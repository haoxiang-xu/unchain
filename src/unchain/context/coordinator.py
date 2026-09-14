from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable

from unchain.journal import (
    BoundExecutionJournal,
    EventCursor,
    JournalEvent,
    JournalSnapshot,
    OperationRef,
    ResourceRef,
    SemanticEventDraft,
    journal_event_sha256,
    journal_event_to_semantic_event,
)

from ..durability import mark_durable_persistence_failure
from .checkpoints import (
    CheckpointRequest,
    build_checkpoint_request,
    checkpoint_event_sha256,
    project_checkpoint_message,
)
from .compiler import (
    ContextCompileResult,
    ContextCompiler,
    _CheckpointBinding,
    _CheckpointConsumption,
    _ContextCompilePass,
    _canonical_journal_message_projection,
    project_canonical_journal_messages,
)
from .models import ContextCompileRequest
from .artifacts import ArtifactService
from .request_factory import _graph_seed_projections
from .model_projection import (
    ContextModelProjectionError,
    ModelContextProjection,
)
from .ports import (
    BoundCheckpointRepository,
    BoundContextBuildRepository,
    CheckpointWriteStatus,
    ContextBuildReceipt,
    PreparedCheckpoint,
)


MAX_CHECKPOINT_PAYLOAD_BYTES = 32 * 1024 * 1024
_CURRENT_ATTEMPT_INPUT_EVENT_TYPES = frozenset(
    {"message.user", "interaction.resolved", "tool_result"}
)
_PENDING_INPUT_EVENT_TYPES = {
    "message.user": "message.user",
    "interaction_resolved": "interaction.resolved",
    "tool_result": "tool_result",
}
_ATTEMPT_TERMINAL_EVENT_TYPES = frozenset(
    {
        "run_completed",
        "run_failed",
        "run_cancelled",
        "run_canceled",
        "run_aborted",
        "run.completed",
        "run.failed",
        "run.cancelled",
        "run.canceled",
        "run.aborted",
    }
)
_DURABLE_INTERNAL_EVENT_TYPES = frozenset(
    {
        "tool.catalog_snapshot",
        "provider.wire_snapshot",
        "provider.turn_result",
    }
)


class ContextCompileCoordinatorError(RuntimeError):
    """A stable failure in the durable two-pass compile protocol."""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _operation(operation_id: str, payload: Mapping[str, Any]) -> OperationRef:
    digest = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return OperationRef(
        operation_id=operation_id,
        payload_sha256=digest,
    )


def _snapshot_descriptor(snapshot: JournalSnapshot) -> dict[str, Any]:
    return {
        "snapshot_sha256": snapshot.snapshot_sha256,
        "high_water": (snapshot.high_water.to_dict() if snapshot.high_water else None),
        "event_count": snapshot.event_count,
    }


@dataclass(frozen=True)
class _PreparedJournalView:
    snapshot: JournalSnapshot
    generation_events: tuple[JournalEvent, ...]
    semantic_events: tuple[Mapping[str, Any], ...]
    events_by_cursor: Mapping[tuple[str, int], JournalEvent]
    input_receipt: JournalEvent


def _event_claim(raw: Mapping[str, Any]) -> tuple[str, int, str]:
    inner = raw.get("event")
    event = inner if isinstance(inner, Mapping) else raw
    payload = event.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    event_id = raw.get("event_id", event.get("event_id", payload.get("event_id")))
    store_seq = raw.get("store_seq", event.get("store_seq", payload.get("store_seq")))
    event_type = raw.get("type", event.get("type", payload.get("type")))
    if (
        not isinstance(event_id, str)
        or not event_id
        or event_id != event_id.strip()
        or isinstance(store_seq, bool)
        or not isinstance(store_seq, int)
        or store_seq <= 0
        or not isinstance(event_type, str)
        or not event_type.strip()
    ):
        raise ContextCompileCoordinatorError(
            "semantic event claim has no exact journal cursor"
        )
    return event_id, store_seq, event_type.strip()


def _validated_pending_task_inputs(
    *,
    request: ContextCompileRequest,
    events_by_cursor: Mapping[tuple[str, int], JournalEvent],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[JournalEvent, ...]]:
    rebuilt: list[Mapping[str, Any]] = []
    receipts: list[JournalEvent] = []
    previous_store_seq = 0
    for raw in request.pending_task_inputs or ():
        event_id = str(raw.get("event_id") or "").strip()
        store_seq = raw.get("store_seq")
        pending_type = str(raw.get("type") or "").strip()
        expected_event_type = _PENDING_INPUT_EVENT_TYPES.get(pending_type)
        if (
            not event_id
            or isinstance(store_seq, bool)
            or not isinstance(store_seq, int)
            or store_seq <= previous_store_seq
            or expected_event_type is None
        ):
            raise ContextCompileCoordinatorError(
                "pending task input has invalid journal identity"
            )
        receipt = events_by_cursor.get((event_id, store_seq))
        if receipt is None or receipt.event_type != expected_event_type:
            raise ContextCompileCoordinatorError(
                "pending task input is absent from the stable journal snapshot"
            )
        if expected_event_type == "interaction.resolved":
            interaction_id = str(receipt.payload.get("interaction_id") or "").strip()
            matching_requests = tuple(
                event
                for event in events_by_cursor.values()
                if event.event_type == "interaction.requested"
                and event.store_seq < receipt.store_seq
                and event.attempt.generation == receipt.attempt.generation
                and str(event.payload.get("interaction_id") or "").strip()
                == interaction_id
            )
            matching_resolutions = tuple(
                event
                for event in events_by_cursor.values()
                if event.event_type == "interaction.resolved"
                and event.attempt.generation == receipt.attempt.generation
                and str(event.payload.get("interaction_id") or "").strip()
                == interaction_id
            )
            if (
                not interaction_id
                or len(matching_requests) != 1
                or matching_resolutions != (receipt,)
            ):
                raise ContextCompileCoordinatorError(
                    "resolved pending task input has no unique interaction request and resolution"
                )
        raw_ref = raw.get("content_ref")
        try:
            content_ref = (
                raw_ref
                if isinstance(raw_ref, ResourceRef)
                else ResourceRef.from_dict(raw_ref)
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise ContextCompileCoordinatorError(
                "pending task input has an invalid durable reference"
            ) from exc
        if content_ref.kind != "artifact" or content_ref not in receipt.resource_refs:
            raise ContextCompileCoordinatorError(
                "pending task input reference is not authorized by its journal receipt"
            )
        payload = receipt.payload
        if expected_event_type == "tool_result":
            call_id = str(
                payload.get("call_id") or payload.get("tool_call_id") or ""
            ).strip()
            tool_name = str(payload.get("tool_name") or "").strip()
            matching_calls = tuple(
                event
                for event in events_by_cursor.values()
                if event.event_type == "tool_call"
                and str(
                    event.payload.get("call_id")
                    or event.payload.get("tool_call_id")
                    or ""
                ).strip()
                == call_id
            )
            matching_results = tuple(
                event
                for event in events_by_cursor.values()
                if event.event_type == "tool_result"
                and str(
                    event.payload.get("call_id")
                    or event.payload.get("tool_call_id")
                    or ""
                ).strip()
                == call_id
            )
            if (
                not call_id
                or not tool_name
                or len(matching_calls) != 1
                or matching_results != (receipt,)
                or matching_calls[0].store_seq >= receipt.store_seq
                or str(matching_calls[0].payload.get("tool_name") or "").strip()
                != tool_name
            ):
                raise ContextCompileCoordinatorError(
                    "tool result causal receipt is invalid"
                )
            raw_authoritative_ref = payload.get("full_output_ref")
            try:
                authoritative_ref = (
                    raw_authoritative_ref
                    if isinstance(raw_authoritative_ref, ResourceRef)
                    else ResourceRef.from_dict(raw_authoritative_ref)
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise ContextCompileCoordinatorError(
                    "tool result input has no authoritative artifact reference"
                ) from exc
            result_payload = payload.get("result")
            result_payload = (
                result_payload if isinstance(result_payload, Mapping) else {}
            )
            content_bytes = payload.get("result_bytes")
            content_sha256 = payload.get("result_sha256")
            preview = str(payload.get("preview") or result_payload.get("preview") or "")
            if authoritative_ref != content_ref:
                raise ContextCompileCoordinatorError(
                    "tool result input reference changed from its receipt"
                )
        else:
            raw_authoritative_ref = payload.get("content_ref")
            if raw_authoritative_ref is not None:
                try:
                    authoritative_ref = (
                        raw_authoritative_ref
                        if isinstance(raw_authoritative_ref, ResourceRef)
                        else ResourceRef.from_dict(raw_authoritative_ref)
                    )
                except (TypeError, ValueError, KeyError) as exc:
                    raise ContextCompileCoordinatorError(
                        "pending task input receipt has an invalid artifact reference"
                    ) from exc
                if authoritative_ref != content_ref:
                    raise ContextCompileCoordinatorError(
                        "pending task input reference changed from its receipt"
                    )
            content_bytes = payload.get("content_bytes")
            content_sha256 = payload.get("content_sha256")
            preview = str(payload.get("preview") or "")
        if (
            isinstance(content_bytes, bool)
            or not isinstance(content_bytes, int)
            or content_bytes < 0
            or not isinstance(content_sha256, str)
            or len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
            or raw.get("content_bytes") != content_bytes
            or raw.get("content_sha256") != content_sha256
            or str(raw.get("preview") or "") != preview
        ):
            raise ContextCompileCoordinatorError(
                "pending task input content receipt is invalid"
            )
        rebuilt.append(
            MappingProxyType(
                {
                    "event_id": event_id,
                    "store_seq": store_seq,
                    "type": pending_type,
                    "preview": preview,
                    "preview_truncated": bool(payload.get("preview_truncated")),
                    "content_ref": content_ref.to_dict(),
                    "content_bytes": content_bytes,
                    "content_sha256": content_sha256,
                }
            )
        )
        receipts.append(receipt)
        previous_store_seq = store_seq
    return tuple(rebuilt), tuple(receipts)


def _prepare_journal_view(
    *,
    request: ContextCompileRequest,
    snapshot: JournalSnapshot,
    artifacts: ArtifactService | None = None,
) -> tuple[ContextCompileRequest, _PreparedJournalView]:
    if not isinstance(snapshot, JournalSnapshot):
        raise ContextCompileCoordinatorError("journal did not return a JournalSnapshot")
    snapshot = JournalSnapshot.from_dict(snapshot.to_dict())
    if snapshot.execution_id != request.execution_id:
        raise ContextCompileCoordinatorError(
            "journal snapshot execution scope does not match the request"
        )
    generation_id = str(request.generation_id or "").strip()
    if not generation_id:
        raise ContextCompileCoordinatorError(
            "durable context compilation requires generation identity"
        )
    generation_events = tuple(
        event
        for event in snapshot.events
        if event.attempt.generation.generation_id == generation_id
    )
    semantic_receipt_events = tuple(
        event
        for event in generation_events
        if event.event_type not in _DURABLE_INTERNAL_EVENT_TYPES
    )
    for event in semantic_receipt_events:
        expected_operation = SemanticEventDraft(
            event_id=event.event_id,
            event_type=event.event_type,
            attempt=event.attempt,
            operation_id=event.operation.operation_id,
            payload=event.payload,
            resource_refs=event.resource_refs,
        ).operation
        if expected_operation != event.operation:
            raise ContextCompileCoordinatorError(
                "journal snapshot event operation receipt is invalid"
            )
    events_by_cursor = {
        (event.event_id, event.store_seq): event for event in generation_events
    }
    for raw in request.semantic_events or ():
        event_id, store_seq, event_type = _event_claim(raw)
        authoritative = events_by_cursor.get((event_id, store_seq))
        if authoritative is None or authoritative.event_type != event_type:
            raise ContextCompileCoordinatorError(
                "semantic event claim conflicts with the journal snapshot"
            )
    pending_task_inputs, pending_input_receipts = _validated_pending_task_inputs(
        request=request,
        events_by_cursor=events_by_cursor,
    )
    graph_seeds = _graph_seed_projections(generation_events, artifacts=artifacts)
    seed_handoff_ids = {seed.handoff.event_id for seed in graph_seeds}
    semantic_events: list[Mapping[str, Any]] = []
    for event in semantic_receipt_events:
        if event.event_id in seed_handoff_ids:
            continue
        semantic = journal_event_to_semantic_event(event)
        semantic["journal_event_sha256"] = journal_event_sha256(event)
        semantic_events.append(MappingProxyType(semantic))
    snapshot_request = replace(
        request,
        semantic_events=tuple(semantic_events),
        pending_task_inputs=pending_task_inputs,
    )
    projected_request = project_canonical_journal_messages(snapshot_request)
    bound_source_indexes = {
        cursor.message_index for cursor in projected_request.source_message_cursors
    }
    for message_index, message in enumerate(projected_request.source_messages):
        role = str(message.get("role") or "").strip().casefold()
        if (
            role not in {"system", "developer"}
            and message_index not in bound_source_indexes
        ):
            raise ContextCompileCoordinatorError(
                "non-system source message is unbound from the journal snapshot"
            )
    attempt_id = str(request.attempt_id or "").strip()
    if not attempt_id:
        raise ContextCompileCoordinatorError(
            "durable context compilation requires attempt identity"
        )
    if any(
        event.attempt.attempt_id == attempt_id
        and event.event_type in _ATTEMPT_TERMINAL_EVENT_TYPES
        for event in generation_events
    ):
        raise ContextCompileCoordinatorError(
            "terminal attempt cannot authorize another context build"
        )
    original_cursor_map = {
        cursor.message_index: (cursor.event_id, cursor.store_seq)
        for cursor in request.source_message_cursors
    }
    if not original_cursor_map and request.source_event_ids:
        original_cursor_map = {
            index: (event_id, request.source_event_store_seqs[index])
            for index, event_id in enumerate(request.source_event_ids)
        }
    original_user_indexes = tuple(
        index
        for index, message in enumerate(request.source_messages)
        if str(message.get("role") or "").strip().casefold() == "user"
    )
    bound_current_user_event: JournalEvent | None = None
    if original_user_indexes:
        current_user_index = original_user_indexes[-1]
        current_user_cursor = original_cursor_map.get(current_user_index)
        current_user_event = (
            events_by_cursor.get(current_user_cursor)
            if current_user_cursor is not None
            else None
        )
        if (
            current_user_event is not None
            and current_user_event.event_type == "message.user"
        ):
            # The model sees the original message, but only the exact step's
            # derived receipt can authorize a build. Re-resolve this against
            # our snapshot rather than trusting a factory-provided mapping.
            bound_current_user_event = next(
                (seed.input_receipt for seed in graph_seeds
                 if seed.source == current_user_event
                 and seed.input_receipt.attempt.attempt_id == attempt_id),
                current_user_event,
            )
    admitted_input_receipts = tuple(
        event
        for event in (
            *((bound_current_user_event,) if bound_current_user_event else ()),
            *pending_input_receipts,
        )
        if event.event_type in _CURRENT_ATTEMPT_INPUT_EVENT_TYPES
    )
    generation_input_receipts = tuple(
        event
        for event in generation_events
        if event.event_type in _CURRENT_ATTEMPT_INPUT_EVENT_TYPES
    )
    if (
        not admitted_input_receipts
        or not generation_input_receipts
        or (
            admitted_input_receipts[-1].event_id,
            admitted_input_receipts[-1].store_seq,
        )
        != (
            generation_input_receipts[-1].event_id,
            generation_input_receipts[-1].store_seq,
        )
        or admitted_input_receipts[-1].attempt.attempt_id != attempt_id
    ):
        raise ContextCompileCoordinatorError(
            "current attempt receipt is absent from the journal snapshot"
        )
    return (
        projected_request,
        _PreparedJournalView(
            snapshot=snapshot,
            generation_events=generation_events,
            semantic_events=tuple(semantic_events),
            events_by_cursor=MappingProxyType(events_by_cursor),
            input_receipt=admitted_input_receipts[-1],
        ),
    )


@dataclass(frozen=True)
class _CheckpointMaterialization:
    source_messages: tuple[Mapping[str, Any], ...]
    dependency_receipts: tuple[JournalEvent, ...]
    refs: tuple[ResourceRef, ...]
    summary: str
    operation: OperationRef


def _checkpoint_materialization(
    *,
    request: ContextCompileRequest,
    checkpoint: CheckpointRequest,
    view: _PreparedJournalView,
) -> _CheckpointMaterialization:
    projection = _canonical_journal_message_projection(request)
    messages_by_cursor = dict(projection.candidates)
    source_cursors = tuple(
        zip(
            checkpoint.source_event_ids,
            checkpoint.source_event_store_seqs,
            strict=True,
        )
    )
    try:
        source_messages = tuple(messages_by_cursor[cursor] for cursor in source_cursors)
        source_events = tuple(
            view.events_by_cursor[cursor] for cursor in source_cursors
        )
    except KeyError as exc:
        raise ContextCompileCoordinatorError(
            "checkpoint request is outside the stable journal snapshot"
        ) from exc
    selected_source_cursors = set(source_cursors)
    expected_dependencies = tuple(
        dependency
        for dependency in projection.projection_dependencies
        if (
            dependency.source_cursor.event_id,
            dependency.source_cursor.store_seq,
        )
        in selected_source_cursors
    )
    rebuilt = build_checkpoint_request(
        source_event_ids=checkpoint.source_event_ids,
        source_event_store_seqs=checkpoint.source_event_store_seqs,
        source_messages=source_messages,
        source_event_sha256s=tuple(
            journal_event_sha256(event) for event in source_events
        ),
        projection_dependencies=expected_dependencies,
    )
    if rebuilt != checkpoint:
        raise ContextCompileCoordinatorError(
            "checkpoint request does not match the stable journal snapshot"
        )
    try:
        dependency_receipts = tuple(
            view.events_by_cursor[
                (
                    dependency.receipt_cursor.event_id,
                    dependency.receipt_cursor.store_seq,
                )
            ]
            for dependency in checkpoint.projection_dependencies
        )
    except KeyError as exc:
        raise ContextCompileCoordinatorError(
            "checkpoint dependency receipt is outside the journal snapshot"
        ) from exc
    for dependency, receipt in zip(
        checkpoint.projection_dependencies,
        dependency_receipts,
        strict=True,
    ):
        if (
            receipt.event_type != dependency.event_type
            or receipt.attempt.attempt_id != dependency.attempt_id
            or journal_event_sha256(receipt) != dependency.event_sha256
        ):
            raise ContextCompileCoordinatorError(
                "checkpoint dependency receipt failed exact verification"
            )
    refs: list[ResourceRef] = []
    seen_refs: set[tuple[str, str, int, str]] = set()
    for event in (*source_events, *dependency_receipts):
        for ref in event.resource_refs:
            if ref.kind != "artifact":
                continue
            identity = (
                ref.kind,
                ref.resource_id,
                ref.revision,
                ref.fragment,
            )
            if identity not in seen_refs:
                seen_refs.add(identity)
                refs.append(ref)
    payload = {
        "schema": "unchain.context_checkpoint_payload.v2",
        "checkpoint_request": checkpoint.to_dict(),
        "source_messages": _plain(source_messages),
        "source_messages_sha256": checkpoint.source_messages_sha256,
        "dependency_receipts": [receipt.to_dict() for receipt in dependency_receipts],
    }
    content = _canonical_bytes(payload)
    if len(content) > MAX_CHECKPOINT_PAYLOAD_BYTES:
        raise ContextCompileCoordinatorError(
            "checkpoint payload exceeds the durable object limit"
        )
    operation_payload = {
        "checkpoint_request": checkpoint.to_dict(),
        "summary_sha256": hashlib.sha256(content).hexdigest(),
        "refs": [ref.to_dict() for ref in refs],
    }
    return _CheckpointMaterialization(
        source_messages=source_messages,
        dependency_receipts=dependency_receipts,
        refs=tuple(refs),
        summary=content.decode("utf-8"),
        operation=_operation(
            f"context-checkpoint.{checkpoint.request_id}",
            operation_payload,
        ),
    )


def _verify_checkpoint_consumption(
    *,
    compiled: _ContextCompilePass,
    checkpoint: CheckpointRequest,
    prepared: PreparedCheckpoint,
) -> _CheckpointConsumption:
    if len(compiled.consumptions) != 1:
        raise ContextCompileCoordinatorError(
            "the second compiler pass omitted its checkpoint consumption proof"
        )
    consumption = compiled.consumptions[0]
    if not isinstance(consumption, _CheckpointConsumption):
        raise ContextCompileCoordinatorError(
            "the second compiler pass returned an invalid consumption proof"
        )
    if (
        consumption.checkpoint_request_id != checkpoint.request_id
        or consumption.checkpoint_ref != prepared.checkpoint_ref
    ):
        raise ContextCompileCoordinatorError(
            "the second compiler pass consumed a different checkpoint"
        )
    expected_message = project_checkpoint_message(
        checkpoint_ref=prepared.checkpoint_ref,
        request=checkpoint,
        omitted_complete_turns=consumption.omitted_complete_turns,
    )
    messages = compiled.result.messages
    if (
        consumption.projected_message_index < 0
        or consumption.projected_message_index >= len(messages)
        or _plain(messages[consumption.projected_message_index]) != expected_message
        or consumption.projected_message_sha256
        != checkpoint_event_sha256(expected_message)
    ):
        raise ContextCompileCoordinatorError(
            "the second compiler pass returned an invalid consumption marker"
        )
    if compiled.result.checkpoint_requests:
        raise ContextCompileCoordinatorError(
            "the second compiler pass did not consume the exact checkpoint"
        )
    if compiled.result.envelope is None or tuple(
        compiled.result.envelope.checkpoint_refs
    ) != (prepared.checkpoint_ref,):
        raise ContextCompileCoordinatorError(
            "the final context envelope omitted its durable checkpoint"
        )
    return consumption


def _context_build_operation_payload(
    *,
    result: ContextCompileResult,
    snapshot: JournalSnapshot,
    consumptions: Sequence[_CheckpointConsumption],
    trigger_cursor: EventCursor,
) -> dict[str, Any]:
    envelope = result.envelope
    if envelope is None:
        raise ContextCompileCoordinatorError(
            "the compiler did not produce a context build envelope"
        )
    return {
        "envelope": envelope.to_dict(),
        "messages_sha256": hashlib.sha256(
            _canonical_bytes(result.messages)
        ).hexdigest(),
        "snapshot": _snapshot_descriptor(snapshot),
        "trigger_cursor": trigger_cursor.to_dict(),
        "checkpoint_consumptions": [
            consumption.to_dict() for consumption in consumptions
        ],
    }


def _context_build_operation_id(
    *,
    envelope,
    trigger_cursor: EventCursor,
) -> str:
    trigger_identity = {
        "execution_id": envelope.execution_id,
        "generation_id": envelope.generation_id,
        "trigger_cursor": trigger_cursor.to_dict(),
    }
    trigger_sha256 = hashlib.sha256(_canonical_bytes(trigger_identity)).hexdigest()
    return f"context-build-trigger.{trigger_sha256}"


class ContextCompileCoordinator:
    """Bind both pure compiler passes to one durable journal snapshot."""

    def __init__(
        self,
        *,
        journal: BoundExecutionJournal,
        checkpoint_repository: BoundCheckpointRepository,
        build_repository: BoundContextBuildRepository,
        partial_attempt_sink: Callable[[ContextCompileRequest, Exception], None],
        model_projection: ModelContextProjection | None = None,
        artifacts: ArtifactService | None = None,
    ) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        if not callable(getattr(checkpoint_repository, "prepare", None)):
            raise TypeError("checkpoint_repository must provide prepare")
        if not callable(getattr(checkpoint_repository, "commit", None)):
            raise TypeError("checkpoint_repository must provide commit")
        if not callable(getattr(checkpoint_repository, "get_by_operation", None)):
            raise TypeError("checkpoint_repository must provide get_by_operation")
        if not callable(getattr(build_repository, "record", None)):
            raise TypeError("build_repository must provide record")
        if not callable(getattr(build_repository, "get_by_operation", None)):
            raise TypeError("build_repository must provide get_by_operation")
        if not callable(getattr(build_repository, "get_by_trigger", None)):
            raise TypeError("build_repository must provide get_by_trigger")
        if not callable(partial_attempt_sink):
            raise TypeError("partial_attempt_sink must be callable")
        if (
            model_projection is not None
            and type(model_projection) is not ModelContextProjection
        ):
            raise TypeError(
                "model_projection must be the official ModelContextProjection"
            )
        if artifacts is not None and (
            not isinstance(artifacts, ArtifactService)
            or artifacts.execution_id != journal.execution_id
        ):
            raise ContextCompileCoordinatorError("artifact service scope mismatch")
        self._artifacts = artifacts
        self._compiler = ContextCompiler()
        self._journal = journal
        self._checkpoint_repository = checkpoint_repository
        self._build_repository = build_repository
        self._partial_attempt_sink = partial_attempt_sink
        self._model_projection = model_projection

    @property
    def journal(self) -> BoundExecutionJournal:
        return self._journal

    @property
    def model_projection(self) -> ModelContextProjection | None:
        return self._model_projection

    def compile(self, request: ContextCompileRequest) -> ContextCompileResult:
        if not isinstance(request, ContextCompileRequest):
            raise TypeError("request must be a ContextCompileRequest")
        if (
            request.checkpoint_ref is not None
            or request.checkpoint_request_id is not None
        ):
            raise ContextCompileCoordinatorError(
                "external pre-bound checkpoints are forbidden"
            )
        execution_id = str(request.execution_id or "")
        if not execution_id:
            raise ContextCompileCoordinatorError(
                "durable context compilation requires build identity"
            )
        for repository in (
            self._journal,
            self._checkpoint_repository,
            self._build_repository,
        ):
            if str(getattr(repository, "execution_id", "")) != execution_id:
                raise ContextCompileCoordinatorError(
                    "context repository execution scope does not match the request"
                )
        try:
            snapshot = self._journal.capture_snapshot()
            prepared_request, journal_view = _prepare_journal_view(
                request=request,
                snapshot=snapshot,
                artifacts=self._artifacts,
            )
            self._verify_input_trigger_claim(request, journal_view.input_receipt)
            first = self._compile_pass(prepared_request)
        except Exception as error:
            self._mark_partial(request, error)
            raise
        if first.consumptions:
            error = ContextCompileCoordinatorError(
                "the first compiler pass returned an unexpected consumption proof"
            )
            self._mark_partial(request, error)
            raise error
        if len(first.result.checkpoint_requests) > 1:
            error = ContextCompileCoordinatorError(
                "the P0 compiler produced multiple checkpoint requests"
            )
            self._mark_partial(request, error)
            raise error

        final = first
        consumptions: tuple[_CheckpointConsumption, ...] = ()
        if first.result.checkpoint_requests:
            checkpoint_request = first.result.checkpoint_requests[0]
            try:
                materialization = _checkpoint_materialization(
                    request=prepared_request,
                    checkpoint=checkpoint_request,
                    view=journal_view,
                )
            except Exception as error:
                self._mark_partial(request, error)
                raise
            try:
                prepared = self._prepare_checkpoint(
                    checkpoint=checkpoint_request,
                    materialization=materialization,
                )
            except Exception as error:
                boundary_error = self._mark_durable_partial(request, error)
                if boundary_error is error:
                    raise
                raise boundary_error from None
            try:
                final = self._compile_pass(
                    prepared_request,
                    checkpoint_binding=_CheckpointBinding(
                        request=checkpoint_request,
                        checkpoint_ref=prepared.checkpoint_ref,
                    ),
                )
                consumption = _verify_checkpoint_consumption(
                    compiled=final,
                    checkpoint=checkpoint_request,
                    prepared=prepared,
                )
            except Exception as error:
                self._mark_partial(request, error)
                raise
            if prepared.status is CheckpointWriteStatus.PREPARED:
                try:
                    prepared = self._commit_checkpoint(prepared)
                except Exception as error:
                    boundary_error = self._mark_durable_partial(request, error)
                    if boundary_error is error:
                        raise
                    raise boundary_error from None
            if prepared.status is not CheckpointWriteStatus.COMMITTED:
                error = ContextCompileCoordinatorError(
                    "checkpoint repository did not return a committed receipt"
                )
                boundary_error = self._mark_durable_partial(request, error)
                raise boundary_error from None
            consumptions = (consumption,)

        if final.result.envelope is None:
            error = ContextCompileCoordinatorError(
                "the compiler did not produce a context build envelope"
            )
            self._mark_partial(request, error)
            raise error
        try:
            model_result = self._project_model_result(
                final.result,
                provider=request.provider or "",
            )
        except Exception as error:
            self._mark_partial(request, error)
            raise
        try:
            recorded = self._record_build(
                model_result,
                snapshot=journal_view.snapshot,
                consumptions=consumptions,
                trigger_cursor=EventCursor(
                    store_seq=journal_view.input_receipt.store_seq,
                    event_id=journal_view.input_receipt.event_id,
                ),
            )
        except Exception as error:
            boundary_error = self._mark_durable_partial(request, error)
            if boundary_error is error:
                raise
            raise boundary_error from None
        if recorded != model_result.envelope:
            error = ContextCompileCoordinatorError(
                "the context build repository changed the recorded envelope"
            )
            boundary_error = self._mark_durable_partial(request, error)
            raise boundary_error from None
        return model_result

    def _project_model_result(
        self,
        result: ContextCompileResult,
        *,
        provider: str,
    ) -> ContextCompileResult:
        projection = self._model_projection
        if projection is None:
            if any("attachments" in message for message in result.messages):
                raise ContextModelProjectionError(
                    "attachment_materializer_unavailable"
                )
            return result
        messages = projection.project(result.messages, provider=provider)
        if messages == result.messages:
            return result
        return replace(result, messages=messages)

    def _compile_pass(
        self,
        request: ContextCompileRequest,
        *,
        checkpoint_binding: _CheckpointBinding | None = None,
    ) -> _ContextCompilePass:
        result = self._compiler._compile_for_coordinator(
            request,
            checkpoint_binding=checkpoint_binding,
        )
        if not isinstance(result, _ContextCompilePass):
            raise TypeError("compiler must return the internal compile pass")
        if not isinstance(result.result, ContextCompileResult):
            raise TypeError("compiler pass must contain ContextCompileResult")
        return result

    def _record_build(
        self,
        result: ContextCompileResult,
        *,
        snapshot: JournalSnapshot,
        consumptions: Sequence[_CheckpointConsumption],
        trigger_cursor: EventCursor,
    ):
        envelope = result.envelope
        if envelope is None:
            raise ContextCompileCoordinatorError(
                "the compiler did not produce a context build envelope"
            )
        operation_payload = _context_build_operation_payload(
            result=result,
            snapshot=snapshot,
            consumptions=consumptions,
            trigger_cursor=trigger_cursor,
        )
        operation = _operation(
            _context_build_operation_id(
                envelope=envelope,
                trigger_cursor=trigger_cursor,
            ),
            operation_payload,
        )
        receipt = self._build_repository.get_by_operation(operation=operation)
        if receipt is None:
            try:
                receipt = self._build_repository.record(
                    envelope=envelope,
                    operation=operation,
                    trigger_cursor=trigger_cursor,
                )
            except Exception as error:
                receipt = self._build_repository.get_by_operation(operation=operation)
                if receipt is None:
                    raise error
        if (
            not isinstance(receipt, ContextBuildReceipt)
            or receipt.operation != operation
            or receipt.envelope != envelope
            or receipt.trigger_cursor != trigger_cursor
        ):
            raise ContextCompileCoordinatorError(
                "context build repository changed the operation receipt"
            )
        return receipt.envelope

    def _verify_input_trigger_claim(
        self,
        request: ContextCompileRequest,
        input_receipt: JournalEvent,
    ) -> None:
        trigger_cursor = EventCursor(
            store_seq=input_receipt.store_seq,
            event_id=input_receipt.event_id,
        )
        receipt = self._build_repository.get_by_trigger(
            trigger_cursor=trigger_cursor,
        )
        if receipt is None:
            return
        if (
            not isinstance(receipt, ContextBuildReceipt)
            or receipt.trigger_cursor != trigger_cursor
            or receipt.envelope.execution_id != request.execution_id
            or receipt.envelope.generation_id != request.generation_id
            or receipt.envelope.attempt_id != request.attempt_id
        ):
            raise ContextCompileCoordinatorError(
                "input receipt has an invalid durable build binding"
            )
        if receipt.envelope.build_id != request.build_id:
            raise ContextCompileCoordinatorError(
                "input receipt already authorized a distinct model build"
            )

    def _prepare_checkpoint(
        self,
        *,
        checkpoint: CheckpointRequest,
        materialization: _CheckpointMaterialization,
    ) -> PreparedCheckpoint:
        receipt = self._checkpoint_repository.get_by_operation(
            operation=materialization.operation
        )
        if receipt is None:
            try:
                receipt = self._checkpoint_repository.prepare(
                    source_range=checkpoint.source_range,
                    summary=materialization.summary,
                    refs=materialization.refs,
                    operation=materialization.operation,
                )
            except Exception as error:
                receipt = self._checkpoint_repository.get_by_operation(
                    operation=materialization.operation
                )
                if receipt is None:
                    raise error
        if (
            not isinstance(receipt, PreparedCheckpoint)
            or receipt.operation != materialization.operation
            or receipt.status
            not in {
                CheckpointWriteStatus.PREPARED,
                CheckpointWriteStatus.COMMITTED,
            }
        ):
            raise ContextCompileCoordinatorError(
                "checkpoint repository changed the preparation identity"
            )
        return receipt

    def _commit_checkpoint(
        self,
        prepared: PreparedCheckpoint,
    ) -> PreparedCheckpoint:
        try:
            receipt = self._checkpoint_repository.commit(prepared=prepared)
        except Exception as error:
            receipt = self._checkpoint_repository.get_by_operation(
                operation=prepared.operation
            )
            if receipt is None:
                raise error
        if (
            not isinstance(receipt, PreparedCheckpoint)
            or receipt.operation != prepared.operation
            or receipt.preparation_id != prepared.preparation_id
            or receipt.checkpoint_ref != prepared.checkpoint_ref
            or receipt.status is not CheckpointWriteStatus.COMMITTED
        ):
            raise ContextCompileCoordinatorError(
                "checkpoint repository changed the committed receipt"
            )
        return receipt

    def _mark_durable_partial(
        self,
        request: ContextCompileRequest,
        error: Exception,
    ) -> Exception:
        boundary_error = mark_durable_persistence_failure(error)
        if not isinstance(boundary_error, Exception):
            raise TypeError("durable repository failures must be Exception instances")
        self._mark_partial(request, boundary_error)
        return boundary_error

    def _mark_partial(
        self,
        request: ContextCompileRequest,
        error: Exception,
    ) -> None:
        try:
            self._partial_attempt_sink(request, error)
        except Exception:
            pass


__all__ = [
    "ContextCompileCoordinator",
    "ContextCompileCoordinatorError",
    "MAX_CHECKPOINT_PAYLOAD_BYTES",
]
