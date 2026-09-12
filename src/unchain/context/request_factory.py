from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundExecutionJournal,
    EventCursor,
    JournalEvent,
    JournalSnapshot,
    ResourceRef,
    journal_event_sha256,
    journal_event_to_semantic_event,
)
from unchain.journal.models import _required_text, _thaw_json
from unchain.kernel.harness import HarnessContext

from .artifacts import ArtifactService
from .attachments import normalize_host_resolved_attachments
from .budget import estimate_context_tokens, resolve_context_budget
from .models import ContextCompileRequest, HandoffEnvelope, SourceMessageCursor
from .graph_checkpoint import GraphExecutionPlan


_CURRENT_INPUT_EVENT_TYPES = frozenset(
    {"message.user", "interaction.resolved", "tool_result"}
)


class JournalContextRequestFactoryError(RuntimeError):
    """A stable journal snapshot could not authorize a context request."""


class ModelWindowFallbackPolicy(Protocol):
    def __call__(self, provider: str, model: str) -> int:
        ...


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _positive_limit(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _provider_identity(context: HarnessContext) -> tuple[str, str]:
    try:
        provider = _required_text(
            context.state.provider_state.provider,
            "provider",
            maximum=512,
        )
        model = _required_text(
            context.state.provider_state.model,
            "model",
            maximum=512,
        )
    except (TypeError, ValueError) as exc:
        raise JournalContextRequestFactoryError(
            "current provider and model identity are required"
        ) from exc
    return provider, model


def _current_instruction_messages(
    context: HarnessContext,
) -> tuple[Mapping[str, Any], ...]:
    selected: list[Mapping[str, Any]] = []
    for message in context.latest_messages():
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role") or "").strip().casefold()
        if role in {"system", "developer"}:
            selected.append(copy.deepcopy(dict(message)))
    return tuple(selected)


def _current_tool_schema(
    context: HarnessContext,
    *,
    provider: str,
) -> int:
    toolkit = context.event.get("toolkit")
    if toolkit is None:
        return 0
    serializer = getattr(toolkit, "to_provider_json", None)
    if not callable(serializer):
        raise JournalContextRequestFactoryError(
            "current toolkit cannot provide a provider schema"
        )
    try:
        schema = serializer(provider)
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise JournalContextRequestFactoryError(
            "current tool schema is not finite JSON"
        ) from exc
    if not isinstance(schema, Sequence) or isinstance(
        schema,
        (str, bytes, bytearray),
    ):
        raise JournalContextRequestFactoryError("current tool schema must be an array")
    if not schema:
        return 0
    estimate = estimate_context_tokens(
        (
            {
                "role": "system",
                "content": "CURRENT_TOOL_SCHEMA\n" + encoded,
            },
        )
    )
    return estimate.total_tokens


def _validated_snapshot(
    journal: BoundExecutionJournal,
    *,
    max_events: int,
    max_bytes: int,
) -> JournalSnapshot:
    snapshot = journal.capture_snapshot(
        max_events=max_events,
        max_bytes=max_bytes,
    )
    if not isinstance(snapshot, JournalSnapshot):
        raise JournalContextRequestFactoryError(
            "journal did not return a stable snapshot"
        )
    snapshot = JournalSnapshot.from_dict(snapshot.to_dict())
    if snapshot.execution_id != journal.execution_id:
        raise JournalContextRequestFactoryError(
            "journal snapshot escaped its execution scope"
        )
    return snapshot


def _canonical_user_message(event: JournalEvent) -> dict[str, Any]:
    message = event.payload.get("message")
    raw_attachments = event.payload.get("attachments")
    raw_attachment_refs = event.payload.get("attachment_refs")
    try:
        attachments = (
            normalize_host_resolved_attachments(raw_attachments)
            if raw_attachments is not None
            else ()
        )
        if raw_attachment_refs is None:
            attachment_refs = ()
        elif isinstance(raw_attachment_refs, Sequence) and not isinstance(
            raw_attachment_refs,
            (str, bytes, bytearray),
        ):
            attachment_refs = tuple(
                value
                if isinstance(value, ResourceRef)
                else ResourceRef.from_dict(value)
                for value in raw_attachment_refs
            )
        else:
            raise TypeError("attachment_refs must be an array")
    except (TypeError, ValueError, KeyError) as exc:
        raise JournalContextRequestFactoryError(
            "canonical journal user attachments are invalid"
        ) from exc
    expected_message_fields = {"role", "content"}
    if attachments:
        expected_message_fields.add("attachments")
    if (
        not isinstance(message, Mapping)
        or set(message) != expected_message_fields
        or message.get("role") != "user"
        or not isinstance(message.get("content"), str)
        or (
            not str(message.get("content") or "").strip()
            and not attachments
        )
    ):
        raise JournalContextRequestFactoryError(
            "canonical journal user message is invalid"
        )
    raw_ref = event.payload.get("content_ref")
    try:
        content_ref = (
            raw_ref
            if isinstance(raw_ref, ResourceRef)
            else ResourceRef.from_dict(raw_ref)
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise JournalContextRequestFactoryError(
            "canonical journal user message has no artifact descriptor"
        ) from exc
    if content_ref.kind != "artifact" or content_ref not in event.resource_refs:
        raise JournalContextRequestFactoryError(
            "canonical journal user artifact descriptor is unauthorized"
        )
    expected_attachment_refs = tuple(
        attachment.artifact.ref for attachment in attachments
    )
    if attachments:
        try:
            message_attachments = normalize_host_resolved_attachments(
                message.get("attachments")
            )
        except (TypeError, ValueError, KeyError) as exc:
            raise JournalContextRequestFactoryError(
                "canonical journal user attachment descriptors are unauthorized"
            ) from exc
        if (
            attachment_refs != expected_attachment_refs
            or tuple(event.resource_refs) != (content_ref, *attachment_refs)
            or message_attachments != attachments
        ):
            raise JournalContextRequestFactoryError(
                "canonical journal user attachment descriptors are unauthorized"
            )
    elif raw_attachment_refs is not None or raw_attachments is not None:
        raise JournalContextRequestFactoryError(
            "canonical journal user attachments are incomplete"
        )
    return _thaw_json(message)


def _is_verified_derived_input(
    trigger: JournalEvent,
    *,
    generation_events: Sequence[JournalEvent],
) -> bool:
    """Recognize the official handoff→message receipt for a child attempt."""

    if trigger.event_type != "message.user":
        return False
    try:
        message = _canonical_user_message(trigger)
        attachments = normalize_host_resolved_attachments(
            message.get("attachments")
        )
        if len(attachments) != 1 or attachments[0].kind != "handoff":
            return False
        descriptor = json.loads(str(message.get("content") or ""))
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("schema") != "unchain.derived_handoff_input.v1"
        ):
            return False
        consumer_attempt = AttemptRef.from_dict(
            descriptor["consumer_attempt"]
        )
        source_attempt = AttemptRef.from_dict(descriptor["source_attempt"])
        handoff_cursor = EventCursor.from_dict(descriptor["handoff_event"])
        envelope = HandoffEnvelope.from_dict(descriptor["handoff_envelope"])
        full_output_artifact = ArtifactRef.from_dict(
            descriptor["full_output_artifact"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False
    if (
        consumer_attempt != trigger.attempt
        or source_attempt == consumer_attempt
        or source_attempt.generation != consumer_attempt.generation
        or envelope.child_attempt != source_attempt
        or envelope.full_output_ref != full_output_artifact.ref
        or envelope.byte_length != full_output_artifact.byte_length
        or envelope.sha256 != full_output_artifact.sha256
        or full_output_artifact.media_type != "application/json"
        or attachments[0].artifact != full_output_artifact
        or handoff_cursor.store_seq >= trigger.store_seq
    ):
        return False
    handoff_events = tuple(
        event
        for event in generation_events
        if event.store_seq == handoff_cursor.store_seq
        and event.event_id == handoff_cursor.event_id
    )
    if len(handoff_events) != 1:
        return False
    handoff = handoff_events[0]
    expected_refs = tuple(
        dict.fromkeys((envelope.full_output_ref, *envelope.artifact_refs))
    )
    try:
        persisted_envelope = HandoffEnvelope.from_dict(
            handoff.payload["handoff_envelope"]
        )
        persisted_artifact = ArtifactRef.from_dict(
            handoff.payload["full_output_artifact"]
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        handoff.attempt == trigger.attempt
        and handoff.event_type == "handoff.recorded"
        and persisted_envelope == envelope
        and persisted_artifact == full_output_artifact
        and handoff.resource_refs == expected_refs
    )


@dataclass(frozen=True)
class _GraphSeedProjection:
    source: JournalEvent
    input_receipt: JournalEvent
    handoff: JournalEvent


def _graph_seed_projections(
    events: Sequence[JournalEvent],
    *,
    artifacts: ArtifactService | None = None,
) -> tuple[_GraphSeedProjection, ...]:
    """Resolve root graph seeds from receipts, never from model-facing text.

    The input receipt still authorizes the step. Its original source is only
    the message projection; the seed artifact and all journal bytes stay put.
    """
    by_cursor = {(event.event_id, event.store_seq): event for event in events}
    projections: list[_GraphSeedProjection] = []
    for admission in events:
        if admission.event_type != "graph.execution.admitted":
            continue
        try:
            if set(admission.payload) != {"graph_plan_id", "graph_scope_id", "plan"}:
                raise ValueError("graph admission fields")
            plan = GraphExecutionPlan.from_dict(_thaw_json(admission.payload["plan"]))
            if (
                plan.orchestration_attempt != admission.attempt
                or admission.payload["graph_plan_id"] != plan.plan_id
                or admission.payload["graph_scope_id"] != plan.scope_id
            ):
                raise ValueError("graph admission identity")
            step = plan.steps[0]
            source = by_cursor[(plan.initial_input_cursor.event_id,
                                plan.initial_input_cursor.store_seq)]
            if source.attempt != plan.orchestration_attempt:
                raise ValueError("graph seed source")
            if source.event_type == "interaction.resolved":
                continue
            if source.event_type != "message.user":
                raise ValueError("graph seed source type")
            source_message = _canonical_user_message(source)
            # A recipe-ref/subagent coordinator has an intentional derived
            # task as its seed. It is not a root user message to unwrap.
            if any(attachment.kind == "handoff" for attachment in
                   normalize_host_resolved_attachments(source_message.get("attachments"))):
                continue
            starts = [event for event in events
                      if event.event_type == "graph.step.started"
                      and event.attempt == step.attempt]
            if not starts:
                continue
            if len(starts) != 1:
                raise ValueError("ambiguous graph seed start")
            started = starts[0]
            if (
                set(started.payload) != {"graph_plan_id", "graph_scope_id", "step",
                                         "handoff_cursor", "input_cursor"}
                or started.payload["graph_plan_id"] != plan.plan_id
                or started.payload["graph_scope_id"] != plan.scope_id
                or _thaw_json(started.payload["step"]) != step.to_dict()
            ):
                raise ValueError("graph seed start identity")
            input_cursor = EventCursor.from_dict(started.payload["input_cursor"])
            handoff_cursor = EventCursor.from_dict(started.payload["handoff_cursor"])
            trigger = by_cursor[(input_cursor.event_id, input_cursor.store_seq)]
            handoff = by_cursor[(handoff_cursor.event_id, handoff_cursor.store_seq)]
            if trigger.attempt != step.attempt or not _is_verified_derived_input(
                trigger, generation_events=events,
            ):
                raise ValueError("graph seed derived receipt")
            descriptor = json.loads(trigger.payload["message"]["content"])
            if set(descriptor) != {"schema", "consumer_attempt", "source_attempt",
                                   "handoff_event", "handoff_envelope", "full_output_artifact"}:
                raise ValueError("graph seed descriptor fields")
            envelope = HandoffEnvelope.from_dict(descriptor["handoff_envelope"])
            artifact = ArtifactRef.from_dict(descriptor["full_output_artifact"])
            seed_value = {
                "schema": "unchain.graph_input_seed.v1",
                "input_event": source.to_dict(),
            }
            seed_bytes = _canonical_bytes(seed_value)
            if artifacts is not None:
                # Graph seed artifacts use the general sanitizer, while the
                # canonical user event can retain provenance-only handles.
                seed_bytes = artifacts.prepare_json_value(
                    seed_value,
                    operation_id="graph-seed-projection-verification",
                ).sanitized
                if artifacts.read_full(
                    artifact, remaining_budget_bytes=artifact.byte_length,
                ) != seed_bytes:
                    raise ValueError("graph seed stored artifact")
            if (
                descriptor["source_attempt"] != source.attempt.to_dict()
                or descriptor["handoff_event"] != handoff_cursor.to_dict()
                or envelope.source_event_range.start != plan.initial_input_cursor
                or envelope.source_event_range.end != plan.initial_input_cursor
                or envelope.artifact_refs
                or envelope.status.value != "complete"
                or artifact.sha256 != hashlib.sha256(seed_bytes).hexdigest()
                or artifact.byte_length != len(seed_bytes)
                or started.resource_refs != (artifact.ref,)
                or not source.store_seq < admission.store_seq < handoff.store_seq < trigger.store_seq < started.store_seq
            ):
                raise ValueError("graph seed artifact or lineage")
            projections.append(_GraphSeedProjection(source, trigger, handoff))
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise JournalContextRequestFactoryError(
                "graph seed projection does not match its durable receipts"
            ) from exc
    return tuple(projections)


def _pending_tool_result(event: JournalEvent) -> dict[str, Any]:
    payload = event.payload
    raw_ref = payload.get("full_output_ref")
    try:
        content_ref = (
            raw_ref
            if isinstance(raw_ref, ResourceRef)
            else ResourceRef.from_dict(raw_ref)
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise JournalContextRequestFactoryError(
            "tool result has no complete artifact descriptor"
        ) from exc
    content_bytes = payload.get("result_bytes")
    content_sha256 = payload.get("result_sha256")
    preview = payload.get("preview")
    preview_truncated = payload.get("preview_truncated")
    if (
        content_ref.kind != "artifact"
        or content_ref not in event.resource_refs
        or isinstance(content_bytes, bool)
        or not isinstance(content_bytes, int)
        or content_bytes < 0
        or not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
        or not isinstance(preview, str)
        or not isinstance(preview_truncated, bool)
    ):
        raise JournalContextRequestFactoryError(
            "tool result has no complete artifact descriptor"
        )
    return {
        "event_id": event.event_id,
        "store_seq": event.store_seq,
        "type": "tool_result",
        "preview": preview,
        "preview_truncated": preview_truncated,
        "content_ref": content_ref.to_dict(),
        "content_bytes": content_bytes,
        "content_sha256": content_sha256,
    }


def _pending_interaction_resolution(
    event: JournalEvent,
    *,
    generation_events: Sequence[JournalEvent],
) -> dict[str, Any]:
    payload = event.payload
    interaction_id = str(payload.get("interaction_id") or "").strip()
    matching_requests = tuple(
        candidate
        for candidate in generation_events
        if candidate.event_type == "interaction.requested"
        and candidate.store_seq < event.store_seq
        and str(candidate.payload.get("interaction_id") or "").strip() == interaction_id
    )
    matching_resolutions = tuple(
        candidate
        for candidate in generation_events
        if candidate.event_type == "interaction.resolved"
        and str(candidate.payload.get("interaction_id") or "").strip() == interaction_id
    )
    if (
        not interaction_id
        or len(matching_requests) != 1
        or matching_resolutions != (event,)
    ):
        raise JournalContextRequestFactoryError(
            "interaction resolution has no unique interaction request"
        )
    raw_ref = payload.get("content_ref")
    try:
        content_ref = (
            raw_ref
            if isinstance(raw_ref, ResourceRef)
            else ResourceRef.from_dict(raw_ref)
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise JournalContextRequestFactoryError(
            "interaction resolution has no complete artifact descriptor"
        ) from exc
    content_bytes = payload.get("content_bytes")
    content_sha256 = payload.get("content_sha256")
    preview = payload.get("preview")
    preview_truncated = payload.get("preview_truncated")
    if (
        content_ref.kind != "artifact"
        or content_ref not in event.resource_refs
        or isinstance(content_bytes, bool)
        or not isinstance(content_bytes, int)
        or content_bytes < 0
        or not isinstance(content_sha256, str)
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
        or not isinstance(preview, str)
        or not isinstance(preview_truncated, bool)
    ):
        raise JournalContextRequestFactoryError(
            "interaction resolution has no complete artifact descriptor"
        )
    return {
        "event_id": event.event_id,
        "store_seq": event.store_seq,
        "type": "interaction_resolved",
        "preview": preview,
        "preview_truncated": preview_truncated,
        "content_ref": content_ref.to_dict(),
        "content_bytes": content_bytes,
        "content_sha256": content_sha256,
    }


class JournalContextRequestFactory:
    """Construct one compile request from an exact immutable journal prefix."""

    def __init__(
        self,
        *,
        attempt: AttemptRef,
        journal: BoundExecutionJournal,
        model_window_fallback: ModelWindowFallbackPolicy,
        artifacts: ArtifactService | None = None,
        output_reserve_tokens: int | None = None,
        transport_margin_tokens: int | None = None,
        snapshot_max_events: int = 10_000,
        snapshot_max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(attempt, AttemptRef):
            attempt = AttemptRef.from_dict(attempt)
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        if journal.execution_id != attempt.generation.execution_id:
            raise JournalContextRequestFactoryError(
                "journal execution does not match the bound attempt"
            )
        if not callable(model_window_fallback):
            raise TypeError("model_window_fallback must be callable")
        if artifacts is not None and (
            not isinstance(artifacts, ArtifactService)
            or artifacts.execution_id != journal.execution_id
        ):
            raise JournalContextRequestFactoryError("artifact service scope mismatch")
        self._artifacts = artifacts
        self._attempt = attempt
        self._journal = journal
        self._model_window_fallback = model_window_fallback
        self._output_reserve_tokens = output_reserve_tokens
        self._transport_margin_tokens = transport_margin_tokens
        self._snapshot_max_events = _positive_limit(
            snapshot_max_events,
            "snapshot_max_events",
        )
        self._snapshot_max_bytes = _positive_limit(
            snapshot_max_bytes,
            "snapshot_max_bytes",
        )

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    @property
    def journal(self) -> BoundExecutionJournal:
        return self._journal

    def __call__(self, context: HarnessContext) -> ContextCompileRequest:
        if not isinstance(context, HarnessContext):
            raise TypeError("context must be a HarnessContext")
        run_id = str(context.event.get("run_id") or "").strip()
        if run_id != self._attempt.attempt_id:
            raise JournalContextRequestFactoryError(
                "harness context does not match the bound attempt"
            )
        provider, model = _provider_identity(context)
        budget = self._resolve_budget(context, provider=provider, model=model)
        tool_schema_tokens = _current_tool_schema(
            context,
            provider=provider,
        )
        snapshot = _validated_snapshot(
            self._journal,
            max_events=self._snapshot_max_events,
            max_bytes=self._snapshot_max_bytes,
        )
        generation_events = tuple(
            event
            for event in snapshot.events
            if event.attempt.generation == self._attempt.generation
        )
        generation_inputs = tuple(
            event
            for event in generation_events
            if event.event_type in _CURRENT_INPUT_EVENT_TYPES
        )
        if not generation_inputs:
            raise JournalContextRequestFactoryError(
                "journal snapshot has no current input receipt"
            )
        attempt_inputs = tuple(
            event for event in generation_inputs if event.attempt == self._attempt
        )
        if not attempt_inputs:
            raise JournalContextRequestFactoryError(
                "journal snapshot has no current-attempt input receipt"
            )
        trigger = attempt_inputs[-1]
        if generation_inputs[-1] != trigger and not _is_verified_derived_input(
            trigger,
            generation_events=generation_events,
        ):
            raise JournalContextRequestFactoryError(
                "latest input receipt belongs to a foreign attempt"
            )
        source_messages = list(_current_instruction_messages(context))
        graph_seeds = _graph_seed_projections(generation_events, artifacts=self._artifacts)
        visible_trigger = next(
            (seed.source for seed in graph_seeds if seed.input_receipt == trigger),
            trigger,
        )
        seed_handoff_ids = {seed.handoff.event_id for seed in graph_seeds}
        source_cursors: tuple[SourceMessageCursor, ...] = ()
        pending_task_inputs: tuple[Mapping[str, Any], ...] | None = None
        if trigger.event_type == "message.user":
            source_messages.append(_canonical_user_message(visible_trigger))
            source_cursors = (
                SourceMessageCursor(
                    message_index=len(source_messages) - 1,
                    event_id=visible_trigger.event_id,
                    store_seq=visible_trigger.store_seq,
                ),
            )
        elif trigger.event_type == "tool_result":
            pending_task_inputs = (_pending_tool_result(trigger),)
        elif trigger.event_type == "interaction.resolved":
            pending_task_inputs = (
                _pending_interaction_resolution(
                    trigger,
                    generation_events=generation_events,
                ),
            )
        else:  # pragma: no cover - guarded by _CURRENT_INPUT_EVENT_TYPES
            raise JournalContextRequestFactoryError("unsupported current input receipt")
        semantic_events: list[Mapping[str, Any]] = []
        for event in generation_events:
            if event.event_id in seed_handoff_ids:
                continue
            projected = journal_event_to_semantic_event(event)
            projected["journal_event_sha256"] = journal_event_sha256(event)
            semantic_events.append(projected)
        build_id = self._build_id(
            trigger=trigger,
            provider=provider,
            model=model,
        )
        return ContextCompileRequest(
            case="journal-runtime",
            source_messages=tuple(source_messages),
            current_generation=self._attempt.generation.generation_id,
            fixed_overhead_tokens=tool_schema_tokens,
            semantic_events=tuple(semantic_events),
            pending_task_inputs=pending_task_inputs,
            budget=budget,
            source_message_cursors=source_cursors,
            provider=provider,
            model=model,
            build_id=build_id,
            execution_id=self._attempt.generation.execution_id,
            generation_id=self._attempt.generation.generation_id,
            attempt_id=self._attempt.attempt_id,
        )

    def _resolve_budget(
        self,
        context: HarnessContext,
        *,
        provider: str,
        model: str,
    ):
        declared = context.state.provider_state.max_context_window_tokens
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
            raise JournalContextRequestFactoryError(
                "declared model context window is invalid"
            )
        window = declared
        if window == 0:
            try:
                window = self._model_window_fallback(provider, model)
            except Exception as exc:
                raise JournalContextRequestFactoryError(
                    "model context fallback resolution failed"
                ) from exc
            if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
                raise JournalContextRequestFactoryError(
                    "model context fallback must be a finite positive integer"
                )
        try:
            return resolve_context_budget(
                context_window_tokens=window,
                output_reserve_tokens=self._output_reserve_tokens,
                transport_margin_tokens=self._transport_margin_tokens,
            )
        except (TypeError, ValueError) as exc:
            raise JournalContextRequestFactoryError(
                "model context budget is unusable"
            ) from exc

    def _build_id(
        self,
        *,
        trigger: JournalEvent,
        provider: str,
        model: str,
    ) -> str:
        identity = {
            "domain": "unchain.context_build.trigger.v1",
            "attempt": self._attempt.to_dict(),
            "trigger": {
                "event_id": trigger.event_id,
                "store_seq": trigger.store_seq,
            },
            "provider": provider,
            "model": model,
        }
        return "context-build-" + hashlib.sha256(_canonical_bytes(identity)).hexdigest()


__all__ = [
    "JournalContextRequestFactory",
    "JournalContextRequestFactoryError",
    "ModelWindowFallbackPolicy",
]
