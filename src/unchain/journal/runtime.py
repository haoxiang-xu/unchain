from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from .models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    JournalAppendRequest,
    JournalAppendResult,
    JournalEvent,
    JournalPage,
    OperationRef,
    PendingArtifact,
    ResourceRef,
    ToolExecutionReceiptLookup,
    _freeze_json,
    _record_tuple,
    _required_text,
    _thaw_json,
)
from .ports import BoundExecutionJournal, BoundToolReceiptIndex
from .snapshot import JournalSnapshot
from .resource_limits import JsonResourceLimits, validate_json_resource


_MAX_REPLAY_PAGE_SIZE = 1_000
_MAX_REPLAY_PAGES = 100_000
_TOOL_STARTED_TYPES = frozenset({"tool.started"})
_TOOL_SEALED_TYPES = frozenset({"tool.subagent_completion.sealed"})
_TOOL_RESULT_TYPES = frozenset({"tool.result", "tool_result"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SEMANTIC_EVENT_PROJECTION_LIMITS = JsonResourceLimits(
    max_items=10_000,
    max_bytes=32 * 1024 * 1024,
    max_depth=64,
    max_nodes=1_000_000,
)


class DurableJournalError(RuntimeError):
    """Base error for the provider-neutral durable journal runtime."""


class DurableJournalScopeError(DurableJournalError):
    """A projector or repository crossed the runtime's bound execution."""


class DurableJournalIntegrityError(DurableJournalError):
    """A durable receipt or replay page contradicted the requested mutation."""


class SemanticEventProjector(Protocol):
    def __call__(self, event: Mapping[str, Any]) -> SemanticEventDraft | None:
        ...


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_operation_ref(
    operation_id: object,
    *,
    domain: str,
    payload: Mapping[str, Any],
) -> OperationRef:
    """Bind an operation identity to the exact normalized semantic payload."""

    normalized_id = _required_text(
        operation_id,
        "operation_id",
        identifier=True,
    )
    normalized_domain = _required_text(domain, "domain", identifier=True)
    frozen = _freeze_json(
        {"domain": normalized_domain, "payload": payload},
        path="operation",
    )
    if not isinstance(frozen, Mapping):
        raise TypeError("operation payload must be an object")
    digest = hashlib.sha256(_canonical_json(_thaw_json(frozen))).hexdigest()
    return OperationRef(normalized_id, digest)


@dataclass(frozen=True)
class SemanticEventDraft:
    """Stable, host-projected semantic event before its durable append."""

    event_id: str
    event_type: str
    attempt: AttemptRef
    operation_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    resource_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event_id",
            _required_text(self.event_id, "event_id", identifier=True),
        )
        object.__setattr__(
            self,
            "event_type",
            _required_text(self.event_type, "event_type", identifier=True),
        )
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(self, "attempt", AttemptRef.from_dict(self.attempt))
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, "operation_id", identifier=True),
        )
        frozen = _freeze_json(self.payload, path="payload")
        if not isinstance(frozen, Mapping):
            raise TypeError("payload must be an object")
        object.__setattr__(self, "payload", frozen)
        object.__setattr__(
            self,
            "resource_refs",
            _record_tuple(self.resource_refs, ResourceRef, "resource_refs"),
        )

    @property
    def operation(self) -> OperationRef:
        return build_operation_ref(
            self.operation_id,
            domain="journal.semantic_event",
            payload={
                "event_id": self.event_id,
                "event_type": self.event_type,
                "attempt": self.attempt.to_dict(),
                "payload": _thaw_json(self.payload),
                "resource_refs": [ref.to_dict() for ref in self.resource_refs],
            },
        )

    def to_append_request(self) -> JournalAppendRequest:
        return JournalAppendRequest(
            event_id=self.event_id,
            event_type=self.event_type,
            attempt=self.attempt,
            operation=self.operation,
            payload=self.payload,
            resource_refs=self.resource_refs,
        )


@dataclass(frozen=True)
class PreparedSemanticEvent:
    """A draft plus the artifacts that must be claimed atomically with it."""

    draft: SemanticEventDraft
    artifacts: tuple[PendingArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.draft, SemanticEventDraft):
            raise TypeError("draft must be a SemanticEventDraft")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, PendingArtifact) for item in artifacts):
            raise TypeError("artifacts must be PendingArtifact records")
        object.__setattr__(self, "artifacts", artifacts)


def _runtime_event_is_ephemeral(event: Mapping[str, Any]) -> bool:
    raw_type = event.get("type")
    if not isinstance(raw_type, str):
        return False
    if raw_type == "provider_retry":
        return True
    event_type = re.sub(r"[^a-z0-9]+", "_", raw_type.casefold()).strip("_")
    if not event_type:
        return False
    if any(
        marker in event_type
        for marker in ("reasoning", "thinking", "chain_of_thought", "hidden_cot")
    ):
        return True
    return event_type.endswith("delta") and any(
        token in event_type
        for token in (
            "analysis",
            "content",
            "message",
            "response",
            "stream",
            "token",
        )
    )


def _same_persisted_event(
    request: JournalAppendRequest,
    persisted: JournalEvent,
) -> bool:
    return (
        persisted.event_id == request.event_id
        and persisted.event_type == request.event_type
        and persisted.attempt == request.attempt
        and persisted.operation == request.operation
        and persisted.payload == request.payload
        and persisted.resource_refs == request.resource_refs
    )


def _semantic_projection_resource_adapter(value: Any) -> Any:
    if isinstance(value, ResourceRef):
        return value.to_dict()
    return value


def journal_event_to_semantic_event(event: JournalEvent) -> dict[str, Any]:
    """Project a canonical journal record into compiler semantic-event shape.

    The outer journal event type is authoritative.  A payload cannot hide or
    replace it with a colliding ``type`` key.
    """

    if not isinstance(event, JournalEvent):
        raise TypeError("event must be a JournalEvent")
    validate_json_resource(
        {
            "type": event.event_type,
            "event_id": event.event_id,
            "store_seq": event.store_seq,
            "execution_id": event.attempt.generation.execution_id,
            "generation_id": event.attempt.generation.generation_id,
            "attempt_id": event.attempt.attempt_id,
            "payload": event.payload,
            "resource_refs": event.resource_refs,
        },
        boundary="semantic_event_projection",
        limits=SEMANTIC_EVENT_PROJECTION_LIMITS,
        record_adapter=_semantic_projection_resource_adapter,
    )
    projected = _thaw_json(event.payload)
    if not isinstance(projected, dict):
        raise DurableJournalIntegrityError("journal payload is not an object")
    projected.update(
        {
            "type": event.event_type,
            "event_id": event.event_id,
            "store_seq": event.store_seq,
            "execution_id": event.attempt.generation.execution_id,
            "generation_id": event.attempt.generation.generation_id,
            "attempt_id": event.attempt.attempt_id,
            "resource_refs": [ref.to_dict() for ref in event.resource_refs],
        }
    )
    return projected


class SideEffectRecoveryState(StrEnum):
    NOT_STARTED = "not-started"
    UNCERTAIN_AFTER_START = "uncertain-after-start"
    SEALED_COMPLETION_FINALIZABLE = "sealed-completion-finalizable"
    TERMINAL_RESULT_REUSABLE = "terminal-result-reusable"
    CORRUPT = "corrupt"


@dataclass(frozen=True)
class ToolSideEffectRecovery:
    state: SideEffectRecoveryState
    call_id: str
    started_event: JournalEvent | None = None
    sealed_event: JournalEvent | None = None
    result_event: JournalEvent | None = None
    reusable_result_artifact: ArtifactRef | None = None
    reason: str = ""
    intent_event: JournalEvent | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )
        if not isinstance(self.state, SideEffectRecoveryState):
            object.__setattr__(self, "state", SideEffectRecoveryState(self.state))
        if self.started_event is not None and not isinstance(
            self.started_event, JournalEvent
        ):
            raise TypeError("started_event must be a JournalEvent")
        if self.result_event is not None and not isinstance(
            self.result_event, JournalEvent
        ):
            raise TypeError("result_event must be a JournalEvent")
        if self.sealed_event is not None and not isinstance(
            self.sealed_event, JournalEvent
        ):
            raise TypeError("sealed_event must be a JournalEvent")
        if self.intent_event is not None and not isinstance(
            self.intent_event, JournalEvent
        ):
            raise TypeError("intent_event must be a JournalEvent")
        if self.reusable_result_artifact is not None and not isinstance(
            self.reusable_result_artifact, ArtifactRef
        ):
            object.__setattr__(
                self,
                "reusable_result_artifact",
                ArtifactRef.from_dict(self.reusable_result_artifact),
            )
        object.__setattr__(
            self,
            "reason",
            str(self.reason or "").strip()[:512],
        )

    @property
    def auto_execute_allowed(self) -> bool:
        return self.state is SideEffectRecoveryState.NOT_STARTED

    @property
    def reusable_result_ref(self) -> ResourceRef | None:
        if self.reusable_result_artifact is None:
            return None
        return self.reusable_result_artifact.ref


class DurableEventSink:
    """Synchronously project, validate, and append semantic runtime events."""

    def __init__(
        self,
        journal: BoundExecutionJournal,
        attempt: AttemptRef,
        projector: SemanticEventProjector,
    ) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        if not isinstance(attempt, AttemptRef):
            attempt = AttemptRef.from_dict(attempt)
        if journal.execution_id != attempt.generation.execution_id:
            raise DurableJournalScopeError(
                "journal execution does not match the bound attempt execution"
            )
        if not callable(projector):
            raise TypeError("projector must be callable")
        self._journal = journal
        self._attempt = attempt
        self._projector = projector

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    @property
    def journal(self) -> BoundExecutionJournal:
        return self._journal

    @property
    def projector(self) -> SemanticEventProjector:
        return self._projector

    def __call__(self, event: Mapping[str, Any]) -> JournalAppendResult | None:
        if not isinstance(event, Mapping):
            raise TypeError("durable event must be an object")
        if _runtime_event_is_ephemeral(event):
            return None
        draft = self._projector(event)
        if draft is None:
            return None
        if _runtime_event_is_ephemeral({"type": draft.event_type}):
            return None
        return self.append_projected(draft)

    def append_projected(
        self,
        draft: SemanticEventDraft,
    ) -> JournalAppendResult:
        """Append one draft produced by this sink's bound projector."""

        if not isinstance(draft, SemanticEventDraft):
            raise TypeError("draft must be a SemanticEventDraft")
        if _runtime_event_is_ephemeral({"type": draft.event_type}):
            raise DurableJournalIntegrityError(
                "ephemeral events cannot be appended as projected drafts"
            )
        if draft.attempt != self._attempt:
            raise DurableJournalScopeError(
                "projected event attempt does not match the bound attempt"
            )
        request = draft.to_append_request()
        result = self._journal.append(request=request)
        if not isinstance(result, JournalAppendResult):
            raise DurableJournalIntegrityError(
                "journal append did not return a JournalAppendResult"
            )
        if not _same_persisted_event(request, result.event):
            raise DurableJournalIntegrityError(
                "persisted event does not match the exact append request"
            )
        return result

    def append_prepared(
        self,
        prepared: PreparedSemanticEvent,
        *,
        precondition: Callable[[JournalSnapshot], None] | None = None,
    ) -> JournalAppendResult:
        """Atomically claim ``prepared.artifacts`` and append its draft.

        Falls back to :meth:`append_projected` semantics (no artifact claim,
        no precondition) is not possible: a journal that cannot provide
        ``append_with_artifacts`` raises rather than silently splitting the
        claim and the append into two transactions.
        """

        if not isinstance(prepared, PreparedSemanticEvent):
            raise TypeError("prepared must be a PreparedSemanticEvent")
        draft = prepared.draft
        if _runtime_event_is_ephemeral({"type": draft.event_type}):
            raise DurableJournalIntegrityError(
                "ephemeral events cannot be appended as projected drafts"
            )
        if draft.attempt != self._attempt:
            raise DurableJournalScopeError(
                "projected event attempt does not match the bound attempt"
            )
        request = draft.to_append_request()
        result = self._journal.append_with_artifacts(
            request=request,
            artifacts=prepared.artifacts,
            precondition=precondition,
        )
        if not isinstance(result, JournalAppendResult):
            raise DurableJournalIntegrityError(
                "journal append did not return a JournalAppendResult"
            )
        if not _same_persisted_event(request, result.event):
            raise DurableJournalIntegrityError(
                "persisted event does not match the exact append request"
            )
        return result

    def replay(self, *, page_size: int = 100) -> tuple[JournalEvent, ...]:
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= _MAX_REPLAY_PAGE_SIZE
        ):
            raise ValueError(
                f"page_size must be between 1 and {_MAX_REPLAY_PAGE_SIZE}"
            )
        selected: list[JournalEvent] = []
        seen_event_ids: set[str] = set()
        seen_operation_ids: set[str] = set()
        after: EventCursor | None = None
        previous_store_seq = 0
        for _page_number in range(_MAX_REPLAY_PAGES):
            page = self._journal.read(after=after, limit=page_size)
            if not isinstance(page, JournalPage):
                raise DurableJournalIntegrityError(
                    "journal replay did not return a JournalPage"
                )
            if len(page.events) > page_size:
                raise DurableJournalIntegrityError(
                    "journal replay exceeded the requested page size"
                )
            for event in page.events:
                generation = event.attempt.generation
                if generation.execution_id != self._attempt.generation.execution_id:
                    raise DurableJournalScopeError(
                        "replayed event belongs to a foreign execution"
                    )
                if generation.generation_id != self._attempt.generation.generation_id:
                    raise DurableJournalScopeError(
                        "replayed event belongs to a foreign generation"
                    )
                if event.store_seq <= previous_store_seq:
                    raise DurableJournalIntegrityError(
                        "journal replay cursor did not advance"
                    )
                previous_store_seq = event.store_seq
                if event.event_id in seen_event_ids:
                    raise DurableJournalIntegrityError(
                        "journal replay repeated an event identity"
                    )
                seen_event_ids.add(event.event_id)
                if event.attempt.attempt_id == self._attempt.attempt_id:
                    if event.operation.operation_id in seen_operation_ids:
                        raise DurableJournalIntegrityError(
                            "journal replay repeated an operation identity"
                        )
                    seen_operation_ids.add(event.operation.operation_id)
                    selected.append(event)
            if not page.has_more:
                return tuple(selected)
            if page.next_cursor is None or (
                after is not None
                and page.next_cursor.store_seq <= after.store_seq
            ):
                raise DurableJournalIntegrityError(
                    "journal replay pagination did not advance"
                )
            after = page.next_cursor
        raise DurableJournalIntegrityError("journal replay exceeded the page limit")

    def recover_tool_side_effect(
        self,
        call_id: object,
        *,
        page_size: int = 100,
    ) -> ToolSideEffectRecovery:
        normalized_call_id = _required_text(call_id, "call_id", identifier=True)
        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= _MAX_REPLAY_PAGE_SIZE
        ):
            raise ValueError(
                f"page_size must be between 1 and {_MAX_REPLAY_PAGE_SIZE}"
            )
        if not isinstance(self._journal, BoundToolReceiptIndex):
            raise DurableJournalIntegrityError(
                "durable tool recovery requires a receipt index"
            )
        lookup = self._journal.lookup_tool_execution_receipts(
            attempt=self._attempt,
            call_id=normalized_call_id,
        )
        if not isinstance(lookup, ToolExecutionReceiptLookup):
            raise DurableJournalIntegrityError(
                "tool receipt index returned an invalid lookup"
            )
        if lookup.attempt != self._attempt or lookup.call_id != normalized_call_id:
            raise DurableJournalIntegrityError(
                "tool receipt index crossed its requested subject"
            )
        if lookup.overflow:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                reason="tool boundary cardinality exceeds the indexed limit",
            )

        intents: list[JournalEvent] = []
        started: list[JournalEvent] = []
        sealed: list[JournalEvent] = []
        results: list[JournalEvent] = []
        tool_names: dict[str, str] = {}
        malformed_tool_identity = False
        for event in lookup.events:
            event_tool_name = event.payload.get("tool_name")
            if (
                not isinstance(event_tool_name, str)
                or event_tool_name != event_tool_name.strip()
                or not event_tool_name
            ):
                malformed_tool_identity = True
            else:
                tool_names[event.event_id] = event_tool_name
            if event.event_type == "tool_call":
                intents.append(event)
            elif event.event_type in _TOOL_STARTED_TYPES:
                started.append(event)
            elif event.event_type in _TOOL_SEALED_TYPES:
                sealed.append(event)
            else:
                results.append(event)
        if malformed_tool_identity:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                reason="tool boundary is missing a stable tool identity",
            )
        if (
            len(intents) > 1
            or len(started) > 1
            or len(sealed) > 1
            or len(results) > 1
        ):
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                intent_event=intents[0] if len(intents) == 1 else None,
                started_event=started[0] if len(started) == 1 else None,
                sealed_event=sealed[0] if len(sealed) == 1 else None,
                result_event=results[0] if len(results) == 1 else None,
                reason="tool boundary cardinality is invalid",
            )
        intent = intents[0] if intents else None
        start = started[0] if started else None
        seal = sealed[0] if sealed else None
        result = results[0] if results else None
        if intent is None:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="tool boundary is missing its durable call intent",
            )
        if start is not None and start.store_seq <= intent.store_seq:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                intent_event=intent,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="tool start does not follow tool call intent",
            )
        if start is None and (seal is not None or result is not None):
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                intent_event=intent,
                sealed_event=seal,
                result_event=result,
                reason="tool completion exists without a durable tool start",
            )
        if start is None:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.NOT_STARTED,
                normalized_call_id,
                intent_event=intent,
            )
        if seal is not None and seal.store_seq <= start.store_seq:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                intent_event=intent,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="sealed completion does not follow tool start",
            )
        if result is not None and seal is not None and result.store_seq <= seal.store_seq:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                intent_event=intent,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="tool result does not follow sealed completion",
            )
        if result is None:
            if seal is not None:
                return ToolSideEffectRecovery(
                    SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE,
                    normalized_call_id,
                    started_event=start,
                    sealed_event=seal,
                    intent_event=intent,
                )
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.UNCERTAIN_AFTER_START,
                normalized_call_id,
                started_event=start,
                sealed_event=seal,
                reason="tool start is durable but no terminal result is durable",
                intent_event=intent,
            )
        if len({tool_names[event.event_id] for event in lookup.events}) != 1:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="tool boundary has a mismatched stable tool identity",
                intent_event=intent,
            )
        if result.store_seq <= start.store_seq:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="tool result does not follow tool start",
                intent_event=intent,
            )
        reusable_artifact = self._reusable_result_ref(result)
        if reusable_artifact is None:
            return ToolSideEffectRecovery(
                SideEffectRecoveryState.CORRUPT,
                normalized_call_id,
                started_event=start,
                sealed_event=seal,
                result_event=result,
                reason="tool result lacks a complete durable artifact descriptor",
                intent_event=intent,
            )
        return ToolSideEffectRecovery(
            SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE,
            normalized_call_id,
            started_event=start,
            sealed_event=seal,
            result_event=result,
            reusable_result_artifact=reusable_artifact,
            intent_event=intent,
        )

    @staticmethod
    def _reusable_result_ref(event: JournalEvent) -> ArtifactRef | None:
        raw_ref = event.payload.get("full_output_ref")
        try:
            ref = (
                raw_ref
                if isinstance(raw_ref, ResourceRef)
                else ResourceRef.from_dict(raw_ref)
            )
        except (TypeError, ValueError):
            return None
        byte_length = event.payload.get("result_bytes")
        digest = event.payload.get("result_sha256")
        if (
            ref.kind != "artifact"
            or ref not in event.resource_refs
            or isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            return None
        try:
            return ArtifactRef(
                ref=ref,
                media_type="application/json",
                byte_length=byte_length,
                sha256=digest,
                preview=str(event.payload.get("preview") or ""),
            )
        except (TypeError, ValueError):
            return None


__all__ = [
    "DurableEventSink",
    "DurableJournalError",
    "DurableJournalIntegrityError",
    "DurableJournalScopeError",
    "PreparedSemanticEvent",
    "SemanticEventDraft",
    "SemanticEventProjector",
    "SideEffectRecoveryState",
    "ToolSideEffectRecovery",
    "build_operation_ref",
    "journal_event_to_semantic_event",
]
