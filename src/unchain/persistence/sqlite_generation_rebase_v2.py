"""Atomic generation rebase over the canonical Context V2 SQLite plane.

This service is the only writer in this module.  One ``BEGIN IMMEDIATE`` owns
the host generation lifecycle CAS, imported canonical journal events and
operations, the bootstrap/rebase manifest and head, and the initial attempt
binding.  It never composes the independently transactional legacy bootstrap
or host-generation services.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Iterator, Mapping

from unchain.context.artifacts import MAX_ARTIFACT_BYTES, ArtifactService
from unchain.context.attachments import HostResolvedAttachment
from unchain.context.graph_checkpoint import (
    GraphCheckpointError,
    GraphCheckpointService,
    GraphExecutionPlan,
    GraphStepBinding,
    GraphTerminalStatus,
    JournalGraphCheckpointRepository,
    locate_graph_execution_plan,
)
from unchain.context.models import HandoffEnvelope
from unchain.context.ports import ContextRepositoryError, ContextScopeError

from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    EventRange,
    GenerationRef,
    OperationRef,
    ResourceRef,
    SemanticEventDraft,
)
from unchain.journal.models import (
    JournalEvent,
    ModelValidationError,
    _bounded_int,
    _record_tuple,
    _required_text,
    _sha256,
    _thaw_json,
)
from unchain.journal.interaction_resolution_compat import (
    InteractionResolutionCompatibilityError,
    interaction_resolution_compatibility_record,
    legacy_interaction_resolution_supersession_pairs,
    legacy_interaction_resolution_supersessions,
)
from unchain.journal.graph_attempt_quiescence import (
    ATTEMPT_TERMINAL_EQUIVALENTS,
    CANONICAL_ATTEMPT_TERMINALS,
    GRAPH_STEP_COMPLETED,
    GRAPH_STEP_SEALS,
    GRAPH_STEP_SEAL_TERMINALS,
    MAX_ITERATIONS_TERMINALS,
    select_attempt_terminal,
)
from unchain.journal.interaction_cycles import (
    DURABLE_INTERACTION_REQUESTS,
    DURABLE_INTERACTION_RESOLUTIONS,
    INTERACTION_LIVENESS_EXEMPT_EVENT_TYPES,
    INTERACTION_REQUESTS,
    INTERACTION_RESOLUTIONS,
    LIVE_INTERACTION_OUTCOMES,
)
from unchain.persistence.sqlite_v2 import (
    SQLiteContextV2Store,
    SQLiteContextV2StoreIntegrityError,
    _SQLiteBoundContextV2Repository,
)


_MAX_MESSAGES = 10_000
_MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
_LEGACY_CAPTURE_STATUS = "legacy_partial"
_INTERACTION_REQUEST_EVENT_TYPES = DURABLE_INTERACTION_REQUESTS
_INTERACTION_RESOLUTION_EVENT_TYPES = DURABLE_INTERACTION_RESOLUTIONS
_GRAPH_INTERACTION_REQUEST_EVENT_TYPES = INTERACTION_REQUESTS
_GRAPH_INTERACTION_RESOLUTION_EVENT_TYPES = INTERACTION_RESOLUTIONS
_ATTEMPT_TERMINAL_EVENT_TYPES = CANONICAL_ATTEMPT_TERMINALS
_TOOL_INTENT_EVENT_TYPES = frozenset({"tool_call"})
_TOOL_STARTED_EVENT_TYPES = frozenset({"tool.started"})
_TOOL_SEALED_EVENT_TYPES = frozenset({"tool.subagent_completion.sealed"})
_TOOL_RESULT_EVENT_TYPES = frozenset({"tool.result", "tool_result"})
_TOOL_LIFECYCLE_EVENT_TYPES = (
    _TOOL_INTENT_EVENT_TYPES
    | _TOOL_STARTED_EVENT_TYPES
    | _TOOL_SEALED_EVENT_TYPES
    | _TOOL_RESULT_EVENT_TYPES
)
_POST_TERMINAL_AUDIT_EVENT_TYPES = frozenset(
    {
        "memory.curator.enqueued",
        "memory.curator.noop",
        "memory.curator.isolated",
        "memory.curator.failed",
    }
)


class GenerationRebaseFailureReason(StrEnum):
    """Closed reason vocabulary consumed across the Unchain/PuPu boundary."""

    INFRASTRUCTURE_UNAVAILABLE = "infrastructure_unavailable"
    JOURNAL_AUTHORITY_INVALID = "journal_authority_invalid"
    CURRENT_RECEIPT_UNAVAILABLE = "current_receipt_unavailable"
    HOST_SNAPSHOT_UNSANITIZED = "host_snapshot_unsanitized"
    CHECKPOINT_PREPARED = "checkpoint_prepared"
    PENDING_INTERACTION = "pending_interaction"
    ATTEMPT_OPEN = "attempt_open"
    TOOL_OPEN = "tool_open"
    OPERATION_IDENTITY_CONFLICT = "operation_identity_conflict"
    HEAD_REVISION_CONFLICT = "head_revision_conflict"
    SOURCE_GENERATION_CONFLICT = "source_generation_conflict"
    CHAT_BINDING_CONFLICT = "chat_binding_conflict"
    INTERACTION_RESOLUTION_DUPLICATED = "interaction_resolution_duplicated"
    INTERACTION_REQUEST_DUPLICATED = "interaction_request_duplicated"
    INTERACTION_LIFECYCLE_NOT_PAIRED = "interaction_lifecycle_not_paired"
    TOOL_CALL_IDENTITY_UNSTABLE = "tool_call_identity_unstable"
    TOOL_LIFECYCLE_NOT_PAIRED = "tool_lifecycle_not_paired"
    TOOL_START_PRECEDES_INTENT = "tool_start_precedes_intent"
    TOOL_SEAL_PRECEDES_START = "tool_seal_precedes_start"
    TOOL_RESULT_PRECEDES_START = "tool_result_precedes_start"
    TOOL_RESULT_PRECEDES_SEAL = "tool_result_precedes_seal"
    TOOL_IDENTITY_CHANGED = "tool_identity_changed"
    ATTEMPT_DUPLICATE_TERMINAL = "attempt_duplicate_terminal"
    ATTEMPT_CONTINUED_AFTER_TERMINAL = "attempt_continued_after_terminal"
    GRAPH_ATTEMPT_KIND_AMBIGUOUS = "graph_attempt_kind_ambiguous"
    GRAPH_PLAN_DESCRIPTOR_INVALID = "graph_plan_descriptor_invalid"
    GRAPH_STEP_TERMINAL_AMBIGUOUS = "graph_step_terminal_ambiguous"
    GRAPH_STEP_SEAL_DUPLICATED = "graph_step_seal_duplicated"
    GRAPH_STEP_SEAL_NOT_LAST = "graph_step_seal_not_last"
    GRAPH_STEP_SEAL_NOT_ADJACENT = "graph_step_seal_not_adjacent"
    GRAPH_STEP_SEAL_MISMATCHED_TERMINAL = (
        "graph_step_seal_mismatched_terminal"
    )
    GRAPH_STEP_SEAL_FOREIGN = "graph_step_seal_foreign"
    GRAPH_STEP_SEQUENCE_INVALID = "graph_step_sequence_invalid"
    GRAPH_EXECUTION_SEAL_DUPLICATED = "graph_execution_seal_duplicated"
    GRAPH_EXECUTION_SEAL_MISMATCHED = "graph_execution_seal_mismatched"
    GRAPH_STEP_SEAL_MISSING = "graph_step_seal_missing"
    GRAPH_EXECUTION_SEAL_MISSING = "graph_execution_seal_missing"


_FAILURE_DETAIL_SCHEMA = "unchain.generation_rebase_failure.v1"
_FAILURE_SUBJECT_KEYS = frozenset(
    {
        "execution_id",
        "generation_id",
        "attempt_id",
        "orchestration_attempt_id",
        "call_id",
        "interaction_id",
        "step_index",
        "event_type",
        "store_seq",
        "event_id",
        "graph_plan_id",
        "graph_scope_id",
    }
)


def _generation_rebase_failure_detail(
    reason: GenerationRebaseFailureReason,
    subject: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not isinstance(reason, GenerationRebaseFailureReason):
        raise TypeError("generation rebase failure reason is not closed")
    normalized: dict[str, str | int] = {}
    if subject is not None:
        if not isinstance(subject, Mapping) or len(subject) > 12:
            raise TypeError("generation rebase failure subject is invalid")
        for raw_key, raw_value in subject.items():
            if raw_key not in _FAILURE_SUBJECT_KEYS:
                raise ValueError("generation rebase failure subject key is forbidden")
            if isinstance(raw_value, str):
                if not raw_value or len(raw_value) > 256 or "\x00" in raw_value:
                    raise ValueError(
                        "generation rebase failure subject text is invalid"
                    )
                normalized[raw_key] = raw_value
            elif (
                not isinstance(raw_value, bool)
                and isinstance(raw_value, int)
                and raw_value >= 0
            ):
                normalized[raw_key] = raw_value
            else:
                raise TypeError("generation rebase failure subject value is invalid")
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(raw) > 2048:
        raise ValueError("generation rebase failure subject is too large")
    return MappingProxyType(
        {
            "schema": _FAILURE_DETAIL_SCHEMA,
            "reason": reason.value,
            "subject": MappingProxyType(normalized),
        }
    )


class GenerationRebaseError(RuntimeError):
    """Base failure at the atomic generation-rebase boundary."""

    code = "unavailable"
    retryable = True
    default_reason = GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE

    def __init__(
        self,
        message: str,
        *,
        reason: GenerationRebaseFailureReason | None = None,
        subject: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        selected = self.default_reason if reason is None else reason
        self.reason = selected.value
        self.detail = _generation_rebase_failure_detail(selected, subject)


class GenerationRebaseConflict(GenerationRebaseError):
    """A generation identity, CAS precondition, or operation payload drifted."""

    code = "generation_conflict"
    default_reason = GenerationRebaseFailureReason.SOURCE_GENERATION_CONFLICT


class GenerationRebaseUnavailable(GenerationRebaseError):
    """SQLite or storage infrastructure could not complete the boundary."""

    default_reason = GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE


class GenerationRebasePreflightBlocked(GenerationRebaseConflict):
    """Current durable state does not permit a generation cutover."""

    code = "in_progress"
    default_reason = GenerationRebaseFailureReason.ATTEMPT_OPEN


class GenerationRebaseRecoveryRequired(GenerationRebasePreflightBlocked):
    """A canonical graph terminal needs exactly one deterministic seal."""

    code = "recovery_required"


class GenerationRebaseJournalIncompatible(GenerationRebaseConflict):
    """Durable business state is deterministic but cannot be safely rebased."""

    code = "journal_incompatible"
    retryable = False
    default_reason = GenerationRebaseFailureReason.JOURNAL_AUTHORITY_INVALID


def generation_rebase_failure_detail(
    error: object,
) -> Mapping[str, Any] | None:
    """Read a validated detail without requiring new exception subclasses."""

    detail = getattr(error, "detail", None)
    if not isinstance(detail, Mapping) or set(detail) != {
        "schema",
        "reason",
        "subject",
    }:
        return None
    if detail.get("schema") != _FAILURE_DETAIL_SCHEMA:
        return None
    raw_reason = detail.get("reason")
    try:
        reason = GenerationRebaseFailureReason(raw_reason)
    except (TypeError, ValueError):
        return None
    subject = detail.get("subject")
    try:
        return _generation_rebase_failure_detail(reason, subject)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class GenerationRebaseRecoveryResult:
    """Bounded result of one host-invoked graph recovery action."""

    action: str
    reason: str
    execution_id: str
    generation_id: str
    appended_event_count: int
    artifact_count: int
    schema: str = "unchain.generation_rebase_recovery.v1"

    def __post_init__(self) -> None:
        if self.schema != "unchain.generation_rebase_recovery.v1":
            raise ValueError("generation rebase recovery schema is unsupported")
        if self.action not in {
            "step_recovered",
            "execution_finalized",
            "unchanged",
        }:
            raise ValueError("generation rebase recovery action is invalid")
        if self.reason not in {
            GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING.value,
            GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISSING.value,
        }:
            raise ValueError("generation rebase recovery reason is invalid")
        for field_name in ("execution_id", "generation_id"):
            _required_text(getattr(self, field_name), field_name, identifier=True)
        for field_name in ("appended_event_count", "artifact_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or value not in {0, 1}:
                raise ValueError(
                    "generation rebase recovery counts must be zero or one"
                )


@dataclass(frozen=True)
class _GraphStepQuiescence:
    status: str
    step: GraphStepBinding
    terminal: JournalEvent | None = None
    seal: JournalEvent | None = None
    output_artifact: ArtifactRef | None = None


def _failure_subject_for_event(
    event: JournalEvent,
    *,
    plan: GraphExecutionPlan | None = None,
    step_index: int | None = None,
) -> dict[str, Any]:
    subject: dict[str, Any] = {
        "execution_id": event.attempt.generation.execution_id,
        "generation_id": event.attempt.generation.generation_id,
        "attempt_id": event.attempt.attempt_id,
        "event_type": event.event_type,
        "store_seq": event.store_seq,
        "event_id": event.event_id,
    }
    if plan is not None:
        subject.update(
            {
                "orchestration_attempt_id": (
                    plan.orchestration_attempt.attempt_id
                ),
                "graph_plan_id": plan.plan_id,
                "graph_scope_id": plan.scope_id,
            }
        )
    if step_index is not None:
        subject["step_index"] = step_index
    return subject


def _journal_incompatible(
    message: str,
    *,
    reason: GenerationRebaseFailureReason,
    event: JournalEvent,
    plan: GraphExecutionPlan | None = None,
    step_index: int | None = None,
) -> GenerationRebaseJournalIncompatible:
    return GenerationRebaseJournalIncompatible(
        message,
        reason=reason,
        subject=_failure_subject_for_event(
            event,
            plan=plan,
            step_index=step_index,
        ),
    )


def _parse_graph_plan_admission(event: JournalEvent) -> GraphExecutionPlan:
    if (
        event.event_type != "graph.execution.admitted"
        or set(event.payload)
        != {"graph_plan_id", "graph_scope_id", "plan"}
        or event.resource_refs
    ):
        raise _journal_incompatible(
            "generation rebase graph plan descriptor is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID,
            event=event,
        )
    try:
        plan = GraphExecutionPlan.from_dict(event.payload.get("plan"))
    except (AttributeError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph plan descriptor is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID,
            event=event,
        ) from error
    if (
        plan.orchestration_attempt != event.attempt
        or event.payload.get("graph_plan_id") != plan.plan_id
        or event.payload.get("graph_scope_id") != plan.scope_id
    ):
        raise _journal_incompatible(
            "generation rebase graph plan descriptor changed identity",
            reason=GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID,
            event=event,
            plan=plan,
        )
    return plan


def _closed_cursor(value: object) -> EventCursor:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "store_seq",
        "event_id",
    }:
        raise ValueError("cursor is not closed")
    cursor = EventCursor.from_dict(value)
    if cursor.to_dict() != dict(value):
        raise ValueError("cursor is not canonical")
    return cursor


def _event_at_graph_cursor(
    events: tuple[JournalEvent, ...],
    cursor: EventCursor,
) -> JournalEvent:
    matches = tuple(
        event
        for event in events
        if event.store_seq == cursor.store_seq and event.event_id == cursor.event_id
    )
    if len(matches) != 1:
        raise ValueError("graph cursor has no exact journal event")
    return matches[0]


def _graph_interaction_id(event: JournalEvent) -> str:
    raw = event.payload.get("interaction_id")
    request = event.payload.get("interaction_request")
    if raw is None and isinstance(request, Mapping):
        raw = request.get("interaction_id")
    if raw is None:
        for field_name in ("confirmation_id", "request_id", "call_id"):
            candidate = event.payload.get(field_name)
            if candidate is not None:
                raw = candidate
                break
    return _required_text(raw, "interaction_id", identifier=True)


def _graph_interaction_aliases(event: JournalEvent) -> frozenset[str]:
    candidates: list[object] = []

    def collect(value: object) -> None:
        if not isinstance(value, Mapping):
            return
        for field_name in (
            "interaction_id",
            "confirmation_id",
            "request_id",
            "call_id",
            "tool_call_id",
        ):
            candidates.append(value.get(field_name))

    collect(event.payload)
    request = event.payload.get("interaction_request")
    collect(request)
    if isinstance(request, Mapping):
        collect(request.get("payload"))
        collect(request.get("subject"))
    aliases: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        aliases.add(_required_text(candidate, "interaction_alias", identifier=True))
    return frozenset(aliases)


def _validate_graph_step_interaction_cycles(
    events: tuple[JournalEvent, ...],
    *,
    journal_events: tuple[JournalEvent, ...],
    start: JournalEvent,
    plan: GraphExecutionPlan,
    expected_step: GraphStepBinding,
    artifact_repository: _SQLiteBoundContextV2Repository,
) -> None:
    resumes = tuple(
        event
        for event in events
        if event.event_type == "graph.step.resume.admitted"
    )
    anchor = resumes[0] if resumes else start
    step_admitted_interaction_ids = frozenset(
        str(resume.payload.get("interaction_id")) for resume in resumes
    )

    def incompatible(message: str, event: JournalEvent) -> None:
        raise _journal_incompatible(
            message,
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=event,
            plan=plan,
            step_index=expected_step.index,
        )

    try:
        admitted_resolution_store_seqs = {
            _closed_cursor(resume.payload.get("resolution_cursor")).store_seq
            for resume in resumes
        }
        supersession_pairs = legacy_interaction_resolution_supersession_pairs(
            tuple(
                interaction_resolution_compatibility_record(
                    ordinal=event.store_seq,
                    event_type=event.event_type,
                    interaction_id=_graph_interaction_id(event),
                    execution_id=event.attempt.generation.execution_id,
                    generation_id=event.attempt.generation.generation_id,
                    attempt_id=event.attempt.attempt_id,
                    payload=event.payload,
                    resource_refs=event.resource_refs,
                )
                for event in journal_events
                if event.event_type
                in {"interaction_resolved", "interaction.resolved"}
            )
        )
    except (
        AttributeError,
        InteractionResolutionCompatibilityError,
        TypeError,
        ValueError,
    ) as error:
        raise _journal_incompatible(
            "generation rebase graph interaction resolution is ambiguous",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=anchor,
            plan=plan,
            step_index=expected_step.index,
        ) from error
    suppressed_resolution_store_seqs = frozenset(
        pair.canonical_ordinal
        if pair.legacy_ordinal in admitted_resolution_store_seqs
        else pair.legacy_ordinal
        for pair in supersession_pairs
    )

    active_interaction_id: str | None = None
    active_request_cursor: EventCursor | None = None
    active_aliases = frozenset[str]()
    active_resolution_cursor: EventCursor | None = None
    active_resume_cursor: EventCursor | None = None
    seen_interaction_ids: set[str] = set()
    seen_request_cursors: set[tuple[int, str]] = set()
    seen_resolution_cursors: set[tuple[int, str]] = set()
    admitted_interaction_ids: set[str] = set()
    admitted_request_cursors: set[tuple[int, str]] = set()
    admitted_resolution_cursors: set[tuple[int, str]] = set()
    admitted_cycles: set[tuple[str, tuple[int, str], tuple[int, str]]] = set()

    def clear_active_interaction() -> None:
        nonlocal active_interaction_id
        nonlocal active_request_cursor
        nonlocal active_aliases
        nonlocal active_resolution_cursor
        nonlocal active_resume_cursor
        active_interaction_id = None
        active_request_cursor = None
        active_aliases = frozenset()
        active_resolution_cursor = None
        active_resume_cursor = None

    for event in events:
        if event.store_seq < start.store_seq:
            continue
        if event.store_seq in suppressed_resolution_store_seqs:
            continue
        if (
            active_interaction_id is not None
            and active_resume_cursor is None
            and active_interaction_id not in step_admitted_interaction_ids
            and event.event_type not in INTERACTION_LIVENESS_EXEMPT_EVENT_TYPES
        ):
            # Match graph checkpoint recovery: runtime activity proves that a
            # cycle without a durable resume admission was answered in-run.
            # A future admission keeps the cycle strict, so malformed durable
            # pauses cannot be reclassified as live.
            clear_active_interaction()
        if event.event_type in _GRAPH_INTERACTION_REQUEST_EVENT_TYPES:
            if event.resource_refs:
                incompatible(
                    "generation rebase graph interaction request carries resources",
                    event,
                )
            try:
                interaction_id = _graph_interaction_id(event)
                request_aliases = _graph_interaction_aliases(event)
            except (TypeError, ValueError) as error:
                raise _journal_incompatible(
                    "generation rebase graph interaction request is invalid",
                    reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                    event=event,
                    plan=plan,
                    step_index=expected_step.index,
                ) from error
            request_key = (event.store_seq, event.event_id)
            if (
                active_interaction_id is not None
                and active_resume_cursor is None
            ):
                incompatible(
                    "generation rebase graph interaction cycles overlap",
                    event,
                )
            if (
                interaction_id in seen_interaction_ids
                or request_key in seen_request_cursors
            ):
                incompatible(
                    "generation rebase graph interaction request is ambiguous",
                    event,
                )
            seen_interaction_ids.add(interaction_id)
            seen_request_cursors.add(request_key)
            active_interaction_id = interaction_id
            active_request_cursor = EventCursor(event.store_seq, event.event_id)
            active_aliases = request_aliases
            active_resolution_cursor = None
            active_resume_cursor = None
            continue
        if event.event_type in _GRAPH_INTERACTION_RESOLUTION_EVENT_TYPES:
            if event.event_type in {"tool_confirmed", "tool_denied"}:
                if event.resource_refs:
                    incompatible(
                        "generation rebase graph tool resolution carries resources",
                        event,
                    )
            else:
                _verified_graph_input_event_artifacts(
                    event,
                    artifact_repository=artifact_repository,
                    plan=plan,
                    step_index=expected_step.index,
                    reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                )
            try:
                interaction_id = _graph_interaction_id(event)
            except (TypeError, ValueError) as error:
                raise _journal_incompatible(
                    "generation rebase graph interaction resolution is invalid",
                    reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                    event=event,
                    plan=plan,
                    step_index=expected_step.index,
                ) from error
            compatible_resolution = event.event_type in LIVE_INTERACTION_OUTCOMES
            if active_interaction_id is None:
                if compatible_resolution:
                    continue
                incompatible(
                    "generation rebase graph resolution has no exact request",
                    event,
                )
            if (
                interaction_id != active_interaction_id
                and not (
                    compatible_resolution
                    and interaction_id in active_aliases
                )
            ):
                if compatible_resolution:
                    continue
                incompatible(
                    "generation rebase graph resolution changed interaction",
                    event,
                )
            if active_resolution_cursor is not None:
                if compatible_resolution and active_resume_cursor is not None:
                    continue
                incompatible(
                    "generation rebase graph resolution is ambiguous",
                    event,
                )
            if (
                compatible_resolution
                and active_interaction_id not in step_admitted_interaction_ids
            ):
                # A live tool outcome closes the cycle in place. Durable
                # resolutions use interaction.resolved and still require the
                # exact admission path below.
                clear_active_interaction()
                continue
            resolution_key = (event.store_seq, event.event_id)
            if (
                resolution_key in seen_resolution_cursors
                or active_request_cursor is None
                or event.store_seq <= active_request_cursor.store_seq
            ):
                incompatible(
                    "generation rebase graph resolution cursor is ambiguous",
                    event,
                )
            seen_resolution_cursors.add(resolution_key)
            active_resolution_cursor = EventCursor(
                event.store_seq,
                event.event_id,
            )
            continue
        if event.event_type != "graph.step.resume.admitted":
            continue
        try:
            interaction_id = _required_text(
                event.payload.get("interaction_id"),
                "interaction_id",
                identifier=True,
            )
            request_cursor = _closed_cursor(event.payload.get("request_cursor"))
            resolution_cursor = _closed_cursor(
                event.payload.get("resolution_cursor")
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise _journal_incompatible(
                "generation rebase graph resume admission is invalid",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                event=event,
                plan=plan,
                step_index=expected_step.index,
            ) from error
        request_key = (request_cursor.store_seq, request_cursor.event_id)
        resolution_key = (
            resolution_cursor.store_seq,
            resolution_cursor.event_id,
        )
        cycle_key = (interaction_id, request_key, resolution_key)
        if (
            active_interaction_id != interaction_id
            or active_request_cursor != request_cursor
            or active_resolution_cursor != resolution_cursor
            or active_resume_cursor is not None
            or interaction_id in admitted_interaction_ids
            or request_key in admitted_request_cursors
            or resolution_key in admitted_resolution_cursors
            or cycle_key in admitted_cycles
        ):
            incompatible(
                "generation rebase graph resume admission is ambiguous",
                event,
            )
        admitted_interaction_ids.add(interaction_id)
        admitted_request_cursors.add(request_key)
        admitted_resolution_cursors.add(resolution_key)
        admitted_cycles.add(cycle_key)
        active_resume_cursor = EventCursor(event.store_seq, event.event_id)


def _assert_bounded_graph_artifact_object(
    artifact: ArtifactRef,
    *,
    artifact_repository: _SQLiteBoundContextV2Repository,
    event: JournalEvent,
    plan: GraphExecutionPlan,
    step_index: int | None,
    reason: GenerationRebaseFailureReason,
) -> None:
    if artifact.byte_length > MAX_ARTIFACT_BYTES:
        raise _journal_incompatible(
            "generation rebase graph artifact exceeds the bounded read limit",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        )
    try:
        actual_size = artifact_repository._store._object_path(
            artifact.sha256
        ).stat().st_size
    except FileNotFoundError as error:
        raise _journal_incompatible(
            "generation rebase graph artifact object is missing",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        ) from error
    except OSError as error:
        raise GenerationRebaseUnavailable(
            "generation rebase graph artifact object stat failed",
            reason=GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE,
            subject=_failure_subject_for_event(
                event,
                plan=plan,
                step_index=step_index,
            ),
        ) from error
    if actual_size != artifact.byte_length:
        raise _journal_incompatible(
            "generation rebase graph artifact object length changed",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        )


def _verified_graph_artifact_bytes(
    artifact: ArtifactRef,
    *,
    artifact_repository: _SQLiteBoundContextV2Repository,
    event: JournalEvent,
    plan: GraphExecutionPlan,
    step_index: int | None,
    reason: GenerationRebaseFailureReason,
) -> bytes:
    _assert_bounded_graph_artifact_object(
        artifact,
        artifact_repository=artifact_repository,
        event=event,
        plan=plan,
        step_index=step_index,
        reason=reason,
    )
    try:
        content = artifact_repository.read_full_verified(artifact=artifact)
    except (ContextScopeError, SQLiteContextV2StoreIntegrityError) as error:
        raise _journal_incompatible(
            "generation rebase graph seal artifact failed exact verification",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        ) from error
    except ContextRepositoryError as error:
        raise GenerationRebaseUnavailable(
            "generation rebase graph seal artifact read failed",
            reason=GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE,
            subject=_failure_subject_for_event(
                event,
                plan=plan,
                step_index=step_index,
            ),
        ) from error
    if (
        type(content) is not bytes
        or len(content) != artifact.byte_length
        or hashlib.sha256(content).hexdigest() != artifact.sha256
    ):
        raise _journal_incompatible(
            "generation rebase graph seal artifact bytes changed",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        )
    return content


def _verified_graph_resource_artifact(
    resource_ref: ResourceRef,
    *,
    artifact_repository: _SQLiteBoundContextV2Repository,
    event: JournalEvent,
    plan: GraphExecutionPlan,
    step_index: int | None,
    reason: GenerationRebaseFailureReason,
) -> ArtifactRef:
    if resource_ref.kind != "artifact" or resource_ref.fragment:
        raise _journal_incompatible(
            "generation rebase graph provenance resource is not a whole artifact",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        )
    try:
        with artifact_repository._store._transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT * FROM artifacts
                WHERE execution_id = ? AND artifact_id = ? AND revision = ?
                """,
                (
                    artifact_repository.execution_id,
                    resource_ref.resource_id,
                    resource_ref.revision,
                ),
            ).fetchone()
            if row is None:
                raise ContextScopeError(
                    "graph provenance artifact does not belong to the execution"
                )
            artifact = artifact_repository._artifact_from_row(row)
            if artifact.ref != resource_ref:
                raise SQLiteContextV2StoreIntegrityError(
                    "graph provenance artifact resource identity changed"
                )
            object_row = connection.execute(
                "SELECT byte_length FROM objects WHERE sha256 = ?",
                (artifact.sha256,),
            ).fetchone()
            if (
                object_row is None
                or object_row["byte_length"] != artifact.byte_length
            ):
                raise SQLiteContextV2StoreIntegrityError(
                    "graph provenance artifact object metadata changed"
                )
        _assert_bounded_graph_artifact_object(
            artifact,
            artifact_repository=artifact_repository,
            event=event,
            plan=plan,
            step_index=step_index,
            reason=reason,
        )
        artifact_repository._store._read_object(
            digest=artifact.sha256,
            byte_length=artifact.byte_length,
        )
    except (ContextScopeError, SQLiteContextV2StoreIntegrityError) as error:
        raise _journal_incompatible(
            "generation rebase graph provenance artifact failed exact verification",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        ) from error
    except (ContextRepositoryError, sqlite3.Error) as error:
        raise GenerationRebaseUnavailable(
            "generation rebase graph provenance artifact read failed",
            reason=GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE,
            subject=_failure_subject_for_event(
                event,
                plan=plan,
                step_index=step_index,
            ),
        ) from error
    return artifact


def _closed_graph_artifact_resource(value: object) -> ResourceRef:
    if not isinstance(value, Mapping):
        raise TypeError("graph artifact resource must be an object")
    resource_ref = ResourceRef.from_dict(value)
    if (
        resource_ref.to_dict() != dict(value)
        or resource_ref.kind != "artifact"
        or resource_ref.fragment
    ):
        raise ValueError("graph artifact resource is not one whole artifact")
    return resource_ref


def _verified_graph_input_event_artifacts(
    event: JournalEvent,
    *,
    artifact_repository: _SQLiteBoundContextV2Repository,
    plan: GraphExecutionPlan,
    step_index: int | None,
    reason: GenerationRebaseFailureReason,
) -> tuple[ArtifactRef, ...]:
    """Verify the exact artifact closure of one graph-trusted input event."""

    message: Mapping[str, Any] | None = None
    try:
        if event.event_type == "message.user":
            message = event.payload.get("message")
            if not isinstance(message, Mapping):
                raise TypeError("graph input message is not an object")
            raw_attachments = event.payload.get("attachments")
            raw_attachment_refs = event.payload.get("attachment_refs")
            message_attachments = message.get("attachments")
            has_attachments = any(
                value is not None
                for value in (
                    raw_attachments,
                    raw_attachment_refs,
                    message_attachments,
                )
            )
            if has_attachments:
                if (
                    not isinstance(raw_attachments, (list, tuple))
                    or not raw_attachments
                    or not isinstance(raw_attachment_refs, (list, tuple))
                    or not isinstance(message_attachments, (list, tuple))
                ):
                    raise ValueError(
                        "graph input attachment descriptors are incomplete"
                    )
                attachments = tuple(
                    HostResolvedAttachment.from_dict(value)
                    for value in raw_attachments
                )
                attachment_refs = tuple(
                    _closed_graph_artifact_resource(value)
                    for value in raw_attachment_refs
                )
                if (
                    tuple(attachment.to_dict() for attachment in attachments)
                    != tuple(_thaw_json(value) for value in raw_attachments)
                    or tuple(_thaw_json(value) for value in message_attachments)
                    != tuple(attachment.to_dict() for attachment in attachments)
                    or attachment_refs
                    != tuple(attachment.artifact.ref for attachment in attachments)
                    or len(set(attachment_refs)) != len(attachment_refs)
                    or set(message) != {"role", "content", "attachments"}
                ):
                    raise ValueError(
                        "graph input attachment descriptors changed identity"
                    )
            else:
                attachments = ()
                attachment_refs = ()
                if (
                    "attachments" in event.payload
                    or "attachment_refs" in event.payload
                    or "attachments" in message
                    or set(message) != {"role", "content"}
                ):
                    raise ValueError(
                        "graph input attachment descriptors are not closed"
                    )
            if message.get("role") != "user" or not isinstance(
                message.get("content"), str
            ):
                raise ValueError("graph input message is not canonical")
            content_ref = _closed_graph_artifact_resource(
                event.payload.get("content_ref")
            )
            expected_refs = (content_ref, *attachment_refs)
            if event.resource_refs != expected_refs:
                raise ValueError("graph input resource closure changed")
        elif event.event_type in {
            "interaction.resolved",
            "interaction_resolved",
        }:
            descriptor_keys = {
                "content_ref",
                "content_bytes",
                "content_sha256",
                "preview",
                "preview_truncated",
            }
            descriptor_present = bool(
                descriptor_keys.intersection(event.payload)
            )
            if (
                event.event_type == "interaction_resolved"
                and not descriptor_present
                and not event.resource_refs
            ):
                return ()
            if not descriptor_keys.issubset(event.payload):
                raise ValueError(
                    "graph interaction response descriptor is incomplete"
                )
            content_ref = _closed_graph_artifact_resource(
                event.payload.get("content_ref")
            )
            attachments = ()
            expected_refs = (content_ref,)
            if event.resource_refs != expected_refs:
                raise ValueError(
                    "graph interaction response resource closure changed"
                )
        else:
            raise ValueError("graph input event type is unsupported")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph input artifact descriptor is invalid",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        ) from error

    verified = tuple(
        _verified_graph_resource_artifact(
            resource_ref,
            artifact_repository=artifact_repository,
            event=event,
            plan=plan,
            step_index=step_index,
            reason=reason,
        )
        for resource_ref in expected_refs
    )
    content_artifact = verified[0]
    content = _verified_graph_artifact_bytes(
        content_artifact,
        artifact_repository=artifact_repository,
        event=event,
        plan=plan,
        step_index=step_index,
        reason=reason,
    )
    try:
        if event.event_type == "message.user":
            content_binding_valid = (
                content_artifact.media_type == "application/json"
                and message is not None
                and content == _canonical_bytes(_thaw_json(message))
            )
        else:
            decoded = json.loads(content.decode("utf-8"))
            content_binding_valid = (
                content_artifact.media_type == "application/json"
                and type(decoded) is dict
                and set(decoded) == {
                    "interaction_id",
                    "response",
                    "submitted_by",
                }
                and decoded.get("interaction_id")
                == event.payload.get("interaction_id")
                and decoded.get("submitted_by")
                == event.payload.get("submitted_by")
                and _canonical_bytes(decoded) == content
            )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        content_binding_valid = False
    preview = event.payload.get("preview")
    preview_truncated = event.payload.get("preview_truncated")
    if (
        not content_binding_valid
        or event.payload.get("content_bytes") != content_artifact.byte_length
        or event.payload.get("content_sha256") != content_artifact.sha256
        or preview != content_artifact.preview
        or type(preview_truncated) is not bool
        or preview_truncated
        != (
            content_artifact.byte_length
            > len(content_artifact.preview.encode("utf-8"))
        )
        or any(
            artifact != attachment.artifact
            for artifact, attachment in zip(
                verified[1:],
                attachments,
                strict=True,
            )
        )
    ):
        raise _journal_incompatible(
            "generation rebase graph input artifact binding changed",
            reason=reason,
            event=event,
            plan=plan,
            step_index=step_index,
        )
    return verified


def _classify_graph_step_attempt(
    events: tuple[JournalEvent, ...],
    *,
    journal_events: tuple[JournalEvent, ...],
    plan: GraphExecutionPlan,
    expected_step: GraphStepBinding,
    artifact_repository: _SQLiteBoundContextV2Repository,
) -> _GraphStepQuiescence:
    graph_events = tuple(
        event for event in events if event.event_type.startswith("graph.")
    )
    allowed_graph_types = {
        "graph.step.started",
        "graph.step.resume.admitted",
        *GRAPH_STEP_SEALS,
    }
    starts = tuple(
        event for event in graph_events if event.event_type == "graph.step.started"
    )
    if (
        len(starts) != 1
        or any(event.event_type not in allowed_graph_types for event in graph_events)
    ):
        anchor = graph_events[0] if graph_events else events[0]
        raise _journal_incompatible(
            "generation rebase graph attempt kind is ambiguous",
            reason=GenerationRebaseFailureReason.GRAPH_ATTEMPT_KIND_AMBIGUOUS,
            event=anchor,
            plan=plan,
            step_index=expected_step.index,
        )
    start = starts[0]
    if set(start.payload) != {
        "graph_plan_id",
        "graph_scope_id",
        "step",
        "handoff_cursor",
        "input_cursor",
    }:
        raise _journal_incompatible(
            "generation rebase graph step start descriptor is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        )
    try:
        persisted_step = GraphStepBinding.from_dict(start.payload.get("step"))
    except (AttributeError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph step binding is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        ) from error
    if (
        persisted_step != expected_step
        or persisted_step.to_dict() != start.payload.get("step")
        or start.attempt != expected_step.attempt
        or start.payload.get("graph_plan_id") != plan.plan_id
        or start.payload.get("graph_scope_id") != plan.scope_id
    ):
        raise _journal_incompatible(
            "generation rebase graph step binding changed identity",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        )

    try:
        handoff_cursor = _closed_cursor(start.payload.get("handoff_cursor"))
        input_cursor = _closed_cursor(start.payload.get("input_cursor"))
        handoff_event = _event_at_graph_cursor(journal_events, handoff_cursor)
        input_event = _event_at_graph_cursor(journal_events, input_cursor)
    except (AttributeError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph step start cursor provenance is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        ) from error
    if (
        handoff_event.attempt != expected_step.attempt
        or handoff_event.event_type != "handoff.recorded"
        or input_event.attempt != expected_step.attempt
        or input_event.event_type != "message.user"
        or not handoff_cursor.store_seq < input_cursor.store_seq < start.store_seq
        or len(start.resource_refs) != 1
        or start.resource_refs[0].kind != "artifact"
        or start.resource_refs[0].fragment
        or start.resource_refs[0] not in handoff_event.resource_refs
        or start.resource_refs[0] not in input_event.resource_refs
    ):
        raise _journal_incompatible(
            "generation rebase graph step start provenance changed",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        )
    try:
        raw_handoff_envelope = handoff_event.payload.get("handoff_envelope")
        if not isinstance(raw_handoff_envelope, Mapping):
            raise TypeError("graph handoff envelope is not an object")
        handoff_envelope = HandoffEnvelope.from_dict(raw_handoff_envelope)
        handoff_artifact = ArtifactRef.from_dict(
            handoff_event.payload.get("full_output_artifact")
        )
        if (
            handoff_envelope.to_dict() != _thaw_json(raw_handoff_envelope)
            or handoff_artifact.to_dict()
            != _thaw_json(
                handoff_event.payload.get("full_output_artifact")
            )
        ):
            raise ValueError("graph handoff descriptor is not exact")
    except (AttributeError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph start artifact descriptor is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        ) from error
    if expected_step.index == 0:
        expected_source_attempt = plan.orchestration_attempt
        expected_source_range = EventRange(
            plan.initial_input_cursor,
            plan.initial_input_cursor,
        )
        expected_source_artifact_refs: tuple[ResourceRef, ...] = ()
        expected_source_artifact: ArtifactRef | None = None
    else:
        predecessor = plan.steps[expected_step.index - 1]
        predecessor_seals = tuple(
            event
            for event in journal_events
            if event.attempt == predecessor.attempt
            and event.event_type == GRAPH_STEP_COMPLETED
            and event.store_seq < handoff_event.store_seq
        )
        try:
            if len(predecessor_seals) != 1:
                raise ValueError(
                    "graph handoff predecessor seal is not unique"
                )
            predecessor_seal = predecessor_seals[0]
            predecessor_artifact = ArtifactRef.from_dict(
                predecessor_seal.payload.get("output_artifact")
            )
            if (
                predecessor_artifact.to_dict()
                != _thaw_json(
                    predecessor_seal.payload.get("output_artifact")
                )
                or predecessor_seal.resource_refs
                != (predecessor_artifact.ref,)
            ):
                raise ValueError(
                    "graph handoff predecessor artifact changed identity"
                )
        except (AttributeError, TypeError, ValueError) as error:
            raise _journal_incompatible(
                "generation rebase graph handoff predecessor is invalid",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                event=handoff_event,
                plan=plan,
                step_index=expected_step.index,
            ) from error
        expected_source_attempt = predecessor.attempt
        predecessor_cursor = EventCursor(
            predecessor_seal.store_seq,
            predecessor_seal.event_id,
        )
        expected_source_range = EventRange(
            predecessor_cursor,
            predecessor_cursor,
        )
        expected_source_artifact_refs = (predecessor_artifact.ref,)
        expected_source_artifact = predecessor_artifact
    if (
        expected_step.source_attempt != expected_source_attempt
        or handoff_envelope.child_attempt != expected_source_attempt
        or handoff_envelope.child_run_id
        != expected_source_attempt.attempt_id
        or handoff_envelope.source_event_range != expected_source_range
        or handoff_envelope.artifact_refs
        != expected_source_artifact_refs
    ):
        raise _journal_incompatible(
            "generation rebase graph handoff source provenance changed",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=handoff_event,
            plan=plan,
            step_index=expected_step.index,
        )
    expected_handoff_refs = tuple(
        dict.fromkeys(
            (
                handoff_envelope.full_output_ref,
                *handoff_envelope.artifact_refs,
            )
        )
    )
    if (
        any(
            resource_ref.kind != "artifact" or resource_ref.fragment
            for resource_ref in expected_handoff_refs
        )
        or handoff_event.resource_refs != expected_handoff_refs
        or handoff_artifact.ref != handoff_envelope.full_output_ref
        or handoff_artifact.media_type != "application/json"
        or handoff_artifact.byte_length != handoff_envelope.byte_length
        or handoff_artifact.sha256 != handoff_envelope.sha256
    ):
        raise _journal_incompatible(
            "generation rebase graph handoff artifact closure changed",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=handoff_event,
            plan=plan,
            step_index=expected_step.index,
        )
    verified_handoff = {
        ref: _verified_graph_resource_artifact(
            ref,
            artifact_repository=artifact_repository,
            event=handoff_event,
            plan=plan,
            step_index=expected_step.index,
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
        )
        for ref in expected_handoff_refs
    }
    input_artifacts = _verified_graph_input_event_artifacts(
        input_event,
        artifact_repository=artifact_repository,
        plan=plan,
        step_index=expected_step.index,
        reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
    )
    shared_provenance = verified_handoff.get(start.resource_refs[0])
    if expected_source_artifact is None:
        source_event = _event_at_graph_cursor(
            journal_events,
            plan.initial_input_cursor,
        )
        expected_source_bytes = _canonical_bytes(
            {
                "schema": "unchain.graph_input_seed.v1",
                "input_event": source_event.to_dict(),
            }
        )
        source_content_matches = (
            handoff_artifact.byte_length == len(expected_source_bytes)
            and handoff_artifact.sha256
            == hashlib.sha256(expected_source_bytes).hexdigest()
        )
    else:
        verified_source = verified_handoff.get(expected_source_artifact.ref)
        source_content_matches = (
            verified_source == expected_source_artifact
            and handoff_artifact.byte_length
            == expected_source_artifact.byte_length
            and handoff_artifact.sha256 == expected_source_artifact.sha256
        )
    if (
        len(input_artifacts) != 2
        or handoff_artifact != input_artifacts[1]
        or handoff_artifact.ref != start.resource_refs[0]
        or shared_provenance is None
        or shared_provenance != handoff_artifact
        or not source_content_matches
    ):
        raise _journal_incompatible(
            "generation rebase graph start artifact binding changed",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        )

    resumes = tuple(
        event
        for event in graph_events
        if event.event_type == "graph.step.resume.admitted"
    )
    for resume in resumes:
        expected_resume_keys = {
            "graph_plan_id",
            "graph_scope_id",
            "step",
            "interaction_id",
            "request_cursor",
            "resolution_cursor",
        }
        try:
            resumed_step = GraphStepBinding.from_dict(resume.payload.get("step"))
            interaction_id = _required_text(
                resume.payload.get("interaction_id"),
                "interaction_id",
                identifier=True,
            )
            request_cursor = _closed_cursor(resume.payload.get("request_cursor"))
            resolution_cursor = _closed_cursor(
                resume.payload.get("resolution_cursor")
            )
            request_event = _event_at_graph_cursor(journal_events, request_cursor)
            resolution_event = _event_at_graph_cursor(
                journal_events,
                resolution_cursor,
            )
            request_interaction_id = _graph_interaction_id(request_event)
            resolution_interaction_id = _graph_interaction_id(resolution_event)
            request_aliases = _graph_interaction_aliases(request_event)
        except (AttributeError, TypeError, ValueError) as error:
            raise _journal_incompatible(
                "generation rebase graph resume admission is invalid",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                event=resume,
                plan=plan,
                step_index=expected_step.index,
            ) from error
        compatible_resolution = resolution_event.event_type in {
            "tool_confirmed",
            "tool_denied",
        }
        if (
            set(resume.payload) != expected_resume_keys
            or resumed_step != expected_step
            or resumed_step.to_dict() != resume.payload.get("step")
            or resume.attempt != expected_step.attempt
            or resume.payload.get("graph_plan_id") != plan.plan_id
            or resume.payload.get("graph_scope_id") != plan.scope_id
            or resume.resource_refs
            or request_event.attempt != expected_step.attempt
            or request_event.event_type
            not in _GRAPH_INTERACTION_REQUEST_EVENT_TYPES
            or resolution_event.attempt != expected_step.attempt
            or resolution_event.event_type
            not in _GRAPH_INTERACTION_RESOLUTION_EVENT_TYPES
            or request_interaction_id != interaction_id
            or (
                resolution_interaction_id != interaction_id
                and not (
                    compatible_resolution
                    and resolution_interaction_id in request_aliases
                )
            )
            or not (
                start.store_seq
                < request_cursor.store_seq
                < resolution_cursor.store_seq
                < resume.store_seq
            )
        ):
            raise _journal_incompatible(
                "generation rebase graph resume admission changed provenance",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                event=resume,
                plan=plan,
                step_index=expected_step.index,
            )

    _validate_graph_step_interaction_cycles(
        events,
        journal_events=journal_events,
        start=start,
        plan=plan,
        expected_step=expected_step,
        artifact_repository=artifact_repository,
    )

    after_start = tuple(
        event for event in events if event.store_seq > start.store_seq
    )
    resource_bearing_terminals = tuple(
        event
        for event in after_start
        if event.event_type in ATTEMPT_TERMINAL_EQUIVALENTS
        and event.resource_refs
    )
    if resource_bearing_terminals:
        raise _journal_incompatible(
            "generation rebase graph runtime terminal carries resources",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=resource_bearing_terminals[0],
            plan=plan,
            step_index=expected_step.index,
        )
    selection = select_attempt_terminal(
        after_start,
        allowed_following_event_types=GRAPH_STEP_SEALS,
    )
    seals = tuple(
        event for event in graph_events if event.event_type in GRAPH_STEP_SEALS
    )
    if selection.ambiguous:
        raise _journal_incompatible(
            "generation rebase graph step has ambiguous canonical terminals",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_TERMINAL_AMBIGUOUS,
            event=start,
            plan=plan,
            step_index=expected_step.index,
        )
    if len(seals) > 1:
        raise _journal_incompatible(
            "generation rebase graph step has duplicate seals",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_DUPLICATED,
            event=seals[1],
            plan=plan,
            step_index=expected_step.index,
        )
    if seals and seals[0].store_seq != events[-1].store_seq:
        raise _journal_incompatible(
            "generation rebase graph step seal is not last",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_NOT_LAST,
            event=seals[0],
            plan=plan,
            step_index=expected_step.index,
        )
    terminal = selection.event
    if terminal is None:
        canonical = tuple(
            event
            for event in after_start
            if event.event_type in CANONICAL_ATTEMPT_TERMINALS
        )
        if canonical or seals:
            anchor = seals[0] if seals else canonical[-1]
            raise _journal_incompatible(
                "generation rebase graph step continued after its terminal",
                reason=(
                    GenerationRebaseFailureReason.GRAPH_STEP_SEAL_NOT_ADJACENT
                    if seals
                    else GenerationRebaseFailureReason.ATTEMPT_CONTINUED_AFTER_TERMINAL
                ),
                event=anchor,
                plan=plan,
                step_index=expected_step.index,
            )
        return _GraphStepQuiescence("blocked", expected_step)
    if not isinstance(terminal, JournalEvent):
        raise TypeError("selected terminal must be a JournalEvent")
    if not seals:
        if terminal.store_seq != events[-1].store_seq:
            raise _journal_incompatible(
                "generation rebase graph step continued after its terminal",
                reason=GenerationRebaseFailureReason.ATTEMPT_CONTINUED_AFTER_TERMINAL,
                event=terminal,
                plan=plan,
                step_index=expected_step.index,
            )
        return _GraphStepQuiescence(
            "recovery_required",
            expected_step,
            terminal=terminal,
        )

    seal = seals[0]
    terminal_position = events.index(terminal)
    if terminal_position + 1 >= len(events) or events[terminal_position + 1] != seal:
        raise _journal_incompatible(
            "generation rebase graph step seal is not terminal-adjacent",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_NOT_ADJACENT,
            event=seal,
            plan=plan,
            step_index=expected_step.index,
        )
    if terminal.event_type not in GRAPH_STEP_SEAL_TERMINALS[seal.event_type]:
        raise _journal_incompatible(
            "generation rebase graph step seal mismatches its terminal",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISMATCHED_TERMINAL,
            event=seal,
            plan=plan,
            step_index=expected_step.index,
        )
    expected_keys = {
        "graph_plan_id",
        "graph_scope_id",
        "step",
        "terminal_cursor",
    }
    if seal.event_type == GRAPH_STEP_COMPLETED:
        expected_keys |= {"output_artifact", "execution_event_range"}
    if set(seal.payload) != expected_keys:
        raise _journal_incompatible(
            "generation rebase graph step seal descriptor is not closed",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=seal,
            plan=plan,
            step_index=expected_step.index,
        )
    try:
        sealed_step = GraphStepBinding.from_dict(seal.payload.get("step"))
        terminal_cursor = _closed_cursor(seal.payload.get("terminal_cursor"))
    except (AttributeError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph step seal descriptor is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=seal,
            plan=plan,
            step_index=expected_step.index,
        ) from error
    if (
        sealed_step != expected_step
        or sealed_step.to_dict() != seal.payload.get("step")
        or seal.attempt != expected_step.attempt
        or seal.payload.get("graph_plan_id") != plan.plan_id
        or seal.payload.get("graph_scope_id") != plan.scope_id
        or terminal_cursor != EventCursor(terminal.store_seq, terminal.event_id)
    ):
        raise _journal_incompatible(
            "generation rebase graph step seal changed identity",
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
            event=seal,
            plan=plan,
            step_index=expected_step.index,
        )
    output_artifact: ArtifactRef | None = None
    if seal.event_type == GRAPH_STEP_COMPLETED:
        try:
            execution_range = EventRange.from_dict(
                seal.payload.get("execution_event_range")
            )
            artifact = ArtifactRef.from_dict(seal.payload.get("output_artifact"))
        except (AttributeError, TypeError, ValueError) as error:
            raise _journal_incompatible(
                "generation rebase completed graph seal is invalid",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEQUENCE_INVALID,
                event=seal,
                plan=plan,
                step_index=expected_step.index,
            ) from error
        if (
            execution_range.to_dict()
            != seal.payload.get("execution_event_range")
            or artifact.to_dict() != seal.payload.get("output_artifact")
            or execution_range.start != EventCursor(start.store_seq, start.event_id)
            or execution_range.end != terminal_cursor
            or seal.resource_refs != (artifact.ref,)
        ):
            raise _journal_incompatible(
                "generation rebase completed graph seal range changed",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEQUENCE_INVALID,
                event=seal,
                plan=plan,
                step_index=expected_step.index,
            )
        _verified_graph_artifact_bytes(
            artifact,
            artifact_repository=artifact_repository,
            event=seal,
            plan=plan,
            step_index=expected_step.index,
            reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
        )
        output_artifact = artifact
        status = "completed"
    else:
        if seal.resource_refs:
            raise _journal_incompatible(
                "generation rebase terminal graph seal carries resources",
                reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_FOREIGN,
                event=seal,
                plan=plan,
                step_index=expected_step.index,
            )
        status = seal.event_type.removeprefix("graph.step.")
    return _GraphStepQuiescence(
        status,
        expected_step,
        terminal=terminal,
        seal=seal,
        output_artifact=output_artifact,
    )


def _validate_graph_execution_seal(
    event: JournalEvent,
    *,
    plan: GraphExecutionPlan,
    final_step: _GraphStepQuiescence | None,
    artifact_repository: _SQLiteBoundContextV2Repository,
) -> None:
    expected_keys = {
        "graph_plan_id",
        "graph_scope_id",
        "status",
        "final_step_index",
        "output_artifact",
        "source_event_range",
    }
    try:
        artifact = ArtifactRef.from_dict(event.payload.get("output_artifact"))
        source_range = EventRange.from_dict(event.payload.get("source_event_range"))
        final_artifact = ArtifactRef.from_dict(
            final_step.seal.payload.get("output_artifact")
            if final_step is not None and final_step.seal is not None
            else None
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise _journal_incompatible(
            "generation rebase graph execution seal is invalid",
            reason=GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED,
            event=event,
            plan=plan,
        ) from error
    if (
        set(event.payload) != expected_keys
        or event.attempt != plan.orchestration_attempt
        or event.payload.get("graph_plan_id") != plan.plan_id
        or event.payload.get("graph_scope_id") != plan.scope_id
        or event.payload.get("status") != "completed"
        or event.payload.get("final_step_index") != len(plan.steps) - 1
        or artifact.to_dict() != event.payload.get("output_artifact")
        or source_range.to_dict() != event.payload.get("source_event_range")
        or final_step is None
        or final_step.status != "completed"
        or final_step.seal is None
        or source_range.start
        != EventCursor(final_step.seal.store_seq, final_step.seal.event_id)
        or source_range.end != source_range.start
        or event.store_seq <= source_range.end.store_seq
        or artifact != final_artifact
        or event.resource_refs != (artifact.ref,)
    ):
        raise _journal_incompatible(
            "generation rebase graph execution seal changed identity",
            reason=GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED,
            event=event,
            plan=plan,
        )
    _verified_graph_artifact_bytes(
        artifact,
        artifact_repository=artifact_repository,
        event=event,
        plan=plan,
        step_index=None,
        reason=GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED,
    )
    if final_step.output_artifact != artifact:
        raise _journal_incompatible(
            "generation rebase graph execution artifact changed from final step",
            reason=GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED,
            event=event,
            plan=plan,
        )


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as error:
        raise ModelValidationError(
            "generation rebase value is not canonical JSON"
        ) from error


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


class GenerationRebaseKind(StrEnum):
    """Host-selected generation transition."""

    CREATE = "create"
    EDIT = "edit"
    REGENERATE = "regenerate"
    RETRY = "retry"


@dataclass(frozen=True)
class GenerationSnapshotMessage:
    """One host-sanitized message imported into the new generation."""

    message_id: str
    role: str
    content: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_id",
            _required_text(self.message_id, "message_id", identifier=True),
        )
        if type(self.role) is not str or self.role not in {"user", "assistant"}:
            raise ModelValidationError(
                "generation snapshot role must be exactly user or assistant"
            )
        if (
            type(self.content) is not str
            or not self.content.strip()
            or "\x00" in self.content
        ):
            raise ModelValidationError(
                "generation snapshot content must be non-empty text"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
        }


@dataclass(frozen=True)
class GenerationRebasePreflight:
    """Host proof for the one fact SQLite cannot derive: snapshot sanitation.

    Interaction and checkpoint clearance are verified from the canonical data
    plane inside the same ``BEGIN IMMEDIATE`` transaction that advances the
    generation head.  They are deliberately not host-supplied booleans.
    """

    proof_id: str
    host_snapshot_sanitized: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proof_id",
            _required_text(self.proof_id, "proof_id", identifier=True),
        )
        object.__setattr__(
            self,
            "host_snapshot_sanitized",
            _exact_bool(
                self.host_snapshot_sanitized,
                "host_snapshot_sanitized",
            ),
        )

    @property
    def permits_rebase(self) -> bool:
        return self.host_snapshot_sanitized

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "host_snapshot_sanitized": self.host_snapshot_sanitized,
        }


@dataclass(frozen=True)
class GenerationTaskStateDescriptor:
    """Content-free descriptor for the task state accompanying a snapshot."""

    descriptor_id: str
    revision: int
    descriptor_sha256: str
    refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "descriptor_id",
            _required_text(self.descriptor_id, "descriptor_id", identifier=True),
        )
        object.__setattr__(
            self,
            "revision",
            _bounded_int(self.revision, "revision", minimum=1),
        )
        object.__setattr__(
            self,
            "descriptor_sha256",
            _sha256(self.descriptor_sha256, "descriptor_sha256"),
        )
        object.__setattr__(
            self,
            "refs",
            _record_tuple(self.refs, ResourceRef, "refs"),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "unchain.legacy_task_state_descriptor.v1",
            "descriptor_id": self.descriptor_id,
            "revision": self.revision,
            "descriptor_sha256": self.descriptor_sha256,
            "refs": [ref.to_dict() for ref in self.refs],
        }

    @classmethod
    def from_record(
        cls,
        value: Mapping[str, Any],
    ) -> GenerationTaskStateDescriptor:
        raw = dict(value)
        if raw.pop("schema", None) != "unchain.legacy_task_state_descriptor.v1":
            raise GenerationRebaseJournalIncompatible(
                "generation task-state descriptor schema changed"
            )
        if set(raw) != {
            "descriptor_id",
            "revision",
            "descriptor_sha256",
            "refs",
        }:
            raise GenerationRebaseJournalIncompatible(
                "generation task-state descriptor shape changed"
            )
        try:
            return cls(
                descriptor_id=raw["descriptor_id"],
                revision=raw["revision"],
                descriptor_sha256=raw["descriptor_sha256"],
                refs=tuple(ResourceRef.from_dict(ref) for ref in raw["refs"]),
            )
        except (TypeError, ValueError) as error:
            raise GenerationRebaseJournalIncompatible(
                "generation task-state descriptor is invalid"
            ) from error


@dataclass(frozen=True)
class GenerationRebaseIntent:
    """Complete host-owned snapshot and exact head transition."""

    owner_chat_id: str
    session_id: str
    execution_id: str
    generation_id: str
    attempt_id: str
    kind: GenerationRebaseKind
    previous_generation_id: str
    expected_head_revision: int
    source_revision: str
    messages: tuple[GenerationSnapshotMessage, ...]
    preflight: GenerationRebasePreflight
    task_state: GenerationTaskStateDescriptor | None = None

    SCHEMA: ClassVar[str] = "unchain.generation_rebase_intent.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "owner_chat_id",
            "session_id",
            "execution_id",
            "generation_id",
            "attempt_id",
            "source_revision",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        try:
            kind = GenerationRebaseKind(self.kind)
        except (TypeError, ValueError) as error:
            raise ModelValidationError("generation rebase kind is invalid") from error
        object.__setattr__(self, "kind", kind)
        revision = _bounded_int(
            self.expected_head_revision,
            "expected_head_revision",
        )
        object.__setattr__(self, "expected_head_revision", revision)
        if kind is GenerationRebaseKind.CREATE:
            if self.previous_generation_id not in (None, ""):
                raise ModelValidationError(
                    "generation create cannot name a previous generation"
                )
            if revision != 0:
                raise ModelValidationError(
                    "generation create requires expected head revision zero"
                )
            object.__setattr__(self, "previous_generation_id", "")
        else:
            previous = _required_text(
                self.previous_generation_id,
                "previous_generation_id",
                identifier=True,
            )
            if previous == self.generation_id:
                raise ModelValidationError(
                    "generation rebase must create a new generation"
                )
            if revision < 1:
                raise ModelValidationError(
                    "generation rebase requires a positive head revision"
                )
            object.__setattr__(self, "previous_generation_id", previous)
        messages = tuple(self.messages)
        if len(messages) > _MAX_MESSAGES:
            raise ModelValidationError(
                "generation snapshot cannot contain more than 10000 messages"
            )
        if any(not isinstance(item, GenerationSnapshotMessage) for item in messages):
            raise TypeError(
                "generation snapshot messages must be GenerationSnapshotMessage records"
            )
        message_ids = [item.message_id for item in messages]
        if len(message_ids) != len(set(message_ids)):
            raise ModelValidationError(
                "generation snapshot message IDs must be unique"
            )
        if sum(len(item.content.encode("utf-8")) for item in messages) > (
            _MAX_SNAPSHOT_BYTES
        ):
            raise ModelValidationError("generation snapshot exceeds the byte limit")
        object.__setattr__(self, "messages", messages)
        if not isinstance(self.preflight, GenerationRebasePreflight):
            raise TypeError("preflight must be a GenerationRebasePreflight")
        if self.task_state is not None and not isinstance(
            self.task_state,
            GenerationTaskStateDescriptor,
        ):
            raise TypeError(
                "task_state must be a GenerationTaskStateDescriptor or None"
            )

    @property
    def attempt(self) -> AttemptRef:
        return AttemptRef(
            GenerationRef(self.execution_id, self.generation_id),
            self.attempt_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "owner_chat_id": self.owner_chat_id,
            "session_id": self.session_id,
            "execution_id": self.execution_id,
            "generation_id": self.generation_id,
            "attempt_id": self.attempt_id,
            "kind": self.kind.value,
            "previous_generation_id": self.previous_generation_id,
            "expected_head_revision": self.expected_head_revision,
            "source_revision": self.source_revision,
            "messages": [item.to_dict() for item in self.messages],
            "preflight": self.preflight.to_dict(),
            "task_state": (
                self.task_state.to_record() if self.task_state is not None else None
            ),
        }


def build_generation_rebase_operation(
    *,
    operation_id: str,
    intent: GenerationRebaseIntent,
) -> OperationRef:
    """Bind one operation ID to the complete rebase snapshot and CAS input."""

    if not isinstance(intent, GenerationRebaseIntent):
        raise TypeError("intent must be a GenerationRebaseIntent")
    return OperationRef(operation_id, _digest(intent.to_dict()))


@dataclass(frozen=True)
class GenerationRebaseRequest:
    intent: GenerationRebaseIntent
    operation: OperationRef

    def __post_init__(self) -> None:
        if not isinstance(self.intent, GenerationRebaseIntent):
            raise TypeError("intent must be a GenerationRebaseIntent")
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(
                self,
                "operation",
                OperationRef.from_dict(self.operation),
            )


@dataclass(frozen=True)
class GenerationRebaseHead:
    owner_chat_id: str
    session_id: str
    execution_id: str
    current_generation_id: str
    current_attempt_id: str
    current_source_revision: str
    revision: int


@dataclass(frozen=True)
class GenerationRebaseReceipt:
    owner_chat_id: str
    session_id: str
    execution_id: str
    generation_id: str
    attempt_id: str
    kind: GenerationRebaseKind
    previous_generation_id: str
    source_revision: str
    head_revision: int
    manifest_sha256: str
    message_count: int
    first_cursor: EventCursor
    last_cursor: EventCursor
    operation: OperationRef
    lifecycle_operation: OperationRef
    attempt_binding_operation: OperationRef
    task_state: GenerationTaskStateDescriptor | None = None
    duplicate: bool = False


def _derived_operation_id(prefix: str, operation: OperationRef, subject: str) -> str:
    digest = hashlib.sha256(
        "\0".join(
            (operation.operation_id, operation.payload_sha256, subject)
        ).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest}"


def _checkpoint_identity(execution_id: str, operation: OperationRef) -> str:
    return hashlib.sha256(
        (
            "unchain.context_checkpoint.v1\0"
            + execution_id
            + "\0"
            + operation.operation_id
            + "\0"
            + operation.payload_sha256
        ).encode("utf-8")
    ).hexdigest()


def _host_transition_payload(intent: GenerationRebaseIntent) -> dict[str, Any]:
    compatible_kind = {
        GenerationRebaseKind.CREATE: "initial",
        GenerationRebaseKind.EDIT: "edit",
        GenerationRebaseKind.REGENERATE: "regenerate",
        GenerationRebaseKind.RETRY: "regenerate",
    }[intent.kind]
    return {
        "schema": "unchain.host_generation_transition.v1",
        "owner_chat_id": intent.owner_chat_id,
        "execution_id": intent.execution_id,
        "session_id": intent.session_id,
        "generation_id": intent.generation_id,
        "kind": compatible_kind,
        "previous_generation_id": intent.previous_generation_id,
        "expected_revision": intent.expected_head_revision,
    }


def _host_attempt_payload(
    intent: GenerationRebaseIntent,
    *,
    head_revision: int,
) -> dict[str, Any]:
    return {
        "schema": "unchain.host_generation_attempt_binding_intent.v1",
        "owner_chat_id": intent.owner_chat_id,
        "execution_id": intent.execution_id,
        "session_id": intent.session_id,
        "generation_id": intent.generation_id,
        "attempt_id": intent.attempt_id,
        "expected_revision": head_revision,
    }


class SQLiteGenerationRebaseV2Service:
    """Own the all-or-nothing generation rebase transaction."""

    _SCHEMA_VERSION: ClassVar[int] = 1

    def __init__(self, store: SQLiteContextV2Store) -> None:
        if type(store) is not SQLiteContextV2Store:
            raise TypeError("store must be the official SQLiteContextV2Store")
        self._store = store
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._store.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _transaction(self, *, immediate: bool) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).casefold() != "wal":
                raise GenerationRebaseUnavailable(
                    "generation rebase SQLite WAL mode is unavailable",
                    reason=(
                        GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                    ),
                )
            connection.executescript(
                """
                BEGIN IMMEDIATE;

                CREATE TABLE IF NOT EXISTS generation_rebase_v2_schema (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO generation_rebase_v2_schema(version)
                VALUES (1);

                CREATE TABLE IF NOT EXISTS host_generation_schema (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO host_generation_schema(version) VALUES (1);

                CREATE TABLE IF NOT EXISTS host_generation_chat_bindings (
                    owner_chat_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    UNIQUE(owner_chat_id, execution_id, session_id),
                    FOREIGN KEY (execution_id) REFERENCES executions(execution_id)
                );

                CREATE TABLE IF NOT EXISTS host_generation_records (
                    owner_chat_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    transition_kind TEXT NOT NULL CHECK(
                        transition_kind IN ('initial', 'edit', 'regenerate')
                    ),
                    previous_generation_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    operation_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (owner_chat_id, generation_id),
                    UNIQUE(execution_id, generation_id),
                    UNIQUE(owner_chat_id, revision),
                    FOREIGN KEY (owner_chat_id, execution_id, session_id)
                        REFERENCES host_generation_chat_bindings(
                            owner_chat_id, execution_id, session_id
                        )
                );

                CREATE TABLE IF NOT EXISTS host_generation_heads (
                    owner_chat_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    current_generation_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    FOREIGN KEY (owner_chat_id, execution_id, session_id)
                        REFERENCES host_generation_chat_bindings(
                            owner_chat_id, execution_id, session_id
                        ),
                    FOREIGN KEY (owner_chat_id, current_generation_id)
                        REFERENCES host_generation_records(
                            owner_chat_id, generation_id
                        )
                );

                CREATE TABLE IF NOT EXISTS host_generation_operations (
                    owner_chat_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    mutation_kind TEXT NOT NULL,
                    result_generation_id TEXT NOT NULL,
                    result_revision INTEGER NOT NULL CHECK(result_revision >= 1),
                    result_attempt_id TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (owner_chat_id, operation_id),
                    FOREIGN KEY (owner_chat_id, result_generation_id)
                        REFERENCES host_generation_records(
                            owner_chat_id, generation_id
                        )
                );

                CREATE TABLE IF NOT EXISTS host_generation_attempt_bindings (
                    owner_chat_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    head_revision INTEGER NOT NULL CHECK(head_revision >= 1),
                    operation_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (execution_id, attempt_id),
                    UNIQUE(owner_chat_id, operation_id),
                    FOREIGN KEY (owner_chat_id, execution_id, session_id)
                        REFERENCES host_generation_chat_bindings(
                            owner_chat_id, execution_id, session_id
                        ),
                    FOREIGN KEY (owner_chat_id, generation_id)
                        REFERENCES host_generation_records(
                            owner_chat_id, generation_id
                        ),
                    FOREIGN KEY (owner_chat_id, operation_id)
                        REFERENCES host_generation_operations(
                            owner_chat_id, operation_id
                        )
                );

                CREATE TABLE IF NOT EXISTS legacy_bootstrap_schema (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO legacy_bootstrap_schema(version) VALUES (1);

                CREATE TABLE IF NOT EXISTS legacy_bootstrap_manifests (
                    owner_chat_id TEXT NOT NULL,
                    source_revision TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    generation_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    manifest_json BLOB NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    first_store_seq INTEGER NOT NULL CHECK(first_store_seq >= 1),
                    last_store_seq INTEGER NOT NULL CHECK(
                        last_store_seq >= first_store_seq
                    ),
                    event_count INTEGER NOT NULL CHECK(event_count >= 1),
                    PRIMARY KEY (owner_chat_id, generation_id),
                    UNIQUE (owner_chat_id, source_revision),
                    UNIQUE (execution_id, generation_id),
                    FOREIGN KEY (execution_id) REFERENCES executions(execution_id),
                    FOREIGN KEY (execution_id, operation_id)
                        REFERENCES operations(execution_id, operation_id)
                );

                CREATE TABLE IF NOT EXISTS legacy_bootstrap_chat_heads (
                    owner_chat_id TEXT PRIMARY KEY,
                    execution_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    current_generation_id TEXT NOT NULL,
                    current_source_revision TEXT NOT NULL,
                    head_revision INTEGER NOT NULL CHECK(head_revision >= 1),
                    FOREIGN KEY (owner_chat_id, current_generation_id)
                        REFERENCES legacy_bootstrap_manifests(
                            owner_chat_id, generation_id
                        )
                );

                COMMIT;
                """
            )
            for table in (
                "generation_rebase_v2_schema",
                "host_generation_schema",
                "legacy_bootstrap_schema",
            ):
                versions = {
                    int(row[0])
                    for row in connection.execute(f"SELECT version FROM {table}")
                }
                if versions != {self._SCHEMA_VERSION}:
                    raise GenerationRebaseUnavailable(
                        "generation rebase SQLite schema is unsupported",
                        reason=(
                            GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                        ),
                    )
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise GenerationRebaseUnavailable(
                    "generation rebase SQLite quick_check failed",
                    reason=(
                        GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                    ),
                )
        except GenerationRebaseError:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise
        except sqlite3.Error as error:
            try:
                connection.rollback()
            except sqlite3.Error:
                pass
            raise GenerationRebaseUnavailable(
                "generation rebase SQLite schema initialization failed",
                reason=(
                    GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                ),
            ) from error
        finally:
            connection.close()

    @staticmethod
    def _target_key(intent: GenerationRebaseIntent) -> str:
        return f"{intent.owner_chat_id}:{intent.generation_id}"

    @staticmethod
    def _message_draft(
        intent: GenerationRebaseIntent,
        message: GenerationSnapshotMessage,
        index: int,
    ) -> SemanticEventDraft:
        identity = {
            "owner_chat_id": intent.owner_chat_id,
            "source_revision": intent.source_revision,
            "execution_id": intent.execution_id,
            "generation_id": intent.generation_id,
            "attempt_id": intent.attempt_id,
            "message_id": message.message_id,
            "message_index": index,
            "role": message.role,
            "kind": intent.kind.value,
        }
        identity_sha256 = _digest(identity)
        return SemanticEventDraft(
            event_id=f"generation-rebase-import-{identity_sha256}",
            event_type=f"message.{message.role}",
            attempt=intent.attempt,
            operation_id=f"generation-rebase-event-{identity_sha256}",
            payload={
                "run_id": intent.attempt_id,
                "message": {
                    "role": message.role,
                    "content": message.content,
                },
                "legacy_provenance": {
                    "source": "host_sanitized_generation_snapshot",
                    "capture_status": _LEGACY_CAPTURE_STATUS,
                    "owner_chat_id": intent.owner_chat_id,
                    "session_id": intent.session_id,
                    "source_revision": intent.source_revision,
                    "message_id": message.message_id,
                    "message_index": index,
                },
                "generation_rebase": {
                    "kind": intent.kind.value,
                    "previous_generation_id": intent.previous_generation_id,
                    "expected_head_revision": intent.expected_head_revision,
                },
            },
        )

    @staticmethod
    def _marker_draft(intent: GenerationRebaseIntent) -> SemanticEventDraft:
        """Persist an empty visible snapshot without inventing a chat message."""

        identity = {
            "owner_chat_id": intent.owner_chat_id,
            "session_id": intent.session_id,
            "source_revision": intent.source_revision,
            "execution_id": intent.execution_id,
            "generation_id": intent.generation_id,
            "attempt_id": intent.attempt_id,
            "kind": intent.kind.value,
            "previous_generation_id": intent.previous_generation_id,
            "expected_head_revision": intent.expected_head_revision,
            "empty_snapshot": True,
        }
        identity_sha256 = _digest(identity)
        return SemanticEventDraft(
            event_id=f"generation-rebase-marker-{identity_sha256}",
            event_type="generation.rebased",
            attempt=intent.attempt,
            operation_id=f"generation-rebase-marker-operation-{identity_sha256}",
            payload={
                "run_id": intent.attempt_id,
                "generation_rebase": {
                    "kind": intent.kind.value,
                    "previous_generation_id": intent.previous_generation_id,
                    "expected_head_revision": intent.expected_head_revision,
                    "empty_snapshot": True,
                    "replacement_message_count": 0,
                },
                "legacy_provenance": {
                    "source": "host_sanitized_generation_snapshot",
                    "capture_status": _LEGACY_CAPTURE_STATUS,
                    "owner_chat_id": intent.owner_chat_id,
                    "session_id": intent.session_id,
                    "source_revision": intent.source_revision,
                },
            },
        )

    @staticmethod
    def _operation_row(
        connection: sqlite3.Connection,
        *,
        execution_id: str,
        operation_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT payload_sha256, target_kind, target_key
            FROM operations
            WHERE execution_id = ? AND operation_id = ?
            """,
            (execution_id, operation_id),
        ).fetchone()

    @classmethod
    def _claim_operation(
        cls,
        connection: sqlite3.Connection,
        *,
        execution_id: str,
        operation: OperationRef,
        target_kind: str,
        target_key: str,
    ) -> bool:
        row = cls._operation_row(
            connection,
            execution_id=execution_id,
            operation_id=operation.operation_id,
        )
        if row is not None:
            if (
                row["payload_sha256"] == operation.payload_sha256
                and row["target_kind"] == target_kind
                and row["target_key"] == target_key
            ):
                return False
            raise GenerationRebaseConflict(
                "generation rebase operation payload or target changed",
                reason=(
                    GenerationRebaseFailureReason.OPERATION_IDENTITY_CONFLICT
                ),
            )
        connection.execute(
            """
            INSERT INTO operations(
                execution_id, operation_id, payload_sha256,
                target_kind, target_key
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                execution_id,
                operation.operation_id,
                operation.payload_sha256,
                target_kind,
                target_key,
            ),
        )
        return True

    @staticmethod
    def _compatible_kind(kind: GenerationRebaseKind) -> str:
        return {
            GenerationRebaseKind.CREATE: "initial",
            GenerationRebaseKind.EDIT: "edit",
            GenerationRebaseKind.REGENERATE: "regenerate",
            GenerationRebaseKind.RETRY: "regenerate",
        }[kind]

    @staticmethod
    def _manifest_kind(kind: GenerationRebaseKind) -> str:
        return "initial" if kind is GenerationRebaseKind.CREATE else kind.value

    @staticmethod
    def _kind_from_manifest(value: object) -> GenerationRebaseKind:
        if value == "initial":
            return GenerationRebaseKind.CREATE
        try:
            return GenerationRebaseKind(value)
        except (TypeError, ValueError) as error:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest kind is invalid"
            ) from error

    @staticmethod
    def _head_rows(
        connection: sqlite3.Connection,
        owner_chat_id: str,
    ) -> tuple[sqlite3.Row | None, sqlite3.Row | None]:
        host = connection.execute(
            "SELECT * FROM host_generation_heads WHERE owner_chat_id = ?",
            (owner_chat_id,),
        ).fetchone()
        bootstrap = connection.execute(
            "SELECT * FROM legacy_bootstrap_chat_heads WHERE owner_chat_id = ?",
            (owner_chat_id,),
        ).fetchone()
        return host, bootstrap

    @staticmethod
    def _assert_matching_heads(
        host: sqlite3.Row | None,
        bootstrap: sqlite3.Row | None,
    ) -> None:
        if (host is None) != (bootstrap is None):
            raise GenerationRebaseJournalIncompatible(
                "generation lifecycle and bootstrap heads diverged"
            )
        if host is None:
            return
        if (
            host["owner_chat_id"] != bootstrap["owner_chat_id"]
            or host["execution_id"] != bootstrap["execution_id"]
            or host["session_id"] != bootstrap["session_id"]
            or host["current_generation_id"]
            != bootstrap["current_generation_id"]
            or int(host["revision"]) != int(bootstrap["head_revision"])
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation lifecycle and bootstrap heads changed independently"
            )

    def _verified_event_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> JournalEvent:
        raw = bytes(row["event_json"])
        if hashlib.sha256(raw).hexdigest() != row["event_sha256"]:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase journal event digest changed"
            )
        try:
            event = JournalEvent.from_dict(json.loads(raw.decode("utf-8")))
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase journal event is unreadable"
            ) from error
        operation = self._operation_row(
            connection,
            execution_id=row["execution_id"],
            operation_id=event.operation.operation_id,
        )
        if (
            _canonical_bytes(event.to_dict()) != raw
            or event.attempt.generation.execution_id != row["execution_id"]
            or event.store_seq != row["store_seq"]
            or event.event_id != row["event_id"]
            or event.attempt.generation.generation_id != row["generation_id"]
            or event.attempt.attempt_id != row["attempt_id"]
            or event.event_type != row["event_type"]
            or event.operation.operation_id != row["operation_id"]
            or operation is None
            or operation["payload_sha256"] != event.operation.payload_sha256
            or operation["target_kind"] != "journal_event"
            or operation["target_key"] != event.event_id
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase journal event authority changed"
            )
        return event

    @staticmethod
    def _interaction_id(event: JournalEvent) -> str:
        direct = event.payload.get("interaction_id")
        request = event.payload.get("interaction_request")
        nested = request.get("interaction_id") if isinstance(request, Mapping) else None
        values = {
            str(value).strip()
            for value in (direct, nested)
            if isinstance(value, str) and value.strip()
        }
        if len(values) != 1:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase interaction identity is ambiguous"
            )
        interaction_id = next(iter(values))
        try:
            return _required_text(
                interaction_id,
                "interaction_id",
                identifier=True,
            )
        except (TypeError, ValueError) as error:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase interaction identity is invalid"
            ) from error

    def _assert_no_pending_interaction(
        self,
        connection: sqlite3.Connection,
        intent: GenerationRebaseIntent,
    ) -> None:
        rows = list(
            connection.execute(
                """
                SELECT * FROM events
                WHERE execution_id = ? AND generation_id = ?
                ORDER BY store_seq
                """,
                (intent.execution_id, intent.previous_generation_id),
            )
        )
        events = tuple(
            self._verified_event_from_row(connection, row) for row in rows
        )
        try:
            suppressed_legacy_resolutions = (
                legacy_interaction_resolution_supersessions(
                    tuple(
                        interaction_resolution_compatibility_record(
                            ordinal=event.store_seq,
                            event_type=event.event_type,
                            interaction_id=self._interaction_id(event),
                            execution_id=event.attempt.generation.execution_id,
                            generation_id=event.attempt.generation.generation_id,
                            attempt_id=event.attempt.attempt_id,
                            payload=event.payload,
                            resource_refs=event.resource_refs,
                        )
                        for event in events
                        if event.event_type
                        in _INTERACTION_RESOLUTION_EVENT_TYPES
                    )
                )
            )
        except InteractionResolutionCompatibilityError as error:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase interaction resolution is duplicated",
                reason=(
                    GenerationRebaseFailureReason.INTERACTION_RESOLUTION_DUPLICATED
                ),
                subject={
                    "execution_id": intent.execution_id,
                    "generation_id": intent.previous_generation_id,
                },
            ) from error
        requests: dict[str, JournalEvent] = {}
        resolutions: dict[str, JournalEvent] = {}
        terminal_store_seqs: dict[str, int] = {}
        for event in events:
            if event.event_type in _INTERACTION_REQUEST_EVENT_TYPES:
                interaction_id = self._interaction_id(event)
                if interaction_id in requests:
                    raise GenerationRebaseJournalIncompatible(
                        "generation rebase interaction request is duplicated",
                        reason=(
                            GenerationRebaseFailureReason.INTERACTION_REQUEST_DUPLICATED
                        ),
                        subject={
                            "execution_id": intent.execution_id,
                            "generation_id": intent.previous_generation_id,
                            "attempt_id": event.attempt.attempt_id,
                            "interaction_id": interaction_id,
                        },
                    )
                requests[interaction_id] = event
            elif event.event_type in _INTERACTION_RESOLUTION_EVENT_TYPES:
                if event.store_seq in suppressed_legacy_resolutions:
                    continue
                interaction_id = self._interaction_id(event)
                if interaction_id in resolutions:
                    raise GenerationRebaseJournalIncompatible(
                        "generation rebase interaction resolution is duplicated",
                        reason=(
                            GenerationRebaseFailureReason.INTERACTION_RESOLUTION_DUPLICATED
                        ),
                        subject={
                            "execution_id": intent.execution_id,
                            "generation_id": intent.previous_generation_id,
                            "attempt_id": event.attempt.attempt_id,
                            "interaction_id": interaction_id,
                        },
                    )
                resolutions[interaction_id] = event
            elif event.event_type in _ATTEMPT_TERMINAL_EVENT_TYPES:
                previous = terminal_store_seqs.get(event.attempt.attempt_id, 0)
                terminal_store_seqs[event.attempt.attempt_id] = max(
                    previous,
                    event.store_seq,
                )

        for interaction_id, resolution in resolutions.items():
            request = requests.get(interaction_id)
            if request is None or resolution.store_seq <= request.store_seq:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase interaction lifecycle is not uniquely paired",
                    reason=(
                        GenerationRebaseFailureReason.INTERACTION_LIFECYCLE_NOT_PAIRED
                    ),
                    subject={
                        "execution_id": intent.execution_id,
                        "generation_id": intent.previous_generation_id,
                        "attempt_id": resolution.attempt.attempt_id,
                        "interaction_id": interaction_id,
                    },
                )

        pending = tuple(
            interaction_id
            for interaction_id, request in requests.items()
            if interaction_id not in resolutions
            and terminal_store_seqs.get(request.attempt.attempt_id, 0)
            <= request.store_seq
        )
        if pending:
            interaction_id = pending[0]
            request = requests[interaction_id]
            raise GenerationRebasePreflightBlocked(
                "generation rebase found a pending durable interaction",
                reason=GenerationRebaseFailureReason.PENDING_INTERACTION,
                subject={
                    "execution_id": intent.execution_id,
                    "generation_id": intent.previous_generation_id,
                    "attempt_id": request.attempt.attempt_id,
                    "interaction_id": interaction_id,
                },
            )

    @staticmethod
    def _checkpoint_tables(
        connection: sqlite3.Connection,
    ) -> tuple[bool, bool]:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        return (
            "context_compiler_v2_schema" in tables,
            "checkpoints" in tables,
        )

    def _assert_no_prepared_checkpoint(
        self,
        connection: sqlite3.Connection,
        intent: GenerationRebaseIntent,
    ) -> None:
        has_schema, has_checkpoints = self._checkpoint_tables(connection)
        if has_schema != has_checkpoints:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase compiler checkpoint schema is incomplete"
            )
        if not has_checkpoints:
            orphan = connection.execute(
                """
                SELECT 1 FROM operations
                WHERE execution_id = ? AND target_kind = 'checkpoint'
                LIMIT 1
                """,
                (intent.execution_id,),
            ).fetchone()
            if orphan is not None:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase found checkpoint operations without a store"
                )
            return
        versions = {
            int(row[0])
            for row in connection.execute(
                "SELECT version FROM context_compiler_v2_schema"
            )
        }
        if versions != {1}:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase compiler checkpoint schema is unsupported"
            )

        rows = list(
            connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE execution_id = ? AND status = 'prepared'
                ORDER BY source_start_seq, source_end_seq, checkpoint_id
                """,
                (intent.execution_id,),
            )
        )
        for row in rows:
            try:
                semantic_raw = bytes(row["semantic_json"])
                artifact_raw = bytes(row["artifact_json"])
            except (TypeError, ValueError) as error:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase checkpoint record bytes are invalid"
                ) from error
            if (
                hashlib.sha256(semantic_raw).hexdigest()
                != row["semantic_sha256"]
                or hashlib.sha256(artifact_raw).hexdigest()
                != row["artifact_sha256"]
            ):
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase checkpoint record digest changed"
                )
            try:
                semantic = json.loads(semantic_raw.decode("utf-8"))
                artifact_record = json.loads(artifact_raw.decode("utf-8"))
                source_range = EventRange.from_dict(semantic["source_range"])
                refs = tuple(
                    ResourceRef.from_dict(value) for value in semantic["refs"]
                )
                artifact = ArtifactRef.from_dict(artifact_record)
                operation = OperationRef(
                    row["operation_id"],
                    row["operation_payload_sha256"],
                )
                summary_sha256 = _sha256(
                    semantic["summary_sha256"],
                    "summary_sha256",
                )
            except (
                KeyError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase checkpoint record is unreadable"
                ) from error
            expected_semantic = {
                "schema": "unchain.sqlite_checkpoint_prepare.v1",
                "source_range": source_range.to_dict(),
                "summary_sha256": summary_sha256,
                "refs": [ref.to_dict() for ref in refs],
            }
            operation_row = self._operation_row(
                connection,
                execution_id=intent.execution_id,
                operation_id=operation.operation_id,
            )
            checkpoint_id = _checkpoint_identity(intent.execution_id, operation)
            if (
                semantic != expected_semantic
                or _canonical_bytes(semantic) != semantic_raw
                or _canonical_bytes(artifact.to_dict()) != artifact_raw
                or row["checkpoint_id"] != checkpoint_id
                or row["preparation_id"] != "preparation-" + checkpoint_id
                or int(row["revision"]) != 1
                or row["source_start_seq"] != source_range.start.store_seq
                or row["source_start_event_id"] != source_range.start.event_id
                or row["source_end_seq"] != source_range.end.store_seq
                or row["source_end_event_id"] != source_range.end.event_id
                or artifact.ref.kind != "artifact"
                or artifact.ref.fragment
                or operation_row is None
                or operation_row["payload_sha256"] != operation.payload_sha256
                or operation_row["target_kind"] != "checkpoint"
                or operation_row["target_key"] != checkpoint_id
            ):
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase checkpoint authority changed"
                )
            endpoint_rows = list(
                connection.execute(
                    """
                    SELECT * FROM events
                    WHERE execution_id = ? AND store_seq IN (?, ?)
                    ORDER BY store_seq
                    """,
                    (
                        intent.execution_id,
                        source_range.start.store_seq,
                        source_range.end.store_seq,
                    ),
                )
            )
            expected_endpoint_count = (
                1
                if source_range.start.store_seq == source_range.end.store_seq
                else 2
            )
            if len(endpoint_rows) != expected_endpoint_count:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase checkpoint endpoints are unavailable"
                )
            endpoints = tuple(
                self._verified_event_from_row(connection, endpoint)
                for endpoint in endpoint_rows
            )
            if (
                endpoints[0].event_id != source_range.start.event_id
                or endpoints[-1].event_id != source_range.end.event_id
            ):
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase checkpoint endpoint identity changed"
                )
            overlaps_current = connection.execute(
                """
                SELECT 1 FROM events
                WHERE execution_id = ? AND generation_id = ?
                  AND store_seq BETWEEN ? AND ?
                LIMIT 1
                """,
                (
                    intent.execution_id,
                    intent.previous_generation_id,
                    source_range.start.store_seq,
                    source_range.end.store_seq,
                ),
            ).fetchone()
            if overlaps_current is not None:
                raise GenerationRebasePreflightBlocked(
                    "generation rebase found a prepared durable checkpoint",
                    reason=GenerationRebaseFailureReason.CHECKPOINT_PREPARED,
                    subject={
                        "execution_id": intent.execution_id,
                        "generation_id": intent.previous_generation_id,
                    },
                )

    def _assert_durable_preflight(
        self,
        connection: sqlite3.Connection,
        intent: GenerationRebaseIntent,
        current_receipt: GenerationRebaseReceipt | None,
    ) -> None:
        if intent.kind is GenerationRebaseKind.CREATE:
            return
        if current_receipt is None:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase current receipt is unavailable",
                reason=GenerationRebaseFailureReason.CURRENT_RECEIPT_UNAVAILABLE,
                subject={
                    "execution_id": intent.execution_id,
                    "generation_id": intent.previous_generation_id,
                },
            )
        self._assert_no_prepared_checkpoint(connection, intent)
        self._assert_no_pending_interaction(connection, intent)
        self._assert_no_open_attempt_or_tool(
            connection,
            intent,
            current_receipt,
        )

    def _assert_no_open_attempt_or_tool(
        self,
        connection: sqlite3.Connection,
        intent: GenerationRebaseIntent,
        current_receipt: GenerationRebaseReceipt,
    ) -> None:
        """Reject a cutover while current-generation work is still live.

        The receipt's manifest range is the atomic import attempt and is
        already verified by ``_receipt_for_generation``.  Only events outside
        that sealed range represent runtime work, even when a host reuses the
        import attempt ID.
        """

        rows = list(
            connection.execute(
                """
                SELECT * FROM events
                WHERE execution_id = ? AND generation_id = ?
                ORDER BY store_seq
                """,
                (intent.execution_id, intent.previous_generation_id),
            )
        )
        events = tuple(
            self._verified_event_from_row(connection, row) for row in rows
        )
        import_start = current_receipt.first_cursor.store_seq
        import_end = current_receipt.last_cursor.store_seq
        runtime_events = tuple(
            event
            for event in events
            if not import_start <= event.store_seq <= import_end
        )
        if not runtime_events:
            return
        artifact_repository = _SQLiteBoundContextV2Repository(
            self._store,
            intent.execution_id,
        )

        blocked_failures: list[
            tuple[str, GenerationRebaseFailureReason, Mapping[str, Any]]
        ] = []
        tool_groups: dict[tuple[str, str], list[JournalEvent]] = {}
        for event in runtime_events:
            if event.event_type not in _TOOL_LIFECYCLE_EVENT_TYPES:
                continue
            call_id = event.payload.get("call_id")
            if (
                not isinstance(call_id, str)
                or not call_id
                or call_id != call_id.strip()
            ):
                raise _journal_incompatible(
                    "generation rebase tool lifecycle has no stable call identity",
                    reason=(
                        GenerationRebaseFailureReason.TOOL_CALL_IDENTITY_UNSTABLE
                    ),
                    event=event,
                )
            tool_groups.setdefault(
                (event.attempt.attempt_id, call_id),
                [],
            ).append(event)

        for lifecycle in tool_groups.values():
            starts = [
                event
                for event in lifecycle
                if event.event_type in _TOOL_STARTED_EVENT_TYPES
            ]
            if not starts:
                continue
            intents = [
                event
                for event in lifecycle
                if event.event_type in _TOOL_INTENT_EVENT_TYPES
            ]
            seals = [
                event
                for event in lifecycle
                if event.event_type in _TOOL_SEALED_EVENT_TYPES
            ]
            results = [
                event
                for event in lifecycle
                if event.event_type in _TOOL_RESULT_EVENT_TYPES
            ]
            if (
                len(intents) != 1
                or len(starts) != 1
                or len(seals) > 1
                or len(results) > 1
            ):
                raise _journal_incompatible(
                    "generation rebase tool lifecycle is not uniquely paired",
                    reason=GenerationRebaseFailureReason.TOOL_LIFECYCLE_NOT_PAIRED,
                    event=lifecycle[-1],
                )
            ordered = intents[0], starts[0]
            if ordered[1].store_seq <= ordered[0].store_seq:
                raise _journal_incompatible(
                    "generation rebase tool start precedes its durable intent",
                    reason=GenerationRebaseFailureReason.TOOL_START_PRECEDES_INTENT,
                    event=starts[0],
                )
            if seals and seals[0].store_seq <= starts[0].store_seq:
                raise _journal_incompatible(
                    "generation rebase sealed tool completion precedes its start",
                    reason=GenerationRebaseFailureReason.TOOL_SEAL_PRECEDES_START,
                    event=seals[0],
                )
            if results and results[0].store_seq <= starts[0].store_seq:
                raise _journal_incompatible(
                    "generation rebase tool result precedes its start",
                    reason=GenerationRebaseFailureReason.TOOL_RESULT_PRECEDES_START,
                    event=results[0],
                )
            if seals and results and results[0].store_seq <= seals[0].store_seq:
                raise _journal_incompatible(
                    "generation rebase tool result precedes sealed completion",
                    reason=GenerationRebaseFailureReason.TOOL_RESULT_PRECEDES_SEAL,
                    event=results[0],
                )
            tool_names = {
                event.payload.get("tool_name") for event in lifecycle
            }
            tool_name = next(iter(tool_names)) if len(tool_names) == 1 else None
            if (
                not isinstance(tool_name, str)
                or not tool_name
                or tool_name != tool_name.strip()
            ):
                raise _journal_incompatible(
                    "generation rebase tool lifecycle identity changed",
                    reason=GenerationRebaseFailureReason.TOOL_IDENTITY_CHANGED,
                    event=lifecycle[-1],
                )
            if not results:
                blocked_failures.append(
                    (
                        "generation rebase found an unfinished durable tool",
                        GenerationRebaseFailureReason.TOOL_OPEN,
                        {
                        "execution_id": lifecycle[-1].attempt.generation.execution_id,
                        "generation_id": lifecycle[-1].attempt.generation.generation_id,
                        "attempt_id": lifecycle[-1].attempt.attempt_id,
                        "call_id": str(lifecycle[-1].payload.get("call_id")),
                        },
                    )
                )

        attempts: dict[str, list[JournalEvent]] = {}
        for event in runtime_events:
            attempts.setdefault(event.attempt.attempt_id, []).append(event)
        ordered_attempts = {
            attempt_id: tuple(sorted(values, key=lambda event: event.store_seq))
            for attempt_id, values in attempts.items()
        }

        plans_by_identity: dict[
            tuple[str, str],
            GraphExecutionPlan,
        ] = {}
        orchestration_groups: dict[str, GraphExecutionPlan] = {}
        step_owners: dict[str, tuple[GraphExecutionPlan, GraphStepBinding]] = {}
        for attempt_id, attempt_events in ordered_attempts.items():
            admissions = tuple(
                event
                for event in attempt_events
                if event.event_type == "graph.execution.admitted"
            )
            if not admissions:
                continue
            if len(admissions) != 1 or any(
                event.event_type == "graph.step.started"
                for event in attempt_events
            ):
                raise _journal_incompatible(
                    "generation rebase graph attempt kind is ambiguous",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_ATTEMPT_KIND_AMBIGUOUS
                    ),
                    event=admissions[-1],
                )
            plan = _parse_graph_plan_admission(admissions[0])
            if (
                plan.execution_id != intent.execution_id
                or plan.orchestration_attempt.generation.generation_id
                != intent.previous_generation_id
            ):
                raise _journal_incompatible(
                    "generation rebase graph plan escaped the current generation",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                    ),
                    event=admissions[0],
                    plan=plan,
                )
            try:
                initial_input = _event_at_graph_cursor(
                    runtime_events,
                    plan.initial_input_cursor,
                )
            except ValueError as error:
                raise _journal_incompatible(
                    "generation rebase graph initial input is unavailable",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                    ),
                    event=admissions[0],
                    plan=plan,
                ) from error
            if (
                initial_input.attempt != plan.orchestration_attempt
                or initial_input.event_type
                not in {"message.user", "interaction.resolved"}
                or initial_input.store_seq >= admissions[0].store_seq
            ):
                raise _journal_incompatible(
                    "generation rebase graph initial input changed provenance",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                    ),
                    event=admissions[0],
                    plan=plan,
                )
            _verified_graph_input_event_artifacts(
                initial_input,
                artifact_repository=artifact_repository,
                plan=plan,
                step_index=None,
                reason=(
                    GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                ),
            )
            identity = (plan.plan_id, plan.scope_id)
            if identity in plans_by_identity or attempt_id in orchestration_groups:
                raise _journal_incompatible(
                    "generation rebase graph plan admission is ambiguous",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                    ),
                    event=admissions[0],
                    plan=plan,
                )
            plans_by_identity[identity] = plan
            orchestration_groups[attempt_id] = plan
            for step in plan.steps:
                if step.attempt.attempt_id in step_owners:
                    raise _journal_incompatible(
                        "generation rebase graph step belongs to multiple plans",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                        ),
                        event=admissions[0],
                        plan=plan,
                        step_index=step.index,
                    )
                step_owners[step.attempt.attempt_id] = (plan, step)

        step_states: dict[tuple[str, int], _GraphStepQuiescence] = {}
        recovery_failures: list[
            tuple[
                GenerationRebaseFailureReason,
                JournalEvent,
                GraphExecutionPlan,
                int | None,
            ]
        ] = []
        for attempt_id, attempt_events in ordered_attempts.items():
            graph_events = tuple(
                event
                for event in attempt_events
                if event.event_type.startswith("graph.")
            )
            if attempt_id in orchestration_groups:
                allowed = {
                    "graph.execution.admitted",
                    "graph.execution.completed",
                }
                if any(event.event_type not in allowed for event in graph_events):
                    raise _journal_incompatible(
                        "generation rebase graph orchestration kind is ambiguous",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_ATTEMPT_KIND_AMBIGUOUS
                        ),
                        event=graph_events[-1],
                        plan=orchestration_groups[attempt_id],
                    )
                continue

            owner = step_owners.get(attempt_id)
            starts = tuple(
                event
                for event in graph_events
                if event.event_type == "graph.step.started"
            )
            if owner is not None and not starts:
                plan, step = owner
                raise _journal_incompatible(
                    "generation rebase graph step attempt has no durable start",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_ATTEMPT_KIND_AMBIGUOUS
                    ),
                    event=attempt_events[0],
                    plan=plan,
                    step_index=step.index,
                )
            if starts:
                if owner is None:
                    start = starts[0]
                    identity = (
                        start.payload.get("graph_plan_id"),
                        start.payload.get("graph_scope_id"),
                    )
                    plan = plans_by_identity.get(identity)
                    if plan is not None:
                        matching = tuple(
                            step
                            for step in plan.steps
                            if step.attempt == start.attempt
                        )
                        if len(matching) == 1:
                            owner = (plan, matching[0])
                if owner is None:
                    raise _journal_incompatible(
                        "generation rebase graph step has no exact admitted plan",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID
                        ),
                        event=starts[0],
                    )
                plan, step = owner
                state = _classify_graph_step_attempt(
                    attempt_events,
                    journal_events=runtime_events,
                    plan=plan,
                    expected_step=step,
                    artifact_repository=artifact_repository,
                )
                key = (plan.plan_id, step.index)
                if key in step_states:
                    raise _journal_incompatible(
                        "generation rebase graph step state is ambiguous",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_STEP_SEQUENCE_INVALID
                        ),
                        event=starts[0],
                        plan=plan,
                        step_index=step.index,
                    )
                step_states[key] = state
                if state.status == "recovery_required":
                    if state.terminal is None:
                        raise TypeError("graph recovery state requires a terminal")
                    recovery_failures.append(
                        (
                            GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING,
                            state.terminal,
                            plan,
                            step.index,
                        )
                    )
                elif state.status == "blocked":
                    blocked_failures.append(
                        (
                            "generation rebase found an unfinished durable attempt",
                            GenerationRebaseFailureReason.ATTEMPT_OPEN,
                            _failure_subject_for_event(
                                starts[0],
                                plan=plan,
                                step_index=step.index,
                            ),
                        )
                    )
                continue

            if graph_events:
                raise _journal_incompatible(
                    "generation rebase graph attempt kind is ambiguous",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_ATTEMPT_KIND_AMBIGUOUS
                    ),
                    event=graph_events[0],
                )

            selection = select_attempt_terminal(
                attempt_events,
                allowed_following_event_types=_POST_TERMINAL_AUDIT_EVENT_TYPES,
            )
            if selection.ambiguous:
                raise _journal_incompatible(
                    "generation rebase attempt has duplicate terminal events",
                    reason=(
                        GenerationRebaseFailureReason.ATTEMPT_DUPLICATE_TERMINAL
                    ),
                    event=attempt_events[-1],
                )
            terminal = selection.event
            if terminal is None:
                canonical = tuple(
                    event
                    for event in attempt_events
                    if event.event_type in CANONICAL_ATTEMPT_TERMINALS
                )
                if canonical:
                    raise _journal_incompatible(
                        "generation rebase attempt continued after its terminal event",
                        reason=(
                            GenerationRebaseFailureReason.ATTEMPT_CONTINUED_AFTER_TERMINAL
                        ),
                        event=canonical[-1],
                    )
                blocked_failures.append(
                    (
                        "generation rebase found an unfinished durable attempt",
                        GenerationRebaseFailureReason.ATTEMPT_OPEN,
                        _failure_subject_for_event(attempt_events[-1]),
                    )
                )

        for orchestration_attempt_id, plan in orchestration_groups.items():
            attempt_events = ordered_attempts[orchestration_attempt_id]
            admission = next(
                event
                for event in attempt_events
                if event.event_type == "graph.execution.admitted"
            )
            execution_seals = tuple(
                event
                for event in attempt_events
                if event.event_type == "graph.execution.completed"
            )
            if len(execution_seals) > 1:
                raise _journal_incompatible(
                    "generation rebase graph execution seal is duplicated",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_DUPLICATED
                    ),
                    event=execution_seals[-1],
                    plan=plan,
                )
            canonical_terminals = tuple(
                event
                for event in attempt_events
                if event.event_type in CANONICAL_ATTEMPT_TERMINALS
            )
            resource_bearing_terminals = tuple(
                event
                for event in attempt_events
                if event.event_type in ATTEMPT_TERMINAL_EQUIVALENTS
                and event.resource_refs
            )
            if resource_bearing_terminals:
                raise _journal_incompatible(
                    "generation rebase root graph terminal carries resources",
                    reason=(
                        GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED
                    ),
                    event=resource_bearing_terminals[0],
                    plan=plan,
                )
            if len(canonical_terminals) > 1:
                raise _journal_incompatible(
                    "generation rebase orchestration has duplicate terminals",
                    reason=(
                        GenerationRebaseFailureReason.ATTEMPT_DUPLICATE_TERMINAL
                    ),
                    event=canonical_terminals[-1],
                    plan=plan,
                )
            terminal = canonical_terminals[0] if canonical_terminals else None
            following_terminal = (
                ()
                if terminal is None
                else tuple(
                    event
                    for event in attempt_events
                    if event.store_seq > terminal.store_seq
                )
            )
            if terminal is not None and any(
                event.event_type not in _POST_TERMINAL_AUDIT_EVENT_TYPES
                for event in following_terminal
            ):
                raise _journal_incompatible(
                    "generation rebase orchestration continued after its terminal",
                    reason=(
                        GenerationRebaseFailureReason.ATTEMPT_CONTINUED_AFTER_TERMINAL
                    ),
                    event=terminal,
                    plan=plan,
                )

            states = tuple(
                step_states.get((plan.plan_id, step.index))
                for step in plan.steps
            )
            completed = all(
                state is not None and state.status == "completed"
                for state in states
            )
            dead_indexes = tuple(
                index
                for index, state in enumerate(states)
                if state is not None and state.status in {"failed", "cancelled"}
            )
            active_indexes = tuple(
                index for index, state in enumerate(states) if state is not None
            )
            for index in active_indexes:
                if any(
                    states[predecessor] is None
                    or states[predecessor].status != "completed"
                    for predecessor in range(index)
                ):
                    state = states[index]
                    if state is None:
                        raise TypeError("active graph step state is unavailable")
                    raise _journal_incompatible(
                        "generation rebase graph step sequence is not a prefix",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_STEP_SEQUENCE_INVALID
                        ),
                        event=state.seal or state.terminal or admission,
                        plan=plan,
                        step_index=index,
                    )
            if execution_seals:
                _validate_graph_execution_seal(
                    execution_seals[0],
                    plan=plan,
                    final_step=states[-1],
                    artifact_repository=artifact_repository,
                )

            if terminal is not None:
                if (
                    not completed
                    or len(execution_seals) != 1
                    or execution_seals[0].store_seq >= terminal.store_seq
                ):
                    raise _journal_incompatible(
                        "generation rebase root graph terminal is not fully sealed",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED
                        ),
                        event=terminal,
                        plan=plan,
                    )
                continue

            if execution_seals:
                if execution_seals[0] != attempt_events[-1] or not completed:
                    raise _journal_incompatible(
                        "generation rebase graph execution seal is not terminal",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISMATCHED
                        ),
                        event=execution_seals[0],
                        plan=plan,
                    )
                continue

            if dead_indexes:
                dead_index = dead_indexes[0]
                if (
                    len(dead_indexes) != 1
                    or any(
                        states[index] is None
                        or states[index].status != "completed"
                        for index in range(dead_index)
                    )
                    or any(
                        plan.steps[index].attempt.attempt_id in ordered_attempts
                        for index in range(dead_index + 1, len(plan.steps))
                    )
                    or admission != attempt_events[-1]
                ):
                    raise _journal_incompatible(
                        "generation rebase graph dead prefix is invalid",
                        reason=(
                            GenerationRebaseFailureReason.GRAPH_STEP_SEQUENCE_INVALID
                        ),
                        event=states[dead_index].seal or admission,
                        plan=plan,
                        step_index=dead_index,
                    )
                continue

            if completed:
                if admission == attempt_events[-1]:
                    recovery_failures.append(
                        (
                            GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISSING,
                            admission,
                            plan,
                            None,
                        )
                    )
                else:
                    blocked_failures.append(
                        (
                            "generation rebase found an unfinished durable attempt",
                            GenerationRebaseFailureReason.ATTEMPT_OPEN,
                            _failure_subject_for_event(
                                attempt_events[-1],
                                plan=plan,
                            ),
                        )
                    )
                continue
            blocked_failures.append(
                (
                    "generation rebase found an unfinished durable attempt",
                    GenerationRebaseFailureReason.ATTEMPT_OPEN,
                    _failure_subject_for_event(admission, plan=plan),
                )
            )

        if recovery_failures:
            reason, event, plan, step_index = recovery_failures[0]
            raise GenerationRebaseRecoveryRequired(
                "generation rebase graph checkpoint recovery is required",
                reason=reason,
                subject=_failure_subject_for_event(
                    event,
                    plan=plan,
                    step_index=step_index,
                ),
            )
        if blocked_failures:
            message, reason, subject = blocked_failures[0]
            raise GenerationRebasePreflightBlocked(
                message,
                reason=reason,
                subject=subject,
            )

    @staticmethod
    def _ensure_initial_execution(
        connection: sqlite3.Connection,
        intent: GenerationRebaseIntent,
    ) -> None:
        execution = connection.execute(
            "SELECT next_store_seq FROM executions WHERE execution_id = ?",
            (intent.execution_id,),
        ).fetchone()
        if execution is None:
            connection.execute(
                "INSERT INTO executions(execution_id) VALUES (?)",
                (intent.execution_id,),
            )
            return
        event = connection.execute(
            "SELECT 1 FROM events WHERE execution_id = ? LIMIT 1",
            (intent.execution_id,),
        ).fetchone()
        operation = connection.execute(
            "SELECT 1 FROM operations WHERE execution_id = ? LIMIT 1",
            (intent.execution_id,),
        ).fetchone()
        if int(execution["next_store_seq"]) != 1 or event is not None or (
            operation is not None
        ):
            raise GenerationRebaseConflict(
                "generation create cannot claim a non-empty execution",
                reason=GenerationRebaseFailureReason.CHAT_BINDING_CONFLICT,
            )

    @staticmethod
    def _validate_preflight(intent: GenerationRebaseIntent) -> None:
        if not intent.preflight.permits_rebase:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase preflight found an unsanitized host snapshot",
                reason=GenerationRebaseFailureReason.HOST_SNAPSHOT_UNSANITIZED,
                subject={
                    "execution_id": intent.execution_id,
                    "generation_id": intent.previous_generation_id
                    or intent.generation_id,
                    "attempt_id": intent.attempt_id,
                },
            )

    def rebase(
        self,
        request: GenerationRebaseRequest,
    ) -> GenerationRebaseReceipt:
        """Atomically create, import, bind, and select one generation."""

        if not isinstance(request, GenerationRebaseRequest):
            raise TypeError("request must be a GenerationRebaseRequest")
        intent = request.intent
        expected_operation = build_generation_rebase_operation(
            operation_id=request.operation.operation_id,
            intent=intent,
        )
        if expected_operation != request.operation:
            raise GenerationRebaseConflict(
                "generation rebase operation payload hash changed",
                reason=(
                    GenerationRebaseFailureReason.OPERATION_IDENTITY_CONFLICT
                ),
            )
        self._validate_preflight(intent)
        target_key = self._target_key(intent)
        next_revision = intent.expected_head_revision + 1
        transition_payload = _host_transition_payload(intent)
        lifecycle_operation = OperationRef(
            _derived_operation_id(
                "generation-transition",
                request.operation,
                intent.generation_id,
            ),
            _digest(transition_payload),
        )
        attempt_payload = _host_attempt_payload(
            intent,
            head_revision=next_revision,
        )
        attempt_operation = OperationRef(
            _derived_operation_id(
                "generation-attempt-binding",
                request.operation,
                intent.attempt_id,
            ),
            _digest(attempt_payload),
        )
        try:
            with self._transaction(immediate=True) as connection:
                existing_primary = self._operation_row(
                    connection,
                    execution_id=intent.execution_id,
                    operation_id=request.operation.operation_id,
                )
                if existing_primary is not None:
                    if (
                        existing_primary["payload_sha256"]
                        != request.operation.payload_sha256
                        or existing_primary["target_kind"]
                        != "legacy_bootstrap_manifest"
                        or existing_primary["target_key"] != target_key
                    ):
                        raise GenerationRebaseConflict(
                            "generation rebase operation ID was reused",
                            reason=(
                                GenerationRebaseFailureReason.OPERATION_IDENTITY_CONFLICT
                            ),
                        )
                    receipt = self._receipt_for_generation(
                        connection,
                        owner_chat_id=intent.owner_chat_id,
                        execution_id=intent.execution_id,
                        session_id=intent.session_id,
                        generation_id=intent.generation_id,
                    )
                    if receipt is None or receipt.operation != request.operation:
                        raise GenerationRebaseJournalIncompatible(
                            "generation rebase operation receipt is incomplete"
                        )
                    return replace(receipt, duplicate=True)

                current_receipt = None
                host_head, bootstrap_head = self._head_rows(
                    connection,
                    intent.owner_chat_id,
                )
                self._assert_matching_heads(host_head, bootstrap_head)
                if intent.kind is GenerationRebaseKind.CREATE:
                    if host_head is not None:
                        raise GenerationRebaseConflict(
                            "generation create requires an empty chat head",
                            reason=(
                                GenerationRebaseFailureReason.CHAT_BINDING_CONFLICT
                            ),
                        )
                    orphaned_owner_state = any(
                        connection.execute(query, (intent.owner_chat_id,)).fetchone()
                        is not None
                        for query in (
                            "SELECT 1 FROM host_generation_chat_bindings "
                            "WHERE owner_chat_id = ? LIMIT 1",
                            "SELECT 1 FROM host_generation_records "
                            "WHERE owner_chat_id = ? LIMIT 1",
                            "SELECT 1 FROM legacy_bootstrap_manifests "
                            "WHERE owner_chat_id = ? LIMIT 1",
                        )
                    )
                    if orphaned_owner_state:
                        raise GenerationRebaseJournalIncompatible(
                            "generation create found owner state without a current head"
                        )
                    foreign_binding = connection.execute(
                        """
                        SELECT owner_chat_id FROM host_generation_chat_bindings
                        WHERE execution_id = ?
                        """,
                        (intent.execution_id,),
                    ).fetchone()
                    if foreign_binding is not None:
                        raise GenerationRebaseConflict(
                            "generation execution is already bound to a chat",
                            reason=(
                                GenerationRebaseFailureReason.CHAT_BINDING_CONFLICT
                            ),
                        )
                    self._ensure_initial_execution(connection, intent)
                    connection.execute(
                        """
                        INSERT INTO host_generation_chat_bindings(
                            owner_chat_id, execution_id, session_id
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            intent.owner_chat_id,
                            intent.execution_id,
                            intent.session_id,
                        ),
                    )
                else:
                    if host_head is None or bootstrap_head is None:
                        raise GenerationRebaseConflict(
                            "generation rebase has no current generation",
                            reason=(
                                GenerationRebaseFailureReason.SOURCE_GENERATION_CONFLICT
                            ),
                        )
                    if (
                        host_head["execution_id"] != intent.execution_id
                        or host_head["session_id"] != intent.session_id
                    ):
                        raise GenerationRebaseConflict(
                            "generation rebase binding is outside the durable chat",
                            reason=(
                                GenerationRebaseFailureReason.CHAT_BINDING_CONFLICT
                            ),
                        )
                    if (
                        host_head["current_generation_id"]
                        != intent.previous_generation_id
                    ):
                        raise GenerationRebaseConflict(
                            "generation rebase previous generation is not current",
                            reason=(
                                GenerationRebaseFailureReason.SOURCE_GENERATION_CONFLICT
                            ),
                        )
                    if int(host_head["revision"]) != intent.expected_head_revision:
                        raise GenerationRebaseConflict(
                            "generation rebase head revision is not current",
                            reason=(
                                GenerationRebaseFailureReason.HEAD_REVISION_CONFLICT
                            ),
                        )
                    current_receipt = self._receipt_for_generation(
                        connection,
                        owner_chat_id=intent.owner_chat_id,
                        execution_id=intent.execution_id,
                        session_id=intent.session_id,
                        generation_id=intent.previous_generation_id,
                    )
                    if (
                        current_receipt is None
                        or current_receipt.head_revision
                        != intent.expected_head_revision
                        or current_receipt.source_revision
                        != bootstrap_head["current_source_revision"]
                    ):
                        raise GenerationRebaseJournalIncompatible(
                            "generation rebase current receipt is incomplete"
                        )

                self._assert_durable_preflight(
                    connection,
                    intent,
                    current_receipt,
                )

                if connection.execute(
                    """
                    SELECT 1 FROM host_generation_records
                    WHERE owner_chat_id = ? AND generation_id = ?
                    """,
                    (intent.owner_chat_id, intent.generation_id),
                ).fetchone() is not None:
                    raise GenerationRebaseConflict(
                        "generation ID already belongs to a lifecycle record",
                        reason=(
                            GenerationRebaseFailureReason.SOURCE_GENERATION_CONFLICT
                        ),
                    )
                if connection.execute(
                    """
                    SELECT 1 FROM legacy_bootstrap_manifests
                    WHERE owner_chat_id = ? AND source_revision = ?
                    """,
                    (intent.owner_chat_id, intent.source_revision),
                ).fetchone() is not None:
                    raise GenerationRebaseConflict(
                        "generation source revision is already imported",
                        reason=(
                            GenerationRebaseFailureReason.SOURCE_GENERATION_CONFLICT
                        ),
                    )
                if connection.execute(
                    """
                    SELECT 1 FROM host_generation_attempt_bindings
                    WHERE execution_id = ? AND attempt_id = ?
                    """,
                    (intent.execution_id, intent.attempt_id),
                ).fetchone() is not None:
                    raise GenerationRebaseConflict(
                        "generation attempt ID is already bound",
                        reason=(
                            GenerationRebaseFailureReason.SOURCE_GENERATION_CONFLICT
                        ),
                    )

                if not self._claim_operation(
                    connection,
                    execution_id=intent.execution_id,
                    operation=request.operation,
                    target_kind="legacy_bootstrap_manifest",
                    target_key=target_key,
                ):
                    raise GenerationRebaseJournalIncompatible(
                        "generation rebase operation replay lost its receipt"
                    )
                execution = connection.execute(
                    "SELECT next_store_seq FROM executions WHERE execution_id = ?",
                    (intent.execution_id,),
                ).fetchone()
                if execution is None:
                    raise GenerationRebaseJournalIncompatible(
                        "generation rebase journal head is unavailable"
                    )
                next_store_seq = int(execution["next_store_seq"])
                events: list[JournalEvent] = []
                manifest_events: list[dict[str, Any]] = []
                drafts: tuple[
                    tuple[SemanticEventDraft, GenerationSnapshotMessage | None],
                    ...,
                ] = (
                    tuple(
                        (self._message_draft(intent, message, index), message)
                        for index, message in enumerate(intent.messages)
                    )
                    if intent.messages
                    else ((self._marker_draft(intent), None),)
                )
                for index, (draft, message) in enumerate(drafts):
                    event = JournalEvent(
                        event_id=draft.event_id,
                        event_type=draft.event_type,
                        attempt=draft.attempt,
                        operation=draft.operation,
                        store_seq=next_store_seq + index,
                        payload=draft.payload,
                        resource_refs=draft.resource_refs,
                    )
                    if not self._claim_operation(
                        connection,
                        execution_id=intent.execution_id,
                        operation=event.operation,
                        target_kind="journal_event",
                        target_key=event.event_id,
                    ):
                        raise GenerationRebaseConflict(
                            "generation rebase event operation already exists",
                            reason=(
                                GenerationRebaseFailureReason.OPERATION_IDENTITY_CONFLICT
                            ),
                        )
                    event_json = _canonical_bytes(event.to_dict())
                    event_sha256 = hashlib.sha256(event_json).hexdigest()
                    connection.execute(
                        """
                        INSERT INTO events(
                            execution_id, store_seq, event_id, generation_id,
                            attempt_id, event_type, operation_id,
                            event_json, event_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            intent.execution_id,
                            event.store_seq,
                            event.event_id,
                            intent.generation_id,
                            intent.attempt_id,
                            event.event_type,
                            event.operation.operation_id,
                            event_json,
                            event_sha256,
                        ),
                    )
                    events.append(event)
                    descriptor = {
                        "event_id": event.event_id,
                        "store_seq": event.store_seq,
                        "event_sha256": event_sha256,
                    }
                    if message is not None:
                        descriptor.update(
                            {
                                "message_id": message.message_id,
                                "role": message.role,
                            }
                        )
                    else:
                        descriptor["record_kind"] = "generation_marker"
                    manifest_events.append(descriptor)
                advanced = connection.execute(
                    """
                    UPDATE executions SET next_store_seq = ?
                    WHERE execution_id = ? AND next_store_seq = ?
                    """,
                    (
                        next_store_seq + len(events),
                        intent.execution_id,
                        next_store_seq,
                    ),
                )
                if advanced.rowcount != 1:
                    raise GenerationRebaseConflict(
                        "generation rebase journal head changed",
                        reason=(
                            GenerationRebaseFailureReason.HEAD_REVISION_CONFLICT
                        ),
                    )

                manifest = {
                    "schema": "unchain.legacy_bootstrap_manifest.v1",
                    "owner_chat_id": intent.owner_chat_id,
                    "source_revision": intent.source_revision,
                    "session_id": intent.session_id,
                    "execution_id": intent.execution_id,
                    "generation_id": intent.generation_id,
                    "attempt_id": intent.attempt_id,
                    "capture_status": _LEGACY_CAPTURE_STATUS,
                    "payload_sha256": request.operation.payload_sha256,
                    "primary_operation_id": request.operation.operation_id,
                    "rebase": {
                        "kind": self._manifest_kind(intent.kind),
                        "previous_generation_id": intent.previous_generation_id,
                    },
                    "preflight_proof_id": intent.preflight.proof_id,
                    "task_state": (
                        intent.task_state.to_record()
                        if intent.task_state is not None
                        else None
                    ),
                    "events": manifest_events,
                }
                manifest_json = _canonical_bytes(manifest)
                manifest_sha256 = hashlib.sha256(manifest_json).hexdigest()
                connection.execute(
                    """
                    INSERT INTO legacy_bootstrap_manifests(
                        owner_chat_id, source_revision, session_id,
                        execution_id, generation_id, attempt_id,
                        operation_id, payload_sha256,
                        manifest_json, manifest_sha256,
                        first_store_seq, last_store_seq, event_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        intent.source_revision,
                        intent.session_id,
                        intent.execution_id,
                        intent.generation_id,
                        intent.attempt_id,
                        request.operation.operation_id,
                        request.operation.payload_sha256,
                        manifest_json,
                        manifest_sha256,
                        events[0].store_seq,
                        events[-1].store_seq,
                        len(events),
                    ),
                )

                connection.execute(
                    """
                    INSERT INTO host_generation_records(
                        owner_chat_id, execution_id, session_id, generation_id,
                        transition_kind, previous_generation_id, revision,
                        operation_id, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        intent.execution_id,
                        intent.session_id,
                        intent.generation_id,
                        self._compatible_kind(intent.kind),
                        intent.previous_generation_id,
                        next_revision,
                        lifecycle_operation.operation_id,
                        lifecycle_operation.payload_sha256,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO host_generation_operations(
                        owner_chat_id, operation_id, payload_sha256,
                        mutation_kind, result_generation_id, result_revision
                    ) VALUES (?, ?, ?, 'transition', ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        lifecycle_operation.operation_id,
                        lifecycle_operation.payload_sha256,
                        intent.generation_id,
                        next_revision,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO host_generation_operations(
                        owner_chat_id, operation_id, payload_sha256,
                        mutation_kind, result_generation_id,
                        result_revision, result_attempt_id
                    ) VALUES (?, ?, ?, 'attempt_binding', ?, ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        attempt_operation.operation_id,
                        attempt_operation.payload_sha256,
                        intent.generation_id,
                        next_revision,
                        intent.attempt_id,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO host_generation_attempt_bindings(
                        owner_chat_id, execution_id, session_id, generation_id,
                        attempt_id, head_revision, operation_id, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        intent.execution_id,
                        intent.session_id,
                        intent.generation_id,
                        intent.attempt_id,
                        next_revision,
                        attempt_operation.operation_id,
                        attempt_operation.payload_sha256,
                    ),
                )

                if host_head is None:
                    connection.execute(
                        """
                        INSERT INTO host_generation_heads(
                            owner_chat_id, execution_id, session_id,
                            current_generation_id, revision
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            intent.owner_chat_id,
                            intent.execution_id,
                            intent.session_id,
                            intent.generation_id,
                            next_revision,
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO legacy_bootstrap_chat_heads(
                            owner_chat_id, execution_id, session_id,
                            current_generation_id, current_source_revision,
                            head_revision
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            intent.owner_chat_id,
                            intent.execution_id,
                            intent.session_id,
                            intent.generation_id,
                            intent.source_revision,
                            next_revision,
                        ),
                    )
                else:
                    host_updated = connection.execute(
                        """
                        UPDATE host_generation_heads
                        SET current_generation_id = ?, revision = ?
                        WHERE owner_chat_id = ? AND execution_id = ?
                          AND session_id = ? AND current_generation_id = ?
                          AND revision = ?
                        """,
                        (
                            intent.generation_id,
                            next_revision,
                            intent.owner_chat_id,
                            intent.execution_id,
                            intent.session_id,
                            intent.previous_generation_id,
                            intent.expected_head_revision,
                        ),
                    )
                    bootstrap_updated = connection.execute(
                        """
                        UPDATE legacy_bootstrap_chat_heads
                        SET current_generation_id = ?,
                            current_source_revision = ?,
                            head_revision = ?
                        WHERE owner_chat_id = ? AND execution_id = ?
                          AND session_id = ? AND current_generation_id = ?
                          AND head_revision = ?
                        """,
                        (
                            intent.generation_id,
                            intent.source_revision,
                            next_revision,
                            intent.owner_chat_id,
                            intent.execution_id,
                            intent.session_id,
                            intent.previous_generation_id,
                            intent.expected_head_revision,
                        ),
                    )
                    if host_updated.rowcount != 1 or (
                        bootstrap_updated.rowcount != 1
                    ):
                        raise GenerationRebaseConflict(
                            "generation rebase head changed during compare-and-swap",
                            reason=(
                                GenerationRebaseFailureReason.HEAD_REVISION_CONFLICT
                            ),
                        )

                receipt = self._receipt_for_generation(
                    connection,
                    owner_chat_id=intent.owner_chat_id,
                    execution_id=intent.execution_id,
                    session_id=intent.session_id,
                    generation_id=intent.generation_id,
                )
                if receipt is None or receipt.operation != request.operation:
                    raise GenerationRebaseJournalIncompatible(
                        "generation rebase receipt was not durably completed"
                    )
                return receipt
        except GenerationRebaseError:
            raise
        except sqlite3.IntegrityError as error:
            raise GenerationRebaseConflict(
                "generation rebase conflicted with durable state",
                reason=GenerationRebaseFailureReason.CHAT_BINDING_CONFLICT,
            ) from error
        except sqlite3.Error as error:
            raise GenerationRebaseUnavailable(
                "generation rebase SQLite transaction failed",
                reason=(
                    GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                ),
            ) from error

    @staticmethod
    def _decode_manifest(row: sqlite3.Row) -> Mapping[str, Any]:
        raw = bytes(row["manifest_json"])
        if hashlib.sha256(raw).hexdigest() != row["manifest_sha256"]:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest digest changed"
            )
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest is unreadable"
            ) from error
        if type(decoded) is not dict or _canonical_bytes(decoded) != raw:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest is not canonical"
            )
        expected = {
            "schema",
            "owner_chat_id",
            "source_revision",
            "session_id",
            "execution_id",
            "generation_id",
            "attempt_id",
            "capture_status",
            "payload_sha256",
            "primary_operation_id",
            "rebase",
            "preflight_proof_id",
            "task_state",
            "events",
        }
        if set(decoded) != expected or decoded["schema"] != (
            "unchain.legacy_bootstrap_manifest.v1"
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest shape changed"
            )
        for field_name in (
            "owner_chat_id",
            "source_revision",
            "session_id",
            "execution_id",
            "generation_id",
            "attempt_id",
            "payload_sha256",
        ):
            if decoded[field_name] != row[field_name]:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase manifest index changed"
                )
        if (
            decoded["primary_operation_id"] != row["operation_id"]
            or decoded["capture_status"] != _LEGACY_CAPTURE_STATUS
            or type(decoded["rebase"]) is not dict
            or set(decoded["rebase"])
            != {"kind", "previous_generation_id"}
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest authority changed"
            )
        events = decoded["events"]
        if (
            type(events) is not list
            or len(events) != row["event_count"]
            or not events
            or events[0].get("store_seq") != row["first_store_seq"]
            or events[-1].get("store_seq") != row["last_store_seq"]
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest event range changed"
            )
        event_kinds = tuple(
            (
                "generation_marker"
                if type(event) is dict
                and event.get("record_kind") == "generation_marker"
                else "message"
                if type(event) is dict
                and set(event)
                == {
                    "message_id",
                    "role",
                    "event_id",
                    "store_seq",
                    "event_sha256",
                }
                else "invalid"
            )
            for event in events
        )
        if event_kinds != ("generation_marker",) and (
            not event_kinds or any(kind != "message" for kind in event_kinds)
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase manifest message accounting changed"
            )
        return MappingProxyType(decoded)

    @classmethod
    def _verify_events(
        cls,
        connection: sqlite3.Connection,
        *,
        row: sqlite3.Row,
        manifest: Mapping[str, Any],
        expected_head_revision: int,
    ) -> None:
        persisted = list(
            connection.execute(
                """
                SELECT * FROM events
                WHERE execution_id = ?
                  AND store_seq BETWEEN ? AND ?
                ORDER BY store_seq
                """,
                (
                    row["execution_id"],
                    row["first_store_seq"],
                    row["last_store_seq"],
                ),
            )
        )
        expected = manifest["events"]
        if len(persisted) != row["event_count"] or len(persisted) != len(expected):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase journal range is incomplete"
            )
        for event_row, descriptor in zip(persisted, expected, strict=True):
            if type(descriptor) is not dict:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase event descriptor is invalid"
                )
            raw = bytes(event_row["event_json"])
            if hashlib.sha256(raw).hexdigest() != event_row["event_sha256"]:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase event digest changed"
                )
            try:
                event = JournalEvent.from_dict(json.loads(raw.decode("utf-8")))
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as error:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase event is unreadable"
                ) from error
            operation = cls._operation_row(
                connection,
                execution_id=row["execution_id"],
                operation_id=event.operation.operation_id,
            )
            descriptor_kind = (
                "generation_marker"
                if descriptor.get("record_kind") == "generation_marker"
                else "message"
            )
            legacy_provenance = event.payload.get("legacy_provenance")
            common_invalid = (
                _canonical_bytes(event.to_dict()) != raw
                or event.event_id != descriptor.get("event_id")
                or event.store_seq != descriptor.get("store_seq")
                or event_row["event_sha256"] != descriptor.get("event_sha256")
                or event_row["event_id"] != event.event_id
                or event_row["store_seq"] != event.store_seq
                or event_row["generation_id"]
                != event.attempt.generation.generation_id
                or event_row["attempt_id"] != event.attempt.attempt_id
                or event_row["event_type"] != event.event_type
                or event_row["operation_id"] != event.operation.operation_id
                or event.attempt.generation.execution_id != row["execution_id"]
                or event.attempt.generation.generation_id != row["generation_id"]
                or event.attempt.attempt_id != row["attempt_id"]
                or event.payload.get("run_id") != row["attempt_id"]
                or not isinstance(legacy_provenance, Mapping)
                or legacy_provenance.get("source")
                != "host_sanitized_generation_snapshot"
                or legacy_provenance.get("capture_status")
                != _LEGACY_CAPTURE_STATUS
                or legacy_provenance.get("owner_chat_id") != row["owner_chat_id"]
                or legacy_provenance.get("session_id") != row["session_id"]
                or legacy_provenance.get("source_revision")
                != row["source_revision"]
                or operation is None
                or operation["payload_sha256"] != event.operation.payload_sha256
                or operation["target_kind"] != "journal_event"
                or operation["target_key"] != event.event_id
            )
            if descriptor_kind == "message":
                message = event.payload.get("message")
                kind_invalid = (
                    set(descriptor)
                    != {
                        "message_id",
                        "role",
                        "event_id",
                        "store_seq",
                        "event_sha256",
                    }
                    or event.event_type
                    not in {"message.user", "message.assistant"}
                    or not isinstance(message, Mapping)
                    or message.get("role") != descriptor.get("role")
                    or event.event_type != f"message.{descriptor.get('role')}"
                    or legacy_provenance.get("message_id")
                    != descriptor.get("message_id")
                )
            elif descriptor_kind == "generation_marker":
                generation_rebase = event.payload.get("generation_rebase")
                kind_invalid = (
                    set(descriptor)
                    != {
                        "record_kind",
                        "event_id",
                        "store_seq",
                        "event_sha256",
                    }
                    or event.event_type != "generation.rebased"
                    or not isinstance(generation_rebase, Mapping)
                    or dict(generation_rebase)
                    != {
                        "kind": cls._kind_from_manifest(
                            manifest["rebase"]["kind"]
                        ).value,
                        "previous_generation_id": manifest["rebase"][
                            "previous_generation_id"
                        ],
                        "expected_head_revision": expected_head_revision,
                        "empty_snapshot": True,
                        "replacement_message_count": 0,
                    }
                )
            else:
                kind_invalid = True
            if common_invalid or kind_invalid:
                raise GenerationRebaseJournalIncompatible(
                    "generation rebase event binding changed"
                )

    def _receipt_for_generation(
        self,
        connection: sqlite3.Connection,
        *,
        owner_chat_id: str,
        execution_id: str,
        session_id: str,
        generation_id: str,
    ) -> GenerationRebaseReceipt | None:
        manifest_row = connection.execute(
            """
            SELECT * FROM legacy_bootstrap_manifests
            WHERE owner_chat_id = ? AND generation_id = ?
            """,
            (owner_chat_id, generation_id),
        ).fetchone()
        if manifest_row is None:
            return None
        lifecycle = connection.execute(
            """
            SELECT * FROM host_generation_records
            WHERE owner_chat_id = ? AND generation_id = ?
            """,
            (owner_chat_id, generation_id),
        ).fetchone()
        attempt = connection.execute(
            """
            SELECT * FROM host_generation_attempt_bindings
            WHERE owner_chat_id = ? AND execution_id = ?
              AND generation_id = ? AND attempt_id = ?
            """,
            (
                owner_chat_id,
                execution_id,
                generation_id,
                manifest_row["attempt_id"],
            ),
        ).fetchone()
        if lifecycle is None or attempt is None:
            raise GenerationRebaseJournalIncompatible(
                "generation rebase lifecycle receipt is incomplete"
            )
        manifest = self._decode_manifest(manifest_row)
        kind = self._kind_from_manifest(manifest["rebase"]["kind"])
        operation = OperationRef(
            manifest_row["operation_id"],
            manifest_row["payload_sha256"],
        )
        primary = self._operation_row(
            connection,
            execution_id=execution_id,
            operation_id=operation.operation_id,
        )
        lifecycle_operation = OperationRef(
            lifecycle["operation_id"],
            lifecycle["payload_sha256"],
        )
        attempt_operation = OperationRef(
            attempt["operation_id"],
            attempt["payload_sha256"],
        )
        lifecycle_receipt = connection.execute(
            """
            SELECT * FROM host_generation_operations
            WHERE owner_chat_id = ? AND operation_id = ?
            """,
            (owner_chat_id, lifecycle_operation.operation_id),
        ).fetchone()
        attempt_receipt = connection.execute(
            """
            SELECT * FROM host_generation_operations
            WHERE owner_chat_id = ? AND operation_id = ?
            """,
            (owner_chat_id, attempt_operation.operation_id),
        ).fetchone()
        expected_compatible_kind = self._compatible_kind(kind)
        if (
            manifest_row["execution_id"] != execution_id
            or manifest_row["session_id"] != session_id
            or lifecycle["execution_id"] != execution_id
            or lifecycle["session_id"] != session_id
            or lifecycle["transition_kind"] != expected_compatible_kind
            or lifecycle["previous_generation_id"]
            != manifest["rebase"]["previous_generation_id"]
            or int(lifecycle["revision"]) != int(attempt["head_revision"])
            or primary is None
            or primary["payload_sha256"] != operation.payload_sha256
            or primary["target_kind"] != "legacy_bootstrap_manifest"
            or primary["target_key"] != f"{owner_chat_id}:{generation_id}"
            or lifecycle_receipt is None
            or lifecycle_receipt["payload_sha256"]
            != lifecycle_operation.payload_sha256
            or lifecycle_receipt["mutation_kind"] != "transition"
            or lifecycle_receipt["result_generation_id"] != generation_id
            or int(lifecycle_receipt["result_revision"])
            != int(lifecycle["revision"])
            or attempt_receipt is None
            or attempt_receipt["payload_sha256"]
            != attempt_operation.payload_sha256
            or attempt_receipt["mutation_kind"] != "attempt_binding"
            or attempt_receipt["result_generation_id"] != generation_id
            or attempt_receipt["result_attempt_id"] != attempt["attempt_id"]
        ):
            raise GenerationRebaseJournalIncompatible(
                "generation rebase durable authorities diverged"
            )
        self._verify_events(
            connection,
            row=manifest_row,
            manifest=manifest,
            expected_head_revision=int(lifecycle["revision"]) - 1,
        )
        task_state = (
            GenerationTaskStateDescriptor.from_record(manifest["task_state"])
            if manifest["task_state"] is not None
            else None
        )
        return GenerationRebaseReceipt(
            owner_chat_id=owner_chat_id,
            session_id=session_id,
            execution_id=execution_id,
            generation_id=generation_id,
            attempt_id=manifest_row["attempt_id"],
            kind=kind,
            previous_generation_id=manifest["rebase"][
                "previous_generation_id"
            ],
            source_revision=manifest_row["source_revision"],
            head_revision=int(lifecycle["revision"]),
            manifest_sha256=manifest_row["manifest_sha256"],
            message_count=(
                0
                if manifest["events"][0].get("record_kind")
                == "generation_marker"
                else len(manifest["events"])
            ),
            first_cursor=EventCursor(
                manifest["events"][0]["store_seq"],
                manifest["events"][0]["event_id"],
            ),
            last_cursor=EventCursor(
                manifest["events"][-1]["store_seq"],
                manifest["events"][-1]["event_id"],
            ),
            operation=operation,
            lifecycle_operation=lifecycle_operation,
            attempt_binding_operation=attempt_operation,
            task_state=task_state,
        )

    def current(
        self,
        *,
        owner_chat_id: str,
        execution_id: str,
        session_id: str,
    ) -> GenerationRebaseHead | None:
        """Read the one head only when lifecycle and bootstrap agree."""

        owner = _required_text(owner_chat_id, "owner_chat_id", identifier=True)
        execution = _required_text(execution_id, "execution_id", identifier=True)
        session = _required_text(session_id, "session_id", identifier=True)
        try:
            with self._transaction(immediate=False) as connection:
                host, bootstrap = self._head_rows(connection, owner)
                self._assert_matching_heads(host, bootstrap)
                if host is None or bootstrap is None:
                    return None
                if host["execution_id"] != execution or host["session_id"] != session:
                    raise GenerationRebaseConflict(
                        "generation head read is outside the durable chat binding",
                        reason=(
                            GenerationRebaseFailureReason.CHAT_BINDING_CONFLICT
                        ),
                    )
                receipt = self._receipt_for_generation(
                    connection,
                    owner_chat_id=owner,
                    execution_id=execution,
                    session_id=session,
                    generation_id=host["current_generation_id"],
                )
                if (
                    receipt is None
                    or receipt.head_revision != int(host["revision"])
                    or receipt.source_revision
                    != bootstrap["current_source_revision"]
                ):
                    raise GenerationRebaseJournalIncompatible(
                        "generation head has no complete durable receipt"
                    )
                return GenerationRebaseHead(
                    owner_chat_id=owner,
                    session_id=session,
                    execution_id=execution,
                    current_generation_id=host["current_generation_id"],
                    current_attempt_id=receipt.attempt_id,
                    current_source_revision=bootstrap["current_source_revision"],
                    revision=int(host["revision"]),
                )
        except GenerationRebaseError:
            raise
        except sqlite3.Error as error:
            raise GenerationRebaseUnavailable(
                "generation rebase current-head read failed",
                reason=(
                    GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                ),
            ) from error

    def receipt_for_generation(
        self,
        *,
        owner_chat_id: str,
        execution_id: str,
        session_id: str,
        generation_id: str,
    ) -> GenerationRebaseReceipt | None:
        """Read and verify one immutable historical generation receipt."""

        owner = _required_text(owner_chat_id, "owner_chat_id", identifier=True)
        execution = _required_text(execution_id, "execution_id", identifier=True)
        session = _required_text(session_id, "session_id", identifier=True)
        generation = _required_text(
            generation_id,
            "generation_id",
            identifier=True,
        )
        try:
            with self._transaction(immediate=False) as connection:
                return self._receipt_for_generation(
                    connection,
                    owner_chat_id=owner,
                    execution_id=execution,
                    session_id=session,
                    generation_id=generation,
                )
        except GenerationRebaseError:
            raise
        except sqlite3.Error as error:
            raise GenerationRebaseUnavailable(
                "generation rebase receipt read failed",
                reason=(
                    GenerationRebaseFailureReason.INFRASTRUCTURE_UNAVAILABLE
                ),
            ) from error


def _recovery_authority_snapshot(
    service: SQLiteGenerationRebaseV2Service,
    execution_id: str,
) -> dict[str, dict[object, tuple[Any, ...]]]:
    connection = service._connect()
    try:
        execution_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT execution_id, next_store_seq
                FROM executions WHERE execution_id = ?
                """,
                (execution_id,),
            )
        )
        event_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT store_seq, event_id, generation_id, attempt_id,
                       event_type, operation_id, event_json, event_sha256
                FROM events WHERE execution_id = ? ORDER BY store_seq
                """,
                (execution_id,),
            )
        )
        operation_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT operation_id, payload_sha256, target_kind, target_key
                FROM operations WHERE execution_id = ? ORDER BY operation_id
                """,
                (execution_id,),
            )
        )
        receipt_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT store_seq, receipt_kind, generation_id, attempt_id,
                       call_id, iteration
                FROM event_receipts
                WHERE execution_id = ? ORDER BY store_seq, receipt_kind
                """,
                (execution_id,),
            )
        )
        artifact_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT artifact_id, revision, logical_kind, logical_key,
                       object_sha256, media_type, byte_length, preview,
                       operation_id, artifact_json, artifact_record_sha256
                FROM artifacts WHERE execution_id = ?
                ORDER BY artifact_id, revision
                """,
                (execution_id,),
            )
        )
        object_rows = tuple(
            tuple(row)
            for row in connection.execute(
                """
                SELECT DISTINCT objects.sha256, objects.byte_length
                FROM objects
                JOIN artifacts ON artifacts.object_sha256 = objects.sha256
                WHERE artifacts.execution_id = ? ORDER BY objects.sha256
                """,
                (execution_id,),
            )
        )
    finally:
        connection.close()
    return {
        "execution": {row[0]: row for row in execution_rows},
        "events": {row[1]: row for row in event_rows},
        "operations": {row[0]: row for row in operation_rows},
        "event_receipts": {
            (row[0], row[1]): row for row in receipt_rows
        },
        "artifacts": {(row[0], row[1]): row for row in artifact_rows},
        "objects": {row[0]: row for row in object_rows},
    }


def _recovery_authority_additions(
    before: Mapping[object, tuple[Any, ...]],
    after: Mapping[object, tuple[Any, ...]],
    *,
    authority: str,
) -> dict[object, tuple[Any, ...]]:
    if any(after.get(key) != value for key, value in before.items()):
        raise GraphCheckpointError(
            f"generation rebase recovery mutated existing {authority} authority"
        )
    return {key: value for key, value in after.items() if key not in before}


def _verify_recovery_authority_delta(
    *,
    before: Mapping[str, Mapping[object, tuple[Any, ...]]],
    after: Mapping[str, Mapping[object, tuple[Any, ...]]],
    target: JournalEvent,
) -> tuple[int, int]:
    additions = {
        authority: _recovery_authority_additions(
            before[authority],
            after[authority],
            authority=authority,
        )
        for authority in (
            "events",
            "operations",
            "event_receipts",
            "artifacts",
            "objects",
        )
    }
    if additions["event_receipts"]:
        raise GraphCheckpointError(
            "generation rebase recovery appended an unexpected event receipt"
        )
    added_events = additions["events"]
    if set(added_events) - {target.event_id} or len(added_events) > 1:
        raise GraphCheckpointError(
            "generation rebase recovery appended a foreign journal event"
        )
    if added_events:
        added_event_row = added_events[target.event_id]
        if (
            added_event_row[0] != target.store_seq
            or added_event_row[2] != target.attempt.generation.generation_id
            or added_event_row[3] != target.attempt.attempt_id
            or added_event_row[4] != target.event_type
            or added_event_row[5] != target.operation.operation_id
            or bytes(added_event_row[6]) != _canonical_bytes(target.to_dict())
            or added_event_row[7]
            != hashlib.sha256(bytes(added_event_row[6])).hexdigest()
        ):
            raise GraphCheckpointError(
                "generation rebase recovery appended a changed journal event"
            )

    referenced_artifacts = {
        (ref.resource_id, ref.revision)
        for ref in target.resource_refs
        if ref.kind == "artifact" and not ref.fragment
    }
    added_artifacts = additions["artifacts"]
    if set(added_artifacts) - referenced_artifacts or len(added_artifacts) > 1:
        raise GraphCheckpointError(
            "generation rebase recovery appended a foreign artifact"
        )
    allowed_operations = {
        row[8] for row in added_artifacts.values()
    } | ({target.operation.operation_id} if added_events else set())
    if set(additions["operations"]) - allowed_operations:
        raise GraphCheckpointError(
            "generation rebase recovery appended a foreign operation"
        )
    allowed_objects = {row[4] for row in added_artifacts.values()}
    if set(additions["objects"]) - allowed_objects:
        raise GraphCheckpointError(
            "generation rebase recovery appended a foreign object"
        )

    before_execution = before["execution"]
    after_execution = after["execution"]
    if set(before_execution) != set(after_execution) or len(after_execution) != 1:
        raise GraphCheckpointError(
            "generation rebase recovery changed execution authority"
        )
    execution_id = next(iter(after_execution))
    if after_execution[execution_id][1] != (
        before_execution[execution_id][1] + len(added_events)
    ):
        raise GraphCheckpointError(
            "generation rebase recovery changed the journal head unexpectedly"
        )
    return len(added_events), len(added_artifacts)


def recover_generation_rebase_attempt(
    *,
    service: SQLiteGenerationRebaseV2Service,
    request: GenerationRebaseRequest,
    failure: GenerationRebaseRecoveryRequired,
    artifact_sanitizer: Callable[[bytes, str], bytes],
) -> GenerationRebaseRecoveryResult:
    """Perform one exact graph recovery action for one classified failure.

    This is the narrow host seam for a sidecar.  It never admits a plan and
    never calls ``rebase``; the caller may replay its frozen request once only
    after this function returns.
    """

    if type(service) is not SQLiteGenerationRebaseV2Service:
        raise TypeError("service must be SQLiteGenerationRebaseV2Service")
    if not isinstance(request, GenerationRebaseRequest):
        raise TypeError("request must be GenerationRebaseRequest")
    if not isinstance(failure, GenerationRebaseRecoveryRequired):
        raise TypeError("failure must be GenerationRebaseRecoveryRequired")
    if not callable(artifact_sanitizer):
        raise TypeError("artifact_sanitizer must be callable")

    detail = generation_rebase_failure_detail(failure)
    if detail is None:
        raise GraphCheckpointError(
            "generation rebase recovery detail is unavailable"
        )
    try:
        reason = GenerationRebaseFailureReason(detail["reason"])
    except (KeyError, TypeError, ValueError) as error:
        raise GraphCheckpointError(
            "generation rebase recovery reason is invalid"
        ) from error
    if reason not in {
        GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING,
        GenerationRebaseFailureReason.GRAPH_EXECUTION_SEAL_MISSING,
    } or failure.reason != reason.value:
        raise GraphCheckpointError(
            "generation rebase failure is not recoverable"
        )
    subject = detail["subject"]
    if not isinstance(subject, Mapping):
        raise GraphCheckpointError(
            "generation rebase recovery subject is unavailable"
        )
    common_keys = {
        "execution_id",
        "generation_id",
        "attempt_id",
        "orchestration_attempt_id",
        "event_type",
        "store_seq",
        "event_id",
        "graph_plan_id",
        "graph_scope_id",
    }
    expected_keys = (
        common_keys | {"step_index"}
        if reason is GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING
        else common_keys
    )
    if set(subject) != expected_keys:
        raise GraphCheckpointError(
            "generation rebase recovery subject is not closed"
        )

    intent = request.intent
    if request.operation != build_generation_rebase_operation(
        operation_id=request.operation.operation_id,
        intent=intent,
    ):
        raise GraphCheckpointError(
            "generation rebase recovery request operation changed identity"
        )
    if (
        intent.kind is GenerationRebaseKind.CREATE
        or subject["execution_id"] != intent.execution_id
        or subject["generation_id"] != intent.previous_generation_id
    ):
        raise GraphCheckpointError(
            "generation rebase recovery request changed identity"
        )
    journal = service._store.bind_execution(intent.execution_id)
    plan = locate_graph_execution_plan(
        journal,
        generation_id=intent.previous_generation_id,
        orchestration_attempt_id=str(subject["orchestration_attempt_id"]),
    )
    if (
        plan.plan_id != subject["graph_plan_id"]
        or plan.scope_id != subject["graph_scope_id"]
        or plan.orchestration_attempt.attempt_id
        != subject["orchestration_attempt_id"]
    ):
        raise GraphCheckpointError(
            "generation rebase recovery plan changed identity"
        )

    snapshot = journal.capture_snapshot()
    evidence = tuple(
        event
        for event in snapshot.events
        if event.store_seq == subject["store_seq"]
        and event.event_id == subject["event_id"]
    )
    if len(evidence) != 1:
        raise GraphCheckpointError(
            "generation rebase recovery evidence is unavailable"
        )
    evidence_event = evidence[0]
    if (
        evidence_event.event_type != subject["event_type"]
        or evidence_event.attempt.attempt_id != subject["attempt_id"]
        or evidence_event.attempt.generation.execution_id
        != subject["execution_id"]
        or evidence_event.attempt.generation.generation_id
        != subject["generation_id"]
    ):
        raise GraphCheckpointError(
            "generation rebase recovery evidence changed identity"
        )

    step: GraphStepBinding | None = None
    if reason is GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING:
        step_index = subject["step_index"]
        if (
            isinstance(step_index, bool)
            or not isinstance(step_index, int)
            or not 0 <= step_index < len(plan.steps)
        ):
            raise GraphCheckpointError(
                "generation rebase recovery step index is invalid"
            )
        step = plan.steps[step_index]
        if (
            step.attempt != evidence_event.attempt
            or evidence_event.event_type not in ATTEMPT_TERMINAL_EQUIVALENTS
        ):
            raise GraphCheckpointError(
                "generation rebase recovery step evidence is invalid"
            )
    elif (
        evidence_event.event_type != "graph.execution.admitted"
        or evidence_event.attempt != plan.orchestration_attempt
    ):
        raise GraphCheckpointError(
            "generation rebase recovery execution evidence is invalid"
        )

    def target_events(events: tuple[JournalEvent, ...]) -> tuple[JournalEvent, ...]:
        if step is not None:
            return tuple(
                event
                for event in events
                if event.attempt == step.attempt
                and event.event_type in GRAPH_STEP_SEALS
                and event.payload.get("graph_plan_id") == plan.plan_id
                and event.payload.get("graph_scope_id") == plan.scope_id
            )
        return tuple(
            event
            for event in events
            if event.attempt == plan.orchestration_attempt
            and event.event_type == "graph.execution.completed"
            and event.payload.get("graph_plan_id") == plan.plan_id
            and event.payload.get("graph_scope_id") == plan.scope_id
        )

    generation_events = tuple(
        event
        for event in snapshot.events
        if event.attempt.generation.execution_id == intent.execution_id
        and event.attempt.generation.generation_id
        == intent.previous_generation_id
    )
    admissions = tuple(
        event
        for event in generation_events
        if event.event_type == "graph.execution.admitted"
        and event.attempt == plan.orchestration_attempt
    )
    if len(admissions) != 1 or _parse_graph_plan_admission(admissions[0]) != plan:
        raise GraphCheckpointError(
            "generation rebase recovery plan admission changed identity"
        )
    try:
        initial_input = _event_at_graph_cursor(
            generation_events,
            plan.initial_input_cursor,
        )
    except ValueError as error:
        raise GraphCheckpointError(
            "generation rebase recovery initial input is unavailable"
        ) from error
    if (
        initial_input.attempt != plan.orchestration_attempt
        or initial_input.event_type not in {"message.user", "interaction.resolved"}
        or initial_input.store_seq >= admissions[0].store_seq
    ):
        raise GraphCheckpointError(
            "generation rebase recovery initial input changed provenance"
        )
    _verified_graph_input_event_artifacts(
        initial_input,
        artifact_repository=journal,
        plan=plan,
        step_index=None,
        reason=GenerationRebaseFailureReason.GRAPH_PLAN_DESCRIPTOR_INVALID,
    )

    step_states: dict[int, _GraphStepQuiescence] = {}
    for candidate_step in plan.steps:
        candidate_events = tuple(
            event
            for event in generation_events
            if event.attempt == candidate_step.attempt
        )
        if not candidate_events:
            continue
        step_states[candidate_step.index] = _classify_graph_step_attempt(
            candidate_events,
            journal_events=generation_events,
            plan=plan,
            expected_step=candidate_step,
            artifact_repository=journal,
        )

    before_targets = target_events(snapshot.events)
    if len(before_targets) > 1:
        raise GraphCheckpointError(
            "generation rebase recovery target is ambiguous"
        )

    if step is not None:
        state = step_states.get(step.index)
        if state is None or state.terminal != evidence_event:
            raise GraphCheckpointError(
                "generation rebase recovery step evidence is no longer current"
            )
        if state.seal is not None:
            if before_targets != (state.seal,):
                raise GraphCheckpointError(
                    "generation rebase recovery step seal changed identity"
                )
            unchanged_before = _recovery_authority_snapshot(
                service,
                intent.execution_id,
            )
            unchanged_after = _recovery_authority_snapshot(
                service,
                intent.execution_id,
            )
            if unchanged_after != unchanged_before:
                raise GraphCheckpointError(
                    "generation rebase stale recovery observed a concurrent write"
                )
            return GenerationRebaseRecoveryResult(
                action="unchanged",
                reason=reason.value,
                execution_id=intent.execution_id,
                generation_id=intent.previous_generation_id,
                appended_event_count=0,
                artifact_count=0,
            )
        if state.status != "recovery_required" or before_targets:
            raise GraphCheckpointError(
                "generation rebase recovery step is not missing one exact seal"
            )
    else:
        if set(step_states) != set(range(len(plan.steps))) or any(
            state.status != "completed" or state.seal is None
            for state in step_states.values()
        ):
            raise GraphCheckpointError(
                "generation rebase execution recovery has an incomplete step"
            )
        if before_targets:
            execution_seal = before_targets[0]
            _validate_graph_execution_seal(
                execution_seal,
                plan=plan,
                final_step=step_states[len(plan.steps) - 1],
                artifact_repository=journal,
            )
            unchanged_before = _recovery_authority_snapshot(
                service,
                intent.execution_id,
            )
            unchanged_after = _recovery_authority_snapshot(
                service,
                intent.execution_id,
            )
            if unchanged_after != unchanged_before:
                raise GraphCheckpointError(
                    "generation rebase stale recovery observed a concurrent write"
                )
            return GenerationRebaseRecoveryResult(
                action="unchanged",
                reason=reason.value,
                execution_id=intent.execution_id,
                generation_id=intent.previous_generation_id,
                appended_event_count=0,
                artifact_count=0,
            )

    before_authority = _recovery_authority_snapshot(
        service,
        intent.execution_id,
    )
    checkpoints = GraphCheckpointService(
        repository=JournalGraphCheckpointRepository(journal),
        artifacts=ArtifactService(
            journal,
            sanitizer=artifact_sanitizer,
        ),
        derived_ingress_resolver=lambda _consumer, _source: (
            _raise_recovery_ingress_unavailable()
        ),
    )
    if reason is GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING:
        if step is None:
            raise TypeError("graph step recovery requires one step")
        scan = checkpoints.repository.scan(plan)
        terminal = checkpoints._terminal_after_start(scan, step.index)
        if terminal != evidence_event:
            raise GraphCheckpointError(
                "generation rebase recovery terminal changed before sealing"
            )
        terminal_cursor = EventCursor(terminal.store_seq, terminal.event_id)
        if terminal.event_type in GRAPH_STEP_SEAL_TERMINALS[GRAPH_STEP_COMPLETED]:
            output = checkpoints._completed_output(scan, step, terminal)
            checkpoints._seal_completed_terminal(
                plan,
                step,
                terminal,
                scan.snapshot_events,
                output,
            )
        elif terminal.event_type in GRAPH_STEP_SEAL_TERMINALS["graph.step.failed"]:
            checkpoints.repository.terminal(
                plan,
                step,
                status=GraphTerminalStatus.FAILED,
                terminal_cursor=terminal_cursor,
            )
        elif terminal.event_type in GRAPH_STEP_SEAL_TERMINALS[
            "graph.step.cancelled"
        ]:
            checkpoints.repository.terminal(
                plan,
                step,
                status=GraphTerminalStatus.CANCELLED,
                terminal_cursor=terminal_cursor,
            )
        else:
            raise GraphCheckpointError(
                "generation rebase recovery terminal family changed"
            )
        requested_action = "step_recovered"
    else:
        scan = checkpoints.repository.scan(plan)
        if (
            scan.recovery.terminal_status is not None
            or len(scan.recovery.completed_steps) != len(plan.steps)
        ):
            raise GraphCheckpointError(
                "generation rebase execution recovery is not exactly finalizable"
            )
        checkpoints.repository.finalize(
            plan,
            scan.recovery.completed_steps[-1],
        )
        requested_action = "execution_finalized"

    after_snapshot = journal.capture_snapshot()
    after_targets = target_events(after_snapshot.events)
    if len(after_targets) != 1:
        raise GraphCheckpointError(
            "generation rebase recovery did not converge to one seal"
        )
    after_generation_events = tuple(
        event
        for event in after_snapshot.events
        if event.attempt.generation.execution_id == intent.execution_id
        and event.attempt.generation.generation_id
        == intent.previous_generation_id
    )
    if step is not None:
        after_step_events = tuple(
            event
            for event in after_generation_events
            if event.attempt == step.attempt
        )
        after_state = _classify_graph_step_attempt(
            after_step_events,
            journal_events=after_generation_events,
            plan=plan,
            expected_step=step,
            artifact_repository=journal,
        )
        if after_state.terminal != evidence_event or after_state.seal != after_targets[0]:
            raise GraphCheckpointError(
                "generation rebase recovery sealed a changed step"
            )
    else:
        _validate_graph_execution_seal(
            after_targets[0],
            plan=plan,
            final_step=step_states[len(plan.steps) - 1],
            artifact_repository=journal,
        )
    after_authority = _recovery_authority_snapshot(
        service,
        intent.execution_id,
    )
    appended_event_count, artifact_count = _verify_recovery_authority_delta(
        before=before_authority,
        after=after_authority,
        target=after_targets[0],
    )
    if appended_event_count != 1:
        raise GraphCheckpointError(
            "generation rebase recovery did not append exactly one target seal"
        )
    return GenerationRebaseRecoveryResult(
        action=requested_action,
        reason=reason.value,
        execution_id=intent.execution_id,
        generation_id=intent.previous_generation_id,
        appended_event_count=appended_event_count,
        artifact_count=artifact_count,
    )


def _raise_recovery_ingress_unavailable() -> None:
    raise GraphCheckpointError(
        "generation rebase recovery cannot create derived ingress"
    )


__all__ = [
    "GenerationRebaseConflict",
    "GenerationRebaseError",
    "GenerationRebaseFailureReason",
    "GenerationRebaseHead",
    "GenerationRebaseIntent",
    "GenerationRebaseJournalIncompatible",
    "GenerationRebaseKind",
    "GenerationRebasePreflight",
    "GenerationRebasePreflightBlocked",
    "GenerationRebaseRecoveryRequired",
    "GenerationRebaseRecoveryResult",
    "GenerationRebaseReceipt",
    "GenerationRebaseRequest",
    "GenerationRebaseUnavailable",
    "GenerationSnapshotMessage",
    "GenerationTaskStateDescriptor",
    "SQLiteGenerationRebaseV2Service",
    "build_generation_rebase_operation",
    "generation_rebase_failure_detail",
    "recover_generation_rebase_attempt",
]
