from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from unchain.journal.models import (
    AttemptRef,
    ContextBuildStatus,
    EventRange,
    ModelValidationError,
    ResourceRef,
    _bounded_int,
    _freeze_json,
    _record_data,
    _record_tuple,
    _required_text,
    _sha256,
    _thaw_json,
)
from unchain.journal.resource_limits import (
    JsonResourceLimits,
    enforce_item_limit,
    validate_json_resource,
)


def _text_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an array")
    return tuple(
        _required_text(item, f"{field_name}[{index}]", maximum=16_384)
        for index, item in enumerate(value)
    )


def _mapping_tuple(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an array")
    records: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        frozen = _freeze_json(item, path=f"{field_name}[{index}]")
        if not isinstance(frozen, Mapping):
            raise TypeError(f"{field_name}[{index}] must be an object")
        records.append(frozen)
    return tuple(records)


def _int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an array")
    return tuple(
        _bounded_int(item, f"{field_name}[{index}]", minimum=1)
        for index, item in enumerate(value)
    )


def _identifier_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an array")
    identifiers: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or item != item.strip():
            raise ModelValidationError(f"{field_name}[{index}] is invalid")
        identifiers.append(
            _required_text(
                item,
                f"{field_name}[{index}]",
                identifier=True,
            )
        )
    return tuple(identifiers)


MAX_PINNED_TASK_STATE_ITEMS = 256
MAX_PINNED_TASK_STATE_BYTES = 256 * 1024
CONTEXT_COMPILE_REQUEST_LIMITS = JsonResourceLimits(
    max_items=10_000,
    max_bytes=32 * 1024 * 1024,
    max_depth=64,
    max_nodes=1_000_000,
)


class HandoffStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ContextBudget:
    SCHEMA: ClassVar[str] = "unchain.context_budget.v1"

    context_window_tokens: int
    output_reserve_tokens: int
    transport_margin_tokens: int
    available_input_tokens: int
    pressure_threshold_tokens: int

    def __post_init__(self) -> None:
        for field_name in (
            "context_window_tokens",
            "output_reserve_tokens",
            "transport_margin_tokens",
            "available_input_tokens",
            "pressure_threshold_tokens",
        ):
            minimum = 1 if field_name == "context_window_tokens" else 0
            object.__setattr__(
                self,
                field_name,
                _bounded_int(getattr(self, field_name), field_name, minimum=minimum),
            )
        expected_available = (
            self.context_window_tokens
            - self.output_reserve_tokens
            - self.transport_margin_tokens
        )
        if expected_available <= 0 or self.available_input_tokens != expected_available:
            raise ModelValidationError("available_input_tokens must equal window minus reserves")
        if not 0 < self.pressure_threshold_tokens <= self.available_input_tokens:
            raise ModelValidationError("pressure_threshold_tokens exceeds available input")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "context_window_tokens": self.context_window_tokens,
            "output_reserve_tokens": self.output_reserve_tokens,
            "transport_margin_tokens": self.transport_margin_tokens,
            "available_input_tokens": self.available_input_tokens,
            "pressure_threshold_tokens": self.pressure_threshold_tokens,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextBudget:
        fields = frozenset(
            {
                "context_window_tokens",
                "output_reserve_tokens",
                "transport_margin_tokens",
                "available_input_tokens",
                "pressure_threshold_tokens",
            }
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(**{field_name: raw[field_name] for field_name in fields})


@dataclass(frozen=True)
class ContextBuildEnvelope:
    SCHEMA: ClassVar[str] = "unchain.context_build_envelope.v1"

    build_id: str
    execution_id: str
    generation_id: str
    attempt_id: str
    provider: str
    model: str
    budget: ContextBudget
    source_range: EventRange | None = None
    included_ranges: tuple[EventRange, ...] = ()
    transformed_ranges: tuple[EventRange, ...] = ()
    checkpoint_refs: tuple[ResourceRef, ...] = ()
    artifact_refs: tuple[ResourceRef, ...] = ()
    estimated_input_tokens: int = 0
    status: ContextBuildStatus = ContextBuildStatus.COMPLETE

    def __post_init__(self) -> None:
        for field_name in ("build_id", "execution_id", "generation_id", "attempt_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model", _required_text(self.model, "model"))
        if not isinstance(self.budget, ContextBudget):
            object.__setattr__(self, "budget", ContextBudget.from_dict(self.budget))
        if self.source_range is not None and not isinstance(self.source_range, EventRange):
            object.__setattr__(self, "source_range", EventRange.from_dict(self.source_range))
        object.__setattr__(
            self, "included_ranges", _record_tuple(self.included_ranges, EventRange, "included_ranges")
        )
        object.__setattr__(
            self,
            "transformed_ranges",
            _record_tuple(self.transformed_ranges, EventRange, "transformed_ranges"),
        )
        object.__setattr__(
            self, "checkpoint_refs", _record_tuple(self.checkpoint_refs, ResourceRef, "checkpoint_refs")
        )
        object.__setattr__(
            self, "artifact_refs", _record_tuple(self.artifact_refs, ResourceRef, "artifact_refs")
        )
        object.__setattr__(
            self,
            "estimated_input_tokens",
            _bounded_int(self.estimated_input_tokens, "estimated_input_tokens"),
        )
        try:
            object.__setattr__(self, "status", ContextBuildStatus(self.status))
        except ValueError as exc:
            raise ModelValidationError("invalid context build status") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "build_id": self.build_id,
            "execution_id": self.execution_id,
            "generation_id": self.generation_id,
            "attempt_id": self.attempt_id,
            "provider": self.provider,
            "model": self.model,
            "budget": self.budget.to_dict(),
            "source_range": self.source_range.to_dict() if self.source_range else None,
            "included_ranges": [item.to_dict() for item in self.included_ranges],
            "transformed_ranges": [item.to_dict() for item in self.transformed_ranges],
            "checkpoint_refs": [item.to_dict() for item in self.checkpoint_refs],
            "artifact_refs": [item.to_dict() for item in self.artifact_refs],
            "estimated_input_tokens": self.estimated_input_tokens,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextBuildEnvelope:
        fields = frozenset(
            {
                "build_id",
                "execution_id",
                "generation_id",
                "attempt_id",
                "provider",
                "model",
                "budget",
                "source_range",
                "included_ranges",
                "transformed_ranges",
                "checkpoint_refs",
                "artifact_refs",
                "estimated_input_tokens",
                "status",
            }
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            build_id=raw["build_id"],
            execution_id=raw["execution_id"],
            generation_id=raw["generation_id"],
            attempt_id=raw["attempt_id"],
            provider=raw["provider"],
            model=raw["model"],
            budget=ContextBudget.from_dict(raw["budget"]),
            source_range=(
                EventRange.from_dict(raw["source_range"])
                if raw["source_range"] is not None
                else None
            ),
            included_ranges=_record_tuple(raw["included_ranges"], EventRange, "included_ranges"),
            transformed_ranges=_record_tuple(
                raw["transformed_ranges"], EventRange, "transformed_ranges"
            ),
            checkpoint_refs=_record_tuple(raw["checkpoint_refs"], ResourceRef, "checkpoint_refs"),
            artifact_refs=_record_tuple(raw["artifact_refs"], ResourceRef, "artifact_refs"),
            estimated_input_tokens=raw["estimated_input_tokens"],
            status=raw["status"],
        )


@dataclass(frozen=True)
class HandoffEnvelope:
    SCHEMA: ClassVar[str] = "unchain.handoff_envelope.v2"

    child_run_id: str
    child_attempt: AttemptRef
    status: HandoffStatus
    summary: str
    full_output_ref: ResourceRef
    artifact_refs: tuple[ResourceRef, ...]
    source_event_range: EventRange
    byte_length: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "child_run_id",
            _required_text(self.child_run_id, "child_run_id", identifier=True),
        )
        if not isinstance(self.child_attempt, AttemptRef):
            object.__setattr__(
                self,
                "child_attempt",
                AttemptRef.from_dict(self.child_attempt),
            )
        if self.child_attempt.attempt_id != self.child_run_id:
            raise ModelValidationError(
                "child_run_id must match child_attempt.attempt_id"
            )
        try:
            object.__setattr__(self, "status", HandoffStatus(self.status))
        except ValueError as exc:
            raise ModelValidationError("invalid handoff status") from exc
        object.__setattr__(self, "summary", _required_text(self.summary, "summary", maximum=16_384))
        if not isinstance(self.full_output_ref, ResourceRef):
            object.__setattr__(
                self, "full_output_ref", ResourceRef.from_dict(self.full_output_ref)
            )
        object.__setattr__(
            self, "artifact_refs", _record_tuple(self.artifact_refs, ResourceRef, "artifact_refs")
        )
        if not isinstance(self.source_event_range, EventRange):
            object.__setattr__(
                self,
                "source_event_range",
                EventRange.from_dict(self.source_event_range),
            )
        object.__setattr__(self, "byte_length", _bounded_int(self.byte_length, "byte_length"))
        object.__setattr__(self, "sha256", _sha256(self.sha256))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "child_run_id": self.child_run_id,
            "child_attempt": self.child_attempt.to_dict(),
            "status": self.status.value,
            "summary": self.summary,
            "full_output_ref": self.full_output_ref.to_dict(),
            "artifact_refs": [item.to_dict() for item in self.artifact_refs],
            "source_event_range": self.source_event_range.to_dict(),
            "byte_length": self.byte_length,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HandoffEnvelope:
        fields = frozenset(
            {
                "child_run_id",
                "child_attempt",
                "status",
                "summary",
                "full_output_ref",
                "artifact_refs",
                "source_event_range",
                "byte_length",
                "sha256",
            }
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            child_run_id=raw["child_run_id"],
            child_attempt=AttemptRef.from_dict(raw["child_attempt"]),
            status=raw["status"],
            summary=raw["summary"],
            full_output_ref=ResourceRef.from_dict(raw["full_output_ref"]),
            artifact_refs=_record_tuple(raw["artifact_refs"], ResourceRef, "artifact_refs"),
            source_event_range=EventRange.from_dict(raw["source_event_range"]),
            byte_length=raw["byte_length"],
            sha256=raw["sha256"],
        )


@dataclass(frozen=True)
class PinnedTaskState:
    SCHEMA: ClassVar[str] = "unchain.pinned_task_state.v1"

    state_id: str
    revision: int
    objective: str
    success_criteria: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    confirmed_decisions: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    active_plan: tuple[str, ...] = ()
    artifact_refs: tuple[ResourceRef, ...] = ()
    memory_refs: tuple[ResourceRef, ...] = ()
    source_event_refs: tuple[ResourceRef, ...] = ()
    status: str = "in_progress"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "state_id", _required_text(self.state_id, "state_id", identifier=True)
        )
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))
        object.__setattr__(
            self, "objective", _required_text(self.objective, "objective", maximum=32_768)
        )
        for field_name in (
            "success_criteria",
            "constraints",
            "confirmed_decisions",
            "open_questions",
            "active_plan",
        ):
            object.__setattr__(self, field_name, _text_tuple(getattr(self, field_name), field_name))
        for field_name in ("artifact_refs", "memory_refs", "source_event_refs"):
            object.__setattr__(
                self,
                field_name,
                _record_tuple(getattr(self, field_name), ResourceRef, field_name),
            )
        object.__setattr__(
            self, "status", _required_text(self.status, "status", identifier=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "state_id": self.state_id,
            "revision": self.revision,
            "objective": self.objective,
            "success_criteria": list(self.success_criteria),
            "constraints": list(self.constraints),
            "confirmed_decisions": list(self.confirmed_decisions),
            "open_questions": list(self.open_questions),
            "active_plan": list(self.active_plan),
            "artifact_refs": [item.to_dict() for item in self.artifact_refs],
            "memory_refs": [item.to_dict() for item in self.memory_refs],
            "source_event_refs": [item.to_dict() for item in self.source_event_refs],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PinnedTaskState:
        fields = frozenset(
            {
                "state_id",
                "revision",
                "objective",
                "success_criteria",
                "constraints",
                "confirmed_decisions",
                "open_questions",
                "active_plan",
                "artifact_refs",
                "memory_refs",
                "source_event_refs",
                "status",
            }
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            state_id=raw["state_id"],
            revision=raw["revision"],
            objective=raw["objective"],
            success_criteria=_text_tuple(raw["success_criteria"], "success_criteria"),
            constraints=_text_tuple(raw["constraints"], "constraints"),
            confirmed_decisions=_text_tuple(raw["confirmed_decisions"], "confirmed_decisions"),
            open_questions=_text_tuple(raw["open_questions"], "open_questions"),
            active_plan=_text_tuple(raw["active_plan"], "active_plan"),
            artifact_refs=_record_tuple(raw["artifact_refs"], ResourceRef, "artifact_refs"),
            memory_refs=_record_tuple(raw["memory_refs"], ResourceRef, "memory_refs"),
            source_event_refs=_record_tuple(
                raw["source_event_refs"], ResourceRef, "source_event_refs"
            ),
            status=raw["status"],
        )


@dataclass(frozen=True)
class ContextTaskStateUnavailable:
    """Content-free audit marker for task state unsafe to compile."""

    SCHEMA: ClassVar[str] = "unchain.context_task_state_unavailable.v1"

    state_ref: ResourceRef
    item_count: int
    content_bytes: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.state_ref, ResourceRef):
            object.__setattr__(
                self,
                "state_ref",
                ResourceRef.from_dict(self.state_ref),
            )
        if self.state_ref.kind != "task_state" or self.state_ref.fragment:
            raise ModelValidationError(
                "state_ref must be an unfragmented task-state reference"
            )
        object.__setattr__(
            self,
            "item_count",
            _bounded_int(self.item_count, "item_count"),
        )
        object.__setattr__(
            self,
            "content_bytes",
            _bounded_int(self.content_bytes, "content_bytes", minimum=1),
        )
        object.__setattr__(
            self,
            "reason",
            _required_text(self.reason, "reason", identifier=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "state_ref": self.state_ref.to_dict(),
            "item_count": self.item_count,
            "content_bytes": self.content_bytes,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> ContextTaskStateUnavailable:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset(
                {"state_ref", "item_count", "content_bytes", "reason"}
            ),
        )
        return cls(
            state_ref=ResourceRef.from_dict(raw["state_ref"]),
            item_count=raw["item_count"],
            content_bytes=raw["content_bytes"],
            reason=raw["reason"],
        )


@dataclass(frozen=True)
class OpaqueSecretHandle:
    SCHEMA: ClassVar[str] = "unchain.opaque_secret_handle.v1"

    handle_id: str
    label: str
    scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "handle_id", _required_text(self.handle_id, "handle_id", identifier=True)
        )
        object.__setattr__(self, "label", _required_text(self.label, "label", maximum=128))
        object.__setattr__(self, "scope", _required_text(self.scope, "scope", identifier=True))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "handle_id": self.handle_id,
            "label": self.label,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OpaqueSecretHandle:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"handle_id", "label", "scope"}),
        )
        return cls(raw["handle_id"], raw["label"], raw["scope"])


@dataclass(frozen=True)
class SourceMessageCursor:
    """Explicitly bind one compiler message to its durable journal cursor."""

    SCHEMA: ClassVar[str] = "unchain.source_message_cursor.v1"

    message_index: int
    event_id: str
    store_seq: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_index",
            _bounded_int(self.message_index, "message_index"),
        )
        object.__setattr__(
            self,
            "event_id",
            _required_text(self.event_id, "event_id", identifier=True),
        )
        object.__setattr__(
            self,
            "store_seq",
            _bounded_int(self.store_seq, "store_seq", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "message_index": self.message_index,
            "event_id": self.event_id,
            "store_seq": self.store_seq,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SourceMessageCursor:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"message_index", "event_id", "store_seq"}),
        )
        return cls(
            message_index=raw["message_index"],
            event_id=raw["event_id"],
            store_seq=raw["store_seq"],
        )


def _compile_request_resource_adapter(value: Any) -> Any:
    if isinstance(
        value,
        (
            ContextBudget,
            ContextTaskStateUnavailable,
            ResourceRef,
            SourceMessageCursor,
        ),
    ):
        return value.to_dict()
    return value


def _validate_compile_request_resources(request: ContextCompileRequest) -> None:
    limits = CONTEXT_COMPILE_REQUEST_LIMITS
    for field_name in (
        "source_messages",
        "semantic_events",
        "pending_task_inputs",
        "source_event_ids",
        "source_event_store_seqs",
        "source_message_cursors",
    ):
        value = getattr(request, field_name)
        if value is None or not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            continue
        enforce_item_limit(
            value,
            boundary=f"context_compile_request.{field_name}",
            limits=limits,
        )
    resource_record: dict[str, Any] = {
        "schema": request.SCHEMA,
        "case": request.case,
        "source_messages": request.source_messages,
    }
    optional_fields = (
        "current_generation",
        "fixed_overhead_tokens",
        "semantic_events",
        "task_state",
        "task_state_unavailable",
        "pending_task_inputs",
        "legacy_profile",
        "legacy_query",
        "capture_quality",
        "budget",
        "checkpoint_ref",
        "checkpoint_request_id",
        "source_event_ids",
        "source_event_store_seqs",
        "source_message_cursors",
        "provider",
        "model",
        "build_id",
        "execution_id",
        "generation_id",
        "attempt_id",
    )
    for field_name in optional_fields:
        value = getattr(request, field_name)
        if value is None or (
            field_name
            in {
                "source_event_ids",
                "source_event_store_seqs",
                "source_message_cursors",
            }
            and value == ()
        ):
            continue
        resource_record[field_name] = value
    validate_json_resource(
        resource_record,
        boundary="context_compile_request",
        limits=limits,
        record_adapter=_compile_request_resource_adapter,
    )


@dataclass(frozen=True)
class ContextCompileRequest:
    """Provider-neutral compiler input accepted at the Unchain boundary."""

    SCHEMA: ClassVar[str] = "unchain.context_v2.request.v1"

    case: str
    source_messages: tuple[Mapping[str, Any], ...]
    current_generation: str | None = None
    fixed_overhead_tokens: int | None = None
    semantic_events: tuple[Mapping[str, Any], ...] | None = None
    task_state: Mapping[str, Any] | None = None
    pending_task_inputs: tuple[Mapping[str, Any], ...] | None = None
    legacy_profile: Mapping[str, Any] | None = None
    legacy_query: str | None = None
    capture_quality: str | None = None
    budget: ContextBudget | None = None
    checkpoint_ref: ResourceRef | None = None
    checkpoint_request_id: str | None = None
    source_event_ids: tuple[str, ...] = ()
    source_event_store_seqs: tuple[int, ...] = ()
    source_message_cursors: tuple[SourceMessageCursor, ...] = ()
    provider: str | None = None
    model: str | None = None
    build_id: str | None = None
    execution_id: str | None = None
    generation_id: str | None = None
    attempt_id: str | None = None
    task_state_unavailable: ContextTaskStateUnavailable | None = None

    def __post_init__(self) -> None:
        _validate_compile_request_resources(self)
        object.__setattr__(self, "case", _required_text(self.case, "case", identifier=True))
        object.__setattr__(
            self, "source_messages", _mapping_tuple(self.source_messages, "source_messages")
        )
        if self.current_generation is not None:
            object.__setattr__(
                self,
                "current_generation",
                _required_text(self.current_generation, "current_generation", identifier=True),
            )
        if self.fixed_overhead_tokens is not None:
            object.__setattr__(
                self,
                "fixed_overhead_tokens",
                _bounded_int(self.fixed_overhead_tokens, "fixed_overhead_tokens"),
            )
        if self.semantic_events is not None:
            object.__setattr__(
                self, "semantic_events", _mapping_tuple(self.semantic_events, "semantic_events")
            )
        if self.task_state is not None:
            frozen = _freeze_json(self.task_state, path="task_state")
            if not isinstance(frozen, Mapping):
                raise TypeError("task_state must be an object")
            object.__setattr__(self, "task_state", frozen)
        if self.task_state_unavailable is not None:
            if not isinstance(
                self.task_state_unavailable,
                ContextTaskStateUnavailable,
            ):
                object.__setattr__(
                    self,
                    "task_state_unavailable",
                    ContextTaskStateUnavailable.from_dict(
                        self.task_state_unavailable
                    ),
                )
            if self.task_state is not None:
                raise ModelValidationError(
                    "task state content and unavailable marker are mutually exclusive"
                )
        if self.pending_task_inputs is not None:
            object.__setattr__(
                self,
                "pending_task_inputs",
                _mapping_tuple(self.pending_task_inputs, "pending_task_inputs"),
            )
        if self.legacy_profile is not None:
            frozen = _freeze_json(self.legacy_profile, path="legacy_profile")
            if not isinstance(frozen, Mapping):
                raise TypeError("legacy_profile must be an object")
            object.__setattr__(self, "legacy_profile", frozen)
        if self.legacy_query is not None:
            object.__setattr__(
                self,
                "legacy_query",
                _required_text(self.legacy_query, "legacy_query", maximum=4096),
            )
        if self.capture_quality is not None:
            try:
                quality = ContextBuildStatus(self.capture_quality)
            except ValueError as exc:
                raise ModelValidationError("invalid capture_quality") from exc
            object.__setattr__(self, "capture_quality", quality.value)
        if (
            self.task_state_unavailable is not None
            and self.capture_quality != ContextBuildStatus.UNAVAILABLE.value
        ):
            raise ModelValidationError(
                "unavailable task state requires unavailable capture quality"
            )
        if self.budget is not None and not isinstance(self.budget, ContextBudget):
            object.__setattr__(self, "budget", ContextBudget.from_dict(self.budget))
        if self.checkpoint_ref is not None and not isinstance(
            self.checkpoint_ref, ResourceRef
        ):
            object.__setattr__(
                self,
                "checkpoint_ref",
                ResourceRef.from_dict(self.checkpoint_ref),
            )
        if self.checkpoint_ref is not None and self.checkpoint_ref.kind != "checkpoint":
            raise ModelValidationError("checkpoint_ref must reference a checkpoint")
        if self.checkpoint_request_id is not None:
            object.__setattr__(
                self,
                "checkpoint_request_id",
                _required_text(
                    self.checkpoint_request_id,
                    "checkpoint_request_id",
                    identifier=True,
                ),
            )
        if (self.checkpoint_ref is None) != (self.checkpoint_request_id is None):
            raise ModelValidationError(
                "checkpoint_ref and checkpoint_request_id must be supplied together"
            )
        object.__setattr__(
            self,
            "source_event_ids",
            _identifier_tuple(self.source_event_ids, "source_event_ids"),
        )
        object.__setattr__(
            self,
            "source_event_store_seqs",
            _int_tuple(self.source_event_store_seqs, "source_event_store_seqs"),
        )
        if len(self.source_event_ids) != len(self.source_event_store_seqs):
            raise ModelValidationError(
                "source event identifiers and store sequences must be aligned"
            )
        if self.source_event_ids and len(self.source_event_ids) != len(
            self.source_messages
        ):
            raise ModelValidationError(
                "source event provenance requires one cursor per source message"
            )
        object.__setattr__(
            self,
            "source_message_cursors",
            _record_tuple(
                self.source_message_cursors,
                SourceMessageCursor,
                "source_message_cursors",
            ),
        )
        if self.source_message_cursors and self.source_event_ids:
            raise ModelValidationError(
                "source provenance must use either explicit cursors or legacy arrays"
            )
        if self.source_message_cursors:
            message_indexes = tuple(
                cursor.message_index for cursor in self.source_message_cursors
            )
            event_ids = tuple(cursor.event_id for cursor in self.source_message_cursors)
            store_seqs = tuple(cursor.store_seq for cursor in self.source_message_cursors)
            if any(index >= len(self.source_messages) for index in message_indexes):
                raise ModelValidationError(
                    "source message cursor index is outside source_messages"
                )
            if any(
                current <= previous
                for previous, current in zip(message_indexes, message_indexes[1:])
            ):
                raise ModelValidationError(
                    "source message cursor indexes must be strictly increasing"
                )
            if len(set(event_ids)) != len(event_ids):
                raise ModelValidationError(
                    "source message cursor event identifiers must be unique"
                )
            if any(
                current <= previous
                for previous, current in zip(store_seqs, store_seqs[1:])
            ):
                raise ModelValidationError(
                    "source message cursor store sequences must be strictly increasing"
                )
        for field_name in ("provider", "model"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _required_text(value, field_name, maximum=512),
                )
        identity_fields = ("build_id", "execution_id", "generation_id", "attempt_id")
        present_identity = [
            field_name
            for field_name in identity_fields
            if getattr(self, field_name) is not None
        ]
        if present_identity and len(present_identity) != len(identity_fields):
            raise ModelValidationError(
                "build identity fields must be supplied together"
            )
        for field_name in identity_fields:
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _required_text(value, field_name, identifier=True),
                )
        if present_identity and (self.provider is None or self.model is None):
            raise ModelValidationError(
                "provider and model are required with build identity"
            )
        if present_identity and self.budget is None:
            raise ModelValidationError("budget is required with build identity")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.SCHEMA,
            "case": self.case,
            "source_messages": [_thaw_json(item) for item in self.source_messages],
        }
        optional = (
            "current_generation",
            "fixed_overhead_tokens",
            "semantic_events",
            "task_state",
            "task_state_unavailable",
            "pending_task_inputs",
            "legacy_profile",
            "legacy_query",
            "capture_quality",
            "budget",
            "checkpoint_ref",
            "checkpoint_request_id",
            "source_event_ids",
            "source_event_store_seqs",
            "source_message_cursors",
            "provider",
            "model",
            "build_id",
            "execution_id",
            "generation_id",
            "attempt_id",
        )
        for field_name in optional:
            value = getattr(self, field_name)
            if value is None:
                continue
            if field_name in {
                "source_event_ids",
                "source_event_store_seqs",
                "source_message_cursors",
            } and value == ():
                continue
            if isinstance(value, (ContextBudget, ResourceRef)):
                result[field_name] = value.to_dict()
            elif field_name == "task_state_unavailable":
                result[field_name] = value.to_dict()
            elif field_name == "source_message_cursors":
                result[field_name] = [item.to_dict() for item in value]
            else:
                result[field_name] = _thaw_json(value)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ContextCompileRequest:
        optional = frozenset(
            {
                "current_generation",
                "fixed_overhead_tokens",
                "semantic_events",
                "task_state",
                "task_state_unavailable",
                "pending_task_inputs",
                "legacy_profile",
                "legacy_query",
                "capture_quality",
                "budget",
                "checkpoint_ref",
                "checkpoint_request_id",
                "source_event_ids",
                "source_event_store_seqs",
                "source_message_cursors",
                "provider",
                "model",
                "build_id",
                "execution_id",
                "generation_id",
                "attempt_id",
            }
        )
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"case", "source_messages"}),
            optional=optional,
        )
        return cls(
            case=raw["case"],
            source_messages=raw["source_messages"],
            **{field_name: raw[field_name] for field_name in optional if field_name in raw},
        )


__all__ = [
    "ContextBudget",
    "ContextBuildEnvelope",
    "ContextBuildStatus",
    "ContextCompileRequest",
    "ContextTaskStateUnavailable",
    "HandoffEnvelope",
    "HandoffStatus",
    "MAX_PINNED_TASK_STATE_BYTES",
    "MAX_PINNED_TASK_STATE_ITEMS",
    "OpaqueSecretHandle",
    "PinnedTaskState",
    "SourceMessageCursor",
]
