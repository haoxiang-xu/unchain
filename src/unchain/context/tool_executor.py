from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from unchain.execution import ExecutionFence, ExecutionGuard, ExecutionLease
from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    GenerationRef,
    ResourceRef,
    SideEffectRecoveryState,
)
from unchain.journal.models import (
    _freeze_json,
    _required_text,
    _sha256,
    _thaw_json,
)

from .artifacts import (
    ArtifactService,
    ToolCompletionArtifactization,
    ToolResultArtifactization,
    _bounded_utf8_preview,
)
from .tool_boundary import (
    DurableToolApprovalState,
    DurableToolAuthorization,
    DurableToolBoundary,
    DurableToolBoundaryCorruptError,
    DurableToolExecutionDisposition,
    DurableToolExecutionSubject,
    DurableToolResultReceipt,
    DurableToolRouteKind,
    _canonical_digest,
    _iteration,
)


_DURABLE_TOOL_INVOCATION_AUTHORITY = object()
_MAX_BOUND_TOOL_INVOCATIONS = 128


def _same_execution_guard_chain(
    left: tuple[tuple[Any, ...], ...],
    right: tuple[tuple[Any, ...], ...],
) -> bool:
    if len(left) != len(right):
        return False
    for left_node, right_node in zip(left, right, strict=True):
        if not left_node or not right_node or left_node[0] != right_node[0]:
            return False
        kind = left_node[0]
        if kind == "root":
            if len(left_node) != 4 or len(right_node) != 4:
                return False
            if any(
                left_value is not right_value
                for left_value, right_value in zip(
                    left_node[1:],
                    right_node[1:],
                    strict=True,
                )
            ):
                return False
        elif kind == "borrowed":
            if (
                len(left_node) != 3
                or len(right_node) != 3
                or left_node[1] is not right_node[1]
                or left_node[2] != right_node[2]
            ):
                return False
        elif kind == "batch":
            if (
                len(left_node) != 3
                or len(right_node) != 3
                or left_node[1] is not right_node[1]
                or left_node[2] is not right_node[2]
            ):
                return False
        else:
            return False
    return True


def _resolve_official_execution_guard(
    guard: ExecutionGuard,
) -> tuple[Any, ExecutionGuard, tuple[tuple[Any, ...], ...]]:
    """Return the projected lease and authentic root for an official chain."""

    from unchain.execution import _BorrowedExecutionGuard
    from unchain.subagents.plugin import _BatchExecutionGuard

    current = guard
    seen: set[int] = set()
    wrappers: list[tuple[str, Any, Any]] = []
    while type(current) is not ExecutionGuard:
        current_id = id(current)
        if current_id in seen:
            raise DurableToolExecutorContractError(
                "official execution guard chain is cyclic"
            )
        seen.add(current_id)
        if type(current) is _BorrowedExecutionGuard:
            wrappers.append(("borrowed", current, current._session_id))
            current = current._parent
            continue
        if type(current) is _BatchExecutionGuard:
            if type(current._failure_event) is not threading.Event:
                raise DurableToolExecutorContractError(
                    "batch execution guard has an invalid abort capability"
                )
            wrappers.append(("batch", current, current._failure_event))
            current = current._delegate
            continue
        raise DurableToolExecutorContractError(
            "durable tool execution requires an official execution guard"
        )

    if id(current) in seen:
        raise DurableToolExecutorContractError(
            "official execution guard chain is cyclic"
        )
    root = current
    lease = ExecutionGuard.assert_active(root)
    if type(lease) is not ExecutionLease:
        raise DurableToolExecutorContractError(
            "root execution guard returned an invalid lease"
        )
    root_execution_id = lease.execution_id

    for kind, wrapper, binding in reversed(wrappers):
        if kind == "borrowed":
            session_id = binding
            if not (
                session_id == root_execution_id
                or session_id.startswith(f"{root_execution_id}:")
            ):
                raise DurableToolExecutorContractError(
                    "borrowed execution guard is outside its root execution"
                )
            with wrapper._borrow_lock:
                _BorrowedExecutionGuard._ensure_active(wrapper)
                lease = ExecutionLease(
                    execution_id=session_id,
                    owner_id=lease.owner_id,
                    fencing_token=lease.fencing_token,
                    acquired_at_ms=lease.acquired_at_ms,
                    expires_at_ms=lease.expires_at_ms,
                )
            continue
        if threading.Event.is_set(binding):
            raise DurableToolExecutorContractError(
                "batch execution guard is aborted"
            )

    chain = tuple(wrappers) + (
        ("root", root, root._runtime, root._runtime.store),
    )
    return lease, root, chain


def _assert_official_execution_guard_active(
    guard: ExecutionGuard,
) -> ExecutionFence:
    lease, _root, _chain = _resolve_official_execution_guard(guard)
    return lease.fence


class DurableToolExecutorError(RuntimeError):
    """Base error for the Context V2 durable tool executor."""


class DurableToolExecutorContractError(DurableToolExecutorError):
    """A host invocation attempted to bypass the typed durable contract."""


class DurableToolInvocationFailedError(DurableToolExecutorError):
    """Secret-safe terminal wrapper for an arbitrary tool exception."""

    code = "durable_tool_invocation_failed"

    def __init__(self) -> None:
        self.__suppress_context__ = True
        super().__init__(self.code)


def _coerce_execution_subject(value: Any) -> DurableToolExecutionSubject:
    try:
        if type(value) is DurableToolExecutionSubject:
            parsed = value
        elif isinstance(value, Mapping):
            parsed = DurableToolExecutionSubject.from_dict(value)
        else:
            raise DurableToolExecutorContractError(
                "execution subject must be an exact subject or object"
            )
        text_fields = (
            parsed.original_arguments_sha256,
            parsed.effective_arguments_sha256,
            parsed.approval_request_sha256,
            parsed.approval_receipt_sha256,
            parsed.route_manifest_sha256,
            parsed.terminal_handler_manifest_sha256,
        )
        if (
            any(type(item) is not str for item in text_fields)
            or type(parsed.approval_state) is not DurableToolApprovalState
            or type(parsed.route_kind) is not DurableToolRouteKind
        ):
            raise DurableToolExecutorContractError(
                "execution subject contains non-exact scalar records"
            )
        return DurableToolExecutionSubject(
            intent_cursor=_canonical_event_cursor(
                parsed.intent_cursor,
                field_name="execution subject intent cursor",
            ),
            original_arguments_sha256=parsed.original_arguments_sha256,
            effective_arguments_sha256=parsed.effective_arguments_sha256,
            approval_state=parsed.approval_state,
            approval_request_sha256=parsed.approval_request_sha256,
            approval_receipt_sha256=parsed.approval_receipt_sha256,
            route_kind=parsed.route_kind,
            route_manifest_sha256=parsed.route_manifest_sha256,
            terminal_handler_manifest_sha256=(
                parsed.terminal_handler_manifest_sha256
            ),
            execution_fence=_canonical_execution_fence(
                parsed.execution_fence,
                field_name="execution subject fence",
            ),
        )
    except (DurableToolBoundaryCorruptError, TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "execution subject object is invalid"
        ) from exc


def _canonical_event_cursor(
    value: Any,
    *,
    field_name: str,
) -> EventCursor:
    if (
        type(value) is not EventCursor
        or type(value.store_seq) is not int
        or type(value.event_id) is not str
    ):
        raise DurableToolExecutorContractError(
            f"{field_name} must be an exact EventCursor snapshot"
        )
    try:
        return EventCursor(
            store_seq=value.store_seq,
            event_id=value.event_id,
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            f"{field_name} is invalid"
        ) from exc


def _canonical_execution_fence(
    value: Any,
    *,
    field_name: str,
) -> ExecutionFence:
    if (
        type(value) is not ExecutionFence
        or type(value.execution_id) is not str
        or type(value.owner_id) is not str
        or type(value.fencing_token) is not int
    ):
        raise DurableToolExecutorContractError(
            f"{field_name} must be an exact ExecutionFence snapshot"
        )
    try:
        return ExecutionFence(
            execution_id=value.execution_id,
            owner_id=value.owner_id,
            fencing_token=value.fencing_token,
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            f"{field_name} is invalid"
        ) from exc


def _canonical_generation_ref(value: Any) -> GenerationRef:
    if (
        type(value) is not GenerationRef
        or type(value.execution_id) is not str
        or type(value.generation_id) is not str
    ):
        raise DurableToolExecutorContractError(
            "generation reference must be an exact snapshot"
        )
    try:
        return GenerationRef(
            execution_id=value.execution_id,
            generation_id=value.generation_id,
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "generation reference is invalid"
        ) from exc


def _canonical_attempt_ref(value: Any) -> AttemptRef:
    if type(value) is not AttemptRef or type(value.attempt_id) is not str:
        raise DurableToolExecutorContractError(
            "attempt reference must be an exact snapshot"
        )
    try:
        return AttemptRef(
            generation=_canonical_generation_ref(value.generation),
            attempt_id=value.attempt_id,
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "attempt reference is invalid"
        ) from exc


def _canonical_resource_ref(value: Any) -> ResourceRef:
    if (
        type(value) is not ResourceRef
        or type(value.kind) is not str
        or type(value.resource_id) is not str
        or type(value.revision) is not int
        or type(value.fragment) is not str
    ):
        raise DurableToolExecutorContractError(
            "resource reference must be an exact snapshot"
        )
    try:
        return ResourceRef(
            kind=value.kind,
            resource_id=value.resource_id,
            revision=value.revision,
            fragment=value.fragment,
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "resource reference is invalid"
        ) from exc


def _canonical_artifact_ref(value: Any) -> ArtifactRef:
    if (
        type(value) is not ArtifactRef
        or type(value.media_type) is not str
        or type(value.byte_length) is not int
        or type(value.sha256) is not str
        or type(value.preview) is not str
    ):
        raise DurableToolExecutorContractError(
            "artifact reference must be an exact snapshot"
        )
    try:
        return ArtifactRef(
            ref=_canonical_resource_ref(value.ref),
            media_type=value.media_type,
            byte_length=value.byte_length,
            sha256=value.sha256,
            preview=value.preview,
        )
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "artifact reference is invalid"
        ) from exc


def _canonical_json_snapshot(value: Any, *, field_name: str) -> Any:
    try:
        encoded = json.dumps(
            _thaw_json(_freeze_json(value, path=field_name)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return _freeze_json(json.loads(encoded), path=field_name)
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            f"{field_name} is not canonical JSON"
        ) from exc


def _same_canonical_json(left: Any, right: Any) -> bool:
    left_snapshot = _canonical_json_snapshot(
        left,
        field_name="left JSON comparison value",
    )
    right_snapshot = _canonical_json_snapshot(
        right,
        field_name="right JSON comparison value",
    )
    return _canonical_digest(
        {"value": _thaw_json(left_snapshot)}
    ) == _canonical_digest({"value": _thaw_json(right_snapshot)})


def _canonical_authorization(
    value: Any,
    *,
    authority: object,
) -> DurableToolAuthorization:
    if (
        type(value) is not DurableToolAuthorization
        or value._authority is not authority
        or type(value.disposition) is not DurableToolExecutionDisposition
        or type(value.tool_name) is not str
        or type(value.call_id) is not str
        or type(value.iteration) is not int
    ):
        raise DurableToolExecutorContractError(
            "tool authorization is not runtime-owned"
        )
    try:
        return DurableToolAuthorization(
            disposition=value.disposition,
            tool_name=value.tool_name,
            call_id=value.call_id,
            iteration=value.iteration,
            execution_subject=_coerce_execution_subject(
                value.execution_subject
            ),
            started_cursor=_canonical_event_cursor(
                value.started_cursor,
                field_name="tool authorization started cursor",
            ),
            current_execution_fence=_canonical_execution_fence(
                value.current_execution_fence,
                field_name="tool authorization current fence",
            ),
            result_artifact=(
                None
                if value.result_artifact is None
                else _canonical_artifact_ref(value.result_artifact)
            ),
            completion_artifact=(
                None
                if value.completion_artifact is None
                else _canonical_artifact_ref(value.completion_artifact)
            ),
            result_cursor=(
                None
                if value.result_cursor is None
                else _canonical_event_cursor(
                    value.result_cursor,
                    field_name="tool authorization result cursor",
                )
            ),
            visible_result=_canonical_json_snapshot(
                value.visible_result,
                field_name="tool authorization visible result",
            ),
            _authority=authority,
        )
    except DurableToolExecutorContractError:
        raise
    except (DurableToolBoundaryCorruptError, TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "tool authorization contains invalid records"
        ) from exc


def _canonical_result_receipt(
    value: Any,
    *,
    authority: object,
) -> DurableToolResultReceipt:
    if (
        type(value) is not DurableToolResultReceipt
        or value._authority is not authority
        or type(value.tool_name) is not str
        or type(value.call_id) is not str
        or type(value.iteration) is not int
        or type(value.execution_subject_sha256) is not str
        or type(value.duplicate) is not bool
    ):
        raise DurableToolExecutorContractError(
            "durable result receipt is not runtime-owned"
        )
    try:
        return DurableToolResultReceipt(
            attempt=_canonical_attempt_ref(value.attempt),
            tool_name=value.tool_name,
            call_id=value.call_id,
            iteration=value.iteration,
            execution_subject=_coerce_execution_subject(
                value.execution_subject
            ),
            execution_subject_sha256=value.execution_subject_sha256,
            visible_result=_canonical_json_snapshot(
                value.visible_result,
                field_name="durable result receipt visible result",
            ),
            artifact=_canonical_artifact_ref(value.artifact),
            cursor=_canonical_event_cursor(
                value.cursor,
                field_name="durable result receipt cursor",
            ),
            model_projection=(
                None
                if value.model_projection is None
                else _canonical_json_snapshot(
                    value.model_projection,
                    field_name="durable result receipt model projection",
                )
            ),
            completion_artifact=(
                None
                if value.completion_artifact is None
                else _canonical_artifact_ref(value.completion_artifact)
            ),
            duplicate=value.duplicate,
            _authority=authority,
        )
    except DurableToolExecutorContractError:
        raise
    except (DurableToolBoundaryCorruptError, TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            "durable result receipt contains invalid records"
        ) from exc


@dataclass(frozen=True)
class DurableToolExecutionRequest:
    tool_name: str
    call_id: str
    iteration: int
    subject: DurableToolExecutionSubject
    tool_result_policy: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_name",
            _required_text(self.tool_name, "tool_name", identifier=True),
        )
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )
        object.__setattr__(self, "iteration", _iteration(self.iteration))
        object.__setattr__(
            self,
            "subject",
            _coerce_execution_subject(self.subject),
        )
        policy = self.tool_result_policy
        if not isinstance(policy, str) or not policy.strip():
            raise DurableToolExecutorContractError(
                "tool result policy must be a non-empty string"
            )
        object.__setattr__(self, "tool_result_policy", policy.strip().lower())


@dataclass(frozen=True)
class DurableToolInvocation:
    route_kind: DurableToolRouteKind
    route_manifest_sha256: str
    terminal_handler_manifest_sha256: str
    effective_arguments: Any
    terminal_handler: Any = field(repr=False, compare=False)
    _executor_authority: object = field(repr=False, compare=False)
    _authority: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._authority is not _DURABLE_TOOL_INVOCATION_AUTHORITY:
            raise DurableToolExecutorContractError(
                "durable tool invocation must be runtime-owned"
            )
        if not isinstance(self.route_kind, DurableToolRouteKind):
            object.__setattr__(
                self,
                "route_kind",
                DurableToolRouteKind(self.route_kind),
            )
        object.__setattr__(
            self,
            "route_manifest_sha256",
            _sha256(self.route_manifest_sha256, "route_manifest_sha256"),
        )
        object.__setattr__(
            self,
            "terminal_handler_manifest_sha256",
            _sha256(
                self.terminal_handler_manifest_sha256,
                "terminal_handler_manifest_sha256",
            ),
        )
        object.__setattr__(
            self,
            "effective_arguments",
            _canonical_json_snapshot(
                self.effective_arguments,
                field_name="tool invocation effective arguments",
            ),
        )
        if not callable(self.terminal_handler):
            raise TypeError("durable tool terminal handler must be callable")


@dataclass(frozen=True)
class _BoundDurableToolInvocation:
    invocation: DurableToolInvocation
    route_kind: DurableToolRouteKind
    route_manifest_sha256: str
    terminal_handler_manifest_sha256: str
    effective_arguments: Any
    terminal_handler: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class DurableToolCompletionDraft:
    result: Any
    should_observe: bool = True
    state_transition: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result",
            _canonical_json_snapshot(
                self.result,
                field_name="tool completion result",
            ),
        )
        if not isinstance(self.should_observe, bool):
            raise TypeError("should_observe must be a boolean")
        if self.state_transition is not None:
            from .tool_transitions import DurableToolStateTransitionDraft

            if type(self.state_transition) is not DurableToolStateTransitionDraft:
                raise DurableToolExecutorContractError(
                    "tool completion state transition must be runtime-owned"
                )


@dataclass(frozen=True)
class DurableToolCompletionEnvelope:
    SCHEMA: ClassVar[str] = "unchain.durable_tool_completion.v2"

    attempt: AttemptRef
    tool_name: str
    call_id: str
    iteration: int
    execution_subject: DurableToolExecutionSubject
    execution_subject_sha256: str
    result_artifact: ArtifactRef
    visible_result: Any
    should_observe: bool
    transition: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(
                self,
                "attempt",
                AttemptRef.from_dict(self.attempt),
            )
        object.__setattr__(
            self,
            "tool_name",
            _required_text(self.tool_name, "tool_name", identifier=True),
        )
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )
        object.__setattr__(self, "iteration", _iteration(self.iteration))
        object.__setattr__(
            self,
            "execution_subject",
            _coerce_execution_subject(self.execution_subject),
        )
        object.__setattr__(
            self,
            "execution_subject_sha256",
            _sha256(
                self.execution_subject_sha256,
                "execution_subject_sha256",
            ),
        )
        if self.execution_subject_sha256 != self.execution_subject.sha256:
            raise DurableToolExecutorContractError(
                "completion execution subject digest changed"
            )
        if not isinstance(self.result_artifact, ArtifactRef):
            object.__setattr__(
                self,
                "result_artifact",
                ArtifactRef.from_dict(self.result_artifact),
            )
        object.__setattr__(
            self,
            "visible_result",
            _freeze_json(self.visible_result, path="completion.visible_result"),
        )
        if not isinstance(self.should_observe, bool):
            raise TypeError("should_observe must be a boolean")
        if self.transition is not None:
            from .tool_transitions import DurableToolStateTransitionEnvelope

            if type(self.transition) is not DurableToolStateTransitionEnvelope:
                raise DurableToolExecutorContractError(
                    "tool completion transition must be runtime-owned"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.to_dict(),
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "iteration": self.iteration,
            "execution_subject": self.execution_subject.to_dict(),
            "execution_subject_sha256": self.execution_subject_sha256,
            "result_artifact": self.result_artifact.to_dict(),
            "visible_result": _thaw_json(self.visible_result),
            "should_observe": self.should_observe,
            "transition": (
                None if self.transition is None else self.transition.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Any) -> DurableToolCompletionEnvelope:
        if not isinstance(value, dict):
            try:
                value = dict(value)
            except (TypeError, ValueError) as exc:
                raise DurableToolExecutorContractError(
                    "tool completion must be an object"
                ) from exc
        expected = {
            "schema",
            "attempt",
            "tool_name",
            "call_id",
            "iteration",
            "execution_subject",
            "execution_subject_sha256",
            "result_artifact",
            "visible_result",
            "should_observe",
            "transition",
        }
        if set(value) != expected or value.get("schema") != cls.SCHEMA:
            raise DurableToolExecutorContractError(
                "tool completion schema is invalid"
            )
        transition = value["transition"]
        if transition is not None:
            from .tool_transitions import DurableToolStateTransitionEnvelope

            transition = DurableToolStateTransitionEnvelope.from_dict(
                transition
            )
        return cls(
            attempt=AttemptRef.from_dict(value["attempt"]),
            tool_name=value["tool_name"],
            call_id=value["call_id"],
            iteration=value["iteration"],
            execution_subject=DurableToolExecutionSubject.from_dict(
                value["execution_subject"]
            ),
            execution_subject_sha256=value["execution_subject_sha256"],
            result_artifact=ArtifactRef.from_dict(value["result_artifact"]),
            visible_result=value["visible_result"],
            should_observe=value["should_observe"],
            transition=transition,
        )


@dataclass(frozen=True)
class DurableToolCompletionReceipt:
    attempt: AttemptRef
    tool_name: str
    call_id: str
    iteration: int
    execution_subject: DurableToolExecutionSubject
    current_execution_fence: ExecutionFence
    visible_result: Any
    should_observe: bool
    result_artifact: ArtifactRef
    completion_artifact: ArtifactRef
    journal_cursor: EventCursor
    transition: Any = None
    reused: bool = False
    model_projection: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(
                self,
                "attempt",
                AttemptRef.from_dict(self.attempt),
            )
        object.__setattr__(
            self,
            "tool_name",
            _required_text(self.tool_name, "tool_name", identifier=True),
        )
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )
        object.__setattr__(self, "iteration", _iteration(self.iteration))
        object.__setattr__(
            self,
            "execution_subject",
            _coerce_execution_subject(self.execution_subject),
        )
        if not isinstance(self.current_execution_fence, ExecutionFence):
            raise TypeError("current_execution_fence must be an ExecutionFence")
        object.__setattr__(
            self,
            "visible_result",
            _freeze_json(self.visible_result, path="receipt.visible_result"),
        )
        if not isinstance(self.should_observe, bool):
            raise TypeError("should_observe must be a boolean")
        if not isinstance(self.result_artifact, ArtifactRef):
            object.__setattr__(
                self,
                "result_artifact",
                ArtifactRef.from_dict(self.result_artifact),
            )
        if not isinstance(self.completion_artifact, ArtifactRef):
            object.__setattr__(
                self,
                "completion_artifact",
                ArtifactRef.from_dict(self.completion_artifact),
            )
        if not isinstance(self.journal_cursor, EventCursor):
            object.__setattr__(
                self,
                "journal_cursor",
                EventCursor.from_dict(self.journal_cursor),
            )
        if self.model_projection is not None:
            if not isinstance(self.model_projection, Mapping) or set(
                self.model_projection
            ) != {"result", "metadata"}:
                raise DurableToolExecutorContractError(
                    "tool completion receipt model projection schema is invalid"
                )
            if not isinstance(self.model_projection["result"], Mapping) or not isinstance(
                self.model_projection["metadata"], Mapping
            ):
                raise DurableToolExecutorContractError(
                    "tool completion receipt model projection payload is invalid"
                )
            object.__setattr__(
                self,
                "model_projection",
                _freeze_json(
                    self.model_projection,
                    path="receipt.model_projection",
                ),
            )
        if self.transition is not None:
            from .tool_transitions import DurableToolStateTransitionEnvelope

            if type(self.transition) is not DurableToolStateTransitionEnvelope:
                raise DurableToolExecutorContractError(
                    "tool completion receipt transition must be runtime-owned"
                )
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be a boolean")


class DurableToolExecutor:
    """Own the only safe transition from a tool side effect to model visibility."""

    def __init__(
        self,
        *,
        boundary: DurableToolBoundary,
        artifacts: ArtifactService,
        execution_guard: ExecutionGuard,
        expected_guard_chain: tuple[tuple[Any, ...], ...] | None = None,
        bootstrap_binding: Any | None = None,
        completion_policy: Any | None = None,
    ) -> None:
        if type(boundary) is not DurableToolBoundary:
            raise TypeError("boundary must be the official DurableToolBoundary")
        if type(artifacts) is not ArtifactService:
            raise TypeError("artifacts must be the official ArtifactService")
        if completion_policy is not None and not callable(completion_policy):
            raise TypeError("completion_policy must be callable")
        if boundary.projector.artifacts is not artifacts:
            raise DurableToolExecutorContractError(
                "executor boundary and artifacts do not share one service"
            )
        lease, root_guard, guard_chain = _resolve_official_execution_guard(
            execution_guard
        )
        if bootstrap_binding is not None:
            from .runtime import (
                _BoundExecutionGuard,
                _CONTEXT_EXECUTION_BINDING_AUTHORITY,
            )

            if (
                type(bootstrap_binding) is not _BoundExecutionGuard
                or bootstrap_binding._authority
                is not _CONTEXT_EXECUTION_BINDING_AUTHORITY
                or bootstrap_binding.attempt != boundary.attempt
                or bootstrap_binding.guard is not execution_guard
                or bootstrap_binding.root_guard is not root_guard
                or bootstrap_binding.execution_id != lease.execution_id
                or bootstrap_binding.root_execution_id
                != root_guard.lease.execution_id
                or bootstrap_binding.owner_id != lease.owner_id
                or lease.fencing_token < bootstrap_binding.fencing_token
                or not _same_execution_guard_chain(
                    bootstrap_binding.chain, guard_chain
                )
            ):
                raise DurableToolExecutorContractError(
                    "execution guard does not match its bootstrap binding"
                )
        if lease.execution_id != boundary.attempt.generation.execution_id and (
            bootstrap_binding is None
            or root_guard.lease.execution_id
            != boundary.attempt.generation.execution_id
        ):
            raise DurableToolExecutorContractError(
                "execution guard does not match the durable attempt"
            )
        if expected_guard_chain is not None and not _same_execution_guard_chain(
            tuple(expected_guard_chain),
            guard_chain,
        ):
            raise DurableToolExecutorContractError(
                "execution guard chain changed after bootstrap binding"
            )
        if lease.execution_id != boundary.attempt.generation.execution_id:
            boundary._bind_projected_execution(bootstrap_binding)
        self._boundary = boundary
        self._artifacts = artifacts
        self._execution_guard = execution_guard
        self._root_execution_guard = root_guard
        self._execution_guard_chain = guard_chain
        self._completion_policy = completion_policy
        self._execution_lock = threading.RLock()
        self._invocation_authority = object()
        self._invocation_binding_lock = threading.Lock()
        self._invocation_bindings: dict[int, _BoundDurableToolInvocation] = {}

    @property
    def boundary(self) -> DurableToolBoundary:
        return self._boundary

    @property
    def artifacts(self) -> ArtifactService:
        return self._artifacts

    def bind_invocation(
        self,
        *,
        request: DurableToolExecutionRequest,
        effective_arguments: Any,
        terminal_handler: Any,
    ) -> DurableToolInvocation:
        """Freeze the only arguments the terminal handler may receive."""

        if type(request) is not DurableToolExecutionRequest:
            raise DurableToolExecutorContractError(
                "invocation binding requires a DurableToolExecutionRequest"
            )
        if type(request.subject) is not DurableToolExecutionSubject:
            raise DurableToolExecutorContractError(
                "invocation binding requires an exact execution subject"
            )
        if _coerce_execution_subject(request.subject) != request.subject:
            raise DurableToolExecutorContractError(
                "invocation binding execution subject is not canonical"
            )
        frozen_arguments = _canonical_json_snapshot(
            effective_arguments,
            field_name="tool invocation effective arguments",
        )
        if (
            _canonical_digest(_thaw_json(frozen_arguments))
            != request.subject.effective_arguments_sha256
        ):
            raise DurableToolExecutorContractError(
                "tool invocation effective arguments do not match its "
                "durable subject"
            )
        if not callable(terminal_handler):
            raise DurableToolExecutorContractError(
                "tool invocation terminal handler must be callable"
            )
        invocation = DurableToolInvocation(
            route_kind=request.subject.route_kind,
            route_manifest_sha256=request.subject.route_manifest_sha256,
            terminal_handler_manifest_sha256=(
                request.subject.terminal_handler_manifest_sha256
            ),
            effective_arguments=frozen_arguments,
            terminal_handler=terminal_handler,
            _executor_authority=self._invocation_authority,
            _authority=_DURABLE_TOOL_INVOCATION_AUTHORITY,
        )
        binding = _BoundDurableToolInvocation(
            invocation=invocation,
            route_kind=invocation.route_kind,
            route_manifest_sha256=invocation.route_manifest_sha256,
            terminal_handler_manifest_sha256=(
                invocation.terminal_handler_manifest_sha256
            ),
            effective_arguments=invocation.effective_arguments,
            terminal_handler=invocation.terminal_handler,
        )
        with self._invocation_binding_lock:
            if len(self._invocation_bindings) >= _MAX_BOUND_TOOL_INVOCATIONS:
                raise DurableToolExecutorContractError(
                    "durable tool invocation binding capacity exceeded"
                )
            self._invocation_bindings[id(invocation)] = binding
        return invocation

    def execute(
        self,
        *,
        request: DurableToolExecutionRequest,
        guard: ExecutionGuard | None = None,
        invocation: DurableToolInvocation | None,
    ) -> DurableToolCompletionReceipt:
        with self._execution_lock:
            return self._execute_once(
                request=request,
                guard=guard,
                invocation=invocation,
            )

    def _execute_once(
        self,
        *,
        request: DurableToolExecutionRequest,
        guard: ExecutionGuard | None = None,
        invocation: DurableToolInvocation | None,
    ) -> DurableToolCompletionReceipt:
        if type(request) is not DurableToolExecutionRequest:
            raise DurableToolExecutorContractError(
                "executor requires a DurableToolExecutionRequest"
            )
        if type(request.subject) is not DurableToolExecutionSubject:
            raise DurableToolExecutorContractError(
                "executor requires an exact execution subject"
            )
        if _coerce_execution_subject(request.subject) != request.subject:
            raise DurableToolExecutorContractError(
                "executor execution subject is not canonical"
            )
        if guard is not None and guard is not self._execution_guard:
            raise DurableToolExecutorContractError(
                "tool execution guard is not the attempt bootstrap binding"
            )
        initial_recovery = self._boundary.sink.recover_tool_side_effect(
            request.call_id
        )
        bound_invocation = None
        if initial_recovery.state is SideEffectRecoveryState.NOT_STARTED:
            if type(invocation) is not DurableToolInvocation:
                raise DurableToolExecutorContractError(
                    "fresh execution requires a DurableToolInvocation"
                )
            bound_invocation = self._validate_invocation(
                request,
                invocation,
            )
        current_fence = self._assert_live_guard(request)
        authorization = self._boundary.authorize_execution(
            tool_name=request.tool_name,
            call_id=request.call_id,
            iteration=request.iteration,
            subject=request.subject,
        )
        authorization = self._validate_authorization(
            request,
            authorization,
            current_fence=current_fence,
        )
        if authorization.disposition is DurableToolExecutionDisposition.REUSE:
            self._assert_live_guard(request)
            recovered = self._recover_completion(
                request=request,
                authorization=authorization,
                current_fence=current_fence,
            )
            self._assert_live_guard(request)
            return recovered
        if authorization.disposition is DurableToolExecutionDisposition.FINALIZE:
            self._assert_live_guard(request)
            finalized = self._finalize_sealed_completion(
                request=request,
                authorization=authorization,
                current_fence=current_fence,
            )
            self._assert_live_guard(request)
            return finalized

        if bound_invocation is None:
            raise DurableToolExecutorContractError(
                "fresh execution lost its runtime-owned invocation"
            )

        self._assert_live_guard(request)
        invocation_failed = False
        try:
            raw_draft = bound_invocation.terminal_handler(
                _thaw_json(bound_invocation.effective_arguments)
            )
        except Exception:
            invocation_failed = True
        if invocation_failed:
            raise DurableToolInvocationFailedError()
        if type(raw_draft) is not DurableToolCompletionDraft:
            raise DurableToolExecutorContractError(
                "tool invocation must return a typed completion draft"
            )
        if self._completion_policy is not None:
            raw_draft = self._completion_policy(
                request=request,
                raw_draft=raw_draft,
                effective_arguments=_thaw_json(bound_invocation.effective_arguments),
                artifacts=self._artifacts,
            )
            if type(raw_draft) is not DurableToolCompletionDraft:
                raise DurableToolExecutorContractError(
                    "tool completion policy must return a typed completion draft"
                )
        return self._persist_completed_draft(
            request=request, authorization=authorization, raw_draft=raw_draft,
            reused=False,
        )

    def _persist_completed_draft(self, *, request, authorization, raw_draft, reused):
        completion_subject = authorization.execution_subject
        self._assert_live_guard(request)
        result_artifactization = self._artifacts.artifactize_tool_result(
            _thaw_json(raw_draft.result),
            operation_id="artifact.tool-result." + completion_subject.sha256,
        )
        result_artifactization = (
            ArtifactService.verify_tool_result_artifactization(
                self._artifacts,
                result_artifactization,
            )
        )
        transition = None
        if raw_draft.state_transition is not None:
            from .tool_transitions import (
                DurableToolStateTransitionEnvelope,
                canonical_state_sha256,
            )

            transition_draft = raw_draft.state_transition
            next_state_artifact = self._artifacts.persist_exact_json(
                transition_draft.next_state,
                operation_id=(
                    "artifact.tool-next-state." + completion_subject.sha256
                ),
                operation_binding={
                    "kind": "durable_tool_next_state",
                    "attempt": self._boundary.attempt.to_dict(),
                    "call_id": request.call_id,
                    "execution_subject_sha256": completion_subject.sha256,
                },
            )
            transition = DurableToolStateTransitionEnvelope(
                parent_attempt=self._boundary.attempt,
                call_id=request.call_id,
                execution_subject_sha256=completion_subject.sha256,
                operation_id=(
                    "transition.subagent-completion."
                    + completion_subject.sha256
                ),
                kind=transition_draft.kind,
                base_state_sha256=canonical_state_sha256(
                    transition_draft.base_state
                ),
                next_state_artifact=next_state_artifact,
                handoff_refs=transition_draft.handoff_refs,
            )
            from .tool_transitions import load_verified_subagent_state

            load_verified_subagent_state(self._artifacts, transition)
        envelope = DurableToolCompletionEnvelope(
            attempt=self._boundary.attempt,
            tool_name=request.tool_name,
            call_id=request.call_id,
            iteration=request.iteration,
            execution_subject=completion_subject,
            execution_subject_sha256=completion_subject.sha256,
            result_artifact=result_artifactization.artifact,
            visible_result=result_artifactization.visible_result,
            should_observe=raw_draft.should_observe,
            transition=transition,
        )
        completion_artifactization = (
            self._artifacts.artifactize_tool_completion(
                envelope.to_dict(),
                operation_id=(
                    "artifact.tool-completion." + completion_subject.sha256
                ),
            )
        )
        completion_artifactization = (
            ArtifactService.verify_tool_completion_artifactization(
                self._artifacts,
                completion_artifactization,
            )
        )
        persisted_envelope = DurableToolCompletionEnvelope.from_dict(
            _thaw_json(completion_artifactization.completion)
        )
        if _canonical_digest(
            persisted_envelope.to_dict()
        ) != _canonical_digest(envelope.to_dict()):
            raise DurableToolExecutorContractError(
                "tool completion sanitizer changed protected semantics"
            )
        self._assert_live_guard(request)
        if (persisted_envelope.transition is not None
                and self._boundary.sink.recover_tool_side_effect(request.call_id).sealed_event is None):
            self._boundary.persist_sealed_completion(
                authorization,
                result_artifact=result_artifactization.artifact,
                completion_artifact=completion_artifactization.artifact,
                transition=persisted_envelope.transition,
            )
            self._assert_live_guard(request)
        result_kwargs = {
            "artifactization": result_artifactization,
            "completion_artifactization": completion_artifactization,
        }
        if request.tool_result_policy != "default":
            result_kwargs["tool_result_policy"] = request.tool_result_policy
        result_receipt = self._boundary.persist_prepared_result(
            authorization,
            **result_kwargs,
        )
        result_receipt = self._validate_result_receipt(
            request,
            authorization=authorization,
            receipt=result_receipt,
            result_artifactization=result_artifactization,
            completion_artifactization=completion_artifactization,
            envelope=persisted_envelope,
        )
        if (
            result_receipt.artifact != result_artifactization.artifact
            or result_receipt.completion_artifact
            != completion_artifactization.artifact
            or not _same_canonical_json(
                result_receipt.visible_result,
                persisted_envelope.visible_result,
            )
        ):
            raise DurableToolExecutorContractError(
                "durable result receipt changed the prepared completion"
            )
        current_fence = self._assert_live_guard(request)
        return self._receipt(
            envelope=persisted_envelope,
            completion_artifact=result_receipt.completion_artifact,
            cursor=result_receipt.cursor,
            model_projection=result_receipt.model_projection,
            current_fence=current_fence,
            reused=reused,
        )

    def _validate_invocation(
        self,
        request: DurableToolExecutionRequest,
        invocation: DurableToolInvocation,
    ) -> _BoundDurableToolInvocation:
        with self._invocation_binding_lock:
            binding = self._invocation_bindings.pop(id(invocation), None)
        if binding is None or binding.invocation is not invocation:
            raise DurableToolExecutorContractError(
                "tool invocation route is not this executor's runtime binding"
            )
        subject = request.subject
        if (
            invocation._executor_authority is not self._invocation_authority
            or invocation.route_kind is not binding.route_kind
            or invocation.route_manifest_sha256
            != binding.route_manifest_sha256
            or invocation.terminal_handler_manifest_sha256
            != binding.terminal_handler_manifest_sha256
            or invocation.effective_arguments != binding.effective_arguments
            or invocation.terminal_handler is not binding.terminal_handler
            or _canonical_digest(
                _thaw_json(binding.effective_arguments)
            )
            != subject.effective_arguments_sha256
            or
            binding.route_kind is not subject.route_kind
            or binding.route_manifest_sha256
            != subject.route_manifest_sha256
            or binding.terminal_handler_manifest_sha256
            != subject.terminal_handler_manifest_sha256
        ):
            raise DurableToolExecutorContractError(
                "tool invocation route does not match its durable subject"
            )
        return binding

    def _validate_authorization(
        self,
        request: DurableToolExecutionRequest,
        authorization: Any,
        *,
        current_fence: ExecutionFence,
    ) -> DurableToolAuthorization:
        try:
            canonical = _canonical_authorization(
                authorization,
                authority=self._boundary._authority,
            )
            trusted_fence = _canonical_execution_fence(
                current_fence,
                field_name="live execution fence",
            )
        except DurableToolExecutorContractError as exc:
            raise DurableToolExecutorContractError(
                "tool authorization is not this boundary's exact receipt"
            ) from exc
        if (
            canonical.tool_name != request.tool_name
            or canonical.call_id != request.call_id
            or canonical.iteration != request.iteration
            or canonical.current_execution_fence != trusted_fence
        ):
            raise DurableToolExecutorContractError(
                "tool authorization is not this boundary's exact receipt"
            )
        recovery = self._boundary.sink.recover_tool_side_effect(
            request.call_id
        )
        if (
            canonical.disposition
            is DurableToolExecutionDisposition.EXECUTE
        ):
            if (
                canonical.execution_subject != request.subject
                or recovery.state
                is not SideEffectRecoveryState.UNCERTAIN_AFTER_START
            ):
                raise DurableToolExecutorContractError(
                    "execute authorization is not backed by tool.started"
                )
            self._boundary._verify_intent(
                recovery.intent_event,
                tool_name=request.tool_name,
                call_id=request.call_id,
                iteration=request.iteration,
                subject=request.subject,
            )
            recorded_subject = self._boundary._verify_recovery_identity(
                recovery.started_event,
                tool_name=request.tool_name,
                call_id=request.call_id,
                iteration=request.iteration,
                subject=request.subject,
            )
            if (
                recorded_subject != request.subject
            ):
                raise DurableToolExecutorContractError(
                    "execute authorization changed its durable started receipt"
                )
            expected = DurableToolAuthorization(
                disposition=DurableToolExecutionDisposition.EXECUTE,
                tool_name=request.tool_name,
                call_id=request.call_id,
                iteration=request.iteration,
                execution_subject=request.subject,
                started_cursor=self._boundary._cursor(
                    recovery.started_event
                ),
                current_execution_fence=trusted_fence,
                _authority=self._boundary._authority,
            )
            expected = _canonical_authorization(
                expected,
                authority=self._boundary._authority,
            )
            if canonical != expected:
                raise DurableToolExecutorContractError(
                    "execute authorization changed its durable started receipt"
                )
            return expected
        if canonical.disposition is DurableToolExecutionDisposition.FINALIZE:
            if recovery.state is not SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE:
                raise DurableToolExecutorContractError(
                    "finalize authorization is not backed by a sealed completion"
                )
            expected = self._boundary._authorization_from_recovery(
                recovery,
                tool_name=request.tool_name,
                call_id=request.call_id,
                iteration=request.iteration,
                subject=request.subject,
            )
            expected = _canonical_authorization(
                expected,
                authority=self._boundary._authority,
            )
            if canonical != expected or not _same_canonical_json(
                canonical.visible_result,
                expected.visible_result,
            ):
                raise DurableToolExecutorContractError(
                    "finalize authorization changed its sealed completion"
                )
            return expected
        if (
            canonical.disposition
            is not DurableToolExecutionDisposition.REUSE
            or recovery.state
            is not SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE
        ):
            raise DurableToolExecutorContractError(
                "tool authorization has an invalid disposition"
            )
        expected = self._boundary._authorization_from_recovery(
            recovery,
            tool_name=request.tool_name,
            call_id=request.call_id,
            iteration=request.iteration,
            subject=request.subject,
        )
        expected = _canonical_authorization(
            expected,
            authority=self._boundary._authority,
        )
        if canonical != expected or not _same_canonical_json(
            canonical.visible_result,
            expected.visible_result,
        ):
            raise DurableToolExecutorContractError(
                "reuse authorization changed its durable terminal receipt"
            )
        return expected

    def _validate_result_receipt(
        self,
        request: DurableToolExecutionRequest,
        *,
        authorization: DurableToolAuthorization,
        receipt: Any,
        result_artifactization: Any,
        completion_artifactization: ToolCompletionArtifactization,
        envelope: DurableToolCompletionEnvelope,
    ) -> DurableToolResultReceipt:
        try:
            canonical = _canonical_result_receipt(
                receipt,
                authority=self._boundary._authority,
            )
            expected_attempt = _canonical_attempt_ref(
                self._boundary.attempt
            )
            prepared_result_artifact = _canonical_artifact_ref(
                result_artifactization.artifact
            )
            prepared_completion_artifact = _canonical_artifact_ref(
                completion_artifactization.artifact
            )
            prepared_visible_result = _canonical_json_snapshot(
                envelope.visible_result,
                field_name="prepared tool result visible result",
            )
        except DurableToolExecutorContractError as exc:
            raise DurableToolExecutorContractError(
                "durable result receipt is not runtime-owned"
            ) from exc
        recovery = self._boundary.sink.recover_tool_side_effect(
            request.call_id
        )
        if recovery.state is not SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE:
            raise DurableToolExecutorContractError(
                "durable result receipt has no terminal journal event"
            )
        expected = self._boundary._authorization_from_recovery(
            recovery,
            tool_name=request.tool_name,
            call_id=request.call_id,
            iteration=request.iteration,
            subject=request.subject,
        )
        expected = _canonical_authorization(
            expected,
            authority=self._boundary._authority,
        )
        if (
            canonical.attempt != expected_attempt
            or canonical.tool_name != request.tool_name
            or canonical.call_id != request.call_id
            or canonical.iteration != request.iteration
            or canonical.execution_subject
            != authorization.execution_subject
            or canonical.execution_subject_sha256
            != authorization.execution_subject.sha256
            or canonical.artifact != prepared_result_artifact
            or canonical.artifact != expected.result_artifact
            or canonical.completion_artifact
            != prepared_completion_artifact
            or canonical.completion_artifact
            != expected.completion_artifact
            or not _same_canonical_json(
                canonical.visible_result,
                prepared_visible_result,
            )
            or not _same_canonical_json(
                canonical.visible_result,
                expected.visible_result,
            )
            or canonical.cursor != expected.result_cursor
            or (
                canonical.duplicate
                and authorization.disposition
                is not DurableToolExecutionDisposition.FINALIZE
            )
        ):
            raise DurableToolExecutorContractError(
                "durable result receipt changed its journal binding"
            )
        return canonical

    def _assert_live_guard(
        self,
        request: DurableToolExecutionRequest,
    ) -> ExecutionFence:
        lease, root_guard, guard_chain = _resolve_official_execution_guard(
            self._execution_guard
        )
        if root_guard is not self._root_execution_guard:
            raise DurableToolExecutorContractError(
                "execution guard root changed after bootstrap binding"
            )
        if not _same_execution_guard_chain(
            guard_chain,
            self._execution_guard_chain,
        ):
            raise DurableToolExecutorContractError(
                "execution guard chain changed after bootstrap binding"
            )
        # A descendant session borrows the root lease; the subject fence stays
        # scoped to that child even when its journal attempt uses the root ID.
        live_fence = lease.fence
        if live_fence != request.subject.execution_fence:
            raise DurableToolExecutorContractError(
                "request does not match the live execution fence"
            )
        return live_fence

    def _recover_completion(
        self,
        *,
        request: DurableToolExecutionRequest,
        authorization: DurableToolAuthorization,
        current_fence: ExecutionFence,
    ) -> DurableToolCompletionReceipt:
        result_artifact = authorization.result_artifact
        completion_artifact = authorization.completion_artifact
        result_cursor = authorization.result_cursor
        if (
            result_artifact is None
            or completion_artifact is None
            or result_cursor is None
        ):
            raise DurableToolExecutorContractError(
                "reusable tool result has no durable completion artifact"
            )
        content = self._artifacts.read_full(
            completion_artifact,
            remaining_budget_bytes=completion_artifact.byte_length,
        )
        expected_preview = _bounded_utf8_preview(content)
        if completion_artifact.preview != expected_preview:
            raise DurableToolExecutorContractError(
                "durable completion preview changed"
            )
        try:
            decoded = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DurableToolExecutorContractError(
                "durable completion artifact is not valid JSON"
            ) from exc
        envelope = DurableToolCompletionEnvelope.from_dict(decoded)
        if envelope.transition is not None:
            from .tool_transitions import load_verified_subagent_state

            load_verified_subagent_state(
                self._artifacts,
                envelope.transition,
            )
        if (
            envelope.attempt != self._boundary.attempt
            or envelope.tool_name != request.tool_name
            or envelope.call_id != request.call_id
            or envelope.iteration != request.iteration
            or envelope.execution_subject != authorization.execution_subject
            or envelope.result_artifact != result_artifact
            or not _same_canonical_json(
                envelope.visible_result,
                authorization.visible_result,
            )
        ):
            raise DurableToolExecutorContractError(
                "durable completion artifact changed its execution subject"
            )
        return self._receipt(
            envelope=envelope,
            completion_artifact=completion_artifact,
            cursor=result_cursor,
            model_projection=self._model_projection_for_recovered_result(
                request.call_id
            ),
            current_fence=current_fence,
            reused=True,
        )

    def _finalize_sealed_completion(
        self,
        *,
        request: DurableToolExecutionRequest,
        authorization: DurableToolAuthorization,
        current_fence: ExecutionFence,
    ) -> DurableToolCompletionReceipt:
        from ..skeleton.hook_raw_seal import EVENT_TYPE, verify_hook_raw_event

        recovery = self._boundary.sink.recover_tool_side_effect(request.call_id)
        if recovery.sealed_event is not None and recovery.sealed_event.event_type == EVENT_TYPE:
            if self._completion_policy is None:
                raise DurableToolExecutorContractError(
                    "sealed Hook Tool result has no admitted completion policy"
                )
            _payload, raw_artifact, raw_draft, effective_arguments = verify_hook_raw_event(
                recovery.sealed_event, self._artifacts,
                attempt=self._boundary.attempt, tool_name=request.tool_name,
                call_id=request.call_id, iteration=request.iteration,
                recorded_subject=authorization.execution_subject,
            )
            if raw_artifact != authorization.result_artifact:
                raise DurableToolExecutorContractError(
                    "sealed Hook Tool artifact changed during finalization"
                )
            self._assert_live_guard(request)
            final_draft = self._completion_policy(
                request=request, raw_draft=raw_draft,
                effective_arguments=effective_arguments, artifacts=self._artifacts,
            )
            if type(final_draft) is not DurableToolCompletionDraft:
                raise DurableToolExecutorContractError(
                    "Hook completion policy must return a typed draft"
                )
            return self._persist_completed_draft(
                request=request, authorization=authorization,
                raw_draft=final_draft, reused=True,
            )
        result_artifact = authorization.result_artifact
        completion_artifact = authorization.completion_artifact
        if result_artifact is None or completion_artifact is None:
            raise DurableToolExecutorContractError(
                "sealed completion has incomplete artifact descriptors"
            )
        content = self._artifacts.read_full(
            completion_artifact,
            remaining_budget_bytes=completion_artifact.byte_length,
        )
        try:
            envelope = DurableToolCompletionEnvelope.from_dict(
                json.loads(content.decode("utf-8"))
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DurableToolExecutorContractError(
                "sealed completion artifact is not valid JSON"
            ) from exc
        if (
            envelope.attempt != self._boundary.attempt
            or envelope.tool_name != request.tool_name
            or envelope.call_id != request.call_id
            or envelope.iteration != request.iteration
            or envelope.execution_subject != authorization.execution_subject
            or envelope.result_artifact != result_artifact
            or envelope.transition is None
            or not _same_canonical_json(
                envelope.visible_result,
                authorization.visible_result,
            )
        ):
            raise DurableToolExecutorContractError(
                "sealed completion artifact changed its execution subject"
            )
        from .tool_transitions import load_verified_subagent_state

        load_verified_subagent_state(
            self._artifacts,
            envelope.transition,
        )
        result_artifactization = ToolResultArtifactization(
            artifact=result_artifact,
            visible_result=_thaw_json(envelope.visible_result),
            result_bytes=result_artifact.byte_length,
            result_sha256=result_artifact.sha256,
        )
        completion_artifactization = ToolCompletionArtifactization(
            artifact=completion_artifact,
            completion=envelope.to_dict(),
        )
        result_kwargs = {
            "artifactization": result_artifactization,
            "completion_artifactization": completion_artifactization,
        }
        if request.tool_result_policy != "default":
            result_kwargs["tool_result_policy"] = request.tool_result_policy
        result_receipt = self._boundary.persist_prepared_result(
            authorization,
            **result_kwargs,
        )
        result_receipt = self._validate_result_receipt(
            request,
            authorization=authorization,
            receipt=result_receipt,
            result_artifactization=result_artifactization,
            completion_artifactization=completion_artifactization,
            envelope=envelope,
        )
        return self._receipt(
            envelope=envelope,
            completion_artifact=result_receipt.completion_artifact,
            cursor=result_receipt.cursor,
            model_projection=result_receipt.model_projection,
            current_fence=current_fence,
            reused=True,
        )

    @staticmethod
    def _receipt(
        *,
        envelope: DurableToolCompletionEnvelope,
        completion_artifact: ArtifactRef,
        cursor: EventCursor,
        model_projection: Any,
        current_fence: ExecutionFence,
        reused: bool,
    ) -> DurableToolCompletionReceipt:
        trusted_attempt = _canonical_attempt_ref(envelope.attempt)
        trusted_subject = _coerce_execution_subject(
            envelope.execution_subject
        )
        trusted_result_artifact = _canonical_artifact_ref(
            envelope.result_artifact
        )
        trusted_completion_artifact = _canonical_artifact_ref(
            completion_artifact
        )
        trusted_cursor = _canonical_event_cursor(
            cursor,
            field_name="final tool receipt journal cursor",
        )
        trusted_fence = _canonical_execution_fence(
            current_fence,
            field_name="final tool receipt execution fence",
        )
        return DurableToolCompletionReceipt(
            attempt=trusted_attempt,
            tool_name=envelope.tool_name,
            call_id=envelope.call_id,
            iteration=envelope.iteration,
            execution_subject=trusted_subject,
            current_execution_fence=trusted_fence,
            visible_result=_thaw_json(
                _canonical_json_snapshot(
                    envelope.visible_result,
                    field_name="final tool receipt visible result",
                )
            ),
            should_observe=envelope.should_observe,
            result_artifact=trusted_result_artifact,
            completion_artifact=trusted_completion_artifact,
            journal_cursor=trusted_cursor,
            model_projection=model_projection,
            transition=envelope.transition,
            reused=reused,
        )

    def _model_projection_for_recovered_result(self, call_id: str) -> Any:
        recovery = self._boundary.sink.recover_tool_side_effect(call_id)
        event = recovery.result_event
        if event is None:
            return None
        projection = event.payload.get("model_projection")
        if projection is None:
            return None
        return _canonical_json_snapshot(
            projection,
            field_name="recovered tool result model projection",
        )


__all__ = [
    "DurableToolCompletionDraft",
    "DurableToolCompletionEnvelope",
    "DurableToolCompletionReceipt",
    "DurableToolExecutionRequest",
    "DurableToolExecutor",
    "DurableToolExecutorContractError",
    "DurableToolExecutorError",
    "DurableToolInvocation",
    "DurableToolInvocationFailedError",
]
