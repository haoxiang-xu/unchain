from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from unchain.execution import ExecutionFence
from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundToolReceiptIndex,
    DurableEventSink,
    EventCursor,
    JournalAppendResult,
    JournalConflictError,
    JournalEvent,
    SideEffectRecoveryState,
)
from unchain.journal.models import (
    _freeze_json,
    _required_text,
    _sha256,
    _thaw_json,
)

from .artifacts import (
    MAX_INLINE_TOOL_RESULT_BYTES,
    ArtifactService,
    ArtifactServiceError,
    ToolCompletionArtifactization,
    ToolResultArtifactization,
    _bounded_utf8_preview,
)
from .projector import CanonicalSemanticEventProjector


class DurableToolBoundaryError(RuntimeError):
    """Base error for the fail-closed durable tool execution boundary."""


class DurableToolExecutionUncertainError(DurableToolBoundaryError):
    """A side effect may have run and therefore must not be replayed."""


class DurableToolBoundaryCorruptError(DurableToolBoundaryError):
    """Durable tool receipts contradict the requested execution identity."""


class DurableToolExecutionDisposition(StrEnum):
    EXECUTE = "execute"
    FINALIZE = "finalize"
    REUSE = "reuse"


class DurableToolApprovalState(StrEnum):
    NOT_REQUIRED = "not_required"
    APPROVED = "approved"
    DENIED = "denied"


class DurableToolRouteKind(StrEnum):
    NORMAL = "normal"
    PLUGIN = "plugin"


def _iteration(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DurableToolBoundaryCorruptError("tool iteration is invalid")
    return value


def _canonical_digest(value: Any) -> str:
    frozen = _freeze_json(value, path="durable_tool_subject")
    content = json.dumps(
        _thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _fence_to_dict(fence: ExecutionFence) -> dict[str, Any]:
    return {
        "execution_id": fence.execution_id,
        "owner_id": fence.owner_id,
        "fencing_sequence": fence.fencing_token,
    }


def _fence_from_dict(value: Mapping[str, Any]) -> ExecutionFence:
    if not isinstance(value, Mapping) or set(value) != {
        "execution_id",
        "owner_id",
        "fencing_sequence",
    }:
        raise DurableToolBoundaryCorruptError("execution fence is invalid")
    return ExecutionFence(
        execution_id=value["execution_id"],
        owner_id=value["owner_id"],
        fencing_token=value["fencing_sequence"],
    )


@dataclass(frozen=True)
class DurableToolExecutionSubject:
    """Secret-free durable identity of one approved tool invocation."""

    SCHEMA: ClassVar[str] = "unchain.durable_tool_execution_subject.v1"

    intent_cursor: EventCursor
    original_arguments_sha256: str
    effective_arguments_sha256: str
    approval_state: DurableToolApprovalState
    approval_request_sha256: str
    approval_receipt_sha256: str
    route_kind: DurableToolRouteKind
    route_manifest_sha256: str
    terminal_handler_manifest_sha256: str
    execution_fence: ExecutionFence

    def __post_init__(self) -> None:
        if type(self.intent_cursor) is not EventCursor:
            if not isinstance(self.intent_cursor, Mapping):
                raise DurableToolBoundaryCorruptError(
                    "tool intent cursor must be an exact EventCursor"
                )
            intent_cursor = EventCursor.from_dict(self.intent_cursor)
        else:
            intent_cursor = EventCursor.from_dict(
                EventCursor.to_dict(self.intent_cursor)
            )
        object.__setattr__(self, "intent_cursor", intent_cursor)
        if intent_cursor.store_seq < 1:
            raise DurableToolBoundaryCorruptError(
                "tool call intent cursor must be durable"
            )
        object.__setattr__(
            self,
            "original_arguments_sha256",
            _sha256(
                self.original_arguments_sha256,
                "original_arguments_sha256",
            ),
        )
        object.__setattr__(
            self,
            "effective_arguments_sha256",
            _sha256(
                self.effective_arguments_sha256,
                "effective_arguments_sha256",
            ),
        )
        if not isinstance(self.approval_state, DurableToolApprovalState):
            object.__setattr__(
                self,
                "approval_state",
                DurableToolApprovalState(self.approval_state),
            )
        request_digest = str(self.approval_request_sha256 or "").strip()
        receipt_digest = str(self.approval_receipt_sha256 or "").strip()
        if self.approval_state in {
            DurableToolApprovalState.APPROVED,
            DurableToolApprovalState.DENIED,
        }:
            if not request_digest or not receipt_digest:
                raise DurableToolBoundaryCorruptError(
                    "approval request and receipt digests are both required"
                )
            request_digest = _sha256(
                request_digest,
                "approval_request_sha256",
            )
            receipt_digest = _sha256(
                receipt_digest,
                "approval_receipt_sha256",
            )
        elif request_digest or receipt_digest:
            raise DurableToolBoundaryCorruptError(
                "not_required approval cannot contain request or receipt digests"
            )
        object.__setattr__(self, "approval_request_sha256", request_digest)
        object.__setattr__(self, "approval_receipt_sha256", receipt_digest)
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
        if type(self.execution_fence) is not ExecutionFence:
            if not isinstance(self.execution_fence, Mapping):
                raise DurableToolBoundaryCorruptError(
                    "tool execution fence must be an exact ExecutionFence"
                )
            execution_fence = _fence_from_dict(self.execution_fence)
        else:
            execution_fence = ExecutionFence(
                execution_id=self.execution_fence.execution_id,
                owner_id=self.execution_fence.owner_id,
                fencing_token=self.execution_fence.fencing_token,
            )
        object.__setattr__(self, "execution_fence", execution_fence)

    @property
    def sha256(self) -> str:
        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "intent_cursor": self.intent_cursor.to_dict(),
            "original_arguments_sha256": self.original_arguments_sha256,
            "effective_arguments_sha256": self.effective_arguments_sha256,
            "approval_state": self.approval_state.value,
            "approval_request_sha256": self.approval_request_sha256,
            "approval_receipt_sha256": self.approval_receipt_sha256,
            "route_kind": self.route_kind.value,
            "route_manifest_sha256": self.route_manifest_sha256,
            "terminal_handler_manifest_sha256": (
                self.terminal_handler_manifest_sha256
            ),
            "execution_fence": _fence_to_dict(self.execution_fence),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> DurableToolExecutionSubject:
        if cls is not DurableToolExecutionSubject:
            raise DurableToolBoundaryCorruptError(
                "execution subject subclasses are not accepted"
            )
        if not isinstance(value, Mapping):
            raise TypeError("execution subject must be an object")
        raw = dict(value)
        expected = {
            "schema",
            "intent_cursor",
            "original_arguments_sha256",
            "effective_arguments_sha256",
            "approval_state",
            "approval_request_sha256",
            "approval_receipt_sha256",
            "route_kind",
            "route_manifest_sha256",
            "terminal_handler_manifest_sha256",
            "execution_fence",
        }
        if set(raw) != expected or raw.get("schema") != cls.SCHEMA:
            raise DurableToolBoundaryCorruptError(
                "execution subject schema is invalid"
            )
        return DurableToolExecutionSubject(
            intent_cursor=EventCursor.from_dict(raw["intent_cursor"]),
            original_arguments_sha256=raw["original_arguments_sha256"],
            effective_arguments_sha256=raw["effective_arguments_sha256"],
            approval_state=raw["approval_state"],
            approval_request_sha256=raw["approval_request_sha256"],
            approval_receipt_sha256=raw["approval_receipt_sha256"],
            route_kind=raw["route_kind"],
            route_manifest_sha256=raw["route_manifest_sha256"],
            terminal_handler_manifest_sha256=(
                raw["terminal_handler_manifest_sha256"]
            ),
            execution_fence=_fence_from_dict(raw["execution_fence"]),
        )


@dataclass(frozen=True)
class DurableToolAuthorization:
    disposition: DurableToolExecutionDisposition
    tool_name: str
    call_id: str
    iteration: int
    execution_subject: DurableToolExecutionSubject
    started_cursor: EventCursor
    current_execution_fence: ExecutionFence | None = None
    result_artifact: ArtifactRef | None = None
    completion_artifact: ArtifactRef | None = None
    result_cursor: EventCursor | None = None
    visible_result: Any = None
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, DurableToolExecutionDisposition):
            object.__setattr__(
                self,
                "disposition",
                DurableToolExecutionDisposition(self.disposition),
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
        if type(self.execution_subject) is not DurableToolExecutionSubject:
            if not isinstance(self.execution_subject, Mapping):
                raise DurableToolBoundaryCorruptError(
                    "authorization execution subject must be an exact subject"
                )
            object.__setattr__(
                self,
                "execution_subject",
                DurableToolExecutionSubject.from_dict(self.execution_subject),
            )
        if not isinstance(self.started_cursor, EventCursor):
            object.__setattr__(
                self,
                "started_cursor",
                EventCursor.from_dict(self.started_cursor),
            )
        current_fence = self.current_execution_fence
        if current_fence is None:
            current_fence = self.execution_subject.execution_fence
        elif not isinstance(current_fence, ExecutionFence):
            current_fence = _fence_from_dict(current_fence)
        object.__setattr__(self, "current_execution_fence", current_fence)
        recorded_fence = self.execution_subject.execution_fence
        if current_fence.execution_id != recorded_fence.execution_id:
            raise DurableToolBoundaryCorruptError(
                "current execution fence crossed the recorded execution"
            )
        if current_fence.fencing_token < recorded_fence.fencing_token or (
            current_fence.fencing_token == recorded_fence.fencing_token
            and current_fence.owner_id != recorded_fence.owner_id
        ):
            raise DurableToolBoundaryCorruptError(
                "current execution fence is stale or contradictory"
            )
        if self.result_artifact is not None and not isinstance(
            self.result_artifact,
            ArtifactRef,
        ):
            object.__setattr__(
                self,
                "result_artifact",
                ArtifactRef.from_dict(self.result_artifact),
            )
        if self.completion_artifact is not None and not isinstance(
            self.completion_artifact,
            ArtifactRef,
        ):
            object.__setattr__(
                self,
                "completion_artifact",
                ArtifactRef.from_dict(self.completion_artifact),
            )
        if self.result_cursor is not None and not isinstance(
            self.result_cursor,
            EventCursor,
        ):
            object.__setattr__(
                self,
                "result_cursor",
                EventCursor.from_dict(self.result_cursor),
            )
        frozen = _freeze_json(self.visible_result, path="visible_result")
        object.__setattr__(self, "visible_result", frozen)
        if self.disposition is DurableToolExecutionDisposition.EXECUTE:
            if current_fence != recorded_fence:
                raise DurableToolBoundaryCorruptError(
                    "execute authorization requires its recorded execution fence"
                )
            if (
                self.result_artifact is not None
                or self.completion_artifact is not None
                or self.result_cursor is not None
                or self.visible_result is not None
            ):
                raise DurableToolBoundaryCorruptError(
                    "execute authorization cannot contain a terminal result"
                )
        elif self.result_artifact is None or self.result_cursor is None:
            raise DurableToolBoundaryCorruptError(
                "reuse authorization requires a durable result artifact and cursor"
            )

    @property
    def should_execute(self) -> bool:
        return self.disposition is DurableToolExecutionDisposition.EXECUTE


@dataclass(frozen=True)
class DurableToolResultReceipt:
    attempt: AttemptRef
    tool_name: str
    call_id: str
    iteration: int
    execution_subject: DurableToolExecutionSubject
    execution_subject_sha256: str
    visible_result: Any
    artifact: ArtifactRef
    cursor: EventCursor
    completion_artifact: ArtifactRef | None = None
    duplicate: bool = False
    _authority: object = field(default=None, repr=False, compare=False)
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
        if type(self.execution_subject) is not DurableToolExecutionSubject:
            if not isinstance(self.execution_subject, Mapping):
                raise DurableToolBoundaryCorruptError(
                    "result receipt execution subject must be an exact subject"
                )
            object.__setattr__(
                self,
                "execution_subject",
                DurableToolExecutionSubject.from_dict(self.execution_subject),
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
            raise DurableToolBoundaryCorruptError(
                "tool result receipt execution subject digest changed"
            )
        object.__setattr__(
            self,
            "visible_result",
            _freeze_json(self.visible_result, path="visible_result"),
        )
        if not isinstance(self.artifact, ArtifactRef):
            object.__setattr__(
                self,
                "artifact",
                ArtifactRef.from_dict(self.artifact),
            )
        if not isinstance(self.cursor, EventCursor):
            object.__setattr__(self, "cursor", EventCursor.from_dict(self.cursor))
        if self.model_projection is not None:
            if not isinstance(self.model_projection, Mapping) or set(
                self.model_projection
            ) != {"result", "metadata"}:
                raise DurableToolBoundaryCorruptError(
                    "tool result model projection schema is invalid"
                )
            if not isinstance(self.model_projection["result"], Mapping) or not isinstance(
                self.model_projection["metadata"], Mapping
            ):
                raise DurableToolBoundaryCorruptError(
                    "tool result model projection payload is invalid"
                )
            object.__setattr__(
                self,
                "model_projection",
                _freeze_json(self.model_projection, path="model_projection"),
            )
        if self.completion_artifact is not None and not isinstance(
            self.completion_artifact,
            ArtifactRef,
        ):
            object.__setattr__(
                self,
                "completion_artifact",
                ArtifactRef.from_dict(self.completion_artifact),
            )
        if not isinstance(self.duplicate, bool):
            raise TypeError("duplicate must be a boolean")


class DurableToolBoundary:
    """Persist side-effect intent and sanitized output around one tool call.

    A fresh ``tool.started`` receipt is the only state that yields an execution
    authorization.  On restart, a started call without a terminal result is
    intentionally uncertain and cannot be executed automatically.
    """

    def __init__(
        self,
        *,
        attempt: AttemptRef,
        projector: CanonicalSemanticEventProjector,
        sink: DurableEventSink,
    ) -> None:
        if not isinstance(attempt, AttemptRef):
            attempt = AttemptRef.from_dict(attempt)
        if type(projector) is not CanonicalSemanticEventProjector:
            raise TypeError(
                "projector must be the official CanonicalSemanticEventProjector"
            )
        if type(sink) is not DurableEventSink:
            raise TypeError("sink must be the official DurableEventSink")
        if projector.attempt != attempt or sink.attempt != attempt:
            raise DurableToolBoundaryCorruptError(
                "tool boundary components do not share one exact attempt"
            )
        if sink.projector is not projector:
            raise DurableToolBoundaryCorruptError(
                "tool boundary sink does not use the bound projector"
            )
        if not isinstance(sink.journal, BoundToolReceiptIndex):
            raise DurableToolBoundaryCorruptError(
                "tool boundary requires an indexed receipt journal"
            )
        self._attempt = attempt
        self._projector = projector
        self._sink = sink
        self._authority = object()
        self._projected_execution_id: str | None = None
        self._projected_execution_lock = threading.Lock()

    def _bind_projected_execution(self, binding: Any) -> None:
        """Freeze the one child session allowed to use a root-journal attempt."""

        from .runtime import (
            _BoundExecutionGuard,
            _CONTEXT_EXECUTION_BINDING_AUTHORITY,
        )

        if (
            type(binding) is not _BoundExecutionGuard
            or binding._authority is not _CONTEXT_EXECUTION_BINDING_AUTHORITY
            or binding.attempt != self._attempt
            or binding.root_execution_id
            != self._attempt.generation.execution_id
            or binding.execution_id == binding.root_execution_id
        ):
            raise DurableToolBoundaryCorruptError(
                "projected execution binding does not match the tool attempt"
            )
        with self._projected_execution_lock:
            if self._projected_execution_id not in {
                None,
                binding.execution_id,
            }:
                raise DurableToolBoundaryCorruptError(
                    "tool attempt is bound to a different projected execution"
                )
            self._projected_execution_id = binding.execution_id

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    @property
    def projector(self) -> CanonicalSemanticEventProjector:
        return self._projector

    @property
    def sink(self) -> DurableEventSink:
        return self._sink

    def authorize_execution(
        self,
        *,
        tool_name: object,
        call_id: object,
        iteration: object,
        subject: DurableToolExecutionSubject,
    ) -> DurableToolAuthorization:
        normalized_tool = _required_text(
            tool_name,
            "tool_name",
            identifier=True,
        )
        normalized_call = _required_text(
            call_id,
            "call_id",
            identifier=True,
        )
        normalized_iteration = _iteration(iteration)
        if type(subject) is not DurableToolExecutionSubject:
            if not isinstance(subject, Mapping):
                raise DurableToolBoundaryCorruptError(
                    "authorization requires an exact execution subject"
                )
            subject = DurableToolExecutionSubject.from_dict(subject)
        recovery = self._sink.recover_tool_side_effect(normalized_call)
        self._verify_intent(
            recovery.intent_event,
            tool_name=normalized_tool,
            call_id=normalized_call,
            iteration=normalized_iteration,
            subject=subject,
        )
        if recovery.state is SideEffectRecoveryState.NOT_STARTED:
            draft = self._projector(
                {
                    "type": "tool.started",
                    "run_id": self._attempt.attempt_id,
                    "iteration": normalized_iteration,
                    "tool_name": normalized_tool,
                    "call_id": normalized_call,
                    "execution_subject": subject.to_dict(),
                    "execution_subject_sha256": subject.sha256,
                }
            )
            if draft is None:  # pragma: no cover - official projector invariant
                raise DurableToolBoundaryCorruptError(
                    "tool start did not produce a durable draft"
                )
            try:
                appended = self._sink.append_projected(draft)
            except JournalConflictError as exc:
                raise DurableToolBoundaryCorruptError(
                    "tool started atomic claim conflicted"
                ) from exc
            if appended.duplicate:
                return self._authorization_after_race(
                    tool_name=normalized_tool,
                    call_id=normalized_call,
                    iteration=normalized_iteration,
                    subject=subject,
                )
            return DurableToolAuthorization(
                disposition=DurableToolExecutionDisposition.EXECUTE,
                tool_name=normalized_tool,
                call_id=normalized_call,
                iteration=normalized_iteration,
                execution_subject=subject,
                started_cursor=appended.cursor,
                _authority=self._authority,
            )
        return self._authorization_from_recovery(
            recovery,
            tool_name=normalized_tool,
            call_id=normalized_call,
            iteration=normalized_iteration,
            subject=subject,
        )

    def persist_result(
        self,
        authorization: DurableToolAuthorization,
        result: Any,
    ) -> DurableToolResultReceipt:
        if (
            type(authorization) is not DurableToolAuthorization
            or authorization._authority is not self._authority
            or authorization.disposition
            is not DurableToolExecutionDisposition.EXECUTE
        ):
            raise DurableToolBoundaryCorruptError(
                "tool result requires this boundary's execute authorization"
            )
        recovery = self._sink.recover_tool_side_effect(authorization.call_id)
        self._verify_intent(
            recovery.intent_event,
            tool_name=authorization.tool_name,
            call_id=authorization.call_id,
            iteration=authorization.iteration,
            subject=authorization.execution_subject,
        )
        if recovery.state is SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE:
            recovered = self._authorization_from_recovery(
                recovery,
                tool_name=authorization.tool_name,
                call_id=authorization.call_id,
                iteration=authorization.iteration,
                subject=authorization.execution_subject,
            )
            return DurableToolResultReceipt(
                attempt=self._attempt,
                tool_name=authorization.tool_name,
                call_id=authorization.call_id,
                iteration=authorization.iteration,
                execution_subject=authorization.execution_subject,
                execution_subject_sha256=(
                    authorization.execution_subject.sha256
                ),
                visible_result=_thaw_json(recovered.visible_result),
                artifact=recovered.result_artifact,
                cursor=self._result_cursor(recovery.result_event),
                model_projection=self._model_projection_from_event(
                    recovery.result_event
                ),
                completion_artifact=recovered.completion_artifact,
                duplicate=True,
                _authority=self._authority,
            )
        if recovery.state is not SideEffectRecoveryState.UNCERTAIN_AFTER_START:
            raise DurableToolBoundaryCorruptError(
                "tool result has no exact durable started receipt"
            )
        self._verify_recovery_identity(
            recovery.started_event,
            tool_name=authorization.tool_name,
            call_id=authorization.call_id,
            iteration=authorization.iteration,
            subject=authorization.execution_subject,
        )
        if self._cursor(recovery.started_event) != authorization.started_cursor:
            raise DurableToolBoundaryCorruptError(
                "tool started receipt changed after authorization"
            )
        draft = self._projector(
            {
                "type": "tool_result",
                "run_id": self._attempt.attempt_id,
                "iteration": authorization.iteration,
                "tool_name": authorization.tool_name,
                "call_id": authorization.call_id,
                "execution_subject": authorization.execution_subject.to_dict(),
                "execution_subject_sha256": (
                    authorization.execution_subject.sha256
                ),
                "result": result,
            }
        )
        if draft is None:  # pragma: no cover - official projector invariant
            raise DurableToolBoundaryCorruptError(
                "tool result did not produce a durable draft"
            )
        try:
            appended = self._sink.append_projected(draft)
        except JournalConflictError as exc:
            raise DurableToolBoundaryCorruptError(
                "tool result atomic claim conflicted"
            ) from exc
        return self._result_receipt(
            appended,
            authorization=authorization,
        )

    def persist_prepared_result(
        self,
        authorization: DurableToolAuthorization,
        *,
        artifactization: ToolResultArtifactization,
        completion_artifactization: ToolCompletionArtifactization,
        tool_result_policy: str = "default",
    ) -> DurableToolResultReceipt:
        """Atomically journal references to two already-persisted artifacts."""

        if (
            type(authorization) is not DurableToolAuthorization
            or authorization._authority is not self._authority
            or authorization.disposition not in {
                DurableToolExecutionDisposition.EXECUTE,
                DurableToolExecutionDisposition.FINALIZE,
            }
        ):
            raise DurableToolBoundaryCorruptError(
                "prepared tool result requires this boundary's execute authorization"
            )
        if type(artifactization) is not ToolResultArtifactization:
            raise TypeError(
                "prepared tool result requires ToolResultArtifactization"
            )
        if type(completion_artifactization) is not ToolCompletionArtifactization:
            raise TypeError(
                "prepared tool result requires ToolCompletionArtifactization"
            )
        if not isinstance(tool_result_policy, str) or not tool_result_policy.strip():
            raise DurableToolBoundaryCorruptError(
                "prepared tool result policy is invalid"
            )
        tool_result_policy = tool_result_policy.strip().lower()
        try:
            artifactization = ArtifactService.verify_tool_result_artifactization(
                self._projector.artifacts,
                artifactization,
            )
            completion_artifactization = (
                ArtifactService.verify_tool_completion_artifactization(
                    self._projector.artifacts,
                    completion_artifactization,
                )
            )
            self._verify_completion_binding(
                completion_artifactization.completion,
                tool_name=authorization.tool_name,
                call_id=authorization.call_id,
                iteration=authorization.iteration,
                subject=authorization.execution_subject,
                result_artifact=artifactization.artifact,
                visible_result=artifactization.visible_result,
            )
        except ArtifactServiceError as exc:
            raise DurableToolBoundaryCorruptError(
                "prepared tool artifact failed verification"
            ) from exc

        recovery = self._sink.recover_tool_side_effect(authorization.call_id)
        self._verify_intent(
            recovery.intent_event,
            tool_name=authorization.tool_name,
            call_id=authorization.call_id,
            iteration=authorization.iteration,
            subject=authorization.execution_subject,
        )
        if recovery.state is SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE:
            recovered = self._authorization_from_recovery(
                recovery,
                tool_name=authorization.tool_name,
                call_id=authorization.call_id,
                iteration=authorization.iteration,
                subject=authorization.execution_subject,
            )
            if (
                recovered.result_artifact != artifactization.artifact
                or recovered.completion_artifact
                != completion_artifactization.artifact
            ):
                raise DurableToolBoundaryCorruptError(
                    "prepared tool result raced with a different completion"
                )
            return DurableToolResultReceipt(
                attempt=self._attempt,
                tool_name=authorization.tool_name,
                call_id=authorization.call_id,
                iteration=authorization.iteration,
                execution_subject=authorization.execution_subject,
                execution_subject_sha256=(
                    authorization.execution_subject.sha256
                ),
                visible_result=_thaw_json(recovered.visible_result),
                artifact=recovered.result_artifact,
                cursor=self._result_cursor(recovery.result_event),
                model_projection=self._model_projection_from_event(
                    recovery.result_event
                ),
                completion_artifact=recovered.completion_artifact,
                duplicate=True,
                _authority=self._authority,
            )
        if recovery.state not in {
            SideEffectRecoveryState.UNCERTAIN_AFTER_START,
            SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE,
        }:
            raise DurableToolBoundaryCorruptError(
                "prepared tool result has no exact durable started receipt"
            )
        self._verify_recovery_identity(
            recovery.started_event,
            tool_name=authorization.tool_name,
            call_id=authorization.call_id,
            iteration=authorization.iteration,
            subject=authorization.execution_subject,
        )
        if self._cursor(recovery.started_event) != authorization.started_cursor:
            raise DurableToolBoundaryCorruptError(
                "tool started receipt changed after authorization"
            )
        completion = _thaw_json(completion_artifactization.completion)
        transition = completion.get("transition")
        if recovery.state is SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE:
            from ..skeleton.hook_raw_seal import EVENT_TYPE, verify_hook_raw_event
            if recovery.sealed_event is not None and recovery.sealed_event.event_type == EVENT_TYPE:
                recorded_subject = self._verify_recovery_identity(
                    recovery.started_event,
                    tool_name=authorization.tool_name,
                    call_id=authorization.call_id,
                    iteration=authorization.iteration,
                    subject=authorization.execution_subject,
                    allow_newer_fence=True,
                )
                verify_hook_raw_event(recovery.sealed_event, self._projector.artifacts,
                    attempt=self._attempt, tool_name=authorization.tool_name,
                    call_id=authorization.call_id, iteration=authorization.iteration,
                    recorded_subject=recorded_subject)
            else:
                self._verify_sealed_completion(
                    recovery.sealed_event,
                    authorization=authorization,
                    result_artifact=artifactization.artifact,
                    completion_artifact=completion_artifactization.artifact,
                    transition=transition,
                )
        elif transition is not None:
            raise DurableToolBoundaryCorruptError(
                "stateful tool completion has no durable seal"
            )
        draft = self._projector.project_prepared_tool_result(
            {
                "type": "tool_result",
                "run_id": self._attempt.attempt_id,
                "iteration": authorization.iteration,
                "tool_name": authorization.tool_name,
                "call_id": authorization.call_id,
                "tool_result_policy": tool_result_policy,
                "execution_subject": authorization.execution_subject.to_dict(),
                "execution_subject_sha256": (
                    authorization.execution_subject.sha256
                ),
            },
            artifactization=artifactization,
            completion_artifact=completion_artifactization.artifact,
        )
        try:
            appended = self._sink.append_projected(draft)
        except JournalConflictError as exc:
            raise DurableToolBoundaryCorruptError(
                "prepared tool result atomic claim conflicted"
            ) from exc
        return self._result_receipt(
            appended,
            authorization=authorization,
        )

    def persist_sealed_completion(
        self,
        authorization: DurableToolAuthorization,
        *,
        result_artifact: ArtifactRef,
        completion_artifact: ArtifactRef,
        transition: Any,
    ) -> EventCursor:
        """Atomically claim a stateful completion before parent result append."""

        from .tool_transitions import DurableToolStateTransitionEnvelope

        if (
            type(authorization) is not DurableToolAuthorization
            or authorization._authority is not self._authority
            or authorization.disposition
            is not DurableToolExecutionDisposition.EXECUTE
            or type(transition) is not DurableToolStateTransitionEnvelope
        ):
            raise DurableToolBoundaryCorruptError(
                "sealed completion requires runtime-owned authorization"
            )
        if (
            transition.parent_attempt != self._attempt
            or transition.call_id != authorization.call_id
            or transition.execution_subject_sha256
            != authorization.execution_subject.sha256
        ):
            raise DurableToolBoundaryCorruptError(
                "sealed completion transition changed its parent binding"
            )
        recovery = self._sink.recover_tool_side_effect(authorization.call_id)
        if recovery.state is SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE:
            self._verify_sealed_completion(
                recovery.sealed_event,
                authorization=authorization,
                result_artifact=result_artifact,
                completion_artifact=completion_artifact,
                transition=transition.to_dict(),
            )
            return self._cursor(recovery.sealed_event)
        if recovery.state is not SideEffectRecoveryState.UNCERTAIN_AFTER_START:
            raise DurableToolBoundaryCorruptError(
                "sealed completion has no exact durable started receipt"
            )
        self._verify_recovery_identity(
            recovery.started_event,
            tool_name=authorization.tool_name,
            call_id=authorization.call_id,
            iteration=authorization.iteration,
            subject=authorization.execution_subject,
        )
        if self._cursor(recovery.started_event) != authorization.started_cursor:
            raise DurableToolBoundaryCorruptError(
                "tool started receipt changed before sealed completion"
            )
        draft = self._projector.project_sealed_tool_completion(
            {
                "type": "tool.subagent_completion.sealed",
                "run_id": self._attempt.attempt_id,
                "iteration": authorization.iteration,
                "tool_name": authorization.tool_name,
                "call_id": authorization.call_id,
                "execution_subject": authorization.execution_subject.to_dict(),
                "execution_subject_sha256": (
                    authorization.execution_subject.sha256
                ),
            },
            result_artifact=result_artifact,
            completion_artifact=completion_artifact,
            transition=transition.to_dict(),
            next_state_artifact=transition.next_state_artifact,
            handoff_refs=transition.handoff_refs,
        )
        try:
            appended = self._sink.append_projected(draft)
        except JournalConflictError as exc:
            raise DurableToolBoundaryCorruptError(
                "sealed completion atomic claim conflicted"
            ) from exc
        return appended.cursor

    def _authorization_after_race(
        self,
        *,
        tool_name: str,
        call_id: str,
        iteration: int,
        subject: DurableToolExecutionSubject,
    ) -> DurableToolAuthorization:
        recovery = self._sink.recover_tool_side_effect(call_id)
        return self._authorization_from_recovery(
            recovery,
            tool_name=tool_name,
            call_id=call_id,
            iteration=iteration,
            subject=subject,
        )

    def _verify_sealed_completion(
        self,
        event: JournalEvent | None,
        *,
        authorization: DurableToolAuthorization,
        result_artifact: ArtifactRef,
        completion_artifact: ArtifactRef,
        transition: Any,
    ) -> None:
        from .tool_transitions import DurableToolStateTransitionEnvelope

        if event is None or event.event_type != "tool.subagent_completion.sealed":
            raise DurableToolBoundaryCorruptError(
                "sealed completion receipt is missing"
            )
        expected_fields = {
            "run_id",
            "iteration",
            "tool_name",
            "call_id",
            "execution_subject",
            "execution_subject_sha256",
            "result_artifact",
            "completion_artifact",
            "next_state_artifact",
            "transition",
            "handoff_refs",
        }
        raw = _thaw_json(event.payload)
        actual_fields = set(raw) if isinstance(raw, dict) else set()
        if (
            not isinstance(raw, dict)
            or frozenset(actual_fields) not in {
                frozenset(expected_fields),
                frozenset(expected_fields | {"parent_run_id"}),
            }
            or (
                "parent_run_id" in raw
                and (
                    type(raw["parent_run_id"]) is not str
                    or not raw["parent_run_id"]
                    or raw["parent_run_id"] == raw.get("run_id")
                )
            )
        ):
            raise DurableToolBoundaryCorruptError(
                "sealed completion payload schema is invalid"
            )
        try:
            sealed_result = ArtifactRef.from_dict(raw["result_artifact"])
            sealed_completion = ArtifactRef.from_dict(
                raw["completion_artifact"]
            )
            sealed_next = ArtifactRef.from_dict(raw["next_state_artifact"])
            sealed_transition = DurableToolStateTransitionEnvelope.from_dict(
                raw["transition"]
            )
            supplied_transition = (
                transition
                if type(transition) is DurableToolStateTransitionEnvelope
                else DurableToolStateTransitionEnvelope.from_dict(transition)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise DurableToolBoundaryCorruptError(
                "sealed completion descriptor is invalid"
            ) from exc
        if (
            raw["run_id"] != self._attempt.attempt_id
            or raw["iteration"] != authorization.iteration
            or raw["tool_name"] != authorization.tool_name
            or raw["call_id"] != authorization.call_id
            or raw["execution_subject"]
            != authorization.execution_subject.to_dict()
            or raw["execution_subject_sha256"]
            != authorization.execution_subject.sha256
            or sealed_result != result_artifact
            or sealed_completion != completion_artifact
            or sealed_next != sealed_transition.next_state_artifact
            or sealed_transition != supplied_transition
        ):
            raise DurableToolBoundaryCorruptError(
                "sealed completion changed its parent binding"
            )
        declared_refs = (
            result_artifact.ref,
            completion_artifact.ref,
            sealed_transition.next_state_artifact.ref,
            *sealed_transition.handoff_refs,
        )
        if (
            tuple(event.resource_refs) != declared_refs
            or raw["handoff_refs"]
            != [ref.to_dict() for ref in sealed_transition.handoff_refs]
        ):
            raise DurableToolBoundaryCorruptError(
                "sealed completion resource declarations changed"
            )

    def _authorization_from_recovery(
        self,
        recovery,
        *,
        tool_name: str,
        call_id: str,
        iteration: int,
        subject: DurableToolExecutionSubject,
    ) -> DurableToolAuthorization:
        if recovery.state is SideEffectRecoveryState.CORRUPT:
            raise DurableToolBoundaryCorruptError(
                recovery.reason or "durable tool receipts are corrupt"
            )
        self._verify_intent(
            recovery.intent_event,
            tool_name=tool_name,
            call_id=call_id,
            iteration=iteration,
            subject=subject,
        )
        recorded_subject = self._verify_recovery_identity(
            recovery.started_event,
            tool_name=tool_name,
            call_id=call_id,
            iteration=iteration,
            subject=subject,
            allow_newer_fence=True,
        )
        if recovery.state is SideEffectRecoveryState.UNCERTAIN_AFTER_START:
            raise DurableToolExecutionUncertainError(
                "tool execution may have started; automatic replay is forbidden"
            )
        if recovery.state is SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE:
            sealed_event = recovery.sealed_event
            if sealed_event is None:
                raise DurableToolBoundaryCorruptError(
                    "sealed completion receipt is missing"
                )
            from ..skeleton.hook_raw_seal import EVENT_TYPE, verify_hook_raw_event
            if sealed_event.event_type == EVENT_TYPE:
                _payload, raw_artifact, _draft, _arguments = verify_hook_raw_event(
                    sealed_event, self._projector.artifacts,
                    attempt=self._attempt, tool_name=tool_name, call_id=call_id,
                    iteration=iteration, recorded_subject=recorded_subject,
                )
                return DurableToolAuthorization(
                    disposition=DurableToolExecutionDisposition.FINALIZE,
                    tool_name=tool_name,
                    call_id=call_id,
                    iteration=iteration,
                    execution_subject=recorded_subject,
                    started_cursor=self._cursor(recovery.started_event),
                    current_execution_fence=subject.execution_fence,
                    result_artifact=raw_artifact,
                    completion_artifact=raw_artifact,
                    result_cursor=self._cursor(sealed_event),
                    visible_result=None,
                    _authority=self._authority,
                )
            raw = _thaw_json(sealed_event.payload)
            try:
                result_artifact = ArtifactRef.from_dict(
                    raw["result_artifact"]
                )
                completion_artifact = ArtifactRef.from_dict(
                    raw["completion_artifact"]
                )
                completion = self._read_verified_completion(
                    completion_artifact
                )
                from .tool_executor import DurableToolCompletionEnvelope

                envelope = DurableToolCompletionEnvelope.from_dict(
                    _thaw_json(completion)
                )
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                raise DurableToolBoundaryCorruptError(
                    "sealed completion artifact binding is invalid"
                ) from exc
            if envelope.transition is None:
                raise DurableToolBoundaryCorruptError(
                    "sealed completion has no state transition"
                )
            executing = DurableToolAuthorization(
                disposition=DurableToolExecutionDisposition.EXECUTE,
                tool_name=tool_name,
                call_id=call_id,
                iteration=iteration,
                execution_subject=recorded_subject,
                started_cursor=self._cursor(recovery.started_event),
                _authority=self._authority,
            )
            self._verify_sealed_completion(
                sealed_event,
                authorization=executing,
                result_artifact=result_artifact,
                completion_artifact=completion_artifact,
                transition=envelope.transition,
            )
            try:
                result_content = self._projector.artifacts.read_full(
                    result_artifact,
                    remaining_budget_bytes=result_artifact.byte_length,
                )
                decoded_result = json.loads(result_content.decode("utf-8"))
            except (ArtifactServiceError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DurableToolBoundaryCorruptError(
                    "sealed result artifact failed verification"
                ) from exc
            expected_preview = _bounded_utf8_preview(result_content)
            if result_artifact.preview != expected_preview:
                raise DurableToolBoundaryCorruptError(
                    "sealed result artifact preview changed"
                )
            visible_result = (
                decoded_result
                if result_artifact.byte_length <= MAX_INLINE_TOOL_RESULT_BYTES
                else {
                    "preview": expected_preview,
                    "full_output_ref": result_artifact.ref.to_dict(),
                    "content_bytes": result_artifact.byte_length,
                    "content_sha256": result_artifact.sha256,
                }
            )
            self._verify_completion_binding(
                completion,
                tool_name=tool_name,
                call_id=call_id,
                iteration=iteration,
                subject=recorded_subject,
                result_artifact=result_artifact,
                visible_result=visible_result,
            )
            return DurableToolAuthorization(
                disposition=DurableToolExecutionDisposition.FINALIZE,
                tool_name=tool_name,
                call_id=call_id,
                iteration=iteration,
                execution_subject=recorded_subject,
                started_cursor=self._cursor(recovery.started_event),
                current_execution_fence=subject.execution_fence,
                result_artifact=result_artifact,
                completion_artifact=completion_artifact,
                result_cursor=self._cursor(sealed_event),
                visible_result=visible_result,
                _authority=self._authority,
            )
        if recovery.state is not SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE:
            raise DurableToolBoundaryCorruptError(
                "durable tool recovery changed unexpectedly"
            )
        self._verify_recovery_identity(
            recovery.result_event,
            tool_name=tool_name,
            call_id=call_id,
            iteration=iteration,
            subject=recorded_subject,
        )
        if recovery.reusable_result_artifact is None:
            raise DurableToolBoundaryCorruptError(
                "durable tool result artifact is missing"
            )
        result_event = recovery.result_event
        if result_event is None or "result" not in result_event.payload:
            raise DurableToolBoundaryCorruptError(
                "durable tool result has no model-visible value"
            )
        result_artifact = self._artifact_from_event(result_event)
        if (
            result_artifact.ref != recovery.reusable_result_artifact.ref
            or result_artifact.media_type
            != recovery.reusable_result_artifact.media_type
            or result_artifact.byte_length
            != recovery.reusable_result_artifact.byte_length
            or result_artifact.sha256
            != recovery.reusable_result_artifact.sha256
            or result_artifact.preview
            != recovery.reusable_result_artifact.preview
        ):
            raise DurableToolBoundaryCorruptError(
                "durable recovery artifact descriptor changed"
            )
        verified_content = None
        if result_artifact.byte_length <= MAX_INLINE_TOOL_RESULT_BYTES:
            try:
                verified_content = self._projector.artifacts.read_full(
                    result_artifact,
                    remaining_budget_bytes=result_artifact.byte_length,
                )
            except ArtifactServiceError as exc:
                raise DurableToolBoundaryCorruptError(
                    "durable recovery artifact failed verification"
                ) from exc
        visible_result = self._verified_visible_result(
            result_event,
            result_artifact,
            verified_content,
        )
        completion_artifact = self._completion_artifact_from_event(result_event)
        if completion_artifact is not None:
            try:
                completion = self._read_verified_completion(
                    completion_artifact
                )
            except ArtifactServiceError as exc:
                raise DurableToolBoundaryCorruptError(
                    "durable completion artifact failed verification"
                ) from exc
            verified_completion = self._verify_completion_binding(
                completion,
                tool_name=tool_name,
                call_id=call_id,
                iteration=iteration,
                subject=recorded_subject,
                result_artifact=result_artifact,
                visible_result=visible_result,
            )
            transition = verified_completion["transition"]
            from ..skeleton.hook_raw_seal import EVENT_TYPE, verify_hook_raw_event
            hook_seal = (recovery.sealed_event is not None
                and recovery.sealed_event.event_type == EVENT_TYPE)
            if hook_seal:
                verify_hook_raw_event(recovery.sealed_event, self._projector.artifacts,
                    attempt=self._attempt, tool_name=tool_name, call_id=call_id,
                    iteration=iteration, recorded_subject=recorded_subject)
            if transition is None and recovery.sealed_event is not None and not hook_seal:
                raise DurableToolBoundaryCorruptError(
                    "durable seal has no completion transition"
                )
            if transition is not None:
                if recovery.sealed_event is None:
                    raise DurableToolBoundaryCorruptError(
                        "stateful tool completion has no durable seal"
                    )
                executing = DurableToolAuthorization(
                    disposition=DurableToolExecutionDisposition.EXECUTE,
                    tool_name=tool_name,
                    call_id=call_id,
                    iteration=iteration,
                    execution_subject=recorded_subject,
                    started_cursor=self._cursor(recovery.started_event),
                    _authority=self._authority,
                )
                if not hook_seal:
                    self._verify_sealed_completion(
                        recovery.sealed_event,
                        authorization=executing,
                        result_artifact=result_artifact,
                        completion_artifact=completion_artifact,
                        transition=transition,
                    )
        return DurableToolAuthorization(
            disposition=DurableToolExecutionDisposition.REUSE,
            tool_name=tool_name,
            call_id=call_id,
            iteration=iteration,
            execution_subject=recorded_subject,
            started_cursor=self._cursor(recovery.started_event),
            current_execution_fence=subject.execution_fence,
            result_artifact=result_artifact,
            completion_artifact=completion_artifact,
            result_cursor=self._result_cursor(result_event),
            visible_result=visible_result,
            _authority=self._authority,
        )

    @staticmethod
    def _verified_visible_result(
        event: JournalEvent,
        artifact: ArtifactRef,
        content: bytes | None,
    ) -> Any:
        if artifact.byte_length <= MAX_INLINE_TOOL_RESULT_BYTES:
            if not isinstance(content, bytes):
                raise DurableToolBoundaryCorruptError(
                    "inline durable tool result requires verified content"
                )
            try:
                decoded = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DurableToolBoundaryCorruptError(
                    "durable tool result artifact is not valid JSON"
                ) from exc
            expected_preview = _bounded_utf8_preview(content)
            if artifact.preview != expected_preview:
                raise DurableToolBoundaryCorruptError(
                    "durable tool result artifact preview changed"
                )
            expected_visible = decoded
        else:
            expected_visible = {
                "preview": artifact.preview,
                "full_output_ref": artifact.ref.to_dict(),
                "content_bytes": artifact.byte_length,
                "content_sha256": artifact.sha256,
            }
        actual_visible = _thaw_json(event.payload.get("result"))
        try:
            matches = _canonical_digest(
                {"result": actual_visible}
            ) == _canonical_digest({"result": expected_visible})
        except (TypeError, ValueError) as exc:
            raise DurableToolBoundaryCorruptError(
                "durable tool model-visible result is invalid"
            ) from exc
        if not matches:
            raise DurableToolBoundaryCorruptError(
                "durable tool model-visible result disagrees with artifact"
            )
        return expected_visible

    def _verify_completion_binding(
        self,
        completion: Any,
        *,
        tool_name: str,
        call_id: str,
        iteration: int,
        subject: DurableToolExecutionSubject,
        result_artifact: ArtifactRef,
        visible_result: Any,
    ) -> dict[str, Any]:
        raw = _thaw_json(completion)
        expected_fields = {
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
        if (
            not isinstance(raw, dict)
            or set(raw) != expected_fields
            or raw.get("schema") != "unchain.durable_tool_completion.v2"
            or type(raw.get("should_observe")) is not bool
        ):
            raise DurableToolBoundaryCorruptError(
                "durable tool completion identity is invalid"
            )
        expected = {
            "schema": "unchain.durable_tool_completion.v2",
            "attempt": self._attempt.to_dict(),
            "tool_name": tool_name,
            "call_id": call_id,
            "iteration": iteration,
            "execution_subject": subject.to_dict(),
            "execution_subject_sha256": subject.sha256,
            "result_artifact": result_artifact.to_dict(),
            "visible_result": _thaw_json(visible_result),
            "should_observe": raw["should_observe"],
            "transition": raw["transition"],
        }
        try:
            matches = _canonical_digest(raw) == _canonical_digest(expected)
        except (TypeError, ValueError) as exc:
            raise DurableToolBoundaryCorruptError(
                "durable tool completion identity is invalid"
            ) from exc
        if not matches:
            raise DurableToolBoundaryCorruptError(
                "durable tool completion identity changed"
            )
        return raw

    @staticmethod
    def _verify_recovery_identity(
        event: JournalEvent | None,
        *,
        tool_name: str,
        call_id: str,
        iteration: int,
        subject: DurableToolExecutionSubject,
        allow_newer_fence: bool = False,
    ) -> DurableToolExecutionSubject:
        if event is None:
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt is missing"
            )
        if (
            event.payload.get("tool_name") != tool_name
            or event.payload.get("call_id") != call_id
            or event.payload.get("iteration") != iteration
        ):
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt identity does not match the request"
            )
        if event.store_seq <= subject.intent_cursor.store_seq:
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt does not follow its tool call intent"
            )
        return DurableToolBoundary._verify_event_subject(
            event,
            subject,
            allow_newer_fence=allow_newer_fence,
        )

    def _verify_intent(
        self,
        intent: JournalEvent | None,
        *,
        tool_name: str,
        call_id: str,
        iteration: int,
        subject: DurableToolExecutionSubject,
    ) -> None:
        expected_execution_id = (
            self._projected_execution_id
            or self._attempt.generation.execution_id
        )
        if subject.execution_fence.execution_id != expected_execution_id:
            raise DurableToolBoundaryCorruptError(
                "execution fence does not match the tool execution"
            )
        if intent is None or intent.event_type != "tool_call":
            raise DurableToolBoundaryCorruptError(
                "tool call intent is missing"
            )
        if (
            intent.payload.get("tool_name") != tool_name
            or intent.payload.get("iteration") != iteration
        ):
            raise DurableToolBoundaryCorruptError(
                "tool call intent identity does not match the request"
            )
        if self._cursor(intent) != subject.intent_cursor:
            raise DurableToolBoundaryCorruptError(
                "tool call intent cursor changed"
            )
        arguments_digest = _canonical_digest(
            _thaw_json(intent.payload.get("arguments", {}))
        )
        if arguments_digest != subject.original_arguments_sha256:
            raise DurableToolBoundaryCorruptError(
                "original arguments digest changed"
            )

    @staticmethod
    def _verify_event_subject(
        event: JournalEvent,
        expected: DurableToolExecutionSubject,
        *,
        allow_newer_fence: bool = False,
    ) -> DurableToolExecutionSubject:
        try:
            actual = DurableToolExecutionSubject.from_dict(
                event.payload.get("execution_subject")
            )
            actual_digest = _sha256(
                event.payload.get("execution_subject_sha256"),
                "execution_subject_sha256",
            )
        except (TypeError, ValueError) as exc:
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt execution subject is invalid"
            ) from exc
        if actual_digest != actual.sha256:
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt execution subject digest changed"
            )
        comparisons = (
            ("intent cursor", actual.intent_cursor, expected.intent_cursor),
            (
                "original arguments",
                actual.original_arguments_sha256,
                expected.original_arguments_sha256,
            ),
            (
                "effective arguments",
                actual.effective_arguments_sha256,
                expected.effective_arguments_sha256,
            ),
            ("approval state", actual.approval_state, expected.approval_state),
            (
                "approval request",
                actual.approval_request_sha256,
                expected.approval_request_sha256,
            ),
            (
                "approval receipt",
                actual.approval_receipt_sha256,
                expected.approval_receipt_sha256,
            ),
            ("route kind", actual.route_kind, expected.route_kind),
            (
                "route manifest",
                actual.route_manifest_sha256,
                expected.route_manifest_sha256,
            ),
            (
                "terminal handler",
                actual.terminal_handler_manifest_sha256,
                expected.terminal_handler_manifest_sha256,
            ),
        )
        for field_name, actual_value, expected_value in comparisons:
            if actual_value != expected_value:
                raise DurableToolBoundaryCorruptError(
                    f"durable tool receipt {field_name} changed"
                )
        recorded_fence = actual.execution_fence
        current_fence = expected.execution_fence
        if allow_newer_fence:
            fence_changed = (
                current_fence.execution_id != recorded_fence.execution_id
                or current_fence.fencing_token < recorded_fence.fencing_token
                or (
                    current_fence.fencing_token == recorded_fence.fencing_token
                    and current_fence.owner_id != recorded_fence.owner_id
                )
            )
        else:
            fence_changed = current_fence != recorded_fence
        if fence_changed:
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt execution fence changed"
            )
        return actual

    @staticmethod
    def _cursor(event: JournalEvent | None) -> EventCursor:
        if event is None:
            raise DurableToolBoundaryCorruptError(
                "durable tool receipt is missing"
            )
        return EventCursor(event.store_seq, event.event_id)

    @classmethod
    def _result_cursor(cls, event: JournalEvent | None) -> EventCursor:
        return cls._cursor(event)

    def _result_receipt(
        self,
        appended: JournalAppendResult,
        *,
        authorization: DurableToolAuthorization,
    ) -> DurableToolResultReceipt:
        event = appended.event
        if event.event_type != "tool_result" or "result" not in event.payload:
            raise DurableToolBoundaryCorruptError(
                "persisted tool result receipt is invalid"
            )
        artifact = self._artifact_from_event(event)
        completion_artifact = self._completion_artifact_from_event(event)
        completion = None
        try:
            content = self._projector.artifacts.read_full(
                artifact,
                remaining_budget_bytes=artifact.byte_length,
            )
            if completion_artifact is not None:
                completion = self._read_verified_completion(
                    completion_artifact
                )
        except ArtifactServiceError as exc:
            raise DurableToolBoundaryCorruptError(
                "persisted tool artifact failed verification"
            ) from exc
        visible_result = self._verified_visible_result(
            event,
            artifact,
            content,
        )
        DurableToolBoundary._verify_event_subject(
            event,
            authorization.execution_subject,
        )
        if completion is not None:
            self._verify_completion_binding(
                completion,
                tool_name=authorization.tool_name,
                call_id=authorization.call_id,
                iteration=authorization.iteration,
                subject=authorization.execution_subject,
                result_artifact=artifact,
                visible_result=visible_result,
            )
        return DurableToolResultReceipt(
            attempt=event.attempt,
            tool_name=authorization.tool_name,
            call_id=authorization.call_id,
            iteration=authorization.iteration,
            execution_subject=authorization.execution_subject,
            execution_subject_sha256=(
                authorization.execution_subject.sha256
            ),
            visible_result=visible_result,
            artifact=artifact,
            cursor=appended.cursor,
            model_projection=self._model_projection_from_event(event),
            completion_artifact=completion_artifact,
            duplicate=appended.duplicate,
            _authority=self._authority,
        )

    @staticmethod
    def _model_projection_from_event(event: JournalEvent | None) -> Any:
        if event is None:
            return None
        projection = event.payload.get("model_projection")
        if projection is None:
            return None
        if not isinstance(projection, Mapping):
            raise DurableToolBoundaryCorruptError(
                "persisted tool result model projection is invalid"
            )
        return _thaw_json(_freeze_json(projection, path="model_projection"))

    @staticmethod
    def _artifact_from_event(event: JournalEvent) -> ArtifactRef:
        raw_ref = event.payload.get("full_output_ref")
        raw_bytes = event.payload.get("result_bytes")
        raw_sha256 = event.payload.get("result_sha256")
        try:
            artifact = ArtifactRef(
                ref=raw_ref,
                media_type="application/json",
                byte_length=raw_bytes,
                sha256=raw_sha256,
                preview=str(event.payload.get("preview") or ""),
            )
        except (TypeError, ValueError) as exc:
            raise DurableToolBoundaryCorruptError(
                "persisted tool result artifact descriptor is invalid"
            ) from exc
        if artifact.ref not in event.resource_refs:
            raise DurableToolBoundaryCorruptError(
                "persisted tool result artifact ref is not declared"
            )
        return artifact

    @staticmethod
    def _completion_artifact_from_event(
        event: JournalEvent,
    ) -> ArtifactRef | None:
        field_names = (
            "completion_ref",
            "completion_bytes",
            "completion_sha256",
            "completion_preview",
        )
        present = tuple(name in event.payload for name in field_names)
        if not any(present):
            return None
        if not all(present):
            raise DurableToolBoundaryCorruptError(
                "persisted tool completion descriptor is incomplete"
            )
        try:
            artifact = ArtifactRef(
                ref=event.payload["completion_ref"],
                media_type="application/json",
                byte_length=event.payload["completion_bytes"],
                sha256=event.payload["completion_sha256"],
                preview=event.payload["completion_preview"],
            )
        except (TypeError, ValueError) as exc:
            raise DurableToolBoundaryCorruptError(
                "persisted tool completion descriptor is invalid"
            ) from exc
        if artifact.ref not in event.resource_refs:
            raise DurableToolBoundaryCorruptError(
                "persisted tool completion ref is not declared"
            )
        return artifact

    def _read_verified_completion(
        self,
        artifact: ArtifactRef,
    ) -> Any:
        content = self._projector.artifacts.read_full(
            artifact,
            remaining_budget_bytes=artifact.byte_length,
        )
        try:
            decoded = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactServiceError(
                "durable completion artifact is not valid JSON"
            ) from exc
        verified = ArtifactService.verify_tool_completion_artifactization(
            self._projector.artifacts,
            ToolCompletionArtifactization(
                artifact=artifact,
                completion=decoded,
            ),
        )
        return verified.completion


__all__ = [
    "DurableToolApprovalState",
    "DurableToolAuthorization",
    "DurableToolBoundary",
    "DurableToolBoundaryCorruptError",
    "DurableToolBoundaryError",
    "DurableToolExecutionDisposition",
    "DurableToolExecutionSubject",
    "DurableToolExecutionUncertainError",
    "DurableToolRouteKind",
    "DurableToolResultReceipt",
]
