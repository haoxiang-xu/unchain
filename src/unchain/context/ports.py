from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from unchain.journal import (
    ArtifactRef,
    EventCursor,
    EventRange,
    OperationRef,
    ResourceRef,
)
from unchain.journal.models import _required_text

from .models import ContextBuildEnvelope
from .task_state import ContextTaskStateReadOutcome


class ContextRepositoryError(RuntimeError):
    """Base error for an execution-bound context capability."""


class ContextConflictError(ContextRepositoryError):
    """An idempotency or revision precondition conflicted."""


class ContextScopeError(ContextRepositoryError):
    """A record did not belong to the bound execution."""


class _ExecutionBoundPort(ABC):
    def __init__(self, execution_id: str) -> None:
        self._execution_id = _required_text(
            execution_id,
            "execution_id",
            identifier=True,
        )

    @property
    def execution_id(self) -> str:
        return self._execution_id


class BoundArtifactRepository(_ExecutionBoundPort):
    @abstractmethod
    def put(
        self,
        *,
        content: bytes,
        media_type: str,
        operation: OperationRef,
        preview: str = "",
    ) -> ArtifactRef:
        """Persist bytes before returning a durable artifact reference."""

    @abstractmethod
    def read_verified(
        self,
        *,
        artifact: ArtifactRef,
        offset: int = 0,
        limit: int = 65_536,
    ) -> bytes:
        """Scope-authorize and verify the whole object before returning a slice."""

    @abstractmethod
    def read_full_verified(self, *, artifact: ArtifactRef) -> bytes:
        """Scope-authorize and return one fully verified immutable object."""

    def artifact_id_for(self, *, logical_kind: str, logical_key: str) -> str:
        """Return the deterministic artifact id a claim would receive.

        Lets a caller compute the identity an artifact will get from
        ``put()`` before it is actually written, so it can be referenced from
        an event drafted in the same transaction. A repository that cannot
        compute this ahead of a real write raises rather than guessing.
        """

        raise NotImplementedError(
            "artifact repository cannot pre-compute artifact ids"
        )


class CheckpointWriteStatus(StrEnum):
    PREPARED = "prepared"
    COMMITTED = "committed"


@dataclass(frozen=True)
class PreparedCheckpoint:
    """A durable operation receipt whose content is hidden until commit."""

    preparation_id: str
    checkpoint_ref: ResourceRef
    operation: OperationRef
    status: CheckpointWriteStatus = CheckpointWriteStatus.PREPARED
    duplicate: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preparation_id",
            _required_text(
                self.preparation_id,
                "preparation_id",
                identifier=True,
            ),
        )
        if not isinstance(self.checkpoint_ref, ResourceRef):
            object.__setattr__(
                self,
                "checkpoint_ref",
                ResourceRef.from_dict(self.checkpoint_ref),
            )
        if (
            self.checkpoint_ref.kind != "checkpoint"
            or self.checkpoint_ref.fragment
        ):
            raise ValueError(
                "prepared checkpoint must contain an unfragmented checkpoint ref"
            )
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(
                self,
                "operation",
                OperationRef.from_dict(self.operation),
            )
        if not isinstance(self.status, CheckpointWriteStatus):
            object.__setattr__(
                self,
                "status",
                CheckpointWriteStatus(self.status),
            )
        if not isinstance(self.duplicate, bool):
            raise TypeError("duplicate must be a boolean")


@dataclass(frozen=True)
class ContextBuildReceipt:
    """Durable idempotency receipt for one recorded context build."""

    envelope: ContextBuildEnvelope
    operation: OperationRef
    trigger_cursor: EventCursor
    duplicate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.envelope, ContextBuildEnvelope):
            object.__setattr__(
                self,
                "envelope",
                ContextBuildEnvelope.from_dict(self.envelope),
            )
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(
                self,
                "operation",
                OperationRef.from_dict(self.operation),
            )
        if not isinstance(self.trigger_cursor, EventCursor):
            object.__setattr__(
                self,
                "trigger_cursor",
                EventCursor.from_dict(self.trigger_cursor),
            )
        if not isinstance(self.duplicate, bool):
            raise TypeError("duplicate must be a boolean")


class BoundCheckpointRepository(_ExecutionBoundPort):
    def checkpoint_ref_for(self, *, operation: OperationRef) -> ResourceRef:
        """Return the deterministic ref a checkpoint preparation will receive.

        This read-only preview lets the context compiler price the exact
        checkpoint marker before ``prepare()`` performs a durable write.  A
        repository that cannot guarantee the future identity raises rather
        than returning an estimate.
        """

        raise NotImplementedError(
            "checkpoint repository cannot pre-compute checkpoint refs"
        )

    def list_committed_refs(self, *, limit: int = 32) -> tuple[ResourceRef, ...]:
        """Return a bounded newest-first list of committed local checkpoints.

        Older implementations do not expose discovery.  They remain safe: the
        coordinator simply compiles without reuse and may create a new
        checkpoint under pressure.
        """

        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        return ()

    def get_by_ref(self, *, ref: ResourceRef) -> PreparedCheckpoint | None:
        """Return durable checkpoint metadata for one exact local ref.

        Automatic compiler reuse requires this metadata to distinguish its
        reserved operation identities from host-authored checkpoint content.
        Older adapters can leave the method unsupported; the coordinator then
        disables automatic reuse instead of interpreting arbitrary summaries.
        """

        raise NotImplementedError(
            "checkpoint repository cannot resolve checkpoint metadata by ref"
        )

    @abstractmethod
    def prepare(
        self,
        *,
        source_range: EventRange,
        summary: str,
        refs: tuple[ResourceRef, ...],
        operation: OperationRef,
    ) -> PreparedCheckpoint:
        """Durably reserve an idempotent ref without authorizing content reads."""

    @abstractmethod
    def commit(self, *, prepared: PreparedCheckpoint) -> PreparedCheckpoint:
        """Commit one proven preparation and return a committed operation receipt."""

    @abstractmethod
    def get_by_operation(
        self,
        *,
        operation: OperationRef,
    ) -> PreparedCheckpoint | None:
        """Recover a matching prepared/committed receipt or fail on conflict."""

    @abstractmethod
    def read(self, *, ref: ResourceRef, offset: int = 0, limit: int = 65_536) -> bytes:
        """Read a bounded checkpoint slice."""


class BoundContextBuildRepository(_ExecutionBoundPort):
    @abstractmethod
    def record(
        self,
        *,
        envelope: ContextBuildEnvelope,
        operation: OperationRef,
        trigger_cursor: EventCursor,
    ) -> ContextBuildReceipt:
        """Atomically claim one input trigger and persist its reproducible build."""

    @abstractmethod
    def get_by_operation(
        self,
        *,
        operation: OperationRef,
    ) -> ContextBuildReceipt | None:
        """Recover a matching build receipt or fail on operation conflict."""

    @abstractmethod
    def get_by_trigger(
        self,
        *,
        trigger_cursor: EventCursor,
    ) -> ContextBuildReceipt | None:
        """Return the sole build authorized by one durable input receipt."""

    @abstractmethod
    def latest(self, *, generation_id: str) -> ContextBuildEnvelope | None:
        """Return the latest build for a generation in this execution."""


class BoundContextTaskStateReader(ABC):
    """Task-state read capability bound to one immutable host execution."""

    def __init__(self, binding_id: str) -> None:
        self._binding_id = _required_text(
            binding_id,
            "binding_id",
            identifier=True,
        )

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @abstractmethod
    def read_for_context(self) -> ContextTaskStateReadOutcome:
        """Return content or a content-free unavailable marker."""


__all__ = [
    "BoundArtifactRepository",
    "BoundCheckpointRepository",
    "BoundContextBuildRepository",
    "BoundContextTaskStateReader",
    "CheckpointWriteStatus",
    "ContextConflictError",
    "ContextBuildReceipt",
    "ContextRepositoryError",
    "ContextScopeError",
    "PreparedCheckpoint",
]
