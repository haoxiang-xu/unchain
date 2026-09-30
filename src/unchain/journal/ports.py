from __future__ import annotations

from abc import ABC, abstractmethod

from collections.abc import Callable

from .models import (
    AttemptRef,
    EventCursor,
    JournalAppendRequest,
    JournalAppendResult,
    JournalPage,
    PendingArtifact,
    ToolExecutionReceiptLookup,
    _required_text,
)
from .snapshot import JournalSnapshot


class JournalRepositoryError(RuntimeError):
    """Base error for an execution-bound journal capability."""


class JournalConflictError(JournalRepositoryError):
    """An idempotency key or optimistic write precondition conflicted."""


class JournalScopeError(JournalRepositoryError):
    """A record did not belong to the capability's bound execution."""


class BoundExecutionJournal(ABC):
    """Journal capability constructed for exactly one execution."""

    def __init__(self, execution_id: str) -> None:
        self._execution_id = _required_text(
            execution_id,
            "execution_id",
            identifier=True,
        )

    @property
    def execution_id(self) -> str:
        return self._execution_id

    @abstractmethod
    def append(self, *, request: JournalAppendRequest) -> JournalAppendResult:
        """Durably append or idempotently replay one semantic event."""

    @abstractmethod
    def read(self, *, after: EventCursor | None = None, limit: int = 100) -> JournalPage:
        """Read integrity-verified persisted events after an optional cursor."""

    @abstractmethod
    def capture_snapshot(
        self,
        *,
        max_events: int = 10_000,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> JournalSnapshot:
        """Atomically capture a bounded execution high-water snapshot."""

    def snapshot_integrity_revision(self) -> int | None:
        """Return a durable mutation revision when the adapter exposes one.

        `None` preserves the generic safe behavior: callers must validate a
        full durable snapshot before reusing a cached prefix.
        """

        return None

    def capture_snapshot_with_integrity_revision(
        self,
        *,
        max_events: int = 10_000,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> tuple[JournalSnapshot, int | None]:
        """Capture a snapshot together with its durable mutation revision.

        Adapters with a compact revision should override this so both values
        come from one durable read transaction. The default remains safe for
        adapters without one because later prefix validation rehydrates the
        durable prefix instead of trusting the revision.
        """

        return (
            self.capture_snapshot(max_events=max_events, max_bytes=max_bytes),
            self.snapshot_integrity_revision(),
        )

    def snapshot_prefix_is_current(
        self,
        *,
        snapshot: JournalSnapshot,
        integrity_revision: int | None = None,
    ) -> bool:
        """Check whether a previously captured durable prefix is still exact.

        Implementations with a compact journal identity should override this
        method. The generic fallback deliberately rehydrates from durable
        authority so a cache cannot turn an unknown adapter into a stale
        history source.
        """

        if not isinstance(snapshot, JournalSnapshot):
            raise TypeError("snapshot must be a JournalSnapshot")
        if snapshot.execution_id != self.execution_id:
            return False
        current = self.capture_snapshot()
        return current.events[: snapshot.event_count] == snapshot.events

    def append_with_artifacts(
        self,
        *,
        request: JournalAppendRequest,
        artifacts: tuple[PendingArtifact, ...],
        precondition: Callable[[JournalSnapshot], None] | None = None,
    ) -> JournalAppendResult:
        """Atomically claim artifacts and append one event in a single write.

        Order inside the write transaction: exact operation replay check
        (a duplicate returns the original result without re-running
        ``precondition`` or re-claiming the artifacts), a current-state
        snapshot handed to ``precondition`` (raise to reject the write with
        nothing persisted), the artifact rows, then the event row. A journal
        that cannot provide this atomicity refuses it rather than silently
        falling back to a non-atomic sequence.
        """

        raise JournalRepositoryError(
            "journal cannot claim artifacts and append an event atomically"
        )


class BoundToolReceiptIndex(BoundExecutionJournal):
    """Execution-bound journal with an exact indexed tool receipt lookup."""

    @abstractmethod
    def lookup_tool_execution_receipts(
        self,
        *,
        attempt: AttemptRef,
        call_id: str,
    ) -> ToolExecutionReceiptLookup:
        """Atomically return the exhaustive receipt set for one tool call."""


__all__ = [
    "BoundExecutionJournal",
    "BoundToolReceiptIndex",
    "JournalConflictError",
    "JournalRepositoryError",
    "JournalScopeError",
]
