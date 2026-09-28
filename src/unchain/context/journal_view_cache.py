from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from unchain.journal import (
    BoundExecutionJournal,
    JournalScopeError,
    JournalSnapshot,
    capture_journal_snapshot,
)


class JournalSnapshotSource(Protocol):
    """Returns a complete, verified snapshot for one bound journal."""

    @property
    def journal(self) -> BoundExecutionJournal:
        ...

    def capture_snapshot(self) -> JournalSnapshot:
        ...


class JournalViewCacheError(RuntimeError):
    """A process-local journal view could not be verified or refreshed."""


def _positive_limit(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _serialized_size(snapshot: JournalSnapshot) -> int:
    return len(
        json.dumps(
            snapshot.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


def _validated_snapshot(
    snapshot: JournalSnapshot,
    *,
    journal: BoundExecutionJournal,
    max_events: int,
    max_bytes: int,
) -> JournalSnapshot:
    if not isinstance(snapshot, JournalSnapshot):
        raise JournalViewCacheError("journal did not return a stable snapshot")
    normalized = JournalSnapshot.from_dict(snapshot.to_dict())
    if normalized.execution_id != journal.execution_id:
        raise JournalViewCacheError("journal snapshot escaped its execution scope")
    if normalized.event_count > max_events:
        raise JournalViewCacheError("journal snapshot exceeds the event limit")
    if _serialized_size(normalized) > max_bytes:
        raise JournalViewCacheError("journal snapshot exceeds the byte limit")
    return normalized


class FullJournalSnapshotSource:
    """The durable, cache-disabled snapshot source used for control runs."""

    def __init__(
        self,
        journal: BoundExecutionJournal,
        *,
        max_events: int = 10_000,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        self._journal = journal
        self._max_events = _positive_limit(max_events, "max_events")
        self._max_bytes = _positive_limit(max_bytes, "max_bytes")

    @property
    def journal(self) -> BoundExecutionJournal:
        return self._journal

    def capture_snapshot(self) -> JournalSnapshot:
        journal = self.journal
        return _validated_snapshot(
            journal.capture_snapshot(
                max_events=self._max_events,
                max_bytes=self._max_bytes,
            ),
            journal=journal,
            max_events=self._max_events,
            max_bytes=self._max_bytes,
        )


@dataclass(frozen=True)
class JournalViewCacheMetrics:
    full_snapshot_reads: int
    tail_reads: int
    cache_hits: int
    refresh_fallbacks: int


class RunLocalJournalViewCache:
    """Keep one verified journal prefix in process memory for a live run.

    The cache only stores a derived copy of a complete durable snapshot.  Each
    refresh reads the suffix after that snapshot's exact high-water cursor and
    rebuilds a new complete snapshot before it can be used.  Durable journal
    writes remain the source of truth and are intentionally outside this class.
    """

    _registry_lock = threading.RLock()
    _registry: weakref.WeakKeyDictionary[
        BoundExecutionJournal, dict[tuple[int, int], "RunLocalJournalViewCache"]
    ] = weakref.WeakKeyDictionary()

    @classmethod
    def for_journal(
        cls,
        journal: BoundExecutionJournal,
        *,
        max_events: int = 10_000,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> "RunLocalJournalViewCache":
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        limits = (
            _positive_limit(max_events, "max_events"),
            _positive_limit(max_bytes, "max_bytes"),
        )
        with cls._registry_lock:
            try:
                cached_by_limits = cls._registry.get(journal)
            except TypeError:
                # A non-weak-referenceable adapter is safe but cannot share a
                # registry entry. The caller still receives an isolated cache.
                return cls(
                    journal,
                    max_events=max_events,
                    max_bytes=max_bytes,
                )
            if cached_by_limits is None:
                cached_by_limits = {}
                cls._registry[journal] = cached_by_limits
            cached = cached_by_limits.get(limits)
            if cached is None:
                cached = cls(
                    journal,
                    max_events=limits[0],
                    max_bytes=limits[1],
                )
                cached_by_limits[limits] = cached
            return cached

    def __init__(
        self,
        journal: BoundExecutionJournal,
        *,
        max_events: int = 10_000,
        max_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        self._journal_ref = weakref.ref(journal)
        self._max_events = _positive_limit(max_events, "max_events")
        self._max_bytes = _positive_limit(max_bytes, "max_bytes")
        self._lock = threading.RLock()
        self._snapshot: JournalSnapshot | None = None
        self._integrity_revision: int | None = None
        self._full_snapshot_reads = 0
        self._tail_reads = 0
        self._cache_hits = 0
        self._refresh_fallbacks = 0

    @property
    def journal(self) -> BoundExecutionJournal:
        journal = self._journal_ref()
        if journal is None:
            raise JournalViewCacheError("bound journal is no longer available")
        return journal

    def metrics(self) -> JournalViewCacheMetrics:
        with self._lock:
            return JournalViewCacheMetrics(
                full_snapshot_reads=self._full_snapshot_reads,
                tail_reads=self._tail_reads,
                cache_hits=self._cache_hits,
                refresh_fallbacks=self._refresh_fallbacks,
            )

    def invalidate(self) -> None:
        """Discard a derived view after a caller rejects its semantic content."""
        with self._lock:
            self._snapshot = None

    def capture_snapshot(self) -> JournalSnapshot:
        with self._lock:
            if self._snapshot is None:
                return self._capture_full()
            try:
                journal = self.journal
                if not journal.snapshot_prefix_is_current(
                    snapshot=self._snapshot,
                    integrity_revision=self._integrity_revision,
                ):
                    self._refresh_fallbacks += 1
                    return self._capture_full()
                refreshed = self._refresh_from_tail(self._snapshot, journal=journal)
                if (
                    self._integrity_revision is not None
                    and journal.snapshot_integrity_revision()
                    != self._integrity_revision
                ):
                    self._refresh_fallbacks += 1
                    return self._capture_full()
            except JournalScopeError:
                self._refresh_fallbacks += 1
                return self._capture_full()
            if refreshed is None:
                self._refresh_fallbacks += 1
                return self._capture_full()
            self._snapshot = refreshed
            self._cache_hits += 1
            return refreshed

    def _capture_full(self) -> JournalSnapshot:
        self._full_snapshot_reads += 1
        journal = self.journal
        captured, revision = journal.capture_snapshot_with_integrity_revision(
            max_events=self._max_events,
            max_bytes=self._max_bytes,
        )
        snapshot = _validated_snapshot(
            captured,
            journal=journal,
            max_events=self._max_events,
            max_bytes=self._max_bytes,
        )
        self._snapshot = snapshot
        self._integrity_revision = revision
        return snapshot

    def _refresh_from_tail(
        self,
        snapshot: JournalSnapshot,
        *,
        journal: BoundExecutionJournal,
    ) -> JournalSnapshot | None:
        remaining = self._max_events - snapshot.event_count
        # Read one event even at the limit so a newly appended event triggers
        # the same bounded full-snapshot failure as the cache-disabled path.
        self._tail_reads += 1
        page = journal.read(
            after=snapshot.high_water,
            limit=max(1, remaining),
        )
        if page.has_more:
            return None
        if not page.events:
            if page.next_cursor not in (None, snapshot.high_water):
                return None
            return snapshot
        if snapshot.high_water is None:
            expected_sequence = 1
        else:
            expected_sequence = snapshot.high_water.store_seq + 1
        if page.events[0].store_seq != expected_sequence:
            return None
        if any(
            earlier.store_seq + 1 != later.store_seq
            for earlier, later in zip(page.events, page.events[1:])
        ):
            return None
        if any(
            event.attempt.generation.execution_id != journal.execution_id
            for event in page.events
        ):
            return None
        try:
            rebuilt = capture_journal_snapshot(
                execution_id=journal.execution_id,
                events=(*snapshot.events, *page.events),
            )
            return _validated_snapshot(
                rebuilt,
                journal=journal,
                max_events=self._max_events,
                max_bytes=self._max_bytes,
            )
        except (TypeError, ValueError, KeyError):
            return None


__all__ = [
    "FullJournalSnapshotSource",
    "JournalSnapshotSource",
    "JournalViewCacheError",
    "JournalViewCacheMetrics",
    "RunLocalJournalViewCache",
]
