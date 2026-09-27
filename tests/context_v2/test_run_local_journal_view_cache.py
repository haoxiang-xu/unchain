from __future__ import annotations

import gc
import sqlite3
import weakref

import pytest

from unchain.context.journal_view_cache import RunLocalJournalViewCache
from unchain.journal import (
    AttemptRef,
    BoundExecutionJournal,
    EventCursor,
    GenerationRef,
    JournalAppendResult,
    JournalConflictError,
    JournalEvent,
    JournalPage,
    JournalScopeError,
    SemanticEventDraft,
    capture_journal_snapshot,
)
from unchain.persistence.sqlite_v2 import (
    SQLiteContextV2Store,
    SQLiteContextV2StoreIntegrityError,
)


class _Journal(BoundExecutionJournal):
    def __init__(self) -> None:
        super().__init__("execution-1")
        self.events = []
        self.operations = {}
        self.capture_calls = 0
        self.read_cursors = []

    def append(self, *, request):
        previous = self.operations.get(request.operation.operation_id)
        if previous is not None:
            previous_request, event = previous
            if previous_request != request:
                raise JournalConflictError("operation payload changed")
            return JournalAppendResult(
                event=event,
                cursor=EventCursor(event.store_seq, event.event_id),
                duplicate=True,
            )
        event = JournalEvent(
            event_id=request.event_id,
            event_type=request.event_type,
            attempt=request.attempt,
            operation=request.operation,
            store_seq=len(self.events) + 1,
            payload=request.payload,
            resource_refs=request.resource_refs,
        )
        self.events.append(event)
        self.operations[request.operation.operation_id] = (request, event)
        return JournalAppendResult(
            event=event,
            cursor=EventCursor(event.store_seq, event.event_id),
        )

    def read(self, *, after=None, limit=100):
        self.read_cursors.append(after)
        if after is None:
            start = 0
        else:
            if (
                after.store_seq > len(self.events)
                or self.events[after.store_seq - 1].event_id != after.event_id
            ):
                raise JournalScopeError("cursor is not in this journal")
            start = after.store_seq
        page = self.events[start : start + limit]
        return JournalPage(
            events=tuple(page),
            next_cursor=(
                EventCursor(page[-1].store_seq, page[-1].event_id)
                if page
                else after
            ),
            has_more=start + len(page) < len(self.events),
        )

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        self.capture_calls += 1
        snapshot = capture_journal_snapshot(
            execution_id=self.execution_id,
            events=tuple(self.events),
        )
        if snapshot.event_count > max_events:
            raise ValueError("journal snapshot exceeds test event limit")
        return snapshot

    def snapshot_prefix_is_current(self, *, snapshot, integrity_revision=None):
        return tuple(self.events[: snapshot.event_count]) == snapshot.events


def _append(journal: _Journal, event_id: str) -> None:
    attempt = AttemptRef(GenerationRef("execution-1", "generation-1"), "run-1")
    journal.append(
        request=SemanticEventDraft(
            event_id=event_id,
            event_type="run_started",
            attempt=attempt,
            operation_id=f"operation-{event_id}",
            payload={"run_id": "run-1", "status": event_id},
        ).to_append_request()
    )


def test_cache_reuses_a_verified_prefix_and_merges_durable_suffix() -> None:
    journal = _Journal()
    for index in range(1, 9):
        _append(journal, f"event-{index}")
    cache = RunLocalJournalViewCache(journal)

    first = cache.capture_snapshot()
    _append(journal, "event-9")
    second = cache.capture_snapshot()
    third = cache.capture_snapshot()

    assert first.event_count == 8
    assert tuple(event.event_id for event in second.events)[-1] == "event-9"
    assert third == second
    assert journal.capture_calls == 1
    assert journal.read_cursors == [first.high_water, second.high_water]
    assert cache.metrics().full_snapshot_reads == 1
    assert cache.metrics().tail_reads == 2
    assert cache.metrics().cache_hits == 2
    assert cache.metrics().refresh_fallbacks == 0


def test_cache_rehydrates_from_durable_authority_when_cursor_is_invalid() -> None:
    journal = _Journal()
    for index in range(1, 9):
        _append(journal, f"event-{index}")
    cache = RunLocalJournalViewCache(journal)
    cache.capture_snapshot()
    journal.events.clear()
    journal.operations.clear()
    _append(journal, "event-9")

    recovered = cache.capture_snapshot()

    assert tuple(event.event_id for event in recovered.events) == ("event-9",)
    assert journal.capture_calls == 2
    assert cache.metrics().refresh_fallbacks == 1


def _sqlite_journal(tmp_path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    )
    return store, store.bind_execution("execution-1")


def _append_sqlite(journal, event_id: str) -> None:
    attempt = AttemptRef(GenerationRef("execution-1", "generation-1"), "run-1")
    journal.append(
        request=SemanticEventDraft(
            event_id=event_id,
            event_type="run_started",
            attempt=attempt,
            operation_id=f"operation-{event_id}",
            payload={"run_id": "run-1", "status": event_id},
        ).to_append_request()
    )


def test_sqlite_cache_fails_closed_when_a_cached_prefix_is_deleted(tmp_path) -> None:
    store, journal = _sqlite_journal(tmp_path)
    for index in range(1, 9):
        _append_sqlite(journal, f"event-{index}")
    cache = RunLocalJournalViewCache(journal)
    cache.capture_snapshot()

    # Keep the high-water row in place: a tail-only read would otherwise appear valid.
    with store._transaction(immediate=True) as connection:
        connection.execute(
            "DELETE FROM events WHERE execution_id = ? AND store_seq = ?",
            ("execution-1", 1),
        )

    with pytest.raises(ValueError, match="complete contiguous prefix"):
        cache.capture_snapshot()


@pytest.mark.parametrize(
    "statement",
    (
        "UPDATE events SET event_json=CAST('{}' AS BLOB) WHERE store_seq=1",
        "UPDATE events SET generation_id='foreign-generation' WHERE store_seq=1",
        "UPDATE operations SET target_key='foreign-event' WHERE operation_id='operation-event-1'",
    ),
)
def test_sqlite_cache_fails_closed_on_mutated_durable_prefix(tmp_path, statement) -> None:
    store, journal = _sqlite_journal(tmp_path)
    for index in range(1, 9):
        _append_sqlite(journal, f"event-{index}")
    cache = RunLocalJournalViewCache(journal)
    cache.capture_snapshot()

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(statement)

    with pytest.raises(SQLiteContextV2StoreIntegrityError):
        cache.capture_snapshot()


@pytest.mark.parametrize(
    "statement",
    (
        "INSERT OR REPLACE INTO events SELECT execution_id, store_seq, event_id, "
        "generation_id, attempt_id, event_type, operation_id, CAST('{}' AS BLOB), "
        "event_sha256 FROM events WHERE store_seq = 1",
        "INSERT OR REPLACE INTO operations SELECT execution_id, operation_id, "
        "payload_sha256, target_kind, 'other-event' FROM operations "
        "WHERE operation_id = 'operation-event-1'",
    ),
    ids=("event-replacement", "operation-replacement"),
)
def test_sqlite_cache_fails_closed_on_replaced_durable_prefix(tmp_path, statement) -> None:
    store, journal = _sqlite_journal(tmp_path)
    for index in range(1, 9):
        _append_sqlite(journal, f"event-{index}")
    cache = RunLocalJournalViewCache(journal)
    cache.capture_snapshot()

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA recursive_triggers").fetchone()[0] == 0
        connection.execute(statement)

    with pytest.raises(SQLiteContextV2StoreIntegrityError):
        cache.capture_snapshot()


def test_sqlite_registry_does_not_retain_completed_run(tmp_path) -> None:
    store, journal = _sqlite_journal(tmp_path)
    _append_sqlite(journal, "event-1")
    cache = RunLocalJournalViewCache.for_journal(journal)
    cache.capture_snapshot()
    journal_ref, cache_ref = weakref.ref(journal), weakref.ref(cache)

    del cache, journal, store
    gc.collect()

    assert cache_ref() is None
    assert journal_ref() is None
