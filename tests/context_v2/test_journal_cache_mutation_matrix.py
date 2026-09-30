"""Durable mutations must invalidate every affected execution's read cache."""

import sqlite3

import pytest

from unchain.context.journal_view_cache import RunLocalJournalViewCache
from unchain.journal import AttemptRef, GenerationRef, SemanticEventDraft
from unchain.persistence.sqlite_v2 import (
    SQLiteContextV2Store,
    SQLiteContextV2StoreIntegrityError,
)


def _append(journal, index):
    request = SemanticEventDraft(
        event_id=f"{journal.execution_id}-event-{index}",
        event_type="run_started",
        attempt=AttemptRef(GenerationRef(journal.execution_id, "generation"), "run"),
        operation_id=f"operation-{index}",
        payload={"run_id": "run", "status": "started"},
    ).to_append_request()
    return journal.append(request=request)


def _fixture(tmp_path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    )
    journal = store.bind_execution("destination")
    other = store.bind_execution("source")
    for index in range(1, 9):
        _append(journal, index)
    _append(other, 1)
    cache = RunLocalJournalViewCache(journal)
    cache.capture_snapshot()
    return store, journal, other, cache


@pytest.mark.parametrize("table", ["events", "operations"])
@pytest.mark.parametrize("recursive", [0, 1])
@pytest.mark.parametrize("cross_execution", [False, True])
def test_update_replace_invalidates_affected_executions(
    tmp_path, table, recursive, cross_execution,
):
    store, journal, other, cache = _fixture(tmp_path)
    before = journal.snapshot_integrity_revision()
    other_before = other.snapshot_integrity_revision()
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(f"PRAGMA recursive_triggers={recursive}")
        if cross_execution:
            connection.execute(
                f"UPDATE OR REPLACE {table} SET execution_id='destination' "
                "WHERE execution_id='source' AND operation_id='operation-1'"
            )
        else:
            assignment = (
                "event_json=CAST('{}' AS BLOB)" if table == "events"
                else "target_key='wrong-event'"
            )
            connection.execute(
                f"UPDATE OR REPLACE {table} SET {assignment} "
                "WHERE execution_id='destination' AND operation_id='operation-1'"
            )
    assert journal.snapshot_integrity_revision() > before
    assert (other.snapshot_integrity_revision() > other_before) == cross_execution
    with pytest.raises(SQLiteContextV2StoreIntegrityError):
        journal.capture_snapshot()
    with pytest.raises(SQLiteContextV2StoreIntegrityError):
        cache.capture_snapshot()


@pytest.mark.parametrize("entity,table", [("event", "events"), ("operation", "operations")])
def test_reopen_upgrades_old_v3_trigger_and_invalidates_old_cache(tmp_path, entity, table):
    store, journal, other, cache = _fixture(tmp_path)
    before = journal.snapshot_integrity_revision()
    # Simulate a v3 store from the preceding candidate. Corruption that occurred
    # under its old trigger must not stay invisible to a surviving warm cache.
    name = f"context_v2_{entity}_integrity_update"
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(f"DROP TRIGGER {name}")
        connection.execute(f"""CREATE TRIGGER {name} AFTER UPDATE ON {table}
            BEGIN UPDATE executions SET integrity_revision=integrity_revision+1
            WHERE execution_id=OLD.execution_id; END""")
        connection.execute(
            f"UPDATE OR REPLACE {table} SET execution_id='destination' "
            "WHERE execution_id='source' AND operation_id='operation-1'"
        )
    assert journal.snapshot_integrity_revision() == before
    SQLiteContextV2Store(
        database_path=store.database_path, object_directory=store.object_directory,
    )
    assert journal.snapshot_integrity_revision() > before
    with pytest.raises(SQLiteContextV2StoreIntegrityError):
        cache.capture_snapshot()
    # Ensure the new definition is effective for later mutations too.
    after_upgrade = journal.snapshot_integrity_revision()
    _append(other, 2)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            f"UPDATE OR REPLACE {table} SET execution_id='destination' "
            "WHERE execution_id='source' AND operation_id='operation-2'"
        )
    assert journal.snapshot_integrity_revision() > after_upgrade


def test_append_replay_and_unchanged_reopen_keep_warm_cache(tmp_path):
    store, journal, other, cache = _fixture(tmp_path)
    before = journal.snapshot_integrity_revision()
    _append(journal, 9)
    assert _append(journal, 9).duplicate
    SQLiteContextV2Store(
        database_path=store.database_path, object_directory=store.object_directory,
    )
    assert journal.snapshot_integrity_revision() == before
    assert cache.capture_snapshot() == journal.capture_snapshot()
    assert cache.metrics().full_snapshot_reads == 1
    assert cache.metrics().tail_reads == 1
