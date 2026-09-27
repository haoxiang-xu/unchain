from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from unchain.context import (
    ArtifactService,
    CheckpointWriteStatus,
    ContextBuildEnvelope,
    ContextConflictError,
    ContextRepositoryError,
    ContextScopeError,
    resolve_context_budget,
)
from unchain.journal import (
    AttemptRef,
    EventCursor,
    EventRange,
    GenerationRef,
    JournalAppendRequest,
    OperationRef,
    ResourceRef,
)
from unchain.journal.runtime import build_operation_ref
from unchain.persistence.sqlite_context_compiler_v2 import (
    SQLiteContextCompilerV2Store,
)
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


def _operation(operation_id: str, payload: dict) -> OperationRef:
    return build_operation_ref(
        operation_id,
        domain="test.sqlite_context_compiler_v2",
        payload=payload,
    )


def _append_trigger(repository, *, event_id: str = "event-current", store_seq: int = 1):
    del store_seq
    attempt = AttemptRef(
        GenerationRef("execution-a", "generation-a"),
        "attempt-a",
    )
    payload = {
        "run_id": attempt.attempt_id,
        "message": {"role": "user", "content": "current"},
    }
    operation = _operation("message-current", payload)
    return repository.append(
        request=JournalAppendRequest(
            event_id=event_id,
            event_type="message.user",
            attempt=attempt,
            operation=operation,
            payload=payload,
        )
    ).event


def _open(root: Path):
    database_path = root / "context_v2.sqlite3"
    object_directory = root / "objects"
    context_store = SQLiteContextV2Store(
        database_path=database_path,
        object_directory=object_directory,
    )
    journal = context_store.bind_execution("execution-a")
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    compiler_store = SQLiteContextCompilerV2Store(context_store=context_store)
    capabilities = compiler_store.bind_execution(
        "execution-a",
        artifacts=artifacts,
    )
    return database_path, journal, capabilities


def _envelope(*, build_id: str = "build-a") -> ContextBuildEnvelope:
    return ContextBuildEnvelope(
        build_id=build_id,
        execution_id="execution-a",
        generation_id="generation-a",
        attempt_id="attempt-a",
        provider="openai",
        model="synthetic",
        budget=resolve_context_budget(context_window_tokens=8_192),
        estimated_input_tokens=512,
    )


def test_checkpoint_prepare_commit_and_read_survive_cold_restart(
    tmp_path: Path,
) -> None:
    _, journal, capabilities = _open(tmp_path)
    first = _append_trigger(journal)
    source_range = EventRange(
        EventCursor(first.store_seq, first.event_id),
        EventCursor(first.store_seq, first.event_id),
    )
    operation = _operation(
        "checkpoint-a",
        {"source_range": source_range.to_dict(), "summary_sha256": "a" * 64},
    )
    prospective_ref = capabilities.checkpoints.checkpoint_ref_for(
        operation=operation
    )

    prepared = capabilities.checkpoints.prepare(
        source_range=source_range,
        summary='{"decision":"keep full history"}',
        refs=(),
        operation=operation,
    )
    replay = capabilities.checkpoints.prepare(
        source_range=source_range,
        summary='{"decision":"keep full history"}',
        refs=(),
        operation=operation,
    )

    assert prepared.status is CheckpointWriteStatus.PREPARED
    assert prepared.checkpoint_ref == prospective_ref
    assert prepared.duplicate is False
    assert replay.checkpoint_ref == prepared.checkpoint_ref
    assert replay.duplicate is True
    with pytest.raises(ContextScopeError, match="committed"):
        capabilities.checkpoints.read(ref=prepared.checkpoint_ref)

    committed = capabilities.checkpoints.commit(prepared=prepared)
    assert committed.status is CheckpointWriteStatus.COMMITTED
    assert capabilities.checkpoints.list_committed_refs() == (
        committed.checkpoint_ref,
    )
    assert capabilities.checkpoints.read(ref=committed.checkpoint_ref) == (
        b'{"decision":"keep full history"}'
    )

    _, _, reopened = _open(tmp_path)
    recovered = reopened.checkpoints.get_by_operation(operation=operation)
    recommitted = reopened.checkpoints.commit(prepared=committed)

    assert recovered is not None
    assert reopened.checkpoints.checkpoint_ref_for(
        operation=operation
    ) == prospective_ref
    assert recovered.status is CheckpointWriteStatus.COMMITTED
    assert recovered.checkpoint_ref == committed.checkpoint_ref
    assert recommitted.duplicate is True
    assert reopened.checkpoints.list_committed_refs() == (
        committed.checkpoint_ref,
    )
    assert (
        reopened.checkpoints.read(
            ref=committed.checkpoint_ref,
            offset=13,
            limit=4,
        )
        == b"keep"
    )


def test_checkpoint_operation_and_semantic_drift_fail_closed(tmp_path: Path) -> None:
    _, journal, capabilities = _open(tmp_path)
    event = _append_trigger(journal)
    source_range = EventRange(
        EventCursor(event.store_seq, event.event_id),
        EventCursor(event.store_seq, event.event_id),
    )
    operation = _operation("checkpoint-a", {"version": 1})
    capabilities.checkpoints.prepare(
        source_range=source_range,
        summary="stable",
        refs=(),
        operation=operation,
    )

    with pytest.raises(ContextConflictError, match="operation"):
        capabilities.checkpoints.get_by_operation(
            operation=OperationRef(operation.operation_id, "f" * 64)
        )
    with pytest.raises(ContextConflictError, match="payload"):
        capabilities.checkpoints.prepare(
            source_range=source_range,
            summary="changed",
            refs=(),
            operation=operation,
        )


def test_context_build_claims_one_trigger_and_survives_restart(tmp_path: Path) -> None:
    _, journal, capabilities = _open(tmp_path)
    trigger = _append_trigger(journal)
    cursor = EventCursor(trigger.store_seq, trigger.event_id)
    envelope = _envelope()
    operation = _operation(
        "build-a",
        {"envelope": envelope.to_dict(), "trigger": cursor.to_dict()},
    )

    created = capabilities.context_builds.record(
        envelope=envelope,
        operation=operation,
        trigger_cursor=cursor,
    )
    replay = capabilities.context_builds.record(
        envelope=envelope,
        operation=operation,
        trigger_cursor=cursor,
    )

    assert created.duplicate is False
    assert replay.duplicate is True

    _, _, reopened = _open(tmp_path)
    assert reopened.context_builds.get_by_operation(operation=operation) == created
    assert reopened.context_builds.get_by_trigger(trigger_cursor=cursor) == created
    assert reopened.context_builds.latest(generation_id="generation-a") == envelope

    other_envelope = _envelope(build_id="build-other")
    with pytest.raises(ContextConflictError, match="trigger"):
        reopened.context_builds.record(
            envelope=other_envelope,
            operation=_operation("build-other", {"version": 2}),
            trigger_cursor=cursor,
        )


def test_compiler_ports_reject_foreign_scope_and_unknown_journal_ranges(
    tmp_path: Path,
) -> None:
    _, journal, capabilities = _open(tmp_path)
    trigger = _append_trigger(journal)
    cursor = EventCursor(trigger.store_seq, trigger.event_id)

    foreign = ContextBuildEnvelope(
        build_id="build-foreign",
        execution_id="execution-foreign",
        generation_id="generation-a",
        attempt_id="attempt-a",
        provider="openai",
        model="synthetic",
        budget=resolve_context_budget(context_window_tokens=8_192),
    )
    with pytest.raises(ContextScopeError, match="execution"):
        capabilities.context_builds.record(
            envelope=foreign,
            operation=_operation("build-foreign", {"version": 1}),
            trigger_cursor=cursor,
        )

    unknown = EventCursor(cursor.store_seq + 1, "event-unknown")
    with pytest.raises(ContextScopeError, match="journal"):
        capabilities.context_builds.record(
            envelope=_envelope(),
            operation=_operation("build-unknown", {"version": 1}),
            trigger_cursor=unknown,
        )
    with pytest.raises(ContextScopeError, match="journal"):
        capabilities.checkpoints.prepare(
            source_range=EventRange(unknown, unknown),
            summary="missing",
            refs=(ResourceRef("artifact", "missing", 1),),
            operation=_operation("checkpoint-unknown", {"version": 1}),
        )


def test_checkpoint_insert_failure_leaves_no_receipt_and_retry_is_safe(
    tmp_path: Path,
) -> None:
    database_path, journal, capabilities = _open(tmp_path)
    event = _append_trigger(journal)
    source_range = EventRange(
        EventCursor(event.store_seq, event.event_id),
        EventCursor(event.store_seq, event.event_id),
    )
    operation = _operation("checkpoint-fault", {"version": 1})
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TRIGGER fail_checkpoint_insert
        BEFORE INSERT ON checkpoints
        BEGIN
            SELECT RAISE(ABORT, 'injected checkpoint failure');
        END;
        """
    )
    connection.close()

    with pytest.raises(ContextRepositoryError, match="checkpoint persistence"):
        capabilities.checkpoints.prepare(
            source_range=source_range,
            summary="durable before reference",
            refs=(),
            operation=operation,
        )
    assert capabilities.checkpoints.get_by_operation(operation=operation) is None

    connection = sqlite3.connect(database_path)
    connection.execute("DROP TRIGGER fail_checkpoint_insert")
    connection.close()
    retried = capabilities.checkpoints.prepare(
        source_range=source_range,
        summary="durable before reference",
        refs=(),
        operation=operation,
    )
    assert retried.duplicate is False


def test_context_build_write_is_atomic_under_sqlite_failure(tmp_path: Path) -> None:
    database_path, journal, capabilities = _open(tmp_path)
    trigger = _append_trigger(journal)
    cursor = EventCursor(trigger.store_seq, trigger.event_id)
    envelope = _envelope()
    operation = _operation("build-fault", {"version": 1})
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TRIGGER fail_context_build_insert
        BEFORE INSERT ON context_builds
        BEGIN
            SELECT RAISE(ABORT, 'injected build failure');
        END;
        """
    )
    connection.close()

    with pytest.raises(ContextRepositoryError, match="build persistence"):
        capabilities.context_builds.record(
            envelope=envelope,
            operation=operation,
            trigger_cursor=cursor,
        )
    assert capabilities.context_builds.get_by_operation(operation=operation) is None

    connection = sqlite3.connect(database_path)
    connection.execute("DROP TRIGGER fail_context_build_insert")
    connection.close()
    receipt = capabilities.context_builds.record(
        envelope=envelope,
        operation=operation,
        trigger_cursor=cursor,
    )
    assert receipt.duplicate is False


def test_context_build_digest_corruption_is_detected_after_restart(
    tmp_path: Path,
) -> None:
    database_path, journal, capabilities = _open(tmp_path)
    trigger = _append_trigger(journal)
    cursor = EventCursor(trigger.store_seq, trigger.event_id)
    envelope = _envelope()
    operation = _operation("build-corrupt", {"version": 1})
    capabilities.context_builds.record(
        envelope=envelope,
        operation=operation,
        trigger_cursor=cursor,
    )
    connection = sqlite3.connect(database_path)
    connection.execute(
        "UPDATE context_builds SET envelope_sha256 = ? WHERE operation_id = ?",
        (hashlib.sha256(b"changed").hexdigest(), operation.operation_id),
    )
    connection.commit()
    connection.close()

    _, _, reopened = _open(tmp_path)
    with pytest.raises(ContextRepositoryError, match="digest"):
        reopened.context_builds.get_by_operation(operation=operation)
