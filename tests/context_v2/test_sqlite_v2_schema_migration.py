from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

from unchain.journal.models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    GenerationRef,
    OperationRef,
    ResourceRef,
)
from unchain.journal.provider_result import ProviderTurnResultEnvelope
from unchain.kernel.types import ModelTurnResult
from unchain.persistence.sqlite_v2 import (
    SQLiteContextV2Store,
    SQLiteContextV2StoreIntegrityError,
)
from unchain.providers.request_lease import (
    ProviderRequestLeaseCoordinator,
    ProviderRequestStatus,
    ProviderTurnResultBinding,
)


ATTEMPT = AttemptRef(
    GenerationRef("execution-schema-migration", "generation-schema-migration"),
    "attempt-schema-migration",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _store(tmp_path) -> SQLiteContextV2Store:
    return SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )


def _operation(index: int) -> OperationRef:
    return OperationRef(f"schema-migration-operation-{index}", f"{index:x}" * 64)


def _claim(repository):
    return ProviderRequestLeaseCoordinator(repository).claim_initial(
        attempt=ATTEMPT,
        iteration=1,
        envelope_sha256="a" * 64,
        route="primary",
        route_sha256="b" * 64,
        operation=_operation(1),
    )


def _rewrite_all_leases_as_v1(database_path) -> None:
    with sqlite3.connect(database_path) as connection:
        rows = list(
            connection.execute(
                "SELECT rowid, lease_json FROM provider_request_lease_revisions"
            )
        )
        for rowid, encoded in rows:
            value = json.loads(bytes(encoded).decode("utf-8"))
            value["schema"] = "unchain.provider_request_lease.v1"
            del value["result_binding"]
            del value["predecessor_sha256"]
            legacy = _canonical_bytes(value)
            connection.execute(
                """
                UPDATE provider_request_lease_revisions
                SET lease_json = ?, lease_sha256 = ?
                WHERE rowid = ?
                """,
                (legacy, hashlib.sha256(legacy).hexdigest(), rowid),
            )
        connection.execute("DELETE FROM context_v2_schema WHERE version = 2")


def test_unknown_schema_version_is_rejected_before_database_mutation(tmp_path) -> None:
    database_path = tmp_path / "memory_v2" / "context_v2.sqlite3"
    database_path.parent.mkdir(parents=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE context_v2_schema(version INTEGER PRIMARY KEY)"
        )
        connection.execute("INSERT INTO context_v2_schema(version) VALUES (99)")

    with pytest.raises(SQLiteContextV2StoreIntegrityError, match="schema|version"):
        _store(tmp_path)

    with sqlite3.connect(database_path) as connection:
        versions = {
            row[0]
            for row in connection.execute("SELECT version FROM context_v2_schema")
        }
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert versions == {99}
    assert tables == {"context_v2_schema"}


@pytest.mark.parametrize("terminal", (False, True))
def test_v1_started_and_failed_leases_are_canonically_upgraded(
    tmp_path,
    terminal: bool,
) -> None:
    store = _store(tmp_path)
    repository = store.bind_execution(ATTEMPT.generation.execution_id)
    started = _claim(repository)
    expected = started
    if terminal:
        expected = ProviderRequestLeaseCoordinator(repository).record_failure(
            started,
            classification="transient",
            retryable=True,
            visible_output=False,
            operation=_operation(2),
        )
    _rewrite_all_leases_as_v1(store.database_path)

    reopened = _store(tmp_path)
    recovered = reopened.bind_execution(ATTEMPT.generation.execution_id).load(
        subject=started.subject
    )

    assert recovered == expected
    with sqlite3.connect(store.database_path) as connection:
        versions = {
            row[0]
            for row in connection.execute("SELECT version FROM context_v2_schema")
        }
        schemas = {
            json.loads(bytes(row[0]).decode("utf-8"))["schema"]
            for row in connection.execute(
                "SELECT lease_json FROM provider_request_lease_revisions"
            )
        }
    assert versions == {1, 2, 3}
    assert schemas == {"unchain.provider_request_lease.v2"}


def test_v1_completed_lease_fails_closed_and_rolls_back_entire_migration(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    repository = store.bind_execution(ATTEMPT.generation.execution_id)
    started = _claim(repository)
    ProviderRequestLeaseCoordinator(repository).record_completed_result(
        started,
        result_binding=ProviderTurnResultBinding(
            route_sha256=started.route_sha256,
            result_sha256="c" * 64,
            artifact=ArtifactRef(
                ResourceRef("artifact", "legacy-result", 1),
                "application/json",
                1,
                "d" * 64,
                "",
            ),
            cursor=EventCursor(1, "legacy-result-event"),
        ),
        visible_output=True,
        operation=_operation(2),
    )
    _rewrite_all_leases_as_v1(store.database_path)

    with pytest.raises(
        SQLiteContextV2StoreIntegrityError,
        match="legacy|completed|result",
    ):
        _store(tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        versions = {
            row[0]
            for row in connection.execute("SELECT version FROM context_v2_schema")
        }
        schemas = [
            json.loads(bytes(row[0]).decode("utf-8"))["schema"]
            for row in connection.execute(
                "SELECT lease_json FROM provider_request_lease_revisions ORDER BY revision"
            )
        ]
    assert versions == {1, 3}
    assert schemas == [
        "unchain.provider_request_lease.v1",
        "unchain.provider_request_lease.v1",
    ]
    assert started.status is ProviderRequestStatus.STARTED


@pytest.mark.parametrize(
    "corruption",
    ("missing_revision", "missing_head", "stale_head"),
)
def test_lease_head_corruption_fails_closed_before_restart_recovery(
    tmp_path,
    corruption: str,
) -> None:
    store = _store(tmp_path)
    repository = store.bind_execution(ATTEMPT.generation.execution_id)
    started = _claim(repository)
    if corruption == "stale_head":
        ProviderRequestLeaseCoordinator(repository).record_failure(
            started,
            classification="transient",
            retryable=True,
            visible_output=False,
            operation=_operation(2),
        )
    with sqlite3.connect(store.database_path) as connection:
        if corruption == "missing_revision":
            connection.execute(
                """
                UPDATE provider_request_lease_heads
                SET current_revision = 999
                WHERE execution_id = ?
                """,
                (ATTEMPT.generation.execution_id,),
            )
        elif corruption == "missing_head":
            connection.execute(
                """
                DELETE FROM provider_request_lease_heads
                WHERE execution_id = ?
                """,
                (ATTEMPT.generation.execution_id,),
            )
        else:
            connection.execute(
                """
                UPDATE provider_request_lease_heads
                SET current_revision = 1
                WHERE execution_id = ?
                """,
                (ATTEMPT.generation.execution_id,),
            )

    with pytest.raises(
        SQLiteContextV2StoreIntegrityError,
        match="lease|head|revision",
    ):
        _store(tmp_path)

    with pytest.raises(Exception, match="lease|head|revision|durable"):
        ProviderRequestLeaseCoordinator(repository).recover(started.subject)


@pytest.mark.parametrize("surviving_evidence", ("lease_operation", "result_receipt"))
def test_removed_lease_rows_cannot_erase_surviving_send_evidence(
    tmp_path,
    surviving_evidence: str,
) -> None:
    store = _store(tmp_path)
    repository = store.bind_execution(ATTEMPT.generation.execution_id)
    started = _claim(repository)
    if surviving_evidence == "result_receipt":
        envelope = ProviderTurnResultEnvelope.from_model_turn_result(
            subject=started.subject,
            route_sha256=started.route_sha256,
            visible_output=True,
            result=ModelTurnResult(
                assistant_messages=[{"role": "assistant", "content": "durable"}],
                tool_calls=[],
                final_text="durable",
                response_id="schema-migration-result",
            ),
        )
        repository.persist_provider_turn_result_cas(
            started_lease=started,
            envelope=envelope,
            artifact_operation=_operation(3),
            event_operation=_operation(4),
            event_id="schema-migration-result-event",
        )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            "DELETE FROM provider_request_lease_heads WHERE execution_id = ?",
            (ATTEMPT.generation.execution_id,),
        )
        connection.execute(
            "DELETE FROM provider_request_lease_revisions WHERE execution_id = ?",
            (ATTEMPT.generation.execution_id,),
        )
        if surviving_evidence == "result_receipt":
            connection.execute(
                """
                DELETE FROM operations
                WHERE execution_id = ? AND target_kind = 'provider_request_lease'
                """,
                (ATTEMPT.generation.execution_id,),
            )

    with pytest.raises(
        SQLiteContextV2StoreIntegrityError,
        match="lease|evidence|result|operation",
    ):
        _store(tmp_path)

    with pytest.raises(Exception, match="lease|evidence|result|operation|durable"):
        ProviderRequestLeaseCoordinator(repository).recover(started.subject)
