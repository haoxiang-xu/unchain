"""The existing-only capability can verify graph bytes without creating rows."""
from unchain.persistence.sqlite_v2 import SQLiteContextV2ReadOnlyJournal, _SQLiteBoundContextV2Repository
from dataclasses import replace
from hashlib import sha256
import pytest
from unchain.journal import OperationRef
from unchain.persistence.sqlite_v2 import (SQLiteContextV2Store,
    SQLiteContextV2StoreError, SQLiteContextV2StoreIntegrityError,
    open_existing_execution_journal_readonly)
from unchain.persistence.sqlite_generation_lifecycle_v2 import (
    HostGenerationAttemptBindingIntent, HostGenerationAttemptBindingRequest,
    HostGenerationConflict, HostGenerationTransition, HostGenerationTransitionKind,
    HostGenerationTransitionRequest, SQLiteHostGenerationLifecycleV2,
    build_host_generation_attempt_binding_operation, build_host_generation_transition_operation,
    read_existing_host_generation_binding)


def test_readonly_graph_capabilities_are_exposed_without_constructing_store():
    assert callable(getattr(SQLiteContextV2ReadOnlyJournal, "artifact_id_for", None))
    assert callable(getattr(SQLiteContextV2ReadOnlyJournal, "read_full_verified", None))
    # Identity calculation is pure and shared with the actual writable adapter.
    journal = object.__new__(SQLiteContextV2ReadOnlyJournal)
    journal._execution_id = "read-only-graph"
    assert journal.artifact_id_for(logical_kind="artifact", logical_key="key") == _SQLiteBoundContextV2Repository._artifact_id(
        execution_id="read-only-graph", logical_kind="artifact", logical_key="key")


def _bytes(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()}


def test_existing_reader_verifies_real_artifact_and_rejects_corruption_without_writes(tmp_path):
    store = SQLiteContextV2Store(database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects")
    writer = store.bind_execution("reader-execution")
    content = b'{"fixed":"definition"}'
    artifact = writer.put(content=content, media_type="application/json", preview="",
        operation=OperationRef("reader-artifact", "0" * 64))
    before = _bytes(tmp_path)
    reader = open_existing_execution_journal_readonly(database_path=store.database_path,
        object_directory=store.object_directory, execution_id="reader-execution")
    assert reader.read_full_verified(artifact=artifact) == content
    assert reader.artifact_id_for(logical_kind="artifact", logical_key="fixed") == writer.artifact_id_for(
        logical_kind="artifact", logical_key="fixed")
    assert _bytes(tmp_path) == before
    (store.object_directory / artifact.sha256).write_bytes(b"corrupt")
    corrupted = _bytes(tmp_path)
    with pytest.raises(SQLiteContextV2StoreIntegrityError):
        reader.read_full_verified(artifact=artifact)
    assert _bytes(tmp_path) == corrupted


def test_existing_generation_read_never_initializes_absent_database(tmp_path):
    with pytest.raises(SQLiteContextV2StoreError):
        read_existing_host_generation_binding(database_path=tmp_path / "absent" / "context.sqlite3",
            owner_chat_id="owner", execution_id="execution", session_id="session", attempt_id="root")
    assert not (tmp_path / "absent").exists()


def test_existing_generation_read_preserves_real_head_binding_and_detects_rebase(tmp_path):
    store = SQLiteContextV2Store(database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects")
    lifecycle = SQLiteHostGenerationLifecycleV2(store)
    transition = HostGenerationTransition(owner_chat_id="owner", execution_id="execution",
        session_id="session", generation_id="generation-1", kind=HostGenerationTransitionKind.INITIAL,
        previous_generation_id="", expected_revision=0)
    lifecycle.advance(HostGenerationTransitionRequest(transition,
        build_host_generation_transition_operation(operation_id="initial", transition=transition)))
    intent = HostGenerationAttemptBindingIntent(owner_chat_id="owner", execution_id="execution",
        session_id="session", generation_id="generation-1", attempt_id="root", expected_revision=1)
    lifecycle.bind_current_attempt(HostGenerationAttemptBindingRequest(intent,
        build_host_generation_attempt_binding_operation(operation_id="root-bind", intent=intent)))
    scope = dict(database_path=store.database_path, owner_chat_id="owner", execution_id="execution",
        session_id="session", attempt_id="root")
    before = _bytes(tmp_path)
    head, binding = read_existing_host_generation_binding(**scope)
    assert head.current_generation_id == binding.generation_id == "generation-1"
    with pytest.raises(HostGenerationConflict):
        read_existing_host_generation_binding(**dict(scope, session_id="foreign"))
    assert read_existing_host_generation_binding(**dict(scope, owner_chat_id="foreign")) == (None, None)
    assert _bytes(tmp_path) == before
    edit = replace(transition, generation_id="generation-2", kind=HostGenerationTransitionKind.EDIT,
        previous_generation_id="generation-1", expected_revision=1)
    lifecycle.advance(HostGenerationTransitionRequest(edit,
        build_host_generation_transition_operation(operation_id="edit", transition=edit)))
    before = _bytes(tmp_path)
    head, binding = read_existing_host_generation_binding(**scope)
    assert head.current_generation_id == "generation-2" and binding.generation_id == "generation-1"
    assert _bytes(tmp_path) == before
