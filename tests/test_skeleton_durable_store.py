"""SQLite admission and replay tests for the bounded CP-10 store."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from unchain.journal import (
    AttemptRef, GenerationRef, JournalAppendRequest, JournalConflictError,
    OperationRef, ResourceRef, build_operation_ref,
)
from unchain.skeleton.durable_protocol import canonical_bytes, execution_artifacts
from unchain.skeleton.durable_store import DurableStoreError, WorkflowExecutionStore
from unchain.skeleton.preparation import prepare_workflow
from unchain.skeleton.recovery import workflow_definition_digest


def _plan():
    fixture = Path(__file__).parent / "fixtures" / "skeleton_recovery" / "workflow.json"
    return prepare_workflow(json.loads(fixture.read_text()))


def _journal(tmp_path, execution_id="exec-store"):
    from unchain.persistence import SQLiteContextV2Store

    repository = SQLiteContextV2Store(
        database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects",
    ).bind_execution(execution_id)
    return repository


def _attempt(execution_id="exec-store"):
    return AttemptRef(GenerationRef(execution_id, "generation-store"), "attempt-store")


def _state(plan, attempt, graph_id, inputs_ref, **changes):
    value = {
        "schema": "unchain.skeleton_execution_state.v2",
        "graph_id": graph_id,
        "attempt": attempt.to_dict(),
        "definition_digest": workflow_definition_digest(plan),
        "call_stack": [{
            "execution_id": attempt.generation.execution_id,
            "graph_invocation_id": graph_id,
            "attempt_id": attempt.attempt_id,
            "definition_id": plan.document_id,
            "definition_digest": workflow_definition_digest(plan),
        }],
        "inputs_ref": inputs_ref,
        "attachments": {},
        "completed": [],
        "active": None,
        "status": "ready",
        "paused": False,
        "error": None,
        "root_committed": False,
    }
    value.update(changes)
    return value


def _store(tmp_path, graph_id="graph-store"):
    plan, attempt, journal = _plan(), _attempt(), _journal(tmp_path)
    return WorkflowExecutionStore(journal, plan, graph_id=graph_id, attempt=attempt)


def _raw_request(store, state, *, caller_operation, revision, previous_event_id, resource_refs=None):
    raw = canonical_bytes(state)
    graph_key = hashlib.sha256(store.graph_id.encode("utf-8", "surrogatepass")).hexdigest()
    payload = {
        "schema": "unchain.skeleton_workflow_execution_journal.v2",
        "graph_key": graph_key,
        "revision": revision,
        "previous_event_id": previous_event_id,
        "encoded_state": base64.b64encode(raw).decode("ascii"),
        "state_sha256": hashlib.sha256(raw).hexdigest(),
    }
    operation_id = "skeleton.execution." + graph_key + "." + hashlib.sha256(caller_operation.encode()).hexdigest()
    if resource_refs is None:
        resource_refs = (ResourceRef("artifact", state["inputs_ref"]["ref"]["id"], 1),)
    return JournalAppendRequest(
        event_id="event." + operation_id,
        event_type="skeleton.workflow_execution.saved",
        attempt=store.attempt,
        operation=build_operation_ref(operation_id, domain="skeleton.workflow_execution.saved", payload=payload),
        payload=payload,
        resource_refs=resource_refs,
    )


def _legacy_envelope(plan, attempt, graph_id):
    invocation = {"execution_id": attempt.generation.execution_id, "graph_invocation_id": graph_id, "attempt_id": attempt.attempt_id}
    from unchain.skeleton.recovery import build_workflow_recovery_envelope

    return build_workflow_recovery_envelope(
        plan, invocation=invocation,
        call_stack=[{**invocation, "definition_id": plan.document_id, "definition_digest": workflow_definition_digest(plan)}],
        completed_nodes=[
            {"node_id": "start", "node_invocation_id": "start-call", "attempt_id": "start-attempt", "input_receipt_id": "start-input", "output_receipt_id": "start-output"},
            {"node_id": "if", "node_invocation_id": "if-call", "attempt_id": "if-attempt", "input_receipt_id": "if-input", "output_receipt_id": "if-output"},
            {"node_id": "left", "node_invocation_id": "left-call", "attempt_id": "left-attempt", "input_receipt_id": "left-input", "output_receipt_id": "left-output"},
        ],
        selected_edges=[
            {"edge_id": "start-if", "source_node_id": "start", "source_port_id": "out", "target_node_id": "if", "target_port_id": "in"},
            {"edge_id": "if-left", "source_node_id": "if", "source_port_id": "true", "target_node_id": "left", "target_port_id": "in"},
            {"edge_id": "left-join", "source_node_id": "left", "source_port_id": "out", "target_node_id": "join", "target_port_id": "in"},
        ],
        active={"kind": "waiting_interaction", "node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "interaction_id": "interaction", "request_receipt_id": "request"},
    )


def test_rejects_invalid_admission_without_event_or_artifact(tmp_path):
    """The write admission guard was written before the valid write path."""
    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    state = _state(store._plan, store._attempt, store._graph_id, descriptor, status="running")
    with pytest.raises(DurableStoreError, match="active record required"):
        store.save(state, expected_revision=0, operation_id="bad-admission", artifacts=(pending,))
    assert store._journal.capture_snapshot().events == ()


def _state_with_malformed_code_profile_kind(store, kind):
    inputs_ref, inputs_pending = store.prepare_json({"input": 1}, key="root.inputs")
    source = b"def main(input, context): return {'answer': 'ok'}\n"
    source_ref, source_pending = store.prepare_bytes(
        source, media_type="text/x-python", key="code.left.source",
    )
    binding = {
        "schema": "unchain.skeleton.code-binding.v1",
        "source_ref": source_ref,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "profile": {
            "id": "builtin", "kind": kind, "command": ["/python"], "sha256": "a" * 64,
        },
        "artifacts": [],
    }
    state = _state(
        store.plan, store.attempt, store.graph_id, inputs_ref,
        code_bindings={"left": binding},
    )
    return state, (inputs_pending, source_pending)


@pytest.mark.parametrize("kind", ([], {}))
def test_malformed_code_profile_kind_rejects_before_write_with_structured_error(tmp_path, kind):
    store = _store(tmp_path)
    state, pending = _state_with_malformed_code_profile_kind(store, kind)
    with pytest.raises(DurableStoreError) as caught:
        store.save(state, expected_revision=0, operation_id="invalid-profile-kind", artifacts=pending)
    assert caught.value.code == "invalid_type"
    assert caught.value.path[-2:] == ("profile", "kind")
    assert store.journal.capture_snapshot().events == ()


@pytest.mark.parametrize("kind", ([], {}))
def test_cold_load_rejects_malformed_code_profile_kind_with_structured_error(tmp_path, kind):
    store = _store(tmp_path)
    state, pending = _state_with_malformed_code_profile_kind(store, kind)
    request = _raw_request(
        store, state, caller_operation="malformed-profile-kind", revision=1,
        previous_event_id=None,
        resource_refs=tuple(ref.ref for ref in execution_artifacts(state)),
    )
    store.journal.append_with_artifacts(request=request, artifacts=pending)
    cold = _store(tmp_path)
    with pytest.raises(DurableStoreError) as caught:
        cold.load()
    assert caught.value.code == "invalid_type"
    assert caught.value.path[-2:] == ("profile", "kind")


def test_save_load_exact_replay_after_progress_and_atomic_artifact(tmp_path):
    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(store._plan, store._attempt, store._graph_id, descriptor)
    first = store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    assert first.revision == 1
    assert store.read_json(descriptor) == {"input": 1}

    progressed = _state(store._plan, store._attempt, store._graph_id, descriptor, paused=True)
    second = store.save(progressed, expected_revision=1, operation_id="pause")
    assert second.revision == 2
    replay = store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    assert replay == first
    assert store.load() == second


def test_real_sqlite_cas_race_rejects_stale_writer(tmp_path):
    one, two = _store(tmp_path), _store(tmp_path)
    descriptor, pending = one.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(one._plan, one._attempt, one._graph_id, descriptor)
    one.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    with pytest.raises(JournalConflictError):
        two.save(initial, expected_revision=0, operation_id="different-stale", artifacts=())


def test_concurrent_same_revision_has_one_winner(tmp_path):
    one, two = _store(tmp_path), _store(tmp_path)
    descriptor, pending = one.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(one.plan, one.attempt, one.graph_id, descriptor)
    one.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    candidate = _state(one.plan, one.attempt, one.graph_id, descriptor, paused=True)
    gate, outcomes = threading.Barrier(2), []

    def writer(store, operation):
        gate.wait()
        try:
            outcomes.append(store.save(candidate, expected_revision=1, operation_id=operation))
        except JournalConflictError:
            outcomes.append("conflict")

    first = threading.Thread(target=writer, args=(one, "race-one"))
    second = threading.Thread(target=writer, args=(two, "race-two"))
    first.start(); second.start(); first.join(); second.join()
    assert sum(value == "conflict" for value in outcomes) == 1
    assert sum(getattr(value, "revision", None) == 2 for value in outcomes) == 1


def test_rollback_after_artifact_row_leaves_no_receipt(tmp_path, monkeypatch):
    from unchain.persistence.sqlite_v2 import _SQLiteBoundContextV2Repository

    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(store.plan, store.attempt, store.graph_id, descriptor)
    original = _SQLiteBoundContextV2Repository._append_with_connection

    def explode(self, connection, request):
        assert connection.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
        raise RuntimeError("between artifact and event")

    monkeypatch.setattr(_SQLiteBoundContextV2Repository, "_append_with_connection", explode)
    with pytest.raises(RuntimeError, match="between artifact"):
        store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    monkeypatch.setattr(_SQLiteBoundContextV2Repository, "_append_with_connection", original)
    assert store._journal.capture_snapshot().events == ()
    assert store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,)).revision == 1


def test_real_sqlite_loads_in_a_fresh_process(tmp_path):
    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(store.plan, store.attempt, store.graph_id, descriptor)
    store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    source_root = Path(__file__).parents[1] / "src"
    fixture = Path(__file__).parent / "fixtures" / "skeleton_recovery" / "workflow.json"
    script = f"""
import json
from pathlib import Path
from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.preparation import prepare_workflow
plan = prepare_workflow(json.loads(Path({str(fixture)!r}).read_text()))
journal = SQLiteContextV2Store(database_path={str(tmp_path / 'context.sqlite3')!r}, object_directory={str(tmp_path / 'objects')!r}).bind_execution('exec-store')
store = WorkflowExecutionStore(journal, plan, graph_id='graph-store', attempt=AttemptRef(GenerationRef('exec-store', 'generation-store'), 'attempt-store'))
assert store.load().revision == 1
"""
    environment = dict(os.environ, PYTHONPATH=str(source_root))
    result = subprocess.run(
        [sys.executable, "-c", script], env=environment, text=True,
        capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_rejects_unpersisted_reference_and_tampered_content(tmp_path):
    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(store._plan, store._attempt, store._graph_id, descriptor)
    with pytest.raises(DurableStoreError):
        store.save(initial, expected_revision=0, operation_id="missing", artifacts=())
    store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    object_path = store._journal._store.object_directory / descriptor["sha256"]
    object_path.write_bytes(b'{"input":2}')
    with pytest.raises(DurableStoreError):
        store.load()


def test_rejects_non_object_json_for_a_receipt_before_write(tmp_path):
    store = _store(tmp_path)
    descriptor, pending = store.prepare_bytes(b"[]", media_type="application/json", key="root.inputs")
    state = _state(store.plan, store.attempt, store.graph_id, descriptor)
    with pytest.raises(DurableStoreError, match="must contain an object"):
        store.save(state, expected_revision=0, operation_id="array-input", artifacts=(pending,))
    assert store._journal.capture_snapshot().events == ()


def test_rejects_valid_same_execution_artifact_from_another_graph(tmp_path):
    journal, plan, attempt = _journal(tmp_path), _plan(), _attempt()
    left = WorkflowExecutionStore(journal, plan, graph_id="left-graph", attempt=attempt)
    right = WorkflowExecutionStore(journal, plan, graph_id="right-graph", attempt=attempt)
    foreign_ref, foreign_pending = right.prepare_json({"input": 1}, key="root.inputs")
    # The descriptor has the correct execution scope and bytes, but the graph
    # qualifier in its operation-derived id binds it to the other invocation.
    borrowed = _state(plan, attempt, "left-graph", foreign_ref)
    with pytest.raises(DurableStoreError, match="receipt slot"):
        left.save(borrowed, expected_revision=0, operation_id="borrow", artifacts=(foreign_pending,))


def test_rejects_foreign_header_and_missing_resource_refs_before_filtering(tmp_path):
    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(store.plan, store.attempt, store.graph_id, descriptor)
    first = store.save(initial, expected_revision=0, operation_id="create", artifacts=(pending,))
    request = _raw_request(store, initial, caller_operation="foreign-header", revision=2,
                           previous_event_id=first.event_id)
    broken = dict(request.payload); broken["graph_key"] = "0" * 64
    operation = build_operation_ref(request.operation.operation_id, domain="skeleton.workflow_execution.saved", payload=broken)
    store._journal.append(request=JournalAppendRequest(
        event_id=request.event_id, event_type=request.event_type, attempt=request.attempt,
        operation=operation, payload=broken, resource_refs=request.resource_refs,
    ))
    with pytest.raises(DurableStoreError, match="graph key"):
        store.load()


def test_rejects_missing_event_resource_refs_and_invalid_first_transition(tmp_path):
    store = _store(tmp_path)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    initial = _state(store.plan, store.attempt, store.graph_id, descriptor)
    request = _raw_request(store, initial, caller_operation="missing-resources", revision=1,
                           previous_event_id=None, resource_refs=())
    store._journal.append_with_artifacts(request=request, artifacts=(pending,))
    with pytest.raises(DurableStoreError, match="resources disagree"):
        store.load()

    other = _store(tmp_path / "first", graph_id="first-graph")
    ref, item = other.prepare_json({"input": 1}, key="root.inputs")
    forged = _state(other.plan, other.attempt, other.graph_id, ref, paused=True)
    first_request = _raw_request(other, forged, caller_operation="forged-first", revision=1,
                                 previous_event_id=None)
    other._journal.append_with_artifacts(request=first_request, artifacts=(item,))
    with pytest.raises(DurableStoreError, match="new execution"):
        other.load()


@pytest.mark.parametrize("mutation", ("unknown_payload", "schema_v3", "unknown_state", "bad_operation", "bad_event"))
def test_closed_v2_wire_rejects_corrupt_foreign_headers_before_filtering(tmp_path, mutation):
    store = _store(tmp_path / mutation)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    state = _state(store.plan, store.attempt, store.graph_id, descriptor)
    first = store.save(state, expected_revision=0, operation_id="create", artifacts=(pending,))
    raw_state = dict(state)
    if mutation == "unknown_state":
        raw_state["unknown"] = True
    request = _raw_request(store, raw_state, caller_operation="bad-" + mutation,
                           revision=2, previous_event_id=first.event_id)
    payload = dict(request.payload)
    if mutation == "unknown_payload":
        payload["unknown"] = True
    if mutation == "schema_v3":
        payload["schema"] = "unchain.skeleton_workflow_execution_journal.v3"
    operation = build_operation_ref(request.operation.operation_id, domain="skeleton.workflow_execution.saved", payload=payload)
    event_id = request.event_id
    if mutation == "bad_operation":
        operation = OperationRef(request.operation.operation_id, "0" * 64)
    if mutation == "bad_event":
        event_id = "event.forged"
    store._journal.append(request=JournalAppendRequest(
        event_id=event_id, event_type=request.event_type, attempt=request.attempt,
        operation=operation, payload=payload, resource_refs=request.resource_refs,
    ))
    absent = WorkflowExecutionStore(store._journal, store.plan, graph_id="absent-graph", attempt=store.attempt)
    with pytest.raises(DurableStoreError):
        absent.load()


def test_rejects_v1_only_graph_as_fresh_execution(tmp_path):
    from unchain.skeleton.recovery_journal import save_workflow_recovery

    plan, attempt, journal = _plan(), _attempt("execution"), _journal(tmp_path, "execution")
    store = WorkflowExecutionStore(journal, plan, graph_id="graph", attempt=attempt)
    envelope = _legacy_envelope(plan, attempt, "graph")
    save_workflow_recovery(journal, plan, envelope, attempt=attempt, operation_id="legacy", expected_revision=0)
    descriptor, pending = store.prepare_json({"input": 1}, key="root.inputs")
    state = _state(plan, attempt, "graph", descriptor)
    request = _raw_request(store, state, caller_operation="new-v2", revision=1, previous_event_id=None)
    journal.append_with_artifacts(request=request, artifacts=(pending,))
    with pytest.raises(DurableStoreError, match="v1 recovery"):
        store.load()


def test_valid_foreign_v1_coexists_but_malformed_header_fails_closed(tmp_path):
    from unchain.skeleton.recovery_journal import save_workflow_recovery

    plan, attempt, journal = _plan(), _attempt("execution"), _journal(tmp_path, "execution")
    saved = save_workflow_recovery(
        journal, plan, _legacy_envelope(plan, attempt, "foreign"),
        attempt=attempt, operation_id="foreign-v1", expected_revision=0,
    )
    target = WorkflowExecutionStore(journal, plan, graph_id="target", attempt=attempt)
    assert target.load() is None
    payload = dict(saved.event.payload)
    payload["graph_key"] = "0" * 64
    source_key = saved.event.payload["graph_key"]
    operation_id = "skeleton.workflow-recovery.op." + source_key + "." + hashlib.sha256(b"malformed-v1").hexdigest()
    journal.append(request=JournalAppendRequest(
        event_id="skeleton.workflow-recovery.event." + hashlib.sha256(operation_id.encode("ascii")).hexdigest(),
        event_type=saved.event.event_type, attempt=attempt,
        operation=build_operation_ref(operation_id, domain="skeleton.workflow_recovery.saved", payload=payload),
        payload=payload, resource_refs=(),
    ))
    with pytest.raises(DurableStoreError):
        target.load()
