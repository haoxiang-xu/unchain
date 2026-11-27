from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from unchain.journal import (
    AttemptRef, BoundExecutionJournal, GenerationRef, JournalAppendRequest,
    JournalConflictError, JournalRepositoryError, OperationRef, build_operation_ref,
)
from unchain.skeleton import (
    build_workflow_recovery_envelope,
    prepare_workflow,
    workflow_definition_digest,
    validate_workflow_recovery_envelope,
)
from unchain.skeleton.recovery_journal import (
    WorkflowRecoveryJournalError,
    load_workflow_recovery,
    save_workflow_recovery,
)


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_recovery"


def _plan():
    return prepare_workflow(json.loads((FIXTURES / "workflow.json").read_text()))


def _records(plan):
    invocation = {
        "execution_id": "execution",
        "graph_invocation_id": "graph",
        "attempt_id": "attempt",
    }
    return {
        "invocation": invocation,
        "call_stack": [
            {
                **invocation,
                "definition_id": "recovery-workflow",
                "definition_digest": workflow_definition_digest(plan),
            }
        ],
        "completed_nodes": [
            {
                "node_id": "start",
                "node_invocation_id": "start-call",
                "attempt_id": "start-attempt",
                "input_receipt_id": "start-input",
                "output_receipt_id": "start-output",
            },
            {
                "node_id": "if",
                "node_invocation_id": "if-call",
                "attempt_id": "if-attempt",
                "input_receipt_id": "if-input",
                "output_receipt_id": "if-output",
            },
            {
                "node_id": "left",
                "node_invocation_id": "left-call",
                "attempt_id": "left-attempt",
                "input_receipt_id": "left-input",
                "output_receipt_id": "left-output",
            },
        ],
        "selected_edges": [
            {
                "edge_id": "start-if",
                "source_node_id": "start",
                "source_port_id": "out",
                "target_node_id": "if",
                "target_port_id": "in",
            },
            {
                "edge_id": "if-left",
                "source_node_id": "if",
                "source_port_id": "true",
                "target_node_id": "left",
                "target_port_id": "in",
            },
            {
                "edge_id": "left-join",
                "source_node_id": "left",
                "source_port_id": "out",
                "target_node_id": "join",
                "target_port_id": "in",
            },
        ],
        "active": {
            "kind": "waiting_interaction",
            "node_id": "join",
            "node_invocation_id": "join-call",
            "attempt_id": "join-attempt",
            "interaction_id": "interaction",
            "request_receipt_id": "request",
        },
        "terminal": None,
    }


def _envelope(plan, **changes):
    values = _records(plan)
    values.update(changes)
    return build_workflow_recovery_envelope(plan, **values)


def _graph_envelope(plan, graph_invocation_id):
    values = _records(plan)
    values["invocation"]["graph_invocation_id"] = graph_invocation_id
    values["call_stack"][-1]["graph_invocation_id"] = graph_invocation_id
    return build_workflow_recovery_envelope(plan, **values)


def _journal(tmp_path):
    from unchain.persistence import SQLiteContextV2Store

    store = SQLiteContextV2Store(
        database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects",
    )
    return store.bind_execution("execution")


def _terminal_envelope(plan):
    values = _records(plan)
    values["completed_nodes"] += [
        {"node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "input_receipt_id": "join-input", "output_receipt_id": "join-output"},
        {"node_id": "end", "node_invocation_id": "end-call", "attempt_id": "end-attempt", "input_receipt_id": "end-input", "output_receipt_id": "end-output"},
    ]
    values["selected_edges"] += [{"edge_id": "join-end", "source_node_id": "join", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"}]
    values["active"] = None
    values["terminal"] = {"kind": "end_completed", "node_id": "end", "node_invocation_id": "end-call", "end_receipt_id": "end-output"}
    return build_workflow_recovery_envelope(plan, **values)


def _attempt():
    return AttemptRef(GenerationRef("execution", "generation"), "attempt")


def test_save_load_transition_and_exact_retry_after_later_revision(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    waiting = _envelope(plan)
    first = save_workflow_recovery(
        journal, plan, waiting, attempt=_attempt(), operation_id="save-waiting", expected_revision=0
    )
    unknown = _envelope(
        plan,
        active={
            "kind": "outcome_unknown",
            "node_id": "join",
            "node_invocation_id": "join-call",
            "attempt_id": "join-attempt",
            "intent_receipt_id": "intent",
        },
    )
    second = save_workflow_recovery(
        journal, plan, unknown, attempt=_attempt(), operation_id="save-unknown", expected_revision=1
    )
    retried = save_workflow_recovery(
        journal, plan, waiting, attempt=_attempt(), operation_id="save-waiting", expected_revision=0
    )
    loaded = load_workflow_recovery(journal, plan, graph_invocation_id="graph")

    assert first.duplicate is False
    assert second.duplicate is False
    assert retried.duplicate is True and retried.event == first.event
    assert loaded is not None and loaded.revision == 2 and loaded.envelope == unknown
    loaded.envelope["active"]["intent_receipt_id"] = "mutated"
    assert load_workflow_recovery(journal, plan, graph_invocation_id="graph").envelope == unknown


def test_same_operation_id_conflicts_when_revision_or_body_changes(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    waiting = _envelope(plan)
    save_workflow_recovery(
        journal, plan, waiting, attempt=_attempt(), operation_id="stable-operation", expected_revision=0
    )
    unknown = _envelope(
        plan,
        active={
            "kind": "outcome_unknown",
            "node_id": "join",
            "node_invocation_id": "join-call",
            "attempt_id": "join-attempt",
            "intent_receipt_id": "intent",
        },
    )
    with pytest.raises(JournalConflictError):
        save_workflow_recovery(
            journal,
            plan,
            unknown,
            attempt=_attempt(),
            operation_id="stable-operation",
            expected_revision=1,
        )
    assert load_workflow_recovery(journal, plan, graph_invocation_id="graph").revision == 1


def test_rejects_valid_cp7_parent_call_stack_rewrite_before_commit(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    save_workflow_recovery(
        journal, plan, _envelope(plan), attempt=_attempt(), operation_id="save-waiting", expected_revision=0
    )
    changed = _records(plan)
    changed["call_stack"].insert(
        0,
        {
            "execution_id": "execution",
            "graph_invocation_id": "parent",
            "attempt_id": "parent-attempt",
            "definition_id": "parent-definition",
            "definition_digest": "0" * 64,
            "caller_node_invocation_id": "parent-call",
        },
    )
    changed["active"] = {
        "kind": "outcome_unknown",
        "node_id": "join",
        "node_invocation_id": "join-call",
        "attempt_id": "join-attempt",
        "intent_receipt_id": "intent",
    }
    rewritten = build_workflow_recovery_envelope(plan, **changed)
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        save_workflow_recovery(
            journal, plan, rewritten, attempt=_attempt(), operation_id="save-unknown", expected_revision=1
        )
    assert error.value.code == "identity_mismatch"
    assert len(journal.capture_snapshot().events) == 1


def test_rejects_non_native_caller_operation_id_without_normalizing(tmp_path):
    plan = _plan()
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        save_workflow_recovery(
            _journal(tmp_path),
            plan,
            _envelope(plan),
            attempt=_attempt(),
            operation_id=" save-waiting ",
            expected_revision=0,
        )
    assert (error.value.code, error.value.path) == ("invalid_value", ("operation_id",))


def test_unicode_graph_ids_remain_distinct_and_preserve_surrogates(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    graph_ids = ("graph-\U0001f600", "graph-\ud83d\ude00", "graph-\ud800")
    for index, graph_id in enumerate(graph_ids):
        save_workflow_recovery(
            journal,
            plan,
            _graph_envelope(plan, graph_id),
            attempt=_attempt(),
            operation_id=f"unicode-{index}",
            expected_revision=0,
        )
    assert [
        load_workflow_recovery(journal, plan, graph_invocation_id=graph_id).envelope["invocation"]["graph_invocation_id"]
        for graph_id in graph_ids
    ] == list(graph_ids)


def test_unknown_recovery_refuses_new_revision_but_exact_retry_is_retained(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    waiting = _envelope(plan)
    save_workflow_recovery(
        journal, plan, waiting, attempt=_attempt(), operation_id="save-waiting", expected_revision=0
    )
    unknown = _envelope(
        plan,
        active={
            "kind": "outcome_unknown",
            "node_id": "join",
            "node_invocation_id": "join-call",
            "attempt_id": "join-attempt",
            "intent_receipt_id": "intent",
        },
    )
    saved = save_workflow_recovery(
        journal, plan, unknown, attempt=_attempt(), operation_id="save-unknown", expected_revision=1
    )
    retried = save_workflow_recovery(
        journal, plan, unknown, attempt=_attempt(), operation_id="save-unknown", expected_revision=1
    )
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        save_workflow_recovery(
            journal, plan, unknown, attempt=_attempt(), operation_id="new-operation", expected_revision=2
        )
    assert retried.duplicate is True and retried.event == saved.event
    assert error.value.code == "invalid_transition"
    assert len(journal.capture_snapshot().events) == 2


def test_terminal_initial_descriptor_refuses_new_revision(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    terminal = _terminal_envelope(plan)
    save_workflow_recovery(journal, plan, terminal, attempt=_attempt(), operation_id="terminal", expected_revision=0)
    with pytest.raises(WorkflowRecoveryJournalError):
        save_workflow_recovery(journal, plan, terminal, attempt=_attempt(), operation_id="after-terminal", expected_revision=1)
    assert load_workflow_recovery(journal, plan, graph_invocation_id="graph").envelope == terminal


def test_racing_real_sqlite_writers_leave_one_revision_winner(tmp_path):
    plan = _plan()
    barrier = threading.Barrier(2)
    results = []
    errors = []

    from unchain.journal.ports import BoundExecutionJournal

    class GatedJournal(BoundExecutionJournal):
        def __init__(self, inner):
            super().__init__(inner.execution_id)
            self.inner = inner

        def append(self, *, request):
            return self.inner.append(request=request)

        def read(self, *, after=None, limit=100):
            return self.inner.read(after=after, limit=limit)

        def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
            snapshot = self.inner.capture_snapshot(max_events=max_events, max_bytes=max_bytes)
            barrier.wait(timeout=10)
            return snapshot

        def append_with_artifacts(self, *, request, artifacts, precondition=None):
            return self.inner.append_with_artifacts(request=request, artifacts=artifacts, precondition=precondition)

    journals = (GatedJournal(_journal(tmp_path)), GatedJournal(_journal(tmp_path)))

    def writer(journal, operation_id):
        try:
            results.append(save_workflow_recovery(journal, plan, _envelope(plan), attempt=_attempt(), operation_id=operation_id, expected_revision=0))
        except JournalConflictError:
            results.append("conflict")
        except BaseException as exc:  # Assert thread failures after both joins.
            errors.append(exc)

    first = threading.Thread(target=writer, args=(journals[0], "race-one"))
    second = threading.Thread(target=writer, args=(journals[1], "race-two"))
    first.start(); second.start(); first.join(timeout=10); second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    if errors:
        raise errors[0]
    assert len([result for result in results if result != "conflict"]) == 1
    assert results.count("conflict") == 1
    assert len(_journal(tmp_path).capture_snapshot().events) == 1


def test_cold_process_loads_a_real_sqlite_descriptor(tmp_path):
    plan = _plan()
    journal = _journal(tmp_path)
    waiting = _envelope(plan)
    save_workflow_recovery(journal, plan, waiting, attempt=_attempt(), operation_id="cold-save", expected_revision=0)
    code = """import json, sys
from pathlib import Path
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_workflow
from unchain.skeleton.recovery_journal import load_workflow_recovery
root=Path(sys.argv[1]); plan=prepare_workflow(json.loads(Path(sys.argv[2]).read_text()))
journal=SQLiteContextV2Store(database_path=root/'context.sqlite3',object_directory=root/'objects').bind_execution('execution')
record=load_workflow_recovery(journal,plan,graph_invocation_id='graph')
assert record is not None and record.revision == 1 and record.envelope['active']['kind'] == 'waiting_interaction'
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), str(FIXTURES / "workflow.json")], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_explicit_adapter_import_does_not_load_provider_modules():
    code = """import sys
blocked=('unchain.providers','openai','anthropic','google','ollama')
hits=[]
class Guard:
 def find_spec(self,name,path=None,target=None):
  if any(name==item or name.startswith(item+'.') for item in blocked):
   hits.append(name); raise ImportError(name)
sys.meta_path.insert(0,Guard())
import unchain.skeleton.recovery_journal
assert hits == []
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_racing_real_sqlite_writers_leave_one_revision_one_winner(tmp_path):
    plan = _plan()
    save_workflow_recovery(_journal(tmp_path), plan, _envelope(plan), attempt=_attempt(), operation_id="seed", expected_revision=0)
    unknown = _envelope(plan, active={"kind": "outcome_unknown", "node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "intent_receipt_id": "intent"})
    barrier = threading.Barrier(2)
    results = []
    errors = []
    from unchain.journal.ports import BoundExecutionJournal

    class GatedJournal(BoundExecutionJournal):
        def __init__(self, inner):
            super().__init__(inner.execution_id); self.inner = inner
        def append(self, *, request): return self.inner.append(request=request)
        def read(self, *, after=None, limit=100): return self.inner.read(after=after, limit=limit)
        def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
            result = self.inner.capture_snapshot(max_events=max_events, max_bytes=max_bytes); barrier.wait(timeout=10); return result
        def append_with_artifacts(self, *, request, artifacts, precondition=None):
            return self.inner.append_with_artifacts(request=request, artifacts=artifacts, precondition=precondition)

    journals = (GatedJournal(_journal(tmp_path)), GatedJournal(_journal(tmp_path)))

    def writer(journal, operation_id):
        try:
            results.append(save_workflow_recovery(journal, plan, unknown, attempt=_attempt(), operation_id=operation_id, expected_revision=1))
        except JournalConflictError:
            results.append("conflict")
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=writer, args=(journals[0], "race-one")); second = threading.Thread(target=writer, args=(journals[1], "race-two"))
    first.start(); second.start(); first.join(timeout=10); second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    if errors:
        raise errors[0]
    assert len([result for result in results if result != "conflict"]) == 1
    assert results.count("conflict") == 1
    assert len(_journal(tmp_path).capture_snapshot().events) == 2


def _save(journal, plan, envelope, operation="seed", revision=0, attempt=None):
    return save_workflow_recovery(
        journal, plan, envelope, attempt=attempt or _attempt(),
        operation_id=operation, expected_revision=revision,
    )


def _replace_wire_envelope(payload, raw):
    payload["encoded_envelope"] = base64.b64encode(raw).decode("ascii")
    payload["envelope_sha256"] = hashlib.sha256(raw).hexdigest()


def _append_wire(
    journal,
    payload,
    *,
    operation="corrupt",
    attempt=None,
    wrong_digest=False,
    native_graph_key=None,
):
    # Use the real journal producer with an independently specified wire record.
    # Deliberately bypass only the Skeleton writer, as a mismatched writer or
    # corrupt persisted descriptor must still be rejected by its strict reader.
    key = native_graph_key or payload.get("graph_key", "0" * 64)
    native = "skeleton.workflow-recovery.op." + key + "." + hashlib.sha256(operation.encode()).hexdigest()
    event_id = "skeleton.workflow-recovery.event." + hashlib.sha256(native.encode()).hexdigest()
    ref = build_operation_ref(native, domain="skeleton.workflow_recovery.saved", payload=payload)
    if wrong_digest:
        ref = OperationRef(native, "0" * 64)
    return journal.append(request=JournalAppendRequest(
        event_id=event_id, event_type="skeleton.workflow_recovery.saved",
        attempt=attempt or _attempt(), operation=ref, payload=payload,
    ))


@pytest.mark.parametrize("queried_graph", ["graph", "absent"])
def test_contradictory_foreign_header_fails_before_graph_filtering(
    tmp_path,
    queried_graph,
):
    plan, journal = _plan(), _journal(tmp_path)
    first = _save(journal, plan, _envelope(plan)).event
    payload = first.to_dict()["payload"]
    original_graph_key = payload["graph_key"]
    foreign = _graph_envelope(plan, "foreign-graph")
    raw = json.dumps(
        foreign,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["graph_key"] = hashlib.sha256(b"foreign-graph").hexdigest()
    _replace_wire_envelope(payload, raw)
    _append_wire(
        journal,
        payload,
        operation="foreign-bad-tail",
        native_graph_key=original_graph_key,
    )

    with pytest.raises(WorkflowRecoveryJournalError) as error:
        load_workflow_recovery(
            journal,
            plan,
            graph_invocation_id=queried_graph,
        )
    assert (error.value.code, error.value.path) == (
        "identity_mismatch",
        ("snapshot", "events", 1, "operation", "operation_id"),
    )


def test_valid_foreign_graph_with_different_definition_coexists(tmp_path):
    plan, journal = _plan(), _journal(tmp_path)
    _save(journal, plan, _envelope(plan))

    changed_document = json.loads((FIXTURES / "workflow.json").read_text())
    changed_document["nodes"][2]["config"]["foreign_definition"] = True
    foreign_plan = prepare_workflow(changed_document)
    foreign = _graph_envelope(foreign_plan, "foreign-graph")
    _save(journal, foreign_plan, foreign)

    assert load_workflow_recovery(
        journal,
        plan,
        graph_invocation_id="graph",
    ).envelope == _envelope(plan)
    assert load_workflow_recovery(
        journal,
        foreign_plan,
        graph_invocation_id="foreign-graph",
    ).envelope == foreign


def test_malformed_foreign_attempt_identity_fails_before_graph_filtering(tmp_path):
    plan, journal = _plan(), _journal(tmp_path)
    _save(journal, plan, _envelope(plan))
    foreign = _graph_envelope(plan, "foreign-graph")
    foreign["invocation"].pop("execution_id")
    raw = json.dumps(
        foreign,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = journal.capture_snapshot().events[0].to_dict()["payload"]
    payload["graph_key"] = hashlib.sha256(b"foreign-graph").hexdigest()
    _replace_wire_envelope(payload, raw)
    _append_wire(journal, payload, operation="foreign-missing-execution")

    with pytest.raises(WorkflowRecoveryJournalError) as error:
        load_workflow_recovery(journal, plan, graph_invocation_id="graph")
    assert (error.value.code, error.value.path) == (
        "invalid_type",
        (
            "snapshot", "events", 1, "payload", "encoded_envelope",
            "invocation", "execution_id",
        ),
    )


def test_absence_two_interactions_then_end_and_interleaved_graphs(tmp_path):
    plan, journal = _plan(), _journal(tmp_path)
    assert load_workflow_recovery(journal, plan, graph_invocation_id="graph") is None
    first = _envelope(plan)
    first["completed_nodes"] = first["completed_nodes"][:2]
    first["selected_edges"] = first["selected_edges"][:2]
    first["active"] = dict(kind="waiting_interaction", node_id="left",
        node_invocation_id="left-call", attempt_id="left-attempt",
        interaction_id="first-interaction", request_receipt_id="first-request")
    _save(journal, plan, first)
    other = _envelope(plan)
    other["invocation"]["graph_invocation_id"] = "other"
    other["call_stack"][-1]["graph_invocation_id"] = "other"
    _save(journal, plan, other)  # Same caller operation ID is graph-scoped.
    _save(journal, plan, _envelope(plan), "second-wait", 1)
    terminal = _terminal_envelope(plan)
    _save(journal, plan, terminal, "end", 2)
    result = load_workflow_recovery(_journal(tmp_path), plan, graph_invocation_id="graph")
    assert result.revision == 3 and result.store_seq == 4 and result.envelope == terminal
    assert load_workflow_recovery(journal, plan, graph_invocation_id="other").revision == 1


@pytest.mark.parametrize("change,code", [
    ("noop", "invalid_transition"), ("receipt", "invalid_transition"),
    ("branch", "invalid_transition"), ("active", "identity_mismatch"),
    ("interaction", "identity_mismatch"), ("completion", "identity_mismatch"),
])
def test_valid_cp7_rewrites_fail_before_persistence(tmp_path, change, code):
    plan, journal = _plan(), _journal(tmp_path)
    original = _envelope(plan)
    _save(journal, plan, original)
    candidate = deepcopy(original)
    if change == "receipt":
        candidate["completed_nodes"][0]["output_receipt_id"] = "replacement"
    elif change == "branch":
        candidate["completed_nodes"][2]["node_id"] = "right"
        candidate["selected_edges"][1].update(edge_id="if-right", source_port_id="false", target_node_id="right")
        candidate["selected_edges"][2].update(edge_id="right-join", source_node_id="right")
    elif change == "active":
        candidate["active"]["node_invocation_id"] = "new-call"
    elif change == "interaction":
        candidate["active"]["interaction_id"] = "replacement-interaction"
    elif change == "completion":
        candidate = _terminal_envelope(plan)
        candidate["completed_nodes"][-2]["node_invocation_id"] = "replacement-join"
    validate_workflow_recovery_envelope(candidate, plan)
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        _save(journal, plan, candidate, "rewrite", 1)
    assert error.value.code == code
    assert journal.capture_snapshot().event_count == 1
    assert load_workflow_recovery(journal, plan, graph_invocation_id="graph").envelope == original


@pytest.mark.parametrize("kind", ["execution", "attempt", "generation", "definition"])
def test_foreign_identity_rejected_without_writing(tmp_path, kind):
    plan, journal = _plan(), _journal(tmp_path)
    _save(journal, plan, _envelope(plan))
    attempt = _attempt()
    candidate = _terminal_envelope(plan)
    if kind == "execution":
        attempt = AttemptRef(GenerationRef("foreign", "generation"), "attempt")
    elif kind == "attempt":
        attempt = AttemptRef(GenerationRef("execution", "generation"), "foreign")
    elif kind == "generation":
        attempt = AttemptRef(GenerationRef("execution", "foreign"), "attempt")
    else:
        document = json.loads((FIXTURES / "workflow.json").read_text())
        document["nodes"][2]["config"]["changed"] = True
        plan = prepare_workflow(document)
        candidate = _terminal_envelope(plan)
    with pytest.raises(WorkflowRecoveryJournalError):
        _save(journal, plan, candidate, "foreign", 1, attempt)
    assert journal.capture_snapshot().event_count == 1


@pytest.mark.parametrize("revision", [True, -1, 1.0, "1", 50])
def test_invalid_revision_is_not_coerced(tmp_path, revision):
    plan, journal = _plan(), _journal(tmp_path)
    with pytest.raises(WorkflowRecoveryJournalError):
        _save(journal, plan, _envelope(plan), revision=revision)
    assert journal.capture_snapshot().event_count == 0


@pytest.mark.parametrize("change,code", [
    ("schema", "unknown_variant"), ("codec", "unknown_variant"),
    ("extra", "unknown_field"), ("base64", "invalid_value"),
    ("digest", "digest_mismatch"), ("whitespace", "invalid_value"),
    ("duplicates", "duplicate_name"), ("huge-integer", "invalid_json"),
    ("graph", "identity_mismatch"), ("revision", "revision_mismatch"),
    ("previous", "identity_mismatch"), ("operation", "digest_mismatch"),
    ("envelope-version", "unknown_variant"),
])
def test_damaged_tail_never_falls_back_to_old_checkpoint(tmp_path, change, code):
    plan, journal = _plan(), _journal(tmp_path)
    event = _save(journal, plan, _envelope(plan)).event
    payload = event.to_dict()["payload"]
    payload["revision"] = 2
    payload["previous_event"] = dict(event_id=event.event_id, store_seq=event.store_seq,
        operation_id=event.operation.operation_id, payload_sha256=event.operation.payload_sha256)
    raw = json.dumps(_terminal_envelope(plan), ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode()
    _replace_wire_envelope(payload, raw)
    if change == "schema": payload["schema"] = "unchain.skeleton_workflow_recovery_journal.v9"
    elif change == "codec": payload["codec"] = "unknown"
    elif change == "extra": payload["extra"] = True
    elif change == "base64": payload["encoded_envelope"] = "!"
    elif change == "digest": payload["envelope_sha256"] = "0" * 64
    elif change == "whitespace": _replace_wire_envelope(payload, b" " + raw)
    elif change == "duplicates": _replace_wire_envelope(payload, b'{"invocation":{},"invocation":{}}')
    elif change == "huge-integer": _replace_wire_envelope(payload, b'{"x":' + b"9" * 5000 + b"}")
    elif change == "graph": payload["graph_key"] = "0" * 64
    elif change == "revision": payload["revision"] = 3
    elif change == "previous": payload["previous_event"]["event_id"] = "foreign"
    elif change == "envelope-version":
        value = json.loads(raw); value["schema"] = "unchain.skeleton_workflow_recovery.v9"
        _replace_wire_envelope(payload, json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    _append_wire(journal, payload, wrong_digest=change == "operation")
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        load_workflow_recovery(journal, plan, graph_invocation_id="graph")
    assert error.value.code == code
    assert error.value.path[:3] == ("snapshot", "events", 1)
    # Malformed header identities cannot vanish through graph filtering.
    if change in {"schema", "codec", "extra", "graph"}:
        with pytest.raises(WorkflowRecoveryJournalError):
            load_workflow_recovery(journal, plan, graph_invocation_id="absent")


def test_no_atomic_capability_refuses_instead_of_falling_back(tmp_path):
    inner = _journal(tmp_path)

    class NonAtomic(BoundExecutionJournal):
        def append(self, *, request):
            pytest.fail("non-atomic append must never be called")
        def read(self, **kwargs):
            return inner.read(**kwargs)
        def capture_snapshot(self, **kwargs):
            return inner.capture_snapshot(**kwargs)

    plan = _plan()
    with pytest.raises(JournalRepositoryError, match="atomically"):
        _save(NonAtomic("execution"), plan, _envelope(plan))
    assert inner.capture_snapshot().event_count == 0


def test_snapshot_capacity_is_checked_before_commit_and_never_truncated(tmp_path, monkeypatch):
    from dataclasses import replace
    import unchain.journal.snapshot as snapshots
    from unchain.journal.resource_limits import BoundaryResourceLimitError

    plan, journal = _plan(), _journal(tmp_path)
    _save(journal, plan, _envelope(plan))
    # Lower the shared cap to exercise the exact same real SQLite transaction
    # and bounded snapshot validator without seeding 10,000 fsynced objects.
    with monkeypatch.context() as patch:
        patch.setattr(snapshots, "JOURNAL_SNAPSHOT_LIMITS",
                      replace(snapshots.JOURNAL_SNAPSHOT_LIMITS, max_items=1))
        with pytest.raises(BoundaryResourceLimitError):
            _save(journal, plan, _terminal_envelope(plan), "overflow", 1)
        assert journal.capture_snapshot().event_count == 1
    _save(journal, plan, _terminal_envelope(plan), "overflow", 1)
    with monkeypatch.context() as patch:
        patch.setattr(snapshots, "JOURNAL_SNAPSHOT_LIMITS",
                      replace(snapshots.JOURNAL_SNAPSHOT_LIMITS, max_items=1))
        with pytest.raises(BoundaryResourceLimitError):
            load_workflow_recovery(journal, plan, graph_invocation_id="graph")
    assert journal.capture_snapshot().event_count == 2


def test_envelope_byte_ceiling_rejects_before_writing(tmp_path):
    plan, journal = _plan(), _journal(tmp_path)
    value = _envelope(plan)
    value["active"]["request_receipt_id"] = "x" * (8 * 1024 * 1024)
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        _save(journal, plan, value)
    assert error.value.code == "limit_bytes"
    assert journal.capture_snapshot().event_count == 0


def test_reader_enforces_actual_eight_mib_decoded_byte_ceiling(tmp_path):
    plan, journal = _plan(), _journal(tmp_path)
    event = _save(journal, plan, _envelope(plan)).event
    payload = event.to_dict()["payload"]
    # 8 MiB is congruent to 2 mod 3: one extra decoded byte has the SAME
    # base64 length. Checking only the encoded length would miss this input.
    _replace_wire_envelope(payload, b"x" * (8 * 1024 * 1024 + 1))
    _append_wire(journal, payload)
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        load_workflow_recovery(journal, plan, graph_invocation_id="graph")
    assert error.value.code == "limit_bytes"


def test_persisted_records_do_not_cross_old_new_checkpoint_readers(tmp_path):
    from unchain.context.graph_checkpoint import GraphExecutionPlan

    plan, journal = _plan(), _journal(tmp_path)
    event = _save(journal, plan, _envelope(plan)).event
    loaded = load_workflow_recovery(_journal(tmp_path), plan, graph_invocation_id="graph")
    with pytest.raises(ValueError):
        GraphExecutionPlan.from_dict(loaded.envelope)
    payload = event.to_dict()["payload"]
    encoded = loaded.envelope
    encoded["schema"] = "unchain.graph_execution_plan.v1"
    _replace_wire_envelope(payload, json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode())
    _append_wire(journal, payload)
    with pytest.raises(WorkflowRecoveryJournalError) as error:
        load_workflow_recovery(journal, plan, graph_invocation_id="graph")
    assert error.value.code == "unknown_variant"


def test_real_store_save_and_load_do_not_initialize_clients_or_read_credentials(tmp_path, monkeypatch):
    import openai

    plan, journal = _plan(), _journal(tmp_path)
    original_getitem = os._Environ.__getitem__
    attempts = []

    def guarded_env(env, key):
        label = str(key).lower()
        if any(part in label for part in ("api_key", "api_base", "base_url", "credential", "access_token")):
            attempts.append(label)
            raise AssertionError("credential read")
        return original_getitem(env, key)

    def forbidden_client(*args, **kwargs):
        attempts.append("client")
        raise AssertionError("model client initialized")

    monkeypatch.setattr(os._Environ, "__getitem__", guarded_env)
    for module_name, module in tuple(sys.modules.items()):
        if module_name.startswith(("openai", "anthropic", "ollama", "google.genai")) and module is not None:
            for name in ("OpenAI", "AsyncOpenAI", "Anthropic", "AsyncAnthropic", "Client", "AsyncClient"):
                if name in vars(module):
                    monkeypatch.setattr(module, name, forbidden_client)
    # Negative controls prove neither guard is merely decorative.
    with pytest.raises(AssertionError): os.environ.get("OPENAI_API_KEY")
    with pytest.raises(AssertionError): openai.OpenAI()
    assert len(attempts) == 2
    attempts.clear()
    _save(journal, plan, _envelope(plan))
    assert load_workflow_recovery(journal, plan, graph_invocation_id="graph").revision == 1
    assert attempts == []
