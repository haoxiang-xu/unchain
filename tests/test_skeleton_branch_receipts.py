"""#351: the chosen and skipped outlets share one durable selection commit."""
import json

import pytest

from tests.test_skeleton_durable_execution import FIXTURES, code, document, runner
from unchain.skeleton import WorkflowNodeCompleted


@pytest.mark.parametrize("flag,selected,skipped", [(True, "b", "c"), (False, "c", "b")])
def test_if_selection_records_only_untaken_outlet_and_survives_reconstruction(tmp_path, flag, selected, skipped):
    calls = []
    run = runner(tmp_path, code(calls))
    run.create({"flag": flag})
    run.run(max_steps=2)
    record = run.store.load()
    branch = record.state["completed"][-1]
    assert branch["node_id"] == "if" and branch["edge_id"] == selected
    expected = {"schema": "unchain.skeleton.branch_selection.v1", "node_id": "if",
                "selected_edge_id": selected, "skipped_edges": [{"edge_id": skipped, "status": "skipped"}]}
    assert run.store.read_json(branch["result_ref"]) == expected
    # The receipt is an artifact of the actual atomic Journal state commit,
    # not a UI projection, and the rejoin is not labelled skipped.
    snapshot = run.store.journal.capture_snapshot()
    event = next(event for event in snapshot.events if event.event_id == record.event_id)
    assert event.event_type == "skeleton.workflow_execution.saved"
    assert branch["result_ref"]["ref"]["id"] in [ref.resource_id for ref in event.resource_refs]
    assert calls == []
    cold = runner(tmp_path, code(calls))
    cold.run()
    assert calls == ["left" if flag else "right", "join"]
    assert cold.store.read_json(branch["result_ref"]) == expected
    completed = cold.store.load()
    assert completed.state["completed"][-1]["node_id"] == "end"
    assert cold.run().status == "completed"
    assert cold.store.load() == completed


@pytest.mark.parametrize("value,selected,skipped", [("match", "b", "c"), ("other", "c", "b")])
def test_switch_receipt_includes_case_or_default_without_skipping_the_end(tmp_path, value, selected, skipped):
    document = json.loads((FIXTURES / "switch_workflow.json").read_text())
    document["nodes"][1]["config"]["execution"]["value"]["value"] = value
    calls = []

    def execute(context, _inputs):
        calls.append(context.node_id)
        return WorkflowNodeCompleted({"value": context.node_id})

    run = runner(tmp_path, execute, doc=document)
    run.create({})
    assert run.run().status == "completed"
    rows = run.store.load().state["completed"]
    branch = next(row for row in rows if row["node_id"] == "switch")
    assert run.store.read_json(branch["result_ref"]) == {
        "schema": "unchain.skeleton.branch_selection.v1", "node_id": "switch",
        "selected_edge_id": selected, "skipped_edges": [{"edge_id": skipped, "status": "skipped"}],
    }
    assert calls == ["case" if value == "match" else "fallback"]
    assert rows[-1]["node_id"] == "end"


def test_failed_condition_has_no_selection_receipt(tmp_path):
    doc = document()
    doc["nodes"][1]["config"]["execution"]["condition"] = {
        "kind": "compare", "operator": "contains",
        "left": {"kind": "literal", "value": 7}, "right": {"kind": "literal", "value": "x"},
    }
    calls = []
    run = runner(tmp_path, code(calls), doc=doc)
    run.create({"flag": True})
    assert run.run().status == "failed"
    assert calls == []
    assert [row["node_id"] for row in run.store.load().state["completed"]] == ["start"]


def test_selection_receipt_cannot_escape_a_failed_state_commit(tmp_path, monkeypatch):
    class BeforeCommit(BaseException):
        pass

    run = runner(tmp_path, code([]))
    run.create({"flag": True})
    run.step()
    before = run.store.load()
    save = run.store.save

    def reject(*_args, **_kwargs):
        raise BeforeCommit()

    monkeypatch.setattr(run.store, "save", reject)
    with pytest.raises(BeforeCommit):
        run.step()
    assert run.store.load() == before
    monkeypatch.setattr(run.store, "save", save)
    run.step()
    after = run.store.load()
    assert after.revision == before.revision + 1
    assert run.store.read_json(after.state["completed"][-1]["result_ref"])["skipped_edges"] == [
        {"edge_id": "c", "status": "skipped"},
    ]


def test_multiple_skipped_outlets_do_not_skip_their_shared_target(tmp_path):
    doc = json.loads((FIXTURES / "switch_workflow.json").read_text())
    doc["nodes"][1]["config"]["execution"]["cases"].append({"id": "second", "value": "second"})
    doc["edges"].append({"id": "f", "kind": "flow", "source_node_id": "switch",
                         "source_port_id": "second", "target_node_id": "case", "target_port_id": "in"})
    calls = []

    def execute(context, _inputs):
        calls.append(context.node_id)
        return WorkflowNodeCompleted({"value": context.node_id})

    run = runner(tmp_path, execute, doc=doc)
    run.create({})
    assert run.run().status == "completed"
    branch = next(row for row in run.store.load().state["completed"] if row["node_id"] == "switch")
    assert run.store.read_json(branch["result_ref"])["skipped_edges"] == [
        {"edge_id": "c", "status": "skipped"}, {"edge_id": "f", "status": "skipped"},
    ]
    assert calls == ["case"]


def test_historical_empty_branch_result_remains_resumable(tmp_path, monkeypatch):
    calls = []
    run = runner(tmp_path, code(calls))
    original = run._commit_node

    def old_commit(record, node_id, resolved, result, port, *args):
        # Reproduce the exact former raw-result shape without changing Journal
        # identities, schema, state transitions, artifacts or hashing rules.
        return original(record, node_id, resolved, {} if node_id == "if" else result, port, *args)

    monkeypatch.setattr(run, "_commit_node", old_commit)
    run.create({"flag": False})
    run.run(max_steps=2)
    branch = run.store.load().state["completed"][-1]
    assert run.store.read_json(branch["result_ref"]) == {}
    cold = runner(tmp_path, code(calls))
    assert cold.run().outputs == {"answer": "right"}
    assert calls == ["right", "join"]
    assert cold.store.read_json(branch["result_ref"]) == {}
