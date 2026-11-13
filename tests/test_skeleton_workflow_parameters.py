from copy import deepcopy

import pytest

from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import (WorkflowNodeCompleted, diagnose_skeleton_draft,
    prepare_workflow, run_workflow, validate_skeleton_document)
from unchain.skeleton.durable_execution import DurableWorkflowRunner, WorkflowNodeWaiting
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError


def signature():
    return {"type": "object", "properties": {"city": {"type": "string"},
        "days": {"type": "integer", "default": 1}}, "required": ["city"],
        "additionalProperties": False}


def document():
    def out(name):
        return {"name": name, "source": {"kind": "result", "path": [name]}}
    def binding(name, node):
        return {"name": name, "value": {"kind": "ref", "node_id": node, "path": [name]}, "missing": "error"}
    return {"schema_version": 1, "kind": "workflow", "id": "weather", "name": "Weather",
        "input_schema": signature(), "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": [out("city"), out("days")]}}},
            {"id": "step", "type": "code", "config": {"execution": {"inputs": [binding("city", "start"), binding("days", "start")], "outputs": [out("answer")]}}},
            {"id": "end", "type": "end", "config": {"execution": {"inputs": [binding("answer", "step")], "outputs": [{"name": "answer", "source": {"kind": "input", "path": ["answer"]}}]}}},
        ], "edges": [{"id": a + b, "kind": "flow", "source_node_id": a, "source_port_id": "out", "target_node_id": b, "target_port_id": "in"} for a, b in [("start", "step"), ("step", "end")]]}


def durable(tmp_path, doc, handler, graph="graph"):
    journal = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite3", object_directory=tmp_path / "objects").bind_execution("parameters")
    store = WorkflowExecutionStore(journal, prepare_workflow(doc), graph_id=graph,
        attempt=AttemptRef(GenerationRef("parameters", graph), graph + "-attempt"))
    return DurableWorkflowRunner(store, {"code": handler})


def test_function_signature_survives_preparation_and_defaults_reach_handler():
    doc = document(); before = deepcopy(doc); calls = []
    plan = prepare_workflow(doc)
    def handler(_, args):
        calls.append(args)
        return WorkflowNodeCompleted({"answer": args})
    assert run_workflow(plan, {"city": "Shanghai"}, {"code": handler}).outputs == {"answer": {"city": "Shanghai", "days": 1}}
    assert run_workflow(plan, {"city": "Paris", "days": 3}, {"code": handler}).outputs["answer"]["days"] == 3
    assert doc == before and plan.snapshot()["input_schema"] == signature()
    assert calls == [{"city": "Shanghai", "days": 1}, {"city": "Paris", "days": 3}]


@pytest.mark.parametrize("args,code,path", [
    ({}, "missing_value", ("inputs", "city")),
    ({"city": 1}, "invalid_type", ("inputs", "city")),
    ({"city": "X", "days": True}, "invalid_type", ("inputs", "days")),
    ({"city": "X", "days": 1.5}, "invalid_type", ("inputs", "days")),
    ({"city": "X", "days": None}, "invalid_type", ("inputs", "days")),
    ({"city": "X", "extra": 1}, "unknown_field", ("inputs", "extra")),
])
def test_bad_arguments_fail_before_effects_and_durable_publication(tmp_path, args, code, path):
    calls = []; doc = document()
    handler = lambda *_: calls.append(1)
    with pytest.raises(SkeletonValidationError) as exc:
        run_workflow(prepare_workflow(doc), args, {"code": handler})
    assert (exc.value.code, exc.value.path) == (code, path)
    runner = durable(tmp_path, doc, handler)
    with pytest.raises(SkeletonValidationError) as exc:
        runner.create(args)
    assert (exc.value.code, exc.value.path) == (code, path)
    assert runner.store.load() is None and calls == []


@pytest.mark.parametrize("mutate,code,path", [
    (lambda s: s.update(extra=True), "unknown_field", ("document", "input_schema", "extra")),
    (lambda s: s.update(additionalProperties=True), "invalid_value", ("document", "input_schema", "additionalProperties")),
    (lambda s: s["properties"]["days"].update(default="one"), "invalid_type", ("document", "input_schema", "properties", "days", "default")),
    (lambda s: s.update(required=["city", "city"]), "duplicate_name", ("document", "input_schema", "required", 1)),
    (lambda s: s["properties"]["city"].update({"$ref": "https://invalid.example/schema"}), "unknown_field", ("document", "input_schema", "properties", "city", "$ref")),
    (lambda s: s.update(required=[]), "missing_field", ("document", "input_schema", "properties", "city", "default")),
])
def test_invalid_signatures_are_repairable_but_cannot_prepare(mutate, code, path):
    doc = document(); mutate(doc["input_schema"])
    assert validate_skeleton_document(doc) == doc
    diagnostics = diagnose_skeleton_draft(doc)
    assert any((d.code, d.path) == (code, path) for d in diagnostics)
    with pytest.raises(SkeletonValidationError) as exc:
        prepare_workflow(doc)
    assert (exc.value.code, exc.value.path) == (code, path)


def test_signature_is_not_allowed_on_agents():
    doc = document(); doc["kind"] = "agent"
    with pytest.raises(SkeletonValidationError) as exc:
        validate_skeleton_document(doc)
    assert (exc.value.code, exc.value.path) == ("invalid_value", ("document", "input_schema"))


def test_old_workflows_keep_the_original_contract():
    doc = document(); del doc["input_schema"]
    result = run_workflow(prepare_workflow(doc), {"city": "X", "days": "untyped", "extra": True},
        {"code": lambda _, args: WorkflowNodeCompleted({"answer": args})})
    assert result.outputs == {"answer": {"city": "X", "days": "untyped"}}


def test_parked_run_keeps_signature_and_defaults_while_new_run_uses_new_definition(tmp_path):
    a = document(); b = deepcopy(a); b["input_schema"]["properties"]["days"]["default"] = 7
    first = durable(tmp_path, a, lambda *_: WorkflowNodeWaiting("interaction", "approve", {}))
    first.create({"city": "X"}); assert first.run().status == "waiting"
    captured = first.store.read_json(first.store.load().state["inputs_ref"])
    assert captured == {"city": "X", "days": 1}
    cold = durable(tmp_path, a, lambda *_: pytest.fail("parked operation redispatched"))
    active = cold.store.load().state["active"]
    cold.complete_wait("approve", {"answer": captured}, node_invocation_id=active["node_invocation_id"])
    result = cold.run()
    assert result.outputs == {"answer": captured}
    second = durable(tmp_path, b, lambda _, args: WorkflowNodeCompleted({"answer": args}), graph="second")
    second.create({"city": "X"}); assert second.run().outputs["answer"]["days"] == 7
    assert cold.store.plan.snapshot()["input_schema"] == a["input_schema"]


def test_mutable_defaults_are_detached_per_call():
    doc = document(); doc["input_schema"]["properties"]["days"] = {"type": "array", "default": []}
    seen = []
    def handler(_, args):
        seen.append(deepcopy(args["days"])); args["days"].append(1)
        return WorkflowNodeCompleted({"answer": args})
    plan = prepare_workflow(doc)
    for _ in range(2): run_workflow(plan, {"city": "X"}, {"code": handler})
    assert seen == [[], []] and doc["input_schema"]["properties"]["days"]["default"] == []


@pytest.mark.parametrize("value,success", [("typed", True), (5, False)])
def test_pinned_workflow_call_checks_declared_types_and_expands_defaults(tmp_path, value, success):
    from tests.test_skeleton_workflow_runtime import child_document, parent_document, host
    from unchain.skeleton.workflow_runtime import PublishedWorkflow
    child = child_document()
    child["input_schema"] = {"type": "object", "properties": {
        "value": {"type": "string"}, "unused": {"type": "boolean", "default": False}},
        "required": ["value"], "additionalProperties": False}
    child["nodes"][0]["config"]["execution"]["outputs"].append({
        "name": "unused", "source": {"kind": "result", "path": ["unused"]}})
    child["nodes"][1]["config"]["execution"]["inputs"].append({
        "name": "unused", "value": {"kind": "ref", "node_id": "start", "path": ["unused"]}, "missing": "error"})
    seen = []
    def handler(_, args):
        seen.append(args)
        return WorkflowNodeCompleted({"value": args["value"], "version": "rev-1"})
    runner, runtime = host(tmp_path, parent_document(), lambda wid, rev: PublishedWorkflow(wid, rev, child), handler)
    runner.create({"value": value})
    if success:
        assert runner.run().outputs == {"value": "typed", "version": "rev-1"}
        assert seen == [{"value": "typed", "unused": False}]
    else:
        before = runner.store.load()
        with pytest.raises(SkeletonValidationError) as error:
            runtime.preflight_inputs("call", {"value": value})
        assert (error.value.code, error.value.path) == ("invalid_type", ("inputs", "value"))
        assert runner.store.load() == before and seen == []
