"""R3 follow-up: stable cold diagnostics and intrinsic empty IO surfaces."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from unchain.skeleton import (
    WorkflowExecutionError,
    WorkflowNodeCompleted,
    diagnose_skeleton_draft,
    prepare_workflow,
    run_workflow,
)


def _workflow(filename="if_workflow.json"):
    return json.loads((Path(__file__).parent / "fixtures" / "skeleton_execution" / filename).read_text())


def _end(document):
    return document["nodes"][-1]["config"]["execution"]


def _binding(expression, policy="error"):
    row = dict(name="answer", value=expression, missing=policy)
    if policy == "default":
        row["default"] = "fallback"
    return row


def _ref(node_id, path):
    return dict(kind="ref", node_id=node_id, path=path)


@pytest.mark.parametrize("nested", [False, True], ids=["row-required-fields", "nested-required-fields"])
def test_draft_diagnostics_are_identical_across_cold_hash_seeds(nested):
    document = _workflow()
    if nested:
        _end(document)["inputs"] = [_binding(dict(kind="first_present", values=[
            dict(kind="literal", value=1), dict(kind="ref")]))]
        _end(document)["outputs"] = [dict(name="answer", source={})]
    else:
        _end(document)["inputs"] = [{}]
        _end(document)["outputs"] = [{}]
    before = deepcopy(document)
    program = """import json, sys
from unchain.skeleton import diagnose_skeleton_draft
document = json.load(sys.stdin)
print(json.dumps([{'code': item.code, 'path': item.path, 'message': item.message}
                  for item in diagnose_skeleton_draft(document)], sort_keys=True))
"""
    responses = []
    for seed in range(8):
        environment = dict(os.environ, PYTHONHASHSEED=str(seed))
        completed = subprocess.run([sys.executable, "-c", program], input=json.dumps(document),
                                   env=environment, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr
        responses.append(json.loads(completed.stdout))
    assert all(response and len(response) == 2 for response in responses)
    assert responses == [responses[0]] * len(responses), responses
    assert document == before


@pytest.mark.parametrize("filename,node_id", [
    ("if_workflow.json", "if"), ("switch_workflow.json", "switch"),
])
@pytest.mark.parametrize("policy", ["error", "null", "default"])
def test_branch_named_export_is_diagnosed_without_an_output_section(filename, node_id, policy):
    document = _workflow(filename)
    _end(document)["inputs"] = [_binding(_ref(node_id, ["not-exported"]), policy)]
    _end(document)["outputs"] = [dict(name="answer", source=dict(kind="input", path=["answer"]))]
    before = deepcopy(document)
    expected = ("document", "nodes", len(document["nodes"]) - 1,
                "config", "execution", "inputs", 0, "value", "path", 0)
    found = diagnose_skeleton_draft(document)
    assert any(item.code == "missing_value" and item.path == expected for item in found), found
    assert document == before


@pytest.mark.parametrize("filename,node_id", [
    ("if_workflow.json", "if"), ("switch_workflow.json", "switch"),
])
def test_branch_whole_export_is_valid_and_really_executes(filename, node_id):
    document = _workflow(filename)
    _end(document)["inputs"] = [_binding(_ref(node_id, []))]
    _end(document)["outputs"] = [dict(name="answer", source=dict(kind="input", path=["answer"]))]
    assert diagnose_skeleton_draft(document) == ()
    called = []

    def code(context, _):
        called.append(context.node_id)
        return WorkflowNodeCompleted(dict(answer="ok", value="ok"))

    result = run_workflow(prepare_workflow(document), dict(flag=True), dict(code=code))
    assert result.outputs == dict(answer={})
    assert node_id in result.completed_node_ids and len(called) == 1


def _start_mapping(path):
    document = _workflow()
    start = document["nodes"][0]["config"]["execution"]
    start["outputs"] = [dict(name="flag", source=dict(kind="result", path=["flag"])),
                        dict(name="whole", source=dict(kind="input", path=path))]
    _end(document)["inputs"] = [_binding(_ref("start", ["whole"]))]
    return document


def test_start_named_input_mapping_is_diagnosed_without_an_input_section():
    document = _start_mapping(["not-an-input"])
    found = diagnose_skeleton_draft(document)
    expected = ("document", "nodes", 0, "config", "execution", "outputs", 1, "source", "path", 0)
    assert any(item.code == "missing_value" and item.path == expected for item in found), found


def test_start_whole_input_mapping_is_valid_and_really_executes():
    document = _start_mapping([])
    assert diagnose_skeleton_draft(document) == ()
    result = run_workflow(prepare_workflow(document), dict(flag=True),
                          dict(code=lambda *_: WorkflowNodeCompleted(dict(answer="ok"))))
    assert result.outputs == dict(answer={})


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("policy", ["error", "null", "default"])
def test_exclusive_first_present_keeps_present_null_and_missing_semantics(flag, policy):
    document = _workflow()
    _end(document)["inputs"][0]["missing"] = policy
    if policy == "default":
        _end(document)["inputs"][0]["default"] = "fallback"
    assert diagnose_skeleton_draft(document) == ()
    calls = []

    def code(context, _):
        calls.append(context.node_id)
        return WorkflowNodeCompleted(dict(answer=None if context.node_id == "left" else "right"))

    result = run_workflow(prepare_workflow(document), dict(flag=flag), dict(code=code))
    assert result.outputs == dict(answer=None if flag else "right")
    assert calls == ["left" if flag else "right"]


@pytest.mark.parametrize("policy", ["error", "null", "default"])
def test_missing_dynamic_leaf_preserves_policy_instead_of_guessing_json(policy):
    document = _workflow()
    _end(document)["inputs"] = [_binding(_ref("left", ["answer", "absent"]), policy)]
    assert diagnose_skeleton_draft(document) == ()
    plan = prepare_workflow(document)
    handlers = dict(code=lambda *_: WorkflowNodeCompleted(dict(answer={})))
    if policy == "error":
        with pytest.raises(WorkflowExecutionError) as caught:
            run_workflow(plan, dict(flag=True), handlers)
        assert caught.value.code == "missing_value"
    else:
        result = run_workflow(plan, dict(flag=True), handlers)
        assert result.outputs == dict(answer=None if policy == "null" else "fallback")


def test_required_first_present_with_only_impossible_sources_rejects_before_handlers():
    document = _workflow()
    expression = dict(kind="first_present", values=[
        _ref("missing", []), dict(kind="first_present", values=[
            _ref("if", ["not-exported"]), _ref("end", ["answer"])])])
    _end(document)["inputs"] = [_binding(expression)]
    calls = []
    with pytest.raises(WorkflowExecutionError) as caught:
        run_workflow(prepare_workflow(document), dict(flag=True),
                     dict(code=lambda *_: calls.append("called") or WorkflowNodeCompleted(dict(answer="ok"))))
    assert caught.value.code == "missing_value"
    assert caught.value.path == ("document", "nodes", 4, "config", "execution", "inputs", 0, "value")
    assert calls == []


@pytest.mark.parametrize("alternative,expected", [
    (_ref("left", ["answer"]), "left"),
    (dict(kind="literal", value="literal"), "literal"),
    (dict(kind="literal", value=None), None),
], ids=["legitimate-branch", "literal", "present-null"])
def test_required_first_present_with_one_valid_alternative_really_executes(alternative, expected):
    document = _workflow()
    _end(document)["inputs"] = [_binding(dict(kind="first_present", values=[
        _ref("missing", ["answer"]), deepcopy(alternative)]))]
    calls = []

    def code(context, _):
        calls.append(context.node_id)
        return WorkflowNodeCompleted(dict(answer=context.node_id))

    result = run_workflow(prepare_workflow(document), dict(flag=True), dict(code=code))
    assert result.outputs == dict(answer=expected)
    assert calls == ["left"]


@pytest.mark.parametrize("expression", [
    _ref("missing", ["answer"]), _ref("end", ["answer"]),
    _ref("if", ["not-exported"]), _ref("start", ["not-exported"]),
], ids=["unknown-node", "self-node", "empty-branch-export", "undeclared-start-export"])
@pytest.mark.parametrize("policy", ["null", "default"])
def test_optional_impossible_direct_reference_keeps_runtime_fallback(expression, policy):
    document = _workflow()
    _end(document)["inputs"] = [_binding(deepcopy(expression), policy)]
    calls = []

    def code(context, _):
        calls.append(context.node_id)
        return WorkflowNodeCompleted(dict(answer=context.node_id))

    result = run_workflow(prepare_workflow(document), dict(flag=True), dict(code=code))
    assert result.outputs == dict(answer=None if policy == "null" else "fallback")
    assert calls == ["left"]


@pytest.mark.parametrize("node_index", [0, 2], ids=["intrinsic-empty-start-inputs", "declared-empty-code-inputs"])
def test_impossible_local_input_output_mapping_rejects_before_handlers(node_index):
    document = _workflow()
    mappings = document["nodes"][node_index]["config"]["execution"]["outputs"]
    mappings.append(dict(name="impossible", source=dict(kind="input", path=["not-declared"])))
    calls = []
    with pytest.raises(WorkflowExecutionError) as caught:
        run_workflow(prepare_workflow(document), dict(flag=True),
                     dict(code=lambda *_: calls.append("called") or WorkflowNodeCompleted(dict(answer="ok"))))
    assert caught.value.code == "missing_value"
    assert caught.value.path == ("document", "nodes", node_index, "config", "execution",
                                 "outputs", 1, "source", "path")
    assert calls == []


def test_kernel_interaction_hook_fixture_really_runs_with_event_payload():
    from tests.test_skeleton_kernel_interaction import _hook_workflow

    result = run_workflow(prepare_workflow(_hook_workflow()), dict(event="before_tool_call"), {})
    assert result.outputs == dict(event="before_tool_call")
    assert result.completed_node_ids == ("start", "end")
