"""AC-016-D1–D5: draft IO diagnostics without complete model configuration."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from unchain.skeleton import SkeletonValidationError, diagnose_skeleton_draft
from unchain.skeleton.agent_preparation import prepare_agent_graph


def flow(identifier, source, target, outlet="out"):
    return dict(id=identifier, kind="flow", source_node_id=source, source_port_id=outlet,
                target_node_id=target, target_port_id="in")


def output(name="payload", kind="result", path=None):
    return dict(name=name, source=dict(kind=kind, path=[name] if path is None else path))


def binding(name="x", **extra):
    return dict(name=name, value=dict(kind="ref", node_id="start", path=["payload"]),
                missing="error", **extra)


def draft(inputs=None, outputs=None):
    return dict(schema_version=1, kind="workflow", id="draft", name="Draft", nodes=[
        dict(id="start", type="start", config=dict(execution=dict(outputs=[output()]))),
        dict(id="end", type="end", config=dict(execution=dict(
            inputs=[binding()] if inputs is None else inputs,
            outputs=[output("answer", "input", ["x"])] if outputs is None else outputs))),
    ], edges=[flow("one", "start", "end")])


@pytest.mark.parametrize("inputs,code,suffix", [
    ([dict(name="x", value=dict(kind="literal", value=1), missing="null", default=2)],
     "unknown_field", (0, "default")),
    ([dict(name="x", value=dict(kind="first_present", values="bad"), missing="error")],
     "invalid_value", (0, "value", "values")),
    ([binding(), binding()], "duplicate_name", (1, "name")),
    ([dict(name="x", value=dict(kind="env"), missing="error")],
     "unknown_variant", (0, "value", "kind")),
])
def test_binding_diagnostics_have_exact_row_paths(inputs, code, suffix):
    document = draft(inputs=inputs)
    before = deepcopy(document)
    found = diagnose_skeleton_draft(document)
    assert any((item.code, item.path) == (code, ("document", "nodes", 1, "config", "execution", "inputs") + suffix) for item in found)
    assert document == before


@pytest.mark.parametrize("outputs,code,suffix", [
    ([output("a", "input", ["x"]), output("a", "input", ["x"])], "duplicate_name", (1, "name")),
    ([dict(name="a", source=dict(kind="input", path=["x"]), type="string")], "unknown_field", (0, "type")),
    ([output("a", "result", [])], "invalid_value", (0, "source", "kind")),
])
def test_output_diagnostics_have_exact_row_paths(outputs, code, suffix):
    found = diagnose_skeleton_draft(draft(outputs=outputs))
    assert any((item.code, item.path) == (code, ("document", "nodes", 1, "config", "execution", "outputs") + suffix) for item in found)


def test_independent_rows_and_sections_are_reported_together():
    document = draft(inputs=[
        dict(name="x", value=dict(kind="literal", value=1), missing="null", default=2),
        dict(name="y", value=dict(kind="first_present", values="bad"), missing="error"),
    ], outputs=[output("a", "result", []), output("a", "input", ["x"])])
    paths = {item.path for item in diagnose_skeleton_draft(document)}
    prefix = ("document", "nodes", 1, "config", "execution")
    assert {prefix + ("inputs", 0, "default"), prefix + ("inputs", 1, "value", "values"),
            prefix + ("outputs", 0, "source", "kind"), prefix + ("outputs", 1, "name")} <= paths


@pytest.mark.parametrize("kernel", [None, {"schema_version": 99}])
def test_agent_io_diagnostics_do_not_wait_for_kernel_configuration(kernel):
    document = draft(inputs=[], outputs=[])
    document["kind"] = "agent"
    node = dict(id="agent", type="agent", config=dict(execution=dict(
        inputs=[dict(name="x", value=dict(kind="literal", value=1), missing="null", default=2)],
        outputs=[])))
    document["nodes"].insert(1, node)
    if kernel is not None:
        node["config"]["kernel"] = kernel
    document["edges"] = [flow("one", "start", "agent"), flow("two", "agent", "end")]
    assert any(item.path == ("document", "nodes", 1, "config", "execution", "inputs", 0, "default")
               for item in diagnose_skeleton_draft(document))
    with pytest.raises(SkeletonValidationError) as caught:
        prepare_agent_graph(document)
    assert caught.value.path[:4] == ("document", "nodes", 1, "config")
    if kernel is not None:
        assert caught.value.path[4] == "kernel"


@pytest.mark.parametrize("source,path,suffix", [
    ("missing", ["payload"], ("node_id",)),
    ("end", ["answer"], ("node_id",)),
    ("start", ["not-exported"], ("path", 0)),
])
def test_static_reference_problems_are_diagnosed(source, path, suffix):
    item = binding()
    item["value"] = dict(kind="ref", node_id=source, path=path)
    found = diagnose_skeleton_draft(draft(inputs=[item]))
    prefix = ("document", "nodes", 1, "config", "execution", "inputs", 0, "value")
    assert any(diagnostic.code == "missing_value" and diagnostic.path == prefix + suffix for diagnostic in found)


def test_dynamic_json_leaf_and_exclusive_first_present_are_not_guessed():
    document = draft()
    document["nodes"][1]["config"]["execution"]["inputs"][0]["value"]["path"] += ["customer", "id"]
    assert diagnose_skeleton_draft(document) == ()
    document = draft()
    document["nodes"][1:1] = [
        dict(id="if", type="if_else", config={}),
        dict(id="left", type="code", config=dict(execution=dict(inputs=[], outputs=[output("value")]))),
        dict(id="right", type="code", config=dict(execution=dict(inputs=[], outputs=[output("value")]))),
    ]
    document["edges"] = [flow("a", "start", "if"), flow("b", "if", "left", "true"),
                         flow("c", "if", "right", "false"), flow("d", "left", "end"), flow("e", "right", "end")]
    document["nodes"][-1]["config"]["execution"]["inputs"][0]["value"] = dict(kind="first_present", values=[
        dict(kind="ref", node_id="left", path=["value"]), dict(kind="ref", node_id="right", path=["value"])])
    assert diagnose_skeleton_draft(document) == ()


def test_diagnostics_keep_root_node_edge_order_and_report_real_row_indexes():
    document = draft(inputs=[binding(), dict(name="y", value=dict(kind="first_present", values=[]), missing="error")])
    document["nodes"][0]["config"]["execution"]["outputs"] = [dict(name="payload", source=dict(kind="other", path=[]))]
    document["edges"][0]["source_port_id"] = "bad"
    found = diagnose_skeleton_draft(document)
    node_positions = [item.path[2] for item in found if len(item.path) > 2 and item.path[1] == "nodes"]
    assert node_positions == sorted(node_positions)
    assert any(item.path == ("document", "nodes", 1, "config", "execution", "inputs", 1, "value", "values") for item in found)
    assert found[-1].path[:2] == ("document", "edges")


@pytest.mark.parametrize("patch", [{"schema_version": 2}, {"unexpected": 1}])
def test_outer_schema_errors_still_raise(patch):
    document = draft()
    document.update(patch)
    with pytest.raises(SkeletonValidationError):
        diagnose_skeleton_draft(document)


@pytest.mark.parametrize("execution,code,suffix", [
    (None, "invalid_type", ()),
    ({}, "missing_field", ("inputs",)),
    ({"inputs": {}, "outputs": []}, "invalid_type", ("inputs",)),
    ({"inputs": [], "outputs": {}}, "invalid_type", ("outputs",)),
])
def test_declared_execution_diagnoses_incomplete_or_malformed_io(execution, code, suffix):
    document = draft()
    document["nodes"][1]["config"]["execution"] = execution
    prefix = ("document", "nodes", 1, "config", "execution")
    assert any(item.code == code and item.path == prefix + suffix for item in diagnose_skeleton_draft(document))


def test_absent_execution_preserves_structural_only_drafts():
    document = draft()
    for node in document["nodes"]:
        node["config"] = {}
    assert diagnose_skeleton_draft(document) == ()


def test_nested_first_present_reference_keeps_its_actual_expression_path():
    document = draft()
    document["nodes"][1]["config"]["execution"]["inputs"][0]["value"] = dict(kind="first_present", values=[
        dict(kind="literal", value=1), dict(kind="first_present", values=[
            dict(kind="ref", node_id="start", path=["missing"]), dict(kind="literal", value=2)])])
    assert any(item.path == ("document", "nodes", 1, "config", "execution", "inputs", 0,
                            "value", "values", 1, "values", 0, "path", 0)
               for item in diagnose_skeleton_draft(document))


def test_existing_but_non_upstream_node_cannot_supply_an_input():
    document = draft()
    document["nodes"].append(dict(id="other", type="code", config=dict(execution=dict(
        inputs=[], outputs=[output("value")]))))
    document["nodes"][1]["config"]["execution"]["inputs"][0]["value"] = dict(
        kind="ref", node_id="other", path=["value"])
    assert any(item.code == "missing_value" and item.path == (
        "document", "nodes", 1, "config", "execution", "inputs", 0, "value", "node_id")
        for item in diagnose_skeleton_draft(document))


def test_output_mapping_names_a_declared_input_without_guessing_json_leaves():
    document = draft(outputs=[output("a", "input", ["missing"])])
    assert any(item.code == "missing_value" and item.path == (
        "document", "nodes", 1, "config", "execution", "outputs", 0, "source", "path", 0)
        for item in diagnose_skeleton_draft(document))
    document["nodes"][1]["config"]["execution"]["outputs"][0]["source"]["path"] = ["x", "unknown-leaf"]
    assert diagnose_skeleton_draft(document) == ()


@pytest.mark.parametrize("node_type,section", [("start", "inputs"), ("if_else", "outputs"), ("hooks_pool", "execution")])
def test_declared_io_respects_node_io_surface(node_type, section):
    document = draft()
    node = document["nodes"][0]
    node["type"] = node_type
    if node_type == "hooks_pool":
        document["kind"] = "agent"
    node["config"]["execution"]["inputs"] = []
    suffix = () if section == "execution" else (section,)
    assert any(item.code == "unknown_field" and item.path == (
        "document", "nodes", 0, "config", "execution") + suffix
        for item in diagnose_skeleton_draft(document))


def test_draft_io_diagnosis_does_not_import_providers():
    document = draft(inputs=[dict(name="x", value=dict(kind="literal", value=1), missing="null", default=2)])
    program = "import json,sys; from unchain.skeleton import diagnose_skeleton_draft; diagnostics=diagnose_skeleton_draft(json.loads(sys.argv[1])); assert diagnostics; assert not any(n.startswith('unchain.providers') for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", program, json.dumps(document)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
