from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from unchain.skeleton import SkeletonValidationError, validate_skeleton_document


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_schema"
ALL_NODE_TYPES = {
    "start", "end", "agent", "toolkit_pool", "subagent_pool", "if_else",
    "switch_case", "tool_call", "skill_read", "code", "workflow_call", "hooks_pool",
}
WORKFLOW_FORBIDDEN = {"agent", "toolkit_pool", "subagent_pool", "hooks_pool"}


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def document(kind="agent", nodes=None, edges=None):
    return {
        "schema_version": 1, "kind": kind, "id": "doc", "name": "Document",
        "nodes": [] if nodes is None else nodes, "edges": [] if edges is None else edges,
    }


def node(node_id, node_type, config=None):
    return {"id": node_id, "type": node_type, "config": {} if config is None else config}


def edge(edge_id, source, target, kind="flow"):
    return {
        "id": edge_id, "kind": kind, "source_node_id": source, "source_port_id": "out",
        "target_node_id": target, "target_port_id": "in",
    }


def test_fixtures_validate_and_are_detached():
    data = fixture("agent_document.json")
    validated = validate_skeleton_document(data)
    assert validated == data
    data["nodes"][1]["config"]["nested"].append("mutated")
    assert validated["nodes"][1]["config"]["nested"] == [True, 1]
    assert validate_skeleton_document(fixture("workflow_document.json"))["kind"] == "workflow"


def test_repeated_input_config_occurrences_are_independently_detached():
    shared_config = {"nested": [1]}
    source = document(nodes=[node("first", "start", shared_config), node("second", "end", shared_config)])
    validated = validate_skeleton_document(source)
    validated["nodes"][0]["config"]["nested"].append(2)
    assert validated["nodes"][1]["config"] == {"nested": [1]}
    assert source["nodes"][0]["config"] == {"nested": [1]}


def test_every_admitted_agent_node_type_is_valid():
    assert validate_skeleton_document(document(nodes=[node(f"node-{index}", node_type) for index, node_type in enumerate(ALL_NODE_TYPES)]))


@pytest.mark.parametrize("node_type", sorted(WORKFLOW_FORBIDDEN))
def test_workflow_rejects_platform_node_types_at_type_path(node_type):
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document("workflow", [node("node", node_type)]))
    assert error.value.path == ("document", "nodes", 0, "type")


def test_empty_draft_is_saveable_without_topology_rules():
    assert validate_skeleton_document(document("workflow")) == document("workflow")


@pytest.mark.parametrize("mutate, path", [
    (lambda value: value.update({"extra": True}), ("document", "extra")),
    (lambda value: value.__setitem__("schema_version", True), ("document", "schema_version")),
    (lambda value: value.__setitem__("kind", "other"), ("document", "kind")),
    (lambda value: value.__setitem__("name", ""), ("document", "name")),
])
def test_root_contract_errors_have_paths(mutate, path):
    value = document()
    mutate(value)
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(value)
    assert error.value.path == path


def test_closed_node_edge_and_config_contract():
    for bad_node in [
        {**node("n", "start"), "extra": 1},
        {**node("n", "unknown"), "config": {}},
        {**node("n", "start"), "config": []},
    ]:
        with pytest.raises(SkeletonValidationError):
            validate_skeleton_document(document(nodes=[bad_node]))
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document(nodes=[node("n", "start")], edges=[{**edge("e", "n", "n"), "extra": 1}]))
    assert error.value.path == ("document", "edges", 0, "extra")


def test_duplicate_ids_and_unknown_edge_endpoints_fail():
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document(nodes=[node("n", "start"), node("n", "end")]))
    assert error.value.path == ("document", "nodes", 1, "id")
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document(nodes=[node("n", "start")], edges=[edge("e", "missing", "n")]))
    assert error.value.path == ("document", "edges", 0, "source_node_id")
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document(nodes=[node("n", "start")], edges=[edge("e", "n", "missing")]))
    assert error.value.path == ("document", "edges", 0, "target_node_id")


class _String(str):
    pass


class _Dict(dict):
    pass


class _List(list):
    pass


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), ("tuple",), {1: "key"}, _String("custom"), _Dict(), _List(),
])
def test_strict_json_values_are_rejected(bad):
    with pytest.raises(SkeletonValidationError):
        validate_skeleton_document(document(nodes=[node("n", "start", {"bad": bad})]))


def test_provider_free_import():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import unchain.skeleton; assert 'unchain.providers' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
