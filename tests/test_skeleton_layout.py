"""Display-only Skeleton node layout (``document.layout``).

``layout`` is an optional CLOSED top-level object: ``{"nodes": {<node id>:
{"x": number, "y": number}}}``. The schema admits and preserves it exactly,
and every prepared execution snapshot drops it, from the root and from each
embedded subagent document, so moving a node can never change a plan or a
definition digest.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from unchain.runtime.runtime_protocol import runtime_protocol_manifest
from unchain.skeleton import SkeletonValidationError, prepare_workflow, validate_skeleton_document
from unchain.skeleton.agent_preparation import execution_definition_digest, prepare_agent_graph
from unchain.skeleton.recovery import workflow_definition_digest
from unchain.skeleton.validation import diagnose_skeleton_draft

from tests.test_skeleton_agent_preparation import agent_document, child_agent, with_subagent_pool

FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_preparation"


def workflow_document():
    return json.loads((FIXTURES / "workflow.json").read_text())


def two_node_document():
    return {
        "schema_version": 1, "kind": "workflow", "id": "wf-1", "name": "Workflow",
        "nodes": [
            {"id": "start", "type": "start", "config": {}},
            {"id": "end", "type": "end", "config": {}},
        ],
        "edges": [],
    }


def layout_for(document, dx=0):
    return {"nodes": {node["id"]: {"x": 80 + i * 260 + dx, "y": 80.5} for i, node in enumerate(document["nodes"])}}


@pytest.mark.parametrize("layout", [
    {"nodes": {}},
    {"nodes": {"start": {"x": 0, "y": 0}}},
    {"nodes": {"start": {"x": -1000000, "y": 1000000}, "end": {"x": 12.25, "y": -3}}},
])
def test_valid_layout_is_preserved_exactly_and_detached(layout):
    document = two_node_document()
    document["layout"] = deepcopy(layout)
    validated = validate_skeleton_document(document)
    assert validated == document
    assert validated["layout"] is not document["layout"]
    validated["layout"]["nodes"]["moved"] = {"x": 1, "y": 1}
    assert document["layout"] == layout


def test_document_without_layout_is_unchanged():
    document = two_node_document()
    assert "layout" not in validate_skeleton_document(document)


@pytest.mark.parametrize("value", [None, "x", 3, True, [], [{}]])
def test_non_object_layout_is_invalid_type(value):
    document = two_node_document()
    document["layout"] = value
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == ("invalid_type", ("document", "layout"))


def test_layout_requires_nodes():
    document = two_node_document()
    document["layout"] = {}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == ("missing_field", ("document", "layout", "nodes"))


@pytest.mark.parametrize("extra", ["viewport", "zoom", "Nodes"])
def test_extra_layout_key_is_unknown_field(extra):
    document = two_node_document()
    document["layout"] = {"nodes": {}, extra: {}}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == ("unknown_field", ("document", "layout", extra))


@pytest.mark.parametrize("value", [None, [], "x", 1])
def test_non_object_layout_nodes_is_invalid_type(value):
    document = two_node_document()
    document["layout"] = {"nodes": value}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == ("invalid_type", ("document", "layout", "nodes"))


def test_layout_for_a_node_not_in_the_document_is_rejected():
    document = two_node_document()
    document["layout"] = {"nodes": {"ghost": {"x": 1, "y": 2}}}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == ("missing_value", ("document", "layout", "nodes", "ghost"))


@pytest.mark.parametrize("point,code,tail", [
    ({"x": 1}, "missing_field", ("y",)),
    ({"x": 1, "y": 2, "z": 3}, "unknown_field", ("z",)),
    ({"x": "1", "y": 2}, "invalid_type", ("x",)),
    ({"x": True, "y": 2}, "invalid_type", ("x",)),
    ({"x": 1, "y": None}, "invalid_type", ("y",)),
    ({"x": 1000001, "y": 0}, "invalid_value", ("x",)),
    ({"x": 0, "y": -1000000.5}, "invalid_value", ("y",)),
    ([1, 2], "invalid_type", ()),
])
def test_bad_point_is_rejected_at_its_path(point, code, tail):
    document = two_node_document()
    document["layout"] = {"nodes": {"start": point}}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == (code, ("document", "layout", "nodes", "start") + tail)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinate_is_rejected(value):
    document = two_node_document()
    document["layout"] = {"nodes": {"start": {"x": value, "y": 0}}}
    with pytest.raises(SkeletonValidationError):
        validate_skeleton_document(document)


def test_layout_does_not_reopen_the_root_envelope():
    document = two_node_document()
    document["layout"] = {"nodes": {}}
    document["viewport"] = {}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert (error.value.code, error.value.path) == ("unknown_field", ("document", "viewport"))


def test_layout_is_dropped_from_the_prepared_agent_plan_and_digest():
    plain = agent_document()
    moved = deepcopy(plain)
    moved["layout"] = layout_for(plain)
    elsewhere = deepcopy(plain)
    elsewhere["layout"] = layout_for(plain, dx=999)
    base = prepare_agent_graph(plain)
    plan = prepare_agent_graph(moved)
    assert plan == base
    assert "layout" not in plan.snapshot()
    assert execution_definition_digest(plan) == execution_definition_digest(base)
    assert execution_definition_digest(prepare_agent_graph(elsewhere)) == execution_definition_digest(base)
    assert moved["layout"] == layout_for(plain)


def test_layout_is_dropped_from_the_prepared_workflow_plan_and_digest():
    plain = workflow_document()
    moved = deepcopy(plain)
    moved["layout"] = layout_for(plain)
    base = prepare_workflow(plain)
    plan = prepare_workflow(moved)
    assert plan == base
    assert "layout" not in plan.snapshot()
    assert workflow_definition_digest(plan) == workflow_definition_digest(base)


def test_embedded_child_layout_never_reaches_the_parent_plan_or_digest():
    child = child_agent()
    plain_parent = with_subagent_pool(agent_document(), {"kind": "embedded", "document": deepcopy(child)})
    child["layout"] = layout_for(child)
    moved_parent = with_subagent_pool(agent_document(), {"kind": "embedded", "document": child})
    plan = prepare_agent_graph(moved_parent)
    assert plan == prepare_agent_graph(plain_parent)
    assert "layout" not in json.dumps(plan.snapshot(), default=str)
    assert execution_definition_digest(plan) == execution_definition_digest(prepare_agent_graph(plain_parent))
    assert child["layout"] == layout_for(child)


def test_embedded_child_invalid_layout_is_rejected_at_its_nested_path():
    child = child_agent()
    child["layout"] = {"nodes": {"ghost": {"x": 0, "y": 0}}}
    parent = with_subagent_pool(agent_document(), {"kind": "embedded", "document": child})
    with pytest.raises(SkeletonValidationError) as error:
        prepare_agent_graph(parent)
    assert error.value.code == "missing_value"
    assert error.value.path[-3:] == ("layout", "nodes", "ghost")
    assert "document" in error.value.path[:-3]


def test_draft_diagnostics_report_invalid_layout():
    document = two_node_document()
    document["layout"] = {"nodes": {"start": {"x": "a", "y": 0}}}
    with pytest.raises(SkeletonValidationError) as error:
        diagnose_skeleton_draft(document)
    assert error.value.path == ("document", "layout", "nodes", "start", "x")


def test_manifest_advertises_layout_in_minor_ten_without_losing_old_features():
    protocol = next(item for item in runtime_protocol_manifest()["protocols"] if item["id"] == "skeleton_execution")
    assert (protocol["major"], protocol["minor"]) == (1, 11)
    assert "document_layout_v1" in protocol["features"]
    assert "document_presentation_v1" in protocol["features"]
    assert "node_display_label_v1" in protocol["features"]
    assert "workflow_parameters_v1" in protocol["features"]
    assert len(protocol["features"]) == 25
