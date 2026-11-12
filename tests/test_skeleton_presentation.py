"""Display-only Skeleton document presentation (icon + colour).

``presentation`` is an optional CLOSED top-level object on the version-one
document. It is admitted and preserved exactly by the schema validator and
is removed from every prepared execution snapshot, so it can never change a
plan, a definition digest, or anything a model sees.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from unchain.runtime.runtime_protocol import runtime_protocol_manifest
from unchain.skeleton import SkeletonValidationError, prepare_workflow, validate_skeleton_document
from unchain.skeleton.agent_preparation import (
    checked_execution_plan, execution_definition_digest, prepare_agent_graph,
)
from unchain.skeleton.recovery import workflow_definition_digest
from unchain.skeleton.validation import diagnose_skeleton_draft

from tests.test_skeleton_agent_preparation import agent_document, child_agent, with_subagent_pool

FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_preparation"


def workflow_document():
    return json.loads((FIXTURES / "workflow.json").read_text())


def minimal_document():
    return {
        "schema_version": 1, "kind": "agent", "id": "agent-1", "name": "Agent",
        "description": "", "nodes": [], "edges": [],
    }


VALID = [
    {"icon": "robot", "color": "#1a2b3c"},
    {"icon": "a"},
    {"color": "#000000"},
    {"icon": "x" * 40, "color": "#ffffff"},
    {"icon": "chat_bubble_2"},
    {},
]


@pytest.mark.parametrize("presentation", VALID)
def test_valid_presentation_is_preserved_exactly_and_detached(presentation):
    document = minimal_document()
    document["presentation"] = deepcopy(presentation)

    validated = validate_skeleton_document(document)

    assert validated == document
    assert validated["presentation"] == presentation
    assert set(validated["presentation"]) == set(presentation)
    assert validated["presentation"] is not document["presentation"]
    validated["presentation"]["icon"] = "changed"
    assert document["presentation"] == presentation


def test_document_without_presentation_is_unchanged():
    document = minimal_document()
    validated = validate_skeleton_document(document)
    assert validated == document
    assert "presentation" not in validated


@pytest.mark.parametrize("value", [None, "robot", 3, True, [], ["robot"]])
def test_non_object_presentation_rejected_at_presentation_path(value):
    document = minimal_document()
    document["presentation"] = value
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert error.value.code == "invalid_type"
    assert error.value.path == ("document", "presentation")


@pytest.mark.parametrize("extra", ["label", "Icon", "colour", "background"])
def test_extra_presentation_key_rejected_as_unknown_field(extra):
    document = minimal_document()
    document["presentation"] = {"icon": "robot", "color": "#112233", extra: "x"}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert error.value.code == "unknown_field"
    assert error.value.path == ("document", "presentation", extra)


@pytest.mark.parametrize("field", ["icon", "color"])
@pytest.mark.parametrize("value", [None, 3, True, [], {}, 1.5])
def test_non_string_presentation_field_rejected_as_invalid_type(field, value):
    document = minimal_document()
    document["presentation"] = {field: value}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert error.value.code == "invalid_type"
    assert error.value.path == ("document", "presentation", field)


@pytest.mark.parametrize("icon", [
    "", "Robot", "ROBOT", "robot-face", "robot face", " robot", "robot\n", "x" * 41,
    "roböt", "ｒobot", "robot.", "\u0000",
])
def test_bad_icon_pattern_rejected_as_invalid_value(icon):
    document = minimal_document()
    document["presentation"] = {"icon": icon}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert error.value.code == "invalid_value"
    assert error.value.path == ("document", "presentation", "icon")


@pytest.mark.parametrize("color", [
    "", "#ABCDEF", "#AbCdEf", "#FFFFFF", "123456", "#12345", "#1234567", "#12345g",
    "#fff", "red", " #123456", "#123456\n", "#１２３４５６",
    "rgb(0,0,0)", "##12345",
])
def test_bad_or_uppercase_color_rejected_as_invalid_value(color):
    document = minimal_document()
    document["presentation"] = {"color": color}
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert error.value.code == "invalid_value"
    assert error.value.path == ("document", "presentation", "color")


def test_presentation_does_not_reopen_the_root_envelope():
    document = minimal_document()
    document["presentation"] = {"icon": "robot"}
    document["icon"] = "robot"
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(document)
    assert error.value.code == "unknown_field"
    assert error.value.path == ("document", "icon")


def test_presentation_is_dropped_from_the_prepared_agent_plan_and_digest():
    plain = agent_document()
    decorated = deepcopy(plain)
    decorated["presentation"] = {"icon": "robot", "color": "#1a2b3c"}
    other = deepcopy(plain)
    other["presentation"] = {"icon": "star", "color": "#ffffff"}

    base = prepare_agent_graph(plain)
    plan = prepare_agent_graph(decorated)

    assert plan == base
    assert "presentation" not in plan.snapshot()
    assert plan.snapshot() == base.snapshot()
    assert checked_execution_plan(plan) == base
    assert execution_definition_digest(plan) == execution_definition_digest(base)
    assert execution_definition_digest(prepare_agent_graph(other)) == execution_definition_digest(base)
    assert decorated["presentation"] == {"icon": "robot", "color": "#1a2b3c"}


def test_presentation_is_dropped_from_the_prepared_workflow_plan_and_digest():
    plain = workflow_document()
    decorated = deepcopy(plain)
    decorated["presentation"] = {"icon": "flow", "color": "#00ff00"}

    base = prepare_workflow(plain)
    plan = prepare_workflow(decorated)

    assert plan == base
    assert "presentation" not in plan.snapshot()
    assert workflow_definition_digest(plan) == workflow_definition_digest(base)


def test_invalid_presentation_blocks_preparation_with_the_schema_error():
    document = agent_document()
    document["presentation"] = {"color": "#ABCDEF"}
    with pytest.raises(SkeletonValidationError) as error:
        prepare_agent_graph(document)
    assert (error.value.code, error.value.path) == ("invalid_value", ("document", "presentation", "color"))


def test_embedded_subagent_document_admits_presentation_and_child_plan_ignores_it():
    child = child_agent()
    child["presentation"] = {"icon": "owl", "color": "#abcdef"}
    parent = with_subagent_pool(agent_document(), {"kind": "embedded", "document": child})
    prepare_agent_graph(parent)

    plain_child = child_agent()
    assert execution_definition_digest(prepare_agent_graph(child)) == execution_definition_digest(
        prepare_agent_graph(plain_child))


def test_embedded_subagent_invalid_presentation_is_rejected_at_its_nested_path():
    child = child_agent()
    child["presentation"] = {"icon": "Owl"}
    parent = with_subagent_pool(agent_document(), {"kind": "embedded", "document": child})
    with pytest.raises(SkeletonValidationError) as error:
        prepare_agent_graph(parent)
    assert error.value.code == "invalid_value"
    assert error.value.path[-2:] == ("presentation", "icon")
    assert "document" in error.value.path[:-2]


def test_draft_diagnostics_report_invalid_presentation():
    document = minimal_document()
    document["presentation"] = {"icon": "robot", "color": "#XYZ123"}
    with pytest.raises(SkeletonValidationError) as error:
        diagnose_skeleton_draft(document)
    assert error.value.path == ("document", "presentation", "color")


def test_manifest_advertises_presentation_in_minor_nine_without_losing_old_features():
    protocol = next(item for item in runtime_protocol_manifest()["protocols"] if item["id"] == "skeleton_execution")
    assert (protocol["major"], protocol["minor"]) == (1, 9)
    assert "document_presentation_v1" in protocol["features"]
    assert "node_display_label_v1" in protocol["features"]
    assert len(protocol["features"]) == 23
