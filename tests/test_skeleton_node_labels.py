from __future__ import annotations

from copy import deepcopy

import pytest

from unchain.runtime.runtime_protocol import runtime_protocol_manifest
from unchain.skeleton import SkeletonValidationError, validate_skeleton_document


NODE_TYPES = (
    "start", "end", "agent", "toolkit_pool", "subagent_pool", "if_else",
    "switch_case", "tool_call", "skill_read", "code", "workflow_call", "hooks_pool",
)


def document():
    return {
        "schema_version": 1, "kind": "agent", "id": "same-resource", "name": "Agent",
        "nodes": [
            {"id": f"node-{index}", "type": node_type, "config": {"opaque": [index]}}
            for index, node_type in enumerate(NODE_TYPES)
        ],
        "edges": [{
            "id": "same-edge", "kind": "flow", "source_node_id": "node-0",
            "source_port_id": "out", "target_node_id": "node-1", "target_port_id": "in",
        }],
    }


@pytest.mark.parametrize("label", ["Display name", "  审核 e\u0301\n", "\ufeff", "\u0085name\u0085"])
def test_common_node_label_preserves_text_without_changing_graph(label):
    original = document()
    labelled = deepcopy(original)
    for node in labelled["nodes"]:
        node["label"] = label

    validated = validate_skeleton_document(labelled)

    assert validated == labelled
    for node in validated["nodes"]:
        assert node.pop("label") == label
    assert validated == original
    assert all(node["label"] == label for node in labelled["nodes"])


def test_historical_label_free_document_is_unchanged_and_detached():
    original = document()
    validated = validate_skeleton_document(original)
    assert validated == original
    assert all("label" not in node for node in validated["nodes"])
    validated["nodes"][0]["config"]["opaque"].append("edit")
    assert original["nodes"][0]["config"]["opaque"] == [0]


@pytest.mark.parametrize("label", [None, True, 3, [], {}])
def test_label_rejects_non_strings_at_label_path(label):
    value = document()
    value["nodes"][0]["label"] = label
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(value)
    assert error.value.code == "invalid_type"
    assert error.value.path == ("document", "nodes", 0, "label")


@pytest.mark.parametrize("label", ["", " \t\r\n", "\u0085", "\u001c\u001d\u001e\u001f", "\u00a0\u2007\u202f\u3000"])
def test_label_rejects_python_whitespace_only_at_label_path(label):
    value = document()
    value["nodes"][0]["label"] = label
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(value)
    assert error.value.code == "invalid_value"
    assert error.value.path == ("document", "nodes", 0, "label")


def test_label_does_not_open_node_envelope_to_other_metadata():
    value = document()
    value["nodes"][0].update(label="Accepted", display_name="Not admitted")
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(value)
    assert error.value.code == "unknown_field"
    assert error.value.path == ("document", "nodes", 0, "display_name")


@pytest.mark.parametrize("missing", ["id", "type", "config"])
def test_label_does_not_replace_required_node_fields(missing):
    value = document()
    value["nodes"][0]["label"] = "Accepted"
    del value["nodes"][0][missing]
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_document(value)
    assert error.value.code == "missing_field"
    assert error.value.path == ("document", "nodes", 0, missing)


def test_manifest_advertises_label_in_minor_eight_without_losing_old_features():
    # Labels were introduced at minor 8 and remain in minor 10 (presentation 9, layout 10).
    protocol = next(item for item in runtime_protocol_manifest()["protocols"] if item["id"] == "skeleton_execution")
    assert (protocol["major"], protocol["minor"]) == (1, 10)
    assert set(protocol["features"]) == {
        "code_python_v1", "document_layout_v1", "document_presentation_v1", "draft_diagnostics_v1", "durable_exclusive_dag_v2",
        "end_only_root_completion_v1", "explicit_bindings_v1", "hook_embedded_workflow_v1",
        "hook_enable_v1", "hook_phase_receipts_v1", "hook_tool_set_v1",
        "kernel_profile_auth_v1", "kernel_profiles_v1", "legacy_conversion_v1",
        "node_display_label_v1", "non_llm_workflow_v1", "pinned_workflow_calls_v1",
        "production_host_bindings_v1", "skill_read_v1", "structured_output_v1",
        "subagent_graphs_v1", "subagent_native_observation_v1",
        "subagent_tool_settlement_projection_v1", "tool_call_v1",
    }
