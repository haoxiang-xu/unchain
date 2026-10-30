"""B-AC8-02: scoped, provider-free draft diagnostics for embedded Agent graphs."""
from copy import deepcopy
import json
import subprocess
import sys

import pytest

from unchain.skeleton import SkeletonValidationError, diagnose_skeleton_draft


def flow(identifier, source, target, *, kind="flow"):
    attach = kind == "attach"
    return dict(id=identifier, kind=kind, source_node_id=source,
                source_port_id="attach_top" if attach else "out",
                target_node_id=target, target_port_id="attach_bot" if attach else "in")


def agent(identifier="draft"):
    # No Kernel is configured: incomplete Agent drafts remain meaningful here.
    return dict(schema_version=1, kind="agent", id=identifier, name=identifier, nodes=[
        dict(id="start", type="start", config={}),
        dict(id="worker", type="agent", config={}),
        dict(id="end", type="end", config={}),
    ], edges=[flow("one", "start", "worker"), flow("two", "worker", "end")])


def entry(child, index):
    return dict(id=f"entry-{index}", name=f"Child {index}", description="",
                source=dict(kind="embedded", document=child), allowed_modes=["delegate"],
                output_mode="summary", memory_policy="ephemeral", parallel_safe=False,
                allowed_tools=None, disabled_tools=[])


def embed(parent, *children):
    parent["nodes"].append(dict(id="pool", type="subagent_pool", config=dict(
        subagents=[entry(child, index) for index, child in enumerate(children)])))
    parent["edges"].append(flow("attached", "worker", "pool", kind="attach"))
    return parent


def child_prefix(index=0):
    return ("nodes", 3, "config", "subagents", index, "source", "document")


def markers(document):
    return [(item.code, item.path) for item in diagnose_skeleton_draft(document)]


def invalid_io(document):
    document["nodes"][1]["config"]["execution"] = dict(inputs=[dict(
        name="x", value=dict(kind="literal", value=1), missing="null", default=2)], outputs=[])
    return document


def remove_end(document):
    document["nodes"] = [node for node in document["nodes"] if node["type"] != "end"]
    document["edges"] = [edge for edge in document["edges"] if edge["target_node_id"] != "end"]
    return document


IO_DEFAULT = ("nodes", 1, "config", "execution", "inputs", 0, "default")


def test_root_and_depth_two_io_failures_are_both_reported_at_their_own_scope():
    leaf = invalid_io(agent("leaf"))
    root = invalid_io(embed(agent("root"), embed(agent("child"), leaf)))
    found = markers(root)
    assert ("unknown_field", ("document",) + IO_DEFAULT) in found
    assert ("unknown_field", ("document",) + child_prefix() + child_prefix() + IO_DEFAULT) in found


def test_sibling_structure_io_and_bad_envelope_are_collected_independently():
    broken_structure = remove_end(agent("no-end"))
    broken_io = invalid_io(agent("bad-io"))
    broken_envelope = agent("bad-envelope")
    broken_envelope["nodes"] = {"toString": None}
    root = embed(agent("root"), broken_structure, broken_io, broken_envelope)
    found = markers(root)
    assert ("end_count", ("document",) + child_prefix(0) + ("nodes",)) in found
    assert ("unknown_field", ("document",) + child_prefix(1) + IO_DEFAULT) in found
    assert ("invalid_type", ("document",) + child_prefix(2) + ("nodes",)) in found


@pytest.mark.parametrize("child,code,tail", [
    (None, "invalid_type", ()),
    ({"schema_version": 2, "kind": "agent", "id": "v2", "name": "v2", "nodes": [], "edges": []},
     "invalid_value", ("schema_version",)),
    ({**agent("extra"), "unexpected": 1}, "unknown_field", ("unexpected",)),
    ({**agent("bad-edges"), "edges": "not-an-array"}, "invalid_type", ("edges",)),
])
def test_inner_envelope_errors_are_diagnostics_without_rejecting_the_outer_draft(child, code, tail):
    root = embed(agent("root"), child)
    before = deepcopy(root)
    assert (code, ("document",) + child_prefix() + tail) in markers(root)
    assert root == before


def test_embedded_workflow_is_reported_as_wrong_kind_at_its_own_envelope():
    child = agent("workflow")
    child["kind"] = "workflow"
    child["nodes"].pop(1)
    child["edges"] = [flow("direct", "start", "end")]
    assert ("wrong_kind", ("document",) + child_prefix() + ("kind",)) in markers(
        embed(agent("root"), child))


def test_same_local_node_ids_do_not_leak_parent_or_sibling_io_surfaces():
    root, valid, invalid = agent("root"), agent("valid"), agent("invalid")
    root["nodes"][0]["config"]["execution"] = dict(outputs=[
        dict(name="parent-only", source=dict(kind="result", path=[]))])
    for child, requested in [(valid, "payload"), (invalid, "parent-only")]:
        child["nodes"][0]["config"]["execution"] = dict(outputs=[
            dict(name="payload", source=dict(kind="result", path=[]))])
        child["nodes"][1]["config"]["execution"] = dict(inputs=[dict(
            name="x", missing="error", value=dict(kind="ref", node_id="start", path=[requested]))],
            outputs=[])
    found = markers(embed(root, valid, invalid))
    expected = ("missing_value", ("document",) + child_prefix(1) +
                ("nodes", 1, "config", "execution", "inputs", 0, "value", "path", 0))
    assert found == [expected]


def test_unconfigured_child_kernel_does_not_hide_declared_io():
    child = invalid_io(agent("child"))
    assert "kernel" not in child["nodes"][1]["config"]
    found = markers(embed(agent("root"), child))
    assert found == [("unknown_field", ("document",) + child_prefix() + IO_DEFAULT)]


def test_the_same_child_object_in_two_placements_is_diagnosed_at_both_paths():
    child = invalid_io(agent("shared-object"))
    root = embed(agent("root"), child, child)
    assert root["nodes"][3]["config"]["subagents"][0]["source"]["document"] is child
    found = markers(root)
    assert found == [
        ("unknown_field", ("document",) + child_prefix(index) + IO_DEFAULT)
        for index in range(2)
    ]


def nested_chain(depth):
    document = agent(f"depth-{depth}")
    for index in reversed(range(depth)):
        document = embed(agent(f"depth-{index}"), document)
    return document


def test_six_embedded_agent_levels_remain_admissible_to_draft_diagnostics():
    assert markers(nested_chain(6)) == []


def test_seventh_embedded_level_reports_the_existing_agent_depth_limit():
    assert ("subagent_depth", ("document",) + child_prefix() * 7) in markers(nested_chain(7))


def test_agent_references_are_not_followed_or_guessed_by_pure_draft_diagnosis():
    root = embed(agent("root"), agent("child"))
    root["nodes"][3]["config"]["subagents"][0]["source"] = dict(
        kind="agent_ref", agent_id="missing-or-self-or-archived")
    assert markers(root) == []


@pytest.mark.parametrize("patch,code,path", [
    ({"schema_version": 2}, "invalid_value", ("document", "schema_version")),
    ({"unexpected": 1}, "unknown_field", ("document", "unexpected")),
    ({"nodes": {}}, "invalid_type", ("document", "nodes")),
])
def test_outer_schema_admission_stays_a_rejection(patch, code, path):
    root = embed(agent("root"), invalid_io(agent("child")))
    root.update(patch)
    with pytest.raises(SkeletonValidationError) as rejected:
        diagnose_skeleton_draft(root)
    assert (rejected.value.code, rejected.value.path) == (code, path)


def test_nested_diagnostics_do_not_mutate_the_draft_and_have_stable_sibling_order():
    root = embed(agent("root"), invalid_io(agent("first")), invalid_io(agent("second")))
    before = deepcopy(root)
    first = diagnose_skeleton_draft(root)
    second = diagnose_skeleton_draft(root)
    assert first == second
    assert [(item.code, item.path) for item in first] == [
        ("unknown_field", ("document",) + child_prefix(index) + IO_DEFAULT)
        for index in range(2)
    ]
    assert root == before


def test_actual_nested_diagnosis_does_not_import_model_providers():
    root = embed(agent("root"), invalid_io(agent("child")))
    program = (
        "import json,sys; from unchain.skeleton import diagnose_skeleton_draft; "
        "items=diagnose_skeleton_draft(json.loads(sys.argv[1])); "
        "assert items; assert not any(n.startswith('unchain.providers') for n in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", program, json.dumps(root)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
