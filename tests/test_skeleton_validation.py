from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from unchain.skeleton import SkeletonDiagnostic, SkeletonValidationError, diagnose_skeleton_draft, validate_skeleton_for_execution


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_validation"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def node(identifier, kind):
    return {"id": identifier, "type": kind, "config": {}}


def edge(identifier, source, target, source_port="out", target_port="in", kind="flow"):
    return {"id": identifier, "kind": kind, "source_node_id": source, "source_port_id": source_port, "target_node_id": target, "target_port_id": target_port}


def document(nodes, edges, kind="agent"):
    return {"schema_version": 1, "kind": kind, "id": "doc", "name": "Document", "nodes": nodes, "edges": edges}


def codes(value):
    return [item.code for item in diagnose_skeleton_draft(value)]


def test_fixtures_complete_graphs_and_incomplete_draft():
    assert diagnose_skeleton_draft(fixture("complete_agent.json")) == ()
    assert diagnose_skeleton_draft(fixture("complete_workflow.json")) == ()
    diagnostics = diagnose_skeleton_draft(fixture("incomplete_draft.json"))
    assert [item.code for item in diagnostics] == ["start_count", "end_count", "agent_required"]
    with pytest.raises(SkeletonValidationError) as error:
        validate_skeleton_for_execution(fixture("incomplete_draft.json"))
    assert (error.value.code, error.value.path) == (diagnostics[0].code, diagnostics[0].path)


def test_diagnostic_is_immutable_and_preflight_returns_detached_document():
    assert SkeletonDiagnostic("x", ("p",), "message").code == "x"
    with pytest.raises(Exception):
        SkeletonDiagnostic("x", (), "message").code = "y"
    source = fixture("complete_workflow.json")
    result = validate_skeleton_for_execution(source)
    result["nodes"][1]["config"]["changed"] = True
    assert source["nodes"][1]["config"] == {}


@pytest.mark.parametrize("kind", ["start", "end"])
def test_missing_and_multiple_terminals(kind):
    assert f"{kind}_count" in codes(document([], []))
    assert f"{kind}_count" in codes(document([node("one", kind), node("two", kind)], []))


def test_agent_and_pool_attachment_rules():
    assert "agent_required" in codes(document([node("s", "start"), node("e", "end")], []))
    value = document([node("s", "start"), node("a", "agent"), node("p", "toolkit_pool"), node("e", "end")], [
        edge("bad", "a", "p"),
    ])
    found = codes(value)
    assert "flow_pool" in found and "pool_attachment_required" in found
    value["edges"] = [edge("bad", "a", "p", "out", "in", "attach")]
    assert "attach_ports" in codes(value)
    value["edges"] = [edge("bad", "s", "p", "attach_top", "attach_bot", "attach")]
    assert "attach_endpoints" in codes(value)


@pytest.mark.parametrize("source_port, target_port", [("attach_top", "attach_top"), ("attach_bot", "attach_bot")])
def test_attach_layout_ports_do_not_imply_direction(source_port, target_port):
    value = document([node("s", "start"), node("a", "agent"), node("p", "toolkit_pool"), node("e", "end")], [
        edge("f1", "s", "a"), edge("f2", "a", "e"), edge("attach", "a", "p", source_port, target_port, "attach"),
    ])
    assert diagnose_skeleton_draft(value) == ()


@pytest.mark.parametrize("source_port, target_port, paths", [
    ("bad", "attach_top", [("document", "edges", 2, "source_port_id")]),
    ("attach_top", "bad", [("document", "edges", 2, "target_port_id")]),
    ("bad-source", "bad-target", [
        ("document", "edges", 2, "source_port_id"),
        ("document", "edges", 2, "target_port_id"),
    ]),
])
def test_invalid_attach_ports_report_each_invalid_field(source_port, target_port, paths):
    value = document([node("s", "start"), node("a", "agent"), node("p", "toolkit_pool"), node("e", "end")], [
        edge("f1", "s", "a"), edge("f2", "a", "e"), edge("attach", "a", "p", source_port, target_port, "attach"),
    ])
    diagnostics = diagnose_skeleton_draft(value)
    assert [item.path for item in diagnostics if item.code == "attach_ports"] == paths


def test_flow_ports_outlets_and_branch_requirements():
    nodes = [node("s", "start"), node("if", "if_else"), node("sw", "switch_case"), node("e", "end")]
    value = document(nodes, [
        edge("f1", "s", "if"), edge("f2", "if", "sw", "out"), edge("f3", "sw", "e", "case"),
    ])
    found = codes(value)
    assert "flow_source_port" in found and "if_outlet_required" in found and "switch_default_required" in found
    duplicate = document([node("s", "start"), node("a", "agent"), node("e", "end")], [
        edge("f1", "s", "a"), edge("f2", "s", "a"), edge("f3", "a", "e"),
    ])
    assert "duplicate_flow_outlet" in codes(duplicate)


def test_invalid_flow_endpoints_and_ports():
    nodes = [node("s", "start"), node("a", "agent"), node("e", "end")]
    value = document(nodes, [edge("a", "e", "a"), edge("b", "a", "s"), edge("c", "s", "a", "out", "wrong")])
    found = codes(value)
    assert {"flow_end_source", "flow_start_target", "flow_target_port"} <= set(found)


def test_single_invalid_flow_reports_every_independent_edge_problem():
    nodes = [node("s", "start"), node("a", "agent"), node("e", "end")]
    value = document(nodes, [edge("bad", "e", "s", "wrong-source", "wrong-target")])
    diagnostics = diagnose_skeleton_draft(value)
    edge_diagnostics = [item for item in diagnostics if item.code.startswith("flow_") and item.path[1] == "edges"]
    assert [(item.code, item.path) for item in edge_diagnostics] == [
        ("flow_end_source", ("document", "edges", 0, "source_node_id")),
        ("flow_source_port", ("document", "edges", 0, "source_port_id")),
        ("flow_start_target", ("document", "edges", 0, "target_node_id")),
        ("flow_target_port", ("document", "edges", 0, "target_port_id")),
    ]


def test_duplicate_outlet_is_emitted_in_edge_document_order():
    nodes = [node("s", "start"), node("a", "agent"), node("e", "end")]
    value = document(nodes, [
        edge("one", "s", "a"), edge("two", "s", "a"), edge("bad", "a", "e", "wrong"),
    ])
    relevant = [item.code for item in diagnose_skeleton_draft(value) if item.code in {"duplicate_flow_outlet", "flow_source_port"}]
    assert relevant == ["duplicate_flow_outlet", "flow_source_port"]


def test_flow_completeness_reachability_cycles_and_branch_rejoin():
    incomplete = document([node("s", "start"), node("a", "agent"), node("e", "end")], [edge("f", "s", "a")])
    assert "flow_output_required" in codes(incomplete)
    disconnected = document([node("s", "start"), node("a", "agent"), node("e", "end")], [edge("f", "s", "e")])
    assert {"flow_input_required", "flow_output_required", "unreachable_node", "cannot_reach_end"} <= set(codes(disconnected))
    cyclic = document([node("s", "start"), node("a", "agent"), node("e", "end")], [edge("f1", "s", "a"), edge("f2", "a", "a"), edge("f3", "a", "e")])
    assert "flow_cycle" in codes(cyclic)
    downstream = document([node("s", "start"), node("a", "agent"), node("b", "code"), node("e", "end")], [
        edge("f1", "s", "a"), edge("f2", "a", "a"), edge("f3", "a", "b"), edge("f4", "b", "e"),
    ])
    cycle_paths = [item.path for item in diagnose_skeleton_draft(downstream) if item.code == "flow_cycle"]
    assert cycle_paths == [("document", "nodes", 1)]
    overlapping = document([node("s", "start"), node("a", "if_else"), node("b", "code"), node("c", "code"), node("e", "end")], [
        edge("f1", "s", "a"), edge("f2", "a", "b", "true"), edge("f3", "b", "a"),
        edge("f4", "a", "c", "false"), edge("f5", "c", "b"),
    ])
    cycle_paths = [item.path for item in diagnose_skeleton_draft(overlapping) if item.code == "flow_cycle"]
    assert cycle_paths == [
        ("document", "nodes", 1), ("document", "nodes", 2), ("document", "nodes", 3),
    ]
    branch = document([node("s", "start"), node("if", "if_else"), node("a", "agent"), node("c", "code"), node("e", "end")], [
        edge("f1", "s", "if"), edge("f2", "if", "a", "true"), edge("f3", "if", "c", "false"), edge("f4", "a", "e"), edge("f5", "c", "e"),
    ])
    assert diagnose_skeleton_draft(branch) == ()


def test_diagnostic_order_and_exact_path():
    value = document([], [])
    diagnostics = diagnose_skeleton_draft(value)
    assert [(item.code, item.path) for item in diagnostics] == [
        ("start_count", ("document", "nodes")), ("end_count", ("document", "nodes")), ("agent_required", ("document", "nodes")),
    ]


def test_diagnostic_order_is_root_then_node_order_then_edge_order():
    value = document([node("s", "start"), node("s2", "start"), node("a", "agent"), node("p", "toolkit_pool"), node("e", "end")], [
        edge("bad-attach", "a", "p", "bad", "attach_top", "attach"),
    ])
    diagnostics = diagnose_skeleton_draft(value)
    assert diagnostics[0].code == "start_count"
    edge_position = next(index for index, item in enumerate(diagnostics) if item.path[1] == "edges")
    node_indexes = [item.path[2] for item in diagnostics[1:edge_position] if len(item.path) > 2]
    assert node_indexes == sorted(node_indexes)
    assert diagnostics[edge_position].path == ("document", "edges", 0, "source_port_id")


def test_malformed_schema_still_raises_instead_of_becoming_a_diagnostic():
    malformed = {"schema_version": 1, "kind": "agent"}
    with pytest.raises(SkeletonValidationError):
        diagnose_skeleton_draft(malformed)


def test_provider_free_import():
    result = subprocess.run([sys.executable, "-c", "import sys; import unchain.skeleton; assert 'unchain.providers' not in sys.modules"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
