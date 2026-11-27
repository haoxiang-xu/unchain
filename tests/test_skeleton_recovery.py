from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

import pytest

from unchain.context.graph_checkpoint import GraphExecutionPlan
from unchain.skeleton import (
    RecoveryEnvelopeError,
    build_workflow_recovery_envelope,
    prepare_workflow,
    validate_workflow_recovery_envelope,
    workflow_definition_digest,
)


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_recovery"


def workflow() -> dict:
    return json.loads((FIXTURES / "workflow.json").read_text())


def records() -> dict:
    return {
        "invocation": {"execution_id": "execution", "graph_invocation_id": "graph", "attempt_id": "attempt"},
        "call_stack": [{"execution_id": "execution", "graph_invocation_id": "graph", "attempt_id": "attempt", "definition_id": "recovery-workflow", "definition_digest": "0" * 64}],
        "completed_nodes": [
            {"node_id": "start", "node_invocation_id": "start-call", "attempt_id": "start-attempt", "input_receipt_id": "start-input", "output_receipt_id": "start-output"},
            {"node_id": "if", "node_invocation_id": "if-call", "attempt_id": "if-attempt", "input_receipt_id": "if-input", "output_receipt_id": "if-output"},
            {"node_id": "left", "node_invocation_id": "left-call", "attempt_id": "left-attempt", "input_receipt_id": "left-input", "output_receipt_id": "left-output"},
        ],
        "selected_edges": [
            {"edge_id": "start-if", "source_node_id": "start", "source_port_id": "out", "target_node_id": "if", "target_port_id": "in"},
            {"edge_id": "if-left", "source_node_id": "if", "source_port_id": "true", "target_node_id": "left", "target_port_id": "in"},
            {"edge_id": "left-join", "source_node_id": "left", "source_port_id": "out", "target_node_id": "join", "target_port_id": "in"},
        ],
        "active": {"kind": "waiting_interaction", "node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "interaction_id": "interaction", "request_receipt_id": "request"},
        "terminal": None,
    }


def envelope(plan=None, **changes):
    plan = prepare_workflow(workflow()) if plan is None else plan
    value = records()
    value.update(changes)
    value["call_stack"][0]["definition_digest"] = workflow_definition_digest(plan)
    return build_workflow_recovery_envelope(plan, **value)


def test_partial_branch_waiting_unknown_terminal_and_detachment():
    plan = prepare_workflow(workflow())
    waiting = envelope(plan)
    assert waiting["selected_edges"][1]["edge_id"] == "if-left"
    source = records()
    source["call_stack"][0]["definition_digest"] = workflow_definition_digest(plan)
    first = build_workflow_recovery_envelope(plan, **source)
    source["active"]["interaction_id"] = "changed"
    first["active"]["interaction_id"] = "returned-change"
    assert envelope(plan)["active"]["interaction_id"] == "interaction"

    unknown = envelope(plan, active={"kind": "outcome_unknown", "node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "intent_receipt_id": "intent"})
    assert unknown["active"]["kind"] == "outcome_unknown"
    completed = records()["completed_nodes"] + [
        {"node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "input_receipt_id": "join-input", "output_receipt_id": "join-output"},
        {"node_id": "end", "node_invocation_id": "end-call", "attempt_id": "end-attempt", "input_receipt_id": "end-input", "output_receipt_id": "end-output"},
    ]
    selected = records()["selected_edges"] + [{"edge_id": "join-end", "source_node_id": "join", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"}]
    terminal = envelope(plan, completed_nodes=completed, selected_edges=selected, active=None, terminal={"kind": "end_completed", "node_id": "end", "node_invocation_id": "end-call", "end_receipt_id": "end-output"})
    assert terminal["terminal"]["kind"] == "end_completed"


def test_identity_digests_and_checked_plan_are_not_trusted():
    plan = prepare_workflow(workflow())
    changed = workflow(); changed["nodes"][2]["config"]["nested"] = ["changed"]
    topology = workflow(); topology["edges"][1]["id"] = "different"
    assert workflow_definition_digest(plan) != workflow_definition_digest(prepare_workflow(changed))
    topology_plan = prepare_workflow(topology)
    topology_records = records()
    topology_records["selected_edges"][1]["edge_id"] = "different"
    topology_records["call_stack"][0]["definition_digest"] = workflow_definition_digest(topology_plan)
    topology_envelope = build_workflow_recovery_envelope(topology_plan, **topology_records)
    assert envelope(plan)["definition"]["topology_digest"] != topology_envelope["definition"]["topology_digest"]
    forged = object.__new__(type(plan))
    with pytest.raises(RecoveryEnvelopeError) as error:
        workflow_definition_digest(forged)
    assert (error.value.code, error.value.path) == ("invalid_plan", ("plan",))
    malformed = object.__new__(type(plan))
    for name, value in (("document_id", "recovery-workflow"), ("node_ids", ()), ("topological_node_ids", ()), ("flow_edges", ()), ("_encoded_snapshot", ())):
        object.__setattr__(malformed, name, value)
    with pytest.raises(RecoveryEnvelopeError) as error:
        workflow_definition_digest(malformed)
    assert (error.value.code, error.value.path) == ("invalid_plan", ("plan",))
    deep_encoded: object = 0
    for _ in range(1_100):
        deep_encoded = ("array", (deep_encoded,))
    deep_plan = replace(plan, _encoded_snapshot=deep_encoded)
    with pytest.raises(RecoveryEnvelopeError) as error:
        workflow_definition_digest(deep_plan)
    assert (error.value.code, error.value.path) == ("invalid_plan", ("plan",))
    with pytest.raises(RecoveryEnvelopeError) as error:
        build_workflow_recovery_envelope(deep_plan, **records())
    assert (error.value.code, error.value.path) == ("invalid_plan", ("plan",))


def test_digest_preserves_full_json_integer_and_string_domain_across_process_limits():
    document = workflow()
    document["nodes"][2]["config"] = {
        "huge_positive": 10 ** 5000,
        "huge_negative": -(10 ** 5001),
        "surrogate": "\ud800",
    }
    plan = prepare_workflow(document)
    original_limit = sys.get_int_max_str_digits()
    original_digest = workflow_definition_digest(plan)
    original_envelope = envelope(plan)
    try:
        sys.set_int_max_str_digits(640)
        constrained_digest = workflow_definition_digest(plan)
        constrained_envelope = envelope(plan)
        assert validate_workflow_recovery_envelope(constrained_envelope, plan) == constrained_envelope
        sys.set_int_max_str_digits(0)
        unlimited_digest = workflow_definition_digest(plan)
        unlimited_envelope = envelope(plan)
    finally:
        sys.set_int_max_str_digits(original_limit)
    assert original_digest == constrained_digest == unlimited_digest
    assert original_envelope["definition"] == constrained_envelope["definition"] == unlimited_envelope["definition"]


def test_digest_distinguishes_unicode_scalars_from_explicit_surrogates():
    scalar_value = workflow()
    scalar_value["nodes"][2]["config"] = {"value": "\U00010000"}
    surrogate_value = workflow()
    surrogate_value["nodes"][2]["config"] = {"value": "\ud800\udc00"}
    assert scalar_value != surrogate_value
    assert workflow_definition_digest(prepare_workflow(scalar_value)) != workflow_definition_digest(prepare_workflow(surrogate_value))

    scalar_key = workflow()
    scalar_key["nodes"][2]["config"] = {"\U00010000": "value"}
    surrogate_key = workflow()
    surrogate_key["nodes"][2]["config"] = {"\ud800\udc00": "value"}
    assert scalar_key != surrogate_key
    assert workflow_definition_digest(prepare_workflow(scalar_key)) != workflow_definition_digest(prepare_workflow(surrogate_key))


def test_rejects_identity_path_terminal_and_closed_field_failures():
    plan = prepare_workflow(workflow())
    invalid = envelope(plan); invalid["completed_nodes"][0]["node_id"] = "right"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("path_mismatch", ("envelope", "completed_nodes", 0, "node_id"))
    invalid = envelope(plan); invalid["selected_edges"][1]["target_node_id"] = "right"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("edge_mismatch", ("envelope", "selected_edges", 1))
    invalid = envelope(plan); invalid["selected_edges"][1], invalid["selected_edges"][2] = invalid["selected_edges"][2], invalid["selected_edges"][1]
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("path_mismatch", ("envelope", "selected_edges", 1, "source_node_id"))
    invalid = envelope(plan); invalid["completed_nodes"][0]["provider"] = "forbidden"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("unknown_field", ("envelope", "completed_nodes", 0, "provider"))
    invalid = envelope(plan); invalid["terminal"] = {"kind": "end_completed", "node_id": "end", "node_invocation_id": "end-call", "end_receipt_id": "end-output"}
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert error.value.code == "invalid_value"


def test_call_stack_schema_and_old_checkpoint_readers_are_mutually_closed():
    plan = prepare_workflow(workflow())
    invalid = envelope(plan); invalid["call_stack"][0]["graph_invocation_id"] = "other"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert error.value.code == "lineage_mismatch"
    invalid = envelope(plan); invalid["schema"] = "unchain.graph_execution_plan.v1"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("unknown_variant", ("envelope", "schema"))
    with pytest.raises(ValueError):
        GraphExecutionPlan.from_dict(envelope(plan))
    with pytest.raises(RecoveryEnvelopeError):
        validate_workflow_recovery_envelope({"schema": "unchain.graph_execution_plan.v1"}, plan)


def test_rejects_graph_and_receipt_identity_collisions():
    plan = prepare_workflow(workflow())
    for field, value in (("node_invocation_id", "left-call"), ("attempt_id", "left-attempt")):
        invalid = envelope(plan)
        invalid["active"][field] = value
        with pytest.raises(RecoveryEnvelopeError) as error:
            validate_workflow_recovery_envelope(invalid, plan)
        assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "active", field))
    invalid = envelope(plan); invalid["completed_nodes"][1]["input_receipt_id"] = "start-output"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "completed_nodes", 1, "input_receipt_id"))
    invalid = envelope(plan); invalid["active"]["request_receipt_id"] = "left-output"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "active", "request_receipt_id"))
    invalid = envelope(plan); invalid["active"] = {"kind": "outcome_unknown", "node_id": "join", "node_invocation_id": "join-call", "attempt_id": "join-attempt", "intent_receipt_id": "left-output"}
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "active", "intent_receipt_id"))
    invalid = envelope(plan); invalid["completed_nodes"][0]["attempt_id"] = "attempt"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "completed_nodes", 0, "attempt_id"))
    invalid = envelope(plan); invalid["active"]["attempt_id"] = "attempt"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "active", "attempt_id"))


def test_rejects_call_stack_identity_collisions():
    plan = prepare_workflow(workflow())
    parent = {"execution_id": "execution", "graph_invocation_id": "parent", "attempt_id": "parent-attempt", "definition_id": "parent-definition", "definition_digest": "1" * 64, "caller_node_invocation_id": "caller"}
    invalid = envelope(plan); invalid["call_stack"] = [dict(parent), invalid["call_stack"][0]]
    invalid["call_stack"][0]["execution_id"] = "other"
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("lineage_mismatch", ("envelope", "call_stack", 0, "execution_id"))
    for field, value in (("graph_invocation_id", "graph"), ("attempt_id", "attempt")):
        invalid = envelope(plan); invalid["call_stack"] = [dict(parent, **{field: value}), invalid["call_stack"][0]]
        with pytest.raises(RecoveryEnvelopeError) as error:
            validate_workflow_recovery_envelope(invalid, plan)
        assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "call_stack", 1, field))
    second_parent = dict(parent, graph_invocation_id="parent-two", attempt_id="parent-attempt-two")
    invalid = envelope(plan); invalid["call_stack"] = [dict(parent), second_parent, invalid["call_stack"][0]]
    with pytest.raises(RecoveryEnvelopeError) as error:
        validate_workflow_recovery_envelope(invalid, plan)
    assert (error.value.code, error.value.path) == ("duplicate_name", ("envelope", "call_stack", 1, "caller_node_invocation_id"))


def test_fresh_process_imports_recovery_without_provider_packages():
    code = '''import importlib, json, sys
blocked = ("unchain.agent", "unchain.agent.model_io", "unchain.providers", "openai", "anthropic", "google", "ollama")
class Guard:
 def find_spec(self, name, path=None, target=None):
  if any(name == root or name.startswith(root + ".") for root in blocked): raise ImportError("blocked: " + name)
sys.meta_path.insert(0, Guard())
from unchain.skeleton import prepare_workflow, workflow_definition_digest
plan = prepare_workflow(json.load(open(sys.argv[1])))
assert len(workflow_definition_digest(plan)) == 64
'''
    result = subprocess.run([sys.executable, "-c", code, str(FIXTURES / "workflow.json")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
