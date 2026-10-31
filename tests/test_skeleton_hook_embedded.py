"""Owned Workflow bodies are pinned without pretending they are Store revisions."""
from copy import deepcopy
import pytest

from unchain.skeleton.agent_preparation import prepare_agent_graph, execution_definition_digest
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hook_runtime import SkeletonHookRuntime
from unchain.skeleton.hooks_config import validate_hooks_pool_config
from unchain.skeleton.validation import diagnose_skeleton_draft
from pathlib import Path
import json

# Fixture builders copied from the existing Hook-runtime test, not the producer.
def _agent_document():
    document = json.loads((Path(__file__).parent / "fixtures" / "skeleton_execution" / "if_rejoin_workflow.json").read_text())
    document["kind"] = "agent"
    for node in document["nodes"]:
        if node["type"] == "code":
            node["type"] = "agent"
            node["config"]["kernel"] = {
                "schema_version": 1,
                "profile": {"id": "test.profile", "provider": "openai", "model": "test-model",
                            "protocol": "openai-responses", "endpoint": "https://example.test/v1",
                            "credential_ref": "test.key", "context_window": 8192, "output_limit": 2048,
                            "capabilities": {"json_schema": True, "images": False, "files": False}},
                "prompt": "test", "max_iterations": 3, "output_tokens": 512,
                "history": "root", "overflow": "error", "json_schema": None,
            }
    return document

def _entry(*, hook_id="redact", phase="after_tool_call", order=20, force_approval=False):
    return {"id": hook_id, "order": order, "event": phase, "tool_name": "echo",
            "workflow": {"workflow_id": "redaction-workflow", "revision_id": "rev-1"},
            "force_approval": force_approval}

def _with_pool(document, entries):
    value = deepcopy(document)
    value["nodes"].append({"id": "hooks", "type": "hooks_pool", "config": {"hooks": entries}})
    kernel = next(node for node in value["nodes"] if node["type"] == "agent")
    value["edges"].append({"id": "attach-hooks", "kind": "attach",
                           "source_node_id": kernel["id"], "source_port_id": "attach_bot",
                           "target_node_id": "hooks", "target_port_id": "attach_top"})
    return value

def _workflow(workflow_id="redaction-workflow"):
    return {
        "schema_version": 1, "kind": "workflow", "id": workflow_id,
        "name": "Hook workflow",
        "nodes": [
            {"id": "start", "type": "start", "config": {
                "execution": {"outputs": [{"name": "value", "source": {"kind": "result", "path": ["value"]}}]}}},
            {"id": "end", "type": "end", "config": {
                "execution": {
                    "inputs": [{"name": "value", "value": {"kind": "ref", "node_id": "start", "path": ["value"]}, "missing": "error"}],
                    "outputs": [{"name": "value", "source": {"kind": "input", "path": ["value"]}}],
                }}},
        ],
        "edges": [{"id": "start-end", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"}],
    }


def embedded_entry():
    return {**_entry(), "workflow": {"kind": "embedded", "document": _workflow("owned-flow")}}


def test_embedded_body_is_admitted_and_detached_while_pin_wire_is_unchanged():
    entry = embedded_entry()
    answer = validate_hooks_pool_config({"hooks": [entry]}, ("config",))
    assert answer == (entry,)
    entry["workflow"]["document"]["name"] = "changed"
    assert answer[0]["workflow"]["document"]["name"] == "Hook workflow"
    legacy = _entry()
    assert validate_hooks_pool_config({"hooks": [legacy]}, ("config",)) == (legacy,)


def test_embedded_binding_never_uses_published_resolver_and_old_plan_is_fixed():
    raw = _with_pool(_agent_document(), [embedded_entry()])
    def absent(*_args):
        pytest.fail("an owned embedded body is not a published resource reference")
    old = SkeletonHookRuntime(prepare_agent_graph(raw), absent)
    bound = old.bindings_for_node("left")[0]
    assert type(bound.workflow).__name__ == "EmbeddedHookWorkflow"
    assert bound.workflow.workflow_id == "owned-flow"
    assert bound.workflow.revision_id == "embedded-sha256-" + execution_definition_digest(bound.plan)
    raw["nodes"][-1]["config"]["hooks"][0]["workflow"]["document"]["name"] = "new body"
    newer = SkeletonHookRuntime(prepare_agent_graph(raw), absent).bindings_for_node("left")[0]
    assert bound.workflow.revision_id != newer.workflow.revision_id
    assert old.bindings_for_node("left")[0].workflow.document["name"] == "Hook workflow"
    bound.workflow.document["name"] = "caller mutation"
    assert old.bindings_for_node("left")[0].workflow.document["name"] == "Hook workflow"


@pytest.mark.parametrize("mutation,code,tail", [
    ("extra", "unknown_field", ("extra",)),
    ("mixed", "unknown_field", ("workflow_id",)),
    ("future", "unknown_variant", ("kind",)),
    ("agent", "wrong_kind", ("document", "kind")),
    ("missing-document", "missing_field", ("document",)),
    ("invalid-body", "invalid_type", ("document",)),
])
def test_closed_embedded_source_rejects_at_its_own_address(mutation, code, tail):
    entry = embedded_entry(); source = entry["workflow"]
    if mutation == "extra": source["extra"] = True
    if mutation == "mixed": source["workflow_id"] = "pretend"
    if mutation == "future": source["kind"] = "future"
    if mutation == "agent": source["document"]["kind"] = "agent"
    if mutation == "missing-document": del source["document"]
    if mutation == "invalid-body": source["document"] = None
    with pytest.raises(SkeletonValidationError) as err:
        validate_hooks_pool_config({"hooks": [entry]}, ("config",))
    assert err.value.code == code
    assert err.value.path == ("config", "hooks", 0, "workflow") + tail


def test_embedded_workflow_cannot_contain_an_agent_or_any_pool():
    for kind in ["agent", "toolkit_pool", "subagent_pool", "hooks_pool"]:
        entry = embedded_entry()
        entry["workflow"]["document"]["nodes"].append({"id": "forbidden", "type": kind, "config": {}})
        with pytest.raises(SkeletonValidationError) as err:
            validate_hooks_pool_config({"hooks": [entry]}, ("config",))
        assert err.value.path[:5] == ("config", "hooks", 0, "workflow", "document")


def test_embedded_diagnostics_keep_parent_and_sibling_addresses_separate():
    bad = embedded_entry(); good = embedded_entry(); good["id"] = "healthy"
    bad["workflow"]["document"]["nodes"].pop()
    bad["workflow"]["document"]["edges"] = []
    doc = _with_pool(_agent_document(), [bad, good])
    answer = diagnose_skeleton_draft(doc)
    prefix = ("document", "nodes", len(doc["nodes"])-1, "config", "hooks", 0, "workflow", "document")
    assert any(d.code == "end_count" and d.path[:len(prefix)] == prefix for d in answer)
    healthy = prefix[:5] + (1,) + prefix[6:]
    assert not any(d.path[:len(healthy)] == healthy for d in answer)


def test_new_contract_is_closed_detached_and_old_enable_contract_is_unchanged():
    import unchain.skeleton.hooks_config as hooks
    getter = getattr(hooks, "hook_workflow_contract", None)
    assert callable(getter)
    expected = {"schema": "unchain.skeleton_hook_workflow.v1", "field": "workflow",
        "reference_fields": ["revision_id", "workflow_id"], "embedded_fields": ["document", "kind"],
        "embedded_kind": "embedded", "document_kind": "workflow", "identity": "embedded-sha256"}
    assert getter() == expected
    changed = getter(); changed["embedded_fields"].append("extra")
    assert getter() == expected
    assert hooks.hook_entry_contract() == {"schema": "unchain.skeleton_hook_entry.v1",
        "required_fields": ["event", "force_approval", "id", "order", "tool_name", "workflow"],
        "optional_fields": ["enabled"], "omitted_enabled": True}
