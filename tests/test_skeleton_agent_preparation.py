from copy import deepcopy
import json
from pathlib import Path

import pytest

from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_workflow, WorkflowNodeCompleted
from unchain.skeleton.agent_preparation import prepare_agent_graph, execution_definition_digest
from unchain.skeleton.subagent_config import attached_subagents
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.recovery import workflow_definition_digest


def kernel_config(provider="openai", protocol="openai-responses"):
    return {
        "schema_version": 1,
        "profile": {"id": provider + ".test", "provider": provider, "model": "test-model",
                    "protocol": protocol, "endpoint": "https://example.test/v1",
                    "credential_ref": provider + ".key", "context_window": 8192,
                    "output_limit": 2048,
                    "capabilities": {"json_schema": True, "images": True, "files": False}},
        "prompt": "test instruction", "max_iterations": 3, "output_tokens": 512,
        "history": "root", "overflow": "error", "json_schema": None,
    }


def agent_document():
    doc = json.loads((Path(__file__).parent / "fixtures/skeleton_execution/if_rejoin_workflow.json").read_text())
    doc["kind"] = "agent"
    for node in doc["nodes"]:
        if node["type"] == "code":
            node["type"] = "agent"
            node["config"]["kernel"] = kernel_config()
    return doc


def child_agent(resource_id="child", provider="anthropic"):
    child = agent_document()
    child["id"] = resource_id
    child["name"] = resource_id.title()
    for node in child["nodes"]:
        if node["type"] == "agent":
            node["config"]["kernel"] = kernel_config(provider, "anthropic")
    return child


def with_subagent_pool(document, source):
    result = deepcopy(document)
    result["nodes"].append({
        "id": "children", "type": "subagent_pool", "config": {"subagents": [{
            "id": "reviewer", "name": "Reviewer", "description": "Review one task",
            "source": source,
            "allowed_modes": ["delegate", "handoff", "worker"],
            "output_mode": "summary", "memory_policy": "ephemeral",
            "parallel_safe": True, "allowed_tools": None, "disabled_tools": [],
        }]},
    })
    kernel = next(node for node in result["nodes"] if node["type"] == "agent")
    result["edges"].append({
        "id": "attach-children", "kind": "attach",
        "source_node_id": kernel["id"], "source_port_id": "attach_bot",
        "target_node_id": "children", "target_port_id": "attach_top",
    })
    return result


def test_agent_plan_is_distinct_and_workflow_v1_stays_closed():
    doc = agent_document()
    plan = prepare_agent_graph(doc)
    assert plan.snapshot() == doc
    with pytest.raises(SkeletonValidationError):
        prepare_workflow(doc)
    with pytest.raises(SkeletonValidationError):
        workflow_definition_digest(plan)
    old_digest = execution_definition_digest(plan)
    doc["nodes"][2]["config"]["kernel"]["profile"]["credential_ref"] = "different-key"
    assert execution_definition_digest(prepare_agent_graph(doc)) != old_digest
    assert execution_definition_digest(plan) == old_digest


def test_subagent_pool_accepts_closed_refs_and_embedded_agent_graphs():
    referenced = with_subagent_pool(agent_document(), {"kind": "agent_ref", "agent_id": "shared-reviewer"})
    ref_plan = prepare_agent_graph(referenced)
    assert attached_subagents(ref_plan.snapshot())["left"][0]["source"] == {
        "kind": "agent_ref", "agent_id": "shared-reviewer",
    }

    embedded = with_subagent_pool(agent_document(), {
        "kind": "embedded", "document": child_agent(),
    })
    embedded_plan = prepare_agent_graph(embedded)
    assert attached_subagents(embedded_plan.snapshot())["left"][0]["source"]["document"]["kind"] == "agent"


@pytest.mark.parametrize("mutate", [
    lambda entry: entry["source"].update({"kind": "workflow_ref", "workflow_id": "bad"}),
    lambda entry: entry.update({"allowed_modes": ["delegate", "delegate"]}),
    lambda entry: entry.update({"memory_policy": "root"}),
    lambda entry: entry.update({"unknown": True}),
])
def test_subagent_pool_rejects_ambiguous_or_open_entries(mutate):
    document = with_subagent_pool(agent_document(), {"kind": "agent_ref", "agent_id": "shared-reviewer"})
    entry = document["nodes"][-1]["config"]["subagents"][0]
    mutate(entry)
    with pytest.raises(SkeletonValidationError):
        prepare_agent_graph(document)


def test_embedded_subagent_must_be_an_executable_agent_and_depth_is_bounded():
    wrong_kind = child_agent()
    wrong_kind["kind"] = "workflow"
    with pytest.raises(SkeletonValidationError) as wrong:
        prepare_agent_graph(with_subagent_pool(agent_document(), {
            "kind": "embedded", "document": wrong_kind,
        }))
    assert wrong.value.code in {"invalid_value", "wrong_kind"}

    nested = child_agent("level-0")
    for depth in range(7):
        nested = with_subagent_pool(child_agent(f"level-{depth + 1}"), {
            "kind": "embedded", "document": nested,
        })
    with pytest.raises(SkeletonValidationError) as too_deep:
        prepare_agent_graph(nested)
    assert too_deep.value.code == "subagent_depth"


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("output_tokens", 0),
    ("output_tokens", 2049), ("max_iterations", False), ("history", "all"),
    ("overflow", "truncate"), ("prompt", None),
])
def test_invalid_kernel_preflight_even_on_unselected_branch(field, value):
    doc = agent_document()
    right = next(n for n in doc["nodes"] if n["id"] == "right")
    right["config"]["kernel"][field] = value
    with pytest.raises(SkeletonValidationError):
        prepare_agent_graph(doc)


@pytest.mark.parametrize("field,value", [
    ("protocol", "openai-chat"), ("context_window", False), ("output_limit", 0),
    ("endpoint", "https://user:secret@example.test"), ("endpoint", "https://example.test?api_key=secret"),
    ("endpoint", "file:///private"), ("credential_ref", ""),
    ("capabilities", {"json_schema": "yes", "images": True, "files": False}),
])
def test_profile_admission(field, value):
    doc = agent_document()
    doc["nodes"][2]["config"]["kernel"]["profile"][field] = value
    with pytest.raises(SkeletonValidationError):
        prepare_agent_graph(doc)


def test_typed_agent_uses_existing_durable_receipts_without_workflow_disguise(tmp_path):
    doc = agent_document()
    for node in doc["nodes"]:
        if node["type"] == "agent":
            node["config"]["kernel"]["json_schema"] = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    plan = prepare_agent_graph(doc)
    journal = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite", object_directory=tmp_path / "objects").bind_execution("execution")
    attempt = AttemptRef(GenerationRef("execution", "generation"), "attempt")
    def runner(handler):
        return DurableWorkflowRunner(WorkflowExecutionStore(journal, plan, graph_id="agent", attempt=attempt), {"agent": handler})
    calls = []
    def handler(context, inputs):
        calls.append(context.node_id)
        return WorkflowNodeCompleted({"message": json.dumps({"answer": context.node_id}), "context": {}})
    run = runner(handler)
    run.create({"flag": True})
    assert run.run().outputs == {"answer": "left"}
    assert calls == ["left", "join"]
    assert run.plan.snapshot()["kind"] == "agent"
    assert runner(lambda *_: pytest.fail("cold completed node dispatched")).run().status == "completed"
    changed = deepcopy(doc)
    changed["nodes"][2]["config"]["kernel"]["profile"]["endpoint"] = "https://other.example.test/v1"
    store = WorkflowExecutionStore(journal, prepare_agent_graph(changed), graph_id="agent", attempt=attempt)
    with pytest.raises(SkeletonValidationError):
        store.load()
