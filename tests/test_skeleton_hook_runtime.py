"""CP16 Hook dependencies are pinned before a Kernel can start."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hook_runtime import SkeletonHookRuntime
from unchain.skeleton.preparation import prepare_workflow
from unchain.skeleton.workflow_runtime import PublishedWorkflow



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


def test_hook_dependencies_are_resolved_once_and_pinned_before_kernel_dispatch():
    document = _with_pool(_agent_document(), [
        _entry(hook_id="redact", phase="after_tool_call", order=20),
        _entry(hook_id="gate", phase="before_tool_call", order=10, force_approval=True),
    ])
    requested = []

    def resolver(workflow_id, revision_id):
        requested.append((workflow_id, revision_id))
        return PublishedWorkflow(workflow_id, revision_id, _workflow(workflow_id))

    runtime = SkeletonHookRuntime(prepare_agent_graph(document), resolver)

    bindings = runtime.bindings_for_node("left")
    assert [binding.hook["id"] for binding in bindings] == ["gate", "redact"]
    assert [binding.workflow.revision_id for binding in bindings] == ["rev-1", "rev-1"]
    assert requested == [("redaction-workflow", "rev-1")] * 2
    bindings[0].hook["workflow"]["revision_id"] = "mutated"
    assert runtime.bindings_for_node("left")[0].hook["workflow"]["revision_id"] == "rev-1"


@pytest.mark.parametrize("returned", [
    PublishedWorkflow("redaction-workflow", "rev-1", {"kind": "agent"}),
    PublishedWorkflow("wrong-id", "rev-1", _workflow()),
    PublishedWorkflow("redaction-workflow", "wrong-revision", _workflow()),
])
def test_hook_preflight_rejects_non_exact_or_non_workflow_dependencies(returned):
    document = _with_pool(_agent_document(), [_entry()])
    with pytest.raises(SkeletonValidationError):
        SkeletonHookRuntime(prepare_agent_graph(document), lambda *_args: deepcopy(returned))


def test_hook_preflight_rejects_a_structural_workflow_the_real_runner_cannot_execute(tmp_path):
    workflow = _workflow()
    workflow["nodes"][-1]["config"]["execution"]["outputs"][0]["source"]["kind"] = "result"
    workflow_plan = prepare_workflow(workflow)
    journal = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    ).bind_execution("execution")
    store = WorkflowExecutionStore(
        journal,
        workflow_plan,
        graph_id="hook-child",
        attempt=AttemptRef(GenerationRef("execution", "generation"), "hook-child"),
    )
    with pytest.raises(SkeletonValidationError):
        DurableWorkflowRunner(store, {})

    document = _with_pool(_agent_document(), [_entry()])
    with pytest.raises(SkeletonValidationError):
        SkeletonHookRuntime(
            prepare_agent_graph(document),
            lambda *_args: PublishedWorkflow("redaction-workflow", "rev-1", workflow),
        )


def test_hook_ids_are_unique_across_all_pools_attached_to_one_kernel():
    document = _with_pool(_agent_document(), [_entry()])
    second_pool = deepcopy(document["nodes"][-1])
    second_pool["id"] = "second-hooks"
    document["nodes"].append(second_pool)
    second_edge = deepcopy(document["edges"][-1])
    second_edge["id"] = "attach-second-hooks"
    second_edge["target_node_id"] = "second-hooks"
    document["edges"].append(second_edge)

    with pytest.raises(SkeletonValidationError) as error:
        SkeletonHookRuntime(
            prepare_agent_graph(document),
            lambda *_args: PublishedWorkflow(
                "redaction-workflow", "rev-1", _workflow(),
            ),
        )
    assert error.value.code == "duplicate_name"


def test_duplicate_attachment_cannot_schedule_one_hook_twice():
    document = _with_pool(_agent_document(), [_entry()])
    duplicate = deepcopy(document["edges"][-1])
    duplicate["id"] = "duplicate-attachment"
    document["edges"].append(duplicate)

    with pytest.raises(SkeletonValidationError) as error:
        SkeletonHookRuntime(
            prepare_agent_graph(document),
            lambda *_args: PublishedWorkflow(
                "redaction-workflow", "rev-1", _workflow(),
            ),
        )
    assert error.value.code == "duplicate_name"
