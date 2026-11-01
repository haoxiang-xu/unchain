"""A denied parent may park an after-Hook without authorizing the parent Tool."""
from copy import deepcopy
from dataclasses import replace

import pytest

import tests.test_skeleton_kernel_interaction as base
from unchain.context.runtime import ContextRuntime
from unchain.context.tool_boundary import DurableToolRouteKind
from unchain.context.tool_executor import DurableToolExecutorContractError
from unchain.execution import ExecutionRuntime
from unchain.interaction.durable import build_interaction_request
from unchain.interaction.runtime import response_contract_for_kind
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.subagents.types import SubagentWaitSignal


@pytest.mark.parametrize("approved,fault", [
    (False, None), (True, None), (False, "terminal"),
    (False, "route"), (False, "kind"), (False, "call"),
])
def test_after_hook_child_wait_retains_denial_terminal_authority(tmp_path, monkeypatch, approved, fault):
    document = base._document_with_before_hook()
    hooks = next(node for node in document["nodes"] if node["type"] == "hooks_pool")["config"]["hooks"]
    hooks[0]["event"] = "after_tool_call"
    sessions = JsonFileSessionStore(tmp_path / "sessions")
    parent_effects, hook_effects = [], []
    native = base._StrictOpenAIClient([base._tool_response(), base._final_response()])

    def hook_runner(*, binding, inputs, parent_context, **kwargs):
        if kwargs.get("hook_resume") is not None:
            hook_effects.append("approved audit")
            return {"decision": "continue"}
        child = build_interaction_request(
            session_id="interaction-session", kind="tool_approval", source_run_id="audit-child",
            occurrence="audit", payload={"tool_name": "audit", "call_id": "audit-call"},
            response_contract=response_contract_for_kind("tool_approval"), created_revision=0,
        )
        call = {"call_id": inputs["call_id"], "tool_name": inputs["tool_name"]}
        raw = {
            "schema": "unchain.hook_child_wait.v1", "parent_graph_id": "interaction-graph",
            "parent_node_invocation_id": parent_context.node_invocation_id,
            "parent_tool_call": call, "hook_id": binding.hook["id"],
            "workflow_id": binding.workflow.workflow_id, "revision_id": binding.workflow.revision_id,
            "definition_digest": "a" * 64, "child_graph_id": "audit-child",
            "child_attempt_id": "audit-attempt", "child_node_invocation_id": "audit-node",
            "child_wait_kind": "interaction", "child_wait_id": child.interaction_id,
            "interaction_request": child.to_dict(),
        }
        raise SubagentWaitSignal(child.interaction_id, raw, parent_tool_call=call)

    project = ContextRuntime._project_after_hook_child_wait
    projection_errors = []

    def project_fault(self, context, binding, bundle, wait):
        if fault == "terminal":
            binding = replace(binding, request=replace(binding.request, subject=replace(
                binding.request.subject, terminal_handler_manifest_sha256="0" * 64)))
        elif fault == "route":
            binding = replace(binding, request=replace(binding.request, subject=replace(
                binding.request.subject, route_manifest_sha256="0" * 64)))
        elif fault == "kind":
            binding = replace(binding, route=replace(binding.route, route_kind=DurableToolRouteKind.PLUGIN))
        elif fault == "call":
            binding = replace(binding, request=replace(binding.request, call_id="another-call"))
        try:
            return project(self, context, binding, bundle, wait)
        except DurableToolExecutorContractError as error:
            projection_errors.append(str(error))
            raise

    monkeypatch.setattr(ContextRuntime, "_project_after_hook_child_wait", project_fault)

    def stack(guard):
        return base._stack(tmp_path, memory_runtime=KernelMemoryRuntime.from_config(store=sessions),
            client=native, guard=guard, tool_calls=parent_effects, document=document,
            hook_workflow_runner=hook_runner)

    with ExecutionRuntime(sessions).scope("interaction-session") as guard:
        store, handler = stack(guard)
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        assert runner.run().status == "waiting"
        original = deepcopy(store.load().state["active"])
    with ExecutionRuntime(sessions).scope("interaction-session") as guard:
        store, handler = stack(guard)
        runner = DurableWorkflowRunner(store, {"agent": handler})
        result = runner.resume_kernel_interaction(original["wait"]["id"], approved,
            node_invocation_id=original["node_invocation_id"])
        if fault:
            assert result.status == "outcome_unknown"
            expected = "after Hook wait has no parent intent" if fault == "call" else "after Hook route changed"
            assert projection_errors == [expected]
            assert parent_effects == hook_effects == []
            assert len(native.requests) == 1
            return
        assert result.status == "waiting", result.error
        assert parent_effects == (["approved value"] if approved else [])
        assert hook_effects == []
        waiting = deepcopy(store.load().state["active"])
        before = [event.to_dict() for event in store.journal.capture_snapshot().events]
    with ExecutionRuntime(sessions).scope("interaction-session") as guard:
        store, handler = stack(guard)
        runner = DurableWorkflowRunner(store, {"agent": handler})
        result = runner.resume_kernel_interaction(waiting["wait"]["id"], True,
            node_invocation_id=waiting["node_invocation_id"])
        if result.status == "result_received":
            result = runner.run()
        assert result.status == "completed", result.error
        events = [event.to_dict() for event in store.journal.capture_snapshot().events]
        assert events[:len(before)] == before
        receipts = [event for event in events if event["event_type"] == "tool_result"
            and event["payload"].get("call_id") == "approval-call"]
        assert len(receipts) == 1
        assert receipts[0]["payload"]["execution_subject"]["approval_state"] == ("approved" if approved else "denied")
        if not approved:
            assert receipts[0]["payload"]["result"]["denied"] is True
    assert parent_effects == (["approved value"] if approved else [])
    assert hook_effects == ["approved audit"]
    assert projection_errors == []
    assert len(native.requests) == 2
