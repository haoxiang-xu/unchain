from copy import deepcopy
from types import SimpleNamespace

import pytest

from unchain.journal import AttemptRef, GenerationRef, JournalAppendRequest, build_operation_ref
from unchain.journal.models import _thaw_json
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.kernel_runtime import SkeletonKernelHandler, build_node_bundle
from unchain.skeleton.workflow_runtime import PublishedWorkflow
from tests.test_skeleton_agent_preparation import agent_document


class Stream:
    def __init__(self, tool=False, final_text='{"answer":"processed"}', final_messages=None):
        self.tool = tool
        self.final_text = final_text
        self.final_messages = final_messages
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def __iter__(self):
        if self.tool:
            yield SimpleNamespace(type="response.completed", response=SimpleNamespace(id="tool-response", output=[{"type":"function_call","call_id":"same-call","name":"probe","arguments":"{}"}], usage={"input_tokens":7,"output_tokens":3,"total_tokens":10}))
            return
        yield SimpleNamespace(type="response.output_text.delta", delta=self.final_text)
        messages = self.final_messages or [self.final_text]
        yield SimpleNamespace(type="response.completed", response=SimpleNamespace(
            id="reply", output=[
                {"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":text}]}
                for text in messages
            ],
            usage={"input_tokens":7,"output_tokens":3,"total_tokens":10}))


def runtime_fixture(tmp_path, *, output_schema=None, tool=None, execution_guard=None,
                    kernel_updates=None, final_text='{"answer":"processed"}', final_messages=None,
                    json_output=True, prompt_input=False, hook_entries=None,
                    hook_workflow_resolver=None, lifecycle=None):
    doc = agent_document()
    if hook_entries is not None:
        kernel = next(node for node in doc["nodes"] if node["type"] == "agent")
        doc["nodes"].append({
            "id": "hooks", "type": "hooks_pool", "config": {"hooks": hook_entries},
        })
        doc["edges"].append({
            "id": "attach-hooks", "kind": "attach",
            "source_node_id": kernel["id"], "source_port_id": "attach_bot",
            "target_node_id": "hooks", "target_port_id": "attach_top",
        })
    if prompt_input:
        left = next(node for node in doc["nodes"] if node["id"] == "left")
        left["config"]["execution"]["inputs"].append({
            "name": "flag",
            "value": {"kind": "ref", "node_id": "start", "path": ["flag"]},
            "missing": "error",
        })
    for n in doc["nodes"]:
        if n["type"] == "agent":
            n["config"]["kernel"]["json_schema"] = (
                output_schema or {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":False}
            ) if json_output else None
    for n in doc["nodes"]:
        if n["type"] == "agent":
            n["config"]["kernel"].update(kernel_updates or {})
    disk = SQLiteContextV2Store(database_path=tmp_path / "state.sqlite", object_directory=tmp_path / "objects")
    compiler = SQLiteContextCompilerV2Store(context_store=disk)
    journal = disk.bind_execution("execution")
    store = WorkflowExecutionStore(journal, prepare_agent_graph(doc), graph_id="graph",
                                   attempt=AttemptRef(GenerationRef("execution","generation"),"root"))
    sends=[]
    def bind(config):
        if lifecycle is not None:
            lifecycle.append("bind")
        count = 0
        def send(**kwargs):
            nonlocal count
            count += 1
            sends.append(deepcopy(kwargs))
            return Stream(tool is not None and count == 1, final_text, final_messages)
        return bind_kernel_profile(config, profile_resolver=lambda _:deepcopy(config["profile"]),
            credential_resolver=lambda _:"test-key",
            client_factories={"openai":lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=send))})
    def bundle(attempt, view, config):
        return build_node_bundle(journal=journal, compiler_store=compiler, view=view, config=config,
            parent_attempt_id="root", artifact_sanitizer=lambda content,_media:content,
            payload_sanitizer=lambda _kind,payload:payload, partial_attempt_sink=lambda *args:None)
    handler=SkeletonKernelHandler(store, bind_profile=bind, bundle_builder=bundle,
        tools_for_node=lambda _context: () if tool is None else (tool,), execution_guard=execution_guard,
        hook_workflow_resolver=hook_workflow_resolver)
    return store, handler, sends


def _hook_workflow(workflow_id):
    return {
        "schema_version": 1, "kind": "workflow", "id": workflow_id,
        "name": "Hook workflow",
        "nodes": [
            {
                "id": "start", "type": "start",
                "config": {"execution": {"outputs": [{
                    "name": "value", "source": {"kind": "result", "path": ["value"]},
                }]}},
            },
            {
                "id": "end", "type": "end",
                "config": {"execution": {
                    "inputs": [{
                        "name": "value",
                        "value": {"kind": "ref", "node_id": "start", "path": ["value"]},
                        "missing": "error",
                    }],
                    "outputs": [{
                        "name": "value", "source": {"kind": "input", "path": ["value"]},
                    }],
                }},
            },
        ],
        "edges": [{
            "id": "start-end", "kind": "flow", "source_node_id": "start", "source_port_id": "out",
            "target_node_id": "end", "target_port_id": "in",
        }],
    }


def _hook_entry():
    return {
        "id": "policy", "order": 10, "event": "before_tool_call", "tool_name": None,
        "workflow": {"workflow_id": "policy-workflow", "revision_id": "rev-1"},
        "force_approval": False,
    }


def test_hook_dependencies_are_admitted_before_profiles_or_model_requests(tmp_path):
    lifecycle = []

    def resolve(workflow_id, revision_id):
        lifecycle.append("resolve")
        return PublishedWorkflow(workflow_id, revision_id, _hook_workflow(workflow_id))

    store, handler, sends = runtime_fixture(
        tmp_path, hook_entries=[_hook_entry()], hook_workflow_resolver=resolve,
        lifecycle=lifecycle,
    )

    assert lifecycle[0] == "resolve"
    assert lifecycle[1:] and set(lifecycle[1:]) == {"bind"}
    assert sends == []
    assert handler.hook_runtime.bindings_for_node("left")[0].workflow.revision_id == "rev-1"

    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})
    assert runner.run().status == "completed"
    assert sends


def test_hook_dependencies_require_a_resolver_before_profile_binding(tmp_path):
    lifecycle = []

    with pytest.raises(ValueError, match="hook_workflow_resolver"):
        runtime_fixture(tmp_path, hook_entries=[_hook_entry()], lifecycle=lifecycle)

    assert lifecycle == []


def test_actual_kernel_loop_and_durable_provider_boundary(tmp_path):
    store, handler, sends = runtime_fixture(tmp_path)
    journal = store.journal
    failures=[]
    def invoke(*args):
        try:
            return handler(*args)
        except Exception:
            import traceback
            failures.append(traceback.format_exc())
            raise
    runner=DurableWorkflowRunner(store,{"agent":invoke})
    runner.create({"flag":True})
    result=runner.run()
    assert result.status == "completed", "\n".join(failures)
    assert result.outputs == {"answer":"processed"}
    assert len(sends) == 2
    assert all(send["max_output_tokens"] == 512 for send in sends)
    assert runner.run().outputs == result.outputs
    assert len(sends) == 2
    events=journal.capture_snapshot().events
    assert all(e.payload.get("parent_run_id") == "root" for e in events if e.event_type in {"message.user","final_message","run_completed"})
    results=[store.read_json(_thaw_json(e.payload)["result_ref"]) for e in events if e.event_type == "skeleton.kernel_result.saved"]
    assert len(results)==2
    assert results[0]["context"]["usage"]["total_tokens"]==10
    assert "messages" not in results[0]["context"]


def test_kernel_prompt_resolves_bound_input_before_provider_request(tmp_path):
    store, handler, sends = runtime_fixture(
        tmp_path, kernel_updates={"prompt": "Review {{#input.flag#}}"}, prompt_input=True,
    )
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})
    assert runner.run().status == "completed"
    assert sends[0]["input"][0] == {"role": "system", "content": "Review True"}


class SimulatedProcessDeath(BaseException):
    pass


def test_cold_recovery_imports_saved_kernel_result_without_resending(tmp_path, monkeypatch):
    store, handler, sends = runtime_fixture(tmp_path)
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})
    original = store.save
    def fail_graph_raw(state, **kwargs):
        if state["status"] == "result_received":
            raise SimulatedProcessDeath()
        return original(state, **kwargs)
    monkeypatch.setattr(store, "save", fail_graph_raw)
    with pytest.raises(SimulatedProcessDeath):
        runner.run()
    assert len(sends) == 1
    assert runner.inspect().status == "running"
    cold_store, cold_handler, cold_sends = runtime_fixture(tmp_path)
    cold = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
    assert cold.recover().status == "result_received"
    assert cold.run().outputs == {"answer": "processed"}
    assert len(cold_sends) == 1  # only the downstream node
    assert cold.recover().status == "completed"
    assert len(sends) + len(cold_sends) == 2


def test_cold_recovery_imports_completed_kernel_receipts_before_handler_result(tmp_path, monkeypatch):
    store, handler, sends = runtime_fixture(tmp_path)
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})

    def fail_handler_receipt(_context, _output):
        raise SimulatedProcessDeath()

    monkeypatch.setattr(handler, "_save_completion", fail_handler_receipt)
    with pytest.raises(SimulatedProcessDeath):
        runner.run()
    active = store.load().state["active"]
    terminal = [
        event.event_type
        for event in store.journal.capture_snapshot().events
        if event.attempt.attempt_id == active["node_attempt_id"]
        and event.event_type in {"provider.turn_result", "final_message", "run_completed"}
    ]
    assert terminal == ["provider.turn_result", "final_message", "run_completed"]
    assert runner.inspect().status == "running"
    assert len(sends) == 1

    cold_store, cold_handler, cold_sends = runtime_fixture(tmp_path)
    cold = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
    assert cold.recover().status == "result_received"
    recovered = [
        cold_store.read_json(_thaw_json(event.payload)["result_ref"])
        for event in cold_store.journal.capture_snapshot().events
        if event.event_type == "skeleton.kernel_result.saved"
    ]
    assert len(recovered) == 1
    assert recovered[0]["message"] == '{"answer":"processed"}'
    assert recovered[0]["context"]["provider"] == "openai"
    assert recovered[0]["context"]["model"] == "test-model"
    assert recovered[0]["context"]["profile_id"] == "openai.test"
    assert recovered[0]["context"]["usage"]["total_tokens"] == 10
    assert cold.run().outputs == {"answer": "processed"}
    assert len(cold_sends) == 1  # the downstream Kernel only
    assert len(sends) + len(cold_sends) == 2


@pytest.mark.parametrize("messages", [
    ["Preamble.", "  Final answer.\n"],
    ["Preamble.", " \t\n"],
])
def test_cold_recovery_preserves_last_message_from_multi_message_provider_result(
    tmp_path, monkeypatch, messages,
):
    if messages[-1].startswith(" ") and "Final" not in messages[-1]:
        assert not messages[-1].strip()
    normal_root = tmp_path / "normal"
    normal_root.mkdir()
    normal_store, normal_handler, _ = runtime_fixture(
        normal_root, final_messages=messages, json_output=False,
    )
    normal = DurableWorkflowRunner(normal_store, {"agent": normal_handler})
    normal.create({"flag": True})
    while normal.inspect().status == "ready":
        normal.step()
    normal_message = normal_store.read_json(
        normal_store.load().state["active"]["result_ref"]
    )["message"]
    assert normal_message == messages[-1]

    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    store, handler, sends = runtime_fixture(
        recovery_root, final_messages=messages, json_output=False,
    )
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})

    def fail_handler_receipt(_context, _output):
        raise SimulatedProcessDeath()

    monkeypatch.setattr(handler, "_save_completion", fail_handler_receipt)
    with pytest.raises(SimulatedProcessDeath):
        runner.run()
    assert len(sends) == 1

    cold_store, cold_handler, cold_sends = runtime_fixture(
        recovery_root, final_messages=messages, json_output=False,
    )
    cold = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
    assert cold.recover().status == "result_received"
    recovered = cold_store.read_json(cold_store.load().state["active"]["result_ref"])
    assert recovered["message"] == normal_message
    assert cold_sends == []


@pytest.mark.parametrize("raw_text", [
    '  {"answer":"processed"}\n',
    " \t\n",
])
def test_cold_recovery_preserves_exact_kernel_message(tmp_path, monkeypatch, raw_text):
    normal_root = tmp_path / "normal"
    normal_root.mkdir()
    normal_store, normal_handler, normal_sends = runtime_fixture(
        normal_root, final_text=raw_text,
    )
    normal = DurableWorkflowRunner(normal_store, {"agent": normal_handler})
    normal.create({"flag": True})
    while normal.inspect().status == "ready":
        normal.step()
    assert normal.inspect().status == "result_received"
    normal_message = normal_store.read_json(
        normal_store.load().state["active"]["result_ref"]
    )["message"]

    recovery_root = tmp_path / "recovery"
    recovery_root.mkdir()
    store, handler, sends = runtime_fixture(recovery_root, final_text=raw_text)
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})

    def fail_handler_receipt(_context, _output):
        raise SimulatedProcessDeath()

    monkeypatch.setattr(handler, "_save_completion", fail_handler_receipt)
    with pytest.raises(SimulatedProcessDeath):
        runner.run()

    cold_store, cold_handler, cold_sends = runtime_fixture(
        recovery_root, final_text=raw_text,
    )
    cold = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
    assert cold.recover().status == "result_received"
    recovered_message = cold_store.read_json(
        cold_store.load().state["active"]["result_ref"]
    )["message"]

    assert normal_message == recovered_message == raw_text
    assert len(normal_sends) == len(sends) == 1
    assert cold_sends == []


def test_cold_recovery_rejects_ambiguous_kernel_terminal_evidence(tmp_path, monkeypatch):
    store, handler, sends = runtime_fixture(tmp_path)
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"flag": True})

    def fail_handler_receipt(_context, _output):
        raise SimulatedProcessDeath()

    monkeypatch.setattr(handler, "_save_completion", fail_handler_receipt)
    with pytest.raises(SimulatedProcessDeath):
        runner.run()
    final = next(
        event for event in store.journal.capture_snapshot().events
        if event.event_type == "final_message"
    )
    payload = _thaw_json(final.payload)
    operation_id = "test.duplicate-kernel-final"
    store.journal.append(request=JournalAppendRequest(
        event_id="event." + operation_id,
        event_type="final_message",
        attempt=final.attempt,
        operation=build_operation_ref(
            operation_id,
            domain="test.kernel_recovery",
            payload=payload,
        ),
        payload=payload,
    ))

    cold_store, cold_handler, cold_sends = runtime_fixture(tmp_path)
    cold = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
    with pytest.raises(ValueError, match="incomplete or ambiguous"):
        cold.recover()
    assert cold.inspect().status == "running"
    assert len(sends) == 1
    assert cold_sends == []


def test_invalid_json_output_keeps_raw_result_and_never_retries_model(tmp_path):
    store, handler, sends = runtime_fixture(tmp_path, output_schema={"type":"object", "properties":{"answer":{"type":"integer"}}, "required":["answer"]})
    runner = DurableWorkflowRunner(store, {"agent":handler})
    runner.create({"flag":True})
    result = runner.run()
    assert result.status == "failed"
    assert len(sends) == 1
    active = store.load().state["active"]
    assert store.read_json(active["result_ref"])["message"] == '{"answer":"processed"}'
    assert runner.resume().status == "failed"
    assert runner.recover().status == "failed"
    assert len(sends) == 1


def test_model_tool_turn_stays_private_across_two_kernel_nodes(tmp_path):
    from unchain.execution import ExecutionRuntime
    from unchain.memory import InMemorySessionStore
    from unchain.tools import Tool
    calls = []
    def probe():
        calls.append("tool")
        return {"private": "PRIVATE_TOOL_BYTES"}
    runtime = ExecutionRuntime(InMemorySessionStore())
    with runtime.scope("execution") as guard:
        store, handler, sends = runtime_fixture(tmp_path,
            tool=Tool.from_callable(probe, name="probe", output_policy="artifact_only"), execution_guard=guard)
        runner = DurableWorkflowRunner(store,{"agent":handler})
        runner.create({"flag":True})
        result = runner.run()
        assert result.status == "completed", result
    assert calls == ["tool", "tool"]
    assert len(sends) == 4
    assert "PRIVATE_TOOL_BYTES" not in repr(sends)
    assert "function_call_output" in repr(sends[1])
    assert "function_call_output" not in repr(sends[2])
    assert runner.run().status == "completed"
    assert calls == ["tool", "tool"]


@pytest.mark.parametrize("updates", [{"prompt":"mandatory " * 20000}, {"output_tokens":1900, "profile":{"id":"small", "provider":"openai", "model":"test-model", "protocol":"openai-responses", "endpoint":"https://example.test/v1", "credential_ref":"key", "context_window":2000, "output_limit":1950, "capabilities":{"json_schema":True,"images":True,"files":False}}}])
def test_required_input_or_output_reservation_overflow_fails_before_intent(tmp_path, updates):
    store, handler, sends = runtime_fixture(tmp_path, kernel_updates=updates)
    runner = DurableWorkflowRunner(store,{"agent":handler})
    runner.create({"flag":True})
    result=runner.run()
    assert result.status=="failed"
    assert result.error["code"]=="context_overflow"
    assert store.load().state["active"] is None
    assert sends==[]
