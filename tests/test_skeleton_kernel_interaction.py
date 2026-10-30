"""Durable Kernel interaction resumption through the actual Agent boundary."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from unchain.execution import ExecutionLeaseError, ExecutionRuntime
from unchain.journal import AttemptRef, GenerationRef
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.skeleton import WorkflowNodeCompleted
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.skeleton.kernel_runtime import SkeletonKernelHandler, build_node_bundle
from unchain.tools import Tool
from unchain.skeleton.workflow_runtime import PublishedWorkflow


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _document():
    kernel = {
        "schema_version": 1,
        "profile": {
            "id": "interaction.openai",
            "provider": "openai",
            "model": "strict-test-model",
            "protocol": "openai-responses",
            "endpoint": "https://example.test/v1",
            "credential_ref": "interaction-key",
            "context_window": 8192,
            "output_limit": 1024,
            "capabilities": {"json_schema": True, "images": False, "files": False},
        },
        "prompt": "Use the approved tool and return JSON.",
        "max_iterations": 3,
        "output_tokens": 256,
        "history": "root",
        "overflow": "error",
        "json_schema": deepcopy(_SCHEMA),
    }
    return {
        "schema_version": 1,
        "kind": "agent",
        "id": "kernel-interaction",
        "name": "Kernel interaction",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": []}}},
            {
                "id": "agent",
                "type": "agent",
                "config": {
                    "execution": {
                        "inputs": [],
                        "outputs": [{"name": "answer", "source": {"kind": "result", "path": ["answer"]}}],
                    },
                    "kernel": kernel,
                },
            },
            {
                "id": "end",
                "type": "end",
                "config": {
                    "execution": {
                        "inputs": [{
                            "name": "answer",
                            "value": {"kind": "ref", "node_id": "agent", "path": ["answer"]},
                            "missing": "error",
                        }],
                        "outputs": [{"name": "answer", "source": {"kind": "input", "path": ["answer"]}}],
                    }
                },
            },
        ],
        "edges": [
            {"id": "start-agent", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "agent", "target_port_id": "in"},
            {"id": "agent-end", "kind": "flow", "source_node_id": "agent", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


def _hook_workflow():
    """A valid pinned Workflow; its runner is supplied by the Hook host."""
    return {
        "schema_version": 1,
        "kind": "workflow",
        "id": "before-tool-policy",
        "name": "Before tool policy",
        "nodes": [
            {
                "id": "start",
                "type": "start",
                "config": {
                    "execution": {
                        "outputs": [
                            {
                                "name": "event",
                                "source": {"kind": "result", "path": ["event"]},
                            }
                        ]
                    }
                },
            },
            {
                "id": "end",
                "type": "end",
                "config": {
                    "execution": {
                        "inputs": [
                            {
                                "name": "event",
                                "value": {
                                    "kind": "ref",
                                    "node_id": "start",
                                    "path": ["event"],
                                },
                                "missing": "error",
                            }
                        ],
                        "outputs": [
                            {
                                "name": "event",
                                "source": {"kind": "input", "path": ["event"]},
                            }
                        ],
                    }
                },
            },
        ],
        "edges": [
            {
                "id": "start-end",
                "kind": "flow",
                "source_node_id": "start",
                "source_port_id": "out",
                "target_node_id": "end",
                "target_port_id": "in",
            }
        ],
    }


def _document_with_before_hook(*, force_approval=False):
    document = _document()
    document["nodes"].append(
        {
            "id": "hooks",
            "type": "hooks_pool",
            "config": {
                "hooks": [
                    {
                        "id": "policy",
                        "order": 10,
                        "event": "before_tool_call",
                        "tool_name": "dangerous_tool",
                        "workflow": {
                            "workflow_id": "before-tool-policy",
                            "revision_id": "rev-1",
                        },
                        "force_approval": force_approval,
                    }
                ]
            },
        }
    )
    document["edges"].append(
        {
            "id": "attach-hooks",
            "kind": "attach",
            "source_node_id": "agent",
            "source_port_id": "attach_bot",
            "target_node_id": "hooks",
            "target_port_id": "attach_top",
        }
    )
    return document


class _OpenAIStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.completed", response=self.response)


class _StrictOpenAIClient:
    def __init__(self, responses):
        self.responses_to_send = list(responses)
        self.requests = []
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **request):
        self.requests.append(deepcopy(request))
        if not self.responses_to_send:
            raise AssertionError("unexpected native OpenAI request")
        return _OpenAIStream(self.responses_to_send.pop(0))


class _SimulatedProcessDeath(BaseException):
    pass


def _tool_response(*, call_id="approval-call", value="approved value"):
    return SimpleNamespace(
        id="approval-tool-turn",
        output=[{
            "type": "function_call",
            "call_id": call_id,
            "name": "dangerous_tool",
            "arguments": '{"value":"' + value + '"}',
        }],
        usage={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )


def _final_response():
    return SimpleNamespace(
        id="approval-final-turn",
        output=[{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": '{"answer":"approved result"}'}],
        }],
        usage={"input_tokens": 6, "output_tokens": 3, "total_tokens": 9},
    )


def _stack(tmp_path, *, memory_runtime, client, guard, tool_calls, prompt=None,
           document=None, hook_workflow_runner=None,
           tool_requires_confirmation=True):
    disk = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    )
    compiler = SQLiteContextCompilerV2Store(context_store=disk)
    journal = disk.bind_execution("interaction-session")
    document = _document() if document is None else deepcopy(document)
    if prompt is not None:
        next(
            node for node in document["nodes"] if node["type"] == "agent"
        )["config"]["kernel"]["prompt"] = prompt
    plan = prepare_agent_graph(document)
    attempt = AttemptRef(GenerationRef("interaction-session", "generation"), "root")
    store = WorkflowExecutionStore(journal, plan, graph_id="interaction-graph", attempt=attempt)

    def bind(config):
        profile = deepcopy(config["profile"])
        return bind_kernel_profile(
            config,
            profile_resolver=lambda profile_id: profile if profile_id == profile["id"] else None,
            credential_resolver=lambda credential_ref: "test-key" if credential_ref == "interaction-key" else None,
            client_factories={"openai": lambda **_kwargs: client},
        )

    def bundle(*, attempt, view, config):
        return build_node_bundle(
            journal=journal,
            compiler_store=compiler,
            view=view,
            config=config,
            parent_attempt_id="root",
            artifact_sanitizer=lambda content, _media: content,
            payload_sanitizer=lambda _kind, payload: payload,
            partial_attempt_sink=lambda *_args: None,
        )

    def dangerous_tool(value: str):
        tool_calls.append(value)
        return {"applied": value}

    tool = Tool.from_callable(
        dangerous_tool,
        name="dangerous_tool",
        requires_confirmation=tool_requires_confirmation,
    )
    def resolve_hook_workflow(workflow_id, revision_id):
        assert (workflow_id, revision_id) == ("before-tool-policy", "rev-1")
        return PublishedWorkflow(workflow_id, revision_id, _hook_workflow())
    handler = SkeletonKernelHandler(
        store,
        bind_profile=bind,
        bundle_builder=bundle,
        tools_for_node=lambda _context: (tool,),
        execution_guard=guard,
        durability_runtime=memory_runtime,
        hook_workflow_resolver=(
            resolve_hook_workflow if hook_workflow_runner is not None else None
        ),
        hook_workflow_runner=hook_workflow_runner,
    )
    return store, handler


def test_before_hook_modifies_arguments_and_requires_one_parent_approval(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    observed = []

    def run_hook(*, binding, inputs, parent_context, parent_store):
        observed.append((binding.hook["id"], deepcopy(inputs), parent_context.node_id, parent_store.graph_id))
        return {
            "decision": "continue",
            "arguments": {"value": "hooked value"},
            "require_approval": False,
        }

    first_client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
            document=_document_with_before_hook(force_approval=True),
            hook_workflow_runner=run_hook,
            tool_requires_confirmation=False,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        waiting = first_runner.run()

    assert waiting.status == "waiting"
    assert tool_calls == []
    assert observed == [
        (
            "policy",
            {
                "event": "before_tool_call",
                "call_id": "approval-call",
                "tool_name": "dangerous_tool",
                "arguments": {"value": "approved value"},
            },
            "agent",
            "interaction-graph",
        )
    ]
    wait_id = first_store.load().state["active"]["wait"]["id"]
    node_invocation_id = first_store.load().state["active"]["node_invocation_id"]

    final_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        resumed_store, resumed_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
            document=_document_with_before_hook(force_approval=True),
            hook_workflow_runner=run_hook,
            tool_requires_confirmation=False,
        )
        resumed_runner = DurableWorkflowRunner(resumed_store, {"agent": resumed_handler})
        assert resumed_runner.resume_kernel_interaction(
            wait_id,
            True,
            node_invocation_id=node_invocation_id,
        ).status == "result_received"
        assert resumed_runner.run().outputs == {"answer": "approved result"}

    assert tool_calls == ["hooked value"]


def test_before_hook_block_prevents_the_tool_side_effect(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    hook_calls = []

    def run_hook(*, binding, inputs, parent_context, parent_store):
        del binding, inputs, parent_context, parent_store
        hook_calls.append("called")
        return {"decision": "block", "reason": "policy denied"}

    # Blocked results remain observable: the existing tool observer gets one
    # turn before the primary model receives the final tool-result message.
    client = _StrictOpenAIClient(
        [_tool_response(), _final_response(), _final_response()]
    )
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=client,
            guard=guard,
            tool_calls=tool_calls,
            document=_document_with_before_hook(),
            hook_workflow_runner=run_hook,
            tool_requires_confirmation=False,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        result = runner.run()

    assert result.status == "completed"
    assert hook_calls == ["called"]
    assert tool_calls == []


def test_before_hook_arguments_are_revalidated_before_tool_execution(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []

    def run_hook(*, binding, inputs, parent_context, parent_store):
        del binding, inputs, parent_context, parent_store
        return {
            "decision": "continue",
            "arguments": {"value": 7},
            "require_approval": False,
        }

    client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=client,
            guard=guard,
            tool_calls=tool_calls,
            document=_document_with_before_hook(),
            hook_workflow_runner=run_hook,
            tool_requires_confirmation=False,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        result = runner.run()

    assert result.status == "outcome_unknown"
    assert tool_calls == []


def test_before_hook_validates_each_modified_argument_before_the_next_hook(tmp_path) -> None:
    """An invalid first modification must not reach a later Hook or the tool."""
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    observed = []
    document = _document_with_before_hook()
    pool = next(node for node in document["nodes"] if node["type"] == "hooks_pool")
    first = pool["config"]["hooks"][0]
    first["id"] = "first"
    first["order"] = 10
    second = deepcopy(first)
    second["id"] = "second"
    second["order"] = 20
    pool["config"]["hooks"] = [second, first]

    def run_hook(*, binding, inputs, parent_context, parent_store):
        del parent_context, parent_store
        observed.append((binding.hook["id"], deepcopy(inputs["arguments"])))
        if binding.hook["id"] == "first":
            return {
                "decision": "continue",
                "arguments": {"value": 7},
                "require_approval": False,
            }
        return {
            "decision": "continue",
            "arguments": {"value": "repaired too late"},
            "require_approval": False,
        }

    client = _StrictOpenAIClient([_tool_response(), _final_response(), _final_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=client,
            guard=guard,
            tool_calls=tool_calls,
            document=document,
            hook_workflow_runner=run_hook,
            tool_requires_confirmation=False,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        result = runner.run()

    assert result.status == "completed"
    assert observed == [("first", {"value": "approved value"})]
    assert tool_calls == []


def test_before_hook_propagates_execution_lease_loss_without_continuing_the_model(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []

    def run_hook(*, binding, inputs, parent_context, parent_store):
        del binding, inputs, parent_context, parent_store
        raise ExecutionLeaseError("Hook parent no longer owns execution")

    client = _StrictOpenAIClient([_tool_response(), _final_response(), _final_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=client,
            guard=guard,
            tool_calls=tool_calls,
            document=_document_with_before_hook(),
            hook_workflow_runner=run_hook,
            tool_requires_confirmation=False,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        result = runner.run()

    assert result.status == "outcome_unknown"
    assert tool_calls == []
    assert len(client.requests) == 1


def test_kernel_approval_cold_resume_uses_durable_interaction_without_resending_initial_turn(
    tmp_path,
) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        waiting = first_runner.run()

    assert waiting.status == "waiting"
    assert tool_calls == []
    assert len(first_client.requests) == 1
    active = first_store.load().state["active"]
    wait_id = active["wait"]["id"]
    node_invocation_id = active["node_invocation_id"]

    cold_store = JsonFileSessionStore(tmp_path / "sessions")
    cold_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold_store).scope("interaction-session") as guard:
        resumed_store, resumed_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=cold_store),
            client=cold_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        resumed_runner = DurableWorkflowRunner(
            resumed_store,
            {"agent": resumed_handler},
        )
        resumed = resumed_runner.resume_kernel_interaction(
            wait_id,
            True,
            node_invocation_id=node_invocation_id,
        )
        with pytest.raises(SkeletonValidationError, match="Kernel wait is not current"):
            resumed_runner.resume_kernel_interaction(
                wait_id,
                True,
                node_invocation_id=node_invocation_id,
            )
        completed = resumed_runner.run()

    assert resumed.status == "result_received"
    assert completed.status == "completed"
    assert completed.outputs == {"answer": "approved result"}
    assert tool_calls == ["approved value"]
    assert len(first_client.requests) == 1
    assert len(cold_client.requests) == 1
    assert cold_client.requests[0]["model"] == "strict-test-model"
    assert cold_client.requests[0]["text"]["format"]["strict"] is True


@pytest.mark.parametrize("crash_before_graph_wait", [False, True])
def test_kernel_empty_prompt_approval_resume_matches_producer_manifest(
    tmp_path, crash_before_graph_wait,
) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
            prompt="",
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        if crash_before_graph_wait:
            original_save = first_store.save

            def lose_graph_wait(state, **kwargs):
                if state["status"] == "waiting":
                    raise _SimulatedProcessDeath()
                return original_save(state, **kwargs)

            first_store.save = lose_graph_wait
            with pytest.raises(_SimulatedProcessDeath):
                first_runner.run()
            assert first_runner.inspect().status == "running"
        else:
            assert first_runner.run().status == "waiting"
        node_invocation_id = first_store.load().state["active"]["node_invocation_id"]

    cold_session_store = JsonFileSessionStore(tmp_path / "sessions")
    memory = KernelMemoryRuntime.from_config(store=cold_session_store)
    wait_id = memory.load_execution_checkpoint(
        "interaction-session"
    )["interaction_ref"]["interaction_id"]
    final_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold_session_store).scope("interaction-session") as guard:
        cold_store, cold_handler = _stack(
            tmp_path,
            memory_runtime=memory,
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
            prompt="",
        )
        cold_runner = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
        if crash_before_graph_wait:
            assert cold_runner.recover().status == "waiting"
        assert cold_runner.resume_kernel_interaction(
            wait_id,
            True,
            node_invocation_id=node_invocation_id,
        ).status == "result_received"
        assert cold_runner.run().outputs == {"answer": "approved result"}

    assert tool_calls == ["approved value"]
    assert len(first_client.requests) == len(final_client.requests) == 1


def test_kernel_recovery_restores_approval_saved_before_graph_wait(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        original_save = first_store.save

        def lose_graph_wait(state, **kwargs):
            if state["status"] == "waiting":
                raise _SimulatedProcessDeath()
            return original_save(state, **kwargs)

        first_store.save = lose_graph_wait
        with pytest.raises(_SimulatedProcessDeath):
            first_runner.run()
        assert first_runner.inspect().status == "running"
        node_invocation_id = first_store.load().state["active"]["node_invocation_id"]

    cold_session_store = JsonFileSessionStore(tmp_path / "sessions")
    checkpoint = KernelMemoryRuntime.from_config(
        store=cold_session_store
    ).load_execution_checkpoint("interaction-session")
    wait_id = checkpoint["interaction_ref"]["interaction_id"]
    final_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold_session_store).scope("interaction-session") as guard:
        cold_store, cold_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=cold_session_store),
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        cold_runner = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
        recovered = cold_runner.recover()
        active = cold_store.load().state["active"]
        assert recovered.status == "waiting"
        assert active["wait"]["id"] == wait_id
        assert cold_runner.resume_kernel_interaction(
            wait_id,
            True,
            node_invocation_id=node_invocation_id,
        ).status == "result_received"
        assert cold_runner.run().outputs == {"answer": "approved result"}

    assert tool_calls == ["approved value"]
    assert len(first_client.requests) == len(final_client.requests) == 1


def test_kernel_completed_recovery_uses_assistant_history_across_tool_turns(
    tmp_path, monkeypatch,
) -> None:
    document = _document()
    agent = document["nodes"][1]
    agent["config"]["kernel"]["json_schema"] = None
    agent["config"]["execution"]["outputs"][0]["source"]["path"] = ["message"]
    monkeypatch.setattr(
        "tests.test_skeleton_kernel_interaction._document",
        lambda: deepcopy(document),
    )
    tool_response = _tool_response()
    tool_response.output.insert(0, {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "I will run the tool."}],
    })
    final_response = _final_response()
    final_response.output[0]["content"][0]["text"] = " \t\n"
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([tool_response])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        assert runner.run().status == "waiting"
        active = store.load().state["active"]

    resumed_sessions = JsonFileSessionStore(tmp_path / "sessions")
    final_client = _StrictOpenAIClient([final_response])
    with ExecutionRuntime(resumed_sessions).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=resumed_sessions),
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})

        def lose_handler_receipt(*_args):
            raise _SimulatedProcessDeath()

        monkeypatch.setattr(handler, "_save_completion", lose_handler_receipt)
        with pytest.raises(_SimulatedProcessDeath):
            runner.resume_kernel_interaction(
                active["wait"]["id"],
                True,
                node_invocation_id=active["node_invocation_id"],
            )
        assert runner.inspect().status == "running"

    cold_sessions = JsonFileSessionStore(tmp_path / "sessions")
    cold_client = _StrictOpenAIClient([])
    with ExecutionRuntime(cold_sessions).scope("interaction-session") as guard:
        cold_store, cold_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=cold_sessions),
            client=cold_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        cold = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
        assert cold.recover().status == "result_received"
        assert cold.run().outputs == {"answer": " \t\n"}

    assert tool_calls == ["approved value"]
    assert len(first_client.requests) == len(final_client.requests) == 1
    assert cold_client.requests == []


def test_kernel_recovery_rejects_foreign_approval_checkpoint(tmp_path, monkeypatch) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        original_save = first_store.save

        def lose_graph_wait(state, **kwargs):
            if state["status"] == "waiting":
                raise _SimulatedProcessDeath()
            return original_save(state, **kwargs)

        first_store.save = lose_graph_wait
        with pytest.raises(_SimulatedProcessDeath):
            first_runner.run()

    cold_session_store = JsonFileSessionStore(tmp_path / "sessions")
    cold_memory = KernelMemoryRuntime.from_config(store=cold_session_store)
    checkpoint = cold_memory.load_execution_checkpoint("interaction-session")

    def foreign_checkpoint(session_id):
        assert session_id == "interaction-session"
        changed = deepcopy(checkpoint)
        changed["source_run_id"] = "foreign-node-attempt"
        return changed

    monkeypatch.setattr(cold_memory, "load_execution_checkpoint", foreign_checkpoint)
    cold_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold_session_store).scope("interaction-session") as guard:
        cold_store, cold_handler = _stack(
            tmp_path,
            memory_runtime=cold_memory,
            client=cold_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        cold_runner = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
        with pytest.raises(ValueError, match="checkpoint changed node identity"):
            cold_runner.recover()
        assert cold_runner.inspect().status == "running"

    assert tool_calls == []
    assert len(first_client.requests) == 1
    assert cold_client.requests == []


def test_kernel_interaction_rejects_bad_or_generic_resumes_without_dispatching_tool(
    tmp_path,
) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        store, handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=client,
            guard=guard,
            tool_calls=tool_calls,
        )
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        waiting = runner.run()
        active = store.load().state["active"]
        wait_id = active["wait"]["id"]
        node_invocation_id = active["node_invocation_id"]
        execution_revision = store.load().revision
        session_before = session_store.load("interaction-session")

        with pytest.raises(SkeletonValidationError, match="Kernel wait is not current"):
            runner.resume_kernel_interaction(
                "foreign-wait",
                True,
                node_invocation_id=node_invocation_id,
            )
        with pytest.raises(SkeletonValidationError, match="official interaction resume"):
            runner.complete_wait(wait_id, {}, node_invocation_id=node_invocation_id)
        with pytest.raises(ValueError, match="tool approval response"):
            runner.resume_kernel_interaction(
                wait_id,
                "approve",
                node_invocation_id=node_invocation_id,
            )

        assert waiting.status == "waiting"
        assert store.load().revision == execution_revision
        assert session_store.load("interaction-session") == session_before
        assert tool_calls == []
        assert len(client.requests) == 1

        assert runner.pause().paused is True
        with pytest.raises(SkeletonValidationError, match="Kernel wait is not current"):
            runner.resume_kernel_interaction(
                wait_id,
                True,
                node_invocation_id=node_invocation_id,
            )
        assert tool_calls == []
        assert len(client.requests) == 1

        runner.resume()
        assert runner.cancel().status == "cancelled"
        with pytest.raises(SkeletonValidationError, match="Kernel wait is not current"):
            runner.resume_kernel_interaction(
                wait_id,
                True,
                node_invocation_id=node_invocation_id,
            )
        assert tool_calls == []
        assert len(client.requests) == 1


def test_kernel_second_approval_uses_a_second_cold_resume_without_replaying_first_turn(
    tmp_path,
) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response(call_id="first-call", value="first")])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        assert first_runner.run().status == "waiting"
        first_active = first_store.load().state["active"]

    first_cold_store = JsonFileSessionStore(tmp_path / "sessions")
    second_client = _StrictOpenAIClient([_tool_response(call_id="second-call", value="second")])
    with ExecutionRuntime(first_cold_store).scope("interaction-session") as guard:
        second_store, second_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=first_cold_store),
            client=second_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        second_runner = DurableWorkflowRunner(second_store, {"agent": second_handler})
        assert second_runner.resume_kernel_interaction(
            first_active["wait"]["id"],
            True,
            node_invocation_id=first_active["node_invocation_id"],
        ).status == "waiting"
        second_active = second_store.load().state["active"]

    final_store = JsonFileSessionStore(tmp_path / "sessions")
    final_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(final_store).scope("interaction-session") as guard:
        final_execution_store, final_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=final_store),
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        final_runner = DurableWorkflowRunner(final_execution_store, {"agent": final_handler})
        assert final_runner.resume_kernel_interaction(
            second_active["wait"]["id"],
            True,
            node_invocation_id=second_active["node_invocation_id"],
        ).status == "result_received"
        assert final_runner.run().outputs == {"answer": "approved result"}

    assert tool_calls == ["first", "second"]
    assert [len(client.requests) for client in (first_client, second_client, final_client)] == [1, 1, 1]


def test_kernel_recovery_restores_second_approval_saved_before_graph_wait(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response(call_id="first-call", value="first")])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        assert first_runner.run().status == "waiting"
        first_active = first_store.load().state["active"]

    second_session_store = JsonFileSessionStore(tmp_path / "sessions")
    second_client = _StrictOpenAIClient([_tool_response(call_id="second-call", value="second")])
    with ExecutionRuntime(second_session_store).scope("interaction-session") as guard:
        second_store, second_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=second_session_store),
            client=second_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        second_runner = DurableWorkflowRunner(second_store, {"agent": second_handler})
        original_save = second_store.save

        def lose_second_graph_wait(state, **kwargs):
            if state["status"] == "waiting":
                raise _SimulatedProcessDeath()
            return original_save(state, **kwargs)

        second_store.save = lose_second_graph_wait
        with pytest.raises(_SimulatedProcessDeath):
            second_runner.resume_kernel_interaction(
                first_active["wait"]["id"],
                True,
                node_invocation_id=first_active["node_invocation_id"],
            )
        assert second_runner.inspect().status == "running"

    final_session_store = JsonFileSessionStore(tmp_path / "sessions")
    checkpoint = KernelMemoryRuntime.from_config(
        store=final_session_store
    ).load_execution_checkpoint("interaction-session")
    wait_id = checkpoint["interaction_ref"]["interaction_id"]
    final_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(final_session_store).scope("interaction-session") as guard:
        final_store, final_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=final_session_store),
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        final_runner = DurableWorkflowRunner(final_store, {"agent": final_handler})
        assert final_runner.recover().status == "waiting"
        assert final_runner.resume_kernel_interaction(
            wait_id,
            True,
            node_invocation_id=first_active["node_invocation_id"],
        ).status == "result_received"
        assert final_runner.run().outputs == {"answer": "approved result"}

    assert tool_calls == ["first", "second"]
    assert [len(client.requests) for client in (first_client, second_client, final_client)] == [1, 1, 1]


def test_kernel_rejected_approval_never_executes_the_tool(tmp_path) -> None:
    session_store = JsonFileSessionStore(tmp_path / "sessions")
    tool_calls = []
    first_client = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(session_store).scope("interaction-session") as guard:
        first_store, first_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
            client=first_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        first_runner = DurableWorkflowRunner(first_store, {"agent": first_handler})
        first_runner.create({})
        assert first_runner.run().status == "waiting"
        active = first_store.load().state["active"]

    cold_store = JsonFileSessionStore(tmp_path / "sessions")
    final_client = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold_store).scope("interaction-session") as guard:
        resumed_store, resumed_handler = _stack(
            tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=cold_store),
            client=final_client,
            guard=guard,
            tool_calls=tool_calls,
        )
        runner = DurableWorkflowRunner(resumed_store, {"agent": resumed_handler})
        assert runner.resume_kernel_interaction(
            active["wait"]["id"],
            False,
            node_invocation_id=active["node_invocation_id"],
        ).status == "result_received"
        assert runner.run().outputs == {"answer": "approved result"}

    assert tool_calls == []
    assert len(first_client.requests) == len(final_client.requests) == 1
