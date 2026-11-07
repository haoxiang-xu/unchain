from __future__ import annotations

import json
from pathlib import Path

from unchain.kernel.types import KernelRunResult
from unchain.execution import (
    ExecutionCancelledError,
    ExecutionLeaseConfig,
    ExecutionRuntime,
)
from unchain.memory import InMemorySessionStore
from unchain.runtime.runtime_protocol import runtime_protocol_manifest
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.subagent_runtime import SkeletonSubagentAgent
from unchain.subagents.executor import SubagentExecutor
from unchain.subagents.plugin import SubagentToolPlugin
from unchain.subagents.plugin import _ChildRunError
from unchain.subagents.types import SubagentPolicy
from unchain.agent import Agent


def _agent_document():
    document = json.loads(
        (Path(__file__).parent / "fixtures/skeleton_execution/if_rejoin_workflow.json").read_text()
    )
    document["kind"] = "agent"
    for node in document["nodes"]:
        if node["type"] == "code":
            node["type"] = "agent"
            node["config"]["kernel"] = {
                "schema_version": 1,
                "profile": {
                    "id": "openai.test", "provider": "openai", "model": "test-model",
                    "protocol": "openai-responses", "endpoint": "https://example.test/v1",
                    "credential_ref": "openai.key", "context_window": 8192,
                    "output_limit": 2048,
                    "capabilities": {"json_schema": True, "images": True, "files": False},
                },
                "prompt": "test instruction", "max_iterations": 3, "output_tokens": 512,
                "history": "root", "overflow": "error", "json_schema": None,
            }
    return document


def _plugin():
    return SubagentToolPlugin(
        parent_agent=Agent(name="parent", provider="openai"),
        templates=(),
        policy=SubagentPolicy(),
        executor=SubagentExecutor(),
    )


def _run_native(run_child, *, callback=None, **options):
    facade = SkeletonSubagentAgent(
        name="native",
        plan=prepare_agent_graph(_agent_document()),
        run_child=run_child,
        entry_id="native",
    )
    result = _plugin()._run_child(
        agent=facade,
        mode="delegate",
        child_id="native",
        lineage=["parent", "native"],
        template_name="native",
        session_id="root-session:native",
        memory_namespace="",
        input_messages="work",
        max_iterations=1,
        callback=callback,
        **options,
    )
    return result


def _completed():
    return KernelRunResult(
        messages=[{"role": "assistant", "content": "done"}],
        status="completed",
    )


def test_exact_native_facade_receives_observation_gate_without_parent_forward():
    parent_events = []
    received = {}
    events = [
        {"type": "response_received", "run_id": "native-node"},
        {"type": "human_input_requested", "request": {"prompt": "secret"}},
    ]

    def probe(_adapter, _messages, kwargs):
        gate = kwargs["_native_observation_gate"]
        received["gate"] = gate
        received["results"] = [gate(event) for event in events]
        return _completed()

    _run_native(probe, callback=parent_events.append)

    assert callable(received["gate"])
    assert received["results"] == [True, False]
    assert parent_events == []


def test_native_observation_captures_board_delta_once_by_copy():
    returned = {}
    item = {
        "item_id": "board-one",
        "board_id": "default",
        "content": "original",
    }
    event = {
        "type": "tool_result",
        "tool_name": "write_agent_board",
        "result": {
            "mode": "agent_board_write",
            "status": "written",
            "item": item,
        },
    }

    def probe(_adapter, _messages, kwargs):
        gate = kwargs["_native_observation_gate"]
        assert gate(event) is True
        assert gate(event) is True
        item["content"] = "mutated"
        return _completed()

    result = _run_native(probe, callback=lambda _event: returned.setdefault("parent", True))

    assert result.subagent_state["blackboards"]["default"] == [{
        "item_id": "board-one",
        "board_id": "default",
        "content": "original",
    }]
    assert returned == {}


def test_native_observation_checks_batch_before_capture_and_forward():
    import threading

    failure_event = threading.Event()
    parent_events = []
    after_gate = []
    board_event = {
        "type": "tool_result",
        "tool_name": "write_agent_board",
        "result": {
            "mode": "agent_board_write",
            "status": "written",
            "item": {"item_id": "board-one", "board_id": "default"},
        },
    }

    def probe(_adapter, _messages, kwargs):
        failure_event.set()
        kwargs["_native_observation_gate"](board_event)
        after_gate.append(True)
        return _completed()

    try:
        _run_native(probe, callback=parent_events.append, batch_failure_event=failure_event)
    except _ChildRunError as error:
        assert after_gate == []
        assert "subagent_batch_durable_failure" in str(error.original)
        assert error.subagent_state == {}
    else:
        raise AssertionError("batch failure must stop native observation")
    assert after_gate == []
    assert parent_events == []


def test_native_observation_checks_cancelled_parent_before_capture_and_forward():
    runtime = ExecutionRuntime(
        InMemorySessionStore(),
        ExecutionLeaseConfig(ttl_ms=60_000, heartbeat_interval_ms=0),
    )
    root_guard = runtime.acquire("root-session", owner_id="attempt-1")
    parent_events = []
    after_gate = []
    board_event = {
        "type": "tool_result",
        "tool_name": "write_agent_board",
        "result": {
            "mode": "agent_board_write",
            "status": "written",
            "item": {"item_id": "board-one", "board_id": "default"},
        },
    }

    def probe(_adapter, _messages, kwargs):
        runtime.request_cancel("root-session", "attempt-1", reason="user_stop")
        kwargs["_native_observation_gate"](board_event)
        after_gate.append(True)
        return _completed()

    try:
        try:
            _run_native(probe, callback=parent_events.append, execution_guard=root_guard)
        except _ChildRunError as error:
            assert after_gate == []
            assert isinstance(error.original, ExecutionCancelledError)
            assert error.subagent_state == {}
        else:
            raise AssertionError("cancelled parent must stop native observation")
        assert after_gate == []
        assert parent_events == []
    finally:
        root_guard.release()


def test_native_gate_keyword_is_not_injected_into_claim_only_adapter():
    class ClaimOnlyAdapter:
        name = "claim-only"
        manages_subagent_completion = True

        def run(self, _messages, **kwargs):
            assert "_native_observation_gate" not in kwargs
            kwargs["callback"]({"type": "custom_child_progress"})
            return _completed()

    events = []
    result = _plugin()._run_child(
        agent=ClaimOnlyAdapter(),
        mode="delegate",
        child_id="claim-only",
        lineage=["parent", "claim-only"],
        template_name="claim-only",
        session_id="root-session:claim-only",
        memory_namespace="",
        input_messages="work",
        max_iterations=1,
        callback=events.append,
    )
    assert result.status == "completed"
    assert events == [{"type": "custom_child_progress"}]


def test_public_native_facade_gate_argument_compatibility():
    received = {}

    def run_child(_adapter, _messages, kwargs):
        received.update(kwargs)
        return _completed()

    facade = SkeletonSubagentAgent(
        name="native",
        plan=prepare_agent_graph(_agent_document()),
        run_child=run_child,
        entry_id="native",
    )
    facade.run("work")
    assert "_native_observation_gate" not in received

    for invalid in (None, 42):
        try:
            facade.run("work", _native_observation_gate=invalid)
        except TypeError:
            pass
        else:
            raise AssertionError("present non-callable gate must be rejected")

    gate = lambda _event: True
    facade.run("work", _native_observation_gate=gate)
    assert received["_native_observation_gate"] is gate


def test_manifest_advertises_native_observation_feature_at_minor_seven():
    protocols = runtime_protocol_manifest()["protocols"]
    skeleton = next(item for item in protocols if item["id"] == "skeleton_execution")
    # Native observation was introduced at minor 7 and remains in minor 8.
    assert skeleton["minor"] == 8
    assert "subagent_native_observation_v1" in skeleton["features"]
    assert "node_display_label_v1" in skeleton["features"]
    assert skeleton["features"] == sorted(set(skeleton["features"]))
    assert len(skeleton["features"]) == 22
