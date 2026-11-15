"""Direct durable Human Input must remain a graph wait, not handler_error.

The tool arguments are the actual normalized App #486 run 43680507496
response (qa486-human / human-call), replayed through the real Kernel and
Journal with the strict existing offline SDK client.
"""
from copy import deepcopy
import json
from types import SimpleNamespace
from pathlib import Path
import hashlib
import os
import sys

import pytest

from unchain.execution import ExecutionRuntime
from unchain.input.human_input import build_ask_user_question_tool
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from tests.test_skeleton_kernel_interaction import (
    _stack, _StrictOpenAIClient, _final_response, _tool_response,
)

ARGS = {
    "title": "QA486 question", "question": "Choose the offline answer",
    "selection_mode": "single",
    "options": [{"label": "Use fixture", "value": "fixture"}],
    "allow_other": False,
}


def test_direct_human_input_is_retained_and_resumes_on_a_rebuilt_stack(tmp_path, capfd):
    import unchain
    import unchain.runtime.runtime_protocol as protocol
    import unchain.skeleton.kernel_runtime as kernel
    expected = Path(__file__).resolve().parents[1] / "src"
    modules = {name: {"file": str(Path(module.__file__).resolve()),
        "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
        for name, module in (("unchain", unchain),
            ("unchain.runtime.runtime_protocol", protocol),
            ("unchain.skeleton.kernel_runtime", kernel))}
    assert all(Path(row["file"]).is_relative_to(expected) for row in modules.values())
    with capfd.disabled():
        print("UNCHAIN_HUMAN_WAIT_CONSUMER " + json.dumps({"pid": os.getpid(),
            "argv": sys.argv, "modules": modules,
            "manifest_digest": protocol.runtime_protocol_manifest()["manifest_digest"]}), flush=True)
    sessions = JsonFileSessionStore(tmp_path / "sessions")
    first = _StrictOpenAIClient([SimpleNamespace(
        id="human-turn", output=[{"type": "function_call", "call_id": "human-call",
            "name": "ask_user_question", "arguments": json.dumps(ARGS)}], usage={})])
    calls = []
    with ExecutionRuntime(sessions).scope("interaction-session") as guard:
        store, handler = _stack(tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=sessions),
            client=first, guard=guard, tool_calls=calls)
        handler.tools_for_node = lambda _context: (build_ask_user_question_tool(),)
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        result = runner.run()
        assert result.status == "waiting", store.load().state["error"]
        state = store.load().state
        wait = deepcopy(state["active"]["wait"])
        invocation = state["active"]["node_invocation_id"]
        request = store.read_json(wait["request_ref"])
        assert wait["kind"] == "interaction"
        assert request["kind"] == "human_input"
        assert request["payload"]["request_id"] == "human-call"
        assert request["interaction_id"] == wait["id"]
        assert calls == []
    assert len(first.requests) == 1

    cold = JsonFileSessionStore(tmp_path / "sessions")
    final = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold).scope("interaction-session") as guard:
        restored, handler = _stack(tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=cold),
            client=final, guard=guard, tool_calls=calls)
        handler.tools_for_node = lambda _context: (build_ask_user_question_tool(),)
        runner = DurableWorkflowRunner(restored, {"agent": handler})
        assert restored.load().state["active"]["wait"] == wait
        resumed = runner.resume_kernel_interaction(wait["id"], {
            "request_id": "human-call", "selected_values": ["fixture"], "other_text": None,
        }, node_invocation_id=invocation)
        assert resumed.status == "result_received"
        assert runner.run().outputs == {"answer": "approved result"}
    assert len(final.requests) == 1
    assert "fixture" in json.dumps(final.requests)
    assert calls == []


def test_human_checkpoint_cannot_relabel_a_tool_approval(tmp_path, monkeypatch):
    sessions = JsonFileSessionStore(tmp_path / "sessions")
    calls = []
    first = _StrictOpenAIClient([_tool_response()])
    with ExecutionRuntime(sessions).scope("interaction-session") as guard:
        store, handler = _stack(tmp_path,
            memory_runtime=KernelMemoryRuntime.from_config(store=sessions),
            client=first, guard=guard, tool_calls=calls)
        runner = DurableWorkflowRunner(store, {"agent": handler})
        runner.create({})
        assert runner.run().status == "waiting"
        wait = deepcopy(store.load().state["active"]["wait"])
        invocation = store.load().state["active"]["node_invocation_id"]
    cold = JsonFileSessionStore(tmp_path / "sessions")
    memory = KernelMemoryRuntime.from_config(store=cold)
    final = _StrictOpenAIClient([_final_response()])
    with ExecutionRuntime(cold).scope("interaction-session") as guard:
        store, handler = _stack(tmp_path, memory_runtime=memory,
            client=final, guard=guard, tool_calls=calls)
        runner = DurableWorkflowRunner(store, {"agent": handler})
        original = memory.load_execution_checkpoint
        def relabel(*args, **kwargs):
            checkpoint = deepcopy(original(*args, **kwargs))
            checkpoint["status"] = "awaiting_human_input"
            return checkpoint
        monkeypatch.setattr(memory, "load_execution_checkpoint", relabel)
        with pytest.raises(ValueError, match="Human input checkpoint changed interaction kind"):
            runner.resume_kernel_interaction(wait["id"], True, node_invocation_id=invocation)
        assert store.load().state["status"] == "waiting"
        assert calls == [] and final.requests == []
        monkeypatch.setattr(memory, "load_execution_checkpoint", original)
        assert runner.resume_kernel_interaction(wait["id"], True,
            node_invocation_id=invocation).status == "result_received"
        assert runner.run().outputs == {"answer": "approved result"}
    assert calls == ["approved value"]
