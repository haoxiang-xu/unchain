import json

import httpx
import pytest

from unchain.kernel import ModelTurnResult
from unchain.kernel.types import ToolCall
from unchain.runtime import build_runtime_loop
from unchain.toolkits import CoreToolkit
from unchain.toolkits.builtin.core import web_fetch as web
from unchain.tools.messages import get_provider_message_builder
from unchain.tools.web_fetch_guard import web_fetch_failure_limit


def _call_message(provider, call):
    if provider == "openai":
        return {"type": "function_call", "call_id": call.call_id, "name": call.name, "arguments": json.dumps(call.arguments)}
    if provider == "anthropic":
        return {"role": "assistant", "content": [{"type": "tool_use", "id": call.call_id, "name": call.name, "input": call.arguments}]}
    if provider == "gemini":
        return {"role": "model", "parts": [{"function_call": {"id": call.call_id, "name": call.name, "args": call.arguments}}]}
    return {"role": "assistant", "tool_calls": [{"id": call.call_id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}]}


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama"])
@pytest.mark.parametrize("different_urls,batch_size,limit", [(False, 1, 3), (True, 2, 6)])
def test_all_403_research_finishes_with_bounded_fetches_and_approvals(monkeypatch, tmp_path, provider, different_urls, batch_size, limit):
    attempts = []
    approvals = []
    client = httpx.Client
    def respond(request):
        attempts.append(str(request.url))
        return httpx.Response(403, text="Access denied by origin", request=request)
    monkeypatch.setattr(web.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(respond), **kw))
    monkeypatch.setattr(web, "validate_public_url", lambda url: (url, None))
    class RepeatingModel:
        calls = 0
        def fetch_turn(self, request):
            self.calls += 1
            calls = [ToolCall(
                call_id=f"fetch-{self.calls}-{index}", name="web_fetch",
                arguments={"url": f"https://example.com/{self.calls}-{index}" if different_urls else "https://example.com/blocked"},
            ) for index in range(batch_size)]
            return ModelTurnResult(assistant_messages=[_call_message(provider, call) for call in calls], tool_calls=calls)
    model = RepeatingModel()
    toolkit = CoreToolkit(workspace_roots=[str(tmp_path)])
    result = build_runtime_loop(model_io=model).run(
        [{"role": "user", "content": "Research these web sources"}],
        provider=provider, model="offline-test-double", toolkit=toolkit,
        max_iterations=10,
        on_tool_confirm=lambda request: approvals.append(request) or {"approved": True},
    )
    assert result.status == "completed"
    assert len(attempts) == len(approvals) == limit
    assert model.calls == limit // batch_size
    assert result.messages[-1]["role"] == "assistant"
    assert "HTTP 403" in result.messages[-1]["content"]
    assert "cannot verify" in result.messages[-1]["content"]


def _exchange(provider, index, *, ok=False, name="web_fetch"):
    call = ToolCall(call_id=f"fetch-{index}", name=name, arguments={"url": "https://example.com/page"})
    return [_call_message(provider, call), *get_provider_message_builder(provider).build_tool_result_messages(
        tool_call=call,
        tool_result={"ok": ok, "url": "https://example.com/page", "status_code": 200 if ok else 403, "result": "body"},
    )]


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "ollama"])
def test_budget_survives_serialization_and_resets_for_success_or_new_user(provider):
    messages = [{"role": "user", "content": "start"}]
    messages += _exchange(provider, 1) + _exchange(provider, 2)
    assert web_fetch_failure_limit(messages) is None
    # Cold replay uses only existing native transcript bytes, not service state.
    restored = json.loads(json.dumps(messages))
    assert web_fetch_failure_limit(restored + _exchange(provider, 3))
    assert web_fetch_failure_limit(restored + _exchange(provider, 3, ok=True) + _exchange(provider, 4)) is None
    assert web_fetch_failure_limit(restored + _exchange(provider, 3) + _exchange(provider, 4, ok=True)) is None
    assert web_fetch_failure_limit(restored + [{"role": "user", "content": "try again"}] + _exchange(provider, 3)) is None
    assert web_fetch_failure_limit([{"role": "user", "content": "start"}] + sum((_exchange(provider, index, name="another_tool") for index in range(4)), [])) is None


def test_cold_checkpoint_restore_retains_fetch_failure_budget(tmp_path):
    from unchain.kernel import RunState
    from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
    from unchain.memory.checkpoint_state import build_execution_checkpoint

    state = RunState()
    state.seed_messages([{"role": "user", "content": "Research sources"}] + _exchange("ollama", 1) + _exchange("ollama", 2))
    state.session_state.session_id = "fetch-recovery"
    state.provider_state.provider = "ollama"
    state.provider_state.model = "offline-test-double"
    state.iteration = 2
    checkpoint = build_execution_checkpoint(state, status="max_iterations", run_id="before-restart")
    runtime = KernelMemoryRuntime.from_config(store=JsonFileSessionStore(base_dir=tmp_path))
    runtime.save_execution_checkpoint("fetch-recovery", checkpoint)
    cold_runtime = KernelMemoryRuntime.from_config(store=JsonFileSessionStore(base_dir=tmp_path))
    restored, _, _, _ = cold_runtime.bootstrap_session(
        session_id="fetch-recovery", memory_namespace=None, incoming_messages=[],
        resume_mode=False, provider="ollama", model="offline-test-double",
    )
    assert restored == state.transcript
    assert web_fetch_failure_limit(restored) is None
    assert "HTTP 403" in web_fetch_failure_limit(restored + _exchange("ollama", 3))


def test_cold_approval_resume_stops_at_third_failure_without_another_model_turn(tmp_path):
    from tests.test_durable_tool_approval import _QueueModelIO, _tool_turn
    from unchain.interaction import INTERACTION_JOURNAL_KEY
    from unchain.interaction.runtime import DurableInteractionRuntime
    from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
    from unchain.tools import Toolkit

    store = JsonFileSessionStore(base_dir=tmp_path)
    calls = []
    toolkit = Toolkit()
    def fetch(url):
        calls.append(url)
        return {"ok": False, "url": url, "status_code": 403, "result": "Access denied"}
    toolkit.register(fetch, name="web_fetch", requires_confirmation=True)
    third = ToolCall(call_id="fetch-3", name="web_fetch", arguments={"url": "https://example.com/page"})
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO([_tool_turn(third)]),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    suspended = first_loop.run(
        [{"role": "user", "content": "Research sources"}] + _exchange("openai", 1) + _exchange("openai", 2),
        session_id="approval-fetch", provider="openai", model="gpt-5",
        toolkit=toolkit, max_iterations=6,
    )
    assert suspended.status == "awaiting_interaction"
    assert calls == []
    cold_store = JsonFileSessionStore(base_dir=tmp_path)
    cold_memory = KernelMemoryRuntime.from_config(store=cold_store)
    interaction = DurableInteractionRuntime(cold_memory)
    pending = interaction.load_active("approval-fetch")
    interaction.record_receipt(
        "approval-fetch", interaction_id=pending.request.interaction_id,
        response={"approved": True}, submitted_by="ui:test",
        expected_revision=pending.session_snapshot.revision,
    )
    model = _QueueModelIO([])  # Any request after this approval is a regression.
    resumed = build_runtime_loop(model_io=model, memory_runtime=cold_memory).resume_interaction(
        session_id="approval-fetch", toolkit=toolkit,
    )
    assert resumed.status == "completed"
    assert calls == ["https://example.com/page"]
    assert model.requests == []
    assert "HTTP 403" in resumed.messages[-1]["content"]
    assert cold_store.load("approval-fetch")[INTERACTION_JOURNAL_KEY]["active_id"] is None
