"""Provider-consumer checks for the new Claude catalog entries (#354)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from unchain.providers import AnthropicModelIO
from unchain.runtime import build_runtime_loop
from unchain.runtime.payloads import load_model_capabilities
from unchain.tools import Toolkit
from tests.test_provider_replay import _anthropic_factory, _anthropic_thinking_tool_events


@pytest.mark.parametrize("model,effort", [
    ("claude-opus-5-5", "medium"), ("claude-sonnet-5", "high"),
])
def test_new_claude_requests_exclude_rejected_controls_and_replay_signed_tools(model, effort):
    caps = load_model_capabilities()[model]
    assert caps["max_context_window_tokens"] == 1000000
    assert caps["max_output_tokens"] == 128000
    assert caps["default_reasoning_effort"] == effort
    requests, calls = [], []
    events = _anthropic_thinking_tool_events(signature="model-bound-signature")
    done = [SimpleNamespace(type="content_block_delta", index=0,
                            delta=SimpleNamespace(type="text_delta", text="done")),
            SimpleNamespace(type="content_block_stop", index=0)]
    factory = _anthropic_factory([events, done], requests)

    def strict_factory(**kwargs):
        client = factory(**kwargs)
        stream = client.messages.stream

        def checked_stream(**wire):
            assert set(wire) == {"model", "messages", "max_tokens", "output_config", "tools", "system"}
            assert wire["model"] == model
            assert wire["max_tokens"] == 128000
            assert wire["output_config"] == {"effort": effort}
            assert "thinking" not in wire
            return stream(**wire)

        client.messages.stream = checked_stream
        return client

    toolkit = Toolkit()

    def demo_tool(x: int):
        calls.append(x)
        return {"value": x + 1}

    toolkit.register(demo_tool, name="demo_tool")
    io = AnthropicModelIO(model=model, api_key="fixture-key", client_factory=strict_factory)
    result = build_runtime_loop(model_io=io).run(
        [{"role": "user", "content": "Use the tool"}], provider="anthropic", model=model,
        toolkit=toolkit, max_iterations=2,
        payload={"thinking": {"type": "disabled"}, "temperature": 0.1,
                 "top_p": 0.5, "top_k": 3, "tool_choice": {"type": "any"}},
    )
    assert result.status == "completed"
    assert calls == [2]
    thinking = [block for message in requests[1]["messages"]
                if isinstance(message.get("content"), list)
                for block in message["content"] if block.get("type") == "thinking"]
    assert thinking == [{"type": "thinking", "thinking": "plan",
                         "signature": "model-bound-signature"}]
