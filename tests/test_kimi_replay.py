from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from unchain.agent import Agent, MemoryModule, ToolsModule
from unchain.kernel import ModelTurnRequest
from unchain.kernel.provider_replay import ProviderReplayFrameError, set_provider_replay_frame
from unchain.kernel.state import RunState
from unchain.memory import JsonFileSessionStore, MemoryManager
from unchain.providers import AnthropicModelIO, HyperspaceModelIO
from unchain.providers.model_turn_runtime import build_model_turn_request
from unchain.runtime import build_runtime_loop
from unchain.tools import Toolkit

from tests.test_provider_replay import _anthropic_factory, _anthropic_thinking_tool_events


MODEL = "kimi-k2.7-code"
ENDPOINT = "https://api.moonshot.ai/anthropic"


def _io(events, requests, *, endpoint=ENDPOINT, model=MODEL):
    return HyperspaceModelIO(
        model=model, api_key="fixture-key", base_url=endpoint,
        client_factory=_anthropic_factory(events, requests),
    )


def _tool_events(call_id="toolu_1"):
    events = _anthropic_thinking_tool_events(signature="")
    for event in events:
        block = getattr(event, "content_block", None)
        if getattr(block, "type", None) == "tool_use":
            block.id = call_id
    return events


def _done_events():
    return [
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="text_delta", text="done")),
        SimpleNamespace(type="content_block_stop", index=0),
    ]


def _assert_wire_thinking(messages, expected_count):
    blocks = [block for message in messages
              if message.get("role") == "assistant" and isinstance(message.get("content"), list)
              for block in message["content"] if block.get("type") == "thinking"]
    assert blocks == [{"type": "thinking", "thinking": "plan"}] * expected_count


@pytest.mark.parametrize("endpoint", [ENDPOINT, "https://api.moonshot.cn/anthropic/"])
def test_kimi_unsigned_thinking_tool_capture(endpoint):
    turn = _io([_tool_events()], [], endpoint=endpoint).fetch_turn(
        ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}])
    )
    assert len(turn.tool_calls) == 1
    assert turn.reasoning_items == [{"type": "thinking", "text": "plan"}]
    _assert_wire_thinking(turn.provider_replay_frame["items"], 1)
    assert turn.provider_replay_frame["replay_profile"] == {
        "profile": "kimi.unsigned-thinking.v1", "endpoint": endpoint.rstrip("/"), "model": MODEL,
    }


@pytest.mark.parametrize("endpoint,model", [
    ("https://api.anthropic.com", MODEL),
    ("https://api.moonshot.ai.evil.example/anthropic", MODEL),
    ("http://api.moonshot.ai/anthropic", MODEL),
    (ENDPOINT + "?route=other", MODEL),
    ("http://localhost:6655/anthropic", MODEL),
    (ENDPOINT, "kimi-k3"), (ENDPOINT, "kimi-k2.6"),
])
def test_other_routes_keep_signature_requirement(endpoint, model):
    with pytest.raises(ProviderReplayFrameError, match="signature"):
        _io([_tool_events()], [], endpoint=endpoint, model=model).fetch_turn(
            ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}])
        )


def test_kimi_repeated_tools_and_second_user_turn():
    requests, calls = [], []
    toolkit = Toolkit()

    def demo_tool(x: int):
        calls.append(x)
        return {"value": x + 1}

    toolkit.register(demo_tool, name="demo_tool")
    model_io = _io([_tool_events(), _tool_events("toolu_2"), _done_events()], requests)
    result = build_runtime_loop(model_io=model_io).run(
        [{"role": "user", "content": "two tools"}], provider="hyperspace",
        model=MODEL, toolkit=toolkit, max_iterations=3,
    )
    assert result.status == "completed"
    assert calls == [2, 2]
    _assert_wire_thinking(requests[1]["messages"], 1)
    _assert_wire_thinking(requests[2]["messages"], 2)
    assert "thinking" not in json.dumps(result.messages)


def test_kimi_cold_restart_preserves_replay_and_does_not_repeat_tool(tmp_path):
    calls, first_requests, resumed_requests = [], [], []

    def demo_tool(x: int):
        calls.append(x)
        return {"value": x + 1}

    def make_agent(events, requests):
        return Agent(
            name="kimi-cold", provider="hyperspace", model=MODEL,
            modules=(ToolsModule(tools=(demo_tool,)),
                     MemoryModule(memory=MemoryManager(store=JsonFileSessionStore(tmp_path)))),
            model_io_factory=lambda spec, context: _io(events, requests),
        )

    stopped = make_agent([_tool_events()], first_requests).run(
        "use tool", session_id="kimi-cold", max_iterations=1,
    )
    assert stopped.status == "max_iterations"
    checkpoint = JsonFileSessionStore(tmp_path).load("kimi-cold")["execution_checkpoint"]
    assert checkpoint["replay_frame"]["replay_profile"]["model"] == MODEL
    completed = make_agent([_done_events(), _done_events()], resumed_requests)
    assert completed.run("continue", session_id="kimi-cold", max_iterations=1).status == "completed"
    _assert_wire_thinking(resumed_requests[0]["messages"], 1)
    assert calls == [2]
    assert completed.run("second message", session_id="kimi-cold", max_iterations=1).status == "completed"
    assert calls == [2]
    # Completed legacy memory stores semantic history. Verify the next turn's
    # tool pair remains intact; live Kimi acceptance of that history is separate.
    later_blocks = [block for message in resumed_requests[1]["messages"]
                    if isinstance(message.get("content"), list) for block in message["content"]]
    assert [b["id"] for b in later_blocks if b.get("type") == "tool_use"] == ["toolu_1"]
    assert [b["tool_use_id"] for b in later_blocks if b.get("type") == "tool_result"] == ["toolu_1"]


@pytest.mark.parametrize("mutation", ["profile", "endpoint", "model", "extra", "native", "no-live-io", "malformed-endpoint", "active-endpoint", "active-model"])
def test_profile_mismatch_rejected_before_provider_request(mutation):
    model_io = _io([_tool_events()], [])
    turn = model_io.fetch_turn(ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}]))
    frame = copy.deepcopy(turn.provider_replay_frame)
    state = RunState()
    state.provider_state.provider = "hyperspace"
    state.provider_state.model = MODEL
    state.seed_messages([{"role": "user", "content": "use tool"}, *turn.assistant_messages])
    if mutation in {"profile", "endpoint", "model"}:
        frame["replay_profile"][mutation] = "wrong"
    elif mutation == "extra":
        frame["replay_profile"]["allow_unsigned"] = True
    elif mutation == "malformed-endpoint":
        frame["replay_profile"]["endpoint"] = []
    elif mutation == "active-endpoint":
        model_io = _io([], [], endpoint="https://api.moonshot.cn/anthropic")
    elif mutation == "active-model":
        state.provider_state.model = "kimi-k3"
        model_io = _io([], [], model="kimi-k3")
    elif mutation == "native":
        state.provider_state.provider = "anthropic"
        model_io = AnthropicModelIO(model=MODEL, api_key="fixture-key", client_factory=lambda **kwargs: None)
    else:
        model_io = None
    set_provider_replay_frame(state, frame)
    with pytest.raises(ProviderReplayFrameError, match="profile"):
        build_model_turn_request(state, model_io=model_io)


def test_kimi_actual_wire_model_must_match_profile():
    requests = []
    model_io = _io([_tool_events()], requests)
    model_io.default_payloads = {MODEL: {"model": "claude-other"}}
    with pytest.raises(ProviderReplayFrameError, match="wire model"):
        model_io.fetch_turn(ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}]))
    assert requests == []


@pytest.mark.parametrize("value", [123, False, [], {"x": 1}, None])
@pytest.mark.parametrize("field", ["signature", "thinking"])
def test_kimi_rejects_malformed_stream_delta_before_coercion(field, value):
    events = _tool_events()
    delta = events[2 if field == "signature" else 1].delta
    setattr(delta, field, value)
    with pytest.raises(ProviderReplayFrameError, match=field):
        _io([events], []).fetch_turn(
            ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}])
        )


@pytest.mark.parametrize("signature", [None, "", "signed"])
def test_kimi_sdk_final_message_thinking_shape(signature):
    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return SimpleNamespace(id="kimi-final", content=[
                {"type": "thinking", "thinking": "plan", "signature": signature},
                {"type": "tool_use", "id": "toolu_1", "name": "demo_tool", "input": {"x": 2}},
            ])

    io = HyperspaceModelIO(model=MODEL, api_key="fixture-key", base_url=ENDPOINT,
        client_factory=lambda **kwargs: SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream())))
    turn = io.fetch_turn(ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}]))
    expected = {"type": "thinking", "thinking": "plan"}
    if signature:
        expected["signature"] = signature
    assert turn.provider_replay_frame["items"][-1]["content"][0] == expected


@pytest.mark.parametrize("metadata", [{}, {"caller": None},
    {"caller": {"type": "direct"}}, {"caller": ""}, {"future_metadata": None}])
@pytest.mark.parametrize("kimi", [True, False])
def test_sdk_tool_metadata_through_canonical_compiler(metadata, kimi):
    from unchain.context.compiler import _native_tool_call_messages
    from unchain.providers.context_assembler import (
        ProviderContextProjectionError, _rehydrate, _segments_for,
    )

    tool = {"type": "tool_use", "id": "toolu_1", "name": "demo_tool", "input": {"x": 2}, **metadata}
    thinking = {"type": "thinking", "thinking": "plan", "signature": "signed"}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return SimpleNamespace(id="sdk-final", content=[thinking, tool])

    cls = HyperspaceModelIO if kimi else AnthropicModelIO
    kwargs = {"base_url": ENDPOINT} if kimi else {}
    io = cls(model=MODEL if kimi else "claude-sonnet-4-5", api_key="fixture-key",
        client_factory=lambda **kw: SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream())),
        **kwargs)
    turn = io.fetch_turn(ModelTurnRequest(messages=[{"role": "user", "content": "use tool"}]))
    expected_tool = copy.deepcopy(tool)
    if kimi and expected_tool.get("caller") is None:
        expected_tool.pop("caller", None)
    frame = turn.provider_replay_frame
    assert frame["items"][-1]["content"] == [thinking, expected_tool]
    assert turn.assistant_messages[-1]["content"] == [expected_tool]
    provider = "hyperspace" if kimi else "anthropic"
    canonical = _native_tool_call_messages(provider, [({}, call) for call in turn.tool_calls])
    result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "3"}]}
    canonical.append(result)
    segments = _segments_for(frame["format"], frame["items"], allow_unsigned_thinking=kimi)
    if set(expected_tool) - {"type", "id", "name", "input"}:
        with pytest.raises(ProviderContextProjectionError, match="mutated ambiguously"):
            _rehydrate(provider, canonical, segments)
    else:
        assert _rehydrate(provider, canonical, segments) == [
            {"role": "assistant", "content": [thinking, expected_tool]}, result,
        ]


def test_kimi_official_context_boundary_cold_approval_resume(tmp_path):
    from tests.context_v2.test_context_provider_turn_approval_cross_provider import _anthropic_approval_events
    from tests.context_v2.test_context_provider_turn_approval_resume import _approval_toolkit
    from tests.context_v2.test_context_provider_turn_cross_provider import _runtime
    from unchain.memory import InMemorySessionStore, KernelMemoryRuntime

    calls, requests = [], []
    session_store = InMemorySessionStore()
    thinking = _tool_events()[:4]
    tool_events = _anthropic_approval_events(send_number=1, call_id="approved-kimi")
    for event in tool_events:
        if hasattr(event, "index"):
            event.index += 1
    first_io = _io([thinking + tool_events], requests)
    first_io.fetch_turn = lambda request: (_ for _ in ()).throw(AssertionError("legacy path"))
    runtime = _runtime(tmp_path)
    loop = build_runtime_loop(
        harnesses=list(runtime.build_harnesses()), model_io=first_io,
        memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
        semantic_context_owner=runtime.owner_id,
    )
    suspended = loop.run(
        messages=[{"role": "user", "content": "write after approval"}],
        callback=runtime.compose_event_callback(None), session_id="kimi-approved",
        provider="hyperspace", model=MODEL, toolkit=_approval_toolkit(calls),
        run_id="kimi-approved-attempt", max_iterations=2,
    )
    assert suspended.status == "awaiting_interaction"
    assert calls == []
    reopened = _runtime(tmp_path)
    second_io = _io([_done_events()], requests)
    second_io.fetch_turn = lambda request: (_ for _ in ()).throw(AssertionError("legacy path"))
    resumed_loop = build_runtime_loop(
        harnesses=list(reopened.build_harnesses()), model_io=second_io,
        memory_runtime=KernelMemoryRuntime.from_config(store=session_store),
        semantic_context_owner=reopened.owner_id,
    )
    result = resumed_loop.resume_interaction(
        session_id="kimi-approved", response={"approved": True},
        callback=reopened.compose_event_callback(None), toolkit=_approval_toolkit(calls),
    )
    assert result.status == "completed"
    assert calls == ["durable"]
    assert len(requests) == 2
    _assert_wire_thinking(requests[1]["messages"], 1)


def test_kimi_durable_owned_provider_branch_preserves_repeated_thinking(tmp_path):
    from tests.context_v2.test_provider_turn_execution_service import _service, ATTEMPT
    from unchain.providers.durable_turn_runtime import DurableProviderTurnMode
    from unchain.providers.turn_ownership import ProviderTurnOwnership
    from unchain.run_bundle import RunIdentity

    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None, relation="root",
    )

    class Factory:
        def bind(self, *, identity):
            service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
            return ProviderTurnOwnership(identity=identity, service=service, ledger=service.store, factory=self)

    requests, calls = [], []

    def demo_tool(x: int):
        calls.append(x)
        return {"value": x + 1}

    io = _io([_tool_events(), _tool_events("toolu_2"), _done_events()], requests)
    io.fetch_turn = lambda request: (_ for _ in ()).throw(AssertionError("legacy path"))
    result = Agent(
        name="kimi-owned", provider="hyperspace", model=MODEL,
        modules=(ToolsModule(tools=(demo_tool,)),), model_io_factory=lambda spec, context: io,
    ).run(
        "use two tools", max_iterations=3, session_id=identity.execution_id,
        run_id=identity.run_id, _run_bundle_identity=identity,
        _provider_turn_ownership_factory=Factory(),
    )
    assert result.status == "completed"
    assert calls == [2, 2]
    assert len(requests) == 3
    _assert_wire_thinking(requests[1]["messages"], 1)
    _assert_wire_thinking(requests[2]["messages"], 2)
