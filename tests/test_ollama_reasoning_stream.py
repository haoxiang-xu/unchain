from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from unchain.events import RuntimeEventBridge
from unchain.kernel import ModelTurnRequest
from unchain.providers.ollama import OllamaModelIO
from unchain.tools import Toolkit


class _Response:
    status_code = 200

    def __init__(self, lines):
        self.lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield from self.lines

    def read(self):
        return b""


def _io(lines):
    return OllamaModelIO(
        model="qwen3",
        stream_factory=lambda *args, **kwargs: _Response(lines),
        default_payloads={},
        model_capabilities={},
    )


def _request(*, events, emit_stream=True, toolkit=None):
    return ModelTurnRequest(
        messages=[{"role": "user", "content": "hello"}],
        callback=events.append,
        emit_stream=emit_stream,
        run_id="run-274",
        iteration=3,
        toolkit=toolkit or Toolkit(),
    )


def _line(*, thinking=None, content=None, tool_calls=None, done=False, **extra):
    message = {}
    if thinking is not None:
        message["thinking"] = thinking
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps({"message": message, "done": done, **extra})


def test_reasoning_deltas_are_emitted_before_same_chunk_text_and_split_chunks():
    events = []
    class _ObservedResponse(_Response):
        def iter_lines(self):
            yield self.lines[0]
            assert [event["type"] for event in events] == ["request_messages", "reasoning"]
            yield self.lines[1]
            assert [event["type"] for event in events] == [
                "request_messages", "reasoning", "reasoning", "token_delta",
            ]
            yield from self.lines[2:]

    io = OllamaModelIO(
        model="qwen3",
        stream_factory=lambda *args, **kwargs: _ObservedResponse([
            _line(thinking="plan "),
            _line(thinking="carefully", content="answer "),
            _line(content="now"),
            _line(done=True),
        ]),
        default_payloads={},
        model_capabilities={},
    )
    result = io.fetch_turn(_request(events=events))

    assert result.final_text == "answer now"
    assert [event["type"] for event in events] == [
        "request_messages",
        "reasoning",
        "reasoning",
        "token_delta",
        "token_delta",
    ]
    assert [event["delta"] for event in events if event["type"] == "reasoning"] == [
        "plan ",
        "carefully",
    ]
    assert [event["delta"] for event in events if event["type"] == "token_delta"] == [
        "answer ",
        "now",
    ]
    assert events[1] == {
        "type": "reasoning",
        "run_id": "run-274",
        "iteration": 3,
        "provider": "ollama",
        "delta": "plan ",
    }


@pytest.mark.parametrize("message", [{}, {"thinking": ""}, {"thinking": None}])
def test_empty_or_missing_thinking_emits_no_reasoning(message):
    events = []
    _io([json.dumps({"message": {**message, "content": "ok"}, "done": True})]).fetch_turn(
        _request(events=events)
    )
    assert [event for event in events if event["type"] == "reasoning"] == []


def test_emit_stream_false_suppresses_reasoning_and_text_callbacks_but_replays_thinking():
    events = []
    result = _io([_line(thinking="private", content="visible", done=True)]).fetch_turn(
        _request(events=events, emit_stream=False)
    )
    assert [event["type"] for event in events] == ["request_messages"]
    assert result.final_text == "visible"
    assert result.reasoning_items == [{"type": "thinking", "text": "private"}]


def test_callback_none_and_tool_call_early_return_preserve_replay_and_usage():
    toolkit = Toolkit()
    toolkit.register(lambda value=None: value, name="demo")
    events = []
    result = _io(
        [
            _line(thinking="use tool", content="", tool_calls=[{
                "id": "call-1",
                "function": {"name": "demo", "arguments": {"value": 1}},
            }], prompt_eval_count=4, eval_count=5)
        ]
    ).fetch_turn(_request(events=events, emit_stream=True, toolkit=toolkit))
    assert [event["type"] for event in events] == ["request_messages", "reasoning"]
    assert events[1]["delta"] == "use tool"
    assert result.reasoning_items == [{"type": "thinking", "text": "use tool"}]
    assert result.consumed_tokens == 9
    assert result.tool_calls[0].name == "demo"
    assert result.tool_calls[0].arguments == {"value": 1}
    assert result.provider_replay_frame["items"][-1]["thinking"] == "use tool"
    assert result.provider_replay_frame["items"][-1]["tool_calls"][0]["id"] == "call-1"

    no_callback = _io([_line(thinking="silent", content="ok", done=True)]).fetch_turn(
        ModelTurnRequest(messages=[{"role": "user", "content": "hello"}], emit_stream=True)
    )
    assert no_callback.final_text == "ok"


def test_runtime_event_bridge_accepts_strict_reasoning_shape():
    raw_events = []
    _io([_line(thinking="think", content="answer", done=True)]).fetch_turn(
        _request(events=raw_events)
    )
    raw_reasoning = next(event for event in raw_events if event["type"] == "reasoning")
    assert set(raw_reasoning) == {"type", "run_id", "iteration", "provider", "delta"}
    bridge = RuntimeEventBridge(session_id="session-274", root_run_id="run-274")
    event = bridge.normalize(raw_reasoning)[0].to_dict()
    assert event["type"] == "step.delta"
    assert set(event["payload"]) == {"step_id", "step_type", "kind", "delta"}
    assert event["payload"] == {
        "step_id": "model:run-274:turn-3:response",
        "step_type": "model_response",
        "kind": "reasoning",
        "delta": "think",
    }


def test_exact_route_buffers_reasoning_until_release_and_preserves_wire_body():
    spec = importlib.util.spec_from_file_location(
        "exact_provider_helpers", Path(__file__).with_name("test_exact_provider_route_transport.py")
    )
    helpers = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helpers)
    from unchain.providers.exact_route_transport import OllamaExactRouteTransport

    request = {
        "model": "qwen3",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    catalog, envelope = helpers._envelope(provider="ollama", model="qwen3", request=request)
    sent = []

    def stream_factory(method, url, **kwargs):
        sent.append((method, url, kwargs))
        return _Response([_line(thinking="plan", content="ok", done=True)])

    external = []
    transport = OllamaExactRouteTransport(
        model_io=OllamaModelIO(model="qwen3", stream_factory=stream_factory, default_payloads={}, model_capabilities={}),
        catalog=catalog,
        callback=external.append,
        run_id="run-274",
        emit_stream=True,
    )
    result = transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=0)
    assert result.final_text == "ok"
    assert external == []
    assert [event["type"] for event in transport.buffered_events] == [
        "request_messages", "reasoning", "token_delta",
    ]
    transport.release_buffered_events()
    assert [event["type"] for event in external] == [
        "request_messages", "reasoning", "token_delta",
    ]
    assert sent[0][2]["json"] == request
