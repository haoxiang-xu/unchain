from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from threading import Event, Thread

import pytest

from unchain.events import RuntimeEventBridge
from unchain.context import ContextCompiler, ContextRuntime
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


def test_exact_route_previews_thinking_before_provider_completion_without_early_journal_write():
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
    paused = Event()
    continue_stream = Event()

    class _PausedResponse(_Response):
        def iter_lines(self):
            yield _line(thinking="plan")
            paused.set()
            assert continue_stream.wait(5), "timed out waiting for the second chunk"
            yield _line(content="ok", done=True)

    durable = []
    external = []
    runtime = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=lambda _context: None,
        durable_event_sink=durable.append,
        partial_attempt_sink=lambda _event, _error: None,
    )
    transport = OllamaExactRouteTransport(
        model_io=OllamaModelIO(
            model="qwen3",
            stream_factory=lambda *_args, **_kwargs: _PausedResponse([]),
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
        callback=runtime.compose_event_callback(external.append),
        run_id="run-274",
        emit_stream=True,
    )
    result = {}

    def send():
        try:
            result["turn"] = transport.send(
                envelope=envelope, route=envelope.routes[0], retry_ordinal=0
            )
        except BaseException as error:
            result["error"] = error

    worker = Thread(target=send, daemon=True)
    worker.start()
    try:
        assert paused.wait(5), "provider did not reach the pause"
        assert [event["delta"] for event in external if event["type"] == "reasoning"] == ["plan"]
        assert [event for event in durable if event["type"] == "reasoning"] == []
        assert worker.is_alive()
    finally:
        continue_stream.set()
        worker.join(5)

    assert not worker.is_alive()
    assert "error" not in result, result.get("error")
    assert result["turn"].final_text == "ok"
    transport.release_buffered_events()
    assert [event["delta"] for event in durable if event["type"] == "reasoning"] == ["plan"]
    assert [event["delta"] for event in external if event["type"] == "reasoning"] == ["plan"]


def test_exact_route_failed_preview_is_reset_and_not_committed_on_retry():
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
    sends = []

    class _FailingResponse(_Response):
        def iter_lines(self):
            yield _line(thinking="failed draft")
            raise RuntimeError("stream interrupted")

    def stream_factory(*_args, **_kwargs):
        sends.append(True)
        if len(sends) == 1:
            return _FailingResponse([])
        return _Response([_line(thinking="accepted plan"), _line(content="ok", done=True)])

    durable = []
    external = []
    runtime = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=lambda _context: None,
        durable_event_sink=durable.append,
        partial_attempt_sink=lambda _event, _error: None,
    )
    transport = OllamaExactRouteTransport(
        model_io=OllamaModelIO(
            model="qwen3",
            stream_factory=stream_factory,
            default_payloads={},
            model_capabilities={},
        ),
        catalog=catalog,
        callback=runtime.compose_event_callback(external.append),
        run_id="run-274",
        emit_stream=True,
    )

    with pytest.raises(RuntimeError, match="stream interrupted"):
        transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=0)
    assert [event["type"] for event in external] == [
        "reasoning", "reasoning_preview_discarded",
    ]
    failed_preview_id = external[0]["provisional_reasoning_id"]
    assert external[1]["provisional_reasoning_id"] == failed_preview_id
    assert durable == []
    assert transport.buffered_events == ()

    result = transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=1)
    assert result.final_text == "ok"
    assert [event["delta"] for event in external if event["type"] == "reasoning"] == [
        "failed draft", "accepted plan",
    ]
    assert external[-1]["provisional_reasoning_id"] != failed_preview_id
    transport.release_buffered_events()
    assert [event["delta"] for event in durable if event["type"] == "reasoning"] == [
        "accepted plan",
    ]
    assert len([event for event in external if event["type"] == "reasoning"]) == 2


def test_release_persistence_failure_resets_visible_preview():
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
    external = []
    durable = []
    failure = RuntimeError("journal unavailable")

    def persist(event):
        if event["type"] == "reasoning":
            raise failure
        durable.append(event)

    runtime = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=lambda _context: None,
        durable_event_sink=persist,
        partial_attempt_sink=lambda _event, _error: None,
    )
    transport = OllamaExactRouteTransport(
        model_io=_io([_line(thinking="plan", content="ok", done=True)]),
        catalog=catalog,
        callback=runtime.compose_event_callback(external.append),
        run_id="run-274",
        emit_stream=True,
    )

    transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=0)
    assert [event["type"] for event in external] == ["reasoning"]
    with pytest.raises(RuntimeError, match="journal unavailable") as raised:
        transport.release_buffered_events()
    assert raised.value is failure
    assert [event["type"] for event in external] == [
        "reasoning", "request_messages", "reasoning_preview_discarded",
    ]
    assert [event["type"] for event in durable] == ["request_messages"]
    assert transport.buffered_events == ()


def test_late_reasoning_commit_failure_keeps_only_already_durable_preview():
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
    durable = []
    external = []
    failure = RuntimeError("second reasoning append failed")

    def persist(event):
        if event["type"] == "reasoning" and event["delta"] == "second":
            raise failure
        durable.append(event)

    runtime = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=lambda _context: None,
        durable_event_sink=persist,
        partial_attempt_sink=lambda _event, _error: None,
    )
    transport = OllamaExactRouteTransport(
        model_io=_io([
            _line(thinking="first"),
            _line(thinking="second", content="ok", done=True),
        ]),
        catalog=catalog,
        callback=runtime.compose_event_callback(external.append),
        run_id="run-274",
        emit_stream=True,
    )

    transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=0)
    previews = [event for event in external if event["type"] == "reasoning"]
    assert [event["delta"] for event in previews] == ["first", "second"]
    assert previews[0]["provisional_reasoning_id"] != previews[1]["provisional_reasoning_id"]
    with pytest.raises(RuntimeError, match="second reasoning append failed") as raised:
        transport.release_buffered_events()
    assert raised.value is failure
    assert [event["delta"] for event in durable if event["type"] == "reasoning"] == ["first"]
    resets = [event for event in external if event["type"] == "reasoning_preview_discarded"]
    assert [event["provisional_reasoning_id"] for event in resets] == [
        previews[1]["provisional_reasoning_id"],
    ]
    assert transport.buffered_events == ()


def test_provisional_callback_rejects_wrong_provider_and_shape_without_host_or_journal_effect():
    durable = []
    external = []
    runtime = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=lambda _context: None,
        durable_event_sink=durable.append,
        partial_attempt_sink=lambda _event, _error: None,
    )
    parent = runtime.compose_event_callback(external.append)
    callback = runtime.compose_event_callback(parent)
    raw = {
        "type": "reasoning", "run_id": "run-274", "iteration": 1,
        "provider": "ollama", "delta": "thinking",
    }
    preview_id = "a" * 32
    for invalid in (
        {**raw, "provider": "openai"},
        {**raw, "delta": ""},
        {**raw, "unexpected": True},
    ):
        with pytest.raises(ValueError, match="exact Ollama event"):
            callback.emit_provisional_reasoning(invalid, preview_id)
    with pytest.raises(ValueError, match="lowercase hex"):
        callback.emit_provisional_reasoning(raw, "invalid")
    assert durable == []
    assert external == []

    callback.emit_provisional_reasoning(raw, preview_id)
    assert durable == []
    assert external == [{**raw, "provisional_reasoning_id": preview_id}]
    callback.commit_provisional_reasoning(raw)
    assert durable == [raw]
    assert len(external) == 1


def test_nested_distinct_context_owners_commit_preview_to_each_journal_once():
    parent_durable = []
    child_durable = []
    external = []

    def runtime(owner_id, durable):
        return ContextRuntime._for_test(
            owner_id=owner_id,
            compiler=ContextCompiler(),
            request_factory=lambda _context: None,
            durable_event_sink=durable.append,
            partial_attempt_sink=lambda _event, _error: None,
        )

    parent = runtime("parent", parent_durable).compose_event_callback(external.append)
    child = runtime("child", child_durable).compose_event_callback(parent)
    raw = {
        "type": "reasoning", "run_id": "child-run", "iteration": 1,
        "provider": "ollama", "delta": "plan",
    }
    child.emit_provisional_reasoning(raw, "c" * 32)
    assert parent_durable == child_durable == []
    assert len(external) == 1
    child.commit_provisional_reasoning(raw)
    assert parent_durable == [raw]
    assert child_durable == [raw]
    assert len(external) == 1


def test_nested_journal_failure_retains_preview_already_committed_by_first_owner():
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
    child_durable = []
    parent_durable = []
    external = []
    failure = RuntimeError("parent journal unavailable")

    def parent_sink(event):
        if event["type"] == "reasoning":
            raise failure
        parent_durable.append(event)

    def runtime(owner_id, sink):
        return ContextRuntime._for_test(
            owner_id=owner_id,
            compiler=ContextCompiler(),
            request_factory=lambda _context: None,
            durable_event_sink=sink,
            partial_attempt_sink=lambda _event, _error: None,
        )

    parent = runtime("parent", parent_sink).compose_event_callback(external.append)
    child = runtime("child", child_durable.append).compose_event_callback(parent)
    transport = OllamaExactRouteTransport(
        model_io=_io([_line(thinking="retained", content="ok", done=True)]),
        catalog=catalog,
        callback=child,
        run_id="run-274",
        emit_stream=True,
    )
    transport.send(envelope=envelope, route=envelope.routes[0], retry_ordinal=0)
    assert [event["type"] for event in external] == ["reasoning"]
    with pytest.raises(RuntimeError, match="parent journal unavailable") as raised:
        transport.release_buffered_events()
    assert raised.value is failure
    assert [event["delta"] for event in child_durable if event["type"] == "reasoning"] == [
        "retained",
    ]
    assert [event for event in parent_durable if event["type"] == "reasoning"] == []
    assert [event["type"] for event in external] == ["reasoning", "request_messages"]
    assert transport.buffered_events == ()


def test_runtime_bridge_projects_preview_and_reset_with_strict_identity():
    bridge = RuntimeEventBridge(session_id="session-274", root_run_id="run-274")
    preview_id = "b" * 32
    raw = {
        "type": "reasoning", "run_id": "run-274", "iteration": 2,
        "provider": "ollama", "delta": "thinking",
        "provisional_reasoning_id": preview_id,
    }
    preview = bridge.normalize(raw)[0].to_dict()
    assert preview["type"] == "step.delta"
    assert preview["metadata"] == {
        "raw_type": "reasoning", "provider": "ollama",
        "provisional_reasoning_id": preview_id,
    }
    assert preview["payload"] == {
        "step_id": "model:run-274:turn-2:response",
        "step_type": "model_response", "kind": "reasoning", "delta": "thinking",
    }

    reset = bridge.normalize({
        "type": "reasoning_preview_discarded", "run_id": "run-274",
        "iteration": 2, "provider": "ollama",
        "provisional_reasoning_id": preview_id,
    })[0].to_dict()
    assert reset["type"] == "step.delta"
    assert reset["payload"] == {
        "step_id": "model:run-274:turn-2:response",
        "step_type": "model_response", "kind": "reasoning_reset",
        "preview_id": preview_id,
    }
    assert bridge.normalize({**raw, "provisional_reasoning_id": "invalid"}) == []
    assert bridge.normalize({**raw, "provisional_reasoning_id": None}) == []
    assert bridge.normalize({**raw, "delta": ""}) == []
    assert bridge.normalize({
        "type": "reasoning_preview_discarded", "run_id": "run-274",
        "iteration": 2, "provider": "ollama",
    }) == []
    assert bridge.normalize({
        "type": "reasoning_preview_discarded", "run_id": "run-274",
        "iteration": 2, "provider": "openai",
        "provisional_reasoning_id": preview_id,
    }) == []
