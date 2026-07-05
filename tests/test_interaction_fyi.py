import threading

from unchain.agent import Agent, InteractionModule, ToolsModule
from unchain.capabilities import ContextTarget, InsertMessagesOp
from unchain.events.normalizer import RuntimeEventNormalizerContext, normalize_raw_event
from unchain.events.types import RUNTIME_EVENT_TYPES
from unchain.interaction.fyi import FyiChannel, FyiInjectionHarness, FyiMessage, wrap_fyi
from unchain.kernel import ModelTurnResult, ToolCall
from unchain.kernel.harness import HarnessContext
from unchain.kernel.loop import KernelLoop
from unchain.kernel.state import RunState


def test_fyi_channel_post_and_drain_preserves_fifo_order():
    channel = FyiChannel()
    id1 = channel.post("first")
    id2 = channel.post("second", origin="system")

    assert channel.pending_count() == 2
    drained = channel.drain()
    assert channel.pending_count() == 0
    assert channel.drain() == []

    assert [m.text for m in drained] == ["first", "second"]
    assert [m.origin for m in drained] == ["user", "system"]
    assert drained[0].message_id == id1
    assert drained[1].message_id == id2
    assert isinstance(drained[0], FyiMessage)


def test_fyi_channel_is_thread_safe_under_concurrent_posts():
    channel = FyiChannel()

    def worker(prefix: str) -> None:
        for i in range(100):
            channel.post(f"{prefix}-{i}")

    threads = [threading.Thread(target=worker, args=(str(t),)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert channel.pending_count() == 400
    assert len(channel.drain()) == 400


def test_wrap_fyi_user_message_format():
    wrapped = wrap_fyi(FyiMessage(text="also support Chinese"))
    assert wrapped["role"] == "user"
    assert wrapped["content"].startswith("<fyi_message>")
    assert wrapped["content"].rstrip().endswith("</fyi_message>")
    assert "also support Chinese" in wrapped["content"]
    assert "Do not ignore this message." in wrapped["content"]


def test_wrap_fyi_system_origin_uses_side_note_wording():
    wrapped = wrap_fyi(FyiMessage(text="Q: eta? A: about 2 more steps", origin="system"))
    assert "side assistant already replied" in wrapped["content"]


def _make_context(phase: str = "before_model") -> HarnessContext:
    state = RunState(transcript=[{"role": "user", "content": "task"}])
    return HarnessContext(state=state, phase=phase, event={})


def test_harness_returns_none_when_channel_empty():
    harness = FyiInjectionHarness(channel=FyiChannel())
    assert harness.build_delta(_make_context()) is None


def test_harness_drains_channel_into_conversation_insert_delta_with_event():
    channel = FyiChannel()
    mid = channel.post("new requirement")
    harness = FyiInjectionHarness(channel=channel)

    delta = harness.build_delta(_make_context())

    assert channel.pending_count() == 0
    insert_ops = [op for op in delta.context_ops if isinstance(op, InsertMessagesOp)]
    assert len(insert_ops) == 1
    assert insert_ops[0].target == ContextTarget.CONVERSATION
    assert insert_ops[0].messages[0]["role"] == "user"
    assert "new requirement" in insert_ops[0].messages[0]["content"]

    event_ops = [op for op in delta.context_ops if getattr(op, "type", "") == "fyi_injected"]
    assert len(event_ops) == 1
    assert event_ops[0].payload["count"] == 1
    assert event_ops[0].payload["messages"][0]["message_id"] == mid

    assert harness.name == "fyi_injection"
    assert harness.phases == ("before_model",)
    assert harness.order == 180


def test_fyi_injection_is_model_visible_and_persisted_same_turn_e2e():
    """Regression for the bug where fyi injection was invisible in the same
    turn: went through capabilities.RunDelta.ops (dead code for apply_run_delta)
    instead of context_ops, so state.latest_messages() never saw it until the
    *next* iteration's transcript rebuild (or never, if the run ended first).

    This drives the real dispatch path (KernelLoop.dispatch_phase ->
    apply_run_delta) instead of asserting on the shape of the returned delta.
    """
    channel = FyiChannel()
    mid = channel.post("the user changed their mind mid-task")
    harness = FyiInjectionHarness(channel=channel)
    loop = KernelLoop(harnesses=[harness])

    state = loop.seed_state([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "original task"},
    ])

    captured_events = []
    loop.dispatch_phase(
        state,
        phase="before_model",
        event={"callback": captured_events.append, "run_id": "run-1", "iteration": 0},
    )

    # (1) Same-turn model visibility: the version graph the model reads from
    # must already contain the injected message.
    latest = state.latest_messages()
    assert any(
        message.get("role") == "user" and "the user changed their mind mid-task" in message.get("content", "")
        for message in latest
    ), latest

    # (1) Persistence: the flat transcript (what final result.messages is
    # built from) must also contain it.
    assert any(
        message.get("role") == "user" and "the user changed their mind mid-task" in message.get("content", "")
        for message in state.transcript
    ), state.transcript

    # Original messages are preserved, fyi message appended at the end.
    assert latest[0] == {"role": "system", "content": "sys"}
    assert latest[1] == {"role": "user", "content": "original task"}
    assert "the user changed their mind mid-task" in latest[2]["content"]

    # (2) Channel drained.
    assert channel.pending_count() == 0
    assert channel.drain() == []

    # (2) Structured event emitted through dispatch_phase's event path.
    fyi_events = [event for event in captured_events if event.get("type") == "fyi_injected"]
    assert len(fyi_events) == 1
    assert fyi_events[0]["count"] == 1
    assert fyi_events[0]["messages"][0]["message_id"] == mid
    assert fyi_events[0]["messages"][0]["text"] == "the user changed their mind mid-task"


def test_fyi_posted_mid_run_reaches_next_model_request():
    channel = FyiChannel()
    seen_requests: list[list[dict]] = []

    def echo(text: str) -> dict:
        return {"echo": text}

    class FakeModelIO:
        provider = "ollama"
        model = "llama3"

        def __init__(self):
            self.calls = 0

        def fetch_turn(self, request):
            seen_requests.append(list(request.messages))
            self.calls += 1
            if self.calls == 1:
                # Simulate the user chiming in while the first model turn is
                # being handled.
                channel.post("please also handle Chinese input")
                return ModelTurnResult(
                    assistant_messages=[{"role": "assistant", "content": ""}],
                    tool_calls=[ToolCall(call_id="c1", name="echo", arguments={"text": "hi"})],
                    final_text="",
                )
            return ModelTurnResult(
                assistant_messages=[{"role": "assistant", "content": "done"}],
                tool_calls=[],
                final_text="done",
            )

    events: list[dict] = []
    agent = Agent(
        name="fyi_agent",
        provider="ollama",
        model="llama3",
        instructions="Be concise.",
        model_io_factory=lambda spec, ctx: FakeModelIO(),
        modules=(
            ToolsModule(tools=(echo,)),
            InteractionModule(fyi_channel=channel),
        ),
    )
    result = agent.run("do the task", callback=events.append, max_iterations=3)

    assert result.status == "completed"
    # The second model request should contain the wrapped fyi message.
    second = seen_requests[1]
    fyi_msgs = [m for m in second if "<fyi_message>" in str(m.get("content", ""))]
    assert len(fyi_msgs) == 1
    assert "handle Chinese input" in fyi_msgs[0]["content"]
    # The fyi message lands in the final transcript.
    assert any("<fyi_message>" in str(m.get("content", "")) for m in result.messages)
    # The callback event stream carries the raw fyi_injected event; normalizing
    # it to "interaction.fyi_injected" is a separate opt-in step
    # (unchain.events.normalizer.normalize_raw_event), not applied here.
    assert any("fyi_injected" in str(e.get("type", "")) or "fyi_injected" in str(e.get("event", "")) for e in events)


def test_fyi_injected_event_normalizes_to_interaction_namespace():
    assert "interaction.fyi_injected" in RUNTIME_EVENT_TYPES

    # Raw event shape as actually produced on the callback stream: emit_event
    # (kernel/loop.py) flattens the EmitEventOp payload keys alongside
    # type/run_id/iteration -- there is no nested "payload" key.
    raw = {
        "type": "fyi_injected",
        "run_id": "run-1",
        "iteration": 0,
        "count": 1,
        "messages": [{"message_id": "abc", "origin": "user", "text": "hello"}],
    }
    context = RuntimeEventNormalizerContext(session_id="thread-1", root_run_id="run-1")

    events = normalize_raw_event(raw, context=context)

    assert len(events) == 1
    event = events[0]
    assert event.type == "interaction.fyi_injected"
    assert event.visibility == "user"
    assert event.surface.slot == "trace_inline"
    assert event.surface.scope == "turn"
    assert event.payload["count"] == 1
    assert event.payload["messages"][0]["message_id"] == "abc"
    assert event.payload["messages"][0]["origin"] == "user"
    assert event.payload["messages"][0]["text"] == "hello"
