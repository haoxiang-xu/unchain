import threading

from unchain.interaction.fyi import FyiChannel, FyiInjectionHarness, FyiMessage, wrap_fyi
from unchain.kernel.delta import AppendMessagesOp
from unchain.kernel.harness import HarnessContext
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


def test_harness_drains_channel_into_append_delta_with_event():
    channel = FyiChannel()
    mid = channel.post("new requirement")
    harness = FyiInjectionHarness(channel=channel)

    delta = harness.build_delta(_make_context())

    assert channel.pending_count() == 0
    append_ops = [op for op in delta.ops if isinstance(op, AppendMessagesOp)]
    assert len(append_ops) == 1
    assert append_ops[0].messages[0]["role"] == "user"
    assert "new requirement" in append_ops[0].messages[0]["content"]
    assert delta.state_updates["transcript_append"] == append_ops[0].messages

    event_ops = [op for op in delta.context_ops if getattr(op, "type", "") == "fyi_injected"]
    assert len(event_ops) == 1
    assert event_ops[0].payload["count"] == 1
    assert event_ops[0].payload["messages"][0]["message_id"] == mid

    assert harness.name == "fyi_injection"
    assert harness.phases == ("before_model",)
    assert harness.order == 180
