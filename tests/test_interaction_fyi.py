import threading

from unchain.interaction.fyi import FyiChannel, FyiMessage


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
