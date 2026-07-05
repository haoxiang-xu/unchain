import threading

from unchain.interaction.steer import SteerBuffer, merge_steered_texts


def test_merge_single_text_returns_it_verbatim():
    assert merge_steered_texts(["just one thing"]) == "just one thing"


def test_merge_multiple_texts_keeps_order_as_numbered_list():
    merged = merge_steered_texts(["fix the header", "then add tests"])
    assert merged.index("fix the header") < merged.index("then add tests")
    assert "1. fix the header" in merged
    assert "2. then add tests" in merged
    assert merged.startswith("The user sent several follow-up requests")


def test_steer_buffer_drain_merged():
    buf = SteerBuffer()
    assert buf.drain_merged() is None
    buf.post("a")
    buf.post("b")
    assert buf.pending_count() == 2
    merged = buf.drain_merged()
    assert "1. a" in merged and "2. b" in merged
    assert buf.pending_count() == 0
    assert buf.drain_merged() is None


def test_steer_buffer_thread_safety():
    buf = SteerBuffer()
    threads = [threading.Thread(target=lambda i=i: buf.post(f"t{i}")) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert buf.pending_count() == 50
