from unchain.interaction.btw import ProgressDigest, build_btw_prompt


def test_digest_collects_iterations_and_tools():
    digest = ProgressDigest()
    digest({"type": "iteration_started", "iteration": 0})
    digest({"type": "tool_call", "tool_name": "read_file", "arguments": {"path": "a.py"}})
    digest({"type": "final_message", "content": "working on parsing"})
    digest({"type": "iteration_started", "iteration": 1})

    summary = digest.summary()
    assert "iteration 2" in summary or "iterations: 2" in summary
    assert "read_file" in summary


def test_digest_is_bounded():
    digest = ProgressDigest(max_entries=5)
    for i in range(100):
        digest({"type": "tool_call", "tool_name": f"tool_{i}"})
    summary = digest.summary()
    assert "tool_99" in summary
    assert "tool_0" not in summary  # 只保留最近 max_entries 条


def test_digest_uses_real_tool_name_key():
    # Real emitter (tools/execution.py) emits tool_name=, not tool=.
    digest = ProgressDigest()
    digest({"type": "tool_call", "tool_name": "read_file", "call_id": "c1"})
    summary = digest.summary()
    assert "read_file" in summary


def test_digest_uses_real_final_message_content_key():
    # Real emitter (kernel/lifecycle_events.py build_final_message_payload) emits content=.
    digest = ProgressDigest()
    digest({"type": "final_message", "content": "working on parsing"})
    summary = digest.summary()
    assert "working on parsing" in summary


def test_digest_never_raises_on_non_dict_or_missing_keys():
    digest = ProgressDigest()
    digest(None)
    digest("not a dict")
    digest({})
    digest(123)
    digest([1, 2, 3])
    # digest should still be usable afterwards
    summary = digest.summary()
    assert "iterations: 0" in summary


def test_build_btw_prompt_contains_task_digest_and_question():
    messages = build_btw_prompt(
        original_task="refactor the parser",
        digest_summary="iterations: 2; recent tools: read_file",
        question="how long will this take?",
    )
    assert messages[0]["role"] == "system"
    assert "side assistant" in messages[0]["content"]
    assert "refactor the parser" in messages[0]["content"]
    assert "read_file" in messages[0]["content"]
    assert messages[-1] == {"role": "user", "content": "how long will this take?"}
