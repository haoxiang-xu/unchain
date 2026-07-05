from unchain.interaction.btw import ProgressDigest, build_btw_prompt


def test_digest_collects_iterations_and_tools():
    digest = ProgressDigest()
    digest({"type": "iteration_started", "iteration": 0})
    digest({"type": "tool_started", "tool": "read_file", "arguments": {"path": "a.py"}})
    digest({"type": "response_received", "final_text": "working on parsing"})
    digest({"type": "iteration_started", "iteration": 1})

    summary = digest.summary()
    assert "iteration 2" in summary or "iterations: 2" in summary
    assert "read_file" in summary


def test_digest_is_bounded():
    digest = ProgressDigest(max_entries=5)
    for i in range(100):
        digest({"type": "tool_started", "tool": f"tool_{i}"})
    summary = digest.summary()
    assert "tool_99" in summary
    assert "tool_0" not in summary  # 只保留最近 max_entries 条


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
