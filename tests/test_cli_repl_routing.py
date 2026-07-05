import threading
from types import SimpleNamespace

from unchain.cli import repl
from unchain.cli.repl import _fallback_renderer, _run_worker, build_followup_messages, route_input


def test_route_input_prefixes():
    assert route_input("/btw how long?") == ("btw", "how long?")
    assert route_input("/fyi also support Chinese") == ("fyi", "also support Chinese")
    assert route_input("/steer next refactor the tests") == ("steer", "next refactor the tests")


def test_route_input_bare_text_is_unknown():
    assert route_input("hello there") == ("unknown", "hello there")


def test_route_input_empty_and_whitespace():
    assert route_input("") == ("empty", "")
    assert route_input("   ") == ("empty", "")
    assert route_input("/fyi   ") == ("empty", "")


def test_fallback_renderer_prints_content_on_final_message(capsys):
    # This is the rich-absent path: it is the *only* thing that prints the
    # model's answer when TerminalRenderer isn't available, so it must work.
    _fallback_renderer({"type": "final_message", "content": "the answer is 42"})
    captured = capsys.readouterr()
    assert "the answer is 42" in captured.out


def test_fallback_renderer_ignores_other_events(capsys):
    _fallback_renderer({"type": "token_delta", "delta": "x"})
    _fallback_renderer(None)
    _fallback_renderer("not a dict")
    _fallback_renderer({})
    captured = capsys.readouterr()
    assert captured.out == ""


class _RaisingAgent:
    def run(self, task, callback=None):
        raise RuntimeError("boom")


class _OkAgent:
    def run(self, task, callback=None):
        return "ok"


def test_run_worker_sets_done_when_agent_raises(capsys):
    done = threading.Event()
    _run_worker(_RaisingAgent(), "do something", lambda event: None, done)
    assert done.is_set()
    captured = capsys.readouterr()
    assert "[run failed]" in captured.out
    assert "boom" in captured.out


def test_run_worker_sets_done_on_success():
    done = threading.Event()
    _run_worker(_OkAgent(), "do something", lambda event: None, done)
    assert done.is_set()


def test_run_worker_stores_result_in_holder_on_success():
    done = threading.Event()
    result_holder = []
    _run_worker(_OkAgent(), "do something", lambda event: None, done, result_holder=result_holder)
    assert done.is_set()
    assert result_holder == ["ok"]


def test_run_worker_leaves_holder_empty_when_agent_raises(capsys):
    done = threading.Event()
    result_holder = []
    _run_worker(
        _RaisingAgent(), "do something", lambda event: None, done, result_holder=result_holder
    )
    assert done.is_set()
    assert result_holder == []
    captured = capsys.readouterr()
    assert "[run failed]" in captured.out


def test_build_followup_messages_appends_merged_as_user_turn():
    prior_messages = [
        {"role": "user", "content": "调研三家 CI 服务商的定价"},
        {"role": "assistant", "content": "已完成初步调研"},
    ]
    result = build_followup_messages(prior_messages, "顺便把 GitHub Actions 也加进对比")

    assert result == [
        {"role": "user", "content": "调研三家 CI 服务商的定价"},
        {"role": "assistant", "content": "已完成初步调研"},
        {"role": "user", "content": "顺便把 GitHub Actions 也加进对比"},
    ]
    # must not mutate the caller's prior_messages list
    assert result is not prior_messages
    assert len(prior_messages) == 2


def test_build_followup_messages_with_empty_prior_history():
    result = build_followup_messages([], "merged steer text")
    assert result == [{"role": "user", "content": "merged steer text"}]


class _FakeProgressDigest:
    def summary(self) -> str:
        return "iterations: 1"


def test_side_answer_uses_last_assistant_text(monkeypatch, capsys):
    class _FakeSideAgent:
        def __init__(self, **kwargs):
            pass

        def run(self, messages, max_iterations=1):
            return SimpleNamespace(
                messages=[
                    {"role": "user", "content": "how long?"},
                    {"role": "assistant", "content": "  about 5 minutes  "},
                ]
            )

    monkeypatch.setattr(repl, "Agent", _FakeSideAgent)
    args = SimpleNamespace(provider="ollama", model="llama3", side_model=None)
    repl._side_answer(args, "the task", _FakeProgressDigest(), "how long?")
    captured = capsys.readouterr()
    assert "[btw] about 5 minutes" in captured.out


def test_side_answer_prints_failure_instead_of_raising(monkeypatch, capsys):
    class _RaisingSideAgent:
        def __init__(self, **kwargs):
            raise RuntimeError("no provider configured")

    monkeypatch.setattr(repl, "Agent", _RaisingSideAgent)
    args = SimpleNamespace(provider="ollama", model="llama3", side_model=None)
    repl._side_answer(args, "the task", _FakeProgressDigest(), "how long?")
    captured = capsys.readouterr()
    assert "[btw failed]" in captured.out
    assert "no provider configured" in captured.out
