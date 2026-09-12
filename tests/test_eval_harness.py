import json
import os
from pathlib import Path

from unchain.kernel import KernelRunResult

from tests.evals.cases import build_eval_case, get_eval_case
from tests.evals.env import filter_model_specs, load_root_env
from tests.evals.judge import build_judge_messages
from tests.evals.notebook_sessions import describe_pending_request, load_notebook_session, resume_notebook_session, start_notebook_session
from tests.evals.runner import (
    _build_standard_payload,
    _coerce_agent_run_output,
    _max_context_window_tokens,
    run_benchmark_suite,
    run_single_model_eval,
)
from tests.evals.scoring import score_run_artifact
from tests.evals.serialization import persist_judge_report, persist_run_artifact
from tests.evals.types import JudgeReport, ModelSpec, RunArtifact
from tests.evals.workspace import diff_workspace_snapshots, prepare_workspace, snapshot_workspace


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_root_env_reads_repo_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")

    loaded = load_root_env(tmp_path)

    assert loaded["OPENAI_API_KEY"] == "from-dotenv"
    assert os.environ["OPENAI_API_KEY"] == "from-dotenv"


def test_filter_model_specs_skips_missing_keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    ready, skipped = filter_model_specs(
        [
            {
                "provider": "openai",
                "model": "gpt-5",
                "label": "gpt-5",
                "api_key_env": "OPENAI_API_KEY",
            },
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
                "label": "claude",
                "api_key_env": "ANTHROPIC_API_KEY",
            },
        ]
    )

    assert len(ready) == 1
    assert ready[0][0].label == "gpt-5"
    assert len(skipped) == 1
    assert skipped[0]["label"] == "claude"


def test_filter_model_specs_uses_current_provider_and_auth_policy():
    ready, skipped = filter_model_specs(
        [
            {"provider": "ollama", "model": "qwen3", "label": "local"},
            {"provider": "unsupported-provider", "model": "unknown", "label": "unsupported"},
            {"provider": "gemini", "model": "gemini-2.5-flash", "label": "gemini"},
            {"provider": "hyperspace", "model": "hyperspace--claude", "label": "hyperspace"},
        ],
        env={},
    )

    assert [(spec.label, api_key) for spec, api_key, _ in ready] == [("local", None)]
    skipped_by_label = {item["label"]: item["reason"] for item in skipped}
    assert "unsupported provider" in skipped_by_label["unsupported"]
    assert "GEMINI_API_KEY/GOOGLE_API_KEY" in skipped_by_label["gemini"]
    assert "HYPERSPACE_API_KEY" in skipped_by_label["hyperspace"]


def test_eval_model_capabilities_resolve_snapshot_names():
    openai_snapshot = ModelSpec(
        provider="openai",
        model="gpt-5-2025-08-07",
        label="openai-snapshot",
    )
    anthropic_snapshot = ModelSpec(
        provider="anthropic",
        model="claude-opus-4-6-20260301",
        label="anthropic-snapshot",
    )

    assert _max_context_window_tokens(openai_snapshot, REPO_ROOT) == 1_047_576
    assert _max_context_window_tokens(anthropic_snapshot, REPO_ROOT) == 200_000
    assert _build_standard_payload(
        model_spec=openai_snapshot,
        repo_root=REPO_ROOT,
        max_output_tokens=321,
    ) == {
        "max_output_tokens": 321,
        "store": False,
        "reasoning": {"effort": "medium"},
    }


def test_legacy_agent_tuple_receives_current_bundle_metadata():
    messages, bundle = _coerce_agent_run_output(
        ([{"role": "assistant", "content": "done"}], {"status": "completed"}),
        callback_events=[
            {
                "type": "run_completed",
                "bundle": {"consumed_tokens": 8, "last_turn_tokens": 3},
            }
        ],
        model="gpt-5-snapshot",
        max_context_window_tokens=1_000,
    )

    assert messages[-1]["content"] == "done"
    assert bundle["model"] == "gpt-5-snapshot"
    assert bundle["consumed_tokens"] == 8
    assert bundle["max_context_window_tokens"] == 1_000
    assert bundle["context_window_used_pct"] == 0.3


def test_prepare_workspace_creates_isolated_fixture_copies():
    case = get_eval_case("fixture_debug", REPO_ROOT)

    with prepare_workspace(case, repo_root=REPO_ROOT) as first_workspace:
        target = first_workspace / "src" / "reporting.py"
        original = target.read_text(encoding="utf-8")
        target.write_text(original + "\n# mutated\n", encoding="utf-8")
        assert "# mutated" in target.read_text(encoding="utf-8")

    with prepare_workspace(case, repo_root=REPO_ROOT) as second_workspace:
        target = second_workspace / "src" / "reporting.py"
        assert "# mutated" not in target.read_text(encoding="utf-8")


def test_workspace_snapshot_diff_reports_source_changes_and_ignores_caches(tmp_path):
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("before\n", encoding="utf-8")
    before = snapshot_workspace(tmp_path)

    source.write_text("after\n", encoding="utf-8")
    created = tmp_path / "notes.txt"
    created.write_text("new\n", encoding="utf-8")
    cache = tmp_path / "__pycache__" / "module.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")

    changes = diff_workspace_snapshots(before, snapshot_workspace(tmp_path))
    assert changes == {
        "created_paths": ["notes.txt"],
        "modified_paths": ["src/module.py"],
        "deleted_paths": [],
    }


def test_artifact_serialization_writes_json_files(tmp_path):
    run_artifact = RunArtifact(
        run_id="fixture_debug__demo",
        suite_id="suite-1",
        case_id="fixture_debug",
        case_title="Debug A Failing Fixture",
        provider="openai",
        model="gpt-5",
        model_label="gpt-5",
        status="completed",
        started_at="2026-03-19T00:00:00+00:00",
        duration_seconds=1.23,
        workspace_mode="fixture_copy",
        workspace_source="tests/evals/fixtures/fixture_debug",
        final_answer="Use pytest and inspect src/reporting.py.",
        messages=[{"role": "assistant", "content": "ok"}],
    )
    judge_report = JudgeReport(
        case_id="fixture_debug",
        case_title="Debug A Failing Fixture",
        provider="openai",
        model="gpt-5",
        model_label="gpt-5",
        judge_provider="openai",
        judge_model="gpt-5",
        judge_label="judge",
        status="completed",
        overall_score=88,
        summary="Strong run.",
    )

    run_path = persist_run_artifact(tmp_path / "run.json", run_artifact)
    judge_path = persist_judge_report(tmp_path / "judge.json", judge_report)

    run_payload = json.loads(run_path.read_text(encoding="utf-8"))
    judge_payload = json.loads(judge_path.read_text(encoding="utf-8"))

    assert run_payload["run_id"] == "fixture_debug__demo"
    assert judge_payload["overall_score"] == 88


def test_rule_based_scorer_rewards_expected_evidence():
    case = get_eval_case("fixture_debug", REPO_ROOT)
    run_artifact = RunArtifact(
        run_id="fixture_debug__demo",
        suite_id="suite-1",
        case_id="fixture_debug",
        case_title=case.title,
        provider="openai",
        model="gpt-5",
        model_label="gpt-5",
        status="completed",
        started_at="2026-03-19T00:00:00+00:00",
        duration_seconds=1.5,
        workspace_mode=case.workspace_mode,
        workspace_source=case.workspace_source,
        final_answer=(
            "I ran pytest -q. The failure comes from src/reporting.py and tests/test_reporting.py. "
            "The bug is floor division in format_success_rate, so 3/4 becomes 0 before multiplying."
        ),
        callback_events=[
            {"type": "tool_call", "tool_name": "shell", "arguments": {"command": "pytest -q"}},
            {"type": "tool_call", "tool_name": "read", "arguments": {"path": "src/reporting.py"}},
            {"type": "tool_result", "tool_name": "shell", "result": {"error": "assertion failed"}},
        ],
    )
    run_artifact.tool_usage = {
        "total_calls": 2,
        "by_tool": {"shell": 1, "read": 1},
        "failed_calls": [{"tool": "shell", "error": "assertion failed"}],
        "blocked_attempts": [],
        "denied_tools": [],
        "terminal_commands": ["pytest -q"],
    }

    result = score_run_artifact(run_artifact, case)

    assert result["score_pct"] >= 80
    assert any(check["name"] == "tool:shell" and check["passed"] for check in result["checks"])
    assert not next(
        check["passed"]
        for check in result["checks"]
        if check["name"] == "path:tests/test_reporting.py"
    )

    run_artifact.status = "max_iterations"
    ineligible = score_run_artifact(run_artifact, case)
    assert ineligible["raw_score_pct"] >= 80
    assert ineligible["score_pct"] == 0
    assert ineligible["eligible"] is False

    run_artifact.status = "completed"
    run_artifact.tool_usage["workspace_changes"] = ["src/reporting.py"]
    changed_workspace = score_run_artifact(run_artifact, case)
    assert changed_workspace["score_pct"] == 0
    assert changed_workspace["eligibility_failures"] == ["workspace_changed"]


def test_build_judge_messages_embeds_message_list():
    case = get_eval_case("repo_trace", REPO_ROOT)
    run_artifact = RunArtifact(
        run_id="repo_trace__demo",
        suite_id="suite-1",
        case_id="repo_trace",
        case_title=case.title,
        provider="openai",
        model="gpt-5",
        model_label="gpt-5",
        status="completed",
        started_at="2026-03-19T00:00:00+00:00",
        duration_seconds=1.0,
        workspace_mode=case.workspace_mode,
        workspace_source=case.workspace_source,
        final_answer="See src/unchain/agent/agent.py and src/unchain/kernel/loop.py.",
        messages=[{"role": "assistant", "content": "See src/unchain/agent/agent.py and src/unchain/kernel/loop.py."}],
    )

    messages = build_judge_messages(case=case, run_artifact=run_artifact, rubric_weights=case.rubric_weights)
    payload = json.loads(messages[0]["content"].split("\n\n", 1)[1])

    assert payload["run_artifact"]["message_list"] == run_artifact.messages


class FakeAgent:
    candidate_messages = [
        {"role": "user", "content": "prompt"},
        {
            "role": "assistant",
            "content": (
                "I ran pytest -q and inspected src/reporting.py plus tests/test_reporting.py. "
                "The bug is floor division in the rate calculation."
            ),
        },
    ]
    judge_seen_message_list = None

    def __init__(self, *, name, provider, model, api_key=None, instructions="", tools=None, **kwargs):
        self.name = name
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.instructions = instructions
        self.tools = tools or []

    def run(self, messages, payload=None, response_format=None, callback=None, max_iterations=None, **kwargs):
        del payload, max_iterations, kwargs
        if self.name.startswith("candidate-"):
            if callback is not None:
                callback({"type": "tool_call", "tool_name": "shell", "arguments": {"command": "pytest -q"}})
                callback({"type": "tool_result", "tool_name": "shell", "result": {"ok": False, "error": "assertion failed"}})
            return list(self.candidate_messages), {
                "status": "completed",
                "consumed_tokens": 42,
                "input_tokens": 30,
                "output_tokens": 12,
                "context_window_used_pct": 0.5,
                "max_context_window_tokens": 1000,
            }

        assert response_format is not None
        assert isinstance(messages, list)
        payload = json.loads(messages[0]["content"].split("\n\n", 1)[1])
        FakeAgent.judge_seen_message_list = payload["run_artifact"]["message_list"]
        return [
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "overall_score": 91,
                        "rubric_scores": {
                            "correctness": 92,
                            "debugging": 90,
                            "tool_strategy": 88,
                            "efficiency": 94,
                        },
                        "summary": "Solid transcript.",
                        "failure_modes": [],
                        "debug_notes": ["The diagnosis is grounded."],
                        "recommendations": ["Keep the root-cause explanation concise."],
                        "prompt_suggestions": ["Ask for explicit failing command output."],
                        "tooling_suggestions": ["Use read for tighter evidence snippets."],
                    }
                ),
            }
        ], {"status": "completed", "consumed_tokens": 11, "input_tokens": 7, "output_tokens": 4}


def test_run_benchmark_suite_passes_saved_message_list_to_judge(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = run_benchmark_suite(
        repo_root=REPO_ROOT,
        model_specs=[
            {
                "provider": "openai",
                "model": "gpt-5",
                "label": "gpt-5",
                "api_key_env": "OPENAI_API_KEY",
            }
        ],
        judge_model_spec={
            "provider": "openai",
            "model": "gpt-5",
            "label": "judge",
            "api_key_env": "OPENAI_API_KEY",
        },
        selected_case_ids=["fixture_debug"],
        artifacts_root=tmp_path / "artifacts",
        candidate_agent_cls=FakeAgent,
        judge_agent_cls=FakeAgent,
    )

    run_artifact_path = Path(result["leaderboard_rows"][0]["run_artifact_path"])
    saved_run_payload = json.loads(run_artifact_path.read_text(encoding="utf-8"))

    assert FakeAgent.judge_seen_message_list == saved_run_payload["messages"]
    assert saved_run_payload["token_usage"]["consumed_tokens"] == 42
    assert saved_run_payload["token_usage"]["input_tokens"] == 30
    assert saved_run_payload["token_usage"]["output_tokens"] == 12
    assert result["leaderboard_rows"][0]["input_tokens"] == 30
    assert result["leaderboard_rows"][0]["output_tokens"] == 12


def test_run_single_model_eval_runs_ollama_without_api_key(tmp_path):
    case = build_eval_case(
        case_id="ollama_no_key",
        title="Ollama Without API Key",
        task_prompt="Inspect the fixture.",
        workspace_mode="fixture_copy",
        workspace_source="tests/evals/fixtures/fixture_debug",
        rule_checks={"min_tool_calls": 0},
    )

    result = run_single_model_eval(
        repo_root=REPO_ROOT,
        case=case,
        model_spec={
            "provider": "ollama",
            "model": "qwen3",
            "label": "local-candidate",
        },
        judge_model_spec={
            "provider": "ollama",
            "model": "qwen3",
            "label": "local-judge",
        },
        artifacts_root=tmp_path / "ollama-no-key",
        candidate_agent_cls=FakeAgent,
        judge_agent_cls=FakeAgent,
    )

    assert result["run_artifact"]["status"] == "completed"
    assert result["run_artifact"]["skip_reason"] is None
    assert result["judge_report"]["status"] == "completed"


def test_run_single_model_eval_supports_notebook_defined_case(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    case = build_eval_case(
        case_id="custom_notebook_case",
        title="Custom Notebook Case",
        task_prompt="Inspect the repo and explain the structure.",
        workspace_mode="repo_copy",
        workspace_source=".",
        allowed_toolkits=["core"],
        toolkit_options={},
        rule_checks={"min_tool_calls": 0},
        candidate_instructions="Only use core tools.",
    )

    result = run_single_model_eval(
        repo_root=REPO_ROOT,
        case=case,
        model_spec={
            "provider": "openai",
            "model": "gpt-5",
            "label": "gpt-5",
            "api_key_env": "OPENAI_API_KEY",
        },
        judge_model_spec={
            "provider": "openai",
            "model": "gpt-5",
            "label": "judge",
            "api_key_env": "OPENAI_API_KEY",
        },
        artifacts_root=tmp_path / "single-run",
        candidate_agent_cls=FakeAgent,
        judge_agent_cls=FakeAgent,
    )

    assert result["case"]["id"] == "custom_notebook_case"
    assert result["model_spec"]["label"] == "gpt-5"
    assert Path(result["summary_path"]).exists()


def test_run_single_model_eval_uses_central_default_judge_model(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    case = build_eval_case(
        case_id="default_judge_case",
        title="Default Judge Case",
        task_prompt="Inspect the repo and explain the structure.",
        workspace_mode="repo_copy",
        workspace_source=".",
        allowed_toolkits=["core"],
        toolkit_options={},
        rule_checks={"min_tool_calls": 0},
    )

    result = run_single_model_eval(
        repo_root=REPO_ROOT,
        case=case,
        model_spec={
            "provider": "openai",
            "model": "gpt-5",
            "label": "gpt-5",
            "api_key_env": "OPENAI_API_KEY",
        },
        artifacts_root=tmp_path / "default-judge-run",
        candidate_agent_cls=FakeAgent,
        judge_agent_cls=FakeAgent,
    )

    assert result["judge_model_spec"]["provider"] == "anthropic"
    assert result["judge_model_spec"]["model"] == "claude-opus-4-6"


class FakeInteractiveAgent:
    last_run_tool_runtime_config = None
    last_resume_tool_runtime_config = None

    def __init__(self, *, name, provider, model, api_key=None, instructions="", tools=None, **kwargs):
        self.name = name
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.instructions = instructions
        self.tools = tools or []

    def run(self, messages, payload=None, callback=None, max_iterations=None, session_id=None, memory_namespace=None, **kwargs):
        type(self).last_run_tool_runtime_config = kwargs.get("tool_runtime_config")
        del messages, payload, max_iterations, kwargs
        assert session_id == memory_namespace
        if callback is not None:
            callback(
                {
                    "type": "tool_call",
                    "tool_name": "ask_user_question",
                    "arguments": {
                        "title": "Theme",
                        "question": "Pick a visual direction",
                    },
                }
            )
        return KernelRunResult(
            messages=[
                {"role": "assistant", "content": "I need to confirm the visual direction first."}
            ],
            status="awaiting_human_input",
            human_input_request={
                "request_id": "req_1",
                "title": "Theme",
                "question": "Pick a visual direction",
                "selection_mode": "single",
                "options": [
                    {"label": "Arcade", "value": "arcade"},
                    {"label": "Minimal", "value": "minimal"},
                ],
                "allow_other": False,
                "min_selected": 1,
                "max_selected": 1,
            },
            continuation={
                "type": "human_input_continuation",
                "request": {
                    "request_id": "req_1",
                    "kind": "selector",
                    "title": "Theme",
                    "question": "Pick a visual direction",
                    "selection_mode": "single",
                    "options": [
                        {"label": "Arcade", "value": "arcade"},
                        {"label": "Minimal", "value": "minimal"},
                    ],
                    "allow_other": False,
                    "min_selected": 1,
                    "max_selected": 1,
                },
                "payload": {},
                "provider": self.provider,
                "model": self.model,
                "call_id": "req_1",
                "request_id": "req_1",
                "session_id": session_id,
                "memory_namespace": memory_namespace,
            },
            consumed_tokens=7,
            input_tokens=4,
            output_tokens=3,
            last_turn_tokens=7,
            last_turn_input_tokens=4,
            last_turn_output_tokens=3,
            iteration=1,
        )

    def resume_human_input(
        self,
        *,
        conversation,
        continuation,
        response,
        callback=None,
        session_id=None,
        memory_namespace=None,
        **kwargs,
    ):
        type(self).last_resume_tool_runtime_config = kwargs.get("tool_runtime_config")
        del continuation, kwargs
        assert session_id == memory_namespace
        assert response["request_id"] == "req_1"
        assert response["selected_values"] == ["arcade"]
        if callback is not None:
            callback({"type": "tool_call", "tool_name": "write", "arguments": {"path": "app/index.html"}})
        return KernelRunResult(
            messages=conversation
            + [
                {
                    "role": "assistant",
                    "content": "Done. Created app/index.html, app/style.css, and app/game.js.",
                }
            ],
            status="completed",
            consumed_tokens=18,
            input_tokens=11,
            output_tokens=7,
            last_turn_tokens=11,
            last_turn_input_tokens=7,
            last_turn_output_tokens=4,
            iteration=2,
        )


def test_notebook_session_start_and_resume_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    test_dir = tmp_path / "tetris_test"
    (test_dir / "artifacts").mkdir(parents=True)
    (test_dir / "app").mkdir(parents=True)

    case = build_eval_case(
        case_id="tetris_case",
        title="Tetris",
        task_prompt="Build a tetris game and ask when uncertain.",
        workspace_mode="persistent_test_folder",
        workspace_source=str(test_dir),
        allowed_toolkits=["core"],
        toolkit_options={"tool_result_budget": {"max_result_chars": 1_234}},
        rule_checks={"required_tool_names": ["ask_user_question"]},
        candidate_instructions="Ask before deciding.",
    )

    started_state = start_notebook_session(
        repo_root=REPO_ROOT,
        test_dir=test_dir,
        case=case,
        model_spec={
            "provider": "openai",
            "model": "gpt-5",
            "label": "gpt-5",
            "api_key_env": "OPENAI_API_KEY",
        },
        state_path=test_dir / "artifacts" / "session_state.json",
        agent_cls=FakeInteractiveAgent,
    )

    pending = describe_pending_request(started_state)
    assert started_state["run_artifact"]["status"] == "awaiting_human_input"
    assert pending["request_id"] == "req_1"

    resumed_state = resume_notebook_session(
        state_path=test_dir / "artifacts" / "session_state.json",
        user_response={"request_id": "req_1", "selected_values": ["arcade"]},
        agent_cls=FakeInteractiveAgent,
    )

    saved = load_notebook_session(test_dir / "artifacts" / "session_state.json")
    assert resumed_state["run_artifact"]["status"] == "completed"
    assert "app/index.html" in resumed_state["run_artifact"]["final_answer"]
    assert saved["run_artifact"]["status"] == "completed"
    expected_runtime_config = {"tool_result_budget": {"max_result_chars": 1_234}}
    assert FakeInteractiveAgent.last_run_tool_runtime_config == expected_runtime_config
    assert FakeInteractiveAgent.last_resume_tool_runtime_config == expected_runtime_config
