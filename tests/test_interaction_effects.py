from __future__ import annotations


def _human_input_request():
    from unchain.interaction import HumanInputOption, HumanInputRequest

    return HumanInputRequest(
        request_id="input-1",
        kind="selector",
        title="Choose stack",
        question="Which stack?",
        selection_mode="single",
        options=[
            HumanInputOption(label="React", value="react"),
            HumanInputOption(label="Vue", value="vue", description="Use Vue."),
        ],
        allow_other=True,
        other_label="Other",
        other_placeholder="Type one",
        min_selected=1,
        max_selected=1,
    )


def test_interaction_surface_owns_human_input_effect_helpers():
    from unchain.capabilities import EmitEventOp, RequestSuspendOp
    from unchain.interaction import (
        build_human_input_requested_event,
        build_human_input_suspend_request,
    )

    request = _human_input_request()
    requested = build_human_input_requested_event(request)
    suspended = build_human_input_suspend_request(
        request,
        continuation={"type": "human_input_continuation", "request_id": "input-1"},
    )

    assert isinstance(requested, EmitEventOp)
    assert requested.type == "human_input_requested"
    assert requested.reason == "interaction.requested"
    assert requested.payload == {
        "request_id": "input-1",
        "kind": "selector",
        "title": "Choose stack",
        "question": "Which stack?",
        "selection_mode": "single",
        "options": [
            {"label": "React", "value": "react", "description": ""},
            {"label": "Vue", "value": "vue", "description": "Use Vue."},
        ],
        "allow_other": True,
        "other_label": "Other",
        "other_placeholder": "Type one",
        "min_selected": 1,
        "max_selected": 1,
    }
    assert isinstance(suspended, RequestSuspendOp)
    assert suspended.kind == "human_input"
    assert suspended.reason == "interaction.awaiting_human_input"
    assert suspended.payload == {
        "continuation": {"type": "human_input_continuation", "request_id": "input-1"},
        "request": request.to_dict(),
    }


def test_interaction_surface_builds_human_input_continuation_payload():
    from unchain.interaction import build_human_input_continuation
    from unchain.kernel.provider_replay import set_provider_replay_frame
    from unchain.kernel.state import RunState
    from unchain.schemas import ResponseFormat

    request = _human_input_request()
    state = RunState()
    state.latest_version_id = "version-1"
    state.provider_state.provider = "openai"
    state.provider_state.model = "gpt-test"
    state.provider_state.previous_response_id = "resp-1"
    state.provider_state.use_previous_response_chain = True
    state.provider_state.max_context_window_tokens = 4096
    state.session_state.session_id = "session-1"
    state.session_state.memory_namespace = "memory-1"
    state.token_state.consumed_tokens = 30
    state.token_state.input_tokens = 10
    state.token_state.output_tokens = 20
    state.token_state.last_turn_tokens = 12
    state.token_state.last_turn_input_tokens = 5
    state.token_state.last_turn_output_tokens = 7
    state.workspace_change_state = {"files": [{"path": "app.py"}]}
    set_provider_replay_frame(
        state,
        {
            "format": "openai.responses.v1",
            "complete": False,
            "items": [{"role": "user", "content": "pick one"}],
            "incomplete_reason": "external remote prefix",
        },
    )

    continuation = build_human_input_continuation(
        request=request,
        payload={"ok": True, "nested": {"choice": "react"}, "drop": object(), "items": ("a", 1)},
        response_format=ResponseFormat(
            "selection",
            {"type": "object", "required": ["choice"]},
        ),
        next_iteration=3,
        max_iterations=8,
        state=state,
        run_id="run-1",
    )
    state.workspace_change_state["files"][0]["path"] = "mutated.py"

    assert continuation["type"] == "human_input_continuation"
    assert continuation["run_id"] == "run-1"
    assert continuation["provider"] == "openai"
    assert continuation["model"] == "gpt-test"
    assert continuation["request_id"] == "input-1"
    assert continuation["payload"] == {"ok": True, "nested": {"choice": "react"}, "items": ["a", 1]}
    assert continuation["response_format"] == {
        "name": "selection",
        "schema": {"type": "object", "required": ["choice"]},
        "required": ["choice"],
    }
    assert continuation["context_version_id"] == "version-1"
    assert continuation["iteration"] == 3
    assert continuation["max_iterations"] == 8
    assert continuation["previous_response_id"] == "resp-1"
    assert continuation["use_openai_previous_response_chain"] is True
    assert "provider_replay_frame" not in continuation
    assert continuation["provider_replay_handle"]["id"].startswith(
        "provider_replay_"
    )
    assert continuation["session_id"] == "session-1"
    assert continuation["memory_namespace"] == "memory-1"
    assert continuation["max_context_window_tokens"] == 4096
    assert continuation["consumed_tokens"] == 30
    assert continuation["input_tokens"] == 10
    assert continuation["output_tokens"] == 20
    assert continuation["last_turn_tokens"] == 12
    assert continuation["last_turn_input_tokens"] == 5
    assert continuation["last_turn_output_tokens"] == 7
    assert continuation["workspace_change_state"] == {"files": [{"path": "app.py"}]}


def test_human_input_continuation_hides_provider_replay_secrets():
    import json
    import pickle
    from dataclasses import asdict

    from unchain.interaction import (
        build_human_input_continuation,
        prepare_human_input_resume_plan,
    )
    from unchain.kernel.provider_replay import set_provider_replay_frame
    from unchain.kernel.results import build_kernel_run_result
    from unchain.kernel.state import RunState

    state = RunState()
    conversation = [{"role": "user", "content": "pick one"}]
    state.seed_messages(conversation)
    state.provider_state.provider = "anthropic"
    set_provider_replay_frame(
        state,
        {
            "format": "anthropic.messages.v1",
            "complete": True,
            "items": [
                conversation[0],
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "PRIVATE HIDDEN CHAIN OF THOUGHT",
                            "signature": "opaque-signature",
                        },
                        {"type": "text", "text": "Choose."},
                    ],
                },
            ],
        },
    )

    continuation = build_human_input_continuation(
        request=_human_input_request(),
        payload={},
        response_format=None,
        next_iteration=1,
        max_iterations=4,
        state=state,
        run_id="private-replay",
    )

    serialized = json.dumps(continuation)
    assert "PRIVATE HIDDEN CHAIN OF THOUGHT" not in serialized
    assert "opaque-signature" not in serialized
    assert "provider_replay_frame" not in continuation
    assert continuation["provider_replay_handle"]["id"].startswith(
        "provider_replay_"
    )

    plan = prepare_human_input_resume_plan(
        conversation=conversation,
        continuation=continuation,
    )
    assert plan.provider_replay_frame is not None
    assert plan.provider_replay_frame["items"][1]["content"][0]["thinking"] == (
        "PRIVATE HIDDEN CHAIN OF THOUGHT"
    )

    result_payload = asdict(build_kernel_run_result(state, status="completed"))
    serialized_result = json.dumps(result_payload)
    assert "PRIVATE HIDDEN CHAIN OF THOUGHT" not in serialized_result
    assert "opaque-signature" not in serialized_result
    assert result_payload["provider_replay_handle"]["scope"] == "process"
    pickled_handle = pickle.dumps(continuation["provider_replay_handle"])
    assert b"PRIVATE HIDDEN CHAIN OF THOUGHT" not in pickled_handle
    assert b"opaque-signature" not in pickled_handle


def test_interaction_surface_prepares_and_hydrates_human_input_resume_plan():
    from unchain.interaction import hydrate_human_input_resume_state, prepare_human_input_resume_plan
    from unchain.kernel.provider_replay import current_provider_replay_frame
    from unchain.kernel.state import RunState

    conversation = [{"role": "user", "content": "pick one"}]
    continuation = {
        "type": "human_input_continuation",
        "provider": "openai",
        "model": "gpt-test",
        "run_id": "run-1",
        "session_id": "session-1",
        "memory_namespace": "memory-1",
        "payload": {"topic": "ui"},
        "response_format": {
            "name": "selection",
            "schema": {"type": "object", "required": ["choice"]},
            "required": ["choice"],
        },
        "iteration": 4,
        "max_iterations": 9,
        "previous_response_id": "resp-1",
        "use_openai_previous_response_chain": True,
        "provider_replay_frame": {
            "format": "openai.responses.v1",
            "complete": False,
            "items": [{"role": "user", "content": "pick one"}],
            "incomplete_reason": "external remote prefix",
        },
        "max_context_window_tokens": 4096,
        "consumed_tokens": 30,
        "input_tokens": 10,
        "output_tokens": 20,
        "last_turn_tokens": 12,
        "last_turn_input_tokens": 5,
        "last_turn_output_tokens": 7,
        "workspace_change_state": {"files": [{"path": "app.py"}]},
    }

    plan = prepare_human_input_resume_plan(
        conversation=conversation,
        continuation=continuation,
        payload=None,
        response_format=None,
        fallback_provider="anthropic",
        fallback_model="claude-test",
        session_id="session-1",
        memory_namespace=None,
        run_id=None,
        run_id_factory=lambda: "new-run",
    )
    state = RunState()
    state.seed_messages(conversation)
    hydrate_human_input_resume_state(state, plan)
    continuation["workspace_change_state"]["files"][0]["path"] = "mutated.py"

    assert plan.provider == "openai"
    assert plan.model == "gpt-test"
    assert plan.session_id == "session-1"
    assert plan.memory_namespace == "memory-1"
    assert plan.run_id == "run-1"
    assert plan.payload == {"topic": "ui"}
    assert plan.response_format is not None
    assert plan.response_format.name == "selection"
    assert plan.response_format.required == ["choice"]
    assert plan.max_iterations == 9
    assert plan.max_context_window_tokens == 4096
    assert state.provider_state.provider == "openai"
    assert state.provider_state.model == "gpt-test"
    assert state.provider_state.max_context_window_tokens == 4096
    assert state.session_state.session_id == "session-1"
    assert state.session_state.memory_namespace == "memory-1"
    assert state.iteration == 4
    assert state.provider_state.previous_response_id == "resp-1"
    assert state.provider_state.use_previous_response_chain is True
    assert current_provider_replay_frame(state) == {
        "format": "openai.responses.v1",
        "complete": False,
        "items": conversation,
        "incomplete_reason": "external remote prefix",
    }
    assert state.run_status == "running"
    assert state.token_state.consumed_tokens == 30
    assert state.token_state.input_tokens == 10
    assert state.token_state.output_tokens == 20
    assert state.token_state.last_turn_tokens == 12
    assert state.token_state.last_turn_input_tokens == 5
    assert state.token_state.last_turn_output_tokens == 7
    assert state.workspace_change_state == {"files": [{"path": "app.py"}]}
    assert state.component_bucket("workspace_changes")["state"] == {"files": [{"path": "app.py"}]}


def test_human_input_resume_rejects_legacy_remote_continuation_without_replay():
    import pytest

    from unchain.interaction import (
        hydrate_human_input_resume_state,
        prepare_human_input_resume_plan,
    )
    from unchain.kernel.state import RunState
    from unchain.providers.context_assembler import ProviderContextProjectionError
    from unchain.providers.model_turn_runtime import build_model_turn_request

    conversation = [{"role": "user", "content": "pick one"}]
    plan = prepare_human_input_resume_plan(
        conversation=conversation,
        continuation={
            "type": "human_input_continuation",
            "provider": "openai",
            "model": "gpt-test",
            "previous_response_id": "resp-legacy",
            "use_openai_previous_response_chain": True,
        },
    )
    state = RunState()
    state.seed_messages(conversation)

    hydrate_human_input_resume_state(state, plan)

    with pytest.raises(ProviderContextProjectionError, match="handle is unavailable"):
        build_model_turn_request(state)


def test_interaction_resume_plan_errors_do_not_name_kernel_loop():
    import pytest
    from unchain.interaction import prepare_human_input_resume_plan

    with pytest.raises(NotImplementedError) as exc_info:
        prepare_human_input_resume_plan(
            conversation=[],
            continuation={"provider": "unsupported-provider"},
        )

    message = str(exc_info.value)
    assert "human input resume" in message
    assert "KernelLoop" not in message


def test_interaction_resume_plan_generates_run_id_without_kernel_factory():
    from unchain.interaction import prepare_human_input_resume_plan

    plan = prepare_human_input_resume_plan(
        conversation=[],
        continuation={"provider": "openai"},
    )

    assert isinstance(plan.run_id, str)
    assert plan.run_id


def test_human_input_resume_harness_lives_under_interaction_boundary():
    from unchain.interaction import HumanInputResumeHarness
    from unchain.tools import HumanInputResumeHarness as LegacyHumanInputResumeHarness

    assert HumanInputResumeHarness is LegacyHumanInputResumeHarness
    assert HumanInputResumeHarness.__module__ == "unchain.interaction.resume"
