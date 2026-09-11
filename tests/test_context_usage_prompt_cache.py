"""Runtime telemetry must not invalidate instructions or become conversation history."""
import copy
import json

import pytest

from unchain.kernel import KernelLoop
from unchain.optimizers import ContextUsageOptimizer, ContextUsageOptimizerConfig
from unchain.optimizers.common import latest_user_query
from unchain.providers.model_turn_runtime import build_model_turn_request


PROVIDERS = ["openai", "anthropic", "hyperspace", "ollama", "gemini"]


def _state(provider="gemini"):
    loop = KernelLoop(harnesses=[ContextUsageOptimizer()])
    messages = [
        {"role": "system", "content": "Stable instructions"},
        {"role": "user", "content": "Reference material. " * 1000},
        {"role": "assistant", "content": "Ready"},
        {"role": "user", "content": "Summarize the reference"},
    ]
    state = loop.seed_state(messages)
    state.provider_state.provider = provider
    state.provider_state.max_context_window_tokens = 100000
    return loop, state, messages


@pytest.mark.parametrize("provider", PROVIDERS)
def test_status_is_request_only_suffix_and_never_a_new_user_query(provider):
    loop, state, messages = _state(provider)
    version = state.latest_version_id
    loop.dispatch_phase(state, phase="before_model")
    request = build_model_turn_request(state)
    assert state.latest_messages() == messages
    assert state.transcript == messages
    assert state.latest_version_id == version
    assert latest_user_query(state.latest_messages()) == "Summarize the reference"
    assert request.messages[:-1] == messages
    assert set(request.messages[-1]) == {"role", "content"}
    assert request.messages[-1]["role"] == "user"
    assert "[Context Status]" in request.messages[-1]["content"]
    assert "Runtime context" in request.messages[-1]["content"]
    assert build_model_turn_request(state).messages == request.messages


def test_repeated_optimization_does_not_count_or_accumulate_its_own_note():
    loop, state, messages = _state()
    loop.dispatch_phase(state, phase="before_model")
    first = build_model_turn_request(state).messages
    first_tokens = state.optimizer_state["context_usage"]["current_tokens"]
    loop.dispatch_phase(state, phase="before_model")
    assert build_model_turn_request(state).messages == first
    assert state.optimizer_state["context_usage"]["current_tokens"] == first_tokens
    state.iteration += 1
    updated = messages + [{"role": "assistant", "content": "Answer"}, {"role": "user", "content": "Explain more"}]
    state.rebuild_working_version(updated)
    loop.dispatch_phase(state, phase="before_model")
    second = build_model_turn_request(state).messages
    assert second[:-1] == updated
    assert second[:len(messages)] == first[:len(messages)]
    assert second[-1] != first[-1]


def test_user_authored_status_like_instructions_are_preserved():
    loop, state, messages = _state()
    messages.insert(0, {"role": "system", "content": "[Context Status] This is my instruction, keep it."})
    state.rebuild_working_version(messages)
    loop.dispatch_phase(state, phase="before_model")
    assert build_model_turn_request(state).messages[:-1] == messages


@pytest.mark.parametrize("disable", ["config", "window"])
def test_disabling_status_clears_previously_generated_note(disable):
    loop, state, messages = _state()
    loop.dispatch_phase(state, phase="before_model")
    if disable == "config":
        optimizer = ContextUsageOptimizer(ContextUsageOptimizerConfig(enabled=False))
        from unchain.kernel.harness import HarnessContext
        delta = optimizer.build_delta(HarnessContext(state, "before_model"))
        if delta is not None:
            state.apply_delta(delta)
    else:
        state.provider_state.max_context_window_tokens = 0
        loop.dispatch_phase(state, phase="before_model")
    assert build_model_turn_request(state).messages == messages


@pytest.mark.parametrize("change", ["version", "iteration"])
def test_stale_note_is_not_reused_for_another_context(change):
    loop, state, messages = _state()
    loop.dispatch_phase(state, phase="before_model")
    if change == "version":
        state.rebuild_working_version(messages)
    else:
        state.iteration += 1
    assert build_model_turn_request(state).messages == messages


@pytest.mark.parametrize("mutation", ["unknown", "version", "boolean_count"])
def test_request_note_contract_rejects_invalid_records(mutation):
    loop, state, _ = _state()
    loop.dispatch_phase(state, phase="before_model")
    record = state.optimizer_state["context_usage"]["request_note"]
    if mutation == "unknown":
        record["unexpected"] = True
    elif mutation == "version":
        record["schema_version"] = 2
    else:
        record["current_tokens"] = True
    with pytest.raises(ValueError, match="context usage"):
        build_model_turn_request(state)


def _wire(provider, request):
    from tests.test_provider_wire_preparer import _prepared_turn_for_request, _ModelIO, ATTEMPT
    from unchain.providers.wire_preparer import build_prepared_provider_request_payload, prepare_provider_wire
    from unchain.providers.wire_envelope import ProviderWireEnvelope

    model_io = _ModelIO(provider)
    payload = build_prepared_provider_request_payload(
        provider=provider, messages=request.messages, effective_payload={},
        request_model=model_io.model, response_format={"kind": "none", "value": None},
        previous_response_id=request.previous_response_id,
        fallback_messages=request.fallback_messages, context_mode=request.context_mode,
    )
    prepared, draft = _prepared_turn_for_request(model_io=model_io, request_payload=payload)
    envelope = prepare_provider_wire(
        prepared, model_io=model_io, attempt=ATTEMPT, iteration=7,
        transport_target_sha256="a" * 64,
    )
    # Production strict wire validation, not a permissive mock consumer.
    restored = ProviderWireEnvelope.from_dict(json.loads(json.dumps(envelope.to_dict())))
    assert restored == envelope
    assert restored.verify_against_catalog(draft.catalog) is restored
    wire = restored.request_copy()
    if provider == "gemini":
        from google.genai import types
        for content in wire["contents"]:
            types.Content.model_validate(content)
        types.GenerateContentConfig.model_validate(wire["config"])
    return wire


@pytest.mark.parametrize("provider", PROVIDERS)
def test_final_wire_changes_only_status_suffix_when_usage_changes(provider):
    loop, state, _ = _state(provider)
    loop.dispatch_phase(state, phase="before_model")
    first = _wire(provider, build_model_turn_request(state))
    state.provider_state.max_context_window_tokens *= 2
    loop.dispatch_phase(state, phase="before_model")
    second = _wire(provider, build_model_turn_request(state))
    assert first != second
    if provider == "gemini":
        assert "[Context Status]" not in first["config"]["system_instruction"]
        first_text = first["contents"][-1]["parts"][-1].pop("text")
        second_text = second["contents"][-1]["parts"][-1].pop("text")
    else:
        key = "input" if provider == "openai" else "messages"
        first_tail, second_tail = first[key].pop(), second[key].pop()
        if provider in {"anthropic", "hyperspace"}:
            assert "[Context Status]" not in json.dumps(first["system"])
            first_text = first_tail["content"][0]["text"]
            second_text = second_tail["content"][0]["text"]
        else:
            first_text, second_text = first_tail["content"], second_tail["content"]
    assert "[Context Status]" in first_text and "[Context Status]" in second_text
    assert first == second


@pytest.mark.parametrize("provider", PROVIDERS)
def test_status_follows_complete_tool_results_without_mutating_them(provider):
    loop, state, messages = _state(provider)
    if provider == "openai":
        tool_turn = [
            {"type": "function_call", "call_id": "c1", "name": "search", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "found"},
        ]
    elif provider in {"anthropic", "hyperspace"}:
        tool_turn = [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "search", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "found"}]},
        ]
    elif provider == "gemini":
        tool_turn = [
            {"role": "model", "parts": [{"function_call": {"id": "c1", "name": "search", "args": {}}, "thought_signature": "c2ln"}]},
            {"role": "user", "parts": [{"function_response": {"id": "c1", "name": "search", "response": {"result": "found"}}}]},
        ]
    else:
        tool_turn = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "found"},
        ]
    state.rebuild_working_version(messages + tool_turn)
    loop.dispatch_phase(state, phase="before_model")
    request = build_model_turn_request(state)
    assert request.messages[-3:-1] == tool_turn
    assert state.latest_messages() == messages + tool_turn
    wire = _wire(provider, request)
    if provider == "gemini":
        assert wire["contents"][-1]["parts"][0] == tool_turn[-1]["parts"][0]
        assert wire["contents"][-2]["parts"] == tool_turn[-2]["parts"]
    else:
        key = "input" if provider == "openai" else "messages"
        assert wire[key][-3:-1] == tool_turn


def test_remote_continuation_and_fallback_receive_same_note():
    from unchain.kernel.provider_replay import set_provider_replay_frame
    loop, state, _ = _state("openai")
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    state.rebuild_working_version(messages)
    set_provider_replay_frame(state, {
        "format": "openai.responses.v1", "complete": True, "items": messages,
    })
    state.provider_state.use_previous_response_chain = True
    state.provider_state.previous_response_id = "response-1"
    state.remote_continuation_input = [{"role": "user", "content": "continue"}]
    loop.dispatch_phase(state, phase="before_model")
    request = build_model_turn_request(state)
    assert request.previous_response_id == "response-1"
    assert request.messages[0] == state.remote_continuation_input[0]
    assert request.messages[-1] == request.fallback_messages[-1]
    assert request.fallback_messages[:-1] == messages


def test_old_checkpoint_statistics_do_not_inject_or_delete_messages():
    _, state, messages = _state()
    state.optimizer_state["context_usage"] = {"current_tokens": 100, "max_tokens": 100000}
    assert build_model_turn_request(state).messages == messages


def test_request_is_independent_of_later_optimizer_updates():
    loop, state, _ = _state()
    loop.dispatch_phase(state, phase="before_model")
    request = build_model_turn_request(state)
    frozen_messages = copy.deepcopy(request.messages)
    frozen_wire = _wire("gemini", request)
    state.provider_state.max_context_window_tokens *= 2
    loop.dispatch_phase(state, phase="before_model")
    assert request.messages == frozen_messages
    assert _wire("gemini", request) == frozen_wire


@pytest.mark.parametrize("provider", PROVIDERS)
def test_checkpoint_reopen_refreshes_note_without_replaying_old_telemetry(provider, tmp_path):
    from unchain.kernel.provider_replay import set_provider_replay_frame
    from unchain.memory.checkpoint_state import (
        build_execution_checkpoint, validate_execution_checkpoint,
        restore_fresh_checkpoint_messages,
    )
    loop, state, messages = _state(provider)
    state.session_state.session_id = "context-status-session"
    loop.dispatch_phase(state, phase="before_model")
    request = build_model_turn_request(state)
    # Replay can retain the actual old wire suffix. Semantic context must not.
    replay_items = copy.deepcopy(request.messages)
    if provider == "gemini":
        from unchain.providers.gemini import translate_gemini_messages
        replay_items, _ = translate_gemini_messages(replay_items)
    formats = {"openai": "openai.responses.v1", "anthropic": "anthropic.messages.v1",
               "hyperspace": "anthropic.messages.v1", "ollama": "ollama.chat.v1", "gemini": "gemini.contents.v1"}
    set_provider_replay_frame(state, {"format": formats[provider], "complete": True, "items": replay_items})
    checkpoint = build_execution_checkpoint(state, status="max_iterations", run_id="status-run")
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(checkpoint))
    restored = validate_execution_checkpoint(json.loads(path.read_text()))
    restored_messages = restore_fresh_checkpoint_messages(restored, incoming_messages=[])
    assert restored_messages == messages
    restarted = loop.seed_state(restored_messages)
    restarted.provider_state.provider = provider
    restarted.provider_state.max_context_window_tokens = 200000
    set_provider_replay_frame(restarted, restored["replay_frame"])
    loop.dispatch_phase(restarted, phase="before_model")
    refreshed = build_model_turn_request(restarted)
    assert refreshed.messages[:-1] == messages
    assert refreshed.messages[-1] != request.messages[-1]
    assert restarted.transcript == messages


def test_empty_context_does_not_turn_telemetry_into_a_task():
    loop, state, _ = _state()
    state.seed_messages([])
    loop.dispatch_phase(state, phase="before_model")
    assert build_model_turn_request(state).messages == []
