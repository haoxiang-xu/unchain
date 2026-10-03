"""Ticket 390 lifecycle coverage with real durable context boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest

from test_context_provider_turn_cross_provider import (
    _run_mixed_anthropic_compatible_tool_turn,
    _runtime,
)

from unchain.context import (
    ContextCompileRequest,
    ContextCompiler,
    estimate_context_tokens,
    resolve_context_budget,
)
from unchain.context.compiler import (
    ContextBudgetExceededError,
    _CurrentProviderTurnProjection,
)
from unchain.context.coordinator import (
    ContextCompileCoordinator,
    ContextCompileCoordinatorError,
)
from unchain.execution import ExecutionRuntime
from unchain.journal import ResourceRef, SemanticEventDraft
from unchain.kernel.loop import KernelLoop
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.providers import (
    AnthropicModelIO,
    GeminiModelIO,
    HyperspaceModelIO,
    OpenAIModelIO,
)
from unchain.runtime import build_runtime_loop
from unchain.retry import RetryConfig
from unchain.tools import Toolkit

from test_ticket390_signed_gemini import _run_signed_gemini_mixed_tool_turn


def _anthropic_tool_model(send_calls, *, tool_message=None):
    tool_message = tool_message or [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "Inspect the workspace first."},
        {
            "type": "tool_use",
            "id": "phase2-anthropic-call",
            "name": "probe",
            "input": {"query": "durable"},
        },
    ]

    class _Stream:
        def __init__(self, content):
            self.content = content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return SimpleNamespace(id=f"phase2-anthropic-{len(send_calls)}", content=self.content)

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream(tool_message if len(send_calls) == 1 else [
                {"type": "text", "text": "anthropic complete"}
            ])

    class _Client:
        messages = _Messages()

    return AnthropicModelIO(
        model="phase2-anthropic", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _openai_final_model(send_calls, text="openai complete"):
    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="phase2-openai-final",
                    output=[{
                        "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }],
                    usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                ),
            )

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        responses = _Responses()

    return OpenAIModelIO(
        model="phase2-openai", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _openai_tool_model(send_calls):
    class _Stream:
        def __init__(self, send_number):
            self._send_number = send_number

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            output = (
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "OpenAI is checking the workspace.",
                            }
                        ],
                    },
                    {
                        "type": "function_call",
                        "call_id": "phase2-openai-call",
                        "name": "probe",
                        "arguments": '{"query":"switch"}',
                    },
                ]
                if self._send_number == 1
                else [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "openai complete"}
                        ],
                    }
                ]
            )
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id=f"phase2-openai-{self._send_number}", output=output,
                    usage={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                ),
            )

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream(len(send_calls))

    class _Client:
        responses = _Responses()

    return OpenAIModelIO(
        model="phase2-openai", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _anthropic_final_model(send_calls, text="anthropic complete"):
    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return SimpleNamespace(
                id="phase2-anthropic-final",
                content=[{"type": "text", "text": text}],
            )

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        messages = _Messages()

    return AnthropicModelIO(
        model="phase2-anthropic", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _hyperspace_final_model(send_calls, text="hyperspace complete"):
    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return SimpleNamespace(id="phase2-hyperspace-final",
                                   content=[{"type": "text", "text": text}])

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        messages = _Messages()

    return HyperspaceModelIO(
        model="phase2-hyperspace", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _gemini_final_model(send_calls, text="gemini complete"):
    from google.genai import types

    response = types.GenerateContentResponse.model_validate({
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]},
                       "finish_reason": "STOP"}]
    })

    class _Models:
        def generate_content_stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return iter([response])

    class _Client:
        models = _Models()

        def close(self):
            return None

    return GeminiModelIO(
        model="phase2-gemini", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _toolkit(invocations, *, requires_confirmation=False):
    toolkit = Toolkit()

    def probe(query=""):
        invocations.append(query)
        return {"query": query, "status": "complete"}

    toolkit.register(probe, name="probe", requires_confirmation=requires_confirmation)
    return toolkit


def _compiled_current_anthropic_exchange(*, assistant_messages):
    call_specs = (
        ("phase2-accounting-1", {"query": "first"}),
        ("phase2-accounting-2", {"query": "second"}),
    )
    events = []
    pending_inputs = []
    for index, (call_id, arguments) in enumerate(call_specs, start=1):
        result = {"query": arguments["query"], "status": "complete"}
        encoded = json.dumps(
            result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        ref = ResourceRef("artifact", f"phase2-accounting-{index}", 1)
        events.extend((
            {
                "type": "tool_call", "event_id": f"phase2-call-{index}",
                "store_seq": index, "iteration": 1, "call_id": call_id,
                "tool_name": "probe", "arguments": arguments,
                "source_provider": "anthropic",
                "execution_id": "phase2-execution",
                "generation_id": "phase2-generation",
            },
            {
                "type": "tool_result", "event_id": f"phase2-result-{index}",
                "store_seq": index + 10, "iteration": 1, "call_id": call_id,
                "tool_name": "probe", "result": result,
                "full_output_ref": ref.to_dict(), "result_bytes": len(encoded),
                "result_sha256": hashlib.sha256(encoded).hexdigest(),
                "execution_id": "phase2-execution",
                "generation_id": "phase2-generation",
            },
        ))
        pending_inputs.append(
            {
                "event_id": f"phase2-result-{index}", "store_seq": index + 10,
                "type": "tool_result", "preview": "complete",
                "preview_truncated": False, "content_ref": ref.to_dict(),
                "content_bytes": len(encoded),
                "content_sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    request = ContextCompileRequest(
        case="phase2-current-native-accounting",
        source_messages=({"role": "user", "content": "inspect both"},),
        semantic_events=tuple(events), pending_task_inputs=tuple(pending_inputs),
        budget=resolve_context_budget(context_window_tokens=16_384),
        provider="anthropic", model="phase2-anthropic", build_id="phase2-build",
        execution_id="phase2-execution", generation_id="phase2-generation",
        attempt_id="phase2-attempt",
    )
    projection = _CurrentProviderTurnProjection(
        subject=None, subject_sha256="", result_cursor=None, result_sha256="",
        replay_frame_sha256=None, assistant_messages=tuple(assistant_messages),
        call_ids=tuple(call_id for call_id, _arguments in call_specs),
    )
    return ContextCompiler()._compile_for_coordinator(
        request, current_provider_turn=projection
    ).result


def _run_cold_resume_process(root, phase, provider="anthropic"):
    repository_root = Path(__file__).resolve().parents[2]
    fixture = repository_root / "tests/context_v2/fixtures/ticket_390/cold_resume_process.py"
    environment = os.environ.copy()
    pythonpath = [
        str(repository_root / "src"),
        str(repository_root / "tests/context_v2"),
    ]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    completed = subprocess.run(
        [sys.executable, str(fixture), phase, provider, str(root)],
        cwd=repository_root, env=environment, check=True,
        capture_output=True, text=True,
    )
    return json.loads(completed.stdout)


def test_cold_resume_replays_mixed_anthropic_turn_from_file_backed_state(tmp_path):
    session_id = "phase2-cold-anthropic"
    session_directory = tmp_path / "sessions"
    sends, invocations = [], []
    first_runtime = _runtime(tmp_path)
    first_store = JsonFileSessionStore(session_directory)
    first_loop = build_runtime_loop(
        harnesses=list(first_runtime.build_harnesses()),
        model_io=_anthropic_tool_model(sends),
        memory_runtime=KernelMemoryRuntime.from_config(
            store=first_store
        ),
        execution_runtime=ExecutionRuntime(first_store),
        semantic_context_owner=first_runtime.owner_id,
    )

    suspended = first_loop.run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=first_runtime.compose_event_callback(None),
        session_id=session_id, provider="anthropic", model="phase2-anthropic",
        toolkit=_toolkit(invocations, requires_confirmation=True),
        run_id="phase2-cold-attempt", max_iterations=2,
    )

    assert suspended.status == "awaiting_interaction"
    assert len(sends) == 1
    assert invocations == []

    restarted_runtime = _runtime(tmp_path)
    restarted_store = JsonFileSessionStore(session_directory)
    restarted_loop = build_runtime_loop(
        harnesses=list(restarted_runtime.build_harnesses()),
        model_io=_anthropic_tool_model(sends),
        memory_runtime=KernelMemoryRuntime.from_config(
            store=restarted_store
        ),
        execution_runtime=ExecutionRuntime(restarted_store),
        semantic_context_owner=restarted_runtime.owner_id,
    )
    resumed = restarted_loop.resume_interaction(
        session_id=session_id, response={"approved": True},
        callback=restarted_runtime.compose_event_callback(None),
        toolkit=_toolkit(invocations, requires_confirmation=True),
    )

    assert resumed.status == "completed"
    assert invocations == ["durable"]
    assert len(sends) == 2
    replayed = next(
        message["content"] for message in sends[1]["messages"]
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    assert replayed == [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "Inspect the workspace first."},
        {
            "type": "tool_use", "id": "phase2-anthropic-call", "name": "probe",
            "input": {"query": "durable"},
        },
    ]


def test_cold_resume_projects_legacy_anthropic_sdk_text_helper_only_on_wire(tmp_path):
    session_id = "phase2-cold-anthropic-legacy-sdk-helper"
    session_directory = tmp_path / "sessions"
    sends, invocations = [], []
    first_runtime = _runtime(tmp_path)
    first_store = JsonFileSessionStore(session_directory)
    source_model = _anthropic_tool_model(sends)
    original_fetch_stream = source_model._fetch_turn_streaming

    def fetch_legacy_result(client, request, request_kwargs):
        result = original_fetch_stream(client, request, request_kwargs)
        if result.tool_calls:
            result.assistant_messages[0]["content"][0]["parsed_output"] = {
                "legacy": "sdk-only"
            }
            result.provider_replay_frame["items"][-1]["content"][1][
                "parsed_output"
            ] = {"legacy": "sdk-only"}
        return result

    source_model._fetch_turn_streaming = fetch_legacy_result
    first_loop = build_runtime_loop(
        harnesses=list(first_runtime.build_harnesses()),
        model_io=source_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=first_store),
        execution_runtime=ExecutionRuntime(first_store),
        semantic_context_owner=first_runtime.owner_id,
    )

    suspended = first_loop.run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=first_runtime.compose_event_callback(None),
        session_id=session_id, provider="anthropic", model="phase2-anthropic",
        toolkit=_toolkit(invocations, requires_confirmation=True),
        run_id="phase2-cold-legacy-sdk-helper", max_iterations=2,
    )

    assert suspended.status == "awaiting_interaction"
    checkpoint = first_store.load(session_id)["execution_checkpoint"]
    assert checkpoint["replay_frame"]["items"][-1]["content"][1]["parsed_output"] == {
        "legacy": "sdk-only"
    }

    restarted_runtime = _runtime(tmp_path)
    restarted_store = JsonFileSessionStore(session_directory)
    restarted_loop = build_runtime_loop(
        harnesses=list(restarted_runtime.build_harnesses()),
        model_io=_anthropic_tool_model(sends),
        memory_runtime=KernelMemoryRuntime.from_config(store=restarted_store),
        execution_runtime=ExecutionRuntime(restarted_store),
        semantic_context_owner=restarted_runtime.owner_id,
    )
    resumed = restarted_loop.resume_interaction(
        session_id=session_id, response={"approved": True},
        callback=restarted_runtime.compose_event_callback(None),
        toolkit=_toolkit(invocations, requires_confirmation=True),
    )

    assert resumed.status == "completed"
    assert invocations == ["durable"]
    assert len(sends) == 2
    replayed = next(
        message["content"] for message in sends[1]["messages"]
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    assert replayed == [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "Inspect the workspace first."},
        {
            "type": "tool_use", "id": "phase2-anthropic-call", "name": "probe",
            "input": {"query": "durable"},
        },
    ]


def test_fresh_process_recovers_file_backed_native_tool_turn(tmp_path):
    for provider in ("anthropic", "hyperspace", "gemini"):
        provider_root = tmp_path / provider
        seeded = _run_cold_resume_process(provider_root, "seed", provider)
        resumed = _run_cold_resume_process(provider_root, "resume", provider)

        assert seeded == {
            "invocations": [], "send_count": 1,
            "status": "awaiting_interaction", "replayed_assistant": None,
            "second_message_status": None,
            "second_message_has_source_native_call": None,
            "second_message_history": None,
        }
        assert resumed["invocations"] == ["process"]
        assert resumed["send_count"] == 2
        assert resumed["status"] == "completed"
        assert resumed["second_message_status"] == "completed"
        replayed = resumed["replayed_assistant"]
        assert replayed is not None
        assert any(item["text"] == "Inspect the durable workspace." for item in replayed)
        call = next(item for item in replayed if item.get("call_name") == "probe")
        assert call["call_id"] == f"{provider}-process-cold-call"
        assert call["arguments"] == {"query": "process"}
        assert any(item["signature"] for item in replayed)
        assert resumed["second_message_has_source_native_call"] is False
        assert resumed["second_message_history"].count("process cold resume complete") == 1


@pytest.mark.parametrize("provider", ["anthropic", "hyperspace", "gemini"])
def test_mixed_native_continuation_retries_the_same_durable_request(
    tmp_path, provider
):
    repositories, sends, invocations = [], [], []
    runtime = _runtime(tmp_path, repository_observer=repositories.append)
    retry = RetryConfig(
        max_retries=1, base_delay_ms=0, max_delay_ms=0, jitter_ratio=0
    )
    if provider == "gemini":
        result, _, _ = _run_signed_gemini_mixed_tool_turn(
            tmp_path,
            runtime=runtime,
            send_calls=sends,
            tool_calls=invocations,
            transient_second_send=True,
            retry_config=retry,
        )
    else:
        result = _run_mixed_anthropic_compatible_tool_turn(
            runtime,
            provider=provider,
            send_calls=sends,
            tool_calls=invocations,
            transient_second_send=True,
            retry_config=retry,
        )

    assert result.status == "completed"
    assert invocations == ["durable"]
    assert len(sends) == 3
    assert sends[1] == sends[2]
    receipts = repositories[0].load_receipts(
        root_run_id=(
            "signed-gemini-attempt"
            if provider == "gemini"
            else "attempt-mixed-anthropic"
        ),
        owner_run_id=(
            "signed-gemini-attempt"
            if provider == "gemini"
            else "attempt-mixed-anthropic"
        ),
        attempt_id=(
            "signed-gemini-attempt"
            if provider == "gemini"
            else "attempt-mixed-anthropic"
        ),
    )
    assert sorted(
        (receipt.identity.iteration, receipt.identity.retry_ordinal)
        for receipt in receipts
    ) == [(0, 0), (1, 0), (1, 1)]
    if provider == "gemini":
        replayed = next(
            content for content in sends[1]["contents"]
            if content.role == "model"
            and any(part.function_call for part in content.parts)
        )
        assert [part.text for part in replayed.parts if part.text] == [
            "Inspect the workspace first."
        ]
        assert [part.function_call.id for part in replayed.parts
                if part.function_call] == ["gemini-signed-call"]
        assert replayed.parts[1].thought_signature == b"ticket390-synthetic-signature"
        result_ids = [
            part.function_response.id
            for content in sends[1]["contents"]
            for part in content.parts
            if part.function_response
        ]
        assert result_ids == ["gemini-signed-call"]
    else:
        assistant = next(
            message for message in sends[1]["messages"]
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), list)
            and any(block.get("type") == "tool_use" for block in message["content"])
        )
        assert [block["text"] for block in assistant["content"]
                if block.get("type") == "text"] == [
                    "I will inspect the workspace first."
                ]
        assert [block["id"] for block in assistant["content"]
                if block.get("type") == "tool_use"] == ["mixed-call-1"]
        result_ids = [
            block["tool_use_id"]
            for message in sends[1]["messages"]
            for block in message.get("content", [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert result_ids == ["mixed-call-1"]


@pytest.mark.parametrize("provider", ["anthropic", "hyperspace", "gemini"])
@pytest.mark.parametrize("terminal_type", ["run_failed", "run_cancelled"])
def test_mixed_native_terminal_attempt_never_sends_next_request(
    tmp_path, provider, terminal_type
):
    repositories, sends, invocations, terminal_events = [], [], [], []
    base_runtime = _runtime(tmp_path, repository_observer=repositories.append)

    def append_terminal_after_tool_result(event):
        if event.get("type") != "tool_result" or terminal_events:
            return
        journal = repositories[0]
        result_event = next(
            item for item in journal.capture_snapshot().events
            if item.event_type == "tool_result"
        )
        draft = SemanticEventDraft(
            event_id=f"ticket390-{provider}-{terminal_type}-before-continuation",
            event_type=terminal_type,
            attempt=result_event.attempt,
            operation_id=f"ticket390-{provider}-{terminal_type}-terminal-op",
            payload={
                "run_id": result_event.attempt.attempt_id,
                "status": "cancelled" if terminal_type == "run_cancelled" else "failed",
                "error_code": "synthetic_terminal_before_continuation",
            },
            resource_refs=(),
        )
        journal.append(request=draft.to_append_request())
        terminal_events.append(draft.event_id)

    class RuntimeWithTerminalEvent:
        def build_harnesses(self):
            return base_runtime.build_harnesses()

        def compose_event_callback(self, _callback):
            return base_runtime.compose_event_callback(
                append_terminal_after_tool_result
            )

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="terminal attempt cannot authorize another context build",
    ):
        if provider == "gemini":
            _run_signed_gemini_mixed_tool_turn(
                tmp_path,
                runtime=RuntimeWithTerminalEvent(),
                send_calls=sends,
                tool_calls=invocations,
            )
        else:
            _run_mixed_anthropic_compatible_tool_turn(
                RuntimeWithTerminalEvent(),
                provider=provider,
                send_calls=sends,
                tool_calls=invocations,
            )
    assert len(terminal_events) == 1
    assert invocations == ["durable"]
    assert len(sends) == 1


def test_anthropic_history_switches_to_openai_without_native_wire_leak(tmp_path):
    session_id = "phase2-switch-anthropic-openai"
    anthropic_sends, openai_sends, invocations = [], [], []
    first_runtime = _runtime(tmp_path)
    first = KernelLoop(
        model_io=_anthropic_tool_model(anthropic_sends),
        harnesses=list(first_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(tmp_path / "sessions")),
    ).run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=first_runtime.compose_event_callback(None),
        session_id=session_id, provider="anthropic", model="phase2-anthropic",
        toolkit=_toolkit(invocations), run_id="phase2-anthropic-attempt", max_iterations=2,
    )

    assert first.status == "completed"
    assert len(anthropic_sends) == 2
    second_runtime = _runtime(tmp_path)
    second = KernelLoop(
        model_io=_openai_final_model(openai_sends),
        harnesses=list(second_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(tmp_path / "sessions")),
    ).run(
        messages=[{"role": "user", "content": "continue with openai"}],
        callback=second_runtime.compose_event_callback(None),
        session_id=session_id, provider="openai", model="phase2-openai",
        toolkit=Toolkit(), run_id="phase2-openai-attempt", max_iterations=1,
    )

    assert second.status == "completed"
    rendered = repr(openai_sends[0]["input"])
    assert rendered.count("anthropic complete") == 1
    assert "signature" not in rendered
    assert "tool_use" not in rendered


def test_hyperspace_history_switches_to_openai_without_native_wire_leak(tmp_path):
    session_id = "phase2-switch-hyperspace-openai"
    store_path = tmp_path / "sessions"
    source_runtime = _runtime(tmp_path)
    source_sends, invocations = [], []
    source = _run_mixed_anthropic_compatible_tool_turn(
        source_runtime,
        provider="hyperspace",
        send_calls=source_sends,
        tool_calls=invocations,
        session_store=JsonFileSessionStore(store_path),
        session_id=session_id,
    )
    assert source.status == "completed"
    assert invocations == ["durable"]

    target_runtime = _runtime(tmp_path)
    target_sends = []
    target = KernelLoop(
        model_io=_openai_final_model(target_sends),
        harnesses=list(target_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(store_path)),
    ).run(
        messages=[{"role": "user", "content": "continue with openai"}],
        callback=target_runtime.compose_event_callback(None),
        session_id=session_id, provider="openai", model="phase2-openai",
        toolkit=Toolkit(), run_id="phase2-hyperspace-openai-attempt", max_iterations=1,
    )
    assert target.status == "completed"
    rendered = repr(target_sends[0]["input"])
    assert rendered.count("mixed tool turn complete") == 1
    assert "tool_use" not in rendered
    assert "signature" not in rendered


def test_gemini_history_switches_to_openai_without_native_wire_leak(tmp_path):
    session_id = "phase2-switch-gemini-openai"
    store_path = tmp_path / "sessions"
    source_runtime = _runtime(tmp_path)
    source, source_sends, invocations = _run_signed_gemini_mixed_tool_turn(
        tmp_path,
        runtime=source_runtime,
        session_store=JsonFileSessionStore(store_path),
        session_id=session_id,
    )
    assert source.status == "completed"
    assert invocations == ["durable"]
    assert len(source_sends) == 2

    target_runtime = _runtime(tmp_path)
    target_sends = []
    target = KernelLoop(
        model_io=_openai_final_model(target_sends),
        harnesses=list(target_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(store_path)),
    ).run(
        messages=[{"role": "user", "content": "continue with openai"}],
        callback=target_runtime.compose_event_callback(None),
        session_id=session_id, provider="openai", model="phase2-openai",
        toolkit=Toolkit(), run_id="phase2-gemini-openai-attempt", max_iterations=1,
    )
    assert target.status == "completed"
    rendered = repr(target_sends[0]["input"])
    assert rendered.count("'content': 'complete'") == 1
    assert "function_call" not in rendered
    assert "thought_signature" not in rendered


@pytest.mark.parametrize("target_provider", ["hyperspace", "gemini"])
def test_openai_history_switches_to_additional_provider_without_native_wire_leak(
    tmp_path, target_provider
):
    session_id = f"phase2-switch-openai-{target_provider}"
    store_path = tmp_path / "sessions"
    source_runtime = _runtime(tmp_path)
    source_sends, invocations = [], []
    source = KernelLoop(
        model_io=_openai_tool_model(source_sends),
        harnesses=list(source_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(store_path)),
    ).run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=source_runtime.compose_event_callback(None),
        session_id=session_id, provider="openai", model="phase2-openai",
        toolkit=_toolkit(invocations), run_id="phase2-openai-attempt", max_iterations=2,
    )
    assert source.status == "completed"
    assert invocations == ["switch"]

    target_runtime = _runtime(tmp_path)
    target_sends = []
    target_model = (
        _hyperspace_final_model(target_sends)
        if target_provider == "hyperspace"
        else _gemini_final_model(target_sends)
    )
    target = KernelLoop(
        model_io=target_model,
        harnesses=list(target_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(store_path)),
    ).run(
        messages=[{"role": "user", "content": f"continue with {target_provider}"}],
        callback=target_runtime.compose_event_callback(None),
        session_id=session_id, provider=target_provider,
        model=f"phase2-{target_provider}", toolkit=Toolkit(),
        run_id=f"phase2-openai-{target_provider}-attempt", max_iterations=1,
    )
    assert target.status == "completed"
    rendered = repr(target_sends[0])
    assert rendered.count("openai complete") == 1
    assert "OpenAI is checking the workspace." not in rendered
    if target_provider == "gemini":
        assert not any(
            part.function_call or part.thought_signature
            for content in target_sends[0]["contents"]
            for part in content.parts
        )
    else:
        assert "function_call" not in rendered


def test_openai_history_switches_to_anthropic_without_native_wire_leak(tmp_path):
    session_id = "phase2-switch-openai-anthropic"
    openai_sends, anthropic_sends, invocations = [], [], []
    first_runtime = _runtime(tmp_path)
    first = KernelLoop(
        model_io=_openai_tool_model(openai_sends),
        harnesses=list(first_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(tmp_path / "sessions")),
    ).run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=first_runtime.compose_event_callback(None),
        session_id=session_id, provider="openai", model="phase2-openai",
        toolkit=_toolkit(invocations), run_id="phase2-openai-attempt", max_iterations=2,
    )

    assert first.status == "completed"
    assert invocations == ["switch"]
    assert len(openai_sends) == 2

    second_runtime = _runtime(tmp_path)
    second = KernelLoop(
        model_io=_anthropic_final_model(anthropic_sends),
        harnesses=list(second_runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(tmp_path / "sessions")),
    ).run(
        messages=[{"role": "user", "content": "continue with anthropic"}],
        callback=second_runtime.compose_event_callback(None),
        session_id=session_id, provider="anthropic", model="phase2-anthropic",
        toolkit=Toolkit(), run_id="phase2-anthropic-attempt", max_iterations=1,
    )

    assert second.status == "completed"
    rendered = repr(anthropic_sends[0]["messages"])
    assert rendered.count("openai complete") == 1
    assert "function_call" not in rendered
    assert "OpenAI is checking the workspace." not in rendered


def test_anthropic_parallel_current_turn_keeps_text_and_calls_as_one_exchange(tmp_path):
    runtime = _runtime(tmp_path)
    sends, invocations = [], []
    assistant_message = [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "I will inspect both files."},
        {
            "type": "tool_use",
            "id": "phase2-parallel-1",
            "name": "probe",
            "input": {"query": "first"},
        },
        {
            "type": "tool_use",
            "id": "phase2-parallel-2",
            "name": "probe",
            "input": {"query": "second"},
        },
        {"type": "text", "text": "Then I will compare the results."},
    ]
    result = KernelLoop(
        model_io=_anthropic_tool_model(sends, tool_message=assistant_message),
        harnesses=list(runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(JsonFileSessionStore(tmp_path / "sessions")),
    ).run(
        messages=[{"role": "user", "content": "inspect both files"}],
        callback=runtime.compose_event_callback(None),
        session_id="phase2-parallel-anthropic", provider="anthropic",
        model="phase2-anthropic", toolkit=_toolkit(invocations),
        run_id="phase2-parallel-attempt", max_iterations=2,
    )

    assert result.status == "completed"
    assert invocations == ["first", "second"]
    assert len(sends) == 2
    replayed = next(
        message["content"] for message in sends[1]["messages"]
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    assert replayed == assistant_message


def test_current_native_text_is_token_charged_once_and_not_split_from_results():
    visible_text = "This visible text must be budgeted exactly once."
    assistant_messages = (
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": visible_text},
                {
                    "type": "tool_use", "id": "phase2-accounting-1",
                    "name": "probe", "input": {"query": "first"},
                },
                {
                    "type": "tool_use", "id": "phase2-accounting-2",
                    "name": "probe", "input": {"query": "second"},
                },
            ],
        },
    )
    compiled = _compiled_current_anthropic_exchange(
        assistant_messages=assistant_messages
    )
    compiled_data = compiled.to_dict()
    messages = compiled_data["messages"]
    diagnostics = compiled_data["diagnostics"]

    assert json.dumps(messages, ensure_ascii=False).count(visible_text) == 1
    assistant_index = next(
        index for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    result_index = assistant_index + 1
    assert [
        block["tool_use_id"] for block in messages[result_index]["content"]
        if block.get("type") == "tool_result"
    ] == ["phase2-accounting-1", "phase2-accounting-2"]
    assert diagnostics["after_estimated_tokens"] == estimate_context_tokens(
        messages
    ).total_tokens
    assert set(diagnostics["atomic_call_ids"]) >= {
        "phase2-accounting-1", "phase2-accounting-2"
    }


def test_verified_parallel_exchange_survives_sqlite_checkpoint_under_pressure(
    tmp_path, monkeypatch
):
    session_id = "phase2-checkpointed-native-turn"
    session_path = tmp_path / "sessions"
    compiled_contexts = []
    original_compile = ContextCompileCoordinator.compile

    def capture_real_compilation(coordinator, request):
        compiled = original_compile(coordinator, request)
        compiled_contexts.append(compiled)
        return compiled

    monkeypatch.setattr(
        ContextCompileCoordinator, "compile", capture_real_compilation
    )
    runtime = _runtime(
        tmp_path,
        checkpoint_capabilities=True,
        context_window_tokens=8_192,
    )

    for index in (1, 2):
        store = JsonFileSessionStore(session_path)
        seeded = KernelLoop(
            model_io=_anthropic_final_model([], text=f"history answer {index}"),
            harnesses=list(runtime.build_harnesses()),
            execution_runtime=ExecutionRuntime(store),
        ).run(
            messages=[{"role": "user", "content": f"history {index} " + "x" * 14_000}],
            callback=runtime.compose_event_callback(None),
            session_id=session_id, provider="anthropic", model="phase2-anthropic",
            toolkit=Toolkit(), run_id=f"phase2-pressure-history-{index}",
            max_iterations=1, max_context_window_tokens=8_192,
        )
        assert seeded.status == "completed"

    sends, invocations = [], []
    result = _run_mixed_anthropic_compatible_tool_turn(
        runtime,
        provider="anthropic",
        send_calls=sends,
        tool_calls=invocations,
        call_specs=(("pressure-call-1", "probe", {"query": "first"}),
                    ("pressure-call-2", "probe", {"query": "second"})),
        messages=[{"role": "user", "content": "inspect both current files"}],
        session_store=JsonFileSessionStore(session_path),
        session_id=session_id,
        sibling_text_before="I will inspect both files.",
        sibling_text_after="Then I will compare the results.",
        max_context_window_tokens=8_192,
    )

    assert result.status == "completed"
    assert invocations == ["first", "second"]
    assert len(sends) == 2
    second_messages = sends[1]["messages"]
    serialized_wire = json.dumps(second_messages, ensure_ascii=False)
    assert serialized_wire.count("I will inspect both files.") == 1
    assert serialized_wire.count("Then I will compare the results.") == 1
    assistant_index = next(
        index for index, message in enumerate(second_messages)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    assistant_content = second_messages[assistant_index]["content"]
    assert [block.get("text") for block in assistant_content
            if block.get("type") == "text"] == [
                "I will inspect both files.", "Then I will compare the results."
            ]
    result_messages = second_messages[assistant_index + 1:assistant_index + 3]
    result_ids = [
        block["tool_use_id"]
        for message in result_messages
        for block in message.get("content", [])
        if block.get("type") == "tool_result"
    ]
    assert result_ids == ["pressure-call-1", "pressure-call-2"]

    matching_compilations = [
        compiled for compiled in compiled_contexts
        if "I will inspect both files." in json.dumps(
            compiled.to_dict()["messages"], ensure_ascii=False
        )
    ]
    assert len(matching_compilations) == 1
    compiled = matching_compilations[0]
    compiled_messages = compiled.to_dict()["messages"]
    serialized_compiled = json.dumps(compiled_messages, ensure_ascii=False)
    assert serialized_compiled.count("I will inspect both files.") == 1
    assert serialized_compiled.count("Then I will compare the results.") == 1
    assert compiled.diagnostics["after_estimated_tokens"] == (
        estimate_context_tokens(compiled_messages).total_tokens
    )

    with sqlite3.connect(tmp_path / "memory_v2" / "context_v2.sqlite3") as db:
        checkpoint_rows = db.execute(
            "SELECT status FROM checkpoints WHERE execution_id=?",
            (session_id,),
        ).fetchall()
        current_build = db.execute(
            "SELECT envelope_json FROM context_builds WHERE execution_id=? "
            "ORDER BY trigger_store_seq DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    assert checkpoint_rows
    assert {row[0] for row in checkpoint_rows} == {"committed"}
    assert current_build is not None
    current_envelope = json.loads(current_build[0])
    assert current_envelope["checkpoint_refs"]
    assert current_envelope["estimated_input_tokens"] == (
        compiled.diagnostics["after_estimated_tokens"]
    )


def test_oversized_verified_native_exchange_stops_before_second_provider_send(tmp_path):
    runtime = _runtime(
        tmp_path,
        checkpoint_capabilities=True,
        context_window_tokens=8_192,
    )
    sends, invocations = [], []
    with pytest.raises(ContextBudgetExceededError):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime,
            provider="anthropic",
            send_calls=sends,
            tool_calls=invocations,
            messages=[{"role": "user", "content": "inspect the file"}],
            sibling_text_before="required provider text " + "x" * 80_000,
            session_id="phase2-oversized-native-exchange",
            max_context_window_tokens=8_192,
        )
    assert invocations == ["durable"]
    assert len(sends) == 1
