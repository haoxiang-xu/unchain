from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from unchain.context.artifacts import ArtifactService
from unchain.context.budget import resolve_context_budget
from unchain.context.coordinator import (
    ContextCompileCoordinator,
    ContextCompileCoordinatorError,
)
from unchain.context.factory import ContextExecutionBundle, DurableContextRuntimeFactory
from unchain.context.handoff import DurableHandoffRecorder, HandoffService
from unchain.context.ingress import ContextInputIngress, HostResolvedCurrentInput
from unchain.context.models import ContextCompileRequest
from unchain.context.ports import ContextBuildReceipt
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.context.provider_execution import ContextProviderTurnExecutionService
from unchain.context.request_factory import JournalContextRequestFactory
from unchain.context.runtime import ContextRuntime
from unchain.context.tool_boundary import DurableToolBoundary
from unchain.execution import ExecutionRuntime
from unchain.journal import ArtifactRef, DurableEventSink, ResourceRef
from unchain.kernel.loop import KernelLoop
from unchain.memory import InMemorySessionStore
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.providers import (
    AnthropicModelIO,
    GeminiModelIO,
    HyperspaceModelIO,
    OllamaModelIO,
)
from unchain.providers.durable_turn_runtime import DurableProviderTurnMode
from unchain.tools import Toolkit


TARGET_SHA256 = "d" * 64


class _CheckpointRepository:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id

    def prepare(self, **kwargs):
        raise AssertionError(kwargs)

    def commit(self, **kwargs):
        raise AssertionError(kwargs)

    def get_by_operation(self, **_kwargs):
        return None


class _BuildRepository:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self._operations = {}
        self._triggers = {}

    def record(self, *, envelope, operation, trigger_cursor):
        receipt = ContextBuildReceipt(
            envelope=envelope,
            operation=operation,
            trigger_cursor=trigger_cursor,
        )
        self._operations[operation.operation_id] = receipt
        self._triggers[trigger_cursor] = receipt
        return receipt

    def get_by_operation(self, *, operation):
        return self._operations.get(operation.operation_id)

    def get_by_trigger(self, *, trigger_cursor):
        return self._triggers.get(trigger_cursor)


def _current_input(context, attempt):
    users = [
        message
        for message in context.latest_messages()
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not users:
        return None
    return HostResolvedCurrentInput(
        attempt=attempt,
        content=str(users[-1].get("content") or ""),
    )


def _runtime(
    tmp_path,
    *,
    provider_turn_result_reader_factory=None,
    repository_observer=None,
    checkpoint_capabilities=False,
    context_window_tokens=16_384,
):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )

    def build(attempt):
        repository = store.bind_execution(attempt.generation.execution_id)
        if repository_observer is not None:
            repository_observer(repository)
        artifacts = ArtifactService(
            repository,
            sanitizer=lambda content, media_type: content,
        )
        projector = CanonicalSemanticEventProjector(
            attempt=attempt,
            artifacts=artifacts,
            payload_sanitizer=lambda event_type, payload: payload,
        )
        sink = DurableEventSink(repository, attempt, projector)
        handoffs = HandoffService(artifacts)
        provider_turn_result_reader = repository.read_full_verified
        if provider_turn_result_reader_factory is not None:
            provider_turn_result_reader = provider_turn_result_reader_factory(repository)
        compiler_capabilities = (
            SQLiteContextCompilerV2Store(context_store=store).bind_execution(
                attempt.generation.execution_id,
                artifacts=artifacts,
            )
            if checkpoint_capabilities
            else None
        )
        return ContextExecutionBundle(
            attempt=attempt,
            journal=repository,
            projector=projector,
            durable_event_sink=sink,
            coordinator=ContextCompileCoordinator(
                journal=repository,
                checkpoint_repository=(
                    compiler_capabilities.checkpoints
                    if compiler_capabilities is not None
                    else _CheckpointRepository(attempt.generation.execution_id)
                ),
                build_repository=(
                    compiler_capabilities.context_builds
                    if compiler_capabilities is not None
                    else _BuildRepository(attempt.generation.execution_id)
                ),
                partial_attempt_sink=lambda request, error: None,
                provider_turn_result_reader=provider_turn_result_reader,
            ),
            artifacts=artifacts,
            handoffs=handoffs,
            ingress=ContextInputIngress(
                attempt=attempt,
                projector=projector,
                sink=sink,
            ),
            request_factory=JournalContextRequestFactory(
                attempt=attempt,
                journal=repository,
                model_window_fallback=lambda provider, model: context_window_tokens,
            ),
            tool_boundary=DurableToolBoundary(
                attempt=attempt,
                projector=projector,
                sink=sink,
            ),
            handoff_recorder=DurableHandoffRecorder(
                attempt=attempt,
                handoffs=handoffs,
                projector=projector,
                sink=sink,
            ),
            partial_attempt_sink=lambda event, error: None,
            provider_turn_service=ContextProviderTurnExecutionService(
                attempt=attempt,
                store=repository,
                mode=DurableProviderTurnMode.ENFORCE_TEST,
                transport_target_sha256=TARGET_SHA256,
                sleep=lambda _seconds: None,
            ),
        )

    return ContextRuntime.from_factory(
        owner_id="context-v2-cross-provider-turn",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, execution_id: (
                f"generation-{execution_id}"
            ),
            current_input_resolver=_current_input,
        ),
        provider_turns_enabled=True,
    )


class _AnthropicStream:
    def __init__(self, text: str) -> None:
        self._text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(usage={"input_tokens": 3, "output_tokens": 0}),
        )
        yield SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="text_delta", text=self._text),
        )
        yield SimpleNamespace(
            type="message_delta",
            usage={"input_tokens": 3, "output_tokens": 2},
        )


def _anthropic_family_model_io(provider: str, send_calls: list[dict]):
    text = f"exact {provider}"

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _AnthropicStream(text)

    class _Client:
        messages = _Messages()

    common = {
        "model": f"{provider}-boundary-model",
        "api_key": "test-key",
        "client_factory": lambda **_kwargs: _Client(),
        "default_payloads": {},
        "model_capabilities": {},
    }
    if provider == "anthropic":
        model_io = AnthropicModelIO(**common)
    else:
        model_io = HyperspaceModelIO(**common)
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    return model_io, text


def _ollama_model_io(send_calls: list[dict]):
    text = "exact ollama"

    class _Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield json.dumps(
                {
                    "message": {"role": "assistant", "content": text},
                    "prompt_eval_count": 3,
                    "eval_count": 2,
                    "done": True,
                }
            )

        def read(self):
            return b""

    def stream_factory(method, url, **kwargs):
        send_calls.append(
            {
                "method": method,
                "url": url,
                **copy.deepcopy(kwargs),
            }
        )
        return _Response()

    model_io = OllamaModelIO(
        model="ollama-boundary-model",
        base_url="http://ollama.test",
        stream_factory=stream_factory,
        default_payloads={},
        model_capabilities={},
    )
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    return model_io, text


def _provider_model_io(provider: str, send_calls: list[dict]):
    if provider in {"anthropic", "hyperspace"}:
        return _anthropic_family_model_io(provider, send_calls)
    return _ollama_model_io(send_calls)


@pytest.mark.parametrize("provider", ["anthropic", "hyperspace", "ollama"])
def test_enabled_context_runtime_owns_empty_tool_provider_turn_for_non_openai_providers(
    tmp_path,
    provider,
):
    runtime = _runtime(tmp_path)
    send_calls: list[dict] = []
    model_io, expected_text = _provider_model_io(provider, send_calls)
    execution_id = f"execution-boundary-{provider}"
    attempt_id = f"attempt-boundary-{provider}"
    loop = KernelLoop(
        model_io=model_io,
        harnesses=list(runtime.build_harnesses()),
    )

    result = loop.run(
        messages=[{"role": "user", "content": f"use durable {provider}"}],
        callback=runtime.compose_event_callback(None),
        session_id=execution_id,
        provider=provider,
        model=model_io.model,
        toolkit=Toolkit(),
        run_id=attempt_id,
        max_iterations=1,
    )

    assert result.messages[-1]["content"] == expected_text
    assert len(send_calls) == 1
    sent_request = send_calls[0]
    if provider == "ollama":
        assert sent_request["method"] == "POST"
        assert sent_request["url"] == "http://ollama.test/api/chat"
        sent_request = sent_request["json"]
    assert sent_request["model"] == model_io.model
    assert "tools" not in sent_request

    reopened_store = SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )
    reopened = reopened_store.bind_execution(execution_id)
    events = reopened.capture_snapshot().events
    wire_events = [
        event for event in events if event.event_type == "provider.wire_snapshot"
    ]
    result_events = [
        event for event in events if event.event_type == "provider.turn_result"
    ]

    assert len(wire_events) == 1
    assert len(result_events) == 1
    assert wire_events[0].payload["provider"] == provider
    wire_artifact = ArtifactRef.from_dict(wire_events[0].payload["wire_artifact"])
    result_artifact = ArtifactRef.from_dict(result_events[0].payload["result_artifact"])
    wire_payload = json.loads(
        reopened.read_provider_wire_full_verified(artifact=wire_artifact)
    )
    result_payload = json.loads(
        reopened.read_provider_turn_result_full_verified(artifact=result_artifact)
    )
    assert wire_payload["provider"] == provider
    assert result_payload["result"]["final_text"] == expected_text


def _run_mixed_anthropic_compatible_tool_turn(
    runtime,
    *,
    provider: str,
    send_calls,
    tool_calls,
    after_tool=None,
    messages=None,
    session_store=None,
    call_specs=(
        ("mixed-call-1", "probe", {"query": "durable"}),
    ),
    session_id="execution-mixed-anthropic",
    sibling_text_before="I will inspect the workspace first.",
    sibling_text_after=None,
    max_context_window_tokens=None,
    transient_second_send=False,
    retry_config=None,
):

    tool_blocks = [
        {
            "type": "tool_use",
            "id": call_id,
            "name": name,
            "input": arguments,
        }
        for call_id, name, arguments in call_specs
    ]

    class _Stream:
        def __init__(self, content: list[dict]) -> None:
            self._content = content

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return SimpleNamespace(id=f"mixed-turn-{len(send_calls)}", content=self._content)

    responses = [
        [
            {"type": "thinking", "thinking": "plan", "signature": "signed"},
            {"type": "text", "text": sibling_text_before},
            *tool_blocks,
            *(
                [{"type": "text", "text": sibling_text_after}]
                if sibling_text_after is not None
                else []
            ),
        ],
        [{"type": "text", "text": "mixed tool turn complete"}],
    ]

    class _Messages:
        def stream(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            if transient_second_send and len(send_calls) == 2:
                class RetryableSendError(RuntimeError):
                    status_code = 429

                raise RetryableSendError("retry the exact second request")
            return _Stream(responses.pop(0))

    class _Client:
        messages = _Messages()

    model_io = (AnthropicModelIO if provider == "anthropic" else HyperspaceModelIO)(
        model=f"{provider}-mixed-boundary",
        api_key="test-key",
        client_factory=lambda **_kwargs: _Client(),
        default_payloads={},
        model_capabilities={},
    )
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    toolkit = Toolkit()

    def probe(query: str = "") -> dict[str, str]:
        tool_calls.append(query)
        if after_tool is not None:
            after_tool()
        return {"query": query, "status": "complete"}

    toolkit.register(probe, name="probe")
    loop = KernelLoop(
        model_io=model_io,
        harnesses=list(runtime.build_harnesses()),
        retry_config=retry_config,
        execution_runtime=ExecutionRuntime(
            session_store or InMemorySessionStore()
        ),
    )

    return loop.run(
        messages=messages or [{"role": "user", "content": "call the probe"}],
        callback=runtime.compose_event_callback(None),
        session_id=session_id,
        provider=provider,
        model=model_io.model,
        toolkit=toolkit,
        run_id="attempt-mixed-anthropic",
        max_iterations=2,
        max_context_window_tokens=max_context_window_tokens,
    )

def test_durable_anthropic_mixed_text_and_tool_turn_replays_complete_semantics(
    tmp_path,
):
    """The second durable provider turn keeps visible text beside tool_use."""

    send_calls: list[dict] = []
    tool_calls: list[str] = []
    result = _run_mixed_anthropic_compatible_tool_turn(
        _runtime(tmp_path),
        provider="anthropic",
        send_calls=send_calls,
        tool_calls=tool_calls,
    )

    assert result.status == "completed"
    assert result.messages[-1]["content"] == "mixed tool turn complete"
    assert tool_calls == ["durable"]
    assert len(send_calls) == 2
    replayed = send_calls[1]["messages"][-2]["content"]
    assert replayed == [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "I will inspect the workspace first."},
        {
            "type": "tool_use",
            "id": "mixed-call-1",
            "name": "probe",
            "input": {"query": "durable"},
        },
    ]


def test_durable_hyperspace_mixed_text_and_tool_turn_replays_complete_semantics(
    tmp_path,
):
    """The shared durable repair retains text for DeepSeek's native route."""

    send_calls: list[dict] = []
    tool_calls: list[str] = []
    result = _run_mixed_anthropic_compatible_tool_turn(
        _runtime(tmp_path),
        provider="hyperspace",
        send_calls=send_calls,
        tool_calls=tool_calls,
    )

    assert result.status == "completed"
    assert tool_calls == ["durable"]
    assert len(send_calls) == 2
    assert send_calls[1]["messages"][-2]["content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "I will inspect the workspace first."},
        {
            "type": "tool_use",
            "id": "mixed-call-1",
            "name": "probe",
            "input": {"query": "durable"},
        },
    ]


def test_durable_hyperspace_four_tool_turn_replays_complete_semantics(tmp_path):
    """DeepSeek-compatible replay keeps every parallel call in order."""

    send_calls: list[dict] = []
    tool_calls: list[str] = []
    call_specs = tuple(
        (f"deepseek-call-{index}", "probe", {"query": f"durable-{index}"})
        for index in range(1, 5)
    )

    result = _run_mixed_anthropic_compatible_tool_turn(
        _runtime(tmp_path),
        provider="hyperspace",
        send_calls=send_calls,
        tool_calls=tool_calls,
        call_specs=call_specs,
    )

    assert result.status == "completed"
    assert tool_calls == ["durable-1", "durable-2", "durable-3", "durable-4"]
    assert len(send_calls) == 2
    replayed = next(
        message["content"]
        for message in send_calls[1]["messages"]
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    assert replayed == [
        {"type": "thinking", "thinking": "plan", "signature": "signed"},
        {"type": "text", "text": "I will inspect the workspace first."},
        *[
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": arguments,
            }
            for call_id, name, arguments in call_specs
        ],
    ]


def test_durable_gemini_mixed_text_and_tool_turn_replays_complete_semantics(
    tmp_path,
):
    """The second Gemini request retains text and the function call together."""

    from google.genai import types

    runtime = _runtime(tmp_path)
    send_calls: list[dict] = []
    tool_calls: list[str] = []
    responses = [
        [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "I will inspect the workspace first."},
                                {
                                    "function_call": {
                                        "id": "gemini-call-1",
                                        "name": "probe",
                                        "args": {"query": "durable-1"},
                                    }
                                },
                                {
                                    "function_call": {
                                        "id": "gemini-call-2",
                                        "name": "probe",
                                        "args": {"query": "durable-2"},
                                    }
                                },
                            ],
                        },
                        "finish_reason": "STOP",
                    }
                ]
            }
        ],
        [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "mixed tool turn complete"}],
                        },
                        "finish_reason": "STOP",
                    }
                ]
            }
        ],
    ]

    def client_factory(**_kwargs):
        def generate_content_stream(**kwargs):
            send_calls.append(kwargs)
            return iter(
                types.GenerateContentResponse.model_validate(item)
                for item in responses.pop(0)
            )

        return SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=generate_content_stream),
            close=lambda: None,
        )

    model_io = GeminiModelIO(
        model="gemini-mixed-boundary",
        api_key="test-key",
        client_factory=client_factory,
        default_payloads={},
        model_capabilities={},
    )
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    toolkit = Toolkit()

    def probe(query: str = "") -> dict[str, str]:
        tool_calls.append(query)
        return {"query": query, "status": "complete"}

    toolkit.register(probe, name="probe")
    loop = KernelLoop(
        model_io=model_io,
        harnesses=list(runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(InMemorySessionStore()),
    )

    result = loop.run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=runtime.compose_event_callback(None),
        session_id="execution-mixed-gemini",
        provider="gemini",
        model=model_io.model,
        toolkit=toolkit,
        run_id="attempt-mixed-gemini",
        max_iterations=2,
    )

    assert result.status == "completed"
    assert tool_calls == ["durable-1", "durable-2"]
    assert len(send_calls) == 2
    replayed = next(
        content.model_dump(mode="json", exclude_none=True)
        for content in send_calls[1]["contents"]
        if content.role == "model"
        and any(part.function_call for part in content.parts)
    )
    assert replayed == {
        "role": "model",
        "parts": [
            {"text": "I will inspect the workspace first."},
            {
                "function_call": {
                    "id": "gemini-call-1",
                    "name": "probe",
                    "args": {"query": "durable-1"},
                }
            },
            {
                "function_call": {
                    "id": "gemini-call-2",
                    "name": "probe",
                    "args": {"query": "durable-2"},
                }
            },
        ],
    }


def test_durable_native_tool_turn_rejects_tampered_provider_result_before_replay(
    tmp_path,
):
    """A changed result artifact cannot create a second provider request."""

    send_calls: list[dict] = []
    tool_calls: list[str] = []
    runtime = _runtime(
        tmp_path,
        provider_turn_result_reader_factory=lambda _repository: (
            lambda *, artifact: b"{}"
        ),
    )

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="provider result is invalid",
    ):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime,
            provider="anthropic",
            send_calls=send_calls,
            tool_calls=tool_calls,
        )

    assert tool_calls == ["durable"]
    assert len(send_calls) == 1


def test_durable_native_tool_turn_without_result_reader_stops_before_replay(
    tmp_path,
):
    """A durable native turn cannot silently use call-only reconstruction."""

    send_calls: list[dict] = []
    tool_calls: list[str] = []
    runtime = _runtime(
        tmp_path,
        provider_turn_result_reader_factory=lambda _repository: None,
    )

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="artifact reader is unavailable",
    ):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime,
            provider="anthropic",
            send_calls=send_calls,
            tool_calls=tool_calls,
        )

    assert tool_calls == ["durable"]
    assert len(send_calls) == 1


def test_durable_native_tool_turn_rejects_later_wire_receipt_before_replay(
    tmp_path,
):
    """A second wire receipt for the same turn stops the next provider send."""

    from unchain.journal import JournalAppendRequest, OperationRef

    send_calls: list[dict] = []
    tool_calls: list[str] = []
    repositories = []
    runtime = _runtime(
        tmp_path,
        repository_observer=repositories.append,
    )

    def append_later_wire_receipt() -> None:
        repository = repositories[0]
        original = next(
            event
            for event in repository.capture_snapshot().events
            if event.event_type == "provider.wire_snapshot"
        )
        repository.append(
            request=JournalAppendRequest(
                event_id="ticket-390-later-wire-event",
                event_type=original.event_type,
                attempt=original.attempt,
                operation=OperationRef(
                    "ticket-390-later-wire-operation",
                    "f" * 64,
                ),
                payload=original.payload,
                resource_refs=original.resource_refs,
            )
        )

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="conflicting provider wire snapshots",
    ):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime,
            provider="anthropic",
            send_calls=send_calls,
            tool_calls=tool_calls,
            after_tool=append_later_wire_receipt,
        )

    assert tool_calls == ["durable"]
    assert len(send_calls) == 1


def test_verified_current_provider_turn_rejects_missing_receipt_without_reader():
    """A durable native batch cannot use synthetic replay without its receipt."""

    from unchain.context.coordinator import _verified_current_provider_turn
    result_ref = ArtifactRef(
        ref=ResourceRef(
            "artifact", "missing-provider-result", 1
        ),
        media_type="application/json",
        byte_length=2,
        sha256="a" * 64,
        preview="",
    )
    request = ContextCompileRequest(
        case="missing-provider-result-reader",
        source_messages=({"role": "user", "content": "call the probe"},),
        fixed_overhead_tokens=0,
        semantic_events=(
            {
                "type": "tool_call",
                "event_id": "call-event",
                "store_seq": 1,
                "attempt_id": "attempt-1",
                "run_id": "attempt-1",
                "iteration": 0,
                "source_provider": "anthropic",
                "call_id": "call-1",
                "tool_name": "probe",
                "arguments": {"query": "durable"},
            },
            {
                "type": "tool_result",
                "event_id": "result-event",
                "store_seq": 2,
                "attempt_id": "attempt-1",
                "run_id": "attempt-1",
                "iteration": 0,
                "source_provider": "anthropic",
                "call_id": "call-1",
                "tool_name": "probe",
                "result": {"status": "complete"},
            },
        ),
        pending_task_inputs=(
            {
                "event_id": "result-event",
                "store_seq": 2,
                "type": "tool_result",
                "content_ref": result_ref.ref.to_dict(),
                "content_bytes": 2,
                "content_sha256": "b" * 64,
            },
        ),
        budget=resolve_context_budget(context_window_tokens=16_384),
        provider="anthropic",
        model="test-model",
        build_id="build-1",
        execution_id="execution-1",
        generation_id="generation-1",
        attempt_id="attempt-1",
    )

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="provider result is missing",
    ):
        _verified_current_provider_turn(
            request=request,
            generation_events=(),
            artifact_reader=None,
        )


def test_verified_current_provider_turn_rejects_visible_output_receipt_mutation():
    """A receipt cannot relabel the visibility recorded in its result artifact."""

    from unchain.context.coordinator import _verified_current_provider_turn
    from unchain.journal.models import (
        AttemptRef,
        GenerationRef,
        JournalEvent,
        OperationRef,
    )
    from unchain.journal.provider_result import (
        ProviderTurnResultEnvelope,
        build_provider_turn_result_event_payload,
    )
    from unchain.kernel.types import ModelTurnResult, ToolCall
    from unchain.providers.request_lease import ProviderRequestSubject
    from unchain.providers.wire_envelope import (
        ProviderWireEnvelope,
        ProviderWireRoute,
    )

    attempt = AttemptRef(
        GenerationRef("execution-visible-output", "generation-visible-output"),
        "attempt-visible-output",
    )
    wire = ProviderWireEnvelope(
        attempt=attempt,
        iteration=0,
        provider="openai",
        configured_model="test-model",
        request_model="test-model",
        adapter_revision="unchain.openai.responses.request.v1",
        transport_kind="openai.responses.create",
        transport_target_sha256="1" * 64,
        source_request_sha256="2" * 64,
        source_payload_sha256="3" * 64,
        catalog_sha256="4" * 64,
        prompt_sha256="5" * 64,
        tool_schema_sha256="6" * 64,
        required_betas=(),
        base_anthropic_betas=(),
        routes=(
            ProviderWireRoute(
                name="primary",
                request={
                    "model": "test-model",
                    "input": [{"role": "user", "content": "call the probe"}],
                    "stream": True,
                    "store": False,
                },
            ),
        ),
    )
    wire_bytes = wire.canonical_bytes()
    wire_artifact = ArtifactRef(
        ref=ResourceRef("artifact", "visible-output-wire", 1),
        media_type="application/json",
        byte_length=len(wire_bytes),
        sha256=hashlib.sha256(wire_bytes).hexdigest(),
        preview="",
    )
    wire_event = JournalEvent(
        event_id="visible-output-wire-event",
        event_type="provider.wire_snapshot",
        attempt=attempt,
        operation=OperationRef("visible-output-wire-operation", "7" * 64),
        store_seq=1,
        payload={
            "iteration": 0,
            "provider": "openai",
            "adapter_revision": "unchain.openai.responses.request.v1",
            "catalog_sha256": wire.catalog_sha256,
            "envelope_sha256": wire.envelope_sha256,
            "wire_artifact": wire_artifact.to_dict(),
        },
        resource_refs=(wire_artifact.ref,),
    )
    result = ProviderTurnResultEnvelope.from_model_turn_result(
        subject=ProviderRequestSubject(
            attempt=attempt,
            iteration=0,
            envelope_sha256=wire.envelope_sha256,
            route="primary",
            retry_ordinal=0,
        ),
        route_sha256=wire.routes[0].route_sha256,
        visible_output=True,
        result=ModelTurnResult(
            assistant_messages=[
                {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "probe",
                    "arguments": {"query": "durable"},
                }
            ],
            tool_calls=[ToolCall("call-1", "probe", {"query": "durable"})],
        ),
    )
    result_bytes = result.canonical_bytes()
    result_artifact = ArtifactRef(
        ref=ResourceRef("artifact", "visible-output-result", 1),
        media_type="application/json",
        byte_length=len(result_bytes),
        sha256=hashlib.sha256(result_bytes).hexdigest(),
        preview="",
    )
    result_event = JournalEvent(
        event_id="visible-output-result-event",
        event_type="provider.turn_result",
        attempt=attempt,
        operation=OperationRef("visible-output-result-operation", "8" * 64),
        store_seq=2,
        payload={
            **build_provider_turn_result_event_payload(
                envelope=result,
                artifact=result_artifact,
            ),
            "visible_output": False,
        },
        resource_refs=(result_artifact.ref,),
    )
    tool_result_ref = ResourceRef("artifact", "visible-output-tool-result", 1)
    request = ContextCompileRequest(
        case="visible-output-receipt-mutation",
        source_messages=({"role": "user", "content": "call the probe"},),
        fixed_overhead_tokens=0,
        semantic_events=(
            {
                "type": "tool_call",
                "event_id": "visible-output-call-event",
                "store_seq": 3,
                "attempt_id": attempt.attempt_id,
                "run_id": attempt.attempt_id,
                "iteration": 0,
                "source_provider": "openai",
                "call_id": "call-1",
                "tool_name": "probe",
                "arguments": {"query": "durable"},
            },
            {
                "type": "tool_result",
                "event_id": "visible-output-tool-result-event",
                "store_seq": 4,
                "attempt_id": attempt.attempt_id,
                "run_id": attempt.attempt_id,
                "iteration": 0,
                "source_provider": "openai",
                "call_id": "call-1",
                "tool_name": "probe",
                "result": {"status": "complete"},
            },
        ),
        pending_task_inputs=(
            {
                "event_id": "visible-output-tool-result-event",
                "store_seq": 4,
                "type": "tool_result",
                "content_ref": tool_result_ref.to_dict(),
                "content_bytes": 2,
                "content_sha256": "9" * 64,
            },
        ),
        budget=resolve_context_budget(context_window_tokens=16_384),
        provider="openai",
        model="test-model",
        build_id="build-visible-output",
        execution_id=attempt.generation.execution_id,
        generation_id=attempt.generation.generation_id,
        attempt_id=attempt.attempt_id,
    )
    contents = {
        wire_artifact.ref: wire_bytes,
        result_artifact.ref: result_bytes,
    }

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="provider result is invalid",
    ):
        _verified_current_provider_turn(
            request=request,
            generation_events=(wire_event, result_event),
            artifact_reader=lambda *, artifact: contents[artifact.ref],
        )

    valid_result_event = JournalEvent(
        event_id="visible-output-valid-result-event",
        event_type="provider.turn_result",
        attempt=attempt,
        operation=OperationRef("visible-output-valid-result-operation", "c" * 64),
        store_seq=2,
        payload=build_provider_turn_result_event_payload(
            envelope=result,
            artifact=result_artifact,
        ),
        resource_refs=(result_artifact.ref,),
    )
    projection = _verified_current_provider_turn(
        request=request,
        generation_events=(wire_event, valid_result_event),
        artifact_reader=lambda *, artifact: contents[artifact.ref],
    )

    assert projection is not None
    assert projection.subject == result.subject
    assert projection.subject_sha256 == result.subject_sha256
    assert projection.result_cursor.store_seq == valid_result_event.store_seq
    assert projection.result_cursor.event_id == valid_result_event.event_id
    assert projection.result_sha256 == result.result_sha256
    assert projection.replay_frame_sha256 is None


def test_provider_result_requires_matching_wire_subject_and_route():
    """A result from another request cannot authorize native-tool replay."""

    from unchain.context.coordinator import (
        _verified_provider_wire_snapshot_for_result,
    )
    from unchain.journal.models import (
        AttemptRef,
        GenerationRef,
        JournalEvent,
        OperationRef,
    )
    from unchain.journal.provider_result import (
        ProviderTurnResultEnvelope,
        build_provider_turn_result_event_payload,
        provider_turn_result_event_fields,
    )
    from unchain.providers.request_lease import ProviderRequestSubject
    from unchain.providers.wire_envelope import (
        ProviderWireEnvelope,
        ProviderWireRoute,
    )
    from unchain.kernel.types import ModelTurnResult

    attempt = AttemptRef(
        GenerationRef("execution-wire-binding", "generation-wire-binding"),
        "attempt-wire-binding",
    )
    wire = ProviderWireEnvelope(
        attempt=attempt,
        iteration=0,
        provider="openai",
        configured_model="test-model",
        request_model="test-model",
        adapter_revision="unchain.openai.responses.request.v1",
        transport_kind="openai.responses.create",
        transport_target_sha256="1" * 64,
        source_request_sha256="2" * 64,
        source_payload_sha256="3" * 64,
        catalog_sha256="4" * 64,
        prompt_sha256="5" * 64,
        tool_schema_sha256="6" * 64,
        required_betas=(),
        base_anthropic_betas=(),
        routes=(
            ProviderWireRoute(
                name="primary",
                request={
                    "model": "test-model",
                    "input": [{"role": "user", "content": "hello"}],
                    "stream": True,
                    "store": False,
                },
            ),
        ),
    )
    wire_bytes = wire.canonical_bytes()
    wire_artifact = ArtifactRef(
        ref=ResourceRef("artifact", "wire-binding", 1),
        media_type="application/json",
        byte_length=len(wire_bytes),
        sha256=hashlib.sha256(wire_bytes).hexdigest(),
        preview="",
    )
    wire_event = JournalEvent(
        event_id="wire-binding-event",
        event_type="provider.wire_snapshot",
        attempt=attempt,
        operation=OperationRef("wire-binding-operation", "7" * 64),
        store_seq=1,
        payload={
            "iteration": 0,
            "provider": "openai",
            "adapter_revision": "unchain.openai.responses.request.v1",
            "catalog_sha256": wire.catalog_sha256,
            "envelope_sha256": wire.envelope_sha256,
            "wire_artifact": wire_artifact.to_dict(),
        },
        resource_refs=(wire_artifact.ref,),
    )
    result = ProviderTurnResultEnvelope.from_model_turn_result(
        subject=ProviderRequestSubject(
            attempt=attempt,
            iteration=0,
            envelope_sha256="f" * 64,
            route="primary",
            retry_ordinal=0,
        ),
        route_sha256=wire.routes[0].route_sha256,
        visible_output=True,
        result=ModelTurnResult(
            assistant_messages=[],
            tool_calls=[],
            final_text="",
            response_id=None,
            reasoning_items=None,
            consumed_tokens=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            provider_replay_frame=None,
        ),
    )
    result_bytes = result.canonical_bytes()
    result_artifact = ArtifactRef(
        ref=ResourceRef("artifact", "result-binding", 1),
        media_type="application/json",
        byte_length=len(result_bytes),
        sha256=hashlib.sha256(result_bytes).hexdigest(),
        preview="",
    )
    result_event = JournalEvent(
        event_id="result-binding-event",
        event_type="provider.turn_result",
        attempt=attempt,
        operation=OperationRef("result-binding-operation", "8" * 64),
        store_seq=2,
        payload=build_provider_turn_result_event_payload(
            envelope=result,
            artifact=result_artifact,
        ),
        resource_refs=(result_artifact.ref,),
    )
    contents = {
        wire_artifact.ref: wire_bytes,
        result_artifact.ref: result_bytes,
    }

    with pytest.raises(ContextCompileCoordinatorError, match="wire snapshot"):
        _verified_provider_wire_snapshot_for_result(
            provider="openai",
            iteration=0,
            result_event=result_event,
            result_fields=provider_turn_result_event_fields(result_event),
            generation_events=(wire_event, result_event),
            artifact_reader=lambda *, artifact: contents[artifact.ref],
        )

    matching_result = ProviderTurnResultEnvelope.from_model_turn_result(
        subject=ProviderRequestSubject(
            attempt=attempt,
            iteration=0,
            envelope_sha256=wire.envelope_sha256,
            route="primary",
            retry_ordinal=0,
        ),
        route_sha256=wire.routes[0].route_sha256,
        visible_output=True,
        result=ModelTurnResult(
            assistant_messages=[],
            tool_calls=[],
            final_text="",
            response_id=None,
            reasoning_items=None,
            consumed_tokens=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            provider_replay_frame=None,
        ),
    )
    matching_result_bytes = matching_result.canonical_bytes()
    matching_result_artifact = ArtifactRef(
        ref=ResourceRef("artifact", "matching-result-binding", 1),
        media_type="application/json",
        byte_length=len(matching_result_bytes),
        sha256=hashlib.sha256(matching_result_bytes).hexdigest(),
        preview="",
    )
    matching_result_event = JournalEvent(
        event_id="matching-result-binding-event",
        event_type="provider.turn_result",
        attempt=attempt,
        operation=OperationRef("matching-result-binding-operation", "a" * 64),
        store_seq=2,
        payload=build_provider_turn_result_event_payload(
            envelope=matching_result,
            artifact=matching_result_artifact,
        ),
        resource_refs=(matching_result_artifact.ref,),
    )
    later_wire_event = JournalEvent(
        event_id="later-wire-binding-event",
        event_type="provider.wire_snapshot",
        attempt=attempt,
        operation=OperationRef("later-wire-binding-operation", "b" * 64),
        store_seq=3,
        payload=wire_event.payload,
        resource_refs=wire_event.resource_refs,
    )
    contents[matching_result_artifact.ref] = matching_result_bytes

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="conflicting provider wire snapshots",
    ):
        _verified_provider_wire_snapshot_for_result(
            provider="openai",
            iteration=0,
            result_event=matching_result_event,
            result_fields=provider_turn_result_event_fields(matching_result_event),
            generation_events=(wire_event, matching_result_event, later_wire_event),
            artifact_reader=lambda *, artifact: contents[artifact.ref],
        )

    route_mismatch = ProviderTurnResultEnvelope.from_model_turn_result(
        subject=ProviderRequestSubject(
            attempt=attempt,
            iteration=0,
            envelope_sha256=wire.envelope_sha256,
            route="primary",
            retry_ordinal=0,
        ),
        route_sha256="e" * 64,
        visible_output=True,
        result=ModelTurnResult(
            assistant_messages=[],
            tool_calls=[],
            final_text="",
            response_id=None,
            reasoning_items=None,
            consumed_tokens=0,
            input_tokens=0,
            output_tokens=0,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
            provider_replay_frame=None,
        ),
    )
    route_mismatch_bytes = route_mismatch.canonical_bytes()
    route_mismatch_artifact = ArtifactRef(
        ref=ResourceRef("artifact", "route-binding", 1),
        media_type="application/json",
        byte_length=len(route_mismatch_bytes),
        sha256=hashlib.sha256(route_mismatch_bytes).hexdigest(),
        preview="",
    )
    route_mismatch_event = JournalEvent(
        event_id="route-binding-event",
        event_type="provider.turn_result",
        attempt=attempt,
        operation=OperationRef("route-binding-operation", "9" * 64),
        store_seq=2,
        payload=build_provider_turn_result_event_payload(
            envelope=route_mismatch,
            artifact=route_mismatch_artifact,
        ),
        resource_refs=(route_mismatch_artifact.ref,),
    )
    contents[route_mismatch_artifact.ref] = route_mismatch_bytes

    with pytest.raises(ContextCompileCoordinatorError, match="wire snapshot"):
        _verified_provider_wire_snapshot_for_result(
            provider="openai",
            iteration=0,
            result_event=route_mismatch_event,
            result_fields=provider_turn_result_event_fields(route_mismatch_event),
            generation_events=(wire_event, route_mismatch_event),
            artifact_reader=lambda *, artifact: contents[artifact.ref],
        )


@pytest.mark.parametrize(
    ("provider", "assistant_messages", "expected_calls"),
    [
        (
            "anthropic",
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "probe",
                            "input": {"query": "changed"},
                        }
                    ],
                }
            ],
            (("call-1", "probe", {"query": "durable"}),),
        ),
        (
            "hyperspace",
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call-1",
                            "name": "renamed_probe",
                            "input": {"query": "durable"},
                        }
                    ],
                }
            ],
            (("call-1", "probe", {"query": "durable"}),),
        ),
        (
            "gemini",
            [
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "id": "call-b",
                                "name": "second",
                                "args": {"position": 2},
                            }
                        },
                        {
                            "function_call": {
                                "id": "call-a",
                                "name": "first",
                                "args": {"position": 1},
                            }
                        },
                    ],
                }
            ],
            (
                ("call-a", "first", {"position": 1}),
                ("call-b", "second", {"position": 2}),
            ),
        ),
    ],
)
def test_verified_provider_turn_rejects_semantic_call_mismatch(
    provider,
    assistant_messages,
    expected_calls,
):
    from unchain.context.coordinator import _validate_semantic_tool_call_group

    with pytest.raises(ContextCompileCoordinatorError, match="does not match"):
        _validate_semantic_tool_call_group(
            provider=provider,
            assistant_messages=assistant_messages,
            expected_calls=expected_calls,
        )


def test_verified_provider_turn_accepts_openai_computer_call_semantics():
    """Computer-use calls use actions instead of function-call arguments."""

    from unchain.context.coordinator import _validate_semantic_tool_call_group

    _validate_semantic_tool_call_group(
        provider="openai",
        assistant_messages=(
            {
                "type": "computer_call",
                "call_id": "computer-call-1",
                "actions": ({"type": "click", "x": 4, "y": 8},),
            },
        ),
        expected_calls=(
            (
                "computer-call-1",
                "computer",
                {
                    "provider": "openai",
                    "protocol": "openai.responses.computer.v1",
                    "actions": [{"type": "click", "x": 4, "y": 8}],
                },
            ),
        ),
    )
