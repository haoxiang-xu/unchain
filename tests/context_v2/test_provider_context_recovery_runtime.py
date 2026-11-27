from __future__ import annotations

import copy
import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import openai
import pytest

from unchain.journal import AttemptRef, GenerationRef, JournalAppendRequest, OperationRef
from unchain.journal.runtime import SemanticEventDraft
from unchain.kernel.loop import KernelLoop
from unchain.providers import OpenAIModelIO
from unchain.tools import Toolkit

from test_context_provider_turn_cross_provider import _runtime, _AnthropicStream


EXECUTION = "execution-context-recovery"
ATTEMPT = "attempt-context-recovery"


def _seed_history(repository):
    if repository.capture_snapshot().events:
        return
    attempt = AttemptRef(GenerationRef(EXECUTION, f"generation-{EXECUTION}"), ATTEMPT)
    for index, role in enumerate(("user", "assistant")):
        payload = {"run_id": ATTEMPT, "message": {"role": role, "content": "Earlier completed work. " * 250}}
        draft = SemanticEventDraft(
            event_id=f"old-{index}", event_type=f"message.{role}", attempt=attempt,
            operation_id=f"old-operation-{index}", payload=payload,
        )
        repository.append(request=JournalAppendRequest(
            event_id=f"old-{index}", event_type=f"message.{role}", attempt=attempt,
            operation=draft.operation,
            payload=payload,
        ))


def _model(sent, *, reject_count=1, code="context_length_exceeded", message="Input exceeds model context window", text="continued task"):
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __iter__(self):
            yield SimpleNamespace(type="response.completed", response=SimpleNamespace(
                id="accepted-after-compaction", output=[{"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": text}]}],
                usage={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}))

    class Responses:
        def create(self, **kwargs):
            sent.append(copy.deepcopy(kwargs))
            if len(sent) <= reject_count:
                raise openai.BadRequestError("rejected", response=httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1/responses")),
                    body={"error": {"code": code, "message": message}})
            return Stream()

    client = SimpleNamespace(responses=Responses())
    return OpenAIModelIO(model="gpt-test", api_key="test-key", client_factory=lambda **kwargs: client,
                         default_payloads={}, model_capabilities={})


def _run(runtime, model, *, toolkit=None, max_iterations=1, execution_runtime=None,
         run_id=ATTEMPT, user_input="Continue the current task."):
    return KernelLoop(model_io=model, harnesses=list(runtime.build_harnesses()), execution_runtime=execution_runtime).run(
        messages=[{"role": "system", "content": "Keep the current user instructions."},
                  {"role": "user", "content": user_input}],
        callback=runtime.compose_event_callback(None), session_id=EXECUTION, provider=model.provider,
        model=model.model, toolkit=toolkit or Toolkit(), run_id=run_id, max_iterations=max_iterations)


def test_rejected_input_is_recompiled_and_rebuilt_before_retry(tmp_path):
    repositories = []
    def observe(repository):
        repositories.append(repository)
        _seed_history(repository)
    runtime = _runtime(tmp_path, repository_observer=observe, checkpoint_capabilities=True)
    sent = []
    result = _run(runtime, _model(sent))
    assert result.messages[-1]["content"] == "continued task"
    assert len(sent) == 2
    assert len(json.dumps(sent[1])) < len(json.dumps(sent[0]))
    assert "Continue the current task." in json.dumps(sent[1])
    assert "Keep the current user instructions." in json.dumps(sent[1])
    events = repositories[-1].capture_snapshot().events
    assert len([event for event in events if event.event_type == "context.provider_recovery"]) == 1
    assert len([event for event in events if event.event_type == "provider.wire_snapshot"]) == 2


def test_generic_validation_error_never_starts_compaction(tmp_path):
    from unchain.providers.durable_turn_runtime import DurableProviderTurnTerminalError
    sent = []
    runtime = _runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True)
    with pytest.raises(DurableProviderTurnTerminalError):
        _run(runtime, _model(sent, code="invalid_tool"))
    assert len(sent) == 1


def test_conflicting_context_code_does_not_authorize_recovery(tmp_path):
    from unchain.providers.durable_turn_runtime import DurableProviderTurnTerminalError
    sent = []
    runtime = _runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True)
    with pytest.raises(DurableProviderTurnTerminalError):
        _run(runtime, _model(sent, message="Invalid API key"))
    assert len(sent) == 1


@pytest.mark.parametrize("provider", ["anthropic", "hyperspace", "gemini", "ollama"])
def test_protocol_adapters_recover_only_rejected_input(tmp_path, provider):
    from unchain.providers import AnthropicModelIO, HyperspaceModelIO, GeminiModelIO, OllamaModelIO
    from google.genai import types
    from google.genai.errors import APIError
    import anthropic

    sent = []
    def send(kwargs):
        sent.append(copy.deepcopy(kwargs))
        if len(sent) == 1:
            if provider in {"anthropic", "hyperspace"}:
                raise anthropic.BadRequestError("rejected", response=httpx.Response(400, request=httpx.Request("POST", "https://provider.test/v1/messages")),
                    body={"error": {"type": "invalid_request_error", "message": "prompt is too long: 20000 tokens > 16000 maximum"}})
            message = "The input token count (20000) exceeds the maximum number of tokens allowed (16000)."
            response = httpx.Response(400, request=httpx.Request("POST", "https://provider.test/generate"),
                                      json={"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": message}})
            if provider == "gemini":
                raise APIError(400, response.json(), response)
            response.raise_for_status()

    if provider in {"anthropic", "hyperspace"}:
        def stream(**kwargs):
            send(kwargs)
            return _AnthropicStream("continued task")
        client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
        cls = AnthropicModelIO if provider == "anthropic" else HyperspaceModelIO
        model = cls(model="claude-test", api_key="test-key", client_factory=lambda **kwargs: client,
                    default_payloads={}, model_capabilities={})
    elif provider == "gemini":
        def generate(**kwargs):
            send(kwargs)
            return iter([types.GenerateContentResponse.model_validate({"candidates": [{"content": {
                "role": "model", "parts": [{"text": "continued task"}]}, "finish_reason": "STOP"}]})])
        model = GeminiModelIO(model="gemini-test", api_key="test-key", default_payloads={}, model_capabilities={},
            client_factory=lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate), close=lambda: None))
    else:
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def read(self): return b""
            def iter_lines(self):
                yield json.dumps({"message": {"role": "assistant", "content": "continued task"}, "done": True,
                    "prompt_eval_count": 3, "eval_count": 2})
        def ollama_stream(method, url, **kwargs):
            if not sent:
                sent.append(copy.deepcopy(kwargs))
                response = httpx.Response(400, request=httpx.Request(method, url),
                    json={"error": "The input token count (20000) exceeds the maximum number of tokens allowed (16000)."})
                return nullcontext(response)
            send(kwargs)
            return Response()
        model = OllamaModelIO(model="ollama-test", stream_factory=ollama_stream,
            default_payloads={}, model_capabilities={})
    runtime = _runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True)
    result = _run(runtime, model)
    assert result.messages[-1]["content"] == "continued task"
    assert len(sent) == 2


def test_cold_restart_after_recovery_record_does_not_resend_rejected_wire(tmp_path, monkeypatch):
    import unchain.context.provider_recovery as recovery
    class Crash(BaseException): pass
    original = recovery.prepare_provider_context_recovery
    def crash_after_record(**kwargs):
        original(**kwargs)
        raise Crash()
    sent = []
    with monkeypatch.context() as patch:
        patch.setattr(recovery, "prepare_provider_context_recovery", crash_after_record)
        with pytest.raises(Crash):
            _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), _model(sent))
    assert len(sent) == 1
    result = _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), _model(sent))
    assert result.messages[-1]["content"] == "continued task"
    assert len(sent) == 2
    assert len(json.dumps(sent[1])) < len(json.dumps(sent[0]))


@pytest.mark.parametrize("stage", ["compiled", "result_committed"])
def test_cold_restart_at_recovery_commit_boundaries(tmp_path, monkeypatch, stage):
    import unchain.context.provider_recovery as recovery
    from unchain.providers.durable_turn_runtime import DurableProviderTurnRuntime
    class Crash(BaseException): pass
    sent = []
    with monkeypatch.context() as patch:
        if stage == "compiled":
            original = recovery.verify_provider_context_reduction
            def crash_compiled(*args):
                original(*args)
                raise Crash()
            patch.setattr(recovery, "verify_provider_context_reduction", crash_compiled)
        else:
            def crash_complete(*args, **kwargs): raise Crash()
            patch.setattr(DurableProviderTurnRuntime, "_complete", crash_complete)
        with pytest.raises(Crash):
            _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), _model(sent))
    assert len(sent) == (1 if stage == "compiled" else 2)
    result = _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), _model(sent))
    assert result.messages[-1]["content"] == "continued task"
    assert len(sent) == 2


def test_unknown_send_outcome_after_compaction_is_never_sent_again(tmp_path):
    from unchain.providers.durable_turn_runtime import DurableProviderTurnUncertainError
    from unchain.context.coordinator import ContextCompileCoordinatorError
    sent = []
    class Responses:
        def create(self, **kwargs):
            sent.append(copy.deepcopy(kwargs))
            if len(sent) == 1:
                raise openai.BadRequestError("rejected", response=httpx.Response(400, request=httpx.Request("POST", "https://provider.test/responses")),
                    body={"error": {"code": "context_length_exceeded"}})
            raise ConnectionError("response lost after send")
    client = SimpleNamespace(responses=Responses())
    def model():
        return OpenAIModelIO(model="gpt-test", api_key="test-key", client_factory=lambda **kwargs: client,
            default_payloads={}, model_capabilities={})
    with pytest.raises(DurableProviderTurnUncertainError):
        _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), model())
    # A controlled uncertain failure has already terminated the owner attempt.
    # Cold admission remains terminal; it must not silently reopen and resend.
    with pytest.raises(ContextCompileCoordinatorError, match="terminal attempt"):
        _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), model())
    assert len(sent) == 2


def test_repeated_rejections_are_bounded_and_irreducible_context_is_explicit(tmp_path):
    from unchain.context.provider_recovery import ContextLimitRecoveryError
    from unchain.context.compiler import ContextBudgetExceededError
    sent = []
    with pytest.raises((ContextLimitRecoveryError, ContextBudgetExceededError)):
        _run(_runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True), _model(sent, reject_count=10))
    assert 1 <= len(sent) <= 3


def test_recovery_preserves_tool_pair_without_reexecuting_completed_tool(tmp_path):
    import anthropic
    from unchain.execution import ExecutionRuntime
    from unchain.memory import InMemorySessionStore
    from unchain.providers import AnthropicModelIO
    sent, effects = [], []
    class Stream:
        def __init__(self, content): self.content = content
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __iter__(self): return iter(())
        def get_final_message(self):
            return SimpleNamespace(id=f"tool-turn-{len(sent)}", content=self.content)
    def stream(**kwargs):
        sent.append(copy.deepcopy(kwargs))
        if len(sent) == 1:
            return Stream([{"type": "tool_use", "id": "probe-once", "name": "probe", "input": {"query": "side effect"}}])
        if len(sent) == 2:
            raise anthropic.BadRequestError("rejected", response=httpx.Response(400, request=httpx.Request("POST", "https://provider.test/messages")),
                body={"error": {"type": "invalid_request_error", "message": "prompt is too long"}})
        return Stream([{"type": "text", "text": "continued task"}])
    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    model = AnthropicModelIO(model="claude-test", api_key="test-key", client_factory=lambda **kwargs: client,
                            default_payloads={}, model_capabilities={})
    toolkit = Toolkit()
    def probe(query: str = "") -> dict:
        effects.append(query)
        return {"status": "complete", "query": query}
    toolkit.register(probe, name="probe")
    runtime = _runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True)
    result = _run(runtime, model, toolkit=toolkit, max_iterations=2,
                  execution_runtime=ExecutionRuntime(InMemorySessionStore()))
    assert result.messages[-1]["content"] == "continued task"
    assert effects == ["side effect"]
    assert len(sent) == 3
    assert '"type": "tool_use"' in json.dumps(sent[-1])
    assert '"type": "tool_result"' in json.dumps(sent[-1])
    assert "Continue the current task." in json.dumps(sent[-1])


def test_second_input_in_same_chat_does_not_reuse_prior_recovery(tmp_path):
    first, second = [], []
    runtime = _runtime(tmp_path, repository_observer=_seed_history, checkpoint_capabilities=True)
    _run(runtime, _model(first, text="The first task is finished. " * 250))
    result = _run(runtime, _model(second, reject_count=0), run_id="second-attempt", user_input="Follow the second instruction.")
    assert result.messages[-1]["content"] == "continued task"
    assert len(first) == 2
    assert len(second) == 1
    assert "Follow the second instruction." in json.dumps(second[-1])
