from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from unchain.journal import AttemptRef, GenerationRef
from unchain.context.composition import CONTEXT_COMPOSITION_EXTENSION_KEY
from unchain.context import ContextCompiler, ContextRuntime
from unchain.kernel.run_ledger import build_model_attempt_receipt
from unchain.kernel.state import RunState
from unchain.kernel.types import ModelTurnResult
from unchain.persistence import SQLiteContextV2Store
from unchain.providers import OllamaModelIO, OpenAIModelIO
from unchain.providers.base import ModelTurnRequest
from unchain.providers.durable_turn_runtime import (
    DurableProviderTurnError,
    DurableProviderTurnMode,
    DurableProviderTurnUncertainError,
    ExactProviderRouteFailure,
    ExactProviderRouteFailureKind,
    ExactProviderRouteTransport,
)
from unchain.providers.model_turn_runtime import build_model_turn_request
from unchain.retry import RetryConfig, RetriesExhaustedError
from unchain.run_bundle import ProviderCallUsage, RunIdentity
from unchain.providers.turn_ownership import ProviderTurnOwnership
from unchain.tools import Toolkit


ATTEMPT = AttemptRef(
    GenerationRef("execution-provider-service", "generation-provider-service"),
    "attempt-provider-service",
)
TARGET_SHA256 = "b" * 64


class _OpenAIStream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.output_text.delta", delta="durable ")
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="response-provider-service",
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "durable result"}],
                    }
                ],
                usage={"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
            ),
        )


def _model_io(send_calls: list[dict]):
    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _OpenAIStream()

    class _Client:
        responses = _Responses()

    return OpenAIModelIO(
        model="gpt-test",
        api_key="test-key",
        client_factory=lambda **_kwargs: _Client(),
        default_payloads={},
        model_capabilities={},
    )


def _request(events: list[dict] | None = None) -> ModelTurnRequest:
    return ModelTurnRequest(
        messages=[{"role": "user", "content": "execute exactly once"}],
        payload={},
        callback=(events.append if events is not None else None),
        run_id=ATTEMPT.attempt_id,
        iteration=2,
        toolkit=Toolkit(),
        emit_stream=True,
    )


def _composition_manifest(*, fallback: bool = False):
    def route(name, mode, retained, surface):
        return {
            "route_name": name,
            "context_mode": mode,
            "provider_retained": retained,
            "manifest_items": 1,
            "wire_surfaces": 1,
            "contributions": [
                {
                    "category": "skills",
                    "subtype": "expanded_invocation",
                    "surface": surface,
                    "utf8_bytes": 4,
                    "source_count": 1,
                }
            ],
        }

    routes = [
        route(
            "primary",
            "remote_continuation" if fallback else "semantic",
            fallback,
            "provider_state" if fallback else "messages",
        )
    ]
    if fallback:
        routes.append(
            route(
                "openai_previous_response_fallback",
                "local_replay",
                False,
                "messages",
            )
        )
    return {
        "schema": "unchain.context/internal_context_composition_v1",
        "method": "utf8_heuristic_v2",
        "context_window_tokens": 272_000,
        "routes": routes,
    }


def _repository(tmp_path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )
    return store.bind_execution(ATTEMPT.generation.execution_id)


def _service(tmp_path, mode):
    from unchain.context.provider_execution import (
        ContextProviderTurnExecutionService,
    )
    return ContextProviderTurnExecutionService(
        attempt=ATTEMPT,
        store=_repository(tmp_path),
        mode=mode,
        transport_target_sha256=TARGET_SHA256,
        sleep=lambda _seconds: None,
    )


def _run_receipt_factory():
    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    def build(attempt, started_at, completed_at, outcome, classification, result):
        return build_model_attempt_receipt(
            identity=identity,
            provider="openai",
            model="gpt-test",
            iteration=2,
            retry_ordinal=attempt,
            purpose="agent_turn",
            request_digest="e" * 64,
            route="openai.responses.create",
            started_at=started_at,
            completed_at=completed_at,
            turn=result,
            status=outcome,
            classification=classification,
        )

    return build


class _FailingTransport(ExactProviderRouteTransport):
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def send(self, *, envelope, route, retry_ordinal):
        del envelope, route, retry_ordinal
        self.calls += 1
        raise self.error

    def discard_buffered_events(self):
        return None

    def release_buffered_events(self):
        return None


class _SequenceTransport(ExactProviderRouteTransport):
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def send(self, *, envelope, route, retry_ordinal):
        self.calls.append((route.name, retry_ordinal))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def discard_buffered_events(self):
        return None

    def release_buffered_events(self):
        return None


def _turn_result(text="composition result"):
    return ModelTurnResult(
        assistant_messages=[{"role": "assistant", "content": text}],
        tool_calls=[],
        final_text=text,
        response_id="response-composition",
        consumed_tokens=4,
        input_tokens=2,
        output_tokens=2,
    )


def test_ollama_enforce_previews_before_result_cas_and_journals_afterward(tmp_path):
    durable = []
    external = []
    context = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=lambda _context: None,
        durable_event_sink=durable.append,
        partial_attempt_sink=lambda _event, _error: None,
    )
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            yield json.dumps({"message": {"thinking": "plan"}, "done": False})
            assert [event["delta"] for event in external if event["type"] == "reasoning"] == ["plan"]
            assert [event for event in durable if event["type"] == "reasoning"] == []
            assert not any(
                event.event_type == "provider.turn_result"
                for event in service.store.capture_snapshot().events
            )
            yield json.dumps({"message": {"content": "ok"}, "done": True})

    io = OllamaModelIO(
        model="qwen3",
        stream_factory=lambda *_args, **_kwargs: Response(),
        default_payloads={},
        model_capabilities={},
    )
    request = ModelTurnRequest(
        messages=[{"role": "user", "content": "hello"}],
        callback=context.compose_event_callback(external.append),
        run_id=ATTEMPT.attempt_id,
        iteration=2,
        toolkit=Toolkit(),
        emit_stream=True,
    )
    result = service.fetch_prepared(
        model_io=io, request=request, retry_config=RetryConfig(max_retries=0)
    )

    assert result.final_text == "ok"
    assert any(
        event.event_type == "provider.turn_result"
        for event in service.store.capture_snapshot().events
    )
    assert [event["delta"] for event in durable if event["type"] == "reasoning"] == ["plan"]
    assert [event["delta"] for event in external if event["type"] == "reasoning"] == ["plan"]

def test_off_is_a_read_only_legacy_fallthrough_without_durable_authority(tmp_path):
    send_calls: list[dict] = []
    service = _service(tmp_path, DurableProviderTurnMode.OFF)

    result = service.fetch_prepared(
        model_io=_model_io(send_calls),
        request=_request(),
        retry_config=RetryConfig(max_retries=0),
    )

    assert result is None
    assert send_calls == []
    assert service.store.capture_snapshot().events == ()


def test_shadow_persists_exact_authority_but_never_sends(tmp_path):
    send_calls: list[dict] = []
    service = _service(tmp_path, DurableProviderTurnMode.SHADOW)

    result = service.fetch_prepared(
        model_io=_model_io(send_calls),
        request=_request(),
        retry_config=RetryConfig(max_retries=0),
    )

    assert result is None
    assert send_calls == []
    assert [event.event_type for event in service.store.capture_snapshot().events] == [
        "tool.catalog_snapshot",
        "provider.wire_snapshot",
    ]


def test_enforce_sends_once_then_releases_buffered_callbacks(tmp_path):
    send_calls: list[dict] = []
    events: list[dict] = []
    attempts: list[int] = []
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)

    result = service.fetch_prepared(
        model_io=_model_io(send_calls),
        request=_request(events),
        retry_config=RetryConfig(max_retries=0),
        before_attempt=attempts.append,
    )

    assert result.final_text == "durable result"
    assert len(send_calls) == 1
    assert attempts == [0]
    assert [event["type"] for event in events] == [
        "request_messages",
        "token_delta",
    ]


def test_production_enforce_mode_uses_the_official_transport_identity(tmp_path):
    from unchain.context.provider_execution import (
        ContextProviderTurnExecutionService,
        official_provider_transport_target_sha256,
    )

    service = ContextProviderTurnExecutionService(
        attempt=ATTEMPT,
        store=_repository(tmp_path),
        mode=DurableProviderTurnMode.ENFORCE,
        transport_target_sha256=official_provider_transport_target_sha256(),
        sleep=lambda _seconds: None,
    )
    sends = []
    result = service.fetch_prepared(
        model_io=_model_io(sends),
        request=_request(),
        retry_config=RetryConfig(max_retries=0),
    )

    assert result.final_text == "durable result"
    assert len(sends) == 1


def test_restart_recovers_result_without_resend_or_callback_replay(tmp_path):
    first_sends: list[dict] = []
    first_events: list[dict] = []
    first = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    expected = first.fetch_prepared(
        model_io=_model_io(first_sends),
        request=_request(first_events),
        retry_config=RetryConfig(max_retries=0),
    )

    restart_sends: list[dict] = []
    restart_events: list[dict] = []
    restarted = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    recovered = restarted.fetch_prepared(
        model_io=_model_io(restart_sends),
        request=_request(restart_events),
        retry_config=RetryConfig(max_retries=0),
    )

    assert recovered.final_text == expected.final_text
    assert recovered.assistant_messages == expected.assistant_messages
    assert recovered.provider_call_usage is None
    assert len(first_sends) == 1
    assert restart_sends == []
    assert restart_events == []


def test_occurrence_namespace_preserves_owner_iteration_and_replays_distinct_calls(
    tmp_path,
):
    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    def bound_owner():
        service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
        owner = ProviderTurnOwnership(
            identity=identity,
            service=service,
            ledger=service.store,
        )
        state = RunState()
        state.session_state.session_id = identity.execution_id
        state.run_ledger.initialize(
            state=state,
            run_id=identity.run_id,
            explicit_identity=identity,
        )
        state.run_ledger.bind_provider_turn_ownership(owner)
        return owner, state

    first_owner, first_state = bound_owner()
    first_sends: list[dict] = []
    first_model = _model_io(first_sends)
    for occurrence in ("same-wire:first", "same-wire:second"):
        result = first_owner.fetch_turn(
            state=first_state,
            model_io=first_model,
            request=_request(),
            occurrence_id=occurrence,
            purpose="tool_observation",
            iteration=2,
            request_sha256="a" * 64,
            retry_config=RetryConfig(max_retries=0),
            provider="openai",
            model="gpt-test",
        )
        assert result.final_text == "durable result"

    assert len(first_sends) == 2
    assert len(first_state.run_ledger.receipts) == 2
    assert len(set(first_state.run_ledger.receipts)) == 2

    result_events = [
        event
        for event in first_owner.service.store.capture_snapshot().events
        if event.event_type == "provider.turn_result"
    ]
    assert len(result_events) == 2
    assert {event.attempt.attempt_id for event in result_events} == {
        ATTEMPT.attempt_id
    }
    assert {event.payload["iteration"] for event in result_events} == {2}
    assert len(
        {event.attempt.generation.generation_id for event in result_events}
    ) == 2

    cold_owner, cold_state = bound_owner()
    cold_sends: list[dict] = []
    cold_model = _model_io(cold_sends)
    for occurrence in ("same-wire:first", "same-wire:second"):
        recovered = cold_owner.fetch_turn(
            state=cold_state,
            model_io=cold_model,
            request=_request(),
            occurrence_id=occurrence,
            purpose="tool_observation",
            iteration=2,
            request_sha256="a" * 64,
            retry_config=RetryConfig(max_retries=0),
            provider="openai",
            model="gpt-test",
        )
        assert recovered.final_text == "durable result"

    assert cold_sends == []
    assert len(cold_state.run_ledger.receipts) == 2


@pytest.mark.parametrize(
    ("purpose", "occurrence_id"),
    (
        ("tool_exposure", "tool_exposure:root"),
        ("tool_observation", "tool_observation:batch-1"),
        ("web_extract", "web_extract:call-1"),
    ),
)
def test_auxiliary_send_crash_after_atomic_cas_cold_replays_without_resend(
    tmp_path,
    purpose,
    occurrence_id,
):
    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    def bound_owner():
        service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
        owner = ProviderTurnOwnership(
            identity=identity,
            service=service,
            ledger=service.store,
        )
        state = RunState()
        state.session_state.session_id = identity.execution_id
        state.run_ledger.initialize(
            state=state,
            run_id=identity.run_id,
            explicit_identity=identity,
        )
        state.run_ledger.bind_provider_turn_ownership(owner)
        return owner, state

    owner, state = bound_owner()
    sends: list[dict] = []
    original_append = state.run_ledger.append

    class _ProcessCrash(BaseException):
        pass

    def crash_after_durable_cas(receipt):
        original_append(receipt)
        raise _ProcessCrash()

    state.run_ledger.append = crash_after_durable_cas
    with pytest.raises(_ProcessCrash):
        owner.fetch_turn(
            state=state,
            model_io=_model_io(sends),
            request=_request(),
            occurrence_id=occurrence_id,
            purpose=purpose,
            iteration=2,
            request_sha256="c" * 64,
            retry_config=RetryConfig(max_retries=0),
            provider="openai",
            model="gpt-test",
        )
    assert len(sends) == 1

    cold_owner, cold_state = bound_owner()
    cold_sends: list[dict] = []
    recovered = cold_owner.fetch_turn(
        state=cold_state,
        model_io=_model_io(cold_sends),
        request=_request(),
        occurrence_id=occurrence_id,
        purpose=purpose,
        iteration=2,
        request_sha256="c" * 64,
        retry_config=RetryConfig(max_retries=0),
        provider="openai",
        model="gpt-test",
    )
    assert recovered.final_text == "durable result"
    assert cold_sends == []
    assert len(cold_state.run_ledger.receipts) == 1


def test_web_extract_uses_owned_atomic_send_and_cold_replays(
    tmp_path,
    monkeypatch,
):
    from unchain.agent.model_io import ModelIOFactoryRegistry
    from unchain.toolkits.builtin.core.web_fetch import run_extract_model

    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    def bound_state():
        service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
        owner = ProviderTurnOwnership(
            identity=identity,
            service=service,
            ledger=service.store,
        )
        state = RunState()
        state.session_state.session_id = identity.execution_id
        state.run_ledger.initialize(
            state=state,
            run_id=identity.run_id,
            explicit_identity=identity,
        )
        state.run_ledger.bind_provider_turn_ownership(owner)
        return state

    sends: list[dict] = []
    monkeypatch.setattr(
        ModelIOFactoryRegistry,
        "create",
        lambda _registry, **_kwargs: _model_io(sends),
    )
    state = bound_state()
    execution_context = SimpleNamespace(
        run_state=state,
        iteration=7,
        call_id="web-call-7",
    )
    first = run_extract_model(
        url="https://example.test/page",
        content="The durable fact is 42.",
        prompt="What is the durable fact?",
        extract_model_config={
            "provider": "openai",
            "model": "gpt-test",
            "payload": {},
        },
        execution_context=execution_context,
    )
    assert first == "durable result"
    assert len(sends) == 1
    receipt = next(iter(state.run_ledger.receipts.values()))
    assert receipt.identity.iteration == 7
    assert receipt.identity.purpose == "web_extract"

    cold_sends: list[dict] = []
    monkeypatch.setattr(
        ModelIOFactoryRegistry,
        "create",
        lambda _registry, **_kwargs: _model_io(cold_sends),
    )
    cold_state = bound_state()
    recovered = run_extract_model(
        url="https://example.test/page",
        content="The durable fact is 42.",
        prompt="What is the durable fact?",
        extract_model_config={
            "provider": "openai",
            "model": "gpt-test",
            "payload": {},
        },
        execution_context=SimpleNamespace(
            run_state=cold_state,
            iteration=7,
            call_id="web-call-7",
        ),
    )
    assert recovered == first
    assert cold_sends == []
    assert len(cold_state.run_ledger.receipts) == 1


def test_tool_observation_harness_uses_owned_send_and_cold_replays(tmp_path):
    from unchain.kernel.harness import HarnessContext
    from unchain.kernel.loop import KernelLoop
    from unchain.tools.execution import ToolExecutionHarness
    from unchain.tools.types import ToolBatchState

    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    def bound_context(send_calls):
        service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
        owner = ProviderTurnOwnership(
            identity=identity,
            service=service,
            ledger=service.store,
        )
        state = RunState()
        state.session_state.session_id = identity.execution_id
        state.provider_state.provider = "openai"
        state.provider_state.model = "gpt-test"
        state.iteration = 5
        state.seed_messages([{"role": "user", "content": "inspect result"}])
        state.tool_batch_state = ToolBatchState(
            result_messages=[
                {
                    "role": "user",
                    "content": '{"ok":true}',
                }
            ],
            should_observe=True,
            executed_call_ids=["tool-call-5"],
        )
        state.run_ledger.initialize(
            state=state,
            run_id=identity.run_id,
            explicit_identity=identity,
        )
        state.run_ledger.bind_provider_turn_ownership(owner)
        model_io = _model_io(send_calls)
        loop = KernelLoop(model_io=model_io)
        return state, HarnessContext(
            state=state,
            phase="after_tool_batch",
            event={
                "payload": {},
                "response_format": None,
                "run_id": identity.run_id,
                "loop": loop,
                "model_io": model_io,
                "callback": None,
                "max_iterations": 6,
                "tool_runtime_plugins": [],
            },
        )

    sends: list[dict] = []
    first_state, first_context = bound_context(sends)
    delta = ToolExecutionHarness().build_delta(first_context)
    assert delta is not None
    assert len(sends) == 1
    first_receipt = next(iter(first_state.run_ledger.receipts.values()))
    assert first_receipt.identity.iteration == 5
    assert first_receipt.identity.purpose == "tool_observation"

    cold_sends: list[dict] = []
    cold_state, cold_context = bound_context(cold_sends)
    cold_delta = ToolExecutionHarness().build_delta(cold_context)
    assert cold_delta is not None
    assert cold_sends == []
    assert tuple(cold_state.run_ledger.receipts) == (
        first_receipt.provider_call_id,
    )


def test_memory_off_agent_and_tool_selector_share_owner_and_cold_replay(
    tmp_path,
):
    from unchain.agent import Agent, ToolOptimizerModule, ToolsModule
    from unchain.providers.turn_ownership import ProviderTurnOwnershipFactory
    from unchain.tools import Tool, ToolOptimizerConfig

    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    class _Factory:
        def bind(self, *, identity: RunIdentity):
            assert identity == globals_identity
            service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
            return ProviderTurnOwnership(
                identity=identity,
                service=service,
                ledger=service.store,
                factory=self,
            )

    globals_identity = identity
    factory = _Factory()
    assert isinstance(factory, ProviderTurnOwnershipFactory)

    toolkit = Toolkit()
    for index in range(2):
        toolkit.register(
            Tool.from_callable(
                lambda value="", index=index: {"index": index, "value": value},
                name=f"owned_tool_{index}",
                description=f"Owned tool {index}",
                parameters=[],
            )
        )

    def run_once(send_calls):
        model_io = _model_io(send_calls)
        agent = Agent(
            name="owned-memory-off-agent",
            provider="openai",
            model="gpt-test",
            modules=(
                ToolsModule(tools=(toolkit,)),
                ToolOptimizerModule(
                    config=ToolOptimizerConfig(
                        max_direct_tools=1,
                        trigger_tool_count=1,
                    )
                ),
            ),
            model_io_factory=lambda _spec, _context: model_io,
        )
        return agent.run(
            "finish without tools",
            max_iterations=1,
            session_id=identity.execution_id,
            run_id=identity.run_id,
            _run_bundle_identity=identity,
            _provider_turn_ownership_factory=factory,
        )

    first_sends: list[dict] = []
    first = run_once(first_sends)
    assert first.status == "completed"
    assert len(first_sends) == 2
    assert first.run_bundle is not None
    assert {
        receipt["identity"]["purpose"]
        for receipt in first.run_bundle["provider_calls"]
    } == {"tool_exposure", "agent_turn"}

    cold_sends: list[dict] = []
    cold = run_once(cold_sends)
    assert cold.status == "completed"
    assert cold_sends == []
    assert cold.run_bundle == first.run_bundle


def test_guard_failure_preserves_uncertain_fence_without_network_or_callbacks(
    tmp_path,
):
    send_calls: list[dict] = []
    events: list[dict] = []
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)

    def fail_guard(_attempt: int) -> None:
        raise RuntimeError("execution lease was lost")

    with pytest.raises(DurableProviderTurnUncertainError):
        service.fetch_prepared(
            model_io=_model_io(send_calls),
            request=_request(events),
            retry_config=RetryConfig(max_retries=0),
            before_attempt=fail_guard,
        )

    assert send_calls == []
    assert events == []
    with pytest.raises(DurableProviderTurnUncertainError):
        service.fetch_prepared(
            model_io=_model_io(send_calls),
            request=_request(events),
            retry_config=RetryConfig(max_retries=0),
        )
    assert send_calls == []


def test_closed_mode_rejects_prior_enforce_evidence_instead_of_falling_through(
    tmp_path,
):
    send_calls: list[dict] = []
    _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io(send_calls),
        request=_request(),
        retry_config=RetryConfig(max_retries=0),
    )

    with pytest.raises(DurableProviderTurnError, match="durable evidence"):
        _service(tmp_path, DurableProviderTurnMode.OFF).fetch_prepared(
            model_io=_model_io([]),
            request=_request(),
            retry_config=RetryConfig(max_retries=0),
        )


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_error"),
    [
        (
            ExactProviderRouteFailure(
                ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
                RuntimeError("temporary"),
            ),
            "failed",
            RetriesExhaustedError,
        ),
        (RuntimeError("connection state unknown"), "uncertain", DurableProviderTurnUncertainError),
    ],
)
def test_failed_and_uncertain_sends_persist_one_atomic_receipt(
    tmp_path,
    monkeypatch,
    error,
    expected_status,
    expected_error,
):
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    transport = _FailingTransport(error)
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )
    observed = []

    with pytest.raises(expected_error):
        service.fetch_prepared(
            model_io=_model_io([]),
            request=_request(),
            retry_config=RetryConfig(max_retries=0),
            run_receipt_factory=_run_receipt_factory(),
            run_receipt_observed=observed.append,
        )

    receipts = service.store.load_receipts(
        root_run_id=ATTEMPT.attempt_id,
        owner_run_id=ATTEMPT.attempt_id,
        attempt_id=ATTEMPT.attempt_id,
    )
    assert transport.calls == 1
    assert receipts == tuple(observed)
    assert len(receipts) == 1
    assert receipts[0].status == expected_status
    assert receipts[0].timing.started_at is not None
    assert receipts[0].timing.completed_at is not None


def test_exact_route_composition_enriches_the_valid_base_receipt(tmp_path):
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    request = replace(
        _request(),
        internal_context_composition_v1=_composition_manifest(),
    )

    result = service.fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )

    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]
    assert extension["quality"] == "reconciled_estimate"
    assert extension["attributed_tokens"] == 1
    assert extension["residual_tokens"] == 1
    assert extension["wire"]["route_name"] == "primary"
    assert extension["coverage"] == {
        "status": "complete",
        "manifest_items": 1,
        "matched_items": 1,
        "wire_surfaces": 1,
        "matched_surfaces": 1,
    }


@pytest.mark.parametrize(
    ("case", "expected_quality", "expected_tokens", "expected_residual", "expected_total"),
    (
        ("reconciled", "reconciled_estimate", 1, 1, 2),
        ("provider_total_unavailable", "estimated", 1, None, None),
        # v2: the heuristic's raw estimate (3 tokens from 12 bytes at the
        # 4.5 divisor) overshoots the 2-token provider total, but coverage
        # is complete and a total is known, so this now scales onto that
        # total (see _scale_categories_to_target) instead of giving up as
        # "estimated" the way v1 did — attributed_tokens becomes exactly
        # the provider total, residual becomes exactly 0.
        ("heuristic_overestimate", "reconciled_estimate", 2, 0, 2),
        ("known_instrumentation_loss", "partial", 1, None, 2),
    ),
)
def test_composition_quality_table_preserves_authoritative_usage_semantics(
    tmp_path,
    monkeypatch,
    case,
    expected_quality,
    expected_tokens,
    expected_residual,
    expected_total,
):
    manifest = _composition_manifest()
    turn = replace(
        _turn_result(case),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {
                "input_tokens": 2,
                "output_tokens": 2,
                "total_tokens": 4,
            }
        ),
    )
    if case == "provider_total_unavailable":
        turn = replace(
            turn,
            provider_call_usage=ProviderCallUsage(source="unavailable"),
        )
    elif case == "heuristic_overestimate":
        manifest["routes"][0]["contributions"][0]["utf8_bytes"] = 12
    elif case == "known_instrumentation_loss":
        manifest["routes"][0]["manifest_items"] = 2
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(
        tmp_path,
        DurableProviderTurnMode.ENFORCE_TEST,
    ).fetch_prepared(
        model_io=_model_io([]),
        request=replace(
            _request(),
            internal_context_composition_v1=manifest,
        ),
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    receipt = result.provider_call_receipt
    extension = receipt.extensions[CONTEXT_COMPOSITION_EXTENSION_KEY]

    assert extension["quality"] == expected_quality
    assert extension["attributed_tokens"] == expected_tokens
    assert extension["residual_tokens"] == expected_residual
    assert extension["coverage"]["status"] == (
        "partial" if case == "known_instrumentation_loss" else "complete"
    )
    assert receipt.usage.input_total_tokens == expected_total
    assert "input_total_tokens" not in extension


def test_heuristic_token_estimate_uses_the_calibrated_bytes_per_token(
    tmp_path,
    monkeypatch,
):
    """Locks the bytes-per-token divisor at 4.5 (see _HEURISTIC_BYTES_PER_TOKEN
    in composition.py — calibrated 2026-08-21 against real o200k_base
    tokenization of real content: builtin tool schemas, source text, README
    markdown, an assistant reply). 17 bytes is deliberately chosen because
    the old bytes/4 divisor and the new 4.5 divisor round to DIFFERENT token
    counts (5 vs 4) — a byte count where they agreed could not catch a
    regression back to the old, more biased value. A generous provider
    total keeps this below the reconcile-scaling threshold (tested
    separately) — this test is only about the raw divisor."""
    manifest = _composition_manifest()
    manifest["routes"][0]["contributions"][0]["utf8_bytes"] = 17

    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 1000, "output_tokens": 2, "total_tokens": 1002}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=replace(_request(), internal_context_composition_v1=manifest),
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["attributed_tokens"] == 4


def _multi_leaf_manifest():
    """3 categories x 2 subtypes = 6 leaves, deliberately uneven byte sizes
    (100/50/30/20/15/5) so scaling has to do real apportionment work rather
    than a trivial even split. Hand-verified against a standalone script
    before being hardcoded here (see the plan this test implements) — this
    is not a "run the code, paste what it prints" assertion."""
    contributions = [
        {"category": "instructions", "subtype": "core_system", "surface": "messages", "utf8_bytes": 100, "source_count": 1},
        {"category": "instructions", "subtype": "agent_instructions", "surface": "messages", "utf8_bytes": 50, "source_count": 1},
        {"category": "skills", "subtype": "catalog_metadata", "surface": "messages", "utf8_bytes": 30, "source_count": 1},
        {"category": "skills", "subtype": "loaded_body", "surface": "messages", "utf8_bytes": 20, "source_count": 1},
        {"category": "tool_definitions", "subtype": "provider_schema", "surface": "messages", "utf8_bytes": 15, "source_count": 1},
        {"category": "tool_definitions", "subtype": "prompt_guidance", "surface": "messages", "utf8_bytes": 5, "source_count": 1},
    ]
    return {
        "schema": "unchain.context/internal_context_composition_v1",
        "method": "utf8_heuristic_v2",
        "context_window_tokens": 272_000,
        "routes": [
            {
                "route_name": "primary",
                "context_mode": "semantic",
                "provider_retained": False,
                "manifest_items": 6,
                "wire_surfaces": 1,
                "contributions": contributions,
            }
        ],
    }


def test_scaling_apportions_the_billed_total_across_categories_by_share(
    tmp_path,
    monkeypatch,
):
    """The core of the scaled-reconcile feature: at the raw 4.5 divisor
    these 6 leaves sum to 53 attributed tokens (bytes -> tokens: 100->23,
    50->12, 30->7, 20->5, 15->4, 5->2), overshooting a 30-token provider
    total. Post-scale every leaf must still hold >=1 token (the smallest,
    5 bytes, would floor to 0 without the +1 baseline — this fixture
    exists specifically to exercise that floor), and the total must land
    on exactly 30, not "close to" 30."""
    manifest = _multi_leaf_manifest()
    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 30, "output_tokens": 2, "total_tokens": 32}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=replace(_request(), internal_context_composition_v1=manifest),
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["quality"] == "reconciled_estimate"
    assert extension["attributed_tokens"] == 30
    assert extension["residual_tokens"] == 0

    by_category = {c["id"]: c for c in extension["categories"]}
    assert by_category["instructions"]["tokens"] == 18
    assert by_category["skills"]["tokens"] == 7
    assert by_category["tool_definitions"]["tokens"] == 5
    assert sum(c["tokens"] for c in extension["categories"]) == 30

    subtypes_by_id = {
        (c["id"], s["id"]): s["tokens"]
        for c in extension["categories"]
        for s in c["subtypes"]
    }
    assert subtypes_by_id[("instructions", "core_system")] == 11
    assert subtypes_by_id[("instructions", "agent_instructions")] == 7
    assert subtypes_by_id[("skills", "catalog_metadata")] == 4
    assert subtypes_by_id[("skills", "loaded_body")] == 3
    assert subtypes_by_id[("tool_definitions", "provider_schema")] == 3
    # The smallest contributor (5 raw bytes) floors to 0 share of the pool —
    # the +1 baseline is the only thing keeping it >=1.
    assert subtypes_by_id[("tool_definitions", "prompt_guidance")] == 2
    assert all(tokens >= 1 for tokens in subtypes_by_id.values())


def test_scaling_guard_declines_when_provider_total_is_below_leaf_count(
    tmp_path,
    monkeypatch,
):
    """Every leaf must keep >=1 token — scaling below the leaf count would
    have to invent tokens nobody measured, so the guard must decline and
    quality must fall back to "estimated" rather than fabricate a total."""
    manifest = _multi_leaf_manifest()
    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=replace(_request(), internal_context_composition_v1=manifest),
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["quality"] == "estimated"
    assert extension["residual_tokens"] is None
    assert extension["attributed_tokens"] == 53


def test_incomplete_coverage_never_scales_even_when_attributed_exceeds_input(
    tmp_path,
    monkeypatch,
):
    """Scaling onto an incomplete manifest would be false precision — some
    of what the provider billed was never attributed to any category at
    all, so there is nothing honest to anchor a proportion to. This must
    stay "partial" with a null residual even though attributed (53) is
    well above the provider total (30), the exact condition that triggers
    scaling when coverage IS complete."""
    manifest = _multi_leaf_manifest()
    manifest["routes"][0]["manifest_items"] = 7  # one more than matched -> incomplete
    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 30, "output_tokens": 2, "total_tokens": 32}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=replace(_request(), internal_context_composition_v1=manifest),
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["quality"] == "partial"
    assert extension["residual_tokens"] is None
    assert extension["attributed_tokens"] == 53
    assert extension["coverage"]["status"] == "partial"


def test_composition_method_is_pinned_to_v2(tmp_path, monkeypatch):
    """Regression lock: method must read utf8_heuristic_v2, not the old v1 —
    an archaeologist reading old vs new receipts depends on this being
    accurate, and it is trivial to silently drift back to v1 by reverting
    only CONTEXT_COMPOSITION_METHOD without anyone noticing (every fixture
    in this file hardcodes its OWN method string, so a revert would not
    even fail manifest validation)."""
    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 1000, "output_tokens": 2, "total_tokens": 1002}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=replace(_request(), internal_context_composition_v1=_composition_manifest()),
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["method"] == "utf8_heuristic_v2"


def test_toolkit_tool_schema_is_measured_end_to_end_to_reconciled_estimate(
    tmp_path,
    monkeypatch,
):
    """P1 regression: every other test in this module hand-builds its
    manifest via _composition_manifest(), bypassing
    build_internal_context_composition entirely. This is the one place a
    REAL toolkit goes through the real builder and the real wire-match, the
    only path that actually proves a tool-bearing call reaches
    reconciled_estimate instead of the "partial forever" quality tools used
    to be stuck at."""
    state = RunState()
    state.seed_messages([{"role": "user", "content": "look something up"}])
    state.provider_state.provider = "openai"
    state.iteration = 2
    toolkit = Toolkit()

    @toolkit.tool
    def lookup(query: str) -> str:
        return query

    request = build_model_turn_request(
        state,
        toolkit=toolkit,
        run_id=ATTEMPT.attempt_id,
        emit_stream=True,
    )
    [primary_route] = [
        route
        for route in request.internal_context_composition_v1["routes"]
        if route["route_name"] == "primary"
    ]
    assert any(
        item["category"] == "tool_definitions"
        for item in primary_route["contributions"]
    )

    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 1000, "output_tokens": 2, "total_tokens": 1002}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["quality"] == "reconciled_estimate"
    assert extension["coverage"]["status"] == "complete"
    assert extension["residual_tokens"] is not None
    assert extension["residual_tokens"] >= 0
    assert (
        extension["attributed_tokens"] + extension["residual_tokens"] == 1000
    )
    categories = {category["id"] for category in extension["categories"]}
    assert "tool_definitions" in categories


def test_declared_tool_schema_absent_from_the_actual_wire_stays_partial(
    tmp_path,
    monkeypatch,
):
    """Negative twin of the test above: the manifest declares a tool_schema
    contribution (a real toolkit was attached) but the model_io actually
    used to physically send strips tools off the wire (capability gate). The
    matcher must not credit a surface that never made it onto the request —
    coverage stays partial rather than quietly rounding up to complete."""
    state = RunState()
    state.seed_messages([{"role": "user", "content": "look something up"}])
    state.provider_state.provider = "openai"
    state.iteration = 2
    toolkit = Toolkit()

    @toolkit.tool
    def lookup(query: str) -> str:
        return query

    request = build_model_turn_request(
        state,
        toolkit=toolkit,
        run_id=ATTEMPT.attempt_id,
        emit_stream=True,
    )

    turn = replace(
        _turn_result(),
        provider_call_usage=ProviderCallUsage.from_openai_usage(
            {"input_tokens": 1000, "output_tokens": 2, "total_tokens": 1002}
        ),
    )
    transport = _SequenceTransport([turn])
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )
    # supports_tools=False forces openai.py to omit "tools" from the wire
    # kwargs even though the toolkit and the manifest both carry one. Keyed
    # by model name — _model_capability resolves it via _resolve_model_key,
    # a flat dict here would silently no-op and defeat the whole test.
    model_io = OpenAIModelIO(
        model="gpt-test",
        api_key="test-key",
        client_factory=lambda **_kwargs: SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kw: _OpenAIStream())
        ),
        default_payloads={},
        model_capabilities={"gpt-test": {"supports_tools": False}},
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=model_io,
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["quality"] == "partial"
    assert extension["residual_tokens"] is None
    assert extension["coverage"]["status"] == "partial"
    assert extension["coverage"]["matched_surfaces"] < extension["coverage"]["wire_surfaces"]


def test_fallback_receipts_use_distinct_physical_ordinals_and_exact_routes(
    tmp_path,
    monkeypatch,
):
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    fallback_failure = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
        RuntimeError("remote state unavailable"),
    )
    transport = _SequenceTransport(
        [fallback_failure, _turn_result("local replay")]
    )
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )
    request = replace(
        _request(),
        previous_response_id="response-previous",
        fallback_messages=[{"role": "user", "content": "local replay"}],
        context_mode="remote_continuation",
        internal_context_composition_v1=_composition_manifest(fallback=True),
    )

    result = service.fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    receipts = service.store.load_receipts(
        root_run_id=ATTEMPT.attempt_id,
        owner_run_id=ATTEMPT.attempt_id,
        attempt_id=ATTEMPT.attempt_id,
    )

    assert transport.calls == [
        ("primary", 0),
        ("openai_previous_response_fallback", 0),
    ]
    receipts = sorted(receipts, key=lambda item: item.identity.retry_ordinal)
    assert [receipt.identity.retry_ordinal for receipt in receipts] == [0, 1]
    assert len({receipt.provider_call_id for receipt in receipts}) == 2
    assert [
        receipt.extensions[CONTEXT_COMPOSITION_EXTENSION_KEY]["wire"][
            "route_name"
        ]
        for receipt in receipts
    ] == ["primary", "openai_previous_response_fallback"]
    assert result.provider_call_receipt.identity.retry_ordinal == 1


def test_cold_retry_reopens_with_the_same_physical_receipt_identity(
    tmp_path,
    monkeypatch,
):
    from unchain.context.provider_execution import (
        ContextProviderTurnExecutionService,
    )

    transient = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("temporary before process restart"),
    )
    first_transport = _SequenceTransport([transient])
    selected_transport = {"value": first_transport}
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: selected_transport["value"],
    )

    class ProcessRestart(RuntimeError):
        pass

    def restart_during_backoff(_seconds):
        raise ProcessRestart("restart during backoff")

    request = replace(
        _request(),
        internal_context_composition_v1=_composition_manifest(),
    )
    first_service = ContextProviderTurnExecutionService(
        attempt=ATTEMPT,
        store=_repository(tmp_path),
        mode=DurableProviderTurnMode.ENFORCE_TEST,
        transport_target_sha256=TARGET_SHA256,
        sleep=restart_during_backoff,
    )
    retry_config = RetryConfig(
        max_retries=1,
        base_delay_ms=0,
        max_delay_ms=0,
        jitter_ratio=0,
    )

    with pytest.raises(ProcessRestart, match="restart during backoff"):
        first_service.fetch_prepared(
            model_io=_model_io([]),
            request=request,
            retry_config=retry_config,
            run_receipt_factory=_run_receipt_factory(),
        )

    second_transport = _SequenceTransport([_turn_result("cold retry")])
    selected_transport["value"] = second_transport
    second_service = ContextProviderTurnExecutionService(
        attempt=ATTEMPT,
        store=_repository(tmp_path),
        mode=DurableProviderTurnMode.ENFORCE_TEST,
        transport_target_sha256=TARGET_SHA256,
        sleep=lambda _seconds: None,
    )
    result = second_service.fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=retry_config,
        run_receipt_factory=_run_receipt_factory(),
    )
    receipts = sorted(
        second_service.store.load_receipts(
            root_run_id=ATTEMPT.attempt_id,
            owner_run_id=ATTEMPT.attempt_id,
            attempt_id=ATTEMPT.attempt_id,
        ),
        key=lambda item: item.identity.retry_ordinal,
    )

    assert first_transport.calls == [("primary", 0)]
    assert second_transport.calls == [("primary", 1)]
    assert [receipt.identity.retry_ordinal for receipt in receipts] == [0, 1]
    assert len({receipt.provider_call_id for receipt in receipts}) == 2
    assert result.provider_call_receipt == receipts[1]

    replay_transport = _SequenceTransport([])
    selected_transport["value"] = replay_transport
    replay_service = ContextProviderTurnExecutionService(
        attempt=ATTEMPT,
        store=_repository(tmp_path),
        mode=DurableProviderTurnMode.ENFORCE_TEST,
        transport_target_sha256=TARGET_SHA256,
        sleep=lambda _seconds: None,
    )
    replay = replay_service.fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=retry_config,
        run_receipt_factory=_run_receipt_factory(),
    )

    assert replay.final_text == "cold retry"
    assert replay.provider_call_receipt is None
    assert replay_transport.calls == []


def test_base_receipt_factory_error_is_never_downgraded_by_composition(tmp_path):
    class BaseReceiptFailure(RuntimeError):
        pass

    request = replace(
        _request(),
        internal_context_composition_v1=_composition_manifest(),
    )

    def fail_factory(*_args):
        raise BaseReceiptFailure("base receipt failed")

    with pytest.raises(BaseReceiptFailure, match="base receipt failed"):
        _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
            model_io=_model_io([]),
            request=request,
            retry_config=RetryConfig(max_retries=0),
            run_receipt_factory=fail_factory,
        )


def test_base_receipt_physical_identity_mismatch_fails_before_ledger_append(
    tmp_path,
):
    base_factory = _run_receipt_factory()

    def changed_factory(*args):
        receipt = base_factory(*args)
        return replace(
            receipt,
            identity=replace(
                receipt.identity,
                retry_ordinal=receipt.identity.retry_ordinal + 1,
            ),
            extensions=dict(receipt.extensions),
            provider_call_id="",
        )

    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    with pytest.raises(RuntimeError, match="physical send subject"):
        service.fetch_prepared(
            model_io=_model_io([]),
            request=replace(
                _request(),
                internal_context_composition_v1=_composition_manifest(),
            ),
            retry_config=RetryConfig(max_retries=0),
            run_receipt_factory=changed_factory,
        )

    assert service.store.load_receipts(
        root_run_id=ATTEMPT.attempt_id,
        owner_run_id=ATTEMPT.attempt_id,
        attempt_id=ATTEMPT.attempt_id,
    ) == ()


@pytest.mark.parametrize(
    "transport_error",
    [
        ExactProviderRouteFailure(
            ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
            RuntimeError("temporary"),
        ),
        RuntimeError("connection state unknown"),
    ],
)
def test_failed_or_uncertain_receipt_identity_mismatch_never_reaches_ledger(
    tmp_path,
    monkeypatch,
    transport_error,
):
    base_factory = _run_receipt_factory()

    def changed_factory(*args):
        receipt = base_factory(*args)
        return replace(
            receipt,
            identity=replace(
                receipt.identity,
                attempt_id="conflicting-attempt",
            ),
            extensions=dict(receipt.extensions),
            provider_call_id="",
        )

    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    transport = _FailingTransport(transport_error)
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )
    observed = []

    with pytest.raises(RuntimeError, match="physical send subject"):
        service.fetch_prepared(
            model_io=_model_io([]),
            request=replace(
                _request(),
                internal_context_composition_v1=_composition_manifest(),
            ),
            retry_config=RetryConfig(max_retries=0),
            run_receipt_factory=changed_factory,
            run_receipt_observed=observed.append,
        )

    assert transport.calls == 1
    assert observed == []
    assert service.store.load_receipts(
        root_run_id=ATTEMPT.attempt_id,
        owner_run_id=ATTEMPT.attempt_id,
        attempt_id=ATTEMPT.attempt_id,
    ) == ()


def test_known_uninstrumented_wire_sources_force_partial_composition(tmp_path):
    toolkit = Toolkit()

    @toolkit.tool
    def lookup(query: str) -> str:
        return query

    manifest = _composition_manifest()
    manifest["routes"][0]["manifest_items"] = 4
    manifest["routes"][0]["wire_surfaces"] = 2
    request = replace(
        _request(),
        messages=[
            {"role": "system", "content": "system rules"},
            {"role": "user", "content": "request"},
        ],
        toolkit=toolkit,
        internal_context_composition_v1=manifest,
    )

    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )
    extension = result.provider_call_receipt.extensions[
        CONTEXT_COMPOSITION_EXTENSION_KEY
    ]

    assert extension["quality"] == "partial"
    assert extension["residual_tokens"] is None
    assert extension["coverage"] == {
        "status": "partial",
        "manifest_items": 4,
        "matched_items": 1,
        "wire_surfaces": 2,
        "matched_surfaces": 1,
    }


@pytest.mark.parametrize("with_manifest", [False, True])
def test_reserved_composition_extension_is_owned_only_by_official_enrichment(
    tmp_path,
    with_manifest,
):
    base_factory = _run_receipt_factory()

    def injected_factory(*args):
        receipt = base_factory(*args)
        return replace(
            receipt,
            extensions={
                **dict(receipt.extensions),
                CONTEXT_COMPOSITION_EXTENSION_KEY: {
                    "schema": "mutant.context_composition.v0",
                    "content": "must-not-persist",
                    "label": "mutant",
                },
            },
        )

    request = _request()
    if with_manifest:
        request = replace(
            request,
            internal_context_composition_v1=_composition_manifest(),
        )
    result = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST).fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=injected_factory,
    )
    extensions = result.provider_call_receipt.extensions

    if with_manifest:
        official = extensions[CONTEXT_COMPOSITION_EXTENSION_KEY]
        assert official["schema"] == "unchain.context/context_composition_v1"
        assert "content" not in official
        assert "label" not in official
    else:
        assert CONTEXT_COMPOSITION_EXTENSION_KEY not in extensions


@pytest.mark.parametrize("failure_type", [TypeError, RuntimeError])
def test_composition_failure_preserves_the_valid_base_receipt(
    tmp_path,
    monkeypatch,
    caplog,
    failure_type,
):
    """P3 regression: enrich_provider_call_receipt's except block used to
    swallow this failure with zero trace. It must now log exactly once, and
    that one line must carry only the exception's type name — never the raw
    exception message, which can (as constructed here) hold receipt-adjacent
    content that must not reach logs."""
    secret_marker = "composition-only-failure-CONTENT-MUST-NOT-LOG"

    def fail_composition(**_kwargs):
        raise failure_type(secret_marker)

    monkeypatch.setattr(
        "unchain.context.composition.build_context_composition_extension",
        fail_composition,
    )
    request = replace(
        _request(),
        internal_context_composition_v1=_composition_manifest(),
    )

    with caplog.at_level("WARNING", logger="unchain.context.composition"):
        result = _service(
            tmp_path,
            DurableProviderTurnMode.ENFORCE_TEST,
        ).fetch_prepared(
            model_io=_model_io([]),
            request=request,
            retry_config=RetryConfig(max_retries=0),
            run_receipt_factory=_run_receipt_factory(),
        )

    assert result.final_text == "durable result"
    assert (
        CONTEXT_COMPOSITION_EXTENSION_KEY
        not in result.provider_call_receipt.extensions
    )

    composition_records = [
        r for r in caplog.records if r.name == "unchain.context.composition"
    ]
    assert len(composition_records) == 1
    logged_message = composition_records[0].getMessage()
    assert failure_type.__name__ in logged_message
    assert secret_marker not in logged_message


def test_malformed_composition_builder_output_never_reaches_the_receipt(
    tmp_path,
    monkeypatch,
):
    private_marker = "PRIVATE_RAW_MUST_NOT_PERSIST"
    monkeypatch.setattr(
        "unchain.context.composition.build_context_composition_extension",
        lambda **_kwargs: {
            "schema": "mutant.context.v0",
            "content": private_marker,
        },
    )
    request = replace(
        _request(),
        internal_context_composition_v1=_composition_manifest(),
    )

    result = _service(
        tmp_path,
        DurableProviderTurnMode.ENFORCE_TEST,
    ).fetch_prepared(
        model_io=_model_io([]),
        request=request,
        retry_config=RetryConfig(max_retries=0),
        run_receipt_factory=_run_receipt_factory(),
    )

    assert result.final_text == "durable result"
    assert (
        CONTEXT_COMPOSITION_EXTENSION_KEY
        not in result.provider_call_receipt.extensions
    )
    assert private_marker not in str(result.provider_call_receipt.extensions)
