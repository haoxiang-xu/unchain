from __future__ import annotations

import hashlib
import json
import traceback

import pytest

from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.durability import is_durable_persistence_failure
from unchain.journal.models import AttemptRef, GenerationRef, OperationRef
from unchain.journal.provider_result import recover_provider_turn_result
from unchain.journal.provider_wire import (
    persist_provider_wire_snapshot,
    recover_provider_wire_authority,
)
from unchain.kernel.types import ModelTurnResult
from unchain.kernel.run_ledger import build_model_attempt_receipt
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store
from unchain.providers.durable_turn_runtime import (
    DurableProviderTurnError,
    DurableProviderTurnMode,
    DurableProviderTurnRuntime,
    DurableProviderTurnStatus,
    DurableProviderTurnTerminalError,
    DurableProviderTurnUncertainError,
    ExactProviderRouteFailure,
    ExactProviderRouteFailureKind,
    ExactProviderRouteTransport,
)
from unchain.providers.request_lease import (
    ProviderRequestLeaseCoordinator,
    ProviderRequestStatus,
    ProviderRequestSubject,
)
from unchain.providers.wire_envelope import ProviderWireEnvelope, ProviderWireRoute
from unchain.providers.uncertainty_diagnostic import ProviderUncertaintyDiagnostic
from unchain.retry import RetryConfig, RetriesExhaustedError
from unchain.run_bundle import RunIdentity


ATTEMPT = AttemptRef(
    GenerationRef("execution-durable-turn", "generation-durable-turn"),
    "attempt-durable-turn",
)
ITERATION = 4


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _operation(name: str, digit: str) -> OperationRef:
    return OperationRef(name, digit * 64)


def _store(tmp_path) -> SQLiteContextV2Store:
    return SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )


def _catalog() -> ToolCatalogEnvelope:
    return ToolCatalogEnvelope(
        attempt=ATTEMPT,
        iteration=ITERATION,
        provider="openai",
        model="frontier-model",
        semantic_schemas=[],
        entries=[],
        required_betas_sha256=_json_sha256([]),
        prompt_sha256="0" * 64,
        exposure_plan_sha256="1" * 64,
    )


def _envelope(
    catalog: ToolCatalogEnvelope,
    *,
    fallback: bool = False,
) -> ProviderWireEnvelope:
    common = {
        "model": "frontier-model",
        "stream": True,
        "store": False,
    }
    primary = {
        **common,
        "input": [{"role": "user", "content": "hello"}],
    }
    routes = [ProviderWireRoute(name="primary", request=primary)]
    if fallback:
        primary["previous_response_id"] = "response-before-restart"
        routes = [
            ProviderWireRoute(name="primary", request=primary),
            ProviderWireRoute(
                name="openai_previous_response_fallback",
                request={
                    **common,
                    "input": [{"role": "user", "content": "complete replay"}],
                },
            ),
        ]
    return ProviderWireEnvelope(
        attempt=ATTEMPT,
        iteration=ITERATION,
        provider="openai",
        configured_model="frontier-model",
        request_model="frontier-model",
        adapter_revision="unchain.openai.responses.request.v1",
        transport_kind="openai.responses.create",
        transport_target_sha256="2" * 64,
        source_request_sha256="3" * 64,
        source_payload_sha256="4" * 64,
        catalog_sha256=catalog.catalog_sha256,
        prompt_sha256=catalog.prompt_sha256,
        tool_schema_sha256=catalog.tool_schema_sha256,
        required_betas=(),
        base_anthropic_betas=(),
        routes=routes,
    )


def _authority(tmp_path, *, fallback: bool = False):
    store = _store(tmp_path)
    repository = store.bind_execution(ATTEMPT.generation.execution_id)
    catalog = _catalog()
    envelope = _envelope(catalog, fallback=fallback)
    receipt = persist_provider_wire_snapshot(
        repository,
        envelope=envelope,
        catalog=catalog,
        artifact_operation=_operation("durable-turn-wire-artifact", "8"),
        event_operation=_operation("durable-turn-wire-event", "9"),
        event_id="durable-turn-wire-snapshot",
        expected_artifact_revision=0,
    )
    authority = recover_provider_wire_authority(
        repository,
        attempt=ATTEMPT,
        iteration=ITERATION,
        catalog=catalog,
        expected_provider="openai",
        expected_adapter_revision=envelope.adapter_revision,
        expected_envelope_sha256=envelope.envelope_sha256,
        expected_artifact=receipt.artifact,
        expected_cursor=receipt.cursor,
    )
    return store, repository, authority


def _result(text: str = "durable result") -> ModelTurnResult:
    return ModelTurnResult(
        assistant_messages=[{"role": "assistant", "content": text}],
        tool_calls=[],
        final_text=text,
        response_id="response-durable-turn",
        consumed_tokens=3,
        input_tokens=2,
        output_tokens=1,
    )


def _receipt_factory():
    identity = RunIdentity(
        execution_id=ATTEMPT.generation.execution_id,
        attempt_id=ATTEMPT.attempt_id,
        root_run_id=ATTEMPT.attempt_id,
        run_id=ATTEMPT.attempt_id,
        parent_run_id=None,
        relation="root",
    )

    def build(send_context, completed_at, outcome, classification, result):
        return build_model_attempt_receipt(
            identity=identity,
            provider="openai",
            model="frontier-model",
            iteration=ITERATION,
            retry_ordinal=send_context.physical_ordinal,
            purpose="agent_turn",
            request_digest="d" * 64,
            route="openai.responses.create",
            started_at="2026-08-13T18:00:00Z",
            completed_at=completed_at,
            turn=result,
            status=outcome,
            classification=classification,
        )

    return build


class _Transport(ExactProviderRouteTransport):
    def __init__(self, outcomes, *, before_send=None) -> None:
        self.outcomes = list(outcomes)
        self.calls = []
        self.before_send = before_send

    def send(self, *, envelope, route, retry_ordinal):
        if self.before_send is not None:
            self.before_send(envelope, route, retry_ordinal)
        self.calls.append((envelope, route, retry_ordinal))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _runtime(
    repository, transport, *, mode=DurableProviderTurnMode.ENFORCE_TEST, sleep=None
):
    return DurableProviderTurnRuntime(
        mode=mode,
        store=repository,
        transport=transport,
        sleep=(sleep if sleep is not None else lambda _seconds: None),
    )


def test_terminal_http_diagnostic_survives_real_adapter_and_sqlite_reopen(tmp_path):
    import httpx
    from openai import BadRequestError
    from unchain.providers.openai import OpenAIModelIO
    from unchain.providers.exact_route_transport import OpenAIExactRouteTransport
    from unchain.providers.durable_turn_runtime import DurableProviderTurnTerminalError

    _, repository, authority = _authority(tmp_path)
    secret = "PRIVATE_DIAGNOSTIC_SENTINEL"
    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            raise BadRequestError(
                secret,
                response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid/v1/responses")),
                body={"code": "invalid_function_parameters", "param": "tools[3].parameters", "message": secret},
            )

    class Client:
        responses = Responses()

    transport = OpenAIExactRouteTransport(
        model_io=OpenAIModelIO(model="frontier-model", api_key=secret,
            client_factory=lambda **kwargs: Client(), default_payloads={}, model_capabilities={}),
        catalog=_catalog(),
    )
    with pytest.raises(DurableProviderTurnTerminalError) as first:
        _runtime(repository, transport).execute(authority=authority, retry_config=RetryConfig(max_retries=0))
    assert "HTTP 400" in str(first.value)
    assert "invalid_function_parameters" in str(first.value)
    assert "tools[3].parameters" in str(first.value)
    assert secret not in str(first.value)
    import traceback
    assert secret not in "".join(traceback.format_exception(first.value))
    assert len(calls) == 1
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnTerminalError) as recovered:
        _runtime(reopened, no_send).execute(authority=authority, retry_config=RetryConfig(max_retries=0))
    assert str(recovered.value) == str(first.value)
    assert no_send.calls == []
    assert secret.encode() not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


def test_diagnostic_lease_schema_keeps_legacy_bytes_and_rejects_corruption(tmp_path):
    from unchain.providers.request_lease import ProviderRequestLease
    from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic
    from dataclasses import replace

    _, repository, authority = _authority(tmp_path)
    subject = ProviderRequestSubject(ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 0)
    coordinator = ProviderRequestLeaseCoordinator(repository)
    started = coordinator.claim_initial(attempt=ATTEMPT, iteration=ITERATION,
        envelope_sha256=subject.envelope_sha256, route="primary",
        route_sha256=authority.envelope.routes[0].route_sha256,
        operation=_operation("diagnostic-legacy-claim", "6"))
    old_bytes = json.dumps(started.to_dict(), sort_keys=True)
    assert started.to_dict()["schema"] == "unchain.provider_request_lease.v2"
    assert json.dumps(ProviderRequestLease.from_durable_dict(started.to_dict()).to_dict(), sort_keys=True) == old_bytes
    failed = coordinator.record_failure(started, classification="non_retryable",
        retryable=False, visible_output=False, operation=_operation("diagnostic-failure", "5"),
        failure_diagnostic=ProviderFailureDiagnostic(401, "invalid_api_key"))
    assert failed.to_dict()["schema"] == "unchain.provider_request_lease.v3"
    assert ProviderRequestLease.from_durable_dict(failed.to_dict()) == failed
    with pytest.raises(ValueError):
        replace(started, failure_diagnostic=failed.failure_diagnostic)
    for change in ({"failure_diagnostic": None}, {"unexpected": True}, {"schema": "unchain.provider_request_lease.v2"}):
        value = failed.to_dict()
        value.update(change)
        with pytest.raises((TypeError, ValueError)):
            ProviderRequestLease.from_durable_dict(value)


@pytest.mark.parametrize(
    ("mode", "status"),
    (
        (DurableProviderTurnMode.OFF, DurableProviderTurnStatus.BYPASSED),
        (DurableProviderTurnMode.SHADOW, DurableProviderTurnStatus.SHADOWED),
    ),
)
def test_closed_modes_never_claim_or_send(tmp_path, mode, status) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    outcome = _runtime(repository, transport, mode=mode).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
    )

    assert outcome.status is status
    assert outcome.result is None
    assert transport.calls == []
    subject = ProviderRequestSubject(
        ATTEMPT,
        ITERATION,
        authority.envelope.envelope_sha256,
        "primary",
        0,
    )
    assert repository.load(subject=subject) is None


@pytest.mark.parametrize(
    "mode",
    (DurableProviderTurnMode.OFF, DurableProviderTurnMode.SHADOW),
)
@pytest.mark.parametrize("evidence", ("started", "completed"))
def test_closed_modes_fail_when_exact_subject_has_enforce_evidence(
    tmp_path,
    mode,
    evidence,
) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    route = authority.envelope.routes[0]
    if evidence == "started":
        ProviderRequestLeaseCoordinator(repository).claim_initial(
            attempt=ATTEMPT,
            iteration=ITERATION,
            envelope_sha256=authority.envelope.envelope_sha256,
            route=route.name,
            route_sha256=route.route_sha256,
            operation=_operation("mode-downgrade-started", "7"),
        )
    else:
        _runtime(repository, _Transport([_result()])).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=0),
        )

    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnError, match="durable evidence"):
        _runtime(repository, no_send, mode=mode).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=0),
        )

    assert no_send.calls == []


def test_success_persists_result_and_restart_reuses_it_without_network(
    tmp_path,
) -> None:
    store, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    first = _runtime(repository, transport).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
    )

    assert first.status is DurableProviderTurnStatus.COMPLETED
    assert first.result.final_text == "durable result"
    assert first.recovered is False
    assert [(call[1].name, call[2]) for call in transport.calls] == [("primary", 0)]

    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    recovered = _runtime(reopened, no_send).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
    )

    assert recovered.status is DurableProviderTurnStatus.COMPLETED
    assert recovered.result == first.result
    assert recovered.recovered is True
    assert no_send.calls == []
    assert store.database_path.exists()


def test_retry_safe_failure_gets_a_new_lease_before_the_second_send(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    failure = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("temporary"),
    )
    started_before_send = []

    def assert_started(envelope, route, retry_ordinal):
        subject = ProviderRequestSubject(
            ATTEMPT,
            ITERATION,
            envelope.envelope_sha256,
            route.name,
            retry_ordinal,
        )
        lease = repository.load(subject=subject)
        assert lease.status is ProviderRequestStatus.STARTED
        assert lease.route_sha256 == route.route_sha256
        started_before_send.append(subject)

    transport = _Transport(
        [failure, _result("after retry")],
        before_send=assert_started,
    )
    sleeps = []

    outcome = _runtime(repository, transport, sleep=sleeps.append).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=1,
            base_delay_ms=1,
            max_delay_ms=1,
            jitter_ratio=0,
        ),
    )

    assert outcome.result.final_text == "after retry"
    assert [(call[1].name, call[2]) for call in transport.calls] == [
        ("primary", 0),
        ("primary", 1),
    ]
    assert [subject.retry_ordinal for subject in started_before_send] == [0, 1]
    assert sleeps == [0.001]
    first_subject = ProviderRequestSubject(
        ATTEMPT,
        ITERATION,
        authority.envelope.envelope_sha256,
        "primary",
        0,
    )
    second_subject = ProviderRequestSubject(
        ATTEMPT,
        ITERATION,
        authority.envelope.envelope_sha256,
        "primary",
        1,
    )
    assert repository.load(subject=first_subject).status is ProviderRequestStatus.FAILED
    assert (
        repository.load(subject=second_subject).status
        is ProviderRequestStatus.COMPLETED
    )
    no_send = _Transport([])
    replay = _runtime(repository, no_send).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=0,
            base_delay_ms=0,
            max_delay_ms=0,
            jitter_ratio=0,
        ),
    )
    assert replay.result == outcome.result
    assert replay.recovered is True
    assert no_send.calls == []


def test_cold_retry_preserves_the_physical_ordinal_after_restart(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transient = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("temporary before restart"),
    )
    first_transport = _Transport([transient])
    first_contexts = []

    class ProcessRestart(RuntimeError):
        pass

    def restart_during_backoff(_seconds):
        raise ProcessRestart("restart during retry backoff")

    with pytest.raises(ProcessRestart, match="restart during retry backoff"):
        _runtime(
            repository,
            first_transport,
            sleep=restart_during_backoff,
        ).execute(
            authority=authority,
            retry_config=RetryConfig(
                max_retries=1,
                base_delay_ms=0,
                max_delay_ms=0,
                jitter_ratio=0,
            ),
            after_send=lambda context, *_rest: first_contexts.append(context),
        )

    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    second_transport = _Transport([_result("cold retry")])
    second_contexts = []
    outcome = _runtime(reopened, second_transport).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=1,
            base_delay_ms=0,
            max_delay_ms=0,
            jitter_ratio=0,
        ),
        after_send=lambda context, *_rest: second_contexts.append(context),
        build_run_receipt=_receipt_factory(),
    )

    assert [(call[1].name, call[2]) for call in first_transport.calls] == [
        ("primary", 0)
    ]
    assert [(call[1].name, call[2]) for call in second_transport.calls] == [
        ("primary", 1)
    ]
    assert [context.physical_ordinal for context in first_contexts] == [0]
    assert [context.physical_ordinal for context in second_contexts] == [1]
    assert outcome.run_receipt.identity.retry_ordinal == 1

    no_send = _Transport([])
    replay = _runtime(reopened, no_send).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=1),
    )
    assert replay.result.final_text == "cold retry"
    assert replay.recovered is True
    assert no_send.calls == []


def test_fallback_retry_uses_consecutive_physical_ordinals(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path, fallback=True)
    primary_fallback = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
        RuntimeError("remote continuation unavailable"),
    )
    fallback_transient = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("local replay transport is temporarily unavailable"),
    )
    transport = _Transport(
        [primary_fallback, fallback_transient, _result("fallback retry")]
    )
    contexts = []

    outcome = _runtime(repository, transport).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=1,
            base_delay_ms=0,
            max_delay_ms=0,
            jitter_ratio=0,
        ),
        after_send=lambda context, *_rest: contexts.append(context),
        build_run_receipt=_receipt_factory(),
    )

    assert [(call[1].name, call[2]) for call in transport.calls] == [
        ("primary", 0),
        ("openai_previous_response_fallback", 0),
        ("openai_previous_response_fallback", 1),
    ]
    assert [context.physical_ordinal for context in contexts] == [0, 1, 2]
    assert outcome.run_receipt.identity.retry_ordinal == 2


def test_primary_retry_cannot_transition_to_fallback(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path, fallback=True)
    transient = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("retry primary once"),
    )
    late_fallback = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
        RuntimeError("late fallback is ambiguous"),
    )
    transport = _Transport([transient, late_fallback])

    with pytest.raises(BaseException) as caught:
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(
                max_retries=1,
                base_delay_ms=0,
                max_delay_ms=0,
                jitter_ratio=0,
            ),
        )

    assert is_durable_persistence_failure(caught.value)
    assert [(call[1].name, call[2]) for call in transport.calls] == [
        ("primary", 0),
        ("primary", 1),
    ]


def test_before_send_runs_immediately_before_every_transport_send(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    failure = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("temporary"),
    )
    events = []

    def record_transport_send(_envelope, route, retry_ordinal):
        events.append(("send", route.name, retry_ordinal))

    transport = _Transport(
        [failure, _result("after retry")],
        before_send=record_transport_send,
    )

    outcome = _runtime(repository, transport).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=1,
            base_delay_ms=0,
            max_delay_ms=0,
            jitter_ratio=0,
        ),
        before_send=lambda context: events.append(
            ("guard", context.physical_ordinal)
        ),
    )

    assert outcome.result.final_text == "after retry"
    assert events == [
        ("guard", 0),
        ("send", "primary", 0),
        ("guard", 1),
        ("send", "primary", 1),
    ]


def test_before_send_failure_leaves_started_lease_without_sending(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    def fail_guard(_context):
        raise RuntimeError("execution lease lost")

    with pytest.raises(DurableProviderTurnUncertainError):
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=0),
            before_send=fail_guard,
        )

    assert transport.calls == []
    subject = ProviderRequestSubject(
        ATTEMPT,
        ITERATION,
        authority.envelope.envelope_sha256,
        "primary",
        0,
    )
    assert repository.load(subject=subject).status is ProviderRequestStatus.STARTED


def test_unclassified_transport_failure_is_uncertain_and_never_resent(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([RuntimeError("unknown network outcome")])

    with pytest.raises(DurableProviderTurnUncertainError):
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=3),
        )
    assert len(transport.calls) == 1

    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError):
        _runtime(repository, no_send).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=3),
        )
    assert no_send.calls == []


def test_openai_previous_response_fallback_is_a_separate_durable_send(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path, fallback=True)
    fallback = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
        RuntimeError("previous response is unavailable"),
    )
    transport = _Transport([fallback, _result("local replay")])

    outcome = _runtime(repository, transport).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
    )

    assert outcome.result.final_text == "local replay"
    assert [(call[1].name, call[2]) for call in transport.calls] == [
        ("primary", 0),
        ("openai_previous_response_fallback", 0),
    ]
    primary = repository.load(
        subject=ProviderRequestSubject(
            ATTEMPT,
            ITERATION,
            authority.envelope.envelope_sha256,
            "primary",
            0,
        )
    )
    fallback_lease = repository.load(
        subject=ProviderRequestSubject(
            ATTEMPT,
            ITERATION,
            authority.envelope.envelope_sha256,
            "openai_previous_response_fallback",
            0,
        )
    )
    assert primary.classification == "previous_response_fallback"
    assert fallback_lease.status is ProviderRequestStatus.COMPLETED
    no_send = _Transport([])
    replay = _runtime(repository, no_send).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
    )
    assert replay.result == outcome.result
    assert replay.recovered is True
    assert no_send.calls == []


@pytest.mark.parametrize("recovered_primary_failure", (False, True))
def test_before_send_runs_for_fresh_and_recovered_openai_fallback(
    tmp_path,
    recovered_primary_failure,
) -> None:
    _store_value, repository, authority = _authority(tmp_path, fallback=True)
    fallback_failure = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
        RuntimeError("previous response is unavailable"),
    )
    outcomes = [fallback_failure, _result("local replay")]
    if recovered_primary_failure:
        primary_route = authority.envelope.routes[0]
        coordinator = ProviderRequestLeaseCoordinator(repository)
        started = coordinator.claim_initial(
            attempt=ATTEMPT,
            iteration=ITERATION,
            envelope_sha256=authority.envelope.envelope_sha256,
            route=primary_route.name,
            route_sha256=primary_route.route_sha256,
            operation=_operation("fallback-primary-started", "5"),
        )
        coordinator.record_failure(
            started,
            classification="previous_response_fallback",
            retryable=True,
            visible_output=False,
            operation=_operation("fallback-primary-failed", "6"),
        )
        outcomes = [_result("local replay")]

    events = []

    def record_transport_send(_envelope, route, retry_ordinal):
        events.append(("send", route.name, retry_ordinal))

    transport = _Transport(outcomes, before_send=record_transport_send)
    outcome = _runtime(repository, transport).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
        before_send=lambda context: events.append(
            ("guard", context.physical_ordinal)
        ),
    )

    assert outcome.result.final_text == "local replay"
    if recovered_primary_failure:
        assert events == [
            ("guard", 1),
            ("send", "openai_previous_response_fallback", 0),
        ]
    else:
        assert events == [
            ("guard", 0),
            ("send", "primary", 0),
            ("guard", 1),
            ("send", "openai_previous_response_fallback", 0),
        ]


def test_result_persistence_failure_is_durable_and_never_retried(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    def fail_result_persistence(*, request):
        del request
        raise OSError("disk full")

    repository._persist_provider_turn_result_cas = fail_result_persistence
    with pytest.raises(BaseException) as caught:
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=3),
        )

    assert is_durable_persistence_failure(caught.value)
    assert len(transport.calls) == 1
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError):
        _runtime(repository, no_send).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=3),
        )
    assert no_send.calls == []


def test_restart_finalizes_receipted_started_lease_without_resending(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result("persisted before crash")])
    original_cas = repository.compare_and_swap

    def fail_completion(*, subject, expected_revision, replacement):
        if replacement.status is ProviderRequestStatus.COMPLETED:
            raise OSError("completion lease write failed")
        return original_cas(
            subject=subject,
            expected_revision=expected_revision,
            replacement=replacement,
        )

    repository.compare_and_swap = fail_completion
    with pytest.raises(BaseException) as caught:
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=0),
            build_run_receipt=_receipt_factory(),
        )
    assert is_durable_persistence_failure(caught.value)
    assert len(transport.calls) == 1

    durable_run_receipts = repository.load_receipts(
        root_run_id=ATTEMPT.attempt_id,
        owner_run_id=ATTEMPT.attempt_id,
        attempt_id=ATTEMPT.attempt_id,
    )
    assert len(durable_run_receipts) == 1
    assert durable_run_receipts[0].status == "completed"

    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    outcome = _runtime(reopened, _Transport([])).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
    )

    assert outcome.result.final_text == "persisted before crash"
    assert outcome.recovered is True
    subject = ProviderRequestSubject(
        ATTEMPT,
        ITERATION,
        authority.envelope.envelope_sha256,
        "primary",
        0,
    )
    completed = reopened.load(subject=subject)
    receipt = recover_provider_turn_result(
        reopened,
        subject=subject,
        expected_route_sha256=completed.route_sha256,
    )
    assert completed.result_binding.result_sha256 == receipt.envelope.result_sha256


def test_exhausted_retry_is_durable_terminal_and_not_sent_after_restart(
    tmp_path,
) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    failures = [
        ExactProviderRouteFailure(
            ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
            RuntimeError(f"temporary-{index}"),
        )
        for index in range(2)
    ]
    transport = _Transport(failures)

    with pytest.raises(RetriesExhaustedError):
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(
                max_retries=1,
                base_delay_ms=0,
                max_delay_ms=0,
                jitter_ratio=0,
            ),
        )
    assert len(transport.calls) == 2

    no_send = _Transport([])
    with pytest.raises(Exception, match="terminal|transient|failed"):
        _runtime(repository, no_send).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=2),
        )
    assert no_send.calls == []


def test_after_send_closes_retry_attempts_immediately(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    retry_failure = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE,
        RuntimeError("retry safe"),
    )
    events = []
    transport = _Transport(
        [retry_failure, _result("after retry")],
        before_send=lambda _envelope, route, ordinal: events.append(
            ("send", route.name, ordinal)
        ),
    )

    outcome = _runtime(repository, transport).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=1,
            base_delay_ms=0,
            max_delay_ms=0,
            jitter_ratio=0,
        ),
        after_send=lambda context, completed_at, status, classification: events.append(
            ("after", context, completed_at, status, classification)
        ),
    )

    assert outcome.result.final_text == "after retry"
    assert [event[0] for event in events] == [
        "send",
        "after",
        "send",
        "after",
    ]
    after = [event for event in events if event[0] == "after"]
    assert [(event[1].physical_ordinal, event[3], event[4]) for event in after] == [
        (0, "failed", "transient_retry_safe"),
        (1, "completed", "success"),
    ]
    assert all(event[2].endswith("Z") for event in after)


def test_after_send_closes_fallback_and_success_attempts(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path, fallback=True)
    fallback_failure = ExactProviderRouteFailure(
        ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
        RuntimeError("fallback"),
    )
    events = []

    outcome = _runtime(
        repository,
        _Transport([fallback_failure, _result("local replay")]),
    ).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=0),
        after_send=lambda *event: events.append(event),
    )

    assert outcome.result.final_text == "local replay"
    assert [(event[0].physical_ordinal, event[2], event[3]) for event in events] == [
        (0, "failed", "previous_response_fallback"),
        (1, "completed", "success"),
    ]


def test_after_send_closes_uncertain_final_error_once(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    events = []

    with pytest.raises(DurableProviderTurnUncertainError):
        _runtime(repository, _Transport([RuntimeError("unknown")])).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=3),
            after_send=lambda *event: events.append(event),
        )

    assert len(events) == 1
    assert events[0][0].physical_ordinal == 0
    assert events[0][2:] == ("uncertain", "uncertain")


GEMINI_MODEL = "gemini-3.8-flash"


def _gemini_authority(tmp_path):
    """A persisted Gemini wire envelope, recovered the way a real turn does."""

    store = _store(tmp_path)
    repository = store.bind_execution(ATTEMPT.generation.execution_id)
    catalog = ToolCatalogEnvelope(
        attempt=ATTEMPT,
        iteration=ITERATION,
        provider="gemini",
        model=GEMINI_MODEL,
        semantic_schemas=[],
        entries=[],
        required_betas_sha256=_json_sha256([]),
        prompt_sha256="0" * 64,
        exposure_plan_sha256="1" * 64,
    )
    envelope = ProviderWireEnvelope(
        attempt=ATTEMPT,
        iteration=ITERATION,
        provider="gemini",
        configured_model=GEMINI_MODEL,
        request_model=GEMINI_MODEL,
        adapter_revision="unchain.gemini.contents.request.v1",
        transport_kind="gemini.models.generate_content_stream",
        transport_target_sha256="2" * 64,
        source_request_sha256="3" * 64,
        source_payload_sha256="4" * 64,
        catalog_sha256=catalog.catalog_sha256,
        prompt_sha256=catalog.prompt_sha256,
        tool_schema_sha256=catalog.tool_schema_sha256,
        required_betas=(),
        base_anthropic_betas=(),
        routes=[
            ProviderWireRoute(
                name="primary",
                request={
                    "model": GEMINI_MODEL,
                    "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
                    "config": {
                        "automatic_function_calling": {"disable": True},
                        "max_output_tokens": 64,
                    },
                },
            )
        ],
    )
    receipt = persist_provider_wire_snapshot(
        repository,
        envelope=envelope,
        catalog=catalog,
        artifact_operation=_operation("gemini-turn-wire-artifact", "8"),
        event_operation=_operation("gemini-turn-wire-event", "9"),
        event_id="gemini-turn-wire-snapshot",
        expected_artifact_revision=0,
    )
    authority = recover_provider_wire_authority(
        repository,
        attempt=ATTEMPT,
        iteration=ITERATION,
        catalog=catalog,
        expected_provider="gemini",
        expected_adapter_revision=envelope.adapter_revision,
        expected_envelope_sha256=envelope.envelope_sha256,
        expected_artifact=receipt.artifact,
        expected_cursor=receipt.cursor,
    )
    return store, repository, authority, catalog


def _gemini_transport(catalog, outcomes):
    """The real Gemini transport over a fake SDK client.

    Each outcome is either an SDK exception, raised from
    ``generate_content_stream``, or the text of a one-chunk success stream.
    """

    from types import SimpleNamespace

    from google.genai import types
    from unchain.providers.exact_route_transport import GeminiExactRouteTransport
    from unchain.providers.gemini import GeminiModelIO

    queue = list(outcomes)
    calls = []

    def factory(**_kwargs):
        def send(**kwargs):
            calls.append(kwargs["model"])
            outcome = queue.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if isinstance(outcome, dict):
                return iter([types.GenerateContentResponse.model_validate(outcome)])
            return iter(
                [
                    types.GenerateContentResponse.model_validate(
                        {
                            "candidates": [
                                {
                                    "content": {
                                        "role": "model",
                                        "parts": [{"text": outcome}],
                                    },
                                    "finish_reason": "STOP",
                                }
                            ],
                            "usage_metadata": {
                                "prompt_token_count": 3,
                                "candidates_token_count": 2,
                                "total_token_count": 5,
                            },
                        }
                    )
                ]
            )

        return SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=send), close=lambda: None
        )

    transport = GeminiExactRouteTransport(
        model_io=GeminiModelIO(
            model=GEMINI_MODEL, api_key="secret", client_factory=factory
        ),
        catalog=catalog,
    )
    return transport, calls


def _google_error(status: int, provider_status: str, message: str = "private body"):
    import httpx
    from google.genai import errors

    error_type = errors.ServerError if status >= 500 else errors.ClientError
    return error_type(
        status,
        {"error": {"code": status, "message": message, "status": provider_status}},
        httpx.Response(
            status,
            request=httpx.Request("POST", "https://example.invalid/gemini"),
        ),
    )


def test_gemini_503_is_retried_through_the_real_transport_and_completes(tmp_path) -> None:
    """SEQ-386-1 / AC-386-1: a provider 503 is a retry-safe transient failure."""

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(
        catalog, [_google_error(503, "UNAVAILABLE"), "after the overload"]
    )
    sleeps = []

    outcome = _runtime(repository, transport, sleep=sleeps.append).execute(
        authority=authority,
        retry_config=RetryConfig(
            max_retries=2, base_delay_ms=1, max_delay_ms=1, jitter_ratio=0
        ),
    )

    assert outcome.status is DurableProviderTurnStatus.COMPLETED
    assert outcome.result.final_text == "after the overload"
    assert calls == [GEMINI_MODEL, GEMINI_MODEL]
    assert len(sleeps) == 1
    first = repository.load(
        subject=ProviderRequestSubject(
            ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 0
        )
    )
    assert first.status is ProviderRequestStatus.FAILED
    assert first.classification == "transient"
    assert first.retryable is True


def test_v2_diagnostic_lease_uses_v4_and_fails_closed(tmp_path) -> None:
    """BC-386-2 / AC-386-4: v2 diagnostics ride a v4 lease; every mismatch is refused."""

    from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic
    from unchain.providers.request_lease import ProviderRequestLease

    _, repository, authority = _authority(tmp_path)
    coordinator = ProviderRequestLeaseCoordinator(repository)
    started = coordinator.claim_initial(
        attempt=ATTEMPT, iteration=ITERATION,
        envelope_sha256=authority.envelope.envelope_sha256, route="primary",
        route_sha256=authority.envelope.routes[0].route_sha256,
        operation=_operation("v4-lease-claim", "6"),
    )
    diagnostic = ProviderFailureDiagnostic(
        404, provider_status="NOT_FOUND", replacement_model="gemini-3.8-flash"
    )
    failed = coordinator.record_failure(
        started, classification="non_retryable", retryable=False,
        visible_output=False, operation=_operation("v4-lease-failure", "5"),
        failure_diagnostic=diagnostic,
    )
    record = failed.to_dict()
    assert record["schema"] == "unchain.provider_request_lease.v4"
    assert record["failure_diagnostic"]["schema"] == "unchain.provider_failure_diagnostic.v2"
    assert ProviderRequestLease.from_durable_dict(record) == failed
    # The same record read back from SQLite after a cold reopen.
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    assert reopened.load(subject=failed.subject) == failed

    v1_body = ProviderFailureDiagnostic(404).to_dict()
    mismatches = (
        {"schema": "unchain.provider_request_lease.v3"},        # v2 body under the v3 schema
        {"failure_diagnostic": v1_body},                        # v1 body under the v4 schema
        {"failure_diagnostic": None},
        {"unexpected": True},
        {"schema": "unchain.provider_request_lease.v2"},
        {"schema": "unchain.provider_request_lease.v5"},
    )
    for change in mismatches:
        value = failed.to_dict()
        value.update(change)
        with pytest.raises((TypeError, ValueError)):
            ProviderRequestLease.from_durable_dict(value)

    # A v3 record whose diagnostic is v1 keeps its schema and still loads.
    legacy = coordinator.record_failure(
        coordinator.claim_initial(
            attempt=ATTEMPT, iteration=ITERATION + 1,
            envelope_sha256=authority.envelope.envelope_sha256, route="primary",
            route_sha256=authority.envelope.routes[0].route_sha256,
            operation=_operation("v3-lease-claim", "7"),
        ),
        classification="non_retryable", retryable=False, visible_output=False,
        operation=_operation("v3-lease-failure", "4"),
        failure_diagnostic=ProviderFailureDiagnostic(401, "invalid_api_key"),
    )
    assert legacy.to_dict()["schema"] == "unchain.provider_request_lease.v3"
    assert ProviderRequestLease.from_durable_dict(legacy.to_dict()) == legacy


def test_gemini_404_replacement_reaches_the_error_and_survives_sqlite_reopen(tmp_path) -> None:
    """SEQ-386-2 / AC-386-3: the suggestion is shown, recovered without a resend, never the raw text."""

    import traceback

    from unchain.providers.durable_turn_runtime import DurableProviderTurnTerminalError

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    sentinel = "PRIVATE_PROVIDER_SENTINEL"
    message = (
        f"This model models/gemini-2.5-flash is no longer available to new users {sentinel}. "
        "Please update your code to use models/gemini-3.8-flash for the latest features."
    )
    transport, calls = _gemini_transport(catalog, [_google_error(404, "NOT_FOUND", message)])

    with pytest.raises(DurableProviderTurnTerminalError) as first:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=0)
        )
    assert str(first.value) == (
        "durable_provider_turn_terminal_failed:non_retryable; "
        "Provider model or endpoint was not found "
        "(HTTP 404, status=NOT_FOUND; suggested model: gemini-3.8-flash)"
    )
    assert sentinel not in str(first.value)
    assert sentinel not in "".join(traceback.format_exception(first.value))
    assert calls == [GEMINI_MODEL]

    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnTerminalError) as recovered:
        _runtime(reopened, no_send).execute(
            authority=authority, retry_config=RetryConfig(max_retries=0)
        )
    assert str(recovered.value) == str(first.value)
    assert no_send.calls == []
    assert sentinel.encode() not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


def test_gemini_503_budget_exhaustion_names_the_status_and_attempts(tmp_path) -> None:
    """An exhausted Gemini budget is terminal after the bounded three sends."""

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(
        catalog, [_google_error(503, "UNAVAILABLE") for _ in range(3)]
    )

    with pytest.raises(DurableProviderTurnTerminalError) as caught:
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(
                max_retries=99, base_delay_ms=1, max_delay_ms=1, jitter_ratio=0
            ),
        )

    assert len(calls) == 3
    assert caught.value.code == "durable_provider_turn_terminal_failed"
    assert str(caught.value) == "Provider temporarily busy (HTTP 503). Please try again later."
    assert "private" not in "".join(traceback.format_exception(caught.value))
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnTerminalError) as recovered:
        _runtime(reopened, no_send).execute(authority=authority, retry_config=RetryConfig(max_retries=99))
    assert str(recovered.value) == str(caught.value)
    assert no_send.calls == []


def test_gemini_500_stays_uncertain_but_names_the_status(tmp_path) -> None:
    """D1 / AC-386-5: 500 may have been processed, so it is never resent, but it says why."""

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(catalog, [_google_error(500, "INTERNAL")])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert caught.value.code == "durable_provider_turn_uncertain"
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic(
        "provider_stream_error", "reading_response", 500, "INTERNAL"
    )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: The provider reported an error while "
        "processing the request (reason=provider_stream_error, phase=reading_response, "
        "HTTP 500, code=INTERNAL)"
    )
    assert calls == [GEMINI_MODEL]
    assert caught.value.__cause__ is None
    assert "private" not in "".join(traceback.format_exception(caught.value))

    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError) as again:
        _runtime(_store(tmp_path).bind_execution(ATTEMPT.generation.execution_id), no_send).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert no_send.calls == []
    assert str(again.value) == str(caught.value)
    assert again.value.diagnostic == caught.value.diagnostic
    assert b"private body" not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


def test_an_unexpected_send_error_is_named_by_its_class_only(tmp_path) -> None:
    """An unknown failure keeps a closed category without message or class text."""

    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([RuntimeError("PRIVATE unknown network outcome")])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: A local error interrupted provider "
        "response processing (reason=local_processing_error, phase=reading_response)"
    )
    assert "PRIVATE" not in str(caught.value)
    assert len(transport.calls) == 1
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("local_processing_error", "reading_response")
    assert caught.value.__cause__ is None
    assert "RuntimeError" not in str(caught.value)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_an_unsafe_exception_class_name_is_not_shown(tmp_path) -> None:
    odd = type("PRIVATE text; not a name", (RuntimeError,), {})
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([odd("opaque")])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=1)
        )
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("local_processing_error", "reading_response")
    assert "PRIVATE" not in str(caught.value)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    assert len(transport.calls) == 1
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_a_failed_before_send_says_the_request_was_not_started(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    def before_send(_context):
        raise ValueError("PRIVATE guard detail")

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=1),
            before_send=before_send,
        )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: The request preparation callback failed "
        "before the provider send (reason=before_send_failed, phase=before_send)"
    )
    assert transport.calls == []
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("before_send_failed", "before_send")
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_an_unusable_result_is_named_by_its_type(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([{"final_text": "PRIVATE dict result"}])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=1)
        )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: The provider adapter returned an invalid "
        "result (reason=invalid_result, phase=result_processing)"
    )
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("invalid_result", "result_processing")
    assert len(transport.calls) == 1
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_an_answer_that_cannot_be_sealed_says_it_was_not_recorded(tmp_path, monkeypatch) -> None:
    from unchain.providers import durable_turn_runtime as runtime_module

    def reject(**_kwargs):
        raise OverflowError("PRIVATE answer field")

    monkeypatch.setattr(
        runtime_module.ProviderTurnResultEnvelope, "from_model_turn_result", reject
    )
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=1)
        )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: The provider adapter returned an invalid "
        "result (reason=invalid_result, phase=result_processing)"
    )
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("invalid_result", "result_processing")
    assert len(transport.calls) == 1
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_an_invalid_run_receipt_says_the_call_record_failed(tmp_path) -> None:
    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=1),
            build_run_receipt=lambda *_args: {"not": "a receipt"},
        )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: The provider adapter returned an invalid "
        "result (reason=invalid_result, phase=result_processing)"
    )
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("invalid_result", "result_processing")
    assert len(transport.calls) == 1
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_gemini_503_retry_survives_a_restart_without_resending_the_failed_ordinal(tmp_path) -> None:
    """SEQ-386-1 restart cell: the recorded transient lease is reused, not sent again."""

    class _Crash(BaseException):
        pass

    def crash(_seconds):
        raise _Crash()

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(catalog, [_google_error(503, "UNAVAILABLE")])
    retry = RetryConfig(max_retries=2, base_delay_ms=1, max_delay_ms=1, jitter_ratio=0)

    with pytest.raises(_Crash):
        _runtime(repository, transport, sleep=crash).execute(
            authority=authority, retry_config=retry
        )
    assert calls == [GEMINI_MODEL]

    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    resumed, resumed_calls = _gemini_transport(catalog, ["after the restart"])
    outcome = _runtime(reopened, resumed).execute(authority=authority, retry_config=retry)

    assert outcome.result.final_text == "after the restart"
    assert resumed_calls == [GEMINI_MODEL]
    ordinals = [
        reopened.load(
            subject=ProviderRequestSubject(
                ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", ordinal
            )
        ).status
        for ordinal in (0, 1)
    ]
    assert ordinals == [ProviderRequestStatus.FAILED, ProviderRequestStatus.COMPLETED]


def test_a_hostile_provider_error_cannot_replace_the_uncertain_error(tmp_path) -> None:
    """Message evidence is best effort: reading it must never change the error type."""

    class _Hostile(RuntimeError):
        @property
        def status_code(self):
            raise RuntimeError("reading the status must not escape")

    _store_value, repository, authority = _authority(tmp_path)
    transport = _Transport([_Hostile("opaque")])

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=1)
        )
    assert str(caught.value) == (
        "Provider request outcome is uncertain: A local error interrupted provider "
        "response processing (reason=local_processing_error, phase=reading_response)"
    )
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("local_processing_error", "reading_response")
    assert caught.value.__cause__ is None
    assert "_Hostile" not in str(caught.value)
    assert len(transport.calls) == 1
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def test_gemini_timeout_stays_uncertain_but_says_it_timed_out(tmp_path) -> None:
    """A transport timeout has no HTTP status; the text must still say what happened."""

    import httpx

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(
        catalog, [httpx.ReadTimeout("private host and key must not appear")]
    )

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert calls == [GEMINI_MODEL]
    assert str(caught.value) == (
        "Provider request outcome is uncertain: The provider request timed out "
        "(reason=timeout, phase=requesting)"
    )
    assert "private" not in str(caught.value)
    assert caught.value.diagnostic == ProviderUncertaintyDiagnostic("timeout", "requesting")
    assert caught.value.__cause__ is None
    assert "private" not in "".join(traceback.format_exception(caught.value))
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)


def _assert_same_uncertainty_after_reopen(tmp_path, authority, error) -> None:
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError) as recovered:
        _runtime(reopened, no_send).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert no_send.calls == []
    assert recovered.value.diagnostic == error.diagnostic
    assert str(recovered.value) == str(error)
    assert b"PRIVATE" not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


def _uncertain_text_for(tmp_path, outcome) -> str:
    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(catalog, [outcome])
    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert calls == [GEMINI_MODEL]  # never resent
    assert caught.value.__cause__ is None
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    _assert_same_uncertainty_after_reopen(tmp_path, authority, caught.value)
    return str(caught.value)


@pytest.mark.parametrize(
    ("outcome_factory", "expected"),
    [
        (
            lambda: __import__("httpx").RemoteProtocolError("PRIVATE peer closed"),
            "Provider request outcome is uncertain: The connection was interrupted "
            "(reason=connection_interrupted, phase=requesting)",
        ),
        (
            lambda: __import__("httpx").ReadError("PRIVATE reset"),
            "Provider request outcome is uncertain: The connection was interrupted "
            "(reason=connection_interrupted, phase=requesting)",
        ),
        (
            lambda: __import__("httpx").ConnectError("PRIVATE dns"),
            "Provider request outcome is uncertain: The connection was interrupted "
            "(reason=connection_interrupted, phase=requesting)",
        ),
    ],
)
def test_uncertain_failures_without_http_status_say_what_happened(
    tmp_path, outcome_factory, expected
) -> None:
    """Transport failures use fixed local categories and never invent HTTP metadata."""

    text = _uncertain_text_for(tmp_path, outcome_factory())
    assert text == expected
    assert "PRIVATE" not in text
    assert "HTTP" not in text


@pytest.mark.parametrize(
    ("outcome_factory", "reason"),
    [
        (
            lambda: {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "PRIVATE partial"}]},
                        "finish_reason": "MALFORMED_FUNCTION_CALL",
                    }
                ]
            },
            "MALFORMED_FUNCTION_CALL",
        ),
        (
            lambda: {"prompt_feedback": {"block_reason": "SAFETY"}},
            "PROMPT_BLOCKED",
        ),
        (
            lambda: {
                "candidates": [
                    {"content": {"role": "model", "parts": []}, "finish_reason": "STOP"}
                ]
            },
            "EMPTY_RESPONSE",
        ),
    ],
)
def test_explicit_gemini_response_outcomes_are_terminal_without_http_metadata(
    tmp_path, outcome_factory, reason
) -> None:
    """An explicit response outcome is settled, unlike an interrupted stream."""

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(catalog, [outcome_factory()])
    with pytest.raises(DurableProviderTurnTerminalError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert caught.value.code == "durable_provider_turn_terminal_failed"
    assert str(caught.value) == (
        f"durable_provider_turn_terminal_failed:non_retryable; "
        f"Gemini generation stopped (reason={reason})"
    )
    assert caught.value.diagnostic.http_status is None
    assert caught.value.diagnostic.provider_code == reason
    assert calls == [GEMINI_MODEL]
    assert "HTTP" not in str(caught.value)
    assert "PRIVATE" not in "".join(traceback.format_exception(caught.value))
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnTerminalError) as recovered:
        _runtime(reopened, no_send).execute(
            authority=authority, retry_config=RetryConfig(max_retries=3)
        )
    assert str(recovered.value) == str(caught.value)
    assert recovered.value.diagnostic == caught.value.diagnostic
    assert no_send.calls == []
    assert b"PRIVATE" not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


def test_retry_wait_hook_receives_each_retryable_failure(tmp_path) -> None:
    """AC-386-8: the hook replaces the bare sleep and carries closed retry facts."""

    from unchain.providers.durable_turn_runtime import ProviderRetryWait

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(
        catalog,
        [
            _google_error(503, "UNAVAILABLE", "PRIVATE overloaded text"),
            _google_error(429, "RESOURCE_EXHAUSTED"),
            "after two waits",
        ],
    )
    waits = []
    sleeps = []

    def retry_wait(info, sleep):
        assert type(info) is ProviderRetryWait
        waits.append(info)
        sleep(info.delay_ms / 1000.0)

    outcome = _runtime(repository, transport, sleep=sleeps.append).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=4, base_delay_ms=10, max_delay_ms=40, jitter_ratio=0),
        retry_wait=retry_wait,
    )

    assert outcome.result.final_text == "after two waits"
    assert calls == [GEMINI_MODEL] * 3
    # HTTP 503 is capped at two retries; the later HTTP 429 retains the configured four.
    assert [(w.attempt_failed, w.next_attempt, w.max_attempts) for w in waits] == [(1, 2, 3), (2, 3, 5)]
    assert [w.delay_ms for w in waits] == [10, 20]
    assert [(w.diagnostic.http_status, w.diagnostic.provider_status) for w in waits] == [
        (503, "UNAVAILABLE"),
        (429, "RESOURCE_EXHAUSTED"),
    ]
    assert sleeps == [0.01, 0.02]
    assert "PRIVATE" not in repr(waits)


def test_retry_wait_hook_that_raises_stops_before_the_next_send(tmp_path) -> None:
    """SEQ-386-3 Stop cell: an interrupted wait never sends the next try."""

    class StoppedWhileWaiting(RuntimeError):
        code = "execution_cancelled"

    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(catalog, [_google_error(503, "UNAVAILABLE")])

    def retry_wait(info, sleep):
        raise StoppedWhileWaiting("stopped")

    with pytest.raises(StoppedWhileWaiting):
        _runtime(repository, transport).execute(
            authority=authority,
            retry_config=RetryConfig(max_retries=4, base_delay_ms=1, max_delay_ms=1, jitter_ratio=0),
            retry_wait=retry_wait,
        )
    assert calls == [GEMINI_MODEL]
    first = repository.load(
        subject=ProviderRequestSubject(ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 0)
    )
    assert (first.status, first.classification, first.retryable) == (
        ProviderRequestStatus.FAILED, "transient", True,
    )
    assert repository.load(
        subject=ProviderRequestSubject(ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 1)
    ) is None


def test_without_a_hook_the_old_delay_is_used(tmp_path) -> None:
    _store_value, repository, authority, catalog = _gemini_authority(tmp_path)
    transport, calls = _gemini_transport(catalog, [_google_error(503, "UNAVAILABLE"), "ok"])
    sleeps = []
    _runtime(repository, transport, sleep=sleeps.append).execute(
        authority=authority,
        retry_config=RetryConfig(max_retries=2, base_delay_ms=7, max_delay_ms=7, jitter_ratio=0),
    )
    assert sleeps == [0.007]
    assert calls == [GEMINI_MODEL, GEMINI_MODEL]
