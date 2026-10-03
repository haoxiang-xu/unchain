"""Uncertain outcome detail must survive recovery without authorizing a resend."""
from __future__ import annotations

from dataclasses import replace
import importlib
import json
import traceback

import httpx
import pytest

from unchain.context.provider_execution import ContextProviderTurnExecutionService
from unchain.providers.durable_turn_runtime import DurableProviderTurnUncertainError
from unchain.providers.request_lease import (
    ProviderRequestDisposition,
    ProviderRequestLease,
    ProviderRequestLeaseCoordinator,
    ProviderRequestStatus,
    ProviderRequestTransitionError,
)

from tests.context_v2.test_gemini_response_outcomes import (
    PARTIAL,
    PRIVATE,
    harness,
    sse,
    stream,
)


def diagnostic_api():
    # Import within tests so the initial red run shows every missing contract.
    return importlib.import_module("unchain.providers.uncertainty_diagnostic")


def test_uncertainty_record_has_closed_fields_and_exact_types():
    record = {
        "schema": "unchain.provider_uncertainty_diagnostic.v1",
        "reason": "stream_incomplete",
        "phase": "reading_response",
        "http_status": None,
        "provider_code": "",
    }
    diagnostic = diagnostic_api().ProviderUncertaintyDiagnostic.from_dict(record)
    assert diagnostic.to_dict() == record
    for changed in (
        dict(record, schema="unchain.provider_uncertainty_diagnostic.v2"),
        dict(record, reason=PRIVATE),
        dict(record, phase=PRIVATE),
        dict(record, http_status=True),
        dict(record, http_status="503"),
        dict(record, http_status=200),
        dict(record, provider_code=PRIVATE),
        dict(record, provider_code="UNAVAILABLE"),
        dict(record, exception=PRIVATE),
        {key: value for key, value in record.items() if key != "phase"},
    ):
        with pytest.raises((ValueError, TypeError)):
            diagnostic_api().ProviderUncertaintyDiagnostic.from_dict(changed)


@pytest.mark.parametrize("body,reason,phase", [
    (sse(PARTIAL), "stream_incomplete", "reading_response"),
    (sse(PARTIAL) + "data: {invalid\n\n", "response_parse_error", "reading_response"),
    (sse(PARTIAL, {"error": {"code": 503, "status": "UNAVAILABLE", "message": PRIVATE}}), "provider_stream_error", "reading_response"),
    (sse(PARTIAL, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": PRIVATE}}), "provider_stream_error", "reading_response"),
    (sse({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "done"}]}}]}, {"candidates": [{"finishReason": "MAX_TOKENS"}]}), "conflicting_finish_reason", "processing_response"),
    (sse({"candidates": [{"finishReason": "UNRECOGNIZED_FUTURE_FINISH"}]}), "unsupported_finish_reason", "processing_response"),
    (sse({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"functionCall": {"args": {}}}]}}]}), "malformed_tool_call", "processing_response"),
])
def test_real_sdk_uncertainty_is_detailed_after_cold_reopen(tmp_path, body, reason, phase):
    run, sends, _, records = harness(tmp_path, lambda _ordinal: stream(body))
    first = None
    for _ in range(2):
        with pytest.raises(DurableProviderTurnUncertainError) as caught:
            run()
        error = caught.value
        assert error.code == "durable_provider_turn_uncertain"
        assert error.diagnostic.reason == reason
        assert error.diagnostic.phase == phase
        assert reason in str(error) and phase in str(error)
        assert PRIVATE not in str(error)
        assert PRIVATE not in "".join(traceback.format_exception(error))
        if first is None:
            first = error.diagnostic.to_dict()
        else:
            assert error.diagnostic.to_dict() == first
    leases, receipt_count = records()
    assert len(sends) == 1 and receipt_count == 0
    assert len(leases) == 2
    assert all(lease["status"] == "started" for lease in leases)
    current = leases[-1]
    assert current["schema"] == "unchain.provider_request_lease.v4"
    assert current["revision"] == 2
    assert current["uncertainty_diagnostic"] == first
    assert current["retryable"] is False and current["result_binding"] is None
    assert PRIVATE not in json.dumps(leases)
    if reason == "provider_stream_error":
        assert first["http_status"] == 200
        assert first["provider_code"] == ("RESOURCE_EXHAUSTED" if '"code": 429' in body else "UNAVAILABLE")


@pytest.mark.parametrize("error_type,reason", [
    (httpx.ReadError, "connection_interrupted"),
    (httpx.ReadTimeout, "timeout"),
    (RuntimeError, "local_processing_error"),
])
def test_real_sdk_midstream_failures_do_not_leak_or_automatically_retry(tmp_path, error_type, reason):
    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield sse(PARTIAL).encode()
            raise error_type(PRIVATE)

    run, sends, _, records = harness(tmp_path, lambda _ordinal: httpx.Response(
        200, headers={"Content-Type": "text/event-stream"}, stream=BrokenStream(),
    ))
    for _ in range(2):
        with pytest.raises(DurableProviderTurnUncertainError) as caught:
            run()
        assert caught.value.diagnostic.reason == reason
        assert caught.value.diagnostic.phase == "reading_response"
        assert PRIVATE not in str(caught.value)
        assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    leases, receipt_count = records()
    assert len(sends) == 1 and receipt_count == 0
    assert leases[-1]["status"] == "started"
    assert PRIVATE not in json.dumps(leases)


def test_uncertainty_metadata_cannot_be_attached_to_a_terminal_lease():
    api = diagnostic_api()
    from tests.test_provider_request_lease import _claim_primary, _coordinator

    coordinator, _ = _coordinator()
    started = _claim_primary(coordinator)
    diagnostic = api.ProviderUncertaintyDiagnostic("stream_incomplete", "reading_response")
    annotated = replace(started, revision=2, uncertainty_diagnostic=diagnostic)
    assert annotated.status is ProviderRequestStatus.STARTED
    assert ProviderRequestLease.from_durable_dict(annotated.to_dict()) == annotated
    with pytest.raises((ValueError, TypeError)):
        replace(annotated, status=ProviderRequestStatus.FAILED, classification="non_retryable")
    with pytest.raises((ValueError, TypeError)):
        replace(annotated, retryable=True)
    with pytest.raises((ValueError, TypeError)):
        replace(annotated, classification="transient")


def test_uncertainty_cas_preserves_no_resend_and_legacy_lease_bytes():
    api = diagnostic_api()
    from tests.test_provider_request_lease import _claim_primary, _coordinator, _operation

    coordinator, port = _coordinator()
    started = _claim_primary(coordinator)
    old_record = started.to_dict()
    assert old_record["schema"] == "unchain.provider_request_lease.v2"
    assert "uncertainty_diagnostic" not in old_record
    diagnostic = api.ProviderUncertaintyDiagnostic("stream_incomplete", "reading_response")
    annotated = coordinator.record_uncertainty(started, diagnostic=diagnostic, operation=_operation(2))
    assert annotated.status is ProviderRequestStatus.STARTED
    assert annotated.revision == started.revision + 1
    assert annotated.retryable is False and annotated.result_binding is None
    assert ProviderRequestLease.from_durable_dict(old_record).to_dict() == old_record
    recovered = ProviderRequestLeaseCoordinator(port).recover(started.subject)
    assert recovered.disposition is ProviderRequestDisposition.UNCERTAIN
    assert recovered.auto_resend_allowed is False
    assert recovered.lease.uncertainty_diagnostic == diagnostic
    with pytest.raises(ProviderRequestTransitionError):
        coordinator.claim_retry(annotated, operation=_operation(3))
    assert not any(call[0] == "cas" and call[2] == 2 for call in port.calls)


def test_old_undiagnosed_uncertainty_has_an_honest_recovery_description():
    error = DurableProviderTurnUncertainError()
    assert error.code == "durable_provider_turn_uncertain"
    assert error.diagnostic.reason == "unrecorded_outcome"
    assert error.diagnostic.phase == "recovery"
    assert "unrecorded_outcome" in str(error)


@pytest.mark.parametrize("failure,reason,phase,send_count", [
    ("guard", "before_send_failed", "before_send", 0),
    ("result", "invalid_result", "result_processing", 1),
])
def test_durable_nonstream_failures_retain_their_stage_on_recovery(tmp_path, failure, reason, phase, send_count):
    from tests.test_durable_provider_turn_runtime import (
        ATTEMPT, ITERATION, _Transport, _authority, _result, _runtime, _store,
    )
    from unchain.providers.request_lease import ProviderRequestSubject
    from unchain.retry import RetryConfig

    _, repository, authority = _authority(tmp_path)
    transport = _Transport([None if failure == "result" else _result()])

    def fail_guard(_context):
        raise RuntimeError(PRIVATE)

    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=2),
            before_send=fail_guard if failure == "guard" else None,
        )
    first = caught.value.diagnostic.to_dict()
    assert first["reason"] == reason and first["phase"] == phase
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    assert len(transport.calls) == send_count
    subject = ProviderRequestSubject(ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 0)
    assert repository.load(subject=subject).status is ProviderRequestStatus.STARTED
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError) as recovered:
        _runtime(reopened, no_send).execute(authority=authority, retry_config=RetryConfig(max_retries=2))
    assert recovered.value.diagnostic.to_dict() == first
    assert no_send.calls == []
    assert PRIVATE.encode() not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


def test_failed_uncertainty_annotation_keeps_original_error_and_no_resend(tmp_path, monkeypatch):
    run, sends, _, records = harness(tmp_path, lambda _ordinal: stream(sse(PARTIAL)))

    def unavailable_diagnostic_store(*_args, **_kwargs):
        raise RuntimeError(PRIVATE)

    monkeypatch.setattr(ProviderRequestLeaseCoordinator, "record_uncertainty", unavailable_diagnostic_store)
    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        run()
    assert caught.value.diagnostic.reason == "stream_incomplete"
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    with pytest.raises(DurableProviderTurnUncertainError) as recovered:
        run()
    assert recovered.value.diagnostic.reason == "unrecorded_outcome"
    assert recovered.value.diagnostic.phase == "recovery"
    assert len(sends) == 1 and records()[1] == 0
    assert records()[0][-1]["status"] == "started"


def test_real_sdk_request_validation_has_safe_detail_before_client_creation():
    from unchain.providers import GeminiModelIO, ModelTurnRequest
    from unchain.tools import Toolkit

    client_calls = []

    def forbidden_client(**kwargs):
        client_calls.append(kwargs)
        pytest.fail("invalid local request must not create a provider client")

    model = GeminiModelIO(model="gemini-3.6-flash", api_key="offline-fixture", client_factory=forbidden_client)
    request = ModelTurnRequest(messages=[{"role": "user", "content": "hello"}], run_id="validation", toolkit=Toolkit())
    wire = {
        "model": "gemini-3.6-flash", "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
        "config": {"temperature": PRIVATE},
    }
    with pytest.raises(diagnostic_api().ProviderUncertaintyObservation) as caught:
        model._fetch_prepared(request, wire)
    assert caught.value.diagnostic.reason == "request_validation_failed"
    assert caught.value.diagnostic.phase == "request_preparation"
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    assert client_calls == []


@pytest.mark.parametrize("failure,reason", [
    ("observer", "local_processing_error"),
    ("receipt", "invalid_result"),
])
def test_durable_observer_and_receipt_failures_preserve_original_safe_diagnostic(tmp_path, failure, reason):
    from tests.test_durable_provider_turn_runtime import (
        ATTEMPT, ITERATION, _Transport, _authority, _result, _runtime, _store,
    )
    from unchain.providers.request_lease import ProviderRequestSubject
    from unchain.retry import RetryConfig

    _, repository, authority = _authority(tmp_path)
    transport = _Transport([RuntimeError(PRIVATE) if failure == "observer" else _result()])

    def broken_observer(*_args):
        raise RuntimeError(PRIVATE)

    kwargs = {"after_send": broken_observer} if failure == "observer" else {"build_run_receipt": lambda *_args: None}
    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(authority=authority, retry_config=RetryConfig(max_retries=2), **kwargs)
    first = caught.value.diagnostic.to_dict()
    assert first["reason"] == reason
    if failure == "receipt":
        assert first["phase"] == "result_processing"
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    assert len(transport.calls) == 1
    subject = ProviderRequestSubject(ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 0)
    assert repository.load(subject=subject).status is ProviderRequestStatus.STARTED
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError) as recovered:
        _runtime(reopened, no_send).execute(authority=authority, retry_config=RetryConfig(max_retries=2))
    assert recovered.value.diagnostic.to_dict() == first
    assert no_send.calls == []
    assert PRIVATE.encode() not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


@pytest.mark.parametrize("error_type,reason", [
    (httpx.ConnectError, "connection_interrupted"),
    (httpx.ConnectTimeout, "timeout"),
])
def test_real_sdk_connection_establishment_is_reported_as_requesting(tmp_path, error_type, reason):
    def disconnected(_ordinal):
        raise error_type(PRIVATE)

    run, sends, _, records = harness(tmp_path, disconnected)
    first = None
    for _ in range(2):
        with pytest.raises(DurableProviderTurnUncertainError) as caught:
            run()
        diagnostic = caught.value.diagnostic.to_dict()
        assert diagnostic["reason"] == reason
        assert diagnostic["phase"] == "requesting"
        assert PRIVATE not in "".join(traceback.format_exception(caught.value))
        if first is None:
            first = diagnostic
        else:
            assert diagnostic == first
    assert len(sends) == 1 and records()[1] == 0
    assert records()[0][-1]["status"] == "started"
    assert PRIVATE not in json.dumps(records()[0])


@pytest.mark.parametrize("callback", ["build_run_receipt", "after_send"])
def test_ordinary_post_result_callback_failure_is_safe_and_cold_recoverable(tmp_path, callback):
    from tests.test_durable_provider_turn_runtime import (
        ATTEMPT, ITERATION, _Transport, _authority, _result, _runtime, _store,
    )
    from unchain.providers.request_lease import ProviderRequestSubject
    from unchain.retry import RetryConfig

    _, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])

    def broken_callback(*_args):
        raise RuntimeError(PRIVATE)

    # Both callbacks run before result persistence in the live completion path.
    with pytest.raises(DurableProviderTurnUncertainError) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=2),
            **{callback: broken_callback},
        )
    first = caught.value.diagnostic.to_dict()
    assert first["reason"] == "local_processing_error"
    assert first["phase"] == "result_processing"
    assert PRIVATE not in str(caught.value)
    assert PRIVATE not in "".join(traceback.format_exception(caught.value))
    subject = ProviderRequestSubject(ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", 0)
    lease = repository.load(subject=subject)
    assert lease.status is ProviderRequestStatus.STARTED
    assert lease.uncertainty_diagnostic.to_dict() == first
    assert len(transport.calls) == 1
    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    with pytest.raises(DurableProviderTurnUncertainError) as recovered:
        _runtime(reopened, no_send).execute(authority=authority, retry_config=RetryConfig(max_retries=2))
    assert recovered.value.diagnostic.to_dict() == first
    assert no_send.calls == []
    assert PRIVATE.encode() not in (tmp_path / "memory_v2" / "context_v2.sqlite3").read_bytes()


@pytest.mark.parametrize("callback", ["build_run_receipt", "after_send"])
@pytest.mark.parametrize("kind", ["durability", "cancelled"])
def test_authoritative_post_result_callback_failures_are_not_reclassified(tmp_path, callback, kind):
    from tests.test_durable_provider_turn_runtime import _Transport, _authority, _result, _runtime
    from unchain.durability import is_durable_persistence_failure, mark_durable_persistence_failure
    from unchain.retry import RetryConfig

    _, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])
    if kind == "durability":
        original = mark_durable_persistence_failure(RuntimeError("ledger rejected"))
    else:
        class Cancelled(RuntimeError):
            code = "execution_cancelled"
        original = Cancelled("execution cancelled")

    def authoritative_failure(*_args):
        raise original

    with pytest.raises(type(original)) as caught:
        _runtime(repository, transport).execute(
            authority=authority, retry_config=RetryConfig(max_retries=2),
            **{callback: authoritative_failure},
        )
    assert caught.value is original
    if kind == "durability":
        assert is_durable_persistence_failure(caught.value)
    else:
        assert caught.value.code == "execution_cancelled"
    assert len(transport.calls) == 1


def test_completed_cold_replay_does_not_invoke_live_completion_callbacks(tmp_path):
    from tests.test_durable_provider_turn_runtime import ATTEMPT, _Transport, _authority, _result, _runtime, _store
    from unchain.retry import RetryConfig

    _, repository, authority = _authority(tmp_path)
    transport = _Transport([_result()])
    first = _runtime(repository, transport).execute(authority=authority, retry_config=RetryConfig(max_retries=0))

    def forbidden_callback(*_args):
        pytest.fail("completed replay must not invoke live completion callbacks")

    reopened = _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)
    no_send = _Transport([])
    recovered = _runtime(reopened, no_send).execute(
        authority=authority, retry_config=RetryConfig(max_retries=0),
        build_run_receipt=forbidden_callback, after_send=forbidden_callback,
    )
    assert recovered.recovered and recovered.result.final_text == first.result.final_text
    assert no_send.calls == []
