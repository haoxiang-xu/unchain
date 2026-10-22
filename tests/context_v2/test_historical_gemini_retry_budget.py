"""Cold #386 request history obeys #390's cap without losing durable outcomes."""

from __future__ import annotations

import pytest

from tests.test_durable_provider_turn_runtime import (
    ATTEMPT,
    ITERATION,
    _gemini_authority,
    _gemini_transport,
    _operation,
    _result,
    _runtime,
    _store,
)
from unchain.journal.provider_result import ProviderTurnResultEnvelope
from unchain.providers.durable_turn_runtime import (
    DurableProviderTurnStatus,
    DurableProviderTurnTerminalError,
    DurableProviderTurnUncertainError,
)
from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic
from unchain.providers.request_lease import (
    ProviderRequestLeaseCoordinator,
    ProviderRequestStatus,
    ProviderRequestSubject,
    ProviderTurnResultBinding,
)
from unchain.retry import RetryConfig


RETRY = RetryConfig(max_retries=10, base_delay_ms=0, max_delay_ms=0, jitter_ratio=0)


def _historical_failures(tmp_path, status=503):
    """#386 used the normal retry budget and could leave retryable ordinal 2."""
    _, repository, authority, catalog = _gemini_authority(tmp_path)
    coordinator = ProviderRequestLeaseCoordinator(repository)
    lease = coordinator.claim_initial(
        attempt=ATTEMPT,
        iteration=ITERATION,
        envelope_sha256=authority.envelope.envelope_sha256,
        route="primary",
        route_sha256=authority.envelope.routes[0].route_sha256,
        operation=_operation("historical-claim-0", "1"),
    )
    for ordinal in range(3):
        if ordinal:
            lease = coordinator.claim_retry(
                lease, operation=_operation(f"historical-claim-{ordinal}", str(ordinal + 1))
            )
        lease = coordinator.record_failure(
            lease,
            classification="transient",
            retryable=True,
            visible_output=False,
            operation=_operation(f"historical-failed-{ordinal}", str(ordinal + 5)),
            failure_diagnostic=ProviderFailureDiagnostic(
                status,
                provider_status="UNAVAILABLE" if status == 503 else "RESOURCE_EXHAUSTED",
            ),
        )
    return repository, authority, catalog, lease


def _subject(authority, ordinal):
    return ProviderRequestSubject(
        ATTEMPT, ITERATION, authority.envelope.envelope_sha256, "primary", ordinal
    )


def _reopened(tmp_path):
    return _store(tmp_path).bind_execution(ATTEMPT.generation.execution_id)


def test_historical_503_budget_is_exhausted_before_a_new_ordinal_is_claimed(tmp_path):
    _, authority, catalog, _ = _historical_failures(tmp_path)
    reopened = _reopened(tmp_path)
    transport, calls = _gemini_transport(catalog, ["must not send a fourth time"])
    waits = []

    with pytest.raises(DurableProviderTurnTerminalError) as caught:
        _runtime(reopened, transport, sleep=waits.append).execute(
            authority=authority, retry_config=RETRY
        )

    assert caught.value.diagnostic.http_status == 503
    assert calls == [] and waits == []
    assert reopened.load(subject=_subject(authority, 3)) is None


@pytest.mark.parametrize("sealed", [False, True], ids=["receipt_only", "completed_lease"])
def test_existing_fourth_ordinal_result_is_recovered_even_after_the_new_cap(tmp_path, sealed):
    repository, authority, catalog, failed = _historical_failures(tmp_path)
    coordinator = ProviderRequestLeaseCoordinator(repository)
    started = coordinator.claim_retry(failed, operation=_operation("historical-claim-3", "4"))
    result = ProviderTurnResultEnvelope.from_model_turn_result(
        subject=started.subject,
        route_sha256=started.route_sha256,
        visible_output=True,
        result=_result("already completed before the new cap"),
    )
    receipt = repository.persist_provider_turn_result_cas(
        started_lease=started,
        envelope=result,
        artifact_operation=_operation("historical-result-artifact", "a"),
        event_operation=_operation("historical-result-event", "b"),
        event_id="historical-fourth-result",
    )
    if sealed:
        coordinator.record_completed_result(
            started,
            result_binding=ProviderTurnResultBinding(
                route_sha256=result.route_sha256,
                result_sha256=result.result_sha256,
                artifact=receipt.artifact,
                cursor=receipt.cursor,
            ),
            visible_output=True,
            operation=_operation("historical-completed-3", "c"),
        )
    reopened = _reopened(tmp_path)
    transport, calls = _gemini_transport(catalog, [])
    waits = []

    outcome = _runtime(reopened, transport, sleep=waits.append).execute(
        authority=authority, retry_config=RETRY
    )

    assert outcome.status is DurableProviderTurnStatus.COMPLETED
    assert outcome.recovered is True
    assert outcome.result.final_text == "already completed before the new cap"
    assert calls == [] and waits == []
    assert reopened.load(subject=_subject(authority, 3)).status is ProviderRequestStatus.COMPLETED


def test_existing_fourth_ordinal_started_lease_stays_uncertain_and_is_never_resent(tmp_path):
    repository, authority, catalog, failed = _historical_failures(tmp_path)
    started = ProviderRequestLeaseCoordinator(repository).claim_retry(
        failed, operation=_operation("historical-claim-3", "4")
    )
    reopened = _reopened(tmp_path)
    transport, calls = _gemini_transport(catalog, [])
    waits = []

    with pytest.raises(DurableProviderTurnUncertainError):
        _runtime(reopened, transport, sleep=waits.append).execute(
            authority=authority, retry_config=RETRY
        )

    assert calls == [] and waits == []
    assert reopened.load(subject=started.subject).lease_sha256 == started.lease_sha256


def test_historical_429_keeps_the_configured_budget_and_can_send_the_fourth_ordinal(tmp_path):
    _, authority, catalog, _ = _historical_failures(tmp_path, status=429)
    reopened = _reopened(tmp_path)
    transport, calls = _gemini_transport(catalog, ["normal rate-limit budget"])

    outcome = _runtime(reopened, transport).execute(authority=authority, retry_config=RETRY)

    assert outcome.result.final_text == "normal rate-limit budget"
    assert len(calls) == 1
    assert reopened.load(subject=_subject(authority, 3)).status is ProviderRequestStatus.COMPLETED
