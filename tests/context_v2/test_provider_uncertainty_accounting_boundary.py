"""Authoritative ledger faults must survive the uncertain observer guard."""

from __future__ import annotations

from dataclasses import replace
import traceback

import pytest

from unchain.context.provider_execution import ContextProviderTurnExecutionError
from unchain.durability import is_durable_persistence_failure
from unchain.providers.durable_turn_runtime import (
    DurableProviderTurnMode,
    DurableProviderTurnRuntime,
    DurableProviderTurnUncertainError,
)
from unchain.retry import RetryConfig

from tests.context_v2.test_provider_turn_execution_service import (
    ATTEMPT,
    _FailingTransport,
    _model_io,
    _request,
    _run_receipt_factory,
    _service,
)


PRIVATE = "PRIVATE provider or observer exception must not reach the user"


@pytest.mark.parametrize("fault, expected_message", [
    ("missing_ledger", "durable provider store lacks the accounting ledger"),
    ("changed_receipt", "durable provider store changed the accounting receipt"),
    ("unmatched_start", "provider attempt completed without a matching start"),
])
def test_controlled_accounting_fault_preserves_boundary_after_uncertain_send(
    tmp_path, monkeypatch, fault, expected_message,
):
    service = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    transport = _FailingTransport(RuntimeError(PRIVATE))
    monkeypatch.setattr(
        "unchain.context.provider_execution._exact_transport",
        lambda **_kwargs: transport,
    )
    observed = []
    received_by_ledger = []
    if fault == "missing_ledger":
        monkeypatch.setattr(service.store, "append_receipt", None)
    elif fault == "changed_receipt":
        def return_foreign_receipt(receipt):
            received_by_ledger.append(receipt)
            return replace(
                receipt,
                identity=replace(receipt.identity, attempt_id="foreign-attempt"),
                extensions=dict(receipt.extensions),
                provider_call_id="",
            )

        monkeypatch.setattr(service.store, "append_receipt", return_foreign_receipt)
    else:
        execute = DurableProviderTurnRuntime.execute

        def execute_with_unpaired_completion(runtime, **kwargs):
            after_send = kwargs["after_send"]

            def unmatched_completion(_context, completed_at, outcome, classification):
                after_send(None, completed_at, outcome, classification)

            return execute(runtime, **{**kwargs, "after_send": unmatched_completion})

        monkeypatch.setattr(
            DurableProviderTurnRuntime, "execute", execute_with_unpaired_completion,
        )

    with pytest.raises(ContextProviderTurnExecutionError, match=expected_message) as caught:
        service.fetch_prepared(
            model_io=_model_io([]), request=_request(),
            retry_config=RetryConfig(max_retries=2),
            run_receipt_factory=_run_receipt_factory(),
            run_receipt_observed=observed.append,
        )
    error = caught.value
    assert str(error) == expected_message
    assert is_durable_persistence_failure(error)
    assert PRIVATE not in "".join(traceback.format_exception(error))
    assert transport.calls == 1
    assert observed == []
    assert len(received_by_ledger) == (1 if fault == "changed_receipt" else 0)
    assert all(receipt.identity.attempt_id == ATTEMPT.attempt_id
               for receipt in received_by_ledger)
    assert service.store.load_receipts(
        root_run_id=ATTEMPT.attempt_id,
        owner_run_id=ATTEMPT.attempt_id,
        attempt_id=ATTEMPT.attempt_id,
    ) == ()

    # A genuine SQLite reopen still cannot re-send the uncertain request or
    # consume the foreign accounting receipt returned above.
    reopened = _service(tmp_path, DurableProviderTurnMode.ENFORCE_TEST)
    with pytest.raises(DurableProviderTurnUncertainError) as recovered:
        reopened.fetch_prepared(
            model_io=_model_io([]), request=_request(),
            retry_config=RetryConfig(max_retries=2),
            run_receipt_factory=_run_receipt_factory(),
            run_receipt_observed=observed.append,
        )
    assert recovered.value.diagnostic.reason == "local_processing_error"
    assert transport.calls == 1
    assert observed == []
    assert PRIVATE not in "".join(traceback.format_exception(recovered.value))
    assert PRIVATE.encode() not in (
        tmp_path / "memory_v2" / "context_v2.sqlite3"
    ).read_bytes()
