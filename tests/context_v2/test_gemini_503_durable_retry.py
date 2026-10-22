"""Real SDK and SQLite contract for an explicit Gemini HTTP 503 rejection."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
from google import genai

from unchain.context.provider_execution import (
    ContextProviderTurnExecutionService,
    official_provider_transport_target_sha256,
)
from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.providers import GeminiModelIO, ModelTurnRequest
from unchain.providers.durable_turn_runtime import (
    DurableProviderTurnMode,
    DurableProviderTurnTerminalError,
    DurableProviderTurnUncertainError,
)
from unchain.retry import RetryConfig
from unchain.tools import Toolkit


_OVERLOADED = {
    "error": {
        "code": 503,
        "message": "private provider response content must not be persisted",
        "status": "UNAVAILABLE",
    }
}
_COMPLETE = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "complete"}]},
            "finishReason": "STOP",
        }
    ]
}
_PARTIAL = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "partial"}]}}
    ]
}


def _sse(*values):
    return "".join(f"data: {json.dumps(value)}\n\n" for value in values)


def _harness(tmp_path, handler, *, sleep=None, callback=None):
    sends = []
    waits = []

    def observed_send(request):
        sends.append(request.content)
        return handler(request, len(sends))

    def client_factory(**kwargs):
        options = kwargs.pop("http_options")
        options["client_args"] = {"transport": httpx.MockTransport(observed_send)}
        return genai.Client(**kwargs, http_options=options)

    attempt = AttemptRef(
        GenerationRef("gemini-503-execution", "gemini-503-generation"),
        "gemini-503-attempt",
    )
    store = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    )
    model = GeminiModelIO(
        model="gemini-3.6-flash",
        api_key="offline-fixture",
        client_factory=client_factory,
    )
    request = ModelTurnRequest(
        messages=[{"role": "user", "content": "hello"}],
        callback=callback,
        run_id=attempt.attempt_id,
        toolkit=Toolkit(),
    )

    def run(max_retries=10):
        service = ContextProviderTurnExecutionService(
            attempt=attempt,
            store=store.bind_execution(attempt.generation.execution_id),
            mode=DurableProviderTurnMode.ENFORCE,
            transport_target_sha256=official_provider_transport_target_sha256(),
            sleep=waits.append if sleep is None else sleep,
        )
        return service.fetch_prepared(
            model_io=model,
            request=request,
            retry_config=RetryConfig(max_retries=max_retries, jitter_ratio=0),
        )

    def leases():
        with sqlite3.connect(tmp_path / "journal.sqlite3") as connection:
            return [
                json.loads(row[0])
                for row in connection.execute(
                    "SELECT lease_json FROM provider_request_lease_revisions "
                    "ORDER BY rowid"
                )
            ]

    return run, sends, waits, leases


def test_pre_stream_503_retries_frozen_request_and_recovers_after_reopen(tmp_path):
    def handler(_request, ordinal):
        if ordinal == 1:
            return httpx.Response(503, headers={"Retry-After": "2"}, json=_OVERLOADED)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(_COMPLETE),
        )

    run, sends, waits, leases = _harness(tmp_path, handler)
    assert run().final_text == "complete"
    assert len(sends) == 2 and sends[0] == sends[1]
    assert waits[0] == 2.0
    persisted = leases()
    assert any(
        row["status"] == "failed"
        and row["failure_diagnostic"]["http_status"] == 503
        for row in persisted
    )
    assert any(row["status"] == "completed" for row in persisted)
    assert run().final_text == "complete"
    assert len(sends) == 2
    assert waits == [2.0]
    assert "private provider response content" not in json.dumps(persisted)


def test_sse_503_after_output_is_uncertain_and_never_resent(tmp_path):
    def handler(_request, _ordinal):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(_PARTIAL, _OVERLOADED),
        )

    progress = []
    run, sends, _waits, leases = _harness(
        tmp_path, handler, callback=progress.append
    )
    with pytest.raises(DurableProviderTurnUncertainError):
        run()
    with pytest.raises(DurableProviderTurnUncertainError):
        run()
    assert len(sends) == 1
    assert leases()[0]["status"] == "started"
    assert not any(event.get("type") == "provider_retry" for event in progress)


def test_network_disconnect_has_no_http_rejection_proof_and_is_not_resent(tmp_path):
    def handler(_request, _ordinal):
        raise httpx.ReadError("connection lost")

    run, sends, _waits, leases = _harness(tmp_path, handler)
    with pytest.raises(DurableProviderTurnUncertainError):
        run()
    with pytest.raises(DurableProviderTurnUncertainError):
        run()
    assert len(sends) == 1
    assert leases()[0]["status"] == "started"


def test_retry_progress_reports_only_committed_safe_state(tmp_path):
    def handler(_request, ordinal):
        if ordinal < 3:
            return httpx.Response(503, json=_OVERLOADED)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(_COMPLETE),
        )

    progress = []
    run, sends, waits, leases = _harness(
        tmp_path, handler, callback=progress.append
    )
    assert run().final_text == "complete"
    notices = [event for event in progress if event.get("type") == "provider_retry"]
    assert [event["retry_ordinal"] for event in notices] == [1, 2]
    assert all(event["provider"] == "gemini" for event in notices)
    assert all(event["http_status"] == 503 for event in notices)
    assert all(event["max_retries"] == 2 for event in notices)
    assert [event["delay_ms"] for event in notices] == [500, 1000]
    assert all(
        set(event)
        == {
            "type", "run_id", "iteration", "timestamp", "provider",
            "http_status", "retry_ordinal", "max_retries", "delay_ms",
        }
        for event in notices
    )
    assert len(sends) == 3 and len(waits) == 2
    assert [row["status"] for row in leases()].count("failed") == 2


def test_repeated_pre_stream_503_exhausts_a_small_budget_with_diagnostic(tmp_path):
    def handler(_request, _ordinal):
        return httpx.Response(503, json=_OVERLOADED)

    run, sends, waits, leases = _harness(tmp_path, handler)
    with pytest.raises(DurableProviderTurnTerminalError) as caught:
        run()
    assert "temporarily busy (HTTP 503)" in str(caught.value)
    assert len(sends) == 3 and len(set(sends)) == 1
    assert 0 < waits[0] < waits[1]
    persisted = leases()
    assert persisted[-1]["status"] == "failed"
    assert persisted[-1]["retryable"] is False
    assert persisted[-1]["failure_diagnostic"]["http_status"] == 503
    with pytest.raises(DurableProviderTurnTerminalError) as resumed:
        run()
    assert "HTTP 503" in str(resumed.value)
    assert len(sends) == 3


def test_reopen_after_saved_503_retries_the_same_request(tmp_path):
    class SimulatedProcessExit(BaseException):
        pass

    first_sleep = True

    def exit_during_backoff(_delay):
        nonlocal first_sleep
        if first_sleep:
            first_sleep = False
            raise SimulatedProcessExit()

    def handler(_request, ordinal):
        if ordinal == 1:
            return httpx.Response(503, json=_OVERLOADED)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(_COMPLETE),
        )

    run, sends, _waits, leases = _harness(
        tmp_path, handler, sleep=exit_during_backoff
    )
    with pytest.raises(SimulatedProcessExit):
        run()
    assert len(sends) == 1
    assert leases()[-1]["status"] == "failed"
    assert leases()[-1]["retryable"] is True

    assert run().final_text == "complete"
    assert len(sends) == 2 and sends[0] == sends[1]
