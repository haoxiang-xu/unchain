"""#386 AC-386-9 (kernel part): retry waits are reported and stay stoppable."""

from __future__ import annotations

import pytest

# Loading the context package first avoids an existing import cycle between
# the durable runtime and the journal (same order as
# test_durable_provider_turn_runtime.py).
from unchain.context.tool_catalog import ToolCatalogEnvelope  # noqa: F401
from unchain.kernel.loop import KernelLoop
from unchain.providers.durable_turn_runtime import ProviderRetryWait
from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic

RETRY_KEYS = {
    "type", "run_id", "iteration", "provider", "attempt_failed", "next_attempt",
    "max_attempts", "delay_ms", "remaining_ms", "http_status", "provider_status",
}


def _wait(delay_ms: int, diagnostic=ProviderFailureDiagnostic(503, provider_status="UNAVAILABLE")):
    return ProviderRetryWait(
        attempt_failed=2,
        next_attempt=3,
        max_attempts=11,
        delay_ms=delay_ms,
        diagnostic=diagnostic,
    )


def _hook(callback, guard=None):
    loop = KernelLoop.__new__(KernelLoop)
    return loop._provider_retry_wait(
        callback=callback,
        run_id="run-1",
        iteration=4,
        provider="gemini",
        execution_guard=guard,
    )


def test_wait_reports_its_start_and_every_second_then_returns():
    events, sleeps = [], []
    _hook(events.append)(_wait(2500), sleeps.append)

    assert sleeps == [1.0, 1.0, 0.5]
    assert [event["remaining_ms"] for event in events] == [2500, 1500, 500, 0]
    for event in events:
        assert set(event) == RETRY_KEYS
        assert event == {
            **event,
            "type": "provider_retry",
            "run_id": "run-1",
            "iteration": 4,
            "provider": "gemini",
            "attempt_failed": 2,
            "next_attempt": 3,
            "max_attempts": 11,
            "delay_ms": 2500,
            "http_status": 503,
            "provider_status": "UNAVAILABLE",
        }


def test_wait_without_http_evidence_reports_empty_status_fields():
    events = []
    _hook(events.append)(_wait(0, diagnostic=None), lambda _seconds: None)
    assert len(events) == 1
    assert (events[0]["http_status"], events[0]["provider_status"]) == (None, "")


def test_a_stopped_turn_ends_the_wait_at_the_next_heartbeat():
    class Stopped(RuntimeError):
        pass

    sleeps = []
    calls = []

    def callback(event):
        calls.append(event["remaining_ms"])
        if len(calls) == 2:
            raise Stopped("stopped")

    with pytest.raises(Stopped):
        _hook(callback)(_wait(30000), sleeps.append)
    assert sleeps == [1.0]
    assert calls == [30000, 29000]


def test_a_lost_execution_guard_ends_the_wait():
    class Lost(RuntimeError):
        pass

    class Guard:
        def __init__(self):
            self.checks = 0

        def assert_active(self):
            self.checks += 1
            raise Lost("lease lost")

    events, sleeps = [], []
    guard = Guard()
    with pytest.raises(Lost):
        _hook(events.append, guard)(_wait(5000), sleeps.append)
    assert sleeps == [1.0]
    assert guard.checks == 1
    assert [event["remaining_ms"] for event in events] == [5000]
