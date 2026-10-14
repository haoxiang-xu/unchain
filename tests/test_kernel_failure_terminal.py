"""#389: a failed or cancelled kernel run emits exactly one canonical terminal."""

from __future__ import annotations

import pytest

from unchain.events.normalizer import RuntimeEventNormalizerContext, normalize_raw_event
from unchain.execution import (
    ActiveExecutionLeaseError,
    ExecutionCancelledError,
    ExecutionLeaseExpiredError,
    ExecutionLeaseNotOwnedError,
    StaleExecutionLeaseError,
)
from unchain.kernel import ModelTurnResult
from unchain.runtime import build_runtime_loop

TERMINALS = {"run_completed", "run_failed", "run_cancelled"}
CONTEXT = RuntimeEventNormalizerContext(session_id="session-389", root_run_id="run-389")


class _RaisingModelIO:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.requests = []

    def fetch_turn(self, request):
        self.requests.append(request)
        raise self.error


class _OkModelIO:
    def fetch_turn(self, request):
        return ModelTurnResult(
            assistant_messages=[{"role": "assistant", "content": "done"}],
            tool_calls=[],
            final_text="done",
            response_id="resp-ok",
        )


def _run(model_io, *, callback):
    return build_runtime_loop(model_io=model_io).run(
        [{"role": "user", "content": "hello"}],
        provider="openai",
        model="gpt-4.1",
        max_iterations=1,
        callback=callback,
        run_id="run-389",
    )


def _terminals(events):
    return [event for event in events if event["type"] in TERMINALS]


def test_provider_failure_emits_one_run_failed_and_still_raises():
    events = []
    with pytest.raises(RuntimeError, match="private provider text"):
        _run(_RaisingModelIO(RuntimeError("private provider text")), callback=events.append)

    terminals = _terminals(events)
    assert [event["type"] for event in terminals] == ["run_failed"]
    failed = terminals[0]
    assert failed["run_id"] == "run-389"
    assert failed["status"] == "failed"
    assert failed["code"] == "kernel_run_failed"
    assert failed["message"] == "Run failed"
    assert failed["recoverable"] is False
    assert "private provider text" not in repr(failed)
    # The terminal comes after run_started and is the last event of the run.
    assert events[0]["type"] == "run_started"
    assert events[-1] is failed


def test_failure_code_uses_the_sanitized_exception_code():
    class CodedError(RuntimeError):
        code = "provider_overloaded"

    events = []
    with pytest.raises(CodedError):
        _run(_RaisingModelIO(CodedError("x")), callback=events.append)
    assert _terminals(events)[0]["code"] == "provider_overloaded"


def test_cancelled_execution_emits_one_run_cancelled():
    events = []
    with pytest.raises(ExecutionCancelledError):
        _run(_RaisingModelIO(ExecutionCancelledError("stopped")), callback=events.append)

    terminals = _terminals(events)
    assert [event["type"] for event in terminals] == ["run_cancelled"]
    cancelled = terminals[0]
    assert cancelled["status"] == "cancelled"
    assert cancelled["code"] == "execution_cancelled"
    assert cancelled["message"] == "Run cancelled"
    assert cancelled["recoverable"] is False
    assert "stopped" not in repr(cancelled)


def test_successful_run_still_emits_only_run_completed():
    events = []
    result = _run(_OkModelIO(), callback=events.append)
    assert result.status == "completed"
    assert [event["type"] for event in _terminals(events)] == ["run_completed"]


def test_a_failing_callback_never_masks_the_original_error():
    def callback(event):
        if event["type"] == "run_failed":
            raise OSError("journal unavailable")

    with pytest.raises(RuntimeError, match="provider down"):
        _run(_RaisingModelIO(RuntimeError("provider down")), callback=callback)


def test_no_second_terminal_when_the_completed_terminal_was_already_attempted():
    events = []

    def callback(event):
        events.append(event)
        if event["type"] == "run_completed":
            raise OSError("journal unavailable after the terminal")

    with pytest.raises(OSError):
        _run(_OkModelIO(), callback=callback)
    assert [event["type"] for event in _terminals(events)] == ["run_completed"]


def test_run_failed_projects_to_the_v4_failed_event_with_the_fixed_text():
    events = []
    with pytest.raises(RuntimeError):
        _run(_RaisingModelIO(RuntimeError("private")), callback=events.append)
    failed = _terminals(events)[0]

    drafts = normalize_raw_event(failed, context=CONTEXT)
    assert [draft.type for draft in drafts] == ["run.failed"]
    assert drafts[0].payload == {
        "status": "failed",
        "error": {"code": "kernel_run_failed", "message": "Run failed"},
        "recoverable": False,
    }


def test_run_cancelled_has_no_v4_projection_so_stop_adds_no_ui_event():
    events = []
    with pytest.raises(ExecutionCancelledError):
        _run(_RaisingModelIO(ExecutionCancelledError("stopped")), callback=events.append)
    cancelled = _terminals(events)[0]
    assert normalize_raw_event(cancelled, context=CONTEXT) == []


def test_a_cancel_carried_by_the_host_error_code_is_cancelled_not_failed():
    """The sidecar cancels with a plain RuntimeError that carries the code."""

    class HostCancel(RuntimeError):
        code = "execution_cancelled"

    events = []
    with pytest.raises(HostCancel):
        _run(_RaisingModelIO(HostCancel("execution attempt was cancelled")), callback=events.append)
    terminals = _terminals(events)
    assert [event["type"] for event in terminals] == ["run_cancelled"]
    assert terminals[0]["code"] == "execution_cancelled"
    assert terminals[0]["status"] == "cancelled"


@pytest.mark.parametrize(
    "error",
    [
        ExecutionLeaseNotOwnedError("lost"),
        ExecutionLeaseExpiredError("expired"),
        StaleExecutionLeaseError("superseded"),
        ActiveExecutionLeaseError("taken"),
    ],
)
def test_lost_ownership_emits_no_terminal_for_an_attempt_another_owner_may_hold(error):
    events = []
    with pytest.raises(type(error)):
        _run(_RaisingModelIO(error), callback=events.append)
    assert _terminals(events) == []



@pytest.mark.parametrize("execution_id,attempt_id,deferred", [
    ("run-389", "run-389", True),
    ("other-execution", "run-389", False),
    ("run-389", "other-attempt", False),
])
def test_durable_settlement_deferral_requires_exact_identity(execution_id, attempt_id, deferred):
    from unchain.interaction import DurableInteractionSettlementPending
    error = DurableInteractionSettlementPending(
        execution_id=execution_id, attempt_id=attempt_id, interaction_id="interaction-389",
    )
    events = []
    with pytest.raises(DurableInteractionSettlementPending):
        _run(_RaisingModelIO(error), callback=events.append)
    assert bool(_terminals(events)) is not deferred
    assert str(error) == "Durable interaction settlement is pending"
