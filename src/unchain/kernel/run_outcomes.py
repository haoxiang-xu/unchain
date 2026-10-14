from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .lifecycle_events import (
    build_final_message_payload,
    build_run_completed_payload,
    build_run_failure_payload,
    build_run_max_iterations_payload,
)
from .results import build_kernel_run_result
from .state import RunState
from .types import KernelRunResult

EmitEvent = Callable[..., None]
DispatchRunFinalizing = Callable[..., None]
DispatchOnSuspend = Callable[..., None]


def _completed_iteration(state: RunState) -> int:
    return max(0, int(state.iteration) - 1)


def finish_completed_run(
    state: RunState,
    *,
    callback: Any,
    run_id: str,
    emit_event: EmitEvent,
    dispatch_run_finalizing: DispatchRunFinalizing,
) -> KernelRunResult:
    state.run_status = "completed"
    dispatch_run_finalizing(
        state,
        callback=callback,
        run_id=run_id,
        iteration=_completed_iteration(state),
        status="completed",
    )
    emit_event(
        callback,
        "final_message",
        run_id,
        **build_final_message_payload(state),
    )
    emit_event(
        callback,
        "run_completed",
        run_id,
        **build_run_completed_payload(state, status="completed"),
    )
    return build_kernel_run_result(state, status="completed")


# Events that end an attempt in the journal. A run that has attempted one of
# these never gets a second, failure-shaped terminal.
TERMINAL_EVENT_TYPES = frozenset(
    {
        "run_completed",
        "run_failed",
        "run_cancelled",
        "run_max_iterations",
    }
)


def emit_run_failure_terminal(
    *,
    callback: Any,
    run_id: str,
    iteration: int,
    error_code: str,
    cancelled: bool,
    emit_event: EmitEvent,
) -> None:
    """Emit the one canonical terminal for a failed or cancelled run.

    Journal consumers (rebase preflight, projector, graph recovery) already
    treat ``run_failed`` and ``run_cancelled`` as terminals; nothing produced
    them before, so the attempt stayed open forever.
    """

    emit_event(
        callback,
        "run_cancelled" if cancelled else "run_failed",
        run_id,
        **build_run_failure_payload(
            iteration=iteration,
            error_code=error_code,
            cancelled=cancelled,
        ),
    )


def finish_max_iterations_run(
    state: RunState,
    *,
    callback: Any,
    run_id: str,
    emit_run_max_iterations: bool,
    emit_event: EmitEvent,
    dispatch_run_finalizing: DispatchRunFinalizing,
) -> KernelRunResult:
    state.run_status = "max_iterations"
    dispatch_run_finalizing(
        state,
        callback=callback,
        run_id=run_id,
        iteration=int(state.iteration),
        status="max_iterations",
    )
    if emit_run_max_iterations:
        emit_event(
            callback,
            "run_max_iterations",
            run_id,
            **build_run_max_iterations_payload(state),
        )
    return build_kernel_run_result(state, status="max_iterations")


def finish_awaiting_human_input_run(
    state: RunState,
    *,
    callback: Any,
    run_id: str,
    dispatch_on_suspend: DispatchOnSuspend,
) -> KernelRunResult:
    state.run_status = "awaiting_human_input"
    dispatch_on_suspend(
        state,
        callback=callback,
        run_id=run_id,
        iteration=int(state.iteration),
        status="awaiting_human_input",
    )
    return build_kernel_run_result(state, status="awaiting_human_input")


def finish_awaiting_interaction_run(
    state: RunState,
    *,
    callback: Any,
    run_id: str,
    dispatch_on_suspend: DispatchOnSuspend,
) -> KernelRunResult:
    state.run_status = "awaiting_interaction"
    dispatch_on_suspend(
        state,
        callback=callback,
        run_id=run_id,
        iteration=int(state.iteration),
        status="awaiting_interaction",
    )
    return build_kernel_run_result(state, status="awaiting_interaction")


__all__ = [
    "finish_awaiting_human_input_run",
    "finish_awaiting_interaction_run",
    "finish_completed_run",
    "finish_max_iterations_run",
]
