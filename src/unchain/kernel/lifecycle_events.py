from __future__ import annotations

from typing import Any

from .results import build_legacy_run_bundle
from .state import RunState
from .types import ModelTurnResult


def _completed_iteration(state: RunState) -> int:
    return max(0, int(state.iteration) - 1)


def last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages or []):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def build_run_started_payload(state: RunState) -> dict[str, Any]:
    return {
        "iteration": int(state.iteration),
        "provider": state.provider_state.provider,
        "model": state.provider_state.model,
    }


def build_run_max_iterations_payload(state: RunState) -> dict[str, Any]:
    return {
        "iteration": int(state.iteration),
        "bundle": build_legacy_run_bundle(state, status="max_iterations"),
    }


def build_max_iterations_decision_payload(state: RunState, *, max_iterations: int) -> dict[str, Any]:
    return {
        "iteration": int(state.iteration),
        "max_iterations": int(max_iterations),
        "consumed_tokens": int(state.token_state.consumed_tokens),
    }


def build_iteration_started_payload(state: RunState) -> dict[str, Any]:
    return {"iteration": int(state.iteration)}


def build_response_received_payload(state: RunState, turn: ModelTurnResult) -> dict[str, Any]:
    return {
        "iteration": _completed_iteration(state),
        "response_id": turn.response_id,
        "has_tool_calls": bool(turn.tool_calls),
        "status": state.run_status,
        "bundle": build_legacy_run_bundle(
            state,
            status="running" if turn.tool_calls else "completed",
        ),
    }


def build_iteration_completed_payload(
    state: RunState,
    *,
    has_tool_calls: bool,
) -> dict[str, Any]:
    return {
        "iteration": _completed_iteration(state),
        "has_tool_calls": bool(has_tool_calls),
    }


def build_final_message_payload(state: RunState) -> dict[str, Any]:
    return {
        "iteration": _completed_iteration(state),
        "content": last_assistant_text(state.transcript),
    }


def build_run_completed_payload(state: RunState, *, status: str) -> dict[str, Any]:
    return {
        "iteration": _completed_iteration(state),
        "status": status,
        "bundle": build_legacy_run_bundle(state, status=status),
    }


def build_provider_retry_payload(
    wait: Any,
    *,
    provider: str,
    remaining_ms: int,
) -> dict[str, Any]:
    """Closed payload of a ``provider_retry`` event (BC-386-4).

    Status fields come only from the closed failure diagnostic, never from
    provider text.
    """

    diagnostic = getattr(wait, "diagnostic", None)
    return {
        "provider": str(provider or ""),
        "attempt_failed": int(wait.attempt_failed),
        "next_attempt": int(wait.next_attempt),
        "max_attempts": int(wait.max_attempts),
        "delay_ms": int(wait.delay_ms),
        "remaining_ms": max(0, int(remaining_ms)),
        "http_status": diagnostic.http_status if diagnostic is not None else None,
        "provider_status": diagnostic.provider_status if diagnostic is not None else "",
    }


__all__ = [
    "build_provider_retry_payload",
    "build_final_message_payload",
    "build_iteration_completed_payload",
    "build_iteration_started_payload",
    "build_max_iterations_decision_payload",
    "build_response_received_payload",
    "build_run_completed_payload",
    "build_run_max_iterations_payload",
    "build_run_started_payload",
    "last_assistant_text",
]
