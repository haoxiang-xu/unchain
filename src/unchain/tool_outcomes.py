from __future__ import annotations

from typing import Any


DURABLE_TOOL_RESULT_OUTCOMES = frozenset(
    {"success", "error", "denied", "cancelled"}
)


def classify_durable_tool_result(result: Any) -> str:
    """Classify a sealed tool result without inspecting a presentation wrapper."""

    if not isinstance(result, dict):
        return "success"
    if result.get("denied") is True or result.get("status") == "denied":
        return "denied"
    if (
        result.get("cancelled") is True
        or result.get("canceled") is True
        or result.get("status") in {"cancelled", "canceled"}
    ):
        return "cancelled"
    if (
        result.get("ok") is False
        or result.get("error") is not None
        or result.get("status") in {"error", "failed"}
    ):
        return "error"
    return "success"
