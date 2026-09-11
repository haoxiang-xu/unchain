from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..kernel.state import RunState
from .base import BaseContextOptimizer, OptimizerContext
from .common import estimate_tokens


@dataclass(frozen=True)
class ContextUsageOptimizerConfig:
    enabled: bool = True


class ContextUsageOptimizer(BaseContextOptimizer):
    """Measure usage for a request-only [Context Status] suffix.

    Runs at order 55 (after all other optimizers) so the usage stats reflect
    the post-optimization message state. Helps the agent make informed
    decisions about delegation vs direct tool use.
    """

    def __init__(
        self,
        config: ContextUsageOptimizerConfig | None = None,
        *,
        phases=("before_model",),
        order: int = 55,
    ) -> None:
        super().__init__(name="context_usage", phases=phases, order=order)
        self.config = config or ContextUsageOptimizerConfig()

    def build_optimizer_delta(self, context: OptimizerContext):
        bucket = context.optimizer_state()
        max_tokens = context.max_context_window_tokens
        if not self.config.enabled or max_tokens <= 0:
            if bucket.get("request_note") is None:
                return None
            bucket["request_note"] = None
            return self.state_only_delta(bucket=bucket)

        messages = context.latest_messages()
        current_tokens = estimate_tokens(messages)
        usage_pct = current_tokens / max_tokens if max_tokens > 0 else 0.0
        remaining_tokens = max(0, max_tokens - current_tokens)

        # Do not edit semantic history: a dynamic system message invalidates
        # the cache prefix; a synthetic user message becomes the latest task.
        # The request builder projects this record only after native replay.
        bucket["request_note"] = {
            "schema_version": 1,
            "context_version_id": context.latest_version_id,
            "iteration": context.state.iteration,
            "current_tokens": current_tokens,
            "max_tokens": max_tokens,
        }
        bucket["usage_pct"] = round(usage_pct, 4)
        bucket["current_tokens"] = current_tokens
        bucket["max_tokens"] = max_tokens
        bucket["remaining_tokens"] = remaining_tokens

        return self.state_only_delta(
            bucket=bucket,
            trace={
                "usage_pct": round(usage_pct, 4),
                "current_tokens": current_tokens,
                "remaining_tokens": remaining_tokens,
            },
        )


def context_usage_request_note(state: RunState) -> dict[str, Any] | None:
    """Project only current, runtime-owned telemetry, never matching prompt text."""
    record = state.optimizer_state.get("context_usage", {}).get("request_note")
    if record is None:
        return None
    fields = {"schema_version", "context_version_id", "iteration", "current_tokens", "max_tokens"}
    if (
        type(record) is not dict
        or set(record) != fields
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or not isinstance(record["context_version_id"], str)
        or not record["context_version_id"]
        or any(type(record[key]) is not int or record[key] < 0
               for key in ("iteration", "current_tokens", "max_tokens"))
        or record["max_tokens"] == 0
    ):
        raise ValueError("invalid context usage request note")
    if (
        record["context_version_id"] != state.latest_version_id
        or record["iteration"] != state.iteration
    ):
        return None
    current, maximum = record["current_tokens"], record["max_tokens"]
    return {
        "role": "user",
        "content": (
            "Runtime context information (not a new user request):\n"
            f"[Context Status] {current / maximum:.0%} used "
            f"({current:,}/{maximum:,} tokens). "
            f"~{max(0, maximum - current):,} remaining."
        ),
    }
