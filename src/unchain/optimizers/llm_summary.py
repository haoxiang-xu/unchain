from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from ..execution import ExecutionLeaseError
from .base import BaseContextOptimizer, OptimizerContext
from .common import estimate_tokens, replace_non_system_span, split_system_and_non_system, split_turns


SummaryGenerator = Callable[[str, list[dict[str, Any]], int, str], str]


def _flatten_turns(turns: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for turn in turns:
        flattened.extend(turn)
    return flattened


@dataclass(frozen=True)
class LlmSummaryOptimizerConfig:
    summary_trigger_pct: float = 0.75
    summary_target_pct: float = 0.45
    max_summary_chars: int = 2400
    summary_generator: SummaryGenerator | None = None


class LlmSummaryOptimizer(BaseContextOptimizer):
    def __init__(
        self,
        config: LlmSummaryOptimizerConfig | None = None,
        *,
        phases=("before_model",),
        order: int = 20,
    ) -> None:
        super().__init__(name="llm_summary", phases=phases, order=order)
        self.config = config or LlmSummaryOptimizerConfig()

    def build_optimizer_delta(self, context: OptimizerContext):
        bucket = context.optimizer_state()
        bucket["summary_triggered"] = False
        bucket["source_preservation_required"] = False
        bucket.pop("summary_skip_reason", None)
        bucket.pop("summary_fallback_reason", None)

        max_context_window_tokens = context.max_context_window_tokens
        if max_context_window_tokens <= 0:
            bucket["summary_skip_reason"] = "invalid_max_context_window"
            return self.state_only_delta(bucket=bucket)

        messages = context.latest_messages()
        estimated_tokens = estimate_tokens(messages)
        usage_pct = estimated_tokens / max_context_window_tokens
        bucket["summary_usage_pct"] = round(usage_pct, 4)

        if usage_pct < float(self.config.summary_trigger_pct):
            bucket["summary_skip_reason"] = "below_trigger_threshold"
            return self.state_only_delta(bucket=bucket)

        systems, non_system = split_system_and_non_system(messages)
        turns = split_turns(non_system)
        if len(turns) <= 1:
            bucket["summary_skip_reason"] = "insufficient_turns"
            return self.state_only_delta(bucket=bucket)

        target_tokens = max(1, int(max_context_window_tokens * float(self.config.summary_target_pct)))
        summary_budget_tokens = max(32, int(math.ceil(int(self.config.max_summary_chars) / 4.0)))
        running_tokens = estimate_tokens(systems) + summary_budget_tokens

        kept_turns: list[list[dict[str, Any]]] = []
        for turn in reversed(turns):
            turn_tokens = estimate_tokens(turn)
            if kept_turns and running_tokens + turn_tokens > target_tokens:
                break
            kept_turns.insert(0, turn)
            running_tokens += turn_tokens

        if len(kept_turns) >= len(turns):
            bucket["summary_skip_reason"] = "no_old_turns_to_summarize"
            return self.state_only_delta(bucket=bucket)

        old_turn_count = len(turns) - len(kept_turns)
        old_messages = _flatten_turns(turns[:old_turn_count])
        previous_summary = str(bucket.get("summary", ""))
        summary_generator = self.config.summary_generator
        if not callable(summary_generator):
            bucket["summary_fallback_reason"] = "summary_generator_missing"
            bucket["source_preservation_required"] = True
            return self.state_only_delta(bucket=bucket)

        try:
            summary_text = summary_generator(
                previous_summary,
                old_messages,
                int(self.config.max_summary_chars),
                context.model,
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            bucket["summary_fallback_reason"] = f"summary_generation_failed: {exc}"
            bucket["source_preservation_required"] = True
            return self.state_only_delta(bucket=bucket)

        summary_text = str(summary_text or "").strip()
        if not summary_text:
            bucket["summary_fallback_reason"] = "empty_summary"
            bucket["source_preservation_required"] = True
            return self.state_only_delta(bucket=bucket)

        if len(summary_text) > int(self.config.max_summary_chars):
            summary_text = summary_text[: int(self.config.max_summary_chars)].rstrip()

        bucket["summary"] = summary_text
        bucket["summary_triggered"] = True
        bucket["summary_old_turn_count"] = old_turn_count
        bucket["summary_kept_turn_count"] = len(kept_turns)
        bucket["summary_target_tokens"] = target_tokens
        bucket["source_preservation_required"] = False
        bucket.pop("summary_skip_reason", None)
        bucket.pop("summary_fallback_reason", None)

        updated_messages = replace_non_system_span(
            messages,
            _flatten_turns(kept_turns),
            injected_system_messages=[
                {"role": "system", "content": f"[CONTEXT SUMMARY]\n{summary_text}"},
            ],
        )
        return self.replace_messages_delta(
            context,
            updated_messages,
            bucket=bucket,
            trace={
                "summary_triggered": True,
                "summary_old_turn_count": old_turn_count,
                "summary_kept_turn_count": len(kept_turns),
            },
        )
