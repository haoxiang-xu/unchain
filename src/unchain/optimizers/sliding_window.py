from __future__ import annotations

from dataclasses import dataclass

from .base import BaseContextOptimizer, OptimizerContext
from .common import estimate_tokens, replace_non_system_span, split_system_and_non_system, split_turns


def _flatten_turns(turns: list[list[dict]]) -> list[dict]:
    flattened: list[dict] = []
    for turn in turns:
        flattened.extend(turn)
    return flattened


@dataclass(frozen=True)
class SlidingWindowOptimizerConfig:
    max_window_pct: float = 0.7
    max_window_tokens: int | None = None


class SlidingWindowOptimizer(BaseContextOptimizer):
    def __init__(
        self,
        config: SlidingWindowOptimizerConfig | None = None,
        *,
        phases=("before_model",),
        order: int = 25,
    ) -> None:
        super().__init__(name="sliding_window", phases=phases, order=order)
        self.config = config or SlidingWindowOptimizerConfig()

    def _compute_effective_budget(self, context: OptimizerContext) -> int | None:
        max_ctx = context.max_context_window_tokens
        pct_budget = int(max_ctx * self.config.max_window_pct) if max_ctx > 0 else None
        abs_budget = self.config.max_window_tokens

        if pct_budget is not None and abs_budget is not None:
            return min(pct_budget, abs_budget)
        return pct_budget or abs_budget

    def build_optimizer_delta(self, context: OptimizerContext):
        budget = self._compute_effective_budget(context)
        if budget is None or budget <= 0:
            return None

        messages = context.latest_messages()
        system_msgs, non_system = split_system_and_non_system(messages)
        system_tokens = estimate_tokens(system_msgs)
        remaining_budget = budget - system_tokens

        turns = split_turns(non_system)
        total_turn_count = len(turns)
        forced_keep_latest_turn = False

        if remaining_budget <= 0:
            kept_turns: list[list[dict]] = []
        else:
            kept_turns = []
            tokens_used = 0
            for turn in reversed(turns):
                turn_tokens = estimate_tokens(turn)
                if tokens_used + turn_tokens > remaining_budget:
                    break
                kept_turns.insert(0, turn)
                tokens_used += turn_tokens

        # Never let optimization erase all non-system context when there are
        # still turns available. Anthropic requires at least one chat message,
        # and preserving the latest full turn keeps tool/result pairs intact.
        if total_turn_count > 0 and not kept_turns:
            kept_turns = [turns[-1]]
            forced_keep_latest_turn = True

        kept_turn_count = len(kept_turns)
        dropped_turn_count = total_turn_count - kept_turn_count
        tokens_used_final = estimate_tokens(_flatten_turns(kept_turns))

        bucket = context.optimizer_state()
        bucket["effective_budget_tokens"] = budget
        bucket["system_tokens"] = system_tokens
        bucket["kept_turn_count"] = kept_turn_count
        bucket["total_turn_count"] = total_turn_count
        bucket["dropped_turn_count"] = dropped_turn_count
        bucket["tokens_used"] = tokens_used_final
        bucket["forced_keep_latest_turn"] = forced_keep_latest_turn

        trace = {
            "kept_turn_count": kept_turn_count,
            "total_turn_count": total_turn_count,
            "dropped_turn_count": dropped_turn_count,
            "budget_tokens": budget,
            "used_tokens": tokens_used_final,
            "forced_keep_latest_turn": forced_keep_latest_turn,
        }

        updated = replace_non_system_span(messages, _flatten_turns(kept_turns))
        return self.replace_messages_delta(context, updated, bucket=bucket, trace=trace)
