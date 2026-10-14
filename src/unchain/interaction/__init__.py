from __future__ import annotations

from .btw import ProgressDigest, build_btw_prompt
from .human_input import (
    ASK_USER_QUESTION_TOOL_NAME,
    HUMAN_INPUT_KIND_SELECTOR,
    HUMAN_INPUT_OTHER_VALUE,
    HumanInputOption,
    HumanInputRequest,
    HumanInputResponse,
    build_ask_user_question_tool,
    is_human_input_tool_name,
)
from .effects import (
    INTERACTION_EFFECT_CREATED_BY,
    build_human_input_continuation,
    build_human_input_requested_event,
    build_human_input_suspend_request,
    build_max_budget_continuation,
    build_max_budget_suspend_request,
    build_tool_approval_continuation,
    build_tool_approval_suspend_request,
)
from .durable import (
    INTERACTION_JOURNAL_KEY,
    INTERACTION_KIND_HUMAN_INPUT,
    INTERACTION_KIND_MAX_BUDGET,
    INTERACTION_KIND_TOOL_APPROVAL,
    DurableInteractionSettlementPending,
    InteractionAlreadyAppliedError,
    InteractionError,
    InteractionIntegrityError,
    InteractionNotPendingError,
    InteractionReceipt,
    InteractionReceiptConflictError,
    InteractionRequest,
    build_interaction_receipt,
    build_interaction_request,
)
from .fyi import FyiChannel, FyiInjectionHarness, FyiMessage, wrap_fyi
from .resume import (
    HumanInputResumeHarness,
    HumanInputResumePlan,
    hydrate_human_input_resume_state,
    parse_human_input_request,
    prepare_human_input_resume_plan,
)
from .queue_turns import QueuedTurnBuffer, merge_queued_turn_texts

# Deprecated aliases (steer -> queued-turns rename, 0.2.0): resolved lazily in
# __getattr__ below so that `from unchain.interaction import SteerBuffer` keeps
# working with a DeprecationWarning, while plain `import unchain.interaction`
# stays silent. Shim removal is slated for the next minor release.
_DEPRECATED_ALIASES = {
    "SteerBuffer": "QueuedTurnBuffer",
    "merge_steered_texts": "merge_queued_turn_texts",
}


def __getattr__(name: str):
    replacement = _DEPRECATED_ALIASES.get(name)
    if replacement is not None:
        import warnings

        warnings.warn(
            f"unchain.interaction.{name} is deprecated and will be removed in "
            f"the next minor release — use unchain.interaction.{replacement} "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[replacement]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DurableInteractionSettlementPending",
    "ASK_USER_QUESTION_TOOL_NAME",
    "HUMAN_INPUT_KIND_SELECTOR",
    "HUMAN_INPUT_OTHER_VALUE",
    "HumanInputOption",
    "HumanInputRequest",
    "HumanInputResponse",
    "HumanInputResumeHarness",
    "HumanInputResumePlan",
    "INTERACTION_EFFECT_CREATED_BY",
    "INTERACTION_JOURNAL_KEY",
    "INTERACTION_KIND_HUMAN_INPUT",
    "INTERACTION_KIND_MAX_BUDGET",
    "INTERACTION_KIND_TOOL_APPROVAL",
    "InteractionAlreadyAppliedError",
    "InteractionError",
    "InteractionIntegrityError",
    "InteractionNotPendingError",
    "InteractionReceipt",
    "InteractionReceiptConflictError",
    "InteractionRequest",
    "FyiChannel",
    "FyiInjectionHarness",
    "FyiMessage",
    "ProgressDigest",
    "QueuedTurnBuffer",
    "build_ask_user_question_tool",
    "build_btw_prompt",
    "build_human_input_continuation",
    "build_human_input_requested_event",
    "build_human_input_suspend_request",
    "build_interaction_receipt",
    "build_interaction_request",
    "build_max_budget_continuation",
    "build_max_budget_suspend_request",
    "build_tool_approval_continuation",
    "build_tool_approval_suspend_request",
    "hydrate_human_input_resume_state",
    "is_human_input_tool_name",
    "merge_queued_turn_texts",
    "parse_human_input_request",
    "prepare_human_input_resume_plan",
    "wrap_fyi",
]
