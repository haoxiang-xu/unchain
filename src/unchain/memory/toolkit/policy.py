"""Model-visible policy for proposing durable Memory V2 candidates."""

from __future__ import annotations

from dataclasses import dataclass

from unchain.kernel.delta import HarnessDelta, InsertMessagesOp, ReplaceSpanOp
from unchain.tools import ToolPromptSpec
from unchain.tools.base import BaseToolHarness, ToolContext


MEMORY_PROPOSAL_POLICY_VERSION = "unchain.memory.proposal_policy.v1"
MEMORY_LEARNING_POLICY_VERSION = "unchain.memory.learning_policy.v1"
MEMORY_LEARNING_BLOCK_START = "<memory_learning>"
MEMORY_LEARNING_BLOCK_END = "</memory_learning>"

MEMORY_PROPOSE_PROMPT_SPEC = ToolPromptSpec(
    purpose=(
        "Propose one auditable chat-memory candidate for future work; this never "
        "writes formal or long-term memory. Policy version: "
        f"{MEMORY_PROPOSAL_POLICY_VERSION}. Except for an explicit user request to "
        "remember, preserve, correct, or replace information, call this tool only "
        "when every Evidence, Future value, Durability, and Novelty test below passes."
    ),
    when_to_use=(
        (
            "Explicit intent — Handle a user's explicit request to remember, preserve, "
            "correct, or replace durable information, subject to secret and duplicate "
            "checks. When memory_search is available, search first; if equivalent "
            "memory already exists, do not create a duplicate."
        ),
        (
            "Evidence — The candidate comes from a direct user statement, a confirmed "
            "decision, or a verified tool or test result, not a guess or an unaccepted "
            "assistant proposal."
        ),
        (
            "Future value — After current context is gone, losing the information would "
            "likely cause a materially worse future action or force the user to repeat "
            "important information."
        ),
        (
            "Durability — The information remains useful beyond the current exchange. "
            "High-signal examples are stable user preferences and standing collaboration "
            "rules; confirmed project decisions, invariants, conventions, and constraints; "
            "verified reusable procedures and failure shields; durable environment facts "
            "that are costly to rediscover; and references to important durable artifacts."
        ),
        (
            "Novelty — The candidate adds, corrects, or materially contextualizes existing "
            "knowledge instead of paraphrasing it. For a likely correction, search when "
            "possible, use the intended target path, and explain what changed in rationale; "
            "never silently overwrite a conflict."
        ),
    ),
    when_not_to_use=(
        (
            "Never store passwords, tokens, credentials, private keys, secret values, or "
            "provider credentials. Secrets belong only in the Secret Vault."
        ),
        (
            "Do not store guesses, inferred personal traits, unaccepted assistant ideas, "
            "or lessons that have not been confirmed by the user or verified by execution."
        ),
        (
            "Do not store greetings, acknowledgements, small talk, one-off requests, "
            "temporary state, tentative ideas, rejected options, or common public knowledge."
        ),
        (
            "Do not store current objectives, progress, active plans, or open questions; "
            "those belong in Pinned Task State. Do not copy raw transcripts, logs, hidden "
            "reasoning, or full tool or subagent outputs into memory."
        ),
        (
            "Do not proactively propose highly sensitive personal information without the "
            "user's explicit request, and never infer sensitive attributes as facts."
        ),
    ),
    examples=(
        (
            "A user establishes a standing project rule or confirms an architectural "
            "invariant: propose the durable rule with its conditions and source references."
        ),
        (
            "A tested workaround reliably prevents a recurring failure: propose the "
            "verified failure shield, not the raw debug transcript."
        ),
        (
            "A user requests CSV formatting only for the current answer: do not propose it."
        ),
        (
            "A tool returns a large log or document: preserve it as an ArtifactRef and "
            "propose only a durable, verified takeaway or index when useful."
        ),
    ),
    advanced_tips=(
        (
            "Create one coherent, standalone candidate. Preserve exact names, dates, "
            "quantities, conditions, uncertainty, meaningful path and description, and "
            "source references when available."
        ),
        (
            "A normal proposal is chat-scoped. Long-term memory requires a separate, "
            "user-confirmed promotion."
        ),
        (
            "After memory_propose returns a non-empty candidate_ref, say only that a "
            "memory candidate was proposed. Any tool error means no candidate was "
            "proposed: do not claim success, and correct model-supplied arguments before "
            "retrying. Claim formal memory was saved only when a curator or formal-write "
            "result confirms it. If nothing passes this policy, do not call the tool, "
            "and never interrupt the primary task merely to manufacture a memory."
        ),
    ),
)


def _is_memory_learning_message(message: dict[str, object]) -> bool:
    if not isinstance(message, dict) or message.get("role") != "system":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    return stripped.startswith(MEMORY_LEARNING_BLOCK_START) and stripped.endswith(
        MEMORY_LEARNING_BLOCK_END
    )


def render_memory_learning_block(toolkit: object) -> str:
    """Render the root-only terminal learning instruction when its tool exists."""

    tools = getattr(toolkit, "tools", None)
    if not isinstance(tools, dict) or "memory_propose" not in tools:
        return ""
    return (
        f"{MEMORY_LEARNING_BLOCK_START}\n"
        f"Policy version: {MEMORY_LEARNING_POLICY_VERSION}.\n"
        "Before returning the final answer for this root conversation, evaluate "
        "whether one durable fact from the conversation satisfies the complete "
        "memory_propose policy shown in the tools guidance. If one qualifies, call "
        "memory_propose before the final answer. If none qualifies, continue to the "
        "final answer without calling it; the completion audit will record that "
        "outcome. Never weaken the Evidence, Future value, Durability, Novelty, "
        "duplicate, sensitivity, or secret checks merely to create a memory. Never "
        "copy a raw transcript, hidden reasoning, or full tool output. If a candidate "
        "was already proposed successfully during this run, do not propose it again. "
        "Treat the proposal as successful only when the tool result contains a non-empty "
        "candidate_ref; a tool error means nothing was proposed and must never be "
        "reported as saved or proposed.\n"
        f"{MEMORY_LEARNING_BLOCK_END}"
    )


@dataclass
class MemoryLearningPromptHarness(BaseToolHarness):
    """Keep one root-only learning instruction aligned with tool exposure."""

    name: str = "memory_learning_prompt"
    phases: tuple[str, ...] = ("before_model",)
    order: int = 255

    def build_tool_delta(self, context: ToolContext) -> HarnessDelta | None:
        messages = context.latest_messages()
        rendered = render_memory_learning_block(context.toolkit)
        existing = [
            index
            for index, message in enumerate(messages)
            if _is_memory_learning_message(message)
        ]
        if not rendered:
            if not existing:
                return None
            return HarnessDelta(
                created_by=self.created_by,
                ops=(
                    ReplaceSpanOp(
                        start=existing[0],
                        end=existing[-1] + 1,
                        messages=[],
                    ),
                ),
            )

        new_message = {"role": "system", "content": rendered}
        if existing:
            if len(existing) == 1 and messages[existing[0]] == new_message:
                return None
            return HarnessDelta(
                created_by=self.created_by,
                ops=(
                    ReplaceSpanOp(
                        start=existing[0],
                        end=existing[-1] + 1,
                        messages=[new_message],
                    ),
                ),
            )

        insert_index = 0
        while insert_index < len(messages):
            message = messages[insert_index]
            if not isinstance(message, dict) or message.get("role") != "system":
                break
            insert_index += 1
        return HarnessDelta(
            created_by=self.created_by,
            ops=(InsertMessagesOp(index=insert_index, messages=[new_message]),),
        )


__all__ = [
    "MEMORY_LEARNING_BLOCK_END",
    "MEMORY_LEARNING_BLOCK_START",
    "MEMORY_LEARNING_POLICY_VERSION",
    "MEMORY_PROPOSAL_POLICY_VERSION",
    "MEMORY_PROPOSE_PROMPT_SPEC",
    "MemoryLearningPromptHarness",
    "render_memory_learning_block",
]
