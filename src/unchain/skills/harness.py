from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..kernel.delta import HarnessDelta, InsertMessagesOp, ReplaceSpanOp
from ..tools.base import BaseToolHarness, ToolContext
from .activation import (
    ActivationQueue,
    ActiveSkillSet,
    find_user_invocation_tokens,
    is_new_user_turn,
    latest_real_user_text,
    user_turn_activation_label,
    user_turn_identity,
)
from .models import SkillDiagnostic
from .journal import JournalSkillState
from .registry import SkillRegistry
from .rendering import (
    is_active_skills_message,
    is_skill_catalog_message,
    render_skill_catalog,
)

SKILLS_STATE_BUCKET = "skills"


def _leading_system_count(messages: Sequence[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "developer"}:
            break
        count += 1
    return count


def _block_ops(
    messages: Sequence[dict[str, Any]],
    existing: list[int],
    rendered: str,
    *,
    remove_when_empty: bool,
) -> tuple[Any, ...] | None:
    """Ops that insert / replace / remove one delimited system block; None when unchanged."""

    if not rendered:
        if not existing or not remove_when_empty:
            return None
        return (ReplaceSpanOp(start=existing[0], end=existing[-1] + 1, messages=[]),)
    new_message = {"role": "system", "content": rendered}
    if existing:
        if len(existing) == 1 and messages[existing[0]] == new_message:
            return None
        return (ReplaceSpanOp(start=existing[0], end=existing[-1] + 1, messages=[new_message]),)
    return (InsertMessagesOp(index=_leading_system_count(messages), messages=[new_message]),)


def _block_delta(
    *,
    created_by: str,
    messages: list[dict[str, Any]],
    existing: list[int],
    rendered: str,
    remove_when_empty: bool,
    state_updates: dict[str, Any] | None = None,
) -> HarnessDelta | None:
    """Insert / replace / remove one delimited system block (ToolPromptHarness pattern)."""

    ops = _block_ops(messages, existing, rendered, remove_when_empty=remove_when_empty)
    if ops is None:
        if not state_updates:
            return None
        return HarnessDelta(created_by=created_by, state_updates=state_updates)
    return HarnessDelta(created_by=created_by, ops=ops, state_updates=state_updates or {})


def _transcript_with_block(
    transcript: Sequence[dict[str, Any]],
    rendered: str,
    is_block: Any,
) -> list[dict[str, Any]] | None:
    """The transcript with the block inserted/replaced, or None when already current.

    The kernel rebuilds the working messages from `state.transcript` on every
    step and persists the transcript in execution checkpoints, so a durable
    block must live there too; model-context edits alone evaporate.
    """

    if not rendered:
        return None
    new_message = {"role": "system", "content": rendered}
    existing = [index for index, message in enumerate(transcript) if is_block(message)]
    updated = [dict(message) for message in transcript]
    if existing:
        if len(existing) == 1 and transcript[existing[0]] == new_message:
            return None
        updated[existing[0] : existing[-1] + 1] = [new_message]
        return updated
    updated.insert(_leading_system_count(transcript), new_message)
    return updated


@dataclass
class SkillCatalogHarness(BaseToolHarness):
    """Keeps one `<available_skills>` system block in sync with the registry inventory."""

    registry: SkillRegistry = field(kw_only=True)
    tool_name: str = "skill"
    description_max_length: int = 500
    name: str = "skill_catalog"
    phases: tuple[str, ...] = ("before_model",)
    order: int = 260

    def build_tool_delta(self, context: ToolContext) -> HarnessDelta | None:
        messages = context.latest_messages()
        inventory = self.registry.list()
        rendered = render_skill_catalog(
            inventory.skills,
            tool_name=self.tool_name,
            description_max_length=self.description_max_length,
        )
        existing = [index for index, message in enumerate(messages) if is_skill_catalog_message(message)]
        bucket = context.state.component_bucket(SKILLS_STATE_BUCKET)
        bucket["inventory_revision"] = inventory.revision
        bucket["inventory_diagnostics"] = [_diagnostic_dict(item) for item in inventory.diagnostics]
        return _block_delta(
            created_by=self.created_by,
            messages=messages,
            existing=existing,
            rendered=rendered,
            remove_when_empty=True,
        )


@dataclass
class SkillActivationHarness(BaseToolHarness):
    """Projects the durable activation snapshot into one `<active_skills>` system block.

    Runs before each model call (explicit `/name` activations from the latest
    real user message) and after each tool batch (activations queued by the
    `skill` tool), so tool activations are committed with their tool result.
    The rendered block is the only durable authority: on every run it is parsed
    back first, then new activations are merged and the block re-rendered.
    """

    registry: SkillRegistry = field(kw_only=True)
    queue: ActivationQueue = field(kw_only=True)
    journal_binding: Callable | None = field(default=None, kw_only=True, repr=False)
    name: str = "skill_activation"
    phases: tuple[str, ...] = ("before_model", "after_tool_batch")
    order: int = 262

    def build_tool_delta(self, context: ToolContext) -> HarnessDelta | None:
        messages = context.latest_messages()
        active, existing = ActiveSkillSet.from_messages(messages)
        binding = self.journal_binding(context.harness_context) if self.journal_binding else None
        journal = JournalSkillState(*binding) if binding is not None else None
        if journal is not None and journal.block is not None:
            active = ActiveSkillSet.from_block(journal.block)
        diagnostics: list[SkillDiagnostic] = []
        changed = False

        if context.phase == "before_model":
            changed |= self._apply_user_invocations(
                messages, active, diagnostics, transcript=context.state.transcript,
                journal=journal,
            )
        changed |= self._apply_queued(context, active, diagnostics)

        self.queue.remember(active)
        bucket = context.state.component_bucket(SKILLS_STATE_BUCKET)
        bucket["activation"] = active.to_state()
        if diagnostics:
            bucket["activation_diagnostics"] = [_diagnostic_dict(item) for item in diagnostics]
        del changed  # the block comparisons below decide whether a delta is needed
        rendered = active.render()
        if journal is not None:
            journal.save(rendered)
        transcript = _transcript_with_block(
            context.state.transcript, rendered, is_active_skills_message
        )
        return _block_delta(
            created_by=self.created_by,
            messages=messages,
            existing=existing,
            rendered=rendered,
            remove_when_empty=False,
            state_updates={"transcript": transcript} if transcript is not None else None,
        )

    def _apply_user_invocations(
        self,
        messages: list[dict[str, Any]],
        active: ActiveSkillSet,
        diagnostics: list[SkillDiagnostic],
        *,
        transcript: Sequence[dict[str, Any]] = (),
        journal: JournalSkillState | None = None,
    ) -> bool:
        located = latest_real_user_text(messages)
        if located is None:
            return False
        _index, text = located
        turn = (journal.turn(text) if journal is not None else None) or user_turn_identity(
            text, transcript=transcript, messages=messages
        )
        if not is_new_user_turn(turn, active.processed_turn):
            # Replay of an already processed turn (tool loop, retry, resume,
            # cold restart): the persisted snapshot is authoritative. No live
            # resolution, no file read — even if a higher-ranked source or a
            # new revision appeared in the meantime.
            return False
        tokens = find_user_invocation_tokens(text)
        if not tokens:
            # A turn without slash tokens binds nothing; leave the marker on
            # the last turn that did (a later /name turn still has a higher
            # ordinal, so it is recognised as new).
            return False
        if len(active) == 0 and not any(self.registry.resolve(t) is not None for t in tokens):
            # Nothing to record durably (no block exists and nothing resolves).
            return False
        active.processed_turn = turn
        changed = True  # the header marker changes even when no entry does
        label = user_turn_activation_label(turn)
        reserved = {item.casefold() for item in self.registry.config.reserved_commands}
        for token in tokens:
            if token.casefold() in reserved:
                continue
            summary = self.registry.resolve(token)
            if summary is None:
                continue  # unknown tokens stay literal
            if not summary.user_invocable:
                diagnostics.append(
                    SkillDiagnostic(
                        kind="user_invocation_denied",
                        name=summary.name,
                        source=summary.source,
                        source_id=summary.source_id,
                        message=f"skill '{summary.name}' is not user-invocable",
                    )
                )
                continue
            loaded = self.registry.get(summary.name)
            if loaded is None:
                diagnostics.append(
                    SkillDiagnostic(
                        kind="inaccessible",
                        name=summary.name,
                        source=summary.source,
                        source_id=summary.source_id,
                        message=f"skill '{summary.name}' could not be loaded",
                    )
                )
                continue
            try:
                status = active.apply(loaded, activation=label)
            except ValueError as exc:
                diagnostics.append(
                    SkillDiagnostic(
                        kind="body_delimiter",
                        name=summary.name,
                        source=summary.source,
                        source_id=summary.source_id,
                        message=str(exc),
                    )
                )
                continue
            changed |= status != "already_active"
        return changed

    def _apply_queued(
        self,
        context: ToolContext,
        active: ActiveSkillSet,
        diagnostics: list[SkillDiagnostic],
    ) -> bool:
        pending = self.queue.drain()
        if not pending:
            return False
        call_ids = _skill_tool_call_ids(context, tool_name=self.registry.config.tool_name)
        changed = False
        for position, (loaded, activation) in enumerate(pending):
            label = activation
            if activation == "tool" and position < len(call_ids):
                label = f"tool:{call_ids[position]}"
            try:
                status = active.apply(loaded, activation=label)
            except ValueError as exc:
                diagnostics.append(
                    SkillDiagnostic(
                        kind="body_delimiter",
                        name=loaded.summary.name,
                        source=loaded.summary.source,
                        source_id=loaded.summary.source_id,
                        message=str(exc),
                    )
                )
                continue
            changed |= status != "already_active"
        return changed


def _skill_tool_call_ids(context: ToolContext, *, tool_name: str) -> list[str]:
    tool_calls = context.raw_event.get("tool_calls")
    if not isinstance(tool_calls, Sequence):
        return []
    ids: list[str] = []
    for call in tool_calls:
        name = getattr(call, "name", None)
        call_id = getattr(call, "call_id", None)
        if name == tool_name and isinstance(call_id, str) and call_id:
            ids.append(call_id)
    return ids


def _diagnostic_dict(item: SkillDiagnostic) -> dict[str, str]:
    return {
        "kind": item.kind,
        "name": item.name,
        "source": item.source,
        "source_id": item.source_id,
        "message": item.message,
    }


__all__ = [
    "SKILLS_STATE_BUCKET",
    "SkillActivationHarness",
    "SkillCatalogHarness",
    "is_active_skills_message",
]
