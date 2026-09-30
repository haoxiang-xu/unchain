from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .models import ActiveSkill, LoadedSkill, SkillIdentity
from .rendering import (
    ActiveSkillsParseError,
    is_active_skills_message,
    is_skill_content_message,
    parse_active_skills_block,
    parse_active_skills_turn,
    render_active_skills_block,
    render_skill_content,
)

ACTIVE_SKILLS_SNAPSHOT_VERSION = 1

# Whitespace-delimited slash tokens anywhere in the user's text, e.g. `/plan-first`.
# Same grammar as PuPu's command tokenizer (`/[a-zA-Z0-9_-]+`) so legacy
# underscore spellings can still resolve through aliases.
_USER_TOKEN_RE = re.compile(r"^/([A-Za-z0-9_-]+)$")

_ACTIVATION_STATUSES = ("activated", "already_active", "superseded")


class SkillActivationStateError(RuntimeError):
    """The persisted `<active_skills>` block cannot be trusted (unsupported version or corrupt)."""


@dataclass
class ActivationQueue:
    """Hands tool-driven activations from the `skill` tool to the activation harness.

    The tool runs inside tool execution and cannot mutate run state itself; it
    pushes the loaded snapshot here and the harness drains the queue in the
    same iteration (`after_tool_batch`) so the block is committed with the
    tool result.
    """

    _pending: list[tuple[LoadedSkill, str]] = field(default_factory=list)
    # identity key -> revision of the snapshot as last projected by the harness;
    # lets the tool report already_active / superseded truthfully.
    known: dict[str, str] = field(default_factory=dict)

    def push(self, loaded: LoadedSkill, *, activation: str = "tool") -> str:
        """Queue an activation and return the status the projection will settle to."""

        self._pending.append((loaded, activation))
        current = self.known.get(loaded.summary.identity.key)
        if current is None:
            return "activated"
        if current == loaded.revision:
            return "already_active"
        return "superseded"

    def remember(self, active: "ActiveSkillSet") -> None:
        self.known = {entry.identity.key: entry.revision for entry in active.entries}

    def drain(self) -> tuple[tuple[LoadedSkill, str], ...]:
        pending = tuple(self._pending)
        self._pending.clear()
        return pending

    def __len__(self) -> int:
        return len(self._pending)


class ActiveSkillSet:
    """Ordered set of active skills keyed by identity; one revision per identity."""

    def __init__(
        self, entries: Iterable[ActiveSkill] = (), *, processed_turn: str | None = None
    ) -> None:
        self._entries: dict[str, ActiveSkill] = {}
        for entry in entries:
            self._entries[entry.identity.key] = entry
        # "<ordinal>:<hash>" of the last real user turn whose /name tokens were
        # resolved (persisted in the block header, see rendering).
        self.processed_turn: str | None = processed_turn

    @classmethod
    def from_block(cls, content: str) -> "ActiveSkillSet":
        try:
            entries = parse_active_skills_block(content)
        except ActiveSkillsParseError as exc:
            raise SkillActivationStateError(str(exc)) from exc
        return cls(entries, processed_turn=parse_active_skills_turn(content))

    @classmethod
    def from_messages(
        cls, messages: Sequence[Mapping[str, Any]]
    ) -> tuple["ActiveSkillSet", list[int]]:
        """Rebuild the set from the persisted block(s); return the block indexes."""

        indexes = [
            index for index, message in enumerate(messages) if is_active_skills_message(message)
        ]
        if not indexes:
            return cls(), []
        # The block is the durable authority. Only the first block is trusted;
        # duplicates are collapsed on the next render.
        return cls.from_block(str(messages[indexes[0]]["content"])), indexes

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    @property
    def entries(self) -> tuple[ActiveSkill, ...]:
        return tuple(self._entries.values())

    def get(self, identity: SkillIdentity) -> ActiveSkill | None:
        return self._entries.get(identity.key)

    def apply(self, loaded: LoadedSkill, *, activation: str) -> str:
        """Activate `loaded`; returns `activated`, `already_active` or `superseded`."""

        identity = loaded.summary.identity
        candidate = ActiveSkill(
            identity=identity,
            revision=loaded.revision,
            activation=activation,
            body=loaded.body,
            base_dir=loaded.summary.base_dir,
            tools=tuple(loaded.tools),
        )
        # Reject bodies that would break the strict block grammar before they
        # can be persisted.
        render_skill_content(candidate)
        current = self._entries.get(identity.key)
        if current is None:
            self._entries[identity.key] = candidate
            return "activated"
        if current.revision == loaded.revision:
            # Same content: keep the entry but refresh its provenance so the
            # block records the turn/call that most recently asked for it.
            if current.activation != activation:
                self._entries[identity.key] = candidate
            return "already_active"
        self._entries[identity.key] = candidate
        return "superseded"

    def render(self) -> str:
        return render_active_skills_block(self.entries, processed_turn=self.processed_turn)

    def to_state(self) -> dict[str, Any]:
        return {
            "version": ACTIVE_SKILLS_SNAPSHOT_VERSION,
            "active": {
                entry.identity.key: {
                    "name": entry.identity.name,
                    "source": entry.identity.source,
                    "source_id": entry.identity.source_id,
                    "revision": entry.revision,
                    "activation": entry.activation,
                    "base_dir": str(entry.base_dir) if entry.base_dir is not None else "",
                    "tools": list(entry.tools),
                    "body_bytes": len(entry.body.encode("utf-8")),
                }
                for entry in self.entries
            },
        }


def _is_real_user_message(message: Mapping[str, Any]) -> bool:
    return (
        message.get("role") == "user"
        and not is_skill_content_message(message)
        and _message_text(message) is not None
    )


def user_turn_identity(
    text: str,
    *,
    transcript: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Durable identity of the latest real user turn: ``"<ordinal>:<text hash>"``.

    The ordinal counts real user messages in the durable transcript (the
    conversation the kernel persists and replays), falling back to the
    working messages when the transcript carries none. Two identical
    submissions therefore get different ordinals, while replaying the same
    turn (tool loop, retry, resume, cold restart) yields the same identity.
    """

    source = transcript if any(_is_real_user_message(m) for m in transcript) else messages
    ordinal = sum(1 for m in source if isinstance(m, Mapping) and _is_real_user_message(m))
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{ordinal}:{digest}"


def is_new_user_turn(current: str, processed: str | None) -> bool:
    """Whether ``current`` is a turn not yet processed relative to ``processed``.

    A turn is new when its ordinal is higher than the recorded one, or when its
    text differs (guards against a transcript compaction lowering the ordinal
    while a genuinely new message arrived). Same ordinal and same text is a
    replay and must never re-resolve against the live registry.
    """

    if processed is None:
        return True
    if current == processed:
        return False
    cur_ordinal, _, cur_hash = current.partition(":")
    old_ordinal, _, old_hash = processed.partition(":")
    try:
        if int(cur_ordinal) > int(old_ordinal):
            return True
    except ValueError:
        return True
    return cur_hash != old_hash


def user_turn_activation_label(turn_identity: str) -> str:
    """Provenance label stored on entries activated from ``turn_identity``."""

    return f"user:{turn_identity}"


def find_user_invocation_tokens(text: str) -> list[str]:
    """Return `/name` tokens (without the slash) in textual order, deduplicated, case-preserved."""

    seen: set[str] = set()
    tokens: list[str] = []
    for word in text.split():
        match = _USER_TOKEN_RE.match(word)
        if match is None:
            continue
        token = match.group(1)
        folded = token.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        tokens.append(token)
    return tokens


def _message_text(message: Mapping[str, Any]) -> str | None:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (bytes, bytearray)):
        parts: list[str] = []
        saw_text = False
        for block in content:
            if not isinstance(block, Mapping):
                return None
            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
                saw_text = True
            elif block_type in {"tool_result", "function_response"}:
                return None
        return "\n".join(parts) if saw_text else None
    return None


def latest_real_user_text(messages: Sequence[Mapping[str, Any]]) -> tuple[int, str] | None:
    """Index and text of the latest user-authored message (never a synthetic skill message)."""

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        if is_skill_content_message(message):
            continue
        text = _message_text(message)
        if text is None:
            continue
        return index, text
    return None


__all__ = [
    "ACTIVE_SKILLS_SNAPSHOT_VERSION",
    "ActivationQueue",
    "ActiveSkillSet",
    "SkillActivationStateError",
    "find_user_invocation_tokens",
    "is_new_user_turn",
    "latest_real_user_text",
    "user_turn_activation_label",
    "user_turn_identity",
]
