from __future__ import annotations

import hashlib
import json

from ..journal import AttemptRef, BoundExecutionJournal, SemanticEventDraft
from .activation import ActiveSkillSet, SkillActivationStateError

SNAPSHOT_EVENT = "skills.activation_snapshot"
SNAPSHOT_SCHEMA = "unchain.skills.activation_snapshot.v1"


class JournalSkillState:
    """Generation-scoped snapshots, independent of a trimmed model transcript."""

    def __init__(self, attempt: AttemptRef, journal: BoundExecutionJournal) -> None:
        if journal.execution_id != attempt.generation.execution_id:
            raise SkillActivationStateError("skill journal execution mismatch")
        self.attempt = attempt
        self.journal = journal
        self.events = tuple(
            event for event in journal.capture_snapshot().events
            if event.attempt.generation == attempt.generation
        )
        self.block = None
        for event in self.events:
            if event.event_type != SNAPSHOT_EVENT:
                continue
            payload = event.payload
            if (
                set(payload) != {"schema", "block"}
                or payload["schema"] != SNAPSHOT_SCHEMA
                or not isinstance(payload["block"], str)
                or not payload["block"]
            ):
                raise SkillActivationStateError("invalid journal skill snapshot")
            # Validate every recorded snapshot, including its version, before use.
            ActiveSkillSet.from_block(payload["block"])
            self.block = payload["block"]

    def turn(self, text: str) -> str | None:
        # Graph handoff inputs have their own receipts, but their synthetic
        # content does not equal the real user text. Bind to the original input.
        matches = [
            event for event in self.events
            if event.event_type == "message.user"
            and event.payload.get("message", {}).get("content") == text
        ]
        if not matches:
            return None
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return f"{matches[-1].store_seq}:{digest}"

    def save(self, block: str) -> None:
        if not block or block == self.block:
            return
        payload = {"schema": SNAPSHOT_SCHEMA, "block": block}
        digest = hashlib.sha256(json.dumps(
            {"attempt": self.attempt.to_dict(), "payload": payload},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")).hexdigest()
        self.journal.append(request=SemanticEventDraft(
            event_id=f"skills-{digest}", event_type=SNAPSHOT_EVENT,
            attempt=self.attempt, operation_id=f"skills-{digest}",
            payload=payload,
        ).to_append_request())
        self.block = block
