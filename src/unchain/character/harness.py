from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

from ..kernel.delta import HarnessDelta, InsertMessagesOp
from ..kernel.harness import BaseRuntimeHarness, HarnessContext
from .spec import CharacterSpec
from .state import CharacterState


_COMPONENT_KEY = "character"


@dataclass
class CharacterNarrativeHarness(BaseRuntimeHarness):
    """Harness that manages character narrative state across the agent run lifecycle.

    Phases:
    - bootstrap: load state from memory, initialize if first run, check time-delayed events
    - before_model: inject character context (identity + traits + schedule + active narrative)
    - before_commit: evaluate outcomes, advance narrative, persist state
    """
    spec: CharacterSpec = field(default=None, repr=False)
    name: str = "character_narrative"
    phases: tuple[str, ...] = ("bootstrap", "before_model", "before_commit")
    order: int = 15

    @property
    def created_by(self) -> str:
        return f"character.{self.name}"

    def applies(self, context: HarnessContext) -> bool:
        return context.phase in self.phases

    def build_delta(self, context: HarnessContext) -> HarnessDelta | None:
        if context.phase == "bootstrap":
            return self._on_bootstrap(context)
        if context.phase == "before_model":
            return self._on_before_model(context)
        if context.phase == "before_commit":
            return self._on_before_commit(context)
        return None

    def _on_bootstrap(self, context: HarnessContext) -> HarnessDelta | None:
        bucket = context.state.component_bucket(_COMPONENT_KEY)
        raw_state = bucket.get("state")
        now = time.time()

        if isinstance(raw_state, dict) and raw_state:
            char_state = CharacterState.from_dict(raw_state)
        else:
            char_state = CharacterState.initial(self.spec)

        # Check for newly eligible time-delayed events
        char_state.check_time_delayed_events(self.spec, now)

        # Mark pending events as in_progress if eligible
        for event in char_state.active_events(self.spec, now):
            bm = char_state.bookmarks.get(event.id)
            if bm is not None and bm.status == "pending":
                bm.status = "in_progress"
                if bm.reached_at is None:
                    bm.reached_at = now

        bucket["state"] = char_state.to_dict()

        return HarnessDelta(
            created_by=self.created_by,
            state_updates={
                "metadata": {
                    "character_state": char_state.to_dict(),
                },
            },
            trace={
                "character_activated_at": char_state.activated_at,
                "pending_events": [
                    eid for eid, bm in char_state.bookmarks.items()
                    if bm.status in ("pending", "in_progress")
                ],
            },
        )

    def _on_before_model(self, context: HarnessContext) -> HarnessDelta | None:
        bucket = context.state.component_bucket(_COMPONENT_KEY)
        raw_state = bucket.get("state")
        if not isinstance(raw_state, dict):
            return None

        char_state = CharacterState.from_dict(raw_state)
        now = time.time()

        # Build character system message
        parts: list[str] = []

        # Identity
        if self.spec.identity:
            parts.append(f"[CHARACTER]\n{self.spec.identity}")

        # Current traits
        if char_state.traits:
            trait_lines = "\n".join(f"- {k}: {v}" for k, v in char_state.traits.items())
            parts.append(f"[CURRENT TRAITS]\n{trait_lines}")

        # Schedule
        if char_state.schedule:
            schedule_lines = "\n".join(
                f"- {b.get('label', b.get('id', '?'))}: {b.get('time_range', '')} — {b.get('behavior', '')}"
                for b in char_state.schedule
                if b.get("active", True)
            )
            if schedule_lines:
                parts.append(f"[CURRENT SCHEDULE]\n{schedule_lines}")

        # Active narrative events
        active = char_state.active_events(self.spec, now)
        if active:
            narrative_lines = "\n\n".join(
                f"**{event.name}**\n{event.context}" for event in active
            )
            parts.append(f"[ACTIVE NARRATIVE]\n{narrative_lines}")

        # Bookmarked in-progress events
        in_progress = [
            (eid, bm) for eid, bm in char_state.bookmarks.items()
            if bm.status == "in_progress" and bm.bookmark
        ]
        if in_progress:
            bookmark_lines = "\n".join(
                f"- {eid}: {bm.bookmark}" for eid, bm in in_progress
            )
            parts.append(f"[NARRATIVE BOOKMARKS]\n{bookmark_lines}")

        if not parts:
            return None

        system_content = "\n\n".join(parts)
        character_message = {"role": "system", "content": system_content}

        # Insert as the first system message
        return HarnessDelta(
            created_by=self.created_by,
            ops=(InsertMessagesOp(index=0, messages=[character_message]),),
            trace={
                "active_event_count": len(active),
                "injected_system_chars": len(system_content),
            },
        )

    def _on_before_commit(self, context: HarnessContext) -> HarnessDelta | None:
        bucket = context.state.component_bucket(_COMPONENT_KEY)
        raw_state = bucket.get("state")
        if not isinstance(raw_state, dict):
            return None

        char_state = CharacterState.from_dict(raw_state)
        now = time.time()
        events_completed: list[str] = []

        # Evaluate outcomes for all in_progress events
        for event_id, bm in list(char_state.bookmarks.items()):
            if bm.status != "in_progress":
                continue
            event = self.spec.narrative.get(event_id)
            if event is None:
                continue
            outcome_id = char_state.resolve_outcome(event)
            if outcome_id is not None:
                char_state.apply_outcome(event, outcome_id, now)
                events_completed.append(f"{event_id}:{outcome_id}")

        # Check for newly unlocked events after outcome application
        char_state.check_time_delayed_events(self.spec, now)

        # Persist updated state
        bucket["state"] = char_state.to_dict()

        return HarnessDelta(
            created_by=self.created_by,
            state_updates={
                "metadata": {
                    "character_state": char_state.to_dict(),
                },
            },
            trace={
                "events_completed": events_completed,
                "trait_snapshot": copy.deepcopy(char_state.traits),
            },
        )
