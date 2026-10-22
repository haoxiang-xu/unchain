from __future__ import annotations

import copy
from typing import Any, Callable, Sequence
from datetime import datetime

from .decision import (
    CharacterIdentitySpec,
    CharacterObligation,
    evaluate_character,
    decide_character_response,
)
from .naming import (
    make_character_self_namespace,
    make_character_relationship_namespace,
    make_character_direct_session_id,
)
from .instructions import build_character_instructions


def build_character_agent_config(
    *,
    character: CharacterIdentitySpec | dict[str, Any],
    thread_id: str,
    human_id: str = "local_user",
    profile_loader: Callable[[str], dict[str, Any]] | None = None,
    now: datetime | str | None = None,
    obligations: Sequence[CharacterObligation | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a config dict for running an unchain Agent with character context.

    This replaces the old CharacterAgent.build_config() method.
    Returns a dict with all the information needed to create and run an agent
    with character identity, schedule awareness, and decision context.
    """
    spec = CharacterIdentitySpec.coerce(character)
    session_id = make_character_direct_session_id(spec.id, thread_id)
    self_namespace = make_character_self_namespace(spec.id)
    relationship_namespace = make_character_relationship_namespace(spec.id, human_id)

    evaluation = evaluate_character(spec, now=now, obligations=obligations)
    decision = decide_character_response(spec, evaluation=evaluation)

    load_profile = profile_loader if callable(profile_loader) else (lambda _namespace: {})
    self_profile = load_profile(self_namespace) or {}
    relationship_profile = load_profile(relationship_namespace) or {}

    instructions = build_character_instructions(
        spec,
        evaluation=evaluation,
        decision=decision,
        self_namespace=self_namespace,
        relationship_namespace=relationship_namespace,
        self_profile=self_profile,
        relationship_profile=relationship_profile,
    )

    return {
        "character": spec.to_dict(),
        "session_id": session_id,
        "self_namespace": self_namespace,
        "relationship_namespace": relationship_namespace,
        "run_memory_namespace": relationship_namespace,
        "evaluation": evaluation.to_dict(),
        "decision": decision.to_dict(),
        "instructions": instructions,
        "self_profile": copy.deepcopy(self_profile),
        "relationship_profile": copy.deepcopy(relationship_profile),
    }
