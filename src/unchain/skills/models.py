"""Canonical data records for agent skills: identity, revision, and inventory.

These frozen dataclasses are the shared vocabulary used by the skills
registry, activation harness, and rendering helpers. They carry no I/O and no
parsing logic (see ``skills/frontmatter.py`` and ``skills/registry.py`` for
that); this module only defines the shapes and the small pure functions
(name/description validation, revision hashing) that every other skills
module depends on.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Any

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
SKILL_NAME_MAX_LENGTH = 64
SKILL_DESCRIPTION_MAX_LENGTH = 1024

SKILL_DIAGNOSTIC_KINDS = frozenset(
    {
        "invalid",
        "shadowed",
        "alias_shadowed",
        "alias_ambiguous",
        "reserved",
        "inaccessible",
        "body_delimiter",
        "user_invocation_denied",
        "unknown_skill",
        "model_invocation_denied",
    }
)


def is_valid_skill_name(name: object) -> bool:
    """Return True when ``name`` is a valid skill name.

    Valid names are non-empty strings of at most ``SKILL_NAME_MAX_LENGTH``
    characters matching ``SKILL_NAME_RE`` (lowercase alphanumerics with
    single hyphen separators, no leading/trailing/doubled hyphens).
    """
    if not isinstance(name, str):
        return False
    if not name or len(name) > SKILL_NAME_MAX_LENGTH:
        return False
    return bool(SKILL_NAME_RE.match(name))


def description_error(text: object) -> str | None:
    """Return None when ``text`` is a valid skill description, else an error message."""
    if not isinstance(text, str):
        return "description must be a string"
    if not text.strip():
        return "description must not be empty"
    if len(text) > SKILL_DESCRIPTION_MAX_LENGTH:
        return f"description must be at most {SKILL_DESCRIPTION_MAX_LENGTH} characters"
    return None


@dataclass(frozen=True)
class SkillIdentity:
    """Identifies one skill across sources: filesystem SKILL.md, toolkit, etc."""

    source: str
    source_id: str
    name: str

    @property
    def key(self) -> str:
        return f"{self.source}:{self.source_id}:{self.name}"


def compute_skill_revision(
    *,
    body: str,
    tools: Sequence[str],
    model_invocable: bool,
    user_invocable: bool,
) -> str:
    """Compute the content-addressed revision for a skill's activatable contract.

    The revision changes only when the activated behavior changes: the
    instruction body, the declared tool list, or either invocation policy
    flag. It is intentionally independent of description, name, source, or
    filesystem path.
    """
    payload = {
        "body": body,
        "model_invocable": model_invocable,
        "tools": list(tools),
        "user_invocable": user_invocable,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class SkillSummary:
    """Catalog-facing view of a discovered skill (no body)."""

    identity: SkillIdentity
    description: str
    rank: int
    path: Path | None
    base_dir: Path | None
    model_invocable: bool = True
    user_invocable: bool = True
    aliases: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return self.identity.name

    @property
    def source(self) -> str:
        return self.identity.source

    @property
    def source_id(self) -> str:
        return self.identity.source_id


@dataclass(frozen=True)
class LoadedSkill:
    """A freshly re-read skill, body included, ready for activation."""

    summary: SkillSummary
    body: str
    revision: str
    tools: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillDiagnostic:
    """A non-fatal note about a skill discovery or invocation problem."""

    kind: str
    name: str
    source: str
    source_id: str
    message: str

    def __post_init__(self) -> None:
        if self.kind not in SKILL_DIAGNOSTIC_KINDS:
            raise ValueError(f"unknown skill diagnostic kind: {self.kind!r}")


@dataclass(frozen=True)
class SkillInventory:
    """The full result of a registry scan: winning skills, diagnostics, revision."""

    skills: tuple[SkillSummary, ...]
    diagnostics: tuple[SkillDiagnostic, ...]
    revision: str


def compute_inventory_revision(
    skills_with_revisions: Iterable[tuple[str, str]],
    *,
    alias_map: Mapping[str, str] | None = None,
    reserved_commands: Iterable[str] = (),
) -> str:
    """Order-independent revision of the *effective* inventory.

    Binds not only each winner's identity + content revision but also the
    effective alias -> identity resolution map and the reserved-command set,
    so any change that alters what a `/name` token resolves to (an alias moving
    from one skill to another, a reserved name appearing) yields a new revision.
    A host that compares this value before dispatch therefore cannot execute a
    different skill than the one its menu showed.
    """
    pairs = sorted(skills_with_revisions, key=lambda pair: pair[0])
    canonical = json.dumps(
        {
            "skills": [list(pair) for pair in pairs],
            "aliases": sorted((str(alias), str(key)) for alias, key in (alias_map or {}).items()),
            "reserved": sorted(str(item) for item in reserved_commands),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class ActiveSkill:
    """One activated skill entry as it lives in the durable <active_skills> block."""

    identity: SkillIdentity
    revision: str
    activation: str
    body: str
    base_dir: Path | None
    tools: tuple[str, ...] = ()


__all__ = [
    "SKILL_NAME_RE",
    "SKILL_NAME_MAX_LENGTH",
    "SKILL_DESCRIPTION_MAX_LENGTH",
    "SKILL_DIAGNOSTIC_KINDS",
    "is_valid_skill_name",
    "description_error",
    "SkillIdentity",
    "compute_skill_revision",
    "SkillSummary",
    "LoadedSkill",
    "SkillDiagnostic",
    "SkillInventory",
    "compute_inventory_revision",
    "ActiveSkill",
]
