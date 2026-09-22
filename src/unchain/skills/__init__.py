"""Agent skills: SKILL.md discovery, catalog, activation tool and /name invocation.

Versioned contract surface (advertised as runtime protocol ``skills`` 1.0):

- ``SKILLS_CONTRACT_VERSION`` — the descriptor / inventory / resolution shape.
- ``ACTIVE_SKILLS_SNAPSHOT_VERSION`` — the durable ``<active_skills>`` block format.
"""

from __future__ import annotations

from .activation import (
    ACTIVE_SKILLS_SNAPSHOT_VERSION,
    ActivationQueue,
    ActiveSkillSet,
    SkillActivationStateError,
    find_user_invocation_tokens,
    latest_real_user_text,
)
from .frontmatter import ParsedSkillFile, SkillParseError, coerce_bool, parse_skill_file
from .harness import SkillActivationHarness, SkillCatalogHarness
from .models import (
    ActiveSkill,
    LoadedSkill,
    SkillDiagnostic,
    SkillIdentity,
    SkillInventory,
    SkillSummary,
    compute_inventory_revision,
    compute_skill_revision,
    is_valid_skill_name,
)
from .registry import (
    TOOLKIT_SKILL_RANK,
    SkillRegistry,
    SkillRoot,
    SkillsConfig,
    resolve_project_root,
)
from .rendering import (
    ActiveSkillsParseError,
    parse_active_skills_block,
    render_active_skills_block,
    render_skill_catalog,
    render_skill_content,
    render_skill_loaded_envelope,
)
from .tools import build_skill_tool

SKILLS_PROTOCOL_ID = "skills"
SKILLS_CONTRACT_VERSION = 1

__all__ = [
    "ACTIVE_SKILLS_SNAPSHOT_VERSION",
    "SKILLS_CONTRACT_VERSION",
    "SKILLS_PROTOCOL_ID",
    "TOOLKIT_SKILL_RANK",
    "ActivationQueue",
    "ActiveSkill",
    "ActiveSkillSet",
    "ActiveSkillsParseError",
    "LoadedSkill",
    "ParsedSkillFile",
    "SkillActivationHarness",
    "SkillActivationStateError",
    "SkillCatalogHarness",
    "SkillDiagnostic",
    "SkillIdentity",
    "SkillInventory",
    "SkillParseError",
    "SkillRegistry",
    "SkillRoot",
    "SkillSummary",
    "SkillsConfig",
    "build_skill_tool",
    "coerce_bool",
    "compute_inventory_revision",
    "compute_skill_revision",
    "find_user_invocation_tokens",
    "is_valid_skill_name",
    "latest_real_user_text",
    "parse_active_skills_block",
    "parse_skill_file",
    "render_active_skills_block",
    "render_skill_catalog",
    "render_skill_content",
    "render_skill_loaded_envelope",
    "resolve_project_root",
]
