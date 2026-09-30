"""Skill discovery and resolution registry (ticket #327 decision D3).

``SkillRegistry`` scans a fixed, ranked set of filesystem roots for
``SKILL.md`` (or ``<name>.md``) files and merges them with skills embedded in
the runtime toolkit (``SkillDescriptor`` entries), resolving name conflicts
deterministically (lowest rank, then lowest identity key). It never caches
across calls to ``list()``/``get()``/``resolve()`` -- the durable copy of an
activated skill is the rendered ``<active_skills>`` snapshot (see
``skills/activation.py``), not anything held in this registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..tools.models import SkillDescriptor
from .frontmatter import SkillParseError, coerce_bool, parse_skill_file
from .models import (
    LoadedSkill,
    SkillDiagnostic,
    SkillIdentity,
    SkillInventory,
    SkillSummary,
    compute_inventory_revision,
    compute_skill_revision,
    description_error,
    is_valid_skill_name,
)

if TYPE_CHECKING:
    from ..tools.toolkit import Toolkit

__all__ = [
    "SKILL_FILE_NAME",
    "TOOLKIT_SKILL_RANK",
    "SkillsConfig",
    "SkillRoot",
    "resolve_project_root",
    "SkillRegistry",
]

SKILL_FILE_NAME = "SKILL.md"
TOOLKIT_SKILL_RANK = 600

_RESERVED_FRONTMATTER_KEYS = frozenset(
    {"name", "description", "disable-model-invocation", "user-invocable"}
)


@dataclass(frozen=True)
class SkillsConfig:
    """Configuration for skill discovery and resolution (ticket #327 D3)."""

    project_root: str | Path | None = None
    include_project_dirs: bool = True
    extra_dirs: tuple[str | Path, ...] = ()
    include_user_dirs: bool = True
    catalog_description_max_length: int = 500
    tool_name: str = "skill"
    reserved_commands: tuple[str, ...] = ()
    home: str | Path | None = None
    # Programmatic sources without a toolkit (e.g. a host's installed skill
    # packs). Same rank as toolkit-embedded skills (600); each descriptor keeps
    # its own `source` / `source_id` identity.
    extra_skills: tuple[SkillDescriptor, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(item, SkillDescriptor) for item in self.extra_skills):
            raise TypeError("extra_skills must contain SkillDescriptor instances")
        object.__setattr__(self, "extra_skills", tuple(self.extra_skills))
        if self.catalog_description_max_length < 3:
            raise ValueError("catalog_description_max_length must be >= 3")
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool_name must be a non-empty string")

        object.__setattr__(
            self, "extra_dirs", tuple(Path(entry) for entry in self.extra_dirs)
        )

        normalized_reserved: list[str] = []
        for raw in self.reserved_commands:
            text = str(raw).strip()
            if text.startswith("/"):
                text = text.lstrip("/")
            text = text.lower()
            if text:
                normalized_reserved.append(text)
        object.__setattr__(self, "reserved_commands", tuple(normalized_reserved))

        if self.project_root is not None:
            object.__setattr__(self, "project_root", Path(self.project_root))
        if self.home is not None:
            object.__setattr__(self, "home", Path(self.home))

    @classmethod
    def coerce(cls, value: "SkillsConfig | dict | None") -> "SkillsConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError(f"cannot coerce {value!r} into a SkillsConfig")


@dataclass(frozen=True)
class SkillRoot:
    """One ranked filesystem root to scan for skills."""

    source: str
    rank: int
    path: Path


def resolve_project_root(start: Path) -> Path:
    """Return the nearest ancestor of ``start`` (including ``start``) that has
    a ``.git`` entry (file or directory), else ``start.resolve()``."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


# ---------------------------------------------------------------------------
# Internal scan bookkeeping (not part of the public contract)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    identity: SkillIdentity
    rank: int
    path: Path | None
    base_dir: Path | None
    description: str
    model_invocable: bool
    user_invocable: bool
    aliases: tuple[str, ...]
    body: str
    tools: tuple[str, ...]
    metadata: dict[str, Any]
    revision: str


@dataclass(frozen=True)
class _WinnerRecord:
    summary: SkillSummary
    body: str
    revision: str
    tools: tuple[str, ...]
    metadata: dict[str, Any]


def _scan_root_entry(
    root: SkillRoot,
    entry: Path,
    seen_real_paths: set[Path],
    diagnostics: list[SkillDiagnostic],
    candidates: list[_Candidate],
) -> None:
    if entry.name.startswith("."):
        return

    base_dir: Path | None
    if entry.is_dir():
        skill_file = entry / SKILL_FILE_NAME
        if not skill_file.is_file():
            return
        candidate_name = entry.name
        skill_path = skill_file
        base_dir = entry
    elif entry.is_file():
        if entry.name == SKILL_FILE_NAME or entry.suffix != ".md":
            return
        candidate_name = entry.stem
        skill_path = entry
        base_dir = root.path
    else:
        return

    try:
        resolved = skill_path.resolve()
    except OSError as exc:
        diagnostics.append(
            SkillDiagnostic(
                kind="inaccessible",
                name=candidate_name,
                source=root.source,
                source_id=str(skill_path),
                message=str(exc),
            )
        )
        return

    if resolved in seen_real_paths:
        return
    seen_real_paths.add(resolved)

    try:
        text = resolved.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeDecodeError) as exc:
        diagnostics.append(
            SkillDiagnostic(
                kind="inaccessible",
                name=candidate_name,
                source=root.source,
                source_id=str(resolved),
                message=str(exc),
            )
        )
        return

    try:
        parsed = parse_skill_file(text)
    except SkillParseError as exc:
        diagnostics.append(
            SkillDiagnostic(
                kind="invalid",
                name=candidate_name,
                source=root.source,
                source_id=str(resolved),
                message=str(exc),
            )
        )
        return

    fields = parsed.fields
    field_name = fields.get("name")
    if not isinstance(field_name, str) or field_name != candidate_name:
        diagnostics.append(
            SkillDiagnostic(
                kind="invalid",
                name=candidate_name,
                source=root.source,
                source_id=str(resolved),
                message=(
                    f"skill name {field_name!r} does not match expected name "
                    f"{candidate_name!r}"
                ),
            )
        )
        return
    if not is_valid_skill_name(field_name):
        diagnostics.append(
            SkillDiagnostic(
                kind="invalid",
                name=candidate_name,
                source=root.source,
                source_id=str(resolved),
                message=f"invalid skill name: {field_name!r}",
            )
        )
        return

    description = fields.get("description")
    description_err = description_error(description)
    if description_err is not None:
        diagnostics.append(
            SkillDiagnostic(
                kind="invalid",
                name=candidate_name,
                source=root.source,
                source_id=str(resolved),
                message=description_err,
            )
        )
        return

    try:
        model_invocable = not coerce_bool(
            fields.get("disable-model-invocation"), default=False
        )
        user_invocable = coerce_bool(fields.get("user-invocable"), default=True)
    except SkillParseError as exc:
        diagnostics.append(
            SkillDiagnostic(
                kind="invalid",
                name=candidate_name,
                source=root.source,
                source_id=str(resolved),
                message=str(exc),
            )
        )
        return

    identity = SkillIdentity(source=root.source, source_id=str(resolved), name=candidate_name)
    metadata = {
        key: value for key, value in fields.items() if key not in _RESERVED_FRONTMATTER_KEYS
    }
    revision = compute_skill_revision(
        body=parsed.body,
        tools=(),
        model_invocable=model_invocable,
        user_invocable=user_invocable,
    )
    candidates.append(
        _Candidate(
            identity=identity,
            rank=root.rank,
            path=resolved,
            base_dir=base_dir,
            description=description,
            model_invocable=model_invocable,
            user_invocable=user_invocable,
            aliases=(),
            body=parsed.body,
            tools=(),
            metadata=metadata,
            revision=revision,
        )
    )


def _scan_toolkit_skills(
    runtime_toolkit: "Toolkit | None",
    diagnostics: list[SkillDiagnostic],
    candidates: list[_Candidate],
    extra_skills: tuple[SkillDescriptor, ...] = (),
) -> None:
    descriptors = tuple(getattr(runtime_toolkit, "skills", ()) or ()) + tuple(extra_skills)
    for descriptor in descriptors:
        name = descriptor.name
        source = descriptor.source or "toolkit"
        source_id = descriptor.source_id or ""
        if not is_valid_skill_name(name):
            diagnostics.append(
                SkillDiagnostic(
                    kind="invalid",
                    name=str(name),
                    source=source,
                    source_id=source_id,
                    message=f"invalid skill name: {name!r}",
                )
            )
            continue
        description_err = description_error(descriptor.description)
        if description_err is not None:
            diagnostics.append(
                SkillDiagnostic(
                    kind="invalid",
                    name=name,
                    source=source,
                    source_id=source_id,
                    message=description_err,
                )
            )
            continue

        identity = SkillIdentity(source=source, source_id=source_id, name=name)
        tools = tuple(descriptor.tools)
        rendered_body = descriptor.body.replace(
            "{tools}", ", ".join(f"`{tool}`" for tool in tools)
        )
        revision = compute_skill_revision(
            body=rendered_body,
            tools=tools,
            model_invocable=descriptor.model_invocable,
            user_invocable=descriptor.user_invocable,
        )
        candidates.append(
            _Candidate(
                identity=identity,
                rank=TOOLKIT_SKILL_RANK,
                path=None,
                base_dir=descriptor.base_dir,
                description=descriptor.description,
                model_invocable=descriptor.model_invocable,
                user_invocable=descriptor.user_invocable,
                aliases=tuple(descriptor.aliases),
                body=rendered_body,
                tools=tools,
                metadata=dict(descriptor.metadata),
                revision=revision,
            )
        )


def _select_winners(candidates: list[_Candidate]) -> dict[str, _Candidate]:
    groups: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.identity.name, []).append(candidate)
    winners: dict[str, _Candidate] = {}
    for name, group in groups.items():
        winners[name] = min(group, key=lambda c: (c.rank, c.identity.key))
    return winners


def _emit_shadowed_diagnostics(
    candidates: list[_Candidate],
    winners: dict[str, _Candidate],
    diagnostics: list[SkillDiagnostic],
) -> None:
    for candidate in candidates:
        winner = winners[candidate.identity.name]
        if candidate is winner:
            continue
        diagnostics.append(
            SkillDiagnostic(
                kind="shadowed",
                name=candidate.identity.name,
                source=candidate.identity.source,
                source_id=candidate.identity.source_id,
                message=(
                    f"skill '{candidate.identity.name}' is shadowed by a higher-priority "
                    f"skill from source '{winner.identity.source}'"
                ),
            )
        )


def _resolve_aliases(
    ordered_winners: list[_Candidate],
    winners: dict[str, _Candidate],
    diagnostics: list[SkillDiagnostic],
) -> dict[str, str]:
    winner_name_folds = {name.casefold() for name in winners}
    groups: dict[str, list[tuple[str, _Candidate]]] = {}
    for candidate in ordered_winners:
        for alias in candidate.aliases:
            groups.setdefault(alias.casefold(), []).append((alias, candidate))

    resolved: dict[str, str] = {}
    for folded, entries in groups.items():
        display_alias = entries[0][0]
        if folded in winner_name_folds:
            candidate = entries[0][1]
            diagnostics.append(
                SkillDiagnostic(
                    kind="alias_shadowed",
                    name=display_alias,
                    source=candidate.identity.source,
                    source_id=candidate.identity.source_id,
                    message=(
                        f"alias '{display_alias}' matches an existing skill name and was dropped"
                    ),
                )
            )
            continue
        distinct = {candidate.identity.key: candidate for _, candidate in entries}
        if len(distinct) > 1:
            first_candidate = entries[0][1]
            diagnostics.append(
                SkillDiagnostic(
                    kind="alias_ambiguous",
                    name=display_alias,
                    source=first_candidate.identity.source,
                    source_id=first_candidate.identity.source_id,
                    message=f"alias '{display_alias}' is claimed by multiple skills and was dropped",
                )
            )
            continue
        candidate = entries[0][1]
        resolved[folded] = candidate.identity.name
    return resolved


def _emit_reserved_diagnostics(
    ordered_winners: list[_Candidate],
    reserved_commands: tuple[str, ...],
    diagnostics: list[SkillDiagnostic],
) -> None:
    if not reserved_commands:
        return
    reserved = set(reserved_commands)
    for candidate in ordered_winners:
        if candidate.identity.name in reserved:
            diagnostics.append(
                SkillDiagnostic(
                    kind="reserved",
                    name=candidate.identity.name,
                    source=candidate.identity.source,
                    source_id=candidate.identity.source_id,
                    message=(
                        f"skill name '{candidate.identity.name}' collides with a reserved "
                        "command and will not resolve via /name"
                    ),
                )
            )


class SkillRegistry:
    """Discovers and resolves skills across filesystem roots and the runtime toolkit.

    Never caches across public calls: each of ``list()``, ``get()``, and
    ``resolve()`` re-scans the filesystem and re-reads ``runtime_toolkit.skills``
    (via ``getattr`` so skills appended to the toolkit after construction are
    still discovered).
    """

    def __init__(
        self,
        config: "SkillsConfig | dict | None",
        *,
        runtime_toolkit: "Toolkit | None" = None,
    ) -> None:
        self.config: SkillsConfig = SkillsConfig.coerce(config)
        self.runtime_toolkit = runtime_toolkit
        self._winner_cache: dict[str, _WinnerRecord] = {}
        self._alias_index: dict[str, str] = {}

    def roots(self) -> tuple[SkillRoot, ...]:
        result: list[SkillRoot] = []
        if self.config.include_project_dirs:
            project_root = self.config.project_root
            if project_root is None:
                project_root = resolve_project_root(Path.cwd())
            result.append(
                SkillRoot("project-unchain", 100, project_root / ".unchain" / "skills")
            )
            result.append(
                SkillRoot("project-agents", 200, project_root / ".agents" / "skills")
            )
        for extra in self.config.extra_dirs:
            result.append(SkillRoot("custom", 300, Path(extra)))
        if self.config.include_user_dirs:
            home = self.config.home if self.config.home is not None else Path.home()
            result.append(SkillRoot("user-unchain", 400, home / ".unchain" / "skills"))
            result.append(SkillRoot("user-agents", 500, home / ".agents" / "skills"))
        return tuple(result)

    def list(self) -> SkillInventory:
        diagnostics: list[SkillDiagnostic] = []
        candidates: list[_Candidate] = []
        seen_real_paths: set[Path] = set()

        for root in self.roots():
            if not root.path.is_dir():
                continue
            try:
                entries = sorted(root.path.iterdir(), key=lambda item: item.name)
            except OSError:
                continue
            for entry in entries:
                _scan_root_entry(root, entry, seen_real_paths, diagnostics, candidates)

        _scan_toolkit_skills(
            self.runtime_toolkit, diagnostics, candidates, self.config.extra_skills
        )

        winners = _select_winners(candidates)
        ordered_winners = [
            candidate for candidate in candidates if winners.get(candidate.identity.name) is candidate
        ]

        _emit_shadowed_diagnostics(candidates, winners, diagnostics)
        alias_index = _resolve_aliases(ordered_winners, winners, diagnostics)
        _emit_reserved_diagnostics(ordered_winners, self.config.reserved_commands, diagnostics)

        # Only aliases that actually resolve to this winner are published; a
        # dropped (shadowed) or ambiguous alias must never be advertised as a
        # command the runtime will then refuse or route elsewhere.
        effective_aliases: dict[str, list[str]] = {}
        for candidate in ordered_winners:
            for alias in candidate.aliases:
                if alias_index.get(alias.casefold()) == candidate.identity.name:
                    effective_aliases.setdefault(candidate.identity.name, []).append(alias)

        winner_cache: dict[str, _WinnerRecord] = {}
        summaries: list[SkillSummary] = []
        for candidate in ordered_winners:
            summary = SkillSummary(
                identity=candidate.identity,
                description=candidate.description,
                rank=candidate.rank,
                path=candidate.path,
                base_dir=candidate.base_dir,
                model_invocable=candidate.model_invocable,
                user_invocable=candidate.user_invocable,
                aliases=tuple(effective_aliases.get(candidate.identity.name, ())),
            )
            summaries.append(summary)
            winner_cache[candidate.identity.name] = _WinnerRecord(
                summary=summary,
                body=candidate.body,
                revision=candidate.revision,
                tools=candidate.tools,
                metadata=dict(candidate.metadata),
            )

        summaries.sort(key=lambda item: item.name)
        revision = compute_inventory_revision(
            ((item.identity.key, winner_cache[item.name].revision) for item in summaries),
            alias_map={
                alias: winners[name].identity.key for alias, name in alias_index.items()
            },
            reserved_commands=self.config.reserved_commands,
        )

        self._winner_cache = winner_cache
        self._alias_index = alias_index

        return SkillInventory(
            skills=tuple(summaries),
            diagnostics=tuple(diagnostics),
            revision=revision,
        )

    def get(self, name: str) -> LoadedSkill | None:
        self.list()
        record = self._winner_cache.get(name)
        if record is None:
            return None
        return LoadedSkill(
            summary=record.summary,
            body=record.body,
            revision=record.revision,
            tools=record.tools,
            metadata=dict(record.metadata),
        )

    def resolve(self, token: str) -> SkillSummary | None:
        inventory = self.list()
        stripped = token[1:] if token.startswith("/") else token
        folded = stripped.casefold()
        for summary in inventory.skills:
            if summary.name.casefold() == folded:
                return summary
        winner_name = self._alias_index.get(folded)
        if winner_name is None:
            return None
        record = self._winner_cache.get(winner_name)
        return record.summary if record is not None else None
