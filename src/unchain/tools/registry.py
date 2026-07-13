from __future__ import annotations

import importlib
import inspect
import logging
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

try:  # pragma: no cover - Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - Python 3.9/3.10 fallback
    import tomli as tomllib

try:  # pragma: no cover - Python 3.10+
    from importlib.metadata import entry_points
except ImportError:  # pragma: no cover - Python 3.9 fallback
    from importlib_metadata import entry_points

from .toolkit import Toolkit as RuntimeToolkit

_ICON_SUFFIXES = {".svg", ".png"}
_ENTRY_POINT_GROUPS = ("unchain.toolkits", "unchain.toolkits")
_ARTIFACT_FALLBACK_RENDERERS = {"markdown", "text", "table", "kv", "log", "link", "json"}
_PUBLIC_BUILTIN_TOOLKIT_DIRS = {"agent_reach", "core", "plan"}
_LOGGER = logging.getLogger(__name__)


def _builtin_artifact_icon(name: str) -> "IconDescriptor":
    return IconDescriptor.from_builtin(name, "", "")


_BUILTIN_ARTIFACT_KIND_DEFINITIONS = (
    {
        "kind": "file_diff",
        "display_name": "Files changed",
        "description": "Immutable snapshots of file changes produced by tools.",
        "icon": "file_edit",
        "fallback_renderer": "json",
    },
    {
        "kind": "workspace_change_set",
        "display_name": "Workspace changes",
        "description": "Run-scoped net workspace changes with safe restore metadata.",
        "icon": "file_edit",
        "fallback_renderer": "json",
    },
    {
        "kind": "plan",
        "display_name": "Plan",
        "description": "Plan snapshots produced by PlanToolkit.",
        "icon": "check_list",
        "fallback_renderer": "markdown",
    },
    {
        "kind": "markdown",
        "display_name": "Markdown",
        "description": "Markdown artifact snapshots.",
        "icon": "markdown",
        "fallback_renderer": "markdown",
    },
    {
        "kind": "table",
        "display_name": "Table",
        "description": "Tabular artifact snapshots.",
        "icon": "data",
        "fallback_renderer": "table",
    },
    {
        "kind": "kv",
        "display_name": "Metadata",
        "description": "Key-value artifact snapshots.",
        "icon": "information",
        "fallback_renderer": "kv",
    },
    {
        "kind": "log",
        "display_name": "Log",
        "description": "Log artifact snapshots.",
        "icon": "terminal",
        "fallback_renderer": "log",
    },
    {
        "kind": "link",
        "display_name": "Link",
        "description": "Link artifact snapshots.",
        "icon": "link",
        "fallback_renderer": "link",
    },
)
_BUILTIN_ARTIFACT_KIND_NAMES = {
    item["kind"] for item in _BUILTIN_ARTIFACT_KIND_DEFINITIONS
}

_SKILL_PHASES = frozenset({"composer", "streaming", "always"})
_SKILL_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _read_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"expected boolean, got {type(value).__name__}")


def _require_str(section: dict[str, Any], key: str, manifest_path: Path) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{manifest_path}: missing or invalid string for '{key}'")
    return value.strip()


def _optional_str(section: dict[str, Any], key: str) -> str | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid string for '{key}'")
    return value.strip()


def _optional_int(section: dict[str, Any], key: str, *, default: int = 0) -> int:
    value = section.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"invalid integer for '{key}'")
    return value


def _string_list(section: dict[str, Any], key: str) -> tuple[str, ...]:
    value = section.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"invalid string list for '{key}'")
    return tuple(item.strip() for item in value)


def _resolve_asset(root: Path, relative_path: str, *, label: str, required_suffixes: set[str] | None = None) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay within toolkit directory: {candidate}") from exc
    if not candidate.is_file():
        raise ValueError(f"{label} not found: {candidate}")
    if required_suffixes is not None and candidate.suffix.lower() not in required_suffixes:
        raise ValueError(f"{label} must use one of {sorted(required_suffixes)}: {candidate}")
    return candidate


def _looks_like_icon_asset(value: str) -> bool:
    return Path(value).suffix.lower() in _ICON_SUFFIXES


def _looks_like_emoji_icon(value: str) -> bool:
    return any(unicodedata.category(char) == "So" for char in value)


def _require_builtin_icon_field(
    section: dict[str, Any],
    key: str,
    manifest_path: Path,
    *,
    icon_name: str,
) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{manifest_path}: builtin toolkit icon '{icon_name}' requires non-empty '{key}'"
        )
    return value.strip()


def _import_object(import_path: str, import_roots: Sequence[Path] = ()) -> Any:
    if ":" not in import_path:
        raise ValueError(f"invalid import path '{import_path}'; expected 'module:attribute'")

    module_name, attribute_path = import_path.split(":", 1)
    if not module_name or not attribute_path:
        raise ValueError(f"invalid import path '{import_path}'")

    with _temporary_sys_path(import_roots):
        module = importlib.import_module(module_name)

    obj: Any = module
    for part in attribute_path.split("."):
        if not hasattr(obj, part):
            raise ValueError(f"import path '{import_path}' is missing attribute '{part}'")
        obj = getattr(obj, part)
    return obj


@contextmanager
def _temporary_sys_path(paths: Sequence[Path]) -> Iterator[None]:
    additions: list[str] = []
    normalized = [str(path) for path in paths if str(path)]
    for path in reversed(normalized):
        if path in sys.path:
            continue
        sys.path.insert(0, path)
        additions.append(path)
    try:
        yield
    finally:
        for path in additions:
            while path in sys.path:
                sys.path.remove(path)


def _normalize_roots(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        for candidate in (path, path.parent):
            if candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
    return tuple(normalized)


def _entry_points_for_group(group: str) -> list[Any]:
    discovered = entry_points()
    if hasattr(discovered, "select"):
        selected = list(discovered.select(group=group))
    elif isinstance(discovered, dict):  # pragma: no cover - legacy stdlib shape
        selected = list(discovered.get(group, []))
    else:  # pragma: no cover - older importlib_metadata shape
        selected = [entry_point for entry_point in discovered if getattr(entry_point, "group", None) == group]
    return sorted(selected, key=lambda item: item.name)


def _find_manifest_near_factory(factory: Any) -> Path:
    module = inspect.getmodule(factory)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise ValueError(f"cannot determine module path for factory '{factory}'")

    start = Path(module_file).resolve().parent
    for candidate_dir in (start, *start.parents):
        manifest_path = candidate_dir / "toolkit.toml"
        if manifest_path.is_file():
            return manifest_path
    raise ValueError(f"could not find toolkit.toml near factory '{factory}'")


def _matches_factory_identity(left: Any, right: Any) -> bool:
    if left is right:
        return True
    return (
        getattr(left, "__module__", None) == getattr(right, "__module__", None)
        and getattr(left, "__qualname__", getattr(left, "__name__", None))
        == getattr(right, "__qualname__", getattr(right, "__name__", None))
    )


@dataclass
class ToolRegistryConfig:
    local_roots: tuple[str, ...] = ()
    enabled_plugins: tuple[str, ...] = ()
    include_builtin: bool = True
    validate: bool = True

    def __init__(
        self,
        local_roots: Sequence[str | Path] | None = None,
        enabled_plugins: Sequence[str] | None = None,
        include_builtin: bool = True,
        validate: bool = True,
    ):
        self.local_roots = tuple(str(Path(path).resolve()) for path in (local_roots or ()))
        self.enabled_plugins = tuple(str(plugin).strip() for plugin in (enabled_plugins or ()) if str(plugin).strip())
        self.include_builtin = bool(include_builtin)
        self.validate = bool(validate)

    @classmethod
    def coerce(cls, value: ToolRegistryConfig | dict[str, Any] | None) -> ToolRegistryConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("config must be ToolRegistryConfig, dict, or None")


@dataclass
class ToolDescriptor:
    name: str
    title: str
    description: str
    icon_path: Path | None
    icon: "IconDescriptor"
    hidden: bool = False
    requires_confirmation: bool = False
    observe: bool = False

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "icon_path": str(self.icon_path) if self.icon_path else "",
            "icon": self.icon.to_summary(),
            "hidden": self.hidden,
            "requires_confirmation": self.requires_confirmation,
            "observe": self.observe,
        }


@dataclass
class SkillDescriptor:
    name: str
    title: str
    description: str
    body: str
    tools: tuple[str, ...] = ()
    phase: str = "composer"

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "body": self.body,
            "tools": list(self.tools),
            "phase": self.phase,
        }


@dataclass
class IconDescriptor:
    type: str
    path: Path | None = None
    name: str | None = None
    color: str | None = None
    background_color: str | None = None
    emoji: str | None = None

    @classmethod
    def from_file(cls, path: Path) -> "IconDescriptor":
        return cls(type="file", path=path)

    @classmethod
    def from_emoji(cls, emoji: str) -> "IconDescriptor":
        return cls(type="emoji", emoji=emoji)

    @classmethod
    def from_builtin(
        cls,
        name: str,
        color: str,
        background_color: str,
    ) -> "IconDescriptor":
        return cls(
            type="builtin",
            name=name,
            color=color,
            background_color=background_color,
        )

    def to_summary(self) -> dict[str, Any]:
        if self.type == "file":
            return {
                "type": "file",
                "path": str(self.path) if self.path else "",
            }
        if self.type == "emoji":
            return {
                "type": "emoji",
                "emoji": self.emoji or "",
            }
        return {
            "type": "builtin",
            "name": self.name or "",
            "color": self.color or "",
            "background_color": self.background_color or "",
        }


@dataclass
class ArtifactKindDescriptor:
    kind: str
    display_name: str
    description: str
    icon: IconDescriptor
    fallback_renderer: str
    toolkit_id: str

    def to_summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "display_name": self.display_name,
            "description": self.description,
            "icon": self.icon.to_summary(),
            "fallback_renderer": self.fallback_renderer,
            "toolkit_id": self.toolkit_id,
        }


@dataclass
class ToolkitDescriptor:
    id: str
    name: str
    description: str
    factory: str
    version: str | None
    tags: tuple[str, ...]
    manifest_path: Path
    root_path: Path
    readme_path: Path
    icon_path: Path | None
    icon: IconDescriptor
    source: str
    display_category: str | None = None
    display_order: int = 0
    hidden: bool = False
    compat_python: str | None = None
    compat_legacy: str | None = None
    tools: dict[str, ToolDescriptor] = field(default_factory=dict)
    artifact_kinds: dict[str, ArtifactKindDescriptor] = field(default_factory=dict)
    skills: dict[str, SkillDescriptor] = field(default_factory=dict)
    import_roots: tuple[Path, ...] = field(default_factory=tuple, repr=False)

    def sorted_tools(self) -> list[ToolDescriptor]:
        return sorted(
            self.tools.values(),
            key=lambda item: (item.title.casefold(), item.name.casefold()),
        )

    def sorted_artifact_kinds(self) -> list[ArtifactKindDescriptor]:
        return sorted(
            self.artifact_kinds.values(),
            key=lambda item: item.kind.casefold(),
        )

    def sorted_skills(self) -> list[SkillDescriptor]:
        return sorted(
            self.skills.values(),
            key=lambda item: (item.title.casefold(), item.name.casefold()),
        )

    def to_summary(self, *, include_tools: bool = True) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "factory": self.factory,
            "version": self.version,
            "tags": list(self.tags),
            "source": self.source,
            "manifest_path": str(self.manifest_path),
            "root_path": str(self.root_path),
            "readme_path": str(self.readme_path),
            "icon_path": str(self.icon_path) if self.icon_path else "",
            "icon": self.icon.to_summary(),
            "tool_count": len(self.tools),
            "display": {
                "category": self.display_category,
                "order": self.display_order,
                "hidden": self.hidden,
            },
            "compat": {
                "python": self.compat_python,
                "legacy": self.compat_legacy,
            },
            "artifact_kinds": [
                artifact_kind.to_summary()
                for artifact_kind in self.sorted_artifact_kinds()
            ],
            "skills": [skill.to_summary() for skill in self.sorted_skills()],
        }
        if include_tools:
            payload["tools"] = [tool.to_summary() for tool in self.sorted_tools()]
        return payload

    def to_metadata(self, *, include_tools: bool = True) -> dict[str, Any]:
        payload = self.to_summary(include_tools=include_tools)
        payload["readme_markdown"] = _read_markdown(self.readme_path)
        return payload


def _resolve_toolkit_icon(
    root: Path,
    toolkit_section: dict[str, Any],
    manifest_path: Path,
) -> tuple[Path | None, IconDescriptor]:
    icon_value = _require_str(toolkit_section, "icon", manifest_path)
    if _looks_like_icon_asset(icon_value):
        icon_path = _resolve_asset(
            root,
            icon_value,
            label=f"{manifest_path}: toolkit icon",
            required_suffixes=_ICON_SUFFIXES,
        )
        return icon_path, IconDescriptor.from_file(icon_path)
    if _looks_like_emoji_icon(icon_value):
        return None, IconDescriptor.from_emoji(icon_value)

    color = _require_builtin_icon_field(
        toolkit_section,
        "color",
        manifest_path,
        icon_name=icon_value,
    )
    background_color = _require_builtin_icon_field(
        toolkit_section,
        "backgroundcolor",
        manifest_path,
        icon_name=icon_value,
    )
    return None, IconDescriptor.from_builtin(icon_value, color, background_color)


def _builtin_artifact_kinds() -> dict[str, ArtifactKindDescriptor]:
    return {
        str(item["kind"]): ArtifactKindDescriptor(
            kind=str(item["kind"]),
            display_name=str(item["display_name"]),
            description=str(item["description"]),
            icon=_builtin_artifact_icon(str(item["icon"])),
            fallback_renderer=str(item["fallback_renderer"]),
            toolkit_id="builtin",
        )
        for item in _BUILTIN_ARTIFACT_KIND_DEFINITIONS
    }


def _resolve_artifact_kind_icon(
    root: Path,
    artifact_kind: dict[str, Any],
    manifest_path: Path,
    kind: str,
) -> IconDescriptor:
    icon_value = _require_str(artifact_kind, "icon", manifest_path)
    if _looks_like_icon_asset(icon_value):
        icon_path = _resolve_asset(
            root,
            icon_value,
            label=f"{manifest_path}: artifact icon for '{kind}'",
            required_suffixes=_ICON_SUFFIXES,
        )
        return IconDescriptor.from_file(icon_path)

    color = _optional_str(artifact_kind, "color") or ""
    background_color = (
        _optional_str(artifact_kind, "backgroundcolor")
        or _optional_str(artifact_kind, "background_color")
        or ""
    )
    return IconDescriptor.from_builtin(icon_value, color, background_color)


def _parse_artifact_kinds(
    root: Path,
    manifest_path: Path,
    toolkit_id: str,
    artifact_kinds_section: list[Any],
) -> dict[str, ArtifactKindDescriptor]:
    artifact_kinds: dict[str, ArtifactKindDescriptor] = {}
    for artifact_kind in artifact_kinds_section:
        if not isinstance(artifact_kind, dict):
            raise ValueError(f"{manifest_path}: invalid [[artifact_kinds]] entry")

        kind = _require_str(artifact_kind, "kind", manifest_path)
        if kind in _BUILTIN_ARTIFACT_KIND_NAMES:
            _LOGGER.warning(
                "%s: toolkit '%s' cannot override builtin artifact kind '%s'; ignoring manifest entry",
                manifest_path,
                toolkit_id,
                kind,
            )
            continue
        if kind in artifact_kinds:
            raise ValueError(f"{manifest_path}: duplicate artifact kind '{kind}'")

        fallback_renderer = _require_str(artifact_kind, "fallback_renderer", manifest_path)
        if fallback_renderer not in _ARTIFACT_FALLBACK_RENDERERS:
            raise ValueError(
                f"{manifest_path}: invalid artifact fallback_renderer "
                f"'{fallback_renderer}' for '{kind}'"
            )

        artifact_kinds[kind] = ArtifactKindDescriptor(
            kind=kind,
            display_name=_require_str(artifact_kind, "display_name", manifest_path),
            description=_optional_str(artifact_kind, "description") or "",
            icon=_resolve_artifact_kind_icon(root, artifact_kind, manifest_path, kind),
            fallback_renderer=fallback_renderer,
            toolkit_id=toolkit_id,
        )
    return artifact_kinds


class ToolkitRegistry:
    """Discover toolkit metadata from builtin, local, and plugin sources."""

    def __init__(self, config: ToolRegistryConfig | dict[str, Any] | None = None):
        self.config = ToolRegistryConfig.coerce(config)
        self._toolkits: dict[str, ToolkitDescriptor] = {}
        self._discover()

    @property
    def toolkits(self) -> dict[str, ToolkitDescriptor]:
        return dict(self._toolkits)

    def list_toolkits(self, *, include_tools: bool = True) -> list[dict[str, Any]]:
        return [
            descriptor.to_summary(include_tools=include_tools)
            for descriptor in self._sorted_toolkits()
        ]

    def list_artifact_kinds(self) -> list[dict[str, Any]]:
        artifact_kinds = _builtin_artifact_kinds()
        custom: dict[str, ArtifactKindDescriptor] = {}

        for descriptor in self._sorted_toolkits():
            for artifact_kind in descriptor.sorted_artifact_kinds():
                kind = artifact_kind.kind
                if kind in artifact_kinds:
                    _LOGGER.warning(
                        "%s: toolkit '%s' cannot override builtin artifact kind '%s'; ignoring manifest entry",
                        descriptor.manifest_path,
                        descriptor.id,
                        kind,
                    )
                    continue

                existing = custom.get(kind)
                if existing is None:
                    custom[kind] = artifact_kind
                    continue

                winner = min(
                    (existing, artifact_kind),
                    key=lambda item: item.toolkit_id.casefold(),
                )
                loser = artifact_kind if winner is existing else existing
                custom[kind] = winner
                _LOGGER.warning(
                    "duplicate artifact kind '%s' from toolkit '%s' ignored; toolkit '%s' wins",
                    kind,
                    loser.toolkit_id,
                    winner.toolkit_id,
                )

        return [
            artifact_kind.to_summary()
            for artifact_kind in (
                list(artifact_kinds.values())
                + sorted(custom.values(), key=lambda item: item.kind.casefold())
            )
        ]

    def get(self, toolkit_id: str) -> ToolkitDescriptor | None:
        return self._toolkits.get(toolkit_id)

    def require(self, toolkit_id: str) -> ToolkitDescriptor:
        descriptor = self.get(toolkit_id)
        if descriptor is None:
            raise KeyError(f"unknown toolkit id: {toolkit_id}")
        return descriptor

    def get_toolkit_metadata(self, toolkit_id: str, tool_name: str | None = None) -> dict[str, Any]:
        descriptor = self.require(toolkit_id)
        if tool_name is None:
            return descriptor.to_metadata(include_tools=True)

        tool_descriptor = descriptor.tools.get(tool_name)
        if tool_descriptor is None:
            raise KeyError(f"unknown tool '{tool_name}' for toolkit '{toolkit_id}'")

        return {
            "toolkit": descriptor.to_metadata(include_tools=False),
            "tool": tool_descriptor.to_summary(),
        }

    def instantiate_toolkit(self, toolkit_id: str) -> RuntimeToolkit:
        descriptor = self.require(toolkit_id)
        factory = _import_object(descriptor.factory, descriptor.import_roots)
        try:
            runtime_toolkit = factory()
        except TypeError as exc:
            raise ValueError(
                f"{descriptor.manifest_path}: toolkit factory '{descriptor.factory}' "
                "must be callable without required arguments"
            ) from exc

        if not isinstance(runtime_toolkit, RuntimeToolkit):
            raise ValueError(
                f"{descriptor.manifest_path}: toolkit factory '{descriptor.factory}' "
                "did not return a unchain.tools.Toolkit"
            )
        self._apply_runtime_tool_metadata(runtime_toolkit, descriptor)
        return runtime_toolkit

    def _apply_runtime_tool_metadata(
        self,
        runtime_toolkit: RuntimeToolkit,
        descriptor: ToolkitDescriptor,
    ) -> None:
        for tool_name, tool_descriptor in descriptor.tools.items():
            runtime_tool = runtime_toolkit.tools.get(tool_name)
            if runtime_tool is None:
                continue
            runtime_tool.icon_path = str(tool_descriptor.icon_path) if tool_descriptor.icon_path else ""
            runtime_tool.icon = tool_descriptor.icon.to_summary()

    def _sorted_toolkits(self) -> list[ToolkitDescriptor]:
        return sorted(
            self._toolkits.values(),
            key=lambda item: (item.display_order, item.name.casefold(), item.id.casefold()),
        )

    def _discover(self) -> None:
        if self.config.include_builtin:
            self._discover_builtin_toolkits()
        self._discover_local_toolkits()
        self._discover_plugin_toolkits()

    def _discover_builtin_toolkits(self) -> None:
        builtin_root = Path(__file__).resolve().parents[1] / "toolkits" / "builtin"
        for manifest_path in sorted(builtin_root.rglob("toolkit.toml")):
            if manifest_path.parent.name not in _PUBLIC_BUILTIN_TOOLKIT_DIRS:
                continue
            self._register_descriptor(
                self._load_descriptor(
                    manifest_path=manifest_path,
                    source="builtin",
                    import_roots=(),
                )
            )

    def _discover_local_toolkits(self) -> None:
        for raw_root in self.config.local_roots:
            root = Path(raw_root).resolve()
            manifests = self._find_local_manifests(root)
            import_roots = _normalize_roots((root,))
            for manifest_path in manifests:
                self._register_descriptor(
                    self._load_descriptor(
                        manifest_path=manifest_path,
                        source="local",
                        import_roots=import_roots,
                    )
                )

    def _discover_plugin_toolkits(self) -> None:
        enabled = set(self.config.enabled_plugins)
        if not enabled:
            return

        seen: set[str] = set()
        for entry_point_group in _ENTRY_POINT_GROUPS:
            for entry_point in _entry_points_for_group(entry_point_group):
                if entry_point.name not in enabled or entry_point.name in seen:
                    continue

                factory = entry_point.load()
                manifest_path = _find_manifest_near_factory(factory)
                descriptor = self._load_descriptor(
                    manifest_path=manifest_path,
                    source="plugin",
                    import_roots=(),
                    entry_point_name=entry_point.name,
                    entry_point_factory=factory,
                )
                self._register_descriptor(descriptor)
                seen.add(entry_point.name)

        missing = sorted(enabled - seen)
        if missing:
            raise ValueError(
                "enabled plugin(s) not found in any of "
                f"{_ENTRY_POINT_GROUPS!r}: {', '.join(missing)}"
            )

    def _find_local_manifests(self, root: Path) -> list[Path]:
        if not root.exists():
            raise ValueError(f"local root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"local root is not a directory: {root}")

        direct_manifest = root / "toolkit.toml"
        if direct_manifest.is_file():
            return [direct_manifest]

        manifests = sorted(root.rglob("toolkit.toml"))
        if manifests:
            return manifests
        raise ValueError(f"no toolkit.toml found under local root: {root}")

    def _register_descriptor(self, descriptor: ToolkitDescriptor) -> None:
        existing = self._toolkits.get(descriptor.id)
        if existing is not None:
            raise ValueError(
                f"duplicate toolkit id '{descriptor.id}' in {descriptor.manifest_path} "
                f"(already registered from {existing.manifest_path})"
            )
        self._toolkits[descriptor.id] = descriptor

    def _load_descriptor(
        self,
        *,
        manifest_path: Path,
        source: str,
        import_roots: Sequence[Path],
        entry_point_name: str | None = None,
        entry_point_factory: Any | None = None,
    ) -> ToolkitDescriptor:
        manifest_path = manifest_path.resolve()
        data = _read_toml(manifest_path)
        root_path = manifest_path.parent.resolve()

        toolkit_section = data.get("toolkit")
        if not isinstance(toolkit_section, dict):
            raise ValueError(f"{manifest_path}: missing [toolkit] section")

        display_section = data.get("display", {})
        if display_section is None:
            display_section = {}
        if not isinstance(display_section, dict):
            raise ValueError(f"{manifest_path}: [display] must be a table")

        compat_section = data.get("compat", {})
        if compat_section is None:
            compat_section = {}
        if not isinstance(compat_section, dict):
            raise ValueError(f"{manifest_path}: [compat] must be a table")

        tools_section = data.get("tools", [])
        if tools_section is None:
            tools_section = []
        if not isinstance(tools_section, list):
            raise ValueError(f"{manifest_path}: [[tools]] must be an array of tables")

        artifact_kinds_section = data.get("artifact_kinds", [])
        if artifact_kinds_section is None:
            artifact_kinds_section = []
        if not isinstance(artifact_kinds_section, list):
            raise ValueError(f"{manifest_path}: [[artifact_kinds]] must be an array of tables")

        skills_section = data.get("skills", [])
        if skills_section is None:
            skills_section = []
        if not isinstance(skills_section, list):
            raise ValueError(f"{manifest_path}: [[skills]] must be an array of tables")

        toolkit_id = _require_str(toolkit_section, "id", manifest_path)
        toolkit_name = _require_str(toolkit_section, "name", manifest_path)
        toolkit_description = _require_str(toolkit_section, "description", manifest_path)
        factory = _require_str(toolkit_section, "factory", manifest_path)
        version = _optional_str(toolkit_section, "version")
        tags = _string_list(toolkit_section, "tags")
        readme_path = _resolve_asset(
            root_path,
            _require_str(toolkit_section, "readme", manifest_path),
            label=f"{manifest_path}: toolkit readme",
        )
        icon_path, icon = _resolve_toolkit_icon(root_path, toolkit_section, manifest_path)

        tools: dict[str, ToolDescriptor] = {}
        for tool_item in tools_section:
            if not isinstance(tool_item, dict):
                raise ValueError(f"{manifest_path}: invalid [[tools]] entry")
            tool_name = _require_str(tool_item, "name", manifest_path)
            if tool_name in tools:
                raise ValueError(f"{manifest_path}: duplicate tool '{tool_name}'")
            if "readme" in tool_item:
                raise ValueError(
                    f"{manifest_path}: tool-level readme is not supported for '{tool_name}'; "
                    "document tools in the toolkit README instead"
                )
            tool_icon_rel = _optional_str(tool_item, "icon")
            if tool_icon_rel is not None:
                tool_icon_path = _resolve_asset(
                    root_path,
                    tool_icon_rel,
                    label=f"{manifest_path}: tool icon for '{tool_name}'",
                    required_suffixes=_ICON_SUFFIXES,
                )
                tool_icon = IconDescriptor.from_file(tool_icon_path)
            else:
                tool_icon_path = icon_path
                tool_icon = icon
            tools[tool_name] = ToolDescriptor(
                name=tool_name,
                title=_optional_str(tool_item, "title") or tool_name,
                description=_require_str(tool_item, "description", manifest_path),
                icon_path=tool_icon_path,
                icon=tool_icon,
                hidden=_coerce_bool(tool_item.get("hidden"), default=False),
                requires_confirmation=_coerce_bool(tool_item.get("requires_confirmation"), default=False),
                observe=_coerce_bool(tool_item.get("observe"), default=False),
            )

        skills: dict[str, SkillDescriptor] = {}
        for skill_item in skills_section:
            if not isinstance(skill_item, dict):
                raise ValueError(f"{manifest_path}: invalid [[skills]] entry")
            skill_name = _require_str(skill_item, "name", manifest_path)
            if not _SKILL_NAME_RE.match(skill_name):
                raise ValueError(
                    f"{manifest_path}: skill name '{skill_name}' must match [a-zA-Z0-9_-]+"
                )
            if skill_name in skills:
                raise ValueError(f"{manifest_path}: duplicate skill '{skill_name}'")
            skill_phase = _optional_str(skill_item, "phase") or "composer"
            if skill_phase not in _SKILL_PHASES:
                raise ValueError(
                    f"{manifest_path}: skill '{skill_name}' has invalid phase '{skill_phase}'"
                )
            skill_tools = _string_list(skill_item, "tools")
            for tool_ref in skill_tools:
                if tool_ref not in tools:
                    raise ValueError(
                        f"{manifest_path}: skill '{skill_name}' references unknown tool '{tool_ref}'"
                    )
            skills[skill_name] = SkillDescriptor(
                name=skill_name,
                title=_optional_str(skill_item, "title") or skill_name,
                description=_require_str(skill_item, "description", manifest_path),
                body=_require_str(skill_item, "body", manifest_path),
                tools=tuple(skill_tools),
                phase=skill_phase,
            )

        artifact_kinds = _parse_artifact_kinds(
            root_path,
            manifest_path,
            toolkit_id,
            artifact_kinds_section,
        )

        descriptor = ToolkitDescriptor(
            id=toolkit_id,
            name=toolkit_name,
            description=toolkit_description,
            factory=factory,
            version=version,
            tags=tags,
            manifest_path=manifest_path,
            root_path=root_path,
            readme_path=readme_path,
            icon_path=icon_path,
            icon=icon,
            source=source,
            display_category=_optional_str(display_section, "category"),
            display_order=_optional_int(display_section, "order", default=0),
            hidden=_coerce_bool(display_section.get("hidden"), default=False),
            compat_python=_optional_str(compat_section, "python"),
            compat_legacy=_optional_str(compat_section, "legacy"),
            tools=tools,
            artifact_kinds=artifact_kinds,
            skills=skills,
            import_roots=tuple(import_roots),
        )

        if self.config.validate:
            self._validate_descriptor(
                descriptor,
                entry_point_name=entry_point_name,
                entry_point_factory=entry_point_factory,
            )

        return descriptor

    def _validate_descriptor(
        self,
        descriptor: ToolkitDescriptor,
        *,
        entry_point_name: str | None = None,
        entry_point_factory: Any | None = None,
    ) -> None:
        if entry_point_name is not None and descriptor.id != entry_point_name:
            raise ValueError(
                f"{descriptor.manifest_path}: toolkit.id '{descriptor.id}' must match entry point name '{entry_point_name}'"
            )

        factory = _import_object(descriptor.factory, descriptor.import_roots)
        if entry_point_factory is not None and not _matches_factory_identity(factory, entry_point_factory):
            raise ValueError(
                f"{descriptor.manifest_path}: manifest factory '{descriptor.factory}' does not match plugin entry point '{entry_point_name}'"
            )

        try:
            runtime_toolkit = factory()
        except TypeError as exc:
            raise ValueError(
                f"{descriptor.manifest_path}: toolkit factory '{descriptor.factory}' must be callable without required arguments"
            ) from exc

        if not isinstance(runtime_toolkit, RuntimeToolkit):
            raise ValueError(
                f"{descriptor.manifest_path}: toolkit factory '{descriptor.factory}' did not return a unchain.tools.Toolkit"
            )

        actual_tools = runtime_toolkit.tools
        manifest_tool_names = set(descriptor.tools)
        actual_tool_names = set(actual_tools)

        missing_from_manifest = sorted(actual_tool_names - manifest_tool_names)
        if missing_from_manifest:
            raise ValueError(
                f"{descriptor.manifest_path}: runtime tools missing from manifest: {', '.join(missing_from_manifest)}"
            )

        missing_from_runtime = sorted(manifest_tool_names - actual_tool_names)
        if missing_from_runtime:
            raise ValueError(
                f"{descriptor.manifest_path}: manifest tools missing from runtime registration: {', '.join(missing_from_runtime)}"
            )

        for tool_name, tool_descriptor in descriptor.tools.items():
            runtime_tool = actual_tools[tool_name]
            if runtime_tool.observe != tool_descriptor.observe:
                raise ValueError(
                    f"{descriptor.manifest_path}: tool '{tool_name}' observe={tool_descriptor.observe} "
                    f"does not match runtime observe={runtime_tool.observe}"
                )
            if runtime_tool.requires_confirmation != tool_descriptor.requires_confirmation:
                raise ValueError(
                    f"{descriptor.manifest_path}: tool '{tool_name}' requires_confirmation="
                    f"{tool_descriptor.requires_confirmation} does not match runtime "
                    f"requires_confirmation={runtime_tool.requires_confirmation}"
                )


def list_toolkits(
    config: ToolRegistryConfig | dict[str, Any] | None = None,
    *,
    include_tools: bool = True,
) -> list[dict[str, Any]]:
    """Return JSON-safe toolkit summaries for UI and developer tooling."""
    return ToolkitRegistry(config).list_toolkits(include_tools=include_tools)


def list_artifact_kinds(
    config: ToolRegistryConfig | dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return JSON-safe artifact kind metadata for UI catalog consumers."""
    return ToolkitRegistry(config).list_artifact_kinds()


def get_toolkit_metadata(
    toolkit_id: str,
    tool_name: str | None = None,
    *,
    config: ToolRegistryConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return full metadata for one toolkit or one tool within a toolkit."""
    return ToolkitRegistry(config).get_toolkit_metadata(toolkit_id, tool_name=tool_name)


__all__ = [
    "ArtifactKindDescriptor",
    "ToolDescriptor",
    "ToolRegistryConfig",
    "ToolkitDescriptor",
    "ToolkitRegistry",
    "get_toolkit_metadata",
    "list_artifact_kinds",
    "list_toolkits",
]
