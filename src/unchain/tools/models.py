from __future__ import annotations

import inspect
import json
import re
import types
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any, Callable, Union, get_args, get_origin, get_type_hints

_UNION_TYPE = getattr(types, "UnionType", None)


def _escape_control_chars_inside_json_strings(raw: str) -> str:
    """Escape raw control chars that appear inside JSON string literals."""
    chars: list[str] = []
    in_string = False
    escaped = False

    for ch in raw:
        if in_string:
            if escaped:
                chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                chars.append(ch)
                escaped = True
                continue
            if ch == "\"":
                chars.append(ch)
                in_string = False
                continue
            if ch == "\n":
                chars.append("\\n")
                continue
            if ch == "\r":
                chars.append("\\r")
                continue
            if ch == "\t":
                chars.append("\\t")
                continue
            chars.append(ch)
            continue

        chars.append(ch)
        if ch == "\"":
            in_string = True
            escaped = False

    return "".join(chars)


def _annotation_to_json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        return "array"
    if origin in (dict,):
        return "object"
    union_origins = (Union,) if _UNION_TYPE is None else (Union, _UNION_TYPE)
    if origin in union_origins:
        union_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(union_args) == 1:
            return _annotation_to_json_type(union_args[0])
        return "string"
    if origin is not None:
        origin_args = get_args(annotation)
        if origin_args:
            return _annotation_to_json_type(origin)

    if annotation in (int,):
        return "integer"
    if annotation in (float,):
        return "number"
    if annotation in (bool,):
        return "boolean"
    if annotation in (list, tuple, set):
        return "array"
    if annotation in (dict,):
        return "object"
    return "string"


def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        origin_args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        item_schema: dict[str, Any] = {"type": "string"}
        if origin_args:
            item_schema = _annotation_to_json_schema(origin_args[0])
        if "type" not in item_schema:
            item_schema = {"type": "string"}
        return {"type": "array", "items": item_schema}

    return {"type": _annotation_to_json_type(annotation)}


def _parse_docstring(func: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", {}

    lines = doc.splitlines()
    summary = lines[0].strip()
    parameter_descriptions: dict[str, str] = {}

    for line in lines:
        match = re.match(r"^\s*:param\s+([A-Za-z_]\w*)\s*:\s*(.+)$", line)
        if match:
            parameter_descriptions[match.group(1)] = match.group(2).strip()

    in_args_block = False
    current_parameter: str | None = None

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped in {"Args:", "Arguments:", "Parameters:"}:
            in_args_block = True
            current_parameter = None
            continue

        if not in_args_block:
            continue

        if not stripped:
            current_parameter = None
            continue

        if re.match(r"^[A-Z][A-Za-z ]+:$", stripped):
            in_args_block = False
            current_parameter = None
            continue

        parameter_match = re.match(
            r"^([A-Za-z_]\w*)(?:\s*\([^)]+\))?\s*:\s*(.+)$",
            stripped,
        )
        if parameter_match:
            current_parameter = parameter_match.group(1)
            parameter_descriptions[current_parameter] = parameter_match.group(2).strip()
            continue

        if current_parameter and line.startswith((" ", "\t")):
            parameter_descriptions[current_parameter] = (
                f"{parameter_descriptions[current_parameter]} {stripped}".strip()
            )
        else:
            current_parameter = None

    return summary, parameter_descriptions


@dataclass
class ToolParameter:
    name: str
    description: str
    type_: str
    required: bool = False
    pattern: str | None = None
    items: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        json_parameter: dict[str, Any] = {
            "type": self.type_,
            "description": self.description,
        }
        if self.pattern is not None:
            json_parameter["pattern"] = self.pattern
        if self.items is not None:
            json_parameter["items"] = self.items
        elif self.type_ == "array":
            json_parameter["items"] = {"type": "string"}
        return json_parameter


@dataclass
class ToolHistoryOptimizationContext:
    tool_name: str
    call_id: str
    kind: str
    provider: str
    session_id: str
    latest_messages: list[dict[str, Any]]
    max_chars: int
    preview_chars: int
    include_hash: bool = True


@dataclass(frozen=True)
class ToolExecutionContext:
    session_id: str
    run_id: str
    provider: str
    model: str
    iteration: int
    memory_namespace: str = ""
    session_store: Any = None
    tool_runtime_config: dict[str, Any] = field(default_factory=dict)
    tool_name: str = ""
    call_id: str = ""
    turn_id: str = ""
    workspace_changes: Any = None
    # Internal-only owner state for nested provider-call accounting. It is
    # never projected into tool arguments or model-visible tool results.
    run_state: Any = None


@dataclass(frozen=True)
class ToolConfirmationPolicy:
    requires_confirmation: bool = True
    description: str = ""
    render_component: dict[str, Any] | None = None
    interact_type: str = "confirmation"
    interact_config: dict[str, Any] | list[Any] | None = None

    @classmethod
    def from_raw(
        cls,
        raw: bool | dict[str, Any] | "ToolConfirmationPolicy" | None,
    ) -> "ToolConfirmationPolicy":
        if isinstance(raw, ToolConfirmationPolicy):
            return raw
        if isinstance(raw, bool):
            return cls(requires_confirmation=raw)
        if isinstance(raw, dict):
            render_component = raw.get("render_component")
            interact_type_raw = raw.get("interact_type", "confirmation")
            interact_type = (
                interact_type_raw
                if isinstance(interact_type_raw, str) and interact_type_raw
                else "confirmation"
            )
            interact_config_raw = raw.get("interact_config")
            interact_config = (
                interact_config_raw
                if isinstance(interact_config_raw, (dict, list))
                else None
            )
            return cls(
                requires_confirmation=bool(raw.get("requires_confirmation", True)),
                description=str(raw.get("description") or ""),
                render_component=render_component if isinstance(render_component, dict) else None,
                interact_type=interact_type,
                interact_config=interact_config,
            )
        return cls()


@dataclass(frozen=True)
class ToolPromptSpec:
    purpose: str = ""
    when_to_use: tuple[str, ...] = ()
    when_not_to_use: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    advanced_tips: tuple[str, ...] = ()

    @staticmethod
    def _normalize_lines(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            text = value.strip()
            return (text,) if text else ()
        if isinstance(value, (list, tuple)):
            normalized: list[str] = []
            for item in value:
                text = str(item or "").strip()
                if text:
                    normalized.append(text)
            return tuple(normalized)
        raise TypeError("tool prompt spec list fields must be strings or string lists")

    @classmethod
    def from_raw(cls, raw: "ToolPromptSpec | dict[str, Any] | None") -> "ToolPromptSpec | None":
        if raw is None:
            return None
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, dict):
            return cls(
                purpose=str(raw.get("purpose") or "").strip(),
                when_to_use=cls._normalize_lines(raw.get("when_to_use")),
                when_not_to_use=cls._normalize_lines(raw.get("when_not_to_use")),
                examples=cls._normalize_lines(raw.get("examples")),
                advanced_tips=cls._normalize_lines(raw.get("advanced_tips")),
            )
        raise TypeError("tool prompt spec must be ToolPromptSpec, dict, or None")


@dataclass
class NormalizedToolHistoryRecord:
    tool_name: str
    call_id: str
    kind: str
    payload: Any
    provider: str
    message_index: int
    location_type: str
    payload_format: str
    block_index: int | None = None
    part_index: int | None = None
    field_name: str | None = None


HistoryPayloadOptimizer = Callable[[Any, ToolHistoryOptimizationContext], Any]


@dataclass
class ToolConfirmationRequest:
    tool_name: str
    call_id: str
    arguments: dict[str, Any]
    description: str = ""
    interact_type: str = "confirmation"
    interact_config: dict[str, Any] | list[Any] | None = None
    render_component: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "tool_confirmation_request",
            "tool_name": self.tool_name,
            "call_id": self.call_id,
            "arguments": self.arguments,
            "description": self.description,
        }
        if self.interact_type != "confirmation":
            payload["interact_type"] = self.interact_type
        if self.interact_config is not None:
            payload["interact_config"] = self.interact_config
        if self.render_component is not None:
            payload["render_component"] = self.render_component
        return payload


@dataclass
class ToolConfirmationResponse:
    approved: bool = True
    modified_arguments: dict[str, Any] | None = None
    reason: str = ""

    @classmethod
    def from_raw(
        cls,
        raw: bool | dict[str, Any] | "ToolConfirmationResponse",
    ) -> "ToolConfirmationResponse":
        if isinstance(raw, ToolConfirmationResponse):
            return raw
        if isinstance(raw, bool):
            return cls(approved=raw)
        if isinstance(raw, dict):
            return cls(
                approved=raw.get("approved", True),
                modified_arguments=raw.get("modified_arguments"),
                reason=raw.get("reason", ""),
            )
        return cls(approved=bool(raw))


@dataclass(frozen=True)
class SkillDescriptor:
    """One skill embedded in a toolkit (`[[skills]]` manifest table or `Toolkit(skills=...)`).

    The first five fields keep the historical positional shape; policy and
    source fields are keyword-only so existing callers stay valid. `source` /
    `source_id` identify where the skill came from (a toolkit id for manifest
    skills); the skills registry never merges descriptors by name alone.
    """

    name: str
    description: str
    body: str
    tools: tuple[str, ...] = ()
    base_dir: Path | None = None
    _: KW_ONLY
    model_invocable: bool = True
    user_invocable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()
    source: str = "toolkit"
    source_id: str = ""

    def to_summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "body": self.body,
            "tools": list(self.tools),
            "base_dir": str(self.base_dir) if self.base_dir is not None else "",
            "model_invocable": self.model_invocable,
            "user_invocable": self.user_invocable,
            "metadata": dict(self.metadata),
            "aliases": list(self.aliases),
            "source": self.source,
            "source_id": self.source_id,
        }


__all__ = [
    "HistoryPayloadOptimizer",
    "ToolConfirmationPolicy",
    "ToolExecutionContext",
    "NormalizedToolHistoryRecord",
    "SkillDescriptor",
    "ToolConfirmationRequest",
    "ToolConfirmationResponse",
    "ToolHistoryOptimizationContext",
    "ToolParameter",
    "ToolPromptSpec",
    "_annotation_to_json_schema",
    "_escape_control_chars_inside_json_strings",
    "_parse_docstring",
]
