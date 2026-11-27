"""Closed Subagent Pool configuration for Agent Skeletons."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .errors import SkeletonValidationError


_MODES = {"delegate", "handoff", "worker"}
_OUTPUT_MODES = {"summary", "last_message", "full_trace"}
_MEMORY_POLICIES = {"ephemeral", "scoped_persistent"}


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _closed(value: Any, path: tuple[str | int, ...], required: set[str]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("invalid_type", path, "value must be an object")
    missing = required - set(value)
    if missing:
        _fail("missing_field", path + (sorted(missing)[0],), "required field is missing")
    extra = set(value) - required
    if extra:
        _fail("unknown_field", path + (sorted(extra)[0],), "unknown field")
    return value


def _text(value: Any, path: tuple[str | int, ...], label: str) -> str:
    if type(value) is not str:
        _fail("invalid_type", path, f"{label} must be a string")
    if not value.strip():
        _fail("invalid_value", path, f"{label} must be non-empty")
    return value


def validate_subagent_pool_config(
    config: Any,
    path: tuple[str | int, ...],
    *,
    validate_embedded: Callable[[dict[str, Any], tuple[str | int, ...]], None],
) -> tuple[dict[str, Any], ...]:
    """Validate one closed pool and return detached entry snapshots."""
    root = _closed(config, path, {"subagents"})
    raw_entries = root["subagents"]
    if type(raw_entries) is not list:
        _fail("invalid_type", path + ("subagents",), "subagents must be an array")
    if not raw_entries:
        _fail("missing_value", path + ("subagents",), "subagent pool must contain an Agent")
    entries: list[dict[str, Any]] = []
    ids: set[str] = set()
    names: set[str] = set()
    required = {
        "id", "name", "description", "source", "allowed_modes", "output_mode",
        "memory_policy", "parallel_safe", "allowed_tools", "disabled_tools",
    }
    for index, raw in enumerate(raw_entries):
        entry_path = path + ("subagents", index)
        entry = _closed(raw, entry_path, required)
        entry_id = _text(entry["id"], entry_path + ("id",), "subagent id")
        name = _text(entry["name"], entry_path + ("name",), "subagent name")
        if entry_id in ids:
            _fail("duplicate_name", entry_path + ("id",), "subagent id is duplicated")
        if name in names:
            _fail("duplicate_name", entry_path + ("name",), "subagent name is duplicated")
        ids.add(entry_id)
        names.add(name)
        if type(entry["description"]) is not str:
            _fail("invalid_type", entry_path + ("description",), "description must be a string")
        modes = entry["allowed_modes"]
        if type(modes) is not list or not modes:
            _fail("invalid_type", entry_path + ("allowed_modes",), "allowed_modes must be a non-empty array")
        if any(type(mode) is not str or mode not in _MODES for mode in modes):
            _fail("unknown_variant", entry_path + ("allowed_modes",), "allowed_modes contains an unknown mode")
        if len(modes) != len(set(modes)):
            _fail("duplicate_name", entry_path + ("allowed_modes",), "allowed_modes contains duplicates")
        if entry["output_mode"] not in _OUTPUT_MODES:
            _fail("unknown_variant", entry_path + ("output_mode",), "output_mode is unknown")
        if entry["memory_policy"] not in _MEMORY_POLICIES:
            _fail("unknown_variant", entry_path + ("memory_policy",), "memory_policy is unknown")
        if type(entry["parallel_safe"]) is not bool:
            _fail("invalid_type", entry_path + ("parallel_safe",), "parallel_safe must be a boolean")
        for field in ("allowed_tools", "disabled_tools"):
            tools = entry[field]
            if field == "allowed_tools" and tools is None:
                continue
            if type(tools) is not list or any(type(tool) is not str or not tool.strip() for tool in tools):
                _fail("invalid_type", entry_path + (field,), f"{field} must contain non-empty strings")
            if len(tools) != len(set(tools)):
                _fail("duplicate_name", entry_path + (field,), f"{field} contains duplicates")
        if entry["allowed_tools"] is not None and set(entry["allowed_tools"]) & set(entry["disabled_tools"]):
            _fail("invalid_value", entry_path + ("disabled_tools",), "a tool cannot be both allowed and disabled")

        source_path = entry_path + ("source",)
        if type(entry["source"]) is not dict or type(entry["source"].get("kind")) is not str:
            _fail("invalid_type", source_path, "subagent source must be an object with kind")
        source = entry["source"]
        if source["kind"] == "agent_ref":
            source = _closed(source, source_path, {"kind", "agent_id"})
            _text(source["agent_id"], source_path + ("agent_id",), "Agent resource id")
        elif source["kind"] == "embedded":
            source = _closed(source, source_path, {"kind", "document"})
            if type(source["document"]) is not dict:
                _fail("invalid_type", source_path + ("document",), "embedded Agent must be an object")
            validate_embedded(source["document"], source_path + ("document",))
        else:
            _fail("unknown_variant", source_path + ("kind",), "subagent source must be agent_ref or embedded")
        entries.append(deepcopy(entry))
    return tuple(entries)


def attached_subagents(document: dict[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return validated pool entries grouped by their attached Kernel node."""
    nodes = {node["id"]: node for node in document["nodes"]}
    pools = {
        node_id: tuple(deepcopy(node["config"]["subagents"]))
        for node_id, node in nodes.items()
        if node["type"] == "subagent_pool"
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for edge in document["edges"]:
        if edge["kind"] != "attach":
            continue
        source, target = edge["source_node_id"], edge["target_node_id"]
        pool_id = source if source in pools else target if target in pools else None
        agent_id = target if pool_id == source else source if pool_id == target else None
        if pool_id is not None and agent_id is not None and nodes[agent_id]["type"] == "agent":
            result.setdefault(agent_id, []).extend(deepcopy(pools[pool_id]))
    return {node_id: tuple(entries) for node_id, entries in result.items()}


__all__ = ["attached_subagents", "validate_subagent_pool_config"]
