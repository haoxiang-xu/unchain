"""Closed, provider-free Hook Pool admission for Skeleton Agent graphs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import SkeletonValidationError


_EVENTS = {"before_tool_call", "after_tool_call"}
_MAX_ORDER = 1_000_000


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


def validate_hooks_pool_config(
    config: Any,
    path: tuple[str | int, ...],
) -> tuple[dict[str, Any], ...]:
    """Validate a fixed Hook Pool and return entries in execution order.

    A Hook always invokes one pinned, published Workflow.  The resource
    resolver verifies the referenced resource is a Workflow at run preflight;
    this producer-side validator owns only the closed document wire.
    """
    root = _closed(config, path, {"hooks"})
    raw_entries = root["hooks"]
    if type(raw_entries) is not list:
        _fail("invalid_type", path + ("hooks",), "hooks must be an array")
    if not raw_entries:
        _fail("missing_value", path + ("hooks",), "hooks pool must contain a Hook")

    entries: list[dict[str, Any]] = []
    ids: set[str] = set()
    required = {"id", "order", "event", "tool_name", "workflow", "force_approval"}
    for index, raw in enumerate(raw_entries):
        entry_path = path + ("hooks", index)
        entry = _closed(raw, entry_path, required)
        hook_id = _text(entry["id"], entry_path + ("id",), "hook id")
        if hook_id in ids:
            _fail("duplicate_name", entry_path + ("id",), "hook id is duplicated")
        ids.add(hook_id)
        if type(entry["order"]) is not int or isinstance(entry["order"], bool):
            _fail("invalid_type", entry_path + ("order",), "hook order must be an integer")
        if not 0 <= entry["order"] <= _MAX_ORDER:
            _fail("invalid_value", entry_path + ("order",), "hook order is outside the supported range")
        if type(entry["event"]) is not str or entry["event"] not in _EVENTS:
            _fail("unknown_variant", entry_path + ("event",), "hook event is unknown")
        if entry["tool_name"] is not None:
            _text(entry["tool_name"], entry_path + ("tool_name",), "tool name")
        workflow_path = entry_path + ("workflow",)
        workflow = _closed(entry["workflow"], workflow_path, {"workflow_id", "revision_id"})
        _text(workflow["workflow_id"], workflow_path + ("workflow_id",), "Workflow resource id")
        _text(workflow["revision_id"], workflow_path + ("revision_id",), "Workflow revision id")
        if type(entry["force_approval"]) is not bool:
            _fail("invalid_type", entry_path + ("force_approval",), "force_approval must be a boolean")
        entries.append(deepcopy(entry))
    return tuple(sorted(entries, key=lambda entry: (entry["order"], entry["id"])))


def attached_hooks(document: dict[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    """Return Hook entries grouped by their attached Kernel node in stable order."""
    nodes = {node["id"]: node for node in document["nodes"]}
    pools = {
        node_id: tuple(deepcopy(node["config"]["hooks"]))
        for node_id, node in nodes.items()
        if node["type"] == "hooks_pool"
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for edge in document["edges"]:
        if edge["kind"] != "attach":
            continue
        source, target = edge["source_node_id"], edge["target_node_id"]
        pool_id = source if source in pools else target if target in pools else None
        agent_id = target if pool_id == source else source if pool_id == target else None
        if pool_id is not None and agent_id is not None and nodes[agent_id]["type"] == "agent":
            grouped.setdefault(agent_id, []).extend(deepcopy(pools[pool_id]))
    return {
        node_id: tuple(sorted(entries, key=lambda entry: (entry["order"], entry["id"])))
        for node_id, entries in grouped.items()
    }


__all__ = ["attached_hooks", "validate_hooks_pool_config"]
