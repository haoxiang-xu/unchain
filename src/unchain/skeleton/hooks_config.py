"""Closed, provider-free Hook Pool admission for Skeleton Agent graphs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .errors import SkeletonValidationError


_EVENTS = {"before_tool_call", "after_tool_call"}
_MAX_ORDER = 1_000_000
_ENTRY_REQUIRED = frozenset({"id", "order", "event", "tool_name", "workflow", "force_approval"})
_ENTRY_OPTIONAL = frozenset({"enabled"})


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _closed(value: Any, path: tuple[str | int, ...], required: set[str],
            optional: set[str] = frozenset()) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("invalid_type", path, "value must be an object")
    missing = required - set(value)
    if missing:
        _fail("missing_field", path + (sorted(missing)[0],), "required field is missing")
    extra = set(value) - required - optional
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

    A Hook invokes a pinned reference or an owned embedded Workflow. Resource
    existence belongs to the host; embedded bodies are checked in place.
    """
    root = _closed(config, path, {"hooks"})
    raw_entries = root["hooks"]
    if type(raw_entries) is not list:
        _fail("invalid_type", path + ("hooks",), "hooks must be an array")
    if not raw_entries:
        _fail("missing_value", path + ("hooks",), "hooks pool must contain a Hook")

    entries: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_entries):
        entry_path = path + ("hooks", index)
        entry = _closed(raw, entry_path, _ENTRY_REQUIRED, _ENTRY_OPTIONAL)
        if "enabled" in entry and type(entry["enabled"]) is not bool:
            _fail("invalid_type", entry_path + ("enabled",), "enabled must be a boolean")
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
        matcher = entry["tool_name"]
        if type(matcher) is list:
            if not matcher:
                _fail("invalid_value", entry_path + ("tool_name",), "tool set must not be empty")
            names: set[str] = set()
            for at, value in enumerate(matcher):
                name_path = entry_path + ("tool_name", at)
                name = _text(value, name_path, "tool name")
                if name in names:
                    _fail("duplicate_name", name_path, "tool name is duplicated")
                names.add(name)
        elif matcher is not None:
            _text(matcher, entry_path + ("tool_name",), "tool name")
        workflow_path = entry_path + ("workflow",)
        workflow = entry["workflow"]
        if type(workflow) is dict and "kind" in workflow:
            if workflow["kind"] != "embedded":
                _fail("unknown_variant", workflow_path + ("kind",), "Hook Workflow source is unknown")
            _closed(workflow, workflow_path, {"kind", "document"})
            body_path = workflow_path + ("document",)
            if type(workflow["document"]) is not dict:
                _fail("invalid_type", body_path, "embedded Workflow must be an object")
            if workflow["document"].get("kind") != "workflow":
                _fail("wrong_kind", body_path + ("kind",), "embedded Hook requires Workflow kind")
            from .preparation import prepare_workflow
            from .execution import _validate_all
            try:
                plan = prepare_workflow(workflow["document"])
                _validate_all(plan.snapshot())
            except SkeletonValidationError as exc:
                raise SkeletonValidationError(exc.code, body_path + exc.path[1:], str(exc)) from exc
        else:
            workflow = _closed(workflow, workflow_path, {"workflow_id", "revision_id"})
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
        node_id: tuple(deepcopy(hook) for hook in node["config"]["hooks"]
                       if hook.get("enabled") is not False)
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
        for node_id, entries in grouped.items() if entries
    }


def hook_entry_contract() -> dict[str, Any]:
    """Advertise this imported validator's closed fields, detached from storage.

    Omission retains historical enabled behavior. No default is inserted into
    a document, so historical definition digests and parked runs stay intact.
    Disabled entries remain structurally validated and stored, but are not
    executable bindings and need no dependency resolution.
    """
    return {"schema": "unchain.skeleton_hook_entry.v1",
            "required_fields": sorted(_ENTRY_REQUIRED),
            "optional_fields": sorted(_ENTRY_OPTIONAL), "omitted_enabled": True}


def hook_matches_tool(matcher: Any, tool_name: str) -> bool:
    """Match an admitted all/single/set selector against the invocation identity.

    No normalization, title lookup, glob or permission change is involved.
    The closed validator owns list/member validity before bindings are made.
    """
    if type(tool_name) is not str:
        return False
    if matcher is None:
        return True
    if type(matcher) is str:
        return matcher == tool_name
    return type(matcher) is list and tool_name in matcher


def hook_tool_set_contract() -> dict[str, Any]:
    """The versioned set extension; the legacy enable getter stays unchanged."""
    return {"schema": "unchain.skeleton_hook_tool_set.v1", "field": "tool_name",
            "legacy_types": ["null", "string"], "set_type": "array", "minimum_size": 1,
            "duplicates": "reject", "names": "nonblank_exact", "normalization": "none"}


def hook_workflow_contract() -> dict[str, Any]:
    """Advertise owned bodies separately from the unchanged enable contract."""
    return {"schema": "unchain.skeleton_hook_workflow.v1", "field": "workflow",
            "reference_fields": ["revision_id", "workflow_id"],
            "embedded_fields": ["document", "kind"], "embedded_kind": "embedded",
            "document_kind": "workflow", "identity": "embedded-sha256"}


__all__ = ["attached_hooks", "hook_entry_contract", "hook_tool_set_contract", "hook_workflow_contract",
           "hook_matches_tool", "validate_hooks_pool_config"]
