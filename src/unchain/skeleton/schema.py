"""Provider-free validation for the version-one Skeleton envelope."""

from __future__ import annotations

from typing import Any

from .io import _closed_object, _error, _validate_json

_NODE_TYPES = {
    "start", "end", "agent", "toolkit_pool", "subagent_pool", "if_else",
    "switch_case", "tool_call", "skill_read", "code", "workflow_call", "hooks_pool",
}
_WORKFLOW_FORBIDDEN_TYPES = {"agent", "toolkit_pool", "subagent_pool", "hooks_pool"}


def _nonempty_string(value: Any, path: tuple[str | int, ...], label: str) -> None:
    if type(value) is not str:
        _error("invalid_type", path, f"{label} must be a string")
    if not value:
        _error("invalid_value", path, f"{label} must be non-empty")


def _clone_json_tree(value: Any) -> Any:
    """Clone every JSON occurrence without deepcopy memo alias preservation."""
    if type(value) is dict:
        return {key: _clone_json_tree(child) for key, child in value.items()}
    if type(value) is list:
        return [_clone_json_tree(child) for child in value]
    return value


def validate_skeleton_document(document: Any) -> dict[str, Any]:
    """Validate and detach a closed version-one Skeleton document."""
    _validate_json(document, ("document",))
    root = _closed_object(
        document,
        ("document",),
        {"schema_version", "kind", "id", "name", "nodes", "edges"},
        {"description"},
    )
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        _error("invalid_value", ("document", "schema_version"), "schema_version must be integer 1")
    if type(root["kind"]) is not str:
        _error("invalid_type", ("document", "kind"), "kind must be a string")
    if root["kind"] not in {"agent", "workflow"}:
        _error("unknown_variant", ("document", "kind"), "kind must be agent or workflow")
    _nonempty_string(root["id"], ("document", "id"), "id")
    _nonempty_string(root["name"], ("document", "name"), "name")
    if "description" in root and type(root["description"]) is not str:
        _error("invalid_type", ("document", "description"), "description must be a string")
    if type(root["nodes"]) is not list:
        _error("invalid_type", ("document", "nodes"), "nodes must be an array")
    if type(root["edges"]) is not list:
        _error("invalid_type", ("document", "edges"), "edges must be an array")

    node_ids: set[str] = set()
    for index, node in enumerate(root["nodes"]):
        path = ("document", "nodes", index)
        obj = _closed_object(node, path, {"id", "type", "config"})
        _nonempty_string(obj["id"], path + ("id",), "node id")
        if obj["id"] in node_ids:
            _error("duplicate_name", path + ("id",), "node id is duplicated")
        node_ids.add(obj["id"])
        if type(obj["type"]) is not str:
            _error("invalid_type", path + ("type",), "node type must be a string")
        if obj["type"] not in _NODE_TYPES:
            _error("unknown_variant", path + ("type",), "unknown node type")
        if root["kind"] == "workflow" and obj["type"] in _WORKFLOW_FORBIDDEN_TYPES:
            _error("invalid_value", path + ("type",), "node type is forbidden in workflow")
        if type(obj["config"]) is not dict:
            _error("invalid_type", path + ("config",), "node config must be an object")

    edge_ids: set[str] = set()
    for index, edge in enumerate(root["edges"]):
        path = ("document", "edges", index)
        obj = _closed_object(
            edge,
            path,
            {"id", "kind", "source_node_id", "source_port_id", "target_node_id", "target_port_id"},
        )
        _nonempty_string(obj["id"], path + ("id",), "edge id")
        if obj["id"] in edge_ids:
            _error("duplicate_name", path + ("id",), "edge id is duplicated")
        edge_ids.add(obj["id"])
        if type(obj["kind"]) is not str:
            _error("invalid_type", path + ("kind",), "edge kind must be a string")
        if obj["kind"] not in {"flow", "attach"}:
            _error("unknown_variant", path + ("kind",), "edge kind must be flow or attach")
        for field in ("source_node_id", "source_port_id", "target_node_id", "target_port_id"):
            _nonempty_string(obj[field], path + (field,), field)
        if obj["source_node_id"] not in node_ids:
            _error("missing_value", path + ("source_node_id",), "edge source node is unavailable")
        if obj["target_node_id"] not in node_ids:
            _error("missing_value", path + ("target_node_id",), "edge target node is unavailable")

    return _clone_json_tree(root)
