"""Provider-free validation for the version-one Skeleton envelope."""

from __future__ import annotations

import re
from typing import Any

from .io import _closed_object, _error, _validate_json

_NODE_TYPES = {
    "start", "end", "agent", "toolkit_pool", "subagent_pool", "if_else",
    "switch_case", "tool_call", "skill_read", "code", "workflow_call", "hooks_pool",
}
_WORKFLOW_FORBIDDEN_TYPES = {"agent", "toolkit_pool", "subagent_pool", "hooks_pool"}
# Display-only document presentation. The host owns the icon catalogue; the
# runtime admits only the closed shape and these exact ASCII patterns.
_PRESENTATION_ICON = re.compile(r"[a-z0-9_]{1,40}")
_PRESENTATION_COLOR = re.compile(r"#[0-9a-f]{6}")
# Display-only node coordinates. Bounded so a stored drawing cannot carry a
# value no canvas can place; the host owns what a coordinate means.
_LAYOUT_COORDINATE_LIMIT = 1_000_000


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


def _validate_presentation(value: Any) -> None:
    """Admit the closed, display-only ``{icon?, color?}`` object."""
    path = ("document", "presentation")
    presentation = _closed_object(value, path, set(), {"icon", "color"})
    for field, pattern, label in (
        ("icon", _PRESENTATION_ICON, "presentation icon must match ^[a-z0-9_]{1,40}$"),
        ("color", _PRESENTATION_COLOR, "presentation color must be lowercase #rrggbb"),
    ):
        if field not in presentation:
            continue
        if type(presentation[field]) is not str:
            _error("invalid_type", path + (field,), f"presentation {field} must be a string")
        if pattern.fullmatch(presentation[field]) is None:
            _error("invalid_value", path + (field,), label)


def _validate_layout(value: Any, node_ids: set[str]) -> None:
    """Admit the closed, display-only ``{nodes: {<node id>: {x, y}}}`` object."""
    path = ("document", "layout")
    layout = _closed_object(value, path, {"nodes"})
    nodes = layout["nodes"]
    if type(nodes) is not dict:
        _error("invalid_type", path + ("nodes",), "layout nodes must be an object")
    for node_id, point in nodes.items():
        point_path = path + ("nodes", node_id)
        if node_id not in node_ids:
            _error("missing_value", point_path, "layout names a node that is not in the document")
        point = _closed_object(point, point_path, {"x", "y"})
        for axis in ("x", "y"):
            coordinate = point[axis]
            if type(coordinate) not in (int, float):
                _error("invalid_type", point_path + (axis,), f"layout {axis} must be a number")
            if not (-_LAYOUT_COORDINATE_LIMIT <= coordinate <= _LAYOUT_COORDINATE_LIMIT):
                _error("invalid_value", point_path + (axis,), f"layout {axis} is out of range")


def without_display_layout(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Remove ``layout`` from a validated snapshot and every embedded child.

    Layout is display-only. It is dropped from the root and from each embedded
    subagent document the snapshot carries, so moving a node, here or inside an
    embedded child, can never change a plan or a definition digest.
    """
    snapshot.pop("layout", None)
    for node in snapshot.get("nodes", ()):
        if type(node) is not dict or node.get("type") != "subagent_pool":
            continue
        config = node.get("config")
        entries = config.get("subagents") if type(config) is dict else None
        for entry in entries if type(entries) is list else ():
            source = entry.get("source") if type(entry) is dict else None
            if type(source) is dict and source.get("kind") == "embedded" and type(source.get("document")) is dict:
                without_display_layout(source["document"])
    return snapshot


def validate_skeleton_document(document: Any) -> dict[str, Any]:
    """Validate and detach a closed version-one Skeleton document."""
    _validate_json(document, ("document",))
    root = _closed_object(
        document,
        ("document",),
        {"schema_version", "kind", "id", "name", "nodes", "edges"},
        {"description", "presentation", "layout"},
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
    if "presentation" in root:
        _validate_presentation(root["presentation"])
    if type(root["nodes"]) is not list:
        _error("invalid_type", ("document", "nodes"), "nodes must be an array")
    if type(root["edges"]) is not list:
        _error("invalid_type", ("document", "edges"), "edges must be an array")

    node_ids: set[str] = set()
    for index, node in enumerate(root["nodes"]):
        path = ("document", "nodes", index)
        obj = _closed_object(node, path, {"id", "type", "config"}, {"label"})
        if "label" in obj:
            if type(obj["label"]) is not str:
                _error("invalid_type", path + ("label",), "node label must be a string")
            if not obj["label"].strip():
                _error("invalid_value", path + ("label",), "node label must be non-blank")
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

    if "layout" in root:
        _validate_layout(root["layout"], node_ids)

    return _clone_json_tree(root)
