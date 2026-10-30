"""Closed, provider-free recovery envelopes for prepared Workflows."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .errors import SkeletonValidationError
from .io import _closed_object as _io_closed_object, _validate_json as _io_validate_json
from .preparation import PreparedWorkflowPlan, prepare_workflow


_SCHEMA = "unchain.skeleton_workflow_recovery.v1"
_HEX = set("0123456789abcdef")


class RecoveryEnvelopeError(SkeletonValidationError):
    """Raised when a Workflow recovery envelope is not admissible."""


def _closed_object(
    value: Any,
    path: tuple[str | int, ...],
    required: set[str],
    optional: set[str] = set(),
) -> dict[str, Any]:
    try:
        return _io_closed_object(value, path, required, optional)
    except SkeletonValidationError as error:
        raise RecoveryEnvelopeError(error.code, error.path, str(error)) from error


def _validate_json(value: Any, path: tuple[str | int, ...]) -> None:
    try:
        _io_validate_json(value, path)
    except SkeletonValidationError as error:
        raise RecoveryEnvelopeError(error.code, error.path, str(error)) from error


def _clone(value: Any) -> Any:
    if type(value) is dict:
        return {key: _clone(child) for key, child in value.items()}
    if type(value) is list:
        return [_clone(child) for child in value]
    return value


def _integer_decimal(value: int) -> bytes:
    """Encode any Python integer without consulting the process digit limit."""
    if value == 0:
        return b"0"
    negative = value < 0
    remaining = -value if negative else value
    chunks: list[int] = []
    while remaining:
        remaining, chunk = divmod(remaining, 1_000_000_000)
        chunks.append(chunk)
    pieces = [str(chunks.pop()).encode("ascii")]
    while chunks:
        pieces.append(f"{chunks.pop():09d}".encode("ascii"))
    return (b"-" if negative else b"") + b"".join(pieces)


def _string_json(value: str) -> bytes:
    encoded = bytearray(b'"')
    escapes = {
        '"': b'\\"',
        "\\": b"\\\\",
        "\b": b"\\b",
        "\f": b"\\f",
        "\n": b"\\n",
        "\r": b"\\r",
        "\t": b"\\t",
    }
    for character in value:
        escaped = escapes.get(character)
        if escaped is not None:
            encoded.extend(escaped)
            continue
        codepoint = ord(character)
        if codepoint < 0x20 or 0xD800 <= codepoint <= 0xDFFF:
            encoded.extend(f"\\u{codepoint:04x}".encode("ascii"))
        else:
            encoded.extend(character.encode("utf-8"))
    encoded.append(ord('"'))
    return bytes(encoded)


def _canonical_value(value: Any, parts: list[bytes]) -> None:
    if value is None:
        parts.append(b"null")
    elif type(value) is bool:
        parts.append(b"true" if value else b"false")
    elif type(value) is int:
        parts.append(_integer_decimal(value))
    elif type(value) is float:
        parts.append(json.dumps(value, allow_nan=False).encode("ascii"))
    elif type(value) is str:
        parts.append(_string_json(value))
    elif type(value) is list:
        parts.append(b"[")
        for index, child in enumerate(value):
            if index:
                parts.append(b",")
            _canonical_value(child, parts)
        parts.append(b"]")
    elif type(value) is dict:
        parts.append(b"{")
        for index, key in enumerate(sorted(value)):
            if index:
                parts.append(b",")
            parts.append(_string_json(key))
            parts.append(b":")
            _canonical_value(value[key], parts)
        parts.append(b"}")
    else:
        raise TypeError("value is not JSON")


def _canonical(value: Any, path: tuple[str | int, ...]) -> bytes:
    try:
        parts: list[bytes] = []
        _canonical_value(value, parts)
        return b"".join(parts)
    except (TypeError, ValueError, RecursionError):
        _raise("invalid_json", path, "value cannot be canonically encoded as JSON")


def _raise(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise RecoveryEnvelopeError(code, path, message)


def _text(value: Any, path: tuple[str | int, ...], label: str) -> str:
    if type(value) is not str:
        _raise("invalid_type", path, f"{label} must be a string")
    if not value:
        _raise("invalid_value", path, f"{label} must be non-empty")
    return value


def _digest(value: Any, path: tuple[str | int, ...]) -> str:
    return hashlib.sha256(_canonical(value, path)).hexdigest()


def _is_digest(value: Any) -> bool:
    return type(value) is str and len(value) == 64 and all(char in _HEX for char in value)


def _checked_plan(plan: Any) -> PreparedWorkflowPlan:
    if type(plan) is not PreparedWorkflowPlan:
        _raise("invalid_plan", ("plan",), "plan must be a PreparedWorkflowPlan")
    try:
        checked = prepare_workflow(plan.snapshot())
        if (
            plan.document_id != checked.document_id
            or plan.node_ids != checked.node_ids
            or plan.topological_node_ids != checked.topological_node_ids
            or plan.flow_edges != checked.flow_edges
        ):
            _raise("invalid_plan", ("plan",), "plan does not match its checked workflow snapshot")
    except (SkeletonValidationError, ValueError, TypeError, AttributeError, IndexError, KeyError, RecursionError) as error:
        if isinstance(error, RecoveryEnvelopeError):
            raise
        _raise("invalid_plan", ("plan",), "plan snapshot is not a prepared workflow")
    return checked


def workflow_definition_digest(plan: PreparedWorkflowPlan) -> str:
    """Return the canonical definition SHA-256 for a checked Workflow plan."""
    checked = _checked_plan(plan)
    return _digest(checked.snapshot(), ("plan", "snapshot"))


def _topology_digest(plan: PreparedWorkflowPlan) -> str:
    return _digest(
        {
            "node_ids": list(plan.node_ids),
            "flow_edges": [
                {
                    "id": edge.id,
                    "source_node_id": edge.source_node_id,
                    "source_port_id": edge.source_port_id,
                    "target_node_id": edge.target_node_id,
                    "target_port_id": edge.target_port_id,
                }
                for edge in plan.flow_edges
            ],
        },
        ("plan", "topology"),
    )


def _definition(plan: PreparedWorkflowPlan) -> dict[str, str]:
    return {
        "id": plan.document_id,
        "definition_digest": workflow_definition_digest(plan),
        "topology_digest": _topology_digest(plan),
    }


def _frame(value: Any, path: tuple[str | int, ...], final: bool) -> dict[str, Any]:
    required = {"execution_id", "graph_invocation_id", "attempt_id", "definition_id", "definition_digest"}
    optional = set() if final else {"caller_node_invocation_id"}
    obj = _closed_object(value, path, required, optional)
    for key in required:
        _text(obj[key], path + (key,), key)
    if not final:
        if "caller_node_invocation_id" not in obj:
            _raise("missing_field", path + ("caller_node_invocation_id",), "caller node invocation is required")
        _text(obj["caller_node_invocation_id"], path + ("caller_node_invocation_id",), "caller_node_invocation_id")
    if not _is_digest(obj["definition_digest"]):
        _raise("invalid_value", path + ("definition_digest",), "definition_digest must be SHA-256 hex")
    return obj


def _active(value: Any, path: tuple[str | int, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _closed_object(value, path, {"kind", "node_id", "node_invocation_id", "attempt_id"}, {"interaction_id", "request_receipt_id", "intent_receipt_id"})
    kind = _text(obj["kind"], path + ("kind",), "active kind")
    for key in ("node_id", "node_invocation_id", "attempt_id"):
        _text(obj[key], path + (key,), key)
    if kind == "waiting_interaction":
        if set(obj) != {"kind", "node_id", "node_invocation_id", "attempt_id", "interaction_id", "request_receipt_id"}:
            _raise("invalid_shape", path, "waiting interaction fields are invalid")
        _text(obj["interaction_id"], path + ("interaction_id",), "interaction_id")
        _text(obj["request_receipt_id"], path + ("request_receipt_id",), "request_receipt_id")
    elif kind == "outcome_unknown":
        if set(obj) != {"kind", "node_id", "node_invocation_id", "attempt_id", "intent_receipt_id"}:
            _raise("invalid_shape", path, "unknown outcome fields are invalid")
        _text(obj["intent_receipt_id"], path + ("intent_receipt_id",), "intent_receipt_id")
    else:
        _raise("unknown_variant", path + ("kind",), "unknown active record kind")
    return obj


def _completed(value: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    obj = _closed_object(value, path, {"node_id", "node_invocation_id", "attempt_id", "input_receipt_id", "output_receipt_id"})
    for key in obj:
        _text(obj[key], path + (key,), key)
    return obj


def _selected(value: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    obj = _closed_object(value, path, {"edge_id", "source_node_id", "source_port_id", "target_node_id", "target_port_id"})
    for key in obj:
        _text(obj[key], path + (key,), key)
    return obj


def _terminal(value: Any, path: tuple[str | int, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    obj = _closed_object(value, path, {"kind", "node_id", "node_invocation_id", "end_receipt_id"})
    if obj["kind"] != "end_completed":
        _raise("unknown_variant", path + ("kind",), "terminal kind must be end_completed")
    for key in ("node_id", "node_invocation_id", "end_receipt_id"):
        _text(obj[key], path + (key,), key)
    return obj


def validate_workflow_recovery_envelope(envelope: Any, plan: PreparedWorkflowPlan) -> dict[str, Any]:
    """Validate an in-memory v1 envelope against a checked Workflow plan."""
    checked = _checked_plan(plan)
    _validate_json(envelope, ("envelope",))
    root = _closed_object(envelope, ("envelope",), {"schema", "definition", "invocation", "call_stack", "completed_nodes", "selected_edges", "active", "terminal"})
    if root["schema"] != _SCHEMA:
        _raise("unknown_variant", ("envelope", "schema"), "recovery schema is unsupported")
    definition = _closed_object(root["definition"], ("envelope", "definition"), {"id", "definition_digest", "topology_digest"})
    expected_definition = _definition(checked)
    for key, expected in expected_definition.items():
        if definition.get(key) != expected:
            _raise("identity_mismatch", ("envelope", "definition", key), "definition identity does not match plan")
    invocation = _closed_object(root["invocation"], ("envelope", "invocation"), {"execution_id", "graph_invocation_id", "attempt_id"})
    for key in invocation:
        _text(invocation[key], ("envelope", "invocation", key), key)
    if type(root["call_stack"]) is not list or not root["call_stack"]:
        _raise("invalid_value", ("envelope", "call_stack"), "call_stack must be a non-empty array")
    frames = [_frame(frame, ("envelope", "call_stack", index), index == len(root["call_stack"]) - 1) for index, frame in enumerate(root["call_stack"])]
    final = frames[-1]
    if any(final[key] != invocation[key] for key in invocation) or final["definition_id"] != expected_definition["id"] or final["definition_digest"] != expected_definition["definition_digest"]:
        _raise("lineage_mismatch", ("envelope", "call_stack", len(frames) - 1), "final call frame must match invocation and definition")
    graph_ids: set[str] = set()
    call_attempt_ids: set[str] = set()
    caller_ids: set[str] = set()
    for index, frame in enumerate(frames):
        if frame["execution_id"] != invocation["execution_id"]:
            _raise("lineage_mismatch", ("envelope", "call_stack", index, "execution_id"), "call frame execution must match invocation")
        graph_id = frame["graph_invocation_id"]
        if graph_id in graph_ids:
            _raise("duplicate_name", ("envelope", "call_stack", index, "graph_invocation_id"), "graph invocation ids must be unique")
        graph_ids.add(graph_id)
        attempt_id = frame["attempt_id"]
        if attempt_id in call_attempt_ids:
            _raise("duplicate_name", ("envelope", "call_stack", index, "attempt_id"), "call frame attempt ids must be unique")
        call_attempt_ids.add(attempt_id)
        if index < len(frames) - 1:
            caller_id = frame["caller_node_invocation_id"]
            if caller_id in caller_ids:
                _raise("duplicate_name", ("envelope", "call_stack", index, "caller_node_invocation_id"), "caller node invocations must be unique")
            caller_ids.add(caller_id)

    if type(root["completed_nodes"]) is not list:
        _raise("invalid_type", ("envelope", "completed_nodes"), "completed_nodes must be an array")
    if type(root["selected_edges"]) is not list:
        _raise("invalid_type", ("envelope", "selected_edges"), "selected_edges must be an array")
    completed = [_completed(item, ("envelope", "completed_nodes", index)) for index, item in enumerate(root["completed_nodes"])]
    selected = [_selected(item, ("envelope", "selected_edges", index)) for index, item in enumerate(root["selected_edges"])]
    active = _active(root["active"], ("envelope", "active"))
    terminal = _terminal(root["terminal"], ("envelope", "terminal"))
    snapshot = checked.snapshot()
    nodes = {node["id"]: node for node in snapshot["nodes"]}
    starts = [node["id"] for node in snapshot["nodes"] if node["type"] == "start"]
    ends = [node["id"] for node in snapshot["nodes"] if node["type"] == "end"]
    plan_edges = {edge.id: edge for edge in checked.flow_edges}
    seen_nodes: set[str] = set()
    seen_invocations: set[str] = set()
    seen_attempts: set[str] = set()
    seen_receipts: set[str] = set()
    for index, item in enumerate(completed):
        node_id = item["node_id"]
        if node_id not in nodes:
            _raise("foreign_node", ("envelope", "completed_nodes", index, "node_id"), "completed node is not in plan")
        if node_id in seen_nodes:
            _raise("duplicate_name", ("envelope", "completed_nodes", index, "node_id"), "completed node is duplicated")
        if item["node_invocation_id"] in seen_invocations:
            _raise("duplicate_name", ("envelope", "completed_nodes", index, "node_invocation_id"), "node invocation is duplicated")
        if item["attempt_id"] in seen_attempts:
            _raise("duplicate_name", ("envelope", "completed_nodes", index, "attempt_id"), "node attempt is duplicated")
        if item["attempt_id"] in call_attempt_ids:
            _raise("duplicate_name", ("envelope", "completed_nodes", index, "attempt_id"), "node attempt conflicts with call stack")
        for field in ("input_receipt_id", "output_receipt_id"):
            if item[field] in seen_receipts:
                _raise("duplicate_name", ("envelope", "completed_nodes", index, field), "receipt id is duplicated")
            seen_receipts.add(item[field])
        seen_nodes.add(node_id); seen_invocations.add(item["node_invocation_id"])
        seen_attempts.add(item["attempt_id"])
    if completed and completed[0]["node_id"] != starts[0]:
        _raise("path_mismatch", ("envelope", "completed_nodes", 0, "node_id"), "completed path must begin at start")
    selected_by_source: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(selected):
        edge = plan_edges.get(item["edge_id"])
        if edge is None:
            _raise("foreign_edge", ("envelope", "selected_edges", index, "edge_id"), "selected edge is not in plan")
        if item["source_node_id"] != edge.source_node_id or item["source_port_id"] != edge.source_port_id or item["target_node_id"] != edge.target_node_id or item["target_port_id"] != edge.target_port_id:
            _raise("edge_mismatch", ("envelope", "selected_edges", index), "selected edge does not match plan")
        if item["source_node_id"] in selected_by_source:
            _raise("duplicate_name", ("envelope", "selected_edges", index, "source_node_id"), "selected edge source is duplicated")
        if item["source_node_id"] not in seen_nodes:
            _raise("path_mismatch", ("envelope", "selected_edges", index, "source_node_id"), "selected edge must start at a completed node")
        selected_by_source[item["source_node_id"]] = item
    expected_sources = [item["node_id"] for item in completed if item["node_id"] != ends[0]]
    if len(selected) != len(expected_sources):
        _raise("path_mismatch", ("envelope", "selected_edges"), "selected edges must cover completed non-end nodes")
    for index, node_id in enumerate(expected_sources):
        if selected[index]["source_node_id"] != node_id:
            _raise("path_mismatch", ("envelope", "selected_edges", index, "source_node_id"), "selected edges must follow completed receipt order")
    for index, item in enumerate(completed):
        node_id = item["node_id"]
        if node_id == ends[0]:
            if node_id in selected_by_source:
                _raise("path_mismatch", ("envelope", "selected_edges"), "end cannot select an edge")
            continue
        selected_edge = selected_by_source.get(node_id)
        if selected_edge is None:
            _raise("missing_value", ("envelope", "selected_edges"), "completed non-end node needs a selected edge")
        if index + 1 < len(completed) and selected_edge["target_node_id"] != completed[index + 1]["node_id"]:
            _raise("path_mismatch", ("envelope", "selected_edges"), "selected edge must continue completed path")
    if active is not None:
        if active["node_id"] not in nodes:
            _raise("foreign_node", ("envelope", "active", "node_id"), "active node is not in plan")
        if active["node_id"] in seen_nodes:
            _raise("path_mismatch", ("envelope", "active", "node_id"), "active node is already completed")
        if active["node_invocation_id"] in seen_invocations:
            _raise("duplicate_name", ("envelope", "active", "node_invocation_id"), "active node invocation is duplicated")
        if active["attempt_id"] in seen_attempts:
            _raise("duplicate_name", ("envelope", "active", "attempt_id"), "active node attempt is duplicated")
        if active["attempt_id"] in call_attempt_ids:
            _raise("duplicate_name", ("envelope", "active", "attempt_id"), "active node attempt conflicts with call stack")
        active_receipts = ("interaction_id", "request_receipt_id") if active["kind"] == "waiting_interaction" else ("intent_receipt_id",)
        for field in active_receipts:
            if active[field] in seen_receipts:
                _raise("duplicate_name", ("envelope", "active", field), "active receipt conflicts with completed receipt")
        if active["node_id"] == ends[0]:
            _raise("invalid_value", ("envelope", "active", "node_id"), "end cannot be active")
        if completed:
            last = selected_by_source.get(completed[-1]["node_id"])
            if last is None or last["target_node_id"] != active["node_id"]:
                _raise("path_mismatch", ("envelope", "active", "node_id"), "active node must follow completed path")
        elif active["node_id"] != starts[0]:
            _raise("path_mismatch", ("envelope", "active", "node_id"), "initial active node must be start")
    if terminal is not None:
        if active is not None:
            _raise("invalid_value", ("envelope", "terminal"), "terminal cannot coexist with active node")
        if terminal["node_id"] != ends[0]:
            _raise("invalid_value", ("envelope", "terminal", "node_id"), "terminal node must be end")
        if not completed or completed[-1]["node_id"] != ends[0]:
            _raise("path_mismatch", ("envelope", "terminal", "node_id"), "completed path must reach end")
        end = completed[-1]
        if terminal["node_invocation_id"] != end["node_invocation_id"] or terminal["end_receipt_id"] != end["output_receipt_id"]:
            _raise("identity_mismatch", ("envelope", "terminal"), "terminal must match end completion receipt")
    elif active is None:
        _raise("missing_value", ("envelope", "active"), "an active or terminal record is required")
    return _clone(root)


def build_workflow_recovery_envelope(
    plan: PreparedWorkflowPlan,
    *,
    invocation: dict[str, Any],
    call_stack: list[dict[str, Any]],
    completed_nodes: list[dict[str, Any]] | None = None,
    selected_edges: list[dict[str, Any]] | None = None,
    active: dict[str, Any] | None = None,
    terminal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate a detached v1 envelope from explicit receipt records."""
    checked = _checked_plan(plan)
    envelope = {
        "schema": _SCHEMA,
        "definition": _definition(checked),
        "invocation": invocation,
        "call_stack": call_stack,
        "completed_nodes": [] if completed_nodes is None else completed_nodes,
        "selected_edges": [] if selected_edges is None else selected_edges,
        "active": active,
        "terminal": terminal,
    }
    return validate_workflow_recovery_envelope(envelope, checked)
