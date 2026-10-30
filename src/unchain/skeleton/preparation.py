"""Provider-free preparation of structurally complete Workflow definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import SkeletonValidationError
from .validation import validate_skeleton_for_execution


def _freeze(value: Any) -> Any:
    if type(value) is dict:
        return ("object", tuple((key, _freeze(child)) for key, child in value.items()))
    if type(value) is list:
        return ("array", tuple(_freeze(child) for child in value))
    return value


def _thaw(value: Any) -> Any:
    if type(value) is tuple and value[0] == "object":
        return {key: _thaw(child) for key, child in value[1]}
    if type(value) is tuple and value[0] == "array":
        return [_thaw(child) for child in value[1]]
    return value


@dataclass(frozen=True)
class PreparedFlowEdge:
    id: str
    source_node_id: str
    source_port_id: str
    target_node_id: str
    target_port_id: str


@dataclass(frozen=True)
class PreparedWorkflowPlan:
    document_id: str
    node_ids: tuple[str, ...]
    topological_node_ids: tuple[str, ...]
    flow_edges: tuple[PreparedFlowEdge, ...]
    _encoded_snapshot: tuple[Any, ...]

    def snapshot(self) -> dict[str, Any]:
        return _thaw(self._encoded_snapshot)


def prepare_workflow(document: Any) -> PreparedWorkflowPlan:
    """Preflight and encode a Workflow without importing model infrastructure."""
    snapshot = validate_skeleton_for_execution(document)
    if snapshot["kind"] != "workflow":
        raise SkeletonValidationError("wrong_kind", ("document", "kind"), "preparation requires workflow kind")
    node_ids = tuple(node["id"] for node in snapshot["nodes"])
    order = {node_id: index for index, node_id in enumerate(node_ids)}
    flow = [edge for edge in snapshot["edges"] if edge["kind"] == "flow"]
    outgoing = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in flow:
        outgoing[edge["source_node_id"]].append(edge["target_node_id"])
        indegree[edge["target_node_id"]] += 1
    available = [node_id for node_id in node_ids if indegree[node_id] == 0]
    topological: list[str] = []
    while available:
        available.sort(key=order.__getitem__)
        current = available.pop(0)
        topological.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
    return PreparedWorkflowPlan(
        document_id=snapshot["id"],
        node_ids=node_ids,
        topological_node_ids=tuple(topological),
        flow_edges=tuple(PreparedFlowEdge(edge["id"], edge["source_node_id"], edge["source_port_id"], edge["target_node_id"], edge["target_port_id"]) for edge in flow),
        _encoded_snapshot=_freeze(snapshot),
    )
