"""Prepare an Agent graph while preserving the strict Workflow entry point."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .errors import SkeletonValidationError
from .kernel_config import validate_kernel_config
from .preparation import PreparedWorkflowPlan, PreparedFlowEdge, _freeze
from .recovery import _checked_plan, _canonical, workflow_definition_digest
from .validation import validate_skeleton_for_execution


@dataclass(frozen=True)
class PreparedAgentPlan(PreparedWorkflowPlan):
    """A distinct plan type; it cannot pass the Workflow-only v1 readers."""


def prepare_agent_graph(document, *, _subagent_depth=0):
    snapshot = validate_skeleton_for_execution(document)
    if snapshot["kind"] != "agent":
        raise SkeletonValidationError("wrong_kind", ("document", "kind"), "Agent preparation requires agent kind")
    if type(_subagent_depth) is not int or _subagent_depth < 0 or _subagent_depth > 6:
        raise SkeletonValidationError("subagent_depth", ("document",), "subagent graph depth exceeds 6")
    from .hooks_config import validate_hooks_pool_config
    from .subagent_config import validate_subagent_pool_config
    for index, node in enumerate(snapshot["nodes"]):
        if node["type"] == "agent":
            if set(node["config"]) != {"execution", "kernel"}:
                raise SkeletonValidationError("invalid_shape", ("document", "nodes", index, "config"), "Agent node requires execution and kernel configuration")
            validate_kernel_config(node["config"]["kernel"], ("document", "nodes", index, "config", "kernel"))
        elif node["type"] == "subagent_pool":
            def validate_embedded(child, child_path):
                if _subagent_depth >= 6:
                    raise SkeletonValidationError("subagent_depth", child_path, "subagent graph depth exceeds 6")
                try:
                    prepare_agent_graph(child, _subagent_depth=_subagent_depth + 1)
                except SkeletonValidationError as exc:
                    raise SkeletonValidationError(exc.code, child_path + exc.path[1:], str(exc)) from exc
            validate_subagent_pool_config(
                node["config"],
                ("document", "nodes", index, "config"),
                validate_embedded=validate_embedded,
            )
        elif node["type"] == "hooks_pool":
            validate_hooks_pool_config(
                node["config"],
                ("document", "nodes", index, "config"),
            )
    node_ids = tuple(node["id"] for node in snapshot["nodes"])
    order = {node_id: index for index, node_id in enumerate(node_ids)}
    flow = [edge for edge in snapshot["edges"] if edge["kind"] == "flow"]
    outgoing = {node_id: [] for node_id in node_ids}
    indegree = dict.fromkeys(node_ids, 0)
    for edge in flow:
        outgoing[edge["source_node_id"]].append(edge["target_node_id"])
        indegree[edge["target_node_id"]] += 1
    available = [key for key in node_ids if indegree[key] == 0]
    topological = []
    while available:
        available.sort(key=order.__getitem__)
        current = available.pop(0)
        topological.append(current)
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                available.append(target)
    return PreparedAgentPlan(snapshot["id"], node_ids, tuple(topological),
        tuple(PreparedFlowEdge(e["id"], e["source_node_id"], e["source_port_id"], e["target_node_id"], e["target_port_id"]) for e in flow),
        _freeze(snapshot))


def checked_execution_plan(plan):
    if type(plan) is not PreparedAgentPlan:
        return _checked_plan(plan)
    rebuilt = prepare_agent_graph(plan.snapshot())
    if rebuilt != plan:
        raise SkeletonValidationError("invalid_plan", ("plan",), "Agent plan differs from its fixed definition")
    return rebuilt


def execution_definition_digest(plan):
    if type(plan) is not PreparedAgentPlan:
        return workflow_definition_digest(plan)
    snapshot = checked_execution_plan(plan).snapshot()
    return sha256(_canonical(snapshot, ("document",))).hexdigest()
