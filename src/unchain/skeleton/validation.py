"""Provider-free structural draft diagnostics and execution preflight."""

from __future__ import annotations

from collections import deque
from typing import Any

from .diagnostics import SkeletonDiagnostic
from .errors import SkeletonValidationError
from .schema import validate_skeleton_document

_POOLS = {"toolkit_pool", "subagent_pool", "hooks_pool"}


def _diagnostic(items: list[SkeletonDiagnostic], code: str, path: tuple[str | int, ...], message: str) -> None:
    items.append(SkeletonDiagnostic(code, path, message))


def _walk(starts: set[str], adjacency: dict[str, list[str]]) -> set[str]:
    visited = set(starts)
    queue = deque(starts)
    while queue:
        current = queue.popleft()
        for target in adjacency.get(current, []):
            if target not in visited:
                visited.add(target)
                queue.append(target)
    return visited


def _diagnose(document: Any) -> tuple[dict[str, Any], tuple[SkeletonDiagnostic, ...]]:
    skeleton = validate_skeleton_document(document)
    nodes = skeleton["nodes"]
    edges = skeleton["edges"]
    diagnostics: list[SkeletonDiagnostic] = []
    node_diagnostics: list[list[SkeletonDiagnostic]] = [[] for _ in nodes]
    edge_diagnostics: list[list[SkeletonDiagnostic]] = [[] for _ in edges]
    node_by_id = {node["id"]: (index, node) for index, node in enumerate(nodes)}
    starts = [(index, node) for index, node in enumerate(nodes) if node["type"] == "start"]
    ends = [(index, node) for index, node in enumerate(nodes) if node["type"] == "end"]
    agents = [(index, node) for index, node in enumerate(nodes) if node["type"] == "agent"]

    if len(starts) != 1:
        _diagnostic(diagnostics, "start_count", ("document", "nodes"), "graph requires exactly one start node")
    if len(ends) != 1:
        _diagnostic(diagnostics, "end_count", ("document", "nodes"), "graph requires exactly one end node")
    if skeleton["kind"] == "agent" and not agents:
        _diagnostic(diagnostics, "agent_required", ("document", "nodes"), "agent graph requires an agent node")

    valid_flow: list[tuple[int, dict[str, Any]]] = []
    pool_attached: set[str] = set()
    outgoing_ports: set[tuple[str, str]] = set()
    for edge_index, edge in enumerate(edges):
        path = ("document", "edges", edge_index)
        source_type = node_by_id[edge["source_node_id"]][1]["type"]
        target_type = node_by_id[edge["target_node_id"]][1]["type"]
        if edge["kind"] == "attach":
            source_port_ok = edge["source_port_id"] in {"attach_top", "attach_bot"}
            target_port_ok = edge["target_port_id"] in {"attach_top", "attach_bot"}
            ports_ok = source_port_ok and target_port_ok
            endpoints_ok = (source_type == "agent" and target_type in _POOLS) or (
                target_type == "agent" and source_type in _POOLS
            )
            if not source_port_ok:
                _diagnostic(edge_diagnostics[edge_index], "attach_ports", path + ("source_port_id",), "attach port must be attach_top or attach_bot")
            if not target_port_ok:
                _diagnostic(edge_diagnostics[edge_index], "attach_ports", path + ("target_port_id",), "attach port must be attach_top or attach_bot")
            if not endpoints_ok:
                _diagnostic(edge_diagnostics[edge_index], "attach_endpoints", path, "attach connects an agent and a pool")
            if ports_ok and endpoints_ok:
                pool_attached.add(edge["target_node_id"] if target_type in _POOLS else edge["source_node_id"])
            continue

        edge_ok = True
        if source_type in _POOLS or target_type in _POOLS:
            _diagnostic(edge_diagnostics[edge_index], "flow_pool", path, "pools cannot participate in flow edges")
            edge_ok = False
        if source_type == "end":
            _diagnostic(edge_diagnostics[edge_index], "flow_end_source", path + ("source_node_id",), "end cannot source a flow edge")
            edge_ok = False
        source_port = edge["source_port_id"]
        if source_type == "if_else":
            port_ok = source_port in {"true", "false"}
        elif source_type == "switch_case":
            port_ok = bool(source_port)
        else:
            port_ok = source_port == "out"
        if not port_ok:
            _diagnostic(edge_diagnostics[edge_index], "flow_source_port", path + ("source_port_id",), "flow source port is invalid for node type")
            edge_ok = False
        if target_type == "start":
            _diagnostic(edge_diagnostics[edge_index], "flow_start_target", path + ("target_node_id",), "start cannot target a flow edge")
            edge_ok = False
        if edge["target_port_id"] != "in":
            _diagnostic(edge_diagnostics[edge_index], "flow_target_port", path + ("target_port_id",), "flow target port must be in")
            edge_ok = False
        if edge_ok:
            key = (edge["source_node_id"], edge["source_port_id"])
            if key in outgoing_ports:
                _diagnostic(edge_diagnostics[edge_index], "duplicate_flow_outlet", ("document", "edges", edge_index, "source_port_id"), "flow outlet has multiple edges")
            outgoing_ports.add(key)
            valid_flow.append((edge_index, edge))

    for index, node in enumerate(nodes):
        if node["type"] in _POOLS and node["id"] not in pool_attached:
            _diagnostic(node_diagnostics[index], "pool_attachment_required", ("document", "nodes", index), "pool must attach to an agent")

    outgoing: dict[str, list[str]] = {node_id: [] for node_id in node_by_id}
    incoming: dict[str, list[str]] = {node_id: [] for node_id in node_by_id}
    ports_by_source: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    for _, edge in valid_flow:
        source = edge["source_node_id"]
        target = edge["target_node_id"]
        outgoing[source].append(target)
        incoming[target].append(source)
        ports_by_source[source].add(edge["source_port_id"])

    for index, node in enumerate(nodes):
        node_id, node_type = node["id"], node["type"]
        if node_type in _POOLS:
            continue
        path = ("document", "nodes", index)
        if node_type == "start" and incoming[node_id]:
            _diagnostic(node_diagnostics[index], "start_flow_input", path, "start cannot have flow input")
        if node_type == "end" and outgoing[node_id]:
            _diagnostic(node_diagnostics[index], "end_flow_output", path, "end cannot have flow output")
        if node_type != "start" and not incoming[node_id]:
            _diagnostic(node_diagnostics[index], "flow_input_required", path, "execution node requires flow input")
        if node_type != "end" and not outgoing[node_id]:
            _diagnostic(node_diagnostics[index], "flow_output_required", path, "execution node requires flow output")
        if node_type == "if_else":
            for outlet in ("true", "false"):
                if outlet not in ports_by_source[node_id]:
                    _diagnostic(node_diagnostics[index], "if_outlet_required", path, f"if_else requires {outlet} outlet")
        if node_type == "switch_case" and "default" not in ports_by_source[node_id]:
            _diagnostic(node_diagnostics[index], "switch_default_required", path, "switch_case requires default outlet")

    execution_ids = [node["id"] for node in nodes if node["type"] not in _POOLS]
    if len(starts) == 1:
        from_start = _walk({starts[0][1]["id"]}, outgoing)
        for index, node in enumerate(nodes):
            if node["type"] not in _POOLS and node["id"] not in from_start:
                _diagnostic(node_diagnostics[index], "unreachable_node", ("document", "nodes", index), "execution node is unreachable from start")
    if len(ends) == 1:
        reverse = {node_id: [] for node_id in node_by_id}
        for source, targets in outgoing.items():
            for target in targets:
                reverse[target].append(source)
        to_end = _walk({ends[0][1]["id"]}, reverse)
        for index, node in enumerate(nodes):
            if node["type"] not in _POOLS and node["id"] not in to_end:
                _diagnostic(node_diagnostics[index], "cannot_reach_end", ("document", "nodes", index), "execution node cannot reach end")

    reverse_flow = {node_id: [] for node_id in execution_ids}
    for source in execution_ids:
        for target in outgoing[source]:
            reverse_flow[target].append(source)
    visited: set[str] = set()
    finish_order: list[str] = []
    for root in execution_ids:
        if root in visited:
            continue
        visited.add(root)
        stack: list[list[Any]] = [[root, 0]]
        while stack:
            current, next_index = stack[-1]
            if next_index == len(outgoing[current]):
                finish_order.append(current)
                stack.pop()
                continue
            target = outgoing[current][next_index]
            stack[-1][1] += 1
            if target not in visited:
                visited.add(target)
                stack.append([target, 0])

    assigned: set[str] = set()
    cycle_nodes: set[str] = set()
    for root in reversed(finish_order):
        if root in assigned:
            continue
        assigned.add(root)
        component: list[str] = []
        stack = [root]
        while stack:
            current = stack.pop()
            component.append(current)
            for target in reverse_flow[current]:
                if target not in assigned:
                    assigned.add(target)
                    stack.append(target)
        if len(component) > 1 or component[0] in outgoing[component[0]]:
            cycle_nodes.update(component)
    for index, node in enumerate(nodes):
        if node["id"] in cycle_nodes:
            _diagnostic(node_diagnostics[index], "flow_cycle", ("document", "nodes", index), "flow graph contains a cycle")
    return skeleton, tuple(diagnostics + [item for group in node_diagnostics for item in group] + [item for group in edge_diagnostics for item in group])


def diagnose_skeleton_draft(document: Any) -> tuple[SkeletonDiagnostic, ...]:
    """Diagnose a draft and its embedded Agents without resolving resources.

    Only the outer envelope is a hard admission gate. An invalid embedded
    envelope belongs to a repairable parent draft and is reported at its own
    occurrence, just like its structural and IO problems.
    """
    return _diagnose_draft_tree(document, depth=0)


def _diagnose_draft_tree(document: Any, *, depth: int) -> tuple[SkeletonDiagnostic, ...]:
    from .draft_io import diagnose_execution_io

    skeleton, structural = _diagnose(document)
    diagnostics = list(structural + diagnose_execution_io(skeleton))
    if depth and skeleton["kind"] != "agent":
        _diagnostic(diagnostics, "wrong_kind", ("document", "kind"),
                    "Agent preparation requires agent kind")

    for node_index, node in enumerate(skeleton["nodes"]):
        if node["type"] == "hooks_pool" and skeleton["kind"] == "agent":
            hooks = node["config"].get("hooks")
            for hook_index, hook in enumerate(hooks if type(hooks) is list else []):
                workflow = hook.get("workflow") if type(hook) is dict else None
                if type(workflow) is not dict or workflow.get("kind") != "embedded":
                    continue
                prefix = ("document", "nodes", node_index, "config", "hooks",
                          hook_index, "workflow", "document")
                if "document" not in workflow:
                    _diagnostic(diagnostics, "missing_field", prefix, "required field is missing")
                    continue
                body = workflow["document"]
                try:
                    if type(body) is dict and body.get("kind") != "workflow":
                        raise SkeletonValidationError("wrong_kind", ("document", "kind"),
                                                      "embedded Hook requires Workflow kind")
                    child_diagnostics = _diagnose_draft_tree(body, depth=0)
                except SkeletonValidationError as exc:
                    child_diagnostics = (SkeletonDiagnostic(exc.code, exc.path, str(exc)),)
                diagnostics.extend(SkeletonDiagnostic(item.code, prefix + item.path[1:], item.message)
                                   for item in child_diagnostics)
        if node["type"] != "subagent_pool":
            continue
        entries = node["config"].get("subagents")
        if type(entries) is not list:
            continue
        for entry_index, entry in enumerate(entries):
            source = entry.get("source") if type(entry) is dict else None
            if type(source) is not dict or source.get("kind") != "embedded":
                continue
            prefix = ("document", "nodes", node_index, "config", "subagents",
                      entry_index, "source", "document")
            if depth >= 6:
                _diagnostic(diagnostics, "subagent_depth", prefix,
                            "subagent graph depth exceeds 6")
                continue
            if "document" not in source:
                _diagnostic(diagnostics, "missing_field", prefix,
                            "required field is missing")
                continue
            try:
                child_diagnostics = _diagnose_draft_tree(source["document"], depth=depth + 1)
            except SkeletonValidationError as exc:
                child_diagnostics = (SkeletonDiagnostic(exc.code, exc.path, str(exc)),)
            # Each occurrence has its own scope, even when two entries share a
            # document id or the caller supplied the same Python object twice.
            diagnostics.extend(SkeletonDiagnostic(item.code, prefix + item.path[1:], item.message)
                               for item in child_diagnostics)

    def position(item: SkeletonDiagnostic) -> tuple[int, int]:
        if len(item.path) > 2 and type(item.path[2]) is int:
            if item.path[1] == "nodes":
                return 1, item.path[2]
            if item.path[1] == "edges":
                return 2, item.path[2]
        return 0, 0

    return tuple(sorted(diagnostics, key=position))


def validate_skeleton_for_execution(document: Any) -> dict[str, Any]:
    """Reject the first draft diagnostic or return a detached schema document."""
    skeleton, diagnostics = _diagnose(document)
    if diagnostics:
        first = diagnostics[0]
        raise SkeletonValidationError(first.code, first.path, first.message)
    return skeleton
