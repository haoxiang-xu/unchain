"""Static IO surfaces for schema-admitted drafts and executable snapshots."""
from __future__ import annotations

from typing import Any

from .errors import SkeletonValidationError

_BRANCHES = {"if_else", "switch_case"}
_POOLS = {"toolkit_pool", "subagent_pool", "hooks_pool"}


def _names(rows: list[Any]) -> set[str]:
    return {row["name"] for row in rows if type(row) is dict
            and type(row.get("name")) is str and row["name"]}


class StaticIOSurfaces:
    """Inspect flow ancestry and known names without guessing dynamic values."""

    def __init__(self, snapshot: dict[str, Any]):
        self.nodes = {node["id"]: node for node in snapshot["nodes"]}
        self.incoming: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in snapshot["edges"]:
            if edge["kind"] == "flow":
                self.incoming[edge["target_node_id"]].append(edge["source_node_id"])
        self.ancestors: dict[str, set[str]] = {}

    def upstream(self, node_id: str) -> set[str]:
        if node_id not in self.ancestors:
            visited = {node_id}
            pending = list(self.incoming[node_id])
            while pending:
                current = pending.pop()
                if current not in visited:
                    visited.add(current)
                    pending.extend(self.incoming[current])
            self.ancestors[node_id] = visited - {node_id}
        return self.ancestors[node_id]

    def names(self, node_id: str, section: str) -> set[str] | None:
        node = self.nodes[node_id]
        # These sections never exist, even in a fully configured node.
        if (section == "inputs" and node["type"] == "start"
                or section == "outputs" and node["type"] in _BRANCHES):
            return set()
        execution = node["config"].get("execution")
        if type(execution) is not dict or type(execution.get(section)) is not list:
            return None  # A supported surface has not been declared yet.
        return _names(execution[section])

    def reference_problem(self, value: dict[str, Any], node_id: str
                          ) -> tuple[tuple[str | int, ...], str] | None:
        source = value["node_id"]
        if (source not in self.nodes or source not in self.upstream(node_id)
                or self.nodes[source]["type"] in _POOLS):
            return ("node_id",), "reference must name an upstream flow node"
        names = self.names(source, "outputs")
        parts = value["path"]
        if parts and names is not None and parts[0] not in names:
            return ("path", 0), "reference names an undeclared node output"
        return None

    def expression_possible(self, expression: dict[str, Any], node_id: str) -> bool:
        pending = [expression]
        while pending:
            value = pending.pop()
            if value["kind"] == "literal":
                return True  # Null is present; missing is a separate sentinel.
            if value["kind"] == "first_present":
                pending.extend(value["values"])
            elif self.reference_problem(value, node_id) is None:
                return True
        return False

    def output_problem(self, source: dict[str, Any], node_id: str) -> str | None:
        if source["kind"] == "input" and source["path"]:
            names = self.names(node_id, "inputs")
            if names is not None and source["path"][0] not in names:
                return "output names an undeclared node input"
        return None


def validate_static_execution_io(snapshot: dict[str, Any]) -> None:
    """Reject provably missing required values after existing config admission."""
    surfaces = StaticIOSurfaces(snapshot)
    for index, node in enumerate(snapshot["nodes"]):
        if node["type"] in _POOLS:
            continue
        execution = node["config"]["execution"]
        path = ("document", "nodes", index, "config", "execution")
        for row_index, row in enumerate(execution.get("inputs", [])):
            if row["missing"] == "error" and not surfaces.expression_possible(row["value"], node["id"]):
                raise SkeletonValidationError("missing_value", path + ("inputs", row_index, "value"),
                                              "required input has no available static source")
        for row_index, row in enumerate(execution.get("outputs", [])):
            problem = surfaces.output_problem(row["source"], node["id"])
            if problem is not None:
                raise SkeletonValidationError("missing_value", path + ("outputs", row_index, "source", "path"),
                                              problem)

    # Preserve all existing config/grammar and static IO error priorities first.
    from .branch_admission import analyze_branch_execution

    for index, node in enumerate(snapshot["nodes"]):
        if node["type"] in _BRANCHES:
            path = ("document", "nodes", index, "config", "execution")
            rejection = analyze_branch_execution(node, surfaces, path).rejection
            if rejection is not None:
                raise SkeletonValidationError(rejection.code, rejection.path, rejection.message)
