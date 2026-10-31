"""Provider-free execution of one prepared Skeleton Workflow path."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .conditions import _validate_condition, _validate_operand, evaluate_condition, select_case
from .errors import SkeletonValidationError
from .io import _MISSING, _closed_object, _read_path, _validate_expression, _validate_json, _validate_path, map_outputs, resolve_inputs
from .preparation import PreparedWorkflowPlan, prepare_workflow


class WorkflowExecutionError(SkeletonValidationError):
    """A closed Workflow execution failure."""


@dataclass(frozen=True)
class WorkflowNodeContext:
    document_id: str
    node_id: str
    node_type: str


@dataclass(frozen=True)
class WorkflowNodeCompleted:
    result: dict[str, Any]
    selected_port: str = "out"


@dataclass(frozen=True)
class WorkflowRunResult:
    outputs: dict[str, Any]
    completed_node_ids: tuple[str, ...]
    selected_edge_ids: tuple[str, ...]


_HANDLERS = {"code", "tool_call", "skill_read", "workflow_call"}


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise WorkflowExecutionError(code, path, message)


def _clone(value: Any) -> Any:
    if type(value) is dict:
        return {key: _clone(child) for key, child in value.items()}
    if type(value) is list:
        return [_clone(child) for child in value]
    return value


def _mapped(operation: Callable[[], Any], roots: dict[str, tuple[str | int, ...]]) -> Any:
    """Run a helper and translate its local error roots into public runner paths."""
    try:
        return operation()
    except SkeletonValidationError as error:
        path = error.path
        if path and type(path[0]) is str and path[0] in roots:
            path = roots[path[0]] + path[1:]
        raise WorkflowExecutionError(error.code, path, str(error)) from error


def _resolve_node_inputs(
    bindings: list[dict[str, Any]],
    upstream_outputs: dict[str, Any],
    config_path: tuple[str | int, ...],
    runtime_path: tuple[str | int, ...],
) -> dict[str, Any]:
    return _mapped(
        lambda: resolve_inputs(bindings, upstream_outputs),
        {
            "bindings": config_path + ("inputs",),
            "outputs": runtime_path + ("upstream_outputs",),
            "inputs": runtime_path + ("resolved_inputs",),
        },
    )


def _map_node_outputs(
    mappings: list[dict[str, Any]],
    inputs: dict[str, Any],
    result: dict[str, Any],
    config_path: tuple[str | int, ...],
    runtime_path: tuple[str | int, ...],
) -> dict[str, Any]:
    return _mapped(
        lambda: map_outputs(mappings, inputs, result),
        {
            "mappings": config_path + ("outputs",),
            "inputs": runtime_path + ("resolved_inputs",),
            "result": runtime_path + ("result",),
            "outputs": runtime_path + ("outputs",),
        },
    )


def _checked(plan: Any) -> PreparedWorkflowPlan:
    if type(plan) is not PreparedWorkflowPlan:
        _fail("invalid_plan", ("plan",), "plan must be a PreparedWorkflowPlan")
    try:
        rebuilt = prepare_workflow(plan.snapshot())
        if (plan.document_id, plan.node_ids, plan.topological_node_ids, plan.flow_edges) != (rebuilt.document_id, rebuilt.node_ids, rebuilt.topological_node_ids, rebuilt.flow_edges):
            _fail("invalid_plan", ("plan",), "plan does not match its snapshot")
        return rebuilt
    except (SkeletonValidationError, ValueError, TypeError, AttributeError, IndexError, KeyError, RecursionError) as error:
        if isinstance(error, WorkflowExecutionError):
            raise
        _fail("invalid_plan", ("plan",), "plan snapshot is not a prepared workflow")


def _execution(node: dict[str, Any], path: tuple[str | int, ...]) -> dict[str, Any]:
    config = node["config"]
    if type(config) is not dict or "execution" not in config:
        _fail("missing_field", path + ("config", "execution"), "execution config is required")
    value = config["execution"]
    _validate_json(value, path + ("config", "execution"))
    if type(value) is not dict:
        _fail("invalid_type", path + ("config", "execution"), "execution config must be an object")
    return value


def _validate_bindings(value: Any, path: tuple[str | int, ...]) -> None:
    if type(value) is not list:
        _fail("invalid_type", path, "inputs must be an array")
    names: set[str] = set()
    for index, item in enumerate(value):
        item_path = path + (index,)
        obj = _closed_object(item, item_path, {"name", "value", "missing"}, {"default"})
        if type(obj["name"]) is not str:
            _fail("invalid_type", item_path + ("name",), "binding name must be a string")
        if not obj["name"] or obj["name"] in names:
            _fail("invalid_value" if obj["name"] not in names else "duplicate_name", item_path + ("name",), "binding name is invalid")
        names.add(obj["name"])
        if type(obj["missing"]) is not str:
            _fail("invalid_type", item_path + ("missing",), "missing policy must be a string")
        if obj["missing"] not in {"error", "null", "default"}:
            _fail("invalid_value", item_path + ("missing",), "unknown missing policy")
        if obj["missing"] == "default" and "default" not in obj:
            _fail("missing_field", item_path + ("default",), "default policy requires default")
        if obj["missing"] != "default" and "default" in obj:
            _fail("unknown_field", item_path + ("default",), "default is only allowed for default policy")
        if "default" in obj: _validate_json(obj["default"], item_path + ("default",))
        _validate_expression(obj["value"], item_path + ("value",))


def _validate_outputs(value: Any, path: tuple[str | int, ...]) -> None:
    if type(value) is not list:
        _fail("invalid_type", path, "outputs must be an array")
    names: set[str] = set()
    for index, item in enumerate(value):
        obj = _closed_object(item, path + (index,), {"name", "source"})
        if type(obj["name"]) is not str:
            _fail("invalid_type", path + (index, "name"), "output name must be a string")
        if not obj["name"] or obj["name"] in names:
            _fail("invalid_value" if obj["name"] not in names else "duplicate_name", path + (index, "name"), "output name is invalid")
        names.add(obj["name"])
        source = _closed_object(obj["source"], path + (index, "source"), {"kind", "path"})
        if type(source["kind"]) is not str:
            _fail("invalid_type", path + (index, "source", "kind"), "output source kind must be a string")
        if source["kind"] not in {"input", "result"}:
            _fail("unknown_variant", path + (index, "source", "kind"), "output source is invalid")
        _validate_path(source["path"], path + (index, "source", "path"))


def _validate_all(snapshot: dict[str, Any], *, agent_graph: bool = False) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(snapshot["nodes"]):
        path = ("document", "nodes", index)
        node_type = node["type"]
        if node_type == "tool_call":
            config = _closed_object(node["config"], path + ("config",), {"execution", "tool"})
            tool = _closed_object(
                config["tool"], path + ("config", "tool"),
                {"name", "on_error", "approval_timeout_ms"},
            )
            if type(tool["name"]) is not str or not tool["name"]:
                _fail("invalid_value", path + ("config", "tool", "name"), "tool name must be non-empty")
            if type(tool["on_error"]) is not str or tool["on_error"] not in {"stop", "continue"}:
                _fail("unknown_variant", path + ("config", "tool", "on_error"), "tool error policy is invalid")
            if (type(tool["approval_timeout_ms"]) is not int or isinstance(tool["approval_timeout_ms"], bool)
                    or not 1 <= tool["approval_timeout_ms"] <= 86_400_000):
                _fail("invalid_value", path + ("config", "tool", "approval_timeout_ms"), "tool approval timeout is invalid")
        elif node_type == "skill_read":
            config = _closed_object(node["config"], path + ("config",), {"execution", "skill"})
            skill = _closed_object(
                config["skill"], path + ("config", "skill"),
                {"source", "source_id", "name"},
            )
            for key in ("source", "source_id", "name"):
                if type(skill[key]) is not str or not skill[key]:
                    _fail("invalid_value", path + ("config", "skill", key), "skill identity field must be non-empty")
        elif node_type == "code" and "code" in node["config"]:
            config = _closed_object(node["config"], path + ("config",), {"execution", "code"})
            from .code_runtime import admit_code_config
            admit_code_config(config["code"], path + ("config", "code"))
        elif node_type == "workflow_call":
            config = _closed_object(node["config"], path + ("config",), {"execution", "workflow"})
            workflow = _closed_object(
                config["workflow"], path + ("config", "workflow"),
                {"workflow_id", "revision_id"},
            )
            for key in ("workflow_id", "revision_id"):
                if type(workflow[key]) is not str or not workflow[key]:
                    _fail(
                        "invalid_value",
                        path + ("config", "workflow", key),
                        "Workflow Call identity must be non-empty",
                    )
        if agent_graph and node["type"] in {"toolkit_pool", "subagent_pool", "hooks_pool"}:
            continue
        cfg = _execution(node, path); node_type = node["type"]
        required = ({"outputs"} if node_type == "start" else {"inputs", "outputs"} if node_type in _HANDLERS or node_type == "end" or (agent_graph and node_type == "agent") else {"inputs", "condition"} if node_type == "if_else" else {"inputs", "value", "cases"} if node_type == "switch_case" else None)
        if required is None:
            _fail("unknown_variant", path + ("type",), "node type is not executable in WorkflowRunner")
        if set(cfg) != required:
            _fail("invalid_shape", path + ("config", "execution"), "execution config fields are invalid")
        if "inputs" in cfg: _validate_bindings(cfg["inputs"], path + ("config", "execution", "inputs"))
        if "outputs" in cfg: _validate_outputs(cfg["outputs"], path + ("config", "execution", "outputs"))
        if node_type == "end":
            for output_index, output in enumerate(cfg["outputs"]):
                if output["source"]["kind"] != "input":
                    _fail("invalid_value", path + ("config", "execution", "outputs", output_index, "source", "kind"), "end outputs may only use inputs")
        if node_type == "if_else": _validate_condition(cfg["condition"], path + ("config", "execution", "condition"))
        if node_type == "switch_case":
            _validate_operand(cfg["value"], path + ("config", "execution", "value"))
            _mapped(
                lambda: select_case(None, cfg["cases"]),
                {"cases": path + ("config", "execution", "cases"), "value": path + ("config", "execution", "value")},
            )
        configs[node["id"]] = cfg
    # Keep Kernel/config/grammar error precedence; inspect values only afterward.
    from .static_io import validate_static_execution_io
    validate_static_execution_io(snapshot)
    return configs


def _value(operand: dict[str, Any], inputs: dict[str, Any], path: tuple[str | int, ...]) -> Any:
    if operand["kind"] == "literal": return operand["value"]
    value = _read_path(inputs, operand["path"])
    if value is _MISSING: _fail("missing_value", path + ("path",), "switch input is unavailable")
    return value


def run_workflow(plan: PreparedWorkflowPlan, inputs: Any, handlers: Any) -> WorkflowRunResult:
    """Run one exclusive Workflow path using only injected deterministic handlers."""
    try:
        checked = _checked(plan); _validate_json(inputs, ("inputs",))
        if type(inputs) is not dict: _fail("invalid_type", ("inputs",), "inputs must be an object")
        if type(handlers) is not dict: _fail("invalid_type", ("handlers",), "handlers must be an object")
        snapshot = checked.snapshot(); configs = _validate_all(snapshot)
        nodes = {node["id"]: node for node in snapshot["nodes"]}
        node_positions = {node["id"]: index for index, node in enumerate(snapshot["nodes"])}
        start = next(node["id"] for node in snapshot["nodes"] if node["type"] == "start")
        edges: dict[tuple[str, str], tuple[str, str]] = {}
        for edge in checked.flow_edges:
            key = (edge.source_node_id, edge.source_port_id)
            if key in edges: _fail("duplicate_name", ("document", "edges"), "flow outlet is duplicated")
            edges[key] = (edge.id, edge.target_node_id)
        for node_id, cfg in configs.items():
            node_type = nodes[node_id]["type"]
            node_path = ("document", "nodes", node_positions[node_id], "config", "execution")
            actual = {port for source, port in edges if source == node_id}
            if node_type == "if_else" and actual != {"true", "false"}:
                _fail("invalid_value", node_path, "if outlets must be true and false")
            if node_type == "switch_case":
                declared = {case["id"] for case in cfg["cases"]} | {"default"}
                if actual != declared:
                    _fail("invalid_value", node_path + ("cases",), "switch outlets must match declared cases and default")
        exports: dict[str, Any] = {}; completed: list[str] = []; selected: list[str] = []; current = start
        while True:
            node = nodes[current]; kind = node["type"]; cfg = configs[current]; resolved: dict[str, Any] = {}
            node_index = node_positions[current]
            config_path = ("document", "nodes", node_index, "config", "execution")
            runtime_path = ("execution", "nodes", node_index)
            if kind == "start":
                result = inputs; port = "out"
                exported = _map_node_outputs(cfg["outputs"], {}, result, config_path, runtime_path)
            else:
                resolved = _resolve_node_inputs(cfg["inputs"], exports, config_path, runtime_path)
                if kind == "end":
                    exported = _map_node_outputs(cfg["outputs"], resolved, {}, config_path, runtime_path)
                    completed.append(current)
                    return WorkflowRunResult(_clone(exported), tuple(completed), tuple(selected))
                if kind == "if_else":
                    condition = _mapped(
                        lambda: evaluate_condition(cfg["condition"], resolved),
                        {"condition": config_path + ("condition",)},
                    )
                    port = "true" if condition else "false"; exported = {}
                elif kind == "switch_case":
                    value = _value(cfg["value"], resolved, config_path + ("value",))
                    port = _mapped(
                        lambda: select_case(value, cfg["cases"]),
                        {"cases": config_path + ("cases",), "value": config_path + ("value",)},
                    )
                    exported = {}
                else:
                    handler = handlers.get(kind)
                    if not callable(handler): _fail("missing_value", runtime_path + ("handler",), "handler is required")
                    try: completion = handler(WorkflowNodeContext(checked.document_id, current, kind), _clone(resolved))
                    except Exception as error: _fail("handler_error", runtime_path + ("handler",), "handler failed")
                    if type(completion) is not WorkflowNodeCompleted: _fail("invalid_type", runtime_path + ("result",), "handler must return WorkflowNodeCompleted")
                    _validate_json(completion.result, runtime_path + ("result",))
                    if type(completion.result) is not dict: _fail("invalid_type", runtime_path + ("result",), "handler result must be an object")
                    if completion.selected_port != "out": _fail("invalid_value", runtime_path + ("selected_port",), "handler port must be out")
                    port = "out"
                    exported = _map_node_outputs(cfg["outputs"], resolved, completion.result, config_path, runtime_path)
            completed.append(current); exports[current] = exported
            edge = edges.get((current, port))
            if edge is None: _fail("missing_value", runtime_path + ("selected_port",), "selected outlet is unavailable")
            selected.append(edge[0]); current = edge[1]
    except SkeletonValidationError as error:
        if isinstance(error, WorkflowExecutionError): raise
        raise WorkflowExecutionError(error.code, error.path, str(error)) from error
