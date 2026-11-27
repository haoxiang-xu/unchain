"""Strict, provider-free input and output mapping helpers."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .errors import SkeletonValidationError

_MAX_DEPTH = 64
_MAX_NODES = 100_000
_MISSING = object()


def _error(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _expect_object(value: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    if type(value) is not dict:
        _error("invalid_type", path, "expected an object")
    return value


def _closed_object(
    value: Any, path: tuple[str | int, ...], required: set[str], optional: set[str] = set()
) -> dict[str, Any]:
    obj = _expect_object(value, path)
    for key in obj:
        if type(key) is not str:
            _error("invalid_json", path, "object keys must be strings")
        if key not in required | optional:
            _error("unknown_field", path + (key,), "unknown field")
    for key in sorted(required):
        if key not in obj:
            _error("missing_field", path + (key,), "required field is missing")
    return obj


def _validate_json(value: Any, path: tuple[str | int, ...]) -> None:
    """Validate JSON without recursively walking caller-supplied values."""
    stack = [(value, path, 0)]
    count = 0
    while stack:
        current, current_path, depth = stack.pop()
        count += 1
        if count > _MAX_NODES:
            _error("limit_nodes", current_path, "JSON value exceeds the node limit")
        if depth > _MAX_DEPTH:
            _error("limit_depth", current_path, "JSON value exceeds the depth limit")
        if current is None or type(current) in (str, bool):
            continue
        if type(current) is int:
            continue
        if type(current) is float:
            if not math.isfinite(current):
                _error("invalid_json", current_path, "JSON number must be finite")
            continue
        if type(current) is list:
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], current_path + (index,), depth + 1))
            continue
        if type(current) is dict:
            for key, child in current.items():
                if type(key) is not str:
                    _error("invalid_json", current_path, "object keys must be strings")
                stack.append((child, current_path + (key,), depth + 1))
            continue
        _error("invalid_json", current_path, "value is not JSON")


def _detach_document(value: dict[str, Any], path: tuple[str | int, ...]) -> dict[str, Any]:
    """Admit a whole public document before independently copying its fields."""
    _validate_json(value, path)
    # A separate deepcopy per field deliberately prevents repeated source
    # values from sharing mutable state in the returned public document.
    return {name: deepcopy(field) for name, field in value.items()}


def _validate_path(value: Any, path: tuple[str | int, ...]) -> list[str | int]:
    if type(value) is not list:
        _error("invalid_type", path, "path must be an array")
    result: list[str | int] = []
    for index, part in enumerate(value):
        item_path = path + (index,)
        if type(part) is str:
            result.append(part)
        elif type(part) is int and part >= 0:
            result.append(part)
        else:
            _error("invalid_path", item_path, "path item must be a string or non-negative integer")
    return result


def _read_path(value: Any, path: list[str | int]) -> Any:
    current = value
    for part in path:
        if isinstance(part, str):
            if not isinstance(current, dict) or part not in current:
                return _MISSING
        else:
            if not isinstance(current, list) or part >= len(current):
                return _MISSING
        current = current[part]
    return current


def _validate_expression(value: Any, path: tuple[str | int, ...], depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        _error("limit_depth", path, "expression exceeds the depth limit")
    obj = _expect_object(value, path)
    if "kind" not in obj:
        _error("missing_field", path + ("kind",), "required field is missing")
    kind = obj["kind"]
    if type(kind) is not str:
        _error("invalid_type", path + ("kind",), "kind must be a string")
    if kind == "literal":
        _closed_object(obj, path, {"kind", "value"})
        _validate_json(obj["value"], path + ("value",))
    elif kind == "ref":
        _closed_object(obj, path, {"kind", "node_id", "path"})
        if type(obj["node_id"]) is not str or not obj["node_id"]:
            _error("invalid_value", path + ("node_id",), "node_id must be a non-empty string")
        _validate_path(obj["path"], path + ("path",))
    elif kind == "first_present":
        _closed_object(obj, path, {"kind", "values"})
        values = obj["values"]
        if type(values) is not list or len(values) < 2:
            _error("invalid_value", path + ("values",), "first_present requires at least two values")
        for index, expression in enumerate(values):
            _validate_expression(expression, path + ("values", index), depth + 1)
    else:
        _error("unknown_variant", path + ("kind",), "unknown value expression")


def _evaluate_expression(expression: dict[str, Any], outputs: dict[str, Any]) -> Any:
    kind = expression["kind"]
    if kind == "literal":
        return expression["value"]
    if kind == "ref":
        if expression["node_id"] not in outputs:
            return _MISSING
        return _read_path(outputs[expression["node_id"]], expression["path"])
    for candidate in expression["values"]:
        value = _evaluate_expression(candidate, outputs)
        if value is not _MISSING:
            return value
    return _MISSING


def resolve_inputs(bindings: Any, outputs: Any) -> dict[str, Any]:
    """Resolve closed bindings against completed node exports."""
    _validate_json(bindings, ("bindings",))
    if type(bindings) is not list:
        _error("invalid_type", ("bindings",), "bindings must be an array")
    _validate_json(outputs, ("outputs",))
    if type(outputs) is not dict:
        _error("invalid_type", ("outputs",), "outputs must be an object")
    resolved: dict[str, Any] = {}
    seen: set[str] = set()
    parsed: list[tuple[int, dict[str, Any]]] = []
    for index, binding in enumerate(bindings):
        path = ("bindings", index)
        obj = _closed_object(binding, path, {"name", "value", "missing"}, {"default"})
        if type(obj["name"]) is not str or not obj["name"]:
            _error("invalid_value", path + ("name",), "name must be a non-empty string")
        if obj["name"] in seen:
            _error("duplicate_name", path + ("name",), "binding name is duplicated")
        seen.add(obj["name"])
        if type(obj["missing"]) is not str:
            _error("invalid_type", path + ("missing",), "missing must be a string")
        if obj["missing"] not in {"error", "null", "default"}:
            _error("invalid_value", path + ("missing",), "unknown missing policy")
        if obj["missing"] == "default":
            if "default" not in obj:
                _error("missing_field", path + ("default",), "default policy requires default")
            _validate_json(obj["default"], path + ("default",))
        elif "default" in obj:
            _error("unknown_field", path + ("default",), "default is only allowed for default policy")
        _validate_expression(obj["value"], path + ("value",))
        parsed.append((index, obj))
    for index, binding in parsed:
        value = _evaluate_expression(binding["value"], outputs)
        if value is _MISSING:
            if binding["missing"] == "error":
                _error("missing_value", ("bindings", index, "value"), "binding value is unavailable")
            value = None if binding["missing"] == "null" else binding["default"]
        resolved[binding["name"]] = value
    return _detach_document(resolved, ("inputs",))


def map_outputs(mappings: Any, inputs: Any, result: Any) -> dict[str, Any]:
    """Export selected values from a node's inputs or result."""
    _validate_json(mappings, ("mappings",))
    if type(mappings) is not list:
        _error("invalid_type", ("mappings",), "mappings must be an array")
    _validate_json(inputs, ("inputs",))
    _validate_json(result, ("result",))
    if type(inputs) is not dict:
        _error("invalid_type", ("inputs",), "inputs must be an object")
    parsed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, mapping in enumerate(mappings):
        path = ("mappings", index)
        obj = _closed_object(mapping, path, {"name", "source"})
        if type(obj["name"]) is not str or not obj["name"]:
            _error("invalid_value", path + ("name",), "name must be a non-empty string")
        if obj["name"] in seen:
            _error("duplicate_name", path + ("name",), "output name is duplicated")
        seen.add(obj["name"])
        source = _closed_object(obj["source"], path + ("source",), {"kind", "path"})
        if type(source["kind"]) is not str:
            _error("invalid_type", path + ("source", "kind"), "source kind must be a string")
        if source["kind"] not in {"input", "result"}:
            _error("unknown_variant", path + ("source", "kind"), "source must be input or result")
        _validate_path(source["path"], path + ("source", "path"))
        parsed.append(obj)
    exported: dict[str, Any] = {}
    for index, mapping in enumerate(parsed):
        source = mapping["source"]
        value = _read_path(inputs if source["kind"] == "input" else result, source["path"])
        if value is _MISSING:
            _error("missing_value", ("mappings", index, "source", "path"), "output value is unavailable")
        exported[mapping["name"]] = value
    return _detach_document(exported, ("outputs",))
