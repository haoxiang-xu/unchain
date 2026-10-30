"""Strict condition and Switch helpers for the Skeleton protocol."""

from __future__ import annotations

from typing import Any

from .io import _MISSING, _closed_object, _error, _read_path, _validate_json, _validate_path

_UNARY_OPERATORS = {"is_null", "is_not_null"}
_BINARY_OPERATORS = {"eq", "ne", "gt", "gte", "lt", "lte", "contains"}


def _is_number(value: Any) -> bool:
    return type(value) in (int, float)


def _strict_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool-is-int equivalence."""
    if _is_number(left) and _is_number(right):
        return left == right
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return len(left) == len(right) and left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    return left == right


def _validate_operand(value: Any, path: tuple[str | int, ...]) -> None:
    obj = _closed_object(value, path, {"kind"}, {"path", "value"})
    if type(obj["kind"]) is not str:
        _error("invalid_type", path + ("kind",), "operand kind must be a string")
    if obj["kind"] == "input":
        _closed_object(obj, path, {"kind", "path"})
        _validate_path(obj["path"], path + ("path",))
    elif obj["kind"] == "literal":
        _closed_object(obj, path, {"kind", "value"})
        _validate_json(obj["value"], path + ("value",))
    else:
        _error("unknown_variant", path + ("kind",), "unknown operand")


def _validate_condition(value: Any, path: tuple[str | int, ...], depth: int = 0) -> None:
    if depth > 64:
        _error("limit_depth", path, "condition exceeds the depth limit")
    obj = _closed_object(value, path, {"kind"}, {"conditions", "operator", "left", "right"})
    if type(obj["kind"]) is not str:
        _error("invalid_type", path + ("kind",), "condition kind must be a string")
    if obj["kind"] in {"all", "any"}:
        _closed_object(obj, path, {"kind", "conditions"})
        conditions = obj["conditions"]
        if type(conditions) is not list or not conditions:
            _error("invalid_value", path + ("conditions",), "condition group must be non-empty")
        for index, child in enumerate(conditions):
            _validate_condition(child, path + ("conditions", index), depth + 1)
        return
    if obj["kind"] != "compare":
        _error("unknown_variant", path + ("kind",), "unknown condition")
    _closed_object(obj, path, {"kind", "operator", "left"}, {"right"})
    if type(obj["operator"]) is not str:
        _error("invalid_type", path + ("operator",), "operator must be a string")
    operator = obj["operator"]
    if operator in _UNARY_OPERATORS:
        if "right" in obj:
            _error("unknown_field", path + ("right",), "unary operator forbids right")
    elif operator in _BINARY_OPERATORS:
        if "right" not in obj:
            _error("missing_field", path + ("right",), "binary operator requires right")
    else:
        _error("unknown_variant", path + ("operator",), "unknown comparison operator")
    _validate_operand(obj["left"], path + ("left",))
    if "right" in obj:
        _validate_operand(obj["right"], path + ("right",))


def _evaluate_operand(operand: dict[str, Any], inputs: dict[str, Any], path: tuple[str | int, ...]) -> tuple[Any, tuple[str | int, ...]]:
    if operand["kind"] == "literal":
        return operand["value"], path + ("value",)
    return _read_path(inputs, operand["path"]), path + ("path",)


def _compare(
    operator: str,
    left: Any,
    right: Any = _MISSING,
    left_path: tuple[str | int, ...] = ("condition",),
    right_path: tuple[str | int, ...] = ("condition",),
) -> bool:
    if left is _MISSING:
        _error("missing_value", left_path, "condition input is unavailable")
    if operator not in _UNARY_OPERATORS and right is _MISSING:
        _error("missing_value", right_path, "condition input is unavailable")
    if operator == "is_null":
        return left is None
    if operator == "is_not_null":
        return left is not None
    if left is None or right is None:
        _error("invalid_value", left_path if left is None else right_path, "null is only valid with unary operators")
    if operator in {"eq", "ne"}:
        if not (_is_number(left) and _is_number(right)) and type(left) is not type(right):
            _error("invalid_type", right_path, "comparison operand types differ")
        equal = _strict_equal(left, right)
        return equal if operator == "eq" else not equal
    if operator in {"gt", "gte", "lt", "lte"}:
        if not _is_number(left):
            _error("invalid_type", left_path, "ordering requires numbers")
        if not _is_number(right):
            _error("invalid_type", right_path, "ordering requires numbers")
        return {"gt": left > right, "gte": left >= right, "lt": left < right, "lte": left <= right}[operator]
    if type(left) is str and type(right) is str:
        return right in left
    if type(left) is str:
        _error("invalid_type", right_path, "string contains requires a string")
    if type(left) is list:
        return any(_strict_equal(item, right) for item in left)
    _error("invalid_type", left_path, "contains requires strings or a list")


def _evaluate_condition(
    condition: dict[str, Any], inputs: dict[str, Any], path: tuple[str | int, ...] = ("condition",)
) -> bool:
    kind = condition["kind"]
    if kind == "all":
        return all(_evaluate_condition(child, inputs, path + ("conditions", index)) for index, child in enumerate(condition["conditions"]))
    if kind == "any":
        return any(_evaluate_condition(child, inputs, path + ("conditions", index)) for index, child in enumerate(condition["conditions"]))
    operator = condition["operator"]
    left, left_path = _evaluate_operand(condition["left"], inputs, path + ("left",))
    if operator in _UNARY_OPERATORS:
        return _compare(operator, left, left_path=left_path)
    right, right_path = _evaluate_operand(condition["right"], inputs, path + ("right",))
    return _compare(
        operator,
        left,
        right,
        left_path,
        right_path,
    )


def evaluate_condition(condition: Any, inputs: Any) -> bool:
    """Validate and evaluate a closed condition AST against JSON inputs."""
    _validate_json(condition, ("condition",))
    _validate_json(inputs, ("inputs",))
    if type(inputs) is not dict:
        _error("invalid_type", ("inputs",), "inputs must be an object")
    _validate_condition(condition, ("condition",))
    return _evaluate_condition(condition, inputs)


def _case_key(value: Any) -> tuple[str, Any]:
    if type(value) is bool:
        return ("bool", value)
    if _is_number(value):
        return ("number", value)
    if type(value) is str:
        return ("string", value)
    return ("null", None)


def select_case(value: Any, cases: Any) -> str:
    """Return a strictly matching scalar case ID, or the reserved default."""
    _validate_json(value, ("value",))
    _validate_json(cases, ("cases",))
    if type(cases) is not list:
        _error("invalid_type", ("cases",), "cases must be an array")
    seen_ids: set[str] = set()
    seen_values: set[tuple[str, Any]] = set()
    parsed: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        path = ("cases", index)
        obj = _closed_object(case, path, {"id", "value"})
        if type(obj["id"]) is not str or not obj["id"]:
            _error("invalid_value", path + ("id",), "case id must be a non-empty string")
        if obj["id"] == "default":
            _error("invalid_value", path + ("id",), "default is reserved")
        if obj["id"] in seen_ids:
            _error("duplicate_name", path + ("id",), "case id is duplicated")
        seen_ids.add(obj["id"])
        _validate_json(obj["value"], path + ("value",))
        if type(obj["value"]) not in (type(None), bool, int, float, str):
            _error("invalid_type", path + ("value",), "case value must be a JSON scalar")
        key = _case_key(obj["value"])
        if key in seen_values:
            _error("duplicate_value", path + ("value",), "case value is duplicated")
        seen_values.add(key)
        parsed.append(obj)
    for case in parsed:
        if _strict_equal(value, case["value"]):
            return case["id"]
    return "default"
