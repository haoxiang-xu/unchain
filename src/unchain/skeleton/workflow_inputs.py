"""Declared Workflow function signatures and provider-free argument admission."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .io import _closed_object, _error, _validate_json

_TYPES = {"string", "number", "integer", "boolean", "object", "array", "null"}
_PATH = ("document", "input_schema")


def _matches(value: Any, kind: str | None) -> bool:
    if kind is None:
        return True
    return {"string": type(value) is str, "number": type(value) in (int, float),
        "integer": type(value) is int or (type(value) is float and value.is_integer()),
        "boolean": type(value) is bool, "object": type(value) is dict,
        "array": type(value) is list, "null": value is None}[kind]


def workflow_input_schema(document: dict[str, Any]) -> dict[str, Any] | None:
    """Admit the bounded, closed signature; absent means the retained contract.

    A parameter is required or has an explicit default. Defaults are ordinary
    values, not coercions: null, false, zero and empty text remain distinct.
    """
    if "input_schema" not in document:
        return None
    if document.get("kind") != "workflow":
        _error("invalid_value", _PATH, "only a Workflow can declare parameters")
    schema = _closed_object(document["input_schema"], _PATH,
        {"type", "properties", "required", "additionalProperties"}, {"description"})
    if schema["type"] != "object":
        _error("invalid_value", _PATH + ("type",), "Workflow parameters must describe an object")
    if schema["additionalProperties"] is not False:
        _error("invalid_value", _PATH + ("additionalProperties",), "undeclared Workflow arguments are forbidden")
    if "description" in schema and type(schema["description"]) is not str:
        _error("invalid_type", _PATH + ("description",), "description must be text")
    if type(schema["properties"]) is not dict:
        _error("invalid_type", _PATH + ("properties",), "parameters must be an object")
    if type(schema["required"]) is not list:
        _error("invalid_type", _PATH + ("required",), "required must be an array")
    required: set[str] = set()
    for index, name in enumerate(schema["required"]):
        at = _PATH + ("required", index)
        if type(name) is not str:
            _error("invalid_type", at, "required parameter name must be text")
        if name in required:
            _error("duplicate_name", at, "required parameter is duplicated")
        if name not in schema["properties"]:
            _error("missing_value", at, "required parameter is not declared")
        required.add(name)
    for name, field in schema["properties"].items():
        at = _PATH + ("properties", name)
        if not name.strip():
            _error("invalid_value", at, "parameter name must be nonblank")
        field = _closed_object(field, at, set(), {"type", "description", "default"})
        kind = field.get("type")
        if "type" in field and (type(kind) is not str or kind not in _TYPES):
            _error("invalid_type", at + ("type",), "parameter type is invalid")
        if "description" in field and type(field["description"]) is not str:
            _error("invalid_type", at + ("description",), "description must be text")
        if name in required and "default" in field:
            _error("invalid_value", at + ("default",), "a required parameter cannot have a default")
        if name not in required and "default" not in field:
            _error("missing_field", at + ("default",), "an optional parameter needs an explicit default")
        if "default" in field and not _matches(field["default"], kind):
            _error("invalid_type", at + ("default",), "default does not match the parameter type")
    return deepcopy(schema)


def bind_workflow_inputs(document: dict[str, Any], inputs: Any) -> dict[str, Any]:
    """Validate before execution/persistence and detach default-expanded args."""
    _validate_json(inputs, ("inputs",))
    if type(inputs) is not dict:
        _error("invalid_type", ("inputs",), "Workflow arguments must be an object")
    schema = workflow_input_schema(document)
    if schema is None:
        return deepcopy(inputs)
    fields = schema["properties"]
    for name in inputs:
        if name not in fields:
            _error("unknown_field", ("inputs", name), "Workflow argument is not declared")
    result = deepcopy(inputs)
    required = set(schema["required"])
    for name, field in fields.items():
        if name not in result:
            if name in required:
                _error("missing_value", ("inputs", name), "required Workflow argument is missing")
            result[name] = deepcopy(field["default"])
        if not _matches(result[name], field.get("type")):
            _error("invalid_type", ("inputs", name), "Workflow argument does not match its declared type")
    _validate_json(result, ("inputs",))
    return result
