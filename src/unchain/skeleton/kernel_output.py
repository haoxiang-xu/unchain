"""Strict, provider-free admission for structured kernel outputs."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .errors import SkeletonValidationError
from .io import _MAX_DEPTH, _MAX_NODES


_DRAFT_2020_12 = "https://json-schema.org/draft/2020-12/schema"
_ALLOWED_KEYWORDS = frozenset(
    {
        "$schema",
        "$defs",
        "$ref",
        "title",
        "description",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "anyOf",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
    }
)
_JSON_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})
_RESERVED_ROOT_PROPERTIES = frozenset({"message", "context"})
# This matches the Skeleton JSON content limit without coupling output admission
# to the durable execution protocol module.
_MAX_JSON_BYTES = 8 * 1024 * 1024


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


class _Leave:
    def __init__(self, value_id: int):
        self.value_id = value_id


_LEAVE = _Leave(-1)


def _validate_json_tree(value: Any, path: tuple[str | int, ...]) -> None:
    """Validate the Skeleton JSON limits without recursing through input."""
    stack: list[tuple[Any, tuple[str | int, ...], int]] = [(value, path, 0)]
    active: set[int] = set()
    count = 0
    while stack:
        current, current_path, depth = stack.pop()
        if isinstance(current, _Leave):
            active.remove(current.value_id)
            continue
        count += 1
        if count > _MAX_NODES:
            _fail("limit_nodes", current_path, "JSON value exceeds the node limit")
        if depth > _MAX_DEPTH:
            _fail("limit_depth", current_path, "JSON value exceeds the depth limit")
        if current is None or type(current) in (str, bool, int):
            continue
        if type(current) is float:
            if not math.isfinite(current):
                _fail("invalid_json", current_path, "JSON number must be finite")
            continue
        if type(current) not in (dict, list):
            _fail("invalid_json", current_path, "value is not JSON")
        current_id = id(current)
        if current_id in active:
            _fail("invalid_json", current_path, "JSON value cannot contain a cycle")
        active.add(current_id)
        stack.append((_Leave(current_id), current_path, depth))
        if type(current) is dict:
            for key, child in reversed(tuple(current.items())):
                if type(key) is not str:
                    _fail("invalid_json", current_path, "object keys must be strings")
                stack.append((child, current_path + (key,), depth + 1))
        else:
            for index in range(len(current) - 1, -1, -1):
                stack.append((current[index], current_path + (index,), depth + 1))


def _require_object(value: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("invalid_schema", path, "schema nodes must be objects")
    return value


def _require_nonnegative_integer(value: Any, path: tuple[str | int, ...]) -> None:
    if type(value) is not int or value < 0:
        _fail("invalid_schema", path, "schema limit must be a non-negative integer")


def _require_number(value: Any, path: tuple[str | int, ...]) -> None:
    if type(value) is int:
        return
    if type(value) is not float or not math.isfinite(value):
        _fail("invalid_schema", path, "schema bound must be a finite JSON number")


def _decode_pointer_token(token: str, path: tuple[str | int, ...]) -> str:
    result: list[str] = []
    index = 0
    while index < len(token):
        character = token[index]
        if character != "~":
            result.append(character)
            index += 1
            continue
        if index + 1 == len(token) or token[index + 1] not in {"0", "1"}:
            _fail("invalid_reference", path, "reference pointer is invalid")
        result.append("~" if token[index + 1] == "0" else "/")
        index += 2
    return "".join(result)


def _resolve_local_reference(
    root: dict[str, Any], reference: str, path: tuple[str | int, ...]
) -> Any:
    """Resolve supported JSON Pointer ``~0``/``~1`` escapes under ``#/$defs/``.

    Percent-encoded URI fragments are rejected so this bounded resolver cannot
    disagree with the validator's URI-fragment handling.
    """
    if "%" in reference:
        _fail("unsupported_reference", path, "percent-encoded references are unsupported")
    if not reference.startswith("#/$defs/"):
        _fail("unsupported_reference", path, "only local definition references are supported")
    current: Any = root
    for raw_token in reference[2:].split("/"):
        token = _decode_pointer_token(raw_token, path)
        if type(current) is dict:
            if token not in current:
                _fail("invalid_reference", path, "reference target is unavailable")
            current = current[token]
        elif type(current) is list:
            if not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
                _fail("invalid_reference", path, "reference pointer is invalid")
            index = int(token)
            if index >= len(current):
                _fail("invalid_reference", path, "reference target is unavailable")
            current = current[index]
        else:
            _fail("invalid_reference", path, "reference target is unavailable")
    return current


class _SchemaWalker:
    def __init__(self, root: dict[str, Any]):
        self.root = root
        self.nodes: dict[int, dict[str, Any]] = {}
        self.edges: dict[int, set[int]] = {}
        self.references: list[tuple[dict[str, Any], str, tuple[str | int, ...]]] = []

    def walk(self, value: Any, path: tuple[str | int, ...], *, root: bool = False) -> dict[str, Any]:
        schema = _require_object(value, path)
        node_id = id(schema)
        if node_id in self.nodes:
            return schema
        self.nodes[node_id] = schema
        self.edges[node_id] = set()
        for keyword, child in schema.items():
            keyword_path = path + (keyword,)
            if keyword not in _ALLOWED_KEYWORDS:
                _fail("unknown_keyword", keyword_path, "schema keyword is unsupported")
            if keyword == "$schema":
                if type(child) is not str or child != _DRAFT_2020_12:
                    _fail("invalid_schema", keyword_path, "schema dialect is unsupported")
            elif keyword == "$defs":
                definitions = _require_object(child, keyword_path)
                for name, definition in definitions.items():
                    child_schema = self.walk(definition, keyword_path + (name,))
                    self.edges[node_id].add(id(child_schema))
            elif keyword == "$ref":
                if type(child) is not str:
                    _fail("invalid_schema", keyword_path, "reference must be a string")
                self.references.append((schema, child, keyword_path))
            elif keyword in {"title", "description"}:
                if type(child) is not str:
                    _fail("invalid_schema", keyword_path, "schema annotation must be a string")
            elif keyword == "type":
                self._validate_type(child, keyword_path, root=root)
            elif keyword == "properties":
                properties = _require_object(child, keyword_path)
                if root and _RESERVED_ROOT_PROPERTIES.intersection(properties):
                    _fail("reserved_name", keyword_path, "root output property is reserved")
                for name, property_schema in properties.items():
                    child_schema = self.walk(property_schema, keyword_path + (name,))
                    self.edges[node_id].add(id(child_schema))
            elif keyword == "required":
                if type(child) is not list or any(type(name) is not str for name in child):
                    _fail("invalid_schema", keyword_path, "required must be an array of strings")
                if len(set(child)) != len(child):
                    _fail("invalid_schema", keyword_path, "required names must be unique")
            elif keyword == "additionalProperties":
                if type(child) is not bool:
                    _fail("invalid_schema", keyword_path, "additionalProperties must be boolean")
            elif keyword == "items":
                child_schema = self.walk(child, keyword_path)
                self.edges[node_id].add(id(child_schema))
            elif keyword == "enum":
                if type(child) is not list or not child:
                    _fail("invalid_schema", keyword_path, "enum must be a non-empty array")
                _validate_json_tree(child, keyword_path)
            elif keyword == "const":
                _validate_json_tree(child, keyword_path)
            elif keyword == "anyOf":
                if type(child) is not list or not child:
                    _fail("invalid_schema", keyword_path, "anyOf must be a non-empty array")
                for index, alternative in enumerate(child):
                    child_schema = self.walk(alternative, keyword_path + (index,))
                    self.edges[node_id].add(id(child_schema))
            elif keyword in {"minItems", "maxItems", "minLength", "maxLength"}:
                _require_nonnegative_integer(child, keyword_path)
            elif keyword in {"minimum", "maximum"}:
                _require_number(child, keyword_path)
        return schema

    def _validate_type(self, value: Any, path: tuple[str | int, ...], *, root: bool) -> None:
        if type(value) is str:
            types = [value]
        elif type(value) is list and value and all(type(item) is str for item in value):
            types = value
        else:
            _fail("invalid_schema", path, "type must be a JSON type or non-empty array of JSON types")
        if any(item not in _JSON_TYPES for item in types) or len(set(types)) != len(types):
            _fail("invalid_schema", path, "type contains an unsupported JSON type")
        if root and value != "object":
            _fail("invalid_schema", path, "root schema type must be object")

    def resolve_references(self) -> None:
        for source, reference, path in self.references:
            target = _resolve_local_reference(self.root, reference, path)
            target_id = id(target)
            if type(target) is not dict or target_id not in self.nodes:
                _fail("invalid_reference", path, "reference target must be a schema object")
            self.edges[id(source)].add(target_id)
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(node_id: int) -> None:
            if node_id in visiting:
                _fail("reference_cycle", ("schema",), "schema references cannot form a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child_id in self.edges[node_id]:
                visit(child_id)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(id(self.root))

    def reject_effective_root_reservations(self) -> None:
        """Reject reserved properties exposed by root ``$ref``/``anyOf`` chains.

        A schema nested below ``properties`` describes a nested object, so it is
        deliberately outside this traversal and may use the reserved names.
        """
        active: set[int] = set()

        def visit(schema: dict[str, Any]) -> None:
            schema_id = id(schema)
            if schema_id in active:
                return
            active.add(schema_id)
            properties = schema.get("properties")
            if type(properties) is dict and _RESERVED_ROOT_PROPERTIES.intersection(properties):
                _fail("reserved_name", ("schema", "properties"), "root output property is reserved")
            reference = schema.get("$ref")
            if type(reference) is str:
                target = _resolve_local_reference(self.root, reference, ("schema", "$ref"))
                if type(target) is dict:
                    visit(target)
            alternatives = schema.get("anyOf")
            if type(alternatives) is list:
                for alternative in alternatives:
                    if type(alternative) is dict:
                        visit(alternative)
            active.remove(schema_id)

        visit(self.root)


def _enforce_schema_size(schema: Any) -> None:
    try:
        encoded = json.dumps(
            schema,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8", "surrogatepass")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise SkeletonValidationError(
            "invalid_schema", ("schema",), "schema cannot be encoded as bounded JSON"
        ) from exc
    if len(encoded) > _MAX_JSON_BYTES:
        _fail("limit_bytes", ("schema",), "schema exceeds eight MiB")


def validate_output_schema(schema: Any) -> dict[str, Any]:
    """Validate and detach the bounded Draft 2020-12 output-schema subset."""
    _validate_json_tree(schema, ("schema",))
    _enforce_schema_size(schema)
    root = _require_object(schema, ("schema",))
    walker = _SchemaWalker(root)
    walker.walk(root, ("schema",), root=True)
    if root.get("type") != "object":
        _fail("invalid_schema", ("schema", "type"), "root schema type must be object")
    walker.resolve_references()
    walker.reject_effective_root_reservations()
    try:
        Draft202012Validator.check_schema(root)
    except (SchemaError, TypeError, ValueError, RecursionError) as exc:
        raise SkeletonValidationError(
            "invalid_schema", ("schema",), "schema is not valid Draft 2020-12"
        ) from exc
    return deepcopy(root)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_name", ("output",), "JSON object contains a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> None:
    _fail("invalid_json", ("output",), "JSON number must be finite")


def parse_kernel_output(text: Any, schema: Any) -> dict[str, Any]:
    """Parse a final kernel JSON object against a validated output schema."""
    admitted_schema = validate_output_schema(schema)
    if type(text) is not str:
        _fail("invalid_type", ("output",), "kernel output must be text")
    try:
        raw = text.encode("utf-8", "surrogatepass")
    except UnicodeError as exc:
        raise SkeletonValidationError(
            "invalid_json", ("output",), "kernel output is not valid Unicode text"
        ) from exc
    if len(raw) > _MAX_JSON_BYTES:
        _fail("limit_bytes", ("output",), "kernel output exceeds eight MiB")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
        )
    except SkeletonValidationError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError, UnicodeError) as exc:
        raise SkeletonValidationError("invalid_json", ("output",), "kernel output is not valid JSON") from exc
    if type(value) is not dict:
        _fail("invalid_type", ("output",), "kernel output must be a JSON object")
    _validate_json_tree(value, ("output",))
    if _RESERVED_ROOT_PROPERTIES.intersection(value):
        _fail("reserved_name", ("output",), "kernel output contains a reserved property")
    validator = Draft202012Validator(admitted_schema)
    if next(validator.iter_errors(value), None) is not None:
        _fail("output_invalid", ("output",), "kernel output does not satisfy its schema")
    return deepcopy(value)
