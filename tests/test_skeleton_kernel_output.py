from __future__ import annotations

import json

import pytest

from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.kernel_output import parse_kernel_output, validate_output_schema


def schema(**overrides):
    result = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "answer": {"$ref": "#/$defs/answer"},
        },
        "required": ["answer"],
        "additionalProperties": False,
        "$defs": {
            "answer": {"type": ["string", "null"]},
        },
    }
    result.update(overrides)
    return result


def test_valid_schema_and_output_are_detached_and_preserve_unicode_and_null():
    source = schema()
    admitted = validate_output_schema(source)
    source["$defs"]["answer"]["type"] = "integer"
    assert admitted["$defs"]["answer"]["type"] == ["string", "null"]

    parsed = parse_kernel_output('{"answer":"\u5b8c\u6210"}', admitted)
    assert parsed == {"answer": "完成"}
    assert parse_kernel_output('{"answer":null}', admitted) == {"answer": None}


@pytest.mark.parametrize(
    "mutate, expected_code",
    [
        (lambda value: value.update({"pattern": "x"}), "unknown_keyword"),
        (lambda value: value["$defs"]["answer"].update({"pattern": "x"}), "unknown_keyword"),
        (lambda value: value.update({"$schema": "https://example.invalid/schema"}), "invalid_schema"),
        (lambda value: value.update({"type": ["object", "null"]}), "invalid_schema"),
        (lambda value: value.update({"additionalProperties": {}}), "invalid_schema"),
        (lambda value: value.update({"properties": {"context": {"type": "string"}}}), "reserved_name"),
    ],
)
def test_schema_subset_is_closed_and_root_contract_is_strict(mutate, expected_code):
    value = schema()
    mutate(value)
    with pytest.raises(SkeletonValidationError) as error:
        validate_output_schema(value)
    assert error.value.code == expected_code


@pytest.mark.parametrize(
    "value, expected_code",
    [
        ({"type": "object", "$ref": "https://example.invalid/schema"}, "unsupported_reference"),
        ({"type": "object", "$ref": "#/properties/answer"}, "unsupported_reference"),
        (
            {
                "type": "object",
                "$defs": {"loop": {"$ref": "#/$defs/loop"}},
            },
            "reference_cycle",
        ),
        (
            {
                "type": "object",
                "$defs": {
                    "a": {"$ref": "#/$defs/b"},
                    "b": {"$ref": "#/$defs/a"},
                },
            },
            "reference_cycle",
        ),
    ],
)
def test_references_are_local_resolved_and_acyclic(value, expected_code):
    with pytest.raises(SkeletonValidationError) as error:
        validate_output_schema(value)
    assert error.value.code == expected_code


@pytest.mark.parametrize(
    "value",
    [
        {
            "type": "object",
            "$defs": {"result": {"type": "object", "properties": {"message": {"type": "string"}}}},
            "$ref": "#/$defs/result",
        },
        {
            "type": "object",
            "anyOf": [{"type": "object", "properties": {"context": {"type": "string"}}}],
        },
    ],
)
def test_effective_root_ref_and_anyof_cannot_expose_reserved_properties(value):
    with pytest.raises(SkeletonValidationError) as error:
        validate_output_schema(value)
    assert error.value.code == "reserved_name"


def test_nested_object_may_use_reserved_properties():
    value = schema(properties={"answer": {"type": "object", "properties": {"message": {"type": "string"}}}})
    assert parse_kernel_output('{"answer":{"message":"nested"}}', value) == {
        "answer": {"message": "nested"}
    }


def test_percent_encoded_reference_is_explicitly_unsupported_but_pointer_escapes_work():
    percent_encoded = {
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/a%2Fb"}},
        "$defs": {"a/b": {"type": "string"}},
    }
    with pytest.raises(SkeletonValidationError) as error:
        validate_output_schema(percent_encoded)
    assert error.value.code == "unsupported_reference"

    escaped = {
        "type": "object",
        "properties": {"answer": {"$ref": "#/$defs/a~1b"}},
        "$defs": {"a/b": {"type": "string"}},
    }
    assert parse_kernel_output('{"answer":"ok"}', escaped) == {"answer": "ok"}


@pytest.mark.parametrize(
    "text",
    [
        '{"answer":1}',
        '{"answer":"ok","extra":true}',
        '{}',
        '{"answer":"ok","context":{}}',
    ],
)
def test_output_rejects_types_properties_required_and_reserved_collisions(text):
    with pytest.raises(SkeletonValidationError) as error:
        parse_kernel_output(text, schema())
    assert error.value.code in {"output_invalid", "reserved_name"}


def test_output_rejects_duplicate_keys_and_nonfinite_numbers_without_echoing_values():
    for text, expected_code in [
        ('{"answer":"first","answer":"secret"}', "duplicate_name"),
        ('{"answer":NaN}', "invalid_json"),
        ('{"answer":Infinity}', "invalid_json"),
    ]:
        with pytest.raises(SkeletonValidationError) as error:
            parse_kernel_output(text, schema())
        assert error.value.code == expected_code
        assert "secret" not in str(error.value)


def test_output_respects_eight_mebibyte_skeleton_json_limit():
    oversized = " " * (8 * 1024 * 1024 + 1)
    with pytest.raises(SkeletonValidationError) as error:
        parse_kernel_output(oversized, schema())
    assert error.value.code == "limit_bytes"


def test_schema_respects_eight_mebibyte_limit_and_huge_bounds_fail_structurally():
    with pytest.raises(SkeletonValidationError) as error:
        validate_output_schema({"type": "object", "description": "x" * (8 * 1024 * 1024 + 1)})
    assert error.value.code == "limit_bytes"

    with pytest.raises(SkeletonValidationError) as error:
        validate_output_schema({"type": "object", "minimum": 10 ** 10_000})
    assert error.value.code == "invalid_schema"


def test_output_enum_validation_rejects_unlisted_value_without_echoing_it():
    value = schema(properties={"answer": {"enum": ["allowed", None]}})
    with pytest.raises(SkeletonValidationError) as error:
        parse_kernel_output(json.dumps({"answer": "secret"}), value)
    assert error.value.code == "output_invalid"
    assert "secret" not in str(error.value)
