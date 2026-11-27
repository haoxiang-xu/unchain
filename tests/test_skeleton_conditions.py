from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from unchain.skeleton import SkeletonValidationError, evaluate_condition, select_case


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_conditions"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def compare(operator, left, right=None):
    condition = {"kind": "compare", "operator": operator, "left": left}
    if right is not None:
        condition["right"] = right
    return condition


def literal(value):
    return {"kind": "literal", "value": value}


def input_at(*path):
    return {"kind": "input", "path": list(path)}


def test_condition_fixture_nested_selection():
    data = fixture("nested_condition.json")
    assert evaluate_condition(**data) is True


def test_switch_fixture():
    assert select_case(**fixture("switch_cases.json")) == "beta-case"


def test_provider_free_import():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import unchain.skeleton; assert 'unchain.providers' not in sys.modules"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_short_circuit_skips_missing_read_but_validates_hidden_children():
    condition = {"kind": "any", "conditions": [
        compare("eq", literal(1), literal(1)),
        compare("eq", input_at("missing"), literal(1)),
    ]}
    assert evaluate_condition(condition, {}) is True
    condition["conditions"][1] = {"kind": "compare", "operator": "bogus", "left": literal(1), "right": literal(1)}
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(condition, {})
    assert error.value.path == ("condition", "conditions", 1, "operator")


def test_unary_null_and_binary_null_semantics():
    assert evaluate_condition(compare("is_null", input_at("value")), {"value": None}) is True
    assert evaluate_condition(compare("is_not_null", input_at("value")), {"value": None}) is False
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(compare("eq", input_at("value"), literal(1)), {"value": None})
    assert error.value.path == ("condition", "left", "path")
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition({**compare("is_null", input_at("value")), "right": literal(None)}, {"value": None})
    assert error.value.path == ("condition", "right")
    for condition, expected_path in [
        (compare("eq", literal(None), literal(1)), ("condition", "left", "value")),
        (compare("eq", literal(1), literal(None)), ("condition", "right", "value")),
    ]:
        with pytest.raises(SkeletonValidationError) as error:
            evaluate_condition(condition, {})
        assert error.value.path == expected_path


def test_numeric_and_strict_nested_equality():
    assert evaluate_condition(compare("eq", literal(1), literal(1.0)), {}) is True
    assert evaluate_condition(compare("gt", literal(2), literal(1.5)), {}) is True
    assert evaluate_condition(compare("eq", literal([True]), literal([1])), {}) is False
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(compare("eq", literal(True), literal(1)), {})
    assert error.value.path == ("condition", "right", "value")


def test_contains_and_invalid_combinations():
    assert evaluate_condition(compare("contains", literal("abcdef"), literal("bcd")), {}) is True
    assert evaluate_condition(compare("contains", literal([{"x": True}]), literal({"x": 1})), {}) is False
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(compare("contains", literal("abc"), literal(1)), {})
    assert error.value.path == ("condition", "right", "value")


def test_invalid_ordering_reports_the_first_invalid_operand():
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(compare("gt", literal("not-a-number"), literal(1)), {})
    assert error.value.path == ("condition", "left", "value")
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(compare("gt", literal(1), literal("not-a-number")), {})
    assert error.value.path == ("condition", "right", "value")


def test_nested_group_runtime_error_keeps_nested_operand_path():
    condition = {"kind": "all", "conditions": [
        compare("eq", literal(1), literal(1)),
        {"kind": "any", "conditions": [compare("eq", literal(True), literal(1))]},
    ]}
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(condition, {})
    assert error.value.path == ("condition", "conditions", 1, "conditions", 0, "right", "value")


def test_missing_input_has_exact_path_when_evaluated():
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(compare("eq", input_at("missing"), literal(1)), {})
    assert error.value.code == "missing_value"
    assert error.value.path == ("condition", "left", "path")


@pytest.mark.parametrize("bad", [
    {"kind": "all", "conditions": []},
    {"kind": "compare", "operator": "eq", "left": literal(1)},
    {"kind": "compare", "operator": "unknown", "left": literal(1), "right": literal(1)},
    {"kind": "all", "conditions": [], "extra": 1},
])
def test_malformed_conditions_are_structured(bad):
    with pytest.raises(SkeletonValidationError):
        evaluate_condition(bad, {})


@pytest.mark.parametrize("bad", [("tuple",), {1: "bad"}])
def test_python_only_values_are_rejected(bad):
    with pytest.raises(SkeletonValidationError):
        evaluate_condition(compare("eq", literal(bad), literal(bad)), {})
    with pytest.raises(SkeletonValidationError):
        select_case(bad, [])


def test_switch_duplicate_default_and_strict_matching():
    with pytest.raises(SkeletonValidationError):
        select_case(1, [{"id": "one", "value": 1}, {"id": "float", "value": 1.0}])
    assert select_case(True, [{"id": "one", "value": 1}, {"id": "bool", "value": True}]) == "bool"
    with pytest.raises(SkeletonValidationError):
        select_case("x", [{"id": "default", "value": "x"}])
    with pytest.raises(SkeletonValidationError):
        select_case("x", [{"id": "list", "value": []}])


def test_switch_default_for_composite_or_unmatched_values():
    cases = [{"id": "one", "value": 1}]
    assert select_case([1], cases) == "default"
    assert select_case("other", cases) == "default"
