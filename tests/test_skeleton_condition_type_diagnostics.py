"""CP17: literal type errors terminate groups without unreachable diagnostics."""
from copy import deepcopy

import pytest

from unchain.skeleton import WorkflowExecutionError
from tests.test_skeleton_condition_admission import (
    _assert_static_rejection,
    _branch,
    _compare,
    _diagnostic_pairs,
    _document,
    _group,
    _input,
    _literal,
    _missing,
    _prefix,
    _run,
)


def _terminal_document(kind, condition):
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _group(kind, condition, _missing("following"))
    return document


def _assert_terminal_runtime_error(document, *, code="invalid_type", suffix, flag=1):
    before = deepcopy(document)
    assert _diagnostic_pairs(document) == []
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, dict(flag=flag), calls)
    expected = _prefix(document) + ("condition", "conditions", 0) + suffix
    assert (error.value.code, error.value.path) == (code, expected)
    assert calls == ["before"], "existing type errors remain runtime errors, rather than new admission failures"
    assert document == before


@pytest.mark.parametrize("kind", ["all", "any"])
@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
@pytest.mark.parametrize("invalid_side,invalid_literal", [("left", "not-a-number"), ("right", True)])
def test_literal_type_comparison_stops_before_later_static_missing(kind, operator, invalid_side, invalid_literal):
    operands = dict(left=_input(["flag"]), right=_input(["flag"]))
    operands[invalid_side] = _literal(invalid_literal)
    document = _terminal_document(kind, _compare(operands["left"], operands["right"], operator=operator))
    _assert_terminal_runtime_error(document, suffix=(invalid_side, "value"))


@pytest.mark.parametrize("kind", ["all", "any"])
@pytest.mark.parametrize("invalid_literal", [True, 4, {"x": 1}], ids=["bool", "number", "object"])
def test_invalid_contains_literal_left_stops_before_later_static_missing(kind, invalid_literal):
    document = _terminal_document(kind, _compare(_literal(invalid_literal), _input(["flag"]), operator="contains"))
    _assert_terminal_runtime_error(document, suffix=("left", "value"))


@pytest.mark.parametrize("operator,invalid_side,invalid_literal", [
    ("gt", "left", "not-a-number"),
    ("lte", "right", True),
    ("contains", "left", {"x": 1}),
])
def test_known_missing_still_wins_over_other_literal_type_error(operator, invalid_side, invalid_literal):
    missing_side = "right" if invalid_side == "left" else "left"
    operands = dict(left=_input(["absent"]), right=_input(["absent"]))
    operands[invalid_side] = _literal(invalid_literal)
    document = _terminal_document("any", _compare(operands["left"], operands["right"], operator=operator))
    expected = _prefix(document) + ("condition", "conditions", 0, missing_side, "path")
    before = deepcopy(document)
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)
    assert document == before


@pytest.mark.parametrize("operator,invalid_literal", [("gt", "not-a-number"), ("contains", True)])
def test_null_runtime_priority_before_invalid_literal_type_is_preserved(operator, invalid_literal):
    document = _terminal_document("all", _compare(_literal(invalid_literal), _literal(None), operator=operator))
    _assert_terminal_runtime_error(document, code="invalid_value", suffix=("right", "value"))


@pytest.mark.parametrize("operator", ["gt", "gte", "lt", "lte"])
@pytest.mark.parametrize("number", [2, 2.0], ids=["int", "float"])
def test_valid_numeric_literals_allow_dynamic_success_and_reachable_missing_diagnosis(operator, number):
    condition = _compare(_literal(number), _input(["flag"]), operator=operator)
    true_flag, false_flag = (1, 3) if operator in {"gt", "gte"} else (3, 1)
    for kind, safe_flag, unsafe_flag, selected in [
        ("any", true_flag, false_flag, "left"),
        ("all", false_flag, true_flag, "right"),
    ]:
        document = _terminal_document(kind, condition)
        before = deepcopy(document)
        expected = _prefix(document) + ("condition", "conditions", 1, "left", "path")
        assert _diagnostic_pairs(document) == [("missing_value", expected)]
        calls = []
        assert _run(document, dict(flag=safe_flag), calls).outputs == dict(answer=selected)
        assert calls == ["before", selected]
        calls = []
        with pytest.raises(WorkflowExecutionError) as error:
            _run(document, dict(flag=unsafe_flag), calls)
        assert (error.value.code, error.value.path) == ("missing_value", expected)
        assert calls == ["before"]
        assert document == before


@pytest.mark.parametrize("kind", ["all", "any"])
@pytest.mark.parametrize("literal_left,true_flag,false_flag", [
    ("apple", "app", "missing"),
    ([1, "x"], 1, 9),
], ids=["string", "list"])
def test_valid_contains_literal_left_keeps_dynamic_success_and_reachable_missing_diagnosis(kind, literal_left, true_flag, false_flag):
    document = _terminal_document(kind, _compare(_literal(literal_left), _input(["flag"]), operator="contains"))
    before = deepcopy(document)
    expected = _prefix(document) + ("condition", "conditions", 1, "left", "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    safe_flag, unsafe_flag, selected = (true_flag, false_flag, "left") if kind == "any" else (false_flag, true_flag, "right")
    calls = []
    assert _run(document, dict(flag=safe_flag), calls).outputs == dict(answer=selected)
    assert calls == ["before", selected]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, dict(flag=unsafe_flag), calls)
    assert (error.value.code, error.value.path) == ("missing_value", expected)
    assert calls == ["before"]
    assert document == before


def test_hidden_bad_grammar_is_validated_before_guaranteed_literal_type_error():
    document = _terminal_document("any", _compare(_literal("not-a-number"), _input(["flag"]), operator="gt"))
    _branch(document)["condition"]["conditions"][1]["operator"] = "unsupported"
    before = deepcopy(document)
    expected = _prefix(document) + ("condition", "conditions", 1, "operator")
    assert _diagnostic_pairs(document) == [("unknown_variant", expected)]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == ("unknown_variant", expected)
    assert calls == []
    assert document == before
