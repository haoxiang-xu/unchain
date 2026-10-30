"""CP17: real branch admission preserves ordered condition evaluation semantics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from unchain.skeleton import (
    SkeletonValidationError,
    WorkflowExecutionError,
    WorkflowNodeCompleted,
    diagnose_skeleton_draft,
    evaluate_condition,
    prepare_workflow,
    run_workflow,
)


def _document(switch=False, *, preceding_handler=False):
    filename = "switch_workflow.json" if switch else "if_workflow.json"
    source = json.loads((Path(__file__).parent / "fixtures" / "skeleton_execution" / filename).read_text())
    if preceding_handler:
        branch = source["nodes"][1]["id"]
        source["nodes"].insert(1, dict(id="before", type="code", config=dict(execution=dict(inputs=[], outputs=[]))))
        source["edges"][0]["target_node_id"] = "before"
        source["edges"].append(dict(id="before-branch", kind="flow", source_node_id="before", source_port_id="out",
                                    target_node_id=branch, target_port_id="in"))
    return source


def _branch(document):
    return next(node["config"]["execution"] for node in document["nodes"]
                if node["type"] in {"if_else", "switch_case"})


def _prefix(document):
    index = next(index for index, node in enumerate(document["nodes"])
                 if node["type"] in {"if_else", "switch_case"})
    return ("document", "nodes", index, "config", "execution")


def _input(path):
    return dict(kind="input", path=path)


def _literal(value):
    return dict(kind="literal", value=value)


def _compare(left, right=None, *, operator="eq"):
    value = dict(kind="compare", operator=operator, left=left)
    if operator not in {"is_null", "is_not_null"}:
        value["right"] = _literal(True) if right is None else right
    return value


def _fixed(value):
    return _compare(_literal(value), _literal(True))


def _missing(name="absent"):
    return _compare(_input([name]))


def _group(kind, *children):
    return dict(kind=kind, conditions=list(children))


def _diagnostic_pairs(document):
    return [(item.code, item.path) for item in diagnose_skeleton_draft(document)]


def _run(document, inputs=None, calls=None):
    calls = [] if calls is None else calls

    def code(context, _):
        calls.append(context.node_id)
        return WorkflowNodeCompleted(dict(answer=context.node_id, value=context.node_id))

    return run_workflow(prepare_workflow(document), dict(flag=True) if inputs is None else inputs, dict(code=code))


def _assert_static_rejection(document, expected):
    """A real code node preceding the branch proves rejection is admission-time."""
    before = deepcopy(document)
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == ("missing_value", expected)
    assert calls == [], "branch admission must happen before an earlier code handler"
    assert document == before


@pytest.mark.parametrize("operator", ["eq", "ne", "gt", "gte", "lt", "lte", "contains", "is_null", "is_not_null"])
def test_every_compare_reports_full_missing_operand_and_rejects_before_any_effect(operator):
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _compare(_input(["absent"]), operator=operator)
    expected = _prefix(document) + ("condition", "left", "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)


def test_left_and_right_diagnosed_in_order_and_first_error_is_left():
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _compare(_input(["left-absent"]), _input(["right-absent"]))
    prefix = _prefix(document) + ("condition",)
    assert _diagnostic_pairs(document) == [("missing_value", prefix + (side, "path")) for side in ("left", "right")]
    _assert_static_rejection(document, prefix + ("left", "path"))


def test_right_missing_operand_has_its_own_complete_path():
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _compare(_literal(True), _input(["absent"]))
    expected = _prefix(document) + ("condition", "right", "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)


@pytest.mark.parametrize("missing_side", ["left", "right"])
def test_missing_operand_precedes_other_operand_null_error(missing_side):
    document = _document(preceding_handler=True)
    operands = {"left": _literal(None), "right": _literal(None)}
    operands[missing_side] = _input(["absent"])
    _branch(document)["condition"] = _compare(operands["left"], operands["right"])
    expected = _prefix(document) + ("condition", missing_side, "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)


@pytest.mark.parametrize("switch", [False, True], ids=["if", "switch"])
def test_integer_root_is_statically_missing_in_resolved_input_object(switch):
    document = _document(switch, preceding_handler=True)
    execution = _branch(document)
    if switch:
        execution["value"] = _input([0])
        suffix = ("value", "path")
    else:
        execution["condition"] = _compare(_input([0]))
        suffix = ("condition", "left", "path")
    expected = _prefix(document) + suffix
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)


def test_switch_undeclared_name_is_diagnosed_and_rejected_before_code():
    document = _document(True, preceding_handler=True)
    _branch(document)["value"] = _input(["absent"])
    expected = _prefix(document) + ("value", "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)


@pytest.mark.parametrize("kind,decisive,selected", [
    ("all", False, "right"), ("any", True, "left"),
])
def test_decisive_literal_short_circuit_does_not_diagnose_or_reject_hidden_missing(kind, decisive, selected):
    document = _document()
    _branch(document)["condition"] = _group(kind, _fixed(decisive), _missing())
    before = deepcopy(document)
    assert _diagnostic_pairs(document) == []
    assert _run(document).outputs == dict(answer=selected)
    assert document == before


@pytest.mark.parametrize("kind,tail", [("all", False), ("any", True)])
def test_decisive_literal_after_missing_does_not_make_missing_unreachable(kind, tail):
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _group(kind, _missing(), _fixed(tail))
    expected = _prefix(document) + ("condition", "conditions", 0, "left", "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    _assert_static_rejection(document, expected)


@pytest.mark.parametrize("kind,safe_flag,selected", [("all", False, "right"), ("any", True, "left")])
def test_dynamic_short_circuit_may_succeed_so_diagnostic_is_not_blanket_admission_failure(kind, safe_flag, selected):
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _group(kind, _compare(_input(["flag"])), _missing())
    expected = _prefix(document) + ("condition", "conditions", 1, "left", "path")
    assert _diagnostic_pairs(document) == [("missing_value", expected)]
    calls = []
    assert _run(document, dict(flag=safe_flag), calls).outputs == dict(answer=selected)
    assert calls == ["before", selected]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, dict(flag=not safe_flag), calls)
    assert (error.value.code, error.value.path) == ("missing_value", expected)
    assert calls == ["before"], "the admitted dynamic failure remains a runtime failure"


@pytest.mark.parametrize("outer,inner", [("any", "all"), ("all", "any")])
def test_nested_dynamic_groups_with_only_missing_terminal_outcomes_reject(outer, inner):
    document = _document(preceding_handler=True)
    # A successfully resolved named binding is present. Unary is_null always
    # produces a boolean, while eq True can additionally fail on null/type.
    dynamic = _compare(_input(["flag"]), operator="is_null")
    _branch(document)["condition"] = _group(outer, _group(inner, dynamic, _missing("nested")), _missing("last"))
    prefix = _prefix(document) + ("condition",)
    expected = prefix + ("conditions", 0, "conditions", 1, "left", "path")
    assert _diagnostic_pairs(document) == [
        ("missing_value", expected),
        ("missing_value", prefix + ("conditions", 1, "left", "path")),
    ]
    _assert_static_rejection(document, expected)


@pytest.mark.parametrize("flag,code,suffix", [
    (None, "invalid_value", ("left", "path")),
    ("text", "invalid_type", ("right", "value")),
])
def test_nested_dynamic_compare_other_error_remains_admitted(flag, code, suffix):
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _group(
        "any", _group("all", _compare(_input(["flag"])), _missing("nested")), _missing("last"))
    prefix = _prefix(document) + ("condition",)
    assert _diagnostic_pairs(document) == [
        ("missing_value", prefix + ("conditions", 0, "conditions", 1, "left", "path")),
        ("missing_value", prefix + ("conditions", 1, "left", "path")),
    ]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, dict(flag=flag), calls)
    expected = prefix + ("conditions", 0, "conditions", 0) + suffix
    assert (error.value.code, error.value.path) == (code, expected)
    assert calls == ["before"], "OTHER_ERROR possibility prevents blanket static rejection"


def test_nested_known_short_circuit_keeps_later_dynamic_operand_and_hides_only_skipped_subtree():
    document = _document()
    _branch(document)["condition"] = _group("any", _group("all", _fixed(False), _missing()), _compare(_input(["flag"])))
    assert _diagnostic_pairs(document) == []
    assert _run(document, dict(flag=True)).outputs == dict(answer="left")
    assert _run(document, dict(flag=False)).outputs == dict(answer="right")


@pytest.mark.parametrize("switch", [False, True], ids=["if", "switch"])
def test_whole_input_object_path_is_present_and_runs(switch):
    document = _document(switch)
    if switch:
        _branch(document)["value"] = _input([])
    else:
        _branch(document)["condition"] = _compare(_input([]), _literal(dict(flag=True)))
    assert _diagnostic_pairs(document) == []
    result = _run(document)
    assert result.outputs == (dict(value="fallback") if switch else dict(answer="left"))


@pytest.mark.parametrize("switch", [False, True], ids=["if", "switch"])
def test_declared_root_deeper_json_path_is_unknown_until_runtime(switch):
    document = _document(switch, preceding_handler=True)
    execution = _branch(document)
    execution["inputs"] = [dict(name="payload", value=_literal({}), missing="error")]
    if switch:
        execution["value"] = _input(["payload", "dynamic"])
        suffix = ("value", "path")
    else:
        execution["condition"] = _compare(_input(["payload", "dynamic"]))
        suffix = ("condition", "left", "path")
    assert _diagnostic_pairs(document) == []
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == ("missing_value", _prefix(document) + suffix)
    assert calls == ["before"]
    execution["inputs"][0]["value"] = _literal(dict(dynamic="match" if switch else True))
    assert _run(document).outputs == (dict(value="case") if switch else dict(answer="left"))


def test_not_yet_declared_draft_inputs_do_not_invent_static_missing():
    document = _document()
    del _branch(document)["inputs"]
    _branch(document)["condition"] = _missing()
    assert _diagnostic_pairs(document) == [("missing_field", _prefix(document) + ("inputs",))]


def test_draft_condition_diagnostics_do_not_wait_for_kernel_configuration():
    document = _document()
    document["kind"] = "agent"
    document["nodes"][2]["type"] = "agent"
    _branch(document)["condition"] = _missing()
    expected = _prefix(document) + ("condition", "left", "path")
    assert ("missing_value", expected) in _diagnostic_pairs(document)
    assert "kernel" not in document["nodes"][2]["config"]


@pytest.mark.parametrize("kind,decisive", [("all", False), ("any", True)])
def test_hidden_malformed_ast_is_grammar_error_even_behind_decisive_short_circuit(kind, decisive):
    document = _document(preceding_handler=True)
    invalid = _missing()
    invalid["operator"] = "unsupported"
    _branch(document)["condition"] = _group(kind, _fixed(decisive), invalid)
    expected = _prefix(document) + ("condition", "conditions", 1, "operator")
    assert _diagnostic_pairs(document) == [("unknown_variant", expected)]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == ("unknown_variant", expected)
    assert calls == []


@pytest.mark.parametrize("switch,operand,code,suffix", [
    (False, dict(kind="input", path=[True]), "invalid_path", ("condition", "left", "path", 0)),
    (True, dict(kind="input", path="absent"), "invalid_type", ("value", "path")),
    (True, dict(kind="input", path=[], extra=True), "unknown_field", ("value", "extra")),
])
def test_declared_operand_grammar_has_precise_draft_paths(switch, operand, code, suffix):
    document = _document(switch, preceding_handler=True)
    if switch:
        _branch(document)["value"] = operand
    else:
        _branch(document)["condition"] = _compare(operand)
    expected = _prefix(document) + suffix
    assert _diagnostic_pairs(document) == [(code, expected)]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == (code, expected)
    assert calls == []


def test_later_node_grammar_error_precedes_earlier_static_branch_error():
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _missing()
    document["nodes"][4]["config"]["execution"]["outputs"][0]["source"]["path"] = [True]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == (
        "invalid_path", ("document", "nodes", 4, "config", "execution", "outputs", 0, "source", "path", 0))
    assert calls == []


def test_existing_required_ref_static_error_precedes_new_branch_static_error():
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = _missing()
    document["nodes"][4]["config"]["execution"]["inputs"] = [
        dict(name="x", value=dict(kind="ref", node_id="unknown", path=[]), missing="error")]
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == (
        "missing_value", ("document", "nodes", 4, "config", "execution", "inputs", 0, "value"))
    assert calls == []


@pytest.mark.parametrize("operator,selected", [("is_null", "left"), ("is_not_null", "right")])
def test_null_is_present_and_unary_operators_run_without_missing_diagnostic(operator, selected):
    document = _document()
    _branch(document)["condition"] = _compare(_literal(None), operator=operator)
    assert _diagnostic_pairs(document) == []
    assert _run(document).outputs == dict(answer=selected)


@pytest.mark.parametrize("condition,code,suffix", [
    (_compare(_literal(None), _literal(True)), "invalid_value", ("left", "value")),
    (_compare(_literal(1), _literal(True)), "invalid_type", ("right", "value")),
    (_group("any", _compare(_literal(1), _literal(True)), _missing()), "invalid_type", ("conditions", 0, "right", "value")),
])
def test_literal_runtime_errors_remain_admitted_and_stop_later_operand_reads(condition, code, suffix):
    document = _document(preceding_handler=True)
    _branch(document)["condition"] = condition
    assert _diagnostic_pairs(document) == []
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, calls=calls)
    assert (error.value.code, error.value.path) == (code, _prefix(document) + ("condition",) + suffix)
    assert calls == ["before"], "other existing runtime errors are outside static missing rejection"


@pytest.mark.parametrize("kind", ["all", "any"])
@pytest.mark.parametrize("null_side", ["left", "right"])
def test_literal_null_comparison_stops_before_later_static_missing(kind, null_side):
    document = _document(preceding_handler=True)
    operands = {"left": _input(["flag"]), "right": _input(["flag"])}
    operands[null_side] = _literal(None)
    _branch(document)["condition"] = _group(
        kind, _compare(operands["left"], operands["right"]), _missing())
    before = deepcopy(document)
    # Any successful resolution of the named flag still fails this comparison.
    # A later group's child can never be evaluated, so it has no diagnostic.
    assert _diagnostic_pairs(document) == []
    calls = []
    with pytest.raises(WorkflowExecutionError) as error:
        _run(document, dict(flag=True), calls)
    expected = _prefix(document) + ("condition", "conditions", 0, null_side, "value")
    assert (error.value.code, error.value.path) == ("invalid_value", expected)
    assert calls == ["before"], "the existing null error remains admitted until evaluation"
    assert document == before


@pytest.mark.parametrize("switch", [False, True], ids=["if", "switch"])
def test_durable_runner_refuses_static_missing_before_create_or_handler(tmp_path, switch):
    from unchain.journal import AttemptRef, GenerationRef
    from unchain.persistence import SQLiteContextV2Store
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    from unchain.skeleton.durable_store import WorkflowExecutionStore

    document = _document(switch)
    if switch:
        _branch(document)["value"] = _input(["absent"])
        suffix = ("value", "path")
    else:
        _branch(document)["condition"] = _missing()
        suffix = ("condition", "left", "path")
    before = deepcopy(document)
    repository = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite3", object_directory=tmp_path / "objects").bind_execution("execution")
    store = WorkflowExecutionStore(repository, prepare_workflow(document), graph_id="graph",
                                   attempt=AttemptRef(GenerationRef("execution", "generation"), "attempt"))
    calls = []
    with pytest.raises(SkeletonValidationError) as error:
        DurableWorkflowRunner(store, dict(code=lambda *_: calls.append("effect")))
    assert (error.value.code, error.value.path) == ("missing_value", _prefix(document) + suffix)
    assert calls == []
    assert store.load() is None
    assert document == before


def test_public_raw_evaluator_contract_still_fails_missing_unary_and_validates_hidden_ast():
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(_compare(_input(["absent"]), operator="is_null"), {})
    assert (error.value.code, error.value.path) == ("missing_value", ("condition", "left", "path"))
    invalid = _missing()
    invalid["operator"] = "unsupported"
    with pytest.raises(SkeletonValidationError) as error:
        evaluate_condition(_group("any", _fixed(True), invalid), {})
    assert (error.value.code, error.value.path) == ("unknown_variant", ("condition", "conditions", 1, "operator"))
