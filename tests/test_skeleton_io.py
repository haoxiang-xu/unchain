from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from unchain.skeleton import SkeletonValidationError, map_outputs, resolve_inputs


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_io"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def test_resolve_inputs_fixture_and_detaches_values():
    data = fixture("resolve_inputs_success.json")
    resolved = resolve_inputs(**data)
    assert resolved == {"from_output": {"answer": 42}, "fallback": None, "defaulted": {"safe": True}}
    data["outputs"]["node-a"]["items"][1]["answer"] = 0
    assert resolved["from_output"] == {"answer": 42}


def test_provider_free_import_does_not_load_provider_package():
    result = subprocess.run(
        [sys.executable, "-c", "import sys; import unchain.skeleton; assert 'unchain.providers' not in sys.modules"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_missing_null_and_default_policies():
    bindings = [
        {"name": "null", "value": {"kind": "ref", "node_id": "n", "path": ["x"]}, "missing": "null"},
        {"name": "default", "value": {"kind": "ref", "node_id": "n", "path": ["x"]}, "missing": "default", "default": [1]},
    ]
    assert resolve_inputs(bindings, {}) == {"null": None, "default": [1]}
    with pytest.raises(SkeletonValidationError, match="unavailable") as error:
        resolve_inputs([{**bindings[0], "missing": "error"}], {})
    assert error.value.code == "missing_value"


def test_literal_keys_are_not_paths_and_null_is_present():
    bindings = [{"name": "x", "value": {"kind": "literal", "value": {"a.b": None}}, "missing": "error"}]
    assert resolve_inputs(bindings, {}) == {"x": {"a.b": None}}


def test_first_present_stops_at_present_null():
    bindings = [{
        "name": "x",
        "value": {
            "kind": "first_present",
            "values": [
                {"kind": "ref", "node_id": "node", "path": ["value"]},
                {"kind": "literal", "value": "later"},
            ],
        },
        "missing": "error",
    }]
    assert resolve_inputs(bindings, {"node": {"value": None}}) == {"x": None}


@pytest.mark.parametrize("bad", [
    [{"name": "x", "value": {"kind": "literal", "value": 1}, "missing": "error", "extra": 1}],
    [{"name": "x", "value": {"kind": "literal", "value": 1}, "missing": "default"}],
    [{"name": "x", "value": {"kind": "literal", "value": 1}, "missing": "null", "default": 1}],
    [{"name": "x", "value": {"kind": "ref", "node_id": "n", "path": [True]}, "missing": "error"}],
    [{"name": "x", "value": {"kind": "bogus"}, "missing": "error"}],
    [{"name": "x", "value": {"kind": "first_present", "values": [{"kind": "literal", "value": 1}, {"kind": "bogus"}]}, "missing": "error"}],
])
def test_invalid_bindings_are_structured_errors(bad):
    with pytest.raises(SkeletonValidationError) as error:
        resolve_inputs(bad, {})
    assert error.value.path


@pytest.mark.parametrize("missing", [[], {}])
def test_non_string_missing_policy_is_structured(missing):
    with pytest.raises(SkeletonValidationError) as error:
        resolve_inputs([{"name": "x", "value": {"kind": "literal", "value": 1}, "missing": missing}], {})
    assert error.value.code == "invalid_type"
    assert error.value.path == ("bindings", 0, "missing")


def test_duplicate_and_unavailable_binding_fail():
    with pytest.raises(SkeletonValidationError) as error:
        resolve_inputs([
            {"name": "x", "value": {"kind": "literal", "value": 1}, "missing": "error"},
            {"name": "x", "value": {"kind": "literal", "value": 2}, "missing": "error"},
        ], {})
    assert error.value.code == "duplicate_name"


def test_map_outputs_fixture_and_input_alias_isolation():
    data = fixture("map_outputs_success.json")
    mapped = map_outputs(**data)
    assert mapped == {"request_id": "r-1", "answer": [1, 2, 3]}
    data["result"]["answer"].append(4)
    assert mapped["answer"] == [1, 2, 3]


@pytest.mark.parametrize("mapping", [
    {"name": "x", "source": {"kind": "node", "path": []}},
    {"name": "x", "source": {"kind": "input", "path": [True]}},
    {"name": "x", "source": {"kind": "result", "path": []}, "extra": 1},
])
def test_map_outputs_rejects_forbidden_sources_and_bad_envelopes(mapping):
    with pytest.raises(SkeletonValidationError):
        map_outputs([mapping], {}, {})


@pytest.mark.parametrize("kind", [[], {}])
def test_non_string_output_source_kind_is_structured(kind):
    with pytest.raises(SkeletonValidationError) as error:
        map_outputs([{"name": "x", "source": {"kind": kind, "path": []}}], {}, {})
    assert error.value.code == "invalid_type"
    assert error.value.path == ("mappings", 0, "source", "kind")


def test_map_outputs_rejects_missing_and_duplicate_names():
    with pytest.raises(SkeletonValidationError, match="unavailable"):
        map_outputs([{"name": "x", "source": {"kind": "result", "path": ["none"]}}], {}, {})
    with pytest.raises(SkeletonValidationError) as error:
        map_outputs([
            {"name": "x", "source": {"kind": "input", "path": []}},
            {"name": "x", "source": {"kind": "input", "path": []}},
        ], {}, {})
    assert error.value.code == "duplicate_name"


class _Int(int):
    pass


class _String(str):
    pass


class _List(list):
    pass


class _Dict(dict):
    pass


@pytest.mark.parametrize("value", [
    float("nan"), float("inf"), ("tuple",), {1: "key"}, _Int(1), _String("value"), _List(), _Dict(),
])
def test_non_json_values_fail(value):
    with pytest.raises(SkeletonValidationError):
        resolve_inputs([{"name": "x", "value": {"kind": "literal", "value": value}, "missing": "error"}], {})


def test_huge_integer_is_detached_without_json_encoder_failure():
    huge_integer = 10 ** 5000
    assert resolve_inputs(
        [{"name": "x", "value": {"kind": "literal", "value": huge_integer}, "missing": "error"}], {}
    ) == {"x": huge_integer}


def test_resolve_inputs_rejects_aggregate_node_limit_overflow():
    bindings = [
        {"name": f"value_{index}", "value": {"kind": "ref", "node_id": "node", "path": ["value"]}, "missing": "error"}
        for index in range(100)
    ]
    with patch("unchain.skeleton.io.deepcopy") as copy_value:
        with pytest.raises(SkeletonValidationError) as error:
            resolve_inputs(bindings, {"node": {"value": list(range(1000))}})
    copy_value.assert_not_called()
    assert error.value.code == "limit_nodes"
    assert error.value.path[0] == "inputs"


def test_map_outputs_rejects_aggregate_node_limit_overflow():
    mappings = [
        {"name": f"value_{index}", "source": {"kind": "result", "path": ["value"]}}
        for index in range(100)
    ]
    with patch("unchain.skeleton.io.deepcopy") as copy_value:
        with pytest.raises(SkeletonValidationError) as error:
            map_outputs(mappings, {}, {"value": list(range(1000))})
    copy_value.assert_not_called()
    assert error.value.code == "limit_nodes"
    assert error.value.path[0] == "outputs"


def test_map_outputs_rejects_final_wrapped_depth_overflow():
    value = []
    for _ in range(64):
        value = [value]
    with pytest.raises(SkeletonValidationError) as error:
        map_outputs([{"name": "value", "source": {"kind": "result", "path": []}}], {}, value)
    assert error.value.code == "limit_depth"
    assert error.value.path[0] == "outputs"


def test_map_outputs_accepts_near_boundary_aggregate():
    mappings = [
        {"name": f"value_{index}", "source": {"kind": "result", "path": ["value"]}}
        for index in range(99)
    ]
    mapped = map_outputs(mappings, {}, {"value": list(range(1000))})
    assert len(mapped) == 99
    assert mapped["value_0"] == list(range(1000))


def test_duplicate_source_exports_are_independent_from_each_other_and_source():
    source = {"value": [1]}
    mapped = map_outputs([
        {"name": "first", "source": {"kind": "result", "path": ["value"]}},
        {"name": "second", "source": {"kind": "result", "path": ["value"]}},
    ], {}, source)
    mapped["first"].append(2)
    assert mapped == {"first": [1, 2], "second": [1]}
    assert source == {"value": [1]}


def test_depth_and_node_limits_are_structured_errors():
    deep = value = []
    for _ in range(65):
        child = []
        value.append(child)
        value = child
    with pytest.raises(SkeletonValidationError) as error:
        resolve_inputs([{"name": "x", "value": {"kind": "literal", "value": deep}, "missing": "error"}], {})
    assert error.value.code == "limit_depth"
    with pytest.raises(SkeletonValidationError) as error:
        resolve_inputs([{"name": "x", "value": {"kind": "literal", "value": list(range(100001))}, "missing": "error"}], {})
    assert error.value.code == "limit_nodes"
