from __future__ import annotations
import json
from pathlib import Path
import pytest
from unchain.skeleton import SkeletonValidationError, convert_legacy_definition, materialize_legacy_agent
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.execution import _validate_all
from unchain.skeleton import diagnose_skeleton_draft, validate_skeleton_for_execution

FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_legacy"
def fixture(name): return json.loads((FIXTURES / name).read_text())

def test_graphless_recipe_preserves_provenance_and_sentinel():
    source = fixture("graphless_recipe.json")
    result = convert_legacy_definition("recipe", source, resource_id="caller-id")
    assert result["id"] == "caller-id"
    assert result["nodes"][1]["config"]["prompt"] == {"kind":"builtin_developer_prompt"}
    assert result["nodes"][0]["config"]["legacy_source"]["document"] == source
    result["nodes"][1]["config"]["legacy_recipe"]["name"] = "changed"
    assert source["name"] == "Default"

def test_soul_defaults_and_invalids():
    result = convert_legacy_definition("soul", fixture("soul_defaults.json"), resource_id="soul-id")
    assert result["nodes"][1]["config"]["legacy_template"]["allowed_modes"] == ["delegate", "worker"]
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("other", {}, resource_id="id")
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("recipe", {}, resource_id="")
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("soul", "bad", resource_id="id")

def test_template_and_graph_recipe():
    template = {"name":"T","description":"Template","instructions":"Do work","allowed_modes":["worker"]}
    assert convert_legacy_definition("skeleton_template", template, resource_id="t")["nodes"][1]["config"]["prompt"]["text"] == "Do work"
    graph = fixture("graphless_recipe.json")
    graph["nodes"]=[{"id":"old-start","type":"start"},{"id":"a","type":"agent","override":{"prompt":"Hi","prompt_format":"soul"}},{"id":"p","type":"toolpool"},{"id":"old-end","type":"end"}]
    graph["edges"]=[{"id":"old-edge","source_node_id":"old-start","source_port_id":"in","target_node_id":"a","target_port_id":"out"}]
    result=convert_legacy_definition("recipe", graph, resource_id="g")
    assert result["nodes"][2]["type"] == "toolkit_pool"
    assert result["nodes"][0]["id"] == "old-start" and result["edges"][0]["id"] == "old-edge"
    assert result["edges"][0]["source_port_id"] == "out" and result["edges"][0]["target_port_id"] == "in"

def test_sentinel_is_exact_and_attach_inference_does_not_reverse():
    source = fixture("graphless_recipe.json")
    source["agent"]["prompt"] = "prefix {{USE_BUILTIN_DEVELOPER_PROMPT}} suffix"
    assert convert_legacy_definition("recipe", source, resource_id="x")["nodes"][1]["config"]["prompt"]["kind"] == "text"
    source["nodes"] = [{"id":"s","type":"start"},{"id":"a","type":"agent"},{"id":"p","type":"toolpool"},{"id":"e","type":"end"}]
    source["edges"] = [{"id":"attach","source_node_id":"a","source_port_id":"in","target_node_id":"p","target_port_id":"attach_top"}]
    output = convert_legacy_definition("recipe", source, resource_id="x")
    assert output["edges"][0] == {"id":"attach","kind":"attach","source_node_id":"a","source_port_id":"in","target_node_id":"p","target_port_id":"attach_top"}

def test_sentinel_prefix_and_port_classification_are_historical():
    source = fixture("graphless_recipe.json")
    source["agent"]["prompt"] = " {{#start.text#}} {{#start.images#}} {{#start.files#}} {{USE_BUILTIN_DEVELOPER_PROMPT}} "
    assert convert_legacy_definition("recipe", source, resource_id="x")["nodes"][1]["config"]["prompt"]["kind"] == "builtin_developer_prompt"
    source["agent"]["prompt"] = "{{#start.text#}}{{USE_BUILTIN_DEVELOPER_PROMPT}}"
    assert convert_legacy_definition("recipe", source, resource_id="x")["nodes"][1]["config"]["prompt"]["kind"] == "text"
    source["nodes"]=[{"id":"s","type":"start"},{"id":"a","type":"agent"},{"id":"e","type":"end"}]
    source["edges"]=[{"id":"f","source_node_id":"a","source_port_id":" in ","target_node_id":"s","target_port_id":" out "}]
    edge_value = convert_legacy_definition("recipe", source, resource_id="x")["edges"][0]
    assert (edge_value["source_node_id"], edge_value["target_node_id"], edge_value["source_port_id"], edge_value["target_port_id"]) == ("s", "a", " out ", " in ")

@pytest.mark.parametrize("bad", [[], {}])
def test_unhashable_prompt_formats_are_structured(bad):
    source = fixture("graphless_recipe.json")
    source["agent"]["prompt_format"] = bad
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("recipe", source, resource_id="x")

def test_template_metadata_and_bad_inputs_are_structured():
    valid = {"name":"T","description":"D","instructions":"I","model":"  ","allowed_tools":["x", " "]}
    metadata = convert_legacy_definition("skeleton_template", valid, resource_id="t")["nodes"][1]["config"]["legacy_template"]
    assert metadata["model"] is None and metadata["allowed_tools"] == ["x"]
    for bad in [{"name":"T","description":"D","instructions":"I","parallel_safe":"yes"}, {"name":"T","description":"D","instructions":"I","allowed_modes":[1]}]:
        with pytest.raises(SkeletonValidationError): convert_legacy_definition("skeleton_template", bad, resource_id="t")

def test_declarative_graph_template_and_explicit_soul_fixtures():
    graph = fixture("graph_recipe.json")
    converted = convert_legacy_definition("recipe", graph, resource_id="graph-id")
    assert converted["id"] == "graph-id" and converted["nodes"][0]["id"] == "start-old"
    assert converted["nodes"][2]["type"] == "toolkit_pool"
    assert converted["nodes"][1]["config"]["prompt"] == {"kind":"text","text":"Node"}
    assert converted["nodes"][0]["config"]["legacy_source"]["document"] == graph
    assert converted["nodes"][1]["config"]["legacy_fields"]["capability_refs"] == ["cap"]
    assert converted["edges"][0]["id"] == "edge-old" and converted["edges"][0]["source_node_id"] == "start-old"
    assert diagnose_skeleton_draft(converted) == () and validate_skeleton_for_execution(converted)["id"] == "graph-id"
    converted["nodes"][0]["config"]["legacy_source"]["document"]["name"] = "changed"
    assert graph["name"] == "Graph"
    template = convert_legacy_definition("skeleton_template", fixture("full_template.json"), resource_id="template-id")
    metadata = template["nodes"][1]["config"]["legacy_template"]
    assert metadata == {"allowed_modes":["delegate","worker"],"output_mode":"full_trace","memory_policy":"scoped_persistent","parallel_safe":False,"allowed_tools":["one","two"],"model":"model-y"}
    soul = convert_legacy_definition("soul", fixture("soul_explicit.json"), resource_id="soul-id")
    assert soul["nodes"][1]["config"]["legacy_template"]["allowed_tools"] == ["one", "two"]
    assert soul["nodes"][1]["config"]["legacy_template"]["model"] == "model-z"

def test_skeleton_prompt_nonfinite_json_is_structured_error():
    source = fixture("graphless_recipe.json")
    source["agent"] = {"prompt_format":"skeleton", "prompt":"{\"instructions\": \"x\", \"bad\": NaN}"}
    with pytest.raises(SkeletonValidationError):
        convert_legacy_definition("recipe", source, resource_id="x")

def test_graph_prompts_do_not_inherit_root_agent_prompt_or_format():
    source = fixture("graph_recipe.json")
    source.pop("agent")
    source["nodes"][1].pop("override")
    result = convert_legacy_definition("recipe", source, resource_id="x")
    assert result["nodes"][1]["config"]["prompt"] == {"kind":"text", "text":""}
    source["nodes"][1]["override"] = {"prompt": 42}
    assert convert_legacy_definition("recipe", source, resource_id="x")["nodes"][1]["config"]["prompt"] == {"kind":"text", "text":"42"}

def test_versioned_sources_are_not_legacy_templates_or_recipes():
    for kind in ("recipe", "skeleton_template"):
        with pytest.raises(SkeletonValidationError):
            convert_legacy_definition(kind, {"schema_version": 2}, resource_id="x")

def test_graph_root_agent_validation_and_empty_nodes_follow_legacy_rules():
    source = fixture("graph_recipe.json")
    source["agent"] = {"prompt_format": [], "prompt": "x"}
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("recipe", source, resource_id="x")
    source = fixture("graphless_recipe.json")
    source["nodes"] = []
    assert validate_skeleton_for_execution(convert_legacy_definition("recipe", source, resource_id="x"))["id"] == "x"
    source.pop("agent")
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("recipe", source, resource_id="x")

def test_huge_graph_override_prompt_is_structured_error():
    source = fixture("graph_recipe.json")
    source["nodes"][1]["override"] = {"prompt": 10 ** 5000}
    with pytest.raises(SkeletonValidationError) as error:
        convert_legacy_definition("recipe", source, resource_id="x")
    assert error.value.path == ("source", "nodes", 1, "override", "prompt")

def test_null_root_agent_and_whitespace_node_types_follow_old_normalization():
    source = fixture("graph_recipe.json")
    source["agent"] = None
    source["nodes"] = [{**item, "type": f" {item['type']} "} for item in source["nodes"]]
    result = convert_legacy_definition("recipe", source, resource_id="x")
    assert result["nodes"][1]["type"] == "agent" and validate_skeleton_for_execution(result)["id"] == "x"

def test_empty_nodes_keep_inert_legacy_edges_only_in_provenance():
    source = fixture("graphless_recipe.json")
    source["nodes"] = []
    source["edges"] = [{"id": "inert", "kind": None}]
    result = convert_legacy_definition("recipe", source, resource_id="x")
    assert validate_skeleton_for_execution(result)["id"] == "x"
    assert result["nodes"][0]["config"]["legacy_source"]["document"]["edges"] == source["edges"]
    assert [edge["id"] for edge in result["edges"]] == ["start-agent", "agent-end"]
    source["edges"][0]["id"] = "mutated"
    assert result["nodes"][0]["config"]["legacy_source"]["document"]["edges"][0]["id"] == "inert"
    source["edges"] = "bad"
    with pytest.raises(SkeletonValidationError): convert_legacy_definition("recipe", source, resource_id="x")

def test_empty_nodes_null_edges_match_legacy_normalization_and_preserve_null():
    source = fixture("graphless_recipe.json")
    source["nodes"] = []
    source["edges"] = None
    result = convert_legacy_definition("recipe", source, resource_id="x")
    assert validate_skeleton_for_execution(result)["id"] == "x"
    assert result["nodes"][0]["config"]["legacy_source"]["document"]["edges"] is None
    source["name"] = "changed"
    assert result["nodes"][1]["config"]["legacy_recipe"]["name"] == "Default"


def _kernel(model="test-model"):
    return {
        "schema_version": 1,
        "profile": {
            "id": "test", "provider": "openai", "model": model,
            "protocol": "openai-responses", "endpoint": "https://example.test/v1",
            "credential_ref": "test.key", "context_window": 8192,
            "output_limit": 2048,
            "capabilities": {"json_schema": True, "images": True, "files": True},
        },
        "prompt": "resolved builtin", "max_iterations": 8, "output_tokens": 512,
        "history": "root", "overflow": "error", "json_schema": None,
    }


def test_legacy_agent_materialization_is_strict_and_detached():
    source = fixture("graphless_recipe.json")
    converted = convert_legacy_definition("recipe", source, resource_id="legacy")
    materialized = materialize_legacy_agent(converted, _kernel())
    plan = prepare_agent_graph(materialized)
    start, agent, end = plan.snapshot()["nodes"]
    assert set(start["config"]) == {"execution"}
    assert set(agent["config"]) == {"execution", "kernel"}
    assert set(end["config"]) == {"execution"}
    assert [item["name"] for item in agent["config"]["execution"]["outputs"]] == ["message", "context"]
    assert [item["name"] for item in end["config"]["execution"]["outputs"]] == ["message", "context"]
    assert agent["config"]["kernel"]["prompt"] == "resolved builtin"
    materialized["name"] = "changed"
    assert converted["name"] == "Default"


def test_legacy_agent_materialization_requires_matching_explicit_model():
    source = fixture("graph_recipe.json")
    converted = convert_legacy_definition("recipe", source, resource_id="legacy")
    with pytest.raises(SkeletonValidationError) as error:
        materialize_legacy_agent(converted, _kernel("different-model"))
    assert error.value.code == "identity_mismatch"
    assert prepare_agent_graph(materialize_legacy_agent(converted, _kernel("model-x")))


def test_legacy_multi_kernel_preserves_root_config_and_prompt_dependencies():
    source = {
        "name": "Chain", "model": "openai:root-model", "max_iterations": 3,
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "first", "type": "agent", "override": {
                "prompt_format": "soul", "prompt": "First", "model": "override-model",
            }},
            {"id": "second", "type": "agent", "override": {
                "prompt_format": "soul", "prompt": "Review {{#first.output#}}",
            }},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"id": source + "-" + target, "kind": "flow",
             "source_node_id": source, "source_port_id": "out",
             "target_node_id": target, "target_port_id": "in"}
            for source, target in (("start", "first"), ("first", "second"), ("second", "end"))
        ],
    }
    converted = convert_legacy_definition("recipe", source, resource_id="chain")
    with pytest.raises(SkeletonValidationError) as mismatch:
        materialize_legacy_agent(converted, _kernel("override-model"))
    assert mismatch.value.code == "identity_mismatch"

    source["model"] = None
    source["nodes"][1]["override"]["model"] = ""
    converted = convert_legacy_definition("recipe", source, resource_id="chain")
    materialized = materialize_legacy_agent(converted, _kernel())
    agents = [node for node in materialized["nodes"] if node["type"] == "agent"]
    assert [node["config"]["kernel"]["max_iterations"] for node in agents] == [3, 3]
    second = agents[1]
    assert second["config"]["kernel"]["prompt"] == "Review {{#input.__legacy_prompt_0#}}"
    prompt_binding = next(
        item for item in second["config"]["execution"]["inputs"]
        if item["name"] == "__legacy_prompt_0"
    )
    assert prompt_binding["value"] == {
        "kind": "ref", "node_id": "first", "path": ["message"],
    }


@pytest.mark.parametrize("prompt", ["Author edited prompt", "", "  Use {{#input.x#}}  "])
def test_explicit_node_kernel_is_authoritative_over_legacy_values(prompt):
    source = fixture("graphless_recipe.json")
    source["model"] = "openai:old-model"
    source["max_iterations"] = 3
    source["agent"]["prompt"] = "Old legacy prompt"
    converted = convert_legacy_definition("recipe", source, resource_id="edited")
    chosen = _kernel("new-model")
    chosen["profile"].update({
        "id": "new-anthropic", "provider": "anthropic", "protocol": "anthropic",
        "endpoint": "https://chosen.example.test", "credential_ref": "chosen.key",
    })
    chosen.update(prompt=prompt, max_iterations=12)
    converted["nodes"][1]["config"]["kernel"] = chosen
    converted["nodes"][1]["config"]["execution"] = {
        "inputs": [{"name": "x", "value": {"kind": "ref", "node_id": "start", "path": ["text"]}, "missing": "error"}],
        "outputs": [{"name": "message", "source": {"kind": "result", "path": ["message"]}}],
    }
    original = json.loads(json.dumps(converted))
    fallback = _kernel("old-model")

    materialized = materialize_legacy_agent(converted, fallback)
    assert materialized["nodes"][1]["config"]["kernel"] == chosen
    assert _validate_all(prepare_agent_graph(materialized).snapshot(), agent_graph=True)
    assert converted == original
    materialized["nodes"][1]["config"]["kernel"]["profile"]["credential_ref"] = "changed"
    assert converted == original


def test_all_explicit_nodes_need_no_fallback_and_preserve_execution_surfaces():
    converted = convert_legacy_definition("recipe", fixture("graphless_recipe.json"), resource_id="explicit")
    start, agent, end = converted["nodes"]
    start["config"]["execution"] = {
        "outputs": [{"name": "question", "source": {"kind": "result", "path": ["text"]}}],
    }
    agent["config"]["kernel"] = {**_kernel(), "prompt": "Use {{#input.x#}}"}
    agent["config"]["execution"] = {
        "inputs": [{"name": "x", "value": {"kind": "ref", "node_id": "start", "path": ["question"]}, "missing": "error"}],
        "outputs": [{"name": "reply", "source": {"kind": "result", "path": ["message"]}}],
    }
    end["config"]["execution"] = {
        "inputs": [{"name": "final", "value": {"kind": "ref", "node_id": "agent", "path": ["reply"]}, "missing": "error"}],
        "outputs": [{"name": "answer", "source": {"kind": "input", "path": ["final"]}}],
    }
    # These compatibility-only values are no longer the author's Kernel.
    agent["config"]["prompt"] = {"kind": "unreadable old prompt"}
    agent["config"]["legacy_recipe"]["max_iterations"] = "old invalid value"

    materialized = materialize_legacy_agent(converted, None)
    for before, after in zip(converted["nodes"], materialized["nodes"]):
        assert after["config"]["execution"] == before["config"]["execution"]
    assert materialized["nodes"][1]["config"]["kernel"] == agent["config"]["kernel"]
    assert _validate_all(prepare_agent_graph(materialized).snapshot(), agent_graph=True)


def test_explicit_outputs_feed_generated_downstream_bindings():
    converted = convert_legacy_definition("recipe", fixture("graphless_recipe.json"), resource_id="surface")
    converted["nodes"][1]["config"]["kernel"] = _kernel()
    converted["nodes"][1]["config"]["execution"] = {
        "inputs": [],
        "outputs": [{"name": "reply", "source": {"kind": "result", "path": ["message"]}}],
    }
    materialized = materialize_legacy_agent(converted, None)
    end_execution = materialized["nodes"][-1]["config"]["execution"]
    assert end_execution["inputs"] == [{
        "name": "reply", "value": {"kind": "ref", "node_id": "agent", "path": ["reply"]}, "missing": "null",
    }]
    assert end_execution["outputs"] == [{"name": "reply", "source": {"kind": "input", "path": ["reply"]}}]
    assert _validate_all(prepare_agent_graph(materialized).snapshot(), agent_graph=True)


def test_mixed_explicit_and_fallback_nodes_keep_distinct_profiles_and_legacy_defaults():
    source = {
        "name": "Mixed", "model": "openai:old-model", "max_iterations": 3,
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "first", "type": "agent", "override": {"prompt_format": "soul", "prompt": "Old first"}},
            {"id": "second", "type": "agent", "override": {"prompt_format": "soul", "prompt": "Review {{#first.output#}}"}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"id": a + "-" + b, "kind": "flow", "source_node_id": a, "source_port_id": "out", "target_node_id": b, "target_port_id": "in"}
            for a, b in (("start", "first"), ("first", "second"), ("second", "end"))
        ],
    }
    converted = convert_legacy_definition("recipe", source, resource_id="mixed")
    chosen = _kernel("new-model")
    chosen["profile"].update({"id": "new", "provider": "anthropic", "protocol": "anthropic", "credential_ref": "new.key"})
    chosen.update(prompt="New first", max_iterations=12)
    converted["nodes"][1]["config"]["kernel"] = chosen
    fallback = _kernel("old-model")
    fallback["profile"].update(id="old", credential_ref="old.key")

    materialized = materialize_legacy_agent(converted, fallback)
    first, second = materialized["nodes"][1:3]
    assert first["config"]["kernel"] == chosen
    assert second["config"]["kernel"]["profile"] == fallback["profile"]
    assert second["config"]["kernel"]["max_iterations"] == 3
    assert second["config"]["kernel"]["prompt"] == "Review {{#input.__legacy_prompt_0#}}"
    assert _validate_all(prepare_agent_graph(materialized).snapshot(), agent_graph=True)
    with pytest.raises(SkeletonValidationError) as error:
        materialize_legacy_agent(converted, None)
    assert error.value.code == "invalid_type"
    assert error.value.path == ("legacy_kernel",)


@pytest.mark.parametrize("value,code,tail", [
    (None, "invalid_type", ()),
    ({**_kernel(), "temperature": 0.5}, "unknown_field", ("temperature",)),
    ({**_kernel(), "schema_version": 2}, "unknown_variant", ("schema_version",)),
])
def test_invalid_explicit_kernel_never_falls_back(value, code, tail):
    converted = convert_legacy_definition("recipe", fixture("graphless_recipe.json"), resource_id="invalid")
    converted["nodes"][1]["config"]["kernel"] = value
    with pytest.raises(SkeletonValidationError) as error:
        materialize_legacy_agent(converted, _kernel())
    assert error.value.code == code
    assert error.value.path == ("document", "nodes", 1, "config", "kernel") + tail


@pytest.mark.parametrize("execution", [None, 7, {"inputs": [], "outputs": [], "unknown": True}])
def test_invalid_explicit_execution_is_preserved_until_strict_preparation(execution):
    converted = convert_legacy_definition("recipe", fixture("graphless_recipe.json"), resource_id="invalid-io")
    converted["nodes"][1]["config"]["kernel"] = _kernel()
    converted["nodes"][1]["config"]["execution"] = execution
    materialized = materialize_legacy_agent(converted, None)
    assert materialized["nodes"][1]["config"]["execution"] == execution
    with pytest.raises(SkeletonValidationError) as error:
        _validate_all(prepare_agent_graph(materialized).snapshot(), agent_graph=True)
    assert error.value.path[:5] == ("document", "nodes", 1, "config", "execution")


def test_untouched_inline_child_keeps_own_prompt_and_fallback_iterations():
    converted = convert_legacy_definition("skeleton_template", {
        "name": "Child", "description": "Child", "instructions": "Child own prompt",
    }, resource_id="child")
    fallback = {**_kernel(), "prompt": "Parent edited prompt", "max_iterations": 12}
    materialized = materialize_legacy_agent(converted, fallback)
    assert materialized["nodes"][1]["config"]["kernel"]["prompt"] == "Child own prompt"
    assert materialized["nodes"][1]["config"]["kernel"]["max_iterations"] == 12
    assert prepare_agent_graph(materialized)
