"""Pure conversion of historical recipe and template definitions."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from .io import _error, _validate_json
from .kernel_config import validate_kernel_config
from .schema import validate_skeleton_document

_SENTINEL = "{{USE_BUILTIN_DEVELOPER_PROMPT}}"
_TYPES = {"start", "end", "agent", "toolkit_pool", "subagent_pool"}


_START_PREFIX = re.compile(r"^\s*\{\{#start\.text#\}\}\s*\{\{#start\.images#\}\}\s*\{\{#start\.files#\}\}\s*")
_LEGACY_VARIABLE = re.compile(r"\{\{#([^.}]+)\.([^#}]+)#\}\}")


def _port_kind(value: str) -> str:
    port = value.strip()
    if port in {"in", "out"}:
        return port
    return "attach" if port in {"attach_top", "attach_bot"} else ""


def _text_prompt(value: Any, path: tuple[str | int, ...], prompt_format: Any = "soul", *, recipe_prompt: bool = False) -> dict[str, Any]:
    if type(value) is not str:
        _error("invalid_type", path, "prompt must be a string")
    if recipe_prompt and _START_PREFIX.sub("", value).strip() == _SENTINEL:
        return {"kind": "builtin_developer_prompt"}
    if prompt_format == "skeleton":
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            _error("invalid_value", path, "skeleton prompt must be valid JSON")
        _validate_json(parsed, path)
        if type(parsed) is not dict or type(parsed.get("instructions")) is not str:
            _error("invalid_value", path, "skeleton prompt requires string instructions")
        return {"kind": "text", "text": parsed["instructions"]}
    return {"kind": "text", "text": value}


def _template_metadata(source: dict[str, Any], path: tuple[str | int, ...]) -> dict[str, Any]:
    modes = source.get("allowed_modes", ["delegate", "worker"])
    if type(modes) is not list or any(type(item) is not str for item in modes):
        _error("invalid_type", path + ("allowed_modes",), "allowed_modes must be a list of strings")
    output = source.get("output_mode", "summary")
    memory = source.get("memory_policy", "ephemeral")
    parallel = source.get("parallel_safe", True)
    tools = source.get("allowed_tools")
    model = source.get("model")
    if type(output) is not str or type(memory) is not str or type(parallel) is not bool:
        _error("invalid_type", path, "template metadata is invalid")
    if tools is not None and (type(tools) is not list or any(type(item) is not str for item in tools)):
        _error("invalid_type", path + ("allowed_tools",), "allowed_tools must be null or a list of strings")
    if model is not None and type(model) is not str:
        _error("invalid_type", path + ("model",), "model must be null or a string")
    return {"description": source["description"], "allowed_modes": modes, "output_mode": output, "memory_policy": memory, "parallel_safe": parallel, "allowed_tools": None if tools is None else [item.strip() for item in tools if item.strip()], "model": model.strip() or None if type(model) is str else None}


def _fixed(resource_id: str, name: str, provenance: dict[str, Any], prompt: dict[str, Any], template: dict[str, Any] | None = None) -> dict[str, Any]:
    agent_config: dict[str, Any] = {"prompt": prompt}
    if template is not None:
        agent_config["legacy_template"] = template
    result = {"schema_version": 1, "kind": "agent", "id": resource_id, "name": name, "nodes": [
        {"id": "start", "type": "start", "config": {"legacy_source": provenance}},
        {"id": "agent", "type": "agent", "config": agent_config},
        {"id": "end", "type": "end", "config": {}},
    ], "edges": [
        {"id": "start-agent", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "agent", "target_port_id": "in"},
        {"id": "agent-end", "kind": "flow", "source_node_id": "agent", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
    ]}
    if template is not None:
        result["description"] = template["description"]
    return result


def _validate_inert_edges(edges: Any) -> None:
    if edges is None:
        return
    if type(edges) is not list:
        _error("invalid_type", ("source", "edges"), "edges must be a list")
    seen: set[str] = set()
    for index, edge in enumerate(edges):
        if type(edge) is not dict or type(edge.get("id")) is not str or not edge["id"]:
            _error("invalid_type", ("source", "edges", index), "legacy edge is invalid")
        if edge["id"] in seen or edge.get("kind") is not None and type(edge.get("kind")) is not str:
            _error("invalid_value", ("source", "edges", index), "legacy edge is invalid")
        seen.add(edge["id"])


def _recipe(source: dict[str, Any], resource_id: str) -> dict[str, Any]:
    if "name" not in source:
        _error("missing_field", ("source", "name"), "required recipe field is missing")
    if type(source["name"]) is not str or not source["name"]:
        _error("invalid_value", ("source", "name"), "recipe name must be non-empty")
    if "description" in source and type(source["description"]) is not str:
        _error("invalid_type", ("source", "description"), "description must be a string")
    if source.get("model") is not None and type(source.get("model")) is not str:
        _error("invalid_type", ("source", "model"), "model must be string or null")
    maximum = source.get("max_iterations")
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        _error("invalid_value", ("source", "max_iterations"), "max_iterations must be positive integer or null")
    graph_present = "nodes" in source
    graph_nodes, graph_edges = source.get("nodes"), source.get("edges", [])
    has_graph = graph_present and type(graph_nodes) is list and bool(graph_nodes)
    supplied_agent = source.get("agent")
    if supplied_agent is not None and (type(supplied_agent) is not dict or type(supplied_agent.get("prompt_format")) is not str or supplied_agent.get("prompt_format") not in {"soul", "skeleton"} or type(supplied_agent.get("prompt", "")) is not str):
        _error("invalid_type", ("source", "agent"), "agent prompt object is required")
    if not has_graph and type(supplied_agent) is not dict:
        _error("invalid_type", ("source", "agent"), "agent prompt object is required")
    if type(source.get("toolkits", [])) is not list or type(source.get("subagent_pool", [])) is not list or type(source.get("merge_with_user_selected", True)) is not bool:
        _error("invalid_type", ("source",), "recipe compatibility fields are invalid")
    root_agent = supplied_agent if type(supplied_agent) is dict else None
    root_prompt = _text_prompt(root_agent.get("prompt", ""), ("source", "agent", "prompt"), root_agent.get("prompt_format", "soul"), recipe_prompt=True) if root_agent and not has_graph else {"kind": "text", "text": ""}
    provenance = {"format": "recipe", "document": source}
    if not has_graph and not graph_present:
        result = _fixed(resource_id, source["name"], provenance, root_prompt)
        result["nodes"][1]["config"]["legacy_recipe"] = source
        return result
    if graph_present and type(graph_nodes) is list and not graph_nodes:
        _validate_inert_edges(graph_edges)
        result = _fixed(resource_id, source["name"], provenance, root_prompt)
        result["nodes"][1]["config"]["legacy_recipe"] = source
        return result
    if type(graph_nodes) is not list or type(graph_edges) is not list or not graph_nodes:
        _error("invalid_type", ("source", "nodes"), "recipe graph nodes and edges must be non-empty arrays")
    nodes = []
    first_agent = None
    start_index = None
    start_count = end_count = 0
    seen_nodes: set[str] = set()
    for index, old in enumerate(graph_nodes):
        if type(old) is not dict or type(old.get("id")) is not str:
            _error("invalid_type", ("source", "nodes", index), "legacy node requires id and type")
        try:
            node_type = str(old.get("type") or "").strip()
        except ValueError:
            _error("invalid_value", ("source", "nodes", index, "type"), "legacy node type is invalid")
        node_type = "toolkit_pool" if node_type == "toolpool" else node_type
        if not old["id"] or old["id"] in seen_nodes:
            _error("invalid_value", ("source", "nodes", index, "id"), "node id is invalid")
        seen_nodes.add(old["id"])
        if node_type not in _TYPES:
            _error("unknown_variant", ("source", "nodes", index, "type"), "unsupported legacy node type")
        fields = {key: value for key, value in old.items() if key not in {"id", "type"}}
        config: dict[str, Any] = {"legacy_fields": fields}
        if node_type == "agent":
            override = old.get("override") if type(old.get("override")) is dict else {}
            prompt_format = override.get("prompt_format", "soul")
            if "prompt_format" in override and (type(prompt_format) is not str or prompt_format not in {"soul", "skeleton"}):
                _error("invalid_value", ("source", "nodes", index, "override", "prompt_format"), "prompt format is invalid")
            raw_prompt = override.get("prompt", "")
            if type(raw_prompt) is not str:
                try:
                    raw_prompt = str(raw_prompt or "")
                except ValueError:
                    _error("invalid_value", ("source", "nodes", index, "override", "prompt"), "legacy prompt projection is invalid")
            config["prompt"] = _text_prompt(raw_prompt, ("source", "nodes", index, "override", "prompt"), prompt_format, recipe_prompt=True)
            if first_agent is None:
                first_agent = len(nodes)
        if node_type == "start":
            start_index = len(nodes)
            start_count += 1
        if node_type == "end":
            end_count += 1
        nodes.append({"id": old["id"], "type": node_type, "config": config})
    if first_agent is None:
        _error("invalid_value", ("source", "nodes"), "recipe graph requires agent node")
    if start_count != 1 or end_count != 1:
        _error("invalid_value", ("source", "nodes"), "recipe graph requires start and end nodes")
    nodes[first_agent]["config"]["legacy_recipe"] = source
    nodes[start_index]["config"]["legacy_source"] = provenance
    edges = []
    seen_edges: set[str] = set()
    for index, old in enumerate(graph_edges):
        if type(old) is not dict or any(type(old.get(key)) is not str or not old.get(key) for key in ("id", "source_node_id", "source_port_id", "target_node_id", "target_port_id")):
            _error("invalid_type", ("source", "edges", index), "legacy edge is invalid")
        if old["id"] in seen_edges or old["source_node_id"] not in seen_nodes or old["target_node_id"] not in seen_nodes or old.get("kind") is not None and type(old.get("kind")) is not str:
            _error("invalid_value", ("source", "edges", index), "legacy edge is invalid")
        seen_edges.add(old["id"])
        source_id, target_id, source_port, target_port = old["source_node_id"], old["target_node_id"], old["source_port_id"], old["target_port_id"]
        source_kind, target_kind = _port_kind(source_port), _port_kind(target_port)
        kind = old.get("kind") if old.get("kind") in {"flow", "attach"} else ("attach" if "attach" in {source_kind, target_kind} else "flow")
        if kind == "flow" and source_kind == "in" and target_kind == "out":
            source_id, target_id, source_port, target_port = target_id, source_id, target_port, source_port
        edges.append({"id": old["id"], "kind": kind, "source_node_id": source_id, "source_port_id": source_port, "target_node_id": target_id, "target_port_id": target_port})
    return {"schema_version": 1, "kind": "agent", "id": resource_id, "name": source["name"], "nodes": nodes, "edges": edges}


def _soul(source: str, resource_id: str) -> dict[str, Any]:
    if not source.startswith("---"):
        _error("invalid_value", ("source",), "soul requires frontmatter")
    parts = source.split("\n---", 1)
    if len(parts) != 2:
        _error("invalid_value", ("source",), "soul frontmatter is not terminated")
    header, body = parts[0][3:].lstrip("\n"), parts[1].lstrip("\n")
    metadata: dict[str, Any] = {}
    for line in header.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            _error("invalid_value", ("source",), "soul frontmatter line is invalid")
        key, value = (part.strip() for part in line.split(":", 1))
        if not key:
            _error("invalid_value", ("source",), "soul frontmatter key is empty")
        metadata[key] = [item.strip() for item in value[1:-1].split(",") if item.strip()] if value.startswith("[") and value.endswith("]") else value
    if type(metadata.get("name")) is not str or not metadata["name"].strip() or type(metadata.get("description")) is not str or not metadata["description"].strip() or not body.strip():
        _error("invalid_value", ("source",), "soul requires name, description, and body")
    if "tools" in metadata and type(metadata["tools"]) is not list:
        _error("invalid_type", ("source", "tools"), "soul tools must be a list")
    if "model" in metadata and type(metadata["model"]) is not str:
        _error("invalid_type", ("source", "model"), "soul model must be a string")
    template = {"description": metadata["description"], "allowed_modes": ["delegate", "worker"], "output_mode": "summary", "memory_policy": "ephemeral", "parallel_safe": True, "allowed_tools": metadata.get("tools"), "model": metadata.get("model", "").strip() or None}
    return _fixed(resource_id, metadata["name"].strip(), {"format": "soul", "text": source}, {"kind": "text", "text": body.strip()}, template)


def convert_legacy_definition(source_kind: Any, source: Any, *, resource_id: Any) -> dict[str, Any]:
    if type(resource_id) is not str or not resource_id:
        _error("invalid_value", ("resource_id",), "resource_id must be a non-empty string")
    if type(source_kind) is not str or source_kind not in {"recipe", "skeleton_template", "soul"}:
        _error("unknown_variant", ("source_kind",), "unknown legacy source kind")
    if source_kind == "soul":
        if type(source) is not str:
            _error("invalid_type", ("source",), "soul source must be text")
        result = _soul(source, resource_id)
    else:
        _validate_json(source, ("source",))
        if type(source) is not dict:
            _error("invalid_type", ("source",), "legacy source must be an object")
        if "schema_version" in source:
            _error("invalid_value", ("source", "schema_version"), "new skeleton is not a legacy source")
        if source_kind == "recipe":
            result = _recipe(source, resource_id)
            if "description" in source:
                result["description"] = source["description"]
        else:
            if type(source.get("name")) is not str or type(source.get("instructions")) is not str:
                _error("invalid_value", ("source",), "template requires name and instructions")
            if not source["name"].strip() or type(source.get("description")) is not str or not source["description"].strip() or not source["instructions"].strip():
                _error("invalid_value", ("source",), "template name, description, and instructions are required")
            metadata = _template_metadata(source, ("source",))
            result = _fixed(resource_id, source["name"].strip(), {"format": "skeleton_template", "document": source}, {"kind": "text", "text": source["instructions"].strip()}, metadata)
    return validate_skeleton_document(result)


def _legacy_output_names(node: dict[str, Any]) -> list[str]:
    execution = node.get("config", {}).get("execution")
    outputs = execution.get("outputs") if type(execution) is dict else None
    if type(outputs) is list and all(type(item) is dict and type(item.get("name")) is str for item in outputs):
        return [item["name"] for item in outputs]
    if node["type"] == "start":
        fields = node["config"].get("legacy_fields", {})
        outputs = fields.get("outputs") if type(fields) is dict else None
        if type(outputs) is list:
            names = [item.get("name") for item in outputs if type(item) is dict]
            if names and all(type(name) is str and name for name in names) and len(names) == len(set(names)):
                return names
        return ["text", "images", "files"]
    return ["message", "context"]


def _execution_outputs(names: list[str], kind: str) -> list[dict[str, Any]]:
    return [{"name": name, "source": {"kind": kind, "path": [name]}} for name in names]


def _execution_inputs(source_id: str, names: list[str]) -> list[dict[str, Any]]:
    return [{
        "name": name,
        "value": {"kind": "ref", "node_id": source_id, "path": [name]},
        "missing": "error" if name in {"text", "message"} else "null",
    } for name in names]


def _model_matches(requested: str, profile: dict[str, Any]) -> bool:
    value = requested.strip()
    if ":" not in value:
        return value == profile["model"]
    provider, model = value.split(":", 1)
    return provider == profile["provider"] and model == profile["model"]


def _legacy_prompt_bindings(text: str, nodes: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    bindings: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        source_id, legacy_name = match.group(1), match.group(2)
        source = nodes.get(source_id)
        if source is None:
            return ""
        name = "message" if source["type"] == "agent" and legacy_name == "output" else legacy_name
        if name not in _legacy_output_names(source):
            return ""
        binding_name = f"__legacy_prompt_{len(bindings)}"
        bindings.append({
            "name": binding_name,
            "value": {"kind": "ref", "node_id": source_id, "path": [name]},
            "missing": "error",
        })
        return "{{#input." + binding_name + "#}}"

    return _LEGACY_VARIABLE.sub(replace, text), bindings


def materialize_legacy_agent(document: Any, kernel: Any) -> dict[str, Any]:
    """Produce one strict, detached Agent execution snapshot from legacy data.

    An explicit node Kernel is the authored configuration, including its
    model, prompt and iteration limit. Only nodes without one project their
    legacy settings onto the host-supplied fallback ``kernel``. The fallback
    is not needed for an entirely configured graph. Explicit execution
    bindings are retained for the normal strict execution validator.
    """
    checked = validate_skeleton_document(document)
    if checked["kind"] != "agent":
        _error("wrong_kind", ("document", "kind"), "legacy materialization requires an Agent")
    fixed_kernel = None
    flow = [edge for edge in checked["edges"] if edge["kind"] == "flow"]
    incoming: dict[str, str] = {}
    for edge in flow:
        target = edge["target_node_id"]
        if target in incoming:
            _error("invalid_value", ("document", "edges"), "legacy Agent must have one flow predecessor")
        incoming[target] = edge["source_node_id"]
    by_id = {node["id"]: node for node in checked["nodes"]}
    legacy_recipe = next((
        node["config"].get("legacy_recipe")
        for node in checked["nodes"]
        if type(node["config"].get("legacy_recipe")) is dict
    ), None)
    outputs_by_id: dict[str, list[str]] = {}
    result = deepcopy(checked)
    for index, node in enumerate(result["nodes"]):
        node_type = node["type"]
        if node_type in {"toolkit_pool", "subagent_pool", "hooks_pool"}:
            continue
        if node_type == "start":
            names = _legacy_output_names(node)
            execution = node["config"]["execution"] if "execution" in node["config"] else {
                "outputs": _execution_outputs(names, "result"),
            }
            node["config"] = {"execution": execution}
            outputs_by_id[node["id"]] = _legacy_output_names(node)
            continue
        source_id = incoming.get(node["id"])
        if source_id is None or source_id not in by_id:
            _error("missing_value", ("document", "nodes", index), "legacy execution predecessor is missing")
        source_names = outputs_by_id.get(source_id) or _legacy_output_names(by_id[source_id])
        inputs = _execution_inputs(source_id, source_names)
        if node_type == "agent":
            if "kernel" in node["config"]:
                # Presence is intentional, including an empty prompt or the
                # default iteration limit. Invalid authored data must never
                # silently fall back to a different model or configuration.
                selected = validate_kernel_config(node["config"]["kernel"],
                    ("document", "nodes", index, "config", "kernel"))
            else:
                if fixed_kernel is None:
                    fixed_kernel = validate_kernel_config(kernel, ("legacy_kernel",))
                selected = deepcopy(fixed_kernel)
                prompt = node["config"].get("prompt")
                if type(prompt) is not dict or prompt.get("kind") not in {"text", "builtin_developer_prompt"}:
                    _error("invalid_shape", ("document", "nodes", index, "config", "prompt"), "legacy prompt is invalid")
                if prompt["kind"] == "text":
                    text = _START_PREFIX.sub("", prompt["text"]).strip()
                    selected["prompt"], prompt_inputs = _legacy_prompt_bindings(text, by_id)
                    inputs.extend(prompt_inputs)
                legacy = legacy_recipe or node["config"].get("legacy_template")
                fields = node["config"].get("legacy_fields", {})
                override = fields.get("override", {}) if type(fields) is dict else {}
                override_model = override.get("model") if type(override) is dict else None
                requested_model = (
                    override_model if type(override_model) is str and override_model.strip()
                    else legacy.get("model") if type(legacy) is dict else None
                )
                if (type(requested_model) is str and requested_model.strip()
                        and not _model_matches(requested_model, selected["profile"])):
                    _error("identity_mismatch", ("document", "nodes", index, "config", "model"), "selected model differs from legacy Agent")
                maximum = legacy.get("max_iterations") if type(legacy) is dict else None
                if maximum is not None:
                    selected["max_iterations"] = maximum
            names = ["message", "context"]
            execution = node["config"]["execution"] if "execution" in node["config"] else {
                "inputs": inputs, "outputs": _execution_outputs(names, "result"),
            }
            node["config"] = {
                "execution": execution,
                "kernel": selected,
            }
            outputs_by_id[node["id"]] = _legacy_output_names(node)
        elif node_type == "end":
            execution = node["config"]["execution"] if "execution" in node["config"] else {
                "inputs": inputs, "outputs": _execution_outputs(source_names, "input"),
            }
            node["config"] = {"execution": execution}
            outputs_by_id[node["id"]] = _legacy_output_names(node)
        else:
            _error("unknown_variant", ("document", "nodes", index, "type"), "legacy Agent contains an unsupported execution node")
    return validate_skeleton_document(result)
