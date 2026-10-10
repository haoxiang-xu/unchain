"""CP16 closed Hook Pool admission before runtime dispatch is wired."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hooks_config import attached_hooks, validate_hooks_pool_config


def _agent_document():
    document = json.loads(
        (Path(__file__).parent / "fixtures" / "skeleton_execution" / "if_rejoin_workflow.json").read_text()
    )
    document["kind"] = "agent"
    for node in document["nodes"]:
        if node["type"] == "code":
            node["type"] = "agent"
            node["config"]["kernel"] = {
                "schema_version": 1,
                "profile": {
                    "id": "test.profile", "provider": "openai", "model": "test-model",
                    "protocol": "openai-responses", "endpoint": "https://example.test/v1",
                    "credential_ref": "test.key", "context_window": 8192, "output_limit": 2048,
                    "capabilities": {"json_schema": True, "images": False, "files": False},
                },
                "prompt": "test", "max_iterations": 3, "output_tokens": 512,
                "history": "root", "overflow": "error", "json_schema": None,
            }
    return document


def _entry(*, hook_id="redact", phase="after_tool_call", tool_name="echo", order=20,
           force_approval=False):
    return {
        "id": hook_id,
        "order": order,
        "event": phase,
        "tool_name": tool_name,
        "workflow": {"workflow_id": "redaction-workflow", "revision_id": "rev-1"},
        "force_approval": force_approval,
    }


def _with_pool(document, entries):
    value = deepcopy(document)
    value["nodes"].append({"id": "hooks", "type": "hooks_pool", "config": {"hooks": entries}})
    kernel = next(node for node in value["nodes"] if node["type"] == "agent")
    value["edges"].append({
        "id": "attach-hooks", "kind": "attach",
        "source_node_id": kernel["id"], "source_port_id": "attach_bot",
        "target_node_id": "hooks", "target_port_id": "attach_top",
    })
    return value


def test_hook_pool_is_closed_is_detached_and_groups_by_attached_kernel():
    document = _with_pool(_agent_document(), [
        _entry(hook_id="format", phase="after_tool_call", tool_name=None, order=30),
        _entry(hook_id="gate", phase="before_tool_call", tool_name="echo", order=10,
               force_approval=True),
    ])

    prepared = prepare_agent_graph(document)
    hooks = attached_hooks(prepared.snapshot())
    assert [entry["id"] for entry in hooks["left"]] == ["gate", "format"]
    assert hooks["left"][0]["tool_name"] == "echo"
    assert hooks["left"][1]["tool_name"] is None
    hooks["left"][0]["workflow"]["revision_id"] = "mutated"
    assert attached_hooks(prepared.snapshot())["left"][0]["workflow"]["revision_id"] == "rev-1"


@pytest.mark.parametrize("mutate", [
    lambda value: value.update({"event": "after_model"}),
    lambda value: value.update({"tool_name": ""}),
    lambda value: value.update({"order": True}),
    lambda value: value.update({"force_approval": "yes"}),
    lambda value: value["workflow"].pop("revision_id"),
    lambda value: value.update({"unknown": True}),
])
def test_hook_pool_rejects_unknown_or_open_entries_before_any_runtime(mutate):
    entry = _entry()
    mutate(entry)
    with pytest.raises(SkeletonValidationError):
        validate_hooks_pool_config({"hooks": [entry]}, ("pool",))


def test_hook_ids_are_unique_but_equal_order_uses_stable_id_tiebreaker():
    first = _entry(hook_id="zeta", order=20)
    second = _entry(hook_id="alpha", order=20)
    entries = validate_hooks_pool_config({"hooks": [first, second]}, ("pool",))
    assert [entry["id"] for entry in entries] == ["alpha", "zeta"]
    second["id"] = "zeta"
    with pytest.raises(SkeletonValidationError) as error:
        validate_hooks_pool_config({"hooks": [first, second]}, ("pool",))
    assert error.value.code == "duplicate_name"
