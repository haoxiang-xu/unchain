"""Deterministic Tool/Skill handlers share the checked Agent execution seam."""
from copy import deepcopy
from dataclasses import replace

import pytest

from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_agent_graph, WorkflowNodeCompleted
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.tool_runtime import SkeletonToolRuntime, SkeletonSkillRuntime
from unchain.tools import Toolkit

from test_skeleton_tool_runtime import _tool_document, _skill_document
from test_skeleton_kernel_profile import _config


def _agent_plan(kind):
    identity = {"source": "skillpack", "source_id": "pack", "name": "review"}
    document = _tool_document(approval=False) if kind == "tool" else _skill_document(identity)
    document["kind"] = "agent"
    document["nodes"].insert(2, {"id": "kernel", "type": "agent", "config": {
        "kernel": _config(provider="openai", protocol="openai-responses"),
        "execution": {"inputs": [{"name": "value", "value": {"kind": "ref",
            "node_id": "step", "path": ["value"]}, "missing": "error"}],
            "outputs": [{"name": "value", "source": {"kind": "result", "path": ["context", "value"]}}]}}})
    document["nodes"][-1]["config"]["execution"]["inputs"][0]["value"]["node_id"] = "kernel"
    document["edges"][-1]["target_node_id"] = "kernel"
    document["edges"].append({"id": "kernel-end", "kind": "flow", "source_node_id": "kernel",
        "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"})
    return prepare_agent_graph(document), identity


@pytest.mark.parametrize("kind", ["tool", "skill"])
def test_official_deterministic_handler_runs_on_actual_agent_plan(tmp_path, kind):
    plan, identity = _agent_plan(kind)
    calls = []
    toolkit = Toolkit()
    toolkit.register(lambda text: calls.append(text) or {"echo": text}, name="echo")

    class Reader:
        def read_exact(self, requested):
            assert requested == identity
            return {"identity": deepcopy(identity), "body": "review instructions", "revision": "fixed"}

    handler = (SkeletonToolRuntime(plan, toolkit, "session", lambda: 0)
               if kind == "tool" else SkeletonSkillRuntime(plan, Reader()))
    assert handler.for_plan(plan).plan == plan
    journal = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects").bind_execution("execution")
    store = WorkflowExecutionStore(journal, plan, graph_id="root",
        attempt=AttemptRef(GenerationRef("execution", "generation"), "attempt"))
    runner = DurableWorkflowRunner(store, {"tool_call" if kind == "tool" else "skill_read": handler,
        "agent": lambda _context, inputs: WorkflowNodeCompleted({"message": "done", "context": inputs})})
    runner.create({"text": "hello"})
    result = runner.run()
    assert result.status == "completed"
    value = result.outputs["value"]
    if kind == "tool":
        assert value == {"result": {"echo": "hello"}, "ok": True, "error": None}
        assert calls == ["hello"]
    else:
        assert value["name"] == "review"
        assert value["content"] == "review instructions"


@pytest.mark.parametrize("kind", ["tool", "skill"])
def test_deterministic_handlers_reject_agent_plan_snapshot_identity_drift(kind):
    plan, _identity = _agent_plan(kind)
    forged = replace(plan, document_id="forged")
    with pytest.raises(SkeletonValidationError):
        if kind == "tool":
            SkeletonToolRuntime(forged, Toolkit(), "session", lambda: 0)
        else:
            SkeletonSkillRuntime(forged, object())
