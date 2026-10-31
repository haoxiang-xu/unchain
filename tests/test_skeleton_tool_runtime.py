"""CP12 durable Tool Call and Skill Read integration tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from unchain.interaction.durable import InteractionIntegrityError
from unchain.journal import AttemptRef, GenerationRef, JournalConflictError
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import (
    prepare_workflow,
    SkeletonSkillRuntime,
    SkeletonToolRuntime,
    SkeletonValidationError,
)
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.tools.toolkit import Toolkit


def _start_outputs():
    return [{"name": "text", "source": {"kind": "result", "path": ["text"]}}]


def _end_inputs():
    return [{"name": "value", "value": {"kind": "ref", "node_id": "step", "path": ["value"]}, "missing": "error"}]


def _end_outputs():
    return [{"name": "value", "source": {"kind": "input", "path": ["value"]}}]


def _document(kind: str, step_type: str, config: dict) -> dict:
    return {
        "schema_version": 1,
        "kind": kind,
        "id": "workflow-tool",
        "name": "Workflow tool",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": _start_outputs()}}},
            {"id": "step", "type": step_type, "config": config},
            {"id": "end", "type": "end", "config": {"execution": {"inputs": _end_inputs(), "outputs": _end_outputs()}}},
        ],
        "edges": [
            {"id": "start-step", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "step", "target_port_id": "in"},
            {"id": "step-end", "kind": "flow", "source_node_id": "step", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


def _tool_document(*, approval: bool = True, on_error: str = "stop") -> dict:
    return _document("workflow", "tool_call", {
        "execution": {
            "inputs": [{"name": "text", "value": {"kind": "ref", "node_id": "start", "path": ["text"]}, "missing": "error"}],
            "outputs": [{"name": "value", "source": {"kind": "result", "path": []}}],
        },
        "tool": {"name": "echo", "on_error": on_error, "approval_timeout_ms": 1000},
    })


def _skill_document(identity: dict) -> dict:
    document = _document("workflow", "skill_read", {
        "execution": {
            "inputs": [],
            "outputs": [{"name": "value", "source": {"kind": "result", "path": []}}],
        },
        "skill": identity,
    })
    document["nodes"][0]["config"]["execution"]["outputs"] = []
    return document


def _runner(tmp_path: Path, document: dict, handlers: dict, *, graph: str = "graph") -> DurableWorkflowRunner:
    repo = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite3", object_directory=tmp_path / "objects").bind_execution("execution")
    plan = prepare_workflow(document)
    store = WorkflowExecutionStore(repo, plan, graph_id=graph, attempt=AttemptRef(GenerationRef("execution", "generation"), "attempt"))
    return DurableWorkflowRunner(store, handlers)


def _toolkit(calls: list[dict], *, approval: bool) -> Toolkit:
    toolkit = Toolkit()
    @toolkit.tool(name="echo", requires_confirmation=approval)
    def echo(text: str):
        calls.append({"text": text})
        return {"echo": text}
    return toolkit


def test_confirmable_tool_waits_then_executes_once_and_exposes_detached_schema(tmp_path):
    calls: list[dict] = []
    toolkit = _toolkit(calls, approval=True)
    runtime = SkeletonToolRuntime(prepare_workflow(_tool_document()), toolkit, session_id="session", clock_ms=lambda: 100)
    specs = runtime.tool_schemas()
    assert specs[0]["name"] == "echo"
    specs[0]["parameters"]["properties"]["text"]["type"] = "changed"
    assert runtime.tool_schemas()[0]["parameters"]["properties"]["text"]["type"] == "string"

    run = _runner(tmp_path, _tool_document(), {"tool_call": runtime})
    run.create({"text": "hello"})
    waiting = run.run()
    assert waiting.status == "waiting" and calls == []
    active = run.store.load().state["active"]
    request = run.store.read_json(active["wait"]["request_ref"])
    assert request["interaction"]["kind"] == "tool_approval"
    assert "session" not in json.dumps(request["interaction"]["subject"])

    resumed = run.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "ok"}, node_invocation_id=active["node_invocation_id"])
    assert resumed.status == "result_received" and calls == [{"text": "hello"}]
    assert run.run().outputs == {"value": {"result": {"echo": "hello"}, "ok": True, "error": None}}


def test_tool_approval_receipt_is_durable_and_cold_resume_never_reexecutes(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=True)
    first = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    first.create({"text": "once"})
    first.run()
    active = first.store.load().state["active"]
    original_save = first.store.save
    class _Crash(BaseException):
        pass
    def crash_after_receipt(state, **kwargs):
        if state["status"] == "running" and state["active"]["wait"] is None:
            raise _Crash()
        return original_save(state, **kwargs)
    first.store.save = crash_after_receipt
    with pytest.raises(_Crash):
        first.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "ok"}, node_invocation_id=active["node_invocation_id"])
    assert calls == []
    cold = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    active = cold.store.load().state["active"]
    result = cold.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "ok"}, node_invocation_id=active["node_invocation_id"])
    assert result.status == "result_received" and calls == [{"text": "once"}]
    assert cold.run().status == "completed"


@pytest.mark.parametrize("on_error, expected_status", [("stop", "failed"), ("continue", "completed")])
def test_tool_rejection_is_never_success_and_follows_the_configured_error_policy(tmp_path, on_error, expected_status):
    calls: list[dict] = []
    document = _tool_document(on_error=on_error)
    toolkit = _toolkit(calls, approval=True)
    run = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    run.create({"text": "no"})
    run.run()
    active = run.store.load().state["active"]
    assert run.resume_tool_interaction(active["wait"]["id"], {"approved": False, "modified_arguments": None, "reason": "no"}, node_invocation_id=active["node_invocation_id"]).status == "result_received"
    result = run.run()
    assert calls == [] and result.status == expected_status
    if on_error == "continue":
        assert result.outputs == {"value": {"result": {"denied": True, "tool": "echo", "reason": "no"}, "ok": False, "error": {"code": "denied"}}}


def test_duplicate_or_expired_tool_approval_cannot_dispatch(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=True)
    run = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    run.create({"text": "once"})
    run.run()
    active = run.store.load().state["active"]
    run.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "ok"}, node_invocation_id=active["node_invocation_id"])
    with pytest.raises(Exception):
        run.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "duplicate"}, node_invocation_id=active["node_invocation_id"])
    assert calls == [{"text": "once"}]

    expired = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)}, graph="expired")
    expired.create({"text": "never"})
    expired.run()
    active = expired.store.load().state["active"]
    assert expired.expire_tool_interaction(active["wait"]["id"], node_invocation_id=active["node_invocation_id"], now_ms=1101).status == "failed"
    assert calls == [{"text": "once"}]


@pytest.mark.parametrize("on_error, expected_status", [("stop", "failed"), ("continue", "completed")])
def test_tool_runtime_error_is_a_known_result_not_a_retryable_success(tmp_path, on_error, expected_status):
    document = _tool_document(on_error=on_error)
    toolkit = Toolkit()
    @toolkit.tool(name="echo", requires_confirmation=False)
    def echo(text: str):
        raise RuntimeError("expected failure")
    run = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    run.create({"text": "error"})
    result = run.run()
    assert result.status == expected_status
    if on_error == "continue":
        assert result.outputs == {"value": {"result": {"error": "expected failure", "tool": "echo"}, "ok": False, "error": {"code": "tool_error"}}}


def test_waiting_tool_cancellation_prevents_later_approval_dispatch(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=True)
    run = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    run.create({"text": "cancelled"})
    run.run()
    active = run.store.load().state["active"]
    assert run.cancel().status == "cancelled"
    with pytest.raises(Exception):
        run.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "late"}, node_invocation_id=active["node_invocation_id"])
    assert calls == []


def test_cancellation_during_policy_resolution_prevents_tool_dispatch(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=False)
    entered, release = Event(), Event()

    def resolve(arguments, context):
        entered.set()
        assert release.wait(10)
        return {"requires_confirmation": False}

    toolkit.get("echo").confirmation_resolver = resolve
    run = _runner(tmp_path, document, {
        "tool_call": SkeletonToolRuntime(
            prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100,
        ),
    })
    run.create({"text": "cancel-before-effect"})
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(run.run)
        try:
            assert entered.wait(10) and calls == []
            assert run.cancel().status == "cancelled"
        finally:
            release.set()
        with pytest.raises(JournalConflictError):
            pending.result(timeout=10)
    assert run.inspect().status == "cancelled" and calls == []


def test_cancellation_during_approval_revalidation_prevents_tool_dispatch(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=True)
    entered, release = Event(), Event()
    resolutions = 0

    def resolve(arguments, context):
        nonlocal resolutions
        resolutions += 1
        if resolutions > 1:
            entered.set()
            assert release.wait(10)
        return {"requires_confirmation": True}

    toolkit.get("echo").confirmation_resolver = resolve
    run = _runner(tmp_path, document, {
        "tool_call": SkeletonToolRuntime(
            prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100,
        ),
    })
    run.create({"text": "cancel-approval"})
    assert run.run().status == "waiting"
    active = run.store.load().state["active"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(
            run.resume_tool_interaction,
            active["wait"]["id"],
            {"approved": True, "modified_arguments": None, "reason": "ok"},
            node_invocation_id=active["node_invocation_id"],
        )
        try:
            assert entered.wait(10) and calls == []
            assert run.cancel().status == "cancelled"
        finally:
            release.set()
        with pytest.raises(JournalConflictError):
            pending.result(timeout=10)
    assert run.inspect().status == "cancelled" and calls == []


def test_tool_approval_wait_rejects_generic_completion(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=True)
    run = _runner(tmp_path, document, {
        "tool_call": SkeletonToolRuntime(
            prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100,
        ),
    })
    run.create({"text": "pending"})
    assert run.run().status == "waiting"
    active = run.store.load().state["active"]
    with pytest.raises(SkeletonValidationError, match="official interaction resume"):
        run.complete_wait(
            active["wait"]["id"],
            {"result": {"echo": "fabricated"}, "ok": True, "error": None},
            node_invocation_id=active["node_invocation_id"],
        )
    assert run.run().status == "waiting" and calls == []

def test_tool_changed_policy_invalidates_the_stored_approval_request(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    toolkit = _toolkit(calls, approval=True)
    run = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    run.create({"text": "guarded"})
    run.run()
    active = run.store.load().state["active"]
    toolkit.get("echo").confirmation_resolver = lambda arguments, context: {"description": "changed"}
    with pytest.raises(InteractionIntegrityError):
        run.resume_tool_interaction(active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "ok"}, node_invocation_id=active["node_invocation_id"])
    assert calls == [] and run.inspect().status == "waiting"


def test_two_confirmable_tool_nodes_need_two_distinct_approvals(tmp_path):
    calls: list[dict] = []
    document = _tool_document()
    first = document["nodes"][1]
    first["id"] = "first"
    document["nodes"].insert(2, {
        "id": "step",
        "type": "tool_call",
        "config": {
            "execution": {
                "inputs": [{"name": "text", "value": {"kind": "ref", "node_id": "first", "path": ["value", "result", "echo"]}, "missing": "error"}],
                "outputs": [{"name": "value", "source": {"kind": "result", "path": []}}],
            },
            "tool": {"name": "echo", "on_error": "stop", "approval_timeout_ms": 1000},
        },
    })
    document["edges"] = [
        {"id": "start-first", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "first", "target_port_id": "in"},
        {"id": "first-step", "kind": "flow", "source_node_id": "first", "source_port_id": "out", "target_node_id": "step", "target_port_id": "in"},
        {"id": "step-end", "kind": "flow", "source_node_id": "step", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
    ]
    toolkit = _toolkit(calls, approval=True)
    run = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100)})
    run.create({"text": "twice"})
    run.run()
    first_active = run.store.load().state["active"]
    run.resume_tool_interaction(first_active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "one"}, node_invocation_id=first_active["node_invocation_id"])
    run.run()
    second_active = run.store.load().state["active"]
    assert second_active["node_invocation_id"] != first_active["node_invocation_id"] and calls == [{"text": "twice"}]
    run.resume_tool_interaction(second_active["wait"]["id"], {"approved": True, "modified_arguments": None, "reason": "two"}, node_invocation_id=second_active["node_invocation_id"])
    assert run.run().outputs["value"]["result"] == {"echo": "twice"}
    assert calls == [{"text": "twice"}, {"text": "twice"}]


def test_tool_effect_without_a_durable_result_is_never_reissued_after_cold_recovery(tmp_path):
    document = _tool_document()
    document_path = tmp_path / "tool-workflow.json"
    document_path.write_text(json.dumps(document))
    script = r'''
import json, os, sys
from pathlib import Path
from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_workflow
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.tool_runtime import SkeletonToolRuntime
from unchain.tools.toolkit import Toolkit
root=Path(sys.argv[1]); document=json.loads(Path(sys.argv[2]).read_text())
toolkit=Toolkit()
@toolkit.tool(name='echo', requires_confirmation=False)
def echo(text):
    with (root/'effect').open('ab', buffering=0) as handle:
        handle.write((text + chr(10)).encode()); os.fsync(handle.fileno())
    os._exit(17)
repo=SQLiteContextV2Store(database_path=root/'journal.sqlite3', object_directory=root/'objects').bind_execution('execution')
plan=prepare_workflow(document)
store=WorkflowExecutionStore(repo,plan,graph_id='graph',attempt=AttemptRef(GenerationRef('execution','generation'),'attempt'))
run=DurableWorkflowRunner(store, {'tool_call': SkeletonToolRuntime(plan, toolkit, session_id='session', clock_ms=lambda: 1)})
run.create({'text':'once'}); run.run()
'''
    crashed = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path), str(document_path)], capture_output=True, text=True)
    assert crashed.returncode == 17, crashed.stderr
    assert (tmp_path / "effect").read_bytes() == b"once\n"
    calls: list[dict] = []
    toolkit = _toolkit(calls, approval=False)
    cold = _runner(tmp_path, document, {"tool_call": SkeletonToolRuntime(prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 1)})
    assert cold.recover().status == "outcome_unknown"
    assert cold.run().status == "outcome_unknown" and calls == []


def test_tool_runtime_import_never_initializes_a_model_provider():
    script = r'''
import builtins, sys
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.startswith(('unchain.providers', 'openai', 'anthropic', 'google.genai', 'ollama')):
        raise AssertionError('provider import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import unchain.skeleton.tool_runtime
assert 'unchain.providers' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


class _Registry:
    def __init__(self, loaded):
        self.loaded = loaded
        self.calls = 0

    def read_exact(self, identity):
        self.calls += 1
        self.requested_identity = identity
        return self.loaded


def test_skill_read_snapshots_identity_and_body_but_new_graph_rereads(tmp_path):
    identity = {"source": "project-agents", "source_id": "skill.md", "name": "demo"}
    registry = _Registry({"identity": identity, "body": "first", "revision": "sha256:" + "a" * 64})
    document = _skill_document(identity)
    run = _runner(tmp_path, document, {"skill_read": SkeletonSkillRuntime(prepare_workflow(document), registry)})
    run.create({})
    assert run.run().outputs == {"value": {"content": "first", "name": "demo"}}
    raw = run.store.read_json(run.store.load().state["completed"][1]["result_ref"])
    assert raw["_skeleton_skill"] == {"identity": identity, "revision": "sha256:" + "a" * 64}
    registry.loaded = {"identity": identity, "body": "changed", "revision": "sha256:" + "b" * 64}
    cold = _runner(tmp_path, document, {"skill_read": SkeletonSkillRuntime(prepare_workflow(document), registry)})
    assert cold.run().outputs == {"value": {"content": "first", "name": "demo"}}
    assert registry.calls == 1
    next_run = _runner(tmp_path, document, {"skill_read": SkeletonSkillRuntime(prepare_workflow(document), registry)}, graph="next")
    next_run.create({})
    assert next_run.run().outputs == {"value": {"content": "changed", "name": "demo"}}
    assert registry.calls == 2


def test_skill_identity_drift_fails_before_publication(tmp_path):
    identity = {"source": "project-agents", "source_id": "skill.md", "name": "demo"}
    registry = _Registry({"identity": {**identity, "source_id": "other.md"}, "body": "wrong", "revision": "sha256:" + "a" * 64})
    document = _skill_document(identity)
    run = _runner(tmp_path, document, {"skill_read": SkeletonSkillRuntime(prepare_workflow(document), registry)})
    run.create({})
    assert run.run().status == "failed"


def test_official_tool_runtime_must_match_the_store_definition(tmp_path):
    document = _tool_document()
    other = json.loads(json.dumps(document))
    other["nodes"][1]["config"]["tool"]["name"] = "different"
    calls: list[dict] = []
    toolkit = _toolkit(calls, approval=False)
    original = _runner(tmp_path, document, {
        "tool_call": SkeletonToolRuntime(
            prepare_workflow(document), toolkit, session_id="session", clock_ms=lambda: 100,
        ),
    })
    original.create({"text": "fixed-plan"})

    @toolkit.tool(name="different", requires_confirmation=False)
    def different(text: str):
        calls.append({"wrong_tool": text})
        return {"echo": text}

    with pytest.raises(SkeletonValidationError, match="official handler definition"):
        _runner(tmp_path, document, {
            "tool_call": SkeletonToolRuntime(
                prepare_workflow(other), toolkit, session_id="session", clock_ms=lambda: 100,
            ),
        })
    assert calls == []


def test_official_skill_runtime_must_match_the_store_definition(tmp_path):
    identity = {"source": "project-agents", "source_id": "skill.md", "name": "demo"}
    document = _skill_document(identity)
    other = json.loads(json.dumps(document))
    other["nodes"][1]["config"]["skill"]["source_id"] = "other.md"
    registry = _Registry({"identity": identity, "body": "wrong", "revision": "sha256:" + "a" * 64})
    original = _runner(tmp_path, document, {
        "skill_read": SkeletonSkillRuntime(prepare_workflow(document), registry),
    })
    original.create({})
    with pytest.raises(SkeletonValidationError, match="official handler definition"):
        _runner(tmp_path, document, {
            "skill_read": SkeletonSkillRuntime(prepare_workflow(other), registry),
        })
    assert registry.calls == 0
