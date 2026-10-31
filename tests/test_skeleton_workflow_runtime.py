"""CP14 fixed-revision Workflow Call and parent/child recovery tests."""

from copy import deepcopy
import sys
import threading

import pytest

from unchain.journal import AttemptRef, GenerationRef, JournalConflictError
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import WorkflowNodeCompleted
from unchain.skeleton import FileCodeSourceResolver, PythonEnvironmentProfile, SkeletonCodeRuntime
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner, WorkflowNodeWaiting
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.preparation import prepare_workflow
from unchain.skeleton.workflow_runtime import PublishedWorkflow, SkeletonWorkflowCallRuntime


def _execution(inputs, outputs):
    return {"inputs": inputs, "outputs": outputs}


def _binding(name, node_id, path):
    return {"name": name, "value": {"kind": "ref", "node_id": node_id, "path": path}, "missing": "error"}


def _output(name, kind, path):
    return {"name": name, "source": {"kind": kind, "path": path}}


def child_document(workflow_id="child", *, suffix=""):
    return {
        "schema_version": 1, "kind": "workflow", "id": workflow_id,
        "name": "Child" + suffix,
        "nodes": [
            {"id": "start", "type": "start", "config": {
                "execution": {"outputs": [_output("value", "result", ["value"])]}}},
            {"id": "code", "type": "code", "config": {"execution": _execution(
                [_binding("value", "start", ["value"])],
                [_output("value", "result", ["value"]),
                 _output("version", "result", ["version"])])}},
            {"id": "end", "type": "end", "config": {"execution": _execution(
                [_binding("value", "code", ["value"]), _binding("version", "code", ["version"])],
                [_output("value", "input", ["value"]), _output("version", "input", ["version"])])}},
        ],
        "edges": [
            {"id": "s-c", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "code", "target_port_id": "in"},
            {"id": "c-e", "kind": "flow", "source_node_id": "code", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


def parent_document(*, revision="rev-1", workflow_id="child", parent_id="parent"):
    return {
        "schema_version": 1, "kind": "workflow", "id": parent_id, "name": "Parent",
        "nodes": [
            {"id": "start", "type": "start", "config": {
                "execution": {"outputs": [_output("value", "result", ["value"])]}}},
            {"id": "call", "type": "workflow_call", "config": {
                "workflow": {"workflow_id": workflow_id, "revision_id": revision},
                "execution": _execution([_binding("value", "start", ["value"])],
                    [_output("value", "result", ["value"]), _output("version", "result", ["version"])])}},
            {"id": "end", "type": "end", "config": {"execution": _execution(
                [_binding("value", "call", ["value"]), _binding("version", "call", ["version"])],
                [_output("value", "input", ["value"]), _output("version", "input", ["version"])])}},
        ],
        "edges": [
            {"id": "s-call", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "call", "target_port_id": "in"},
            {"id": "call-e", "kind": "flow", "source_node_id": "call", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


def agent_parent_document(*, revision="rev-1", workflow_id="child"):
    return {
        "schema_version": 1, "kind": "agent", "id": "agent-parent", "name": "Agent Parent",
        "nodes": [
            {"id": "start", "type": "start", "config": {
                "execution": {"outputs": [_output("value", "result", ["value"])]}}},
            {"id": "kernel", "type": "agent", "config": {
                "kernel": {
                    "schema_version": 1,
                    "profile": {
                        "id": "agent.test", "provider": "openai", "model": "test-model",
                        "protocol": "openai-responses", "endpoint": "https://example.test/v1",
                        "credential_ref": "agent.key", "context_window": 8192,
                        "output_limit": 2048,
                        "capabilities": {"json_schema": True, "images": False, "files": False},
                    },
                    "prompt": "test", "max_iterations": 3, "output_tokens": 512,
                    "history": "root", "overflow": "error", "json_schema": None,
                },
                "execution": _execution(
                    [_binding("value", "start", ["value"])],
                    [_output("value", "result", ["message"])],
                ),
            }},
            {"id": "call", "type": "workflow_call", "config": {
                "workflow": {"workflow_id": workflow_id, "revision_id": revision},
                "execution": _execution(
                    [_binding("value", "kernel", ["value"])],
                    [_output("value", "result", ["value"]),
                     _output("version", "result", ["version"])],
                ),
            }},
            {"id": "end", "type": "end", "config": {"execution": _execution(
                [_binding("value", "call", ["value"]),
                 _binding("version", "call", ["version"])],
                [_output("value", "input", ["value"]),
                 _output("version", "input", ["version"])],
            )}},
        ],
        "edges": [
            {"id": "s-k", "kind": "flow", "source_node_id": "start",
             "source_port_id": "out", "target_node_id": "kernel", "target_port_id": "in"},
            {"id": "k-c", "kind": "flow", "source_node_id": "kernel",
             "source_port_id": "out", "target_node_id": "call", "target_port_id": "in"},
            {"id": "c-e", "kind": "flow", "source_node_id": "call",
             "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


def host(tmp_path, document, resolver, code, *, graph="parent-graph"):
    journal = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    ).bind_execution("execution")
    plan = prepare_workflow(document)
    store = WorkflowExecutionStore(journal, plan, graph_id=graph,
        attempt=AttemptRef(GenerationRef("execution", "generation"), "root-attempt"))
    runtime = SkeletonWorkflowCallRuntime(store, resolver, handlers={"code": code})
    return DurableWorkflowRunner(store, {"workflow_call": runtime}), runtime


def test_published_workflow_call_returns_only_child_end_and_pins_revision(tmp_path):
    definitions = {
        ("child", "rev-1"): child_document(suffix=" one"),
        ("child", "rev-2"): child_document(suffix=" two"),
    }
    calls = []
    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definitions[(workflow_id, revision_id)]))
    def code(context, inputs):
        calls.append(context.graph_id)
        return WorkflowNodeCompleted({"value": inputs["value"] + "!", "version": "rev-1"})
    parent, _runtime = host(tmp_path, parent_document(), resolver, code)
    parent.create({"value": "fixed"})
    definitions[("child", "rev-1")] = child_document(suffix=" changed after create")
    result = parent.run()
    assert result.status == "completed"
    assert result.outputs == {"value": "fixed!", "version": "rev-1"}
    assert len(calls) == 1 and ".child." in calls[0]
    assert parent.store.load().state["call_stack"][-1]["definition_id"] == "parent"


def test_child_wait_survives_cold_parent_and_completes_exact_invocation(tmp_path):
    definition = child_document()
    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))
    def waiting(context, inputs):
        return WorkflowNodeWaiting("interaction", "approve-child", {"value": inputs["value"]})
    parent, runtime = host(tmp_path, parent_document(), resolver, waiting)
    parent.create({"value": "cold"})
    first = parent.run()
    assert first.status == "waiting"
    parent_state = parent.store.load().state
    invocation = parent_state["active"]["node_invocation_id"]

    cold_parent, cold_runtime = host(tmp_path, parent_document(), resolver, waiting)
    _record, _active, _wait, child, _child_runtime = cold_runtime._active_child(
        cold_parent, invocation,
    )
    child_state = child.store.load().state
    child.complete_wait("approve-child", {"value": "approved", "version": "rev-1"},
        node_invocation_id=child_state["active"]["node_invocation_id"])
    assert child.run().outputs == {"value": "approved", "version": "rev-1"}
    result = cold_runtime.resume_child(cold_parent, node_invocation_id=invocation)
    assert result.status == "completed"
    assert result.outputs == {"value": "approved", "version": "rev-1"}


def test_failed_waiting_child_preserves_parent_wait_in_terminal_receipt(tmp_path):
    definition = child_document()
    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))
    parent, runtime = host(tmp_path, parent_document(), resolver,
        lambda _context, _inputs: WorkflowNodeWaiting("interaction", "approve-child", {}))
    parent.create({"value": "x"})
    assert parent.run().status == "waiting"
    active = parent.store.load().state["active"]
    _record, _active, _wait, child, _child_runtime = runtime._active_child(
        parent, active["node_invocation_id"],
    )
    assert child.cancel().status == "cancelled"
    result = runtime.resume_child(parent, node_invocation_id=active["node_invocation_id"])
    assert result.status == "failed"
    terminal = parent.store.load().state
    assert terminal["active"]["wait"] == active["wait"]
    assert terminal["error"]["code"] == "child_cancelled"


def test_unknown_waiting_child_preserves_parent_wait_in_terminal_receipt(tmp_path):
    definition = child_document()
    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))
    parent, _runtime = host(tmp_path, parent_document(), resolver,
        lambda _context, _inputs: WorkflowNodeWaiting("interaction", "approve-child", {}))
    parent.create({"value": "x"})
    assert parent.run().status == "waiting"
    record = parent.store.load()
    wait = deepcopy(record.state["active"]["wait"])
    error = SkeletonValidationError("child_outcome_unknown", ("execution", "child"),
                                    "Child effect may have occurred")
    altered = deepcopy(record.state)
    altered["status"] = "outcome_unknown"
    altered["error"] = {"code": "child_outcome_unknown", "path": ["execution", "child"]}
    altered["active"]["wait"]["id"] = "different-child"
    with pytest.raises(SkeletonValidationError):
        parent.store.save(altered, expected_revision=record.revision,
                          operation_id="probe.changed-child-wait")
    terminal = parent._failed(record, error, unknown=True)
    assert terminal.state["status"] == "outcome_unknown"
    assert terminal.state["active"]["wait"] == wait
    assert parent.inspect().status == "outcome_unknown"


def test_parent_cancel_propagates_to_waiting_child(tmp_path):
    definition = child_document()
    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))
    parent, runtime = host(tmp_path, parent_document(), resolver,
        lambda context, inputs: WorkflowNodeWaiting("interaction", "hold", {}))
    parent.create({"value": "x"}); parent.run()
    state = parent.store.load().state
    invocation = state["active"]["node_invocation_id"]
    _record, _active, _wait, child, _child_runtime = runtime._active_child(parent, invocation)
    assert child.inspect().status == "waiting"
    assert parent.cancel().status == "cancelled"
    assert child.inspect().status == "cancelled"


def test_parent_cancel_revokes_running_child_before_effect(tmp_path):
    definition = child_document()
    entered = threading.Event()
    release = threading.Event()
    effects = []

    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))

    def code(context, inputs):
        entered.set()
        assert release.wait(5)
        context.assert_current()
        effects.append("late")
        return WorkflowNodeCompleted({"value": inputs["value"], "version": "rev-1"})

    parent, _runtime = host(tmp_path, parent_document(), resolver, code)
    parent.create({"value": "x"})
    errors = []
    def run_parent():
        try:
            parent.run()
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=run_parent)
    thread.start()
    try:
        assert entered.wait(5)
        assert parent.cancel().status == "cancelled"
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert effects == []


def test_cold_parent_recovers_completed_child_without_replaying_effect(tmp_path):
    definition = child_document()
    effects = []

    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))

    def code(_context, inputs):
        effects.append("effect")
        return WorkflowNodeCompleted({"value": inputs["value"], "version": "rev-1"})

    parent, _runtime = host(tmp_path, parent_document(), resolver, code)
    parent.create({"value": "receipt"})
    parent._seal_completion = lambda *_args: (_ for _ in ()).throw(SystemExit(73))
    with pytest.raises(SystemExit):
        parent.run()
    assert effects == ["effect"]

    cold, _runtime = host(
        tmp_path, parent_document(), resolver,
        lambda *_args: (_ for _ in ()).throw(AssertionError("child replayed")),
    )
    result = cold.recover()
    if result.status in {"ready", "result_received"}:
        result = cold.run()
    assert result.status == "completed"
    assert result.outputs == {"value": "receipt", "version": "rev-1"}
    assert effects == ["effect"]


def test_nested_workflow_rebinds_official_code_runtime_per_child_plan(tmp_path):
    leaf = child_document("leaf")
    code_node = next(node for node in leaf["nodes"] if node["id"] == "code")
    code_node["config"]["code"] = {
        "source": {"kind": "file", "path": "main.py"},
        "environment_profile_id": "test", "timeout_ms": 5000,
        "max_result_bytes": 1048576, "max_log_bytes": 65536,
        "result_schema": {"type": "object"},
    }
    middle = parent_document(revision="rev-leaf", workflow_id="leaf", parent_id="middle")
    root = parent_document(revision="rev-middle", workflow_id="middle")
    definitions = {"leaf": leaf, "middle": middle}
    (tmp_path / "main.py").write_text(
        "def main(input, context):\n"
        "    return {'value': input['value'] + '!', 'version': 'rev-1'}\n",
        encoding="utf-8",
    )
    runtime = SkeletonCodeRuntime(
        prepare_workflow(leaf), source_resolver=FileCodeSourceResolver(tmp_path),
        profile_resolver=lambda _profile_id: PythonEnvironmentProfile.venv("test", sys.executable),
        staging_root=tmp_path / "staging",
    )

    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definitions[workflow_id]))

    parent, _runtime = host(tmp_path, root, resolver, runtime)
    parent.create({"value": "nested"})
    result = parent.run()
    assert result.status == "completed"
    assert result.outputs == {"value": "nested!", "version": "rev-1"}


def test_nested_child_wait_cold_resume_and_cancel_propagate_recursively(tmp_path):
    definitions = {
        ("child", "rev-child"): parent_document(
            revision="rev-grandchild", workflow_id="grandchild", parent_id="child",
        ),
        ("grandchild", "rev-grandchild"): child_document("grandchild"),
    }

    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(
            workflow_id, revision_id, deepcopy(definitions[(workflow_id, revision_id)]),
        )

    def waiting(_context, inputs):
        return WorkflowNodeWaiting("interaction", "deep-wait", {"value": inputs["value"]})

    root_document = parent_document(revision="rev-child", workflow_id="child")
    root, _runtime = host(tmp_path, root_document, resolver, waiting)
    root.create({"value": "nested"})
    assert root.run().status == "waiting"

    cold_root, cold_runtime = host(tmp_path, root_document, resolver, waiting)
    root_record = cold_root.store.load()
    root_invocation = root_record.state["active"]["node_invocation_id"]
    _record, _active, _wait, child, child_runtime = cold_runtime._active_child(
        cold_root, root_invocation,
    )
    child_record = child.store.load()
    child_invocation = child_record.state["active"]["node_invocation_id"]
    _record, _active, _wait, grandchild, _runtime = child_runtime._active_child(
        child, child_invocation,
    )
    grandchild_state = grandchild.store.load().state
    grandchild.complete_wait(
        "deep-wait", {"value": "done", "version": "rev-grandchild"},
        node_invocation_id=grandchild_state["active"]["node_invocation_id"],
    )
    assert grandchild.run().status == "completed"
    completed = cold_runtime.resume_child(cold_root, node_invocation_id=root_invocation)
    assert completed.status == "completed"
    assert completed.outputs == {"value": "done", "version": "rev-grandchild"}

    cancel_root, cancel_runtime = host(
        tmp_path, root_document, resolver, waiting, graph="cancel-root",
    )
    cancel_root.create({"value": "cancel"})
    assert cancel_root.run().status == "waiting"
    cancel_record = cancel_root.store.load()
    cancel_invocation = cancel_record.state["active"]["node_invocation_id"]
    _record, _active, _wait, cancel_child, cancel_child_runtime = (
        cancel_runtime._active_child(cancel_root, cancel_invocation)
    )
    cancel_child_record = cancel_child.store.load()
    cancel_child_invocation = cancel_child_record.state["active"]["node_invocation_id"]
    _record, _active, _wait, cancel_grandchild, _runtime = (
        cancel_child_runtime._active_child(cancel_child, cancel_child_invocation)
    )
    assert cancel_root.cancel().status == "cancelled"
    assert cancel_child.inspect().status == "cancelled"
    assert cancel_grandchild.inspect().status == "cancelled"


def test_nested_workflow_call_keeps_ancestor_authority_at_the_leaf(tmp_path):
    """A nested child must poll the root authority before its Code side effect."""
    definitions = {
        ("child", "rev-child"): parent_document(
            revision="rev-grandchild", workflow_id="grandchild", parent_id="child",
        ),
        ("grandchild", "rev-grandchild"): child_document("grandchild"),
    }
    authority = {"active": True}
    started = threading.Event()
    release = threading.Event()
    effect = []

    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(
            workflow_id, revision_id, deepcopy(definitions[(workflow_id, revision_id)]),
        )

    def code(context, inputs):
        started.set()
        assert release.wait(5), "test controller did not release the nested child"
        context.assert_current()
        effect.append(inputs["value"])
        return WorkflowNodeCompleted({"value": inputs["value"], "version": "rev-grandchild"})

    def assert_root_authority():
        if not authority["active"]:
            raise RuntimeError("root execution authority ended")

    initial, _runtime = host(
        tmp_path,
        parent_document(revision="rev-child", workflow_id="child"),
        resolver,
        code,
    )
    root = DurableWorkflowRunner(
        initial.store,
        initial.handlers,
        ancestor_assert_current=assert_root_authority,
    )
    root.create({"value": "must not write"})

    def end_root_authority():
        assert started.wait(5), "nested Code handler did not start"
        authority["active"] = False
        release.set()

    controller = threading.Thread(target=end_root_authority)
    controller.start()
    result = root.run()
    controller.join(timeout=5)

    assert not controller.is_alive()
    assert result.status == "outcome_unknown"
    assert effect == []


def test_agent_graph_can_call_a_published_workflow_revision(tmp_path):
    definition = child_document()

    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definition))

    journal = SQLiteContextV2Store(
        database_path=tmp_path / "agent-journal.sqlite3",
        object_directory=tmp_path / "agent-objects",
    ).bind_execution("agent-execution")
    plan = prepare_agent_graph(agent_parent_document())
    store = WorkflowExecutionStore(
        journal, plan, graph_id="agent-parent",
        attempt=AttemptRef(GenerationRef("agent-execution", "generation"), "root-attempt"),
    )

    def kernel(_context, inputs):
        return WorkflowNodeCompleted({
            "message": inputs["value"] + "-agent", "context": {},
        })

    def code(_context, inputs):
        return WorkflowNodeCompleted({"value": inputs["value"] + "-child", "version": "rev-1"})

    runtime = SkeletonWorkflowCallRuntime(
        store, resolver, handlers={"agent": kernel, "code": code},
    )
    runner = DurableWorkflowRunner(
        store, {"agent": kernel, "workflow_call": runtime},
    )
    runner.create({"value": "start"})
    result = runner.run()
    assert result.status == "completed"
    assert result.outputs == {"value": "start-agent-child", "version": "rev-1"}


@pytest.mark.parametrize("failure", ["missing", "wrong_kind", "recursive"])
def test_reference_failures_are_explicit_before_child_dispatch(tmp_path, failure):
    parent_doc = parent_document(
        workflow_id="parent" if failure == "recursive" else "child",
        parent_id="parent",
    )
    calls = []
    def resolver(workflow_id, revision_id):
        if failure == "missing":
            raise SkeletonValidationError("resource_missing", ("resource",), "missing")
        document = parent_document(parent_id="parent") if failure == "recursive" else child_document()
        if failure == "wrong_kind":
            document["kind"] = "agent"
        return PublishedWorkflow(workflow_id, revision_id, document)
    with pytest.raises(SkeletonValidationError) as caught:
        host(tmp_path, parent_doc, resolver, lambda *_: calls.append("dispatched"))
    assert caught.value.code in {"resource_missing", "wrong_kind", "recursive_reference"}
    assert calls == []


def test_workflow_call_config_is_closed_and_requires_fixed_revision(tmp_path):
    document = parent_document()
    document["nodes"][1]["config"]["workflow"]["latest"] = True
    journal = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    ).bind_execution("execution")
    plan = prepare_workflow(document)
    store = WorkflowExecutionStore(journal, plan, graph_id="graph",
        attempt=AttemptRef(GenerationRef("execution", "generation"), "attempt"))
    with pytest.raises(SkeletonValidationError) as caught:
        DurableWorkflowRunner(store, {})
    assert caught.value.code in {"unknown_field", "invalid_shape"}


def test_cancelled_parent_retries_interrupted_child_propagation(tmp_path, monkeypatch):
    definition = child_document()
    resolver = lambda workflow_id, revision_id: PublishedWorkflow(
        workflow_id, revision_id, deepcopy(definition),
    )
    waiting = lambda _context, inputs: WorkflowNodeWaiting(
        "interaction", "approve", {"value": inputs["value"]},
    )
    parent, runtime = host(tmp_path, parent_document(), resolver, waiting)
    parent.create({"value": "cancel"})
    assert parent.run().status == "waiting"
    active = parent.store.load().state["active"]
    child, _child_runtime, _published, _plan = runtime._child_for_active(active)
    monkeypatch.setattr(runtime, "cancel_child", lambda *_args, **_kwargs: None)
    assert parent.cancel().status == "cancelled"
    assert child.inspect().status == "waiting"

    cold, cold_runtime = host(tmp_path, parent_document(), resolver, waiting)
    assert cold.cancel().status == "cancelled"
    cold_child, *_ = cold_runtime._child_for_active(active)
    assert cold_child.inspect().status == "cancelled"


def test_cancel_child_accepts_a_raced_terminal_descendant(tmp_path, monkeypatch):
    definition = child_document()
    resolver = lambda workflow_id, revision_id: PublishedWorkflow(
        workflow_id, revision_id, deepcopy(definition),
    )
    waiting = lambda _context, inputs: WorkflowNodeWaiting(
        "interaction", "approve", {"value": inputs["value"]},
    )
    parent, runtime = host(tmp_path, parent_document(), resolver, waiting)
    parent.create({"value": "cancel"})
    assert parent.run().status == "waiting"
    active = parent.store.load().state["active"]
    child, *_ = runtime._child_for_active(active)
    original_cancel = DurableWorkflowRunner.cancel
    raced = {"done": False}

    def cancel_after_terminal_write(runner):
        if runner.store.graph_id == child.store.graph_id and not raced["done"]:
            raced["done"] = True
            original_cancel(runner)
            raise JournalConflictError("workflow execution revision changed")
        return original_cancel(runner)

    monkeypatch.setattr(DurableWorkflowRunner, "cancel", cancel_after_terminal_write)

    result = runtime.cancel_child(parent, node_invocation_id=active["node_invocation_id"])

    assert raced["done"]
    assert result.status == "cancelled"
    assert child.inspect().status == "cancelled"


def test_cancelled_parent_cold_recovery_finishes_child_propagation(tmp_path, monkeypatch):
    definition = child_document()
    resolver = lambda workflow_id, revision_id: PublishedWorkflow(
        workflow_id, revision_id, deepcopy(definition),
    )
    waiting = lambda _context, inputs: WorkflowNodeWaiting(
        "interaction", "approve", {"value": inputs["value"]},
    )
    parent, runtime = host(tmp_path, parent_document(), resolver, waiting)
    parent.create({"value": "cancel"})
    assert parent.run().status == "waiting"
    active = parent.store.load().state["active"]
    child, *_ = runtime._child_for_active(active)
    monkeypatch.setattr(runtime, "cancel_child", lambda *_args, **_kwargs: None)
    assert parent.cancel().status == "cancelled"
    assert child.inspect().status == "waiting"

    cold, cold_runtime = host(tmp_path, parent_document(), resolver, waiting)
    assert cold.recover().status == "cancelled"
    cold_child, *_ = cold_runtime._child_for_active(active)
    assert cold_child.inspect().status == "cancelled"


def test_child_failure_is_authoritative_during_normal_dispatch(tmp_path):
    definition = child_document()
    resolver = lambda workflow_id, revision_id: PublishedWorkflow(
        workflow_id, revision_id, deepcopy(definition),
    )
    calls = []

    def missing_outputs(_context, _inputs):
        calls.append("called")
        return WorkflowNodeCompleted({})

    parent, _runtime = host(
        tmp_path, parent_document(), resolver, missing_outputs,
    )
    parent.create({"value": "failure"})
    result = parent.run()
    assert result.status == "failed"
    assert result.error["code"] == "child_failed"
    assert calls == ["called"]


@pytest.mark.parametrize("code", ["child_failed", "child_cancelled"])
def test_custom_workflow_handler_cannot_assert_authoritative_child_failure(tmp_path, code):
    resolver = lambda workflow_id, revision_id: PublishedWorkflow(
        workflow_id, revision_id, child_document(),
    )
    parent, _runtime = host(
        tmp_path, parent_document(), resolver,
        lambda _context, _inputs: pytest.fail("child handler must not run"),
    )
    calls = []

    def custom_handler(_context, _inputs):
        calls.append("called")
        raise SkeletonValidationError(
            code, ("custom",), "no authoritative child receipt",
        )

    runner = DurableWorkflowRunner(
        parent.store, {"workflow_call": custom_handler},
    )
    runner.create({"value": "failure"})
    result = runner.run()
    assert result.status == "outcome_unknown"
    assert result.error["code"] == "handler_error"
    assert calls == ["called"]


def test_cold_recovery_seals_durably_failed_child(tmp_path, monkeypatch):
    definition = child_document()
    resolver = lambda workflow_id, revision_id: PublishedWorkflow(
        workflow_id, revision_id, deepcopy(definition),
    )
    parent, runtime = host(
        tmp_path, parent_document(), resolver,
        lambda _context, _inputs: WorkflowNodeCompleted({}),
    )
    parent.create({"value": "failure"})

    class ProcessLost(BaseException):
        pass

    monkeypatch.setattr(parent, "_failed", lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessLost()))
    with pytest.raises(ProcessLost):
        parent.run()
    active = parent.store.load().state["active"]
    child, *_ = runtime._child_for_active(active)
    assert child.inspect().status == "failed"

    cold, _cold_runtime = host(tmp_path, parent_document(), resolver, lambda *_: pytest.fail("replayed"))
    result = cold.recover()
    assert result.status == "failed"
    assert result.error["code"] == "child_failed"
