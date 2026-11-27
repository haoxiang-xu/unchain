"""CP10 observable execution/recovery tests using the real durable store."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from unchain.journal import AttemptRef, GenerationRef, JournalConflictError
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_workflow, WorkflowNodeCompleted
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.durable_execution import DurableWorkflowRunner, WorkflowNodeWaiting
from unchain.skeleton.durable_protocol import validate_execution_state
from unchain.skeleton.durable_store import WorkflowExecutionStore


FIXTURES = Path(__file__).parent / "fixtures" / "skeleton_execution"


def document():
    return json.loads((FIXTURES / "if_rejoin_workflow.json").read_text())


def runner(tmp_path, handler=None, *, doc=None, graph="graph", generation="generation", attempt="attempt"):
    repo = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite3", object_directory=tmp_path / "objects").bind_execution("execution")
    plan = prepare_workflow(document() if doc is None else doc)
    store = WorkflowExecutionStore(repo, plan, graph_id=graph,
                                   attempt=AttemptRef(GenerationRef("execution", generation), attempt))
    return DurableWorkflowRunner(store, {"code": handler} if handler else {})


def code(calls):
    def execute(context, inputs):
        context.assert_current()
        calls.append(context.node_id)
        return WorkflowNodeCompleted({"answer": context.node_id})
    return execute


@pytest.mark.parametrize("flag,branch,edges", [(True, "left", ("a", "b", "d", "join-end")),
                                               (False, "right", ("a", "c", "e", "join-end"))])
def test_real_durable_branch_rejoin_and_second_run(tmp_path, flag, branch, edges):
    calls = []; run = runner(tmp_path, code(calls))
    run.create({"flag": flag})
    result = run.run()
    assert result.status == "completed" and result.outputs == {"answer": branch}
    assert result.completed_node_ids == ("start", "if", branch, "join", "end")
    assert result.selected_edge_ids == edges and calls == [branch, "join"]
    again = runner(tmp_path, lambda *_: pytest.fail("completed handler repeated"))
    assert again.create({"flag": flag}).completion_id == result.completion_id
    assert again.run() == result
    with pytest.raises(JournalConflictError):
        again.create({"flag": not flag})
    second = runner(tmp_path, code(calls), graph="graph2", generation="generation2", attempt="attempt2")
    second.create({"flag": not flag})
    assert second.run().completion_id != result.completion_id


def test_end_uses_explicit_binding_not_last_handler_and_commits_once(tmp_path):
    calls = []; run = runner(tmp_path, code(calls)); run.create({"flag": True})
    with pytest.raises(SkeletonValidationError):
        run.commit_root()
    result = run.run()
    assert calls[-1] == "join" and result.outputs == {"answer": "left"}
    assert not result.root_committed
    committed = run.commit_root()
    assert committed.root_committed and committed.completion_id == result.completion_id
    assert runner(tmp_path).commit_root() == committed


def test_pause_at_branch_preserves_selection_and_resume(tmp_path):
    calls = []; run = runner(tmp_path, code(calls)); run.create({"flag": False})
    run.run(max_steps=2)
    paused = run.pause()
    assert paused.selected_edge_ids == ("a", "c") and paused.paused
    assert run.run() == paused and calls == []
    cold = runner(tmp_path, code(calls))
    assert cold.resume().outputs == {"answer": "right"}
    assert calls == ["right", "join"]


class Crash(BaseException):
    pass


@pytest.mark.parametrize("status,when,expected,calls_before", [
    ("running", "before", "ready", 0),
    ("running", "after", "outcome_unknown", 0),
    ("result_received", "before", "outcome_unknown", 1),
    ("result_received", "after", "result_received", 1),
    ("ready", "after", "ready", 1),
    ("completed", "after", "completed", 2),
])
def test_fault_windows_never_repeat_completed_or_uncertain_handlers(tmp_path, monkeypatch, status, when, expected, calls_before):
    calls = []; run = runner(tmp_path, code(calls)); run.create({"flag": True}); run.run(max_steps=2)
    save = run.store.save
    fired = False

    def crashing(state, **kwargs):
        nonlocal fired
        hit = not fired and state["status"] == status
        if hit:
            fired = True
            if when == "before":
                raise Crash()
        result = save(state, **kwargs)
        if hit and when == "after":
            raise Crash()
        return result

    monkeypatch.setattr(run.store, "save", crashing)
    with pytest.raises(Crash):
        run.run()
    assert len(calls) == calls_before
    cold = runner(tmp_path, code(calls))
    assert cold.recover().status == expected
    final = cold.run()
    if expected == "outcome_unknown":
        assert final.status == expected and len(calls) == calls_before and final.outputs is None
    else:
        assert final.status == "completed" and calls == ["left", "join"]


def test_handler_crash_is_explicitly_unknown_not_failed_or_retried(tmp_path):
    calls = []
    def crash(ctx, inputs):
        calls.append(ctx.node_id)
        raise RuntimeError("secret value must not enter failure record")
    run = runner(tmp_path, crash); run.create({"flag": True})
    result = run.run()
    assert result.status == "outcome_unknown" and calls == ["left"]
    assert "secret" not in str(run.store.load().state)
    assert runner(tmp_path, code(calls)).run().status == "outcome_unknown"
    assert calls == ["left"]


def test_saved_raw_result_survives_invalid_output_mapping(tmp_path):
    calls = []
    def incomplete(ctx, inputs):
        calls.append(ctx.node_id)
        return WorkflowNodeCompleted({"other": "durable"})
    run = runner(tmp_path, incomplete); run.create({"flag": True})
    result = run.run()
    assert result.status == "failed" and result.error["code"] == "missing_value"
    raw = run.store.load().state["active"]["result_ref"]
    assert run.store.read_json(raw) == {"other": "durable"}
    assert runner(tmp_path, code(calls)).run().status == "failed" and calls == ["left"]


@pytest.mark.parametrize("wait_kind", ["interaction", "child"])
def test_wait_identity_cold_completion_and_next_wait(tmp_path, wait_kind):
    calls = []
    def waiting(ctx, inputs):
        calls.append(ctx.node_id)
        return WorkflowNodeWaiting(wait_kind, "wait-" + ctx.node_invocation_id, {"action": ctx.node_id})
    run = runner(tmp_path, waiting); run.create({"flag": True})
    assert run.run().status == "waiting"
    for expected in ("left", "join"):
        cold = runner(tmp_path, waiting)
        active = cold.store.load().state["active"]
        assert cold.store.read_json(active["wait"]["request_ref"]) == {"action": expected}
        before = cold.inspect().revision
        with pytest.raises(SkeletonValidationError):
            cold.complete_wait("wrong", {"answer": "unapproved"}, node_invocation_id=active["node_invocation_id"])
        assert cold.inspect().revision == before
        cold.complete_wait(active["wait"]["id"], {"answer": expected + "-approved"}, node_invocation_id=active["node_invocation_id"])
        with pytest.raises(SkeletonValidationError):
            cold.complete_wait(active["wait"]["id"], {"answer": "duplicate"}, node_invocation_id=active["node_invocation_id"])
        result = cold.run()
    assert result.outputs == {"answer": "left-approved"} and calls == ["left", "join"]


@pytest.mark.parametrize("control", ["cancel", "recover", "pause"])
def test_control_race_fences_owner_or_seals_paused_result(tmp_path, control):
    entered = threading.Event(); release = threading.Event(); calls = []; outcomes = []
    def blocked(ctx, inputs):
        calls.append(ctx.node_id); entered.set()
        assert release.wait(10)
        return WorkflowNodeCompleted({"answer": ctx.node_id})
    run = runner(tmp_path, blocked); run.create({"flag": True})
    def execute():
        try:
            outcomes.append(run.run())
        except Exception as error:
            outcomes.append(error)
    thread = threading.Thread(target=execute); thread.start()
    assert entered.wait(10)
    other = runner(tmp_path, code(calls))
    assert other.run().status == "running"  # observing is not takeover
    getattr(other, control)()
    release.set(); thread.join(10)
    assert not thread.is_alive() and calls == ["left"]
    if control == "pause":
        assert other.inspect().paused and other.inspect().status == "result_received"
        assert other.resume().outputs == {"answer": "left"}
        assert calls == ["left", "join"]
    else:
        assert isinstance(outcomes[0], JournalConflictError)
        assert other.run().status == ("cancelled" if control == "cancel" else "outcome_unknown")


def test_cancel_ready_and_waiting_does_not_create_success(tmp_path):
    run = runner(tmp_path); run.create({"flag": True})
    assert run.cancel().outputs is None
    assert run.cancel() == run.run() and run.inspect().status == "cancelled"
    second = runner(tmp_path, lambda *_: WorkflowNodeWaiting("interaction", "wait", {}), graph="second")
    second.create({"flag": False}); second.run(); second.cancel()
    with pytest.raises(SkeletonValidationError):
        second.complete_wait("wait", {"answer": "late"}, node_invocation_id=second.store.load().state["active"]["node_invocation_id"])
    assert second.inspect().status == "cancelled"


def test_input_and_binary_attachments_are_snapshots(tmp_path):
    source = tmp_path / "source.bin"; source.write_bytes(b"original\x00bytes")
    inputs = {"flag": True}; calls = []
    def read(ctx, values):
        calls.append(ctx.read_attachment("file"))
        with pytest.raises(SkeletonValidationError):
            ctx.read_attachment("not-granted")
        return WorkflowNodeCompleted({"answer": "ok"})
    run = runner(tmp_path, read)
    run.create(inputs, attachments={"file": (source.read_bytes(), "application/octet-stream")})
    inputs["flag"] = False; source.unlink(); run.pause()
    assert runner(tmp_path, read).resume().outputs == {"answer": "ok"}
    assert calls == [b"original\x00bytes", b"original\x00bytes"]


def test_child_frame_never_commits_root(tmp_path):
    run = runner(tmp_path, code([]), graph="child", attempt="child-attempt")
    run.create({"flag": True})
    # Build a separately admitted child, never mutate a saved lineage.
    frames = run.store.load().state["call_stack"]
    frames.insert(0, {"execution_id": "execution", "graph_invocation_id": "parent", "attempt_id": "parent-attempt",
                      "definition_id": "parent-definition", "definition_digest": "a" * 64,
                      "caller_node_invocation_id": "parent-node"})
    child = runner(tmp_path, code([]), graph="real-child", attempt="real-child-attempt")
    frames[-1]["graph_invocation_id"] = "real-child"; frames[-1]["attempt_id"] = "real-child-attempt"
    child.create({"flag": True}, call_stack=frames)
    assert child.run().status == "completed"
    with pytest.raises(SkeletonValidationError):
        child.commit_root()


@pytest.mark.parametrize("field,value", [("media_type", " application/json "), ("preview", None), ("fragment", None)])
def test_artifact_descriptor_normalization_is_refused(tmp_path, field, value):
    run = runner(tmp_path); run.create({"flag": True})
    state = run.store.load().state
    target = state["inputs_ref"]["ref"] if field == "fragment" else state["inputs_ref"]
    target[field] = value
    with pytest.raises(SkeletonValidationError):
        validate_execution_state(state, run.plan)


def test_end_commit_lost_acknowledgement_is_idempotent(tmp_path, monkeypatch):
    run = runner(tmp_path, code([])); run.create({"flag": True}); run.run()
    save = run.store.save
    def lose_ack(state, **kwargs):
        save(state, **kwargs)
        raise Crash()
    monkeypatch.setattr(run.store, "save", lose_ack)
    with pytest.raises(Crash):
        run.commit_root()
    cold = runner(tmp_path)
    before = cold.inspect()
    assert before.root_committed and cold.commit_root() == before


def test_cold_process_continues_from_received_result(tmp_path):
    calls = []; run = runner(tmp_path, code(calls)); run.create({"flag": True}); run.run(max_steps=3)
    assert run.inspect().status == "result_received" and calls == ["left"]
    script = r'''
import json, sys
from pathlib import Path
from unchain.journal import AttemptRef,GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_workflow,WorkflowNodeCompleted
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.durable_execution import DurableWorkflowRunner
root=Path(sys.argv[1]); doc=json.loads(Path(sys.argv[2]).read_text())
repo=SQLiteContextV2Store(database_path=root/'journal.sqlite3',object_directory=root/'objects').bind_execution('execution')
store=WorkflowExecutionStore(repo,prepare_workflow(doc),graph_id='graph',attempt=AttemptRef(GenerationRef('execution','generation'),'attempt'))
calls=[]
def code(ctx, inputs):
    calls.append(ctx.node_id)
    assert ctx.node_id == 'join'
    return WorkflowNodeCompleted({'answer':'join'})
result=DurableWorkflowRunner(store,{'code':code}).run()
print(json.dumps({'calls':calls,'outputs':result.outputs,'status':result.status}))
'''
    done = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path), str(FIXTURES / "if_rejoin_workflow.json")],
                          capture_output=True, text=True, check=True)
    assert json.loads(done.stdout) == {"calls": ["join"], "outputs": {"answer": "left"}, "status": "completed"}


def test_two_runners_racing_on_intent_dispatch_only_once(tmp_path, monkeypatch):
    calls = []; first = runner(tmp_path, code(calls)); first.create({"flag": True}); first.run(max_steps=2)
    second = runner(tmp_path, code(calls)); barrier = threading.Barrier(2); results = []
    for current in (first, second):
        original = current.store.save
        def save(state, _save=original, **kwargs):
            if state["status"] == "running" and state["active"]["node_id"] == "left":
                barrier.wait(timeout=10)
            return _save(state, **kwargs)
        monkeypatch.setattr(current.store, "save", save)
    def execute(current):
        try:
            results.append(current.step())
        except Exception as exc:
            results.append(exc)
    threads = [threading.Thread(target=execute, args=(current,)) for current in (first, second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(15)
        assert not thread.is_alive()
    assert calls == ["left"] and sum(isinstance(value, JournalConflictError) for value in results) == 1
    assert runner(tmp_path).inspect().status == "result_received"


def test_pause_between_result_guard_and_cas_does_not_lose_success(tmp_path, monkeypatch):
    calls = []; run = runner(tmp_path, code(calls)); run.create({"flag": True})
    original = run.store.save; paused = False
    def save(state, **kwargs):
        nonlocal paused
        if state["status"] == "result_received" and not paused:
            paused = True
            runner(tmp_path).pause()
        return original(state, **kwargs)
    monkeypatch.setattr(run.store, "save", save)
    result = run.run()
    assert result.paused and result.status == "result_received" and calls == ["left"]
    assert runner(tmp_path, code(calls)).resume().status == "completed" and calls == ["left", "join"]


def test_provider_free_modules_and_real_store_execution_without_credentials(tmp_path):
    script = r'''
import builtins, json, os, sys
from pathlib import Path
original_import=builtins.__import__
def blocked(name,*args,**kwargs):
    if name.startswith(('unchain.providers','unchain.agent','unchain.kernel','openai','anthropic','google.genai','ollama')):
        raise AssertionError('unexpected provider import: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=blocked
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton import prepare_workflow,WorkflowNodeCompleted
from unchain.journal import AttemptRef,GenerationRef
builtins.__import__=original_import
# The legacy concrete persistence package still imports SDK definitions.
# Block actual client creation and credential access, not those imports.
from unchain.persistence import SQLiteContextV2Store
import openai, anthropic
def no_client(*args,**kwargs): raise AssertionError('model client initialized')
for module,names in ((openai,('OpenAI','AsyncOpenAI','AzureOpenAI','AsyncAzureOpenAI')),
                     (anthropic,('Anthropic','AsyncAnthropic'))):
    for name in names:
        if hasattr(module,name): setattr(module,name,no_client)
original_getitem=type(os.environ).__getitem__
def no_key(self,key):
    if any(token in key.upper() for token in ('API_KEY','APIKEY','ACCESS_TOKEN')):
        raise AssertionError('credential access: '+key)
    return original_getitem(self,key)
type(os.environ).__getitem__=no_key
root=Path(sys.argv[1]); doc=json.loads(Path(sys.argv[2]).read_text())
repo=SQLiteContextV2Store(database_path=root/'plain.sqlite3',object_directory=root/'plain-objects').bind_execution('execution')
store=WorkflowExecutionStore(repo,prepare_workflow(doc),graph_id='plain',attempt=AttemptRef(GenerationRef('execution','g'),'a'))
builtins.__import__=blocked
run=DurableWorkflowRunner(store,{'code':lambda ctx,_:WorkflowNodeCompleted({'answer':ctx.node_id})})
run.create({'flag':False}); assert run.run().outputs == {'answer':'right'}
assert run.commit_root().root_committed
print('provider-free execution OK')
'''
    done = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path), str(FIXTURES / "if_rejoin_workflow.json")],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "provider-free execution OK"


def test_missing_binding_fails_before_any_handler_and_end_stays_uncommitted(tmp_path):
    calls = []; run = runner(tmp_path, code(calls)); run.create({})
    result = run.run()
    assert result.status == "failed" and result.error["code"] == "missing_value"
    assert not result.root_committed and result.outputs is None and calls == []


def test_unselected_invalid_config_is_rejected_before_creating_run(tmp_path):
    doc = document(); doc["nodes"][3]["config"]["execution"]["unexpected"] = True
    with pytest.raises(SkeletonValidationError):
        runner(tmp_path, code([]), doc=doc)


def test_switch_real_durable_execution(tmp_path):
    doc = json.loads((FIXTURES / "switch_workflow.json").read_text())
    calls = []
    def execute(ctx, inputs):
        calls.append(ctx.node_id)
        return WorkflowNodeCompleted({"value": ctx.node_id})
    run = runner(tmp_path, execute, doc=doc); run.create({})
    assert run.run().outputs == {"value": "case"} and calls == ["case"]


def test_delayed_wait_response_cannot_complete_another_node(tmp_path):
    run = runner(tmp_path, lambda ctx,_: WorkflowNodeWaiting("interaction", "approve", {"node": ctx.node_id}))
    run.create({"flag": True}); run.run()
    first = run.store.load().state["active"]
    run.complete_wait("approve", {"answer": "first"}, node_invocation_id=first["node_invocation_id"])
    run.run()
    second = run.store.load().state["active"]
    assert first["node_invocation_id"] != second["node_invocation_id"]
    before = run.inspect().revision
    with pytest.raises(SkeletonValidationError):
        run.complete_wait("approve", {"answer": "delayed-first"}, node_invocation_id=first["node_invocation_id"])
    assert run.inspect().revision == before and run.inspect().status == "waiting"
    run.complete_wait("approve", {"answer": "second"}, node_invocation_id=second["node_invocation_id"])
    assert run.run().outputs == {"answer": "first"}


@pytest.mark.parametrize("waiting", [WorkflowNodeWaiting([], "wait", {}), WorkflowNodeWaiting("interaction", [], {}),
                                    WorkflowNodeWaiting("interaction", "wait", []), WorkflowNodeWaiting("unknown", "wait", {})])
def test_invalid_wait_result_is_classified_unknown(tmp_path, waiting):
    calls = []
    def execute(ctx, inputs):
        calls.append(ctx.node_id)
        return waiting
    run = runner(tmp_path, execute); run.create({"flag": True})
    result = run.run()
    assert result.status == "outcome_unknown" and result.outputs is None and calls == ["left"]
    assert runner(tmp_path, code(calls)).run().status == "outcome_unknown" and calls == ["left"]


def test_process_exit_after_external_effect_never_reissues_it(tmp_path):
    run = runner(tmp_path); run.create({"flag": True})
    script = r'''
import json, os, sys
from pathlib import Path
from unchain.journal import AttemptRef,GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import prepare_workflow
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.durable_execution import DurableWorkflowRunner
root=Path(sys.argv[1]); doc=json.loads(Path(sys.argv[2]).read_text())
repo=SQLiteContextV2Store(database_path=root/'journal.sqlite3',object_directory=root/'objects').bind_execution('execution')
store=WorkflowExecutionStore(repo,prepare_workflow(doc),graph_id='graph',attempt=AttemptRef(GenerationRef('execution','generation'),'attempt'))
def effect(ctx, inputs):
    with (root/'effect').open('ab',buffering=0) as f:
        f.write(b'once\n'); os.fsync(f.fileno())
    os._exit(17)
DurableWorkflowRunner(store,{'code':effect}).run()
'''
    done = subprocess.run([sys.executable, "-B", "-c", script, str(tmp_path), str(FIXTURES / "if_rejoin_workflow.json")],
                          capture_output=True, text=True)
    assert done.returncode == 17, done.stderr
    assert (tmp_path / "effect").read_bytes() == b"once\n"
    cold = runner(tmp_path, lambda *_: pytest.fail("unknown side effect reissued"))
    assert cold.inspect().status == "running"
    assert cold.recover().status == "outcome_unknown"
    assert cold.run().status == "outcome_unknown"
    assert (tmp_path / "effect").read_bytes() == b"once\n"


def test_fixed_definition_and_attempt_reject_changed_cold_reader(tmp_path):
    run = runner(tmp_path); run.create({"flag": True})
    doc = document(); doc["name"] = "changed"
    with pytest.raises(SkeletonValidationError):
        runner(tmp_path, doc=doc).inspect()
    with pytest.raises(SkeletonValidationError):
        runner(tmp_path, attempt="another-attempt").inspect()
    assert run.inspect().status == "ready"
