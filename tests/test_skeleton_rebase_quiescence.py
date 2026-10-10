"""Generation edit proofs against real Skeleton/Kernel SQLite producers."""
from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest

from unchain.context import ArtifactService, CanonicalSemanticEventProjector
from unchain.journal import AttemptRef, DurableEventSink, GenerationRef, build_operation_ref
from unchain.journal.models import _thaw_json
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner, WorkflowNodeCompleted, WorkflowNodeWaiting
from unchain.skeleton.durable_protocol import canonical_bytes, validate_execution_transition
from unchain.skeleton.durable_store import DurableStoreError, WorkflowExecutionStore
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.skeleton.kernel_runtime import SkeletonKernelHandler, build_node_bundle
from unchain.skeleton.rebase_quiescence import SkeletonRebaseQuiescenceError, classify_skeleton_attempts


def _doc():
    profile = {"id": "offline-profile", "provider": "openai", "model": "offline-model",
        "protocol": "openai-responses", "endpoint": "https://offline.invalid/v1",
        "credential_ref": "offline", "context_window": 8192, "output_limit": 1024,
        "capabilities": {"json_schema": False, "images": False, "files": False}}
    kernel = {"schema_version": 1, "profile": profile, "prompt": "Return a reply",
        "max_iterations": 2, "output_tokens": 100, "history": "root", "overflow": "error", "json_schema": None}
    output = {"name": "message", "source": {"kind": "result", "path": ["message"]}}
    return {"schema_version": 1, "kind": "agent", "id": "proof-agent", "name": "Proof",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": []}}},
            {"id": "kernel", "type": "agent", "config": {"kernel": kernel,
                "execution": {"inputs": [], "outputs": [output]}}},
            {"id": "end", "type": "end", "config": {"execution": {
                "inputs": [{"name": "message", "value": {"kind": "ref", "node_id": "kernel", "path": ["message"]}, "missing": "error"}],
                "outputs": [{"name": "message", "source": {"kind": "input", "path": ["message"]}}]}}}],
        "edges": [{"id": "a", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "kernel", "target_port_id": "in"},
                  {"id": "b", "kind": "flow", "source_node_id": "kernel", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"}]}


class _Stream:
    def __enter__(self): return self
    def __exit__(self, *_args): return False
    def __iter__(self):
        yield SimpleNamespace(type="response.completed", response=SimpleNamespace(
            id="offline-response", output=[{"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": "  actual native reply  "}]}],
            usage={"input_tokens": 4, "output_tokens": 3, "total_tokens": 7}))


def _produce(tmp_path, *, mutation=None):
    disk = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite", object_directory=tmp_path / "objects")
    journal = disk.bind_execution("proof-execution")
    attempt = AttemptRef(GenerationRef("proof-execution", "proof-generation"), "proof-root")
    plan = prepare_agent_graph(_doc())
    store = WorkflowExecutionStore(journal, plan, graph_id="proof-graph", attempt=attempt)
    calls = []
    def factory(**config):
        calls.append(config)
        def create(**request):
            assert request["model"] == "offline-model"
            return _Stream()
        return SimpleNamespace(responses=SimpleNamespace(create=create))
    def binding(config):
        profile = deepcopy(config["profile"])
        return bind_kernel_profile(config, profile_resolver=lambda _id: profile,
            credential_resolver=lambda _id: "offline-value", client_factories={"openai": factory})
    compiler = SQLiteContextCompilerV2Store(context_store=disk)
    def bundle(*, view, config, **_):
        return build_node_bundle(journal=journal, compiler_store=compiler, view=view, config=config,
            parent_attempt_id=attempt.attempt_id, artifact_sanitizer=lambda content, _media: content,
            payload_sanitizer=lambda _kind, payload: payload, partial_attempt_sink=lambda *_args: None)
    handler = SkeletonKernelHandler(store, bind_profile=binding, bundle_builder=bundle)
    deferred_children = []
    if mutation == "missing_seal":
        handler._save_completion = lambda _context, output: WorkflowNodeCompleted(output)
    elif mutation == "forged_seal":
        original = handler._request
        def forged_request(context, ref):
            request = original(context, ref)
            payload = _thaw_json(request.payload)
            payload["node_id"] = "different-node"
            return replace(request, payload=payload, operation=build_operation_ref(
                request.operation.operation_id, domain=request.event_type, payload=payload))
        handler._request = forged_request
    elif mutation == "late_child_completion":
        original_save = handler._save_completion
        def defer(context, output):
            parent_state = store.load().state
            deferred_children.append(_child(journal, attempt, parent_state,
                                           context.node_invocation_id, "deferred-child"))
            return original_save(context, output)
        handler._save_completion = defer
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({})
    assert runner.run().status == "completed"
    assert len(calls) == 1
    return SimpleNamespace(disk=disk, journal=journal, store=store, runner=runner, attempt=attempt,
                           calls=calls, deferred_children=deferred_children)


@pytest.fixture
def produced(tmp_path):
    return _produce(tmp_path)


def _events(produced):
    return produced.journal.capture_snapshot().events


def _seal_root(produced, *, commit=True, final=True, content=None):
    state = produced.store.load().state
    output = produced.store.read_json(state["completed"][-1]["output_ref"])
    artifacts = ArtifactService(produced.journal, sanitizer=lambda value, _media: value)
    projector = CanonicalSemanticEventProjector(attempt=produced.attempt, artifacts=artifacts,
        payload_sanitizer=lambda _kind, payload: payload)
    sink = DurableEventSink(produced.journal, produced.attempt, projector)
    if final:
        sink({"type": "final_message", "run_id": produced.attempt.attempt_id,
              "iteration": len(state["completed"]), "content": output["message"] if content is None else content})
    sink({"type": "run_completed", "run_id": produced.attempt.attempt_id,
          "iteration": len(state["completed"]), "status": "completed"})
    if commit:
        produced.runner.commit_root()
    return _events(produced)


def _mutate(events, target, **changes):
    return tuple(replace(event, **changes) if event.event_id == target.event_id else event for event in events)


def _changed_state(event, mutate):
    payload = _thaw_json(event.payload)
    import json
    state = json.loads(base64.b64decode(payload["encoded_state"]))
    mutate(state)
    raw = canonical_bytes(state)
    payload["encoded_state"] = base64.b64encode(raw).decode("ascii")
    payload["state_sha256"] = sha256(raw).hexdigest()
    return dict(payload=payload, operation=build_operation_ref(event.operation.operation_id,
        domain="skeleton.workflow_execution.saved", payload=payload))


def test_real_kernel_and_root_native_seals_are_quiescent_without_new_provider_send(produced):
    events = _seal_root(produced)
    seals = [e for e in events if e.event_type == "skeleton.kernel_result.saved"]
    expected = {produced.attempt.attempt_id: "quiescent", seals[0].attempt.attempt_id: "quiescent"}
    assert classify_skeleton_attempts(events, produced.journal) == expected
    # A second pure read performs no provider call or journal append.
    assert classify_skeleton_attempts(events, produced.journal) == expected
    assert _events(produced) == events and len(produced.calls) == 1


def test_real_standalone_completed_graph_needs_no_chat_terminal(produced):
    assert set(classify_skeleton_attempts(_events(produced), produced.journal).values()) == {"quiescent"}
    produced.runner.commit_root()
    assert set(classify_skeleton_attempts(_events(produced), produced.journal).values()) == {"quiescent"}


def test_requested_prefix_never_hides_actual_later_terminal_or_seal(produced):
    events = _events(produced)
    first = next(e for e in events if e.event_type == "skeleton.workflow_execution.saved")
    assert set(classify_skeleton_attempts(tuple(e for e in events if e.store_seq <= first.store_seq), produced.journal).values()) == {"quiescent"}


def test_actual_persisted_waiting_and_cancelled_graphs_remain_open(tmp_path):
    disk = SQLiteContextV2Store(database_path=tmp_path / "waiting.sqlite", object_directory=tmp_path / "objects")
    journal = disk.bind_execution("proof-execution")
    attempt = AttemptRef(GenerationRef("proof-execution", "proof-generation"), "waiting-root")
    store = WorkflowExecutionStore(journal, prepare_agent_graph(_doc()), graph_id="waiting-graph", attempt=attempt)
    runner = DurableWorkflowRunner(store, {"agent": lambda *_args: WorkflowNodeWaiting("interaction", "original-wait", {"request": "original"})})
    runner.create({})
    assert runner.run().status == "waiting"
    assert classify_skeleton_attempts(journal.capture_snapshot().events, journal) == {"waiting-root": "open"}
    runner.cancel()
    assert classify_skeleton_attempts(journal.capture_snapshot().events, journal) == {"waiting-root": "open"}


@pytest.mark.parametrize("mutation", ["missing_seal", "forged_seal"])
def test_actual_persisted_native_producer_missing_or_forged_result_seal_is_rejected(tmp_path, mutation):
    produced = _produce(tmp_path, mutation=mutation)
    events = _seal_root(produced)
    assert events == produced.journal.capture_snapshot().events
    # A producer fault suppresses or corrupts only the result boundary. The
    # real SQLite journal, native provider result, End and root seal remain.
    with pytest.raises(SkeletonRebaseQuiescenceError) as caught:
        classify_skeleton_attempts(events, produced.journal)
    assert caught.value.code in {"kernel_seal_missing", "kernel_seal_identity"}


@pytest.mark.parametrize("owner", ["root", "kernel"])
def test_actual_journal_append_after_terminal_is_rejected(produced, owner):
    events = _seal_root(produced)
    attempt = produced.attempt if owner == "root" else next(e.attempt for e in events if e.event_type == "skeleton.kernel_result.saved")
    projector = CanonicalSemanticEventProjector(attempt=attempt,
        artifacts=ArtifactService(produced.journal, sanitizer=lambda value, _media: value),
        payload_sanitizer=lambda _kind, payload: payload)
    sink = DurableEventSink(produced.journal, attempt, projector)
    sink({"type": "response_received", "run_id": attempt.attempt_id, "iteration": 99})
    assert produced.journal.capture_snapshot().events[-1].event_type == "response_received"
    with pytest.raises(SkeletonRebaseQuiescenceError) as caught:
        classify_skeleton_attempts(_events(produced), produced.journal)
    assert caught.value.code in {"kernel_after_terminal", "graph_root_seal"}


def test_actual_canonical_root_terminal_cannot_commit_another_end_message(produced):
    events = _seal_root(produced, content="A different terminal message")
    with pytest.raises(SkeletonRebaseQuiescenceError) as caught:
        classify_skeleton_attempts(events, produced.journal)
    assert caught.value.code == "graph_root_seal"


@pytest.mark.parametrize("mutation", ["duplicate", "event_id", "operation", "resource", "graph", "node", "definition", "unknown"])
def test_original_kernel_seal_is_required_and_cannot_be_rekeyed_or_widened(produced, mutation):
    events = _seal_root(produced)
    seal = next(e for e in events if e.event_type == "skeleton.kernel_result.saved")
    if mutation == "duplicate": changed = tuple(events) + (replace(seal, event_id="extra-seal", store_seq=events[-1].store_seq + 1),)
    elif mutation == "event_id": changed = _mutate(events, seal, event_id="rekeyed-seal")
    elif mutation == "operation": changed = _mutate(events, seal, operation=build_operation_ref("other", domain=seal.event_type, payload=seal.payload))
    elif mutation == "resource": changed = _mutate(events, seal, resource_refs=())
    else:
        payload = _thaw_json(seal.payload)
        key = {"graph": "graph_id", "node": "node_id", "definition": "definition_digest", "unknown": "unknown"}[mutation]
        payload[key] = "0" * 64 if key == "definition_digest" else "altered"
        changed = _mutate(events, seal, payload=payload, operation=build_operation_ref(seal.operation.operation_id, domain=seal.event_type, payload=payload))
    with pytest.raises(SkeletonRebaseQuiescenceError): classify_skeleton_attempts(changed, produced.journal)


@pytest.mark.parametrize("target_type", ["skeleton.kernel_result.saved", "skeleton.workflow_execution.saved"])
def test_unrelated_work_after_terminal_is_never_an_allowed_seal(produced, target_type):
    events = _seal_root(produced)
    target = next(e for e in events if e.event_type == target_type)
    extra = replace(target, event_id="extra-work", event_type="provider.turn_result",
                    store_seq=events[-1].store_seq + 1, resource_refs=())
    with pytest.raises(SkeletonRebaseQuiescenceError): classify_skeleton_attempts((*events, extra), produced.journal)


@pytest.mark.parametrize("mutation", ["missing_commit", "missing_final", "content", "iteration", "duplicate_terminal", "changed_commit"])
def test_root_seal_requires_exact_end_output_and_unique_terminal(produced, mutation):
    events = _seal_root(produced, commit=mutation != "missing_commit", final=mutation != "missing_final")
    if mutation in {"missing_commit", "missing_final"}:
        with pytest.raises(SkeletonRebaseQuiescenceError): classify_skeleton_attempts(events, produced.journal)
        return
    final = next(e for e in events if e.attempt == produced.attempt and e.event_type == "final_message")
    terminal = next(e for e in events if e.attempt == produced.attempt and e.event_type == "run_completed")
    if mutation == "missing_commit": changed = events[:-1]
    elif mutation == "missing_final": changed = tuple(e for e in events if e is not final)
    elif mutation == "duplicate_terminal": changed = (*events, replace(terminal, event_id="duplicate-terminal", store_seq=events[-1].store_seq + 1))
    elif mutation == "changed_commit": changed = _mutate(events, events[-1], **_changed_state(events[-1], lambda state: state["completed"].pop()))
    else:
        payload = _thaw_json(final.payload)
        payload["content" if mutation == "content" else "iteration"] = "other" if mutation == "content" else 0
        changed = _mutate(events, final, payload=payload)
    with pytest.raises(SkeletonRebaseQuiescenceError): classify_skeleton_attempts(changed, produced.journal)


@pytest.mark.parametrize("mutation", ["revision", "digest", "predecessor", "resource", "unknown"])
def test_every_graph_receipt_is_checked_before_terminal_claim(produced, mutation):
    events = _seal_root(produced)
    graph = next(e for e in events if e.event_type == "skeleton.workflow_execution.saved")
    payload = _thaw_json(graph.payload)
    if mutation == "resource": changed = _mutate(events, graph, resource_refs=())
    else:
        key = {"revision": "revision", "digest": "state_sha256", "predecessor": "previous_event_id", "unknown": "unknown"}[mutation]
        payload[key] = 2 if key == "revision" else "altered"
        changed = _mutate(events, graph, payload=payload, operation=build_operation_ref(graph.operation.operation_id, domain=graph.event_type, payload=payload))
    with pytest.raises(SkeletonRebaseQuiescenceError): classify_skeleton_attempts(changed, produced.journal)


def test_read_only_planless_store_cannot_write_even_initial_ready_state(produced):
    before = _events(produced)
    read_only = WorkflowExecutionStore(produced.journal, None, graph_id=produced.store.graph_id, attempt=produced.attempt)
    with pytest.raises(DurableStoreError) as caught:
        read_only.save(produced.store.load().state, expected_revision=0, operation_id="must-not-write")
    assert caught.value.code == "missing_plan" and _events(produced) == before


def test_self_contained_transition_retains_identity_and_terminal_guards(produced):
    store = produced.store
    chain = WorkflowExecutionStore(produced.journal, None, graph_id=store.graph_id, attempt=store.attempt)._snapshot(
        produced.journal.capture_snapshot(), plan=None, verify_contents=True)
    for before, after in zip(chain, chain[1:]): validate_execution_transition(before.state, after.state, None)
    state = deepcopy(chain[-1].state)
    state["definition_digest"] = "0" * 64
    with pytest.raises(ValueError): validate_execution_transition(chain[-1].state, state, None)


def test_typed_plan_still_rejects_handler_completion_without_an_execution_intent(produced):
    store = produced.store
    chain = store._snapshot(produced.journal.capture_snapshot(), plan=store.plan, verify_contents=True)
    before = next(item.state for item in chain if item.state["status"] == "ready"
                  and len(item.state["completed"]) == 1)
    current = deepcopy(before)
    kernel = next(row for row in chain[-1].state["completed"] if row["node_id"] == "kernel")
    current["completed"].append(deepcopy(kernel))
    # A planless reader verifies closed receipt identities only; real writes
    # still have the prepared plan and enforce the required handler intent.
    validate_execution_transition(before, current, None)
    with pytest.raises(ValueError, match="handler requires an execution intent"):
        validate_execution_transition(before, current, store.plan)
    with pytest.raises(DurableStoreError, match="journal-only execution stores cannot write"):
        WorkflowExecutionStore(store.journal, None, graph_id=store.graph_id, attempt=store.attempt).save(
            current, expected_revision=1, operation_id="untyped-dispatch-bypass")
    state = deepcopy(chain[-1].state)
    state["completed"].pop()
    with pytest.raises(ValueError): validate_execution_transition(chain[-1].state, state, None)


@pytest.mark.parametrize("depth", [1, 2])
def test_actual_fixed_revision_workflow_calls_keep_all_nested_frames_quiescent(tmp_path, depth):
    from tests.test_skeleton_workflow_runtime import host, parent_document, child_document
    from unchain.skeleton.workflow_runtime import PublishedWorkflow
    definitions = {"leaf": child_document("leaf")}
    if depth == 2:
        definitions["middle"] = parent_document(parent_id="middle", workflow_id="leaf")
    target = "middle" if depth == 2 else "leaf"
    def resolver(workflow_id, revision_id):
        return PublishedWorkflow(workflow_id, revision_id, deepcopy(definitions[workflow_id]))
    def code(_context, inputs):
        return WorkflowNodeCompleted({"value": inputs["value"] + "!", "version": "rev-1"})
    parent, _ = host(tmp_path, parent_document(workflow_id=target), resolver, code)
    parent.create({"value": "actual"})
    assert parent.run().outputs == {"value": "actual!", "version": "rev-1"}
    events = parent.store.journal.capture_snapshot().events
    proof = classify_skeleton_attempts(events, parent.store.journal)
    assert len(proof) == depth + 1 and set(proof.values()) == {"quiescent"}


def _child(journal, parent_attempt, parent_state, caller_invocation, graph_id):
    from unchain.skeleton import prepare_workflow
    from unchain.skeleton.agent_preparation import execution_definition_digest
    document = {"schema_version": 1, "kind": "workflow", "id": graph_id, "name": "Child",
        "nodes": [{"id": "start", "type": "start", "config": {"execution": {"outputs": []}}},
                  {"id": "end", "type": "end", "config": {"execution": {"inputs": [], "outputs": []}}}],
        "edges": [{"id": "a", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"}]}
    plan = prepare_workflow(document)
    frames = deepcopy(parent_state["call_stack"])
    frames[-1]["caller_node_invocation_id"] = caller_invocation
    attempt = AttemptRef(parent_attempt.generation, graph_id + "-owner")
    frame = {"execution_id": attempt.generation.execution_id, "graph_invocation_id": graph_id,
        "attempt_id": attempt.attempt_id, "definition_id": plan.document_id,
        "definition_digest": execution_definition_digest(plan)}
    child_store = WorkflowExecutionStore(journal, plan, graph_id=graph_id, attempt=attempt)
    child = DurableWorkflowRunner(child_store, {})
    child.create({}, call_stack=frames + [frame])
    return child


def test_real_child_graph_cannot_claim_a_completed_parent_caller(produced):
    _seal_root(produced)
    parent_state = produced.store.load().state
    caller = next(row for row in parent_state["completed"] if row["node_id"] == "kernel")
    child = _child(produced.journal, produced.attempt, parent_state, caller["node_invocation_id"], "late-child")
    assert child.run().status == "completed"
    with pytest.raises(SkeletonRebaseQuiescenceError) as caught:
        classify_skeleton_attempts(_events(produced), produced.journal)
    assert caught.value.code == "graph_lineage_invocation"


def test_actual_child_admitted_while_parent_active_cannot_finish_after_caller_result(tmp_path):
    produced = _produce(tmp_path, mutation="late_child_completion")
    _seal_root(produced)
    child, = produced.deferred_children
    assert child.run().status == "completed"
    with pytest.raises(SkeletonRebaseQuiescenceError) as caught:
        classify_skeleton_attempts(_events(produced), produced.journal)
    assert caught.value.code == "graph_lineage_completion"
