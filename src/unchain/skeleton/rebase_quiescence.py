"""Read-only proofs that Skeleton attempts no longer own runnable work.

This is deliberately not a terminal-event allowlist. Every graph receipt and
its immutable artifacts must validate first. A native Kernel terminal may be
followed only by its exact result receipt; a chat root terminal may be followed
by one validated completion audit and its exact End acknowledgement. Legacy
attempts remain unclassified.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from hashlib import sha256

from unchain.context import ArtifactService, CanonicalSemanticEventProjector
from unchain.journal import (
    ArtifactRef, AttemptRef, JournalAppendRequest, JournalEvent, SemanticEventDraft,
    build_operation_ref,
)
from unchain.journal.graph_attempt_quiescence import CANONICAL_ATTEMPT_TERMINALS
from unchain.journal.models import _thaw_json

from .durable_protocol import canonical_bytes, decode_json
from .durable_store import WorkflowExecutionStore


_GRAPH_EVENT = "skeleton.workflow_execution.saved"
_KERNEL_EVENT = "skeleton.kernel_result.saved"


class SkeletonRebaseQuiescenceError(ValueError):
    """Skeleton durable evidence is inconsistent; no edit may cut it over."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__("Skeleton execution quiescence evidence is invalid")


def _reject(code):
    raise SkeletonRebaseQuiescenceError(code)


def _matches_request(event, request):
    return (event.attempt == request.attempt and event.event_id == request.event_id
            and event.event_type == request.event_type and event.operation == request.operation
            and event.resource_refs == request.resource_refs
            and canonical_bytes(_thaw_json(event.payload)) == canonical_bytes(_thaw_json(request.payload)))


def _root_completion_audit(event, attempt):
    """Validate the existing Host's content-free v1 enqueue decision, not work."""
    statuses = {"memory.curator.enqueued": "Enqueued", "memory.curator.noop": "NoOp",
                "memory.curator.isolated": "Isolated", "memory.curator.failed": "Failed"}
    payload = _thaw_json(event.payload)
    keys = {"schema", "backend", "status", "enqueue_status", "reason", "trigger",
            "run_id", "job_id", "candidate_count", "worker_status"}
    status = statuses.get(event.event_type)
    if (event.attempt != attempt or set(payload) != keys or status is None
            or payload["schema"] != "pupu.memory_curator_completion.v1"
            or payload["backend"] != "unchain" or payload["status"] != status
            or payload["enqueue_status"] != status or payload["trigger"] != "completed_root_run"
            or payload["run_id"] != attempt.attempt_id or type(payload["reason"]) is not str
            or not payload["reason"] or type(payload["job_id"]) is not str
            or type(payload["candidate_count"]) is not int or payload["candidate_count"] < 0
            or payload["worker_status"] != ("Pending" if status == "Enqueued" else "NotScheduled")
            or (status == "Enqueued" and not payload["job_id"])):
        _reject("graph_root_audit")
    operation_id = event.operation.operation_id
    digest = operation_id.removeprefix("memory-curator-audit.")
    if (operation_id != "memory-curator-audit." + digest or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)):
        _reject("graph_root_audit")
    request = SemanticEventDraft(event_id="memory-curator-audit-" + digest,
        event_type=event.event_type, attempt=attempt, operation_id=operation_id,
        payload=payload).to_append_request()
    if not _matches_request(event, request):
        _reject("graph_root_audit")


def _native_completion(events, journal, attempt, *, parent_id=None, expected_message=None,
                       expected_iteration=None):
    terminals = [e for e in events if e.event_type in CANONICAL_ATTEMPT_TERMINALS]
    finals = [e for e in events if e.event_type == "final_message"]
    if not terminals and not finals:
        return None
    if len(terminals) != 1 or len(finals) != 1 or terminals[0].event_type != "run_completed":
        _reject("native_completion_ambiguous")
    final, terminal = finals[0], terminals[0]
    f, t = _thaw_json(final.payload), _thaw_json(terminal.payload)
    extra = {"parent_run_id"} if parent_id is not None else set()
    if (set(f) != {"run_id", "iteration", "content"} | extra
            or set(t) != {"run_id", "iteration", "status"} | extra
            or f["run_id"] != attempt.attempt_id or t["run_id"] != attempt.attempt_id
            or type(f["iteration"]) is not int or f["iteration"] < 0
            or t["iteration"] != f["iteration"] or t["status"] != "completed"
            or type(f["content"]) is not str or final.store_seq >= terminal.store_seq
            or (parent_id is not None and (f["parent_run_id"] != parent_id or t["parent_run_id"] != parent_id))
            or (expected_message is not None and f["content"] != expected_message)
            or (expected_iteration is not None and f["iteration"] != expected_iteration)):
        _reject("native_completion_identity")
    # Bounded final/terminal projection does not put artifacts or call a model.
    artifacts = ArtifactService(journal, sanitizer=lambda value, _media: value)
    def sanitize(_kind, payload):
        return {**payload, **({"parent_run_id": parent_id} if parent_id is not None else {})}
    projector = CanonicalSemanticEventProjector(attempt=attempt, artifacts=artifacts,
                                                payload_sanitizer=sanitize)
    for event in (final, terminal):
        draft = projector({"type": event.event_type, **_thaw_json(event.payload)})
        if draft is None or not _matches_request(event, draft.to_append_request()):
            _reject("native_completion_receipt")
    return final, terminal


def _kernel_seal(event, store, state, row):
    attempt = AttemptRef(store.attempt.generation, row["node_attempt_id"])
    identity = {"schema": "unchain.skeleton_kernel_result.v1", "graph_id": state["graph_id"],
                "node_id": row["node_id"], "node_invocation_id": row["node_invocation_id"],
                "attempt": attempt.to_dict(), "definition_digest": state["definition_digest"]}
    payload = _thaw_json(event.payload)
    if set(payload) != set(identity) | {"result_ref"} or any(payload[k] != v for k, v in identity.items()):
        _reject("kernel_seal_identity")
    operation_id = "skeleton.kernel-result." + sha256(canonical_bytes(identity)).hexdigest()
    ref = ArtifactRef.from_dict(payload["result_ref"])
    if ref.to_dict() != payload["result_ref"]:
        _reject("kernel_seal_artifact")
    request = JournalAppendRequest(event_id="event." + operation_id, event_type=_KERNEL_EVENT,
        attempt=attempt, operation=build_operation_ref(operation_id, domain=_KERNEL_EVENT, payload=payload),
        payload=payload, resource_refs=(ref.ref,))
    if not _matches_request(event, request):
        _reject("kernel_seal_receipt")
    expected_id = store.journal.artifact_id_for(logical_kind="artifact",
        logical_key=store._artifact_operation_id(row["node_invocation_id"] + ".kernel-result"))
    if ref.ref.resource_id != expected_id:
        _reject("kernel_seal_artifact")
    result = store.read_json(payload["result_ref"])
    if type(result) is not dict:
        _reject("kernel_result_shape")
    if (set(result) == {"schema", "code"} and result["schema"] == "unchain.kernel_failure.v1"
            and result["code"] in {"context_overflow", "unsupported_capability"}):
        return result, False
    if set(result) != {"message", "context"} or type(result["message"]) is not str or type(result["context"]) is not dict:
        _reject("kernel_result_shape")
    return result, True


def _resolved_wait(store, state, groups):
    """Prove the original parked interaction and its one canonical receipt."""
    from unchain.interaction.durable import InteractionRequest
    active = state["active"]
    wait = active["wait"]
    events = groups.get(active["node_attempt_id"], [])
    if not any(event.event_type == "interaction.resolved" and event.payload.get("interaction_id") == wait["id"]
               for event in events):
        return False
    request_value = store.read_json(wait["request_ref"])
    request = InteractionRequest.from_dict(request_value)
    attempt = AttemptRef(store.attempt.generation, active["node_attempt_id"])
    if request.interaction_id != wait["id"] or request.source_run_id != attempt.attempt_id:
        _reject("cancel_wait_identity")
    requested = [e for e in events if e.event_type == "interaction.requested"
                 and e.payload.get("interaction_id") == request.interaction_id]
    resolved = [e for e in events if e.event_type == "interaction.resolved"
                and e.payload.get("interaction_id") == request.interaction_id]
    if not resolved:
        return False
    if len(requested) != 1 or len(resolved) != 1:
        _reject("cancel_wait_ambiguous")
    original, answer = requested[0], resolved[0]
    if (_thaw_json(original.payload).get("interaction_request") != request.to_dict()
            or original.store_seq >= answer.store_seq or events[-1] != answer):
        _reject("cancel_wait_receipt")
    value = answer.payload
    from unchain.journal import ResourceRef
    artifact = ArtifactRef(ResourceRef.from_dict(value["content_ref"]), "application/json",
                           value["content_bytes"], value["content_sha256"], value.get("preview", ""))
    response = decode_json(store.journal.read_full_verified(artifact=artifact))
    if (type(response) is not dict or set(response) != {"interaction_id", "response", "submitted_by"}
            or response["interaction_id"] != request.interaction_id or response["submitted_by"] != "user"):
        _reject("cancel_wait_response")
    parent = original.payload.get("parent_run_id")
    if parent is not None and parent != store.attempt.attempt_id:
        _reject("cancel_wait_parent")
    artifacts = ArtifactService(store.journal, sanitizer=lambda content, _media: content)
    def sanitize(_kind, payload):
        return {**payload, **({"parent_run_id": parent} if parent is not None else {})}
    projector = CanonicalSemanticEventProjector(attempt=attempt, artifacts=artifacts,
                                                payload_sanitizer=sanitize)
    expected_request = projector({"type": "interaction_requested", "run_id": attempt.attempt_id,
        "iteration": original.payload["iteration"], "interaction_request": request.to_dict()})
    expected_answer = projector.prepare_interaction_resolution(interaction_id=request.interaction_id,
        response=response["response"], submitted_by="user")
    if (not _matches_request(original, expected_request.to_append_request())
            or not _matches_request(answer, expected_answer.draft.to_append_request())):
        _reject("cancel_wait_canonical")
    return True


def _classify(events, journal):
    requested = tuple(events)
    if not requested:
        return {}
    # Rebase excludes its sealed import range from runtime_events; that leaves
    # holes and is not itself a journal snapshot. Use the verified full prefix,
    # then bind the requested generation to its exact persisted events.
    snapshot = journal.capture_snapshot()
    persisted = {event.event_id: event for event in snapshot.events}
    generation = requested[0].attempt.generation
    if (any(event.attempt.generation != generation or persisted.get(event.event_id) != event for event in requested)
            or len({event.event_id for event in requested}) != len(requested)):
        _reject("snapshot_scope")
    generation_events = tuple(event for event in snapshot.events if event.attempt.generation == generation)
    groups = defaultdict(list)
    for event in generation_events:
        groups[event.attempt.attempt_id].append(event)
    decoded = []
    for index, event in enumerate(snapshot.events):
        if event.event_type == _GRAPH_EVENT and event.attempt.generation == generation:
            # The decoder obtains graph identity from a closed, digest checked
            # state; it does not trust the payload's graph-key as a filter.
            probe = WorkflowExecutionStore(journal, None, graph_id="quiescence-probe", attempt=event.attempt)
            decoded.append(probe._decode_event(event, index))
    stores, chains, owners = {}, {}, {}
    for item in decoded:
        graph_id = item.state["graph_id"]
        if graph_id not in stores:
            store = WorkflowExecutionStore(journal, None, graph_id=graph_id, attempt=item.event.attempt)
            chain = store._snapshot(snapshot, plan=None, verify_contents=True)
            stores[graph_id], chains[graph_id] = store, chain
            owner = store.attempt.attempt_id
            if owner in owners:
                _reject("graph_owner_ambiguous")
            owners[owner] = graph_id

    # Every child frame must name an actual ancestor and its actual invocation,
    # in the same generation, before that child's first receipt.
    for graph_id, chain in chains.items():
        state = chain[-1].state
        for index, frame in enumerate(state["call_stack"][:-1]):
            parent_chain = chains.get(frame["graph_invocation_id"])
            if parent_chain is None:
                _reject("graph_lineage_missing")
            parent_state = parent_chain[-1].state
            clean_frame = {k: v for k, v in frame.items() if k != "caller_node_invocation_id"}
            expected_frames = state["call_stack"][:index] + [clean_frame]
            if (parent_state["call_stack"] != expected_frames
                    or stores[graph_id].attempt.generation != stores[parent_state["graph_id"]].attempt.generation):
                _reject("graph_lineage_identity")
            caller = frame["caller_node_invocation_id"]
            preceding = [ancestor for ancestor in parent_chain if ancestor.event.store_seq < chain[0].event.store_seq]
            parent_at_admission = preceding[-1].state if preceding else None
            if (parent_at_admission is None or parent_at_admission["status"] not in {"running", "waiting"}
                    or parent_at_admission["active"] is None
                    or parent_at_admission["active"]["node_invocation_id"] != caller):
                _reject("graph_lineage_invocation")
            if state["status"] == "completed":
                completed_at = next(item.event.store_seq for item in chain if item.state["status"] == "completed")
                closed_caller = [ancestor for ancestor in parent_chain if any(
                    row["node_invocation_id"] == caller for row in ancestor.state["completed"])
                    or (ancestor.state["active"] is not None
                        and ancestor.state["active"]["node_invocation_id"] == caller
                        and ancestor.state["active"]["result_ref"] is not None)]
                if closed_caller and completed_at >= closed_caller[0].event.store_seq:
                    _reject("graph_lineage_completion")

    result, node_owners, valid_seals = {}, {}, set()
    for graph_id, chain in chains.items():
        store, state = stores[graph_id], chain[-1].state
        owner = store.attempt.attempt_id
        result[owner] = "open"
        rows = {row["node_attempt_id"]: row for item in chain
                for row in item.state["completed"] + ([item.state["active"]] if item.state["active"] else [])}
        latest_rows = {row["node_attempt_id"]: row for row in state["completed"]}
        for node_attempt_id, row in rows.items():
            node_events = groups.get(node_attempt_id, [])
            seals = [e for e in node_events if e.event_type == _KERNEL_EVENT]
            native = [e for e in node_events if e.event_type in CANONICAL_ATTEMPT_TERMINALS or e.event_type == "final_message"]
            if not seals and not native:
                continue
            if node_attempt_id in node_owners or node_attempt_id in owners:
                _reject("kernel_owner_ambiguous")
            node_owners[node_attempt_id] = graph_id
            result[node_attempt_id] = "open"
            if not seals:
                if any(e.event_type == "run_completed" for e in native):
                    _reject("kernel_seal_missing")
                continue
            if len(seals) != 1:
                _reject("kernel_seal_ambiguous")
            seal = seals[0]
            output, success = _kernel_seal(seal, store, state, row)
            valid_seals.add(seal.event_id)
            if not success:
                continue
            completion = _native_completion(node_events, journal, seal.attempt, parent_id=owner,
                                             expected_message=output["message"].strip())
            if completion is None:
                _reject("kernel_terminal_missing")
            _, terminal = completion
            if seal.store_seq <= terminal.store_seq or [e for e in node_events if e.store_seq > terminal.store_seq] != [seal]:
                _reject("kernel_after_terminal")
            committed_row = latest_rows.get(node_attempt_id)
            if committed_row is not None:
                if canonical_bytes(store.read_json(committed_row["result_ref"])) != canonical_bytes(output):
                    _reject("kernel_parent_result")
                saved_raw = next((item for item in chain if item.state["active"] is not None
                    and item.state["active"]["node_attempt_id"] == node_attempt_id
                    and item.state["active"]["result_ref"] == committed_row["result_ref"]), None)
                if saved_raw is None or seal.store_seq >= saved_raw.event.store_seq:
                    _reject("kernel_parent_order")
                result[node_attempt_id] = "quiescent"

        owner_events = groups[owner]
        if state["status"] != "completed":
            if any(e.event_type in {"run_completed", "final_message"} for e in owner_events):
                _reject("graph_terminal_without_end")
            continue
        native = _native_completion(owner_events, journal, store.attempt)
        end = state["completed"][-1]
        if end["edge_id"] is not None:
            _reject("graph_end_missing")
        completed_receipt = next(item for item in chain if item.state["status"] == "completed")
        after_end = [e for e in owner_events if e.store_seq > completed_receipt.event.store_seq]
        if native is not None:
            final, terminal = native
            outputs = store.read_json(end["output_ref"])
            root_seal = [final, terminal]
            if len(after_end) == 4 and after_end[2].event_type.startswith("memory.curator."):
                _root_completion_audit(after_end[2], store.attempt)
                root_seal.append(after_end[2])
            root_seal.append(chain[-1].event)
            if (len(state["call_stack"]) != 1 or not state["root_committed"]
                    or type(outputs.get("message")) is not str
                    or final.payload["content"] != outputs["message"]
                    or final.payload["iteration"] != len(state["completed"])
                    or completed_receipt.event.store_seq >= final.store_seq
                    or len(chain) < 2 or chain[-2] != completed_receipt
                    or chain[-1].event.store_seq <= terminal.store_seq
                    or after_end != root_seal):
                _reject("graph_root_seal")
        elif after_end:
            # A standalone root acknowledgement has no native chat terminal;
            # its exact false→true transition was already checked by the store.
            if not state["root_committed"] or after_end != [chain[-1].event] or chain[-2] != completed_receipt:
                _reject("graph_after_end")
        result[owner] = "quiescent"

    # Failed/cancelled graph authority can also be fully closed. A ready
    # preflight failure owns no active work; a saved raw result has already
    # ended its effect. A parked cancellation needs its original canonical
    # interaction receipt, or fully closed actual descendant graphs.
    for _ in range(len(chains) + 1):
        changed = False
        for graph_id, chain in chains.items():
            store, state = stores[graph_id], chain[-1].state
            owner = store.attempt.attempt_id
            if result[owner] == "quiescent" or state["status"] not in {"failed", "cancelled"}:
                continue
            if any(e.store_seq > chain[-1].event.store_seq for e in groups[owner]):
                continue
            active = state["active"]
            closed = active is None
            if active is not None and active["result_ref"] is not None:
                closed = len(chain) > 1 and chain[-2].state["status"] == "result_received"
            if active is not None and active["wait"] is not None and state["status"] == "cancelled":
                if active["wait"]["kind"] == "interaction":
                    closed = _resolved_wait(store, state, groups)
                    if closed:
                        result[active["node_attempt_id"]] = "quiescent"
                else:
                    children = [child[-1].state for child in chains.values()
                        if len(child[-1].state["call_stack"]) == len(state["call_stack"]) + 1
                        and child[-1].state["call_stack"][-2]["graph_invocation_id"] == graph_id
                        and child[-1].state["call_stack"][-2]["caller_node_invocation_id"] == active["node_invocation_id"]]
                    closed = bool(children) and all(result.get(child["attempt"]["attempt_id"]) == "quiescent" for child in children)
                    if closed:
                        result[active["node_attempt_id"]] = "quiescent"
            if closed:
                result[owner] = "quiescent"
                changed = True
        if not changed:
            break
    if any(e.event_type == _KERNEL_EVENT and e.event_id not in valid_seals for e in generation_events):
        _reject("kernel_graph_missing")
    return result


def classify_skeleton_attempts(events: Sequence[JournalEvent], journal) -> dict[str, str]:
    """Classify only exact Skeleton graph and native Kernel attempt evidence.

    Open tool/checkpoint and legacy-attempt guards remain the caller's job.
    """
    try:
        return _classify(events, journal)
    except SkeletonRebaseQuiescenceError:
        raise
    except Exception as exc:
        raise SkeletonRebaseQuiescenceError("invalid_durable_evidence") from exc


__all__ = ["SkeletonRebaseQuiescenceError", "classify_skeleton_attempts"]
