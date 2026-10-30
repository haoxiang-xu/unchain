"""Audit-only raw Tool receipt that precedes after-Hook child execution."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from unchain.journal import AttemptRef, ArtifactRef, JournalAppendRequest, ResourceRef, build_operation_ref
from unchain.journal.models import _thaw_json

from .errors import SkeletonValidationError


EVENT_TYPE = "tool.hook_raw.sealed"
SCHEMA = "unchain.skeleton_hook_raw_seal.v1"
FIELDS = frozenset({
    "schema", "parent_graph_id", "parent_node_invocation_id", "run_id",
    "iteration", "tool_name", "call_id", "execution_subject",
    "execution_subject_sha256", "hook_manifest_sha256", "raw_artifact",
})
RAW_FIELDS = frozenset({"result", "should_observe", "state_transition", "effective_arguments"})


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _transition_data(value):
    if value is None:
        return None
    return {
        "kind": value.kind,
        "base_state": value.base_state,
        "next_state": value.next_state,
        "handoff_refs": [ref.to_dict() for ref in value.handoff_refs],
    }


def verify_hook_raw_event(event, artifacts, *, attempt, tool_name, call_id,
                          iteration, recorded_subject):
    """Verify one audit seal without making its bytes model-visible."""
    if event is None or event.event_type != EVENT_TYPE or event.attempt != attempt:
        raise SkeletonValidationError("invalid_receipt", ("hooks",),
            "Hook raw seal is missing or escaped its Tool attempt")
    payload = _thaw_json(event.payload)
    if type(payload) is not dict or set(payload) != FIELDS or payload.get("schema") != SCHEMA:
        raise SkeletonValidationError("invalid_receipt", ("hooks",),
            "Hook raw seal schema is invalid")
    if (any(type(payload[field]) is not str or not payload[field].strip()
            for field in ("parent_graph_id", "parent_node_invocation_id"))
            or type(payload["hook_manifest_sha256"]) is not str
            or len(payload["hook_manifest_sha256"]) != 64
            or any(char not in "0123456789abcdef"
                for char in payload["hook_manifest_sha256"])):
        raise SkeletonValidationError("invalid_receipt", ("hooks",),
            "Hook raw seal parent identity or manifest digest is invalid")
    if (payload["run_id"] != attempt.attempt_id
            or payload["tool_name"] != tool_name
            or payload["call_id"] != call_id
            or payload["iteration"] != iteration
            or payload["execution_subject"] != recorded_subject.to_dict()
            or payload["execution_subject_sha256"] != recorded_subject.sha256):
        raise SkeletonValidationError("identity_mismatch", ("hooks",),
            "Hook raw seal changed its durable Tool subject")
    key = ("hook-raw." + payload["parent_graph_id"] + "."
           + payload["parent_node_invocation_id"] + "." + call_id)
    operation_id = "skeleton.hook_raw." + sha256(key.encode()).hexdigest()
    if (event.event_id != "event." + operation_id
            or event.operation != build_operation_ref(
                operation_id, domain=EVENT_TYPE, payload=payload)):
        raise SkeletonValidationError("invalid_receipt", ("hooks",),
            "Hook raw seal operation changed")
    try:
        artifact = ArtifactRef.from_dict(payload["raw_artifact"])
        if (artifact.media_type != "application/octet-stream" or artifact.preview
                or tuple(event.resource_refs) != (artifact.ref,)):
            raise ValueError("raw artifact is publicly previewable")
        content = artifacts.read_full(artifact, remaining_budget_bytes=artifact.byte_length)
        raw = json.loads(content.decode("utf-8"))
        if type(raw) is not dict or set(raw) != RAW_FIELDS or _json(raw) != content:
            raise ValueError("raw Tool completion is not canonical")
        from unchain.context.tool_boundary import _canonical_digest
        if (type(raw["effective_arguments"]) is not dict
                or _canonical_digest(raw["effective_arguments"])
                != recorded_subject.effective_arguments_sha256):
            raise ValueError("raw Tool arguments changed")
        from unchain.context.tool_executor import DurableToolCompletionDraft
        from unchain.context.tool_transitions import DurableToolStateTransitionDraft
        transition = raw["state_transition"]
        if transition is not None:
            if type(transition) is not dict or set(transition) != {
                "kind", "base_state", "next_state", "handoff_refs",
            }:
                raise ValueError("raw Tool transition is invalid")
            transition = DurableToolStateTransitionDraft(
                kind=transition["kind"], base_state=transition["base_state"],
                next_state=transition["next_state"],
                handoff_refs=tuple(ResourceRef.from_dict(ref)
                    for ref in transition["handoff_refs"]),
            )
        draft = DurableToolCompletionDraft(result=raw["result"],
            should_observe=raw["should_observe"], state_transition=transition)
    except (TypeError, ValueError, UnicodeError, RuntimeError, KeyError) as exc:
        raise SkeletonValidationError("invalid_receipt", ("hooks",),
            "Hook raw artifact is invalid") from exc
    return payload, artifact, draft, deepcopy(raw["effective_arguments"])


class HookRawSealStore:
    """One exact raw-result seal in the indexed Tool lifecycle.

    Its binary artifact has an empty preview. Only the final visible Tool
    result becomes a ``tool_result`` journal event consumed by the compiler.
    """

    def __init__(self, parent_store, parent_context, *, manifest_sha256):
        self._store = parent_store
        self._context = parent_context
        self._manifest = manifest_sha256
        self._attempt = AttemptRef(parent_store.attempt.generation,
            parent_context.node_attempt_id)

    def _events(self, call_id):
        lookup = self._store.journal.lookup_tool_execution_receipts(
            attempt=self._attempt, call_id=call_id)
        if lookup.overflow:
            raise SkeletonValidationError("invalid_receipt", ("hooks",),
                "Hook Tool receipt index overflowed")
        events = [event for event in lookup.events if event.event_type == EVENT_TYPE]
        if len(events) > 1:
            raise SkeletonValidationError("duplicate_receipt", ("hooks",),
                "Hook Tool has duplicate raw result seals")
        return events

    def load(self, request, artifacts):
        events = self._events(request.call_id)
        if not events:
            return None
        event = events[0]
        from unchain.context.tool_boundary import DurableToolExecutionSubject
        raw_subject = _thaw_json(event.payload).get("execution_subject")
        try:
            recorded_subject = DurableToolExecutionSubject.from_dict(raw_subject)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise SkeletonValidationError("invalid_receipt", ("hooks",),
                "Hook raw seal execution subject is invalid") from exc
        payload, artifact, draft, effective_arguments = verify_hook_raw_event(event, artifacts,
            attempt=self._attempt, tool_name=request.tool_name,
            call_id=request.call_id, iteration=request.iteration,
            recorded_subject=recorded_subject)
        expected = {
            "parent_graph_id": self._store.graph_id,
            "parent_node_invocation_id": self._context.node_invocation_id,
            "run_id": self._attempt.attempt_id,
            "iteration": request.iteration,
            "tool_name": request.tool_name,
            "call_id": request.call_id,
            "hook_manifest_sha256": self._manifest,
        }
        if (event.attempt != self._attempt
                or any(payload.get(key) != value for key, value in expected.items())
            or payload.get("execution_subject_sha256") != recorded_subject.sha256):
            raise SkeletonValidationError("identity_mismatch", ("hooks",),
                "Hook raw seal changed its parent Tool or pinned manifest")
        return event, artifact, draft, effective_arguments

    def record(self, request, raw_draft, artifacts, effective_arguments):
        self._context.assert_current()
        existing = self.load(request, artifacts)
        if existing is not None:
            if (existing[2].result != raw_draft.result
                    or existing[2].should_observe != raw_draft.should_observe
                    or existing[2].state_transition != raw_draft.state_transition
                    or existing[3] != effective_arguments):
                raise SkeletonValidationError("identity_mismatch", ("hooks",),
                    "Hook raw result changed after sealing")
            return existing
        raw = {
            "result": _thaw_json(raw_draft.result),
            "should_observe": raw_draft.should_observe,
            "state_transition": _transition_data(raw_draft.state_transition),
            "effective_arguments": deepcopy(effective_arguments),
        }
        content = _json(raw)
        key = ("hook-raw." + self._store.graph_id + "."
               + self._context.node_invocation_id + "." + request.call_id)
        descriptor, pending = self._store.prepare_bytes(
            content, media_type="application/octet-stream", key=key)
        operation_id = "skeleton.hook_raw." + sha256(key.encode()).hexdigest()
        payload = {
            "schema": SCHEMA,
            "parent_graph_id": self._store.graph_id,
            "parent_node_invocation_id": self._context.node_invocation_id,
            "run_id": self._attempt.attempt_id,
            "iteration": request.iteration,
            "tool_name": request.tool_name,
            "call_id": request.call_id,
            "execution_subject": request.subject.to_dict(),
            "execution_subject_sha256": request.subject.sha256,
            "hook_manifest_sha256": self._manifest,
            "raw_artifact": descriptor,
        }
        journal_request = JournalAppendRequest(
            event_id="event." + operation_id,
            event_type=EVENT_TYPE,
            attempt=self._attempt,
            operation=build_operation_ref(operation_id, domain=EVENT_TYPE, payload=payload),
            payload=payload,
            resource_refs=(ResourceRef.from_dict(descriptor["ref"]),),
        )

        def precondition(snapshot):
            matching = [event for event in snapshot.events
                if event.attempt == self._attempt
                and event.payload.get("call_id") == request.call_id
                and event.event_type in {"tool.started", EVENT_TYPE, "tool_result", "tool.result"}]
            if len(matching) != 1 or matching[0].event_type != "tool.started":
                raise SkeletonValidationError("invalid_receipt", ("hooks",),
                    "Hook raw seal requires one unfinished Tool start")
        self._store.journal.append_with_artifacts(
            request=journal_request, artifacts=(pending,), precondition=precondition)
        sealed = self.load(request, artifacts)
        if sealed is None:
            raise SkeletonValidationError("invalid_receipt", ("hooks",),
                "Hook raw result was not durable")
        return sealed


__all__ = ["EVENT_TYPE", "HookRawSealStore", "verify_hook_raw_event"]
