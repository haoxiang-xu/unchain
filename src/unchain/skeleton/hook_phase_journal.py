"""Durable, parent-scoped receipts for completed Hook Workflow decisions."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json

from unchain.journal import AttemptRef, JournalAppendRequest, build_operation_ref
from unchain.journal.models import _thaw_json

from .agent_preparation import execution_definition_digest
from .errors import SkeletonValidationError


EVENT_TYPE = "skeleton.hook_phase.receipt"
SCHEMA = "unchain.skeleton_hook_phase_receipt.v1"
FIELDS = frozenset({
    "schema", "parent_graph_id", "parent_node_invocation_id", "call_id",
    "tool_name", "phase", "hook_id", "workflow_id", "revision_id",
    "definition_digest", "input_sha256", "decision",
})


def _digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


class HookPhaseJournal:
    """Record one exact Hook decision after its child graph has completed.

    A completed child without this receipt is safe to revisit: the child graph
    replays its own durable End instead of executing completed tools again.
    """

    def __init__(self, parent_store, parent_context, *, call_id, tool_name):
        self._store = parent_store
        self._context = parent_context
        self._call_id = call_id
        self._tool_name = tool_name
        self._attempt = AttemptRef(parent_store.attempt.generation,
            parent_context.node_attempt_id)

    def _identity(self, binding, phase, inputs):
        hook = binding.hook
        if phase not in {"before_tool_call", "after_tool_call"} or hook["event"] != phase:
            raise SkeletonValidationError("identity_mismatch", ("hooks", hook["id"]),
                "Hook phase changed")
        identity = {
            "schema": SCHEMA,
            "parent_graph_id": self._store.graph_id,
            "parent_node_invocation_id": self._context.node_invocation_id,
            "call_id": self._call_id,
            "tool_name": self._tool_name,
            "phase": phase,
            "hook_id": hook["id"],
            "workflow_id": binding.workflow.workflow_id,
            "revision_id": binding.workflow.revision_id,
            "definition_digest": execution_definition_digest(binding.plan),
            "input_sha256": _digest(inputs),
        }
        key = _digest(identity)
        return identity, f"skeleton.hook_phase.{key}"

    def _find(self, identity):
        matches = []
        snapshot = self._store.journal.capture_snapshot()
        for event in snapshot.events:
            if event.event_type != EVENT_TYPE or event.attempt != self._attempt:
                continue
            raw = _thaw_json(event.payload)
            if type(raw) is not dict or set(raw) != FIELDS or raw.get("schema") != SCHEMA:
                raise SkeletonValidationError("invalid_receipt", ("hooks",),
                    "Hook phase receipt schema is invalid")
            if (raw["parent_graph_id"] == identity["parent_graph_id"]
                    and raw["parent_node_invocation_id"] == identity["parent_node_invocation_id"]
                    and raw["call_id"] == identity["call_id"]
                    and raw["hook_id"] == identity["hook_id"]
                    and raw["phase"] == identity["phase"]):
                if any(raw[field] != value for field, value in identity.items()):
                    raise SkeletonValidationError("identity_mismatch", ("hooks", identity["hook_id"]),
                        "Hook phase receipt changed its pinned input or revision")
                operation_id = "skeleton.hook_phase." + _digest(identity)
                if (event.event_id != "event." + operation_id
                        or event.operation != build_operation_ref(
                            operation_id, domain=EVENT_TYPE, payload=raw)
                        or event.resource_refs):
                    raise SkeletonValidationError("invalid_receipt", ("hooks", identity["hook_id"]),
                        "Hook phase receipt operation changed")
                matches.append(raw)
        if len(matches) > 1:
            raise SkeletonValidationError("duplicate_receipt", ("hooks", identity["hook_id"]),
                "Hook phase has duplicate durable decisions")
        return deepcopy(matches[0]["decision"]) if matches else None

    def replay(self, binding, phase, inputs):
        identity, _ = self._identity(binding, phase, inputs)
        return self._find(identity)

    def record(self, binding, phase, inputs, decision):
        self._context.assert_current()
        identity, operation_id = self._identity(binding, phase, inputs)
        existing = self._find(identity)
        if existing is not None:
            if existing != decision:
                raise SkeletonValidationError("identity_mismatch", ("hooks", identity["hook_id"]),
                    "Hook phase decision changed after persistence")
            return existing
        payload = {**identity, "decision": deepcopy(decision)}
        request = JournalAppendRequest(
            event_id="event." + operation_id,
            event_type=EVENT_TYPE,
            attempt=self._attempt,
            operation=build_operation_ref(operation_id, domain=EVENT_TYPE, payload=payload),
            payload=payload,
        )
        self._store.journal.append(request=request)
        recorded = self._find(identity)
        if recorded != decision:
            raise SkeletonValidationError("identity_mismatch", ("hooks", identity["hook_id"]),
                "Hook phase decision was not durably recorded")
        return recorded


__all__ = ["HookPhaseJournal"]
