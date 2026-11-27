"""Durable, provider-free recovery descriptors for prepared Workflows.

This module is deliberately an explicit Skeleton submodule.  Importing
``unchain.skeleton`` must remain independent of the journal/persistence stack.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from unchain.journal import (
    AttemptRef,
    BoundExecutionJournal,
    JournalAppendRequest,
    JournalAppendResult,
    JournalConflictError,
    JournalEvent,
    JournalRepositoryError,
    JournalSnapshot,
    OperationRef,
    build_operation_ref,
    capture_journal_snapshot,
)

from .errors import SkeletonValidationError
from .preparation import PreparedWorkflowPlan
from .recovery import validate_workflow_recovery_envelope


_EVENT_TYPE = "skeleton.workflow_recovery.saved"
_PAYLOAD_SCHEMA = "unchain.skeleton_workflow_recovery_journal.v1"
_CODEC = "json-utf8-surrogatepass-base64.v1"
_OPERATION_DOMAIN = "skeleton.workflow_recovery.saved"
_MAX_DECODED_BYTES = 8 * 1024 * 1024
_MAX_ENCODED_BYTES = 4 * ((_MAX_DECODED_BYTES + 2) // 3)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class WorkflowRecoveryJournalError(SkeletonValidationError):
    """A persisted Workflow recovery descriptor is malformed or inconsistent."""


@dataclass(frozen=True)
class WorkflowRecoveryRecord:
    """One detached, validated durable descriptor and its journal identity."""

    revision: int
    envelope: dict[str, Any]
    event_id: str
    store_seq: int
    operation_id: str


@dataclass(frozen=True)
class _RecoveryEvent:
    event: JournalEvent
    snapshot_index: int
    revision: int
    graph_key: str
    previous_event: dict[str, Any] | None
    envelope: dict[str, Any]


def _raise(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise WorkflowRecoveryJournalError(code, path, message)


def _exact_text(value: Any, path: tuple[str | int, ...], label: str) -> str:
    if type(value) is not str:
        _raise("invalid_type", path, f"{label} must be text")
    if not value:
        _raise("invalid_value", path, f"{label} must be non-empty")
    return value


def _exact_positive_int(value: Any, path: tuple[str | int, ...], label: str) -> int:
    if type(value) is not int:
        _raise("invalid_type", path, f"{label} must be an integer")
    if value < 1:
        _raise("invalid_value", path, f"{label} must be positive")
    return value


def _exact_nonnegative_int(value: Any, path: tuple[str | int, ...], label: str) -> int:
    if type(value) is not int:
        _raise("invalid_type", path, f"{label} must be an integer")
    if value < 0:
        _raise("invalid_value", path, f"{label} must be non-negative")
    return value


def _closed_mapping(
    value: Any,
    path: tuple[str | int, ...],
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise("invalid_type", path, "value must be an object")
    keys = set(value)
    for key in keys:
        if type(key) is not str:
            _raise("invalid_json", path, "object keys must be strings")
    missing = required - keys
    unknown = keys - required
    if missing:
        _raise("missing_field", path + (sorted(missing)[0],), "required field is missing")
    if unknown:
        _raise("unknown_field", path + (sorted(unknown)[0],), "unknown field")
    return value


def _clone(value: Any) -> Any:
    # CP-7 has already established that every envelope is JSON.  deepcopy also
    # breaks any MappingProxy instances introduced by journal model freezing.
    return deepcopy(value)


def _graph_key(graph_invocation_id: str) -> str:
    return hashlib.sha256(graph_invocation_id.encode("utf-8", "surrogatepass")).hexdigest()


def _canonical_envelope_bytes(envelope: Any, path: tuple[str | int, ...]) -> bytes:
    try:
        return json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8", "surrogatepass")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise WorkflowRecoveryJournalError(
            "invalid_json", path, "envelope cannot be canonically encoded"
        ) from exc


def _encode_envelope(envelope: dict[str, Any]) -> tuple[str, str]:
    raw = _canonical_envelope_bytes(envelope, ("envelope",))
    if len(raw) > _MAX_DECODED_BYTES:
        _raise("limit_bytes", ("envelope",), "encoded envelope exceeds eight MiB")
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def _strict_json_object(raw: bytes, path: tuple[str | int, ...]) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _raise("duplicate_name", path, "JSON object contains a duplicate key")
            result[key] = value
        return result

    try:
        decoded = json.loads(
            raw.decode("utf-8", "surrogatepass"),
            object_pairs_hook=reject_duplicates,
        )
    except WorkflowRecoveryJournalError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise WorkflowRecoveryJournalError(
            "invalid_json", path, "encoded envelope is not valid JSON"
        ) from exc
    if type(decoded) is not dict:
        _raise("invalid_type", path, "encoded envelope must be an object")
    return decoded


def _decode_envelope(
    encoded: Any,
    digest: Any,
    path: tuple[str | int, ...],
) -> dict[str, Any]:
    text = _exact_text(encoded, path + ("encoded_envelope",), "encoded envelope")
    if len(text) > _MAX_ENCODED_BYTES:
        _raise("limit_bytes", path + ("encoded_envelope",), "encoded envelope exceeds eight MiB")
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        _raise("invalid_value", path + ("envelope_sha256",), "envelope digest is invalid")
    try:
        transport = text.encode("ascii")
        raw = base64.b64decode(transport, validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise WorkflowRecoveryJournalError(
            "invalid_value", path + ("encoded_envelope",), "encoded envelope is not strict base64"
        ) from exc
    if len(raw) > _MAX_DECODED_BYTES:
        _raise("limit_bytes", path + ("encoded_envelope",), "decoded envelope exceeds eight MiB")
    if base64.b64encode(raw).decode("ascii") != text:
        _raise("invalid_value", path + ("encoded_envelope",), "encoded envelope is not canonical base64")
    if hashlib.sha256(raw).hexdigest() != digest:
        _raise("digest_mismatch", path + ("envelope_sha256",), "encoded envelope digest changed")
    envelope = _strict_json_object(raw, path + ("encoded_envelope",))
    if _canonical_envelope_bytes(envelope, path + ("encoded_envelope",)) != raw:
        _raise("invalid_value", path + ("encoded_envelope",), "encoded envelope is not canonical JSON")
    return envelope


def _envelope_graph_id(envelope: Mapping[str, Any], path: tuple[str | int, ...]) -> str:
    invocation = envelope.get("invocation")
    if type(invocation) is not dict:
        _raise("invalid_payload", path + ("encoded_envelope", "invocation"), "envelope invocation is invalid")
    return _exact_text(
        invocation.get("graph_invocation_id"),
        path + ("encoded_envelope", "invocation", "graph_invocation_id"),
        "graph invocation id",
    )


def _event_identity(event: JournalEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "store_seq": event.store_seq,
        "operation_id": event.operation.operation_id,
        "payload_sha256": event.operation.payload_sha256,
    }


def _validate_previous_event(value: Any, path: tuple[str | int, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    record = _closed_mapping(
        value,
        path,
        {"event_id", "store_seq", "operation_id", "payload_sha256"},
    )
    event_id = _exact_text(record["event_id"], path + ("event_id",), "event id")
    operation_id = _exact_text(record["operation_id"], path + ("operation_id",), "operation id")
    if type(record["payload_sha256"]) is not str or _SHA256.fullmatch(record["payload_sha256"]) is None:
        _raise("invalid_value", path + ("payload_sha256",), "payload digest is invalid")
    return {
        "event_id": event_id,
        "store_seq": _exact_positive_int(record["store_seq"], path + ("store_seq",), "store sequence"),
        "operation_id": operation_id,
        "payload_sha256": record["payload_sha256"],
    }


def _payload_for(
    *,
    graph_key: str,
    revision: int,
    previous_event: dict[str, Any] | None,
    envelope: dict[str, Any],
) -> dict[str, Any]:
    encoded, digest = _encode_envelope(envelope)
    return {
        "schema": _PAYLOAD_SCHEMA,
        "graph_key": graph_key,
        "revision": revision,
        "previous_event": previous_event,
        "codec": _CODEC,
        "encoded_envelope": encoded,
        "envelope_sha256": digest,
    }


def _decode_recovery_event(
    event: JournalEvent,
    snapshot_index: int,
    path: tuple[str | int, ...],
) -> _RecoveryEvent:
    if event.event_type != _EVENT_TYPE:
        _raise("invalid_value", path + ("event_type",), "event type is not a recovery descriptor")
    payload = _closed_mapping(
        event.payload,
        path + ("payload",),
        {
            "schema",
            "graph_key",
            "revision",
            "previous_event",
            "codec",
            "encoded_envelope",
            "envelope_sha256",
        },
    )
    if payload["schema"] != _PAYLOAD_SCHEMA:
        _raise("unknown_variant", path + ("payload", "schema"), "recovery payload schema is unsupported")
    graph_key = payload["graph_key"]
    if type(graph_key) is not str or _SHA256.fullmatch(graph_key) is None:
        _raise("invalid_value", path + ("payload", "graph_key"), "graph key is invalid")
    if payload["codec"] != _CODEC:
        _raise("unknown_variant", path + ("payload", "codec"), "recovery payload codec is unsupported")
    envelope = _decode_envelope(payload["encoded_envelope"], payload["envelope_sha256"], path + ("payload",))
    graph_id = _envelope_graph_id(envelope, path + ("payload",))
    if _graph_key(graph_id) != graph_key:
        _raise("identity_mismatch", path + ("payload", "graph_key"), "graph key does not match encoded envelope")
    return _RecoveryEvent(
        event=event,
        snapshot_index=snapshot_index,
        revision=_exact_positive_int(payload["revision"], path + ("payload", "revision"), "revision"),
        graph_key=graph_key,
        previous_event=_validate_previous_event(payload["previous_event"], path + ("payload", "previous_event")),
        envelope=envelope,
    )


def _native_operation_id(graph_key: str, caller_operation_id: str) -> str:
    caller_key = hashlib.sha256(caller_operation_id.encode("utf-8", "surrogatepass")).hexdigest()
    return f"skeleton.workflow-recovery.op.{graph_key}.{caller_key}"


def _native_event_id(native_operation_id: str) -> str:
    return "skeleton.workflow-recovery.event." + hashlib.sha256(
        native_operation_id.encode("ascii")
    ).hexdigest()


def _validate_caller_operation_id(operation_id: Any) -> str:
    if type(operation_id) is not str:
        _raise("invalid_type", ("operation_id",), "operation id must be text")
    try:
        # This is only a validation probe: the durable operation id is the
        # graph-scoped deterministic value below.
        probe = OperationRef(operation_id, "0" * 64)
    except (TypeError, ValueError) as exc:
        raise WorkflowRecoveryJournalError(
            "invalid_value", ("operation_id",), "operation id is not a journal-native identifier"
        ) from exc
    if probe.operation_id != operation_id:
        _raise("invalid_value", ("operation_id",), "operation id must already be a journal-native identifier")
    return operation_id


def _validate_scope(
    journal: Any,
    attempt: Any,
    envelope: dict[str, Any],
) -> tuple[BoundExecutionJournal, AttemptRef, str]:
    if not isinstance(journal, BoundExecutionJournal):
        _raise("invalid_type", ("journal",), "journal must be a BoundExecutionJournal")
    if not isinstance(attempt, AttemptRef):
        _raise("invalid_type", ("attempt",), "attempt must be an AttemptRef")
    invocation = envelope["invocation"]
    execution_id = invocation["execution_id"]
    attempt_id = invocation["attempt_id"]
    if attempt.generation.execution_id != execution_id or journal.execution_id != execution_id:
        _raise("identity_mismatch", ("envelope", "invocation", "execution_id"), "attempt or journal escaped envelope execution")
    if attempt.attempt_id != attempt_id:
        _raise("identity_mismatch", ("envelope", "invocation", "attempt_id"), "attempt does not match envelope")
    return journal, attempt, invocation["graph_invocation_id"]


def _validate_snapshot(snapshot: Any, journal: BoundExecutionJournal, path: tuple[str | int, ...]) -> JournalSnapshot:
    if not isinstance(snapshot, JournalSnapshot):
        _raise("invalid_snapshot", path, "journal snapshot is invalid")
    if snapshot.execution_id != journal.execution_id:
        _raise("identity_mismatch", path + ("execution_id",), "journal snapshot escaped bound execution")
    return snapshot


def _validate_recovery_event_identity(
    recovery: _RecoveryEvent,
    path: tuple[str | int, ...],
) -> None:
    """Validate identities that are independent of the caller's Workflow plan."""
    invocation = recovery.envelope["invocation"]
    event = recovery.event
    execution_id = _exact_text(
        invocation.get("execution_id"),
        path + ("payload", "encoded_envelope", "invocation", "execution_id"),
        "execution id",
    )
    attempt_id = _exact_text(
        invocation.get("attempt_id"),
        path + ("payload", "encoded_envelope", "invocation", "attempt_id"),
        "attempt id",
    )
    if event.attempt.generation.execution_id != execution_id:
        _raise(
            "identity_mismatch",
            path + ("attempt", "generation", "execution_id"),
            "event execution changed",
        )
    if event.attempt.attempt_id != attempt_id:
        _raise(
            "identity_mismatch",
            path + ("attempt", "attempt_id"),
            "event attempt changed",
        )
    expected_operation = build_operation_ref(
        event.operation.operation_id,
        domain=_OPERATION_DOMAIN,
        payload=dict(event.payload),
    )
    if event.operation != expected_operation:
        _raise(
            "digest_mismatch",
            path + ("operation",),
            "operation content digest changed",
        )
    if re.fullmatch(
        rf"skeleton\.workflow-recovery\.op\.{recovery.graph_key}\.[0-9a-f]{{64}}",
        event.operation.operation_id,
    ) is None:
        _raise(
            "identity_mismatch",
            path + ("operation", "operation_id"),
            "operation is not scoped to graph",
        )
    if event.event_id != _native_event_id(event.operation.operation_id):
        _raise(
            "identity_mismatch",
            path + ("event_id",),
            "event identity changed",
        )


def _validate_chain(
    snapshot: JournalSnapshot,
    *,
    plan: PreparedWorkflowPlan,
    graph_invocation_id: str,
    expected_attempt: AttemptRef | None,
) -> tuple[_RecoveryEvent, ...]:
    graph_key = _graph_key(graph_invocation_id)
    all_recovery: list[_RecoveryEvent] = []
    for snapshot_index, event in enumerate(snapshot.events):
        if event.event_type == _EVENT_TYPE:
            event_path = ("snapshot", "events", snapshot_index)
            recovery = _decode_recovery_event(
                event,
                snapshot_index,
                event_path,
            )
            # These checks depend only on the persisted event itself.  Run
            # them before graph filtering so a contradictory graph identity
            # cannot hide corruption or make the reader fall back.
            _validate_recovery_event_identity(recovery, event_path)
            all_recovery.append(recovery)
    matching = tuple(item for item in all_recovery if item.graph_key == graph_key)
    if not matching:
        return ()

    first = matching[0]
    baseline_definition = first.envelope.get("definition")
    baseline_invocation = first.envelope.get("invocation")
    baseline_call_stack = first.envelope.get("call_stack")
    previous: _RecoveryEvent | None = None
    for index, current in enumerate(matching, start=1):
        event_path = ("snapshot", "events", current.snapshot_index)
        try:
            validated = validate_workflow_recovery_envelope(current.envelope, plan)
        except SkeletonValidationError as exc:
            raise WorkflowRecoveryJournalError(
                exc.code,
                event_path + ("payload", "encoded_envelope") + exc.path,
                str(exc),
            ) from exc
        if validated["invocation"]["graph_invocation_id"] != graph_invocation_id:
            _raise("identity_mismatch", event_path + ("payload", "graph_key"), "recovery graph identity collided")
        if expected_attempt is not None and current.event.attempt != expected_attempt:
            _raise("identity_mismatch", event_path + ("attempt",), "persisted attempt or generation changed")
        if current.event.attempt.generation != first.event.attempt.generation:
            _raise("identity_mismatch", event_path + ("attempt", "generation"), "generation changed across revisions")
        if validated["definition"] != baseline_definition:
            _raise("identity_mismatch", event_path + ("payload", "definition"), "definition changed across revisions")
        if validated["invocation"] != baseline_invocation:
            _raise("identity_mismatch", event_path + ("payload", "invocation"), "invocation changed across revisions")
        if validated["call_stack"] != baseline_call_stack:
            _raise("identity_mismatch", event_path + ("payload", "call_stack"), "call stack changed across revisions")
        if current.revision != index:
            _raise("revision_mismatch", event_path + ("payload", "revision"), "recovery revisions must be contiguous")
        if current.previous_event != (_event_identity(previous.event) if previous else None):
            _raise("identity_mismatch", event_path + ("payload", "previous_event"), "previous event link changed")
        if previous is not None:
            _validate_transition(previous.envelope, validated, event_path)
        previous = current
    return matching


def _validate_transition(previous: dict[str, Any], current: dict[str, Any], path: tuple[str | int, ...]) -> None:
    if previous == current:
        _raise("invalid_transition", path + ("payload",), "recovery revision cannot be a no-op")
    if previous["terminal"] is not None:
        _raise("invalid_transition", path + ("payload", "terminal"), "terminal recovery cannot advance")
    previous_active = previous["active"]
    if previous_active is None:
        _raise("invalid_transition", path + ("payload", "active"), "recovery state has no active node")
    if previous_active["kind"] == "outcome_unknown":
        _raise("invalid_transition", path + ("payload", "active"), "unknown outcome cannot advance")
    for field in ("definition", "invocation", "call_stack"):
        if current[field] != previous[field]:
            _raise("identity_mismatch", path + ("payload", field), f"{field} changed across revisions")
    previous_completed = previous["completed_nodes"]
    current_completed = current["completed_nodes"]
    previous_selected = previous["selected_edges"]
    current_selected = current["selected_edges"]
    if current_completed[: len(previous_completed)] != previous_completed:
        _raise("invalid_transition", path + ("payload", "completed_nodes"), "completed path was rewritten")
    if current_selected[: len(previous_selected)] != previous_selected:
        _raise("invalid_transition", path + ("payload", "selected_edges"), "selected edge path was rewritten")
    current_active = current["active"]
    if current_active is not None and current_active["node_id"] == previous_active["node_id"]:
        for field in ("node_id", "node_invocation_id", "attempt_id"):
            if current_active[field] != previous_active[field]:
                _raise("identity_mismatch", path + ("payload", "active", field), "active node identity changed")
        if previous_active["kind"] == "waiting_interaction":
            if current_active["kind"] == "waiting_interaction":
                for field in ("interaction_id", "request_receipt_id"):
                    if current_active[field] != previous_active[field]:
                        _raise("identity_mismatch", path + ("payload", "active", field), "waiting interaction changed")
            elif current_active["kind"] != "outcome_unknown":
                _raise("invalid_transition", path + ("payload", "active", "kind"), "waiting interaction has invalid transition")
        return
    if len(current_completed) <= len(previous_completed):
        _raise("invalid_transition", path + ("payload", "completed_nodes"), "active node did not complete")
    completed = current_completed[len(previous_completed)]
    for field in ("node_id", "node_invocation_id", "attempt_id"):
        if completed[field] != previous_active[field]:
            _raise("identity_mismatch", path + ("payload", "completed_nodes", len(previous_completed), field), "completed node replaced active identity")


def _validate_projected_snapshot(snapshot: JournalSnapshot, request: JournalAppendRequest) -> None:
    candidate = JournalEvent(
        event_id=request.event_id,
        event_type=request.event_type,
        attempt=request.attempt,
        operation=request.operation,
        store_seq=len(snapshot.events) + 1,
        payload=request.payload,
    )
    # Reuse the journal snapshot verifier so event count, contiguous sequence,
    # and resource limits match the persisted journal's snapshot contract.
    capture_journal_snapshot(
        execution_id=snapshot.execution_id,
        events=(*snapshot.events, candidate),
    )
    try:
        encoded = json.dumps(
            [event.to_dict() for event in (*snapshot.events, candidate)],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise JournalRepositoryError("projected journal snapshot is not canonical JSON") from exc
    if len(encoded) > 32 * 1024 * 1024:
        raise JournalRepositoryError("journal snapshot byte limit exceeded")


def save_workflow_recovery(
    journal: BoundExecutionJournal,
    plan: PreparedWorkflowPlan,
    envelope: dict[str, Any],
    *,
    attempt: AttemptRef,
    operation_id: str,
    expected_revision: int,
) -> JournalAppendResult:
    """Atomically append one checked Workflow recovery descriptor.

    The journal's replay check happens before the precondition, so retrying an
    older successful operation continues to return its original event even
    after a later recovery revision has been saved.
    """
    try:
        checked_envelope = validate_workflow_recovery_envelope(envelope, plan)
    except SkeletonValidationError as exc:
        raise WorkflowRecoveryJournalError(exc.code, exc.path, str(exc)) from exc
    journal, attempt, graph_invocation_id = _validate_scope(journal, attempt, checked_envelope)
    caller_operation_id = _validate_caller_operation_id(operation_id)
    revision = _exact_nonnegative_int(expected_revision, ("expected_revision",), "expected revision")
    graph_key = _graph_key(graph_invocation_id)

    # This pre-read is only for reconstructing the exact retry request.  The
    # atomic callback below independently reads, validates, and authorizes the
    # actual write from the transaction's current snapshot.
    initial_snapshot = _validate_snapshot(journal.capture_snapshot(), journal, ("snapshot",))
    initial_chain = _validate_chain(
        initial_snapshot,
        plan=plan,
        graph_invocation_id=graph_invocation_id,
        expected_attempt=attempt,
    )
    if revision > len(initial_chain):
        _raise("revision_mismatch", ("expected_revision",), "expected revision is ahead of durable recovery")
    previous_event = _event_identity(initial_chain[revision - 1].event) if revision else None
    payload = _payload_for(
        graph_key=graph_key,
        revision=revision + 1,
        previous_event=previous_event,
        envelope=checked_envelope,
    )
    native_operation_id = _native_operation_id(graph_key, caller_operation_id)
    request = JournalAppendRequest(
        event_id=_native_event_id(native_operation_id),
        event_type=_EVENT_TYPE,
        attempt=attempt,
        operation=build_operation_ref(
            native_operation_id,
            domain=_OPERATION_DOMAIN,
            payload=payload,
        ),
        payload=payload,
    )

    def precondition(snapshot: JournalSnapshot) -> None:
        current = _validate_snapshot(snapshot, journal, ("transaction", "snapshot"))
        chain = _validate_chain(
            current,
            plan=plan,
            graph_invocation_id=graph_invocation_id,
            expected_attempt=attempt,
        )
        if len(chain) != revision:
            raise JournalConflictError("workflow recovery revision changed")
        if chain and (chain[-1].envelope["terminal"] is not None or chain[-1].envelope["active"]["kind"] == "outcome_unknown"):
            _raise("invalid_transition", ("transaction", "snapshot"), "terminal or unknown recovery cannot advance")
        if chain:
            _validate_transition(chain[-1].envelope, checked_envelope, ("transaction", "snapshot"))
        _validate_projected_snapshot(current, request)

    return journal.append_with_artifacts(request=request, artifacts=(), precondition=precondition)


def load_workflow_recovery(
    journal: BoundExecutionJournal,
    plan: PreparedWorkflowPlan,
    *,
    graph_invocation_id: str,
) -> WorkflowRecoveryRecord | None:
    """Load the latest fully validated durable descriptor for one graph."""
    if not isinstance(journal, BoundExecutionJournal):
        _raise("invalid_type", ("journal",), "journal must be a BoundExecutionJournal")
    graph_id = _exact_text(graph_invocation_id, ("graph_invocation_id",), "graph invocation id")
    snapshot = _validate_snapshot(journal.capture_snapshot(), journal, ("snapshot",))
    chain = _validate_chain(
        snapshot,
        plan=plan,
        graph_invocation_id=graph_id,
        expected_attempt=None,
    )
    if not chain:
        return None
    latest = chain[-1]
    return WorkflowRecoveryRecord(
        revision=latest.revision,
        envelope=_clone(latest.envelope),
        event_id=latest.event.event_id,
        store_seq=latest.event.store_seq,
        operation_id=latest.event.operation.operation_id,
    )


__all__ = [
    "WorkflowRecoveryJournalError",
    "WorkflowRecoveryRecord",
    "load_workflow_recovery",
    "save_workflow_recovery",
]
