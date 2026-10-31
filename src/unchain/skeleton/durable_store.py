"""Atomic, provider-free persistence for CP-10 Workflow execution state.

The store deliberately owns no database connection.  It consumes the
execution-bound journal's atomic ``append_with_artifacts`` capability and
checks every durable receipt before exposing it to a runner.
"""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundExecutionJournal,
    JournalAppendRequest,
    JournalConflictError,
    JournalEvent,
    JournalRepositoryError,
    JournalSnapshot,
    PendingArtifact,
    ResourceRef,
    build_operation_ref,
    capture_journal_snapshot,
)

from .durable_protocol import (
    canonical_bytes,
    decode_json,
    execution_artifacts,
    artifact_ref,
    validate_execution_state,
    validate_execution_transition,
)
from .errors import SkeletonValidationError
from .recovery_journal import (
    WorkflowRecoveryJournalError,
    _decode_recovery_event,
    _validate_recovery_event_identity,
)


_EVENT_TYPE = "skeleton.workflow_execution.saved"
_PAYLOAD_SCHEMA = "unchain.skeleton_workflow_execution_journal.v2"
_OPERATION_DOMAIN = "skeleton.workflow_execution.saved"
_ARTIFACT_OPERATION_DOMAIN = "context.artifact.put"
_MAX_STATE_BYTES = 8 * 1024 * 1024
_MAX_ENCODED_BYTES = 4 * ((_MAX_STATE_BYTES + 2) // 3)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DurableStoreError(SkeletonValidationError):
    """A durable execution receipt, artifact, or write is inconsistent."""


@dataclass(frozen=True)
class ExecutionRecord:
    """The latest validated execution state and its journal position."""

    revision: int
    state: dict[str, Any]
    event_id: str
    store_seq: int


@dataclass(frozen=True)
class _ExecutionEvent:
    event: JournalEvent
    snapshot_index: int
    revision: int
    graph_key: str
    previous_event_id: str | None
    state: dict[str, Any]


def _raise(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise DurableStoreError(code, path, message)


def _exact_text(value: Any, path: tuple[str | int, ...], label: str) -> str:
    if type(value) is not str:
        _raise("invalid_type", path, f"{label} must be text")
    if not value:
        _raise("invalid_value", path, f"{label} must be non-empty")
    return value


def _positive_int(value: Any, path: tuple[str | int, ...], label: str) -> int:
    if type(value) is not int:
        _raise("invalid_type", path, f"{label} must be an integer")
    if value < 1:
        _raise("invalid_value", path, f"{label} must be positive")
    return value


def _nonnegative_int(value: Any, path: tuple[str | int, ...], label: str) -> int:
    if type(value) is not int:
        _raise("invalid_type", path, f"{label} must be an integer")
    if value < 0:
        _raise("invalid_value", path, f"{label} must be non-negative")
    return value


def _closed_mapping(value: Any, path: tuple[str | int, ...], required: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise("invalid_type", path, "value must be an object")
    keys = set(value)
    if any(type(key) is not str for key in keys):
        _raise("invalid_json", path, "object keys must be strings")
    missing = required - keys
    unknown = keys - required
    if missing:
        _raise("missing_field", path + (sorted(missing)[0],), "required field is missing")
    if unknown:
        _raise("unknown_field", path + (sorted(unknown)[0],), "unknown field")
    return value


def _graph_key(graph_id: str) -> str:
    return hashlib.sha256(graph_id.encode("utf-8", "surrogatepass")).hexdigest()


def _strict_b64decode(value: Any, path: tuple[str | int, ...]) -> bytes:
    text = _exact_text(value, path, "encoded state")
    if len(text) > _MAX_ENCODED_BYTES:
        _raise("limit_bytes", path, "encoded state exceeds limit")
    try:
        raw = base64.b64decode(text.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise DurableStoreError("invalid_value", path, "encoded state is not strict base64") from exc
    if len(raw) > _MAX_STATE_BYTES:
        _raise("limit_bytes", path, "decoded state exceeds limit")
    if base64.b64encode(raw).decode("ascii") != text:
        _raise("invalid_value", path, "encoded state is not canonical base64")
    return raw


def _encode_state(state: dict[str, Any]) -> tuple[str, str, bytes]:
    raw = canonical_bytes(state)
    if len(raw) > _MAX_STATE_BYTES:
        _raise("limit_bytes", ("state",), "state exceeds limit")
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest(), raw


def _decode_state(encoded: Any, digest: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    if type(digest) is not str or _SHA256.fullmatch(digest) is None:
        _raise("invalid_value", path + ("state_sha256",), "state digest is invalid")
    raw = _strict_b64decode(encoded, path + ("encoded_state",))
    if hashlib.sha256(raw).hexdigest() != digest:
        _raise("digest_mismatch", path + ("state_sha256",), "state digest changed")
    try:
        decoded = decode_json(raw)
    except SkeletonValidationError as exc:
        raise DurableStoreError(exc.code, path + ("encoded_state",) + exc.path, str(exc)) from exc
    except (TypeError, ValueError, UnicodeError) as exc:
        raise DurableStoreError("invalid_json", path + ("encoded_state",), "state is not valid JSON") from exc
    if type(decoded) is not dict:
        _raise("invalid_type", path + ("encoded_state",), "state must be an object")
    try:
        if canonical_bytes(decoded) != raw:
            _raise("invalid_value", path + ("encoded_state",), "state is not canonical JSON")
    except SkeletonValidationError as exc:
        raise DurableStoreError(exc.code, path + ("encoded_state",) + exc.path, str(exc)) from exc
    return decoded


def _validate_artifact_descriptor(value: Any, path: tuple[str | int, ...]) -> ArtifactRef:
    try:
        artifact = artifact_ref(value, path)
    except SkeletonValidationError as exc:
        raise DurableStoreError(exc.code, path + exc.path, str(exc)) from exc
    if artifact.to_dict() != value:
        _raise("invalid_value", path, "artifact descriptor is not full canonical shape")
    return artifact


class WorkflowExecutionStore:
    """Persist CP-10 state in one execution-bound journal capability."""

    def __init__(self, journal: Any, plan: Any, *, graph_id: str, attempt: AttemptRef) -> None:
        if not isinstance(journal, BoundExecutionJournal):
            _raise("invalid_type", ("journal",), "journal must be a BoundExecutionJournal")
        for name in ("artifact_id_for", "read_full_verified"):
            if not callable(getattr(journal, name, None)):
                _raise("unsupported_capability", ("journal", name), f"journal lacks {name}")
        if not isinstance(attempt, AttemptRef):
            _raise("invalid_type", ("attempt",), "attempt must be an AttemptRef")
        graph_id = _exact_text(graph_id, ("graph_id",), "graph id")
        if journal.execution_id != attempt.generation.execution_id:
            _raise("identity_mismatch", ("attempt", "generation", "execution_id"), "attempt escaped journal execution")
        self._journal = journal
        self._plan = plan
        self._graph_id = graph_id
        self._graph_key = _graph_key(graph_id)
        self._attempt = attempt

    @property
    def plan(self) -> Any:
        return self._plan

    @property
    def journal(self) -> BoundExecutionJournal:
        return self._journal

    @property
    def graph_id(self) -> str:
        return self._graph_id

    @property
    def attempt(self) -> AttemptRef:
        return self._attempt

    def _artifact_operation_id(self, key: Any) -> str:
        key = _exact_text(key, ("key",), "artifact key")
        digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()
        return f"skeleton.execution.artifact.{self._graph_key}.{digest}"

    def _artifact_for_pending(self, pending: PendingArtifact) -> dict[str, Any]:
        artifact_id = self._journal.artifact_id_for(
            logical_kind="artifact", logical_key=pending.operation.operation_id
        )
        artifact = ArtifactRef(
            ref=ResourceRef("artifact", artifact_id, 1),
            media_type=pending.media_type,
            byte_length=len(pending.content),
            sha256=hashlib.sha256(pending.content).hexdigest(),
            preview="",
        )
        return artifact.to_dict()

    def prepare_bytes(
        self, content: bytes, *, media_type: str, key: str
    ) -> tuple[dict[str, Any], PendingArtifact]:
        if type(content) is not bytes:
            _raise("invalid_type", ("content",), "content must be exact bytes")
        media_type = _exact_text(media_type, ("media_type",), "media type")
        if "/" not in media_type:
            _raise("invalid_value", ("media_type",), "media type must be a MIME type")
        operation_id = self._artifact_operation_id(key)
        payload = {
            "media_type": media_type,
            "byte_length": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "preview": "",
            "binding": {},
        }
        pending = PendingArtifact(
            content=content,
            media_type=media_type,
            operation=build_operation_ref(
                operation_id, domain=_ARTIFACT_OPERATION_DOMAIN, payload=payload
            ),
            preview="",
        )
        return self._artifact_for_pending(pending), pending

    def prepare_json(self, value: Any, *, key: str) -> tuple[dict[str, Any], PendingArtifact]:
        try:
            content = canonical_bytes(value)
        except SkeletonValidationError as exc:
            raise DurableStoreError(exc.code, ("value",) + exc.path, str(exc)) from exc
        return self.prepare_bytes(content, media_type="application/json", key=key)

    def read_bytes(self, ref: Any) -> bytes:
        artifact = _validate_artifact_descriptor(ref, ("ref",))
        try:
            content = self._journal.read_full_verified(artifact=artifact)
        except DurableStoreError:
            raise
        except Exception as exc:
            raise DurableStoreError("artifact_unavailable", ("ref",), "artifact cannot be read and verified") from exc
        if type(content) is not bytes:
            _raise("invalid_artifact", ("ref",), "artifact reader did not return bytes")
        if len(content) != artifact.byte_length or hashlib.sha256(content).hexdigest() != artifact.sha256:
            _raise("digest_mismatch", ("ref",), "artifact bytes disagree with descriptor")
        return content

    def read_json(self, ref: Any) -> Any:
        artifact = _validate_artifact_descriptor(ref, ("ref",))
        if artifact.media_type != "application/json":
            _raise("invalid_value", ("ref", "media_type"), "artifact is not JSON")
        content = self.read_bytes(ref)
        try:
            value = decode_json(content)
            if canonical_bytes(value) != content:
                _raise("invalid_value", ("ref",), "JSON artifact is not canonical")
            return value
        except DurableStoreError:
            raise
        except SkeletonValidationError as exc:
            raise DurableStoreError(exc.code, ("ref",) + exc.path, str(exc)) from exc
        except (TypeError, ValueError, UnicodeError) as exc:
            raise DurableStoreError("invalid_json", ("ref",), "artifact is not valid JSON") from exc

    def _validate_state(self, state: Any, *, plan: Any) -> dict[str, Any]:
        try:
            checked = validate_execution_state(state, plan=plan)
        except SkeletonValidationError as exc:
            raise DurableStoreError(exc.code, ("state",) + exc.path, str(exc)) from exc
        if type(checked) is not dict:
            _raise("invalid_state", ("state",), "state validator did not return an object")
        if checked.get("graph_id") != self._graph_id:
            _raise("identity_mismatch", ("state", "graph_id"), "state escaped requested graph")
        try:
            state_attempt = AttemptRef.from_dict(checked.get("attempt"))
        except (TypeError, ValueError) as exc:
            raise DurableStoreError("invalid_state", ("state", "attempt"), "state attempt is invalid") from exc
        if state_attempt != self._attempt:
            _raise("identity_mismatch", ("state", "attempt"), "state attempt changed")
        return deepcopy(checked)

    def _decode_event(self, event: JournalEvent, index: int) -> _ExecutionEvent:
        path = ("snapshot", "events", index)
        if event.event_type != _EVENT_TYPE:
            _raise("invalid_value", path + ("event_type",), "event is not a workflow execution receipt")
        payload = _closed_mapping(
            event.payload,
            path + ("payload",),
            {"schema", "graph_key", "revision", "previous_event_id", "encoded_state", "state_sha256"},
        )
        if payload["schema"] != _PAYLOAD_SCHEMA:
            _raise("unknown_variant", path + ("payload", "schema"), "execution payload schema is unsupported")
        graph_key = payload["graph_key"]
        if type(graph_key) is not str or _SHA256.fullmatch(graph_key) is None:
            _raise("invalid_value", path + ("payload", "graph_key"), "graph key is invalid")
        state = _decode_state(payload["encoded_state"], payload["state_sha256"], path + ("payload",))
        # This validation intentionally has no plan.  It is required before
        # graph filtering so malformed foreign receipts cannot be hidden.
        try:
            checked = validate_execution_state(state, plan=None)
        except SkeletonValidationError as exc:
            raise DurableStoreError(exc.code, path + ("payload", "encoded_state") + exc.path, str(exc)) from exc
        if type(checked) is not dict or checked != state:
            _raise("invalid_state", path + ("payload", "encoded_state"), "self-contained state changed during validation")
        graph_id = checked.get("graph_id")
        if type(graph_id) is not str or _graph_key(graph_id) != graph_key:
            _raise("identity_mismatch", path + ("payload", "graph_key"), "graph key disagrees with state")
        try:
            state_attempt = AttemptRef.from_dict(checked.get("attempt"))
        except (TypeError, ValueError) as exc:
            raise DurableStoreError("invalid_state", path + ("payload", "encoded_state", "attempt"), "state attempt is invalid") from exc
        if event.attempt != state_attempt:
            _raise("identity_mismatch", path + ("attempt",), "event attempt disagrees with state")
        expected_operation = build_operation_ref(
            event.operation.operation_id, domain=_OPERATION_DOMAIN, payload=dict(event.payload)
        )
        if event.operation != expected_operation:
            _raise("digest_mismatch", path + ("operation",), "operation digest changed")
        if re.fullmatch(rf"skeleton\.execution\.{graph_key}\.[0-9a-f]{{64}}", event.operation.operation_id) is None:
            _raise("identity_mismatch", path + ("operation", "operation_id"), "operation is not graph scoped")
        if event.event_id != f"event.{event.operation.operation_id}":
            _raise("identity_mismatch", path + ("event_id",), "event identity changed")
        expected_refs = tuple(ref.ref for ref in execution_artifacts(checked))
        if event.resource_refs != expected_refs:
            _raise("identity_mismatch", path + ("resource_refs",), "event resources disagree with encoded state")
        previous_event_id = payload["previous_event_id"]
        if previous_event_id is not None:
            previous_event_id = _exact_text(previous_event_id, path + ("payload", "previous_event_id"), "previous event id")
        return _ExecutionEvent(
            event=event,
            snapshot_index=index,
            revision=_positive_int(payload["revision"], path + ("payload", "revision"), "revision"),
            graph_key=graph_key,
            previous_event_id=previous_event_id,
            state=deepcopy(checked),
        )

    def _snapshot(
        self, snapshot: Any, *, plan: Any, verify_contents: bool = False
    ) -> tuple[_ExecutionEvent, ...]:
        if not isinstance(snapshot, JournalSnapshot):
            _raise("invalid_snapshot", ("snapshot",), "journal returned an invalid snapshot")
        if snapshot.execution_id != self._journal.execution_id:
            _raise("identity_mismatch", ("snapshot", "execution_id"), "snapshot escaped journal execution")
        all_events: list[_ExecutionEvent] = []
        saw_v1_for_graph = False
        for index, event in enumerate(snapshot.events):
            if event.event_type == _EVENT_TYPE:
                all_events.append(self._decode_event(event, index))
            elif event.event_type == "skeleton.workflow_recovery.saved":
                # Validate v1 identity before filtering too.  A forged v1
                # graph key must not hide earlier incompatible progress and
                # make a new v2 graph look fresh.
                path = ("snapshot", "events", index)
                try:
                    legacy = _decode_recovery_event(event, index, path)
                    _validate_recovery_event_identity(legacy, path)
                except WorkflowRecoveryJournalError as exc:
                    raise DurableStoreError(exc.code, exc.path, str(exc)) from exc
                if legacy.graph_key == self._graph_key:
                    saw_v1_for_graph = True
        matching = tuple(item for item in all_events if item.graph_key == self._graph_key)
        if saw_v1_for_graph:
            _raise("unsupported_recovery", ("snapshot",), "v1 recovery cannot start a CP-10 execution")
        if not matching:
            return ()
        previous: _ExecutionEvent | None = None
        content_states: list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]] = []
        for revision, current in enumerate(matching, start=1):
            path = ("snapshot", "events", current.snapshot_index)
            checked = self._validate_state(current.state, plan=plan)
            if checked != current.state:
                _raise("invalid_state", path + ("payload", "encoded_state"), "plan validation changed state")
            descriptors = self._state_artifacts(checked)
            content_states.append((checked, descriptors))
            if current.event.attempt != self._attempt:
                _raise("identity_mismatch", path + ("attempt",), "persisted attempt changed")
            if current.revision != revision:
                _raise("revision_mismatch", path + ("payload", "revision"), "revisions must be contiguous")
            if current.previous_event_id != (previous.event.event_id if previous else None):
                _raise("identity_mismatch", path + ("payload", "previous_event_id"), "predecessor link changed")
            if previous is not None:
                try:
                    validate_execution_transition(previous.state, current.state, plan)
                except SkeletonValidationError as exc:
                    raise DurableStoreError(exc.code, path + ("payload", "encoded_state") + exc.path, str(exc)) from exc
            else:
                try:
                    validate_execution_transition(None, current.state, plan)
                except SkeletonValidationError as exc:
                    raise DurableStoreError(exc.code, path + ("payload", "encoded_state") + exc.path, str(exc)) from exc
            previous = current
        if verify_contents:
            self._verify_snapshot_contents(content_states)
        return matching

    def _capture_chain(self, *, plan: Any) -> tuple[_ExecutionEvent, ...]:
        try:
            snapshot = self._journal.capture_snapshot()
        except Exception as exc:
            raise DurableStoreError("snapshot_failed", ("snapshot",), "journal snapshot failed") from exc
        return self._snapshot(snapshot, plan=plan, verify_contents=True)

    def _state_artifacts(self, state: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        try:
            refs = execution_artifacts(state)
        except SkeletonValidationError as exc:
            raise DurableStoreError(exc.code, ("state",) + exc.path, str(exc)) from exc
        if not isinstance(refs, Iterable) or isinstance(refs, (str, bytes, bytearray, dict)):
            _raise("invalid_state", ("state",), "execution artifacts must be an array")
        result: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for index, ref in enumerate(refs):
            raw = ref.to_dict() if isinstance(ref, ArtifactRef) else ref
            artifact = _validate_artifact_descriptor(raw, ("state", "artifacts", index))
            identity = (artifact.ref.resource_id, artifact.ref.revision)
            if identity in seen:
                _raise("duplicate_artifact", ("state", "artifacts", index), "artifact is referenced more than once")
            seen.add(identity)
            result.append(artifact.to_dict())
        self._validate_artifact_provenance(state)
        return tuple(result)

    def _validate_artifact_provenance(self, state: dict[str, Any]) -> None:
        """Bind every receipt slot to its one graph-qualified content key."""
        slots: list[tuple[Any, str, tuple[str | int, ...]]] = [
            (state["inputs_ref"], "root.inputs", ("state", "inputs_ref")),
        ]
        slots.extend(
            (ref, "attachment." + name, ("state", "attachments", name))
            for name, ref in state["attachments"].items()
        )
        slots.extend(
            (binding["source_ref"], "code." + node_id + ".source", ("state", "code_bindings", node_id, "source_ref"))
            for node_id, binding in state.get("code_bindings", {}).items()
        )
        for index, row in enumerate(state["completed"]):
            invocation = row["node_invocation_id"]
            slots.extend((row[field], invocation + suffix, ("state", "completed", index, field)) for field, suffix in (
                ("input_ref", ".input"), ("result_ref", ".raw"), ("output_ref", ".output"),
            ))
        active = state["active"]
        if active is not None:
            invocation = active["node_invocation_id"]
            slots.append((active["input_ref"], invocation + ".input", ("state", "active", "input_ref")))
            if active["result_ref"] is not None:
                slots.append((active["result_ref"], invocation + ".raw", ("state", "active", "result_ref")))
            if active["wait"] is not None:
                wait = active["wait"]
                slots.append((wait["request_ref"], invocation + ".wait." + wait["id"], ("state", "active", "wait", "request_ref")))
                if wait["receipt_ref"] is not None:
                    slots.append((wait["receipt_ref"], invocation + ".wait." + wait["id"] + ".receipt", ("state", "active", "wait", "receipt_ref")))
        for value, key, path in slots:
            artifact = _validate_artifact_descriptor(value, path)
            expected = self._journal.artifact_id_for(
                logical_kind="artifact", logical_key=self._artifact_operation_id(key)
            )
            if artifact.ref.resource_id != expected:
                _raise("identity_mismatch", path + ("ref", "id"), "artifact is not bound to its execution receipt slot")

    def _verify_snapshot_contents(
        self, states: list[tuple[dict[str, Any], tuple[dict[str, Any], ...]]]
    ) -> None:
        """Read each immutable object once, then validate JSON receipt values."""
        contents: dict[tuple[str, int], bytes] = {}
        descriptors: dict[tuple[str, int], dict[str, Any]] = {}
        receipts: list[tuple[dict[str, Any], tuple[str | int, ...]]] = []
        for state, refs in states:
            for descriptor in refs:
                identity = (descriptor["ref"]["id"], descriptor["ref"]["revision"])
                descriptors.setdefault(identity, descriptor)
            receipts.extend(self._receipt_values(state))
        for identity, descriptor in descriptors.items():
            contents[identity] = self.read_bytes(descriptor)
        for raw, path in receipts:
            artifact = _validate_artifact_descriptor(raw, path)
            if artifact.media_type != "application/json":
                _raise("invalid_value", path + ("media_type",), "execution receipt must be JSON")
            content = contents[(artifact.ref.resource_id, artifact.ref.revision)]
            try:
                value = decode_json(content)
            except SkeletonValidationError as exc:
                raise DurableStoreError(exc.code, path + exc.path, str(exc)) from exc
            if type(value) is not dict:
                _raise("invalid_type", path, "execution receipt must contain an object")

    @staticmethod
    def _receipt_values(state: dict[str, Any]) -> list[tuple[dict[str, Any], tuple[str | int, ...]]]:
        values = [(state["inputs_ref"], ("state", "inputs_ref"))]
        for index, row in enumerate(state["completed"]):
            values.extend((row[field], ("state", "completed", index, field)) for field in ("input_ref", "result_ref", "output_ref"))
        active = state["active"]
        if active is not None:
            values.append((active["input_ref"], ("state", "active", "input_ref")))
            if active["result_ref"] is not None:
                values.append((active["result_ref"], ("state", "active", "result_ref")))
            if active["wait"] is not None:
                values.append((active["wait"]["request_ref"], ("state", "active", "wait", "request_ref")))
        return values

    def _validate_artifacts_before_write(
        self, state: dict[str, Any], artifacts: tuple[PendingArtifact, ...]
    ) -> tuple[dict[str, Any], ...]:
        declared = self._state_artifacts(state)
        declared_by_identity = {
            (ref["ref"]["id"], ref["ref"]["revision"]): ref for ref in declared
        }
        pending_descriptors: list[dict[str, Any]] = []
        pending_identities: set[tuple[str, int]] = set()
        for index, pending in enumerate(artifacts):
            if not isinstance(pending, PendingArtifact):
                _raise("invalid_type", ("artifacts", index), "artifacts must be PendingArtifact records")
            if pending.preview != "":
                _raise("invalid_value", ("artifacts", index, "preview"), "execution artifacts require an empty preview")
            expected_payload = {
                "media_type": pending.media_type,
                "byte_length": len(pending.content),
                "sha256": hashlib.sha256(pending.content).hexdigest(),
                "preview": "",
                "binding": {},
            }
            expected_operation = build_operation_ref(
                pending.operation.operation_id,
                domain=_ARTIFACT_OPERATION_DOMAIN,
                payload=expected_payload,
            )
            if pending.operation != expected_operation:
                _raise("digest_mismatch", ("artifacts", index, "operation"), "artifact operation payload changed")
            descriptor = self._artifact_for_pending(pending)
            identity = (descriptor["ref"]["id"], descriptor["ref"]["revision"])
            if identity in pending_identities:
                _raise("duplicate_artifact", ("artifacts", index), "pending artifact is duplicated")
            pending_identities.add(identity)
            if declared_by_identity.get(identity) != descriptor:
                _raise("artifact_mismatch", ("artifacts", index), "pending artifact is not exactly declared by state")
            pending_descriptors.append(descriptor)
        for descriptor in declared:
            identity = (descriptor["ref"]["id"], descriptor["ref"]["revision"])
            try:
                self.read_bytes(descriptor)
            except DurableStoreError as exc:
                # A freshly prepared artifact has no durable row yet.  The
                # concrete repository reports that as a scope miss; any other
                # read failure (including a checksum failure) is corruption.
                cause = exc.__cause__
                if identity not in pending_identities or cause is None or cause.__class__.__name__ != "ContextScopeError":
                    raise
        pending_content = {
            (descriptor["ref"]["id"], descriptor["ref"]["revision"]): pending.content
            for descriptor, pending in zip(pending_descriptors, artifacts)
        }
        for raw, path in self._receipt_values(state):
            artifact = _validate_artifact_descriptor(raw, path)
            identity = (artifact.ref.resource_id, artifact.ref.revision)
            content = pending_content.get(identity)
            if content is None:
                content = self.read_bytes(artifact.to_dict())
            if artifact.media_type != "application/json":
                _raise("invalid_value", path + ("media_type",), "execution receipt must be JSON")
            try:
                value = decode_json(content)
            except SkeletonValidationError as exc:
                raise DurableStoreError(exc.code, path + exc.path, str(exc)) from exc
            if type(value) is not dict:
                _raise("invalid_type", path, "execution receipt must contain an object")
        return tuple(pending_descriptors)

    @staticmethod
    def _validate_projected_snapshot(snapshot: JournalSnapshot, request: JournalAppendRequest) -> None:
        candidate = JournalEvent(
            event_id=request.event_id,
            event_type=request.event_type,
            attempt=request.attempt,
            operation=request.operation,
            store_seq=len(snapshot.events) + 1,
            payload=request.payload,
            resource_refs=request.resource_refs,
        )
        try:
            projected = capture_journal_snapshot(
                execution_id=snapshot.execution_id, events=(*snapshot.events, candidate)
            )
            encoded = json.dumps(
                [event.to_dict() for event in projected.events],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8", "surrogatepass")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise DurableStoreError("limit_bytes", ("transaction", "snapshot"), "projected snapshot is invalid") from exc
        if len(encoded) > 32 * 1024 * 1024:
            _raise("limit_bytes", ("transaction", "snapshot"), "projected snapshot exceeds byte limit")

    def load(self) -> ExecutionRecord | None:
        chain = self._capture_chain(plan=self._plan)
        if not chain:
            return None
        latest = chain[-1]
        return ExecutionRecord(latest.revision, deepcopy(latest.state), latest.event.event_id, latest.event.store_seq)

    def save(
        self,
        state: Any,
        *,
        expected_revision: int,
        operation_id: str,
        artifacts: Iterable[PendingArtifact] = (),
    ) -> ExecutionRecord:
        if self._plan is None:
            _raise("missing_plan", ("plan",), "journal-only execution stores cannot write")
        checked_state = self._validate_state(state, plan=self._plan)
        revision = _nonnegative_int(expected_revision, ("expected_revision",), "expected revision")
        caller_operation = _exact_text(operation_id, ("operation_id",), "operation id")
        artifacts = tuple(artifacts)
        self._validate_artifacts_before_write(checked_state, artifacts)
        chain = self._capture_chain(plan=self._plan)
        if revision > len(chain):
            _raise("revision_mismatch", ("expected_revision",), "expected revision is ahead of durable state")
        previous_event_id = chain[revision - 1].event.event_id if revision else None
        encoded_state, state_sha256, _ = _encode_state(checked_state)
        payload = {
            "schema": _PAYLOAD_SCHEMA,
            "graph_key": self._graph_key,
            "revision": revision + 1,
            "previous_event_id": previous_event_id,
            "encoded_state": encoded_state,
            "state_sha256": state_sha256,
        }
        operation_key = hashlib.sha256(caller_operation.encode("utf-8", "surrogatepass")).hexdigest()
        native_operation = f"skeleton.execution.{self._graph_key}.{operation_key}"
        request = JournalAppendRequest(
            event_id=f"event.{native_operation}",
            event_type=_EVENT_TYPE,
            attempt=self._attempt,
            operation=build_operation_ref(native_operation, domain=_OPERATION_DOMAIN, payload=payload),
            payload=payload,
            resource_refs=tuple(
                ResourceRef(ref["ref"]["kind"], ref["ref"]["id"], ref["ref"]["revision"], ref["ref"]["fragment"])
                for ref in self._state_artifacts(checked_state)
            ),
        )

        def precondition(snapshot: JournalSnapshot) -> None:
            current = self._snapshot(snapshot, plan=self._plan, verify_contents=False)
            if len(current) != revision:
                raise JournalConflictError("workflow execution revision changed")
            if current:
                try:
                    validate_execution_transition(current[-1].state, checked_state, self._plan)
                except SkeletonValidationError as exc:
                    raise DurableStoreError(exc.code, ("transaction", "state") + exc.path, str(exc)) from exc
            else:
                try:
                    validate_execution_transition(None, checked_state, self._plan)
                except SkeletonValidationError as exc:
                    raise DurableStoreError(exc.code, ("transaction", "state") + exc.path, str(exc)) from exc
            self._validate_projected_snapshot(snapshot, request)

        try:
            result = self._journal.append_with_artifacts(
                request=request, artifacts=artifacts, precondition=precondition
            )
        except DurableStoreError:
            raise
        except JournalConflictError:
            raise
        except JournalRepositoryError as exc:
            raise DurableStoreError("write_failed", ("journal",), "atomic execution save failed") from exc
        record = self._decode_event(result.event, result.event.store_seq - 1)
        # append_with_artifacts returns the original event on an exact replay.
        # Parsing it, rather than trusting the proposed state, prevents an old
        # receipt from granting a new dispatch after later revisions exist.
        if record.graph_key != self._graph_key or record.event.attempt != self._attempt:
            _raise("identity_mismatch", ("result",), "journal replay escaped execution graph")
        return ExecutionRecord(record.revision, deepcopy(record.state), record.event.event_id, record.event.store_seq)


__all__ = ["DurableStoreError", "ExecutionRecord", "WorkflowExecutionStore"]
