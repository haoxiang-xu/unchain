"""Immutable model-visible event selection over a complete shared journal."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

from unchain.journal import AttemptRef, JournalAppendRequest, JournalSnapshot, build_operation_ref
from unchain.journal.snapshot import journal_event_sha256

_SCHEMA = "unchain.node_memory_view.v1"
_SCHEMA_V2 = "unchain.node_memory_view.v2"
_EVENT = "context.node_memory_view"
_HISTORY_TYPES = frozenset({"message.user", "message.assistant", "final_message", "run_completed"})


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class NodeMemoryView:
    attempt: AttemptRef
    definition_digest: str
    # Original event identities, never renumbered or reconstructed content.
    history: tuple[tuple[str, int, str], ...] = ()
    # This root's attachments enter the node only through its bound input.
    # Past root attachments remain available under the root history policy.
    current_root_attempt_id: str | None = None

    def __post_init__(self):
        if type(self.attempt) is not AttemptRef:
            raise TypeError("node view requires an exact attempt")
        if (type(self.definition_digest) is not str or len(self.definition_digest) != 64
                or any(c not in "0123456789abcdef" for c in self.definition_digest)):
            raise ValueError("node view definition digest is invalid")
        if type(self.history) is not tuple or len(self.history) > 10000:
            raise ValueError("node view history is invalid")
        if self.current_root_attempt_id is not None and (
                type(self.current_root_attempt_id) is not str
                or not self.current_root_attempt_id
                or self.current_root_attempt_id != self.current_root_attempt_id.strip()
                or self.current_root_attempt_id == self.attempt.attempt_id):
            raise ValueError("node view current root identity is invalid")
        previous = 0
        ids = set()
        for row in self.history:
            if (type(row) is not tuple or len(row) != 3 or type(row[0]) is not str or not row[0]
                    or type(row[1]) is not int or row[1] <= previous
                    or type(row[2]) is not str or len(row[2]) != 64
                    or any(c not in "0123456789abcdef" for c in row[2]) or row[0] in ids):
                raise ValueError("node view history identity is invalid")
            previous = row[1]
            ids.add(row[0])

    def to_dict(self):
        value = {"schema": _SCHEMA, "attempt": self.attempt.to_dict(),
                 "definition_digest": self.definition_digest,
                 "history": [{"event_id": i, "store_seq": s, "sha256": h} for i, s, h in self.history]}
        if self.current_root_attempt_id is not None:
            value["schema"] = _SCHEMA_V2
            value["current_root_attempt_id"] = self.current_root_attempt_id
        return value

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict:
            raise ValueError("node view schema is invalid")
        fields = {"schema", "attempt", "definition_digest", "history"}
        if value.get("schema") == _SCHEMA_V2:
            fields.add("current_root_attempt_id")
            if value.get("current_root_attempt_id") is None:
                raise ValueError("node view current root identity is invalid")
        elif value.get("schema") != _SCHEMA:
            raise ValueError("node view schema is invalid")
        if set(value) != fields:
            raise ValueError("node view schema is invalid")
        attempt = AttemptRef.from_dict(value["attempt"])
        if attempt.to_dict() != value["attempt"] or type(value["history"]) is not list:
            raise ValueError("node view representation is invalid")
        rows = []
        for row in value["history"]:
            if type(row) is not dict or set(row) != {"event_id", "store_seq", "sha256"}:
                raise ValueError("node view history representation is invalid")
            rows.append((row["event_id"], row["store_seq"], row["sha256"]))
        return cls(attempt, value["definition_digest"], tuple(rows), value.get("current_root_attempt_id"))

    @property
    def digest(self):
        return sha256(_bytes(self.to_dict())).hexdigest()

    @property
    def operation_id(self):
        return "node-memory-view." + sha256(_bytes(self.attempt.to_dict())).hexdigest()

    def _request(self):
        payload = self.to_dict()
        return JournalAppendRequest(event_id="event." + self.operation_id,
            event_type=_EVENT, attempt=self.attempt,
            operation=build_operation_ref(self.operation_id, domain=_EVENT, payload=payload), payload=payload)

    def _history_events(self, snapshot):
        if type(snapshot) is not JournalSnapshot or snapshot.execution_id != self.attempt.generation.execution_id:
            raise ValueError("node view journal scope differs")
        # Revalidate the complete prefix before selecting from it.
        snapshot = JournalSnapshot.from_dict(snapshot.to_dict())
        selected = []
        for event_id, seq, digest in self.history:
            if seq > len(snapshot.events):
                raise ValueError("node view history receipt is unavailable")
            event = snapshot.events[seq - 1]
            payload = event.payload
            if (event.event_id != event_id or journal_event_sha256(event) != digest
                    or event.attempt.generation != self.attempt.generation
                    or event.attempt == self.attempt or event.event_type not in _HISTORY_TYPES
                    or payload.get("parent_run_id")
                    or any(payload.get(key) is not None for key in (
                        "workflow_node_id", "workflow_step_index", "workflow_step_count"))
                    or payload.get("run_id") != event.attempt.attempt_id):
                raise ValueError("node view history is not a public root receipt")
            selected.append(event)
        return tuple(selected)

    def persist(self, journal):
        if journal.execution_id != self.attempt.generation.execution_id:
            raise ValueError("node view journal scope differs")
        def before_first_event(snapshot):
            self._history_events(snapshot)
            if any(e.attempt == self.attempt for e in snapshot.events):
                raise ValueError("node view must precede the node attempt")
        return journal.append_with_artifacts(request=self._request(), artifacts=(),
                                            precondition=before_first_event)

    def select(self, snapshot):
        history = self._history_events(snapshot)
        expected = self._request()
        receipt = next((e for e in snapshot.events if e.event_id == expected.event_id), None)
        if (receipt is None or receipt.event_type != _EVENT or receipt.attempt != self.attempt
                or receipt.operation != expected.operation or dict(receipt.payload) != expected.payload
                or receipt.resource_refs):
            raise ValueError("node view durable binding is missing or changed")
        if any(e.store_seq >= receipt.store_seq for e in history) or any(
            e.attempt == self.attempt and e.store_seq < receipt.store_seq for e in snapshot.events
        ):
            raise ValueError("node view binding is out of sequence")
        visible = {e.event_id for e in history}
        return tuple(e for e in snapshot.events if e.event_id in visible or
                     (e.attempt == self.attempt and e.event_type != _EVENT))
