from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from unchain.journal import EventCursor, EventRange, ResourceRef


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CheckpointProjectionError(ValueError):
    """Raised when checkpoint coverage is not exact and reproducible."""


def _canonical_bytes(value: Any) -> bytes:
    def plain(candidate: Any) -> Any:
        if isinstance(candidate, Mapping):
            return {str(key): plain(item) for key, item in candidate.items()}
        if isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes, bytearray)
        ):
            return [plain(item) for item in candidate]
        return candidate

    return json.dumps(
        plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def checkpoint_event_sha256(value: Mapping[str, Any]) -> str:
    """Hash one normalized semantic journal event for manifest binding."""

    if not isinstance(value, Mapping):
        raise TypeError("checkpoint event must be an object")
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sequence_digest(values: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = _canonical_bytes(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CheckpointProjectionError(f"{field_name} is invalid")
    return value


def _identifier(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value) is None
    ):
        raise CheckpointProjectionError(f"{field_name} is invalid")
    return value


@dataclass(frozen=True)
class CheckpointProjectionDependency:
    """A journal receipt required to authorize one materialized message."""

    SCHEMA: ClassVar[str] = "unchain.checkpoint_projection_dependency.v1"

    source_cursor: EventCursor
    receipt_cursor: EventCursor
    event_type: str
    attempt_id: str
    purpose: str
    status: str
    event_sha256: str
    workflow_node_id: str = ""
    workflow_step_index: int | None = None
    workflow_step_count: int | None = None
    iteration: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_cursor, EventCursor):
            object.__setattr__(
                self,
                "source_cursor",
                EventCursor.from_dict(self.source_cursor),
            )
        if not isinstance(self.receipt_cursor, EventCursor):
            object.__setattr__(
                self,
                "receipt_cursor",
                EventCursor.from_dict(self.receipt_cursor),
            )
        if (
            self.source_cursor.store_seq <= 0
            or self.receipt_cursor.store_seq <= self.source_cursor.store_seq
        ):
            raise CheckpointProjectionError(
                "checkpoint receipt must follow its source event"
            )
        if self.event_type != "run_completed":
            raise CheckpointProjectionError(
                "checkpoint dependency event type is invalid"
            )
        object.__setattr__(
            self,
            "attempt_id",
            _identifier(self.attempt_id, "checkpoint dependency attempt_id"),
        )
        if self.purpose != "assistant_commit":
            raise CheckpointProjectionError(
                "checkpoint dependency purpose is invalid"
            )
        if self.status != "completed":
            raise CheckpointProjectionError(
                "checkpoint dependency status is invalid"
            )
        object.__setattr__(
            self,
            "event_sha256",
            _sha256(self.event_sha256, "checkpoint dependency event_sha256"),
        )
        node_id = self.workflow_node_id
        if node_id:
            node_id = _identifier(node_id, "checkpoint dependency workflow_node_id")
        elif node_id != "":
            raise CheckpointProjectionError(
                "checkpoint dependency workflow_node_id is invalid"
            )
        object.__setattr__(self, "workflow_node_id", node_id)
        step_index = self.workflow_step_index
        step_count = self.workflow_step_count
        if (step_index is None) != (step_count is None):
            raise CheckpointProjectionError(
                "checkpoint dependency workflow step identity is incomplete"
            )
        if step_index is not None:
            if (
                isinstance(step_index, bool)
                or not isinstance(step_index, int)
                or isinstance(step_count, bool)
                or not isinstance(step_count, int)
                or step_count <= 0
                or step_index < 0
                or step_index >= step_count
            ):
                raise CheckpointProjectionError(
                    "checkpoint dependency workflow step identity is invalid"
                )
        if self.iteration is not None and (
            isinstance(self.iteration, bool)
            or not isinstance(self.iteration, int)
            or self.iteration < 0
        ):
            raise CheckpointProjectionError(
                "checkpoint dependency iteration is invalid"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "source_cursor": self.source_cursor.to_dict(),
            "receipt_cursor": self.receipt_cursor.to_dict(),
            "event_type": self.event_type,
            "attempt_id": self.attempt_id,
            "purpose": self.purpose,
            "status": self.status,
            "workflow_node_id": self.workflow_node_id,
            "workflow_step_index": self.workflow_step_index,
            "workflow_step_count": self.workflow_step_count,
            "iteration": self.iteration,
            "event_sha256": self.event_sha256,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> CheckpointProjectionDependency:
        expected = {
            "schema",
            "source_cursor",
            "receipt_cursor",
            "event_type",
            "attempt_id",
            "purpose",
            "status",
            "workflow_node_id",
            "workflow_step_index",
            "workflow_step_count",
            "iteration",
            "event_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CheckpointProjectionError(
                "checkpoint projection dependency is invalid"
            )
        if value.get("schema") != cls.SCHEMA:
            raise CheckpointProjectionError(
                "checkpoint projection dependency schema is invalid"
            )
        return cls(
            source_cursor=EventCursor.from_dict(value.get("source_cursor")),
            receipt_cursor=EventCursor.from_dict(value.get("receipt_cursor")),
            event_type=value.get("event_type"),
            attempt_id=value.get("attempt_id"),
            purpose=value.get("purpose"),
            status=value.get("status"),
            workflow_node_id=value.get("workflow_node_id"),
            workflow_step_index=value.get("workflow_step_index"),
            workflow_step_count=value.get("workflow_step_count"),
            iteration=value.get("iteration"),
            event_sha256=value.get("event_sha256"),
        )


def _validate_source_events(
    event_ids: tuple[Any, ...],
    store_seqs: tuple[Any, ...],
) -> None:
    if not event_ids or len(event_ids) != len(store_seqs):
        raise CheckpointProjectionError(
            "checkpoint events and store sequences must be non-empty and aligned"
        )
    if any(
        not isinstance(event_id, str)
        or not event_id
        or event_id != event_id.strip()
        for event_id in event_ids
    ):
        raise CheckpointProjectionError("checkpoint event identifiers are invalid")
    if len(set(event_ids)) != len(event_ids):
        raise CheckpointProjectionError(
            "checkpoint event identifiers must be unique"
        )
    if any(
        isinstance(store_seq, bool)
        or not isinstance(store_seq, int)
        or store_seq <= 0
        for store_seq in store_seqs
    ):
        raise CheckpointProjectionError("checkpoint store sequences are invalid")
    if any(
        current <= previous
        for previous, current in zip(store_seqs, store_seqs[1:])
    ):
        raise CheckpointProjectionError(
            "checkpoint store sequences must be strictly increasing"
        )


def _legacy_checkpoint_request_id(
    event_ids: tuple[str, ...],
    store_seqs: tuple[int, ...],
    source_messages_sha256: str,
) -> str:
    return "checkpoint-" + hashlib.sha256(
        _canonical_bytes(
            {
                "source_event_ids": event_ids,
                "source_event_store_seqs": store_seqs,
                "source_messages_sha256": source_messages_sha256,
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class LegacyCheckpointRequest:
    """A v1 manifest that can be inspected but never authorize model input."""

    SCHEMA: ClassVar[str] = "unchain.checkpoint_request.v1"

    request_id: str
    source_range: EventRange
    source_event_ids: tuple[str, ...]
    source_event_store_seqs: tuple[int, ...]
    source_messages_sha256: str

    @property
    def authoritative(self) -> bool:
        return False

    @property
    def event_count(self) -> int:
        return len(self.source_event_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "request_id": self.request_id,
            "source_range": self.source_range.to_dict(),
            "source_event_ids": list(self.source_event_ids),
            "source_event_store_seqs": list(self.source_event_store_seqs),
            "source_messages_sha256": self.source_messages_sha256,
            "event_count": self.event_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LegacyCheckpointRequest:
        expected = {
            "schema",
            "request_id",
            "source_range",
            "source_event_ids",
            "source_event_store_seqs",
            "source_messages_sha256",
            "event_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CheckpointProjectionError("checkpoint request record is invalid")
        if value.get("schema") != cls.SCHEMA:
            raise CheckpointProjectionError("checkpoint request schema is invalid")
        event_ids = tuple(value.get("source_event_ids") or ())
        store_seqs = tuple(value.get("source_event_store_seqs") or ())
        _validate_source_events(event_ids, store_seqs)
        digest = _sha256(
            value.get("source_messages_sha256"),
            "checkpoint source digest",
        )
        source_range = EventRange.from_dict(value.get("source_range"))
        if (
            source_range.start.event_id != event_ids[0]
            or source_range.start.store_seq != store_seqs[0]
            or source_range.end.event_id != event_ids[-1]
            or source_range.end.store_seq != store_seqs[-1]
            or value.get("event_count") != len(event_ids)
        ):
            raise CheckpointProjectionError(
                "checkpoint source range is inconsistent"
            )
        expected_request_id = _legacy_checkpoint_request_id(
            event_ids,
            store_seqs,
            digest,
        )
        if value.get("request_id") != expected_request_id:
            raise CheckpointProjectionError(
                "checkpoint request digest is inconsistent"
            )
        return cls(
            request_id=expected_request_id,
            source_range=source_range,
            source_event_ids=event_ids,
            source_event_store_seqs=store_seqs,
            source_messages_sha256=digest,
        )


@dataclass(frozen=True)
class CheckpointRequest:
    SCHEMA: ClassVar[str] = "unchain.checkpoint_request.v2"

    request_id: str
    source_range: EventRange
    source_event_ids: tuple[str, ...]
    source_event_store_seqs: tuple[int, ...]
    source_event_sha256s: tuple[str, ...]
    source_messages_sha256: str
    projection_dependencies: tuple[CheckpointProjectionDependency, ...] = ()

    def __post_init__(self) -> None:
        event_ids = tuple(self.source_event_ids)
        store_seqs = tuple(self.source_event_store_seqs)
        _validate_source_events(event_ids, store_seqs)
        event_sha256s = tuple(self.source_event_sha256s)
        if len(event_sha256s) != len(event_ids):
            raise CheckpointProjectionError(
                "checkpoint source event digests are unaligned"
            )
        event_sha256s = tuple(
            _sha256(item, "checkpoint source event digest")
            for item in event_sha256s
        )
        dependencies = tuple(
            item
            if isinstance(item, CheckpointProjectionDependency)
            else CheckpointProjectionDependency.from_dict(item)
            for item in self.projection_dependencies
        )
        _validate_dependencies(
            event_ids=event_ids,
            store_seqs=store_seqs,
            dependencies=dependencies,
        )
        source_messages_sha256 = _sha256(
            self.source_messages_sha256,
            "checkpoint source digest",
        )
        source_range = self.source_range
        if not isinstance(source_range, EventRange):
            source_range = EventRange.from_dict(source_range)
        if source_range != _coverage_range(event_ids, store_seqs, dependencies):
            raise CheckpointProjectionError(
                "checkpoint source range is inconsistent"
            )
        expected_request_id = _checkpoint_request_id(
            event_ids,
            store_seqs,
            event_sha256s,
            source_messages_sha256,
            dependencies,
        )
        if self.request_id != expected_request_id:
            raise CheckpointProjectionError(
                "checkpoint request digest is inconsistent"
            )
        object.__setattr__(self, "source_range", source_range)
        object.__setattr__(self, "source_event_ids", event_ids)
        object.__setattr__(self, "source_event_store_seqs", store_seqs)
        object.__setattr__(self, "source_event_sha256s", event_sha256s)
        object.__setattr__(
            self,
            "source_messages_sha256",
            source_messages_sha256,
        )
        object.__setattr__(self, "projection_dependencies", dependencies)

    @property
    def authoritative(self) -> bool:
        return True

    @property
    def event_count(self) -> int:
        return len(self.source_event_ids)

    @property
    def dependency_count(self) -> int:
        return len(self.projection_dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "request_id": self.request_id,
            "source_range": self.source_range.to_dict(),
            "source_event_ids": list(self.source_event_ids),
            "source_event_store_seqs": list(self.source_event_store_seqs),
            "source_event_sha256s": list(self.source_event_sha256s),
            "source_messages_sha256": self.source_messages_sha256,
            "projection_dependencies": [
                dependency.to_dict()
                for dependency in self.projection_dependencies
            ],
            "event_count": self.event_count,
            "dependency_count": self.dependency_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointRequest:
        expected = {
            "schema",
            "request_id",
            "source_range",
            "source_event_ids",
            "source_event_store_seqs",
            "source_event_sha256s",
            "source_messages_sha256",
            "projection_dependencies",
            "event_count",
            "dependency_count",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CheckpointProjectionError("checkpoint request record is invalid")
        if value.get("schema") != cls.SCHEMA:
            raise CheckpointProjectionError("checkpoint request schema is invalid")
        event_ids = tuple(value.get("source_event_ids") or ())
        store_seqs = tuple(value.get("source_event_store_seqs") or ())
        _validate_source_events(event_ids, store_seqs)
        event_sha256s = tuple(value.get("source_event_sha256s") or ())
        if len(event_sha256s) != len(event_ids):
            raise CheckpointProjectionError(
                "checkpoint source event digests are unaligned"
            )
        event_sha256s = tuple(
            _sha256(item, "checkpoint source event digest")
            for item in event_sha256s
        )
        raw_dependencies = value.get("projection_dependencies")
        if not isinstance(raw_dependencies, Sequence) or isinstance(
            raw_dependencies, (str, bytes, bytearray)
        ):
            raise CheckpointProjectionError(
                "checkpoint projection dependencies are invalid"
            )
        dependencies = tuple(
            item
            if isinstance(item, CheckpointProjectionDependency)
            else CheckpointProjectionDependency.from_dict(item)
            for item in raw_dependencies
        )
        digest = _sha256(
            value.get("source_messages_sha256"),
            "checkpoint source digest",
        )
        source_range = EventRange.from_dict(value.get("source_range"))
        _validate_dependencies(
            event_ids=event_ids,
            store_seqs=store_seqs,
            dependencies=dependencies,
        )
        expected_range = _coverage_range(event_ids, store_seqs, dependencies)
        if (
            source_range != expected_range
            or value.get("event_count") != len(event_ids)
            or value.get("dependency_count") != len(dependencies)
        ):
            raise CheckpointProjectionError(
                "checkpoint source range is inconsistent"
            )
        expected_request_id = _checkpoint_request_id(
            event_ids,
            store_seqs,
            event_sha256s,
            digest,
            dependencies,
        )
        if value.get("request_id") != expected_request_id:
            raise CheckpointProjectionError(
                "checkpoint request digest is inconsistent"
            )
        return cls(
            request_id=expected_request_id,
            source_range=source_range,
            source_event_ids=event_ids,
            source_event_store_seqs=store_seqs,
            source_event_sha256s=event_sha256s,
            source_messages_sha256=digest,
            projection_dependencies=dependencies,
        )


def _validate_dependencies(
    *,
    event_ids: tuple[str, ...],
    store_seqs: tuple[int, ...],
    dependencies: tuple[CheckpointProjectionDependency, ...],
) -> None:
    source_cursors = set(zip(event_ids, store_seqs, strict=True))
    cursors_by_event_id = dict(zip(event_ids, store_seqs, strict=True))
    cursors_by_store_seq = dict(zip(store_seqs, event_ids, strict=True))
    dependency_sources: set[tuple[str, int]] = set()
    receipt_cursors: set[tuple[str, int]] = set()
    for dependency in dependencies:
        source = (
            dependency.source_cursor.event_id,
            dependency.source_cursor.store_seq,
        )
        receipt = (
            dependency.receipt_cursor.event_id,
            dependency.receipt_cursor.store_seq,
        )
        if source not in source_cursors:
            raise CheckpointProjectionError(
                "checkpoint dependency source is outside the manifest"
            )
        if (
            receipt[0] in cursors_by_event_id
            or receipt[1] in cursors_by_store_seq
        ):
            raise CheckpointProjectionError(
                "checkpoint dependency receipt cursor aliases a source cursor"
            )
        if source in dependency_sources or receipt in receipt_cursors:
            raise CheckpointProjectionError(
                "checkpoint projection dependencies must be unique"
            )
        dependency_sources.add(source)
        receipt_cursors.add(receipt)
        cursors_by_event_id[receipt[0]] = receipt[1]
        cursors_by_store_seq[receipt[1]] = receipt[0]


def _coverage_range(
    event_ids: tuple[str, ...],
    store_seqs: tuple[int, ...],
    dependencies: tuple[CheckpointProjectionDependency, ...],
) -> EventRange:
    cursors = [
        EventCursor(store_seq=store_seq, event_id=event_id)
        for event_id, store_seq in zip(event_ids, store_seqs, strict=True)
    ]
    cursors.extend(dependency.receipt_cursor for dependency in dependencies)
    ordered = sorted(cursors, key=lambda cursor: (cursor.store_seq, cursor.event_id))
    return EventRange(start=ordered[0], end=ordered[-1])


def _checkpoint_request_id(
    event_ids: tuple[str, ...],
    store_seqs: tuple[int, ...],
    source_event_sha256s: tuple[str, ...],
    source_messages_sha256: str,
    dependencies: tuple[CheckpointProjectionDependency, ...],
) -> str:
    request_payload = {
        "schema": CheckpointRequest.SCHEMA,
        "source_event_ids": event_ids,
        "source_event_store_seqs": store_seqs,
        "source_event_sha256s": source_event_sha256s,
        "source_messages_sha256": source_messages_sha256,
        "projection_dependencies": [
            dependency.to_dict() for dependency in dependencies
        ],
    }
    return "checkpoint-" + hashlib.sha256(
        _canonical_bytes(request_payload)
    ).hexdigest()


def build_checkpoint_request(
    *,
    source_event_ids: Sequence[str],
    source_event_store_seqs: Sequence[int],
    source_messages: Sequence[Mapping[str, Any]],
    source_event_sha256s: Sequence[str] | None = None,
    projection_dependencies: Sequence[CheckpointProjectionDependency] = (),
) -> CheckpointRequest:
    """Describe exact sparse journal sources and their eligibility receipts."""

    event_ids = tuple(source_event_ids)
    store_seqs = tuple(source_event_store_seqs)
    _validate_source_events(event_ids, store_seqs)
    if len(source_messages) != len(event_ids):
        raise CheckpointProjectionError(
            "checkpoint requires one source message per source event"
        )
    normalized_messages = tuple(dict(message) for message in source_messages)
    if source_event_sha256s is None:
        event_sha256s = tuple(
            checkpoint_event_sha256(
                {
                    "event_id": event_id,
                    "store_seq": store_seq,
                    "message": message,
                    "provenance": "prevalidated_source_message",
                }
            )
            for event_id, store_seq, message in zip(
                event_ids,
                store_seqs,
                normalized_messages,
                strict=True,
            )
        )
    else:
        event_sha256s = tuple(
            _sha256(item, "checkpoint source event digest")
            for item in source_event_sha256s
        )
        if len(event_sha256s) != len(event_ids):
            raise CheckpointProjectionError(
                "checkpoint source event digests are unaligned"
            )
    dependencies = tuple(
        item
        if isinstance(item, CheckpointProjectionDependency)
        else CheckpointProjectionDependency.from_dict(item)
        for item in projection_dependencies
    )
    _validate_dependencies(
        event_ids=event_ids,
        store_seqs=store_seqs,
        dependencies=dependencies,
    )
    source_messages_sha256 = _sequence_digest(normalized_messages)
    source_range = _coverage_range(event_ids, store_seqs, dependencies)
    request_id = _checkpoint_request_id(
        event_ids,
        store_seqs,
        event_sha256s,
        source_messages_sha256,
        dependencies,
    )
    return CheckpointRequest(
        request_id=request_id,
        source_range=source_range,
        source_event_ids=tuple(event_id.strip() for event_id in event_ids),
        source_event_store_seqs=store_seqs,
        source_event_sha256s=event_sha256s,
        source_messages_sha256=source_messages_sha256,
        projection_dependencies=dependencies,
    )


def read_checkpoint_request_record(
    value: Mapping[str, Any],
) -> CheckpointRequest | LegacyCheckpointRequest:
    """Read v2 authority or a content-inert v1 audit record."""

    if not isinstance(value, Mapping):
        raise TypeError("checkpoint request record must be an object")
    if value.get("schema") == CheckpointRequest.SCHEMA:
        return CheckpointRequest.from_dict(value)
    if value.get("schema") == LegacyCheckpointRequest.SCHEMA:
        return LegacyCheckpointRequest.from_dict(value)
    raise CheckpointProjectionError("checkpoint request schema is invalid")


def project_checkpoint_message(
    *,
    checkpoint_ref: ResourceRef,
    request: CheckpointRequest,
    omitted_complete_turns: int,
) -> dict[str, Any]:
    """Project a durable v2 checkpoint as untrusted provider-neutral context."""

    if not isinstance(request, CheckpointRequest) or not request.authoritative:
        raise CheckpointProjectionError(
            "only a checkpoint request v2 can authorize model projection"
        )
    if checkpoint_ref.kind != "checkpoint":
        raise CheckpointProjectionError(
            "checkpoint_ref must reference a checkpoint"
        )
    if (
        isinstance(omitted_complete_turns, bool)
        or not isinstance(omitted_complete_turns, int)
        or omitted_complete_turns <= 0
    ):
        raise CheckpointProjectionError(
            "omitted_complete_turns must be positive"
        )
    payload = {
        "schema_version": "memory_v2.context.v2",
        "omitted_complete_turns": omitted_complete_turns,
        "checkpoint_ref": checkpoint_ref.to_dict(),
        "source_event_range": {
            "first_event_id": request.source_range.start.event_id,
            "last_event_id": request.source_range.end.event_id,
            "first_store_seq": request.source_range.start.store_seq,
            "last_store_seq": request.source_range.end.store_seq,
            "event_count": request.event_count,
            "dependency_count": request.dependency_count,
            "source_messages_sha256": request.source_messages_sha256,
            "checkpoint_request_id": request.request_id,
        },
        "trust": "UNTRUSTED_DATA",
    }
    from .untrusted import untrusted_context_message

    return untrusted_context_message("MEMORY_V2_CHECKPOINT", payload)


__all__ = [
    "CheckpointProjectionDependency",
    "CheckpointProjectionError",
    "CheckpointRequest",
    "LegacyCheckpointRequest",
    "build_checkpoint_request",
    "checkpoint_event_sha256",
    "project_checkpoint_message",
    "read_checkpoint_request_record",
]
