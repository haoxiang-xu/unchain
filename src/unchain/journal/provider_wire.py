"""Bound journal receipt and raw-byte recovery for provider wire envelopes.

This module defines only the process/durable-storage boundary.  Concrete CAS,
artifact, and journal repositories remain host responsibilities.
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.providers.wire_envelope import (
    MAX_PROVIDER_WIRE_BYTES,
    ProviderWireEnvelope,
)

from .models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    GenerationRef,
    JournalAppendRequest,
    JournalAppendResult,
    JournalEvent,
    ModelValidationError,
    OperationRef,
    ResourceRef,
    _record_data,
)
from .ports import BoundExecutionJournal
from .resource_limits import JsonResourceLimits, validate_json_resource


PROVIDER_WIRE_SNAPSHOT_EVENT_TYPE = "provider.wire_snapshot"
MAX_PROVIDER_WIRE_RECEIPTS = 2
MAX_PROVIDER_WIRE_ITERATION = 2**31 - 1
PROVIDER_WIRE_RECEIPT_LOOKUP_LIMITS = JsonResourceLimits(
    max_items=MAX_PROVIDER_WIRE_RECEIPTS,
    max_bytes=256 * 1024,
    max_depth=32,
    max_nodes=4_096,
)
_PROVIDER_REVISIONS = {
    "gemini": "unchain.gemini.contents.request.v1",
    "openai": "unchain.openai.responses.request.v1",
    "anthropic": "unchain.anthropic.messages.request.v1",
    "hyperspace": "unchain.hyperspace.anthropic-messages.request.v1",
    "ollama": "unchain.ollama.chat.request.v1",
}
_PAYLOAD_FIELDS = frozenset(
    {
        "iteration",
        "provider",
        "adapter_revision",
        "catalog_sha256",
        "envelope_sha256",
        "wire_artifact",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProviderWireReceiptIntegrityError(RuntimeError):
    """Durable wire bytes, descriptor, event, or index disagreed."""


class ProviderWireReceiptNotFound(ProviderWireReceiptIntegrityError):
    """The bounded receipt index proved that no wire snapshot exists."""


def _require_iteration(value: object) -> int:
    if type(value) is not int:
        raise TypeError("iteration must be an exact integer")
    if not 0 <= value <= MAX_PROVIDER_WIRE_ITERATION:
        raise ModelValidationError(
            f"iteration must be between 0 and {MAX_PROVIDER_WIRE_ITERATION}"
        )
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ModelValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _require_attempt(value: object, field_name: str) -> AttemptRef:
    if type(value) is not AttemptRef or type(value.generation) is not GenerationRef:
        raise TypeError(f"{field_name} must be an exact AttemptRef")
    return value


def _require_provider(value: object) -> str:
    if type(value) is not str or value not in _PROVIDER_REVISIONS:
        raise ModelValidationError("provider is unsupported")
    return value


def _require_revision(value: object, *, provider: str) -> str:
    if type(value) is not str or value != _PROVIDER_REVISIONS[provider]:
        raise ModelValidationError("adapter_revision is unsupported for the provider")
    return value


def _validate_artifact(
    artifact: object,
    *,
    content: bytes | None = None,
) -> ArtifactRef:
    if type(artifact) is not ArtifactRef or type(artifact.ref) is not ResourceRef:
        raise ProviderWireReceiptIntegrityError(
            "wire artifact must be an exact ArtifactRef"
        )
    if artifact.ref.kind != "artifact" or artifact.ref.fragment:
        raise ProviderWireReceiptIntegrityError(
            "wire receipt must identify one whole artifact"
        )
    if artifact.media_type != "application/json":
        raise ProviderWireReceiptIntegrityError(
            "wire artifact must be a JSON artifact with application/json media type"
        )
    if artifact.preview:
        raise ProviderWireReceiptIntegrityError("wire artifact preview must be empty")
    if artifact.byte_length > MAX_PROVIDER_WIRE_BYTES:
        raise ProviderWireReceiptIntegrityError(
            "wire artifact exceeds the 64 MiB provider wire size limit"
        )
    if content is not None:
        if type(content) is not bytes:
            raise ProviderWireReceiptIntegrityError(
                "wire artifact readback must return exact bytes"
            )
        if artifact.byte_length != len(content):
            raise ProviderWireReceiptIntegrityError(
                "wire artifact bytes length does not match its descriptor"
            )
        if artifact.sha256 != hashlib.sha256(content).hexdigest():
            raise ProviderWireReceiptIntegrityError(
                "wire artifact sha256 digest does not match its bytes"
            )
    return artifact


@dataclass(frozen=True)
class _ReceiptFields:
    iteration: int
    provider: str
    adapter_revision: str
    catalog_sha256: str
    envelope_sha256: str
    artifact: ArtifactRef


def _receipt_fields(event: object) -> _ReceiptFields:
    if type(event) is not JournalEvent:
        raise TypeError("provider wire receipts require exact JournalEvent records")
    _require_attempt(event.attempt, "provider wire receipt attempt")
    if type(event.operation) is not OperationRef:
        raise TypeError("provider wire receipt operation must be exact")
    if any(type(ref) is not ResourceRef for ref in event.resource_refs):
        raise TypeError("provider wire receipt resource refs must be exact")
    if event.event_type != PROVIDER_WIRE_SNAPSHOT_EVENT_TYPE:
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt has an unsupported event type"
        )
    if frozenset(event.payload) != _PAYLOAD_FIELDS:
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt payload fields are not canonical"
        )
    try:
        iteration = _require_iteration(event.payload.get("iteration"))
        provider = _require_provider(event.payload.get("provider"))
        adapter_revision = _require_revision(
            event.payload.get("adapter_revision"),
            provider=provider,
        )
        catalog_sha256 = _require_sha256(
            event.payload.get("catalog_sha256"),
            "catalog_sha256",
        )
        envelope_sha256 = _require_sha256(
            event.payload.get("envelope_sha256"),
            "envelope_sha256",
        )
    except (TypeError, ValueError) as exc:
        raise ProviderWireReceiptIntegrityError(str(exc)) from exc
    raw_artifact = event.payload.get("wire_artifact")
    if not isinstance(raw_artifact, Mapping):
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt lacks its artifact descriptor"
        )
    try:
        artifact = ArtifactRef.from_dict(raw_artifact)
    except (TypeError, ValueError) as exc:
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt artifact descriptor is malformed"
        ) from exc
    _validate_artifact(artifact)
    if event.resource_refs != (artifact.ref,):
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt resource refs must contain exactly its artifact"
        )
    return _ReceiptFields(
        iteration=iteration,
        provider=provider,
        adapter_revision=adapter_revision,
        catalog_sha256=catalog_sha256,
        envelope_sha256=envelope_sha256,
        artifact=artifact,
    )


@dataclass(frozen=True)
class ProviderWireReceiptLookup:
    """One exhaustive bounded lookup for an attempt/iteration wire snapshot."""

    SCHEMA: ClassVar[str] = "unchain.provider_wire_receipt_lookup.v1"

    attempt: AttemptRef
    iteration: int
    events: tuple[JournalEvent, ...] = ()
    overflow: bool = False

    def __post_init__(self) -> None:
        _require_attempt(self.attempt, "attempt")
        object.__setattr__(self, "iteration", _require_iteration(self.iteration))
        if type(self.events) is not tuple:
            raise TypeError("events must be an exact tuple")
        if len(self.events) > MAX_PROVIDER_WIRE_RECEIPTS:
            raise ModelValidationError(
                "provider wire receipt lookup may contain at most two events"
            )
        if type(self.overflow) is not bool:
            raise TypeError("overflow must be an exact boolean")

        previous_store_seq = 0
        event_ids: set[str] = set()
        operation_ids: set[str] = set()
        for event in self.events:
            fields = _receipt_fields(event)
            if event.attempt != self.attempt:
                raise ModelValidationError(
                    "provider wire receipt belongs to a foreign attempt"
                )
            if fields.iteration != self.iteration:
                raise ModelValidationError("provider wire receipt iteration changed")
            if event.store_seq <= previous_store_seq:
                raise ModelValidationError(
                    "provider wire receipts must be strictly ordered"
                )
            previous_store_seq = event.store_seq
            if (
                event.event_id in event_ids
                or event.operation.operation_id in operation_ids
            ):
                raise ModelValidationError(
                    "provider wire receipt identity is duplicated"
                )
            event_ids.add(event.event_id)
            operation_ids.add(event.operation.operation_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.to_dict(),
            "iteration": self.iteration,
            "events": [event.to_dict() for event in self.events],
            "overflow": self.overflow,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProviderWireReceiptLookup:
        if type(value) is not dict:
            raise TypeError("provider wire receipt lookup must be an exact dict")
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"attempt", "iteration", "events", "overflow"}),
        )
        raw_events = raw["events"]
        if type(raw_events) is not list:
            raise TypeError("events must be an exact list")
        if len(raw_events) > MAX_PROVIDER_WIRE_RECEIPTS:
            raise ModelValidationError(
                "provider wire receipt lookup may contain at most two events"
            )
        if any(type(event) is not dict for event in raw_events):
            raise TypeError("each provider wire receipt event must be an exact dict")
        validate_json_resource(
            raw,
            boundary="provider_wire_receipt_lookup",
            limits=PROVIDER_WIRE_RECEIPT_LOOKUP_LIMITS,
        )
        return cls(
            attempt=AttemptRef.from_dict(raw["attempt"]),
            iteration=raw["iteration"],
            events=tuple(JournalEvent.from_dict(event) for event in raw_events),
            overflow=raw["overflow"],
        )


class BoundProviderWireStore(BoundExecutionJournal):
    """Execution-bound CAS, exact-read, append, and receipt-index contract."""

    @abstractmethod
    def write_provider_wire_cas(
        self,
        *,
        content: bytes,
        media_type: str,
        preview: str,
        operation: OperationRef,
        expected_revision: int,
    ) -> ArtifactRef:
        """CAS-write the complete immutable wire bytes."""

    @abstractmethod
    def read_provider_wire_full_verified(self, *, artifact: ArtifactRef) -> bytes:
        """Scope-authorize and return the complete verified object."""

    @abstractmethod
    def lookup_provider_wire_receipts(
        self,
        *,
        attempt: AttemptRef,
        iteration: int,
    ) -> ProviderWireReceiptLookup:
        """Return the exhaustive bounded receipt set for one subject."""


@dataclass(frozen=True)
class ProviderWireSnapshotReceipt:
    envelope: ProviderWireEnvelope
    artifact: ArtifactRef
    event: JournalEvent
    cursor: EventCursor

    def __post_init__(self) -> None:
        if type(self.envelope) is not ProviderWireEnvelope:
            raise TypeError("envelope must be an exact ProviderWireEnvelope")
        _validate_artifact(self.artifact, content=self.envelope.canonical_bytes())
        fields = _receipt_fields(self.event)
        if type(self.cursor) is not EventCursor:
            raise TypeError("cursor must be an exact EventCursor")
        if (
            self.cursor.store_seq != self.event.store_seq
            or self.cursor.event_id != self.event.event_id
        ):
            raise ProviderWireReceiptIntegrityError(
                "provider wire receipt cursor changed"
            )
        if (
            self.event.attempt != self.envelope.attempt
            or fields.iteration != self.envelope.iteration
            or fields.provider != self.envelope.provider
            or fields.adapter_revision != self.envelope.adapter_revision
            or fields.catalog_sha256 != self.envelope.catalog_sha256
            or fields.envelope_sha256 != self.envelope.envelope_sha256
            or fields.artifact != self.artifact
        ):
            raise ProviderWireReceiptIntegrityError(
                "provider wire snapshot receipt disagrees with its envelope"
            )


@dataclass(frozen=True)
class RecoveredProviderWireAuthority:
    """Plain-data result returned only after durable bytes are reconstructed."""

    envelope: ProviderWireEnvelope
    catalog: ToolCatalogEnvelope
    artifact: ArtifactRef
    event: JournalEvent
    cursor: EventCursor


def _same_append_request(
    request: JournalAppendRequest,
    event: JournalEvent,
) -> bool:
    return (
        event.event_id == request.event_id
        and event.event_type == request.event_type
        and event.attempt == request.attempt
        and event.operation == request.operation
        and dict(event.payload) == dict(request.payload)
        and event.resource_refs == request.resource_refs
    )


def _verify_readback(
    store: BoundProviderWireStore,
    *,
    artifact: ArtifactRef,
    expected: bytes,
) -> None:
    readback = store.read_provider_wire_full_verified(artifact=artifact)
    _validate_artifact(artifact, content=readback)
    if readback != expected:
        raise ProviderWireReceiptIntegrityError(
            "provider wire readback bytes differ from the canonical envelope"
        )


def persist_provider_wire_snapshot(
    store: BoundProviderWireStore,
    *,
    envelope: ProviderWireEnvelope,
    catalog: ToolCatalogEnvelope,
    artifact_operation: OperationRef,
    event_operation: OperationRef,
    event_id: str,
    expected_artifact_revision: int,
) -> ProviderWireSnapshotReceipt:
    """CAS-write, full-read, then append the exact durable wire receipt."""

    if not isinstance(store, BoundProviderWireStore):
        raise TypeError("store must be a BoundProviderWireStore")
    if type(envelope) is not ProviderWireEnvelope:
        raise TypeError("envelope must be an exact ProviderWireEnvelope")
    if type(catalog) is not ToolCatalogEnvelope:
        raise TypeError("catalog must be an exact ToolCatalogEnvelope")
    if type(artifact_operation) is not OperationRef:
        raise TypeError("artifact_operation must be an exact OperationRef")
    if type(event_operation) is not OperationRef:
        raise TypeError("event_operation must be an exact OperationRef")
    if type(expected_artifact_revision) is not int or expected_artifact_revision < 0:
        raise ValueError("expected_artifact_revision must be a non-negative integer")
    if store.execution_id != envelope.attempt.generation.execution_id:
        raise ProviderWireReceiptIntegrityError(
            "provider wire store execution scope does not match the envelope"
        )
    try:
        envelope.verify_against_catalog(catalog)
    except (TypeError, ValueError) as exc:
        raise ProviderWireReceiptIntegrityError(
            "provider wire envelope does not match the catalog"
        ) from exc

    content = envelope.canonical_bytes()
    artifact = store.write_provider_wire_cas(
        content=content,
        media_type="application/json",
        preview="",
        operation=artifact_operation,
        expected_revision=expected_artifact_revision,
    )
    _validate_artifact(artifact, content=content)
    _verify_readback(store, artifact=artifact, expected=content)

    payload = {
        "iteration": envelope.iteration,
        "provider": envelope.provider,
        "adapter_revision": envelope.adapter_revision,
        "catalog_sha256": envelope.catalog_sha256,
        "envelope_sha256": envelope.envelope_sha256,
        "wire_artifact": artifact.to_dict(),
    }
    request = JournalAppendRequest(
        event_id=event_id,
        event_type=PROVIDER_WIRE_SNAPSHOT_EVENT_TYPE,
        attempt=envelope.attempt,
        operation=event_operation,
        payload=payload,
        resource_refs=(artifact.ref,),
    )
    result = store.append(request=request)
    if type(result) is not JournalAppendResult:
        raise ProviderWireReceiptIntegrityError(
            "provider wire append did not return an exact result"
        )
    if not _same_append_request(request, result.event):
        raise ProviderWireReceiptIntegrityError(
            "persisted provider wire event differs from its append request"
        )
    return ProviderWireSnapshotReceipt(
        envelope=envelope,
        artifact=artifact,
        event=result.event,
        cursor=result.cursor,
    )


def _read_envelope(
    store: BoundProviderWireStore,
    *,
    artifact: ArtifactRef,
    catalog: ToolCatalogEnvelope,
) -> ProviderWireEnvelope:
    raw = store.read_provider_wire_full_verified(artifact=artifact)
    _validate_artifact(artifact, content=raw)
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderWireReceiptIntegrityError(
            "provider wire artifact is not valid UTF-8 JSON"
        ) from exc
    if type(decoded) is not dict:
        raise ProviderWireReceiptIntegrityError(
            "provider wire artifact must contain one JSON object"
        )
    try:
        envelope = ProviderWireEnvelope.from_dict(decoded)
        if envelope.canonical_bytes() != raw:
            raise ProviderWireReceiptIntegrityError(
                "provider wire artifact bytes are not canonical"
            )
        envelope.verify_against_catalog(catalog)
    except ProviderWireReceiptIntegrityError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProviderWireReceiptIntegrityError(
            "provider wire artifact envelope or catalog verification failed"
        ) from exc
    return envelope


def recover_provider_wire_authority(
    store: BoundProviderWireStore,
    *,
    attempt: AttemptRef,
    iteration: int,
    catalog: ToolCatalogEnvelope,
    expected_provider: str,
    expected_adapter_revision: str,
    expected_envelope_sha256: str,
    expected_artifact: ArtifactRef,
    expected_cursor: EventCursor,
) -> RecoveredProviderWireAuthority:
    """Recover one exact indexed receipt and reconstruct its canonical bytes."""

    if not isinstance(store, BoundProviderWireStore):
        raise TypeError("store must be a BoundProviderWireStore")
    attempt = _require_attempt(attempt, "attempt")
    iteration = _require_iteration(iteration)
    try:
        expected_provider = _require_provider(expected_provider)
        expected_adapter_revision = _require_revision(
            expected_adapter_revision,
            provider=expected_provider,
        )
        expected_envelope_sha256 = _require_sha256(
            expected_envelope_sha256,
            "expected_envelope_sha256",
        )
    except (TypeError, ValueError) as exc:
        raise ProviderWireReceiptIntegrityError(str(exc)) from exc
    if type(catalog) is not ToolCatalogEnvelope:
        raise TypeError("catalog must be an exact ToolCatalogEnvelope")
    _validate_artifact(expected_artifact)
    if type(expected_cursor) is not EventCursor:
        raise TypeError("expected_cursor must be an exact EventCursor")
    if store.execution_id != attempt.generation.execution_id:
        raise ProviderWireReceiptIntegrityError(
            "provider wire store execution scope does not match the attempt"
        )
    if (
        catalog.attempt != attempt
        or catalog.iteration != iteration
        or catalog.provider != expected_provider
    ):
        raise ProviderWireReceiptIntegrityError(
            "provider wire catalog crossed the requested subject or provider"
        )

    try:
        lookup = store.lookup_provider_wire_receipts(
            attempt=attempt,
            iteration=iteration,
        )
    except Exception as exc:
        raise ProviderWireReceiptIntegrityError(
            f"provider wire index violated the requested subject: {exc}"
        ) from exc
    if type(lookup) is not ProviderWireReceiptLookup:
        raise ProviderWireReceiptIntegrityError(
            "provider wire index did not return the exact lookup record"
        )
    if lookup.attempt != attempt or lookup.iteration != iteration:
        raise ProviderWireReceiptIntegrityError(
            "provider wire index crossed its requested subject"
        )
    if lookup.overflow:
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt lookup overflowed its bounded result"
        )
    if not lookup.events:
        raise ProviderWireReceiptNotFound("provider wire receipt was not found")
    if len(lookup.events) != 1:
        raise ProviderWireReceiptIntegrityError(
            "provider wire recovery requires exactly one receipt; conflict detected"
        )

    event = lookup.events[0]
    fields = _receipt_fields(event)
    cursor = EventCursor(event.store_seq, event.event_id)
    if cursor != expected_cursor:
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt cursor does not match the expected cursor"
        )
    if event.attempt != attempt or fields.iteration != iteration:
        raise ProviderWireReceiptIntegrityError(
            "provider wire receipt crossed its requested subject"
        )
    if fields.provider != expected_provider:
        raise ProviderWireReceiptIntegrityError("provider wire provider changed")
    if fields.adapter_revision != expected_adapter_revision:
        raise ProviderWireReceiptIntegrityError(
            "provider wire adapter revision changed"
        )
    if fields.catalog_sha256 != catalog.catalog_sha256:
        raise ProviderWireReceiptIntegrityError("provider wire catalog digest changed")
    if fields.envelope_sha256 != expected_envelope_sha256:
        raise ProviderWireReceiptIntegrityError("provider wire envelope digest changed")
    if fields.artifact != expected_artifact:
        raise ProviderWireReceiptIntegrityError(
            "provider wire artifact descriptor changed"
        )

    envelope = _read_envelope(
        store,
        artifact=fields.artifact,
        catalog=catalog,
    )
    if (
        envelope.attempt != attempt
        or envelope.iteration != iteration
        or envelope.provider != fields.provider
        or envelope.adapter_revision != fields.adapter_revision
        or envelope.catalog_sha256 != fields.catalog_sha256
        or envelope.envelope_sha256 != fields.envelope_sha256
    ):
        raise ProviderWireReceiptIntegrityError(
            "provider wire artifact envelope disagrees with its receipt"
        )
    return RecoveredProviderWireAuthority(
        envelope=envelope,
        catalog=catalog,
        artifact=fields.artifact,
        event=event,
        cursor=cursor,
    )


__all__ = [
    "BoundProviderWireStore",
    "MAX_PROVIDER_WIRE_BYTES",
    "MAX_PROVIDER_WIRE_ITERATION",
    "MAX_PROVIDER_WIRE_RECEIPTS",
    "PROVIDER_WIRE_RECEIPT_LOOKUP_LIMITS",
    "PROVIDER_WIRE_SNAPSHOT_EVENT_TYPE",
    "ProviderWireReceiptIntegrityError",
    "ProviderWireReceiptLookup",
    "ProviderWireReceiptNotFound",
    "ProviderWireSnapshotReceipt",
    "RecoveredProviderWireAuthority",
    "persist_provider_wire_snapshot",
    "recover_provider_wire_authority",
]
