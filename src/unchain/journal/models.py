from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAINTEXT_SECRET_FIELDS = frozenset(
    {
        "access_token",
        "api_secret",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "aws_secret_access_key",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "id_token",
        "passwd",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "secret_value",
        "session_token",
        "signing_key",
        "webhook_secret",
    }
)
_CREDENTIAL_METADATA_SUFFIXES = {
    "_budget": "integer",
    "_bytes": "integer",
    "_configured": "boolean",
    "_count": "integer",
    "_counts": "integer",
    "_digest": "digest",
    "_handle": "handle",
    "_hash": "digest",
    "_id": "identifier",
    "_ids": "identifiers",
    "_length": "integer",
    "_limit": "integer",
    "_limits": "integer",
    "_present": "boolean",
    "_ref": "reference",
    "_refs": "references",
    "_sha256": "digest",
    "_usage": "usage",
}
_SAFE_TOKEN_METRIC_FIELDS = frozenset(
    {
        "attributed_tokens",
        "available_input_tokens",
        "cache_read_tokens",
        "cache_write_1h_tokens",
        "cache_write_5m_tokens",
        "cache_write_tokens",
        "cached_tokens",
        "completion_tokens",
        "consumed_tokens",
        "context_window_tokens",
        "estimated_input_tokens",
        "fixed_overhead_tokens",
        "input_tokens",
        "max_tokens",
        "output_reserve_tokens",
        "output_tokens",
        "pressure_threshold_tokens",
        "prompt_tokens",
        "reasoning_tokens",
        "residual_tokens",
        "tokens",
        "total_tokens",
        "transport_margin_tokens",
        "uncached_tokens",
        "visible_tokens",
    }
)
TOOL_EXECUTION_RECEIPT_TYPES = frozenset(
    {
        "tool_call",
        "tool.started",
        "tool.subagent_completion.sealed",
        "tool.hook_raw.sealed",
        "tool_result",
        "tool.result",
    }
)
MAX_TOOL_EXECUTION_RECEIPTS = 4


class ModelValidationError(ValueError):
    """Raised when a Context V2 boundary record is invalid."""


class ContextBuildStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    LEGACY = "legacy"
    UNAVAILABLE = "unavailable"


def _normalize_key(value: str) -> str:
    compatible = unicodedata.normalize("NFKC", value)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", compatible)
    return re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")


def _credential_metadata_kind(normalized_key: str) -> str | None:
    if normalized_key in _SAFE_TOKEN_METRIC_FIELDS:
        # Canonical usage contracts represent an unavailable observation as
        # null.  Keep that exception limited to the explicitly registered
        # metric names; arbitrary token-shaped keys remain credential fields.
        return "nullable_integer"
    segments = normalized_key.split("_")
    sensitive_segment = re.compile(
        r"^(?:authorization|bearer|cookie|credential|passwd|password|secret|token)s?[0-9]*$"
    )
    sensitive = normalized_key in _PLAINTEXT_SECRET_FIELDS or any(
        sensitive_segment.fullmatch(segment) is not None for segment in segments
    )
    key_prefixes = {"access", "api", "encryption", "private", "signing"}
    sensitive = sensitive or any(
        (
            segments[index] in key_prefixes
            and re.fullmatch(r"keys?[0-9]*", segments[index + 1]) is not None
        )
        or (
            segments[index] == "pass"
            and re.fullmatch(r"words?[0-9]*", segments[index + 1]) is not None
        )
        for index in range(len(segments) - 1)
    )
    if not sensitive:
        return None
    for suffix, metadata_kind in _CREDENTIAL_METADATA_SUFFIXES.items():
        if normalized_key.endswith(suffix):
            return metadata_kind
    return "plaintext"


def _validate_credential_metadata(
    normalized_key: str,
    value: Any,
    *,
    path: str,
) -> None:
    metadata_kind = _credential_metadata_kind(normalized_key)
    if metadata_kind is None:
        return
    if metadata_kind == "plaintext":
        raise ModelValidationError(f"{path} is a plaintext secret field; use an opaque handle")
    invalid_message = f"{path} is not valid opaque credential metadata"
    if metadata_kind == "identifier":
        if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind == "identifiers":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            raise ModelValidationError(invalid_message)
        if any(not isinstance(item, str) or _IDENTIFIER_RE.fullmatch(item) is None for item in value):
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind == "digest":
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind == "boolean":
        if not isinstance(value, bool):
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind == "nullable_integer":
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind == "usage":
        def valid_usage(item: Any) -> bool:
            if isinstance(item, bool):
                return False
            if isinstance(item, int):
                return item >= 0
            if isinstance(item, Mapping):
                return all(
                    isinstance(key, str) and valid_usage(child)
                    for key, child in item.items()
                )
            return False

        if not valid_usage(value):
            raise ModelValidationError(invalid_message)
        return
    if metadata_kind in {"reference", "references"}:
        values = value if metadata_kind == "references" else (value,)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ModelValidationError(invalid_message)
        for item in values:
            if not isinstance(item, Mapping):
                raise ModelValidationError(invalid_message)
            try:
                ResourceRef.from_dict(item)
            except (TypeError, ValueError) as exc:
                raise ModelValidationError(invalid_message) from exc
        return
    if metadata_kind == "handle":
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "handle_id",
            "label",
            "scope",
        }:
            raise ModelValidationError(invalid_message)
        if value.get("schema") != "unchain.opaque_secret_handle.v1":
            raise ModelValidationError(invalid_message)
        _required_text(value.get("handle_id"), "handle_id", identifier=True)
        _required_text(value.get("label"), "label", maximum=128)
        _required_text(value.get("scope"), "scope", identifier=True)
        return
    raise ModelValidationError(invalid_message)


def _required_text(
    value: Any,
    field_name: str,
    *,
    maximum: int = 256,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized or len(normalized) > maximum or "\x00" in normalized:
        raise ModelValidationError(f"{field_name} is invalid")
    if any(ord(character) < 32 for character in normalized):
        raise ModelValidationError(f"{field_name} contains control characters")
    if identifier and _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise ModelValidationError(f"{field_name} is not a valid identifier")
    return normalized


def _optional_text(value: Any, field_name: str, *, maximum: int = 4096) -> str:
    if value in (None, ""):
        return ""
    return _required_text(value, field_name, maximum=maximum)


def _bounded_int(value: Any, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < minimum:
        raise ModelValidationError(f"{field_name} must be at least {minimum}")
    return value


def _positive_revision(value: Any) -> int:
    return _bounded_int(value, "revision", minimum=1)


def _sha256(value: Any, field_name: str = "sha256") -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ModelValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _freeze_json(value: Any, *, path: str = "payload") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelValidationError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        raw_keys = list(value)
        for raw_key in raw_keys:
            if not isinstance(raw_key, str) or not raw_key or "\x00" in raw_key:
                raise ModelValidationError(f"{path} contains an invalid object key")
        for raw_key in sorted(raw_keys):
            _validate_credential_metadata(
                _normalize_key(raw_key),
                value[raw_key],
                path=f"{path}.{raw_key}",
            )
            frozen[raw_key] = _freeze_json(value[raw_key], path=f"{path}.{raw_key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} contains a non-JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _record_data(
    value: Mapping[str, Any],
    *,
    schema: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("record must be an object")
    raw = dict(value)
    if raw.get("schema") != schema:
        raise ModelValidationError(f"expected schema {schema}")
    allowed = required | optional | {"schema"}
    missing = required - raw.keys()
    unknown = raw.keys() - allowed
    if missing:
        raise ModelValidationError(f"missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ModelValidationError(f"unknown fields: {', '.join(sorted(unknown))}")
    return raw


def _record_tuple(values: Any, record_type: type[Any], field_name: str) -> tuple[Any, ...]:
    if values is None:
        return ()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be an array")
    return tuple(
        item if isinstance(item, record_type) else record_type.from_dict(item)
        for item in values
    )


@dataclass(frozen=True)
class ResourceRef:
    SCHEMA: ClassVar[str] = "unchain.resource_ref.v1"

    kind: str
    resource_id: str
    revision: int
    fragment: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _required_text(self.kind, "kind", identifier=True))
        object.__setattr__(
            self,
            "resource_id",
            _required_text(self.resource_id, "resource_id", identifier=True),
        )
        object.__setattr__(self, "revision", _positive_revision(self.revision))
        fragment = _optional_text(self.fragment, "fragment", maximum=512)
        if (
            fragment.startswith("/")
            or "\\" in fragment
            or any(segment in (".", "..") for segment in fragment.split("/"))
        ):
            raise ModelValidationError("fragment must be reference-local")
        object.__setattr__(self, "fragment", fragment)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind,
            "id": self.resource_id,
            "revision": self.revision,
            "fragment": self.fragment,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResourceRef:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"kind", "id", "revision"}),
            optional=frozenset({"fragment"}),
        )
        return cls(
            kind=raw["kind"],
            resource_id=raw["id"],
            revision=raw["revision"],
            fragment=raw.get("fragment", ""),
        )


@dataclass(frozen=True)
class ArtifactRef:
    SCHEMA: ClassVar[str] = "unchain.artifact_ref.v1"

    ref: ResourceRef
    media_type: str
    byte_length: int
    sha256: str
    preview: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ResourceRef):
            object.__setattr__(self, "ref", ResourceRef.from_dict(self.ref))
        media_type = _required_text(self.media_type, "media_type", maximum=255)
        if "/" not in media_type:
            raise ModelValidationError("media_type must be a MIME type")
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "byte_length", _bounded_int(self.byte_length, "byte_length"))
        object.__setattr__(self, "sha256", _sha256(self.sha256))
        object.__setattr__(self, "preview", _optional_text(self.preview, "preview", maximum=4096))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "ref": self.ref.to_dict(),
            "media_type": self.media_type,
            "byte_length": self.byte_length,
            "sha256": self.sha256,
            "preview": self.preview,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactRef:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"ref", "media_type", "byte_length", "sha256"}),
            optional=frozenset({"preview"}),
        )
        return cls(
            ref=ResourceRef.from_dict(raw["ref"]),
            media_type=raw["media_type"],
            byte_length=raw["byte_length"],
            sha256=raw["sha256"],
            preview=raw.get("preview", ""),
        )


@dataclass(frozen=True)
class EventCursor:
    SCHEMA: ClassVar[str] = "unchain.event_cursor.v1"

    store_seq: int
    event_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "store_seq", _bounded_int(self.store_seq, "store_seq"))
        object.__setattr__(
            self, "event_id", _required_text(self.event_id, "event_id", identifier=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "store_seq": self.store_seq, "event_id": self.event_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EventCursor:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"store_seq", "event_id"}),
        )
        return cls(store_seq=raw["store_seq"], event_id=raw["event_id"])


@dataclass(frozen=True)
class EventRange:
    SCHEMA: ClassVar[str] = "unchain.event_range.v1"

    start: EventCursor
    end: EventCursor

    def __post_init__(self) -> None:
        if not isinstance(self.start, EventCursor):
            object.__setattr__(self, "start", EventCursor.from_dict(self.start))
        if not isinstance(self.end, EventCursor):
            object.__setattr__(self, "end", EventCursor.from_dict(self.end))
        if self.start.store_seq > self.end.store_seq:
            raise ModelValidationError("event range start must not follow end")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.SCHEMA, "start": self.start.to_dict(), "end": self.end.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EventRange:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"start", "end"}),
        )
        return cls(EventCursor.from_dict(raw["start"]), EventCursor.from_dict(raw["end"]))


@dataclass(frozen=True)
class GenerationRef:
    SCHEMA: ClassVar[str] = "unchain.generation_ref.v1"

    execution_id: str
    generation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_id",
            _required_text(self.execution_id, "execution_id", identifier=True),
        )
        object.__setattr__(
            self,
            "generation_id",
            _required_text(self.generation_id, "generation_id", identifier=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "execution_id": self.execution_id,
            "generation_id": self.generation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GenerationRef:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"execution_id", "generation_id"}),
        )
        return cls(raw["execution_id"], raw["generation_id"])


@dataclass(frozen=True)
class AttemptRef:
    SCHEMA: ClassVar[str] = "unchain.attempt_ref.v1"

    generation: GenerationRef
    attempt_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.generation, GenerationRef):
            object.__setattr__(self, "generation", GenerationRef.from_dict(self.generation))
        object.__setattr__(
            self, "attempt_id", _required_text(self.attempt_id, "attempt_id", identifier=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "generation": self.generation.to_dict(),
            "attempt_id": self.attempt_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AttemptRef:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"generation", "attempt_id"}),
        )
        return cls(GenerationRef.from_dict(raw["generation"]), raw["attempt_id"])


@dataclass(frozen=True)
class OperationRef:
    SCHEMA: ClassVar[str] = "unchain.operation_ref.v1"

    operation_id: str
    payload_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, "operation_id", identifier=True),
        )
        object.__setattr__(self, "payload_sha256", _sha256(self.payload_sha256, "payload_sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "operation_id": self.operation_id,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OperationRef:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"operation_id", "payload_sha256"}),
        )
        return cls(raw["operation_id"], raw["payload_sha256"])


@dataclass(frozen=True)
class PendingArtifact:
    """One content-addressed artifact claimed in the same transaction as an event.

    The artifact keeps ``put()`` semantics (``logical_kind="artifact"``,
    ``logical_key=operation.operation_id``, revision 1) but is only written
    once the journal has re-validated the event inside its own write
    transaction.
    """

    content: bytes
    media_type: str
    operation: OperationRef
    preview: str = ""

    def __post_init__(self) -> None:
        if type(self.content) is not bytes:
            raise TypeError("artifact content must be exact bytes")
        object.__setattr__(
            self,
            "media_type",
            _required_text(self.media_type, "media_type", maximum=255),
        )
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(
                self, "operation", OperationRef.from_dict(self.operation)
            )
        if not isinstance(self.preview, str):
            raise TypeError("artifact preview must be text")


@dataclass(frozen=True)
class JournalAppendRequest:
    SCHEMA: ClassVar[str] = "unchain.journal_append_request.v1"

    event_id: str
    event_type: str
    attempt: AttemptRef
    operation: OperationRef
    payload: Mapping[str, Any] = field(default_factory=dict)
    resource_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "event_id", _required_text(self.event_id, "event_id", identifier=True)
        )
        object.__setattr__(
            self, "event_type", _required_text(self.event_type, "event_type", identifier=True)
        )
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(self, "attempt", AttemptRef.from_dict(self.attempt))
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(self, "operation", OperationRef.from_dict(self.operation))
        frozen_payload = _freeze_json(self.payload)
        if not isinstance(frozen_payload, Mapping):
            raise TypeError("payload must be an object")
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(
            self,
            "resource_refs",
            _record_tuple(self.resource_refs, ResourceRef, "resource_refs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "attempt": self.attempt.to_dict(),
            "operation": self.operation.to_dict(),
            "payload": _thaw_json(self.payload),
            "resource_refs": [item.to_dict() for item in self.resource_refs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JournalAppendRequest:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset(
                {"event_id", "event_type", "attempt", "operation", "payload"}
            ),
            optional=frozenset({"resource_refs"}),
        )
        return cls(
            event_id=raw["event_id"],
            event_type=raw["event_type"],
            attempt=AttemptRef.from_dict(raw["attempt"]),
            operation=OperationRef.from_dict(raw["operation"]),
            payload=raw["payload"],
            resource_refs=_record_tuple(raw.get("resource_refs", ()), ResourceRef, "resource_refs"),
        )


@dataclass(frozen=True)
class JournalEvent:
    SCHEMA: ClassVar[str] = "unchain.journal_event.v1"

    event_id: str
    event_type: str
    attempt: AttemptRef
    operation: OperationRef
    store_seq: int
    payload: Mapping[str, Any] = field(default_factory=dict)
    resource_refs: tuple[ResourceRef, ...] = ()

    def __post_init__(self) -> None:
        request = JournalAppendRequest(
            event_id=self.event_id,
            event_type=self.event_type,
            attempt=self.attempt,
            operation=self.operation,
            payload=self.payload,
            resource_refs=self.resource_refs,
        )
        object.__setattr__(self, "event_id", request.event_id)
        object.__setattr__(self, "event_type", request.event_type)
        object.__setattr__(self, "attempt", request.attempt)
        object.__setattr__(self, "operation", request.operation)
        object.__setattr__(self, "payload", request.payload)
        object.__setattr__(self, "resource_refs", request.resource_refs)
        object.__setattr__(
            self,
            "store_seq",
            _bounded_int(self.store_seq, "store_seq", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "attempt": self.attempt.to_dict(),
            "operation": self.operation.to_dict(),
            "store_seq": self.store_seq,
            "payload": _thaw_json(self.payload),
            "resource_refs": [item.to_dict() for item in self.resource_refs],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JournalEvent:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset(
                {"event_id", "event_type", "attempt", "operation", "store_seq", "payload"}
            ),
            optional=frozenset({"resource_refs"}),
        )
        return cls(
            event_id=raw["event_id"],
            event_type=raw["event_type"],
            attempt=AttemptRef.from_dict(raw["attempt"]),
            operation=OperationRef.from_dict(raw["operation"]),
            store_seq=raw["store_seq"],
            payload=raw["payload"],
            resource_refs=_record_tuple(raw.get("resource_refs", ()), ResourceRef, "resource_refs"),
        )


@dataclass(frozen=True)
class ToolExecutionReceiptLookup:
    """One atomic, exhaustive lookup for a single tool execution subject."""

    SCHEMA: ClassVar[str] = "unchain.tool_execution_receipt_lookup.v1"

    attempt: AttemptRef
    call_id: str
    events: tuple[JournalEvent, ...] = ()
    overflow: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(
                self,
                "attempt",
                AttemptRef.from_dict(self.attempt),
            )
        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )
        events = _record_tuple(self.events, JournalEvent, "events")
        if len(events) > MAX_TOOL_EXECUTION_RECEIPTS:
            raise ModelValidationError(
                "tool execution receipt lookup may contain at most four events"
            )
        if not isinstance(self.overflow, bool):
            raise TypeError("overflow must be a boolean")

        previous_store_seq = 0
        event_ids: set[str] = set()
        operation_ids: set[str] = set()
        cursors: set[tuple[int, str]] = set()
        for event in events:
            if event.attempt != self.attempt:
                raise ModelValidationError(
                    "tool execution receipt belongs to a foreign attempt"
                )
            if event.event_type not in TOOL_EXECUTION_RECEIPT_TYPES:
                raise ModelValidationError(
                    "tool execution receipt has an unsupported event type"
                )
            if event.payload.get("call_id") != self.call_id:
                raise ModelValidationError(
                    "tool execution receipt call identity changed"
                )
            if event.store_seq <= previous_store_seq:
                raise ModelValidationError(
                    "tool execution receipts must be strictly ordered"
                )
            previous_store_seq = event.store_seq
            cursor = (event.store_seq, event.event_id)
            if (
                event.event_id in event_ids
                or event.operation.operation_id in operation_ids
                or cursor in cursors
            ):
                raise ModelValidationError(
                    "tool execution receipt identity is duplicated"
                )
            event_ids.add(event.event_id)
            operation_ids.add(event.operation.operation_id)
            cursors.add(cursor)
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "attempt": self.attempt.to_dict(),
            "call_id": self.call_id,
            "events": [event.to_dict() for event in self.events],
            "overflow": self.overflow,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> ToolExecutionReceiptLookup:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"attempt", "call_id", "events", "overflow"}),
        )
        return cls(
            attempt=AttemptRef.from_dict(raw["attempt"]),
            call_id=raw["call_id"],
            events=_record_tuple(raw["events"], JournalEvent, "events"),
            overflow=raw["overflow"],
        )


@dataclass(frozen=True)
class JournalAppendResult:
    SCHEMA: ClassVar[str] = "unchain.journal_append_result.v1"

    event: JournalEvent
    cursor: EventCursor
    duplicate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.event, JournalEvent):
            object.__setattr__(self, "event", JournalEvent.from_dict(self.event))
        if not isinstance(self.cursor, EventCursor):
            object.__setattr__(self, "cursor", EventCursor.from_dict(self.cursor))
        if (
            self.event.event_id != self.cursor.event_id
            or self.event.store_seq != self.cursor.store_seq
        ):
            raise ModelValidationError("append result cursor must identify its persisted event")
        if not isinstance(self.duplicate, bool):
            raise TypeError("duplicate must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "event": self.event.to_dict(),
            "cursor": self.cursor.to_dict(),
            "duplicate": self.duplicate,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JournalAppendResult:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"event", "cursor", "duplicate"}),
        )
        return cls(
            JournalEvent.from_dict(raw["event"]),
            EventCursor.from_dict(raw["cursor"]),
            raw["duplicate"],
        )


@dataclass(frozen=True)
class JournalPage:
    SCHEMA: ClassVar[str] = "unchain.journal_page.v1"

    events: tuple[JournalEvent, ...] = ()
    next_cursor: EventCursor | None = None
    has_more: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", _record_tuple(self.events, JournalEvent, "events"))
        if self.next_cursor is not None and not isinstance(self.next_cursor, EventCursor):
            object.__setattr__(self, "next_cursor", EventCursor.from_dict(self.next_cursor))
        if not isinstance(self.has_more, bool):
            raise TypeError("has_more must be a boolean")
        if any(
            earlier.store_seq >= later.store_seq
            for earlier, later in zip(self.events, self.events[1:])
        ):
            raise ModelValidationError("journal page event sequences must be strictly increasing")
        if len({event.event_id for event in self.events}) != len(self.events):
            raise ModelValidationError("journal page event ids must be unique")
        if len(
            {event.attempt.generation.execution_id for event in self.events}
        ) > 1:
            raise ModelValidationError("journal page events must belong to one execution")
        if self.events:
            final_event = self.events[-1]
            if self.next_cursor is None or (
                self.next_cursor.store_seq != final_event.store_seq
                or self.next_cursor.event_id != final_event.event_id
            ):
                raise ModelValidationError("next_cursor must identify the final event")
        elif self.has_more:
            raise ModelValidationError("an empty journal page cannot have more events")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "events": [event.to_dict() for event in self.events],
            "next_cursor": self.next_cursor.to_dict() if self.next_cursor else None,
            "has_more": self.has_more,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> JournalPage:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"events", "next_cursor", "has_more"}),
        )
        return cls(
            events=_record_tuple(raw["events"], JournalEvent, "events"),
            next_cursor=(
                EventCursor.from_dict(raw["next_cursor"])
                if raw["next_cursor"] is not None
                else None
            ),
            has_more=raw["has_more"],
        )


__all__ = [
    "ArtifactRef",
    "AttemptRef",
    "ContextBuildStatus",
    "EventCursor",
    "EventRange",
    "GenerationRef",
    "JournalAppendResult",
    "JournalAppendRequest",
    "JournalEvent",
    "JournalPage",
    "ModelValidationError",
    "OperationRef",
    "ResourceRef",
]
