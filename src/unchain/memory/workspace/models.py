from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar
from urllib.parse import unquote, urlparse

from unchain.journal.models import (
    ModelValidationError,
    ResourceRef,
    _bounded_int,
    _freeze_json,
    _optional_text,
    _record_data,
    _record_tuple,
    _required_text,
    _sha256,
    _thaw_json,
)
from unchain.memory.curator.models import (
    CandidateStatus,
    ConsolidationJobStatus as JobStatus,
)


_HOST_ROOTS = frozenset(
    {
        "applications",
        "bin",
        "cores",
        "dev",
        "documents and settings",
        "etc",
        "home",
        "library",
        "mnt",
        "network",
        "opt",
        "private",
        "program files",
        "program files (x86)",
        "programdata",
        "proc",
        "root",
        "run",
        "sbin",
        "sys",
        "system",
        "tmp",
        "users",
        "usr",
        "var",
        "volumes",
        "windows",
    }
)
_WINDOWS_DRIVE_RE = re.compile(r"^/[A-Za-z]:(?:/|$)")
_URL_CREDENTIAL_SEGMENT_RE = re.compile(
    r"^(?:auth|authorization|bearer|code|cookie|credential|jwt|key|passwd|"
    r"password|sas|secret|sig|signature|token)s?[0-9]*$"
)
_URL_CREDENTIAL_COMPOUNDS = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "apisecret",
        "authtoken",
        "bearertoken",
        "clientsecret",
        "credential",
        "credentials",
        "encryptionkey",
        "githubtoken",
        "idtoken",
        "oauthsecret",
        "oauthtoken",
        "password",
        "passwd",
        "privatekey",
        "refreshtoken",
        "secret",
        "secretkey",
        "sessioncookie",
        "sessiontoken",
        "signingkey",
        "webhooksecret",
    }
)


def _fully_unquote_url_component(value: str) -> str:
    decoded = value
    for _ in range(16):
        next_value = unquote(decoded)
        if next_value == decoded:
            return decoded
        decoded = next_value
    if unquote(decoded) != decoded:
        raise ModelValidationError("link URL contains credential-like nested encoding")
    return decoded


def _url_key_is_sensitive(value: str) -> bool:
    decoded = _fully_unquote_url_component(value)
    compatible = unicodedata.normalize("NFKC", decoded)
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", compatible)
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.casefold()).strip("_")
    segments = tuple(segment for segment in normalized.split("_") if segment)
    collapsed = "".join(segments)
    if collapsed in _URL_CREDENTIAL_COMPOUNDS:
        return True
    if any(_URL_CREDENTIAL_SEGMENT_RE.fullmatch(segment) for segment in segments):
        return True
    return any(
        left in {"access", "api", "encryption", "private", "signing"}
        and re.fullmatch(r"keys?[0-9]*", right) is not None
        for left, right in zip(segments, segments[1:])
    )


def _url_component_has_sensitive_key(value: str) -> bool:
    decoded = _fully_unquote_url_component(value)
    for field in re.split(r"[&;?]", decoded):
        key = field.split("=", 1)[0]
        if _url_key_is_sensitive(key):
            return True
    return False


def _url_path_has_embedded_credential(parsed: Any) -> bool:
    try:
        unicode_host = unicodedata.normalize("NFKC", parsed.hostname or "")
        host = unicode_host.encode("idna").decode("ascii").casefold().rstrip(".")
    except (UnicodeError, ValueError):
        return True
    decoded_path = _fully_unquote_url_component(parsed.path)
    path = unicodedata.normalize("NFKC", decoded_path).casefold()
    if host == "hooks.slack.com" and path.startswith("/services/"):
        return True
    if host in {"discord.com", "discordapp.com"} and re.match(
        r"^/api(?:/v[0-9]+)?/webhooks/[^/]+/[^/]+",
        path,
    ):
        return True
    if host.endswith(".webhook.office.com") or (
        host == "outlook.office.com" and "/webhook/" in path
    ):
        return True
    if host == "maker.ifttt.com" and "/with/key/" in path:
        return True
    if host == "api.telegram.org" and re.match(r"^/bot[^/]+/", path):
        return True
    segments = tuple(segment for segment in path.split("/") if segment)
    if any(
        "=" in segment and _url_component_has_sensitive_key(segment)
        for segment in segments
    ):
        return True
    for marker, candidate in zip(segments, segments[1:]):
        if candidate and _url_key_is_sensitive(marker):
            return True
    return False


def canonical_memory_link_url(value: Any) -> str:
    """Return validated HTTP(S) link text without accepting credential material."""

    link_url = _required_text(value, "link_url", maximum=8192)
    parsed = urlparse(link_url)
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
        raise ModelValidationError("link entries require an HTTP(S) URL only")
    if (
        parsed.username is not None
        or parsed.password is not None
        or _url_component_has_sensitive_key(parsed.query)
        or _url_component_has_sensitive_key(parsed.fragment)
        or _url_path_has_embedded_credential(parsed)
    ):
        raise ModelValidationError("link URLs cannot contain credentials")
    return link_url


def _virtual_path(value: Any, field_name: str = "path") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    if any(
        ord(character) < 32
        or ord(character) == 127
        or unicodedata.category(character).startswith("C")
        or unicodedata.category(character) in {"Zl", "Zp"}
        for character in value
    ):
        raise ModelValidationError(f"{field_name} contains control characters")
    normalized = unicodedata.normalize("NFKC", value)
    if normalized != normalized.strip() or unquote(normalized) != normalized:
        raise ModelValidationError(f"{field_name} is not canonical")
    lower = normalized.casefold()
    if (
        not normalized.startswith("/")
        or normalized.startswith("//")
        or normalized.startswith("\\")
        or "\\" in normalized
        or lower.startswith("file:")
        or _WINDOWS_DRIVE_RE.match(normalized)
        or "\x00" in normalized
    ):
        raise ModelValidationError(f"{field_name} must be a canonical virtual path")
    segments = normalized.split("/")[1:]
    if any(segment in ("", ".", "..") for segment in segments):
        if normalized != "/":
            raise ModelValidationError(f"{field_name} contains an invalid segment")
    if segments and segments[0].casefold() in _HOST_ROOTS:
        raise ModelValidationError(f"{field_name} looks like a host filesystem path")
    if segments and (segments[0].casefold() == "file:" or segments[0].endswith(":")):
        raise ModelValidationError(f"{field_name} looks like a host filesystem path")
    if len(normalized) > 2048:
        raise ModelValidationError(f"{field_name} is too long")
    return normalized


def canonical_virtual_path(value: Any, field_name: str = "path") -> str:
    """Validate and return one provider-neutral virtual workspace path."""

    return _virtual_path(value, field_name)


def _tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError("tags must be an array")
    normalized = tuple(
        _required_text(item, f"tags[{index}]", maximum=64, identifier=True)
        for index, item in enumerate(value)
    )
    if len(set(normalized)) != len(normalized):
        raise ModelValidationError("tags must be unique")
    return normalized


def canonical_memory_tags(value: Any) -> tuple[str, ...]:
    """Normalize a provider-neutral memory tag collection."""

    return _tags(value)


class MemoryEntryKind(StrEnum):
    FOLDER = "folder"
    MARKDOWN = "markdown"
    IMAGE = "image"
    LINK = "link"


class PromotionStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class MemorySpace:
    SCHEMA: ClassVar[str] = "unchain.memory_space.v1"

    space_id: str
    namespace: str
    name: str
    description: str
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "space_id", _required_text(self.space_id, "space_id", identifier=True)
        )
        object.__setattr__(
            self, "namespace", _required_text(self.namespace, "namespace", identifier=True)
        )
        object.__setattr__(self, "name", _required_text(self.name, "name", maximum=256))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "description", maximum=4096),
        )
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "space_id": self.space_id,
            "namespace": self.namespace,
            "name": self.name,
            "description": self.description,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemorySpace:
        fields = frozenset({"space_id", "namespace", "name", "description", "revision"})
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(**{field_name: raw[field_name] for field_name in fields})


@dataclass(frozen=True)
class MemoryEntry:
    SCHEMA: ClassVar[str] = "unchain.memory_entry.v1"

    entry_id: str
    space_id: str
    path: str
    name: str
    description: str
    kind: MemoryEntryKind
    revision: int
    updated_seq: int = 0
    content_ref: ResourceRef | None = None
    source_refs: tuple[ResourceRef, ...] = ()
    tags: tuple[str, ...] = ()
    media_type: str = ""
    link_url: str = ""
    deleted: bool = False

    def __post_init__(self) -> None:
        for field_name in ("entry_id", "space_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        object.__setattr__(self, "path", _virtual_path(self.path))
        object.__setattr__(self, "name", _required_text(self.name, "name", maximum=256))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "description", maximum=8192),
        )
        try:
            object.__setattr__(self, "kind", MemoryEntryKind(self.kind))
        except ValueError as exc:
            raise ModelValidationError("invalid memory entry kind") from exc
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))
        object.__setattr__(self, "updated_seq", _bounded_int(self.updated_seq, "updated_seq"))
        if self.content_ref is not None and not isinstance(self.content_ref, ResourceRef):
            object.__setattr__(self, "content_ref", ResourceRef.from_dict(self.content_ref))
        object.__setattr__(
            self, "source_refs", _record_tuple(self.source_refs, ResourceRef, "source_refs")
        )
        object.__setattr__(self, "tags", _tags(self.tags))
        media_type = _optional_text(self.media_type, "media_type", maximum=255)
        link_url = _optional_text(self.link_url, "link_url", maximum=8192)
        if media_type and "/" not in media_type:
            raise ModelValidationError("media_type must be a MIME type")
        if self.kind is MemoryEntryKind.FOLDER:
            if self.content_ref is not None or media_type or link_url:
                raise ModelValidationError("folder entries cannot carry content metadata")
        elif self.kind is MemoryEntryKind.MARKDOWN:
            if link_url:
                raise ModelValidationError("markdown entries cannot carry a link URL")
            if media_type and media_type not in {"text/markdown", "text/plain"}:
                raise ModelValidationError("markdown entries require a text MIME type")
        elif self.kind is MemoryEntryKind.IMAGE:
            if link_url:
                raise ModelValidationError("image entries cannot carry a link URL")
            if media_type and not media_type.startswith("image/"):
                raise ModelValidationError("image entries require an image MIME type")
        elif self.kind is MemoryEntryKind.LINK:
            if self.content_ref is not None or media_type:
                raise ModelValidationError("link entries require an HTTP(S) URL only")
            link_url = canonical_memory_link_url(link_url)
        object.__setattr__(self, "media_type", media_type)
        object.__setattr__(self, "link_url", link_url)
        if not isinstance(self.deleted, bool):
            raise TypeError("deleted must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entry_id": self.entry_id,
            "space_id": self.space_id,
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "revision": self.revision,
            "updated_seq": self.updated_seq,
            "content_ref": self.content_ref.to_dict() if self.content_ref else None,
            "source_refs": [item.to_dict() for item in self.source_refs],
            "tags": list(self.tags),
            "media_type": self.media_type,
            "link_url": self.link_url,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryEntry:
        required = frozenset(
            {
                "entry_id",
                "space_id",
                "path",
                "name",
                "description",
                "kind",
                "revision",
                "content_ref",
                "source_refs",
                "tags",
                "deleted",
            }
        )
        optional = frozenset({"media_type", "link_url", "updated_seq"})
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=required,
            optional=optional,
        )
        return cls(
            entry_id=raw["entry_id"],
            space_id=raw["space_id"],
            path=raw["path"],
            name=raw["name"],
            description=raw["description"],
            kind=raw["kind"],
            revision=raw["revision"],
            updated_seq=raw.get("updated_seq", 0),
            content_ref=(
                ResourceRef.from_dict(raw["content_ref"])
                if raw["content_ref"] is not None
                else None
            ),
            source_refs=_record_tuple(raw["source_refs"], ResourceRef, "source_refs"),
            tags=_tags(raw["tags"]),
            media_type=raw.get("media_type", ""),
            link_url=raw.get("link_url", ""),
            deleted=raw["deleted"],
        )


@dataclass(frozen=True)
class MemoryEntryPage:
    SCHEMA: ClassVar[str] = "unchain.memory_entry_page.v1"

    entries: tuple[MemoryEntry, ...] = ()
    next_cursor: str | None = None
    has_more: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entries",
            _record_tuple(self.entries, MemoryEntry, "entries"),
        )
        if self.next_cursor is not None:
            object.__setattr__(
                self,
                "next_cursor",
                _required_text(self.next_cursor, "next_cursor", identifier=True),
            )
        if not isinstance(self.has_more, bool):
            raise TypeError("has_more must be a boolean")
        if self.has_more != (self.next_cursor is not None):
            raise ModelValidationError("next_cursor must be present exactly when has_more is true")
        if self.has_more and not self.entries:
            raise ModelValidationError("a continuation page must contain at least one entry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entries": [entry.to_dict() for entry in self.entries],
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryEntryPage:
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=frozenset({"entries", "next_cursor", "has_more"}),
        )
        return cls(
            entries=_record_tuple(raw["entries"], MemoryEntry, "entries"),
            next_cursor=raw["next_cursor"],
            has_more=raw["has_more"],
        )


@dataclass(frozen=True)
class MemoryChildEntry:
    SCHEMA: ClassVar[str] = "unchain.memory_child_entry.v1"

    entry: MemoryEntry
    has_children: bool = False
    orphaned: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.entry, MemoryEntry):
            object.__setattr__(self, "entry", MemoryEntry.from_dict(self.entry))
        if not isinstance(self.has_children, bool):
            raise TypeError("has_children must be a boolean")
        if not isinstance(self.orphaned, bool):
            raise TypeError("orphaned must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entry": self.entry.to_dict(),
            "has_children": self.has_children,
            "orphaned": self.orphaned,
        }


@dataclass(frozen=True)
class MemoryChildPage:
    SCHEMA: ClassVar[str] = "unchain.memory_child_page.v1"

    space_id: str
    space_revision: int
    parent_path: str
    order_version: str
    entries: tuple[MemoryChildEntry, ...] = ()
    next_cursor: str | None = None
    has_more: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "space_id", _required_text(self.space_id, "space_id", identifier=True)
        )
        object.__setattr__(
            self,
            "space_revision",
            _bounded_int(self.space_revision, "space_revision", minimum=1),
        )
        object.__setattr__(self, "parent_path", _virtual_path(self.parent_path, "parent_path"))
        object.__setattr__(
            self,
            "order_version",
            _required_text(self.order_version, "order_version", maximum=128, identifier=True),
        )
        object.__setattr__(
            self,
            "entries",
            _record_tuple(self.entries, MemoryChildEntry, "entries"),
        )
        if self.next_cursor is not None:
            object.__setattr__(
                self,
                "next_cursor",
                _required_text(self.next_cursor, "next_cursor", maximum=8192),
            )
        if not isinstance(self.has_more, bool):
            raise TypeError("has_more must be a boolean")
        if self.has_more != (self.next_cursor is not None):
            raise ModelValidationError(
                "next_cursor must be present exactly when has_more is true"
            )
        if self.has_more and not self.entries:
            raise ModelValidationError(
                "a continuation page must contain at least one child"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "space_id": self.space_id,
            "space_revision": self.space_revision,
            "parent_path": self.parent_path,
            "order_version": self.order_version,
            "entries": [item.to_dict() for item in self.entries],
            "next_cursor": self.next_cursor,
            "has_more": self.has_more,
        }


@dataclass(frozen=True)
class VectorProjectionPoint:
    chunk_id: str
    entry_id: str
    entry_revision: int
    ordinal: int
    x: float
    y: float

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "entry_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        object.__setattr__(
            self,
            "entry_revision",
            _bounded_int(self.entry_revision, "entry_revision", minimum=1),
        )
        object.__setattr__(self, "ordinal", _bounded_int(self.ordinal, "ordinal"))
        for field_name in ("x", "y"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be a finite number")
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ModelValidationError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, normalized)


@dataclass(frozen=True)
class VectorProjectionPage:
    space_id: str
    backend_identity: str
    chunker_version: str
    corpus_epoch: int
    basis_id: str
    basis_version: int
    algorithm: str
    dimension: int
    status: str
    stale: bool
    eligible_entries: int
    indexed_entries: int
    eligible_chunks: int
    indexed_chunks: int
    projected_chunks: int
    points: tuple[VectorProjectionPoint, ...] = ()
    next_cursor: str | None = None
    has_more: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "space_id",
            "backend_identity",
            "chunker_version",
            "basis_id",
            "algorithm",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, maximum=512),
            )
        for field_name in (
            "corpus_epoch",
            "basis_version",
            "dimension",
            "eligible_entries",
            "indexed_entries",
            "eligible_chunks",
            "indexed_chunks",
            "projected_chunks",
        ):
            minimum = 1 if field_name in {"corpus_epoch", "basis_version", "dimension"} else 0
            object.__setattr__(
                self,
                field_name,
                _bounded_int(getattr(self, field_name), field_name, minimum=minimum),
            )
        if self.status not in {"complete", "partial", "disabled", "warming", "degraded"}:
            raise ModelValidationError("projection status is invalid")
        if not isinstance(self.stale, bool) or not isinstance(self.has_more, bool):
            raise TypeError("projection flags must be booleans")
        object.__setattr__(
            self,
            "points",
            _record_tuple(self.points, VectorProjectionPoint, "points"),
        )
        if self.next_cursor is not None:
            object.__setattr__(
                self,
                "next_cursor",
                _required_text(self.next_cursor, "next_cursor", maximum=8192),
            )
        if self.has_more != (self.next_cursor is not None):
            raise ModelValidationError("projection cursor and has_more disagree")


@dataclass(frozen=True)
class EntryRevision:
    SCHEMA: ClassVar[str] = "unchain.entry_revision.v1"

    entry_id: str
    revision: int
    content_ref: ResourceRef
    content_sha256: str
    byte_length: int
    source_refs: tuple[ResourceRef, ...]
    operation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entry_id", _required_text(self.entry_id, "entry_id", identifier=True)
        )
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))
        if not isinstance(self.content_ref, ResourceRef):
            object.__setattr__(self, "content_ref", ResourceRef.from_dict(self.content_ref))
        object.__setattr__(
            self, "content_sha256", _sha256(self.content_sha256, "content_sha256")
        )
        object.__setattr__(self, "byte_length", _bounded_int(self.byte_length, "byte_length"))
        object.__setattr__(
            self, "source_refs", _record_tuple(self.source_refs, ResourceRef, "source_refs")
        )
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, "operation_id", identifier=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "entry_id": self.entry_id,
            "revision": self.revision,
            "content_ref": self.content_ref.to_dict(),
            "content_sha256": self.content_sha256,
            "byte_length": self.byte_length,
            "source_refs": [item.to_dict() for item in self.source_refs],
            "operation_id": self.operation_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EntryRevision:
        fields = frozenset(
            {"entry_id", "revision", "content_ref", "content_sha256", "byte_length", "source_refs", "operation_id"}
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            entry_id=raw["entry_id"],
            revision=raw["revision"],
            content_ref=ResourceRef.from_dict(raw["content_ref"]),
            content_sha256=raw["content_sha256"],
            byte_length=raw["byte_length"],
            source_refs=_record_tuple(raw["source_refs"], ResourceRef, "source_refs"),
            operation_id=raw["operation_id"],
        )


@dataclass(frozen=True)
class MemoryLink:
    SCHEMA: ClassVar[str] = "unchain.memory_link.v1"

    link_id: str
    source_entry_ref: ResourceRef
    target_ref: ResourceRef
    relation: str
    revision: int

    def __post_init__(self) -> None:
        for field_name in ("link_id", "relation"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        if not isinstance(self.source_entry_ref, ResourceRef):
            object.__setattr__(
                self,
                "source_entry_ref",
                ResourceRef.from_dict(self.source_entry_ref),
            )
        if not isinstance(self.target_ref, ResourceRef):
            object.__setattr__(self, "target_ref", ResourceRef.from_dict(self.target_ref))
        if self.source_entry_ref.kind != "memory" or self.target_ref.kind != "memory":
            raise ModelValidationError("memory links require revisioned memory references")
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))

    @property
    def source_entry_id(self) -> str:
        return self.source_entry_ref.resource_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "link_id": self.link_id,
            "source_entry_ref": self.source_entry_ref.to_dict(),
            "target_ref": self.target_ref.to_dict(),
            "relation": self.relation,
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryLink:
        fields = frozenset(
            {"link_id", "source_entry_ref", "target_ref", "relation", "revision"}
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            raw["link_id"],
            ResourceRef.from_dict(raw["source_entry_ref"]),
            ResourceRef.from_dict(raw["target_ref"]),
            raw["relation"],
            raw["revision"],
        )


@dataclass(frozen=True)
class MemoryCandidate:
    SCHEMA: ClassVar[str] = "unchain.memory_candidate.v1"

    candidate_id: str
    path: str
    name: str
    description: str
    kind: MemoryEntryKind
    content_ref: ResourceRef
    source_refs: tuple[ResourceRef, ...]
    status: CandidateStatus
    revision: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "candidate_id",
            _required_text(self.candidate_id, "candidate_id", identifier=True),
        )
        object.__setattr__(self, "path", _virtual_path(self.path))
        object.__setattr__(self, "name", _required_text(self.name, "name", maximum=256))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "description", maximum=8192),
        )
        try:
            object.__setattr__(self, "kind", MemoryEntryKind(self.kind))
            object.__setattr__(self, "status", CandidateStatus(self.status))
        except ValueError as exc:
            raise ModelValidationError("invalid candidate kind or status") from exc
        if not isinstance(self.content_ref, ResourceRef):
            object.__setattr__(self, "content_ref", ResourceRef.from_dict(self.content_ref))
        object.__setattr__(
            self, "source_refs", _record_tuple(self.source_refs, ResourceRef, "source_refs")
        )
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))
        object.__setattr__(self, "reason", _required_text(self.reason, "reason", maximum=8192))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "candidate_id": self.candidate_id,
            "path": self.path,
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "content_ref": self.content_ref.to_dict(),
            "source_refs": [item.to_dict() for item in self.source_refs],
            "status": self.status.value,
            "revision": self.revision,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MemoryCandidate:
        fields = frozenset(
            {"candidate_id", "path", "name", "description", "kind", "content_ref", "source_refs", "status", "revision", "reason"}
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            candidate_id=raw["candidate_id"],
            path=raw["path"],
            name=raw["name"],
            description=raw["description"],
            kind=raw["kind"],
            content_ref=ResourceRef.from_dict(raw["content_ref"]),
            source_refs=_record_tuple(raw["source_refs"], ResourceRef, "source_refs"),
            status=raw["status"],
            revision=raw["revision"],
            reason=raw["reason"],
        )


@dataclass(frozen=True)
class ConsolidationJob:
    SCHEMA: ClassVar[str] = "unchain.consolidation_job.v1"

    job_id: str
    candidate_refs: tuple[ResourceRef, ...]
    status: JobStatus
    revision: int
    operation_id: str
    failure_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _required_text(self.job_id, "job_id", identifier=True))
        object.__setattr__(
            self,
            "candidate_refs",
            _record_tuple(self.candidate_refs, ResourceRef, "candidate_refs"),
        )
        if not self.candidate_refs:
            raise ModelValidationError("candidate_refs must not be empty")
        try:
            object.__setattr__(self, "status", JobStatus(self.status))
        except ValueError as exc:
            raise ModelValidationError("invalid consolidation job status") from exc
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))
        object.__setattr__(
            self,
            "operation_id",
            _required_text(self.operation_id, "operation_id", identifier=True),
        )
        object.__setattr__(
            self,
            "failure_reason",
            _optional_text(self.failure_reason, "failure_reason", maximum=8192),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "job_id": self.job_id,
            "candidate_refs": [item.to_dict() for item in self.candidate_refs],
            "status": self.status.value,
            "revision": self.revision,
            "operation_id": self.operation_id,
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConsolidationJob:
        fields = frozenset(
            {"job_id", "candidate_refs", "status", "revision", "operation_id", "failure_reason"}
        )
        raw = _record_data(value, schema=cls.SCHEMA, required=fields)
        return cls(
            job_id=raw["job_id"],
            candidate_refs=_record_tuple(raw["candidate_refs"], ResourceRef, "candidate_refs"),
            status=raw["status"],
            revision=raw["revision"],
            operation_id=raw["operation_id"],
            failure_reason=raw["failure_reason"],
        )


@dataclass(frozen=True)
class PromotionProposal:
    SCHEMA: ClassVar[str] = "unchain.promotion_proposal.v1"

    proposal_id: str
    source_entry_ref: ResourceRef
    target_namespace: str
    target_path: str
    diff: Mapping[str, Any]
    reason: str
    status: PromotionStatus
    revision: int
    source_refs: tuple[ResourceRef, ...] = ()
    target_entry_ref: ResourceRef | None = None
    applied_entry_ref: ResourceRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposal_id",
            _required_text(self.proposal_id, "proposal_id", identifier=True),
        )
        if not isinstance(self.source_entry_ref, ResourceRef):
            object.__setattr__(
                self, "source_entry_ref", ResourceRef.from_dict(self.source_entry_ref)
            )
        object.__setattr__(
            self,
            "target_namespace",
            _required_text(self.target_namespace, "target_namespace", identifier=True),
        )
        object.__setattr__(self, "target_path", _virtual_path(self.target_path, "target_path"))
        frozen_diff = _freeze_json(self.diff, path="diff")
        if not isinstance(frozen_diff, Mapping):
            raise TypeError("diff must be an object")
        object.__setattr__(self, "diff", frozen_diff)
        object.__setattr__(self, "reason", _required_text(self.reason, "reason", maximum=8192))
        try:
            object.__setattr__(self, "status", PromotionStatus(self.status))
        except ValueError as exc:
            raise ModelValidationError("invalid promotion status") from exc
        object.__setattr__(self, "revision", _bounded_int(self.revision, "revision", minimum=1))
        object.__setattr__(
            self, "source_refs", _record_tuple(self.source_refs, ResourceRef, "source_refs")
        )
        for field_name in ("target_entry_ref", "applied_entry_ref"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, ResourceRef):
                value = ResourceRef.from_dict(value)
                object.__setattr__(self, field_name, value)
            if value is not None and value.kind != "memory":
                raise ModelValidationError(f"{field_name} must be a memory reference")
        if self.status is PromotionStatus.APPLIED and self.applied_entry_ref is None:
            raise ModelValidationError("applied promotions require an applied entry reference")
        if self.status is not PromotionStatus.APPLIED and self.applied_entry_ref is not None:
            raise ModelValidationError("only applied promotions may carry an applied entry reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "proposal_id": self.proposal_id,
            "source_entry_ref": self.source_entry_ref.to_dict(),
            "target_namespace": self.target_namespace,
            "target_path": self.target_path,
            "diff": _thaw_json(self.diff),
            "reason": self.reason,
            "status": self.status.value,
            "revision": self.revision,
            "source_refs": [item.to_dict() for item in self.source_refs],
            "target_entry_ref": (
                self.target_entry_ref.to_dict() if self.target_entry_ref else None
            ),
            "applied_entry_ref": (
                self.applied_entry_ref.to_dict() if self.applied_entry_ref else None
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromotionProposal:
        fields = frozenset(
            {
                "proposal_id",
                "source_entry_ref",
                "target_namespace",
                "target_path",
                "diff",
                "reason",
                "status",
                "revision",
                "source_refs",
            }
        )
        optional = frozenset({"target_entry_ref", "applied_entry_ref"})
        raw = _record_data(
            value,
            schema=cls.SCHEMA,
            required=fields,
            optional=optional,
        )
        return cls(
            proposal_id=raw["proposal_id"],
            source_entry_ref=ResourceRef.from_dict(raw["source_entry_ref"]),
            target_namespace=raw["target_namespace"],
            target_path=raw["target_path"],
            diff=raw["diff"],
            reason=raw["reason"],
            status=raw["status"],
            revision=raw["revision"],
            source_refs=_record_tuple(raw["source_refs"], ResourceRef, "source_refs"),
            target_entry_ref=(
                ResourceRef.from_dict(raw["target_entry_ref"])
                if raw.get("target_entry_ref") is not None
                else None
            ),
            applied_entry_ref=(
                ResourceRef.from_dict(raw["applied_entry_ref"])
                if raw.get("applied_entry_ref") is not None
                else None
            ),
        )


__all__ = [
    "CandidateStatus",
    "ConsolidationJob",
    "EntryRevision",
    "JobStatus",
    "MemoryCandidate",
    "MemoryEntry",
    "MemoryEntryPage",
    "MemoryEntryKind",
    "MemoryLink",
    "MemorySpace",
    "PromotionProposal",
    "PromotionStatus",
    "canonical_memory_tags",
    "canonical_memory_link_url",
    "canonical_virtual_path",
]
