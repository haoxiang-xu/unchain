from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from unchain.journal import ResourceRef
from unchain.journal.models import _required_text, _sha256


MAX_LIST_RESULTS = 200
MAX_SEARCH_RESULTS = 100
MAX_PAGE_READ_BYTES = 32 * 1024
MAX_CONTEXT_CONTENT_PAGE_READ_BYTES = 8 * 1024
MAX_CONTEXT_CONTENT_SERIALIZED_BYTES = 12 * 1024
MAX_FULL_READ_BYTES = 128 * 1024
MAX_WRITE_BYTES = 256 * 1024
MAX_HISTORY_REVISIONS = 50
MAX_CHECKPOINT_EVENT_PAGE_SIZE = 50
MAX_TASK_STATE_ITEMS = 200
MAX_TASK_STATE_ITEM_CHARS = 4096


class MemoryToolkitError(ValueError):
    """A stable, model-safe Memory V2 toolkit boundary error."""


class ReferencePurpose(StrEnum):
    """Why an external reference is being decoded by the bound host codec."""

    MEMORY = "memory"
    CONTEXT_CONTENT = "context_content"
    CHECKPOINT = "checkpoint"
    SOURCE = "source"
    CANDIDATE = "candidate"
    TASK_EVENT = "task_event"
    ARTIFACT_OR_MEMORY = "artifact_or_memory"


@dataclass(frozen=True)
class MemoryToolkitRunBinding:
    """Opaque host binding used for deterministic mutation identities."""

    binding_id: str
    session_id: str
    attempt_id: str
    run_id: str

    def __post_init__(self) -> None:
        for field_name in ("binding_id", "session_id", "attempt_id", "run_id"):
            object.__setattr__(
                self,
                field_name,
                _required_text(
                    getattr(self, field_name),
                    field_name,
                    maximum=512,
                    identifier=True,
                ),
            )


@dataclass(frozen=True)
class MemoryToolContentPage:
    """One structured page returned by a scope-bound read capability."""

    ref: ResourceRef
    media_type: str
    data: bytes
    offset: int
    total_bytes: int
    sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ResourceRef):
            object.__setattr__(self, "ref", ResourceRef.from_dict(self.ref))
        media_type = _required_text(self.media_type, "media_type", maximum=255)
        if "/" not in media_type:
            raise MemoryToolkitError("media_type must be a MIME type")
        object.__setattr__(self, "media_type", media_type)
        if not isinstance(self.data, bytes):
            raise TypeError("data must be bytes")
        for field_name in ("offset", "total_bytes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MemoryToolkitError(f"{field_name} must be a non-negative integer")
        if (
            self.offset > self.total_bytes
            or self.offset + len(self.data) > self.total_bytes
        ):
            raise MemoryToolkitError("content page exceeds the declared byte range")
        if self.sha256:
            object.__setattr__(self, "sha256", _sha256(self.sha256))

    @property
    def next_offset(self) -> int | None:
        candidate = self.offset + len(self.data)
        return candidate if candidate < self.total_bytes else None

    @property
    def truncated(self) -> bool:
        return self.next_offset is not None


@dataclass(frozen=True)
class CandidateProposalRequest:
    path: str
    description: str
    kind: str
    content: bytes | None
    media_type: str
    url: str
    source_refs: tuple[ResourceRef, ...]
    rationale: str
    confidence: float | None
    sensitivity: str
    operation_id: str


@dataclass(frozen=True)
class MemoryUpsertRequest:
    path: str
    description: str
    expected_space_revision: int
    entry_ref: ResourceRef | None
    kind: str
    content: bytes | None
    media_type: str
    url: str
    source_refs: tuple[ResourceRef, ...]
    operation_id: str


@dataclass(frozen=True)
class TaskStateUpdateRequest:
    expected_revision: int
    patch: Mapping[str, Any]
    source_refs: tuple[ResourceRef, ...]
    operation_id: str


def require_sha256(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise MemoryToolkitError("stored content hash is unavailable")
    return value


__all__ = [
    "CandidateProposalRequest",
    "MAX_CHECKPOINT_EVENT_PAGE_SIZE",
    "MAX_CONTEXT_CONTENT_PAGE_READ_BYTES",
    "MAX_CONTEXT_CONTENT_SERIALIZED_BYTES",
    "MAX_FULL_READ_BYTES",
    "MAX_HISTORY_REVISIONS",
    "MAX_LIST_RESULTS",
    "MAX_PAGE_READ_BYTES",
    "MAX_SEARCH_RESULTS",
    "MAX_TASK_STATE_ITEM_CHARS",
    "MAX_TASK_STATE_ITEMS",
    "MAX_WRITE_BYTES",
    "MemoryToolContentPage",
    "MemoryToolkitError",
    "MemoryToolkitRunBinding",
    "MemoryUpsertRequest",
    "ReferencePurpose",
    "TaskStateUpdateRequest",
    "require_sha256",
]
