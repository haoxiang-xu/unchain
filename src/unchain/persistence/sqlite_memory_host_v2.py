"""Production SQLite host adapter for Memory V2 consolidation.

This module composes the durable curator repository with one bound chat
workspace.  It deliberately exposes only the consolidation role: candidate
bytes can be read, a frozen candidate can create a new chat-memory entry, and
an existing-path conflict can produce a durable server-computed review.  It
does not run a model, promote long-term memory, or mutate pinned task state.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from unchain.memory.module import (
    MemoryAttachment,
    MemoryAttachmentRequest,
    MemoryCompletionFactory,
)
from unchain.journal import ResourceRef
from unchain.journal.models import _required_text
from unchain.memory.curator.models import (
    CandidateStatus,
    ConsolidationJob,
    ConsolidationJobStatus,
    CuratorLeaseFence,
    FrozenCandidateSnapshot,
)
from unchain.memory.curator.ports import (
    BoundCurationRepository,
    BoundCuratorMutationGuard,
)
from unchain.memory.toolkit.capabilities import (
    BoundContextMemoryCapability,
    BoundExternalReferenceCodec,
    BoundMemoryReadCapability,
    ConsolidationMemoryToolkitCapabilities,
    NormalMemoryToolkitCapabilities,
)
from unchain.memory.toolkit.models import (
    MAX_PAGE_READ_BYTES,
    MemoryToolContentPage,
    MemoryToolkitRunBinding,
)
from unchain.memory.toolkit.services import WorkspaceMemoryCapability
from unchain.memory.workspace.models import MemoryEntry, MemoryEntryKind
from unchain.memory.workspace.paths import canonical_entry_path, virtual_name
from unchain.memory.workspace.ports import (
    RepositoryConflictError,
    RepositoryNotFoundError,
    WorkspaceRepositoryError,
)
from unchain.memory.workspace.service import MemoryWorkspaceService


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_REVIEW_SCHEMA = "unchain.memory_review_diff.v1"
_SCHEMA_VERSION = 1


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _path_identity(path: str) -> str:
    return unicodedata.normalize("NFKC", canonical_entry_path(path)).casefold()


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SQLiteMemoryHostV2Error(f"{field_name}_invalid")
    return value


def _non_negative_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SQLiteMemoryHostV2Error(f"{field_name}_invalid")
    return value


def _deduplicate_refs(
    candidates: tuple[FrozenCandidateSnapshot, ...],
) -> tuple[ResourceRef, ...]:
    result: list[ResourceRef] = []
    for candidate in candidates:
        for ref in candidate.source_refs:
            if ref not in result:
                result.append(ref)
    return tuple(result)


class SQLiteMemoryHostV2Error(RuntimeError):
    """Stable failure at the SQLite Memory V2 host boundary."""

    def __init__(self, code: str) -> None:
        normalized = ""
        if isinstance(code, str):
            normalized = re.sub(r"[^a-z0-9_:-]+", "_", code.casefold()).strip("_")
        self.code = normalized[:128] or "memory_host_error"
        super().__init__(self.code)


class SQLiteMemoryHostV2IntegrityError(SQLiteMemoryHostV2Error):
    """Durable state no longer matches the frozen consolidation effect."""


def initialize_sqlite_memory_host_v2_schema(
    *,
    database_path: str | Path,
) -> None:
    """Install the host-owned review schema in an already-owned SQLite plane.

    The schema is deliberately independent of a bound consolidation factory so
    empty-plane bootstrap can establish the same canonical closure without
    inventing a synthetic agent, workspace, or review proposal.
    """

    connection = sqlite3.connect(
        Path(database_path),
        timeout=30.0,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 30000")
    try:
        mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).casefold() != "wal":
            raise SQLiteMemoryHostV2IntegrityError("sqlite_wal_unavailable")
        connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS memory_host_v2_schema (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR IGNORE INTO memory_host_v2_schema(version) VALUES (1);

            CREATE TABLE IF NOT EXISTS memory_review_proposals (
                review_id TEXT PRIMARY KEY,
                binding_id TEXT NOT NULL,
                job_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                candidate_revision INTEGER NOT NULL
                    CHECK(candidate_revision >= 1),
                binding_revision INTEGER NOT NULL
                    CHECK(binding_revision >= 1),
                target_space_id TEXT NOT NULL,
                target_entry_id TEXT NOT NULL,
                target_revision INTEGER NOT NULL
                    CHECK(target_revision >= 1),
                mode TEXT NOT NULL,
                semantic_json BLOB NOT NULL,
                semantic_sha256 TEXT NOT NULL,
                review_json BLOB NOT NULL,
                review_sha256 TEXT NOT NULL,
                first_operation_id TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                UNIQUE (
                    binding_id,
                    job_id,
                    candidate_id,
                    candidate_revision,
                    target_space_id,
                    target_entry_id,
                    target_revision,
                    mode
                )
            );
            COMMIT;
            """
        )
        versions = {
            int(row[0])
            for row in connection.execute(
                "SELECT version FROM memory_host_v2_schema"
            )
        }
        if versions != {_SCHEMA_VERSION}:
            raise SQLiteMemoryHostV2IntegrityError("memory_host_schema_unsupported")
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


class MemoryCompletionFactoryResolver(Protocol):
    """Host policy deciding whether one attached run gets a terminal projector."""

    def resolve(
        self,
        request: MemoryAttachmentRequest,
    ) -> MemoryCompletionFactory | None:
        ...


class _SQLiteBoundChatReadCapability:
    """Toolkit-facing read-only view of one bound chat workspace."""

    def __init__(
        self,
        *,
        binding_id: str,
        workspace: MemoryWorkspaceService,
    ) -> None:
        if not isinstance(workspace, MemoryWorkspaceService):
            raise TypeError("workspace must be a MemoryWorkspaceService")
        self.binding_id = _required_text(
            binding_id,
            "binding_id",
            maximum=512,
            identifier=True,
        )
        if workspace.binding_id != self.binding_id:
            raise SQLiteMemoryHostV2Error("workspace_binding_mismatch")
        self._workspace = workspace

    @property
    def space_id(self) -> str:
        return self._workspace.space.space_id

    @property
    def space_revision(self) -> int:
        return self._workspace.space.revision

    def list_entries(
        self,
        *,
        path: str,
        recursive: bool,
        limit: int,
    ) -> Mapping[str, Any]:
        page = self._workspace.list(
            parent_path=path or "/",
            recursive=recursive,
            limit=limit,
        )
        return {
            "entries": page.entries,
            "truncated": page.has_more,
            "next_cursor": page.next_cursor,
        }

    def search_entries(
        self,
        *,
        query: str,
        limit: int,
    ) -> Mapping[str, Any]:
        result = self._workspace.search(query, limit=limit)
        return {
            "query": query,
            "backend": "lexical_fallback" if result.lexical_fallback else "hybrid",
            "vector_status": "degraded" if result.vector_error else "ready",
            "results": tuple(
                {
                    "entry": hit.entry,
                    "score": hit.score,
                    "matched_by": hit.matched_by,
                    "source_refs": hit.source_refs,
                }
                for hit in result.hits
            ),
        }

    def read_content(
        self,
        *,
        ref: ResourceRef,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage:
        page = self._workspace.read(ref, offset=offset, limit=limit)
        digest = (
            _sha256(page.data)
            if page.offset == 0 and len(page.data) == page.total_bytes
            else ""
        )
        return MemoryToolContentPage(
            ref=ref,
            media_type=page.media_type,
            data=page.data,
            offset=page.offset,
            total_bytes=page.total_bytes,
            sha256=digest,
        )


class SQLiteMemoryAttachmentFactory:
    """Attach the exact normal-agent Memory V2 capabilities for one run."""

    def __init__(
        self,
        *,
        binding_id: str,
        repository: BoundCurationRepository,
        workspace: MemoryWorkspaceService,
        references: BoundExternalReferenceCodec,
        context: BoundContextMemoryCapability,
        completion_factory_resolver: MemoryCompletionFactoryResolver
        | None = None,
        proposal_source_refs_resolver: Callable[
            [MemoryAttachmentRequest], Sequence[ResourceRef]
        ]
        | None = None,
        long_term: BoundMemoryReadCapability | None = None,
        allowed_long_term_refs: Sequence[ResourceRef] = (),
    ) -> None:
        self.binding_id = _required_text(
            binding_id,
            "binding_id",
            maximum=512,
            identifier=True,
        )
        if not isinstance(repository, BoundCurationRepository):
            raise TypeError("repository must be a BoundCurationRepository")
        if repository.binding_id != self.binding_id:
            raise SQLiteMemoryHostV2Error("repository_binding_mismatch")
        if not isinstance(workspace, MemoryWorkspaceService):
            raise TypeError("workspace must be a MemoryWorkspaceService")
        if workspace.binding_id != self.binding_id:
            raise SQLiteMemoryHostV2Error("workspace_binding_mismatch")
        if getattr(repository, "target_space_id", "") != workspace.space.space_id:
            raise SQLiteMemoryHostV2Error("workspace_scope_mismatch")
        if not callable(getattr(repository, "bind_candidate_proposals", None)):
            raise SQLiteMemoryHostV2Error("candidate_repository_incomplete")
        for label, capability in (("references", references), ("context", context)):
            if getattr(capability, "binding_id", "") != self.binding_id:
                raise SQLiteMemoryHostV2Error(f"{label}_binding_mismatch")
        if completion_factory_resolver is not None and not callable(
            getattr(completion_factory_resolver, "resolve", None)
        ):
            raise TypeError("completion_factory_resolver must provide resolve(request)")
        if proposal_source_refs_resolver is not None and not callable(
            proposal_source_refs_resolver
        ):
            raise TypeError("proposal_source_refs_resolver must be callable")

        if isinstance(allowed_long_term_refs, (str, bytes, bytearray)):
            raise TypeError("allowed_long_term_refs must be a sequence")
        allowed = tuple(allowed_long_term_refs)
        if len(set(allowed)) != len(allowed):
            raise SQLiteMemoryHostV2Error("long_term_refs_duplicate")
        if allowed and long_term is None:
            raise SQLiteMemoryHostV2Error("long_term_capability_missing")
        if long_term is not None:
            if getattr(long_term, "binding_id", "") != self.binding_id:
                raise SQLiteMemoryHostV2Error("long_term_binding_mismatch")
            long_term_space_id = str(getattr(long_term, "space_id", ""))
            if not long_term_space_id or long_term_space_id == workspace.space.space_id:
                raise SQLiteMemoryHostV2Error("long_term_scope_mismatch")
            if any(
                not isinstance(ref, ResourceRef)
                or ref.kind != "memory"
                or ref.fragment != long_term_space_id
                for ref in allowed
            ):
                raise SQLiteMemoryHostV2Error("long_term_ref_scope_mismatch")
            for method_name in ("list_entries", "search_entries", "read_content"):
                if not callable(getattr(long_term, method_name, None)):
                    raise SQLiteMemoryHostV2Error("long_term_capability_incomplete")

        self.repository = repository
        self.workspace = workspace
        self.references = references
        self.context = context
        self.completion_factory_resolver = completion_factory_resolver
        self.proposal_source_refs_resolver = proposal_source_refs_resolver
        self.long_term = long_term
        self.allowed_long_term_refs = allowed

    def attach(
        self,
        request: MemoryAttachmentRequest,
    ) -> MemoryAttachment:
        if not isinstance(request, MemoryAttachmentRequest):
            raise TypeError("request must be a MemoryAttachmentRequest")
        binding = MemoryToolkitRunBinding(
            binding_id=self.binding_id,
            session_id=request.session_id,
            attempt_id=request.attempt_id,
            run_id=request.run_id,
        )
        completion_factory = None
        resolver = self.completion_factory_resolver
        if resolver is not None:
            completion_factory = resolver.resolve(request)
            if completion_factory is not None and not callable(
                getattr(completion_factory, "build", None)
            ):
                raise SQLiteMemoryHostV2Error("completion_factory_invalid")

        candidates = self.repository.bind_candidate_proposals(
            binding=binding,
            root_run_id=request.root_run_id,
        )
        if getattr(candidates, "binding_id", "") != self.binding_id:
            raise SQLiteMemoryHostV2Error("candidate_binding_mismatch")
        chat = _SQLiteBoundChatReadCapability(
            binding_id=self.binding_id,
            workspace=self.workspace,
        )
        capabilities = NormalMemoryToolkitCapabilities(
            references=self.references,
            context=self.context,
            chat=chat,
            candidates=candidates,
            proposal_source_refs_resolver=(
                None
                if self.proposal_source_refs_resolver is None
                else lambda: self.proposal_source_refs_resolver(request)
            ),
            long_term=self.long_term,
            allowed_long_term_refs=self.allowed_long_term_refs,
        )
        return MemoryAttachment(
            binding=binding,
            capabilities=capabilities,
            completion_factory=completion_factory,
        )


class SQLiteConsolidationCapabilityFactory:
    """Build an exact consolidation bundle for one SQLite-backed chat scope."""

    def __init__(
        self,
        *,
        binding_id: str,
        database_path: str | Path,
        repository: BoundCurationRepository,
        workspace: MemoryWorkspaceService,
        references: BoundExternalReferenceCodec,
        context: BoundContextMemoryCapability,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.binding_id = _required_text(
            binding_id,
            "binding_id",
            maximum=512,
            identifier=True,
        )
        if not isinstance(repository, BoundCurationRepository):
            raise TypeError("repository must be a BoundCurationRepository")
        if repository.binding_id != self.binding_id:
            raise SQLiteMemoryHostV2Error("repository_binding_mismatch")
        if not isinstance(workspace, MemoryWorkspaceService):
            raise TypeError("workspace must be a MemoryWorkspaceService")
        if workspace.binding_id != self.binding_id:
            raise SQLiteMemoryHostV2Error("workspace_binding_mismatch")
        for label, capability in (("references", references), ("context", context)):
            if getattr(capability, "binding_id", "") != self.binding_id:
                raise SQLiteMemoryHostV2Error(f"{label}_binding_mismatch")
        target_space_id = getattr(repository, "target_space_id", "")
        if target_space_id != workspace.space.space_id:
            raise SQLiteMemoryHostV2Error("workspace_scope_mismatch")

        self.database_path = Path(database_path)
        expected_database = self.database_path.resolve()
        repository_store = getattr(repository, "_store", None)
        workspace_store = getattr(workspace.repository, "_store", None)
        for label, store in (
            ("repository", repository_store),
            ("workspace", workspace_store),
        ):
            store_path = getattr(store, "database_path", None)
            if store_path is None or Path(store_path).resolve() != expected_database:
                raise SQLiteMemoryHostV2Error(f"{label}_database_mismatch")
        for method_name in ("read_candidate", "read_candidate_content"):
            if not callable(getattr(repository, method_name, None)):
                raise SQLiteMemoryHostV2Error("candidate_repository_incomplete")

        self.repository = repository
        self.workspace = workspace
        self.references = references
        self.context = context
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        initialize_sqlite_memory_host_v2_schema(database_path=self.database_path)

    def build(
        self,
        *,
        binding: MemoryToolkitRunBinding,
        job: ConsolidationJob,
        mutation_guard: BoundCuratorMutationGuard,
    ) -> ConsolidationMemoryToolkitCapabilities:
        if not isinstance(binding, MemoryToolkitRunBinding):
            raise TypeError("binding must be a MemoryToolkitRunBinding")
        if not isinstance(job, ConsolidationJob):
            raise TypeError("job must be a ConsolidationJob")
        if job.status is not ConsolidationJobStatus.LEASED:
            raise SQLiteMemoryHostV2Error("job_not_leased")
        digest = hashlib.sha256(
            f"{job.job_id}:{job.revision}".encode("utf-8")
        ).hexdigest()
        expected_binding = MemoryToolkitRunBinding(
            binding_id=self.binding_id,
            session_id=job.trigger.session_id,
            attempt_id=f"memory-curator-attempt-{digest}",
            run_id=f"memory-curator-run-{digest}",
        )
        if binding != expected_binding:
            raise SQLiteMemoryHostV2Error("run_binding_mismatch")
        current = self.repository.read_job(job_id=job.job_id)
        if current != job:
            raise SQLiteMemoryHostV2Error("job_revision_changed")
        fence = CuratorLeaseFence.from_job(self.binding_id, job)
        if (
            not isinstance(mutation_guard, BoundCuratorMutationGuard)
            or mutation_guard.fence != fence
        ):
            raise SQLiteMemoryHostV2Error("lease_guard_mismatch")
        if any(
            candidate.target_space_id != self.workspace.space.space_id
            or candidate.outcome is not CandidateStatus.PROCESSING
            for candidate in job.candidates
        ):
            raise SQLiteMemoryHostV2Error("candidate_scope_mismatch")
        for candidate in job.candidates:
            durable = self.repository.read_candidate(ref=candidate.candidate_ref)
            if durable != candidate:
                raise SQLiteMemoryHostV2Error("candidate_revision_changed")

        source_refs = _deduplicate_refs(job.candidates)
        chat = WorkspaceMemoryCapability(
            binding_id=self.binding_id,
            service=self.workspace,
            mutation_source_refs=source_refs,
        )
        consolidation = _SQLiteBoundConsolidationCapability(
            factory=self,
            job=job,
            mutation_guard=mutation_guard,
        )
        return ConsolidationMemoryToolkitCapabilities(
            binding_id=self.binding_id,
            references=self.references,
            context=self.context,
            chat=chat,
            consolidation=consolidation,
            job_id=job.job_id,
            candidate_refs=tuple(
                candidate.candidate_ref for candidate in job.candidates
            ),
            lease_fence=fence,
            mutation_guard=mutation_guard,
            source_refs=source_refs,
        )


class _SQLiteBoundConsolidationCapability:
    def __init__(
        self,
        *,
        factory: SQLiteConsolidationCapabilityFactory,
        job: ConsolidationJob,
        mutation_guard: BoundCuratorMutationGuard,
    ) -> None:
        self.binding_id = factory.binding_id
        self._factory = factory
        self._repository = factory.repository
        self._workspace = factory.workspace
        self._job = job
        self._mutation_guard = mutation_guard
        self._fence = CuratorLeaseFence.from_job(self.binding_id, job)
        self._candidates = {
            candidate.candidate_ref: candidate for candidate in job.candidates
        }

    def _candidate(
        self,
        *,
        job_id: str,
        ref: ResourceRef,
    ) -> FrozenCandidateSnapshot:
        if job_id != self._job.job_id or not isinstance(ref, ResourceRef):
            raise SQLiteMemoryHostV2Error("candidate_scope_mismatch")
        candidate = self._candidates.get(ref)
        if candidate is None:
            raise SQLiteMemoryHostV2Error("candidate_scope_mismatch")
        durable = self._repository.read_candidate(ref=ref)
        if durable != candidate:
            raise SQLiteMemoryHostV2Error("candidate_revision_changed")
        return candidate

    def _mutation_candidate(
        self,
        *,
        job_id: str,
        candidate_ref: ResourceRef,
        expected_binding_revision: int,
        mutation_guard: BoundCuratorMutationGuard,
        operation_id: str,
    ) -> FrozenCandidateSnapshot:
        candidate = self._candidate(job_id=job_id, ref=candidate_ref)
        expected = _positive_integer(
            expected_binding_revision,
            "expected_binding_revision",
        )
        if expected != candidate.binding_revision:
            raise SQLiteMemoryHostV2Error("candidate_binding_revision_changed")
        if (
            mutation_guard is not self._mutation_guard
            or mutation_guard.fence != self._fence
        ):
            raise SQLiteMemoryHostV2Error("lease_guard_mismatch")
        normalized_operation = _required_text(
            operation_id,
            "operation_id",
            maximum=256,
            identifier=True,
        )
        if _IDENTIFIER_RE.fullmatch(normalized_operation) is None:
            raise SQLiteMemoryHostV2Error("operation_id_invalid")
        mutation_guard.assert_active()
        return candidate

    @staticmethod
    def _synthetic_candidate_bytes(
        candidate: FrozenCandidateSnapshot,
    ) -> tuple[str, bytes]:
        if candidate.kind == "link":
            return "text/uri-list", candidate.link_url.encode("utf-8")
        if candidate.kind == "folder":
            return "application/json", _canonical_json_bytes(
                {
                    "description": candidate.description,
                    "kind": candidate.kind,
                    "name": candidate.name,
                    "path": candidate.target_path,
                }
            )
        raise SQLiteMemoryHostV2Error("candidate_content_kind_invalid")

    def read_candidate(
        self,
        *,
        job_id: str,
        ref: ResourceRef,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage:
        candidate = self._candidate(job_id=job_id, ref=ref)
        page_offset = _non_negative_integer(offset, "offset")
        page_limit = _positive_integer(limit, "limit")
        if page_limit > MAX_PAGE_READ_BYTES:
            raise SQLiteMemoryHostV2Error("limit_invalid")
        if candidate.kind == "file":
            page = self._repository.read_candidate_content(
                ref=ref,
                offset=page_offset,
                limit=page_limit,
            )
            if (
                not isinstance(page, MemoryToolContentPage)
                or page.ref != ref
                or page.offset != page_offset
                or page.total_bytes != candidate.byte_length
                or page.sha256 != candidate.content_sha256
                or page.media_type != candidate.media_type
                or len(page.data) > page_limit
            ):
                raise SQLiteMemoryHostV2IntegrityError(
                    "candidate_content_metadata_changed"
                )
            return page
        media_type, content = self._synthetic_candidate_bytes(candidate)
        if page_offset > len(content):
            raise SQLiteMemoryHostV2Error("offset_invalid")
        return MemoryToolContentPage(
            ref=ref,
            media_type=media_type,
            data=content[page_offset : page_offset + page_limit],
            offset=page_offset,
            total_bytes=len(content),
            sha256=_sha256(content),
        )

    def _read_full_candidate(self, candidate: FrozenCandidateSnapshot) -> bytes:
        if candidate.kind != "file":
            return self._synthetic_candidate_bytes(candidate)[1]
        content = bytearray()
        offset = 0
        while offset < candidate.byte_length:
            page = self.read_candidate(
                job_id=self._job.job_id,
                ref=candidate.candidate_ref,
                offset=offset,
                limit=MAX_PAGE_READ_BYTES,
            )
            if not page.data:
                raise SQLiteMemoryHostV2IntegrityError(
                    "candidate_content_pagination_stalled"
                )
            content.extend(page.data)
            offset += len(page.data)
        payload = bytes(content)
        if (
            len(payload) != candidate.byte_length
            or _sha256(payload) != candidate.content_sha256
        ):
            raise SQLiteMemoryHostV2IntegrityError("candidate_content_digest_changed")
        return payload

    @staticmethod
    def _workspace_kind(candidate: FrozenCandidateSnapshot) -> MemoryEntryKind:
        if candidate.kind == "folder":
            return MemoryEntryKind.FOLDER
        if candidate.kind == "link":
            return MemoryEntryKind.LINK
        if candidate.kind != "file":
            raise SQLiteMemoryHostV2Error("candidate_storage_kind_invalid")
        if candidate.media_type == "text/markdown":
            return MemoryEntryKind.MARKDOWN
        if candidate.media_type.startswith("image/"):
            return MemoryEntryKind.IMAGE
        raise SQLiteMemoryHostV2Error("candidate_media_type_unsupported")

    def _effect_semantic(
        self,
        candidate: FrozenCandidateSnapshot,
    ) -> Mapping[str, Any]:
        return {
            "binding_id": self.binding_id,
            "job_id": self._job.job_id,
            "candidate_ref": candidate.candidate_ref.to_dict(),
            "binding_revision": candidate.binding_revision,
            "target_space_id": candidate.target_space_id,
            "target_path": candidate.target_path,
            "name": candidate.name,
            "description": candidate.description,
            "kind": candidate.kind,
            "media_type": candidate.media_type,
            "link_url": candidate.link_url,
            "source_refs": [ref.to_dict() for ref in candidate.source_refs],
            "payload_sha256": candidate.payload_sha256,
            "content_sha256": candidate.content_sha256,
            "byte_length": candidate.byte_length,
        }

    def _stable_effect_operation_id(
        self,
        candidate: FrozenCandidateSnapshot,
    ) -> str:
        digest = _sha256(_canonical_json_bytes(self._effect_semantic(candidate)))
        return f"memory-curator-apply-{digest}"

    def _stable_entry_id(self, candidate: FrozenCandidateSnapshot) -> str:
        operation_id = self._stable_effect_operation_id(candidate)
        digest = hashlib.sha256(
            f"{candidate.target_space_id}\0{operation_id}".encode("utf-8")
        ).hexdigest()
        return f"memory-{digest[:32]}"

    @staticmethod
    def _entry_ref(entry: MemoryEntry) -> ResourceRef:
        return ResourceRef(
            "memory",
            entry.entry_id,
            entry.revision,
            entry.space_id,
        )

    def _read_full_entry(self, entry: MemoryEntry) -> bytes:
        if entry.kind is MemoryEntryKind.LINK:
            return entry.link_url.encode("utf-8")
        if entry.kind is MemoryEntryKind.FOLDER:
            return b""
        ref = self._entry_ref(entry)
        content = bytearray()
        offset = 0
        while True:
            page = self._workspace.read(
                ref,
                offset=offset,
                limit=MAX_PAGE_READ_BYTES,
            )
            if page.offset != offset or page.offset + len(page.data) > page.total_bytes:
                raise SQLiteMemoryHostV2IntegrityError(
                    "workspace_content_metadata_changed"
                )
            content.extend(page.data)
            offset += len(page.data)
            if offset == page.total_bytes:
                break
            if not page.data:
                raise SQLiteMemoryHostV2IntegrityError(
                    "workspace_content_pagination_stalled"
                )
        return bytes(content)

    def _validate_applied_entry(
        self,
        *,
        candidate: FrozenCandidateSnapshot,
        entry: MemoryEntry,
        content: bytes,
    ) -> None:
        expected_kind = self._workspace_kind(candidate)
        expected_media_type = (
            candidate.media_type
            if expected_kind in {MemoryEntryKind.MARKDOWN, MemoryEntryKind.IMAGE}
            else ""
        )
        expected_link = (
            candidate.link_url if expected_kind is MemoryEntryKind.LINK else ""
        )
        if (
            entry.entry_id != self._stable_entry_id(candidate)
            or entry.space_id != candidate.target_space_id
            or entry.path != candidate.target_path
            or entry.name != candidate.name
            or entry.name != virtual_name(candidate.target_path)
            or entry.description != candidate.description
            or entry.kind is not expected_kind
            or entry.revision != 1
            or entry.source_refs != candidate.source_refs
            or entry.tags
            or entry.media_type != expected_media_type
            or entry.link_url != expected_link
            or entry.deleted
        ):
            raise SQLiteMemoryHostV2IntegrityError("workspace_effect_metadata_changed")
        if expected_kind in {MemoryEntryKind.MARKDOWN, MemoryEntryKind.IMAGE}:
            stored = self._read_full_entry(entry)
            if stored != content or _sha256(stored) != candidate.content_sha256:
                raise SQLiteMemoryHostV2IntegrityError(
                    "workspace_effect_content_changed"
                )

    def _existing_effect(
        self,
        *,
        candidate: FrozenCandidateSnapshot,
        content: bytes,
    ) -> MemoryEntry | None:
        try:
            entry = self._workspace.repository.read_current_entry(
                entry_id=self._stable_entry_id(candidate),
            )
        except RepositoryNotFoundError:
            return None
        self._validate_applied_entry(
            candidate=candidate,
            entry=entry,
            content=content,
        )
        return entry

    def _find_path_conflict(
        self,
        candidate: FrozenCandidateSnapshot,
    ) -> MemoryEntry | None:
        target_path = canonical_entry_path(candidate.target_path)
        target_identity = _path_identity(target_path)
        parent = target_path.rsplit("/", 1)[0] or "/"
        cursor: str | None = None
        scanned = 0
        while scanned < 10_000:
            page = self._workspace.repository.list_entries(
                parent_path=parent,
                include_deleted=False,
                limit=200,
                cursor=cursor,
            )
            for entry in page.entries:
                if _path_identity(entry.path) == target_identity:
                    return entry
            scanned += len(page.entries)
            if not page.has_more:
                return None
            if page.next_cursor is None or page.next_cursor == cursor:
                raise SQLiteMemoryHostV2IntegrityError(
                    "workspace_listing_did_not_advance"
                )
            cursor = page.next_cursor
        raise SQLiteMemoryHostV2Error("workspace_listing_limit_exceeded")

    def _applied_result(
        self,
        candidate: FrozenCandidateSnapshot,
        entry: MemoryEntry,
    ) -> Mapping[str, Any]:
        return {
            "outcome": "applied",
            "candidate_ref": candidate.candidate_ref,
            "target_space_id": candidate.target_space_id,
            "result_ref": self._entry_ref(entry),
        }

    def _conflict_result(
        self,
        candidate: FrozenCandidateSnapshot,
        entry: MemoryEntry,
    ) -> Mapping[str, Any]:
        return {
            "outcome": "conflict",
            "reason": "path_exists",
            "candidate_ref": candidate.candidate_ref,
            "target_space_id": candidate.target_space_id,
            "target_entry_ref": self._entry_ref(entry),
            "server_review_required": True,
        }

    def apply_new(
        self,
        *,
        job_id: str,
        candidate_ref: ResourceRef,
        expected_binding_revision: int,
        expected_space_revision: int,
        mutation_guard: BoundCuratorMutationGuard,
        operation_id: str,
    ) -> Mapping[str, Any]:
        candidate = self._mutation_candidate(
            job_id=job_id,
            candidate_ref=candidate_ref,
            expected_binding_revision=expected_binding_revision,
            mutation_guard=mutation_guard,
            operation_id=operation_id,
        )
        space_revision = _positive_integer(
            expected_space_revision,
            "expected_space_revision",
        )
        content = (
            self._read_full_candidate(candidate) if candidate.kind == "file" else b""
        )
        replay = self._existing_effect(candidate=candidate, content=content)
        if replay is not None:
            return self._applied_result(candidate, replay)
        conflict = self._find_path_conflict(candidate)
        if conflict is not None:
            return self._conflict_result(candidate, conflict)

        # Re-check the lease immediately before the workspace side effect.  The
        # workspace itself provides the durable CAS and operation receipt.
        mutation_guard.assert_active()
        stable_operation = self._stable_effect_operation_id(candidate)
        try:
            kind = self._workspace_kind(candidate)
            common = {
                "path": candidate.target_path,
                "description": candidate.description,
                "expected_space_revision": space_revision,
                "source_refs": candidate.source_refs,
                "operation_id": stable_operation,
            }
            if kind is MemoryEntryKind.FOLDER:
                entry = self._workspace.create_folder(**common)
            elif kind is MemoryEntryKind.LINK:
                entry = self._workspace.create_link(
                    **common,
                    url=candidate.link_url,
                )
            elif kind is MemoryEntryKind.MARKDOWN:
                entry = self._workspace.write_markdown(
                    **common,
                    content=content,
                )
            else:
                entry = self._workspace.write_image(
                    **common,
                    content=content,
                    media_type=candidate.media_type,
                )
        except RepositoryConflictError:
            replay = self._existing_effect(candidate=candidate, content=content)
            if replay is not None:
                return self._applied_result(candidate, replay)
            conflict = self._find_path_conflict(candidate)
            if conflict is not None:
                return self._conflict_result(candidate, conflict)
            raise
        self._validate_applied_entry(
            candidate=candidate,
            entry=entry,
            content=content,
        )
        return self._applied_result(candidate, entry)

    def _entry_diff_record(self, entry: MemoryEntry) -> Mapping[str, Any]:
        content = self._read_full_entry(entry)
        return {
            "ref": self._entry_ref(entry).to_dict(),
            "path": entry.path,
            "name": entry.name,
            "description": entry.description,
            "kind": entry.kind.value,
            "media_type": entry.media_type,
            "link_url": entry.link_url,
            "content_sha256": _sha256(content) if content else "",
            "byte_length": len(content),
            "source_refs_sha256": _sha256(
                _canonical_json_bytes([ref.to_dict() for ref in entry.source_refs])
            ),
            "source_ref_count": len(entry.source_refs),
        }

    def _candidate_diff_record(
        self,
        candidate: FrozenCandidateSnapshot,
    ) -> Mapping[str, Any]:
        kind = self._workspace_kind(candidate)
        return {
            "ref": candidate.candidate_ref.to_dict(),
            "path": candidate.target_path,
            "name": candidate.name,
            "description": candidate.description,
            "kind": kind.value,
            "media_type": candidate.media_type
            if kind
            in {
                MemoryEntryKind.MARKDOWN,
                MemoryEntryKind.IMAGE,
            }
            else "",
            "link_url": candidate.link_url,
            "content_sha256": candidate.content_sha256,
            "byte_length": candidate.byte_length,
            "source_refs_sha256": _sha256(
                _canonical_json_bytes([ref.to_dict() for ref in candidate.source_refs])
            ),
            "source_ref_count": len(candidate.source_refs),
        }

    def _persist_review(
        self,
        *,
        candidate: FrozenCandidateSnapshot,
        target: MemoryEntry,
        mode: str,
        operation_id: str,
        review_diff: Mapping[str, Any],
    ) -> ResourceRef:
        semantic = {
            "binding_id": self.binding_id,
            "job_id": self._job.job_id,
            "candidate": candidate.to_dict(),
            "target": target.to_dict(),
            "mode": mode,
        }
        semantic_json = _canonical_json_bytes(semantic)
        semantic_sha256 = _sha256(semantic_json)
        review_id = f"memory-review-{semantic_sha256[:32]}"
        review_json = _canonical_json_bytes(review_diff)
        review_sha256 = _sha256(review_json)
        now_ms = _non_negative_integer(self._factory._clock_ms(), "clock_ms")
        connection = self._factory._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM memory_review_proposals WHERE review_id = ?",
                (review_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO memory_review_proposals(
                        review_id, binding_id, job_id, candidate_id,
                        candidate_revision, binding_revision, target_space_id,
                        target_entry_id, target_revision, mode, semantic_json,
                        semantic_sha256, review_json, review_sha256,
                        first_operation_id, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_id,
                        self.binding_id,
                        self._job.job_id,
                        candidate.candidate_ref.resource_id,
                        candidate.candidate_ref.revision,
                        candidate.binding_revision,
                        candidate.target_space_id,
                        target.entry_id,
                        target.revision,
                        mode,
                        semantic_json,
                        semantic_sha256,
                        review_json,
                        review_sha256,
                        operation_id,
                        now_ms,
                    ),
                )
            else:
                if (
                    row["binding_id"] != self.binding_id
                    or row["job_id"] != self._job.job_id
                    or row["candidate_id"] != candidate.candidate_ref.resource_id
                    or row["candidate_revision"] != candidate.candidate_ref.revision
                    or row["binding_revision"] != candidate.binding_revision
                    or row["target_space_id"] != candidate.target_space_id
                    or row["target_entry_id"] != target.entry_id
                    or row["target_revision"] != target.revision
                    or row["mode"] != mode
                    or bytes(row["semantic_json"]) != semantic_json
                    or row["semantic_sha256"] != semantic_sha256
                    or bytes(row["review_json"]) != review_json
                    or row["review_sha256"] != review_sha256
                ):
                    raise SQLiteMemoryHostV2IntegrityError(
                        "memory_review_record_changed"
                    )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return ResourceRef(
            "memory_review",
            review_id,
            1,
            candidate.target_space_id,
        )

    def propose_review(
        self,
        *,
        job_id: str,
        candidate_ref: ResourceRef,
        expected_binding_revision: int,
        target_entry_id: str,
        expected_target_revision: int,
        mode: str,
        mutation_guard: BoundCuratorMutationGuard,
        operation_id: str,
    ) -> Mapping[str, Any]:
        candidate = self._mutation_candidate(
            job_id=job_id,
            candidate_ref=candidate_ref,
            expected_binding_revision=expected_binding_revision,
            mutation_guard=mutation_guard,
            operation_id=operation_id,
        )
        target_id = _required_text(
            target_entry_id,
            "target_entry_id",
            maximum=512,
            identifier=True,
        )
        target_revision = _positive_integer(
            expected_target_revision,
            "expected_target_revision",
        )
        normalized_mode = _required_text(mode, "mode", maximum=32).casefold()
        if normalized_mode != "overwrite":
            raise SQLiteMemoryHostV2Error("review_mode_unsupported")
        try:
            target = self._workspace.repository.read_current_entry(
                entry_id=target_id,
            )
        except RepositoryNotFoundError as exc:
            raise SQLiteMemoryHostV2Error("review_target_not_found") from exc
        if (
            target.space_id != candidate.target_space_id
            or target.revision != target_revision
            or _path_identity(target.path) != _path_identity(candidate.target_path)
            or target.deleted
        ):
            raise SQLiteMemoryHostV2Error("review_target_changed")

        candidate_record = self._candidate_diff_record(candidate)
        target_record = self._entry_diff_record(target)
        changes = tuple(
            field_name
            for field_name in (
                "description",
                "kind",
                "media_type",
                "link_url",
                "content_sha256",
                "byte_length",
                "source_refs_sha256",
            )
            if candidate_record[field_name] != target_record[field_name]
        )
        review_diff = {
            "schema": _REVIEW_SCHEMA,
            "mode": normalized_mode,
            "candidate": candidate_record,
            "target": target_record,
            "changes": changes,
            "requires_user_confirmation": True,
        }
        # The persisted review is the only semantic side effect.  Fence it as
        # close to the write transaction as this independently-owned store permits.
        mutation_guard.assert_active()
        review_ref = self._persist_review(
            candidate=candidate,
            target=target,
            mode=normalized_mode,
            operation_id=operation_id,
            review_diff=review_diff,
        )
        return {
            "outcome": "awaiting_user",
            "candidate_ref": candidate.candidate_ref,
            "target_space_id": candidate.target_space_id,
            "result_ref": review_ref,
            "review_diff": review_diff,
            "requires_user_confirmation": True,
        }


__all__ = [
    "MemoryCompletionFactoryResolver",
    "SQLiteConsolidationCapabilityFactory",
    "SQLiteMemoryAttachmentFactory",
    "SQLiteMemoryHostV2Error",
    "SQLiteMemoryHostV2IntegrityError",
    "initialize_sqlite_memory_host_v2_schema",
]
