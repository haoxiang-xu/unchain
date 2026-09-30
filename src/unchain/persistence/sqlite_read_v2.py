"""Scope-bound read facade over the SQLite Context/Memory V2 data plane.

The facade accepts durable model references and virtual workspace paths only.
Host identity resolution stays outside this module; a product host must bind an
explicit owner, chat workspace, and exhaustive execution allow-list first.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from unchain.context import ArtifactReadPage, ArtifactService, CheckpointWriteStatus
from unchain.journal import (
    ArtifactRef,
    EventCursor,
    EventRange,
    JournalPage,
    ResourceRef,
)
from unchain.journal.models import _required_text
from unchain.memory.toolkit.models import MemoryToolContentPage
from unchain.memory.workspace import (
    MemoryEntry,
    MemoryChildPage,
    MemoryEntryKind,
    MemoryEntryPage,
    MemorySpace,
    VectorProjectionPage,
    WorkspaceSearchResult,
    WorkspaceSearchService,
)
from unchain.memory.workspace.paths import canonical_parent_path
from unchain.memory.workspace.ports import (
    RepositoryNotFoundError,
    RepositoryScopeError,
)

from .sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from .sqlite_memory_v2 import (
    SQLiteMemoryV2Store,
    _SQLiteBoundMemoryWorkspaceRepository,
)
from .sqlite_v2 import (
    SQLiteContextV2Store,
    _SQLiteBoundContextV2Repository,
    serialized_context_v2_database_access,
)


_MAX_EXECUTIONS = 10_000
_MAX_LIST_RESULTS = 200
_MAX_LIST_SCAN = 10_000
_SCAN_PAGE_SIZE = 200
_MAX_CONTEXT_CONTENT_PAGE_BYTES = 65_536
_MAX_CHECKPOINT_EVENT_PAGE_SIZE = 50
_CHECKPOINT_EVENT_FRAGMENT = re.compile(r"^event/([1-9][0-9]*)$")


class SQLiteContextV2ReadError(RuntimeError):
    """The verified SQLite read plane is unavailable or inconsistent."""


class SQLiteContextV2ReadScopeError(RepositoryScopeError):
    """A requested record is outside the immutable bound chat scope."""


@dataclass(frozen=True, slots=True)
class SQLiteContextV2StoreReadStatus:
    """Database-level health only; it carries no chat or execution scope."""

    available: bool
    schema_version: int
    journal_mode: str
    lexical_backend: str
    vector_status: str = "disabled"

    SCHEMA = "unchain.sqlite_context_v2_store_read_status.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "available": self.available,
            "schema_version": self.schema_version,
            "journal_mode": self.journal_mode,
            "lexical_backend": self.lexical_backend,
            "vector_status": self.vector_status,
        }


def read_sqlite_context_v2_store_status(
    database_path: str | Path,
) -> SQLiteContextV2StoreReadStatus:
    """Verify the shared data plane without inventing a chat capability."""

    try:
        path = Path(database_path).expanduser().resolve()
        if not path.is_file():
            raise SQLiteContextV2ReadError("Context V2 database is unavailable")
        with serialized_context_v2_database_access(path):
            connection = sqlite3.connect(
                f"{path.as_uri()}?mode=ro",
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
            try:
                connection.execute("PRAGMA query_only = ON")
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise SQLiteContextV2ReadError("SQLite quick_check failed")
                context_versions = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT version FROM context_v2_schema ORDER BY version"
                    )
                }
                memory_versions = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT version FROM memory_v2_schema ORDER BY version"
                    )
                }
                tables = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                    )
                }
                required = {
                    "executions",
                    "events",
                    "artifacts",
                    "spaces",
                    "entries",
                    "index_state",
                }
                if (
                    # Status is queried before runtime construction/migration.
                    # These schemas share the read-only health surface; report
                    # the actual version without upgrading the database here.
                    context_versions not in ({1, 2}, {1, 2, 3})
                    or memory_versions != {1}
                    or not required.issubset(tables)
                ):
                    raise SQLiteContextV2ReadError(
                        "Context V2 database schema is unsupported"
                    )
                journal_mode = str(
                    connection.execute("PRAGMA journal_mode").fetchone()[0]
                ).casefold()
                if journal_mode != "wal":
                    raise SQLiteContextV2ReadError("Context V2 journal mode is unavailable")
                fts_available = "workspace_entries_fts" in tables
                if fts_available:
                    degraded = connection.execute(
                        "SELECT 1 FROM index_state "
                        "WHERE index_name = 'workspace_fts' AND status != 'ready' "
                        "LIMIT 1"
                    ).fetchone()
                    fts_available = degraded is None
            finally:
                connection.close()
    except SQLiteContextV2ReadError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError) as error:
        raise SQLiteContextV2ReadError(
            "Context V2 database status is unavailable"
        ) from error
    return SQLiteContextV2StoreReadStatus(
        available=True,
        schema_version=max(context_versions),
        journal_mode=journal_mode,
        lexical_backend="fts5" if fts_available else "degraded",
    )


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class ContextV2ReadScope:
    """Product-neutral durable scope selected before any read is exposed."""

    owner_chat_id: str
    execution_ids: tuple[str, ...]
    space_id: str

    def __post_init__(self) -> None:
        owner = _required_text(
            self.owner_chat_id,
            "owner_chat_id",
            maximum=512,
            identifier=True,
        )
        space_id = _required_text(
            self.space_id,
            "space_id",
            maximum=512,
            identifier=True,
        )
        try:
            values = tuple(self.execution_ids)
        except TypeError as error:
            raise TypeError(
                "execution_ids must be an iterable of identifiers"
            ) from error
        if not values or len(values) > _MAX_EXECUTIONS:
            raise ValueError("execution_ids must contain between 1 and 10000 items")
        normalized = tuple(
            _required_text(
                value,
                "execution_id",
                maximum=512,
                identifier=True,
            )
            for value in values
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError("execution_ids must not contain duplicates")
        object.__setattr__(self, "owner_chat_id", owner)
        object.__setattr__(self, "execution_ids", tuple(sorted(normalized)))
        object.__setattr__(self, "space_id", space_id)


@dataclass(frozen=True, slots=True)
class SQLiteContextV2ReadStatus:
    available: bool
    owner_chat_id: str
    execution_count: int
    space_id: str
    space_revision: int
    journal: str = "available"
    artifacts: str = "available"
    workspace: str = "available"
    search: str = "ready"

    SCHEMA = "unchain.sqlite_context_v2_read_status.v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.SCHEMA,
            "available": self.available,
            "owner_chat_id": self.owner_chat_id,
            "execution_count": self.execution_count,
            "space_id": self.space_id,
            "space_revision": self.space_revision,
            "journal": self.journal,
            "artifacts": self.artifacts,
            "workspace": self.workspace,
            "search": self.search,
        }


@dataclass(frozen=True, slots=True)
class VerifiedContentPage:
    """One integrity-verified page identified by a durable semantic ref."""

    ref: ResourceRef
    media_type: str
    data: bytes
    offset: int
    total_bytes: int
    sha256: str

    @property
    def next_offset(self) -> int:
        return self.offset + len(self.data)

    @property
    def has_more(self) -> bool:
        return self.next_offset < self.total_bytes


class SQLiteContextV2ReadService:
    """Bind official SQLite stores to one explicit, immutable read scope."""

    def __init__(
        self,
        *,
        context_store: SQLiteContextV2Store,
        memory_store: SQLiteMemoryV2Store,
        compiler_store: SQLiteContextCompilerV2Store | None = None,
    ) -> None:
        if not isinstance(context_store, SQLiteContextV2Store):
            raise TypeError("context_store must be a SQLiteContextV2Store")
        if not isinstance(memory_store, SQLiteMemoryV2Store):
            raise TypeError("memory_store must be a SQLiteMemoryV2Store")
        if compiler_store is not None and not isinstance(
            compiler_store,
            SQLiteContextCompilerV2Store,
        ):
            raise TypeError(
                "compiler_store must be a SQLiteContextCompilerV2Store or None"
            )
        context_database = context_store.database_path.expanduser().resolve()
        memory_database = memory_store.database_path.expanduser().resolve()
        context_objects = context_store.object_directory.expanduser().resolve()
        memory_objects = memory_store.object_directory.expanduser().resolve()
        if context_database != memory_database or context_objects != memory_objects:
            raise SQLiteContextV2ReadError(
                "context and memory stores must share one durable data plane"
            )
        if compiler_store is not None and (
            compiler_store.database_path.expanduser().resolve() != context_database
            or compiler_store.object_directory.expanduser().resolve() != context_objects
        ):
            raise SQLiteContextV2ReadError(
                "compiler store must share the Context V2 durable data plane"
            )
        self._context_store = context_store
        self._memory_store = memory_store
        self._compiler_store = compiler_store
        self._database_path = context_database

    def _execution_exists(self, execution_id: str) -> bool:
        try:
            with serialized_context_v2_database_access(self._database_path):
                connection = sqlite3.connect(
                    f"{self._database_path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=30.0,
                    isolation_level=None,
                )
                try:
                    connection.execute("PRAGMA query_only = ON")
                    row = connection.execute(
                        "SELECT 1 FROM executions WHERE execution_id = ?",
                        (execution_id,),
                    ).fetchone()
                finally:
                    connection.close()
        except sqlite3.Error as error:
            raise SQLiteContextV2ReadError(
                "durable execution scope is unavailable"
            ) from error
        return row is not None

    def bind(self, scope: ContextV2ReadScope) -> BoundSQLiteContextV2ReadService:
        if not isinstance(scope, ContextV2ReadScope):
            raise TypeError("scope must be a ContextV2ReadScope")
        if any(not self._execution_exists(value) for value in scope.execution_ids):
            raise SQLiteContextV2ReadScopeError(
                "one or more executions are outside the durable scope"
            )
        try:
            space = self._memory_store._load_space(
                space_id=scope.space_id,
                owner_chat_id=scope.owner_chat_id,
            )
        except RepositoryScopeError as error:
            raise SQLiteContextV2ReadScopeError(
                "workspace owner is outside the durable scope"
            ) from error
        workspace = _SQLiteBoundMemoryWorkspaceRepository(
            self._memory_store,
            space,
            scope.owner_chat_id,
        )
        journals = {
            execution_id: _SQLiteBoundContextV2Repository(
                self._context_store,
                execution_id,
            )
            for execution_id in scope.execution_ids
        }
        artifacts = {
            execution_id: ArtifactService(
                journal,
                sanitizer=lambda content, media_type: content,
            )
            for execution_id, journal in journals.items()
        }
        checkpoints = (
            {
                execution_id: self._compiler_store.bind_execution(
                    execution_id,
                    artifacts=artifacts[execution_id],
                ).checkpoints
                for execution_id in scope.execution_ids
            }
            if self._compiler_store is not None
            else {}
        )
        return BoundSQLiteContextV2ReadService(
            database_path=self._database_path,
            scope=scope,
            journals=journals,
            artifacts=artifacts,
            checkpoints=checkpoints,
            workspace=workspace,
        )


class BoundSQLiteContextV2ReadService:
    """Read-only operations that cannot select a new owner or workspace."""

    def __init__(
        self,
        *,
        database_path,
        scope: ContextV2ReadScope,
        journals,
        artifacts,
        checkpoints,
        workspace,
    ) -> None:
        self._database_path = database_path
        self._scope = scope
        self._journals = dict(journals)
        self._artifacts = dict(artifacts)
        self._checkpoints = dict(checkpoints)
        self._workspace = workspace
        self._search = WorkspaceSearchService(repository=workspace)

    @property
    def scope(self) -> ContextV2ReadScope:
        return self._scope

    @property
    def workspace_space(self) -> MemorySpace:
        """Return the immutable workspace descriptor selected at bind time."""

        return self._workspace.space

    def _journal(self, execution_id: object):
        normalized = _required_text(
            execution_id,
            "execution_id",
            maximum=512,
            identifier=True,
        )
        journal = self._journals.get(normalized)
        if journal is None:
            raise SQLiteContextV2ReadScopeError(
                "execution is outside the bound read scope"
            )
        return journal

    def status(self) -> SQLiteContextV2ReadStatus:
        try:
            with serialized_context_v2_database_access(self._database_path):
                connection = sqlite3.connect(
                    f"{self._database_path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=30.0,
                    isolation_level=None,
                )
                try:
                    connection.execute("PRAGMA query_only = ON")
                    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise SQLiteContextV2ReadError("SQLite quick_check failed")
                    row = connection.execute(
                        "SELECT status FROM index_state "
                        "WHERE index_name = 'workspace_fts' AND scope_id = ?",
                        (self._scope.space_id,),
                    ).fetchone()
                finally:
                    connection.close()
        except sqlite3.Error as error:
            raise SQLiteContextV2ReadError("read status is unavailable") from error
        search = "ready" if row is not None and row[0] == "ready" else "fallback"
        return SQLiteContextV2ReadStatus(
            available=True,
            owner_chat_id=self._scope.owner_chat_id,
            execution_count=len(self._scope.execution_ids),
            space_id=self._scope.space_id,
            space_revision=self._workspace.space.revision,
            search=search,
        )

    def read_events(
        self,
        *,
        execution_id: str,
        after: EventCursor | None = None,
        limit: int = 100,
    ) -> JournalPage:
        return self._journal(execution_id).read(after=after, limit=limit)

    def read_events_after_store_seq(
        self,
        *,
        execution_id: str,
        after_store_seq: int = 0,
        limit: int = 100,
        attempt_id: str | None = None,
    ) -> JournalPage:
        """Read an HTTP-style integer cursor inside one bound execution.

        The canonical journal cursor also carries an event id.  Product HTTP
        surfaces historically expose only ``store_seq``; resolving that
        presentation detail belongs here so hosts never query Unchain's SQLite
        schema directly.
        """

        start = _bounded_integer(
            after_store_seq,
            "after_store_seq",
            minimum=0,
            maximum=2**63 - 1,
        )
        page_limit = _bounded_integer(
            limit,
            "limit",
            minimum=1,
            maximum=10_000,
        )
        normalized_attempt = (
            _required_text(
                attempt_id,
                "attempt_id",
                maximum=512,
                identifier=True,
            )
            if attempt_id is not None
            else None
        )
        journal = self._journal(execution_id)
        try:
            with journal._store._transaction(immediate=False) as connection:
                query = (
                    "SELECT * FROM events WHERE execution_id = ? " "AND store_seq > ?"
                )
                parameters: list[object] = [journal.execution_id, start]
                if normalized_attempt is not None:
                    query += " AND attempt_id = ?"
                    parameters.append(normalized_attempt)
                query += " ORDER BY store_seq LIMIT ?"
                parameters.append(page_limit + 1)
                rows = list(connection.execute(query, tuple(parameters)))
                has_more = len(rows) > page_limit
                events = tuple(
                    journal._event_from_row(connection, row)
                    for row in rows[:page_limit]
                )
        except sqlite3.Error as error:
            raise SQLiteContextV2ReadError(
                "durable journal page is unavailable"
            ) from error
        next_cursor = (
            EventCursor(events[-1].store_seq, events[-1].event_id) if events else None
        )
        return JournalPage(
            events=events,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def read_artifact_page(
        self,
        *,
        execution_id: str,
        artifact: ArtifactRef,
        offset: int = 0,
        limit: int = 65_536,
    ) -> ArtifactReadPage:
        journal = self._journal(execution_id)
        service = self._artifacts[journal.execution_id]
        return service.read_page(artifact, offset=offset, limit=limit)

    def _unique_scoped_row(
        self,
        *,
        table: str,
        identifier_field: str,
        identifier: str,
        revision: int,
        kind: str,
    ):
        placeholders = ",".join("?" for _ in self._scope.execution_ids)
        try:
            with serialized_context_v2_database_access(self._database_path):
                connection = sqlite3.connect(
                    f"{self._database_path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=30.0,
                    isolation_level=None,
                )
                connection.row_factory = sqlite3.Row
                try:
                    connection.execute("PRAGMA query_only = ON")
                    rows = list(
                        connection.execute(
                            f"SELECT * FROM {table} WHERE {identifier_field} = ? "
                            f"AND revision = ? AND execution_id IN ({placeholders}) "
                            "LIMIT 2",
                            (
                                identifier,
                                revision,
                                *self._scope.execution_ids,
                            ),
                        )
                    )
                finally:
                    connection.close()
        except sqlite3.Error as error:
            raise SQLiteContextV2ReadError(
                f"durable {kind} descriptor is unavailable"
            ) from error
        if len(rows) != 1:
            raise SQLiteContextV2ReadScopeError(
                f"{kind} is absent or ambiguous in the bound execution scope"
            )
        return rows[0]

    def read_unique_artifact(
        self,
        *,
        ref: ResourceRef,
        offset: int = 0,
        limit: int = 65_536,
    ) -> VerifiedContentPage:
        if not isinstance(ref, ResourceRef) or ref.kind != "artifact" or ref.fragment:
            raise SQLiteContextV2ReadScopeError(
                "artifact ref is outside the bound execution scope"
            )
        row = self._unique_scoped_row(
            table="artifacts",
            identifier_field="artifact_id",
            identifier=ref.resource_id,
            revision=ref.revision,
            kind="artifact",
        )
        execution_id = str(row["execution_id"])
        artifact = self._journal(execution_id)._artifact_from_row(row)
        if artifact.ref != ref:
            raise SQLiteContextV2ReadScopeError(
                "artifact descriptor changed outside the bound scope"
            )
        page = self.read_artifact_page(
            execution_id=execution_id,
            artifact=artifact,
            offset=offset,
            limit=limit,
        )
        return VerifiedContentPage(
            ref=ref,
            media_type=artifact.media_type,
            data=page.data,
            offset=page.offset,
            total_bytes=artifact.byte_length,
            sha256=artifact.sha256,
        )

    def read_unique_checkpoint(
        self,
        *,
        ref: ResourceRef,
        offset: int = 0,
        limit: int = 65_536,
    ) -> VerifiedContentPage:
        if (
            not isinstance(ref, ResourceRef)
            or ref.kind != "checkpoint"
            or ref.fragment
            or ref.revision != 1
        ):
            raise SQLiteContextV2ReadScopeError(
                "checkpoint ref is outside the bound execution scope"
            )
        if not self._checkpoints:
            raise SQLiteContextV2ReadError(
                "official checkpoint read capability is unavailable"
            )
        row = self._unique_scoped_row(
            table="checkpoints",
            identifier_field="checkpoint_id",
            identifier=ref.resource_id,
            revision=ref.revision,
            kind="checkpoint",
        )
        execution_id = str(row["execution_id"])
        port = self._checkpoints.get(execution_id)
        if port is None:
            raise SQLiteContextV2ReadScopeError(
                "checkpoint is outside the bound execution scope"
            )
        receipt, _semantic, artifact = port._decode(row, duplicate=False)
        if receipt.checkpoint_ref != ref:
            raise SQLiteContextV2ReadScopeError(
                "checkpoint descriptor changed outside the bound scope"
            )
        data = port.read(ref=ref, offset=offset, limit=limit)
        return VerifiedContentPage(
            ref=ref,
            media_type=artifact.media_type,
            data=data,
            offset=offset,
            total_bytes=artifact.byte_length,
            sha256=artifact.sha256,
        )

    def _unique_scoped_event(self, *, ref: ResourceRef):
        if (
            not isinstance(ref, ResourceRef)
            or ref.kind != "context_event"
            or ref.revision != 1
            or ref.fragment not in {"", "content"}
        ):
            raise SQLiteContextV2ReadScopeError(
                "context event ref is outside the bound execution scope"
            )
        placeholders = ",".join("?" for _ in self._scope.execution_ids)
        try:
            with serialized_context_v2_database_access(self._database_path):
                connection = sqlite3.connect(
                    f"{self._database_path.as_uri()}?mode=ro",
                    uri=True,
                    timeout=30.0,
                    isolation_level=None,
                )
                connection.row_factory = sqlite3.Row
                try:
                    connection.execute("PRAGMA query_only = ON")
                    rows = list(
                        connection.execute(
                            "SELECT * FROM events WHERE event_id = ? "
                            f"AND execution_id IN ({placeholders}) LIMIT 2",
                            (ref.resource_id, *self._scope.execution_ids),
                        )
                    )
                    if len(rows) != 1:
                        raise SQLiteContextV2ReadScopeError(
                            "context event is absent or ambiguous in the bound scope"
                        )
                    execution_id = str(rows[0]["execution_id"])
                    event = self._journal(execution_id)._event_from_row(
                        connection,
                        rows[0],
                    )
                finally:
                    connection.close()
        except SQLiteContextV2ReadScopeError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextV2ReadError(
                "durable context event is unavailable"
            ) from error
        if event.event_id != ref.resource_id:
            raise SQLiteContextV2ReadScopeError(
                "context event descriptor changed outside the bound scope"
            )
        return event

    def _checkpoint_record(self, *, ref: ResourceRef):
        if (
            not isinstance(ref, ResourceRef)
            or ref.kind != "checkpoint"
            or ref.revision != 1
            or ref.fragment
        ):
            raise SQLiteContextV2ReadScopeError(
                "checkpoint ref is outside the bound execution scope"
            )
        if not self._checkpoints:
            raise SQLiteContextV2ReadError(
                "official checkpoint read capability is unavailable"
            )
        row = self._unique_scoped_row(
            table="checkpoints",
            identifier_field="checkpoint_id",
            identifier=ref.resource_id,
            revision=ref.revision,
            kind="checkpoint",
        )
        execution_id = str(row["execution_id"])
        port = self._checkpoints.get(execution_id)
        if port is None:
            raise SQLiteContextV2ReadScopeError(
                "checkpoint is outside the bound execution scope"
            )
        receipt, semantic, artifact = port._decode(row, duplicate=False)
        if (
            receipt.checkpoint_ref != ref
            or receipt.status is not CheckpointWriteStatus.COMMITTED
        ):
            raise SQLiteContextV2ReadScopeError(
                "checkpoint is not a committed record in the bound scope"
            )
        return execution_id, semantic["source_range"], artifact

    @staticmethod
    def _memory_content_page(
        *,
        ref: ResourceRef,
        media_type: str,
        content: bytes,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage:
        page_offset = _bounded_integer(
            offset,
            "offset",
            minimum=0,
            maximum=2**63 - 1,
        )
        page_limit = _bounded_integer(
            limit,
            "limit",
            minimum=1,
            maximum=_MAX_CONTEXT_CONTENT_PAGE_BYTES,
        )
        if page_offset > len(content):
            raise ValueError("offset exceeds the content byte length")
        return MemoryToolContentPage(
            ref=ref,
            media_type=media_type,
            data=content[page_offset : page_offset + page_limit],
            offset=page_offset,
            total_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    @staticmethod
    def _event_content(event, *, content_only: bool) -> tuple[str, bytes]:
        event_record = event.to_dict()
        if not content_only:
            return (
                "application/json",
                json.dumps(
                    event_record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
        payload = event_record["payload"]
        if "content" not in payload:
            raise SQLiteContextV2ReadScopeError(
                "context event has no disclosed content payload"
            )
        content = payload["content"]
        if isinstance(content, str):
            return "text/plain", content.encode("utf-8")
        return (
            "application/json",
            json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
        )

    def authorize_context_ref(self, *, ref: ResourceRef) -> ResourceRef:
        """Resolve one structured ref inside the immutable bound lineage."""

        if not isinstance(ref, ResourceRef):
            raise TypeError("ref must be a ResourceRef")
        if ref.kind == "artifact" and not ref.fragment:
            self.read_unique_artifact(ref=ref, offset=0, limit=1)
            return ref
        if ref.kind == "context_event" and ref.fragment in {"", "content"}:
            self._unique_scoped_event(ref=ref)
            return ref
        if ref.kind == "checkpoint" and ref.revision == 1:
            base = ResourceRef("checkpoint", ref.resource_id, 1)
            self._checkpoint_record(ref=base)
            if ref.fragment:
                match = _CHECKPOINT_EVENT_FRAGMENT.fullmatch(ref.fragment)
                if match is None:
                    raise SQLiteContextV2ReadScopeError(
                        "checkpoint fragment is outside the bound scope"
                    )
                page = self.read_checkpoint_events(
                    ref=base,
                    after_position=int(match.group(1)) - 1,
                    limit=1,
                )
                if not page["events"]:
                    raise SQLiteContextV2ReadScopeError(
                        "checkpoint event position is outside the bound scope"
                    )
            return ref
        raise SQLiteContextV2ReadScopeError(
            "context reference is outside the bound execution scope"
        )

    def read_content(
        self,
        *,
        ref: ResourceRef,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage:
        """Read artifact, event, or checkpoint bytes through one public port."""

        self.authorize_context_ref(ref=ref)
        if ref.kind == "artifact":
            page = self.read_unique_artifact(
                ref=ref,
                offset=offset,
                limit=limit,
            )
            return MemoryToolContentPage(
                ref=ref,
                media_type=page.media_type,
                data=page.data,
                offset=page.offset,
                total_bytes=page.total_bytes,
                sha256=page.sha256,
            )
        if ref.kind == "context_event":
            event = self._unique_scoped_event(ref=ref)
            if ref.fragment == "content" and "content_ref" in event.payload:
                try:
                    content_ref = ResourceRef.from_dict(event.payload["content_ref"])
                except (TypeError, ValueError) as error:
                    raise SQLiteContextV2ReadError(
                        "context event content reference is invalid"
                    ) from error
                if (
                    content_ref.kind != "artifact"
                    or content_ref.fragment
                    or content_ref not in event.resource_refs
                ):
                    raise SQLiteContextV2ReadError(
                        "context event content reference changed"
                    )
                page = self.read_unique_artifact(
                    ref=content_ref,
                    offset=offset,
                    limit=limit,
                )
                return MemoryToolContentPage(
                    ref=ref,
                    media_type=page.media_type,
                    data=page.data,
                    offset=page.offset,
                    total_bytes=page.total_bytes,
                    sha256=page.sha256,
                )
            media_type, content = self._event_content(
                event,
                content_only=ref.fragment == "content",
            )
            return self._memory_content_page(
                ref=ref,
                media_type=media_type,
                content=content,
                offset=offset,
                limit=limit,
            )
        if not ref.fragment:
            page = self.read_unique_checkpoint(
                ref=ref,
                offset=offset,
                limit=limit,
            )
            return MemoryToolContentPage(
                ref=ref,
                media_type=page.media_type,
                data=page.data,
                offset=page.offset,
                total_bytes=page.total_bytes,
                sha256=page.sha256,
            )
        position = int(_CHECKPOINT_EVENT_FRAGMENT.fullmatch(ref.fragment).group(1))
        checkpoint_ref = ResourceRef("checkpoint", ref.resource_id, 1)
        event_page = self.read_checkpoint_events(
            ref=checkpoint_ref,
            after_position=position - 1,
            limit=1,
        )
        event_ref = event_page["events"][0]["event_ref"]
        event = self._unique_scoped_event(ref=event_ref)
        media_type, content = self._event_content(event, content_only=False)
        return self._memory_content_page(
            ref=ref,
            media_type=media_type,
            content=content,
            offset=offset,
            limit=limit,
        )

    def read_checkpoint_events(
        self,
        *,
        ref: ResourceRef,
        after_position: int,
        limit: int,
    ) -> Mapping[str, Any]:
        """Page immutable journal event metadata within checkpoint coverage."""

        cursor = _bounded_integer(
            after_position,
            "after_position",
            minimum=0,
            maximum=2**63 - 1,
        )
        page_limit = _bounded_integer(
            limit,
            "limit",
            minimum=1,
            maximum=_MAX_CHECKPOINT_EVENT_PAGE_SIZE,
        )
        execution_id, raw_range, _artifact = self._checkpoint_record(ref=ref)
        source_range = (
            raw_range
            if isinstance(raw_range, EventRange)
            else EventRange.from_dict(raw_range)
        )
        journal = self._journal(execution_id)
        try:
            with journal._store._transaction(immediate=False) as connection:
                boundary_rows = list(
                    connection.execute(
                        "SELECT * FROM events WHERE execution_id = ? "
                        "AND store_seq IN (?, ?) ORDER BY store_seq",
                        (
                            execution_id,
                            source_range.start.store_seq,
                            source_range.end.store_seq,
                        ),
                    )
                )
                if not boundary_rows:
                    raise SQLiteContextV2ReadError(
                        "checkpoint event coverage is unavailable"
                    )
                first = journal._event_from_row(connection, boundary_rows[0])
                last = journal._event_from_row(connection, boundary_rows[-1])
                if (
                    first.event_id != source_range.start.event_id
                    or last.event_id != source_range.end.event_id
                ):
                    raise SQLiteContextV2ReadError(
                        "checkpoint event coverage changed"
                    )
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM events WHERE execution_id = ? "
                        "AND store_seq BETWEEN ? AND ?",
                        (
                            execution_id,
                            source_range.start.store_seq,
                            source_range.end.store_seq,
                        ),
                    ).fetchone()[0]
                )
                rows = list(
                    connection.execute(
                        "SELECT * FROM events WHERE execution_id = ? "
                        "AND store_seq BETWEEN ? AND ? ORDER BY store_seq "
                        "LIMIT ? OFFSET ?",
                        (
                            execution_id,
                            source_range.start.store_seq,
                            source_range.end.store_seq,
                            page_limit + 1,
                            cursor,
                        ),
                    )
                )
                has_more = len(rows) > page_limit
                events = tuple(
                    journal._event_from_row(connection, row)
                    for row in rows[:page_limit]
                )
        except SQLiteContextV2ReadError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextV2ReadError(
                "checkpoint event page is unavailable"
            ) from error
        visible_events = tuple(
            MappingProxyType(
                {
                    "position": cursor + index,
                    "store_seq": event.store_seq,
                    "event_type": event.event_type,
                    "event_ref": ResourceRef("context_event", event.event_id, 1),
                    "content_ref": ResourceRef(
                        "checkpoint",
                        ref.resource_id,
                        1,
                        f"event/{cursor + index}",
                    ),
                }
            )
            for index, event in enumerate(events, start=1)
        )
        next_position = cursor + len(visible_events)
        return MappingProxyType(
            {
                "checkpoint_ref": ref,
                "coverage": MappingProxyType(
                    {
                        "first_store_seq": source_range.start.store_seq,
                        "last_store_seq": source_range.end.store_seq,
                        "ceiling_position": total,
                    }
                ),
                "after_position": cursor,
                "next_after_position": next_position,
                "has_more": has_more,
                "events": visible_events,
            }
        )

    def read_workspace_content(
        self,
        *,
        ref: ResourceRef,
        offset: int = 0,
        limit: int = 32 * 1024,
    ) -> VerifiedContentPage:
        entry = self.get_workspace_entry(ref=ref)
        page_offset = _bounded_integer(
            offset,
            "offset",
            minimum=0,
            maximum=2**63 - 1,
        )
        page_limit = _bounded_integer(
            limit,
            "limit",
            minimum=1,
            maximum=65_536,
        )
        if entry.kind is MemoryEntryKind.FOLDER:
            raise RepositoryNotFoundError("folder entries do not have readable content")
        if entry.kind is MemoryEntryKind.LINK:
            content = entry.link_url.encode("utf-8")
            media_type = "text/uri-list"
        else:
            if entry.content_ref is None:
                raise RepositoryNotFoundError("workspace entry content is unavailable")
            full = self._workspace.read_content(
                ref=entry.content_ref,
                offset=0,
                limit=1024 * 1024,
            )
            if full.offset != 0 or full.has_more:
                raise SQLiteContextV2ReadError(
                    "workspace content exceeds the verified P0 read bound"
                )
            content = full.data
            media_type = entry.media_type or full.media_type
        if page_offset > len(content):
            raise ValueError("offset exceeds the workspace content length")
        return VerifiedContentPage(
            ref=ref,
            media_type=media_type,
            data=content[page_offset : page_offset + page_limit],
            offset=page_offset,
            total_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def _workspace_page(
        self,
        *,
        parent_path: str,
        recursive: bool,
        include_deleted: bool,
        limit: int,
        cursor: str | None,
    ) -> MemoryEntryPage:
        parent = canonical_parent_path(parent_path)
        if not isinstance(recursive, bool):
            raise TypeError("recursive must be a boolean")
        if not isinstance(include_deleted, bool):
            raise TypeError("include_deleted must be a boolean")
        page_limit = _bounded_integer(
            limit,
            "limit",
            minimum=1,
            maximum=_MAX_LIST_RESULTS,
        )
        if cursor is not None:
            cursor = _required_text(cursor, "cursor", identifier=True)
        entries: list[MemoryEntry] = []
        repository_cursor = None
        seen_cursors: set[str] = set()
        scanned = 0
        while scanned < _MAX_LIST_SCAN:
            page = self._workspace.list_entries(
                parent_path=parent,
                include_deleted=include_deleted,
                limit=min(_SCAN_PAGE_SIZE, _MAX_LIST_SCAN - scanned),
                cursor=repository_cursor,
            )
            scanned += len(page.entries)
            prefix = parent.rstrip("/") + "/"
            for entry in page.entries:
                if entry.space_id != self._scope.space_id:
                    raise SQLiteContextV2ReadScopeError(
                        "workspace returned an entry outside the bound scope"
                    )
                entry_parent = entry.path.rsplit("/", 1)[0] or "/"
                if recursive or entry_parent == parent:
                    if parent == "/" or entry.path.startswith(prefix):
                        entries.append(entry)
            if not page.has_more:
                break
            if page.next_cursor is None or page.next_cursor in seen_cursors:
                raise SQLiteContextV2ReadError("workspace pagination did not advance")
            seen_cursors.add(page.next_cursor)
            repository_cursor = page.next_cursor
        if page.has_more:
            raise SQLiteContextV2ReadError(
                "workspace listing exceeds the safe scan bound"
            )
        entries.sort(key=lambda entry: (entry.path.casefold(), entry.entry_id))
        start = 0
        if cursor is not None:
            matches = [
                index for index, entry in enumerate(entries) if entry.entry_id == cursor
            ]
            if not matches:
                raise SQLiteContextV2ReadScopeError(
                    "cursor is outside the bound workspace listing"
                )
            start = matches[0] + 1
        selected = tuple(entries[start : start + page_limit])
        has_more = start + len(selected) < len(entries)
        return MemoryEntryPage(
            entries=selected,
            next_cursor=selected[-1].entry_id if selected and has_more else None,
            has_more=has_more,
        )

    def list_workspace(
        self,
        *,
        parent_path: str = "/",
        include_deleted: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> MemoryEntryPage:
        return self._workspace_page(
            parent_path=parent_path,
            recursive=False,
            include_deleted=include_deleted,
            limit=limit,
            cursor=cursor,
        )

    def list_workspace_children(
        self,
        *,
        parent_path: str = "/",
        expected_space_revision: int,
        limit: int = 100,
        cursor: str | None = None,
    ) -> MemoryChildPage:
        return self._workspace.list_children(
            parent_path=parent_path,
            expected_space_revision=expected_space_revision,
            limit=limit,
            cursor=cursor,
        )

    def list_workspace_projection_points(self, **kwargs) -> VectorProjectionPage:
        """Read current, authorized, persisted projection receipts only."""

        return self._workspace.list_projection_points(**kwargs)

    def current_workspace_projection_contract(
        self,
        *,
        backend_identity: str,
    ) -> dict[str, Any] | None:
        return self._workspace.current_projection_contract(
            backend_identity=backend_identity,
        )

    def workspace_tree(
        self,
        *,
        parent_path: str = "/",
        include_deleted: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> MemoryEntryPage:
        return self._workspace_page(
            parent_path=parent_path,
            recursive=True,
            include_deleted=include_deleted,
            limit=limit,
            cursor=cursor,
        )

    def get_workspace_entry(
        self,
        *,
        entry_id: str | None = None,
        ref: ResourceRef | None = None,
    ) -> MemoryEntry:
        if (entry_id is None) == (ref is None):
            raise ValueError("provide exactly one of entry_id or ref")
        if ref is not None:
            if (
                not isinstance(ref, ResourceRef)
                or ref.kind != "memory"
                or ref.fragment != self._scope.space_id
            ):
                raise SQLiteContextV2ReadScopeError(
                    "memory reference is outside the bound workspace scope"
                )
            entry = self._workspace.read_entry(ref=ref)
        else:
            entry = self._workspace.read_current_entry(
                entry_id=_required_text(entry_id, "entry_id", identifier=True)
            )
        if entry.space_id != self._scope.space_id:
            raise SQLiteContextV2ReadScopeError(
                "workspace returned an entry outside the bound scope"
            )
        return entry

    def search_workspace(
        self,
        query: str,
        *,
        limit: int = 20,
    ) -> WorkspaceSearchResult:
        return self._search.search(query, limit=limit)


__all__ = [
    "BoundSQLiteContextV2ReadService",
    "ContextV2ReadScope",
    "SQLiteContextV2ReadError",
    "SQLiteContextV2ReadScopeError",
    "SQLiteContextV2ReadService",
    "SQLiteContextV2ReadStatus",
    "SQLiteContextV2StoreReadStatus",
    "VerifiedContentPage",
    "read_sqlite_context_v2_store_status",
]
