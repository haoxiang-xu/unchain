"""Durable SQLite/CAS repositories for the Memory V2 workspace slice.

The store deliberately shares the Context V2 database and object directory,
while keeping workspace receipts separate from execution-journal receipts.
Every repository returned here is bound to one host-selected scope; callers do
not get to supply a different chat or space on individual operations.
"""

from __future__ import annotations

import hashlib
import base64
import json
import math
import os
import re
import sqlite3
import tempfile
import unicodedata
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from unchain.context import PinnedTaskState
from unchain.journal import OperationRef, ResourceRef
from unchain.journal.models import _bounded_int, _required_text
from unchain.memory.workspace import (
    MemoryEntry,
    MemoryChildEntry,
    MemoryChildPage,
    MemoryEntryKind,
    MemoryEntryPage,
    MemoryLink,
    MemorySpace,
    VectorProjectionPage,
    VectorProjectionPoint,
)
from unchain.memory.workspace.paths import canonical_parent_path
from unchain.memory.workspace.ports import (
    BoundMemoryWorkspaceRepository,
    BoundPinnedTaskStateRepository,
    BoundWorkspaceContentRepository,
    BoundWorkspaceHistoryRepository,
    BoundWorkspaceHierarchyRepository,
    BoundWorkspaceProjectionRepository,
    BoundWorkspaceLinkRepository,
    BoundWorkspaceMutationRepository,
    RepositoryConflictError,
    RepositoryNotFoundError,
    RepositoryScopeError,
    RepositorySearchUnavailableError,
    WorkspaceContentPage,
    WorkspaceLinkRequest,
    WorkspaceMutationRequest,
    WorkspaceRepositoryError,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_REPOSITORY_PAGE = 10_000
_MAX_CHILD_PAGE = 500
_CHILD_ORDER_VERSION = "unchain.path_key_entry_id.v1"
_PROJECTION_ALGORITHM = "cosine_hash_2d_v1"


class SQLiteMemoryV2StoreError(WorkspaceRepositoryError):
    """Base failure for the SQLite Memory V2 persistence slice."""


class SQLiteMemoryV2StoreIntegrityError(SQLiteMemoryV2StoreError):
    """Durable metadata or content-addressed bytes failed verification."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise SQLiteMemoryV2StoreIntegrityError(
            "durable memory record is not canonical JSON"
        ) from exc


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _positive_limit(value: object, *, maximum: int = _MAX_REPOSITORY_PAGE) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"limit must be between 1 and {maximum}")
    return value


def _non_negative_offset(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("offset must be a non-negative integer")
    return value


def _path_key(path: str) -> str:
    return unicodedata.normalize("NFKC", path).casefold()


def _name_key(name: str) -> str:
    return unicodedata.normalize("NFKC", name).casefold()


def _child_cursor_encode(
    *,
    space_id: str,
    parent_path: str,
    space_revision: int,
    path_key: str,
    entry_id: str,
) -> str:
    payload = _canonical_json_bytes(
        {
            "schema": "unchain.memory_child_cursor.v1",
            "space_id": space_id,
            "parent_path": parent_path,
            "space_revision": space_revision,
            "order_version": _CHILD_ORDER_VERSION,
            "path_key": path_key,
            "entry_id": entry_id,
        }
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _child_cursor_decode(
    value: str,
    *,
    space_id: str,
    parent_path: str,
    space_revision: int,
) -> tuple[str, str]:
    token = _required_text(value, "cursor", maximum=8192)
    try:
        padding = "=" * (-len(token) % 4)
        raw = base64.b64decode(token + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RepositoryScopeError("child cursor is invalid") from exc
    expected = {
        "schema",
        "space_id",
        "parent_path",
        "space_revision",
        "order_version",
        "path_key",
        "entry_id",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RepositoryScopeError("child cursor contract is invalid")
    if (
        payload["schema"] != "unchain.memory_child_cursor.v1"
        or payload["space_id"] != space_id
        or payload["parent_path"] != parent_path
        or payload["space_revision"] != space_revision
        or payload["order_version"] != _CHILD_ORDER_VERSION
    ):
        raise RepositoryConflictError("child cursor snapshot changed")
    return (
        _required_text(payload["path_key"], "cursor.path_key", maximum=2048),
        _required_text(payload["entry_id"], "cursor.entry_id", identifier=True),
    )


def _projection_cursor_encode(*, space_id: str, backend_identity: str, basis_id: str, basis_version: int, corpus_epoch: int, chunk_id: str) -> str:
    payload = _canonical_json_bytes(
        {
            "schema": "unchain.vector_projection_cursor.v1",
            "space_id": space_id,
            "backend_identity": backend_identity,
            "basis_id": basis_id,
            "basis_version": basis_version,
            "corpus_epoch": corpus_epoch,
            "chunk_id": chunk_id,
        }
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _projection_cursor_decode(value: str, *, space_id: str, backend_identity: str, basis_id: str, basis_version: int, corpus_epoch: int) -> str:
    token = _required_text(value, "cursor", maximum=8192)
    try:
        padding = "=" * (-len(token) % 4)
        payload = json.loads(
            base64.b64decode(token + padding, altchars=b"-_", validate=True).decode("utf-8")
        )
    except (UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RepositoryScopeError("projection cursor is invalid") from exc
    expected = {"schema", "space_id", "backend_identity", "basis_id", "basis_version", "corpus_epoch", "chunk_id"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise RepositoryScopeError("projection cursor contract is invalid")
    if (
        payload["schema"] != "unchain.vector_projection_cursor.v1"
        or payload["space_id"] != space_id
        or payload["backend_identity"] != backend_identity
        or payload["basis_id"] != basis_id
        or payload["basis_version"] != basis_version
        or payload["corpus_epoch"] != corpus_epoch
    ):
        raise RepositoryConflictError("projection cursor snapshot changed")
    return _required_text(payload["chunk_id"], "cursor.chunk_id", identifier=True)


class SQLiteMemoryV2Store:
    """Own Memory V2 tables in the shared Context V2 SQLite/CAS data plane."""

    def __init__(
        self,
        *,
        database_path: str | os.PathLike[str],
        object_directory: str | os.PathLike[str],
    ) -> None:
        self.database_path = Path(database_path)
        self.object_directory = Path(object_directory)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.object_directory.mkdir(parents=True, exist_ok=True)
        self._fts_available = False
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

    @contextmanager
    def _transaction(self, *, immediate: bool) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).casefold() != "wal":
                raise SQLiteMemoryV2StoreIntegrityError(
                    "SQLite refused WAL journal mode"
                )
            connection.executescript(
                """
                BEGIN IMMEDIATE;

                CREATE TABLE IF NOT EXISTS memory_v2_schema (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO memory_v2_schema(version) VALUES (1);

                CREATE TABLE IF NOT EXISTS objects (
                    sha256 TEXT PRIMARY KEY,
                    byte_length INTEGER NOT NULL CHECK(byte_length >= 0)
                );

                CREATE TABLE IF NOT EXISTS spaces (
                    space_id TEXT PRIMARY KEY,
                    owner_chat_id TEXT NOT NULL,
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    space_json BLOB NOT NULL,
                    space_sha256 TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS entries (
                    space_id TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
                    path_key TEXT NOT NULL,
                    name_key TEXT NOT NULL,
                    deleted INTEGER NOT NULL CHECK(deleted IN (0, 1)),
                    updated_seq INTEGER NOT NULL CHECK(updated_seq >= 0),
                    PRIMARY KEY (space_id, entry_id),
                    FOREIGN KEY (space_id) REFERENCES spaces(space_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_entries_active_path
                    ON entries(space_id, path_key) WHERE deleted = 0;
                CREATE INDEX IF NOT EXISTS idx_entries_listing
                    ON entries(space_id, deleted, path_key, entry_id);

                CREATE TABLE IF NOT EXISTS entry_revisions (
                    space_id TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    path_key TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    entry_json BLOB NOT NULL,
                    entry_sha256 TEXT NOT NULL,
                    content_resource_id TEXT,
                    object_sha256 TEXT,
                    byte_length INTEGER,
                    PRIMARY KEY (space_id, entry_id, revision),
                    FOREIGN KEY (space_id) REFERENCES spaces(space_id),
                    FOREIGN KEY (object_sha256) REFERENCES objects(sha256),
                    CHECK (
                        (content_resource_id IS NULL
                            AND object_sha256 IS NULL AND byte_length IS NULL)
                        OR
                        (content_resource_id IS NOT NULL
                            AND object_sha256 IS NOT NULL AND byte_length >= 0)
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_entry_revision_content
                    ON entry_revisions(
                        space_id, content_resource_id, revision
                    ) WHERE content_resource_id IS NOT NULL;

                CREATE TABLE IF NOT EXISTS links (
                    space_id TEXT NOT NULL,
                    link_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    source_entry_id TEXT NOT NULL,
                    source_revision INTEGER NOT NULL CHECK(source_revision >= 1),
                    target_entry_id TEXT NOT NULL,
                    target_revision INTEGER NOT NULL CHECK(target_revision >= 1),
                    relation TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    link_json BLOB NOT NULL,
                    link_sha256 TEXT NOT NULL,
                    source_refs_json BLOB NOT NULL,
                    source_refs_sha256 TEXT NOT NULL,
                    PRIMARY KEY (space_id, link_id, revision),
                    FOREIGN KEY (space_id) REFERENCES spaces(space_id)
                );
                CREATE INDEX IF NOT EXISTS idx_links_source
                    ON links(space_id, source_entry_id, source_revision, link_id);
                CREATE INDEX IF NOT EXISTS idx_links_target
                    ON links(space_id, target_entry_id, target_revision, link_id);

                CREATE TABLE IF NOT EXISTS memory_operation_receipts (
                    scope_kind TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    target_kind TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    result_id TEXT NOT NULL,
                    result_revision INTEGER NOT NULL CHECK(result_revision >= 1),
                    PRIMARY KEY (scope_kind, scope_id, operation_id)
                );

                CREATE TABLE IF NOT EXISTS task_state_heads (
                    binding_id TEXT NOT NULL,
                    state_id TEXT NOT NULL,
                    current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
                    PRIMARY KEY (binding_id, state_id)
                );
                CREATE TABLE IF NOT EXISTS task_state_revisions (
                    binding_id TEXT NOT NULL,
                    state_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    operation_id TEXT NOT NULL,
                    state_json BLOB NOT NULL,
                    state_sha256 TEXT NOT NULL,
                    PRIMARY KEY (binding_id, state_id, revision)
                );

                CREATE TABLE IF NOT EXISTS index_state (
                    index_name TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 1 CHECK(revision >= 1),
                    detail TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (index_name, scope_id)
                );

                CREATE TABLE IF NOT EXISTS vector_projection_receipts (
                    space_id TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    entry_revision INTEGER NOT NULL CHECK(entry_revision >= 1),
                    backend_identity TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    basis_id TEXT NOT NULL,
                    basis_version INTEGER NOT NULL CHECK(basis_version >= 1),
                    algorithm TEXT NOT NULL,
                    dimension INTEGER NOT NULL CHECK(dimension >= 1),
                    expected_chunks INTEGER NOT NULL CHECK(expected_chunks >= 0),
                    indexed_chunks INTEGER NOT NULL CHECK(indexed_chunks >= 0),
                    content_digest TEXT NOT NULL,
                    complete INTEGER NOT NULL CHECK(complete IN (0, 1)),
                    PRIMARY KEY (
                        space_id, entry_id, entry_revision,
                        backend_identity, basis_id, basis_version
                    )
                );
                CREATE TABLE IF NOT EXISTS vector_projection_points (
                    space_id TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    entry_revision INTEGER NOT NULL CHECK(entry_revision >= 1),
                    backend_identity TEXT NOT NULL,
                    basis_id TEXT NOT NULL,
                    basis_version INTEGER NOT NULL CHECK(basis_version >= 1),
                    chunk_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    embedding_digest TEXT NOT NULL,
                    external_receipt_id TEXT NOT NULL,
                    PRIMARY KEY (
                        space_id, backend_identity, basis_id,
                        basis_version, chunk_id
                    )
                );
                CREATE INDEX IF NOT EXISTS idx_vector_projection_page
                    ON vector_projection_points(
                        space_id, backend_identity, basis_id,
                        basis_version, chunk_id
                    );
                CREATE TABLE IF NOT EXISTS vector_projection_watermarks (
                    space_id TEXT NOT NULL,
                    backend_identity TEXT NOT NULL,
                    chunker_version TEXT NOT NULL,
                    basis_id TEXT NOT NULL,
                    basis_version INTEGER NOT NULL CHECK(basis_version >= 1),
                    algorithm TEXT NOT NULL,
                    dimension INTEGER NOT NULL CHECK(dimension >= 1),
                    corpus_epoch INTEGER NOT NULL CHECK(corpus_epoch >= 1),
                    PRIMARY KEY (space_id, backend_identity, basis_id, basis_version)
                );

                COMMIT;
                """
            )
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise SQLiteMemoryV2StoreIntegrityError(
                    "SQLite quick_check failed for Memory V2"
                )
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        self._initialize_fts()

    def _initialize_fts(self) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS workspace_entries_fts
                USING fts5(
                    space_id UNINDEXED,
                    entry_id UNINDEXED,
                    path,
                    name,
                    description,
                    tags,
                    tokenize = 'unicode61 remove_diacritics 2'
                )
                """
            )
            unavailable = connection.execute(
                "SELECT 1 FROM index_state WHERE index_name = 'workspace_fts' "
                "AND status != 'ready' LIMIT 1"
            ).fetchone()
            count = connection.execute(
                "SELECT COUNT(*) FROM workspace_entries_fts"
            ).fetchone()[0]
            active_count = connection.execute(
                "SELECT COUNT(*) FROM entries WHERE deleted = 0"
            ).fetchone()[0]
            if unavailable is not None or count != active_count:
                connection.execute("DELETE FROM workspace_entries_fts")
                rows = connection.execute(
                    """
                    SELECT r.entry_json, r.entry_sha256
                    FROM entries AS e
                    JOIN entry_revisions AS r
                      ON r.space_id = e.space_id
                     AND r.entry_id = e.entry_id
                     AND r.revision = e.current_revision
                    WHERE e.deleted = 0
                    ORDER BY e.space_id, e.path_key, e.entry_id
                    """
                )
                for row in rows:
                    entry = self._decode_entry_record(row)
                    self._insert_fts_row(connection, entry)
            connection.execute(
                """
                INSERT INTO index_state(index_name, scope_id, status, revision, detail)
                SELECT 'workspace_fts', space_id, 'ready', revision, ''
                FROM spaces WHERE 1
                ON CONFLICT(index_name, scope_id) DO UPDATE SET
                    status = 'ready',
                    revision = excluded.revision,
                    detail = ''
                """
            )
            connection.commit()
            self._fts_available = True
        except sqlite3.Error:
            connection.rollback()
            self._fts_available = False
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO index_state(index_name, scope_id, status, revision, detail)
                    SELECT 'workspace_fts', space_id, 'unavailable', revision,
                           'sqlite_fts_unavailable'
                    FROM spaces WHERE 1
                    ON CONFLICT(index_name, scope_id) DO UPDATE SET
                        status = 'unavailable',
                        revision = excluded.revision,
                        detail = 'sqlite_fts_unavailable'
                    """
                )
                connection.commit()
            except sqlite3.Error:
                connection.rollback()
        finally:
            connection.close()

    def bind_workspace(
        self,
        *,
        space: MemorySpace,
        owner_chat_id: str | None = None,
    ) -> _SQLiteBoundMemoryWorkspaceRepository:
        if not isinstance(space, MemorySpace):
            raise TypeError("space must be a MemorySpace")
        if owner_chat_id is None:
            if space.namespace == "chat":
                raise RepositoryScopeError("chat workspace requires an owner binding")
            owner = ""
        else:
            owner = _required_text(
                owner_chat_id,
                "owner_chat_id",
                maximum=512,
                identifier=True,
            )
        try:
            with self._transaction(immediate=True) as connection:
                row = connection.execute(
                    "SELECT * FROM spaces WHERE space_id = ?",
                    (space.space_id,),
                ).fetchone()
                if row is None:
                    if space.revision != 1:
                        raise RepositoryConflictError(
                            "new workspace must start at revision one"
                        )
                    encoded = _canonical_json_bytes(space.to_dict())
                    connection.execute(
                        """
                        INSERT INTO spaces(
                            space_id, owner_chat_id, namespace, name, description,
                            revision, space_json, space_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            space.space_id,
                            owner,
                            space.namespace,
                            space.name,
                            space.description,
                            space.revision,
                            encoded,
                            _sha256(encoded),
                        ),
                    )
                    persisted = space
                else:
                    persisted = self._space_from_row(row)
                    if row["owner_chat_id"] != owner:
                        raise RepositoryScopeError(
                            "workspace belongs to another owner binding"
                        )
                    if (
                        persisted.namespace != space.namespace
                        or persisted.name != space.name
                        or persisted.description != space.description
                    ):
                        raise RepositoryConflictError(
                            "workspace identity metadata changed"
                        )
                status = "ready" if self._fts_available else "unavailable"
                detail = "" if self._fts_available else "sqlite_fts_unavailable"
                connection.execute(
                    """
                    INSERT INTO index_state(
                        index_name, scope_id, status, revision, detail
                    ) VALUES ('workspace_fts', ?, ?, ?, ?)
                    ON CONFLICT(index_name, scope_id) DO UPDATE SET
                        status = excluded.status,
                        revision = excluded.revision,
                        detail = excluded.detail
                    """,
                    (persisted.space_id, status, persisted.revision, detail),
                )
        except sqlite3.IntegrityError as exc:
            raise RepositoryConflictError("workspace binding conflicted") from exc
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite workspace binding failed") from exc
        return _SQLiteBoundMemoryWorkspaceRepository(self, persisted, owner)

    def bind_task_state(
        self,
        *,
        binding_id: str,
        state_id: str | None = None,
    ) -> _SQLiteBoundPinnedTaskStateRepository:
        return _SQLiteBoundPinnedTaskStateRepository(
            self,
            binding_id=binding_id,
            state_id=state_id,
        )

    @staticmethod
    def _decode_entry_record(row: sqlite3.Row) -> MemoryEntry:
        raw = bytes(row["entry_json"])
        if _sha256(raw) != row["entry_sha256"]:
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory entry descriptor digest changed"
            )
        try:
            decoded = json.loads(raw.decode("utf-8"))
            entry = MemoryEntry.from_dict(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory entry descriptor is malformed"
            ) from exc
        if _canonical_json_bytes(entry.to_dict()) != raw:
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory entry descriptor is not canonical"
            )
        return entry

    def _space_from_row(self, row: sqlite3.Row) -> MemorySpace:
        raw = bytes(row["space_json"])
        if _sha256(raw) != row["space_sha256"]:
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory space descriptor digest changed"
            )
        try:
            space = MemorySpace.from_dict(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory space descriptor is malformed"
            ) from exc
        if (
            _canonical_json_bytes(space.to_dict()) != raw
            or space.space_id != row["space_id"]
            or space.namespace != row["namespace"]
            or space.name != row["name"]
            or space.description != row["description"]
            or space.revision != row["revision"]
        ):
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory space indexed metadata changed"
            )
        return space

    def _load_space(self, *, space_id: str, owner_chat_id: str) -> MemorySpace:
        try:
            with self._transaction(immediate=False) as connection:
                row = connection.execute(
                    "SELECT * FROM spaces WHERE space_id = ?",
                    (space_id,),
                ).fetchone()
                if row is None or row["owner_chat_id"] != owner_chat_id:
                    raise RepositoryScopeError("workspace is outside the bound scope")
                return self._space_from_row(row)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite workspace read failed") from exc

    def _object_path(self, digest: str) -> Path:
        if _SHA256_RE.fullmatch(digest) is None:
            raise SQLiteMemoryV2StoreIntegrityError("object digest is not canonical")
        return self.object_directory / digest

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        # Windows cannot open directories with os.open. Object contents are
        # already fsynced before publication; directory fsync is POSIX-only.
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _install_object(self, content: bytes) -> tuple[str, int]:
        if type(content) is not bytes:
            raise TypeError("workspace content must be exact bytes")
        digest = _sha256(content)
        target = self._object_path(digest)
        if target.exists():
            self._read_object(digest=digest, byte_length=len(content))
            return digest, len(content)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".memory-v2-object-",
            dir=self.object_directory,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                # Another writer installed this content digest first; verify its bytes and digest below.
                pass
            self._fsync_directory(self.object_directory)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                # Cleanup is complete when the temporary object is already absent.
                pass
        self._read_object(digest=digest, byte_length=len(content))
        return digest, len(content)

    def _read_object(self, *, digest: str, byte_length: int) -> bytes:
        try:
            content = self._object_path(digest).read_bytes()
        except OSError as exc:
            raise SQLiteMemoryV2StoreIntegrityError(
                "workspace object is missing or unreadable"
            ) from exc
        if len(content) != byte_length or _sha256(content) != digest:
            raise SQLiteMemoryV2StoreIntegrityError(
                "workspace object length or digest changed"
            )
        return content

    @staticmethod
    def _insert_fts_row(connection: sqlite3.Connection, entry: MemoryEntry) -> None:
        connection.execute(
            """
            INSERT INTO workspace_entries_fts(
                space_id, entry_id, path, name, description, tags
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.space_id,
                entry.entry_id,
                entry.path,
                entry.name,
                entry.description,
                " ".join(entry.tags),
            ),
        )

    def _refresh_fts_entry(
        self,
        connection: sqlite3.Connection,
        entry: MemoryEntry,
    ) -> None:
        if not self._fts_available:
            return
        try:
            connection.execute(
                "DELETE FROM workspace_entries_fts WHERE space_id = ? AND entry_id = ?",
                (entry.space_id, entry.entry_id),
            )
            if not entry.deleted:
                self._insert_fts_row(connection, entry)
            connection.execute(
                """
                UPDATE index_state SET status = 'ready', revision = ?, detail = ''
                WHERE index_name = 'workspace_fts' AND scope_id = ?
                """,
                (entry.updated_seq, entry.space_id),
            )
        except sqlite3.OperationalError:
            self._fts_available = False
            connection.execute(
                """
                UPDATE index_state
                SET status = 'unavailable', detail = 'sqlite_fts_unavailable'
                WHERE index_name = 'workspace_fts'
                """
            )

    @staticmethod
    def _claim_receipt(
        connection: sqlite3.Connection,
        *,
        scope_kind: str,
        scope_id: str,
        operation: OperationRef,
        target_kind: str,
        target_key: str,
        result_id: str,
        result_revision: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO memory_operation_receipts(
                scope_kind, scope_id, operation_id, payload_sha256,
                target_kind, target_key, result_id, result_revision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scope_kind,
                scope_id,
                operation.operation_id,
                operation.payload_sha256,
                target_kind,
                target_key,
                result_id,
                result_revision,
            ),
        )


class _SQLiteBoundMemoryWorkspaceRepository(
    BoundMemoryWorkspaceRepository,
    BoundWorkspaceHierarchyRepository,
    BoundWorkspaceProjectionRepository,
    BoundWorkspaceMutationRepository,
    BoundWorkspaceContentRepository,
    BoundWorkspaceHistoryRepository,
    BoundWorkspaceLinkRepository,
):
    """All workspace ports bound to one durable chat/space capability."""

    def __init__(
        self,
        store: SQLiteMemoryV2Store,
        space: MemorySpace,
        owner_chat_id: str,
    ) -> None:
        BoundMemoryWorkspaceRepository.__init__(self, space)
        # All remaining workspace ports share this same capability initializer.
        BoundWorkspaceHierarchyRepository.__init__(self, space)
        self._store = store
        self._owner_chat_id = owner_chat_id

    @property
    def space(self) -> MemorySpace:
        current = self._store._load_space(
            space_id=self._space.space_id,
            owner_chat_id=self._owner_chat_id,
        )
        self._space = current
        return current

    @property
    def _space_id(self) -> str:
        return self._space.space_id

    def _receipt(
        self,
        connection: sqlite3.Connection,
        *,
        operation: OperationRef,
        target_kind: str,
        target_key: str,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """
            SELECT * FROM memory_operation_receipts
            WHERE scope_kind = 'workspace' AND scope_id = ? AND operation_id = ?
            """,
            (self._space_id, operation.operation_id),
        ).fetchone()
        if row is None:
            return None
        if (
            row["payload_sha256"] != operation.payload_sha256
            or row["target_kind"] != target_kind
            or row["target_key"] != target_key
        ):
            raise RepositoryConflictError("operation payload or durable target changed")
        return row

    def _entry_from_row(self, row: sqlite3.Row) -> MemoryEntry:
        entry = self._store._decode_entry_record(row)
        if entry.space_id != self._space_id:
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory entry crossed its durable space"
            )
        if (
            entry.entry_id != row["entry_id"]
            or entry.revision != row["revision"]
            or _path_key(entry.path) != row["path_key"]
        ):
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory entry indexed metadata changed"
            )
        has_object = row["object_sha256"] is not None
        if has_object != (entry.content_ref is not None):
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory entry content reference changed"
            )
        if entry.content_ref is not None and (
            entry.content_ref.kind != "memory_content"
            or entry.content_ref.resource_id != row["content_resource_id"]
            or entry.content_ref.revision > entry.revision
            or entry.content_ref.fragment != self._space_id
        ):
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory content reference is inconsistent"
            )
        return entry

    def _entry_revision_row(
        self,
        connection: sqlite3.Connection,
        *,
        entry_id: str,
        revision: int,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM entry_revisions
            WHERE space_id = ? AND entry_id = ? AND revision = ?
            """,
            (self._space_id, entry_id, revision),
        ).fetchone()

    def _current_entry_row(
        self,
        connection: sqlite3.Connection,
        *,
        entry_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT r.*
            FROM entries AS e
            JOIN entry_revisions AS r
              ON r.space_id = e.space_id
             AND r.entry_id = e.entry_id
             AND r.revision = e.current_revision
            WHERE e.space_id = ? AND e.entry_id = ?
            """,
            (self._space_id, entry_id),
        ).fetchone()

    def _require_memory_ref(self, ref: ResourceRef) -> None:
        if not isinstance(ref, ResourceRef):
            raise TypeError("ref must be a ResourceRef")
        if ref.kind != "memory" or ref.fragment != self._space_id:
            raise RepositoryScopeError(
                "memory reference is outside the bound workspace"
            )

    def _load_space_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM spaces WHERE space_id = ?",
            (self._space_id,),
        ).fetchone()
        if row is None or row["owner_chat_id"] != self._owner_chat_id:
            raise RepositoryScopeError("workspace is outside the bound owner scope")
        self._store._space_from_row(row)
        return row

    def _advance_space(
        self,
        connection: sqlite3.Connection,
        *,
        current_row: sqlite3.Row,
        expected_revision: int,
    ) -> MemorySpace:
        current = self._store._space_from_row(current_row)
        advanced = replace(current, revision=expected_revision + 1)
        encoded = _canonical_json_bytes(advanced.to_dict())
        updated = connection.execute(
            """
            UPDATE spaces
            SET revision = ?, space_json = ?, space_sha256 = ?
            WHERE space_id = ? AND owner_chat_id = ? AND revision = ?
            """,
            (
                advanced.revision,
                encoded,
                _sha256(encoded),
                self._space_id,
                self._owner_chat_id,
                expected_revision,
            ),
        )
        if updated.rowcount != 1:
            raise RepositoryConflictError("space revision changed")
        return advanced

    def list_entries(
        self,
        *,
        parent_path: str = "/",
        include_deleted: bool = False,
        limit: int = 100,
        cursor: str | None = None,
    ) -> MemoryEntryPage:
        parent = canonical_parent_path(parent_path)
        if not isinstance(include_deleted, bool):
            raise TypeError("include_deleted must be a boolean")
        page_limit = _positive_limit(limit)
        if cursor is not None:
            cursor = _required_text(cursor, "cursor", identifier=True)
        prefix = _path_key(parent).rstrip("/") + "/"
        try:
            with self._store._transaction(immediate=False) as connection:
                rows = list(
                    connection.execute(
                        """
                        SELECT r.*
                        FROM entries AS e
                        JOIN entry_revisions AS r
                          ON r.space_id = e.space_id
                         AND r.entry_id = e.entry_id
                         AND r.revision = e.current_revision
                        WHERE e.space_id = ?
                          AND (? OR e.deleted = 0)
                          AND (? = '/' OR substr(e.path_key, 1, length(?)) = ?)
                        ORDER BY e.path_key, e.entry_id
                        """,
                        (
                            self._space_id,
                            int(include_deleted),
                            parent,
                            prefix,
                            prefix,
                        ),
                    )
                )
                entries = tuple(self._entry_from_row(row) for row in rows)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite workspace listing failed") from exc
        start = 0
        if cursor is not None:
            matches = [
                index for index, entry in enumerate(entries) if entry.entry_id == cursor
            ]
            if not matches:
                raise RepositoryScopeError(
                    "cursor is outside the bound workspace listing"
                )
            start = matches[0] + 1
        selected = entries[start : start + page_limit]
        has_more = start + len(selected) < len(entries)
        return MemoryEntryPage(
            entries=selected,
            next_cursor=selected[-1].entry_id if selected and has_more else None,
            has_more=has_more,
        )

    def list_children(
        self,
        *,
        parent_path: str = "/",
        expected_space_revision: int,
        limit: int = 100,
        cursor: str | None = None,
    ) -> MemoryChildPage:
        parent = canonical_parent_path(parent_path)
        page_limit = _positive_limit(limit, maximum=_MAX_CHILD_PAGE)
        current_space = self.space
        if (
            isinstance(expected_space_revision, bool)
            or not isinstance(expected_space_revision, int)
            or expected_space_revision < 1
        ):
            raise ValueError("expected_space_revision must be a positive integer")
        if current_space.revision != expected_space_revision:
            raise RepositoryConflictError("space revision changed")
        last_path_key = ""
        last_entry_id = ""
        if cursor is not None:
            last_path_key, last_entry_id = _child_cursor_decode(
                cursor,
                space_id=self._space_id,
                parent_path=parent,
                space_revision=expected_space_revision,
            )
        parent_key = _path_key(parent)
        prefix = parent_key.rstrip("/") + "/"
        root = parent == "/"
        try:
            with self._store._transaction(immediate=False) as connection:
                rows = list(
                    connection.execute(
                        """
                        SELECT r.*, e.path_key,
                               CASE WHEN ? AND instr(substr(e.path_key, 2), '/') > 0
                                    THEN 1 ELSE 0 END AS orphaned,
                               EXISTS (
                                 SELECT 1
                                 FROM entries AS ce
                                 JOIN entry_revisions AS cr
                                   ON cr.space_id = ce.space_id
                                  AND cr.entry_id = ce.entry_id
                                  AND cr.revision = ce.current_revision
                                 WHERE ce.space_id = e.space_id
                                   AND ce.deleted = 0
                                   AND substr(ce.path_key, 1, length(e.path_key) + 1)
                                       = e.path_key || '/'
                                   AND instr(substr(ce.path_key, length(e.path_key) + 2), '/') = 0
                               ) AS has_children
                        FROM entries AS e
                        JOIN entry_revisions AS r
                          ON r.space_id = e.space_id
                         AND r.entry_id = e.entry_id
                         AND r.revision = e.current_revision
                        WHERE e.space_id = ?
                          AND e.deleted = 0
                          AND (
                            (
                              ?
                              AND (
                                instr(substr(e.path_key, 2), '/') = 0
                                OR NOT EXISTS (
                                  SELECT 1
                                  FROM entries AS pe
                                  JOIN entry_revisions AS pr
                                    ON pr.space_id = pe.space_id
                                   AND pr.entry_id = pe.entry_id
                                   AND pr.revision = pe.current_revision
                                  WHERE pe.space_id = e.space_id
                                    AND pe.deleted = 0
                                    AND json_extract(pr.entry_json, '$.kind') = 'folder'
                                    AND substr(e.path_key, 1, length(pe.path_key) + 1)
                                        = pe.path_key || '/'
                                    AND instr(substr(e.path_key, length(pe.path_key) + 2), '/') = 0
                                )
                              )
                            )
                            OR (
                              NOT ?
                              AND substr(e.path_key, 1, length(?)) = ?
                              AND instr(substr(e.path_key, length(?) + 1), '/') = 0
                            )
                          )
                          AND (
                            ? = '' OR e.path_key > ?
                            OR (e.path_key = ? AND e.entry_id > ?)
                          )
                        ORDER BY e.path_key, e.entry_id
                        LIMIT ?
                        """,
                        (
                            int(root),
                            self._space_id,
                            int(root),
                            int(root),
                            prefix,
                            prefix,
                            prefix,
                            last_path_key,
                            last_path_key,
                            last_path_key,
                            last_entry_id,
                            page_limit + 1,
                        ),
                    )
                )
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite child listing failed") from exc
        has_more = len(rows) > page_limit
        selected_rows = rows[:page_limit]
        children = tuple(
            MemoryChildEntry(
                entry=self._entry_from_row(row),
                has_children=bool(row["has_children"]),
                orphaned=bool(row["orphaned"]),
            )
            for row in selected_rows
        )
        next_cursor = None
        if has_more and selected_rows:
            last = selected_rows[-1]
            next_cursor = _child_cursor_encode(
                space_id=self._space_id,
                parent_path=parent,
                space_revision=expected_space_revision,
                path_key=last["path_key"],
                entry_id=last["entry_id"],
            )
        return MemoryChildPage(
            space_id=self._space_id,
            space_revision=expected_space_revision,
            parent_path=parent,
            order_version=_CHILD_ORDER_VERSION,
            entries=children,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def supersede_projection(
        self,
        *,
        entry_ref: ResourceRef,
        deleted: bool,
        backend_identity: str,
    ) -> tuple[str, ...]:
        if not isinstance(entry_ref, ResourceRef) or entry_ref.fragment != self._space_id:
            raise RepositoryScopeError("projection entry belongs to another workspace")
        backend = _required_text(backend_identity, "backend_identity", maximum=512)
        with self._store._transaction(immediate=True) as connection:
            head = connection.execute(
                "SELECT current_revision, deleted FROM entries WHERE space_id = ? AND entry_id = ?",
                (self._space_id, entry_ref.resource_id),
            ).fetchone()
            if head is None or int(head["current_revision"]) != entry_ref.revision:
                raise RepositoryConflictError("projection entry revision is not current")
            if bool(head["deleted"]) != bool(deleted):
                raise RepositoryConflictError("projection deletion state is not current")
            external_ids = tuple(
                row[0]
                for row in connection.execute(
                    "SELECT external_receipt_id FROM vector_projection_points WHERE space_id = ? AND entry_id = ? AND backend_identity = ?",
                    (self._space_id, entry_ref.resource_id, backend),
                )
            )
            connection.execute(
                "DELETE FROM vector_projection_points WHERE space_id = ? AND entry_id = ? AND backend_identity = ?",
                (self._space_id, entry_ref.resource_id, backend),
            )
            connection.execute(
                "DELETE FROM vector_projection_receipts WHERE space_id = ? AND entry_id = ? AND backend_identity = ?",
                (self._space_id, entry_ref.resource_id, backend),
            )
            if deleted:
                connection.execute(
                    "UPDATE vector_projection_watermarks SET corpus_epoch = (SELECT revision FROM spaces WHERE space_id = ?) WHERE space_id = ? AND backend_identity = ?",
                    (self._space_id, self._space_id, backend),
                )
        return external_ids

    def commit_projection(
        self,
        *,
        entry_ref: ResourceRef,
        backend_identity: str,
        chunker_version: str,
        basis_id: str,
        basis_version: int,
        algorithm: str,
        dimension: int,
        content_digest: str,
        points: Sequence[Mapping[str, Any]],
    ) -> None:
        if not isinstance(entry_ref, ResourceRef) or entry_ref.fragment != self._space_id:
            raise RepositoryScopeError("projection entry belongs to another workspace")
        backend = _required_text(backend_identity, "backend_identity", maximum=512)
        chunker = _required_text(chunker_version, "chunker_version", maximum=128)
        basis = _required_text(basis_id, "basis_id", maximum=512)
        algo = _required_text(algorithm, "algorithm", maximum=128)
        if algo != _PROJECTION_ALGORITHM:
            raise ValueError("projection algorithm is unsupported")
        basis_revision = _bounded_int(basis_version, "basis_version", minimum=1)
        vector_dimension = _bounded_int(dimension, "dimension", minimum=1)
        digest = _required_text(content_digest, "content_digest", maximum=128)
        if isinstance(points, (str, bytes, bytearray)) or not isinstance(points, Sequence):
            raise TypeError("points must be a sequence")
        rebuilt: list[tuple[Any, ...]] = []
        seen: set[str] = set()
        for raw in points:
            if not isinstance(raw, Mapping):
                raise TypeError("projection point must be a mapping")
            chunk_id = _required_text(raw.get("chunk_id"), "chunk_id", identifier=True)
            if chunk_id in seen:
                raise ValueError("projection chunk ids must be unique")
            seen.add(chunk_id)
            ordinal = _bounded_int(raw.get("ordinal"), "ordinal")
            x = raw.get("x")
            y = raw.get("y")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in (x, y)):
                raise ValueError("projection coordinates must be finite")
            rebuilt.append(
                (
                    self._space_id,
                    entry_ref.resource_id,
                    entry_ref.revision,
                    backend,
                    basis,
                    basis_revision,
                    chunk_id,
                    ordinal,
                    float(x),
                    float(y),
                    _required_text(raw.get("embedding_digest"), "embedding_digest", maximum=128),
                    _required_text(raw.get("external_receipt_id"), "external_receipt_id", maximum=512),
                )
            )
        with self._store._transaction(immediate=True) as connection:
            head = connection.execute(
                "SELECT current_revision, deleted FROM entries WHERE space_id = ? AND entry_id = ?",
                (self._space_id, entry_ref.resource_id),
            ).fetchone()
            if head is None or int(head["current_revision"]) != entry_ref.revision or bool(head["deleted"]):
                raise RepositoryConflictError("projection entry revision is not current")
            connection.execute(
                "DELETE FROM vector_projection_points WHERE space_id = ? AND entry_id = ? AND backend_identity = ?",
                (self._space_id, entry_ref.resource_id, backend),
            )
            connection.execute(
                "DELETE FROM vector_projection_receipts WHERE space_id = ? AND entry_id = ? AND backend_identity = ?",
                (self._space_id, entry_ref.resource_id, backend),
            )
            connection.execute(
                """
                INSERT INTO vector_projection_receipts(
                    space_id, entry_id, entry_revision, backend_identity,
                    chunker_version, basis_id, basis_version, algorithm,
                    dimension, expected_chunks, indexed_chunks, content_digest, complete
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    self._space_id, entry_ref.resource_id, entry_ref.revision,
                    backend, chunker, basis, basis_revision, algo,
                    vector_dimension, len(rebuilt), len(rebuilt), digest,
                ),
            )
            connection.executemany(
                """
                INSERT INTO vector_projection_points(
                    space_id, entry_id, entry_revision, backend_identity,
                    basis_id, basis_version, chunk_id, ordinal, x, y,
                    embedding_digest, external_receipt_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rebuilt,
            )
            space_revision = int(connection.execute("SELECT revision FROM spaces WHERE space_id = ?", (self._space_id,)).fetchone()[0])
            connection.execute(
                """
                INSERT INTO vector_projection_watermarks(
                    space_id, backend_identity, chunker_version, basis_id,
                    basis_version, algorithm, dimension, corpus_epoch
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(space_id, backend_identity, basis_id, basis_version)
                DO UPDATE SET chunker_version=excluded.chunker_version,
                    algorithm=excluded.algorithm, dimension=excluded.dimension,
                    corpus_epoch=excluded.corpus_epoch
                """,
                (self._space_id, backend, chunker, basis, basis_revision, algo, vector_dimension, space_revision),
            )

    def list_projection_points(
        self,
        *,
        backend_identity: str,
        chunker_version: str,
        basis_id: str,
        basis_version: int,
        algorithm: str,
        dimension: int,
        corpus_epoch: int | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> VectorProjectionPage:
        backend = _required_text(backend_identity, "backend_identity", maximum=512)
        chunker = _required_text(chunker_version, "chunker_version", maximum=128)
        basis = _required_text(basis_id, "basis_id", maximum=512)
        basis_revision = _bounded_int(basis_version, "basis_version", minimum=1)
        vector_dimension = _bounded_int(dimension, "dimension", minimum=1)
        if algorithm != _PROJECTION_ALGORITHM:
            raise ValueError("projection algorithm is unsupported")
        page_limit = _positive_limit(limit, maximum=_MAX_CHILD_PAGE)
        current_epoch = self.space.revision
        if corpus_epoch is not None and corpus_epoch != current_epoch:
            raise RepositoryConflictError("projection corpus epoch changed")
        last_chunk_id = ""
        if cursor is not None:
            last_chunk_id = _projection_cursor_decode(
                cursor,
                space_id=self._space_id,
                backend_identity=backend,
                basis_id=basis,
                basis_version=basis_revision,
                corpus_epoch=current_epoch,
            )
        with self._store._transaction(immediate=False) as connection:
            watermark = connection.execute(
                """
                SELECT corpus_epoch FROM vector_projection_watermarks
                WHERE space_id=? AND backend_identity=? AND basis_id=? AND basis_version=?
                  AND chunker_version=? AND algorithm=? AND dimension=?
                """,
                (self._space_id, backend, basis, basis_revision, chunker, algorithm, vector_dimension),
            ).fetchone()
            eligible_entries = int(connection.execute("SELECT COUNT(*) FROM entries WHERE space_id=? AND deleted=0", (self._space_id,)).fetchone()[0])
            counts = connection.execute(
                """
                SELECT COUNT(*) AS entries, COALESCE(SUM(r.expected_chunks),0) AS expected,
                       COALESCE(SUM(r.indexed_chunks),0) AS indexed
                FROM vector_projection_receipts AS r
                JOIN entries AS e ON e.space_id=r.space_id AND e.entry_id=r.entry_id
                    AND e.current_revision=r.entry_revision AND e.deleted=0
                WHERE r.space_id=? AND r.backend_identity=? AND r.basis_id=?
                  AND r.basis_version=? AND r.complete=1
                """,
                (self._space_id, backend, basis, basis_revision),
            ).fetchone()
            rows = list(
                connection.execute(
                    """
                    SELECT p.* FROM vector_projection_points AS p
                    JOIN entries AS e ON e.space_id=p.space_id AND e.entry_id=p.entry_id
                        AND e.current_revision=p.entry_revision AND e.deleted=0
                    WHERE p.space_id=? AND p.backend_identity=? AND p.basis_id=?
                      AND p.basis_version=? AND (?='' OR p.chunk_id>?)
                    ORDER BY p.chunk_id LIMIT ?
                    """,
                    (self._space_id, backend, basis, basis_revision, last_chunk_id, last_chunk_id, page_limit + 1),
                )
            )
        has_more = len(rows) > page_limit
        selected = rows[:page_limit]
        next_cursor = (
            _projection_cursor_encode(
                space_id=self._space_id,
                backend_identity=backend,
                basis_id=basis,
                basis_version=basis_revision,
                corpus_epoch=current_epoch,
                chunk_id=selected[-1]["chunk_id"],
            )
            if has_more and selected
            else None
        )
        indexed_entries = int(counts["entries"])
        complete = (
            watermark is not None
            and int(watermark["corpus_epoch"]) == current_epoch
            and indexed_entries == eligible_entries
        )
        return VectorProjectionPage(
            space_id=self._space_id,
            backend_identity=backend,
            chunker_version=chunker,
            corpus_epoch=current_epoch,
            basis_id=basis,
            basis_version=basis_revision,
            algorithm=algorithm,
            dimension=vector_dimension,
            status="complete" if complete else "partial",
            stale=False,
            eligible_entries=eligible_entries,
            indexed_entries=indexed_entries,
            eligible_chunks=int(counts["expected"]),
            indexed_chunks=int(counts["indexed"]),
            projected_chunks=int(counts["indexed"]),
            points=tuple(
                VectorProjectionPoint(
                    chunk_id=row["chunk_id"],
                    entry_id=row["entry_id"],
                    entry_revision=int(row["entry_revision"]),
                    ordinal=int(row["ordinal"]),
                    x=float(row["x"]),
                    y=float(row["y"]),
                )
                for row in selected
            ),
            next_cursor=next_cursor,
            has_more=has_more,
        )

    def authorize_projection_hits(
        self,
        *,
        backend_identity: str,
        chunk_ids: Sequence[str],
    ) -> dict[str, ResourceRef]:
        backend = _required_text(backend_identity, "backend_identity", maximum=512)
        normalized = tuple(
            _required_text(chunk_id, "chunk_id", identifier=True)
            for chunk_id in chunk_ids[:500]
        )
        if not normalized:
            return {}
        placeholders = ",".join("?" for _ in normalized)
        with self._store._transaction(immediate=False) as connection:
            rows = connection.execute(
                f"""
                SELECT p.chunk_id, p.entry_id, p.entry_revision
                FROM vector_projection_points AS p
                JOIN entries AS e ON e.space_id=p.space_id AND e.entry_id=p.entry_id
                    AND e.current_revision=p.entry_revision AND e.deleted=0
                WHERE p.space_id=? AND p.backend_identity=?
                  AND p.chunk_id IN ({placeholders})
                """,
                (self._space_id, backend, *normalized),
            )
            return {
                row["chunk_id"]: ResourceRef(
                    "memory",
                    row["entry_id"],
                    int(row["entry_revision"]),
                    self._space_id,
                )
                for row in rows
            }

    def current_projection_contract(self, *, backend_identity: str) -> dict[str, Any] | None:
        backend = _required_text(backend_identity, "backend_identity", maximum=512)
        with self._store._transaction(immediate=False) as connection:
            row = connection.execute(
                """
                SELECT chunker_version, basis_id, basis_version, algorithm,
                       dimension, corpus_epoch
                FROM vector_projection_watermarks
                WHERE space_id=? AND backend_identity=?
                ORDER BY corpus_epoch DESC, basis_version DESC LIMIT 1
                """,
                (self._space_id, backend),
            ).fetchone()
        return dict(row) if row is not None else None

    def search(self, *, query: str, limit: int = 20) -> tuple[MemoryEntry, ...]:
        normalized = _required_text(query, "query", maximum=4096)
        page_limit = _positive_limit(limit, maximum=100)
        if not self._store._fts_available:
            raise RepositorySearchUnavailableError("workspace FTS is unavailable")
        folded = unicodedata.normalize("NFKC", normalized).casefold()
        tokens = tuple(re.findall(r"\w+", normalized, flags=re.UNICODE))
        match_expression = " AND ".join(
            '"' + token.replace('"', '""') + '"' for token in tokens
        )
        try:
            with self._store._transaction(immediate=False) as connection:
                state = connection.execute(
                    """
                    SELECT status FROM index_state
                    WHERE index_name = 'workspace_fts' AND scope_id = ?
                    """,
                    (self._space_id,),
                ).fetchone()
                if state is None or state["status"] != "ready":
                    raise RepositorySearchUnavailableError(
                        "workspace FTS is unavailable"
                    )
                exact_rows = list(
                    connection.execute(
                        """
                        SELECT r.*, CASE
                            WHEN e.path_key = ? THEN 0 ELSE 1 END AS exact_rank
                        FROM entries AS e
                        JOIN entry_revisions AS r
                          ON r.space_id = e.space_id
                         AND r.entry_id = e.entry_id
                         AND r.revision = e.current_revision
                        WHERE e.space_id = ? AND e.deleted = 0
                          AND (e.path_key = ? OR e.name_key = ?)
                        ORDER BY exact_rank, e.path_key, e.entry_id
                        """,
                        (folded, self._space_id, folded, folded),
                    )
                )
                fts_rows: list[sqlite3.Row] = []
                if match_expression:
                    fts_rows = list(
                        connection.execute(
                            """
                            SELECT r.*, bm25(workspace_entries_fts) AS rank
                            FROM workspace_entries_fts
                            JOIN entries AS e
                              ON e.space_id = workspace_entries_fts.space_id
                             AND e.entry_id = workspace_entries_fts.entry_id
                            JOIN entry_revisions AS r
                              ON r.space_id = e.space_id
                             AND r.entry_id = e.entry_id
                             AND r.revision = e.current_revision
                            WHERE workspace_entries_fts MATCH ?
                              AND e.space_id = ? AND e.deleted = 0
                            ORDER BY rank, e.path_key, e.entry_id
                            LIMIT ?
                            """,
                            (match_expression, self._space_id, page_limit),
                        )
                    )
                ordered: list[MemoryEntry] = []
                seen: set[str] = set()
                for row in (*exact_rows, *fts_rows):
                    entry = self._entry_from_row(row)
                    if entry.entry_id not in seen:
                        ordered.append(entry)
                        seen.add(entry.entry_id)
                    if len(ordered) >= page_limit:
                        break
                return tuple(ordered)
        except RepositorySearchUnavailableError:
            raise
        except sqlite3.OperationalError as exc:
            self._store._fts_available = False
            try:
                with self._store._transaction(immediate=True) as connection:
                    connection.execute(
                        """
                        UPDATE index_state
                        SET status = 'unavailable', detail = 'sqlite_fts_unavailable'
                        WHERE index_name = 'workspace_fts'
                        """
                    )
            except sqlite3.Error:
                # The unavailable index marker is best effort; the original search failure is still raised below.
                pass
            raise RepositorySearchUnavailableError(
                "workspace FTS is unavailable"
            ) from exc
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite workspace search failed") from exc

    def read_entry(self, *, ref: ResourceRef) -> MemoryEntry:
        self._require_memory_ref(ref)
        try:
            with self._store._transaction(immediate=False) as connection:
                row = self._entry_revision_row(
                    connection,
                    entry_id=ref.resource_id,
                    revision=ref.revision,
                )
                if row is None:
                    raise RepositoryNotFoundError("entry revision was not found")
                return self._entry_from_row(row)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError(
                "SQLite workspace entry read failed"
            ) from exc

    def read_current_entry(self, *, entry_id: str) -> MemoryEntry:
        normalized = _required_text(entry_id, "entry_id", identifier=True)
        try:
            with self._store._transaction(immediate=False) as connection:
                row = self._current_entry_row(connection, entry_id=normalized)
                if row is None:
                    raise RepositoryNotFoundError("entry was not found")
                return self._entry_from_row(row)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite current entry read failed") from exc

    def compare_and_swap(
        self,
        *,
        entry: MemoryEntry,
        expected_revision: int | None,
        operation: OperationRef,
    ) -> MemoryEntry:
        return self.apply(
            request=WorkspaceMutationRequest(
                entry=entry,
                expected_revision=expected_revision,
                expected_space_revision=self.space.revision,
                operation=operation,
            )
        )

    def _write_entry_revision(
        self,
        connection: sqlite3.Connection,
        *,
        entry: MemoryEntry,
        operation_id: str,
        object_sha256: str | None,
        byte_length: int | None,
    ) -> None:
        encoded = _canonical_json_bytes(entry.to_dict())
        content_resource_id = (
            entry.content_ref.resource_id if entry.content_ref is not None else None
        )
        connection.execute(
            """
            INSERT INTO entry_revisions(
                space_id, entry_id, revision, path_key, operation_id,
                entry_json, entry_sha256, content_resource_id,
                object_sha256, byte_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self._space_id,
                entry.entry_id,
                entry.revision,
                _path_key(entry.path),
                operation_id,
                encoded,
                _sha256(encoded),
                content_resource_id,
                object_sha256,
                byte_length,
            ),
        )
        connection.execute(
            """
            INSERT INTO entries(
                space_id, entry_id, current_revision, path_key, name_key,
                deleted, updated_seq
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(space_id, entry_id) DO UPDATE SET
                current_revision = excluded.current_revision,
                path_key = excluded.path_key,
                name_key = excluded.name_key,
                deleted = excluded.deleted,
                updated_seq = excluded.updated_seq
            """,
            (
                self._space_id,
                entry.entry_id,
                entry.revision,
                _path_key(entry.path),
                _name_key(entry.name),
                int(entry.deleted),
                entry.updated_seq,
            ),
        )
        self._store._refresh_fts_entry(connection, entry)

    def apply(self, *, request: WorkspaceMutationRequest) -> MemoryEntry:
        if not isinstance(request, WorkspaceMutationRequest):
            raise TypeError("request must be a WorkspaceMutationRequest")
        entry = request.entry
        if entry.space_id != self._space_id:
            raise RepositoryScopeError("entry belongs to another workspace")
        installed: tuple[str, int] | None = None
        if request.content is not None:
            installed = self._store._install_object(request.content)
        target_key = f"{entry.entry_id}@{entry.revision}"
        try:
            with self._store._transaction(immediate=True) as connection:
                receipt = self._receipt(
                    connection,
                    operation=request.operation,
                    target_kind="entry",
                    target_key=target_key,
                )
                if receipt is not None:
                    row = self._entry_revision_row(
                        connection,
                        entry_id=receipt["result_id"],
                        revision=receipt["result_revision"],
                    )
                    if row is None:
                        raise SQLiteMemoryV2StoreIntegrityError(
                            "workspace operation receipt lost its entry revision"
                        )
                    return self._entry_from_row(row)

                space_row = self._load_space_row(connection)
                if space_row["revision"] != request.expected_space_revision:
                    raise RepositoryConflictError("space revision changed")
                if entry.updated_seq != request.expected_space_revision + 1:
                    raise RepositoryConflictError(
                        "updated sequence did not advance with the space"
                    )
                current_row = self._current_entry_row(
                    connection,
                    entry_id=entry.entry_id,
                )
                current = (
                    self._entry_from_row(current_row)
                    if current_row is not None
                    else None
                )
                actual_revision = current.revision if current is not None else None
                if actual_revision != request.expected_revision:
                    raise RepositoryConflictError("entry revision changed")
                expected_next = 1 if current is None else current.revision + 1
                if entry.revision != expected_next:
                    raise RepositoryConflictError(
                        "entry revision must advance exactly once"
                    )

                descendants: list[tuple[MemoryEntry, sqlite3.Row]] = []
                if current is not None and current.kind is MemoryEntryKind.FOLDER:
                    prefix = _path_key(current.path).rstrip("/") + "/"
                    descendant_rows = list(
                        connection.execute(
                            """
                            SELECT r.*
                            FROM entries AS e
                            JOIN entry_revisions AS r
                              ON r.space_id = e.space_id
                             AND r.entry_id = e.entry_id
                             AND r.revision = e.current_revision
                            WHERE e.space_id = ? AND e.deleted = 0
                              AND substr(e.path_key, 1, length(?)) = ?
                            ORDER BY e.path_key, e.entry_id
                            """,
                            (self._space_id, prefix, prefix),
                        )
                    )
                    descendants = [
                        (self._entry_from_row(row), row) for row in descendant_rows
                    ]
                if entry.deleted and descendants and not request.recursive:
                    raise RepositoryConflictError("folder contains active descendants")

                collision = connection.execute(
                    """
                    SELECT 1 FROM entries
                    WHERE space_id = ? AND path_key = ? AND deleted = 0
                      AND entry_id != ?
                    LIMIT 1
                    """,
                    (self._space_id, _path_key(entry.path), entry.entry_id),
                ).fetchone()
                if collision is not None:
                    raise RepositoryConflictError("path collision")

                persisted = entry
                object_sha256: str | None = None
                byte_length: int | None = None
                if installed is not None:
                    object_sha256, byte_length = installed
                    content_ref = ResourceRef(
                        "memory_content",
                        f"{entry.entry_id}-content",
                        entry.revision,
                        self._space_id,
                    )
                    persisted = replace(entry, content_ref=content_ref)
                    connection.execute(
                        "INSERT OR IGNORE INTO objects(sha256, byte_length) VALUES (?, ?)",
                        (object_sha256, byte_length),
                    )
                    object_row = connection.execute(
                        "SELECT byte_length FROM objects WHERE sha256 = ?",
                        (object_sha256,),
                    ).fetchone()
                    if object_row is None or object_row["byte_length"] != byte_length:
                        raise SQLiteMemoryV2StoreIntegrityError(
                            "workspace object catalog changed"
                        )
                elif current_row is not None and current is not None:
                    if (
                        persisted.content_ref is None
                        and current.content_ref is not None
                    ):
                        persisted = replace(persisted, content_ref=current.content_ref)
                    elif persisted.content_ref != current.content_ref:
                        raise RepositoryConflictError(
                            "content reference changed without content bytes"
                        )
                    object_sha256 = current_row["object_sha256"]
                    byte_length = current_row["byte_length"]

                if persisted.kind in {MemoryEntryKind.FOLDER, MemoryEntryKind.LINK}:
                    if persisted.content_ref is not None or object_sha256 is not None:
                        raise RepositoryConflictError(
                            "entry kind cannot reference durable content"
                        )
                elif persisted.content_ref is None or object_sha256 is None:
                    raise RepositoryConflictError(
                        "content entry requires durable object bytes"
                    )

                if entry.deleted and request.recursive:
                    for descendant, descendant_row in descendants:
                        tombstone = replace(
                            descendant,
                            revision=descendant.revision + 1,
                            updated_seq=entry.updated_seq,
                            source_refs=entry.source_refs,
                            deleted=True,
                        )
                        self._write_entry_revision(
                            connection,
                            entry=tombstone,
                            operation_id=request.operation.operation_id,
                            object_sha256=descendant_row["object_sha256"],
                            byte_length=descendant_row["byte_length"],
                        )

                self._write_entry_revision(
                    connection,
                    entry=persisted,
                    operation_id=request.operation.operation_id,
                    object_sha256=object_sha256,
                    byte_length=byte_length,
                )
                advanced = self._advance_space(
                    connection,
                    current_row=space_row,
                    expected_revision=request.expected_space_revision,
                )
                self._store._claim_receipt(
                    connection,
                    scope_kind="workspace",
                    scope_id=self._space_id,
                    operation=request.operation,
                    target_kind="entry",
                    target_key=target_key,
                    result_id=persisted.entry_id,
                    result_revision=persisted.revision,
                )
                self._space = advanced
                return persisted
        except RepositoryConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            message = str(exc).casefold()
            if (
                "idx_entries_active_path" in message
                or "entries.space_id, entries.path_key" in message
            ):
                raise RepositoryConflictError("path collision") from exc
            raise RepositoryConflictError(
                "workspace mutation operation or revision conflicted"
            ) from exc
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite workspace mutation failed") from exc

    def read_content(
        self,
        *,
        ref: ResourceRef,
        offset: int = 0,
        limit: int = 32 * 1024,
    ) -> WorkspaceContentPage:
        if not isinstance(ref, ResourceRef):
            raise TypeError("ref must be a ResourceRef")
        if ref.kind != "memory_content" or ref.fragment != self._space_id:
            raise RepositoryScopeError(
                "content reference is outside the bound workspace"
            )
        page_offset = _non_negative_offset(offset)
        page_limit = _positive_limit(limit, maximum=1024 * 1024)
        try:
            with self._store._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM entry_revisions
                    WHERE space_id = ? AND content_resource_id = ? AND revision = ?
                    LIMIT 1
                    """,
                    (self._space_id, ref.resource_id, ref.revision),
                ).fetchone()
                if row is None:
                    raise RepositoryNotFoundError("workspace content was not found")
                entry = self._entry_from_row(row)
                byte_length = row["byte_length"]
                if page_offset > byte_length:
                    raise ValueError("offset exceeds the workspace content length")
                content = self._store._read_object(
                    digest=row["object_sha256"],
                    byte_length=byte_length,
                )
                return WorkspaceContentPage(
                    ref=ref,
                    media_type=entry.media_type or "application/octet-stream",
                    data=content[page_offset : page_offset + page_limit],
                    offset=page_offset,
                    total_bytes=byte_length,
                )
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError(
                "SQLite workspace content read failed"
            ) from exc

    def list_revisions(
        self,
        *,
        ref: ResourceRef,
        before_revision: int | None = None,
        limit: int = 20,
    ) -> tuple[MemoryEntry, ...]:
        self._require_memory_ref(ref)
        page_limit = _positive_limit(limit, maximum=100)
        if before_revision is not None and (
            isinstance(before_revision, bool)
            or not isinstance(before_revision, int)
            or before_revision < 1
        ):
            raise ValueError("before_revision must be a positive integer")
        ceiling = before_revision if before_revision is not None else ref.revision + 1
        try:
            with self._store._transaction(immediate=False) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM entries WHERE space_id = ? AND entry_id = ?",
                    (self._space_id, ref.resource_id),
                ).fetchone()
                if exists is None:
                    raise RepositoryNotFoundError("entry was not found")
                rows = list(
                    connection.execute(
                        """
                        SELECT * FROM entry_revisions
                        WHERE space_id = ? AND entry_id = ? AND revision < ?
                        ORDER BY revision DESC
                        LIMIT ?
                        """,
                        (self._space_id, ref.resource_id, ceiling, page_limit),
                    )
                )
                return tuple(self._entry_from_row(row) for row in rows)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError(
                "SQLite workspace history read failed"
            ) from exc

    def _current_ref_entry(
        self,
        connection: sqlite3.Connection,
        ref: ResourceRef,
    ) -> MemoryEntry:
        self._require_memory_ref(ref)
        row = self._current_entry_row(connection, entry_id=ref.resource_id)
        if row is None:
            raise RepositoryNotFoundError("linked entry was not found")
        entry = self._entry_from_row(row)
        if entry.revision != ref.revision or entry.deleted:
            raise RepositoryConflictError("linked entry reference is not current")
        return entry

    def _link_from_row(self, row: sqlite3.Row) -> MemoryLink:
        raw = bytes(row["link_json"])
        if _sha256(raw) != row["link_sha256"]:
            raise SQLiteMemoryV2StoreIntegrityError("memory link digest changed")
        try:
            link = MemoryLink.from_dict(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SQLiteMemoryV2StoreIntegrityError("memory link is malformed") from exc
        if (
            _canonical_json_bytes(link.to_dict()) != raw
            or link.link_id != row["link_id"]
            or link.revision != row["revision"]
            or link.source_entry_ref.resource_id != row["source_entry_id"]
            or link.source_entry_ref.revision != row["source_revision"]
            or link.target_ref.resource_id != row["target_entry_id"]
            or link.target_ref.revision != row["target_revision"]
            or link.relation != row["relation"]
            or link.source_entry_ref.fragment != self._space_id
            or link.target_ref.fragment != self._space_id
        ):
            raise SQLiteMemoryV2StoreIntegrityError(
                "memory link indexed metadata changed"
            )
        return link

    def create_link(self, *, request: WorkspaceLinkRequest) -> MemoryLink:
        if not isinstance(request, WorkspaceLinkRequest):
            raise TypeError("request must be a WorkspaceLinkRequest")
        self._require_memory_ref(request.source_entry_ref)
        self._require_memory_ref(request.link.source_entry_ref)
        self._require_memory_ref(request.link.target_ref)
        if request.link.source_entry_ref != request.source_entry_ref:
            raise RepositoryScopeError("link source reference changed")
        target_key = f"{request.link.link_id}@{request.link.revision}"
        try:
            with self._store._transaction(immediate=True) as connection:
                receipt = self._receipt(
                    connection,
                    operation=request.operation,
                    target_kind="link",
                    target_key=target_key,
                )
                if receipt is not None:
                    row = connection.execute(
                        """
                        SELECT * FROM links
                        WHERE space_id = ? AND link_id = ? AND revision = ?
                        """,
                        (
                            self._space_id,
                            receipt["result_id"],
                            receipt["result_revision"],
                        ),
                    ).fetchone()
                    if row is None:
                        raise SQLiteMemoryV2StoreIntegrityError(
                            "workspace operation receipt lost its memory link"
                        )
                    return self._link_from_row(row)
                space_row = self._load_space_row(connection)
                if space_row["revision"] != request.expected_space_revision:
                    raise RepositoryConflictError("space revision changed")
                self._current_ref_entry(connection, request.source_entry_ref)
                self._current_ref_entry(connection, request.link.target_ref)
                if request.link.revision != 1:
                    raise RepositoryConflictError(
                        "new memory link must start at revision one"
                    )
                encoded = _canonical_json_bytes(request.link.to_dict())
                provenance = _canonical_json_bytes(
                    [ref.to_dict() for ref in request.source_refs]
                )
                connection.execute(
                    """
                    INSERT INTO links(
                        space_id, link_id, revision,
                        source_entry_id, source_revision,
                        target_entry_id, target_revision, relation,
                        operation_id, link_json, link_sha256,
                        source_refs_json, source_refs_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._space_id,
                        request.link.link_id,
                        request.link.revision,
                        request.link.source_entry_ref.resource_id,
                        request.link.source_entry_ref.revision,
                        request.link.target_ref.resource_id,
                        request.link.target_ref.revision,
                        request.link.relation,
                        request.operation.operation_id,
                        encoded,
                        _sha256(encoded),
                        provenance,
                        _sha256(provenance),
                    ),
                )
                advanced = self._advance_space(
                    connection,
                    current_row=space_row,
                    expected_revision=request.expected_space_revision,
                )
                self._store._claim_receipt(
                    connection,
                    scope_kind="workspace",
                    scope_id=self._space_id,
                    operation=request.operation,
                    target_kind="link",
                    target_key=target_key,
                    result_id=request.link.link_id,
                    result_revision=request.link.revision,
                )
                self._space = advanced
                return request.link
        except RepositoryConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise RepositoryConflictError(
                "memory link operation or identifier conflicted"
            ) from exc
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite memory link write failed") from exc

    def list_links(
        self,
        *,
        source_entry_ref: ResourceRef,
        limit: int = 100,
    ) -> tuple[MemoryLink, ...]:
        self._require_memory_ref(source_entry_ref)
        page_limit = _positive_limit(limit, maximum=200)
        try:
            with self._store._transaction(immediate=False) as connection:
                rows = list(
                    connection.execute(
                        """
                        SELECT * FROM links
                        WHERE space_id = ? AND source_entry_id = ?
                          AND source_revision = ?
                        ORDER BY link_id, revision
                        LIMIT ?
                        """,
                        (
                            self._space_id,
                            source_entry_ref.resource_id,
                            source_entry_ref.revision,
                            page_limit,
                        ),
                    )
                )
                return tuple(self._link_from_row(row) for row in rows)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite memory link read failed") from exc

    def list_backlinks(
        self,
        *,
        target_entry_ref: ResourceRef,
        limit: int = 100,
    ) -> tuple[MemoryLink, ...]:
        self._require_memory_ref(target_entry_ref)
        page_limit = _positive_limit(limit, maximum=200)
        try:
            with self._store._transaction(immediate=False) as connection:
                rows = list(
                    connection.execute(
                        """
                        SELECT * FROM links
                        WHERE space_id = ? AND target_entry_id = ?
                          AND target_revision = ?
                        ORDER BY link_id, revision
                        LIMIT ?
                        """,
                        (
                            self._space_id,
                            target_entry_ref.resource_id,
                            target_entry_ref.revision,
                            page_limit,
                        ),
                    )
                )
                return tuple(self._link_from_row(row) for row in rows)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError(
                "SQLite memory backlink read failed"
            ) from exc


class _SQLiteBoundPinnedTaskStateRepository(BoundPinnedTaskStateRepository):
    """Pinned task-state port bound to one durable chat lineage."""

    def __init__(
        self,
        store: SQLiteMemoryV2Store,
        *,
        binding_id: str,
        state_id: str | None,
    ) -> None:
        super().__init__(binding_id, state_id)
        self._store = store

    def _state_from_row(self, row: sqlite3.Row) -> PinnedTaskState:
        raw = bytes(row["state_json"])
        if _sha256(raw) != row["state_sha256"]:
            raise SQLiteMemoryV2StoreIntegrityError("task state digest changed")
        try:
            state = PinnedTaskState.from_dict(json.loads(raw.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise SQLiteMemoryV2StoreIntegrityError("task state is malformed") from exc
        if (
            _canonical_json_bytes(state.to_dict()) != raw
            or state.state_id != self.state_id
            or state.state_id != row["state_id"]
            or state.revision != row["revision"]
        ):
            raise SQLiteMemoryV2StoreIntegrityError(
                "task state indexed metadata changed"
            )
        return state

    def _revision_row(
        self,
        connection: sqlite3.Connection,
        *,
        revision: int,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT * FROM task_state_revisions
            WHERE binding_id = ? AND state_id = ? AND revision = ?
            """,
            (self.binding_id, self.state_id, revision),
        ).fetchone()

    def current(self) -> PinnedTaskState | None:
        try:
            with self._store._transaction(immediate=False) as connection:
                head = connection.execute(
                    """
                    SELECT current_revision FROM task_state_heads
                    WHERE binding_id = ? AND state_id = ?
                    """,
                    (self.binding_id, self.state_id),
                ).fetchone()
                if head is None:
                    return None
                row = self._revision_row(
                    connection,
                    revision=head["current_revision"],
                )
                if row is None:
                    raise SQLiteMemoryV2StoreIntegrityError(
                        "task state head lost its revision"
                    )
                return self._state_from_row(row)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite task state read failed") from exc

    def replay(self, *, operation: OperationRef) -> PinnedTaskState | None:
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        try:
            with self._store._transaction(immediate=False) as connection:
                receipt = connection.execute(
                    """
                    SELECT * FROM memory_operation_receipts
                    WHERE scope_kind = 'task_state' AND scope_id = ?
                      AND operation_id = ?
                    """,
                    (self.binding_id, operation.operation_id),
                ).fetchone()
                if receipt is None:
                    return None
                if (
                    receipt["payload_sha256"] != operation.payload_sha256
                    or receipt["target_kind"] != "task_state"
                    or receipt["target_key"] != self.state_id
                    or receipt["result_id"] != self.state_id
                ):
                    raise RepositoryConflictError(
                        "task state operation payload changed"
                    )
                row = self._revision_row(
                    connection,
                    revision=receipt["result_revision"],
                )
                if row is None:
                    raise SQLiteMemoryV2StoreIntegrityError(
                        "task state receipt lost its revision"
                    )
                return self._state_from_row(row)
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite task state replay failed") from exc

    def compare_and_swap(
        self,
        *,
        state: PinnedTaskState,
        expected_revision: int | None,
        operation: OperationRef,
    ) -> PinnedTaskState:
        if not isinstance(state, PinnedTaskState):
            raise TypeError("state must be a PinnedTaskState")
        if state.state_id != self.state_id:
            raise RepositoryScopeError("task state is outside the bound lineage")
        replayed = self.replay(operation=operation)
        if replayed is not None:
            return replayed
        try:
            with self._store._transaction(immediate=True) as connection:
                receipt = connection.execute(
                    """
                    SELECT * FROM memory_operation_receipts
                    WHERE scope_kind = 'task_state' AND scope_id = ?
                      AND operation_id = ?
                    """,
                    (self.binding_id, operation.operation_id),
                ).fetchone()
                if receipt is not None:
                    if receipt["payload_sha256"] != operation.payload_sha256:
                        raise RepositoryConflictError(
                            "task state operation payload changed"
                        )
                    row = self._revision_row(
                        connection,
                        revision=receipt["result_revision"],
                    )
                    if row is None:
                        raise SQLiteMemoryV2StoreIntegrityError(
                            "task state receipt lost its revision"
                        )
                    return self._state_from_row(row)
                head = connection.execute(
                    """
                    SELECT current_revision FROM task_state_heads
                    WHERE binding_id = ? AND state_id = ?
                    """,
                    (self.binding_id, self.state_id),
                ).fetchone()
                actual = head["current_revision"] if head is not None else None
                if actual != expected_revision:
                    raise RepositoryConflictError("task state revision changed")
                expected_next = 1 if actual is None else actual + 1
                if state.revision != expected_next:
                    raise RepositoryConflictError(
                        "task state revision must advance exactly once"
                    )
                encoded = _canonical_json_bytes(state.to_dict())
                connection.execute(
                    """
                    INSERT INTO task_state_revisions(
                        binding_id, state_id, revision, operation_id,
                        state_json, state_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.binding_id,
                        self.state_id,
                        state.revision,
                        operation.operation_id,
                        encoded,
                        _sha256(encoded),
                    ),
                )
                if head is None:
                    connection.execute(
                        """
                        INSERT INTO task_state_heads(
                            binding_id, state_id, current_revision
                        ) VALUES (?, ?, ?)
                        """,
                        (self.binding_id, self.state_id, state.revision),
                    )
                else:
                    updated = connection.execute(
                        """
                        UPDATE task_state_heads SET current_revision = ?
                        WHERE binding_id = ? AND state_id = ?
                          AND current_revision = ?
                        """,
                        (
                            state.revision,
                            self.binding_id,
                            self.state_id,
                            expected_revision,
                        ),
                    )
                    if updated.rowcount != 1:
                        raise RepositoryConflictError("task state revision changed")
                self._store._claim_receipt(
                    connection,
                    scope_kind="task_state",
                    scope_id=self.binding_id,
                    operation=operation,
                    target_kind="task_state",
                    target_key=self.state_id,
                    result_id=self.state_id,
                    result_revision=state.revision,
                )
                return state
        except RepositoryConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise RepositoryConflictError(
                "task state operation or revision conflicted"
            ) from exc
        except sqlite3.Error as exc:
            raise SQLiteMemoryV2StoreError("SQLite task state CAS failed") from exc


__all__ = [
    "SQLiteMemoryV2Store",
    "SQLiteMemoryV2StoreError",
    "SQLiteMemoryV2StoreIntegrityError",
]
