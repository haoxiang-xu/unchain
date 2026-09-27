"""Durable SQLite ports for Context V2 checkpoints and build envelopes.

The compiler-state slice shares the official :class:`SQLiteContextV2Store`
database and its content-addressed artifact service.  A checkpoint's complete
payload is persisted as an artifact before the checkpoint receipt becomes
visible; build claims and their input-trigger claim are committed atomically.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from unchain.context.artifacts import ArtifactService
from unchain.context.models import ContextBuildEnvelope
from unchain.context.ports import (
    BoundCheckpointRepository,
    BoundContextBuildRepository,
    CheckpointWriteStatus,
    ContextBuildReceipt,
    ContextConflictError,
    ContextRepositoryError,
    ContextScopeError,
    PreparedCheckpoint,
)
from unchain.journal import (
    ArtifactRef,
    EventCursor,
    EventRange,
    OperationRef,
    ResourceRef,
)
from unchain.journal.models import _required_text

from .sqlite_v2 import SQLiteContextV2Store, serialized_context_v2_database_access


_SCHEMA_VERSION = 1


class SQLiteContextCompilerV2Error(ContextRepositoryError):
    """Stable persistence failure for the Context V2 compiler-state slice."""


class SQLiteContextCompilerV2IntegrityError(SQLiteContextCompilerV2Error):
    """Durable compiler metadata no longer matches its canonical digest."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as error:
        raise SQLiteContextCompilerV2IntegrityError(
            "compiler record is not canonical JSON"
        ) from error


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _canonical_record(raw: object, *, field_name: str) -> Mapping[str, Any]:
    try:
        encoded = bytes(raw)
        decoded = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SQLiteContextCompilerV2IntegrityError(
            f"{field_name} is not valid canonical JSON"
        ) from error
    if not isinstance(decoded, Mapping) or _canonical_json_bytes(decoded) != encoded:
        raise SQLiteContextCompilerV2IntegrityError(
            f"{field_name} is not canonical JSON"
        )
    return decoded


def _verified_record(
    raw: object,
    digest: object,
    *,
    field_name: str,
) -> Mapping[str, Any]:
    try:
        encoded = bytes(raw)
    except (TypeError, ValueError) as error:
        raise SQLiteContextCompilerV2IntegrityError(
            f"{field_name} bytes are invalid"
        ) from error
    if not isinstance(digest, str) or _sha256(encoded) != digest:
        raise SQLiteContextCompilerV2IntegrityError(f"{field_name} digest changed")
    return _canonical_record(encoded, field_name=field_name)


def _resource_refs(values: Sequence[ResourceRef]) -> tuple[ResourceRef, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("refs must be a sequence")
    return tuple(
        value if isinstance(value, ResourceRef) else ResourceRef.from_dict(value)
        for value in values
    )


def _checkpoint_identity(execution_id: str, operation: OperationRef) -> str:
    return hashlib.sha256(
        (
            "unchain.context_checkpoint.v1\0"
            + execution_id
            + "\0"
            + operation.operation_id
            + "\0"
            + operation.payload_sha256
        ).encode("utf-8")
    ).hexdigest()


def _checkpoint_semantic(
    *,
    source_range: EventRange,
    summary: str,
    refs: tuple[ResourceRef, ...],
) -> dict[str, Any]:
    return {
        "schema": "unchain.sqlite_checkpoint_prepare.v1",
        "source_range": source_range.to_dict(),
        "summary_sha256": _sha256(summary.encode("utf-8")),
        "refs": [ref.to_dict() for ref in refs],
    }


def _build_semantic(
    *,
    envelope: ContextBuildEnvelope,
    trigger_cursor: EventCursor,
) -> dict[str, Any]:
    return {
        "schema": "unchain.sqlite_context_build.v1",
        "envelope": envelope.to_dict(),
        "trigger_cursor": trigger_cursor.to_dict(),
    }


@dataclass(frozen=True)
class SQLiteContextCompilerV2Capabilities:
    """The two official compiler ports bound to one execution."""

    checkpoints: BoundCheckpointRepository
    context_builds: BoundContextBuildRepository

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoints, BoundCheckpointRepository):
            raise TypeError("checkpoints must be a BoundCheckpointRepository")
        if not isinstance(self.context_builds, BoundContextBuildRepository):
            raise TypeError("context_builds must be a BoundContextBuildRepository")
        if self.checkpoints.execution_id != self.context_builds.execution_id:
            raise ContextScopeError("compiler capabilities crossed executions")

    @property
    def execution_id(self) -> str:
        return self.checkpoints.execution_id


class SQLiteContextCompilerV2Store:
    """Add compiler-state tables to one official Context V2 SQLite store."""

    def __init__(self, *, context_store: SQLiteContextV2Store) -> None:
        if type(context_store) is not SQLiteContextV2Store:
            raise TypeError("context_store must be the official SQLiteContextV2Store")
        self.context_store = context_store
        self.database_path = Path(context_store.database_path)
        self.object_directory = Path(context_store.object_directory)
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
        with serialized_context_v2_database_access(self.database_path):
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
        with serialized_context_v2_database_access(self.database_path):
            connection = self._connect()
            try:
                mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                if str(mode).casefold() != "wal":
                    raise SQLiteContextCompilerV2IntegrityError(
                        "SQLite refused WAL journal mode"
                    )
                connection.executescript(
                    """
                    BEGIN IMMEDIATE;

                    CREATE TABLE IF NOT EXISTS context_compiler_v2_schema (
                        version INTEGER PRIMARY KEY,
                        applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    INSERT OR IGNORE INTO context_compiler_v2_schema(version)
                    VALUES (1);

                    CREATE TABLE IF NOT EXISTS checkpoints (
                        execution_id TEXT NOT NULL,
                        checkpoint_id TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK(revision = 1),
                        preparation_id TEXT NOT NULL,
                        operation_id TEXT NOT NULL,
                        operation_payload_sha256 TEXT NOT NULL,
                        semantic_json BLOB NOT NULL,
                        semantic_sha256 TEXT NOT NULL,
                        source_start_seq INTEGER NOT NULL CHECK(source_start_seq >= 1),
                        source_start_event_id TEXT NOT NULL,
                        source_end_seq INTEGER NOT NULL CHECK(source_end_seq >= source_start_seq),
                        source_end_event_id TEXT NOT NULL,
                        artifact_json BLOB NOT NULL,
                        artifact_sha256 TEXT NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('prepared', 'committed')),
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        committed_at TEXT,
                        PRIMARY KEY (execution_id, checkpoint_id, revision),
                        UNIQUE (execution_id, preparation_id),
                        UNIQUE (execution_id, operation_id),
                        FOREIGN KEY (execution_id)
                            REFERENCES executions(execution_id),
                        FOREIGN KEY (execution_id, operation_id)
                            REFERENCES operations(execution_id, operation_id),
                        FOREIGN KEY (execution_id, source_start_seq)
                            REFERENCES events(execution_id, store_seq),
                        FOREIGN KEY (execution_id, source_end_seq)
                            REFERENCES events(execution_id, store_seq)
                    );
                    CREATE INDEX IF NOT EXISTS idx_checkpoints_source_range
                    ON checkpoints(execution_id, source_start_seq, source_end_seq);

                    CREATE TABLE IF NOT EXISTS context_builds (
                        execution_id TEXT NOT NULL,
                        build_id TEXT NOT NULL,
                        generation_id TEXT NOT NULL,
                        attempt_id TEXT NOT NULL,
                        operation_id TEXT NOT NULL,
                        operation_payload_sha256 TEXT NOT NULL,
                        trigger_store_seq INTEGER NOT NULL CHECK(trigger_store_seq >= 1),
                        trigger_event_id TEXT NOT NULL,
                        semantic_sha256 TEXT NOT NULL,
                        envelope_json BLOB NOT NULL,
                        envelope_sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (execution_id, build_id),
                        UNIQUE (execution_id, operation_id),
                        UNIQUE (execution_id, trigger_store_seq, trigger_event_id),
                        FOREIGN KEY (execution_id)
                            REFERENCES executions(execution_id),
                        FOREIGN KEY (execution_id, operation_id)
                            REFERENCES operations(execution_id, operation_id),
                        FOREIGN KEY (execution_id, trigger_store_seq)
                            REFERENCES events(execution_id, store_seq)
                    );
                    CREATE INDEX IF NOT EXISTS idx_context_builds_latest
                    ON context_builds(
                        execution_id, generation_id, trigger_store_seq DESC, build_id
                    );

                    COMMIT;
                    """
                )
                versions = {
                    int(row[0])
                    for row in connection.execute(
                        "SELECT version FROM context_compiler_v2_schema"
                    )
                }
                if versions != {_SCHEMA_VERSION}:
                    raise SQLiteContextCompilerV2IntegrityError(
                        "SQLite compiler-state schema version is unsupported"
                    )
                if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise SQLiteContextCompilerV2IntegrityError(
                        "SQLite compiler-state quick_check failed"
                    )
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                raise SQLiteContextCompilerV2IntegrityError(
                    "SQLite compiler-state schema initialization failed"
                ) from error
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    def bind_execution(
        self,
        execution_id: str,
        *,
        artifacts: ArtifactService,
    ) -> SQLiteContextCompilerV2Capabilities:
        normalized = _required_text(
            execution_id,
            "execution_id",
            identifier=True,
        )
        if not isinstance(artifacts, ArtifactService):
            raise TypeError("artifacts must be an ArtifactService")
        if artifacts.execution_id != normalized:
            raise ContextScopeError("artifact service belongs to another execution")
        with self._transaction(immediate=False) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM executions WHERE execution_id = ?",
                    (normalized,),
                ).fetchone()
                is None
            ):
                raise ContextScopeError("execution is not present in the context store")
        checkpoints = _SQLiteBoundCheckpointRepository(
            store=self,
            execution_id=normalized,
            artifacts=artifacts,
        )
        builds = _SQLiteBoundContextBuildRepository(
            store=self,
            execution_id=normalized,
        )
        return SQLiteContextCompilerV2Capabilities(
            checkpoints=checkpoints,
            context_builds=builds,
        )


class _SQLiteBoundCompilerPort:
    def __init__(self, *, store: SQLiteContextCompilerV2Store) -> None:
        self._store = store

    def _operation_row(
        self,
        connection: sqlite3.Connection,
        operation_id: str,
    ) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT payload_sha256, target_kind, target_key
            FROM operations
            WHERE execution_id = ? AND operation_id = ?
            """,
            (self.execution_id, operation_id),
        ).fetchone()

    def _claim_operation(
        self,
        connection: sqlite3.Connection,
        *,
        operation: OperationRef,
        target_kind: str,
        target_key: str,
    ) -> bool:
        previous = self._operation_row(connection, operation.operation_id)
        if previous is not None:
            if (
                previous["payload_sha256"] != operation.payload_sha256
                or previous["target_kind"] != target_kind
                or previous["target_key"] != target_key
            ):
                raise ContextConflictError(
                    "operation identity is already bound to another payload or target"
                )
            return False
        connection.execute(
            """
            INSERT INTO operations(
                execution_id, operation_id, payload_sha256,
                target_kind, target_key
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                self.execution_id,
                operation.operation_id,
                operation.payload_sha256,
                target_kind,
                target_key,
            ),
        )
        return True

    def _event_row(
        self,
        connection: sqlite3.Connection,
        cursor: EventCursor,
    ) -> sqlite3.Row:
        row = connection.execute(
            """
            SELECT event_id, generation_id, attempt_id
            FROM events WHERE execution_id = ? AND store_seq = ?
            """,
            (self.execution_id, cursor.store_seq),
        ).fetchone()
        if row is None or row["event_id"] != cursor.event_id:
            raise ContextScopeError("cursor is outside the bound journal")
        return row


class _SQLiteBoundCheckpointRepository(
    _SQLiteBoundCompilerPort,
    BoundCheckpointRepository,
):
    def __init__(
        self,
        *,
        store: SQLiteContextCompilerV2Store,
        execution_id: str,
        artifacts: ArtifactService,
    ) -> None:
        BoundCheckpointRepository.__init__(self, execution_id)
        _SQLiteBoundCompilerPort.__init__(self, store=store)
        self._artifacts = artifacts

    @staticmethod
    def _checkpoint_ref(checkpoint_id: str) -> ResourceRef:
        return ResourceRef("checkpoint", checkpoint_id, 1)

    def checkpoint_ref_for(self, *, operation: OperationRef) -> ResourceRef:
        if not isinstance(operation, OperationRef):
            operation = OperationRef.from_dict(operation)
        return self._checkpoint_ref(
            _checkpoint_identity(self.execution_id, operation)
        )

    def list_committed_refs(self, *, limit: int = 32) -> tuple[ResourceRef, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 < limit <= 64:
            raise ValueError("limit must be between 1 and 64")
        try:
            with self._store._transaction(immediate=False) as connection:
                rows = connection.execute(
                    """
                    SELECT checkpoint_id FROM checkpoints
                    WHERE execution_id = ? AND revision = 1 AND status = 'committed'
                    ORDER BY committed_at DESC, checkpoint_id DESC
                    LIMIT ?
                    """,
                    (self.execution_id, limit),
                ).fetchall()
            return tuple(self._checkpoint_ref(str(row["checkpoint_id"])) for row in rows)
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "checkpoint discovery failed"
            ) from error

    def _row_by_operation(
        self,
        connection: sqlite3.Connection,
        operation: OperationRef,
    ) -> sqlite3.Row | None:
        claimed = self._operation_row(connection, operation.operation_id)
        if claimed is None:
            return None
        if (
            claimed["payload_sha256"] != operation.payload_sha256
            or claimed["target_kind"] != "checkpoint"
        ):
            raise ContextConflictError(
                "checkpoint operation is bound to another payload or target"
            )
        row = connection.execute(
            """
            SELECT * FROM checkpoints
            WHERE execution_id = ? AND operation_id = ?
            """,
            (self.execution_id, operation.operation_id),
        ).fetchone()
        if row is None or row["checkpoint_id"] != claimed["target_key"]:
            raise SQLiteContextCompilerV2IntegrityError(
                "checkpoint operation has no exact durable target"
            )
        return row

    def _decode(
        self,
        row: sqlite3.Row,
        *,
        duplicate: bool,
    ) -> tuple[PreparedCheckpoint, Mapping[str, Any], ArtifactRef]:
        semantic = _verified_record(
            row["semantic_json"],
            row["semantic_sha256"],
            field_name="checkpoint semantic record",
        )
        artifact_record = _verified_record(
            row["artifact_json"],
            row["artifact_sha256"],
            field_name="checkpoint artifact record",
        )
        try:
            source_range = EventRange.from_dict(semantic["source_range"])
            refs = tuple(ResourceRef.from_dict(value) for value in semantic["refs"])
            artifact = ArtifactRef.from_dict(artifact_record)
            status = CheckpointWriteStatus(row["status"])
            operation = OperationRef(
                row["operation_id"],
                row["operation_payload_sha256"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SQLiteContextCompilerV2IntegrityError(
                "checkpoint durable record is invalid"
            ) from error
        expected_semantic = {
            "schema": "unchain.sqlite_checkpoint_prepare.v1",
            "source_range": source_range.to_dict(),
            "summary_sha256": semantic.get("summary_sha256"),
            "refs": [ref.to_dict() for ref in refs],
        }
        checkpoint_id = _checkpoint_identity(self.execution_id, operation)
        preparation_id = "preparation-" + checkpoint_id
        if (
            semantic != expected_semantic
            or row["execution_id"] != self.execution_id
            or row["checkpoint_id"] != checkpoint_id
            or int(row["revision"]) != 1
            or row["preparation_id"] != preparation_id
            or row["source_start_seq"] != source_range.start.store_seq
            or row["source_start_event_id"] != source_range.start.event_id
            or row["source_end_seq"] != source_range.end.store_seq
            or row["source_end_event_id"] != source_range.end.event_id
            or artifact.ref.kind != "artifact"
            or artifact.ref.fragment
        ):
            raise SQLiteContextCompilerV2IntegrityError(
                "checkpoint indexed fields changed"
            )
        receipt = PreparedCheckpoint(
            preparation_id=preparation_id,
            checkpoint_ref=self._checkpoint_ref(checkpoint_id),
            operation=operation,
            status=status,
            duplicate=duplicate,
        )
        return receipt, semantic, artifact

    def _existing_prepare(
        self,
        connection: sqlite3.Connection,
        *,
        operation: OperationRef,
        semantic: Mapping[str, Any],
    ) -> PreparedCheckpoint | None:
        row = self._row_by_operation(connection, operation)
        if row is None:
            return None
        receipt, durable_semantic, _artifact = self._decode(row, duplicate=True)
        if durable_semantic != semantic:
            raise ContextConflictError("checkpoint operation payload changed")
        return receipt

    def prepare(
        self,
        *,
        source_range: EventRange,
        summary: str,
        refs: tuple[ResourceRef, ...],
        operation: OperationRef,
    ) -> PreparedCheckpoint:
        if not isinstance(source_range, EventRange):
            source_range = EventRange.from_dict(source_range)
        if not isinstance(summary, str):
            raise TypeError("summary must be text")
        if not isinstance(operation, OperationRef):
            operation = OperationRef.from_dict(operation)
        normalized_refs = _resource_refs(refs)
        semantic = _checkpoint_semantic(
            source_range=source_range,
            summary=summary,
            refs=normalized_refs,
        )
        checkpoint_id = _checkpoint_identity(self.execution_id, operation)
        with self._store._transaction(immediate=False) as connection:
            existing = self._existing_prepare(
                connection,
                operation=operation,
                semantic=semantic,
            )
            if existing is not None:
                return existing
            self._event_row(connection, source_range.start)
            self._event_row(connection, source_range.end)

        artifact_operation_id = "checkpoint-object-" + _sha256(
            (
                operation.operation_id
                + "\0"
                + operation.payload_sha256
                + "\0"
                + _sha256(_canonical_json_bytes(semantic))
            ).encode("utf-8")
        )
        artifact = self._artifacts.persist(
            summary.encode("utf-8"),
            media_type="application/json",
            operation_id=artifact_operation_id,
        )
        semantic_bytes = _canonical_json_bytes(semantic)
        artifact_bytes = _canonical_json_bytes(artifact.to_dict())
        try:
            with self._store._transaction(immediate=True) as connection:
                existing = self._existing_prepare(
                    connection,
                    operation=operation,
                    semantic=semantic,
                )
                if existing is not None:
                    return existing
                self._event_row(connection, source_range.start)
                self._event_row(connection, source_range.end)
                created = self._claim_operation(
                    connection,
                    operation=operation,
                    target_kind="checkpoint",
                    target_key=checkpoint_id,
                )
                if not created:  # pragma: no cover - handled by existing lookup
                    raise SQLiteContextCompilerV2IntegrityError(
                        "checkpoint operation was claimed without a receipt"
                    )
                connection.execute(
                    """
                    INSERT INTO checkpoints(
                        execution_id, checkpoint_id, revision, preparation_id,
                        operation_id, operation_payload_sha256,
                        semantic_json, semantic_sha256,
                        source_start_seq, source_start_event_id,
                        source_end_seq, source_end_event_id,
                        artifact_json, artifact_sha256, status
                    ) VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'prepared')
                    """,
                    (
                        self.execution_id,
                        checkpoint_id,
                        "preparation-" + checkpoint_id,
                        operation.operation_id,
                        operation.payload_sha256,
                        semantic_bytes,
                        _sha256(semantic_bytes),
                        source_range.start.store_seq,
                        source_range.start.event_id,
                        source_range.end.store_seq,
                        source_range.end.event_id,
                        artifact_bytes,
                        _sha256(artifact_bytes),
                    ),
                )
                row = self._row_by_operation(connection, operation)
                if row is None:  # pragma: no cover - transaction invariant
                    raise SQLiteContextCompilerV2IntegrityError(
                        "checkpoint receipt disappeared during prepare"
                    )
                receipt, durable_semantic, durable_artifact = self._decode(
                    row,
                    duplicate=False,
                )
                if durable_semantic != semantic or durable_artifact != artifact:
                    raise SQLiteContextCompilerV2IntegrityError(
                        "checkpoint prepare receipt changed"
                    )
                return receipt
        except ContextRepositoryError:
            raise
        except sqlite3.IntegrityError as error:
            raise ContextConflictError("checkpoint persistence conflicted") from error
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "checkpoint persistence failed"
            ) from error

    def commit(self, *, prepared: PreparedCheckpoint) -> PreparedCheckpoint:
        if not isinstance(prepared, PreparedCheckpoint):
            prepared = PreparedCheckpoint(**prepared)
        if prepared.checkpoint_ref.kind != "checkpoint":
            raise ContextScopeError("prepared receipt is not a checkpoint")
        with self._store._transaction(immediate=False) as connection:
            row = self._row_by_operation(connection, prepared.operation)
            if row is None:
                raise ContextScopeError("checkpoint preparation is unavailable")
            current, _semantic, artifact = self._decode(row, duplicate=False)
        if (
            current.preparation_id != prepared.preparation_id
            or current.checkpoint_ref != prepared.checkpoint_ref
        ):
            raise ContextConflictError("checkpoint preparation identity changed")
        self._artifacts.read_full(artifact, remaining_budget_bytes=artifact.byte_length)
        if current.status is CheckpointWriteStatus.COMMITTED:
            return replace(current, duplicate=True)
        try:
            with self._store._transaction(immediate=True) as connection:
                row = self._row_by_operation(connection, prepared.operation)
                if row is None:
                    raise SQLiteContextCompilerV2IntegrityError(
                        "checkpoint preparation disappeared during commit"
                    )
                current, _semantic, _artifact = self._decode(row, duplicate=False)
                if (
                    current.preparation_id != prepared.preparation_id
                    or current.checkpoint_ref != prepared.checkpoint_ref
                ):
                    raise ContextConflictError(
                        "checkpoint preparation identity changed"
                    )
                if current.status is CheckpointWriteStatus.COMMITTED:
                    return replace(current, duplicate=True)
                updated = connection.execute(
                    """
                    UPDATE checkpoints
                    SET status = 'committed', committed_at = CURRENT_TIMESTAMP
                    WHERE execution_id = ? AND checkpoint_id = ?
                      AND revision = 1 AND status = 'prepared'
                    """,
                    (self.execution_id, current.checkpoint_ref.resource_id),
                )
                if updated.rowcount != 1:
                    raise ContextConflictError("checkpoint commit CAS failed")
                return replace(
                    current,
                    status=CheckpointWriteStatus.COMMITTED,
                )
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "checkpoint commit persistence failed"
            ) from error

    def get_by_operation(
        self,
        *,
        operation: OperationRef,
    ) -> PreparedCheckpoint | None:
        if not isinstance(operation, OperationRef):
            operation = OperationRef.from_dict(operation)
        try:
            with self._store._transaction(immediate=False) as connection:
                row = self._row_by_operation(connection, operation)
                if row is None:
                    return None
                receipt, _semantic, _artifact = self._decode(row, duplicate=False)
                return receipt
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "checkpoint receipt read failed"
            ) from error

    def get_by_ref(self, *, ref: ResourceRef) -> PreparedCheckpoint | None:
        if not isinstance(ref, ResourceRef):
            ref = ResourceRef.from_dict(ref)
        if ref.kind != "checkpoint" or ref.fragment or ref.revision != 1:
            raise ContextScopeError("checkpoint ref is outside the bound repository")
        try:
            with self._store._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE execution_id = ? AND checkpoint_id = ? AND revision = 1
                    """,
                    (self.execution_id, ref.resource_id),
                ).fetchone()
                if row is None:
                    return None
                receipt, _semantic, _artifact = self._decode(
                    row,
                    duplicate=False,
                )
                if receipt.checkpoint_ref != ref:
                    raise SQLiteContextCompilerV2IntegrityError(
                        "checkpoint ref changed outside the bound repository"
                    )
                claimed = self._operation_row(
                    connection,
                    receipt.operation.operation_id,
                )
                if (
                    claimed is None
                    or claimed["payload_sha256"]
                    != receipt.operation.payload_sha256
                    or claimed["target_kind"] != "checkpoint"
                    or claimed["target_key"] != ref.resource_id
                ):
                    raise SQLiteContextCompilerV2IntegrityError(
                        "checkpoint operation claim is invalid"
                    )
                return receipt
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "checkpoint metadata read failed"
            ) from error

    def read(
        self,
        *,
        ref: ResourceRef,
        offset: int = 0,
        limit: int = 65_536,
    ) -> bytes:
        if not isinstance(ref, ResourceRef):
            ref = ResourceRef.from_dict(ref)
        if ref.kind != "checkpoint" or ref.fragment or ref.revision != 1:
            raise ContextScopeError("checkpoint ref is outside the bound repository")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        try:
            with self._store._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM checkpoints
                    WHERE execution_id = ? AND checkpoint_id = ? AND revision = 1
                    """,
                    (self.execution_id, ref.resource_id),
                ).fetchone()
                if row is None:
                    raise ContextScopeError("checkpoint is unavailable")
                receipt, _semantic, artifact = self._decode(row, duplicate=False)
                if receipt.status is not CheckpointWriteStatus.COMMITTED:
                    raise ContextScopeError("checkpoint is not committed")
            return self._artifacts.read_page(
                artifact,
                offset=offset,
                limit=limit,
            ).data
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error("checkpoint read failed") from error


class _SQLiteBoundContextBuildRepository(
    _SQLiteBoundCompilerPort,
    BoundContextBuildRepository,
):
    def __init__(
        self,
        *,
        store: SQLiteContextCompilerV2Store,
        execution_id: str,
    ) -> None:
        BoundContextBuildRepository.__init__(self, execution_id)
        _SQLiteBoundCompilerPort.__init__(self, store=store)

    def _decode(self, row: sqlite3.Row, *, duplicate: bool) -> ContextBuildReceipt:
        envelope_record = _verified_record(
            row["envelope_json"],
            row["envelope_sha256"],
            field_name="context build envelope",
        )
        try:
            envelope = ContextBuildEnvelope.from_dict(envelope_record)
            operation = OperationRef(
                row["operation_id"],
                row["operation_payload_sha256"],
            )
            cursor = EventCursor(
                row["trigger_store_seq"],
                row["trigger_event_id"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SQLiteContextCompilerV2IntegrityError(
                "context build durable record is invalid"
            ) from error
        semantic = _build_semantic(envelope=envelope, trigger_cursor=cursor)
        if (
            row["execution_id"] != self.execution_id
            or envelope.to_dict() != envelope_record
            or envelope.execution_id != self.execution_id
            or row["build_id"] != envelope.build_id
            or row["generation_id"] != envelope.generation_id
            or row["attempt_id"] != envelope.attempt_id
            or row["semantic_sha256"] != _sha256(_canonical_json_bytes(semantic))
        ):
            raise SQLiteContextCompilerV2IntegrityError(
                "context build indexed fields or semantic digest changed"
            )
        return ContextBuildReceipt(
            envelope=envelope,
            operation=operation,
            trigger_cursor=cursor,
            duplicate=duplicate,
        )

    def _row_by_operation(
        self,
        connection: sqlite3.Connection,
        operation: OperationRef,
    ) -> sqlite3.Row | None:
        claimed = self._operation_row(connection, operation.operation_id)
        if claimed is None:
            return None
        if (
            claimed["payload_sha256"] != operation.payload_sha256
            or claimed["target_kind"] != "context_build"
        ):
            raise ContextConflictError(
                "context build operation is bound to another payload or target"
            )
        row = connection.execute(
            """
            SELECT * FROM context_builds
            WHERE execution_id = ? AND operation_id = ?
            """,
            (self.execution_id, operation.operation_id),
        ).fetchone()
        if row is None or row["build_id"] != claimed["target_key"]:
            raise SQLiteContextCompilerV2IntegrityError(
                "context build operation has no exact durable target"
            )
        return row

    def record(
        self,
        *,
        envelope: ContextBuildEnvelope,
        operation: OperationRef,
        trigger_cursor: EventCursor,
    ) -> ContextBuildReceipt:
        if not isinstance(envelope, ContextBuildEnvelope):
            envelope = ContextBuildEnvelope.from_dict(envelope)
        if not isinstance(operation, OperationRef):
            operation = OperationRef.from_dict(operation)
        if not isinstance(trigger_cursor, EventCursor):
            trigger_cursor = EventCursor.from_dict(trigger_cursor)
        if envelope.execution_id != self.execution_id:
            raise ContextScopeError("context build belongs to another execution")
        semantic = _build_semantic(
            envelope=envelope,
            trigger_cursor=trigger_cursor,
        )
        semantic_sha256 = _sha256(_canonical_json_bytes(semantic))
        envelope_bytes = _canonical_json_bytes(envelope.to_dict())
        try:
            with self._store._transaction(immediate=True) as connection:
                existing = self._row_by_operation(connection, operation)
                if existing is not None:
                    receipt = self._decode(existing, duplicate=True)
                    if (
                        receipt.envelope != envelope
                        or receipt.trigger_cursor != trigger_cursor
                    ):
                        raise ContextConflictError(
                            "context build operation payload changed"
                        )
                    return receipt
                event = self._event_row(connection, trigger_cursor)
                if (
                    event["generation_id"] != envelope.generation_id
                    or event["attempt_id"] != envelope.attempt_id
                ):
                    raise ContextScopeError(
                        "context build trigger belongs to another attempt"
                    )
                trigger_claim = connection.execute(
                    """
                    SELECT * FROM context_builds
                    WHERE execution_id = ? AND trigger_store_seq = ?
                      AND trigger_event_id = ?
                    """,
                    (
                        self.execution_id,
                        trigger_cursor.store_seq,
                        trigger_cursor.event_id,
                    ),
                ).fetchone()
                if trigger_claim is not None:
                    raise ContextConflictError(
                        "context build trigger is already claimed"
                    )
                build_claim = connection.execute(
                    """
                    SELECT 1 FROM context_builds
                    WHERE execution_id = ? AND build_id = ?
                    """,
                    (self.execution_id, envelope.build_id),
                ).fetchone()
                if build_claim is not None:
                    raise ContextConflictError("context build ID is already claimed")
                self._claim_operation(
                    connection,
                    operation=operation,
                    target_kind="context_build",
                    target_key=envelope.build_id,
                )
                connection.execute(
                    """
                    INSERT INTO context_builds(
                        execution_id, build_id, generation_id, attempt_id,
                        operation_id, operation_payload_sha256,
                        trigger_store_seq, trigger_event_id, semantic_sha256,
                        envelope_json, envelope_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.execution_id,
                        envelope.build_id,
                        envelope.generation_id,
                        envelope.attempt_id,
                        operation.operation_id,
                        operation.payload_sha256,
                        trigger_cursor.store_seq,
                        trigger_cursor.event_id,
                        semantic_sha256,
                        envelope_bytes,
                        _sha256(envelope_bytes),
                    ),
                )
                row = self._row_by_operation(connection, operation)
                if row is None:  # pragma: no cover - transaction invariant
                    raise SQLiteContextCompilerV2IntegrityError(
                        "context build receipt disappeared during record"
                    )
                return self._decode(row, duplicate=False)
        except ContextRepositoryError:
            raise
        except sqlite3.IntegrityError as error:
            raise ContextConflictError(
                "context build persistence conflicted"
            ) from error
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "context build persistence failed"
            ) from error

    def get_by_operation(
        self,
        *,
        operation: OperationRef,
    ) -> ContextBuildReceipt | None:
        if not isinstance(operation, OperationRef):
            operation = OperationRef.from_dict(operation)
        try:
            with self._store._transaction(immediate=False) as connection:
                row = self._row_by_operation(connection, operation)
                return None if row is None else self._decode(row, duplicate=False)
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "context build receipt read failed"
            ) from error

    def get_by_trigger(
        self,
        *,
        trigger_cursor: EventCursor,
    ) -> ContextBuildReceipt | None:
        if not isinstance(trigger_cursor, EventCursor):
            trigger_cursor = EventCursor.from_dict(trigger_cursor)
        try:
            with self._store._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM context_builds
                    WHERE execution_id = ? AND trigger_store_seq = ?
                      AND trigger_event_id = ?
                    """,
                    (
                        self.execution_id,
                        trigger_cursor.store_seq,
                        trigger_cursor.event_id,
                    ),
                ).fetchone()
                return None if row is None else self._decode(row, duplicate=False)
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "context build trigger read failed"
            ) from error

    def latest(self, *, generation_id: str) -> ContextBuildEnvelope | None:
        normalized = _required_text(
            generation_id,
            "generation_id",
            identifier=True,
        )
        try:
            with self._store._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM context_builds
                    WHERE execution_id = ? AND generation_id = ?
                    ORDER BY trigger_store_seq DESC, build_id DESC
                    LIMIT 1
                    """,
                    (self.execution_id, normalized),
                ).fetchone()
                if row is None:
                    return None
                return self._decode(row, duplicate=False).envelope
        except ContextRepositoryError:
            raise
        except sqlite3.Error as error:
            raise SQLiteContextCompilerV2Error(
                "latest context build read failed"
            ) from error


__all__ = [
    "SQLiteContextCompilerV2Capabilities",
    "SQLiteContextCompilerV2Error",
    "SQLiteContextCompilerV2IntegrityError",
    "SQLiteContextCompilerV2Store",
]
