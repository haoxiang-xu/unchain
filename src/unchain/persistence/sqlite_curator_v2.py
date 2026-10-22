"""SQLite/WAL persistence for Memory V2 candidates and curator jobs.

This module deliberately owns only the candidate/job side of the shared
``context_v2.sqlite3`` data plane.  Workspace mutations remain host supplied;
the repository exposes an exact lease fence that those adapters can check at
their own durable mutation boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from unchain.journal import OperationRef, ResourceRef
from unchain.journal.models import _required_text
from unchain.memory.curator._validation import canonical_candidate_path
from unchain.memory.curator.models import (
    CandidateOrigin,
    CandidateResolution,
    CandidateStatus,
    ConsolidationJob,
    ConsolidationJobStatus,
    CuratorLeaseFence,
    EnqueueRequest,
    FrozenCandidateSnapshot,
    Lease,
    RootRunCompletion,
    RunCaptureStatus,
    SourceRunStatus,
)
from unchain.memory.curator.ports import (
    BoundCurationRepository,
    BoundCuratorMutationGuard,
    CurationConflictError,
    CurationRepositoryError,
)
from unchain.memory.toolkit.models import (
    CandidateProposalRequest,
    MemoryToolContentPage,
    MemoryToolkitRunBinding,
)


_SHA256_LENGTH = 64
_SCHEMA_VERSION = 1


class SQLiteCuratorV2IntegrityError(CurationRepositoryError):
    """Durable curator metadata or CAS bytes failed exact verification."""


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
        raise SQLiteCuratorV2IntegrityError("record_not_canonical_json") from exc


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _semantic_digest(value: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(value))


def _exact_non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _exact_positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _candidate_from_dict(value: Mapping[str, Any]) -> FrozenCandidateSnapshot:
    required = {
        "candidate_ref",
        "target_space_id",
        "binding_revision",
        "outcome",
        "origin",
        "target_path",
        "name",
        "description",
        "kind",
        "media_type",
        "content_ref",
        "link_url",
        "source_refs",
        "source_agent_run_id",
        "source_tool_call_id",
        "rationale",
        "confidence",
        "sensitivity",
        "payload_sha256",
        "content_sha256",
        "byte_length",
        "result_ref",
        "review_diff",
        "error_code",
    }
    raw = dict(value)
    if set(raw) != required:
        raise SQLiteCuratorV2IntegrityError("candidate_record_shape_changed")
    try:
        candidate = FrozenCandidateSnapshot(
            candidate_ref=ResourceRef.from_dict(raw["candidate_ref"]),
            target_space_id=raw["target_space_id"],
            binding_revision=raw["binding_revision"],
            outcome=raw["outcome"],
            origin=raw["origin"],
            target_path=raw["target_path"],
            name=raw["name"],
            description=raw["description"],
            kind=raw["kind"],
            media_type=raw["media_type"],
            content_ref=(
                ResourceRef.from_dict(raw["content_ref"])
                if raw["content_ref"] is not None
                else None
            ),
            link_url=raw["link_url"],
            source_refs=tuple(
                ResourceRef.from_dict(item) for item in raw["source_refs"]
            ),
            source_agent_run_id=raw["source_agent_run_id"],
            source_tool_call_id=raw["source_tool_call_id"],
            rationale=raw["rationale"],
            confidence=raw["confidence"],
            sensitivity=raw["sensitivity"],
            payload_sha256=raw["payload_sha256"],
            content_sha256=raw["content_sha256"],
            byte_length=raw["byte_length"],
            result_ref=(
                ResourceRef.from_dict(raw["result_ref"])
                if raw["result_ref"] is not None
                else None
            ),
            review_diff=raw["review_diff"],
            error_code=raw["error_code"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SQLiteCuratorV2IntegrityError("candidate_record_invalid") from exc
    if candidate.to_dict() != raw:
        raise SQLiteCuratorV2IntegrityError("candidate_record_not_canonical")
    return candidate


def _completion_from_dict(value: Mapping[str, Any]) -> RootRunCompletion:
    raw = dict(value)
    if set(raw) != {
        "session_id",
        "attempt_id",
        "run_id",
        "is_root_run",
        "run_status",
        "capture_status",
        "trigger_key",
    }:
        raise SQLiteCuratorV2IntegrityError("root_completion_shape_changed")
    try:
        completion = RootRunCompletion(
            session_id=raw["session_id"],
            attempt_id=raw["attempt_id"],
            run_id=raw["run_id"],
            is_root_run=raw["is_root_run"],
            run_status=SourceRunStatus(raw["run_status"]),
            capture_status=RunCaptureStatus(raw["capture_status"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SQLiteCuratorV2IntegrityError("root_completion_invalid") from exc
    if completion.to_dict() != raw:
        raise SQLiteCuratorV2IntegrityError("root_completion_not_canonical")
    return completion


def _job_to_dict(job: ConsolidationJob) -> dict[str, Any]:
    if not isinstance(job, ConsolidationJob):
        raise TypeError("job must be a ConsolidationJob")
    lease = None
    if job.lease is not None:
        lease = {
            "owner": job.lease.owner,
            "token": job.lease.token,
            "expires_at_ms": job.lease.expires_at_ms,
        }
    return {
        "schema": "unchain.consolidation_job.v2",
        "job_id": job.job_id,
        "trigger": job.trigger.to_dict(),
        "candidates": [item.to_dict() for item in job.candidates],
        "status": job.status.value,
        "revision": job.revision,
        "operation_id": job.operation_id,
        "created_at_ms": job.created_at_ms,
        "updated_at_ms": job.updated_at_ms,
        "lease": lease,
        "attempt_count": job.attempt_count,
        "next_attempt_at_ms": job.next_attempt_at_ms,
        "last_error_code": job.last_error_code,
    }


def _job_from_dict(value: Mapping[str, Any]) -> ConsolidationJob:
    raw = dict(value)
    required = {
        "schema",
        "job_id",
        "trigger",
        "candidates",
        "status",
        "revision",
        "operation_id",
        "created_at_ms",
        "updated_at_ms",
        "lease",
        "attempt_count",
        "next_attempt_at_ms",
        "last_error_code",
    }
    if set(raw) != required or raw.get("schema") != "unchain.consolidation_job.v2":
        raise SQLiteCuratorV2IntegrityError("consolidation_job_shape_changed")
    lease_value = raw["lease"]
    if lease_value is not None:
        if not isinstance(lease_value, Mapping) or set(lease_value) != {
            "owner",
            "token",
            "expires_at_ms",
        }:
            raise SQLiteCuratorV2IntegrityError("lease_record_shape_changed")
        try:
            lease = Lease(
                owner=lease_value["owner"],
                token=lease_value["token"],
                expires_at_ms=lease_value["expires_at_ms"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SQLiteCuratorV2IntegrityError("lease_record_invalid") from exc
    else:
        lease = None
    try:
        job = ConsolidationJob(
            job_id=raw["job_id"],
            trigger=_completion_from_dict(raw["trigger"]),
            candidates=tuple(_candidate_from_dict(item) for item in raw["candidates"]),
            status=ConsolidationJobStatus(raw["status"]),
            revision=raw["revision"],
            operation_id=raw["operation_id"],
            created_at_ms=raw["created_at_ms"],
            updated_at_ms=raw["updated_at_ms"],
            lease=lease,
            attempt_count=raw["attempt_count"],
            next_attempt_at_ms=raw["next_attempt_at_ms"],
            last_error_code=raw["last_error_code"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SQLiteCuratorV2IntegrityError("consolidation_job_invalid") from exc
    if _job_to_dict(job) != raw:
        raise SQLiteCuratorV2IntegrityError("consolidation_job_not_canonical")
    return job


class SQLiteCuratorV2Store:
    """Own candidate and curation tables in the shared Context V2 data plane."""

    def __init__(
        self,
        *,
        database_path: str | os.PathLike[str],
        object_directory: str | os.PathLike[str],
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.object_directory = Path(object_directory)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.object_directory.mkdir(parents=True, exist_ok=True)
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
        connection = self._connect()
        try:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).casefold() != "wal":
                raise SQLiteCuratorV2IntegrityError("sqlite_wal_unavailable")
            connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS curator_v2_schema (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                INSERT OR IGNORE INTO curator_v2_schema(version) VALUES (1);

                CREATE TABLE IF NOT EXISTS objects (
                    sha256 TEXT PRIMARY KEY,
                    byte_length INTEGER NOT NULL CHECK(byte_length >= 0)
                );

                CREATE TABLE IF NOT EXISTS curation_scopes (
                    binding_id TEXT PRIMARY KEY,
                    owner_chat_id TEXT NOT NULL,
                    target_space_id TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0)
                );

                CREATE TABLE IF NOT EXISTS curation_run_scopes (
                    binding_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    root_run_id TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    PRIMARY KEY (binding_id, session_id, attempt_id, run_id),
                    FOREIGN KEY (binding_id) REFERENCES curation_scopes(binding_id)
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    binding_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    current_record_revision INTEGER NOT NULL
                        CHECK(current_record_revision >= 1),
                    status TEXT NOT NULL,
                    object_sha256 TEXT,
                    byte_length INTEGER NOT NULL CHECK(byte_length >= 0),
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    updated_at_ms INTEGER NOT NULL CHECK(updated_at_ms >= 0),
                    FOREIGN KEY (binding_id) REFERENCES curation_scopes(binding_id),
                    FOREIGN KEY (object_sha256) REFERENCES objects(sha256)
                );
                CREATE INDEX IF NOT EXISTS idx_candidates_source_status
                    ON candidates(
                        binding_id,
                        session_id,
                        attempt_id,
                        run_id,
                        status,
                        candidate_id
                    );

                CREATE TABLE IF NOT EXISTS candidate_revisions (
                    candidate_id TEXT NOT NULL,
                    record_revision INTEGER NOT NULL CHECK(record_revision >= 1),
                    snapshot_json BLOB NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    PRIMARY KEY (candidate_id, record_revision),
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS consolidation_jobs (
                    job_id TEXT PRIMARY KEY,
                    binding_id TEXT NOT NULL,
                    trigger_key TEXT NOT NULL,
                    current_revision INTEGER NOT NULL CHECK(current_revision >= 1),
                    status TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_token TEXT,
                    lease_expires_at_ms INTEGER,
                    next_attempt_at_ms INTEGER NOT NULL CHECK(next_attempt_at_ms >= 0),
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    updated_at_ms INTEGER NOT NULL CHECK(updated_at_ms >= 0),
                    UNIQUE (binding_id, trigger_key),
                    FOREIGN KEY (binding_id) REFERENCES curation_scopes(binding_id)
                );
                CREATE INDEX IF NOT EXISTS idx_consolidation_jobs_claim
                    ON consolidation_jobs(
                        binding_id,
                        status,
                        next_attempt_at_ms,
                        lease_expires_at_ms,
                        created_at_ms,
                        job_id
                    );

                CREATE TABLE IF NOT EXISTS consolidation_job_revisions (
                    job_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK(revision >= 1),
                    job_json BLOB NOT NULL,
                    job_sha256 TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    PRIMARY KEY (job_id, revision),
                    FOREIGN KEY (job_id) REFERENCES consolidation_jobs(job_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS candidate_bindings (
                    candidate_id TEXT NOT NULL,
                    binding_revision INTEGER NOT NULL CHECK(binding_revision >= 1),
                    job_id TEXT,
                    target_space_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_ref_json BLOB,
                    review_diff_json BLOB NOT NULL,
                    error_code TEXT NOT NULL,
                    candidate_record_revision INTEGER NOT NULL
                        CHECK(candidate_record_revision >= 1),
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    PRIMARY KEY (candidate_id, binding_revision),
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (job_id) REFERENCES consolidation_jobs(job_id)
                );

                CREATE TABLE IF NOT EXISTS curator_operation_receipts (
                    binding_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    operation_kind TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    semantic_sha256 TEXT NOT NULL,
                    result_kind TEXT NOT NULL,
                    result_key TEXT,
                    result_revision INTEGER,
                    result_count INTEGER,
                    created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                    PRIMARY KEY (binding_id, operation_id),
                    FOREIGN KEY (binding_id) REFERENCES curation_scopes(binding_id)
                );
                """
            )
            run_scope_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(curation_run_scopes)"
                )
            }
            if "root_run_id" not in run_scope_columns:
                connection.execute(
                    """
                    CREATE TABLE curation_run_scopes_with_root_lineage (
                        binding_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        attempt_id TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        root_run_id TEXT NOT NULL,
                        created_at_ms INTEGER NOT NULL CHECK(created_at_ms >= 0),
                        PRIMARY KEY (binding_id, session_id, attempt_id, run_id),
                        FOREIGN KEY (binding_id)
                            REFERENCES curation_scopes(binding_id)
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO curation_run_scopes_with_root_lineage(
                        binding_id,
                        session_id,
                        attempt_id,
                        run_id,
                        root_run_id,
                        created_at_ms
                    )
                    SELECT
                        binding_id,
                        session_id,
                        attempt_id,
                        run_id,
                        run_id,
                        created_at_ms
                    FROM curation_run_scopes
                    """
                )
                connection.execute("DROP TABLE curation_run_scopes")
                connection.execute(
                    """
                    ALTER TABLE curation_run_scopes_with_root_lineage
                    RENAME TO curation_run_scopes
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_curation_run_scopes_root
                ON curation_run_scopes(
                    binding_id,
                    session_id,
                    root_run_id,
                    attempt_id,
                    run_id
                )
                """
            )
            versions = {
                int(row[0])
                for row in connection.execute("SELECT version FROM curator_v2_schema")
            }
            if versions != {_SCHEMA_VERSION}:
                raise SQLiteCuratorV2IntegrityError(
                    "curator_schema_version_unsupported"
                )
            if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise SQLiteCuratorV2IntegrityError("sqlite_quick_check_failed")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

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

    def _object_path(self, digest: str) -> Path:
        if (
            not isinstance(digest, str)
            or len(digest) != _SHA256_LENGTH
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SQLiteCuratorV2IntegrityError("object_digest_invalid")
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

    def _read_object(self, *, digest: str, byte_length: int) -> bytes:
        try:
            content = self._object_path(digest).read_bytes()
        except OSError as exc:
            raise SQLiteCuratorV2IntegrityError("candidate_object_unreadable") from exc
        if len(content) != byte_length or _sha256(content) != digest:
            raise SQLiteCuratorV2IntegrityError("candidate_object_digest_changed")
        return content

    def _install_object(self, content: bytes) -> tuple[str, int]:
        if type(content) is not bytes:
            raise TypeError("candidate content must be exact bytes")
        digest = _sha256(content)
        target = self._object_path(digest)
        if target.exists():
            self._read_object(digest=digest, byte_length=len(content))
            return digest, len(content)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".memory-v2-candidate-",
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

    def bind_curation(
        self,
        *,
        binding_id: str,
        owner_chat_id: str,
        target_space_id: str,
    ) -> _SQLiteBoundCurationRepository:
        normalized_binding = _required_text(
            binding_id,
            "binding_id",
            maximum=512,
            identifier=True,
        )
        normalized_owner = _required_text(
            owner_chat_id,
            "owner_chat_id",
            maximum=512,
            identifier=True,
        )
        normalized_space = _required_text(
            target_space_id,
            "target_space_id",
            maximum=512,
            identifier=True,
        )
        now_ms = _exact_non_negative_int(self._clock_ms(), "clock_ms")
        try:
            with self._transaction(immediate=True) as connection:
                row = connection.execute(
                    "SELECT * FROM curation_scopes WHERE binding_id = ?",
                    (normalized_binding,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO curation_scopes(
                            binding_id, owner_chat_id, target_space_id, created_at_ms
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            normalized_binding,
                            normalized_owner,
                            normalized_space,
                            now_ms,
                        ),
                    )
                elif (
                    row["owner_chat_id"] != normalized_owner
                    or row["target_space_id"] != normalized_space
                ):
                    raise CurationRepositoryError("curation_scope_mismatch")
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("curation_scope_write_failed") from exc
        return _SQLiteBoundCurationRepository(
            store=self,
            binding_id=normalized_binding,
            owner_chat_id=normalized_owner,
            target_space_id=normalized_space,
        )


class _SQLiteBoundCandidateProposalCapability:
    def __init__(
        self,
        repository: _SQLiteBoundCurationRepository,
        binding: MemoryToolkitRunBinding,
    ) -> None:
        self._repository = repository
        self._binding = binding
        self.binding_id = binding.binding_id

    def propose(
        self,
        *,
        request: CandidateProposalRequest,
    ) -> FrozenCandidateSnapshot:
        return self._repository._propose(binding=self._binding, request=request)


class _SQLiteCuratorMutationGuard(BoundCuratorMutationGuard):
    def __init__(
        self,
        repository: _SQLiteBoundCurationRepository,
        fence: CuratorLeaseFence,
    ) -> None:
        self._repository = repository
        self.fence = fence

    def assert_active(self) -> None:
        self._repository._assert_fence(
            self.fence,
            now_ms=_exact_non_negative_int(
                self._repository._store._clock_ms(),
                "clock_ms",
            ),
        )


class _SQLiteBoundCurationRepository(BoundCurationRepository):
    def __init__(
        self,
        *,
        store: SQLiteCuratorV2Store,
        binding_id: str,
        owner_chat_id: str,
        target_space_id: str,
    ) -> None:
        self._store = store
        self.binding_id = binding_id
        self.owner_chat_id = owner_chat_id
        self.target_space_id = target_space_id

    def _scope_row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM curation_scopes WHERE binding_id = ?",
            (self.binding_id,),
        ).fetchone()
        if (
            row is None
            or row["owner_chat_id"] != self.owner_chat_id
            or row["target_space_id"] != self.target_space_id
        ):
            raise CurationRepositoryError("curation_scope_mismatch")
        return row

    def bind_candidate_proposals(
        self,
        *,
        binding: MemoryToolkitRunBinding,
        root_run_id: str | None = None,
    ) -> _SQLiteBoundCandidateProposalCapability:
        if not isinstance(binding, MemoryToolkitRunBinding):
            raise TypeError("binding must be a MemoryToolkitRunBinding")
        if binding.binding_id != self.binding_id:
            raise CurationRepositoryError("candidate_binding_scope_mismatch")
        normalized_root_run_id = _required_text(
            binding.run_id if root_run_id is None else root_run_id,
            "root_run_id",
            maximum=512,
            identifier=True,
        )
        now_ms = _exact_non_negative_int(self._store._clock_ms(), "clock_ms")
        try:
            with self._store._transaction(immediate=True) as connection:
                self._scope_row(connection)
                existing = connection.execute(
                    """
                    SELECT root_run_id FROM curation_run_scopes
                    WHERE binding_id = ?
                      AND session_id = ?
                      AND attempt_id = ?
                      AND run_id = ?
                    """,
                    (
                        self.binding_id,
                        binding.session_id,
                        binding.attempt_id,
                        binding.run_id,
                    ),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO curation_run_scopes(
                            binding_id,
                            session_id,
                            attempt_id,
                            run_id,
                            root_run_id,
                            created_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.binding_id,
                            binding.session_id,
                            binding.attempt_id,
                            binding.run_id,
                            normalized_root_run_id,
                            now_ms,
                        ),
                    )
                elif existing["root_run_id"] != normalized_root_run_id:
                    raise CurationRepositoryError(
                        "candidate_root_run_scope_mismatch"
                    )
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationRepositoryError("candidate_run_scope_mismatch") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError("candidate_scope_write_failed") from exc
        return _SQLiteBoundCandidateProposalCapability(self, binding)

    def _assert_completion_scope(
        self,
        connection: sqlite3.Connection,
        completion: RootRunCompletion,
    ) -> None:
        if not isinstance(completion, RootRunCompletion):
            raise TypeError("completion must be a RootRunCompletion")
        self._scope_row(connection)
        row = connection.execute(
            """
            SELECT binding_id FROM curation_run_scopes
            WHERE binding_id = ?
              AND session_id = ?
              AND attempt_id = ?
              AND run_id = ?
            """,
            (
                self.binding_id,
                completion.session_id,
                completion.attempt_id,
                completion.run_id,
            ),
        ).fetchone()
        if row is None:
            raise CurationRepositoryError("root_completion_scope_mismatch")

    @staticmethod
    def _decode_record(raw_value: object, digest: object) -> Mapping[str, Any]:
        try:
            raw = bytes(raw_value)
        except (TypeError, ValueError) as exc:
            raise SQLiteCuratorV2IntegrityError("durable_record_not_bytes") from exc
        if not isinstance(digest, str) or _sha256(raw) != digest:
            raise SQLiteCuratorV2IntegrityError("durable_record_digest_changed")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SQLiteCuratorV2IntegrityError("durable_record_invalid_json") from exc
        if not isinstance(decoded, Mapping) or _canonical_json_bytes(decoded) != raw:
            raise SQLiteCuratorV2IntegrityError("durable_record_not_canonical")
        return decoded

    def _candidate_revision(
        self,
        connection: sqlite3.Connection,
        *,
        candidate_id: str,
        record_revision: int,
    ) -> FrozenCandidateSnapshot:
        row = connection.execute(
            """
            SELECT snapshot_json, snapshot_sha256
            FROM candidate_revisions
            WHERE candidate_id = ? AND record_revision = ?
            """,
            (candidate_id, record_revision),
        ).fetchone()
        if row is None:
            raise SQLiteCuratorV2IntegrityError("candidate_revision_missing")
        candidate = _candidate_from_dict(
            self._decode_record(row["snapshot_json"], row["snapshot_sha256"])
        )
        if candidate.candidate_ref.resource_id != candidate_id:
            raise SQLiteCuratorV2IntegrityError("candidate_revision_identity_changed")
        return candidate

    def _current_candidate(
        self,
        connection: sqlite3.Connection,
        *,
        candidate_id: str,
    ) -> FrozenCandidateSnapshot:
        row = connection.execute(
            """
            SELECT binding_id, current_record_revision, status
            FROM candidates WHERE candidate_id = ?
            """,
            (candidate_id,),
        ).fetchone()
        if row is None or row["binding_id"] != self.binding_id:
            raise CurationRepositoryError("candidate_scope_mismatch")
        candidate = self._candidate_revision(
            connection,
            candidate_id=candidate_id,
            record_revision=row["current_record_revision"],
        )
        if candidate.outcome.value != row["status"]:
            raise SQLiteCuratorV2IntegrityError("candidate_head_status_changed")
        return candidate

    def _job_revision(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        revision: int,
    ) -> ConsolidationJob:
        row = connection.execute(
            """
            SELECT job_json, job_sha256
            FROM consolidation_job_revisions
            WHERE job_id = ? AND revision = ?
            """,
            (job_id, revision),
        ).fetchone()
        if row is None:
            raise SQLiteCuratorV2IntegrityError("consolidation_job_revision_missing")
        job = _job_from_dict(self._decode_record(row["job_json"], row["job_sha256"]))
        if job.job_id != job_id or job.revision != revision:
            raise SQLiteCuratorV2IntegrityError("consolidation_job_identity_changed")
        return job

    def _current_job(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
    ) -> ConsolidationJob:
        row = connection.execute(
            """
            SELECT binding_id, current_revision, status
            FROM consolidation_jobs WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if row is None or row["binding_id"] != self.binding_id:
            raise CurationRepositoryError("consolidation_job_scope_mismatch")
        job = self._job_revision(
            connection,
            job_id=job_id,
            revision=row["current_revision"],
        )
        if job.status.value != row["status"]:
            raise SQLiteCuratorV2IntegrityError("consolidation_job_head_changed")
        return job

    def _receipt(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        operation_kind: str,
        payload_sha256: str,
        semantic_sha256: str,
    ) -> sqlite3.Row | None:
        row = connection.execute(
            """
            SELECT * FROM curator_operation_receipts
            WHERE binding_id = ? AND operation_id = ?
            """,
            (self.binding_id, operation_id),
        ).fetchone()
        if row is None:
            return None
        if (
            row["binding_id"] != self.binding_id
            or row["operation_kind"] != operation_kind
            or row["payload_sha256"] != payload_sha256
            or row["semantic_sha256"] != semantic_sha256
        ):
            raise CurationConflictError("operation_payload_conflict")
        return row

    def _receipt_result(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> FrozenCandidateSnapshot | ConsolidationJob | int | None:
        result_kind = row["result_kind"]
        if result_kind == "candidate":
            return self._candidate_revision(
                connection,
                candidate_id=row["result_key"],
                record_revision=row["result_revision"],
            )
        if result_kind == "job":
            return self._job_revision(
                connection,
                job_id=row["result_key"],
                revision=row["result_revision"],
            )
        if result_kind == "count":
            return int(row["result_count"])
        if result_kind == "none":
            return None
        raise SQLiteCuratorV2IntegrityError("operation_receipt_result_invalid")

    def _record_receipt(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        operation_kind: str,
        payload_sha256: str,
        semantic_sha256: str,
        result_kind: str,
        result_key: str | None = None,
        result_revision: int | None = None,
        result_count: int | None = None,
        now_ms: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO curator_operation_receipts(
                operation_id,
                binding_id,
                operation_kind,
                payload_sha256,
                semantic_sha256,
                result_kind,
                result_key,
                result_revision,
                result_count,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                self.binding_id,
                operation_kind,
                payload_sha256,
                semantic_sha256,
                result_kind,
                result_key,
                result_revision,
                result_count,
                now_ms,
            ),
        )

    def _propose(
        self,
        *,
        binding: MemoryToolkitRunBinding,
        request: CandidateProposalRequest,
    ) -> FrozenCandidateSnapshot:
        if not isinstance(request, CandidateProposalRequest):
            raise TypeError("request must be a CandidateProposalRequest")
        if binding.binding_id != self.binding_id:
            raise CurationRepositoryError("candidate_binding_scope_mismatch")
        operation_id = _required_text(
            request.operation_id,
            "operation_id",
            maximum=256,
            identifier=True,
        )
        if request.content is not None and type(request.content) is not bytes:
            raise TypeError("candidate content must be exact bytes")
        kind = str(request.kind or "").casefold()
        content_kinds = {"markdown", "image"}
        if kind in content_kinds:
            valid_payload = (
                type(request.content) is bytes
                and isinstance(request.media_type, str)
                and "/" in request.media_type
                and request.url == ""
            )
        elif kind == "link":
            valid_payload = (
                request.content is None
                and request.media_type == ""
                and isinstance(request.url, str)
                and bool(request.url.strip())
            )
        elif kind == "folder":
            valid_payload = (
                request.content is None
                and request.media_type == ""
                and request.url == ""
            )
        else:
            valid_payload = False
        if not valid_payload:
            raise CurationRepositoryError("candidate_payload_shape_invalid")
        if not isinstance(request.source_refs, tuple) or any(
            not isinstance(item, ResourceRef) or item.kind != "context_event"
            for item in request.source_refs
        ):
            raise CurationRepositoryError("candidate_payload_source_refs_invalid")
        normalized_path = canonical_candidate_path(request.path)
        name = normalized_path.rsplit("/", 1)[-1]
        content = request.content
        content_sha256 = _sha256(content) if content is not None else ""
        byte_length = len(content) if content is not None else 0
        semantic = {
            "binding_id": self.binding_id,
            "session_id": binding.session_id,
            "attempt_id": binding.attempt_id,
            "run_id": binding.run_id,
            "path": normalized_path,
            "description": request.description,
            "kind": request.kind,
            "content_sha256": content_sha256,
            "byte_length": byte_length,
            "media_type": request.media_type,
            "url": request.url,
            "source_refs": [item.to_dict() for item in request.source_refs],
            "rationale": request.rationale,
            "confidence": request.confidence,
            "sensitivity": request.sensitivity,
        }
        semantic_sha256 = _semantic_digest(semantic)
        candidate_id = (
            "candidate-"
            + hashlib.sha256(
                f"{self.binding_id}\0{operation_id}".encode("utf-8")
            ).hexdigest()[:32]
        )
        content_ref = (
            ResourceRef(
                "memory_candidate_content",
                candidate_id,
                1,
                self.target_space_id,
            )
            if content is not None
            else None
        )
        candidate = FrozenCandidateSnapshot(
            candidate_ref=ResourceRef("memory_candidate", candidate_id, 1),
            origin=CandidateOrigin.AGENT_PROPOSAL,
            target_path=normalized_path,
            name=name,
            description=request.description,
            kind=request.kind,
            media_type=request.media_type,
            source_refs=request.source_refs,
            payload_sha256=semantic_sha256,
            content_sha256=content_sha256,
            byte_length=byte_length,
            content_ref=content_ref,
            link_url=request.url,
            source_agent_run_id=binding.run_id,
            rationale=request.rationale,
            confidence=request.confidence,
            sensitivity=request.sensitivity,
        )
        if content is not None:
            installed_sha256, installed_bytes = self._store._install_object(content)
            if (
                installed_sha256 != candidate.content_sha256
                or installed_bytes != candidate.byte_length
            ):
                raise SQLiteCuratorV2IntegrityError("candidate_object_install_changed")
        now_ms = _exact_non_negative_int(self._store._clock_ms(), "clock_ms")
        try:
            with self._store._transaction(immediate=True) as connection:
                self._assert_completion_scope(
                    connection,
                    RootRunCompletion(
                        session_id=binding.session_id,
                        attempt_id=binding.attempt_id,
                        run_id=binding.run_id,
                        is_root_run=True,
                        run_status=SourceRunStatus.COMPLETED,
                        capture_status=RunCaptureStatus.COMPLETE,
                    ),
                )
                replay = self._receipt(
                    connection,
                    operation_id=operation_id,
                    operation_kind="candidate_propose",
                    payload_sha256=semantic_sha256,
                    semantic_sha256=semantic_sha256,
                )
                if replay is not None:
                    result = self._receipt_result(connection, replay)
                    if not isinstance(result, FrozenCandidateSnapshot):
                        raise SQLiteCuratorV2IntegrityError(
                            "candidate_operation_receipt_changed"
                        )
                    return result
                if (
                    connection.execute(
                        "SELECT 1 FROM candidates WHERE candidate_id = ?",
                        (candidate_id,),
                    ).fetchone()
                    is not None
                ):
                    raise CurationConflictError("candidate_identity_conflict")
                if content is not None:
                    object_row = connection.execute(
                        "SELECT byte_length FROM objects WHERE sha256 = ?",
                        (content_sha256,),
                    ).fetchone()
                    if (
                        object_row is not None
                        and object_row["byte_length"] != byte_length
                    ):
                        raise SQLiteCuratorV2IntegrityError(
                            "candidate_object_metadata_changed"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO objects(sha256, byte_length) VALUES (?, ?)",
                        (content_sha256, byte_length),
                    )
                connection.execute(
                    """
                    INSERT INTO candidates(
                        candidate_id,
                        binding_id,
                        session_id,
                        attempt_id,
                        run_id,
                        current_record_revision,
                        status,
                        object_sha256,
                        byte_length,
                        created_at_ms,
                        updated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        self.binding_id,
                        binding.session_id,
                        binding.attempt_id,
                        binding.run_id,
                        candidate.outcome.value,
                        content_sha256 or None,
                        byte_length,
                        now_ms,
                        now_ms,
                    ),
                )
                encoded = _canonical_json_bytes(candidate.to_dict())
                connection.execute(
                    """
                    INSERT INTO candidate_revisions(
                        candidate_id,
                        record_revision,
                        snapshot_json,
                        snapshot_sha256,
                        operation_id,
                        created_at_ms
                    ) VALUES (?, 1, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        encoded,
                        _sha256(encoded),
                        operation_id,
                        now_ms,
                    ),
                )
                self._record_receipt(
                    connection,
                    operation_id=operation_id,
                    operation_kind="candidate_propose",
                    payload_sha256=semantic_sha256,
                    semantic_sha256=semantic_sha256,
                    result_kind="candidate",
                    result_key=candidate_id,
                    result_revision=1,
                    now_ms=now_ms,
                )
                return candidate
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationConflictError("candidate_operation_conflict") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError("candidate_write_failed") from exc

    def read_candidate(
        self,
        *,
        ref: ResourceRef,
    ) -> FrozenCandidateSnapshot:
        if not isinstance(ref, ResourceRef):
            raise TypeError("ref must be a ResourceRef")
        if ref.kind != "memory_candidate" or ref.revision != 1 or ref.fragment:
            raise CurationRepositoryError("candidate_scope_mismatch")
        try:
            with self._store._transaction(immediate=False) as connection:
                self._scope_row(connection)
                return self._current_candidate(
                    connection,
                    candidate_id=ref.resource_id,
                )
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("candidate_read_failed") from exc

    def read_candidate_content(
        self,
        *,
        ref: ResourceRef,
        offset: int,
        limit: int,
    ) -> MemoryToolContentPage:
        if not isinstance(ref, ResourceRef):
            raise TypeError("ref must be a ResourceRef")
        if ref.kind != "memory_candidate" or ref.revision != 1 or ref.fragment:
            raise CurationRepositoryError("candidate_scope_mismatch")
        offset = _exact_non_negative_int(offset, "offset")
        limit = _exact_non_negative_int(limit, "limit")
        try:
            with self._store._transaction(immediate=False) as connection:
                candidate = self._current_candidate(
                    connection,
                    candidate_id=ref.resource_id,
                )
                if candidate.candidate_ref != ref:
                    raise CurationRepositoryError("candidate_scope_mismatch")
                if not candidate.content_sha256 or candidate.content_ref is None:
                    raise CurationRepositoryError("candidate_content_unavailable")
                row = connection.execute(
                    "SELECT byte_length FROM objects WHERE sha256 = ?",
                    (candidate.content_sha256,),
                ).fetchone()
                if row is None or row["byte_length"] != candidate.byte_length:
                    raise SQLiteCuratorV2IntegrityError(
                        "candidate_object_metadata_changed"
                    )
            content = self._store._read_object(
                digest=candidate.content_sha256,
                byte_length=candidate.byte_length,
            )
            return MemoryToolContentPage(
                ref=candidate.candidate_ref,
                media_type=candidate.media_type,
                data=content[offset : offset + limit],
                offset=offset,
                total_bytes=len(content),
                sha256=candidate.content_sha256,
            )
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("candidate_content_read_failed") from exc

    def find_job_by_trigger(self, *, trigger_key: str) -> ConsolidationJob | None:
        normalized = _required_text(
            trigger_key,
            "trigger_key",
            maximum=256,
            identifier=True,
        )
        try:
            with self._store._transaction(immediate=False) as connection:
                self._scope_row(connection)
                row = connection.execute(
                    """
                    SELECT job_id, current_revision
                    FROM consolidation_jobs
                    WHERE binding_id = ? AND trigger_key = ?
                    """,
                    (self.binding_id, normalized),
                ).fetchone()
                if row is None:
                    return None
                return self._job_revision(
                    connection,
                    job_id=row["job_id"],
                    revision=row["current_revision"],
                )
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("consolidation_job_read_failed") from exc

    def list_pending_candidates(
        self,
        *,
        completion: RootRunCompletion,
        limit: int,
    ) -> tuple[FrozenCandidateSnapshot, ...]:
        limit = _exact_positive_int(limit, "limit")
        try:
            with self._store._transaction(immediate=False) as connection:
                self._assert_completion_scope(connection, completion)
                rows = connection.execute(
                    """
                    SELECT
                        candidate.candidate_id,
                        candidate.current_record_revision
                    FROM candidates AS candidate
                    JOIN curation_run_scopes AS source_scope
                      ON source_scope.binding_id = candidate.binding_id
                     AND source_scope.session_id = candidate.session_id
                     AND source_scope.attempt_id = candidate.attempt_id
                     AND source_scope.run_id = candidate.run_id
                    WHERE candidate.binding_id = ?
                      AND candidate.session_id = ?
                      AND source_scope.root_run_id = ?
                      AND candidate.status = ?
                    ORDER BY candidate.candidate_id
                    LIMIT ?
                    """,
                    (
                        self.binding_id,
                        completion.session_id,
                        completion.run_id,
                        CandidateStatus.PENDING.value,
                        limit,
                    ),
                ).fetchall()
                return tuple(
                    self._candidate_revision(
                        connection,
                        candidate_id=row["candidate_id"],
                        record_revision=row["current_record_revision"],
                    )
                    for row in rows
                )
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("candidate_list_failed") from exc

    def _write_candidate_transition(
        self,
        connection: sqlite3.Connection,
        *,
        before: FrozenCandidateSnapshot,
        after: FrozenCandidateSnapshot,
        job_id: str | None,
        operation_id: str,
        now_ms: int,
    ) -> None:
        if before.candidate_ref != after.candidate_ref:
            raise CurationConflictError("candidate_identity_conflict")
        current = self._current_candidate(
            connection,
            candidate_id=before.candidate_ref.resource_id,
        )
        if current != before:
            raise CurationConflictError("candidate_revision_conflict")
        if before == after:
            return
        row = connection.execute(
            """
            SELECT current_record_revision FROM candidates WHERE candidate_id = ?
            """,
            (before.candidate_ref.resource_id,),
        ).fetchone()
        next_record_revision = int(row["current_record_revision"]) + 1
        updated = connection.execute(
            """
            UPDATE candidates
            SET current_record_revision = ?, status = ?, updated_at_ms = ?
            WHERE candidate_id = ? AND current_record_revision = ?
            """,
            (
                next_record_revision,
                after.outcome.value,
                now_ms,
                before.candidate_ref.resource_id,
                next_record_revision - 1,
            ),
        )
        if updated.rowcount != 1:
            raise CurationConflictError("candidate_revision_conflict")
        encoded = _canonical_json_bytes(after.to_dict())
        connection.execute(
            """
            INSERT INTO candidate_revisions(
                candidate_id,
                record_revision,
                snapshot_json,
                snapshot_sha256,
                operation_id,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                after.candidate_ref.resource_id,
                next_record_revision,
                encoded,
                _sha256(encoded),
                operation_id,
                now_ms,
            ),
        )
        if after.is_durable_binding:
            result_ref_json = (
                _canonical_json_bytes(after.result_ref.to_dict())
                if after.result_ref is not None
                else None
            )
            connection.execute(
                """
                INSERT INTO candidate_bindings(
                    candidate_id,
                    binding_revision,
                    job_id,
                    target_space_id,
                    status,
                    result_ref_json,
                    review_diff_json,
                    error_code,
                    candidate_record_revision,
                    created_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    after.candidate_ref.resource_id,
                    after.binding_revision,
                    job_id,
                    after.target_space_id,
                    after.outcome.value,
                    result_ref_json,
                    _canonical_json_bytes(after.to_dict()["review_diff"]),
                    after.error_code,
                    next_record_revision,
                    now_ms,
                ),
            )

    def isolate_source_candidates(
        self,
        *,
        completion: RootRunCompletion,
        reason: str,
        operation: OperationRef,
    ) -> int:
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        normalized_reason = _required_text(
            reason,
            "reason",
            maximum=128,
            identifier=True,
        )
        semantic_sha256 = _semantic_digest(
            {
                "trigger_key": completion.trigger_key,
                "reason": normalized_reason,
            }
        )
        now_ms = _exact_non_negative_int(self._store._clock_ms(), "clock_ms")
        try:
            with self._store._transaction(immediate=True) as connection:
                self._assert_completion_scope(connection, completion)
                replay = self._receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="isolate_source_candidates",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                )
                if replay is not None:
                    result = self._receipt_result(connection, replay)
                    if type(result) is not int:
                        raise SQLiteCuratorV2IntegrityError(
                            "isolation_operation_receipt_changed"
                        )
                    return result
                candidates = self.list_pending_candidates_in_transaction(
                    connection,
                    completion=completion,
                    limit=201,
                )
                for candidate in candidates:
                    isolated = replace(
                        candidate.bind(
                            target_space_id=self.target_space_id,
                            binding_revision=1,
                            outcome=CandidateStatus.ISOLATED,
                            storage_kind=(
                                "file"
                                if candidate.kind in {"markdown", "image"}
                                else candidate.kind
                            ),
                            content_ref=candidate.content_ref,
                        ),
                        error_code=normalized_reason,
                    )
                    self._write_candidate_transition(
                        connection,
                        before=candidate,
                        after=isolated,
                        job_id=None,
                        operation_id=operation.operation_id,
                        now_ms=now_ms,
                    )
                count = len(candidates)
                self._record_receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="isolate_source_candidates",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                    result_kind="count",
                    result_count=count,
                    now_ms=now_ms,
                )
                return count
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationConflictError("isolation_operation_conflict") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError("candidate_isolation_failed") from exc

    def list_pending_candidates_in_transaction(
        self,
        connection: sqlite3.Connection,
        *,
        completion: RootRunCompletion,
        limit: int,
    ) -> tuple[FrozenCandidateSnapshot, ...]:
        rows = connection.execute(
            """
            SELECT
                candidate.candidate_id,
                candidate.current_record_revision
            FROM candidates AS candidate
            JOIN curation_run_scopes AS source_scope
              ON source_scope.binding_id = candidate.binding_id
             AND source_scope.session_id = candidate.session_id
             AND source_scope.attempt_id = candidate.attempt_id
             AND source_scope.run_id = candidate.run_id
            WHERE candidate.binding_id = ?
              AND candidate.session_id = ?
              AND source_scope.root_run_id = ?
              AND candidate.status = ?
            ORDER BY candidate.candidate_id
            LIMIT ?
            """,
            (
                self.binding_id,
                completion.session_id,
                completion.run_id,
                CandidateStatus.PENDING.value,
                limit,
            ),
        ).fetchall()
        return tuple(
            self._candidate_revision(
                connection,
                candidate_id=row["candidate_id"],
                record_revision=row["current_record_revision"],
            )
            for row in rows
        )

    def _insert_job(
        self,
        connection: sqlite3.Connection,
        *,
        job: ConsolidationJob,
        operation_id: str,
        now_ms: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO consolidation_jobs(
                job_id,
                binding_id,
                trigger_key,
                current_revision,
                status,
                lease_owner,
                lease_token,
                lease_expires_at_ms,
                next_attempt_at_ms,
                created_at_ms,
                updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?)
            """,
            (
                job.job_id,
                self.binding_id,
                job.trigger.trigger_key,
                job.revision,
                job.status.value,
                job.next_attempt_at_ms,
                job.created_at_ms,
                job.updated_at_ms,
            ),
        )
        self._insert_job_revision(
            connection,
            job=job,
            operation_id=operation_id,
            now_ms=now_ms,
        )

    @staticmethod
    def _insert_job_revision(
        connection: sqlite3.Connection,
        *,
        job: ConsolidationJob,
        operation_id: str,
        now_ms: int,
    ) -> None:
        encoded = _canonical_json_bytes(_job_to_dict(job))
        connection.execute(
            """
            INSERT INTO consolidation_job_revisions(
                job_id,
                revision,
                job_json,
                job_sha256,
                operation_id,
                created_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                job.job_id,
                job.revision,
                encoded,
                _sha256(encoded),
                operation_id,
                now_ms,
            ),
        )

    def _transition_job(
        self,
        connection: sqlite3.Connection,
        *,
        before: ConsolidationJob,
        after: ConsolidationJob,
        operation_id: str,
        now_ms: int,
    ) -> None:
        current = self._current_job(connection, job_id=before.job_id)
        if current != before or after.revision != before.revision + 1:
            raise CurationConflictError("consolidation_job_revision_conflict")
        lease_owner = after.lease.owner if after.lease is not None else None
        lease_token = after.lease.token if after.lease is not None else None
        lease_expiry = after.lease.expires_at_ms if after.lease is not None else None
        updated = connection.execute(
            """
            UPDATE consolidation_jobs
            SET current_revision = ?,
                status = ?,
                lease_owner = ?,
                lease_token = ?,
                lease_expires_at_ms = ?,
                next_attempt_at_ms = ?,
                updated_at_ms = ?
            WHERE job_id = ? AND binding_id = ? AND current_revision = ?
            """,
            (
                after.revision,
                after.status.value,
                lease_owner,
                lease_token,
                lease_expiry,
                after.next_attempt_at_ms,
                after.updated_at_ms,
                before.job_id,
                self.binding_id,
                before.revision,
            ),
        )
        if updated.rowcount != 1:
            raise CurationConflictError("consolidation_job_revision_conflict")
        self._insert_job_revision(
            connection,
            job=after,
            operation_id=operation_id,
            now_ms=now_ms,
        )

    def enqueue(
        self,
        *,
        request: EnqueueRequest,
        operation: OperationRef,
    ) -> ConsolidationJob:
        if not isinstance(request, EnqueueRequest):
            raise TypeError("request must be an EnqueueRequest")
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        semantic_sha256 = _semantic_digest(
            {
                "trigger": request.trigger.to_dict(),
                "candidates": [item.to_dict() for item in request.candidates],
            }
        )
        now_ms = _exact_non_negative_int(self._store._clock_ms(), "clock_ms")
        try:
            with self._store._transaction(immediate=True) as connection:
                self._assert_completion_scope(connection, request.trigger)
                replay = self._receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="enqueue",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                )
                if replay is not None:
                    result = self._receipt_result(connection, replay)
                    if not isinstance(result, ConsolidationJob):
                        raise SQLiteCuratorV2IntegrityError(
                            "enqueue_operation_receipt_changed"
                        )
                    return result
                if (
                    connection.execute(
                        """
                    SELECT 1 FROM consolidation_jobs
                    WHERE binding_id = ? AND trigger_key = ?
                    """,
                        (self.binding_id, request.trigger.trigger_key),
                    ).fetchone()
                    is not None
                ):
                    raise CurationConflictError("trigger_already_has_job")
                bound_candidates = []
                for requested in request.candidates:
                    current = self._current_candidate(
                        connection,
                        candidate_id=requested.candidate_ref.resource_id,
                    )
                    if (
                        current != requested
                        or current.outcome is not CandidateStatus.PENDING
                    ):
                        raise CurationConflictError("candidate_revision_conflict")
                    bound_candidates.append(
                        current.bind(
                            target_space_id=self.target_space_id,
                            binding_revision=1,
                            outcome=CandidateStatus.QUEUED,
                            storage_kind=(
                                "file"
                                if current.kind in {"markdown", "image"}
                                else current.kind
                            ),
                            content_ref=current.content_ref,
                        )
                    )
                job_id = (
                    "curation-job-"
                    + hashlib.sha256(
                        f"{self.binding_id}\0{request.trigger.trigger_key}".encode(
                            "utf-8"
                        )
                    ).hexdigest()[:32]
                )
                job = ConsolidationJob.pending(
                    job_id=job_id,
                    trigger=request.trigger,
                    candidates=tuple(bound_candidates),
                    operation_id=operation.operation_id,
                    now_ms=now_ms,
                )
                self._insert_job(
                    connection,
                    job=job,
                    operation_id=operation.operation_id,
                    now_ms=now_ms,
                )
                for before, after in zip(request.candidates, bound_candidates):
                    self._write_candidate_transition(
                        connection,
                        before=before,
                        after=after,
                        job_id=job.job_id,
                        operation_id=operation.operation_id,
                        now_ms=now_ms,
                    )
                self._record_receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="enqueue",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                    result_kind="job",
                    result_key=job.job_id,
                    result_revision=job.revision,
                    now_ms=now_ms,
                )
                return job
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationConflictError("enqueue_operation_conflict") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError("consolidation_job_enqueue_failed") from exc

    def read_job(self, *, job_id: str) -> ConsolidationJob:
        normalized = _required_text(
            job_id,
            "job_id",
            maximum=512,
            identifier=True,
        )
        try:
            with self._store._transaction(immediate=False) as connection:
                self._scope_row(connection)
                return self._current_job(connection, job_id=normalized)
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("consolidation_job_read_failed") from exc

    def _assert_fence_in_transaction(
        self,
        connection: sqlite3.Connection,
        fence: CuratorLeaseFence,
        *,
        now_ms: int,
    ) -> ConsolidationJob:
        if not isinstance(fence, CuratorLeaseFence):
            raise TypeError("fence must be a CuratorLeaseFence")
        if fence.binding_id != self.binding_id:
            raise CurationConflictError("lease_fence_scope_mismatch")
        current = self._current_job(connection, job_id=fence.job_id)
        if (
            current.status is not ConsolidationJobStatus.LEASED
            or current.revision != fence.job_revision
            or current.lease is None
            or current.lease.owner != fence.lease_owner
            or current.lease.token != fence.lease_token
        ):
            raise CurationConflictError("lease_fence_lost")
        if current.lease.expires_at_ms <= now_ms:
            raise CurationConflictError("lease_expired")
        return current

    def _assert_fence(self, fence: CuratorLeaseFence, *, now_ms: int) -> None:
        try:
            with self._store._transaction(immediate=False) as connection:
                self._assert_fence_in_transaction(
                    connection,
                    fence,
                    now_ms=now_ms,
                )
        except CurationRepositoryError:
            raise
        except sqlite3.Error as exc:
            raise CurationRepositoryError("lease_fence_read_failed") from exc

    def bind_mutation_guard(
        self,
        *,
        job: ConsolidationJob,
    ) -> BoundCuratorMutationGuard:
        if not isinstance(job, ConsolidationJob):
            raise TypeError("job must be a ConsolidationJob")
        fence = CuratorLeaseFence.from_job(self.binding_id, job)
        self._assert_fence(
            fence,
            now_ms=_exact_non_negative_int(self._store._clock_ms(), "clock_ms"),
        )
        return _SQLiteCuratorMutationGuard(self, fence)

    def claim_next(
        self,
        *,
        worker_id: str,
        now_ms: int,
        lease_ms: int,
        operation: OperationRef,
    ) -> ConsolidationJob | None:
        worker = _required_text(
            worker_id,
            "worker_id",
            maximum=512,
            identifier=True,
        )
        now_ms = _exact_non_negative_int(now_ms, "now_ms")
        lease_ms = _exact_positive_int(lease_ms, "lease_ms")
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        semantic_sha256 = _semantic_digest({"worker_id": worker, "lease_ms": lease_ms})
        try:
            with self._store._transaction(immediate=True) as connection:
                self._scope_row(connection)
                replay = self._receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="claim_next",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                )
                if replay is not None:
                    result = self._receipt_result(connection, replay)
                    if result is not None and not isinstance(result, ConsolidationJob):
                        raise SQLiteCuratorV2IntegrityError(
                            "claim_operation_receipt_changed"
                        )
                    return result
                row = connection.execute(
                    """
                    SELECT job_id, current_revision
                    FROM consolidation_jobs
                    WHERE binding_id = ?
                      AND (
                        (status = ? AND next_attempt_at_ms <= ?)
                        OR
                        (status = ? AND lease_expires_at_ms <= ?)
                      )
                    ORDER BY created_at_ms, job_id
                    LIMIT 1
                    """,
                    (
                        self.binding_id,
                        ConsolidationJobStatus.PENDING.value,
                        now_ms,
                        ConsolidationJobStatus.LEASED.value,
                        now_ms,
                    ),
                ).fetchone()
                if row is None:
                    self._record_receipt(
                        connection,
                        operation_id=operation.operation_id,
                        operation_kind="claim_next",
                        payload_sha256=operation.payload_sha256,
                        semantic_sha256=semantic_sha256,
                        result_kind="none",
                        now_ms=now_ms,
                    )
                    return None
                current = self._job_revision(
                    connection,
                    job_id=row["job_id"],
                    revision=row["current_revision"],
                )
                token = (
                    "lease-"
                    + hashlib.sha256(
                        (
                            f"{self.binding_id}\0{current.job_id}\0{current.revision}"
                            f"\0{operation.operation_id}"
                        ).encode("utf-8")
                    ).hexdigest()[:32]
                )
                claimed = current.with_lease(
                    Lease(
                        owner=worker,
                        token=token,
                        expires_at_ms=now_ms + lease_ms,
                    ),
                    revision=current.revision + 1,
                    now_ms=now_ms,
                )
                if current.status is ConsolidationJobStatus.PENDING:
                    for before, after in zip(current.candidates, claimed.candidates):
                        self._write_candidate_transition(
                            connection,
                            before=before,
                            after=after,
                            job_id=current.job_id,
                            operation_id=operation.operation_id,
                            now_ms=now_ms,
                        )
                self._transition_job(
                    connection,
                    before=current,
                    after=claimed,
                    operation_id=operation.operation_id,
                    now_ms=now_ms,
                )
                self._record_receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="claim_next",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                    result_kind="job",
                    result_key=claimed.job_id,
                    result_revision=claimed.revision,
                    now_ms=now_ms,
                )
                return claimed
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationConflictError("claim_operation_conflict") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError("consolidation_job_claim_failed") from exc

    def _exact_guard(
        self,
        mutation_guard: BoundCuratorMutationGuard,
        job: ConsolidationJob,
    ) -> _SQLiteCuratorMutationGuard:
        if (
            not isinstance(mutation_guard, _SQLiteCuratorMutationGuard)
            or mutation_guard._repository._store.database_path
            != self._store.database_path
            or mutation_guard._repository.binding_id != self.binding_id
        ):
            raise CurationConflictError("lease_mutation_guard_mismatch")
        expected = CuratorLeaseFence.from_job(self.binding_id, job)
        if mutation_guard.fence != expected:
            raise CurationConflictError("lease_mutation_guard_mismatch")
        return mutation_guard

    def reconcile_and_complete(
        self,
        *,
        job: ConsolidationJob,
        resolutions: tuple[CandidateResolution, ...],
        mutation_guard: BoundCuratorMutationGuard,
        operation: OperationRef,
        now_ms: int,
    ) -> ConsolidationJob:
        if not isinstance(job, ConsolidationJob):
            raise TypeError("job must be a ConsolidationJob")
        if not isinstance(resolutions, tuple) or any(
            not isinstance(item, CandidateResolution) for item in resolutions
        ):
            raise TypeError("resolutions must contain CandidateResolution values")
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        now_ms = _exact_non_negative_int(now_ms, "now_ms")
        guard = self._exact_guard(mutation_guard, job)
        semantic_sha256 = _semantic_digest(
            {
                "job_id": job.job_id,
                "job_revision": job.revision,
                "lease_token": job.lease.token if job.lease is not None else "",
                "resolutions": [item.to_dict() for item in resolutions],
            }
        )
        try:
            with self._store._transaction(immediate=True) as connection:
                replay = self._receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="reconcile_and_complete",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                )
                if replay is not None:
                    result = self._receipt_result(connection, replay)
                    if not isinstance(result, ConsolidationJob):
                        raise SQLiteCuratorV2IntegrityError(
                            "reconcile_operation_receipt_changed"
                        )
                    return result
                current = self._assert_fence_in_transaction(
                    connection,
                    guard.fence,
                    now_ms=now_ms,
                )
                if current != job:
                    raise CurationConflictError("lease_fence_lost")
                expected = {
                    item.candidate_ref: item.target_space_id
                    for item in current.candidates
                }
                actual = {
                    item.candidate_ref: item.target_space_id for item in resolutions
                }
                if len(actual) != len(resolutions) or actual != expected:
                    raise CurationConflictError("candidate_resolution_set_mismatch")
                by_ref = {item.candidate_ref: item for item in resolutions}
                reconciled = tuple(
                    candidate.with_outcome(
                        by_ref[candidate.candidate_ref].candidate_status,
                        result_ref=by_ref[candidate.candidate_ref].result_ref,
                        review_diff=by_ref[candidate.candidate_ref].review_diff,
                    )
                    for candidate in current.candidates
                )
                completed = current.terminal(
                    status=ConsolidationJobStatus.COMPLETED,
                    revision=current.revision + 1,
                    now_ms=now_ms,
                    candidates=reconciled,
                )
                for before, after in zip(current.candidates, reconciled):
                    self._write_candidate_transition(
                        connection,
                        before=before,
                        after=after,
                        job_id=current.job_id,
                        operation_id=operation.operation_id,
                        now_ms=now_ms,
                    )
                self._transition_job(
                    connection,
                    before=current,
                    after=completed,
                    operation_id=operation.operation_id,
                    now_ms=now_ms,
                )
                self._record_receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind="reconcile_and_complete",
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                    result_kind="job",
                    result_key=completed.job_id,
                    result_revision=completed.revision,
                    now_ms=now_ms,
                )
                return completed
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationConflictError("reconcile_operation_conflict") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError("consolidation_job_reconcile_failed") from exc

    def fail(
        self,
        *,
        job: ConsolidationJob,
        error_code: str,
        retry_at_ms: int,
        operation: OperationRef,
        now_ms: int,
    ) -> ConsolidationJob:
        if not isinstance(job, ConsolidationJob):
            raise TypeError("job must be a ConsolidationJob")
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        normalized_error = _required_text(
            error_code,
            "error_code",
            maximum=128,
            identifier=True,
        )
        retry_at_ms = _exact_non_negative_int(retry_at_ms, "retry_at_ms")
        now_ms = _exact_non_negative_int(now_ms, "now_ms")
        semantic_sha256 = _semantic_digest(
            {
                "job_id": job.job_id,
                "job_revision": job.revision,
                "lease_token": job.lease.token if job.lease is not None else "",
                "error_code": normalized_error,
                "retry_at_ms": retry_at_ms,
            }
        )
        return self._terminal_or_retry(
            job=job,
            operation=operation,
            operation_kind="fail",
            semantic_sha256=semantic_sha256,
            now_ms=now_ms,
            retry_at_ms=retry_at_ms,
            error_code=normalized_error,
        )

    def cancel(
        self,
        *,
        job: ConsolidationJob,
        reason: str,
        operation: OperationRef,
        now_ms: int,
    ) -> ConsolidationJob:
        if not isinstance(job, ConsolidationJob):
            raise TypeError("job must be a ConsolidationJob")
        if not isinstance(operation, OperationRef):
            raise TypeError("operation must be an OperationRef")
        normalized_reason = _required_text(
            reason,
            "reason",
            maximum=128,
            identifier=True,
        )
        now_ms = _exact_non_negative_int(now_ms, "now_ms")
        semantic_sha256 = _semantic_digest(
            {
                "job_id": job.job_id,
                "job_revision": job.revision,
                "reason": normalized_reason,
            }
        )
        return self._terminal_or_retry(
            job=job,
            operation=operation,
            operation_kind="cancel",
            semantic_sha256=semantic_sha256,
            now_ms=now_ms,
            retry_at_ms=None,
            error_code=normalized_reason,
        )

    def _terminal_or_retry(
        self,
        *,
        job: ConsolidationJob,
        operation: OperationRef,
        operation_kind: str,
        semantic_sha256: str,
        now_ms: int,
        retry_at_ms: int | None,
        error_code: str,
    ) -> ConsolidationJob:
        try:
            with self._store._transaction(immediate=True) as connection:
                replay = self._receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind=operation_kind,
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                )
                if replay is not None:
                    result = self._receipt_result(connection, replay)
                    if not isinstance(result, ConsolidationJob):
                        raise SQLiteCuratorV2IntegrityError(
                            "job_terminal_operation_receipt_changed"
                        )
                    return result
                current = self._current_job(connection, job_id=job.job_id)
                if current != job:
                    raise CurationConflictError("consolidation_job_revision_conflict")
                if operation_kind == "fail":
                    fence = CuratorLeaseFence.from_job(self.binding_id, current)
                    self._assert_fence_in_transaction(
                        connection,
                        fence,
                        now_ms=now_ms,
                    )
                    if retry_at_ms is not None and retry_at_ms > now_ms:
                        updated = current.retry(
                            revision=current.revision + 1,
                            retry_at_ms=retry_at_ms,
                            error_code=error_code,
                            now_ms=now_ms,
                        )
                    else:
                        updated = current.terminal(
                            status=ConsolidationJobStatus.FAILED,
                            revision=current.revision + 1,
                            now_ms=now_ms,
                            error_code=error_code,
                        )
                else:
                    updated = current.terminal(
                        status=ConsolidationJobStatus.CANCELLED,
                        revision=current.revision + 1,
                        now_ms=now_ms,
                        error_code=error_code,
                    )
                for before, after in zip(current.candidates, updated.candidates):
                    self._write_candidate_transition(
                        connection,
                        before=before,
                        after=after,
                        job_id=current.job_id,
                        operation_id=operation.operation_id,
                        now_ms=now_ms,
                    )
                self._transition_job(
                    connection,
                    before=current,
                    after=updated,
                    operation_id=operation.operation_id,
                    now_ms=now_ms,
                )
                self._record_receipt(
                    connection,
                    operation_id=operation.operation_id,
                    operation_kind=operation_kind,
                    payload_sha256=operation.payload_sha256,
                    semantic_sha256=semantic_sha256,
                    result_kind="job",
                    result_key=updated.job_id,
                    result_revision=updated.revision,
                    now_ms=now_ms,
                )
                return updated
        except CurationRepositoryError:
            raise
        except sqlite3.IntegrityError as exc:
            raise CurationConflictError(f"{operation_kind}_operation_conflict") from exc
        except sqlite3.Error as exc:
            raise CurationRepositoryError(
                f"consolidation_job_{operation_kind}_failed"
            ) from exc


__all__ = ["SQLiteCuratorV2IntegrityError", "SQLiteCuratorV2Store"]
