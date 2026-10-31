"""Host-authoritative generation lifecycle persistence for Context V2.

The host supplies every identity and transition explicitly.  This adapter only
persists immutable generation records plus the per-chat current head; it never
infers a generation from a transcript, run mode, or provider payload.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar, Iterator

from unchain.journal import OperationRef
from unchain.journal.models import (
    ModelValidationError,
    _bounded_int,
    _required_text,
)
from unchain.persistence.sqlite_v2 import (
    SQLiteContextV2Store,
    existing_context_v2_readonly_connection,
    serialized_context_v2_database_access,
)


def read_existing_host_generation_binding(*, database_path, owner_chat_id, execution_id, session_id, attempt_id):
    """Read a validated head and attempt atomically without initializing storage.

    This can also run inside a journal write precondition: it acquires no
    second write lock and the outer transaction prevents a concurrent rebase.
    """
    owner = _required_text(owner_chat_id, "owner_chat_id", identifier=True)
    execution = _required_text(execution_id, "execution_id", identifier=True)
    session = _required_text(session_id, "session_id", identifier=True)
    attempt = _required_text(attempt_id, "attempt_id", identifier=True)
    try:
        with existing_context_v2_readonly_connection(database_path) as connection:
            head_row = connection.execute("SELECT * FROM host_generation_heads WHERE owner_chat_id=?", (owner,)).fetchone()
            binding_row = connection.execute("SELECT * FROM host_generation_attempt_bindings WHERE owner_chat_id=? AND execution_id=? AND attempt_id=?",
                (owner, execution, attempt)).fetchone()
            head = SQLiteHostGenerationLifecycleV2._head_from_row(head_row) if head_row is not None else None
            binding = SQLiteHostGenerationLifecycleV2._attempt_binding_from_row(binding_row) if binding_row is not None else None
            if head is not None and (head.execution_id != execution or head.session_id != session):
                raise HostGenerationConflict("host generation read binding does not match durable scope")
            if binding is not None and binding.session_id != session:
                raise HostGenerationConflict("host generation attempt read binding does not match durable scope")
            return head, binding
    except sqlite3.Error as exc:
        raise HostGenerationUnavailable("existing host generation read failed") from exc


class HostGenerationLifecycleError(RuntimeError):
    """Base failure at the durable host-generation boundary."""


class HostGenerationConflict(HostGenerationLifecycleError):
    """A host identity, operation, or compare-and-swap precondition drifted."""


class HostGenerationUnavailable(HostGenerationLifecycleError):
    """The durable lifecycle store could not complete an operation."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ModelValidationError(
            "host generation value is not canonical JSON"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class HostGenerationTransitionKind(StrEnum):
    """Explicit host-selected generation transition."""

    INITIAL = "initial"
    EDIT = "edit"
    REGENERATE = "regenerate"


@dataclass(frozen=True)
class HostGenerationTransition:
    """One exact host request to create and select a generation."""

    owner_chat_id: str
    execution_id: str
    session_id: str
    generation_id: str
    kind: HostGenerationTransitionKind
    previous_generation_id: str
    expected_revision: int

    SCHEMA: ClassVar[str] = "unchain.host_generation_transition.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "owner_chat_id",
            "execution_id",
            "session_id",
            "generation_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        try:
            kind = HostGenerationTransitionKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ModelValidationError(
                "host generation transition kind is invalid"
            ) from exc
        object.__setattr__(self, "kind", kind)
        revision = _bounded_int(self.expected_revision, "expected_revision")
        object.__setattr__(self, "expected_revision", revision)
        if kind is HostGenerationTransitionKind.INITIAL:
            if self.previous_generation_id not in (None, ""):
                raise ModelValidationError(
                    "initial host generation cannot name a previous generation"
                )
            if revision != 0:
                raise ModelValidationError(
                    "initial host generation requires expected_revision zero"
                )
            object.__setattr__(self, "previous_generation_id", "")
            return
        previous = _required_text(
            self.previous_generation_id,
            "previous_generation_id",
            identifier=True,
        )
        if previous == self.generation_id:
            raise ModelValidationError(
                "host generation transition must create a new generation"
            )
        if revision < 1:
            raise ModelValidationError(
                "host generation rebase requires a positive expected_revision"
            )
        object.__setattr__(self, "previous_generation_id", previous)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "owner_chat_id": self.owner_chat_id,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "generation_id": self.generation_id,
            "kind": self.kind.value,
            "previous_generation_id": self.previous_generation_id,
            "expected_revision": self.expected_revision,
        }


def build_host_generation_transition_operation(
    *,
    operation_id: str,
    transition: HostGenerationTransition,
) -> OperationRef:
    """Bind an operation id to the complete canonical transition payload."""

    if not isinstance(transition, HostGenerationTransition):
        raise TypeError("transition must be a HostGenerationTransition")
    return OperationRef(operation_id, _sha256(transition.to_dict()))


@dataclass(frozen=True)
class HostGenerationTransitionRequest:
    transition: HostGenerationTransition
    operation: OperationRef

    def __post_init__(self) -> None:
        if not isinstance(self.transition, HostGenerationTransition):
            raise TypeError("transition must be a HostGenerationTransition")
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(
                self,
                "operation",
                OperationRef.from_dict(self.operation),
            )


@dataclass(frozen=True)
class HostGenerationRecord:
    owner_chat_id: str
    execution_id: str
    session_id: str
    generation_id: str
    kind: HostGenerationTransitionKind
    previous_generation_id: str
    revision: int
    operation: OperationRef


@dataclass(frozen=True)
class HostGenerationHead:
    owner_chat_id: str
    execution_id: str
    session_id: str
    current_generation_id: str
    revision: int


@dataclass(frozen=True)
class HostGenerationTransitionReceipt:
    record: HostGenerationRecord
    head: HostGenerationHead
    duplicate: bool = False


@dataclass(frozen=True)
class HostGenerationAttemptBindingIntent:
    """Explicit host intent to bind one attempt to the current generation."""

    owner_chat_id: str
    execution_id: str
    session_id: str
    generation_id: str
    attempt_id: str
    expected_revision: int

    SCHEMA: ClassVar[str] = "unchain.host_generation_attempt_binding_intent.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "owner_chat_id",
            "execution_id",
            "session_id",
            "generation_id",
            "attempt_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name, identifier=True),
            )
        object.__setattr__(
            self,
            "expected_revision",
            _bounded_int(
                self.expected_revision,
                "expected_revision",
                minimum=1,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "owner_chat_id": self.owner_chat_id,
            "execution_id": self.execution_id,
            "session_id": self.session_id,
            "generation_id": self.generation_id,
            "attempt_id": self.attempt_id,
            "expected_revision": self.expected_revision,
        }


def build_host_generation_attempt_binding_operation(
    *,
    operation_id: str,
    intent: HostGenerationAttemptBindingIntent,
) -> OperationRef:
    """Bind an operation id to the complete canonical attempt intent."""

    if not isinstance(intent, HostGenerationAttemptBindingIntent):
        raise TypeError("intent must be a HostGenerationAttemptBindingIntent")
    return OperationRef(operation_id, _sha256(intent.to_dict()))


@dataclass(frozen=True)
class HostGenerationAttemptBindingRequest:
    intent: HostGenerationAttemptBindingIntent
    operation: OperationRef

    def __post_init__(self) -> None:
        if not isinstance(self.intent, HostGenerationAttemptBindingIntent):
            raise TypeError("intent must be a HostGenerationAttemptBindingIntent")
        if not isinstance(self.operation, OperationRef):
            object.__setattr__(
                self,
                "operation",
                OperationRef.from_dict(self.operation),
            )


@dataclass(frozen=True)
class HostGenerationAttemptBinding:
    owner_chat_id: str
    execution_id: str
    session_id: str
    generation_id: str
    attempt_id: str
    head_revision: int
    operation: OperationRef


@dataclass(frozen=True)
class HostGenerationAttemptBindingReceipt:
    binding: HostGenerationAttemptBinding
    duplicate: bool = False


class SQLiteHostGenerationLifecycleV2:
    """Persist exact host-selected generations in one SQLite/WAL data plane."""

    _SCHEMA_VERSION: ClassVar[int] = 1

    def __init__(self, store: SQLiteContextV2Store) -> None:
        if not isinstance(store, SQLiteContextV2Store):
            raise TypeError("store must be a SQLiteContextV2Store")
        self._store = store
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._store.database_path,
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
        with serialized_context_v2_database_access(self._store.database_path):
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
        with serialized_context_v2_database_access(self._store.database_path):
            try:
                connection = self._connect()
                try:
                    mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
                    if str(mode).casefold() != "wal":
                        raise HostGenerationUnavailable(
                            "host generation SQLite WAL mode is unavailable"
                        )
                finally:
                    connection.close()
                with self._transaction(immediate=True) as connection:
                    connection.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS host_generation_schema (
                            version INTEGER PRIMARY KEY,
                            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                        );
                        INSERT OR IGNORE INTO host_generation_schema(version) VALUES (1);

                        CREATE TABLE IF NOT EXISTS host_generation_chat_bindings (
                            owner_chat_id TEXT PRIMARY KEY,
                            execution_id TEXT NOT NULL UNIQUE,
                            session_id TEXT NOT NULL,
                            UNIQUE(owner_chat_id, execution_id, session_id),
                            FOREIGN KEY (execution_id)
                                REFERENCES executions(execution_id)
                        );

                        CREATE TABLE IF NOT EXISTS host_generation_records (
                            owner_chat_id TEXT NOT NULL,
                            execution_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            generation_id TEXT NOT NULL,
                            transition_kind TEXT NOT NULL CHECK(
                                transition_kind IN ('initial', 'edit', 'regenerate')
                            ),
                            previous_generation_id TEXT NOT NULL,
                            revision INTEGER NOT NULL CHECK(revision >= 1),
                            operation_id TEXT NOT NULL,
                            payload_sha256 TEXT NOT NULL,
                            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (owner_chat_id, generation_id),
                            UNIQUE(execution_id, generation_id),
                            UNIQUE(owner_chat_id, revision),
                            FOREIGN KEY (owner_chat_id, execution_id, session_id)
                                REFERENCES host_generation_chat_bindings(
                                    owner_chat_id,
                                    execution_id,
                                    session_id
                                )
                        );

                        CREATE TABLE IF NOT EXISTS host_generation_heads (
                            owner_chat_id TEXT PRIMARY KEY,
                            execution_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            current_generation_id TEXT NOT NULL,
                            revision INTEGER NOT NULL CHECK(revision >= 1),
                            FOREIGN KEY (owner_chat_id, execution_id, session_id)
                                REFERENCES host_generation_chat_bindings(
                                    owner_chat_id,
                                    execution_id,
                                    session_id
                                ),
                            FOREIGN KEY (owner_chat_id, current_generation_id)
                                REFERENCES host_generation_records(
                                    owner_chat_id,
                                    generation_id
                                )
                        );

                        CREATE TABLE IF NOT EXISTS host_generation_operations (
                            owner_chat_id TEXT NOT NULL,
                            operation_id TEXT NOT NULL,
                            payload_sha256 TEXT NOT NULL,
                            mutation_kind TEXT NOT NULL,
                            result_generation_id TEXT NOT NULL,
                            result_revision INTEGER NOT NULL CHECK(result_revision >= 1),
                            result_attempt_id TEXT NOT NULL DEFAULT '',
                            PRIMARY KEY (owner_chat_id, operation_id),
                            FOREIGN KEY (owner_chat_id, result_generation_id)
                                REFERENCES host_generation_records(
                                    owner_chat_id,
                                    generation_id
                                )
                        );

                        CREATE TABLE IF NOT EXISTS host_generation_attempt_bindings (
                            owner_chat_id TEXT NOT NULL,
                            execution_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            generation_id TEXT NOT NULL,
                            attempt_id TEXT NOT NULL,
                            head_revision INTEGER NOT NULL CHECK(head_revision >= 1),
                            operation_id TEXT NOT NULL,
                            payload_sha256 TEXT NOT NULL,
                            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (execution_id, attempt_id),
                            UNIQUE(owner_chat_id, operation_id),
                            FOREIGN KEY (owner_chat_id, execution_id, session_id)
                                REFERENCES host_generation_chat_bindings(
                                    owner_chat_id,
                                    execution_id,
                                    session_id
                                ),
                            FOREIGN KEY (owner_chat_id, generation_id)
                                REFERENCES host_generation_records(
                                    owner_chat_id,
                                    generation_id
                                ),
                            FOREIGN KEY (owner_chat_id, operation_id)
                                REFERENCES host_generation_operations(
                                    owner_chat_id,
                                    operation_id
                                )
                        );
                        """
                    )
                    versions = {
                        int(row[0])
                        for row in connection.execute(
                            "SELECT version FROM host_generation_schema"
                        )
                    }
                    if versions != {self._SCHEMA_VERSION}:
                        raise HostGenerationUnavailable(
                            "host generation SQLite schema is unsupported"
                        )
            except HostGenerationLifecycleError:
                raise
            except sqlite3.Error as exc:
                raise HostGenerationUnavailable(
                    "host generation SQLite schema initialization failed"
                ) from exc

    @staticmethod
    def _head_from_row(row: sqlite3.Row) -> HostGenerationHead:
        return HostGenerationHead(
            owner_chat_id=row["owner_chat_id"],
            execution_id=row["execution_id"],
            session_id=row["session_id"],
            current_generation_id=row["current_generation_id"],
            revision=int(row["revision"]),
        )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> HostGenerationRecord:
        return HostGenerationRecord(
            owner_chat_id=row["owner_chat_id"],
            execution_id=row["execution_id"],
            session_id=row["session_id"],
            generation_id=row["generation_id"],
            kind=HostGenerationTransitionKind(row["transition_kind"]),
            previous_generation_id=row["previous_generation_id"],
            revision=int(row["revision"]),
            operation=OperationRef(row["operation_id"], row["payload_sha256"]),
        )

    def advance(
        self,
        request: HostGenerationTransitionRequest,
    ) -> HostGenerationTransitionReceipt:
        """Create exactly one explicit generation and advance the chat head."""

        if not isinstance(request, HostGenerationTransitionRequest):
            raise TypeError("request must be a HostGenerationTransitionRequest")
        transition = request.transition
        expected_operation = build_host_generation_transition_operation(
            operation_id=request.operation.operation_id,
            transition=transition,
        )
        if expected_operation != request.operation:
            raise HostGenerationConflict(
                "host generation operation payload hash changed"
            )
        try:
            with self._transaction(immediate=True) as connection:
                existing = connection.execute(
                    """
                    SELECT payload_sha256, mutation_kind,
                           result_generation_id, result_revision
                    FROM host_generation_operations
                    WHERE owner_chat_id = ? AND operation_id = ?
                    """,
                    (transition.owner_chat_id, request.operation.operation_id),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["payload_sha256"] != request.operation.payload_sha256
                        or existing["mutation_kind"] != "transition"
                    ):
                        raise HostGenerationConflict(
                            "host generation operation id was reused with another payload"
                        )
                    record = connection.execute(
                        """
                        SELECT * FROM host_generation_records
                        WHERE owner_chat_id = ? AND generation_id = ?
                        """,
                        (
                            transition.owner_chat_id,
                            existing["result_generation_id"],
                        ),
                    ).fetchone()
                    if record is None:
                        raise HostGenerationUnavailable(
                            "host generation operation receipt is incomplete"
                        )
                    parsed_record = self._record_from_row(record)
                    return HostGenerationTransitionReceipt(
                        parsed_record,
                        HostGenerationHead(
                            owner_chat_id=parsed_record.owner_chat_id,
                            execution_id=parsed_record.execution_id,
                            session_id=parsed_record.session_id,
                            current_generation_id=parsed_record.generation_id,
                            revision=int(existing["result_revision"]),
                        ),
                        duplicate=True,
                    )

                head = connection.execute(
                    "SELECT * FROM host_generation_heads WHERE owner_chat_id = ?",
                    (transition.owner_chat_id,),
                ).fetchone()
                next_revision = transition.expected_revision + 1
                if transition.kind is HostGenerationTransitionKind.INITIAL:
                    if head is not None:
                        raise HostGenerationConflict(
                            "host generation chat already has a current generation"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO executions(execution_id) VALUES (?)",
                        (transition.execution_id,),
                    )
                    connection.execute(
                        """
                        INSERT INTO host_generation_chat_bindings(
                            owner_chat_id, execution_id, session_id
                        ) VALUES (?, ?, ?)
                        """,
                        (
                            transition.owner_chat_id,
                            transition.execution_id,
                            transition.session_id,
                        ),
                    )
                else:
                    if head is None:
                        raise HostGenerationConflict(
                            "host generation rebase has no current generation"
                        )
                    if (
                        head["execution_id"] != transition.execution_id
                        or head["session_id"] != transition.session_id
                    ):
                        raise HostGenerationConflict(
                            "host generation rebase binding does not match durable scope"
                        )
                    if (
                        head["current_generation_id"]
                        != transition.previous_generation_id
                    ):
                        raise HostGenerationConflict(
                            "host generation previous generation is not current"
                        )
                    if int(head["revision"]) != transition.expected_revision:
                        raise HostGenerationConflict(
                            "host generation expected revision is not current"
                        )
                connection.execute(
                    """
                    INSERT INTO host_generation_records(
                        owner_chat_id, execution_id, session_id, generation_id,
                        transition_kind, previous_generation_id, revision,
                        operation_id, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transition.owner_chat_id,
                        transition.execution_id,
                        transition.session_id,
                        transition.generation_id,
                        transition.kind.value,
                        transition.previous_generation_id,
                        next_revision,
                        request.operation.operation_id,
                        request.operation.payload_sha256,
                    ),
                )
                if transition.kind is HostGenerationTransitionKind.INITIAL:
                    connection.execute(
                        """
                        INSERT INTO host_generation_heads(
                            owner_chat_id, execution_id, session_id,
                            current_generation_id, revision
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            transition.owner_chat_id,
                            transition.execution_id,
                            transition.session_id,
                            transition.generation_id,
                            next_revision,
                        ),
                    )
                else:
                    updated = connection.execute(
                        """
                        UPDATE host_generation_heads
                        SET current_generation_id = ?, revision = ?
                        WHERE owner_chat_id = ?
                          AND execution_id = ?
                          AND session_id = ?
                          AND current_generation_id = ?
                          AND revision = ?
                        """,
                        (
                            transition.generation_id,
                            next_revision,
                            transition.owner_chat_id,
                            transition.execution_id,
                            transition.session_id,
                            transition.previous_generation_id,
                            transition.expected_revision,
                        ),
                    )
                    if updated.rowcount != 1:
                        raise HostGenerationConflict(
                            "host generation head changed during compare-and-swap"
                        )
                connection.execute(
                    """
                    INSERT INTO host_generation_operations(
                        owner_chat_id, operation_id, payload_sha256,
                        mutation_kind, result_generation_id, result_revision
                    ) VALUES (?, ?, ?, 'transition', ?, ?)
                    """,
                    (
                        transition.owner_chat_id,
                        request.operation.operation_id,
                        request.operation.payload_sha256,
                        transition.generation_id,
                        next_revision,
                    ),
                )
                record = connection.execute(
                    """
                    SELECT * FROM host_generation_records
                    WHERE owner_chat_id = ? AND generation_id = ?
                    """,
                    (transition.owner_chat_id, transition.generation_id),
                ).fetchone()
                current = connection.execute(
                    "SELECT * FROM host_generation_heads WHERE owner_chat_id = ?",
                    (transition.owner_chat_id,),
                ).fetchone()
                if record is None or current is None:
                    raise HostGenerationUnavailable(
                        "host generation initial write did not persist"
                    )
                return HostGenerationTransitionReceipt(
                    self._record_from_row(record),
                    self._head_from_row(current),
                )
        except HostGenerationLifecycleError:
            raise
        except sqlite3.IntegrityError as exc:
            raise HostGenerationConflict(
                "host generation initial identity conflicts with durable state"
            ) from exc
        except sqlite3.Error as exc:
            raise HostGenerationUnavailable(
                "host generation SQLite transition failed"
            ) from exc

    def current(
        self,
        *,
        owner_chat_id: str,
        execution_id: str,
        session_id: str,
    ) -> HostGenerationHead | None:
        """Read the current head only through its exact host binding."""

        owner = _required_text(owner_chat_id, "owner_chat_id", identifier=True)
        execution = _required_text(execution_id, "execution_id", identifier=True)
        session = _required_text(session_id, "session_id", identifier=True)
        try:
            with self._transaction(immediate=False) as connection:
                row = connection.execute(
                    "SELECT * FROM host_generation_heads WHERE owner_chat_id = ?",
                    (owner,),
                ).fetchone()
                if row is None:
                    return None
                if row["execution_id"] != execution or row["session_id"] != session:
                    raise HostGenerationConflict(
                        "host generation read binding does not match durable scope"
                    )
                return self._head_from_row(row)
        except HostGenerationLifecycleError:
            raise
        except sqlite3.Error as exc:
            raise HostGenerationUnavailable(
                "host generation SQLite current-head read failed"
            ) from exc

    def generation(
        self,
        *,
        owner_chat_id: str,
        execution_id: str,
        session_id: str,
        generation_id: str,
    ) -> HostGenerationRecord | None:
        """Read one immutable record only through its exact host binding."""

        owner = _required_text(owner_chat_id, "owner_chat_id", identifier=True)
        execution = _required_text(execution_id, "execution_id", identifier=True)
        session = _required_text(session_id, "session_id", identifier=True)
        generation = _required_text(
            generation_id,
            "generation_id",
            identifier=True,
        )
        try:
            with self._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM host_generation_records
                    WHERE owner_chat_id = ? AND generation_id = ?
                    """,
                    (owner, generation),
                ).fetchone()
                if row is None:
                    return None
                if row["execution_id"] != execution or row["session_id"] != session:
                    raise HostGenerationConflict(
                        "host generation record binding does not match durable scope"
                    )
                return self._record_from_row(row)
        except HostGenerationLifecycleError:
            raise
        except sqlite3.Error as exc:
            raise HostGenerationUnavailable(
                "host generation SQLite record read failed"
            ) from exc

    @staticmethod
    def _attempt_binding_from_row(
        row: sqlite3.Row,
    ) -> HostGenerationAttemptBinding:
        return HostGenerationAttemptBinding(
            owner_chat_id=row["owner_chat_id"],
            execution_id=row["execution_id"],
            session_id=row["session_id"],
            generation_id=row["generation_id"],
            attempt_id=row["attempt_id"],
            head_revision=int(row["head_revision"]),
            operation=OperationRef(row["operation_id"], row["payload_sha256"]),
        )

    def bind_current_attempt(
        self,
        request: HostGenerationAttemptBindingRequest,
    ) -> HostGenerationAttemptBindingReceipt:
        """Bind an attempt only when its explicit generation is still current."""

        if not isinstance(request, HostGenerationAttemptBindingRequest):
            raise TypeError("request must be a HostGenerationAttemptBindingRequest")
        intent = request.intent
        expected_operation = build_host_generation_attempt_binding_operation(
            operation_id=request.operation.operation_id,
            intent=intent,
        )
        if expected_operation != request.operation:
            raise HostGenerationConflict(
                "host generation attempt operation payload hash changed"
            )
        try:
            with self._transaction(immediate=True) as connection:
                existing_operation = connection.execute(
                    """
                    SELECT payload_sha256, mutation_kind, result_attempt_id
                    FROM host_generation_operations
                    WHERE owner_chat_id = ? AND operation_id = ?
                    """,
                    (intent.owner_chat_id, request.operation.operation_id),
                ).fetchone()
                if existing_operation is not None:
                    if (
                        existing_operation["payload_sha256"]
                        != request.operation.payload_sha256
                        or existing_operation["mutation_kind"] != "attempt_binding"
                    ):
                        raise HostGenerationConflict(
                            "host generation operation id was reused with another payload"
                        )
                    row = connection.execute(
                        """
                        SELECT * FROM host_generation_attempt_bindings
                        WHERE owner_chat_id = ? AND execution_id = ?
                          AND attempt_id = ?
                        """,
                        (
                            intent.owner_chat_id,
                            intent.execution_id,
                            existing_operation["result_attempt_id"],
                        ),
                    ).fetchone()
                    if row is None:
                        raise HostGenerationUnavailable(
                            "host generation attempt receipt is incomplete"
                        )
                    return HostGenerationAttemptBindingReceipt(
                        self._attempt_binding_from_row(row),
                        duplicate=True,
                    )

                head = connection.execute(
                    "SELECT * FROM host_generation_heads WHERE owner_chat_id = ?",
                    (intent.owner_chat_id,),
                ).fetchone()
                if head is None:
                    raise HostGenerationConflict(
                        "host generation attempt has no current generation"
                    )
                if (
                    head["execution_id"] != intent.execution_id
                    or head["session_id"] != intent.session_id
                ):
                    raise HostGenerationConflict(
                        "host generation attempt binding does not match durable scope"
                    )
                if head["current_generation_id"] != intent.generation_id:
                    raise HostGenerationConflict(
                        "host generation attempt does not name the current generation"
                    )
                if int(head["revision"]) != intent.expected_revision:
                    raise HostGenerationConflict(
                        "host generation attempt expected revision is not current"
                    )
                existing_attempt = connection.execute(
                    """
                    SELECT 1 FROM host_generation_attempt_bindings
                    WHERE execution_id = ? AND attempt_id = ?
                    """,
                    (intent.execution_id, intent.attempt_id),
                ).fetchone()
                if existing_attempt is not None:
                    raise HostGenerationConflict(
                        "host generation attempt is already bound"
                    )
                connection.execute(
                    """
                    INSERT INTO host_generation_operations(
                        owner_chat_id, operation_id, payload_sha256,
                        mutation_kind, result_generation_id, result_revision,
                        result_attempt_id
                    ) VALUES (?, ?, ?, 'attempt_binding', ?, ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        request.operation.operation_id,
                        request.operation.payload_sha256,
                        intent.generation_id,
                        intent.expected_revision,
                        intent.attempt_id,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO host_generation_attempt_bindings(
                        owner_chat_id, execution_id, session_id, generation_id,
                        attempt_id, head_revision, operation_id, payload_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        intent.owner_chat_id,
                        intent.execution_id,
                        intent.session_id,
                        intent.generation_id,
                        intent.attempt_id,
                        intent.expected_revision,
                        request.operation.operation_id,
                        request.operation.payload_sha256,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT * FROM host_generation_attempt_bindings
                    WHERE execution_id = ? AND attempt_id = ?
                    """,
                    (intent.execution_id, intent.attempt_id),
                ).fetchone()
                if row is None:
                    raise HostGenerationUnavailable(
                        "host generation attempt binding did not persist"
                    )
                return HostGenerationAttemptBindingReceipt(
                    self._attempt_binding_from_row(row)
                )
        except HostGenerationLifecycleError:
            raise
        except sqlite3.IntegrityError as exc:
            raise HostGenerationConflict(
                "host generation attempt binding conflicts with durable state"
            ) from exc
        except sqlite3.Error as exc:
            raise HostGenerationUnavailable(
                "host generation SQLite attempt binding failed"
            ) from exc

    def attempt_binding(
        self,
        *,
        owner_chat_id: str,
        execution_id: str,
        session_id: str,
        attempt_id: str,
    ) -> HostGenerationAttemptBinding | None:
        """Read one immutable attempt binding through its exact host scope."""

        owner = _required_text(owner_chat_id, "owner_chat_id", identifier=True)
        execution = _required_text(execution_id, "execution_id", identifier=True)
        session = _required_text(session_id, "session_id", identifier=True)
        attempt = _required_text(attempt_id, "attempt_id", identifier=True)
        try:
            with self._transaction(immediate=False) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM host_generation_attempt_bindings
                    WHERE owner_chat_id = ? AND execution_id = ?
                      AND attempt_id = ?
                    """,
                    (owner, execution, attempt),
                ).fetchone()
                if row is None:
                    return None
                if row["session_id"] != session:
                    raise HostGenerationConflict(
                        "host generation attempt read binding does not match durable scope"
                    )
                return self._attempt_binding_from_row(row)
        except HostGenerationLifecycleError:
            raise
        except sqlite3.Error as exc:
            raise HostGenerationUnavailable(
                "host generation SQLite attempt read failed"
            ) from exc
