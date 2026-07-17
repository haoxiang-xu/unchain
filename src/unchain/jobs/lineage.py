from __future__ import annotations

import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .models import (
    DurableJobConflictError,
    DurableJobSnapshot,
    DurableJobStoreCorruptionError,
)
from .store import JsonFileJobStore, _exclusive_lock


JOB_LINEAGE_SCHEMA_VERSION = 1
JOB_LINEAGE_CANCELLATION_SCHEMA_VERSION = 1
_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{32}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if required and not normalized:
        raise ValueError(f"{field} must be a non-empty string")
    if "\0" in normalized:
        raise ValueError(f"{field} must not contain NUL")
    return normalized


@dataclass(frozen=True, slots=True)
class DurableJobLineage:
    store_id: str
    job_id: str
    execution_id: str
    logical_execution_id: str
    created_by_attempt_id: str
    request_digest: str
    lineage_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOB_LINEAGE_SCHEMA_VERSION,
            "store_id": self.store_id,
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "logical_execution_id": self.logical_execution_id,
            "created_by_attempt_id": self.created_by_attempt_id,
            "request_digest": self.request_digest,
            "lineage_digest": self.lineage_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        store_id: str,
        job_id: str,
        execution_id: str,
        logical_execution_id: str,
        created_by_attempt_id: str = "",
        request_digest: str,
    ) -> "DurableJobLineage":
        core = {
            "schema_version": JOB_LINEAGE_SCHEMA_VERSION,
            "store_id": _text(store_id, "store_id"),
            "job_id": _text(job_id, "job_id"),
            "execution_id": _text(execution_id, "execution_id"),
            "logical_execution_id": _text(
                logical_execution_id,
                "logical_execution_id",
            ),
            "created_by_attempt_id": _text(
                created_by_attempt_id,
                "created_by_attempt_id",
                required=False,
            ),
            "request_digest": _text(request_digest, "request_digest"),
        }
        if not _JOB_ID_RE.fullmatch(core["job_id"]):
            raise ValueError("job_id is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", core["request_digest"]):
            raise ValueError("request_digest is invalid")
        return cls(
            store_id=core["store_id"],
            job_id=core["job_id"],
            execution_id=core["execution_id"],
            logical_execution_id=core["logical_execution_id"],
            created_by_attempt_id=core["created_by_attempt_id"],
            request_digest=core["request_digest"],
            lineage_digest=_digest(core),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "DurableJobLineage":
        if not isinstance(raw, dict):
            raise DurableJobStoreCorruptionError(
                "durable job lineage must be a JSON object"
            )
        expected = {
            "schema_version",
            "store_id",
            "job_id",
            "execution_id",
            "logical_execution_id",
            "created_by_attempt_id",
            "request_digest",
            "lineage_digest",
        }
        if (
            set(raw) != expected
            or raw.get("schema_version") != JOB_LINEAGE_SCHEMA_VERSION
        ):
            raise DurableJobStoreCorruptionError(
                "durable job lineage fields or schema version are invalid"
            )
        try:
            lineage = cls.build(
                store_id=raw["store_id"],
                job_id=raw["job_id"],
                execution_id=raw["execution_id"],
                logical_execution_id=raw["logical_execution_id"],
                created_by_attempt_id=raw["created_by_attempt_id"],
                request_digest=raw["request_digest"],
            )
        except (TypeError, ValueError) as exc:
            raise DurableJobStoreCorruptionError(
                "durable job lineage identity is invalid"
            ) from exc
        if raw.get("lineage_digest") != lineage.lineage_digest:
            raise DurableJobStoreCorruptionError(
                "durable job lineage digest does not match"
            )
        return lineage


@dataclass(frozen=True, slots=True)
class DurableJobLineageCancellation:
    store_id: str
    execution_id: str
    logical_execution_id: str
    requested_at_ms: int
    reason: str
    cancellation_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOB_LINEAGE_CANCELLATION_SCHEMA_VERSION,
            "store_id": self.store_id,
            "execution_id": self.execution_id,
            "logical_execution_id": self.logical_execution_id,
            "requested_at_ms": self.requested_at_ms,
            "reason": self.reason,
            "cancellation_digest": self.cancellation_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        store_id: str,
        execution_id: str,
        logical_execution_id: str,
        requested_at_ms: int,
        reason: str = "user_stop",
    ) -> "DurableJobLineageCancellation":
        if (
            isinstance(requested_at_ms, bool)
            or not isinstance(requested_at_ms, int)
            or requested_at_ms < 0
        ):
            raise ValueError("requested_at_ms must be a non-negative integer")
        normalized_reason = _text(reason, "reason", required=False) or "user_stop"
        core = {
            "schema_version": JOB_LINEAGE_CANCELLATION_SCHEMA_VERSION,
            "store_id": _text(store_id, "store_id"),
            "execution_id": _text(execution_id, "execution_id"),
            "logical_execution_id": _text(
                logical_execution_id,
                "logical_execution_id",
            ),
            "requested_at_ms": requested_at_ms,
            "reason": normalized_reason,
        }
        return cls(
            store_id=core["store_id"],
            execution_id=core["execution_id"],
            logical_execution_id=core["logical_execution_id"],
            requested_at_ms=core["requested_at_ms"],
            reason=core["reason"],
            cancellation_digest=_digest(core),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "DurableJobLineageCancellation":
        if not isinstance(raw, dict):
            raise DurableJobStoreCorruptionError(
                "durable job lineage cancellation must be a JSON object"
            )
        expected = {
            "schema_version",
            "store_id",
            "execution_id",
            "logical_execution_id",
            "requested_at_ms",
            "reason",
            "cancellation_digest",
        }
        if (
            set(raw) != expected
            or raw.get("schema_version")
            != JOB_LINEAGE_CANCELLATION_SCHEMA_VERSION
        ):
            raise DurableJobStoreCorruptionError(
                "durable job lineage cancellation fields are invalid"
            )
        try:
            cancellation = cls.build(
                store_id=raw["store_id"],
                execution_id=raw["execution_id"],
                logical_execution_id=raw["logical_execution_id"],
                requested_at_ms=raw["requested_at_ms"],
                reason=raw["reason"],
            )
        except (TypeError, ValueError) as exc:
            raise DurableJobStoreCorruptionError(
                "durable job lineage cancellation identity is invalid"
            ) from exc
        if raw.get("cancellation_digest") != cancellation.cancellation_digest:
            raise DurableJobStoreCorruptionError(
                "durable job lineage cancellation digest does not match"
            )
        return cancellation


class JsonFileJobLineageStore:
    """Optional exact-owner metadata stored beside a durable job.

    The immutable process spec remains untouched for backwards compatibility.
    A missing lineage sidecar therefore means legacy/unscoped; it never means
    that a session-wide cancellation may claim the job.
    """

    def __init__(self, store: JsonFileJobStore) -> None:
        if not isinstance(store, JsonFileJobStore):
            raise TypeError("store must be a JsonFileJobStore")
        self.store = store

    def _scope_paths(
        self,
        execution_id: str,
        logical_execution_id: str,
        *,
        create: bool,
    ) -> dict[str, Path]:
        normalized_execution = _text(execution_id, "execution_id")
        normalized_logical = _text(
            logical_execution_id,
            "logical_execution_id",
        )
        root = self.store.namespace_dir / "lineages"
        execution_dir = root / hashlib.sha256(
            normalized_execution.encode("utf-8")
        ).hexdigest()[:32]
        if create:
            root_created = not root.exists()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            execution_created = not execution_dir.exists()
            execution_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if os.name != "nt":
                os.chmod(root, 0o700)
                os.chmod(execution_dir, 0o700)
                if root_created:
                    self.store._fsync_dir(root)
                    self.store._fsync_dir(self.store.namespace_dir)
                if execution_created:
                    self.store._fsync_dir(execution_dir)
                    self.store._fsync_dir(root)
        logical_digest = hashlib.sha256(
            normalized_logical.encode("utf-8")
        ).hexdigest()
        return {
            "lock": execution_dir / f".{logical_digest}.lock",
            "cancellation": execution_dir / f"{logical_digest}.cancel.json",
        }

    @contextmanager
    def lock_scope(
        self,
        *,
        execution_id: str,
        logical_execution_id: str,
    ) -> Iterator[None]:
        """Serialize scoped start and Stop using one durable lock order.

        Callers must acquire this lock before any job lock.  The scope lock is
        intentionally not acquired by bind/load/list_lineage so callers can
        hold it across reserve -> bind -> launch or tombstone -> scan -> marker.
        """

        paths = self._scope_paths(
            execution_id,
            logical_execution_id,
            create=True,
        )
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            yield

    def cancellation_locked(
        self,
        *,
        execution_id: str,
        logical_execution_id: str,
    ) -> DurableJobLineageCancellation | None:
        """Read a cancellation while the caller owns ``lock_scope``."""

        normalized_execution = _text(execution_id, "execution_id")
        normalized_logical = _text(
            logical_execution_id,
            "logical_execution_id",
        )
        paths = self._scope_paths(
            normalized_execution,
            normalized_logical,
            create=True,
        )
        path = paths["cancellation"]
        if not path.exists():
            return None
        cancellation = DurableJobLineageCancellation.from_dict(
            self.store._read_json(path, label="job lineage cancellation")
        )
        if (
            cancellation.store_id != self.store.store_id
            or cancellation.execution_id != normalized_execution
            or cancellation.logical_execution_id != normalized_logical
        ):
            raise DurableJobStoreCorruptionError(
                "durable job lineage cancellation identity does not match its path"
            )
        return cancellation

    def request_cancel_locked(
        self,
        *,
        execution_id: str,
        logical_execution_id: str,
        reason: str = "user_stop",
    ) -> DurableJobLineageCancellation:
        """Persist an idempotent tombstone while ``lock_scope`` is held."""

        existing = self.cancellation_locked(
            execution_id=execution_id,
            logical_execution_id=logical_execution_id,
        )
        if existing is not None:
            return existing
        cancellation = DurableJobLineageCancellation.build(
            store_id=self.store.store_id,
            execution_id=execution_id,
            logical_execution_id=logical_execution_id,
            requested_at_ms=self.store.now_ms(),
            reason=reason,
        )
        paths = self._scope_paths(
            execution_id,
            logical_execution_id,
            create=True,
        )
        self.store._write_json_atomic(
            paths["cancellation"],
            cancellation.to_dict(),
        )
        return cancellation

    def cancellation(
        self,
        *,
        execution_id: str,
        logical_execution_id: str,
    ) -> DurableJobLineageCancellation | None:
        with self.lock_scope(
            execution_id=execution_id,
            logical_execution_id=logical_execution_id,
        ):
            return self.cancellation_locked(
                execution_id=execution_id,
                logical_execution_id=logical_execution_id,
            )

    @staticmethod
    def _path(paths: dict[str, Any]):
        return paths["dir"] / "lineage.json"

    def reservation_guard(
        self,
        *,
        execution_id: str,
        logical_execution_id: str,
        created_by_attempt_id: str = "",
    ):
        """Build the callback used by the store's atomic scoped reservation."""

        normalized_execution = _text(execution_id, "execution_id")
        normalized_logical = _text(
            logical_execution_id,
            "logical_execution_id",
        )
        normalized_attempt = _text(
            created_by_attempt_id,
            "created_by_attempt_id",
            required=False,
        )

        def guard(
            paths: dict[str, Path],
            spec: dict[str, Any],
            creating_spec: bool,
        ) -> None:
            self.store._ensure_owner(spec, normalized_execution)
            requested = DurableJobLineage.build(
                store_id=self.store.store_id,
                job_id=str(spec["job_id"]),
                execution_id=normalized_execution,
                logical_execution_id=normalized_logical,
                created_by_attempt_id=normalized_attempt,
                request_digest=str(spec["request_digest"]),
            )
            existing = self._read_locked(
                paths,
                spec=spec,
                match_request_digest=not creating_spec,
            )
            if existing is not None:
                if (
                    existing.logical_execution_id == requested.logical_execution_id
                    and existing.request_digest == requested.request_digest
                ):
                    return
                raise DurableJobConflictError(
                    "durable job lineage is already bound to a different owner"
                )
            if not creating_spec:
                raise DurableJobConflictError(
                    "existing unscoped durable job cannot be claimed by a logical execution"
                )
            self.store._write_json_atomic(self._path(paths), requested.to_dict())
            # The sidecar write fsyncs the job directory; also persist the job
            # directory entry before the spec can become visible.
            self.store._fsync_dir(self.store.jobs_dir)

        return guard

    def bind(
        self,
        job_id: str,
        *,
        execution_id: str,
        logical_execution_id: str = "",
        created_by_attempt_id: str = "",
    ) -> DurableJobLineage | None:
        normalized_logical = _text(
            logical_execution_id,
            "logical_execution_id",
            required=False,
        )
        normalized_attempt = _text(
            created_by_attempt_id,
            "created_by_attempt_id",
            required=False,
        )
        if normalized_attempt and not normalized_logical:
            raise ValueError("created_by_attempt_id requires a logical_execution_id")
        paths = self.store.paths_for_worker(job_id)
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            spec = self.store._read_spec(paths["spec"])
            self.store._ensure_owner(spec, execution_id)
            existing = self._read_locked(paths, spec=spec)
            if not normalized_logical:
                if existing is not None:
                    raise DurableJobConflictError(
                        "durable job is already bound to a logical execution"
                    )
                return None
            requested = DurableJobLineage.build(
                store_id=self.store.store_id,
                job_id=job_id,
                execution_id=str(spec["execution_id"]),
                logical_execution_id=normalized_logical,
                created_by_attempt_id=normalized_attempt,
                request_digest=str(spec["request_digest"]),
            )
            if existing is not None:
                if existing == requested:
                    return existing
                raise DurableJobConflictError(
                    "durable job lineage is already bound to a different owner"
                )
            self.store._write_json_atomic(self._path(paths), requested.to_dict())
            return requested

    def load(
        self,
        job_id: str,
        *,
        execution_id: str,
    ) -> DurableJobLineage | None:
        paths = self.store.paths_for_worker(job_id)
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            spec = self.store._read_spec(paths["spec"])
            self.store._ensure_owner(spec, execution_id)
            return self._read_locked(paths, spec=spec)

    def list_lineage(
        self,
        *,
        execution_id: str,
        logical_execution_id: str,
    ) -> list[DurableJobSnapshot]:
        logical_id = _text(logical_execution_id, "logical_execution_id")
        matches: list[DurableJobSnapshot] = []
        for snapshot in self.store.list_execution(execution_id):
            lineage = self.load(snapshot.job_id, execution_id=execution_id)
            if lineage is not None and lineage.logical_execution_id == logical_id:
                matches.append(snapshot)
        return matches

    def _read_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
        match_request_digest: bool = True,
    ) -> DurableJobLineage | None:
        path = self._path(paths)
        if not path.exists():
            return None
        lineage = DurableJobLineage.from_dict(
            self.store._read_json(path, label="job lineage")
        )
        if (
            lineage.store_id != self.store.store_id
            or lineage.job_id != spec["job_id"]
            or lineage.execution_id != spec["execution_id"]
            or (
                match_request_digest
                and lineage.request_digest != spec["request_digest"]
            )
        ):
            raise DurableJobStoreCorruptionError(
                "durable job lineage does not belong to its job"
            )
        return lineage


__all__ = [
    "JOB_LINEAGE_CANCELLATION_SCHEMA_VERSION",
    "JOB_LINEAGE_SCHEMA_VERSION",
    "DurableJobLineageCancellation",
    "DurableJobLineage",
    "JsonFileJobLineageStore",
]
