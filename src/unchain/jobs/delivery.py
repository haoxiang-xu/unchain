from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from .lineage import JsonFileJobLineageStore
from .models import (
    DurableJobConflictError,
    DurableJobSnapshot,
    DurableJobStoreCorruptionError,
)
from .store import JsonFileJobStore, _exclusive_lock


JOB_RESULT_RECEIPT_SCHEMA_VERSION = 1
JOB_DELIVERY_JOURNAL_SCHEMA_VERSION = 1
_RECEIPT_ID_RE = re.compile(r"^job_receipt_[0-9a-f]{32}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        return json.loads(_canonical_json(value))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError(f"{label} must be strict JSON") from exc


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


def _sha256(value: Any, field: str) -> str:
    normalized = _text(value, field)
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return normalized


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _receipt_core(
    *,
    store_id: str,
    job_id: str,
    execution_id: str,
    logical_execution_id: str,
    created_by_attempt_id: str,
    source_attempt_id: str,
    delivery_key: str,
    action: str,
    operation_digest: str,
    job_revision: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    result_copy = _json_copy(result, label="result")
    if not isinstance(result_copy, dict):
        raise ValueError("result must be a JSON object")
    return {
        "schema_version": JOB_RESULT_RECEIPT_SCHEMA_VERSION,
        "store_id": _text(store_id, "store_id"),
        "job_id": _text(job_id, "job_id"),
        "execution_id": _text(execution_id, "execution_id"),
        "logical_execution_id": _text(
            logical_execution_id,
            "logical_execution_id",
            required=False,
        ),
        "created_by_attempt_id": _text(
            created_by_attempt_id,
            "created_by_attempt_id",
            required=False,
        ),
        "source_attempt_id": _text(
            source_attempt_id,
            "source_attempt_id",
            required=False,
        ),
        "delivery_key": _text(delivery_key, "delivery_key"),
        "action": _text(action, "action"),
        "operation_digest": _sha256(operation_digest, "operation_digest"),
        "job_revision": _non_negative_int(job_revision, "job_revision"),
        "result": result_copy,
        "result_digest": _digest(result_copy),
    }


@dataclass(frozen=True, slots=True)
class JobResultReceipt:
    schema_version: int
    receipt_id: str
    store_id: str
    job_id: str
    execution_id: str
    logical_execution_id: str
    created_by_attempt_id: str
    source_attempt_id: str
    delivery_key: str
    action: str
    operation_digest: str
    job_revision: int
    result_digest: str
    receipt_digest: str
    prepared_at_ms: int
    _result_json: str = field(repr=False)

    @property
    def result(self) -> dict[str, Any]:
        value = json.loads(self._result_json)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "store_id": self.store_id,
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "logical_execution_id": self.logical_execution_id,
            "created_by_attempt_id": self.created_by_attempt_id,
            "source_attempt_id": self.source_attempt_id,
            "delivery_key": self.delivery_key,
            "action": self.action,
            "operation_digest": self.operation_digest,
            "job_revision": self.job_revision,
            "result": self.result,
            "result_digest": self.result_digest,
            "receipt_digest": self.receipt_digest,
            "prepared_at_ms": self.prepared_at_ms,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        """Return non-model-visible metadata used by the commit/ack protocol."""

        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "receipt_digest": self.receipt_digest,
            "store_id": self.store_id,
            "job_id": self.job_id,
            "execution_id": self.execution_id,
            "logical_execution_id": self.logical_execution_id,
            "created_by_attempt_id": self.created_by_attempt_id,
            "source_attempt_id": self.source_attempt_id,
            "delivery_key": self.delivery_key,
            "action": self.action,
            "operation_digest": self.operation_digest,
            "job_revision": self.job_revision,
            "result_digest": self.result_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        store_id: str,
        job_id: str,
        execution_id: str,
        logical_execution_id: str,
        created_by_attempt_id: str,
        source_attempt_id: str,
        delivery_key: str,
        action: str,
        operation_digest: str,
        job_revision: int,
        result: dict[str, Any],
        prepared_at_ms: int,
    ) -> "JobResultReceipt":
        core = _receipt_core(
            store_id=store_id,
            job_id=job_id,
            execution_id=execution_id,
            logical_execution_id=logical_execution_id,
            created_by_attempt_id=created_by_attempt_id,
            source_attempt_id=source_attempt_id,
            delivery_key=delivery_key,
            action=action,
            operation_digest=operation_digest,
            job_revision=job_revision,
            result=result,
        )
        prepared_at = _non_negative_int(prepared_at_ms, "prepared_at_ms")
        receipt_digest = _digest(
            {
                **core,
                "prepared_at_ms": prepared_at,
            }
        )
        return cls(
            schema_version=JOB_RESULT_RECEIPT_SCHEMA_VERSION,
            receipt_id=f"job_receipt_{receipt_digest[:32]}",
            store_id=core["store_id"],
            job_id=core["job_id"],
            execution_id=core["execution_id"],
            logical_execution_id=core["logical_execution_id"],
            created_by_attempt_id=core["created_by_attempt_id"],
            source_attempt_id=core["source_attempt_id"],
            delivery_key=core["delivery_key"],
            action=core["action"],
            operation_digest=core["operation_digest"],
            job_revision=core["job_revision"],
            result_digest=core["result_digest"],
            receipt_digest=receipt_digest,
            prepared_at_ms=prepared_at,
            _result_json=_canonical_json(core["result"]),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> "JobResultReceipt":
        if not isinstance(raw, dict):
            raise DurableJobStoreCorruptionError(
                "durable job result receipt must be a JSON object"
            )
        expected = {
            "schema_version",
            "receipt_id",
            "store_id",
            "job_id",
            "execution_id",
            "logical_execution_id",
            "created_by_attempt_id",
            "source_attempt_id",
            "delivery_key",
            "action",
            "operation_digest",
            "job_revision",
            "result",
            "result_digest",
            "receipt_digest",
            "prepared_at_ms",
        }
        if (
            set(raw) != expected
            or raw.get("schema_version") != JOB_RESULT_RECEIPT_SCHEMA_VERSION
        ):
            raise DurableJobStoreCorruptionError(
                "durable job result receipt fields or schema version are invalid"
            )
        try:
            receipt = cls.build(
                store_id=raw["store_id"],
                job_id=raw["job_id"],
                execution_id=raw["execution_id"],
                logical_execution_id=raw["logical_execution_id"],
                created_by_attempt_id=raw["created_by_attempt_id"],
                source_attempt_id=raw["source_attempt_id"],
                delivery_key=raw["delivery_key"],
                action=raw["action"],
                operation_digest=raw["operation_digest"],
                job_revision=raw["job_revision"],
                result=raw["result"],
                prepared_at_ms=raw["prepared_at_ms"],
            )
        except (TypeError, ValueError) as exc:
            raise DurableJobStoreCorruptionError(
                "durable job result receipt is invalid"
            ) from exc
        if (
            raw.get("receipt_id") != receipt.receipt_id
            or raw.get("result_digest") != receipt.result_digest
            or raw.get("receipt_digest") != receipt.receipt_digest
        ):
            raise DurableJobStoreCorruptionError(
                "durable job result receipt digest does not match"
            )
        return receipt


@dataclass(frozen=True, slots=True)
class JobResultAcknowledgement:
    receipt_id: str
    applied_checkpoint_id: str
    acknowledged_at_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": JOB_RESULT_RECEIPT_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "applied_checkpoint_id": self.applied_checkpoint_id,
            "acknowledged_at_ms": self.acknowledged_at_ms,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> "JobResultAcknowledgement":
        if not isinstance(raw, dict) or set(raw) != {
            "schema_version",
            "receipt_id",
            "applied_checkpoint_id",
            "acknowledged_at_ms",
        }:
            raise DurableJobStoreCorruptionError(
                "durable job receipt acknowledgement is invalid"
            )
        if raw.get("schema_version") != JOB_RESULT_RECEIPT_SCHEMA_VERSION:
            raise DurableJobStoreCorruptionError(
                "durable job receipt acknowledgement schema is invalid"
            )
        try:
            receipt_id = _text(raw["receipt_id"], "receipt_id")
            checkpoint_id = _text(
                raw["applied_checkpoint_id"],
                "applied_checkpoint_id",
            )
            acknowledged_at_ms = _non_negative_int(
                raw["acknowledged_at_ms"],
                "acknowledged_at_ms",
            )
        except (TypeError, ValueError) as exc:
            raise DurableJobStoreCorruptionError(
                "durable job receipt acknowledgement fields are invalid"
            ) from exc
        if not _RECEIPT_ID_RE.fullmatch(receipt_id):
            raise DurableJobStoreCorruptionError(
                "durable job receipt acknowledgement id is invalid"
            )
        return cls(
            receipt_id=receipt_id,
            applied_checkpoint_id=checkpoint_id,
            acknowledged_at_ms=acknowledged_at_ms,
        )


def build_job_result_payload(
    snapshot: DurableJobSnapshot,
    *,
    action: str,
    logs: dict[str, Any],
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stdout = str(logs.get("stdout") or "")
    stderr = str(logs.get("stderr") or "")
    stdout_start_offset = int(logs.get("stdout_start_offset") or 0)
    stderr_start_offset = int(logs.get("stderr_start_offset") or 0)
    next_stdout_offset = int(logs.get("stdout_offset") or 0)
    next_stderr_offset = int(logs.get("stderr_offset") or 0)
    consumer_truncated = bool(logs.get("truncated"))
    result: dict[str, Any] = {
        "ok": snapshot.ok,
        "action": action,
        "status": snapshot.status,
        "task_id": snapshot.job_id,
        "job_id": snapshot.job_id,
        "execution_id": snapshot.execution_id,
        "adapter": snapshot.adapter,
        "durable": True,
        "background": True,
        "stdout": stdout,
        "stderr": stderr,
        "completed": snapshot.completed,
        "returncode": snapshot.returncode,
        "timed_out": snapshot.timed_out,
        "cancelled": snapshot.cancelled,
        "outcome_unknown": snapshot.status == "outcome_unknown",
        "outcome_unknown_reason": snapshot.outcome_unknown_reason,
        "error": snapshot.error,
        "stdout_truncated": snapshot.stdout_truncated,
        "stderr_truncated": snapshot.stderr_truncated,
        "truncated": (
            consumer_truncated or snapshot.stdout_truncated or snapshot.stderr_truncated
        ),
        "stdout_offset": stdout_start_offset,
        "stderr_offset": stderr_start_offset,
        "next_stdout_offset": next_stdout_offset,
        "next_stderr_offset": next_stderr_offset,
        "offset_unit": str(logs.get("offset_unit") or "utf8_bytes"),
        "stdout_available": int(logs.get("stdout_available") or 0),
        "stderr_available": int(logs.get("stderr_available") or 0),
    }
    if updates:
        copied_updates = _json_copy(updates, label="result updates")
        if not isinstance(copied_updates, dict):
            raise ValueError("result updates must be a JSON object")
        overlap = set(result).intersection(copied_updates)
        if overlap:
            raise ValueError(
                f"result updates cannot replace standard fields: {sorted(overlap)}"
            )
        result.update(copied_updates)
    return result


class JsonFileJobDeliveryJournal:
    """Crash-idempotent prepare/ack delivery for durable job results.

    Receipt payloads are immutable files. The journal is the single atomic
    authority for both committed log offsets and acknowledgement application,
    so a session/checkpoint commit can happen before one idempotent ack.
    """

    def __init__(
        self,
        store: JsonFileJobStore,
        *,
        lineage_store: JsonFileJobLineageStore | None = None,
    ) -> None:
        if not isinstance(store, JsonFileJobStore):
            raise TypeError("store must be a JsonFileJobStore")
        self.store = store
        self.lineage_store = lineage_store or JsonFileJobLineageStore(store)

    @staticmethod
    def _delivery_paths(paths: dict[str, Any]) -> dict[str, Any]:
        return {
            "journal": paths["dir"] / "delivery-journal.json",
            "receipts": paths["dir"] / "delivery-receipts",
        }

    def lookup(
        self,
        job_id: str,
        *,
        execution_id: str,
        delivery_key: str,
        action: str,
        operation_digest: str,
        logical_execution_id: str = "",
        created_by_attempt_id: str = "",
    ) -> JobResultReceipt | None:
        identity = self._normalize_delivery_identity(
            delivery_key=delivery_key,
            action=action,
            operation_digest=operation_digest,
            logical_execution_id=logical_execution_id,
            created_by_attempt_id=created_by_attempt_id,
        )
        paths = self.store.paths_for_worker(job_id)
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            spec = self.store._read_spec(paths["spec"])
            self.store._ensure_owner(spec, execution_id)
            lineage = self._require_lineage_locked(
                paths,
                spec=spec,
                logical_execution_id=identity["logical_execution_id"],
                created_by_attempt_id=identity["created_by_attempt_id"],
            )
            journal = self._read_journal_locked(paths, spec=spec)
            if journal is None:
                journal = self._new_journal_locked(paths, spec=spec)
            existing = self._lookup_locked(
                paths,
                journal=journal,
                identity=identity,
                lineage=lineage,
            )
            if existing is not None:
                return existing

            # A host can crash after the immutable receipt is fsynced but
            # before its journal entry is published.  Recover that receipt in
            # lookup, before callers perform wait/cancel side effects again.
            orphan = self._find_orphan_receipt_locked(
                paths,
                spec=spec,
                journal=journal,
                identity=identity,
                lineage=lineage,
            )
            if orphan is None:
                return None
            self._publish_receipt_locked(paths, journal=journal, receipt=orphan)
            return orphan

    def prepare(
        self,
        job_id: str,
        *,
        execution_id: str,
        delivery_key: str,
        action: str,
        operation_digest: str,
        max_output_chars: int,
        logical_execution_id: str = "",
        created_by_attempt_id: str = "",
        source_attempt_id: str = "",
        result_updates: dict[str, Any] | None = None,
    ) -> JobResultReceipt:
        if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int):
            raise TypeError("max_output_chars must be an integer")
        limit = max(0, max_output_chars)
        identity = self._normalize_delivery_identity(
            delivery_key=delivery_key,
            action=action,
            operation_digest=operation_digest,
            logical_execution_id=logical_execution_id,
            created_by_attempt_id=created_by_attempt_id,
        )
        source_attempt = _text(
            source_attempt_id,
            "source_attempt_id",
            required=False,
        )
        paths = self.store.paths_for_worker(job_id)
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            spec = self.store._read_spec(paths["spec"])
            self.store._ensure_owner(spec, execution_id)
            lineage = self._require_lineage_locked(
                paths,
                spec=spec,
                logical_execution_id=identity["logical_execution_id"],
                created_by_attempt_id=identity["created_by_attempt_id"],
            )
            journal = self._read_journal_locked(paths, spec=spec)
            if journal is None:
                journal = self._new_journal_locked(paths, spec=spec)
            existing = self._lookup_locked(
                paths,
                journal=journal,
                identity=identity,
                lineage=lineage,
            )
            if existing is not None:
                return existing

            orphan = self._find_orphan_receipt_locked(
                paths,
                spec=spec,
                journal=journal,
                identity=identity,
                lineage=lineage,
            )
            if orphan is not None:
                self._publish_receipt_locked(paths, journal=journal, receipt=orphan)
                return orphan

            state = self.store._read_state(
                paths["state"],
                expected_job_id=job_id,
            )
            self.store._ensure_worker_token_matches(spec, state)
            snapshot = self.store._snapshot(spec, state)
            cursor = journal["committed_cursor"]
            final = snapshot.completed and snapshot.status != "outcome_unknown"
            stdout = self.store._read_log_chunk(
                paths["stdout"],
                offset=int(cursor["stdout_offset"]),
                max_chars=limit,
                final=final,
            )
            stderr = self.store._read_log_chunk(
                paths["stderr"],
                offset=int(cursor["stderr_offset"]),
                max_chars=limit,
                final=final,
            )
            logs = self._logs_result(stdout, stderr)
            result = build_job_result_payload(
                snapshot,
                action=identity["action"],
                logs=logs,
                updates=result_updates,
            )
            receipt = JobResultReceipt.build(
                store_id=self.store.store_id,
                job_id=job_id,
                execution_id=str(spec["execution_id"]),
                logical_execution_id=(
                    lineage.logical_execution_id if lineage is not None else ""
                ),
                created_by_attempt_id=(
                    lineage.created_by_attempt_id if lineage is not None else ""
                ),
                source_attempt_id=source_attempt,
                delivery_key=identity["delivery_key"],
                action=identity["action"],
                operation_digest=identity["operation_digest"],
                job_revision=snapshot.revision,
                result=result,
                prepared_at_ms=self.store.now_ms(),
            )
            self._write_immutable_receipt_locked(paths, receipt)
            self._publish_receipt_locked(paths, journal=journal, receipt=receipt)
            return receipt

    def acknowledge(
        self,
        job_id: str,
        *,
        execution_id: str,
        receipt_id: str,
        applied_checkpoint_id: str,
        expected_receipt_digest: str | None = None,
        expected_result_digest: str | None = None,
        expected_delivery_key: str | None = None,
        expected_operation_digest: str | None = None,
    ) -> JobResultAcknowledgement:
        normalized_receipt_id = _text(receipt_id, "receipt_id")
        checkpoint_id = _text(
            applied_checkpoint_id,
            "applied_checkpoint_id",
        )
        if not _RECEIPT_ID_RE.fullmatch(normalized_receipt_id):
            raise ValueError("receipt_id is invalid")
        paths = self.store.paths_for_worker(job_id)
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            spec = self.store._read_spec(paths["spec"])
            self.store._ensure_owner(spec, execution_id)
            journal = self._read_journal_locked(paths, spec=spec)
            if journal is None:
                raise DurableJobConflictError(
                    "durable job has no result delivery journal"
                )
            delivery_key, entry = self._entry_for_receipt(
                journal,
                normalized_receipt_id,
            )
            receipt = self._read_receipt_locked(paths, entry=entry)
            expected_identity = {
                "receipt_digest": (
                    _sha256(expected_receipt_digest, "expected_receipt_digest")
                    if expected_receipt_digest is not None
                    else receipt.receipt_digest
                ),
                "result_digest": (
                    _sha256(expected_result_digest, "expected_result_digest")
                    if expected_result_digest is not None
                    else receipt.result_digest
                ),
                "delivery_key": (
                    _text(expected_delivery_key, "expected_delivery_key")
                    if expected_delivery_key is not None
                    else receipt.delivery_key
                ),
                "operation_digest": (
                    _sha256(
                        expected_operation_digest,
                        "expected_operation_digest",
                    )
                    if expected_operation_digest is not None
                    else receipt.operation_digest
                ),
            }
            actual_identity = {
                "receipt_digest": receipt.receipt_digest,
                "result_digest": receipt.result_digest,
                "delivery_key": receipt.delivery_key,
                "operation_digest": receipt.operation_digest,
            }
            if expected_identity != actual_identity:
                raise DurableJobConflictError(
                    "durable job receipt acknowledgement identity does not match"
                )
            application_raw = entry.get("application")
            if application_raw is not None:
                application = JobResultAcknowledgement.from_dict(application_raw)
                if application.applied_checkpoint_id != checkpoint_id:
                    raise DurableJobConflictError(
                        "durable job receipt was acknowledged by a different checkpoint"
                    )
                return application

            cursor = journal["committed_cursor"]
            expected_offsets = (
                receipt.result["stdout_offset"],
                receipt.result["stderr_offset"],
            )
            current_offsets = (
                cursor["stdout_offset"],
                cursor["stderr_offset"],
            )
            if current_offsets != expected_offsets:
                raise DurableJobConflictError(
                    "durable job committed cursor no longer matches its pending receipt"
                )
            application = JobResultAcknowledgement(
                receipt_id=receipt.receipt_id,
                applied_checkpoint_id=checkpoint_id,
                acknowledged_at_ms=self.store.now_ms(),
            )
            entries = copy.deepcopy(journal["entries"])
            entries[delivery_key]["application"] = application.to_dict()
            updated = {
                **journal,
                "committed_cursor": {
                    "stdout_offset": receipt.result["next_stdout_offset"],
                    "stderr_offset": receipt.result["next_stderr_offset"],
                    "revision": int(cursor["revision"]) + 1,
                },
                "entries": entries,
                "revision": int(journal["revision"]) + 1,
            }
            self.store._write_json_atomic(
                self._delivery_paths(paths)["journal"],
                updated,
            )
            return application

    def pending_for_job(
        self,
        job_id: str,
        *,
        execution_id: str,
    ) -> list[JobResultReceipt]:
        paths = self.store.paths_for_worker(job_id)
        with _exclusive_lock(paths["lock"]):
            self.store._assert_store_identity()
            spec = self.store._read_spec(paths["spec"])
            self.store._ensure_owner(spec, execution_id)
            journal = self._read_journal_locked(paths, spec=spec)
            if journal is None:
                return []
            pending: list[JobResultReceipt] = []
            for delivery_key in journal["order"]:
                entry = journal["entries"][delivery_key]
                if entry.get("application") is None:
                    pending.append(self._read_receipt_locked(paths, entry=entry))
            return pending

    def consume_legacy_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
        max_output_chars: int,
        final: bool,
    ) -> dict[str, Any] | None:
        """Use the journal cursor after receipt delivery has been enabled.

        The caller must hold the job lock. A pending receipt blocks destructive
        legacy reads until its checkpoint application has been acknowledged.
        """

        journal = self._read_journal_locked(paths, spec=spec)
        if journal is None:
            empty_journal = self._new_journal_locked(paths, spec=spec)
            self._ensure_no_orphan_receipts_locked(
                paths,
                spec=spec,
                journal=empty_journal,
            )
            return None
        self._ensure_no_orphan_receipts_locked(
            paths,
            spec=spec,
            journal=journal,
        )
        self._ensure_no_pending(journal)
        cursor = journal["committed_cursor"]
        stdout = self.store._read_log_chunk(
            paths["stdout"],
            offset=int(cursor["stdout_offset"]),
            max_chars=max_output_chars,
            final=final,
        )
        stderr = self.store._read_log_chunk(
            paths["stderr"],
            offset=int(cursor["stderr_offset"]),
            max_chars=max_output_chars,
            final=final,
        )
        updated = {
            **journal,
            "committed_cursor": {
                "stdout_offset": stdout["next_offset"],
                "stderr_offset": stderr["next_offset"],
                "revision": int(cursor["revision"]) + 1,
            },
            "revision": int(journal["revision"]) + 1,
        }
        self.store._write_json_atomic(
            self._delivery_paths(paths)["journal"],
            updated,
        )
        return self._logs_result(stdout, stderr)

    def _ensure_no_orphan_receipts_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
        journal: dict[str, Any],
    ) -> None:
        """Block destructive legacy reads behind any immutable orphan.

        A receipt file is the frozen result authority even if its journal
        publication crashed.  Legacy cursor advancement must not make that
        frozen result unrecoverable.
        """

        receipt_dir = self._delivery_paths(paths)["receipts"]
        if not receipt_dir.exists():
            return
        registered_ids = {
            entry["receipt_id"] for entry in journal["entries"].values()
        }
        for receipt_path in sorted(receipt_dir.glob("job_receipt_*.json")):
            receipt = JobResultReceipt.from_dict(
                self.store._read_json(
                    receipt_path,
                    label="orphan job result receipt",
                )
            )
            if receipt_path.name != f"{receipt.receipt_id}.json":
                raise DurableJobStoreCorruptionError(
                    "durable job result receipt filename does not match its id"
                )
            if (
                receipt.store_id != self.store.store_id
                or receipt.job_id != spec["job_id"]
                or receipt.execution_id != spec["execution_id"]
            ):
                raise DurableJobStoreCorruptionError(
                    "durable job result receipt does not belong to its job"
                )
            if receipt.receipt_id not in registered_ids:
                raise DurableJobConflictError(
                    "unregistered immutable job result receipt blocks legacy log consumption"
                )

    def _lookup_locked(
        self,
        paths: dict[str, Any],
        *,
        journal: dict[str, Any],
        identity: dict[str, str],
        lineage: Any,
    ) -> JobResultReceipt | None:
        entry = journal["entries"].get(identity["delivery_key"])
        if entry is None:
            self._ensure_no_pending(journal)
            return None
        receipt = self._read_receipt_locked(paths, entry=entry)
        expected_logical = lineage.logical_execution_id if lineage is not None else ""
        expected_attempt = lineage.created_by_attempt_id if lineage is not None else ""
        if (
            receipt.delivery_key != identity["delivery_key"]
            or receipt.action != identity["action"]
            or receipt.operation_digest != identity["operation_digest"]
            or receipt.logical_execution_id != expected_logical
            or receipt.created_by_attempt_id != expected_attempt
        ):
            raise DurableJobConflictError(
                "durable job delivery key is already bound to a different operation"
            )
        return receipt

    @staticmethod
    def _normalize_delivery_identity(
        *,
        delivery_key: str,
        action: str,
        operation_digest: str,
        logical_execution_id: str,
        created_by_attempt_id: str,
    ) -> dict[str, str]:
        return {
            "delivery_key": _text(delivery_key, "delivery_key"),
            "action": _text(action, "action"),
            "operation_digest": _sha256(operation_digest, "operation_digest"),
            "logical_execution_id": _text(
                logical_execution_id,
                "logical_execution_id",
                required=False,
            ),
            "created_by_attempt_id": _text(
                created_by_attempt_id,
                "created_by_attempt_id",
                required=False,
            ),
        }

    def _require_lineage_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
        logical_execution_id: str,
        created_by_attempt_id: str,
    ):
        lineage = self.lineage_store._read_locked(paths, spec=spec)
        stored_logical = lineage.logical_execution_id if lineage is not None else ""
        # ``created_by_attempt_id`` is creation provenance, not the current
        # delivery attempt.  A resumed runtime may legitimately carry a new
        # attempt id.  The immutable receipt always copies creation provenance
        # from the sidecar and stores the current attempt separately as
        # ``source_attempt_id``.
        del created_by_attempt_id
        if logical_execution_id != stored_logical:
            raise DurableJobConflictError(
                "durable job receipt owner does not match its logical execution"
            )
        return lineage

    def _find_orphan_receipt_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
        journal: dict[str, Any],
        identity: dict[str, str],
        lineage: Any,
    ) -> JobResultReceipt | None:
        """Recover a receipt durably written before its journal publication.

        The receipt file is the frozen payload.  Retrying must publish that
        exact file instead of reading a later log tail or assigning a new
        timestamp.  This closes the only prepare-side crash window.
        """

        receipt_dir = self._delivery_paths(paths)["receipts"]
        if not receipt_dir.exists():
            return None
        registered_ids = {entry["receipt_id"] for entry in journal["entries"].values()}
        expected_logical = lineage.logical_execution_id if lineage is not None else ""
        expected_attempt = lineage.created_by_attempt_id if lineage is not None else ""
        candidate: JobResultReceipt | None = None
        for receipt_path in sorted(receipt_dir.glob("job_receipt_*.json")):
            receipt = JobResultReceipt.from_dict(
                self.store._read_json(
                    receipt_path,
                    label="orphan job result receipt",
                )
            )
            if receipt_path.name != f"{receipt.receipt_id}.json":
                raise DurableJobStoreCorruptionError(
                    "durable job result receipt filename does not match its id"
                )
            if (
                receipt.store_id != self.store.store_id
                or receipt.job_id != spec["job_id"]
                or receipt.execution_id != spec["execution_id"]
            ):
                raise DurableJobStoreCorruptionError(
                    "durable job result receipt does not belong to its job"
                )
            if receipt.receipt_id in registered_ids:
                continue
            cursor = journal["committed_cursor"]
            if (
                receipt.result["stdout_offset"] != cursor["stdout_offset"]
                or receipt.result["stderr_offset"] != cursor["stderr_offset"]
            ):
                raise DurableJobStoreCorruptionError(
                    "unregistered durable job receipt is not aligned with the "
                    "committed cursor"
                )
            if receipt.delivery_key != identity["delivery_key"]:
                raise DurableJobConflictError(
                    "another orphan durable job receipt reserves the committed cursor"
                )
            if (
                receipt.action != identity["action"]
                or receipt.operation_digest != identity["operation_digest"]
                or receipt.logical_execution_id != expected_logical
                or receipt.created_by_attempt_id != expected_attempt
            ):
                raise DurableJobConflictError(
                    "orphan durable job delivery key belongs to another operation"
                )
            if candidate is not None and candidate != receipt:
                raise DurableJobStoreCorruptionError(
                    "multiple orphan durable job receipts claim one delivery key"
                )
            candidate = receipt
        return candidate

    def _publish_receipt_locked(
        self,
        paths: dict[str, Any],
        *,
        journal: dict[str, Any],
        receipt: JobResultReceipt,
    ) -> None:
        entries = copy.deepcopy(journal["entries"])
        if receipt.delivery_key in entries:
            raise DurableJobConflictError(
                "durable job delivery key is already registered"
            )
        entries[receipt.delivery_key] = {
            "receipt_id": receipt.receipt_id,
            "receipt_digest": receipt.receipt_digest,
            "operation_digest": receipt.operation_digest,
            "action": receipt.action,
            "application": None,
        }
        order = list(journal["order"])
        order.append(receipt.delivery_key)
        updated = {
            **journal,
            "entries": entries,
            "order": order,
            "revision": int(journal["revision"]) + 1,
        }
        self.store._write_json_atomic(
            self._delivery_paths(paths)["journal"],
            updated,
        )

    def _new_journal_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
    ) -> dict[str, Any]:
        legacy_cursor = self.store._read_cursor(
            paths["cursor"],
            expected_job_id=str(spec["job_id"]),
        )
        return {
            "schema_version": JOB_DELIVERY_JOURNAL_SCHEMA_VERSION,
            "store_id": self.store.store_id,
            "job_id": str(spec["job_id"]),
            "committed_cursor": {
                "stdout_offset": int(legacy_cursor["stdout_offset"]),
                "stderr_offset": int(legacy_cursor["stderr_offset"]),
                "revision": int(legacy_cursor["revision"]),
            },
            "entries": {},
            "order": [],
            "revision": 0,
        }

    def _read_journal_locked(
        self,
        paths: dict[str, Any],
        *,
        spec: dict[str, Any],
    ) -> dict[str, Any] | None:
        path = self._delivery_paths(paths)["journal"]
        if not path.exists():
            return None
        raw = self.store._read_json(path, label="job delivery journal")
        expected = {
            "schema_version",
            "store_id",
            "job_id",
            "committed_cursor",
            "entries",
            "order",
            "revision",
        }
        if (
            set(raw) != expected
            or raw.get("schema_version") != JOB_DELIVERY_JOURNAL_SCHEMA_VERSION
            or raw.get("store_id") != self.store.store_id
            or raw.get("job_id") != spec["job_id"]
        ):
            raise DurableJobStoreCorruptionError(
                "durable job delivery journal identity is invalid"
            )
        cursor = raw.get("committed_cursor")
        if not isinstance(cursor, dict) or set(cursor) != {
            "stdout_offset",
            "stderr_offset",
            "revision",
        }:
            raise DurableJobStoreCorruptionError(
                "durable job delivery cursor is invalid"
            )
        try:
            for field_name in ("stdout_offset", "stderr_offset", "revision"):
                _non_negative_int(cursor[field_name], field_name)
            _non_negative_int(raw.get("revision"), "revision")
        except ValueError as exc:
            raise DurableJobStoreCorruptionError(
                "durable job delivery cursor fields are invalid"
            ) from exc
        entries = raw.get("entries")
        order = raw.get("order")
        if not isinstance(entries, dict) or not isinstance(order, list):
            raise DurableJobStoreCorruptionError(
                "durable job delivery entries are invalid"
            )
        if len(order) != len(set(order)) or set(order) != set(entries):
            raise DurableJobStoreCorruptionError(
                "durable job delivery order does not match its entries"
            )
        for delivery_key in order:
            if not isinstance(delivery_key, str) or not delivery_key:
                raise DurableJobStoreCorruptionError(
                    "durable job delivery key is invalid"
                )
            entry = entries.get(delivery_key)
            if not isinstance(entry, dict) or set(entry) != {
                "receipt_id",
                "receipt_digest",
                "operation_digest",
                "action",
                "application",
            }:
                raise DurableJobStoreCorruptionError(
                    "durable job delivery entry is invalid"
                )
            if (
                not isinstance(entry["receipt_id"], str)
                or not _RECEIPT_ID_RE.fullmatch(entry["receipt_id"])
                or not isinstance(entry["receipt_digest"], str)
                or not _SHA256_RE.fullmatch(entry["receipt_digest"])
                or not isinstance(entry["operation_digest"], str)
                or not _SHA256_RE.fullmatch(entry["operation_digest"])
                or not isinstance(entry["action"], str)
                or not entry["action"]
            ):
                raise DurableJobStoreCorruptionError(
                    "durable job delivery entry fields are invalid"
                )
            if entry["application"] is not None:
                application = JobResultAcknowledgement.from_dict(entry["application"])
                if application.receipt_id != entry["receipt_id"]:
                    raise DurableJobStoreCorruptionError(
                        "durable job delivery application has the wrong receipt"
                    )
        return raw

    def _write_immutable_receipt_locked(
        self,
        paths: dict[str, Any],
        receipt: JobResultReceipt,
    ) -> None:
        delivery_paths = self._delivery_paths(paths)
        receipt_path = delivery_paths["receipts"] / f"{receipt.receipt_id}.json"
        if receipt_path.exists():
            existing = JobResultReceipt.from_dict(
                self.store._read_json(receipt_path, label="job result receipt")
            )
            if existing != receipt:
                raise DurableJobConflictError(
                    "durable job result receipt id already has different content"
                )
            return
        self.store._write_json_atomic(receipt_path, receipt.to_dict())
        # ``_write_json_atomic`` fsyncs the receipt directory itself. Also
        # persist the directory entry in the job folder before publishing the
        # journal so the orphan-recovery contract survives a host crash.
        self.store._fsync_dir(delivery_paths["receipts"].parent)

    def _read_receipt_locked(
        self,
        paths: dict[str, Any],
        *,
        entry: dict[str, Any],
    ) -> JobResultReceipt:
        receipt_id = str(entry["receipt_id"])
        receipt_path = self._delivery_paths(paths)["receipts"] / f"{receipt_id}.json"
        receipt = JobResultReceipt.from_dict(
            self.store._read_json(receipt_path, label="job result receipt")
        )
        if (
            receipt.receipt_id != receipt_id
            or receipt.receipt_digest != entry["receipt_digest"]
            or receipt.operation_digest != entry["operation_digest"]
            or receipt.action != entry["action"]
        ):
            raise DurableJobStoreCorruptionError(
                "durable job delivery entry does not match its immutable receipt"
            )
        return receipt

    @staticmethod
    def _entry_for_receipt(
        journal: dict[str, Any],
        receipt_id: str,
    ) -> tuple[str, dict[str, Any]]:
        for delivery_key in journal["order"]:
            entry = journal["entries"][delivery_key]
            if entry["receipt_id"] == receipt_id:
                return delivery_key, entry
        raise DurableJobConflictError(
            "durable job result receipt is not registered in its journal"
        )

    @staticmethod
    def _ensure_no_pending(journal: dict[str, Any]) -> None:
        if any(
            journal["entries"][delivery_key].get("application") is None
            for delivery_key in journal["order"]
        ):
            raise DurableJobConflictError(
                "durable job has a pending result receipt that must be acknowledged"
            )

    @staticmethod
    def _logs_result(
        stdout: dict[str, Any],
        stderr: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "stdout": stdout["text"],
            "stderr": stderr["text"],
            "stdout_start_offset": stdout["start_offset"],
            "stderr_start_offset": stderr["start_offset"],
            "stdout_offset": stdout["next_offset"],
            "stderr_offset": stderr["next_offset"],
            "stdout_available": stdout["available"],
            "stderr_available": stderr["available"],
            "offset_unit": "utf8_bytes",
            "truncated": stdout["has_more"] or stderr["has_more"],
        }


__all__ = [
    "JOB_DELIVERY_JOURNAL_SCHEMA_VERSION",
    "JOB_RESULT_RECEIPT_SCHEMA_VERSION",
    "JobResultAcknowledgement",
    "JobResultReceipt",
    "JsonFileJobDeliveryJournal",
    "build_job_result_payload",
]
