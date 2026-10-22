from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from weakref import WeakValueDictionary

from .environment import JobEnvironmentProfile
from .models import (
    JOB_SCHEMA_VERSION,
    JOB_STATUSES,
    TERMINAL_JOB_STATUSES,
    DurableJobConflictError,
    DurableJobNotFoundError,
    DurableJobOwnershipError,
    DurableJobSnapshot,
    DurableJobStoreCorruptionError,
)

if os.name == "nt":  # pragma: no cover - exercised on Windows
    import msvcrt
else:  # pragma: no cover - platform selection itself is trivial
    import fcntl


_JOB_ID_RE = re.compile(r"^job_[0-9a-f]{32}$")
_STORE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
STORE_MANIFEST_SCHEMA_VERSION = 1
_THREAD_LOCKS: WeakValueDictionary[str, threading.RLock] = WeakValueDictionary()
_THREAD_LOCKS_GUARD = threading.Lock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        lock = _THREAD_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _THREAD_LOCKS[key] = lock
        return lock


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    lock = _thread_lock(path)
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("a+b") as lock_file:
            if os.name != "nt":
                os.chmod(path, 0o600)
            if os.name == "nt":  # pragma: no cover - exercised on Windows
                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if os.name == "nt":  # pragma: no cover - exercised on Windows
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class JsonFileJobStore:
    """Crash-safe local store for detached process jobs.

    Immutable launch specs, mutable worker state, append-only logs, and the
    consumer cursor are stored separately so a new runtime can reattach without
    serializing a process-local ``Popen`` object.
    """

    def __init__(self, base_dir: str | Path, *, clock_ms=None) -> None:
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.stores_dir = self.base_dir / "stores"
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self.base_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._manifest_path = self.base_dir / "store.json"
        self._manifest_lock_path = self.base_dir / ".store.lock"
        with _exclusive_lock(self._manifest_lock_path):
            if self._manifest_path.exists():
                manifest = self._read_store_manifest()
            else:
                if self._has_unbound_job_data():
                    raise DurableJobStoreCorruptionError(
                        "durable job store manifest is missing while jobs exist"
                    )
                store_id = uuid.uuid4().hex
                jobs_dir = self.stores_dir / store_id / "jobs"
                jobs_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
                if os.name != "nt":
                    os.chmod(self.stores_dir, 0o700)
                    os.chmod(jobs_dir.parent, 0o700)
                    os.chmod(jobs_dir, 0o700)
                    self._fsync_dir(jobs_dir)
                    self._fsync_dir(jobs_dir.parent)
                    self._fsync_dir(self.stores_dir)
                manifest = {
                    "schema_version": STORE_MANIFEST_SCHEMA_VERSION,
                    "store_id": store_id,
                    "created_at_ms": self.now_ms(),
                }
                self._write_json_atomic(self._manifest_path, manifest)
        self.store_id = str(manifest["store_id"])
        self.namespace_dir = self.stores_dir / self.store_id
        self.jobs_dir = self.namespace_dir / "jobs"
        if not self.jobs_dir.is_dir():
            raise DurableJobStoreCorruptionError(
                "durable job store namespace is missing"
            )
        if os.name != "nt":
            os.chmod(self.base_dir, 0o700)
            os.chmod(self.stores_dir, 0o700)
            os.chmod(self.namespace_dir, 0o700)
            os.chmod(self.jobs_dir, 0o700)

    def now_ms(self) -> int:
        value = self._clock_ms()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError("clock_ms must return a non-negative integer timestamp")
        return value

    @staticmethod
    def durable_job_id(execution_id: str, idempotency_key: str) -> str:
        identity = f"durable-process-job-v1\0{execution_id}\0{idempotency_key}"
        return f"job_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]}"

    def reserve_process_job(
        self,
        *,
        execution_id: str,
        adapter: str,
        argv: list[str],
        cwd: str,
        timeout_ms: int,
        idempotency_key: str,
        intent_digest: str,
        max_log_bytes: int,
        environment_digest: str | None = None,
    ) -> tuple[DurableJobSnapshot, bool]:
        execution_id = self._required_text(execution_id, "execution_id")
        adapter = self._required_text(adapter, "adapter")
        idempotency_key = self._required_text(idempotency_key, "idempotency_key")
        intent_digest = self._required_text(intent_digest, "intent_digest")
        if not re.fullmatch(r"[0-9a-f]{64}", intent_digest):
            raise ValueError("intent_digest must be a lowercase SHA-256 digest")
        resolved_environment_digest = str(environment_digest or "").strip()
        if not resolved_environment_digest:
            resolved_environment_digest = JobEnvironmentProfile.capture().digest
        if not re.fullmatch(r"[0-9a-f]{64}", resolved_environment_digest):
            raise ValueError(
                "environment_digest must be a lowercase SHA-256 digest"
            )
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
            raise ValueError("argv must be a non-empty list of strings")
        cwd_path = Path(self._required_text(cwd, "cwd"))
        if not cwd_path.is_absolute():
            raise ValueError("cwd must be absolute")
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        if (
            isinstance(max_log_bytes, bool)
            or not isinstance(max_log_bytes, int)
            or max_log_bytes <= 0
        ):
            raise ValueError("max_log_bytes must be a positive integer")

        job_id = self.durable_job_id(execution_id, idempotency_key)
        paths = self._paths(job_id, create=True)
        request = {
            "store_id": self.store_id,
            "execution_id": execution_id,
            "adapter": adapter,
            "argv": list(argv),
            "cwd": str(cwd_path),
            "timeout_ms": timeout_ms,
            "idempotency_digest": _digest(idempotency_key),
            "intent_digest": intent_digest,
            "environment_digest": resolved_environment_digest,
            "max_log_bytes": max_log_bytes,
        }
        request_digest = _digest(request)
        created = False
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            if paths["spec"].exists():
                spec = self._read_spec(paths["spec"])
                if spec["request_digest"] != request_digest:
                    raise DurableJobConflictError(
                        "the durable job idempotency key is already bound to a different intent"
                    )
            else:
                created_at_ms = self.now_ms()
                spec = {
                    "schema_version": JOB_SCHEMA_VERSION,
                    "job_id": job_id,
                    **request,
                    "request_digest": request_digest,
                    "worker_token": uuid.uuid4().hex,
                    "created_at_ms": created_at_ms,
                }
                self._write_json_atomic(paths["spec"], spec)
                created = True

            self._ensure_initial_files(paths, spec)
        return self.load(job_id, execution_id=execution_id), created

    def load(self, job_id: str, *, execution_id: str) -> DurableJobSnapshot:
        paths = self._paths(job_id)
        if not paths["spec"].exists() or not paths["state"].exists():
            raise DurableJobNotFoundError(f"durable job not found: {job_id}")
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            self._ensure_owner(spec, execution_id)
            state = self._read_state(paths["state"], expected_job_id=job_id)
            self._ensure_worker_token_matches(spec, state)
            return self._snapshot(spec, state)

    def load_spec(self, job_id: str) -> dict[str, Any]:
        paths = self._paths(job_id)
        if not paths["spec"].exists():
            raise DurableJobNotFoundError(f"durable job not found: {job_id}")
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            return self._read_spec(paths["spec"])

    def contains(self, job_id: str) -> bool:
        try:
            paths = self._paths(job_id)
        except DurableJobNotFoundError:
            return False
        exists = paths["spec"].exists()
        self._assert_store_identity()
        return exists

    def update_state(
        self,
        job_id: str,
        *,
        worker_token: str,
        status: str | None = None,
        **updates: Any,
    ) -> DurableJobSnapshot:
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            if str(spec["worker_token"]) != str(worker_token or ""):
                raise DurableJobConflictError("durable job worker token does not match")
            state = self._read_state(paths["state"], expected_job_id=job_id)
            self._ensure_worker_token_matches(spec, state)
            next_status = str(status or state["status"])
            self._validate_transition(str(state["status"]), next_status)
            allowed_updates = {
                "worker_pid",
                "child_pid",
                "returncode",
                "timed_out",
                "cancelled",
                "outcome_unknown_reason",
                "error",
                "stdout_truncated",
                "stderr_truncated",
            }
            unknown = set(updates) - allowed_updates
            if unknown:
                raise ValueError(f"unsupported durable job state fields: {sorted(unknown)}")
            next_state = {
                **state,
                **updates,
                "status": next_status,
                "updated_at_ms": self.now_ms(),
                "revision": int(state["revision"]) + 1,
            }
            self._validate_state(next_state, expected_job_id=job_id)
            self._write_json_atomic(paths["state"], next_state)
        return self._snapshot(spec, next_state)

    def claim_worker(self, job_id: str, *, worker_token: str, worker_pid: int) -> bool:
        if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
            raise ValueError("worker_pid must be a positive integer")
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            if str(spec["worker_token"]) != str(worker_token or ""):
                raise DurableJobConflictError(
                    "durable job worker token does not match"
                )
            state = self._read_state(paths["state"], expected_job_id=job_id)
            self._ensure_worker_token_matches(spec, state)
            if state["status"] != "queued":
                return False
            if paths["claim"].exists():
                return False
            payload = {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "worker_token": worker_token,
                "claim_id": uuid.uuid4().hex,
                "worker_pid": int(worker_pid),
                "claimed_at_ms": self.now_ms(),
            }
            self._write_json_atomic(paths["claim"], payload)
            return True

    def clear_stale_claim(
        self,
        job_id: str,
        *,
        worker_token: str,
        expected_claim: dict[str, Any],
    ) -> bool:
        """Compare-and-delete one exact stale claim.

        A worker token is stable for the lifetime of a job, so it cannot by
        itself fence retries.  ``claim_id`` is the per-attempt generation.  A
        supervisor may only delete the same generation that it proved stale;
        it must never delete a replacement claim created after that proof.
        """

        self._validate_claim(expected_claim, expected_job_id=job_id)
        if expected_claim.get("worker_token") != worker_token:
            return False
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            if spec["worker_token"] != worker_token:
                return False
            state = self._read_state(paths["state"], expected_job_id=job_id)
            self._ensure_worker_token_matches(spec, state)
            if state["status"] != "queued":
                return False
            claim = self._read_optional_json(paths["claim"], label="worker claim")
            if claim is None:
                return False
            self._validate_claim(claim, expected_job_id=job_id)
            if claim.get("worker_token") != worker_token:
                return False
            if claim != expected_claim:
                return False
            paths["claim"].unlink(missing_ok=True)
            self._fsync_dir(paths["dir"])
            return True

    def worker_claim_is_current(
        self,
        job_id: str,
        *,
        worker_token: str,
        worker_pid: int,
    ) -> bool:
        """Return whether this live wrapper still owns the current claim."""

        claim = self.read_claim(job_id)
        if claim is None:
            return False
        return (
            claim.get("worker_token") == worker_token
            and claim.get("worker_pid") == worker_pid
        )

    def read_claim(self, job_id: str) -> dict[str, Any] | None:
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            claim = self._read_optional_json(
                paths["claim"],
                label="worker claim",
            )
            if claim is None:
                return None
            self._validate_claim(claim, expected_job_id=job_id)
            if claim["worker_token"] != spec["worker_token"]:
                raise DurableJobStoreCorruptionError(
                    "durable job worker claim token does not match its immutable spec"
                )
            return claim

    def write_heartbeat(self, job_id: str, *, worker_token: str, worker_pid: int) -> bool:
        if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
            raise ValueError("worker_pid must be a positive integer")
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            if spec["worker_token"] != worker_token:
                raise DurableJobConflictError("durable job worker token does not match")
            state = self._read_state(paths["state"], expected_job_id=job_id)
            self._ensure_worker_token_matches(spec, state)
            if state["status"] in TERMINAL_JOB_STATUSES:
                return False
            claim = self._read_optional_json(paths["claim"], label="worker claim")
            if claim is None:
                raise DurableJobConflictError(
                    "durable job heartbeat requires a current worker claim"
                )
            self._validate_claim(claim, expected_job_id=job_id)
            if (
                claim["worker_token"] != worker_token
                or claim["worker_pid"] != worker_pid
            ):
                raise DurableJobConflictError(
                    "durable job heartbeat worker does not own the current claim"
                )
            if state["worker_pid"] is not None and state["worker_pid"] != worker_pid:
                raise DurableJobConflictError(
                    "durable job heartbeat worker does not match durable state"
                )
            previous = self._read_optional_json(
                paths["heartbeat"],
                label="heartbeat",
            )
            previous_seq = 0
            if previous is not None:
                self._validate_heartbeat(
                    previous,
                    expected_job_id=job_id,
                    spec=spec,
                )
                previous_seq = int(previous["heartbeat_seq"])
            self._write_json_atomic(
                paths["heartbeat"],
                {
                    "schema_version": JOB_SCHEMA_VERSION,
                    "job_id": job_id,
                    "worker_token": worker_token,
                    "worker_pid": int(worker_pid),
                    "heartbeat_seq": previous_seq + 1,
                    "updated_at_ms": self.now_ms(),
                },
                durable=False,
            )
        return True

    def read_heartbeat(self, job_id: str) -> dict[str, Any] | None:
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            heartbeat = self._read_optional_json(
                paths["heartbeat"],
                label="heartbeat",
            )
            if heartbeat is None:
                return None
            self._validate_heartbeat(
                heartbeat,
                expected_job_id=job_id,
                spec=spec,
            )
            return heartbeat

    def transition_to_outcome_unknown(
        self,
        job_id: str,
        *,
        execution_id: str,
        expected_revision: int,
        expected_status: str,
        reason: str,
        error: str,
        heartbeat_stale_before_ms: int | None = None,
        expected_heartbeat_seq: int | None = None,
        expected_claim_id: str | None = None,
    ) -> DurableJobSnapshot:
        """Persist a monotonic unknown outcome using revision and lease CAS.

        When a heartbeat cutoff is supplied, both heartbeat publication and
        this transition use the job lock.  A late-but-fresh heartbeat therefore
        wins the race and leaves the job nonterminal; once this transition
        wins, later heartbeats and state transitions cannot revive the job.
        """

        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        if expected_status not in JOB_STATUSES:
            raise ValueError("expected_status is invalid")
        reason = self._required_text(reason, "reason")
        error = self._required_text(error, "error")
        if heartbeat_stale_before_ms is not None and (
            isinstance(heartbeat_stale_before_ms, bool)
            or not isinstance(heartbeat_stale_before_ms, int)
            or heartbeat_stale_before_ms < 0
        ):
            raise ValueError(
                "heartbeat_stale_before_ms must be a non-negative integer"
            )
        if expected_heartbeat_seq is not None and (
            isinstance(expected_heartbeat_seq, bool)
            or not isinstance(expected_heartbeat_seq, int)
            or expected_heartbeat_seq < 0
        ):
            raise ValueError(
                "expected_heartbeat_seq must be a non-negative integer"
            )
        if expected_claim_id is not None and not re.fullmatch(
            r"[0-9a-f]{32}",
            expected_claim_id,
        ):
            raise ValueError("expected_claim_id must be a claim generation")

        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            self._ensure_owner(spec, execution_id)
            state = self._read_state(paths["state"], expected_job_id=job_id)
            self._ensure_worker_token_matches(spec, state)
            if state["status"] in TERMINAL_JOB_STATUSES:
                return self._snapshot(spec, state)
            if (
                int(state["revision"]) != expected_revision
                or str(state["status"]) != expected_status
            ):
                return self._snapshot(spec, state)

            current_claim: dict[str, Any] | None = None
            if expected_claim_id is not None:
                current_claim = self._read_optional_json(
                    paths["claim"],
                    label="worker claim",
                )
                if current_claim is None:
                    return self._snapshot(spec, state)
                self._validate_claim(current_claim, expected_job_id=job_id)
                if current_claim["claim_id"] != expected_claim_id:
                    return self._snapshot(spec, state)

            if (
                heartbeat_stale_before_ms is not None
                or expected_heartbeat_seq is not None
            ):
                heartbeat = self._read_optional_json(
                    paths["heartbeat"],
                    label="heartbeat",
                )
                current_heartbeat_seq = 0
                if heartbeat is not None:
                    self._validate_heartbeat(
                        heartbeat,
                        expected_job_id=job_id,
                        spec=spec,
                    )
                    current_heartbeat_seq = int(heartbeat["heartbeat_seq"])
                if (
                    expected_heartbeat_seq is not None
                    and current_heartbeat_seq != expected_heartbeat_seq
                ):
                    return self._snapshot(spec, state)
                if heartbeat is not None and heartbeat_stale_before_ms is not None:
                    expected_worker_pid = state["worker_pid"]
                    if expected_worker_pid is None and current_claim is not None:
                        expected_worker_pid = current_claim["worker_pid"]
                    if (
                        heartbeat["worker_pid"] == expected_worker_pid
                        and heartbeat["updated_at_ms"] >= heartbeat_stale_before_ms
                    ):
                        return self._snapshot(spec, state)

            self._validate_transition(str(state["status"]), "outcome_unknown")
            next_state = {
                **state,
                "status": "outcome_unknown",
                "outcome_unknown_reason": reason,
                "error": error,
                "returncode": None,
                "updated_at_ms": max(int(state["updated_at_ms"]), self.now_ms()),
                "revision": int(state["revision"]) + 1,
            }
            self._validate_state(next_state, expected_job_id=job_id)
            self._write_json_atomic(paths["state"], next_state)
            return self._snapshot(spec, next_state)

    def request_cancel(self, job_id: str, *, execution_id: str) -> None:
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            self._ensure_owner(spec, execution_id)
            self._write_json_atomic(
                paths["cancel"],
                {
                    "schema_version": JOB_SCHEMA_VERSION,
                    "job_id": job_id,
                    "execution_id": execution_id,
                    "requested_at_ms": self.now_ms(),
                },
            )

    def cancel_requested(self, job_id: str) -> bool:
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            marker = self._read_optional_json(
                paths["cancel"],
                label="cancel marker",
            )
            if marker is None:
                return False
            if (
                marker.get("schema_version") != JOB_SCHEMA_VERSION
                or marker.get("job_id") != job_id
                or marker.get("execution_id") != spec["execution_id"]
            ):
                raise DurableJobStoreCorruptionError(
                    "durable job cancel marker identity is invalid"
                )
            requested_at_ms = marker.get("requested_at_ms")
            if (
                isinstance(requested_at_ms, bool)
                or not isinstance(requested_at_ms, int)
                or requested_at_ms < 0
            ):
                raise DurableJobStoreCorruptionError(
                    "durable job cancel marker timestamp is invalid"
                )
            return True

    def consume_logs(
        self,
        job_id: str,
        *,
        execution_id: str,
        max_output_chars: int,
        final: bool = False,
    ) -> dict[str, Any]:
        if isinstance(max_output_chars, bool) or not isinstance(max_output_chars, int):
            raise TypeError("max_output_chars must be an integer")
        limit = max(0, max_output_chars)
        paths = self._paths(job_id)
        with _exclusive_lock(paths["lock"]):
            self._assert_store_identity()
            spec = self._read_spec(paths["spec"])
            self._ensure_owner(spec, execution_id)
            cursor = self._read_cursor(
                paths["cursor"],
                expected_job_id=job_id,
            )
            stdout = self._read_log_chunk(
                paths["stdout"],
                offset=int(cursor["stdout_offset"]),
                max_chars=limit,
                final=final,
            )
            stderr = self._read_log_chunk(
                paths["stderr"],
                offset=int(cursor["stderr_offset"]),
                max_chars=limit,
                final=final,
            )
            next_cursor = {
                "schema_version": JOB_SCHEMA_VERSION,
                "job_id": job_id,
                "stdout_offset": stdout["next_offset"],
                "stderr_offset": stderr["next_offset"],
                "revision": int(cursor["revision"]) + 1,
            }
            self._write_json_atomic(paths["cursor"], next_cursor)
        return {
            "stdout": stdout["text"],
            "stderr": stderr["text"],
            "stdout_start_offset": stdout["start_offset"],
            "stderr_start_offset": stderr["start_offset"],
            "stdout_offset": next_cursor["stdout_offset"],
            "stderr_offset": next_cursor["stderr_offset"],
            "stdout_available": stdout["available"],
            "stderr_available": stderr["available"],
            "offset_unit": "utf8_bytes",
            "truncated": stdout["has_more"] or stderr["has_more"],
        }

    def list_execution(self, execution_id: str) -> list[DurableJobSnapshot]:
        execution_id = self._required_text(execution_id, "execution_id")
        self._assert_store_identity()
        snapshots: list[DurableJobSnapshot] = []
        for job_dir in sorted(self.jobs_dir.glob("job_*")):
            if not job_dir.is_dir() or not _JOB_ID_RE.fullmatch(job_dir.name):
                continue
            paths = self._paths(job_dir.name)
            with _exclusive_lock(paths["lock"]):
                self._assert_store_identity()
                # A crash may leave the directory (and an atomic-write temp)
                # before the immutable spec exists.  It has no owner and no
                # launchable intent, so it must not poison unrelated reattach.
                if not paths["spec"].exists():
                    continue
                spec = self._read_spec(paths["spec"])
                try:
                    self._ensure_owner(spec, execution_id)
                except DurableJobOwnershipError:
                    continue
                self._ensure_initial_files(paths, spec)
                state = self._read_state(
                    paths["state"],
                    expected_job_id=job_dir.name,
                )
                self._ensure_worker_token_matches(spec, state)
                snapshots.append(self._snapshot(spec, state))
        return snapshots

    def paths_for_worker(self, job_id: str) -> dict[str, Path]:
        return dict(self._paths(job_id))

    def _paths(self, job_id: str, *, create: bool = False) -> dict[str, Path]:
        self._assert_store_identity()
        if not isinstance(job_id, str) or not _JOB_ID_RE.fullmatch(job_id):
            raise DurableJobNotFoundError(f"durable job not found: {job_id}")
        job_dir = self.jobs_dir / job_id
        if create:
            job_dir.mkdir(parents=False, exist_ok=True, mode=0o700)
            if os.name != "nt":
                os.chmod(job_dir, 0o700)
        return {
            "dir": job_dir,
            "lock": job_dir / ".job.lock",
            "spec": job_dir / "spec.json",
            "state": job_dir / "state.json",
            "cursor": job_dir / "cursor.json",
            "claim": job_dir / "worker-claim.json",
            "heartbeat": job_dir / "heartbeat.json",
            "cancel": job_dir / "cancel-request.json",
            "stdout": job_dir / "stdout.log",
            "stderr": job_dir / "stderr.log",
        }

    def _ensure_initial_files(
        self,
        paths: dict[str, Path],
        spec: dict[str, Any],
    ) -> None:
        """Finish only a provably pre-launch reservation after a crash."""

        job_id = str(spec["job_id"])
        if not paths["state"].exists():
            launch_artifacts = (
                "claim",
                "heartbeat",
                "cancel",
                "stdout",
                "stderr",
            )
            if any(paths[name].exists() for name in launch_artifacts):
                raise DurableJobStoreCorruptionError(
                    "durable job state is missing after launch artifacts were created"
                )
            created_at_ms = int(spec["created_at_ms"])
            self._write_json_atomic(
                paths["state"],
                {
                    "schema_version": JOB_SCHEMA_VERSION,
                    "job_id": job_id,
                    "status": "queued",
                    "created_at_ms": created_at_ms,
                    "updated_at_ms": created_at_ms,
                    "revision": 0,
                    "worker_token": str(spec["worker_token"]),
                    "worker_pid": None,
                    "child_pid": None,
                    "returncode": None,
                    "timed_out": False,
                    "cancelled": False,
                    "outcome_unknown_reason": "",
                    "error": "",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                },
            )
        if not paths["cursor"].exists():
            state = self._read_state(
                paths["state"],
                expected_job_id=job_id,
            )
            launch_artifacts = (
                "claim",
                "heartbeat",
                "cancel",
                "stdout",
                "stderr",
            )
            if (
                state["status"] != "queued"
                or int(state["revision"]) != 0
                or any(paths[name].exists() for name in launch_artifacts)
            ):
                raise DurableJobStoreCorruptionError(
                    "durable job cursor is missing after launch could have begun"
                )
            self._write_json_atomic(
                paths["cursor"],
                {
                    "schema_version": JOB_SCHEMA_VERSION,
                    "job_id": job_id,
                    "stdout_offset": 0,
                    "stderr_offset": 0,
                    "revision": 0,
                },
            )

    def _read_spec(self, path: Path) -> dict[str, Any]:
        raw = self._read_json(path, label="job spec")
        required = {
            "schema_version",
            "job_id",
            "store_id",
            "execution_id",
            "adapter",
            "argv",
            "cwd",
            "timeout_ms",
            "idempotency_digest",
            "intent_digest",
            "environment_digest",
            "max_log_bytes",
            "request_digest",
            "worker_token",
            "created_at_ms",
        }
        if not required.issubset(raw):
            raise DurableJobStoreCorruptionError("durable job spec is missing required fields")
        if raw["schema_version"] != JOB_SCHEMA_VERSION:
            raise DurableJobStoreCorruptionError("unsupported durable job spec schema version")
        if raw["store_id"] != self.store_id:
            raise DurableJobStoreCorruptionError(
                "durable job spec belongs to a different store identity"
            )
        if not _JOB_ID_RE.fullmatch(str(raw["job_id"])) or path.parent.name != raw["job_id"]:
            raise DurableJobStoreCorruptionError("durable job spec id is invalid")
        if not isinstance(raw["argv"], list) or not raw["argv"] or any(
            not isinstance(item, str) for item in raw["argv"]
        ):
            raise DurableJobStoreCorruptionError("durable job spec argv is invalid")
        for field in (
            "execution_id",
            "adapter",
            "cwd",
            "idempotency_digest",
            "intent_digest",
            "environment_digest",
            "request_digest",
            "worker_token",
        ):
            if not isinstance(raw[field], str) or not raw[field]:
                raise DurableJobStoreCorruptionError(f"durable job spec {field} is invalid")
        for field in ("timeout_ms", "max_log_bytes", "created_at_ms"):
            if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 0:
                raise DurableJobStoreCorruptionError(f"durable job spec {field} is invalid")
        if raw["timeout_ms"] <= 0 or raw["max_log_bytes"] <= 0:
            raise DurableJobStoreCorruptionError(
                "durable job spec timing or log limit is invalid"
            )
        if not Path(raw["cwd"]).is_absolute():
            raise DurableJobStoreCorruptionError("durable job spec cwd is not absolute")
        for field in (
            "idempotency_digest",
            "intent_digest",
            "environment_digest",
            "request_digest",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", raw[field]):
                raise DurableJobStoreCorruptionError(
                    f"durable job spec {field} is not a SHA-256 digest"
                )
        if not re.fullmatch(r"[0-9a-f]{32}", raw["worker_token"]):
            raise DurableJobStoreCorruptionError(
                "durable job spec worker token is invalid"
            )
        request = {
            "store_id": raw["store_id"],
            "execution_id": raw["execution_id"],
            "adapter": raw["adapter"],
            "argv": raw["argv"],
            "cwd": raw["cwd"],
            "timeout_ms": raw["timeout_ms"],
            "idempotency_digest": raw["idempotency_digest"],
            "intent_digest": raw["intent_digest"],
            "environment_digest": raw["environment_digest"],
            "max_log_bytes": raw["max_log_bytes"],
        }
        if _digest(request) != raw["request_digest"]:
            raise DurableJobStoreCorruptionError(
                "durable job immutable launch spec digest does not match"
            )
        return raw

    def _read_state(self, path: Path, *, expected_job_id: str) -> dict[str, Any]:
        raw = self._read_json(path, label="job state")
        self._validate_state(raw, expected_job_id=expected_job_id)
        return raw

    def _validate_state(self, raw: dict[str, Any], *, expected_job_id: str) -> None:
        required = {
            "schema_version",
            "job_id",
            "status",
            "created_at_ms",
            "updated_at_ms",
            "revision",
            "worker_token",
            "worker_pid",
            "child_pid",
            "returncode",
            "timed_out",
            "cancelled",
            "outcome_unknown_reason",
            "error",
            "stdout_truncated",
            "stderr_truncated",
        }
        if not required.issubset(raw):
            raise DurableJobStoreCorruptionError("durable job state is missing required fields")
        if raw["schema_version"] != JOB_SCHEMA_VERSION or raw["job_id"] != expected_job_id:
            raise DurableJobStoreCorruptionError("durable job state identity is invalid")
        if raw["status"] not in JOB_STATUSES:
            raise DurableJobStoreCorruptionError("durable job state status is invalid")
        for field in ("created_at_ms", "updated_at_ms", "revision"):
            if isinstance(raw[field], bool) or not isinstance(raw[field], int) or raw[field] < 0:
                raise DurableJobStoreCorruptionError(f"durable job state {field} is invalid")
        if raw["updated_at_ms"] < raw["created_at_ms"]:
            raise DurableJobStoreCorruptionError(
                "durable job state update timestamp predates creation"
            )
        for field in ("worker_pid", "child_pid", "returncode"):
            value = raw[field]
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise DurableJobStoreCorruptionError(f"durable job state {field} is invalid")
            if field in {"worker_pid", "child_pid"} and value is not None and value <= 0:
                raise DurableJobStoreCorruptionError(f"durable job state {field} is invalid")
        for field in ("timed_out", "cancelled", "stdout_truncated", "stderr_truncated"):
            if not isinstance(raw[field], bool):
                raise DurableJobStoreCorruptionError(f"durable job state {field} is invalid")
        for field in ("worker_token", "outcome_unknown_reason", "error"):
            if not isinstance(raw[field], str):
                raise DurableJobStoreCorruptionError(f"durable job state {field} is invalid")

    def _read_cursor(self, path: Path, *, expected_job_id: str) -> dict[str, Any]:
        raw = self._read_json(path, label="job log cursor")
        if raw.get("schema_version") != JOB_SCHEMA_VERSION:
            raise DurableJobStoreCorruptionError("unsupported durable job cursor schema version")
        if raw.get("job_id") != expected_job_id:
            raise DurableJobStoreCorruptionError("durable job cursor identity is invalid")
        for field in ("stdout_offset", "stderr_offset", "revision"):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DurableJobStoreCorruptionError(f"durable job cursor {field} is invalid")
        return raw

    @staticmethod
    def _validate_claim(raw: dict[str, Any], *, expected_job_id: str) -> None:
        if (
            raw.get("schema_version") != JOB_SCHEMA_VERSION
            or raw.get("job_id") != expected_job_id
        ):
            raise DurableJobStoreCorruptionError(
                "durable job worker claim identity is invalid"
            )
        worker_token = raw.get("worker_token")
        if not isinstance(worker_token, str) or not worker_token:
            raise DurableJobStoreCorruptionError(
                "durable job worker claim token is invalid"
            )
        claim_id = raw.get("claim_id")
        if not isinstance(claim_id, str) or not re.fullmatch(
            r"[0-9a-f]{32}",
            claim_id,
        ):
            raise DurableJobStoreCorruptionError(
                "durable job worker claim generation is invalid"
            )
        worker_pid = raw.get("worker_pid")
        if (
            isinstance(worker_pid, bool)
            or not isinstance(worker_pid, int)
            or worker_pid <= 0
        ):
            raise DurableJobStoreCorruptionError(
                "durable job worker claim worker_pid is invalid"
            )
        claimed_at_ms = raw.get("claimed_at_ms")
        if (
            isinstance(claimed_at_ms, bool)
            or not isinstance(claimed_at_ms, int)
            or claimed_at_ms < 0
        ):
            raise DurableJobStoreCorruptionError(
                "durable job worker claim claimed_at_ms is invalid"
            )

    @staticmethod
    def _validate_heartbeat(
        raw: dict[str, Any],
        *,
        expected_job_id: str,
        spec: dict[str, Any],
    ) -> None:
        required = {
            "schema_version",
            "job_id",
            "worker_token",
            "worker_pid",
            "heartbeat_seq",
            "updated_at_ms",
        }
        if not required.issubset(raw):
            raise DurableJobStoreCorruptionError(
                "durable job heartbeat is missing fields"
            )
        if (
            raw["schema_version"] != JOB_SCHEMA_VERSION
            or raw["job_id"] != expected_job_id
        ):
            raise DurableJobStoreCorruptionError(
                "durable job heartbeat identity is invalid"
            )
        if not isinstance(raw["worker_token"], str) or not raw["worker_token"]:
            raise DurableJobStoreCorruptionError(
                "durable job heartbeat token is invalid"
            )
        for field in ("worker_pid", "heartbeat_seq", "updated_at_ms"):
            value = raw[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DurableJobStoreCorruptionError(
                    f"durable job heartbeat {field} is invalid"
                )
        if raw["worker_pid"] <= 0 or raw["heartbeat_seq"] <= 0:
            raise DurableJobStoreCorruptionError(
                "durable job heartbeat identity or sequence is invalid"
            )
        if raw["worker_token"] != spec["worker_token"]:
            raise DurableJobStoreCorruptionError(
                "durable job heartbeat token does not match its immutable spec"
            )

    @staticmethod
    def _validate_transition(current: str, next_status: str) -> None:
        allowed = {
            "queued": {"queued", "starting", "cancelled", "failed", "outcome_unknown"},
            "starting": {
                "starting",
                "running",
                "cancelled",
                "failed",
                "timed_out",
                "outcome_unknown",
            },
            "running": {
                "running",
                "completed",
                "failed",
                "timed_out",
                "cancelled",
                "outcome_unknown",
            },
        }
        if current in TERMINAL_JOB_STATUSES:
            if next_status != current:
                raise DurableJobConflictError(
                    f"terminal durable job cannot transition from {current} to {next_status}"
                )
            return
        if next_status not in allowed.get(current, set()):
            raise DurableJobConflictError(
                f"invalid durable job transition from {current} to {next_status}"
            )

    @staticmethod
    def _snapshot(spec: dict[str, Any], state: dict[str, Any]) -> DurableJobSnapshot:
        return DurableJobSnapshot(
            job_id=str(spec["job_id"]),
            execution_id=str(spec["execution_id"]),
            adapter=str(spec["adapter"]),
            status=str(state["status"]),
            intent_digest=str(spec["intent_digest"]),
            environment_digest=str(spec["environment_digest"]),
            created_at_ms=int(state["created_at_ms"]),
            updated_at_ms=int(state["updated_at_ms"]),
            revision=int(state["revision"]),
            worker_token=str(state["worker_token"]),
            worker_pid=state["worker_pid"],
            child_pid=state["child_pid"],
            returncode=state["returncode"],
            timed_out=bool(state["timed_out"]),
            cancelled=bool(state["cancelled"]),
            outcome_unknown_reason=str(state["outcome_unknown_reason"]),
            error=str(state["error"]),
            stdout_truncated=bool(state["stdout_truncated"]),
            stderr_truncated=bool(state["stderr_truncated"]),
        )

    @staticmethod
    def _ensure_owner(spec: dict[str, Any], execution_id: str) -> None:
        if str(spec["execution_id"]) != str(execution_id or ""):
            raise DurableJobOwnershipError(f"durable job not found: {spec['job_id']}")

    @staticmethod
    def _ensure_worker_token_matches(
        spec: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        if state["worker_token"] != spec["worker_token"]:
            raise DurableJobStoreCorruptionError(
                "durable job state worker token does not match its immutable spec"
            )
        if state["created_at_ms"] != spec["created_at_ms"]:
            raise DurableJobStoreCorruptionError(
                "durable job state creation timestamp does not match its immutable spec"
            )

    @staticmethod
    def _required_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")
        return value.strip()

    def _read_json(self, path: Path, *, label: str) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DurableJobNotFoundError(f"durable job file not found: {path.name}") from exc
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DurableJobStoreCorruptionError(f"cannot decode {label}: {path.name}") from exc
        if not isinstance(raw, dict):
            raise DurableJobStoreCorruptionError(f"{label} must be a JSON object")
        return raw

    def _read_store_manifest(self) -> dict[str, Any]:
        raw = self._read_json(self._manifest_path, label="job store manifest")
        if raw.get("schema_version") != STORE_MANIFEST_SCHEMA_VERSION:
            raise DurableJobStoreCorruptionError(
                "unsupported durable job store manifest schema version"
            )
        store_id = raw.get("store_id")
        if not isinstance(store_id, str) or not _STORE_ID_RE.fullmatch(store_id):
            raise DurableJobStoreCorruptionError(
                "durable job store manifest id is invalid"
            )
        created_at_ms = raw.get("created_at_ms")
        if (
            isinstance(created_at_ms, bool)
            or not isinstance(created_at_ms, int)
            or created_at_ms < 0
        ):
            raise DurableJobStoreCorruptionError(
                "durable job store manifest timestamp is invalid"
            )
        return raw

    def _assert_store_identity(self) -> None:
        try:
            manifest = self._read_store_manifest()
        except DurableJobNotFoundError as exc:
            raise DurableJobStoreCorruptionError(
                "durable job store manifest is missing"
            ) from exc
        if manifest["store_id"] != self.store_id:
            raise DurableJobStoreCorruptionError(
                "durable job store identity changed while the store was open"
            )

    def _has_unbound_job_data(self) -> bool:
        legacy_jobs_dir = self.base_dir / "jobs"
        if legacy_jobs_dir.exists() and any(legacy_jobs_dir.iterdir()):
            return True
        if not self.stores_dir.exists():
            return False
        for path in self.stores_dir.rglob("*"):
            if path.is_file() or (
                path.is_dir() and _JOB_ID_RE.fullmatch(path.name)
            ):
                return True
        return False

    def _read_optional_json(self, path: Path, *, label: str) -> dict[str, Any] | None:
        if not path.exists():
            return None
        return self._read_json(path, label=label)

    @staticmethod
    def _read_log_chunk(
        path: Path,
        *,
        offset: int,
        max_chars: int,
        final: bool,
    ) -> dict[str, Any]:
        """Read one bounded UTF-8 chunk without rescanning the whole log.

        Cursor offsets are bytes.  The worker writes valid UTF-8, and the
        incremental decoder holds a trailing partial code point until the next
        poll.  A terminal read may flush an incomplete final sequence as the
        Unicode replacement character.
        """

        try:
            available = path.stat().st_size
        except FileNotFoundError:
            available = 0
        if offset > available:
            raise DurableJobStoreCorruptionError(
                f"durable job log {path.name} is shorter than its persisted cursor"
            )
        start_offset = max(0, offset)
        if max_chars <= 0 or start_offset >= available:
            return {
                "text": "",
                "start_offset": start_offset,
                "next_offset": start_offset,
                "available": available,
                "has_more": start_offset < available,
            }

        # Four bytes are enough to complete any UTF-8 code point.  Reading up
        # to 4x the character budget lets us return the requested number of
        # characters without splitting a multibyte sequence.
        read_limit = max(4, max_chars * 4)
        try:
            with path.open("rb") as handle:
                handle.seek(start_offset)
                payload = handle.read(read_limit)
        except FileNotFoundError:
            return {
                "text": "",
                "start_offset": 0,
                "next_offset": 0,
                "available": 0,
                "has_more": False,
            }

        reached_snapshot_end = start_offset + len(payload) >= available
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        decoded = decoder.decode(
            payload,
            final=bool(final and reached_snapshot_end),
        )
        buffered, _ = decoder.getstate()
        complete_payload_length = len(payload) - len(buffered)
        text = decoded[:max_chars]
        if len(decoded) > max_chars:
            # Durable logs are normalized to valid UTF-8 by the worker, so
            # re-encoding the returned prefix gives its exact byte length.
            consumed = len(text.encode("utf-8"))
        else:
            consumed = complete_payload_length
        next_offset = start_offset + consumed
        return {
            "text": text,
            "start_offset": start_offset,
            "next_offset": next_offset,
            "available": available,
            "has_more": next_offset < available,
        }

    def _write_json_atomic(
        self,
        path: Path,
        payload: dict[str, Any],
        *,
        durable: bool = True,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        encoded = (_canonical_json(payload) + "\n").encode("utf-8")
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb", closefd=True) as handle:
                handle.write(encoded)
                handle.flush()
                if durable:
                    os.fsync(handle.fileno())
            os.replace(temp_path, path)
            if os.name != "nt":
                os.chmod(path, 0o600)
            if durable:
                self._fsync_dir(path.parent)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                # A successful atomic replace already consumed the temporary path.
                pass

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        if os.name == "nt":  # pragma: no cover - directory fsync is POSIX-only
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["STORE_MANIFEST_SCHEMA_VERSION", "JsonFileJobStore"]
