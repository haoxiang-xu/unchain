from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .environment import JobEnvironmentProfile
from .models import (
    TERMINAL_JOB_STATUSES,
    DurableJobConflictError,
    DurableJobSnapshot,
)
from .store import JsonFileJobStore


@dataclass(frozen=True)
class _SuspectObservation:
    signature: tuple[Any, ...]
    first_seen_monotonic: float


class DurableJobResult(dict[str, Any]):
    """A JSON-friendly result that also exposes fields as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ProcessJobSupervisor:
    """Launch and observe detached local-process jobs through durable files.

    The supervisor deliberately owns no ``Popen`` handles for user commands.
    A short-lived, independently detached worker owns each command instead, so
    closing or crashing the runtime that started a job does not kill the job.

    Every public lookup takes an ``execution_id``.  The job store applies that
    ownership check before logs are read or cancellation is requested.
    """

    def __init__(
        self,
        store: JsonFileJobStore,
        *,
        python_executable: str | os.PathLike[str] | None = None,
        worker_command_prefix: Sequence[str | os.PathLike[str]] | None = None,
        worker_environment_overlay: Mapping[str, str] | None = None,
        heartbeat_stale_ms: int | None = None,
        launch_grace_ms: int | None = None,
        poll_interval_s: float | None = None,
        default_max_log_bytes: int = 50 * 1024 * 1024,
        poll_interval_ms: int | None = None,
        heartbeat_interval_ms: int | None = None,
        heartbeat_stale_after_ms: int | None = None,
        cancel_grace_ms: int = 1_000,
        startup_timeout_ms: int | None = None,
        suspect_grace_ms: int | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        environment: Mapping[str, str] | JobEnvironmentProfile | None = None,
    ) -> None:
        if not isinstance(store, JsonFileJobStore):
            raise TypeError("store must be a JsonFileJobStore")
        self.store = store
        self.python_executable = str(python_executable or sys.executable)
        if python_executable is not None and worker_command_prefix is not None:
            raise ValueError(
                "python_executable and worker_command_prefix are mutually exclusive"
            )
        self.worker_command_prefix = self._normalize_worker_command_prefix(
            worker_command_prefix
        )
        self.worker_environment_overlay = self._normalize_worker_environment_overlay(
            worker_environment_overlay
        )
        self.environment_profile = (
            environment
            if isinstance(environment, JobEnvironmentProfile)
            else JobEnvironmentProfile.capture(environment)
        )
        if heartbeat_stale_ms is not None and heartbeat_stale_after_ms is not None:
            raise ValueError(
                "provide only one of heartbeat_stale_ms or heartbeat_stale_after_ms"
            )
        resolved_heartbeat_stale_ms = (
            heartbeat_stale_ms
            if heartbeat_stale_ms is not None
            else (
                heartbeat_stale_after_ms
                if heartbeat_stale_after_ms is not None
                else 5_000
            )
        )
        self.heartbeat_stale_ms = self._positive_int(
            resolved_heartbeat_stale_ms,
            "heartbeat_stale_ms",
        )
        if launch_grace_ms is not None and startup_timeout_ms is not None:
            raise ValueError("provide only one of launch_grace_ms or startup_timeout_ms")
        resolved_launch_grace_ms = (
            launch_grace_ms
            if launch_grace_ms is not None
            else (startup_timeout_ms if startup_timeout_ms is not None else 2_000)
        )
        self.launch_grace_ms = self._positive_int(
            resolved_launch_grace_ms,
            "launch_grace_ms",
        )
        if poll_interval_s is not None and poll_interval_ms is not None:
            raise ValueError("provide only one of poll_interval_s or poll_interval_ms")
        resolved_poll_interval_s: float = (
            float(poll_interval_s)
            if poll_interval_s is not None
            else (
                self._positive_int(poll_interval_ms, "poll_interval_ms") / 1000.0
                if poll_interval_ms is not None
                else 0.05
            )
        )
        if isinstance(resolved_poll_interval_s, bool) or not isinstance(
            resolved_poll_interval_s,
            (int, float),
        ):
            raise TypeError("poll_interval_s must be a positive number")
        if resolved_poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be a positive number")
        self.poll_interval_s = float(resolved_poll_interval_s)
        resolved_heartbeat_interval_ms = (
            heartbeat_interval_ms
            if heartbeat_interval_ms is not None
            else min(1_000, max(1, self.heartbeat_stale_ms // 3))
        )
        self.heartbeat_interval_ms = self._positive_int(
            resolved_heartbeat_interval_ms,
            "heartbeat_interval_ms",
        )
        if self.heartbeat_interval_ms >= self.heartbeat_stale_ms:
            raise ValueError(
                "heartbeat_interval_ms must be smaller than the stale-heartbeat window"
            )
        self.suspect_grace_ms = self._positive_int(
            (
                suspect_grace_ms
                if suspect_grace_ms is not None
                else self.heartbeat_stale_ms
            ),
            "suspect_grace_ms",
        )
        if monotonic_clock is not None and not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        self._monotonic_clock = monotonic_clock or time.monotonic
        self.cancel_grace_ms = self._positive_int(cancel_grace_ms, "cancel_grace_ms")
        self.default_max_log_bytes = self._positive_int(
            default_max_log_bytes,
            "default_max_log_bytes",
        )
        self._closed = False
        self._spawn_lock = threading.Lock()
        self._worker_wrappers: dict[str, subprocess.Popen[Any]] = {}
        self._last_spawn_at: dict[str, float] = {}
        self._suspect_lock = threading.Lock()
        self._suspect_observations: dict[str, _SuspectObservation] = {}

    def start(
        self,
        *,
        execution_id: str,
        idempotency_key: str,
        argv: list[str],
        cwd: str,
        timeout_ms: int,
        intent_digest: str | None = None,
        adapter: str = "local_process",
        max_log_bytes: int | None = None,
    ) -> DurableJobSnapshot:
        """Reserve one stable job id and ensure its worker has been launched."""

        if self._closed:
            raise RuntimeError("process job supervisor is closed")
        resolved_max_log_bytes = (
            self.default_max_log_bytes
            if max_log_bytes is None
            else self._positive_int(max_log_bytes, "max_log_bytes")
        )
        resolved_intent_digest = str(intent_digest or "").strip()
        if not resolved_intent_digest:
            encoded_intent = json.dumps(
                {
                    "adapter": adapter,
                    "argv": argv,
                    "cwd": str(Path(cwd).resolve()),
                    "timeout_ms": timeout_ms,
                    "environment_digest": self.environment_profile.digest,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            resolved_intent_digest = hashlib.sha256(encoded_intent).hexdigest()
        snapshot, _created = self.store.reserve_process_job(
            execution_id=execution_id,
            adapter=adapter,
            argv=argv,
            cwd=cwd,
            timeout_ms=timeout_ms,
            idempotency_key=idempotency_key,
            intent_digest=resolved_intent_digest,
            max_log_bytes=resolved_max_log_bytes,
            environment_digest=self.environment_profile.digest,
        )
        return self._reconcile(snapshot, allow_launch=True)

    def inspect(
        self,
        job_id: str,
        *,
        execution_id: str,
    ) -> DurableJobSnapshot:
        snapshot = self.store.load(job_id, execution_id=execution_id)
        return self._reconcile(snapshot, allow_launch=not self._closed)

    def poll(
        self,
        job_id: str,
        *,
        execution_id: str,
        max_output_chars: int = 20_000,
    ) -> DurableJobResult:
        snapshot = self.inspect(job_id, execution_id=execution_id)
        logs = self.store.consume_logs(
            job_id,
            execution_id=execution_id,
            max_output_chars=max_output_chars,
            final=self._logs_are_final(snapshot),
        )
        return self._result(snapshot, action="poll", logs=logs)

    def wait(
        self,
        job_id: str,
        *,
        execution_id: str,
        timeout_ms: int = 120_000,
        max_output_chars: int = 20_000,
    ) -> DurableJobResult:
        resolved_timeout_ms = self._non_negative_int(timeout_ms, "timeout_ms")
        deadline = time.monotonic() + (resolved_timeout_ms / 1000.0)
        snapshot = self.inspect(job_id, execution_id=execution_id)
        while not snapshot.completed and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            time.sleep(min(self.poll_interval_s, max(0.0, remaining)))
            snapshot = self.inspect(job_id, execution_id=execution_id)

        logs = self.store.consume_logs(
            job_id,
            execution_id=execution_id,
            max_output_chars=max_output_chars,
            final=self._logs_are_final(snapshot),
        )
        result = self._result(snapshot, action="wait", logs=logs)
        result["wait_timeout_ms"] = resolved_timeout_ms
        result["wait_timed_out"] = not snapshot.completed
        return result

    def cancel(
        self,
        job_id: str,
        *,
        execution_id: str,
        wait_timeout_ms: int | None = None,
        timeout_ms: int | None = None,
        max_output_chars: int = 20_000,
    ) -> DurableJobResult:
        """Request cooperative cancellation without signalling an unverified PID.

        ``timeout_ms`` is accepted as a compatibility alias.  It controls only
        how long this caller waits for the worker's durable terminal receipt;
        the worker remains responsible for terminating its own live child.
        """

        if wait_timeout_ms is not None and timeout_ms is not None:
            raise ValueError("provide only one of wait_timeout_ms or timeout_ms")
        requested_wait_ms = (
            wait_timeout_ms
            if wait_timeout_ms is not None
            else (timeout_ms if timeout_ms is not None else 2_000)
        )
        resolved_wait_ms = self._non_negative_int(requested_wait_ms, "wait_timeout_ms")

        # Read without the normal queued-job recovery launch.  The cancel
        # marker must reach durable storage before any replacement worker is
        # allowed to start, otherwise a queued command could briefly run while
        # the caller is trying to cancel it.
        snapshot = self.store.load(job_id, execution_id=execution_id)
        snapshot = self._reconcile(snapshot, allow_launch=False)
        cancel_requested = False
        if snapshot.status not in TERMINAL_JOB_STATUSES:
            self.store.request_cancel(job_id, execution_id=execution_id)
            cancel_requested = True
            # A queued reservation with no live worker still needs a worker to
            # record the cancellation as a durable terminal outcome.
            snapshot = self._reconcile(snapshot, allow_launch=not self._closed)

        deadline = time.monotonic() + (resolved_wait_ms / 1000.0)
        while not snapshot.completed and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            time.sleep(min(self.poll_interval_s, max(0.0, remaining)))
            snapshot = self.inspect(job_id, execution_id=execution_id)

        logs = self.store.consume_logs(
            job_id,
            execution_id=execution_id,
            max_output_chars=max_output_chars,
            final=self._logs_are_final(snapshot),
        )
        result = self._result(snapshot, action="cancel", logs=logs)
        result["cancel_requested"] = cancel_requested
        result["cancel_wait_timeout_ms"] = resolved_wait_ms
        result["cancel_wait_timed_out"] = not snapshot.completed
        return result

    def reattach(self, execution_id: str) -> list[DurableJobSnapshot]:
        """Return all jobs owned by an execution after reconciling liveness."""

        snapshots: list[DurableJobSnapshot] = []
        for snapshot in self.store.list_execution(execution_id):
            snapshots.append(
                self._reconcile(snapshot, allow_launch=not self._closed)
            )
        return snapshots

    def contains(self, job_id: str) -> bool:
        return self.store.contains(job_id)

    def close(self) -> None:
        """Detach this supervisor; intentionally do not cancel running jobs."""

        self._closed = True
        with self._suspect_lock:
            self._suspect_observations.clear()

    def _reconcile(
        self,
        snapshot: DurableJobSnapshot,
        *,
        allow_launch: bool,
    ) -> DurableJobSnapshot:
        if snapshot.completed:
            self._clear_suspicion(snapshot.job_id)
            return snapshot

        heartbeat = self.store.read_heartbeat(snapshot.job_id)
        heartbeat_fresh = self._heartbeat_is_fresh(snapshot, heartbeat)
        if snapshot.status in {"starting", "running"}:
            if heartbeat_fresh:
                self._clear_suspicion(snapshot.job_id)
                return snapshot
            # Give a newly claimed worker a bounded interval to publish its
            # first heartbeat.  After that, absence is an ambiguous outcome:
            # no new worker is launched because the command may still exist.
            if self.store.now_ms() - snapshot.updated_at_ms <= self.launch_grace_ms:
                self._clear_suspicion(snapshot.job_id)
                return snapshot
            if not self._suspect_grace_elapsed(
                snapshot,
                heartbeat=heartbeat,
                reason="worker_heartbeat_lost",
            ):
                return snapshot
            return self._persist_outcome_unknown(
                snapshot,
                reason="worker_heartbeat_lost",
                recheck_heartbeat=True,
                heartbeat=heartbeat,
            )

        if snapshot.status != "queued":
            return snapshot

        claim = self.store.read_claim(snapshot.job_id)
        if claim is None:
            self._clear_suspicion(snapshot.job_id)
            if allow_launch:
                self._spawn_worker(snapshot)
                return self.store.load(
                    snapshot.job_id,
                    execution_id=snapshot.execution_id,
                )
            return snapshot

        if not self._claim_matches(snapshot, claim):
            return self._persist_outcome_unknown(
                snapshot,
                reason="worker_claim_invalid",
                recheck_heartbeat=False,
                heartbeat=heartbeat,
            )
        if heartbeat_fresh:
            self._clear_suspicion(snapshot.job_id)
            return snapshot

        claimed_at_ms = claim.get("claimed_at_ms")
        if not isinstance(claimed_at_ms, int) or isinstance(claimed_at_ms, bool):
            return self._persist_outcome_unknown(
                snapshot,
                reason="worker_claim_invalid",
                recheck_heartbeat=False,
                heartbeat=heartbeat,
            )
        if self.store.now_ms() - claimed_at_ms <= self.launch_grace_ms:
            self._clear_suspicion(snapshot.job_id)
            return snapshot

        worker_pid = claim.get("worker_pid")
        if not isinstance(worker_pid, int) or isinstance(worker_pid, bool) or worker_pid <= 0:
            return self._persist_outcome_unknown(
                snapshot,
                reason="worker_claim_invalid",
                recheck_heartbeat=False,
                heartbeat=heartbeat,
            )
        if self._pid_may_be_alive(worker_pid):
            claim_id = str(claim.get("claim_id") or "")
            if not self._suspect_grace_elapsed(
                snapshot,
                heartbeat=heartbeat,
                reason="worker_heartbeat_lost_before_launch",
                claim_id=claim_id,
            ):
                return snapshot
            return self._persist_outcome_unknown(
                snapshot,
                reason="worker_heartbeat_lost_before_launch",
                recheck_heartbeat=True,
                heartbeat=heartbeat,
                expected_claim_id=claim_id,
            )

        # The worker protocol writes STARTING before it can launch a child.
        # Therefore a dead worker whose durable state is still QUEUED can be
        # retried without duplicating the user command.
        self._clear_suspicion(snapshot.job_id)
        if allow_launch and self.store.clear_stale_claim(
            snapshot.job_id,
            worker_token=snapshot.worker_token,
            expected_claim=claim,
        ):
            self._spawn_worker(snapshot)
            return self.store.load(
                snapshot.job_id,
                execution_id=snapshot.execution_id,
            )
        return snapshot

    def _spawn_worker(self, snapshot: DurableJobSnapshot) -> None:
        if self._closed:
            return
        with self._spawn_lock:
            if snapshot.environment_digest != self.environment_profile.digest:
                # A mismatched observer must not terminalize shared state: a
                # matching supervisor may already be between Popen(wrapper)
                # and the worker's durable claim.  Fail this caller closed and
                # leave QUEUED available to the matching execution profile.
                raise DurableJobConflictError(
                    "queued durable job environment profile does not match "
                    "the current supervisor"
                )
            existing = self._worker_wrappers.get(snapshot.job_id)
            if existing is not None and existing.poll() is None:
                return
            now = time.monotonic()
            last_spawn_at = self._last_spawn_at.get(snapshot.job_id, 0.0)
            retry_cooldown_s = max(
                0.1,
                min(1.0, self.launch_grace_ms / 1000.0),
            )
            if now - last_spawn_at < retry_cooldown_s:
                return
            self._last_spawn_at[snapshot.job_id] = now
            command = [
                *self.worker_command_prefix,
                "--store-dir",
                str(self.store.base_dir),
                "--job-id",
                snapshot.job_id,
                "--worker-token",
                snapshot.worker_token,
                "--heartbeat-interval-ms",
                str(self.heartbeat_interval_ms),
                "--cancel-grace-ms",
                str(self.cancel_grace_ms),
            ]
            worker_environment = self.environment_profile.to_environment()
            worker_environment.update(dict(self.worker_environment_overlay))
            popen_kwargs: dict[str, Any] = {
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "close_fds": True,
                "shell": False,
                "env": worker_environment,
            }
            if os.name == "nt":  # pragma: no cover - exercised on Windows
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                creationflags |= getattr(subprocess, "DETACHED_PROCESS", 0)
                if creationflags:
                    popen_kwargs["creationflags"] = creationflags
            else:
                popen_kwargs["start_new_session"] = True
            worker = subprocess.Popen(command, **popen_kwargs)
            self._worker_wrappers[snapshot.job_id] = worker
            # Reap normally while this runtime is alive.  If the runtime exits
            # abruptly, the detached worker is re-parented and keeps running.
            def _reap() -> None:
                worker.wait()
                with self._spawn_lock:
                    current = self._worker_wrappers.get(snapshot.job_id)
                    if current is worker:
                        self._worker_wrappers.pop(snapshot.job_id, None)

            reaper = threading.Thread(
                target=_reap,
                name=f"durable-job-reaper-{snapshot.job_id}",
                daemon=True,
            )
            reaper.start()

    def _normalize_worker_command_prefix(
        self,
        value: Sequence[str | os.PathLike[str]] | None,
    ) -> tuple[str, ...]:
        if value is None:
            return (
                self.python_executable,
                "-m",
                "unchain.jobs._worker",
            )
        if isinstance(value, (str, bytes, os.PathLike)):
            raise TypeError("worker_command_prefix must be a sequence of arguments")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, (str, os.PathLike)):
                raise TypeError(
                    "worker_command_prefix arguments must be strings or paths"
                )
            argument = os.fspath(item)
            if not isinstance(argument, str):
                raise TypeError("worker_command_prefix paths must resolve to strings")
            if not argument or "\0" in argument:
                raise ValueError(
                    "worker_command_prefix arguments must be non-empty and NUL-free"
                )
            normalized.append(argument)
        if not normalized:
            raise ValueError("worker_command_prefix must not be empty")
        return tuple(normalized)

    @staticmethod
    def _normalize_worker_environment_overlay(
        value: Mapping[str, str] | None,
    ) -> tuple[tuple[str, str], ...]:
        if value is None:
            return ()
        if not isinstance(value, Mapping):
            raise TypeError("worker_environment_overlay must be a string mapping")
        normalized: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str):
                raise TypeError(
                    "worker_environment_overlay keys and values must be strings"
                )
            if not key or "\0" in key or "=" in key or "\0" in item:
                raise ValueError("worker_environment_overlay contains an invalid entry")
            normalized_key = key.upper() if os.name == "nt" else key
            previous = normalized.get(normalized_key)
            if previous is not None and previous != item:
                raise ValueError(
                    "worker_environment_overlay contains conflicting keys"
                )
            normalized[normalized_key] = item
        return tuple(sorted(normalized.items()))

    def _heartbeat_is_fresh(
        self,
        snapshot: DurableJobSnapshot,
        heartbeat: dict[str, Any] | None,
    ) -> bool:
        if not isinstance(heartbeat, dict):
            return False
        if heartbeat.get("worker_token") != snapshot.worker_token:
            return False
        updated_at_ms = heartbeat.get("updated_at_ms")
        worker_pid = heartbeat.get("worker_pid")
        if (
            isinstance(updated_at_ms, bool)
            or not isinstance(updated_at_ms, int)
            or isinstance(worker_pid, bool)
            or not isinstance(worker_pid, int)
            or worker_pid <= 0
        ):
            return False
        if snapshot.worker_pid is not None and worker_pid != snapshot.worker_pid:
            return False
        age_ms = max(0, self.store.now_ms() - updated_at_ms)
        return age_ms <= self.heartbeat_stale_ms

    @staticmethod
    def _claim_matches(snapshot: DurableJobSnapshot, claim: dict[str, Any]) -> bool:
        return (
            claim.get("job_id") == snapshot.job_id
            and claim.get("worker_token") == snapshot.worker_token
        )

    @staticmethod
    def _pid_may_be_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            return exc.errno != errno.ESRCH
        return True

    def _persist_outcome_unknown(
        self,
        snapshot: DurableJobSnapshot,
        *,
        reason: str,
        recheck_heartbeat: bool,
        heartbeat: dict[str, Any] | None,
        expected_claim_id: str | None = None,
    ) -> DurableJobSnapshot:
        stale_before_ms = None
        expected_heartbeat_seq = None
        if recheck_heartbeat:
            stale_before_ms = max(
                0,
                self.store.now_ms() - self.heartbeat_stale_ms,
            )
            expected_heartbeat_seq = int(
                heartbeat.get("heartbeat_seq", 0)
                if isinstance(heartbeat, dict)
                else 0
            )
        result = self.store.transition_to_outcome_unknown(
            snapshot.job_id,
            execution_id=snapshot.execution_id,
            expected_revision=snapshot.revision,
            expected_status=snapshot.status,
            reason=reason,
            error="cannot safely prove the detached worker state",
            heartbeat_stale_before_ms=stale_before_ms,
            expected_heartbeat_seq=expected_heartbeat_seq,
            expected_claim_id=expected_claim_id,
        )
        self._clear_suspicion(snapshot.job_id)
        return result

    def _suspect_grace_elapsed(
        self,
        snapshot: DurableJobSnapshot,
        *,
        heartbeat: dict[str, Any] | None,
        reason: str,
        claim_id: str = "",
    ) -> bool:
        signature = (
            snapshot.status,
            snapshot.revision,
            snapshot.worker_token,
            snapshot.worker_pid,
            reason,
            claim_id,
            heartbeat.get("worker_pid") if isinstance(heartbeat, dict) else None,
            heartbeat.get("heartbeat_seq", 0) if isinstance(heartbeat, dict) else 0,
            heartbeat.get("updated_at_ms") if isinstance(heartbeat, dict) else None,
        )
        now = self._monotonic_now()
        with self._suspect_lock:
            current = self._suspect_observations.get(snapshot.job_id)
            if current is None or current.signature != signature:
                self._suspect_observations[snapshot.job_id] = _SuspectObservation(
                    signature=signature,
                    first_seen_monotonic=now,
                )
                return False
            elapsed_ms = max(0.0, now - current.first_seen_monotonic) * 1_000.0
            return elapsed_ms >= self.suspect_grace_ms

    def _clear_suspicion(self, job_id: str) -> None:
        with self._suspect_lock:
            self._suspect_observations.pop(job_id, None)

    def _monotonic_now(self) -> float:
        value = self._monotonic_clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("monotonic_clock must return a non-negative number")
        if value < 0:
            raise ValueError("monotonic_clock must return a non-negative number")
        return float(value)

    @staticmethod
    def _logs_are_final(snapshot: DurableJobSnapshot) -> bool:
        # ``outcome_unknown`` is terminal for orchestration, not proof that the
        # external process has stopped writing.
        return snapshot.completed and snapshot.status != "outcome_unknown"

    @staticmethod
    def _result(
        snapshot: DurableJobSnapshot,
        *,
        action: str,
        logs: dict[str, Any],
    ) -> DurableJobResult:
        stdout = str(logs.get("stdout") or "")
        stderr = str(logs.get("stderr") or "")
        stdout_start_offset = int(logs.get("stdout_start_offset") or 0)
        stderr_start_offset = int(logs.get("stderr_start_offset") or 0)
        next_stdout_offset = int(logs.get("stdout_offset") or 0)
        next_stderr_offset = int(logs.get("stderr_offset") or 0)
        consumer_truncated = bool(logs.get("truncated"))
        result = DurableJobResult(
            ok=snapshot.ok,
            action=action,
            status=snapshot.status,
            task_id=snapshot.job_id,
            job_id=snapshot.job_id,
            execution_id=snapshot.execution_id,
            adapter=snapshot.adapter,
            durable=True,
            background=True,
            stdout=stdout,
            stderr=stderr,
            completed=snapshot.completed,
            returncode=snapshot.returncode,
            timed_out=snapshot.timed_out,
            cancelled=snapshot.cancelled,
            outcome_unknown=snapshot.status == "outcome_unknown",
            outcome_unknown_reason=snapshot.outcome_unknown_reason,
            error=snapshot.error,
            stdout_truncated=snapshot.stdout_truncated,
            stderr_truncated=snapshot.stderr_truncated,
            truncated=(
                consumer_truncated
                or snapshot.stdout_truncated
                or snapshot.stderr_truncated
            ),
            stdout_offset=stdout_start_offset,
            stderr_offset=stderr_start_offset,
            next_stdout_offset=next_stdout_offset,
            next_stderr_offset=next_stderr_offset,
            offset_unit=str(logs.get("offset_unit") or "utf8_bytes"),
            stdout_available=int(logs.get("stdout_available") or 0),
            stderr_available=int(logs.get("stderr_available") or 0),
        )
        return result

    @staticmethod
    def _positive_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be a positive integer")
        if value <= 0:
            raise ValueError(f"{field} must be a positive integer")
        return value

    @staticmethod
    def _non_negative_int(value: Any, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field} must be a non-negative integer")
        if value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value


__all__ = ["DurableJobResult", "ProcessJobSupervisor"]
