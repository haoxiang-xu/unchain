from __future__ import annotations

import argparse
import codecs
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, BinaryIO, Sequence

from .environment import JobEnvironmentProfile
from .models import DurableJobSnapshot
from .store import JsonFileJobStore


_READER_JOIN_TIMEOUT_S = 2.0


class _HeartbeatLease:
    """Publish worker liveness until a terminal receipt has been committed."""

    def __init__(
        self,
        *,
        store: JsonFileJobStore,
        job_id: str,
        worker_token: str,
        worker_pid: int,
        interval_ms: int,
    ) -> None:
        self.store = store
        self.job_id = job_id
        self.worker_token = worker_token
        self.worker_pid = worker_pid
        self.interval_s = max(0.001, interval_ms / 1_000.0)
        self.last_error = ""
        self._stop = threading.Event()
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"durable-job-heartbeat-{job_id}",
            daemon=True,
        )

    def start(self) -> None:
        if not self.store.write_heartbeat(
            self.job_id,
            worker_token=self.worker_token,
            worker_pid=self.worker_pid,
        ):
            raise RuntimeError("durable job became terminal before worker startup")
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        if not self._started:
            return
        self._thread.join(timeout=max(1.0, self.interval_s * 4.0))

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                if not self.store.write_heartbeat(
                    self.job_id,
                    worker_token=self.worker_token,
                    worker_pid=self.worker_pid,
                ):
                    return
                self.last_error = ""
            except Exception as exc:
                # A transient filesystem error must not permanently disable
                # the lease.  The main worker still owns the command and will
                # either commit a receipt or stop this thread on exit.
                self.last_error = f"{type(exc).__name__}: {exc}"


class _LogPump:
    """Drain one child pipe into a bounded, append-only durable log."""

    def __init__(
        self,
        *,
        pipe: BinaryIO,
        destination: Path,
        max_bytes: int,
        label: str,
    ) -> None:
        self.pipe = pipe
        self.destination = destination
        self.max_bytes = max_bytes
        self.label = label
        self.truncated = False
        self.error = ""
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._copy,
            name=f"durable-job-{label}-pump",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def finish(self) -> None:
        self._thread.join(timeout=_READER_JOIN_TIMEOUT_S)
        if self._thread.is_alive():
            with self._lock:
                self.truncated = True
                if not self.error:
                    self.error = f"{self.label} pipe remained open after command exit"
            try:
                self.pipe.close()
            except Exception:
                # The pipe may already be closed; preserve the recorded truncation and finish the bounded join.
                pass
            self._thread.join(timeout=0.25)

    def _copy(self) -> None:
        descriptor: int | None = None
        try:
            self.destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            descriptor = os.open(
                self.destination,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                written = os.fstat(descriptor).st_size
            except OSError:
                written = self.destination.stat().st_size if self.destination.exists() else 0

            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            accepting_output = written < self.max_bytes
            while True:
                # ``BufferedReader.read(size)`` may wait for the entire size or
                # EOF.  Reading the descriptor directly makes output visible
                # to a reattached runtime while the command is still running.
                chunk = os.read(self.pipe.fileno(), 64 * 1024)
                if not chunk:
                    break
                normalized = decoder.decode(chunk, final=False).encode("utf-8")
                accepted = (
                    self._bounded_utf8_prefix(
                        normalized,
                        max(0, self.max_bytes - written),
                    )
                    if accepting_output
                    else b""
                )
                if accepted:
                    self._write_all(descriptor, accepted)
                    written += len(accepted)
                if len(accepted) < len(normalized):
                    accepting_output = False
                    with self._lock:
                        self.truncated = True
            final_payload = decoder.decode(b"", final=True).encode("utf-8")
            accepted = (
                self._bounded_utf8_prefix(
                    final_payload,
                    max(0, self.max_bytes - written),
                )
                if accepting_output
                else b""
            )
            if accepted:
                self._write_all(descriptor, accepted)
                written += len(accepted)
            if len(accepted) < len(final_payload):
                with self._lock:
                    self.truncated = True
            os.fsync(descriptor)
        except Exception as exc:
            with self._lock:
                self.error = f"cannot persist {self.label}: {type(exc).__name__}: {exc}"
                self.truncated = True
        finally:
            try:
                self.pipe.close()
            except Exception:
                # Closing the pipe is best effort; any read or durable-log failure is already recorded above.
                pass
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    # The descriptor may already be closed; retain the primary log persistence result.
                    pass

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write while persisting durable job output")
            view = view[written:]

    @staticmethod
    def _bounded_utf8_prefix(payload: bytes, limit: int) -> bytes:
        if limit <= 0:
            return b""
        if len(payload) <= limit:
            return payload
        return payload[:limit].decode("utf-8", errors="ignore").encode("utf-8")


def _popen_kwargs(*, cwd: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
        "text": False,
        "close_fds": True,
        "env": dict(os.environ),
    }
    if os.name == "nt":  # pragma: no cover - exercised on Windows
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if creationflags:
            kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _terminate_child(
    process: subprocess.Popen[bytes],
    *,
    grace_s: float,
) -> None:
    """Terminate only the child whose live ``Popen`` this worker owns."""

    if process.poll() is not None:
        return
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        # The child exited between poll and signal; the wait below still reaps it.
        pass
    except Exception:
        try:
            process.terminate()
        except Exception:
            # The process may exit during fallback termination; wait and escalate after the grace period.
            pass
    try:
        process.wait(timeout=grace_s)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.name == "nt":  # pragma: no cover - exercised on Windows
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        # The child exited between the grace wait and kill; continue to reap it below.
        pass
    except Exception:
        try:
            process.kill()
        except Exception:
            # Killing is best effort after timeout; the final bounded wait still attempts to reap the child.
            pass
    try:
        process.wait(timeout=grace_s)
    except Exception:
        # Cleanup must not replace the worker outcome when the OS cannot reap the child within the grace period.
        pass


def _record_orchestration_failure(
    store: JsonFileJobStore,
    snapshot: DurableJobSnapshot,
    *,
    worker_token: str,
    process: subprocess.Popen[bytes] | None,
    error: BaseException,
    cancel_grace_s: float,
) -> None:
    message = f"durable worker failed: {type(error).__name__}: {error}"
    try:
        current = store.load(snapshot.job_id, execution_id=snapshot.execution_id)
    except Exception:
        return
    if current.completed:
        return

    returncode: int | None = None
    if process is not None:
        _terminate_child(process, grace_s=cancel_grace_s)
        returncode = process.poll()
        # If the worker cannot prove that its own child stopped, leave the
        # nonterminal state in place.  A supervisor will surface it as
        # outcome_unknown after the heartbeat expires.
        if returncode is None:
            return
    try:
        store.update_state(
            snapshot.job_id,
            worker_token=worker_token,
            status="failed",
            worker_pid=os.getpid(),
            child_pid=process.pid if process is not None else current.child_pid,
            returncode=returncode,
            error=message,
        )
    except Exception:
        return


def run_worker(
    *,
    store_dir: str | Path,
    job_id: str,
    worker_token: str,
    heartbeat_interval_ms: int = 1_000,
    cancel_grace_ms: int = 1_000,
) -> int:
    store = JsonFileJobStore(store_dir)
    spec = store.load_spec(job_id)
    if JobEnvironmentProfile.capture(os.environ).digest != spec["environment_digest"]:
        # The immutable profile is checked again inside the detached wrapper,
        # before it can claim the job or launch user code.
        return 4
    if str(spec["worker_token"]) != str(worker_token or ""):
        return 2
    worker_pid = os.getpid()
    if not store.claim_worker(
        job_id,
        worker_token=worker_token,
        worker_pid=worker_pid,
    ):
        # Concurrent supervisors may launch multiple wrappers, but the claim
        # ensures only one wrapper can ever reach the user command.
        return 0

    lease = _HeartbeatLease(
        store=store,
        job_id=job_id,
        worker_token=worker_token,
        worker_pid=worker_pid,
        interval_ms=heartbeat_interval_ms,
    )
    snapshot: DurableJobSnapshot | None = None
    process: subprocess.Popen[bytes] | None = None
    stdout_pump: _LogPump | None = None
    stderr_pump: _LogPump | None = None
    try:
        # Start the lease immediately after the durable claim.  The independent
        # thread remains active through command teardown, log drain/fsync, and
        # the terminal state commit below.
        lease.start()
        snapshot = store.load(job_id, execution_id=str(spec["execution_id"]))
        # This durable STARTING receipt is written before Popen.  Consequently
        # a stale claim paired with QUEUED state is safe for a supervisor to
        # retry; no user command can have started in that state.
        snapshot = store.update_state(
            job_id,
            worker_token=worker_token,
            status="starting",
            worker_pid=worker_pid,
        )
        if store.cancel_requested(job_id):
            store.update_state(
                job_id,
                worker_token=worker_token,
                status="cancelled",
                worker_pid=worker_pid,
                cancelled=True,
                error="cancelled before command launch",
            )
            return 0

        # The claim generation is the final launch fence.  A stale supervisor
        # must not be able to replace this claim between acquisition and
        # Popen; if ownership is ever lost, fail closed before user code runs.
        if not store.worker_claim_is_current(
            job_id,
            worker_token=worker_token,
            worker_pid=worker_pid,
        ):
            store.transition_to_outcome_unknown(
                job_id,
                execution_id=snapshot.execution_id,
                expected_revision=snapshot.revision,
                expected_status=snapshot.status,
                reason="worker_claim_lost_before_launch",
                error="worker claim changed before command launch",
            )
            return 3

        process = subprocess.Popen(
            list(spec["argv"]),
            **_popen_kwargs(cwd=str(spec["cwd"])),
        )
        snapshot = store.update_state(
            job_id,
            worker_token=worker_token,
            status="running",
            worker_pid=worker_pid,
            child_pid=process.pid,
        )
        paths = store.paths_for_worker(job_id)
        assert process.stdout is not None
        assert process.stderr is not None
        stdout_pump = _LogPump(
            pipe=process.stdout,
            destination=paths["stdout"],
            max_bytes=int(spec["max_log_bytes"]),
            label="stdout",
        )
        stderr_pump = _LogPump(
            pipe=process.stderr,
            destination=paths["stderr"],
            max_bytes=int(spec["max_log_bytes"]),
            label="stderr",
        )
        stdout_pump.start()
        stderr_pump.start()

        deadline = time.monotonic() + (int(spec["timeout_ms"]) / 1000.0)
        cancelled = False
        timed_out = False
        while process.poll() is None:
            if store.cancel_requested(job_id):
                cancelled = True
                _terminate_child(process, grace_s=cancel_grace_ms / 1000.0)
                break
            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_child(process, grace_s=cancel_grace_ms / 1000.0)
                break
            time.sleep(heartbeat_interval_ms / 1000.0)

        try:
            returncode = process.wait(timeout=cancel_grace_ms / 1000.0)
        except subprocess.TimeoutExpired:
            _terminate_child(process, grace_s=cancel_grace_ms / 1000.0)
            returncode = process.poll()

        stdout_pump.finish()
        stderr_pump.finish()
        capture_errors = [
            pump.error
            for pump in (stdout_pump, stderr_pump)
            if pump.error
        ]
        stdout_truncated = stdout_pump.truncated
        stderr_truncated = stderr_pump.truncated

        if returncode is None:
            # Do not fabricate completion when even this owning worker could
            # not prove its child stopped.  Heartbeat expiry will make the
            # ambiguity explicit to a reattached supervisor.
            return 3
        if cancelled:
            status = "cancelled"
            error = "command cancelled"
        elif timed_out:
            status = "timed_out"
            error = f"command timed out after {spec['timeout_ms']} ms"
        elif capture_errors:
            status = "failed"
            error = "; ".join(capture_errors)
        elif returncode == 0:
            status = "completed"
            error = ""
        else:
            status = "failed"
            error = f"command exited with return code {returncode}"

        store.update_state(
            job_id,
            worker_token=worker_token,
            status=status,
            worker_pid=worker_pid,
            child_pid=process.pid,
            returncode=returncode,
            timed_out=timed_out,
            cancelled=cancelled,
            error=error,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        return 0
    except BaseException as exc:
        # This is the worker process boundary: stop the owned child, record
        # failure where possible, and return a nonzero process exit status.
        if snapshot is not None:
            _record_orchestration_failure(
                store,
                snapshot,
                worker_token=worker_token,
                process=process,
                error=exc,
                cancel_grace_s=cancel_grace_ms / 1000.0,
            )
        return 1
    finally:
        lease.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one durable local-process job")
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--worker-token", required=True)
    parser.add_argument("--heartbeat-interval-ms", type=int, default=1_000)
    parser.add_argument("--cancel-grace-ms", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_worker(
        store_dir=args.store_dir,
        job_id=args.job_id,
        worker_token=args.worker_token,
        heartbeat_interval_ms=max(1, args.heartbeat_interval_ms),
        cancel_grace_ms=max(1, args.cancel_grace_ms),
    )


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
