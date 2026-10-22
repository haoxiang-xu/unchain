from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShellExecutorSpec:
    family: str
    program: str
    argv: list[str]
    platform: str


@dataclass
class _BackgroundTask:
    task_id: str
    session_key: str
    process: subprocess.Popen[bytes]
    cwd: Path
    shell_family: str
    platform: str
    lock: threading.Lock = field(default_factory=threading.Lock)
    stdout_buffer: str = ""
    stderr_buffer: str = ""
    stdout_offset: int = 0
    stderr_offset: int = 0
    timed_out: bool = False
    killed: bool = False
    reader_threads: list[threading.Thread] = field(default_factory=list)

    def append_output(self, stream_name: str, text: str) -> None:
        with self.lock:
            if stream_name == "stdout":
                self.stdout_buffer += text
            else:
                self.stderr_buffer += text

    def consume_increment(self, stream_name: str, max_chars: int) -> tuple[str, bool]:
        with self.lock:
            if stream_name == "stdout":
                buffer = self.stdout_buffer
                offset = self.stdout_offset
            else:
                buffer = self.stderr_buffer
                offset = self.stderr_offset

            remainder = buffer[offset:]
            if len(remainder) > max_chars:
                chunk = remainder[:max_chars]
                new_offset = offset + len(chunk)
                truncated = True
            else:
                chunk = remainder
                new_offset = len(buffer)
                truncated = False

            if stream_name == "stdout":
                self.stdout_offset = new_offset
            else:
                self.stderr_offset = new_offset
            return chunk, truncated

    def join_readers(self, timeout: float = 0.05) -> None:
        for thread in list(self.reader_threads):
            thread.join(timeout=timeout)


class ShellRuntime:
    DEFAULT_TIMEOUT_MS = 120_000
    MAX_TIMEOUT_MS = 600_000
    DEFAULT_MAX_OUTPUT_CHARS = 20_000
    MAX_MAX_OUTPUT_CHARS = 100_000
    DEFAULT_YIELD_TIME_MS = 300
    _POSIX_READ_ONLY_COMMANDS = {
        "pwd",
        "ls",
        "find",
        "rg",
        "grep",
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "file",
        "which",
        "echo",
        "printf",
        "git",
    }
    _POWERSHELL_READ_ONLY_COMMANDS = {
        "get-location",
        "get-childitem",
        "get-content",
        "get-item",
        "test-path",
        "resolve-path",
        "select-string",
        "get-filehash",
        "git",
    }
    _READ_ONLY_GIT_SUBCOMMANDS = {"status", "diff", "log"}
    _HIGH_RISK_COMMAND_PATTERN = re.compile(
        r"\b("
        r"rm|mv|cp|touch|mkdir|rmdir|install|curl|wget|ssh|scp|rsync|chmod|chown|sudo|"
        r"apt(?:-get)?|yum|dnf|brew|pip(?:3)?|npm|pnpm|yarn|uv|poetry|"
        r"git\s+(?:add|commit|push|checkout|switch|branch|merge|rebase|reset|clean|stash|apply|am)"
        r")\b",
        re.IGNORECASE,
    )
    _POSIX_SUSPICIOUS_TOKENS = ("<<", ">>", ">", "<", "$(", "`", "\n")
    _POWERSHELL_SUSPICIOUS_TOKENS = ("<<", ">>", ">", "<", "$(", "`", "\n")

    def __init__(self, workspace_roots: list[Path]):
        self.workspace_roots = [Path(root).resolve() for root in workspace_roots]
        self.workspace_root = self.workspace_roots[0]
        self.cwd_by_session: dict[str, Path] = {}
        self.background_tasks: dict[str, _BackgroundTask] = {}

    @classmethod
    def detect_executor(
        cls,
        *,
        platform_name: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ShellExecutorSpec:
        resolved_platform = str(platform_name or sys.platform).lower()
        environment = env or os.environ

        if resolved_platform.startswith("win"):
            pwsh = shutil.which("pwsh")
            if pwsh:
                program = pwsh
            else:
                program = shutil.which("powershell.exe") or "powershell.exe"
            return ShellExecutorSpec(
                family="powershell",
                program=program,
                argv=[program, "-NoLogo", "-NoProfile", "-Command"],
                platform=resolved_platform,
            )

        shell_env = str(environment.get("SHELL", "") or "").strip()
        if shell_env:
            candidate = Path(shell_env)
            if candidate.is_absolute() and candidate.exists():
                program = str(candidate)
            else:
                program = shutil.which(shell_env) or "/bin/sh"
        else:
            program = "/bin/sh"
        return ShellExecutorSpec(
            family="posix",
            program=program,
            argv=[program, "-lc"],
            platform=resolved_platform,
        )

    @classmethod
    def is_low_risk_command(cls, command: str, family: str) -> bool:
        text = str(command or "").strip()
        if not text:
            return False
        if cls._HIGH_RISK_COMMAND_PATTERN.search(text):
            return False

        if family == "powershell":
            if any(token in text for token in cls._POWERSHELL_SUSPICIOUS_TOKENS):
                return False
            if re.search(r"[;|&]", text):
                return False
            command_name, arguments = cls._parse_powershell_command(text)
            if not command_name:
                return False
            if command_name == "git":
                return bool(arguments and arguments[0].lower() in cls._READ_ONLY_GIT_SUBCOMMANDS)
            return command_name in cls._POWERSHELL_READ_ONLY_COMMANDS

        if any(token in text for token in cls._POSIX_SUSPICIOUS_TOKENS):
            return False
        if re.search(r"[;|&]", text):
            return False
        try:
            argv = shlex.split(text, posix=True)
        except ValueError:
            return False
        if not argv:
            return False
        executable = Path(argv[0]).name.lower()
        if executable == "git":
            return len(argv) >= 2 and argv[1].lower() in cls._READ_ONLY_GIT_SUBCOMMANDS
        return executable in cls._POSIX_READ_ONLY_COMMANDS

    def default_cwd_for_session(self, session_key: str) -> Path:
        return self.cwd_by_session.get(session_key, self.workspace_root)

    def resolve_cwd(self, session_key: str, cwd: str | None) -> tuple[Path | None, str | None]:
        base = self.default_cwd_for_session(session_key)
        target = str(cwd or "").strip()
        path_obj = Path(target) if target else base
        resolved = path_obj.resolve() if path_obj.is_absolute() else (base / path_obj).resolve()
        for root in self.workspace_roots:
            try:
                resolved.relative_to(root)
                if not resolved.exists():
                    return None, f"cwd does not exist: {resolved}"
                if not resolved.is_dir():
                    return None, f"cwd is not a directory: {resolved}"
                return resolved, None
            except ValueError:
                continue
        return None, "cwd is outside all workspace roots"

    def run(
        self,
        *,
        session_key: str,
        command: str,
        cwd: str | None = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        run_in_background: bool = False,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        yield_time_ms: int = DEFAULT_YIELD_TIME_MS,
    ) -> dict[str, Any]:
        spec = self.detect_executor()
        result = self._base_result(
            action="run",
            shell_family=spec.family,
            platform=spec.platform,
            cwd=str(self.default_cwd_for_session(session_key)),
        )
        result["background"] = bool(run_in_background)

        if not isinstance(command, str) or not command.strip():
            result["error"] = "command is required"
            return result

        resolved_cwd, cwd_error = self.resolve_cwd(session_key, cwd)
        if cwd_error:
            result["error"] = cwd_error
            result["cwd"] = cwd or result["cwd"]
            return result
        assert resolved_cwd is not None
        result["cwd"] = str(resolved_cwd)

        resolved_timeout_ms = self._normalize_timeout_ms(timeout_ms)
        resolved_max_output_chars = self._normalize_max_output_chars(max_output_chars)
        resolved_yield_time_ms = self._normalize_yield_time_ms(yield_time_ms)

        if run_in_background:
            return self._run_background(
                spec=spec,
                session_key=session_key,
                command=command,
                cwd=resolved_cwd,
                timeout_ms=resolved_timeout_ms,
                max_output_chars=resolved_max_output_chars,
                yield_time_ms=resolved_yield_time_ms,
            )

        return self._run_foreground(
            spec=spec,
            session_key=session_key,
            command=command,
            cwd=resolved_cwd,
            timeout_ms=resolved_timeout_ms,
            max_output_chars=resolved_max_output_chars,
        )

    def poll(
        self,
        task_id: str,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        *,
        session_key: str = "__default__",
    ) -> dict[str, Any]:
        task = self._task_for_session(task_id, session_key=session_key)
        if task is None:
            return {
                **self._base_result(action="poll"),
                "task_id": str(task_id or ""),
                "status": "missing",
                "completed": True,
                "error": f"task not found: {task_id}",
            }

        if task.process.poll() is not None:
            task.join_readers()

        resolved_max_output_chars = self._normalize_max_output_chars(max_output_chars)
        stdout_chunk, stdout_truncated = task.consume_increment("stdout", resolved_max_output_chars)
        stderr_chunk, stderr_truncated = task.consume_increment("stderr", resolved_max_output_chars)
        completed = task.process.poll() is not None
        returncode = task.process.returncode if completed else None
        status = "running"
        if task.killed:
            status = "killed"
        elif task.timed_out:
            status = "timed_out"
        elif completed:
            status = "completed"
        return {
            **self._base_result(
                action="poll",
                shell_family=task.shell_family,
                platform=task.platform,
                cwd=str(task.cwd),
                task_id=task.task_id,
            ),
            "ok": bool(completed and returncode == 0 and not task.timed_out and not task.killed),
            "status": status,
            "background": True,
            "stdout": stdout_chunk,
            "stderr": stderr_chunk,
            "completed": completed,
            "returncode": returncode,
            "timed_out": task.timed_out,
            "truncated": stdout_truncated or stderr_truncated,
        }

    def wait(
        self,
        task_id: str,
        *,
        session_key: str = "__default__",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        """Wait once for a background task without spending model iterations."""

        task = self._task_for_session(task_id, session_key=session_key)
        if task is None:
            return {
                **self._base_result(action="wait"),
                "task_id": str(task_id or ""),
                "status": "missing",
                "completed": True,
                "wait_timed_out": False,
                "error": f"task not found: {task_id}",
            }

        resolved_timeout_ms = self._normalize_timeout_ms(timeout_ms)
        wait_timed_out = False
        if task.process.poll() is None:
            try:
                task.process.wait(timeout=resolved_timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                wait_timed_out = True

        result = self.poll(
            task_id=task.task_id,
            max_output_chars=max_output_chars,
            session_key=session_key,
        )
        result["action"] = "wait"
        result["wait_timeout_ms"] = resolved_timeout_ms
        result["wait_timed_out"] = bool(wait_timed_out and not result["completed"])
        return result

    def kill(
        self,
        task_id: str,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        *,
        session_key: str = "__default__",
    ) -> dict[str, Any]:
        task = self._task_for_session(task_id, session_key=session_key)
        if task is None:
            return {
                **self._base_result(action="kill"),
                "task_id": str(task_id or ""),
                "status": "missing",
                "completed": True,
                "error": f"task not found: {task_id}",
            }

        if task.process.poll() is None:
            self._terminate_task(task, timed_out=False)
        task.join_readers()
        resolved_max_output_chars = self._normalize_max_output_chars(max_output_chars)
        stdout_chunk, stdout_truncated = task.consume_increment("stdout", resolved_max_output_chars)
        stderr_chunk, stderr_truncated = task.consume_increment("stderr", resolved_max_output_chars)
        return {
            **self._base_result(
                action="kill",
                shell_family=task.shell_family,
                platform=task.platform,
                cwd=str(task.cwd),
                task_id=task.task_id,
            ),
            "ok": False,
            "status": "killed",
            "background": True,
            "stdout": stdout_chunk,
            "stderr": stderr_chunk,
            "completed": True,
            "returncode": task.process.returncode,
            "timed_out": task.timed_out,
            "truncated": stdout_truncated or stderr_truncated,
        }

    def shutdown(self) -> None:
        for task in list(self.background_tasks.values()):
            try:
                self.kill(
                    task.task_id,
                    max_output_chars=self.DEFAULT_MAX_OUTPUT_CHARS,
                    session_key=task.session_key,
                )
            except Exception:
                continue
        self.background_tasks.clear()

    def _run_foreground(
        self,
        *,
        spec: ShellExecutorSpec,
        session_key: str,
        command: str,
        cwd: Path,
        timeout_ms: int,
        max_output_chars: int,
    ) -> dict[str, Any]:
        capture_fd, capture_path = tempfile.mkstemp(prefix="unchain-shell-cwd-", suffix=".txt")
        os.close(capture_fd)
        wrapped_command = self._wrap_command(command=command, cwd_capture_path=capture_path, family=spec.family)
        argv = [*spec.argv, wrapped_command]

        try:
            completed = subprocess.run(
                argv,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=max(1, timeout_ms / 1000.0),
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_text = self._decode_bytes(exc.stdout)
            stderr_text = self._decode_bytes(exc.stderr)
            stdout_chunk, stdout_truncated = self._truncate_text(stdout_text, max_output_chars)
            stderr_chunk, stderr_truncated = self._truncate_text(stderr_text, max_output_chars)
            self._safe_unlink(capture_path)
            return {
                **self._base_result(
                    action="run",
                    shell_family=spec.family,
                    platform=spec.platform,
                    cwd=str(cwd),
                ),
                "background": False,
                "status": "timed_out",
                "stdout": stdout_chunk,
                "stderr": stderr_chunk,
                "timed_out": True,
                "truncated": stdout_truncated or stderr_truncated,
                "error": f"command timed out after {int(timeout_ms / 1000)}s",
            }
        except Exception as exc:
            self._safe_unlink(capture_path)
            return {
                **self._base_result(
                    action="run",
                    shell_family=spec.family,
                    platform=spec.platform,
                    cwd=str(cwd),
                ),
                "background": False,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        stdout_text = self._decode_bytes(completed.stdout)
        stderr_text = self._decode_bytes(completed.stderr)
        stdout_chunk, stdout_truncated = self._truncate_text(stdout_text, max_output_chars)
        stderr_chunk, stderr_truncated = self._truncate_text(stderr_text, max_output_chars)
        final_cwd = self._read_captured_cwd(capture_path, fallback=cwd)
        self._safe_unlink(capture_path)
        self.cwd_by_session[session_key] = final_cwd
        return {
            **self._base_result(
                action="run",
                shell_family=spec.family,
                platform=spec.platform,
                cwd=str(final_cwd),
            ),
            "ok": completed.returncode == 0,
            "status": "completed",
            "background": False,
            "returncode": completed.returncode,
            "stdout": stdout_chunk,
            "stderr": stderr_chunk,
            "timed_out": False,
            "truncated": stdout_truncated or stderr_truncated,
        }

    def _run_background(
        self,
        *,
        spec: ShellExecutorSpec,
        session_key: str,
        command: str,
        cwd: Path,
        timeout_ms: int,
        max_output_chars: int,
        yield_time_ms: int,
    ) -> dict[str, Any]:
        popen_kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL,
            "text": False,
            "shell": False,
        }
        if spec.family == "powershell":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if creationflags:
                popen_kwargs["creationflags"] = creationflags
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen([*spec.argv, command], **popen_kwargs)
        task_id = uuid.uuid4().hex
        task = _BackgroundTask(
            task_id=task_id,
            session_key=session_key,
            process=process,
            cwd=cwd,
            shell_family=spec.family,
            platform=spec.platform,
        )
        self.background_tasks[task_id] = task
        self._start_reader(task, "stdout", process.stdout)
        self._start_reader(task, "stderr", process.stderr)
        self._start_timeout_watcher(task, timeout_ms=timeout_ms)

        if yield_time_ms > 0:
            time.sleep(min(yield_time_ms / 1000.0, 1.0))

        return {
            **self._base_result(
                action="run",
                shell_family=spec.family,
                platform=spec.platform,
                cwd=str(cwd),
                task_id=task_id,
            ),
            "ok": True,
            "status": "running",
            "background": True,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
            "truncated": False,
        }

    def _task_for_session(
        self,
        task_id: str,
        *,
        session_key: str,
    ) -> _BackgroundTask | None:
        task = self.background_tasks.get(str(task_id or ""))
        if task is None or task.session_key != str(session_key or ""):
            return None
        return task

    def _start_timeout_watcher(self, task: _BackgroundTask, *, timeout_ms: int) -> None:
        def _watch() -> None:
            try:
                task.process.wait(timeout=max(1, timeout_ms / 1000.0))
            except subprocess.TimeoutExpired:
                self._terminate_task(task, timed_out=True)

        watcher = threading.Thread(target=_watch, name=f"shell-watch-{task.task_id}", daemon=True)
        watcher.start()

    def _start_reader(self, task: _BackgroundTask, stream_name: str, pipe: Any) -> None:
        if pipe is None:
            return

        def _reader() -> None:
            try:
                while True:
                    chunk = pipe.read(4096)
                    if not chunk:
                        break
                    task.append_output(stream_name, self._decode_bytes(chunk))
            finally:
                try:
                    pipe.close()
                except Exception:
                    # Closing must not suppress a read failure or interruption.
                    pass

        reader = threading.Thread(target=_reader, name=f"shell-{stream_name}-{task.task_id}", daemon=True)
        task.reader_threads.append(reader)
        reader.start()

    def _terminate_task(self, task: _BackgroundTask, *, timed_out: bool) -> None:
        if task.process.poll() is not None:
            if timed_out:
                task.timed_out = True
            return
        try:
            if task.shell_family == "powershell":
                task.process.kill()
            else:
                os.killpg(task.process.pid, signal.SIGKILL)
        except Exception:
            try:
                task.process.kill()
            except Exception:
                # The process may exit during signaling; still attempt the bounded wait below.
                pass
        try:
            task.process.wait(timeout=2.0)
        except Exception:
            # Cancellation/timeout cleanup remains bounded even if the OS cannot reap the process promptly.
            pass
        if timed_out:
            task.timed_out = True
        else:
            task.killed = True

    def _wrap_command(self, *, command: str, cwd_capture_path: str, family: str) -> str:
        if family == "powershell":
            escaped_path = cwd_capture_path.replace("'", "''")
            return (
                "$ErrorActionPreference = 'Continue'\n"
                f"{command}\n"
                "$status = if ($null -ne $LASTEXITCODE) { [int]$LASTEXITCODE } elseif ($?) { 0 } else { 1 }\n"
                f"[System.IO.File]::WriteAllText('{escaped_path}', (Get-Location).Path)\n"
                "exit $status"
            )
        quoted_path = shlex.quote(cwd_capture_path)
        return f"{command}\nunchain_status=$?\npwd > {quoted_path}\nexit $unchain_status"

    def _read_captured_cwd(self, capture_path: str, *, fallback: Path) -> Path:
        try:
            raw = Path(capture_path).read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            return fallback
        if not raw:
            return fallback
        resolved = Path(raw).resolve()
        for root in self.workspace_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        return fallback

    def _normalize_timeout_ms(self, value: Any) -> int:
        try:
            timeout_ms = int(value)
        except (TypeError, ValueError):
            timeout_ms = self.DEFAULT_TIMEOUT_MS
        timeout_ms = max(1_000, timeout_ms)
        return min(self.MAX_TIMEOUT_MS, timeout_ms)

    def _normalize_max_output_chars(self, value: Any) -> int:
        try:
            max_output_chars = int(value)
        except (TypeError, ValueError):
            max_output_chars = self.DEFAULT_MAX_OUTPUT_CHARS
        max_output_chars = max(0, max_output_chars)
        return min(self.MAX_MAX_OUTPUT_CHARS, max_output_chars)

    def _normalize_yield_time_ms(self, value: Any) -> int:
        try:
            yield_time_ms = int(value)
        except (TypeError, ValueError):
            yield_time_ms = self.DEFAULT_YIELD_TIME_MS
        return max(0, min(5_000, yield_time_ms))

    def _base_result(
        self,
        *,
        action: str,
        shell_family: str = "",
        platform: str = "",
        cwd: str = "",
        task_id: str = "",
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "action": action,
            "status": "error",
            "shell_family": shell_family,
            "platform": platform,
            "cwd": cwd,
            "task_id": task_id,
            "error": "",
        }

    def _truncate_text(self, text: str, max_chars: int) -> tuple[str, bool]:
        if max_chars < 0:
            return "", bool(text)
        if len(text) <= max_chars:
            return text, False
        return text[:max_chars], True

    def _decode_bytes(self, value: bytes | str | None) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return value.decode("utf-8", errors="replace")

    def _safe_unlink(self, path: str) -> None:
        try:
            os.unlink(path)
        except FileNotFoundError:
            return

    @staticmethod
    def _parse_powershell_command(command: str) -> tuple[str, list[str]]:
        stripped = command.strip()
        if not stripped:
            return "", []
        tokens = re.findall(r"(?:[^\s\"']+|\"[^\"]*\"|'[^']*')+", stripped)
        if not tokens:
            return "", []
        command_name = Path(tokens[0].strip("\"'")).name.lower()
        arguments = [token.strip("\"'") for token in tokens[1:]]
        return command_name, arguments


__all__ = ["ShellExecutorSpec", "ShellRuntime"]
