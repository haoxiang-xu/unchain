"""Authenticated local-process host for Skeleton Python Code nodes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .durable_execution import DurableNodeContext
from .errors import SkeletonValidationError
from .execution import WorkflowNodeCompleted
from .io import _validate_json
from .agent_preparation import checked_execution_plan


_RECEIPT_SCHEMA = "unchain.skeleton.code-worker.v1"
_PUBLIC_RECEIPT_SCHEMA = "unchain.skeleton.code-receipt.v1"
_MAX_SOURCE_BYTES = 4 * 1024 * 1024


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PythonEnvironmentProfile:
    """One host-local Python launch profile; never serialized into a Skeleton."""

    profile_id: str
    kind: str
    command: tuple[str, ...]

    @classmethod
    def venv(cls, profile_id: str, executable: str | os.PathLike[str]) -> "PythonEnvironmentProfile":
        return cls(profile_id, "venv", (os.fspath(executable),))

    @classmethod
    def bundled(cls, profile_id: str, command: Sequence[str | os.PathLike[str]]) -> "PythonEnvironmentProfile":
        if isinstance(command, (str, bytes, os.PathLike)):
            _fail("invalid_type", ("environment", "command"), "bundled command must be an argument array")
        return cls(profile_id, "bundled", tuple(os.fspath(item) for item in command))

    def admitted(self) -> "PythonEnvironmentProfile":
        if type(self.profile_id) is not str or not self.profile_id:
            _fail("invalid_value", ("environment", "profile_id"), "profile id is required")
        if self.kind not in {"venv", "bundled"}:
            _fail("unknown_variant", ("environment", "kind"), "Python profile kind is invalid")
        if not self.command or any(type(item) is not str or not item or "\0" in item for item in self.command):
            _fail("invalid_value", ("environment", "command"), "runtime command is invalid")
        executable = Path(self.command[0])
        if not executable.is_absolute():
            _fail("invalid_path", ("environment", "command", 0), "runtime executable must be absolute")
        resolved = executable.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            _fail("missing_runtime", ("environment", "command", 0), "runtime executable is unavailable")
        return self

    def fingerprint(self) -> str:
        admitted = self.admitted()
        artifacts: list[dict[str, str]] = []
        for index, item in enumerate(admitted.command):
            candidate = Path(item)
            if candidate.is_absolute() and candidate.resolve().is_file():
                artifacts.append({"index": str(index), "path": str(candidate.resolve()), "sha256": _file_sha256(candidate.resolve())})
            if admitted.kind == "venv" and index == 0:
                configuration = candidate.parent.parent / "pyvenv.cfg"
                if configuration.is_file():
                    artifacts.append({
                        "index": "venv-config",
                        "path": str(configuration),
                        "sha256": _file_sha256(configuration),
                    })
        return _sha256_bytes(_canonical({"kind": admitted.kind, "command": list(admitted.command), "artifacts": artifacts}))


@dataclass(frozen=True)
class CodeSourceSnapshot:
    relative_path: str
    content: bytes
    sha256: str


class FileCodeSourceResolver:
    """Read one regular relative source file beneath a fixed host root."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            _fail("missing_value", ("source_root",), "code source root is unavailable")

    def __call__(self, relative_path: str) -> CodeSourceSnapshot:
        if type(relative_path) is not str or not relative_path or "\0" in relative_path:
            _fail("invalid_path", ("code", "source", "path"), "code path is invalid")
        lexical = Path(relative_path)
        if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
            _fail("invalid_path", ("code", "source", "path"), "code path must be a normalized relative path")
        candidate = self.root.joinpath(lexical)
        current = self.root
        for part in lexical.parts:
            current = current / part
            if current.is_symlink():
                _fail("invalid_path", ("code", "source", "path"), "code path cannot traverse a symbolic link")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.root)
            mode = resolved.stat().st_mode
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise SkeletonValidationError("missing_value", ("code", "source", "path"), "code source is unavailable") from exc
        if not stat.S_ISREG(mode) or not os.access(resolved, os.R_OK):
            _fail("invalid_path", ("code", "source", "path"), "code source must be a readable regular file")
        content = resolved.read_bytes()
        if len(content) > _MAX_SOURCE_BYTES:
            _fail("limit_bytes", ("code", "source", "path"), "code source exceeds four MiB")
        return CodeSourceSnapshot(relative_path, content, _sha256_bytes(content))


@dataclass(frozen=True)
class _CodeBinding:
    source: CodeSourceSnapshot
    profile: PythonEnvironmentProfile
    environment_sha256: str
    timeout_ms: int
    max_result_bytes: int
    max_log_bytes: int
    result_schema: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]


_BINDING_SCHEMA = "unchain.skeleton.code-binding.v1"


_WORKER_SOURCE = r'''
import importlib.util, json, os, pathlib, sys, traceback

MAX_DEPTH=64
MAX_NODES=100000
def strict_json(value, depth=0, seen=None):
    if seen is None: seen=[0]
    seen[0] += 1
    if seen[0] > MAX_NODES or depth > MAX_DEPTH: raise ValueError('JSON structure exceeds limit')
    if value is None or type(value) in (str, bool, int): return
    if type(value) is float:
        if value != value or value in (float('inf'), -float('inf')): raise ValueError('invalid JSON number')
        return
    if type(value) is list:
        for item in value: strict_json(item, depth + 1, seen)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str: raise TypeError('JSON object keys must be strings')
            strict_json(item, depth + 1, seen)
        return
    raise TypeError('value is not JSON')

def write_receipt(path, value):
    target = pathlib.Path(path)
    temporary = target.with_suffix('.tmp')
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    with temporary.open('wb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, target)

request_path, result_path = sys.argv[-2:]
request = json.loads(pathlib.Path(request_path).read_text(encoding='utf-8'))
identity = request['identity']
try:
    spec = importlib.util.spec_from_file_location('unchain_skeleton_user_code', request['source_path'])
    if spec is None or spec.loader is None: raise RuntimeError('source loader unavailable')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    entrypoint = getattr(module, 'main', None)
    if not callable(entrypoint): raise RuntimeError('main(input, context) is required')
    result = entrypoint(request['input'], request['context'])
    if type(result) is not dict:
        outcome = {'status':'error','code':'invalid_type'}
    else:
        try:
            strict_json(result)
            encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
            outcome = {'status':'error','code':'limit_bytes'} if len(encoded) > request['max_result_bytes'] else {'status':'success','result':result}
        except (TypeError, ValueError, OverflowError, RecursionError):
            outcome = {'status':'error','code':'invalid_json'}
    write_receipt(result_path, {'schema':'unchain.skeleton.code-worker.v1','identity':identity,'outcome':outcome})
except BaseException:
    traceback.print_exc(file=sys.stderr)
    raise
'''


class SkeletonCodeRuntime:
    """Execute fixed Python source through an explicitly supplied local profile."""

    def __init__(
        self,
        plan: Any,
        *,
        source_resolver: Callable[[str], CodeSourceSnapshot],
        profile_resolver: Callable[[str], PythonEnvironmentProfile | None],
        staging_root: str | os.PathLike[str],
        poll_interval_s: float = 0.05,
        cancel_grace_s: float = 0.5,
    ) -> None:
        plan = checked_execution_plan(plan)
        if not callable(source_resolver) or not callable(profile_resolver):
            _fail("invalid_type", ("resolver",), "Code runtime resolvers must be callable")
        if isinstance(poll_interval_s, bool) or not isinstance(poll_interval_s, (int, float)) or poll_interval_s <= 0:
            _fail("invalid_value", ("poll_interval_s",), "poll interval must be positive")
        if isinstance(cancel_grace_s, bool) or not isinstance(cancel_grace_s, (int, float)) or cancel_grace_s <= 0:
            _fail("invalid_value", ("cancel_grace_s",), "cancel grace must be positive")
        self.plan = plan
        self.staging_root = Path(staging_root)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        try:
            self.staging_root.chmod(0o700)
        except OSError:
            pass
        self.poll_interval_s = float(poll_interval_s)
        self.cancel_grace_s = float(cancel_grace_s)
        self.source_resolver = source_resolver
        self.profile_resolver = profile_resolver
        self._configs: dict[str, dict[str, Any]] = {}
        self._execution_bindings: dict[
            str,
            tuple[str, dict[str, Any], Callable[[dict[str, Any]], bytes]],
        ] = {}
        snapshot = plan.snapshot()
        for index, node in enumerate(snapshot["nodes"]):
            if node["type"] != "code":
                continue
            path = ("document", "nodes", index, "config", "code")
            config = admit_code_config(node["config"].get("code"), path)
            self._configs[node["id"]] = config

    def for_plan(self, plan: Any) -> "SkeletonCodeRuntime":
        """Bind the same host policy to one fixed child definition."""
        return SkeletonCodeRuntime(
            plan,
            source_resolver=self.source_resolver,
            profile_resolver=self.profile_resolver,
            staging_root=self.staging_root,
            poll_interval_s=self.poll_interval_s,
            cancel_grace_s=self.cancel_grace_s,
        )

    def prepare_execution_bindings(self, attachments: dict[str, dict[str, Any]], prepare_bytes: Callable[..., tuple[dict[str, Any], Any]]):
        """Freeze source/profile identities before the execution's first receipt."""
        bindings: dict[str, dict[str, Any]] = {}
        pending = []
        profiles: dict[str, tuple[PythonEnvironmentProfile, str]] = {}
        for node_id, config in self._configs.items():
            source = self.source_resolver(config["source"]["path"])
            if type(source) is not CodeSourceSnapshot or source.sha256 != _sha256_bytes(source.content):
                _fail("identity_mismatch", ("code", "source"), "source resolver returned an invalid snapshot")
            profile_id = config["environment_profile_id"]
            fixed = profiles.get(profile_id)
            if fixed is None:
                profile = self.profile_resolver(profile_id)
                if type(profile) is not PythonEnvironmentProfile or profile.profile_id != profile_id:
                    _fail("missing_runtime", ("environment_profile_id",), "Python runtime profile is unavailable")
                profile = profile.admitted(); fixed = (profile, profile.fingerprint()); profiles[profile_id] = fixed
            profile, environment_sha256 = fixed
            source_ref, item = prepare_bytes(source.content, media_type="text/x-python", key="code." + node_id + ".source")
            pending.append(item)
            mapped = []
            for artifact in config["artifacts"]:
                attachment = artifact["attachment"]
                if attachment not in attachments:
                    _fail("missing_value", ("code", "artifacts", artifact["name"]), "Code artifact is not granted to this execution")
                mapped.append({"name": artifact["name"], "attachment": attachment, "ref": deepcopy(attachments[attachment])})
            bindings[node_id] = {
                "schema": _BINDING_SCHEMA, "source_ref": source_ref, "source_sha256": source.sha256,
                "profile": {"id": profile_id, "kind": profile.kind, "command": list(profile.command), "sha256": environment_sha256},
                "artifacts": mapped,
            }
        return bindings, tuple(pending)

    def bind_node_execution(
        self,
        node_id: str,
        binding: dict[str, Any],
        read_bytes: Callable[[dict[str, Any]], bytes],
        node_attempt_id: str,
    ) -> None:
        if (node_id not in self._configs or type(binding) is not dict
                or not callable(read_bytes) or type(node_attempt_id) is not str or not node_attempt_id):
            _fail("identity_mismatch", ("code_bindings",), "Code execution bindings are unavailable")
        copied = deepcopy(binding)
        existing = self._execution_bindings.get(node_attempt_id)
        if existing is not None and (existing[0] != node_id or _canonical(existing[1]) != _canonical(copied)):
            _fail("identity_mismatch", ("node_attempt_id",), "Code node attempt cannot be rebound")
        self._execution_bindings[node_attempt_id] = (node_id, copied, read_bytes)

    def __call__(self, context: DurableNodeContext, resolved: dict[str, Any]) -> WorkflowNodeCompleted:
        binding = self.preflight_inputs(context.node_id, resolved, context.node_attempt_id)
        try:
            context.assert_current()
            directory = Path(tempfile.mkdtemp(prefix="code-", dir=self.staging_root))
        except BaseException:
            self._execution_bindings.pop(context.node_attempt_id, None)
            raise
        try:
            source_path = directory / "main.py"
            source_path.write_bytes(binding.source.content)
            try:
                source_path.chmod(0o600)
                directory.chmod(0o700)
            except OSError:
                pass
            if _file_sha256(source_path) != binding.source.sha256:
                _fail("identity_mismatch", ("code", "source"), "staged Code source changed")
            staged_artifacts: dict[str, dict[str, Any]] = {}
            artifacts_directory = directory / "artifacts"
            for artifact in binding.artifacts:
                content = context.read_attachment(artifact["attachment"])
                ref = artifact["ref"]
                if len(content) != ref["byte_length"] or _sha256_bytes(content) != ref["sha256"]:
                    _fail("identity_mismatch", ("code", "artifacts", artifact["name"]), "Code artifact changed after it was bound")
                artifacts_directory.mkdir(mode=0o700, exist_ok=True)
                target = artifacts_directory / artifact["name"]
                target.write_bytes(content)
                target.chmod(0o600)
                staged_artifacts[artifact["name"]] = {
                    "path": str(target), "sha256": ref["sha256"],
                    "byte_length": ref["byte_length"], "media_type": ref["media_type"],
                }
            identity = {
                "document_id": context.document_id, "graph_id": context.graph_id,
                "node_id": context.node_id, "node_invocation_id": context.node_invocation_id,
                "node_attempt_id": context.node_attempt_id, "code_sha256": binding.source.sha256,
                "environment_sha256": binding.environment_sha256,
            }
            request = {
                "schema": "unchain.skeleton.code-request.v1", "identity": identity,
                "source_path": str(source_path), "input": deepcopy(resolved),
                "context": {
                    "document_id": context.document_id, "graph_id": context.graph_id,
                    "node_id": context.node_id, "node_type": context.node_type,
                    "node_invocation_id": context.node_invocation_id,
                    "node_attempt_id": context.node_attempt_id, "work_directory": str(directory),
                    "artifacts": staged_artifacts,
                },
                "max_result_bytes": binding.max_result_bytes,
            }
            request_path = directory / "request.json"; result_path = directory / "result.json"
            request_path.write_bytes(_canonical(request))
            command = self._command(binding.profile, request_path, result_path)
            # This is the final durable ownership check before a process can exist.
            context.assert_current()
            popen_kwargs: dict[str, Any] = {"cwd": str(directory), "shell": False, "stdin": subprocess.DEVNULL, "close_fds": True}
            if os.name == "nt":  # pragma: no cover - Windows CI/release evidence
                popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs)
            stdout_capture = self._start_log_drain(process.stdout, binding.max_log_bytes)
            stderr_capture = self._start_log_drain(process.stderr, binding.max_log_bytes)
            deadline = time.monotonic() + binding.timeout_ms / 1000
            try:
                while process.poll() is None:
                    context.assert_current()
                    if time.monotonic() >= deadline:
                        self._terminate_tree(process)
                        raise RuntimeError("Code process timed out; external outcome is unknown")
                    time.sleep(self.poll_interval_s)
                context.assert_current()
                # The worker has exited. Reap any descendants that inherited its
                # private process group before accepting the receipt or closing logs.
                self._terminate_tree(process)
            except BaseException:
                self._terminate_tree(process)
                raise
            finally:
                stdout_capture["thread"].join(timeout=self.cancel_grace_s + 1)
                stderr_capture["thread"].join(timeout=self.cancel_grace_s + 1)
            logs = {
                **self._captured_log(stdout_capture, "stdout"),
                **self._captured_log(stderr_capture, "stderr"),
            }
            if process.returncode != 0 or not result_path.is_file():
                raise RuntimeError("Code process exited without an authenticated result")
            if result_path.stat().st_size > binding.max_result_bytes + 64 * 1024:
                raise RuntimeError("Code receipt exceeds its authenticated result allowance")
            try:
                receipt = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
                raise RuntimeError("Code process returned an invalid receipt") from exc
            self._admit_receipt(receipt, identity)
            outcome = receipt["outcome"]
            public = {
                "schema": _PUBLIC_RECEIPT_SCHEMA, "result": None,
                "code_sha256": binding.source.sha256,
                "environment_sha256": binding.environment_sha256,
                "logs": logs,
            }
            if outcome["status"] == "error":
                public["error"] = {"code": outcome["code"]}
            else:
                result = outcome["result"]
                _validate_json(result, ("code", "result"))
                if type(result) is not dict:
                    _fail("invalid_type", ("code", "result"), "Code result must be an object")
                if next(Draft202012Validator(binding.result_schema).iter_errors(result), None) is not None:
                    public["error"] = {"code": "output_invalid"}
                else:
                    public["result"] = deepcopy(result)
                    public["error"] = None
            return WorkflowNodeCompleted({"_skeleton_code": public})
        finally:
            shutil.rmtree(directory, ignore_errors=True)
            self._execution_bindings.pop(context.node_attempt_id, None)

    def preflight_inputs(self, node_id: str, resolved: dict[str, Any], node_attempt_id: str) -> _CodeBinding:
        config = self._configs.get(node_id)
        bound = self._execution_bindings.get(node_attempt_id)
        if config is None or bound is None or bound[0] != node_id:
            _fail("identity_mismatch", ("node_id",), "Code node attempt is not in the fixed execution")
        raw = bound[1]
        read_bytes = bound[2]
        _validate_json(resolved, ("resolved_inputs",))
        if type(resolved) is not dict:
            _fail("invalid_type", ("resolved_inputs",), "Code input must be an object")
        source_content = read_bytes(raw["source_ref"])
        if _sha256_bytes(source_content) != raw["source_sha256"]:
            _fail("identity_mismatch", ("code", "source"), "persisted Code source does not match its identity")
        profile_id = raw["profile"]["id"]
        profile = self.profile_resolver(profile_id)
        if type(profile) is not PythonEnvironmentProfile or profile.profile_id != profile_id:
            _fail("environment_changed", ("environment",), "Python runtime profile is unavailable after binding")
        profile = profile.admitted()
        if (profile.kind != raw["profile"]["kind"] or list(profile.command) != raw["profile"]["command"]
                or profile.fingerprint() != raw["profile"]["sha256"]):
            _fail("environment_changed", ("environment",), "Python runtime profile changed after binding")
        return _CodeBinding(
            CodeSourceSnapshot(config["source"]["path"], source_content, raw["source_sha256"]), profile,
            raw["profile"]["sha256"], config["timeout_ms"], config["max_result_bytes"],
            config["max_log_bytes"], config["result_schema"], tuple(deepcopy(raw["artifacts"])),
        )

    @staticmethod
    def _command(profile: PythonEnvironmentProfile, request: Path, result: Path) -> list[str]:
        if profile.kind == "venv" or (profile.kind == "bundled" and len(profile.command) == 1):
            return [profile.command[0], "-I", "-c", _WORKER_SOURCE, str(request), str(result)]
        return [*profile.command, str(request), str(result)]

    @staticmethod
    def _start_log_drain(pipe: Any, limit: int) -> dict[str, Any]:
        capture: dict[str, Any] = {"data": bytearray(), "truncated": False}
        def drain() -> None:
            try:
                while True:
                    chunk = pipe.read(64 * 1024)
                    if not chunk:
                        break
                    remaining = limit - len(capture["data"])
                    if remaining > 0:
                        capture["data"].extend(chunk[:remaining])
                    if len(chunk) > max(0, remaining):
                        capture["truncated"] = True
            finally:
                pipe.close()
        thread = threading.Thread(target=drain, name="skeleton-code-log-drain", daemon=True)
        capture["thread"] = thread
        thread.start()
        return capture

    @staticmethod
    def _captured_log(capture: dict[str, Any], name: str) -> dict[str, Any]:
        return {
            name: bytes(capture["data"]).decode("utf-8", "replace"),
            name + "_truncated": bool(capture["truncated"]),
        }

    def _terminate_tree(self, process: subprocess.Popen[Any]) -> None:
        if os.name == "nt":  # pragma: no cover - Windows CI/release evidence
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        else:
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
            try: process.wait(timeout=self.cancel_grace_s)
            except subprocess.TimeoutExpired: pass
            # A parent can exit on TERM while an inherited child ignores it. The
            # process group remains ours until it no longer exists, so clean it
            # independently of the leader's lifecycle.
            try:
                os.killpg(process.pid, 0)
                os.killpg(process.pid, signal.SIGTERM)
                time.sleep(self.cancel_grace_s)
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try: process.wait(timeout=self.cancel_grace_s)
        except subprocess.TimeoutExpired: process.kill(); process.wait()

    @staticmethod
    def _admit_receipt(value: Any, identity: dict[str, Any]) -> None:
        if type(value) is not dict or set(value) != {"schema", "identity", "outcome"} or value.get("schema") != _RECEIPT_SCHEMA:
            raise RuntimeError("Code receipt envelope is invalid")
        if value["identity"] != identity:
            raise RuntimeError("Code receipt identity does not match the invocation")
        outcome = value["outcome"]
        if type(outcome) is not dict or outcome.get("status") not in {"success", "error"}:
            raise RuntimeError("Code receipt outcome is invalid")
        if outcome["status"] == "success":
            if set(outcome) != {"status", "result"}:
                raise RuntimeError("Code success receipt fields are invalid")
        elif set(outcome) != {"status", "code"} or outcome["code"] not in {"invalid_type", "invalid_json", "limit_bytes"}:
            raise RuntimeError("Code error receipt fields are invalid")


def admit_code_config(value: Any, path: tuple[str | int, ...]) -> dict[str, Any]:
    """Admit the closed portable portion of one Code node configuration."""
    if type(value) is not dict:
        _fail("missing_field", path, "official Code runtime requires config.code")
    required = {"source", "environment_profile_id", "timeout_ms", "max_result_bytes", "max_log_bytes", "result_schema"}
    allowed = required | {"artifacts"}
    if not required <= set(value) or not set(value) <= allowed:
        unknown = next(iter(set(value) - allowed), None)
        _fail("unknown_field" if unknown else "missing_field", path + ((unknown,) if unknown else ()), "Code config fields are invalid")
    source = value["source"]
    if type(source) is not dict or set(source) != {"kind", "path"} or source.get("kind") != "file" or type(source.get("path")) is not str:
        _fail("invalid_shape", path + ("source",), "Code source must be one file reference")
    if type(value["environment_profile_id"]) is not str or not value["environment_profile_id"]:
        _fail("invalid_value", path + ("environment_profile_id",), "environment profile id is required")
    for key, ceiling in (("timeout_ms", 86_400_000), ("max_result_bytes", 8 * 1024 * 1024), ("max_log_bytes", 8 * 1024 * 1024)):
        item = value[key]
        if type(item) is not int or isinstance(item, bool) or not 1 <= item <= ceiling:
            _fail("invalid_value", path + (key,), key + " is outside the supported range")
    _validate_json(value["result_schema"], path + ("result_schema",))
    if type(value["result_schema"]) is not dict or value["result_schema"].get("type") != "object":
        _fail("invalid_schema", path + ("result_schema",), "Code result schema must describe an object")
    try:
        Draft202012Validator.check_schema(value["result_schema"])
    except (SchemaError, TypeError, ValueError) as exc:
        raise SkeletonValidationError("invalid_schema", path + ("result_schema",), "Code result schema is invalid") from exc
    artifacts = value.get("artifacts", [])
    if type(artifacts) is not list:
        _fail("invalid_type", path + ("artifacts",), "Code artifacts must be an array")
    names = set()
    admitted_artifacts = []
    for index, artifact in enumerate(artifacts):
        item_path = path + ("artifacts", index)
        if type(artifact) is not dict or set(artifact) != {"name", "attachment"}:
            _fail("invalid_shape", item_path, "Code artifact mapping is invalid")
        name, attachment = artifact["name"], artifact["attachment"]
        if type(name) is not str:
            _fail("invalid_type", item_path + ("name",), "Code artifact name must be a string")
        if not name or name in {".", ".."} or len(name) > 128 or any(
                character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in name):
            _fail("invalid_value", item_path + ("name",), "Code artifact name is invalid")
        if name in names:
            _fail("duplicate_name", item_path + ("name",), "Code artifact name is duplicated")
        if type(attachment) is not str or not attachment:
            _fail("invalid_value", item_path + ("attachment",), "Code artifact attachment is invalid")
        names.add(name); admitted_artifacts.append({"name": name, "attachment": attachment})
    admitted = deepcopy(value); admitted["artifacts"] = admitted_artifacts
    return admitted
