from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..kernel.types import ToolCall
from ..tools.common import emit_loop_event
from ..tools.confirmation import prepare_tool_confirmation
from ..tools.models import ToolConfirmationResponse, ToolExecutionContext
from ..tools.receipts import (
    DurableToolReceiptIntegrityError,
    build_durable_tool_receipt,
    durable_tool_receipt_state_update,
    validate_durable_tool_receipt,
)
from ..tools.runtime import ToolRuntimeOutcome, ToolRuntimePlugin
from .environment import JobEnvironmentProfile
from .models import (
    DurableJobConflictError,
    DurableJobNotFoundError,
    DurableJobOwnershipError,
    DurableJobSnapshot,
)
from .store import STORE_MANIFEST_SCHEMA_VERSION

if TYPE_CHECKING:
    from .process import ProcessJobSupervisor


_DURABLE_JOB_PREFIX = "job_"
_SHELL_ACTIONS = frozenset({"run", "poll", "wait", "kill"})
_DURABLE_JOB_RECEIPT_SOURCE = "unchain.jobs.durable_shell"


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class DurableShellJobPlugin(ToolRuntimePlugin):
    """Route durable background ``shell`` calls through a process supervisor.

    The plugin deliberately does not register or replace the shell tool.  The
    original tool therefore remains the source of its schema and confirmation
    policy.  Only approved background starts and ``job_`` lifecycle calls are
    intercepted here; all legacy foreground and process-local tasks continue
    through the normal toolkit execution path.
    """

    supervisor: ProcessJobSupervisor
    _approved_intents: dict[tuple[str, str], tuple[str, "_ShellJobIntent"]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def can_handle(self, *, tool_call: ToolCall, context: Any) -> bool:
        if tool_call.name != "shell":
            return False
        if self._source_shell_runtime_or_none(context) is None:
            return False
        arguments = tool_call.arguments
        if not isinstance(arguments, dict):
            return False
        action = str(arguments.get("action") or "").strip().lower()
        if action == "run":
            return bool(arguments.get("run_in_background"))
        if action not in {"poll", "wait", "kill"}:
            return False
        task_id = str(arguments.get("task_id") or "")
        return task_id.startswith(_DURABLE_JOB_PREFIX)

    def durable_runtime_manifest(
        self,
        *,
        tool_call: ToolCall,
        context: Any,
    ) -> dict[str, Any]:
        """Bind approval replay to this store and normalized shell intent."""

        self._validate_runtime_session(
            context,
            execution_id=self._execution_id(context),
        )
        store = getattr(self.supervisor, "store", None)
        base_dir = getattr(store, "base_dir", None)
        store_id = getattr(store, "store_id", None)
        if base_dir is None or not isinstance(store_id, str) or not store_id:
            raise RuntimeError(
                "durable shell job supervisor has no stable store namespace"
            )
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        action = str(arguments.get("action") or "").strip().lower()
        logical_execution_id, _ = self._runtime_lineage_fields(context)
        operation: dict[str, Any] = {
            "action": action,
            "task_id": str(arguments.get("task_id") or ""),
        }
        if action == "run":
            cache_key = (self._execution_id(context), tool_call.call_id)
            self._approved_intents.pop(cache_key, None)
            try:
                intent = self._resolve_job_intent(
                    execution_id=cache_key[0],
                    arguments=arguments,
                    context=context,
                )
                if cache_key not in self._approved_intents:
                    while len(self._approved_intents) >= 256:
                        self._approved_intents.pop(next(iter(self._approved_intents)))
                self._approved_intents[cache_key] = (
                    _canonical_digest(arguments),
                    intent,
                )
                operation = {
                    "action": "run",
                    "intent_digest": intent.intent_digest,
                    "environment_digest": intent.environment_digest,
                }
            except Exception as exc:
                operation = {
                    "action": "run",
                    "invalid_intent_digest": _canonical_digest(
                        {
                            "arguments": arguments,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    ),
                }
        if logical_execution_id:
            operation["logical_execution_id"] = logical_execution_id
        path_digest = hashlib.sha256(str(base_dir).encode("utf-8")).hexdigest()
        store_identity = {
            "backend": "json_file",
            "manifest_schema_version": STORE_MANIFEST_SCHEMA_VERSION,
            "store_id": store_id,
            "base_dir_digest": path_digest,
        }
        manifest = {
            "handler": "durable_shell_job",
            "protocol_version": 1,
            "terminal_handler": True,
            "adapter": "shell",
            "store": store_identity,
            "store_namespace_digest": _canonical_digest(store_identity),
            "job_identity": (
                "execution_id+logical_execution_id+shell_call_id:v2"
                if logical_execution_id
                else "execution_id+shell_call_id:v1"
            ),
            "operation": operation,
        }
        if logical_execution_id:
            manifest["logical_execution_id"] = logical_execution_id
        return manifest

    def execute(self, *, tool_call: ToolCall, context: Any) -> ToolRuntimeOutcome:
        original_arguments = (
            tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
        )
        action = str(original_arguments.get("action") or "").strip().lower()
        original_action = action
        requested_task_id = str(original_arguments.get("task_id") or "")
        should_observe = False
        try:
            effective_arguments, should_observe, denied = self._approved_arguments(
                tool_call=tool_call,
                context=context,
            )
            if denied is not None:
                return ToolRuntimeOutcome(
                    handled=True,
                    tool_result=denied,
                    should_observe=should_observe,
                )

            action = str(effective_arguments.get("action") or "").strip().lower()
            requested_task_id = str(
                effective_arguments.get("task_id") or requested_task_id
            )
            if action not in _SHELL_ACTIONS:
                raise ValueError("action must be one of: run, poll, wait, kill")
            execution_id = self._execution_id(context)
            if not execution_id:
                raise ValueError("durable background shell jobs require a session_id")

            if original_action == "run":
                if not bool(effective_arguments.get("run_in_background")):
                    raise ValueError(
                        "approved modified arguments must keep "
                        "run_in_background=true for durable execution"
                    )
                result = self._start(
                    execution_id=execution_id,
                    call_id=tool_call.call_id,
                    arguments=effective_arguments,
                    context=context,
                )
                state_updates: dict[str, Any] = {}
            else:
                result, state_updates = self._lifecycle(
                    action=action,
                    execution_id=execution_id,
                    call_id=tool_call.call_id,
                    arguments=effective_arguments,
                    context=context,
                )
            return ToolRuntimeOutcome(
                handled=True,
                tool_result=result,
                state_updates=state_updates,
                should_observe=should_observe,
            )
        except Exception as exc:
            return ToolRuntimeOutcome(
                handled=True,
                tool_result=self._error_result(
                    action=action,
                    error=exc,
                    task_id=requested_task_id,
                ),
                should_observe=should_observe,
            )
        finally:
            if action == "run":
                self._approved_intents.pop(
                    (self._execution_id(context), tool_call.call_id),
                    None,
                )

    def _approved_arguments(
        self,
        *,
        tool_call: ToolCall,
        context: Any,
    ) -> tuple[dict[str, Any], bool, dict[str, Any] | None]:
        toolkit = context.toolkit
        execution_context = ToolExecutionContext(
            session_id=str(getattr(context, "session_id", "") or ""),
            run_id=str(getattr(context, "run_id", "") or ""),
            provider=str(getattr(context, "provider", "") or ""),
            model=str(getattr(context, "model", "") or ""),
            iteration=int(getattr(context, "iteration", 0) or 0),
            memory_namespace=str(getattr(context, "memory_namespace", "") or ""),
            tool_runtime_config=(
                dict(context.event.get("tool_runtime_config") or {})
                if isinstance(context.event.get("tool_runtime_config"), dict)
                else {}
            ),
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            turn_id=f"{context.run_id}:turn-{context.iteration}",
        )
        preparation = prepare_tool_confirmation(
            toolkit=toolkit,
            tool_call=tool_call,
            execution_context=execution_context,
            execution_guard=getattr(context, "execution_guard", None),
        )
        resolver_error = preparation.to_resolver_error_outcome(tool_name=tool_call.name)
        if resolver_error is not None:
            return (
                {},
                preparation.should_observe,
                copy.deepcopy(resolver_error.tool_result),
            )

        effective = copy.deepcopy(preparation.effective_arguments)
        if not isinstance(effective, dict):
            raise TypeError("shell arguments must be an object")

        raw_response: Any = None
        response_supplied = False
        raw_event = getattr(context, "raw_event", {})
        if (
            preparation.needs_confirmation_response
            and isinstance(raw_event, dict)
            and "interaction_response" in raw_event
        ):
            raw_response = raw_event.get("interaction_response")
            response_supplied = True
        elif preparation.needs_confirmation_response:
            callback = context.event.get("on_tool_confirm")
            if callable(callback):
                raw_response = callback(preparation.request)
                response_supplied = True

        if not response_supplied:
            return effective, preparation.should_observe, None

        response = ToolConfirmationResponse.from_raw(raw_response)
        guard = getattr(context, "execution_guard", None)
        if guard is not None:
            guard.renew()
        if not response.approved:
            reason = response.reason or "User denied execution."
            emit_loop_event(
                context.loop,
                context.callback,
                "tool_denied",
                context.run_id,
                iteration=context.iteration,
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
                reason=reason,
            )
            return (
                effective,
                preparation.should_observe,
                {"denied": True, "tool": tool_call.name, "reason": reason},
            )
        if response.modified_arguments is not None:
            if not isinstance(response.modified_arguments, dict):
                raise TypeError("modified shell arguments must be an object")
            effective = copy.deepcopy(response.modified_arguments)
            emit_loop_event(
                context.loop,
                context.callback,
                "tool_confirmed",
                context.run_id,
                iteration=context.iteration,
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
            )
        return effective, preparation.should_observe, None

    def _start(
        self,
        *,
        execution_id: str,
        call_id: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> dict[str, Any]:
        self._validate_runtime_session(context, execution_id=execution_id)
        cached = self._approved_intents.pop((execution_id, call_id), None)
        arguments_digest = _canonical_digest(arguments)
        if cached is not None and cached[0] == arguments_digest:
            intent = cached[1]
        else:
            intent = self._resolve_job_intent(
                execution_id=execution_id,
                arguments=arguments,
                context=context,
                environment_profile=(
                    cached[1].environment_profile if cached is not None else None
                ),
            )
        logical_execution_id, created_by_attempt_id = self._runtime_lineage_fields(
            context
        )
        idempotency_key = (
            "shell:v2:"
            + _canonical_digest(
                {
                    "schema_version": 2,
                    "logical_execution_id": logical_execution_id,
                    "call_id": call_id,
                }
            )
            if logical_execution_id
            else f"shell:{call_id}"
        )
        snapshot = self.supervisor.start(
            execution_id=execution_id,
            idempotency_key=idempotency_key,
            argv=intent.argv,
            cwd=intent.cwd,
            timeout_ms=intent.timeout_ms,
            intent_digest=intent.intent_digest,
            adapter="shell",
            logical_execution_id=logical_execution_id,
            created_by_attempt_id=created_by_attempt_id,
        )
        return self._snapshot_result(
            snapshot,
            action="run",
            shell_family=intent.shell_family,
            platform=intent.platform,
            cwd=intent.cwd,
        )

    def _resolve_job_intent(
        self,
        *,
        execution_id: str,
        arguments: dict[str, Any],
        context: Any,
        environment_profile: JobEnvironmentProfile | None = None,
    ) -> "_ShellJobIntent":
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command is required")
        shell_runtime = self._source_shell_runtime(context)
        resolved_cwd, cwd_error = shell_runtime.resolve_cwd(
            execution_id,
            arguments.get("cwd"),
        )
        if cwd_error or resolved_cwd is None:
            raise ValueError(cwd_error or "cwd could not be resolved")
        spec = shell_runtime.detect_executor()
        timeout_ms = shell_runtime._normalize_timeout_ms(arguments.get("timeout_ms"))
        argv = [*spec.argv, command]
        cwd = str(resolved_cwd)
        resolved_environment_profile = (
            environment_profile or self.supervisor.environment_profile
        )
        environment_digest = resolved_environment_profile.digest
        return _ShellJobIntent(
            argv=argv,
            cwd=cwd,
            timeout_ms=timeout_ms,
            shell_family=spec.family,
            platform=spec.platform,
            environment_profile=resolved_environment_profile,
            environment_digest=environment_digest,
            intent_digest=_canonical_digest(
                {
                    "adapter": "shell",
                    "argv": argv,
                    "cwd": cwd,
                    "timeout_ms": timeout_ms,
                    "environment_digest": environment_digest,
                }
            ),
        )

    def _lifecycle(
        self,
        *,
        action: str,
        execution_id: str,
        call_id: str,
        arguments: dict[str, Any],
        context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._validate_runtime_session(context, execution_id=execution_id)
        task_id = arguments.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError("task_id is required")
        if not task_id.startswith(_DURABLE_JOB_PREFIX):
            raise ValueError("durable shell lifecycle task_id must start with 'job_'")
        shell_runtime = self._source_shell_runtime(context)
        max_output_chars = shell_runtime._normalize_max_output_chars(
            arguments.get("max_output_chars")
        )
        lineage = self._receipt_lineage(
            context,
            job_id=task_id,
            execution_id=execution_id,
        )
        if lineage is not None:
            _, current_created_by_attempt_id = self._runtime_lineage_fields(context)
            request_digest = _canonical_digest(
                {
                    "tool": "shell",
                    "arguments": arguments,
                }
            )
            delivery_key = f"shell:{call_id}"
            current_attempt_id = str(getattr(context, "run_id", "") or "").strip()
            if action == "poll":
                receipt = self.supervisor.poll_with_receipt(
                    execution_id=execution_id,
                    job_id=task_id,
                    delivery_key=delivery_key,
                    operation_digest=request_digest,
                    logical_execution_id=lineage.logical_execution_id,
                    created_by_attempt_id=current_created_by_attempt_id,
                    source_attempt_id=current_attempt_id,
                    max_output_chars=max_output_chars,
                )
            elif action == "wait":
                timeout_ms = shell_runtime._normalize_timeout_ms(
                    arguments.get("timeout_ms")
                )
                receipt = self.supervisor.wait_with_receipt(
                    execution_id=execution_id,
                    job_id=task_id,
                    delivery_key=delivery_key,
                    operation_digest=request_digest,
                    logical_execution_id=lineage.logical_execution_id,
                    created_by_attempt_id=current_created_by_attempt_id,
                    source_attempt_id=current_attempt_id,
                    timeout_ms=timeout_ms,
                    max_output_chars=max_output_chars,
                )
            else:
                receipt = self.supervisor.cancel_with_receipt(
                    execution_id=execution_id,
                    job_id=task_id,
                    delivery_key=delivery_key,
                    operation_digest=request_digest,
                    logical_execution_id=lineage.logical_execution_id,
                    created_by_attempt_id=current_created_by_attempt_id,
                    source_attempt_id=current_attempt_id,
                    wait_timeout_ms=2_000,
                    max_output_chars=max_output_chars,
                    receipt_action="kill",
                )
            bridge_receipt = build_durable_tool_receipt(
                source=_DURABLE_JOB_RECEIPT_SOURCE,
                receipt_id=receipt.receipt_id,
                execution_id=receipt.execution_id,
                logical_execution_id=receipt.logical_execution_id,
                created_by_attempt_id=receipt.created_by_attempt_id,
                tool_call_id=call_id,
                request_digest=request_digest,
                result_digest=receipt.result_digest,
                payload={
                    "job_receipt": receipt.runtime_metadata(),
                    "stdout_offset": receipt.result["stdout_offset"],
                    "stderr_offset": receipt.result["stderr_offset"],
                    "next_stdout_offset": receipt.result["next_stdout_offset"],
                    "next_stderr_offset": receipt.result["next_stderr_offset"],
                },
            )
            result = receipt.result
            result.setdefault("task_id", task_id)
            result.setdefault("job_id", task_id)
            result.setdefault("durable", True)
            result.setdefault("background", True)
            return (
                result,
                durable_tool_receipt_state_update(context.state, bridge_receipt),
            )

        if action == "poll":
            result = self.supervisor.poll(
                execution_id=execution_id,
                job_id=task_id,
                max_output_chars=max_output_chars,
            )
        elif action == "wait":
            timeout_ms = shell_runtime._normalize_timeout_ms(arguments.get("timeout_ms"))
            result = self.supervisor.wait(
                execution_id=execution_id,
                job_id=task_id,
                timeout_ms=timeout_ms,
                max_output_chars=max_output_chars,
            )
        else:
            result = self.supervisor.cancel(
                execution_id=execution_id,
                job_id=task_id,
                wait_timeout_ms=2_000,
                max_output_chars=max_output_chars,
            )
            result["action"] = "kill"
        result.setdefault("task_id", task_id)
        result.setdefault("job_id", task_id)
        result.setdefault("durable", True)
        result.setdefault("background", True)
        return result, {}

    def acknowledge_durable_receipt(
        self,
        *,
        receipt: dict[str, Any],
        application: dict[str, Any],
    ) -> bool:
        """Commit a prepared job cursor after its session CAS is durable."""

        validated = validate_durable_tool_receipt(receipt)
        if validated["source"] != _DURABLE_JOB_RECEIPT_SOURCE:
            return False
        payload = validated["payload"]
        job_receipt = payload.get("job_receipt")
        if not isinstance(job_receipt, dict):
            raise DurableToolReceiptIntegrityError(
                "durable shell receipt is missing job receipt metadata"
            )
        store = getattr(self.supervisor, "store", None)
        store_id = str(getattr(store, "store_id", "") or "")
        if str(job_receipt.get("store_id") or "") != store_id:
            return False
        if application.get("receipt") != validated:
            raise DurableToolReceiptIntegrityError(
                "durable shell receipt application does not match its receipt"
            )
        application_id = str(application.get("application_id") or "").strip()
        if not application_id:
            raise DurableToolReceiptIntegrityError(
                "durable shell receipt application_id is required"
            )
        expected = {
            "receipt_id": validated["receipt_id"],
            "execution_id": validated["execution_id"],
            "logical_execution_id": validated["logical_execution_id"],
            "created_by_attempt_id": validated["created_by_attempt_id"],
            "tool_call_id": validated["tool_call_id"],
            "request_digest": validated["request_digest"],
            "result_digest": str(job_receipt.get("result_digest") or ""),
        }
        actual = {
            "receipt_id": str(job_receipt.get("receipt_id") or ""),
            "execution_id": str(job_receipt.get("execution_id") or ""),
            "logical_execution_id": str(
                job_receipt.get("logical_execution_id") or ""
            ),
            "created_by_attempt_id": str(
                job_receipt.get("created_by_attempt_id") or ""
            ),
            "tool_call_id": str(job_receipt.get("delivery_key") or "").removeprefix(
                "shell:"
            ),
            "request_digest": str(job_receipt.get("operation_digest") or ""),
            "result_digest": validated["result_digest"],
        }
        if actual != expected:
            raise DurableToolReceiptIntegrityError(
                "durable shell receipt identity does not match its job receipt"
            )
        self.supervisor.acknowledge_result_receipt(
            str(job_receipt.get("job_id") or ""),
            execution_id=str(job_receipt.get("execution_id") or ""),
            receipt_id=validated["receipt_id"],
            applied_checkpoint_id=application_id,
            expected_receipt_digest=str(job_receipt.get("receipt_digest") or ""),
            expected_result_digest=validated["result_digest"],
            expected_delivery_key=str(job_receipt.get("delivery_key") or ""),
            expected_operation_digest=validated["request_digest"],
        )
        return True

    def _receipt_lineage(
        self,
        context: Any,
        *,
        job_id: str,
        execution_id: str,
    ) -> Any | None:
        state = getattr(context, "state", None)
        memory_state = getattr(state, "memory_state", None)
        if (
            not execution_id
            or not isinstance(memory_state, dict)
            or memory_state.get("session_revision_supported") is not True
        ):
            return None
        lineage_store = getattr(self.supervisor, "lineage", None)
        load = getattr(lineage_store, "load", None)
        if not callable(load):
            return None
        lineage = load(job_id, execution_id=execution_id)
        if lineage is None:
            return None
        logical_execution_id, _ = self._runtime_lineage_fields(context)
        if logical_execution_id != lineage.logical_execution_id:
            raise DurableJobConflictError(
                "durable job does not belong to the current logical execution"
            )
        if not lineage.created_by_attempt_id:
            raise DurableJobConflictError(
                "durable job lineage has no creation attempt for receipt delivery"
            )
        return lineage

    @staticmethod
    def _runtime_lineage_fields(context: Any) -> tuple[str, str]:
        config = DurableShellJobPlugin._runtime_job_config(context)
        return (
            str(config.get("logical_execution_id") or "").strip(),
            str(config.get("created_by_attempt_id") or "").strip(),
        )

    @staticmethod
    def _runtime_job_config(context: Any) -> dict[str, Any]:
        event = getattr(context, "event", None)
        runtime_config = (
            event.get("tool_runtime_config")
            if isinstance(event, dict)
            and isinstance(event.get("tool_runtime_config"), dict)
            else {}
        )
        namespaced = runtime_config.get("durable_jobs")
        config = namespaced if isinstance(namespaced, dict) else runtime_config
        return config

    @staticmethod
    def _validate_runtime_session(context: Any, *, execution_id: str) -> None:
        configured_session_id = str(
            DurableShellJobPlugin._runtime_job_config(context).get("session_id")
            or ""
        ).strip()
        if configured_session_id and configured_session_id != execution_id:
            raise DurableJobConflictError(
                "durable jobs runtime session does not match the active session"
            )

    @staticmethod
    def _execution_id(context: Any) -> str:
        return str(getattr(context, "session_id", "") or "").strip()

    @staticmethod
    def _source_shell_runtime(context: Any) -> Any:
        shell_runtime = DurableShellJobPlugin._source_shell_runtime_or_none(context)
        if shell_runtime is None:
            raise RuntimeError("durable shell jobs require the builtin CoreToolkit shell")
        return shell_runtime

    @staticmethod
    def _source_shell_runtime_or_none(context: Any) -> Any | None:
        toolkit = getattr(context, "toolkit", None)
        get_tool = getattr(toolkit, "get", None)
        tool_obj = get_tool("shell") if callable(get_tool) else None
        owner = getattr(getattr(tool_obj, "func", None), "__self__", None)
        from ..toolkits.builtin.core import CoreToolkit

        if not isinstance(owner, CoreToolkit):
            return None
        backend = getattr(owner, "_coding_backend", None)
        shell_runtime = getattr(backend, "_shell_runtime", None)
        return shell_runtime

    @staticmethod
    def _snapshot_result(
        snapshot: DurableJobSnapshot,
        *,
        action: str,
        shell_family: str,
        platform: str,
        cwd: str,
    ) -> dict[str, Any]:
        return {
            # An accepted non-terminal job is a successful start operation;
            # ``snapshot.ok`` intentionally describes only a completed exit 0.
            "ok": not snapshot.completed or snapshot.ok,
            "action": action,
            "status": snapshot.status,
            "shell_family": shell_family,
            "platform": platform,
            "cwd": cwd,
            "task_id": snapshot.job_id,
            "job_id": snapshot.job_id,
            "execution_id": snapshot.execution_id,
            "durable": True,
            "background": True,
            "stdout": "",
            "stderr": "",
            "completed": snapshot.completed,
            "returncode": snapshot.returncode,
            "timed_out": snapshot.timed_out,
            "cancelled": snapshot.cancelled,
            "outcome_unknown": snapshot.status == "outcome_unknown",
            "outcome_unknown_reason": snapshot.outcome_unknown_reason,
            "error": snapshot.error,
            "truncated": snapshot.stdout_truncated or snapshot.stderr_truncated,
            "stdout_offset": 0,
            "stderr_offset": 0,
            "next_stdout_offset": 0,
            "next_stderr_offset": 0,
            "offset_unit": "utf8_bytes",
        }

    @staticmethod
    def _error_result(
        *,
        action: str,
        error: Exception,
        task_id: str = "",
    ) -> dict[str, Any]:
        missing = isinstance(
            error,
            (DurableJobNotFoundError, DurableJobOwnershipError),
        )
        code = str(getattr(error, "code", "") or "")
        return {
            "ok": False,
            "action": action or "run",
            "status": "missing" if missing else "error",
            "task_id": task_id,
            "job_id": task_id,
            "durable": True,
            "background": True,
            "stdout": "",
            "stderr": "",
            "completed": bool(missing),
            "returncode": None,
            "timed_out": False,
            "cancelled": False,
            "outcome_unknown": False,
            "outcome_unknown_reason": "",
            "error": str(error),
            "error_code": code,
            "truncated": False,
            "stdout_offset": 0,
            "stderr_offset": 0,
            "next_stdout_offset": 0,
            "next_stderr_offset": 0,
            "offset_unit": "utf8_bytes",
        }


@dataclass(frozen=True)
class _ShellJobIntent:
    argv: list[str]
    cwd: str
    timeout_ms: int
    shell_family: str
    platform: str
    environment_profile: JobEnvironmentProfile = field(repr=False)
    environment_digest: str
    intent_digest: str


__all__ = ["DurableShellJobPlugin"]
