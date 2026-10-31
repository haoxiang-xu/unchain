"""Deterministic Tool Call and Skill Read hosts for durable Skeleton graphs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from unchain.interaction.durable import (
    InteractionRequest,
    build_interaction_receipt,
    build_interaction_request,
)
from unchain.interaction.runtime import normalize_interaction_response
from unchain.interaction.requests import ensure_interaction_binding_matches
from unchain.kernel.types import ToolCall
from unchain.skills.models import LoadedSkill
from unchain.artifacts import is_failed_tool_result
from unchain.tools.confirmation import execute_confirmable_tool_call, prepare_tool_confirmation
from unchain.tools.models import ToolExecutionContext
from unchain.tools.toolkit import Toolkit

from .durable_execution import DurableNodeContext, WorkflowNodeCompleted, WorkflowNodeWaiting
from .agent_preparation import checked_execution_plan
from .errors import SkeletonValidationError
from .io import _validate_json


_TOOL_WAIT_SCHEMA = "unchain.skeleton_tool_approval_wait.v1"


class _NodeExecutionGuard:
    """Adapt durable node ownership to the formal Tool executor guard contract."""

    def __init__(self, assert_current: Callable[[], Any]) -> None:
        self._assert_current = assert_current

    def assert_active(self) -> Any:
        return self._assert_current()

    def renew(self) -> Any:
        return self._assert_current()


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _digest(value: Any) -> str:
    from unchain.interaction.durable import strict_json_digest
    return strict_json_digest(value)


def _node_config(plan: Any, node_id: str, kind: str) -> dict[str, Any]:
    snapshot = plan.snapshot()
    for index, node in enumerate(snapshot["nodes"]):
        if node["id"] == node_id:
            if node["type"] != kind:
                _fail("identity_mismatch", ("document", "nodes", index, "type"), "node kind changed")
            return deepcopy(node["config"])
    _fail("missing_value", ("node_id",), "node is not in the fixed graph")


@dataclass
class SkeletonToolRuntime:
    """Host adapter which intentionally delegates execution to the existing Tool path."""

    plan: Any
    toolkit: Toolkit
    session_id: str
    clock_ms: Callable[[], int]

    def __post_init__(self) -> None:
        self.plan = checked_execution_plan(self.plan)
        if type(self.toolkit) is not Toolkit:
            _fail("invalid_type", ("toolkit",), "Tool runtime requires a Toolkit")
        if type(self.session_id) is not str or not self.session_id:
            _fail("invalid_value", ("session_id",), "session id is required")
        if not callable(self.clock_ms):
            _fail("invalid_type", ("clock_ms",), "clock must be callable")

    def for_plan(self, plan: Any) -> "SkeletonToolRuntime":
        """Bind the same Toolkit and approval policy to a child definition."""
        return SkeletonToolRuntime(plan, self.toolkit, self.session_id, self.clock_ms)

    def tool_schemas(self) -> list[dict[str, Any]]:
        """Detached existing Toolkit schemas for a UI host; no execution admission."""
        return deepcopy(self.toolkit.to_json())

    def _config(self, context: DurableNodeContext) -> dict[str, Any]:
        return _node_config(self.plan, context.node_id, "tool_call")["tool"]

    def _tool_call(self, context: DurableNodeContext, resolved: dict[str, Any]) -> ToolCall:
        config = self._config(context)
        _validate_json(resolved, ("resolved_inputs",))
        if type(resolved) is not dict:
            _fail("invalid_type", ("resolved_inputs",), "Tool arguments must be an object")
        return ToolCall(call_id=context.node_invocation_id, name=config["name"], arguments=deepcopy(resolved))

    def _execution_context(self, context: DurableNodeContext, call: ToolCall) -> ToolExecutionContext:
        return ToolExecutionContext(
            session_id=self.session_id,
            run_id=context.graph_id,
            provider="",
            model="",
            iteration=0,
            tool_name=call.name,
            call_id=call.call_id,
            turn_id=context.node_invocation_id,
        )

    def _request(self, context: DurableNodeContext, resolved: dict[str, Any], preparation) -> InteractionRequest:
        request = preparation.request
        if request is None:
            _fail("invalid_value", ("tool",), "Tool confirmation request is missing")
        call = self._tool_call(context, resolved)
        subject = {
            "schema": _TOOL_WAIT_SCHEMA,
            "document_id": context.document_id,
            "graph_id": context.graph_id,
            "node_id": context.node_id,
            "node_invocation_id": context.node_invocation_id,
            "node_attempt_id": context.node_attempt_id,
            "tool_name": call.name,
            "arguments_sha256": _digest(preparation.effective_arguments),
            "tool_schema_sha256": _digest(next((item for item in self.toolkit.to_json() if item.get("name") == call.name), {})),
        }
        return build_interaction_request(
            session_id=self.session_id,
            kind="tool_approval",
            source_run_id=context.graph_id,
            occurrence="tool:" + context.node_invocation_id,
            payload=request.to_dict(),
            response_contract={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "modified_arguments": {"type": ["object", "null"]},
                    "reason": {"type": "string"},
                },
                "required": ["approved", "modified_arguments", "reason"],
                "additionalProperties": False,
            },
            created_revision=0,
            subject=subject,
        )

    def __call__(self, context: DurableNodeContext, resolved: dict[str, Any]):
        call = self._tool_call(context, resolved)
        guard = _NodeExecutionGuard(context.assert_current)
        preparation = prepare_tool_confirmation(
            toolkit=self.toolkit,
            tool_call=call,
            execution_context=self._execution_context(context, call),
            execution_guard=guard,
        )
        if preparation.needs_confirmation_response:
            request = self._request(context, resolved, preparation)
            now_ms = self.clock_ms()
            if type(now_ms) is not int or isinstance(now_ms, bool) or now_ms < 0:
                _fail("invalid_value", ("clock_ms",), "clock returned an invalid timestamp")
            timeout_ms = self._config(context)["approval_timeout_ms"]
            return WorkflowNodeWaiting("interaction", request.interaction_id, {
                "schema": _TOOL_WAIT_SCHEMA,
                "interaction": request.to_dict(),
                "expires_at_ms": now_ms + timeout_ms,
            })
        return self._execute(context, resolved, preparation, None)

    def _wait(self, raw: Any) -> tuple[InteractionRequest, int]:
        if type(raw) is not dict or set(raw) != {"schema", "interaction", "expires_at_ms"}:
            _fail("invalid_shape", ("wait",), "Tool wait is invalid")
        if raw["schema"] != _TOOL_WAIT_SCHEMA:
            _fail("unknown_variant", ("wait", "schema"), "Tool wait schema is invalid")
        request = InteractionRequest.from_dict(raw["interaction"])
        if type(raw["expires_at_ms"]) is not int or isinstance(raw["expires_at_ms"], bool) or raw["expires_at_ms"] < 0:
            _fail("invalid_value", ("wait", "expires_at_ms"), "Tool wait deadline is invalid")
        return request, raw["expires_at_ms"]

    def seal_interaction(self, context: DurableNodeContext, resolved: dict[str, Any], raw_wait: Any, raw_receipt: Any,
                         response: Any, *, submitted_by: str) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        stored, expires_at_ms = self._wait(raw_wait)
        call = self._tool_call(context, resolved)
        preparation = prepare_tool_confirmation(
            toolkit=self.toolkit,
            tool_call=call,
            execution_context=self._execution_context(context, call),
        )
        if not preparation.needs_confirmation_response:
            _fail("identity_mismatch", ("wait",), "Tool no longer requires the stored confirmation")
        current = self._request(context, resolved, preparation)
        ensure_interaction_binding_matches(stored, current)
        normalized = normalize_interaction_response(stored, response)
        if raw_receipt is not None:
            from unchain.interaction.durable import validate_interaction_receipt
            stored_receipt = validate_interaction_receipt(raw_receipt, request=stored).to_dict()
            if stored_receipt["response"] != normalized:
                _fail("identity_mismatch", ("response",), "approval conflicts with the persisted receipt")
            return stored_receipt, preparation, normalized
        now_ms = self.clock_ms()
        if type(now_ms) is not int or isinstance(now_ms, bool) or now_ms < 0:
            _fail("invalid_value", ("clock_ms",), "clock returned an invalid timestamp")
        if now_ms > expires_at_ms:
            _fail("interaction_expired", ("wait",), "Tool approval expired")
        receipt = build_interaction_receipt(stored, normalized, submitted_by=submitted_by, submitted_at_ms=now_ms)
        return receipt.to_dict(), preparation, normalized

    def is_expired(self, raw_wait: Any, now_ms: int) -> bool:
        _, expires_at_ms = self._wait(raw_wait)
        return now_ms > expires_at_ms

    def _execute(self, context: DurableNodeContext, resolved: dict[str, Any], preparation, response: dict[str, Any] | None):
        call = self._tool_call(context, resolved)
        guard = _NodeExecutionGuard(context.assert_current)
        kwargs = {}
        if response is not None:
            kwargs["confirmation_response"] = response
        outcome = execute_confirmable_tool_call(
            toolkit=self.toolkit,
            tool_call=call,
            on_tool_confirm=None,
            loop=None,
            callback=None,
            run_id=context.graph_id,
            iteration=0,
            execution_context=self._execution_context(context, call),
            execution_guard=guard,
            prepared_confirmation=preparation,
            **kwargs,
        )
        failed = outcome.denied or is_failed_tool_result(outcome.tool_result)
        code = "denied" if outcome.denied else "tool_error" if failed else None
        return WorkflowNodeCompleted({
            "result": deepcopy(outcome.tool_result),
            "ok": not failed,
            "error": None if code is None else {"code": code},
        })

    def resume_interaction(self, context: DurableNodeContext, resolved: dict[str, Any], preparation, response: dict[str, Any]):
        return self._execute(context, resolved, preparation, response)


@dataclass
class SkeletonSkillRuntime:
    """Read a fully identified skill once; the runner persists the resulting snapshot."""

    plan: Any
    registry: Any

    def __post_init__(self) -> None:
        self.plan = checked_execution_plan(self.plan)

    def for_plan(self, plan: Any) -> "SkeletonSkillRuntime":
        """Bind the same exact-read registry to a child definition."""
        return SkeletonSkillRuntime(plan, self.registry)

    def __call__(self, context: DurableNodeContext, resolved: dict[str, Any]):
        if resolved:
            _fail("invalid_value", ("resolved_inputs",), "Skill Read accepts no runtime inputs")
        identity = _node_config(self.plan, context.node_id, "skill_read")["skill"]
        try:
            loaded = self._read(identity)
        except SkeletonValidationError as error:
            return WorkflowNodeCompleted({"_skeleton_error": {"code": error.code}})
        except Exception:
            # Reading a Skill is side-effect-free. Its failure is a known
            # pre-dispatch result and must not become an unknown external call.
            return WorkflowNodeCompleted({"_skeleton_error": {"code": "skill_unavailable"}})
        return WorkflowNodeCompleted({
            "content": loaded.body,
            "name": loaded.summary.name,
            "_skeleton_skill": {
                "identity": {
                    "source": loaded.summary.identity.source,
                    "source_id": loaded.summary.identity.source_id,
                    "name": loaded.summary.name,
                },
                "revision": loaded.revision,
            },
        })

    def _read(self, identity: dict[str, str]) -> LoadedSkill:
        custom = getattr(self.registry, "read_exact", None)
        if callable(custom):
            raw = custom(deepcopy(identity))
            if type(raw) is not dict:
                _fail("invalid_type", ("skill",), "Skill reader returned an invalid value")
            actual = raw.get("identity")
            if actual != identity or type(raw.get("body")) is not str or type(raw.get("revision")) is not str:
                _fail("identity_mismatch", ("skill",), "Skill reader identity changed")
            from unchain.skills.models import SkillIdentity, SkillSummary
            return LoadedSkill(SkillSummary(SkillIdentity(**actual), "loaded skill", 0, None, None), raw["body"], raw["revision"])
        get_exact = getattr(self.registry, "get_exact", None)
        if callable(get_exact):
            from unchain.skills.models import SkillIdentity
            requested = SkillIdentity(**identity)
            loaded = get_exact(requested)
            if type(loaded) is not LoadedSkill or loaded.summary.identity != requested:
                _fail("identity_mismatch", ("skill",), "Skill identity is unavailable")
            return loaded
        resolve = getattr(self.registry, "resolve", None)
        get = getattr(self.registry, "get", None)
        if not callable(resolve) or not callable(get):
            _fail("invalid_type", ("registry",), "Skill registry does not support exact reads")
        summary = resolve(identity["name"])
        loaded = get(identity["name"])
        if (type(loaded) is not LoadedSkill or summary is None
                or loaded.summary.identity.source != identity["source"]
                or loaded.summary.identity.source_id != identity["source_id"]
                or loaded.summary.name != identity["name"]):
            _fail("identity_mismatch", ("skill",), "Skill identity is unavailable")
        return loaded


__all__ = ["SkeletonSkillRuntime", "SkeletonToolRuntime"]
