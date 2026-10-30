"""Phase-specific, closed decisions emitted by non-LLM Hook Workflows."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .errors import SkeletonValidationError
from .io import _validate_json


HOOK_CHILD_WAIT_SCHEMA = "unchain.hook_child_wait.v1"
HOOK_CHILD_WAIT_FIELDS = frozenset({
    "schema", "parent_graph_id", "parent_node_invocation_id", "parent_tool_call",
    "hook_id", "workflow_id", "revision_id", "definition_digest",
    "child_graph_id", "child_attempt_id", "child_node_invocation_id",
    "child_wait_kind", "child_wait_id", "interaction_request",
})


def validate_hook_child_wait(raw, *, parent_call_id, parent_tool_name):
    """Admit one exact Hook Tool wait across the graph/provider boundary."""
    from unchain.interaction.durable import InteractionRequest

    if (type(raw) is not dict or set(raw) != HOOK_CHILD_WAIT_FIELDS
            or raw.get("schema") != HOOK_CHILD_WAIT_SCHEMA
            or raw.get("child_wait_kind") != "interaction"
            or raw.get("parent_tool_call") != {
                "call_id": parent_call_id, "tool_name": parent_tool_name,
            }):
        raise ValueError("Hook child wait has an invalid parent binding")
    for field in HOOK_CHILD_WAIT_FIELDS - {"schema", "parent_tool_call", "child_wait_kind", "interaction_request"}:
        if type(raw[field]) is not str or not raw[field]:
            raise ValueError("Hook child wait identity is invalid")
    child = InteractionRequest.from_dict(raw["interaction_request"])
    if (child.kind != "tool_approval" or child.interaction_id != raw["child_wait_id"]
            or child.source_run_id != raw["child_graph_id"]):
        raise ValueError("Hook child approval request changed")
    return child


def _fail(code: str, path: tuple[str | int, ...], message: str) -> None:
    raise SkeletonValidationError(code, path, message)


def _closed(value: Any, path: tuple[str | int, ...], required: set[str], optional: set[str] = frozenset()) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("invalid_type", path, "Hook decision must be an object")
    keys = set(value)
    missing = required - keys
    if missing:
        _fail("missing_field", path + (sorted(missing)[0],), "required Hook decision field is missing")
    extra = keys - required - optional
    if extra:
        _fail("unknown_field", path + (sorted(extra)[0],), "unknown Hook decision field")
    return value


def apply_before_decision(
    current_arguments: dict[str, Any],
    approval_required: bool,
    decision: Any,
    path: tuple[str | int, ...],
) -> tuple[dict[str, Any], bool, dict[str, str] | None]:
    """Apply one before-hook decision without changing the original tool identity."""
    if type(current_arguments) is not dict:
        _fail("invalid_type", path + ("arguments",), "current tool arguments must be an object")
    if type(approval_required) is not bool:
        _fail("invalid_type", path + ("approval_required",), "approval state must be a boolean")
    if type(decision) is not dict or type(decision.get("decision")) is not str:
        _fail("invalid_type", path + ("decision",), "Hook decision must declare a string decision")

    kind = decision["decision"]
    if kind == "continue":
        value = _closed(decision, path, {"decision", "require_approval"}, {"arguments"})
        if type(value["require_approval"]) is not bool:
            _fail("invalid_type", path + ("require_approval",), "require_approval must be a boolean")
        arguments = value.get("arguments", current_arguments)
        _validate_json(arguments, path + ("arguments",))
        if type(arguments) is not dict:
            _fail("invalid_type", path + ("arguments",), "Hook arguments must be an object")
        return deepcopy(arguments), approval_required or value["require_approval"], None
    if kind == "block":
        value = _closed(decision, path, {"decision", "reason"})
        if type(value["reason"]) is not str or not value["reason"].strip():
            _fail("invalid_value", path + ("reason",), "block reason must be non-empty")
        return deepcopy(current_arguments), approval_required, {
            "code": "hook_blocked", "reason": value["reason"],
        }
    _fail("unknown_variant", path + ("decision",), "before Hook decision is unknown")


def apply_after_decision(
    current_result: dict[str, Any],
    decision: Any,
    path: tuple[str | int, ...],
) -> dict[str, Any]:
    """Apply one after-hook decision; callers retain the raw result separately."""
    if type(current_result) is not dict:
        _fail("invalid_type", path + ("current_result",), "current tool result must be an object")
    if type(decision) is not dict or type(decision.get("decision")) is not str:
        _fail("invalid_type", path + ("decision",), "Hook decision must declare a string decision")
    kind = decision["decision"]
    if kind == "continue":
        _closed(decision, path, {"decision"})
        return deepcopy(current_result)
    if kind == "replace":
        value = _closed(decision, path, {"decision", "result"})
        _validate_json(value["result"], path + ("result",))
        if type(value["result"]) is not dict:
            _fail("invalid_type", path + ("result",), "replacement tool result must be an object")
        return deepcopy(value["result"])
    _fail("unknown_variant", path + ("decision",), "after Hook decision is unknown")


class SkeletonBeforeToolHookDispatcher:
    """Run matching admitted before Hooks without giving them tool authority.

    The host supplies the deterministic Workflow invocation capability. This
    adapter owns event input, ordering, decision validation, and the fact that
    a Hook can only change arguments for the already selected tool.
    """

    def __init__(
        self,
        *,
        hook_runtime,
        kernel_node_id: str,
        parent_context: Any,
        parent_store: Any,
        run_hook_workflow: Callable[..., dict[str, Any]] | None,
    ) -> None:
        from .hook_runtime import SkeletonHookRuntime

        if type(hook_runtime) is not SkeletonHookRuntime:
            raise TypeError("before Hook dispatcher requires an admitted Hook runtime")
        if type(kernel_node_id) is not str or not kernel_node_id:
            raise ValueError("before Hook dispatcher requires a Kernel node id")
        if run_hook_workflow is not None and not callable(run_hook_workflow):
            raise TypeError("before Hook workflow runner must be callable")
        self._bindings = hook_runtime.bindings_for_node(kernel_node_id)
        self._parent_context = parent_context
        self._parent_store = parent_store
        self._run_hook_workflow = run_hook_workflow

    def __call__(self, *, context, tool_call, execution_context):
        del execution_context
        from .hook_phase_journal import HookPhaseJournal
        from unchain.execution import ExecutionLeaseError
        from unchain.context.runtime import ContextToolPreparationPolicyResult
        from unchain.kernel.types import ToolCall
        from unchain.tools.confirmation import normalize_tool_arguments

        arguments = deepcopy(tool_call.arguments)
        require_approval = False
        phases = HookPhaseJournal(self._parent_store, self._parent_context,
            call_id=tool_call.call_id, tool_name=tool_call.name)
        for binding in self._bindings:
            hook = binding.hook
            if hook["event"] != "before_tool_call":
                continue
            if hook["tool_name"] is not None and hook["tool_name"] != tool_call.name:
                continue
            if self._run_hook_workflow is None:
                return ContextToolPreparationPolicyResult(
                    arguments=arguments,
                    force_approval=require_approval,
                    blocked={
                        "code": "hook_runner_unavailable",
                        "reason": "The Hook Workflow runner is unavailable.",
                    },
                )
            inputs = {
                "event": "before_tool_call",
                "call_id": tool_call.call_id,
                "tool_name": tool_call.name,
                "arguments": deepcopy(arguments),
            }
            try:
                resume_request = context.event.get("interaction_request")
                resume_subject = resume_request.get("subject") if type(resume_request) is dict else None
                hook_wait = resume_subject.get("hook_child_wait") if type(resume_subject) is dict else None
                resume = (
                    {"request": deepcopy(resume_request),
                     "response": deepcopy(context.event["interaction_response"])}
                    if type(hook_wait) is dict and hook_wait.get("hook_id") == hook["id"]
                    else None
                )
                decision = phases.replay(binding, "before_tool_call", inputs)
                if decision is None:
                    decision = self._run_hook_workflow(
                        binding=binding,
                        inputs=inputs,
                        parent_context=self._parent_context,
                        parent_store=self._parent_store,
                        **({"hook_resume": resume} if resume is not None else {}),
                    )
                arguments, require_approval, blocked = apply_before_decision(
                    arguments,
                    require_approval or hook["force_approval"],
                    decision,
                    ("hooks", hook["id"]),
                )
                if type(decision) is dict and "arguments" in decision and blocked is None:
                    try:
                        arguments = normalize_tool_arguments(
                            toolkit=context.event.get("toolkit"),
                            tool_call=ToolCall(
                                call_id=tool_call.call_id,
                                name=tool_call.name,
                                arguments=arguments,
                            ),
                            validate_schema=True,
                        )
                    except ExecutionLeaseError:
                        raise
                    except Exception:
                        return ContextToolPreparationPolicyResult(
                            arguments=arguments,
                            force_approval=require_approval,
                            blocked={
                                "code": "hook_arguments_invalid",
                                "reason": "Hook Workflow produced invalid tool arguments.",
                            },
                        )
                phases.record(binding, "before_tool_call", inputs, decision)
            except ExecutionLeaseError:
                raise
            except SkeletonValidationError as error:
                if error.code in {
                    "child_cancelled",
                    "child_failed",
                    "child_outcome_unknown",
                    "child_waiting",
                    "hook_incomplete",
                    "identity_mismatch",
                }:
                    raise
                return ContextToolPreparationPolicyResult(
                    arguments=arguments,
                    force_approval=require_approval or hook["force_approval"],
                    blocked={
                        "code": "hook_" + error.code,
                        "reason": "Hook Workflow did not produce a valid decision.",
                    },
                )
            except Exception:
                raise
            if blocked is not None:
                return ContextToolPreparationPolicyResult(
                    arguments=arguments,
                    force_approval=require_approval,
                    blocked=blocked,
                )
        return ContextToolPreparationPolicyResult(
            arguments=arguments,
            force_approval=require_approval,
        )


class SkeletonAfterToolHookDispatcher:
    """Run pinned after Hooks before a parent result becomes visible."""

    def __init__(self, *, hook_runtime, kernel_node_id, parent_context,
                 parent_store, run_hook_workflow):
        from .hook_runtime import SkeletonHookRuntime
        if type(hook_runtime) is not SkeletonHookRuntime:
            raise TypeError("after Hook dispatcher requires an admitted Hook runtime")
        if type(kernel_node_id) is not str or not kernel_node_id:
            raise ValueError("after Hook dispatcher requires a Kernel node id")
        if run_hook_workflow is not None and not callable(run_hook_workflow):
            raise TypeError("after Hook Workflow runner must be callable")
        self._bindings = hook_runtime.bindings_for_node(kernel_node_id)
        self._parent_context = parent_context
        self._parent_store = parent_store
        self._run_hook_workflow = run_hook_workflow

    def __call__(self, *, context, request, raw_draft, effective_arguments, artifacts):
        from .agent_preparation import execution_definition_digest
        from .hook_phase_journal import HookPhaseJournal, _digest
        from .hook_raw_seal import HookRawSealStore
        from unchain.context.tool_executor import DurableToolCompletionDraft
        from unchain.journal.models import _thaw_json

        matching = [binding for binding in self._bindings
            if binding.hook["event"] == "after_tool_call"
            and (binding.hook["tool_name"] is None
                 or binding.hook["tool_name"] == request.tool_name)]
        if not matching:
            return raw_draft
        manifest_sha256 = _digest([{
            "hook": binding.hook,
            "workflow_id": binding.workflow.workflow_id,
            "revision_id": binding.workflow.revision_id,
            "definition_digest": execution_definition_digest(binding.plan),
        } for binding in matching])
        raw_seal = HookRawSealStore(self._parent_store, self._parent_context,
            manifest_sha256=manifest_sha256)
        raw_seal.record(request, raw_draft, artifacts, effective_arguments)
        raw = _thaw_json(raw_draft.result)
        if type(raw) is not dict:
            raise SkeletonValidationError("invalid_type", ("hooks", "raw_result"),
                "after Hook requires an object tool result")
        visible = deepcopy(raw)
        raw_available = True
        phases = HookPhaseJournal(self._parent_store, self._parent_context,
            call_id=request.call_id, tool_name=request.tool_name)
        for binding in matching:
            hook = binding.hook
            if self._run_hook_workflow is None:
                raise SkeletonValidationError("hook_runner_unavailable", ("hooks", hook["id"]),
                    "after Hook Workflow runner is unavailable")
            inputs = {
                "event": "after_tool_call",
                "call_id": request.call_id,
                "tool_name": request.tool_name,
                "arguments": deepcopy(effective_arguments),
                "raw_result": deepcopy(raw) if raw_available else {},
                "visible_result": deepcopy(visible),
            }
            decision = phases.replay(binding, "after_tool_call", inputs)
            if decision is None:
                resume_request = context.event.get("interaction_request")
                subject = resume_request.get("subject") if type(resume_request) is dict else None
                hook_wait = subject.get("hook_child_wait") if type(subject) is dict else None
                resume = (
                    {"request": deepcopy(resume_request),
                     "response": deepcopy(context.event["interaction_response"])}
                    if type(hook_wait) is dict and hook_wait.get("hook_id") == hook["id"]
                    else None
                )
                decision = self._run_hook_workflow(
                    binding=binding, inputs=inputs,
                    parent_context=self._parent_context,
                    parent_store=self._parent_store,
                    **({"hook_resume": resume} if resume is not None else {}),
                )
            try:
                visible = apply_after_decision(visible, decision, ("hooks", hook["id"]))
            except SkeletonValidationError as error:
                raise SkeletonValidationError("hook_invalid_decision",
                    ("hooks", hook["id"]),
                    "after Hook produced an invalid decision") from error
            phases.record(binding, "after_tool_call", inputs, decision)
            if decision["decision"] == "replace":
                raw_available = False
        return DurableToolCompletionDraft(
            result=visible,
            should_observe=raw_draft.should_observe,
            state_transition=raw_draft.state_transition,
        )


__all__ = [
    "SkeletonAfterToolHookDispatcher",
    "SkeletonBeforeToolHookDispatcher",
    "apply_after_decision",
    "apply_before_decision",
]
