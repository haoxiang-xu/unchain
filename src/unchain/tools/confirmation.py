from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ..capabilities import CapabilityOutcome, RunContext
from ..execution import ExecutionGuard, ExecutionLeaseError
from ..toolkits.base import BuiltinToolkit
from .models import (
    ToolConfirmationPolicy,
    ToolConfirmationRequest,
    ToolConfirmationResponse,
    ToolExecutionContext,
)
from .toolkit import Toolkit
from ..kernel.types import ToolCall
from .common import emit_loop_event


@dataclass(frozen=True)
class ToolExecutionOutcome:
    tool_result: dict[str, Any]
    should_observe: bool
    denied: bool = False
    deny_reason: str = ""
    effective_arguments: Any = None
    confirmation_policy: ToolConfirmationPolicy | None = None
    capability_outcome: CapabilityOutcome | None = None


@dataclass(frozen=True)
class ToolConfirmationPreparation:
    should_observe: bool
    effective_arguments: Any
    confirmation_policy: ToolConfirmationPolicy | None
    requires_confirmation: bool
    request: ToolConfirmationRequest | None
    resolver_error: str | None = None

    @property
    def needs_confirmation_response(self) -> bool:
        return (
            self.resolver_error is None
            and self.requires_confirmation
            and self.request is not None
        )

    def to_resolver_error_outcome(self, *, tool_name: str) -> ToolExecutionOutcome | None:
        if self.resolver_error is None:
            return None
        return ToolExecutionOutcome(
            tool_result={
                "error": f"tool confirmation resolver failed: {self.resolver_error}",
                "tool": tool_name,
            },
            should_observe=self.should_observe,
            effective_arguments=self.effective_arguments,
            confirmation_policy=self.confirmation_policy,
        )


_CONFIRMATION_RESPONSE_UNSET = object()


def _resolve_builtin_toolkit_owner(toolkit: Toolkit, tool_name: str) -> BuiltinToolkit | None:
    tool_obj = toolkit.get(tool_name)
    if tool_obj is None:
        return None
    owner = getattr(getattr(tool_obj, "func", None), "__self__", None)
    return owner if isinstance(owner, BuiltinToolkit) else None


@contextmanager
def _tool_execution_scope(
    *,
    toolkit: Toolkit,
    tool_name: str,
    execution_context: ToolExecutionContext | None,
):
    owner = _resolve_builtin_toolkit_owner(toolkit, tool_name)
    if owner is None or execution_context is None:
        yield
        return

    owner.push_execution_context(execution_context)
    try:
        yield
    finally:
        owner.pop_execution_context()


def _tool_result_from_capability_outcome(
    outcome: CapabilityOutcome,
    *,
    tool_name: str,
) -> dict[str, Any]:
    del tool_name
    value = outcome.value
    if isinstance(value, dict):
        return value
    return {"result": value}


def prepare_tool_confirmation(
    *,
    toolkit: Toolkit,
    tool_call: ToolCall,
    execution_context: ToolExecutionContext | None = None,
    execution_guard: ExecutionGuard | None = None,
) -> ToolConfirmationPreparation:
    tool_obj = toolkit.get(tool_call.name)
    should_observe = bool(tool_obj is not None and tool_obj.observe)
    effective_arguments = copy.deepcopy(tool_call.arguments)
    confirmation_policy: ToolConfirmationPolicy | None = None
    requires_confirmation = bool(tool_obj is not None and tool_obj.requires_confirmation)

    if tool_obj is not None:
        try:
            # Use the executor's parser before policy evaluation. Providers may
            # encode the same object as JSON; policy, UI and execution must see
            # the same arguments, including the existing JSON repair behavior.
            effective_arguments = copy.deepcopy(tool_obj._parse_arguments(effective_arguments))
            if not isinstance(effective_arguments, dict):
                raise ValueError("tool arguments must decode to an object")
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            return ToolConfirmationPreparation(
                should_observe=should_observe,
                effective_arguments=effective_arguments,
                confirmation_policy=None,
                requires_confirmation=requires_confirmation,
                request=None,
                resolver_error=f"{type(exc).__name__}: {exc}",
            )

    if tool_obj is not None and callable(getattr(tool_obj, "confirmation_resolver", None)):
        resolver_arguments = copy.deepcopy(effective_arguments)
        try:
            confirmation_policy = ToolConfirmationPolicy.from_raw(
                tool_obj.confirmation_resolver(resolver_arguments, execution_context)
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            return ToolConfirmationPreparation(
                should_observe=should_observe,
                effective_arguments=effective_arguments,
                confirmation_policy=confirmation_policy,
                requires_confirmation=requires_confirmation,
                request=None,
                resolver_error=f"{type(exc).__name__}: {exc}",
            )
        if execution_guard is not None:
            execution_guard.assert_active()

    if confirmation_policy is not None:
        requires_confirmation = requires_confirmation and confirmation_policy.requires_confirmation

    confirmation_request: ToolConfirmationRequest | None = None
    if tool_obj is not None and requires_confirmation:
        tool_render = getattr(tool_obj, "render_component", None)
        if isinstance(tool_render, dict) and tool_render:
            effective_render = dict(tool_render)
        else:
            effective_render = {"version": 1, "type": "confirmation", "config": {}}
        policy_render = confirmation_policy.render_component if confirmation_policy is not None else None
        if isinstance(policy_render, dict) and policy_render:
            effective_render = dict(policy_render)

        policy_interact_type = (
            confirmation_policy.interact_type
            if confirmation_policy is not None
            else "confirmation"
        )
        policy_interact_config = (
            confirmation_policy.interact_config
            if confirmation_policy is not None
            else None
        )

        confirmation_request = ToolConfirmationRequest(
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            arguments=copy.deepcopy(effective_arguments),
            description=(
                confirmation_policy.description
                if confirmation_policy is not None and confirmation_policy.description
                else tool_obj.description
            ),
            interact_type=policy_interact_type,
            interact_config=policy_interact_config,
            render_component=effective_render,
        )

    return ToolConfirmationPreparation(
        should_observe=should_observe,
        effective_arguments=effective_arguments,
        confirmation_policy=confirmation_policy,
        requires_confirmation=requires_confirmation,
        request=confirmation_request,
    )


def execute_confirmable_tool_call(
    *,
    toolkit: Toolkit,
    tool_call: ToolCall,
    on_tool_confirm: Any,
    loop: Any,
    callback: Any,
    run_id: str,
    iteration: int,
    execution_context: ToolExecutionContext | None = None,
    execution_guard: ExecutionGuard | None = None,
    prepared_confirmation: ToolConfirmationPreparation | None = None,
    confirmation_response: Any = _CONFIRMATION_RESPONSE_UNSET,
) -> ToolExecutionOutcome:
    if prepared_confirmation is None:
        preparation = prepare_tool_confirmation(
            toolkit=toolkit,
            tool_call=tool_call,
            execution_context=execution_context,
            execution_guard=execution_guard,
        )
    else:
        preparation = prepared_confirmation
    resolver_error_outcome = preparation.to_resolver_error_outcome(tool_name=tool_call.name)
    if resolver_error_outcome is not None:
        return resolver_error_outcome

    tool_obj = toolkit.get(tool_call.name)
    should_observe = preparation.should_observe
    effective_arguments = (
        preparation.effective_arguments
        if prepared_confirmation is None
        else copy.deepcopy(preparation.effective_arguments)
    )
    denied = False
    deny_reason = ""
    confirmation_policy = preparation.confirmation_policy

    raw_response = _CONFIRMATION_RESPONSE_UNSET
    if preparation.needs_confirmation_response:
        if confirmation_response is not _CONFIRMATION_RESPONSE_UNSET:
            raw_response = confirmation_response
        elif callable(on_tool_confirm):
            raw_response = on_tool_confirm(preparation.request)

    if raw_response is not _CONFIRMATION_RESPONSE_UNSET:
        response = ToolConfirmationResponse.from_raw(raw_response)
        if execution_guard is not None:
            execution_guard.renew()
        if not response.approved:
            denied = True
            deny_reason = response.reason
            emit_loop_event(
                loop,
                callback,
                "tool_denied",
                run_id,
                iteration=iteration,
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
                reason=deny_reason,
            )
        elif response.modified_arguments is not None:
            effective_arguments = copy.deepcopy(response.modified_arguments)
            emit_loop_event(
                loop,
                callback,
                "tool_confirmed",
                run_id,
                iteration=iteration,
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
            )

    if denied:
        tool_result = {
            "denied": True,
            "tool": tool_call.name,
            "reason": deny_reason or "User denied execution.",
        }
    else:
        if execution_guard is not None:
            execution_guard.renew()
        with _tool_execution_scope(
            toolkit=toolkit,
            tool_name=tool_call.name,
            execution_context=execution_context,
        ):
            if tool_obj is None:
                tool_result = toolkit.execute(tool_call.name, effective_arguments)
            else:
                capability_outcome = tool_obj.invoke(
                    {
                        "tool_name": tool_call.name,
                        "call_id": tool_call.call_id,
                        "arguments": effective_arguments,
                    },
                    RunContext(
                        phase="on_tool_call",
                        event={
                            "tool_call": tool_call,
                            "execution_context": execution_context,
                        },
                    ),
                )
                tool_result = _tool_result_from_capability_outcome(
                    capability_outcome,
                    tool_name=tool_call.name,
                )
        if execution_guard is not None:
            execution_guard.assert_active()

    if execution_guard is not None:
        execution_guard.assert_active()

    return ToolExecutionOutcome(
        tool_result=tool_result,
        should_observe=should_observe,
        denied=denied,
        deny_reason=deny_reason,
        effective_arguments=effective_arguments,
        confirmation_policy=confirmation_policy,
        capability_outcome=capability_outcome if not denied and tool_obj is not None else None,
    )
