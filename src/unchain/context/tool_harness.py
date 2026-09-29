from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..input.human_input import is_human_input_tool_name
from ..capabilities import CapabilityOutcome, RunDelta
from ..interaction.effects import build_tool_approval_suspend_request
from ..kernel.delta import HarnessDelta
from ..kernel.harness import BaseRuntimeHarness, HarnessContext, RuntimePhase
from ..kernel.types import ToolCall
from ..tools.common import (
    append_executed_call_id,
    copy_messages,
    emit_loop_event,
)
from ..tools.messages import get_provider_message_builder
from ..tools.types import ToolBatchState

if TYPE_CHECKING:
    from .runtime import ContextRuntime

from .tool_permits import DurableToolApprovalPending


@dataclass
class ContextToolAuthorityHarness(BaseRuntimeHarness):
    """Run V2 tools before the Legacy execution harness can select a route."""

    name: str = "context_v2_tool_authority"
    phases: tuple[RuntimePhase, ...] = ("on_tool_call",)
    order: int = 90
    runtime: ContextRuntime = field(repr=False, kw_only=True)

    def build_delta(self, context: HarnessContext) -> HarnessDelta | None:
        tool_call = context.event.get("tool_call")
        if not isinstance(tool_call, ToolCall):
            return None
        if is_human_input_tool_name(tool_call.name):
            return None

        batch_state = context.state.tool_batch_state.copy()
        if (
            tool_call.call_id in batch_state.executed_call_ids
            or batch_state.awaiting_human_input
            or context.state.run_status == "awaiting_interaction"
        ):
            return None

        emit_loop_event(
            context.event.get("loop"),
            context.event.get("callback"),
            "tool_call",
            str(context.event.get("run_id") or "kernel"),
            iteration=int(context.state.iteration),
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            arguments=copy.deepcopy(tool_call.arguments),
            source_provider=str(context.state.provider_state.provider or ""),
        )
        permit = self.runtime.prepare_tool_execution(context)
        if type(permit) is DurableToolApprovalPending:
            interaction_request = permit.interaction_request
            continuation = permit.continuation
            suspend_op = build_tool_approval_suspend_request(
                interaction_request=interaction_request,
                continuation=continuation,
            )
            return CapabilityOutcome(
                delta=RunDelta(
                    created_by=f"context.{self.name}",
                    context_ops=(suspend_op,),
                    state_updates={
                        "run_status": "awaiting_interaction",
                        "last_continuation": continuation,
                        "pending_tool_calls": list(
                            context.event.get("tool_calls") or []
                        ),
                        "tool_batch_state": batch_state,
                    },
                    trace={
                        "awaiting_tool_approval": True,
                        "tool_call": tool_call.name,
                        "call_id": tool_call.call_id,
                        "interaction_id": interaction_request["interaction_id"],
                        "context_v2_tool_authority": True,
                    },
                )
            )
        receipt = self.runtime.execute_prepared_tool(context, permit)

        project_result = getattr(self.runtime, "project_tool_result_for_model", None)
        if callable(project_result):
            visible_result = project_result(context, receipt)
        else:  # pragma: no cover - compatibility for isolated harness fakes
            from ..journal.models import _thaw_json

            visible_result = _thaw_json(receipt.visible_result)
        if not isinstance(visible_result, dict):
            visible_result = {"result": visible_result}
        builder = get_provider_message_builder(
            str(context.state.provider_state.provider or "")
        )
        result_messages = copy_messages(batch_state.result_messages)
        result_messages.extend(
            builder.build_tool_result_messages(
                tool_call=tool_call,
                tool_result=visible_result,
            )
        )
        emit_persisted_tool_result = getattr(
            self.runtime,
            "emit_persisted_tool_result",
            None,
        )
        if callable(emit_persisted_tool_result):
            emit_persisted_tool_result(
                context,
                receipt,
                result=visible_result,
            )
        else:  # pragma: no cover - compatibility for isolated harness fakes
            emit_loop_event(
                context.event.get("loop"),
                context.event.get("callback"),
                "tool_result",
                str(context.event.get("run_id") or "kernel"),
                iteration=int(context.state.iteration),
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
                result=copy.deepcopy(visible_result),
            )
        state_updates = {
            "tool_batch_state": ToolBatchState(
                result_messages=result_messages,
                should_observe=(
                    batch_state.should_observe or bool(receipt.should_observe)
                ),
                awaiting_human_input=False,
                human_input_request=batch_state.human_input_request,
                human_input_tool_call_id=(batch_state.human_input_tool_call_id),
                executed_call_ids=append_executed_call_id(
                    batch_state,
                    tool_call.call_id,
                ),
            ),
        }
        if receipt.transition is not None:
            replacement = self.runtime.materialize_tool_transition(
                context,
                receipt,
            )
            if receipt.transition.kind == "subagent_terminal_handoff":
                if replacement is None:
                    state_updates = {}
                else:
                    terminal_keys = {
                        "subagent_state",
                        "transcript",
                        "run_status",
                        "pending_tool_calls",
                        "tool_batch_state",
                        "last_continuation",
                        "next_model_input",
                    }
                    if (
                        type(replacement) is not dict
                        or set(replacement) != terminal_keys
                    ):
                        from .tool_executor import (
                            DurableToolExecutorContractError,
                        )

                        raise DurableToolExecutorContractError(
                            "terminal handoff transition is not one exact state update"
                        )
                    state_updates = replacement
            elif replacement is not None:
                state_updates["subagent_state"] = replacement
        return HarnessDelta(
            created_by=f"context.{self.name}",
            state_updates=state_updates,
            trace={
                "tool_call": tool_call.name,
                "call_id": tool_call.call_id,
                "durable_tool_completion": True,
                "durable_tool_reused": bool(receipt.reused),
                "completion_ref": receipt.completion_artifact.to_dict(),
                "result_ref": receipt.result_artifact.to_dict(),
            },
        )


__all__ = ["ContextToolAuthorityHarness"]
