from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from ..capabilities import CapabilityOutcome, CreateArtifactOp, RunDelta
from ..artifacts import (
    ArtifactOwner,
    artifacts_from_code_diff_policy,
    canonicalize_artifacts,
    extract_authored_artifacts,
    is_failed_tool_result,
    plan_artifact_from_tool_result,
)
from ..input.human_input import is_human_input_tool_name
from ..interaction import (
    INTERACTION_EFFECT_CREATED_BY,
    build_human_input_continuation,
    build_human_input_suspend_request,
)
from ..toolkits.base import BuiltinToolkit
from ..workspace_changes import WorkspaceChangeTracker
from .toolkit import Toolkit
from ..kernel.delta import HarnessDelta
from ..kernel.types import ToolCall
from .base import BaseToolHarness, ToolContext
from .common import append_executed_call_id, copy_messages, emit_loop_event
from .confirmation import execute_confirmable_tool_call
from .human_input import parse_human_input_request
from .messages import coalesce_provider_tool_result_messages, get_provider_message_builder
from .models import ToolExecutionContext
from .observation import ToolObservationRunner, inject_observation, observation_token_state
from .result_budget import ToolResultBudgetConfig, ToolResultBudgetController
from .runtime import run_tool_runtime_plugins
from .types import ToolBatchState


def _artifact_owner(context: ToolContext, tool_call: ToolCall) -> ArtifactOwner:
    return ArtifactOwner(
        run_id=context.run_id,
        turn_id=f"{context.run_id}:turn-{context.iteration}",
        tool_name=tool_call.name,
        call_id=tool_call.call_id,
    )


def _builtin_toolkit_owner(toolkit: Toolkit, tool_name: str) -> BuiltinToolkit | None:
    tool_obj = toolkit.get(tool_name)
    owner = getattr(getattr(tool_obj, "func", None), "__self__", None) if tool_obj is not None else None
    return owner if isinstance(owner, BuiltinToolkit) else None


def _workspace_change_tracker(
    context: ToolContext,
    toolkit: Toolkit,
    tool_call: ToolCall,
) -> WorkspaceChangeTracker | None:
    owner = _builtin_toolkit_owner(toolkit, tool_call.name)
    workspace_roots = getattr(owner, "workspace_roots", None) if owner is not None else None
    if not isinstance(workspace_roots, list) or not workspace_roots:
        return None
    return WorkspaceChangeTracker.from_state(
        context.state.workspace_change_state,
        run_id=context.run_id,
        workspace_roots=workspace_roots,
    )


def _workspace_change_state_update(
    tracker: WorkspaceChangeTracker | None,
) -> dict[str, Any]:
    if tracker is None or not tracker.has_changes():
        return {}
    return {"workspace_change_state": tracker.to_state()}


def _emit_artifact_events(
    context: ToolContext,
    tool_call: ToolCall,
    artifacts: list[dict],
) -> None:
    existing_ids = {
        artifact.get("artifact_id")
        for artifact in context.state.artifacts
        if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str)
    }
    seen_ids = set(existing_ids)
    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id") if isinstance(artifact, dict) else None
        if not isinstance(artifact_id, str) or not artifact_id:
            continue
        event_type = "artifact_updated" if artifact_id in seen_ids else "artifact_created"
        seen_ids.add(artifact_id)
        snapshot = artifact.get("snapshot") if isinstance(artifact.get("snapshot"), dict) else {}
        plan_id = snapshot.get("plan_id") if isinstance(snapshot, dict) else None
        emit_loop_event(
            context.loop,
            context.callback,
            event_type,
            context.run_id,
            iteration=context.iteration,
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            artifact_id=artifact_id,
            plan_id=plan_id if isinstance(plan_id, str) and plan_id else None,
            artifact=copy.deepcopy(artifact),
        )


def _create_artifact_ops(artifacts: list[dict], *, reason: str) -> tuple[CreateArtifactOp, ...]:
    return tuple(
        CreateArtifactOp(artifact=copy.deepcopy(artifact), reason=reason)
        for artifact in artifacts
        if isinstance(artifact, dict)
    )


def _canonical_artifacts_for_tool_result(
    context: ToolContext,
    tool_call: ToolCall,
    tool_result: dict,
    authored_artifacts: list[dict],
    *,
    confirmation_policy: object | None = None,
    effective_arguments: object | None = None,
) -> list[dict]:
    if is_failed_tool_result(tool_result):
        return []

    descriptors = list(authored_artifacts)
    has_authored_file_diff = any(
        isinstance(descriptor, dict) and descriptor.get("kind") == "file_diff"
        for descriptor in descriptors
    )
    arguments_changed = effective_arguments is not None and effective_arguments != tool_call.arguments
    interact_type = getattr(confirmation_policy, "interact_type", None)
    interact_config = getattr(confirmation_policy, "interact_config", None)
    if interact_type == "code_diff" and not has_authored_file_diff and not arguments_changed:
        descriptors.extend(
            artifacts_from_code_diff_policy(
                interact_config,
                call_id=tool_call.call_id,
                tool_name=tool_call.name,
            )
        )

    plan_descriptor = plan_artifact_from_tool_result(
        tool_name=tool_call.name,
        tool_result=tool_result,
    )
    if plan_descriptor is not None:
        descriptors.append(plan_descriptor)

    return canonicalize_artifacts(
        descriptors,
        owner=_artifact_owner(context, tool_call),
        existing_artifacts=context.state.artifacts,
    )


@dataclass
class ToolExecutionHarness(BaseToolHarness):
    name: str = "tool_execution"
    phases: tuple[str, ...] = ("after_model", "on_tool_call", "after_tool_batch")
    order: int = 100

    def build_tool_delta(self, context: ToolContext) -> HarnessDelta | None:
        if context.phase == "after_model":
            return self._after_model(context)
        if context.phase == "on_tool_call":
            return self._on_tool_call(context)
        if context.phase == "after_tool_batch":
            return self._after_tool_batch(context)
        return None

    def _after_model(self, context: ToolContext) -> HarnessDelta | None:
        tool_calls = list(context.state.pending_tool_calls)
        if not tool_calls:
            return None

        includes_human_input = any(is_human_input_tool_name(tool_call.name) for tool_call in tool_calls)
        includes_handoff = any(tool_call.name == "handoff_to_subagent" for tool_call in tool_calls)
        if includes_human_input and len(tool_calls) > 1:
            error_text = "ask_user_question must be the only tool call in a turn"
            builder = get_provider_message_builder(context.provider)
            result_messages = [
                builder.build_tool_result_message(
                    tool_call=tool_call,
                    tool_result={"error": error_text, "tool": tool_call.name},
                )
                for tool_call in tool_calls
            ]
            return HarnessDelta(
                created_by=self.created_by,
                state_updates={
                    "tool_batch_state": ToolBatchState(
                        result_messages=result_messages,
                        should_observe=False,
                        awaiting_human_input=False,
                        human_input_request=None,
                        human_input_tool_call_id=None,
                        executed_call_ids=[tool_call.call_id for tool_call in tool_calls],
                    ),
                    "run_status": "running",
                },
                trace={
                    "mixed_human_input_batch": True,
                    "tool_call_count": len(tool_calls),
                },
            )
        if includes_handoff and len(tool_calls) > 1:
            error_text = "handoff_to_subagent must be the only tool call in a turn"
            builder = get_provider_message_builder(context.provider)
            result_messages = [
                builder.build_tool_result_message(
                    tool_call=tool_call,
                    tool_result={"error": error_text, "tool": tool_call.name},
                )
                for tool_call in tool_calls
            ]
            return HarnessDelta(
                created_by=self.created_by,
                state_updates={
                    "tool_batch_state": ToolBatchState(
                        result_messages=result_messages,
                        should_observe=False,
                        awaiting_human_input=False,
                        human_input_request=None,
                        human_input_tool_call_id=None,
                        executed_call_ids=[tool_call.call_id for tool_call in tool_calls],
                    ),
                    "run_status": "running",
                },
                trace={
                    "mixed_handoff_batch": True,
                    "tool_call_count": len(tool_calls),
                },
            )

        return HarnessDelta(
            created_by=self.created_by,
            state_updates={
                "tool_batch_state": ToolBatchState(),
                "run_status": "running",
            },
            trace={"tool_call_count": len(tool_calls)},
        )

    def _on_tool_call(self, context: ToolContext) -> HarnessDelta | None:
        toolkit = context.toolkit if context.toolkit is not None else Toolkit()
        on_tool_confirm = context.event.get("on_tool_confirm")
        tool_call = context.event.get("tool_call")
        if not isinstance(tool_call, ToolCall):
            return None

        batch_state = context.state.tool_batch_state.copy()
        if tool_call.call_id in batch_state.executed_call_ids or batch_state.awaiting_human_input:
            return None

        emit_loop_event(
            context.loop,
            context.callback,
            "tool_call",
            context.run_id,
            iteration=context.iteration,
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            arguments=copy.deepcopy(tool_call.arguments),
        )

        workspace_change_tracker = None
        workspace_snapshot_before = None
        if not is_human_input_tool_name(tool_call.name):
            workspace_change_tracker = _workspace_change_tracker(context, toolkit, tool_call)
            if workspace_change_tracker is not None:
                workspace_snapshot_before = workspace_change_tracker.capture_text_snapshot()

        plugin_outcome = run_tool_runtime_plugins(
            context.tool_runtime_plugins,
            tool_call=tool_call,
            context=context,
            execution_guard=context.execution_guard,
        )
        if context.execution_guard is not None:
            context.execution_guard.assert_active()
        if plugin_outcome is not None:
            builder = get_provider_message_builder(context.provider)
            result_messages = copy_messages(batch_state.result_messages)
            visible_tool_result = plugin_outcome.tool_result
            authored_artifacts: list[dict] = []
            if isinstance(plugin_outcome.tool_result, dict):
                visible_tool_result, authored_artifacts = extract_authored_artifacts(plugin_outcome.tool_result)
            if plugin_outcome.result_messages:
                result_messages.extend(copy_messages(plugin_outcome.result_messages))
            elif isinstance(visible_tool_result, dict):
                result_messages.extend(
                    builder.build_tool_result_messages(tool_call=tool_call, tool_result=visible_tool_result)
                )
            emitted_artifacts: list[dict] = []
            if isinstance(visible_tool_result, dict):
                emitted_artifacts = _canonical_artifacts_for_tool_result(
                    context,
                    tool_call,
                    visible_tool_result,
                    authored_artifacts,
                )
                emit_loop_event(
                    context.loop,
                    context.callback,
                    "tool_result",
                    context.run_id,
                    iteration=context.iteration,
                    tool_name=tool_call.name,
                    call_id=tool_call.call_id,
                    result=copy.deepcopy(visible_tool_result),
                )
            artifact_ops = _create_artifact_ops(
                emitted_artifacts,
                reason="tool_runtime_plugin_artifact",
            )
            state_updates = copy.deepcopy(plugin_outcome.state_updates)
            if workspace_change_tracker is not None:
                workspace_change_tracker.record_text_snapshot_changes(
                    workspace_snapshot_before,
                    tool_name=tool_call.name,
                    call_id=tool_call.call_id,
                    turn_id=f"{context.run_id}:turn-{context.iteration}",
                )
                state_updates.update(_workspace_change_state_update(workspace_change_tracker))
            state_updates["tool_batch_state"] = ToolBatchState(
                result_messages=result_messages,
                should_observe=batch_state.should_observe or bool(plugin_outcome.should_observe),
                awaiting_human_input=False,
                human_input_request=batch_state.human_input_request,
                human_input_tool_call_id=batch_state.human_input_tool_call_id,
                executed_call_ids=append_executed_call_id(batch_state, tool_call.call_id),
            )
            if artifact_ops:
                return CapabilityOutcome(
                    delta=RunDelta(
                        created_by=self.created_by,
                        context_ops=artifact_ops,
                        state_updates=state_updates,
                        trace={
                            "tool_call": tool_call.name,
                            "call_id": tool_call.call_id,
                        },
                        suspend=plugin_outcome.suspend_override,
                    )
                )
            return HarnessDelta(
                created_by=self.created_by,
                state_updates=state_updates,
                suspend=plugin_outcome.suspend_override,
            )

        if is_human_input_tool_name(tool_call.name):
            try:
                request = parse_human_input_request(tool_call)
            except Exception as exc:
                builder = get_provider_message_builder(context.provider)
                tool_result = {
                    "error": str(exc),
                    "tool": tool_call.name,
                }
                result_messages = copy_messages(batch_state.result_messages)
                result_messages.append(
                    builder.build_tool_result_message(tool_call=tool_call, tool_result=tool_result)
                )
                emit_loop_event(
                    context.loop,
                    context.callback,
                    "tool_result",
                    context.run_id,
                    iteration=context.iteration,
                    tool_name=tool_call.name,
                    call_id=tool_call.call_id,
                    result=tool_result,
                )
                return HarnessDelta(
                    created_by=self.created_by,
                    state_updates={
                        "tool_batch_state": ToolBatchState(
                            result_messages=result_messages,
                            should_observe=batch_state.should_observe,
                            awaiting_human_input=False,
                            human_input_request=batch_state.human_input_request,
                            human_input_tool_call_id=batch_state.human_input_tool_call_id,
                            executed_call_ids=append_executed_call_id(batch_state, tool_call.call_id),
                        ),
                    },
                )

            return CapabilityOutcome(
                delta=HarnessDelta(
                    created_by=INTERACTION_EFFECT_CREATED_BY,
                    state_updates={
                        "tool_batch_state": ToolBatchState(
                            result_messages=copy_messages(batch_state.result_messages),
                            should_observe=False,
                            awaiting_human_input=True,
                            human_input_request=request,
                            human_input_tool_call_id=tool_call.call_id,
                            executed_call_ids=append_executed_call_id(batch_state, tool_call.call_id),
                        ),
                    },
                    trace={
                        "tool_call": tool_call.name,
                        "call_id": tool_call.call_id,
                    },
                ),
            )

        outcome = execute_confirmable_tool_call(
            toolkit=toolkit,
            tool_call=tool_call,
            on_tool_confirm=on_tool_confirm,
            loop=context.loop,
            callback=context.callback,
            run_id=context.run_id,
            iteration=context.iteration,
            execution_context=ToolExecutionContext(
                session_id=context.session_id,
                run_id=context.run_id,
                provider=context.provider,
                model=context.model,
                iteration=context.iteration,
                memory_namespace=context.memory_namespace,
                tool_runtime_config=context.event.get("tool_runtime_config")
                if isinstance(context.event.get("tool_runtime_config"), dict)
                else {},
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
                turn_id=f"{context.run_id}:turn-{context.iteration}",
                workspace_changes=workspace_change_tracker,
            ),
            execution_guard=context.execution_guard,
        )
        if context.execution_guard is not None:
            context.execution_guard.assert_active()
        if workspace_change_tracker is not None:
            workspace_change_tracker.record_text_snapshot_changes(
                workspace_snapshot_before,
                tool_name=tool_call.name,
                call_id=tool_call.call_id,
                turn_id=f"{context.run_id}:turn-{context.iteration}",
            )
        should_observe = batch_state.should_observe or outcome.should_observe
        builder = get_provider_message_builder(context.provider)
        visible_tool_result, authored_artifacts = extract_authored_artifacts(outcome.tool_result)
        emitted_artifacts = []
        if isinstance(visible_tool_result, dict):
            emitted_artifacts = _canonical_artifacts_for_tool_result(
                context,
                tool_call,
                visible_tool_result,
                authored_artifacts,
                confirmation_policy=outcome.confirmation_policy,
                effective_arguments=outcome.effective_arguments,
            )
        result_messages = copy_messages(batch_state.result_messages)
        result_messages.extend(
            builder.build_tool_result_messages(tool_call=tool_call, tool_result=visible_tool_result)
        )
        emit_loop_event(
            context.loop,
            context.callback,
            "tool_result",
            context.run_id,
            iteration=context.iteration,
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            result=copy.deepcopy(visible_tool_result),
        )
        artifact_ops = _create_artifact_ops(
            emitted_artifacts,
            reason="tool_result_artifact",
        )
        state_updates = {
            "tool_batch_state": ToolBatchState(
                result_messages=result_messages,
                should_observe=should_observe,
                awaiting_human_input=False,
                human_input_request=batch_state.human_input_request,
                human_input_tool_call_id=batch_state.human_input_tool_call_id,
                executed_call_ids=append_executed_call_id(batch_state, tool_call.call_id),
            ),
            **_workspace_change_state_update(workspace_change_tracker),
        }
        capability_delta = (
            outcome.capability_outcome.delta
            if outcome.capability_outcome is not None
            else None
        )
        capability_state_updates: dict[str, Any] = {}
        context_ops = artifact_ops
        created_by = (
            outcome.capability_outcome.created_by
            if outcome.capability_outcome is not None
            else self.created_by
        )
        trace: dict[str, Any] = {
            "tool_call": tool_call.name,
            "call_id": tool_call.call_id,
        }
        suspend = None
        if isinstance(capability_delta, RunDelta):
            capability_state_updates = copy.deepcopy(capability_delta.state_updates)
            context_ops = tuple(capability_delta.context_ops) + artifact_ops
            created_by = capability_delta.created_by or created_by or self.created_by
            trace = {
                **copy.deepcopy(capability_delta.trace),
                "tool_call": tool_call.name,
                "call_id": tool_call.call_id,
            }
            if capability_delta.created_by:
                trace["capability_created_by"] = capability_delta.created_by
            if created_by != self.created_by:
                trace["applied_by"] = self.created_by
            suspend = capability_delta.suspend

        if context_ops or capability_state_updates or suspend is not None:
            return CapabilityOutcome(
                delta=RunDelta(
                    created_by=created_by,
                    context_ops=context_ops,
                    state_updates={
                        **capability_state_updates,
                        **state_updates,
                    },
                    trace=trace,
                    suspend=suspend,
                ),
            )
        return HarnessDelta(
            created_by=self.created_by,
            state_updates=state_updates,
        )

    def _after_tool_batch(self, context: ToolContext) -> HarnessDelta | None:
        payload = copy.deepcopy(context.event.get("payload") or {})
        response_format = context.event.get("response_format")
        batch_state = context.state.tool_batch_state.copy()

        if batch_state.awaiting_human_input and batch_state.human_input_request is not None:
            continuation = build_human_input_continuation(
                request=batch_state.human_input_request,
                payload=payload,
                response_format=response_format,
                next_iteration=context.iteration + 1,
                max_iterations=int(context.event.get("max_iterations") or 0),
                state=context.state,
                run_id=context.run_id,
            )
            suspend_ops = (
                build_human_input_suspend_request(
                    batch_state.human_input_request,
                    continuation=continuation,
                ),
            )
            return CapabilityOutcome(
                delta=RunDelta(
                    created_by=INTERACTION_EFFECT_CREATED_BY,
                    context_ops=suspend_ops,
                    state_updates={
                        "run_status": "awaiting_human_input",
                        "last_continuation": continuation,
                        "pending_tool_calls": [],
                        "next_model_input": None,
                        "remote_continuation_input": None,
                        "tool_batch_state": batch_state,
                    },
                    trace={"awaiting_human_input": True},
                ),
            )

        result_messages = copy_messages(batch_state.result_messages)
        token_state = {}
        if batch_state.should_observe and result_messages:
            observation = ""
            observation_model_io = context.model_io
            if context.execution_guard is not None and observation_model_io is not None:
                observation_model_io = context.execution_guard.guard_model_io(
                    observation_model_io,
                    "model.tool_observation",
                )
            observation, observe_usage = ToolObservationRunner(
                model_io=observation_model_io,
            ).observe_tool_batch(
                full_messages=context.state.transcript,
                tool_messages=result_messages,
                payload=payload,
                iteration=context.iteration,
                provider=context.provider,
            )
            if context.execution_guard is not None:
                context.execution_guard.assert_active()
            token_state = observation_token_state(
                consumed_tokens=context.state.token_state.consumed_tokens,
                input_tokens=context.state.token_state.input_tokens,
                output_tokens=context.state.token_state.output_tokens,
                observe_usage=observe_usage,
            )
            if observation:
                inject_observation(result_messages[-1], observation)
                emit_loop_event(
                    context.loop,
                    context.callback,
                    "observation",
                    context.run_id,
                    iteration=context.iteration,
                    content=observation,
                )

        budget_stats: dict[str, Any] | None = None
        tool_runtime_config = context.event.get("tool_runtime_config")
        raw_budget_config = (
            tool_runtime_config.get("tool_result_budget")
            if isinstance(tool_runtime_config, dict)
            else None
        )
        budget_config = ToolResultBudgetConfig.from_raw(raw_budget_config)
        budget_outcome = ToolResultBudgetController(budget_config).budget_messages(
            provider=context.provider,
            toolkit=context.toolkit,
            tool_calls=list(context.event.get("tool_calls") or []),
            result_messages=result_messages,
            session_id=context.session_id,
            latest_messages=context.state.transcript,
        )
        result_messages = budget_outcome.messages
        result_messages = coalesce_provider_tool_result_messages(
            context.provider,
            result_messages,
        )
        budget_stats = budget_outcome.stats.to_dict()

        if not result_messages:
            return HarnessDelta(
                created_by=self.created_by,
                state_updates={
                    "pending_tool_calls": [],
                    "tool_batch_state": ToolBatchState(),
                    "run_status": context.state.run_status,
                    "last_continuation": None,
                    "next_model_input": None,
                    "remote_continuation_input": None,
                },
            )

        next_model_input = None
        remote_continuation_input = None
        if context.provider == "openai" and context.state.provider_state.use_previous_response_chain:
            if context.state.provider_state.previous_response_id:
                remote_continuation_input = copy_messages(result_messages)
            if isinstance(context.state.next_model_input, list):
                next_model_input = copy_messages(context.state.next_model_input)
                next_model_input.extend(copy_messages(result_messages))
        elif isinstance(context.state.next_model_input, list):
            next_model_input = copy_messages(context.state.next_model_input)
            next_model_input.extend(copy_messages(result_messages))

        return HarnessDelta.append(
            created_by=self.created_by,
            messages=result_messages,
            state_updates={
                "transcript_append": result_messages,
                "pending_tool_calls": [],
                "tool_batch_state": ToolBatchState(),
                "run_status": "running",
                "last_continuation": None,
                "next_model_input": next_model_input,
                "remote_continuation_input": remote_continuation_input,
                "provider_replay_append": result_messages,
                "token_state": token_state,
                "suspend_state": {
                    "signal_kind": None,
                    "payload": {},
                },
                **({"optimizer_state": {"tool_result_budget": budget_stats}} if budget_stats is not None else {}),
            },
            trace={
                "result_message_count": len(result_messages),
                "observed": bool(batch_state.should_observe),
                **({"tool_result_budget": budget_stats} if budget_stats is not None else {}),
            },
        )
