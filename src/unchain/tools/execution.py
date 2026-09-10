from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
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
from ..interaction.requests import build_human_interaction_request
from ..interaction.effects import (
    build_tool_approval_continuation,
    build_tool_approval_suspend_request,
)
from ..interaction.durable import InteractionIntegrityError
from ..interaction.requests import (
    build_tool_approval_interaction_request,
    ensure_interaction_binding_matches,
)
from ..toolkits.base import BuiltinToolkit
from ..workspace_changes import WorkspaceChangeTracker
from .toolkit import Toolkit
from ..kernel.delta import HarnessDelta
from ..kernel.run_ledger import (
    provider_call_route,
    record_model_turn,
    record_unobserved_model_attempt,
    request_sha256,
)
from ..kernel.types import ToolCall
from ..run_bundle import (
    RunMetricEvidenceRef,
    canonical_sha256,
    opaque_metric_evidence_ref,
)
from .base import BaseToolHarness, ToolContext
from .common import append_executed_call_id, copy_messages, emit_loop_event
from .confirmation import execute_confirmable_tool_call, prepare_tool_confirmation
from .human_input import parse_human_input_request
from .messages import coalesce_provider_tool_result_messages, get_provider_message_builder
from .models import ToolExecutionContext
from .observation import ToolObservationRunner, inject_observation, observation_token_state
from .result_budget import ToolResultBudgetConfig, ToolResultBudgetController
from .runtime import (
    run_tool_runtime_plugins,
    snapshot_durable_tool_exposure_plan,
    snapshot_durable_tool_runtime_route,
)
from .types import ToolBatchState
from .web_fetch_guard import web_fetch_failure_limit


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


def _durable_tool_exposure_plan(context: ToolContext) -> dict[str, Any] | None:
    return snapshot_durable_tool_exposure_plan(context.tool_runtime_plugins)


def _durable_tool_runtime_subject(
    context: Any,
    tool_call: ToolCall,
    *,
    extra_subject: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    subject = copy.deepcopy(extra_subject) if isinstance(extra_subject, dict) else {}
    route = snapshot_durable_tool_runtime_route(
        getattr(context, "tool_runtime_plugins", None),
        tool_call=tool_call,
        context=context,
    )
    if route is not None:
        subject["tool_runtime_route"] = route
    return subject or None


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

    @staticmethod
    def _record_tool_result_metric(
        context: ToolContext,
        tool_call: ToolCall,
        result: Any,
        artifacts: list[dict] | None = None,
    ) -> None:
        if getattr(context.state.run_ledger, "identity", None) is None:
            return
        failed = isinstance(result, dict) and is_failed_tool_result(result)
        refs: dict[str, RunMetricEvidenceRef] = {}
        for artifact in artifacts or []:
            if not isinstance(artifact, dict):
                continue
            ref_id = str(
                artifact.get("artifact_id")
                or artifact.get("id")
                or artifact.get("logical_key")
                or ""
            ).strip()
            if ref_id:
                revision = artifact.get("revision")
                logical_occurrence = (
                    f"{ref_id}:revision:{revision}"
                    if isinstance(revision, int) and not isinstance(revision, bool)
                    else ref_id
                )
                ref = opaque_metric_evidence_ref(
                    kind="artifact",
                    source_id=logical_occurrence,
                )
                refs[ref.ref_id] = ref
        ordered_refs = tuple(refs[key] for key in sorted(refs))
        for ref in ordered_refs:
            context.state.run_ledger.record_metric_event(
                kind="artifact",
                subject_id=ref.ref_id,
                outcome="completed",
                evidence_refs=(ref,),
            )
        manifest_ref = (
            opaque_metric_evidence_ref(
                kind="artifact",
                source_id=(
                    "manifest:"
                    + canonical_sha256([ref.ref_id for ref in ordered_refs])
                ),
            )
            if ordered_refs
            else None
        )
        context.state.run_ledger.record_metric_event(
            kind="tool_result",
            subject_id=tool_call.call_id,
            outcome="failed" if failed else "completed",
            error_category="tool" if failed else None,
            error_code="tool_result_failed" if failed else None,
            evidence_refs=(manifest_ref,) if manifest_ref is not None else (),
        )
        if failed:
            context.state.run_ledger.record_metric_event(
                kind="error",
                subject_id=f"tool-result:{tool_call.call_id}",
                outcome="failed",
                error_category="tool",
                error_code="tool_result_failed",
            )

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
        if getattr(context.state.run_ledger, "identity", None) is not None:
            for tool_call in tool_calls:
                context.state.run_ledger.record_metric_event(
                    kind="tool_call",
                    subject_id=tool_call.call_id,
                )

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
            for tool_call in tool_calls:
                self._record_tool_result_metric(
                    context,
                    tool_call,
                    {"error": error_text, "tool": tool_call.name},
                )
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
            for tool_call in tool_calls:
                self._record_tool_result_metric(
                    context,
                    tool_call,
                    {"error": error_text, "tool": tool_call.name},
                )
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
        if (
            tool_call.call_id in batch_state.executed_call_ids
            or batch_state.awaiting_human_input
            or context.state.run_status == "awaiting_interaction"
        ):
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

        execution_context = ToolExecutionContext(
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
            run_state=context.state,
        )
        confirmation_preparation = prepare_tool_confirmation(
            toolkit=toolkit,
            tool_call=tool_call,
            execution_context=execution_context,
            execution_guard=context.execution_guard,
        )
        interaction_response_supplied = (
            "interaction_response" in context.raw_event
        )
        stored_interaction_request = context.raw_event.get("interaction_request")
        stored_request_bound = False
        if (
            confirmation_preparation.needs_confirmation_response
            and context.session_id
            and getattr(context.loop, "interaction_runtime", None) is not None
        ):
            confirmation_request = confirmation_preparation.request
            if confirmation_request is None:
                raise RuntimeError(
                    "tool confirmation preparation requires a request"
                )
            current_interaction_request = build_tool_approval_interaction_request(
                state=context.state,
                toolkit=toolkit,
                tool_call=tool_call,
                confirmation_request=confirmation_request,
                run_id=context.run_id,
                source_request=(
                    stored_interaction_request
                    if isinstance(stored_interaction_request, dict)
                    else None
                ),
                extra_subject=_durable_tool_runtime_subject(
                    context,
                    tool_call,
                ),
            )
            if isinstance(stored_interaction_request, dict):
                ensure_interaction_binding_matches(
                    stored_interaction_request,
                    current_interaction_request,
                )
                if not interaction_response_supplied:
                    raise RuntimeError(
                        "durable tool approval resume requires a receipt response"
                    )
                stored_request_bound = True
            else:
                continuation = build_tool_approval_continuation(
                    interaction_request=current_interaction_request,
                    payload=copy.deepcopy(context.event.get("payload") or {}),
                    response_format=context.event.get("response_format"),
                    max_iterations=int(context.event.get("max_iterations") or 0),
                    tool_calls=list(context.event.get("tool_calls") or []),
                    batch_state=batch_state,
                    state=context.state,
                    run_id=context.run_id,
                    tool_exposure_plan=_durable_tool_exposure_plan(context),
                )
                suspend_op = build_tool_approval_suspend_request(
                    interaction_request=current_interaction_request,
                    continuation=continuation,
                )
                return CapabilityOutcome(
                    delta=RunDelta(
                        created_by=INTERACTION_EFFECT_CREATED_BY,
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
                            "interaction_id": (
                                current_interaction_request.interaction_id
                            ),
                        },
                    )
                )

        nested_confirmation_policy = None
        if (
            not stored_request_bound
            and context.session_id
            and getattr(context.loop, "interaction_runtime", None) is not None
        ):
            for plugin in context.tool_runtime_plugins:
                can_handle = getattr(plugin, "can_handle", None)
                prepare_nested = getattr(
                    plugin,
                    "prepare_durable_confirmation",
                    None,
                )
                if (
                    not callable(can_handle)
                    or not can_handle(tool_call=tool_call, context=context)
                    or not callable(prepare_nested)
                ):
                    continue
                nested = prepare_nested(tool_call=tool_call, context=context)
                if not isinstance(nested, dict):
                    break
                nested_preparation = nested.get("preparation")
                nested_confirmation_policy = getattr(
                    nested_preparation,
                    "confirmation_policy",
                    None,
                )
                nested_request = getattr(nested_preparation, "request", None)
                if not bool(
                    getattr(
                        nested_preparation,
                        "needs_confirmation_response",
                        False,
                    )
                ) or nested_request is None:
                    if isinstance(stored_interaction_request, dict):
                        raise InteractionIntegrityError(
                            "tool confirmation policy changed while approval was pending"
                        )
                    break
                nested_tool_call = nested.get("tool_call")
                nested_toolkit = nested.get("toolkit")
                if not isinstance(nested_tool_call, ToolCall) or not isinstance(
                    nested_toolkit,
                    Toolkit,
                ):
                    raise RuntimeError(
                        "tool runtime plugin returned invalid confirmation preparation"
                    )
                interaction_request = build_tool_approval_interaction_request(
                    state=context.state,
                    toolkit=nested_toolkit,
                    tool_call=nested_tool_call,
                    confirmation_request=nested_request,
                    run_id=context.run_id,
                    source_request=(
                        stored_interaction_request
                        if isinstance(stored_interaction_request, dict)
                        else None
                    ),
                    extra_subject=_durable_tool_runtime_subject(
                        nested.get("runtime_context", context),
                        nested_tool_call,
                        extra_subject=(
                            nested.get("extra_subject")
                            if isinstance(nested.get("extra_subject"), dict)
                            else None
                        ),
                    ),
                )
                if isinstance(stored_interaction_request, dict):
                    ensure_interaction_binding_matches(
                        stored_interaction_request,
                        interaction_request,
                    )
                    if not interaction_response_supplied:
                        raise InteractionIntegrityError(
                            "durable tool approval resume requires a receipt response"
                        )
                    stored_request_bound = True
                    break
                continuation = build_tool_approval_continuation(
                    interaction_request=interaction_request,
                    payload=copy.deepcopy(context.event.get("payload") or {}),
                    response_format=context.event.get("response_format"),
                    max_iterations=int(context.event.get("max_iterations") or 0),
                    tool_calls=list(context.event.get("tool_calls") or []),
                    batch_state=batch_state,
                    state=context.state,
                    run_id=context.run_id,
                    tool_exposure_plan=_durable_tool_exposure_plan(context),
                )
                suspend_op = build_tool_approval_suspend_request(
                    interaction_request=interaction_request,
                    continuation=continuation,
                )
                return CapabilityOutcome(
                    delta=RunDelta(
                        created_by=INTERACTION_EFFECT_CREATED_BY,
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
                            "tool_call": nested_tool_call.name,
                            "call_id": nested_tool_call.call_id,
                            "interaction_id": interaction_request.interaction_id,
                            "deferred_tool": True,
                        },
                    )
                )

        if (
            isinstance(stored_interaction_request, dict)
            and not stored_request_bound
        ):
            raise InteractionIntegrityError(
                "tool confirmation policy changed while approval was pending"
            )

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
                # Plugin execution still runs under the confirmation the
                # user approved: the nested preparation's policy when the
                # plugin routed one, else the outer preparation's. Dropping
                # it here silently lost every code_diff-derived file_diff
                # artifact on the plugin path.
                emitted_artifacts = _canonical_artifacts_for_tool_result(
                    context,
                    tool_call,
                    visible_tool_result,
                    authored_artifacts,
                    confirmation_policy=(
                        nested_confirmation_policy
                        if nested_confirmation_policy is not None
                        else confirmation_preparation.confirmation_policy
                    ),
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
            self._record_tool_result_metric(
                context,
                tool_call,
                visible_tool_result,
                emitted_artifacts,
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
                self._record_tool_result_metric(
                    context,
                    tool_call,
                    tool_result,
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

        execution_kwargs: dict[str, Any] = {
            "toolkit": toolkit,
            "tool_call": tool_call,
            "on_tool_confirm": on_tool_confirm,
            "loop": context.loop,
            "callback": context.callback,
            "run_id": context.run_id,
            "iteration": context.iteration,
            "execution_context": execution_context,
            "execution_guard": context.execution_guard,
            "prepared_confirmation": confirmation_preparation,
        }
        if interaction_response_supplied:
            execution_kwargs["confirmation_response"] = context.raw_event.get(
                "interaction_response"
            )
        outcome = execute_confirmable_tool_call(
            **execution_kwargs,
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
        self._record_tool_result_metric(
            context,
            tool_call,
            visible_tool_result,
            emitted_artifacts,
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

        if context.state.run_status == "awaiting_interaction":
            return None

        if batch_state.awaiting_human_input and batch_state.human_input_request is not None:
            interaction_request = None
            if (
                context.session_id
                and getattr(context.loop, "interaction_runtime", None) is not None
            ):
                interaction_request = build_human_interaction_request(
                    state=context.state,
                    request=batch_state.human_input_request,
                    run_id=context.run_id,
                )
            continuation = build_human_input_continuation(
                request=batch_state.human_input_request,
                payload=payload,
                response_format=response_format,
                next_iteration=context.iteration + 1,
                max_iterations=int(context.event.get("max_iterations") or 0),
                state=context.state,
                run_id=context.run_id,
                interaction_request=interaction_request,
                tool_exposure_plan=_durable_tool_exposure_plan(context),
            )
            suspend_ops = (
                build_human_input_suspend_request(
                    batch_state.human_input_request,
                    continuation=continuation,
                    interaction_request=interaction_request,
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
        fetch_limit = web_fetch_failure_limit(context.state.transcript + result_messages)
        if batch_state.should_observe and result_messages and not fetch_limit:
            observation = ""
            observation_model_io = context.model_io
            provider_turn_ownership = (
                context.state.run_ledger.provider_turn_ownership
            )
            if (
                provider_turn_ownership is None
                and context.execution_guard is not None
                and observation_model_io is not None
            ):
                observation_model_io = context.execution_guard.guard_model_io(
                    observation_model_io,
                    "model.tool_observation",
                )

            def observation_request_digest(request: Any) -> str:
                return request_sha256(
                    state=context.state,
                    payload=request.payload,
                    toolkit=request.toolkit,
                    response_format=request.response_format,
                    openai_text_format=request.openai_text_format,
                    provider=str(context.state.provider_state.provider or "unknown"),
                    model=str(
                        context.state.provider_state.model or "unknown-model"
                    ),
                    messages=request.messages,
                )

            def record_observation_turn(
                turn: Any,
                request: Any,
                occurred_at: str,
            ) -> None:
                record_model_turn(
                    context.state,
                    turn,
                    iteration=context.iteration,
                    retry_ordinal=0,
                    purpose="tool_observation",
                    request_digest=observation_request_digest(request),
                    payload=request.payload,
                    started_at=occurred_at,
                    completed_at=(
                        datetime.now(timezone.utc)
                        .isoformat(timespec="microseconds")
                        .replace("+00:00", "Z")
                    ),
                    route=provider_call_route(
                        str(context.state.provider_state.provider or "unknown")
                    ),
                )

            def record_observation_failure(
                _error: BaseException,
                request: Any,
                occurred_at: str,
            ) -> None:
                record_unobserved_model_attempt(
                    context.state,
                    iteration=context.iteration,
                    retry_ordinal=0,
                    purpose="tool_observation",
                    request_digest=observation_request_digest(request),
                    payload=request.payload,
                    started_at=occurred_at,
                    completed_at=(
                        datetime.now(timezone.utc)
                        .isoformat(timespec="microseconds")
                        .replace("+00:00", "Z")
                    ),
                    route=provider_call_route(
                        str(context.state.provider_state.provider or "unknown")
                    ),
                    status="uncertain",
                )

            fetch_observation_turn = None
            if provider_turn_ownership is not None:
                observation_occurrence_id = (
                    "tool_observation:"
                    + canonical_sha256(
                        {
                            "iteration": max(0, int(context.iteration)),
                            "call_ids": list(batch_state.executed_call_ids),
                        }
                    )
                )

                def fetch_observation_turn(request):
                    raw_digest = observation_request_digest(request)
                    return provider_turn_ownership.fetch_turn(
                        state=context.state,
                        model_io=observation_model_io,
                        request=request,
                        occurrence_id=observation_occurrence_id,
                        purpose="tool_observation",
                        iteration=context.iteration,
                        request_sha256=raw_digest,
                        retry_config=context.loop.retry_config,
                        before_attempt=(
                            (lambda _attempt: context.execution_guard.renew())
                            if context.execution_guard is not None
                            else None
                        ),
                        provider=str(
                            context.state.provider_state.provider or "unknown"
                        ),
                        model=str(
                            context.state.provider_state.model or "unknown-model"
                        ),
                    )

            observation, observe_usage = ToolObservationRunner(
                model_io=observation_model_io,
            ).observe_tool_batch(
                full_messages=context.state.transcript,
                tool_messages=result_messages,
                payload=payload,
                iteration=context.iteration,
                provider=context.provider,
                on_model_turn=(
                    None
                    if provider_turn_ownership is not None
                    else record_observation_turn
                ),
                on_model_failure=(
                    None
                    if provider_turn_ownership is not None
                    else record_observation_failure
                ),
                fetch_model_turn=fetch_observation_turn,
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
        output_manager = context.output_manager
        if output_manager.legacy_budget_enabled:
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
            budget_stats = budget_outcome.stats.to_dict()
        result_messages = coalesce_provider_tool_result_messages(
            context.provider,
            result_messages,
        )
        if fetch_limit:
            result_messages.append({"role": "assistant", "content": fetch_limit})

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
                "run_status": "completed" if fetch_limit else "running",
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
                "tool_output_projection": output_manager.active,
                **({"tool_result_budget": budget_stats} if budget_stats is not None else {}),
            },
        )
