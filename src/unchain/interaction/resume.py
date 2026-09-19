from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..input.human_input import ASK_USER_QUESTION_TOOL_NAME, HumanInputRequest, HumanInputResponse
from ..kernel.delta import HarnessDelta
from ..kernel.provider_replay import (
    set_provider_replay_frame,
    validate_provider_replay_frame,
)
from ..kernel.replay_handle import load_provider_replay_handle
from ..kernel.state import RunState
from ..kernel.types import ToolCall
from ..schemas import ResponseFormat
from ..tools.base import BaseToolHarness, ToolContext
from ..tools.common import emit_loop_event
from ..tools.messages import get_provider_message_builder
from ..tools.types import ToolBatchState

_SUPPORTED_HUMAN_INPUT_RESUME_PROVIDERS = {
    "openai",
    "anthropic",
    "ollama",
    "hyperspace",
    "gemini",
}


@dataclass(frozen=True)
class HumanInputResumePlan:
    conversation: list[dict[str, Any]]
    payload: dict[str, Any]
    response_format: ResponseFormat | None
    provider: str
    model: str | None
    session_id: str | None
    memory_namespace: str | None
    run_id: str
    iteration: int
    max_iterations: int
    previous_response_id: str | None
    use_previous_response_chain: bool
    max_context_window_tokens: int
    consumed_tokens: int
    input_tokens: int
    output_tokens: int
    last_turn_tokens: int
    last_turn_input_tokens: int
    last_turn_output_tokens: int
    workspace_change_state: dict[str, Any] | None
    provider_replay_frame: dict[str, Any] | None = None
    provider_replay_required: bool = False


def parse_human_input_request(tool_call: ToolCall) -> HumanInputRequest:
    return HumanInputRequest.from_tool_arguments(
        tool_call.arguments,
        request_id=tool_call.call_id,
    )


def _deserialize_response_format(raw: dict[str, Any] | None) -> ResponseFormat | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    schema = raw.get("schema")
    required = raw.get("required")
    if not isinstance(name, str) or not isinstance(schema, dict):
        return None
    required_list = required if isinstance(required, list) else None
    return ResponseFormat(name=name, schema=schema, required=required_list)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _int_from_continuation(continuation: dict[str, Any], key: str, default: int = 0) -> int:
    return int(continuation.get(key) or default)


def prepare_human_input_resume_plan(
    *,
    conversation: list[dict[str, Any]],
    continuation: dict[str, Any],
    payload: dict[str, Any] | None = None,
    response_format: ResponseFormat | None = None,
    fallback_provider: str | None = None,
    fallback_model: str | None = None,
    session_id: str | None = None,
    memory_namespace: str | None = None,
    run_id: str | None = None,
    run_id_factory: Callable[[], str] | None = None,
) -> HumanInputResumePlan:
    if not isinstance(conversation, list):
        raise TypeError("conversation must be a list of provider-projected messages")
    if not isinstance(continuation, dict):
        raise TypeError("continuation must be a dict returned by KernelRunResult.continuation")

    resolved_provider = str(continuation.get("provider") or fallback_provider or "")
    if resolved_provider not in _SUPPORTED_HUMAN_INPUT_RESUME_PROVIDERS:
        raise NotImplementedError(
            "human input resume currently supports only provider in "
            "{'openai', 'anthropic', 'ollama', 'hyperspace'}, "
            f"got {resolved_provider!r}"
        )

    expected_session_id = continuation.get("session_id")
    if isinstance(expected_session_id, str) and session_id is not None and session_id != expected_session_id:
        raise ValueError("resume_human_input requires the same session_id as the suspended run")

    resolved_run_id = str(run_id or continuation.get("run_id") or (run_id_factory or (lambda: str(uuid.uuid4())))())
    resolved_workspace_change_state = continuation.get("workspace_change_state")
    raw_replay_frame = continuation.get("provider_replay_frame")
    replay_handle = continuation.get("provider_replay_handle")
    replay_frame = (
        validate_provider_replay_frame(raw_replay_frame)
        if isinstance(raw_replay_frame, dict)
        else load_provider_replay_handle(replay_handle)
    )
    if isinstance(replay_frame, dict):
        replay_frame = validate_provider_replay_frame(replay_frame)

    return HumanInputResumePlan(
        conversation=copy.deepcopy(conversation),
        payload=dict(payload) if payload is not None else copy.deepcopy(continuation.get("payload") or {}),
        response_format=(
            response_format
            if response_format is not None
            else _deserialize_response_format(continuation.get("response_format"))
        ),
        provider=resolved_provider,
        model=_optional_str(continuation.get("model")) or fallback_model,
        session_id=session_id if session_id is not None else _optional_str(expected_session_id),
        memory_namespace=(
            memory_namespace if memory_namespace is not None else _optional_str(continuation.get("memory_namespace"))
        ),
        run_id=resolved_run_id,
        iteration=_int_from_continuation(continuation, "iteration"),
        max_iterations=_int_from_continuation(continuation, "max_iterations", 6),
        previous_response_id=_optional_str(continuation.get("previous_response_id")),
        use_previous_response_chain=bool(continuation.get("use_openai_previous_response_chain", False)),
        max_context_window_tokens=max(0, _int_from_continuation(continuation, "max_context_window_tokens")),
        consumed_tokens=_int_from_continuation(continuation, "consumed_tokens"),
        input_tokens=_int_from_continuation(continuation, "input_tokens"),
        output_tokens=_int_from_continuation(continuation, "output_tokens"),
        last_turn_tokens=_int_from_continuation(continuation, "last_turn_tokens"),
        last_turn_input_tokens=_int_from_continuation(continuation, "last_turn_input_tokens"),
        last_turn_output_tokens=_int_from_continuation(continuation, "last_turn_output_tokens"),
        workspace_change_state=(
            copy.deepcopy(resolved_workspace_change_state)
            if isinstance(resolved_workspace_change_state, dict)
            else None
        ),
        provider_replay_frame=replay_frame,
        provider_replay_required=bool(
            replay_handle
            or (
                continuation.get("use_openai_previous_response_chain", False)
                and continuation.get("previous_response_id")
            )
        ),
    )


def hydrate_human_input_resume_state(state: RunState, plan: HumanInputResumePlan) -> RunState:
    state.provider_state.provider = plan.provider
    state.provider_state.model = plan.model
    state.provider_state.max_context_window_tokens = plan.max_context_window_tokens
    state.session_state.session_id = plan.session_id
    state.session_state.memory_namespace = plan.memory_namespace
    state.iteration = plan.iteration
    state.provider_state.previous_response_id = plan.previous_response_id
    state.provider_state.use_previous_response_chain = plan.use_previous_response_chain
    state.token_state.consumed_tokens = plan.consumed_tokens
    state.token_state.input_tokens = plan.input_tokens
    state.token_state.output_tokens = plan.output_tokens
    state.token_state.last_turn_tokens = plan.last_turn_tokens
    state.token_state.last_turn_input_tokens = plan.last_turn_input_tokens
    state.token_state.last_turn_output_tokens = plan.last_turn_output_tokens
    if plan.workspace_change_state is not None:
        state.workspace_change_state = copy.deepcopy(plan.workspace_change_state)
        state.component_bucket("workspace_changes")["state"] = copy.deepcopy(plan.workspace_change_state)
    replay_frame = plan.provider_replay_frame
    if isinstance(replay_frame, dict):
        set_provider_replay_frame(state, replay_frame)
    elif plan.provider_replay_required:
        state.metadata["provider_replay_required"] = True
    state.run_status = "running"
    return state


@dataclass
class HumanInputResumeHarness(BaseToolHarness):
    name: str = "human_input_resume"
    phases: tuple[str, ...] = ("on_resume",)
    order: int = 100

    def build_tool_delta(self, context: ToolContext) -> HarnessDelta | None:
        continuation = context.event.get("continuation")
        response = context.event.get("response")
        if not isinstance(continuation, dict):
            raise TypeError("continuation must be a dict returned by KernelRunResult.continuation")
        if continuation.get("type") != "human_input_continuation":
            raise ValueError("continuation must be a human_input_continuation payload")

        request = HumanInputRequest.from_dict(continuation.get("request"))
        human_response = HumanInputResponse.from_raw(response, request=request)
        tool_call = ToolCall(
            call_id=str(continuation.get("call_id") or request.request_id),
            name=ASK_USER_QUESTION_TOOL_NAME,
            arguments={},
        )
        builder = get_provider_message_builder(context.provider)
        tool_result = human_response.to_tool_result()
        tool_message = builder.build_tool_result_message(
            tool_call=tool_call,
            tool_result=tool_result,
        )
        emit_loop_event(
            context.loop,
            context.callback,
            "tool_result",
            context.run_id,
            iteration=max(0, _int_from_continuation(continuation, "iteration") - 1),
            tool_name=tool_call.name,
            call_id=tool_call.call_id,
            result=copy.deepcopy(tool_result),
            interaction_id=str(continuation.get("interaction_id") or ""),
            interaction_resume_replay=True,
        )

        checkpoint_restored = bool(
            context.state.memory_state.get("execution_checkpoint_restored", False)
        )
        use_previous_response_chain = bool(
            continuation.get("use_openai_previous_response_chain", False)
        ) and not checkpoint_restored
        previous_response_id = (
            None if checkpoint_restored else continuation.get("previous_response_id")
        )
        remote_continuation_input = (
            [copy.deepcopy(tool_message)]
            if context.provider == "openai"
            and use_previous_response_chain
            and isinstance(previous_response_id, str)
            and previous_response_id
            else None
        )

        return HarnessDelta.append(
            created_by=self.created_by,
            messages=[tool_message],
            state_updates={
                "transcript_append": [tool_message],
                "pending_tool_calls": [],
                "tool_batch_state": ToolBatchState(),
                "run_status": "running",
                "last_continuation": None,
                "next_model_input": None,
                "remote_continuation_input": remote_continuation_input,
                "provider_replay_append": [tool_message],
                "provider_state": {
                    "previous_response_id": previous_response_id,
                    "use_previous_response_chain": use_previous_response_chain,
                },
                "suspend_state": {
                    "signal_kind": None,
                    "payload": {},
                },
            },
            trace={
                "request_id": request.request_id,
                "resumed_from_continuation": True,
                "execution_checkpoint_restored": checkpoint_restored,
            },
        )


__all__ = [
    "HumanInputResumePlan",
    "HumanInputResumeHarness",
    "hydrate_human_input_resume_state",
    "parse_human_input_request",
    "prepare_human_input_resume_plan",
]
