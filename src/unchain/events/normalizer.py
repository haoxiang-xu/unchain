from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .types import RuntimeEventLinks, RuntimeEventSurface, Visibility


@dataclass(frozen=True)
class RuntimeEventNormalizerContext:
    session_id: str
    root_run_id: str
    root_agent_id: str = "developer"


@dataclass(frozen=True)
class RuntimeEventDraft:
    type: str
    run_id: str
    agent_id: str
    turn_id: str | None = None
    links: RuntimeEventLinks = field(default_factory=RuntimeEventLinks)
    surface: RuntimeEventSurface = field(default_factory=RuntimeEventSurface)
    visibility: Visibility = "user"
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _str_value(value: Any, fallback: str = "") -> str:
    if isinstance(value, str) and value:
        return value
    return fallback


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _turn_id(run_id: str, raw: dict[str, Any]) -> str | None:
    iteration = _int_value(raw.get("iteration"))
    if iteration is None:
        return None
    return f"{run_id}:turn-{iteration}"


def _base_run_id(raw: dict[str, Any], context: RuntimeEventNormalizerContext) -> str:
    return _str_value(raw.get("run_id"), context.root_run_id)


def _base_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"raw_type": _str_value(raw.get("type"), "event")}
    provider = raw.get("provider")
    if isinstance(provider, str) and provider:
        metadata["provider"] = provider
    return metadata


def _status_from_tool_result(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("denied") is True:
            return "denied"
        if result.get("error") is not None:
            return "error"
    return "success"


def _trace_surface(group: str = "trace", *, default_state: str = "collapsed") -> RuntimeEventSurface:
    return RuntimeEventSurface(
        slot="trace_inline",
        scope="turn",
        group=group,
        default_state=default_state,
        priority=100,
    )


def _debug_surface() -> RuntimeEventSurface:
    return RuntimeEventSurface(
        slot="debug",
        scope="run",
        group="debug",
        default_state="hidden",
        priority=1000,
    )


def _artifact_payload(raw: dict[str, Any]) -> dict[str, Any]:
    artifact = raw.get("artifact")
    if isinstance(artifact, dict):
        return copy.deepcopy(artifact)

    payload: dict[str, Any] = {}
    reserved = {
        "type",
        "run_id",
        "iteration",
        "tool_name",
        "call_id",
        "tool_call_id",
        "plan_id",
        "artifact",
    }
    for key, value in raw.items():
        if key in reserved:
            continue
        if key == "artifact_id" or key in {
            "schema_version",
            "kind",
            "title",
            "summary",
            "revision",
            "status",
            "owner",
            "snapshot",
            "source",
            "presentation",
        }:
            payload[key] = copy.deepcopy(value)
    return payload


def _artifact_surface(artifact: dict[str, Any]) -> RuntimeEventSurface:
    presentation = artifact.get("presentation")
    presentation = presentation if isinstance(presentation, dict) else {}
    surface_name = _str_value(presentation.get("surface"))
    group = _str_value(presentation.get("group"), _str_value(artifact.get("kind"), "artifact"))
    collapsed = presentation.get("collapsed")
    default_state = "collapsed" if collapsed is True else "expanded"
    if surface_name == "run_summary":
        return RuntimeEventSurface(
            slot="run_summary",
            scope="run",
            group=group,
            default_state=default_state,
            priority=50,
        )
    return RuntimeEventSurface(
        slot="iteration_summary",
        scope="turn",
        group=group,
        default_state=default_state,
        priority=100,
    )


def _renderer_from_selection_mode(selection_mode: str, kind: str = "") -> str:
    if selection_mode in {"single", "multi", "text_input"}:
        return selection_mode
    if selection_mode in {"multiple", "multi_select"}:
        return "multi"
    if kind in {"text", "freeform"}:
        return "text_input"
    return "confirmation"


def _interaction_payload(raw: dict[str, Any], *, raw_type: str) -> tuple[str, dict[str, Any]]:
    interaction_id = _str_value(
        raw.get("confirmation_id"),
        _str_value(raw.get("request_id"), _str_value(raw.get("call_id"))),
    )
    interact_type = _str_value(raw.get("interact_type"))
    selection_mode = _str_value(raw.get("selection_mode"))
    raw_kind = _str_value(raw.get("kind"))

    if raw_type == "continuation_request":
        kind = "continuation"
        renderer = "confirmation"
    elif interact_type == "code_diff":
        kind = "code_diff"
        renderer = "code_diff"
    elif interact_type:
        renderer = interact_type
        kind = "choice" if interact_type in {"single", "multi"} else interact_type
    elif raw_type == "human_input_requested":
        renderer = _renderer_from_selection_mode(selection_mode, raw_kind)
        kind = "choice" if renderer in {"single", "multi"} else "text" if renderer == "text_input" else "confirmation"
    else:
        kind = "confirmation"
        renderer = "confirmation"

    options = raw.get("options")
    target_arguments = raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {}
    payload: dict[str, Any] = {
        "interaction_id": interaction_id,
        "kind": kind,
        "blocking": True,
        "renderer": renderer,
        "title": _str_value(raw.get("title"), _str_value(raw.get("description"))),
        "prompt": _str_value(raw.get("question")),
        "options": copy.deepcopy(options) if isinstance(options, list) else [],
        "target": {
            "tool_call_id": _str_value(raw.get("call_id")),
            "tool_name": _str_value(raw.get("tool_name")),
            "toolkit_id": _str_value(raw.get("toolkit_id")),
            "arguments": copy.deepcopy(target_arguments),
        },
        "config": copy.deepcopy(raw.get("interact_config")) if isinstance(raw.get("interact_config"), dict) else {},
    }
    if selection_mode:
        payload["selection_mode"] = selection_mode
        payload["config"].setdefault("selection_mode", selection_mode)
    for key in ("allow_other", "other_label", "other_placeholder", "min_selected", "max_selected"):
        if key in raw:
            payload[key] = copy.deepcopy(raw[key])
    return interaction_id, payload


def normalize_raw_event(
    raw_event: dict[str, Any],
    *,
    context: RuntimeEventNormalizerContext,
) -> list[RuntimeEventDraft]:
    if not isinstance(raw_event, dict):
        return []
    raw_type = _str_value(raw_event.get("type"))
    if not raw_type:
        return []

    run_id = _base_run_id(raw_event, context)
    agent_id = context.root_agent_id
    turn_id = _turn_id(run_id, raw_event)
    metadata = _base_metadata(raw_event)

    if raw_type == "run_started":
        return [
            RuntimeEventDraft(
                type="run.started",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                surface=_debug_surface(),
                visibility="debug",
                payload={
                    "status": "running",
                    "provider": _str_value(raw_event.get("provider")),
                    "model": _str_value(raw_event.get("model")),
                },
                metadata=metadata,
            )
        ]

    if raw_type == "run_completed":
        payload: dict[str, Any] = {"status": _str_value(raw_event.get("status"), "completed")}
        if isinstance(raw_event.get("bundle"), dict):
            payload["usage"] = copy.deepcopy(raw_event["bundle"])
        return [
            RuntimeEventDraft(
                type="run.completed",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                surface=_debug_surface(),
                visibility="debug",
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type == "run_failed":
        return [
            RuntimeEventDraft(
                type="run.failed",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                surface=_debug_surface(),
                payload={
                    "status": "failed",
                    "error": {
                        "code": _str_value(raw_event.get("code"), "run_failed"),
                        "message": _str_value(raw_event.get("message"), "Run failed"),
                    },
                    "recoverable": bool(raw_event.get("recoverable", False)),
                },
                metadata=metadata,
            )
        ]

    if raw_type == "iteration_started":
        return [
            RuntimeEventDraft(
                type="turn.started",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                surface=_debug_surface(),
                visibility="debug",
                payload={"iteration": raw_event.get("iteration")},
                metadata=metadata,
            )
        ]

    if raw_type == "iteration_completed":
        return [
            RuntimeEventDraft(
                type="turn.completed",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                surface=_debug_surface(),
                visibility="debug",
                payload={
                    "iteration": raw_event.get("iteration"),
                    "has_tool_calls": bool(raw_event.get("has_tool_calls", False)),
                },
                metadata=metadata,
            )
        ]

    if raw_type == "request_messages":
        step_id = f"model:{turn_id or run_id}:request"
        payload = {
            "step_id": step_id,
            "step_type": "model_request",
            "provider": _str_value(raw_event.get("provider")),
            "model": _str_value(raw_event.get("model")),
            "messages": copy.deepcopy(raw_event.get("messages")) if isinstance(raw_event.get("messages"), list) else [],
            "tool_names": copy.deepcopy(raw_event.get("tool_names")) if isinstance(raw_event.get("tool_names"), list) else [],
        }
        previous_response_id = _str_value(raw_event.get("previous_response_id"))
        if previous_response_id:
            payload["previous_response_id"] = previous_response_id
        return [
            RuntimeEventDraft(
                type="step.started",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(step_id=step_id),
                surface=_debug_surface(),
                visibility="debug",
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type in {"token_delta", "reasoning"}:
        kind = "reasoning" if raw_type == "reasoning" else "text"
        delta = _str_value(raw_event.get("delta"), _str_value(raw_event.get("content")))
        step_id = f"model:{turn_id or run_id}:response"
        payload = {
            "step_id": step_id,
            "step_type": "model_response",
            "kind": kind,
            "delta": delta,
        }
        accumulated_text = raw_event.get("accumulated_text")
        if isinstance(accumulated_text, str):
            payload["accumulated_text"] = accumulated_text
        return [
            RuntimeEventDraft(
                type="step.delta",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(step_id=step_id),
                surface=_trace_surface("model"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type in {"response_received", "final_message"}:
        step_id = f"model:{turn_id or run_id}:response"
        payload: dict[str, Any] = {
            "step_id": step_id,
            "step_type": "model_response",
            "status": _str_value(raw_event.get("status"), "completed"),
        }
        if raw_type == "response_received":
            payload["response_id"] = _str_value(raw_event.get("response_id"))
            payload["has_tool_calls"] = bool(raw_event.get("has_tool_calls", False))
            if isinstance(raw_event.get("bundle"), dict):
                payload["usage"] = copy.deepcopy(raw_event["bundle"])
        else:
            payload["final_text"] = _str_value(raw_event.get("content"))
        return [
            RuntimeEventDraft(
                type="step.completed",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(step_id=step_id),
                surface=_trace_surface("model"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type == "tool_call":
        call_id = _str_value(raw_event.get("call_id"))
        step_id = f"tool:{call_id or 'unknown'}"
        payload: dict[str, Any] = {
            "step_id": step_id,
            "step_type": "tool",
            "tool_name": _str_value(raw_event.get("tool_name"), "tool"),
            "call_id": call_id,
        }
        for key in (
            "tool_display_name",
            "toolkit_id",
            "toolkit_name",
            "description",
            "confirmation_id",
            "requires_confirmation",
            "interact_type",
            "interact_config",
            "arguments",
        ):
            if key in raw_event:
                payload[key] = copy.deepcopy(raw_event[key])
        return [
            RuntimeEventDraft(
                type="step.started",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(step_id=step_id, tool_call_id=call_id or None),
                surface=_trace_surface("tool"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type == "tool_result":
        call_id = _str_value(raw_event.get("call_id"))
        step_id = f"tool:{call_id or 'unknown'}"
        result = copy.deepcopy(raw_event.get("result"))
        payload = {
            "step_id": step_id,
            "step_type": "tool",
            "tool_name": _str_value(raw_event.get("tool_name"), "tool"),
            "call_id": call_id,
            "status": _status_from_tool_result(result),
            "result": result,
        }
        display_name = _str_value(raw_event.get("tool_display_name"))
        if display_name:
            payload["tool_display_name"] = display_name
        return [
            RuntimeEventDraft(
                type="step.completed",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(step_id=step_id, tool_call_id=call_id or None),
                surface=_trace_surface("tool"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type in {"tool_confirmation_requested", "input_requested", "continuation_request", "human_input_requested"}:
        interaction_id, payload = _interaction_payload(raw_event, raw_type=raw_type)
        call_id = _str_value(raw_event.get("call_id"))
        return [
            RuntimeEventDraft(
                type="interaction.requested",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(
                    interaction_id=interaction_id or None,
                    tool_call_id=call_id or None,
                    step_id=f"tool:{call_id}" if call_id else None,
                ),
                surface=_trace_surface("interaction", default_state="expanded"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type == "fyi_injected":
        messages = raw_event.get("messages")
        payload = {
            "count": _int_value(raw_event.get("count")) or 0,
            "messages": copy.deepcopy(messages) if isinstance(messages, list) else [],
        }
        return [
            RuntimeEventDraft(
                type="interaction.fyi_injected",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                surface=_trace_surface("interaction"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type in {"tool_confirmed", "tool_denied"}:
        call_id = _str_value(raw_event.get("call_id"))
        interaction_id = _str_value(
            raw_event.get("confirmation_id"),
            _str_value(raw_event.get("request_id"), call_id),
        )
        response = copy.deepcopy(raw_event.get("user_response"))
        if response is None and "response" in raw_event:
            response = copy.deepcopy(raw_event.get("response"))
        if raw_type == "tool_denied":
            outcome = "denied"
        else:
            outcome = "submitted" if response is not None else "approved"
        payload = {
            "interaction_id": interaction_id,
            "outcome": outcome,
            "response": response,
            "reason": _str_value(raw_event.get("reason")),
        }
        return [
            RuntimeEventDraft(
                type="interaction.resolved",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(
                    interaction_id=interaction_id or None,
                    tool_call_id=call_id or None,
                    step_id=f"tool:{call_id}" if call_id else None,
                ),
                surface=_trace_surface("interaction"),
                payload=payload,
                metadata=metadata,
            )
        ]

    if raw_type in {"artifact_created", "artifact_updated"}:
        artifact = _artifact_payload(raw_event)
        artifact_id = _str_value(artifact.get("artifact_id"), _str_value(raw_event.get("artifact_id")))
        snapshot = artifact.get("snapshot") if isinstance(artifact.get("snapshot"), dict) else {}
        plan_id = _str_value(raw_event.get("plan_id"), _str_value(snapshot.get("plan_id")))
        change_set_id = _str_value(snapshot.get("change_set_id"))
        call_id = _str_value(raw_event.get("call_id"), _str_value(raw_event.get("tool_call_id")))
        return [
            RuntimeEventDraft(
                type="artifact.created" if raw_type == "artifact_created" else "artifact.updated",
                run_id=run_id,
                agent_id=agent_id,
                turn_id=turn_id,
                links=RuntimeEventLinks(
                    tool_call_id=call_id or None,
                    step_id=f"tool:{call_id}" if call_id else None,
                    artifact_id=artifact_id or None,
                    workspace_change_set_id=change_set_id or None,
                    plan_id=plan_id or None,
                ),
                surface=_artifact_surface(artifact),
                payload=artifact,
                metadata=metadata,
            )
        ]

    if raw_type in {"subagent_started", "subagent_completed", "subagent_failed"}:
        child_run_id = _str_value(raw_event.get("child_run_id"))
        subagent_id = _str_value(raw_event.get("subagent_id"), child_run_id or agent_id)
        parent_run_id = _str_value(raw_event.get("root_run_id"), context.root_run_id)
        event_type = {
            "subagent_started": "run.started",
            "subagent_completed": "run.completed",
            "subagent_failed": "run.failed",
        }[raw_type]
        payload: dict[str, Any] = {
            "agent_id": subagent_id,
            "parent_id": _str_value(raw_event.get("parent_id")),
            "mode": _str_value(raw_event.get("mode")),
            "template": _str_value(raw_event.get("template")),
            "lineage": copy.deepcopy(raw_event.get("lineage")) if isinstance(raw_event.get("lineage"), list) else [],
        }
        if raw_type == "subagent_completed":
            payload["status"] = _str_value(raw_event.get("status"), "completed")
        if raw_type == "subagent_failed":
            payload["status"] = _str_value(raw_event.get("status"), "failed")
            payload["error"] = {
                "code": _str_value(raw_event.get("code"), "subagent_failed"),
                "message": _str_value(raw_event.get("message"), "Subagent failed"),
            }
        return [
            RuntimeEventDraft(
                type=event_type,
                run_id=child_run_id,
                agent_id=subagent_id,
                links=RuntimeEventLinks(parent_run_id=parent_run_id or None),
                surface=_debug_surface(),
                payload=payload,
                metadata=metadata,
            )
        ]

    return []
