from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..interaction import hydrate_human_input_resume_state, prepare_human_input_resume_plan
from ..schemas import ResponseFormat
from .state import RunState

SUPPORTED_PROVIDERS = {"openai", "anthropic", "ollama", "hyperspace", "gemini"}


@dataclass(frozen=True)
class RunInvocationPlan:
    state: RunState
    payload: dict[str, Any]
    run_id: str


@dataclass(frozen=True)
class ResumeInvocationPlan:
    state: RunState
    payload: dict[str, Any]
    response_format: ResponseFormat | None
    run_id: str
    max_iterations: int


def infer_provider(model_io: Any) -> str | None:
    if model_io is None:
        return None
    provider = getattr(model_io, "provider", None)
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    engine = getattr(model_io, "engine", None)
    engine_provider = getattr(engine, "provider", None)
    if isinstance(engine_provider, str) and engine_provider.strip():
        return engine_provider.strip()
    if model_io.__class__.__name__ == "OpenAIModelIO":
        return "openai"
    return None


def infer_model(model_io: Any) -> str | None:
    if model_io is None:
        return None
    model = getattr(model_io, "model", None)
    if isinstance(model, str) and model.strip():
        return model.strip()
    engine = getattr(model_io, "engine", None)
    engine_model = getattr(engine, "model", None)
    if isinstance(engine_model, str) and engine_model.strip():
        return engine_model.strip()
    return None


def validate_provider(provider: str) -> str:
    if provider not in SUPPORTED_PROVIDERS:
        raise NotImplementedError(
            "KernelLoop.run currently supports only provider in "
            "{'openai', 'anthropic', 'ollama', 'hyperspace'}, "
            f"got {provider!r}"
        )
    return provider


def effective_payload_store(model_io: Any, payload: dict[str, Any]) -> Any:
    if "store" in payload:
        return payload.get("store")
    merged_payload = getattr(model_io, "_merged_payload", None)
    if callable(merged_payload):
        try:
            return merged_payload(payload).get("store")
        except Exception:
            return None
    return None


def prepare_fresh_run_invocation(
    *,
    messages: list[dict[str, Any]],
    payload: dict[str, Any] | None,
    model_io: Any,
    provider: str | None,
    model: str | None,
    previous_response_id: str | None,
    session_id: str | None,
    memory_namespace: str | None,
    max_context_window_tokens: int | None,
    run_id: str | None,
    run_id_factory: Callable[[], str],
) -> RunInvocationPlan:
    resolved_payload = dict(payload or {})
    resolved_provider = validate_provider(str(provider or infer_provider(model_io) or "openai"))
    resolved_model = model or infer_model(model_io)
    resolved_run_id = str(run_id or run_id_factory())

    state = RunState()
    state.seed_messages(messages)
    state.provider_state.provider = resolved_provider
    state.provider_state.model = resolved_model
    state.provider_state.max_context_window_tokens = max(
        0,
        int(max_context_window_tokens or 0),
    )
    state.provider_state.previous_response_id = previous_response_id
    state.provider_state.use_previous_response_chain = (
        resolved_provider == "openai"
        and effective_payload_store(model_io, resolved_payload) is not False
    )
    state.session_state.session_id = session_id
    state.session_state.memory_namespace = memory_namespace
    state.run_status = "running"

    return RunInvocationPlan(
        state=state,
        payload=resolved_payload,
        run_id=resolved_run_id,
    )


def prepare_state_for_execution(state: RunState, *, model_io: Any) -> None:
    provider = validate_provider(str(state.provider_state.provider or infer_provider(model_io) or ""))
    state.provider_state.provider = provider
    if not state.provider_state.model:
        state.provider_state.model = infer_model(model_io)


def prepare_resume_run_invocation(
    *,
    conversation: list[dict[str, Any]],
    continuation: dict[str, Any],
    payload: dict[str, Any] | None,
    response_format: ResponseFormat | None,
    fallback_provider: str | None,
    fallback_model: str | None,
    session_id: str | None,
    memory_namespace: str | None,
    run_id: str | None,
    run_id_factory: Callable[[], str],
) -> ResumeInvocationPlan:
    plan = prepare_human_input_resume_plan(
        conversation=conversation,
        continuation=continuation,
        payload=payload,
        response_format=response_format,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        session_id=session_id,
        memory_namespace=memory_namespace,
        run_id=run_id,
        run_id_factory=run_id_factory,
    )
    state = RunState()
    state.seed_messages(plan.conversation)
    hydrate_human_input_resume_state(state, plan)
    return ResumeInvocationPlan(
        state=state,
        payload=plan.payload,
        response_format=plan.response_format,
        run_id=plan.run_id,
        max_iterations=plan.max_iterations,
    )


__all__ = [
    "ResumeInvocationPlan",
    "RunInvocationPlan",
    "SUPPORTED_PROVIDERS",
    "effective_payload_store",
    "infer_model",
    "infer_provider",
    "prepare_fresh_run_invocation",
    "prepare_resume_run_invocation",
    "prepare_state_for_execution",
    "validate_provider",
]
