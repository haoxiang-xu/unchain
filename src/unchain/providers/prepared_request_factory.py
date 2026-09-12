"""Resolve native adapter semantics into a canonical prepared request."""

from __future__ import annotations

import copy
from typing import Any

from .base import ModelTurnRequest
from .wire_preparer import (
    ProviderWirePreparationError,
    build_prepared_provider_request_payload,
)


_SUPPORTED_PROVIDERS = frozenset(
    {"openai", "anthropic", "hyperspace", "ollama", "gemini"}
)


def _required_callable(value: object, name: str):
    method = getattr(value, name, None)
    if not callable(method):
        raise TypeError(f"model_io must provide {name}()")
    return method


def _response_format_value(response_format: object, method_name: str) -> Any:
    method = getattr(response_format, method_name, None)
    if not callable(method):
        raise TypeError(f"response_format must provide {method_name}()")
    return method()


def resolve_prepared_provider_request_payload(
    *,
    model_io: object,
    request: ModelTurnRequest,
) -> dict[str, Any]:
    """Freeze the effective native-provider request before durable preparation."""

    if type(request) is not ModelTurnRequest:
        raise TypeError("request must be an exact ModelTurnRequest")

    provider = getattr(model_io, "provider", None)
    if provider not in _SUPPORTED_PROVIDERS:
        raise ProviderWirePreparationError("provider is unsupported")
    configured_model = getattr(model_io, "model", None)
    if not isinstance(configured_model, str) or not configured_model.strip():
        raise ProviderWirePreparationError(
            "model_io must expose a non-empty configured model"
        )

    merged_payload = _required_callable(model_io, "_merged_payload")(request.payload)
    if type(merged_payload) is not dict:
        raise TypeError("model_io effective payload must be an exact object")
    effective_payload = copy.deepcopy(merged_payload)
    request_model = configured_model.strip()
    response_format: dict[str, Any] = {"kind": "none", "value": None}

    if provider == "openai":
        supports_reasoning = bool(
            _required_callable(model_io, "_model_capability")(
                "supports_reasoning", False
            )
        )
        if effective_payload.get("store") is False and supports_reasoning:
            raw_include = effective_payload.get("include")
            include = (
                [item for item in raw_include if isinstance(item, str)]
                if isinstance(raw_include, list)
                else []
            )
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            effective_payload["include"] = include

        openai_format = request.openai_text_format
        if openai_format is None and request.response_format is not None:
            openai_format = _response_format_value(request.response_format, "to_openai")
        if openai_format is not None:
            response_format = {
                "kind": "openai_text",
                "value": copy.deepcopy(openai_format),
            }
    elif provider == "gemini":
        if request.response_format is not None:
            from .gemini_schema import sanitize_gemini_schema

            value = _response_format_value(request.response_format, "to_gemini")
            effective_payload.update(value)
            effective_payload["response_schema"] = sanitize_gemini_schema(
                value["response_schema"]
            )
    elif provider in {"anthropic", "hyperspace"}:
        request_model = _required_callable(model_io, "_provider_request_model")()
        if not isinstance(request_model, str) or not request_model.strip():
            raise ProviderWirePreparationError(
                "provider request model must be non-empty text"
            )
        request_model = request_model.strip()
        if "max_tokens" not in effective_payload:
            effective_payload["max_tokens"] = _required_callable(
                model_io, "_model_capability"
            )("max_output_tokens", 4096)
        if request.response_format is not None:
            response_format = {
                "kind": "anthropic_instruction",
                "value": _response_format_value(
                    request.response_format, "to_anthropic"
                ),
            }
    else:
        if request.response_format is not None:
            response_format = {
                "kind": "ollama_format",
                "value": _response_format_value(request.response_format, "to_ollama"),
            }

    return build_prepared_provider_request_payload(
        provider=provider,
        messages=request.copied_messages(),
        effective_payload=effective_payload,
        request_model=request_model,
        response_format=response_format,
        previous_response_id=request.previous_response_id,
        fallback_messages=request.fallback_messages,
        context_mode=request.context_mode,
    )


__all__ = ["resolve_prepared_provider_request_payload"]
