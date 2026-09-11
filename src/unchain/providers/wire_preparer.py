"""Pure construction of immutable provider wire envelopes."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.journal.models import AttemptRef

from .context_assembler import _openai_computer_call_semantic
from .prepared_turn import (
    FrozenProviderToolkit,
    PreparedProviderTurn,
    _consume_prepared_provider_turn,
)
from .native import (
    _translate_content_blocks_for_anthropic,
    _translate_content_blocks_for_ollama,
    _translate_content_blocks_for_openai,
)
from .wire_envelope import ProviderWireEnvelope, ProviderWireRoute


_PROFILES = {
    "gemini": (
        "unchain.gemini.contents.request.v1",
        "gemini.models.generate_content_stream",
    ),
    "openai": (
        "unchain.openai.responses.request.v1",
        "openai.responses.create",
    ),
    "anthropic": (
        "unchain.anthropic.messages.request.v1",
        "anthropic.messages.stream",
    ),
    "hyperspace": (
        "unchain.hyperspace.anthropic-messages.request.v1",
        "hyperspace.anthropic.messages.stream",
    ),
    "ollama": (
        "unchain.ollama.chat.request.v1",
        "ollama.api.chat.post",
    ),
}
_PREPARED_REQUEST_SCHEMA = "unchain.prepared_provider_request.v1"
_CONTEXT_MODES = frozenset({"semantic", "local_replay", "remote_continuation"})
_RESPONSE_FORMAT_KINDS = {
    "gemini": frozenset({"none"}),
    "openai": frozenset({"none", "openai_text"}),
    "anthropic": frozenset({"none", "anthropic_instruction"}),
    "hyperspace": frozenset({"none", "anthropic_instruction"}),
    "ollama": frozenset({"none", "ollama_format"}),
}


class ProviderWirePreparationError(ValueError):
    """The assembled turn cannot produce deterministic provider wire data."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ProviderWirePreparationError(
            "provider wire source must be strict canonical JSON"
        ) from exc


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(_canonical_bytes(value).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise ProviderWirePreparationError(
            "provider wire source JSON could not be detached"
        ) from exc


def build_prepared_provider_request_payload(
    *,
    provider: str,
    messages: list[dict[str, Any]],
    effective_payload: dict[str, Any],
    request_model: str,
    response_format: dict[str, Any],
    previous_response_id: str | None,
    fallback_messages: list[dict[str, Any]] | None,
    context_mode: str,
) -> dict[str, Any]:
    """Build the exact provider-neutral request frozen by a prepared turn."""

    if provider not in _PROFILES:
        raise ProviderWirePreparationError("provider is unsupported")
    copied_messages = _json_copy(messages)
    if type(copied_messages) is not list or not copied_messages:
        raise ProviderWirePreparationError(
            "prepared provider messages must be a non-empty exact array"
        )
    if any(type(message) is not dict for message in copied_messages):
        raise TypeError("prepared provider messages require exact JSON objects")
    copied_payload = _json_copy(effective_payload)
    if type(copied_payload) is not dict:
        raise TypeError("effective_payload must be an exact JSON object")
    if type(request_model) is not str or not request_model.strip():
        raise ProviderWirePreparationError("request_model must be non-empty text")
    if request_model != request_model.strip():
        raise ProviderWirePreparationError("request_model must be canonical text")

    copied_format = _json_copy(response_format)
    if type(copied_format) is not dict or set(copied_format) != {"kind", "value"}:
        raise ProviderWirePreparationError(
            "response_format must use the exact kind/value shape"
        )
    format_kind = copied_format.get("kind")
    format_value = copied_format.get("value")
    if format_kind not in _RESPONSE_FORMAT_KINDS[provider]:
        raise ProviderWirePreparationError(
            "response_format kind does not match provider"
        )
    if format_kind == "none":
        if format_value is not None:
            raise ProviderWirePreparationError("none response format requires null")
    elif format_kind == "openai_text":
        if type(format_value) is not dict:
            raise TypeError("OpenAI response format must be an exact JSON object")
    elif format_kind == "anthropic_instruction":
        if type(format_value) is not str or not format_value.strip():
            raise ProviderWirePreparationError(
                "Anthropic response instruction must be non-empty text"
            )
    elif type(format_value) not in {dict, str} or (
        type(format_value) is str and not format_value
    ):
        raise ProviderWirePreparationError(
            "Ollama response format must be non-empty JSON"
        )

    if context_mode not in _CONTEXT_MODES:
        raise ProviderWirePreparationError("context_mode is unsupported")
    copied_fallback = _json_copy(fallback_messages)
    if context_mode == "remote_continuation":
        if provider != "openai":
            raise ProviderWirePreparationError(
                "remote_continuation is only supported by OpenAI"
            )
        if type(previous_response_id) is not str or not previous_response_id:
            raise ProviderWirePreparationError(
                "remote_continuation requires previous_response_id"
            )
        if type(copied_fallback) is not list or not copied_fallback:
            raise ProviderWirePreparationError(
                "remote_continuation requires a non-empty fallback replay"
            )
        if any(type(message) is not dict for message in copied_fallback):
            raise TypeError("fallback replay requires exact JSON objects")
    elif previous_response_id is not None or copied_fallback is not None:
        raise ProviderWirePreparationError(
            "local context modes cannot carry remote continuation fields"
        )

    return {
        "schema": _PREPARED_REQUEST_SCHEMA,
        "provider": provider,
        "messages": copied_messages,
        "effective_payload": copied_payload,
        "request_model": request_model,
        "response_format": copied_format,
        "previous_response_id": previous_response_id,
        "fallback_messages": copied_fallback,
        "context_mode": context_mode,
    }


def _resolve_model_key(model: str, registry: dict[str, Any]) -> str | None:
    if model in registry:
        return model
    normalized_model = model.replace(".", "-")
    best: str | None = None
    for raw_key in registry:
        key = str(raw_key)
        normalized_key = key.replace(".", "-")
        if (
            model.startswith(key)
            or model.startswith(normalized_key)
            or normalized_model.startswith(key)
            or normalized_model.startswith(normalized_key)
            or key.startswith(model)
            or key.startswith(normalized_model)
            or normalized_key.startswith(model)
            or normalized_key.startswith(normalized_model)
        ) and (best is None or len(key) > len(best)):
            best = key
    return best


def _model_capabilities(preparation: _ProviderWirePreparationInput) -> dict[str, Any]:
    key = _resolve_model_key(
        preparation.configured_model,
        preparation.model_capabilities,
    )
    selected = preparation.model_capabilities.get(key, {}) if key else {}
    return copy.deepcopy(selected) if isinstance(selected, dict) else {}


def _merged_payload(preparation: _ProviderWirePreparationInput) -> dict[str, Any]:
    key = _resolve_model_key(
        preparation.configured_model,
        preparation.default_payloads,
    )
    selected = preparation.default_payloads.get(key, {}) if key else {}
    defaults = copy.deepcopy(selected) if isinstance(selected, dict) else {}
    user_payload = _json_copy(preparation.payload)
    if type(user_payload) is not dict:
        raise TypeError("payload must be an exact JSON object")
    for name in tuple(defaults):
        if name in user_payload:
            defaults[name] = user_payload[name]
    capabilities = _model_capabilities(preparation)
    allowed = capabilities.get("allowed_payload_keys")
    if isinstance(allowed, list) and allowed:
        allowed_names = {name for name in allowed if isinstance(name, str)}
        for name, value in user_payload.items():
            if name in allowed_names and name not in defaults:
                defaults[name] = value
        defaults = {
            name: value for name, value in defaults.items() if name in allowed_names
        }
    return {
        name: value
        for name, value in defaults.items()
        if value is not None or name in user_payload
    }


def _normalize_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    copied = _json_copy(messages)
    if type(copied) is not list:
        raise TypeError("messages must be an exact JSON array")
    for item in copied:
        if type(item) is not dict:
            raise TypeError("messages must contain exact JSON objects")
        item.pop("status", None)
        if item.get("type") == "computer_call":
            semantic = _openai_computer_call_semantic(item)
            item.clear()
            item.update(semantic)
        if item.get("type") == "function_call" and not (
            isinstance(item.get("call_id") or item.get("id"), str)
            and (item.get("call_id") or item.get("id"))
        ):
            raise ProviderWirePreparationError(
                "function_call requires a stable call id before wire preparation"
            )
        if item.get("type") == "function_call" and "call_id" not in item:
            item["call_id"] = item["id"]
    _translate_content_blocks_for_openai(copied)
    return copied


def _source_request_body(preparation: _ProviderWirePreparationInput) -> dict[str, Any]:
    return {
        "attempt": preparation.attempt.to_dict(),
        "iteration": preparation.iteration,
        "provider": preparation.provider,
        "configured_model": preparation.configured_model,
        "transport_target_sha256": preparation.transport_target_sha256,
        "messages": _json_copy(preparation.messages),
        "payload": _json_copy(preparation.payload),
        "toolkit_sha256": preparation.toolkit.toolkit_sha256,
        "catalog_sha256": preparation.catalog.catalog_sha256,
        "openai_text_format": _json_copy(preparation.openai_text_format),
        "anthropic_response_instruction": preparation.anthropic_response_instruction,
        "ollama_format": _json_copy(preparation.ollama_format),
        "previous_response_id": preparation.previous_response_id,
        "fallback_messages": _json_copy(preparation.fallback_messages),
    }


def _openai_routes(
    preparation: _ProviderWirePreparationInput,
    *,
    merged_payload: dict[str, Any],
    capabilities: dict[str, Any],
) -> tuple[ProviderWireRoute, ...]:
    messages = _normalize_openai_messages(preparation.messages)
    payload = copy.deepcopy(merged_payload)
    if payload.get("store") is False and capabilities.get("supports_reasoning", False):
        raw_include = payload.get("include")
        include = (
            [item for item in raw_include if isinstance(item, str)]
            if isinstance(raw_include, list)
            else []
        )
        if "reasoning.encrypted_content" not in include:
            include.append("reasoning.encrypted_content")
        payload["include"] = include
    request: dict[str, Any] = {
        "model": preparation.configured_model,
        "input": messages,
        **payload,
        "stream": True,
    }
    tools = preparation.toolkit.to_provider_json("openai")
    if tools and capabilities.get("supports_tools", True):
        request["tools"] = tools
    if preparation.openai_text_format is not None:
        text = dict(request["text"]) if isinstance(request.get("text"), dict) else {}
        text["format"] = _json_copy(preparation.openai_text_format)
        request["text"] = text
    routes: list[ProviderWireRoute] = []
    if preparation.previous_response_id is not None:
        if not preparation.previous_response_id:
            raise ProviderWirePreparationError("previous_response_id must be non-empty")
        if preparation.fallback_messages is None:
            raise ProviderWirePreparationError(
                "previous_response_id requires complete fallback messages"
            )
        request["previous_response_id"] = preparation.previous_response_id
        routes.append(ProviderWireRoute("primary", request))
        fallback = copy.deepcopy(request)
        fallback.pop("previous_response_id", None)
        fallback["input"] = _normalize_openai_messages(preparation.fallback_messages)
        routes.append(ProviderWireRoute("openai_previous_response_fallback", fallback))
        return tuple(routes)
    routes.append(ProviderWireRoute("primary", request))
    return tuple(routes)


def _anthropic_route(
    preparation: _ProviderWirePreparationInput,
    *,
    merged_payload: dict[str, Any],
    capabilities: dict[str, Any],
    request_model: str,
) -> tuple[ProviderWireRoute, tuple[str, ...]]:
    source_messages = _json_copy(preparation.messages)
    if type(source_messages) is not list:
        raise TypeError("messages must be an exact JSON array")
    system_parts: list[str] = []
    chat_messages: list[dict[str, Any]] = []
    for message in source_messages:
        if type(message) is not dict:
            raise TypeError("messages must contain exact JSON objects")
        if message.get("role") in {"system", "developer"}:
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                system_parts.append(content.strip())
            elif content not in (None, ""):
                system_parts.append(str(content))
            continue
        chat_messages.append(message)
    _translate_content_blocks_for_anthropic(chat_messages)
    instruction = preparation.anthropic_response_instruction
    if instruction:
        if not isinstance(instruction, str):
            raise TypeError("anthropic_response_instruction must be text")
        system_parts.append(instruction)
    if not chat_messages:
        raise ProviderWirePreparationError(
            "Anthropic request has no chat messages after preprocessing"
        )

    payload = copy.deepcopy(merged_payload)
    default_max = capabilities.get("max_output_tokens", 4096)
    max_tokens = payload.pop("max_tokens", default_max)
    request: dict[str, Any] = {
        "model": request_model,
        "messages": chat_messages,
        "max_tokens": max_tokens,
        **payload,
    }
    system_prompt = "\n\n".join(system_parts)
    if system_prompt:
        request["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    tools = preparation.toolkit.to_provider_json(preparation.provider)
    if tools and capabilities.get("supports_tools", True):
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        request["tools"] = tools

    required_betas = preparation.toolkit.required_betas(preparation.provider)
    raw_headers = request.get("extra_headers")
    if raw_headers is None:
        headers: dict[str, Any] = {}
    elif type(raw_headers) is dict:
        headers = copy.deepcopy(raw_headers)
    else:
        raise TypeError("extra_headers must be an exact JSON object")
    raw_base = headers.get("anthropic-beta")
    base_betas: list[str] = []
    if isinstance(raw_base, str):
        for beta in raw_base.split(","):
            normalized = beta.strip()
            if normalized and normalized not in base_betas:
                base_betas.append(normalized)
    elif raw_base is not None:
        raise TypeError("anthropic-beta must be text")
    merged_betas = list(base_betas)
    for beta in required_betas:
        if beta not in merged_betas:
            merged_betas.append(beta)
    if merged_betas:
        headers["anthropic-beta"] = ",".join(merged_betas)
        request["extra_headers"] = headers

    last = chat_messages[-1]
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    elif isinstance(content, list) and content:
        tail = content[-1]
        if not isinstance(tail, dict):
            raise ProviderWirePreparationError(
                "Anthropic message tail must be an object"
            )
        tail["cache_control"] = {"type": "ephemeral"}
    else:
        raise ProviderWirePreparationError(
            "Anthropic message tail requires cacheable content"
        )
    return ProviderWireRoute("primary", request), tuple(base_betas)


def _ollama_route(
    preparation: _ProviderWirePreparationInput,
    *,
    merged_payload: dict[str, Any],
    capabilities: dict[str, Any],
) -> ProviderWireRoute:
    messages = _json_copy(preparation.messages)
    if type(messages) is not list:
        raise TypeError("messages must be an exact JSON array")
    if any(type(message) is not dict for message in messages):
        raise TypeError("messages must contain exact JSON objects")
    _translate_content_blocks_for_ollama(messages)

    request: dict[str, Any] = {
        "model": preparation.configured_model,
        "messages": messages,
        "stream": True,
    }
    tools = preparation.toolkit.to_provider_json("ollama")
    if tools and capabilities.get("supports_tools", True):
        request["tools"] = tools
        request["tool_choice"] = "auto"
    if merged_payload:
        request["options"] = copy.deepcopy(merged_payload)
    if preparation.ollama_format is not None:
        request["format"] = _json_copy(preparation.ollama_format)
    return ProviderWireRoute("primary", request)


@dataclass(frozen=True, slots=True)
class _ProviderWirePreparationInput:
    """Complete provider-neutral input available after turn assembly."""

    attempt: AttemptRef
    iteration: int
    catalog: ToolCatalogEnvelope
    provider: str
    configured_model: str
    transport_target_sha256: str
    messages: list[dict[str, Any]]
    payload: dict[str, Any]
    toolkit: FrozenProviderToolkit
    default_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    openai_text_format: dict[str, Any] | None = None
    anthropic_response_instruction: str = ""
    ollama_format: dict[str, Any] | str | None = None
    previous_response_id: str | None = None
    fallback_messages: list[dict[str, Any]] | None = None
    source_request_sha256: str | None = None


def _prepare_provider_wire_from_input(
    preparation: _ProviderWirePreparationInput,
) -> ProviderWireEnvelope:
    if type(preparation) is not _ProviderWirePreparationInput:
        raise TypeError("preparation must be an exact internal input")
    if preparation.provider not in _PROFILES:
        raise ProviderWirePreparationError("provider is unsupported")
    if preparation.catalog.attempt != preparation.attempt:
        raise ProviderWirePreparationError("catalog attempt changed")
    if preparation.catalog.iteration != preparation.iteration:
        raise ProviderWirePreparationError("catalog iteration changed")
    if preparation.catalog.provider != preparation.provider:
        raise ProviderWirePreparationError("catalog provider changed")
    if preparation.catalog.model != preparation.configured_model:
        raise ProviderWirePreparationError("catalog model changed")
    if preparation.toolkit.provider != preparation.provider:
        raise ProviderWirePreparationError("toolkit provider changed")

    capabilities = _model_capabilities(preparation)
    merged_payload = _merged_payload(preparation)
    request_model = preparation.configured_model
    if preparation.provider in {"anthropic", "hyperspace"}:
        resolved = capabilities.get("provider_model", request_model)
        if isinstance(resolved, str) and resolved.strip():
            request_model = resolved.strip()

    if preparation.provider == "openai":
        routes = _openai_routes(
            preparation,
            merged_payload=merged_payload,
            capabilities=capabilities,
        )
        base_anthropic_betas: tuple[str, ...] = ()
    elif preparation.provider in {"anthropic", "hyperspace"}:
        route, base_anthropic_betas = _anthropic_route(
            preparation,
            merged_payload=merged_payload,
            capabilities=capabilities,
            request_model=request_model,
        )
        routes = (route,)
    elif preparation.provider == "gemini":
        routes = (_gemini_route(preparation, merged_payload=merged_payload),)
        base_anthropic_betas = ()
    elif preparation.provider == "ollama":
        routes = (
            _ollama_route(
                preparation,
                merged_payload=merged_payload,
                capabilities=capabilities,
            ),
        )
        base_anthropic_betas = ()
    else:
        raise NotImplementedError(
            f"{preparation.provider} wire preparation is not implemented"
        )

    adapter_revision, transport_kind = _PROFILES[preparation.provider]
    envelope = ProviderWireEnvelope(
        attempt=preparation.attempt,
        iteration=preparation.iteration,
        provider=preparation.provider,
        configured_model=preparation.configured_model,
        request_model=request_model,
        adapter_revision=adapter_revision,
        transport_kind=transport_kind,
        transport_target_sha256=preparation.transport_target_sha256,
        source_request_sha256=(
            preparation.source_request_sha256
            if preparation.source_request_sha256 is not None
            else _canonical_sha256(_source_request_body(preparation))
        ),
        source_payload_sha256=_canonical_sha256(merged_payload),
        catalog_sha256=preparation.catalog.catalog_sha256,
        prompt_sha256=preparation.catalog.prompt_sha256,
        tool_schema_sha256=preparation.catalog.tool_schema_sha256,
        required_betas=preparation.toolkit.required_betas(preparation.provider),
        base_anthropic_betas=base_anthropic_betas,
        routes=routes,
    )
    return envelope.verify_against_catalog(preparation.catalog)


def _decoded_prepared_request(draft: Any) -> dict[str, Any]:
    raw = draft._request_payload_copy()
    expected_fields = {
        "schema",
        "provider",
        "messages",
        "effective_payload",
        "request_model",
        "response_format",
        "previous_response_id",
        "fallback_messages",
        "context_mode",
    }
    if type(raw) is not dict or set(raw) != expected_fields:
        raise ProviderWirePreparationError(
            "prepared provider request uses an unsupported record shape"
        )
    if raw.get("schema") != _PREPARED_REQUEST_SCHEMA:
        raise ProviderWirePreparationError(
            "prepared provider request schema is unsupported"
        )
    rebuilt = build_prepared_provider_request_payload(
        provider=raw.get("provider"),
        messages=raw.get("messages"),
        effective_payload=raw.get("effective_payload"),
        request_model=raw.get("request_model"),
        response_format=raw.get("response_format"),
        previous_response_id=raw.get("previous_response_id"),
        fallback_messages=raw.get("fallback_messages"),
        context_mode=raw.get("context_mode"),
    )
    if rebuilt != raw or raw["provider"] != draft.provider:
        raise ProviderWirePreparationError(
            "prepared provider request changed from its provider turn"
        )
    return rebuilt


def prepare_provider_wire(
    prepared: PreparedProviderTurn,
    *,
    model_io: object,
    attempt: AttemptRef,
    iteration: int,
    transport_target_sha256: str,
) -> ProviderWireEnvelope:
    """Consume one exact prepared turn and construct its immutable wire bytes."""

    draft = _consume_prepared_provider_turn(
        prepared,
        model_io=model_io,
        attempt=attempt,
        iteration=iteration,
    )
    raw = _decoded_prepared_request(draft)
    response_format = raw["response_format"]
    format_kind = response_format["kind"]
    format_value = response_format["value"]
    capabilities = {
        draft.model: {
            "provider_model": raw["request_model"],
            "supports_tools": draft.toolkit.supports_tools,
        }
    }
    effective_payload = raw["effective_payload"]
    return _prepare_provider_wire_from_input(
        _ProviderWirePreparationInput(
            attempt=draft.attempt,
            iteration=draft.iteration,
            catalog=draft.catalog,
            provider=draft.provider,
            configured_model=draft.model,
            transport_target_sha256=transport_target_sha256,
            messages=raw["messages"],
            payload=effective_payload,
            toolkit=draft.toolkit,
            default_payloads={draft.model: effective_payload},
            model_capabilities=capabilities,
            openai_text_format=(format_value if format_kind == "openai_text" else None),
            anthropic_response_instruction=(
                format_value if format_kind == "anthropic_instruction" else ""
            ),
            ollama_format=(format_value if format_kind == "ollama_format" else None),
            previous_response_id=raw["previous_response_id"],
            fallback_messages=raw["fallback_messages"],
            source_request_sha256=draft._request_payload_sha256,
        )
    )


__all__ = [
    "ProviderWirePreparationError",
    "build_prepared_provider_request_payload",
    "prepare_provider_wire",
]


def _gemini_route(preparation, *, merged_payload):
    from .gemini import translate_gemini_messages

    contents, system = translate_gemini_messages(preparation.messages)
    config = _json_copy(merged_payload)
    config["automatic_function_calling"] = {"disable": True}
    if system:
        config["system_instruction"] = system
    request = {
        "model": preparation.configured_model,
        "contents": contents,
        "config": config,
    }
    tools = preparation.toolkit.to_provider_json("gemini")
    if tools:
        request["tools"] = tools
    return ProviderWireRoute("primary", request)
