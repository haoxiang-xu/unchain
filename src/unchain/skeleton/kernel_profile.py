"""Bind a closed Kernel Loop profile to one fresh native provider adapter.

This is deliberately a small host boundary.  It admits the provider-free
Kernel configuration first, compares the definition profile with the host's
resolved profile, and only then resolves a credential for the new adapter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .kernel_config import validate_kernel_config


_NATIVE_PROTOCOLS = {
    "openai": "openai-responses",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "ollama": "ollama",
    "hyperspace": "anthropic",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _frozen_json(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: _frozen_json(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_frozen_json(item) for item in value)
    return value


def _thawed_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thawed_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thawed_json(item) for item in value]
    return deepcopy(value)


def _config_snapshot(config: object) -> dict[str, Any]:
    if type(config) is not dict:
        return validate_kernel_config(config)
    # Node callers naturally hold this value under ``config.kernel`` while
    # KernelLoop callers already hold the kernel object.  Both still go
    # through the same closed validator.
    if set(config) == {"kernel"}:
        return validate_kernel_config(config["kernel"], ("kernel",))
    return validate_kernel_config(config)


def _resolve_profile(
    config: dict[str, Any], profile_resolver: Callable[[str], object]
) -> dict[str, Any]:
    if not callable(profile_resolver):
        raise TypeError("profile_resolver must be callable")
    definition = config["profile"]
    resolved = profile_resolver(definition["id"])
    # Validate the host object with the same closed contract before comparing
    # it.  This rejects a resolver that adds fields or changes JSON types.
    candidate = deepcopy(config)
    candidate["profile"] = resolved
    admitted = validate_kernel_config(candidate)
    if admitted["profile"] != definition:
        raise ValueError("resolved profile does not exactly match the Kernel definition")
    return deepcopy(admitted["profile"])


def _transport_for(profile: dict[str, Any]) -> str:
    brand = profile["provider"].lower()
    protocol = profile["protocol"]
    expected = _NATIVE_PROTOCOLS.get(brand)
    if expected is not None and protocol != expected:
        raise ValueError("native provider brand contradicts its protocol")
    if protocol == "openai-responses":
        return "openai"
    if protocol == "anthropic":
        # The public Anthropic brand is the sole route that uses the official
        # adapter.  Every shipped or custom Anthropic-compatible route is
        # endpoint-bound through HyperspaceModelIO.
        return "anthropic" if brand == "anthropic" else "hyperspace"
    return protocol


def _response_format(schema: dict[str, Any] | None, *, protocol: str):
    # OpenAI, Anthropic, and Gemini receive strict structured-output settings
    # directly in ``native_payload``.  Their existing ResponseFormat adapters
    # respectively duplicate/alter that setting (and Anthropic would append a
    # prompt instruction), so the runtime must pass null for those protocols.
    if schema is None or protocol != "ollama":
        return None
    from ..schemas import ResponseFormat

    return ResponseFormat(name="kernel_output", schema=deepcopy(schema))


def _native_payload(
    *, protocol: str, output_tokens: int, context_window: int, schema: dict[str, Any] | None
) -> dict[str, Any]:
    if protocol == "openai-responses":
        payload: dict[str, Any] = {"max_output_tokens": output_tokens}
        if schema is not None:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "kernel_output",
                    "schema": deepcopy(schema),
                    "strict": True,
                }
            }
        return payload
    if protocol == "anthropic":
        payload = {"max_tokens": output_tokens}
        if schema is not None:
            payload["output_config"] = {
                "format": {"type": "json_schema", "schema": deepcopy(schema)}
            }
        return payload
    if protocol == "gemini":
        payload = {"max_output_tokens": output_tokens}
        if schema is not None:
            payload["response_mime_type"] = "application/json"
            payload["response_json_schema"] = deepcopy(schema)
        return payload
    if protocol == "ollama":
        return {"num_predict": output_tokens, "num_ctx": context_window}
    raise ValueError("unsupported Kernel provider protocol")


def _catalog(model: str, payload: dict[str, Any], *, output_tokens: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    # Give each new adapter one exact key.  This prevents the providers'
    # compatibility matching from selecting a shared/global default entry.
    return (
        {model: deepcopy(payload)},
        {
            model: {
                "provider_model": model,
                "max_output_tokens": output_tokens,
                "supports_tools": True,
                "allowed_payload_keys": list(payload),
            }
        },
    )


def _pinned_sdk_factory(factory: Callable[..., Any], endpoint: str) -> Callable[..., Any]:
    def create(*, api_key: str, **kwargs: Any) -> Any:
        options = dict(kwargs)
        options["base_url"] = endpoint
        options["max_retries"] = 0
        return factory(api_key=api_key, **options)

    return create


def _pinned_gemini_factory(factory: Callable[..., Any], endpoint: str) -> Callable[..., Any]:
    def create(*, api_key: str, **kwargs: Any) -> Any:
        options = dict(kwargs)
        http_options = dict(options.pop("http_options", {}) or {})
        http_options["base_url"] = endpoint
        http_options["retry_options"] = {"attempts": 1}
        return factory(api_key=api_key, http_options=http_options, **options)

    return create


def _pinned_ollama_stream(
    stream_factory: Callable[..., Any], endpoint: str, credential: str | None
) -> Callable[..., Any]:
    def stream(method: str, _url: str, **kwargs: Any) -> Any:
        options = dict(kwargs)
        if credential is not None:
            headers = dict(options.get("headers", {}) or {})
            headers["Authorization"] = f"Bearer {credential}"
            options["headers"] = headers
        return stream_factory(method, f"{endpoint.rstrip('/')}/api/chat", **options)

    return stream


@dataclass(frozen=True)
class KernelProfileBinding:
    """Credential-safe, immutable metadata around one fresh native adapter."""

    config: Mapping[str, Any]
    model_io: Any = field(repr=False, compare=False)
    transport_provider: str
    native_payload: Mapping[str, Any]
    response_format: Any | None = field(repr=False, compare=False)
    profile_digest: str

    def payload_copy(self) -> dict[str, Any]:
        """Return a detached native payload suitable for ``ModelTurnRequest``."""
        return _thawed_json(self.native_payload)

    @property
    def provider(self) -> str:
        """The provider identity expected by the Kernel runtime."""
        return self.transport_provider

    @property
    def payload(self) -> dict[str, Any]:
        """A detached native request payload for one Kernel turn."""
        return self.payload_copy()


def bind_kernel_profile(
    config: object,
    *,
    profile_resolver: Callable[[str], object],
    credential_resolver: Callable[[str], object],
    client_factories: dict[str, Callable[..., Any]] | None = None,
) -> KernelProfileBinding:
    """Bind one fixed Kernel profile without consulting provider catalogs.

    Profile admission and exact host comparison happen before credential lookup.
    Native provider imports intentionally remain inside this function so merely
    loading Skeleton validation never imports an SDK or client implementation.
    """
    admitted = _config_snapshot(config)
    profile = _resolve_profile(admitted, profile_resolver)
    transport = _transport_for(profile)
    if not callable(credential_resolver):
        raise TypeError("credential_resolver must be callable")
    if client_factories is not None and type(client_factories) is not dict:
        raise TypeError("client_factories must be an exact object or null")
    factories = client_factories or {}
    allowed_factory_keys = {transport, profile["provider"]}
    for key, value in factories.items():
        if type(key) is not str or key not in allowed_factory_keys:
            raise ValueError("client factory key does not match this profile transport")
        if not callable(value):
            raise TypeError("client factory must be callable")

    raw_credential = credential_resolver(profile["credential_ref"])
    if transport == "ollama" and raw_credential is None:
        credential: str | None = None
    elif isinstance(raw_credential, str) and raw_credential.strip():
        credential = raw_credential
    else:
        raise ValueError("credential resolver returned no usable credential")

    schema = deepcopy(admitted["json_schema"])
    payload = _native_payload(
        protocol=profile["protocol"],
        output_tokens=admitted["output_tokens"],
        context_window=profile["context_window"],
        schema=schema,
    )
    defaults, capabilities = _catalog(
        profile["model"], payload, output_tokens=admitted["output_tokens"]
    )
    factory = factories.get(transport, factories.get(profile["provider"]))
    endpoint = profile["endpoint"].rstrip("/")

    if transport == "openai":
        from ..providers.openai import OpenAIModelIO
        from openai import OpenAI

        model_io = OpenAIModelIO(
            model=profile["model"], api_key=credential,
            client_factory=_pinned_sdk_factory(factory if callable(factory) else OpenAI, endpoint),
            default_payloads=defaults, model_capabilities=capabilities,
        )
    elif transport == "anthropic":
        from ..providers.anthropic import AnthropicModelIO
        import anthropic

        model_io = AnthropicModelIO(
            model=profile["model"], api_key=credential,
            client_factory=_pinned_sdk_factory(
                factory if callable(factory) else anthropic.Anthropic, endpoint
            ),
            default_payloads=defaults, model_capabilities=capabilities,
        )
    elif transport == "hyperspace":
        from ..providers.hyperspace import HyperspaceModelIO
        import anthropic

        model_io = HyperspaceModelIO(
            model=profile["model"], api_key=credential, base_url=endpoint,
            client_factory=_pinned_sdk_factory(
                factory if callable(factory) else anthropic.Anthropic, endpoint
            ),
            default_payloads=defaults, model_capabilities=capabilities,
        )
    elif transport == "gemini":
        from ..providers.gemini import GeminiModelIO
        from google import genai

        model_io = GeminiModelIO(
            model=profile["model"], api_key=credential,
            client_factory=_pinned_gemini_factory(
                factory if callable(factory) else genai.Client, endpoint
            ),
            default_payloads=defaults, model_capabilities=capabilities,
        )
    else:
        from ..providers.ollama import OllamaModelIO
        import httpx

        model_io = OllamaModelIO(
            model=profile["model"], base_url=endpoint,
            stream_factory=_pinned_ollama_stream(
                factory if callable(factory) else httpx.stream, endpoint, credential
            ),
            default_payloads=defaults, model_capabilities=capabilities,
        )

    return KernelProfileBinding(
        config=_frozen_json(admitted),
        model_io=model_io,
        transport_provider=transport,
        native_payload=_frozen_json(payload),
        response_format=_response_format(schema, protocol=profile["protocol"]),
        # This contains only the opaque credential reference, never resolved
        # secret bytes.  Changing a selected host identity changes the digest.
        profile_digest=sha256(_canonical_bytes(profile)).hexdigest(),
    )


__all__ = ["KernelProfileBinding", "bind_kernel_profile"]
