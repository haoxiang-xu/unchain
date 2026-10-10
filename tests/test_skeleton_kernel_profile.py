from __future__ import annotations

import copy
import hashlib

import pytest

from unchain.kernel import ModelTurnRequest
from unchain.context.tool_catalog import ToolCatalogSnapshot
from unchain.journal.models import ArtifactRef, AttemptRef, EventCursor, GenerationRef, ResourceRef
from unchain.providers import prepared_turn
from unchain.providers.prepared_request_factory import (
    resolve_prepared_provider_request_payload,
)
from unchain.providers.wire_preparer import prepare_provider_wire
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.tools.handler_registry import DurableToolHandlerRegistry


def _config(*, provider: str, protocol: str, json_schema=None):
    return {
        "schema_version": 1,
        "profile": {
            "id": "profile-a",
            "provider": provider,
            "model": "model-a",
            "protocol": protocol,
            "endpoint": "https://profile.example/v1",
            "credential_ref": "credential-a",
            "context_window": 8192,
            "output_limit": 512,
            "capabilities": {"json_schema": json_schema is not None, "images": False, "files": False},
        },
        "prompt": "reply",
        "max_iterations": 2,
        "output_tokens": 256,
        "history": "root",
        "overflow": "error",
        "json_schema": json_schema,
    }


def _schema():
    return {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }


def _binding(config, *, factories=None, calls=None):
    def profile_resolver(profile_id):
        assert profile_id == "profile-a"
        return copy.deepcopy(config["profile"])

    def credential_resolver(ref):
        if calls is not None:
            calls.append(("credential", ref))
        return "secret-value"

    return bind_kernel_profile(
        config,
        profile_resolver=profile_resolver,
        credential_resolver=credential_resolver,
        client_factories=factories,
    )


def _prepared_turn(binding, payload):
    attempt = AttemptRef(GenerationRef("kernel-profile", "generation"), "attempt")
    draft = prepared_turn._build_provider_turn_draft(
        model_io=binding.model_io,
        registry=DurableToolHandlerRegistry(),
        resolutions=(),
        attempt=attempt,
        iteration=0,
        supports_tools=True,
        request_payload=payload,
        prompt_sha256="0" * 64,
        exposure_plan_sha256="1" * 64,
    )
    catalog_bytes = draft.catalog.canonical_bytes()
    snapshot = ToolCatalogSnapshot(
        envelope=draft.catalog,
        event_cursor=EventCursor(1, "kernel-profile-catalog"),
        artifact=ArtifactRef(
            ref=ResourceRef("artifact", "kernel-profile-catalog", 1),
            media_type="application/json",
            byte_length=len(catalog_bytes),
            sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            preview="",
        ),
    )
    return prepared_turn._issue_prepared_provider_turn(
        draft=draft,
        catalog_authority=prepared_turn._issue_persisted_tool_catalog_authority(snapshot),
    )


@pytest.mark.parametrize(
    ("provider", "protocol", "transport", "payload"),
    [
        ("openai", "openai-responses", "openai", {"max_output_tokens": 256}),
        ("anthropic", "anthropic", "anthropic", {"max_tokens": 256}),
        ("gemini", "gemini", "gemini", {"max_output_tokens": 256}),
        ("ollama", "ollama", "ollama", {"num_predict": 256, "num_ctx": 8192}),
        ("shipped-route", "anthropic", "hyperspace", {"max_tokens": 256}),
    ],
)
def test_binds_fresh_protocol_adapter_with_explicit_catalog(provider, protocol, transport, payload):
    config = _config(provider=provider, protocol=protocol)
    first = _binding(config)
    second = _binding(config)

    assert first.transport_provider == transport
    assert first.model_io is not second.model_io
    assert first.payload_copy() == payload
    assert first.model_io.default_payloads == {"model-a": payload}
    assert first.model_io.model_capabilities["model-a"]["allowed_payload_keys"] == list(payload)
    assert "secret-value" not in repr(first)
    assert "secret-value" not in repr(first.native_payload)


@pytest.mark.parametrize(
    ("provider", "protocol", "expected"),
    [
        ("openai", "openai-responses", {"max_output_tokens": 256, "text": {"format": {"type": "json_schema", "name": "kernel_output", "schema": _schema(), "strict": True}}}),
        ("anthropic", "anthropic", {"max_tokens": 256, "output_config": {"format": {"type": "json_schema", "schema": _schema()}}}),
        ("gemini", "gemini", {"max_output_tokens": 256, "response_mime_type": "application/json", "response_json_schema": _schema()}),
        ("ollama", "ollama", {"num_predict": 256, "num_ctx": 8192}),
    ],
)
def test_native_structured_payload_survives_prepared_request_resolution(provider, protocol, expected):
    binding = _binding(_config(provider=provider, protocol=protocol, json_schema=_schema()))
    prepared = resolve_prepared_provider_request_payload(
        model_io=binding.model_io,
        request=ModelTurnRequest(
            messages=[{"role": "user", "content": "hello"}],
            payload=binding.payload,
            response_format=binding.response_format,
        ),
    )
    assert prepared["effective_payload"] == expected
    if protocol == "ollama":
        assert binding.response_format is not None
        assert prepared["response_format"]["kind"] == "ollama_format"
    else:
        assert binding.response_format is None
        assert prepared["response_format"] == {"kind": "none", "value": None}
    envelope = prepare_provider_wire(
        _prepared_turn(binding, prepared),
        model_io=binding.model_io,
        attempt=AttemptRef(GenerationRef("kernel-profile", "generation"), "attempt"),
        iteration=0,
        transport_target_sha256="a" * 64,
    )
    wire = envelope.request_copy()
    if protocol == "gemini":
        assert wire["config"] == expected | {"automatic_function_calling": {"disable": True}}
        assert "response_schema" not in wire["config"]
    elif protocol == "ollama":
        assert wire["options"] == expected
    else:
        for name, value in expected.items():
            assert wire[name] == value
        if protocol == "anthropic":
            assert "system" not in wire


def test_profile_is_compared_before_credential_resolution():
    config = _config(provider="openai", protocol="openai-responses")
    calls = []

    def profile_resolver(_profile_id):
        wrong = copy.deepcopy(config["profile"])
        wrong["endpoint"] = "https://wrong.example/v1"
        return wrong

    with pytest.raises(ValueError, match="exactly match"):
        bind_kernel_profile(
            config,
            profile_resolver=profile_resolver,
            credential_resolver=lambda ref: calls.append(ref),
        )
    assert calls == []


def test_json_schema_capability_is_rejected_before_any_host_resolution():
    config = _config(provider="openai", protocol="openai-responses", json_schema=_schema())
    config["profile"]["capabilities"]["json_schema"] = False
    calls = []
    with pytest.raises(ValueError):
        bind_kernel_profile(
            config,
            profile_resolver=lambda _profile_id: calls.append("profile"),
            credential_resolver=lambda _credential_ref: calls.append("credential"),
        )
    assert calls == []


def test_bindings_never_share_their_model_catalogs():
    first_config = _config(provider="openai", protocol="openai-responses")
    second_config = _config(provider="openai", protocol="openai-responses")
    second_config["profile"]["model"] = "model-b"
    first = _binding(first_config)
    second = _binding(second_config)
    first.model_io.default_payloads["model-a"]["max_output_tokens"] = 1
    assert second.model_io.default_payloads == {"model-b": {"max_output_tokens": 256}}


def test_profile_digest_uses_the_opaque_credential_reference_not_secret_bytes():
    first_config = _config(provider="openai", protocol="openai-responses")
    second_config = _config(provider="openai", protocol="openai-responses")
    second_config["profile"]["credential_ref"] = "credential-b"
    first = _binding(first_config)
    second = _binding(second_config)
    assert first.profile_digest != second.profile_digest
    assert "secret-value" not in repr(first)


@pytest.mark.parametrize("factories", [{"openai": object()}, {"wrong": lambda: None}])
def test_invalid_client_factory_configuration_is_rejected(factories):
    with pytest.raises((TypeError, ValueError)):
        _binding(
            _config(provider="openai", protocol="openai-responses"),
            factories=factories,
        )


@pytest.mark.parametrize(
    ("provider", "protocol"),
    [("openai", "anthropic"), ("anthropic", "gemini"), ("gemini", "ollama"), ("ollama", "openai-responses")],
)
def test_native_brand_protocol_contradictions_are_rejected(provider, protocol):
    with pytest.raises(ValueError, match="contradicts"):
        _binding(_config(provider=provider, protocol=protocol))


@pytest.mark.parametrize(
    ("provider", "protocol", "transport"),
    [
        ("openai", "openai-responses", "openai"),
        ("anthropic", "anthropic", "anthropic"),
        ("shipped-route", "anthropic", "hyperspace"),
        ("gemini", "gemini", "gemini"),
    ],
)
def test_sdk_factories_pin_endpoint_and_disable_sdk_retries_without_secret_leakage(provider, protocol, transport):
    seen = []

    def factory(**kwargs):
        seen.append(kwargs)
        return object()

    binding = _binding(
        _config(provider=provider, protocol=protocol),
        factories={transport: factory},
    )
    binding.model_io._client_factory(
        api_key="secret-value",
        http_options={"timeout": 12, "retry_options": {"attempts": 99}},
    )
    expected = {"api_key": "secret-value"}
    if transport == "gemini":
        expected["http_options"] = {
            "timeout": 12,
            "base_url": "https://profile.example/v1",
            "retry_options": {"attempts": 1},
        }
    else:
        expected |= {"base_url": "https://profile.example/v1", "max_retries": 0,
                     "http_options": {"timeout": 12, "retry_options": {"attempts": 99}}}
    assert seen == [expected]
    assert "secret-value" not in repr(binding)


def test_ollama_authentication_is_attached_only_to_the_pinned_stream():
    seen = []

    def stream_factory(method, url, **kwargs):
        seen.append((method, url, kwargs))
        return object()

    binding = _binding(
        _config(provider="ollama", protocol="ollama"),
        factories={"ollama": stream_factory},
    )
    binding.model_io._stream_factory("POST", "https://elsewhere.invalid/api/chat", json={})
    assert seen == [
        ("POST", "https://profile.example/v1/api/chat", {"json": {}, "headers": {"Authorization": "Bearer secret-value"}})
    ]


def test_ollama_explicit_credential_free_profile_keeps_the_stream_unauthenticated():
    config = _config(provider="ollama", protocol="ollama")
    calls = []
    binding = bind_kernel_profile(
        config,
        profile_resolver=lambda _profile_id: copy.deepcopy(config["profile"]),
        credential_resolver=lambda ref: calls.append(ref) or None,
        client_factories={"ollama": lambda method, url, **kwargs: (method, url, kwargs)},
    )
    assert calls == ["credential-a"]
    assert binding.model_io._stream_factory("POST", "https://wrong.invalid", json={}) == (
        "POST", "https://profile.example/v1/api/chat", {"json": {}}
    )
    # Authenticated protocols never admit a missing credential.
    with pytest.raises(ValueError):
        bind_kernel_profile(
            _config(provider="openai", protocol="openai-responses"),
            profile_resolver=lambda profile_id: copy.deepcopy(_config(provider="openai", protocol="openai-responses")["profile"]),
            credential_resolver=lambda _ref: None,
        )
    assert "secret-value" not in repr(binding)
