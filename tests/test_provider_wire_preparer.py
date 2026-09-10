from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from unchain.kernel import ModelTurnRequest
from unchain.context.tool_catalog import ToolCatalogSnapshot
from unchain.journal.models import (
    ArtifactRef,
    AttemptRef,
    EventCursor,
    GenerationRef,
    ResourceRef,
)
from unchain.providers import (
    AnthropicModelIO,
    HyperspaceModelIO,
    OllamaModelIO,
    OpenAIModelIO,
)
from unchain.providers import prepared_turn, wire_preparer as wire_preparer_module
from unchain.providers.prepared_request_factory import resolve_prepared_provider_request_payload
from unchain.providers.wire_preparer import (
    build_prepared_provider_request_payload,
    prepare_provider_wire as prepare_bound_provider_wire,
)
from unchain.providers.wire_envelope import ProviderWireEnvelope
from unchain.tools.handler_registry import (
    DurableToolHandlerBinding,
    DurableToolHandlerRegistry,
    tool_config_sha256,
)
from unchain.tools.tool import Tool


ATTEMPT = AttemptRef(
    GenerationRef("execution-wire-preparer", "generation-wire-preparer"),
    "attempt-wire-preparer",
)


def test_uncatalogued_ollama_window_survives_durable_wire_preparation():
    model_io = OllamaModelIO(model="unknown-local:latest", default_payloads={}, model_capabilities={})
    request = ModelTurnRequest(messages=[{"role": "user", "content": "Hello"}], payload={"num_ctx": 32768, "unknown_option": True})
    payload = resolve_prepared_provider_request_payload(model_io=model_io, request=request)
    assert payload["effective_payload"] == {"num_ctx": 32768}
    prepared, draft = _prepared_turn_for_request(model_io=model_io, request_payload=payload)
    envelope = prepare_bound_provider_wire(prepared, model_io=model_io, attempt=ATTEMPT, iteration=7, transport_target_sha256="9" * 64)
    wire = envelope.request_copy()
    assert set(wire) == {"model", "messages", "stream", "options", "tools", "tool_choice"}
    assert wire["tools"] == draft.toolkit.to_provider_json("ollama")
    assert wire["tool_choice"] == "auto"
    assert wire["options"] == {"num_ctx": 32768}
ProviderWirePreparationInput = wire_preparer_module._ProviderWirePreparationInput
prepare_provider_wire = wire_preparer_module._prepare_provider_wire_from_input


class _ModelIO:
    def __init__(self, provider: str, model: str = "frontier-model") -> None:
        self.provider = provider
        self.model = model


class _OpenAIStream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="response-current",
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
                usage={"input_tokens": 1, "output_tokens": 1},
            ),
        )


def _openai_capture_factory(captured_requests):
    class _Responses:
        def create(self, **kwargs):
            captured_requests.append(copy.deepcopy(kwargs))
            if kwargs.get("previous_response_id"):
                raise ValueError("previous_response not_found")
            return _OpenAIStream()

    class _Client:
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = _Responses()

    return _Client


class _AnthropicStream:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        return iter(())


def _anthropic_capture_factory(captured_requests):
    class _Messages:
        def stream(self, **kwargs):
            captured_requests.append(copy.deepcopy(kwargs))
            return _AnthropicStream()

    class _Client:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.messages = _Messages()

    return _Client


class _OllamaResponse:
    status_code = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        return iter([json.dumps({"message": {"content": "done"}, "done": True})])


class _ResponseFormat:
    def to_anthropic(self):
        return "Return strict JSON."

    def to_ollama(self):
        return {"type": "object"}


def _search(query: str = "") -> dict[str, str]:
    return {"query": query}


def _toolkit_and_catalog(
    provider: str,
    *,
    required_betas: list[str] | None = None,
):
    registry = DurableToolHandlerRegistry()
    tool = Tool(
        name="search",
        description="Search",
        func=_search,
        required_betas=(
            {provider: required_betas} if required_betas is not None else None
        ),
    )
    resolution = registry.register(
        DurableToolHandlerBinding(
            handler_id="host.search",
            revision=1,
            config_sha256=tool_config_sha256(tool),
            kind="stable",
        ),
        tool=tool,
        handler=_search,
    )
    draft = prepared_turn._build_provider_turn_draft(
        model_io=_ModelIO(provider),
        registry=registry,
        resolutions=(resolution,),
        attempt=ATTEMPT,
        iteration=7,
        supports_tools=True,
        request_payload={"messages": [{"role": "user", "content": "hello"}]},
        prompt_sha256="0" * 64,
        exposure_plan_sha256="1" * 64,
    )
    return draft.toolkit, draft.catalog


def _prepared_turn_for_request(
    *,
    model_io,
    request_payload,
    required_betas: list[str] | None = None,
):
    registry = DurableToolHandlerRegistry()
    tool = Tool(
        name="search",
        description="Search",
        func=_search,
        required_betas=(
            {model_io.provider: required_betas} if required_betas is not None else None
        ),
    )
    resolution = registry.register(
        DurableToolHandlerBinding(
            handler_id="host.search",
            revision=1,
            config_sha256=tool_config_sha256(tool),
            kind="stable",
        ),
        tool=tool,
        handler=_search,
    )
    draft = prepared_turn._build_provider_turn_draft(
        model_io=model_io,
        registry=registry,
        resolutions=(resolution,),
        attempt=ATTEMPT,
        iteration=7,
        supports_tools=True,
        request_payload=request_payload,
        prompt_sha256="0" * 64,
        exposure_plan_sha256="1" * 64,
    )
    catalog_bytes = draft.catalog.canonical_bytes()
    snapshot = ToolCatalogSnapshot(
        envelope=draft.catalog,
        event_cursor=EventCursor(1, "event-wire-preparer-catalog"),
        artifact=ArtifactRef(
            ref=ResourceRef("artifact", "wire-preparer-catalog", 1),
            media_type="application/json",
            byte_length=len(catalog_bytes),
            sha256=hashlib.sha256(catalog_bytes).hexdigest(),
            preview="",
        ),
    )
    authority = prepared_turn._issue_persisted_tool_catalog_authority(snapshot)
    prepared = prepared_turn._issue_prepared_provider_turn(
        draft=draft,
        catalog_authority=authority,
    )
    return prepared, draft


def test_provider_wire_preparer_public_contract_exists() -> None:
    assert callable(build_prepared_provider_request_payload)
    assert callable(prepare_bound_provider_wire)
    assert "_ProviderWirePreparationInput" not in wire_preparer_module.__all__


def test_prepared_provider_request_payload_has_one_exact_canonical_shape() -> None:
    payload = build_prepared_provider_request_payload(
        provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        effective_payload={"temperature": 0.2},
        request_model="frontier-model",
        response_format={"kind": "none", "value": None},
        previous_response_id=None,
        fallback_messages=None,
        context_mode="semantic",
    )

    assert payload == {
        "schema": "unchain.prepared_provider_request.v1",
        "provider": "openai",
        "messages": [{"role": "user", "content": "hello"}],
        "effective_payload": {"temperature": 0.2},
        "request_model": "frontier-model",
        "response_format": {"kind": "none", "value": None},
        "previous_response_id": None,
        "fallback_messages": None,
        "context_mode": "semantic",
    }


def test_remote_continuation_rejects_empty_fallback_before_preparation() -> None:
    with pytest.raises(ValueError, match="non-empty fallback"):
        build_prepared_provider_request_payload(
            provider="openai",
            messages=[{"role": "user", "content": "current delta"}],
            effective_payload={},
            request_model="frontier-model",
            response_format={"kind": "none", "value": None},
            previous_response_id="response-previous",
            fallback_messages=[],
            context_mode="remote_continuation",
        )


def test_public_preparer_consumes_one_exact_prepared_turn() -> None:
    model_io = _ModelIO("openai")
    request_payload = build_prepared_provider_request_payload(
        provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        effective_payload={"temperature": 0.2},
        request_model="frontier-model",
        response_format={"kind": "none", "value": None},
        previous_response_id=None,
        fallback_messages=None,
        context_mode="semantic",
    )
    prepared, draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
    )

    envelope = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="b" * 64,
    )

    assert envelope.source_request_sha256 == draft._request_payload_sha256
    assert envelope.catalog_sha256 == draft.catalog.catalog_sha256
    assert envelope.request_copy() == {
        "model": "frontier-model",
        "input": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "stream": True,
        "tools": draft.toolkit.to_provider_json("openai"),
    }


def test_public_preparer_rejects_another_model_io_consumer() -> None:
    model_io = _ModelIO("openai")
    request_payload = build_prepared_provider_request_payload(
        provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        effective_payload={},
        request_model="frontier-model",
        response_format={"kind": "none", "value": None},
        previous_response_id=None,
        fallback_messages=None,
        context_mode="semantic",
    )
    prepared, _draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
    )

    with pytest.raises(ValueError, match="another model_io"):
        prepare_bound_provider_wire(
            prepared,
            model_io=_ModelIO("openai"),
            attempt=ATTEMPT,
            iteration=7,
            transport_target_sha256="c" * 64,
        )


def test_openai_prepares_primary_and_exact_previous_response_fallback() -> None:
    toolkit, catalog = _toolkit_and_catalog("openai")
    messages = [{"role": "user", "content": "current delta"}]
    fallback = [{"role": "user", "content": "complete replay"}]
    payload = {"store": False, "temperature": 0.2}
    original = copy.deepcopy((messages, fallback, payload))
    preparation = ProviderWirePreparationInput(
        attempt=ATTEMPT,
        iteration=7,
        catalog=catalog,
        provider="openai",
        configured_model="frontier-model",
        transport_target_sha256="2" * 64,
        messages=messages,
        payload=payload,
        toolkit=toolkit,
        default_payloads={"frontier-model": {"store": True, "temperature": 1.0}},
        model_capabilities={
            "frontier-model": {
                "supports_reasoning": True,
                "supports_tools": True,
                "allowed_payload_keys": ["store", "temperature", "include"],
            }
        },
        openai_text_format={"type": "json_object"},
        previous_response_id="response-previous",
        fallback_messages=fallback,
    )

    envelope = prepare_provider_wire(preparation)

    primary = envelope.request_copy("primary")
    replay = envelope.request_copy("openai_previous_response_fallback")
    assert envelope.provider == "openai"
    assert envelope.request_model == "frontier-model"
    assert primary["input"] == messages
    assert primary["previous_response_id"] == "response-previous"
    assert primary["store"] is False
    assert primary["temperature"] == 0.2
    assert primary["include"] == ["reasoning.encrypted_content"]
    assert primary["text"] == {"format": {"type": "json_object"}}
    assert primary["tools"] == toolkit.to_provider_json("openai")
    assert "previous_response_id" not in replay
    assert replay["input"] == fallback
    assert {
        key: value
        for key, value in primary.items()
        if key not in {"input", "previous_response_id"}
    } == {key: value for key, value in replay.items() if key != "input"}
    assert (messages, fallback, payload) == original
    assert envelope.verify_against_catalog(catalog) is envelope


def test_anthropic_prepares_system_cache_control_tools_and_beta_header() -> None:
    toolkit, catalog = _toolkit_and_catalog(
        "anthropic",
        required_betas=["computer-use-2025-11-24"],
    )
    messages = [
        {"role": "system", "content": "system policy"},
        {"role": "developer", "content": "developer policy"},
        {"role": "user", "content": "hello"},
    ]
    payload = {
        "max_tokens": 2048,
        "temperature": 0.3,
        "extra_headers": {"anthropic-beta": "existing-beta"},
    }
    original = copy.deepcopy((messages, payload))

    envelope = prepare_provider_wire(
        ProviderWirePreparationInput(
            attempt=ATTEMPT,
            iteration=7,
            catalog=catalog,
            provider="anthropic",
            configured_model="frontier-model",
            transport_target_sha256="3" * 64,
            messages=messages,
            payload=payload,
            toolkit=toolkit,
            default_payloads={
                "frontier-model": {
                    "max_tokens": 4096,
                    "temperature": 1.0,
                    "extra_headers": {},
                }
            },
            model_capabilities={
                "frontier-model": {
                    "supports_tools": True,
                    "max_output_tokens": 8192,
                }
            },
            anthropic_response_instruction="Respond as JSON.",
        )
    )

    request = envelope.request_copy()
    assert envelope.adapter_revision == "unchain.anthropic.messages.request.v1"
    assert envelope.transport_kind == "anthropic.messages.stream"
    assert envelope.base_anthropic_betas == ("existing-beta",)
    assert envelope.required_betas == ("computer-use-2025-11-24",)
    assert request["model"] == "frontier-model"
    assert request["max_tokens"] == 2048
    assert request["temperature"] == 0.3
    assert request["extra_headers"] == {
        "anthropic-beta": "existing-beta,computer-use-2025-11-24"
    }
    assert request["system"] == [
        {
            "type": "text",
            "text": "system policy\n\ndeveloper policy\n\nRespond as JSON.",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert request["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        }
    ]
    assert request["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert (messages, payload) == original
    assert envelope.verify_against_catalog(catalog) is envelope


def test_hyperspace_uses_anthropic_body_with_provider_model_alias() -> None:
    toolkit, catalog = _toolkit_and_catalog("hyperspace")

    envelope = prepare_provider_wire(
        ProviderWirePreparationInput(
            attempt=ATTEMPT,
            iteration=7,
            catalog=catalog,
            provider="hyperspace",
            configured_model="frontier-model",
            transport_target_sha256="4" * 64,
            messages=[{"role": "user", "content": "hello"}],
            payload={"max_tokens": 1024},
            toolkit=toolkit,
            default_payloads={"frontier-model": {"max_tokens": 4096}},
            model_capabilities={
                "frontier-model": {
                    "provider_model": "anthropic/claude-frontier",
                    "supports_tools": True,
                }
            },
        )
    )

    assert envelope.provider == "hyperspace"
    assert envelope.configured_model == "frontier-model"
    assert envelope.request_model == "anthropic/claude-frontier"
    assert envelope.adapter_revision == (
        "unchain.hyperspace.anthropic-messages.request.v1"
    )
    assert envelope.transport_kind == "hyperspace.anthropic.messages.stream"
    assert envelope.transport_target_sha256 == "4" * 64
    assert envelope.request_copy()["model"] == "anthropic/claude-frontier"
    assert envelope.request_copy()["max_tokens"] == 1024
    assert envelope.verify_against_catalog(catalog) is envelope


def test_ollama_prepares_options_format_tools_and_auto_tool_choice() -> None:
    toolkit, catalog = _toolkit_and_catalog("ollama")
    messages = [{"role": "user", "content": "hello"}]
    payload = {"temperature": 0.4, "num_ctx": 8192}
    original = copy.deepcopy((messages, payload))

    envelope = prepare_provider_wire(
        ProviderWirePreparationInput(
            attempt=ATTEMPT,
            iteration=7,
            catalog=catalog,
            provider="ollama",
            configured_model="frontier-model",
            transport_target_sha256="5" * 64,
            messages=messages,
            payload=payload,
            toolkit=toolkit,
            default_payloads={"frontier-model": {"temperature": 0.8, "num_ctx": 4096}},
            model_capabilities={
                "frontier-model": {
                    "supports_tools": True,
                    "allowed_payload_keys": ["temperature", "num_ctx"],
                }
            },
            ollama_format={"type": "object"},
        )
    )

    request = envelope.request_copy()
    assert envelope.adapter_revision == "unchain.ollama.chat.request.v1"
    assert envelope.transport_kind == "ollama.api.chat.post"
    assert request == {
        "model": "frontier-model",
        "messages": messages,
        "stream": True,
        "tools": toolkit.to_provider_json("ollama"),
        "tool_choice": "auto",
        "options": {"temperature": 0.4, "num_ctx": 8192},
        "format": {"type": "object"},
    }
    assert (messages, payload) == original
    assert envelope.verify_against_catalog(catalog) is envelope


def test_provider_wire_preparation_is_deterministic_and_round_trips() -> None:
    model_io = _ModelIO("openai")
    request_payload = build_prepared_provider_request_payload(
        provider="openai",
        messages=[{"role": "user", "content": "hello"}],
        effective_payload={"temperature": 0.2},
        request_model="frontier-model",
        response_format={"kind": "none", "value": None},
        previous_response_id=None,
        fallback_messages=None,
        context_mode="semantic",
    )
    prepared, _draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
    )

    first = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="6" * 64,
    )
    second = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="6" * 64,
    )
    recovered = ProviderWireEnvelope.from_dict(first.to_dict())

    assert second == first
    assert second.canonical_bytes() == first.canonical_bytes()
    assert second.envelope_sha256 == first.envelope_sha256
    assert recovered == first
    assert recovered.canonical_bytes() == first.canonical_bytes()


def test_openai_prepared_routes_match_native_adapter_requests() -> None:
    messages = [{"role": "user", "content": "current delta"}]
    fallback = [{"role": "user", "content": "complete replay"}]
    payload = {"store": False, "temperature": 0.2}
    defaults = {"frontier-model": {"store": True, "temperature": 1.0}}
    capabilities = {
        "frontier-model": {
            "supports_reasoning": True,
            "supports_tools": True,
            "allowed_payload_keys": ["store", "temperature", "include"],
        }
    }
    captured = []
    model_io = OpenAIModelIO(
        model="frontier-model",
        api_key="test-key",
        client_factory=_openai_capture_factory(captured),
        default_payloads=defaults,
        model_capabilities=capabilities,
    )
    request_payload = build_prepared_provider_request_payload(
        provider="openai",
        messages=messages,
        effective_payload={
            "store": False,
            "temperature": 0.2,
            "include": ["reasoning.encrypted_content"],
        },
        request_model="frontier-model",
        response_format={
            "kind": "openai_text",
            "value": {"type": "json_object"},
        },
        previous_response_id="response-previous",
        fallback_messages=fallback,
        context_mode="remote_continuation",
    )
    prepared, draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
    )
    envelope = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="7" * 64,
    )

    model_io.fetch_turn(
        ModelTurnRequest(
            messages=messages,
            payload=payload,
            toolkit=draft.toolkit,
            previous_response_id="response-previous",
            openai_text_format={"type": "json_object"},
            fallback_messages=fallback,
        )
    )

    assert captured == [
        envelope.request_copy("primary"),
        envelope.request_copy("openai_previous_response_fallback"),
    ]


def test_openai_computer_call_preparation_matches_native_adapter_request() -> None:
    messages = [
        {
            "type": "computer_call",
            "id": "computer-item-1",
            "actions": [
                {
                    "type": "click",
                    "x": 10,
                    "y": 20,
                    "button": None,
                }
            ],
            "pending_safety_checks": [],
            "status": "completed",
            "provider_only_noise": "remove-me",
        }
    ]
    captured = []
    model_io = OpenAIModelIO(
        model="frontier-model",
        api_key="test-key",
        client_factory=_openai_capture_factory(captured),
        default_payloads={},
        model_capabilities={},
    )
    request_payload = build_prepared_provider_request_payload(
        provider="openai",
        messages=messages,
        effective_payload={},
        request_model="frontier-model",
        response_format={"kind": "none", "value": None},
        previous_response_id=None,
        fallback_messages=None,
        context_mode="local_replay",
    )
    prepared, draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
    )
    envelope = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="a" * 64,
    )

    model_io.fetch_turn(ModelTurnRequest(messages=messages, toolkit=draft.toolkit))

    assert captured == [envelope.request_copy()]


@pytest.mark.parametrize(
    ("provider", "adapter_class"),
    [
        ("anthropic", AnthropicModelIO),
        ("hyperspace", HyperspaceModelIO),
    ],
)
def test_anthropic_family_prepared_route_matches_native_adapter_request(
    provider,
    adapter_class,
) -> None:
    messages = [
        {"role": "system", "content": "System rule."},
        {"role": "user", "content": "hello"},
    ]
    payload = {
        "max_tokens": 1024,
        "extra_headers": {"anthropic-beta": "base-beta"},
    }
    defaults = {
        "frontier-model": {
            "max_tokens": 4096,
            "extra_headers": {},
        }
    }
    capabilities = {
        "frontier-model": {
            "provider_model": "anthropic/claude-frontier",
            "supports_tools": True,
            "allowed_payload_keys": ["max_tokens", "extra_headers"],
        }
    }
    captured = []
    model_io = adapter_class(
        model="frontier-model",
        api_key="test-key",
        client_factory=_anthropic_capture_factory(captured),
        default_payloads=defaults,
        model_capabilities=capabilities,
    )
    request_payload = build_prepared_provider_request_payload(
        provider=provider,
        messages=messages,
        effective_payload=payload,
        request_model="anthropic/claude-frontier",
        response_format={
            "kind": "anthropic_instruction",
            "value": "Return strict JSON.",
        },
        previous_response_id=None,
        fallback_messages=None,
        context_mode="semantic",
    )
    prepared, draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
        required_betas=["computer-use-2025-01-24"],
    )
    envelope = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="8" * 64,
    )

    model_io.fetch_turn(
        ModelTurnRequest(
            messages=messages,
            payload=payload,
            response_format=_ResponseFormat(),
            toolkit=draft.toolkit,
        )
    )

    assert captured == [envelope.request_copy()]


def test_ollama_prepared_route_matches_native_adapter_request() -> None:
    messages = [{"role": "user", "content": "hello"}]
    payload = {"temperature": 0.4, "num_ctx": 8192}
    defaults = {"frontier-model": {"temperature": 0.8, "num_ctx": 4096}}
    capabilities = {
        "frontier-model": {
            "supports_tools": True,
            "allowed_payload_keys": ["temperature", "num_ctx"],
        }
    }
    captured = []

    def stream_factory(method, url, **kwargs):
        captured.append({"method": method, "url": url, **copy.deepcopy(kwargs)})
        return _OllamaResponse()

    model_io = OllamaModelIO(
        model="frontier-model",
        stream_factory=stream_factory,
        default_payloads=defaults,
        model_capabilities=capabilities,
    )
    request_payload = build_prepared_provider_request_payload(
        provider="ollama",
        messages=messages,
        effective_payload=payload,
        request_model="frontier-model",
        response_format={"kind": "ollama_format", "value": {"type": "object"}},
        previous_response_id=None,
        fallback_messages=None,
        context_mode="semantic",
    )
    prepared, draft = _prepared_turn_for_request(
        model_io=model_io,
        request_payload=request_payload,
    )
    envelope = prepare_bound_provider_wire(
        prepared,
        model_io=model_io,
        attempt=ATTEMPT,
        iteration=7,
        transport_target_sha256="9" * 64,
    )
    model_io.fetch_turn(
        ModelTurnRequest(
            messages=messages,
            payload=payload,
            response_format=_ResponseFormat(),
            toolkit=draft.toolkit,
        )
    )

    assert len(captured) == 1
    assert captured[0]["method"] == "POST"
    assert captured[0]["url"].endswith("/api/chat")
    assert captured[0]["timeout"] is None
    assert captured[0]["json"] == envelope.request_copy()
