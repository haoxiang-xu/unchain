"""PuPu #411: the synchronous native Argon contract, with no live requests."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
from google.genai import types

import unchain.context.tool_catalog
from unchain.providers import GeminiModelIO, ModelTurnRequest
from unchain.providers.prepared_request_factory import resolve_prepared_provider_request_payload
from unchain.providers.wire_envelope import ProviderWireEnvelope
from unchain.providers.wire_preparer import prepare_provider_wire
from unchain.runtime.payloads import load_default_payloads, load_model_capabilities
from unchain.tools import Toolkit

MODEL = "gemini-4-argon"


def test_argon_catalog_has_exact_synchronous_capabilities_and_defaults():
    assert load_model_capabilities()[MODEL] == {
        "provider": "gemini",
        "max_context_window_tokens": 1_048_576,
        "max_output_tokens": 65_536,
        "supports_tools": True,
        "supports_response_format": True,
        "supports_previous_response_id": False,
        "supports_reasoning": True,
        "reasoning_efforts": ["low", "medium", "high"],
        "default_reasoning_effort": "medium",
        "input_modalities": ["text", "image", "pdf"],
        "input_source_types": {"image": ["url", "base64"], "pdf": ["url", "base64"]},
        "allowed_payload_keys": ["max_output_tokens", "thinking_config"],
    }
    assert load_default_payloads()[MODEL] == {
        "max_output_tokens": 65_536,
        "thinking_config": {"thinking_level": "medium", "include_thoughts": True},
    }


def _io(captured, chunks=None):
    def factory(**kwargs):
        assert kwargs == {
            "api_key": "test-key",
            "http_options": {"timeout": 900_000, "retry_options": {"attempts": 1}},
        }

        def send(**wire):
            assert set(wire) == {"model", "contents", "config"}
            assert wire["model"] == MODEL
            assert isinstance(wire["config"], types.GenerateContentConfig)
            config = wire["config"].model_dump(mode="json", exclude_none=True, exclude_unset=True)
            assert set(config) == {
                "max_output_tokens", "thinking_config", "automatic_function_calling", "tools",
            }
            assert config["automatic_function_calling"] == {"disable": True}
            assert set(config["thinking_config"]) == {"thinking_level", "include_thoughts"}
            assert config["max_output_tokens"] == 65_536
            assert config["tools"][0]["function_declarations"][0]["name"] == "search"
            captured.append(copy.deepcopy(wire))
            return iter(types.GenerateContentResponse.model_validate(c) for c in (
                chunks or [{"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finish_reason": "STOP"}]}]
            ))

        return SimpleNamespace(models=SimpleNamespace(generate_content_stream=send), close=lambda: None)

    return GeminiModelIO(model=MODEL, api_key="test-key", client_factory=factory)


def _toolkit():
    toolkit = Toolkit()

    def search(query: str):
        return query

    toolkit.register(search, name="search")
    return toolkit


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_argon_direct_stream_projects_only_compatible_settings(level):
    captured, events = [], []
    io = _io(captured)
    result = io.fetch_turn(ModelTurnRequest(
        messages=[{"role": "user", "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1n"}},
        ]}], toolkit=_toolkit(), emit_stream=True, callback=events.append,
        payload={
            "thinking_config": {"thinking_level": level, "include_thoughts": True},
            "temperature": 0.2, "top_p": 0.3, "top_k": 7,
            "cached_content": "caches/forbidden", "background": True, "unknown": True,
            "automatic_continuation": True, "continuation_token": "forbidden",
        },
    ))
    assert result.final_text == "ok"
    assert [event["delta"] for event in events if event["type"] == "token_delta"] == ["ok"]
    assert len(captured) == 1
    assert captured[0]["config"].thinking_config.thinking_level.value.lower() == level
    assert captured[0]["contents"][0].parts[1].inline_data.data == b"img"


def _envelope(io, payload):
    from tests.test_provider_wire_preparer import ATTEMPT, _prepared_turn_for_request

    request = ModelTurnRequest(messages=[{"role": "user", "content": "hello"}], payload=payload)
    frozen = resolve_prepared_provider_request_payload(model_io=io, request=request)
    prepared, draft = _prepared_turn_for_request(model_io=io, request_payload=frozen)
    envelope = prepare_provider_wire(
        prepared, model_io=io, attempt=ATTEMPT, iteration=7, transport_target_sha256="9" * 64,
    )
    return envelope, draft


def test_argon_durable_wire_has_exact_projection_and_replay_identity():
    io = GeminiModelIO(model=MODEL, api_key="test-key")
    envelope, draft = _envelope(io, {
        "temperature": 0.2, "cached_content": "caches/no", "unknown": True,
        "automatic_continuation": True, "max_output_tokens": 65_536,
    })
    wire = envelope.request_copy()
    assert set(wire) == {"model", "contents", "config", "tools"}
    assert wire["model"] == MODEL
    assert wire["config"] == {
        "max_output_tokens": 65_536,
        "thinking_config": {"thinking_level": "medium", "include_thoughts": True},
        "automatic_function_calling": {"disable": True},
    }
    assert wire["tools"][0]["name"] == "search"
    assert ProviderWireEnvelope.from_dict(envelope.to_dict()) == envelope
    assert envelope.verify_against_catalog(draft.catalog) is envelope
    assert envelope.request_copy() == wire
    wire["config"]["max_output_tokens"] = 1
    assert envelope.request_copy()["config"]["max_output_tokens"] == 65_536


@pytest.mark.parametrize("payload", [
    {"max_output_tokens": 65_537}, {"max_output_tokens": 1_048_576},
    {"max_output_tokens": 0}, {"max_output_tokens": -1}, {"max_output_tokens": True},
    {"max_output_tokens": 1.5}, {"max_output_tokens": "65536"}, {"max_output_tokens": None},
    {"thinking_config": {"thinking_level": "minimal"}},
    {"thinking_config": {"thinking_level": "max"}},
    {"thinking_config": {"thinking_budget": 1024}},
    {"thinking_config": {"thinking_level": "low", "unknown": True}},
    {"thinking_config": {"include_thoughts": "yes"}},
    {"thinking_config": None},
])
@pytest.mark.parametrize("path", ["direct", "durable"])
def test_argon_rejects_unsupported_limits_and_thinking_before_send(payload, path):
    clients = []

    def forbidden(**kwargs):
        clients.append(kwargs)
        pytest.fail("invalid Argon config must not create a client")

    io = GeminiModelIO(model=MODEL, api_key="test-key", client_factory=forbidden)
    with pytest.raises(ValueError, match="Argon"):
        if path == "direct":
            io.fetch_turn(ModelTurnRequest(messages=[{"role": "user", "content": "hello"}], payload=payload))
        else:
            _envelope(io, payload)
    assert clients == []


@pytest.mark.parametrize("extra", [
    {"temperature": 1}, {"cached_content": "caches/no"},
    {"automatic_continuation": True}, {"unknown": True},
])
def test_argon_forged_final_config_is_rejected_before_sdk_client(extra):
    io = GeminiModelIO(model=MODEL, api_key="test-key", client_factory=lambda **_: pytest.fail("must not send"))
    with pytest.raises(ValueError, match="Argon"):
        io._fetch_prepared(ModelTurnRequest(messages=[]), {
            "model": MODEL, "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "config": {**load_default_payloads().get(MODEL, {}), **extra},
        })


def test_argon_two_signed_function_cycles_then_second_normal_turn():
    captured = []
    messages = [{"role": "user", "content": "search twice"}]
    for index in (1, 2):
        io = _io(captured, [{"candidates": [{
            "content": {"role": "model", "parts": [{
                "thought_signature": "c2ln", "function_call": {"id": f"call-{index}", "name": "search", "args": {"query": "x"}},
            }]}, "finish_reason": "STOP",
        }]}])
        result = io.fetch_turn(ModelTurnRequest(messages=messages, toolkit=_toolkit()))
        assert result.tool_calls[0].call_id == f"call-{index}"
        messages = result.provider_replay_frame["items"] + [{"role": "user", "parts": [{
            "function_response": {"id": f"call-{index}", "name": "search", "response": {"result": "found"}},
        }]}]
    _io(captured).fetch_turn(ModelTurnRequest(messages=messages, toolkit=_toolkit()))
    messages += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "thanks"}]
    _io(captured).fetch_turn(ModelTurnRequest(messages=messages, toolkit=_toolkit()))
    assert len(captured) == 4
    parts = [part for content in captured[-1]["contents"] for part in content.parts]
    assert [part.function_response.id for part in parts if part.function_response] == ["call-1", "call-2"]
    assert [part.thought_signature for part in parts if part.function_call] == [b"sig", b"sig"]


def test_existing_flash_configuration_is_unchanged():
    io = GeminiModelIO(model="gemini-3.6-flash", api_key="test-key")
    assert io._merged_payload({"temperature": 0.2})["temperature"] == 0.2
    assert load_default_payloads()["gemini-3.6-flash"]["thinking_config"]["thinking_level"] == "medium"
