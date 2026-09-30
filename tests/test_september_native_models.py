from __future__ import annotations

from copy import deepcopy

import pytest
from google.genai import types as genai_types
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from pydantic import TypeAdapter

# Import context before prepared_request_factory to complete its package imports.
import unchain.context.tool_catalog
from unchain.providers import GeminiModelIO, ModelTurnRequest, OpenAIModelIO
from unchain.providers.prepared_request_factory import (
    resolve_prepared_provider_request_payload,
)
from unchain.runtime.payloads import load_default_payloads, load_model_capabilities


OPENAI_MODELS = ("gpt-6-sol", "gpt-6-luna")
GEMINI_EFFORTS = {
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
    "gemini-3.5-flash-lite": ("minimal", "low", "medium", "high"),
}


@pytest.mark.parametrize("model", OPENAI_MODELS)
def test_new_openai_models_are_registered_with_responses_defaults_and_limits(model):
    capabilities = load_model_capabilities()[model]
    default = load_default_payloads()[model]

    assert capabilities == {
        "provider": "openai",
        "max_context_window_tokens": 1_050_000,
        "max_output_tokens": 128_000,
        "supports_tools": True,
        "supports_response_format": True,
        "supports_previous_response_id": True,
        "supports_reasoning": True,
        "reasoning_efforts": ["none", "low", "medium", "high", "xhigh", "max"],
        "default_reasoning_effort": "medium",
        "input_modalities": ["text", "image"],
        "input_source_types": {"image": ["url", "base64"]},
        "allowed_payload_keys": [
            "instructions",
            "max_output_tokens",
            "truncation",
            "store",
            "include",
            "reasoning",
            "tool_choice",
        ],
    }
    assert default == {
        "instructions": "",
        "max_output_tokens": 128_000,
        "truncation": "auto",
        "store": False,
        "include": [],
        "reasoning": {"effort": "medium"},
    }


@pytest.mark.parametrize("model", OPENAI_MODELS)
@pytest.mark.parametrize(
    "effort", ["none", "low", "medium", "high", "xhigh", "max"]
)
def test_new_openai_models_prepare_exact_responses_payload_without_sampling(model, effort):
    model_io = OpenAIModelIO(
        model=model,
        api_key="test-key",
        client_factory=lambda **_kwargs: None,
        default_payloads=load_default_payloads(),
        model_capabilities=load_model_capabilities(),
    )
    prepared = resolve_prepared_provider_request_payload(
        model_io=model_io,
        request=ModelTurnRequest(
            messages=[{"role": "user", "content": "hello"}],
            payload={
                "reasoning": {"effort": effort},
                "temperature": 0.1,
                "top_p": 0.2,
            },
        ),
    )

    assert prepared["provider"] == "openai"
    assert prepared["request_model"] == model
    payload = prepared["effective_payload"]
    assert set(payload) == {
        "instructions",
        "max_output_tokens",
        "truncation",
        "store",
        "include",
        "reasoning",
    }
    assert payload == {
        "instructions": "",
        "max_output_tokens": 128_000,
        "truncation": "auto",
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "reasoning": {"effort": effort},
    }
    assert "temperature" not in payload
    assert "top_p" not in payload
    sdk_wire = {
        "model": prepared["request_model"],
        "input": prepared["messages"],
        **deepcopy(payload),
        "stream": True,
    }
    assert set(sdk_wire) == {
        "model",
        "input",
        "instructions",
        "max_output_tokens",
        "truncation",
        "store",
        "include",
        "reasoning",
        "stream",
    }
    if effort == "max":
        # This environment's OpenAI SDK Literal lags the official model docs.
        assert sdk_wire["reasoning"]["effort"] == "max"
    else:
        TypeAdapter(ResponseCreateParamsStreaming).validate_python(sdk_wire)


@pytest.mark.parametrize("model,efforts", GEMINI_EFFORTS.items())
def test_new_gemini_models_prepare_sdk_valid_configs_without_sampling(model, efforts):
    capabilities = load_model_capabilities()[model]
    default = load_default_payloads()[model]
    assert capabilities["provider"] == "gemini"
    assert capabilities["max_context_window_tokens"] == 1_048_576
    assert capabilities["max_output_tokens"] == 65_536
    assert capabilities["supports_tools"] is True
    assert capabilities["supports_response_format"] is True
    assert capabilities["supports_previous_response_id"] is False
    assert capabilities["supports_reasoning"] is True
    assert capabilities["reasoning_efforts"] == list(efforts)
    assert capabilities["default_reasoning_effort"] == (
        "minimal" if "minimal" in efforts else "medium"
    )
    assert capabilities["input_modalities"] == ["text", "image", "pdf"]
    assert capabilities["allowed_payload_keys"] == [
        "max_output_tokens",
        "thinking_config",
    ]
    assert set(default) == {"max_output_tokens", "thinking_config"}
    assert default["max_output_tokens"] == 65_536
    sdk_default = genai_types.GenerateContentConfig.model_validate(deepcopy(default))
    assert (
        sdk_default.thinking_config.thinking_level.name.lower()
        == capabilities["default_reasoning_effort"]
    )

    for effort in efforts:
        model_io = GeminiModelIO(
            model=model,
            api_key="test-key",
            client_factory=lambda **_kwargs: None,
            default_payloads=load_default_payloads(),
            model_capabilities=load_model_capabilities(),
        )
        prepared = resolve_prepared_provider_request_payload(
            model_io=model_io,
            request=ModelTurnRequest(
                messages=[{"role": "user", "content": "hello"}],
                payload={
                    "thinking_config": {
                        "thinking_level": effort,
                        "include_thoughts": True,
                    },
                    "temperature": 0.1,
                    "top_p": 0.2,
                    "top_k": 3,
                },
            ),
        )

        assert prepared["provider"] == "gemini"
        assert prepared["request_model"] == model
        payload = prepared["effective_payload"]
        assert set(payload) == {"max_output_tokens", "thinking_config"}
        assert payload == {
            "max_output_tokens": 65_536,
            "thinking_config": {
                "thinking_level": effort,
                "include_thoughts": True,
            },
        }
        # The SDK validates request field types; the model allowlist keeps sampling keys out.
        sdk_config = genai_types.GenerateContentConfig.model_validate(deepcopy(payload))
        assert sdk_config.thinking_config.thinking_level.name.lower() == effort
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload
