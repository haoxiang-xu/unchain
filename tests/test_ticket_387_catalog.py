from __future__ import annotations

import pytest

from unchain.providers import AnthropicModelIO, ModelTurnRequest, OpenAIModelIO
from unchain.providers.prepared_request_factory import (
    resolve_prepared_provider_request_payload,
)
from unchain.runtime.payloads import load_default_payloads, load_model_capabilities


OPENAI_MODELS = {
    "gpt-6.1-sol": ("openai", ["low", "medium", "high", "xhigh", "max"], "medium"),
    "gpt-6.1-terra": (
        "openai",
        ["none", "low", "medium", "high", "xhigh", "max"],
        "medium",
    ),
}
CLAUDE_MODELS = {
    "claude-sonnet-5-5": ("high"),
    "claude-haiku-5-5": ("medium"),
    "claude-fable-5-2": ("high"),
}
CLAUDE_EFFORTS = ["low", "medium", "high", "xhigh", "max"]


def _expected_capabilities(provider: str, efforts: list[str], default: str) -> dict:
    if provider == "openai":
        return {
            "provider": "openai",
            "max_context_window_tokens": 1_050_000,
            "max_output_tokens": 128_000,
            "supports_tools": True,
            "supports_response_format": True,
            "supports_previous_response_id": True,
            "supports_reasoning": True,
            "reasoning_efforts": efforts,
            "default_reasoning_effort": default,
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
    return {
        "provider": "anthropic",
        "max_context_window_tokens": 1_000_000,
        "max_output_tokens": 128_000,
        "supports_tools": True,
        "supports_response_format": False,
        "supports_previous_response_id": False,
        "supports_reasoning": True,
        "reasoning_efforts": efforts,
        "default_reasoning_effort": default,
        "input_modalities": ["text", "image", "pdf"],
        "input_source_types": {
            "image": ["url", "base64"],
            "pdf": ["url", "base64"],
        },
        "allowed_payload_keys": ["max_tokens", "output_config"],
    }


@pytest.mark.parametrize(
    "model,provider_efforts_default",
    [*OPENAI_MODELS.items(), *((model, ("anthropic", CLAUDE_EFFORTS, effort))
                              for model, effort in CLAUDE_MODELS.items())],
)
def test_ticket_387_models_have_exact_capability_and_default_records(
    model, provider_efforts_default
):
    provider, efforts, default_effort = provider_efforts_default
    capabilities = load_model_capabilities()[model]
    defaults = load_default_payloads()[model]

    assert capabilities == _expected_capabilities(provider, efforts, default_effort)
    if provider == "openai":
        expected_defaults = {
            "instructions": "",
            "max_output_tokens": 128_000,
            "truncation": "auto",
            "store": False,
            "include": [],
            "reasoning": {"effort": default_effort},
        }
    else:
        expected_defaults = {
            "max_tokens": 128_000,
            "output_config": {"effort": default_effort},
        }
    assert defaults == expected_defaults


@pytest.mark.parametrize(
    "model,effort",
    [
        *((model, effort) for model, (_, efforts, _) in OPENAI_MODELS.items()
          for effort in efforts),
        *((model, effort) for model in CLAUDE_MODELS for effort in CLAUDE_EFFORTS),
    ],
)
def test_ticket_387_models_prepare_only_allowed_effort_payload(model, effort):
    capabilities = load_model_capabilities()
    defaults = load_default_payloads()
    is_openai = model in OPENAI_MODELS
    if is_openai:
        model_io = OpenAIModelIO(
            model=model,
            api_key="test-key",
            client_factory=lambda **_kwargs: None,
            default_payloads=defaults,
            model_capabilities=capabilities,
        )
        request_payload = {
            "reasoning": {"effort": effort},
            "temperature": 0.1,
            "top_p": 0.2,
            "unknown_setting": {"should": "drop"},
        }
        expected = {
            "instructions": "",
            "max_output_tokens": 128_000,
            "truncation": "auto",
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {"effort": effort},
        }
    else:
        model_io = AnthropicModelIO(
            model=model,
            api_key="test-key",
            client_factory=lambda **_kwargs: None,
            default_payloads=defaults,
            model_capabilities=capabilities,
        )
        request_payload = {
            "output_config": {"effort": effort},
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "temperature": 0.1,
            "top_p": 0.2,
            "top_k": 3,
            "tool_choice": {"type": "tool", "name": "forced_tool"},
            "unknown_setting": {"should": "drop"},
        }
        expected = {
            "max_tokens": 128_000,
            "output_config": {"effort": effort},
        }

    prepared = resolve_prepared_provider_request_payload(
        model_io=model_io,
        request=ModelTurnRequest(
            messages=[{"role": "user", "content": "hello"}],
            payload=request_payload,
        ),
    )

    assert prepared["request_model"] == model
    assert prepared["effective_payload"] == expected
    assert set(prepared["effective_payload"]) == set(expected)
