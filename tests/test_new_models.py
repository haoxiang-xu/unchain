"""Test new model configurations."""

import pytest
from unchain.schemas.models import GPT_4O, CLAUDE_SONNET_35, GEMINI_PRO_15


def test_gpt_4o_configuration():
    """Test GPT-4o model configuration."""
    assert GPT_4O.name == "gpt-4o"
    assert GPT_4O.capabilities.provider == "openai"
    assert GPT_4O.capabilities.provider_model == "gpt-4o"
    assert GPT_4O.capabilities.max_context_window_tokens == 128000
    assert GPT_4O.capabilities.supports_tools is True
    assert GPT_4O.capabilities.supports_response_format is True
    assert "text" in GPT_4O.capabilities.input_modalities
    assert "image" in GPT_4O.capabilities.input_modalities
    assert GPT_4O.default_payload.payload["max_tokens"] == 16384
    assert GPT_4O.default_payload.payload["temperature"] == 0.7


def test_claude_sonnet_35_configuration():
    """Test Claude Sonnet 3.5 model configuration."""
    assert CLAUDE_SONNET_35.name == "claude-sonnet-3.5"
    assert CLAUDE_SONNET_35.capabilities.provider == "anthropic"
    assert CLAUDE_SONNET_35.capabilities.provider_model == "claude-3-5-sonnet-20241022"
    assert CLAUDE_SONNET_35.capabilities.max_context_window_tokens == 200000
    assert CLAUDE_SONNET_35.capabilities.supports_tools is True
    assert CLAUDE_SONNET_35.capabilities.supports_reasoning is True
    assert "text" in CLAUDE_SONNET_35.capabilities.input_modalities
    assert "image" in CLAUDE_SONNET_35.capabilities.input_modalities
    assert CLAUDE_SONNET_35.default_payload.payload["max_tokens"] == 32000


def test_gemini_pro_15_configuration():
    """Test Gemini Pro 1.5 model configuration."""
    assert GEMINI_PRO_15.name == "gemini-2.5-pro"
    assert GEMINI_PRO_15.capabilities.provider == "gemini"
    assert GEMINI_PRO_15.capabilities.provider_model == "gemini-2.5-pro"
    assert GEMINI_PRO_15.capabilities.max_context_window_tokens == 1048576
    assert GEMINI_PRO_15.capabilities.supports_tools is True
    assert "text" in GEMINI_PRO_15.capabilities.input_modalities
    assert "image" in GEMINI_PRO_15.capabilities.input_modalities
    assert "pdf" in GEMINI_PRO_15.capabilities.input_modalities
    assert GEMINI_PRO_15.default_payload.payload["max_output_tokens"] == 65536


def test_model_serialization():
    """Test model configuration serialization."""
    models = [GPT_4O, CLAUDE_SONNET_35, GEMINI_PRO_15]
    
    for model in models:
        # Test to_dict serialization
        model_dict = model.to_dict()
        assert "name" in model_dict
        assert "capabilities" in model_dict
        assert "default_payload" in model_dict
        
        # Test capabilities to_dict
        capabilities_dict = model.capabilities.to_dict()
        assert "provider" in capabilities_dict
        assert "max_context_window_tokens" in capabilities_dict
        
        # Test payload to_dict
        payload_dict = model.default_payload.to_dict()
        assert isinstance(payload_dict, dict)


def test_model_deserialization():
    """Test model configuration deserialization."""
    # Test GPT-4o deserialization
    capabilities_data = {
        "provider": "openai",
        "max_context_window_tokens": 128000,
        "supports_tools": True,
        "supports_response_format": True,
        "input_modalities": ["text", "image"]
    }
    payload_data = {
        "max_tokens": 16384,
        "temperature": 0.7
    }
    
    model = GPT_4O.__class__.from_dict("test-gpt-4o", capabilities_data, payload_data)
    assert model.name == "test-gpt-4o"
    assert model.capabilities.provider == "openai"
    assert model.default_payload.payload["max_tokens"] == 16384
