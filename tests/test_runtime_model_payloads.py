from unchain.runtime.payloads import load_default_payloads, load_model_capabilities


def test_gemini_36_runtime_resources_are_registered():
    from google.genai import types

    capabilities = load_model_capabilities()["gemini-3.6-flash"]
    payload = load_default_payloads()["gemini-3.6-flash"]
    assert capabilities["provider"] == "gemini"
    assert capabilities["max_context_window_tokens"] == 1048576
    assert capabilities["max_output_tokens"] == 65536
    assert capabilities["supports_tools"] is True
    assert capabilities["supports_response_format"] is True
    assert capabilities["reasoning_efforts"] == ["low", "medium", "high"]
    assert payload["thinking_config"] == {
        "thinking_level": "medium", "include_thoughts": True,
    }
    types.GenerateContentConfig.model_validate(payload)


def test_gpt_55_runtime_resources_are_registered():
    capabilities = load_model_capabilities()
    payloads = load_default_payloads()

    assert capabilities["gpt-5.5"]["provider"] == "openai"
    assert capabilities["gpt-5.5"]["supports_tools"] is True
    assert capabilities["gpt-5.5"]["supports_response_format"] is True
    assert capabilities["gpt-5.5"]["supports_previous_response_id"] is True
    assert capabilities["gpt-5.5"]["supports_reasoning"] is True
    assert "reasoning" in capabilities["gpt-5.5"]["allowed_payload_keys"]
    assert payloads["gpt-5.5"]["max_output_tokens"] == 128000
    assert payloads["gpt-5.5"]["reasoning"]["effort"] == "medium"


def test_gpt_6_astra_runtime_resources_are_registered():
    capabilities = load_model_capabilities()
    payloads = load_default_payloads()

    astra = capabilities["gpt-6-astra"]
    assert astra["provider"] == "openai"
    assert astra["max_context_window_tokens"] == 1050000
    assert astra["supports_tools"] is True
    assert astra["supports_response_format"] is True
    assert astra["supports_previous_response_id"] is True
    assert astra["supports_reasoning"] is True
    assert astra["reasoning_efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert "none" not in astra["reasoning_efforts"]
    assert astra["default_reasoning_effort"] == "medium"
    assert astra["input_modalities"] == ["text", "image"]
    assert astra["input_source_types"]["image"] == ["url", "base64"]
    assert "reasoning" in astra["allowed_payload_keys"]
    assert payloads["gpt-6-astra"]["max_output_tokens"] == 128000
    assert payloads["gpt-6-astra"]["reasoning"]["effort"] == "medium"


def test_claude_opus_48_runtime_resources_are_registered():
    capabilities = load_model_capabilities()
    payloads = load_default_payloads()

    opus_48 = capabilities["claude-opus-4-8"]
    assert opus_48["provider"] == "anthropic"
    assert opus_48["max_context_window_tokens"] == 1000000
    assert opus_48["max_output_tokens"] == 128000
    assert opus_48["supports_tools"] is True
    assert opus_48["supports_response_format"] is False
    assert opus_48["supports_previous_response_id"] is False
    assert opus_48["supports_reasoning"] is True
    assert opus_48["input_modalities"] == ["text", "image", "pdf"]
    assert opus_48["input_source_types"]["image"] == ["url", "base64"]
    assert opus_48["input_source_types"]["pdf"] == ["url", "base64"]
    assert opus_48["allowed_payload_keys"] == [
        "max_tokens",
        "thinking",
        "output_config",
        "speed",
    ]
    assert payloads["claude-opus-4-8"]["max_tokens"] == 128000


def test_claude_fable_5_runtime_resources_are_registered():
    capabilities = load_model_capabilities()
    payloads = load_default_payloads()

    fable_5 = capabilities["claude-fable-5"]
    assert fable_5["provider"] == "anthropic"
    assert fable_5["max_context_window_tokens"] == 1000000
    assert fable_5["max_output_tokens"] == 128000
    assert fable_5["supports_tools"] is True
    assert fable_5["supports_response_format"] is False
    assert fable_5["supports_previous_response_id"] is False
    assert fable_5["supports_reasoning"] is True
    assert fable_5["input_modalities"] == ["text", "image", "pdf"]
    assert fable_5["input_source_types"]["image"] == ["url", "base64"]
    assert fable_5["input_source_types"]["pdf"] == ["url", "base64"]
    assert fable_5["allowed_payload_keys"] == [
        "max_tokens",
        "thinking",
        "output_config",
        "fallbacks",
    ]
    assert payloads["claude-fable-5"]["max_tokens"] == 128000


def test_claude_fable_5_1_runtime_resources_are_registered():
    capabilities = load_model_capabilities()
    payloads = load_default_payloads()

    fable_5_1 = capabilities["claude-fable-5-1"]
    assert fable_5_1["provider"] == "anthropic"
    assert fable_5_1["max_context_window_tokens"] == 1000000
    assert fable_5_1["max_output_tokens"] == 128000
    assert fable_5_1["supports_tools"] is True
    assert fable_5_1["supports_response_format"] is False
    assert fable_5_1["supports_previous_response_id"] is False
    assert fable_5_1["supports_reasoning"] is True
    assert fable_5_1["reasoning_efforts"] == ["low", "medium", "high", "xhigh", "max"]
    assert fable_5_1["default_reasoning_effort"] == "high"
    assert fable_5_1["input_modalities"] == ["text", "image", "pdf"]
    assert fable_5_1["input_source_types"]["image"] == ["url", "base64"]
    assert fable_5_1["input_source_types"]["pdf"] == ["url", "base64"]
    assert fable_5_1["allowed_payload_keys"] == [
        "max_tokens",
        "thinking",
        "output_config",
        "fallbacks",
    ]
    assert payloads["claude-fable-5-1"]["max_tokens"] == 128000


def test_claude_haiku_35_runtime_resources_are_removed():
    capabilities = load_model_capabilities()
    payloads = load_default_payloads()

    assert "claude-haiku-3.5" not in capabilities
    assert "claude-haiku-3.5" not in payloads
