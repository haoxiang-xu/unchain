"""Rejected input evidence only: these fixtures never contact a provider."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict
import json

import anthropic
from google.genai import errors as google_errors
import httpx
import openai
import pytest

from unchain.providers.context_limit import ContextLimitRejection, classify_context_limit


CLAUDE = "prompt is too long: 210000 tokens > 200000 maximum"
OPENAI = (
    "This model's maximum context length is 4097 tokens. However, you requested "
    "4927 tokens (3927 in the messages, 1000 in the completion). "
    "Please reduce the length of the messages or completion."
)
OPENAI_MESSAGES = (
    "This model's maximum context length is 128000 tokens. However, your messages "
    "resulted in 130000 tokens. Please reduce the length of the messages."
)
GEMINI = (
    "The input token count (1200000) exceeds the maximum number of tokens "
    "allowed (1048576)."
)


def _response(status=400, body=None):
    return httpx.Response(
        status, json=body, request=httpx.Request("POST", "https://example.invalid/private"),
    )


def _openai(body, status=400):
    return openai.APIStatusError("private SDK rendering", response=_response(status), body=body)


def _anthropic(message=CLAUDE, status=400):
    return anthropic.APIStatusError(
        "private SDK rendering", response=_response(status),
        body={"type": "error", "error": {"type": "invalid_request_error", "message": message}},
    )


def _gemini(message=GEMINI, status=400, provider_status="INVALID_ARGUMENT", http_status=400):
    response = None if http_status is None else _response(http_status)
    return google_errors.ClientError(
        status, {"error": {"code": status, "status": provider_status, "message": message}}, response,
    )


def _http(body, status=400):
    response = _response(status, body)
    return httpx.HTTPStatusError("private HTTP rendering", request=response.request, response=response)


@pytest.mark.parametrize("body", [
    {"code": "context_length_exceeded"},
    {"type": "context_length_exceeded"},
    {"error": {"code": "context_length_exceeded", "type": "invalid_request_error"}},
    {"error": {"code": 400, "metadata": {"error_type": "context_length_exceeded"}}},
    json.dumps({"error": {"code": "context_length_exceeded"}}),
])
@pytest.mark.parametrize("status", [400, 422])
def test_real_openai_sdk_structured_rejections(body, status):
    # OpenRouter's numeric HTTP code must agree with the actual rejection.
    if isinstance(body, dict) and body.get("error", {}).get("code") == 400 and status == 422:
        body = {"error": {"code": status, "metadata": {"error_type": "context_length_exceeded"}}}
    assert classify_context_limit(_openai(body, status)) == ContextLimitRejection(status, "structured_code")


@pytest.mark.parametrize("error", [
    _anthropic(),
    _openai({"error": {"type": "invalid_request_error", "message": OPENAI}}),
    _openai({"message": OPENAI_MESSAGES}),
    _gemini(),
    _gemini(provider_status="FAILED_PRECONDITION"),
    _http({"error": CLAUDE}),  # Ollama uses a JSON error string, not an SDK rendering.
    _http({"error": OPENAI_MESSAGES}, 422),
])
def test_narrow_protocol_messages_on_actual_http_rejection(error):
    result = classify_context_limit(error)
    assert result is not None
    assert result.match_kind == "message"
    assert result.http_status in (400, 422)


@pytest.mark.parametrize("status", [200, 401, 403, 404, 413, 429, 500, 503])
def test_context_code_cannot_override_non_input_http_status(status):
    assert classify_context_limit(_openai({"code": "context_length_exceeded"}, status)) is None


@pytest.mark.parametrize("code", [
    "invalid_api_key", "authentication_error", "permission_error", "permission_denied",
    "insufficient_quota", "rate_limit_exceeded", "billing_hard_limit_reached",
    "content_policy_violation", "request_too_large", "invalid_function_parameters",
    "invalid_tool", "invalid_value", "unsupported_parameter", "missing_required_parameter",
    "max_tokens", "max_output_tokens", "length", "model_context_window_exceeded",
    "deepseek_context_limit", "kimi_context_limit", "unknown_error",
])
def test_conflicting_error_codes_win_over_matching_message_or_metadata(code):
    body = {"error": {"code": code, "message": OPENAI, "metadata": {"error_type": "context_length_exceeded"}}}
    assert classify_context_limit(_openai(body)) is None


@pytest.mark.parametrize("provider_status", [
    "RESOURCE_EXHAUSTED", "PERMISSION_DENIED", "UNAUTHENTICATED", "NOT_FOUND", "INTERNAL",
])
def test_gemini_conflicting_status_is_not_context_recovery(provider_status):
    assert classify_context_limit(_gemini(provider_status=provider_status)) is None


@pytest.mark.parametrize("http_status", [None, 200, 422, 429])
def test_gemini_requires_matching_actual_http_rejection(http_status):
    assert classify_context_limit(_gemini(http_status=http_status)) is None


def test_sdk_status_and_response_status_must_agree():
    error = _openai({"code": "context_length_exceeded"})
    error.status_code = 422
    assert classify_context_limit(error) is None


@pytest.mark.parametrize("message", [
    "private text quoting " + CLAUDE,
    CLAUDE + "\nAnother error: payment required",
    "maximum context length exceeded",
    "request body too large: 200000 bytes",
    "input length must be less than 100 bytes",
    "max_tokens must be less than or equal to 4096",
    "output token count exceeds the maximum number of tokens allowed",
    "context window is 128000 tokens",
    "length", "invalid_prompt", "invalid schema for tools[0]",
    "insufficient quota", "rate limit exceeded", "content policy violation",
    "model_context_window_exceeded",
    OPENAI + " payment required",
    GEMINI.replace("input", "output"),
])
def test_ambiguous_or_unrelated_messages_never_trigger_recovery(message):
    assert classify_context_limit(_anthropic(message)) is None
    assert classify_context_limit(_http({"error": message})) is None


@pytest.mark.parametrize("body", [
    {}, {"error": {"type": "invalid_request_error"}}, {"error": "private"},
    {"error": {"message": OPENAI, "param": "max_output_tokens"}},
    {"error": {"message": OPENAI, "param": "tools[0].parameters"}},
    {"error": {"code": "context_length_exceeded", "type": "authentication_error"}},
    {"error": {"code": "context_length_exceeded", "status": "RESOURCE_EXHAUSTED"}},
    {"error": {"code": "context_length_exceeded", "message": "request body exceeds size limit in bytes"}},
    {"error": {"code": "context_length_exceeded", "message": "quota exhausted"}},
    {"error": {"code": "context_length_exceeded", "message": "output token limit exceeded"}},
    {"error": {"code": "context_length_exceeded", "message": "content policy violation"}},
    {"code": "insufficient_quota", "error": {"code": "context_length_exceeded"}},
    {"error": {"code": 429, "metadata": {"error_type": "context_length_exceeded"}}},
    {"data": {"error": {"code": "context_length_exceeded"}}},
])
def test_status_alone_nested_arbitrary_content_and_conflicts_are_not_evidence(body):
    assert classify_context_limit(_openai(body)) is None


def test_free_form_exception_text_and_generic_spoofed_sdk_fields_are_not_evidence():
    error = RuntimeError(OPENAI)
    error.status_code = 400
    error.body = {"code": "context_length_exceeded"}
    assert classify_context_limit(error) is None
    assert classify_context_limit(RuntimeError(CLAUDE)) is None
    assert classify_context_limit(openai.APIConnectionError(request=_response().request)) is None
    assert classify_context_limit(openai.APITimeoutError(request=_response().request)) is None


def test_successful_or_unread_http_stream_is_not_rejection_evidence():
    for status in (200, 400):
        response = httpx.Response(status, stream=httpx.ByteStream(b'{}'), request=_response().request)
        error = httpx.HTTPStatusError(CLAUDE, request=response.request, response=response)
        assert classify_context_limit(error) is None


@pytest.mark.parametrize("body", [
    '{"error":', '[{"code":"context_length_exceeded"}]',
    '{"error":{"code":"context_length_exceeded"},"padding":"' + 'x' * 20000 + '"}',
    {"error": {"message": "x" * 20000 + OPENAI}},
    {"error": {"error": {"code": "context_length_exceeded"}}},
])
def test_only_bounded_known_envelopes_are_inspected(body):
    assert classify_context_limit(_openai(body)) is None


def test_result_is_closed_frozen_and_never_retains_provider_contents():
    secret = "sk-private-user-prompt-and-endpoint"
    result = classify_context_limit(_openai({"code": "context_length_exceeded", "message": secret}))
    assert asdict(result) == {"http_status": 400, "match_kind": "structured_code"}
    assert secret not in repr(result)
    assert not hasattr(result, "__dict__")
    with pytest.raises(FrozenInstanceError):
        result.http_status = 422


@pytest.mark.parametrize("status,kind", [(200, "message"), (True, "message"), ("400", "message"), (400, "raw")])
def test_result_constructor_rejects_open_or_invalid_values(status, kind):
    with pytest.raises((TypeError, ValueError)):
        ContextLimitRejection(status, kind)


@pytest.mark.parametrize("message", [
    OPENAI.replace("model's", "endpoint's"),
    OPENAI_MESSAGES.replace("model's", "endpoint's"),
    (
        "This endpoint's maximum context length is 16384 tokens. However, you "
        "requested about 16815 tokens (16815 of text input, 0 in the output). "
        "Please reduce the length of either one, or use the \"middle-out\" "
        "transform to compress your prompt automatically."
    ),
])
def test_endpoint_context_limit_uses_equally_bounded_compatible_message(message):
    assert classify_context_limit(_openai({"error": {"code": 400, "message": message}})) == (
        ContextLimitRejection(400, "message")
    )
    assert classify_context_limit(_openai({"error": {"code": "insufficient_quota", "message": message}})) is None
    assert classify_context_limit(_openai({"error": {"message": "private quote: " + message}})) is None


@pytest.mark.parametrize("message", [
    "Invalid API key provided", "Invalid credentials", "Insufficient credits",
    "Account balance too low", "max_tokens must be less than 4096",
    "Request too large", "Access denied",
])
def test_structured_context_code_does_not_override_non_context_message(message):
    assert classify_context_limit(_openai({"code": "context_length_exceeded", "message": message})) is None


@pytest.mark.parametrize("message", ["prompt is too long", "prompt is too long.", "prompt is too long!"])
@pytest.mark.parametrize("status", [400, 422])
def test_plain_claude_documented_message_requires_exact_generic_error_envelope(message, status):
    # https://platform.claude.com/docs/en/build-with-claude/context-windows
    assert classify_context_limit(_anthropic(message, status)) == ContextLimitRejection(status, "message")
    assert classify_context_limit(_http({"error": message}, status)) is None
    assert classify_context_limit(_anthropic("private quote: " + message, status)) is None
    assert classify_context_limit(_anthropic(message + " unknown explanation", status)) is None
