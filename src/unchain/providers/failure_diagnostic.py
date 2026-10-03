"""Closed, non-content-bearing diagnostics for HTTP and response failures."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar


_CODES = frozenset({
    "", "invalid_api_key", "invalid_request_error", "authentication_error",
    "permission_error", "permission_denied", "model_not_found",
    "invalid_function_parameters", "invalid_tool", "invalid_value",
    "unsupported_parameter", "unsupported_value", "missing_required_parameter",
    "unknown_parameter", "context_length_exceeded", "content_policy_violation",
    "insufficient_quota", "rate_limit_exceeded", "billing_hard_limit_reached",
    "INVALID_ARGUMENT", "NOT_FOUND", "PERMISSION_DENIED", "UNAUTHENTICATED",
    "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION", "OUT_OF_RANGE", "UNAVAILABLE",
    "INTERNAL", "DEADLINE_EXCEEDED", "ABORTED", "ALREADY_EXISTS", "CANCELLED",
})
_RESPONSE_CODES = frozenset({
    "SAFETY", "RECITATION", "LANGUAGE", "OTHER", "BLOCKLIST",
    "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL", "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER", "NO_IMAGE", "IMAGE_RECITATION",
    "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE",
    "MALFORMED_RESPONSE", "ESCALATION", "PUP_LIMITED_DISABLED",
    "PROMPT_BLOCKED", "EMPTY_RESPONSE",
})


class GeminiResponseFailure(RuntimeError):
    """An explicit provider outcome, distinct from an interrupted stream."""

    def __init__(self, reason: str) -> None:
        if type(reason) is not str or reason not in _RESPONSE_CODES:
            raise ValueError("Unknown Gemini response failure reason")
        self.reason = reason
        super().__init__(f"Gemini response failed: {reason}")
_PARAMETER_PARTS = frozenset({
    "model", "input", "messages", "content", "type", "role", "text",
    "tools", "function", "name", "parameters", "properties", "required",
    "additionalProperties", "items", "enum", "strict", "tool_choice",
    "reasoning", "effort", "summary", "max_output_tokens", "max_tokens",
    "temperature", "top_p", "include", "store", "stream", "truncation",
    "previous_response_id", "instructions", "parallel_tool_calls", "format",
    "response_format", "service_tier",
})


def _safe_parameter(value: object) -> str:
    if type(value) is not str or len(value) > 160:
        return ""
    # Keep only protocol vocabulary and numeric indexes, never arbitrary
    # property names or provider-supplied content.
    parts = value.split(".")
    for part in parts:
        match = re.fullmatch(r"([A-Za-z_]+)(?:\[([0-9]{1,5})\])?", part)
        if match is None or match[1] not in _PARAMETER_PARTS:
            return ""
    return value


@dataclass(frozen=True, slots=True)
class ProviderFailureDiagnostic:
    SCHEMA: ClassVar[str] = "unchain.provider_failure_diagnostic.v1"
    RESPONSE_SCHEMA: ClassVar[str] = "unchain.provider_failure_diagnostic.v2"

    http_status: int | None
    provider_code: str = ""
    parameter: str = ""

    def __post_init__(self) -> None:
        if self.http_status is None:
            if (
                type(self.provider_code) is not str
                or self.provider_code not in _RESPONSE_CODES
                or type(self.parameter) is not str
                or self.parameter != ""
            ):
                raise ValueError("provider response diagnostic is invalid")
            return
        if type(self.http_status) is not int or not 400 <= self.http_status <= 599:
            raise ValueError("provider diagnostic HTTP status is invalid")
        if type(self.provider_code) is not str or self.provider_code not in _CODES:
            raise ValueError("provider diagnostic code is invalid")
        if type(self.parameter) is not str or _safe_parameter(self.parameter) != self.parameter:
            raise ValueError("provider diagnostic parameter is invalid")

    def to_dict(self) -> dict:
        return {"schema": self.RESPONSE_SCHEMA if self.http_status is None else self.SCHEMA, "http_status": self.http_status,
                "provider_code": self.provider_code, "parameter": self.parameter}

    @classmethod
    def from_dict(cls, value: dict) -> ProviderFailureDiagnostic:
        if type(value) is not dict or set(value) != {"schema", "http_status", "provider_code", "parameter"}:
            raise ValueError("provider diagnostic fields are invalid")
        expected = cls.RESPONSE_SCHEMA if value["http_status"] is None else cls.SCHEMA
        if value["schema"] != expected:
            raise ValueError("provider diagnostic schema is invalid")
        return cls(value["http_status"], value["provider_code"], value["parameter"])

    @classmethod
    def from_exception(cls, error: BaseException) -> ProviderFailureDiagnostic | None:
        if type(error) is GeminiResponseFailure:
            return cls(None, error.reason)
        status = getattr(error, "status_code", None)
        if type(status) is not int:
            status = getattr(getattr(error, "response", None), "status_code", None)
        if type(status) is not int:
            from google.genai.errors import APIError

            if isinstance(error, APIError):
                status = error.code
        if type(status) is not int or not 400 <= status <= 599:
            return None
        body = getattr(error, "body", None)
        body = body if type(body) is dict else {}
        if type(body.get("error")) is dict:
            body = body["error"]
        code = body.get("code", getattr(error, "code", ""))
        from google.genai.errors import APIError

        if isinstance(error, APIError):
            code = error.status
        parameter = body.get("param", getattr(error, "param", ""))
        return cls(status, code if type(code) is str and code in _CODES else "", _safe_parameter(parameter))

    def summary(self) -> str:
        if self.http_status is None:
            return f"Gemini generation stopped (reason={self.provider_code})"
        message = {
            400: "Provider rejected the request",
            401: "Provider rejected the credentials",
            403: "Provider denied access",
            404: "Provider model or endpoint was not found",
            429: "Provider rate or quota limit reached",
        }.get(self.http_status, "Provider HTTP request failed")
        detail = f"{message} (HTTP {self.http_status}"
        if self.provider_code:
            detail += f", code={self.provider_code}"
        if self.parameter:
            detail += f", parameter={self.parameter}"
        return detail + ")"
