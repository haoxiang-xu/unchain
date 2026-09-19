"""Closed, non-content-bearing diagnostics for provider HTTP failures."""

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
})
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

    http_status: int
    provider_code: str = ""
    parameter: str = ""

    def __post_init__(self) -> None:
        if type(self.http_status) is not int or not 400 <= self.http_status <= 599:
            raise ValueError("provider diagnostic HTTP status is invalid")
        if type(self.provider_code) is not str or self.provider_code not in _CODES:
            raise ValueError("provider diagnostic code is invalid")
        if type(self.parameter) is not str or _safe_parameter(self.parameter) != self.parameter:
            raise ValueError("provider diagnostic parameter is invalid")

    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, "http_status": self.http_status,
                "provider_code": self.provider_code, "parameter": self.parameter}

    @classmethod
    def from_dict(cls, value: dict) -> ProviderFailureDiagnostic:
        if type(value) is not dict or set(value) != {"schema", "http_status", "provider_code", "parameter"}:
            raise ValueError("provider diagnostic fields are invalid")
        if value["schema"] != cls.SCHEMA:
            raise ValueError("provider diagnostic schema is invalid")
        return cls(value["http_status"], value["provider_code"], value["parameter"])

    @classmethod
    def from_exception(cls, error: BaseException) -> ProviderFailureDiagnostic | None:
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
        parameter = body.get("param", getattr(error, "param", ""))
        return cls(status, code if type(code) is str and code in _CODES else "", _safe_parameter(parameter))

    def summary(self) -> str:
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
