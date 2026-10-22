"""Closed observations of a request whose durable outcome remains unknown."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import ClassVar

from .failure_diagnostic import ProviderFailureDiagnostic


_REASONS = frozenset({
    "stream_incomplete", "connection_interrupted", "timeout",
    "response_parse_error", "malformed_tool_call", "conflicting_finish_reason",
    "unsupported_finish_reason", "provider_stream_error", "request_validation_failed",
    "local_processing_error", "execution_interrupted", "unrecorded_outcome",
    "invalid_result", "before_send_failed",
})
_PHASES = frozenset({
    "request_preparation", "requesting", "reading_response", "processing_response",
    "before_send", "result_processing", "recovery",
})
_MESSAGES = {
    "stream_incomplete": "The response stream ended without a completion marker",
    "connection_interrupted": "The connection was interrupted",
    "timeout": "The provider request timed out",
    "response_parse_error": "The response could not be decoded",
    "malformed_tool_call": "The response contained an invalid tool call",
    "conflicting_finish_reason": "The response contained conflicting completion markers",
    "unsupported_finish_reason": "The response used an unrecognized completion marker",
    "provider_stream_error": "The provider reported an error while processing the request",
    "request_validation_failed": "The local provider request failed validation",
    "local_processing_error": "A local error interrupted provider response processing",
    "execution_interrupted": "Execution was interrupted before its outcome was recorded",
    "unrecorded_outcome": "The previous request has no recorded outcome; its original cause was not recorded",
    "invalid_result": "The provider adapter returned an invalid result",
    "before_send_failed": "The request preparation callback failed before the provider send",
}


@dataclass(frozen=True, slots=True)
class ProviderUncertaintyDiagnostic:
    SCHEMA: ClassVar[str] = "unchain.provider_uncertainty_diagnostic.v1"

    reason: str
    phase: str
    http_status: int | None = None
    provider_code: str = ""

    def __post_init__(self) -> None:
        if type(self.reason) is not str or self.reason not in _REASONS:
            raise ValueError("uncertainty reason is invalid")
        if type(self.phase) is not str or self.phase not in _PHASES:
            raise ValueError("uncertainty phase is invalid")
        if type(self.provider_code) is not str:
            raise TypeError("uncertainty provider code must be exact text")
        if self.http_status is None:
            if self.provider_code and self.reason != "provider_stream_error":
                raise ValueError("provider code requires an observed HTTP status")
            if self.provider_code:
                ProviderFailureDiagnostic(500, self.provider_code)
        elif self.http_status == 200 and type(self.http_status) is int:
            if self.reason != "provider_stream_error":
                raise ValueError("HTTP 200 metadata requires a provider stream error")
            ProviderFailureDiagnostic(500, self.provider_code)
        else:
            if self.reason != "provider_stream_error":
                raise ValueError("HTTP metadata requires a provider stream error")
            ProviderFailureDiagnostic(self.http_status, self.provider_code)

    def to_dict(self) -> dict:
        return {
            "schema": self.SCHEMA, "reason": self.reason, "phase": self.phase,
            "http_status": self.http_status, "provider_code": self.provider_code,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ProviderUncertaintyDiagnostic:
        if type(value) is not dict or set(value) != {
            "schema", "reason", "phase", "http_status", "provider_code",
        }:
            raise ValueError("uncertainty diagnostic fields are invalid")
        if value["schema"] != cls.SCHEMA:
            raise ValueError("uncertainty diagnostic schema is invalid")
        return cls(value["reason"], value["phase"], value["http_status"], value["provider_code"])

    @classmethod
    def from_exception(
        cls, error: BaseException, *, phase: str,
    ) -> ProviderUncertaintyDiagnostic:
        import httpx
        from google.genai.errors import APIError, UnknownApiResponseError
        from pydantic import ValidationError

        if type(error) is ProviderUncertaintyObservation:
            return error.diagnostic
        if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
            phase = "requesting"
        if isinstance(error, httpx.TimeoutException):
            reason = "timeout"
        elif isinstance(error, (httpx.DecodingError, json.JSONDecodeError, UnicodeDecodeError, UnknownApiResponseError)):
            reason = "response_parse_error"
        elif isinstance(error, httpx.TransportError):
            reason = "connection_interrupted"
        elif isinstance(error, ValidationError):
            reason = "request_validation_failed" if phase == "request_preparation" else "response_parse_error"
        elif isinstance(error, APIError):
            # The SDK's in-stream error code is NOT the enclosing HTTP status:
            # it can report UNAVAILABLE inside an HTTP 200 response.
            status = getattr(getattr(error, "response", None), "status_code", None)
            if type(status) is not int or not (status == 200 or 400 <= status <= 599):
                status = None
            code = error.status
            try:
                ProviderFailureDiagnostic(500, code)
            except (TypeError, ValueError):
                code = ""
            return cls(
                "provider_stream_error", phase, status, code,
            )
        elif not isinstance(error, Exception):
            reason = "execution_interrupted"
        else:
            reason = "local_processing_error"
        return cls(reason, phase)

    def summary(self) -> str:
        detail = f"reason={self.reason}, phase={self.phase}"
        if self.http_status is not None:
            detail += f", HTTP {self.http_status}"
        if self.provider_code:
            detail += f", code={self.provider_code}"
        return f"{_MESSAGES[self.reason]} ({detail})"


class ProviderUncertaintyObservation(RuntimeError):
    """An adapter's safe observation; never a retry authorization."""

    def __init__(self, diagnostic: ProviderUncertaintyDiagnostic) -> None:
        if type(diagnostic) is not ProviderUncertaintyDiagnostic:
            raise TypeError("uncertainty observation requires an exact diagnostic")
        self.diagnostic = diagnostic
        super().__init__(diagnostic.summary())
