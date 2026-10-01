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
# Google RPC status names; the only provider_status values that can be stored.
_PROVIDER_STATUSES = frozenset({
    "UNAVAILABLE", "NOT_FOUND", "RESOURCE_EXHAUSTED", "PERMISSION_DENIED",
    "INVALID_ARGUMENT", "UNAUTHENTICATED", "FAILED_PRECONDITION", "INTERNAL",
    "DEADLINE_EXCEEDED",
})
# Shape of a model id. Persisted replacement models are checked against this
# alone, so an old record stays readable after the catalog drops that model.
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9.-]{0,60}")
# Google names the replacement only in this phrase; the retired model appears
# earlier in the same sentence as "models/<id>" and must never be picked.
_REPLACEMENT_PHRASE = re.compile(r"\buse models/([a-z0-9][a-z0-9.-]{0,60})")


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


def _catalog_replacement_model(message: object) -> str:
    """The model a 404 message tells the caller to use, only if we export it.

    The message text is read here and discarded: only an id that exists in the
    Gemini catalog survives, so provider-supplied content cannot be stored.
    """

    if type(message) is not str:
        return ""
    match = _REPLACEMENT_PHRASE.search(message)
    if match is None:
        return ""
    candidate = match[1].rstrip(".-")
    if not candidate:
        return ""
    from unchain.runtime.payloads import load_model_capabilities

    capabilities = load_model_capabilities().get(candidate)
    if type(capabilities) is dict and capabilities.get("provider") == "gemini":
        return candidate
    return ""


@dataclass(frozen=True, slots=True)
class ProviderFailureDiagnostic:
    SCHEMA: ClassVar[str] = "unchain.provider_failure_diagnostic.v1"
    # Used only when provider_status or replacement_model is set; a record that
    # carries neither keeps the v1 shape and bytes.
    SCHEMA_V2: ClassVar[str] = "unchain.provider_failure_diagnostic.v2"

    http_status: int
    provider_code: str = ""
    parameter: str = ""
    provider_status: str = ""
    replacement_model: str = ""

    def __post_init__(self) -> None:
        if type(self.http_status) is not int or not 400 <= self.http_status <= 599:
            raise ValueError("provider diagnostic HTTP status is invalid")
        if type(self.provider_code) is not str or self.provider_code not in _CODES:
            raise ValueError("provider diagnostic code is invalid")
        if type(self.parameter) is not str or _safe_parameter(self.parameter) != self.parameter:
            raise ValueError("provider diagnostic parameter is invalid")
        if type(self.provider_status) is not str or (
            self.provider_status and self.provider_status not in _PROVIDER_STATUSES
        ):
            raise ValueError("provider diagnostic status is invalid")
        if type(self.replacement_model) is not str or (
            self.replacement_model and _MODEL_ID.fullmatch(self.replacement_model) is None
        ):
            raise ValueError("provider diagnostic replacement model is invalid")

    def to_dict(self) -> dict:
        record = {"schema": self.SCHEMA, "http_status": self.http_status,
                  "provider_code": self.provider_code, "parameter": self.parameter}
        if self.provider_status or self.replacement_model:
            record["schema"] = self.SCHEMA_V2
            record["provider_status"] = self.provider_status
            record["replacement_model"] = self.replacement_model
        return record

    @classmethod
    def from_dict(cls, value: dict) -> ProviderFailureDiagnostic:
        if type(value) is not dict:
            raise ValueError("provider diagnostic fields are invalid")
        v1_keys = {"schema", "http_status", "provider_code", "parameter"}
        schema = value.get("schema")
        if schema == cls.SCHEMA:
            if set(value) != v1_keys:
                raise ValueError("provider diagnostic fields are invalid")
            return cls(value["http_status"], value["provider_code"], value["parameter"])
        if schema == cls.SCHEMA_V2:
            if set(value) != v1_keys | {"provider_status", "replacement_model"}:
                raise ValueError("provider diagnostic fields are invalid")
            diagnostic = cls(
                value["http_status"], value["provider_code"], value["parameter"],
                value["provider_status"], value["replacement_model"],
            )
            if not (diagnostic.provider_status or diagnostic.replacement_model):
                # Would serialize as v1: two spellings of one record are refused.
                raise ValueError("provider diagnostic v2 must carry a new field")
            return diagnostic
        raise ValueError("provider diagnostic schema is invalid")

    @classmethod
    def from_exception(cls, error: BaseException) -> ProviderFailureDiagnostic | None:
        from google.genai.errors import APIError

        status = getattr(error, "status_code", None)
        if type(status) is not int:
            status = getattr(getattr(error, "response", None), "status_code", None)
        if type(status) is not int:
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
        provider_status = replacement_model = ""
        # Only Google's SDK error carries an RPC status and a message whose
        # replacement hint we know how to validate.
        if isinstance(error, APIError):
            raw_status = getattr(error, "status", None)
            if type(raw_status) is str and raw_status in _PROVIDER_STATUSES:
                provider_status = raw_status
            if status == 404:
                replacement_model = _catalog_replacement_model(getattr(error, "message", None))
        return cls(
            status,
            code if type(code) is str and code in _CODES else "",
            _safe_parameter(parameter),
            provider_status,
            replacement_model,
        )

    def summary(self) -> str:
        message = {
            400: "Provider rejected the request",
            401: "Provider rejected the credentials",
            403: "Provider denied access",
            404: "Provider model or endpoint was not found",
            429: "Provider rate or quota limit reached",
            500: "Provider reported an internal error",
            502: "Provider gateway error",
            503: "Provider service is unavailable or overloaded",
            504: "Provider timed out",
        }.get(self.http_status, "Provider HTTP request failed")
        detail = f"{message} (HTTP {self.http_status}"
        if self.provider_code:
            detail += f", code={self.provider_code}"
        if self.parameter:
            detail += f", parameter={self.parameter}"
        if self.provider_status:
            detail += f", status={self.provider_status}"
        if self.replacement_model:
            detail += f"; suggested model: {self.replacement_model}"
        return detail + ")"


def _sdk_enum_names(enum_name: str) -> frozenset[str]:
    try:
        from google.genai import types

        return frozenset(member.name for member in getattr(types, enum_name))
    except Exception:
        return frozenset()


class ProviderResponseEndedError(RuntimeError):
    """A streamed response that ended without a usable answer.

    ``reason`` is kept only when it is a member of the SDK's closed
    FinishReason / BlockedReason enums, so it can be shown to the user.
    The message keeps the historical wording for existing callers.
    """

    _KINDS: ClassVar[frozenset[str]] = frozenset({"finish_reason", "blocked", "empty"})

    def __init__(self, kind: str, reason: str = "") -> None:
        if kind not in self._KINDS:
            raise ValueError("provider response end kind is invalid")
        allowed = (
            _sdk_enum_names("FinishReason")
            if kind == "finish_reason"
            else _sdk_enum_names("BlockedReason")
            if kind == "blocked"
            else frozenset()
        )
        self.kind = kind
        self.reason = reason if type(reason) is str and reason in allowed else ""
        if kind == "blocked":
            message = "Gemini blocked the prompt"
        elif kind == "finish_reason":
            message = f"Gemini generation ended: {self.reason or 'unknown'}"
        else:
            message = "Gemini returned no content"
        super().__init__(message)

    def summary(self) -> str:
        suffix = f" ({self.reason})" if self.reason else ""
        if self.kind == "blocked":
            return f"Provider blocked the prompt{suffix}"
        if self.kind == "finish_reason":
            return f"Provider ended the response early{suffix}"
        return "Provider returned no content"

