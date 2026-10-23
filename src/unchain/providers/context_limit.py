"""Classify explicit rejected input context, without retaining provider content.

Only HTTP 400/422 rejections and known protocol evidence qualify. Successful
length stops, stream errors, and an exception's free-form rendering do not.
This module neither retries a request nor changes the context budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Literal

import httpx


_MAX_JSON_CHARS = 16384
_MAX_MESSAGE_CHARS = 8192
_CONTEXT_CODE = "context_length_exceeded"
# These are generic envelopes, not context evidence in their own right.
_GENERIC_CODES = frozenset({"invalid_request_error", "INVALID_ARGUMENT", "FAILED_PRECONDITION"})
_INPUT_PARAMETERS = frozenset({"input", "messages", "prompt"})
_NUMBER = r"[0-9]{1,12}"
_MESSAGE_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"prompt is too long: {_NUMBER} tokens > {_NUMBER} maximum\.?",
    rf"This (?:model|endpoint)'s maximum context length is {_NUMBER} tokens\. "
    rf"However, you requested {_NUMBER} tokens"
    rf"(?: \({_NUMBER} in the messages, {_NUMBER} in the completion\))?\. "
    r"Please reduce the length of the messages or completion\.",
    rf"This (?:model|endpoint)'s maximum context length is {_NUMBER} tokens\. "
    rf"However, your messages resulted in {_NUMBER} tokens\. "
    r"Please reduce the length of the messages\.",
    rf"This endpoint's maximum context length is {_NUMBER} tokens\. "
    rf"However, you requested about {_NUMBER} tokens "
    rf"\({_NUMBER} of text input, {_NUMBER} in the output\)\. "
    r'Please reduce the length of either one, or use the "middle-out" '
    r"transform to compress your prompt automatically\.",
    rf"The input token count \({_NUMBER}\) exceeds the maximum number of tokens "
    rf"allowed \({_NUMBER}\)\.?",
))
# A context code cannot override an explicitly conflicting failure description.
_CONFLICT_MESSAGE = re.compile(
    r"\b(?:authentication|unauthenticated|unauthorized|permission|forbidden|"
    r"api[ _-]?key|credentials|access denied|credits?|account balance|"
    r"billing|payment|quota|rate[ -]?limit|policy|safety|bytes?|"
    r"request body|request too large|payload too large|output token(?:s| limit)?|"
    r"max_tokens|max_output_tokens|invalid (?:tool|schema)|stream (?:interrupted|error))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ContextLimitRejection:
    """Closed evidence safe to carry across diagnostics/persistence boundaries."""

    http_status: Literal[400, 422]
    match_kind: Literal["structured_code", "message"]

    def __post_init__(self) -> None:
        if type(self.http_status) is not int or self.http_status not in (400, 422):
            raise ValueError("Context rejection requires HTTP 400 or 422")
        if type(self.match_kind) is not str or self.match_kind not in ("structured_code", "message"):
            raise ValueError("Unknown context rejection match kind")


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _body_object(value: object) -> dict | None:
    if type(value) is str:
        if len(value) > _MAX_JSON_CHARS:
            return None
        try:
            value = json.loads(value, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError):
            return None
    return value if type(value) is dict else None


def _http_evidence(error: BaseException) -> tuple[int, list[dict]] | None:
    """Read only actual HTTP rejection plus the supported SDK body fields."""
    from anthropic import APIStatusError as AnthropicStatusError
    from google.genai.errors import APIError as GeminiAPIError
    from openai import APIStatusError as OpenAIStatusError

    response = getattr(error, "response", None)
    if not isinstance(response, httpx.Response):
        return None
    status = response.status_code
    if type(status) is not int or status not in (400, 422):
        return None
    bodies = []
    if isinstance(error, (OpenAIStatusError, AnthropicStatusError)):
        if type(error.status_code) is not int or error.status_code != status:
            return None
        bodies.append(getattr(error, "body", None))
    elif isinstance(error, GeminiAPIError):
        # Gemini's code may describe a streamed error under HTTP 200; it is not
        # independently sufficient evidence of an input that was never accepted.
        if type(error.code) is not int or error.code != status:
            return None
        bodies.extend((getattr(error, "details", None), getattr(error, "response_json", None)))
    elif not isinstance(error, httpx.HTTPStatusError):
        return None
    # Never consume an unread stream or perform I/O while classifying.
    try:
        content = response.content
    except httpx.ResponseNotRead:
        return None
    if content and len(content) <= _MAX_JSON_CHARS:
        try:
            bodies.append(content.decode("utf-8"))
        except UnicodeDecodeError:
            return None
    elif len(content) > _MAX_JSON_CHARS:
        return None
    objects = []
    for value in bodies:
        if value is None:
            continue
        body = _body_object(value)
        # SDK test responses and empty wire responses may contain JSON null.
        if body is None:
            if value not in ("null", ""):
                return None
            continue
        objects.append(body)
    return status, objects


def classify_context_limit(error: BaseException) -> ContextLimitRejection | None:
    """Return closed evidence only for a confirmed provider input rejection.

    Inspect fixed envelope paths only: the root and its immediate ``error``,
    plus ``metadata.error_type`` on those two objects. Never search arbitrary
    nested content or stringify the exception. Unknown/conflicting codes fail
    closed even if another field contains a matching phrase.
    """
    evidence = _http_evidence(error)
    if evidence is None:
        return None
    status, bodies = evidence
    code_match = False
    message_match = False
    for body in bodies:
        nested = body.get("error")
        envelopes = (body, nested) if type(nested) is dict else (body,)
        for envelope in envelopes:
            for key in ("code", "type", "status"):
                value = envelope.get(key)
                if value is None or value == "":
                    continue
                if type(value) is int and key == "code" and value == status:
                    continue
                if value == "error" and key == "type" and envelope is body and type(nested) is dict:
                    continue
                if type(value) is not str or len(value) > 80:
                    return None
                if value == _CONTEXT_CODE and key in ("code", "type"):
                    code_match = True
                elif value not in _GENERIC_CODES:
                    return None
            metadata = envelope.get("metadata")
            if type(metadata) is dict:
                error_type = metadata.get("error_type")
                if error_type == _CONTEXT_CODE:
                    code_match = True
                elif error_type not in (None, ""):
                    return None
            parameter = envelope.get("param")
            if parameter is not None and parameter != "":
                if type(parameter) is not str or parameter not in _INPUT_PARAMETERS:
                    return None
            message = envelope.get("message")
            if envelope is body and type(nested) is str:
                message = nested
            if message is not None:
                if type(message) is not str or len(message) > _MAX_MESSAGE_CHARS:
                    return None
                if _CONFLICT_MESSAGE.search(message):
                    return None
                if any(pattern.fullmatch(message.strip()) for pattern in _MESSAGE_PATTERNS):
                    message_match = True
                # Claude's documented short signature is also explicit, but
                # only as the complete message of its generic error envelope.
                # https://platform.claude.com/docs/en/build-with-claude/context-windows
                if envelope.get("type") == "invalid_request_error" and re.fullmatch(
                    r"prompt is too long[.!]?", message.strip(), re.IGNORECASE,
                ):
                    message_match = True
    if code_match:
        return ContextLimitRejection(status, "structured_code")
    if message_match:
        return ContextLimitRejection(status, "message")
    return None
