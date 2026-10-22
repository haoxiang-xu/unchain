"""Exception classification for retry decisions."""
from __future__ import annotations

from typing import Optional

import httpx

from ..durability import is_durable_persistence_failure

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset(
    {408, 409, 429, 529} | set(range(500, 600))
)


def is_retryable(error: BaseException) -> bool:
    """Return True if the error is a transient network / server failure."""

    if is_durable_persistence_failure(error):
        return False

    if isinstance(
        error,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ),
    ):
        return True

    if isinstance(error, httpx.HTTPStatusError):
        return error.response.status_code in RETRYABLE_STATUS_CODES

    try:
        import anthropic  # type: ignore

        if isinstance(error, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return True
        if isinstance(error, anthropic.APIStatusError):
            status = getattr(error, "status_code", None)
            return status in RETRYABLE_STATUS_CODES
    except ImportError:
        # Anthropic is optional; continue with the remaining provider and generic classifications.
        pass

    try:
        import openai  # type: ignore

        if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(error, openai.APIStatusError):
            status = getattr(error, "status_code", None)
            return status in RETRYABLE_STATUS_CODES
    except ImportError:
        # OpenAI is optional; an unrecognized exception remains non-retryable below.
        pass

    return False


def extract_retry_after_ms(error: BaseException) -> Optional[int]:
    """If the error carries a Retry-After header (integer seconds), return milliseconds."""

    response = getattr(error, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        value = headers.get("retry-after")
    except AttributeError:
        return None
    if not value:
        return None
    try:
        seconds = int(value)
    except (ValueError, TypeError):
        return None
    if seconds < 0:
        return None
    return seconds * 1000
