"""Core retry loop."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable, Optional, TypeVar

from .backoff import compute_delay_ms
from .classifier import extract_retry_after_ms, is_retryable
from .types import RetryAttempt, RetryConfig, RetryContext, RetriesExhaustedError

T = TypeVar("T")


def _completed_at() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _failure_outcome(error: BaseException) -> str:
    """Classify only whether a provider error response was observed.

    Connection, timeout, protocol, and adapter errors remain ``uncertain``:
    the generic retry layer cannot prove whether the remote side processed the
    request.  Errors carrying an HTTP response are closed ``failed`` calls.
    """

    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return "failed" if isinstance(status_code, int) else "uncertain"


def execute_with_retry(
    operation: Callable[[], T],
    config: RetryConfig,
    context: RetryContext,
    *,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Optional[Callable[[], bool]] = None,
    before_attempt: Optional[Callable[[int], None]] = None,
    after_attempt: Optional[
        Callable[[int, str, str, str], None]
    ] = None,
) -> T:
    """Run `operation` with retry on transient errors.

    Raises:
        RetriesExhaustedError: when `config.max_retries` attempts all fail.
        The original exception: when the error is non-retryable, or when
            `should_stop()` returns True before the next retry.
    """

    last_error: Optional[BaseException] = None

    for attempt_number in range(config.max_retries + 1):
        # Admission/lease renewal happens before the physical operation starts.
        # If it fails there was no provider attempt, so no after_attempt fact
        # may be emitted for accounting.
        if before_attempt is not None:
            before_attempt(attempt_number)
        try:
            result = operation()
        except BaseException as exc:  # noqa: BLE001
            # Account for every started attempt, including interruption;
            # terminal exceptions are re-raised unchanged below.
            retryable = is_retryable(exc)
            if after_attempt is not None:
                after_attempt(
                    attempt_number,
                    _completed_at(),
                    _failure_outcome(exc),
                    "retryable" if retryable else "terminal",
                )
            if not retryable:
                raise
            last_error = exc

            if attempt_number >= config.max_retries:
                break

            if should_stop is not None and should_stop():
                raise

            retry_index = attempt_number + 1
            delay_ms = compute_delay_ms(
                attempt=retry_index,
                config=config,
                retry_after_ms=extract_retry_after_ms(exc),
            )

            if context.on_retry is not None:
                context.on_retry(
                    RetryAttempt(
                        attempt=retry_index,
                        error=exc,
                        delay_ms=delay_ms,
                        max_retries=config.max_retries,
                    )
                )

            sleep(delay_ms / 1000.0)
        else:
            if after_attempt is not None:
                after_attempt(
                    attempt_number,
                    _completed_at(),
                    "completed",
                    "success",
                )
            return result

    assert last_error is not None
    raise RetriesExhaustedError(
        last_error=last_error,
        attempts=config.max_retries,
    ) from None
