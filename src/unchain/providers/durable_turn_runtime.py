"""Durable orchestration for one exact provider-wire turn.

This module is intentionally not wired into ``KernelLoop`` yet.  Only the
``enforce_test`` mode may send network requests; ``off`` and ``shadow`` leave
the current production provider path untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Callable

from unchain.durability import (
    is_durable_persistence_failure,
    mark_durable_persistence_failure,
)
from unchain.journal.models import OperationRef
from unchain.journal.provider_result import (
    BoundProviderTurnResultStore,
    ProviderTurnResultEnvelope,
    ProviderTurnResultIntegrityError,
    ProviderTurnResultReceipt,
    ProviderTurnResultReceiptLookup,
    recover_provider_turn_result,
)
from unchain.journal.provider_wire import RecoveredProviderWireAuthority
from unchain.kernel.types import ModelTurnResult
from unchain.run_bundle import ProviderCallReceipt
from unchain.retry import RetryConfig, RetriesExhaustedError
from unchain.retry.backoff import compute_delay_ms
from unchain.retry.classifier import extract_retry_after_ms

from .request_lease import (
    ProviderRequestDisposition,
    ProviderRequestLease,
    ProviderRequestLeaseConflict,
    ProviderRequestLeaseCoordinator,
    ProviderRequestLeasePort,
    ProviderRequestStatus,
    ProviderRequestSubject,
    ProviderTurnResultBinding,
)
from .physical_send import (
    ProviderPhysicalSendContext,
    provider_physical_ordinal,
)
from .wire_envelope import ProviderWireEnvelope, ProviderWireRoute
from .failure_diagnostic import ProviderFailureDiagnostic
from .uncertainty_diagnostic import ProviderUncertaintyDiagnostic


_LOGGER = logging.getLogger(__name__)


def _provider_attempt_completed_at() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class DurableProviderTurnMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"
    ENFORCE_TEST = "enforce_test"


class DurableProviderTurnStatus(StrEnum):
    BYPASSED = "bypassed"
    SHADOWED = "shadowed"
    COMPLETED = "completed"


class DurableProviderTurnError(RuntimeError):
    """Base error for durable provider-turn orchestration."""


@dataclass(frozen=True, slots=True)
class ProviderRetryWait:
    """Closed retry facts; the hook owns the interruptible wait."""

    attempt_failed: int
    next_attempt: int
    max_attempts: int
    delay_ms: int
    diagnostic: ProviderFailureDiagnostic | None


RetryWaitHook = Callable[[ProviderRetryWait, Callable[[float], None]], None]


class DurableProviderTurnUncertainError(DurableProviderTurnError):
    """A send may have happened, but no durable terminal result exists."""

    code = "durable_provider_turn_uncertain"

    def __init__(
        self,
        diagnostic: ProviderUncertaintyDiagnostic | ProviderFailureDiagnostic | None = None,
    ) -> None:
        # Compatibility with #386's public constructor is deliberately closed:
        # arbitrary provider messages or exception class names never become UI text.
        if diagnostic is None:
            diagnostic = ProviderUncertaintyDiagnostic("unrecorded_outcome", "recovery")
        if type(diagnostic) not in {ProviderUncertaintyDiagnostic, ProviderFailureDiagnostic}:
            raise TypeError("uncertain error requires an exact diagnostic")
        self.diagnostic = diagnostic
        self.__suppress_context__ = True
        if type(diagnostic) is ProviderFailureDiagnostic:
            super().__init__(self.code + "; " + diagnostic.summary())
        else:
            super().__init__("Provider request outcome is uncertain: " + diagnostic.summary())


class DurableProviderTurnTerminalError(DurableProviderTurnError):
    """A durable terminal provider failure was recovered without its exception."""

    code = "durable_provider_turn_terminal_failed"

    def __init__(
        self,
        classification: str,
        diagnostic: ProviderFailureDiagnostic | None = None,
    ) -> None:
        self.classification = classification
        self.diagnostic = diagnostic
        self.__suppress_context__ = True
        if (
            classification == "transient"
            and diagnostic is not None
            and diagnostic.http_status == 503
        ):
            super().__init__(
                "Provider temporarily busy (HTTP 503). Please try again later."
            )
        else:
            detail = f"; {diagnostic.summary()}" if diagnostic is not None else ""
            super().__init__(f"{self.code}:{classification}{detail}")


class ExactProviderRouteFailureKind(StrEnum):
    TRANSIENT_RETRY_SAFE = "transient_retry_safe"
    PREVIOUS_RESPONSE_FALLBACK = "previous_response_fallback"
    TERMINAL = "terminal"


class ExactProviderRouteFailure(RuntimeError):
    """A transport-classified failure that is safe to persist as terminal."""

    code = "exact_provider_route_failure"

    def __init__(
        self,
        kind: ExactProviderRouteFailureKind,
        original: BaseException,
    ) -> None:
        try:
            resolved_kind = ExactProviderRouteFailureKind(kind)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "exact provider route failure kind is unsupported"
            ) from exc
        if not isinstance(original, BaseException):
            raise TypeError("original must be an exception")
        self.kind = resolved_kind
        self.original = original
        self.__suppress_context__ = True
        super().__init__(self.code)


class ExactProviderRouteTransport(ABC):
    """Single-send transport that buffers output and never performs fallback."""

    @abstractmethod
    def send(
        self,
        *,
        envelope: ProviderWireEnvelope,
        route: ProviderWireRoute,
        retry_ordinal: int,
    ) -> ModelTurnResult:
        """Send exactly one route once, without emitting user-visible callbacks."""


@dataclass(frozen=True, slots=True)
class DurableProviderTurnOutcome:
    status: DurableProviderTurnStatus
    result: ModelTurnResult | None = None
    recovered: bool = False
    run_receipt: ProviderCallReceipt | None = None

    def __post_init__(self) -> None:
        try:
            status = DurableProviderTurnStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("durable provider turn status is unsupported") from exc
        object.__setattr__(self, "status", status)
        if self.result is not None and type(self.result) is not ModelTurnResult:
            raise TypeError("result must be an exact ModelTurnResult or null")
        if type(self.recovered) is not bool:
            raise TypeError("recovered must be an exact boolean")
        if self.run_receipt is not None and type(self.run_receipt) is not ProviderCallReceipt:
            raise TypeError("run_receipt must be an exact ProviderCallReceipt or null")
        if status is DurableProviderTurnStatus.COMPLETED:
            if self.result is None:
                raise ValueError("completed provider turn requires a result")
            if self.recovered and self.run_receipt is not None:
                raise ValueError("recovered provider turn cannot invent a live receipt")
        elif self.result is not None or self.recovered or self.run_receipt is not None:
            raise ValueError("closed provider turn modes cannot contain a result")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _subject_digest(subject: ProviderRequestSubject) -> str:
    return hashlib.sha256(_canonical_bytes(subject.to_dict())).hexdigest()


def _operation(
    *,
    subject: ProviderRequestSubject,
    phase: str,
    detail: object,
) -> OperationRef:
    payload = {
        "schema": "unchain.durable_provider_turn_operation.v1",
        "phase": phase,
        "subject": subject.to_dict(),
        "detail": detail,
    }
    payload_sha256 = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return OperationRef(
        f"provider-turn-{phase}-{_subject_digest(subject)[:32]}",
        payload_sha256,
    )


def _result_binding(receipt: ProviderTurnResultReceipt) -> ProviderTurnResultBinding:
    return ProviderTurnResultBinding(
        route_sha256=receipt.envelope.route_sha256,
        result_sha256=receipt.envelope.result_sha256,
        artifact=receipt.artifact,
        cursor=receipt.cursor,
    )


class DurableProviderTurnRuntime:
    """Own retries, fallback, result persistence, and restart recovery."""

    def __init__(
        self,
        *,
        mode: DurableProviderTurnMode,
        store: BoundProviderTurnResultStore,
        transport: ExactProviderRouteTransport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            resolved_mode = DurableProviderTurnMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("durable provider turn mode is unsupported") from exc
        if not isinstance(store, BoundProviderTurnResultStore) or not isinstance(
            store, ProviderRequestLeasePort
        ):
            raise TypeError(
                "store must provide provider result persistence and request leases"
            )
        if not isinstance(transport, ExactProviderRouteTransport):
            raise TypeError("transport must be an ExactProviderRouteTransport")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._mode = resolved_mode
        self._store = store
        self._transport = transport
        self._sleep = sleep
        self._coordinator = ProviderRequestLeaseCoordinator(store)

    @staticmethod
    def _validate_retry_config(config: RetryConfig) -> None:
        if type(config) is not RetryConfig:
            raise TypeError("retry_config must be an exact RetryConfig")
        if type(config.max_retries) is not int or config.max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if type(config.base_delay_ms) is not int or config.base_delay_ms < 0:
            raise ValueError("base_delay_ms must be a non-negative integer")
        if type(config.max_delay_ms) is not int or config.max_delay_ms < 0:
            raise ValueError("max_delay_ms must be a non-negative integer")
        if type(config.jitter_ratio) not in {int, float} or config.jitter_ratio < 0:
            raise ValueError("jitter_ratio must be non-negative")

    def _raise_durable(self, error: BaseException) -> None:
        boundary = mark_durable_persistence_failure(error)
        raise boundary from None

    def _recover_lease(self, subject: ProviderRequestSubject):
        try:
            return self._coordinator.recover(subject)
        except BaseException as exc:
            self._raise_durable(exc)

    def _lookup_result(
        self,
        *,
        subject: ProviderRequestSubject,
        route_sha256: str,
    ) -> ProviderTurnResultReceipt | None:
        try:
            lookup = self._store.lookup_provider_turn_result_receipts(subject=subject)
            if (
                type(lookup) is not ProviderTurnResultReceiptLookup
                or lookup.subject != subject
                or lookup.overflow
                or len(lookup.events) > 1
            ):
                raise ProviderTurnResultIntegrityError(
                    "provider result receipt evidence is conflicting"
                )
            if not lookup.events:
                return None
            return recover_provider_turn_result(
                self._store,
                subject=subject,
                expected_route_sha256=route_sha256,
            )
        except BaseException as exc:
            self._raise_durable(exc)

    def _inspect_subject(
        self,
        *,
        subject: ProviderRequestSubject,
        route_sha256: str,
    ) -> tuple[ProviderRequestLease | None, ProviderTurnResultReceipt | None]:
        recovery = self._recover_lease(subject)
        receipt = self._lookup_result(
            subject=subject,
            route_sha256=route_sha256,
        )
        lease = recovery.lease
        if recovery.disposition is ProviderRequestDisposition.NOT_STARTED:
            if lease is not None or receipt is not None:
                self._raise_durable(
                    ProviderTurnResultIntegrityError(
                        "provider result exists without a request lease"
                    )
                )
            return None, None
        if lease is None:
            self._raise_durable(
                ProviderTurnResultIntegrityError(
                    "provider request recovery lost its durable lease"
                )
            )
        if lease.route_sha256 != route_sha256:
            self._raise_durable(
                ProviderTurnResultIntegrityError(
                    "provider request route digest changed"
                )
            )
        if receipt is not None and lease.status is ProviderRequestStatus.FAILED:
            self._raise_durable(
                ProviderTurnResultIntegrityError(
                    "failed provider request also has a durable result"
                )
            )
        if receipt is None and lease.status is ProviderRequestStatus.COMPLETED:
            self._raise_durable(
                ProviderTurnResultIntegrityError(
                    "completed provider request lost its durable result"
                )
            )
        return lease, receipt

    def _claim(
        self,
        *,
        subject: ProviderRequestSubject,
        route: ProviderWireRoute,
        previous: ProviderRequestLease | None,
        fallback_parent: ProviderRequestLease | None,
    ) -> tuple[ProviderRequestLease, bool]:
        operation = _operation(
            subject=subject,
            phase="claim",
            detail={"route_sha256": route.route_sha256},
        )
        try:
            if subject.retry_ordinal > 0:
                if previous is None:
                    raise ProviderTurnResultIntegrityError(
                        "provider retry lost its predecessor lease"
                    )
                lease = self._coordinator.claim_retry(
                    previous,
                    operation=operation,
                )
            elif subject.route == "openai_previous_response_fallback":
                if fallback_parent is None:
                    raise ProviderTurnResultIntegrityError(
                        "provider fallback lost its primary lease"
                    )
                lease = self._coordinator.claim_openai_fallback(
                    fallback_parent,
                    fallback_route_sha256=route.route_sha256,
                    operation=operation,
                )
            else:
                lease = self._coordinator.claim_initial(
                    attempt=subject.attempt,
                    iteration=subject.iteration,
                    envelope_sha256=subject.envelope_sha256,
                    route=subject.route,
                    route_sha256=route.route_sha256,
                    operation=operation,
                )
            return lease, True
        except ProviderRequestLeaseConflict:
            lease, receipt = self._inspect_subject(
                subject=subject,
                route_sha256=route.route_sha256,
            )
            if lease is None or receipt is not None:
                if lease is not None and receipt is not None:
                    return lease, False
                self._raise_durable(
                    ProviderTurnResultIntegrityError(
                        "provider request claim conflicted without a durable winner"
                    )
                )
            return lease, False
        except BaseException as exc:
            self._raise_durable(exc)

    def _record_failure(
        self,
        lease: ProviderRequestLease,
        *,
        classification: str,
        retryable: bool,
        diagnostic: ProviderFailureDiagnostic | None = None,
    ) -> ProviderRequestLease:
        operation = _operation(
            subject=lease.subject,
            phase="failure",
            detail={
                "classification": classification,
                "retryable": retryable,
                "visible_output": False,
                **(
                    {"failure_diagnostic": diagnostic.to_dict()}
                    if diagnostic is not None else {}
                ),
            },
        )
        try:
            return self._coordinator.record_failure(
                lease,
                classification=classification,
                retryable=retryable,
                visible_output=False,
                operation=operation,
                failure_diagnostic=diagnostic,
            )
        except BaseException as exc:
            self._raise_durable(exc)

    def _uncertain_error(
        self,
        lease: ProviderRequestLease,
        diagnostic: ProviderUncertaintyDiagnostic,
    ) -> DurableProviderTurnUncertainError:
        # Observability is best effort. Failure to annotate cannot replace the
        # primary failure, seal a result, or authorize another physical send.
        try:
            self._coordinator.record_uncertainty(
                lease, diagnostic=diagnostic,
                operation=_operation(
                    subject=lease.subject, phase="uncertainty",
                    detail={"uncertainty_diagnostic": diagnostic.to_dict()},
                ),
            )
        except Exception:
            _LOGGER.warning("Provider uncertainty diagnostic could not be persisted")
        return DurableProviderTurnUncertainError(diagnostic)

    @staticmethod
    def _notify_uncertain_after_send(after_send, context, classification: str) -> None:
        if after_send is None:
            return
        try:
            after_send(context, _provider_attempt_completed_at(), "uncertain", classification)
        except Exception as exc:
            if is_durable_persistence_failure(exc) or getattr(exc, "code", None) == "execution_cancelled":
                raise exc from None
            _LOGGER.warning("Provider failure observer could not record uncertainty")

    def _result_callback(self, lease, callback, *args):
        try:
            return callback(*args)
        except BaseException as exc:
            if is_durable_persistence_failure(exc) or getattr(exc, "code", None) == "execution_cancelled":
                raise exc from None
            raise self._uncertain_error(
                lease,
                ProviderUncertaintyDiagnostic.from_exception(exc, phase="result_processing"),
            ) from None

    def _complete(
        self,
        lease: ProviderRequestLease,
        receipt: ProviderTurnResultReceipt,
    ) -> ProviderRequestLease:
        binding = _result_binding(receipt)
        operation = _operation(
            subject=lease.subject,
            phase="completion",
            detail={
                "result_sha256": binding.result_sha256,
                "artifact": binding.artifact.to_dict(),
                "cursor": binding.cursor.to_dict(),
            },
        )
        try:
            return self._coordinator.record_completed_result(
                lease,
                result_binding=binding,
                visible_output=receipt.envelope.visible_output,
                operation=operation,
            )
        except BaseException as exc:
            self._raise_durable(exc)

    def _recover_or_complete(
        self,
        *,
        lease: ProviderRequestLease,
        receipt: ProviderTurnResultReceipt,
    ) -> DurableProviderTurnOutcome:
        binding = _result_binding(receipt)
        if lease.status is ProviderRequestStatus.STARTED:
            lease = self._complete(lease, receipt)
        if (
            lease.status is not ProviderRequestStatus.COMPLETED
            or lease.result_binding != binding
            or lease.visible_output != receipt.envelope.visible_output
        ):
            self._raise_durable(
                ProviderTurnResultIntegrityError(
                    "completed request lease changed its provider result binding"
                )
            )
        return DurableProviderTurnOutcome(
            status=DurableProviderTurnStatus.COMPLETED,
            result=receipt.envelope.to_model_turn_result(),
            recovered=True,
        )

    def _persist_result(
        self,
        *,
        lease: ProviderRequestLease,
        result: ModelTurnResult,
        run_receipt: ProviderCallReceipt | None = None,
    ) -> DurableProviderTurnOutcome:
        try:
            envelope = ProviderTurnResultEnvelope.from_model_turn_result(
                subject=lease.subject,
                route_sha256=lease.route_sha256,
                visible_output=True,
                result=result,
            )
        except BaseException:
            raise self._uncertain_error(
                lease, ProviderUncertaintyDiagnostic("invalid_result", "result_processing"),
            ) from None
        artifact_operation = _operation(
            subject=lease.subject,
            phase="result-artifact",
            detail={"result_sha256": envelope.result_sha256},
        )
        event_operation = _operation(
            subject=lease.subject,
            phase="result-event",
            detail={
                "result_sha256": envelope.result_sha256,
                "route_sha256": envelope.route_sha256,
            },
        )
        event_id = f"provider-turn-result-{_subject_digest(lease.subject)}"
        try:
            receipt = self._store.persist_provider_turn_result_cas(
                started_lease=lease,
                envelope=envelope,
                artifact_operation=artifact_operation,
                event_operation=event_operation,
                event_id=event_id,
                provider_call_receipt=run_receipt,
            )
        except BaseException as exc:
            self._raise_durable(exc)
        self._complete(lease, receipt)
        return DurableProviderTurnOutcome(
            status=DurableProviderTurnStatus.COMPLETED,
            # The durable v1 envelope remains intentionally unchanged.  A live
            # send may nevertheless carry newer ephemeral accounting fields
            # (for example canonical provider usage) to the RunBundle ledger.
            # Cold recovery continues through ``_recover_or_complete`` and is
            # therefore explicitly legacy-partial when v1 lacks those fields.
            result=result,
            recovered=False,
            run_receipt=run_receipt,
        )

    @staticmethod
    def _route(envelope: ProviderWireEnvelope, name: str) -> ProviderWireRoute:
        for route in envelope.routes:
            if route.name == name:
                return route
        raise DurableProviderTurnError(f"provider wire route {name!r} is unavailable")

    def _delay(
        self,
        *,
        ordinal: int,
        retry_config: RetryConfig,
        error: BaseException | None,
        before_sleep: Callable[[int], None] | None = None,
        retry_wait: RetryWaitHook | None = None,
        diagnostic: ProviderFailureDiagnostic | None = None,
        retry_limit: int | None = None,
    ) -> None:
        delay_ms = compute_delay_ms(
            attempt=ordinal,
            config=retry_config,
            retry_after_ms=(
                extract_retry_after_ms(error) if error is not None else None
            ),
        )
        if retry_wait is not None:
            retry_wait(
                ProviderRetryWait(
                    attempt_failed=ordinal,
                    next_attempt=ordinal + 1,
                    max_attempts=(retry_config.max_retries if retry_limit is None else retry_limit) + 1,
                    delay_ms=int(delay_ms),
                    diagnostic=diagnostic,
                ),
                self._sleep,
            )
            return
        if before_sleep is not None:
            before_sleep(delay_ms)
        self._sleep(delay_ms / 1000.0)

    def _execute_route(
        self,
        *,
        authority: RecoveredProviderWireAuthority,
        route_name: str,
        retry_config: RetryConfig,
        before_send: Callable[[ProviderPhysicalSendContext], None] | None,
        after_send: Callable[
            [ProviderPhysicalSendContext, str, str, str], None
        ] | None,
        build_run_receipt: Callable[
            [
                ProviderPhysicalSendContext,
                str,
                str,
                str,
                ModelTurnResult | None,
            ],
            ProviderCallReceipt,
        ] | None,
        on_retry: Callable[[int, int, int], None] | None,
        fallback_parent: ProviderRequestLease | None = None,
        retry_wait: RetryWaitHook | None = None,
    ) -> DurableProviderTurnOutcome:
        envelope = authority.envelope
        route = self._route(envelope, route_name)
        if route_name == "openai_previous_response_fallback":
            if (
                fallback_parent is None
                or fallback_parent.subject.route != "primary"
                or fallback_parent.subject.retry_ordinal != 0
                or fallback_parent.status is not ProviderRequestStatus.FAILED
                or fallback_parent.classification != "previous_response_fallback"
            ):
                self._raise_durable(
                    ProviderTurnResultIntegrityError(
                        "provider fallback requires the failed primary ordinal zero"
                    )
                )
        ordinal = 0
        previous: ProviderRequestLease | None = None
        while True:
            subject = ProviderRequestSubject(
                attempt=envelope.attempt,
                iteration=envelope.iteration,
                envelope_sha256=envelope.envelope_sha256,
                route=route.name,
                retry_ordinal=ordinal,
            )
            send_context = ProviderPhysicalSendContext(
                subject=subject,
                route=route,
                physical_ordinal=provider_physical_ordinal(subject),
            )
            lease, receipt = self._inspect_subject(
                subject=subject,
                route_sha256=route.route_sha256,
            )
            claimed_now = False
            if lease is None:
                # Older #386 records may authorize more 503 retries than the
                # current limit. Inspect existing outcomes first, then bound
                # only a new claim/send; never discard a recorded result.
                previous_diagnostic = previous.failure_diagnostic if previous is not None else None
                retry_limit = (
                    min(retry_config.max_retries, 2)
                    if envelope.provider == "gemini"
                    and previous_diagnostic is not None
                    and previous_diagnostic.http_status == 503
                    else retry_config.max_retries
                )
                if ordinal > retry_limit:
                    raise DurableProviderTurnTerminalError("transient", previous_diagnostic)
                lease, claimed_now = self._claim(
                    subject=subject,
                    route=route,
                    previous=previous,
                    fallback_parent=fallback_parent,
                )
                if not claimed_now:
                    lease, receipt = self._inspect_subject(
                        subject=subject,
                        route_sha256=route.route_sha256,
                    )
            if lease is None:
                self._raise_durable(
                    ProviderTurnResultIntegrityError(
                        "provider request claim did not create a lease"
                    )
                )
            if receipt is not None:
                return self._recover_or_complete(lease=lease, receipt=receipt)
            if lease.status is ProviderRequestStatus.COMPLETED:
                self._raise_durable(
                    ProviderTurnResultIntegrityError(
                        "completed provider request has no recovered result"
                    )
                )
            if lease.status is ProviderRequestStatus.STARTED and not claimed_now:
                raise DurableProviderTurnUncertainError(lease.uncertainty_diagnostic)
            if lease.status is ProviderRequestStatus.FAILED:
                if lease.classification == "previous_response_fallback":
                    self._route(
                        envelope,
                        "openai_previous_response_fallback",
                    )
                    return self._execute_route(
                        authority=authority,
                        route_name="openai_previous_response_fallback",
                        retry_config=retry_config,
                        before_send=before_send,
                        after_send=after_send,
                        build_run_receipt=build_run_receipt,
                        on_retry=on_retry,
                        fallback_parent=lease,
                        retry_wait=retry_wait,
                    )
                if not (
                    lease.classification == "transient"
                    and lease.retryable
                    and not lease.visible_output
                ):
                    raise DurableProviderTurnTerminalError(lease.classification, lease.failure_diagnostic)
                previous = lease
                ordinal += 1
                if ordinal <= retry_config.max_retries:
                    next_subject = ProviderRequestSubject(
                        attempt=envelope.attempt,
                        iteration=envelope.iteration,
                        envelope_sha256=envelope.envelope_sha256,
                        route=route.name,
                        retry_ordinal=ordinal,
                    )
                    next_lease, _ = self._inspect_subject(
                        subject=next_subject,
                        route_sha256=route.route_sha256,
                    )
                    if next_lease is None:
                        gemini_busy = (
                            envelope.provider == "gemini"
                            and lease.failure_diagnostic is not None
                            and lease.failure_diagnostic.http_status == 503
                        )
                        retry_limit = min(retry_config.max_retries, 2) if gemini_busy else retry_config.max_retries
                        if ordinal > retry_limit:
                            raise DurableProviderTurnTerminalError("transient", lease.failure_diagnostic)
                        self._delay(
                            ordinal=ordinal,
                            retry_config=retry_config,
                            error=None,
                            retry_wait=retry_wait,
                            diagnostic=lease.failure_diagnostic,
                            retry_limit=retry_limit,
                            before_sleep=(
                                lambda delay_ms: on_retry(
                                    ordinal, retry_limit, delay_ms
                                )
                                if gemini_busy and on_retry is not None
                                else None
                            ),
                        )
                continue

            if before_send is not None:
                try:
                    before_send(send_context)
                except BaseException:
                    raise self._uncertain_error(
                        lease, ProviderUncertaintyDiagnostic("before_send_failed", "before_send"),
                    ) from None
            try:
                result = self._transport.send(
                    envelope=envelope,
                    route=route,
                    retry_ordinal=ordinal,
                )
            except ExactProviderRouteFailure as exc:
                if after_send is not None:
                    after_send(
                        send_context,
                        _provider_attempt_completed_at(),
                        "failed",
                        exc.kind.value,
                    )
                if is_durable_persistence_failure(exc.original):
                    raise exc.original
                if exc.kind is ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE:
                    diagnostic = ProviderFailureDiagnostic.from_exception(exc.original)
                    gemini_busy = (
                        envelope.provider == "gemini"
                        and diagnostic is not None
                        and diagnostic.http_status == 503
                    )
                    # A busy model can take many seconds to reject each send.
                    # Keep the normal budget for other providers and failures.
                    retry_limit = (
                        min(retry_config.max_retries, 2)
                        if gemini_busy
                        else retry_config.max_retries
                    )
                    retry_budget_remaining = ordinal < retry_limit
                    failed = self._record_failure(
                        lease,
                        classification="transient",
                        retryable=retry_budget_remaining,
                        diagnostic=diagnostic,
                    )
                    if not retry_budget_remaining:
                        if gemini_busy:
                            raise DurableProviderTurnTerminalError(
                                "transient", diagnostic
                            ) from None
                        raise RetriesExhaustedError(
                            last_error=exc.original,
                            attempts=retry_limit,
                            detail=(f"{diagnostic.summary()} after {retry_limit} retries" if diagnostic is not None else ""),
                        ) from None
                    previous = failed
                    ordinal += 1
                    self._delay(
                        ordinal=ordinal,
                        retry_config=retry_config,
                        error=exc.original,
                        retry_wait=retry_wait,
                        diagnostic=diagnostic,
                        retry_limit=retry_limit,
                        before_sleep=(
                            lambda delay_ms: on_retry(
                                ordinal, retry_limit, delay_ms
                            )
                            if gemini_busy and on_retry is not None
                            else None
                        ),
                    )
                    continue
                if exc.kind is ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK:
                    failed = self._record_failure(
                        lease,
                        classification="previous_response_fallback",
                        retryable=True,
                    )
                    return self._execute_route(
                        authority=authority,
                        route_name="openai_previous_response_fallback",
                        retry_config=retry_config,
                        before_send=before_send,
                        after_send=after_send,
                        build_run_receipt=build_run_receipt,
                        on_retry=on_retry,
                        fallback_parent=failed,
                        retry_wait=retry_wait,
                    )
                diagnostic = ProviderFailureDiagnostic.from_exception(exc.original)
                self._record_failure(
                    lease,
                    classification="non_retryable",
                    retryable=False,
                    diagnostic=diagnostic,
                )
                raise DurableProviderTurnTerminalError(
                    "non_retryable", diagnostic
                ) from None
            except BaseException as exc:
                if is_durable_persistence_failure(exc):
                    raise
                error = self._uncertain_error(
                    lease, ProviderUncertaintyDiagnostic.from_exception(exc, phase="reading_response"),
                )
                self._notify_uncertain_after_send(after_send, send_context, "uncertain")
                raise error from None

            if type(result) is not ModelTurnResult:
                error = self._uncertain_error(
                    lease, ProviderUncertaintyDiagnostic("invalid_result", "result_processing"),
                )
                self._notify_uncertain_after_send(after_send, send_context, "invalid_result")
                raise error from None
            completed_at = _provider_attempt_completed_at()
            run_receipt = None
            if build_run_receipt is not None:
                run_receipt = self._result_callback(
                    lease,
                    build_run_receipt,
                    send_context,
                    completed_at,
                    "completed",
                    "success",
                    result,
                )
                if type(run_receipt) is not ProviderCallReceipt:
                    error = self._uncertain_error(
                        lease, ProviderUncertaintyDiagnostic("invalid_result", "result_processing"),
                    )
                    self._notify_uncertain_after_send(after_send, send_context, "invalid_result")
                    raise error from None
            if after_send is not None:
                self._result_callback(
                    lease,
                    after_send,
                    send_context,
                    completed_at,
                    "completed",
                    "success",
                )
            return self._persist_result(
                lease=lease,
                result=result,
                run_receipt=run_receipt,
            )

    def execute(
        self,
        *,
        authority: RecoveredProviderWireAuthority,
        retry_config: RetryConfig,
        before_send: Callable[[ProviderPhysicalSendContext], None] | None = None,
        after_send: Callable[
            [ProviderPhysicalSendContext, str, str, str], None
        ] | None = None,
        build_run_receipt: Callable[
            [
                ProviderPhysicalSendContext,
                str,
                str,
                str,
                ModelTurnResult | None,
            ],
            ProviderCallReceipt,
        ] | None = None,
        on_retry: Callable[[int, int, int], None] | None = None,
        retry_wait: RetryWaitHook | None = None,
    ) -> DurableProviderTurnOutcome:
        self._validate_retry_config(retry_config)
        if on_retry is not None and not callable(on_retry):
            raise TypeError("on_retry must be callable or null")
        if retry_wait is not None and not callable(retry_wait):
            raise TypeError("retry_wait must be callable or null")
        if type(authority) is not RecoveredProviderWireAuthority:
            raise TypeError("authority must be an exact RecoveredProviderWireAuthority")
        envelope = authority.envelope
        if type(envelope) is not ProviderWireEnvelope:
            raise TypeError("authority envelope must be an exact ProviderWireEnvelope")
        if self._store.execution_id != envelope.attempt.generation.execution_id:
            raise DurableProviderTurnError(
                "provider turn store crossed its execution scope"
            )
        envelope.verify_against_catalog(authority.catalog)

        if self._mode not in {
            DurableProviderTurnMode.ENFORCE,
            DurableProviderTurnMode.ENFORCE_TEST,
        }:
            route = self._route(envelope, "primary")
            subject = ProviderRequestSubject(
                attempt=envelope.attempt,
                iteration=envelope.iteration,
                envelope_sha256=envelope.envelope_sha256,
                route=route.name,
                retry_ordinal=0,
            )
            lease, receipt = self._inspect_subject(
                subject=subject,
                route_sha256=route.route_sha256,
            )
            if lease is not None or receipt is not None:
                raise DurableProviderTurnError(
                    "closed provider turn mode conflicts with durable evidence"
                )
            if self._mode is DurableProviderTurnMode.OFF:
                return DurableProviderTurnOutcome(
                    status=DurableProviderTurnStatus.BYPASSED
                )
            return DurableProviderTurnOutcome(
                status=DurableProviderTurnStatus.SHADOWED
            )
        return self._execute_route(
            authority=authority,
            route_name="primary",
            retry_config=retry_config,
            before_send=before_send,
            after_send=after_send,
            build_run_receipt=build_run_receipt,
            on_retry=on_retry,
            retry_wait=retry_wait,
        )


__all__ = [
    "ProviderRetryWait",
    "RetryWaitHook",
    "DurableProviderTurnError",
    "DurableProviderTurnMode",
    "DurableProviderTurnOutcome",
    "DurableProviderTurnRuntime",
    "DurableProviderTurnStatus",
    "DurableProviderTurnTerminalError",
    "DurableProviderTurnUncertainError",
    "ExactProviderRouteFailure",
    "ExactProviderRouteFailureKind",
    "ExactProviderRouteTransport",
]
