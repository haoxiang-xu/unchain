"""Attempt-scoped host execution for one exact durable provider turn."""

from __future__ import annotations

import time
import hashlib
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone

from unchain.journal import AttemptRef, GenerationRef
from unchain.journal.provider_wire import BoundProviderWireStore
from unchain.kernel.types import ModelTurnResult
from unchain.run_bundle import ProviderCallReceipt
from unchain.providers.anthropic import AnthropicModelIO
from unchain.providers.base import ModelTurnRequest
from unchain.providers.durable_turn_runtime import (
    DurableProviderTurnMode,
    DurableProviderTurnRuntime,
    DurableProviderTurnStatus,
    ExactProviderRouteTransport,
)
from unchain.providers.exact_route_transport import (
    GeminiExactRouteTransport,
    AnthropicExactRouteTransport,
    HyperspaceExactRouteTransport,
    OllamaExactRouteTransport,
    OpenAIExactRouteTransport,
)
from unchain.providers.ollama import OllamaModelIO
from unchain.providers.openai import OpenAIModelIO
from unchain.providers.physical_send import ProviderPhysicalSendContext
from unchain.retry import RetryConfig

from .provider_toolkit import ProviderToolkitAuthorityAdapter
from .provider_turns import ProviderTurnAuthorityService


class ContextProviderTurnExecutionError(RuntimeError):
    """The host could not resolve an exact provider transport."""


_OFFICIAL_TRANSPORT_TARGET = (
    "unchain.exact_provider_route_transport.v1|"
    "openai.responses.create|anthropic.messages.stream|"
    "hyperspace.anthropic.messages.stream|ollama.chat"
)


def official_provider_transport_target_sha256() -> str:
    """Return the pinned identity of Unchain's official exact route family."""

    return hashlib.sha256(_OFFICIAL_TRANSPORT_TARGET.encode("utf-8")).hexdigest()


def _exact_transport(
    *,
    model_io: object,
    catalog,
    request: ModelTurnRequest,
) -> ExactProviderRouteTransport:
    provider = getattr(model_io, "provider", None)
    common = {
        "model_io": model_io,
        "catalog": catalog,
        "callback": request.callback,
        "run_id": request.run_id,
        "emit_stream": request.emit_stream,
    }
    from unchain.providers.gemini import GeminiModelIO

    if provider == "gemini" and type(model_io) is GeminiModelIO:
        return GeminiExactRouteTransport(**common)
    if provider == "openai" and type(model_io) is OpenAIModelIO:
        return OpenAIExactRouteTransport(**common)
    if provider == "anthropic" and type(model_io) is AnthropicModelIO:
        return AnthropicExactRouteTransport(**common)
    if provider == "hyperspace":
        return HyperspaceExactRouteTransport(**common)
    if provider == "ollama" and type(model_io) is OllamaModelIO:
        return OllamaExactRouteTransport(**common)
    raise ContextProviderTurnExecutionError(
        "model_io does not have an official exact provider transport"
    )


class ContextProviderTurnExecutionService:
    """Compose durable authority, transport, lease, and result persistence.

    ``off`` is a read-only legacy fallthrough unless prior enforce evidence
    exists. ``shadow`` persists only the request authority. ``enforce_test``
    is the sole mode allowed to send through the exact transport.
    """

    def __init__(
        self,
        *,
        attempt: AttemptRef,
        store: BoundProviderWireStore,
        mode: DurableProviderTurnMode,
        transport_target_sha256: str,
        sleep: Callable[[float], None] = time.sleep,
        toolkit_adapter: ProviderToolkitAuthorityAdapter | None = None,
    ) -> None:
        try:
            resolved_mode = DurableProviderTurnMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("durable provider turn mode is unsupported") from exc
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._attempt = AttemptRef.from_dict(attempt.to_dict())
        self._store = store
        self._mode = resolved_mode
        self._sleep = sleep
        self._transport_target_sha256 = transport_target_sha256
        resolved_toolkit_adapter = (
            toolkit_adapter
            if toolkit_adapter is not None
            else ProviderToolkitAuthorityAdapter()
        )
        if type(resolved_toolkit_adapter) is not ProviderToolkitAuthorityAdapter:
            raise TypeError(
                "toolkit_adapter must be the official "
                "ProviderToolkitAuthorityAdapter or null"
            )
        self._toolkit_adapter = resolved_toolkit_adapter
        self._authority_service = ProviderTurnAuthorityService(
            attempt=self._attempt,
            store=store,
            transport_target_sha256=transport_target_sha256,
            toolkit_adapter=resolved_toolkit_adapter,
        )

    @property
    def attempt(self) -> AttemptRef:
        return AttemptRef.from_dict(self._attempt.to_dict())

    @property
    def store(self) -> BoundProviderWireStore:
        return self._store

    @property
    def mode(self) -> DurableProviderTurnMode:
        return self._mode

    def fetch_prepared(
        self,
        *,
        model_io: object,
        request: ModelTurnRequest,
        retry_config: RetryConfig,
        before_attempt: Callable[[int], None] | None = None,
        after_attempt: Callable[[int, str, str, str], None] | None = None,
        run_receipt_factory: Callable[
            [int, str, str, str, str, ModelTurnResult | None], ProviderCallReceipt
        ] | None = None,
        run_receipt_observed: Callable[[ProviderCallReceipt], None] | None = None,
        occurrence_sha256: str | None = None,
    ) -> ModelTurnResult | None:
        if type(request) is not ModelTurnRequest:
            raise TypeError("request must be an exact ModelTurnRequest")
        if before_attempt is not None and not callable(before_attempt):
            raise TypeError("before_attempt must be callable or null")
        if after_attempt is not None and not callable(after_attempt):
            raise TypeError("after_attempt must be callable or null")
        if run_receipt_factory is not None and not callable(run_receipt_factory):
            raise TypeError("run_receipt_factory must be callable or null")
        if run_receipt_observed is not None and not callable(run_receipt_observed):
            raise TypeError("run_receipt_observed must be callable or null")

        authority_service = self._authority_service
        effective_request = request
        if occurrence_sha256 is not None:
            if (
                type(occurrence_sha256) is not str
                or len(occurrence_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in occurrence_sha256
                )
            ):
                raise ValueError(
                    "occurrence_sha256 must be a lowercase SHA-256 digest or null"
                )
            occurrence_attempt = AttemptRef(
                generation=GenerationRef(
                    self._attempt.generation.execution_id,
                    f"provider-call-{occurrence_sha256}",
                ),
                attempt_id=self._attempt.attempt_id,
            )
            authority_service = ProviderTurnAuthorityService(
                attempt=occurrence_attempt,
                store=self._store,
                transport_target_sha256=self._transport_target_sha256,
                toolkit_adapter=self._toolkit_adapter,
            )
            # Auxiliary calls get a full-digest generation namespace while
            # preserving the owner attempt and its real iteration.  The
            # durable subject is therefore unique without weakening the exact
            # accounting-identity checks used by the atomic result CAS.

        if self._mode is DurableProviderTurnMode.OFF:
            authority = authority_service.recover_existing(
                model_io=model_io,
                request=effective_request,
            )
            if authority is None:
                return None
        else:
            authority = authority_service.prepare(
                model_io=model_io,
                request=effective_request,
            )

        transport = _exact_transport(
            model_io=model_io,
            catalog=authority.catalog,
            request=effective_request,
        )
        runtime = DurableProviderTurnRuntime(
            mode=self._mode,
            store=self._store,
            transport=transport,
            sleep=self._sleep,
        )
        send_number = 0
        active_send_context: ProviderPhysicalSendContext | None = None
        active_process_send_number: int | None = None
        active_started_at: str | None = None

        def accounting_receipt(
            send_context: ProviderPhysicalSendContext,
            started_at: str,
            completed_at: str,
            outcome: str,
            classification: str,
            result: ModelTurnResult | None,
        ) -> ProviderCallReceipt:
            if run_receipt_factory is None:
                raise ContextProviderTurnExecutionError(
                    "provider accounting receipt factory is unavailable"
                )
            base_receipt = run_receipt_factory(
                send_context.physical_ordinal,
                started_at,
                completed_at,
                outcome,
                classification,
                result,
            )
            if type(base_receipt) is not ProviderCallReceipt:
                raise TypeError(
                    "run_receipt_factory must return an exact ProviderCallReceipt"
                )
            identity = base_receipt.identity
            subject = send_context.subject
            if (
                identity.execution_id
                != subject.attempt.generation.execution_id
                or identity.attempt_id != subject.attempt.attempt_id
                or identity.iteration != subject.iteration
                or identity.retry_ordinal != send_context.physical_ordinal
            ):
                raise ContextProviderTurnExecutionError(
                    "provider accounting receipt changed the physical send subject"
                )
            from .composition import enrich_provider_call_receipt

            return enrich_provider_call_receipt(
                receipt=base_receipt,
                manifest=request.internal_context_composition_v1,
                envelope=authority.envelope,
                send_context=send_context,
            )

        def before_send(send_context: ProviderPhysicalSendContext) -> None:
            nonlocal active_send_context, active_process_send_number
            nonlocal active_started_at, send_number
            if type(send_context) is not ProviderPhysicalSendContext:
                raise TypeError(
                    "send_context must be an exact ProviderPhysicalSendContext"
                )
            if active_send_context is not None:
                raise ContextProviderTurnExecutionError(
                    "provider attempt completion was not observed"
                )
            active_send_context = send_context
            active_process_send_number = send_number
            active_started_at = (
                datetime.now(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            if before_attempt is not None:
                before_attempt(send_context.physical_ordinal)
            send_number += 1

        def after_send(
            send_context: ProviderPhysicalSendContext,
            completed_at: str,
            outcome: str,
            classification: str,
        ) -> None:
            nonlocal active_send_context, active_process_send_number
            nonlocal active_started_at
            observed_context = active_send_context
            started_at = active_started_at
            if (
                observed_context is None
                or observed_context != send_context
                or active_process_send_number is None
                or started_at is None
            ):
                raise ContextProviderTurnExecutionError(
                    "provider attempt completed without a matching start"
                )
            active_send_context = None
            active_process_send_number = None
            active_started_at = None
            if outcome != "completed" and run_receipt_factory is not None:
                run_receipt = accounting_receipt(
                    send_context,
                    started_at,
                    completed_at,
                    outcome,
                    classification,
                    None,
                )
                append_receipt = getattr(self._store, "append_receipt", None)
                if not callable(append_receipt):
                    raise ContextProviderTurnExecutionError(
                        "durable provider store lacks the accounting ledger"
                    )
                durable = append_receipt(run_receipt)
                if durable != run_receipt:
                    raise ContextProviderTurnExecutionError(
                        "durable provider store changed the accounting receipt"
                    )
                if run_receipt_observed is not None:
                    run_receipt_observed(run_receipt)
            if after_attempt is not None:
                after_attempt(
                    send_context.physical_ordinal,
                    completed_at,
                    outcome,
                    classification,
                )

        def build_run_receipt(
            send_context: ProviderPhysicalSendContext,
            completed_at: str,
            outcome: str,
            classification: str,
            result: ModelTurnResult | None,
        ) -> ProviderCallReceipt:
            observed_context = active_send_context
            started_at = active_started_at
            if (
                run_receipt_factory is None
                or observed_context is None
                or observed_context != send_context
                or started_at is None
            ):
                raise ContextProviderTurnExecutionError(
                    "provider accounting receipt has no matching live send"
                )
            return accounting_receipt(
                send_context,
                started_at,
                completed_at,
                outcome,
                classification,
                result,
            )

        try:
            outcome = runtime.execute(
                authority=authority,
                retry_config=retry_config,
                before_send=before_send,
                after_send=after_send,
                build_run_receipt=(
                    build_run_receipt if run_receipt_factory is not None else None
                ),
            )
        except BaseException:
            transport.discard_buffered_events()
            raise

        if outcome.status is not DurableProviderTurnStatus.COMPLETED:
            transport.discard_buffered_events()
            return None
        if outcome.result is None:
            transport.discard_buffered_events()
            raise ContextProviderTurnExecutionError(
                "completed provider turn lost its durable result"
            )
        if outcome.recovered:
            transport.discard_buffered_events()
        else:
            transport.release_buffered_events()
        if outcome.run_receipt is not None:
            if run_receipt_observed is not None:
                run_receipt_observed(outcome.run_receipt)
            return replace(
                outcome.result,
                provider_call_receipt=outcome.run_receipt,
            )
        return outcome.result


__all__ = [
    "ContextProviderTurnExecutionError",
    "ContextProviderTurnExecutionService",
    "official_provider_transport_target_sha256",
]
