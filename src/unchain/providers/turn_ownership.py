"""Attempt-scoped durable ownership for every provider send in one run.

This is the Context-compiler-independent production seam.  A host binds one
exact provider execution service and its accounting ledger to a RunIdentity;
kernel, selector, observation, and tool-owned sends then share the same
lease/result/receipt boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

from ..context.provider_execution import ContextProviderTurnExecutionService
from ..kernel.types import ModelTurnResult
from ..run_bundle import RunIdentity, canonical_sha256
from ..run_bundle_ledger import RunBundleLedger
from ..retry import RetryConfig


# Compatibility is intentional: the implementation never depends on a
# Context compiler.  The provider-facing name is the official construction
# API; the old Context name remains the same class object for existing hosts.
ProviderTurnExecutionService = ContextProviderTurnExecutionService


class ProviderTurnOwnershipError(RuntimeError):
    """A provider send escaped or changed its attempt-scoped owner."""


@runtime_checkable
class ProviderTurnOwnershipFactory(Protocol):
    """Host capability which binds one durable owner for an exact run."""

    def bind(self, *, identity: RunIdentity) -> "ProviderTurnOwnership": ...


def _occurrence_sha256(identity: RunIdentity, occurrence_id: str) -> str:
    if type(occurrence_id) is not str:
        raise TypeError("provider occurrence_id must be exact text")
    if (
        not occurrence_id
        or len(occurrence_id) > 4096
        or any(ord(character) < 32 for character in occurrence_id)
    ):
        raise ValueError("provider occurrence_id must be bounded canonical text")
    return canonical_sha256(
        {
            "schema": "unchain.provider_turn_occurrence.v1",
            "identity": identity.to_dict(),
            "occurrence_id": occurrence_id,
        }
    )


@dataclass(frozen=True)
class ProviderTurnOwnership:
    """Inseparable exact-send and accounting capabilities for one run."""

    identity: RunIdentity
    service: ProviderTurnExecutionService
    ledger: RunBundleLedger
    factory: ProviderTurnOwnershipFactory | None = None

    def __post_init__(self) -> None:
        if type(self.identity) is not RunIdentity:
            raise TypeError("identity must be an exact RunIdentity")
        if type(self.service) is not ProviderTurnExecutionService:
            raise TypeError(
                "service must be the official ProviderTurnExecutionService"
            )
        if not isinstance(self.ledger, RunBundleLedger):
            raise TypeError("ledger must implement RunBundleLedger")
        if self.service.store is not self.ledger:
            raise ProviderTurnOwnershipError(
                "provider result CAS and accounting ledger must be the same store"
            )
        attempt = self.service.attempt
        if (
            attempt.generation.execution_id != self.identity.execution_id
            or attempt.attempt_id != self.identity.attempt_id
            or self.ledger.execution_id != self.identity.execution_id
        ):
            raise ProviderTurnOwnershipError(
                "provider owner does not match the run identity"
            )
        from .durable_turn_runtime import DurableProviderTurnMode

        if self.service.mode not in {
            DurableProviderTurnMode.ENFORCE,
            DurableProviderTurnMode.ENFORCE_TEST,
        }:
            raise ProviderTurnOwnershipError(
                "provider owner requires an enforcing durable service"
            )
        if self.factory is not None and not isinstance(
            self.factory,
            ProviderTurnOwnershipFactory,
        ):
            raise TypeError("factory must implement ProviderTurnOwnershipFactory")

    def occurrence_sha256(self, occurrence_id: str) -> str:
        return _occurrence_sha256(self.identity, occurrence_id)

    def logical_request_sha256(
        self,
        *,
        request_sha256: str,
        occurrence_id: str,
    ) -> str:
        if (
            type(request_sha256) is not str
            or len(request_sha256) != 64
            or any(character not in "0123456789abcdef" for character in request_sha256)
        ):
            raise ValueError("request_sha256 must be a lowercase SHA-256 digest")
        return canonical_sha256(
            {
                "schema": "unchain.logical_provider_request.v1",
                "request_sha256": request_sha256,
                "occurrence_sha256": self.occurrence_sha256(occurrence_id),
            }
        )

    def fetch_turn(
        self,
        *,
        state: Any,
        model_io: object,
        request: Any,
        occurrence_id: str,
        purpose: str,
        iteration: int,
        request_sha256: str,
        retry_config: RetryConfig,
        before_attempt: Any = None,
        after_attempt: Any = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> ModelTurnResult:
        # The ledger refers to ownership types. Resolve its receipt builders
        # at the send boundary rather than creating a module import cycle.
        from ..kernel.run_ledger import (
            build_model_attempt_receipt,
            provider_call_route,
        )

        run_ledger = getattr(state, "run_ledger", None)
        if run_ledger is None or run_ledger.identity != self.identity:
            raise ProviderTurnOwnershipError(
                "provider owner is not bound to the active RunLedger"
            )
        if run_ledger.persistence is not self.ledger:
            raise ProviderTurnOwnershipError(
                "provider owner accounting ledger is not attached to the run"
            )
        resolved_provider = str(
            provider or getattr(model_io, "provider", None) or "unknown"
        ).strip().lower()
        resolved_model = str(
            model or getattr(model_io, "model", None) or "unknown-model"
        ).strip()
        logical_digest = self.logical_request_sha256(
            request_sha256=request_sha256,
            occurrence_id=occurrence_id,
        )

        def receipt_factory(
            attempt_number: int,
            started_at: str,
            completed_at: str,
            outcome: str,
            classification: str,
            result: ModelTurnResult | None,
        ):
            return build_model_attempt_receipt(
                identity=self.identity,
                provider=resolved_provider,
                model=resolved_model,
                iteration=iteration,
                retry_ordinal=attempt_number,
                purpose=purpose,
                request_digest=logical_digest,
                route=provider_call_route(resolved_provider),
                payload=request.payload,
                started_at=started_at,
                completed_at=completed_at,
                turn=result,
                status=outcome,
                classification=classification,
            )

        effective_request = replace(
            request,
            run_id=self.identity.attempt_id,
            iteration=max(0, int(iteration)),
        )
        result = self.service.fetch_prepared(
            model_io=model_io,
            request=effective_request,
            retry_config=retry_config,
            before_attempt=before_attempt,
            after_attempt=after_attempt,
            run_receipt_factory=receipt_factory,
            run_receipt_observed=run_ledger.append,
            occurrence_sha256=self.occurrence_sha256(occurrence_id),
        )
        if type(result) is not ModelTurnResult:
            raise ProviderTurnOwnershipError(
                "enforcing provider owner returned no exact ModelTurnResult"
            )
        return result


__all__ = [
    "ProviderTurnExecutionService",
    "ProviderTurnOwnership",
    "ProviderTurnOwnershipError",
    "ProviderTurnOwnershipFactory",
]
