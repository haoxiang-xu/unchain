"""Attempt-scoped persistence of exact provider wire authorities.

This service owns no network transport and performs no provider retry.  It
turns one already-built ``ModelTurnRequest`` into a catalog and provider-wire
snapshot, then reconstructs the same authority after process restart.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from unchain.context.ports import BoundArtifactRepository
from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.journal import ArtifactRef, AttemptRef, EventCursor, OperationRef
from unchain.journal.provider_wire import (
    BoundProviderWireStore,
    ProviderWireReceiptIntegrityError,
    RecoveredProviderWireAuthority,
    persist_provider_wire_snapshot,
    recover_provider_wire_authority,
)
from unchain.journal.tool_catalog import (
    BoundToolCatalogIndex,
    ToolCatalogReceiptIntegrityError,
    recover_tool_catalog_authority,
)
from unchain.providers.base import ModelTurnRequest
from unchain.tools.handler_registry import DurableToolHandlerRegistry
from unchain.tools.toolkit import Toolkit

from .provider_toolkit import ProviderToolkitAuthorityAdapter


class ProviderTurnAuthorityError(RuntimeError):
    """The provider turn could not be bound to exact durable bytes."""


class ProviderTurnAuthorityConflict(ProviderTurnAuthorityError):
    """An existing attempt/iteration disagreed with the current request."""


class ProviderTurnToolAdapterRequired(ProviderTurnAuthorityError):
    """A tool-bearing turn needs the production toolkit host adapter."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise ProviderTurnAuthorityError(
            "provider authority input must be canonical JSON"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _operation(
    *,
    phase: str,
    attempt: AttemptRef,
    iteration: int,
    detail: Mapping[str, Any],
) -> OperationRef:
    body = {
        "schema": "unchain.provider_turn_authority_operation.v1",
        "phase": phase,
        "attempt": attempt.to_dict(),
        "iteration": iteration,
        "detail": dict(detail),
    }
    digest = _sha256(body)
    return OperationRef(
        operation_id=f"provider-authority-{phase}-{digest[:32]}",
        payload_sha256=digest,
    )


def _event_id(kind: str, digest: str) -> str:
    return f"provider-authority-{kind}-{digest[:32]}"


class ProviderTurnAuthorityService:
    """Persist and recover one exact provider request per turn subject."""

    def __init__(
        self,
        *,
        attempt: AttemptRef,
        store: BoundProviderWireStore,
        transport_target_sha256: str,
        toolkit_adapter: ProviderToolkitAuthorityAdapter | None = None,
    ) -> None:
        if type(attempt) is not AttemptRef:
            raise TypeError("attempt must be an exact AttemptRef")
        if not isinstance(store, BoundProviderWireStore):
            raise TypeError("store must be a BoundProviderWireStore")
        if not isinstance(store, BoundToolCatalogIndex):
            raise TypeError("store must provide the tool catalog receipt index")
        if not isinstance(store, BoundArtifactRepository):
            raise TypeError("store must provide the bound artifact repository")
        if store.execution_id != attempt.generation.execution_id:
            raise ProviderTurnAuthorityError(
                "provider authority store crossed its execution scope"
            )
        if (
            type(transport_target_sha256) is not str
            or len(transport_target_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in transport_target_sha256
            )
        ):
            raise ValueError(
                "transport_target_sha256 must be a lowercase SHA-256 digest"
            )
        self._attempt = AttemptRef.from_dict(attempt.to_dict())
        self._store = store
        self._transport_target_sha256 = transport_target_sha256
        if (
            toolkit_adapter is not None
            and type(toolkit_adapter) is not ProviderToolkitAuthorityAdapter
        ):
            raise TypeError(
                "toolkit_adapter must be the official "
                "ProviderToolkitAuthorityAdapter or null"
            )
        self._toolkit_adapter = toolkit_adapter

    @property
    def attempt(self) -> AttemptRef:
        return AttemptRef.from_dict(self._attempt.to_dict())

    @property
    def store(self) -> BoundProviderWireStore:
        return self._store

    def _validate_request(
        self,
        *,
        model_io: object,
        request: ModelTurnRequest,
    ) -> None:
        if type(request) is not ModelTurnRequest:
            raise TypeError("request must be an exact ModelTurnRequest")
        if request.run_id != self._attempt.attempt_id:
            raise ProviderTurnAuthorityConflict("provider request attempt changed")
        if type(request.iteration) is not int or request.iteration < 0:
            raise ProviderTurnAuthorityError("provider request iteration is invalid")
        provider = getattr(model_io, "provider", None)
        model = getattr(model_io, "model", None)
        if not isinstance(provider, str) or not provider:
            raise ProviderTurnAuthorityError("model_io provider is unavailable")
        if not isinstance(model, str) or not model:
            raise ProviderTurnAuthorityError("model_io model is unavailable")
        if not isinstance(request.toolkit, Toolkit):
            raise TypeError("request toolkit must be a Toolkit")
        if request.toolkit.tools and self._toolkit_adapter is None:
            raise ProviderTurnToolAdapterRequired(
                "tool-bearing provider turns require the official toolkit adapter"
            )

    def _build_draft(
        self,
        *,
        model_io: object,
        request: ModelTurnRequest,
    ):
        # These builders import context contracts themselves. Keep that
        # dependency out of context package initialization.
        from unchain.providers.prepared_request_factory import (
            resolve_prepared_provider_request_payload,
        )
        from unchain.providers.prepared_turn import _build_provider_turn_draft

        request_payload = resolve_prepared_provider_request_payload(
            model_io=model_io,
            request=request,
        )
        capability = getattr(model_io, "_model_capability", None)
        supports_tools = (
            bool(capability("supports_tools", True)) if callable(capability) else True
        )
        prompt_sha256 = _sha256(
            {
                "schema": "unchain.provider_prompt_projection.v1",
                "messages": request_payload["messages"],
                "response_format": request_payload["response_format"],
            }
        )
        exposure_plan_sha256 = _sha256(
            {
                "schema": "unchain.provider_tool_exposure_plan.v1",
                "tools": [],
            }
        )
        if request.toolkit.tools:
            registry, resolutions = self._toolkit_adapter.resolve(request.toolkit)
        else:
            registry, resolutions = DurableToolHandlerRegistry(), ()
        return _build_provider_turn_draft(
            model_io=model_io,
            registry=registry,
            resolutions=resolutions,
            attempt=self._attempt,
            iteration=request.iteration,
            supports_tools=supports_tools,
            request_payload=request_payload,
            prompt_sha256=prompt_sha256,
            exposure_plan_sha256=exposure_plan_sha256,
        )

    def _load_catalog(self, *, iteration: int):
        lookup = self._store.lookup_tool_catalog_receipts(
            attempt=self._attempt,
            iteration=iteration,
        )
        if lookup.overflow or len(lookup.events) > 1:
            raise ProviderTurnAuthorityConflict(
                "provider tool catalog receipts are conflicting"
            )
        if not lookup.events:
            return None
        event = lookup.events[0]
        raw_artifact = event.payload.get("catalog_artifact")
        try:
            artifact = ArtifactRef.from_dict(raw_artifact)
            content = self._store.read_full_verified(artifact=artifact)
            decoded = json.loads(content.decode("utf-8"))
            catalog = ToolCatalogEnvelope.from_dict(decoded)
        except Exception as exc:
            raise ProviderTurnAuthorityConflict(
                "existing provider tool catalog cannot be recovered"
            ) from exc
        if (
            catalog.canonical_bytes() != content
            or catalog.attempt != self._attempt
            or catalog.iteration != iteration
            or event.payload.get("catalog_sha256") != catalog.catalog_sha256
        ):
            raise ProviderTurnAuthorityConflict(
                "existing provider tool catalog changed"
            )
        try:
            authority = recover_tool_catalog_authority(
                self._store,
                attempt=self._attempt,
                iteration=iteration,
                expected_catalog_sha256=catalog.catalog_sha256,
                expected_catalog_artifact=artifact,
            )
        except ToolCatalogReceiptIntegrityError as exc:
            raise ProviderTurnAuthorityConflict(
                "existing provider tool catalog authority changed"
            ) from exc
        return catalog, authority

    def _persist_catalog(self, draft):
        from unchain.journal.tool_catalog_persistence import persist_tool_catalog_snapshot
        from unchain.providers.prepared_turn import _issue_persisted_tool_catalog_authority

        detail = {"catalog_sha256": draft.catalog.catalog_sha256}
        snapshot = persist_tool_catalog_snapshot(
            journal=self._store,
            artifacts=self._store,
            envelope=draft.catalog,
            artifact_operation=_operation(
                phase="catalog-artifact",
                attempt=self._attempt,
                iteration=draft.iteration,
                detail=detail,
            ),
            event_operation=_operation(
                phase="catalog-event",
                attempt=self._attempt,
                iteration=draft.iteration,
                detail=detail,
            ),
            event_id=_event_id("catalog", draft.catalog.catalog_sha256),
        )
        return draft.catalog, _issue_persisted_tool_catalog_authority(snapshot)

    def _catalog_authority(self, draft):
        existing = self._load_catalog(iteration=draft.iteration)
        if existing is None:
            return self._persist_catalog(draft)
        catalog, authority = existing
        if catalog.to_dict() != draft.catalog.to_dict():
            raise ProviderTurnAuthorityConflict(
                "existing provider tool catalog changed from the current request"
            )
        return catalog, authority

    def _load_wire_authority(self, *, envelope, catalog):
        lookup = self._store.lookup_provider_wire_receipts(
            attempt=self._attempt,
            iteration=envelope.iteration,
        )
        if lookup.overflow or len(lookup.events) > 1:
            raise ProviderTurnAuthorityConflict(
                "provider wire receipts are conflicting"
            )
        if not lookup.events:
            return None
        event = lookup.events[0]
        try:
            artifact = ArtifactRef.from_dict(event.payload.get("wire_artifact"))
            return recover_provider_wire_authority(
                self._store,
                attempt=self._attempt,
                iteration=envelope.iteration,
                catalog=catalog,
                expected_provider=envelope.provider,
                expected_adapter_revision=envelope.adapter_revision,
                expected_envelope_sha256=envelope.envelope_sha256,
                expected_artifact=artifact,
                expected_cursor=EventCursor(event.store_seq, event.event_id),
            )
        except (ProviderWireReceiptIntegrityError, TypeError, ValueError) as exc:
            raise ProviderTurnAuthorityConflict(
                "existing provider wire changed from the current request"
            ) from exc

    def _wire_authority(self, *, envelope, catalog):
        existing = self._load_wire_authority(
            envelope=envelope,
            catalog=catalog,
        )
        if existing is not None:
            return existing
        detail = {"envelope_sha256": envelope.envelope_sha256}
        persist_provider_wire_snapshot(
            self._store,
            envelope=envelope,
            catalog=catalog,
            artifact_operation=_operation(
                phase="wire-artifact",
                attempt=self._attempt,
                iteration=envelope.iteration,
                detail=detail,
            ),
            event_operation=_operation(
                phase="wire-event",
                attempt=self._attempt,
                iteration=envelope.iteration,
                detail=detail,
            ),
            event_id=_event_id("wire", envelope.envelope_sha256),
            expected_artifact_revision=0,
        )
        existing = self._load_wire_authority(
            envelope=envelope,
            catalog=catalog,
        )
        if existing is None:
            raise ProviderTurnAuthorityConflict(
                "provider wire receipt is missing after persistence"
            )
        return existing

    def _prepare_envelope(self, *, model_io: object, request: ModelTurnRequest):
        from unchain.providers.prepared_turn import _issue_prepared_provider_turn
        from unchain.providers.wire_preparer import prepare_provider_wire

        self._validate_request(model_io=model_io, request=request)
        draft = self._build_draft(model_io=model_io, request=request)
        catalog_authority = self._load_catalog(iteration=draft.iteration)
        if catalog_authority is None:
            return draft, None, None
        catalog, authority = catalog_authority
        if catalog.to_dict() != draft.catalog.to_dict():
            raise ProviderTurnAuthorityConflict(
                "existing provider tool catalog changed from the current request"
            )
        prepared = _issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=authority,
        )
        envelope = prepare_provider_wire(
            prepared,
            model_io=model_io,
            attempt=self._attempt,
            iteration=request.iteration,
            transport_target_sha256=self._transport_target_sha256,
        )
        if envelope.catalog_sha256 != catalog.catalog_sha256:
            raise ProviderTurnAuthorityConflict(
                "provider wire catalog changed during preparation"
            )
        return draft, catalog, envelope

    def recover_existing(
        self,
        *,
        model_io: object,
        request: ModelTurnRequest,
    ) -> RecoveredProviderWireAuthority | None:
        """Recover complete durable authority without creating any evidence."""

        _draft, catalog, envelope = self._prepare_envelope(
            model_io=model_io,
            request=request,
        )
        if catalog is None or envelope is None:
            return None
        return self._load_wire_authority(envelope=envelope, catalog=catalog)

    def prepare(
        self,
        *,
        model_io: object,
        request: ModelTurnRequest,
    ) -> RecoveredProviderWireAuthority:
        from unchain.providers.prepared_turn import _issue_prepared_provider_turn
        from unchain.providers.wire_preparer import prepare_provider_wire

        self._validate_request(model_io=model_io, request=request)
        draft = self._build_draft(model_io=model_io, request=request)
        catalog, catalog_authority = self._catalog_authority(draft)
        prepared = _issue_prepared_provider_turn(
            draft=draft,
            catalog_authority=catalog_authority,
        )
        envelope = prepare_provider_wire(
            prepared,
            model_io=model_io,
            attempt=self._attempt,
            iteration=request.iteration,
            transport_target_sha256=self._transport_target_sha256,
        )
        if envelope.catalog_sha256 != catalog.catalog_sha256:
            raise ProviderTurnAuthorityConflict(
                "provider wire catalog changed during preparation"
            )
        return self._wire_authority(envelope=envelope, catalog=catalog)


__all__ = [
    "ProviderTurnAuthorityConflict",
    "ProviderTurnAuthorityError",
    "ProviderTurnAuthorityService",
    "ProviderTurnToolAdapterRequired",
]
