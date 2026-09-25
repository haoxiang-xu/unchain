"""Single-send transports for persisted provider wire routes.

These adapters never rebuild a provider request.  The persisted
``ProviderWireRoute`` is the only source of SDK request kwargs/body data.
Retries and route transitions belong to ``DurableProviderTurnRuntime``.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Callable

from unchain.context.tool_catalog import ToolCatalogEnvelope
from unchain.durability import is_durable_persistence_failure
from unchain.kernel.provider_replay import (
    redact_provider_replay_secrets,
    strict_json_copy,
)
from unchain.kernel.types import ModelTurnResult

from .anthropic import AnthropicModelIO
from .base import ModelTurnRequest
from .durable_turn_runtime import (
    ExactProviderRouteFailure,
    ExactProviderRouteFailureKind,
    ExactProviderRouteTransport,
)
from .ollama import OllamaModelIO
from .openai import OpenAIModelIO
from .wire_envelope import ProviderWireEnvelope, ProviderWireRoute


def _explicit_status_code(error: BaseException) -> int | None:
    status_code = getattr(error, "status_code", None)
    if type(status_code) is int:
        return status_code
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if type(status_code) is int:
        return status_code
    # google-genai APIError uses code for HTTP status, even without a response.
    from google.genai.errors import APIError

    if isinstance(error, APIError) and type(error.code) is int:
        return error.code
    return None


def _classified_failure_kind(
    error: BaseException,
) -> ExactProviderRouteFailureKind | None:
    status_code = _explicit_status_code(error)
    if status_code in {429, 529}:
        return ExactProviderRouteFailureKind.TRANSIENT_RETRY_SAFE
    if (
        status_code is not None
        and 400 <= status_code < 500
        and status_code not in {408, 409}
    ):
        return ExactProviderRouteFailureKind.TERMINAL
    return None


class _CatalogToolkitView:
    """Read-only provider-schema view used only to build replay metadata."""

    def __init__(self, catalog: ToolCatalogEnvelope) -> None:
        self._provider = catalog.provider
        self._schemas = catalog.to_dict()["semantic_schemas"]

    def to_provider_json(self, provider: str | None = None) -> list[dict[str, Any]]:
        if provider is not None and provider != self._provider:
            return []
        return strict_json_copy(self._schemas)


class _BufferedExactRouteTransport(ExactProviderRouteTransport):
    provider = ""

    def __init__(
        self,
        *,
        catalog: ToolCatalogEnvelope,
        callback: Callable[[dict[str, Any]], None] | None = None,
        run_id: str = "kernel",
        emit_stream: bool = False,
    ) -> None:
        if type(catalog) is not ToolCatalogEnvelope:
            raise TypeError("catalog must be an exact ToolCatalogEnvelope")
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or null")
        if type(run_id) is not str or not run_id:
            raise ValueError("run_id must be non-empty exact text")
        if type(emit_stream) is not bool:
            raise TypeError("emit_stream must be an exact boolean")
        self._catalog = catalog
        self._callback = callback
        self._run_id = run_id
        self._emit_stream = emit_stream
        self._buffered_events: list[dict[str, Any]] = []

    @property
    def buffered_events(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._buffered_events))

    def discard_buffered_events(self) -> None:
        self._buffered_events.clear()

    def release_buffered_events(self) -> None:
        events = self.buffered_events
        self._buffered_events.clear()
        if self._callback is None:
            return
        for event in events:
            self._callback(event)

    def _capture_event(self, event: dict[str, Any]) -> None:
        if type(event) is not dict:
            raise TypeError("provider callback event must be an exact dict")
        copied = strict_json_copy(event)
        if type(copied) is not dict:
            raise TypeError("provider callback event must remain an exact dict")
        self._buffered_events.append(copied)

    def _begin_send(
        self,
        *,
        envelope: ProviderWireEnvelope,
        route: ProviderWireRoute,
        retry_ordinal: int,
        configured_model: str,
    ) -> dict[str, Any]:
        self.discard_buffered_events()
        if type(envelope) is not ProviderWireEnvelope:
            raise TypeError("envelope must be an exact ProviderWireEnvelope")
        if type(route) is not ProviderWireRoute:
            raise TypeError("route must be an exact ProviderWireRoute")
        if type(retry_ordinal) is not int or retry_ordinal < 0:
            raise ValueError("retry_ordinal must be a non-negative exact integer")
        if envelope.provider != self.provider or self._catalog.provider != self.provider:
            raise ValueError("exact route transport provider changed")
        if envelope.configured_model != configured_model:
            raise ValueError("exact route transport model changed")
        envelope.verify_against_catalog(self._catalog)
        if not any(
            candidate.name == route.name
            and candidate.route_sha256 == route.route_sha256
            and candidate.to_dict() == route.to_dict()
            for candidate in envelope.routes
        ):
            raise ValueError("exact route does not belong to the provider envelope")
        return route.request_copy()

    def _result_request(
        self,
        *,
        envelope: ProviderWireEnvelope,
        messages: list[dict[str, Any]],
    ) -> ModelTurnRequest:
        return ModelTurnRequest(
            messages=strict_json_copy(messages),
            callback=self._capture_event,
            run_id=self._run_id,
            iteration=envelope.iteration,
            toolkit=_CatalogToolkitView(self._catalog),
            emit_stream=self._emit_stream,
        )


class OpenAIExactRouteTransport(_BufferedExactRouteTransport):
    """Send one persisted OpenAI Responses route exactly once."""

    provider = "openai"

    def __init__(
        self,
        *,
        model_io: OpenAIModelIO,
        catalog: ToolCatalogEnvelope,
        callback: Callable[[dict[str, Any]], None] | None = None,
        run_id: str = "kernel",
        emit_stream: bool = False,
    ) -> None:
        if type(model_io) is not OpenAIModelIO:
            raise TypeError("model_io must be an exact OpenAIModelIO")
        self._model_io = model_io
        super().__init__(
            catalog=catalog,
            callback=callback,
            run_id=run_id,
            emit_stream=emit_stream,
        )

    def send(
        self,
        *,
        envelope: ProviderWireEnvelope,
        route: ProviderWireRoute,
        retry_ordinal: int,
    ) -> ModelTurnResult:
        request_kwargs = self._begin_send(
            envelope=envelope,
            route=route,
            retry_ordinal=retry_ordinal,
            configured_model=self._model_io.model,
        )
        request_input = request_kwargs.get("input")
        if type(request_input) is not list:
            raise TypeError("persisted OpenAI input must be an exact list")
        request = self._result_request(
            envelope=envelope,
            messages=request_input,
        )
        tools = request_kwargs.get("tools")
        self._model_io._emit_request_messages(
            callback=self._capture_event,
            run_id=self._run_id,
            iteration=envelope.iteration,
            messages=redact_provider_replay_secrets(request_input),
            previous_response_id=request_kwargs.get("previous_response_id"),
            tool_names=self._model_io._tool_names_for_trace(
                tools if isinstance(tools, list) else []
            ),
        )
        client = self._model_io._client_factory(
            api_key=self._model_io.api_key,
            max_retries=0,
        )
        try:
            return self._model_io._fetch_turn_streaming(
                client,
                request,
                request_kwargs,
            )
        except Exception as exc:
            if is_durable_persistence_failure(exc):
                raise
            status_code = _explicit_status_code(exc)
            if (
                route.name == "primary"
                and "previous_response_id" in request_kwargs
                and type(status_code) is int
                and self._model_io._is_previous_response_error(exc)
            ):
                self.discard_buffered_events()
                raise ExactProviderRouteFailure(
                    ExactProviderRouteFailureKind.PREVIOUS_RESPONSE_FALLBACK,
                    exc,
                ) from None
            kind = _classified_failure_kind(exc)
            self.discard_buffered_events()
            if kind is not None:
                raise ExactProviderRouteFailure(kind, exc) from None
            raise


class _AnthropicFamilyExactRouteTransport(_BufferedExactRouteTransport):
    """Shared single-send transport for the Anthropic Messages protocol."""

    def __init__(
        self,
        *,
        model_io: AnthropicModelIO,
        catalog: ToolCatalogEnvelope,
        callback: Callable[[dict[str, Any]], None] | None = None,
        run_id: str = "kernel",
        emit_stream: bool = False,
    ) -> None:
        if not isinstance(model_io, AnthropicModelIO):
            raise TypeError("model_io must be an AnthropicModelIO")
        if model_io.provider != self.provider:
            raise ValueError("model_io provider does not match exact transport")
        self._model_io = model_io
        super().__init__(
            catalog=catalog,
            callback=callback,
            run_id=run_id,
            emit_stream=emit_stream,
        )

    @staticmethod
    def _system_text(request_kwargs: dict[str, Any]) -> str | None:
        system = request_kwargs.get("system")
        if not isinstance(system, list):
            return None
        parts = [
            str(block.get("text") or "")
            for block in system
            if isinstance(block, dict) and block.get("text") not in (None, "")
        ]
        text = "\n\n".join(parts)
        return text or None

    def send(
        self,
        *,
        envelope: ProviderWireEnvelope,
        route: ProviderWireRoute,
        retry_ordinal: int,
    ) -> ModelTurnResult:
        request_kwargs = self._begin_send(
            envelope=envelope,
            route=route,
            retry_ordinal=retry_ordinal,
            configured_model=self._model_io.model,
        )
        messages = request_kwargs.get("messages")
        if type(messages) is not list:
            raise TypeError("persisted Anthropic messages must be an exact list")
        request = self._result_request(
            envelope=envelope,
            messages=messages,
        )
        tools = request_kwargs.get("tools")
        self._model_io._emit_request_messages(
            callback=self._capture_event,
            run_id=self._run_id,
            iteration=envelope.iteration,
            messages=redact_provider_replay_secrets(messages),
            tool_names=self._model_io._tool_names_for_trace(
                tools if isinstance(tools, list) else []
            ),
            system=self._system_text(request_kwargs),
        )
        client = self._model_io._client_factory(
            api_key=self._model_io.api_key,
            timeout=self._model_io._ANTHROPIC_TIMEOUT,
            max_retries=0,
        )
        try:
            return self._model_io._fetch_turn_streaming(
                client,
                request,
                request_kwargs,
            )
        except Exception as exc:
            if is_durable_persistence_failure(exc):
                raise
            kind = _classified_failure_kind(exc)
            self.discard_buffered_events()
            if kind is not None:
                raise ExactProviderRouteFailure(kind, exc) from None
            raise


class AnthropicExactRouteTransport(_AnthropicFamilyExactRouteTransport):
    """Send one persisted Anthropic Messages route exactly once."""

    provider = "anthropic"


class HyperspaceExactRouteTransport(_AnthropicFamilyExactRouteTransport):
    """Send one persisted Hyperspace Anthropic-protocol route exactly once."""

    provider = "hyperspace"


class OllamaExactRouteTransport(_BufferedExactRouteTransport):
    """Send one persisted Ollama chat route exactly once."""

    provider = "ollama"

    def __init__(
        self,
        *,
        model_io: OllamaModelIO,
        catalog: ToolCatalogEnvelope,
        callback: Callable[[dict[str, Any]], None] | None = None,
        run_id: str = "kernel",
        emit_stream: bool = False,
    ) -> None:
        if type(model_io) is not OllamaModelIO:
            raise TypeError("model_io must be an exact OllamaModelIO")
        self._model_io = model_io
        super().__init__(
            catalog=catalog,
            callback=callback,
            run_id=run_id,
            emit_stream=emit_stream,
        )
        self._provisional_reasoning_id: str | None = None
        self._provisional_iteration: int | None = None
        self._preview_ids_by_index: dict[int, str] = {}

    def _capture_event(self, event: dict[str, Any]) -> None:
        super()._capture_event(event)
        if event.get("type") != "reasoning" or not self._emit_stream:
            return
        preview = getattr(self._callback, "emit_provisional_reasoning", None)
        commit = getattr(self._callback, "commit_provisional_reasoning", None)
        discard = getattr(self._callback, "discard_provisional_reasoning", None)
        if not all(callable(method) for method in (preview, commit, discard)):
            return
        if self._provisional_reasoning_id is None:
            raise RuntimeError("Ollama reasoning preview has no physical send identity")
        index = len(self._buffered_events) - 1
        preview_id = uuid.uuid5(
            uuid.UUID(hex=self._provisional_reasoning_id), str(index)
        ).hex
        self._preview_ids_by_index[index] = preview_id
        preview(copy.deepcopy(self._buffered_events[index]), preview_id)

    def discard_buffered_events(self) -> None:
        try:
            if self._preview_ids_by_index:
                discard = getattr(self._callback, "discard_provisional_reasoning", None)
                if callable(discard):
                    first_error = None
                    for preview_id in self._preview_ids_by_index.values():
                        try:
                            discard(
                                preview_id=preview_id,
                                run_id=self._run_id,
                                iteration=self._provisional_iteration,
                            )
                        except BaseException as error:
                            if first_error is None:
                                first_error = error
                    if first_error is not None:
                        raise first_error
        finally:
            super().discard_buffered_events()
            self._preview_ids_by_index.clear()
            self._provisional_reasoning_id = None
            self._provisional_iteration = None

    def release_buffered_events(self) -> None:
        events = self.buffered_events
        try:
            if self._callback is not None:
                commit = getattr(self._callback, "commit_provisional_reasoning", None)
                for index, event in enumerate(events):
                    if index in self._preview_ids_by_index and callable(commit):
                        try:
                            commit(event)
                        except BaseException as error:
                            if getattr(
                                error, "_unchain_provisional_reasoning_journaled", False
                            ):
                                self._preview_ids_by_index.pop(index)
                            raise
                        self._preview_ids_by_index.pop(index)
                    else:
                        self._callback(event)
        except BaseException:
            try:
                self.discard_buffered_events()
            except Exception:
                # Preserve the persistence/host failure if reset is unavailable.
                pass
            raise
        self._buffered_events.clear()
        self._preview_ids_by_index.clear()
        self._provisional_reasoning_id = None
        self._provisional_iteration = None

    def send(
        self,
        *,
        envelope: ProviderWireEnvelope,
        route: ProviderWireRoute,
        retry_ordinal: int,
    ) -> ModelTurnResult:
        request_body = self._begin_send(
            envelope=envelope,
            route=route,
            retry_ordinal=retry_ordinal,
            configured_model=self._model_io.model,
        )
        self._provisional_reasoning_id = uuid.uuid4().hex
        self._provisional_iteration = envelope.iteration
        messages = request_body.get("messages")
        if type(messages) is not list:
            raise TypeError("persisted Ollama messages must be an exact list")
        request = self._result_request(
            envelope=envelope,
            messages=messages,
        )
        tools = request_body.get("tools")
        self._model_io._emit_request_messages(
            callback=self._capture_event,
            run_id=self._run_id,
            iteration=envelope.iteration,
            messages=redact_provider_replay_secrets(messages),
            tool_names=self._model_io._tool_names_for_trace(
                tools if isinstance(tools, list) else []
            ),
        )
        try:
            return self._model_io._fetch_turn_streaming(request_body, request)
        except BaseException:
            try:
                self.discard_buffered_events()
            except Exception:
                # A closed/cancelled host stream may reject its preview reset.
                # Preserve the provider failure and still clear local buffers.
                pass
            raise


__all__ = [
    "AnthropicExactRouteTransport",
    "HyperspaceExactRouteTransport",
    "OllamaExactRouteTransport",
    "OpenAIExactRouteTransport",
]


class GeminiExactRouteTransport(_BufferedExactRouteTransport):
    """Send the frozen Gemini request once, with SDK retries disabled."""

    provider = "gemini"

    def __init__(
        self, *, model_io, catalog, callback=None, run_id="kernel", emit_stream=False
    ):
        from .gemini import GeminiModelIO

        if type(model_io) is not GeminiModelIO:
            raise TypeError("model_io must be GeminiModelIO")
        self._model_io = model_io
        super().__init__(
            catalog=catalog, callback=callback, run_id=run_id, emit_stream=emit_stream
        )

    def send(self, *, envelope, route, retry_ordinal):
        wire = self._begin_send(
            envelope=envelope,
            route=route,
            retry_ordinal=retry_ordinal,
            configured_model=self._model_io.model,
        )
        messages = copy.deepcopy(wire["contents"])
        system = wire["config"].get("system_instruction")
        if system:
            messages.insert(0, {"role": "system", "content": system})
        request = self._result_request(envelope=envelope, messages=messages)
        tools = wire.pop("tools", [])
        if tools:
            wire["config"]["tools"] = [{"function_declarations": tools}]
        try:
            return self._model_io._fetch_prepared(request, wire)
        except Exception as exc:
            if is_durable_persistence_failure(exc):
                raise
            self.discard_buffered_events()
            kind = _classified_failure_kind(exc)
            if kind is not None:
                raise ExactProviderRouteFailure(kind, exc) from None
            raise
