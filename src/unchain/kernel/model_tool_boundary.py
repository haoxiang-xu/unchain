"""Final model/tool boundary contracts for ``KernelLoop.step_once``.

This seam does not cover direct ``KernelLoop.fetch_model_turn`` calls or
provider calls made by internal selector/observation runtimes. Those paths are
explicit rollout blockers until they use an authenticated prepared consumer.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..tools.tool import Tool
from ..tools.toolkit import Toolkit
from .state import RunState
from .types import ModelTurnResult


_FINAL_MODEL_TOOL_BOUNDARY_KIND = "unchain.kernel.final_model_tool_boundary.v1"
_MAX_FINAL_MODEL_TOOL_BOUNDARIES = 4096


def _freeze_boundary_value(
    value: Any,
    *,
    active: set[int] | None = None,
) -> Any:
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    seen = active if active is not None else set()
    value_id = id(value)
    if value_id in seen:
        raise ValueError("final model boundary snapshot cannot contain cycles")
    if isinstance(value, Mapping):
        seen.add(value_id)
        try:
            frozen: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError("final model boundary mappings require string keys")
                frozen[key] = _freeze_boundary_value(item, active=seen)
            return MappingProxyType(frozen)
        finally:
            seen.remove(value_id)
    if isinstance(value, (list, tuple)):
        seen.add(value_id)
        try:
            return tuple(_freeze_boundary_value(item, active=seen) for item in value)
        finally:
            seen.remove(value_id)
    if isinstance(value, (set, frozenset)):
        seen.add(value_id)
        try:
            frozen_items = [_freeze_boundary_value(item, active=seen) for item in value]
            return tuple(sorted(frozen_items, key=repr))
        finally:
            seen.remove(value_id)

    raise TypeError("final model boundary snapshot contains an unsupported value")


@dataclass(frozen=True, slots=True)
class FinalModelToolIdentity:
    """Ordered exact Tool identity captured after all ordinary harnesses."""

    name: str
    tool: Tool
    tool_object_id: int
    handler_object_id: int


@dataclass(frozen=True, slots=True)
class FinalModelToolBoundaryContext:
    """Detached, read-only inputs visible to the final boundary."""

    messages: tuple[Mapping[str, Any], ...]
    payload: Mapping[str, Any]
    tool_runtime_config: Mapping[str, Any]
    openai_text_format: Mapping[str, Any] | None
    provider: str
    model: str
    session_id: str
    memory_namespace: str
    execution_id: str
    generation_id: str
    attempt_id: str
    run_id: str
    iteration: int
    latest_version_id: str | None
    toolkit_object_id: int
    toolkit_prompt_sections: tuple[str, ...]
    tools: tuple[FinalModelToolIdentity, ...]
    tool_runtime_plugin_identities: tuple[tuple[str, int], ...]
    model_io_object_id: int | None


def _snapshot_final_model_tool_boundary_context(
    *,
    state: RunState,
    event: Mapping[str, Any],
    iteration: int | None = None,
) -> FinalModelToolBoundaryContext:
    if not isinstance(state, RunState):
        raise TypeError("final model boundary snapshot requires RunState")
    toolkit = event.get("toolkit")
    if not isinstance(toolkit, Toolkit):
        raise TypeError("final model boundary snapshot requires Toolkit")

    messages = _freeze_boundary_value(state.latest_messages())
    if not isinstance(messages, tuple) or any(
        not isinstance(message, Mapping) for message in messages
    ):
        raise TypeError("final model boundary messages are invalid")
    payload = _freeze_boundary_value(event.get("payload") or {})
    tool_runtime_config = _freeze_boundary_value(event.get("tool_runtime_config") or {})
    raw_openai_text_format = event.get("openai_text_format")
    openai_text_format = (
        _freeze_boundary_value(raw_openai_text_format)
        if raw_openai_text_format is not None
        else None
    )
    if not isinstance(payload, Mapping):
        raise TypeError("final model boundary payload is invalid")
    if not isinstance(tool_runtime_config, Mapping):
        raise TypeError("final model boundary tool runtime config is invalid")
    if openai_text_format is not None and not isinstance(
        openai_text_format,
        Mapping,
    ):
        raise TypeError("final model boundary text format is invalid")

    tool_identities: list[FinalModelToolIdentity] = []
    for name, tool_obj in toolkit.tools.items():
        if type(name) is not str or not name:
            raise TypeError("final model boundary toolkit requires exact tool names")
        if not isinstance(tool_obj, Tool):
            raise TypeError("final model boundary toolkit requires exact Tool entries")
        tool_identities.append(
            FinalModelToolIdentity(
                name=name,
                tool=tool_obj,
                tool_object_id=id(tool_obj),
                handler_object_id=id(getattr(tool_obj, "func", None)),
            )
        )
    tools = tuple(tool_identities)
    prompt_sections = toolkit.prompt_sections
    if type(prompt_sections) is not tuple or any(
        type(section) is not str for section in prompt_sections
    ):
        raise TypeError(
            "final model boundary toolkit prompt_sections must be an exact "
            "string tuple"
        )
    plugin_identities = tuple(
        (
            f"{type(plugin).__module__}.{type(plugin).__qualname__}",
            id(plugin),
        )
        for plugin in event.get("tool_runtime_plugins") or ()
    )
    run_id = str(event.get("run_id") or "kernel")
    attempt_id = str(event.get("attempt_id") or run_id)
    session_id = str(state.session_state.session_id or "")
    execution_id = str(event.get("execution_id") or session_id or attempt_id)
    generation_id = str(
        event.get("generation_id")
        or state.metadata.get("generation_id")
        or state.metadata.get("current_generation")
        or ""
    )
    model_io = event.get("model_io")
    resolved_iteration = int(state.iteration) if iteration is None else iteration
    if (
        isinstance(resolved_iteration, bool)
        or not isinstance(resolved_iteration, int)
        or resolved_iteration < 0
    ):
        raise TypeError("final model boundary iteration must be a non-negative integer")
    return FinalModelToolBoundaryContext(
        messages=messages,
        payload=payload,
        tool_runtime_config=tool_runtime_config,
        openai_text_format=openai_text_format,
        provider=str(state.provider_state.provider or ""),
        model=str(state.provider_state.model or ""),
        session_id=session_id,
        memory_namespace=str(state.session_state.memory_namespace or ""),
        execution_id=execution_id,
        generation_id=generation_id,
        attempt_id=attempt_id,
        run_id=run_id,
        iteration=resolved_iteration,
        latest_version_id=state.latest_version_id,
        toolkit_object_id=id(toolkit),
        toolkit_prompt_sections=prompt_sections,
        tools=tools,
        tool_runtime_plugin_identities=plugin_identities,
        model_io_object_id=id(model_io) if model_io is not None else None,
    )


@dataclass(frozen=True, slots=True)
class FinalModelToolPreparation:
    """Exact model and execution values bound for one model turn."""

    model_toolkit: Toolkit
    execution_toolkit: Toolkit
    execution_binding: object
    prepared_provider_turn: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_toolkit, Toolkit):
            raise TypeError("model_toolkit must be a Toolkit")
        if not isinstance(self.execution_toolkit, Toolkit):
            raise TypeError("execution_toolkit must be a Toolkit")
        if self.execution_binding is None:
            raise TypeError("execution_binding must be sealed and non-null")


def _legacy_fetch_prepared(
    context: FinalModelToolBoundaryContext,
    preparation: FinalModelToolPreparation,
    request: Any,
    retry_config: Any,
    before_attempt: Callable[[int], None] | None,
) -> ModelTurnResult | None:
    del context, preparation, request, retry_config, before_attempt
    return None


def _unsupported_prepare_tool_resume(
    context: FinalModelToolBoundaryContext,
    continuation: Mapping[str, Any],
    interaction_request: Mapping[str, Any],
) -> FinalModelToolPreparation | None:
    del context, continuation, interaction_request
    return None


class FinalModelToolBoundary:
    """Exact issuer-created final-boundary value."""

    __slots__ = ("__issued_record", "__weakref__")

    boundary_kind = _FINAL_MODEL_TOOL_BOUNDARY_KIND

    def __new__(cls, *args, **kwargs):
        del cls, args, kwargs
        raise TypeError("FinalModelToolBoundary is issuer-created")

    def __init_subclass__(cls, **kwargs):
        del cls, kwargs
        raise TypeError("FinalModelToolBoundary cannot be subclassed")

    def __copy__(self):
        raise TypeError("FinalModelToolBoundary cannot be copied or serialized")

    def __deepcopy__(self, memo):
        del memo
        raise TypeError("FinalModelToolBoundary cannot be copied or serialized")

    def __reduce__(self):
        raise TypeError("FinalModelToolBoundary cannot be copied or serialized")

    def __reduce_ex__(self, protocol):
        del protocol
        raise TypeError("FinalModelToolBoundary cannot be copied or serialized")

    def prepare(
        self,
        context: FinalModelToolBoundaryContext,
    ) -> FinalModelToolPreparation:
        return _FINAL_MODEL_TOOL_BOUNDARY_ISSUER.prepare(self, context)

    def validate(
        self,
        context: FinalModelToolBoundaryContext,
        preparation: FinalModelToolPreparation,
        turn: ModelTurnResult,
    ) -> ModelTurnResult:
        return _FINAL_MODEL_TOOL_BOUNDARY_ISSUER.validate(
            self,
            context,
            preparation,
            turn,
        )

    def fetch_prepared(
        self,
        context: FinalModelToolBoundaryContext,
        preparation: FinalModelToolPreparation,
        request: Any,
        *,
        retry_config: Any,
        before_attempt: Callable[[int], None] | None = None,
        after_attempt: Callable[[int, str, str, str], None] | None = None,
    ) -> ModelTurnResult | None:
        return _FINAL_MODEL_TOOL_BOUNDARY_ISSUER.fetch_prepared(
            self,
            context,
            preparation,
            request,
            retry_config,
            before_attempt,
            after_attempt,
        )

    def prepare_tool_resume(
        self,
        context: FinalModelToolBoundaryContext,
        *,
        continuation: Mapping[str, Any],
        interaction_request: Mapping[str, Any],
    ) -> FinalModelToolPreparation | None:
        return _FINAL_MODEL_TOOL_BOUNDARY_ISSUER.prepare_tool_resume(
            self,
            context,
            continuation,
            interaction_request,
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _IssuedBoundaryRecord:
    prepare: Callable[[FinalModelToolBoundaryContext], FinalModelToolPreparation]
    validate: Callable[
        [
            FinalModelToolBoundaryContext,
            FinalModelToolPreparation,
            ModelTurnResult,
        ],
        ModelTurnResult,
    ]
    fetch_prepared: Callable[
        [
            FinalModelToolBoundaryContext,
            FinalModelToolPreparation,
            Any,
            Any,
            Callable[[int], None] | None,
        ],
        ModelTurnResult | None,
    ] = _legacy_fetch_prepared
    prepare_tool_resume: Callable[
        [
            FinalModelToolBoundaryContext,
            Mapping[str, Any],
            Mapping[str, Any],
        ],
        FinalModelToolPreparation | None,
    ] = _unsupported_prepare_tool_resume


@dataclass(frozen=True, slots=True)
class _IssuedBoundaryAuthority:
    boundary_ref: weakref.ReferenceType[FinalModelToolBoundary]
    record_ref: weakref.ReferenceType[_IssuedBoundaryRecord]
    prepare_callback_id: int
    validate_callback_id: int
    fetch_prepared_callback_id: int
    prepare_tool_resume_callback_id: int


class _FinalModelToolBoundaryIssuer:
    def __init__(
        self,
        *,
        max_records: int = _MAX_FINAL_MODEL_TOOL_BOUNDARIES,
    ) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int):
            raise TypeError("final model boundary capacity must be an integer")
        if max_records <= 0:
            raise ValueError("final model boundary capacity must be positive")
        self._lock = threading.RLock()
        self._max_records = max_records
        self._records: dict[int, _IssuedBoundaryAuthority] = {}

    def issue(
        self,
        *,
        prepare: Callable[
            [FinalModelToolBoundaryContext],
            FinalModelToolPreparation,
        ],
        validate: Callable[
            [
                FinalModelToolBoundaryContext,
                FinalModelToolPreparation,
                ModelTurnResult,
            ],
            ModelTurnResult,
        ],
        fetch_prepared: Callable[
            [
                FinalModelToolBoundaryContext,
                FinalModelToolPreparation,
                Any,
                Any,
                Callable[[int], None] | None,
            ],
            ModelTurnResult | None,
        ]
        | None = None,
        prepare_tool_resume: Callable[
            [
                FinalModelToolBoundaryContext,
                Mapping[str, Any],
                Mapping[str, Any],
            ],
            FinalModelToolPreparation | None,
        ]
        | None = None,
    ) -> FinalModelToolBoundary:
        resolved_fetch_prepared = (
            _legacy_fetch_prepared if fetch_prepared is None else fetch_prepared
        )
        resolved_prepare_tool_resume = (
            _unsupported_prepare_tool_resume
            if prepare_tool_resume is None
            else prepare_tool_resume
        )
        if (
            not callable(prepare)
            or not callable(validate)
            or not callable(resolved_fetch_prepared)
            or not callable(resolved_prepare_tool_resume)
        ):
            raise TypeError("final model boundary callbacks must be callable")
        boundary = object.__new__(FinalModelToolBoundary)
        boundary_id = id(boundary)
        record = _IssuedBoundaryRecord(
            prepare=prepare,
            validate=validate,
            fetch_prepared=resolved_fetch_prepared,
            prepare_tool_resume=resolved_prepare_tool_resume,
        )
        object.__setattr__(
            boundary,
            "_FinalModelToolBoundary__issued_record",
            record,
        )

        def remove_record(expired_ref: weakref.ReferenceType[object]) -> None:
            with self._lock:
                existing = self._records.get(boundary_id)
                if existing is not None and (
                    expired_ref is existing.boundary_ref
                    or expired_ref is existing.record_ref
                ):
                    self._records.pop(boundary_id, None)

        boundary_ref = weakref.ref(boundary, remove_record)
        record_ref = weakref.ref(record, remove_record)
        authority = _IssuedBoundaryAuthority(
            boundary_ref=boundary_ref,
            record_ref=record_ref,
            prepare_callback_id=id(prepare),
            validate_callback_id=id(validate),
            fetch_prepared_callback_id=id(resolved_fetch_prepared),
            prepare_tool_resume_callback_id=id(resolved_prepare_tool_resume),
        )
        with self._lock:
            dead_ids = [
                record_id
                for record_id, existing in self._records.items()
                if (existing.boundary_ref() is None or existing.record_ref() is None)
            ]
            for record_id in dead_ids:
                self._records.pop(record_id, None)
            if len(self._records) >= self._max_records:
                raise RuntimeError("final model boundary issuer capacity exceeded")
            existing = self._records.get(boundary_id)
            if existing is not None and existing.boundary_ref() is not None:
                raise RuntimeError("final model boundary identity collision")
            self._records[boundary_id] = authority
        return boundary

    def record_for(
        self,
        value: object,
    ) -> _IssuedBoundaryRecord | None:
        if type(value) is not FinalModelToolBoundary:
            return None
        with self._lock:
            authority = self._records.get(id(value))
            if authority is None or authority.boundary_ref() is not value:
                return None
        try:
            record = object.__getattribute__(
                value,
                "_FinalModelToolBoundary__issued_record",
            )
        except AttributeError:
            return None
        if (
            type(record) is not _IssuedBoundaryRecord
            or authority.record_ref() is not record
            or id(record.prepare) != authority.prepare_callback_id
            or id(record.validate) != authority.validate_callback_id
            or id(record.fetch_prepared) != authority.fetch_prepared_callback_id
            or id(record.prepare_tool_resume)
            != authority.prepare_tool_resume_callback_id
        ):
            return None
        return record

    def prepare(
        self,
        boundary: FinalModelToolBoundary,
        context: FinalModelToolBoundaryContext,
    ) -> FinalModelToolPreparation:
        record = self.record_for(boundary)
        if record is None:
            raise TypeError("invalid final model boundary authority")
        return record.prepare(context)

    def validate(
        self,
        boundary: FinalModelToolBoundary,
        context: FinalModelToolBoundaryContext,
        preparation: FinalModelToolPreparation,
        turn: ModelTurnResult,
    ) -> ModelTurnResult:
        record = self.record_for(boundary)
        if record is None:
            raise TypeError("invalid final model boundary authority")
        return record.validate(context, preparation, turn)

    def fetch_prepared(
        self,
        boundary: FinalModelToolBoundary,
        context: FinalModelToolBoundaryContext,
        preparation: FinalModelToolPreparation,
        request: Any,
        retry_config: Any,
        before_attempt: Callable[[int], None] | None,
        after_attempt: Callable[[int, str, str, str], None] | None,
    ) -> ModelTurnResult | None:
        record = self.record_for(boundary)
        if record is None:
            raise TypeError("invalid final model boundary authority")
        observed_before_attempt = before_attempt
        if after_attempt is not None:
            def observed_before_attempt(attempt: int) -> None:
                if before_attempt is not None:
                    before_attempt(attempt)

            setattr(
                observed_before_attempt,
                "after_attempt",
                after_attempt,
            )
            for attribute in (
                "run_receipt_factory",
                "run_receipt_observed",
                "retry_wait",
            ):
                value = getattr(before_attempt, attribute, None)
                if value is not None:
                    setattr(observed_before_attempt, attribute, value)
        return record.fetch_prepared(
            context,
            preparation,
            request,
            retry_config,
            observed_before_attempt,
        )

    def prepare_tool_resume(
        self,
        boundary: FinalModelToolBoundary,
        context: FinalModelToolBoundaryContext,
        continuation: Mapping[str, Any],
        interaction_request: Mapping[str, Any],
    ) -> FinalModelToolPreparation | None:
        record = self.record_for(boundary)
        if record is None:
            raise TypeError("invalid final model boundary authority")
        return record.prepare_tool_resume(
            context,
            continuation,
            interaction_request,
        )


_FINAL_MODEL_TOOL_BOUNDARY_ISSUER = _FinalModelToolBoundaryIssuer()


@dataclass(frozen=True, slots=True, weakref_slot=True)
class _LoopBoundarySeal:
    boundary: FinalModelToolBoundary


@dataclass(frozen=True, slots=True)
class _LoopBoundaryAuthority:
    loop_ref: weakref.ReferenceType[object]
    seal_ref: weakref.ReferenceType[_LoopBoundarySeal]
    boundary_ref: weakref.ReferenceType[FinalModelToolBoundary]


class _FinalModelToolBoundaryBindingRegistry:
    _STORAGE_ATTRIBUTE = "_KernelLoop__final_model_tool_boundary_seal"

    def __init__(
        self,
        *,
        max_records: int = _MAX_FINAL_MODEL_TOOL_BOUNDARIES,
    ) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int):
            raise TypeError("loop boundary seal capacity must be an integer")
        if max_records <= 0:
            raise ValueError("loop boundary seal capacity must be positive")
        self._lock = threading.RLock()
        self._max_records = max_records
        self._records: dict[int, _LoopBoundaryAuthority] = {}

    def bind(
        self,
        loop: object,
        boundary: FinalModelToolBoundary,
    ) -> None:
        if not _is_authentic_final_model_tool_boundary(boundary):
            raise TypeError("invalid final model boundary registration")
        loop_id = id(loop)
        seal = _LoopBoundarySeal(boundary=boundary)

        def remove_record(expired_ref: weakref.ReferenceType[object]) -> None:
            with self._lock:
                existing = self._records.get(loop_id)
                if existing is not None and expired_ref is existing.loop_ref:
                    self._records.pop(loop_id, None)

        try:
            loop_ref = weakref.ref(loop, remove_record)
        except TypeError as exc:
            raise TypeError(
                "final model boundary owner must support weak refs"
            ) from exc
        seal_ref = weakref.ref(seal)
        boundary_ref = weakref.ref(boundary)
        authority = _LoopBoundaryAuthority(
            loop_ref=loop_ref,
            seal_ref=seal_ref,
            boundary_ref=boundary_ref,
        )
        with self._lock:
            dead_ids = [
                record_id
                for record_id, existing in self._records.items()
                if existing.loop_ref() is None
            ]
            for record_id in dead_ids:
                self._records.pop(record_id, None)
            existing = self._records.get(loop_id)
            if existing is not None and existing.loop_ref() is loop:
                raise ValueError("final model boundary is already registered")
            if len(self._records) >= self._max_records:
                raise RuntimeError("loop boundary seal registry capacity exceeded")
            try:
                self._records[loop_id] = authority
                object.__setattr__(loop, self._STORAGE_ATTRIBUTE, seal)
            except BaseException:
                if self._records.get(loop_id) is authority:
                    self._records.pop(loop_id, None)
                raise

    def resolve(self, loop: object) -> FinalModelToolBoundary | None:
        with self._lock:
            authority = self._records.get(id(loop))
            if authority is None:
                return None
            if authority.loop_ref() is not loop:
                raise RuntimeError("final model boundary seal owner changed")
            seal = authority.seal_ref()
            boundary = authority.boundary_ref()
        if seal is None or boundary is None:
            raise RuntimeError("final model boundary seal expired")
        try:
            current_seal = object.__getattribute__(
                loop,
                self._STORAGE_ATTRIBUTE,
            )
        except AttributeError as exc:
            raise RuntimeError("final model boundary seal was cleared") from exc
        if current_seal is not seal:
            raise RuntimeError("final model boundary seal was replaced")
        if seal.boundary is not boundary:
            raise RuntimeError("final model boundary seal identity changed")
        if not _is_authentic_final_model_tool_boundary(boundary):
            raise RuntimeError("registered final model boundary lost authority")
        return boundary


_FINAL_MODEL_TOOL_BOUNDARY_BINDINGS = _FinalModelToolBoundaryBindingRegistry()


def _bind_final_model_tool_boundary(
    loop: object,
    boundary: FinalModelToolBoundary,
) -> None:
    _FINAL_MODEL_TOOL_BOUNDARY_BINDINGS.bind(loop, boundary)


def _resolve_final_model_tool_boundary(
    loop: object,
) -> FinalModelToolBoundary | None:
    return _FINAL_MODEL_TOOL_BOUNDARY_BINDINGS.resolve(loop)


def _issue_final_model_tool_boundary(
    *,
    prepare: Callable[
        [FinalModelToolBoundaryContext],
        FinalModelToolPreparation,
    ],
    validate: Callable[
        [
            FinalModelToolBoundaryContext,
            FinalModelToolPreparation,
            ModelTurnResult,
        ],
        ModelTurnResult,
    ],
    fetch_prepared: Callable[
        [
            FinalModelToolBoundaryContext,
            FinalModelToolPreparation,
            Any,
            Any,
            Callable[[int], None] | None,
        ],
        ModelTurnResult | None,
    ]
    | None = None,
    prepare_tool_resume: Callable[
        [
            FinalModelToolBoundaryContext,
            Mapping[str, Any],
            Mapping[str, Any],
        ],
        FinalModelToolPreparation | None,
    ]
    | None = None,
) -> FinalModelToolBoundary:
    """Private system factory for official ContextRuntime integration."""

    return _FINAL_MODEL_TOOL_BOUNDARY_ISSUER.issue(
        prepare=prepare,
        validate=validate,
        fetch_prepared=fetch_prepared,
        prepare_tool_resume=prepare_tool_resume,
    )


def _claims_final_model_tool_boundary(value: Any) -> bool:
    return getattr(value, "boundary_kind", None) == _FINAL_MODEL_TOOL_BOUNDARY_KIND


def _is_authentic_final_model_tool_boundary(value: Any) -> bool:
    return _FINAL_MODEL_TOOL_BOUNDARY_ISSUER.record_for(value) is not None


__all__ = [
    "FinalModelToolBoundary",
    "FinalModelToolBoundaryContext",
    "FinalModelToolIdentity",
    "FinalModelToolPreparation",
]
