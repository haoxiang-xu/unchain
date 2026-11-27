from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from unchain.journal import (
    AttemptRef,
    BoundExecutionJournal,
    DurableEventSink,
    GenerationRef,
)
from unchain.kernel.harness import HarnessContext
from unchain.run_bundle_ledger import RunBundleLedger
from unchain.tools.output_management import ToolOutputManager

from .artifacts import ArtifactService
from .coordinator import ContextCompileCoordinator
from .handoff import DurableHandoffRecorder, HandoffService
from .ingress import (
    ContextInputIngress,
    HostResolvedCurrentInput,
    HostResolvedInteractionInput,
)
from .projector import (
    CanonicalSemanticEventProjector,
    SemanticEventProjectionMode,
)
from .provider_execution import ContextProviderTurnExecutionService
from .request_factory import JournalContextRequestFactory
from .tool_boundary import DurableToolBoundary


class ContextExecutionBundleError(RuntimeError):
    """An attempt-scoped Context V2 bundle is missing a durable invariant."""


class ContextGenerationResolver(Protocol):
    def __call__(
        self,
        context: HarnessContext,
        execution_id: str,
    ) -> str:
        ...


class ContextExecutionBundleBuilder(Protocol):
    def __call__(self, attempt: AttemptRef) -> ContextExecutionBundle:
        ...


class ContextCurrentInputResolver(Protocol):
    def __call__(
        self,
        context: HarnessContext,
        attempt: AttemptRef,
    ) -> HostResolvedCurrentInput | HostResolvedInteractionInput | None:
        ...


@dataclass(frozen=True)
class ContextExecutionBundle:
    """Inseparable attempt-scoped journal, compiler, and object capabilities."""

    attempt: AttemptRef
    journal: BoundExecutionJournal
    projector: CanonicalSemanticEventProjector
    durable_event_sink: DurableEventSink
    coordinator: ContextCompileCoordinator
    artifacts: ArtifactService
    handoffs: HandoffService
    ingress: ContextInputIngress
    request_factory: JournalContextRequestFactory
    tool_boundary: DurableToolBoundary
    handoff_recorder: DurableHandoffRecorder
    partial_attempt_sink: Callable[[dict[str, Any], Exception], None]
    provider_turn_service: ContextProviderTurnExecutionService | None = None
    run_bundle_ledger: RunBundleLedger | None = None
    tool_output_manager: ToolOutputManager = field(
        default_factory=ToolOutputManager.active_default,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.attempt, AttemptRef):
            object.__setattr__(
                self,
                "attempt",
                AttemptRef.from_dict(self.attempt),
            )
        if not isinstance(self.journal, BoundExecutionJournal):
            raise TypeError("journal must be a BoundExecutionJournal")
        if type(self.projector) is not CanonicalSemanticEventProjector:
            raise TypeError(
                "projector must be the official CanonicalSemanticEventProjector"
            )
        if type(self.durable_event_sink) is not DurableEventSink:
            raise TypeError("durable_event_sink must be the official DurableEventSink")
        if type(self.coordinator) is not ContextCompileCoordinator:
            raise TypeError(
                "coordinator must be the official ContextCompileCoordinator"
            )
        if not isinstance(self.artifacts, ArtifactService):
            raise TypeError("artifacts must be an ArtifactService")
        if not isinstance(self.handoffs, HandoffService):
            raise TypeError("handoffs must be a HandoffService")
        if type(self.ingress) is not ContextInputIngress:
            raise TypeError("ingress must be the official ContextInputIngress")
        if type(self.request_factory) is not JournalContextRequestFactory:
            raise TypeError(
                "request_factory must be the official " "JournalContextRequestFactory"
            )
        if type(self.tool_boundary) is not DurableToolBoundary:
            raise TypeError("tool_boundary must be the official DurableToolBoundary")
        if type(self.handoff_recorder) is not DurableHandoffRecorder:
            raise TypeError(
                "handoff_recorder must be the official " "DurableHandoffRecorder"
            )
        if not callable(self.partial_attempt_sink):
            raise TypeError("partial_attempt_sink must be callable")
        if (
            self.provider_turn_service is not None
            and type(self.provider_turn_service)
            is not ContextProviderTurnExecutionService
        ):
            raise TypeError(
                "provider_turn_service must be the official "
                "ContextProviderTurnExecutionService or null"
            )
        if self.run_bundle_ledger is None and isinstance(
            self.journal,
            RunBundleLedger,
        ):
            object.__setattr__(self, "run_bundle_ledger", self.journal)
        if self.run_bundle_ledger is not None and not isinstance(
            self.run_bundle_ledger,
            RunBundleLedger,
        ):
            raise TypeError(
                "run_bundle_ledger must implement the RunBundleLedger protocol "
                "or be null"
            )
        if type(self.tool_output_manager) is not ToolOutputManager:
            raise TypeError("tool_output_manager must be the official ToolOutputManager")
        execution_id = self.attempt.generation.execution_id
        if any(
            scope != execution_id
            for scope in (
                self.journal.execution_id,
                self.artifacts.execution_id,
                self.handoffs.execution_id,
            )
        ):
            raise ContextExecutionBundleError(
                "bundle capabilities do not share the attempt execution"
            )
        if self.projector.attempt != self.attempt:
            raise ContextExecutionBundleError(
                "projector does not share the bundle attempt"
            )
        if self.durable_event_sink.attempt != self.attempt:
            raise ContextExecutionBundleError(
                "durable sink does not share the bundle attempt"
            )
        if (
            self.durable_event_sink.journal is not self.journal
            or self.coordinator.journal is not self.journal
        ):
            raise ContextExecutionBundleError(
                "durable sink and coordinator must share the same journal"
            )
        model_projection = self.coordinator.model_projection
        if self.coordinator.node_memory_view != self.request_factory.node_memory_view:
            raise ContextExecutionBundleError(
                "coordinator and request factory must share the same node memory view"
            )
        if (
            model_projection is not None
            and model_projection.artifacts is not self.artifacts
        ):
            raise ContextExecutionBundleError(
                "model projection and bundle must share the same artifacts"
            )
        if self.durable_event_sink.projector is not self.projector:
            raise ContextExecutionBundleError(
                "durable sink and bundle must share the same projector"
            )
        if self.projector.artifacts is not self.artifacts:
            raise ContextExecutionBundleError(
                "projector and bundle must share the same artifact service"
            )
        if self.handoffs.artifacts is not self.artifacts:
            raise ContextExecutionBundleError(
                "handoff service and bundle must share the same artifacts"
            )
        if (
            self.ingress.attempt != self.attempt
            or self.ingress.projector is not self.projector
            or self.ingress.sink is not self.durable_event_sink
        ):
            raise ContextExecutionBundleError(
                "input ingress does not share the exact bundle boundary"
            )
        if (
            self.request_factory.attempt != self.attempt
            or self.request_factory.journal is not self.journal
        ):
            raise ContextExecutionBundleError(
                "request factory does not share the exact bundle journal"
            )
        if (
            self.tool_boundary.attempt != self.attempt
            or self.tool_boundary.projector is not self.projector
            or self.tool_boundary.sink is not self.durable_event_sink
        ):
            raise ContextExecutionBundleError(
                "tool boundary does not share the exact bundle boundary"
            )
        if (
            self.handoff_recorder.attempt != self.attempt
            or self.handoff_recorder.handoffs is not self.handoffs
            or self.handoff_recorder.projector is not self.projector
            or self.handoff_recorder.sink is not self.durable_event_sink
        ):
            raise ContextExecutionBundleError(
                "handoff recorder does not share the exact bundle boundary"
            )
        if self.provider_turn_service is not None and (
            self.provider_turn_service.attempt != self.attempt
            or self.provider_turn_service.store is not self.journal
        ):
            raise ContextExecutionBundleError(
                "provider turn service does not share the exact bundle boundary"
            )
        if (
            self.run_bundle_ledger is not None
            and self.run_bundle_ledger.execution_id != execution_id
        ):
            raise ContextExecutionBundleError(
                "run bundle ledger does not share the attempt execution"
            )

    def bootstrap(
        self,
        current_input: (HostResolvedCurrentInput | HostResolvedInteractionInput | None),
    ) -> None:
        if current_input is not None:
            self.ingress.persist(current_input)


class DurableContextRuntimeFactory:
    """Create one verified durable bundle after bootstrap resolves the run ID."""

    def __init__(
        self,
        *,
        bundle_builder: ContextExecutionBundleBuilder,
        generation_resolver: ContextGenerationResolver,
        current_input_resolver: ContextCurrentInputResolver,
        execution_id_resolver: Callable[[HarnessContext, str], str] | None = None,
        projection_mode: SemanticEventProjectionMode = (
            SemanticEventProjectionMode.CANONICAL
        ),
    ) -> None:
        if not callable(bundle_builder):
            raise TypeError("bundle_builder must be callable")
        if not callable(generation_resolver):
            raise TypeError("generation_resolver must be callable")
        if not callable(current_input_resolver):
            raise TypeError("current_input_resolver must be callable")
        if execution_id_resolver is not None and not callable(execution_id_resolver):
            raise TypeError("execution_id_resolver must be callable or null")
        if not isinstance(projection_mode, SemanticEventProjectionMode):
            raise TypeError("projection_mode must be a SemanticEventProjectionMode")
        self._bundle_builder = bundle_builder
        self._generation_resolver = generation_resolver
        self._current_input_resolver = current_input_resolver
        self._execution_id_resolver = execution_id_resolver
        self._projection_mode = projection_mode
        self._lock = threading.RLock()
        self._bundles: dict[tuple[str, str], ContextExecutionBundle] = {}

    @property
    def projection_mode(self) -> SemanticEventProjectionMode:
        return self._projection_mode

    def execution_id_for_context(self, context: HarnessContext) -> str:
        """Return the root journal execution bound to a live model session."""
        if not isinstance(context, HarnessContext):
            raise TypeError("context must be a HarnessContext")
        attempt_id = str(context.event.get("run_id") or "").strip()
        session_id = str(context.state.session_state.session_id or attempt_id).strip()
        if not session_id:
            raise ContextExecutionBundleError(
                "bootstrap did not provide a stable execution ID"
            )
        execution_id = (
            str(self._execution_id_resolver(context, session_id) or "").strip()
            if self._execution_id_resolver is not None else session_id
        )
        if not execution_id:
            raise ContextExecutionBundleError(
                "execution ID resolver did not return a stable execution ID"
            )
        return execution_id

    def bind(self, context: HarnessContext) -> ContextExecutionBundle:
        if not isinstance(context, HarnessContext):
            raise TypeError("context must be a HarnessContext")
        if context.phase != "bootstrap":
            raise ContextExecutionBundleError(
                "attempt bundles may only be created during bootstrap"
            )
        attempt_id = str(context.event.get("run_id") or "").strip()
        if not attempt_id:
            raise ContextExecutionBundleError(
                "bootstrap did not provide a stable attempt ID"
            )
        execution_id = self.execution_id_for_context(context)
        generation_id = str(
            self._generation_resolver(context, execution_id) or ""
        ).strip()
        if not generation_id:
            raise ContextExecutionBundleError(
                "generation resolver did not return a stable generation ID"
            )
        try:
            attempt = AttemptRef(
                generation=GenerationRef(execution_id, generation_id),
                attempt_id=attempt_id,
            )
        except (TypeError, ValueError) as exc:
            raise ContextExecutionBundleError(
                "bootstrap attempt binding is invalid"
            ) from exc
        return self._bind_exact_attempt(attempt)

    def _bind_exact_attempt(
        self,
        attempt: AttemptRef,
    ) -> ContextExecutionBundle:
        """Build or recover one bundle from an already-proven attempt."""

        if not isinstance(attempt, AttemptRef):
            attempt = AttemptRef.from_dict(attempt)
        key = (
            attempt.generation.execution_id,
            attempt.attempt_id,
        )
        with self._lock:
            existing = self._bundles.get(key)
            if existing is not None:
                if existing.attempt != attempt:
                    raise ContextExecutionBundleError(
                        "cached bundle changed the resolved attempt binding"
                    )
                return existing
            bundle = self._bundle_builder(attempt)
            if not isinstance(bundle, ContextExecutionBundle):
                raise TypeError("bundle_builder must return ContextExecutionBundle")
            if bundle.attempt != attempt:
                raise ContextExecutionBundleError(
                    "bundle builder changed the resolved attempt binding"
                )
            if bundle.projector.projection_mode is not self._projection_mode:
                raise ContextExecutionBundleError(
                    "bundle projector changed the declared projection mode"
                )
            self._bundles[key] = bundle
            return bundle

    def bootstrap(
        self,
        bundle: ContextExecutionBundle,
        context: HarnessContext,
    ) -> None:
        if not isinstance(bundle, ContextExecutionBundle):
            raise TypeError("bundle must be a ContextExecutionBundle")
        current_input = self._current_input_resolver(context, bundle.attempt)
        if current_input is not None and not isinstance(
            current_input,
            (HostResolvedCurrentInput, HostResolvedInteractionInput),
        ):
            raise TypeError(
                "current_input_resolver must return a host-resolved user or "
                "interaction input, or None"
            )
        bundle.bootstrap(current_input)


__all__ = [
    "ContextExecutionBundle",
    "ContextExecutionBundleBuilder",
    "ContextExecutionBundleError",
    "ContextCurrentInputResolver",
    "ContextGenerationResolver",
    "DurableContextRuntimeFactory",
]
