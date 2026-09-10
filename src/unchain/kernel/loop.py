from __future__ import annotations

import copy
import uuid
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from ..execution import (
    ExecutionGuard,
    ExecutionLeaseNotOwnedError,
    ExecutionRuntime,
)
from ..interaction.runtime import (
    DurableInteractionRuntime,
    require_unapplied_callback_receipt,
)
from ..interaction.adapters import DurableMaxBudgetCallbackAdapter
from ..interaction.durable import (
    INTERACTION_KIND_HUMAN_INPUT,
    INTERACTION_KIND_MAX_BUDGET,
    INTERACTION_KIND_TOOL_APPROVAL,
    InteractionIntegrityError,
    InteractionNotPendingError,
)
from ..interaction.requests import ensure_interaction_runtime_matches
from ..interaction.effects import (
    HUMAN_INPUT_CONTINUATION_TYPE,
)
from ..providers.model_turn_runtime import (
    apply_model_turn_result,
    build_model_turn_request,
    fetch_built_model_turn,
    fetch_model_turn,
)
from ..retry import RetryConfig
from ..schemas import ResponseFormat
from ..tools.toolkit import Toolkit
from ..tools.models import ToolConfirmationRequest
from ..tools.runtime import snapshot_durable_tool_exposure_plan
from ..tools.types import ToolBatchState
from .delta import HarnessDelta
from .harness import HarnessContext, RuntimeHarness, RuntimePhase
from .failure import attach_kernel_run_failure
from .lifecycle_events import (
    build_iteration_completed_payload,
    build_iteration_started_payload,
    build_response_received_payload,
    build_run_started_payload,
)
from .model_io import ModelIO
from .model_tool_boundary import (
    FinalModelToolBoundary,
    FinalModelToolPreparation,
    _bind_final_model_tool_boundary,
    _claims_final_model_tool_boundary,
    _resolve_final_model_tool_boundary,
    _snapshot_final_model_tool_boundary_context,
)
from .provider_replay import set_provider_replay_frame
from .run_ledger import (
    build_model_attempt_receipt,
    initialize_run_ledger,
    materialize_state_bundle,
    provider_call_route,
    record_model_turn,
    record_unobserved_model_attempt,
    request_sha256,
)
from .run_limits import resolve_max_iterations_boundary
from .run_outcomes import (
    finish_awaiting_interaction_run,
    finish_awaiting_human_input_run,
    finish_completed_run,
    finish_max_iterations_run,
)
from .run_preparation import (
    infer_model,
    infer_provider,
    prepare_fresh_run_invocation,
    prepare_resume_run_invocation,
    prepare_state_for_execution,
)
from .state import RunState
from .types import KernelRunResult, ModelTurnResult, ToolCall

if TYPE_CHECKING:
    from ..run_bundle import ProviderCallReceipt, RunDescriptor, RunIdentity
    from ..runtime.module_context import AgentRuntimeContext


_DURABLE_BARRIER_PHASES = frozenset({"suspend_persist", "finalize_persist"})


class KernelLoop:
    """Minimal harness-driven loop skeleton for the new kernel."""

    def __init__(
        self,
        *,
        harnesses: list[RuntimeHarness | FinalModelToolBoundary] | None = None,
        model_io: ModelIO | None = None,
        retry_config: RetryConfig | None = None,
        execution_runtime: ExecutionRuntime | None = None,
        interaction_runtime: DurableInteractionRuntime | None = None,
    ) -> None:
        self._harnesses: list[RuntimeHarness] = []
        self._model_io = model_io
        self._retry_config: RetryConfig = (
            retry_config if retry_config is not None else RetryConfig()
        )
        self._execution_runtime = execution_runtime
        self._interaction_runtime = interaction_runtime
        for harness in harnesses or []:
            self.register_harness(harness)

    @property
    def harnesses(self) -> list[RuntimeHarness]:
        return list(self._harnesses)

    @property
    def retry_config(self) -> RetryConfig:
        return self._retry_config

    @property
    def final_model_tool_boundary(self) -> FinalModelToolBoundary | None:
        return _resolve_final_model_tool_boundary(self)

    def register_harness(
        self,
        harness: RuntimeHarness | FinalModelToolBoundary,
    ) -> None:
        claims_final_boundary = _claims_final_model_tool_boundary(harness)
        if isinstance(harness, FinalModelToolBoundary):
            if not claims_final_boundary:
                raise TypeError("invalid final model boundary registration")
            if tuple(getattr(harness, "phases", ())):
                raise TypeError(
                    "final model boundary cannot declare ordinary harness phases"
                )
            _bind_final_model_tool_boundary(self, harness)
            return
        if claims_final_boundary:
            raise TypeError("forged final model boundary registration")

        harness_phases = set(getattr(harness, "phases", ()))
        reserved_phases = harness_phases & _DURABLE_BARRIER_PHASES
        if reserved_phases and getattr(harness, "durable_barrier", False) is not True:
            phase_list = ", ".join(sorted(reserved_phases))
            raise ValueError(
                f"harness '{harness.name}' cannot register reserved durable "
                f"phase(s): {phase_list}"
            )
        for phase in reserved_phases:
            existing = next(
                (
                    item
                    for item in self._harnesses
                    if phase in getattr(item, "phases", ())
                ),
                None,
            )
            if existing is not None:
                raise ValueError(
                    f"durable phase '{phase}' already belongs to harness "
                    f"'{existing.name}'"
                )
        self._harnesses.append(harness)
        self._harnesses.sort(key=lambda item: (item.order, item.name))

    def register_context_optimizer(self, optimizer: RuntimeHarness) -> None:
        semantic_owner = getattr(self, "_semantic_context_owner", None)
        if semantic_owner is not None:
            raise ValueError(
                "context optimizers cannot be registered after a semantic "
                f"context owner is active: {semantic_owner!r}"
            )
        self.register_harness(optimizer)

    @property
    def model_io(self) -> ModelIO | None:
        return self._model_io

    @property
    def execution_runtime(self) -> ExecutionRuntime | None:
        return self._execution_runtime

    @property
    def interaction_runtime(self) -> DurableInteractionRuntime | None:
        return self._interaction_runtime

    @interaction_runtime.setter
    def interaction_runtime(self, value: DurableInteractionRuntime | None) -> None:
        self._interaction_runtime = value

    @model_io.setter
    def model_io(self, value: ModelIO | None) -> None:
        self._model_io = value

    def _validate_execution_guard(
        self,
        state: RunState,
        execution_guard: ExecutionGuard | None,
    ) -> ExecutionGuard | None:
        session_id = str(state.session_state.session_id or "")
        if execution_guard is None:
            if self._execution_runtime is not None and session_id:
                raise ExecutionLeaseNotOwnedError(
                    "execution-lease-enabled KernelLoop requires an active guard",
                    execution_id=session_id,
                )
            return None
        if session_id and execution_guard.lease.execution_id != session_id:
            raise ValueError(
                "execution guard does not belong to the RunState session_id"
            )
        execution_guard.assert_active()
        return execution_guard

    def _scope_for_session(
        self,
        *,
        session_id: str | None,
        execution_guard: ExecutionGuard | None,
    ):
        normalized_session_id = str(session_id or "")
        if execution_guard is not None:
            if (
                normalized_session_id
                and execution_guard.lease.execution_id != normalized_session_id
            ):
                raise ValueError(
                    "execution guard does not belong to the requested session_id"
                )
            execution_guard.assert_active()
            return nullcontext(execution_guard)
        if self._execution_runtime is not None and normalized_session_id:
            return self._execution_runtime.scope(normalized_session_id)
        return nullcontext(None)

    def seed_state(
        self,
        messages: list[dict[str, Any]],
        *,
        provider: str | None = None,
        model: str | None = None,
        session_id: str | None = None,
        memory_namespace: str | None = None,
        max_context_window_tokens: int | None = None,
    ) -> RunState:
        state = RunState()
        state.seed_messages(messages)
        state.provider_state.provider = provider
        state.provider_state.model = model
        state.provider_state.max_context_window_tokens = max(
            0, int(max_context_window_tokens or 0)
        )
        state.session_state.session_id = session_id
        state.session_state.memory_namespace = memory_namespace
        return state

    def dispatch_phase(
        self,
        state: RunState,
        *,
        phase: RuntimePhase,
        event: dict[str, Any] | None = None,
    ) -> RunState:
        context = HarnessContext(state=state, phase=phase, event=event or {})
        for harness in self._iter_phase_harnesses(phase):
            if not harness.applies(context):
                continue
            apply = getattr(harness, "apply", None)
            raw_outcome = (
                apply(context) if callable(apply) else harness.build_delta(context)
            )
            if raw_outcome is None:
                continue

            from ..capabilities import RunDelta, normalize_capability_outcome
            from .application import apply_run_delta

            outcome = normalize_capability_outcome(
                raw_outcome,
                created_by=f"harness.{harness.name}",
            )
            delta = outcome.delta
            if delta is None:
                continue
            if not isinstance(delta, RunDelta):
                raise TypeError(
                    f"harness '{harness.name}' returned {type(delta).__name__}, expected RunDelta"
                )

            def emit_structured_event(op):
                payload = op.payload if isinstance(op.payload, dict) else {}
                raw_iteration = (event or {}).get("iteration", state.iteration)
                try:
                    iteration = int(raw_iteration)
                except (TypeError, ValueError):
                    iteration = int(state.iteration)
                self.emit_event(
                    (event or {}).get("callback"),
                    op.type,
                    str((event or {}).get("run_id") or "kernel"),
                    iteration=iteration,
                    **copy.deepcopy(payload),
                )

            apply_run_delta(state, delta, emit_event=emit_structured_event)
            context = HarnessContext(state=state, phase=phase, event=event or {})
        return state

    def fetch_model_turn(
        self,
        state: RunState,
        *,
        payload: dict[str, Any] | None = None,
        toolkit: Toolkit | None = None,
        callback: Any = None,
        verbose: bool = False,
        run_id: str = "kernel",
        emit_stream: bool = False,
        response_format: Any = None,
        openai_text_format: dict[str, Any] | None = None,
        execution_guard: ExecutionGuard | None = None,
        provider_attempt_callback: Callable[[int], None] | None = None,
        provider_attempt_completed_callback: Callable[
            [int, str, str, str], None
        ] | None = None,
    ):
        def before_attempt(attempt: int) -> None:
            if execution_guard is not None:
                execution_guard.renew()
            if provider_attempt_callback is not None:
                provider_attempt_callback(attempt)

        def after_attempt(
            attempt: int,
            completed_at: str,
            outcome: str,
            classification: str,
        ) -> None:
            if provider_attempt_completed_callback is not None:
                provider_attempt_completed_callback(
                    attempt,
                    completed_at,
                    outcome,
                    classification,
                )

        for attribute in ("run_receipt_factory", "run_receipt_observed"):
            value = getattr(provider_attempt_callback, attribute, None)
            if value is not None:
                setattr(before_attempt, attribute, value)

        return fetch_model_turn(
            model_io=self._model_io,
            retry_config=self._retry_config,
            state=state,
            payload=payload,
            toolkit=toolkit,
            callback=callback,
            verbose=verbose,
            run_id=run_id,
            emit_stream=emit_stream,
            response_format=response_format,
            openai_text_format=openai_text_format,
            before_attempt=(
                before_attempt
                if execution_guard is not None
                or provider_attempt_callback is not None
                else None
            ),
            after_attempt=(
                after_attempt
                if provider_attempt_completed_callback is not None
                else None
            ),
        )

    def apply_model_turn(
        self,
        state: RunState,
        turn: ModelTurnResult,
        *,
        created_by: str = "kernel.model_turn",
    ) -> RunState:
        return apply_model_turn_result(state, turn, created_by=created_by)

    def step_once(
        self,
        state: RunState,
        *,
        payload: dict[str, Any] | None = None,
        toolkit: Toolkit | None = None,
        callback: Any = None,
        verbose: bool = False,
        run_id: str = "kernel",
        emit_stream: bool = False,
        response_format: Any = None,
        openai_text_format: dict[str, Any] | None = None,
        on_tool_confirm: Any = None,
        on_human_input: Any = None,
        max_iterations: int = 0,
        tool_runtime_plugins: list[Any] | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        execution_guard: ExecutionGuard | None = None,
        run_bundle_purpose: str = "agent_turn",
    ) -> ModelTurnResult:
        execution_guard = self._validate_execution_guard(state, execution_guard)
        if state.run_ledger.identity is None:
            initialize_run_ledger(state, run_id=str(run_id or "kernel"))
        runtime_toolkit = toolkit if toolkit is not None else Toolkit()
        current_iteration = int(state.iteration)
        local_next_context = (
            state.next_model_input
            if isinstance(state.next_model_input, list)
            else state.transcript
        )
        state.rebuild_working_version(
            local_next_context,
            created_by="kernel.model_context_snapshot",
            metadata={
                "iteration": current_iteration,
                "transcript_message_count": len(state.transcript),
                "source": (
                    "next_model_input"
                    if local_next_context is state.next_model_input
                    else "transcript"
                ),
            },
        )
        phase_event = {
            "payload": dict(payload or {}),
            "toolkit": runtime_toolkit,
            "callback": callback,
            "verbose": verbose,
            "run_id": run_id,
            "emit_stream": emit_stream,
            "response_format": response_format,
            "openai_text_format": openai_text_format,
            "on_tool_confirm": on_tool_confirm,
            "on_human_input": on_human_input,
            "max_iterations": max_iterations,
            "supports_tools": True,
            "loop": self,
            "model_io": self._model_io,
            "tool_runtime_plugins": list(tool_runtime_plugins or []),
            "tool_runtime_config": copy.deepcopy(tool_runtime_config or {}),
            "execution_guard": execution_guard,
        }
        final_boundary = self.final_model_tool_boundary
        retry_ordinal = 0
        provider_attempts: list[dict[str, Any]] = []
        request_digest: str | None = None

        def provider_attempt_started(attempt: int) -> None:
            nonlocal retry_ordinal
            # One callback invocation means one transport send is about to
            # happen.  Use a process-local monotonic ordinal so a boundary
            # hand-off cannot reuse retry 0 for a second physical send.
            source_attempt = int(attempt)
            if source_attempt < 0:
                raise ValueError("provider attempt ordinal cannot be negative")
            retry_ordinal = max(len(provider_attempts), source_attempt)
            occurred_at = (
                datetime.now(timezone.utc)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            provider_attempts.append(
                {
                    "source_attempt": source_attempt,
                    "retry_ordinal": retry_ordinal,
                    "started_at": occurred_at,
                    "completed_at": None,
                    "outcome": None,
                    "classification": None,
                }
            )

        def provider_attempt_completed(
            attempt: int,
            completed_at: str,
            outcome: str,
            classification: str,
        ) -> None:
            matches = [
                item
                for item in provider_attempts
                if item["source_attempt"] == int(attempt)
                and item["completed_at"] is None
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "provider attempt completion does not match one open send"
                )
            if outcome not in {"completed", "failed", "uncertain"}:
                raise RuntimeError("provider attempt outcome is unsupported")
            matches[0]["completed_at"] = completed_at
            matches[0]["outcome"] = outcome
            matches[0]["classification"] = classification

        def bind_atomic_run_receipt(callback: Callable[[int], None]) -> None:
            if request_digest is None or state.run_ledger.identity is None:
                return
            resolved_provider = str(
                state.provider_state.provider or "unknown"
            ).strip().lower()
            resolved_model = str(
                state.provider_state.model or "unknown-model"
            ).strip()

            def factory(
                attempt_number: int,
                started_at: str,
                completed_at: str,
                outcome: str,
                classification: str,
                result: ModelTurnResult | None,
            ) -> ProviderCallReceipt:
                return build_model_attempt_receipt(
                    identity=state.run_ledger.identity,
                    provider=resolved_provider,
                    model=resolved_model,
                    iteration=current_iteration,
                    retry_ordinal=attempt_number,
                    purpose=run_bundle_purpose,
                    request_digest=request_digest,
                    route=provider_call_route(resolved_provider),
                    payload=payload,
                    started_at=started_at,
                    completed_at=completed_at,
                    turn=result,
                    status=outcome,
                    classification=classification,
                )

            setattr(callback, "run_receipt_factory", factory)
            setattr(callback, "run_receipt_observed", state.run_ledger.append)

        if execution_guard is not None:
            execution_guard.renew()
        self.dispatch_phase(state, phase="before_model", event=phase_event)
        if execution_guard is not None:
            execution_guard.assert_active()
        if self.final_model_tool_boundary is not final_boundary:
            raise RuntimeError("final model boundary changed during before_model")

        preparation: FinalModelToolPreparation | None = None
        try:
            if final_boundary is not None:
                prepare_context = _snapshot_final_model_tool_boundary_context(
                    state=state,
                    event=phase_event,
                )
                preparation = final_boundary.prepare(prepare_context)
                if type(preparation) is not FinalModelToolPreparation:
                    raise TypeError(
                        "final model boundary must return exact "
                        "FinalModelToolPreparation"
                    )
                if not isinstance(preparation.model_toolkit, Toolkit):
                    raise TypeError(
                        "final model boundary preparation requires a Toolkit"
                    )
                if not isinstance(preparation.execution_toolkit, Toolkit):
                    raise TypeError(
                        "final model boundary preparation requires an "
                        "execution Toolkit"
                    )
                if preparation.execution_binding is None:
                    raise TypeError(
                        "final model boundary preparation requires a sealed "
                        "execution binding"
                    )
                phase_event["toolkit"] = preparation.execution_toolkit
                phase_event[
                    "tool_execution_binding"
                ] = preparation.execution_binding
                if preparation.prepared_provider_turn is not None:
                    raise RuntimeError(
                        "prepared provider turn has no authenticated provider "
                        "consumer"
                    )
                if execution_guard is not None:
                    execution_guard.assert_active()
                request = build_model_turn_request(
                    state,
                    payload=payload,
                    toolkit=preparation.model_toolkit,
                    callback=callback,
                    verbose=verbose,
                    run_id=run_id,
                    emit_stream=emit_stream,
                    response_format=response_format,
                    openai_text_format=openai_text_format,
                )
                request_digest = request_sha256(
                    state=state,
                    payload=payload,
                    toolkit=preparation.model_toolkit,
                    response_format=response_format,
                    openai_text_format=openai_text_format,
                    provider=str(state.provider_state.provider or "unknown"),
                    model=str(state.provider_state.model or "unknown-model"),
                )

                def before_attempt(attempt: int) -> None:
                    if execution_guard is not None:
                        execution_guard.renew()
                    provider_attempt_started(attempt)

                bind_atomic_run_receipt(before_attempt)

                turn = final_boundary.fetch_prepared(
                    prepare_context,
                    preparation,
                    request,
                    retry_config=self._retry_config,
                    before_attempt=before_attempt,
                    after_attempt=provider_attempt_completed,
                )
                if turn is None:
                    turn = fetch_built_model_turn(
                        model_io=self._model_io,
                        retry_config=self._retry_config,
                        state=state,
                        request=request,
                        before_attempt=before_attempt,
                        after_attempt=provider_attempt_completed,
                    )
                elif type(turn) is not ModelTurnResult:
                    raise TypeError(
                        "final model boundary prepared fetch must return exact "
                        "ModelTurnResult or None"
                    )
            else:
                raw_request_digest = request_sha256(
                    state=state,
                    payload=payload,
                    toolkit=runtime_toolkit,
                    response_format=response_format,
                    openai_text_format=openai_text_format,
                    provider=str(state.provider_state.provider or "unknown"),
                    model=str(state.provider_state.model or "unknown-model"),
                )
                provider_turn_ownership = (
                    state.run_ledger.provider_turn_ownership
                )
                if provider_turn_ownership is not None:
                    occurrence_id = (
                        f"{run_bundle_purpose}:{max(0, int(current_iteration))}"
                    )
                    request_digest = (
                        provider_turn_ownership.logical_request_sha256(
                            request_sha256=raw_request_digest,
                            occurrence_id=occurrence_id,
                        )
                    )
                    request = build_model_turn_request(
                        state,
                        payload=payload,
                        toolkit=runtime_toolkit,
                        callback=callback,
                        verbose=verbose,
                        run_id=run_id,
                        emit_stream=emit_stream,
                        response_format=response_format,
                        openai_text_format=openai_text_format,
                    )

                    def before_owned_attempt(attempt: int) -> None:
                        if execution_guard is not None:
                            execution_guard.renew()
                        provider_attempt_started(attempt)

                    turn = provider_turn_ownership.fetch_turn(
                        state=state,
                        model_io=self._model_io,
                        request=request,
                        occurrence_id=occurrence_id,
                        purpose=run_bundle_purpose,
                        iteration=current_iteration,
                        request_sha256=raw_request_digest,
                        retry_config=self._retry_config,
                        before_attempt=before_owned_attempt,
                        after_attempt=provider_attempt_completed,
                        provider=str(
                            state.provider_state.provider or "unknown"
                        ),
                        model=str(
                            state.provider_state.model or "unknown-model"
                        ),
                    )
                else:
                    request_digest = raw_request_digest
                    bind_atomic_run_receipt(provider_attempt_started)
                    turn = self.fetch_model_turn(
                        state,
                        payload=payload,
                        toolkit=runtime_toolkit,
                        callback=callback,
                        verbose=verbose,
                        run_id=run_id,
                        emit_stream=emit_stream,
                        response_format=response_format,
                        openai_text_format=openai_text_format,
                        execution_guard=execution_guard,
                        provider_attempt_callback=provider_attempt_started,
                        provider_attempt_completed_callback=(
                            provider_attempt_completed
                        ),
                    )
            if execution_guard is not None:
                execution_guard.assert_active()
            if final_boundary is not None:
                if preparation is None:
                    raise RuntimeError("final model boundary preparation is missing")
                validate_context = _snapshot_final_model_tool_boundary_context(
                    state=state,
                    event=phase_event,
                )
                turn = final_boundary.validate(
                    validate_context,
                    preparation,
                    turn,
                )
                if type(turn) is not ModelTurnResult:
                    raise TypeError(
                        "final model boundary must return exact ModelTurnResult"
                    )
                if execution_guard is not None:
                    execution_guard.assert_active()
        except BaseException:
            if request_digest is not None:
                # A durable result CAS may have committed its accounting fact
                # before a later lease-finalization failure surfaced. Refresh
                # the in-memory exact-once index before projecting the failed
                # run so we never manufacture a conflicting partial receipt.
                if state.run_ledger.persistence is not None:
                    state.run_ledger.attach_persistence(
                        state.run_ledger.persistence
                    )
                for attempt in provider_attempts:
                    outcome = (
                        attempt["outcome"]
                        if attempt["completed_at"] is not None
                        else "uncertain"
                    )
                    record_unobserved_model_attempt(
                        state,
                        iteration=current_iteration,
                        retry_ordinal=attempt["retry_ordinal"],
                        purpose=run_bundle_purpose,
                        request_digest=request_digest,
                        payload=payload,
                        started_at=attempt["started_at"],
                        completed_at=attempt["completed_at"],
                        route=provider_call_route(
                            str(state.provider_state.provider or "unknown")
                        ),
                        status=outcome,
                    )
            raise
        completed_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        if provider_attempts and provider_attempts[-1]["completed_at"] is None:
            # Legacy/custom fetchers may not yet report the optional completion
            # callback.  A successful return closes only the final attempt;
            # prior retry attempts remain explicitly uncertain with null end.
            provider_attempts[-1]["completed_at"] = completed_at
            provider_attempts[-1]["outcome"] = "completed"
            provider_attempts[-1]["classification"] = "legacy_return"
        for attempt in provider_attempts[:-1]:
            outcome = (
                attempt["outcome"]
                if attempt["completed_at"] is not None
                else "uncertain"
            )
            record_unobserved_model_attempt(
                state,
                iteration=current_iteration,
                retry_ordinal=attempt["retry_ordinal"],
                purpose=run_bundle_purpose,
                request_digest=request_digest,
                payload=payload,
                started_at=attempt["started_at"],
                completed_at=attempt["completed_at"],
                route=provider_call_route(
                    str(state.provider_state.provider or "unknown")
                ),
                status=outcome,
            )
        provider_call_started_at = (
            provider_attempts[-1]["started_at"] if provider_attempts else None
        )
        provider_call_completed_at = (
            provider_attempts[-1]["completed_at"] if provider_attempts else None
        )
        record_model_turn(
            state,
            turn,
            iteration=current_iteration,
            retry_ordinal=retry_ordinal,
            purpose=run_bundle_purpose,
            request_digest=request_digest,
            payload=payload,
            started_at=provider_call_started_at,
            completed_at=provider_call_completed_at,
            route=provider_call_route(
                str(state.provider_state.provider or "unknown")
            ),
        )
        if turn.provider_call_receipt is not None:
            # The durable accounting fact has crossed into RunLedger.  Keep it
            # out of transcript/state deltas, whose deep-copy semantics are
            # intentionally limited to model-visible provider data.
            turn = replace(turn, provider_call_receipt=None)
        self.apply_model_turn(state, turn)

        after_model_event = {
            **phase_event,
            "turn_result": turn,
        }
        self.dispatch_phase(state, phase="after_model", event=after_model_event)
        if execution_guard is not None:
            execution_guard.assert_active()

        tool_calls = list(state.pending_tool_calls)
        if tool_calls:
            for index, tool_call in enumerate(tool_calls):
                if execution_guard is not None:
                    execution_guard.renew()
                self.dispatch_phase(
                    state,
                    phase="on_tool_call",
                    event={
                        **after_model_event,
                        "tool_call": tool_call,
                        "tool_call_index": index,
                        "tool_calls": tool_calls,
                    },
                )
                if execution_guard is not None:
                    execution_guard.assert_active()
            self.dispatch_phase(
                state,
                phase="after_tool_batch",
                event={
                    **after_model_event,
                    "tool_calls": tool_calls,
                },
            )
            if execution_guard is not None:
                execution_guard.assert_active()
        else:
            if execution_guard is not None:
                execution_guard.renew()
            self.dispatch_phase(state, phase="before_commit", event=after_model_event)
            if execution_guard is not None:
                execution_guard.assert_active()

        state.iteration = current_iteration + 1
        return turn

    def _iter_phase_harnesses(self, phase: RuntimePhase) -> list[RuntimeHarness]:
        return [harness for harness in self._harnesses if phase in harness.phases]

    def _dispatch_bootstrap(
        self,
        state: RunState,
        *,
        payload: dict[str, Any] | None,
        response_format: ResponseFormat | None,
        callback: Any,
        verbose: bool,
        toolkit: Toolkit | None,
        run_id: str,
        resume_mode: bool,
        continuation: dict[str, Any] | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        execution_guard: ExecutionGuard | None = None,
    ) -> None:
        runtime_toolkit = toolkit if toolkit is not None else Toolkit()
        if execution_guard is not None:
            execution_guard.renew()
        self.dispatch_phase(
            state,
            phase="bootstrap",
            event={
                "payload": dict(payload or {}),
                "toolkit": runtime_toolkit,
                "callback": callback,
                "verbose": verbose,
                "run_id": run_id,
                "response_format": response_format,
                "supports_tools": True,
                "resume_mode": resume_mode,
                "continuation": copy.deepcopy(continuation),
                "loop": self,
                "tool_runtime_config": copy.deepcopy(tool_runtime_config or {}),
                "execution_guard": execution_guard,
            },
        )
        if execution_guard is not None:
            execution_guard.assert_active()

    def _dispatch_run_finalizing(
        self,
        state: RunState,
        *,
        callback: Any,
        run_id: str,
        iteration: int,
        status: str,
        execution_guard: ExecutionGuard | None = None,
    ) -> None:
        event = {
            "callback": callback,
            "run_id": run_id,
            "iteration": int(iteration),
            "status": status,
            "loop": self,
            "execution_guard": execution_guard,
        }
        if execution_guard is not None:
            execution_guard.renew()
        self.dispatch_phase(state, phase="run_finalizing", event=event)
        if execution_guard is not None:
            execution_guard.assert_active()
        self.dispatch_phase(state, phase="finalize_persist", event=event)
        if execution_guard is not None:
            execution_guard.assert_active()

    def _dispatch_on_suspend(
        self,
        state: RunState,
        *,
        callback: Any,
        run_id: str,
        iteration: int,
        status: str,
        execution_guard: ExecutionGuard | None = None,
    ) -> None:
        event = {
            "callback": callback,
            "run_id": run_id,
            "iteration": int(iteration),
            "status": status,
            "loop": self,
            "execution_guard": execution_guard,
        }
        if execution_guard is not None:
            execution_guard.renew()
        self.dispatch_phase(state, phase="on_suspend", event=event)
        if execution_guard is not None:
            execution_guard.assert_active()
        self.dispatch_phase(state, phase="suspend_persist", event=event)
        if execution_guard is not None:
            execution_guard.assert_active()

    def _dispatch_on_resume(
        self,
        state: RunState,
        *,
        continuation: dict[str, Any],
        response: Any,
        callback: Any,
        run_id: str,
        execution_guard: ExecutionGuard | None = None,
    ) -> None:
        if execution_guard is not None:
            execution_guard.renew()
        self.dispatch_phase(
            state,
            phase="on_resume",
            event={
                "continuation": copy.deepcopy(continuation),
                "response": copy.deepcopy(response),
                "callback": callback,
                "run_id": run_id,
                "iteration": int(state.iteration),
                "loop": self,
                "execution_guard": execution_guard,
            },
        )
        if execution_guard is not None:
            execution_guard.assert_active()

    def emit_event(
        self,
        callback: Any,
        event_type: str,
        run_id: str,
        *,
        iteration: int,
        **extra: Any,
    ) -> None:
        if callback is None:
            return
        event = {
            "type": event_type,
            "run_id": run_id,
            "iteration": iteration,
        }
        event.update(copy.deepcopy(extra))
        callback(event)

    def _infer_provider(self) -> str | None:
        return infer_provider(self._model_io)

    def _infer_model(self) -> str | None:
        return infer_model(self._model_io)

    @staticmethod
    def _wrap_max_iterations_callback(
        callback: Any,
        *,
        execution_guard: ExecutionGuard | None,
        expected_revision: Callable[[], int | None],
    ) -> Any:
        if execution_guard is None or not callable(callback):
            return callback

        def guarded_callback(decision: Any) -> Any:
            response = callback(decision)
            revision = expected_revision()
            if revision is not None:
                execution_guard.reacquire(expected_revision=revision)
            else:
                execution_guard.assert_active()
            return response

        return guarded_callback

    @staticmethod
    def _guard_terminal_emitter(
        emit_event: Callable[..., None],
        *,
        execution_guard: ExecutionGuard | None,
    ) -> Callable[..., None]:
        if execution_guard is None:
            return emit_event

        def guarded_emit(*args: Any, **kwargs: Any) -> None:
            execution_guard.assert_active()
            emit_event(*args, **kwargs)
            execution_guard.assert_active()

        return guarded_emit

    @staticmethod
    def _tool_confirmation_request_from_interaction(
        interaction_request: dict[str, Any],
    ) -> ToolConfirmationRequest:
        raw = interaction_request.get("payload")
        if not isinstance(raw, dict):
            raise InteractionIntegrityError(
                "tool approval interaction payload must be an object"
            )
        arguments = raw.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        interact_config = raw.get("interact_config")
        if not isinstance(interact_config, (dict, list)):
            interact_config = None
        render_component = raw.get("render_component")
        if not isinstance(render_component, dict):
            render_component = None
        return ToolConfirmationRequest(
            tool_name=str(raw.get("tool_name") or ""),
            call_id=str(raw.get("call_id") or ""),
            arguments=copy.deepcopy(arguments),
            description=str(raw.get("description") or ""),
            interact_type=str(raw.get("interact_type") or "confirmation"),
            interact_config=copy.deepcopy(interact_config),
            render_component=copy.deepcopy(render_component),
        )

    def _dispatch_tool_approval_resume(
        self,
        state: RunState,
        *,
        continuation: dict[str, Any],
        interaction_request: dict[str, Any],
        response: dict[str, Any],
        payload: dict[str, Any] | None,
        response_format: ResponseFormat | None,
        callback: Any,
        verbose: bool,
        on_tool_confirm: Any,
        on_human_input: Any,
        max_iterations: int,
        toolkit: Toolkit,
        run_id: str,
        tool_runtime_plugins: list[Any] | None,
        tool_runtime_config: dict[str, Any] | None,
        execution_guard: ExecutionGuard | None,
    ) -> None:
        if continuation.get("type") != "tool_approval_continuation":
            raise InteractionIntegrityError(
                "tool approval receipt requires a tool_approval_continuation"
            )
        raw_tool_calls = continuation.get("tool_calls")
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            raise InteractionIntegrityError(
                "tool approval continuation requires pending tool calls"
            )
        tool_calls: list[ToolCall] = []
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict):
                raise InteractionIntegrityError(
                    "tool approval continuation contains an invalid tool call"
                )
            call_id = str(raw_call.get("call_id") or "")
            name = str(raw_call.get("name") or "")
            if not call_id or not name:
                raise InteractionIntegrityError(
                    "tool approval continuation tool call requires id and name"
                )
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=copy.deepcopy(raw_call.get("arguments")),
                )
            )
        raw_batch = continuation.get("tool_batch_state")
        if not isinstance(raw_batch, dict):
            raise InteractionIntegrityError(
                "tool approval continuation requires tool_batch_state"
            )
        result_messages = raw_batch.get("result_messages")
        executed_call_ids = raw_batch.get("executed_call_ids")
        if not isinstance(result_messages, list) or not isinstance(
            executed_call_ids, list
        ):
            raise InteractionIntegrityError(
                "tool approval continuation has invalid batch state"
            )
        tool_iteration = int(
            continuation.get("tool_iteration", max(0, int(state.iteration) - 1))
        )
        next_iteration = int(continuation.get("iteration", tool_iteration + 1))
        phase_event = {
            "payload": dict(payload or {}),
            "toolkit": toolkit,
            "callback": callback,
            "verbose": verbose,
            "run_id": run_id,
            "emit_stream": True,
            "response_format": response_format,
            "on_tool_confirm": on_tool_confirm,
            "on_human_input": on_human_input,
            "max_iterations": max_iterations,
            "supports_tools": True,
            "loop": self,
            "model_io": self._model_io,
            "tool_runtime_plugins": list(tool_runtime_plugins or []),
            "tool_runtime_config": copy.deepcopy(tool_runtime_config or {}),
            "execution_guard": execution_guard,
            "turn_result": state.last_model_turn,
            "tool_calls": tool_calls,
        }
        final_boundary = self.final_model_tool_boundary
        if final_boundary is not None:
            if execution_guard is not None:
                execution_guard.assert_active()
            resume_context = _snapshot_final_model_tool_boundary_context(
                state=state,
                event=phase_event,
                iteration=tool_iteration,
            )
            preparation = final_boundary.prepare_tool_resume(
                resume_context,
                continuation=continuation,
                interaction_request=interaction_request,
            )
            if self.final_model_tool_boundary is not final_boundary:
                raise InteractionIntegrityError(
                    "final model boundary changed during tool approval resume"
                )
            if (
                type(preparation) is not FinalModelToolPreparation
                or not isinstance(preparation.execution_toolkit, Toolkit)
                or preparation.execution_binding is None
            ):
                raise InteractionIntegrityError(
                    "final model boundary tool approval resume requires "
                    "authenticated durable tool binding recovery"
                )
            phase_event["toolkit"] = preparation.execution_toolkit
            phase_event["tool_execution_binding"] = preparation.execution_binding
            if execution_guard is not None:
                execution_guard.assert_active()
        state.pending_tool_calls = list(tool_calls)
        state.tool_batch_state = ToolBatchState(
            result_messages=copy.deepcopy(result_messages),
            should_observe=bool(raw_batch.get("should_observe", False)),
            executed_call_ids=[
                str(item) for item in executed_call_ids if isinstance(item, str)
            ],
        )
        state.component_bucket("tools")[
            "tool_batch_state"
        ] = state.tool_batch_state.copy()
        state.run_status = "running"
        state.last_continuation = None
        state.suspend_state.signal_kind = None
        state.suspend_state.payload = {}

        state.iteration = tool_iteration
        paused_call_id = str(continuation.get("paused_call_id") or "")
        try:
            for index, tool_call in enumerate(tool_calls):
                if execution_guard is not None:
                    execution_guard.renew()
                resume_fields = (
                    {
                        "interaction_request": copy.deepcopy(interaction_request),
                        "interaction_response": copy.deepcopy(response),
                    }
                    if tool_call.call_id == paused_call_id
                    else {}
                )
                self.dispatch_phase(
                    state,
                    phase="on_tool_call",
                    event={
                        **phase_event,
                        **resume_fields,
                        "tool_call": tool_call,
                        "tool_call_index": index,
                    },
                )
                if execution_guard is not None:
                    execution_guard.assert_active()
            self.dispatch_phase(
                state,
                phase="after_tool_batch",
                event=phase_event,
            )
            if execution_guard is not None:
                execution_guard.assert_active()
        except Exception:
            state.iteration = tool_iteration
            state.run_ledger.record_iteration_outcome(
                iteration=tool_iteration,
                outcome="failed",
                error_category="tool",
                error_code="tool_resume_failed",
            )
            raise
        else:
            state.iteration = next_iteration

    def _run_state(
        self,
        state: RunState,
        *,
        payload: dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
        callback: Any = None,
        verbose: bool = False,
        max_iterations: int = 6,
        on_tool_confirm: Any = None,
        on_human_input: Any = None,
        on_max_iterations: Any = None,
        toolkit: Toolkit | None = None,
        run_id: str | None = None,
        skip_bootstrap: bool = False,
        tool_runtime_plugins: list[Any] | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        execution_guard: ExecutionGuard | None = None,
        runtime_context: "AgentRuntimeContext | None" = None,
        run_bundle_identity: "RunIdentity | None" = None,
        run_bundle_descriptor: "RunDescriptor | None" = None,
        continued_from_run_id: str | None = None,
        run_bundle_purpose: str = "agent_turn",
        run_bundle_receipts: tuple["ProviderCallReceipt", ...] = (),
        provider_turn_ownership: Any = None,
    ) -> KernelRunResult:
        if self._model_io is None:
            raise RuntimeError("KernelLoop.model_io is not configured")
        prepare_state_for_execution(state, model_io=self._model_io)
        execution_guard = self._validate_execution_guard(state, execution_guard)
        run_id = str(run_id or uuid.uuid4())
        initialize_run_ledger(
            state,
            run_id=run_id,
            runtime_context=runtime_context,
            explicit_identity=run_bundle_identity,
            descriptor=run_bundle_descriptor,
            continued_from_run_id=continued_from_run_id,
        )
        state.run_ledger.bind_provider_turn_ownership(provider_turn_ownership)
        for receipt in run_bundle_receipts:
            state.run_ledger.append(receipt)
        runtime_toolkit = toolkit if toolkit is not None else Toolkit()
        if not skip_bootstrap:
            self._dispatch_bootstrap(
                state,
                payload=payload,
                response_format=response_format,
                callback=callback,
                verbose=verbose,
                toolkit=runtime_toolkit,
                run_id=run_id,
                resume_mode=False,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
            )
        state.run_ledger.assert_continuation_verified()
        self.emit_event(
            callback,
            "run_started",
            run_id,
            **build_run_started_payload(state),
        )
        effective_max = int(max_iterations)
        if (
            state.memory_state.get("execution_checkpoint_restored") is True
            and state.memory_state.get("execution_checkpoint_status")
            == "max_iterations"
        ):
            # A fresh run that restores a max-iterations checkpoint treats its
            # limit as budget for this invocation.  ``state.iteration`` remains
            # cumulative so telemetry and checkpoint diagnostics stay honest.
            effective_max += int(state.iteration)
        terminal_emit_event = self._guard_terminal_emitter(
            self.emit_event,
            execution_guard=execution_guard,
        )
        while True:

            def current_session_revision() -> int | None:
                revision = state.memory_state.get("session_revision")
                if isinstance(revision, bool) or not isinstance(revision, int):
                    return None
                return revision

            dispatch_run_finalizing = partial(
                self._dispatch_run_finalizing,
                execution_guard=execution_guard,
            )
            dispatch_on_suspend = partial(
                self._dispatch_on_suspend,
                execution_guard=execution_guard,
            )

            # A resumed tool batch can already have reached a terminal limit.
            # Honor that result before asking for more budget or another turn.
            if state.run_status == "completed":
                return finish_completed_run(
                    state,
                    callback=callback,
                    run_id=run_id,
                    emit_event=terminal_emit_event,
                    dispatch_run_finalizing=dispatch_run_finalizing,
                )

            max_wait_revision: int | None = None
            durable_max_wait = (
                callable(on_max_iterations)
                and self._interaction_runtime is not None
                and bool(state.session_state.session_id)
            )

            def persist_before_legacy_max_wait() -> None:
                nonlocal max_wait_revision
                state.run_status = "max_iterations"
                dispatch_on_suspend(
                    state,
                    callback=callback,
                    run_id=run_id,
                    iteration=int(state.iteration),
                    status="max_iterations",
                )
                max_wait_revision = current_session_revision()
                if execution_guard is not None and max_wait_revision is not None:
                    execution_guard.release_for_wait()

            if durable_max_wait:
                if self._interaction_runtime is None or not callable(on_max_iterations):
                    raise InteractionNotPendingError(
                        "max-budget interaction has no callback adapter"
                    )
                max_adapter = DurableMaxBudgetCallbackAdapter(
                    state=state,
                    interaction_runtime=self._interaction_runtime,
                    callback_adapter=on_max_iterations,
                    payload=dict(payload or {}),
                    response_format=response_format,
                    effective_max=effective_max,
                    run_id=run_id,
                    event_callback=callback,
                    emit_event=self.emit_event,
                    dispatch_on_suspend=dispatch_on_suspend,
                    current_session_revision=current_session_revision,
                    tool_exposure_plan_factory=(
                        lambda: snapshot_durable_tool_exposure_plan(
                            tool_runtime_plugins
                        )
                    ),
                    execution_guard=execution_guard,
                )
                boundary_callback = max_adapter.invoke
                before_max_wait = max_adapter.before_wait
            else:
                max_adapter = None
                boundary_callback = self._wrap_max_iterations_callback(
                    on_max_iterations,
                    execution_guard=execution_guard,
                    expected_revision=lambda: max_wait_revision,
                )
                before_max_wait = persist_before_legacy_max_wait

            boundary = resolve_max_iterations_boundary(
                state,
                effective_max=effective_max,
                on_max_iterations=boundary_callback,
                callback=callback,
                run_id=run_id,
                emit_event=self.emit_event,
                before_wait=before_max_wait,
            )
            effective_max = boundary.effective_max
            max_interaction_request = (
                max_adapter.interaction_request if max_adapter is not None else None
            )
            if max_interaction_request is not None:
                state.last_continuation = None
                state.suspend_state.signal_kind = None
                state.suspend_state.payload = {}
            if boundary.should_finish:
                return finish_max_iterations_run(
                    state,
                    callback=callback,
                    run_id=run_id,
                    emit_run_max_iterations=boundary.emit_run_max_iterations_on_finish,
                    emit_event=terminal_emit_event,
                    dispatch_run_finalizing=dispatch_run_finalizing,
                )
            if state.run_status == "max_iterations":
                state.run_status = "running"

            self.emit_event(
                callback,
                "iteration_started",
                run_id,
                **build_iteration_started_payload(state),
            )
            turn = self.step_once(
                state,
                payload=payload,
                toolkit=runtime_toolkit,
                callback=callback,
                verbose=verbose,
                run_id=run_id,
                emit_stream=True,
                response_format=response_format,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                max_iterations=effective_max,
                tool_runtime_plugins=tool_runtime_plugins,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
                run_bundle_purpose=run_bundle_purpose,
            )
            if state.run_status == "awaiting_interaction":
                response_received_emitted = False
                while state.run_status == "awaiting_interaction":
                    if not callable(on_tool_confirm):
                        state.run_ledger.record_iteration_outcome(
                            iteration=max(0, int(state.iteration) - 1),
                            outcome="uncertain",
                        )
                    suspended = finish_awaiting_interaction_run(
                        state,
                        callback=callback,
                        run_id=run_id,
                        dispatch_on_suspend=dispatch_on_suspend,
                    )
                    durable_request = suspended.interaction_request
                    continuation = suspended.continuation
                    if (
                        not isinstance(durable_request, dict)
                        or not isinstance(continuation, dict)
                        or self._interaction_runtime is None
                    ):
                        raise InteractionIntegrityError(
                            "awaiting_interaction requires a durable request and continuation"
                        )
                    wait_revision = current_session_revision()
                    released_for_wait = (
                        execution_guard is not None and wait_revision is not None
                    )
                    if released_for_wait:
                        execution_guard.release_for_wait()
                    if not response_received_emitted:
                        self.emit_event(
                            callback,
                            "response_received",
                            run_id,
                            **build_response_received_payload(state, turn),
                        )
                        response_received_emitted = True
                    self.emit_event(
                        callback,
                        "interaction_requested",
                        run_id,
                        iteration=max(0, int(state.iteration) - 1),
                        interaction_request=copy.deepcopy(durable_request),
                    )
                    if not callable(on_tool_confirm):
                        return suspended
                    if durable_request.get("kind") != INTERACTION_KIND_TOOL_APPROVAL:
                        raise InteractionNotPendingError(
                            "awaiting interaction is not a tool approval"
                        )
                    confirmation_request = (
                        self._tool_confirmation_request_from_interaction(
                            durable_request
                        )
                    )
                    response = on_tool_confirm(confirmation_request)
                    durable_snapshot = require_unapplied_callback_receipt(
                        self._interaction_runtime.record_receipt(
                            str(state.session_state.session_id or ""),
                            interaction_id=str(
                                durable_request.get("interaction_id") or ""
                            ),
                            response=response,
                            submitted_by="callback:on_tool_confirm",
                            expected_revision=wait_revision,
                        )
                    )
                    persisted_revision = durable_snapshot.session_snapshot.revision
                    state.memory_state.update(
                        {
                            "session_revision": persisted_revision,
                            "session_snapshot": copy.deepcopy(
                                durable_snapshot.session_snapshot.state
                            ),
                        }
                    )
                    state.component_bucket("memory")["state"] = copy.deepcopy(
                        state.memory_state
                    )
                    if released_for_wait:
                        execution_guard.reacquire(
                            expected_revision=persisted_revision,
                        )
                    elif execution_guard is not None:
                        execution_guard.assert_active()
                    self._dispatch_tool_approval_resume(
                        state,
                        continuation=continuation,
                        interaction_request=durable_request,
                        response=durable_snapshot.response or {},
                        payload=payload,
                        response_format=response_format,
                        callback=callback,
                        verbose=verbose,
                        on_tool_confirm=on_tool_confirm,
                        on_human_input=on_human_input,
                        max_iterations=effective_max,
                        toolkit=runtime_toolkit,
                        run_id=run_id,
                        tool_runtime_plugins=tool_runtime_plugins,
                        tool_runtime_config=tool_runtime_config,
                        execution_guard=execution_guard,
                    )
                self.emit_event(
                    callback,
                    "iteration_completed",
                    run_id,
                    **build_iteration_completed_payload(
                        state,
                        has_tool_calls=True,
                    ),
                )
                state.run_ledger.record_iteration_outcome(
                    iteration=max(0, int(state.iteration) - 1),
                    outcome="completed",
                )
                continue
            if state.run_status == "awaiting_human_input":
                pending_human_request = (
                    state.tool_batch_state.human_input_request
                )
                if (
                    not callable(on_human_input)
                    or pending_human_request is None
                    or not isinstance(state.last_continuation, dict)
                ):
                    state.run_ledger.record_iteration_outcome(
                        iteration=max(0, int(state.iteration) - 1),
                        outcome="uncertain",
                    )
                suspended = finish_awaiting_human_input_run(
                    state,
                    callback=callback,
                    run_id=run_id,
                    dispatch_on_suspend=dispatch_on_suspend,
                )
                request = state.tool_batch_state.human_input_request
                continuation = suspended.continuation
                durable_request = suspended.interaction_request
                wait_revision = current_session_revision()
                durable_wait = (
                    isinstance(durable_request, dict)
                    and self._interaction_runtime is not None
                    and bool(state.session_state.session_id)
                )
                released_for_wait = False
                if (
                    durable_wait
                    and execution_guard is not None
                    and wait_revision is not None
                ):
                    execution_guard.release_for_wait()
                    released_for_wait = True
                self.emit_event(
                    callback,
                    "response_received",
                    run_id,
                    **build_response_received_payload(state, turn),
                )
                if request is not None:
                    self.emit_event(
                        callback,
                        "human_input_requested",
                        run_id,
                        iteration=max(0, int(state.iteration) - 1),
                        **(
                            {
                                "interaction_id": str(
                                    durable_request.get("interaction_id") or ""
                                ),
                                "interaction_request": copy.deepcopy(durable_request),
                            }
                            if isinstance(durable_request, dict)
                            else {}
                        ),
                        **request.to_dict(),
                    )
                if (
                    not callable(on_human_input)
                    or request is None
                    or continuation is None
                ):
                    return suspended
                if durable_wait:
                    response = on_human_input(request)
                    durable_snapshot = require_unapplied_callback_receipt(
                        self._interaction_runtime.record_receipt(
                            str(state.session_state.session_id or ""),
                            interaction_id=str(
                                durable_request.get("interaction_id") or ""
                            ),
                            response=response,
                            submitted_by="callback:on_human_input",
                            expected_revision=wait_revision,
                        )
                    )
                    response = durable_snapshot.response
                    persisted_revision = durable_snapshot.session_snapshot.revision
                    state.memory_state.update(
                        {
                            "session_revision": persisted_revision,
                            "session_snapshot": copy.deepcopy(
                                durable_snapshot.session_snapshot.state
                            ),
                        }
                    )
                    state.component_bucket("memory")["state"] = copy.deepcopy(
                        state.memory_state
                    )
                    if released_for_wait:
                        execution_guard.reacquire(
                            expected_revision=persisted_revision,
                        )
                    elif execution_guard is not None:
                        execution_guard.assert_active()
                else:
                    if execution_guard is not None and wait_revision is not None:
                        execution_guard.release_for_wait()
                        released_for_wait = True
                    response = on_human_input(request)
                    if released_for_wait:
                        execution_guard.reacquire(
                            expected_revision=wait_revision,
                        )
                    elif execution_guard is not None:
                        execution_guard.assert_active()
                self._dispatch_on_resume(
                    state,
                    continuation=continuation,
                    response=response,
                    callback=callback,
                    run_id=run_id,
                    execution_guard=execution_guard,
                )
                state.run_ledger.record_iteration_outcome(
                    iteration=max(0, int(state.iteration) - 1),
                    outcome="completed",
                )
                continue
            self.emit_event(
                callback,
                "response_received",
                run_id,
                **build_response_received_payload(state, turn),
            )
            if state.run_status == "completed":
                state.run_ledger.record_iteration_outcome(
                    iteration=max(0, int(state.iteration) - 1),
                    outcome="completed",
                )
                return finish_completed_run(
                    state,
                    callback=callback,
                    run_id=run_id,
                    emit_event=terminal_emit_event,
                    dispatch_run_finalizing=dispatch_run_finalizing,
                )
            if turn.tool_calls:
                state.run_ledger.record_iteration_outcome(
                    iteration=max(0, int(state.iteration) - 1),
                    outcome="completed",
                )
                self.emit_event(
                    callback,
                    "iteration_completed",
                    run_id,
                    **build_iteration_completed_payload(state, has_tool_calls=True),
                )
                continue
            state.run_ledger.record_iteration_outcome(
                iteration=max(0, int(state.iteration) - 1),
                outcome="completed",
            )
            return finish_completed_run(
                state,
                callback=callback,
                run_id=run_id,
                emit_event=terminal_emit_event,
                dispatch_run_finalizing=dispatch_run_finalizing,
            )

    def _run_state_with_failure_bundle(
        self,
        state: RunState,
        **kwargs: Any,
    ) -> KernelRunResult:
        try:
            return self._run_state(state, **kwargs)
        except Exception as exc:
            ledger = state.run_ledger
            if ledger.identity is not None:
                raw_code = str(getattr(exc, "code", "kernel_run_failed") or "")
                error_code = (
                    raw_code
                    if raw_code
                    and raw_code[0].islower()
                    and all(
                        character.islower()
                        or character.isdigit()
                        or character in "_.-"
                        for character in raw_code
                    )
                    and len(raw_code) <= 128
                    else "kernel_run_failed"
                )
                module_name = type(exc).__module__.lower()
                error_category = next(
                    (
                        category
                        for category in (
                            "provider",
                            "context",
                            "interaction",
                            "execution",
                            "tool",
                        )
                        if category in module_name
                    ),
                    "runtime",
                )
                already_recorded = any(
                    event.kind == "error"
                    and event.error is not None
                    and event.error.code == error_code
                    for event in ledger.metric_events.values()
                )
                if not already_recorded:
                    ledger.record_metric_event(
                        kind="error",
                        subject_id=(
                            f"run-failure:{max(0, int(state.iteration))}:"
                            f"{error_code}"
                        ),
                        outcome="failed",
                        error_category=error_category,
                        error_code=error_code,
                    )
                direct_attempt_iterations = [
                    receipt.identity.iteration
                    for receipt in ledger.receipts.values()
                    if receipt.identity.owner_run_id == ledger.identity.run_id
                ]
                failed_iteration = (
                    max(direct_attempt_iterations)
                    if direct_attempt_iterations
                    else max(0, int(state.iteration))
                )
                uncertain_iteration = any(
                    receipt.identity.owner_run_id == ledger.identity.run_id
                    and receipt.identity.iteration == failed_iteration
                    and receipt.status == "uncertain"
                    for receipt in ledger.receipts.values()
                )
                ledger.record_iteration_outcome(
                    iteration=failed_iteration,
                    outcome="uncertain" if uncertain_iteration else "failed",
                    error_category=error_category,
                    error_code=error_code,
                )
                state.run_status = "failed"
                from ..run_bundle_v2 import run_bundle_from_dict

                failed_bundle = run_bundle_from_dict(
                    materialize_state_bundle(state, status="failed")
                )
                attach_kernel_run_failure(
                    exc,
                    error_category=error_category,
                    error_code=error_code,
                    run_bundle=failed_bundle,
                )
            raise

    def run(
        self,
        messages: list[dict[str, Any]],
        *,
        payload: dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
        callback: Any = None,
        verbose: bool = False,
        max_iterations: int = 6,
        previous_response_id: str | None = None,
        on_tool_confirm: Any = None,
        on_human_input: Any = None,
        on_max_iterations: Any = None,
        session_id: str | None = None,
        memory_namespace: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_context_window_tokens: int | None = None,
        toolkit: Toolkit | None = None,
        run_id: str | None = None,
        tool_runtime_plugins: list[Any] | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        _provider_replay_frame: dict[str, Any] | None = None,
        _execution_guard: ExecutionGuard | None = None,
        _runtime_context: "AgentRuntimeContext | None" = None,
        _run_bundle_identity: "RunIdentity | None" = None,
        _run_bundle_descriptor: "RunDescriptor | None" = None,
        _continued_from_run_id: str | None = None,
        _run_bundle_purpose: str = "agent_turn",
        _run_bundle_receipts: tuple["ProviderCallReceipt", ...] = (),
        _provider_turn_ownership: Any = None,
    ) -> KernelRunResult:
        plan = prepare_fresh_run_invocation(
            messages=messages,
            payload=payload,
            model_io=self._model_io,
            provider=provider,
            model=model,
            previous_response_id=previous_response_id,
            session_id=session_id,
            memory_namespace=memory_namespace,
            max_context_window_tokens=max_context_window_tokens,
            run_id=run_id,
            run_id_factory=lambda: str(uuid.uuid4()),
        )
        if isinstance(_provider_replay_frame, dict):
            set_provider_replay_frame(plan.state, _provider_replay_frame)
        with self._scope_for_session(
            session_id=session_id,
            execution_guard=_execution_guard,
        ) as execution_guard:
            return self._run_state_with_failure_bundle(
                plan.state,
                payload=plan.payload,
                response_format=response_format,
                callback=callback,
                verbose=verbose,
                max_iterations=max_iterations,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                on_max_iterations=on_max_iterations,
                toolkit=toolkit,
                run_id=plan.run_id,
                tool_runtime_plugins=tool_runtime_plugins,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
                runtime_context=_runtime_context,
                run_bundle_identity=_run_bundle_identity,
                run_bundle_descriptor=_run_bundle_descriptor,
                continued_from_run_id=_continued_from_run_id,
                run_bundle_purpose=_run_bundle_purpose,
                run_bundle_receipts=_run_bundle_receipts,
                provider_turn_ownership=_provider_turn_ownership,
            )

    def resume_human_input(
        self,
        *,
        conversation: list[dict[str, Any]],
        continuation: dict[str, Any],
        response: dict[str, Any] | Any = None,
        payload: dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
        callback: Any = None,
        verbose: bool = False,
        on_tool_confirm: Any = None,
        on_human_input: Any = None,
        on_max_iterations: Any = None,
        session_id: str | None = None,
        memory_namespace: str | None = None,
        toolkit: Toolkit | None = None,
        run_id: str | None = None,
        tool_runtime_plugins: list[Any] | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        _execution_guard: ExecutionGuard | None = None,
        _runtime_context: "AgentRuntimeContext | None" = None,
        _run_bundle_identity: "RunIdentity | None" = None,
        _run_bundle_descriptor: "RunDescriptor | None" = None,
        _run_bundle_purpose: str = "agent_turn",
        _run_bundle_receipts: tuple["ProviderCallReceipt", ...] = (),
        _provider_turn_ownership: Any = None,
    ) -> KernelRunResult:
        plan = prepare_resume_run_invocation(
            conversation=conversation,
            continuation=continuation,
            payload=payload,
            response_format=response_format,
            fallback_provider=self._infer_provider(),
            fallback_model=self._infer_model(),
            session_id=session_id,
            memory_namespace=memory_namespace,
            run_id=run_id,
            run_id_factory=lambda: str(uuid.uuid4()),
        )
        initialize_run_ledger(
            plan.state,
            run_id=plan.run_id,
            runtime_context=_runtime_context,
            explicit_identity=_run_bundle_identity,
            descriptor=_run_bundle_descriptor,
        )
        plan.state.run_ledger.bind_provider_turn_ownership(
            _provider_turn_ownership
        )
        for receipt in _run_bundle_receipts:
            plan.state.run_ledger.append(receipt)
        with self._scope_for_session(
            session_id=session_id,
            execution_guard=_execution_guard,
        ) as execution_guard:
            resolved_response = copy.deepcopy(response)
            if self._interaction_runtime is not None and session_id:
                checkpoint = (
                    self._interaction_runtime.memory_runtime.load_execution_checkpoint(
                        session_id
                    )
                )
                interaction_ref = (
                    checkpoint.get("interaction_ref")
                    if isinstance(checkpoint, dict)
                    else None
                )
                if isinstance(interaction_ref, dict):
                    interaction_id = str(interaction_ref.get("interaction_id") or "")
                    continuation_interaction_id = continuation.get("interaction_id")
                    if (
                        isinstance(continuation_interaction_id, str)
                        and continuation_interaction_id
                        and continuation_interaction_id != interaction_id
                    ):
                        raise InteractionNotPendingError(
                            "resume continuation targets a different interaction"
                        )
                    pending = self._interaction_runtime.load(
                        session_id,
                        interaction_id=interaction_id,
                        require_active=True,
                    )
                    ensure_interaction_runtime_matches(
                        pending.request,
                        provider=self._infer_provider(),
                        model=self._infer_model(),
                        checkpoint=checkpoint,
                        continuation=continuation,
                    )
                    if resolved_response is None:
                        durable_snapshot = self._interaction_runtime.require_receipt(
                            session_id,
                            interaction_id=interaction_id,
                        )
                    else:
                        if pending.request.kind != INTERACTION_KIND_HUMAN_INPUT:
                            raise InteractionNotPendingError(
                                "pending interaction is not a human-input request"
                            )
                        durable_snapshot = self._interaction_runtime.record_receipt(
                            session_id,
                            interaction_id=interaction_id,
                            response=resolved_response,
                            submitted_by="api:resume_human_input",
                            expected_revision=pending.session_snapshot.revision,
                            execution_fence=(
                                execution_guard.fence
                                if execution_guard is not None
                                else None
                            ),
                        )
                    if durable_snapshot.request.kind != INTERACTION_KIND_HUMAN_INPUT:
                        raise InteractionNotPendingError(
                            "pending interaction is not a human-input request"
                        )
                    resolved_response = durable_snapshot.response
                elif resolved_response is None:
                    raise InteractionNotPendingError(
                        "legacy human-input resume requires an explicit response"
                    )
            elif resolved_response is None:
                raise InteractionNotPendingError(
                    "human-input resume requires an explicit response"
                )
            self._dispatch_bootstrap(
                plan.state,
                payload=plan.payload,
                response_format=plan.response_format,
                callback=callback,
                verbose=verbose,
                toolkit=toolkit,
                run_id=plan.run_id,
                resume_mode=True,
                continuation=continuation,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
            )
            self._dispatch_on_resume(
                plan.state,
                continuation=continuation,
                response=resolved_response,
                callback=callback,
                run_id=plan.run_id,
                execution_guard=execution_guard,
            )
            return self._run_state_with_failure_bundle(
                plan.state,
                payload=plan.payload,
                response_format=plan.response_format,
                callback=callback,
                verbose=verbose,
                max_iterations=plan.max_iterations,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                on_max_iterations=on_max_iterations,
                toolkit=toolkit,
                run_id=plan.run_id,
                skip_bootstrap=True,
                tool_runtime_plugins=tool_runtime_plugins,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
                runtime_context=_runtime_context,
                run_bundle_identity=_run_bundle_identity,
                run_bundle_descriptor=_run_bundle_descriptor,
                run_bundle_purpose=_run_bundle_purpose,
                run_bundle_receipts=_run_bundle_receipts,
                provider_turn_ownership=_provider_turn_ownership,
            )

    def resume_interaction(
        self,
        *,
        session_id: str,
        conversation: list[dict[str, Any]] | None = None,
        continuation: dict[str, Any] | None = None,
        response: Any = None,
        submitted_by: str = "api:resume_interaction",
        payload: dict[str, Any] | None = None,
        response_format: ResponseFormat | None = None,
        callback: Any = None,
        verbose: bool = False,
        on_tool_confirm: Any = None,
        on_human_input: Any = None,
        on_max_iterations: Any = None,
        memory_namespace: str | None = None,
        toolkit: Toolkit | None = None,
        run_id: str | None = None,
        tool_runtime_plugins: list[Any] | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        _execution_guard: ExecutionGuard | None = None,
        _runtime_context: "AgentRuntimeContext | None" = None,
        _run_bundle_identity: "RunIdentity | None" = None,
        _run_bundle_descriptor: "RunDescriptor | None" = None,
        _run_bundle_purpose: str = "agent_turn",
        _run_bundle_receipts: tuple["ProviderCallReceipt", ...] = (),
        _provider_turn_ownership: Any = None,
    ) -> KernelRunResult:
        if self._interaction_runtime is None:
            raise InteractionNotPendingError(
                "resume_interaction requires a memory-backed interaction runtime"
            )
        if not isinstance(session_id, str) or not session_id:
            raise InteractionNotPendingError(
                "resume_interaction requires a non-empty session_id"
            )

        with self._scope_for_session(
            session_id=session_id,
            execution_guard=_execution_guard,
        ) as execution_guard:
            checkpoint = (
                self._interaction_runtime.memory_runtime.load_execution_checkpoint(
                    session_id
                )
            )
            if not isinstance(checkpoint, dict):
                raise InteractionNotPendingError(
                    "session has no durable execution checkpoint"
                )
            interaction_ref = checkpoint.get("interaction_ref")
            if not isinstance(interaction_ref, dict):
                raise InteractionNotPendingError(
                    "execution checkpoint has no durable interaction"
                )
            interaction_id = str(interaction_ref.get("interaction_id") or "")
            pending_interaction = self._interaction_runtime.load(
                session_id,
                interaction_id=interaction_id,
                require_active=True,
            )
            ensure_interaction_runtime_matches(
                pending_interaction.request,
                provider=self._infer_provider(),
                model=self._infer_model(),
                checkpoint=checkpoint,
                continuation=(
                    continuation
                    if isinstance(continuation, dict)
                    else checkpoint.get("continuation")
                ),
            )
            if response is None:
                durable_snapshot = self._interaction_runtime.require_receipt(
                    session_id,
                    interaction_id=interaction_id,
                )
            else:
                pending = self._interaction_runtime.load(
                    session_id,
                    interaction_id=interaction_id,
                    require_active=True,
                )
                durable_snapshot = self._interaction_runtime.record_receipt(
                    session_id,
                    interaction_id=interaction_id,
                    response=response,
                    submitted_by=submitted_by,
                    expected_revision=pending.session_snapshot.revision,
                    execution_fence=(
                        execution_guard.fence if execution_guard is not None else None
                    ),
                )
            if durable_snapshot.checkpoint_id != checkpoint.get(
                "checkpoint_id"
            ) or durable_snapshot.request.request_digest != interaction_ref.get(
                "request_digest"
            ):
                raise InteractionIntegrityError(
                    "durable interaction is not bound to the active checkpoint"
                )

            resolved_conversation = (
                copy.deepcopy(conversation)
                if isinstance(conversation, list)
                else copy.deepcopy(checkpoint.get("transcript") or [])
            )
            resolved_continuation = (
                copy.deepcopy(continuation)
                if isinstance(continuation, dict)
                else copy.deepcopy(checkpoint.get("continuation") or {})
            )
            if (
                resolved_continuation.get("interaction_id")
                != durable_snapshot.request.interaction_id
            ):
                raise InteractionIntegrityError(
                    "resume continuation does not match the durable interaction"
                )
            plan = prepare_resume_run_invocation(
                conversation=resolved_conversation,
                continuation=resolved_continuation,
                payload=payload,
                response_format=response_format,
                fallback_provider=self._infer_provider(),
                fallback_model=self._infer_model(),
                session_id=session_id,
                memory_namespace=memory_namespace,
                run_id=run_id,
                run_id_factory=lambda: str(uuid.uuid4()),
            )
            initialize_run_ledger(
                plan.state,
                run_id=plan.run_id,
                runtime_context=_runtime_context,
                explicit_identity=_run_bundle_identity,
                descriptor=_run_bundle_descriptor,
            )
            plan.state.run_ledger.bind_provider_turn_ownership(
                _provider_turn_ownership
            )
            for receipt in _run_bundle_receipts:
                plan.state.run_ledger.append(receipt)
            if (
                durable_snapshot.request.kind == INTERACTION_KIND_TOOL_APPROVAL
                and self.final_model_tool_boundary is not None
            ):
                expected_source_run_id = durable_snapshot.request.source_run_id
                if (
                    not expected_source_run_id
                    or plan.run_id != expected_source_run_id
                    or checkpoint.get("source_run_id") != expected_source_run_id
                    or resolved_continuation.get("run_id") != expected_source_run_id
                ):
                    raise InteractionIntegrityError(
                        "final model boundary tool approval resume changed its "
                        "durable source run"
                    )
            self._dispatch_bootstrap(
                plan.state,
                payload=plan.payload,
                response_format=plan.response_format,
                callback=callback,
                verbose=verbose,
                toolkit=toolkit,
                run_id=plan.run_id,
                resume_mode=True,
                continuation=resolved_continuation,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
            )

            if durable_snapshot.request.kind == INTERACTION_KIND_HUMAN_INPUT:
                if resolved_continuation.get("type") != HUMAN_INPUT_CONTINUATION_TYPE:
                    raise InteractionIntegrityError(
                        "human-input receipt requires a matching continuation"
                    )
                self._dispatch_on_resume(
                    plan.state,
                    continuation=resolved_continuation,
                    response=durable_snapshot.response,
                    callback=callback,
                    run_id=plan.run_id,
                    execution_guard=execution_guard,
                )
                return self._run_state_with_failure_bundle(
                    plan.state,
                    payload=plan.payload,
                    response_format=plan.response_format,
                    callback=callback,
                    verbose=verbose,
                    max_iterations=plan.max_iterations,
                    on_tool_confirm=on_tool_confirm,
                    on_human_input=on_human_input,
                    on_max_iterations=on_max_iterations,
                    toolkit=toolkit,
                    run_id=plan.run_id,
                    skip_bootstrap=True,
                    tool_runtime_plugins=tool_runtime_plugins,
                    tool_runtime_config=tool_runtime_config,
                    execution_guard=execution_guard,
                    runtime_context=_runtime_context,
                    run_bundle_identity=_run_bundle_identity,
                    run_bundle_descriptor=_run_bundle_descriptor,
                    run_bundle_purpose=_run_bundle_purpose,
                    run_bundle_receipts=_run_bundle_receipts,
                    provider_turn_ownership=_provider_turn_ownership,
                )
            if durable_snapshot.request.kind == INTERACTION_KIND_TOOL_APPROVAL:
                self._dispatch_tool_approval_resume(
                    plan.state,
                    continuation=resolved_continuation,
                    interaction_request=durable_snapshot.request.to_dict(),
                    response=durable_snapshot.response or {},
                    payload=plan.payload,
                    response_format=plan.response_format,
                    callback=callback,
                    verbose=verbose,
                    on_tool_confirm=on_tool_confirm,
                    on_human_input=on_human_input,
                    max_iterations=plan.max_iterations,
                    toolkit=toolkit if toolkit is not None else Toolkit(),
                    run_id=plan.run_id,
                    tool_runtime_plugins=tool_runtime_plugins,
                    tool_runtime_config=tool_runtime_config,
                    execution_guard=execution_guard,
                )
                if plan.state.run_status == "awaiting_interaction":
                    return finish_awaiting_interaction_run(
                        plan.state,
                        callback=callback,
                        run_id=plan.run_id,
                        dispatch_on_suspend=partial(
                            self._dispatch_on_suspend,
                            execution_guard=execution_guard,
                        ),
                    )
                self.emit_event(
                    callback,
                    "iteration_completed",
                    plan.run_id,
                    **build_iteration_completed_payload(
                        plan.state,
                        has_tool_calls=True,
                    ),
                )
                return self._run_state_with_failure_bundle(
                    plan.state,
                    payload=plan.payload,
                    response_format=plan.response_format,
                    callback=callback,
                    verbose=verbose,
                    max_iterations=plan.max_iterations,
                    on_tool_confirm=on_tool_confirm,
                    on_human_input=on_human_input,
                    on_max_iterations=on_max_iterations,
                    toolkit=toolkit,
                    run_id=plan.run_id,
                    skip_bootstrap=True,
                    tool_runtime_plugins=tool_runtime_plugins,
                    tool_runtime_config=tool_runtime_config,
                    execution_guard=execution_guard,
                    runtime_context=_runtime_context,
                    run_bundle_identity=_run_bundle_identity,
                    run_bundle_descriptor=_run_bundle_descriptor,
                    run_bundle_purpose=_run_bundle_purpose,
                    run_bundle_receipts=_run_bundle_receipts,
                    provider_turn_ownership=_provider_turn_ownership,
                )
            if durable_snapshot.request.kind != INTERACTION_KIND_MAX_BUDGET:
                raise InteractionNotPendingError("unsupported durable interaction kind")
            if resolved_continuation.get("type") != "max_budget_continuation":
                raise InteractionIntegrityError(
                    "max-budget receipt requires a max_budget_continuation"
                )
            decision = durable_snapshot.response or {}
            approved = bool(decision.get("approved"))
            extra_iterations = (
                int(decision.get("extra_iterations") or 0) if approved else 0
            )
            plan.state.last_continuation = None
            plan.state.suspend_state.signal_kind = None
            plan.state.suspend_state.payload = {}
            return self._run_state_with_failure_bundle(
                plan.state,
                payload=plan.payload,
                response_format=plan.response_format,
                callback=callback,
                verbose=verbose,
                max_iterations=extra_iterations,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                on_max_iterations=(on_max_iterations if approved else None),
                toolkit=toolkit,
                run_id=plan.run_id,
                skip_bootstrap=True,
                tool_runtime_plugins=tool_runtime_plugins,
                tool_runtime_config=tool_runtime_config,
                execution_guard=execution_guard,
                runtime_context=_runtime_context,
                run_bundle_identity=_run_bundle_identity,
                run_bundle_descriptor=_run_bundle_descriptor,
                run_bundle_purpose=_run_bundle_purpose,
                run_bundle_receipts=_run_bundle_receipts,
                provider_turn_ownership=_provider_turn_ownership,
            )
