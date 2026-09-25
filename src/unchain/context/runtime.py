from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from ..durability import (
    find_durable_persistence_failure,
    mark_durable_persistence_failure,
)
from ..kernel.harness import HarnessContext
from ..journal import AttemptRef, ContextBuildStatus
from ..run_bundle import opaque_metric_evidence_ref
from .compiler import ContextCompileResult, ContextCompiler
from .coordinator import ContextCompileCoordinator
from .factory import (
    ContextExecutionBundle,
    ContextExecutionBundleError,
    DurableContextRuntimeFactory,
)
from .models import ContextCompileRequest
from .tool_permits import (
    MAX_PREPARED_TOOL_PERMITS,
    _PreparedToolBinding,
    _is_runtime_tool_permit,
    _mint_tool_permit,
    _new_tool_approval_pending,
)

if TYPE_CHECKING:
    from ..kernel.state import RunState
    from .subagent_input import PreparedSubagentInput


_OWNER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_ACTIVE_DURABLE_EVENTS: ContextVar[tuple[tuple[str, int], ...]] = ContextVar(
    "unchain_active_durable_events",
    default=(),
)
_TEST_CONTEXT_RUNTIME_AUTHORITY = object()
_CONTEXT_EXECUTION_BINDING_AUTHORITY = object()
_TOOL_OUTPUT_SNAPSHOT_EVENT_TYPE = "context.tool_output_snapshot"
_TOOL_OUTPUT_SNAPSHOT_SCHEMA = "unchain.context_tool_output_snapshot.v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class ContextRequestFactory(Protocol):
    def __call__(self, context: HarnessContext) -> ContextCompileRequest:
        ...


class ContextCheckpointPersistenceRequiredError(RuntimeError):
    """The pure compiler requested a durable checkpoint before model use."""

    code = "context_v2_checkpoint_persistence_required"

    def __init__(self, result: ContextCompileResult) -> None:
        self.result = result
        self.checkpoint_requests = tuple(result.checkpoint_requests)
        super().__init__(self.code)


class ContextCheckpointBindingForbiddenError(RuntimeError):
    """A runtime request tried to supply coordinator-owned checkpoint authority."""

    code = "context_v2_checkpoint_binding_forbidden"

    def __init__(self) -> None:
        super().__init__(self.code)


class ContextBuildEnvelopeRequiredError(RuntimeError):
    """A runtime compilation completed without an auditable build envelope."""

    code = "context_v2_build_envelope_required"

    def __init__(self, result: ContextCompileResult) -> None:
        self.result = result
        super().__init__(self.code)


class ContextBuildUnavailableError(RuntimeError):
    """The compiler could not produce context that is safe for model use."""

    code = "context_v2_build_unavailable"

    def __init__(self, result: ContextCompileResult) -> None:
        self.result = result
        super().__init__(self.code)


@dataclass(frozen=True)
class _BoundExecutionGuard:
    guard: Any = field(repr=False, compare=False)
    root_guard: Any = field(repr=False, compare=False)
    execution_id: str
    owner_id: str
    fencing_token: int
    chain: tuple[tuple[Any, ...], ...]


@dataclass(frozen=True)
class _BoundInteractionAuthority:
    loop: Any = field(repr=False, compare=False)
    interaction_runtime: Any = field(repr=False, compare=False)
    memory_runtime: Any = field(repr=False, compare=False)
    memory_store: Any = field(repr=False, compare=False)
    receipt_reader: Any = field(repr=False, compare=False)
    session_id: str


@dataclass(frozen=True)
class _ProviderTurnBoundaryBinding:
    execution_id: str
    attempt_id: str
    iteration: int
    model_io_object_id: int
    source_toolkit_object_id: int
    model_toolkit_object_id: int
    execution_toolkit_object_id: int
    tool_identities: tuple[tuple[str, int, int], ...]


@dataclass(frozen=True)
class ContextSubagentCompletionSink:
    """One parent-attempt-bound sink for completed child results."""

    _runtime: ContextRuntime = field(repr=False, compare=False)
    parent_attempt: AttemptRef
    call_id: str

    def __post_init__(self) -> None:
        if type(self._runtime) is not ContextRuntime:
            raise TypeError("subagent completion sink requires ContextRuntime")
        if not isinstance(self.parent_attempt, AttemptRef):
            object.__setattr__(
                self,
                "parent_attempt",
                AttemptRef.from_dict(self.parent_attempt),
            )
        from ..journal.models import _required_text

        object.__setattr__(
            self,
            "call_id",
            _required_text(self.call_id, "call_id", identifier=True),
        )

    def record(self, *, child_run_id: str, result):
        from ..subagents.types import SubagentResult

        if type(result) is not SubagentResult:
            raise TypeError("subagent completion requires an exact SubagentResult")
        return self._runtime._record_subagent_completion(
            parent_attempt=self.parent_attempt,
            call_id=self.call_id,
            child_run_id=child_run_id,
            result=result,
        )

    def prepare_input(
        self,
        *,
        child_run_id: str,
        child_id: str,
        mode: str,
        lineage,
        template_name: str | None,
        input_messages,
    ):
        """Bind one child task to this sink's exact parent tool intent."""

        return self._runtime._prepare_subagent_input(
            parent_attempt=self.parent_attempt,
            call_id=self.call_id,
            child_run_id=child_run_id,
            child_id=child_id,
            mode=mode,
            lineage=lineage,
            template_name=template_name,
            input_messages=input_messages,
        )


def _resolve_bootstrap_interaction_authority(
    context: HarnessContext,
    *,
    guard_binding: _BoundExecutionGuard | None,
) -> _BoundInteractionAuthority:
    from ..interaction.runtime import DurableInteractionRuntime
    from ..kernel.loop import KernelLoop
    from ..memory.runtime import KernelMemoryRuntime

    session_id = str(context.state.session_state.session_id or "").strip()
    if not session_id:
        raise ContextExecutionBundleError(
            "bootstrap interaction authority requires a session binding"
        )
    loop = context.event.get("loop")
    if loop is None:
        return _BoundInteractionAuthority(
            loop=None,
            interaction_runtime=None,
            memory_runtime=None,
            memory_store=None,
            receipt_reader=None,
            session_id=session_id,
        )
    if type(loop) is not KernelLoop:
        raise ContextExecutionBundleError(
            "bootstrap interaction authority requires the official KernelLoop"
        )
    interaction_runtime = loop.interaction_runtime
    if interaction_runtime is None:
        return _BoundInteractionAuthority(
            loop=loop,
            interaction_runtime=None,
            memory_runtime=None,
            memory_store=None,
            receipt_reader=None,
            session_id=session_id,
        )
    if type(interaction_runtime) is not DurableInteractionRuntime:
        raise ContextExecutionBundleError(
            "bootstrap interaction authority requires the official "
            "DurableInteractionRuntime"
        )
    memory_runtime = interaction_runtime.memory_runtime
    if type(memory_runtime) is not KernelMemoryRuntime:
        raise ContextExecutionBundleError(
            "bootstrap interaction authority requires the official "
            "KernelMemoryRuntime"
        )
    if guard_binding is None:
        raise ContextExecutionBundleError(
            "bootstrap interaction authority requires an execution guard"
        )
    root_store = guard_binding.chain[-1][3]
    memory_store = memory_runtime.store
    if memory_store is not root_store:
        raise ContextExecutionBundleError(
            "bootstrap interaction runtime and execution guard must share "
            "the same session store"
        )
    receipt_reader = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=memory_store)
    )
    return _BoundInteractionAuthority(
        loop=loop,
        interaction_runtime=interaction_runtime,
        memory_runtime=memory_runtime,
        memory_store=memory_store,
        receipt_reader=receipt_reader,
        session_id=session_id,
    )


@dataclass(frozen=True)
class ContextRuntime:
    """Execution-neutral owner of one Context V2 compiler and durability seam."""

    owner_id: str
    request_factory: ContextRequestFactory | None = None
    durable_event_sink: Callable[[dict[str, Any]], None] | None = field(
        default=None,
        repr=False,
    )
    partial_attempt_sink: (Callable[[dict[str, Any], Exception], None] | None) = field(
        default=None, repr=False
    )
    compiler: Any = field(default=None, repr=False)
    execution_factory: DurableContextRuntimeFactory | None = field(
        default=None,
        repr=False,
    )
    provider_turns_enabled: bool = False
    tool_output_management_active: bool = False
    _test_authority: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _failure_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_failures: dict[tuple[str, str], Exception] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _bundle_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_bundles: dict[tuple[str, str], ContextExecutionBundle] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _run_bindings: dict[str, tuple[str, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_tool_output_configs: dict[tuple[str, str], dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _graph_scope_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _graph_scopes: dict[tuple[str, str], tuple[str, int, int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _prepared_subagent_inputs: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _subagent_input_receipts: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _subagent_parent_scopes: dict[tuple[str, str], str] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_execution_guards: dict[tuple[str, str], _BoundExecutionGuard] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _attempt_interaction_authorities: dict[
        tuple[str, str], _BoundInteractionAuthority
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _tool_permit_authority: object = field(
        default_factory=object,
        init=False,
        repr=False,
        compare=False,
    )
    _tool_permit_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _prepared_tool_bindings: dict[int, _PreparedToolBinding] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _prepared_tool_keys: dict[tuple[str, str, str], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _provider_boundary_lock: threading.RLock = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
        compare=False,
    )
    _provider_boundary: Any = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        owner_id = str(self.owner_id or "").strip()
        if _OWNER_ID_RE.fullmatch(owner_id) is None:
            raise ValueError("context runtime owner_id must be a stable identifier")
        object.__setattr__(self, "owner_id", owner_id)
        if type(self.provider_turns_enabled) is not bool:
            raise TypeError("provider_turns_enabled must be an exact boolean")
        if type(self.tool_output_management_active) is not bool:
            raise TypeError("tool_output_management_active must be an exact boolean")
        if self.provider_turns_enabled and self.execution_factory is None:
            raise TypeError("durable provider turns require a factory ContextRuntime")
        if self.execution_factory is not None:
            if type(self.execution_factory) is not DurableContextRuntimeFactory:
                raise TypeError(
                    "execution_factory must be the official "
                    "DurableContextRuntimeFactory"
                )
            if any(
                component is not None
                for component in (
                    self.request_factory,
                    self.durable_event_sink,
                    self.partial_attempt_sink,
                    self.compiler,
                )
            ):
                raise TypeError(
                    "factory ContextRuntime cannot accept separately assembled "
                    "compiler or durability components"
                )
            return
        if not callable(self.request_factory):
            raise TypeError("context runtime request_factory must be callable")
        if not callable(self.durable_event_sink):
            raise TypeError("context runtime durable_event_sink must be callable")
        if not callable(self.partial_attempt_sink):
            raise TypeError("context runtime partial_attempt_sink must be callable")
        if self._test_authority is not _TEST_CONTEXT_RUNTIME_AUTHORITY:
            if type(self.compiler) is not ContextCompileCoordinator:
                raise TypeError(
                    "ContextRuntime requires the official " "ContextCompileCoordinator"
                )
            if not callable(getattr(self.compiler, "compile", None)):
                raise TypeError(
                    "context runtime compiler must provide compile(request)"
                )
        elif self.compiler is not None and not callable(
            getattr(self.compiler, "compile", None)
        ):
            raise TypeError("test compiler must provide compile(request)")

    @classmethod
    def _for_test(cls, **kwargs) -> ContextRuntime:
        """Construct an uncommitted runtime only for isolated unit tests."""

        return cls(
            **kwargs,
            _test_authority=_TEST_CONTEXT_RUNTIME_AUTHORITY,
        )

    @classmethod
    def from_factory(
        cls,
        *,
        owner_id: str,
        execution_factory: DurableContextRuntimeFactory,
        provider_turns_enabled: bool = False,
        tool_output_management_active: bool = False,
    ) -> ContextRuntime:
        return cls(
            owner_id=owner_id,
            execution_factory=execution_factory,
            provider_turns_enabled=provider_turns_enabled,
            tool_output_management_active=tool_output_management_active,
        )

    def bind_context(
        self,
        context: HarnessContext,
        *,
        _binding_authority: object | None = None,
        _shadow_mode: bool = False,
    ) -> None:
        if self.execution_factory is None:
            return
        bundle = self.execution_factory.bind(context)
        if type(_shadow_mode) is not bool:
            raise TypeError("context shadow mode must be an exact boolean")
        from ..tools.output_management import ToolOutputManager

        runtime_config = context.event.get("tool_runtime_config")
        attempt_key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        with self._bundle_lock:
            frozen_output_config = copy.deepcopy(
                self._attempt_tool_output_configs.get(attempt_key)
            )
        deferred_active_output_binding = (
            self.tool_output_management_active
            and not _shadow_mode
            and frozen_output_config is None
            and (
                not isinstance(runtime_config, Mapping)
                or "tool_output_management" not in runtime_config
            )
        )
        # A final model boundary may replace the bootstrap toolkit with a
        # sealed execution toolkit. Active output ownership must wait until
        # the shared tool-authority path prepares the first invocation.
        if frozen_output_config is not None:
            supplied_output_config = (
                {
                    key: copy.deepcopy(runtime_config[key])
                    for key in (
                        "tool_output_management",
                        "tool_output_policy_map",
                    )
                    if key in runtime_config
                }
                if isinstance(runtime_config, Mapping)
                else {}
            )
            if supplied_output_config and supplied_output_config != frozen_output_config:
                raise ContextExecutionBundleError(
                    "tool output snapshot changed after the attempt was bound"
                )
            merged_runtime_config = (
                dict(runtime_config) if isinstance(runtime_config, Mapping) else {}
            )
            merged_runtime_config.update(frozen_output_config)
            runtime_config = merged_runtime_config
            context.event["tool_runtime_config"] = runtime_config
        configured_manager = (
            ToolOutputManager.disabled_default(attempt_id=bundle.attempt.attempt_id)
            if deferred_active_output_binding
            else ToolOutputManager.from_runtime_config(
                runtime_config,
                attempt_id=bundle.attempt.attempt_id,
            )
        )
        if _shadow_mode and configured_manager.active:
            configured_manager = ToolOutputManager(
                active=False,
                default_policy=configured_manager.default_policy,
                policies=configured_manager.policies,
                attempt_id=bundle.attempt.attempt_id,
            )
        if (
            bundle.tool_output_manager.runtime_snapshot()
            != configured_manager.runtime_snapshot()
        ):
            if bundle.projector.bound_tool_output_manager is not None:
                raise ContextExecutionBundleError(
                    "tool output snapshot changed after the attempt was bound"
                )
            object.__setattr__(
                bundle,
                "tool_output_manager",
                configured_manager,
            )
        if not deferred_active_output_binding:
            bundle.projector.bind_tool_output_manager(bundle.tool_output_manager)
            if bundle.tool_output_manager.active:
                with self._bundle_lock:
                    self._attempt_tool_output_configs.setdefault(
                        attempt_key,
                        {
                            key: copy.deepcopy(runtime_config[key])
                            for key in (
                                "tool_output_management",
                                "tool_output_policy_map",
                            )
                            if key in runtime_config
                        },
                    )
        supplied_output_manager = context.event.get("tool_output_manager")
        if supplied_output_manager is not None and (
            supplied_output_manager is not bundle.tool_output_manager
        ):
            raise ContextExecutionBundleError(
                "tool output manager changed the attempt-scoped bundle binding"
            )
        context.event["tool_output_manager"] = bundle.tool_output_manager
        if self.provider_turns_enabled != (bundle.provider_turn_service is not None):
            raise ContextExecutionBundleError(
                "provider turn service does not match the runtime gate"
            )
        from ..kernel.run_ledger import attach_state_run_bundle_ledger

        state_run_ledger = getattr(context.state, "run_ledger", None)
        provider_turn_ownership = getattr(
            state_run_ledger,
            "provider_turn_ownership",
            None,
        )
        if provider_turn_ownership is not None:
            if self.provider_turns_enabled:
                if provider_turn_ownership.ledger is not bundle.run_bundle_ledger:
                    raise ContextExecutionBundleError(
                        "active Context provider ownership changed its accounting ledger"
                    )
                attach_state_run_bundle_ledger(
                    context.state,
                    bundle.run_bundle_ledger,
                )
            # A shadow Context runtime owns semantic events only.  An explicit
            # production provider owner already bound the accounting ledger
            # before bootstrap and must remain authoritative for this run.
        else:
            attach_state_run_bundle_ledger(
                context.state,
                bundle.run_bundle_ledger,
            )
        binding_key = attempt_key
        raw_guard = context.event.get("execution_guard")
        guard_binding = None
        if raw_guard is not None:
            if _binding_authority is not _CONTEXT_EXECUTION_BINDING_AUTHORITY:
                raise ContextExecutionBundleError(
                    "execution guard binding requires the system bootstrap harness"
                )
            from .tool_executor import (
                _resolve_official_execution_guard,
                _same_execution_guard_chain,
            )

            lease, root_guard, guard_chain = _resolve_official_execution_guard(
                raw_guard
            )
            if lease.execution_id != bundle.attempt.generation.execution_id:
                raise ContextExecutionBundleError(
                    "bootstrap execution guard does not match its attempt"
                )
            guard_binding = _BoundExecutionGuard(
                guard=raw_guard,
                root_guard=root_guard,
                execution_id=lease.execution_id,
                owner_id=lease.owner_id,
                fencing_token=lease.fencing_token,
                chain=guard_chain,
            )
        interaction_binding = _resolve_bootstrap_interaction_authority(
            context,
            guard_binding=guard_binding,
        )
        should_bootstrap = False
        with self._bundle_lock:
            previous_key = self._run_bindings.get(bundle.attempt.attempt_id)
            if previous_key is not None and previous_key != binding_key:
                raise ContextExecutionBundleError(
                    "attempt ID is already bound to a different execution"
                )
            existing = self._attempt_bundles.get(binding_key)
            if existing is None:
                self._attempt_bundles[binding_key] = bundle
                self._run_bindings[bundle.attempt.attempt_id] = binding_key
                should_bootstrap = True
            elif existing.attempt != bundle.attempt:
                raise ContextExecutionBundleError(
                    "attempt bundle binding changed during bootstrap"
                )
            previous_interaction = self._attempt_interaction_authorities.get(
                binding_key
            )
            if previous_interaction is not None:
                if (
                    interaction_binding.session_id != previous_interaction.session_id
                    or interaction_binding.loop is not previous_interaction.loop
                    or interaction_binding.interaction_runtime
                    is not previous_interaction.interaction_runtime
                    or interaction_binding.memory_runtime
                    is not previous_interaction.memory_runtime
                    or interaction_binding.memory_store
                    is not previous_interaction.memory_store
                ):
                    raise ContextExecutionBundleError(
                        "bootstrap interaction authority identity changed"
                    )
            self._attempt_interaction_authorities[binding_key] = interaction_binding
            if guard_binding is not None:
                previous_guard = self._attempt_execution_guards.get(binding_key)
                if previous_guard is not None:
                    same_token = (
                        guard_binding.fencing_token == previous_guard.fencing_token
                    )
                    invalid_rebind = (
                        guard_binding.fencing_token < previous_guard.fencing_token
                        or (
                            same_token
                            and (
                                guard_binding.owner_id != previous_guard.owner_id
                                or guard_binding.root_guard
                                is not previous_guard.root_guard
                                or not _same_execution_guard_chain(
                                    guard_binding.chain,
                                    previous_guard.chain,
                                )
                            )
                        )
                    )
                    if invalid_rebind:
                        raise ContextExecutionBundleError(
                            "bootstrap execution guard authority regressed or changed"
                        )
                self._attempt_execution_guards[binding_key] = guard_binding
            elif (
                _binding_authority is _CONTEXT_EXECUTION_BINDING_AUTHORITY
                and binding_key in self._attempt_execution_guards
            ):
                raise ContextExecutionBundleError(
                    "repeated bootstrap omitted its execution guard binding"
                )
        if not should_bootstrap:
            return
        try:
            if not self._bootstrap_prepared_subagent_input(bundle):
                self.execution_factory.bootstrap(bundle, context)
        except Exception:
            with self._bundle_lock:
                if self._attempt_bundles.get(binding_key) is bundle:
                    self._attempt_bundles.pop(binding_key, None)
                    self._run_bindings.pop(bundle.attempt.attempt_id, None)
                    self._attempt_execution_guards.pop(binding_key, None)
                    self._attempt_interaction_authorities.pop(
                        binding_key,
                        None,
                    )
            raise

    def bind_execution_toolkit(self, context: HarnessContext) -> None:
        """Freeze active output policy from the sealed execution toolkit."""

        if not isinstance(context, HarnessContext):
            raise TypeError("execution toolkit binding requires a HarnessContext")
        if not self.tool_output_management_active:
            return
        bundle = self._bundle_for_context(context)
        from ..tools import Toolkit
        from ..tools.output_management import ToolOutputManager

        toolkit = context.event.get("toolkit")
        if not isinstance(toolkit, Toolkit):
            raise ContextExecutionBundleError(
                "active tool output requires an execution Toolkit"
            )
        existing_config = context.event.get("tool_runtime_config")
        merged_config = (
            dict(existing_config) if isinstance(existing_config, Mapping) else {}
        )
        merged_config.pop("tool_output_management", None)
        merged_config.pop("tool_output_policy_map", None)
        merged_config.update(
            ToolOutputManager.active_runtime_config_for_toolkit(
                toolkit,
                attempt_id=bundle.attempt.attempt_id,
            )
        )
        configured_manager = ToolOutputManager.from_runtime_config(
            merged_config,
            attempt_id=bundle.attempt.attempt_id,
        )
        attempt_key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        with self._bundle_lock:
            frozen_output_config = copy.deepcopy(
                self._attempt_tool_output_configs.get(attempt_key)
            )
        candidate_output_config = {
            key: copy.deepcopy(merged_config[key])
            for key in ("tool_output_management", "tool_output_policy_map")
            if key in merged_config
        }
        bound_manager = bundle.projector.bound_tool_output_manager
        if (
            bound_manager is not None
            and bound_manager.runtime_snapshot() != configured_manager.runtime_snapshot()
        ):
            raise ContextExecutionBundleError(
                "execution toolkit changed the frozen tool output policy map"
            )
        if (
            frozen_output_config is not None
            and frozen_output_config != candidate_output_config
        ):
            raise ContextExecutionBundleError(
                "execution toolkit changed the frozen tool output policy map"
            )
        durable_output_config = self._load_or_persist_tool_output_snapshot(
            bundle,
            candidate_output_config,
        )
        if durable_output_config != candidate_output_config:
            raise ContextExecutionBundleError(
                "execution toolkit changed the durable tool output policy map"
            )
        if (
            frozen_output_config is not None
            and frozen_output_config != durable_output_config
        ):
            raise ContextExecutionBundleError(
                "in-memory and durable tool output snapshots disagree"
            )
        if bound_manager is not None:
            if (
                bound_manager.runtime_snapshot()
                != configured_manager.runtime_snapshot()
                or durable_output_config != candidate_output_config
            ):
                raise ContextExecutionBundleError(
                    "execution toolkit changed the frozen tool output policy map"
                )
            restored_config = (
                dict(existing_config) if isinstance(existing_config, Mapping) else {}
            )
            restored_config.update(frozen_output_config or candidate_output_config)
            context.event["tool_runtime_config"] = restored_config
            context.event["tool_output_manager"] = bound_manager
            return
        object.__setattr__(bundle, "tool_output_manager", configured_manager)
        bundle.projector.bind_tool_output_manager(configured_manager)
        with self._bundle_lock:
            self._attempt_tool_output_configs[attempt_key] = durable_output_config
        context.event["tool_runtime_config"] = merged_config
        context.event["tool_output_manager"] = configured_manager

    @staticmethod
    def _tool_output_snapshot_draft(
        bundle: ContextExecutionBundle,
        runtime_config: Mapping[str, Any],
    ):
        """Build the one idempotent durable policy receipt for an attempt."""

        from ..journal.runtime import SemanticEventDraft

        config = copy.deepcopy(dict(runtime_config))
        identity = {
            "execution_id": bundle.attempt.generation.execution_id,
            "generation_id": bundle.attempt.generation.generation_id,
            "attempt_id": bundle.attempt.attempt_id,
        }
        identity_digest = _canonical_sha256(identity)
        config_digest = _canonical_sha256(config)
        return SemanticEventDraft(
            event_id=f"tool-output-snapshot-{identity_digest}",
            event_type=_TOOL_OUTPUT_SNAPSHOT_EVENT_TYPE,
            attempt=bundle.attempt,
            operation_id=f"context-tool-output-snapshot-{identity_digest}",
            payload={
                "schema": _TOOL_OUTPUT_SNAPSHOT_SCHEMA,
                "runtime_config": config,
                "runtime_config_sha256": config_digest,
            },
        )

    @staticmethod
    def _recover_tool_output_snapshot(
        bundle: ContextExecutionBundle,
    ) -> dict[str, Any] | None:
        """Recover and strictly validate the attempt's durable policy receipt."""

        matches = tuple(
            event
            for event in bundle.journal.capture_snapshot().events
            if event.attempt == bundle.attempt
            and event.event_type == _TOOL_OUTPUT_SNAPSHOT_EVENT_TYPE
        )
        if not matches:
            return None
        if len(matches) != 1:
            raise ContextExecutionBundleError(
                "attempt has conflicting durable tool output snapshots"
            )
        payload = matches[0].payload
        if set(payload) != {
            "schema",
            "runtime_config",
            "runtime_config_sha256",
        } or payload.get("schema") != _TOOL_OUTPUT_SNAPSHOT_SCHEMA:
            raise ContextExecutionBundleError("durable tool output snapshot is invalid")
        from ..journal.models import _thaw_json

        config = _thaw_json(payload.get("runtime_config"))
        digest = payload.get("runtime_config_sha256")
        if (
            not isinstance(config, Mapping)
            or not isinstance(digest, str)
            or digest != _canonical_sha256(config)
        ):
            raise ContextExecutionBundleError("durable tool output snapshot is corrupt")
        recovered = copy.deepcopy(dict(config))
        expected_keys = {"tool_output_management", "tool_output_policy_map"}
        if not set(recovered).issubset(expected_keys) or (
            "tool_output_management" not in recovered
        ):
            raise ContextExecutionBundleError("durable tool output snapshot is malformed")
        from ..tools.output_management import ToolOutputManager

        manager = ToolOutputManager.from_runtime_config(
            recovered,
            attempt_id=bundle.attempt.attempt_id,
        )
        if not manager.active:
            raise ContextExecutionBundleError("durable tool output snapshot is inactive")
        return recovered

    def _load_or_persist_tool_output_snapshot(
        self,
        bundle: ContextExecutionBundle,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist before first execution, or return the exact cold receipt."""

        recovered = self._recover_tool_output_snapshot(bundle)
        if recovered is not None:
            return recovered
        draft = self._tool_output_snapshot_draft(bundle, candidate)
        bundle.durable_event_sink.append_projected(draft)
        return copy.deepcopy(dict(candidate))

    def prebind_run_bundle_ledger(
        self,
        *,
        state: RunState,
        run_id: str,
    ):
        """Resolve only the attempt accounting capability before inference."""

        if self.execution_factory is None:
            return None
        context = HarnessContext(
            state=state,
            phase="bootstrap",
            event={"run_id": str(run_id or "")},
        )
        return self.execution_factory.bind(context).run_bundle_ledger

    def prebind_provider_turn_ownership(
        self,
        *,
        state: RunState,
        run_id: str,
        identity,
    ):
        """Resolve the generic exact-send owner before any auxiliary inference."""

        if self.execution_factory is None or not self.provider_turns_enabled:
            return None
        context = HarnessContext(
            state=state,
            phase="bootstrap",
            event={"run_id": str(run_id or "")},
        )
        bundle = self.execution_factory.bind(context)
        if bundle.provider_turn_service is None or bundle.run_bundle_ledger is None:
            raise ContextExecutionBundleError(
                "enforcing provider turn bundle lacks its accounting owner"
            )
        from ..providers.turn_ownership import ProviderTurnOwnership

        return ProviderTurnOwnership(
            identity=identity,
            service=bundle.provider_turn_service,
            ledger=bundle.run_bundle_ledger,
        )

    def skill_activation_journal(self, context: HarnessContext):
        """Return the current owner-bound journal and attempt for SkillsModule."""
        if self.execution_factory is not None:
            bundle = self._bundle_for_context(context)
            return bundle.attempt, bundle.journal
        factory = self.request_factory
        if not hasattr(factory, "journal") or not hasattr(factory, "attempt"):
            raise ContextExecutionBundleError("skills require a bound context journal")
        return factory.attempt, factory.journal

    def compile_context(self, context: HarnessContext) -> ContextCompileResult:
        if self.execution_factory is None:
            request_factory = self.request_factory
            compiler = self.compiler
            attempt_key = self._attempt_key(context.event)
        else:
            bundle = self._bundle_for_context(context)
            request_factory = bundle.request_factory
            compiler = bundle.coordinator
            attempt_key = (
                bundle.attempt.generation.execution_id,
                bundle.attempt.attempt_id,
            )
        self._raise_latched_failure(attempt_key)
        if not callable(request_factory):
            raise TypeError("context runtime request_factory must be callable")
        request = request_factory(context)
        if not isinstance(request, ContextCompileRequest):
            raise TypeError(
                "context runtime request_factory must return ContextCompileRequest"
            )
        if (
            request.checkpoint_ref is not None
            or request.checkpoint_request_id is not None
        ):
            raise ContextCheckpointBindingForbiddenError()
        state = getattr(context, "state", None)
        ledger = getattr(state, "run_ledger", None)
        capture_metrics = getattr(ledger, "identity", None) is not None
        subject_id = "context-request:unavailable"
        result = None
        try:
            request_digest = _canonical_sha256(request.to_dict())
            subject_id = f"context-request:{request_digest}"
            result = compiler.compile(request)
            if not isinstance(result, ContextCompileResult):
                raise TypeError(
                    "context runtime compiler must return ContextCompileResult"
                )
            if (
                result.checkpoint_requests
                or result.diagnostics.get("status") == "checkpoint_required"
            ):
                raise ContextCheckpointPersistenceRequiredError(result)
            if result.envelope is None:
                raise ContextBuildEnvelopeRequiredError(result)
            if result.envelope.status is ContextBuildStatus.UNAVAILABLE:
                raise ContextBuildUnavailableError(result)
        except Exception as exc:
            error_code = str(getattr(exc, "code", "context_build_failed") or "")
            if re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", error_code) is None:
                error_code = "context_build_failed"
            if capture_metrics:
                ledger.record_metric_event(
                    kind="context_build",
                    subject_id=subject_id,
                    outcome="failed",
                    error_category="context",
                    error_code=error_code,
                )
                ledger.record_metric_event(
                    kind="error",
                    subject_id=f"context-error:{subject_id}",
                    outcome="failed",
                    error_category="context",
                    error_code=error_code,
                )
            raise
        assert result is not None and result.envelope is not None
        build_id = str(result.envelope.build_id)
        subject_id = build_id
        evidence = (
            opaque_metric_evidence_ref(
                kind="context_event",
                source_id=build_id,
            ),
        )
        if capture_metrics:
            ledger.record_metric_event(
                kind="context_build",
                subject_id=subject_id,
                evidence_refs=evidence,
            )
            if result.diagnostics.get("compacted") is True:
                ledger.record_metric_event(
                    kind="context_compaction",
                    subject_id=build_id,
                    evidence_refs=evidence,
                )
        return result

    def prepare_subagent_completion_sink(
        self,
        context: HarnessContext,
        *,
        call_id: str,
    ) -> ContextSubagentCompletionSink | None:
        """Bind child completion persistence to the current parent attempt."""

        if self.execution_factory is None:
            return None
        if not isinstance(context, HarnessContext):
            raise TypeError("subagent completion binding requires HarnessContext")
        bundle = self._bundle_for_context(context)
        attempt_key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        self._raise_latched_failure(attempt_key)
        return ContextSubagentCompletionSink(
            _runtime=self,
            parent_attempt=bundle.attempt,
            call_id=call_id,
        )

    def prepared_subagent_input(
        self,
        child_run_id: str,
    ) -> PreparedSubagentInput | None:
        """Return the immutable prepared input for one exact child identity."""

        from ..journal.models import _required_text
        from .subagent_input import PreparedSubagentInput

        normalized_child_run_id = _required_text(
            child_run_id,
            "child_run_id",
            identifier=True,
        )
        with self._bundle_lock:
            prepared = self._prepared_subagent_inputs.get(
                normalized_child_run_id
            )
        if prepared is None:
            return None
        if type(prepared) is not PreparedSubagentInput:
            raise ContextExecutionBundleError(
                "prepared subagent input has an invalid runtime binding"
            )
        if prepared.child_run_id != normalized_child_run_id:
            raise ContextExecutionBundleError(
                "prepared subagent input changed its child identity"
            )
        return prepared

    def _register_prepared_subagent_input(
        self,
        prepared: PreparedSubagentInput,
    ) -> PreparedSubagentInput:
        from .subagent_input import PreparedSubagentInput

        if type(prepared) is not PreparedSubagentInput:
            raise TypeError(
                "prepared must be an exact PreparedSubagentInput"
            )
        child_key = (
            prepared.child_attempt.generation.execution_id,
            prepared.child_attempt.attempt_id,
        )
        parent_run_id = prepared.parent_attempt.attempt_id
        with self._bundle_lock:
            previous = self._prepared_subagent_inputs.get(
                prepared.child_run_id
            )
            if previous is not None and previous != prepared:
                raise ContextExecutionBundleError(
                    "subagent input preparation changed its child binding"
                )
            previous_parent = self._subagent_parent_scopes.get(child_key)
            if previous_parent is not None and previous_parent != parent_run_id:
                raise ContextExecutionBundleError(
                    "prepared child changed its durable parent attempt"
                )
            self._prepared_subagent_inputs[prepared.child_run_id] = (
                previous or prepared
            )
            self._subagent_parent_scopes[child_key] = (
                previous_parent or parent_run_id
            )
            return previous or prepared

    def bind_prepared_subagent_input(
        self,
        child_run_id: str,
        *,
        prepared: PreparedSubagentInput | None = None,
    ) -> ContextExecutionBundle:
        """Cold-bind one prepared child bundle without running its provider."""

        if self.execution_factory is None:
            raise ContextExecutionBundleError(
                "prepared subagent input binding requires a factory runtime"
            )
        if prepared is not None:
            from ..journal.models import _required_text
            from .subagent_input import PreparedSubagentInput

            normalized_child_run_id = _required_text(
                child_run_id,
                "child_run_id",
                identifier=True,
            )
            if type(prepared) is not PreparedSubagentInput:
                raise TypeError(
                    "prepared must be an exact PreparedSubagentInput"
                )
            if prepared.child_run_id != normalized_child_run_id:
                raise ContextExecutionBundleError(
                    "external prepared input changed its child identity"
                )
            prepared = self._register_prepared_subagent_input(prepared)
        else:
            prepared = self.prepared_subagent_input(child_run_id)
        if prepared is None:
            raise ContextExecutionBundleError(
                "subagent input child identity was not prepared"
            )
        parent_key = (
            prepared.parent_attempt.generation.execution_id,
            prepared.parent_attempt.attempt_id,
        )
        child_key = (
            prepared.child_attempt.generation.execution_id,
            prepared.child_attempt.attempt_id,
        )
        self._raise_latched_failure(parent_key)
        self._raise_latched_failure(child_key)
        bundle = self.execution_factory._bind_exact_attempt(
            prepared.child_attempt
        )
        if self.provider_turns_enabled != (bundle.provider_turn_service is not None):
            raise ContextExecutionBundleError(
                "provider turn service does not match the runtime gate"
            )
        registered = False
        with self._bundle_lock:
            previous_key = self._run_bindings.get(
                prepared.child_attempt.attempt_id
            )
            if previous_key is not None and previous_key != child_key:
                raise ContextExecutionBundleError(
                    "prepared child attempt ID is already bound to a different "
                    "execution"
                )
            existing = self._attempt_bundles.get(child_key)
            if existing is None:
                self._attempt_bundles[child_key] = bundle
                self._run_bindings[prepared.child_attempt.attempt_id] = child_key
                registered = True
            elif existing.attempt != prepared.child_attempt:
                raise ContextExecutionBundleError(
                    "prepared child bundle binding changed identity"
                )
            else:
                bundle = existing
        try:
            with self._bundle_lock:
                if (
                    prepared.child_attempt.attempt_id
                    not in self._subagent_input_receipts
                    and not self._bootstrap_prepared_subagent_input(bundle)
                ):
                    raise ContextExecutionBundleError(
                        "prepared child bundle lost its immutable input binding"
                    )
        except Exception:
            if registered:
                with self._bundle_lock:
                    if self._attempt_bundles.get(child_key) is bundle:
                        self._attempt_bundles.pop(child_key, None)
                        self._run_bindings.pop(
                            prepared.child_attempt.attempt_id,
                            None,
                        )
            raise
        return bundle

    def _prepare_subagent_input(
        self,
        *,
        parent_attempt: AttemptRef,
        call_id: str,
        child_run_id: str,
        child_id: str,
        mode: str,
        lineage,
        template_name: str | None,
        input_messages,
    ):
        """Prepare one restart-stable child task before child bootstrap."""

        from .subagent_input import (
            SubagentInputPreparation,
            prepare_subagent_input,
            recover_subagent_result,
        )

        if self.execution_factory is None:
            raise ContextExecutionBundleError(
                "subagent input persistence requires a factory runtime"
            )
        if not isinstance(parent_attempt, AttemptRef):
            parent_attempt = AttemptRef.from_dict(parent_attempt)
        parent_key = (
            parent_attempt.generation.execution_id,
            parent_attempt.attempt_id,
        )
        self._raise_latched_failure(parent_key)
        with self._bundle_lock:
            parent_bundle = self._attempt_bundles.get(parent_key)
        if parent_bundle is None or parent_bundle.attempt != parent_attempt:
            raise ContextExecutionBundleError(
                "subagent input parent attempt is not bound"
            )
        try:
            prepared = prepare_subagent_input(
                parent_bundle,
                call_id=call_id,
                child_run_id=child_run_id,
                child_id=child_id,
                mode=mode,
                lineage=lineage,
                template_name=template_name,
                input_messages=input_messages,
            )
            prepared = self._register_prepared_subagent_input(prepared)
            recovered = recover_subagent_result(parent_bundle, prepared)
            return SubagentInputPreparation(
                prepared=prepared,
                recovered_result=recovered,
            )
        except Exception as error:
            boundary_error = mark_durable_persistence_failure(error)
            first_failure = self._latch_failure(parent_key, boundary_error)
            if not first_failure:
                self._raise_latched_failure(parent_key)
            try:
                parent_bundle.partial_attempt_sink(
                    {
                        "type": "subagent.input.partial",
                        "run_id": parent_attempt.attempt_id,
                        "call_id": str(call_id or ""),
                        "child_run_id": str(child_run_id or ""),
                    },
                    boundary_error,
                )
            except Exception:
                pass
            if boundary_error is error:
                raise
            raise boundary_error from None

    def _bootstrap_prepared_subagent_input(
        self,
        bundle: ContextExecutionBundle,
    ) -> bool:
        """Persist a prepared child handoff/input before its first model turn."""

        from .subagent_input import persist_subagent_input

        with self._bundle_lock:
            prepared = self._prepared_subagent_inputs.get(
                bundle.attempt.attempt_id
            )
            previous_receipt = self._subagent_input_receipts.get(
                bundle.attempt.attempt_id
            )
        if prepared is None:
            return False
        if bundle.attempt != prepared.child_attempt:
            raise ContextExecutionBundleError(
                "subagent bootstrap changed its prepared attempt identity"
            )
        receipt = persist_subagent_input(prepared, bundle)
        with self._bundle_lock:
            if previous_receipt is not None and previous_receipt != receipt:
                raise ContextExecutionBundleError(
                    "subagent bootstrap changed its durable input receipt"
                )
            self._subagent_input_receipts[bundle.attempt.attempt_id] = (
                previous_receipt or receipt
            )
        return True

    def _record_subagent_completion(
        self,
        *,
        parent_attempt: AttemptRef,
        call_id: str,
        child_run_id: str,
        result,
    ):
        """Copy one exact child result into its bound parent execution."""

        from ..journal.models import EventRange, _required_text, _thaw_json
        from ..journal.runtime import SemanticEventDraft
        from ..subagents.types import SubagentResult
        from .host_adapter import ContextArtifactHandoffHostAdapter

        if self.execution_factory is None:
            raise ContextExecutionBundleError(
                "subagent completion persistence requires a factory runtime"
            )
        if not isinstance(parent_attempt, AttemptRef):
            parent_attempt = AttemptRef.from_dict(parent_attempt)
        normalized_call_id = _required_text(
            call_id,
            "call_id",
            identifier=True,
        )
        normalized_child_run_id = _required_text(
            child_run_id,
            "child_run_id",
            identifier=True,
        )
        if type(result) is not SubagentResult:
            raise TypeError("subagent completion requires an exact SubagentResult")

        parent_key = (
            parent_attempt.generation.execution_id,
            parent_attempt.attempt_id,
        )
        self._raise_latched_failure(parent_key)
        with self._bundle_lock:
            parent_bundle = self._attempt_bundles.get(parent_key)
            child_key = self._run_bindings.get(normalized_child_run_id)
            child_bundle = (
                self._attempt_bundles.get(child_key) if child_key is not None else None
            )
        if parent_bundle is None or parent_bundle.attempt != parent_attempt:
            raise ContextExecutionBundleError(
                "subagent completion parent attempt is not bound"
            )

        try:
            if (
                child_bundle is None
                or child_bundle.attempt.attempt_id != normalized_child_run_id
            ):
                raise ContextExecutionBundleError(
                    "subagent completion child attempt is not bound"
                )
            snapshot = child_bundle.journal.capture_snapshot()
            if not any(
                event.attempt == child_bundle.attempt
                for event in snapshot.events
            ):
                raise ContextExecutionBundleError(
                    "subagent completion child journal is empty"
                )

            checkpoint_identity = "\x1f".join(
                (
                    child_bundle.attempt.generation.execution_id,
                    child_bundle.attempt.generation.generation_id,
                    child_bundle.attempt.attempt_id,
                    "subagent-result-checkpoint",
                )
            )
            checkpoint_digest = hashlib.sha256(
                checkpoint_identity.encode("utf-8")
            ).hexdigest()
            result_artifact = child_bundle.artifacts.persist_exact_json(
                result.to_record_dict(),
                operation_id="artifact.subagent-result." + checkpoint_digest,
                operation_binding={
                    "kind": "subagent_result_checkpoint",
                    "child_attempt": child_bundle.attempt.to_dict(),
                    "child_run_id": normalized_child_run_id,
                },
            )
            projected_checkpoint = child_bundle.projector(
                {
                    "type": "subagent_completed",
                    "run_id": normalized_child_run_id,
                    "child_run_id": normalized_child_run_id,
                    "status": result.status,
                }
            )
            if projected_checkpoint is None:
                raise ContextExecutionBundleError(
                    "subagent completion checkpoint was not projected"
                )
            checkpoint_payload = _thaw_json(projected_checkpoint.payload)
            checkpoint_payload["result_checkpoint"] = {
                "schema": "unchain.subagent_result_checkpoint.v1",
                "child_attempt": child_bundle.attempt.to_dict(),
                "result_artifact": result_artifact.to_dict(),
            }
            checkpoint_draft = SemanticEventDraft(
                event_id=projected_checkpoint.event_id,
                event_type=projected_checkpoint.event_type,
                attempt=projected_checkpoint.attempt,
                operation_id=projected_checkpoint.operation_id,
                payload=checkpoint_payload,
                resource_refs=(result_artifact.ref,),
            )
            checkpoint = child_bundle.durable_event_sink.append_projected(
                checkpoint_draft
            )
            source_range = EventRange(
                start=checkpoint.cursor,
                end=checkpoint.cursor,
            )
            identity = "\x1f".join(
                (
                    parent_attempt.generation.execution_id,
                    parent_attempt.generation.generation_id,
                    parent_attempt.attempt_id,
                    normalized_call_id,
                    normalized_child_run_id,
                )
            )
            operation_id = (
                "handoff.subagent."
                + hashlib.sha256(identity.encode("utf-8")).hexdigest()
            )
            adapter = ContextArtifactHandoffHostAdapter(
                recorder=parent_bundle.handoff_recorder,
            )
            return adapter.record_subagent_result(
                result,
                child_attempt=child_bundle.attempt,
                source_event_range=source_range,
                artifact_refs=(),
                operation_id=operation_id,
            )
        except Exception as error:
            boundary_error = mark_durable_persistence_failure(error)
            first_failure = self._latch_failure(parent_key, boundary_error)
            if not first_failure:
                self._raise_latched_failure(parent_key)
            try:
                parent_bundle.partial_attempt_sink(
                    {
                        "type": "subagent.handoff.partial",
                        "run_id": parent_attempt.attempt_id,
                        "call_id": normalized_call_id,
                        "child_run_id": normalized_child_run_id,
                    },
                    boundary_error,
                )
            except Exception:
                pass
            if boundary_error is error:
                raise
            raise boundary_error from None

    def prepare_tool_execution(self, context: HarnessContext):
        """Derive and freeze one tool invocation from durable runtime state."""

        from ..journal import EventCursor, SideEffectRecoveryState
        from ..journal.models import _thaw_json
        from ..kernel.types import ToolCall
        from .tool_boundary import (
            DurableToolApprovalState,
            DurableToolExecutionSubject,
            _canonical_digest,
        )
        from .tool_executor import (
            DurableToolExecutionRequest,
            DurableToolExecutor,
            DurableToolExecutorContractError,
        )

        if not isinstance(context, HarnessContext):
            raise TypeError("tool preparation requires a HarnessContext")
        if context.phase != "on_tool_call":
            raise DurableToolExecutorContractError(
                "durable tool preparation requires the on_tool_call phase"
            )
        if self.execution_factory is None:
            raise DurableToolExecutorContractError(
                "durable tool preparation requires a factory ContextRuntime"
            )
        from .tool_routes import ContextToolRouteResolver

        bundle = self._bundle_for_context(context)
        self.bind_execution_toolkit(context)
        attempt_key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        self._raise_latched_failure(attempt_key)
        with self._bundle_lock:
            guard_binding = self._attempt_execution_guards.get(attempt_key)
            interaction_binding = self._attempt_interaction_authorities.get(attempt_key)
        if guard_binding is None:
            raise DurableToolExecutorContractError(
                "durable tool preparation has no bootstrap guard binding"
            )

        raw_tool_call = context.event.get("tool_call")
        if not isinstance(raw_tool_call, ToolCall):
            raise DurableToolExecutorContractError(
                "durable tool preparation requires one tool call"
            )
        call_id = str(raw_tool_call.call_id or "").strip()
        recovery = bundle.tool_boundary.sink.recover_tool_side_effect(call_id)
        intent = recovery.intent_event
        if (
            recovery.state is SideEffectRecoveryState.CORRUPT
            or intent is None
            or intent.event_type != "tool_call"
        ):
            raise DurableToolExecutorContractError(
                "durable tool preparation requires an exact journal intent"
            )
        tool_name = intent.payload.get("tool_name")
        iteration = intent.payload.get("iteration")
        original_arguments = _thaw_json(intent.payload.get("arguments", {}))
        if (
            not isinstance(tool_name, str)
            or not tool_name
            or isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 0
        ):
            raise DurableToolExecutorContractError(
                "durable tool preparation has an invalid journal intent"
            )
        if (
            raw_tool_call.call_id != call_id
            or raw_tool_call.name != tool_name
            or _canonical_digest(raw_tool_call.arguments or {})
            != _canonical_digest(original_arguments)
        ):
            raise DurableToolExecutorContractError(
                "host tool call does not match its durable journal intent"
            )

        if self.provider_turns_enabled:
            self._verify_provider_tool_execution_binding(
                context,
                bundle=bundle,
                iteration=iteration,
            )

        if recovery.state in {
            SideEffectRecoveryState.SEALED_COMPLETION_FINALIZABLE,
            SideEffectRecoveryState.TERMINAL_RESULT_REUSABLE,
        }:
            started = recovery.started_event
            if (
                started is None
                or started.payload.get("tool_name") != tool_name
                or started.payload.get("call_id") != call_id
                or started.payload.get("iteration") != iteration
            ):
                raise DurableToolExecutorContractError(
                    "cold tool recovery has no exact durable started receipt"
                )
            try:
                recorded_subject = DurableToolExecutionSubject.from_dict(
                    started.payload.get("execution_subject")
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                raise DurableToolExecutorContractError(
                    "cold tool recovery has an invalid execution subject"
                ) from exc
            if (
                started.payload.get("execution_subject_sha256")
                != recorded_subject.sha256
                or recorded_subject.intent_cursor
                != EventCursor(intent.store_seq, intent.event_id)
                or recorded_subject.original_arguments_sha256
                != _canonical_digest(original_arguments)
            ):
                raise DurableToolExecutorContractError(
                    "cold tool recovery changed its durable execution subject"
                )
            recovery_subject = DurableToolExecutionSubject(
                intent_cursor=recorded_subject.intent_cursor,
                original_arguments_sha256=(recorded_subject.original_arguments_sha256),
                effective_arguments_sha256=(
                    recorded_subject.effective_arguments_sha256
                ),
                approval_state=recorded_subject.approval_state,
                approval_request_sha256=(recorded_subject.approval_request_sha256),
                approval_receipt_sha256=(recorded_subject.approval_receipt_sha256),
                route_kind=recorded_subject.route_kind,
                route_manifest_sha256=(recorded_subject.route_manifest_sha256),
                terminal_handler_manifest_sha256=(
                    recorded_subject.terminal_handler_manifest_sha256
                ),
                execution_fence=guard_binding.guard.fence,
            )
            request = DurableToolExecutionRequest(
                tool_name=tool_name,
                call_id=call_id,
                iteration=iteration,
                subject=recovery_subject,
                tool_result_policy=(
                    bundle.tool_output_manager.resolve_policy_for_tool(
                        context.event.get("tool_runtime_config"),
                        tool_name=tool_name,
                    ).name
                ),
            )
            executor = DurableToolExecutor(
                boundary=bundle.tool_boundary,
                artifacts=bundle.artifacts,
                execution_guard=guard_binding.guard,
                expected_guard_chain=guard_binding.chain,
            )
            permit_key = (*attempt_key, call_id)
            with self._tool_permit_lock:
                if permit_key in self._prepared_tool_keys:
                    raise DurableToolExecutorContractError(
                        "durable tool execution is already prepared"
                    )
                if len(self._prepared_tool_bindings) >= MAX_PREPARED_TOOL_PERMITS:
                    raise DurableToolExecutorContractError(
                        "durable tool permit capacity exceeded"
                    )
                permit = _mint_tool_permit(self._tool_permit_authority)
                binding = _PreparedToolBinding(
                    permit=permit,
                    context=context,
                    attempt_key=attempt_key,
                    call_id=call_id,
                    request=request,
                    executor=executor,
                    invocation=None,
                    route=None,
                )
                self._prepared_tool_bindings[id(permit)] = binding
                self._prepared_tool_keys[permit_key] = id(permit)
            return permit

        trusted_event = dict(context.event)
        trusted_tool_call = ToolCall(
            call_id=call_id,
            name=tool_name,
            arguments=original_arguments,
        )
        trusted_event["tool_call"] = trusted_tool_call
        trusted_context = HarnessContext(
            state=context.state,
            phase=context.phase,
            event=trusted_event,
        )
        route_resolver = ContextToolRouteResolver()
        route = route_resolver.resolve(
            trusted_context,
            trusted_tool_call,
        )
        route = route_resolver.verify_route(route)
        toolkit = trusted_event.get("toolkit")
        from ..interaction import (
            INTERACTION_KIND_TOOL_APPROVAL,
            InteractionNotPendingError,
        )
        from ..interaction.effects import build_tool_approval_continuation
        from ..interaction.requests import (
            build_tool_approval_interaction_request,
            ensure_interaction_binding_matches,
        )
        from ..tools.confirmation import prepare_tool_confirmation
        from ..tools.models import (
            ToolConfirmationResponse,
            ToolExecutionContext,
        )
        from ..tools.runtime import snapshot_durable_tool_exposure_plan
        from ..tools.toolkit import Toolkit
        from .tool_executor import DurableToolCompletionDraft

        if not isinstance(toolkit, Toolkit):
            raise DurableToolExecutorContractError(
                "durable tool preparation requires the exposed toolkit"
            )
        execution_context = ToolExecutionContext(
            session_id=str(context.state.session_state.session_id or ""),
            run_id=bundle.attempt.attempt_id,
            provider=str(context.state.provider_state.provider or ""),
            model=str(context.state.provider_state.model or ""),
            iteration=iteration,
            memory_namespace=str(context.state.session_state.memory_namespace or ""),
            tool_runtime_config=(
                copy.deepcopy(context.event.get("tool_runtime_config"))
                if isinstance(context.event.get("tool_runtime_config"), dict)
                else {}
            ),
            tool_name=tool_name,
            call_id=call_id,
            turn_id=f"{bundle.attempt.attempt_id}:turn-{iteration}",
            run_state=context.state,
        )
        confirmation = prepare_tool_confirmation(
            toolkit=toolkit,
            tool_call=trusted_tool_call,
            execution_context=execution_context,
            execution_guard=guard_binding.guard,
        )
        if confirmation.resolver_error is not None:
            raise DurableToolExecutorContractError(
                "durable tool approval policy resolver failed"
            )

        effective_arguments = original_arguments
        approval_state = DurableToolApprovalState.NOT_REQUIRED
        approval_request_sha256 = ""
        approval_receipt_sha256 = ""
        terminal_handler = route.terminal_handler
        terminal_handler_manifest_sha256 = route.terminal_handler_manifest_sha256
        if confirmation.needs_confirmation_response:
            session_id = str(context.state.session_state.session_id or "").strip()
            has_resume_request = "interaction_request" in context.event
            has_resume_response = "interaction_response" in context.event
            if has_resume_request != has_resume_response:
                raise DurableToolExecutorContractError(
                    "durable tool approval resume fields are incomplete"
                )
            durable_snapshot = None
            with self._bundle_lock:
                from ..interaction.runtime import DurableInteractionRuntime
                from ..memory.runtime import KernelMemoryRuntime

                if (
                    interaction_binding is None
                    or session_id != interaction_binding.session_id
                    or type(interaction_binding.interaction_runtime)
                    is not DurableInteractionRuntime
                    or type(interaction_binding.memory_runtime)
                    is not KernelMemoryRuntime
                    or interaction_binding.interaction_runtime.memory_runtime
                    is not interaction_binding.memory_runtime
                    or interaction_binding.memory_runtime.store
                    is not interaction_binding.memory_store
                    or type(interaction_binding.receipt_reader)
                    is not DurableInteractionRuntime
                    or type(interaction_binding.receipt_reader.memory_runtime)
                    is not KernelMemoryRuntime
                    or interaction_binding.receipt_reader.memory_runtime.store
                    is not interaction_binding.memory_store
                ):
                    raise DurableToolExecutorContractError(
                        "durable tool approval bootstrap interaction authority "
                        "changed"
                    )
                if has_resume_request:
                    try:
                        durable_snapshot = DurableInteractionRuntime.require_receipt(
                            interaction_binding.receipt_reader,
                            session_id,
                        )
                    except InteractionNotPendingError:
                        raise DurableToolExecutorContractError(
                            "raw tool approval cannot replace a durable receipt"
                        ) from None
            source_request = (
                durable_snapshot.request if durable_snapshot is not None else None
            )
            confirmation_request = confirmation.request
            if confirmation_request is None:
                raise DurableToolExecutorContractError(
                    "durable tool approval has no confirmation request"
                )
            authority_subject = {
                "schema": "unchain.context_tool_approval.v1",
                "intent_cursor": EventCursor(
                    intent.store_seq,
                    intent.event_id,
                ).to_dict(),
                "original_arguments_sha256": _canonical_digest(original_arguments),
                "route_kind": route.route_kind.value,
                "route_manifest_sha256": route.route_manifest_sha256,
                "terminal_handler_manifest_sha256": (
                    route.terminal_handler_manifest_sha256
                ),
            }
            interaction_request = build_tool_approval_interaction_request(
                state=context.state,
                toolkit=toolkit,
                tool_call=trusted_tool_call,
                confirmation_request=confirmation_request,
                run_id=bundle.attempt.attempt_id,
                source_request=source_request,
                extra_subject={
                    "context_v2_tool_authority": authority_subject,
                },
            )
            if durable_snapshot is None:
                continuation = build_tool_approval_continuation(
                    interaction_request=interaction_request,
                    payload=(
                        copy.deepcopy(context.event.get("payload"))
                        if isinstance(context.event.get("payload"), dict)
                        else {}
                    ),
                    response_format=context.event.get("response_format"),
                    max_iterations=int(context.event.get("max_iterations") or 0),
                    tool_calls=list(context.event.get("tool_calls") or []),
                    batch_state=context.state.tool_batch_state.copy(),
                    state=context.state,
                    run_id=bundle.attempt.attempt_id,
                    tool_exposure_plan=snapshot_durable_tool_exposure_plan(
                        list(context.event.get("tool_runtime_plugins") or [])
                    ),
                )
                return _new_tool_approval_pending(
                    interaction_request=interaction_request.to_dict(),
                    continuation=continuation,
                )
            if (
                durable_snapshot.request.kind != INTERACTION_KIND_TOOL_APPROVAL
                or durable_snapshot.receipt is None
            ):
                raise DurableToolExecutorContractError(
                    "durable tool approval receipt is invalid"
                )
            ensure_interaction_binding_matches(
                durable_snapshot.request,
                interaction_request,
            )
            approval = ToolConfirmationResponse.from_raw(
                durable_snapshot.receipt.response
            )
            approval_request_sha256 = interaction_request.request_digest
            approval_receipt_sha256 = durable_snapshot.receipt.receipt_digest
            if approval.approved:
                approval_state = DurableToolApprovalState.APPROVED
                if approval.modified_arguments is not None:
                    effective_arguments = copy.deepcopy(approval.modified_arguments)
            else:
                approval_state = DurableToolApprovalState.DENIED
                denial_result = {
                    "denied": True,
                    "tool": tool_name,
                    "reason": approval.reason or "User denied execution.",
                }

                def terminal_handler(_effective_arguments):
                    return DurableToolCompletionDraft(
                        result=denial_result,
                        should_observe=False,
                    )

                terminal_handler_manifest_sha256 = _canonical_digest(
                    {
                        "schema": "unchain.context_tool_terminal.v1",
                        "terminal": {
                            "kind": "approval_denied",
                            "tool_name": tool_name,
                            "call_id": call_id,
                        },
                    }
                )
        elif (
            "interaction_request" in context.event
            or "interaction_response" in context.event
        ):
            raise DurableToolExecutorContractError(
                "tool approval policy changed while interaction was pending"
            )

        subject = DurableToolExecutionSubject(
            intent_cursor=EventCursor(intent.store_seq, intent.event_id),
            original_arguments_sha256=_canonical_digest(original_arguments),
            effective_arguments_sha256=_canonical_digest(effective_arguments),
            approval_state=approval_state,
            approval_request_sha256=approval_request_sha256,
            approval_receipt_sha256=approval_receipt_sha256,
            route_kind=route.route_kind,
            route_manifest_sha256=route.route_manifest_sha256,
            terminal_handler_manifest_sha256=(terminal_handler_manifest_sha256),
            execution_fence=guard_binding.guard.fence,
        )
        request = DurableToolExecutionRequest(
            tool_name=tool_name,
            call_id=call_id,
            iteration=iteration,
            subject=subject,
            tool_result_policy=(
                bundle.tool_output_manager.resolve_policy_for_tool(
                    context.event.get("tool_runtime_config"),
                    tool_name=tool_name,
                ).name
            ),
        )
        executor = DurableToolExecutor(
            boundary=bundle.tool_boundary,
            artifacts=bundle.artifacts,
            execution_guard=guard_binding.guard,
            expected_guard_chain=guard_binding.chain,
        )
        invocation = executor.bind_invocation(
            request=request,
            effective_arguments=effective_arguments,
            terminal_handler=terminal_handler,
        )
        permit_key = (*attempt_key, call_id)
        with self._tool_permit_lock:
            if permit_key in self._prepared_tool_keys:
                raise DurableToolExecutorContractError(
                    "durable tool execution is already prepared"
                )
            if len(self._prepared_tool_bindings) >= MAX_PREPARED_TOOL_PERMITS:
                raise DurableToolExecutorContractError(
                    "durable tool permit capacity exceeded"
                )
            permit = _mint_tool_permit(self._tool_permit_authority)
            binding = _PreparedToolBinding(
                permit=permit,
                context=context,
                attempt_key=attempt_key,
                call_id=call_id,
                request=request,
                executor=executor,
                invocation=invocation,
                route=route,
            )
            self._prepared_tool_bindings[id(permit)] = binding
            self._prepared_tool_keys[permit_key] = id(permit)
        return permit

    def execute_prepared_tool(self, context: HarnessContext, permit):
        """Consume one runtime-owned permit and return a sealed durable receipt."""

        from .tool_executor import DurableToolExecutorContractError

        if not isinstance(context, HarnessContext):
            raise TypeError("prepared tool execution requires a HarnessContext")
        if self.execution_factory is None:
            raise DurableToolExecutorContractError(
                "durable prepared tool execution requires a factory ContextRuntime"
            )
        if not _is_runtime_tool_permit(permit, self._tool_permit_authority):
            raise DurableToolExecutorContractError(
                "prepared tool permit is not owned by this runtime"
            )
        with self._tool_permit_lock:
            binding = self._prepared_tool_bindings.pop(id(permit), None)
            if binding is not None:
                self._prepared_tool_keys.pop(
                    (*binding.attempt_key, binding.call_id),
                    None,
                )
        if binding is None or binding.permit is not permit:
            raise DurableToolExecutorContractError(
                "prepared tool permit is invalid, consumed, or expired"
            )
        if binding.context is not context:
            raise DurableToolExecutorContractError(
                "prepared tool permit does not match its exact context binding"
            )
        bundle = self._bundle_for_context(context)
        attempt_key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        if attempt_key != binding.attempt_key:
            raise DurableToolExecutorContractError(
                "prepared tool permit does not match its attempt binding"
            )
        self._raise_latched_failure(attempt_key)
        try:
            return binding.executor.execute(
                request=binding.request,
                invocation=binding.invocation,
            )
        except Exception as error:
            first_failure = self._latch_failure(attempt_key, error)
            if not first_failure:
                self._raise_latched_failure(attempt_key)
            try:
                bundle.partial_attempt_sink(
                    {
                        "type": "tool.execution.partial",
                        "run_id": bundle.attempt.attempt_id,
                        "tool_name": binding.request.tool_name,
                        "call_id": binding.request.call_id,
                        "execution_subject_sha256": (binding.request.subject.sha256),
                    },
                    error,
                )
            except Exception:
                pass
            raise

    def project_tool_result_for_model(self, context: HarnessContext, receipt):
        """Return the attempt manager's fresh view without altering the receipt."""

        from .tool_boundary import DurableToolResultReceipt
        from .tool_executor import DurableToolCompletionReceipt
        from ..journal.models import _thaw_json

        if not isinstance(context, HarnessContext):
            raise TypeError("tool result projection requires a HarnessContext")
        if type(receipt) not in {
            DurableToolResultReceipt,
            DurableToolCompletionReceipt,
        }:
            raise TypeError("tool result projection requires a durable receipt")
        bundle = self._bundle_for_context(context)
        if receipt.attempt != bundle.attempt:
            raise ContextExecutionBundleError(
                "tool result receipt does not match the bound attempt"
            )
        manager = bundle.tool_output_manager
        if not manager.active:
            return _thaw_json(receipt.visible_result)
        projection = receipt.model_projection
        if (
            not isinstance(projection, Mapping)
            or set(projection) != {"result", "metadata"}
            or not isinstance(projection["result"], Mapping)
            or not isinstance(projection["metadata"], Mapping)
        ):
            raise ContextExecutionBundleError(
                "durable tool model projection is missing or invalid"
            )
        return _thaw_json(projection["result"])

    def materialize_tool_transition(self, context: HarnessContext, receipt):
        """Resolve a receipt transition through this attempt's capabilities."""

        from ..kernel.types import ToolCall
        from .tool_executor import (
            DurableToolCompletionReceipt,
            DurableToolExecutorContractError,
        )
        from .tool_transitions import resolve_subagent_transition_cas

        if type(receipt) is not DurableToolCompletionReceipt:
            raise DurableToolExecutorContractError(
                "tool transition requires an exact durable completion receipt"
            )
        transition = receipt.transition
        if transition is None:
            return None
        bundle = self._bundle_for_context(context)
        attempt_key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        self._raise_latched_failure(attempt_key)
        tool_call = context.event.get("tool_call")
        try:
            if (
                receipt.attempt != bundle.attempt
                or transition.parent_attempt != bundle.attempt
                or not isinstance(tool_call, ToolCall)
                or receipt.call_id != tool_call.call_id
                or transition.call_id != tool_call.call_id
                or transition.execution_subject_sha256
                != receipt.execution_subject.sha256
            ):
                raise DurableToolExecutorContractError(
                    "tool transition changed its durable parent binding"
                )
            return resolve_subagent_transition_cas(
                artifacts=bundle.artifacts,
                transition=transition,
                current_state=(
                    context.state
                    if transition.kind == "subagent_terminal_handoff"
                    else context.state.subagent_state
                ),
            )
        except Exception as error:
            first_failure = self._latch_failure(attempt_key, error)
            if not first_failure:
                self._raise_latched_failure(attempt_key)
            try:
                bundle.partial_attempt_sink(
                    {
                        "type": "tool.transition.partial",
                        "run_id": bundle.attempt.attempt_id,
                        "tool_name": receipt.tool_name,
                        "call_id": receipt.call_id,
                        "execution_subject_sha256": (receipt.execution_subject.sha256),
                        "operation_id": transition.operation_id,
                    },
                    error,
                )
            except Exception:
                pass
            raise

    def bind_graph_step_scope(
        self,
        context: HarnessContext,
        *,
        workflow_node_id: str,
        workflow_step_index: int,
        workflow_step_count: int,
    ) -> None:
        """Bind one graph attempt's canonical workflow event scope.

        Runtime callbacks are persisted before a product host sees them.  The
        graph bootstrap therefore owns this binding so lifecycle, tool, and
        interaction events are journaled with their exact step identity rather
        than being decorated later by a UI callback.
        """

        from ..journal.models import _required_text

        if not isinstance(context, HarnessContext):
            raise TypeError("graph step scope requires a HarnessContext")
        node_id = _required_text(
            workflow_node_id,
            "workflow_node_id",
            identifier=True,
        )
        if (
            isinstance(workflow_step_index, bool)
            or not isinstance(workflow_step_index, int)
            or workflow_step_index < 0
        ):
            raise ValueError("workflow_step_index must be non-negative")
        if (
            isinstance(workflow_step_count, bool)
            or not isinstance(workflow_step_count, int)
            or workflow_step_count < 1
            or workflow_step_index >= workflow_step_count
        ):
            raise ValueError("workflow_step_count does not contain the step")
        bundle = self._bundle_for_context(context)
        key = (
            bundle.attempt.generation.execution_id,
            bundle.attempt.attempt_id,
        )
        scope = (node_id, workflow_step_index, workflow_step_count)
        with self._graph_scope_lock:
            existing = self._graph_scopes.get(key)
            if existing is not None and existing != scope:
                raise ContextExecutionBundleError(
                    "graph step scope changed its durable attempt identity"
                )
            self._graph_scopes[key] = scope

    def _event_with_graph_scope(
        self,
        event: Mapping[str, Any],
        attempt_key: tuple[str, str] | None,
    ) -> dict[str, Any]:
        projected = dict(event)
        if attempt_key is None:
            return projected
        with self._graph_scope_lock:
            scope = self._graph_scopes.get(attempt_key)
        if scope is None:
            return projected
        expected = {
            "workflow_node_id": scope[0],
            "workflow_step_index": scope[1],
            "workflow_step_count": scope[2],
        }
        supplied = tuple(key in projected for key in expected)
        if any(supplied) and not all(supplied):
            raise ContextExecutionBundleError(
                "graph runtime event has a partial workflow scope"
            )
        if all(supplied) and any(
            projected[key] != value for key, value in expected.items()
        ):
            raise ContextExecutionBundleError(
                "graph runtime event changed its workflow scope"
            )
        projected.update(expected)
        return projected

    def _event_with_subagent_parent_scope(
        self,
        event: Mapping[str, Any],
        attempt_key: tuple[str, str] | None,
    ) -> dict[str, Any]:
        projected = dict(event)
        if attempt_key is None:
            return projected
        with self._bundle_lock:
            parent_run_id = self._subagent_parent_scopes.get(attempt_key)
        if parent_run_id is None:
            return projected
        supplied_parent = projected.get("parent_run_id")
        if supplied_parent is not None and supplied_parent != parent_run_id:
            raise ContextExecutionBundleError(
                "subagent runtime event changed its durable parent attempt"
            )
        projected["parent_run_id"] = parent_run_id
        return projected

    def persist_event(self, event: Mapping[str, Any]) -> None:
        if not isinstance(event, Mapping):
            raise TypeError("durable event callback requires an object event")
        if self.execution_factory is None:
            durable_event_sink = self.durable_event_sink
            partial_attempt_sink = self.partial_attempt_sink
            attempt_key = self._attempt_key(event)
        else:
            bundle = self._bundle_for_event(event)
            durable_event_sink = bundle.durable_event_sink
            partial_attempt_sink = bundle.partial_attempt_sink
            attempt_key = (
                bundle.attempt.generation.execution_id,
                bundle.attempt.attempt_id,
            )
        self._raise_latched_failure(attempt_key)
        marker_event: dict[str, Any] = {}
        try:
            marker_event = self._event_with_subagent_parent_scope(
                event,
                attempt_key,
            )
            marker_event = self._event_with_graph_scope(
                marker_event,
                attempt_key,
            )
            durable_event = copy.deepcopy(marker_event)
            if not callable(durable_event_sink):
                raise TypeError("context runtime durable_event_sink must be callable")
            durable_event_sink(durable_event)
        except Exception as sink_error:
            boundary_error = mark_durable_persistence_failure(sink_error)
            if not isinstance(boundary_error, Exception):
                raise TypeError("durable sink failures must be Exception instances")
            first_failure = self._latch_failure(attempt_key, boundary_error)
            if not first_failure:
                self._raise_latched_failure(attempt_key)
            try:
                if callable(partial_attempt_sink):
                    partial_attempt_sink(
                        marker_event,
                        boundary_error,
                    )
            except Exception:
                # The original persistence error defines the failed boundary.
                # A secondary marker failure must not replace its type/identity.
                pass
            if boundary_error is sink_error:
                raise
            raise boundary_error from None

    @staticmethod
    def _attempt_key(event: Mapping[str, Any]) -> tuple[str, str] | None:
        attempt_id = str(event.get("attempt_id") or event.get("run_id") or "").strip()
        if not attempt_id:
            return None
        execution_id = str(
            event.get("execution_id")
            or event.get("session_id")
            or event.get("root_run_id")
            or attempt_id
        ).strip()
        return execution_id, attempt_id

    def _bundle_for_context(
        self,
        context: HarnessContext,
    ) -> ContextExecutionBundle:
        attempt_id = str(context.event.get("run_id") or "").strip()
        execution_id = str(context.state.session_state.session_id or attempt_id).strip()
        if not attempt_id or not execution_id:
            raise ContextExecutionBundleError(
                "context compile has no bootstrap attempt binding"
            )
        key = (execution_id, attempt_id)
        with self._bundle_lock:
            bundle = self._attempt_bundles.get(key)
        if bundle is None:
            raise ContextExecutionBundleError(
                "context compile occurred before bootstrap binding"
            )
        return bundle

    def _bundle_for_event(
        self,
        event: Mapping[str, Any],
    ) -> ContextExecutionBundle:
        attempt_id = str(event.get("attempt_id") or event.get("run_id") or "").strip()
        if not attempt_id:
            raise ContextExecutionBundleError(
                "durable event has no bootstrap attempt binding"
            )
        with self._bundle_lock:
            key = self._run_bindings.get(attempt_id)
            bundle = self._attempt_bundles.get(key) if key is not None else None
        if bundle is None:
            raise ContextExecutionBundleError(
                "durable event occurred before bootstrap binding"
            )
        return bundle

    def _latch_failure(
        self,
        attempt_key: tuple[str, str] | None,
        error: Exception,
    ) -> bool:
        if attempt_key is None:
            return True
        with self._failure_lock:
            if attempt_key in self._attempt_failures:
                return False
            self._attempt_failures[attempt_key] = error
            return True

    def _raise_latched_failure(
        self,
        attempt_key: tuple[str, str] | None,
    ) -> None:
        if attempt_key is None:
            return
        with self._failure_lock:
            failure = self._attempt_failures.get(attempt_key)
        if failure is not None:
            raise failure

    def compose_event_callback(
        self,
        host_callback: Callable[[dict[str, Any]], Any] | None,
    ) -> Callable[[dict[str, Any]], Any]:
        def validate_provisional_reasoning(event: dict[str, Any]) -> None:
            if (
                type(event) is not dict
                or set(event) != {"type", "run_id", "iteration", "provider", "delta"}
                or event["type"] != "reasoning"
                or type(event["run_id"]) is not str
                or not event["run_id"]
                or type(event["iteration"]) is not int
                or event["iteration"] < 0
                or event["provider"] != "ollama"
                or type(event["delta"]) is not str
                or not event["delta"]
            ):
                raise ValueError("provisional reasoning requires an exact Ollama event")

        def validate_preview_id(preview_id: str) -> None:
            if (
                type(preview_id) is not str
                or len(preview_id) != 32
                or any(character not in "0123456789abcdef" for character in preview_id)
            ):
                raise ValueError("provisional reasoning ID must be lowercase hex")

        def emit_provisional_reasoning(event: dict[str, Any], preview_id: str) -> Any:
            validate_provisional_reasoning(event)
            validate_preview_id(preview_id)
            if self.execution_factory is not None:
                bundle = self._bundle_for_event(event)
                attempt_key = (
                    bundle.attempt.generation.execution_id,
                    bundle.attempt.attempt_id,
                )
            else:
                attempt_key = self._attempt_key(event)
            self._raise_latched_failure(attempt_key)
            if host_callback is None:
                return None
            nested_preview = getattr(host_callback, "emit_provisional_reasoning", None)
            if callable(nested_preview):
                return nested_preview(event, preview_id)
            preview = copy.deepcopy(event)
            preview["provisional_reasoning_id"] = preview_id
            return host_callback(preview)

        def commit_provisional_reasoning(event: dict[str, Any]) -> None:
            validate_provisional_reasoning(event)
            active_events = _ACTIVE_DURABLE_EVENTS.get()
            event_key = (self.owner_id, id(event))
            if event_key in active_events:
                return
            token = _ACTIVE_DURABLE_EVENTS.set((*active_events, event_key))
            try:
                self.persist_event(event)
                nested_commit = getattr(
                    host_callback, "commit_provisional_reasoning", None
                )
                if callable(nested_commit):
                    try:
                        nested_commit(event)
                    except BaseException as error:
                        # Another owner may fail after this journal has accepted
                        # the event. Its live preview must remain replayable.
                        error._unchain_provisional_reasoning_journaled = True
                        raise
            finally:
                _ACTIVE_DURABLE_EVENTS.reset(token)

        def discard_provisional_reasoning(
            *, preview_id: str, run_id: str, iteration: int
        ) -> Any:
            validate_preview_id(preview_id)
            if type(run_id) is not str or not run_id:
                raise ValueError("provisional reasoning reset requires a run ID")
            if type(iteration) is not int or iteration < 0:
                raise ValueError("provisional reasoning reset requires an iteration")
            if host_callback is None:
                return None
            nested_discard = getattr(host_callback, "discard_provisional_reasoning", None)
            if callable(nested_discard):
                return nested_discard(
                    preview_id=preview_id, run_id=run_id, iteration=iteration
                )
            return host_callback(
                {
                    "type": "reasoning_preview_discarded",
                    "run_id": run_id,
                    "iteration": iteration,
                    "provider": "ollama",
                    "provisional_reasoning_id": preview_id,
                }
            )

        def persist_before_host(event: dict[str, Any]) -> Any:
            active_events = _ACTIVE_DURABLE_EVENTS.get()
            event_key = (self.owner_id, id(event))
            if event_key in active_events:
                if host_callback is None:
                    return None
                return host_callback(event)
            token = _ACTIVE_DURABLE_EVENTS.set((*active_events, event_key))
            try:
                self.persist_event(event)
                if host_callback is None:
                    return None
                return host_callback(event)
            finally:
                _ACTIVE_DURABLE_EVENTS.reset(token)

        persist_before_host.emit_provisional_reasoning = emit_provisional_reasoning
        persist_before_host.commit_provisional_reasoning = commit_provisional_reasoning
        persist_before_host.discard_provisional_reasoning = discard_provisional_reasoning
        return persist_before_host

    @staticmethod
    def _provider_turn_tool_identities(
        context,
    ) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (
                identity.name,
                identity.tool_object_id,
                identity.handler_object_id,
            )
            for identity in context.tools
        )

    def _provider_turn_host(self, context):
        from ..kernel.loop import KernelLoop
        from ..kernel.model_tool_boundary import FinalModelToolBoundaryContext

        if type(context) is not FinalModelToolBoundaryContext:
            raise TypeError(
                "provider turn boundary requires exact final boundary context"
            )
        bundle = self._bundle_for_event({"attempt_id": context.attempt_id})
        attempt = bundle.attempt
        if (
            context.attempt_id != attempt.attempt_id
            or context.run_id != attempt.attempt_id
            or context.execution_id != attempt.generation.execution_id
            or (
                context.generation_id
                and context.generation_id != attempt.generation.generation_id
            )
        ):
            raise ContextExecutionBundleError(
                "provider turn boundary changed its attempt binding"
            )
        service = bundle.provider_turn_service
        if service is None:
            raise ContextExecutionBundleError(
                "provider turn service is unavailable for the bound attempt"
            )
        key = (
            attempt.generation.execution_id,
            attempt.attempt_id,
        )
        with self._bundle_lock:
            interaction = self._attempt_interaction_authorities.get(key)
        loop = interaction.loop if interaction is not None else None
        if type(loop) is not KernelLoop:
            raise ContextExecutionBundleError(
                "provider turn boundary has no exact bound KernelLoop"
            )
        model_io = loop.model_io
        if (
            model_io is None
            or context.model_io_object_id != id(model_io)
            or context.provider != str(getattr(model_io, "provider", "") or "")
            or context.model != str(getattr(model_io, "model", "") or "")
        ):
            raise ContextExecutionBundleError(
                "provider turn boundary changed its model binding"
            )
        return bundle, service, model_io

    def _prepare_provider_turn_boundary(self, context):
        from ..kernel.model_tool_boundary import FinalModelToolPreparation
        from ..tools import Toolkit

        bundle, _service, _model_io = self._provider_turn_host(context)
        tools = {}
        for identity in context.tools:
            if (
                identity.name != identity.tool.name
                or identity.tool_object_id != id(identity.tool)
                or identity.handler_object_id
                != id(getattr(identity.tool, "func", None))
            ):
                raise ContextExecutionBundleError(
                    "provider turn boundary tool identity changed"
                )
            tools[identity.name] = identity.tool
        model_toolkit = Toolkit(
            tools,
            prompt_sections=context.toolkit_prompt_sections,
        )
        execution_toolkit = Toolkit(
            tools,
            prompt_sections=context.toolkit_prompt_sections,
        )
        binding = _ProviderTurnBoundaryBinding(
            execution_id=bundle.attempt.generation.execution_id,
            attempt_id=bundle.attempt.attempt_id,
            iteration=context.iteration,
            model_io_object_id=context.model_io_object_id,
            source_toolkit_object_id=context.toolkit_object_id,
            model_toolkit_object_id=id(model_toolkit),
            execution_toolkit_object_id=id(execution_toolkit),
            tool_identities=self._provider_turn_tool_identities(context),
        )
        return FinalModelToolPreparation(
            model_toolkit=model_toolkit,
            execution_toolkit=execution_toolkit,
            execution_binding=binding,
        )

    def _provider_turn_binding(
        self,
        context,
        preparation,
        *,
        validation: bool,
    ):
        from ..kernel.model_tool_boundary import FinalModelToolPreparation

        if type(preparation) is not FinalModelToolPreparation:
            raise TypeError("provider turn boundary requires exact final preparation")
        binding = preparation.execution_binding
        if type(binding) is not _ProviderTurnBoundaryBinding:
            raise ContextExecutionBundleError(
                "provider turn boundary lost its execution binding"
            )
        bundle, service, model_io = self._provider_turn_host(context)
        expected_toolkit_id = (
            binding.execution_toolkit_object_id
            if validation
            else binding.source_toolkit_object_id
        )
        if (
            binding.execution_id != bundle.attempt.generation.execution_id
            or binding.attempt_id != bundle.attempt.attempt_id
            or binding.iteration != context.iteration
            or binding.model_io_object_id != context.model_io_object_id
            or binding.model_toolkit_object_id != id(preparation.model_toolkit)
            or binding.execution_toolkit_object_id != id(preparation.execution_toolkit)
            or context.toolkit_object_id != expected_toolkit_id
            or binding.tool_identities != self._provider_turn_tool_identities(context)
        ):
            raise ContextExecutionBundleError(
                "provider turn boundary preparation changed"
            )
        return service, model_io

    def _verify_provider_tool_execution_binding(
        self,
        context: HarnessContext,
        *,
        bundle: ContextExecutionBundle,
        iteration: int,
    ) -> None:
        from ..kernel.loop import KernelLoop
        from ..tools import Toolkit

        binding = context.event.get("tool_execution_binding")
        toolkit = context.event.get("toolkit")
        if type(binding) is not _ProviderTurnBoundaryBinding:
            raise ContextExecutionBundleError(
                "provider tool execution lost its final boundary binding"
            )
        if not isinstance(toolkit, Toolkit):
            raise ContextExecutionBundleError(
                "provider tool execution lost its execution toolkit"
            )
        attempt = bundle.attempt
        attempt_key = (
            attempt.generation.execution_id,
            attempt.attempt_id,
        )
        with self._bundle_lock:
            interaction = self._attempt_interaction_authorities.get(attempt_key)
        loop = interaction.loop if interaction is not None else None
        model_io = loop.model_io if type(loop) is KernelLoop else None
        tool_identities = tuple(
            (
                name,
                id(tool),
                id(getattr(tool, "func", None)),
            )
            for name, tool in toolkit.tools.items()
        )
        if (
            binding.execution_id != attempt.generation.execution_id
            or binding.attempt_id != attempt.attempt_id
            or binding.iteration != iteration
            or model_io is None
            or binding.model_io_object_id != id(model_io)
            or binding.execution_toolkit_object_id != id(toolkit)
            or binding.tool_identities != tool_identities
        ):
            raise ContextExecutionBundleError(
                "provider tool execution final boundary binding changed"
            )

    def _fetch_provider_turn_boundary(
        self,
        context,
        preparation,
        request,
        retry_config,
        before_attempt,
    ):
        from ..providers.base import ModelTurnRequest

        service, model_io = self._provider_turn_binding(
            context,
            preparation,
            validation=False,
        )
        if type(request) is not ModelTurnRequest:
            raise TypeError("provider turn boundary requires exact model request")
        if (
            request.run_id != context.attempt_id
            or request.iteration != context.iteration
            or request.toolkit is not preparation.model_toolkit
        ):
            raise ContextExecutionBundleError(
                "provider turn boundary request changed after preparation"
            )
        return service.fetch_prepared(
            model_io=model_io,
            request=request,
            retry_config=retry_config,
            before_attempt=before_attempt,
            after_attempt=getattr(before_attempt, "after_attempt", None),
            run_receipt_factory=getattr(
                before_attempt,
                "run_receipt_factory",
                None,
            ),
            run_receipt_observed=getattr(
                before_attempt,
                "run_receipt_observed",
                None,
            ),
        )

    def _validate_provider_turn_boundary(
        self,
        context,
        preparation,
        turn,
    ):
        from ..kernel.types import ModelTurnResult

        self._provider_turn_binding(
            context,
            preparation,
            validation=True,
        )
        if type(turn) is not ModelTurnResult:
            raise TypeError("provider turn boundary requires exact model turn result")
        return turn

    def _prepare_provider_tool_resume_boundary(
        self,
        context,
        continuation: Mapping[str, Any],
        interaction_request: Mapping[str, Any],
    ):
        from ..interaction import (
            INTERACTION_KIND_TOOL_APPROVAL,
            InteractionRequest,
        )

        bundle, _service, _model_io = self._provider_turn_host(context)
        try:
            request = InteractionRequest.from_dict(dict(interaction_request))
        except (TypeError, ValueError) as exc:
            raise ContextExecutionBundleError(
                "provider tool resume has an invalid durable interaction"
            ) from exc
        tool_calls = continuation.get("tool_calls")
        paused_call_id = str(continuation.get("paused_call_id") or "")
        if (
            request.kind != INTERACTION_KIND_TOOL_APPROVAL
            or request.session_id != bundle.attempt.generation.execution_id
            or request.source_run_id != bundle.attempt.attempt_id
            or continuation.get("type") != "tool_approval_continuation"
            or continuation.get("interaction_id") != request.interaction_id
            or continuation.get("run_id") != bundle.attempt.attempt_id
            or continuation.get("tool_iteration") != context.iteration
            or not paused_call_id
            or not isinstance(tool_calls, list)
            or not any(
                isinstance(item, Mapping)
                and item.get("call_id") == paused_call_id
                and item.get("name") == request.payload.get("tool_name")
                for item in tool_calls
            )
        ):
            raise ContextExecutionBundleError(
                "provider tool resume changed its durable attempt binding"
            )
        return self._prepare_provider_turn_boundary(context)

    def _build_provider_turn_boundary(self):
        from ..kernel.model_tool_boundary import _issue_final_model_tool_boundary

        with self._provider_boundary_lock:
            boundary = self._provider_boundary
            if boundary is None:
                boundary = _issue_final_model_tool_boundary(
                    prepare=self._prepare_provider_turn_boundary,
                    fetch_prepared=self._fetch_provider_turn_boundary,
                    validate=self._validate_provider_turn_boundary,
                    prepare_tool_resume=(self._prepare_provider_tool_resume_boundary),
                )
                object.__setattr__(self, "_provider_boundary", boundary)
            return boundary

    def build_harness(self):
        from .harness import ContextCompilerHarness

        return ContextCompilerHarness(runtime=self)

    def build_harnesses(self):
        compiler = self.build_harness()
        from .tool_harness import ContextToolAuthorityHarness

        tool_authority = ContextToolAuthorityHarness(runtime=self)
        if self.execution_factory is None:
            return (tool_authority, compiler)
        from .harness import ContextExecutionBindingHarness

        components = (
            ContextExecutionBindingHarness(runtime=self),
            tool_authority,
            compiler,
        )
        if not self.provider_turns_enabled:
            return components
        return (*components, self._build_provider_turn_boundary())


__all__ = [
    "ContextBuildEnvelopeRequiredError",
    "ContextBuildUnavailableError",
    "ContextCheckpointBindingForbiddenError",
    "ContextCheckpointPersistenceRequiredError",
    "ContextRequestFactory",
    "ContextRuntime",
    "find_durable_persistence_failure",
    "mark_durable_persistence_failure",
]
