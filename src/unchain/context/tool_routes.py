from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from ..capabilities import CapabilityOutcome, RunContext
from ..input.human_input import is_human_input_tool_name
from ..kernel.harness import HarnessContext
from ..kernel.types import ToolCall
from ..tools.base import ToolContext
from ..tools.runtime import ToolRuntimeOutcome
from ..tools.toolkit import Toolkit
from .tool_boundary import DurableToolRouteKind, _canonical_digest
from .tool_executor import (
    DurableToolCompletionDraft,
    DurableToolExecutorContractError,
)
from .tool_transitions import build_declared_tool_state_transition


_ROUTE_SCHEMA = "unchain.context_tool_route.v1"
_MAX_BOUND_ROUTES = 4_096


def _canonical_copy(value: Any, *, field_name: str) -> Any:
    from ..journal.models import _freeze_json, _thaw_json

    try:
        return _thaw_json(_freeze_json(value, path=field_name))
    except (TypeError, ValueError) as exc:
        raise DurableToolExecutorContractError(
            f"{field_name} must be canonical JSON"
        ) from exc


def _callable_manifest(value: Any) -> dict[str, Any]:
    owner = getattr(value, "__self__", None)
    return {
        "module": str(getattr(value, "__module__", "") or ""),
        "qualname": str(
            getattr(value, "__qualname__", None)
            or getattr(value, "__name__", "")
            or type(value).__qualname__
        ),
        "owner": (
            ""
            if owner is None
            else f"{type(owner).__module__}.{type(owner).__qualname__}"
        ),
    }


def _same_callable_identity(left: Any, right: Any) -> bool:
    left_owner = getattr(left, "__self__", None)
    right_owner = getattr(right, "__self__", None)
    left_function = getattr(left, "__func__", left)
    right_function = getattr(right, "__func__", right)
    return left_owner is right_owner and left_function is right_function


@dataclass(frozen=True)
class ContextToolRoute:
    route_kind: DurableToolRouteKind
    _route_manifest: dict[str, Any] = field(repr=False, compare=False)
    _terminal_handler_manifest: dict[str, Any] = field(
        repr=False,
        compare=False,
    )
    _terminal_handler: Callable[[Any], DurableToolCompletionDraft] = field(
        repr=False,
        compare=False,
    )
    _resolver_authority: object = field(repr=False, compare=False)

    @property
    def route_manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._route_manifest)

    @property
    def route_manifest_sha256(self) -> str:
        return _canonical_digest(self._route_manifest)

    @property
    def terminal_handler_manifest(self) -> dict[str, Any]:
        return copy.deepcopy(self._terminal_handler_manifest)

    @property
    def terminal_handler_manifest_sha256(self) -> str:
        return _canonical_digest(self._terminal_handler_manifest)

    @property
    def terminal_handler(self) -> Callable[[Any], DurableToolCompletionDraft]:
        return self._terminal_handler


@dataclass(frozen=True)
class _RouteBinding:
    route: ContextToolRoute = field(repr=False, compare=False)
    route_kind: DurableToolRouteKind
    route_manifest_sha256: str
    terminal_handler_manifest_sha256: str
    terminal_handler: Callable[[Any], DurableToolCompletionDraft] = field(
        repr=False,
        compare=False,
    )


class ContextToolRouteResolver:
    """Freeze an exposed tool route before approval and side effects."""

    def __init__(self) -> None:
        self._authority = object()
        self._bindings: dict[int, _RouteBinding] = {}

    def resolve(
        self,
        context: HarnessContext,
        tool_call: ToolCall,
    ) -> ContextToolRoute:
        if not isinstance(context, HarnessContext):
            raise TypeError("context tool routing requires a HarnessContext")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("context tool routing requires a ToolCall")
        if is_human_input_tool_name(tool_call.name):
            raise DurableToolExecutorContractError(
                "human-input control routes are not durable tool routes"
            )
        toolkit = context.event.get("toolkit")
        if not isinstance(toolkit, Toolkit):
            raise DurableToolExecutorContractError(
                "durable tool routing requires the exposed toolkit"
            )
        detached_context = HarnessContext(
            state=copy.deepcopy(context.state),
            phase=context.phase,
            event=context.event_payload(),
        )
        tool_context = ToolContext(
            harness_context=detached_context,
            tool_component_name="context_v2_tool_authority",
        )
        plugin_route = self._resolve_plugin_route(
            tool_context=tool_context,
            tool_call=tool_call,
        )
        if plugin_route is not None:
            return self._bind(*plugin_route)
        normal_route = self._resolve_normal_route(
            context=detached_context,
            toolkit=toolkit,
            tool_call=tool_call,
        )
        return self._bind(*normal_route)

    def verify_route(self, route: ContextToolRoute) -> ContextToolRoute:
        if (
            type(route) is not ContextToolRoute
            or route._resolver_authority is not self._authority
        ):
            raise DurableToolExecutorContractError(
                "context tool route is not owned by this resolver authority"
            )
        binding = self._bindings.get(id(route))
        if (
            binding is None
            or binding.route is not route
            or route.route_kind is not binding.route_kind
            or route.route_manifest_sha256 != binding.route_manifest_sha256
            or route.terminal_handler_manifest_sha256
            != binding.terminal_handler_manifest_sha256
            or route.terminal_handler is not binding.terminal_handler
        ):
            raise DurableToolExecutorContractError(
                "context tool route binding changed after resolution"
            )
        return route

    def _bind(
        self,
        route_kind: DurableToolRouteKind,
        route_manifest: dict[str, Any],
        terminal_handler_manifest: dict[str, Any],
        terminal_handler: Callable[[Any], DurableToolCompletionDraft],
    ) -> ContextToolRoute:
        if len(self._bindings) >= _MAX_BOUND_ROUTES:
            raise DurableToolExecutorContractError(
                "context tool route binding capacity exceeded"
            )
        canonical_route = _canonical_copy(
            route_manifest,
            field_name="context_tool_route",
        )
        canonical_terminal = _canonical_copy(
            terminal_handler_manifest,
            field_name="context_tool_terminal_handler",
        )
        route = ContextToolRoute(
            route_kind=route_kind,
            _route_manifest=canonical_route,
            _terminal_handler_manifest=canonical_terminal,
            _terminal_handler=terminal_handler,
            _resolver_authority=self._authority,
        )
        self._bindings[id(route)] = _RouteBinding(
            route=route,
            route_kind=route_kind,
            route_manifest_sha256=route.route_manifest_sha256,
            terminal_handler_manifest_sha256=(route.terminal_handler_manifest_sha256),
            terminal_handler=terminal_handler,
        )
        return route

    def _resolve_normal_route(
        self,
        *,
        context: HarnessContext,
        toolkit: Toolkit,
        tool_call: ToolCall,
    ) -> tuple[
        DurableToolRouteKind,
        dict[str, Any],
        dict[str, Any],
        Callable[[Any], DurableToolCompletionDraft],
    ]:
        tool = toolkit.get(tool_call.name)
        if tool is None:
            raise DurableToolExecutorContractError(
                "durable tool route is not present in the exposed toolkit"
            )
        try:
            frozen_tool = copy.copy(tool)
            for attribute in (
                "parameters",
                "render_component",
                "prompt_spec",
                "icon",
                "provider_native_specs",
                "required_betas",
            ):
                setattr(
                    frozen_tool,
                    attribute,
                    copy.deepcopy(getattr(tool, attribute, None)),
                )
            frozen_tool.func = tool.func
            frozen_tool.confirmation_resolver = tool.confirmation_resolver
        except Exception as exc:
            raise DurableToolExecutorContractError(
                "exposed tool cannot be frozen into a durable route"
            ) from exc
        handler_identity = _callable_manifest(frozen_tool.func)
        tool_schema = _canonical_copy(
            frozen_tool.to_json(),
            field_name="context_tool_schema",
        )
        route_manifest = {
            "schema": _ROUTE_SCHEMA,
            "route_kind": DurableToolRouteKind.NORMAL.value,
            "tool": {
                "name": tool_call.name,
                "schema": tool_schema,
                "handler": handler_identity,
            },
        }
        terminal_manifest = {
            "schema": _ROUTE_SCHEMA,
            "terminal": {
                "kind": "toolkit",
                "tool_name": tool_call.name,
                "handler": handler_identity,
            },
        }
        captured_call_id = tool_call.call_id
        captured_name = tool_call.name

        def invoke(effective_arguments: Any) -> DurableToolCompletionDraft:
            outcome = frozen_tool.invoke(
                {
                    "tool_name": captured_name,
                    "call_id": captured_call_id,
                    "arguments": effective_arguments,
                },
                RunContext(
                    phase="on_tool_call",
                    event={
                        "tool_call": ToolCall(
                            call_id=captured_call_id,
                            name=captured_name,
                            arguments=copy.deepcopy(effective_arguments),
                        ),
                    },
                    state=context.state,
                    latest_version_id=context.latest_version_id,
                ),
            )
            if type(outcome) is not CapabilityOutcome:
                raise DurableToolExecutorContractError(
                    "tool route returned an invalid capability outcome"
                )
            if outcome.delta is not None or outcome.metadata:
                raise DurableToolExecutorContractError(
                    "tool capability delta is an unsupported side channel"
                )
            result = copy.deepcopy(outcome.value)
            if not isinstance(result, dict):
                result = {"result": result}
            return DurableToolCompletionDraft(
                result=result,
                should_observe=bool(frozen_tool.observe),
            )

        return (
            DurableToolRouteKind.NORMAL,
            route_manifest,
            terminal_manifest,
            invoke,
        )

    def _resolve_plugin_route(
        self,
        *,
        tool_context: ToolContext,
        tool_call: ToolCall,
    ) -> (
        tuple[
            DurableToolRouteKind,
            dict[str, Any],
            dict[str, Any],
            Callable[[Any], DurableToolCompletionDraft],
        ]
        | None
    ):
        captured_base_state = tool_context.state.subagent_state.to_dict()
        captured_terminal_handoff_base_state = {
            "subagent_state": tool_context.state.subagent_state.to_dict(),
            "transcript": copy.deepcopy(tool_context.state.transcript),
            "run_status": tool_context.state.run_status,
            "pending_tool_calls": copy.deepcopy(tool_context.state.pending_tool_calls),
            "tool_batch_state": tool_context.state.tool_batch_state.copy(),
            "last_continuation": copy.deepcopy(tool_context.state.last_continuation),
            "next_model_input": copy.deepcopy(tool_context.state.next_model_input),
        }
        matches: list[tuple[Any, Callable[..., Any], dict[str, Any]]] = []
        for plugin in tool_context.tool_runtime_plugins:
            can_handle = getattr(plugin, "can_handle", None)
            if not callable(can_handle) or not can_handle(
                tool_call=tool_call,
                context=tool_context,
            ):
                continue
            manifest_factory = getattr(
                plugin,
                "durable_runtime_manifest",
                None,
            )
            if not callable(manifest_factory):
                raise DurableToolExecutorContractError(
                    "matching tool runtime plugin has no durable manifest"
                )
            manifest = manifest_factory(
                tool_call=tool_call,
                context=tool_context,
            )
            if not isinstance(manifest, dict):
                raise DurableToolExecutorContractError(
                    "tool runtime plugin returned an invalid durable manifest"
                )
            execute = getattr(plugin, "execute", None)
            if not callable(execute):
                raise DurableToolExecutorContractError(
                    "matching tool runtime plugin has no exact handler"
                )
            matches.append(
                (
                    plugin,
                    execute,
                    _canonical_copy(
                        manifest,
                        field_name="context_tool_plugin_manifest",
                    ),
                )
            )
        if not matches:
            return None
        terminal_indexes = [
            index
            for index, (_plugin, _execute, manifest) in enumerate(matches)
            if manifest.get("terminal_handler") is True
        ]
        if not terminal_indexes:
            raise DurableToolExecutorContractError(
                "matching plugin route has no durable terminal manifest"
            )
        terminal_index = terminal_indexes[0]
        route_plugins = [
            {
                "plugin": (f"{type(plugin).__module__}.{type(plugin).__qualname__}"),
                "handler": _callable_manifest(execute),
                "manifest": manifest,
            }
            for plugin, execute, manifest in matches
        ]
        route_manifest = {
            "schema": _ROUTE_SCHEMA,
            "route_kind": DurableToolRouteKind.PLUGIN.value,
            "tool_name": tool_call.name,
            "plugins": route_plugins,
        }
        terminal_manifest = {
            "schema": _ROUTE_SCHEMA,
            "terminal": {
                "kind": "plugin",
                "match_index": terminal_index,
            },
            "handler": route_plugins[terminal_index],
        }
        captured_call_id = tool_call.call_id
        captured_name = tool_call.name
        captured_context = tool_context
        captured_plugins = tuple(plugin for plugin, _execute, _manifest in matches)
        captured_can_handlers = tuple(
            plugin.can_handle for plugin, _execute, _manifest in matches
        )
        captured_handlers = tuple(execute for _plugin, execute, _manifest in matches)
        captured_manifest_factories = tuple(
            plugin.durable_runtime_manifest for plugin, _execute, _manifest in matches
        )
        captured_manifests = tuple(manifest for _plugin, _execute, manifest in matches)

        def invoke(effective_arguments: Any) -> DurableToolCompletionDraft:
            bound_call = ToolCall(
                call_id=captured_call_id,
                name=captured_name,
                arguments=copy.deepcopy(effective_arguments),
            )
            for index, execute in enumerate(captured_handlers[: terminal_index + 1]):
                plugin = captured_plugins[index]
                live_can_handle = getattr(plugin, "can_handle", None)
                live_execute = getattr(plugin, "execute", None)
                live_manifest_factory = getattr(
                    plugin,
                    "durable_runtime_manifest",
                    None,
                )
                if (
                    not callable(live_can_handle)
                    or not callable(live_execute)
                    or not callable(live_manifest_factory)
                    or not _same_callable_identity(
                        live_can_handle,
                        captured_can_handlers[index],
                    )
                    or not _same_callable_identity(live_execute, execute)
                    or not _same_callable_identity(
                        live_manifest_factory,
                        captured_manifest_factories[index],
                    )
                ):
                    raise DurableToolExecutorContractError(
                        "tool runtime plugin handler drifted after routing"
                    )
                if not live_can_handle(
                    tool_call=bound_call,
                    context=captured_context,
                ):
                    raise DurableToolExecutorContractError(
                        "tool runtime plugin capability drifted after routing"
                    )
                revalidated_manifest = _canonical_copy(
                    live_manifest_factory(
                        tool_call=bound_call,
                        context=captured_context,
                    ),
                    field_name="context_tool_plugin_manifest",
                )
                if revalidated_manifest != captured_manifests[index]:
                    raise DurableToolExecutorContractError(
                        "tool runtime plugin manifest drifted after routing"
                    )
                try:
                    outcome = execute(
                        tool_call=bound_call,
                        context=captured_context,
                    )
                except Exception as exc:
                    from ..subagents.types import SubagentWaitSignal
                    if not isinstance(exc, SubagentWaitSignal):
                        raise
                    from ..tools.runtime import subagent_wait_outcome
                    outcome = subagent_wait_outcome(exc, captured_context)
                if type(outcome) is not ToolRuntimeOutcome:
                    raise DurableToolExecutorContractError(
                        "tool runtime plugin returned an invalid outcome"
                    )
                if outcome.result_messages or outcome.suspend_override is not None:
                    raise DurableToolExecutorContractError(
                        "tool runtime plugin used an unsupported side channel"
                    )
                if outcome.handled:
                    if index != terminal_index:
                        raise DurableToolExecutorContractError(
                            "nonterminal plugin handled a fixed terminal route"
                        )
                    result = copy.deepcopy(outcome.tool_result)
                    if not isinstance(result, dict):
                        result = {"result": result}
                    state_transition = None
                    if outcome.state_updates:
                        completion_contract = captured_manifests[index].get(
                            "completion_contract"
                        )
                        if completion_contract is None:
                            raise DurableToolExecutorContractError(
                                "tool runtime plugin used an unsupported side channel"
                            )
                        transition_base_state = captured_base_state
                        transition_variants = (
                            completion_contract.get("state_transition_variants")
                            if isinstance(completion_contract, dict)
                            else None
                        )
                        declares_terminal_handoff_variant = type(
                            transition_variants
                        ) is list and any(
                            type(variant) is dict
                            and variant.get("state_transition")
                            == "subagent_terminal_handoff.v1"
                            for variant in transition_variants
                        )
                        if isinstance(completion_contract, dict) and (
                            completion_contract.get("state_transition")
                            == "subagent_terminal_handoff.v1"
                            or declares_terminal_handoff_variant
                        ):
                            transition_base_state = captured_terminal_handoff_base_state
                        state_transition = build_declared_tool_state_transition(
                            completion_contract=completion_contract,
                            state_updates=outcome.state_updates,
                            base_state=transition_base_state,
                        )
                    return DurableToolCompletionDraft(
                        result=result,
                        should_observe=bool(outcome.should_observe),
                        state_transition=state_transition,
                    )
                if index == terminal_index:
                    raise DurableToolExecutorContractError(
                        "durable terminal plugin declined its fixed route"
                    )
            raise DurableToolExecutorContractError(
                "durable plugin route has no terminal completion"
            )

        return (
            DurableToolRouteKind.PLUGIN,
            route_manifest,
            terminal_manifest,
            invoke,
        )


__all__ = ["ContextToolRoute", "ContextToolRouteResolver"]
