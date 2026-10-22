from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from ..artifacts import extract_authored_artifacts, upsert_artifacts
from ..execution import ExecutionLeaseError
from ..kernel.model_io import ModelIO, ModelTurnRequest
from ..kernel.types import ModelTurnResult, ToolCall
from ..schemas import ResponseFormat
from .confirmation import execute_confirmable_tool_call, prepare_tool_confirmation
from ..interaction.durable import InteractionIntegrityError, strict_json_digest
from ..interaction.requests import (
    build_tool_approval_interaction_request,
    ensure_interaction_binding_matches,
)
from .execution import (
    _canonical_artifacts_for_tool_result,
    _durable_tool_runtime_subject,
    _emit_artifact_events,
    _workspace_change_state_update,
    _workspace_change_tracker,
)
from .models import ToolExecutionContext, ToolPromptSpec
from .runtime import ToolRuntimeOutcome, run_tool_runtime_plugins
from .tool import Tool
from .toolkit import Toolkit


TOOL_SEARCH_NAME = "tool_search"
TOOL_DESCRIBE_NAME = "tool_describe"
TOOL_LOAD_NAME = "tool_load"
TOOL_EXECUTE_DEFERRED_NAME = "tool_execute_deferred"
META_TOOL_NAMES = (
    TOOL_SEARCH_NAME,
    TOOL_DESCRIBE_NAME,
    TOOL_LOAD_NAME,
    TOOL_EXECUTE_DEFERRED_NAME,
)
TOOL_EXPOSURE_PLAN_SCHEMA_VERSION = 1


def _unique_names(names: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _compact_text(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z0-9_:-]+", text or "") if len(term) > 1]


@dataclass(frozen=True)
class ToolOptimizerConfig:
    max_direct_tools: int = 64
    trigger_tool_count: int = 65
    selector_timeout_seconds: int = 30
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_direct_tools", max(1, int(self.max_direct_tools)))
        object.__setattr__(self, "trigger_tool_count", max(1, int(self.trigger_tool_count)))
        object.__setattr__(
            self,
            "selector_timeout_seconds",
            max(1, int(self.selector_timeout_seconds)),
        )
        object.__setattr__(self, "enabled", bool(self.enabled))

    @classmethod
    def coerce(cls, value: "ToolOptimizerConfig" | dict[str, Any] | None) -> "ToolOptimizerConfig" | None:
        if value is None:
            return None
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(**value)
        raise TypeError("tool optimizer config must be ToolOptimizerConfig, dict, or None")


@dataclass
class ToolExposurePlan:
    direct_tool_names: list[str] = field(default_factory=list)
    deferred_tool_names: list[str] = field(default_factory=list)
    loaded_tool_names: list[str] = field(default_factory=list)
    selector_status: str = "not_started"
    fallback_reason: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "direct_tool_names": list(self.direct_tool_names),
            "deferred_tool_names": list(self.deferred_tool_names),
            "loaded_tool_names": list(self.loaded_tool_names),
            "selector_status": self.selector_status,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class ExposureToolRecord:
    handle: str
    name: str
    description: str
    search_hint: str = ""
    toolkit: str = "runtime"
    server: str = ""
    category: str = "local"

    def search_blob(self) -> str:
        return " ".join(
            part
            for part in (
                self.handle,
                self.name,
                self.description,
                self.search_hint,
                self.toolkit,
                self.server,
                self.category,
            )
            if isinstance(part, str) and part.strip()
        ).lower()

    def to_summary(self) -> dict[str, Any]:
        payload = {
            "handle": self.handle,
            "name": self.name,
            "description": self.description,
            "toolkit": self.toolkit,
            "category": self.category,
        }
        if self.server:
            payload["server"] = self.server
        return payload


@dataclass(frozen=True)
class _NestedToolRuntimeContext:
    parent: Any
    toolkit: Toolkit
    tool_runtime_plugins: list[Any]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.parent, name)


class DeferredToolExecutionPlugin:
    def __init__(self, runtime: "ToolExposureRuntime") -> None:
        self.runtime = runtime

    def can_handle(self, *, tool_call: ToolCall, context: Any) -> bool:
        del context
        return tool_call.name == TOOL_EXECUTE_DEFERRED_NAME

    def durable_exposure_plan(self) -> dict[str, Any]:
        """Return the selector plan that must be replayed after a cold pause."""

        return self.runtime.durable_plan_snapshot()

    def _nested_runtime_context(self, context: Any) -> _NestedToolRuntimeContext:
        return _NestedToolRuntimeContext(
            parent=context,
            toolkit=self.runtime.full_toolkit,
            tool_runtime_plugins=[
                plugin
                for plugin in getattr(context, "tool_runtime_plugins", [])
                if plugin is not self
            ],
        )

    @staticmethod
    def _outer_binding(tool_call: ToolCall) -> dict[str, Any]:
        return {
            "outer_tool_name": tool_call.name,
            "outer_call_id": tool_call.call_id,
            "outer_arguments_digest": strict_json_digest(tool_call.arguments),
        }

    def prepare_durable_confirmation(
        self,
        *,
        tool_call: ToolCall,
        context: Any,
    ) -> dict[str, Any] | None:
        payload = self.runtime.parse_execute_deferred_arguments(tool_call.arguments)
        if "error" in payload:
            return None
        target_name = str(payload.get("tool_name") or "").strip()
        if target_name not in self.runtime.full_toolkit.tools or target_name in META_TOOL_NAMES:
            return None
        target_call = ToolCall(
            call_id=tool_call.call_id,
            name=target_name,
            arguments=payload.get("arguments"),
        )
        execution_context = ToolExecutionContext(
            session_id=context.session_id,
            run_id=context.run_id,
            provider=context.provider,
            model=context.model,
            iteration=context.iteration,
            memory_namespace=context.memory_namespace,
            tool_runtime_config=context.event.get("tool_runtime_config")
            if isinstance(context.event.get("tool_runtime_config"), dict)
            else {},
            tool_name=target_name,
            call_id=tool_call.call_id,
            turn_id=f"{context.run_id}:turn-{context.iteration}",
            run_state=context.state,
        )
        preparation = prepare_tool_confirmation(
            toolkit=self.runtime.full_toolkit,
            tool_call=target_call,
            execution_context=execution_context,
            execution_guard=context.execution_guard,
        )
        return {
            "tool_call": target_call,
            "toolkit": self.runtime.full_toolkit,
            "preparation": preparation,
            "extra_subject": self._outer_binding(tool_call),
            "runtime_context": self._nested_runtime_context(context),
        }

    def execute(self, *, tool_call: ToolCall, context: Any) -> ToolRuntimeOutcome:
        payload = self.runtime.parse_execute_deferred_arguments(tool_call.arguments)
        if "error" in payload:
            return ToolRuntimeOutcome(tool_result=payload)

        target_name = str(payload.get("tool_name") or "").strip()
        target_arguments = payload.get("arguments")
        if target_name not in self.runtime.full_toolkit.tools:
            return ToolRuntimeOutcome(
                tool_result={
                    "error": f"deferred tool not found: {target_name}",
                    "tool": target_name,
                }
            )
        if target_name in META_TOOL_NAMES:
            return ToolRuntimeOutcome(
                tool_result={
                    "error": "meta tools cannot be executed through tool_execute_deferred",
                    "tool": target_name,
                }
            )

        target_tool_call = ToolCall(
            call_id=tool_call.call_id,
            name=target_name,
            arguments=target_arguments,
        )
        on_tool_confirm = context.event.get("on_tool_confirm") if hasattr(context, "event") else None
        workspace_tracker = _workspace_change_tracker(context, self.runtime.full_toolkit, target_tool_call)
        workspace_snapshot_before = (
            workspace_tracker.capture_text_snapshot()
            if workspace_tracker is not None
            else None
        )
        execution_context = ToolExecutionContext(
            session_id=context.session_id,
            run_id=context.run_id,
            provider=context.provider,
            model=context.model,
            iteration=context.iteration,
            memory_namespace=context.memory_namespace,
            tool_runtime_config=context.event.get("tool_runtime_config")
            if isinstance(context.event.get("tool_runtime_config"), dict)
            else {},
            tool_name=target_name,
            call_id=tool_call.call_id,
            turn_id=f"{context.run_id}:turn-{context.iteration}",
            workspace_changes=workspace_tracker,
            run_state=context.state,
        )
        preparation = prepare_tool_confirmation(
            toolkit=self.runtime.full_toolkit,
            tool_call=target_tool_call,
            execution_context=execution_context,
            execution_guard=context.execution_guard,
        )
        stored_interaction_request = (
            context.raw_event.get("interaction_request")
            if hasattr(context, "raw_event")
            else None
        )
        interaction_response_supplied = bool(
            hasattr(context, "raw_event")
            and "interaction_response" in context.raw_event
        )
        execution_kwargs: dict[str, Any] = {
            "toolkit": self.runtime.full_toolkit,
            "tool_call": target_tool_call,
            "on_tool_confirm": on_tool_confirm,
            "loop": context.loop,
            "callback": context.callback,
            "run_id": context.run_id,
            "iteration": context.iteration,
            "execution_context": execution_context,
            "execution_guard": context.execution_guard,
            "prepared_confirmation": preparation,
        }
        if isinstance(stored_interaction_request, dict):
            if not preparation.needs_confirmation_response or preparation.request is None:
                raise InteractionIntegrityError(
                    "deferred tool confirmation policy changed while approval was pending"
                )
            current_request = build_tool_approval_interaction_request(
                state=context.state,
                toolkit=self.runtime.full_toolkit,
                tool_call=target_tool_call,
                confirmation_request=preparation.request,
                run_id=context.run_id,
                source_request=stored_interaction_request,
                extra_subject=_durable_tool_runtime_subject(
                    self._nested_runtime_context(context),
                    target_tool_call,
                    extra_subject=self._outer_binding(tool_call),
                ),
            )
            ensure_interaction_binding_matches(
                stored_interaction_request,
                current_request,
            )
            if not interaction_response_supplied:
                raise InteractionIntegrityError(
                    "deferred tool approval requires a durable receipt response"
                )
            execution_kwargs["confirmation_response"] = context.raw_event.get(
                "interaction_response"
            )
        elif (
            preparation.needs_confirmation_response
            and context.session_id
            and getattr(context.loop, "interaction_runtime", None) is not None
        ):
            raise InteractionIntegrityError(
                "deferred tool approval must suspend before plugin execution"
            )
        nested_context = self._nested_runtime_context(context)
        runtime_outcome = run_tool_runtime_plugins(
            nested_context.tool_runtime_plugins,
            tool_call=target_tool_call,
            context=nested_context,
            execution_guard=context.execution_guard,
        )
        if runtime_outcome is not None:
            return runtime_outcome
        outcome = execute_confirmable_tool_call(**execution_kwargs)
        state_updates: dict[str, Any] = {}
        if workspace_tracker is not None:
            workspace_tracker.record_text_snapshot_changes(
                workspace_snapshot_before,
                tool_name=target_name,
                call_id=tool_call.call_id,
                turn_id=f"{context.run_id}:turn-{context.iteration}",
            )
            state_updates.update(_workspace_change_state_update(workspace_tracker))

        visible_tool_result, authored_artifacts = extract_authored_artifacts(outcome.tool_result)
        emitted_artifacts = []
        if isinstance(visible_tool_result, dict):
            emitted_artifacts = _canonical_artifacts_for_tool_result(
                context,
                target_tool_call,
                visible_tool_result,
                authored_artifacts,
                confirmation_policy=outcome.confirmation_policy,
                effective_arguments=outcome.effective_arguments,
            )
            _emit_artifact_events(context, target_tool_call, emitted_artifacts)
        if emitted_artifacts:
            state_updates["artifacts"] = upsert_artifacts(
                context.state.artifacts,
                emitted_artifacts,
            )

        return ToolRuntimeOutcome(
            tool_result=visible_tool_result,
            state_updates=state_updates,
            should_observe=outcome.should_observe,
        )


class ToolExposureRuntime:
    def __init__(
        self,
        *,
        config: ToolOptimizerConfig,
        full_toolkit: Toolkit,
        model_io: ModelIO,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        instructions: str = "",
        payload: dict[str, Any] | None = None,
        callback: Any = None,
        run_id: str | None = None,
        on_model_attempt: Callable[
            [ModelTurnRequest, ModelTurnResult | None, str], None
        ]
        | None = None,
    ) -> None:
        self.config = config
        self.full_toolkit = full_toolkit
        self.model_io = model_io
        self.provider = provider
        self.model = model
        self.messages = copy.deepcopy(messages)
        self.instructions = str(instructions or "")
        self.payload = copy.deepcopy(payload or {})
        self.callback = callback
        self.run_id = str(run_id or "tool_optimizer")
        self.on_model_attempt = on_model_attempt
        self.plan = ToolExposurePlan()
        self.exposed_toolkit: Toolkit = full_toolkit
        self._records_by_name: dict[str, ExposureToolRecord] = {}
        self._loaded_names: list[str] = []
        self._selector_used = False

    @property
    def active(self) -> bool:
        return self._selector_used

    def build_plugins(self) -> list[DeferredToolExecutionPlugin]:
        if not self._selector_used:
            return []
        return [DeferredToolExecutionPlugin(self)]

    def prepare(
        self,
        *,
        replay_plan: dict[str, Any] | None = None,
    ) -> Toolkit:
        if replay_plan is not None:
            return self._prepare_replay_plan(replay_plan)

        if not self.config.enabled:
            self.plan = ToolExposurePlan(
                direct_tool_names=list(self.full_toolkit.tools),
                deferred_tool_names=[],
                loaded_tool_names=[],
                selector_status="disabled",
            )
            self.exposed_toolkit = self.full_toolkit
            return self.exposed_toolkit

        tool_count = len(self.full_toolkit.tools)
        if tool_count <= self.config.trigger_tool_count:
            self.plan = ToolExposurePlan(
                direct_tool_names=list(self.full_toolkit.tools),
                deferred_tool_names=[],
                loaded_tool_names=[],
                selector_status="skipped_small_tool_pool",
            )
            self.exposed_toolkit = self.full_toolkit
            return self.exposed_toolkit

        self._selector_used = True
        self._records_by_name = self._build_records()
        selected_names, selector_status, fallback_reason = self._select_tool_names()
        direct_original = self._resolve_direct_original_names(selected_names)
        deferred = [name for name in self.full_toolkit.tools if name not in set(direct_original)]
        self.plan = ToolExposurePlan(
            direct_tool_names=[*direct_original, *META_TOOL_NAMES],
            deferred_tool_names=deferred,
            loaded_tool_names=[],
            selector_status=selector_status,
            fallback_reason=fallback_reason,
        )
        self.exposed_toolkit = self._build_exposed_toolkit(direct_original)
        return self.exposed_toolkit

    def _catalog_manifest(self) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        for name, tool_obj in self.full_toolkit.tools.items():
            if name in META_TOOL_NAMES:
                continue
            prompt_spec = getattr(tool_obj, "prompt_spec", None)
            tools.append(
                {
                    "name": name,
                    "provider_schema": tool_obj.to_provider_json(self.provider),
                    "always_load": bool(getattr(tool_obj, "always_load", False)),
                    "defer_by_default": bool(
                        getattr(tool_obj, "defer_by_default", False)
                    ),
                    "search_hint": str(getattr(tool_obj, "search_hint", "") or ""),
                    "toolkit": str(getattr(tool_obj, "toolkit_id", "") or "runtime"),
                    "server": str(getattr(tool_obj, "server", "") or ""),
                    "category": str(getattr(tool_obj, "category", "") or "local"),
                    "prompt_spec": (
                        {
                            "purpose": str(getattr(prompt_spec, "purpose", "") or ""),
                            "when_to_use": list(
                                getattr(prompt_spec, "when_to_use", ()) or ()
                            ),
                            "when_not_to_use": list(
                                getattr(prompt_spec, "when_not_to_use", ()) or ()
                            ),
                            "examples": list(
                                getattr(prompt_spec, "examples", ()) or ()
                            ),
                            "advanced_tips": list(
                                getattr(prompt_spec, "advanced_tips", ()) or ()
                            ),
                        }
                        if prompt_spec is not None
                        else None
                    ),
                }
            )
        return {
            "provider": self.provider,
            "prompt_sections": list(self.full_toolkit.prompt_sections),
            "tools": tools,
        }

    def _catalog_digest(self) -> str:
        return strict_json_digest(self._catalog_manifest())

    @staticmethod
    def _plan_name_list(raw: Any, *, field_name: str) -> list[str]:
        if not isinstance(raw, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw
        ):
            raise InteractionIntegrityError(
                f"tool exposure replay {field_name} must be a list of non-empty strings"
            )
        names = [item.strip() for item in raw]
        if len(names) != len(set(names)):
            raise InteractionIntegrityError(
                f"tool exposure replay {field_name} contains duplicate names"
            )
        return names

    def durable_plan_snapshot(self) -> dict[str, Any]:
        if not self._selector_used:
            raise InteractionIntegrityError(
                "tool exposure plan is unavailable because the selector was not active"
            )
        direct_original = [
            name
            for name in self.plan.direct_tool_names
            if name not in META_TOOL_NAMES
        ]
        return {
            "schema_version": TOOL_EXPOSURE_PLAN_SCHEMA_VERSION,
            "provider": self.provider,
            "catalog_digest": self._catalog_digest(),
            "direct_tool_names": direct_original,
            "deferred_tool_names": list(self.plan.deferred_tool_names),
            "loaded_tool_names": list(self.plan.loaded_tool_names),
        }

    def _prepare_replay_plan(self, raw: dict[str, Any]) -> Toolkit:
        if not self.config.enabled:
            raise InteractionIntegrityError(
                "tool exposure replay requires an enabled tool optimizer"
            )
        expected_fields = {
            "schema_version",
            "provider",
            "catalog_digest",
            "direct_tool_names",
            "deferred_tool_names",
            "loaded_tool_names",
        }
        if not isinstance(raw, dict) or set(raw) != expected_fields:
            raise InteractionIntegrityError(
                "tool exposure replay plan does not match schema version 1"
            )
        if raw.get("schema_version") != TOOL_EXPOSURE_PLAN_SCHEMA_VERSION:
            raise InteractionIntegrityError(
                "unsupported tool exposure replay plan schema_version"
            )
        if raw.get("provider") != self.provider:
            raise InteractionIntegrityError(
                "tool exposure replay provider does not match the active runtime"
            )
        catalog_digest = raw.get("catalog_digest")
        if (
            not isinstance(catalog_digest, str)
            or not catalog_digest
            or catalog_digest != self._catalog_digest()
        ):
            raise InteractionIntegrityError(
                "tool exposure replay catalog digest does not match the active toolkit"
            )
        direct_names = self._plan_name_list(
            raw.get("direct_tool_names"),
            field_name="direct_tool_names",
        )
        deferred_names = self._plan_name_list(
            raw.get("deferred_tool_names"),
            field_name="deferred_tool_names",
        )
        loaded_names = self._plan_name_list(
            raw.get("loaded_tool_names"),
            field_name="loaded_tool_names",
        )
        full_names = [
            name
            for name in self.full_toolkit.tools
            if name not in META_TOOL_NAMES
        ]
        direct_set = set(direct_names)
        deferred_set = set(deferred_names)
        full_set = set(full_names)
        if direct_set & deferred_set or direct_set | deferred_set != full_set:
            raise InteractionIntegrityError(
                "tool exposure replay direct/deferred names must exactly partition the active toolkit"
            )
        if not set(loaded_names).issubset(direct_set):
            raise InteractionIntegrityError(
                "tool exposure replay loaded names must be a subset of direct names"
            )

        self._selector_used = True
        self._records_by_name = self._build_records()
        self._loaded_names = list(loaded_names)
        self.plan = ToolExposurePlan(
            direct_tool_names=[*direct_names, *META_TOOL_NAMES],
            deferred_tool_names=list(deferred_names),
            loaded_tool_names=list(loaded_names),
            selector_status="replayed",
            fallback_reason="",
        )
        self.exposed_toolkit = self._build_exposed_toolkit(direct_names)
        return self.exposed_toolkit

    def _build_records(self) -> dict[str, ExposureToolRecord]:
        records: dict[str, ExposureToolRecord] = {}
        for name, tool_obj in self.full_toolkit.tools.items():
            if name in META_TOOL_NAMES:
                continue
            records[name] = ExposureToolRecord(
                handle=name,
                name=name,
                description=_compact_text(tool_obj.description),
                search_hint=_compact_text(getattr(tool_obj, "search_hint", "")),
                toolkit=str(getattr(tool_obj, "toolkit_id", "") or "runtime"),
                server=str(getattr(tool_obj, "server", "") or ""),
                category=str(getattr(tool_obj, "category", "") or "local"),
            )
        return records

    def _always_load_names(self) -> list[str]:
        return [
            name
            for name, tool_obj in self.full_toolkit.tools.items()
            if name not in META_TOOL_NAMES and bool(getattr(tool_obj, "always_load", False))
        ]

    def _selector_response_format(self) -> ResponseFormat:
        return ResponseFormat(
            name="tool_exposure_selection",
            schema={
                "type": "object",
                "properties": {
                    "tool_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["tool_names"],
                "additionalProperties": True,
            },
            required=["tool_names"],
        )

    def _select_tool_names(self) -> tuple[list[str], str, str]:
        try:
            selector_names = self._run_selector()
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            fallback = f"selector_failed: {type(exc).__name__}: {exc}"
            return self._fallback_tool_names(), "fallback", fallback
        if not selector_names:
            return self._fallback_tool_names(), "fallback", "selector_returned_no_valid_tools"
        return selector_names, "selected", ""

    def _run_selector(self) -> list[str]:
        catalog = [
            {
                **record.to_summary(),
                "search_hint": record.search_hint,
            }
            for record in self._records_by_name.values()
        ]
        request_payload = {
            "store": False,
            "temperature": 0,
            "timeout": self.config.selector_timeout_seconds,
            **copy.deepcopy(self.payload),
        }
        request_payload["store"] = False
        request = ModelTurnRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select the smallest ordered set of tools likely needed for the next agent run. "
                        "Return JSON only. Do not call tools."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "agent_instructions": self.instructions,
                            "conversation": self._compact_messages(self.messages),
                            "max_provider_visible_tools": self.config.max_direct_tools,
                            "reserved_meta_tools": list(META_TOOL_NAMES),
                            "always_load": self._always_load_names(),
                            "catalog": catalog,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            payload=request_payload,
            response_format=self._selector_response_format(),
            callback=None,
            verbose=False,
            run_id=f"{self.run_id}:tool-optimizer",
            iteration=0,
            toolkit=Toolkit(),
            emit_stream=False,
            previous_response_id=None,
        )
        occurred_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        try:
            turn = self.model_io.fetch_turn(request)
        except BaseException:
            if self.on_model_attempt is not None:
                self.on_model_attempt(request, None, occurred_at)
            raise
        if self.on_model_attempt is not None:
            self.on_model_attempt(request, turn, occurred_at)
        parsed = self._parse_selector_turn(turn)
        return self._sanitize_selector_names(parsed)

    def _compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compact: list[dict[str, Any]] = []
        for message in messages[-8:]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = message.get("content")
            if isinstance(content, str):
                compact.append({"role": role, "content": _compact_text(content, limit=1200)})
            elif isinstance(content, list):
                compact.append({"role": role, "content": _compact_text(json.dumps(content, default=str), limit=1200)})
            else:
                compact.append({"role": role, "content": _compact_text(content, limit=1200)})
        return compact

    def _parse_selector_turn(self, turn: ModelTurnResult) -> dict[str, Any]:
        text = (turn.final_text or "").strip()
        if not text:
            text = self._last_assistant_text(turn.assistant_messages)
        return self._selector_response_format().parse(text)

    def _last_assistant_text(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages or []):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts).strip()
        return ""

    def _sanitize_selector_names(self, parsed: dict[str, Any]) -> list[str]:
        raw_names = parsed.get("tool_names")
        if not isinstance(raw_names, list):
            raw_names = parsed.get("tools")
        if not isinstance(raw_names, list):
            raw_names = parsed.get("direct_tool_names")
        if not isinstance(raw_names, list):
            return []

        names: list[str] = []
        for item in raw_names:
            if isinstance(item, dict):
                raw_name = item.get("name") or item.get("tool_name") or item.get("handle")
            else:
                raw_name = item
            name = str(raw_name or "").strip()
            if name in self.full_toolkit.tools and name not in META_TOOL_NAMES:
                names.append(name)
        return _unique_names(names)

    def _fallback_tool_names(self) -> list[str]:
        context_text = "\n".join(
            [
                self.instructions,
                *[
                    str(message.get("content") or "")
                    for message in self.messages
                    if isinstance(message, dict)
                ],
            ]
        ).lower()
        context_terms = _terms(context_text)
        scored: list[tuple[int, int, str]] = []
        for name, record in self._records_by_name.items():
            tool_obj = self.full_toolkit.tools[name]
            blob = record.search_blob()
            score = 0
            if name.lower() in context_text:
                score += 50
            for term in context_terms:
                if term == name.lower() or term == record.handle.lower():
                    score += 20
                elif term in name.lower() or term in record.handle.lower():
                    score += 10
                elif term in blob:
                    score += 3
            if bool(getattr(tool_obj, "defer_by_default", False)):
                score -= 5
            scored.append((score, 1 if bool(getattr(tool_obj, "defer_by_default", False)) else 0, name))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [name for _, _, name in scored]

    def _resolve_direct_original_names(self, selected_names: list[str]) -> list[str]:
        always = _unique_names(self._always_load_names())
        reserved_meta_count = len(META_TOOL_NAMES)
        budget_for_original = max(0, self.config.max_direct_tools - reserved_meta_count)
        if len(always) >= budget_for_original:
            return always

        remaining_budget = budget_for_original - len(always)
        direct = list(always)
        for name in selected_names:
            if name in direct or name not in self.full_toolkit.tools or name in META_TOOL_NAMES:
                continue
            direct.append(name)
            if len(direct) >= len(always) + remaining_budget:
                break
        return direct

    def _build_exposed_toolkit(self, direct_original_names: list[str]) -> Toolkit:
        toolkit = Toolkit(prompt_sections=self._prompt_sections())
        for name in direct_original_names:
            tool_obj = self.full_toolkit.get(name)
            if tool_obj is not None:
                toolkit.register(tool_obj)
        for meta_tool in self._build_meta_tools():
            toolkit.register(meta_tool)
        return toolkit

    def _prompt_sections(self) -> tuple[str, ...]:
        base_sections = tuple(getattr(self.full_toolkit, "prompt_sections", ()) or ())
        summary = self._render_deferred_summary()
        if not summary:
            return base_sections
        return (*base_sections, summary)

    def _render_deferred_summary(self) -> str:
        deferred_names = [
            name
            for name in self.full_toolkit.tools
            if name not in self.exposed_tool_names()
            and name not in META_TOOL_NAMES
        ]
        if not deferred_names:
            return ""
        lines = [
            "## Deferred tools",
            (
                "Deferred tools available through tool_search/tool_describe/tool_load/"
                "tool_execute_deferred. Provider-native callable tools are the only "
                "directly callable tools this turn."
            ),
            "Compact catalog:",
        ]
        for name in deferred_names[:200]:
            record = self._records_by_name.get(name)
            if record is None:
                continue
            lines.append(
                f"- handle={record.handle}; name={record.name}; "
                f"toolkit={record.toolkit}; category={record.category}; "
                f"description={record.description}"
            )
        remaining = len(deferred_names) - 200
        if remaining > 0:
            lines.append(f"- ... {remaining} more deferred tools omitted from prompt; use tool_search.")
        return "\n".join(lines)

    def exposed_tool_names(self) -> set[str]:
        if isinstance(self.exposed_toolkit, Toolkit) and self.exposed_toolkit is not self.full_toolkit:
            return set(self.exposed_toolkit.tools)
        return set(self.plan.direct_tool_names)

    def _refresh_prompt_sections(self) -> None:
        if self.exposed_toolkit is self.full_toolkit:
            return
        self.exposed_toolkit.prompt_sections = self._prompt_sections()

    def _build_meta_tools(self) -> tuple[Tool, Tool, Tool, Tool]:
        return (
            Tool.from_callable(
                self.tool_search,
                name=TOOL_SEARCH_NAME,
                description="Search deferred tools that are not directly exposed this turn.",
                parameters=[
                    {
                        "name": "query",
                        "description": "Search query describing the capability or tool name.",
                        "type_": "string",
                        "required": True,
                    },
                    {
                        "name": "max_results",
                        "description": "Maximum number of matches to return.",
                        "type_": "integer",
                        "required": False,
                    },
                ],
                always_load=True,
                prompt_spec=ToolPromptSpec(
                    purpose="Search the compact catalog of deferred tools before loading or executing one.",
                    when_to_use=("A needed capability is not visible in the active native tool list.",),
                    examples=('tool_search(query="github pull request", max_results=5)',),
                ),
            ),
            Tool.from_callable(
                self.tool_describe,
                name=TOOL_DESCRIBE_NAME,
                description="Return full schemas for a small number of deferred tools.",
                parameters=[
                    {
                        "name": "handles",
                        "description": "Deferred tool handles returned by tool_search.",
                        "type_": "array",
                        "items": {"type": "string"},
                        "required": False,
                    },
                    {
                        "name": "names",
                        "description": "Deferred tool names.",
                        "type_": "array",
                        "items": {"type": "string"},
                        "required": False,
                    },
                ],
                always_load=True,
                prompt_spec=ToolPromptSpec(
                    purpose="Inspect exact arguments for deferred tools before loading or same-turn execution.",
                    when_to_use=("You found a deferred tool and need its full argument schema.",),
                    examples=('tool_describe(names=["tool_9"])',),
                ),
            ),
            Tool.from_callable(
                self.tool_load,
                name=TOOL_LOAD_NAME,
                description="Load deferred tools into the active provider-native tool list for later turns.",
                parameters=[
                    {
                        "name": "handles",
                        "description": "Deferred tool handles returned by tool_search.",
                        "type_": "array",
                        "items": {"type": "string"},
                        "required": False,
                    },
                    {
                        "name": "names",
                        "description": "Deferred tool names.",
                        "type_": "array",
                        "items": {"type": "string"},
                        "required": False,
                    },
                ],
                always_load=True,
                prompt_spec=ToolPromptSpec(
                    purpose="Make deferred tools provider-native callable on the next model turn.",
                    when_to_use=("You know which deferred tool should remain available after this turn.",),
                    examples=('tool_load(handles=["tool_9"])',),
                ),
            ),
            Tool.from_callable(
                self.tool_execute_deferred,
                name=TOOL_EXECUTE_DEFERRED_NAME,
                description="Execute a deferred tool in the current turn after describing its schema.",
                parameters=[
                    {
                        "name": "tool_name",
                        "description": "Name or handle of the deferred tool to execute.",
                        "type_": "string",
                        "required": True,
                    },
                    {
                        "name": "arguments",
                        "description": "Arguments object for the target deferred tool.",
                        "type_": "object",
                        "required": False,
                    },
                ],
                always_load=True,
                prompt_spec=ToolPromptSpec(
                    purpose="Run a deferred tool immediately when waiting for the next turn would be inefficient.",
                    when_to_use=("You already know the target deferred tool schema and arguments.",),
                    when_not_to_use=("The tool is already available as a native provider tool.",),
                    examples=('tool_execute_deferred(tool_name="tool_9", arguments={"value": "x"})',),
                ),
            ),
        )

    def _remaining_records(self) -> list[ExposureToolRecord]:
        active = self.exposed_tool_names()
        return [
            record
            for name, record in self._records_by_name.items()
            if name not in active and name not in META_TOOL_NAMES
        ]

    def _resolve_records(
        self,
        handles: list[str] | tuple[str, ...] | str | None = None,
        names: list[str] | tuple[str, ...] | str | None = None,
    ) -> tuple[list[ExposureToolRecord], list[dict[str, Any]]]:
        raw_values: list[str] = []
        for raw_collection in (handles, names):
            if raw_collection is None:
                continue
            if isinstance(raw_collection, str):
                raw_values.append(raw_collection)
            elif isinstance(raw_collection, (list, tuple)):
                raw_values.extend(str(item or "") for item in raw_collection)
            else:
                raw_values.append(str(raw_collection or ""))

        records: list[ExposureToolRecord] = []
        failed: list[dict[str, Any]] = []
        for value in _unique_names(raw_values):
            record = self._records_by_name.get(value)
            if record is None:
                failed.append({"handle": value, "error": "unknown deferred tool handle"})
                continue
            records.append(record)
        return records, failed

    def tool_search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        resolved_max = max(1, min(int(max_results or 5), 50))
        query_text = str(query or "").strip().lower()
        if not query_text:
            matches = self._remaining_records()[:resolved_max]
        else:
            terms = _terms(query_text)
            scored: list[tuple[int, ExposureToolRecord]] = []
            for record in self._remaining_records():
                blob = record.search_blob()
                score = 0
                if query_text == record.handle.lower() or query_text == record.name.lower():
                    score += 100
                for term in terms:
                    if term == record.handle.lower() or term == record.name.lower():
                        score += 20
                    elif term in record.handle.lower() or term in record.name.lower():
                        score += 10
                    elif term in blob:
                        score += 4
                if score > 0:
                    scored.append((score, record))
            scored.sort(key=lambda item: (-item[0], item[1].handle))
            matches = [record for _, record in scored[:resolved_max]]

        return {
            "matches": [record.to_summary() for record in matches],
            "query": str(query or ""),
            "total_matches": len(matches),
            "total_deferred_tools": len(self._remaining_records()),
        }

    def tool_describe(
        self,
        handles: list[str] | str | None = None,
        names: list[str] | str | None = None,
    ) -> dict[str, Any]:
        records, failed = self._resolve_records(handles=handles, names=names)
        tools: list[dict[str, Any]] = []
        for record in records[:10]:
            tool_obj = self.full_toolkit.get(record.name)
            if tool_obj is None:
                failed.append({"handle": record.handle, "error": "tool missing from full toolkit"})
                continue
            schema = tool_obj.to_json()
            schema["handle"] = record.handle
            tools.append(schema)
        return {"tools": tools, "failed": failed}

    def tool_load(
        self,
        handles: list[str] | str | None = None,
        names: list[str] | str | None = None,
    ) -> dict[str, Any]:
        records, failed = self._resolve_records(handles=handles, names=names)
        loaded: list[dict[str, Any]] = []
        already_loaded: list[dict[str, Any]] = []
        for record in records:
            tool_obj = self.full_toolkit.get(record.name)
            if tool_obj is None:
                failed.append({"handle": record.handle, "error": "tool missing from full toolkit"})
                continue
            if record.name in self.exposed_toolkit.tools:
                already_loaded.append(record.to_summary())
                continue
            self.exposed_toolkit.register(tool_obj)
            if record.name not in self._loaded_names:
                self._loaded_names.append(record.name)
            loaded.append(record.to_summary())

        direct_names = _unique_names([*self.plan.direct_tool_names, *self._loaded_names])
        deferred_names = [name for name in self.plan.deferred_tool_names if name not in set(self._loaded_names)]
        self.plan.direct_tool_names = direct_names
        self.plan.deferred_tool_names = deferred_names
        self.plan.loaded_tool_names = list(self._loaded_names)
        self._refresh_prompt_sections()
        return {
            "loaded": loaded,
            "already_loaded": already_loaded,
            "failed": failed,
        }

    def parse_execute_deferred_arguments(self, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if arguments is None:
            return {"error": "tool_execute_deferred requires arguments"}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return {"error": f"invalid JSON arguments: {exc}"}
        elif isinstance(arguments, dict):
            parsed = copy.deepcopy(arguments)
        else:
            return {"error": "invalid tool_execute_deferred arguments type"}
        if not isinstance(parsed, dict):
            return {"error": "tool_execute_deferred arguments must be an object"}
        tool_name = str(parsed.get("tool_name") or parsed.get("name") or parsed.get("handle") or "").strip()
        if not tool_name:
            return {"error": "tool_execute_deferred requires tool_name"}
        target_arguments = parsed.get("arguments")
        if target_arguments is None:
            target_arguments = {}
        if isinstance(target_arguments, str):
            try:
                target_arguments = json.loads(target_arguments or "{}")
            except json.JSONDecodeError:
                # Preserve non-JSON text for the runtime argument validator to reject with the normal tool error.
                pass
        return {"tool_name": tool_name, "arguments": target_arguments}

    def tool_execute_deferred(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        del tool_name, arguments
        return {
            "error": "tool_execute_deferred must be handled by the runtime execution plugin",
            "tool": TOOL_EXECUTE_DEFERRED_NAME,
        }


__all__ = [
    "META_TOOL_NAMES",
    "TOOL_DESCRIBE_NAME",
    "TOOL_EXECUTE_DEFERRED_NAME",
    "TOOL_LOAD_NAME",
    "TOOL_SEARCH_NAME",
    "ToolExposurePlan",
    "ToolExposureRuntime",
    "ToolOptimizerConfig",
]
