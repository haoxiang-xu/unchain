from __future__ import annotations

import copy
from typing import Any, Callable

from ..execution import ExecutionGuard
from ..kernel.types import KernelRunResult
from ..interaction.durable import InteractionReceipt
from ..runtime.module_context import AgentRuntimeContext
from ..run_bundle import RunDescriptor
from ..tools import Tool
from .builder import AgentBuilder, AgentCallContext
from .model_io import ModelIOFactoryRegistry
from .spec import AgentSpec, AgentState


class Agent:
    def __init__(
        self,
        *,
        name: str,
        instructions: str = "",
        provider: str = "openai",
        model: str = "gpt-5",
        api_key: str | None = None,
        modules: tuple[Any, ...] = (),
        allowed_tools: tuple[str, ...] | None = None,
        missing_tool_policy: str = "raise",
        model_io_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Agent name is required")
        self.spec = AgentSpec(
            name=name.strip(),
            instructions=instructions or "",
            provider=provider or "openai",
            model=model or "gpt-5",
            api_key=api_key,
            modules=tuple(modules or ()),
            allowed_tools=tuple(allowed_tools) if allowed_tools is not None else None,
            missing_tool_policy=missing_tool_policy,
        )
        self.state = AgentState()
        self._model_io_registry = ModelIOFactoryRegistry()
        self._model_io_factory = model_io_factory

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def instructions(self) -> str:
        return self.spec.instructions

    @property
    def provider(self) -> str:
        return self.spec.provider

    @property
    def model(self) -> str:
        return self.spec.model

    @property
    def allowed_tools(self) -> tuple[str, ...] | None:
        return self.spec.allowed_tools

    def _resolve_run_bundle_descriptor(self, value: Any = None) -> RunDescriptor:
        if value is not None:
            if type(value) is not RunDescriptor:
                raise TypeError(
                    "_run_bundle_descriptor must be an exact RunDescriptor"
                )
            return value
        orchestration = str(
            getattr(self, "_orchestration_next_mode", "default") or "default"
        )
        if orchestration not in {"default", "developer_waiting_approval"}:
            orchestration = "default"
        return RunDescriptor(
            model=self.model or "unknown-model",
            display_model=str(
                getattr(self, "_display_model", "")
                or f"{self.provider}:{self.model}"
            ),
            active_agent=str(
                getattr(self, "_orchestration_role", "")
                or self.name
                or "unknown"
            ),
            agent_orchestration=orchestration,
        )

    def _normalize_messages(self, messages: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(messages, str):
            normalized = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            normalized = []
            for message in messages:
                if not isinstance(message, dict):
                    raise TypeError("messages must be a string or a list of dict messages")
                normalized.append(copy.deepcopy(message))
        else:
            raise TypeError("messages must be a string or a list of dict messages")
        if self.instructions:
            first = normalized[0] if normalized else None
            if (
                isinstance(first, dict)
                and first.get("role") == "system"
                and first.get("content") == self.instructions
            ):
                return normalized
            return [{"role": "system", "content": self.instructions}, *normalized]
        return normalized

    def _prepare(self, call_context: AgentCallContext):
        builder = AgentBuilder(
            agent=self,
            spec=self.spec,
            state=self.state,
            call_context=call_context,
            model_io_registry=self._model_io_registry,
        )
        if self._model_io_factory is not None:
            builder.set_model_io_factory(self._model_io_factory)
        for module in self.spec.modules:
            module.configure(builder)
        return builder.build()

    def clone(
        self,
        *,
        name: str | None = None,
        instructions: str | None = None,
        modules: tuple[Any, ...] | None = None,
        model: str | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        missing_tool_policy: str | None = None,
    ) -> "Agent":
        return Agent(
            name=name or self.name,
            instructions=self.instructions if instructions is None else instructions,
            provider=self.provider,
            model=self.model if model is None else model,
            api_key=self.spec.api_key,
            modules=tuple(self.spec.modules if modules is None else modules),
            allowed_tools=self.spec.allowed_tools if allowed_tools is None else tuple(allowed_tools),
            missing_tool_policy=(
                self.spec.missing_tool_policy if missing_tool_policy is None else missing_tool_policy
            ),
            model_io_factory=self._model_io_factory,
        )

    def fork_for_subagent(
        self,
        *,
        subagent_name: str,
        mode: str,
        parent_name: str,
        lineage: list[str],
        task: str,
        instructions: str,
        expected_output: str,
        disabled_module_keys: tuple[str, ...] = (),
        model: str | None = None,
        allowed_tools: tuple[str, ...] | None = None,
        missing_tool_policy: str | None = None,
    ) -> "Agent":
        overlay = (
            f'You are subagent "{subagent_name}" created by parent "{parent_name}".\n'
            f"Mode: {mode}\n"
            f"Lineage: {' > '.join(lineage)}\n\n"
            "Only execute the delegated subtask.\n"
            "Do not ask the user directly for clarification. If clarification is required, return a concise structured clarification request via the runtime tools.\n"
            f"Delegated task:\n{task.strip()}\n\n"
        )
        if expected_output.strip():
            overlay += f"Expected output:\n{expected_output.strip()}\n\n"
        if instructions.strip():
            overlay += f"Extra instructions:\n{instructions.strip()}\n"
        disabled = {
            str(key).strip()
            for key in disabled_module_keys
            if isinstance(key, str) and str(key).strip()
        }
        modules = [
            module
            for module in self.spec.modules
            if str(getattr(module, "name", "") or "").strip() not in disabled
        ]
        return self.clone(
            name=subagent_name,
            instructions="\n\n".join(part for part in (self.instructions, overlay.strip()) if part.strip()),
            modules=tuple(modules),
            model=model,
            allowed_tools=allowed_tools,
            missing_tool_policy=missing_tool_policy,
        )

    def run(
        self,
        messages: str | list[dict[str, Any]],
        *,
        payload: dict[str, Any] | None = None,
        response_format: Any = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = False,
        max_iterations: int | None = None,
        max_context_window_tokens: int | None = None,
        previous_response_id: str | None = None,
        on_tool_confirm: Callable[..., Any] | None = None,
        on_human_input: Callable[..., Any] | None = None,
        on_max_iterations: Callable[..., Any] | None = None,
        session_id: str | None = None,
        memory_namespace: str | None = None,
        run_id: str | None = None,
        runtime_context: AgentRuntimeContext | None = None,
        execution_owner_id: str | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        _execution_guard: ExecutionGuard | None = None,
        _run_bundle_identity: Any = None,
        _run_bundle_descriptor: Any = None,
        _continued_from_run_id: str | None = None,
        _provider_turn_ownership_factory: Any = None,
        _initial_iteration: int = 0,
        _recovered_tool_turn: Any = None,
    ) -> KernelRunResult:
        prepared = self._prepare(
            AgentCallContext(
                mode="run",
                input_messages=self._normalize_messages(messages),
                payload=copy.deepcopy(payload) if isinstance(payload, dict) else None,
                response_format=response_format,
                callback=callback,
                verbose=verbose,
                max_iterations=max_iterations,
                max_context_window_tokens=max_context_window_tokens,
                previous_response_id=previous_response_id,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                on_max_iterations=on_max_iterations,
                session_id=session_id,
                memory_namespace=memory_namespace,
                run_id=run_id,
                runtime_context=runtime_context,
                run_bundle_identity=_run_bundle_identity,
                run_bundle_descriptor=self._resolve_run_bundle_descriptor(
                    _run_bundle_descriptor
                ),
                continued_from_run_id=_continued_from_run_id,
                provider_turn_ownership_factory=(
                    _provider_turn_ownership_factory
                ),
                execution_owner_id=execution_owner_id,
                execution_guard=_execution_guard,
                tool_runtime_config=copy.deepcopy(tool_runtime_config)
                if isinstance(tool_runtime_config, dict)
                else None,
                initial_iteration=_initial_iteration,
                recovered_tool_turn=_recovered_tool_turn,
            )
        )
        return prepared.run()

    def resume_human_input(
        self,
        *,
        conversation: list[dict[str, Any]] | None = None,
        continuation: dict[str, Any] | None = None,
        response: dict[str, Any] | Any = None,
        payload: dict[str, Any] | None = None,
        response_format: Any = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = False,
        on_tool_confirm: Callable[..., Any] | None = None,
        on_human_input: Callable[..., Any] | None = None,
        on_max_iterations: Callable[..., Any] | None = None,
        session_id: str | None = None,
        memory_namespace: str | None = None,
        run_id: str | None = None,
        runtime_context: AgentRuntimeContext | None = None,
        execution_owner_id: str | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        _run_bundle_identity: Any = None,
        _run_bundle_descriptor: Any = None,
        _provider_turn_ownership_factory: Any = None,
    ) -> KernelRunResult:
        prepared = self._prepare(
            AgentCallContext(
                mode="resume_human_input",
                conversation=(
                    copy.deepcopy(conversation)
                    if isinstance(conversation, list)
                    else None
                ),
                continuation=(
                    copy.deepcopy(continuation)
                    if isinstance(continuation, dict)
                    else None
                ),
                response=copy.deepcopy(response),
                payload=copy.deepcopy(payload) if isinstance(payload, dict) else None,
                response_format=response_format,
                callback=callback,
                verbose=verbose,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                on_max_iterations=on_max_iterations,
                session_id=session_id,
                memory_namespace=memory_namespace,
                run_id=run_id,
                runtime_context=runtime_context,
                run_bundle_identity=_run_bundle_identity,
                run_bundle_descriptor=self._resolve_run_bundle_descriptor(
                    _run_bundle_descriptor
                ),
                provider_turn_ownership_factory=(
                    _provider_turn_ownership_factory
                ),
                execution_owner_id=execution_owner_id,
                tool_runtime_config=copy.deepcopy(tool_runtime_config)
                if isinstance(tool_runtime_config, dict)
                else None,
            )
        )
        return prepared.resume_human_input()

    def submit_interaction(
        self,
        *,
        session_id: str,
        interaction_id: str,
        response: Any,
        submitted_by: str = "user",
    ) -> InteractionReceipt:
        prepared = self._prepare(
            AgentCallContext(
                mode="submit_interaction",
                session_id=session_id,
                interaction_id=interaction_id,
                response=copy.deepcopy(response),
                submitted_by=submitted_by,
            )
        )
        return prepared.submit_interaction()

    def resume_interaction(
        self,
        *,
        session_id: str,
        conversation: list[dict[str, Any]] | None = None,
        continuation: dict[str, Any] | None = None,
        response: Any = None,
        submitted_by: str = "api:resume_interaction",
        payload: dict[str, Any] | None = None,
        response_format: Any = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
        verbose: bool = False,
        on_tool_confirm: Callable[..., Any] | None = None,
        on_human_input: Callable[..., Any] | None = None,
        on_max_iterations: Callable[..., Any] | None = None,
        memory_namespace: str | None = None,
        run_id: str | None = None,
        runtime_context: AgentRuntimeContext | None = None,
        execution_owner_id: str | None = None,
        tool_runtime_config: dict[str, Any] | None = None,
        _run_bundle_identity: Any = None,
        _run_bundle_descriptor: Any = None,
        _provider_turn_ownership_factory: Any = None,
    ) -> KernelRunResult:
        prepared = self._prepare(
            AgentCallContext(
                mode="resume_interaction",
                conversation=(
                    copy.deepcopy(conversation)
                    if isinstance(conversation, list)
                    else None
                ),
                continuation=(
                    copy.deepcopy(continuation)
                    if isinstance(continuation, dict)
                    else None
                ),
                response=copy.deepcopy(response),
                submitted_by=submitted_by,
                payload=copy.deepcopy(payload) if isinstance(payload, dict) else None,
                response_format=response_format,
                callback=callback,
                verbose=verbose,
                on_tool_confirm=on_tool_confirm,
                on_human_input=on_human_input,
                on_max_iterations=on_max_iterations,
                session_id=session_id,
                memory_namespace=memory_namespace,
                run_id=run_id,
                runtime_context=runtime_context,
                run_bundle_identity=_run_bundle_identity,
                run_bundle_descriptor=self._resolve_run_bundle_descriptor(
                    _run_bundle_descriptor
                ),
                provider_turn_ownership_factory=(
                    _provider_turn_ownership_factory
                ),
                execution_owner_id=execution_owner_id,
                tool_runtime_config=copy.deepcopy(tool_runtime_config)
                if isinstance(tool_runtime_config, dict)
                else None,
            )
        )
        return prepared.resume_interaction()

    def _last_assistant_text(self, messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages or []):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return ""

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        max_iterations: int | None = None,
    ) -> Tool:
        tool_name = name or self.name
        tool_description = description or f"Delegate the task to kernel agent '{self.name}'."

        def _delegate(task: str) -> dict[str, Any]:
            result = self.run(task, max_iterations=max_iterations)
            return {
                "agent": self.name,
                "status": result.status,
                "output": self._last_assistant_text(result.messages),
                "continuation": copy.deepcopy(result.continuation),
                "human_input_request": copy.deepcopy(result.human_input_request),
                "interaction_request": copy.deepcopy(result.interaction_request),
            }

        return Tool.from_callable(
            _delegate,
            name=tool_name,
            description=tool_description,
        )
