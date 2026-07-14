from __future__ import annotations

from typing import Any, Callable

from .models import HistoryPayloadOptimizer, ToolConfirmationPolicy, ToolExecutionContext, ToolParameter, ToolPromptSpec
from .tool import Tool


class Toolkit:
    def __init__(
        self,
        tools: dict[str, Tool] | None = None,
        *,
        prompt_sections: str | list[str] | tuple[str, ...] | None = None,
    ):
        self.tools: dict[str, Tool] = {}
        self.prompt_sections = self._normalize_prompt_sections(prompt_sections)
        for tool_name, tool_obj in (tools or {}).items():
            if isinstance(tool_obj, Tool):
                self.tools[tool_name] = tool_obj

    @staticmethod
    def _normalize_prompt_sections(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            text = value.strip()
            return (text,) if text else ()
        if isinstance(value, (list, tuple)):
            sections: list[str] = []
            for item in value:
                text = str(item or "").strip()
                if text:
                    sections.append(text)
            return tuple(sections)
        raise TypeError(
            "toolkit prompt_sections must be a string, list of strings, "
            "tuple of strings, or None"
        )

    def register(
        self,
        tool_obj: Tool | Callable[..., Any],
        *,
        observe: bool | None = None,
        requires_confirmation: bool | None = None,
        render_component: dict[str, Any] | None = None,
        confirmation_resolver: (
            Callable[[dict[str, Any], ToolExecutionContext | None], ToolConfirmationPolicy | bool | dict[str, Any] | None]
            | None
        ) = None,
        prompt_spec: ToolPromptSpec | dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter | dict[str, Any]] | None = None,
        history_arguments_optimizer: HistoryPayloadOptimizer | None = None,
        history_result_optimizer: HistoryPayloadOptimizer | None = None,
        always_load: bool | None = None,
        defer_by_default: bool | None = None,
        search_hint: str | None = None,
    ) -> Tool:
        if isinstance(tool_obj, Tool):
            if name is not None:
                tool_obj.name = name
            if description is not None:
                tool_obj.description = description
            if parameters is not None:
                tool_obj.parameters = tool_obj._construct_parameters(parameters)
            if observe is not None:
                tool_obj.observe = observe
            if requires_confirmation is not None:
                tool_obj.requires_confirmation = requires_confirmation
            if render_component is not None:
                tool_obj.render_component = render_component
            if confirmation_resolver is not None:
                tool_obj.confirmation_resolver = confirmation_resolver
            if prompt_spec is not None:
                tool_obj.prompt_spec = ToolPromptSpec.from_raw(prompt_spec)
            if history_arguments_optimizer is not None:
                tool_obj.history_arguments_optimizer = history_arguments_optimizer
            if history_result_optimizer is not None:
                tool_obj.history_result_optimizer = history_result_optimizer
            if always_load is not None:
                tool_obj.always_load = bool(always_load)
            if defer_by_default is not None:
                tool_obj.defer_by_default = bool(defer_by_default)
            if search_hint is not None:
                tool_obj.search_hint = str(search_hint or "")
            self.tools[tool_obj.name] = tool_obj
            return tool_obj

        if callable(tool_obj):
            wrapped = Tool.from_callable(
                tool_obj,
                name=name,
                description=description,
                parameters=parameters,
                observe=bool(observe),
                requires_confirmation=bool(requires_confirmation),
                render_component=render_component,
                confirmation_resolver=confirmation_resolver,
                prompt_spec=prompt_spec,
                history_arguments_optimizer=history_arguments_optimizer,
                history_result_optimizer=history_result_optimizer,
                always_load=bool(always_load),
                defer_by_default=bool(defer_by_default),
                search_hint=str(search_hint or ""),
            )
            self.tools[wrapped.name] = wrapped
            return wrapped

        raise ValueError("invalid tool passed to register")

    def register_many(self, *tool_objs: Tool | Callable[..., Any]) -> list[Tool]:
        registered: list[Tool] = []
        for tool_obj in tool_objs:
            registered.append(self.register(tool_obj))
        return registered

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        observe: bool = False,
        requires_confirmation: bool = False,
        name: str | None = None,
        description: str | None = None,
        confirmation_resolver: (
            Callable[[dict[str, Any], ToolExecutionContext | None], ToolConfirmationPolicy | bool | dict[str, Any] | None]
            | None
        ) = None,
        prompt_spec: ToolPromptSpec | dict[str, Any] | None = None,
        parameters: list[ToolParameter | dict[str, Any]] | None = None,
        history_arguments_optimizer: HistoryPayloadOptimizer | None = None,
        history_result_optimizer: HistoryPayloadOptimizer | None = None,
        always_load: bool = False,
        defer_by_default: bool = False,
        search_hint: str = "",
    ):
        if func is not None:
            return self.register(
                func,
                observe=observe,
                requires_confirmation=requires_confirmation,
                name=name,
                description=description,
                confirmation_resolver=confirmation_resolver,
                prompt_spec=prompt_spec,
                parameters=parameters,
                history_arguments_optimizer=history_arguments_optimizer,
                history_result_optimizer=history_result_optimizer,
                always_load=always_load,
                defer_by_default=defer_by_default,
                search_hint=search_hint,
            )

        def decorator(inner: Callable[..., Any]) -> Tool:
            return self.register(
                inner,
                observe=observe,
                requires_confirmation=requires_confirmation,
                name=name,
                description=description,
                confirmation_resolver=confirmation_resolver,
                prompt_spec=prompt_spec,
                parameters=parameters,
                history_arguments_optimizer=history_arguments_optimizer,
                history_result_optimizer=history_result_optimizer,
                always_load=always_load,
                defer_by_default=defer_by_default,
                search_hint=search_hint,
            )

        return decorator

    def get(self, function_name: str) -> Tool | None:
        return self.tools.get(function_name)

    def execute(self, function_name: str, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        tool_obj = self.get(function_name)
        if tool_obj is None:
            return {"error": f"tool not found: {function_name}", "tool": function_name}
        return tool_obj.execute(arguments)

    def to_json(self) -> list[dict[str, Any]]:
        return [tool_obj.to_json() for tool_obj in self.tools.values()]

    def to_provider_json(self, provider: str | None = None) -> list[dict[str, Any]]:
        return [tool_obj.to_provider_json(provider) for tool_obj in self.tools.values()]

    def required_betas(self, provider: str | None = None) -> list[str]:
        """Aggregate provider beta flags declared by the registered tools.

        Order is stable (registration order, first occurrence wins) and
        deduplicated, so a provider can pass them as a single ``anthropic-beta``
        header without repeats.
        """
        collected: list[str] = []
        for tool_obj in self.tools.values():
            for beta in tool_obj.required_betas_for(provider):
                if beta not in collected:
                    collected.append(beta)
        return collected

    def shutdown(self) -> None:
        return None


__all__ = ["Toolkit"]
