from __future__ import annotations

import copy
import inspect
import json
from typing import Any, Callable, get_type_hints

from ..execution import ExecutionLeaseError
from .models import (
    HistoryPayloadOptimizer,
    ToolConfirmationPolicy,
    ToolExecutionContext,
    ToolParameter,
    ToolPromptSpec,
    _annotation_to_json_schema,
    _escape_control_chars_inside_json_strings,
    _parse_docstring,
)


class _InvalidToolArgumentsType(TypeError):
    pass


class Tool:
    def __init__(
        self,
        name: str | Callable[..., Any] = "",
        description: str = "",
        func: Callable[..., Any] | None = None,
        parameters: list[ToolParameter | dict[str, Any]] | None = None,
        observe: bool = False,
        requires_confirmation: bool = False,
        render_component: dict[str, Any] | None = None,
        confirmation_resolver: (
            Callable[[dict[str, Any], ToolExecutionContext | None], ToolConfirmationPolicy | bool | dict[str, Any] | None]
            | None
        ) = None,
        prompt_spec: ToolPromptSpec | dict[str, Any] | None = None,
        history_arguments_optimizer: HistoryPayloadOptimizer | None = None,
        history_result_optimizer: HistoryPayloadOptimizer | None = None,
        icon_path: str = "",
        icon: dict[str, Any] | None = None,
        always_load: bool = False,
        defer_by_default: bool = False,
        search_hint: str = "",
        provider_native_specs: dict[str, dict[str, Any]] | None = None,
        required_betas: dict[str, list[str]] | None = None,
    ):
        if callable(name) and func is None:
            func = name
            name = ""

        self.name = name
        self.description = description
        self.func = func
        self.observe = observe
        self.requires_confirmation = requires_confirmation
        self.render_component = render_component
        self.confirmation_resolver = confirmation_resolver
        self.prompt_spec = ToolPromptSpec.from_raw(prompt_spec)
        self.history_arguments_optimizer = history_arguments_optimizer
        self.history_result_optimizer = history_result_optimizer
        self.icon_path = icon_path
        self.icon = dict(icon) if isinstance(icon, dict) else None
        self.always_load = bool(always_load)
        self.defer_by_default = bool(defer_by_default)
        self.search_hint = str(search_hint or "")
        # Provider-native tool specs (e.g. Anthropic predefined `computer` tool)
        # keyed by normalized provider name. When present for the requested
        # provider, `to_provider_json` returns the native spec instead of the
        # generic function schema. `required_betas` declares provider beta flags
        # a tool needs (e.g. {"anthropic": ["computer-use-2025-11-24"]}).
        self.provider_native_specs = self._construct_provider_native_specs(provider_native_specs)
        self.required_betas = self._construct_required_betas(required_betas)
        self.parameters = self._construct_parameters(parameters)

        if self.func is not None and not self.parameters:
            self.parameters = self._infer_parameters_from_func(self.func)

        if not self.name and self.func is not None:
            self.name = self.func.__name__

        if not self.description and self.func is not None:
            summary, _ = _parse_docstring(self.func)
            self.description = summary

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            return Tool.from_callable(
                func,
                name=self.name or None,
                description=self.description or None,
                parameters=self.parameters or None,
                observe=self.observe,
                requires_confirmation=self.requires_confirmation,
                render_component=self.render_component,
                confirmation_resolver=self.confirmation_resolver,
                prompt_spec=self.prompt_spec,
                history_arguments_optimizer=self.history_arguments_optimizer,
                history_result_optimizer=self.history_result_optimizer,
                icon_path=self.icon_path,
                icon=self.icon,
                always_load=self.always_load,
                defer_by_default=self.defer_by_default,
                search_hint=self.search_hint,
                provider_native_specs=self.provider_native_specs,
                required_betas=self.required_betas,
            )

        if self.func is not None:
            return self.func(*args, **kwargs)

        raise TypeError("Tool object is not callable without a wrapped function")

    def invoke(self, call: Any, context: Any = None):
        del context
        from ..capabilities import normalize_capability_outcome

        if self.func is None:
            return normalize_capability_outcome(
                {"error": "tool function not implemented (Tool -> invoke)"},
                created_by=f"tool.{self.name}",
            )

        arguments = call
        if isinstance(call, dict) and "arguments" in call:
            arguments = call.get("arguments")
        elif hasattr(call, "arguments"):
            arguments = getattr(call, "arguments")

        try:
            parsed_arguments = self._parse_arguments(arguments)
            result = self.func(**parsed_arguments)
            return normalize_capability_outcome(
                result,
                created_by=f"tool.{self.name}",
            )
        except _InvalidToolArgumentsType:
            return normalize_capability_outcome(
                {"error": "invalid tool arguments type"},
                created_by=f"tool.{self.name}",
            )
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            return normalize_capability_outcome(
                {"error": str(exc), "tool": self.name},
                created_by=f"tool.{self.name}",
            )

    @classmethod
    def from_callable(
        cls,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: list[ToolParameter | dict[str, Any]] | None = None,
        observe: bool = False,
        requires_confirmation: bool = False,
        render_component: dict[str, Any] | None = None,
        confirmation_resolver: (
            Callable[[dict[str, Any], ToolExecutionContext | None], ToolConfirmationPolicy | bool | dict[str, Any] | None]
            | None
        ) = None,
        prompt_spec: ToolPromptSpec | dict[str, Any] | None = None,
        history_arguments_optimizer: HistoryPayloadOptimizer | None = None,
        history_result_optimizer: HistoryPayloadOptimizer | None = None,
        icon_path: str = "",
        icon: dict[str, Any] | None = None,
        always_load: bool = False,
        defer_by_default: bool = False,
        search_hint: str = "",
        provider_native_specs: dict[str, dict[str, Any]] | None = None,
        required_betas: dict[str, list[str]] | None = None,
    ) -> "Tool":
        summary, _ = _parse_docstring(func)
        return cls(
            name=name or func.__name__,
            description=description or summary,
            func=func,
            parameters=parameters,
            observe=observe,
            requires_confirmation=requires_confirmation,
            render_component=render_component,
            confirmation_resolver=confirmation_resolver,
            prompt_spec=prompt_spec,
            history_arguments_optimizer=history_arguments_optimizer,
            history_result_optimizer=history_result_optimizer,
            icon_path=icon_path,
            icon=icon,
            always_load=always_load,
            defer_by_default=defer_by_default,
            search_hint=search_hint,
            provider_native_specs=provider_native_specs,
            required_betas=required_betas,
        )

    @staticmethod
    def _construct_provider_native_specs(
        specs: dict[str, dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(specs, dict):
            return {}
        constructed: dict[str, dict[str, Any]] = {}
        for provider, spec in specs.items():
            key = str(provider or "").strip().lower()
            if key and isinstance(spec, dict):
                constructed[key] = copy.deepcopy(spec)
        return constructed

    @staticmethod
    def _construct_required_betas(
        required_betas: dict[str, list[str]] | None,
    ) -> dict[str, list[str]]:
        if not isinstance(required_betas, dict):
            return {}
        constructed: dict[str, list[str]] = {}
        for provider, betas in required_betas.items():
            key = str(provider or "").strip().lower()
            if not key or not isinstance(betas, (list, tuple)):
                continue
            values = [str(beta).strip() for beta in betas if str(beta or "").strip()]
            if values:
                constructed[key] = values
        return constructed

    def _provider_native_spec(self, normalized_provider: str) -> dict[str, Any] | None:
        spec = self.provider_native_specs.get(normalized_provider)
        return copy.deepcopy(spec) if isinstance(spec, dict) else None

    def required_betas_for(self, provider: str | None) -> list[str]:
        normalized = str(provider or "").strip().lower()
        return list(self.required_betas.get(normalized, []))

    def _construct_parameters(
        self,
        parameters: list[ToolParameter | dict[str, Any]] | None,
    ) -> list[ToolParameter]:
        constructed_parameters: list[ToolParameter] = []
        for parameter in parameters or []:
            if isinstance(parameter, ToolParameter):
                constructed_parameters.append(parameter)
            elif isinstance(parameter, dict):
                constructed_parameters.append(
                    ToolParameter(
                        name=parameter.get("name", ""),
                        description=parameter.get("description", ""),
                        type_=parameter.get("type_", "string"),
                        required=parameter.get("required", False),
                        pattern=parameter.get("pattern"),
                        items=parameter.get("items"),
                    )
                )
        return constructed_parameters

    def _infer_parameters_from_func(self, func: Callable[..., Any]) -> list[ToolParameter]:
        inferred: list[ToolParameter] = []
        signature = inspect.signature(func)
        try:
            resolved_hints = get_type_hints(func)
        except Exception:
            resolved_hints = {}
        _, parameter_descriptions = _parse_docstring(func)

        for name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue

            annotation = resolved_hints.get(name, param.annotation)
            if annotation == inspect._empty:
                annotation = str
            schema = _annotation_to_json_schema(annotation)
            type_ = schema.get("type", "string")
            items = schema.get("items") if type_ == "array" else None
            required = param.default == inspect._empty
            inferred.append(
                ToolParameter(
                    name=name,
                    description=parameter_descriptions.get(name, f"Argument {name}"),
                    type_=type_,
                    required=required,
                    items=items,
                )
            )
        return inferred

    def _parameters_json_schema(self) -> dict[str, Any]:
        json_parameters: dict[str, Any] = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        for parameter in self.parameters:
            json_parameters["properties"][parameter.name] = parameter.to_json()
            if parameter.required:
                json_parameters["required"].append(parameter.name)

        return json_parameters

    def to_json(self) -> dict[str, Any]:
        payload = {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self._parameters_json_schema(),
        }
        if self.icon is not None:
            payload["icon_path"] = self.icon_path
            payload["icon"] = self.icon
        return payload

    def to_provider_json(self, provider: str | None = None) -> dict[str, Any]:
        normalized_provider = str(provider or "openai").strip().lower()

        # Provider-native predefined tool (e.g. Anthropic `computer`): the native
        # spec replaces the generic function/input_schema shape. Providers with
        # no native spec fall back to the generic schema below (same execute).
        native_spec = self._provider_native_spec(normalized_provider)
        if native_spec is not None:
            return native_spec

        parameters = self._parameters_json_schema()

        if normalized_provider in {"anthropic", "hyperspace"}:
            return {
                "name": self.name,
                "description": self.description,
                "input_schema": parameters,
            }

        if normalized_provider == "ollama":
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": parameters,
                },
            }

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def _parse_arguments(self, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if not stripped:
                return {}
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                repaired = _escape_control_chars_inside_json_strings(stripped)
                return json.loads(repaired)
        if isinstance(arguments, dict):
            return arguments
        raise _InvalidToolArgumentsType("invalid tool arguments type")

    def execute(self, arguments: dict[str, Any] | str | None) -> dict[str, Any]:
        if self.func is None:
            return {"error": "tool function not implemented (Tool -> execute)"}

        try:
            parsed_arguments = self._parse_arguments(arguments)
            result = self.func(**parsed_arguments)
            if isinstance(result, dict):
                return result
            return {"result": result}
        except _InvalidToolArgumentsType:
            return {"error": "invalid tool arguments type"}
        except ExecutionLeaseError:
            raise
        except Exception as exc:
            return {"error": str(exc), "tool": self.name}


__all__ = ["Tool"]
