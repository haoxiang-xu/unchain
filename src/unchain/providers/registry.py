from __future__ import annotations

from typing import Any


def _default_adapter_classes() -> dict[str, type]:
    from .gemini import GeminiModelIO
    from .anthropic import AnthropicModelIO
    from .hyperspace import HyperspaceModelIO
    from .openai import OpenAIModelIO
    from .ollama import OllamaModelIO

    return {
        "anthropic": AnthropicModelIO,
        "gemini": GeminiModelIO,
        "hyperspace": HyperspaceModelIO,
        "ollama": OllamaModelIO,
        "openai": OpenAIModelIO,
    }


class ProviderAdapterRegistry:
    def __init__(
        self,
        entries: dict[str, type] | None = None,
        *,
        include_defaults: bool = True,
    ) -> None:
        self._entries: dict[str, type] = {}
        if include_defaults:
            for provider, adapter_cls in _default_adapter_classes().items():
                self.register(provider, adapter_cls)
        for provider, adapter_cls in (entries or {}).items():
            self.register(provider, adapter_cls)

    def register(self, provider: str, adapter_cls: type) -> None:
        provider_key = _normalize_provider(provider)
        self._entries[provider_key] = adapter_cls

    def get(self, provider: str) -> type:
        provider_key = _normalize_provider(provider)
        try:
            return self._entries[provider_key]
        except KeyError as exc:
            raise KeyError(f"unknown model provider: {provider_key}") from exc

    def create(self, provider: str, **kwargs: Any):
        return self.get(provider)(**kwargs)

    def as_dict(self) -> dict[str, type]:
        return dict(self._entries)


def _normalize_provider(provider: str) -> str:
    provider_key = str(provider or "").strip().lower()
    if not provider_key:
        raise ValueError("provider must be a non-empty string")
    return provider_key


_DEFAULT_REGISTRY: ProviderAdapterRegistry | None = None


def default_provider_registry() -> ProviderAdapterRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ProviderAdapterRegistry()
    return _DEFAULT_REGISTRY


def get_model_adapter_class(provider: str) -> type:
    return default_provider_registry().get(provider)


def create_model_adapter(provider: str, **kwargs: Any):
    return default_provider_registry().create(provider, **kwargs)


__all__ = [
    "ProviderAdapterRegistry",
    "create_model_adapter",
    "default_provider_registry",
    "get_model_adapter_class",
]
