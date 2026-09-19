from __future__ import annotations

import os
from typing import Any, Callable

from ..kernel.model_io import ModelIO
from ..providers import (
    GeminiModelIO,
    AnthropicModelIO,
    HyperspaceModelIO,
    OllamaModelIO,
    OpenAIModelIO,
)


class ModelIOFactoryRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., ModelIO]] = {
            "gemini": self._create_gemini,
            "openai": self._create_openai,
            "anthropic": self._create_anthropic,
            "ollama": self._create_ollama,
            "hyperspace": self._create_hyperspace,
        }

    def register(self, provider: str, factory: Callable[..., ModelIO]) -> None:
        normalized = str(provider or "").strip().lower()
        if not normalized:
            raise ValueError("provider is required")
        self._factories[normalized] = factory

    def create(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None,
    ) -> ModelIO:
        normalized = str(provider or "").strip().lower()
        factory = self._factories.get(normalized)
        if factory is None:
            raise NotImplementedError(f"no model io factory registered for provider={provider!r}")
        return factory(model=model, api_key=api_key)

    def _create_openai(self, *, model: str, api_key: str | None) -> ModelIO:
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        return OpenAIModelIO(model=model, api_key=resolved_api_key or "")

    def _create_anthropic(self, *, model: str, api_key: str | None) -> ModelIO:
        resolved_api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        return AnthropicModelIO(
            model=model,
            api_key=resolved_api_key or "",
        )

    def _create_ollama(self, *, model: str, api_key: str | None) -> ModelIO:
        del api_key
        return OllamaModelIO(
            model=model,
            base_url=str(os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"),
        )

    def _create_hyperspace(self, *, model: str, api_key: str | None) -> ModelIO:
        resolved_api_key = api_key or os.getenv("HYPERSPACE_API_KEY")
        base_url = os.getenv("HYPERSPACE_BASE_URL")
        kwargs: dict[str, Any] = {
            "model": model,
            "api_key": resolved_api_key or "",
        }
        if base_url:
            kwargs["base_url"] = base_url
        return HyperspaceModelIO(**kwargs)

    def _create_gemini(self, *, model: str, api_key: str | None) -> ModelIO:
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return GeminiModelIO(model=model, api_key=key or "")
