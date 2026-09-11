from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

from .types import ModelSpec, coerce_model_spec


_PROVIDER_ENV_DEFAULTS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "ollama": tuple(),
    "hyperspace": ("HYPERSPACE_API_KEY",),
}
_API_KEY_OPTIONAL_PROVIDERS = frozenset({"ollama"})


def load_root_env(repo_root: str | Path) -> dict[str, str]:
    env_path = Path(repo_root).resolve() / ".env"
    if not env_path.exists():
        return {}
    load_dotenv(env_path, override=False)
    loaded = dotenv_values(env_path)
    return {key: value for key, value in loaded.items() if value is not None}


def get_provider_api_env_names(provider: str) -> tuple[str, ...]:
    return _PROVIDER_ENV_DEFAULTS.get(str(provider or "").strip().lower(), tuple())


def is_supported_provider(provider: str) -> bool:
    return str(provider or "").strip().lower() in _PROVIDER_ENV_DEFAULTS


def provider_requires_api_key(provider: str) -> bool:
    normalized = str(provider or "").strip().lower()
    return normalized in _PROVIDER_ENV_DEFAULTS and normalized not in _API_KEY_OPTIONAL_PROVIDERS


def get_model_spec_skip_reason(
    model_spec: ModelSpec | dict[str, Any],
    *,
    api_key: str | None,
) -> str | None:
    spec = coerce_model_spec(model_spec)
    if not is_supported_provider(spec.provider):
        supported = ", ".join(sorted(_PROVIDER_ENV_DEFAULTS))
        return f"unsupported provider {spec.provider!r}; supported providers: {supported}"
    if api_key or not provider_requires_api_key(spec.provider):
        return None
    expected = spec.api_key_env or "/".join(get_provider_api_env_names(spec.provider))
    return f"missing API key for {spec.label}; expected {expected}"


def resolve_api_key(
    model_spec: ModelSpec | dict[str, Any],
    *,
    env: dict[str, str] | None = None,
) -> tuple[str | None, str | None]:
    spec = coerce_model_spec(model_spec)
    source = os.environ if env is None else env
    if spec.api_key_env:
        value = str(source.get(spec.api_key_env, "")).strip()
        return (value or None), spec.api_key_env
    for env_name in get_provider_api_env_names(spec.provider):
        value = str(source.get(env_name, "")).strip()
        if value:
            return value, env_name
    return None, None


def filter_model_specs(
    model_specs: list[ModelSpec | dict[str, Any]],
    *,
    env: dict[str, str] | None = None,
) -> tuple[list[tuple[ModelSpec, str | None, str | None]], list[dict[str, Any]]]:
    source = os.environ if env is None else env
    ready: list[tuple[ModelSpec, str | None, str | None]] = []
    skipped: list[dict[str, Any]] = []

    for raw_spec in model_specs:
        spec = coerce_model_spec(raw_spec)
        api_key, resolved_env_name = resolve_api_key(spec, env=source)
        skip_reason = get_model_spec_skip_reason(spec, api_key=api_key)
        if skip_reason is None:
            ready.append((spec, api_key, resolved_env_name))
            continue
        skipped.append(
            {
                "provider": spec.provider,
                "model": spec.model,
                "label": spec.label,
                "api_key_env": spec.api_key_env,
                "reason": skip_reason,
            }
        )

    return ready, skipped
