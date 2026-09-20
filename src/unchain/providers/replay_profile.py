"""Narrow provider-specific replay contracts, independent of user payloads."""

from __future__ import annotations

from typing import Any

from ..kernel.provider_replay import ProviderReplayFrameError


_KIMI_ENDPOINTS = frozenset({
    "https://api.moonshot.ai/anthropic",
    "https://api.moonshot.cn/anthropic",
})
_KIMI_MODEL = "kimi-k2.7-code"
_PROFILE = "kimi.unsigned-thinking.v1"


def kimi_replay_profile(*, endpoint: str, model: str) -> dict[str, str] | None:
    endpoint = endpoint.rstrip("/")
    if endpoint not in _KIMI_ENDPOINTS or model != _KIMI_MODEL:
        return None
    return {"profile": _PROFILE, "endpoint": endpoint, "model": model}


def validate_replay_profile(
    value: Any, *, active: Any, provider: str, model: str | None,
) -> bool:
    """Return whether unsigned thinking is admitted for this exact live route."""
    if value is None:
        return False
    if (
        not isinstance(value, dict)
        or set(value) != {"profile", "endpoint", "model"}
        or not all(isinstance(item, str) for item in value.values())
        or value.get("profile") != _PROFILE
        or value.get("endpoint") not in _KIMI_ENDPOINTS
        or value.get("model") != _KIMI_MODEL
        or value != active
        or provider != "hyperspace"
        or model != value["model"]
    ):
        raise ProviderReplayFrameError("provider replay profile does not match the active route")
    return True
