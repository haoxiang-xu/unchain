"""Model-specific admission for Gemini's synchronous streaming subset."""
from __future__ import annotations

from typing import Any

ARGON_MODEL_ID = "gemini-4-argon"


def validate_gemini_model_config(model: str, config: dict[str, Any]) -> None:
    """Reject unsupported Argon settings without rewriting frozen wire data.

    Root payload projection happens through the catalog allowlist. Check the
    resulting config again here so a prepared/replayed request cannot opt into
    explicit caching, sampling or automatic long-decode continuation.
    """
    if model != ARGON_MODEL_ID:
        return
    if type(config) is not dict or set(config) - {
        "max_output_tokens", "thinking_config", "system_instruction",
        "automatic_function_calling", "tools", "response_mime_type", "response_schema",
    }:
        raise ValueError("Argon config contains unsupported fields")
    maximum = config.get("max_output_tokens", 65_536)
    if type(maximum) is not int or not 1 <= maximum <= 65_536:
        raise ValueError("Argon synchronous max_output_tokens must be an integer from 1 to 65536")
    thinking = config.get("thinking_config", {})
    if type(thinking) is not dict or set(thinking) - {"thinking_level", "include_thoughts"}:
        raise ValueError("Argon thinking_config only supports thinking_level and include_thoughts")
    level = thinking.get("thinking_level", "medium")
    if not isinstance(level, str) or level.lower() not in {"low", "medium", "high"}:
        raise ValueError("Argon thinking_level must be low, medium or high")
    if "include_thoughts" in thinking and type(thinking["include_thoughts"]) is not bool:
        raise ValueError("Argon include_thoughts must be boolean")
