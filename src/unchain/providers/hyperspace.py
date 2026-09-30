"""SAP Hyperspace provider — Anthropic-protocol compatible.

SAP Hyperspace exposes an Anthropic Messages API at a custom base URL with
``x-api-key`` authentication. Since the wire protocol is identical to
Anthropic's, ``HyperspaceModelIO`` subclasses ``AnthropicModelIO`` and only
overrides the default ``client_factory`` to point at the Hyperspace base URL.
The two official Moonshot endpoints have a narrowly bound K2.7 Code replay
profile; all other routes retain native Anthropic signature requirements.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx

from .anthropic import AnthropicModelIO
from .replay_profile import kimi_replay_profile

DEFAULT_HYPERSPACE_BASE_URL = "http://localhost:6655/anthropic"


def _default_hyperspace_client_factory(
    base_url: str,
) -> Callable[..., Any]:
    def _factory(api_key: str, **kwargs: Any) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for hyperspace provider — pip install anthropic"
            ) from exc
        return anthropic.Anthropic(api_key=api_key, base_url=base_url, **kwargs)

    return _factory


class HyperspaceModelIO(AnthropicModelIO):
    """SAP Hyperspace adapter.

    Hyperspace speaks the Anthropic Messages API, so all parsing logic
    (thinking blocks, tool_use, prompt caching, token usage) is inherited
    from ``AnthropicModelIO``. Kimi K2.7 Code at the official Moonshot endpoints
    additionally uses a route-bound unsigned-thinking replay profile.
    """

    provider = "hyperspace"

    @property
    def provider_replay_profile(self) -> dict[str, str] | None:
        return kimi_replay_profile(endpoint=self.base_url, model=self.model)

    _ANTHROPIC_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=30.0, pool=10.0)

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_HYPERSPACE_BASE_URL,
        client_factory: Callable[..., Any] | None = None,
        default_payloads: dict[str, dict[str, Any]] | None = None,
        model_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("HyperspaceModelIO requires a non-empty api_key")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("HyperspaceModelIO requires a non-empty base_url")
        self.base_url = base_url.strip()
        if client_factory is None:
            client_factory = _default_hyperspace_client_factory(self.base_url)
        super().__init__(
            model=model,
            api_key=api_key,
            client_factory=client_factory,
            default_payloads=default_payloads,
            model_capabilities=model_capabilities,
        )
