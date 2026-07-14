from __future__ import annotations

import copy
import json
from typing import Any, Protocol, runtime_checkable

from ..kernel.types import ToolCall


# ── rich tool result: content_blocks vocabulary ─────────────────────────────
#
# A tool result dict MAY carry a reserved ``content_blocks`` key. Its absence
# means the legacy behavior is preserved byte-for-byte (zero migration). When
# present, provider builders surface the blocks in each provider's native
# shape. The block vocabulary is append-only; today only ``text`` and ``image``
# are defined. Image blocks are flat: ``{"type":"image","media_type":...,
# "data_b64":...,"width":...,"height":...}`` (width/height optional).

CONTENT_BLOCKS_KEY = "content_blocks"


def _content_blocks(tool_result: Any) -> list[dict] | None:
    """Return the usable content blocks, or None to signal legacy behavior.

    None is returned when the reserved key is absent OR present-but-unusable
    (empty / not a list / no dict blocks), so callers fall back to the exact
    pre-existing behavior in every non-opt-in case.
    """
    if not isinstance(tool_result, dict):
        return None
    raw = tool_result.get(CONTENT_BLOCKS_KEY)
    if not isinstance(raw, list) or not raw:
        return None
    blocks = [block for block in raw if isinstance(block, dict) and block.get("type")]
    return blocks or None


def _image_label(block: dict) -> str:
    """Human placeholder for providers that cannot carry inline image bytes."""
    media_type = block.get("media_type")
    fmt = "image"
    if isinstance(media_type, str) and "/" in media_type:
        fmt = media_type.split("/", 1)[1] or "image"
    width, height = block.get("width"), block.get("height")
    if isinstance(width, int) and isinstance(height, int):
        return f"{width}x{height} {fmt}"
    return fmt


def _text_blocks_joined(blocks: list[dict]) -> str:
    parts = [str(b.get("text") or "") for b in blocks if b.get("type") == "text"]
    return "\n".join(part for part in parts if part)


def _image_blocks(blocks: list[dict]) -> list[dict]:
    return [b for b in blocks if b.get("type") == "image"]


def iter_result_image_blocks(tool_result: Any) -> list[dict]:
    """Host-facing: return the image blocks inside a tool result (mutable refs).

    Used by the host to locate base64 image payloads on an emitted
    ``tool_result`` event before they reach the SSE boundary. Returns the live
    block dicts (not copies) so the host can strip them in place.
    """
    blocks = _content_blocks(tool_result)
    if blocks is None:
        return []
    return _image_blocks(blocks)


def redact_result_image_data(tool_result: Any, *, key: str = "data_b64") -> Any:
    """Host-facing: replace inline image bytes with a reference marker, in place.

    The host calls this in its ``tool_result`` event callback so base64 image
    data never floods the SSE stream. The block keeps ``type``/``media_type``/
    dimensions and gains ``data_omitted``/``byte_len`` so the host can wire an
    artifact id or URL in place of the raw bytes. Returns the same object.
    """
    for block in iter_result_image_blocks(tool_result):
        data = block.get(key)
        if isinstance(data, str) and data:
            block.pop(key, None)
            block["data_omitted"] = True
            block["byte_len"] = len(data)
    return tool_result


@runtime_checkable
class ProviderMessageBuilder(Protocol):
    provider: str

    def build_tool_result_messages(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> list[dict]:
        ...

    def build_tool_result_message(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> dict:
        ...


class _ProviderMessageBuilderBase:
    """Shared proxy: the legacy single-message API delegates to the list API.

    ``build_tool_result_messages`` is the contract; concrete builders override
    it. The list return shape exists so a provider can, in future, surface an
    image as its own follow-up message (e.g. OpenAI image回流).
    """

    def build_tool_result_message(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> dict:
        return self.build_tool_result_messages(
            tool_call=tool_call,
            tool_result=tool_result,
        )[0]


class OpenAIMessageBuilder(_ProviderMessageBuilderBase):
    provider = "openai"

    def build_tool_result_messages(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> list[dict]:
        blocks = _content_blocks(tool_result)
        if blocks is None:
            output = json.dumps(tool_result, default=str, ensure_ascii=False)
        else:
            # DECISION POINT (pupu-llm-expert): images are rendered as a text
            # placeholder here. The list return shape is intentionally reserved
            # so image blocks can later be emitted as a separate user message
            # carrying `input_image` blocks. Do not fold that in without an
            # llm-expert spec — it is model-visible behavior.
            parts: list[str] = []
            for block in blocks:
                if block.get("type") == "text":
                    text = str(block.get("text") or "")
                    if text:
                        parts.append(text)
                elif block.get("type") == "image":
                    parts.append(f"[image: {_image_label(block)}]")
            output = "\n".join(parts)
        return [{
            "type": "function_call_output",
            "call_id": tool_call.call_id,
            "output": output,
        }]


class AnthropicMessageBuilder(_ProviderMessageBuilderBase):
    provider = "anthropic"

    def build_tool_result_messages(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> list[dict]:
        blocks = _content_blocks(tool_result)
        if blocks is None:
            content: Any = json.dumps(tool_result, default=str, ensure_ascii=False)
        else:
            content = self._native_blocks(blocks)
        return [{
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_call.call_id,
                "content": content,
            }],
        }]

    @staticmethod
    def _native_blocks(blocks: list[dict]) -> list[dict]:
        native: list[dict] = []
        for block in blocks:
            btype = block.get("type")
            if btype == "text":
                native.append({"type": "text", "text": str(block.get("text") or "")})
            elif btype == "image":
                native.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.get("media_type") or "image/png",
                        "data": block.get("data_b64") or "",
                    },
                })
            # Unknown block types are dropped: the vocabulary is append-only, so
            # forward-compat means a newer builder handles them; emitting an
            # unknown block would make the Anthropic API reject the request.
        return native


class HyperspaceMessageBuilder(AnthropicMessageBuilder):
    """SAP Hyperspace uses the Anthropic Messages API wire format."""

    provider = "hyperspace"


class GeminiMessageBuilder(_ProviderMessageBuilderBase):
    provider = "gemini"

    def build_tool_result_messages(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> list[dict]:
        blocks = _content_blocks(tool_result)
        if blocks is None:
            return [{
                "role": "user",
                "parts": [{
                    "function_response": {
                        "name": tool_call.name,
                        "response": dict(tool_result),
                    },
                }],
            }]
        parts: list[dict] = [{
            "function_response": {
                "name": tool_call.name,
                "response": {"content": _text_blocks_joined(blocks)},
            },
        }]
        for block in _image_blocks(blocks):
            parts.append({
                "inline_data": {
                    "mime_type": block.get("media_type") or "image/png",
                    "data": block.get("data_b64") or "",
                },
            })
        return [{"role": "user", "parts": parts}]


class OllamaMessageBuilder(_ProviderMessageBuilderBase):
    provider = "ollama"

    def build_tool_result_messages(
        self,
        *,
        tool_call: ToolCall,
        tool_result: dict,
    ) -> list[dict]:
        blocks = _content_blocks(tool_result)
        if blocks is None:
            content = json.dumps(tool_result, default=str, ensure_ascii=False)
        else:
            parts: list[str] = []
            for block in blocks:
                if block.get("type") == "text":
                    text = str(block.get("text") or "")
                    if text:
                        parts.append(text)
                elif block.get("type") == "image":
                    parts.append(f"[image omitted: {_image_label(block)}]")
            content = "\n".join(parts)
        return [{
            "role": "tool",
            "tool_call_id": tool_call.call_id,
            "content": content,
        }]


def get_provider_message_builder(provider: str) -> ProviderMessageBuilder:
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        return OpenAIMessageBuilder()
    if normalized == "anthropic":
        return AnthropicMessageBuilder()
    if normalized == "hyperspace":
        return HyperspaceMessageBuilder()
    if normalized == "gemini":
        return GeminiMessageBuilder()
    if normalized == "ollama":
        return OllamaMessageBuilder()
    raise NotImplementedError(f"provider message builder is not implemented for provider={provider!r}")


def coalesce_provider_tool_result_messages(
    provider: str,
    messages: list[dict],
) -> list[dict]:
    if str(provider or "").strip().lower() not in {"anthropic", "hyperspace"}:
        return copy.deepcopy(messages)

    coalesced: list[dict] = []
    pending_blocks: list[dict] = []

    def flush_pending() -> None:
        if pending_blocks:
            coalesced.append(
                {
                    "role": "user",
                    "content": copy.deepcopy(pending_blocks),
                }
            )
            pending_blocks.clear()

    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        is_tool_result_message = (
            isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(content, list)
            and bool(content)
            and all(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for block in content
            )
        )
        if is_tool_result_message:
            pending_blocks.extend(copy.deepcopy(content))
            continue
        flush_pending()
        if isinstance(message, dict):
            coalesced.append(copy.deepcopy(message))
    flush_pending()
    return coalesced


__all__ = [
    "AnthropicMessageBuilder",
    "CONTENT_BLOCKS_KEY",
    "GeminiMessageBuilder",
    "HyperspaceMessageBuilder",
    "OllamaMessageBuilder",
    "OpenAIMessageBuilder",
    "ProviderMessageBuilder",
    "coalesce_provider_tool_result_messages",
    "get_provider_message_builder",
    "iter_result_image_blocks",
    "redact_result_image_data",
]
