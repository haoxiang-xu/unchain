from __future__ import annotations

import copy
from typing import Any, Callable

from ..kernel.types import TokenUsage
from .message_contract import (
    ProviderMessageContractError,
    validate_anthropic_messages,
    validate_ollama_messages,
    validate_openai_input_items,
)


def _parse_base64_data_url(value: Any, *, default_media_type: str) -> tuple[str, str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw_value = value.strip()
    if not raw_value.startswith("data:"):
        return default_media_type, raw_value

    header, separator, data = raw_value.partition(",")
    if not separator or ";base64" not in header:
        return None
    media_type = header.removeprefix("data:").split(";", 1)[0] or default_media_type
    return media_type, data


def _anthropic_file_source(file_id: Any) -> dict[str, Any] | None:
    if isinstance(file_id, str) and file_id.strip():
        return {"type": "file", "file_id": file_id.strip()}
    return None


def _anthropic_url_source(url: Any) -> dict[str, Any] | None:
    if isinstance(url, str) and url.strip():
        return {"type": "url", "url": url.strip()}
    return None


def _anthropic_base64_source(
    data: Any,
    *,
    media_type: Any,
    default_media_type: str,
) -> dict[str, Any] | None:
    if not isinstance(data, str) or not data:
        return None
    resolved_media_type = (
        media_type if isinstance(media_type, str) and media_type.strip() else default_media_type
    )
    return {"type": "base64", "media_type": resolved_media_type, "data": data}


def _anthropic_source_from_canonical(
    source: Any,
    *,
    default_media_type: str,
) -> dict[str, Any] | None:
    if not isinstance(source, dict):
        return None

    source_type = source.get("type")
    if source_type == "base64":
        return _anthropic_base64_source(
            source.get("data"),
            media_type=source.get("media_type"),
            default_media_type=default_media_type,
        )
    if source_type == "url":
        return _anthropic_url_source(source.get("url"))
    if source_type in ("file", "file_id"):
        return _anthropic_file_source(source.get("file_id"))
    return None


def _anthropic_source_from_input_image(block: dict[str, Any]) -> dict[str, Any] | None:
    image_url = block.get("image_url")
    if isinstance(image_url, dict):
        image_url = image_url.get("url")

    parsed_data_url = _parse_base64_data_url(image_url, default_media_type="image/png")
    if parsed_data_url is not None:
        media_type, data = parsed_data_url
        if isinstance(image_url, str) and image_url.strip().startswith("data:"):
            return _anthropic_base64_source(
                data,
                media_type=media_type,
                default_media_type="image/png",
            )
        return _anthropic_url_source(image_url)

    return _anthropic_file_source(block.get("file_id"))


def _anthropic_source_from_input_file(block: dict[str, Any]) -> dict[str, Any] | None:
    source = _anthropic_file_source(block.get("file_id"))
    if source is not None:
        return source

    source = _anthropic_url_source(block.get("file_url"))
    if source is not None:
        return source

    parsed_file_data = _parse_base64_data_url(
        block.get("file_data"),
        default_media_type="application/pdf",
    )
    if parsed_file_data is not None:
        media_type, data = parsed_file_data
        return _anthropic_base64_source(
            data,
            media_type=media_type,
            default_media_type="application/pdf",
        )

    return _anthropic_source_from_canonical(
        block.get("source"),
        default_media_type="application/pdf",
    )


def _translate_content_blocks_for_anthropic(messages: list[dict[str, Any]]) -> None:
    """Convert unchain canonical content blocks into Anthropic-native format, in place.

    Accept both unchain canonical blocks and OpenAI Responses-style input
    blocks so callers can reuse message payloads across providers.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        new_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue

            btype = block.get("type")
            if btype == "input_text":
                new_content.append({
                    "type": "text",
                    "text": block.get("text", "") or "",
                })
                continue

            if btype == "text" and message.get("role") == "assistant":
                # Older Anthropic result artifacts may contain the SDK-only
                # ParsedTextBlock helper. This is an outbound copy after replay
                # verification, so retain the durable record and reject every
                # other unknown text field through the closed validator below.
                next_block = copy.deepcopy(block)
                next_block.pop("parsed_output", None)
                new_content.append(next_block)
                continue

            if btype == "input_image":
                source = _anthropic_source_from_input_image(block)
                if source is None:
                    new_content.append(block)
                else:
                    new_content.append({"type": "image", "source": source})
                continue

            if btype == "image":
                source = _anthropic_source_from_canonical(
                    block.get("source"),
                    default_media_type="image/png",
                )
                if source is None:
                    new_content.append(block)
                else:
                    next_block = copy.deepcopy(block)
                    next_block["type"] = "image"
                    next_block["source"] = source
                    new_content.append(next_block)
                continue

            if btype == "input_file":
                source = _anthropic_source_from_input_file(block)
                if source is None:
                    new_content.append(block)
                else:
                    new_content.append({"type": "document", "source": source})
                continue

            if btype in ("pdf", "document"):
                source = _anthropic_source_from_canonical(
                    block.get("source"),
                    default_media_type="application/pdf",
                )
                if source is None:
                    if btype == "pdf":
                        next_block = copy.deepcopy(block)
                        next_block["type"] = "document"
                        new_content.append(next_block)
                    else:
                        new_content.append(block)
                else:
                    next_block = copy.deepcopy(block)
                    next_block["type"] = "document"
                    next_block["source"] = source
                    new_content.append(next_block)
                continue

            new_content.append(block)

        message["content"] = new_content
    validate_anthropic_messages(messages)


def _translate_content_blocks_for_openai(messages: list[dict[str, Any]]) -> None:
    """Convert unchain canonical content blocks into OpenAI Responses API format, in place.

    Only user/system/developer messages are rewritten — assistant messages have
    their own output format managed elsewhere. Attachment blocks (``image`` /
    ``pdf``) are always translated. Plain ``{"type": "text", "text": ...}``
    blocks are also upgraded to ``input_text`` when they appear alongside an
    attachment, because Responses API rejects mixing ``text`` and
    ``input_image`` in the same message. Text-only messages are left alone so
    existing text chats keep working unchanged.
    """
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "system", "developer"):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        has_native_input_block = any(
            isinstance(block, dict) and block.get("type") in ("image", "pdf")
            for block in content
        )
        has_canonical_text = any(
            isinstance(block, dict) and block.get("type") == "text"
            for block in content
        )
        if not has_native_input_block and not has_canonical_text:
            continue

        new_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                new_content.append(block)
                continue
            btype = block.get("type")

            if btype == "text":
                new_content.append({
                    "type": "input_text",
                    "text": block.get("text", "") or "",
                })
                continue

            if btype == "image":
                source = block.get("source") if isinstance(block.get("source"), dict) else {}
                source_type = source.get("type")
                if source_type == "base64":
                    media_type = source.get("media_type") or "image/png"
                    data = source.get("data") or ""
                    new_content.append({
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{data}",
                    })
                elif source_type == "url":
                    url = source.get("url") or ""
                    new_content.append({"type": "input_image", "image_url": url})
                else:
                    new_content.append(block)
                continue

            if btype == "pdf":
                source = block.get("source") if isinstance(block.get("source"), dict) else {}
                source_type = source.get("type")
                if source_type == "base64":
                    media_type = source.get("media_type") or "application/pdf"
                    data = source.get("data") or ""
                    filename = source.get("filename") or "document.pdf"
                    new_content.append({
                        "type": "input_file",
                        "filename": filename,
                        "file_data": f"data:{media_type};base64,{data}",
                    })
                elif source_type == "file_id":
                    file_id = source.get("file_id") or ""
                    new_content.append({"type": "input_file", "file_id": file_id})
                elif source_type == "url":
                    # Responses API accepts remote files via file_url
                    url = source.get("url") or ""
                    new_content.append({"type": "input_file", "file_url": url})
                else:
                    new_content.append(block)
                continue

            new_content.append(block)

        message["content"] = new_content
    validate_openai_input_items(messages)


def _translate_content_blocks_for_ollama(messages: list[dict[str, Any]]) -> None:
    """Project canonical user media blocks into Ollama's text/images shape."""

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "developer":
            message["role"] = "system"
        content = message.get("content")
        if not isinstance(content, list):
            continue
        text_parts: list[str] = []
        images: list[str] = []
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                raise ProviderMessageContractError(
                    "ollama",
                    f"messages[{message_index}].content[{block_index}]",
                    "content block must be an object",
                )
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if not isinstance(text, str):
                    raise ProviderMessageContractError(
                        "ollama",
                        f"messages[{message_index}].content[{block_index}].text",
                        "must be text",
                    )
                if text:
                    text_parts.append(text)
                continue
            if block_type == "image":
                source = block.get("source")
                if not isinstance(source, dict) or source.get("type") != "base64":
                    raise ProviderMessageContractError(
                        "ollama",
                        f"messages[{message_index}].content[{block_index}].source",
                        "only base64 image input is supported",
                    )
                data = source.get("data")
                if not isinstance(data, str) or not data:
                    raise ProviderMessageContractError(
                        "ollama",
                        f"messages[{message_index}].content[{block_index}].source.data",
                        "base64 image data is required",
                    )
                images.append(data)
                continue
            raise ProviderMessageContractError(
                "ollama",
                f"messages[{message_index}].content[{block_index}]",
                f"unsupported content block type: {block_type or '<missing>'}",
            )
        message["content"] = "\n".join(text_parts)
        if images:
            if message.get("role") != "user":
                raise ProviderMessageContractError(
                    "ollama",
                    f"messages[{message_index}]",
                    "image input is only supported on user messages",
                )
            existing = message.get("images")
            if existing not in (None, []):
                raise ProviderMessageContractError(
                    "ollama",
                    f"messages[{message_index}].images",
                    "canonical media cannot be combined with preprojected images",
                )
            message["images"] = images
    validate_ollama_messages(messages)


class _NativeModelIOBase:
    provider = ""

    def __init__(
        self,
        *,
        model: str,
        default_payloads: dict[str, dict[str, Any]] | None = None,
        model_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{self.__class__.__name__} requires a non-empty model")

        self.model = model.strip()

        if default_payloads is None or model_capabilities is None:
            from ..runtime.payloads import load_default_payloads, load_model_capabilities

            if default_payloads is None:
                default_payloads = load_default_payloads()
            if model_capabilities is None:
                model_capabilities = load_model_capabilities()

        self.default_payloads = copy.deepcopy(default_payloads or {})
        self.model_capabilities = copy.deepcopy(model_capabilities or {})

    def _resolve_model_key(self, registry: dict[str, Any]) -> str | None:
        if self.model in registry:
            return self.model
        normalized_model = self.model.replace(".", "-")
        best: str | None = None
        for key in registry:
            normalized_key = str(key).replace(".", "-")
            if (
                self.model.startswith(key)
                or self.model.startswith(normalized_key)
                or normalized_model.startswith(key)
                or normalized_model.startswith(normalized_key)
                or key.startswith(self.model)
                or key.startswith(normalized_model)
                or normalized_key.startswith(self.model)
                or normalized_key.startswith(normalized_model)
            ) and (best is None or len(str(key)) > len(best)):
                best = str(key)
        return best

    def _model_capability(self, key: str, default: Any = None) -> Any:
        resolved = self._resolve_model_key(self.model_capabilities)
        model_caps = self.model_capabilities.get(resolved, {}) if resolved else {}
        if not isinstance(model_caps, dict):
            return default
        return model_caps.get(key, default)

    def _provider_request_model(self) -> str:
        resolved_model = self._model_capability("provider_model", self.model)
        if isinstance(resolved_model, str) and resolved_model.strip():
            return resolved_model.strip()
        return self.model

    def _merged_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        resolved_key = self._resolve_model_key(self.default_payloads)
        defaults = copy.deepcopy(self.default_payloads.get(resolved_key, {}) if resolved_key else {})
        if not isinstance(defaults, dict):
            return {}

        user_payload = payload or {}
        for key in list(defaults.keys()):
            if key in user_payload:
                defaults[key] = user_payload[key]

        allowed_keys = self._model_capability("allowed_payload_keys", None)
        if isinstance(allowed_keys, list) and allowed_keys:
            allowed_key_set = {key for key in allowed_keys if isinstance(key, str)}
            for key in user_payload:
                if key in allowed_key_set and key not in defaults:
                    defaults[key] = user_payload[key]
            defaults = {key: value for key, value in defaults.items() if key in allowed_key_set}

        defaults = {key: value for key, value in defaults.items() if value is not None or key in user_payload}
        return defaults

    def _coerce_token_count(self, value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def _normalize_token_usage(self, *, input_tokens: Any, output_tokens: Any) -> TokenUsage:
        resolved_input = self._coerce_token_count(input_tokens)
        resolved_output = self._coerce_token_count(output_tokens)
        return TokenUsage(
            consumed_tokens=resolved_input + resolved_output,
            input_tokens=resolved_input,
            output_tokens=resolved_output,
        )

    def _tool_names_for_trace(self, tools_json: list[dict[str, Any]] | None) -> list[str]:
        tool_names: list[str] = []
        for tool in tools_json or []:
            name = str(tool.get("name", "")).strip()
            if not name and isinstance(tool.get("function"), dict):
                name = str(tool["function"].get("name", "")).strip()
            if name:
                tool_names.append(name)
        return tool_names

    def _emit_request_messages(
        self,
        *,
        callback: Callable[[dict[str, Any]], None] | None,
        run_id: str,
        iteration: int,
        messages: list[dict[str, Any]],
        previous_response_id: str | None = None,
        tool_names: list[str] | None = None,
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "provider": self.provider,
            "messages": copy.deepcopy(messages),
        }
        if previous_response_id is not None:
            payload["previous_response_id"] = previous_response_id
        if tool_names:
            payload["tool_names"] = list(tool_names)
        payload.update(copy.deepcopy(extra))
        self._emit(callback, "request_messages", run_id, iteration=iteration, **payload)

    def _emit(
        self,
        callback: Callable[[dict[str, Any]], None] | None,
        event_type: str,
        run_id: str,
        *,
        iteration: int,
        **extra: Any,
    ) -> None:
        if callback is None:
            return
        event = {
            "type": event_type,
            "run_id": run_id,
            "iteration": iteration,
        }
        event.update(extra)
        callback(event)

    def _as_dict(self, obj: Any) -> dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return copy.deepcopy(obj)
        if hasattr(obj, "model_dump"):
            dumped = obj.model_dump()
            return copy.deepcopy(dumped) if isinstance(dumped, dict) else {}
        if hasattr(obj, "to_dict"):
            dumped = obj.to_dict()
            return copy.deepcopy(dumped) if isinstance(dumped, dict) else {}
        if hasattr(obj, "__dict__"):
            raw = {
                key: value
                for key, value in vars(obj).items()
                if not key.startswith("_")
            }
            return copy.deepcopy(raw)
        return {}


__all__ = [
    "_NativeModelIOBase",
    "_translate_content_blocks_for_anthropic",
    "_translate_content_blocks_for_openai",
]
