from __future__ import annotations

import copy
import json
from typing import Any, Callable

import httpx

from .base import ModelTurnRequest
from .native import _NativeModelIOBase, _translate_content_blocks_for_anthropic
from ..kernel.provider_replay import (
    ProviderReplayFrameError,
    redact_provider_replay_secrets,
    strict_json_copy,
    tool_schema_digest,
    tool_schema_manifest,
)
from ..kernel.types import ModelTurnResult, ToolCall


class AnthropicModelIO(_NativeModelIOBase):
    """Native Anthropic Messages API adapter for the new kernel."""

    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        client_factory: Callable[..., Any] | None = None,
        default_payloads: dict[str, dict[str, Any]] | None = None,
        model_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("AnthropicModelIO requires a non-empty api_key")
        super().__init__(
            model=model,
            default_payloads=default_payloads,
            model_capabilities=model_capabilities,
        )
        self.api_key = api_key
        if client_factory is None:
            try:
                import anthropic
                client_factory = anthropic.Anthropic
            except ImportError:
                raise ImportError("anthropic package is required for anthropic provider — pip install anthropic")
        self._client_factory = client_factory

    _ANTHROPIC_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)

    def _collect_required_betas(self, toolkit: Any) -> list[str]:
        collector = getattr(toolkit, "required_betas", None)
        if not callable(collector):
            return []
        betas = collector(self.provider)
        return [beta for beta in betas if isinstance(beta, str) and beta] if isinstance(betas, list) else []

    def fetch_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        client = self._client_factory(api_key=self.api_key, timeout=self._ANTHROPIC_TIMEOUT)
        request_payload = self._merged_payload(request.payload)

        system_parts: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for message in request.messages:
            if isinstance(message, dict) and message.get("role") in {"system", "developer"}:
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    system_parts.append(content.strip())
                elif content not in (None, ""):
                    system_parts.append(str(content))
                continue
            chat_messages.append(copy.deepcopy(message))

        _translate_content_blocks_for_anthropic(chat_messages)

        if request.response_format is not None:
            system_parts.append(request.response_format.to_anthropic())
        system_prompt = "\n\n".join(part for part in system_parts if isinstance(part, str) and part.strip())

        tools_json = request.toolkit.to_provider_json(self.provider)
        anthropic_tools: list[dict[str, Any]] = []
        if tools_json and self._model_capability("supports_tools", True):
            anthropic_tools = copy.deepcopy(tools_json)

        _default_max = self._model_capability("max_output_tokens", 4096)
        max_tokens = request_payload.pop("max_tokens", _default_max)
        request_kwargs: dict[str, Any] = {
            "model": self._provider_request_model(),
            "messages": chat_messages,
            "max_tokens": max_tokens,
            **request_payload,
        }
        if system_prompt:
            request_kwargs["system"] = [
                {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
            ]
        if anthropic_tools:
            anthropic_tools[-1]["cache_control"] = {"type": "ephemeral"}
            request_kwargs["tools"] = anthropic_tools

        # Provider-native predefined tools (e.g. `computer`) may require an
        # Anthropic beta flag. Aggregate the betas declared by the exposed tools
        # and send them via the `anthropic-beta` request header.
        required_betas = self._collect_required_betas(request.toolkit)
        if required_betas:
            extra_headers = dict(request_kwargs.get("extra_headers") or {})
            extra_headers["anthropic-beta"] = ",".join(required_betas)
            request_kwargs["extra_headers"] = extra_headers
        # Annotate last message for prompt caching.
        if chat_messages:
            _last = chat_messages[-1]
            _content = _last.get("content")
            if isinstance(_content, str):
                _last["content"] = [{"type": "text", "text": _content, "cache_control": {"type": "ephemeral"}}]
            elif isinstance(_content, list) and _content:
                _block = _content[-1]
                if isinstance(_block, dict):
                    _block["cache_control"] = {"type": "ephemeral"}

        self._emit_request_messages(
            callback=request.callback,
            run_id=request.run_id,
            iteration=request.iteration,
            messages=redact_provider_replay_secrets(chat_messages),
            tool_names=self._tool_names_for_trace(tools_json),
            system=system_prompt if system_prompt else None,
        )

        if not chat_messages:
            raise ValueError(
                "Anthropic request has no chat messages after preprocessing. "
                "This usually means context optimization or memory/history "
                "selection dropped the active turn before provider call."
            )

        collected_chunks: list[str] = []
        input_tokens = 0
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0
        blocks_by_index: dict[int, dict[str, Any]] = {}
        tool_json_parts: dict[int, list[str]] = {}
        active_index: int | None = None
        next_fallback_index = 0
        final_message = None

        def resolve_index(event: Any, *, block_type: str) -> int:
            nonlocal next_fallback_index, active_index
            raw_index = getattr(event, "index", None)
            if isinstance(raw_index, int):
                next_fallback_index = max(next_fallback_index, raw_index + 1)
                return raw_index
            if active_index is not None:
                active = blocks_by_index.get(active_index)
                if isinstance(active, dict) and active.get("type") == block_type:
                    return active_index
            index = next_fallback_index
            next_fallback_index += 1
            return index

        def ensure_block(index: int, block_type: str) -> dict[str, Any]:
            block = blocks_by_index.get(index)
            if not isinstance(block, dict):
                block = {"type": block_type}
                blocks_by_index[index] = block
            if block.get("type") != block_type:
                raise ProviderReplayFrameError(
                    "Anthropic stream reused a content block index for a different block type"
                )
            return block

        with client.messages.stream(**request_kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == "content_block_start":
                    block_dict = self._as_dict(getattr(event, "content_block", None))
                    block_type = str(block_dict.get("type") or "")
                    if not block_type:
                        raise ProviderReplayFrameError(
                            "Anthropic content_block_start is missing block type"
                        )
                    index = resolve_index(event, block_type=block_type)
                    if index in blocks_by_index:
                        raise ProviderReplayFrameError(
                            "Anthropic stream emitted duplicate content_block_start index"
                        )
                    block = strict_json_copy(block_dict)
                    if block_type == "thinking":
                        block.setdefault("thinking", "")
                        block.setdefault("signature", "")
                    elif block_type == "text":
                        block.setdefault("text", "")
                    elif block_type == "tool_use":
                        tool_json_parts[index] = []
                    blocks_by_index[index] = block
                    active_index = index
                    continue

                if event_type == "content_block_delta":
                    delta_dict = self._as_dict(getattr(event, "delta", None))
                    delta_type = delta_dict.get("type", "")
                    if delta_type == "thinking_delta":
                        index = resolve_index(event, block_type="thinking")
                        block = ensure_block(index, "thinking")
                        active_index = index
                        thinking_text = delta_dict.get("thinking", "") or ""
                        if thinking_text:
                            block["thinking"] = str(block.get("thinking") or "") + str(
                                thinking_text
                            )
                            if request.emit_stream:
                                self._emit(
                                    request.callback,
                                    "reasoning",
                                    request.run_id,
                                    iteration=request.iteration,
                                    provider=self.provider,
                                    delta=thinking_text,
                                )
                        continue
                    if delta_type == "signature_delta":
                        index = resolve_index(event, block_type="thinking")
                        block = ensure_block(index, "thinking")
                        active_index = index
                        signature = delta_dict.get("signature", "") or ""
                        block["signature"] = str(block.get("signature") or "") + str(
                            signature
                        )
                        continue
                    if delta_type == "text_delta":
                        index = resolve_index(event, block_type="text")
                        block = ensure_block(index, "text")
                        active_index = index
                        text = delta_dict.get("text", "") or ""
                        if text:
                            block["text"] = str(block.get("text") or "") + str(text)
                            collected_chunks.append(text)
                            if request.emit_stream:
                                self._emit(
                                    request.callback,
                                    "token_delta",
                                    request.run_id,
                                    iteration=request.iteration,
                                    provider=self.provider,
                                    delta=text,
                                    accumulated_text="".join(collected_chunks),
                                )
                        continue
                    if delta_type == "input_json_delta":
                        index = resolve_index(event, block_type="tool_use")
                        ensure_block(index, "tool_use")
                        active_index = index
                        partial = delta_dict.get("partial_json", "") or ""
                        if partial:
                            tool_json_parts.setdefault(index, []).append(str(partial))
                    continue

                if event_type == "content_block_stop":
                    raw_index = getattr(event, "index", None)
                    index = raw_index if isinstance(raw_index, int) else active_index
                    if index is None or index not in blocks_by_index:
                        raise ProviderReplayFrameError(
                            "Anthropic content_block_stop has no matching block"
                        )
                    block = blocks_by_index[index]
                    if block.get("type") == "tool_use":
                        raw_json = "".join(tool_json_parts.get(index, []))
                        if raw_json.strip():
                            try:
                                arguments = json.loads(raw_json)
                            except json.JSONDecodeError as exc:
                                raise ProviderReplayFrameError(
                                    "Anthropic tool_use input JSON was truncated or invalid"
                                ) from exc
                            if not isinstance(arguments, dict):
                                raise ProviderReplayFrameError(
                                    "Anthropic tool_use input must decode to an object"
                                )
                            block["input"] = arguments
                        elif not isinstance(block.get("input"), dict):
                            block["input"] = {}
                    active_index = None
                    continue

                if event_type == "message_delta":
                    usage_dict = self._as_dict(getattr(event, "usage", None))
                    if usage_dict:
                        input_tokens = max(input_tokens, self._coerce_token_count(usage_dict.get("input_tokens")))
                        output_tokens = max(output_tokens, self._coerce_token_count(usage_dict.get("output_tokens")))
                        cache_read_input_tokens = max(cache_read_input_tokens, self._coerce_token_count(usage_dict.get("cache_read_input_tokens")))
                        cache_creation_input_tokens = max(cache_creation_input_tokens, self._coerce_token_count(usage_dict.get("cache_creation_input_tokens")))
                    continue

                if event_type == "message_start":
                    msg_dict = self._as_dict(getattr(event, "message", None))
                    usage_dict = msg_dict.get("usage", {}) if isinstance(msg_dict, dict) else {}
                    if isinstance(usage_dict, dict):
                        input_tokens = max(input_tokens, self._coerce_token_count(usage_dict.get("input_tokens")))
                        output_tokens = max(output_tokens, self._coerce_token_count(usage_dict.get("output_tokens")))
                        cache_read_input_tokens = max(cache_read_input_tokens, self._coerce_token_count(usage_dict.get("cache_read_input_tokens")))
                        cache_creation_input_tokens = max(cache_creation_input_tokens, self._coerce_token_count(usage_dict.get("cache_creation_input_tokens")))
                    continue

            get_final_message = getattr(stream, "get_final_message", None)
            if callable(get_final_message):
                final_message = get_final_message()

        raw_blocks: list[dict[str, Any]] = []
        if final_message is not None:
            final_dict = self._as_dict(final_message)
            raw_content = final_dict.get("content")
            if isinstance(raw_content, list):
                raw_blocks = strict_json_copy(
                    [self._as_dict(block) for block in raw_content]
                )
        if not raw_blocks:
            raw_blocks = strict_json_copy(
                [blocks_by_index[index] for index in sorted(blocks_by_index)]
            )

        tool_calls: list[ToolCall] = []
        reasoning_items: list[dict[str, Any]] = []
        semantic_blocks: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for block in raw_blocks:
            block_type = block.get("type")
            if block_type == "thinking":
                thinking_text = str(block.get("thinking") or "")
                reasoning_items.append({"type": "thinking", "text": thinking_text})
                continue
            if block_type == "redacted_thinking":
                if not isinstance(block.get("data"), str) or not block.get("data"):
                    raise ProviderReplayFrameError(
                        "Anthropic redacted_thinking block is missing data"
                    )
                continue
            if block_type == "text":
                text = str(block.get("text") or "")
                if text:
                    text_parts.append(text)
                    semantic_blocks.append({"type": "text", "text": text})
                continue
            if block_type == "tool_use":
                call_id = block.get("id")
                name = block.get("name")
                arguments = block.get("input")
                if not isinstance(call_id, str) or not call_id:
                    raise ProviderReplayFrameError(
                        "Anthropic tool_use block is missing id"
                    )
                if not isinstance(name, str) or not name:
                    raise ProviderReplayFrameError(
                        "Anthropic tool_use block is missing name"
                    )
                if not isinstance(arguments, dict):
                    raise ProviderReplayFrameError(
                        "Anthropic tool_use input must be an object"
                    )
                tool_calls.append(
                    ToolCall(
                        call_id=call_id,
                        name=name,
                        arguments=copy.deepcopy(arguments),
                    )
                )
                semantic_blocks.append(copy.deepcopy(block))

        if tool_calls:
            for block in raw_blocks:
                if block.get("type") == "thinking":
                    if not isinstance(block.get("signature"), str) or not block.get(
                        "signature"
                    ):
                        raise ProviderReplayFrameError(
                            "Anthropic thinking block is missing the replay signature"
                        )

        full_text = "".join(text_parts).strip()
        assistant_messages: list[dict[str, Any]] = []
        if tool_calls:
            assistant_messages.append(
                {"role": "assistant", "content": semantic_blocks}
            )
        elif full_text:
            assistant_messages.append({"role": "assistant", "content": full_text})

        raw_assistant_message = {
            "role": "assistant",
            "content": raw_blocks,
        }
        provider_replay_frame = {
            "format": "anthropic.messages.v1",
            "complete": True,
            "items": strict_json_copy(
                [*copy.deepcopy(request.messages), raw_assistant_message]
            ),
            "mode": "replace",
            "source": "anthropic_final_message_content",
            "tool_schema_digest": tool_schema_digest(
                request.toolkit,
                self.provider,
            ),
            "tool_schema_manifest": tool_schema_manifest(
                request.toolkit,
                self.provider,
            ),
        }

        # Coerce input/output to ints via _normalize_token_usage (consumed_tokens
        # from it is intentionally ignored -- see total_consumed below).
        token_usage = self._normalize_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        # Anthropic's input_tokens excludes cached tokens, so we add
        # cache_read + cache_creation to get the true total processed.
        total_consumed = (
            input_tokens
            + cache_read_input_tokens
            + cache_creation_input_tokens
            + output_tokens
        )
        return ModelTurnResult(
            assistant_messages=assistant_messages,
            tool_calls=tool_calls,
            final_text=full_text,
            response_id=None,
            reasoning_items=reasoning_items or None,
            consumed_tokens=total_consumed,
            input_tokens=token_usage.input_tokens,
            output_tokens=token_usage.output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            provider_replay_frame=provider_replay_frame,
        )


__all__ = ["AnthropicModelIO"]
