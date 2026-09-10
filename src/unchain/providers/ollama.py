from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Callable

import httpx

from .base import ModelTurnRequest
from .native import _NativeModelIOBase, _translate_content_blocks_for_ollama
from ..kernel.provider_replay import (
    redact_provider_replay_secrets,
    strict_json_copy,
    tool_schema_digest,
    tool_schema_manifest,
)
from ..kernel.types import ModelTurnResult, ToolCall
from ..run_bundle import ProviderCallUsage
from .canonical_hash import canonical_json_sha256


class OllamaModelIO(_NativeModelIOBase):
    """Native Ollama chat API adapter for the new kernel."""

    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        stream_factory: Callable[..., Any] | None = None,
        default_payloads: dict[str, dict[str, Any]] | None = None,
        model_capabilities: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(
            model=model,
            default_payloads=default_payloads,
            model_capabilities=model_capabilities,
        )
        self.base_url = str(base_url or "http://localhost:11434").rstrip("/")
        if stream_factory is None:
            stream_factory = httpx.stream
        self._stream_factory = stream_factory

    def _merged_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        merged = super()._merged_payload(payload)
        # num_ctx is a native Ollama option even when the local model has no
        # catalog entry. Keep the host's compiler window on the provider wire.
        if payload is not None and "num_ctx" in payload:
            window = payload["num_ctx"]
            if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
                raise ValueError("Ollama num_ctx must be a positive integer")
            merged["num_ctx"] = window
        return merged

    def fetch_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        messages = copy.deepcopy(request.messages)
        _translate_content_blocks_for_ollama(messages)
        request_body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        tools_json = request.toolkit.to_provider_json(self.provider)
        tools: list[dict[str, Any]] = []
        if tools_json and self._model_capability("supports_tools", True):
            tools = copy.deepcopy(tools_json)

        if tools:
            request_body["tools"] = tools
            request_body["tool_choice"] = "auto"

        merged_payload = self._merged_payload(request.payload)
        if merged_payload:
            request_body["options"] = merged_payload

        if request.response_format is not None:
            request_body["format"] = request.response_format.to_ollama()

        self._emit_request_messages(
            callback=request.callback,
            run_id=request.run_id,
            iteration=request.iteration,
            messages=redact_provider_replay_secrets(
                request_body.get("messages", [])
            ),
            tool_names=self._tool_names_for_trace(tools),
        )

        return self._fetch_turn_streaming(request_body, request)

    def _fetch_turn_streaming(
        self,
        request_body: dict[str, Any],
        request: ModelTurnRequest,
    ) -> ModelTurnResult:
        collected_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        latest_prompt_eval_count = 0
        latest_eval_count = 0
        observed_usage: dict[str, Any] = {}

        with self._stream_factory(
            "POST",
            f"{self.base_url}/api/chat",
            json=request_body,
            timeout=None,
        ) as response:
            if int(getattr(response, "status_code", 0) or 0) >= 400:
                raw_detail = response.read()
                if isinstance(raw_detail, bytes):
                    detail = raw_detail.decode()
                else:
                    detail = str(raw_detail)
                raise ValueError(f"error: {detail} ( kernel.model_io -> OllamaModelIO.fetch_turn )")
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode()

                data = json.loads(line)
                if data.get("error"):
                    raise ValueError(f"error: {data['error']} ( kernel.model_io -> OllamaModelIO.fetch_turn )")
                if isinstance(data.get("prompt_eval_count"), int):
                    latest_prompt_eval_count = data["prompt_eval_count"]
                    observed_usage["prompt_eval_count"] = data["prompt_eval_count"]
                if isinstance(data.get("eval_count"), int):
                    latest_eval_count = data["eval_count"]
                    observed_usage["eval_count"] = data["eval_count"]
                for reasoning_count_key in (
                    "thinking_eval_count",
                    "reasoning_eval_count",
                ):
                    if reasoning_count_key in data:
                        observed_usage[reasoning_count_key] = data[
                            reasoning_count_key
                        ]

                message = data.get("message") or {}
                content_delta = message.get("content", "") or ""
                thinking_delta = message.get("thinking", "") or ""

                if thinking_delta:
                    reasoning_chunks.append(thinking_delta)

                if content_delta:
                    collected_chunks.append(content_delta)
                    if request.emit_stream:
                        self._emit(
                            request.callback,
                            "token_delta",
                            request.run_id,
                            iteration=request.iteration,
                            provider=self.provider,
                            delta=content_delta,
                            accumulated_text="".join(collected_chunks),
                        )

                raw_tool_calls = message.get("tool_calls") or []
                if raw_tool_calls:
                    normalized_tool_calls: list[dict[str, Any]] = []
                    tool_calls: list[ToolCall] = []
                    for raw_tool_call in raw_tool_calls:
                        normalized_call = copy.deepcopy(raw_tool_call)
                        call_id = str(normalized_call.get("id") or str(uuid.uuid4()))
                        normalized_call["id"] = call_id
                        normalized_tool_calls.append(normalized_call)
                        fn = normalized_call.get("function", {}) or {}
                        tool_calls.append(
                            ToolCall(
                                call_id=call_id,
                                name=str(fn.get("name", "") or ""),
                                arguments=copy.deepcopy(fn.get("arguments", {})),
                            )
                        )
                    assistant_message = {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": normalized_tool_calls,
                    }
                    if reasoning_chunks:
                        assistant_message["thinking"] = "".join(reasoning_chunks)

                    provider_replay_frame = {
                        "format": "ollama.chat.v1",
                        "complete": True,
                        "items": strict_json_copy(
                            [
                                *copy.deepcopy(request_body["messages"]),
                                assistant_message,
                            ]
                        ),
                        "mode": "replace",
                        "source": "ollama_chat_message",
                        "tool_schema_digest": tool_schema_digest(
                            request.toolkit,
                            self.provider,
                        ),
                        "tool_schema_manifest": tool_schema_manifest(
                            request.toolkit,
                            self.provider,
                        ),
                    }

                    return ModelTurnResult(
                        assistant_messages=[
                            {
                                "role": "assistant",
                                "content": assistant_message.get("content", ""),
                                "tool_calls": copy.deepcopy(normalized_tool_calls),
                            }
                        ],
                        tool_calls=tool_calls,
                        final_text="",
                        response_id=None,
                        reasoning_items=[{"type": "thinking", "text": "".join(reasoning_chunks)}] if reasoning_chunks else None,
                        consumed_tokens=latest_prompt_eval_count + latest_eval_count,
                        input_tokens=latest_prompt_eval_count,
                        output_tokens=latest_eval_count,
                        provider_replay_frame=provider_replay_frame,
                        provider_call_usage=ProviderCallUsage.from_ollama_usage(
                            observed_usage,
                            reasoning_present=bool(reasoning_chunks),
                        ),
                        provider_raw_usage_sha256=(
                            canonical_json_sha256(observed_usage)
                            if observed_usage
                            else None
                        ),
                    )

                if data.get("done", False):
                    full_message = message.get("content") or "".join(collected_chunks)
                    raw_assistant_message = {
                        "role": "assistant",
                        "content": full_message,
                    }
                    if reasoning_chunks:
                        raw_assistant_message["thinking"] = "".join(reasoning_chunks)
                    return ModelTurnResult(
                        assistant_messages=[{"role": "assistant", "content": full_message}],
                        tool_calls=[],
                        final_text=full_message,
                        response_id=None,
                        reasoning_items=[{"type": "thinking", "text": "".join(reasoning_chunks)}] if reasoning_chunks else None,
                        consumed_tokens=latest_prompt_eval_count + latest_eval_count,
                        input_tokens=latest_prompt_eval_count,
                        output_tokens=latest_eval_count,
                        provider_call_usage=ProviderCallUsage.from_ollama_usage(
                            observed_usage,
                            reasoning_present=bool(reasoning_chunks),
                        ),
                        provider_raw_usage_sha256=(
                            canonical_json_sha256(observed_usage)
                            if observed_usage
                            else None
                        ),
                        provider_replay_frame={
                            "format": "ollama.chat.v1",
                            "complete": True,
                            "items": strict_json_copy(
                                [
                                    *copy.deepcopy(request_body["messages"]),
                                    raw_assistant_message,
                                ]
                            ),
                            "mode": "replace",
                            "source": "ollama_chat_message",
                            "tool_schema_digest": tool_schema_digest(
                                request.toolkit,
                                self.provider,
                            ),
                            "tool_schema_manifest": tool_schema_manifest(
                                request.toolkit,
                                self.provider,
                            ),
                        },
                    )

        raise ValueError("error: unexpected termination of ollama stream. ( kernel.model_io -> OllamaModelIO.fetch_turn )")


__all__ = ["OllamaModelIO"]
