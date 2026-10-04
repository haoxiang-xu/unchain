"""Native Gemini streaming transport; automatic SDK tool execution is disabled."""
from __future__ import annotations

import copy
import logging
import uuid
from typing import Any, Callable

from .base import ModelTurnRequest
from .native import _NativeModelIOBase
from .gemini_schema import sanitize_gemini_schema
from .canonical_hash import canonical_json_sha256
from .failure_diagnostic import GeminiResponseFailure
from .uncertainty_diagnostic import ProviderUncertaintyDiagnostic, ProviderUncertaintyObservation
from ..kernel.provider_replay import tool_schema_digest, tool_schema_manifest
from ..kernel.types import ModelTurnResult, ToolCall
from ..run_bundle import ProviderCallUsage

_LOGGER = logging.getLogger(__name__)


def _gemini_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        raise ValueError("Gemini content must be text or a list of blocks")
    parts = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Gemini content blocks must be objects")
        kind = block.get("type")
        if kind in {"text", "input_text", "output_text"}:
            if set(block) - {"type", "text"} or not isinstance(block.get("text"), str):
                raise ValueError("Invalid Gemini text block")
            parts.append({"text": block["text"]})
        elif kind in {"image", "pdf", "document"}:
            source = block.get("source", {})
            mime = source.get("media_type") or (
                "image/png" if kind == "image" else "application/pdf"
            )
            if source.get("type") == "base64" and source.get("data"):
                parts.append(
                    {"inline_data": {"mime_type": mime, "data": source["data"]}}
                )
            elif source.get("type") == "url" and source.get("url"):
                parts.append(
                    {"file_data": {"mime_type": mime, "file_uri": source["url"]}}
                )
            else:
                raise ValueError("Unsupported Gemini media source")
        else:
            raise ValueError(f"Unsupported Gemini content block: {kind}")
    return parts


def translate_gemini_messages(
    messages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    contents, system = [], []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Gemini messages must be objects")
        role = message.get("role")
        if role in {"system", "developer"}:
            parts = _gemini_parts(message.get("content", ""))
            if any(set(part) != {"text"} for part in parts):
                raise ValueError("Gemini system instructions must be text")
            system.extend(part["text"] for part in parts)
            continue
        if role not in {"user", "assistant", "model"}:
            raise ValueError("Unsupported Gemini message role")
        if "parts" in message:
            if set(message) - {"role", "parts"}:
                raise ValueError("Unknown Gemini native message field")
            parts = copy.deepcopy(message["parts"])
        else:
            if set(message) - {"role", "content"}:
                raise ValueError("Unknown Gemini canonical message field")
            parts = _gemini_parts(message.get("content", ""))
        if not parts:
            continue
        native_role = "model" if role == "assistant" else role
        # Parallel function responses must be one user turn.
        if contents and contents[-1]["role"] == native_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": native_role, "parts": parts})
    return contents, "\n\n".join(system)


class GeminiModelIO(_NativeModelIOBase):
    provider = "gemini"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        client_factory: Callable[..., Any] | None = None,
        default_payloads=None,
        model_capabilities=None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("GeminiModelIO requires a non-empty api_key")
        super().__init__(
            model=model,
            default_payloads=default_payloads,
            model_capabilities=model_capabilities,
        )
        if client_factory is None:
            from google import genai

            client_factory = genai.Client
        self.api_key = api_key
        self._client_factory = client_factory

    def fetch_turn(self, request: ModelTurnRequest) -> ModelTurnResult:
        contents, system = translate_gemini_messages(request.messages)
        if not contents:
            raise ValueError("Gemini request requires chat messages")
        # Public callers retain the pre-send content validation contract.
        validate_gemini_contents(contents)
        config = self._merged_payload(request.payload)
        config["automatic_function_calling"] = {"disable": True}
        if system:
            config["system_instruction"] = system
        tools = request.toolkit.to_provider_json(self.provider)
        if tools and self._model_capability("supports_tools", True):
            config["tools"] = [{"function_declarations": tools}]
        if request.response_format is not None:
            config.update(request.response_format.to_gemini())
            config["response_schema"] = sanitize_gemini_schema(
                config["response_schema"]
            )
        wire = {
            "model": self._provider_request_model(),
            "contents": contents,
            "config": config,
        }
        return self._fetch_prepared(request, wire)

    def _fetch_prepared(
        self, request: ModelTurnRequest, wire: dict[str, Any]
    ) -> ModelTurnResult:
        phase = "request_preparation"
        try:
            from google.genai import types

            contents = wire["contents"]
            config = wire["config"]
            system = config.get("system_instruction", "")
            tools = request.toolkit.to_provider_json(self.provider)
            # Validate against the installed SDK before any network request. SDK
            # models forbid extra fields; JSON replay data is converted only here.
            validate_gemini_contents(contents)
            native_contents = [types.Content.model_validate(item) for item in contents]
            native_config = types.GenerateContentConfig.model_validate(config)
            self._emit_request_messages(
                callback=request.callback,
                run_id=request.run_id,
                iteration=request.iteration,
                messages=_gemini_trace_messages(contents),
                system=system,
                tool_names=self._tool_names_for_trace(tools),
            )
            raw_parts, text_parts, thoughts, tool_calls = [], [], [], []
            observed_usage = {}
            response_id = None
            finish_reason = None
            phase = "requesting"
            client = self._client_factory(
                api_key=self.api_key,
                http_options={"timeout": 120000, "retry_options": {"attempts": 1}},
            )
            stream = None
            try:
                stream = client.models.generate_content_stream(
                    model=wire["model"], contents=native_contents, config=native_config
                )
                phase = "reading_response"
                for chunk in stream:
                    phase = "processing_response"
                    data = (
                        chunk.model_dump(mode="json", exclude_none=True)
                        if hasattr(chunk, "model_dump")
                        else self._as_dict(chunk)
                    )
                    usage = data.get("usage_metadata") or data.get("usageMetadata")
                    if usage:
                        observed_usage.update(usage)
                    response_id = data.get("response_id") or response_id
                    candidates = data.get("candidates") or []
                    if not candidates:
                        if (data.get("prompt_feedback") or {}).get("block_reason"):
                            raise GeminiResponseFailure("PROMPT_BLOCKED")
                        phase = "reading_response"
                        continue
                    candidate = candidates[0]
                    reason = candidate.get("finish_reason")
                    if reason in {"STOP", "MAX_TOKENS"}:
                        if finish_reason is not None and finish_reason != reason:
                            raise ProviderUncertaintyObservation(ProviderUncertaintyDiagnostic("conflicting_finish_reason", phase))
                        finish_reason = reason
                    elif reason and reason != "FINISH_REASON_UNSPECIFIED":
                        try:
                            failure = GeminiResponseFailure(reason)
                        except ValueError:
                            raise ProviderUncertaintyObservation(ProviderUncertaintyDiagnostic("unsupported_finish_reason", phase)) from None
                        raise failure
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        part = copy.deepcopy(part)
                        raw_parts.append(part)
                        text = part.get("text")
                        if text:
                            is_thought = part.get("thought") is True
                            (thoughts if is_thought else text_parts).append(text)
                            if request.emit_stream:
                                self._emit(
                                    request.callback,
                                    "reasoning" if is_thought else "token_delta",
                                    request.run_id,
                                    iteration=request.iteration,
                                    provider=self.provider,
                                    delta=text,
                                    **(
                                        {}
                                        if is_thought
                                        else {"accumulated_text": "".join(text_parts)}
                                    ),
                                )
                        call = part.get("function_call") or part.get("functionCall")
                        if call:
                            if not call.get("name") or not isinstance(
                                call.get("args", {}), dict
                            ):
                                raise ProviderUncertaintyObservation(ProviderUncertaintyDiagnostic("malformed_tool_call", phase))
                            call_id = call.get("id") or f"gemini_{uuid.uuid4().hex}"
                            call["id"] = call_id
                            tool_calls.append(
                                ToolCall(
                                    call_id=call_id,
                                    name=call["name"],
                                    arguments=copy.deepcopy(call.get("args", {})),
                                )
                            )
                    phase = "reading_response"
            finally:
                # Resource cleanup cannot replace the provider's primary outcome.
                # Attempt both resources and never log provider bodies or secrets.
                for label, resource in (("stream", stream), ("client", client)):
                    try:
                        close = getattr(resource, "close", None)
                        if callable(close):
                            close()
                    except Exception:
                        _LOGGER.warning("Gemini %s cleanup failed", label)
            if finish_reason is None:
                raise ProviderUncertaintyObservation(ProviderUncertaintyDiagnostic("stream_incomplete", "reading_response"))
            if not text_parts and not tool_calls:
                raise GeminiResponseFailure("EMPTY_RESPONSE")

            phase = "processing_response"

            def count(snake, camel):
                return self._coerce_token_count(
                    observed_usage.get(snake, observed_usage.get(camel))
                )

            prompt = count("prompt_token_count", "promptTokenCount")
            cached = count("cached_content_token_count", "cachedContentTokenCount")
            visible = count("candidates_token_count", "candidatesTokenCount")
            thinking = count("thoughts_token_count", "thoughtsTokenCount")
            output = visible + thinking
            usage = (
                ProviderCallUsage(
                    input_uncached_tokens=prompt - cached,
                    input_cache_read_tokens=cached,
                    input_cache_write_tokens=0,
                    input_total_tokens=prompt,
                    output_visible_tokens=visible,
                    output_reasoning_tokens=thinking,
                    output_total_tokens=output,
                    total_tokens=prompt + output,
                    source="provider_observed_partial",
                )
                if observed_usage
                else None
            )
            assistant = {"role": "model", "parts": raw_parts}
            return ModelTurnResult(
                assistant_messages=[gemini_semantic_message(assistant)],
                tool_calls=tool_calls,
                final_text="".join(text_parts),
                response_id=response_id,
                reasoning_items=[{"type": "thinking", "text": t} for t in thoughts] or None,
                input_tokens=prompt,
                output_tokens=output,
                consumed_tokens=prompt + output,
                cache_read_input_tokens=cached,
                provider_call_usage=usage,
                provider_raw_usage_sha256=canonical_json_sha256(observed_usage)
                if observed_usage
                else None,
                provider_replay_frame={
                    "format": "gemini.contents.v1",
                    "complete": True,
                    "items": [*copy.deepcopy(request.messages), assistant],
                    "mode": "replace",
                    "source": "gemini_native_parts",
                    "tool_schema_digest": tool_schema_digest(
                        request.toolkit, self.provider
                    ),
                    "tool_schema_manifest": tool_schema_manifest(
                        request.toolkit, self.provider
                    ),
                },
            )

        except (GeminiResponseFailure, ProviderUncertaintyObservation):
            raise
        except BaseException as exc:
            from google.genai.errors import APIError
            # An actual rejected HTTP response retains its existing terminal
            # or bounded-retry handling. An error in HTTP 200 SSE does not.
            response = getattr(exc, "response", None)
            if isinstance(exc, APIError) and type(getattr(response, "status_code", None)) is int and response.status_code >= 400:
                raise
            raise ProviderUncertaintyObservation(
                ProviderUncertaintyDiagnostic.from_exception(exc, phase=phase),
            ) from None


def gemini_semantic_message(message: dict[str, Any]) -> dict[str, Any]:
    if "parts" not in message or message.get("role") not in {"model", "assistant"}:
        return copy.deepcopy(message)
    parts = [
        {
            k: copy.deepcopy(v)
            for k, v in p.items()
            if k not in {"thought_signature", "thoughtSignature"}
        }
        for p in message["parts"]
        if not p.get("thought")
    ]
    if not any("function_call" in p or "functionCall" in p for p in parts):
        return {
            "role": "assistant",
            "content": "".join(p.get("text", "") for p in parts),
        }
    return {"role": "model", "parts": parts}


def validate_gemini_contents(contents: list[dict[str, Any]]) -> None:
    from google.genai import types

    if not isinstance(contents, list) or not contents:
        raise ValueError("Gemini requires a non-empty contents array")
    for message in contents:
        if not isinstance(message, dict) or set(message) != {"role", "parts"}:
            raise ValueError("Gemini contents require exactly role and parts")
        if message["role"] not in {"user", "model"}:
            raise ValueError("Invalid Gemini content role")
        if not isinstance(message["parts"], list) or not message["parts"]:
            raise ValueError("Gemini content requires parts")
        types.Content.model_validate(message)


def _gemini_trace_messages(value):
    if isinstance(value, list):
        return [_gemini_trace_messages(item) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                "[redacted thought signature]"
                if key in {"thought_signature", "thoughtSignature"}
                else _gemini_trace_messages(item)
            )
            for key, item in value.items()
        }
    return value
