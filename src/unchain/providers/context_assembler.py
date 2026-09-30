from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any, Literal

from ..kernel.provider_replay import (
    ProviderReplayFrameError,
    current_provider_replay_frame,
    ensure_replay_tool_schema_compatible,
)
from ..kernel.state import RunState
from ..tools.toolkit import Toolkit
from .replay_profile import validate_replay_profile


class ProviderContextProjectionError(ProviderReplayFrameError):
    """Raised when provider-native replay cannot be safely projected."""


@dataclass(frozen=True)
class ProviderContextAssembly:
    messages: list[dict[str, Any]]
    previous_response_id: str | None = None
    fallback_messages: list[dict[str, Any]] | None = None
    mode: Literal["semantic", "local_replay", "remote_continuation"] = "semantic"


@dataclass(frozen=True)
class _ReplaySegment:
    semantic: dict[str, Any]
    wire_items: tuple[dict[str, Any], ...]
    key: tuple[str, ...] | None = None
    requires_replay: bool = False
    atomic_group: tuple[str, ...] | None = None


_REPLAY_FORMATS = {
    "openai": "openai.responses.v1",
    "anthropic": "anthropic.messages.v1",
    "hyperspace": "anthropic.messages.v1",
    "ollama": "ollama.chat.v1",
    "gemini": "gemini.contents.v1",
}

_OPENAI_OPAQUE_OUTPUT_TYPES = {
    "code_interpreter_call",
    "computer_call",
    "custom_tool_call",
    "file_search_call",
    "image_generation_call",
    "local_shell_call",
    "mcp_approval_request",
    "mcp_call",
    "web_search_call",
}


def _openai_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in item.get("content") or []:
        if not isinstance(block, dict) or block.get("type") not in {"output_text", "text"}:
            continue
        text = block.get("text")
        if text not in (None, ""):
            parts.append(str(text))
    return "".join(parts)


def _openai_refusal(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in item.get("content") or []:
        if not isinstance(block, dict) or block.get("type") != "refusal":
            continue
        refusal = block.get("refusal")
        if refusal not in (None, ""):
            parts.append(str(refusal))
    return "".join(parts)


def _openai_computer_call_semantic(item: dict[str, Any]) -> dict[str, Any]:
    call_id = str(item.get("call_id") or item.get("id") or "")
    actions = item.get("actions")
    if not call_id or not isinstance(actions, list) or not actions:
        raise ProviderContextProjectionError(
            "OpenAI replay computer_call requires call_id and non-empty actions"
        )
    pending_safety_checks = item.get("pending_safety_checks")
    if pending_safety_checks is not None and not isinstance(
        pending_safety_checks,
        list,
    ):
        raise ProviderContextProjectionError(
            "OpenAI computer_call pending_safety_checks must be an array"
        )
    if pending_safety_checks:
        raise ProviderContextProjectionError(
            "computer_turn_terminated: OpenAI returned pending safety checks "
            "that require explicit user acknowledgement"
        )
    semantic = {
        "type": "computer_call",
        "call_id": call_id,
        "actions": [
            {
                key: copy.deepcopy(value)
                for key, value in action.items()
                if value is not None
            }
            if isinstance(action, dict)
            else copy.deepcopy(action)
            for action in actions
        ],
    }
    if item.get("id") not in (None, ""):
        semantic["id"] = copy.deepcopy(item["id"])
    return semantic


def _openai_segments(items: list[dict[str, Any]]) -> list[_ReplaySegment]:
    segments: list[_ReplaySegment] = []
    pending_reasoning: list[dict[str, Any]] = []
    index = 0
    while index < len(items):
        item = copy.deepcopy(items[index])
        item_type = item.get("type")
        if item_type == "reasoning" or (
            item_type in _OPENAI_OPAQUE_OUTPUT_TYPES and item_type != "computer_call"
        ):
            pending_reasoning.append(item)
            index += 1
            continue
        if item_type == "computer_call":
            semantic = _openai_computer_call_semantic(item)
            call_id = semantic["call_id"]
            segments.append(
                _ReplaySegment(
                    semantic=semantic,
                    wire_items=tuple([*pending_reasoning, item]),
                    key=("openai.computer_call", call_id),
                    requires_replay=True,
                )
            )
            pending_reasoning = []
            index += 1
            continue
        if item_type == "function_call":
            calls: list[dict[str, Any]] = []
            while index < len(items) and items[index].get("type") == "function_call":
                calls.append(copy.deepcopy(items[index]))
                index += 1
            call_ids = tuple(
                str(call.get("call_id") or call.get("id") or "")
                for call in calls
            )
            if any(not call_id for call_id in call_ids):
                raise ProviderContextProjectionError(
                    "OpenAI replay function_call is missing call_id"
                )
            atomic_group = (
                ("openai.response_calls", *call_ids)
                if pending_reasoning and len(call_ids) > 1
                else None
            )
            for call_index, (call, call_id) in enumerate(zip(calls, call_ids)):
                semantic = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": call.get("name", ""),
                    "arguments": copy.deepcopy(call.get("arguments", "{}")),
                }
                wire_items = (
                    tuple([*pending_reasoning, call])
                    if call_index == 0
                    else (call,)
                )
                segments.append(
                    _ReplaySegment(
                        semantic=semantic,
                        wire_items=wire_items,
                        key=("openai.function_call", call_id),
                        requires_replay=bool(pending_reasoning),
                        atomic_group=atomic_group,
                    )
                )
            pending_reasoning = []
            continue
        if item.get("role") == "assistant" and isinstance(item.get("tool_calls"), list):
            wire_calls: list[dict[str, Any]] = []
            call_ids: list[str] = []
            for raw_call in item["tool_calls"]:
                if not isinstance(raw_call, dict):
                    raise ProviderContextProjectionError(
                        "OpenAI semantic tool calls must be dictionaries"
                    )
                function = raw_call.get("function")
                source = function if isinstance(function, dict) else raw_call
                call_id = str(raw_call.get("id") or raw_call.get("call_id") or "")
                name = str(source.get("name") or "")
                arguments = source.get("arguments", "{}")
                if isinstance(arguments, dict):
                    arguments = json.dumps(
                        arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                if not call_id or not name or not isinstance(arguments, str):
                    raise ProviderContextProjectionError(
                        "OpenAI semantic tool calls require id, name, and JSON-string arguments"
                    )
                call_ids.append(call_id)
                wire_calls.append(
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                )
            segments.append(
                _ReplaySegment(
                    semantic=item,
                    wire_items=tuple([*pending_reasoning, *wire_calls]),
                    key=("openai.assistant_tool_calls", *call_ids),
                    requires_replay=True,
                    atomic_group=("openai.response_calls", *call_ids),
                )
            )
            pending_reasoning = []
            index += 1
            continue
        if item_type == "message" and item.get("role") == "assistant":
            text = _openai_text(item)
            refusal = _openai_refusal(item)
            if not text and not refusal:
                if pending_reasoning:
                    raise ProviderContextProjectionError(
                        "OpenAI reasoning replay has no semantic assistant anchor"
                    )
                index += 1
                continue
            segments.append(
                _ReplaySegment(
                    semantic={"role": "assistant", "content": text or refusal},
                    wire_items=tuple([*pending_reasoning, item]),
                    requires_replay=bool(pending_reasoning or refusal),
                )
            )
            pending_reasoning = []
            index += 1
            continue
        if pending_reasoning:
            raise ProviderContextProjectionError(
                "OpenAI reasoning replay is not followed by an assistant output"
            )
        segments.append(
            _ReplaySegment(
                semantic=copy.deepcopy(item),
                wire_items=(item,),
            )
        )
        index += 1
    if pending_reasoning:
        raise ProviderContextProjectionError(
            "OpenAI replay ends with unanchored reasoning items"
        )
    return segments


def _anthropic_semantic_assistant(
    item: dict[str, Any],
    *, allow_unsigned_thinking: bool = False,
) -> tuple[dict[str, Any] | None, tuple[str, ...], bool]:
    content = item.get("content")
    if not isinstance(content, list):
        return copy.deepcopy(item), (), False
    semantic_blocks: list[dict[str, Any]] = []
    tool_ids: list[str] = []
    has_opaque = False
    for raw_block in content:
        if not isinstance(raw_block, dict):
            continue
        block = copy.deepcopy(raw_block)
        block_type = block.get("type")
        if block_type == "thinking":
            if allow_unsigned_thinking:
                if not isinstance(block.get("thinking"), str) or (
                    "signature" in block and (
                        not isinstance(block["signature"], str) or not block["signature"]
                    )
                ):
                    raise ProviderContextProjectionError("Kimi thinking replay is malformed")
            elif not isinstance(block.get("signature"), str) or not block.get("signature"):
                raise ProviderContextProjectionError(
                    "Anthropic thinking replay is missing its signature"
                )
            has_opaque = True
            continue
        if block_type == "redacted_thinking":
            if not isinstance(block.get("data"), str) or not block.get("data"):
                raise ProviderContextProjectionError(
                    "Anthropic redacted_thinking replay is missing data"
                )
            has_opaque = True
            continue
        semantic_blocks.append(block)
        if block_type == "tool_use":
            tool_id = str(block.get("id") or "")
            if not tool_id:
                raise ProviderContextProjectionError(
                    "Anthropic replay tool_use is missing id"
                )
            tool_ids.append(tool_id)
    if not semantic_blocks:
        return None, tuple(tool_ids), has_opaque
    if all(block.get("type") == "text" for block in semantic_blocks):
        text = "".join(str(block.get("text") or "") for block in semantic_blocks)
        return {"role": "assistant", "content": text}, tuple(tool_ids), has_opaque
    return {
        "role": "assistant",
        "content": semantic_blocks,
    }, tuple(tool_ids), has_opaque


def _anthropic_segments(
    items: list[dict[str, Any]], *, allow_unsigned_thinking: bool = False,
) -> list[_ReplaySegment]:
    segments: list[_ReplaySegment] = []
    for raw in items:
        item = copy.deepcopy(raw)
        if item.get("role") != "assistant":
            segments.append(_ReplaySegment(semantic=item, wire_items=(item,)))
            continue
        semantic, tool_ids, has_opaque = _anthropic_semantic_assistant(
            item, allow_unsigned_thinking=allow_unsigned_thinking,
        )
        if semantic is None:
            if has_opaque:
                raise ProviderContextProjectionError(
                    "Anthropic thinking replay has no semantic assistant anchor"
                )
            continue
        segments.append(
            _ReplaySegment(
                semantic=semantic,
                wire_items=(item,),
                key=("anthropic.tool_use", *tool_ids) if tool_ids else None,
                requires_replay=has_opaque,
            )
        )
    return segments


def _ollama_segments(items: list[dict[str, Any]]) -> list[_ReplaySegment]:
    segments: list[_ReplaySegment] = []
    for raw in items:
        item = copy.deepcopy(raw)
        if item.get("role") != "assistant":
            segments.append(_ReplaySegment(semantic=item, wire_items=(item,)))
            continue
        semantic = copy.deepcopy(item)
        has_thinking = "thinking" in semantic
        semantic.pop("thinking", None)
        tool_ids = tuple(
            str(call.get("id") or "")
            for call in semantic.get("tool_calls") or []
            if isinstance(call, dict)
        )
        if any(not tool_id for tool_id in tool_ids):
            raise ProviderContextProjectionError(
                "Ollama replay tool call is missing id"
            )
        segments.append(
            _ReplaySegment(
                semantic=semantic,
                wire_items=(item,),
                key=("ollama.tool_calls", *tool_ids) if tool_ids else None,
                requires_replay=has_thinking,
            )
        )
    return segments


def _segments_for(
    format_name: str, items: list[dict[str, Any]], *, allow_unsigned_thinking: bool = False,
) -> list[_ReplaySegment]:
    if any(not isinstance(item, dict) for item in items):
        raise ProviderContextProjectionError(
            "provider replay items must be message dictionaries"
        )
    if format_name == "gemini.contents.v1":
        from .gemini import gemini_semantic_message

        return [
            _ReplaySegment(
                semantic=gemini_semantic_message(item),
                wire_items=(copy.deepcopy(item),),
                key=_message_key("gemini", item),
                requires_replay=any(
                    "thought_signature" in part for part in item.get("parts", [])
                ),
            )
            for item in items
        ]
    if format_name == "openai.responses.v1":
        return _openai_segments(items)
    if format_name == "anthropic.messages.v1":
        return _anthropic_segments(items, allow_unsigned_thinking=allow_unsigned_thinking)
    if format_name == "ollama.chat.v1":
        return _ollama_segments(items)
    raise ProviderContextProjectionError(
        f"unsupported provider replay format: {format_name!r}"
    )


def _message_key(provider: str, message: dict[str, Any]) -> tuple[str, ...] | None:
    if provider == "gemini":
        ids = tuple(
            str(p["function_call"].get("id") or "")
            for p in message.get("parts", [])
            if "function_call" in p
        )
        return ("gemini.function_call", *ids) if ids else None
    if provider == "openai" and message.get("type") == "function_call":
        call_id = str(message.get("call_id") or message.get("id") or "")
        return ("openai.function_call", call_id) if call_id else None
    if provider == "openai" and message.get("role") == "assistant":
        call_ids = tuple(
            str(call.get("id") or call.get("call_id") or "")
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
        )
        return ("openai.assistant_tool_calls", *call_ids) if call_ids else None
    if provider == "openai" and message.get("type") == "computer_call":
        call_id = str(message.get("call_id") or message.get("id") or "")
        return ("openai.computer_call", call_id) if call_id else None
    if provider in {"anthropic", "hyperspace"} and message.get("role") == "assistant":
        tool_ids = tuple(
            str(block.get("id") or "")
            for block in message.get("content") or []
            if isinstance(block, dict) and block.get("type") == "tool_use"
        )
        return ("anthropic.tool_use", *tool_ids) if tool_ids else None
    if provider == "ollama" and message.get("role") == "assistant":
        tool_ids = tuple(
            str(call.get("id") or "")
            for call in message.get("tool_calls") or []
            if isinstance(call, dict)
        )
        return ("ollama.tool_calls", *tool_ids) if tool_ids else None
    return None


def _validate_tool_pairs(provider: str, messages: list[dict[str, Any]]) -> None:
    events: list[tuple[Literal["calls", "results", "other"], list[str]]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if provider == "openai" and message.get("type") == "computer_call":
            call_id = str(message.get("call_id") or message.get("id") or "")
            actions = message.get("actions")
            if not call_id or not isinstance(actions, list) or not actions:
                raise ProviderContextProjectionError(
                    "OpenAI computer_call requires call_id and non-empty actions"
                )
            events.append(("calls", [call_id]))
            index += 1
            continue
        if provider == "openai" and message.get("type") == "computer_call_output":
            ids = []
            while (
                index < len(messages)
                and messages[index].get("type") == "computer_call_output"
            ):
                ids.append(str(messages[index].get("call_id") or ""))
                index += 1
            events.append(("results", ids))
            continue
        if provider == "openai" and message.get("type") == "function_call":
            ids: list[str] = []
            while index < len(messages) and messages[index].get("type") == "function_call":
                ids.append(
                    str(
                        messages[index].get("call_id")
                        or messages[index].get("id")
                        or ""
                    )
                )
                index += 1
            events.append(("calls", ids))
            continue
        if provider == "openai" and message.get("type") == "function_call_output":
            ids = []
            while (
                index < len(messages)
                and messages[index].get("type") == "function_call_output"
            ):
                ids.append(str(messages[index].get("call_id") or ""))
                index += 1
            events.append(("results", ids))
            continue
        if provider in {"anthropic", "hyperspace"}:
            blocks = [
                block
                for block in message.get("content") or []
                if isinstance(block, dict)
            ]
            call_ids = [
                str(block.get("id") or "")
                for block in blocks
                if block.get("type") == "tool_use"
            ]
            result_ids = [
                str(block.get("tool_use_id") or "")
                for block in blocks
                if block.get("type") == "tool_result"
            ]
            if call_ids and result_ids:
                raise ProviderContextProjectionError(
                    "Anthropic replay message mixes tool calls and results"
                )
            if call_ids:
                if message.get("role") != "assistant" or any(
                    not isinstance(block.get("name"), str)
                    or not block.get("name")
                    or not isinstance(block.get("input"), dict)
                    for block in blocks
                    if block.get("type") == "tool_use"
                ):
                    raise ProviderContextProjectionError(
                        "Anthropic tool_use blocks require assistant role, name, and object input"
                    )
                events.append(("calls", call_ids))
            elif result_ids:
                leading_result_count = 0
                for block in blocks:
                    if block.get("type") != "tool_result":
                        break
                    leading_result_count += 1
                if (
                    message.get("role") != "user"
                    or leading_result_count != len(result_ids)
                ):
                    raise ProviderContextProjectionError(
                        "Anthropic tool_result blocks require user role and must come first"
                    )
                events.append(("results", result_ids))
            else:
                events.append(("other", []))
            index += 1
            continue
        if provider == "gemini":
            parts = message.get("parts") or []
            calls = [p["function_call"] for p in parts if "function_call" in p]
            results = [
                p["function_response"] for p in parts if "function_response" in p
            ]
            if calls and results:
                raise ProviderContextProjectionError(
                    "Gemini message mixes calls and results"
                )
            if calls:
                if message.get("role") not in {"model", "assistant"} or any(
                    not c.get("name") or not isinstance(c.get("args"), dict)
                    for c in calls
                ):
                    raise ProviderContextProjectionError("Invalid Gemini function call")
                events.append(("calls", [str(c.get("id") or "") for c in calls]))
            elif results:
                ids = []
                while index < len(messages):
                    batch = [
                        p["function_response"]
                        for p in messages[index].get("parts", [])
                        if "function_response" in p
                    ]
                    if not batch:
                        break
                    if messages[index].get("role") != "user" or any(
                        not isinstance(r.get("response"), dict) for r in batch
                    ):
                        raise ProviderContextProjectionError(
                            "Invalid Gemini function response"
                        )
                    ids.extend(str(r.get("id") or "") for r in batch)
                    index += 1
                events.append(("results", ids))
                continue
            else:
                events.append(("other", []))
            index += 1
            continue
        if provider == "ollama" and message.get("tool_calls"):
            invalid_ollama_call = any(
                not isinstance(call, dict)
                or not isinstance(call.get("function"), dict)
                or not isinstance(call["function"].get("name"), str)
                or not call["function"].get("name")
                or not isinstance(call["function"].get("arguments"), dict)
                for call in message.get("tool_calls") or []
            )
            if message.get("role") != "assistant" or invalid_ollama_call:
                raise ProviderContextProjectionError(
                    "Ollama tool calls require assistant role, function name, and object arguments"
                )
            events.append(
                (
                    "calls",
                    [
                        str(call.get("id") or "")
                        for call in message.get("tool_calls") or []
                        if isinstance(call, dict)
                    ],
                )
            )
            index += 1
            continue
        if provider == "ollama" and message.get("role") == "tool":
            ids = []
            while index < len(messages) and messages[index].get("role") == "tool":
                ids.append(str(messages[index].get("tool_call_id") or ""))
                index += 1
            events.append(("results", ids))
            continue
        events.append(("other", []))
        index += 1

    pending: list[str] | None = None
    for event_type, ids in events:
        if event_type == "calls":
            if pending is not None:
                raise ProviderContextProjectionError(
                    "provider context starts a new tool batch before recording all results"
                )
            if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
                raise ProviderContextProjectionError(
                    "provider context contains invalid or duplicate tool call ids"
                )
            pending = ids
            continue
        if event_type == "results":
            if pending is None:
                raise ProviderContextProjectionError(
                    "provider context contains tool results before matching calls"
                )
            if ids != pending:
                raise ProviderContextProjectionError(
                    "provider context tool results must match each call exactly once and in order"
                )
            pending = None
            continue
        if pending is not None:
            raise ProviderContextProjectionError(
                "provider context separates tool calls from their result batch"
            )
    if pending is not None:
        raise ProviderContextProjectionError(
            "provider context is missing results for a replayed tool batch"
        )


def _project_segments(segments: list[_ReplaySegment]) -> list[dict[str, Any]]:
    return [copy.deepcopy(segment.semantic) for segment in segments]


def _ensure_external_remote_delta(messages: list[dict[str, Any]]) -> None:
    conversation = [
        message
        for message in messages
        if message.get("role") not in {"system", "developer"}
    ]
    if not conversation or any(
        message.get("role") != "user"
        or message.get("type") in {
            "function_call",
            "function_call_output",
            "computer_call",
            "computer_call_output",
        }
        for message in conversation
    ):
        raise ProviderContextProjectionError(
            "external previous_response_id requires new user delta messages, not replayed history"
        )


def _message_fingerprint(message: dict[str, Any]) -> str:
    try:
        return json.dumps(
            message,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProviderContextProjectionError(
            "provider context messages must be strict JSON"
        ) from exc


def _fingerprint_occurrences(
    fingerprints: list[str],
) -> dict[str, list[int]]:
    occurrences: dict[str, list[int]] = {}
    for index, fingerprint in enumerate(fingerprints):
        occurrences.setdefault(fingerprint, []).append(index)
    return occurrences


def _neighbor_groups(
    indexes: list[int],
    fingerprints: list[str],
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[tuple[str, str], int],
    dict[str, int | None],
    dict[str, int | None],
]:
    previous_counts: dict[str, int] = {}
    next_counts: dict[str, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    previous_owner: dict[str, int | None] = {}
    next_owner: dict[str, int | None] = {}
    for index in indexes:
        previous = fingerprints[index - 1] if index > 0 else None
        following = (
            fingerprints[index + 1]
            if index + 1 < len(fingerprints)
            else None
        )
        if previous is not None:
            previous_counts[previous] = previous_counts.get(previous, 0) + 1
            previous_owner[previous] = (
                index
                if previous_counts[previous] == 1
                else None
            )
        if following is not None:
            next_counts[following] = next_counts.get(following, 0) + 1
            next_owner[following] = (
                index
                if next_counts[following] == 1
                else None
            )
        if previous is not None and following is not None:
            pair = (previous, following)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return (
        previous_counts,
        next_counts,
        pair_counts,
        previous_owner,
        next_owner,
    )


def _neighbor_degree(
    index: int,
    fingerprints: list[str],
    previous_counts: dict[str, int],
    next_counts: dict[str, int],
    pair_counts: dict[tuple[str, str], int],
) -> int:
    previous = fingerprints[index - 1] if index > 0 else None
    following = (
        fingerprints[index + 1]
        if index + 1 < len(fingerprints)
        else None
    )
    degree = 0
    if previous is not None:
        degree += previous_counts.get(previous, 0)
    if following is not None:
        degree += next_counts.get(following, 0)
    if previous is not None and following is not None:
        degree -= pair_counts.get((previous, following), 0)
    return degree


def _contextual_replay_ownership(
    replay_fingerprints: list[str],
    target_fingerprints: list[str],
    replay_occurrences: dict[str, list[int]],
    target_occurrences: dict[str, list[int]],
    required_indexes: set[int],
) -> dict[int, list[int]]:
    ownership: dict[int, list[int]] = {}
    required_by_fingerprint: dict[str, list[int]] = {}
    for index in required_indexes:
        required_by_fingerprint.setdefault(
            replay_fingerprints[index],
            [],
        ).append(index)

    for fingerprint, required_group in required_by_fingerprint.items():
        replay_indexes = replay_occurrences.get(fingerprint, [])
        target_indexes = target_occurrences.get(fingerprint, [])
        if not target_indexes:
            continue
        if len(replay_indexes) == 1 and len(target_indexes) == 1:
            ownership[replay_indexes[0]] = [target_indexes[0]]
            continue

        (
            replay_previous_counts,
            replay_next_counts,
            replay_pair_counts,
            replay_previous_owner,
            replay_next_owner,
        ) = _neighbor_groups(replay_indexes, replay_fingerprints)
        (
            target_previous_counts,
            target_next_counts,
            target_pair_counts,
            _,
            _,
        ) = _neighbor_groups(target_indexes, target_fingerprints)
        (
            required_previous_counts,
            required_next_counts,
            required_pair_counts,
            _,
            _,
        ) = _neighbor_groups(required_group, replay_fingerprints)

        for target_index in target_indexes:
            degree = _neighbor_degree(
                target_index,
                target_fingerprints,
                replay_previous_counts,
                replay_next_counts,
                replay_pair_counts,
            )
            required_degree = _neighbor_degree(
                target_index,
                target_fingerprints,
                required_previous_counts,
                required_next_counts,
                required_pair_counts,
            )
            if degree == 0 or (degree > 1 and required_degree > 0):
                raise ProviderContextProjectionError(
                    "provider reasoning replay target is ambiguous across repeated turns"
                )
            if degree != 1:
                continue

            previous = (
                target_fingerprints[target_index - 1]
                if target_index > 0
                else None
            )
            following = (
                target_fingerprints[target_index + 1]
                if target_index + 1 < len(target_fingerprints)
                else None
            )
            owner: int | None = None
            if previous is not None:
                owner = replay_previous_owner.get(previous)
            if following is not None:
                next_owner = replay_next_owner.get(following)
                if owner is None:
                    owner = next_owner
                elif next_owner is not None and next_owner != owner:
                    raise ProviderContextProjectionError(
                        "provider reasoning replay target is ambiguous across repeated turns"
                    )
            if owner is None:
                raise ProviderContextProjectionError(
                    "provider reasoning replay target is ambiguous across repeated turns"
                )
            if owner not in required_indexes:
                continue
            replay_degree = _neighbor_degree(
                owner,
                replay_fingerprints,
                target_previous_counts,
                target_next_counts,
                target_pair_counts,
            )
            if replay_degree != 1:
                raise ProviderContextProjectionError(
                    "provider reasoning replay target is ambiguous across repeated turns"
                )
            ownership.setdefault(owner, []).append(target_index)
    return ownership


def _rehydrate(
    provider: str,
    target: list[dict[str, Any]],
    segments: list[_ReplaySegment],
) -> list[dict[str, Any]]:
    replacements: dict[int, tuple[dict[str, Any], ...]] = {}
    mapped_segments: dict[int, int] = {}
    keyed_targets: dict[tuple[str, ...], list[int]] = {}
    for index, message in enumerate(target):
        key = _message_key(provider, message)
        if key is not None:
            keyed_targets.setdefault(key, []).append(index)

    projected = _project_segments(segments)
    canonical_target = [
        _openai_computer_call_semantic(message)
        if provider == "openai" and message.get("type") == "computer_call"
        else copy.deepcopy(message)
        for message in target
    ]
    replay_fingerprints = [_message_fingerprint(message) for message in projected]
    target_fingerprints = [
        _message_fingerprint(message) for message in canonical_target
    ]
    replay_occurrences = _fingerprint_occurrences(replay_fingerprints)
    target_occurrences = _fingerprint_occurrences(target_fingerprints)
    unkeyed_required_indexes = {
        index
        for index, segment in enumerate(segments)
        if segment.requires_replay and segment.key is None
    }
    contextual_ownership = _contextual_replay_ownership(
        replay_fingerprints,
        target_fingerprints,
        replay_occurrences,
        target_occurrences,
        unkeyed_required_indexes,
    )
    for segment_index, segment in enumerate(segments):
        if not segment.requires_replay:
            continue
        if segment.key is not None:
            indexes = keyed_targets.get(segment.key, [])
            if not indexes and segment.requires_replay:
                source_ids = set(segment.key[1:])
                partial_overlap = any(
                    target_key[0] == segment.key[0]
                    and bool(source_ids & set(target_key[1:]))
                    for target_key in keyed_targets
                )
                if partial_overlap:
                    raise ProviderContextProjectionError(
                        "provider-native parallel tool calls must be retained or removed as one atomic group"
                    )
        else:
            indexes = contextual_ownership.get(segment_index, [])
        if not indexes:
            continue
        if len(indexes) != 1:
            raise ProviderContextProjectionError(
                "provider replay segment has an ambiguous semantic target"
            )
        index = indexes[0]
        if canonical_target[index] != segment.semantic:
            raise ProviderContextProjectionError(
                "provider-native tool/reasoning segment was mutated ambiguously"
            )
        if index in replacements:
            raise ProviderContextProjectionError(
                "multiple provider replay segments map to one semantic message"
            )
        replacements[index] = segment.wire_items
        mapped_segments[segment_index] = index

        if segment.key is None:
            anchored_before = (
                segment_index > 0
                and index > 0
                and canonical_target[index - 1]
                == segments[segment_index - 1].semantic
            )
            anchored_after = (
                segment_index + 1 < len(segments)
                and index + 1 < len(target)
                and canonical_target[index + 1]
                == segments[segment_index + 1].semantic
            )
            if not anchored_before and not anchored_after:
                raise ProviderContextProjectionError(
                    "provider reasoning replay has no stable neighboring anchor"
                )

    ordered_target_indexes = [
        mapped_segments[segment_index]
        for segment_index in sorted(mapped_segments)
    ]
    if ordered_target_indexes != sorted(ordered_target_indexes):
        raise ProviderContextProjectionError(
            "provider-native replay segments changed relative order"
        )

    groups: dict[tuple[str, ...], list[int]] = {}
    for segment_index, segment in enumerate(segments):
        if segment.atomic_group is not None:
            groups.setdefault(segment.atomic_group, []).append(segment_index)
    for member_indexes in groups.values():
        matched = [index for index in member_indexes if index in mapped_segments]
        if matched and len(matched) != len(member_indexes):
            raise ProviderContextProjectionError(
                "provider-native parallel calls must remain an atomic replay group"
            )
        if matched:
            target_indexes = [mapped_segments[index] for index in member_indexes]
            if target_indexes != list(
                range(target_indexes[0], target_indexes[0] + len(target_indexes))
            ):
                raise ProviderContextProjectionError(
                    "provider-native parallel call order changed during projection"
                )

    output: list[dict[str, Any]] = []
    for index, message in enumerate(target):
        replacement = replacements.get(index)
        if replacement is None:
            output.append(copy.deepcopy(message))
        else:
            output.extend(copy.deepcopy(list(replacement)))
    if provider == "openai":
        for item in output:
            if item.get("type") != "function_call":
                continue
            call_id = str(item.get("call_id") or item.get("id") or "")
            name = str(item.get("name") or "")
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, dict):
                item["arguments"] = json.dumps(
                    arguments,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
            elif not isinstance(arguments, str):
                raise ProviderContextProjectionError(
                    "OpenAI function_call arguments must be a JSON string"
                )
            if not call_id or not name:
                raise ProviderContextProjectionError(
                    "OpenAI function_call requires call_id and name"
                )
    _validate_tool_pairs(provider, output)
    return output


class ProviderContextAssembler:
    def assemble(
        self,
        state: RunState,
        *,
        toolkit: Toolkit,
        replay_profile: dict[str, str] | None = None,
    ) -> ProviderContextAssembly:
        provider = str(state.provider_state.provider or "").strip().lower()
        target = state.latest_messages()
        frame = current_provider_replay_frame(state)
        allow_unsigned_thinking = validate_replay_profile(
            frame.get("replay_profile") if frame is not None else None,
            active=replay_profile, provider=provider, model=state.provider_state.model,
        )
        expected_format = _REPLAY_FORMATS.get(provider)
        remote_id = (
            state.provider_state.previous_response_id
            if state.provider_state.use_previous_response_chain
            else None
        )
        remote_input = state.remote_continuation_input

        if frame is None and state.metadata.get("provider_replay_required") is True:
            raise ProviderContextProjectionError(
                "provider replay handle is unavailable and no durable replay frame was restored"
            )
        if frame is None or expected_format is None:
            if (
                frame is None
                and remote_id
                and isinstance(state.next_model_input, list)
            ):
                raise ProviderContextProjectionError(
                    "external previous_response_id context changed without a complete local replay prefix"
                )
            external_remote = (
                frame is None
                and bool(remote_id)
                and not isinstance(state.next_model_input, list)
            )
            if external_remote:
                _ensure_external_remote_delta(
                    remote_input
                    if isinstance(remote_input, list)
                    else target
                )
            use_remote = external_remote
            messages = copy.deepcopy(
                remote_input
                if use_remote and isinstance(remote_input, list)
                else target
            )
            return ProviderContextAssembly(
                messages=messages,
                previous_response_id=remote_id if use_remote else None,
                mode="remote_continuation" if use_remote else "semantic",
            )

        if frame.get("format") != expected_format:
            raise ProviderContextProjectionError(
                "provider replay format does not match the active provider"
            )
        ensure_replay_tool_schema_compatible(
            frame,
            toolkit=toolkit,
            provider=provider,
        )
        if frame.get("complete") is not True:
            if (
                remote_id
                and not isinstance(state.next_model_input, list)
            ):
                segments = _segments_for(
                    expected_format,
                    frame.get("items") or [],
                    allow_unsigned_thinking=allow_unsigned_thinking,
                )
                projected = _project_segments(segments)
                resolved_remote_input = remote_input
                remote_context_matches = target == projected
                if (
                    not isinstance(resolved_remote_input, list)
                    and len(target) > len(projected)
                    and target[: len(projected)] == projected
                ):
                    suffix = copy.deepcopy(target[len(projected) :])
                    _ensure_external_remote_delta(suffix)
                    resolved_remote_input = suffix
                    remote_context_matches = True
                if (
                    not isinstance(resolved_remote_input, list)
                    or not remote_context_matches
                ):
                    raise ProviderContextProjectionError(
                        "current model context changed beyond an incomplete remote replay prefix"
                    )
                _validate_tool_pairs(provider, projected)
                return ProviderContextAssembly(
                    messages=copy.deepcopy(resolved_remote_input),
                    previous_response_id=remote_id,
                    mode="remote_continuation",
                )
            raise ProviderContextProjectionError(
                "provider replay frame is incomplete and cannot build local context"
            )

        raw_items = frame.get("items") or []
        segments = _segments_for(
            expected_format, raw_items, allow_unsigned_thinking=allow_unsigned_thinking,
        )
        fallback_messages = _rehydrate(provider, target, segments)
        projected = _project_segments(segments)
        resolved_remote_input = remote_input
        remote_context_matches = target == projected
        if (
            remote_id
            and not isinstance(state.next_model_input, list)
            and not isinstance(resolved_remote_input, list)
            and len(target) > len(projected)
            and target[: len(projected)] == projected
        ):
            suffix = copy.deepcopy(target[len(projected) :])
            _ensure_external_remote_delta(suffix)
            resolved_remote_input = suffix
            remote_context_matches = True
        if (
            remote_id
            and not isinstance(state.next_model_input, list)
            and isinstance(resolved_remote_input, list)
            and remote_context_matches
        ):
            return ProviderContextAssembly(
                messages=copy.deepcopy(resolved_remote_input),
                previous_response_id=remote_id,
                fallback_messages=fallback_messages,
                mode="remote_continuation",
            )
        return ProviderContextAssembly(
            messages=fallback_messages,
            previous_response_id=None,
            mode="local_replay",
        )


__all__ = [
    "ProviderContextAssembler",
    "ProviderContextAssembly",
    "ProviderContextProjectionError",
]
