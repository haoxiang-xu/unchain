from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from ..execution import ExecutionLeaseError
from ..kernel.types import ToolCall
from .models import ToolHistoryOptimizationContext
from .toolkit import Toolkit


@dataclass(frozen=True)
class ToolResultBudgetConfig:
    max_result_chars: int = 4_000
    max_batch_chars: int = 16_000
    preview_chars: int = 600
    min_chars_to_budget: int = 1_200

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolResultBudgetConfig":
        if isinstance(raw, cls):
            return raw
        defaults = cls()
        if not isinstance(raw, dict):
            return defaults

        def int_field(name: str, minimum: int) -> int:
            value = raw.get(name, getattr(defaults, name))
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = getattr(defaults, name)
            return max(minimum, parsed)

        return cls(
            max_result_chars=int_field("max_result_chars", 1),
            max_batch_chars=int_field("max_batch_chars", 1),
            preview_chars=int_field("preview_chars", 0),
            min_chars_to_budget=int_field("min_chars_to_budget", 0),
        )


@dataclass(frozen=True)
class ToolResultBudgetStats:
    result_count: int = 0
    compacted_count: int = 0
    optimizer_error_count: int = 0
    original_chars: int = 0
    budgeted_chars: int = 0
    saved_chars: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "result_count": int(self.result_count),
            "compacted_count": int(self.compacted_count),
            "optimizer_error_count": int(self.optimizer_error_count),
            "original_chars": int(self.original_chars),
            "budgeted_chars": int(self.budgeted_chars),
            "saved_chars": int(self.saved_chars),
        }


@dataclass(frozen=True)
class ToolResultBudgetOutcome:
    messages: list[dict[str, Any]]
    stats: ToolResultBudgetStats


@dataclass
class _ResultRecord:
    message_index: int
    location_type: str
    tool_name: str
    call_id: str
    payload: Any
    payload_format: str
    original_chars: int
    block_index: int | None = None
    part_index: int | None = None
    # S0 rich tool result: when a tool_result block's content is a block array
    # (text + image), only the text is budgeted; the non-text blocks (images)
    # are preserved verbatim here so screenshots reach the model and the block
    # array stays a valid Anthropic `tool_result.content` on rewrite.
    rich_preserved_blocks: list[Any] | None = None


def _safe_repr(value: Any) -> str:
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        pairs: list[tuple[str, Any]] = []
        for key, item in value.items():
            pairs.append((_canonical_key(key), _canonicalize(item)))
        pairs.sort(key=lambda pair: pair[0])

        canonical: dict[str, Any] = {}
        duplicate_counts: dict[str, int] = {}
        for key, item in pairs:
            duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
            output_key = key if duplicate_counts[key] == 1 else f"{key}#{duplicate_counts[key]}"
            canonical[output_key] = item
        return canonical

    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]

    if isinstance(value, (set, frozenset)):
        items = [_canonicalize(item) for item in value]
        return sorted(items, key=_stable_json_dumps)

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return f"{type(value).__module__}.{type(value).__qualname__}:{_safe_repr(value)}"


def _canonical_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    canonical = _canonicalize(key)
    return f"{type(key).__module__}.{type(key).__qualname__}:{_stable_json_dumps(canonical)}"


def _stable_json_dumps(value: Any) -> str:
    try:
        canonical = _canonicalize(value)
        return json.dumps(
            canonical,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except Exception:
        fallback = f"{type(value).__module__}.{type(value).__qualname__}:{_safe_repr(value)}"
        return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))


def _normalize_payload(value: Any) -> tuple[Any, str]:
    if not isinstance(value, str):
        return copy.deepcopy(value), "native"
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value, "raw_string"
    try:
        parsed = json.loads(stripped)
    except Exception:
        return value, "raw_string"
    if isinstance(parsed, (dict, list)):
        return parsed, "json_string"
    return value, "raw_string"


def _serialize_payload(payload: Any, payload_format: str) -> Any:
    if payload_format == "json_string":
        return _stable_json_dumps(payload)
    if payload_format == "raw_string":
        return payload if isinstance(payload, str) else _stable_json_dumps(payload)
    if isinstance(payload, dict):
        return copy.deepcopy(payload)
    return {"value": copy.deepcopy(payload)}


def _payload_chars(payload: Any, payload_format: str) -> int:
    serialized = _serialize_payload(payload, payload_format)
    if isinstance(serialized, str):
        return len(serialized)
    return len(_stable_json_dumps(serialized))


def _preview_text(text: str, preview_chars: int) -> str:
    limit = int(preview_chars)
    if limit <= 0:
        return ""
    if len(text) <= limit * 2:
        return text
    head = text[:limit]
    tail = text[-limit:]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... <omitted {omitted} chars> ...\n{tail}"


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def _budgeted_chars(payload: Any) -> int:
    return len(_stable_json_dumps(payload))


def _retained_scalar_fields(payload: Any, max_chars: int) -> dict[str, Any]:
    if not isinstance(payload, dict) or max_chars <= 0:
        return {}

    candidates: list[tuple[int, str, Any]] = []
    duplicate_counts: dict[str, int] = {}
    for raw_key, value in payload.items():
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            continue
        key = _canonical_key(raw_key)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        if duplicate_counts[key] > 1:
            key = f"{key}#{duplicate_counts[key]}"
        candidate_chars = _budgeted_chars({key: value})
        candidates.append((candidate_chars, key, copy.deepcopy(value)))

    retained: dict[str, Any] = {}
    for _candidate_chars, key, value in sorted(
        candidates,
        key=lambda item: (item[0], item[1]),
    ):
        candidate = {**retained, key: value}
        if _budgeted_chars(candidate) <= max_chars:
            retained[key] = value
    return retained


def _compact_payload(
    payload: Any,
    *,
    reason: str,
    tool_name: str,
    call_id: str,
    preview_chars: int,
) -> dict[str, Any]:
    original_json = _stable_json_dumps(payload)
    compacted: dict[str, Any] = {
        "compacted": True,
        "reason": reason,
        "tool_name": tool_name,
        "call_id": call_id,
        "original_chars": len(original_json),
        "budgeted_chars": 0,
        "preview": _preview_text(original_json, preview_chars),
        "original_sha1": _sha1_text(original_json),
    }
    retained_fields = _retained_scalar_fields(payload, max(0, int(preview_chars)))
    if retained_fields:
        compacted["retained_fields"] = retained_fields
    previous_size = -1
    while True:
        current_size = _budgeted_chars(compacted)
        if current_size == previous_size:
            break
        compacted["budgeted_chars"] = current_size
        previous_size = current_size
    return compacted


def _compact_payload_for_target(
    record: _ResultRecord,
    *,
    reason: str,
    preview_chars: int,
    target_chars: int | None = None,
) -> dict[str, Any]:
    max_preview_chars = max(0, int(preview_chars))
    if target_chars is None:
        return _compact_payload(
            record.payload,
            reason=reason,
            tool_name=record.tool_name,
            call_id=record.call_id,
            preview_chars=max_preview_chars,
        )

    minimal = _compact_payload(
        record.payload,
        reason=reason,
        tool_name=record.tool_name,
        call_id=record.call_id,
        preview_chars=0,
    )
    if _payload_chars(minimal, record.payload_format) > target_chars:
        return minimal

    best = minimal
    low = 0
    high = max_preview_chars
    while low <= high:
        middle = (low + high) // 2
        candidate = _compact_payload(
            record.payload,
            reason=reason,
            tool_name=record.tool_name,
            call_id=record.call_id,
            preview_chars=middle,
        )
        if _payload_chars(candidate, record.payload_format) <= target_chars:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def _tool_call_maps(
    tool_calls: list[ToolCall],
) -> tuple[dict[str, ToolCall], dict[str, list[ToolCall]]]:
    by_id: dict[str, ToolCall] = {}
    by_name: dict[str, list[ToolCall]] = {}
    for call in tool_calls:
        if call.call_id:
            by_id[call.call_id] = call
        by_name.setdefault(call.name, []).append(call)
    return by_id, by_name


def _resolve_call(
    *,
    call_id: str,
    tool_name: str,
    by_id: dict[str, ToolCall],
    by_name: dict[str, list[ToolCall]],
    used_call_ids: set[str],
) -> tuple[str, str]:
    if call_id and call_id in by_id:
        call = by_id[call_id]
        used_call_ids.add(call.call_id)
        return call.name, call.call_id

    if tool_name:
        for call in by_name.get(tool_name, []):
            if call.call_id not in used_call_ids:
                used_call_ids.add(call.call_id)
                return call.name, call.call_id
        calls = by_name.get(tool_name, [])
        if calls:
            return calls[0].name, calls[0].call_id

    return tool_name, call_id


def _collect_result_records(
    messages: list[dict[str, Any]],
    *,
    provider: str,
    tool_calls: list[ToolCall],
) -> list[_ResultRecord]:
    del provider
    by_id, by_name = _tool_call_maps(tool_calls)
    used_call_ids: set[str] = set()
    records: list[_ResultRecord] = []

    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue

        if message.get("type") == "function_call_output":
            call_id = str(message.get("call_id") or "")
            tool_name, resolved_call_id = _resolve_call(
                call_id=call_id,
                tool_name="",
                by_id=by_id,
                by_name=by_name,
                used_call_ids=used_call_ids,
            )
            payload, payload_format = _normalize_payload(message.get("output", ""))
            records.append(
                _ResultRecord(
                    message_index=message_index,
                    location_type="openai_function_call_output",
                    tool_name=tool_name,
                    call_id=resolved_call_id,
                    payload=payload,
                    payload_format=payload_format,
                    original_chars=_payload_chars(payload, payload_format),
                )
            )

        content = message.get("content")
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = str(block.get("tool_use_id") or "")
                tool_name, resolved_call_id = _resolve_call(
                    call_id=call_id,
                    tool_name="",
                    by_id=by_id,
                    by_name=by_name,
                    used_call_ids=used_call_ids,
                )
                raw_content = block.get("content", "")
                if isinstance(raw_content, list):
                    # S0 rich tool result: content is a block array. Budget only
                    # the text portion — image (and any non-text) blocks are
                    # preserved verbatim, so screenshot base64 is neither counted
                    # toward the budget nor rewritten into an illegal bare dict.
                    text_segments: list[str] = []
                    preserved: list[Any] = []
                    for sub_block in raw_content:
                        if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                            text_segments.append(str(sub_block.get("text") or ""))
                        else:
                            preserved.append(sub_block)
                    payload = "\n".join(text_segments)
                    payload_format = "raw_string"
                    records.append(
                        _ResultRecord(
                            message_index=message_index,
                            location_type="content_block_tool_result",
                            tool_name=tool_name,
                            call_id=resolved_call_id,
                            payload=payload,
                            payload_format=payload_format,
                            original_chars=_payload_chars(payload, payload_format),
                            block_index=block_index,
                            rich_preserved_blocks=preserved,
                        )
                    )
                    continue
                payload, payload_format = _normalize_payload(raw_content)
                records.append(
                    _ResultRecord(
                        message_index=message_index,
                        location_type="content_block_tool_result",
                        tool_name=tool_name,
                        call_id=resolved_call_id,
                        payload=payload,
                        payload_format=payload_format,
                        original_chars=_payload_chars(payload, payload_format),
                        block_index=block_index,
                    )
                )

        if message.get("role") == "tool" and "content" in message:
            call_id = str(message.get("tool_call_id") or "")
            tool_name, resolved_call_id = _resolve_call(
                call_id=call_id,
                tool_name="",
                by_id=by_id,
                by_name=by_name,
                used_call_ids=used_call_ids,
            )
            payload, payload_format = _normalize_payload(message.get("content", ""))
            records.append(
                _ResultRecord(
                    message_index=message_index,
                    location_type="role_tool_content",
                    tool_name=tool_name,
                    call_id=resolved_call_id,
                    payload=payload,
                    payload_format=payload_format,
                    original_chars=_payload_chars(payload, payload_format),
                )
            )

        parts = message.get("parts")
        if isinstance(parts, list):
            for part_index, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                function_response = part.get("function_response")
                if not isinstance(function_response, dict):
                    continue
                tool_name = str(function_response.get("name") or "")
                call_id = str(function_response.get("id") or part.get("id") or "")
                resolved_tool_name, resolved_call_id = _resolve_call(
                    call_id=call_id,
                    tool_name=tool_name,
                    by_id=by_id,
                    by_name=by_name,
                    used_call_ids=used_call_ids,
                )
                payload, payload_format = _normalize_payload(function_response.get("response", {}))
                records.append(
                    _ResultRecord(
                        message_index=message_index,
                        location_type="parts_function_response",
                        tool_name=resolved_tool_name,
                        call_id=resolved_call_id,
                        payload=payload,
                        payload_format=payload_format,
                        original_chars=_payload_chars(payload, payload_format),
                        part_index=part_index,
                    )
                )

    return records


def _write_record_payload(
    messages: list[dict[str, Any]],
    record: _ResultRecord,
    payload: Any,
) -> None:
    if record.message_index < 0 or record.message_index >= len(messages):
        return
    message = messages[record.message_index]
    if not isinstance(message, dict):
        return

    serialized = _serialize_payload(payload, record.payload_format)

    if record.location_type == "openai_function_call_output":
        message["output"] = serialized
        return

    if record.location_type == "content_block_tool_result":
        content = message.get("content")
        if not isinstance(content, list) or record.block_index is None:
            return
        if record.block_index >= len(content) or not isinstance(content[record.block_index], dict):
            return
        if record.rich_preserved_blocks is not None:
            # Rebuild a valid block array: compacted text (if any) first, then
            # the preserved image/non-text blocks verbatim.
            text_str = serialized if isinstance(serialized, str) else _stable_json_dumps(serialized)
            new_blocks: list[Any] = []
            if text_str:
                new_blocks.append({"type": "text", "text": text_str})
            new_blocks.extend(copy.deepcopy(record.rich_preserved_blocks))
            content[record.block_index]["content"] = new_blocks
            return
        content[record.block_index]["content"] = serialized
        return

    if record.location_type == "role_tool_content":
        message["content"] = serialized
        return

    if record.location_type == "parts_function_response":
        parts = message.get("parts")
        if not isinstance(parts, list) or record.part_index is None:
            return
        if record.part_index >= len(parts) or not isinstance(parts[record.part_index], dict):
            return
        function_response = parts[record.part_index].get("function_response")
        if not isinstance(function_response, dict):
            return
        function_response["response"] = serialized


class ToolResultBudgetController:
    def __init__(self, config: ToolResultBudgetConfig | None = None) -> None:
        self.config = config or ToolResultBudgetConfig()

    def budget_messages(
        self,
        *,
        provider: str,
        toolkit: Toolkit | None,
        tool_calls: list[ToolCall],
        result_messages: list[dict[str, Any]],
        session_id: str = "",
        latest_messages: list[dict[str, Any]] | None = None,
        reason: str = "tool_result_budget",
    ) -> ToolResultBudgetOutcome:
        normalized_provider = str(provider or "").strip().lower()
        messages = copy.deepcopy(result_messages)
        records = _collect_result_records(
            messages,
            provider=normalized_provider,
            tool_calls=tool_calls,
        )
        if not records:
            return ToolResultBudgetOutcome(
                messages=messages,
                stats=ToolResultBudgetStats(),
            )

        original_total = sum(record.original_chars for record in records)
        max_result_chars = max(1, int(self.config.max_result_chars))
        max_batch_chars = max(1, int(self.config.max_batch_chars))
        min_chars_to_budget = max(0, int(self.config.min_chars_to_budget))
        compacted_payloads: dict[int, Any] = {}
        current_chars = {
            index: record.original_chars
            for index, record in enumerate(records)
        }
        optimizer_error_count = 0
        optimizer_attempted: set[int] = set()

        def optimizer_payload(index: int) -> tuple[Any | None, bool]:
            nonlocal optimizer_error_count
            if index in optimizer_attempted:
                return None, False
            optimizer_attempted.add(index)
            record = records[index]
            if toolkit is None or not record.tool_name:
                return None, False
            tool_obj = toolkit.get(record.tool_name)
            if tool_obj is None:
                return None, False
            optimizer = getattr(tool_obj, "history_result_optimizer", None)
            if not callable(optimizer):
                return None, False
            try:
                context = ToolHistoryOptimizationContext(
                    tool_name=record.tool_name,
                    call_id=record.call_id,
                    kind="result",
                    provider=normalized_provider,
                    session_id=session_id,
                    latest_messages=copy.deepcopy(latest_messages) if latest_messages is not None else [],
                    max_chars=max_result_chars,
                    preview_chars=max(1, int(self.config.preview_chars)),
                    include_hash=True,
                )
                return optimizer(copy.deepcopy(record.payload), context), True
            except ExecutionLeaseError:
                raise
            except Exception:
                optimizer_error_count += 1
                return None, False

        def apply_payload(index: int, payload: Any) -> bool:
            size = _payload_chars(payload, records[index].payload_format)
            if size >= current_chars[index]:
                return False
            compacted_payloads[index] = payload
            current_chars[index] = size
            return True

        def apply_soft_compaction(index: int) -> None:
            record = records[index]
            optimized, optimizer_used = optimizer_payload(index)
            if optimizer_used:
                if optimized is not None and apply_payload(index, optimized):
                    return
            generic = _compact_payload_for_target(
                record,
                reason=reason,
                preview_chars=self.config.preview_chars,
            )
            apply_payload(index, generic)

        def apply_hard_compaction(index: int, target_chars: int) -> bool:
            record = records[index]
            candidates: list[Any] = []
            optimized, optimizer_used = optimizer_payload(index)
            if optimizer_used and optimized is not None:
                candidates.append(optimized)
            candidates.append(
                _compact_payload_for_target(
                    record,
                    reason=reason,
                    preview_chars=self.config.preview_chars,
                    target_chars=target_chars,
                )
            )
            candidates.sort(key=lambda item: _payload_chars(item, record.payload_format))
            for candidate in candidates:
                if apply_payload(index, candidate):
                    return True
            return False

        for index, record in enumerate(records):
            if (
                record.original_chars >= min_chars_to_budget
                and record.original_chars > max_result_chars
            ):
                apply_soft_compaction(index)

        while sum(current_chars.values()) > max_batch_chars:
            changed = False
            for index in sorted(
                range(len(records)),
                key=lambda candidate: current_chars[candidate],
                reverse=True,
            ):
                total_chars = sum(current_chars.values())
                if total_chars <= max_batch_chars:
                    break
                target_chars = max(0, max_batch_chars - (total_chars - current_chars[index]))
                if apply_hard_compaction(index, target_chars):
                    changed = True
            if not changed:
                break

        for index, payload in compacted_payloads.items():
            _write_record_payload(messages, records[index], payload)

        final_chars = sum(current_chars.values())
        compacted_count = sum(
            1
            for index, record in enumerate(records)
            if _stable_json_dumps(
                _serialize_payload(
                    compacted_payloads.get(index, record.payload),
                    record.payload_format,
                )
            )
            != _stable_json_dumps(
                _serialize_payload(record.payload, record.payload_format)
            )
        )

        return ToolResultBudgetOutcome(
            messages=messages,
            stats=ToolResultBudgetStats(
                result_count=len(records),
                compacted_count=compacted_count,
                optimizer_error_count=optimizer_error_count,
                original_chars=original_total,
                budgeted_chars=final_chars,
                saved_chars=max(0, original_total - final_chars),
            ),
        )


__all__ = [
    "ToolResultBudgetConfig",
    "ToolResultBudgetController",
    "ToolResultBudgetOutcome",
    "ToolResultBudgetStats",
]
