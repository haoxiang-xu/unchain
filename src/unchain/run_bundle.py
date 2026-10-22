"""Strict, deterministic accounting facts for one agent run tree.

The provider-call receipt is the atomic source of truth.  A :class:`RunBundle`
is a pure projection over a set of those receipts; totals are never copied from
parent or child run summaries, so replay and nested runs cannot double-count.

The two public wire schemas in this module are intentionally closed:

* ``unchain.provider_call_usage.v1`` for :class:`ProviderCallReceipt`
* ``unchain.run_bundle.v1`` for :class:`RunBundle`

Only the explicitly namespaced ``extensions`` objects are open.  Raw prompts,
provider requests, hidden reasoning items, secrets, and artifact bytes are not
valid extension data and therefore cannot enter a bundle accidentally.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, ClassVar


PROVIDER_CALL_USAGE_SCHEMA = "unchain.provider_call_usage.v1"
RUN_BUNDLE_SCHEMA = "unchain.run_bundle.v1"
PROVIDER_CALL_SET_UNION_ALGORITHM = "provider_call_set_union.v1"

_MAX_CANONICAL_BYTES = 2 * 1024 * 1024
_MAX_RECEIPTS = 10_000
_MAX_SAFE_INTEGER = 9_007_199_254_740_991
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_EXTENSION_KEY_RE = re.compile(
    r"^[a-z][a-z0-9.-]{1,127}/[a-z][a-z0-9._-]{0,127}$"
)
_RFC3339_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?"
    r"(?P<offset>Z|[+-]\d{2}:\d{2})$"
)
_PROHIBITED_EXTENSION_KEYS = frozenset(
    {
        "api_key",
        "artifact_bytes",
        "attachment",
        "attachments",
        "authorization",
        "chain_of_thought",
        "credential",
        "credentials",
        "messages",
        "password",
        "prompt",
        "provider_request",
        "raw_prompt",
        "raw_payload",
        "raw_request",
        "raw_response",
        "reasoning",
        "reasoning_content",
        "reasoning_item",
        "reasoning_items",
        "request",
        "response",
        "secret",
        "secrets",
        "system_prompt",
        "tool_output",
        "tool_outputs",
    }
)

_USAGE_SOURCES = frozenset(
    {
        "provider_observed",
        "provider_observed_partial",
        "legacy_partial",
        "unavailable",
    }
)
_RECEIPT_STATUSES = frozenset({"completed", "failed", "uncertain"})
_RUN_STATUSES = frozenset(
    {"running", "completed", "failed", "suspended", "cancelled", "uncertain"}
)
_RUN_RELATIONS = frozenset(
    {"root", "subagent", "graph_node", "recipe_node", "auxiliary"}
)
_CHILD_RELATIONS = _RUN_RELATIONS - {"root"}
_COVERAGE_STATUSES = frozenset({"complete", "partial", "unavailable"})
_COST_STATUSES = frozenset({"estimated", "partial", "unavailable"})
_PRICING_STATUSES = frozenset({"estimated", "unavailable"})
_LEGACY_STATUSES = frozenset({"canonical", "legacy_partial"})
_METRIC_EVENT_KINDS = frozenset(
    {
        "artifact",
        "model_attempt",
        "iteration",
        "tool_call",
        "tool_result",
        "interaction",
        "context_build",
        "context_compaction",
        "error",
    }
)
_METRIC_EVENT_OUTCOMES = frozenset(
    {"completed", "failed", "uncertain", "requested", "skipped"}
)
_METRIC_EVIDENCE_KINDS = frozenset(
    {"artifact", "interaction", "context_event"}
)
_ORCHESTRATION_MODES = frozenset(
    {
        "default",
        "developer_waiting_approval",
    }
)
METRIC_EVENT_SET_UNION_ALGORITHM = "unique_metric_event_set_union.v1"
_MAX_METRIC_EVENTS = 50_000
_MAX_METRIC_EVIDENCE_REFS = 16


class RunBundleProtocolError(ValueError):
    """A provider receipt or run bundle violated its closed wire contract."""


def _canonical_text(
    value: object,
    field_name: str,
    *,
    optional: bool = False,
    maximum: int = 256,
) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str:
        raise TypeError(f"{field_name} must be exact text")
    normalized = unicodedata.normalize("NFC", value)
    if (
        normalized != value
        or not value
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise RunBundleProtocolError(
            f"{field_name} must be canonical non-empty text"
        )
    return value


def _slug(value: object, field_name: str) -> str:
    text = _canonical_text(value, field_name, maximum=128)
    assert isinstance(text, str)
    if _SLUG_RE.fullmatch(text) is None:
        raise RunBundleProtocolError(f"{field_name} must be a canonical slug")
    return text


def _enum(value: object, field_name: str, allowed: frozenset[str]) -> str:
    text = _canonical_text(value, field_name, maximum=128)
    assert isinstance(text, str)
    if text not in allowed:
        raise RunBundleProtocolError(f"{field_name} is unsupported")
    return text


def _optional_sha256(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RunBundleProtocolError(
            f"{field_name} must be a bare lowercase SHA-256 digest"
        )
    return value


def _sha256(value: object, field_name: str) -> str:
    result = _optional_sha256(value, field_name)
    if result is None:
        raise RunBundleProtocolError(f"{field_name} is required")
    return result


def _optional_count(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an exact integer or null")
    if value < 0:
        raise RunBundleProtocolError(f"{field_name} must be non-negative")
    if value > _MAX_SAFE_INTEGER:
        raise RunBundleProtocolError(f"{field_name} exceeds the safe integer limit")
    return value


def _count(value: object, field_name: str) -> int:
    result = _optional_count(value, field_name)
    if result is None:
        raise RunBundleProtocolError(f"{field_name} is required")
    return result


def _positive_revision(value: object) -> int:
    if type(value) is not int or value <= 0 or value > _MAX_SAFE_INTEGER:
        raise RunBundleProtocolError("revision must be a positive exact integer")
    return value


def _timestamp(value: object, field_name: str, *, optional: bool = True) -> str | None:
    text = _canonical_text(value, field_name, optional=optional, maximum=40)
    if text is None:
        return None
    try:
        _timestamp_instant(text)
    except (TypeError, ValueError):
        raise RunBundleProtocolError(
            f"{field_name} must be a valid RFC3339 timestamp"
        ) from None
    return text


def _timestamp_instant(value: str) -> int:
    """Return an RFC3339 instant as exact nanoseconds since the Unix epoch."""

    if type(value) is not str:
        raise TypeError("timestamp must be exact text")
    match = _RFC3339_RE.fullmatch(value)
    if match is None:
        raise ValueError("unsupported RFC3339 timestamp")
    base = datetime.strptime(
        f"{match.group('date')}T{match.group('time')}",
        "%Y-%m-%dT%H:%M:%S",
    ).replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = base - epoch
    seconds = delta.days * 86_400 + delta.seconds
    raw_offset = match.group("offset")
    if raw_offset != "Z":
        offset_hour = int(raw_offset[1:3])
        offset_minute = int(raw_offset[4:6])
        if offset_hour > 23 or offset_minute > 59:
            raise ValueError("invalid RFC3339 offset")
        offset_seconds = (offset_hour * 60 + offset_minute) * 60
        seconds += -offset_seconds if raw_offset[0] == "+" else offset_seconds
    fraction = int((match.group("fraction") or "").ljust(9, "0") or "0")
    return seconds * 1_000_000_000 + fraction


def _strict_record(
    value: object,
    *,
    keys: frozenset[str],
    field_name: str,
) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{field_name} must be an exact object")
    if set(value) != keys:
        raise RunBundleProtocolError(f"{field_name} uses an unsupported shape")
    return value


def _validate_json(value: object, *, path: str, check_sensitive_keys: bool) -> None:
    pending: list[tuple[object, str, int]] = [(value, path, 0)]
    nodes = 0
    while pending:
        item, item_path, depth = pending.pop()
        nodes += 1
        if nodes > 100_000:
            raise RunBundleProtocolError(f"{path} exceeds the JSON node limit")
        item_type = type(item)
        if depth > 12:
            raise RunBundleProtocolError(f"{item_path} exceeds the JSON depth limit")
        if item is None or item_type is bool:
            continue
        if item_type is int:
            if check_sensitive_keys and not -_MAX_SAFE_INTEGER <= item <= _MAX_SAFE_INTEGER:
                raise RunBundleProtocolError(
                    f"{item_path} exceeds the safe integer limit"
                )
            continue
        if item_type is str:
            if check_sensitive_keys and len(item) > 16_384:
                raise RunBundleProtocolError(f"{item_path} string is too long")
            continue
        if item_type is float:
            if check_sensitive_keys:
                raise TypeError(f"{item_path} extension numbers must be exact integers")
            if not math.isfinite(item):
                raise RunBundleProtocolError(f"{item_path} must be finite")
            continue
        if item_type is list:
            if check_sensitive_keys and len(item) > 1_000:
                raise RunBundleProtocolError(f"{item_path} array is too large")
            pending.extend(
                (child, f"{item_path}[{index}]", depth + 1)
                for index, child in enumerate(item)
            )
            continue
        if item_type is dict:
            if check_sensitive_keys and len(item) > 1_000:
                raise RunBundleProtocolError(f"{item_path} object is too large")
            for key, child in item.items():
                if type(key) is not str:
                    raise TypeError(f"{item_path} requires exact text object keys")
                if check_sensitive_keys and key.lower() in _PROHIBITED_EXTENSION_KEYS:
                    raise RunBundleProtocolError(
                        f"{item_path}.{key} is prohibited bundle content"
                    )
                pending.append((child, f"{item_path}.{key}", depth + 1))
            continue
        raise TypeError(f"{item_path} requires exact JSON value types")


def _json_copy(
    value: object,
    *,
    path: str,
    check_sensitive_keys: bool = False,
) -> Any:
    _validate_json(value, path=path, check_sensitive_keys=check_sensitive_keys)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, UnicodeError) as exc:
        raise RunBundleProtocolError(f"{path} must be strict canonical JSON") from exc
    if len(encoded) > _MAX_CANONICAL_BYTES:
        raise RunBundleProtocolError(f"{path} exceeds the canonical byte limit")
    return json.loads(encoded)


def _canonical_bytes(value: object) -> bytes:
    copied = _json_copy(value, path="run bundle")
    return json.dumps(
        copied,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Return the bare lowercase SHA-256 of strict canonical JSON."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _freeze_extensions(value: object | None) -> Mapping[str, Any]:
    raw = {} if value is None else value
    if type(raw) is not dict:
        raise TypeError("extensions must be an exact object")
    for key in raw:
        if type(key) is not str or _EXTENSION_KEY_RE.fullmatch(key) is None:
            raise RunBundleProtocolError(
                "extension keys must use the namespace/name form"
            )
    copied = _json_copy(
        raw,
        path="extensions",
        check_sensitive_keys=True,
    )
    return MappingProxyType(copied)


def _extensions_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    return _json_copy(
        dict(value),
        path="extensions",
        check_sensitive_keys=True,
    )


def _present_count(
    sources: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    field_name: str,
) -> int | None:
    values: list[int] = []
    for source in sources:
        for alias in aliases:
            if alias not in source:
                continue
            observed = _optional_count(source[alias], field_name)
            if observed is not None:
                values.append(observed)
    if not values:
        return None
    if any(value != values[0] for value in values[1:]):
        raise RunBundleProtocolError(f"{field_name} aliases disagree")
    return values[0]


@dataclass(frozen=True, slots=True)
class ProviderCallUsage:
    """Nullable, disjoint token counters for exactly one provider send."""

    input_uncached_tokens: int | None = None
    input_cache_read_tokens: int | None = None
    input_cache_write_tokens: int | None = None
    input_cache_write_5m_tokens: int | None = None
    input_cache_write_1h_tokens: int | None = None
    input_total_tokens: int | None = None
    output_visible_tokens: int | None = None
    output_reasoning_tokens: int | None = None
    output_total_tokens: int | None = None
    total_tokens: int | None = None
    source: str = "unavailable"

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "input_uncached_tokens",
        "input_cache_read_tokens",
        "input_cache_write_tokens",
        "input_cache_write_5m_tokens",
        "input_cache_write_1h_tokens",
        "input_total_tokens",
        "output_visible_tokens",
        "output_reasoning_tokens",
        "output_total_tokens",
        "total_tokens",
    )

    def __post_init__(self) -> None:
        for field_name in self._FIELDS:
            object.__setattr__(
                self,
                field_name,
                _optional_count(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "source",
            _enum(self.source, "usage source", _USAGE_SOURCES),
        )
        input_parts = (
            self.input_uncached_tokens,
            self.input_cache_read_tokens,
            self.input_cache_write_tokens,
        )
        if (
            self.input_cache_write_5m_tokens is not None
            and self.input_cache_write_1h_tokens is not None
        ):
            cache_write_total = (
                self.input_cache_write_5m_tokens
                + self.input_cache_write_1h_tokens
            )
            if self.input_cache_write_tokens is None:
                object.__setattr__(
                    self,
                    "input_cache_write_tokens",
                    cache_write_total,
                )
                input_parts = (
                    self.input_uncached_tokens,
                    self.input_cache_read_tokens,
                    self.input_cache_write_tokens,
                )
            elif cache_write_total != self.input_cache_write_tokens:
                raise RunBundleProtocolError(
                    "cache-write TTL components do not equal cache-write total"
                )
        known_input_parts = [value for value in input_parts if value is not None]
        if self.input_total_tokens is not None:
            if sum(known_input_parts) > self.input_total_tokens:
                raise RunBundleProtocolError(
                    "input token components exceed input total"
                )
            if len(known_input_parts) == 3 and sum(known_input_parts) != self.input_total_tokens:
                raise RunBundleProtocolError(
                    "input token components do not equal input total"
                )
        output_parts = (self.output_visible_tokens, self.output_reasoning_tokens)
        known_output_parts = [value for value in output_parts if value is not None]
        if self.output_total_tokens is not None:
            if sum(known_output_parts) > self.output_total_tokens:
                raise RunBundleProtocolError(
                    "output token components exceed output total"
                )
            if len(known_output_parts) == 2 and sum(known_output_parts) != self.output_total_tokens:
                raise RunBundleProtocolError(
                    "output token components do not equal output total"
                )
        if (
            self.total_tokens is not None
            and self.input_total_tokens is not None
            and self.output_total_tokens is not None
            and self.input_total_tokens + self.output_total_tokens
            != self.total_tokens
        ):
            raise RunBundleProtocolError(
                "input and output totals do not equal the provider total"
            )

    @property
    def has_observed_total(self) -> bool:
        return self.total_tokens is not None

    @property
    def is_disjoint_complete(self) -> bool:
        return all(getattr(self, field_name) is not None for field_name in self._FIELDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": {
                "uncached_tokens": self.input_uncached_tokens,
                "cache_read_tokens": self.input_cache_read_tokens,
                "cache_write_tokens": self.input_cache_write_tokens,
                "cache_write_5m_tokens": self.input_cache_write_5m_tokens,
                "cache_write_1h_tokens": self.input_cache_write_1h_tokens,
                "total_tokens": self.input_total_tokens,
            },
            "output": {
                "visible_tokens": self.output_visible_tokens,
                "reasoning_tokens": self.output_reasoning_tokens,
                "total_tokens": self.output_total_tokens,
            },
            "total_tokens": self.total_tokens,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCallUsage:
        raw = _strict_record(
            value,
            keys=frozenset({"input", "output", "total_tokens", "source"}),
            field_name="provider call usage",
        )
        input_usage = _strict_record(
            raw["input"],
            keys=frozenset(
                {
                    "uncached_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "cache_write_5m_tokens",
                    "cache_write_1h_tokens",
                    "total_tokens",
                }
            ),
            field_name="provider call input usage",
        )
        output_usage = _strict_record(
            raw["output"],
            keys=frozenset({"visible_tokens", "reasoning_tokens", "total_tokens"}),
            field_name="provider call output usage",
        )
        return cls(
            input_uncached_tokens=input_usage["uncached_tokens"],
            input_cache_read_tokens=input_usage["cache_read_tokens"],
            input_cache_write_tokens=input_usage["cache_write_tokens"],
            input_cache_write_5m_tokens=input_usage["cache_write_5m_tokens"],
            input_cache_write_1h_tokens=input_usage["cache_write_1h_tokens"],
            input_total_tokens=input_usage["total_tokens"],
            output_visible_tokens=output_usage["visible_tokens"],
            output_reasoning_tokens=output_usage["reasoning_tokens"],
            output_total_tokens=output_usage["total_tokens"],
            total_tokens=raw["total_tokens"],
            source=raw["source"],
        )

    @classmethod
    def from_openai_usage(cls, value: Mapping[str, Any] | None) -> ProviderCallUsage:
        raw = dict(value) if isinstance(value, Mapping) else {}
        input_details_raw = raw.get("input_tokens_details")
        input_details = (
            dict(input_details_raw) if isinstance(input_details_raw, Mapping) else {}
        )
        output_details_raw = raw.get("output_tokens_details")
        output_details = (
            dict(output_details_raw) if isinstance(output_details_raw, Mapping) else {}
        )
        input_total = _present_count([raw], ["input_tokens"], "input total tokens")
        cache_read = _present_count(
            [input_details, raw],
            ["cached_tokens", "cache_read_tokens", "cache_read_input_tokens"],
            "input cache read tokens",
        )
        cache_write = _present_count(
            [input_details, raw],
            [
                "cache_write_tokens",
                "cache_creation_tokens",
                "cache_creation_input_tokens",
                "cache_write_input_tokens",
            ],
            "input cache write tokens",
        )
        input_uncached = None
        if input_total is not None and cache_read is not None and cache_write is not None:
            if cache_read + cache_write > input_total:
                raise RunBundleProtocolError(
                    "OpenAI cache input tokens exceed input_tokens"
                )
            input_uncached = input_total - cache_read - cache_write

        output_total = _present_count([raw], ["output_tokens"], "output total tokens")
        reasoning = _present_count(
            [output_details, raw],
            ["reasoning_tokens", "reasoning_output_tokens"],
            "output reasoning tokens",
        )
        output_visible = None
        if output_total is not None and reasoning is not None:
            if reasoning > output_total:
                raise RunBundleProtocolError(
                    "OpenAI reasoning tokens exceed output_tokens"
                )
            output_visible = output_total - reasoning
        total = _present_count([raw], ["total_tokens"], "total tokens")
        if total is None and input_total is not None and output_total is not None:
            total = input_total + output_total
        usage = cls(
            input_uncached_tokens=input_uncached,
            input_cache_read_tokens=cache_read,
            input_cache_write_tokens=cache_write,
            input_cache_write_5m_tokens=None,
            input_cache_write_1h_tokens=None,
            input_total_tokens=input_total,
            output_visible_tokens=output_visible,
            output_reasoning_tokens=reasoning,
            output_total_tokens=output_total,
            total_tokens=total,
            source="provider_observed_partial",
        )
        return cls(**{**usage._field_values(), "source": cls._observed_source(usage)})

    @classmethod
    def from_anthropic_usage(
        cls,
        value: Mapping[str, Any] | None,
        *,
        reasoning_present: bool,
    ) -> ProviderCallUsage:
        raw = dict(value) if isinstance(value, Mapping) else {}
        input_uncached = _present_count([raw], ["input_tokens"], "input uncached tokens")
        cache_read = _present_count(
            [raw], ["cache_read_input_tokens"], "input cache read tokens"
        )
        cache_write = _present_count(
            [raw], ["cache_creation_input_tokens"], "input cache write tokens"
        )
        cache_creation_raw = raw.get("cache_creation")
        cache_creation = (
            dict(cache_creation_raw)
            if isinstance(cache_creation_raw, Mapping)
            else {}
        )
        cache_write_5m = _present_count(
            [cache_creation, raw],
            ["ephemeral_5m_input_tokens", "cache_creation_5m_input_tokens"],
            "input cache write 5m tokens",
        )
        cache_write_1h = _present_count(
            [cache_creation, raw],
            ["ephemeral_1h_input_tokens", "cache_creation_1h_input_tokens"],
            "input cache write 1h tokens",
        )
        if cache_write is None and cache_write_5m is not None and cache_write_1h is not None:
            cache_write = cache_write_5m + cache_write_1h
        input_total = None
        if None not in (input_uncached, cache_read, cache_write):
            input_total = input_uncached + cache_read + cache_write
        output_total = _present_count([raw], ["output_tokens"], "output total tokens")
        output_details_raw = raw.get("output_tokens_details")
        output_details = (
            dict(output_details_raw) if isinstance(output_details_raw, Mapping) else {}
        )
        reasoning = _present_count(
            [output_details], ["reasoning_tokens"], "output reasoning tokens"
        )
        if reasoning is None and not reasoning_present:
            reasoning = 0
        visible = None
        if output_total is not None and reasoning is not None:
            if reasoning > output_total:
                raise RunBundleProtocolError(
                    "Anthropic reasoning tokens exceed output_tokens"
                )
            visible = output_total - reasoning
        total = (
            input_total + output_total
            if input_total is not None and output_total is not None
            else None
        )
        usage = cls(
            input_uncached_tokens=input_uncached,
            input_cache_read_tokens=cache_read,
            input_cache_write_tokens=cache_write,
            input_cache_write_5m_tokens=cache_write_5m,
            input_cache_write_1h_tokens=cache_write_1h,
            input_total_tokens=input_total,
            output_visible_tokens=visible,
            output_reasoning_tokens=reasoning,
            output_total_tokens=output_total,
            total_tokens=total,
            source="provider_observed_partial",
        )
        return cls(**{**usage._field_values(), "source": cls._observed_source(usage)})

    @classmethod
    def from_ollama_usage(
        cls,
        value: Mapping[str, Any] | None,
        *,
        reasoning_present: bool,
    ) -> ProviderCallUsage:
        raw = dict(value) if isinstance(value, Mapping) else {}
        input_total = _present_count(
            [raw], ["prompt_eval_count"], "input total tokens"
        )
        output_total = _present_count([raw], ["eval_count"], "output total tokens")
        reasoning = _present_count(
            [raw], ["thinking_eval_count", "reasoning_eval_count"], "output reasoning tokens"
        )
        if reasoning is None and not reasoning_present:
            reasoning = 0
        visible = None
        if output_total is not None and reasoning is not None:
            if reasoning > output_total:
                raise RunBundleProtocolError(
                    "Ollama reasoning tokens exceed eval_count"
                )
            visible = output_total - reasoning
        total = (
            input_total + output_total
            if input_total is not None and output_total is not None
            else None
        )
        usage = cls(
            input_uncached_tokens=input_total,
            input_cache_read_tokens=0,
            input_cache_write_tokens=0,
            input_cache_write_5m_tokens=0,
            input_cache_write_1h_tokens=0,
            input_total_tokens=input_total,
            output_visible_tokens=visible,
            output_reasoning_tokens=reasoning,
            output_total_tokens=output_total,
            total_tokens=total,
            source="provider_observed_partial",
        )
        return cls(**{**usage._field_values(), "source": cls._observed_source(usage)})

    @classmethod
    def from_legacy_model_turn_result(cls, result: Any) -> ProviderCallUsage:
        consumed = getattr(result, "consumed_tokens", 0)
        input_tokens = getattr(result, "input_tokens", 0)
        output_tokens = getattr(result, "output_tokens", 0)
        cache_read = getattr(result, "cache_read_input_tokens", 0)
        cache_write = getattr(result, "cache_creation_input_tokens", 0)
        return cls(
            input_uncached_tokens=input_tokens if type(input_tokens) is int and input_tokens > 0 else None,
            input_cache_read_tokens=cache_read if type(cache_read) is int and cache_read > 0 else None,
            input_cache_write_tokens=cache_write if type(cache_write) is int and cache_write > 0 else None,
            input_cache_write_5m_tokens=None,
            input_cache_write_1h_tokens=None,
            input_total_tokens=None,
            output_visible_tokens=None,
            output_reasoning_tokens=None,
            output_total_tokens=output_tokens if type(output_tokens) is int and output_tokens > 0 else None,
            total_tokens=consumed if type(consumed) is int and consumed > 0 else None,
            source="legacy_partial",
        )

    @classmethod
    def sum(cls, usages: Iterable[ProviderCallUsage]) -> ProviderCallUsage:
        items = tuple(usages)
        if not items:
            return cls()
        for item in items:
            if type(item) is not cls:
                raise TypeError("usage aggregation requires exact ProviderCallUsage values")
        values: dict[str, int | None] = {}
        for field_name in cls._FIELDS:
            field_values = [getattr(item, field_name) for item in items]
            values[field_name] = (
                None if any(value is None for value in field_values) else sum(field_values)
            )
        sources = {item.source for item in items}
        if "legacy_partial" in sources:
            source = "legacy_partial"
        elif "unavailable" in sources:
            source = "unavailable" if sources == {"unavailable"} else "provider_observed_partial"
        elif sources == {"provider_observed"} and all(values[name] is not None for name in cls._FIELDS):
            source = "provider_observed"
        else:
            source = "provider_observed_partial"
        return cls(**values, source=source)

    def _field_values(self) -> dict[str, int | None]:
        return {field_name: getattr(self, field_name) for field_name in self._FIELDS}

    @staticmethod
    def _observed_source(usage: ProviderCallUsage) -> str:
        if not any(getattr(usage, field_name) is not None for field_name in usage._FIELDS):
            return "unavailable"
        return "provider_observed" if usage.is_disjoint_complete else "provider_observed_partial"


@dataclass(frozen=True, slots=True)
class ProviderCallIdentity:
    execution_id: str
    attempt_id: str
    root_run_id: str
    owner_run_id: str
    parent_run_id: str | None
    iteration: int
    retry_ordinal: int
    purpose: str
    request_sha256: str
    route: str

    def __post_init__(self) -> None:
        for field_name in (
            "execution_id",
            "attempt_id",
            "root_run_id",
            "owner_run_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "parent_run_id",
            _canonical_text(self.parent_run_id, "parent_run_id", optional=True),
        )
        object.__setattr__(self, "iteration", _count(self.iteration, "iteration"))
        object.__setattr__(
            self,
            "retry_ordinal",
            _count(self.retry_ordinal, "retry_ordinal"),
        )
        object.__setattr__(self, "purpose", _slug(self.purpose, "purpose"))
        object.__setattr__(
            self,
            "request_sha256",
            _sha256(self.request_sha256, "request_sha256"),
        )
        object.__setattr__(self, "route", _slug(self.route, "route"))
        if self.owner_run_id == self.root_run_id and self.parent_run_id is not None:
            raise RunBundleProtocolError("root provider calls cannot have parent_run_id")
        if self.owner_run_id != self.root_run_id and self.parent_run_id is None:
            raise RunBundleProtocolError("child provider calls require parent_run_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "attempt_id": self.attempt_id,
            "root_run_id": self.root_run_id,
            "owner_run_id": self.owner_run_id,
            "parent_run_id": self.parent_run_id,
            "iteration": self.iteration,
            "retry_ordinal": self.retry_ordinal,
            "purpose": self.purpose,
            "request_sha256": self.request_sha256,
            "route": self.route,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCallIdentity:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "execution_id",
                    "attempt_id",
                    "root_run_id",
                    "owner_run_id",
                    "parent_run_id",
                    "iteration",
                    "retry_ordinal",
                    "purpose",
                    "request_sha256",
                    "route",
                }
            ),
            field_name="provider call identity",
        )
        return cls(**raw)


def deterministic_provider_call_id(
    *,
    identity: ProviderCallIdentity,
    provider: str,
    model: str,
) -> str:
    if type(identity) is not ProviderCallIdentity:
        raise TypeError("identity must be an exact ProviderCallIdentity")
    provider_name = _slug(provider, "provider")
    model_name = _canonical_text(model, "model", maximum=256)
    assert isinstance(model_name, str)
    digest = canonical_sha256(
        {
            "domain": "unchain.provider_call_id.v1",
            "identity": identity.to_dict(),
            "provider": provider_name,
            "model": model_name,
        }
    )
    return f"pc_{digest}"


@dataclass(frozen=True, slots=True)
class ProviderCallTiming:
    started_at: str | None = None
    completed_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "started_at",
            _timestamp(self.started_at, "provider call started_at"),
        )
        object.__setattr__(
            self,
            "completed_at",
            _timestamp(self.completed_at, "provider call completed_at"),
        )
        if self.completed_at is not None and self.started_at is None:
            raise RunBundleProtocolError(
                "provider call completed_at requires started_at"
            )
        if (
            self.started_at is not None
            and self.completed_at is not None
            and _timestamp_instant(self.completed_at)
            < _timestamp_instant(self.started_at)
        ):
            raise RunBundleProtocolError(
                "provider call completed_at precedes started_at"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCallTiming:
        return cls(
            **_strict_record(
                value,
                keys=frozenset({"started_at", "completed_at"}),
                field_name="provider call timing",
            )
        )


@dataclass(frozen=True, slots=True)
class ProviderCallIds:
    request_id_sha256: str | None = None
    response_id_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id_sha256",
            _optional_sha256(
                self.request_id_sha256,
                "provider request_id_sha256",
            ),
        )
        object.__setattr__(
            self,
            "response_id_sha256",
            _optional_sha256(
                self.response_id_sha256,
                "provider response_id_sha256",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id_sha256": self.request_id_sha256,
            "response_id_sha256": self.response_id_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCallIds:
        return cls(
            **_strict_record(
                value,
                keys=frozenset({"request_id_sha256", "response_id_sha256"}),
                field_name="provider call ids",
            )
        )


@dataclass(frozen=True, slots=True)
class ProviderBillingDimensions:
    billing_surface: str | None = None
    batch: bool | None = None
    inference_geo: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "billing_surface",
            (
                _slug(self.billing_surface, "billing_surface")
                if self.billing_surface is not None
                else None
            ),
        )
        if self.batch is not None and type(self.batch) is not bool:
            raise TypeError("billing batch must be an exact boolean or null")
        object.__setattr__(
            self,
            "inference_geo",
            (
                _slug(self.inference_geo, "inference_geo")
                if self.inference_geo is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "billing_surface": self.billing_surface,
            "batch": self.batch,
            "inference_geo": self.inference_geo,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderBillingDimensions:
        return cls(
            **_strict_record(
                value,
                keys=frozenset({"billing_surface", "batch", "inference_geo"}),
                field_name="provider billing dimensions",
            )
        )


@dataclass(frozen=True, slots=True)
class PricingSnapshotRef:
    catalog_version: str
    catalog_sha256: str
    source_url: str
    source_sha256: str
    effective_from: str
    provider: str
    billing_surface: str
    model: str
    service_tier: str
    batch: bool
    inference_geo: str
    effective_until: str | None = None
    currency: str = "USD"
    input_uncached_nano_usd_per_million: int | None = None
    input_cache_read_nano_usd_per_million: int | None = None
    input_cache_write_nano_usd_per_million: int | None = None
    input_cache_write_5m_nano_usd_per_million: int | None = None
    input_cache_write_1h_nano_usd_per_million: int | None = None
    output_nano_usd_per_million: int | None = None
    long_context_threshold_input_tokens: int | None = None
    long_context_input_multiplier_ppm: int | None = None
    long_context_output_multiplier_ppm: int | None = None
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "catalog_version",
            _canonical_text(self.catalog_version, "catalog_version", maximum=256),
        )
        object.__setattr__(
            self,
            "catalog_sha256",
            _sha256(self.catalog_sha256, "catalog_sha256"),
        )
        source_url = _canonical_text(self.source_url, "source_url", maximum=2_048)
        assert isinstance(source_url, str)
        if not source_url.startswith("https://"):
            raise RunBundleProtocolError("source_url must use HTTPS")
        object.__setattr__(self, "source_url", source_url)
        object.__setattr__(
            self,
            "source_sha256",
            _sha256(self.source_sha256, "source_sha256"),
        )
        object.__setattr__(
            self,
            "effective_from",
            _timestamp(self.effective_from, "effective_from", optional=False),
        )
        object.__setattr__(
            self,
            "effective_until",
            _timestamp(self.effective_until, "effective_until"),
        )
        if (
            self.effective_until is not None
            and _timestamp_instant(self.effective_until)
            <= _timestamp_instant(self.effective_from)
        ):
            raise RunBundleProtocolError("effective_until must follow effective_from")
        if self.currency != "USD":
            raise RunBundleProtocolError("pricing snapshot currency must be USD")
        object.__setattr__(self, "provider", _slug(self.provider, "provider"))
        object.__setattr__(
            self,
            "billing_surface",
            _slug(self.billing_surface, "billing_surface"),
        )
        object.__setattr__(
            self,
            "model",
            _canonical_text(self.model, "model", maximum=256),
        )
        object.__setattr__(
            self,
            "service_tier",
            _slug(self.service_tier, "service_tier"),
        )
        if type(self.batch) is not bool:
            raise TypeError("batch must be an exact boolean")
        object.__setattr__(
            self,
            "inference_geo",
            _slug(self.inference_geo, "inference_geo"),
        )
        for field_name in (
            "input_uncached_nano_usd_per_million",
            "input_cache_read_nano_usd_per_million",
            "input_cache_write_nano_usd_per_million",
            "input_cache_write_5m_nano_usd_per_million",
            "input_cache_write_1h_nano_usd_per_million",
            "output_nano_usd_per_million",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_count(getattr(self, field_name), field_name),
            )
        long_context_values = (
            self.long_context_threshold_input_tokens,
            self.long_context_input_multiplier_ppm,
            self.long_context_output_multiplier_ppm,
        )
        if any(value is not None for value in long_context_values):
            if any(value is None for value in long_context_values):
                raise RunBundleProtocolError(
                    "long-context pricing rule must be wholly present or null"
                )
            for field_name in (
                "long_context_threshold_input_tokens",
                "long_context_input_multiplier_ppm",
                "long_context_output_multiplier_ppm",
            ):
                value = _count(getattr(self, field_name), field_name)
                if value <= 0:
                    raise RunBundleProtocolError(
                        f"{field_name} must be positive when present"
                    )
                object.__setattr__(self, field_name, value)
        expected_id = f"price_{canonical_sha256(self._body_dict())}"
        if self.snapshot_id:
            observed_id = _canonical_text(
                self.snapshot_id, "snapshot_id", maximum=256
            )
            if observed_id != expected_id:
                raise RunBundleProtocolError(
                    "snapshot_id does not match the immutable pricing snapshot"
                )
        object.__setattr__(self, "snapshot_id", expected_id)

    def _body_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "catalog_sha256": self.catalog_sha256,
            "source_url": self.source_url,
            "source_sha256": self.source_sha256,
            "effective_from": self.effective_from,
            "effective_until": self.effective_until,
            "currency": self.currency,
            "provider": self.provider,
            "billing_surface": self.billing_surface,
            "model": self.model,
            "service_tier": self.service_tier,
            "batch": self.batch,
            "inference_geo": self.inference_geo,
            "rates": {
                "input_uncached_nano_usd_per_million": self.input_uncached_nano_usd_per_million,
                "input_cache_read_nano_usd_per_million": self.input_cache_read_nano_usd_per_million,
                "input_cache_write_nano_usd_per_million": self.input_cache_write_nano_usd_per_million,
                "input_cache_write_5m_nano_usd_per_million": self.input_cache_write_5m_nano_usd_per_million,
                "input_cache_write_1h_nano_usd_per_million": self.input_cache_write_1h_nano_usd_per_million,
                "output_nano_usd_per_million": self.output_nano_usd_per_million,
            },
            "long_context_rule": (
                {
                    "threshold_input_tokens": self.long_context_threshold_input_tokens,
                    "input_multiplier_ppm": self.long_context_input_multiplier_ppm,
                    "output_multiplier_ppm": self.long_context_output_multiplier_ppm,
                }
                if self.long_context_threshold_input_tokens is not None
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"snapshot_id": self.snapshot_id, **self._body_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PricingSnapshotRef:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "snapshot_id",
                    "catalog_version",
                    "catalog_sha256",
                    "source_url",
                    "source_sha256",
                    "effective_from",
                    "effective_until",
                    "currency",
                    "provider",
                    "billing_surface",
                    "model",
                    "service_tier",
                    "batch",
                    "inference_geo",
                    "rates",
                    "long_context_rule",
                }
            ),
            field_name="pricing snapshot",
        )
        rates = _strict_record(
            raw["rates"],
            keys=frozenset(
                {
                    "input_uncached_nano_usd_per_million",
                    "input_cache_read_nano_usd_per_million",
                    "input_cache_write_nano_usd_per_million",
                    "input_cache_write_5m_nano_usd_per_million",
                    "input_cache_write_1h_nano_usd_per_million",
                    "output_nano_usd_per_million",
                }
            ),
            field_name="pricing rates",
        )
        long_context_rule = raw["long_context_rule"]
        if long_context_rule is None:
            long_context = {
                "long_context_threshold_input_tokens": None,
                "long_context_input_multiplier_ppm": None,
                "long_context_output_multiplier_ppm": None,
            }
        else:
            rule = _strict_record(
                long_context_rule,
                keys=frozenset(
                    {
                        "threshold_input_tokens",
                        "input_multiplier_ppm",
                        "output_multiplier_ppm",
                    }
                ),
                field_name="long-context pricing rule",
            )
            long_context = {
                "long_context_threshold_input_tokens": rule["threshold_input_tokens"],
                "long_context_input_multiplier_ppm": rule["input_multiplier_ppm"],
                "long_context_output_multiplier_ppm": rule["output_multiplier_ppm"],
            }
        return cls(
            snapshot_id=raw["snapshot_id"],
            catalog_version=raw["catalog_version"],
            catalog_sha256=raw["catalog_sha256"],
            source_url=raw["source_url"],
            source_sha256=raw["source_sha256"],
            effective_from=raw["effective_from"],
            effective_until=raw["effective_until"],
            currency=raw["currency"],
            provider=raw["provider"],
            billing_surface=raw["billing_surface"],
            model=raw["model"],
            service_tier=raw["service_tier"],
            batch=raw["batch"],
            inference_geo=raw["inference_geo"],
            **rates,
            **long_context,
        )


@dataclass(frozen=True, slots=True)
class ProviderCallPricing:
    status: str = "unavailable"
    basis: str | None = None
    snapshot: PricingSnapshotRef | None = None
    amount_nano_usd: int | None = None
    reason: str | None = "pricing_snapshot_unavailable"
    input_multiplier_ppm: int | None = None
    output_multiplier_ppm: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "pricing status", _PRICING_STATUSES),
        )
        if self.snapshot is not None and type(self.snapshot) is not PricingSnapshotRef:
            if type(self.snapshot) is dict:
                object.__setattr__(
                    self,
                    "snapshot",
                    PricingSnapshotRef.from_dict(self.snapshot),
                )
            else:
                raise TypeError("snapshot must be an exact PricingSnapshotRef or null")
        object.__setattr__(
            self,
            "amount_nano_usd",
            _optional_count(self.amount_nano_usd, "amount_nano_usd"),
        )
        for field_name in ("input_multiplier_ppm", "output_multiplier_ppm"):
            object.__setattr__(
                self,
                field_name,
                _optional_count(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "reason",
            _canonical_text(self.reason, "pricing reason", optional=True, maximum=512),
        )
        if self.status == "estimated":
            if (
                self.basis != "list_price_estimate"
                or self.snapshot is None
                or self.amount_nano_usd is None
                or self.reason is not None
                or self.input_multiplier_ppm is None
                or self.input_multiplier_ppm <= 0
                or self.output_multiplier_ppm is None
                or self.output_multiplier_ppm <= 0
            ):
                raise RunBundleProtocolError(
                    "estimated pricing requires list-price basis, snapshot, amount, and null reason"
                )
        else:
            if (
                self.basis is not None
                or self.amount_nano_usd is not None
                or self.reason is None
                or self.input_multiplier_ppm is not None
                or self.output_multiplier_ppm is not None
            ):
                raise RunBundleProtocolError(
                    "unavailable pricing requires null basis/amount and a reason"
                )

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        snapshot: PricingSnapshotRef | None = None,
    ) -> ProviderCallPricing:
        return cls(reason=reason, snapshot=snapshot)

    @classmethod
    def estimate(
        cls,
        *,
        snapshot: PricingSnapshotRef,
        usage: ProviderCallUsage,
    ) -> ProviderCallPricing:
        if type(snapshot) is not PricingSnapshotRef:
            raise TypeError("snapshot must be an exact PricingSnapshotRef")
        if type(usage) is not ProviderCallUsage:
            raise TypeError("usage must be an exact ProviderCallUsage")
        pairs: list[tuple[int | None, int | None]] = [
            (
                usage.input_uncached_tokens,
                snapshot.input_uncached_nano_usd_per_million,
            ),
            (
                usage.input_cache_read_tokens,
                snapshot.input_cache_read_nano_usd_per_million,
            ),
        ]
        if (
            usage.input_cache_write_5m_tokens is not None
            and usage.input_cache_write_1h_tokens is not None
        ):
            pairs.extend(
                [
                    (
                        usage.input_cache_write_5m_tokens,
                        snapshot.input_cache_write_5m_nano_usd_per_million,
                    ),
                    (
                        usage.input_cache_write_1h_tokens,
                        snapshot.input_cache_write_1h_nano_usd_per_million,
                    ),
                ]
            )
        else:
            pairs.append(
                (
                    usage.input_cache_write_tokens,
                    snapshot.input_cache_write_nano_usd_per_million,
                )
            )
        pairs.append(
            (usage.output_total_tokens, snapshot.output_nano_usd_per_million)
        )
        if any(tokens is None for tokens, _rate in pairs):
            return cls.unavailable("usage_incomplete_for_pricing", snapshot=snapshot)
        if any(tokens and rate is None for tokens, rate in pairs):
            return cls.unavailable("pricing_rate_unavailable", snapshot=snapshot)
        input_total = usage.input_total_tokens
        if input_total is None:
            return cls.unavailable(
                "input_total_unknown_for_pricing_modifier",
                snapshot=snapshot,
            )
        input_multiplier_ppm = 1_000_000
        output_multiplier_ppm = 1_000_000
        if (
            snapshot.long_context_threshold_input_tokens is not None
            and input_total > snapshot.long_context_threshold_input_tokens
        ):
            assert snapshot.long_context_input_multiplier_ppm is not None
            assert snapshot.long_context_output_multiplier_ppm is not None
            input_multiplier_ppm = snapshot.long_context_input_multiplier_ppm
            output_multiplier_ppm = snapshot.long_context_output_multiplier_ppm
        input_numerator = sum(
            int(tokens or 0) * int(rate or 0) for tokens, rate in pairs[:-1]
        )
        output_tokens, output_rate = pairs[-1]
        output_numerator = int(output_tokens or 0) * int(output_rate or 0)
        adjusted_numerator = (
            input_numerator * input_multiplier_ppm
            + output_numerator * output_multiplier_ppm
        )
        amount = (adjusted_numerator + 500_000_000_000) // 1_000_000_000_000
        return cls(
            status="estimated",
            basis="list_price_estimate",
            snapshot=snapshot,
            amount_nano_usd=amount,
            reason=None,
            input_multiplier_ppm=input_multiplier_ppm,
            output_multiplier_ppm=output_multiplier_ppm,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "basis": self.basis,
            "snapshot": self.snapshot.to_dict() if self.snapshot is not None else None,
            "amount_nano_usd": self.amount_nano_usd,
            "reason": self.reason,
            "input_multiplier_ppm": self.input_multiplier_ppm,
            "output_multiplier_ppm": self.output_multiplier_ppm,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCallPricing:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "status",
                    "basis",
                    "snapshot",
                    "amount_nano_usd",
                    "reason",
                    "input_multiplier_ppm",
                    "output_multiplier_ppm",
                }
            ),
            field_name="provider call pricing",
        )
        snapshot = (
            PricingSnapshotRef.from_dict(raw["snapshot"])
            if raw["snapshot"] is not None
            else None
        )
        return cls(
            status=raw["status"],
            basis=raw["basis"],
            snapshot=snapshot,
            amount_nano_usd=raw["amount_nano_usd"],
            reason=raw["reason"],
            input_multiplier_ppm=raw["input_multiplier_ppm"],
            output_multiplier_ppm=raw["output_multiplier_ppm"],
        )


@dataclass(frozen=True, slots=True)
class ProviderCallReceipt:
    """Immutable accounting receipt for one exact provider network send."""

    SCHEMA: ClassVar[str] = PROVIDER_CALL_USAGE_SCHEMA

    identity: ProviderCallIdentity
    provider: str
    model: str
    status: str
    usage: ProviderCallUsage
    service_tier: str | None = None
    timing: ProviderCallTiming = field(default_factory=ProviderCallTiming)
    provider_ids: ProviderCallIds = field(default_factory=ProviderCallIds)
    billing_dimensions: ProviderBillingDimensions = field(
        default_factory=ProviderBillingDimensions
    )
    raw_usage_sha256: str | None = None
    pricing: ProviderCallPricing = field(default_factory=ProviderCallPricing)
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)
    provider_call_id: str = ""

    def __post_init__(self) -> None:
        if type(self.identity) is not ProviderCallIdentity:
            if type(self.identity) is dict:
                object.__setattr__(
                    self,
                    "identity",
                    ProviderCallIdentity.from_dict(self.identity),
                )
            else:
                raise TypeError("identity must be an exact ProviderCallIdentity")
        object.__setattr__(self, "provider", _slug(self.provider, "provider"))
        object.__setattr__(
            self,
            "model",
            _canonical_text(self.model, "model", maximum=256),
        )
        object.__setattr__(
            self,
            "service_tier",
            (
                _slug(self.service_tier, "service_tier")
                if self.service_tier is not None
                else None
            ),
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "provider call status", _RECEIPT_STATUSES),
        )
        if type(self.usage) is not ProviderCallUsage:
            if type(self.usage) is dict:
                object.__setattr__(
                    self,
                    "usage",
                    ProviderCallUsage.from_dict(self.usage),
                )
            else:
                raise TypeError("usage must be an exact ProviderCallUsage")
        for field_name, expected_type in (
            ("timing", ProviderCallTiming),
            ("provider_ids", ProviderCallIds),
            ("billing_dimensions", ProviderBillingDimensions),
        ):
            value = getattr(self, field_name)
            if type(value) is not expected_type:
                if type(value) is dict:
                    object.__setattr__(
                        self,
                        field_name,
                        expected_type.from_dict(value),
                    )
                else:
                    raise TypeError(
                        f"{field_name} must be an exact {expected_type.__name__}"
                    )
        if self.status in {"completed", "failed"} and (
            self.timing.started_at is None
            or self.timing.completed_at is None
        ):
            raise RunBundleProtocolError(
                "completed/failed provider call timing requires both boundaries"
            )
        object.__setattr__(
            self,
            "raw_usage_sha256",
            _optional_sha256(self.raw_usage_sha256, "raw_usage_sha256"),
        )
        if type(self.pricing) is not ProviderCallPricing:
            if type(self.pricing) is dict:
                object.__setattr__(
                    self,
                    "pricing",
                    ProviderCallPricing.from_dict(self.pricing),
                )
            else:
                raise TypeError("pricing must be an exact ProviderCallPricing")
        if self.pricing.snapshot is not None and (
            self.pricing.snapshot.provider != self.provider
            or self.pricing.snapshot.model != self.model
            or self.pricing.snapshot.service_tier != self.service_tier
            or self.pricing.snapshot.billing_surface
            != self.billing_dimensions.billing_surface
            or self.pricing.snapshot.batch != self.billing_dimensions.batch
            or self.pricing.snapshot.inference_geo
            != self.billing_dimensions.inference_geo
        ):
            raise RunBundleProtocolError(
                "pricing snapshot provider/model/tier does not match the receipt"
            )
        object.__setattr__(self, "extensions", _freeze_extensions(self.extensions))
        expected_id = deterministic_provider_call_id(
            identity=self.identity,
            provider=self.provider,
            model=self.model,
        )
        if self.provider_call_id:
            observed_id = _canonical_text(
                self.provider_call_id, "provider_call_id", maximum=256
            )
            if observed_id != expected_id:
                raise RunBundleProtocolError(
                    "provider_call_id does not match its deterministic identity"
                )
        object.__setattr__(self, "provider_call_id", expected_id)

    @property
    def receipt_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "provider_call_id": self.provider_call_id,
            "identity": self.identity.to_dict(),
            "provider": {
                "name": self.provider,
                "model": self.model,
                "service_tier": self.service_tier,
            },
            "status": self.status,
            "timing": self.timing.to_dict(),
            "provider_ids": self.provider_ids.to_dict(),
            "billing_dimensions": self.billing_dimensions.to_dict(),
            "usage": self.usage.to_dict(),
            "raw_usage_sha256": self.raw_usage_sha256,
            "pricing": self.pricing.to_dict(),
            "extensions": _extensions_dict(self.extensions),
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ProviderCallReceipt:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "schema",
                    "provider_call_id",
                    "identity",
                    "provider",
                    "status",
                    "timing",
                    "provider_ids",
                    "billing_dimensions",
                    "usage",
                    "raw_usage_sha256",
                    "pricing",
                    "extensions",
                }
            ),
            field_name="provider call receipt",
        )
        if raw["schema"] != cls.SCHEMA:
            raise RunBundleProtocolError("provider call receipt schema is unsupported")
        provider = _strict_record(
            raw["provider"],
            keys=frozenset({"name", "model", "service_tier"}),
            field_name="provider identity",
        )
        return cls(
            provider_call_id=raw["provider_call_id"],
            identity=ProviderCallIdentity.from_dict(raw["identity"]),
            provider=provider["name"],
            model=provider["model"],
            service_tier=provider["service_tier"],
            status=raw["status"],
            timing=ProviderCallTiming.from_dict(raw["timing"]),
            provider_ids=ProviderCallIds.from_dict(raw["provider_ids"]),
            billing_dimensions=ProviderBillingDimensions.from_dict(
                raw["billing_dimensions"]
            ),
            usage=ProviderCallUsage.from_dict(raw["usage"]),
            raw_usage_sha256=raw["raw_usage_sha256"],
            pricing=ProviderCallPricing.from_dict(raw["pricing"]),
            extensions=raw["extensions"],
        )

    @classmethod
    def from_model_turn_result(
        cls,
        *,
        identity: ProviderCallIdentity,
        provider: str,
        model: str,
        result: Any,
        status: str = "uncertain",
        service_tier: str | None = None,
        raw_usage_sha256: str | None = None,
        timing: ProviderCallTiming | None = None,
        provider_ids: ProviderCallIds | None = None,
        billing_dimensions: ProviderBillingDimensions | None = None,
        pricing: ProviderCallPricing | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> ProviderCallReceipt:
        from .kernel.types import ModelTurnResult

        if type(result) is not ModelTurnResult:
            raise TypeError("result must be an exact ModelTurnResult")
        provider_usage = getattr(result, "provider_call_usage", None)
        usage = (
            provider_usage
            if type(provider_usage) is ProviderCallUsage
            else ProviderCallUsage.from_legacy_model_turn_result(result)
        )
        return cls(
            identity=identity,
            provider=provider,
            model=model,
            service_tier=service_tier,
            status=status,
            usage=usage,
            timing=timing or ProviderCallTiming(),
            provider_ids=provider_ids
            or ProviderCallIds(
                response_id_sha256=(
                    hashlib.sha256(result.response_id.encode("utf-8")).hexdigest()
                    if isinstance(result.response_id, str) and result.response_id
                    else None
                )
            ),
            billing_dimensions=billing_dimensions
            or ProviderBillingDimensions(),
            raw_usage_sha256=raw_usage_sha256,
            pricing=pricing or ProviderCallPricing(),
            extensions=extensions or {},
        )


@dataclass(frozen=True, slots=True)
class RunIdentity:
    execution_id: str
    attempt_id: str
    root_run_id: str
    run_id: str
    parent_run_id: str | None
    relation: str

    def __post_init__(self) -> None:
        for field_name in ("execution_id", "attempt_id", "root_run_id", "run_id"):
            object.__setattr__(
                self,
                field_name,
                _canonical_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "parent_run_id",
            _canonical_text(self.parent_run_id, "parent_run_id", optional=True),
        )
        object.__setattr__(
            self,
            "relation",
            _enum(self.relation, "run relation", _RUN_RELATIONS),
        )
        if self.relation == "root":
            if self.run_id != self.root_run_id or self.parent_run_id is not None:
                raise RunBundleProtocolError(
                    "root run identity requires run_id=root_run_id and null parent"
                )
        elif self.run_id == self.root_run_id or self.parent_run_id is None:
            raise RunBundleProtocolError(
                "child run identity requires a distinct run_id and parent"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "attempt_id": self.attempt_id,
            "root_run_id": self.root_run_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "relation": self.relation,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunIdentity:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "execution_id",
                    "attempt_id",
                    "root_run_id",
                    "run_id",
                    "parent_run_id",
                    "relation",
                }
            ),
            field_name="run identity",
        )
        return cls(**raw)


def deterministic_bundle_id(*, identity: RunIdentity) -> str:
    if type(identity) is not RunIdentity:
        raise TypeError("identity must be an exact RunIdentity")
    digest = canonical_sha256(
        {"domain": "unchain.run_bundle_id.v1", "identity": identity.to_dict()}
    )
    return f"rb_{digest}"


@dataclass(frozen=True, slots=True)
class RunLifecycle:
    status: str
    started_at: str | None = None
    completed_at: str | None = None
    continued_from_run_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "run lifecycle status", _RUN_STATUSES),
        )
        object.__setattr__(
            self,
            "started_at",
            _timestamp(self.started_at, "started_at"),
        )
        object.__setattr__(
            self,
            "completed_at",
            _timestamp(self.completed_at, "completed_at"),
        )
        object.__setattr__(
            self,
            "continued_from_run_id",
            _canonical_text(
                self.continued_from_run_id,
                "continued_from_run_id",
                optional=True,
            ),
        )
        if self.started_at is None:
            raise RunBundleProtocolError("run lifecycle started_at is required")
        if self.status == "running" and self.completed_at is not None:
            raise RunBundleProtocolError(
                "running lifecycle requires null completed_at"
            )
        if (
            self.status in {"completed", "failed", "suspended", "cancelled"}
            and self.completed_at is None
        ):
            raise RunBundleProtocolError(
                "terminal lifecycle requires completed_at"
            )
        if (
            self.started_at is not None
            and self.completed_at is not None
            and _timestamp_instant(self.completed_at)
            < _timestamp_instant(self.started_at)
        ):
            raise RunBundleProtocolError("completed_at must not precede started_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "continued_from_run_id": self.continued_from_run_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunLifecycle:
        raw = _strict_record(
            value,
            keys=frozenset(
                {"status", "started_at", "completed_at", "continued_from_run_id"}
            ),
            field_name="run lifecycle",
        )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    """Content-free presentation metadata for one producer-owned run."""

    model: str = "unknown-model"
    display_model: str = "model-unavailable"
    active_agent: str = "unknown"
    agent_orchestration: str = "default"
    iteration: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model",
            _canonical_text(self.model, "descriptor model", maximum=256),
        )
        object.__setattr__(
            self,
            "display_model",
            _canonical_text(
                self.display_model,
                "descriptor display_model",
                maximum=256,
            ),
        )
        object.__setattr__(
            self,
            "active_agent",
            _canonical_text(
                self.active_agent,
                "descriptor active_agent",
                maximum=256,
            ),
        )
        object.__setattr__(
            self,
            "agent_orchestration",
            _enum(
                self.agent_orchestration,
                "descriptor agent_orchestration",
                _ORCHESTRATION_MODES,
            ),
        )
        object.__setattr__(self, "iteration", _count(self.iteration, "iteration"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "display_model": self.display_model,
            "active_agent": self.active_agent,
            "agent_orchestration": self.agent_orchestration,
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunDescriptor:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "model",
                    "display_model",
                    "active_agent",
                    "agent_orchestration",
                    "iteration",
                }
            ),
            field_name="run descriptor",
        )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class RunMetricError:
    category: str
    code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _slug(self.category, "error category"))
        object.__setattr__(self, "code", _slug(self.code, "error code"))

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "code": self.code}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetricError:
        return cls(
            **_strict_record(
                value,
                keys=frozenset({"category", "code"}),
                field_name="run metric error",
            )
        )


@dataclass(frozen=True, slots=True)
class RunMetricEvidenceRef:
    kind: str
    ref_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, "metric evidence kind", _METRIC_EVIDENCE_KINDS),
        )
        object.__setattr__(
            self,
            "ref_id",
            _canonical_text(self.ref_id, "metric evidence ref_id", maximum=256),
        )
        if not re.fullmatch(
            rf"{re.escape(self.kind)}_[0-9a-f]{{64}}",
            self.ref_id,
        ):
            raise RunBundleProtocolError(
                "metric evidence ref_id must be a kind-bound opaque sha256 id"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref_id": self.ref_id}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetricEvidenceRef:
        return cls(
            **_strict_record(
                value,
                keys=frozenset({"kind", "ref_id"}),
                field_name="run metric evidence ref",
            )
        )


def opaque_metric_evidence_ref(
    *,
    kind: str,
    source_id: str,
) -> RunMetricEvidenceRef:
    resolved_kind = _enum(kind, "metric evidence kind", _METRIC_EVIDENCE_KINDS)
    resolved_source = _canonical_text(
        source_id,
        "metric evidence source_id",
        maximum=4_096,
    )
    return RunMetricEvidenceRef(
        kind=resolved_kind,
        ref_id=(
            f"{resolved_kind}_"
            f"{canonical_sha256({'domain': 'unchain.metric_evidence_ref.v1', 'kind': resolved_kind, 'source_id': resolved_source})}"
        ),
    )


def deterministic_metric_event_id(
    *,
    execution_id: str,
    attempt_id: str,
    root_run_id: str,
    owner_run_id: str,
    parent_run_id: str | None,
    kind: str,
    subject_id: str,
) -> str:
    """Return the replay-stable identity of one logical metric occurrence."""

    material = {
        "domain": "unchain.metric_event_id.v1",
        "execution_id": _canonical_text(execution_id, "execution_id"),
        "attempt_id": _canonical_text(attempt_id, "attempt_id"),
        "root_run_id": _canonical_text(root_run_id, "root_run_id"),
        "owner_run_id": _canonical_text(owner_run_id, "owner_run_id"),
        "parent_run_id": _canonical_text(
            parent_run_id,
            "parent_run_id",
            optional=True,
        ),
        "kind": _enum(kind, "metric event kind", _METRIC_EVENT_KINDS),
        "subject_id": _canonical_text(subject_id, "metric subject_id", maximum=256),
    }
    return f"me_{canonical_sha256(material)}"


@dataclass(frozen=True, slots=True)
class RunMetricEvent:
    execution_id: str
    attempt_id: str
    root_run_id: str
    owner_run_id: str
    parent_run_id: str | None
    kind: str
    subject_id: str
    outcome: str
    error: RunMetricError | None = None
    evidence_refs: tuple[RunMetricEvidenceRef, ...] = ()
    metric_event_id: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "execution_id",
            "attempt_id",
            "root_run_id",
            "owner_run_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _canonical_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "parent_run_id",
            _canonical_text(self.parent_run_id, "parent_run_id", optional=True),
        )
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, "metric event kind", _METRIC_EVENT_KINDS),
        )
        object.__setattr__(
            self,
            "subject_id",
            _canonical_text(self.subject_id, "metric subject_id", maximum=256),
        )
        object.__setattr__(
            self,
            "outcome",
            _enum(self.outcome, "metric event outcome", _METRIC_EVENT_OUTCOMES),
        )
        if self.error is not None and type(self.error) is not RunMetricError:
            raise TypeError("metric event error must be an exact RunMetricError or null")
        if self.error is not None and self.outcome not in {"failed", "uncertain"}:
            raise RunBundleProtocolError(
                "metric event error requires failed or uncertain outcome"
            )
        evidence_refs = tuple(
            sorted(self.evidence_refs, key=lambda item: (item.kind, item.ref_id))
        )
        if any(type(item) is not RunMetricEvidenceRef for item in evidence_refs):
            raise TypeError(
                "metric event evidence_refs must contain exact RunMetricEvidenceRef values"
            )
        if len(evidence_refs) > _MAX_METRIC_EVIDENCE_REFS:
            raise RunBundleProtocolError("metric event exceeds the evidence ref limit")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise RunBundleProtocolError("metric event evidence refs must be unique")
        object.__setattr__(self, "evidence_refs", evidence_refs)
        expected_id = deterministic_metric_event_id(
            execution_id=self.execution_id,
            attempt_id=self.attempt_id,
            root_run_id=self.root_run_id,
            owner_run_id=self.owner_run_id,
            parent_run_id=self.parent_run_id,
            kind=self.kind,
            subject_id=self.subject_id,
        )
        if self.metric_event_id:
            observed_id = _canonical_text(
                self.metric_event_id,
                "metric_event_id",
                maximum=256,
            )
            if observed_id != expected_id:
                raise RunBundleProtocolError(
                    "metric_event_id does not match its deterministic identity"
                )
        object.__setattr__(self, "metric_event_id", expected_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_event_id": self.metric_event_id,
            "execution_id": self.execution_id,
            "attempt_id": self.attempt_id,
            "root_run_id": self.root_run_id,
            "owner_run_id": self.owner_run_id,
            "parent_run_id": self.parent_run_id,
            "kind": self.kind,
            "subject_id": self.subject_id,
            "outcome": self.outcome,
            "error": self.error.to_dict() if self.error is not None else None,
            "evidence_refs": [item.to_dict() for item in self.evidence_refs],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetricEvent:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "metric_event_id",
                    "execution_id",
                    "attempt_id",
                    "root_run_id",
                    "owner_run_id",
                    "parent_run_id",
                    "kind",
                    "subject_id",
                    "outcome",
                    "error",
                    "evidence_refs",
                }
            ),
            field_name="run metric event",
        )
        if type(raw["evidence_refs"]) is not list:
            raise TypeError("metric event evidence_refs must be an exact array")
        return cls(
            metric_event_id=raw["metric_event_id"],
            execution_id=raw["execution_id"],
            attempt_id=raw["attempt_id"],
            root_run_id=raw["root_run_id"],
            owner_run_id=raw["owner_run_id"],
            parent_run_id=raw["parent_run_id"],
            kind=raw["kind"],
            subject_id=raw["subject_id"],
            outcome=raw["outcome"],
            error=(
                RunMetricError.from_dict(raw["error"])
                if raw["error"] is not None
                else None
            ),
            evidence_refs=tuple(
                RunMetricEvidenceRef.from_dict(item)
                for item in raw["evidence_refs"]
            ),
        )


@dataclass(frozen=True, slots=True)
class RunMetricCounters:
    artifacts: int = 0
    model_attempts: int = 0
    iterations: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    interactions: int = 0
    context_builds: int = 0
    context_compactions: int = 0
    errors: int = 0

    def __post_init__(self) -> None:
        for field_name in (
            "artifacts",
            "model_attempts",
            "iterations",
            "tool_calls",
            "tool_results",
            "interactions",
            "context_builds",
            "context_compactions",
            "errors",
        ):
            object.__setattr__(
                self,
                field_name,
                _count(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifacts": self.artifacts,
            "model_attempts": self.model_attempts,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "interactions": self.interactions,
            "context_builds": self.context_builds,
            "context_compactions": self.context_compactions,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetricCounters:
        return cls(
            **_strict_record(
                value,
                keys=frozenset(
                    {
                        "artifacts",
                        "model_attempts",
                        "iterations",
                        "tool_calls",
                        "tool_results",
                        "interactions",
                        "context_builds",
                        "context_compactions",
                        "errors",
                    }
                ),
                field_name="run metric counters",
            )
        )

    @classmethod
    def from_events(cls, events: Iterable[RunMetricEvent]) -> RunMetricCounters:
        names = {
            "artifact": "artifacts",
            "model_attempt": "model_attempts",
            "iteration": "iterations",
            "tool_call": "tool_calls",
            "tool_result": "tool_results",
            "interaction": "interactions",
            "context_build": "context_builds",
            "context_compaction": "context_compactions",
            "error": "errors",
        }
        counts = {field_name: 0 for field_name in names.values()}
        for event in events:
            if type(event) is not RunMetricEvent:
                raise TypeError("metrics events must contain exact RunMetricEvent values")
            counts[names[event.kind]] += 1
        return cls(**counts)


@dataclass(frozen=True, slots=True)
class RunMetrics:
    algorithm: str
    events: tuple[RunMetricEvent, ...]
    direct: RunMetricCounters
    descendant: RunMetricCounters
    all: RunMetricCounters

    def __post_init__(self) -> None:
        if self.algorithm != METRIC_EVENT_SET_UNION_ALGORITHM:
            raise RunBundleProtocolError("run metrics algorithm is unsupported")
        events = tuple(sorted(self.events, key=lambda item: item.metric_event_id))
        if len(events) > _MAX_METRIC_EVENTS:
            raise RunBundleProtocolError("run bundle exceeds the metric event limit")
        if any(type(item) is not RunMetricEvent for item in events):
            raise TypeError("metrics events must contain exact RunMetricEvent values")
        if len({item.metric_event_id for item in events}) != len(events):
            raise RunBundleProtocolError("metric event ids must be unique")
        object.__setattr__(self, "events", events)
        for field_name in ("direct", "descendant", "all"):
            if type(getattr(self, field_name)) is not RunMetricCounters:
                raise TypeError(f"metrics {field_name} must be exact counters")
        if self.all != RunMetricCounters.from_events(events):
            raise RunBundleProtocolError("all metric counters disagree with events")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "events": [item.to_dict() for item in self.events],
            "direct": self.direct.to_dict(),
            "descendant": self.descendant.to_dict(),
            "all": self.all.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunMetrics:
        raw = _strict_record(
            value,
            keys=frozenset({"algorithm", "events", "direct", "descendant", "all"}),
            field_name="run metrics",
        )
        if type(raw["events"]) is not list:
            raise TypeError("run metrics events must be an exact array")
        return cls(
            algorithm=raw["algorithm"],
            events=tuple(RunMetricEvent.from_dict(item) for item in raw["events"]),
            direct=RunMetricCounters.from_dict(raw["direct"]),
            descendant=RunMetricCounters.from_dict(raw["descendant"]),
            all=RunMetricCounters.from_dict(raw["all"]),
        )

    @classmethod
    def from_events(
        cls,
        events: Iterable[RunMetricEvent],
        *,
        direct_run_id: str,
    ) -> RunMetrics:
        ordered = tuple(sorted(events, key=lambda item: item.metric_event_id))
        direct = tuple(item for item in ordered if item.owner_run_id == direct_run_id)
        descendant = tuple(
            item for item in ordered if item.owner_run_id != direct_run_id
        )
        return cls(
            algorithm=METRIC_EVENT_SET_UNION_ALGORITHM,
            events=ordered,
            direct=RunMetricCounters.from_events(direct),
            descendant=RunMetricCounters.from_events(descendant),
            all=RunMetricCounters.from_events(ordered),
        )


@dataclass(frozen=True, slots=True)
class RunChild:
    run_id: str
    attempt_id: str
    parent_run_id: str
    relation: str
    bundle_id: str | None
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _canonical_text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "attempt_id",
            _canonical_text(self.attempt_id, "attempt_id"),
        )
        object.__setattr__(
            self,
            "parent_run_id",
            _canonical_text(self.parent_run_id, "parent_run_id"),
        )
        object.__setattr__(
            self,
            "relation",
            _enum(self.relation, "child relation", _CHILD_RELATIONS),
        )
        object.__setattr__(
            self,
            "bundle_id",
            _canonical_text(self.bundle_id, "bundle_id", optional=True, maximum=256),
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "child status", _RUN_STATUSES),
        )
        if self.run_id == self.parent_run_id:
            raise RunBundleProtocolError("child run cannot be its own parent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "parent_run_id": self.parent_run_id,
            "relation": self.relation,
            "bundle_id": self.bundle_id,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunChild:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "run_id",
                    "attempt_id",
                    "parent_run_id",
                    "relation",
                    "bundle_id",
                    "status",
                }
            ),
            field_name="run child",
        )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class UsageCoverage:
    status: str
    receipt_count: int
    observed_usage_count: int
    missing_usage_count: int
    uncertain_call_count: int
    missing_usage_call_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "coverage status", _COVERAGE_STATUSES),
        )
        for field_name in (
            "receipt_count",
            "observed_usage_count",
            "missing_usage_count",
            "uncertain_call_count",
        ):
            object.__setattr__(
                self,
                field_name,
                _count(getattr(self, field_name), field_name),
            )
        missing_ids = tuple(
            sorted(
                _canonical_text(value, "missing usage call id", maximum=256)
                for value in self.missing_usage_call_ids
            )
        )
        if len(set(missing_ids)) != len(missing_ids):
            raise RunBundleProtocolError("missing usage call ids must be unique")
        object.__setattr__(self, "missing_usage_call_ids", missing_ids)
        if self.receipt_count != self.observed_usage_count + self.missing_usage_count:
            raise RunBundleProtocolError(
                "coverage observed and missing counts must partition receipts"
            )
        if self.missing_usage_count != len(missing_ids):
            raise RunBundleProtocolError(
                "missing usage count must match missing call ids"
            )
        if self.uncertain_call_count > self.receipt_count:
            raise RunBundleProtocolError("uncertain call count exceeds receipts")
        expected_status = (
            "unavailable"
            if self.observed_usage_count == 0
            else ("complete" if self.missing_usage_count == 0 else "partial")
        )
        if self.status != expected_status:
            raise RunBundleProtocolError("coverage status disagrees with its counts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "receipt_count": self.receipt_count,
            "observed_usage_count": self.observed_usage_count,
            "missing_usage_count": self.missing_usage_count,
            "uncertain_call_count": self.uncertain_call_count,
            "missing_usage_call_ids": list(self.missing_usage_call_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UsageCoverage:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "status",
                    "receipt_count",
                    "observed_usage_count",
                    "missing_usage_count",
                    "uncertain_call_count",
                    "missing_usage_call_ids",
                }
            ),
            field_name="usage coverage",
        )
        if type(raw["missing_usage_call_ids"]) is not list:
            raise TypeError("missing_usage_call_ids must be an exact array")
        return cls(
            status=raw["status"],
            receipt_count=raw["receipt_count"],
            observed_usage_count=raw["observed_usage_count"],
            missing_usage_count=raw["missing_usage_count"],
            uncertain_call_count=raw["uncertain_call_count"],
            missing_usage_call_ids=tuple(raw["missing_usage_call_ids"]),
        )

    @classmethod
    def from_receipts(
        cls, receipts: Iterable[ProviderCallReceipt]
    ) -> UsageCoverage:
        items = tuple(receipts)
        observed = tuple(
            receipt
            for receipt in items
            if receipt.status == "completed" and receipt.usage.has_observed_total
        )
        missing = tuple(
            receipt for receipt in items if receipt not in observed
        )
        status = (
            "unavailable"
            if not observed
            else ("complete" if not missing else "partial")
        )
        return cls(
            status=status,
            receipt_count=len(items),
            observed_usage_count=len(observed),
            missing_usage_count=len(missing),
            uncertain_call_count=sum(
                receipt.status == "uncertain" for receipt in items
            ),
            missing_usage_call_ids=tuple(
                receipt.provider_call_id for receipt in missing
            ),
        )


@dataclass(frozen=True, slots=True)
class RunCost:
    status: str
    basis: str | None
    amount_nano_usd: int | None
    currency: str | None
    pricing_snapshot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "cost status", _COST_STATUSES),
        )
        object.__setattr__(
            self,
            "amount_nano_usd",
            _optional_count(self.amount_nano_usd, "amount_nano_usd"),
        )
        snapshot_ids = tuple(
            sorted(
                _canonical_text(value, "pricing snapshot id", maximum=256)
                for value in self.pricing_snapshot_ids
            )
        )
        if len(set(snapshot_ids)) != len(snapshot_ids):
            raise RunBundleProtocolError("pricing snapshot ids must be unique")
        object.__setattr__(self, "pricing_snapshot_ids", snapshot_ids)
        if self.status in {"estimated", "partial"}:
            if (
                self.basis != "list_price_estimate"
                or self.amount_nano_usd is None
                or self.currency != "USD"
            ):
                raise RunBundleProtocolError(
                    "estimated/partial cost requires list-price basis, amount, and USD"
                )
        elif self.basis is not None or self.amount_nano_usd is not None or self.currency is not None:
            raise RunBundleProtocolError(
                "unavailable cost requires null basis, amount, and currency"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "basis": self.basis,
            "amount_nano_usd": self.amount_nano_usd,
            "currency": self.currency,
            "pricing_snapshot_ids": list(self.pricing_snapshot_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunCost:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "status",
                    "basis",
                    "amount_nano_usd",
                    "currency",
                    "pricing_snapshot_ids",
                }
            ),
            field_name="run cost",
        )
        if type(raw["pricing_snapshot_ids"]) is not list:
            raise TypeError("pricing_snapshot_ids must be an exact array")
        return cls(
            status=raw["status"],
            basis=raw["basis"],
            amount_nano_usd=raw["amount_nano_usd"],
            currency=raw["currency"],
            pricing_snapshot_ids=tuple(raw["pricing_snapshot_ids"]),
        )

    @classmethod
    def from_receipts(cls, receipts: Iterable[ProviderCallReceipt]) -> RunCost:
        items = tuple(receipts)
        estimated = tuple(
            receipt for receipt in items if receipt.pricing.status == "estimated"
        )
        snapshot_ids = tuple(
            sorted(
                {
                    receipt.pricing.snapshot.snapshot_id
                    for receipt in items
                    if receipt.pricing.snapshot is not None
                }
            )
        )
        if not estimated:
            return cls(
                status="unavailable",
                basis=None,
                amount_nano_usd=None,
                currency=None,
                pricing_snapshot_ids=snapshot_ids,
            )
        return cls(
            status="estimated" if len(estimated) == len(items) else "partial",
            basis="list_price_estimate",
            amount_nano_usd=sum(
                int(receipt.pricing.amount_nano_usd or 0) for receipt in estimated
            ),
            currency="USD",
            pricing_snapshot_ids=snapshot_ids,
        )


@dataclass(frozen=True, slots=True)
class UsageSlice:
    provider: str
    model: str
    service_tier: str | None
    call_ids: tuple[str, ...]
    usage: ProviderCallUsage
    coverage: UsageCoverage
    cost: RunCost

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _slug(self.provider, "provider"))
        object.__setattr__(
            self,
            "model",
            _canonical_text(self.model, "model", maximum=256),
        )
        object.__setattr__(
            self,
            "service_tier",
            (
                _slug(self.service_tier, "service_tier")
                if self.service_tier is not None
                else None
            ),
        )
        call_ids = tuple(
            sorted(
                _canonical_text(value, "provider call id", maximum=256)
                for value in self.call_ids
            )
        )
        if not call_ids or len(set(call_ids)) != len(call_ids):
            raise RunBundleProtocolError(
                "usage slice call ids must be non-empty and unique"
            )
        object.__setattr__(self, "call_ids", call_ids)
        if type(self.usage) is not ProviderCallUsage:
            raise TypeError("slice usage must be an exact ProviderCallUsage")
        if type(self.coverage) is not UsageCoverage:
            raise TypeError("slice coverage must be an exact UsageCoverage")
        if type(self.cost) is not RunCost:
            raise TypeError("slice cost must be an exact RunCost")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "service_tier": self.service_tier,
            "call_ids": list(self.call_ids),
            "usage": self.usage.to_dict(),
            "coverage": self.coverage.to_dict(),
            "cost": self.cost.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> UsageSlice:
        raw = _strict_record(
            value,
            keys=frozenset(
                {"provider", "model", "service_tier", "call_ids", "usage", "coverage", "cost"}
            ),
            field_name="usage slice",
        )
        if type(raw["call_ids"]) is not list:
            raise TypeError("usage slice call_ids must be an exact array")
        return cls(
            provider=raw["provider"],
            model=raw["model"],
            service_tier=raw["service_tier"],
            call_ids=tuple(raw["call_ids"]),
            usage=ProviderCallUsage.from_dict(raw["usage"]),
            coverage=UsageCoverage.from_dict(raw["coverage"]),
            cost=RunCost.from_dict(raw["cost"]),
        )


@dataclass(frozen=True, slots=True)
class RunAggregation:
    algorithm: str
    direct_call_ids: tuple[str, ...]
    descendant_call_ids: tuple[str, ...]
    all_call_ids: tuple[str, ...]
    direct_usage: ProviderCallUsage
    descendant_usage: ProviderCallUsage
    all_usage: ProviderCallUsage

    def __post_init__(self) -> None:
        if self.algorithm != PROVIDER_CALL_SET_UNION_ALGORITHM:
            raise RunBundleProtocolError("run aggregation algorithm is unsupported")
        for field_name in (
            "direct_call_ids",
            "descendant_call_ids",
            "all_call_ids",
        ):
            values = tuple(
                sorted(
                    _canonical_text(value, "provider call id", maximum=256)
                    for value in getattr(self, field_name)
                )
            )
            if len(set(values)) != len(values):
                raise RunBundleProtocolError(f"{field_name} must be unique")
            object.__setattr__(self, field_name, values)
        if set(self.direct_call_ids) & set(self.descendant_call_ids):
            raise RunBundleProtocolError("direct and descendant call sets overlap")
        if set(self.all_call_ids) != set(self.direct_call_ids) | set(self.descendant_call_ids):
            raise RunBundleProtocolError(
                "all call ids must be the direct/descendant set union"
            )
        for field_name in ("direct_usage", "descendant_usage", "all_usage"):
            if type(getattr(self, field_name)) is not ProviderCallUsage:
                raise TypeError(f"{field_name} must be an exact ProviderCallUsage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "direct_call_ids": list(self.direct_call_ids),
            "descendant_call_ids": list(self.descendant_call_ids),
            "all_call_ids": list(self.all_call_ids),
            "direct_usage": self.direct_usage.to_dict(),
            "descendant_usage": self.descendant_usage.to_dict(),
            "all_usage": self.all_usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunAggregation:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "algorithm",
                    "direct_call_ids",
                    "descendant_call_ids",
                    "all_call_ids",
                    "direct_usage",
                    "descendant_usage",
                    "all_usage",
                }
            ),
            field_name="run aggregation",
        )
        for field_name in ("direct_call_ids", "descendant_call_ids", "all_call_ids"):
            if type(raw[field_name]) is not list:
                raise TypeError(f"{field_name} must be an exact array")
        return cls(
            algorithm=raw["algorithm"],
            direct_call_ids=tuple(raw["direct_call_ids"]),
            descendant_call_ids=tuple(raw["descendant_call_ids"]),
            all_call_ids=tuple(raw["all_call_ids"]),
            direct_usage=ProviderCallUsage.from_dict(raw["direct_usage"]),
            descendant_usage=ProviderCallUsage.from_dict(raw["descendant_usage"]),
            all_usage=ProviderCallUsage.from_dict(raw["all_usage"]),
        )


@dataclass(frozen=True, slots=True)
class LegacyAttribution:
    status: str = "canonical"
    source: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "status",
            _enum(self.status, "legacy status", _LEGACY_STATUSES),
        )
        object.__setattr__(
            self,
            "source",
            _canonical_text(self.source, "legacy source", optional=True, maximum=256),
        )
        object.__setattr__(
            self,
            "reason",
            _canonical_text(self.reason, "legacy reason", optional=True, maximum=512),
        )
        if self.status == "canonical" and (self.source is not None or self.reason is not None):
            raise RunBundleProtocolError(
                "canonical attribution requires null source and reason"
            )
        if self.status == "legacy_partial" and (self.source is None or self.reason is None):
            raise RunBundleProtocolError(
                "legacy_partial attribution requires source and reason"
            )

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "source": self.source, "reason": self.reason}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LegacyAttribution:
        raw = _strict_record(
            value,
            keys=frozenset({"status", "source", "reason"}),
            field_name="legacy attribution",
        )
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class RunEvidence:
    receipt_sha256s: tuple[str, ...] = ()
    raw_usage_sha256s: tuple[str, ...] = ()
    pricing_snapshot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        receipt_sha256s = tuple(sorted(_sha256(value, "receipt sha256") for value in self.receipt_sha256s))
        raw_usage_sha256s = tuple(sorted(_sha256(value, "raw usage sha256") for value in self.raw_usage_sha256s))
        pricing_snapshot_ids = tuple(
            sorted(
                _canonical_text(value, "pricing snapshot id", maximum=256)
                for value in self.pricing_snapshot_ids
            )
        )
        for field_name, values in (
            ("receipt_sha256s", receipt_sha256s),
            ("raw_usage_sha256s", raw_usage_sha256s),
            ("pricing_snapshot_ids", pricing_snapshot_ids),
        ):
            if len(set(values)) != len(values):
                raise RunBundleProtocolError(f"{field_name} must be unique")
            object.__setattr__(self, field_name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_sha256s": list(self.receipt_sha256s),
            "raw_usage_sha256s": list(self.raw_usage_sha256s),
            "pricing_snapshot_ids": list(self.pricing_snapshot_ids),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunEvidence:
        raw = _strict_record(
            value,
            keys=frozenset(
                {"receipt_sha256s", "raw_usage_sha256s", "pricing_snapshot_ids"}
            ),
            field_name="run evidence",
        )
        if any(type(raw[field_name]) is not list for field_name in raw):
            raise TypeError("run evidence members must be exact arrays")
        return cls(
            receipt_sha256s=tuple(raw["receipt_sha256s"]),
            raw_usage_sha256s=tuple(raw["raw_usage_sha256s"]),
            pricing_snapshot_ids=tuple(raw["pricing_snapshot_ids"]),
        )


@dataclass(frozen=True, slots=True)
class RunBundle:
    """A deterministic, renderer-safe projection of a complete run tree."""

    SCHEMA: ClassVar[str] = RUN_BUNDLE_SCHEMA

    identity: RunIdentity
    lifecycle: RunLifecycle
    descriptor: RunDescriptor
    metrics: RunMetrics
    provider_calls: tuple[ProviderCallReceipt, ...]
    children: tuple[RunChild, ...]
    aggregation: RunAggregation
    usage_slices: tuple[UsageSlice, ...]
    coverage: UsageCoverage
    cost: RunCost
    legacy: LegacyAttribution
    evidence: RunEvidence
    revision: int = 1
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False)
    bundle_id: str = ""
    bundle_digest: str = ""

    def __post_init__(self) -> None:
        if type(self.identity) is not RunIdentity:
            raise TypeError("identity must be an exact RunIdentity")
        if type(self.lifecycle) is not RunLifecycle:
            raise TypeError("lifecycle must be an exact RunLifecycle")
        if type(self.descriptor) is not RunDescriptor:
            raise TypeError("descriptor must be an exact RunDescriptor")
        if type(self.metrics) is not RunMetrics:
            raise TypeError("metrics must be exact RunMetrics")
        object.__setattr__(self, "revision", _positive_revision(self.revision))
        provider_calls = tuple(sorted(self.provider_calls, key=lambda item: item.provider_call_id))
        if len(provider_calls) > _MAX_RECEIPTS:
            raise RunBundleProtocolError("run bundle exceeds the provider receipt limit")
        if any(type(item) is not ProviderCallReceipt for item in provider_calls):
            raise TypeError("provider_calls must contain exact ProviderCallReceipt values")
        if len({item.provider_call_id for item in provider_calls}) != len(provider_calls):
            raise RunBundleProtocolError("provider call ids must be unique")
        object.__setattr__(self, "provider_calls", provider_calls)
        children = tuple(sorted(self.children, key=lambda item: item.run_id))
        if any(type(item) is not RunChild for item in children):
            raise TypeError("children must contain exact RunChild values")
        if len({item.run_id for item in children}) != len(children):
            raise RunBundleProtocolError("child run ids must be unique")
        object.__setattr__(self, "children", children)
        for field_name, expected_type in (
            ("aggregation", RunAggregation),
            ("coverage", UsageCoverage),
            ("cost", RunCost),
            ("legacy", LegacyAttribution),
            ("evidence", RunEvidence),
        ):
            if type(getattr(self, field_name)) is not expected_type:
                raise TypeError(f"{field_name} has an unsupported type")
        usage_slices = tuple(
            sorted(
                self.usage_slices,
                key=lambda item: (item.provider, item.model, item.service_tier or ""),
            )
        )
        if any(type(item) is not UsageSlice for item in usage_slices):
            raise TypeError("usage_slices must contain exact UsageSlice values")
        object.__setattr__(self, "usage_slices", usage_slices)
        object.__setattr__(self, "extensions", _freeze_extensions(self.extensions))

        expected_bundle_id = deterministic_bundle_id(identity=self.identity)
        if self.bundle_id:
            observed_id = _canonical_text(self.bundle_id, "bundle_id", maximum=256)
            if observed_id != expected_bundle_id:
                raise RunBundleProtocolError(
                    "bundle_id does not match its deterministic identity"
                )
        object.__setattr__(self, "bundle_id", expected_bundle_id)
        self._validate_projection()
        expected_digest = canonical_sha256(self._body_dict())
        if self.bundle_digest:
            observed_digest = _sha256(self.bundle_digest, "bundle_digest")
            if observed_digest != expected_digest:
                raise RunBundleProtocolError(
                    "bundle_digest does not match the canonical bundle body"
                )
        object.__setattr__(self, "bundle_digest", expected_digest)
        _canonical_bytes(self.to_dict())

    def _validate_projection(self) -> None:
        child_by_id = {child.run_id: child for child in self.children}
        if self.identity.run_id in child_by_id:
            raise RunBundleProtocolError(
                "run topology cannot contain the root as its own child"
            )
        valid_parents = {self.identity.run_id, *child_by_id}
        for child in self.children:
            if child.parent_run_id not in valid_parents:
                raise RunBundleProtocolError(
                    "run topology contains an orphan child"
                )
            if child.bundle_id is None:
                raise RunBundleProtocolError(
                    "materialized run child requires a bundle_id"
                )
            child_identity = RunIdentity(
                execution_id=self.identity.execution_id,
                attempt_id=child.attempt_id,
                root_run_id=self.identity.root_run_id,
                run_id=child.run_id,
                parent_run_id=child.parent_run_id,
                relation=child.relation,
            )
            if child.bundle_id != deterministic_bundle_id(identity=child_identity):
                raise RunBundleProtocolError(
                    "child bundle_id disagrees with its deterministic identity"
                )
        for child in self.children:
            visited = {child.run_id}
            cursor = child
            while cursor.parent_run_id != self.identity.run_id:
                parent = child_by_id.get(cursor.parent_run_id)
                if parent is None:
                    raise RunBundleProtocolError(
                        "run topology child is not rooted at the bundle owner"
                    )
                if parent.run_id in visited:
                    raise RunBundleProtocolError(
                        "run topology contains a child cycle"
                    )
                visited.add(parent.run_id)
                cursor = parent
        owner_attempts = {
            self.identity.run_id: self.identity.attempt_id,
            **{child.run_id: child.attempt_id for child in self.children},
        }
        by_id = {receipt.provider_call_id: receipt for receipt in self.provider_calls}
        for receipt in self.provider_calls:
            call_identity = receipt.identity
            if (
                call_identity.execution_id != self.identity.execution_id
                or call_identity.root_run_id != self.identity.root_run_id
            ):
                raise RunBundleProtocolError(
                    "provider call identity crosses the run bundle boundary"
                )
            expected_attempt_id = owner_attempts.get(call_identity.owner_run_id)
            if expected_attempt_id is None:
                raise RunBundleProtocolError(
                    "provider call owner is absent from run topology"
                )
            if call_identity.attempt_id != expected_attempt_id:
                raise RunBundleProtocolError(
                    "provider call attempt disagrees with its owner topology"
                )
        for event in self.metrics.events:
            if (
                event.execution_id != self.identity.execution_id
                or event.root_run_id != self.identity.root_run_id
            ):
                raise RunBundleProtocolError(
                    "metric event identity crosses the run bundle boundary"
                )
            expected_attempt_id = owner_attempts.get(event.owner_run_id)
            if expected_attempt_id is None:
                raise RunBundleProtocolError(
                    "metric event owner is absent from run topology"
                )
            if event.attempt_id != expected_attempt_id:
                raise RunBundleProtocolError(
                    "metric event attempt disagrees with its owner topology"
                )
            expected_parent = (
                self.identity.parent_run_id
                if event.owner_run_id == self.identity.run_id
                else next(
                    child.parent_run_id
                    for child in self.children
                    if child.run_id == event.owner_run_id
                )
            )
            if event.parent_run_id != expected_parent:
                raise RunBundleProtocolError(
                    "metric event parent disagrees with its owner topology"
                )
        expected_direct_metrics = RunMetricCounters.from_events(
            event
            for event in self.metrics.events
            if event.owner_run_id == self.identity.run_id
        )
        expected_descendant_metrics = RunMetricCounters.from_events(
            event
            for event in self.metrics.events
            if event.owner_run_id != self.identity.run_id
        )
        if (
            self.metrics.direct != expected_direct_metrics
            or self.metrics.descendant != expected_descendant_metrics
        ):
            raise RunBundleProtocolError(
                "direct or descendant metric counters disagree with events"
            )
        model_attempt_events = {
            event.subject_id: event
            for event in self.metrics.events
            if event.kind == "model_attempt"
        }
        if set(model_attempt_events) != set(by_id):
            raise RunBundleProtocolError(
                "model attempt metric events must match provider receipts exactly"
            )
        for call_id, receipt in by_id.items():
            event = model_attempt_events[call_id]
            if event.outcome != receipt.status:
                raise RunBundleProtocolError(
                    "model attempt metric outcome disagrees with provider receipt"
                )
        direct_ids = {
            receipt.provider_call_id
            for receipt in self.provider_calls
            if receipt.identity.owner_run_id == self.identity.run_id
        }
        descendant_ids = set(by_id) - direct_ids
        if (
            set(self.aggregation.direct_call_ids) != direct_ids
            or set(self.aggregation.descendant_call_ids) != descendant_ids
            or set(self.aggregation.all_call_ids) != set(by_id)
        ):
            raise RunBundleProtocolError(
                "aggregation call sets disagree with provider receipts"
            )
        expected_direct_usage = ProviderCallUsage.sum(
            by_id[call_id].usage for call_id in sorted(direct_ids)
        )
        expected_descendant_usage = ProviderCallUsage.sum(
            by_id[call_id].usage for call_id in sorted(descendant_ids)
        )
        expected_all_usage = ProviderCallUsage.sum(
            receipt.usage for receipt in self.provider_calls
        )
        if (
            self.aggregation.direct_usage != expected_direct_usage
            or self.aggregation.descendant_usage != expected_descendant_usage
            or self.aggregation.all_usage != expected_all_usage
        ):
            raise RunBundleProtocolError(
                "aggregation usage disagrees with provider receipts"
            )
        expected_coverage = UsageCoverage.from_receipts(self.provider_calls)
        expected_cost = RunCost.from_receipts(self.provider_calls)
        if self.coverage != expected_coverage or self.cost != expected_cost:
            raise RunBundleProtocolError(
                "bundle coverage or cost disagrees with provider receipts"
            )
        sliced_ids = [call_id for item in self.usage_slices for call_id in item.call_ids]
        if len(sliced_ids) != len(set(sliced_ids)) or set(sliced_ids) != set(by_id):
            raise RunBundleProtocolError(
                "usage slices must partition provider calls exactly once"
            )
        for item in self.usage_slices:
            receipts = tuple(by_id[call_id] for call_id in item.call_ids)
            if any(
                (receipt.provider, receipt.model, receipt.service_tier)
                != (item.provider, item.model, item.service_tier)
                for receipt in receipts
            ):
                raise RunBundleProtocolError("usage slice provider identity disagrees")
            if (
                item.usage != ProviderCallUsage.sum(receipt.usage for receipt in receipts)
                or item.coverage != UsageCoverage.from_receipts(receipts)
                or item.cost != RunCost.from_receipts(receipts)
            ):
                raise RunBundleProtocolError(
                    "usage slice metrics disagree with provider receipts"
                )
        expected_evidence = RunEvidence(
            receipt_sha256s=tuple(
                {
                    receipt.receipt_sha256
                    for receipt in self.provider_calls
                }
            ),
            raw_usage_sha256s=tuple(
                {
                    receipt.raw_usage_sha256
                    for receipt in self.provider_calls
                    if receipt.raw_usage_sha256 is not None
                }
            ),
            pricing_snapshot_ids=tuple(
                {
                    receipt.pricing.snapshot.snapshot_id
                    for receipt in self.provider_calls
                    if receipt.pricing.snapshot is not None
                }
            ),
        )
        if self.evidence != expected_evidence:
            raise RunBundleProtocolError(
                "bundle evidence disagrees with provider receipts"
            )
        contains_legacy = any(
            receipt.usage.source == "legacy_partial"
            for receipt in self.provider_calls
        )
        if contains_legacy != (self.legacy.status == "legacy_partial"):
            raise RunBundleProtocolError(
                "legacy attribution disagrees with provider receipts"
            )

    def _body_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "bundle_id": self.bundle_id,
            "revision": self.revision,
            "identity": self.identity.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "metrics": self.metrics.to_dict(),
            "provider_calls": [receipt.to_dict() for receipt in self.provider_calls],
            "children": [child.to_dict() for child in self.children],
            "aggregation": self.aggregation.to_dict(),
            "usage_slices": [item.to_dict() for item in self.usage_slices],
            "coverage": self.coverage.to_dict(),
            "cost": self.cost.to_dict(),
            "legacy": self.legacy.to_dict(),
            "evidence": self.evidence.to_dict(),
            "extensions": _extensions_dict(self.extensions),
        }

    def to_dict(self) -> dict[str, Any]:
        body = self._body_dict()
        return {
            "schema": body["schema"],
            "bundle_id": body["bundle_id"],
            "revision": body["revision"],
            "bundle_digest": self.bundle_digest,
            "identity": body["identity"],
            "lifecycle": body["lifecycle"],
            "descriptor": body["descriptor"],
            "metrics": body["metrics"],
            "provider_calls": body["provider_calls"],
            "children": body["children"],
            "aggregation": body["aggregation"],
            "usage_slices": body["usage_slices"],
            "coverage": body["coverage"],
            "cost": body["cost"],
            "legacy": body["legacy"],
            "evidence": body["evidence"],
            "extensions": body["extensions"],
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunBundle:
        raw = _strict_record(
            value,
            keys=frozenset(
                {
                    "schema",
                    "bundle_id",
                    "revision",
                    "bundle_digest",
                    "identity",
                    "lifecycle",
                    "descriptor",
                    "metrics",
                    "provider_calls",
                    "children",
                    "aggregation",
                    "usage_slices",
                    "coverage",
                    "cost",
                    "legacy",
                    "evidence",
                    "extensions",
                }
            ),
            field_name="run bundle",
        )
        if raw["schema"] != cls.SCHEMA:
            raise RunBundleProtocolError("run bundle schema is unsupported")
        for field_name in ("provider_calls", "children", "usage_slices"):
            if type(raw[field_name]) is not list:
                raise TypeError(f"{field_name} must be an exact array")
        return cls(
            bundle_id=raw["bundle_id"],
            revision=raw["revision"],
            bundle_digest=raw["bundle_digest"],
            identity=RunIdentity.from_dict(raw["identity"]),
            lifecycle=RunLifecycle.from_dict(raw["lifecycle"]),
            descriptor=RunDescriptor.from_dict(raw["descriptor"]),
            metrics=RunMetrics.from_dict(raw["metrics"]),
            provider_calls=tuple(
                ProviderCallReceipt.from_dict(item) for item in raw["provider_calls"]
            ),
            children=tuple(RunChild.from_dict(item) for item in raw["children"]),
            aggregation=RunAggregation.from_dict(raw["aggregation"]),
            usage_slices=tuple(
                UsageSlice.from_dict(item) for item in raw["usage_slices"]
            ),
            coverage=UsageCoverage.from_dict(raw["coverage"]),
            cost=RunCost.from_dict(raw["cost"]),
            legacy=LegacyAttribution.from_dict(raw["legacy"]),
            evidence=RunEvidence.from_dict(raw["evidence"]),
            extensions=raw["extensions"],
        )


def _metric_event_from_receipt(receipt: ProviderCallReceipt) -> RunMetricEvent:
    identity = receipt.identity
    return RunMetricEvent(
        execution_id=identity.execution_id,
        attempt_id=identity.attempt_id,
        root_run_id=identity.root_run_id,
        owner_run_id=identity.owner_run_id,
        parent_run_id=identity.parent_run_id,
        kind="model_attempt",
        subject_id=receipt.provider_call_id,
        outcome=receipt.status,
    )


class RunBundleReducer:
    """Pure receipt-set reducer used by root, graph, and subagent runtimes."""

    @classmethod
    def reduce(
        cls,
        *,
        identity: RunIdentity,
        lifecycle: RunLifecycle,
        receipts: Iterable[ProviderCallReceipt],
        children: Iterable[RunChild] = (),
        descriptor: RunDescriptor | None = None,
        metric_events: Iterable[RunMetricEvent] = (),
        revision: int = 1,
        legacy: LegacyAttribution | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> RunBundle:
        if type(identity) is not RunIdentity:
            raise TypeError("identity must be an exact RunIdentity")
        if type(lifecycle) is not RunLifecycle:
            raise TypeError("lifecycle must be an exact RunLifecycle")
        receipt_by_id: dict[str, ProviderCallReceipt] = {}
        for receipt in receipts:
            if type(receipt) is not ProviderCallReceipt:
                raise TypeError("receipts must contain exact ProviderCallReceipt values")
            prior = receipt_by_id.get(receipt.provider_call_id)
            if prior is not None and prior.receipt_sha256 != receipt.receipt_sha256:
                raise RunBundleProtocolError(
                    "one provider_call_id has conflicting immutable receipts"
                )
            receipt_by_id[receipt.provider_call_id] = receipt
        ordered_receipts = tuple(
            receipt_by_id[call_id] for call_id in sorted(receipt_by_id)
        )
        child_by_id: dict[str, RunChild] = {}
        for child in children:
            if type(child) is not RunChild:
                raise TypeError("children must contain exact RunChild values")
            prior_child = child_by_id.get(child.run_id)
            if prior_child is not None and prior_child != child:
                raise RunBundleProtocolError(
                    "one child run id has conflicting topology"
                )
            child_by_id[child.run_id] = child
        ordered_children = tuple(
            child_by_id[run_id] for run_id in sorted(child_by_id)
        )
        metric_event_by_id: dict[str, RunMetricEvent] = {}
        for event in metric_events:
            if type(event) is not RunMetricEvent:
                raise TypeError(
                    "metric_events must contain exact RunMetricEvent values"
                )
            prior_event = metric_event_by_id.get(event.metric_event_id)
            if prior_event is not None and prior_event != event:
                raise RunBundleProtocolError(
                    "one metric_event_id has conflicting immutable events"
                )
            metric_event_by_id[event.metric_event_id] = event
        for receipt in ordered_receipts:
            event = _metric_event_from_receipt(receipt)
            prior_event = metric_event_by_id.get(event.metric_event_id)
            if prior_event is not None and prior_event != event:
                raise RunBundleProtocolError(
                    "provider receipt conflicts with its model attempt metric event"
                )
            metric_event_by_id[event.metric_event_id] = event
        ordered_metric_events = tuple(
            metric_event_by_id[event_id]
            for event_id in sorted(metric_event_by_id)
        )
        if descriptor is None:
            descriptor = RunDescriptor()
        if type(descriptor) is not RunDescriptor:
            raise TypeError("descriptor must be an exact RunDescriptor or null")
        direct_receipts = tuple(
            receipt
            for receipt in ordered_receipts
            if receipt.identity.owner_run_id == identity.run_id
        )
        descendant_receipts = tuple(
            receipt
            for receipt in ordered_receipts
            if receipt.identity.owner_run_id != identity.run_id
        )
        aggregation = RunAggregation(
            algorithm=PROVIDER_CALL_SET_UNION_ALGORITHM,
            direct_call_ids=tuple(
                receipt.provider_call_id for receipt in direct_receipts
            ),
            descendant_call_ids=tuple(
                receipt.provider_call_id for receipt in descendant_receipts
            ),
            all_call_ids=tuple(
                receipt.provider_call_id for receipt in ordered_receipts
            ),
            direct_usage=ProviderCallUsage.sum(
                receipt.usage for receipt in direct_receipts
            ),
            descendant_usage=ProviderCallUsage.sum(
                receipt.usage for receipt in descendant_receipts
            ),
            all_usage=ProviderCallUsage.sum(
                receipt.usage for receipt in ordered_receipts
            ),
        )
        grouped: dict[tuple[str, str, str | None], list[ProviderCallReceipt]] = {}
        for receipt in ordered_receipts:
            grouped.setdefault(
                (receipt.provider, receipt.model, receipt.service_tier), []
            ).append(receipt)
        usage_slices = tuple(
            UsageSlice(
                provider=key[0],
                model=key[1],
                service_tier=key[2],
                call_ids=tuple(receipt.provider_call_id for receipt in group),
                usage=ProviderCallUsage.sum(receipt.usage for receipt in group),
                coverage=UsageCoverage.from_receipts(group),
                cost=RunCost.from_receipts(group),
            )
            for key, group in sorted(
                grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
            )
        )
        contains_legacy = any(
            receipt.usage.source == "legacy_partial"
            for receipt in ordered_receipts
        )
        if legacy is None:
            legacy = (
                LegacyAttribution(
                    status="legacy_partial",
                    source="unchain.provider_turn_result.v1",
                    reason="canonical provider usage was unavailable on recovery",
                )
                if contains_legacy
                else LegacyAttribution()
            )
        if type(legacy) is not LegacyAttribution:
            raise TypeError("legacy must be an exact LegacyAttribution or null")
        evidence = RunEvidence(
            receipt_sha256s=tuple(
                {
                    receipt.receipt_sha256
                    for receipt in ordered_receipts
                }
            ),
            raw_usage_sha256s=tuple(
                {
                    receipt.raw_usage_sha256
                    for receipt in ordered_receipts
                    if receipt.raw_usage_sha256 is not None
                }
            ),
            pricing_snapshot_ids=tuple(
                {
                    receipt.pricing.snapshot.snapshot_id
                    for receipt in ordered_receipts
                    if receipt.pricing.snapshot is not None
                }
            ),
        )
        return RunBundle(
            identity=identity,
            lifecycle=lifecycle,
            descriptor=descriptor,
            metrics=RunMetrics.from_events(
                ordered_metric_events,
                direct_run_id=identity.run_id,
            ),
            provider_calls=ordered_receipts,
            children=ordered_children,
            aggregation=aggregation,
            usage_slices=usage_slices,
            coverage=UsageCoverage.from_receipts(ordered_receipts),
            cost=RunCost.from_receipts(ordered_receipts),
            legacy=legacy,
            evidence=evidence,
            revision=revision,
            extensions=extensions or {},
        )

    @classmethod
    def reduce_bundles(
        cls,
        *,
        identity: RunIdentity,
        lifecycle: RunLifecycle,
        bundles: Iterable[RunBundle],
        receipts: Iterable[ProviderCallReceipt] = (),
        children: Iterable[RunChild] = (),
        descriptor: RunDescriptor | None = None,
        metric_events: Iterable[RunMetricEvent] = (),
        revision: int = 1,
        legacy: LegacyAttribution | None = None,
        extensions: Mapping[str, Any] | None = None,
    ) -> RunBundle:
        """Merge public child bundles without reimplementing call-set union."""

        all_receipts = list(receipts)
        all_children = list(children)
        all_metric_events = list(metric_events)
        resolved_descriptor = descriptor
        for bundle in bundles:
            if type(bundle) is not RunBundle:
                raise TypeError("bundles must contain exact RunBundle values")
            if (
                bundle.identity.execution_id != identity.execution_id
                or bundle.identity.root_run_id != identity.root_run_id
            ):
                raise RunBundleProtocolError(
                    "child bundle identity crosses the target run boundary"
                )
            all_receipts.extend(bundle.provider_calls)
            all_children.extend(bundle.children)
            all_metric_events.extend(bundle.metrics.events)
            if bundle.identity == identity and resolved_descriptor is None:
                resolved_descriptor = bundle.descriptor
            if bundle.identity.run_id != identity.run_id:
                assert bundle.identity.parent_run_id is not None
                all_children.append(
                    RunChild(
                        run_id=bundle.identity.run_id,
                        attempt_id=bundle.identity.attempt_id,
                        parent_run_id=bundle.identity.parent_run_id,
                        relation=bundle.identity.relation,
                        bundle_id=bundle.bundle_id,
                        status=bundle.lifecycle.status,
                    )
                )
        return cls.reduce(
            identity=identity,
            lifecycle=lifecycle,
            receipts=all_receipts,
            children=all_children,
            descriptor=resolved_descriptor,
            metric_events=all_metric_events,
            revision=revision,
            legacy=legacy,
            extensions=extensions,
        )


def reproject_run_bundle_extensions(
    bundle: RunBundle,
    *,
    extensions: Mapping[str, Any],
    next_revision: int,
) -> RunBundle:
    """Create the one allowed next revision with additive safe extensions.

    The returned value still passes the complete canonical projection
    validation.  Persist it with the execution-bound ``RunBundleLedger``;
    callers must never mutate a serialized bundle dictionary post hoc.
    """

    if type(bundle) is not RunBundle:
        raise TypeError("bundle must be an exact RunBundle")
    if type(extensions) is not dict:
        raise TypeError("extensions must be an exact object")
    if type(next_revision) is not int or next_revision != bundle.revision + 1:
        raise RunBundleProtocolError(
            "extension reprojection requires the exact next revision"
        )
    additions = dict(_freeze_extensions(extensions))
    overlap = set(bundle.extensions).intersection(additions)
    if overlap:
        raise RunBundleProtocolError(
            "extension reprojection cannot rewrite an existing namespace"
        )
    return RunBundle(
        identity=bundle.identity,
        lifecycle=bundle.lifecycle,
        descriptor=bundle.descriptor,
        metrics=bundle.metrics,
        provider_calls=bundle.provider_calls,
        children=bundle.children,
        aggregation=bundle.aggregation,
        usage_slices=bundle.usage_slices,
        coverage=bundle.coverage,
        cost=bundle.cost,
        legacy=bundle.legacy,
        evidence=bundle.evidence,
        revision=next_revision,
        extensions={**dict(bundle.extensions), **additions},
    )


__all__ = [
    "LegacyAttribution",
    "METRIC_EVENT_SET_UNION_ALGORITHM",
    "PROVIDER_CALL_SET_UNION_ALGORITHM",
    "PROVIDER_CALL_USAGE_SCHEMA",
    "PricingSnapshotRef",
    "ProviderBillingDimensions",
    "ProviderCallIds",
    "ProviderCallIdentity",
    "ProviderCallPricing",
    "ProviderCallReceipt",
    "ProviderCallTiming",
    "ProviderCallUsage",
    "RUN_BUNDLE_SCHEMA",
    "RunAggregation",
    "RunBundle",
    "RunBundleProtocolError",
    "RunBundleReducer",
    "RunChild",
    "RunCost",
    "RunDescriptor",
    "RunEvidence",
    "RunIdentity",
    "RunLifecycle",
    "RunMetricCounters",
    "RunMetricError",
    "RunMetricEvent",
    "RunMetricEvidenceRef",
    "RunMetrics",
    "UsageCoverage",
    "UsageSlice",
    "canonical_sha256",
    "deterministic_bundle_id",
    "deterministic_metric_event_id",
    "deterministic_provider_call_id",
    "opaque_metric_evidence_ref",
    "reproject_run_bundle_extensions",
]
