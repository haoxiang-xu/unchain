from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonResourceLimits:
    """Fail-closed resource ceiling for one JSON boundary record."""

    max_items: int
    max_bytes: int
    max_depth: int
    max_nodes: int

    def __post_init__(self) -> None:
        for field_name in ("max_items", "max_bytes", "max_depth", "max_nodes"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")


class BoundaryResourceLimitError(ValueError):
    """A typed, inspectable rejection instead of truncation or recursion failure."""

    def __init__(
        self,
        *,
        boundary: str,
        dimension: str,
        limit: int,
        observed: int,
    ) -> None:
        self.boundary = boundary
        self.dimension = dimension
        self.limit = limit
        self.observed = observed
        super().__init__(
            f"{boundary} exceeds {dimension} limit: "
            f"observed {observed}, limit {limit}"
        )


def enforce_item_limit(
    value: Sequence[Any],
    *,
    boundary: str,
    limits: JsonResourceLimits,
) -> None:
    observed = len(value)
    if observed > limits.max_items:
        raise BoundaryResourceLimitError(
            boundary=boundary,
            dimension="items",
            limit=limits.max_items,
            observed=observed,
        )


def validate_json_resource(
    value: Any,
    *,
    boundary: str,
    limits: JsonResourceLimits,
    record_adapter: Callable[[Any], Any] | None = None,
) -> None:
    """Bound JSON shape and compact UTF-8 size without recursive overrun."""

    node_count = 0
    byte_count = 0
    active_containers: set[int] = set()

    def reject(dimension: str, limit: int, observed: int) -> None:
        raise BoundaryResourceLimitError(
            boundary=boundary,
            dimension=dimension,
            limit=limit,
            observed=observed,
        )

    def add_bytes(amount: int) -> None:
        nonlocal byte_count
        byte_count += amount
        if byte_count > limits.max_bytes:
            reject("bytes", limits.max_bytes, byte_count)

    def string_bytes(text: str) -> int:
        if type(text) is str:
            if not text:
                return 2

            if len(text) <= 4096:
                try:
                    size = len(json.encoder.encode_basestring(text).encode("utf-8"))
                except UnicodeEncodeError:
                    pass
                else:
                    if byte_count + size > limits.max_bytes:
                        return limits.max_bytes + 1
                    return size
            else:
                size = 2
                try:
                    for offset in range(0, len(text), 4096):
                        encoded = json.encoder.encode_basestring(
                            text[offset : offset + 4096]
                        ).encode("utf-8")
                        size += len(encoded) - 2
                        if byte_count + size > limits.max_bytes:
                            return limits.max_bytes + 1
                except UnicodeEncodeError:
                    pass
                else:
                    return size

        size = 2
        for character in text:
            codepoint = ord(character)
            if character in {'"', "\\"} or character in "\b\t\n\f\r":
                size += 2
            elif codepoint < 0x20:
                size += 6
            else:
                try:
                    size += len(character.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        f"{boundary} contains invalid Unicode"
                    ) from exc
            if byte_count + size > limits.max_bytes:
                return limits.max_bytes + 1
        return size

    def walk(item: Any, depth: int) -> None:
        nonlocal node_count
        if record_adapter is not None:
            item = record_adapter(item)
        if depth > limits.max_depth:
            reject("depth", limits.max_depth, depth)
        node_count += 1
        if node_count > limits.max_nodes:
            reject("nodes", limits.max_nodes, node_count)

        if item is None:
            add_bytes(4)
            return
        if isinstance(item, bool):
            add_bytes(4 if item else 5)
            return
        if isinstance(item, str):
            add_bytes(string_bytes(item))
            return
        if isinstance(item, int):
            try:
                add_bytes(len(str(item).encode("ascii")))
            except ValueError as exc:
                raise ValueError(f"{boundary} contains an invalid integer") from exc
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{boundary} contains a non-finite number")
            add_bytes(
                len(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                )
            )
            return

        is_mapping = isinstance(item, Mapping)
        is_sequence = isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        )
        if not is_mapping and not is_sequence:
            raise TypeError(f"{boundary} contains a non-JSON value")

        identity = id(item)
        if identity in active_containers:
            raise ValueError(f"{boundary} contains a circular JSON value")
        active_containers.add(identity)
        try:
            if is_mapping:
                add_bytes(2 + max(0, len(item) - 1))
                for key, child in item.items():
                    if not isinstance(key, str):
                        raise TypeError(f"{boundary} contains a non-text object key")
                    add_bytes(string_bytes(key) + 1)
                    walk(child, depth + 1)
                return

            add_bytes(2 + max(0, len(item) - 1))
            for child in item:
                walk(child, depth + 1)
        finally:
            active_containers.remove(identity)

    walk(value, 0)


__all__ = [
    "BoundaryResourceLimitError",
    "JsonResourceLimits",
    "enforce_item_limit",
    "validate_json_resource",
]
