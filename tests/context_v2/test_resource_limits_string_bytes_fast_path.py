from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

import unchain.journal.resource_limits as resource_limits
from unchain.journal.resource_limits import (
    BoundaryResourceLimitError,
    JsonResourceLimits,
    validate_json_resource,
)


# Frozen oracle: validate_json_resource from b07c9d67c61c4a6437e9fd08325285530b7b8b6a,
# src/unchain/journal/resource_limits.py (SHA-256
# 878918355289668083ecc30679e53524f2ffebc53be23b4453f088f14652829d).
# Keep this reference's original character loop independent of the candidate.
def _frozen_validate_json_resource(
    value: Any,
    *,
    boundary: str,
    limits: JsonResourceLimits,
    record_adapter: Callable[[Any], Any] | None = None,
) -> None:
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


def _outcome(
    function: Callable[..., Any], value: Any, **kwargs: Any
) -> tuple[Any, ...]:
    try:
        result = function(value, **kwargs)
    except Exception as error:
        cause = error.__cause__
        details: tuple[Any, ...] = (
            "raise",
            type(error),
            error.args,
            getattr(error, "boundary", None),
            getattr(error, "dimension", None),
            getattr(error, "limit", None),
            getattr(error, "observed", None),
            type(cause) if cause is not None else None,
            cause.args if cause is not None else None,
        )
        return details
    return ("return", result)


def _limits(
    max_bytes: int, *, max_depth: int = 64, max_nodes: int = 100_000
) -> JsonResourceLimits:
    return JsonResourceLimits(
        max_items=100_000,
        max_bytes=max_bytes,
        max_depth=max_depth,
        max_nodes=max_nodes,
    )


def _assert_same(
    value: Any,
    *,
    max_bytes: int | None = None,
    limits: JsonResourceLimits | None = None,
    **kwargs: Any,
) -> tuple[Any, ...]:
    selected_limits = limits if limits is not None else _limits(max_bytes or 1)
    call = {"boundary": "string-bytes-fast-path", "limits": selected_limits, **kwargs}
    expected = _outcome(_frozen_validate_json_resource, value, **call)
    actual = _outcome(validate_json_resource, value, **call)
    assert actual == expected
    return actual


@pytest.mark.parametrize(
    "unit",
    [
        "a",
        '"\\',
        "\x00\x01\x1f\b\t\n\f\r",
        "".join(chr(codepoint) for codepoint in range(0x20)),
        "é中",
        "😀",
        "a\x00\\é😀",
    ],
)
@pytest.mark.parametrize("length", [4095, 4096, 4097, 8191, 8192, 8193])
def test_string_boundaries_match_frozen_original(unit: str, length: int) -> None:
    text = (unit * (length // len(unit) + 1))[:length]
    quoted_bytes = len(json.encoder.encode_basestring(text).encode("utf-8"))
    for budget in (quoted_bytes - 1, quoted_bytes, quoted_bytes + 1):
        _assert_same(text, max_bytes=budget)


def test_empty_strings_and_mapping_prefix_accounting_match_frozen_original() -> None:
    empty_root = _assert_same("", max_bytes=1)
    assert empty_root[1:] == (
        BoundaryResourceLimitError,
        ("string-bytes-fast-path exceeds bytes limit: observed 2, limit 1",),
        "string-bytes-fast-path",
        "bytes",
        1,
        2,
        None,
        None,
    )

    empty_key = _assert_same({"": None}, max_bytes=4)
    assert empty_key[1] is BoundaryResourceLimitError
    assert empty_key[6] == 5

    text = "é😀"
    quoted_bytes = len(json.encoder.encode_basestring(text).encode("utf-8"))
    _assert_same(["prefix", text], max_bytes=2 + 1 + 8 + quoted_bytes)
    key_bytes = len(json.encoder.encode_basestring("κ").encode("utf-8"))
    _assert_same({"κ": text}, max_bytes=2 + key_bytes + 1 + quoted_bytes)


@pytest.mark.parametrize(
    ("text", "max_bytes", "dimension"),
    [
        ("abc\ud800", 3, "bytes"),
        ("abc\ud800", 100, "unicode"),
        ("\udfff", 100, "unicode"),
        ("a" * 4093 + "\ud800" + "z" * 8, 8, "bytes"),
        ("a" * 4093 + "\ud800" + "z" * 8, 10000, "unicode"),
        ("a" * 4096 + "\udfff", 100, "bytes"),
        ("a" * 4096 + "\udfff", 10000, "unicode"),
        ("a" * 8192 + "\ud800", 100, "bytes"),
        ("a" * 8192 + "\ud800", 10000, "unicode"),
    ],
)
def test_invalid_unicode_and_quota_precedence_match_frozen_original(
    text: str, max_bytes: int, dimension: str
) -> None:
    outcome = _assert_same(text, max_bytes=max_bytes)
    if dimension == "bytes":
        assert outcome[1] is BoundaryResourceLimitError
        assert outcome[4] == "bytes"
        assert outcome[6] == max_bytes + 1
    else:
        assert outcome[1] is ValueError
        assert outcome[7] is UnicodeEncodeError


class _IterationOnlyText(str):
    def __iter__(self):
        return iter(("x", "é", "\\"))

    def __len__(self):
        raise AssertionError("str subclasses must retain the old iterator path")

    def __getitem__(self, _key):
        raise AssertionError("str subclasses must not be sliced")


class _RaisingIterationText(str):
    def __iter__(self):
        yield "x"
        raise RuntimeError("custom text iterator failed")


def test_string_subclasses_keep_original_iteration_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original = json.encoder.encode_basestring

    def observe(text: str) -> str:
        calls.append(text)
        return original(text)

    monkeypatch.setattr(resource_limits.json.encoder, "encode_basestring", observe)
    _assert_same(_IterationOnlyText("ignored"), max_bytes=100)
    assert calls == []

    _assert_same(_RaisingIterationText("ignored"), max_bytes=100)
    assert calls == []


def test_mixed_json_negative_controls_and_adapter_order_match_frozen_original() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    controls: list[Any] = [cycle, {1: "not-text"}, object(), float("nan")]
    for value in controls:
        _assert_same(value, max_bytes=100)

    _assert_same([[['deep']]], limits=_limits(100, max_depth=2))
    _assert_same([["too many nodes"]], limits=_limits(100, max_nodes=2))

    ordered: list[Any] = []

    def adapter(value: Any) -> Any:
        ordered.append(value)
        return value

    common = {"boundary": "adapter-order", "limits": _limits(100)}
    expected_order: list[Any] = []

    def reference_adapter(value: Any) -> Any:
        expected_order.append(value)
        return value

    assert _outcome(
        _frozen_validate_json_resource,
        {"key": [1]},
        **common,
        record_adapter=reference_adapter,
    ) == _outcome(
        validate_json_resource,
        {"key": [1]},
        **common,
        record_adapter=adapter,
    )
    assert ordered == expected_order == [{"key": [1]}, [1], 1]


def test_encoder_fast_path_is_delegated_and_never_encodes_over_4096_chars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lengths: list[int] = []
    original = json.encoder.encode_basestring

    def observe(text: str) -> str:
        lengths.append(len(text))
        return original(text)

    monkeypatch.setattr(resource_limits.json.encoder, "encode_basestring", observe)

    validate_json_resource("short", boundary="encoder", limits=_limits(100))
    assert lengths == [5]

    lengths.clear()
    validate_json_resource("x" * 8193, boundary="encoder", limits=_limits(9000))
    assert lengths == [4096, 4096, 1]
    assert max(lengths) <= 4096

    lengths.clear()
    _assert_same("", max_bytes=1)
    validate_json_resource(
        _IterationOnlyText("ignored"), boundary="encoder", limits=_limits(100)
    )
    assert lengths == []
