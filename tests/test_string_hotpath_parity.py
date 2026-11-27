"""Frozen Core33 bodies: test-only independent differential oracles."""
from __future__ import annotations
import json, re, unicodedata
from typing import Any
from unchain.journal.models import ModelValidationError
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")

def _integer_decimal(value: int) -> bytes:
    """Encode any Python integer without consulting the process digit limit."""
    if value == 0:
        return b"0"
    negative = value < 0
    remaining = -value if negative else value
    chunks: list[int] = []
    while remaining:
        remaining, chunk = divmod(remaining, 1_000_000_000)
        chunks.append(chunk)
    pieces = [str(chunks.pop()).encode("ascii")]
    while chunks:
        pieces.append(f"{chunks.pop():09d}".encode("ascii"))
    return (b"-" if negative else b"") + b"".join(pieces)

def _string_json(value: str) -> bytes:
    encoded = bytearray(b'"')
    escapes = {
        '"': b'\\"',
        "\\": b"\\\\",
        "\b": b"\\b",
        "\f": b"\\f",
        "\n": b"\\n",
        "\r": b"\\r",
        "\t": b"\\t",
    }
    for character in value:
        escaped = escapes.get(character)
        if escaped is not None:
            encoded.extend(escaped)
            continue
        codepoint = ord(character)
        if codepoint < 0x20 or 0xD800 <= codepoint <= 0xDFFF:
            encoded.extend(f"\\u{codepoint:04x}".encode("ascii"))
        else:
            encoded.extend(character.encode("utf-8"))
    encoded.append(ord('"'))
    return bytes(encoded)

def _canonical_value(value: Any, parts: list[bytes]) -> None:
    if value is None:
        parts.append(b"null")
    elif type(value) is bool:
        parts.append(b"true" if value else b"false")
    elif type(value) is int:
        parts.append(_integer_decimal(value))
    elif type(value) is float:
        parts.append(json.dumps(value, allow_nan=False).encode("ascii"))
    elif type(value) is str:
        parts.append(_string_json(value))
    elif type(value) is list:
        parts.append(b"[")
        for index, child in enumerate(value):
            if index:
                parts.append(b",")
            _canonical_value(child, parts)
        parts.append(b"]")
    elif type(value) is dict:
        parts.append(b"{")
        for index, key in enumerate(sorted(value)):
            if index:
                parts.append(b",")
            parts.append(_string_json(key))
            parts.append(b":")
            _canonical_value(value[key], parts)
        parts.append(b"}")
    else:
        raise TypeError("value is not JSON")

def _required_text(
    value: Any,
    field_name: str,
    *,
    maximum: int = 256,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be text")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized or len(normalized) > maximum or "\x00" in normalized:
        raise ModelValidationError(f"{field_name} is invalid")
    if any(ord(character) < 32 for character in normalized):
        raise ModelValidationError(f"{field_name} contains control characters")
    if identifier and _IDENTIFIER_RE.fullmatch(normalized) is None:
        raise ModelValidationError(f"{field_name} is not a valid identifier")
    return normalized

import hashlib
import random
import pytest
from unchain.skeleton import recovery as candidate_recovery
from unchain.journal import models as candidate_models

def outcome(fn, *args, **kwargs):
    try:
        return ('return', fn(*args, **kwargs))
    except Exception as error:
        return ('error', type(error), str(error))

def corpus():
    fixed = ['', '\"\\/', ''.join(map(chr, range(32))), chr(127), ''.join(map(chr, range(256))),
        'é', 'e\u0301', '漢字', '\U0001f600\uffff\ufdd0', '\ud800', '\udfff', '\ud800\udc00', 'a\x00\ud800\nZ']
    fixed += ['a' * n for n in (1, 32, 255, 4095, 4096, 4097, 65536)]
    rng = random.Random(356)
    alphabet = ['a', 'é', '\u0301', '漢', '\U0001f600', '\ud800', '\udfff', '\x00', '\n', '\"', '\\', chr(127)]
    return fixed + [''.join(rng.choice(alphabet) for _ in range(n)) for n in range(128)]

@pytest.mark.parametrize('value', corpus())
def test_canonical_string_bytes_parity(value):
    assert outcome(candidate_recovery._string_json, value) == outcome(_string_json, value)

@pytest.mark.parametrize('value', corpus() + [None, 1, [], b'text'])
@pytest.mark.parametrize('identifier', [False, True])
@pytest.mark.parametrize('maximum', [0, 1, 255, 256, 4096])
def test_required_text_normalization_and_errors(value, identifier, maximum):
    assert outcome(candidate_models._required_text, value, 'field', maximum=maximum, identifier=identifier) == outcome(_required_text, value, 'field', maximum=maximum, identifier=identifier)

@pytest.mark.parametrize('point', range(160))
def test_required_text_control_range_precedence(point):
    for value in ('a'+chr(point)+'b', chr(point)+'a'+chr(point), 'a'+chr(point)+'\x00'):
        assert outcome(candidate_models._required_text, value, 'field') == outcome(_required_text, value, 'field')

def test_normalization_length_boundaries():
    for value in ('  e\u0301  ', 'e\u0301'*256, 'é'*257, 'a\x00\n', '  ', '\ud800'):
        for maximum in (1, 255, 256):
            assert outcome(candidate_models._required_text, value, 'field', maximum=maximum) == outcome(_required_text, value, 'field', maximum=maximum)

class Iterated(str):
    def __iter__(self):
        return iter('custom\n\ud800')

class StripSubclass(str):
    def strip(self, *args):
        return Iterated('normalized')

class StripError(str):
    def strip(self, *args):
        raise ValueError('custom strip failure')

def test_str_subclass_and_non_str_fallback():
    for value in (Iterated('ignored'), ['x', '\ud800', '\n'], [1], 1):
        assert outcome(candidate_recovery._string_json, value) == outcome(_string_json, value)
    for value in (Iterated('text'), StripSubclass('text'), StripError('text')):
        assert outcome(candidate_models._required_text, value, 'field') == outcome(_required_text, value, 'field')

def test_nested_canonical_bytes_and_digest():
    for value in ({'z': [None, True, False, 0, -10, 10**100, 1.25, -0.0], '\ud800': {'é': '\udfff', 'a': 'x\n'}},
                  {v: [v, {'nested': v}] for v in corpus()[:14]}):
        parts=[]
        _canonical_value(value, parts)
        expected=b''.join(parts)
        assert candidate_recovery._canonical(value, ()) == expected
        assert candidate_recovery._digest(value, ()) == hashlib.sha256(expected).hexdigest()
