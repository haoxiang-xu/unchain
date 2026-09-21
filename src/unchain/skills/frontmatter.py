"""Safe-subset YAML frontmatter parser for ``SKILL.md`` files.

This module implements decision D1 of ticket #327: a bounded, dependency-free
parser for the small slice of YAML used by SKILL.md frontmatter blocks. It is
deliberately not a general YAML parser -- unsupported constructs (flow
mappings, tags, anchors/aliases, tab indentation) are rejected rather than
silently misinterpreted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SkillParseError",
    "ParsedSkillFile",
    "parse_skill_file",
    "coerce_bool",
]


class SkillParseError(ValueError):
    """Raised when a SKILL.md frontmatter block cannot be parsed."""


@dataclass(frozen=True)
class ParsedSkillFile:
    """Result of parsing a SKILL.md-style text file."""

    fields: dict[str, object]
    body: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KEY_LINE_RE = re.compile(r"^([A-Za-z0-9_-]+):(.*)$")

# style -> (kind, chomp): kind is "literal" (|) or "folded" (>); chomp is
# "clip" (default, single trailing newline), "strip" (-, no trailing
# newline), or "keep" (+, preserve all trailing blank lines).
_BLOCK_SCALAR_STYLES: dict[str, tuple[str, str]] = {
    "|": ("literal", "clip"),
    "|-": ("literal", "strip"),
    "|+": ("literal", "keep"),
    ">": ("folded", "clip"),
    ">-": ("folded", "strip"),
    ">+": ("folded", "keep"),
}

# Duplicates among these top-level keys are rejected outright; every other
# duplicate key silently lets the last occurrence win (D1).
_DUPLICATE_SENSITIVE_KEYS = frozenset(
    {"name", "description", "disable-model-invocation", "user-invocable"}
)

_TRUE_STRINGS = {"true", "yes", "on", "1"}
_FALSE_STRINGS = {"false", "no", "off", "0"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_skill_file(text: str) -> ParsedSkillFile:
    """Parse a SKILL.md-style text file into frontmatter fields and a body.

    See the module docstring and ticket #327 decision D1 for the exact
    contract implemented here.
    """

    raw = text
    if raw.startswith("﻿"):
        raw = raw[1:]
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    if not lines or lines[0] != "---":
        raise SkillParseError("missing YAML frontmatter delimited by ---")

    closing_index: int | None = None
    for index in range(1, len(lines)):
        if lines[index] == "---":
            closing_index = index
            break
    if closing_index is None:
        raise SkillParseError("unterminated YAML frontmatter")

    fm_lines = lines[1:closing_index]
    body_lines = lines[closing_index + 1 :]

    _reject_tab_indentation(fm_lines)
    fields, _ = _parse_mapping(fm_lines, 0, len(fm_lines), 0, top_level=True)
    body = _strip_blank_edges(body_lines)
    return ParsedSkillFile(fields=fields, body=body)


def coerce_bool(value: object, *, default: bool) -> bool:
    """Coerce a raw frontmatter field value into a bool.

    ``None`` or a blank/whitespace-only string returns ``default``. The
    strings ``true/yes/on/1`` (case-insensitive) coerce to ``True`` and
    ``false/no/off/0`` (case-insensitive) coerce to ``False``. Anything else
    -- including non-string values such as a parsed list -- raises
    ``SkillParseError``.
    """

    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return default
        lowered = stripped.lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False
        raise SkillParseError(f"expected a boolean, got {value!r}")
    raise SkillParseError(f"expected a boolean, got {value!r}")


# ---------------------------------------------------------------------------
# Indentation / structure helpers
# ---------------------------------------------------------------------------


def _reject_tab_indentation(lines: list[str]) -> None:
    for line in lines:
        prefix_len = len(line) - len(line.lstrip())
        prefix = line[:prefix_len]
        if "\t" in prefix:
            raise SkillParseError(
                f"tab characters are not allowed for indentation: {line!r}"
            )


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _strip_blank_edges(lines: list[str]) -> str:
    start = 0
    end = len(lines)
    while start < end and lines[start].strip() == "":
        start += 1
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return "\n".join(lines[start:end])


def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_comment(line: str) -> bool:
    return line.lstrip().startswith("#")


def _peek_meaningful(
    lines: list[str], start: int, end: int
) -> tuple[int, int] | None:
    """Return (index, indent) of the next non-blank, non-comment line."""

    index = start
    while index < end:
        line = lines[index]
        if _is_blank(line) or _is_comment(line):
            index += 1
            continue
        return index, _indent_of(line)
    return None


# ---------------------------------------------------------------------------
# Mapping / list parsing
# ---------------------------------------------------------------------------


def _parse_mapping(
    lines: list[str],
    start: int,
    end: int,
    indent: int,
    *,
    top_level: bool,
) -> tuple[dict[str, object], int]:
    result: dict[str, object] = {}
    index = start
    while index < end:
        line = lines[index]
        if _is_blank(line):
            index += 1
            continue
        if _is_comment(line):
            index += 1
            continue
        cur_indent = _indent_of(line)
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise SkillParseError(f"invalid frontmatter line: {line.strip()!r}")

        content = line[indent:]
        match = _KEY_LINE_RE.match(content)
        if not match:
            raise SkillParseError(f"invalid frontmatter line: {line.strip()!r}")
        key = match.group(1)
        rest = match.group(2) or ""
        value_token = rest.strip()
        index += 1

        if value_token in _BLOCK_SCALAR_STYLES:
            style, chomp = _BLOCK_SCALAR_STYLES[value_token]
            content_lines, index = _collect_block_scalar(lines, index, end, indent)
            value: object = _render_block_scalar(content_lines, style, chomp)
        elif value_token == "":
            peek = _peek_meaningful(lines, index, end)
            if peek is not None and peek[1] > indent:
                nxt_index, nxt_indent = peek
                nxt_content = lines[nxt_index][nxt_indent:]
                if nxt_content == "-" or nxt_content.startswith("- "):
                    value, index = _parse_list(lines, nxt_index, end, nxt_indent)
                else:
                    value, index = _parse_mapping(
                        lines, nxt_index, end, nxt_indent, top_level=False
                    )
            else:
                value = ""
        elif value_token[0] == "[":
            value = _parse_flow_sequence(value_token)
        elif value_token[0] == "{":
            raise SkillParseError(f"flow mappings are not supported: {value_token!r}")
        else:
            value = _parse_scalar(value_token)

        if top_level and key in _DUPLICATE_SENSITIVE_KEYS and key in result:
            raise SkillParseError(f"duplicate frontmatter key: {key!r}")
        result[key] = value
    return result, index


def _parse_list(
    lines: list[str], start: int, end: int, indent: int
) -> tuple[list[object], int]:
    items: list[object] = []
    index = start
    while index < end:
        line = lines[index]
        if _is_blank(line):
            index += 1
            continue
        if _is_comment(line):
            index += 1
            continue
        cur_indent = _indent_of(line)
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise SkillParseError(f"invalid frontmatter line: {line.strip()!r}")

        content = line[indent:]
        if content == "-":
            item_raw = ""
        elif content.startswith("- "):
            item_raw = content[2:]
        else:
            raise SkillParseError(f"invalid frontmatter line: {line.strip()!r}")
        item_raw_stripped = item_raw.strip()
        index += 1

        if item_raw_stripped == "":
            peek = _peek_meaningful(lines, index, end)
            if peek is not None and peek[1] > indent:
                nxt_index, nxt_indent = peek
                nxt_content = lines[nxt_index][nxt_indent:]
                if nxt_content == "-" or nxt_content.startswith("- "):
                    value, index = _parse_list(lines, nxt_index, end, nxt_indent)
                else:
                    value, index = _parse_mapping(
                        lines, nxt_index, end, nxt_indent, top_level=False
                    )
            else:
                value = ""
        elif item_raw_stripped[0] == "[":
            value = _parse_flow_sequence(item_raw_stripped)
        elif item_raw_stripped[0] == "{":
            raise SkillParseError(
                f"flow mappings are not supported: {item_raw_stripped!r}"
            )
        else:
            value = _parse_scalar(item_raw_stripped)
        items.append(value)
    return items, index


# ---------------------------------------------------------------------------
# Block scalars
# ---------------------------------------------------------------------------


def _collect_block_scalar(
    lines: list[str], start: int, end: int, key_indent: int
) -> tuple[list[str], int]:
    raw_lines: list[str] = []
    index = start
    while index < end:
        line = lines[index]
        if _is_blank(line):
            raw_lines.append("")
            index += 1
            continue
        cur_indent = _indent_of(line)
        if cur_indent <= key_indent:
            break
        raw_lines.append(line)
        index += 1

    block_indent: int | None = None
    for entry in raw_lines:
        if entry != "":
            block_indent = _indent_of(entry)
            break
    if block_indent is None:
        return [], index

    content_lines = [
        "" if entry == "" else entry[block_indent:] for entry in raw_lines
    ]
    return content_lines, index


def _fold(lines: list[str]) -> str:
    parts: list[str] = []
    prev_was_content = False
    for line in lines:
        if line == "":
            parts.append("\n")
            prev_was_content = False
        else:
            if prev_was_content:
                parts.append(" ")
            parts.append(line)
            prev_was_content = True
    return "".join(parts)


def _render_block_scalar(content_lines: list[str], style: str, chomp: str) -> str:
    if not content_lines:
        return ""

    trimmed = list(content_lines)
    while trimmed and trimmed[-1] == "":
        trimmed.pop()

    if style == "literal":
        base_trimmed = "\n".join(trimmed)
        base_full = "\n".join(content_lines)
    else:
        base_trimmed = _fold(trimmed)
        base_full = _fold(content_lines)

    if chomp == "strip":
        return base_trimmed
    if chomp == "keep":
        return base_full + "\n"
    return base_trimmed + "\n"


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


def _strip_plain_comment(token: str) -> str:
    for i in range(1, len(token)):
        if token[i] == "#" and token[i - 1] in " \t":
            return token[: i - 1].rstrip()
    return token


def _parse_scalar(token: str) -> str:
    if token.startswith("!"):
        raise SkillParseError(f"unsupported YAML tag: {token!r}")
    if token.startswith("&"):
        raise SkillParseError(f"unsupported YAML anchor: {token!r}")
    if token.startswith("*"):
        raise SkillParseError(f"unsupported YAML alias: {token!r}")
    if token.startswith("'"):
        return _parse_single_quoted(token)
    if token.startswith('"'):
        return _parse_double_quoted(token)
    return _strip_plain_comment(token).strip()


def _parse_single_quoted(token: str) -> str:
    result: list[str] = []
    index = 1
    length = len(token)
    closed = False
    while index < length:
        char = token[index]
        if char == "'":
            if index + 1 < length and token[index + 1] == "'":
                result.append("'")
                index += 2
                continue
            closed = True
            index += 1
            break
        result.append(char)
        index += 1
    if not closed:
        raise SkillParseError(f"unterminated single-quoted scalar: {token!r}")
    return "".join(result)


def _parse_double_quoted(token: str) -> str:
    result: list[str] = []
    index = 1
    length = len(token)
    closed = False
    while index < length:
        char = token[index]
        if char == "\\":
            if index + 1 >= length:
                raise SkillParseError(
                    f"invalid escape sequence in double-quoted scalar: {token!r}"
                )
            next_char = token[index + 1]
            if next_char == '"':
                result.append('"')
            elif next_char == "\\":
                result.append("\\")
            elif next_char == "n":
                result.append("\n")
            elif next_char == "t":
                result.append("\t")
            else:
                raise SkillParseError(f"unsupported escape sequence: \\{next_char}")
            index += 2
            continue
        if char == '"':
            closed = True
            index += 1
            break
        result.append(char)
        index += 1
    if not closed:
        raise SkillParseError(f"unterminated double-quoted scalar: {token!r}")
    return "".join(result)


def _split_flow_items(inner: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    in_squote = False
    in_dquote = False
    index = 0
    length = len(inner)
    while index < length:
        char = inner[index]
        if in_dquote:
            buf.append(char)
            if char == "\\" and index + 1 < length:
                buf.append(inner[index + 1])
                index += 2
                continue
            if char == '"':
                in_dquote = False
            index += 1
            continue
        if in_squote:
            buf.append(char)
            if char == "'":
                if index + 1 < length and inner[index + 1] == "'":
                    buf.append("'")
                    index += 2
                    continue
                in_squote = False
            index += 1
            continue
        if char == '"':
            in_dquote = True
            buf.append(char)
            index += 1
            continue
        if char == "'":
            in_squote = True
            buf.append(char)
            index += 1
            continue
        if char == ",":
            items.append("".join(buf))
            buf = []
            index += 1
            continue
        buf.append(char)
        index += 1
    if buf or items:
        items.append("".join(buf))
    return items


def _parse_flow_sequence(token: str) -> list[str]:
    if not token.endswith("]"):
        raise SkillParseError(f"unterminated flow sequence: {token!r}")
    inner = token[1:-1]
    items = _split_flow_items(inner)
    result: list[str] = []
    for raw_item in items:
        item = raw_item.strip()
        if item == "":
            continue
        if item[0] == "[":
            raise SkillParseError(f"nested flow sequences are not supported: {item!r}")
        if item[0] == "{":
            raise SkillParseError(f"flow mappings are not supported: {item!r}")
        result.append(_parse_scalar(item))
    return result
