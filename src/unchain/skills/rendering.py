"""Rendering and strict parsing for the skills catalog and active-skills blocks.

Two durable, delimited system-message blocks are produced here:

- ``<available_skills>`` — a lightweight catalog of model-invocable skills
  (name + description), rendered fresh every step from the current
  inventory. See ``render_skill_catalog``.
- ``<active_skills>`` — the durable activation snapshot: full bodies of the
  skills activated so far in this run. It is the only copy of "what is
  active" that survives compaction, resume, retry, and cold restart, so its
  wire format has a strict, versioned parser (``parse_active_skills_block``)
  that is the exact inverse of ``render_active_skills_block``.

This module has no knowledge of the kernel, harnesses, or the registry; it
only turns ``models.py`` records into text and back.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from .models import ActiveSkill, SkillIdentity, SkillSummary

SKILLS_BLOCK_START = "<available_skills>"
SKILLS_BLOCK_HEADER = "# unchain generated skills catalog v1"
SKILLS_BLOCK_END = "</available_skills>"

ACTIVE_BLOCK_START = "<active_skills>"
ACTIVE_BLOCK_HEADER = "# unchain generated active skills v1"
ACTIVE_BLOCK_END = "</active_skills>"

SKILL_CONTENT_START = "<skill_content"
SKILL_CONTENT_END = "</skill_content>"
SKILL_LOADED_START = "<skill_loaded"

_SKILL_LOADED_END = "</skill_loaded>"
_ACTIVE_HEADER_PREFIX = "# unchain generated active skills v"

_NO_BASE_DIR_TEXT = "This skill has no resource directory."
_BASE_DIR_PREFIX = "Base directory for this skill: "
_BASE_DIR_SUFFIX = (
    "Resolve relative paths mentioned by this skill against the base directory before using them. "
    "Load referenced resources only as needed."
)

_REVISION_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

_ENTRY_RE = re.compile(
    r'<skill_content name="(?P<name>[^"]*)" source="(?P<source>[^"]*)" source_id="(?P<source_id>[^"]*)" '
    r'revision="(?P<revision>[^"]*)" activation="(?P<activation>[^"]*)" tools="(?P<tools>[^"]*)">\n'
    r"<skill_resources>\n(?P<resources>.*?)\n</skill_resources>\n"
    r"\n"
    r"<skill_instructions>\n(?P<body>.*?)\n</skill_instructions>\n"
    r"</skill_content>",
    re.DOTALL,
)

_SKILL_LOADED_SENTENCES = {
    "activated": (
        'Skill "{name}" is now active. Its full instructions are in the <active_skills> system block; '
        "follow them for the rest of this run."
    ),
    "already_active": (
        'Skill "{name}" was already active at this revision; its instructions are already in the '
        "<active_skills> system block."
    ),
    "superseded": (
        'Skill "{name}" was re-activated at a new revision; the <active_skills> system block now holds '
        "the updated instructions."
    ),
}


def escape_attr(value: str) -> str:
    """Escape a string for embedding as an XML-like attribute value."""
    value = value.replace("&", "&amp;")
    value = value.replace('"', "&quot;")
    value = value.replace("<", "&lt;")
    value = value.replace(">", "&gt;")
    return value


def unescape_attr(value: str) -> str:
    """Exact inverse of ``escape_attr``."""
    value = value.replace("&quot;", '"')
    value = value.replace("&lt;", "<")
    value = value.replace("&gt;", ">")
    value = value.replace("&amp;", "&")
    return value


def normalize_description(text: str, *, max_length: int) -> str:
    """Collapse whitespace to single spaces and truncate with an ellipsis if needed."""
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) > max_length:
        truncated = collapsed[: max_length - 1].rstrip()
        return f"{truncated}…"
    return collapsed


def render_skill_catalog(
    skills: Iterable[SkillSummary],
    *,
    tool_name: str,
    description_max_length: int,
) -> str:
    """Render the ``<available_skills>`` catalog block, or "" if nothing is model-invocable."""
    invocable = sorted(
        (skill for skill in skills if skill.model_invocable),
        key=lambda skill: skill.name,
    )
    if not invocable:
        return ""

    lines: list[str] = [
        SKILLS_BLOCK_START,
        SKILLS_BLOCK_HEADER,
        "A skill is a reusable set of task-specific instructions. The following skills are available in this session:",
        "",
    ]
    for skill in invocable:
        description = normalize_description(skill.description, max_length=description_max_length)
        lines.append(f"- `{skill.name}`: {description}")
    lines.append("")
    lines.append(
        f"If the user names a skill, or the task clearly matches a skill's description, call the `{tool_name}` "
        "tool with the exact skill name before taking task actions. Load all applicable skills, then follow "
        "their full instructions. This catalog contains summaries only; do not infer or follow a skill's "
        "instructions until it has been loaded."
    )
    lines.append(
        "Skills a user invokes directly, and skills you load, appear in the <active_skills> system block; "
        "follow that block and do not load a skill that is already listed there."
    )
    lines.append(SKILLS_BLOCK_END)
    return "\n".join(lines)


def is_skill_catalog_message(message: object) -> bool:
    """True when ``message`` is a system message carrying the ``<available_skills>`` block."""
    if not isinstance(message, dict) or message.get("role") != "system":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    return (
        stripped.startswith(SKILLS_BLOCK_START)
        and SKILLS_BLOCK_HEADER in stripped
        and stripped.endswith(SKILLS_BLOCK_END)
    )


def render_skill_content(active: ActiveSkill) -> str:
    """Render one ``<skill_content>`` entry for the active-skills block.

    Raises ValueError("body_delimiter") when the body contains a literal
    closing delimiter (``</skill_instructions>`` or ``</skill_content>``),
    which would otherwise break the block's strict parser.
    """
    if "</skill_instructions>" in active.body or "</skill_content>" in active.body:
        raise ValueError("body_delimiter")

    if active.base_dir is not None:
        resources = f"{_BASE_DIR_PREFIX}{active.base_dir}\n{_BASE_DIR_SUFFIX}"
    else:
        resources = _NO_BASE_DIR_TEXT

    opening = (
        f'{SKILL_CONTENT_START} name="{escape_attr(active.identity.name)}" '
        f'source="{escape_attr(active.identity.source)}" '
        f'source_id="{escape_attr(active.identity.source_id)}" '
        f'revision="{escape_attr(active.revision)}" '
        f'activation="{escape_attr(active.activation)}" '
        f'tools="{escape_attr(",".join(active.tools))}">'
    )

    lines = [
        opening,
        "<skill_resources>",
        resources,
        "</skill_resources>",
        "",
        "<skill_instructions>",
        active.body.strip("\n"),
        "</skill_instructions>",
        SKILL_CONTENT_END,
    ]
    return "\n".join(lines)


_PROCESSED_TURN_RE = re.compile(r"^[0-9]+:[0-9a-f]{8,16}$")


def render_active_skills_block(
    active: Sequence[ActiveSkill], *, processed_turn: str | None = None
) -> str:
    """Render the full ``<active_skills>`` system block, or "" if nothing is active.

    ``processed_turn`` (``"<user-turn ordinal>:<text hash>"``) records the last
    real user turn whose ``/name`` tokens were resolved; it rides the header
    line so replay of that same turn never re-resolves against the live
    registry (only a genuinely new turn may pick a new source or revision).
    """
    if not active:
        return ""
    header = ACTIVE_BLOCK_HEADER
    if processed_turn:
        if not _PROCESSED_TURN_RE.match(processed_turn):
            raise ValueError("processed_turn must look like '<ordinal>:<hex>'")
        header = f"{ACTIVE_BLOCK_HEADER} turn={processed_turn}"

    entries = "\n\n".join(render_skill_content(item) for item in active)
    lines = [
        ACTIVE_BLOCK_START,
        header,
        "These skills were activated in this run. Follow their instructions; they stay in force until the run ends.",
        "",
        entries,
        ACTIVE_BLOCK_END,
    ]
    return "\n".join(lines)


def is_active_skills_message(message: object) -> bool:
    """True when ``message`` is a system message carrying the ``<active_skills>`` block."""
    if not isinstance(message, dict) or message.get("role") != "system":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    # Any version header is recognised here; `parse_active_skills_block` is the
    # strict gate that rejects unsupported versions instead of ignoring them.
    return (
        stripped.startswith(ACTIVE_BLOCK_START + "\n" + _ACTIVE_HEADER_PREFIX)
        and stripped.endswith(ACTIVE_BLOCK_END)
    )


class ActiveSkillsParseError(ValueError):
    """Raised when a stored ``<active_skills>`` block cannot be parsed strictly."""


def _parse_resources(text: str) -> Path | None:
    if text == _NO_BASE_DIR_TEXT:
        return None
    if text.startswith(_BASE_DIR_PREFIX):
        rest = text[len(_BASE_DIR_PREFIX):]
        parts = rest.split("\n", 1)
        if len(parts) == 2 and parts[1] == _BASE_DIR_SUFFIX:
            return Path(parts[0])
    raise ActiveSkillsParseError(f"malformed skill_resources block: {text!r}")


def parse_active_skills_block(content: str) -> tuple[ActiveSkill, ...]:
    """Strict inverse of ``render_active_skills_block``.

    Raises ActiveSkillsParseError on any structural deviation: wrong start/
    end delimiters, an unsupported snapshot version, malformed or reordered
    entries, non-blank leftovers between entries, or an invalid revision.
    """
    lines = content.split("\n")
    if not lines or lines[0] != ACTIVE_BLOCK_START:
        raise ActiveSkillsParseError("active skills block must start with <active_skills>")
    if len(lines) < 2:
        raise ActiveSkillsParseError("active skills block is missing its version header")

    header_line = lines[1]
    if not header_line.startswith(_ACTIVE_HEADER_PREFIX):
        raise ActiveSkillsParseError("active skills block is missing its version header")
    header_rest = header_line[len(_ACTIVE_HEADER_PREFIX):].strip()
    version_digits, _sep, header_extra = header_rest.partition(" ")
    version_token = "v" + version_digits
    if version_token != "v1":
        raise ActiveSkillsParseError(f"unsupported active skills snapshot version: {version_token}")
    if header_extra and not (
        header_extra.startswith("turn=") and _PROCESSED_TURN_RE.match(header_extra[5:])
    ):
        raise ActiveSkillsParseError("active skills block header carries an unknown attribute")

    last_index = -1
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            last_index = idx
            break
    if last_index < 0 or lines[last_index] != ACTIVE_BLOCK_END:
        raise ActiveSkillsParseError("active skills block must end with </active_skills>")

    # Fixed layout below the header: exactly one summary sentence line, then
    # exactly one blank line, then the entries region. Requiring the blank
    # line (rather than just scanning for "<skill_content" anywhere) matters:
    # it keeps a corrupted first entry from silently being swallowed as if it
    # were part of the unvalidated summary sentence.
    if last_index < 4:
        raise ActiveSkillsParseError("active skills block is missing its summary line or entries")
    if lines[3] != "":
        raise ActiveSkillsParseError("active skills block is missing the blank line before its entries")

    entries_region = "\n".join(lines[4:last_index])

    matches = list(_ENTRY_RE.finditer(entries_region))
    if not matches:
        raise ActiveSkillsParseError("no active skill entries found in active skills block")

    if matches[0].start() != 0:
        raise ActiveSkillsParseError("unexpected content before the first active skill entry")

    for left, right in zip(matches, matches[1:]):
        gap = entries_region[left.end() : right.start()]
        if gap.strip():
            raise ActiveSkillsParseError("unexpected content between active skill entries")

    trailing_gap = entries_region[matches[-1].end() :]
    if trailing_gap.strip():
        raise ActiveSkillsParseError("unexpected content after the last active skill entry")

    parsed: list[ActiveSkill] = []
    for match in matches:
        revision = unescape_attr(match.group("revision"))
        if not _REVISION_RE.match(revision):
            raise ActiveSkillsParseError(f"invalid skill revision in active skills block: {revision!r}")

        tools_raw = unescape_attr(match.group("tools"))
        tools = tuple(tools_raw.split(",")) if tools_raw else ()

        base_dir = _parse_resources(match.group("resources"))

        identity = SkillIdentity(
            source=unescape_attr(match.group("source")),
            source_id=unescape_attr(match.group("source_id")),
            name=unescape_attr(match.group("name")),
        )
        parsed.append(
            ActiveSkill(
                identity=identity,
                revision=revision,
                activation=unescape_attr(match.group("activation")),
                body=match.group("body"),
                base_dir=base_dir,
                tools=tools,
            )
        )

    return tuple(parsed)


def parse_active_skills_turn(content: str) -> str | None:
    """The ``turn=`` marker of a (already validated) active-skills block, or None."""
    lines = content.split("\n")
    if len(lines) < 2 or not lines[1].startswith(_ACTIVE_HEADER_PREFIX):
        return None
    _version, _sep, extra = lines[1][len(_ACTIVE_HEADER_PREFIX):].strip().partition(" ")
    if extra.startswith("turn=") and _PROCESSED_TURN_RE.match(extra[5:]):
        return extra[5:]
    return None


def render_skill_loaded_envelope(*, name: str, revision: str, status: str) -> str:
    """Render the no-body tool-result envelope reported after a `skill` tool call."""
    sentence_template = _SKILL_LOADED_SENTENCES.get(status)
    if sentence_template is None:
        raise ValueError(f"unknown skill loaded status: {status!r}")
    sentence = sentence_template.format(name=name)
    return (
        f'{SKILL_LOADED_START} name="{escape_attr(name)}" revision="{escape_attr(revision)}" status="{status}">\n'
        f"{sentence}\n"
        f"{_SKILL_LOADED_END}"
    )


def is_skill_content_message(message: object) -> bool:
    """True for a synthetic user-role message carrying skill content/loaded envelopes.

    Used to exclude synthetic text from "latest real user turn" lookups
    (e.g. `/name` scanning), which must only ever see the user's own words.
    """
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    return stripped.startswith(SKILL_CONTENT_START) or stripped.startswith(SKILL_LOADED_START)


__all__ = [
    "SKILLS_BLOCK_START",
    "SKILLS_BLOCK_HEADER",
    "SKILLS_BLOCK_END",
    "ACTIVE_BLOCK_START",
    "ACTIVE_BLOCK_HEADER",
    "ACTIVE_BLOCK_END",
    "SKILL_CONTENT_START",
    "SKILL_CONTENT_END",
    "SKILL_LOADED_START",
    "escape_attr",
    "unescape_attr",
    "normalize_description",
    "render_skill_catalog",
    "is_skill_catalog_message",
    "render_skill_content",
    "render_active_skills_block",
    "is_active_skills_message",
    "ActiveSkillsParseError",
    "parse_active_skills_block",
    "parse_active_skills_turn",
    "render_skill_loaded_envelope",
    "is_skill_content_message",
]
