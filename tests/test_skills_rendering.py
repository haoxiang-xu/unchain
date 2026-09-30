from __future__ import annotations

from pathlib import Path

import pytest

from unchain.skills.models import (
    SKILL_DESCRIPTION_MAX_LENGTH,
    SKILL_DIAGNOSTIC_KINDS,
    SKILL_NAME_MAX_LENGTH,
    ActiveSkill,
    LoadedSkill,
    SkillDiagnostic,
    SkillIdentity,
    SkillInventory,
    SkillSummary,
    compute_inventory_revision,
    compute_skill_revision,
    description_error,
    is_valid_skill_name,
)
from unchain.skills.rendering import (
    ACTIVE_BLOCK_END,
    ACTIVE_BLOCK_HEADER,
    ACTIVE_BLOCK_START,
    SKILL_CONTENT_START,
    SKILL_LOADED_START,
    SKILLS_BLOCK_END,
    SKILLS_BLOCK_HEADER,
    SKILLS_BLOCK_START,
    ActiveSkillsParseError,
    escape_attr,
    is_active_skills_message,
    is_skill_catalog_message,
    is_skill_content_message,
    normalize_description,
    parse_active_skills_block,
    render_active_skills_block,
    render_skill_catalog,
    render_skill_content,
    render_skill_loaded_envelope,
    unescape_attr,
)

# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["a", "a1", "plan-first", "a" * SKILL_NAME_MAX_LENGTH],
)
def test_is_valid_skill_name_accepts(name: str) -> None:
    assert is_valid_skill_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "Plan",
        "plan_first",
        "-plan",
        "plan-",
        "plan--first",
        "a" * (SKILL_NAME_MAX_LENGTH + 1),
        "",
        123,
        None,
        ["plan-first"],
    ],
)
def test_is_valid_skill_name_rejects(name: object) -> None:
    assert is_valid_skill_name(name) is False


def test_description_error_accepts_valid_text() -> None:
    assert description_error("A short, useful description.") is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "x" * (SKILL_DESCRIPTION_MAX_LENGTH + 1),
        123,
        None,
        ["a description"],
    ],
)
def test_description_error_rejects(text: object) -> None:
    error = description_error(text)
    assert isinstance(error, str)
    assert error


def test_description_error_boundary_length_ok() -> None:
    assert description_error("x" * SKILL_DESCRIPTION_MAX_LENGTH) is None


def test_skill_identity_key() -> None:
    identity = SkillIdentity(source="filesystem", source_id="/abs/path/SKILL.md", name="plan-first")
    assert identity.key == "filesystem:/abs/path/SKILL.md:plan-first"


def test_compute_skill_revision_is_deterministic() -> None:
    kwargs = dict(body="Do the thing.", tools=("a", "b"), model_invocable=True, user_invocable=False)
    first = compute_skill_revision(**kwargs)
    second = compute_skill_revision(**kwargs)
    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_compute_skill_revision_sensitive_to_body() -> None:
    base = compute_skill_revision(body="A", tools=(), model_invocable=True, user_invocable=True)
    changed = compute_skill_revision(body="B", tools=(), model_invocable=True, user_invocable=True)
    assert base != changed


def test_compute_skill_revision_sensitive_to_tools() -> None:
    base = compute_skill_revision(body="A", tools=("x",), model_invocable=True, user_invocable=True)
    changed = compute_skill_revision(body="A", tools=("y",), model_invocable=True, user_invocable=True)
    assert base != changed


def test_compute_skill_revision_sensitive_to_model_invocable() -> None:
    base = compute_skill_revision(body="A", tools=(), model_invocable=True, user_invocable=True)
    changed = compute_skill_revision(body="A", tools=(), model_invocable=False, user_invocable=True)
    assert base != changed


def test_compute_skill_revision_sensitive_to_user_invocable() -> None:
    base = compute_skill_revision(body="A", tools=(), model_invocable=True, user_invocable=True)
    changed = compute_skill_revision(body="A", tools=(), model_invocable=True, user_invocable=False)
    assert base != changed


def test_skill_summary_delegates_identity_properties() -> None:
    identity = SkillIdentity(source="toolkit", source_id="demo", name="echo-helper")
    summary = SkillSummary(
        identity=identity,
        description="Echoes text.",
        rank=600,
        path=None,
        base_dir=None,
    )
    assert summary.name == "echo-helper"
    assert summary.source == "toolkit"
    assert summary.source_id == "demo"


def test_loaded_skill_defaults() -> None:
    identity = SkillIdentity(source="toolkit", source_id="demo", name="echo-helper")
    summary = SkillSummary(identity=identity, description="Echoes text.", rank=600, path=None, base_dir=None)
    loaded = LoadedSkill(summary=summary, body="Echo it.", revision="sha256:" + "0" * 64)
    assert loaded.tools == ()
    assert loaded.metadata == {}


def test_skill_diagnostic_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        SkillDiagnostic(kind="not-a-real-kind", name="x", source="filesystem", source_id="id", message="oops")


@pytest.mark.parametrize("kind", sorted(SKILL_DIAGNOSTIC_KINDS))
def test_skill_diagnostic_accepts_known_kinds(kind: str) -> None:
    diagnostic = SkillDiagnostic(kind=kind, name="x", source="filesystem", source_id="id", message="note")
    assert diagnostic.kind == kind


def test_compute_inventory_revision_is_order_independent() -> None:
    pairs = [("filesystem:/a/SKILL.md:plan-first", "sha256:" + "1" * 64), ("toolkit:demo:echo", "sha256:" + "2" * 64)]
    forward = compute_inventory_revision(pairs)
    backward = compute_inventory_revision(list(reversed(pairs)))
    assert forward == backward


def test_compute_inventory_revision_sensitive_to_contents() -> None:
    pairs_a = [("filesystem:/a/SKILL.md:plan-first", "sha256:" + "1" * 64)]
    pairs_b = [("filesystem:/a/SKILL.md:plan-first", "sha256:" + "9" * 64)]
    assert compute_inventory_revision(pairs_a) != compute_inventory_revision(pairs_b)


def test_skill_inventory_holds_its_fields() -> None:
    identity = SkillIdentity(source="filesystem", source_id="/a/SKILL.md", name="plan-first")
    summary = SkillSummary(identity=identity, description="Plans.", rank=100, path=Path("/a/SKILL.md"), base_dir=Path("/a"))
    diagnostic = SkillDiagnostic(kind="shadowed", name="plan-first", source="filesystem", source_id="/b/SKILL.md", message="shadowed by rank 100")
    inventory = SkillInventory(skills=(summary,), diagnostics=(diagnostic,), revision="sha256:" + "0" * 64)
    assert inventory.skills == (summary,)
    assert inventory.diagnostics == (diagnostic,)


# ---------------------------------------------------------------------------
# rendering.py: escape/unescape/normalize
# ---------------------------------------------------------------------------


def test_escape_attr_escapes_all_four_chars() -> None:
    assert escape_attr('&"<>') == "&amp;&quot;&lt;&gt;"


def test_escape_unescape_roundtrip() -> None:
    raw = 'value with & "quotes" <tags> & more'
    assert unescape_attr(escape_attr(raw)) == raw


def test_unescape_attr_is_exact_inverse_for_literal_entity_text() -> None:
    # A raw value that itself contains the literal text "&quot;" must survive
    # an escape/unescape round trip unchanged.
    raw = 'literal &quot; text'
    escaped = escape_attr(raw)
    assert escaped == "literal &amp;quot; text"
    assert unescape_attr(escaped) == raw


def test_normalize_description_collapses_whitespace() -> None:
    assert normalize_description("  a   b\tc\n\nd  ", max_length=100) == "a b c d"


def test_normalize_description_truncates_with_ellipsis() -> None:
    text = "x" * 20
    result = normalize_description(text, max_length=10)
    assert result == "x" * 9 + "…"
    assert len(result) == 10


def test_normalize_description_no_truncation_when_short() -> None:
    assert normalize_description("short", max_length=10) == "short"


# ---------------------------------------------------------------------------
# rendering.py: catalog
# ---------------------------------------------------------------------------


def _summary(name: str, description: str, *, model_invocable: bool = True, rank: int = 100) -> SkillSummary:
    identity = SkillIdentity(source="filesystem", source_id=f"/skills/{name}/SKILL.md", name=name)
    return SkillSummary(
        identity=identity,
        description=description,
        rank=rank,
        path=Path(f"/skills/{name}/SKILL.md"),
        base_dir=Path(f"/skills/{name}"),
        model_invocable=model_invocable,
    )


def test_render_skill_catalog_empty_when_no_skills() -> None:
    assert render_skill_catalog([], tool_name="skill", description_max_length=500) == ""


def test_render_skill_catalog_empty_when_none_model_invocable() -> None:
    skills = [_summary("plan-first", "Plans first.", model_invocable=False)]
    assert render_skill_catalog(skills, tool_name="skill", description_max_length=500) == ""


def test_render_skill_catalog_exact_layout_sorted_and_filtered() -> None:
    skills = [
        _summary("zeta-skill", "Does zeta things."),
        _summary("alpha-skill", "Does alpha things."),
        _summary("hidden-skill", "Never shown.", model_invocable=False),
    ]
    rendered = render_skill_catalog(skills, tool_name="skill", description_max_length=500)

    expected = "\n".join(
        [
            SKILLS_BLOCK_START,
            SKILLS_BLOCK_HEADER,
            "A skill is a reusable set of task-specific instructions. The following skills are available in this session:",
            "",
            "- `alpha-skill`: Does alpha things.",
            "- `zeta-skill`: Does zeta things.",
            "",
            "If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool "
            "with the exact skill name before taking task actions. Load all applicable skills, then follow "
            "their full instructions. This catalog contains summaries only; do not infer or follow a skill's "
            "instructions until it has been loaded.",
            "Skills a user invokes directly, and skills you load, appear in the <active_skills> system block; "
            "follow that block and do not load a skill that is already listed there.",
            SKILLS_BLOCK_END,
        ]
    )
    assert rendered == expected


def test_render_skill_catalog_truncates_long_descriptions() -> None:
    skills = [_summary("plan-first", "x" * 600)]
    rendered = render_skill_catalog(skills, tool_name="skill", description_max_length=20)
    line = [ln for ln in rendered.splitlines() if ln.startswith("- `plan-first`")][0]
    assert line == "- `plan-first`: " + "x" * 19 + "…"


def test_is_skill_catalog_message_true_for_rendered_catalog() -> None:
    skills = [_summary("plan-first", "Plans first.")]
    rendered = render_skill_catalog(skills, tool_name="skill", description_max_length=500)
    message = {"role": "system", "content": rendered}
    assert is_skill_catalog_message(message) is True


def test_is_skill_catalog_message_false_for_user_role() -> None:
    skills = [_summary("plan-first", "Plans first.")]
    rendered = render_skill_catalog(skills, tool_name="skill", description_max_length=500)
    message = {"role": "user", "content": rendered}
    assert is_skill_catalog_message(message) is False


def test_is_skill_catalog_message_false_for_non_dict_and_missing_content() -> None:
    assert is_skill_catalog_message("not a message") is False
    assert is_skill_catalog_message({"role": "system", "content": 123}) is False
    assert is_skill_catalog_message({"role": "system", "content": "unrelated text"}) is False


# ---------------------------------------------------------------------------
# rendering.py: render_skill_content
# ---------------------------------------------------------------------------


def test_render_skill_content_filesystem_skill_exact_output() -> None:
    identity = SkillIdentity(
        source="filesystem",
        source_id="/repo/.unchain/skills/plan-first/SKILL.md",
        name="plan-first",
    )
    active = ActiveSkill(
        identity=identity,
        revision="sha256:" + "0" * 64,
        activation="user",
        body="Step one.\nStep two.",
        base_dir=Path("/repo/.unchain/skills/plan-first"),
        tools=("web_search",),
    )

    expected = "\n".join(
        [
            '<skill_content name="plan-first" source="filesystem" '
            'source_id="/repo/.unchain/skills/plan-first/SKILL.md" '
            'revision="sha256:' + "0" * 64 + '" activation="user" tools="web_search">',
            "<skill_resources>",
            "Base directory for this skill: /repo/.unchain/skills/plan-first",
            "Resolve relative paths mentioned by this skill against the base directory before using them. "
            "Load referenced resources only as needed.",
            "</skill_resources>",
            "",
            "<skill_instructions>",
            "Step one.\nStep two.",
            "</skill_instructions>",
            "</skill_content>",
        ]
    )
    assert render_skill_content(active) == expected


def test_render_skill_content_no_base_dir_skill_exact_output() -> None:
    identity = SkillIdentity(source="toolkit", source_id="demo-toolkit", name="echo-helper")
    active = ActiveSkill(
        identity=identity,
        revision="sha256:" + "f" * 64,
        activation="tool:call_42",
        body="Just do it.",
        base_dir=None,
        tools=(),
    )

    expected = "\n".join(
        [
            '<skill_content name="echo-helper" source="toolkit" source_id="demo-toolkit" '
            'revision="sha256:' + "f" * 64 + '" activation="tool:call_42" tools="">',
            "<skill_resources>",
            "This skill has no resource directory.",
            "</skill_resources>",
            "",
            "<skill_instructions>",
            "Just do it.",
            "</skill_instructions>",
            "</skill_content>",
        ]
    )
    assert render_skill_content(active) == expected


def test_render_skill_content_escapes_quote_and_lt_in_attributes() -> None:
    identity = SkillIdentity(source="toolkit", source_id='weird"<id>', name="echo-helper")
    active = ActiveSkill(
        identity=identity,
        revision="sha256:" + "1" * 64,
        activation="user",
        body="Body text.",
        base_dir=None,
        tools=(),
    )
    rendered = render_skill_content(active)
    assert 'source_id="weird&quot;&lt;id&gt;"' in rendered
    assert '"<id>"' not in rendered


def test_render_skill_content_rejects_skill_instructions_delimiter() -> None:
    identity = SkillIdentity(source="filesystem", source_id="/a/SKILL.md", name="plan-first")
    active = ActiveSkill(
        identity=identity,
        revision="sha256:" + "0" * 64,
        activation="user",
        body="Text with </skill_instructions> inside it.",
        base_dir=None,
        tools=(),
    )
    with pytest.raises(ValueError, match="body_delimiter"):
        render_skill_content(active)


def test_render_skill_content_rejects_skill_content_delimiter() -> None:
    identity = SkillIdentity(source="filesystem", source_id="/a/SKILL.md", name="plan-first")
    active = ActiveSkill(
        identity=identity,
        revision="sha256:" + "0" * 64,
        activation="user",
        body="Text with </skill_content> inside it.",
        base_dir=None,
        tools=(),
    )
    with pytest.raises(ValueError, match="body_delimiter"):
        render_skill_content(active)


# ---------------------------------------------------------------------------
# rendering.py: active skills block round trip + strict parse rejections
# ---------------------------------------------------------------------------


def _active_pair() -> tuple[ActiveSkill, ActiveSkill]:
    first = ActiveSkill(
        identity=SkillIdentity(
            source="filesystem",
            source_id="/repo/.unchain/skills/plan-first/SKILL.md",
            name="plan-first",
        ),
        revision=compute_skill_revision(
            body="Plan the work.\n\nThen execute it.",
            tools=("web_search", "read_file"),
            model_invocable=True,
            user_invocable=True,
        ),
        activation="user",
        body="Plan the work.\n\nThen execute it.",
        base_dir=Path("/repo/.unchain/skills/plan-first"),
        tools=("web_search", "read_file"),
    )
    second = ActiveSkill(
        identity=SkillIdentity(source="toolkit", source_id="demo-toolkit", name="unicode-skill"),
        revision=compute_skill_revision(
            body="Héllo wörld 你好.\n\nSecond paragraph.",
            tools=(),
            model_invocable=True,
            user_invocable=False,
        ),
        activation="tool:call_7",
        body="Héllo wörld 你好.\n\nSecond paragraph.",
        base_dir=None,
        tools=(),
    )
    return first, second


def test_render_active_skills_block_empty_when_no_active() -> None:
    assert render_active_skills_block(()) == ""


def test_render_then_parse_active_skills_block_round_trips() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    parsed = parse_active_skills_block(rendered)
    assert parsed == active


def test_is_active_skills_message_true_and_false() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    assert is_active_skills_message({"role": "system", "content": rendered}) is True
    assert is_active_skills_message({"role": "user", "content": rendered}) is False
    assert is_active_skills_message({"role": "system", "content": "nope"}) is False


def test_parse_active_skills_block_rejects_wrong_header_version() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    tampered = rendered.replace(ACTIVE_BLOCK_HEADER, "# unchain generated active skills v2")
    with pytest.raises(ActiveSkillsParseError, match="unsupported active skills snapshot version: v2"):
        parse_active_skills_block(tampered)


def test_parse_active_skills_block_rejects_missing_end_tag() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    assert rendered.endswith(ACTIVE_BLOCK_END)
    tampered = rendered[: -len(ACTIVE_BLOCK_END)].rstrip("\n")
    with pytest.raises(ActiveSkillsParseError):
        parse_active_skills_block(tampered)


def test_parse_active_skills_block_rejects_garbage_between_entries() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    marker = "\n\n<skill_content name=\"unicode-skill\""
    index = rendered.index(marker)
    tampered = rendered[:index] + "\nSOME UNEXPECTED GARBAGE TEXT\n" + rendered[index:]
    with pytest.raises(ActiveSkillsParseError):
        parse_active_skills_block(tampered)


def test_parse_active_skills_block_rejects_bad_revision() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    good_revision = active[0].revision
    tampered = rendered.replace(f'revision="{good_revision}"', 'revision="sha256:not-valid-hex"', 1)
    with pytest.raises(ActiveSkillsParseError):
        parse_active_skills_block(tampered)


def test_parse_active_skills_block_rejects_swapped_attribute_order() -> None:
    active = _active_pair()
    rendered = render_active_skills_block(active)
    original = 'source="filesystem" source_id="/repo/.unchain/skills/plan-first/SKILL.md"'
    swapped = 'source_id="/repo/.unchain/skills/plan-first/SKILL.md" source="filesystem"'
    assert original in rendered
    tampered = rendered.replace(original, swapped, 1)
    with pytest.raises(ActiveSkillsParseError):
        parse_active_skills_block(tampered)


def test_parse_active_skills_block_rejects_wrong_start_delimiter() -> None:
    with pytest.raises(ActiveSkillsParseError):
        parse_active_skills_block("not the right start\n" + ACTIVE_BLOCK_HEADER + "\n" + ACTIVE_BLOCK_END)


# ---------------------------------------------------------------------------
# rendering.py: skill_loaded envelope
# ---------------------------------------------------------------------------


def test_render_skill_loaded_envelope_activated() -> None:
    rendered = render_skill_loaded_envelope(name="plan-first", revision="sha256:" + "0" * 64, status="activated")
    expected = "\n".join(
        [
            '<skill_loaded name="plan-first" revision="sha256:' + "0" * 64 + '" status="activated">',
            'Skill "plan-first" is now active. Its full instructions are in the <active_skills> system block; '
            "follow them for the rest of this run.",
            "</skill_loaded>",
        ]
    )
    assert rendered == expected


def test_render_skill_loaded_envelope_already_active() -> None:
    rendered = render_skill_loaded_envelope(name="plan-first", revision="sha256:" + "0" * 64, status="already_active")
    expected = "\n".join(
        [
            '<skill_loaded name="plan-first" revision="sha256:' + "0" * 64 + '" status="already_active">',
            'Skill "plan-first" was already active at this revision; its instructions are already in the '
            "<active_skills> system block.",
            "</skill_loaded>",
        ]
    )
    assert rendered == expected


def test_render_skill_loaded_envelope_superseded() -> None:
    rendered = render_skill_loaded_envelope(name="plan-first", revision="sha256:" + "0" * 64, status="superseded")
    expected = "\n".join(
        [
            '<skill_loaded name="plan-first" revision="sha256:' + "0" * 64 + '" status="superseded">',
            'Skill "plan-first" was re-activated at a new revision; the <active_skills> system block now holds '
            "the updated instructions.",
            "</skill_loaded>",
        ]
    )
    assert rendered == expected


def test_render_skill_loaded_envelope_rejects_invalid_status() -> None:
    with pytest.raises(ValueError):
        render_skill_loaded_envelope(name="plan-first", revision="sha256:" + "0" * 64, status="bogus")


# ---------------------------------------------------------------------------
# rendering.py: is_skill_content_message
# ---------------------------------------------------------------------------


def test_is_skill_content_message_true_for_skill_content() -> None:
    identity = SkillIdentity(source="filesystem", source_id="/a/SKILL.md", name="plan-first")
    active = ActiveSkill(
        identity=identity,
        revision="sha256:" + "0" * 64,
        activation="user",
        body="Body.",
        base_dir=None,
        tools=(),
    )
    content = render_skill_content(active)
    assert content.startswith(SKILL_CONTENT_START)
    message = {"role": "user", "content": content}
    assert is_skill_content_message(message) is True


def test_is_skill_content_message_true_for_skill_loaded() -> None:
    content = render_skill_loaded_envelope(name="plan-first", revision="sha256:" + "0" * 64, status="activated")
    assert content.startswith(SKILL_LOADED_START)
    message = {"role": "user", "content": content}
    assert is_skill_content_message(message) is True


def test_is_skill_content_message_false_for_wrong_role() -> None:
    content = render_skill_loaded_envelope(name="plan-first", revision="sha256:" + "0" * 64, status="activated")
    message = {"role": "assistant", "content": content}
    assert is_skill_content_message(message) is False


def test_is_skill_content_message_false_for_ordinary_user_text() -> None:
    message = {"role": "user", "content": "please run /plan-first"}
    assert is_skill_content_message(message) is False


def test_active_block_header_carries_and_validates_the_processed_turn_marker():
    from unchain.skills.rendering import parse_active_skills_turn

    entry = ActiveSkill(
        identity=SkillIdentity("toolkit", "tk", "demo"),
        revision="sha256:" + "a" * 64,
        activation="user:3:0123456789abcdef",
        body="body",
        base_dir=None,
    )
    rendered = render_active_skills_block([entry], processed_turn="3:0123456789abcdef")
    assert rendered.splitlines()[1] == "# unchain generated active skills v1 turn=3:0123456789abcdef"
    assert parse_active_skills_turn(rendered) == "3:0123456789abcdef"
    assert parse_active_skills_block(rendered) == (entry,)
    # Absent marker (older v1 blocks) still parses and reports None.
    plain = render_active_skills_block([entry])
    assert parse_active_skills_turn(plain) is None and parse_active_skills_block(plain) == (entry,)
    with pytest.raises(ValueError):
        render_active_skills_block([entry], processed_turn="not a marker")
    with pytest.raises(ActiveSkillsParseError, match="unknown attribute"):
        parse_active_skills_block(rendered.replace("turn=3:0123456789abcdef", "evil=1"))
    with pytest.raises(ActiveSkillsParseError, match="v2"):
        parse_active_skills_block(rendered.replace("v1 turn=", "v2 turn="))
