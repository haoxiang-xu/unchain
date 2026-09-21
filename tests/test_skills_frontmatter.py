from __future__ import annotations

from pathlib import Path

import pytest

from unchain.skills.frontmatter import (
    ParsedSkillFile,
    SkillParseError,
    coerce_bool,
    parse_skill_file,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "skills"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _parse(name: str) -> ParsedSkillFile:
    return parse_skill_file(_load(name))


# ---------------------------------------------------------------------------
# Plain scalars
# ---------------------------------------------------------------------------


def test_plain_name_and_description_with_body_edge_stripping():
    result = _parse("plain.md")

    assert result.fields == {
        "name": "demo-skill",
        "description": "A simple demo skill for greeting the user.",
    }
    assert result.body == (
        "Follow these steps when the user asks for a greeting.\n\n"
        "1. Read their name from the conversation.\n"
        "2. Reply with a short, friendly greeting."
    )


# ---------------------------------------------------------------------------
# Quoted scalars, escapes, inline comments
# ---------------------------------------------------------------------------


def test_quoted_scalars_keep_internal_colons_and_drop_comments():
    result = _parse("quoted_comments.md")

    assert result.fields == {
        "name": "quote-demo",
        "description": "Use when: x",
        "foo": "bar#notcomment",
        "note": "hello world",
        "quoted-single": "It's a test",
    }
    assert result.body == "Body text."


def test_double_quoted_scalar_supports_backslash_escapes():
    result = _parse("escapes.md")

    assert result.fields["label"] == 'Quote: " Backslash: \\ Tab:\t Newline:\n end'


# ---------------------------------------------------------------------------
# Block scalars
# ---------------------------------------------------------------------------


def test_literal_block_scalar_chomping_variants():
    result = _parse("block_literal.md")

    assert result.fields["clip"] == "Line one.\nLine two.\n"
    assert result.fields["strip"] == "Line one.\nLine two."
    assert result.fields["keep"] == "Line one.\n\n"
    assert result.fields["tags"] == "after"


def test_folded_block_scalar_chomping_variants():
    result = _parse("block_folded.md")

    assert result.fields["clip"] == "Line one. continues here.\nNew paragraph.\n"
    assert result.fields["strip"] == "Only one line here."
    assert result.fields["keep"] == "Folded line.\n\n"
    assert result.fields["next"] == "value"


# ---------------------------------------------------------------------------
# Nested mappings, block lists, flow sequences
# ---------------------------------------------------------------------------


def test_nested_mapping_list_and_flow_sequence():
    result = _parse("nested_and_list.md")

    assert result.fields["interface"] == {
        "display_name": "Ticket Buddy",
        "icon": "ticket",
    }
    assert result.fields["tags"] == ["alpha", "beta", "gamma"]
    assert result.fields["allowed-tools"] == ["Bash", "Read", "Write"]


# ---------------------------------------------------------------------------
# Unknown keys and policy-key string passthrough
# ---------------------------------------------------------------------------


def test_unknown_keys_preserved_verbatim_and_policy_keys_stay_strings():
    result = _parse("unknown_keys.md")

    assert result.fields["custom-field"] == "some value"
    assert result.fields["another-custom"] == "42"
    assert result.fields["disable-model-invocation"] == "true"
    assert result.fields["user-invocable"] == "false"
    # The two policy keys stay strings in `fields`; coercion is a separate step.
    assert coerce_bool(result.fields["disable-model-invocation"], default=False) is True
    assert coerce_bool(result.fields["user-invocable"], default=True) is False


# ---------------------------------------------------------------------------
# Real-world style fixture
# ---------------------------------------------------------------------------


def test_real_world_fixture_folded_description_metadata_and_body_fences():
    result = _parse("real_world.md")

    assert result.fields["name"] == "pdf-form-filler"
    assert result.fields["description"] == (
        "Fill in PDF forms with structured data, validate required fields, and "
        "export a completed copy without altering the original template.\n"
    )
    assert result.fields["metadata"] == {
        "category": "documents",
        "license": "Apache-2.0",
        "version": "1.2.0",
    }
    assert result.fields["allowed-tools"] == ["Read", "Write", "Bash"]

    # The body's own `---` lines (inside a fenced code block) must not be
    # mistaken for frontmatter delimiters, and code fences/links survive
    # untouched.
    assert result.body.startswith("# PDF Form Filler\n\n")
    assert "```bash\npython fill_form.py --input form.pdf --output filled.pdf\n" in result.body
    assert "---\nnot a real fence, just literal text with dashes\n---\n```" in result.body
    assert "[PDF toolkit reference](https://example.com/docs/pdf-toolkit)" in result.body


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_missing_frontmatter_is_rejected():
    with pytest.raises(SkillParseError, match=r"missing YAML frontmatter delimited by ---"):
        _parse("reject_missing_frontmatter.md")


def test_unterminated_frontmatter_is_rejected():
    with pytest.raises(SkillParseError, match=r"unterminated YAML frontmatter"):
        _parse("reject_unterminated.md")


def test_tab_indentation_is_rejected():
    with pytest.raises(SkillParseError, match=r"tab"):
        _parse("reject_tab_indent.md")


def test_yaml_tag_is_rejected():
    with pytest.raises(SkillParseError, match=r"tag"):
        _parse("reject_tag.md")


def test_yaml_anchor_is_rejected():
    with pytest.raises(SkillParseError, match=r"anchor"):
        _parse("reject_anchor.md")


def test_yaml_alias_is_rejected():
    with pytest.raises(SkillParseError, match=r"alias"):
        _parse("reject_alias.md")


def test_duplicate_policy_key_is_rejected():
    with pytest.raises(SkillParseError, match=r"duplicate"):
        _parse("reject_duplicate_name.md")


def test_invalid_top_level_line_is_rejected():
    with pytest.raises(SkillParseError, match=r"invalid frontmatter line"):
        _parse("reject_invalid_line.md")


def test_flow_mapping_is_rejected():
    with pytest.raises(SkillParseError, match=r"flow mapping"):
        parse_skill_file("---\nfoo: {a: 1}\n---\nBody.\n")


# ---------------------------------------------------------------------------
# coerce_bool
# ---------------------------------------------------------------------------


def test_coerce_bool_none_and_blank_use_default():
    assert coerce_bool(None, default=True) is True
    assert coerce_bool(None, default=False) is False
    assert coerce_bool("", default=True) is True
    assert coerce_bool("   ", default=False) is False


@pytest.mark.parametrize("value", ["true", "TRUE", "True", "yes", "Yes", "on", "ON", "1"])
def test_coerce_bool_true_spellings(value):
    assert coerce_bool(value, default=False) is True


@pytest.mark.parametrize("value", ["false", "FALSE", "False", "no", "No", "off", "OFF", "0"])
def test_coerce_bool_false_spellings(value):
    assert coerce_bool(value, default=True) is False


def test_coerce_bool_rejects_unrecognized_string():
    with pytest.raises(SkillParseError, match=r"expected a boolean"):
        coerce_bool("maybe", default=True)


def test_coerce_bool_rejects_non_string_value():
    with pytest.raises(SkillParseError, match=r"expected a boolean"):
        coerce_bool(["x"], default=True)
