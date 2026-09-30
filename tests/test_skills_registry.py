from __future__ import annotations

import os
from pathlib import Path

import pytest

from unchain.skills.registry import (
    SKILL_FILE_NAME,
    TOOLKIT_SKILL_RANK,
    SkillRegistry,
    SkillRoot,
    SkillsConfig,
    resolve_project_root,
)
from unchain.tools import SkillDescriptor, Toolkit


def _write_skill(
    root: Path,
    name: str,
    *,
    body: str | None = None,
    description: str | None = None,
    extra: str = "",
    flat: bool = False,
) -> Path:
    """Write a SKILL.md-style fixture file, as a `<name>/SKILL.md` dir skill
    by default, or a flat `<name>.md` file when `flat=True`."""
    description = description if description is not None else f"{name} description"
    content = f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body or (name + ' body')}\n"
    if flat:
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.md"
        path.write_text(content, encoding="utf-8")
        return path
    (root / name).mkdir(parents=True, exist_ok=True)
    path = root / name / SKILL_FILE_NAME
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    return root


def _skills_dir(project: Path) -> Path:
    return project / ".unchain" / "skills"


def _agents_dir(project: Path) -> Path:
    return project / ".agents" / "skills"


def _list_for(project: Path, tmp_path: Path, **overrides):
    config = SkillsConfig(
        project_root=project, include_user_dirs=False, home=tmp_path / "home", **overrides
    )
    return SkillRegistry(config).list()


# ---------------------------------------------------------------------------
# resolve_project_root
# ---------------------------------------------------------------------------


def test_resolve_project_root_from_nested_dir_finds_git(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    nested = root / "a" / "b" / "c"
    nested.mkdir(parents=True)
    assert resolve_project_root(nested) == root.resolve()


def test_resolve_project_root_without_git_falls_back_to_start(tmp_path):
    plain = tmp_path / "no_git"
    plain.mkdir()
    assert resolve_project_root(plain) == plain.resolve()


# ---------------------------------------------------------------------------
# roots()
# ---------------------------------------------------------------------------


def test_roots_full_order_with_ranks(project, tmp_path):
    home = tmp_path / "home"
    extra1 = tmp_path / "extra1"
    extra2 = tmp_path / "extra2"
    config = SkillsConfig(project_root=project, home=home, extra_dirs=(extra1, extra2))
    registry = SkillRegistry(config)
    assert registry.roots() == (
        SkillRoot("project-unchain", 100, project / ".unchain" / "skills"),
        SkillRoot("project-agents", 200, project / ".agents" / "skills"),
        SkillRoot("custom", 300, extra1),
        SkillRoot("custom", 300, extra2),
        SkillRoot("user-unchain", 400, home / ".unchain" / "skills"),
        SkillRoot("user-agents", 500, home / ".agents" / "skills"),
    )


def test_roots_include_project_dirs_false_skips_project_roots(project, tmp_path):
    config = SkillsConfig(project_root=project, include_project_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)
    assert [r.source for r in registry.roots()] == ["user-unchain", "user-agents"]


def test_roots_include_user_dirs_false_skips_user_roots(project, tmp_path):
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)
    assert [r.source for r in registry.roots()] == ["project-unchain", "project-agents"]


def test_roots_explicit_project_root_used_as_is_even_with_parent_git(tmp_path):
    parent = tmp_path / "parent"
    (parent / ".git").mkdir(parents=True)
    explicit_root = parent / "subproject"
    explicit_root.mkdir()
    config = SkillsConfig(project_root=explicit_root, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)
    roots = registry.roots()
    assert roots[0].path == explicit_root / ".unchain" / "skills"
    assert roots[1].path == explicit_root / ".agents" / "skills"


# ---------------------------------------------------------------------------
# Winner selection, discovery shape, dedupe
# ---------------------------------------------------------------------------


def test_winner_selection_across_ranks_and_shadowed_diagnostic(project, tmp_path):
    _write_skill(_skills_dir(project), "dup", description="from project-unchain")
    _write_skill(_agents_dir(project), "dup", description="from project-agents")
    inventory = _list_for(project, tmp_path)

    assert [s.name for s in inventory.skills] == ["dup"]
    winner = inventory.skills[0]
    assert winner.identity.source == "project-unchain"
    assert winner.description == "from project-unchain"

    shadowed = [d for d in inventory.diagnostics if d.kind == "shadowed"]
    assert len(shadowed) == 1
    assert shadowed[0].source == "project-agents"
    assert "project-unchain" in shadowed[0].message


def test_same_rank_tie_broken_by_identity_key(project, tmp_path):
    extra1 = tmp_path / "extra1"
    extra2 = tmp_path / "extra2"
    _write_skill(extra1, "dup", description="from extra1")
    _write_skill(extra2, "dup", description="from extra2")
    config = SkillsConfig(
        project_root=project,
        include_project_dirs=False,
        include_user_dirs=False,
        extra_dirs=(extra1, extra2),
    )
    inventory = SkillRegistry(config).list()

    (winner,) = inventory.skills
    path1 = (extra1 / "dup" / SKILL_FILE_NAME).resolve()
    path2 = (extra2 / "dup" / SKILL_FILE_NAME).resolve()
    key1 = f"custom:{path1}:dup"
    key2 = f"custom:{path2}:dup"
    expected_path = path1 if key1 < key2 else path2
    assert winner.source_id == str(expected_path)


def test_flat_md_file_accepted(project, tmp_path):
    skills_dir = _skills_dir(project)
    _write_skill(skills_dir, "flatskill", flat=True)
    inventory = _list_for(project, tmp_path)

    (summary,) = inventory.skills
    assert summary.name == "flatskill"
    assert summary.path == (skills_dir / "flatskill.md").resolve()
    assert summary.base_dir == skills_dir


def test_symlinked_directory_to_accepted_skill_is_skipped_silently(project, tmp_path):
    skills_dir = _skills_dir(project)
    _write_skill(skills_dir, "alpha")
    agents_dir = _agents_dir(project)
    agents_dir.mkdir(parents=True, exist_ok=True)
    link = agents_dir / "alpha"
    link.symlink_to(skills_dir / "alpha", target_is_directory=True)

    inventory = _list_for(project, tmp_path)

    assert [s.name for s in inventory.skills] == ["alpha"]
    assert inventory.diagnostics == ()
    assert inventory.skills[0].identity.source == "project-unchain"


def test_root_level_skill_md_and_dotfiles_are_ignored(project, tmp_path):
    skills_dir = _skills_dir(project)
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / SKILL_FILE_NAME).write_text(
        "---\nname: root\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    (skills_dir / ".hidden.md").write_text(
        "---\nname: hidden\ndescription: d\n---\nbody\n", encoding="utf-8"
    )
    (skills_dir / "notes.txt").write_text("not a skill\n", encoding="utf-8")
    (skills_dir / "empty_dir").mkdir()
    hidden_dir = skills_dir / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / SKILL_FILE_NAME).write_text(
        "---\nname: hiddendir\ndescription: d\n---\nbody\n", encoding="utf-8"
    )

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    assert inventory.diagnostics == ()


def test_unreadable_file_is_inaccessible(project, tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("cannot exercise permission denial while running as root")
    path = _write_skill(_skills_dir(project), "locked")
    os.chmod(path, 0o000)
    try:
        inventory = _list_for(project, tmp_path)
    finally:
        os.chmod(path, 0o644)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "inaccessible"
    assert diag.name == "locked"


def test_binary_file_is_inaccessible(project, tmp_path):
    skill_dir = _skills_dir(project) / "binary"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILE_NAME).write_bytes(b"\xff\xfe\x00\x01invalid utf8 \xff")

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "inaccessible"
    assert diag.name == "binary"


# ---------------------------------------------------------------------------
# Invalid cases: exactly one `invalid` diagnostic, no listing
# ---------------------------------------------------------------------------


def test_invalid_missing_frontmatter(project, tmp_path):
    skill_dir = _skills_dir(project) / "nofm"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILE_NAME).write_text("no frontmatter here\n", encoding="utf-8")

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "invalid"


def test_invalid_name_dir_mismatch(project, tmp_path):
    skill_dir = _skills_dir(project) / "realname"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILE_NAME).write_text(
        "---\nname: othername\ndescription: d\n---\nbody\n", encoding="utf-8"
    )

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "invalid"
    assert "does not match" in diag.message


def test_invalid_uppercase_name(project, tmp_path):
    skill_dir = _skills_dir(project) / "Foo"
    skill_dir.mkdir(parents=True)
    (skill_dir / SKILL_FILE_NAME).write_text(
        "---\nname: Foo\ndescription: d\n---\nbody\n", encoding="utf-8"
    )

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "invalid"


def test_invalid_empty_description(project, tmp_path):
    _write_skill(_skills_dir(project), "emptydesc", description="")

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "invalid"


def test_invalid_disable_model_invocation_maybe(project, tmp_path):
    _write_skill(_skills_dir(project), "maybeflag", extra="disable-model-invocation: maybe\n")

    inventory = _list_for(project, tmp_path)

    assert inventory.skills == ()
    (diag,) = inventory.diagnostics
    assert diag.kind == "invalid"


# ---------------------------------------------------------------------------
# Policy flags, metadata, get() behavior
# ---------------------------------------------------------------------------


def test_policy_flags_parsed_and_unknown_keys_land_in_metadata(project, tmp_path):
    _write_skill(
        _skills_dir(project),
        "flagged",
        extra="disable-model-invocation: true\nuser-invocable: false\ncustom-key: custom-value\n",
    )
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)
    inventory = registry.list()

    (summary,) = inventory.skills
    assert summary.model_invocable is False
    assert summary.user_invocable is False

    loaded = registry.get("flagged")
    assert loaded is not None
    assert loaded.metadata == {"custom-key": "custom-value"}


def test_get_rereads_modified_file(project, tmp_path):
    path = _write_skill(_skills_dir(project), "mutable", body="v1")
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)

    first = registry.get("mutable")
    assert first is not None
    assert first.body == "v1"

    path.write_text(
        "---\nname: mutable\ndescription: mutable description\n---\nv2\n", encoding="utf-8"
    )
    second = registry.get("mutable")
    assert second is not None
    assert second.body == "v2"
    assert second.revision != first.revision


def test_get_of_shadowed_name_returns_winner_body(project, tmp_path):
    _write_skill(_skills_dir(project), "dup", body="winner body")
    _write_skill(_agents_dir(project), "dup", body="loser body")
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)

    loaded = registry.get("dup")
    assert loaded is not None
    assert loaded.body == "winner body"
    assert loaded.summary.identity.source == "project-unchain"


def test_get_missing_and_alias_return_none(project, tmp_path):
    toolkit = Toolkit(
        skills=(SkillDescriptor("real", "d", "b", aliases=("shortcut",), source_id="tk-1"),)
    )
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    assert registry.get("missing") is None
    assert registry.get("shortcut") is None  # alias is not a canonical name


# ---------------------------------------------------------------------------
# Toolkit source
# ---------------------------------------------------------------------------


def test_toolkit_skill_rank_and_tools_rendered_in_get(project, tmp_path):
    descriptor = SkillDescriptor(
        "review",
        "Review code.",
        "Use ({tools}) carefully.",
        ("grep", "read_file"),
        source_id="tk-review",
    )
    toolkit = Toolkit(skills=(descriptor,))
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    inventory = registry.list()
    (summary,) = inventory.skills
    assert summary.rank == TOOLKIT_SKILL_RANK
    assert summary.path is None
    assert summary.base_dir is None
    assert summary.identity.source == "toolkit"
    assert summary.identity.source_id == "tk-review"

    loaded = registry.get("review")
    assert loaded is not None
    assert loaded.body == "Use (`grep`, `read_file`) carefully."
    assert loaded.tools == ("grep", "read_file")


def test_toolkit_skill_base_dir_preserved(project, tmp_path):
    base = tmp_path / "resources"
    base.mkdir()
    descriptor = SkillDescriptor("withbase", "Has base dir.", "body", base_dir=base, source_id="tk-2")
    toolkit = Toolkit(skills=(descriptor,))
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    (summary,) = registry.list().skills
    assert summary.base_dir == base


def test_filesystem_skill_shadows_toolkit_skill_of_same_name(project, tmp_path):
    _write_skill(_skills_dir(project), "shared", description="from filesystem")
    descriptor = SkillDescriptor("shared", "from toolkit", "body", source_id="tk-3")
    toolkit = Toolkit(skills=(descriptor,))
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    inventory = registry.list()
    (summary,) = inventory.skills
    assert summary.identity.source == "project-unchain"
    (shadowed,) = [d for d in inventory.diagnostics if d.kind == "shadowed"]
    assert shadowed.source == "toolkit"


def test_toolkit_skills_appended_after_construction_are_discovered(project, tmp_path):
    toolkit = Toolkit()
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    assert registry.list().skills == ()

    toolkit.skills = toolkit.skills + (
        SkillDescriptor("late", "Added later.", "body", source_id="tk-4"),
    )
    inventory = registry.list()
    assert [s.name for s in inventory.skills] == ["late"]


# ---------------------------------------------------------------------------
# Aliases and reserved commands
# ---------------------------------------------------------------------------


def test_alias_resolves_case_insensitively(project, tmp_path):
    descriptor = SkillDescriptor(
        "quiet", "Quiet mode.", "body", aliases=("quiet-legacy",), source_id="tk-5"
    )
    toolkit = Toolkit(skills=(descriptor,))
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    summary = registry.resolve("/QUIET-LEGACY")
    assert summary is not None
    assert summary.name == "quiet"


def test_alias_equal_to_canonical_name_is_dropped(project, tmp_path):
    descriptor_a = SkillDescriptor("alpha", "Alpha.", "body", source_id="tk-a")
    descriptor_b = SkillDescriptor("beta", "Beta.", "body", aliases=("alpha",), source_id="tk-b")
    toolkit = Toolkit(skills=(descriptor_a, descriptor_b))
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    inventory = registry.list()
    (diag,) = [d for d in inventory.diagnostics if d.kind == "alias_shadowed"]
    assert diag.name == "alpha"

    resolved = registry.resolve("alpha")
    assert resolved is not None
    assert resolved.name == "alpha"


def test_alias_claimed_by_two_toolkit_skills_is_ambiguous(project, tmp_path):
    descriptor_a = SkillDescriptor("one", "One.", "body", aliases=("dup-alias",), source_id="tk-1")
    descriptor_b = SkillDescriptor("two", "Two.", "body", aliases=("dup-alias",), source_id="tk-2")
    toolkit = Toolkit(skills=(descriptor_a, descriptor_b))
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config, runtime_toolkit=toolkit)

    inventory = registry.list()
    (diag,) = [d for d in inventory.diagnostics if d.kind == "alias_ambiguous"]
    assert diag.name == "dup-alias"
    assert registry.resolve("dup-alias") is None


def test_reserved_commands_normalized_and_flagged_but_still_listed(project, tmp_path):
    _write_skill(_skills_dir(project), "btw")
    config = SkillsConfig(
        project_root=project,
        include_user_dirs=False,
        home=tmp_path / "home",
        reserved_commands=("/btw",),
    )
    registry = SkillRegistry(config)
    assert registry.config.reserved_commands == ("btw",)

    inventory = registry.list()
    assert [s.name for s in inventory.skills] == ["btw"]
    (diag,) = [d for d in inventory.diagnostics if d.kind == "reserved"]
    assert diag.name == "btw"


# ---------------------------------------------------------------------------
# Inventory revision
# ---------------------------------------------------------------------------


def test_inventory_revision_stable_across_calls_and_changes_on_body_edit(project, tmp_path):
    path = _write_skill(_skills_dir(project), "stable", body="v1")
    config = SkillsConfig(project_root=project, include_user_dirs=False, home=tmp_path / "home")
    registry = SkillRegistry(config)

    first = registry.list().revision
    second = registry.list().revision
    assert first == second

    path.write_text(
        "---\nname: stable\ndescription: stable description\n---\nv2\n", encoding="utf-8"
    )
    third = registry.list().revision
    assert third != first


# ---------------------------------------------------------------------------
# SkillsConfig validation / coerce
# ---------------------------------------------------------------------------


def test_skills_config_validation_errors():
    with pytest.raises(ValueError):
        SkillsConfig(catalog_description_max_length=2)
    with pytest.raises(ValueError):
        SkillsConfig(tool_name="")


def test_skills_config_reserved_commands_and_extra_dirs_normalized():
    config = SkillsConfig(reserved_commands=("/Foo", "bar", "//baz", ""), extra_dirs=("a", "b"))
    assert config.reserved_commands == ("foo", "bar", "baz")
    assert config.extra_dirs == (Path("a"), Path("b"))


def test_skills_config_coerce_cases():
    assert SkillsConfig.coerce(None) == SkillsConfig()

    made = SkillsConfig.coerce({"tool_name": "custom"})
    assert made.tool_name == "custom"

    existing = SkillsConfig(tool_name="already")
    assert SkillsConfig.coerce(existing) is existing

    with pytest.raises(TypeError):
        SkillsConfig.coerce(42)


def test_extra_skills_join_the_registry_at_toolkit_rank(tmp_path):
    from unchain.tools import SkillDescriptor

    home = tmp_path / "home"
    home.mkdir()
    extra = SkillDescriptor("pack-skill", "From a host pack.", "body", source="skillpack", source_id="skillpack.x")
    registry = SkillRegistry(
        SkillsConfig(project_root=tmp_path, include_project_dirs=False, include_user_dirs=False, home=home, extra_skills=(extra,))
    )
    (summary,) = registry.list().skills
    assert summary.rank == 600
    assert summary.identity.source == "skillpack" and summary.identity.source_id == "skillpack.x"
    assert registry.get("pack-skill").body == "body"
    with pytest.raises(TypeError):
        SkillsConfig(extra_skills=("nope",))
