from pathlib import Path

import pytest

from unchain.tools import ToolRegistryConfig, ToolkitRegistry


def _write_icon(path: Path) -> None:
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><rect width="8" height="8" rx="2" fill="#111827"/></svg>\n',
        encoding="utf-8",
    )


def _write_skill_toolkit(root: Path, *, skills_toml: str) -> None:
    package_dir = root / "skilldemo_toolkit"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "from .runtime import SkillDemoToolkit\n\n__all__ = ['SkillDemoToolkit']\n",
        encoding="utf-8",
    )
    (package_dir / "runtime.py").write_text(
        """
from unchain.tools import Toolkit


class SkillDemoToolkit(Toolkit):
    def __init__(self):
        super().__init__()
        self.register(self.echo)

    def echo(self, text: str):
        \"\"\"Echo text back.\"\"\"
        return {\"echo\": text}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (package_dir / "README.md").write_text("# Skill Demo\n", encoding="utf-8")
    _write_icon(package_dir / "icon.svg")
    (package_dir / "toolkit.toml").write_text(
        f"""
[toolkit]
id = "skilldemo"
name = "Skill Demo"
description = "Toolkit with declared skills."
factory = "skilldemo_toolkit:SkillDemoToolkit"
version = "1.0.0"
readme = "README.md"
icon = "icon.svg"
tags = ["local", "test"]

[display]
category = "local"
order = 5
hidden = false

[compat]
python = ">=3.9"
legacy = ">=0"

[[tools]]
name = "echo"
title = "Echo"
description = "Echo text back."
observe = false
requires_confirmation = false

{skills_toml}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _registry(root: Path) -> ToolkitRegistry:
    return ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[root]))


def test_skills_parse_into_descriptor_and_summary(tmp_path, monkeypatch):
    _write_skill_toolkit(
        tmp_path,
        skills_toml="""
[[skills]]
name = "echo-loud"
title = "Echo Loud"
description = "Echo the text emphatically."
body = "Use the echo tool ({tools}) and repeat the result in caps."
tools = ["echo"]
phase = "composer"
""".strip(),
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    descriptor = _registry(tmp_path).require("skilldemo")
    summary = descriptor.to_summary()

    # PuPu-only `title` / `phase` keys are tolerated and ignored by the runtime.
    assert summary["skills"] == [
        {
            "name": "echo-loud",
            "description": "Echo the text emphatically.",
            "body": "Use the echo tool ({tools}) and repeat the result in caps.",
            "tools": ["echo"],
            "base_dir": str(descriptor.root_path),
            "model_invocable": True,
            "user_invocable": True,
            "metadata": {},
            "aliases": [],
            "source": "toolkit",
            "source_id": "skilldemo",
        }
    ]


def test_skills_policy_and_alias_fields(tmp_path, monkeypatch):
    _write_skill_toolkit(
        tmp_path,
        skills_toml="""
[[skills]]
name = "quiet"
description = "Only the user may invoke this."
body = "Do the quiet thing."
disable-model-invocation = true
user-invocable = true
aliases = ["quiet-legacy", "shh"]
""".strip(),
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    (skill,) = _registry(tmp_path).require("skilldemo").sorted_skills()
    assert skill.model_invocable is False
    assert skill.user_invocable is True
    assert skill.aliases == ("quiet-legacy", "shh")
    assert skill.tools == ()
    assert skill.source_id == "skilldemo"


def test_toolkit_without_skills_gets_empty_list(tmp_path, monkeypatch):
    _write_skill_toolkit(tmp_path, skills_toml="")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert _registry(tmp_path).require("skilldemo").to_summary()["skills"] == []


@pytest.mark.parametrize(
    ("skills_toml", "match"),
    [
        (
            '[[skills]]\nname = "bad name"\ndescription = "d"\nbody = "b"',
            "must be kebab-case",
        ),
        (
            '[[skills]]\nname = "Bad_Name"\ndescription = "d"\nbody = "b"',
            "must be kebab-case",
        ),
        (
            '[[skills]]\nname = "x"\ndescription = "d"\nbody = "b"\naliases = ["Not Kebab"]',
            "alias 'Not Kebab' must be kebab-case",
        ),
        (
            '[[skills]]\nname = "x"\ndescription = "d"\nbody = "b"\ntools = ["missing"]',
            "references unknown tool",
        ),
        (
            '[[skills]]\nname = "x"\ndescription = "d"\nbody = "b"\n\n'
            '[[skills]]\nname = "x"\ndescription = "d2"\nbody = "b2"',
            "duplicate skill",
        ),
        ('[[skills]]\nname = "x"\ndescription = "d"', "body"),
    ],
)
def test_invalid_skills_raise(tmp_path, monkeypatch, skills_toml, match):
    _write_skill_toolkit(tmp_path, skills_toml=skills_toml)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match=match):
        _registry(tmp_path)


def test_instantiated_toolkit_carries_skill_descriptors(tmp_path, monkeypatch):
    _write_skill_toolkit(
        tmp_path,
        skills_toml="""
[[skills]]
name = "quick"
description = "Minimal skill."
body = "Do the quick thing."
""".strip(),
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = _registry(tmp_path)
    runtime_toolkit = registry.instantiate_toolkit("skilldemo")

    assert [skill.name for skill in runtime_toolkit.skills] == ["quick"]
    assert runtime_toolkit.skills[0].base_dir == registry.require("skilldemo").root_path
    assert runtime_toolkit.skills[0].source_id == "skilldemo"


def test_toolkit_constructor_validates_skills():
    from unchain.tools import SkillDescriptor, Toolkit

    descriptor = SkillDescriptor("demo", "d", "body")
    assert Toolkit(skills=(descriptor,)).skills == (descriptor,)
    assert Toolkit().skills == ()
    with pytest.raises(TypeError, match="SkillDescriptor"):
        Toolkit(skills=("not a descriptor",))


def test_builder_add_tool_keeps_every_source_qualified_skill():
    from types import SimpleNamespace

    from unchain.agent.builder import AgentBuilder
    from unchain.tools import SkillDescriptor, Toolkit

    first = Toolkit(skills=(SkillDescriptor("dup", "one", "1", source_id="tk-a"),))
    second = Toolkit(
        skills=(
            SkillDescriptor("dup", "two", "2", source_id="tk-b"),
            SkillDescriptor("other", "o", "o", source_id="tk-b"),
        )
    )
    builder = AgentBuilder.__new__(AgentBuilder)
    builder.toolkit = Toolkit()
    builder.spec = SimpleNamespace(name="skills-test")  # only read on tool-name conflicts

    builder.add_tool(first)
    builder.add_tool(second)

    # No first-name-wins merge: the registry resolves duplicates by rank/identity.
    assert [(s.name, s.source_id) for s in builder.toolkit.skills] == [
        ("dup", "tk-a"),
        ("dup", "tk-b"),
        ("other", "tk-b"),
    ]
