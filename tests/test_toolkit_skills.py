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

    summary = _registry(tmp_path).require("skilldemo").to_summary()

    assert summary["skills"] == [
        {
            "name": "echo-loud",
            "title": "Echo Loud",
            "description": "Echo the text emphatically.",
            "body": "Use the echo tool ({tools}) and repeat the result in caps.",
            "tools": ["echo"],
            "phase": "composer",
        }
    ]


def test_skills_defaults_title_and_phase(tmp_path, monkeypatch):
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

    (skill,) = _registry(tmp_path).require("skilldemo").to_summary()["skills"]
    assert skill["title"] == "quick"
    assert skill["phase"] == "composer"
    assert skill["tools"] == []


def test_toolkit_without_skills_gets_empty_list(tmp_path, monkeypatch):
    _write_skill_toolkit(tmp_path, skills_toml="")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert _registry(tmp_path).require("skilldemo").to_summary()["skills"] == []


@pytest.mark.parametrize(
    ("skills_toml", "match"),
    [
        (
            '[[skills]]\nname = "bad name"\ndescription = "d"\nbody = "b"',
            "must match",
        ),
        (
            '[[skills]]\nname = "x"\ndescription = "d"\nbody = "b"\ntools = ["missing"]',
            "references unknown tool",
        ),
        (
            '[[skills]]\nname = "x"\ndescription = "d"\nbody = "b"\nphase = "later"',
            "invalid phase",
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
