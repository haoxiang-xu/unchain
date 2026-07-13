import json
from pathlib import Path

import pytest

from unchain.tools import (
    ToolRegistryConfig,
    ToolkitRegistry,
    get_toolkit_metadata,
    list_artifact_kinds,
    list_toolkits,
)


def _write_icon(path: Path) -> None:
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 8 8"><rect width="8" height="8" rx="2" fill="#111827"/></svg>\n',
        encoding="utf-8",
    )


def _write_toolkit_package(
    root: Path,
    *,
    package_name: str = "demo_toolkit",
    toolkit_id: str = "demo",
    manifest_tool_name: str = "echo",
    include_manifest: bool = True,
    include_toolkit_readme: bool = True,
    include_toolkit_icon: bool = True,
    include_tool_readme_field: bool = False,
    toolkit_icon_value: str = "icon.svg",
    toolkit_color: str | None = None,
    toolkit_backgroundcolor: str | None = None,
    artifact_kinds: str = "",
) -> Path:
    package_dir = root / package_name
    package_dir.mkdir(parents=True, exist_ok=True)

    (package_dir / "__init__.py").write_text(
        f"from .runtime import DemoToolkit\n\n__all__ = ['DemoToolkit']\n",
        encoding="utf-8",
    )
    (package_dir / "runtime.py").write_text(
        """
from unchain.tools import Toolkit


class DemoToolkit(Toolkit):
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

    if include_toolkit_readme:
        (package_dir / "README.md").write_text("# Demo Toolkit\n\nLocal test toolkit.\n", encoding="utf-8")
    if include_toolkit_icon:
        _write_icon(package_dir / "icon.svg")

    if include_manifest:
        tool_readme_line = 'readme = "tools/echo/README.md"\n' if include_tool_readme_field else ""
        color_line = f'color = "{toolkit_color}"\n' if toolkit_color is not None else ""
        background_line = (
            f'backgroundcolor = "{toolkit_backgroundcolor}"\n'
            if toolkit_backgroundcolor is not None
            else ""
        )
        (package_dir / "toolkit.toml").write_text(
            f"""
[toolkit]
id = "{toolkit_id}"
name = "Demo Toolkit"
description = "Local test toolkit."
factory = "{package_name}:DemoToolkit"
version = "1.0.0"
readme = "README.md"
icon = "{toolkit_icon_value}"
{color_line}{background_line}tags = ["local", "test"]

[display]
category = "local"
order = 5
hidden = false

[compat]
python = ">=3.9"
legacy = ">=0"

[[tools]]
name = "{manifest_tool_name}"
title = "Echo"
description = "Echo text back."
{tool_readme_line}observe = false
requires_confirmation = false

{artifact_kinds}
""".strip()
            + "\n",
            encoding="utf-8",
        )

    return package_dir


class _FakeEntryPoint:
    def __init__(self, name: str, loader):
        self.name = name
        self._loader = loader

    def load(self):
        return self._loader()


def test_builtin_registry_lists_expected_toolkits_and_tools():
    registry = ToolkitRegistry()
    toolkit_ids = {item["id"] for item in registry.list_toolkits(include_tools=False)}

    assert toolkit_ids == {
        "agent_reach",
        "core",
        "plan",
    }
    agent_reach_summary = registry.require("agent_reach").to_summary()
    assert agent_reach_summary["tool_count"] == 3
    assert agent_reach_summary["icon"] == {"type": "emoji", "emoji": "👁️"}
    assert registry.require("core").to_summary()["tool_count"] == 9
    assert registry.require("plan").to_summary()["tool_count"] == 5
    plan_skills = registry.require("plan").to_summary()["skills"]
    assert [skill["name"] for skill in plan_skills] == ["plan"]
    assert plan_skills[0]["phase"] == "composer"
    assert plan_skills[0]["tools"] == [
        "plan_start",
        "plan_update",
        "plan_read",
        "plan_finalize",
        "plan_list",
    ]


def test_only_public_builtin_toolkits_ship_registry_manifests():
    builtin_root = Path(__file__).resolve().parents[1] / "src" / "unchain" / "toolkits" / "builtin"
    manifest_dirs = {
        manifest_path.parent.name
        for manifest_path in builtin_root.rglob("toolkit.toml")
    }

    assert manifest_dirs == {"agent_reach", "core", "plan"}


def test_registry_instantiated_agent_reach_tools_include_emoji_icon_metadata():
    registry = ToolkitRegistry()

    runtime_toolkit = registry.instantiate_toolkit("agent_reach")
    tool_json = runtime_toolkit.to_json()

    assert {item["name"] for item in tool_json} == {
        "agent_reach_status",
        "agent_reach_read_web",
        "agent_reach_youtube_metadata",
    }
    assert all(item["icon_path"] == "" for item in tool_json)
    assert all(item["icon"] == {"type": "emoji", "emoji": "👁️"} for item in tool_json)


def test_core_toolkit_description_encourages_user_clarification():
    registry = ToolkitRegistry()

    summary = registry.require("core").to_summary()

    assert "structured user questions" in summary["description"]
    assert "LSP-powered code intelligence" in summary["description"]


def test_get_toolkit_metadata_returns_full_markdown_and_inherited_tool_icon():
    toolkit_metadata = get_toolkit_metadata("core")
    tool_metadata = get_toolkit_metadata("core", "read")

    assert toolkit_metadata["readme_markdown"].startswith("# Core Toolkit")
    assert tool_metadata["toolkit"]["readme_markdown"].startswith("# Core Toolkit")
    assert tool_metadata["tool"]["icon_path"] == tool_metadata["toolkit"]["icon_path"]
    assert tool_metadata["tool"]["icon"] == tool_metadata["toolkit"]["icon"]


def test_list_toolkits_payload_is_json_serializable():
    payload = list_toolkits()
    encoded = json.dumps(payload)

    assert "core" in encoded
    assert '"id": "ask_user"' not in encoded


def test_local_root_requires_a_manifest(tmp_path):
    empty_root = tmp_path / "no_manifest_here"
    empty_root.mkdir()

    with pytest.raises(ValueError, match="no toolkit.toml found under local root"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[empty_root]))


def test_local_root_validates_missing_toolkit_readme(tmp_path, monkeypatch):
    _write_toolkit_package(tmp_path, include_toolkit_readme=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="toolkit readme not found"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_local_root_validates_missing_toolkit_icon(tmp_path, monkeypatch):
    _write_toolkit_package(tmp_path, include_toolkit_icon=False)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="toolkit icon not found"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_local_root_supports_builtin_toolkit_icon_with_colors(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        toolkit_icon_value="terminal",
        toolkit_color="#0f172a",
        toolkit_backgroundcolor="#bae6fd",
        include_toolkit_icon=False,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))
    summary = registry.require("demo").to_summary()
    tool_summary = summary["tools"][0]

    assert summary["icon_path"] == ""
    assert summary["icon"] == {
        "type": "builtin",
        "name": "terminal",
        "color": "#0f172a",
        "background_color": "#bae6fd",
    }
    assert tool_summary["icon_path"] == ""
    assert tool_summary["icon"] == summary["icon"]


def test_local_root_supports_emoji_toolkit_icon_without_colors(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        toolkit_icon_value="👁️",
        include_toolkit_icon=False,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))
    summary = registry.require("demo").to_summary()
    tool_summary = summary["tools"][0]

    assert summary["icon_path"] == ""
    assert summary["icon"] == {"type": "emoji", "emoji": "👁️"}
    assert tool_summary["icon_path"] == ""
    assert tool_summary["icon"] == summary["icon"]


def test_local_root_builtin_toolkit_icon_requires_colors(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        toolkit_icon_value="terminal",
        include_toolkit_icon=False,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="builtin toolkit icon 'terminal' requires non-empty 'color'"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_local_root_validates_tool_name_mismatch(tmp_path, monkeypatch):
    _write_toolkit_package(tmp_path, manifest_tool_name="wrong_name")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="runtime tools missing from manifest"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_local_root_rejects_duplicate_toolkit_ids(tmp_path, monkeypatch):
    _write_toolkit_package(tmp_path, package_name="toolkit_one", toolkit_id="duplicate")
    _write_toolkit_package(tmp_path, package_name="toolkit_two", toolkit_id="duplicate")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="duplicate toolkit id 'duplicate'"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_local_root_rejects_tool_level_readme_field(tmp_path, monkeypatch):
    _write_toolkit_package(tmp_path, include_tool_readme_field=True)
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="tool-level readme is not supported"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_disabled_plugins_are_not_loaded(monkeypatch):
    called = {"loaded": False}

    def _load():
        called["loaded"] = True
        raise AssertionError("plugin loader should not run when plugin is disabled")

    monkeypatch.setattr(
        "unchain.tools.registry._entry_points_for_group",
        lambda group: [_FakeEntryPoint("demo", _load)],
    )

    registry = ToolkitRegistry(ToolRegistryConfig(include_builtin=False, enabled_plugins=[]))

    assert registry.list_toolkits(include_tools=False) == []
    assert called["loaded"] is False


def test_enabled_plugin_is_loaded_only_when_explicitly_enabled(tmp_path, monkeypatch):
    _write_toolkit_package(tmp_path, package_name="plugin_demo", toolkit_id="plugin_demo")
    monkeypatch.syspath_prepend(str(tmp_path))

    def _load():
        module = __import__("plugin_demo", fromlist=["DemoToolkit"])
        return module.DemoToolkit

    monkeypatch.setattr(
        "unchain.tools.registry._entry_points_for_group",
        lambda group: [_FakeEntryPoint("plugin_demo", _load)],
    )

    registry = ToolkitRegistry(
        ToolRegistryConfig(
            include_builtin=False,
            enabled_plugins=["plugin_demo"],
        )
    )
    summary = registry.require("plugin_demo").to_summary()

    assert summary["source"] == "plugin"
    assert summary["tool_count"] == 1
    assert summary["tools"][0]["name"] == "echo"


def test_local_root_parses_artifact_kind_metadata_with_builtin_icon(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        artifact_kinds="""
[[artifact_kinds]]
kind = "benchmark_report"
display_name = "Benchmark"
description = "Benchmark summary artifacts."
icon = "bar_chart"
fallback_renderer = "markdown"
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))
    summary = registry.require("demo").to_summary(include_tools=False)

    assert summary["artifact_kinds"] == [
        {
            "kind": "benchmark_report",
            "display_name": "Benchmark",
            "description": "Benchmark summary artifacts.",
            "icon": {
                "type": "builtin",
                "name": "bar_chart",
                "color": "",
                "background_color": "",
            },
            "fallback_renderer": "markdown",
            "toolkit_id": "demo",
        }
    ]


def test_local_root_parses_artifact_kind_metadata_with_file_icon(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        artifact_kinds="""
[[artifact_kinds]]
kind = "benchmark_report"
display_name = "Benchmark"
description = "Benchmark summary artifacts."
icon = "icon.svg"
fallback_renderer = "markdown"
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    registry = ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))
    [artifact_kind] = registry.require("demo").to_summary(include_tools=False)["artifact_kinds"]

    assert artifact_kind["icon"]["type"] == "file"
    assert artifact_kind["icon"]["path"].endswith("demo_toolkit/icon.svg")


def test_local_root_rejects_invalid_artifact_kind_renderer(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        artifact_kinds="""
[[artifact_kinds]]
kind = "benchmark_report"
display_name = "Benchmark"
description = "Benchmark summary artifacts."
icon = "bar_chart"
fallback_renderer = "chart"
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="invalid artifact fallback_renderer"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_local_root_rejects_artifact_icon_path_outside_toolkit_root(tmp_path, monkeypatch):
    (tmp_path / "outside.svg").write_text("<svg></svg>", encoding="utf-8")
    _write_toolkit_package(
        tmp_path,
        artifact_kinds="""
[[artifact_kinds]]
kind = "benchmark_report"
display_name = "Benchmark"
description = "Benchmark summary artifacts."
icon = "../outside.svg"
fallback_renderer = "markdown"
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ValueError, match="artifact icon for 'benchmark_report' must stay within toolkit directory"):
        ToolkitRegistry(ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path]))


def test_list_artifact_kinds_exposes_builtin_and_toolkit_definitions(tmp_path, monkeypatch):
    _write_toolkit_package(
        tmp_path,
        artifact_kinds="""
[[artifact_kinds]]
kind = "benchmark_report"
display_name = "Benchmark"
description = "Benchmark summary artifacts."
icon = "bar_chart"
fallback_renderer = "markdown"
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    kinds = {
        item["kind"]: item
        for item in list_artifact_kinds(
            ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path])
        )
    }

    assert kinds["file_diff"]["display_name"] == "Files changed"
    assert kinds["workspace_change_set"]["display_name"] == "Workspace changes"
    assert kinds["plan"]["display_name"] == "Plan"
    assert kinds["benchmark_report"]["display_name"] == "Benchmark"
    assert kinds["benchmark_report"]["toolkit_id"] == "demo"


def test_custom_artifact_kind_duplicates_are_deterministic_and_logged(
    tmp_path,
    monkeypatch,
    caplog,
):
    artifact_kinds = """
[[artifact_kinds]]
kind = "shared.report"
display_name = "{display_name}"
description = "Shared report."
icon = "bar_chart"
fallback_renderer = "markdown"
"""
    _write_toolkit_package(
        tmp_path,
        package_name="z_toolkit",
        toolkit_id="z_toolkit",
        artifact_kinds=artifact_kinds.format(display_name="Z Report"),
    )
    _write_toolkit_package(
        tmp_path,
        package_name="a_toolkit",
        toolkit_id="a_toolkit",
        artifact_kinds=artifact_kinds.format(display_name="A Report"),
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with caplog.at_level("WARNING"):
        kinds = {
            item["kind"]: item
            for item in list_artifact_kinds(
                ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path])
            )
        }

    assert kinds["shared.report"]["toolkit_id"] == "a_toolkit"
    assert kinds["shared.report"]["display_name"] == "A Report"
    assert "duplicate artifact kind 'shared.report'" in caplog.text


def test_toolkit_artifact_kind_cannot_override_builtin_kind(tmp_path, monkeypatch, caplog):
    _write_toolkit_package(
        tmp_path,
        artifact_kinds="""
[[artifact_kinds]]
kind = "file_diff"
display_name = "Custom Files"
description = "Should not replace builtin metadata."
icon = "bar_chart"
fallback_renderer = "markdown"
""",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with caplog.at_level("WARNING"):
        kinds = {
            item["kind"]: item
            for item in list_artifact_kinds(
                ToolRegistryConfig(include_builtin=False, local_roots=[tmp_path])
            )
        }

    assert kinds["file_diff"]["display_name"] == "Files changed"
    assert kinds["file_diff"]["toolkit_id"] == "builtin"
    assert "cannot override builtin artifact kind 'file_diff'" in caplog.text
