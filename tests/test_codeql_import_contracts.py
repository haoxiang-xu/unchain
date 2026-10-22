"""Public imports must work independently of earlier imports in the process."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _run_in_fresh_python(source):
    source_root = Path(__file__).resolve().parents[1] / "src"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        str(path) for path in (source_root, env.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module_name", [
    "unchain",
    "unchain.tools",
    "unchain.memory",
    "unchain.subagents",
    "unchain.kernel",
    "unchain.providers.model_io",
    "unchain.agent.modules",
])
def test_all_public_exports_resolve_in_a_fresh_process(module_name):
    _run_in_fresh_python(f"""
        import importlib

        module = importlib.import_module({module_name!r})
        exports = {{name: getattr(module, name) for name in module.__all__}}
        imported = {{}}
        exec("from {module_name} import *", imported)
        for name, value in exports.items():
            assert imported[name] is value, (module.__name__, name)
        try:
            getattr(module, "_nonexistent_public_export")
        except AttributeError:
            pass
        else:
            raise AssertionError("Unknown exports must raise AttributeError")
    """)


@pytest.mark.parametrize("facade,name,definition", [
    ("unchain", "Agent", "unchain.agent.agent"),
    ("unchain.tools", "tool", "unchain.tools.decorators"),
    ("unchain.tools", "Tool", "unchain.tools.tool"),
    ("unchain.memory", "MemoryManager", "unchain.memory.manager"),
    ("unchain.subagents", "SubagentExecutor", "unchain.subagents.executor"),
    ("unchain.kernel", "KernelLoop", "unchain.kernel.loop"),
    ("unchain.kernel", "ModelIO", "unchain.providers.base"),
    ("unchain.providers.model_io", "OpenAIModelIO", "unchain.providers.openai"),
    ("unchain.providers.model_io", "AnthropicModelIO", "unchain.providers.anthropic"),
    ("unchain.providers.model_io", "OllamaModelIO", "unchain.providers.ollama"),
    ("unchain.agent.modules", "CharacterModule", "unchain.character.module"),
])
def test_public_aliases_keep_the_original_object_identity(facade, name, definition):
    _run_in_fresh_python(f"""
        import importlib

        facade = importlib.import_module({facade!r})
        public_object = getattr(facade, {name!r})
        original = getattr(importlib.import_module({definition!r}), {name!r})
        assert public_object is original
        assert getattr(facade, {name!r}) is original
    """)


@pytest.mark.parametrize("facade,name,implementation", [
    ("unchain", "Agent", "unchain.agent.agent"),
    ("unchain.tools", "ToolExecutionHarness", "unchain.tools.execution"),
    ("unchain.memory", "MemoryManager", "unchain.memory.manager"),
    ("unchain.subagents", "SubagentExecutor", "unchain.subagents.executor"),
    ("unchain.kernel", "KernelLoop", "unchain.kernel.loop"),
    ("unchain.agent.modules", "CharacterModule", "unchain.character.module"),
])
def test_lazy_facades_load_implementations_only_when_requested(facade, name, implementation):
    _run_in_fresh_python(f"""
        import importlib
        import sys

        facade = importlib.import_module({facade!r})
        assert {implementation!r} not in sys.modules
        exported_object = getattr(facade, {name!r})
        assert exported_object is getattr(sys.modules[{implementation!r}], {name!r})
    """)


# Each pair corresponds to a distinct dependency cycle reported by CodeQL.
# Test both orders without pytest collection having already warmed sys.modules.
_IMPORT_PAIRS = [
    ("unchain.context.provider_execution", "unchain.providers.turn_ownership"),
    ("unchain.context.provider_turns", "unchain.context.provider_execution"),
    ("unchain.kernel.run_ledger", "unchain.kernel.state"),
    ("unchain.kernel.run_ledger", "unchain.providers.turn_ownership"),
    ("unchain.kernel.state", "unchain.providers.context_assembler"),
    ("unchain.providers.context_assembler", "unchain.providers.openai"),
    ("unchain.providers.context_assembler", "unchain.providers.wire_preparer"),
    ("unchain.providers.exact_route_transport", "unchain.context.provider_execution"),
    ("unchain.providers.openai", "unchain.context.provider_execution"),
    ("unchain.providers.openai", "unchain.providers.exact_route_transport"),
    ("unchain.providers.prepared_request_factory", "unchain.context.provider_turns"),
    ("unchain.providers.wire_preparer", "unchain.context.provider_turns"),
    ("unchain.providers.wire_preparer", "unchain.providers.prepared_request_factory"),
]


@pytest.mark.parametrize("first,second", _IMPORT_PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
def test_provider_and_context_imports_work_in_either_order(first, second, reverse):
    order = [second, first] if reverse else [first, second]
    _run_in_fresh_python(f"""
        import importlib

        for module_name in {order!r}:
            module = importlib.import_module(module_name)
            assert module.__spec__ is not None
    """)
