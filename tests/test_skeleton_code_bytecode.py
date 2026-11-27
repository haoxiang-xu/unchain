"""Default Code execution must not create bytecode beside imported modules."""
from copy import deepcopy
from pathlib import Path
import os
import sys

import pytest

from unchain.skeleton import PythonEnvironmentProfile
from tests.test_skeleton_code_runtime import _runner, _write


@pytest.mark.parametrize("kind", ["bundled", "venv"])
def test_code_worker_preserves_imported_module_tree(tmp_path, kind):
    source = tmp_path / "source"
    modules = tmp_path / "owned-module-library"
    modules.mkdir()
    module = modules / "owned_bytecode_probe.py"
    module.write_text('VALUE = "exact imported value"\n')
    _write(source, "import sys\n"
        "def main(input, context):\n"
        " sys.path.insert(0, input['text'])\n"
        " import owned_bytecode_probe\n"
        " return {'answer': owned_bytecode_probe.VALUE}\n")
    executable = os.environ.get("PUPU_BUNDLED_PYTHON", sys.executable) if kind == "bundled" else sys.executable
    profile = (PythonEnvironmentProfile.bundled("builtin", (executable,)) if kind == "bundled"
        else PythonEnvironmentProfile.venv("builtin", executable))
    runner, _ = _runner(tmp_path, source, profiles={"builtin": profile})
    before = {str(path.relative_to(modules)): path.read_bytes() for path in modules.rglob("*") if path.is_file()}
    runner.create({"text": str(modules)})
    result = runner.run()
    assert result.status == "completed" and result.outputs == {"answer": "exact imported value"}
    after = {str(path.relative_to(modules)): path.read_bytes() for path in modules.rglob("*") if path.is_file()}
    assert after == before
    assert not list(modules.rglob("*.pyc"))


def test_wrapped_host_commands_keep_their_existing_launch_contract(tmp_path):
    # A host-supplied command can already contain its own worker protocol.
    # Interpreter flags only belong on the direct-Python branch.
    from unchain.skeleton.code_runtime import SkeletonCodeRuntime
    profile = PythonEnvironmentProfile.bundled("wrapped", ("/trusted/host", "--code-worker"))
    assert SkeletonCodeRuntime._command(profile, tmp_path / "request.json", tmp_path / "result.json") == [
        "/trusted/host", "--code-worker", str(tmp_path / "request.json"), str(tmp_path / "result.json")]
