"""The public Skeleton API remains lazy and importable in a fresh process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_fresh_process_can_import_every_public_export_without_eager_bootstrap(tmp_path):
    source_root = Path(__file__).resolve().parents[1] / "src"
    script = """
from pathlib import Path
import socket
import sys

def forbidden_network(*args, **kwargs):
    raise AssertionError('importing the public API must not contact a provider')

socket.socket.connect = forbidden_network
socket.socket.connect_ex = forbidden_network
socket.socket.sendto = forbidden_network
socket.create_connection = forbidden_network

import unchain
import unchain.skeleton as skeleton

source_root = Path(sys.argv[1]).resolve()
for module in (unchain, skeleton):
    assert Path(module.__file__).resolve().is_relative_to(source_root), module.__file__
assert not any(name.startswith(('unchain.providers', 'unchain.persistence')) for name in sys.modules)
assert 'unchain.skeleton.kernel_runtime' not in sys.modules
assert 'unchain.skeleton.durable_execution' not in sys.modules

from unchain.runtime import runtime_protocol
assert Path(runtime_protocol.__file__).resolve().is_relative_to(source_root)
assert runtime_protocol.runtime_protocol_manifest()['manifest_digest'].startswith('sha256:')

public = {}
exec('from unchain.skeleton import *', public)
assert len(skeleton.__all__) == len(set(skeleton.__all__))
assert set(public) - {'__builtins__'} == set(skeleton.__all__)
for name in skeleton.__all__:
    assert callable(public[name]), name
    assert public[name] is getattr(skeleton, name), name

try:
    skeleton.not_a_public_skeleton_export
except AttributeError:
    pass
else:
    raise AssertionError('unknown public names must raise AttributeError')
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(source_root)],
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(source_root),
            "PYTHONIOENCODING": "utf-8",
        },
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
