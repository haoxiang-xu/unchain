"""CP13 real-process Code node contract and durable integration."""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import venv

import pytest

from unchain.journal import AttemptRef, GenerationRef, JournalConflictError
from unchain.persistence import SQLiteContextV2Store
from unchain.skeleton import (
    FileCodeSourceResolver,
    PythonEnvironmentProfile,
    SkeletonCodeRuntime,
    SkeletonSkillRuntime,
    execution_definition_digest,
    prepare_workflow,
)
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import DurableStoreError, WorkflowExecutionStore
from unchain.skeleton.errors import SkeletonValidationError


RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _document(*, profile="builtin", timeout_ms=5_000, max_result_bytes=1_048_576,
              max_log_bytes=65_536, source="main.py", result_schema=RESULT_SCHEMA, artifacts=()):
    return {
        "schema_version": 1, "kind": "workflow", "id": "code-workflow", "name": "Code",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": [
                {"name": "text", "source": {"kind": "result", "path": ["text"]}},
            ]}}},
            {"id": "code", "type": "code", "config": {
                "execution": {
                    "inputs": [{"name": "text", "value": {"kind": "ref", "node_id": "start", "path": ["text"]}, "missing": "error"}],
                    "outputs": [{"name": "answer", "source": {"kind": "result", "path": ["answer"]}}],
                },
                "code": {
                    "source": {"kind": "file", "path": source},
                    "environment_profile_id": profile,
                    "timeout_ms": timeout_ms,
                    "max_result_bytes": max_result_bytes,
                    "max_log_bytes": max_log_bytes,
                    "result_schema": deepcopy(result_schema),
                    "artifacts": deepcopy(list(artifacts)),
                },
            }},
            {"id": "end", "type": "end", "config": {"execution": {
                "inputs": [{"name": "answer", "value": {"kind": "ref", "node_id": "code", "path": ["answer"]}, "missing": "error"}],
                "outputs": [{"name": "answer", "source": {"kind": "input", "path": ["answer"]}}],
            }}},
        ],
        "edges": [
            {"id": "a", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "code", "target_port_id": "in"},
            {"id": "b", "kind": "flow", "source_node_id": "code", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


def _profile(profile_id="builtin", executable=None):
    return PythonEnvironmentProfile.venv(profile_id, executable or sys.executable)


def _runner(tmp_path, source_root, *, document=None, profiles=None, staging_root=None):
    document = _document() if document is None else document
    plan = prepare_workflow(document)
    repository = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3", object_directory=tmp_path / "objects",
    ).bind_execution("execution")
    store = WorkflowExecutionStore(
        repository, plan, graph_id="graph",
        attempt=AttemptRef(GenerationRef("execution", "generation"), "attempt"),
    )
    profiles = {"builtin": _profile()} if profiles is None else profiles
    runtime = SkeletonCodeRuntime(
        plan,
        source_resolver=FileCodeSourceResolver(source_root),
        profile_resolver=lambda profile_id: profiles.get(profile_id),
        staging_root=staging_root or (tmp_path / "staging"),
        poll_interval_s=0.01,
    )
    return DurableWorkflowRunner(store, {"code": runtime}), runtime


def _write(root: Path, body: str):
    root.mkdir(parents=True, exist_ok=True)
    (root / "main.py").write_text(body, encoding="utf-8")


def test_real_python_process_keeps_logs_separate_and_persists_fingerprints(tmp_path, capsys):
    source = tmp_path / "source"
    _write(source, "def main(input, context):\n print('user-log:'+input['text'])\n return {'answer': input['text'].upper()}\n")
    run, _ = _runner(tmp_path, source)
    run.create({"text": "héllo"})
    assert run.run().outputs == {"answer": "HÉLLO"}
    assert capsys.readouterr().out == ""
    state = run.store.load().state
    code_row = next(row for row in state["completed"] if row["node_id"] == "code")
    raw = run.store.read_json(code_row["result_ref"])["_skeleton_code"]
    assert raw["schema"] == "unchain.skeleton.code-receipt.v1"
    assert raw["result"] == {"answer": "HÉLLO"}
    assert raw["logs"]["stdout"] == "user-log:héllo\n"
    assert len(raw["code_sha256"]) == len(raw["environment_sha256"]) == 64


def test_real_user_venv_and_standalone_provider_free_execution(tmp_path):
    # Qualify an actual user venv even when CI runs pytest from a base Python.
    # Let Python create it; copying a standalone executable loses its resources.
    environment = tmp_path / "user-venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    source = tmp_path / "source"
    _write(source, "import sys\ndef main(input, context):\n return {'answer': str(sys.prefix != sys.base_prefix)+':'+context['node_id']+':'+input['text']}\n")
    run, _ = _runner(tmp_path, source, profiles={"builtin": _profile("builtin", executable)})
    run.create({"text": "venv"})
    assert run.run().outputs == {"answer": "True:code:venv"}


def test_real_pupu_bundled_python_runs_the_same_protocol_when_supplied(tmp_path):
    executable = os.environ.get("PUPU_BUNDLED_PYTHON")
    if not executable:
        pytest.skip("set PUPU_BUNDLED_PYTHON to the retained PuPu bundled interpreter")
    source = tmp_path / "source"
    _write(source, "def main(input, context):\n return {'answer':'bundled:'+input['text']}\n")
    profile = PythonEnvironmentProfile.bundled("builtin", (executable,))
    run, _ = _runner(tmp_path, source, profiles={"builtin": profile})
    run.create({"text": "candidate"})
    assert run.run().outputs == {"answer": "bundled:candidate"}


def test_real_skill_read_to_code_to_end_uses_explicit_host_injection(tmp_path):
    identity = {"source": "project-agents", "source_id": "skill.md", "name": "demo"}
    document = _document()
    document["nodes"].insert(1, {
        "id": "skill", "type": "skill_read", "config": {
            "execution": {"inputs": [], "outputs": [
                {"name": "content", "source": {"kind": "result", "path": ["content"]}},
            ]},
            "skill": identity,
        },
    })
    document["nodes"][2]["config"]["execution"]["inputs"] = [{
        "name": "text", "value": {"kind": "ref", "node_id": "skill", "path": ["content"]}, "missing": "error",
    }]
    document["edges"] = [
        {"id": "a", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "skill", "target_port_id": "in"},
        {"id": "b", "kind": "flow", "source_node_id": "skill", "source_port_id": "out", "target_node_id": "code", "target_port_id": "in"},
        {"id": "c", "kind": "flow", "source_node_id": "code", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
    ]
    source = tmp_path / "source"
    _write(source, "def main(input, context):\n return {'answer':'skill:'+input['text']}\n")
    plan = prepare_workflow(document)
    repository = SQLiteContextV2Store(database_path=tmp_path / "journal.sqlite3", object_directory=tmp_path / "objects").bind_execution("execution")
    store = WorkflowExecutionStore(repository, plan, graph_id="graph", attempt=AttemptRef(GenerationRef("execution", "g"), "a"))

    class Registry:
        def read_exact(self, requested):
            assert requested == identity
            return {"identity": identity, "body": "body", "revision": "sha256:" + "a" * 64}

    code = SkeletonCodeRuntime(
        plan, source_resolver=FileCodeSourceResolver(source),
        profile_resolver=lambda profile_id: _profile() if profile_id == "builtin" else None,
        staging_root=tmp_path / "staging",
    )
    run = DurableWorkflowRunner(store, {"skill_read": SkeletonSkillRuntime(plan, Registry()), "code": code})
    run.create({"text": "ignored"})
    assert run.run().outputs == {"answer": "skill:body"}


@pytest.mark.parametrize("body, code", [
    ("def main(input, context):\n return {'answer': 1}\n", "output_invalid"),
    ("def main(input, context):\n return {'answer': {1, 2}}\n", "invalid_json"),
    ("def main(input, context):\n return ['not-an-object']\n", "invalid_type"),
])
def test_authenticated_invalid_return_is_known_failure_and_never_retried(tmp_path, body, code):
    source = tmp_path / "source"; _write(source, body)
    run, _ = _runner(tmp_path, source); run.create({"text": "x"})
    result = run.run()
    assert result.status == "failed" and result.error["code"] == code
    assert run.run() == result


def test_source_is_snapshotted_and_staged_file_mutation_never_changes_original(tmp_path):
    source = tmp_path / "source"; marker = tmp_path / "continue"
    _write(source, "from pathlib import Path\nimport time\ndef main(input, context):\n p=Path(input['text'])\n while not p.exists(): time.sleep(.01)\n Path(__file__).write_text('staged-only')\n return {'answer':'original'}\n")
    run, _ = _runner(tmp_path, source); run.create({"text": str(marker)})
    outcome = []
    thread = threading.Thread(target=lambda: outcome.append(run.run())); thread.start()
    deadline = time.time() + 5
    while not list((tmp_path / "staging").glob("**/main.py")) and time.time() < deadline:
        time.sleep(.01)
    (source / "main.py").write_text("def main(input, context): return {'answer':'changed'}", encoding="utf-8")
    marker.write_text("go")
    thread.join(10)
    assert not thread.is_alive() and outcome[0].outputs == {"answer": "original"}
    assert "changed" in (source / "main.py").read_text()


def test_timeout_and_crash_are_outcome_unknown_and_do_not_retry(tmp_path):
    source = tmp_path / "source"
    _write(source, "import time\ndef main(input, context):\n time.sleep(30)\n return {'answer':'late'}\n")
    document = _document(timeout_ms=100)
    run, _ = _runner(tmp_path, source, document=document); run.create({"text": "x"})
    assert run.run().status == "outcome_unknown"
    assert run.run().status == "outcome_unknown"
    other = tmp_path / "other"; _write(other, "import os\ndef main(input, context):\n os._exit(23)\n")
    crash, _ = _runner(tmp_path / "crash", other); crash.create({"text": "x"})
    assert crash.run().status == "outcome_unknown"


def test_timeout_reaps_descendant_process_tree(tmp_path):
    source = tmp_path / "source"; pid_file = tmp_path / "timeout-child.pid"
    _write(source, "import subprocess,sys,time\nfrom pathlib import Path\ndef main(input, context):\n p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n Path(input['text']).write_text(str(p.pid))\n time.sleep(60)\n return {'answer':'late'}\n")
    run, _ = _runner(tmp_path, source, document=_document(timeout_ms=250)); run.create({"text": str(pid_file)})
    assert run.run().status == "outcome_unknown" and pid_file.exists()
    child_pid = int(pid_file.read_text())
    deadline = time.time() + 5
    while _pid_alive(child_pid) and time.time() < deadline: time.sleep(.05)
    assert not _pid_alive(child_pid)


def test_concurrent_cancel_kills_process_group_and_late_result_cannot_publish(tmp_path):
    source = tmp_path / "source"; pid_file = tmp_path / "child.pid"
    _write(source, "import subprocess,sys,time\nfrom pathlib import Path\ndef main(input, context):\n p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n Path(input['text']).write_text(str(p.pid))\n time.sleep(60)\n return {'answer':'late'}\n")
    run, _ = _runner(tmp_path, source); run.create({"text": str(pid_file)})
    errors = []
    thread = threading.Thread(target=lambda: _capture(errors, run.run)); thread.start()
    deadline = time.time() + 8
    while not pid_file.exists() and time.time() < deadline: time.sleep(.01)
    assert pid_file.exists()
    assert run.cancel().status == "cancelled"
    thread.join(10)
    assert not thread.is_alive() and isinstance(errors[0], JournalConflictError)
    child_pid = int(pid_file.read_text())
    deadline = time.time() + 5
    while _pid_alive(child_pid) and time.time() < deadline: time.sleep(.05)
    assert not _pid_alive(child_pid)
    assert run.inspect().status == "cancelled" and run.inspect().outputs is None


def _capture(values, function):
    try: values.append(function())
    except BaseException as error: values.append(error)


def _pid_alive(pid):
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


def test_limits_bad_runtime_path_escape_and_plan_mismatch_fail_closed(tmp_path):
    source = tmp_path / "source"; _write(source, "def main(input, context):\n print('x'*10000)\n return {'answer':'y'*10000}\n")
    document = _document(max_result_bytes=100, max_log_bytes=64)
    run, _ = _runner(tmp_path, source, document=document); run.create({"text": "x"})
    result = run.run()
    assert result.status == "failed" and result.error["code"] == "limit_bytes"

    escaped = _document(source="../outside.py")
    escaped_run, _ = _runner(tmp_path / "escaped", source, document=escaped)
    with pytest.raises(SkeletonValidationError) as error:
        escaped_run.create({"text": "x"})
    assert error.value.code == "invalid_path"

    missing = {"builtin": PythonEnvironmentProfile.venv("builtin", tmp_path / "missing-python")}
    missing_run, _ = _runner(tmp_path / "missing", source, profiles=missing)
    with pytest.raises(SkeletonValidationError) as error:
        missing_run.create({"text": "x"})
    assert error.value.code == "missing_runtime"

    plan_a = prepare_workflow(_document())
    different = _document(); different["name"] = "Different"
    plan_b = prepare_workflow(different)
    runtime = SkeletonCodeRuntime(
        plan_a, source_resolver=FileCodeSourceResolver(source),
        profile_resolver=lambda _: _profile(), staging_root=tmp_path / "mismatch-stage",
    )
    repository = SQLiteContextV2Store(database_path=tmp_path / "mismatch.sqlite", object_directory=tmp_path / "mismatch-objects").bind_execution("mismatch")
    store = WorkflowExecutionStore(repository, plan_b, graph_id="mismatch", attempt=AttemptRef(GenerationRef("mismatch", "g"), "a"))
    with pytest.raises(SkeletonValidationError) as error:
        DurableWorkflowRunner(store, {"code": runtime})
    assert error.value.code == "identity_mismatch"


def test_source_and_environment_content_admission_fail_closed(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    target = source / "target.py"; target.write_text("def main(input, context): return {'answer':'ok'}")
    (source / "main.py").symlink_to(target)
    symlink_run, _ = _runner(tmp_path / "symlink", source)
    with pytest.raises(SkeletonValidationError) as error:
        symlink_run.create({"text": "x"})
    assert error.value.code == "invalid_path"

    (source / "main.py").unlink()
    (source / "main.py").write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    large_run, _ = _runner(tmp_path / "large", source)
    with pytest.raises(SkeletonValidationError) as error:
        large_run.create({"text": "x"})
    assert error.value.code == "limit_bytes"

    (source / "main.py").write_text("def main(input, context): return {'answer':'ok'}")
    non_executable = tmp_path / "not-executable"
    non_executable.write_text("not a runtime")
    non_executable.chmod(0o600)
    permission_run, _ = _runner(
        tmp_path / "permission", source,
        profiles={"builtin": PythonEnvironmentProfile.venv("builtin", non_executable)},
    )
    with pytest.raises(SkeletonValidationError) as error:
        permission_run.create({"text": "x"})
    assert error.value.code == "missing_runtime"

    malformed = _document(); malformed["nodes"][1]["config"]["code"]["unexpected"] = True
    with pytest.raises(SkeletonValidationError) as error:
        _runner(tmp_path / "closed", source, document=malformed)
    assert error.value.code == "unknown_field"
    malformed_plan = prepare_workflow(malformed)
    malformed_repository = SQLiteContextV2Store(
        database_path=tmp_path / "generic.sqlite", object_directory=tmp_path / "generic-objects",
    ).bind_execution("generic")
    malformed_store = WorkflowExecutionStore(
        malformed_repository, malformed_plan, graph_id="generic",
        attempt=AttemptRef(GenerationRef("generic", "g"), "a"),
    )
    with pytest.raises(SkeletonValidationError) as error:
        DurableWorkflowRunner(malformed_store, {"code": lambda *_: None})
    assert error.value.code == "unknown_field"


def test_runtime_binary_change_after_binding_fails_before_process_intent(tmp_path):
    source = tmp_path / "source"; _write(source, "def main(input, context): return {'answer':'never'}")
    executable = tmp_path / "runtime"
    executable.write_text("#!/bin/sh\nexit 70\n")
    executable.chmod(0o700)
    run, _ = _runner(
        tmp_path / "changed", source,
        profiles={"builtin": PythonEnvironmentProfile.bundled("builtin", (executable,))},
    )
    run.create({"text": "x"})
    run.run(max_steps=1)  # complete Start; Code is still ready and has no intent
    executable.write_text("#!/bin/sh\nexit 71\n")
    executable.chmod(0o700)
    result = run.run()
    assert result.status == "failed" and result.error["code"] == "environment_changed"
    assert run.store.load().state["active"] is None


def test_wrong_worker_identity_and_missing_dependency_are_unknown(tmp_path):
    source = tmp_path / "source"; _write(source, "def main(input, context): return {'answer':'ok'}")
    malicious = tmp_path / "malicious.py"
    malicious.write_text("import json,sys\nrequest,result=sys.argv[-2:]\nr=json.load(open(request)); r['identity']['node_id']='other'\njson.dump({'schema':'unchain.skeleton.code-worker.v1','identity':r['identity'],'outcome':{'status':'success','result':{'answer':'forged'}}},open(result,'w'))\n")
    profile = PythonEnvironmentProfile.bundled("builtin", (sys.executable, str(malicious)))
    run, _ = _runner(tmp_path / "wrong", source, profiles={"builtin": profile}); run.create({"text": "x"})
    assert run.run().status == "outcome_unknown"

    dependency = tmp_path / "dependency"; _write(dependency, "import package_that_does_not_exist_356\ndef main(input, context): return {'answer':'x'}")
    failed, _ = _runner(tmp_path / "dependency-run", dependency); failed.create({"text": "x"})
    assert failed.run().status == "outcome_unknown"


def test_log_limit_truncates_each_stream_without_corrupting_success(tmp_path):
    source = tmp_path / "source"
    _write(source, "import sys\ndef main(input, context):\n sys.stdout.write('a'*(2*1024*1024))\n sys.stderr.write('b'*(2*1024*1024))\n return {'answer':'ok'}\n")
    run, _ = _runner(tmp_path, source, document=_document(max_log_bytes=32)); run.create({"text": "x"})
    assert run.run().outputs == {"answer": "ok"}
    row = next(row for row in run.store.load().state["completed"] if row["node_id"] == "code")
    logs = run.store.read_json(row["result_ref"])["_skeleton_code"]["logs"]
    assert len(logs["stdout"].encode()) <= 32 and len(logs["stderr"].encode()) <= 32
    assert logs["stdout_truncated"] and logs["stderr_truncated"]


def test_cold_runner_uses_persisted_source_and_rejects_rebound_profile(tmp_path):
    source = tmp_path / "source"; _write(source, "def main(input, context): return {'answer':'original'}")
    run, _ = _runner(tmp_path, source); run.create({"text": "x"}); run.run(max_steps=1); run.pause()
    _write(source, "def main(input, context): return {'answer':'changed'}")
    recovered, _ = _runner(tmp_path, source)
    assert recovered.resume().outputs == {"answer": "original"}

    source = tmp_path / "environment"; _write(source, "def main(input, context): return {'answer':'never'}")
    run, _ = _runner(tmp_path / "profile", source); run.create({"text": "x"}); run.run(max_steps=1); run.pause()
    alias = tmp_path / "other-python"; alias.symlink_to(sys.executable)
    rebound, _ = _runner(tmp_path / "profile", source, profiles={"builtin": _profile(executable=alias)})
    assert rebound.resume().status == "failed"
    assert rebound.inspect().error["code"] == "environment_changed"


def test_saved_receipt_reaches_end_when_source_was_deleted(tmp_path):
    source = tmp_path / "source"; _write(source, "def main(input, context): return {'answer':'saved'}")
    run, _ = _runner(tmp_path, source); run.create({"text": "x"})
    assert run.run(max_steps=2).status == "result_received"
    (source / "main.py").unlink()
    recovered, _ = _runner(tmp_path, source)
    assert recovered.run().outputs == {"answer": "saved"}


def test_code_artifact_mapping_is_scoped_durable_and_a_private_copy(tmp_path):
    source = tmp_path / "source"
    _write(source, "from pathlib import Path\ndef main(input, context):\n p=Path(context['artifacts']['upload']['path']); before=p.read_bytes(); p.write_bytes(b'changed'); return {'answer':before.decode()}\n")
    document = _document(artifacts=({"name": "upload", "attachment": "file"},))
    run, _ = _runner(tmp_path, source, document=document)
    run.create({"text": "x"}, attachments={"file": (b"durable", "application/octet-stream")})
    run.run(max_steps=1); (source / "main.py").unlink()
    recovered, _ = _runner(tmp_path, source, document=document)
    assert recovered.resume().outputs == {"answer": "durable"}
    assert recovered.store.read_bytes(recovered.store.load().state["attachments"]["file"]) == b"durable"


@pytest.mark.parametrize("expression", ["{'answer':(1, 2)}", "{'answer':{1:'integer-key'}}"])
def test_worker_rejects_python_values_that_json_would_normalize(tmp_path, expression):
    source = tmp_path / "source"; _write(source, "def main(input, context): return " + expression)
    run, _ = _runner(tmp_path, source, document=_document(result_schema={"type": "object"})); run.create({"text": "x"})
    assert run.run().status == "failed"
    assert run.inspect().error["code"] == "invalid_json"


@pytest.mark.parametrize("mode", ("success", "timeout"))
def test_worker_reaps_term_resistant_descendants_after_every_outcome(tmp_path, mode):
    source = tmp_path / "source"; pid_file = tmp_path / "child.pid"
    child = "import os,signal,sys,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    body = (
        "import subprocess,sys,time\nfrom pathlib import Path\ndef main(input, context):\n"
        f" subprocess.Popen([sys.executable,'-c',{child!r},input['text']])\n"
        " while not Path(input['text']).exists(): time.sleep(.01)\n"
        + (" time.sleep(60)\n" if mode == "timeout" else "") + " return {'answer':'done'}\n"
    )
    _write(source, body)
    run, _ = _runner(tmp_path, source, document=_document(timeout_ms=500)); run.create({"text": str(pid_file)})
    child_pid = None
    try:
        run.run(); child_pid = int(pid_file.read_text())
        deadline = time.monotonic() + 2
        while _pid_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(.02)
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(child_pid)], capture_output=True, text=True).stdout.strip()
        assert not state or state.startswith("Z")
    finally:
        if child_pid and _pid_alive(child_pid):
            os.kill(child_pid, 9)


def test_cancellation_observed_before_launch_prevents_process_dispatch(tmp_path, monkeypatch):
    source = tmp_path / "source"; marker = tmp_path / "marker"
    _write(source, "from pathlib import Path\ndef main(input, context): Path(input['text']).write_text('executed'); return {'answer':'done'}")
    run, _ = _runner(tmp_path, source); run.create({"text": str(marker)})
    original = SkeletonCodeRuntime._command
    starts = []
    def cancel_before_dispatch(profile, request, result):
        run.cancel()
        return original(profile, request, result)
    def forbidden_popen(*args, **kwargs):
        starts.append(args)
        raise AssertionError("Code process launched after cancellation")
    monkeypatch.setattr(SkeletonCodeRuntime, "_command", staticmethod(cancel_before_dispatch))
    monkeypatch.setattr(subprocess, "Popen", forbidden_popen)
    with pytest.raises(JournalConflictError):
        run.run()
    assert starts == [] and not marker.exists()


def test_shared_official_runtime_keeps_execution_source_snapshots_isolated(tmp_path):
    source = tmp_path / "source"
    _write(source, "def main(input, context): return {'answer':'first'}")
    first, runtime = _runner(tmp_path, source)
    first.create({"text": "x"}); first.run(max_steps=1)
    _write(source, "def main(input, context): return {'answer':'second'}")
    second_store = WorkflowExecutionStore(
        first.store.journal, first.plan, graph_id="second",
        attempt=AttemptRef(GenerationRef("execution", "generation"), "second-attempt"),
    )
    second = DurableWorkflowRunner(second_store, {"code": runtime})
    second.create({"text": "x"})
    assert first.run().outputs == {"answer": "first"}
    assert second.run().outputs == {"answer": "second"}


def test_unbound_legacy_execution_cannot_inherit_shared_runtime_binding(tmp_path):
    source = tmp_path / "source"
    _write(source, "def main(input, context): return {'answer':'first'}")
    first, runtime = _runner(tmp_path, source)
    first.create({"text": "x"})
    legacy_store = WorkflowExecutionStore(
        first.store.journal, first.plan, graph_id="legacy",
        attempt=AttemptRef(GenerationRef("execution", "generation"), "legacy-attempt"),
    )
    legacy = DurableWorkflowRunner(legacy_store, {"code": lambda *args: None})
    legacy.create({"text": "x"})
    result = DurableWorkflowRunner(legacy_store, {"code": runtime}).run()
    assert result.status == "failed"
    assert result.error["code"] == "identity_mismatch"


def _initial_state(run, inputs_ref, attachments, code_bindings):
    attempt = run.store.attempt
    digest = execution_definition_digest(run.plan)
    return {
        "schema": "unchain.skeleton_execution_state.v2", "graph_id": run.store.graph_id,
        "attempt": attempt.to_dict(), "definition_digest": digest,
        "call_stack": [{
            "execution_id": attempt.generation.execution_id,
            "graph_invocation_id": run.store.graph_id, "attempt_id": attempt.attempt_id,
            "definition_id": run.plan.document_id, "definition_digest": digest,
        }],
        "inputs_ref": inputs_ref, "attachments": attachments, "code_bindings": code_bindings,
        "completed": [], "active": None, "status": "ready", "paused": False,
        "error": None, "root_committed": False,
    }


@pytest.mark.parametrize("mutation", ("mapping", "profile", "missing_node"))
def test_store_rejects_code_bindings_that_disagree_with_fixed_node_config(tmp_path, mutation):
    source = tmp_path / "source"
    _write(source, "def main(input, context): return {'answer':'unused'}")
    document = _document(artifacts=({"name": "upload", "attachment": "public"},))
    run, runtime = _runner(tmp_path, source, document=document)
    inputs, inputs_pending = run.store.prepare_json({"text": "x"}, key="root.inputs")
    attachments = {}; pending = [inputs_pending]
    for name in ("private", "public"):
        ref, artifact = run.store.prepare_bytes(name.encode(), media_type="text/plain", key="attachment." + name)
        attachments[name] = ref; pending.append(artifact)
    bindings, code_pending = runtime.prepare_execution_bindings(attachments, run.store.prepare_bytes)
    pending.extend(code_pending)
    if mutation == "mapping":
        bindings["code"]["artifacts"][0].update(attachment="private", ref=attachments["private"])
    elif mutation == "profile":
        bindings["code"]["profile"]["id"] = "other-profile"
    else:
        bindings = {}
        pending.pop()
    with pytest.raises((SkeletonValidationError, DurableStoreError)):
        run.store.save(
            _initial_state(run, inputs, attachments, bindings), expected_revision=0,
            operation_id="malicious-create", artifacts=tuple(pending),
        )


def test_code_and_attachment_resource_order_is_canonical(tmp_path):
    source = tmp_path / "source"
    _write(source, "def main(input, context): return {'answer':input['text']+'!'}")
    document = _document()
    document["nodes"][1]["id"] = "zcode"
    second = deepcopy(document["nodes"][1]); second["id"] = "acode"
    second["config"]["execution"]["inputs"][0]["value"] = {
        "kind": "ref", "node_id": "zcode", "path": ["answer"],
    }
    document["nodes"].insert(2, second)
    document["nodes"][-1]["config"]["execution"]["inputs"][0]["value"]["node_id"] = "acode"
    document["edges"][0]["target_node_id"] = "zcode"
    document["edges"][1].update(source_node_id="zcode", target_node_id="acode")
    document["edges"].append({
        "id": "c", "kind": "flow", "source_node_id": "acode", "source_port_id": "out",
        "target_node_id": "end", "target_port_id": "in",
    })
    run, _ = _runner(tmp_path, source, document=document)
    run.create({"text": "x"}, attachments={"z": (b"Z", "text/plain"), "a": (b"A", "text/plain")})
    recovered, _ = _runner(tmp_path, source, document=document)
    assert recovered.run().outputs == {"answer": "x!!"}


@pytest.mark.parametrize("name", ([], {}))
def test_non_string_code_artifact_name_has_structured_diagnostic(tmp_path, name):
    source = tmp_path / "source"
    _write(source, "def main(input, context): return {'answer':'x'}")
    with pytest.raises(SkeletonValidationError) as caught:
        _runner(tmp_path, source, document=_document(artifacts=({"name": name, "attachment": "file"},)))
    assert caught.value.code == "invalid_type"
