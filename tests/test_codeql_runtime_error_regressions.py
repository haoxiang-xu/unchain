from __future__ import annotations

import hashlib
import threading

import pytest

from tests.context_v2.test_context_runtime_factory import (
    _bundle,
    _context,
    _current_input,
)
from unchain.context import ContextRuntime, DurableContextRuntimeFactory
from unchain.execution import (
    ExecutionLeaseConfig,
    ExecutionLeaseNotOwnedError,
    ExecutionRuntime,
    _borrow_execution_guard,
)
from unchain.journal import OperationRef
from unchain.kernel import ToolCall
from unchain.memory import InMemorySessionStore
from unchain.persistence import SQLiteContextV2Store
from unchain.subagents.plugin import _BatchExecutionGuard
from unchain.tools import Toolkit
from unchain.tools.confirmation import execute_confirmable_tool_call


@pytest.mark.parametrize("decision", ["approved", "denied", "missing"])
def test_confirmation_returns_an_outcome_on_every_tool_resolution_branch(decision):
    toolkit = Toolkit()
    executions = []

    def write(value: int):
        executions.append(value)
        return {"written": value}

    toolkit.register(write, name="write", requires_confirmation=True)
    outcome = execute_confirmable_tool_call(
        toolkit=toolkit,
        tool_call=ToolCall(
            call_id="call-1",
            name="missing" if decision == "missing" else "write",
            arguments={"value": 7},
        ),
        on_tool_confirm=lambda request: decision == "approved",
        loop=None,
        callback=None,
        run_id="run-1",
        iteration=0,
    )

    if decision == "approved":
        assert executions == [7]
        assert outcome.tool_result == {"written": 7}
        assert outcome.capability_outcome.value == {"written": 7}
    else:
        assert executions == []
        assert outcome.capability_outcome is None
        if decision == "denied":
            assert outcome.denied is True
            assert outcome.tool_result["denied"] is True
        else:
            assert outcome.denied is False
            assert "error" in outcome.tool_result


@pytest.mark.parametrize("guard_kind", ["none", "root", "borrowed", "batch"])
def test_repeated_bootstrap_preserves_its_execution_guard_binding(guard_kind):
    execution_runtime = ExecutionRuntime(
        InMemorySessionStore(),
        ExecutionLeaseConfig(heartbeat_interval_ms=0),
    )
    root = execution_runtime.acquire("execution-1", "owner-1")
    guard = root
    session_id = "execution-1"
    if guard_kind == "borrowed":
        session_id = "execution-1:child-1"
        guard = _borrow_execution_guard(root, session_id=session_id)
    elif guard_kind == "batch":
        guard = _BatchExecutionGuard(root, threading.Event())
    bundles = []

    def build(attempt):
        bundle = _bundle(attempt)
        bundles.append(bundle)
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id="context-v2",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, execution_id: "generation-1",
            current_input_resolver=_current_input,
        ),
    )
    bootstrap = _context(session_id=session_id, run_id="attempt-1")
    if guard_kind != "none":
        bootstrap.event["execution_guard"] = guard

    try:
        harness = runtime.build_harnesses()[0]
        harness.build_delta(bootstrap)
        harness.build_delta(bootstrap)

        assert len(bundles) == 1
        assert bundles[0].attempt.generation.execution_id == session_id
        assert guard.assert_active().execution_id == session_id
    finally:
        guard.release()
        root.release()


def test_borrowed_guard_wait_and_release_preserve_the_parent_lease():
    runtime = ExecutionRuntime(
        InMemorySessionStore(), ExecutionLeaseConfig(heartbeat_interval_ms=0)
    )
    root = runtime.acquire("execution-1", "owner-1")
    child = _borrow_execution_guard(root, session_id="execution-1:child-1")
    try:
        root_fence = root.fence
        child.release_for_wait()
        with pytest.raises(ExecutionLeaseNotOwnedError):
            child.assert_active()
        assert root.assert_active().fence == root_fence

        assert child.reacquire().execution_id == "execution-1:child-1"
        assert child.renew().fencing_token == root_fence.fencing_token
        child.release()
        with pytest.raises(ExecutionLeaseNotOwnedError):
            child.assert_active()
        assert root.assert_active().fence == root_fence
    finally:
        child.release()
        root.release()


def test_batch_abort_blocks_effects_and_keeps_the_delegate_releasable():
    runtime = ExecutionRuntime(
        InMemorySessionStore(), ExecutionLeaseConfig(heartbeat_interval_ms=0)
    )
    root = runtime.acquire("execution-1", "owner-1")
    failure_event = threading.Event()
    batch = _BatchExecutionGuard(root, failure_event)
    try:
        assert batch.renew().fence == root.fence
        batch.release_for_wait()
        assert batch.reacquire().execution_id == "execution-1"
        failure_event.set()
        with pytest.raises(RuntimeError, match="subagent_batch_durable_failure"):
            batch.assert_active()
        assert root.assert_active().execution_id == "execution-1"
        batch.release()
        with pytest.raises(ExecutionLeaseNotOwnedError):
            root.assert_active()
    finally:
        root.release()


def test_sqlite_bound_repository_initializes_artifact_and_result_store_scope(tmp_path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects",
    )
    repository = store.bind_execution("execution-1")
    content = b"verified artifact"
    artifact = repository.put(
        content=content,
        media_type="text/plain",
        operation=OperationRef("artifact-1", hashlib.sha256(content).hexdigest()),
    )

    assert repository.execution_id == "execution-1"
    assert repository.read_full_verified(artifact=artifact) == content
    assert store.bind_execution("execution-1").read_full_verified(artifact=artifact) == content
    with pytest.raises(ValueError):
        store.bind_execution("")
