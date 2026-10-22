"""Cleanup must preserve the failure that stopped a background operation."""

from types import SimpleNamespace
import threading

import pytest

from unchain.jobs._worker import _HeartbeatLease
from unchain.durability import (
    DurablePersistenceTraversalExhaustedError,
    find_durable_persistence_failure,
)
from unchain.retry import RetryConfig, RetryContext
from unchain.retry.executor import execute_with_retry
from unchain.providers.exact_route_transport import OllamaExactRouteTransport
from unchain.providers.ollama import OllamaModelIO
from unchain.subagents.executor import SubagentExecutor
from unchain.subagents.types import SubagentResult
from unchain.toolkits.builtin.core import shell_runtime
from unchain.toolkits.builtin.core.lsp_runtime import LSPRuntime
from tests.test_durable_provider_turn_runtime import _catalog


class _FailingPipe:
    def __init__(self, failure):
        self.failure = failure
        self.closed = False

    def read(self, size):
        raise self.failure

    def readline(self):
        raise self.failure

    def close(self):
        self.closed = True
        raise OSError("pipe close failed")


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("reader_kind", ["lsp", "shell"])
def test_pipe_cleanup_preserves_read_failure(monkeypatch, failure_type, reader_kind):
    failure = failure_type("read failed")
    pipe = _FailingPipe(failure)
    if reader_kind == "lsp":
        reader = lambda: LSPRuntime._stderr_reader(
            None, SimpleNamespace(stderr_lines=[]), pipe
        )
    else:
        class CapturedThread:
            def __init__(self, *, target, **kwargs):
                self.target = target

            def start(self):
                pass

        monkeypatch.setattr(shell_runtime.threading, "Thread", CapturedThread)
        task = SimpleNamespace(task_id="cleanup-test", reader_threads=[])
        shell_runtime.ShellRuntime._start_reader(None, task, "stdout", pipe)
        reader = task.reader_threads[0].target

    with pytest.raises(failure_type) as caught:
        reader()
    assert caught.value is failure
    assert pipe.closed is True


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_heartbeat_does_not_swallow_termination(failure_type):
    failure = failure_type("stop worker")

    def write_heartbeat(*args, **kwargs):
        raise failure

    lease = _HeartbeatLease(
        store=SimpleNamespace(write_heartbeat=write_heartbeat),
        job_id="job-test",
        worker_token="worker-test",
        worker_pid=123,
        interval_ms=1,
    )
    waits = iter([False, True])
    lease._stop = SimpleNamespace(wait=lambda timeout: next(waits))

    with pytest.raises(failure_type) as caught:
        lease._run()
    assert caught.value is failure


def test_heartbeat_retries_transient_write_error():
    calls = []

    def write_heartbeat(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("temporary storage outage")
        return True

    lease = _HeartbeatLease(
        store=SimpleNamespace(write_heartbeat=write_heartbeat),
        job_id="job-test",
        worker_token="worker-test",
        worker_pid=123,
        interval_ms=1,
    )
    waits = iter([False, False, True])
    lease._stop = SimpleNamespace(wait=lambda timeout: next(waits))

    lease._run()

    assert len(calls) == 2
    assert lease.last_error == ""


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_retry_records_interrupted_attempt_and_propagates_without_retry(failure_type):
    failure = failure_type("operation interrupted")
    attempts = []
    completions = []
    delays = []

    def operation():
        attempts.append(True)
        raise failure

    with pytest.raises(failure_type) as caught:
        execute_with_retry(
            operation,
            RetryConfig(max_retries=2),
            RetryContext(run_id="interrupt-test", iteration=0, is_background=False),
            after_attempt=lambda *args: completions.append(args),
            sleep=delays.append,
        )

    assert caught.value is failure
    assert len(attempts) == len(completions) == 1
    assert completions[0][-1] == "terminal"
    assert delays == []


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_subagent_batch_joins_sibling_before_propagating_interrupt(failure_type):
    failure = failure_type("child interrupted")
    sibling_ready = threading.Event()
    sibling_finished = threading.Event()

    def run_item(index, item):
        if index == 0:
            assert sibling_ready.wait(timeout=5)
            raise failure
        sibling_ready.set()
        sibling_finished.set()
        return SubagentResult(
            mode="worker", agent_name="sibling", template_name=None, status="completed"
        )

    with pytest.raises(failure_type) as caught:
        SubagentExecutor(max_parallel_workers=2).execute_batch(
            items=[{}, {}], run_item=run_item
        )

    assert caught.value is failure
    assert sibling_finished.is_set()


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_unreadable_exception_evidence_is_never_classified_retryable(failure_type):
    class UnreadableError(RuntimeError):
        def __getattribute__(self, name):
            if name == "__cause__":
                raise failure_type("cannot inspect provider exception")
            return super().__getattribute__(name)

    result = find_durable_persistence_failure(UnreadableError("provider failed"))

    assert isinstance(result, DurablePersistenceTraversalExhaustedError)


@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_preview_discard_finishes_cleanup_before_propagating_interrupt(failure_type):
    failure = failure_type("preview discard interrupted")
    discarded = []

    class Callback:
        def __call__(self, event):
            pass

        def emit_provisional_reasoning(self, event, preview_id):
            pass

        def commit_provisional_reasoning(self, **kwargs):
            pass

        def discard_provisional_reasoning(self, *, preview_id, **kwargs):
            discarded.append(preview_id)
            if len(discarded) == 1:
                raise failure

    base_catalog = _catalog()
    catalog = type(base_catalog)(
        attempt=base_catalog.attempt,
        iteration=base_catalog.iteration,
        provider="ollama",
        model="local-test",
        semantic_schemas=[],
        entries=[],
        required_betas_sha256=base_catalog.required_betas_sha256,
        prompt_sha256=base_catalog.prompt_sha256,
        exposure_plan_sha256=base_catalog.exposure_plan_sha256,
    )
    transport = OllamaExactRouteTransport(
        model_io=OllamaModelIO(model="local-test", default_payloads={}, model_capabilities={}),
        catalog=catalog,
        callback=Callback(),
        emit_stream=True,
    )
    transport._provisional_reasoning_id = "a" * 32
    transport._provisional_iteration = 1
    transport._capture_event({"type": "reasoning", "delta": "first"})
    transport._capture_event({"type": "reasoning", "delta": "second"})

    with pytest.raises(failure_type) as caught:
        transport.discard_buffered_events()

    assert caught.value is failure
    assert len(discarded) == 2
    assert transport.buffered_events == ()
    assert transport._preview_ids_by_index == {}
    assert transport._provisional_reasoning_id is None
