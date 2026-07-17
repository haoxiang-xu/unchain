from __future__ import annotations

import copy
import hashlib
import json

import pytest

from unchain.kernel import BaseRuntimeHarness, ModelTurnResult, RunState
from unchain.kernel.delta import HarnessDelta
from unchain.kernel.types import ToolCall
from unchain.memory import KernelMemoryRuntime, MemoryManager
from unchain.memory.checkpoint_state import build_execution_checkpoint
from unchain.memory.revision import SessionRevisionConflictError, SessionSnapshot
from unchain.runtime import build_runtime_loop
from unchain.tools import Toolkit
from unchain.tools.receipts import (
    DURABLE_TOOL_RECEIPT_LEDGER_KEY,
    TOOL_RECEIPT_JOURNAL_KEY,
    DurableToolReceiptAckError,
    DurableToolReceiptAckHarness,
    DurableToolReceiptIntegrityError,
    acknowledge_durable_tool_receipt_applications,
    apply_durable_tool_receipt_ledger,
    build_durable_tool_receipt,
    durable_tool_receipt_state_update,
)
from unchain.tools.runtime import ToolRuntimeOutcome


class _RevisionStore:
    def __init__(self) -> None:
        self.state: dict[str, dict] = {}
        self.revisions: dict[str, int] = {}
        self.fail_next_cas = False

    def load(self, session_id: str) -> dict:
        return copy.deepcopy(self.state.get(session_id, {}))

    def save(self, session_id: str, state: dict) -> None:
        self.state[session_id] = copy.deepcopy(state)
        self.revisions[session_id] = self.revisions.get(session_id, 0) + 1

    def load_with_revision(self, session_id: str) -> SessionSnapshot:
        return SessionSnapshot(
            state=self.load(session_id),
            revision=self.revisions.get(session_id, 0),
        )

    def save_if_revision(
        self,
        session_id: str,
        state: dict,
        expected_revision: int,
    ) -> int:
        actual_revision = self.revisions.get(session_id, 0)
        if self.fail_next_cas:
            self.fail_next_cas = False
            raise SessionRevisionConflictError(
                session_id=session_id,
                expected_revision=expected_revision,
                actual_revision=actual_revision + 1,
            )
        if actual_revision != expected_revision:
            raise SessionRevisionConflictError(
                session_id=session_id,
                expected_revision=expected_revision,
                actual_revision=actual_revision,
            )
        next_revision = actual_revision + 1
        self.state[session_id] = copy.deepcopy(state)
        self.revisions[session_id] = next_revision
        return next_revision


class _QueueModelIO:
    provider = "openai"
    model = "gpt-5"

    def __init__(self, turns: list[ModelTurnResult], *, before_fetch=None):
        self.turns = list(turns)
        self.before_fetch = before_fetch
        self.requests = []

    def fetch_turn(self, request):
        if callable(self.before_fetch):
            self.before_fetch()
        self.requests.append(request)
        if not self.turns:
            raise AssertionError("unexpected model turn")
        return self.turns.pop(0)


class _ReceiptPlugin:
    def __init__(self, *, crash_on_ack: bool = False) -> None:
        self.crash_on_ack = crash_on_ack
        self.execute_calls = 0
        self.ack_attempts: list[str] = []
        self.acknowledged: set[str] = set()

    def can_handle(self, *, tool_call, context) -> bool:
        del context
        return tool_call.name == "demo_tool"

    def execute(self, *, tool_call, context) -> ToolRuntimeOutcome:
        self.execute_calls += 1
        result = {"ok": True, "stdout": "durable-output"}
        request_digest = hashlib.sha256(
            json.dumps(
                tool_call.arguments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result_digest = hashlib.sha256(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        receipt = build_durable_tool_receipt(
            source="tests.durable_job_result",
            receipt_id=f"job-result:{tool_call.call_id}",
            execution_id=context.session_id,
            logical_execution_id=f"logical:{context.session_id}",
            created_by_attempt_id=context.run_id,
            tool_call_id=tool_call.call_id,
            request_digest=request_digest,
            result_digest=result_digest,
            payload={
                "job_id": "job-test",
                "stdout_start": 0,
                "stdout_next": len(result["stdout"]),
            },
        )
        return ToolRuntimeOutcome(
            tool_result=result,
            state_updates=durable_tool_receipt_state_update(
                context.state,
                receipt,
            ),
        )

    def acknowledge_durable_receipt(self, *, receipt, application) -> bool:
        del application
        if receipt.get("source") != "tests.durable_job_result":
            return False
        receipt_id = str(receipt.get("receipt_id") or "")
        self.ack_attempts.append(receipt_id)
        if self.crash_on_ack:
            raise RuntimeError("crash after session commit before job ack")
        self.acknowledged.add(receipt_id)
        return True


class _CompleteAfterToolHarness(BaseRuntimeHarness):
    def build_delta(self, context):
        del context
        return HarnessDelta(
            created_by="tests.complete_after_tool",
            state_updates={"run_status": "completed"},
        )


def _tool_turn() -> ModelTurnResult:
    call = ToolCall(call_id="call-job-result", name="demo_tool", arguments={})
    return ModelTurnResult(
        assistant_messages=[
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": "{}",
            }
        ],
        tool_calls=[call],
        response_id="response-tool",
    )


def _final_turn(content: str = "done") -> ModelTurnResult:
    return ModelTurnResult(
        assistant_messages=[{"role": "assistant", "content": content}],
        tool_calls=[],
        final_text=content,
        response_id=f"response-{content}",
    )


def _toolkit() -> Toolkit:
    toolkit = Toolkit()
    toolkit.register(lambda: {"fallback": True}, name="demo_tool")
    return toolkit


def _build_loop(
    *,
    store: _RevisionStore,
    plugin: _ReceiptPlugin,
    model: _QueueModelIO,
    extra_harnesses: list | None = None,
):
    harnesses = [
        DurableToolReceiptAckHarness(plugins=(plugin,)),
        *(extra_harnesses or []),
    ]
    return build_runtime_loop(
        harnesses=harnesses,
        model_io=model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )


def _run(
    loop,
    *,
    session_id: str,
    plugin: _ReceiptPlugin,
    messages=None,
    max_iterations: int = 3,
    callback=None,
):
    return loop.run(
        [{"role": "user", "content": "run the durable job"}]
        if messages is None
        else messages,
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=_toolkit(),
        tool_runtime_plugins=[plugin],
        max_iterations=max_iterations,
        callback=callback,
        run_id=f"attempt:{session_id}",
    )


def _receipt(*, execution_id: str, call_id: str = "call-cross-session") -> dict:
    return build_durable_tool_receipt(
        source="tests.cross_session",
        receipt_id=f"receipt:{call_id}",
        execution_id=execution_id,
        logical_execution_id=f"logical:{execution_id}",
        created_by_attempt_id="attempt:test",
        tool_call_id=call_id,
        request_digest=hashlib.sha256(b"request").hexdigest(),
        result_digest=hashlib.sha256(b"result").hexdigest(),
        payload={"job_id": "job-cross-session"},
    )


def test_receipt_free_empty_session_commit_keeps_legacy_no_store_behavior() -> None:
    store = _RevisionStore()
    manager = MemoryManager(store=store)

    outcome = manager.commit_messages(
        "",
        [{"role": "assistant", "content": "ephemeral"}],
        return_result=True,
    )

    assert outcome is not None
    assert outcome.commit_info["session_id"] == ""
    assert outcome.session_snapshot.revision is None
    assert outcome.session_snapshot.state["messages"] == [
        {"role": "assistant", "content": "ephemeral"}
    ]
    assert store.state == {}


def test_receipt_payload_rejects_non_finite_json_numbers() -> None:
    with pytest.raises(
        DurableToolReceiptIntegrityError,
        match="JSON serializable",
    ):
        build_durable_tool_receipt(
            source="tests.strict-json",
            receipt_id="receipt:nan",
            execution_id="session-strict-json",
            logical_execution_id="logical-strict-json",
            created_by_attempt_id="attempt-strict-json",
            tool_call_id="call-strict-json",
            request_digest=hashlib.sha256(b"request").hexdigest(),
            result_digest=hashlib.sha256(b"result").hexdigest(),
            payload={"not_a_number": float("nan")},
        )


def test_one_tool_call_cannot_gain_a_second_receipt_after_application() -> None:
    first = _receipt(
        execution_id="session-one-call",
        call_id="call-one-receipt",
    )
    first_ledger = durable_tool_receipt_state_update(None, first)[
        DURABLE_TOOL_RECEIPT_LEDGER_KEY
    ]
    session_state: dict = {}
    applied = apply_durable_tool_receipt_ledger(
        session_state,
        ledger=first_ledger,
        application_id="checkpoint-one-receipt",
    )
    second = build_durable_tool_receipt(
        source="tests.cross_session",
        receipt_id="receipt:second-for-same-call",
        execution_id="session-one-call",
        logical_execution_id="logical:session-one-call",
        created_by_attempt_id="attempt:test",
        tool_call_id="call-one-receipt",
        request_digest=hashlib.sha256(b"request").hexdigest(),
        result_digest=hashlib.sha256(b"different-result").hexdigest(),
        payload={"job_id": "job-cross-session"},
    )

    with pytest.raises(
        DurableToolReceiptIntegrityError,
        match="one tool call cannot own multiple durable receipts",
    ):
        durable_tool_receipt_state_update(applied, second)


def test_reused_provider_call_id_is_isolated_by_logical_execution() -> None:
    first = build_durable_tool_receipt(
        source="tests.reused_call_id",
        receipt_id="receipt:first-logical-task",
        execution_id="session-reused-call-id",
        logical_execution_id="logical:first",
        created_by_attempt_id="attempt:first",
        tool_call_id="call-local-provider-reused",
        request_digest=hashlib.sha256(b"first-request").hexdigest(),
        result_digest=hashlib.sha256(b"first-result").hexdigest(),
    )
    second = build_durable_tool_receipt(
        source="tests.reused_call_id",
        receipt_id="receipt:second-logical-task",
        execution_id="session-reused-call-id",
        logical_execution_id="logical:second",
        created_by_attempt_id="attempt:second",
        tool_call_id="call-local-provider-reused",
        request_digest=hashlib.sha256(b"second-request").hexdigest(),
        result_digest=hashlib.sha256(b"second-result").hexdigest(),
    )

    first_ledger = durable_tool_receipt_state_update(None, first)[
        DURABLE_TOOL_RECEIPT_LEDGER_KEY
    ]
    merged = durable_tool_receipt_state_update(first_ledger, second)[
        DURABLE_TOOL_RECEIPT_LEDGER_KEY
    ]

    assert len(merged["pending"]) == 2


def test_cross_session_receipt_is_rejected_by_checkpoint_and_semantic_commit() -> None:
    store = _RevisionStore()
    runtime = KernelMemoryRuntime.from_config(store=store)
    state = RunState()
    state.seed_messages([{"role": "user", "content": "session-b"}])
    state.session_state.session_id = "session-b"
    state.provider_state.provider = "openai"
    state.provider_state.model = "gpt-5"
    state.memory_state["session_revision"] = 0
    checkpoint = build_execution_checkpoint(
        state,
        status="tool_result_ready",
        run_id="attempt:session-b",
    )
    ledger = durable_tool_receipt_state_update(
        None,
        _receipt(execution_id="session-a"),
    )[DURABLE_TOOL_RECEIPT_LEDGER_KEY]

    with pytest.raises(
        DurableToolReceiptIntegrityError,
        match="different execution_id",
    ):
        runtime.save_execution_checkpoint_snapshot(
            "session-b",
            checkpoint,
            expected_revision=0,
            durable_receipt_ledger=ledger,
        )

    with pytest.raises(
        DurableToolReceiptIntegrityError,
        match="different execution_id",
    ):
        runtime.commit_transcript(
            session_id="session-b",
            transcript=state.transcript,
            memory_namespace=None,
            model="gpt-5",
            summary_text="",
            expected_revision=0,
            durable_receipt_ledger=ledger,
        )
    assert store.load("session-b") == {}


def test_copied_session_tombstone_cannot_ack_another_execution() -> None:
    ledger = durable_tool_receipt_state_update(
        None,
        _receipt(execution_id="session-a"),
    )[DURABLE_TOOL_RECEIPT_LEDGER_KEY]
    copied_session_state: dict = {}
    apply_durable_tool_receipt_ledger(
        copied_session_state,
        ledger=ledger,
        application_id="checkpoint-session-a",
    )
    plugin = _ReceiptPlugin()

    with pytest.raises(
        DurableToolReceiptIntegrityError,
        match="different execution_id",
    ):
        acknowledge_durable_tool_receipt_applications(
            [plugin],
            copied_session_state,
            execution_id="session-b",
        )

    assert plugin.ack_attempts == []


def test_checkpoint_cas_failure_never_calls_receipt_ack() -> None:
    session_id = "receipt-cas-failure"
    store = _RevisionStore()
    plugin = _ReceiptPlugin()
    model = _QueueModelIO([_tool_turn(), _final_turn()])
    loop = _build_loop(store=store, plugin=plugin, model=model)

    def fail_checkpoint_after_tool(event):
        if event.get("type") == "tool_result":
            store.fail_next_cas = True

    with pytest.raises(SessionRevisionConflictError):
        _run(
            loop,
            session_id=session_id,
            plugin=plugin,
            callback=fail_checkpoint_after_tool,
        )

    assert plugin.execute_calls == 1
    assert plugin.ack_attempts == []
    assert len(model.requests) == 1
    assert TOOL_RECEIPT_JOURNAL_KEY not in store.load(session_id)


def test_checkpoint_commit_then_ack_crash_cold_bootstrap_does_not_repeat_result() -> None:
    session_id = "receipt-checkpoint-crash"
    store = _RevisionStore()
    crashing_plugin = _ReceiptPlugin(crash_on_ack=True)
    first_model = _QueueModelIO([_tool_turn(), _final_turn("must-not-run")])
    first_loop = _build_loop(
        store=store,
        plugin=crashing_plugin,
        model=first_model,
    )

    with pytest.raises(DurableToolReceiptAckError, match="crash after session commit"):
        _run(first_loop, session_id=session_id, plugin=crashing_plugin)

    persisted = store.load(session_id)
    assert persisted["execution_checkpoint"]["status"] == "tool_result_ready"
    assert TOOL_RECEIPT_JOURNAL_KEY in persisted
    assert len(first_model.requests) == 1

    recovering_plugin = _ReceiptPlugin()

    def assert_checkpoint_ack_before_model() -> None:
        assert len(recovering_plugin.ack_attempts) == 1

    recovered_model = _QueueModelIO(
        [_final_turn("recovered")],
        before_fetch=assert_checkpoint_ack_before_model,
    )
    recovered_loop = _build_loop(
        store=store,
        plugin=recovering_plugin,
        model=recovered_model,
    )
    result = _run(
        recovered_loop,
        session_id=session_id,
        plugin=recovering_plugin,
        messages=[],
    )

    assert result.status == "completed"
    assert recovering_plugin.execute_calls == 0
    assert len(recovering_plugin.acknowledged) == 1
    assert sum(
        message.get("type") == "function_call_output"
        for message in result.messages
    ) == 1
    final_state = store.load(session_id)
    assert "execution_checkpoint" not in final_state
    assert TOOL_RECEIPT_JOURNAL_KEY in final_state


def test_max_iterations_exit_persists_application_before_run_end_ack() -> None:
    session_id = "receipt-max-exit"
    store = _RevisionStore()
    plugin = _ReceiptPlugin()
    model = _QueueModelIO([_tool_turn()])
    loop = _build_loop(store=store, plugin=plugin, model=model)

    result = _run(
        loop,
        session_id=session_id,
        plugin=plugin,
        max_iterations=1,
    )

    assert result.status == "max_iterations"
    assert plugin.ack_attempts == []
    persisted = store.load(session_id)
    assert persisted["execution_checkpoint"]["status"] == "max_iterations"
    acknowledge_durable_tool_receipt_applications(
        [plugin],
        persisted,
        execution_id=session_id,
    )
    assert len(plugin.acknowledged) == 1


def test_semantic_commit_tombstone_cold_bootstrap_acks_without_duplicate_result() -> None:
    session_id = "receipt-semantic-crash"
    store = _RevisionStore()
    first_plugin = _ReceiptPlugin()
    first_model = _QueueModelIO([_tool_turn()])
    complete_after_tool = _CompleteAfterToolHarness(
        name="tests_complete_after_tool",
        phases=("after_tool_batch",),
        order=200,
    )
    first_loop = _build_loop(
        store=store,
        plugin=first_plugin,
        model=first_model,
        extra_harnesses=[complete_after_tool],
    )

    first_result = _run(first_loop, session_id=session_id, plugin=first_plugin)
    assert first_result.status == "completed"
    assert first_plugin.ack_attempts == []
    committed = store.load(session_id)
    assert "execution_checkpoint" not in committed
    assert TOOL_RECEIPT_JOURNAL_KEY in committed

    recovering_plugin = _ReceiptPlugin()

    def assert_semantic_ack_before_model() -> None:
        assert len(recovering_plugin.ack_attempts) == 1

    recovered_model = _QueueModelIO(
        [_final_turn("continued")],
        before_fetch=assert_semantic_ack_before_model,
    )
    recovered_loop = _build_loop(
        store=store,
        plugin=recovering_plugin,
        model=recovered_model,
    )
    recovered = _run(
        recovered_loop,
        session_id=session_id,
        plugin=recovering_plugin,
        messages=[],
    )

    assert recovered.status == "completed"
    assert recovering_plugin.execute_calls == 0
    assert sum(
        message.get("type") == "function_call_output"
        for message in recovered.messages
    ) == 1
    final_state = store.load(session_id)
    assert TOOL_RECEIPT_JOURNAL_KEY in final_state


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_nonstandard_finalize_exit_persists_tool_result_ready_fallback(
    status: str,
) -> None:
    session_id = f"receipt-{status}-exit"
    store = _RevisionStore()
    runtime = KernelMemoryRuntime.from_config(store=store)
    loop = build_runtime_loop(
        model_io=_QueueModelIO([]),
        memory_runtime=runtime,
    )
    state = loop.seed_state(
        [
            {"role": "user", "content": "run"},
            {"type": "function_call_output", "call_id": "call-exit", "output": "ok"},
        ],
        provider="openai",
        model="gpt-5",
        session_id=session_id,
    )
    loop.dispatch_phase(
        state,
        phase="bootstrap",
        event={"resume_mode": False, "toolkit": _toolkit(), "run_id": "attempt:exit"},
    )
    state.metadata.update(
        durable_tool_receipt_state_update(
            state,
            _receipt(execution_id=session_id, call_id="call-exit"),
        )
    )
    state.run_status = status

    loop.dispatch_phase(
        state,
        phase="finalize_persist",
        event={"status": status, "run_id": "attempt:exit"},
    )

    persisted = store.load(session_id)
    assert persisted["execution_checkpoint"]["status"] == "tool_result_ready"
    assert TOOL_RECEIPT_JOURNAL_KEY in persisted
