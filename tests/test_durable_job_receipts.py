from __future__ import annotations

import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from unchain.jobs import (
    DurableJobConflictError,
    DurableJobLineage,
    DurableJobStoreCorruptionError,
    JsonFileJobDeliveryJournal,
    JsonFileJobLineageStore,
    JsonFileJobStore,
    ProcessJobSupervisor,
)
from unchain.jobs.plugin import DurableShellJobPlugin
from unchain.jobs._worker import run_worker
from unchain.kernel.state import RunState
from unchain.kernel.types import ToolCall
from unchain.toolkits import CoreToolkit
from unchain.tools.receipts import (
    DURABLE_TOOL_RECEIPT_LEDGER_KEY,
    apply_durable_tool_receipt_ledger,
    validate_durable_tool_receipt_ledger,
)


EXECUTION_ID = "session-job-receipts"
LOGICAL_EXECUTION_ID = "logical-job-receipts"
CREATED_BY_ATTEMPT_ID = "attempt-created"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reserve_job(
    store: JsonFileJobStore,
    tmp_path: Path,
    *,
    key: str,
    status: str = "completed",
    stdout: str = "你好世界\n",
    stderr: str = "错误\n",
    logical_execution_id: str | None = LOGICAL_EXECUTION_ID,
    created_by_attempt_id: str = CREATED_BY_ATTEMPT_ID,
) -> str:
    snapshot, _ = store.reserve_process_job(
        execution_id=EXECUTION_ID,
        adapter="shell",
        argv=[sys.executable, "-c", "pass"],
        cwd=str(tmp_path.resolve()),
        timeout_ms=1_000,
        idempotency_key=key,
        intent_digest=_digest(f"intent:{key}"),
        max_log_bytes=1_000_000,
    )
    if logical_execution_id is not None:
        JsonFileJobLineageStore(store).bind(
            snapshot.job_id,
            execution_id=EXECUTION_ID,
            logical_execution_id=logical_execution_id,
            created_by_attempt_id=created_by_attempt_id,
        )
    paths = store.paths_for_worker(snapshot.job_id)
    paths["stdout"].write_bytes(stdout.encode("utf-8"))
    paths["stderr"].write_bytes(stderr.encode("utf-8"))
    token = store.load_spec(snapshot.job_id)["worker_token"]
    if status in {"completed", "timed_out"}:
        store.update_state(snapshot.job_id, worker_token=token, status="starting")
        store.update_state(snapshot.job_id, worker_token=token, status="running")
    updates: dict[str, Any] = {}
    if status == "completed":
        updates["returncode"] = 0
    elif status == "failed":
        updates.update(returncode=17, error="command failed")
    elif status == "timed_out":
        updates.update(returncode=-9, timed_out=True, error="timed out")
    elif status == "cancelled":
        updates.update(returncode=-15, cancelled=True, error="cancelled")
    elif status == "outcome_unknown":
        updates.update(outcome_unknown_reason="worker receipt missing")
    else:
        raise AssertionError(f"unsupported test status: {status}")
    store.update_state(
        snapshot.job_id,
        worker_token=token,
        status=status,
        **updates,
    )
    return snapshot.job_id


def _prepare(
    journal: JsonFileJobDeliveryJournal,
    job_id: str,
    *,
    delivery_key: str,
    max_output_chars: int = 20_000,
    created_by_attempt_id: str = "attempt-resumed",
):
    return journal.prepare(
        job_id,
        execution_id=EXECUTION_ID,
        delivery_key=delivery_key,
        action="poll",
        operation_digest=_digest(f"operation:{delivery_key}"),
        logical_execution_id=LOGICAL_EXECUTION_ID,
        created_by_attempt_id=created_by_attempt_id,
        source_attempt_id="attempt-resumed",
        max_output_chars=max_output_chars,
    )


def _plugin_context(
    toolkit: CoreToolkit,
    *,
    revision_supported: bool,
    run_id: str = "attempt-resumed",
    logical_execution_id: str = LOGICAL_EXECUTION_ID,
) -> SimpleNamespace:
    event = {
        "on_tool_confirm": lambda request: True,
        "tool_runtime_config": {
            "durable_jobs": {
                "session_id": EXECUTION_ID,
                "logical_execution_id": logical_execution_id,
                "created_by_attempt_id": run_id,
            }
        },
    }
    state = RunState()
    state.memory_state["session_revision_supported"] = revision_supported
    return SimpleNamespace(
        toolkit=toolkit,
        state=state,
        session_id=EXECUTION_ID,
        run_id=run_id,
        provider="openai",
        model="gpt-5",
        iteration=1,
        memory_namespace="",
        event=event,
        raw_event=event,
        execution_guard=None,
        loop=None,
        callback=None,
    )


def test_prepare_ack_is_replayable_and_advances_utf8_cursor_once(tmp_path) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="utf8-cursor")
    journal = JsonFileJobDeliveryJournal(store)

    first = _prepare(
        journal,
        job_id,
        delivery_key="shell:call-poll-1",
        max_output_chars=2,
    )
    replay = _prepare(
        journal,
        job_id,
        delivery_key="shell:call-poll-1",
        max_output_chars=2,
    )
    assert replay == first
    assert first.result["stdout"] == "你好"
    assert first.result["stdout_offset"] == 0
    assert first.result["next_stdout_offset"] == len("你好".encode("utf-8"))
    assert first.created_by_attempt_id == CREATED_BY_ATTEMPT_ID
    assert first.source_attempt_id == "attempt-resumed"

    paths = store.paths_for_worker(job_id)
    legacy_cursor = json.loads(paths["cursor"].read_text(encoding="utf-8"))
    assert legacy_cursor["stdout_offset"] == 0
    with pytest.raises(DurableJobConflictError, match="pending result receipt"):
        store.consume_logs(
            job_id,
            execution_id=EXECUTION_ID,
            max_output_chars=100,
            final=True,
        )
    with pytest.raises(DurableJobConflictError, match="identity does not match"):
        journal.acknowledge(
            job_id,
            execution_id=EXECUTION_ID,
            receipt_id=first.receipt_id,
            applied_checkpoint_id="checkpoint-1",
            expected_result_digest="f" * 64,
        )

    application = journal.acknowledge(
        job_id,
        execution_id=EXECUTION_ID,
        receipt_id=first.receipt_id,
        applied_checkpoint_id="checkpoint-1",
    )
    assert (
        journal.acknowledge(
            job_id,
            execution_id=EXECUTION_ID,
            receipt_id=first.receipt_id,
            applied_checkpoint_id="checkpoint-1",
        )
        == application
    )
    with pytest.raises(DurableJobConflictError, match="different checkpoint"):
        journal.acknowledge(
            job_id,
            execution_id=EXECUTION_ID,
            receipt_id=first.receipt_id,
            applied_checkpoint_id="checkpoint-other",
        )

    second = _prepare(
        journal,
        job_id,
        delivery_key="shell:call-poll-2",
        max_output_chars=100,
    )
    assert second.result["stdout_offset"] == first.result["next_stdout_offset"]
    assert first.result["stdout"] + second.result["stdout"] == "你好世界\n"
    assert first.result["stderr"] + second.result["stderr"] == "错误\n"


def test_ack_retry_after_uncertain_atomic_write_is_idempotent(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="ack-uncertain")
    journal = JsonFileJobDeliveryJournal(store)
    receipt = _prepare(journal, job_id, delivery_key="shell:call-ack")
    original_write = store._write_json_atomic
    raised = False

    def write_then_raise(path, payload, *, durable=True):
        nonlocal raised
        result = original_write(path, payload, durable=durable)
        if path.name == "delivery-journal.json" and not raised:
            raised = True
            raise RuntimeError("caller lost ack response")
        return result

    monkeypatch.setattr(store, "_write_json_atomic", write_then_raise)
    with pytest.raises(RuntimeError, match="lost ack response"):
        journal.acknowledge(
            job_id,
            execution_id=EXECUTION_ID,
            receipt_id=receipt.receipt_id,
            applied_checkpoint_id="checkpoint-ack",
        )
    monkeypatch.setattr(store, "_write_json_atomic", original_write)

    application = journal.acknowledge(
        job_id,
        execution_id=EXECUTION_ID,
        receipt_id=receipt.receipt_id,
        applied_checkpoint_id="checkpoint-ack",
    )
    assert application.receipt_id == receipt.receipt_id
    follow_up = _prepare(journal, job_id, delivery_key="shell:call-after-ack")
    assert follow_up.result["stdout_offset"] == receipt.result["next_stdout_offset"]


@pytest.mark.parametrize(
    ("status", "completed", "ok", "timed_out", "cancelled", "unknown"),
    [
        ("completed", True, True, False, False, False),
        ("failed", True, False, False, False, False),
        ("timed_out", True, False, True, False, False),
        ("cancelled", True, False, False, True, False),
        ("outcome_unknown", True, False, False, False, True),
    ],
)
def test_receipts_preserve_every_terminal_outcome(
    tmp_path,
    status,
    completed,
    ok,
    timed_out,
    cancelled,
    unknown,
) -> None:
    store = JsonFileJobStore(tmp_path / status)
    job_id = _reserve_job(store, tmp_path, key=status, status=status)
    receipt = _prepare(
        JsonFileJobDeliveryJournal(store),
        job_id,
        delivery_key=f"shell:call-{status}",
    )
    assert receipt.result["status"] == status
    assert receipt.result["completed"] is completed
    assert receipt.result["ok"] is ok
    assert receipt.result["timed_out"] is timed_out
    assert receipt.result["cancelled"] is cancelled
    assert receipt.result["outcome_unknown"] is unknown


def test_prepare_recovers_frozen_orphan_after_journal_write_failure(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="orphan-recovery", stdout="before\n")
    journal = JsonFileJobDeliveryJournal(store)
    original_write = store._write_json_atomic

    def fail_journal(path, payload, *, durable=True):
        if path.name == "delivery-journal.json":
            raise RuntimeError("fault after immutable receipt")
        return original_write(path, payload, durable=durable)

    monkeypatch.setattr(store, "_write_json_atomic", fail_journal)
    with pytest.raises(RuntimeError, match="fault after immutable receipt"):
        _prepare(journal, job_id, delivery_key="shell:call-orphan")

    paths = store.paths_for_worker(job_id)
    receipt_paths = list((paths["dir"] / "delivery-receipts").glob("*.json"))
    assert len(receipt_paths) == 1
    frozen = json.loads(receipt_paths[0].read_text(encoding="utf-8"))
    with paths["stdout"].open("ab") as handle:
        handle.write("after\n".encode("utf-8"))

    with pytest.raises(
        DurableJobConflictError,
        match="unregistered immutable.*blocks legacy",
    ):
        store.consume_logs(
            job_id,
            execution_id=EXECUTION_ID,
            max_output_chars=20_000,
            final=True,
        )

    monkeypatch.setattr(store, "_write_json_atomic", original_write)
    recovered = _prepare(journal, job_id, delivery_key="shell:call-orphan")
    assert recovered.to_dict() == frozen
    assert recovered.result["stdout"] == "before\n"
    assert "after" not in recovered.result["stdout"]


def test_orphan_a_blocks_delivery_b_until_a_is_recovered(tmp_path, monkeypatch) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="orphan-reservation")
    journal = JsonFileJobDeliveryJournal(store)
    original_write = store._write_json_atomic

    def fail_journal(path, payload, *, durable=True):
        if path.name == "delivery-journal.json":
            raise RuntimeError("journal publish failed")
        return original_write(path, payload, durable=durable)

    monkeypatch.setattr(store, "_write_json_atomic", fail_journal)
    with pytest.raises(RuntimeError, match="journal publish failed"):
        _prepare(journal, job_id, delivery_key="shell:call-a")
    monkeypatch.setattr(store, "_write_json_atomic", original_write)

    with pytest.raises(DurableJobConflictError, match="orphan.*reserves"):
        _prepare(journal, job_id, delivery_key="shell:call-b")
    recovered = _prepare(journal, job_id, delivery_key="shell:call-a")
    assert recovered.delivery_key == "shell:call-a"


def test_orphan_after_an_acknowledged_receipt_blocks_legacy_cursor(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="orphan-after-ack", stdout="first\n")
    journal = JsonFileJobDeliveryJournal(store)
    first = _prepare(journal, job_id, delivery_key="shell:call-first")
    journal.acknowledge(
        job_id,
        execution_id=EXECUTION_ID,
        receipt_id=first.receipt_id,
        applied_checkpoint_id="checkpoint-first",
    )
    paths = store.paths_for_worker(job_id)
    with paths["stdout"].open("ab") as handle:
        handle.write("second\n".encode("utf-8"))

    original_write = store._write_json_atomic

    def fail_journal(path, payload, *, durable=True):
        if path.name == "delivery-journal.json":
            raise RuntimeError("journal publish failed")
        return original_write(path, payload, durable=durable)

    monkeypatch.setattr(store, "_write_json_atomic", fail_journal)
    with pytest.raises(RuntimeError, match="journal publish failed"):
        _prepare(journal, job_id, delivery_key="shell:call-second")
    monkeypatch.setattr(store, "_write_json_atomic", original_write)

    with pytest.raises(
        DurableJobConflictError,
        match="unregistered immutable.*blocks legacy",
    ):
        store.consume_logs(
            job_id,
            execution_id=EXECUTION_ID,
            max_output_chars=20_000,
            final=True,
        )
    recovered = _prepare(journal, job_id, delivery_key="shell:call-second")
    assert recovered.result["stdout"] == "second\n"


@pytest.mark.parametrize(
    ("action", "delivery_key", "operation_digest"),
    [
        ("wait", "shell:call-wait-orphan", _digest("wait-orphan")),
        ("cancel", "shell:call-cancel-orphan", _digest("cancel-orphan")),
    ],
)
def test_supervisor_replays_orphan_before_wait_or_cancel_side_effects(
    tmp_path,
    monkeypatch,
    action,
    delivery_key,
    operation_digest,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key=f"{action}-orphan")
    supervisor = ProcessJobSupervisor(store)
    original_write = store._write_json_atomic

    def fail_journal(path, payload, *, durable=True):
        if path.name == "delivery-journal.json":
            raise RuntimeError("journal publish failed")
        return original_write(path, payload, durable=durable)

    call = (
        supervisor.wait_with_receipt
        if action == "wait"
        else supervisor.cancel_with_receipt
    )
    kwargs = {
        "execution_id": EXECUTION_ID,
        "delivery_key": delivery_key,
        "operation_digest": operation_digest,
        "logical_execution_id": LOGICAL_EXECUTION_ID,
        "created_by_attempt_id": "attempt-resumed",
        "source_attempt_id": "attempt-resumed",
        "max_output_chars": 20_000,
    }
    if action == "wait":
        kwargs["timeout_ms"] = 120_000
    else:
        kwargs["wait_timeout_ms"] = 2_000

    try:
        monkeypatch.setattr(store, "_write_json_atomic", fail_journal)
        with pytest.raises(RuntimeError, match="journal publish failed"):
            call(job_id, **kwargs)
        monkeypatch.setattr(store, "_write_json_atomic", original_write)

        def unexpected_post_lookup(*args, **kwargs):
            raise AssertionError("orphan replay must return before job side effects")

        if action == "wait":
            monkeypatch.setattr(supervisor, "inspect", unexpected_post_lookup)
        else:
            monkeypatch.setattr(store, "load", unexpected_post_lookup)
        recovered = call(job_id, **kwargs)
        assert recovered.delivery_key == delivery_key
        assert recovered.action == action
    finally:
        supervisor.close()


def test_cancel_orphan_blocks_a_different_key_before_job_side_effects(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="cancel-orphan-conflict")
    supervisor = ProcessJobSupervisor(store)
    original_write = store._write_json_atomic

    def fail_journal(path, payload, *, durable=True):
        if path.name == "delivery-journal.json":
            raise RuntimeError("journal publish failed")
        return original_write(path, payload, durable=durable)

    common = {
        "execution_id": EXECUTION_ID,
        "logical_execution_id": LOGICAL_EXECUTION_ID,
        "created_by_attempt_id": "attempt-resumed",
        "source_attempt_id": "attempt-resumed",
        "wait_timeout_ms": 0,
        "max_output_chars": 20_000,
    }
    try:
        monkeypatch.setattr(store, "_write_json_atomic", fail_journal)
        with pytest.raises(RuntimeError, match="journal publish failed"):
            supervisor.cancel_with_receipt(
                job_id,
                delivery_key="shell:call-cancel-a",
                operation_digest=_digest("cancel-a"),
                **common,
            )
        monkeypatch.setattr(store, "_write_json_atomic", original_write)

        def unexpected_load(*args, **kwargs):
            raise AssertionError("orphan conflict must precede cancellation")

        monkeypatch.setattr(store, "load", unexpected_load)
        with pytest.raises(DurableJobConflictError, match="orphan.*reserves"):
            supervisor.cancel_with_receipt(
                job_id,
                delivery_key="shell:call-cancel-b",
                operation_digest=_digest("cancel-b"),
                **common,
            )
    finally:
        supervisor.close()


def test_concurrent_same_delivery_key_returns_one_receipt(tmp_path) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="concurrent")
    journal = JsonFileJobDeliveryJournal(store)

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(
            pool.map(
                lambda _: _prepare(
                    journal,
                    job_id,
                    delivery_key="shell:call-concurrent",
                ),
                range(4),
            )
        )
    assert len({receipt.receipt_id for receipt in receipts}) == 1
    paths = store.paths_for_worker(job_id)
    assert len(list((paths["dir"] / "delivery-receipts").glob("*.json"))) == 1


def test_lineage_is_exact_and_cancel_alias_never_selects_legacy_jobs(tmp_path) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    selected = _reserve_job(store, tmp_path, key="selected", status="cancelled")
    _reserve_job(
        store,
        tmp_path,
        key="other-logical",
        status="cancelled",
        logical_execution_id="logical-other",
    )
    _reserve_job(
        store,
        tmp_path,
        key="legacy-unscoped",
        status="cancelled",
        logical_execution_id=None,
    )
    lineage = JsonFileJobLineageStore(store)
    assert (
        lineage.bind(
            selected,
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
            created_by_attempt_id=CREATED_BY_ATTEMPT_ID,
        ).job_id
        == selected
    )
    with pytest.raises(DurableJobConflictError, match="different owner"):
        lineage.bind(
            selected,
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
            created_by_attempt_id="different-creation-attempt",
        )

    supervisor = ProcessJobSupervisor(store)
    try:
        outcome = supervisor.cancel_logical_execution(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
        )
    finally:
        supervisor.close()
    assert outcome["supported"] is True
    assert outcome["disposition"] == "applied"
    assert outcome["cancelled_job_ids"] == [selected]
    assert [job["job_id"] for job in outcome["jobs"]] == [selected]


def test_logical_stop_marks_outcome_unknown_without_touching_other_owners(
    tmp_path,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    selected = _reserve_job(
        store,
        tmp_path,
        key="unknown-selected",
        status="outcome_unknown",
    )
    other = _reserve_job(
        store,
        tmp_path,
        key="unknown-other",
        status="outcome_unknown",
        logical_execution_id="logical-other",
    )
    legacy = _reserve_job(
        store,
        tmp_path,
        key="unknown-legacy",
        status="outcome_unknown",
        logical_execution_id=None,
    )
    supervisor = ProcessJobSupervisor(store)
    try:
        outcome = supervisor.cancel_logical_execution(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
            wait_timeout_ms=0,
        )
    finally:
        supervisor.close()
    assert outcome["cancelled_job_ids"] == [selected]
    assert store.cancel_requested(selected) is True
    assert store.cancel_requested(other) is False
    assert store.cancel_requested(legacy) is False
    assert store.load(
        selected,
        execution_id=EXECUTION_ID,
    ).status == "outcome_unknown"


@pytest.mark.parametrize("receipt_aware", [False, True])
def test_direct_cancel_marks_outcome_unknown_for_cooperative_worker_stop(
    tmp_path,
    receipt_aware,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(
        store,
        tmp_path,
        key=f"unknown-direct-{receipt_aware}",
        status="outcome_unknown",
    )
    supervisor = ProcessJobSupervisor(store)
    try:
        if receipt_aware:
            receipt = supervisor.cancel_with_receipt(
                job_id,
                execution_id=EXECUTION_ID,
                delivery_key="shell:call-cancel-unknown",
                operation_digest=_digest("cancel-unknown"),
                logical_execution_id=LOGICAL_EXECUTION_ID,
                created_by_attempt_id="attempt-resumed",
                source_attempt_id="attempt-resumed",
                wait_timeout_ms=0,
            )
            assert receipt.result["cancel_requested"] is True
        else:
            result = supervisor.cancel(
                job_id,
                execution_id=EXECUTION_ID,
                wait_timeout_ms=0,
            )
            assert result["cancel_requested"] is True
    finally:
        supervisor.close()
    assert store.cancel_requested(job_id) is True


def test_logical_stop_tombstone_blocks_a_late_or_replayed_start(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    try:
        outcome = supervisor.cancel_logical_execution(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
            wait_timeout_ms=0,
        )
        assert outcome["cancelled_job_ids"] == []
        cancellation = supervisor.lineage.cancellation(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
        )
        assert cancellation is not None
        assert cancellation.reason == "user_stop"

        with pytest.raises(DurableJobConflictError, match="has been cancelled"):
            supervisor.start(
                execution_id=EXECUTION_ID,
                idempotency_key="late-after-stop",
                argv=[sys.executable, "-c", "pass"],
                cwd=str(tmp_path.resolve()),
                timeout_ms=1_000,
                intent_digest=_digest("late-after-stop"),
                adapter="shell",
                logical_execution_id=LOGICAL_EXECUTION_ID,
                created_by_attempt_id="attempt-late",
            )
    finally:
        supervisor.close()
    assert store.list_execution(EXECUTION_ID) == []


def test_start_and_logical_stop_share_one_scope_barrier(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    reserved = threading.Event()
    release_start = threading.Event()
    stop_returned = threading.Event()
    original_reserve = store.reserve_process_job

    def reserve_then_pause(**kwargs):
        result = original_reserve(**kwargs)
        reserved.set()
        assert release_start.wait(timeout=5)
        return result

    monkeypatch.setattr(store, "reserve_process_job", reserve_then_pause)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    start_result = {}
    stop_result = {}

    def start_job():
        start_result["snapshot"] = supervisor.start(
            execution_id=EXECUTION_ID,
            idempotency_key="barrier-race",
            argv=[sys.executable, "-c", "pass"],
            cwd=str(tmp_path.resolve()),
            timeout_ms=1_000,
            intent_digest=_digest("barrier-race"),
            adapter="shell",
            logical_execution_id=LOGICAL_EXECUTION_ID,
            created_by_attempt_id=CREATED_BY_ATTEMPT_ID,
        )

    def stop_logical():
        stop_result["outcome"] = supervisor.cancel_logical_execution(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
            wait_timeout_ms=0,
        )
        stop_returned.set()

    start_thread = threading.Thread(target=start_job)
    stop_thread = threading.Thread(target=stop_logical)
    try:
        start_thread.start()
        assert reserved.wait(timeout=5)
        stop_thread.start()
        assert stop_returned.wait(timeout=0.05) is False
        release_start.set()
        start_thread.join(timeout=5)
        stop_thread.join(timeout=5)
        assert not start_thread.is_alive()
        assert not stop_thread.is_alive()
    finally:
        release_start.set()
        start_thread.join(timeout=5)
        stop_thread.join(timeout=5)
        supervisor.close()

    job_id = start_result["snapshot"].job_id
    assert stop_result["outcome"]["cancelled_job_ids"] == [job_id]
    assert store.cancel_requested(job_id) is True


def test_crash_after_reservation_cannot_create_an_unscoped_ghost_job(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    original_reserve = store.reserve_process_job

    def reserve_then_crash(**kwargs):
        original_reserve(**kwargs)
        raise RuntimeError("crash after immutable reservation")

    monkeypatch.setattr(store, "reserve_process_job", reserve_then_crash)
    with pytest.raises(RuntimeError, match="crash after immutable reservation"):
        supervisor.start(
            execution_id=EXECUTION_ID,
            idempotency_key="crash-after-reserve",
            argv=[sys.executable, "-c", "pass"],
            cwd=str(tmp_path.resolve()),
            timeout_ms=1_000,
            intent_digest=_digest("crash-after-reserve"),
            adapter="shell",
            logical_execution_id=LOGICAL_EXECUTION_ID,
            created_by_attempt_id=CREATED_BY_ATTEMPT_ID,
        )
    supervisor.close()

    recovered = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        recovered,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    try:
        scoped = recovered.reattach_lineage(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
        )
        assert len(scoped) == 1
        job_id = scoped[0].job_id
        outcome = recovered.cancel_logical_execution(
            execution_id=EXECUTION_ID,
            logical_execution_id=LOGICAL_EXECUTION_ID,
            wait_timeout_ms=0,
        )
    finally:
        recovered.close()
    assert outcome["cancelled_job_ids"] == [job_id]
    assert store.cancel_requested(job_id) is True


def test_scoped_reservation_cannot_retroactively_claim_a_legacy_job(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    key = "legacy-reservation"
    intent = _digest("legacy-reservation")
    legacy, _ = store.reserve_process_job(
        execution_id=EXECUTION_ID,
        adapter="shell",
        argv=[sys.executable, "-c", "pass"],
        cwd=str(tmp_path.resolve()),
        timeout_ms=1_000,
        idempotency_key=key,
        intent_digest=intent,
        max_log_bytes=supervisor.default_max_log_bytes,
        environment_digest=supervisor.environment_profile.digest,
    )
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    try:
        with pytest.raises(DurableJobConflictError, match="unscoped.*cannot be claimed"):
            supervisor.start(
                execution_id=EXECUTION_ID,
                idempotency_key=key,
                argv=[sys.executable, "-c", "pass"],
                cwd=str(tmp_path.resolve()),
                timeout_ms=1_000,
                intent_digest=intent,
                adapter="shell",
                logical_execution_id=LOGICAL_EXECUTION_ID,
                created_by_attempt_id=CREATED_BY_ATTEMPT_ID,
            )
    finally:
        supervisor.close()
    assert JsonFileJobLineageStore(store).load(
        legacy.job_id,
        execution_id=EXECUTION_ID,
    ) is None


def test_unscoped_start_preserves_legacy_store_call_surface(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    original_reserve = store.reserve_process_job

    def legacy_reserve(**kwargs):
        assert "_reservation_guard" not in kwargs
        return original_reserve(**kwargs)

    monkeypatch.setattr(store, "reserve_process_job", legacy_reserve)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    try:
        snapshot = supervisor.start(
            execution_id=EXECUTION_ID,
            idempotency_key="legacy-call-surface",
            argv=[sys.executable, "-c", "pass"],
            cwd=str(tmp_path.resolve()),
            timeout_ms=1_000,
            intent_digest=_digest("legacy-call-surface"),
            adapter="shell",
        )
    finally:
        supervisor.close()
    assert snapshot.status == "queued"


def test_lineage_request_digest_must_match_owning_job_spec(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    try:
        snapshot = supervisor.start(
            execution_id=EXECUTION_ID,
            idempotency_key="lineage-request-digest",
            argv=[sys.executable, "-c", "pass"],
            cwd=str(tmp_path.resolve()),
            timeout_ms=1_000,
            intent_digest=_digest("lineage-request-digest"),
            adapter="shell",
            logical_execution_id=LOGICAL_EXECUTION_ID,
            created_by_attempt_id=CREATED_BY_ATTEMPT_ID,
        )
        lineage = supervisor.lineage.load(
            snapshot.job_id,
            execution_id=EXECUTION_ID,
        )
        assert lineage is not None
        tampered = DurableJobLineage.build(
            store_id=lineage.store_id,
            job_id=lineage.job_id,
            execution_id=lineage.execution_id,
            logical_execution_id=lineage.logical_execution_id,
            created_by_attempt_id=lineage.created_by_attempt_id,
            request_digest=_digest("different-job-spec"),
        )
        paths = store.paths_for_worker(snapshot.job_id)
        store._write_json_atomic(
            paths["dir"] / "lineage.json",
            tampered.to_dict(),
        )

        with pytest.raises(
            DurableJobStoreCorruptionError,
            match="does not belong to its job",
        ):
            supervisor.lineage.load(
                snapshot.job_id,
                execution_id=EXECUTION_ID,
            )
    finally:
        supervisor.close()


def test_orphan_scoped_reservation_requires_matching_guard_and_intent(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    original_write = store._write_json_atomic

    def fail_spec_after_lineage(path, payload, *, durable=True):
        if path.name == "spec.json":
            raise RuntimeError("crash before spec publication")
        return original_write(path, payload, durable=durable)

    kwargs = {
        "execution_id": EXECUTION_ID,
        "idempotency_key": "orphan-scoped-reservation",
        "argv": [sys.executable, "-c", "pass"],
        "cwd": str(tmp_path.resolve()),
        "timeout_ms": 1_000,
        "intent_digest": _digest("orphan-scoped-reservation"),
        "adapter": "shell",
        "logical_execution_id": LOGICAL_EXECUTION_ID,
        "created_by_attempt_id": CREATED_BY_ATTEMPT_ID,
    }
    try:
        monkeypatch.setattr(store, "_write_json_atomic", fail_spec_after_lineage)
        with pytest.raises(RuntimeError, match="crash before spec publication"):
            supervisor.start(**kwargs)
        monkeypatch.setattr(store, "_write_json_atomic", original_write)

        with pytest.raises(DurableJobConflictError, match="requires its lineage guard"):
            store.reserve_process_job(
                execution_id=EXECUTION_ID,
                adapter="shell",
                argv=[sys.executable, "-c", "pass"],
                cwd=str(tmp_path.resolve()),
                timeout_ms=1_000,
                idempotency_key="orphan-scoped-reservation",
                intent_digest=_digest("orphan-scoped-reservation"),
                max_log_bytes=supervisor.default_max_log_bytes,
                environment_digest=supervisor.environment_profile.digest,
            )
        with pytest.raises(DurableJobConflictError, match="different owner"):
            supervisor.start(
                **{
                    **kwargs,
                    "intent_digest": _digest("different-intent"),
                }
            )

        monkeypatch.setattr(
            supervisor,
            "_reconcile",
            lambda snapshot, *, allow_launch: snapshot,
        )
        recovered = supervisor.start(**kwargs)
    finally:
        supervisor.close()
    lineage = JsonFileJobLineageStore(store).load(
        recovered.job_id,
        execution_id=EXECUTION_ID,
    )
    assert lineage is not None
    assert lineage.logical_execution_id == LOGICAL_EXECUTION_ID


def test_worker_tombstone_fence_blocks_user_popen_when_marker_write_was_lost(
    tmp_path,
    monkeypatch,
) -> None:
    marker = tmp_path / "user-command-ran"
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    snapshot = supervisor.start(
        execution_id=EXECUTION_ID,
        idempotency_key="worker-tombstone-fence",
        argv=[
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(marker)!r}).write_text('ran')",
        ],
        cwd=str(tmp_path.resolve()),
        timeout_ms=1_000,
        intent_digest=_digest("worker-tombstone-fence"),
        adapter="shell",
        logical_execution_id=LOGICAL_EXECUTION_ID,
        created_by_attempt_id=CREATED_BY_ATTEMPT_ID,
    )
    supervisor.cancel_logical_execution(
        execution_id=EXECUTION_ID,
        logical_execution_id=LOGICAL_EXECUTION_ID,
        wait_timeout_ms=0,
    )
    supervisor.close()

    paths = store.paths_for_worker(snapshot.job_id)
    paths["cancel"].unlink()
    assert store.cancel_requested(snapshot.job_id) is False
    assert run_worker(
        store_dir=store.base_dir,
        job_id=snapshot.job_id,
        worker_token=snapshot.worker_token,
        heartbeat_interval_ms=10,
        cancel_grace_ms=10,
    ) == 0
    assert marker.exists() is False
    assert store.cancel_requested(snapshot.job_id) is True
    assert store.load(
        snapshot.job_id,
        execution_id=EXECUTION_ID,
    ).status == "cancelled"


def test_replayed_start_preserves_creation_attempt_but_requires_same_logical_owner(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    kwargs = {
        "execution_id": EXECUTION_ID,
        "idempotency_key": "replayed-start",
        "argv": [sys.executable, "-c", "pass"],
        "cwd": str(tmp_path.resolve()),
        "timeout_ms": 1_000,
        "intent_digest": _digest("replayed-start"),
        "adapter": "shell",
        "logical_execution_id": LOGICAL_EXECUTION_ID,
    }
    try:
        first = supervisor.start(
            **kwargs,
            created_by_attempt_id="attempt-original",
        )
        replay = supervisor.start(
            **kwargs,
            created_by_attempt_id="attempt-resumed",
        )
        assert replay.job_id == first.job_id
        lineage = supervisor.lineage.load(
            first.job_id,
            execution_id=EXECUTION_ID,
        )
        assert lineage.created_by_attempt_id == "attempt-original"
        with pytest.raises(DurableJobConflictError, match="different owner"):
            supervisor.start(
                **{
                    **kwargs,
                    "logical_execution_id": "logical-other",
                },
                created_by_attempt_id="attempt-resumed",
            )
    finally:
        supervisor.close()


def test_plugin_keeps_receipt_metadata_out_of_model_result_and_acks_after_cas(
    tmp_path,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="plugin-bridge", stdout="visible\n")
    supervisor = ProcessJobSupervisor(store)
    toolkit = CoreToolkit(workspace_root=tmp_path)
    plugin = DurableShellJobPlugin(supervisor=supervisor)
    call = ToolCall(
        call_id="call-plugin-poll",
        name="shell",
        arguments={"action": "poll", "task_id": job_id},
    )
    try:
        context = _plugin_context(toolkit, revision_supported=True)
        outcome = plugin.execute(tool_call=call, context=context)
        assert outcome.tool_result["stdout"] == "visible\n"
        assert "receipt_id" not in outcome.tool_result
        assert "receipt_digest" not in outcome.tool_result
        assert "job_receipt" not in outcome.tool_result

        ledger = validate_durable_tool_receipt_ledger(
            outcome.state_updates[DURABLE_TOOL_RECEIPT_LEDGER_KEY]
        )
        assert len(ledger["pending"]) == 1
        assert (
            len(
                supervisor.pending_result_receipts(
                    job_id,
                    execution_id=EXECUTION_ID,
                )
            )
            == 1
        )

        session_state: dict[str, Any] = {}
        application_ledger = apply_durable_tool_receipt_ledger(
            session_state,
            ledger=ledger,
            application_id="checkpoint-plugin",
        )
        application = next(iter(application_ledger["applications"].values()))
        assert (
            plugin.acknowledge_durable_receipt(
                receipt=application["receipt"],
                application=application,
            )
            is True
        )
        assert (
            plugin.acknowledge_durable_receipt(
                receipt=application["receipt"],
                application=application,
            )
            is True
        )
        assert (
            supervisor.pending_result_receipts(
                job_id,
                execution_id=EXECUTION_ID,
            )
            == []
        )
    finally:
        supervisor.close()
        toolkit.shutdown()


def test_plugin_uses_legacy_cursor_without_revision_cas(tmp_path) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    job_id = _reserve_job(store, tmp_path, key="legacy-fallback", stdout="legacy\n")
    supervisor = ProcessJobSupervisor(store)
    toolkit = CoreToolkit(workspace_root=tmp_path)
    plugin = DurableShellJobPlugin(supervisor=supervisor)
    call = ToolCall(
        call_id="call-legacy-poll",
        name="shell",
        arguments={"action": "poll", "task_id": job_id},
    )
    try:
        outcome = plugin.execute(
            tool_call=call,
            context=_plugin_context(toolkit, revision_supported=False),
        )
        assert outcome.tool_result["stdout"] == "legacy\n"
        assert outcome.state_updates == {}
        paths = store.paths_for_worker(job_id)
        assert not (paths["dir"] / "delivery-journal.json").exists()
        assert not (paths["dir"] / "delivery-receipts").exists()
    finally:
        supervisor.close()
        toolkit.shutdown()


def test_durable_runtime_manifest_binds_logical_execution_only(tmp_path) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    toolkit = CoreToolkit(workspace_root=tmp_path)
    plugin = DurableShellJobPlugin(supervisor=supervisor)
    call = ToolCall(
        call_id="call-manifest",
        name="shell",
        arguments={
            "action": "run",
            "command": "printf manifest",
            "run_in_background": True,
        },
    )
    try:
        context = _plugin_context(
            toolkit,
            revision_supported=True,
            run_id="attempt-current",
        )
        manifest = plugin.durable_runtime_manifest(tool_call=call, context=context)
        assert manifest["logical_execution_id"] == LOGICAL_EXECUTION_ID
        assert manifest["operation"]["logical_execution_id"] == LOGICAL_EXECUTION_ID
        assert "created_by_attempt_id" not in manifest
        assert "created_by_attempt_id" not in manifest["operation"]
    finally:
        supervisor.close()
        toolkit.shutdown()


def test_same_session_and_call_id_are_isolated_by_logical_execution(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    toolkit = CoreToolkit(workspace_root=tmp_path)
    plugin = DurableShellJobPlugin(supervisor=supervisor)
    call = ToolCall(
        call_id="call-reused-by-local-provider",
        name="shell",
        arguments={
            "action": "run",
            "command": "printf isolated",
            "run_in_background": True,
        },
    )
    try:
        first = plugin.execute(
            tool_call=call,
            context=_plugin_context(
                toolkit,
                revision_supported=True,
                logical_execution_id="logical-a",
            ),
        )
        second = plugin.execute(
            tool_call=call,
            context=_plugin_context(
                toolkit,
                revision_supported=True,
                logical_execution_id="logical-b",
            ),
        )
        assert first.tool_result["status"] == "queued"
        assert second.tool_result["status"] == "queued"
        assert first.tool_result["job_id"] != second.tool_result["job_id"]
        lineages = {
            supervisor.lineage.load(
                snapshot.job_id,
                execution_id=EXECUTION_ID,
            ).logical_execution_id
            for snapshot in store.list_execution(EXECUTION_ID)
        }
        assert lineages == {"logical-a", "logical-b"}
    finally:
        supervisor.close()
        toolkit.shutdown()


def test_scoped_start_identity_is_not_ambiguous_at_colon_boundaries(
    tmp_path,
    monkeypatch,
) -> None:
    store = JsonFileJobStore(tmp_path / "jobs")
    supervisor = ProcessJobSupervisor(store)
    monkeypatch.setattr(
        supervisor,
        "_reconcile",
        lambda snapshot, *, allow_launch: snapshot,
    )
    toolkit = CoreToolkit(workspace_root=tmp_path)
    plugin = DurableShellJobPlugin(supervisor=supervisor)

    def start(*, logical_execution_id: str, call_id: str):
        return plugin.execute(
            tool_call=ToolCall(
                call_id=call_id,
                name="shell",
                arguments={
                    "action": "run",
                    "command": "printf unambiguous",
                    "run_in_background": True,
                },
            ),
            context=_plugin_context(
                toolkit,
                revision_supported=True,
                logical_execution_id=logical_execution_id,
            ),
        )

    try:
        first = start(logical_execution_id="a:b", call_id="c")
        second = start(logical_execution_id="a", call_id="b:c")
        assert first.tool_result["status"] == "queued"
        assert second.tool_result["status"] == "queued"
        assert first.tool_result["job_id"] != second.tool_result["job_id"]
    finally:
        supervisor.close()
        toolkit.shutdown()
