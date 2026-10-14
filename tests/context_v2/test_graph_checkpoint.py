from __future__ import annotations

import hashlib

import pytest

from unchain.context.artifacts import ArtifactService
from unchain.context.derived_handoff import DerivedHandoffInputIngress
from unchain.context.graph_checkpoint import (
    GraphCheckpointConflict,
    GraphCheckpointError,
    GraphCheckpointService,
    GraphExecutionPlan,
    GraphStepBinding,
    GraphStepDisposition,
    GraphTerminalStatus,
    JournalGraphCheckpointRepository,
)
from unchain.context.handoff import DurableHandoffRecorder, HandoffService
from unchain.context.ingress import ContextInputIngress, HostResolvedCurrentInput
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.journal import (
    AttemptRef,
    DurableEventSink,
    GenerationRef,
    SemanticEventDraft,
)
from unchain.persistence import SQLiteContextV2Store


GENERATION = GenerationRef("graph-execution", "graph-generation")
ORCHESTRATION = AttemptRef(GENERATION, "graph-orchestration")
STEP_0 = AttemptRef(GENERATION, "graph-step-0")
STEP_1 = AttemptRef(GENERATION, "graph-step-1")


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _projector(attempt: AttemptRef, artifacts: ArtifactService):
    return CanonicalSemanticEventProjector(
        attempt=attempt,
        artifacts=artifacts,
        payload_sanitizer=lambda _event_type, payload: payload,
    )


def _open(tmp_path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )
    journal = store.bind_execution(GENERATION.execution_id)
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, _media_type: content,
    )
    projectors = {
        attempt: _projector(attempt, artifacts)
        for attempt in (ORCHESTRATION, STEP_0, STEP_1)
    }
    sinks = {
        attempt: DurableEventSink(journal, attempt, projectors[attempt])
        for attempt in projectors
    }

    def derived_ingress(
        consumer_attempt: AttemptRef,
        source_attempt: AttemptRef,
    ) -> DerivedHandoffInputIngress:
        projector = projectors[consumer_attempt]
        sink = sinks[consumer_attempt]
        ingress = ContextInputIngress(
            attempt=consumer_attempt,
            projector=projector,
            sink=sink,
        )
        return DerivedHandoffInputIngress(
            consumer_attempt=consumer_attempt,
            source_attempt=source_attempt,
            handoff_recorder=DurableHandoffRecorder(
                attempt=consumer_attempt,
                handoffs=HandoffService(artifacts),
                projector=projector,
                sink=sink,
            ),
            input_ingress=ingress,
        )

    service = GraphCheckpointService(
        repository=JournalGraphCheckpointRepository(journal),
        artifacts=artifacts,
        derived_ingress_resolver=derived_ingress,
    )
    return store, journal, artifacts, sinks, service


def _plan(initial_input_cursor, *, topology: str = "topology-v1"):
    return GraphExecutionPlan(
        orchestration_attempt=ORCHESTRATION,
        topology_sha256=_digest(topology),
        initial_input_cursor=initial_input_cursor,
        steps=(
            GraphStepBinding(
                index=0,
                node_id="node-0",
                attempt=STEP_0,
                source_attempt=ORCHESTRATION,
                provider="openai",
                model="gpt-test",
                configuration_sha256=_digest("node-0-config"),
            ),
            GraphStepBinding(
                index=1,
                node_id="node-1",
                attempt=STEP_1,
                source_attempt=STEP_0,
                provider="anthropic",
                model="claude-test",
                configuration_sha256=_digest("node-1-config"),
            ),
        ),
    )


def _bootstrap(tmp_path, *, topology: str = "topology-v1"):
    store, journal, artifacts, sinks, service = _open(tmp_path)
    ingress = ContextInputIngress(
        attempt=ORCHESTRATION,
        projector=sinks[ORCHESTRATION].projector,
        sink=sinks[ORCHESTRATION],
    )
    initial = ingress.persist(
        HostResolvedCurrentInput(
            attempt=ORCHESTRATION,
            content="run the graph",
            message_index=0,
        )
    )
    plan = _plan(initial.cursor, topology=topology)
    service.admit(plan)
    return store, journal, artifacts, sinks, service, plan


def _append(
    sink: DurableEventSink,
    attempt: AttemptRef,
    event_type: str,
    sequence: int,
    **payload,
):
    return sink.append_projected(
        SemanticEventDraft(
            event_id=f"event-{attempt.attempt_id}-{sequence}-{event_type}",
            event_type=event_type,
            attempt=attempt,
            operation_id=f"operation-{attempt.attempt_id}-{sequence}-{event_type}",
            payload={"run_id": attempt.attempt_id, **payload},
        )
    )


def _complete_runtime(sinks, attempt: AttemptRef, *, output: str, base: int = 1):
    _append(sinks[attempt], attempt, "run_started", base, status="running")
    final = _append(
        sinks[attempt],
        attempt,
        "final_message",
        base + 1,
        content=output,
    )
    terminal = _append(
        sinks[attempt],
        attempt,
        "run_completed",
        base + 2,
        status="completed",
    )
    return final, terminal


def test_plan_requires_every_step_to_name_its_immediate_predecessor():
    cursor = type("Cursor", (), {"store_seq": 1, "event_id": "seed"})()
    from unchain.journal import EventCursor

    with pytest.raises(ValueError, match="immediate predecessor"):
        GraphExecutionPlan(
            orchestration_attempt=ORCHESTRATION,
            topology_sha256=_digest("topology"),
            initial_input_cursor=EventCursor(cursor.store_seq, cursor.event_id),
            steps=(
                GraphStepBinding(
                    index=0,
                    node_id="node-0",
                    attempt=STEP_0,
                    source_attempt=ORCHESTRATION,
                    provider="openai",
                    model="gpt-test",
                    configuration_sha256=_digest("node-0"),
                ),
                GraphStepBinding(
                    index=1,
                    node_id="node-1",
                    attempt=STEP_1,
                    source_attempt=ORCHESTRATION,
                    provider="openai",
                    model="gpt-test",
                    configuration_sha256=_digest("node-1"),
                ),
            ),
        )


def test_admit_and_start_zero_persist_handoff_input_before_step_start(tmp_path):
    _store, journal, _artifacts, _sinks, service, plan = _bootstrap(tmp_path)

    receipt = service.start_step(plan, 0)

    assert receipt.disposition is GraphStepDisposition.STARTED
    assert receipt.handoff is not None
    assert receipt.started_cursor is not None
    assert (
        receipt.handoff.handoff_cursor.store_seq
        < receipt.handoff.input_cursor.store_seq
        < receipt.started_cursor.store_seq
    )
    assert [event.event_type for event in journal.capture_snapshot().events] == [
        "message.user",
        "graph.execution.admitted",
        "handoff.recorded",
        "message.user",
        "graph.step.started",
    ]


def test_complete_zero_then_start_one_uses_immediate_predecessor_artifact(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="zero")
    completion = service.complete_step(
        plan,
        0,
        full_output={"output": "zero"},
    )

    started = service.start_step(plan, 1)

    assert started.step.source_attempt == STEP_0
    assert started.handoff is not None
    assert started.handoff.envelope.child_attempt == STEP_0
    assert started.handoff.envelope.artifact_refs == (
        completion.output_artifact.ref,
    )
    events = journal.capture_snapshot().events
    handoff_event = next(
        event
        for event in events
        if event.event_id == started.handoff.handoff_cursor.event_id
    )
    assert completion.output_artifact.ref in handoff_event.resource_refs
    assert started.started_cursor.store_seq > started.handoff.input_cursor.store_seq


def test_restart_returns_skip_for_a_completed_step(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="zero")
    completion = service.complete_step(
        plan,
        0,
        full_output={"output": "zero"},
    )

    _store, _journal, _artifacts, _sinks, restarted = _open(tmp_path)
    replay = restarted.start_step(plan, 0)

    assert replay.disposition is GraphStepDisposition.SKIP_COMPLETED
    assert replay.completion == completion
    assert replay.started_cursor is None
    assert replay.handoff is None


def test_completed_output_reader_returns_only_the_canonical_envelope(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="zero")
    service.complete_step(plan, 0, full_output={"output": "zero"})

    _store, _journal, _artifacts, _sinks, restarted = _open(tmp_path)

    assert restarted.read_completed_output(plan, 0) == {
        "schema": "unchain.graph_step_output.v1",
        "status": "completed",
        "output": "zero",
    }
    with pytest.raises(GraphCheckpointError, match="no completed output"):
        restarted.read_completed_output(plan, 1)


def test_restart_seals_canonical_terminal_left_before_graph_checkpoint(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="sealed-from-journal")
    assert "graph.step.completed" not in {
        event.event_type for event in journal.capture_snapshot().events
    }

    _store, reopened, _artifacts, _sinks, restarted = _open(tmp_path)
    recovery = restarted.recover(plan)

    assert len(recovery.completed_steps) == 1
    assert recovery.next_step_index == 1
    assert recovery.uncertain_step_index is None
    assert restarted._decode_artifact(
        recovery.completed_steps[0].output_artifact
    ) == {
        "schema": "unchain.graph_step_output.v1",
        "status": "completed",
        "output": "sealed-from-journal",
    }
    assert sum(
        event.event_type == "graph.step.completed"
        for event in reopened.capture_snapshot().events
    ) == 1


def test_nested_child_events_do_not_break_parent_completion_or_next_handoff(
    tmp_path,
):
    _store, journal, artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _append(sinks[STEP_0], STEP_0, "run_started", 1, status="running")

    child_attempt = AttemptRef(GENERATION, "graph-step-0-child")
    child_projector = _projector(child_attempt, artifacts)
    child_sink = DurableEventSink(journal, child_attempt, child_projector)
    _append(child_sink, child_attempt, "run_started", 1, status="running")
    _append(
        child_sink,
        child_attempt,
        "final_message",
        2,
        content="nested child output",
    )
    _append(
        child_sink,
        child_attempt,
        "run_completed",
        3,
        status="completed",
    )

    _append(
        sinks[STEP_0],
        STEP_0,
        "final_message",
        2,
        content="zero",
    )
    _append(
        sinks[STEP_0],
        STEP_0,
        "run_completed",
        3,
        status="completed",
    )
    completion = service.complete_step(
        plan,
        0,
        full_output={"output": "zero"},
    )

    assert completion.source_event_range.start == completion.checkpoint_cursor
    assert completion.source_event_range.end == completion.checkpoint_cursor
    checkpoint_event = next(
        event
        for event in journal.capture_snapshot().events
        if event.event_id == completion.checkpoint_cursor.event_id
    )
    assert checkpoint_event.event_type == "graph.step.completed"
    assert checkpoint_event.attempt == STEP_0
    assert checkpoint_event.resource_refs == (completion.output_artifact.ref,)

    next_step = service.start_step(plan, 1)
    assert next_step.disposition is GraphStepDisposition.STARTED
    assert next_step.handoff is not None
    assert next_step.handoff.envelope.child_attempt == STEP_0
    assert next_step.handoff.envelope.source_event_range == (
        completion.source_event_range
    )


def test_started_without_canonical_terminal_fails_closed(tmp_path):
    _store, _journal, _artifacts, _sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)

    recovery = service.recover(plan)

    assert recovery.uncertain_step_index == 0
    assert recovery.next_step_index is None
    with pytest.raises(GraphCheckpointError, match="replay is forbidden"):
        service.start_step(plan, 0)


def test_pending_interaction_is_reported_as_suspended_not_rerunnable(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _append(
        sinks[STEP_0],
        STEP_0,
        "interaction_requested",
        10,
        interaction_id="interaction-1",
        status="pending",
    )

    recovery = service.recover(plan)

    assert recovery.suspended_step_index == 0
    assert recovery.uncertain_step_index is None
    assert recovery.next_step_index is None
    with pytest.raises(GraphCheckpointError, match="interaction resume"):
        service.start_step(plan, 0)


def test_same_orchestration_attempt_cannot_change_topology_after_admission(tmp_path):
    _store, _journal, _artifacts, _sinks, service, plan = _bootstrap(tmp_path)
    changed = _plan(plan.initial_input_cursor, topology="topology-v2")

    with pytest.raises(GraphCheckpointConflict):
        service.admit(changed)


def test_finalize_is_idempotent_and_preserves_the_last_output_ref(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="zero")
    service.complete_step(plan, 0, full_output={"output": "zero"})
    service.start_step(plan, 1)
    _complete_runtime(sinks, STEP_1, output="one", base=20)
    final_completion = service.complete_step(
        plan,
        1,
        full_output="one",
    )

    first = service.finalize(plan)
    replay = service.finalize(plan)

    assert first.is_complete is True
    assert replay == first
    graph_terminals = [
        event
        for event in journal.capture_snapshot().events
        if event.event_type == "graph.execution.completed"
    ]
    assert len(graph_terminals) == 1
    assert graph_terminals[0].resource_refs == (
        final_completion.output_artifact.ref,
    )


def test_completed_step_replay_rejects_changed_full_output(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="zero")
    service.complete_step(
        plan,
        0,
        full_output={"output": "zero"},
    )

    with pytest.raises(GraphCheckpointConflict, match="output"):
        service.complete_step(
            plan,
            0,
            full_output={"output": "changed"},
        )


def _terminate_runtime(sinks, attempt: AttemptRef, terminal_type: str, *, base: int = 1):
    from unchain.kernel.lifecycle_events import build_run_failure_payload

    cancelled = terminal_type == "run_cancelled"
    _append(sinks[attempt], attempt, "run_started", base, status="running")
    return _append(
        sinks[attempt],
        attempt,
        terminal_type,
        base + 1,
        **build_run_failure_payload(
            iteration=0,
            error_code="execution_cancelled" if cancelled else "kernel_run_failed",
            cancelled=cancelled,
        ),
    )


def _step_seals(journal):
    return [
        event.event_type
        for event in journal.capture_snapshot().events
        if event.event_type.startswith("graph.step.")
        and event.event_type
        not in {"graph.step.started", "graph.step.resume.admitted"}
    ]


@pytest.mark.parametrize(
    ("terminal_type", "status", "seal"),
    [
        ("run_failed", GraphTerminalStatus.FAILED, "graph.step.failed"),
        ("run_cancelled", GraphTerminalStatus.CANCELLED, "graph.step.cancelled"),
    ],
)
def test_fail_step_seals_a_failed_or_cancelled_step_at_run_time(
    tmp_path, terminal_type, status, seal
):
    """#389 AC-389-2: the seal is written when the run ends, not by a later recovery."""

    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    terminal = _terminate_runtime(sinks, STEP_0, terminal_type)

    recovery = service.fail_step(plan, 0, status=status)

    assert _step_seals(journal) == [seal]
    assert recovery.terminal_status is status
    assert recovery.next_step_index is None
    assert recovery.uncertain_step_index is None
    sealed = [e for e in journal.capture_snapshot().events if e.event_type == seal][0]
    assert sealed.attempt == STEP_0
    assert sealed.payload["terminal_cursor"] == terminal.cursor.to_dict()
    # A sealed dead step ends the graph: neither a replay of it nor the next
    # step can start.
    with pytest.raises(GraphCheckpointError, match="cannot advance"):
        service.start_step(plan, 0)
    with pytest.raises(GraphCheckpointError, match="cannot advance"):
        service.start_step(plan, 1)


def test_fail_step_is_idempotent_and_agrees_with_recovery(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _terminate_runtime(sinks, STEP_0, "run_failed")

    first = service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)
    again = service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)
    recovered = service.recover(plan)

    assert _step_seals(journal) == ["graph.step.failed"]
    assert first.terminal_status is again.terminal_status is GraphTerminalStatus.FAILED
    assert recovered.terminal_status is GraphTerminalStatus.FAILED


def test_recovery_after_fail_step_does_not_seal_twice_across_a_restart(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _terminate_runtime(sinks, STEP_0, "run_cancelled")
    service.fail_step(plan, 0, status=GraphTerminalStatus.CANCELLED)

    _store, reopened, _artifacts, _sinks, restarted = _open(tmp_path)
    recovery = restarted.recover(plan)

    assert recovery.terminal_status is GraphTerminalStatus.CANCELLED
    assert _step_seals(reopened) == ["graph.step.cancelled"]


def test_journal_with_the_terminal_but_no_seal_still_recovers(tmp_path):
    """Journals written before run-time sealing keep recovering (both orders)."""

    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _terminate_runtime(sinks, STEP_0, "run_failed")
    assert _step_seals(journal) == []

    _store, reopened, _artifacts, _sinks, restarted = _open(tmp_path)
    recovery = restarted.recover(plan)

    assert recovery.terminal_status is GraphTerminalStatus.FAILED
    assert _step_seals(reopened) == ["graph.step.failed"]
    # A later run-time call finds the seal and changes nothing.
    assert (
        restarted.fail_step(plan, 0, status=GraphTerminalStatus.FAILED).terminal_status
        is GraphTerminalStatus.FAILED
    )
    assert _step_seals(reopened) == ["graph.step.failed"]


@pytest.mark.parametrize(
    ("terminal_type", "wrong_status"),
    [
        ("run_failed", GraphTerminalStatus.CANCELLED),
        ("run_cancelled", GraphTerminalStatus.FAILED),
    ],
)
def test_fail_step_refuses_a_status_that_disagrees_with_the_terminal(
    tmp_path, terminal_type, wrong_status
):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _terminate_runtime(sinks, STEP_0, terminal_type)

    with pytest.raises(GraphCheckpointConflict, match="disagrees"):
        service.fail_step(plan, 0, status=wrong_status)
    assert _step_seals(journal) == []


def test_fail_step_refuses_to_reseal_with_a_different_status(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _terminate_runtime(sinks, STEP_0, "run_failed")
    service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)

    with pytest.raises(GraphCheckpointConflict, match="different terminal"):
        service.fail_step(plan, 0, status=GraphTerminalStatus.CANCELLED)
    assert _step_seals(journal) == ["graph.step.failed"]


def test_fail_step_refuses_a_completed_step(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="done")
    service.complete_step(plan, 0, full_output="done")

    with pytest.raises(GraphCheckpointConflict, match="already completed"):
        service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)
    assert _step_seals(journal) == ["graph.step.completed"]


def test_fail_step_requires_a_durable_start_and_a_canonical_terminal(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    with pytest.raises(GraphCheckpointError, match="durable start"):
        service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)

    service.start_step(plan, 0)
    with pytest.raises(GraphCheckpointError, match="canonical"):
        service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)
    _append(sinks[STEP_0], STEP_0, "run_started", 1, status="running")
    with pytest.raises(GraphCheckpointError, match="canonical"):
        service.fail_step(plan, 0, status=GraphTerminalStatus.FAILED)
    assert _step_seals(journal) == []


@pytest.mark.parametrize(
    "status", [GraphTerminalStatus.COMPLETED, "failed", None]
)
def test_fail_step_only_takes_a_failed_or_cancelled_status(tmp_path, status):
    _store, _journal, _artifacts, _sinks, service, plan = _bootstrap(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        service.fail_step(plan, 0, status=status)


@pytest.mark.parametrize("step_index", [-1, 2, True, "0"])
def test_fail_step_rejects_a_step_outside_the_plan(tmp_path, step_index):
    _store, _journal, _artifacts, _sinks, service, plan = _bootstrap(tmp_path)
    with pytest.raises(ValueError, match="outside the graph plan"):
        service.fail_step(plan, step_index, status=GraphTerminalStatus.FAILED)


# --- #389 root terminal: every attempt, including the orchestration attempt,
# ends with one canonical terminal. S1-S4 are the shapes from the plan. ---


def _terminate_root(sinks, terminal_type: str, seq: int = 50):
    from unchain.kernel.lifecycle_events import build_run_failure_payload

    cancelled = terminal_type == "run_cancelled"
    return _append(
        sinks[ORCHESTRATION],
        ORCHESTRATION,
        terminal_type,
        seq,
        **build_run_failure_payload(
            iteration=0,
            error_code="execution_cancelled" if cancelled else "kernel_run_failed",
            cancelled=cancelled,
        ),
    )


def _complete_and_seal(sinks, service, plan, index, attempt, output):
    service.start_step(plan, index)
    _complete_runtime(sinks, attempt, output=output)
    service.complete_step(plan, index, full_output=output)


def _shape(tmp_path, shape: str, terminal_type: str = "run_cancelled"):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    if shape == "S1":
        service.start_step(plan, 0)
        _terminate_runtime(sinks, STEP_0, terminal_type)
        service.fail_step(
            plan,
            0,
            status=(
                GraphTerminalStatus.CANCELLED
                if terminal_type == "run_cancelled"
                else GraphTerminalStatus.FAILED
            ),
        )
    elif shape == "S3":
        _complete_and_seal(sinks, service, plan, 0, STEP_0, "o0")
    elif shape == "S4":
        _complete_and_seal(sinks, service, plan, 0, STEP_0, "o0")
        _complete_and_seal(sinks, service, plan, 1, STEP_1, "o1")
    else:
        assert shape == "S2"
    _terminate_root(sinks, terminal_type)
    return journal, sinks, service, plan


ROOT_SHAPES = [
    ("S1", "run_failed", GraphTerminalStatus.FAILED, 0),
    ("S1", "run_cancelled", GraphTerminalStatus.CANCELLED, 0),
    ("S2", "run_cancelled", GraphTerminalStatus.CANCELLED, 0),
    ("S3", "run_cancelled", GraphTerminalStatus.CANCELLED, 1),
    ("S4", "run_cancelled", GraphTerminalStatus.CANCELLED, None),
]


@pytest.mark.parametrize(("shape", "terminal_type", "status", "next_index"), ROOT_SHAPES)
def test_a_root_terminal_ends_the_graph(tmp_path, shape, terminal_type, status, next_index):
    """#389: a failed or cancelled root is terminal; nothing advances after it."""

    journal, _sinks, service, plan = _shape(tmp_path, shape, terminal_type)

    recovery = service.recover(plan)
    assert recovery.terminal_status is status
    assert recovery.next_step_index is None
    assert recovery.uncertain_step_index is None
    assert recovery.suspended_step_index is None

    if next_index is not None:
        with pytest.raises(GraphCheckpointError, match="cannot advance"):
            service.start_step(plan, next_index)
    with pytest.raises(GraphCheckpointError, match="cannot finalize"):
        service.finalize(plan)
    assert "graph.execution.completed" not in {
        event.event_type for event in journal.capture_snapshot().events
    }


@pytest.mark.parametrize(("shape", "terminal_type", "status", "next_index"), ROOT_SHAPES)
def test_a_root_terminal_survives_a_restart(tmp_path, shape, terminal_type, status, next_index):
    _journal, _sinks, _service, plan = _shape(tmp_path, shape, terminal_type)

    _store, reopened, _artifacts, _sinks, restarted = _open(tmp_path)
    recovery = restarted.recover(plan)

    assert recovery.terminal_status is status
    with pytest.raises(GraphCheckpointError, match="cannot finalize"):
        restarted.finalize(plan)
    if next_index is not None:
        with pytest.raises(GraphCheckpointError, match="cannot advance"):
            restarted.start_step(plan, next_index)
    assert "graph.execution.completed" not in {
        event.event_type for event in reopened.capture_snapshot().events
    }


def test_a_cancelled_root_does_not_block_sealing_existing_step_evidence(tmp_path):
    """Crash between a step's run_completed and its seal, then a cold cancel: the
    completed step is still sealed (S4), and sealing is all that happens."""

    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _complete_runtime(sinks, STEP_0, output="o0")
    _terminate_root(sinks, "run_cancelled")

    recovery = service.recover(plan)

    assert _step_seals(journal) == ["graph.step.completed"]
    assert recovery.terminal_status is GraphTerminalStatus.CANCELLED
    assert len(recovery.completed_steps) == 1
    assert recovery.next_step_index is None
    assert service.complete_step(plan, 0, full_output="o0").step == plan.steps[0]
    with pytest.raises(GraphCheckpointError, match="cannot advance"):
        service.start_step(plan, 1)


def test_a_cancelled_root_does_not_block_fail_step_for_its_dead_step(tmp_path):
    _store, journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    service.start_step(plan, 0)
    _terminate_runtime(sinks, STEP_0, "run_cancelled")
    _terminate_root(sinks, "run_cancelled")

    recovery = service.fail_step(plan, 0, status=GraphTerminalStatus.CANCELLED)

    assert _step_seals(journal) == ["graph.step.cancelled"]
    assert recovery.terminal_status is GraphTerminalStatus.CANCELLED


def test_a_root_run_completed_alone_is_not_a_graph_terminal(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    _append(sinks[ORCHESTRATION], ORCHESTRATION, "run_completed", 50, status="completed")

    recovery = service.recover(plan)

    assert recovery.terminal_status is None
    assert recovery.next_step_index == 0


def test_a_root_terminal_followed_by_an_execution_seal_is_ambiguous(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    _complete_and_seal(sinks, service, plan, 0, STEP_0, "o0")
    _complete_and_seal(sinks, service, plan, 1, STEP_1, "o1")
    _terminate_root(sinks, "run_cancelled")
    # Written behind the service's back (an old build, or a bug): the
    # repository seal is taken from a scan made before the root terminal.
    completion = service.repository.scan(plan).recovery.completed_steps[1]
    service.repository.finalize(plan, completion)

    with pytest.raises(GraphCheckpointError, match="ambiguous"):
        service.recover(plan)


def test_two_root_terminals_are_ambiguous(tmp_path):
    _store, _journal, _artifacts, sinks, service, plan = _bootstrap(tmp_path)
    _terminate_root(sinks, "run_cancelled", 50)
    _terminate_root(sinks, "run_failed", 51)

    with pytest.raises(GraphCheckpointError, match="ambiguous"):
        service.recover(plan)
