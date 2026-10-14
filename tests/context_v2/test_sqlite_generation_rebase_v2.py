from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest
import unchain.persistence.sqlite_generation_rebase_v2 as generation_rebase_module

from unchain.context import (
    MAX_ARTIFACT_BYTES,
    ArtifactService,
    ContextCompileRequest,
    SourceMessageCursor,
    resolve_context_budget,
)
from unchain.context.attachments import HostResolvedAttachment
from unchain.context.coordinator import ContextCompileCoordinator
from unchain.context.derived_handoff import (
    DerivedHandoffInputIngress,
    HostResolvedDerivedHandoffInput,
)
from unchain.context.graph_checkpoint import (
    GraphCheckpointService,
    GraphExecutionPlan,
    GraphStepBinding,
    GraphTerminalStatus,
    JournalGraphCheckpointRepository,
)
from unchain.context.handoff import DurableHandoffRecorder, HandoffService
from unchain.context.ingress import (
    ContextInputIngress,
    HostResolvedCurrentInput,
    HostResolvedInteractionInput,
)
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.context.models import HandoffStatus
from unchain.journal import (
    AttemptRef,
    DurableEventSink,
    EventCursor,
    EventRange,
    GenerationRef,
    OperationRef,
    ResourceRef,
    SemanticEventDraft,
)
from unchain.persistence.sqlite_context_compiler_v2 import (
    SQLiteContextCompilerV2Store,
)
from unchain.persistence.sqlite_generation_lifecycle_v2 import (
    HostGenerationTransitionKind,
    SQLiteHostGenerationLifecycleV2,
)
from unchain.persistence.sqlite_generation_rebase_v2 import (
    GenerationRebaseConflict,
    GenerationRebaseError,
    GenerationRebaseFailureReason,
    GenerationRebaseIntent,
    GenerationRebaseJournalIncompatible,
    GenerationRebaseKind,
    GenerationRebasePreflight,
    GenerationRebasePreflightBlocked,
    GenerationRebaseRecoveryRequired,
    GenerationRebaseRequest,
    GenerationRebaseUnavailable,
    GenerationSnapshotMessage,
    GenerationTaskStateDescriptor,
    SQLiteGenerationRebaseV2Service,
    build_generation_rebase_operation,
    generation_rebase_failure_detail,
    recover_generation_rebase_attempt,
)
from unchain.persistence.sqlite_legacy_bootstrap_v2 import (
    SQLiteLegacyBootstrapService,
)
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


OWNER = "chat-generation-rebase"
SESSION = "session-generation-rebase"
EXECUTION = "execution-generation-rebase"


def _store(root: Path) -> SQLiteContextV2Store:
    return SQLiteContextV2Store(
        database_path=root / "context_v2.sqlite3",
        object_directory=root / "objects",
    )


def _messages(prefix: str, *contents: tuple[str, str]):
    return tuple(
        GenerationSnapshotMessage(
            f"{prefix}-message-{index}",
            role,
            content,
        )
        for index, (role, content) in enumerate(contents, start=1)
    )


def _intent(
    *,
    kind: GenerationRebaseKind = GenerationRebaseKind.CREATE,
    generation_id: str = "generation-1",
    attempt_id: str = "attempt-1",
    previous_generation_id: str = "",
    expected_head_revision: int = 0,
    source_revision: str = "source-revision-1",
    messages=None,
    task_state=None,
) -> GenerationRebaseIntent:
    return GenerationRebaseIntent(
        owner_chat_id=OWNER,
        session_id=SESSION,
        execution_id=EXECUTION,
        generation_id=generation_id,
        attempt_id=attempt_id,
        kind=kind,
        previous_generation_id=previous_generation_id,
        expected_head_revision=expected_head_revision,
        source_revision=source_revision,
        messages=(
            messages
            if messages is not None
            else _messages(
                generation_id,
                ("user", f"prompt for {generation_id}"),
            )
        ),
        preflight=GenerationRebasePreflight(
            proof_id=f"preflight-{generation_id}",
            host_snapshot_sanitized=True,
        ),
        task_state=task_state,
    )


def _request(
    intent: GenerationRebaseIntent,
    *,
    operation_id: str,
) -> GenerationRebaseRequest:
    return GenerationRebaseRequest(
        intent=intent,
        operation=build_generation_rebase_operation(
            operation_id=operation_id,
            intent=intent,
        ),
    )


def _next(
    previous,
    *,
    kind: GenerationRebaseKind,
    ordinal: int,
    messages=None,
):
    intent = _intent(
        kind=kind,
        generation_id=f"generation-{ordinal}",
        attempt_id=f"attempt-{ordinal}",
        previous_generation_id=previous.generation_id,
        expected_head_revision=previous.head_revision,
        source_revision=f"source-revision-{ordinal}",
        messages=messages,
    )
    return intent, _request(intent, operation_id=f"rebase-operation-{ordinal}")


def _append_event(
    store: SQLiteContextV2Store,
    receipt,
    *,
    event_id: str,
    event_type: str,
    interaction_id: str | None = None,
    attempt_id: str | None = None,
    operation_id: str | None = None,
    payload: dict | None = None,
    resource_refs: tuple[ResourceRef, ...] = (),
):
    active_attempt_id = attempt_id or receipt.attempt_id
    attempt = AttemptRef(
        GenerationRef(EXECUTION, receipt.generation_id),
        active_attempt_id,
    )
    event_payload = {
        "run_id": active_attempt_id,
        **(payload or {}),
    }
    if interaction_id is not None:
        event_payload["interaction_id"] = interaction_id
    draft = SemanticEventDraft(
        event_id=event_id,
        event_type=event_type,
        attempt=attempt,
        operation_id=operation_id or f"operation-{event_id}",
        payload=event_payload,
        resource_refs=resource_refs,
    )
    return store.bind_execution(EXECUTION).append(
        request=draft.to_append_request()
    ).event


def _durable_counts(store: SQLiteContextV2Store) -> dict[str, int]:
    with sqlite3.connect(store.database_path) as connection:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "events",
                "operations",
                "legacy_bootstrap_manifests",
                "host_generation_records",
                "host_generation_attempt_bindings",
            )
        }


def _durable_authority_image(
    store: SQLiteContextV2Store,
) -> tuple[tuple[str, ...], tuple[tuple[str, int, bytes], ...]]:
    def object_image(path: Path) -> tuple[str, int, bytes]:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").digest()
        return path.name, path.stat().st_size, digest

    with sqlite3.connect(store.database_path) as connection:
        database = tuple(connection.iterdump())
    objects = tuple(
        object_image(path)
        for path in sorted(store.object_directory.iterdir())
        if path.is_file()
    )
    return database, objects


def _rewrite_event_record(
    store: SQLiteContextV2Store,
    event,
    mutate,
) -> None:
    with sqlite3.connect(store.database_path) as connection:
        raw = connection.execute(
            "SELECT event_json FROM events WHERE execution_id = ? AND store_seq = ?",
            (EXECUTION, event.store_seq),
        ).fetchone()[0]
        record = json.loads(bytes(raw).decode("utf-8"))
        mutate(record)
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        connection.execute(
            """
            UPDATE events SET event_json = ?, event_sha256 = ?
            WHERE execution_id = ? AND store_seq = ?
            """,
            (
                encoded,
                hashlib.sha256(encoded).hexdigest(),
                EXECUTION,
                event.store_seq,
            ),
        )


def _checkpoint_repository(store: SQLiteContextV2Store):
    journal = store.bind_execution(EXECUTION)
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    return SQLiteContextCompilerV2Store(
        context_store=store,
    ).bind_execution(
        EXECUTION,
        artifacts=artifacts,
    ).checkpoints


def _graph_checkpoint_runtime(
    store: SQLiteContextV2Store,
    receipt,
    *,
    step_count: int = 1,
    root_input_kind: str = "message",
    root_attachment: bool = False,
):
    generation = GenerationRef(EXECUTION, receipt.generation_id)
    orchestration = AttemptRef(generation, "graph-orchestration")
    step_attempts = tuple(
        AttemptRef(generation, f"graph-step-{index}")
        for index in range(step_count)
    )
    journal = store.bind_execution(EXECUTION)
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, _media_type: content,
    )
    attempts = (orchestration, *step_attempts)
    projectors = {
        attempt: CanonicalSemanticEventProjector(
            attempt=attempt,
            artifacts=artifacts,
            payload_sanitizer=lambda _event_type, payload: payload,
        )
        for attempt in attempts
    }
    sinks = {
        attempt: DurableEventSink(journal, attempt, projectors[attempt])
        for attempt in attempts
    }

    def derived_ingress(
        consumer_attempt: AttemptRef,
        source_attempt: AttemptRef,
    ) -> DerivedHandoffInputIngress:
        projector = projectors[consumer_attempt]
        sink = sinks[consumer_attempt]
        return DerivedHandoffInputIngress(
            consumer_attempt=consumer_attempt,
            source_attempt=source_attempt,
            handoff_recorder=DurableHandoffRecorder(
                attempt=consumer_attempt,
                handoffs=HandoffService(artifacts),
                projector=projector,
                sink=sink,
            ),
            input_ingress=ContextInputIngress(
                attempt=consumer_attempt,
                projector=projector,
                sink=sink,
            ),
        )

    service = GraphCheckpointService(
        repository=JournalGraphCheckpointRepository(journal),
        artifacts=artifacts,
        derived_ingress_resolver=derived_ingress,
    )
    steps = []
    source = orchestration
    for index, attempt in enumerate(step_attempts):
        steps.append(
            GraphStepBinding(
                index=index,
                node_id=f"node-{index}",
                attempt=attempt,
                source_attempt=source,
                provider="openai",
                model="gpt-test",
                configuration_sha256=hashlib.sha256(
                    f"node-{index}-configuration".encode("utf-8")
                ).hexdigest(),
            )
        )
        source = attempt
    root_ingress = ContextInputIngress(
        attempt=orchestration,
        projector=projectors[orchestration],
        sink=sinks[orchestration],
    )
    if root_input_kind == "message":
        root_attachments = ()
        if root_attachment:
            root_attachment_artifact = artifacts.persist(
                b"root graph attachment",
                media_type="text/plain",
                operation_id="root-graph-attachment",
            )
            root_attachments = (
                HostResolvedAttachment(
                    artifact=root_attachment_artifact,
                    kind="input",
                    name="root.txt",
                    media_type="text/plain",
                ),
            )
        root_input = root_ingress.persist(
            HostResolvedCurrentInput(
                attempt=orchestration,
                content="run the graph",
                message_index=0,
                attachments=root_attachments,
            )
        )
    elif root_input_kind == "interaction":
        if root_attachment:
            raise ValueError("interaction root input cannot carry attachments")
        root_interaction_id = "root-graph-interaction"
        sinks[orchestration].append_projected(
            SemanticEventDraft(
                event_id="root-graph-interaction-request",
                event_type="interaction.requested",
                attempt=orchestration,
                operation_id="root-graph-interaction-request-operation",
                payload={
                    "run_id": orchestration.attempt_id,
                    "interaction_id": root_interaction_id,
                },
            )
        )
        root_input = root_ingress.persist(
            HostResolvedInteractionInput(
                attempt=orchestration,
                interaction_id=root_interaction_id,
                response={"instruction": "run the graph"},
            )
        )
    else:
        raise ValueError("root_input_kind is unsupported")
    plan = GraphExecutionPlan(
        orchestration_attempt=orchestration,
        topology_sha256=hashlib.sha256(b"graph-topology").hexdigest(),
        initial_input_cursor=root_input.cursor,
        steps=tuple(steps),
    )
    service.admit(plan)
    return service, plan, sinks


def _append_graph_runtime_event(
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
            operation_id=(
                f"operation-{attempt.attempt_id}-{sequence}-{event_type}"
            ),
            payload={"run_id": attempt.attempt_id, **payload},
        )
    )


def _complete_graph_runtime(
    sink: DurableEventSink,
    attempt: AttemptRef,
    *,
    output: str,
):
    _append_graph_runtime_event(
        sink,
        attempt,
        "run_started",
        1,
        status="running",
    )
    _append_graph_runtime_event(
        sink,
        attempt,
        "final_message",
        2,
        content=output,
    )
    return _append_graph_runtime_event(
        sink,
        attempt,
        "run_completed",
        3,
        status="completed",
    )


def test_create_commits_all_authorities_and_existing_readers_can_verify(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    task_state = GenerationTaskStateDescriptor(
        "task-state-1",
        1,
        "a" * 64,
        (ResourceRef("context_event", "task-state-source", 1),),
    )
    intent = _intent(
        messages=_messages(
            "generation-1",
            ("user", "initial prompt"),
            ("assistant", "initial answer"),
        ),
        task_state=task_state,
    )

    receipt = service.rebase(
        _request(intent, operation_id="rebase-operation-create")
    )

    assert receipt.kind is GenerationRebaseKind.CREATE
    assert receipt.head_revision == 1
    assert receipt.message_count == 2
    assert receipt.task_state == task_state
    assert service.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == "generation-1"
    snapshot = store.bind_execution(EXECUTION).capture_snapshot()
    assert [event.attempt.generation.generation_id for event in snapshot.events] == [
        "generation-1",
        "generation-1",
    ]
    assert SQLiteLegacyBootstrapService(store).current(OWNER).generation_id == (
        "generation-1"
    )
    lifecycle = SQLiteHostGenerationLifecycleV2(store)
    assert lifecycle.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == "generation-1"
    assert lifecycle.attempt_binding(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
        attempt_id="attempt-1",
    ).generation_id == "generation-1"
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM operations WHERE execution_id = ?",
            (EXECUTION,),
        ).fetchone()[0] == 3
        assert connection.execute(
            "SELECT COUNT(*) FROM host_generation_operations"
        ).fetchone()[0] == 2
        host_head = connection.execute(
            "SELECT current_generation_id, revision FROM host_generation_heads"
        ).fetchone()
        bootstrap_head = connection.execute(
            """
            SELECT current_generation_id, head_revision
            FROM legacy_bootstrap_chat_heads
            """
        ).fetchone()
    assert host_head == bootstrap_head == ("generation-1", 1)


def test_edit_regenerate_and_retry_preserve_every_old_generation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    create_request = _request(
        _intent(),
        operation_id="rebase-operation-1",
    )
    create = service.rebase(create_request)
    edit_intent, edit_request = _next(
        create,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    edit = service.rebase(edit_request)
    regenerate_intent, regenerate_request = _next(
        edit,
        kind=GenerationRebaseKind.REGENERATE,
        ordinal=3,
    )
    regenerate = service.rebase(regenerate_request)
    retry_intent, retry_request = _next(
        regenerate,
        kind=GenerationRebaseKind.RETRY,
        ordinal=4,
        messages=_messages(
            "generation-4",
            ("user", "retry the same logical request"),
        ),
    )
    retry = service.rebase(retry_request)

    assert [
        create.kind,
        edit.kind,
        regenerate.kind,
        retry.kind,
    ] == [
        GenerationRebaseKind.CREATE,
        GenerationRebaseKind.EDIT,
        GenerationRebaseKind.REGENERATE,
        GenerationRebaseKind.RETRY,
    ]
    assert retry.head_revision == 4
    reopened = SQLiteGenerationRebaseV2Service(_store(tmp_path))
    assert reopened.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == "generation-4"
    assert [
        reopened.receipt_for_generation(
            owner_chat_id=OWNER,
            execution_id=EXECUTION,
            session_id=SESSION,
            generation_id=f"generation-{ordinal}",
        ).kind
        for ordinal in range(1, 5)
    ] == [
        GenerationRebaseKind.CREATE,
        GenerationRebaseKind.EDIT,
        GenerationRebaseKind.REGENERATE,
        GenerationRebaseKind.RETRY,
    ]
    assert reopened.rebase(create_request) == replace(create, duplicate=True)
    assert reopened.rebase(retry_request) == replace(retry, duplicate=True)
    assert store.bind_execution(EXECUTION).capture_snapshot().events[0].attempt == (
        _intent().attempt
    )
    lifecycle = SQLiteHostGenerationLifecycleV2(store)
    assert lifecycle.generation(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
        generation_id="generation-4",
    ).kind is HostGenerationTransitionKind.REGENERATE
    assert SQLiteLegacyBootstrapService(store).current(OWNER).generation_id == (
        "generation-4"
    )
    del edit_intent, regenerate_intent, retry_intent


def test_payload_hash_stale_head_and_source_revision_conflicts_leave_head(
    tmp_path: Path,
) -> None:
    service = SQLiteGenerationRebaseV2Service(_store(tmp_path))
    initial_intent = _intent()
    initial_request = _request(
        initial_intent,
        operation_id="rebase-operation-1",
    )
    initial = service.rebase(initial_request)
    assert service.rebase(initial_request) == replace(initial, duplicate=True)

    changed_payload = replace(
        initial_intent,
        messages=_messages("changed", ("user", "changed payload")),
    )
    with pytest.raises(GenerationRebaseConflict, match="operation|payload"):
        service.rebase(
            _request(changed_payload, operation_id="rebase-operation-1")
        )

    stale = _intent(
        kind=GenerationRebaseKind.EDIT,
        generation_id="generation-2",
        attempt_id="attempt-2",
        previous_generation_id="not-current",
        expected_head_revision=1,
        source_revision="source-revision-2",
    )
    with pytest.raises(GenerationRebaseConflict, match="previous|current"):
        service.rebase(_request(stale, operation_id="rebase-operation-stale"))

    reused_source = replace(
        stale,
        previous_generation_id="generation-1",
        source_revision="source-revision-1",
    )
    with pytest.raises(GenerationRebaseConflict, match="source revision"):
        service.rebase(
            _request(reused_source, operation_id="rebase-operation-source-reuse")
        )
    assert service.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == "generation-1"


def test_injected_failure_after_journal_writes_rolls_back_every_authority(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
        messages=_messages(
            "generation-2",
            ("user", "edited prompt"),
            ("assistant", "edited answer"),
        ),
    )
    with sqlite3.connect(store.database_path) as connection:
        before = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "events",
                "operations",
                "legacy_bootstrap_manifests",
                "host_generation_records",
                "host_generation_attempt_bindings",
            )
        }
        connection.executescript(
            """
            CREATE TRIGGER fail_generation_rebase_record
            BEFORE INSERT ON host_generation_records
            WHEN NEW.generation_id = 'generation-2'
            BEGIN
                SELECT RAISE(ABORT, 'injected generation failure');
            END;
            """
        )

    with pytest.raises(GenerationRebaseError):
        service.rebase(edit_request)

    with sqlite3.connect(store.database_path) as connection:
        after = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in before
        }
        assert connection.execute(
            "SELECT current_generation_id, revision FROM host_generation_heads"
        ).fetchone() == ("generation-1", 1)
        assert connection.execute(
            """
            SELECT current_generation_id, head_revision
            FROM legacy_bootstrap_chat_heads
            """
        ).fetchone() == ("generation-1", 1)
        assert connection.execute(
            "SELECT next_store_seq FROM executions WHERE execution_id = ?",
            (EXECUTION,),
        ).fetchone() == (2,)
    assert after == before
    assert service.receipt_for_generation(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
        generation_id="generation-2",
    ) is None
    del edit_intent


def test_current_head_drives_compiler_to_exclude_old_generation(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(
            _intent(
                messages=_messages(
                    "generation-1",
                    ("user", "old prompt must not compile"),
                    ("assistant", "old answer must not compile"),
                )
            ),
            operation_id="rebase-operation-1",
        )
    )
    edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
        messages=_messages(
            "generation-2",
            ("user", "new prompt is current"),
        ),
    )
    edit = service.rebase(edit_request)
    head = service.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    )
    journal = store.bind_execution(EXECUTION)
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    capabilities = SQLiteContextCompilerV2Store(
        context_store=store,
    ).bind_execution(
        EXECUTION,
        artifacts=artifacts,
    )
    request = ContextCompileRequest(
        case="generation-rebase-current-only",
        source_messages=(
            {"role": "user", "content": "new prompt is current"},
        ),
        current_generation=head.current_generation_id,
        budget=resolve_context_budget(context_window_tokens=8_192),
        source_message_cursors=(
            SourceMessageCursor(
                0,
                edit.last_cursor.event_id,
                edit.last_cursor.store_seq,
            ),
        ),
        provider="openai",
        model="synthetic",
        build_id="generation-rebase-build-2",
        execution_id=EXECUTION,
        generation_id=head.current_generation_id,
        attempt_id=head.current_attempt_id,
    )
    result = ContextCompileCoordinator(
        journal=journal,
        checkpoint_repository=capabilities.checkpoints,
        build_repository=capabilities.context_builds,
        partial_attempt_sink=lambda request, error: None,
    ).compile(request)

    contents = [message.get("content") for message in result.messages]
    assert "new prompt is current" in contents
    assert "old prompt must not compile" not in contents
    assert "old answer must not compile" not in contents
    assert [
        event.attempt.generation.generation_id
        for event in journal.capture_snapshot().events
    ] == ["generation-1", "generation-1", "generation-2"]
    del edit_intent


def test_concurrent_rebases_have_one_complete_winner(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first_service = SQLiteGenerationRebaseV2Service(store)
    second_service = SQLiteGenerationRebaseV2Service(store)
    initial = first_service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    intents = (
        _intent(
            kind=GenerationRebaseKind.EDIT,
            generation_id="generation-edit-a",
            attempt_id="attempt-edit-a",
            previous_generation_id=initial.generation_id,
            expected_head_revision=initial.head_revision,
            source_revision="source-revision-edit-a",
        ),
        _intent(
            kind=GenerationRebaseKind.REGENERATE,
            generation_id="generation-edit-b",
            attempt_id="attempt-edit-b",
            previous_generation_id=initial.generation_id,
            expected_head_revision=initial.head_revision,
            source_revision="source-revision-edit-b",
        ),
    )
    barrier = Barrier(2)

    def rebase(index: int):
        barrier.wait()
        service = first_service if index == 0 else second_service
        return service.rebase(
            _request(
                intents[index],
                operation_id=f"rebase-operation-concurrent-{index}",
            )
        )

    successes = []
    conflicts = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(rebase, index) for index in range(2)]
        for future in as_completed(futures):
            try:
                successes.append(future.result())
            except GenerationRebaseConflict as error:
                conflicts.append(error)

    assert len(successes) == 1
    assert len(conflicts) == 1
    assert successes[0].head_revision == 2
    assert first_service.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == successes[0].generation_id
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM host_generation_records"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT COUNT(*) FROM legacy_bootstrap_manifests"
        ).fetchone()[0] == 2


def test_pending_canonical_interaction_blocks_without_rebase_writes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _append_event(
        store,
        initial,
        event_id="interaction-request-1",
        event_type="interaction.requested",
        interaction_id="interaction-1",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebasePreflightBlocked,
        match="pending durable interaction",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before
    assert service.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == initial.generation_id


def test_resolved_canonical_interaction_allows_rebase(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _append_event(
        store,
        initial,
        event_id="interaction-request-1",
        event_type="interaction.requested",
        interaction_id="interaction-1",
    )
    _append_event(
        store,
        initial,
        event_id="interaction-resolution-1",
        event_type="interaction.resolved",
        interaction_id="interaction-1",
        payload={"outcome": "approved"},
    )
    _append_event(
        store,
        initial,
        event_id="run-completed-after-interaction",
        event_type="run_completed",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    edited = service.rebase(edit_request)

    assert edited.head_revision == 2
    assert edited.previous_generation_id == initial.generation_id


def test_authorized_canonical_resolution_supersedes_malformed_legacy_for_rebase(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _append_event(
        store,
        initial,
        event_id="interaction-request-legacy-repair",
        event_type="interaction.requested",
        interaction_id="interaction-legacy-repair",
    )
    _append_event(
        store,
        initial,
        event_id="interaction-resolution-malformed-legacy",
        event_type="interaction_resolved",
        interaction_id="interaction-legacy-repair",
    )
    artifact = ArtifactService(
        store.bind_execution(EXECUTION),
        sanitizer=lambda content, _media_type: content,
    ).persist_exact_json(
        {
            "interaction_id": "interaction-legacy-repair",
            "response": {"answer": "yes"},
            "submitted_by": "ui:test",
        },
        operation_id="artifact-interaction-legacy-repair",
    )
    _append_event(
        store,
        initial,
        event_id="interaction-resolution-canonical-repair",
        event_type="interaction.resolved",
        interaction_id="interaction-legacy-repair",
        payload={
            "submitted_by": "ui:test",
            "content_ref": artifact.ref.to_dict(),
            "content_bytes": artifact.byte_length,
            "content_sha256": artifact.sha256,
            "preview": artifact.preview,
            "preview_truncated": False,
        },
        resource_refs=(artifact.ref,),
    )
    _append_event(
        store,
        initial,
        event_id="run-completed-after-legacy-repair",
        event_type="run_completed",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    edited = service.rebase(edit_request)

    assert edited.head_revision == 2
    assert edited.previous_generation_id == initial.generation_id


def test_unauthorized_canonical_resolution_cannot_hide_legacy_during_rebase(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    interaction_id = "interaction-unauthorized-repair"
    _append_event(
        store,
        initial,
        event_id="interaction-request-unauthorized-repair",
        event_type="interaction.requested",
        interaction_id=interaction_id,
    )
    _append_event(
        store,
        initial,
        event_id="interaction-resolution-malformed-unauthorized",
        event_type="interaction_resolved",
        interaction_id=interaction_id,
    )
    preview = '{"answer":"yes"}'
    content = preview.encode("utf-8")
    _append_event(
        store,
        initial,
        event_id="interaction-resolution-unauthorized-repair",
        event_type="interaction.resolved",
        interaction_id=interaction_id,
        payload={
            "submitted_by": "ui:test",
            "content_ref": {
                "kind": "artifact",
                "id": "payload-only-artifact",
                "revision": 1,
            },
            "content_bytes": len(content),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "preview": preview,
            "preview_truncated": False,
        },
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebaseJournalIncompatible,
        match="resolution is duplicated",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before


def test_ambiguous_interaction_lifecycle_fails_closed_without_rebase_writes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    for ordinal in (1, 2):
        _append_event(
            store,
            initial,
            event_id=f"interaction-request-{ordinal}",
            event_type="interaction.requested",
            interaction_id="interaction-1",
        )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebaseJournalIncompatible,
        match="request is duplicated",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before


def test_current_prepared_checkpoint_blocks_until_committed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    checkpoints = _checkpoint_repository(store)
    prepared = checkpoints.prepare(
        source_range=EventRange(initial.first_cursor, initial.last_cursor),
        summary="durable checkpoint in progress",
        refs=(),
        operation=OperationRef("checkpoint-operation-1", "b" * 64),
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebasePreflightBlocked,
        match="prepared durable checkpoint",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before
    checkpoints.commit(prepared=prepared)
    edited = service.rebase(edit_request)
    assert edited.head_revision == 2


def test_prepared_checkpoint_for_an_old_generation_does_not_block_current(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    edited = service.rebase(edit_request)
    checkpoints = _checkpoint_repository(store)
    checkpoints.prepare(
        source_range=EventRange(initial.first_cursor, initial.last_cursor),
        summary="unfinished checkpoint for archived generation",
        refs=(),
        operation=OperationRef("checkpoint-operation-old", "c" * 64),
    )
    _retry_intent, retry_request = _next(
        edited,
        kind=GenerationRebaseKind.RETRY,
        ordinal=3,
    )

    retried = service.rebase(retry_request)

    assert retried.head_revision == 3
    assert retried.previous_generation_id == edited.generation_id


def test_exact_replay_precedes_new_current_generation_preflight_state(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    edited = service.rebase(edit_request)
    _append_event(
        store,
        edited,
        event_id="run-started-after-edit",
        event_type="run_started",
        attempt_id="attempt-live-after-edit",
    )

    assert service.rebase(edit_request) == replace(edited, duplicate=True)


def test_import_only_current_generation_allows_rebase(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    edited = service.rebase(edit_request)

    assert edited.head_revision == 2


def test_open_real_attempt_blocks_without_rebase_writes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _append_event(
        store,
        initial,
        event_id="run-started-live",
        event_type="run_started",
        attempt_id="attempt-live",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebasePreflightBlocked,
        match="unfinished durable attempt",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before


def test_open_tool_blocks_without_rebase_writes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    for event_id, event_type in (
        ("tool-call-live", "tool_call"),
        ("tool-started-live", "tool.started"),
    ):
        _append_event(
            store,
            initial,
            event_id=event_id,
            event_type=event_type,
            attempt_id="attempt-live",
            payload={"call_id": "call-live", "tool_name": "lookup"},
        )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebasePreflightBlocked,
        match="unfinished durable tool",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before


def test_completed_tool_and_terminal_attempt_allow_rebase(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    for event_id, event_type in (
        ("tool-call-complete", "tool_call"),
        ("tool-started-complete", "tool.started"),
        ("tool-result-complete", "tool_result"),
    ):
        _append_event(
            store,
            initial,
            event_id=event_id,
            event_type=event_type,
            attempt_id="attempt-complete",
            payload={"call_id": "call-complete", "tool_name": "lookup"},
        )
    _append_event(
        store,
        initial,
        event_id="run-completed-tool",
        event_type="run_completed",
        attempt_id="attempt-complete",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    edited = service.rebase(edit_request)

    assert edited.head_revision == 2


def test_terminal_attempt_with_memory_curator_audit_allows_rebase(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _append_event(
        store,
        initial,
        event_id="run-completed-with-curator-audit",
        event_type="run_completed",
        attempt_id="attempt-complete",
    )
    _append_event(
        store,
        initial,
        event_id="memory-curator-noop-after-completion",
        event_type="memory.curator.noop",
        attempt_id="attempt-complete",
        payload={
            "schema": "pupu.memory_curator_completion.v1",
            "status": "NoOp",
        },
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    edited = service.rebase(edit_request)

    assert edited.head_revision == 2


def test_duplicate_tool_lifecycle_fails_without_rebase_writes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    for event_id, event_type in (
        ("tool-call-duplicate", "tool_call"),
        ("tool-started-duplicate-1", "tool.started"),
        ("tool-started-duplicate-2", "tool.started"),
    ):
        _append_event(
            store,
            initial,
            event_id=event_id,
            event_type=event_type,
            attempt_id="attempt-corrupt",
            payload={"call_id": "call-corrupt", "tool_name": "lookup"},
        )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebaseJournalIncompatible,
        match="tool lifecycle is not uniquely paired",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before


def test_duplicate_attempt_terminal_fails_without_rebase_writes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    for event_id, event_type in (
        ("run-started-duplicate", "run_started"),
        ("run-completed-duplicate", "run_completed"),
        ("run-failed-duplicate", "run_failed"),
    ):
        _append_event(
            store,
            initial,
            event_id=event_id,
            event_type=event_type,
            attempt_id="attempt-corrupt",
        )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(
        GenerationRebaseJournalIncompatible,
        match="duplicate terminal events",
    ):
        service.rebase(edit_request)

    assert _durable_counts(store) == before


def test_failure_detail_and_old_consumer_type_order_remain_closed() -> None:
    recovery = GenerationRebaseRecoveryRequired(
        "generation rebase graph checkpoint recovery is required",
        reason=GenerationRebaseFailureReason.GRAPH_STEP_SEAL_MISSING,
        subject={
            "execution_id": EXECUTION,
            "generation_id": "generation-1",
        },
    )
    incompatible = GenerationRebaseJournalIncompatible(
        "generation rebase attempt has duplicate terminal events",
        reason=GenerationRebaseFailureReason.ATTEMPT_DUPLICATE_TERMINAL,
        subject={
            "execution_id": EXECUTION,
            "generation_id": "generation-1",
        },
    )

    assert isinstance(recovery, GenerationRebasePreflightBlocked)
    assert not isinstance(incompatible, GenerationRebasePreflightBlocked)
    assert isinstance(incompatible, GenerationRebaseConflict)
    assert incompatible.retryable is False
    for error in (recovery, incompatible):
        detail = generation_rebase_failure_detail(error)
        assert detail is not None
        assert set(detail) == {"schema", "reason", "subject"}
        assert detail["schema"] == "unchain.generation_rebase_failure.v1"

    def old_consumer(error: GenerationRebaseError) -> str:
        if isinstance(error, GenerationRebasePreflightBlocked):
            return "in_progress"
        if isinstance(error, GenerationRebaseConflict):
            return "conflict"
        return "unavailable"

    assert old_consumer(recovery) == "in_progress"
    assert old_consumer(incompatible) == "conflict"


def test_real_graph_crash_windows_recover_one_seal_at_a_time(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    rebase_service = SQLiteGenerationRebaseV2Service(store)
    initial = rebase_service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    _complete_graph_runtime(
        sinks[step.attempt],
        step.attempt,
        output="completed graph output",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_counts(store)

    with pytest.raises(GenerationRebaseRecoveryRequired) as step_failure_info:
        rebase_service.rebase(edit_request)

    step_failure = step_failure_info.value
    assert step_failure.code == "recovery_required"
    assert step_failure.reason == "graph_step_seal_missing"
    assert _durable_counts(store) == before
    step_recovery = recover_generation_rebase_attempt(
        service=rebase_service,
        request=edit_request,
        failure=step_failure,
        artifact_sanitizer=lambda content, _media_type: content,
    )
    assert step_recovery.action == "step_recovered"
    assert step_recovery.appended_event_count == 1
    assert step_recovery.artifact_count == 1

    before_finalize_preflight = _durable_counts(store)
    with pytest.raises(GenerationRebaseRecoveryRequired) as execution_failure_info:
        rebase_service.rebase(edit_request)
    assert _durable_counts(store) == before_finalize_preflight
    execution_failure = execution_failure_info.value
    assert execution_failure.reason == "graph_execution_seal_missing"
    execution_recovery = recover_generation_rebase_attempt(
        service=rebase_service,
        request=edit_request,
        failure=execution_failure,
        artifact_sanitizer=lambda content, _media_type: content,
    )
    assert execution_recovery.action == "execution_finalized"
    assert execution_recovery.appended_event_count == 1
    assert execution_recovery.artifact_count == 0
    assert recover_generation_rebase_attempt(
        service=rebase_service,
        request=edit_request,
        failure=execution_failure,
        artifact_sanitizer=lambda content, _media_type: content,
    ).action == "unchanged"

    edited = rebase_service.rebase(edit_request)

    assert edited.head_revision == 2


def test_stale_step_recovery_never_seals_a_later_terminal(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    rebase_service = SQLiteGenerationRebaseV2Service(store)
    initial = rebase_service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(
        store,
        initial,
        step_count=2,
    )
    graph_service.start_step(plan, 0)
    first_step = plan.steps[0]
    _complete_graph_runtime(
        sinks[first_step.attempt],
        first_step.attempt,
        output="first completed output",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    with pytest.raises(GenerationRebaseRecoveryRequired) as stale_info:
        rebase_service.rebase(edit_request)
    stale_failure = stale_info.value
    assert stale_failure.reason == "graph_step_seal_missing"

    graph_service.complete_step(
        plan,
        0,
        full_output={"output": "first completed output"},
    )
    graph_service.start_step(plan, 1)
    second_step = plan.steps[1]
    _complete_graph_runtime(
        sinks[second_step.attempt],
        second_step.attempt,
        output="second terminal must remain unsealed",
    )
    before = _durable_authority_image(store)

    recovered = recover_generation_rebase_attempt(
        service=rebase_service,
        request=edit_request,
        failure=stale_failure,
        artifact_sanitizer=lambda content, _media_type: content,
    )

    assert recovered.action == "unchanged"
    assert recovered.appended_event_count == 0
    assert recovered.artifact_count == 0
    assert _durable_authority_image(store) == before
    second_events = tuple(
        event
        for event in store.bind_execution(EXECUTION).capture_snapshot().events
        if event.attempt == second_step.attempt
    )
    assert not any(
        event.event_type in {
            "graph.step.completed",
            "graph.step.failed",
            "graph.step.cancelled",
        }
        for event in second_events
    )


def test_event_sha256_corruption_is_nonretryable_and_zero_write(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            UPDATE events SET event_sha256 = ?
            WHERE execution_id = ? AND store_seq = ?
            """,
            ("0" * 64, EXECUTION, initial.first_cursor.store_seq),
        )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.retryable is False
    assert incompatible.value.reason == "journal_authority_invalid"
    assert _durable_authority_image(store) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "delete",
        "truncate",
        "content_tamper",
        "descriptor_row",
        "oversized",
    ],
)
@pytest.mark.parametrize(
    "artifact_role",
    [
        "plan_initial_input",
        "plan_initial_attachment",
        "plan_initial_resolution",
        "start_provenance",
        "input_content",
        "step_output",
    ],
)
def test_graph_artifact_corruption_is_nonretryable_and_zero_write(
    tmp_path: Path,
    mutation: str,
    artifact_role: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(
        store,
        initial,
        root_input_kind=(
            "interaction"
            if artifact_role == "plan_initial_resolution"
            else "message"
        ),
        root_attachment=artifact_role == "plan_initial_attachment",
    )
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    output = "completed graph output with verified artifact bytes"
    _complete_graph_runtime(
        sinks[step.attempt],
        step.attempt,
        output=output,
    )
    graph_service.complete_step(
        plan,
        0,
        full_output={"output": output},
    )
    graph_service.finalize(plan)
    graph_events = store.bind_execution(EXECUTION).capture_snapshot().events
    plan_initial_input = next(
        event
        for event in graph_events
        if event.store_seq == plan.initial_input_cursor.store_seq
        and event.event_id == plan.initial_input_cursor.event_id
    )
    step_start = next(
        event
        for event in graph_events
        if event.attempt == step.attempt
        and event.event_type == "graph.step.started"
    )
    input_event = next(
        event
        for event in graph_events
        if event.attempt == step.attempt
        and event.event_type == "message.user"
    )
    step_seal = next(
        event
        for event in graph_events
        if event.event_type == "graph.step.completed"
    )
    if artifact_role == "plan_initial_attachment":
        artifact_ref = plan_initial_input.resource_refs[1]
    else:
        artifact_ref = {
            "plan_initial_input": plan_initial_input.resource_refs[0],
            "plan_initial_resolution": plan_initial_input.resource_refs[0],
            "start_provenance": step_start.resource_refs[0],
            "input_content": input_event.resource_refs[0],
            "step_output": step_seal.resource_refs[0],
        }[artifact_role]
    with sqlite3.connect(store.database_path) as connection:
        object_sha256 = connection.execute(
            """
            SELECT object_sha256 FROM artifacts
            WHERE execution_id = ? AND artifact_id = ? AND revision = ?
            """,
            (
                EXECUTION,
                artifact_ref.resource_id,
                artifact_ref.revision,
            ),
        ).fetchone()[0]
    object_path = store.object_directory / object_sha256
    original_bytes = object_path.read_bytes()
    if mutation == "delete":
        object_path.unlink()
    elif mutation == "truncate":
        object_path.write_bytes(original_bytes[: len(original_bytes) // 2])
    elif mutation == "content_tamper":
        object_path.write_bytes(
            bytes([original_bytes[0] ^ 1]) + original_bytes[1:]
        )
    elif mutation == "descriptor_row":
        with sqlite3.connect(store.database_path) as connection:
            connection.execute(
                """
                UPDATE artifacts SET artifact_record_sha256 = ?
                WHERE execution_id = ? AND object_sha256 = ?
                """,
                ("0" * 64, EXECUTION, object_sha256),
            )
    else:
        with object_path.open("r+b") as stream:
            stream.truncate(MAX_ARTIFACT_BYTES + 1)
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.retryable is False
    expected_reason = (
        "graph_plan_descriptor_invalid"
        if artifact_role.startswith("plan_initial_")
        else "graph_step_seal_foreign"
    )
    assert incompatible.value.reason == expected_reason
    assert _durable_authority_image(store) == before


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("plan_extra_ref", "graph_plan_descriptor_invalid"),
        ("plan_content_descriptor_drift", "graph_plan_descriptor_invalid"),
        ("plan_message_artifact_divergence", "graph_plan_descriptor_invalid"),
        ("plan_attachment_descriptor_drift", "graph_plan_descriptor_invalid"),
        ("plan_resolution_extra_ref", "graph_plan_descriptor_invalid"),
        ("plan_resolution_descriptor_drift", "graph_plan_descriptor_invalid"),
        ("plan_resolution_artifact_divergence", "graph_plan_descriptor_invalid"),
        ("handoff_extra_ref", "graph_step_seal_foreign"),
        ("handoff_full_output_descriptor_drift", "graph_step_seal_foreign"),
        ("handoff_envelope_artifact_ref_drift", "graph_step_seal_foreign"),
        ("input_extra_ref", "graph_step_seal_foreign"),
        ("input_nonartifact_ref", "graph_step_seal_foreign"),
        ("input_attachment_descriptor_drift", "graph_step_seal_foreign"),
        ("input_message_artifact_divergence", "graph_step_seal_foreign"),
        ("request_extra_ref", "graph_step_seal_foreign"),
        ("resolution_extra_ref", "graph_step_seal_foreign"),
        ("resolution_descriptor_drift", "graph_step_seal_foreign"),
        ("resolution_artifact_divergence", "graph_step_seal_foreign"),
        ("tool_confirmed_extra_ref", "graph_step_seal_foreign"),
        ("tool_denied_extra_ref", "graph_step_seal_foreign"),
        ("runtime_terminal_extra_ref", "graph_step_seal_foreign"),
        ("root_terminal_extra_ref", "graph_execution_seal_mismatched"),
    ],
)
def test_graph_trusted_event_artifact_closure_is_exact_and_zero_write(
    tmp_path: Path,
    mutation: str,
    expected_reason: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    plan_resolution = mutation.startswith("plan_resolution_")
    graph_service, plan, sinks = _graph_checkpoint_runtime(
        store,
        initial,
        root_input_kind="interaction" if plan_resolution else "message",
        root_attachment=not plan_resolution,
    )
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    journal = store.bind_execution(EXECUTION)
    repository = JournalGraphCheckpointRepository(journal)
    interaction_id = "artifact-closure-interaction"
    request_event = _append_event(
        store,
        initial,
        event_id="artifact-closure-request",
        event_type="interaction.requested",
        interaction_id=interaction_id,
        attempt_id=step.attempt.attempt_id,
    )
    resolution_receipt = ContextInputIngress(
        attempt=step.attempt,
        projector=sinks[step.attempt].projector,
        sink=sinks[step.attempt],
    ).persist(
        HostResolvedInteractionInput(
            attempt=step.attempt,
            interaction_id=interaction_id,
            response={"answer": "continue"},
        )
    )
    repository.resume(
        plan,
        step,
        interaction_id=interaction_id,
        request_cursor=EventCursor(
            request_event.store_seq,
            request_event.event_id,
        ),
        resolution_cursor=resolution_receipt.cursor,
    )
    tool_confirmed = _append_event(
        store,
        initial,
        event_id="artifact-closure-tool-confirmed",
        event_type="tool_confirmed",
        interaction_id=interaction_id,
        attempt_id=step.attempt.attempt_id,
    )
    tool_denied = _append_event(
        store,
        initial,
        event_id="artifact-closure-tool-denied",
        event_type="tool_denied",
        interaction_id=interaction_id,
        attempt_id=step.attempt.attempt_id,
    )
    terminal_receipt = _complete_graph_runtime(
        sinks[step.attempt],
        step.attempt,
        output="artifact closure output",
    )
    graph_service.complete_step(
        plan,
        0,
        full_output={"output": "artifact closure output"},
    )
    graph_service.finalize(plan)
    root_terminal_receipt = _append_graph_runtime_event(
        sinks[plan.orchestration_attempt],
        plan.orchestration_attempt,
        "run_completed",
        10,
        status="completed",
    )
    events = journal.capture_snapshot().events
    plan_input = next(
        event
        for event in events
        if event.store_seq == plan.initial_input_cursor.store_seq
        and event.event_id == plan.initial_input_cursor.event_id
    )
    handoff = next(
        event
        for event in events
        if event.attempt == step.attempt
        and event.event_type == "handoff.recorded"
    )
    step_input = next(
        event
        for event in events
        if event.attempt == step.attempt
        and event.event_type == "message.user"
    )
    step_seal = next(
        event
        for event in events
        if event.attempt == step.attempt
        and event.event_type == "graph.step.completed"
    )
    extra_ref = step_seal.resource_refs[0].to_dict()

    targets = {
        "plan_extra_ref": plan_input,
        "plan_content_descriptor_drift": plan_input,
        "plan_message_artifact_divergence": plan_input,
        "plan_attachment_descriptor_drift": plan_input,
        "plan_resolution_extra_ref": plan_input,
        "plan_resolution_descriptor_drift": plan_input,
        "plan_resolution_artifact_divergence": plan_input,
        "handoff_extra_ref": handoff,
        "handoff_full_output_descriptor_drift": handoff,
        "handoff_envelope_artifact_ref_drift": handoff,
        "input_extra_ref": step_input,
        "input_nonartifact_ref": step_input,
        "input_attachment_descriptor_drift": step_input,
        "input_message_artifact_divergence": step_input,
        "request_extra_ref": request_event,
        "resolution_extra_ref": resolution_receipt.event,
        "resolution_descriptor_drift": resolution_receipt.event,
        "resolution_artifact_divergence": resolution_receipt.event,
        "tool_confirmed_extra_ref": tool_confirmed,
        "tool_denied_extra_ref": tool_denied,
        "runtime_terminal_extra_ref": terminal_receipt.event,
        "root_terminal_extra_ref": root_terminal_receipt.event,
    }

    def mutate(record: dict) -> None:
        if mutation.endswith("extra_ref"):
            record["resource_refs"].append(extra_ref)
        elif mutation in {
            "plan_content_descriptor_drift",
            "plan_resolution_descriptor_drift",
            "resolution_descriptor_drift",
        }:
            record["payload"]["content_sha256"] = "0" * 64
        elif mutation == "plan_attachment_descriptor_drift":
            record["payload"]["attachments"][0]["artifact"]["sha256"] = (
                "0" * 64
            )
        elif mutation in {
            "plan_message_artifact_divergence",
            "input_message_artifact_divergence",
        }:
            record["payload"]["message"]["content"] = "drifted message"
        elif mutation in {
            "plan_resolution_artifact_divergence",
            "resolution_artifact_divergence",
        }:
            record["payload"]["submitted_by"] = "drifted-user"
        elif mutation == "handoff_full_output_descriptor_drift":
            record["payload"]["full_output_artifact"]["sha256"] = "0" * 64
        elif mutation == "handoff_envelope_artifact_ref_drift":
            record["payload"]["handoff_envelope"]["artifact_refs"].append(
                plan_input.resource_refs[0].to_dict()
            )
        elif mutation == "input_nonartifact_ref":
            record["resource_refs"][0]["kind"] = "checkpoint"
        elif mutation == "input_attachment_descriptor_drift":
            record["payload"]["attachments"][0]["artifact"]["sha256"] = (
                "0" * 64
            )
        else:
            raise AssertionError("artifact closure mutation is unsupported")

    _rewrite_event_record(store, targets[mutation], mutate)
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.retryable is False
    assert incompatible.value.reason == expected_reason
    assert _durable_authority_image(store) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "step0_child_attempt",
        "step0_source_range",
        "step0_artifact_refs",
        "step1_child_attempt",
        "step1_source_range",
        "step1_predecessor_ref",
    ],
)
def test_graph_handoff_source_provenance_is_exact_and_zero_write(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(
        store,
        initial,
        step_count=2,
    )
    for step in plan.steps:
        graph_service.start_step(plan, step.index)
        output = f"handoff source output {step.index}"
        _complete_graph_runtime(
            sinks[step.attempt],
            step.attempt,
            output=output,
        )
        graph_service.complete_step(
            plan,
            step.index,
            full_output={"output": output},
        )
    graph_service.finalize(plan)
    events = store.bind_execution(EXECUTION).capture_snapshot().events
    handoffs = tuple(
        next(
            event
            for event in events
            if event.attempt == step.attempt
            and event.event_type == "handoff.recorded"
        )
        for step in plan.steps
    )
    predecessor_seal = next(
        event
        for event in events
        if event.attempt == plan.steps[0].attempt
        and event.event_type == "graph.step.completed"
    )
    target_index = 0 if mutation.startswith("step0_") else 1

    def mutate(record: dict) -> None:
        envelope = record["payload"]["handoff_envelope"]
        if mutation == "step0_child_attempt":
            envelope["child_attempt"] = plan.steps[0].attempt.to_dict()
            envelope["child_run_id"] = plan.steps[0].attempt.attempt_id
        elif mutation == "step0_source_range":
            envelope["source_event_range"] = EventRange(
                EventCursor(handoffs[0].store_seq, handoffs[0].event_id),
                EventCursor(handoffs[0].store_seq, handoffs[0].event_id),
            ).to_dict()
        elif mutation == "step0_artifact_refs":
            predecessor_ref = predecessor_seal.resource_refs[0].to_dict()
            envelope["artifact_refs"].append(predecessor_ref)
            record["resource_refs"].append(predecessor_ref)
        elif mutation == "step1_child_attempt":
            envelope["child_attempt"] = plan.orchestration_attempt.to_dict()
            envelope["child_run_id"] = plan.orchestration_attempt.attempt_id
        elif mutation == "step1_source_range":
            envelope["source_event_range"] = EventRange(
                plan.initial_input_cursor,
                plan.initial_input_cursor,
            ).to_dict()
        elif mutation == "step1_predecessor_ref":
            envelope["artifact_refs"] = []
            record["resource_refs"] = [envelope["full_output_ref"]]
        else:
            raise AssertionError("handoff source mutation is unsupported")

    _rewrite_event_record(store, handoffs[target_index], mutate)
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.retryable is False
    assert incompatible.value.reason == "graph_step_seal_foreign"
    assert _durable_authority_image(store) == before


def test_graph_handoff_full_output_must_equal_its_exact_source(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    step = plan.steps[0]
    ingress = graph_service._derived_ingress_resolver(
        step.attempt,
        step.source_attempt,
    )
    handoff = ingress.persist(
        HostResolvedDerivedHandoffInput(
            consumer_attempt=step.attempt,
            source_attempt=step.source_attempt,
            status=HandoffStatus.COMPLETE,
            full_output={
                "schema": "unchain.graph_input_seed.v1",
                "input_event": {"malicious": True},
            },
            source_event_range=EventRange(
                plan.initial_input_cursor,
                plan.initial_input_cursor,
            ),
            operation_id="malicious-graph-source-handoff",
        )
    )
    JournalGraphCheckpointRepository(
        store.bind_execution(EXECUTION)
    ).start(plan, step, handoff)
    output = "output after malicious source handoff"
    _complete_graph_runtime(
        sinks[step.attempt],
        step.attempt,
        output=output,
    )
    graph_service.complete_step(
        plan,
        0,
        full_output={"output": output},
    )
    graph_service.finalize(plan)
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.retryable is False
    assert incompatible.value.reason == "graph_step_seal_foreign"
    assert _durable_authority_image(store) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "shape",
        "plan",
        "scope",
        "step",
        "attempt",
        "request_cursor",
        "resolution_cursor",
        "resources",
    ],
)
def test_foreign_graph_resume_admission_is_journal_incompatible(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, _sinks = _graph_checkpoint_runtime(
        store,
        initial,
        step_count=2,
    )
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    interaction_id = "foreign-resume-interaction"
    request_event = _append_event(
        store,
        initial,
        event_id="foreign-resume-request",
        event_type="interaction.requested",
        interaction_id=interaction_id,
        attempt_id=step.attempt.attempt_id,
    )
    resolution_event = _append_event(
        store,
        initial,
        event_id="foreign-resume-resolution",
        event_type="interaction.resolved",
        interaction_id=interaction_id,
        attempt_id=step.attempt.attempt_id,
    )
    payload = {
        "graph_plan_id": plan.plan_id,
        "graph_scope_id": plan.scope_id,
        "step": step.to_dict(),
        "interaction_id": interaction_id,
        "request_cursor": EventCursor(
            request_event.store_seq,
            request_event.event_id,
        ).to_dict(),
        "resolution_cursor": EventCursor(
            resolution_event.store_seq,
            resolution_event.event_id,
        ).to_dict(),
    }
    resource_refs: tuple[ResourceRef, ...] = ()
    resume_attempt = step.attempt
    if mutation == "shape":
        payload["foreign_field"] = "forbidden"
    elif mutation == "plan":
        payload["graph_plan_id"] = "foreign-graph-plan"
    elif mutation == "scope":
        payload["graph_scope_id"] = "foreign-graph-scope"
    elif mutation == "step":
        payload["step"] = plan.steps[1].to_dict()
    elif mutation == "attempt":
        resume_attempt = plan.steps[1].attempt
    elif mutation == "request_cursor":
        payload["request_cursor"] = plan.initial_input_cursor.to_dict()
    elif mutation == "resolution_cursor":
        payload["resolution_cursor"] = plan.initial_input_cursor.to_dict()
    elif mutation == "resources":
        start = next(
            event
            for event in store.bind_execution(EXECUTION).capture_snapshot().events
            if event.attempt == step.attempt
            and event.event_type == "graph.step.started"
        )
        resource_refs = start.resource_refs
    store.bind_execution(EXECUTION).append(
        request=SemanticEventDraft(
            event_id=f"foreign-resume-{mutation}",
            event_type="graph.step.resume.admitted",
            attempt=resume_attempt,
            operation_id=f"foreign-resume-operation-{mutation}",
            payload=payload,
            resource_refs=resource_refs,
        ).to_append_request()
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.reason in {
        "graph_attempt_kind_ambiguous",
        "graph_step_seal_foreign",
    }
    assert incompatible.value.retryable is False
    assert _durable_authority_image(store) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_same_evidence",
        "crossed_cycle_cursors",
        "second_cycle_mutation",
    ],
)
def test_noncanonical_graph_resume_cycles_are_nonretryable_and_zero_write(
    tmp_path: Path,
    mutation: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]

    def interaction_cycle(ordinal: int):
        interaction_id = f"resume-cycle-{ordinal}"
        request_event = _append_event(
            store,
            initial,
            event_id=f"resume-cycle-{ordinal}-request",
            event_type="interaction.requested",
            interaction_id=interaction_id,
            attempt_id=step.attempt.attempt_id,
        )
        resolution_event = _append_event(
            store,
            initial,
            event_id=f"resume-cycle-{ordinal}-resolution",
            event_type="interaction.resolved",
            interaction_id=interaction_id,
            attempt_id=step.attempt.attempt_id,
        )
        return interaction_id, request_event, resolution_event

    def append_resume(
        label: str,
        interaction_id: str,
        request_event,
        resolution_event,
    ) -> None:
        store.bind_execution(EXECUTION).append(
            request=SemanticEventDraft(
                event_id=f"resume-cycle-{mutation}-{label}",
                event_type="graph.step.resume.admitted",
                attempt=step.attempt,
                operation_id=f"resume-cycle-{mutation}-{label}-operation",
                payload={
                    "graph_plan_id": plan.plan_id,
                    "graph_scope_id": plan.scope_id,
                    "step": step.to_dict(),
                    "interaction_id": interaction_id,
                    "request_cursor": EventCursor(
                        request_event.store_seq,
                        request_event.event_id,
                    ).to_dict(),
                    "resolution_cursor": EventCursor(
                        resolution_event.store_seq,
                        resolution_event.event_id,
                    ).to_dict(),
                },
            ).to_append_request()
        )

    first_id, first_request, first_resolution = interaction_cycle(1)
    if mutation == "duplicate_same_evidence":
        append_resume("first", first_id, first_request, first_resolution)
        append_resume("duplicate", first_id, first_request, first_resolution)
    elif mutation == "crossed_cycle_cursors":
        second_id, second_request, second_resolution = interaction_cycle(2)
        append_resume("first", first_id, first_request, first_resolution)
        append_resume("second", second_id, second_request, second_resolution)
    else:
        append_resume("first", first_id, first_request, first_resolution)
        second_id, _second_request, second_resolution = interaction_cycle(2)
        append_resume("second", second_id, first_request, second_resolution)

    terminal = _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "run_failed",
        99,
        status="failed",
    )
    JournalGraphCheckpointRepository(
        store.bind_execution(EXECUTION)
    ).terminal(
        plan,
        step,
        status=GraphTerminalStatus.FAILED,
        terminal_cursor=terminal.cursor,
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.code == "journal_incompatible"
    assert incompatible.value.retryable is False
    assert incompatible.value.reason == "graph_step_seal_foreign"
    assert _durable_authority_image(store) == before


def test_two_canonical_graph_resume_cycles_allow_rebase(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    repository = JournalGraphCheckpointRepository(
        store.bind_execution(EXECUTION)
    )
    interaction_ingress = ContextInputIngress(
        attempt=step.attempt,
        projector=sinks[step.attempt].projector,
        sink=sinks[step.attempt],
    )
    for ordinal in (1, 2):
        interaction_id = f"canonical-resume-cycle-{ordinal}"
        request_event = _append_event(
            store,
            initial,
            event_id=f"canonical-resume-cycle-{ordinal}-request",
            event_type="interaction.requested",
            interaction_id=interaction_id,
            attempt_id=step.attempt.attempt_id,
        )
        resolution_event = interaction_ingress.persist(
            HostResolvedInteractionInput(
                attempt=step.attempt,
                interaction_id=interaction_id,
                response={"answer": f"response-{ordinal}"},
            )
        )
        repository.resume(
            plan,
            step,
            interaction_id=interaction_id,
            request_cursor=EventCursor(
                request_event.store_seq,
                request_event.event_id,
            ),
            resolution_cursor=resolution_event.cursor,
        )
    terminal = _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "run_failed",
        99,
        status="failed",
    )
    repository.terminal(
        plan,
        step,
        status=GraphTerminalStatus.FAILED,
        terminal_cursor=terminal.cursor,
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert service.rebase(edit_request).head_revision == 2


def test_boundary_resolved_live_graph_cycles_allow_rebase(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    repository = JournalGraphCheckpointRepository(
        store.bind_execution(EXECUTION)
    )
    interaction_ingress = ContextInputIngress(
        attempt=step.attempt,
        projector=sinks[step.attempt].projector,
        sink=sinks[step.attempt],
    )

    def live_cycle(ordinal: int) -> None:
        interaction_id = f"live-cycle-{ordinal}"
        _append_event(
            store,
            initial,
            event_id=f"live-cycle-{ordinal}-request",
            event_type="interaction.requested",
            interaction_id=interaction_id,
            attempt_id=step.attempt.attempt_id,
        )
        interaction_ingress.persist(
            HostResolvedInteractionInput(
                attempt=step.attempt,
                interaction_id=interaction_id,
                response={"answer": f"response-{ordinal}"},
            )
        )

    live_cycle(1)
    _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "iteration_started",
        17,
        iteration=2,
    )
    live_cycle(2)
    terminal = _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "run_failed",
        99,
        status="failed",
    )
    repository.terminal(
        plan,
        step,
        status=GraphTerminalStatus.FAILED,
        terminal_cursor=terminal.cursor,
    )

    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert service.rebase(edit_request).head_revision == 2


def test_live_tool_outcomes_without_admission_allow_rebase(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    repository = JournalGraphCheckpointRepository(
        store.bind_execution(EXECUTION)
    )
    for ordinal, outcome in ((1, "tool_confirmed"), (2, "tool_denied")):
        interaction_id = f"live-tool-{ordinal}"
        _append_event(
            store,
            initial,
            event_id=f"live-tool-{ordinal}-request",
            event_type="tool_confirmation_requested",
            interaction_id=interaction_id,
            attempt_id=step.attempt.attempt_id,
        )
        _append_event(
            store,
            initial,
            event_id=f"live-tool-{ordinal}-outcome",
            event_type=outcome,
            interaction_id=interaction_id,
            attempt_id=step.attempt.attempt_id,
        )
    terminal = _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "run_failed",
        99,
        status="failed",
    )
    repository.terminal(
        plan,
        step,
        status=GraphTerminalStatus.FAILED,
        terminal_cursor=terminal.cursor,
    )

    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert service.rebase(edit_request).head_revision == 2


def test_late_admission_keeps_durable_cycle_strict(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    repository = JournalGraphCheckpointRepository(
        store.bind_execution(EXECUTION)
    )
    interaction_ingress = ContextInputIngress(
        attempt=step.attempt,
        projector=sinks[step.attempt].projector,
        sink=sinks[step.attempt],
    )
    first_request = _append_event(
        store,
        initial,
        event_id="late-admission-first-request",
        event_type="interaction.requested",
        interaction_id="late-admission-first",
        attempt_id=step.attempt.attempt_id,
    )
    first_resolution = interaction_ingress.persist(
        HostResolvedInteractionInput(
            attempt=step.attempt,
            interaction_id="late-admission-first",
            response={"answer": "first"},
        )
    )
    _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "iteration_started",
        17,
        iteration=2,
    )
    _append_event(
        store,
        initial,
        event_id="late-admission-second-request",
        event_type="interaction.requested",
        interaction_id="late-admission-second",
        attempt_id=step.attempt.attempt_id,
    )
    interaction_ingress.persist(
        HostResolvedInteractionInput(
            attempt=step.attempt,
            interaction_id="late-admission-second",
            response={"answer": "second"},
        )
    )
    repository.resume(
        plan,
        step,
        interaction_id="late-admission-first",
        request_cursor=EventCursor(
            first_request.store_seq,
            first_request.event_id,
        ),
        resolution_cursor=first_resolution.cursor,
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )
    before = _durable_authority_image(store)

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.reason == "graph_step_seal_foreign"
    assert incompatible.value.retryable is False
    assert _durable_authority_image(store) == before


def test_multistep_graph_verification_does_not_retain_output_bytes(
    tmp_path: Path,
) -> None:
    assert (
        "output_bytes"
        not in generation_rebase_module._GraphStepQuiescence.__dataclass_fields__
    )
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(
        store,
        initial,
        step_count=3,
    )
    for step in plan.steps:
        graph_service.start_step(plan, step.index)
        output = f"verified output for step {step.index}"
        _complete_graph_runtime(
            sinks[step.attempt],
            step.attempt,
            output=output,
        )
        graph_service.complete_step(
            plan,
            step.index,
            full_output={"output": output},
        )
    graph_service.finalize(plan)
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert service.rebase(edit_request).head_revision == 2


def test_real_root_terminal_and_dead_graph_prefix_are_quiescent(
    tmp_path: Path,
) -> None:
    root_store = _store(tmp_path / "root")
    root_rebase = SQLiteGenerationRebaseV2Service(root_store)
    root_initial = root_rebase.rebase(
        _request(_intent(), operation_id="rebase-operation-root")
    )
    root_graph, root_plan, root_sinks = _graph_checkpoint_runtime(
        root_store,
        root_initial,
    )
    root_graph.start_step(root_plan, 0)
    root_step = root_plan.steps[0]
    _complete_graph_runtime(
        root_sinks[root_step.attempt],
        root_step.attempt,
        output="root output",
    )
    root_graph.complete_step(
        root_plan,
        0,
        full_output={"output": "root output"},
    )
    root_graph.finalize(root_plan)
    _append_graph_runtime_event(
        root_sinks[root_plan.orchestration_attempt],
        root_plan.orchestration_attempt,
        "run_completed",
        10,
        status="completed",
    )
    _root_intent, root_request = _next(
        root_initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert root_rebase.rebase(root_request).head_revision == 2

    dead_store = _store(tmp_path / "dead")
    dead_rebase = SQLiteGenerationRebaseV2Service(dead_store)
    dead_initial = dead_rebase.rebase(
        _request(_intent(), operation_id="rebase-operation-dead")
    )
    dead_graph, dead_plan, dead_sinks = _graph_checkpoint_runtime(
        dead_store,
        dead_initial,
    )
    dead_graph.start_step(dead_plan, 0)
    dead_step = dead_plan.steps[0]
    _append_graph_runtime_event(
        dead_sinks[dead_step.attempt],
        dead_step.attempt,
        "run_failed",
        1,
        status="failed",
    )
    dead_graph.recover(dead_plan)
    _dead_intent, dead_request = _next(
        dead_initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert dead_rebase.rebase(dead_request).head_revision == 2


def test_max_iterations_never_suppresses_pending_interaction(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _append_event(
        store,
        initial,
        event_id="interaction-request-before-max",
        event_type="interaction.requested",
        interaction_id="interaction-before-max",
        attempt_id="attempt-max-wait",
    )
    _append_event(
        store,
        initial,
        event_id="run-max-iterations-wait",
        event_type="run_max_iterations",
        attempt_id="attempt-max-wait",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    with pytest.raises(GenerationRebasePreflightBlocked) as blocked_info:
        service.rebase(edit_request)

    assert blocked_info.value.reason == "pending_interaction"


def test_max_iterations_is_narrow_terminal_equivalent(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    for event_id, event_type in (
        ("run-started-extended", "run_started"),
        ("run-max-extended", "run_max_iterations"),
        ("run-completed-extended", "run_completed"),
    ):
        _append_event(
            store,
            initial,
            event_id=event_id,
            event_type=event_type,
            attempt_id="attempt-extended",
        )
    _append_event(
        store,
        initial,
        event_id="run-max-final",
        event_type="run.max_iterations",
        attempt_id="attempt-max-final",
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert service.rebase(edit_request).head_revision == 2


@pytest.mark.parametrize(
    "terminal_type",
    ["run_max_iterations", "run_cancelled"],
)
def test_real_graph_terminal_families_form_a_dead_prefix(
    tmp_path: Path,
    terminal_type: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        terminal_type,
        1,
        status="terminal",
    )
    graph_service.recover(plan)
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    assert service.rebase(edit_request).head_revision == 2


def test_graph_seal_followed_by_any_event_is_journal_incompatible(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    _complete_graph_runtime(
        sinks[step.attempt],
        step.attempt,
        output="sealed output",
    )
    graph_service.complete_step(
        plan,
        0,
        full_output={"output": "sealed output"},
    )
    _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "runtime.after_seal",
        10,
    )
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.reason == "graph_step_seal_not_last"


@pytest.mark.parametrize(
    ("mutation", "terminal_type", "expected_reason"),
    (
        (
            "status",
            "run_completed",
            "graph_step_seal_mismatched_terminal",
        ),
        ("cursor", "run_failed", "graph_step_seal_foreign"),
        ("scope", "run_failed", "graph_step_seal_foreign"),
    ),
)
def test_graph_seal_identity_and_status_mismatches_fail_closed(
    tmp_path: Path,
    mutation: str,
    terminal_type: str,
    expected_reason: str,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    terminal = _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        terminal_type,
        1,
        status="terminal",
    )
    terminal_cursor = terminal.cursor
    graph_scope_id = plan.scope_id
    if mutation == "cursor":
        terminal_cursor = EventCursor(
            terminal.cursor.store_seq,
            "mismatched-terminal-event",
        )
    elif mutation == "scope":
        graph_scope_id = "mismatched-graph-scope"
    seal = SemanticEventDraft(
        event_id=f"invalid-seal-{mutation}",
        event_type="graph.step.failed",
        attempt=step.attempt,
        operation_id=f"invalid-seal-operation-{mutation}",
        payload={
            "graph_plan_id": plan.plan_id,
            "graph_scope_id": graph_scope_id,
            "step": step.to_dict(),
            "terminal_cursor": terminal_cursor.to_dict(),
        },
    )
    store.bind_execution(EXECUTION).append(request=seal.to_append_request())
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.reason == expected_reason


def test_completed_graph_seal_execution_range_must_start_at_step_start(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    graph_service, plan, sinks = _graph_checkpoint_runtime(store, initial)
    graph_service.start_step(plan, 0)
    step = plan.steps[0]
    _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "final_message",
        1,
        content="range output",
    )
    terminal = _append_graph_runtime_event(
        sinks[step.attempt],
        step.attempt,
        "run_completed",
        2,
        status="completed",
    )
    journal = store.bind_execution(EXECUTION)
    artifact = ArtifactService(
        journal,
        sanitizer=lambda content, _media_type: content,
    ).persist_exact_json(
        {"output": "range output"},
        operation_id="invalid-range-artifact",
    )
    seal = SemanticEventDraft(
        event_id="invalid-range-seal",
        event_type="graph.step.completed",
        attempt=step.attempt,
        operation_id="invalid-range-seal-operation",
        payload={
            "graph_plan_id": plan.plan_id,
            "graph_scope_id": plan.scope_id,
            "step": step.to_dict(),
            "terminal_cursor": terminal.cursor.to_dict(),
            "execution_event_range": EventRange(
                terminal.cursor,
                terminal.cursor,
            ).to_dict(),
            "output_artifact": artifact.to_dict(),
        },
        resource_refs=(artifact.ref,),
    )
    journal.append(request=seal.to_append_request())
    _edit_intent, edit_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
    )

    with pytest.raises(GenerationRebaseJournalIncompatible) as incompatible:
        service.rebase(edit_request)

    assert incompatible.value.reason == "graph_step_sequence_invalid"


def test_empty_snapshot_uses_a_marker_and_survives_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(_intent(), operation_id="rebase-operation-1")
    )
    _empty_intent, empty_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
        messages=(),
    )

    emptied = service.rebase(empty_request)

    assert emptied.message_count == 0
    assert emptied.first_cursor == emptied.last_cursor
    marker = store.bind_execution(EXECUTION).capture_snapshot().events[-1]
    assert marker.event_type == "generation.rebased"
    assert marker.attempt.generation.generation_id == emptied.generation_id
    assert marker.payload["generation_rebase"]["empty_snapshot"] is True
    assert marker.payload["generation_rebase"]["replacement_message_count"] == 0
    assert "message" not in marker.payload
    reopened = SQLiteGenerationRebaseV2Service(_store(tmp_path))
    assert reopened.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == emptied.generation_id
    assert reopened.receipt_for_generation(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
        generation_id=emptied.generation_id,
    ) == emptied
    assert reopened.rebase(empty_request) == replace(emptied, duplicate=True)


def test_empty_initial_snapshot_has_a_durable_generation_head(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    intent = _intent(messages=())

    receipt = service.rebase(
        _request(intent, operation_id="rebase-operation-empty-create")
    )

    assert receipt.kind is GenerationRebaseKind.CREATE
    assert receipt.message_count == 0
    assert service.current(
        owner_chat_id=OWNER,
        execution_id=EXECUTION,
        session_id=SESSION,
    ).current_generation_id == receipt.generation_id
    snapshot = store.bind_execution(EXECUTION).capture_snapshot()
    assert len(snapshot.events) == 1
    assert snapshot.events[0].event_type == "generation.rebased"


def test_compiler_on_empty_current_generation_does_not_restore_old_messages(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    service = SQLiteGenerationRebaseV2Service(store)
    initial = service.rebase(
        _request(
            _intent(
                messages=_messages(
                    "generation-1",
                    ("user", "old prompt must remain archived"),
                    ("assistant", "old answer must remain archived"),
                )
            ),
            operation_id="rebase-operation-1",
        )
    )
    _empty_intent, empty_request = _next(
        initial,
        kind=GenerationRebaseKind.EDIT,
        ordinal=2,
        messages=(),
    )
    emptied = service.rebase(empty_request)
    journal = store.bind_execution(EXECUTION)
    current_user = journal.append(
        request=SemanticEventDraft(
            event_id="message-after-empty-rebase",
            event_type="message.user",
            attempt=AttemptRef(
                GenerationRef(EXECUTION, emptied.generation_id),
                emptied.attempt_id,
            ),
            operation_id="operation-message-after-empty-rebase",
            payload={
                "run_id": emptied.attempt_id,
                "message": {
                    "role": "user",
                    "content": "new prompt after clearing history",
                },
            },
        ).to_append_request()
    ).event
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    capabilities = SQLiteContextCompilerV2Store(
        context_store=store,
    ).bind_execution(
        EXECUTION,
        artifacts=artifacts,
    )
    request = ContextCompileRequest(
        case="generation-rebase-empty-current",
        source_messages=(
            {"role": "system", "content": "current policy"},
            {"role": "user", "content": "new prompt after clearing history"},
        ),
        current_generation=emptied.generation_id,
        budget=resolve_context_budget(context_window_tokens=8_192),
        source_message_cursors=(
            SourceMessageCursor(
                1,
                current_user.event_id,
                current_user.store_seq,
            ),
        ),
        provider="openai",
        model="synthetic",
        build_id="generation-rebase-empty-build",
        execution_id=EXECUTION,
        generation_id=emptied.generation_id,
        attempt_id=emptied.attempt_id,
    )

    result = ContextCompileCoordinator(
        journal=journal,
        checkpoint_repository=capabilities.checkpoints,
        build_repository=capabilities.context_builds,
        partial_attempt_sink=lambda request, error: None,
    ).compile(request)

    contents = [message.get("content") for message in result.messages]
    assert "current policy" in contents
    assert "new prompt after clearing history" in contents
    assert "old prompt must remain archived" not in contents
    assert "old answer must remain archived" not in contents
    assert all(message.get("role") != "generation.rebased" for message in result.messages)
