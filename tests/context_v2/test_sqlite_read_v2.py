from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path

import pytest

from unchain.context import ArtifactService
from unchain.journal import (
    AttemptRef,
    EventCursor,
    EventRange,
    GenerationRef,
    JournalAppendRequest,
    OperationRef,
    ResourceRef,
)
from unchain.journal.runtime import build_operation_ref
from unchain.memory.workspace import MemorySpace, MemoryWorkspaceService
from unchain.memory.workspace.ports import (
    BoundWorkspaceReferenceAuthorizer,
    RepositoryScopeError,
)
from unchain.memory.toolkit import MemoryToolContentPage
from unchain.persistence.sqlite_memory_v2 import SQLiteMemoryV2Store
from unchain.persistence.sqlite_context_compiler_v2 import (
    SQLiteContextCompilerV2Store,
)
from unchain.persistence.sqlite_read_v2 import (
    ContextV2ReadScope,
    SQLiteContextV2ReadScopeError,
    SQLiteContextV2ReadService,
    read_sqlite_context_v2_store_status,
)
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


class _References(BoundWorkspaceReferenceAuthorizer):
    def authorize(self, *, ref: ResourceRef) -> ResourceRef:
        return ref


def _append(
    repository,
    *,
    execution_id: str,
    index: int,
    attempt_id: str | None = None,
):
    return repository.append(
        request=JournalAppendRequest(
            event_id=f"event-{execution_id}-{index}",
            event_type="message.user",
            attempt=AttemptRef(
                GenerationRef(execution_id, f"generation-{execution_id}"),
                attempt_id or f"attempt-{execution_id}",
            ),
            operation=OperationRef(
                f"operation-{execution_id}-{index}", f"{index:x}" * 64
            ),
            payload={"content": f"message-{execution_id}-{index}"},
        )
    )


def _workspace(
    memory: SQLiteMemoryV2Store,
    *,
    owner_chat_id: str,
    space_id: str,
    event_ref: ResourceRef,
):
    repository = memory.bind_workspace(
        space=MemorySpace(
            space_id,
            "chat",
            f"Workspace {space_id}",
            f"Durable workspace for {owner_chat_id}",
            1,
        ),
        owner_chat_id=owner_chat_id,
    )
    service = MemoryWorkspaceService(
        repository=repository,
        mutations=repository,
        content=repository,
        history=repository,
        links=repository,
        references=_References(f"binding-{space_id}"),
    )
    entry = service.write_markdown(
        path="/notes/Architecture.md",
        description=f"Needle design for {owner_chat_id}",
        content=f"private content for {owner_chat_id}",
        expected_space_revision=1,
        source_refs=(event_ref,),
        operation_id=f"write-{space_id}",
    )
    return entry


def _seed(root: Path):
    database_path = root / "context_v2.sqlite3"
    object_directory = root / "objects"
    context = SQLiteContextV2Store(
        database_path=database_path,
        object_directory=object_directory,
    )
    memory = SQLiteMemoryV2Store(
        database_path=database_path,
        object_directory=object_directory,
    )
    event_refs = {}
    for execution_id in ("execution-a", "execution-b"):
        repository = context.bind_execution(execution_id)
        first = _append(repository, execution_id=execution_id, index=1)
        _append(repository, execution_id=execution_id, index=2)
        event_refs[execution_id] = ResourceRef(
            "context_event",
            first.event.event_id,
            1,
        )
    entry_a = _workspace(
        memory,
        owner_chat_id="chat-a",
        space_id="space-a",
        event_ref=event_refs["execution-a"],
    )
    entry_b = _workspace(
        memory,
        owner_chat_id="chat-b",
        space_id="space-b",
        event_ref=event_refs["execution-b"],
    )
    return database_path, object_directory, context, memory, entry_a, entry_b


def _bind(
    context: SQLiteContextV2Store,
    memory: SQLiteMemoryV2Store,
    *,
    owner_chat_id: str = "chat-a",
    execution_ids=("execution-a",),
    space_id: str = "space-a",
):
    return SQLiteContextV2ReadService(
        context_store=context,
        memory_store=memory,
    ).bind(
        ContextV2ReadScope(
            owner_chat_id=owner_chat_id,
            execution_ids=execution_ids,
            space_id=space_id,
        )
    )


def test_read_service_binds_exact_owner_execution_and_workspace_scope(
    tmp_path: Path,
) -> None:
    _, _, context, memory, entry_a, entry_b = _seed(tmp_path)
    reader = _bind(context, memory)

    first = reader.read_events(execution_id="execution-a", limit=1)
    second = reader.read_events(
        execution_id="execution-a",
        after=first.next_cursor,
        limit=1,
    )

    assert [event.event_id for event in first.events] == ["event-execution-a-1"]
    assert first.has_more is True
    assert [event.event_id for event in second.events] == ["event-execution-a-2"]
    assert second.has_more is False
    assert reader.list_workspace().entries == ()
    assert reader.workspace_tree().entries == (entry_a,)
    assert reader.get_workspace_entry(entry_id=entry_a.entry_id) == entry_a
    with pytest.raises(SQLiteContextV2ReadScopeError, match="execution"):
        reader.read_events(execution_id="execution-b")
    with pytest.raises(RepositoryScopeError, match="workspace|scope"):
        reader.get_workspace_entry(
            ref=ResourceRef("memory", entry_b.entry_id, entry_b.revision, "space-b")
        )
    with pytest.raises(SQLiteContextV2ReadScopeError, match="owner|workspace|scope"):
        _bind(
            context,
            memory,
            owner_chat_id="chat-a",
            execution_ids=("execution-a",),
            space_id="space-b",
        )


def test_route_cursor_reads_after_store_seq_without_requiring_an_event_id(
    tmp_path: Path,
) -> None:
    _, _, context, memory, entry_a, _ = _seed(tmp_path)
    _append(
        context.bind_execution("execution-a"),
        execution_id="execution-a",
        index=3,
        attempt_id="attempt-other",
    )
    reader = _bind(context, memory)

    page = reader.read_events_after_store_seq(
        execution_id="execution-a",
        after_store_seq=1,
        limit=1,
    )
    exhausted = reader.read_events_after_store_seq(
        execution_id="execution-a",
        after_store_seq=999,
        limit=1,
    )
    filtered = reader.read_events_after_store_seq(
        execution_id="execution-a",
        after_store_seq=0,
        limit=10,
        attempt_id="attempt-other",
    )

    assert [event.event_id for event in page.events] == ["event-execution-a-2"]
    assert page.next_cursor is not None
    assert page.next_cursor.store_seq == 2
    assert page.has_more is True
    assert exhausted.events == ()
    assert exhausted.next_cursor is None
    assert exhausted.has_more is False
    assert [event.event_id for event in filtered.events] == ["event-execution-a-3"]
    assert reader.workspace_space == MemorySpace(
        "space-a",
        "chat",
        "Workspace space-a",
        "Durable workspace for chat-a",
        entry_a.updated_seq,
    )
    with pytest.raises(SQLiteContextV2ReadScopeError, match="execution"):
        reader.read_events_after_store_seq(
            execution_id="execution-b",
            after_store_seq=0,
            limit=1,
        )


def test_store_status_is_database_scoped_read_only_and_fails_closed(
    tmp_path: Path,
) -> None:
    database, _, _, _, _, _ = _seed(tmp_path)

    status = read_sqlite_context_v2_store_status(database)

    assert status.to_dict() == {
        "schema": "unchain.sqlite_context_v2_store_read_status.v1",
        "available": True,
        "schema_version": 3,
        "journal_mode": "wal",
        "lexical_backend": "fts5",
        "vector_status": "disabled",
    }
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(Exception, match="unavailable|database"):
        read_sqlite_context_v2_store_status(missing)
    assert not missing.exists()


def test_verified_artifact_pages_and_workspace_reads_survive_restart(
    tmp_path: Path,
) -> None:
    database, objects, context, memory, entry_a, _ = _seed(tmp_path)
    artifact = ArtifactService(
        context.bind_execution("execution-a"),
        sanitizer=lambda content, media_type: content,
    ).persist(
        b"0123456789",
        media_type="text/plain",
        operation_id="artifact-a",
    )

    first = _bind(context, memory).read_artifact_page(
        execution_id="execution-a",
        artifact=artifact,
        offset=0,
        limit=4,
    )
    reopened = _bind(
        SQLiteContextV2Store(database_path=database, object_directory=objects),
        SQLiteMemoryV2Store(database_path=database, object_directory=objects),
    )
    second = reopened.read_artifact_page(
        execution_id="execution-a",
        artifact=artifact,
        offset=first.next_offset,
        limit=6,
    )

    assert first.data == b"0123"
    assert first.has_more is True
    assert second.data == b"456789"
    assert second.has_more is False
    assert reopened.get_workspace_entry(entry_id=entry_a.entry_id) == entry_a
    entry_ref = ResourceRef(
        "memory",
        entry_a.entry_id,
        entry_a.revision,
        entry_a.space_id,
    )
    memory_first = reopened.read_workspace_content(
        ref=entry_ref,
        offset=0,
        limit=8,
    )
    memory_second = reopened.read_workspace_content(
        ref=entry_ref,
        offset=memory_first.next_offset,
        limit=64,
    )
    assert memory_first.data == b"private "
    assert memory_first.has_more is True
    assert memory_second.data == b"content for chat-a"
    assert memory_second.has_more is False
    with pytest.raises(SQLiteContextV2ReadScopeError, match="execution"):
        reopened.read_artifact_page(
            execution_id="execution-b",
            artifact=artifact,
        )


def test_unique_artifact_and_checkpoint_reads_use_only_bound_executions(
    tmp_path: Path,
) -> None:
    _, _, context, memory, _, _ = _seed(tmp_path)
    journal = context.bind_execution("execution-a")
    artifact_service = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    artifact = artifact_service.persist(
        b"scoped artifact payload",
        media_type="text/plain",
        operation_id="scoped-artifact-a",
    )
    event = journal.read(limit=1).events[0]
    cursor = EventCursor(event.store_seq, event.event_id)
    source_range = EventRange(
        cursor,
        cursor,
    )
    compiler = SQLiteContextCompilerV2Store(context_store=context)
    checkpoints = compiler.bind_execution(
        "execution-a",
        artifacts=artifact_service,
    ).checkpoints
    operation = build_operation_ref(
        "scoped-checkpoint-a",
        domain="test.sqlite_read_v2",
        payload={"source_range": source_range.to_dict()},
    )
    prepared = checkpoints.prepare(
        source_range=source_range,
        summary="checkpoint payload for route",
        refs=(),
        operation=operation,
    )
    committed = checkpoints.commit(prepared=prepared)
    reader = SQLiteContextV2ReadService(
        context_store=context,
        memory_store=memory,
        compiler_store=compiler,
    ).bind(
        ContextV2ReadScope(
            owner_chat_id="chat-a",
            execution_ids=("execution-a",),
            space_id="space-a",
        )
    )

    artifact_page = reader.read_unique_artifact(
        ref=artifact.ref,
        offset=7,
        limit=8,
    )
    checkpoint_page = reader.read_unique_checkpoint(
        ref=committed.checkpoint_ref,
        offset=11,
        limit=7,
    )

    assert artifact_page.data == b"artifact"
    assert artifact_page.sha256 == artifact.sha256
    assert checkpoint_page.data == b"payload"
    assert checkpoint_page.ref == committed.checkpoint_ref
    assert checkpoint_page.has_more is True
    with pytest.raises(SQLiteContextV2ReadScopeError, match="artifact|scope"):
        reader.read_unique_artifact(
            ref=ResourceRef("artifact", "artifact-not-owned", 1)
        )
    with pytest.raises(SQLiteContextV2ReadScopeError, match="checkpoint|scope"):
        reader.read_unique_checkpoint(
            ref=ResourceRef("checkpoint", "checkpoint-not-owned", 1)
        )


def test_public_context_capability_reads_scoped_content_and_checkpoint_events(
    tmp_path: Path,
) -> None:
    _, _, context, memory, _, _ = _seed(tmp_path)
    journal = context.bind_execution("execution-a")
    artifact_service = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    artifact = artifact_service.persist(
        b"durable artifact payload",
        media_type="text/plain",
        operation_id="public-reader-artifact",
    )
    events = journal.read(limit=10).events
    source_range = EventRange(
        EventCursor(events[0].store_seq, events[0].event_id),
        EventCursor(events[-1].store_seq, events[-1].event_id),
    )
    compiler = SQLiteContextCompilerV2Store(context_store=context)
    checkpoints = compiler.bind_execution(
        "execution-a",
        artifacts=artifact_service,
    ).checkpoints
    operation = build_operation_ref(
        "public-reader-checkpoint",
        domain="test.sqlite_read_v2",
        payload={"source_range": source_range.to_dict()},
    )
    checkpoint = checkpoints.commit(
        prepared=checkpoints.prepare(
            source_range=source_range,
            summary="durable checkpoint summary",
            refs=(artifact.ref,),
            operation=operation,
        )
    )
    reader = SQLiteContextV2ReadService(
        context_store=context,
        memory_store=memory,
        compiler_store=compiler,
    ).bind(
        ContextV2ReadScope(
            owner_chat_id="chat-a",
            execution_ids=("execution-a",),
            space_id="space-a",
        )
    )

    artifact_page = reader.read_content(
        ref=artifact.ref,
        offset=8,
        limit=8,
    )
    event_ref = ResourceRef("context_event", events[0].event_id, 1)
    event_page = reader.read_content(
        ref=event_ref,
        offset=0,
        limit=65_536,
    )
    event_content_page = reader.read_content(
        ref=ResourceRef("context_event", events[0].event_id, 1, "content"),
        offset=8,
        limit=9,
    )
    checkpoint_page = reader.read_content(
        ref=checkpoint.checkpoint_ref,
        offset=8,
        limit=10,
    )
    first_events = reader.read_checkpoint_events(
        ref=checkpoint.checkpoint_ref,
        after_position=0,
        limit=1,
    )
    second_events = reader.read_checkpoint_events(
        ref=checkpoint.checkpoint_ref,
        after_position=1,
        limit=1,
    )
    derived_ref = first_events["events"][0]["content_ref"]
    derived_page = reader.read_content(
        ref=derived_ref,
        offset=0,
        limit=65_536,
    )

    assert isinstance(artifact_page, MemoryToolContentPage)
    assert artifact_page.data == b"artifact"
    assert artifact_page.sha256 == artifact.sha256
    assert event_page.media_type == "application/json"
    assert json.loads(event_page.data)["event_id"] == events[0].event_id
    assert event_content_page.media_type == "text/plain"
    assert event_content_page.data == b"execution"
    assert checkpoint_page.data == b"checkpoint"
    assert isinstance(first_events, Mapping)
    assert first_events["checkpoint_ref"] == checkpoint.checkpoint_ref
    assert first_events["coverage"] == {
        "first_store_seq": events[0].store_seq,
        "last_store_seq": events[-1].store_seq,
        "ceiling_position": 2,
    }
    assert first_events["after_position"] == 0
    assert first_events["next_after_position"] == 1
    assert first_events["has_more"] is True
    assert first_events["events"] == (
        {
            "position": 1,
            "store_seq": events[0].store_seq,
            "event_type": events[0].event_type,
            "event_ref": event_ref,
            "content_ref": ResourceRef(
                "checkpoint",
                checkpoint.checkpoint_ref.resource_id,
                1,
                "event/1",
            ),
        },
    )
    assert second_events["next_after_position"] == 2
    assert second_events["has_more"] is False
    assert json.loads(derived_page.data)["event_id"] == events[0].event_id
    assert reader.authorize_context_ref(ref=artifact.ref) == artifact.ref
    with pytest.raises(SQLiteContextV2ReadScopeError, match="event|scope"):
        reader.authorize_context_ref(
            ref=ResourceRef("context_event", "event-execution-b-1", 1)
        )


def test_unique_artifact_resolution_fails_closed_on_cross_execution_ambiguity(
    tmp_path: Path,
) -> None:
    database, _, context, memory, _, _ = _seed(tmp_path)
    artifact_a = ArtifactService(
        context.bind_execution("execution-a"),
        sanitizer=lambda content, media_type: content,
    ).persist(
        b"artifact a",
        media_type="text/plain",
        operation_id="ambiguous-artifact-a",
    )
    ArtifactService(
        context.bind_execution("execution-b"),
        sanitizer=lambda content, media_type: content,
    ).persist(
        b"artifact b",
        media_type="text/plain",
        operation_id="ambiguous-artifact-b",
    )
    with sqlite3.connect(database) as connection:
        descriptor = connection.execute(
            "SELECT artifact_json, artifact_record_sha256 FROM artifacts "
            "WHERE execution_id = 'execution-a' AND artifact_id = ?",
            (artifact_a.ref.resource_id,),
        ).fetchone()
        connection.execute(
            "UPDATE artifacts SET artifact_id = ?, artifact_json = ?, "
            "artifact_record_sha256 = ? WHERE execution_id = 'execution-b'",
            (artifact_a.ref.resource_id, descriptor[0], descriptor[1]),
        )
    reader = SQLiteContextV2ReadService(
        context_store=context,
        memory_store=memory,
    ).bind(
        ContextV2ReadScope(
            owner_chat_id="chat-a",
            execution_ids=("execution-a", "execution-b"),
            space_id="space-a",
        )
    )

    with pytest.raises(SQLiteContextV2ReadScopeError, match="ambiguous"):
        reader.read_unique_artifact(ref=artifact_a.ref)


def test_workspace_search_degrades_to_bounded_lexical_scan_when_fts_is_offline(
    tmp_path: Path,
) -> None:
    _, _, context, memory, entry_a, _ = _seed(tmp_path)
    memory._fts_available = False

    result = _bind(context, memory).search_workspace("Needle", limit=5)

    assert [hit.entry for hit in result.hits] == [entry_a]
    assert result.lexical_fallback is True
    assert "lexical_fallback" in result.hits[0].matched_by


def test_status_is_scope_bound_and_does_not_expose_host_paths(tmp_path: Path) -> None:
    _, _, context, memory, _, _ = _seed(tmp_path)

    status = _bind(context, memory).status().to_dict()

    assert status == {
        "schema": "unchain.sqlite_context_v2_read_status.v1",
        "available": True,
        "owner_chat_id": "chat-a",
        "execution_count": 1,
        "space_id": "space-a",
        "space_revision": 2,
        "journal": "available",
        "artifacts": "available",
        "workspace": "available",
        "search": "ready",
    }
    assert str(tmp_path) not in repr(status)
