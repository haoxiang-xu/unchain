from __future__ import annotations

import copy
import json
from collections.abc import Mapping

import pytest

import unchain.journal as journal_api

from unchain.journal.models import (
    AttemptRef,
    EventCursor,
    GenerationRef,
    JournalAppendRequest,
    JournalAppendResult,
    JournalEvent,
    JournalPage,
    OperationRef,
    ResourceRef,
)
from unchain.journal.ports import (
    BoundExecutionJournal,
    JournalConflictError,
    JournalRepositoryError,
    JournalScopeError,
)
from unchain.journal.runtime import (
    DurableEventSink,
    DurableJournalIntegrityError,
    DurableJournalScopeError,
    SemanticEventDraft,
    journal_event_to_semantic_event,
)
from unchain.journal.snapshot import capture_journal_snapshot


class MemoryJournal(BoundExecutionJournal):
    def __init__(self, execution_id: str) -> None:
        super().__init__(execution_id)
        self.events: list[JournalEvent] = []
        self.operations: dict[str, tuple[str, JournalEvent]] = {}
        self.requests: list[JournalAppendRequest] = []
        self.fail_append: Exception | None = None
        self.lie_about_payload = False

    def append(self, *, request: JournalAppendRequest) -> JournalAppendResult:
        if self.fail_append is not None:
            raise self.fail_append
        if request.attempt.generation.execution_id != self.execution_id:
            raise JournalScopeError("foreign execution")
        self.requests.append(request)
        previous = self.operations.get(request.operation.operation_id)
        if previous is not None:
            prior_hash, prior_event = previous
            if prior_hash != request.operation.payload_sha256:
                raise JournalConflictError("operation payload changed")
            cursor = EventCursor(prior_event.store_seq, prior_event.event_id)
            return JournalAppendResult(prior_event, cursor, duplicate=True)
        payload = dict(request.payload)
        if self.lie_about_payload:
            payload["altered"] = True
        event = JournalEvent(
            event_id=request.event_id,
            event_type=request.event_type,
            attempt=request.attempt,
            operation=request.operation,
            store_seq=len(self.events) + 1,
            payload=payload,
            resource_refs=request.resource_refs,
        )
        self.events.append(event)
        self.operations[request.operation.operation_id] = (
            request.operation.payload_sha256,
            event,
        )
        cursor = EventCursor(event.store_seq, event.event_id)
        return JournalAppendResult(event, cursor, duplicate=False)

    def read(self, *, after: EventCursor | None = None, limit: int = 100) -> JournalPage:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("invalid limit")
        offset = after.store_seq if after is not None else 0
        if after is not None:
            if offset > len(self.events) or self.events[offset - 1].event_id != after.event_id:
                raise JournalScopeError("foreign cursor")
        selected = tuple(self.events[offset : offset + limit])
        for event in selected:
            receipt = self.operations.get(event.operation.operation_id)
            if receipt is None or receipt != (
                event.operation.payload_sha256,
                event,
            ):
                raise JournalRepositoryError(
                    "stored event failed integrity verification"
                )
        next_cursor = (
            EventCursor(selected[-1].store_seq, selected[-1].event_id)
            if selected
            else after
        )
        return JournalPage(
            events=selected,
            next_cursor=next_cursor,
            has_more=offset + len(selected) < len(self.events),
        )

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        events = tuple(self.events)
        if len(events) > max_events or len(
            json.dumps([event.to_dict() for event in events]).encode("utf-8")
        ) > max_bytes:
            raise JournalRepositoryError("snapshot limit exceeded")
        for event in events:
            receipt = self.operations.get(event.operation.operation_id)
            if receipt != (event.operation.payload_sha256, event):
                raise JournalRepositoryError("snapshot integrity failure")
        return capture_journal_snapshot(
            execution_id=self.execution_id,
            events=events,
        )


def attempt(
    *,
    execution_id: str = "execution-1",
    generation_id: str = "generation-1",
    attempt_id: str = "attempt-1",
) -> AttemptRef:
    return AttemptRef(
        GenerationRef(execution_id, generation_id),
        attempt_id,
    )


def projector(bound_attempt: AttemptRef):
    def project(event: Mapping[str, object]) -> SemanticEventDraft | None:
        if event.get("skip"):
            return None
        return SemanticEventDraft(
            event_id=event.get("event_id", ""),
            event_type=event.get("semantic_type", event.get("type", "")),
            attempt=event.get("attempt", bound_attempt),
            operation_id=event.get("operation_id", ""),
            payload=event.get("payload", {}),
            resource_refs=event.get("resource_refs", ()),
        )

    return project


def raw_event(
    event_id: str,
    *,
    operation_id: str | None = None,
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "type": "semantic",
        "semantic_type": "message.user",
        "event_id": event_id,
        "operation_id": (
            operation_id if operation_id is not None else f"operation-{event_id}"
        ),
        "payload": dict(payload or {"content": "synthetic"}),
    }


def test_sink_persists_synchronously_and_computes_the_payload_hash_itself() -> None:
    repository = MemoryJournal("execution-1")
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )
    event = raw_event("event-1", payload={"b": 2, "a": 1})
    event["payload_sha256"] = "0" * 64

    result = sink(event)

    assert result is not None
    assert result.event.payload == {"a": 1, "b": 2}
    assert result.event.operation.payload_sha256 != "0" * 64
    assert repository.events == [result.event]

    replay = sink(raw_event("event-1", payload={"a": 1, "b": 2}))
    assert replay is not None and replay.duplicate is True
    assert replay.event == result.event

    with pytest.raises(JournalConflictError):
        sink(raw_event("event-1", payload={"a": 2, "b": 2}))


def test_operation_hash_is_deterministic_for_equivalent_normalized_payloads() -> None:
    first_repository = MemoryJournal("execution-1")
    second_repository = MemoryJournal("execution-1")
    first = DurableEventSink(
        journal=first_repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )
    second = DurableEventSink(
        journal=second_repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )

    first(raw_event("event-1", payload={"outer": {"b": 2, "a": 1}}))
    second(raw_event("event-1", payload={"outer": {"a": 1, "b": 2}}))

    assert (
        first_repository.requests[0].operation.payload_sha256
        == second_repository.requests[0].operation.payload_sha256
    )


@pytest.mark.parametrize(
    "event_type",
    (
        "token_delta",
        "response.token.delta",
        "reasoning_delta",
        "thinking_delta",
        "analysis_delta",
        "hidden_chain_of_thought",
        "content_delta",
        "provider_retry",
    ),
)
def test_sink_ignores_streaming_and_hidden_reasoning_before_projection(
    event_type: str,
) -> None:
    repository = MemoryJournal("execution-1")
    projected: list[Mapping[str, object]] = []

    def forbidden_projector(event: Mapping[str, object]) -> SemanticEventDraft:
        projected.append(copy.deepcopy(event))
        raise AssertionError("ignored runtime events must not reach the projector")

    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=forbidden_projector,
    )

    assert sink({"type": event_type, "content": "synthetic fragment"}) is None
    assert projected == []
    assert repository.events == []


def test_sink_ignores_hidden_reasoning_after_host_projection() -> None:
    repository = MemoryJournal("execution-1")
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )

    wrapped = raw_event("event-1")
    wrapped["semantic_type"] = "reasoning_delta"

    assert sink(wrapped) is None
    assert repository.events == []


def test_sink_rejects_missing_stable_identity_and_foreign_attempts() -> None:
    repository = MemoryJournal("execution-1")
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )

    with pytest.raises(ValueError, match="event_id"):
        sink(raw_event("", operation_id="operation-1"))
    with pytest.raises(ValueError, match="operation_id"):
        sink(raw_event("event-1", operation_id=""))

    foreign = raw_event("event-2")
    foreign["attempt"] = attempt(attempt_id="attempt-2")
    with pytest.raises(DurableJournalScopeError, match="attempt"):
        sink(foreign)

    with pytest.raises(DurableJournalScopeError, match="execution"):
        DurableEventSink(
            journal=MemoryJournal("execution-2"),
            attempt=attempt(),
            projector=projector(attempt()),
        )


def test_sink_fails_closed_when_repository_receipt_does_not_match_request() -> None:
    repository = MemoryJournal("execution-1")
    repository.lie_about_payload = True
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )

    with pytest.raises(DurableJournalIntegrityError, match="persisted event"):
        sink(raw_event("event-1"))


def test_replay_paginates_filters_other_attempts_and_rejects_foreign_generation() -> None:
    repository = MemoryJournal("execution-1")
    current = attempt()
    other_attempt = attempt(attempt_id="attempt-2")
    sink = DurableEventSink(
        journal=repository,
        attempt=current,
        projector=projector(current),
    )
    for index, selected_attempt in enumerate(
        (current, other_attempt, current, current, current),
        start=1,
    ):
        repository.append(
            request=JournalAppendRequest(
                event_id=f"event-{index}",
                event_type="message.user",
                attempt=selected_attempt,
                operation=(
                    OperationRef("legacy-operation", "a" * 64)
                    if selected_attempt == other_attempt
                    else SemanticEventDraft(
                        event_id=f"event-{index}",
                        event_type="message.user",
                        attempt=selected_attempt,
                        operation_id=f"operation-{index}",
                        payload={"index": index},
                    ).operation
                ),
                payload={"index": index},
            )
        )

    replayed = sink.replay(page_size=2)
    assert [event.event_id for event in replayed] == [
        "event-1",
        "event-3",
        "event-4",
        "event-5",
    ]

    foreign_generation_event = JournalEvent(
        event_id="event-5",
        event_type="message.user",
        attempt=attempt(generation_id="generation-2"),
        operation=repository.events[-1].operation,
        store_seq=5,
        payload={"index": 5},
    )
    repository.events[-1] = foreign_generation_event
    repository.operations[foreign_generation_event.operation.operation_id] = (
        foreign_generation_event.operation.payload_sha256,
        foreign_generation_event,
    )
    with pytest.raises(DurableJournalScopeError, match="generation"):
        sink.replay(page_size=2)


def test_resource_refs_are_part_of_the_exact_operation_payload() -> None:
    repository = MemoryJournal("execution-1")
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )
    first = raw_event("event-1")
    first["resource_refs"] = (ResourceRef("artifact", "artifact-1", 1),)
    sink(first)

    changed = raw_event("event-1")
    changed["resource_refs"] = (ResourceRef("artifact", "artifact-2", 1),)
    with pytest.raises(JournalConflictError):
        sink(changed)


def test_journal_event_projection_preserves_outer_event_type_for_the_compiler() -> None:
    repository = MemoryJournal("execution-1")
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )
    persisted = sink(
        raw_event(
            "event-1",
            payload={"content": "synthetic", "type": "untrusted.inner.type"},
        )
    )
    assert persisted is not None

    semantic = journal_event_to_semantic_event(persisted.event)

    assert semantic["type"] == "message.user"
    assert semantic["event_id"] == "event-1"
    assert semantic["store_seq"] == 1
    assert semantic["execution_id"] == "execution-1"
    assert semantic["generation_id"] == "generation-1"
    assert semantic["attempt_id"] == "attempt-1"
    assert semantic["content"] == "synthetic"


def test_durable_journal_runtime_is_publicly_exported() -> None:
    assert journal_api.DurableEventSink is DurableEventSink
    assert journal_api.SemanticEventDraft is SemanticEventDraft
    assert journal_api.SideEffectRecoveryState is not None
    assert journal_api.journal_event_to_semantic_event is journal_event_to_semantic_event


def test_replay_propagates_repository_payload_integrity_failure() -> None:
    repository = MemoryJournal("execution-1")
    sink = DurableEventSink(
        journal=repository,
        attempt=attempt(),
        projector=projector(attempt()),
    )
    persisted = sink(raw_event("event-1"))
    assert persisted is not None
    repository.events[0] = JournalEvent(
        event_id=persisted.event.event_id,
        event_type=persisted.event.event_type,
        attempt=persisted.event.attempt,
        operation=persisted.event.operation,
        store_seq=persisted.event.store_seq,
        payload={"content": "corrupted"},
        resource_refs=persisted.event.resource_refs,
    )

    with pytest.raises(JournalRepositoryError, match="integrity"):
        sink.replay()


def test_replay_accepts_repository_verified_events_from_other_operation_domains() -> None:
    repository = MemoryJournal("execution-1")
    current = attempt()
    sink = DurableEventSink(
        journal=repository,
        attempt=current,
        projector=projector(current),
    )
    repository.append(
        request=JournalAppendRequest(
            event_id="artifact-event-1",
            event_type="artifact.recorded",
            attempt=current,
            operation=OperationRef("artifact-operation-1", "b" * 64),
            payload={
                "artifact_ref": ResourceRef(
                    "artifact",
                    "artifact-1",
                    1,
                ).to_dict(),
            },
            resource_refs=(ResourceRef("artifact", "artifact-1", 1),),
        )
    )

    assert [event.event_id for event in sink.replay()] == ["artifact-event-1"]
