from __future__ import annotations

import hashlib

import pytest

from unchain.context import (
    ArtifactService,
    ContextInputIngress,
    ContextInputIngressError,
    HostResolvedCurrentInput,
    HostResolvedInteractionInput,
    JournalContextRequestFactory,
    JournalContextRequestFactoryError,
    DurableToolApprovalState,
    DurableToolExecutionSubject,
    DurableToolRouteKind,
)
from unchain.context.ports import BoundArtifactRepository, ContextConflictError
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.journal import (
    ArtifactRef,
    AttemptRef,
    BoundExecutionJournal,
    DurableEventSink,
    EventCursor,
    GenerationRef,
    JournalAppendResult,
    JournalConflictError,
    JournalEvent,
    JournalPage,
    ResourceRef,
    SemanticEventDraft,
    capture_journal_snapshot,
)
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.execution import ExecutionFence


def _tool_subject(intent_cursor: EventCursor) -> DurableToolExecutionSubject:
    return DurableToolExecutionSubject(
        intent_cursor=intent_cursor,
        original_arguments_sha256="1" * 64,
        effective_arguments_sha256="2" * 64,
        approval_state=DurableToolApprovalState.NOT_REQUIRED,
        approval_request_sha256="",
        approval_receipt_sha256="",
        route_kind=DurableToolRouteKind.NORMAL,
        route_manifest_sha256="3" * 64,
        terminal_handler_manifest_sha256="4" * 64,
        execution_fence=ExecutionFence("execution-1", "owner-1", 1),
    )


class _ArtifactRepository(BoundArtifactRepository):
    def __init__(self, order: list[str]) -> None:
        super().__init__("execution-1")
        self.order = order
        self.content: dict[str, bytes] = {}
        self.operations = {}

    def put(self, *, content, media_type, operation, preview=""):
        self.order.append("artifact.put")
        previous = self.operations.get(operation.operation_id)
        if previous is not None:
            prior_operation, artifact = previous
            if prior_operation != operation:
                raise ContextConflictError("artifact operation payload changed")
            return artifact
        digest = hashlib.sha256(content).hexdigest()
        artifact = ArtifactRef(
            ref=ResourceRef("artifact", f"object-{operation.operation_id}", 1),
            media_type=media_type,
            byte_length=len(content),
            sha256=digest,
            preview=preview,
        )
        self.operations[operation.operation_id] = (operation, artifact)
        self.content[artifact.ref.resource_id] = content
        return artifact

    def artifact_id_for(self, *, logical_kind, logical_key):
        return f"object-{logical_key}"

    def read_verified(self, *, artifact, offset=0, limit=65_536):
        return self.content[artifact.ref.resource_id][offset : offset + limit]

    def read_full_verified(self, *, artifact):
        return self.content[artifact.ref.resource_id]


class _Journal(BoundExecutionJournal):
    def __init__(self, order: list[str], *, artifacts: BoundArtifactRepository | None = None) -> None:
        super().__init__("execution-1")
        self.order = order
        self.events: list[JournalEvent] = []
        self.operations = {}
        self.capture_calls = 0
        self._artifacts = artifacts

    def append(self, *, request):
        self.order.append("journal.append")
        previous = self.operations.get(request.operation.operation_id)
        if previous is not None:
            prior_request, event = previous
            if prior_request != request:
                raise JournalConflictError("journal operation payload changed")
            return JournalAppendResult(
                event=event,
                cursor=EventCursor(event.store_seq, event.event_id),
                duplicate=True,
            )
        event = JournalEvent(
            event_id=request.event_id,
            event_type=request.event_type,
            attempt=request.attempt,
            operation=request.operation,
            store_seq=len(self.events) + 1,
            payload=request.payload,
            resource_refs=request.resource_refs,
        )
        self.events.append(event)
        self.operations[request.operation.operation_id] = (request, event)
        return JournalAppendResult(
            event=event,
            cursor=EventCursor(event.store_seq, event.event_id),
        )

    def append_with_artifacts(self, *, request, artifacts, precondition=None):
        previous = self.operations.get(request.operation.operation_id)
        if previous is not None:
            return self.append(request=request)
        if precondition is not None:
            precondition(self.capture_snapshot())
        for pending in artifacts:
            self._artifacts.put(
                content=pending.content,
                media_type=pending.media_type,
                operation=pending.operation,
                preview=pending.preview,
            )
        return self.append(request=request)

    def read(self, *, after=None, limit=100):
        start = after.store_seq if after is not None else 0
        return JournalPage(
            events=tuple(self.events[start : start + limit]),
            has_more=False,
        )

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        del max_events, max_bytes
        self.capture_calls += 1
        return capture_journal_snapshot(
            execution_id=self.execution_id,
            events=tuple(self.events),
        )

    def snapshot_prefix_is_current(self, *, snapshot, integrity_revision=None):
        return tuple(self.events[: snapshot.event_count]) == snapshot.events


class _Toolkit:
    def __init__(self) -> None:
        self.providers: list[str] = []

    def to_provider_json(self, provider):
        self.providers.append(provider)
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup one durable value",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]


def _attempt(attempt_id: str = "run-1") -> AttemptRef:
    return AttemptRef(
        generation=GenerationRef("execution-1", "generation-1"),
        attempt_id=attempt_id,
    )


def _bound(attempt: AttemptRef | None = None):
    bound_attempt = attempt or _attempt()
    order: list[str] = []
    repository = _ArtifactRepository(order)
    artifacts = ArtifactService(
        repository,
        sanitizer=lambda content, media_type: content,
    )
    projector = CanonicalSemanticEventProjector(
        attempt=bound_attempt,
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    journal = _Journal(order, artifacts=repository)
    sink = DurableEventSink(journal, bound_attempt, projector)
    ingress = ContextInputIngress(
        attempt=bound_attempt,
        projector=projector,
        sink=sink,
    )
    return order, journal, projector, sink, ingress


def _context(
    *,
    messages,
    toolkit=None,
    window=16_384,
    provider="openai",
    model="gpt-test",
    run_id="run-1",
):
    state = RunState()
    state.seed_messages(list(messages))
    state.session_state.session_id = "execution-1"
    state.provider_state.provider = provider
    state.provider_state.model = model
    state.provider_state.max_context_window_tokens = window
    return HarnessContext(
        state=state,
        phase="before_model",
        event={"run_id": run_id, "toolkit": toolkit or _Toolkit()},
    )


def test_input_ingress_persists_artifact_before_journal_and_replays_exactly() -> None:
    order, journal, _projector, _sink, ingress = _bound()
    current = HostResolvedCurrentInput(
        attempt=_attempt(),
        content="canonical user input",
        message_index=7,
    )

    first = ingress.persist(current)
    second = ingress.persist(current)

    assert order == [
        "artifact.put",
        "journal.append",
        "artifact.put",
        "journal.append",
    ]
    assert first.duplicate is False
    assert second.duplicate is True
    assert second.event == first.event == journal.events[0]
    assert first.event.event_type == "message.user"
    assert first.event.payload["message"] == {
        "role": "user",
        "content": "canonical user input",
    }
    assert first.event.payload["content_ref"] == first.event.resource_refs[0].to_dict()


def _append_interaction_request(
    journal: _Journal,
    *,
    attempt: AttemptRef,
    interaction_id: str = "interaction-human-1",
    operation_suffix: str = "1",
) -> JournalAppendResult:
    return journal.append(
        request=SemanticEventDraft(
            event_id=f"event-interaction-requested-{operation_suffix}",
            event_type="interaction.requested",
            attempt=attempt,
            operation_id=f"operation-interaction-requested-{operation_suffix}",
            payload={
                "run_id": attempt.attempt_id,
                "interaction_id": interaction_id,
                "interaction_request": {
                    "interaction_id": interaction_id,
                    "kind": "human_input",
                },
            },
        ).to_append_request()
    )


def test_interaction_ingress_persists_new_attempt_resolution_and_request_uses_it() -> (
    None
):
    paused_attempt = _attempt("run-paused")
    resumed_attempt = _attempt("run-resumed")
    order, journal, _projector, _sink, ingress = _bound(resumed_attempt)
    requested = _append_interaction_request(journal, attempt=paused_attempt)

    resolved = ingress.persist(
        HostResolvedInteractionInput(
            attempt=resumed_attempt,
            interaction_id="interaction-human-1",
            response={"selected_values": ["react"]},
            submitted_by="ui:test",
        )
    )
    request = JournalContextRequestFactory(
        attempt=resumed_attempt,
        journal=journal,
        model_window_fallback=lambda provider, model: 8_192,
    )(
        _context(
            messages=[
                {"role": "system", "content": "current system"},
                {"role": "user", "content": "transcript must not be guessed"},
            ],
            run_id="run-resumed",
        )
    )

    assert order == ["journal.append", "artifact.put", "journal.append"]
    assert resolved.event.attempt == resumed_attempt
    assert resolved.event.event_type == "interaction.resolved"
    assert request.source_messages == ({"role": "system", "content": "current system"},)
    assert request.source_message_cursors == ()
    assert request.pending_task_inputs == (
        {
            "event_id": resolved.event.event_id,
            "store_seq": resolved.event.store_seq,
            "type": "interaction_resolved",
            "preview": resolved.event.payload["preview"],
            "preview_truncated": resolved.event.payload["preview_truncated"],
            "content_ref": resolved.event.payload["content_ref"],
            "content_bytes": resolved.event.payload["content_bytes"],
            "content_sha256": resolved.event.payload["content_sha256"],
        },
    )
    assert [event["type"] for event in request.semantic_events] == [
        "interaction.requested",
        "interaction.resolved",
    ]
    assert request.semantic_events[0]["event_id"] == requested.event.event_id


def test_interaction_ingress_and_factory_reject_foreign_or_ambiguous_cause() -> None:
    resumed_attempt = _attempt("run-resumed")
    _order, journal, _projector, _sink, ingress = _bound(resumed_attempt)
    with pytest.raises(ContextInputIngressError, match="bound attempt"):
        ingress.persist(
            HostResolvedInteractionInput(
                attempt=_attempt("run-foreign"),
                interaction_id="interaction-human-1",
                response={"selected_values": ["react"]},
            )
        )

    _append_interaction_request(
        journal,
        attempt=_attempt("run-paused-a"),
        operation_suffix="a",
    )
    _append_interaction_request(
        journal,
        attempt=_attempt("run-paused-b"),
        operation_suffix="b",
    )
    ingress.persist(
        HostResolvedInteractionInput(
            attempt=resumed_attempt,
            interaction_id="interaction-human-1",
            response={"selected_values": ["react"]},
        )
    )
    factory = JournalContextRequestFactory(
        attempt=resumed_attempt,
        journal=journal,
        model_window_fallback=lambda provider, model: 8_192,
    )

    with pytest.raises(
        JournalContextRequestFactoryError,
        match="unique interaction request",
    ):
        factory(_context(messages=[], run_id="run-resumed"))


def test_interaction_request_factory_rejects_a_foreign_interaction_id() -> None:
    resumed_attempt = _attempt("run-resumed")
    _order, journal, _projector, _sink, ingress = _bound(resumed_attempt)
    _append_interaction_request(
        journal,
        attempt=_attempt("run-paused"),
        interaction_id="interaction-expected",
    )
    ingress.persist(
        HostResolvedInteractionInput(
            attempt=resumed_attempt,
            interaction_id="interaction-foreign",
            response={"selected_values": ["react"]},
        )
    )
    factory = JournalContextRequestFactory(
        attempt=resumed_attempt,
        journal=journal,
        model_window_fallback=lambda provider, model: 8_192,
    )

    with pytest.raises(
        JournalContextRequestFactoryError,
        match="unique interaction request",
    ):
        factory(_context(messages=[], run_id="run-resumed"))


def test_input_ingress_rejects_empty_foreign_and_changed_inputs() -> None:
    _order, journal, _projector, _sink, ingress = _bound()

    with pytest.raises(ValueError, match="non-empty"):
        HostResolvedCurrentInput(attempt=_attempt(), content="  ")
    with pytest.raises(ContextInputIngressError, match="bound attempt"):
        ingress.persist(
            HostResolvedCurrentInput(
                attempt=_attempt("run-foreign"),
                content="foreign",
            )
        )

    ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="first"))
    with pytest.raises(ContextConflictError, match="payload changed"):
        ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="changed"))
    assert len(journal.events) == 1


def test_input_ingress_requires_the_projector_owned_by_the_exact_sink() -> None:
    _order, _journal, projector, sink, _ingress = _bound()
    _other_order, _other_journal, other_projector, _other_sink, _ = _bound()

    with pytest.raises(ContextInputIngressError, match="same projector"):
        ContextInputIngress(
            attempt=_attempt(),
            projector=other_projector,
            sink=sink,
        )
    assert projector is sink.projector


def test_input_ingress_rejects_a_malformed_projected_receipt_before_append() -> None:
    order: list[str] = []
    artifacts = ArtifactService(
        _ArtifactRepository(order),
        sanitizer=lambda content, media_type: content,
    )

    class _MalformedProjector(CanonicalSemanticEventProjector):
        def project_user_message(self, message, *, message_index=0):
            del message, message_index
            return SemanticEventDraft(
                event_id="event-malformed-input",
                event_type="message.user",
                attempt=_attempt(),
                operation_id="operation-malformed-input",
                payload={
                    "run_id": "run-1",
                    "message": {"role": "user", "content": "unsafe"},
                },
            )

    projector = _MalformedProjector(
        attempt=_attempt(),
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    journal = _Journal(order)
    ingress = ContextInputIngress(
        attempt=_attempt(),
        projector=projector,
        sink=DurableEventSink(journal, _attempt(), projector),
    )

    with pytest.raises(ContextInputIngressError, match="content reference"):
        ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="unsafe"))
    assert journal.events == []


def test_request_factory_uses_current_instructions_but_journal_user_input() -> None:
    _order, journal, _projector, _sink, ingress = _bound()
    receipt = ingress.persist(
        HostResolvedCurrentInput(
            attempt=_attempt(),
            content="journal is canonical",
            message_index=3,
        )
    )
    toolkit = _Toolkit()
    context = _context(
        toolkit=toolkit,
        messages=[
            {"role": "system", "content": "current system"},
            {"role": "developer", "content": "current developer"},
            {"role": "assistant", "content": "transcript-only assistant"},
            {"role": "user", "content": "transcript must not authorize input"},
        ],
    )
    fallback_calls = []
    factory = JournalContextRequestFactory(
        attempt=_attempt(),
        journal=journal,
        model_window_fallback=lambda provider, model: fallback_calls.append(
            (provider, model)
        )
        or 8_192,
    )

    request = factory(context)
    replay = factory(context)

    assert request == replay
    assert request.source_messages == (
        {"role": "system", "content": "current system"},
        {"role": "developer", "content": "current developer"},
        {"role": "user", "content": "journal is canonical"},
    )
    assert request.source_message_cursors[0].message_index == 2
    assert request.source_message_cursors[0].event_id == receipt.event.event_id
    assert request.source_message_cursors[0].store_seq == receipt.event.store_seq
    assert request.semantic_events[-1]["type"] == "message.user"
    assert request.semantic_events[-1]["event_id"] == receipt.event.event_id
    assert request.budget.context_window_tokens == 16_384
    assert request.fixed_overhead_tokens > 0
    assert request.build_id.startswith("context-build-")
    assert request.execution_id == "execution-1"
    assert request.generation_id == "generation-1"
    assert request.attempt_id == "run-1"
    assert toolkit.providers == ["openai", "openai"]
    assert fallback_calls == []
    assert journal.capture_calls == 1


def test_request_factory_uses_explicit_finite_fallback_for_unknown_window() -> None:
    _order, journal, _projector, _sink, ingress = _bound()
    ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="hello"))
    context = _context(messages=[], window=0, provider="anthropic", model="claude-test")
    fallback_calls = []
    factory = JournalContextRequestFactory(
        attempt=_attempt(),
        journal=journal,
        model_window_fallback=lambda provider, model: fallback_calls.append(
            (provider, model)
        )
        or 8_192,
    )

    request = factory(context)

    assert request.budget.context_window_tokens == 8_192
    assert fallback_calls == [("anthropic", "claude-test")]

    invalid = JournalContextRequestFactory(
        attempt=_attempt(),
        journal=journal,
        model_window_fallback=lambda provider, model: 0,
    )
    with pytest.raises(JournalContextRequestFactoryError, match="finite positive"):
        invalid(context)


def test_request_factory_builds_resume_input_from_full_tool_artifact_descriptor() -> (
    None
):
    _order, journal, _projector, sink, ingress = _bound()
    ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="run lookup"))
    intent = sink(
        {
            "type": "tool_call",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-1",
            "arguments": {"query": "durable"},
        }
    )
    assert isinstance(intent, JournalAppendResult)
    subject = _tool_subject(intent.cursor)
    tool_result = sink(
        {
            "type": "tool_result",
            "run_id": "run-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-1",
            "execution_subject": subject.to_dict(),
            "execution_subject_sha256": subject.sha256,
            "result": {"answer": "full output"},
        }
    )
    assert tool_result is not None
    context = _context(
        messages=[
            {"role": "system", "content": "current system"},
            {"role": "user", "content": "stale transcript input"},
        ]
    )
    factory = JournalContextRequestFactory(
        attempt=_attempt(),
        journal=journal,
        model_window_fallback=lambda provider, model: 8_192,
    )

    request = factory(context)

    assert request.source_messages == ({"role": "system", "content": "current system"},)
    assert request.source_message_cursors == ()
    assert request.pending_task_inputs == (
        {
            "event_id": tool_result.event.event_id,
            "store_seq": tool_result.event.store_seq,
            "type": "tool_result",
            "preview": tool_result.event.payload["preview"],
            "preview_truncated": tool_result.event.payload["preview_truncated"],
            "content_ref": tool_result.event.payload["full_output_ref"],
            "content_bytes": tool_result.event.payload["result_bytes"],
            "content_sha256": tool_result.event.payload["result_sha256"],
        },
    )
    assert [event["type"] for event in request.semantic_events] == [
        "message.user",
        "tool_call",
        "tool_result",
    ]


def test_request_factory_fails_closed_for_foreign_latest_input_or_bad_tool_ref() -> (
    None
):
    _order, journal, _projector, _sink, ingress = _bound()
    ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="current"))
    foreign_attempt = _attempt("run-foreign")
    foreign_draft = SemanticEventDraft(
        event_id="event-foreign-input",
        event_type="message.user",
        attempt=foreign_attempt,
        operation_id="operation-foreign-input",
        payload={
            "run_id": "run-foreign",
            "message": {"role": "user", "content": "foreign"},
        },
    )
    journal.append(request=foreign_draft.to_append_request())
    factory = JournalContextRequestFactory(
        attempt=_attempt(),
        journal=journal,
        model_window_fallback=lambda provider, model: 8_192,
    )
    with pytest.raises(JournalContextRequestFactoryError, match="latest input"):
        factory(_context(messages=[]))

    journal.events.pop()
    journal.operations.pop("operation-foreign-input")
    malformed = SemanticEventDraft(
        event_id="event-malformed-tool-result",
        event_type="tool_result",
        attempt=_attempt(),
        operation_id="operation-malformed-tool-result",
        payload={
            "run_id": "run-1",
            "tool_name": "lookup",
            "call_id": "call-bad",
            "result_bytes": 1,
            "result_sha256": "0" * 64,
            "preview": "x",
        },
    )
    journal.append(request=malformed.to_append_request())
    with pytest.raises(JournalContextRequestFactoryError, match="artifact descriptor"):
        factory(_context(messages=[]))


def test_request_factory_build_identity_is_bound_to_trigger_provider_and_model() -> (
    None
):
    _order, journal, _projector, _sink, ingress = _bound()
    ingress.persist(HostResolvedCurrentInput(attempt=_attempt(), content="hello"))
    factory = JournalContextRequestFactory(
        attempt=_attempt(),
        journal=journal,
        model_window_fallback=lambda provider, model: 8_192,
    )
    openai = factory(_context(messages=[], provider="openai", model="gpt-test"))
    anthropic = factory(
        _context(messages=[], provider="anthropic", model="claude-test")
    )

    assert openai.build_id != anthropic.build_id
    assert (
        openai.build_id
        == factory(_context(messages=[], provider="openai", model="gpt-test")).build_id
    )
