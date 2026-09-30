from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from unchain.context import (
    BoundContextTaskStateReader,
    ContextBuildReceipt,
    ContextBuildStatus,
    ContextBuildUnavailableError,
    ContextCompileCoordinator,
    ContextCompileRequest,
    ContextCompiler,
    ContextRuntime,
    ContextTaskStateReadOutcome,
    ContextTaskStateUnavailable,
    PinnedTaskState,
    SourceMessageCursor,
    resolve_context_budget,
)
from unchain.journal import (
    BoundExecutionJournal,
    AttemptRef,
    GenerationRef,
    JournalEvent,
    JournalPage,
    ModelValidationError,
    ResourceRef,
    SemanticEventDraft,
    capture_journal_snapshot,
)
from unchain.memory.workspace import (
    BoundPinnedTaskStateRepository,
    BoundWorkspaceReferenceAuthorizer,
    RepositoryScopeError,
    TaskStateService,
)


def _state(*, constraints: tuple[str, ...]) -> PinnedTaskState:
    return PinnedTaskState(
        state_id="task-state-legacy",
        revision=7,
        objective="classified objective",
        constraints=constraints,
    )


def test_context_task_state_read_returns_a_content_free_marker_on_item_overflow() -> None:
    outcome = ContextTaskStateReadOutcome.from_state(
        _state(
            constraints=tuple(
                f"classified constraint {index}" for index in range(257)
            )
        )
    )

    assert outcome.capture_quality is ContextBuildStatus.UNAVAILABLE
    assert outcome.state is None
    assert outcome.unavailable == ContextTaskStateUnavailable(
        state_ref=ResourceRef("task_state", "task-state-legacy", 7),
        item_count=257,
        content_bytes=outcome.unavailable.content_bytes,
        reason="task_state_limits_exceeded",
    )
    serialized = json.dumps(outcome.to_dict(), sort_keys=True)
    assert "classified objective" not in serialized
    assert "classified constraint" not in serialized


def test_context_task_state_read_returns_a_content_free_marker_on_byte_overflow() -> None:
    outcome = ContextTaskStateReadOutcome.from_state(
        _state(constraints=tuple("x" * 16_000 for _ in range(17)))
    )

    assert outcome.capture_quality is ContextBuildStatus.UNAVAILABLE
    assert outcome.state is None
    assert outcome.unavailable is not None
    assert outcome.unavailable.item_count == 17
    assert outcome.unavailable.content_bytes > 256 * 1024


def test_context_task_state_read_preserves_available_and_absent_states() -> None:
    state = _state(constraints=tuple(f"constraint {index}" for index in range(256)))

    available = ContextTaskStateReadOutcome.from_state(state)
    absent = ContextTaskStateReadOutcome.from_state(None)

    assert available.capture_quality is ContextBuildStatus.COMPLETE
    assert available.state == state
    assert available.unavailable is None
    assert ContextTaskStateReadOutcome.from_dict(available.to_dict()) == available
    assert absent.capture_quality is ContextBuildStatus.COMPLETE
    assert absent.state is None
    assert absent.unavailable is None


def test_context_task_state_reader_is_an_independent_typed_port() -> None:
    state = _state(constraints=())

    class Reader(BoundContextTaskStateReader):
        def read_for_context(self) -> ContextTaskStateReadOutcome:
            return ContextTaskStateReadOutcome.from_state(state)

    reader = Reader("binding-1")

    assert reader.binding_id == "binding-1"
    assert reader.read_for_context().state == state
    assert "read_for_context" not in BoundPinnedTaskStateRepository.__abstractmethods__


class _LegacyRepository(BoundPinnedTaskStateRepository):
    def __init__(self, state: PinnedTaskState) -> None:
        super().__init__("binding-1", state.state_id)
        self.state = state

    def current(self) -> PinnedTaskState | None:
        return self.state

    def replay(self, *, operation):
        del operation
        return None

    def compare_and_swap(self, *, state, expected_revision, operation):
        del state, expected_revision, operation
        raise AssertionError("legacy overflow must not fall back to a write")


class _References(BoundWorkspaceReferenceAuthorizer):
    def authorize(self, *, ref):
        raise RepositoryScopeError(f"unexpected reference: {ref}")


def test_task_state_toolkit_read_keeps_its_explicit_failure_contract() -> None:
    repository = _LegacyRepository(
        _state(constraints=tuple(f"constraint {index}" for index in range(257)))
    )
    service = TaskStateService(
        repository=repository,
        references=_References("binding-1"),
    )

    with pytest.raises(ModelValidationError, match="item budget"):
        service.get()

    assert repository.state.revision == 7


def _unavailable_marker() -> ContextTaskStateUnavailable:
    outcome = ContextTaskStateReadOutcome.from_state(
        _state(constraints=tuple(f"classified {index}" for index in range(257)))
    )
    assert outcome.unavailable is not None
    return outcome.unavailable


def _unavailable_request() -> ContextCompileRequest:
    return ContextCompileRequest(
        case="legacy-task-state-overflow",
        source_messages=({"role": "user", "content": "continue"},),
        current_generation="generation-1",
        task_state_unavailable=_unavailable_marker(),
        capture_quality=ContextBuildStatus.UNAVAILABLE.value,
        budget=resolve_context_budget(context_window_tokens=8_192),
        provider="openai",
        model="synthetic",
        build_id="build-task-state-7",
        execution_id="execution-1",
        generation_id="generation-1",
        attempt_id="attempt-1",
        source_message_cursors=(
            SourceMessageCursor(0, "event-current-user", 1),
        ),
    )


def test_compile_request_round_trips_the_content_free_task_state_marker() -> None:
    request = _unavailable_request()

    assert ContextCompileRequest.from_dict(request.to_dict()) == request
    serialized = json.dumps(request.to_dict(), sort_keys=True)
    assert "classified objective" not in serialized
    assert "classified 0" not in serialized

    with pytest.raises(ModelValidationError, match="task state"):
        ContextCompileRequest(
            case="invalid-task-state-read",
            source_messages=(),
            task_state=_state(constraints=()).to_dict(),
            task_state_unavailable=_unavailable_marker(),
            capture_quality=ContextBuildStatus.UNAVAILABLE.value,
        )


def test_compiler_forms_an_unavailable_envelope_without_task_state_content() -> None:
    result = ContextCompiler().compile(_unavailable_request())

    assert result.messages == ()
    assert result.checkpoint_requests == ()
    assert result.diagnostics["status"] == "task_state_unavailable"
    assert result.diagnostics["capture_quality"] == "unavailable"
    assert result.diagnostics["task_state_unavailable"] == _unavailable_marker().to_dict()
    assert result.envelope is not None
    assert result.envelope.status is ContextBuildStatus.UNAVAILABLE
    assert result.envelope.included_ranges == ()
    assert result.envelope.transformed_ranges == ()
    serialized = json.dumps(result.to_dict(), sort_keys=True)
    assert "classified objective" not in serialized
    assert "classified 0" not in serialized


class _NoCheckpointRepository:
    execution_id = "execution-1"

    def prepare(self, **kwargs):
        raise AssertionError(f"unavailable task state cannot checkpoint: {kwargs}")

    def commit(self, *, prepared):
        raise AssertionError(f"unavailable task state cannot commit: {prepared}")

    def get_by_operation(self, *, operation):
        return None


class _EmptyJournal(BoundExecutionJournal):
    def __init__(self):
        super().__init__("execution-1")

    def append(self, *, request):
        raise AssertionError(f"unexpected append: {request}")

    def read(self, *, after=None, limit=100):
        snapshot = self.capture_snapshot()
        if after not in (None, snapshot.high_water):
            raise AssertionError(f"unexpected live read: {after}, {limit}")
        return JournalPage(events=(), next_cursor=after, has_more=False)

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        del max_events, max_bytes
        attempt = AttemptRef(
            GenerationRef("execution-1", "generation-1"),
            "attempt-1",
        )
        payload = {
            "run_id": "attempt-1",
            "message": {"role": "user", "content": "continue"},
        }
        operation = SemanticEventDraft(
            event_id="event-current-user",
            event_type="message.user",
            attempt=attempt,
            operation_id="operation-current-user",
            payload=payload,
        ).operation
        return capture_journal_snapshot(
            execution_id=self.execution_id,
            events=(
                JournalEvent(
                    event_id="event-current-user",
                    event_type="message.user",
                    attempt=attempt,
                    operation=operation,
                    store_seq=1,
                    payload=payload,
                ),
            ),
        )


class _IdempotentBuildRepository:
    execution_id = "execution-1"

    def __init__(self) -> None:
        self.effects = {}
        self.triggers = {}
        self.record_calls = 0

    def record(self, *, envelope, operation, trigger_cursor):
        self.record_calls += 1
        previous = self.effects.get(operation.operation_id)
        if previous is not None:
            if previous.operation != operation:
                raise AssertionError("operation identity changed")
            return ContextBuildReceipt(
                envelope=previous.envelope,
                operation=previous.operation,
                trigger_cursor=previous.trigger_cursor,
                duplicate=True,
            )
        receipt = ContextBuildReceipt(
            envelope=envelope,
            operation=operation,
            trigger_cursor=trigger_cursor,
        )
        prior_trigger = self.triggers.get(trigger_cursor)
        if (
            prior_trigger is not None
            and prior_trigger.envelope.build_id != envelope.build_id
        ):
            raise AssertionError("input receipt already claimed")
        self.effects[operation.operation_id] = receipt
        self.triggers[trigger_cursor] = receipt
        return receipt

    def get_by_operation(self, *, operation):
        receipt = self.effects.get(operation.operation_id)
        if receipt is not None and receipt.operation != operation:
            raise AssertionError("operation identity changed")
        return receipt

    def get_by_trigger(self, *, trigger_cursor):
        return self.triggers.get(trigger_cursor)


def _coordinator(builds, partials):
    return ContextCompileCoordinator(
        journal=_EmptyJournal(),
        checkpoint_repository=_NoCheckpointRepository(),
        build_repository=builds,
        partial_attempt_sink=lambda request, error: partials.append(
            (request, error)
        ),
    )


def test_coordinator_persists_one_idempotent_unavailable_build_effect() -> None:
    builds = _IdempotentBuildRepository()
    partials = []
    coordinator = _coordinator(builds, partials)

    first = coordinator.compile(_unavailable_request())
    second = coordinator.compile(_unavailable_request())

    assert first == second
    assert first.envelope.status is ContextBuildStatus.UNAVAILABLE
    assert builds.record_calls == 1
    assert len(builds.effects) == 1
    assert partials == []


def test_runtime_blocks_model_use_only_after_unavailable_build_is_persisted() -> None:
    builds = _IdempotentBuildRepository()
    partials = []
    runtime = ContextRuntime(
        owner_id="context-task-state-read",
        request_factory=lambda context: _unavailable_request(),
        durable_event_sink=lambda event: None,
        partial_attempt_sink=lambda event, error: partials.append((event, error)),
        compiler=_coordinator(builds, partials),
    )

    with pytest.raises(ContextBuildUnavailableError) as raised:
        runtime.compile_context(
            SimpleNamespace(
                event={"execution_id": "execution-1", "attempt_id": "attempt-1"}
            )
        )

    assert raised.value.result.envelope.status is ContextBuildStatus.UNAVAILABLE
    assert len(builds.effects) == 1
    assert partials == []
