from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from unchain.context import (
    CheckpointRequest,
    CheckpointWriteStatus,
    ContextBuildReceipt,
    ContextCompileRequest,
    ContextRuntime,
    PreparedCheckpoint,
    SourceMessageCursor,
    resolve_context_budget,
)
from unchain.context.coordinator import (
    ContextCompileCoordinator,
    ContextCompileCoordinatorError,
)
from unchain.durability import DurablePersistenceBoundaryError
from unchain.journal import (
    AttemptRef,
    BoundExecutionJournal,
    EventCursor,
    GenerationRef,
    JournalEvent,
    JournalPage,
    OperationRef,
    ResourceRef,
    SemanticEventDraft,
    capture_journal_snapshot,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _request(*, pressured: bool) -> ContextCompileRequest:
    if pressured:
        messages = (
            {"role": "user", "content": "old " + ("x" * 30_000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        )
    else:
        messages = (
            {"role": "user", "content": "constraint"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "current"},
        )
    return ContextCompileRequest(
        case="durable-coordinator",
        source_messages=messages,
        current_generation="generation-1",
        fixed_overhead_tokens=0,
        budget=resolve_context_budget(context_window_tokens=8_192),
        source_event_ids=("event-1", "event-2", "event-3"),
        source_event_store_seqs=(1, 2, 3),
        provider="openai",
        model="synthetic",
        build_id="build-1",
        execution_id="execution-1",
        generation_id="generation-1",
        attempt_id="attempt-1",
    )


class SnapshotJournal(BoundExecutionJournal):
    def __init__(self, events):
        super().__init__("execution-1")
        self.events = tuple(events)
        self.capture_calls = 0

    def append(self, *, request):
        raise AssertionError(f"unexpected append: {request}")

    def read(self, *, after=None, limit=100):
        if after is None:
            start = 0
        else:
            if (
                after.store_seq > len(self.events)
                or self.events[after.store_seq - 1].event_id != after.event_id
            ):
                raise RuntimeError("journal cursor is invalid")
            start = after.store_seq
        events = self.events[start : start + limit]
        return JournalPage(
            events=events,
            next_cursor=(
                EventCursor(events[-1].store_seq, events[-1].event_id)
                if events
                else after
            ),
            has_more=start + len(events) < len(self.events),
        )

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        del max_events, max_bytes
        self.capture_calls += 1
        return capture_journal_snapshot(
            execution_id=self.execution_id,
            events=self.events,
        )


def _complete_test_journal_prefix(records):
    if not records:
        return records
    existing_sequences = {record.store_seq for record in records}
    for store_seq in range(1, max(existing_sequences) + 1):
        if store_seq in existing_sequences:
            continue
        attempt = AttemptRef(
            GenerationRef("execution-1", "generation-1"),
            "attempt-history",
        )
        event_id = f"filler-event-{store_seq}"
        payload = {
            "run_id": "attempt-history",
            "status": "started",
        }
        operation = SemanticEventDraft(
            event_id=event_id,
            event_type="run_started",
            attempt=attempt,
            operation_id=f"operation-{event_id}",
            payload=payload,
        ).operation
        records.append(
            JournalEvent(
                event_id=event_id,
                event_type="run_started",
                attempt=attempt,
                operation=operation,
                store_seq=store_seq,
                payload=payload,
            )
        )
    records.sort(key=lambda record: record.store_seq)
    return records


def _journal_for_request(request):
    if request.semantic_events:
        records = []
        for raw in request.semantic_events:
            event = dict(raw)
            event_id = event.pop("event_id")
            event_type = event.pop("type")
            store_seq = event.pop("store_seq")
            attempt_id = event.pop("attempt_id")
            event.pop("execution_id", None)
            event.pop("generation_id", None)
            attempt = AttemptRef(
                GenerationRef("execution-1", "generation-1"),
                attempt_id,
            )
            operation_id = f"operation-{event_id}"
            operation = SemanticEventDraft(
                event_id=event_id,
                event_type=event_type,
                attempt=attempt,
                operation_id=operation_id,
                payload=event,
            ).operation
            records.append(
                JournalEvent(
                    event_id=event_id,
                    event_type=event_type,
                    attempt=attempt,
                    operation=operation,
                    store_seq=store_seq,
                    payload=event,
                )
            )
        return SnapshotJournal(_complete_test_journal_prefix(records))
    cursor_map = {
        cursor.message_index: (cursor.event_id, cursor.store_seq)
        for cursor in request.source_message_cursors
    }
    if not cursor_map:
        cursor_map = {
            index: (event_id, request.source_event_store_seqs[index])
            for index, event_id in enumerate(request.source_event_ids)
        }
    records = []
    cursor_indexes = tuple(sorted(cursor_map))
    for position, message_index in enumerate(cursor_indexes):
        event_id, store_seq = cursor_map[message_index]
        message = dict(request.source_messages[message_index])
        attempt_id = (
            "attempt-1"
            if position == len(cursor_indexes) - 1 and message["role"] == "user"
            else "attempt-history"
        )
        attempt = AttemptRef(
            GenerationRef("execution-1", "generation-1"),
            attempt_id,
        )
        payload = {"run_id": attempt_id, "message": message}
        operation = SemanticEventDraft(
            event_id=event_id,
            event_type=f"message.{message['role']}",
            attempt=attempt,
            operation_id=f"operation-{event_id}",
            payload=payload,
        ).operation
        records.append(
            JournalEvent(
                event_id=event_id,
                event_type=f"message.{message['role']}",
                attempt=attempt,
                operation=operation,
                store_seq=store_seq,
                payload=payload,
            )
        )
    return SnapshotJournal(_complete_test_journal_prefix(records))


class RecordingCheckpointRepository:
    execution_id = "execution-1"

    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.calls = []
        self.commit_calls = []
        self.receipts = {}
        self.contents = {}
        self.read_calls = []

    def checkpoint_ref_for(self, *, operation):
        return ResourceRef(
            "checkpoint",
            "checkpoint-" + operation.payload_sha256[:32],
            1,
        )

    def prepare(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.failure is not None:
            raise self.failure
        operation = kwargs["operation"]
        receipt = PreparedCheckpoint(
            preparation_id="preparation-" + operation.payload_sha256[:32],
            checkpoint_ref=self.checkpoint_ref_for(operation=operation),
            operation=operation,
        )
        self.receipts[operation.operation_id] = receipt
        return receipt

    def commit(self, *, prepared):
        self.commit_calls.append(prepared)
        receipt = replace(
            prepared,
            status=CheckpointWriteStatus.COMMITTED,
        )
        self.receipts[receipt.operation.operation_id] = receipt
        prepared_call = next(
            call
            for call in self.calls
            if call["operation"] == receipt.operation
        )
        self.contents[receipt.checkpoint_ref] = prepared_call["summary"].encode("utf-8")
        return receipt

    def get_by_operation(self, *, operation):
        receipt = self.receipts.get(operation.operation_id)
        if receipt is not None and receipt.operation != operation:
            raise RuntimeError("checkpoint operation conflict")
        return receipt

    def list_committed_refs(self, *, limit=32):
        committed = [
            receipt.checkpoint_ref
            for receipt in self.receipts.values()
            if receipt.status is CheckpointWriteStatus.COMMITTED
        ]
        return tuple(reversed(committed[-limit:]))

    def read(self, *, ref, offset=0, limit=65_536):
        self.read_calls.append((ref, offset, limit))
        return self.contents[ref][offset : offset + limit]


class UnsupportedPreviewCheckpointRepository(RecordingCheckpointRepository):
    def checkpoint_ref_for(self, *, operation):
        raise NotImplementedError

    @staticmethod
    def _prepared_ref(operation):
        return ResourceRef(
            "checkpoint",
            "prepared-" + operation.payload_sha256[:32],
            1,
        )

    def prepare(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.failure is not None:
            raise self.failure
        operation = kwargs["operation"]
        receipt = PreparedCheckpoint(
            preparation_id="preparation-" + operation.payload_sha256[:32],
            checkpoint_ref=self._prepared_ref(operation),
            operation=operation,
        )
        self.receipts[operation.operation_id] = receipt
        return receipt


class MissingPreviewCheckpointRepository(UnsupportedPreviewCheckpointRepository):
    checkpoint_ref_for = None


class RecordingBuildRepository:
    execution_id = "execution-1"

    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.calls = []
        self.receipts = {}
        self.triggers = {}

    def record(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.failure is not None:
            raise self.failure
        receipt = ContextBuildReceipt(
            envelope=kwargs["envelope"],
            operation=kwargs["operation"],
            trigger_cursor=kwargs["trigger_cursor"],
        )
        prior_operation = self.receipts.get(receipt.operation.operation_id)
        if prior_operation is not None and prior_operation.operation != receipt.operation:
            raise RuntimeError("build operation conflict")
        previous = self.triggers.get(receipt.trigger_cursor)
        if (
            previous is not None
            and previous.envelope.build_id != receipt.envelope.build_id
        ):
            raise RuntimeError("input receipt already claimed")
        self.receipts[receipt.operation.operation_id] = receipt
        self.triggers[receipt.trigger_cursor] = receipt
        return receipt

    def get_by_operation(self, *, operation):
        receipt = self.receipts.get(operation.operation_id)
        if receipt is not None and receipt.operation != operation:
            raise RuntimeError("build operation conflict")
        return receipt

    def get_by_trigger(self, *, trigger_cursor):
        return self.triggers.get(trigger_cursor)


def _coordinator(
    *,
    request,
    journal=None,
    checkpoints=None,
    builds=None,
    partials=None,
):
    partials = partials if partials is not None else []
    return ContextCompileCoordinator(
        journal=journal or _journal_for_request(request),
        checkpoint_repository=checkpoints or RecordingCheckpointRepository(),
        build_repository=builds or RecordingBuildRepository(),
        partial_attempt_sink=lambda request, error: partials.append((request, error)),
    )


def test_below_pressure_records_the_final_build_without_a_checkpoint() -> None:
    checkpoints = RecordingCheckpointRepository()
    builds = RecordingBuildRepository()
    request = _request(pressured=False)

    result = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
    ).compile(request)

    assert checkpoints.calls == []
    assert len(builds.calls) == 1
    assert builds.calls[0]["envelope"] == result.envelope
    assert builds.calls[0]["operation"].operation_id.startswith(
        "context-build-trigger."
    )
    assert builds.calls[0]["trigger_cursor"] == EventCursor(
        store_seq=3,
        event_id="event-3",
    )


def test_pressure_persists_exact_checkpoint_then_recompiles_and_records_build() -> None:
    checkpoints = RecordingCheckpointRepository()
    builds = RecordingBuildRepository()
    request = _request(pressured=True)

    result = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
    ).compile(request)

    assert result.checkpoint_requests == ()
    assert len(checkpoints.calls) == 1
    payload = json.loads(checkpoints.calls[0]["summary"])
    checkpoint_request = CheckpointRequest.from_dict(
        payload["checkpoint_request"]
    )
    assert checkpoints.calls[0]["source_range"] == checkpoint_request.source_range
    assert result.envelope.checkpoint_refs == (
        checkpoints.commit_calls[0].checkpoint_ref,
    )
    assert payload["schema"] == "unchain.context_checkpoint_payload.v2"
    assert payload["checkpoint_request"] == checkpoint_request.to_dict()
    assert payload["source_messages"] == list(request.source_messages[:2])
    assert payload["source_messages_sha256"] == checkpoint_request.source_messages_sha256
    assert (
        checkpoints.calls[0]["operation"].operation_id
        == f"context-checkpoint.{checkpoint_request.request_id}"
    )
    assert len(builds.calls) == 1


@pytest.mark.parametrize(
    ("retained_chars", "expected_omitted_indexes"),
    (
        (19_000, (0, 1)),
        (19_250, (0, 1, 2, 3)),
    ),
)
def test_checkpoint_planning_prices_projection_before_preparing_boundary_prefix(
    retained_chars,
    expected_omitted_indexes,
) -> None:
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "user", "content": "x" * 30_000},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "y" * retained_chars},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "current"},
        ),
        source_event_ids=tuple(f"event-{index}" for index in range(1, 6)),
        source_event_store_seqs=tuple(range(1, 6)),
    )
    checkpoints = RecordingCheckpointRepository()

    result = _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert (
        result.diagnostics["omitted_source_indexes"]
        == expected_omitted_indexes
    )
    assert len(checkpoints.calls) == 1
    assert len(checkpoints.commit_calls) == 1
    payload = json.loads(checkpoints.calls[0]["summary"])
    assert payload["source_messages"] == list(
        request.source_messages[: len(expected_omitted_indexes)]
    )


@pytest.mark.parametrize(
    "repository_type",
    (UnsupportedPreviewCheckpointRepository, MissingPreviewCheckpointRepository),
)
def test_checkpoint_planning_accepts_retained_repository_without_preview(
    repository_type,
) -> None:
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "user", "content": "x" * 30_000},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "y" * 19_000},
        ),
    )
    checkpoints = repository_type()

    result = _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert result.diagnostics["omitted_source_indexes"] == (0, 1)
    assert len(checkpoints.calls) == 1
    assert len(checkpoints.commit_calls) == 1
    assert result.envelope is not None
    assert result.envelope.checkpoint_refs == (
        checkpoints.commit_calls[0].checkpoint_ref,
    )


def test_checkpoint_retry_expands_prefix_for_unpreviewed_real_ref() -> None:
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "user", "content": "x" * 30_000},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "y" * 19_250},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "current"},
        ),
        source_event_ids=tuple(f"event-{index}" for index in range(1, 6)),
        source_event_store_seqs=tuple(range(1, 6)),
    )
    checkpoints = UnsupportedPreviewCheckpointRepository()

    result = _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert result.diagnostics["omitted_source_indexes"] == (0, 1, 2, 3)
    assert len(checkpoints.calls) == 2
    assert len(checkpoints.commit_calls) == 1
    assert checkpoints.commit_calls[0].operation == checkpoints.calls[1]["operation"]


@pytest.mark.parametrize(
    ("retained_chars", "expected_omitted_indexes"),
    (
        (17_850, (0, 1)),
        (17_900, (0, 1, 2, 3)),
    ),
)
def test_checkpoint_planning_preserves_complete_tool_suffix_at_exact_boundary(
    retained_chars,
    expected_omitted_indexes,
) -> None:
    events = []

    def add(event_type, *, attempt_id="attempt-history", **payload):
        store_seq = len(events) + 1
        events.append(
            {
                "type": event_type,
                "event_id": f"event-{store_seq}",
                "store_seq": store_seq,
                "attempt_id": attempt_id,
                "run_id": attempt_id,
                **payload,
            }
        )

    add("message.user", message={"role": "user", "content": "x" * 30_000})
    add("message.assistant", message={"role": "assistant", "content": "Done."})
    add(
        "message.user",
        message={"role": "user", "content": "y" * retained_chars},
    )
    add(
        "tool_call",
        call_id="call-boundary",
        tool_name="lookup",
        arguments={"query": "price"},
    )
    add(
        "tool_result",
        call_id="call-boundary",
        tool_name="lookup",
        result={"preview": "z" * 500},
        full_output_ref={
            "kind": "artifact",
            "id": "output-boundary",
            "revision": 1,
        },
        result_bytes=500,
        result_sha256="a" * 64,
    )
    add("message.assistant", message={"role": "assistant", "content": "Done."})
    add(
        "message.user",
        attempt_id="attempt-1",
        message={"role": "user", "content": "current"},
    )
    source_events = [
        event for event in events if event["type"].startswith("message.")
    ]
    request = replace(
        _request(pressured=True),
        source_messages=tuple(event["message"] for event in source_events),
        source_event_ids=tuple(event["event_id"] for event in source_events),
        source_event_store_seqs=tuple(
            event["store_seq"] for event in source_events
        ),
        semantic_events=tuple(events),
    )
    checkpoints = RecordingCheckpointRepository()

    result = _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert (
        result.diagnostics["omitted_source_indexes"]
        == expected_omitted_indexes
    )
    assert len(checkpoints.calls) == 1
    assert len(checkpoints.commit_calls) == 1


def test_checkpoint_prepare_must_match_its_prospective_ref() -> None:
    class IdentityChangingRepository(RecordingCheckpointRepository):
        def prepare(self, **kwargs):
            prepared = super().prepare(**kwargs)
            return replace(
                prepared,
                checkpoint_ref=ResourceRef(
                    "checkpoint",
                    "changed-after-preview",
                    1,
                ),
            )

    checkpoints = IdentityChangingRepository()
    request = _request(pressured=True)

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="changed its prospective ref",
    ):
        _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert len(checkpoints.calls) == 1
    assert checkpoints.commit_calls == []


def test_committed_checkpoint_is_reused_for_a_larger_budget_and_new_suffix() -> None:
    checkpoints = RecordingCheckpointRepository()
    first_request = _request(pressured=True)
    _coordinator(
        request=first_request,
        checkpoints=checkpoints,
    ).compile(first_request)

    next_request = replace(
        first_request,
        source_messages=(
            *first_request.source_messages,
            {"role": "user", "content": "next current request"},
        ),
        source_event_ids=("event-1", "event-2", "event-3", "event-4"),
        source_event_store_seqs=(1, 2, 3, 4),
        budget=resolve_context_budget(context_window_tokens=131_072),
        build_id="build-2",
    )
    result = _coordinator(
        request=next_request,
        checkpoints=checkpoints,
    ).compile(next_request)

    assert len(checkpoints.calls) == 1
    assert result.envelope is not None
    assert result.envelope.checkpoint_refs == (checkpoints.commit_calls[0].checkpoint_ref,)
    assert result.messages[-1]["content"] == "next current request"
    assert all(message.get("content") != first_request.source_messages[0]["content"] for message in result.messages)
    assert result.diagnostics["omitted_source_indexes"] == (0, 1)
    assert len(checkpoints.read_calls) == 1
    assert checkpoints.read_calls[0][2] <= 65_536


def test_checkpoint_reuse_does_not_first_compile_the_unbound_history() -> None:
    checkpoints = RecordingCheckpointRepository()
    first_request = _request(pressured=True)
    _coordinator(request=first_request, checkpoints=checkpoints).compile(first_request)
    next_request = replace(
        first_request,
        source_messages=(
            *first_request.source_messages,
            {"role": "user", "content": "next current request"},
        ),
        source_event_ids=("event-1", "event-2", "event-3", "event-4"),
        source_event_store_seqs=(1, 2, 3, 4),
        budget=resolve_context_budget(context_window_tokens=131_072),
        build_id="build-no-unbound-reuse",
    )
    coordinator = _coordinator(request=next_request, checkpoints=checkpoints)
    original_compile_pass = coordinator._compile_pass
    checkpoint_bindings = []

    def record_compile_pass(
        request,
        *,
        checkpoint_binding=None,
        journal_projection=None,
    ):
        checkpoint_bindings.append(checkpoint_binding)
        return original_compile_pass(
            request,
            checkpoint_binding=checkpoint_binding,
            journal_projection=journal_projection,
        )

    coordinator._compile_pass = record_compile_pass

    coordinator.compile(next_request)

    assert len(checkpoint_bindings) == 1
    assert checkpoint_bindings[0] is not None


def test_same_execution_revalidates_the_checkpoint_payload() -> None:
    checkpoints = RecordingCheckpointRepository()
    request = _request(pressured=True)
    coordinator = _coordinator(request=request, checkpoints=checkpoints)

    coordinator.compile(request)
    coordinator.compile(request)
    coordinator.compile(request)

    assert len(checkpoints.read_calls) == 2


def test_same_execution_rejects_a_checkpoint_corrupted_after_warm_reuse() -> None:
    checkpoints = RecordingCheckpointRepository()
    request = _request(pressured=True)
    coordinator = _coordinator(request=request, checkpoints=checkpoints)

    coordinator.compile(request)
    coordinator.compile(request)
    checkpoints.contents[checkpoints.commit_calls[0].checkpoint_ref] = b"{"

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="committed checkpoint payload is invalid",
    ):
        coordinator.compile(request)


def test_checkpoint_reuse_reads_large_payload_in_bounded_pages() -> None:
    checkpoints = RecordingCheckpointRepository()
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "user", "content": "old " + ("x" * 100_000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ),
    )
    _coordinator(request=request, checkpoints=checkpoints).compile(request)
    next_request = replace(
        request,
        source_messages=(*request.source_messages, {"role": "user", "content": "next"}),
        source_event_ids=("event-1", "event-2", "event-3", "event-4"),
        source_event_store_seqs=(1, 2, 3, 4),
        budget=resolve_context_budget(context_window_tokens=131_072),
        build_id="build-large-checkpoint",
    )

    result = _coordinator(request=next_request, checkpoints=checkpoints).compile(next_request)

    assert result.envelope is not None
    assert result.envelope.checkpoint_refs == (checkpoints.commit_calls[0].checkpoint_ref,)
    assert len(checkpoints.read_calls) >= 2
    assert all(limit <= 65_536 for _ref, _offset, limit in checkpoints.read_calls)
    assert [offset for _ref, offset, _limit in checkpoints.read_calls] == sorted(
        offset for _ref, offset, _limit in checkpoints.read_calls
    )


def test_checkpoint_reuse_matches_the_full_multi_turn_prefix() -> None:
    checkpoints = RecordingCheckpointRepository()
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "user", "content": "old one " + ("x" * 30_000)},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "old two " + ("y" * 30_000)},
            {"role": "assistant", "content": "answer two"},
            {"role": "user", "content": "current"},
        ),
        source_event_ids=tuple(f"event-{index}" for index in range(1, 6)),
        source_event_store_seqs=tuple(range(1, 6)),
    )
    first = _coordinator(request=request, checkpoints=checkpoints).compile(request)
    reused = _coordinator(
        request=replace(
            request,
            budget=resolve_context_budget(context_window_tokens=131_072),
            build_id="build-multi-turn-reuse",
        ),
        checkpoints=checkpoints,
    ).compile(
        replace(
            request,
            budget=resolve_context_budget(context_window_tokens=131_072),
            build_id="build-multi-turn-reuse",
        )
    )

    assert first.envelope is not None
    assert len(checkpoints.calls) == 1
    assert reused.envelope is not None
    assert reused.envelope.checkpoint_refs == first.envelope.checkpoint_refs
    assert reused.diagnostics["omitted_source_indexes"] == (0, 1, 2, 3)


def test_growing_suffix_creates_a_new_checkpoint_instead_of_failing() -> None:
    checkpoints = RecordingCheckpointRepository()
    first_request = _request(pressured=True)
    first = _coordinator(
        request=first_request,
        checkpoints=checkpoints,
    ).compile(first_request)
    growing = replace(
        first_request,
        source_messages=(
            *first_request.source_messages[:2],
            {"role": "user", "content": "new large " + ("z" * 30_000)},
            {"role": "assistant", "content": "new answer"},
            {"role": "user", "content": "next"},
        ),
        source_event_ids=tuple(f"event-{index}" for index in range(1, 6)),
        source_event_store_seqs=tuple(range(1, 6)),
        build_id="build-growing-suffix",
    )
    result = _coordinator(request=growing, checkpoints=checkpoints).compile(growing)

    assert first.envelope is not None
    assert result.envelope is not None
    assert len(checkpoints.calls) == 2
    assert result.envelope.checkpoint_refs == (checkpoints.commit_calls[-1].checkpoint_ref,)


def test_old_completed_tool_history_does_not_block_checkpoint_creation() -> None:
    events = [
        {
            "type": "message.user",
            "event_id": "event-1",
            "store_seq": 1,
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "message": {"role": "user", "content": "old " + ("x" * 30_000)},
        }
    ]
    for index in range(20):
        events.extend(
            (
                {
                    "type": "tool_call",
                    "event_id": f"old-call-{index}",
                    "store_seq": 2 + (index * 2),
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "call_id": f"old-call-{index}",
                    "tool_name": "lookup",
                    "arguments": {"query": str(index)},
                },
                {
                    "type": "tool_result",
                    "event_id": f"old-result-{index}",
                    "store_seq": 3 + (index * 2),
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "call_id": f"old-call-{index}",
                    "tool_name": "lookup",
                    "result": {"preview": "z" * 1_200},
                    "full_output_ref": {
                        "kind": "artifact",
                        "id": f"old-output-{index}",
                        "revision": 1,
                    },
                    "result_bytes": 1_200,
                    "result_sha256": "a" * 64,
                },
            )
        )
    events.extend(
        (
            {
                "type": "message.assistant",
                "event_id": "event-42",
                "store_seq": 42,
                "attempt_id": "attempt-history",
                "run_id": "attempt-history",
                "message": {"role": "assistant", "content": "old answer"},
            },
            {
                "type": "message.user",
                "event_id": "event-43",
                "store_seq": 43,
                "attempt_id": "attempt-1",
                "run_id": "attempt-1",
                "message": {"role": "user", "content": "current"},
            },
        )
    )
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "user", "content": "old " + ("x" * 30_000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ),
        source_event_ids=("event-1", "event-42", "event-43"),
        source_event_store_seqs=(1, 42, 43),
        semantic_events=tuple(events),
    )
    checkpoints = RecordingCheckpointRepository()

    result = _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert result.envelope is not None
    assert len(checkpoints.calls) == 1
    assert result.envelope.checkpoint_refs == (checkpoints.commit_calls[0].checkpoint_ref,)


@pytest.mark.parametrize("old_turn_count", (2, 3))
def test_multiple_tool_heavy_old_turns_expand_checkpoint_coverage(
    old_turn_count,
) -> None:
    events = []

    def add(event_type, **payload):
        store_seq = len(events) + 1
        events.append(
            {
                "type": event_type,
                "event_id": f"event-{store_seq}",
                "store_seq": store_seq,
                "attempt_id": "attempt-history",
                "run_id": "attempt-history",
                **payload,
            }
        )

    for turn_index in range(old_turn_count):
        add(
            "message.user",
            message={"role": "user", "content": f"old task {turn_index}"},
        )
        for pair_index in range(20):
            call_id = f"call-{turn_index}-{pair_index}"
            add(
                "tool_call",
                call_id=call_id,
                tool_name="lookup",
                arguments={"query": call_id},
            )
            add(
                "tool_result",
                call_id=call_id,
                tool_name="lookup",
                result={"preview": "z" * 1_200},
                full_output_ref={
                    "kind": "artifact",
                    "id": f"output-{call_id}",
                    "revision": 1,
                },
                result_bytes=1_200,
                result_sha256="a" * 64,
            )
        add(
            "message.assistant",
            message={"role": "assistant", "content": "Done."},
        )
    add("message.user", message={"role": "user", "content": "current"})
    events[-1]["attempt_id"] = "attempt-1"
    events[-1]["run_id"] = "attempt-1"
    source_events = [
        event for event in events if event["type"].startswith("message.")
    ]
    request = replace(
        _request(pressured=False),
        source_messages=tuple(event["message"] for event in source_events),
        source_event_ids=tuple(event["event_id"] for event in source_events),
        source_event_store_seqs=tuple(
            event["store_seq"] for event in source_events
        ),
        semantic_events=tuple(events),
    )
    checkpoints = RecordingCheckpointRepository()

    result = _coordinator(request=request, checkpoints=checkpoints).compile(request)

    assert result.envelope is not None
    assert len(checkpoints.calls) == 1
    assert result.envelope.checkpoint_refs == (
        checkpoints.commit_calls[0].checkpoint_ref,
    )
    assert result.diagnostics["omitted_source_indexes"] == tuple(
        range(old_turn_count * 2)
    )


def test_tool_heavy_suffix_grows_an_existing_checkpoint() -> None:
    checkpoints = RecordingCheckpointRepository()
    first_request = _request(pressured=True)
    first = _coordinator(
        request=first_request,
        checkpoints=checkpoints,
    ).compile(first_request)
    events = [
        {
            "type": "message.user",
            "event_id": "event-1",
            "store_seq": 1,
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "message": dict(first_request.source_messages[0]),
        },
        {
            "type": "message.assistant",
            "event_id": "event-2",
            "store_seq": 2,
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "message": dict(first_request.source_messages[1]),
        },
        {
            "type": "message.user",
            "event_id": "event-3",
            "store_seq": 3,
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "message": {"role": "user", "content": "next old task"},
        },
    ]
    for pair_index in range(20):
        call_id = f"suffix-call-{pair_index}"
        events.extend(
            (
                {
                    "type": "tool_call",
                    "event_id": f"event-{len(events) + 1}",
                    "store_seq": len(events) + 1,
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "call_id": call_id,
                    "tool_name": "lookup",
                    "arguments": {"query": call_id},
                },
                {
                    "type": "tool_result",
                    "event_id": f"event-{len(events) + 2}",
                    "store_seq": len(events) + 2,
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "call_id": call_id,
                    "tool_name": "lookup",
                    "result": {"preview": "z" * 1_200},
                    "full_output_ref": {
                        "kind": "artifact",
                        "id": f"output-{call_id}",
                        "revision": 1,
                    },
                    "result_bytes": 1_200,
                    "result_sha256": "a" * 64,
                },
            )
        )
    assistant_seq = len(events) + 1
    events.append(
        {
            "type": "message.assistant",
            "event_id": f"event-{assistant_seq}",
            "store_seq": assistant_seq,
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "message": {"role": "assistant", "content": "Done."},
        }
    )
    current_seq = len(events) + 1
    events.append(
        {
            "type": "message.user",
            "event_id": f"event-{current_seq}",
            "store_seq": current_seq,
            "attempt_id": "attempt-1",
            "run_id": "attempt-1",
            "message": {"role": "user", "content": "current"},
        }
    )
    source_events = [
        event for event in events if event["type"].startswith("message.")
    ]
    growing = replace(
        first_request,
        source_messages=tuple(event["message"] for event in source_events),
        source_event_ids=tuple(event["event_id"] for event in source_events),
        source_event_store_seqs=tuple(
            event["store_seq"] for event in source_events
        ),
        semantic_events=tuple(events),
        build_id="build-tool-heavy-suffix",
    )

    result = _coordinator(
        request=growing,
        checkpoints=checkpoints,
    ).compile(growing)

    assert first.envelope is not None
    assert result.envelope is not None
    assert len(checkpoints.calls) == 2
    assert result.envelope.checkpoint_refs == (
        checkpoints.commit_calls[-1].checkpoint_ref,
    )
    assert result.diagnostics["omitted_source_indexes"] == (0, 1, 2, 3)


def test_corrupt_committed_checkpoint_fails_before_context_build() -> None:
    checkpoints = RecordingCheckpointRepository()
    request = _request(pressured=True)
    _coordinator(request=request, checkpoints=checkpoints).compile(request)
    checkpoints.contents[checkpoints.commit_calls[0].checkpoint_ref] = b"{"

    with pytest.raises(
        ContextCompileCoordinatorError,
        match="committed checkpoint payload is invalid",
    ):
        _coordinator(request=request, checkpoints=checkpoints).compile(request)


def test_journal_projected_history_survives_the_checkpoint_round_trip() -> None:
    checkpoints = RecordingCheckpointRepository()
    builds = RecordingBuildRepository()
    request = ContextCompileRequest(
        case="durable-projected-history",
        source_messages=({"role": "user", "content": "current"},),
        current_generation="generation-1",
        semantic_events=(
            {
                "type": "message.user",
                "event_id": "event-1",
                "store_seq": 1,
                "generation_id": "generation-1",
                "attempt_id": "attempt-history",
                "run_id": "attempt-history",
                "message": {
                    "role": "user",
                    "content": "old " + ("x" * 30_000),
                },
            },
            {
                "type": "message.assistant",
                "event_id": "event-2",
                "store_seq": 2,
                "generation_id": "generation-1",
                "attempt_id": "attempt-history",
                "run_id": "attempt-history",
                "message": {"role": "assistant", "content": "old answer"},
            },
            {
                "type": "message.user",
                "event_id": "event-3",
                "store_seq": 3,
                "generation_id": "generation-1",
                "attempt_id": "attempt-1",
                "run_id": "attempt-1",
                "message": {"role": "user", "content": "current"},
            },
        ),
        budget=resolve_context_budget(context_window_tokens=8_192),
        source_message_cursors=(SourceMessageCursor(0, "event-3", 3),),
        provider="openai",
        model="synthetic",
        build_id="build-projected",
        execution_id="execution-1",
        generation_id="generation-1",
        attempt_id="attempt-1",
    )

    result = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
    ).compile(request)

    assert len(checkpoints.calls) == 1
    payload = json.loads(checkpoints.calls[0]["summary"])
    assert [message["content"] for message in payload["source_messages"]] == [
        "old " + ("x" * 30_000),
        "old answer",
    ]
    assert result.messages[-1]["content"] == "current"
    assert result.checkpoint_requests == ()


def test_context_runtime_can_use_the_coordinator_as_its_single_compiler() -> None:
    checkpoints = RecordingCheckpointRepository()
    builds = RecordingBuildRepository()
    request = _request(pressured=True)
    coordinator = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
    )
    runtime = ContextRuntime(
        owner_id="context-v2",
        request_factory=lambda context: request,
        durable_event_sink=lambda event: None,
        partial_attempt_sink=lambda event, error: None,
        compiler=coordinator,
    )

    result = runtime.compile_context(
        SimpleNamespace(
            event={"session_id": "execution-1", "attempt_id": "attempt-1"}
        )
    )

    assert result.checkpoint_requests == ()
    assert len(result.envelope.checkpoint_refs) == 1
    assert len(checkpoints.calls) == 1
    assert len(builds.calls) == 1


def test_repeated_compile_uses_identical_checkpoint_and_build_operations() -> None:
    checkpoints = RecordingCheckpointRepository()
    builds = RecordingBuildRepository()
    request = _request(pressured=True)
    coordinator = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
    )

    first = coordinator.compile(request)
    second = coordinator.compile(request)

    assert first.to_dict() == second.to_dict()
    assert len(checkpoints.calls) == 1
    assert len(checkpoints.commit_calls) == 1
    assert len(builds.calls) == 1


def test_one_input_receipt_cannot_authorize_two_distinct_model_builds() -> None:
    checkpoints = RecordingCheckpointRepository()
    builds = RecordingBuildRepository()
    partials = []
    request = _request(pressured=True)
    coordinator = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
        partials=partials,
    )

    first = coordinator.compile(request)
    assert first.envelope is not None

    with pytest.raises(ContextCompileCoordinatorError, match="input receipt"):
        coordinator.compile(
            replace(request, build_id="build-distinct-invocation")
        )

    assert len(checkpoints.calls) == 1
    assert len(checkpoints.commit_calls) == 1
    assert len(builds.calls) == 1
    assert len(partials) == 1


def test_input_trigger_operation_identity_closes_concurrent_claim_race() -> None:
    class RacingBuildRepository(RecordingBuildRepository):
        def get_by_trigger(self, *, trigger_cursor):
            return None

    builds = RacingBuildRepository()
    partials = []
    request = _request(pressured=True)
    coordinator = _coordinator(
        request=request,
        builds=builds,
        partials=partials,
    )

    coordinator.compile(request)

    with pytest.raises(RuntimeError, match="build operation conflict"):
        coordinator.compile(
            replace(request, build_id="build-concurrent-claim")
        )

    assert len(builds.calls) == 1
    assert len(partials) == 1


def test_prepare_success_then_transport_error_recovers_by_operation_receipt() -> None:
    failure = RuntimeError("prepare response lost")

    class UncertainPrepareRepository(RecordingCheckpointRepository):
        def prepare(self, **kwargs):
            receipt = super().prepare(**kwargs)
            raise failure

    checkpoints = UncertainPrepareRepository()
    builds = RecordingBuildRepository()
    partials = []
    request = _request(pressured=True)

    result = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
        partials=partials,
    ).compile(request)

    assert result.checkpoint_requests == ()
    assert len(checkpoints.calls) == 1
    assert len(checkpoints.commit_calls) == 1
    assert len(builds.calls) == 1
    assert partials == []


def test_commit_success_then_transport_error_recovers_without_recommit() -> None:
    failure = RuntimeError("commit response lost")

    class UncertainCommitRepository(RecordingCheckpointRepository):
        def commit(self, *, prepared):
            super().commit(prepared=prepared)
            raise failure

    checkpoints = UncertainCommitRepository()
    builds = RecordingBuildRepository()
    partials = []
    request = _request(pressured=True)

    result = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
        partials=partials,
    ).compile(request)

    assert result.checkpoint_requests == ()
    assert len(checkpoints.commit_calls) == 1
    assert len(builds.calls) == 1
    assert partials == []


def test_build_success_then_transport_error_recovers_recorded_envelope() -> None:
    failure = RuntimeError("build response lost")

    class UncertainBuildRepository(RecordingBuildRepository):
        def record(self, **kwargs):
            super().record(**kwargs)
            raise failure

    checkpoints = RecordingCheckpointRepository()
    builds = UncertainBuildRepository()
    partials = []
    request = _request(pressured=True)

    result = _coordinator(
        request=request,
        checkpoints=checkpoints,
        builds=builds,
        partials=partials,
    ).compile(request)

    assert result.envelope == next(iter(builds.receipts.values())).envelope
    assert len(builds.calls) == 1
    assert partials == []


def test_checkpoint_persistence_failure_is_marked_partial_and_preserves_identity() -> None:
    failure = RuntimeError("checkpoint store unavailable")
    checkpoints = RecordingCheckpointRepository(failure=failure)
    builds = RecordingBuildRepository()
    partials = []
    request = _request(pressured=True)

    with pytest.raises(RuntimeError) as raised:
        _coordinator(
            request=request,
            checkpoints=checkpoints,
            builds=builds,
            partials=partials,
        ).compile(request)

    assert raised.value is failure
    assert getattr(failure, "_unchain_durable_persistence_failure") is True
    assert partials == [(_request(pressured=True), failure)]
    assert builds.calls == []


def test_context_build_failure_is_marked_partial_and_never_returns_model_input() -> None:
    failure = RuntimeError("context build store unavailable")
    builds = RecordingBuildRepository(failure=failure)
    partials = []
    request = _request(pressured=False)

    with pytest.raises(RuntimeError) as raised:
        _coordinator(
            request=request,
            builds=builds,
            partials=partials,
        ).compile(request)

    assert raised.value is failure
    assert getattr(failure, "_unchain_durable_persistence_failure") is True
    assert partials == [(_request(pressured=False), failure)]


def test_unmarkable_repository_failure_uses_returned_safe_boundary_wrapper() -> None:
    class UnmarkableRepositoryError(Exception):
        def __setattr__(self, name, value):
            if name == "_unchain_durable_persistence_failure":
                raise TypeError("immutable exception")
            super().__setattr__(name, value)

    failure = UnmarkableRepositoryError("repository secret")
    builds = RecordingBuildRepository(failure=failure)
    partials = []
    request = _request(pressured=False)

    with pytest.raises(DurablePersistenceBoundaryError) as raised:
        _coordinator(
            request=request,
            builds=builds,
            partials=partials,
        ).compile(request)

    assert raised.value.original is failure
    assert partials == [(_request(pressured=False), raised.value)]


def test_repository_scope_mismatch_fails_before_compilation_or_persistence() -> None:
    checkpoints = RecordingCheckpointRepository()
    checkpoints.execution_id = "different-execution"
    request = _request(pressured=False)

    with pytest.raises(ContextCompileCoordinatorError, match="execution scope"):
        _coordinator(
            request=request,
            checkpoints=checkpoints,
        ).compile(request)

    assert checkpoints.calls == []


def test_checkpoint_payload_hash_matches_the_bytes_sent_to_storage() -> None:
    checkpoints = RecordingCheckpointRepository()
    request = _request(pressured=True)
    journal = _journal_for_request(request)
    _coordinator(
        request=request,
        journal=journal,
        checkpoints=checkpoints,
    ).compile(request)

    call = checkpoints.calls[0]
    summary = json.loads(call["summary"])
    expected_payload = {
        "checkpoint_request": summary["checkpoint_request"],
        "summary_sha256": hashlib.sha256(call["summary"].encode("utf-8")).hexdigest(),
        "refs": [ref.to_dict() for ref in call["refs"]],
    }
    assert call["operation"].payload_sha256 == _canonical_sha256(expected_payload)


def test_checkpoint_payload_uses_exact_sparse_message_cursors() -> None:
    request = replace(
        _request(pressured=True),
        source_messages=(
            {"role": "system", "content": "current policy"},
            {"role": "user", "content": "old " + ("x" * 30_000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ),
        source_event_ids=(),
        source_event_store_seqs=(),
        source_message_cursors=(
            SourceMessageCursor(1, "event-1", 11),
            SourceMessageCursor(2, "event-2", 14),
            SourceMessageCursor(3, "event-3", 19),
        ),
    )
    checkpoints = RecordingCheckpointRepository()

    _coordinator(request=request, checkpoints=checkpoints).compile(request)

    payload = json.loads(checkpoints.calls[0]["summary"])
    assert payload["source_messages"] == list(request.source_messages[1:3])
    assert payload["checkpoint_request"]["source_event_store_seqs"] == [11, 14]
