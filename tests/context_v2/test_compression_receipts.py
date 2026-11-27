from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock

import pytest

from unchain.context.checkpoints import (
    CheckpointProjectionDependency,
    build_checkpoint_request,
)
from unchain.context.compression import (
    CompressionOutcomeUnknown,
    ContextCompressionError,
    ContextCompressionService,
    validate_compression_projection,
)
from unchain.journal import (
    AttemptRef,
    EventCursor,
    GenerationRef,
    JournalAppendRequest,
    JournalEvent,
    build_operation_ref,
)
from unchain.journal.snapshot import journal_event_sha256
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


ATTEMPT = AttemptRef(
    GenerationRef("compression-execution", "compression-generation"),
    "compression-attempt",
)
PROFILE = "a" * 64
MESSAGES = ({"role": "user", "content": "durable history"},)


def _journal(tmp_path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )
    return store.bind_execution(ATTEMPT.generation.execution_id)


def _checkpoint(journal):
    source = journal.append(request=_source_request()).event
    return build_checkpoint_request(
        source_event_ids=(source.event_id,),
        source_event_store_seqs=(source.store_seq,),
        source_event_sha256s=(journal_event_sha256(source),),
        source_messages=MESSAGES,
    )


def _source_request():
    payload = {"run_id": ATTEMPT.attempt_id, "message": MESSAGES[0]}
    return JournalAppendRequest(
        event_id="compression-source",
        event_type="message.user",
        attempt=ATTEMPT,
        operation=build_operation_ref(
            "compression-source-operation",
            domain="compression-source",
            payload=payload,
        ),
        payload=payload,
    )


def _service(journal, callback, *, profile=PROFILE):
    return ContextCompressionService(
        journal,
        attempt=ATTEMPT,
        profile_digest=profile,
        summarize=callback,
    )


def test_sqlite_result_is_replayed_cold_and_source_snapshot_is_pre_intent(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    calls = []
    proof = _service(
        journal,
        lambda **kwargs: calls.append(kwargs) or "brief durable summary",
    ).compress(checkpoint, MESSAGES, target_tokens=256)

    assert calls and set(proof) == {
        "schema", "operation_id", "checkpoint_request_id", "profile_digest",
        "target_tokens", "summary",
    }
    replay = _service(
        journal,
        lambda **_kwargs: pytest.fail("cold replay must not call summarizer"),
    )
    assert replay.compress(checkpoint, MESSAGES, target_tokens=256) == proof
    source = replay.source_snapshot(proof)
    assert [event.event_type for event in source.events] == ["message.user"]


def test_cold_intent_is_unknown_and_never_retried(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    with pytest.raises(CompressionOutcomeUnknown):
        _service(journal, lambda **_kwargs: (_ for _ in ()).throw(RuntimeError())).compress(
            checkpoint, MESSAGES, target_tokens=256
        )
    with pytest.raises(CompressionOutcomeUnknown):
        _service(
            journal,
            lambda **_kwargs: pytest.fail("recorded intent must prevent retry"),
        ).compress(checkpoint, MESSAGES, target_tokens=256)


def test_bad_source_hash_rejects_before_callback(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    called = False

    def summarize(**_kwargs):
        nonlocal called
        called = True
        return "brief"

    with pytest.raises(ContextCompressionError):
        _service(journal, summarize).compress(
            checkpoint,
            ({"role": "user", "content": "different"},),
            target_tokens=256,
        )
    assert not called


def test_profile_change_collides_before_a_second_summarizer_call(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    with pytest.raises(CompressionOutcomeUnknown):
        _service(journal, lambda **_kwargs: (_ for _ in ()).throw(RuntimeError())).compress(
            checkpoint, MESSAGES, target_tokens=256
        )
    called = False

    def summarize(**_kwargs):
        nonlocal called
        called = True
        return "brief"

    with pytest.raises(ContextCompressionError):
        _service(journal, summarize, profile="b" * 64).compress(
            checkpoint, MESSAGES, target_tokens=256
        )
    assert not called


def test_durable_receipts_reject_source_written_after_intent(tmp_path):
    journal = _journal(tmp_path)
    request = _source_request()
    anticipated_source = JournalEvent(
        event_id=request.event_id,
        event_type=request.event_type,
        attempt=request.attempt,
        operation=request.operation,
        store_seq=2,
        payload=request.payload,
    )
    checkpoint = build_checkpoint_request(
        source_event_ids=(anticipated_source.event_id,),
        source_event_store_seqs=(anticipated_source.store_seq,),
        source_event_sha256s=(journal_event_sha256(anticipated_source),),
        source_messages=MESSAGES,
    )
    service = _service(journal, lambda **_kwargs: pytest.fail("must not summarize"))
    _, intent, _ = service._requests(checkpoint=checkpoint, target_tokens=256)
    journal.append(request=intent)
    source = journal.append(request=request).event
    assert source.store_seq == 2

    with pytest.raises(ContextCompressionError, match="precede"):
        service.compress(checkpoint, MESSAGES, target_tokens=256)


def test_durable_receipts_reject_result_written_before_intent(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    service = _service(journal, lambda **_kwargs: pytest.fail("must not summarize"))
    _, intent, result = service._requests(
        checkpoint=checkpoint,
        target_tokens=256,
        summary="brief",
    )
    assert result is not None
    journal.append(request=result)
    journal.append(request=intent)

    with pytest.raises(ContextCompressionError, match="must follow"):
        service.compress(checkpoint, MESSAGES, target_tokens=256)


def test_sequence_and_projection_dependency_mismatch_reject_before_callback(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    source = journal.capture_snapshot().events[0]
    completion_payload = {"run_id": ATTEMPT.attempt_id, "status": "completed"}
    completed = journal.append(
        request=JournalAppendRequest(
            event_id="compression-completed",
            event_type="run_completed",
            attempt=ATTEMPT,
            operation=build_operation_ref(
                "compression-completed-operation",
                domain="compression-completed",
                payload=completion_payload,
            ),
            payload=completion_payload,
        )
    ).event
    dependency = CheckpointProjectionDependency(
        source_cursor=EventCursor(source.store_seq, source.event_id),
        receipt_cursor=EventCursor(completed.store_seq, completed.event_id),
        event_type="run_completed",
        attempt_id=ATTEMPT.attempt_id,
        purpose="assistant_commit",
        status="completed",
        event_sha256=journal_event_sha256(completed),
    )
    dependency_checkpoint = build_checkpoint_request(
        source_event_ids=(source.event_id,),
        source_event_store_seqs=(source.store_seq,),
        source_event_sha256s=(journal_event_sha256(source),),
        source_messages=MESSAGES,
        projection_dependencies=(dependency,),
    )
    forged = build_checkpoint_request(
        source_event_ids=(source.event_id,),
        source_event_store_seqs=(completed.store_seq,),
        source_event_sha256s=(journal_event_sha256(source),),
        source_messages=MESSAGES,
    )
    calls = 0

    def summarize(**_kwargs):
        nonlocal calls
        calls += 1
        return "brief"

    with pytest.raises(ContextCompressionError):
        _service(journal, summarize).compress(forged, MESSAGES, target_tokens=256)
    forged_dependency = build_checkpoint_request(
        source_event_ids=dependency_checkpoint.source_event_ids,
        source_event_store_seqs=dependency_checkpoint.source_event_store_seqs,
        source_event_sha256s=dependency_checkpoint.source_event_sha256s,
        source_messages=MESSAGES,
        projection_dependencies=(replace(dependency, event_sha256="0" * 64),),
    )
    with pytest.raises(ContextCompressionError):
        _service(journal, summarize).compress(
            forged_dependency, MESSAGES, target_tokens=256
        )
    assert calls == 0


def test_two_sqlite_writers_invoke_at_most_one_summarizer(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    gate = Barrier(2)
    lock = Lock()
    calls = 0

    def summarize(**_kwargs):
        nonlocal calls
        with lock:
            calls += 1
        return "brief concurrent summary"

    def run():
        gate.wait()
        try:
            return _service(journal, summarize).compress(
                checkpoint, MESSAGES, target_tokens=256
            )
        except CompressionOutcomeUnknown:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _item: run(), range(2)))
    assert calls == 1
    assert {tuple(sorted(item.items())) for item in outcomes if item is not None}


def test_invalid_public_proof_and_wrong_scope_result_are_rejected(tmp_path):
    journal = _journal(tmp_path)
    checkpoint = _checkpoint(journal)
    with pytest.raises(CompressionOutcomeUnknown):
        _service(journal, lambda **_kwargs: (_ for _ in ()).throw(RuntimeError())).compress(
            checkpoint, MESSAGES, target_tokens=256
        )
    intent = next(
        event for event in journal.capture_snapshot().events
        if event.event_type == "context.compression_intent"
    )
    operation_id = intent.operation.operation_id.removesuffix(".intent")
    wrong_attempt = AttemptRef(ATTEMPT.generation, "other-attempt")
    result_payload = {"schema": "unchain.context_compression_result.v1", "intent": dict(intent.payload), "summary": "brief"}
    journal.append(
        request=JournalAppendRequest(
            event_id=f"event.{operation_id}.result",
            event_type="context.compression_result",
            attempt=wrong_attempt,
            operation=build_operation_ref(
                f"{operation_id}.result",
                domain="context.compression_result",
                payload=result_payload,
            ),
            payload=result_payload,
        )
    )
    with pytest.raises(ContextCompressionError):
        _service(journal, lambda **_kwargs: "brief").compress(
            checkpoint, MESSAGES, target_tokens=256
        )
    with pytest.raises(Exception):
        validate_compression_projection(
            {
                "schema": "unchain.context_compression.v1",
                "operation_id": operation_id,
                "checkpoint_request_id": checkpoint.request_id,
                "profile_digest": PROFILE,
                "target_tokens": 256,
                "summary": "brief",
                "unknown": True,
            },
            checkpoint_request_id=checkpoint.request_id,
        )
