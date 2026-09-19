from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from unchain.context import (
    CheckpointProjectionDependency,
    CheckpointProjectionError,
    build_checkpoint_request,
    project_checkpoint_message,
    read_checkpoint_request_record,
)
from unchain.journal import EventCursor, ResourceRef


def _dependency(
    *,
    receipt_store_seq: int = 3,
    event_sha256: str = "a" * 64,
) -> CheckpointProjectionDependency:
    return CheckpointProjectionDependency(
        source_cursor=EventCursor(store_seq=2, event_id="event-2"),
        receipt_cursor=EventCursor(
            store_seq=receipt_store_seq,
            event_id=f"event-{receipt_store_seq}",
        ),
        event_type="run_completed",
        attempt_id="attempt-1",
        purpose="assistant_commit",
        status="completed",
        workflow_node_id="final-node",
        workflow_step_index=1,
        workflow_step_count=2,
        iteration=0,
        event_sha256=event_sha256,
    )


def test_checkpoint_request_is_a_deterministic_manifest_for_an_exact_range() -> None:
    first = build_checkpoint_request(
        source_event_ids=("event-1", "event-2"),
        source_event_store_seqs=(4, 5),
        source_messages=(
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
        ),
    )
    second = build_checkpoint_request(
        source_event_ids=("event-1", "event-2"),
        source_event_store_seqs=(4, 5),
        source_messages=(
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
        ),
    )

    assert first == second
    assert first.source_range.start.event_id == "event-1"
    assert first.source_range.end.store_seq == 5
    assert first.event_count == 2
    assert "old request" not in json.dumps(first.to_dict())
    assert type(first).from_dict(first.to_dict()) == first


def test_checkpoint_v2_request_id_binds_receipt_identity_and_exact_event_hash() -> None:
    common = {
        "source_event_ids": ("event-1", "event-2"),
        "source_event_store_seqs": (1, 2),
        "source_event_sha256s": ("1" * 64, "2" * 64),
        "source_messages": (
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
        ),
    }

    first = build_checkpoint_request(
        **common,
        projection_dependencies=(_dependency(),),
    )
    changed_receipt = build_checkpoint_request(
        **common,
        projection_dependencies=(_dependency(receipt_store_seq=4),),
    )
    changed_event = build_checkpoint_request(
        **common,
        projection_dependencies=(_dependency(event_sha256="b" * 64),),
    )

    assert first.SCHEMA == "unchain.checkpoint_request.v2"
    assert first.request_id != changed_receipt.request_id
    assert first.request_id != changed_event.request_id
    assert first.source_range.end.store_seq == 3
    assert changed_receipt.source_range.end.store_seq == 4
    assert type(first).from_dict(first.to_dict()) == first


def test_direct_checkpoint_manifest_construction_revalidates_its_digest() -> None:
    request = build_checkpoint_request(
        source_event_ids=("event-1", "event-2"),
        source_event_store_seqs=(1, 2),
        source_event_sha256s=("1" * 64, "2" * 64),
        source_messages=(
            {"role": "user", "content": "old request"},
            {"role": "assistant", "content": "old answer"},
        ),
        projection_dependencies=(_dependency(),),
    )

    with pytest.raises(CheckpointProjectionError, match="digest"):
        replace(request, request_id="checkpoint-" + ("f" * 64))
    with pytest.raises(CheckpointProjectionError, match="unaligned"):
        replace(request, source_event_sha256s=("1" * 64,))


@pytest.mark.parametrize(
    "receipt_cursor",
    (
        EventCursor(store_seq=4, event_id="event-1"),
        EventCursor(store_seq=3, event_id="receipt-event"),
    ),
)
def test_checkpoint_dependency_receipt_cannot_alias_any_source_cursor(
    receipt_cursor: EventCursor,
) -> None:
    dependency = replace(_dependency(), receipt_cursor=receipt_cursor)

    with pytest.raises(CheckpointProjectionError, match="cursor"):
        build_checkpoint_request(
            source_event_ids=("event-1", "event-2", "event-3"),
            source_event_store_seqs=(1, 2, 3),
            source_event_sha256s=("1" * 64, "2" * 64, "3" * 64),
            source_messages=(
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ),
            projection_dependencies=(dependency,),
        )


def test_checkpoint_v1_can_be_read_for_audit_but_not_projected_as_authority() -> None:
    legacy_id_payload = {
        "source_event_ids": ["event-1"],
        "source_event_store_seqs": [1],
        "source_messages_sha256": "b" * 64,
    }
    legacy = {
        "schema": "unchain.checkpoint_request.v1",
        "request_id": "checkpoint-"
        + hashlib.sha256(
            json.dumps(
                legacy_id_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "source_range": {
            "schema": "unchain.event_range.v1",
            "start": {
                "schema": "unchain.event_cursor.v1",
                "store_seq": 1,
                "event_id": "event-1",
            },
            "end": {
                "schema": "unchain.event_cursor.v1",
                "store_seq": 1,
                "event_id": "event-1",
            },
        },
        "source_event_ids": ["event-1"],
        "source_event_store_seqs": [1],
        "source_messages_sha256": "b" * 64,
        "event_count": 1,
    }

    audited = read_checkpoint_request_record(legacy)

    assert audited.SCHEMA == "unchain.checkpoint_request.v1"
    assert audited.authoritative is False
    with pytest.raises(CheckpointProjectionError, match="v2"):
        project_checkpoint_message(
            checkpoint_ref=ResourceRef("checkpoint", "checkpoint-legacy", 1),
            request=audited,
            omitted_complete_turns=1,
        )


def test_checkpoint_request_accepts_deeply_frozen_structured_messages() -> None:
    from unchain.context import ContextCompileRequest

    frozen_messages = ContextCompileRequest(
        case="frozen",
        source_messages=(
            {
                "role": "user",
                "content": [{"type": "text", "text": "old request"}],
            },
        ),
    ).source_messages

    request = build_checkpoint_request(
        source_event_ids=("event-1",),
        source_event_store_seqs=(1,),
        source_messages=frozen_messages,
    )

    assert request.event_count == 1


def test_checkpoint_projection_contains_only_untrusted_coverage_and_structured_ref() -> None:
    request = build_checkpoint_request(
        source_event_ids=("event-1", "event-2"),
        source_event_store_seqs=(1, 2),
        source_messages=(
            {"role": "user", "content": "large old body"},
            {"role": "assistant", "content": "large old answer"},
        ),
    )
    ref = ResourceRef("checkpoint", "checkpoint-1", 1)

    message = project_checkpoint_message(
        checkpoint_ref=ref,
        request=request,
        omitted_complete_turns=1,
    )

    assert set(message) == {"role", "content"}
    assert message["role"] == "assistant"
    assert "ends with this message" in message["content"]
    assert "UNTRUSTED_DATA" in message["content"]
    assert "large old body" not in message["content"]
    assert "pupu://" not in message["content"]
    assert '"kind": "checkpoint"' in message["content"]


@pytest.mark.parametrize(
    ("event_ids", "store_seqs"),
    (
        (("event-1",), ()),
        (("event-1", "event-2"), (2, 1)),
        ((" event-1 ",), (1,)),
        (("event-1", "event-1"), (1, 2)),
        ((), ()),
    ),
)
def test_checkpoint_request_rejects_unaligned_or_non_monotonic_ranges(
    event_ids: tuple[str, ...],
    store_seqs: tuple[int, ...],
) -> None:
    with pytest.raises(CheckpointProjectionError):
        build_checkpoint_request(
            source_event_ids=event_ids,
            source_event_store_seqs=store_seqs,
            source_messages=(),
        )


def test_checkpoint_request_requires_one_source_message_per_event() -> None:
    with pytest.raises(CheckpointProjectionError):
        build_checkpoint_request(
            source_event_ids=("event-1", "event-2"),
            source_event_store_seqs=(1, 2),
            source_messages=({"role": "user", "content": "one message"},),
        )
