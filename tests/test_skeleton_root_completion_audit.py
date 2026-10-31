"""The Host's post-terminal audit does not own runnable graph work."""
from __future__ import annotations

from hashlib import sha256

import pytest

from tests.test_skeleton_rebase_quiescence import _events, _seal_root, produced
from unchain.journal import JournalAppendRequest, build_operation_ref
from unchain.skeleton.durable_protocol import canonical_bytes
from unchain.skeleton.rebase_quiescence import (
    SkeletonRebaseQuiescenceError, classify_skeleton_attempts,
)


_STATUSES = {"enqueued": "Enqueued", "noop": "NoOp",
             "isolated": "Isolated", "failed": "Failed"}


def _audit(produced, disposition="noop", *, change=None, suffix=""):
    status = _STATUSES[disposition]
    payload = {
        "schema": "pupu.memory_curator_completion.v1", "backend": "unchain",
        "status": status, "enqueue_status": status, "reason": "feature_disabled",
        "trigger": "completed_root_run", "run_id": produced.attempt.attempt_id,
        "job_id": "test-job" if disposition == "enqueued" else "",
        "candidate_count": 0,
        "worker_status": "Pending" if disposition == "enqueued" else "NotScheduled",
    }
    if change:
        change(payload)
    digest = sha256(canonical_bytes({"payload": payload, "suffix": suffix})).hexdigest()
    operation_id = "memory-curator-audit." + digest
    event_type = "memory.curator." + disposition
    return produced.journal.append(request=JournalAppendRequest(
        event_id="memory-curator-audit-" + digest, event_type=event_type,
        attempt=produced.attempt,
        operation=build_operation_ref(operation_id, domain=event_type, payload=payload),
        payload=payload,
    )).event


@pytest.mark.parametrize("disposition", tuple(_STATUSES))
def test_known_content_free_root_audit_is_read_only_quiescent(produced, disposition):
    _seal_root(produced, commit=False)
    _audit(produced, disposition)
    produced.runner.commit_root()
    events = _events(produced)
    assert classify_skeleton_attempts(events, produced.journal)[produced.attempt.attempt_id] == "quiescent"
    assert _events(produced) == events and len(produced.calls) == 1


@pytest.mark.parametrize("mutation", (
    "extra", "schema", "run", "status", "enqueue_status", "worker_status",
    "count_bool", "count_negative", "job", "reason", "trigger", "backend",
    "duplicate", "before_terminal", "after_ack",
))
def test_root_audit_is_not_a_generic_post_terminal_allowlist(produced, mutation):
    if mutation == "before_terminal":
        # Still post-End, but before the canonical final/terminal pair.
        _audit(produced)
        _seal_root(produced)
    else:
        _seal_root(produced, commit=mutation == "after_ack")
        changes = {
            "extra": ("unknown", True), "schema": ("schema", "future"),
            "run": ("run_id", "foreign-root"), "status": ("status", "Enqueued"),
            "enqueue_status": ("enqueue_status", "Failed"),
            "worker_status": ("worker_status", "Pending"),
            "count_bool": ("candidate_count", True), "count_negative": ("candidate_count", -1),
            "job": ("job_id", {}), "reason": ("reason", []),
            "trigger": ("trigger", "other"), "backend": ("backend", "other"),
        }
        change = (lambda p: p.update([changes[mutation]])) if mutation in changes else None
        _audit(produced, change=change)
        if mutation == "duplicate":
            _audit(produced, suffix="second-record")
        if mutation != "after_ack":
            produced.runner.commit_root()
    before = _events(produced)
    with pytest.raises(SkeletonRebaseQuiescenceError) as failure:
        classify_skeleton_attempts(before, produced.journal)
    assert failure.value.code in {"graph_root_seal", "graph_root_audit"}
    assert _events(produced) == before and len(produced.calls) == 1
