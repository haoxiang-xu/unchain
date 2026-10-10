from __future__ import annotations

import json
from hashlib import sha256

import pytest

from unchain.context import (
    ContextCompileRequest,
    ContextCompiler,
    ContextCompilerError,
    resolve_context_budget,
)
from unchain.context.compiler import JournalMessageProjectionError
from unchain.journal import AttemptRef, EventCursor, GenerationRef, ResourceRef


def _request(*events: dict) -> ContextCompileRequest:
    return ContextCompileRequest(
        case="compiler-fail-closed",
        source_messages=({"role": "user", "content": "current"},),
        current_generation="generation-1",
        semantic_events=events,
        budget=resolve_context_budget(context_window_tokens=20_000),
        provider="openai",
        model="synthetic",
        build_id="build-fail-closed",
        execution_id="execution-1",
        generation_id="generation-1",
        attempt_id="attempt-current",
    )


def _event(
    event_type: str,
    event_id: str,
    store_seq: int,
    *,
    attempt_id: str = "attempt-history",
    run_id: str | None = None,
    **payload: object,
) -> dict:
    return {
        "type": event_type,
        "event_id": event_id,
        "store_seq": store_seq,
        "execution_id": "execution-1",
        "generation_id": "generation-1",
        "attempt_id": attempt_id,
        "run_id": run_id or attempt_id,
        **payload,
    }


def _history_payload(messages: tuple[dict, ...]) -> dict:
    message = next(
        item
        for item in messages
        if "MEMORY_V2_UNTRUSTED_HISTORY" in str(item.get("content") or "")
    )
    return json.loads(message["content"].split("\n", 2)[2])


def _subagent_settlement(*, outputs: dict, expected_outputs: dict) -> dict:
    """Build a settlement whose descriptor is bound to expected child bytes."""

    consumer = AttemptRef(
        GenerationRef("execution-1", "generation-1"), "attempt-current"
    )
    source = AttemptRef(
        GenerationRef("execution-1", "generation-1"), "attempt-child"
    )
    completion = {
        "schema": "unchain.skeleton_child_completion.v1",
        "status": "completed",
        "outputs": expected_outputs,
        "error": "",
        "summary": expected_outputs["message"],
    }
    encoded = json.dumps(
        completion, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return _event(
        "skeleton.subagent_tool_settlement",
        "event-settlement",
        2,
        attempt_id="attempt-current",
        schema="unchain.skeleton_subagent_tool_settlement.v2",
        consumer_attempt=consumer.to_dict(),
        source_attempt=source.to_dict(),
        call_id="call-child",
        tool_name="delegate_to_subagent",
        status="completed",
        outputs=outputs,
        error="",
        summary=expected_outputs["message"],
        full_output_ref=ResourceRef("artifact", "child-output", 1).to_dict(),
        result_bytes=len(encoded),
        result_sha256=sha256(encoded).hexdigest(),
        handoff_cursor=EventCursor(1, "event-handoff").to_dict(),
    )


def test_subagent_settlement_rejects_output_not_bound_to_child_artifact() -> None:
    event = _subagent_settlement(
        outputs={"message": "FORGED_CHILD_RESULT"},
        expected_outputs={"message": "child done"},
    )

    with pytest.raises(ContextCompilerError, match="does not match its child artifact"):
        ContextCompiler().compile(_request(event))


@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("execution_id", "execution identity"),
        ("generation_id", "generation identity"),
    ],
)
def test_canonical_chat_event_requires_explicit_scope_identity(
    missing_field: str,
    expected_error: str,
) -> None:
    event = _event(
        "message.user",
        "event-1",
        1,
        message={"role": "user", "content": "unbound history"},
    )
    event.pop(missing_field)

    with pytest.raises(ContextCompilerError, match=expected_error):
        ContextCompiler().compile(_request(event))


@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("execution_id", "execution identity"),
        ("generation_id", "generation identity"),
    ],
)
def test_tool_history_requires_explicit_scope_identity(
    missing_field: str,
    expected_error: str,
) -> None:
    artifact_ref = ResourceRef("artifact", "tool-result-1", 1).to_dict()
    events = [
        _event(
            "tool_call",
            "event-1",
            1,
            call_id="call-1",
            tool_name="lookup",
            arguments={"query": "cross-generation"},
        ),
        _event(
            "tool_result",
            "event-2",
            2,
            call_id="call-1",
            tool_name="lookup",
            result={"preview": "cross-generation"},
            full_output_ref=artifact_ref,
            result_bytes=16,
            result_sha256="a" * 64,
        ),
    ]
    for event in events:
        event.pop(missing_field)

    with pytest.raises(ContextCompilerError, match=expected_error):
        ContextCompiler().compile(_request(*events))


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        (
            "artifact.recorded",
            {
                "artifact_ref": ResourceRef(
                    "artifact",
                    "artifact-1",
                    1,
                ).to_dict(),
                "artifact": {"name": "cross-generation artifact"},
            },
        ),
        (
            "handoff.recorded",
            {
                "child_run_id": "child-run-1",
                "status": "complete",
                "summary": "cross-generation handoff",
                "full_output_ref": ResourceRef(
                    "artifact",
                    "handoff-1",
                    1,
                ).to_dict(),
            },
        ),
    ],
)
@pytest.mark.parametrize(
    ("missing_field", "expected_error"),
    [
        ("execution_id", "execution identity"),
        ("generation_id", "generation identity"),
    ],
)
def test_artifact_and_handoff_history_require_explicit_scope_identity(
    event_type: str,
    payload: dict,
    missing_field: str,
    expected_error: str,
) -> None:
    event = _event(event_type, "event-1", 1, **payload)
    event.pop(missing_field)

    with pytest.raises(ContextCompilerError, match=expected_error):
        ContextCompiler().compile(_request(event))


def test_root_assistant_after_success_terminal_fails_closed() -> None:
    terminal = _event(
        "run_completed",
        "event-1",
        1,
        status="completed",
    )
    late_assistant = _event(
        "message.assistant",
        "event-2",
        2,
        message={"role": "assistant", "content": "late output"},
    )

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(_request(terminal, late_assistant))

    assert raised.value.reason == "message_lifecycle_invalid"


def test_terminal_event_without_a_cursor_fails_before_eligibility_filtering() -> None:
    terminal = _event("run_failed", "event-1", 1)
    terminal.pop("event_id")
    terminal.pop("store_seq")

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(_request(terminal))

    assert raised.value.reason == "event_cursor_invalid"


def test_partial_cursor_is_rejected_for_an_ignored_v2_event() -> None:
    ignored = _event("diagnostic.ignored", "event-1", 1)
    ignored.pop("store_seq")

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(_request(ignored))

    assert raised.value.reason == "event_cursor_invalid"


def test_partial_cursor_is_rejected_without_build_identity() -> None:
    request = ContextCompileRequest(
        case="unbound-partial-cursor",
        source_messages=({"role": "user", "content": "current"},),
        semantic_events=(
            {
                "type": "tool_call",
                "event_id": "event-partial",
                "call_id": "call-1",
                "tool_name": "lookup",
                "arguments": {},
            },
        ),
        budget=resolve_context_budget(context_window_tokens=20_000),
    )

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(request)

    assert raised.value.reason == "event_cursor_invalid"


def test_child_runtime_events_cannot_bypass_the_handoff_boundary() -> None:
    artifact_ref = ResourceRef("artifact", "artifact-child", 1).to_dict()
    child_scope = {
        "attempt_id": "attempt-current",
        "run_id": "child-run",
        "parent_run_id": "attempt-current",
    }
    events = (
        _event(
            "tool_call",
            "event-1",
            1,
            call_id="call-child",
            tool_name="lookup",
            arguments={"query": "private child state"},
            **child_scope,
        ),
        _event(
            "tool_result",
            "event-2",
            2,
            call_id="call-child",
            tool_name="lookup",
            result={"preview": "private child result"},
            full_output_ref=artifact_ref,
            result_bytes=20,
            result_sha256="a" * 64,
            **child_scope,
        ),
        _event(
            "artifact.recorded",
            "event-3",
            3,
            artifact_ref=artifact_ref,
            artifact={"name": "private-child-artifact"},
            **child_scope,
        ),
    )

    result = ContextCompiler().compile(_request(*events))

    assert [dict(message) for message in result.messages] == [
        {"role": "user", "content": "current"}
    ]


def test_child_handoff_envelope_is_the_only_child_projection() -> None:
    handoff_ref = ResourceRef("artifact", "handoff-child", 1).to_dict()
    handoff = _event(
        "handoff.recorded",
        "event-1",
        1,
        attempt_id="attempt-current",
        run_id="child-run",
        parent_run_id="attempt-current",
        child_run_id="child-run",
        status="complete",
        summary="bounded child summary",
        full_output_ref=handoff_ref,
    )

    result = ContextCompiler().compile(_request(handoff))

    history = _history_payload(result.messages)
    assert [item["child_run_id"] for item in history["handoffs"]] == [
        "child-run"
    ]


def test_child_tool_result_cannot_override_parent_portable_projection() -> None:
    root_ref = ResourceRef("artifact", "artifact-root", 1).to_dict()
    child_ref = ResourceRef("artifact", "artifact-child", 1).to_dict()
    child_scope = {
        "attempt_id": "attempt-current",
        "run_id": "child-run",
        "parent_run_id": "attempt-current",
    }
    events = (
        _event(
            "tool_call",
            "event-1",
            1,
            call_id="call-shared",
            tool_name="lookup",
            arguments={"query": "root"},
        ),
        _event(
            "tool_result",
            "event-2",
            2,
            call_id="call-shared",
            tool_name="lookup",
            result={"preview": "root result"},
            full_output_ref=root_ref,
            result_bytes=11,
            result_sha256="a" * 64,
        ),
        _event(
            "tool_result",
            "event-3",
            3,
            call_id="call-shared",
            tool_name="lookup",
            result={"preview": "private child result"},
            full_output_ref=child_ref,
            result_bytes=20,
            result_sha256="b" * 64,
            **child_scope,
        ),
    )

    result = ContextCompiler().compile(_request(*events))
    portable = result.projections["unchain.context_v2.comparable.v1"]

    assert portable["closed_tool_exchanges"][0]["full_output_ref"]["id"] == (
        "artifact-root"
    )


def test_exact_duplicate_handoff_is_projected_idempotently_once() -> None:
    handoff = _event(
        "handoff.recorded",
        "event-1",
        1,
        child_run_id="child-run",
        status="complete",
        summary="bounded child summary",
        full_output_ref=ResourceRef("artifact", "handoff-child", 1).to_dict(),
    )

    result = ContextCompiler().compile(_request(handoff, dict(handoff)))

    assert len(_history_payload(result.messages)["handoffs"]) == 1


def test_nested_wrapper_difference_is_not_an_idempotent_duplicate() -> None:
    event = _event(
        "message.user",
        "event-1",
        1,
        message={"role": "user", "content": "history"},
    )
    nested = {
        key: event[key]
        for key in (
            "type",
            "event_id",
            "store_seq",
            "execution_id",
            "generation_id",
            "attempt_id",
            "run_id",
        )
    }
    nested["event"] = dict(event)
    first = {**nested, "wrapper_revision": 1}
    second = {**nested, "wrapper_revision": 2}

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(_request(first, second))

    assert raised.value.reason == "event_payload_conflict"


def test_nested_non_message_cursor_conflict_fails_closed() -> None:
    artifact_ref = ResourceRef("artifact", "artifact-1", 1).to_dict()
    wrapped = {
        "type": "artifact.recorded",
        "event_id": "outer-event",
        "store_seq": 1,
        "execution_id": "execution-1",
        "generation_id": "generation-1",
        "attempt_id": "attempt-history",
        "run_id": "attempt-history",
        "event": {
            "type": "artifact.recorded",
            "event_id": "inner-event",
            "store_seq": 2,
            "execution_id": "execution-1",
            "generation_id": "generation-1",
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "artifact_ref": artifact_ref,
        },
    }

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(_request(wrapped))

    assert raised.value.reason == "event_cursor_conflict"


def test_computer_call_wire_without_an_id_is_not_canonical_chat() -> None:
    event = _event(
        "message.assistant",
        "event-1",
        1,
        message={
            "role": "assistant",
            "content": "looks like ordinary text",
            "computer_call": {"action": "click"},
        },
    )

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(_request(event))

    assert raised.value.reason == "message_payload_invalid"
