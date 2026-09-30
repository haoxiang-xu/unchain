from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from unchain.context import (
    ContextBuildStatus,
    ContextBudgetExceededError,
    ContextCompileRequest,
    ContextCompiler,
    ContextCompilerError,
    SourceMessageCursor,
    resolve_context_budget,
)
from unchain.context.compiler import (
    JournalMessageProjectionError,
    _CheckpointBinding,
)
from unchain.journal import ResourceRef


def _request(
    messages: list[dict],
    *,
    events: list[dict] | None = None,
    task_state: dict | None = None,
    pending_inputs: list[dict] | None = None,
    checkpoint_ref: ResourceRef | None = None,
    source_event_ids: tuple[str, ...] = (),
    source_event_store_seqs: tuple[int, ...] = (),
    source_message_cursors: tuple[SourceMessageCursor, ...] = (),
    window: int = 20_000,
    fixed_overhead_tokens: int = 0,
    build_identity: bool = False,
    provider: str = "openai",
) -> ContextCompileRequest:
    request = ContextCompileRequest(
        case="portable_contract",
        source_messages=tuple(messages),
        fixed_overhead_tokens=fixed_overhead_tokens,
        semantic_events=None if events is None else tuple(events),
        task_state=task_state,
        pending_task_inputs=tuple(pending_inputs or ()),
        budget=resolve_context_budget(context_window_tokens=window),
        source_event_ids=source_event_ids,
        source_event_store_seqs=source_event_store_seqs,
        source_message_cursors=source_message_cursors,
        provider=provider,
        model="synthetic",
        build_id="build-1" if build_identity else None,
        execution_id="execution-1" if build_identity else None,
        generation_id="generation-1" if build_identity else None,
        attempt_id="attempt-1" if build_identity else None,
    )
    if checkpoint_ref is None:
        return request
    preflight = ContextCompiler().compile(request)
    if not preflight.checkpoint_requests:
        return request
    return replace(
        request,
        checkpoint_ref=checkpoint_ref,
        checkpoint_request_id=preflight.checkpoint_requests[0].request_id,
    )


def _marker_payload(messages: tuple[dict, ...], marker: str) -> dict:
    message = next(
        item for item in messages if marker in str(item.get("content") or "")
    )
    return json.loads(message["content"].split("\n", 2)[2])


def _pending_native_tool_request(
    *,
    provider: str,
    source_provider: str | None = None,
    call_ids: tuple[str, ...] = ("call-1",),
    completed_call_ids: tuple[str, ...] | None = None,
    messages: list[dict] | None = None,
    window: int = 20_000,
    checkpoint_ref: ResourceRef | None = None,
    source_event_ids: tuple[str, ...] = (),
    source_event_store_seqs: tuple[int, ...] = (),
    source_events: list[dict] | None = None,
    build_identity: bool = False,
) -> ContextCompileRequest:
    completed = call_ids if completed_call_ids is None else completed_call_ids
    declared_provider = provider if source_provider is None else source_provider
    events: list[dict] = list(source_events or ())
    result_events: dict[str, dict] = {}
    for index, call_id in enumerate(call_ids, start=1):
        events.append(
            {
                "type": "tool_call",
                "event_id": f"event-call-{call_id}",
                "store_seq": 100 + index,
                "iteration": 4,
                "call_id": call_id,
                "tool_name": "approved_write",
                "arguments": {"value": call_id},
                "source_provider": declared_provider,
            }
        )
    for index, call_id in enumerate(completed, start=1):
        result = {"written": call_id}
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        ref = ResourceRef("artifact", f"artifact-{call_id}", 1)
        event = {
            "type": "tool_result",
            "event_id": f"event-result-{call_id}",
            "store_seq": 200 + index,
            "iteration": 4,
            "call_id": call_id,
            "tool_name": "approved_write",
            "result": result,
            "full_output_ref": ref.to_dict(),
            "result_bytes": len(encoded),
            "result_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        result_events[call_id] = event
        events.append(event)
    if build_identity:
        for event in events:
            event.setdefault("execution_id", "execution-1")
            event.setdefault("generation_id", "generation-1")
    pending_call_id = completed[-1]
    pending_result = result_events[pending_call_id]
    return _request(
        messages or [{"role": "user", "content": "perform the approved write"}],
        events=events,
        pending_inputs=[
            {
                "event_id": pending_result["event_id"],
                "store_seq": pending_result["store_seq"],
                "type": "tool_result",
                "preview": "result preview that must not be duplicated",
                "preview_truncated": False,
                "content_ref": pending_result["full_output_ref"],
                "content_bytes": pending_result["result_bytes"],
                "content_sha256": pending_result["result_sha256"],
            }
        ],
        provider=provider,
        window=window,
        checkpoint_ref=checkpoint_ref,
        source_event_ids=source_event_ids,
        source_event_store_seqs=source_event_store_seqs,
        build_identity=build_identity,
    )


def _compile_coordinator_bound(request: ContextCompileRequest):
    assert request.checkpoint_ref is not None
    assert request.checkpoint_request_id is not None
    unbound = replace(
        request,
        checkpoint_ref=None,
        checkpoint_request_id=None,
    )
    compiler = ContextCompiler()
    checkpoint_request = compiler.compile(unbound).checkpoint_requests[0]
    assert checkpoint_request.request_id == request.checkpoint_request_id
    return compiler._compile_for_coordinator(
        unbound,
        checkpoint_binding=_CheckpointBinding(
            request=checkpoint_request,
            checkpoint_ref=request.checkpoint_ref,
        ),
    ).result


def test_below_pressure_preserves_every_message_and_does_not_mutate_input() -> None:
    messages = [
        {"role": "user", "content": "constraint", "stable_id": "message-1"},
        {"role": "assistant", "content": "answer", "stable_id": "message-2"},
        {"role": "user", "content": "current", "stable_id": "message-3"},
    ]
    original = copy.deepcopy(messages)

    result = ContextCompiler().compile(_request(messages))

    assert list(result.messages) == original
    assert messages == original
    assert result.diagnostics["compacted"] is False
    assert result.checkpoint_requests == ()

    with pytest.raises(TypeError):
        result.messages[0]["content"] = "mutated"
    with pytest.raises(TypeError):
        result.diagnostics["budget"]["available_input_tokens"] = 0


def test_below_pressure_preserves_interleaved_system_message_order() -> None:
    messages = [
        {"role": "user", "content": "historical user"},
        {"role": "system", "content": "historical system record"},
        {"role": "user", "content": "current"},
    ]

    result = ContextCompiler().compile(_request(messages))

    assert list(result.messages) == messages


def test_build_identity_produces_a_replayable_context_build_envelope() -> None:
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            source_event_ids=("event-1",),
            source_event_store_seqs=(1,),
            build_identity=True,
        )
    )

    assert result.envelope is not None
    assert result.envelope.build_id == "build-1"
    assert result.envelope.source_range.start.event_id == "event-1"
    assert (
        result.envelope.estimated_input_tokens
        == result.diagnostics["after_estimated_tokens"]
    )
    assert result.envelope.status == ContextBuildStatus.COMPLETE


def test_pressure_uses_a_checkpoint_ref_and_keeps_the_current_user() -> None:
    messages = [
        {"role": "user", "content": "old-1 " + ("x" * 12_000)},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "old-2 " + ("y" * 12_000)},
        {"role": "assistant", "content": "old answer 2"},
        {"role": "user", "content": "current request"},
    ]
    result = _compile_coordinator_bound(
        _request(
            messages,
            window=8_192,
            checkpoint_ref=ResourceRef("checkpoint", "checkpoint-1", 1),
            source_event_ids=(
                "event-1",
                "event-2",
                "event-3",
                "event-4",
                "event-5",
            ),
            source_event_store_seqs=(1, 2, 3, 4, 5),
            build_identity=True,
        )
    )

    assert result.diagnostics["compacted"] is True
    assert result.diagnostics["dropped_turn_count"] >= 1
    assert result.messages[-1]["content"] == "current request"
    assert any(
        "[MEMORY_V2_CHECKPOINT]" in str(item.get("content")) for item in result.messages
    )


def test_pressure_without_a_durable_ref_returns_a_checkpoint_request_not_model_input() -> (
    None
):
    request = _request(
        [
            {"role": "user", "content": "old " + ("x" * 30_000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ],
        window=8_192,
        source_event_ids=("event-1", "event-2", "event-3"),
        source_event_store_seqs=(1, 2, 3),
    )

    result = ContextCompiler().compile(request)

    assert result.messages == ()
    assert result.diagnostics["status"] == "checkpoint_required"
    assert len(result.checkpoint_requests) == 1
    assert result.diagnostics["dropped_turn_count"] == 1


def test_checkpoint_digest_does_not_include_the_retained_current_user() -> None:
    def compile_with_current(content: str):
        return ContextCompiler().compile(
            _request(
                [
                    {"role": "user", "content": "old " + ("x" * 30_000)},
                    {"role": "assistant", "content": "old answer"},
                    {"role": "user", "content": content},
                ],
                window=8_192,
                source_event_ids=("event-1", "event-2", "event-3"),
                source_event_store_seqs=(1, 2, 3),
            )
        )

    first = compile_with_current("current A")
    second = compile_with_current("current B")

    assert first.checkpoint_requests[0].source_messages_sha256 == (
        second.checkpoint_requests[0].source_messages_sha256
    )


def test_pressure_without_exact_checkpoint_coverage_fails_closed() -> None:
    request = _request(
        [
            {"role": "user", "content": "old " + ("x" * 30_000)},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "current"},
        ],
        window=8_192,
    )

    with pytest.raises(ContextBudgetExceededError):
        ContextCompiler().compile(request)


def test_closed_tool_exchange_and_open_call_are_neutral_and_atomic() -> None:
    artifact_ref = ResourceRef("artifact", "artifact-1", 1)
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "tool_call",
                    "event_id": "call-event",
                    "store_seq": 1,
                    "call_id": "call-closed",
                    "tool_name": "lookup",
                    "arguments": {"query": "closed"},
                },
                {
                    "type": "tool_result",
                    "event_id": "result-event",
                    "store_seq": 2,
                    "call_id": "call-closed",
                    "tool_name": "lookup",
                    "result": {"preview": "bounded"},
                    "full_output_ref": artifact_ref.to_dict(),
                    "result_bytes": 500_000,
                    "result_sha256": "a" * 64,
                },
                {
                    "type": "tool_call",
                    "event_id": "open-event",
                    "store_seq": 3,
                    "call_id": "call-open",
                    "tool_name": "lookup",
                    "arguments": {"query": "open"},
                },
            ],
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    assert history["tool_exchanges"][0]["call_id"] == "call-closed"
    assert history["tool_exchanges"][0]["full_output_ref"]["id"] == "artifact-1"
    assert pinned["unfinished_tool_pairs"][0]["call_id"] == "call-open"
    assert result.diagnostics["atomic_call_ids"] == ("call-open",)
    assert not any(item.get("type") == "function_call" for item in result.messages)


def test_closed_tool_exchanges_follow_durable_completion_order() -> None:
    first_ref = ResourceRef("artifact", "artifact-first", 1)
    second_ref = ResourceRef("artifact", "artifact-second", 1)
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "tool_call",
                    "event_id": "call-b-event",
                    "store_seq": 1,
                    "call_id": "call-b",
                    "tool_name": "lookup",
                    "arguments": {"query": "b"},
                },
                {
                    "type": "tool_call",
                    "event_id": "call-a-event",
                    "store_seq": 2,
                    "call_id": "call-a",
                    "tool_name": "lookup",
                    "arguments": {"query": "a"},
                },
                {
                    "type": "tool_result",
                    "event_id": "result-b-event",
                    "store_seq": 3,
                    "call_id": "call-b",
                    "tool_name": "lookup",
                    "result": {"preview": "b"},
                    "full_output_ref": first_ref.to_dict(),
                    "result_bytes": 1,
                    "result_sha256": "b" * 64,
                },
                {
                    "type": "tool_result",
                    "event_id": "result-a-event",
                    "store_seq": 4,
                    "call_id": "call-a",
                    "tool_name": "lookup",
                    "result": {"preview": "a"},
                    "full_output_ref": second_ref.to_dict(),
                    "result_bytes": 1,
                    "result_sha256": "a" * 64,
                },
            ],
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert [item["call_id"] for item in history["tool_exchanges"]] == [
        "call-b",
        "call-a",
    ]


def test_checkpoint_covered_tool_exchange_is_not_reinjected() -> None:
    artifact_ref = ResourceRef("artifact", "covered-tool-output", 1)
    result = _compile_coordinator_bound(
        _request(
            [
                {"role": "user", "content": "old " + ("x" * 30_000)},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "current"},
            ],
            events=[
                {
                    "type": "message.user",
                    "event_id": "source-old",
                    "store_seq": 10,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "message": {"role": "user", "content": "old " + ("x" * 30_000)},
                },
                {
                    "type": "message.assistant",
                    "event_id": "source-answer",
                    "store_seq": 20,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "message": {"role": "assistant", "content": "old answer"},
                },
                {
                    "type": "message.user",
                    "event_id": "source-current",
                    "store_seq": 30,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "attempt_id": "attempt-1",
                    "run_id": "attempt-1",
                    "message": {"role": "user", "content": "current"},
                },
                {
                    "type": "tool_call",
                    "event_id": "covered-call",
                    "store_seq": 1,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "call_id": "covered-call",
                    "tool_name": "lookup",
                    "arguments": {"query": "old"},
                },
                {
                    "type": "tool_result",
                    "event_id": "covered-result",
                    "store_seq": 2,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "call_id": "covered-call",
                    "tool_name": "lookup",
                    "result": {"preview": "old result"},
                    "full_output_ref": artifact_ref.to_dict(),
                    "result_bytes": 10,
                    "result_sha256": "c" * 64,
                },
            ],
            source_event_ids=("source-old", "source-answer", "source-current"),
            source_event_store_seqs=(10, 20, 30),
            window=8_192,
            checkpoint_ref=ResourceRef("checkpoint", "checkpoint-covered", 1),
            build_identity=True,
        )
    )

    assert not any(
        "MEMORY_V2_UNTRUSTED_HISTORY" in str(message.get("content") or "")
        for message in result.messages
    )


def test_completed_tool_exchange_remains_until_checkpoint_coverage() -> None:
    old_ref = ResourceRef("artifact", "old-tool-output", 1)
    current_ref = ResourceRef("artifact", "current-tool-output", 1)
    result = ContextCompiler().compile(
        _request(
            [
                {"role": "user", "content": "old task"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "current task"},
            ],
            events=[
                {
                    "type": "message.user",
                    "event_id": "old-user",
                    "store_seq": 1,
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "message": {"role": "user", "content": "old task"},
                },
                {
                    "type": "tool_call",
                    "event_id": "old-call",
                    "store_seq": 2,
                    "call_id": "old-call",
                    "tool_name": "lookup",
                    "arguments": {"query": "old"},
                },
                {
                    "type": "tool_result",
                    "event_id": "old-result",
                    "store_seq": 3,
                    "call_id": "old-call",
                    "tool_name": "lookup",
                    "result": {"preview": "old result"},
                    "full_output_ref": old_ref.to_dict(),
                    "result_bytes": 10,
                    "result_sha256": "a" * 64,
                },
                {
                    "type": "message.assistant",
                    "event_id": "old-assistant",
                    "store_seq": 4,
                    "attempt_id": "attempt-history",
                    "run_id": "attempt-history",
                    "message": {"role": "assistant", "content": "old answer"},
                },
                {
                    "type": "message.user",
                    "event_id": "current-user",
                    "store_seq": 5,
                    "attempt_id": "attempt-1",
                    "run_id": "attempt-1",
                    "message": {"role": "user", "content": "current task"},
                },
                {
                    "type": "tool_call",
                    "event_id": "current-call",
                    "store_seq": 6,
                    "call_id": "current-call",
                    "tool_name": "lookup",
                    "arguments": {"query": "current"},
                },
                {
                    "type": "tool_result",
                    "event_id": "current-result",
                    "store_seq": 7,
                    "call_id": "current-call",
                    "tool_name": "lookup",
                    "result": {"preview": "current result"},
                    "full_output_ref": current_ref.to_dict(),
                    "result_bytes": 14,
                    "result_sha256": "b" * 64,
                },
            ],
            source_event_ids=("old-user", "old-assistant", "current-user"),
            source_event_store_seqs=(1, 4, 5),
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert [item["call_id"] for item in history["tool_exchanges"]] == [
        "old-call",
        "current-call",
    ]


@pytest.mark.parametrize(
    ("provider", "expected_call", "expected_result"),
    [
        (
            "openai",
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "approved_write",
                "arguments": '{"value":"call-1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": '{"written": "call-1"}',
            },
        ),
        (
            "anthropic",
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "approved_write",
                        "input": {"value": "call-1"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": '{"written": "call-1"}',
                    }
                ],
            },
        ),
        (
            "hyperspace",
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "approved_write",
                        "input": {"value": "call-1"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": '{"written": "call-1"}',
                    }
                ],
            },
        ),
        (
            "ollama",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "approved_write",
                            "arguments": {"value": "call-1"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"written": "call-1"}',
            },
        ),
    ],
)
def test_current_pending_tool_result_is_injected_as_a_native_provider_pair(
    provider,
    expected_call,
    expected_result,
) -> None:
    result = ContextCompiler().compile(_pending_native_tool_request(provider=provider))

    messages = result.to_dict()["messages"]
    call_index = messages.index(expected_call)
    assert messages[call_index + 1] == expected_result
    assert messages.index({"role": "user", "content": "perform the approved write"}) < (
        call_index
    )
    assert not any(
        "MEMORY_V2_UNTRUSTED_HISTORY" in str(message.get("content") or "")
        and "call-1" in str(message.get("content") or "")
        for message in messages
    )

    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    pending = pinned["pending_task_inputs"][0]
    assert pending["content_ref"] == {
        "kind": "artifact",
        "id": "artifact-call-1",
        "revision": 1,
    }
    assert pending["content_sha256"]
    assert pending["content_bytes"] > 0
    assert pending["delivered_as_native_current_tool_result"] is True
    assert "preview" not in pending
    assert "preview_truncated" not in pending


def test_current_native_tool_result_prefers_manager_projection() -> None:
    request = _pending_native_tool_request(provider="openai")
    events = [dict(event) for event in request.semantic_events or ()]
    result = next(event for event in events if event.get("type") == "tool_result")
    result["model_projection"] = {
        "result": {
            "projection": "artifact_only",
            "full_output_ref": result["full_output_ref"],
            "content_bytes": result["result_bytes"],
            "content_sha256": result["result_sha256"],
        },
        "metadata": {"projection_policy": "artifact_only"},
    }

    compiled = ContextCompiler().compile(
        replace(request, semantic_events=tuple(events))
    )

    native_result = next(
        message
        for message in compiled.to_dict()["messages"]
        if message.get("type") == "function_call_output"
    )
    assert '"projection": "artifact_only"' in native_result["output"]
    assert '"written"' not in native_result["output"]
    assert '"metadata"' not in native_result["output"]


def test_cross_provider_pending_tool_result_remains_neutral() -> None:
    result = ContextCompiler().compile(
        _pending_native_tool_request(
            provider="openai",
            source_provider="anthropic",
        )
    )

    assert not any(
        message.get("type") in {"function_call", "function_call_output"}
        for message in result.messages
    )
    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert [item["call_id"] for item in history["tool_exchanges"]] == ["call-1"]
    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    pending = pinned["pending_task_inputs"][0]
    assert pending["preview"] == "result preview that must not be duplicated"
    assert "delivered_as_native_current_tool_result" not in pending


def test_incomplete_parallel_iteration_never_partially_injects_native_history() -> None:
    result = ContextCompiler().compile(
        _pending_native_tool_request(
            provider="anthropic",
            call_ids=("call-1", "call-2"),
            completed_call_ids=("call-1",),
        )
    )

    assert not any(
        isinstance(message.get("content"), list)
        and any(
            isinstance(block, dict) and block.get("type") == "tool_use"
            for block in message["content"]
        )
        for message in result.messages
    )
    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert [item["call_id"] for item in history["tool_exchanges"]] == ["call-1"]
    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    assert [item["call_id"] for item in pinned["unfinished_tool_pairs"]] == ["call-2"]


def test_pressure_keeps_a_parallel_native_batch_complete_ordered_and_at_the_tail() -> (
    None
):
    request = _pending_native_tool_request(
        provider="anthropic",
        call_ids=("call-1", "call-2"),
        messages=[
            {"role": "user", "content": "old " + ("x" * 30_000)},
            {"role": "user", "content": "perform both approved writes"},
        ],
        window=8_192,
        checkpoint_ref=ResourceRef("checkpoint", "checkpoint-native-batch", 1),
        source_event_ids=("event-old-user", "event-current"),
        source_event_store_seqs=(1, 2),
        build_identity=True,
        source_events=[
            {
                "type": "message.user",
                "event_id": "event-old-user",
                "store_seq": 1,
                "attempt_id": "attempt-history",
                "run_id": "attempt-history",
                "message": {
                    "role": "user",
                    "content": "old " + ("x" * 30_000),
                },
            },
            {
                "type": "message.user",
                "event_id": "event-current",
                "store_seq": 2,
                "attempt_id": "attempt-history",
                "run_id": "attempt-history",
                "message": {
                    "role": "user",
                    "content": "perform both approved writes",
                },
            },
        ],
    )
    result = _compile_coordinator_bound(request)

    messages = result.to_dict()["messages"]
    assistant_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_use" for block in message["content"])
    )
    result_index = next(
        index
        for index, message in enumerate(messages)
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_result" for block in message["content"])
    )
    assert result_index == assistant_index + 1
    assert (
        messages.index({"role": "user", "content": "perform both approved writes"})
        < assistant_index
    )
    assert [
        block["id"]
        for block in messages[assistant_index]["content"]
        if block.get("type") == "tool_use"
    ] == ["call-1", "call-2"]
    assert [
        block["tool_use_id"]
        for block in messages[result_index]["content"]
        if block.get("type") == "tool_result"
    ] == ["call-1", "call-2"]
    assert set(result.diagnostics["atomic_call_ids"]) >= {"call-1", "call-2"}


def test_completed_tool_result_without_a_durable_ref_fails_closed() -> None:
    request = _request(
        [{"role": "user", "content": "current"}],
        events=[
            {
                "type": "tool_call",
                "event_id": "call-event",
                "store_seq": 1,
                "call_id": "call-1",
                "tool_name": "lookup",
                "arguments": {},
            },
            {
                "type": "tool_result",
                "event_id": "result-event",
                "store_seq": 2,
                "call_id": "call-1",
                "tool_name": "lookup",
                "result": {"value": "not durable"},
            },
        ],
    )

    with pytest.raises(ContextCompilerError, match="durable"):
        ContextCompiler().compile(request)


def test_pinned_task_state_and_uncovered_inputs_are_always_injected_as_untrusted() -> (
    None
):
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            task_state={
                "stable_id": "task-state-1",
                "objective": "Ship",
                "revision": 2,
                "covered_through_store_seq": 4,
            },
            pending_inputs=[
                {
                    "event_id": "event-5",
                    "store_seq": 5,
                    "type": "message.user",
                    "preview": "earlier decision",
                    "content_ref": ResourceRef("event", "event-5", 1).to_dict(),
                    "content_bytes": 16,
                    "content_sha256": "b" * 64,
                    "inline": False,
                },
                {
                    "event_id": "event-6",
                    "store_seq": 6,
                    "type": "message.user",
                    "preview": "current",
                    "content_ref": ResourceRef("event", "event-6", 1).to_dict(),
                    "content_bytes": 7,
                    "content_sha256": "c" * 64,
                    "inline": False,
                },
            ],
        )
    )

    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    assert pinned["trust"] == "UNTRUSTED_DATA"
    assert pinned["pinned_task_state"] == {
        "stable_id": "task-state-1",
        "objective": "Ship",
        "revision": 2,
    }
    assert pinned["pending_task_inputs"][0]["preview"] == "earlier decision"
    assert "preview" not in pinned["pending_task_inputs"][1]
    assert pinned["pending_task_inputs"][1]["delivered_as_native_current_user"] is True


def test_multiple_root_pending_interactions_fail_closed() -> None:
    request = _request(
        [{"role": "user", "content": "current"}],
        events=[
            {
                "type": "interaction_requested",
                "event_id": "interaction-1",
                "store_seq": 1,
                "interaction_request": {"interaction_id": "interaction-1"},
            },
            {
                "type": "interaction_requested",
                "event_id": "interaction-2",
                "store_seq": 2,
                "interaction_request": {"interaction_id": "interaction-2"},
            },
        ],
    )

    with pytest.raises(ContextCompilerError, match="multiple_pending_interactions"):
        ContextCompiler().compile(request)


def test_canonical_interaction_resolution_closes_only_its_pending_lifecycle() -> None:
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "interaction.requested",
                    "event_id": "requested-closed",
                    "store_seq": 1,
                    "interaction_id": "interaction-closed",
                    "interaction_request": {
                        "interaction_id": "interaction-closed",
                    },
                },
                {
                    "type": "interaction.resolved",
                    "event_id": "resolved-closed",
                    "store_seq": 2,
                    "interaction_id": "interaction-closed",
                },
                {
                    "type": "interaction.requested",
                    "event_id": "requested-open",
                    "store_seq": 3,
                    "interaction_id": "interaction-open",
                    "interaction_request": {
                        "interaction_id": "interaction-open",
                        "question": "Still pending?",
                    },
                },
            ],
        )
    )

    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    assert (
        pinned["pending_interaction"]["request"]["interaction_id"] == "interaction-open"
    )
    assert "interaction-closed" not in str(pinned)


def test_resolved_human_interaction_closes_only_its_correlated_ask_call() -> None:
    response_preview = json.dumps(
        {"selected_values": ["a"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    response_bytes = response_preview.encode("utf-8")
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "continue after my answer"}],
            events=[
                {
                    "type": "tool_call",
                    "event_id": "call-human-event",
                    "store_seq": 1,
                    "call_id": "call-human",
                    "tool_name": "ask_user_question",
                    "arguments": {"question": "Choose one"},
                },
                {
                    "type": "tool_call",
                    "event_id": "call-open-event",
                    "store_seq": 2,
                    "call_id": "call-still-open",
                    "tool_name": "lookup",
                    "arguments": {"query": "pending"},
                },
                {
                    "type": "interaction.requested",
                    "event_id": "interaction-requested-event",
                    "store_seq": 3,
                    "interaction_id": "interaction-human",
                    "interaction_request": {
                        "interaction_id": "interaction-human",
                        "kind": "human_input",
                        "payload": {"request_id": "call-human"},
                    },
                },
                {
                    "type": "interaction.resolved",
                    "event_id": "interaction-resolved-event",
                    "store_seq": 4,
                    "interaction_id": "interaction-human",
                    "content_ref": ResourceRef(
                        "artifact",
                        "interaction-answer",
                        1,
                    ).to_dict(),
                    "content_bytes": len(response_bytes),
                    "content_sha256": hashlib.sha256(response_bytes).hexdigest(),
                    "preview": response_preview,
                    "preview_truncated": False,
                },
            ],
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    [resolved] = history["resolved_human_interactions"]
    assert resolved["interaction_id"] == "interaction-human"
    assert resolved["call_id"] == "call-human"
    assert resolved["response"]["preview"] == response_preview
    assert resolved["response"]["content_ref"] == {
        "kind": "artifact",
        "id": "interaction-answer",
        "revision": 1,
    }
    pinned = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT")
    assert [item["call_id"] for item in pinned["unfinished_tool_pairs"]] == [
        "call-still-open"
    ]
    assert result.diagnostics["atomic_call_ids"] == ("call-still-open",)
    assert not any(
        message.get("type") in {"function_call_output", "tool_result"}
        for message in result.messages
    )


def test_authorized_canonical_resolution_supersedes_one_malformed_legacy_event():
    response_preview = json.dumps(
        {"selected_values": ["a"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    response_bytes = response_preview.encode("utf-8")
    response_ref = ResourceRef("artifact", "interaction-answer-compat", 1)
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "continue after my answer"}],
            events=[
                {
                    "type": "tool_call",
                    "event_id": "call-human-compat-event",
                    "store_seq": 1,
                    "call_id": "call-human-compat",
                    "tool_name": "ask_user_question",
                    "arguments": {"question": "Choose one"},
                },
                {
                    "type": "interaction.requested",
                    "event_id": "interaction-requested-compat-event",
                    "store_seq": 2,
                    "interaction_id": "interaction-human-compat",
                    "interaction_request": {
                        "interaction_id": "interaction-human-compat",
                        "kind": "human_input",
                        "payload": {"request_id": "call-human-compat"},
                    },
                },
                {
                    "type": "interaction_resolved",
                    "event_id": "interaction-resolved-legacy-event",
                    "store_seq": 3,
                    "interaction_id": "interaction-human-compat",
                },
                {
                    "type": "interaction.resolved",
                    "event_id": "interaction-resolved-canonical-event",
                    "store_seq": 4,
                    "interaction_id": "interaction-human-compat",
                    "submitted_by": "ui:test",
                    "content_ref": response_ref.to_dict(),
                    "content_bytes": len(response_bytes),
                    "content_sha256": hashlib.sha256(response_bytes).hexdigest(),
                    "preview": response_preview,
                    "preview_truncated": False,
                    "resource_refs": [response_ref.to_dict()],
                },
            ],
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    [resolved] = history["resolved_human_interactions"]
    assert resolved["interaction_id"] == "interaction-human-compat"
    assert resolved["response"]["content_ref"] == {
        "kind": "artifact",
        "id": "interaction-answer-compat",
        "revision": 1,
    }
    assert result.diagnostics["atomic_call_ids"] == ()


def test_complete_looking_but_unauthorized_resolution_cannot_supersede_legacy():
    response_preview = '{"selected_values":["a"]}'
    response_bytes = response_preview.encode("utf-8")
    with pytest.raises(
        ContextCompilerError,
        match="compatibility is ambiguous",
    ):
        ContextCompiler().compile(
            _request(
                [{"role": "user", "content": "continue"}],
                events=[
                    {
                        "type": "interaction.requested",
                        "event_id": "interaction-requested-untrusted",
                        "store_seq": 1,
                        "interaction_id": "interaction-untrusted",
                        "interaction_request": {
                            "interaction_id": "interaction-untrusted",
                        },
                    },
                    {
                        "type": "interaction_resolved",
                        "event_id": "interaction-resolved-untrusted-legacy",
                        "store_seq": 2,
                        "interaction_id": "interaction-untrusted",
                    },
                    {
                        "type": "interaction.resolved",
                        "event_id": "interaction-resolved-untrusted-dotted",
                        "store_seq": 3,
                        "interaction_id": "interaction-untrusted",
                        "submitted_by": "ui:test",
                        "content_ref": ResourceRef(
                            "artifact", "payload-only-ref", 1
                        ).to_dict(),
                        "content_bytes": len(response_bytes),
                        "content_sha256": hashlib.sha256(
                            response_bytes
                        ).hexdigest(),
                        "preview": response_preview,
                        "preview_truncated": False,
                        "resource_refs": [
                            ResourceRef(
                                "artifact", "foreign-ref", 1
                            ).to_dict()
                        ],
                    },
                ],
            )
        )


def test_many_resolved_human_interactions_do_not_accumulate_under_pressure() -> None:
    messages = [
        {"role": "user", "content": "old " + ("x" * 30_000)},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current request"},
    ]
    events = [
        {
            "type": f"message.{message['role']}",
            "event_id": event_id,
            "store_seq": store_seq,
            "execution_id": "execution-1",
            "generation_id": "generation-1",
            "attempt_id": "attempt-history",
            "run_id": "attempt-history",
            "message": message,
        }
        for message, event_id, store_seq in zip(
            messages,
            ("message-old", "message-answer", "message-current"),
            (1, 2, 3),
            strict=True,
        )
    ]
    store_seq = 100
    response_preview = '{"selected_values":["accepted"]}'
    response_bytes = response_preview.encode("utf-8")
    response_sha256 = hashlib.sha256(response_bytes).hexdigest()
    for index in range(20):
        call_id = f"call-resolved-{index:03d}"
        interaction_id = f"interaction-resolved-{index:03d}"
        events.extend(
            [
                {
                    "type": "tool_call",
                    "event_id": f"tool-call-event-{index:03d}",
                    "store_seq": store_seq,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "call_id": call_id,
                    "tool_name": "ask_user_question",
                    "arguments": {
                        "question": f"Question {index} " + ("q" * 1_200)
                    },
                },
                {
                    "type": "interaction.requested",
                    "event_id": f"interaction-request-event-{index:03d}",
                    "store_seq": store_seq + 1,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "interaction_id": interaction_id,
                    "interaction_request": {
                        "interaction_id": interaction_id,
                        "kind": "human_input",
                        "payload": {"request_id": call_id},
                    },
                },
                {
                    "type": "interaction.resolved",
                    "event_id": f"interaction-resolution-event-{index:03d}",
                    "store_seq": store_seq + 2,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "interaction_id": interaction_id,
                    "content_ref": ResourceRef(
                        "artifact",
                        f"interaction-answer-{index:03d}",
                        1,
                    ).to_dict(),
                    "content_bytes": len(response_bytes),
                    "content_sha256": response_sha256,
                    "preview": response_preview,
                    "preview_truncated": False,
                },
            ]
        )
        store_seq += 3

    result = _compile_coordinator_bound(
        _request(
            messages,
            events=events,
            window=8_192,
            checkpoint_ref=ResourceRef(
                "checkpoint",
                "checkpoint-resolved-interactions",
                1,
            ),
            source_event_ids=("message-old", "message-answer", "message-current"),
            source_event_store_seqs=(1, 2, 3),
            build_identity=True,
        )
    )

    assert result.diagnostics["compacted"] is True
    assert result.diagnostics["atomic_call_ids"] == ()
    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert len(history["resolved_human_interactions"]) == 20
    assert not any(
        "unfinished_tool_pairs" in str(message.get("content") or "")
        for message in result.messages
    )
    assert not any(
        "MEMORY_V2_UNTRUSTED_PINNED_CONTEXT" in str(message.get("content") or "")
        for message in result.messages
    )


def test_child_interaction_is_not_injected_into_the_parent_context() -> None:
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "interaction_requested",
                    "event_id": "child-interaction",
                    "store_seq": 1,
                    "parent_run_id": "parent-run",
                    "interaction_request": {"interaction_id": "child-interaction"},
                }
            ],
        )
    )

    assert result.messages == ({"role": "user", "content": "current"},)


def test_terminal_root_attempt_clears_a_stale_pending_interaction() -> None:
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "interaction_requested",
                    "event_id": "interaction-1",
                    "store_seq": 1,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "attempt_id": "attempt-1",
                    "run_id": "attempt-1",
                    "interaction_request": {"interaction_id": "interaction-1"},
                },
                {
                    "type": "run_failed",
                    "event_id": "terminal-1",
                    "store_seq": 2,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "attempt_id": "attempt-1",
                    "run_id": "attempt-1",
                },
            ],
            build_identity=True,
        )
    )

    assert result.messages == ({"role": "user", "content": "current"},)


def test_native_orphan_tool_result_fails_closed() -> None:
    request = _request(
        [
            {"role": "tool", "tool_call_id": "call-orphan", "content": "result"},
            {"role": "user", "content": "current"},
        ]
    )

    with pytest.raises(ContextCompilerError, match="orphan_tool_result"):
        ContextCompiler().compile(request)


def test_large_open_tool_arguments_cannot_bypass_the_budget() -> None:
    request = _request(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "lookup",
                            "arguments": "x" * 100_000,
                        },
                    }
                ],
            },
            {"role": "user", "content": "current"},
        ],
        window=8_192,
    )

    with pytest.raises(ContextBudgetExceededError):
        ContextCompiler().compile(request)


def test_pressure_never_splits_a_closed_native_tool_pair_across_turns() -> None:
    result = _compile_coordinator_bound(
        _request(
            [
                {"role": "user", "content": "old " + ("x" * 30_000)},
                {
                    "type": "function_call",
                    "call_id": "call-closed",
                    "name": "lookup",
                    "arguments": "{}",
                },
                {
                    "role": "user",
                    "type": "tool_result",
                    "call_id": "call-closed",
                    "output": "bounded result",
                },
                {"role": "user", "content": "current"},
            ],
            window=8_192,
            checkpoint_ref=ResourceRef("checkpoint", "checkpoint-1", 1),
            source_event_ids=("event-1", "event-2", "event-3", "event-4"),
            source_event_store_seqs=(1, 2, 3, 4),
            build_identity=True,
        )
    )

    retained_calls = {
        item.get("call_id")
        for item in result.messages
        if item.get("type") == "function_call"
    }
    retained_results = {
        item.get("call_id")
        for item in result.messages
        if item.get("type") == "tool_result"
    }
    assert retained_calls == retained_results


def test_checkpoint_required_envelope_does_not_claim_any_included_range() -> None:
    result = ContextCompiler().compile(
        _request(
            [
                {"role": "user", "content": "old " + ("x" * 30_000)},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "current"},
            ],
            window=8_192,
            source_event_ids=("event-1", "event-2", "event-3"),
            source_event_store_seqs=(1, 2, 3),
            build_identity=True,
        )
    )

    assert result.envelope is not None
    assert result.envelope.status == ContextBuildStatus.UNAVAILABLE
    assert result.envelope.included_ranges == ()
    assert result.envelope.transformed_ranges == ()


def test_artifact_and_completed_handoff_are_injected_as_untrusted_refs() -> None:
    artifact_ref = ResourceRef("artifact", "artifact-1", 1).to_dict()
    handoff_ref = ResourceRef("artifact", "handoff-1", 1).to_dict()
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "artifact_created",
                    "event_id": "artifact-event",
                    "store_seq": 1,
                    "artifact_ref": artifact_ref,
                    "artifact": {"name": "report"},
                },
                {
                    "type": "subagent_completed",
                    "event_id": "handoff-event",
                    "store_seq": 2,
                    "child_run_id": "child-1",
                    "status": "complete",
                    "handoff_envelope": {
                        "child_run_id": "child-1",
                        "status": "complete",
                        "summary": "finished",
                        "full_output_ref": handoff_ref,
                        "artifact_refs": [artifact_ref],
                    },
                },
            ],
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert history["artifact_refs"][0]["artifact_ref"]["id"] == "artifact-1"
    assert history["handoffs"][0]["full_output_ref"]["id"] == "handoff-1"
    assert history["handoff_refs"] == [
        {"kind": "artifact", "id": "handoff-1", "revision": 1}
    ]


def test_completed_handoff_without_full_output_ref_fails_closed() -> None:
    request = _request(
        [{"role": "user", "content": "current"}],
        events=[
            {
                "type": "subagent_completed",
                "event_id": "handoff-event",
                "store_seq": 1,
                "child_run_id": "child-1",
                "status": "complete",
                "handoff_envelope": {
                    "child_run_id": "child-1",
                    "status": "complete",
                    "summary": "not durable",
                },
            }
        ],
    )

    with pytest.raises(ContextCompilerError, match="handoff"):
        ContextCompiler().compile(request)


def test_current_generation_rejects_stale_branch_messages() -> None:
    request = ContextCompileRequest(
        case="generation-isolation",
        current_generation="generation-2",
        source_messages=(
            {
                "role": "user",
                "content": "stale",
                "generation": "generation-1",
            },
            {
                "role": "user",
                "content": "current",
                "generation": "generation-2",
            },
        ),
        budget=resolve_context_budget(context_window_tokens=20_000),
    )

    with pytest.raises(ContextCompilerError, match="generation"):
        ContextCompiler().compile(request)


def test_checkpoint_uses_explicit_event_mapping_without_fake_system_or_current_cursors() -> (
    None
):
    result = ContextCompiler().compile(
        _request(
            [
                {"role": "system", "content": "current instructions"},
                {"role": "user", "content": "old " + ("x" * 30_000)},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "native current input"},
            ],
            window=8_192,
            source_message_cursors=(
                SourceMessageCursor(1, "event-1", 10),
                SourceMessageCursor(2, "event-2", 11),
            ),
        )
    )

    assert result.messages == ()
    assert result.checkpoint_requests[0].source_event_ids == (
        "event-1",
        "event-2",
    )
    assert result.checkpoint_requests[0].source_event_store_seqs == (10, 11)


def test_envelope_ranges_only_claim_event_backed_messages() -> None:
    result = _compile_coordinator_bound(
        _request(
            [
                {"role": "system", "content": "current instructions"},
                {"role": "user", "content": "old " + ("x" * 30_000)},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "native current input"},
            ],
            window=8_192,
            checkpoint_ref=ResourceRef("checkpoint", "checkpoint-1", 1),
            source_message_cursors=(
                SourceMessageCursor(1, "event-1", 10),
                SourceMessageCursor(2, "event-2", 11),
            ),
            build_identity=True,
        )
    )

    assert result.envelope is not None
    assert result.envelope.included_ranges == ()
    assert len(result.envelope.transformed_ranges) == 1
    assert result.envelope.transformed_ranges[0].start.event_id == "event-1"
    assert result.envelope.transformed_ranges[0].end.event_id == "event-2"
    assert result.envelope.checkpoint_refs == (
        ResourceRef("checkpoint", "checkpoint-1", 1),
    )


def test_checkpoint_ref_is_never_reused_for_expanded_coverage() -> None:
    messages = (
        {"role": "user", "content": "old-1 " + ("x" * 9_000)},
        {"role": "assistant", "content": "old answer 1"},
        {"role": "user", "content": "old-2 " + ("y" * 4_000)},
        {"role": "assistant", "content": "old answer 2"},
        {"role": "user", "content": "z" * 500},
    )
    cursors = tuple(
        SourceMessageCursor(index, f"event-{index}", 100 + (index * 3))
        for index in range(len(messages))
    )
    first_request = ContextCompileRequest(
        case="checkpoint-binding",
        source_messages=messages,
        source_message_cursors=cursors,
        fixed_overhead_tokens=3_680,
        budget=resolve_context_budget(context_window_tokens=8_192),
    )
    first = ContextCompiler().compile(first_request)
    original_checkpoint = first.checkpoint_requests[0]

    with pytest.raises(ContextCompilerError, match="checkpoint_consumption_invalid"):
        ContextCompiler()._compile_for_coordinator(
            first_request,
            checkpoint_binding=_CheckpointBinding(
                request=original_checkpoint,
                checkpoint_ref=ResourceRef("checkpoint", "r" * 256, 1),
            ),
        )


def test_schema_v4_recorded_artifact_and_handoff_envelopes_are_injected() -> None:
    artifact_ref = ResourceRef("artifact", "artifact-recorded", 1).to_dict()
    handoff_ref = ResourceRef("artifact", "handoff-recorded", 1).to_dict()
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "event_id": "artifact-wrapper",
                    "store_seq": 20,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "type": "artifact.recorded",
                    "event": {
                        "schema_version": "context.v2",
                        "type": "artifact.recorded",
                        "payload": {
                            "artifact_ref": {
                                "ref": artifact_ref,
                                "media_type": "text/markdown",
                                "bytes": 12,
                                "sha256": "a" * 64,
                                "preview": "report",
                            }
                        },
                    },
                },
                {
                    "event_id": "handoff-wrapper",
                    "store_seq": 21,
                    "execution_id": "execution-1",
                    "generation_id": "generation-1",
                    "type": "handoff.recorded",
                    "event": {
                        "schema_version": "context.v2",
                        "type": "handoff.recorded",
                        "payload": {
                            "artifact_ref": {
                                "ref": handoff_ref,
                                "media_type": "application/json",
                                "bytes": 24,
                                "sha256": "b" * 64,
                                "preview": "child finished",
                            },
                            "content_bytes": 24,
                            "content_sha256": "b" * 64,
                        },
                    },
                },
            ],
            build_identity=True,
        )
    )

    history = _marker_payload(result.messages, "MEMORY_V2_UNTRUSTED_HISTORY")
    assert history["artifact_refs"][0]["artifact_ref"]["id"] == "artifact-recorded"
    assert history["handoffs"][0]["full_output_ref"]["id"] == "handoff-recorded"
    assert history["handoff_refs"] == [
        {"kind": "artifact", "id": "handoff-recorded", "revision": 1}
    ]
    assert result.envelope is not None
    assert len(result.envelope.transformed_ranges) == 1
    assert result.envelope.transformed_ranges[0].start.event_id == "artifact-wrapper"
    assert result.envelope.transformed_ranges[0].end.event_id == "handoff-wrapper"


def test_reserved_internal_message_metadata_collision_fails_closed() -> None:
    request = _request(
        [
            {
                "role": "user",
                "content": "current",
                "__unchain_context_source_index__": "user-data",
            }
        ]
    )

    with pytest.raises(ContextCompilerError, match="metadata collision"):
        ContextCompiler().compile(request)


def test_nested_and_outer_event_generation_conflicts_fail_closed() -> None:
    request = ContextCompileRequest(
        case="nested-generation-isolation",
        current_generation="generation-2",
        source_messages=({"role": "user", "content": "current"},),
        semantic_events=(
            {
                "event_id": "event-1",
                "store_seq": 1,
                "generation_id": "generation-1",
                "type": "artifact.recorded",
                "event": {
                    "type": "artifact.recorded",
                    "generation_id": "generation-2",
                    "payload": {},
                },
            },
        ),
        budget=resolve_context_budget(context_window_tokens=20_000),
    )

    with pytest.raises(JournalMessageProjectionError) as raised:
        ContextCompiler().compile(request)

    assert raised.value.reason == "event_scope_conflict"


def test_envelope_does_not_claim_ignored_semantic_event_cursor() -> None:
    result = ContextCompiler().compile(
        _request(
            [{"role": "user", "content": "current"}],
            events=[
                {
                    "type": "diagnostic.ignored",
                    "event_id": "ignored-event",
                    "store_seq": 7,
                    "artifact_ref": ResourceRef(
                        "artifact", "ignored-artifact", 1
                    ).to_dict(),
                }
            ],
            build_identity=True,
        )
    )

    assert result.envelope is not None
    assert result.envelope.source_range is None
    assert result.envelope.transformed_ranges == ()
    assert result.envelope.artifact_refs == ()
