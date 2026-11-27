"""Within-turn history pressure must reduce evidence, never its durable source."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from collections.abc import Mapping

import pytest

from unchain.context import (
    ContextBudgetExceededError,
    ContextCompileRequest,
    ContextCompiler,
    resolve_context_budget,
)
from unchain.context_content import decode_context_content_locator, render_context_content_page
from unchain.journal import ResourceRef
from unchain.memory.toolkit.models import MemoryToolContentPage


HISTORY_LIMIT = 256 * 1024
SYSTEM = {"role": "system", "content": "Keep the original task constraints."}
USER = {"role": "user", "content": "Finish this one long-running task."}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _events(count, *, argument="x" * 4096, result_text="", managed=False):
    events = []
    for index in range(count):
        call_id = f"history-{index:04}"
        result = {"step": index, "status": "complete", **({"content": result_text} if result_text else {})}
        raw = _json_bytes(result)
        ref = ResourceRef("artifact", f"output-{index:04}", 1).to_dict()
        output = {
            "type": "tool_result", "event_id": f"result-{index:04}", "store_seq": index * 2 + 2,
            "call_id": call_id, "tool_name": "shell", "result": result,
            "full_output_ref": ref, "result_bytes": len(raw), "result_sha256": hashlib.sha256(raw).hexdigest(),
        }
        if managed:
            output["model_projection"] = {
                "result": {"content_bytes": len(raw), "content_sha256": hashlib.sha256(raw).hexdigest(),
                           "full_output_ref": ref, "inline": True, "preview": raw.decode(), "projection": "default"},
                "metadata": {"projection_policy": "default"},
            }
        events.extend([
            {"type": "tool_call", "event_id": f"call-{index:04}", "store_seq": index * 2 + 1,
             "call_id": call_id, "tool_name": "shell", "arguments": {"command": argument}},
            output,
        ])
    return events


def _request(events, *, window=128_000, **kwargs):
    return ContextCompileRequest(
        case="current-turn-projection", source_messages=(SYSTEM, USER), semantic_events=tuple(events),
        budget=resolve_context_budget(context_window_tokens=window), provider="openai", model="synthetic",
        **kwargs,
    )


def _payload(messages, marker):
    content = next(str(message.get("content") or "") for message in messages
                   if marker in str(message.get("content") or ""))
    return json.loads(content.rsplit("\n", 1)[-1])


def _history(result):
    return _payload(result.messages, "MEMORY_V2_EVIDENCE_HISTORY")


def _read_refs(value):
    refs = []
    if isinstance(value, Mapping):
        if value.get("tool") == "context_content_read":
            arguments = value.get("arguments", {})
            assert arguments.get("offset") == 0
            assert 0 < arguments.get("limit", 0) <= 8192
            refs.append(decode_context_content_locator(
                arguments["ref"], error_message="Projected evidence must have a valid read locator",
            ))
        for nested in value.values():
            refs.extend(_read_refs(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            refs.extend(_read_refs(nested))
    return refs


def _assert_reduction(result, expected_count):
    diagnostic = result.diagnostics
    assert diagnostic["semantic_history_before_bytes"] > diagnostic["semantic_history_after_bytes"]
    assert diagnostic["semantic_history_after_bytes"] <= HISTORY_LIMIT
    assert (diagnostic["semantic_history_before_estimated_tokens"]
            > diagnostic["semantic_history_after_estimated_tokens"] > 0)
    assert 0 < diagnostic["semantic_history_compacted_exchange_count"] <= expected_count
    assert diagnostic["compacted"] is True
    assert result.checkpoint_requests == ()


@pytest.mark.parametrize("argument", ["x" * 4096, '文"\n' * 1366], ids=["ascii", "unicode-json-escaping"])
def test_single_current_turn_survives_local_history_byte_pressure(argument):
    events = _events(64, argument=argument)
    original = copy.deepcopy(events)
    request = _request(events, window=262_144)

    result = ContextCompiler().compile(request)

    exchanges = _history(result)["tool_exchanges"]
    assert [exchange["call_id"] for exchange in exchanges] == [f"history-{index:04}" for index in range(64)]
    assert [exchange["tool_name"] for exchange in exchanges] == ["shell"] * 64
    _assert_reduction(result, 64)
    for index, exchange in enumerate(exchanges):
        projected = exchange["arguments"]
        original_value = _json_bytes(events[index * 2]["arguments"])
        assert ResourceRef("context_event", f"call-{index:04}", 1) in _read_refs(projected)
        assert projected["read_scope"] == "canonical_tool_call_event"
        assert projected["projected_value_bytes"] == len(original_value)
        assert projected["projected_value_sha256"] == hashlib.sha256(original_value).hexdigest()
        assert "content_bytes" not in projected
        assert "content_sha256" not in projected
    assert any(dict(message) == USER for message in result.messages)
    assert any(dict(message) == SYSTEM for message in result.messages)
    assert events == original
    assert ContextCompiler().compile(request).to_dict() == result.to_dict()


def test_large_managed_historical_results_keep_exact_artifact_read_handles():
    events = _events(40, argument="inspect", result_text="result evidence " * 512, managed=True)
    unpressured = ContextCompiler().compile(_request(events[:2], window=262_144))
    original_model_value = _json_bytes(_history(unpressured)["tool_exchanges"][0]["result"])
    result = ContextCompiler().compile(_request(events, window=262_144))

    exchanges = _history(result)["tool_exchanges"]
    assert [exchange["call_id"] for exchange in exchanges] == [f"history-{index:04}" for index in range(40)]
    _assert_reduction(result, 40)
    for index, exchange in enumerate(exchanges):
        assert ResourceRef("artifact", f"output-{index:04}", 1) in _read_refs(exchange["result"])
        assert exchange["result"]["read_scope"] == "full_tool_output"
    first_projected = exchanges[0]["result"]
    assert first_projected["projected_value_bytes"] == len(original_model_value)
    assert first_projected["projected_value_sha256"] == hashlib.sha256(original_model_value).hexdigest()
    assert "content_bytes" not in first_projected
    assert "content_sha256" not in first_projected


def test_token_only_pressure_also_reduces_verbose_current_turn_history():
    result = ContextCompiler().compile(_request(_events(8), window=8192))

    _assert_reduction(result, 8)
    assert result.diagnostics["semantic_history_before_bytes"] < HISTORY_LIMIT
    assert result.diagnostics["after_estimated_tokens"] <= result.diagnostics["budget"]["pressure_threshold_tokens"]
    assert len(_history(result)["tool_exchanges"]) == 8


def test_unpressured_history_preserves_argument_and_result_values():
    events = _events(2, argument="small command")
    result = ContextCompiler().compile(_request(events))

    exchanges = _history(result)["tool_exchanges"]
    assert [item["arguments"] for item in exchanges] == [events[0]["arguments"], events[2]["arguments"]]
    assert [item["result"] for item in exchanges] == [events[1]["result"], events[3]["result"]]
    assert result.diagnostics.get("semantic_history_compacted_exchange_count", 0) == 0
    assert result.diagnostics["compacted"] is False


def test_pressure_preserves_unfinished_pair_pending_approval_and_task_progress():
    protected = [
        {"type": "tool_call", "event_id": "unfinished-call", "store_seq": 1000,
         "call_id": "unfinished", "tool_name": "write", "arguments": {"content": "must remain unchanged " * 100}},
        {"type": "interaction_requested", "event_id": "approval-event", "store_seq": 1001,
         "interaction_request": {"interaction_id": "approval-open", "question": "Approve the unfinished write?"}},
    ]
    task_state = {"stable_id": "task-current", "goal": "Finish the current task",
                  "progress": [{"step": "investigate", "status": "done"}], "next_step": "verify"}
    baseline = ContextCompiler().compile(_request(protected, task_state=task_state))
    result = ContextCompiler().compile(_request(_events(64) + protected, task_state=task_state))

    _assert_reduction(result, 64)
    assert (_payload(result.messages, "MEMORY_V2_EVIDENCE_PINNED_CONTEXT")
            == _payload(baseline.messages, "MEMORY_V2_EVIDENCE_PINNED_CONTEXT"))
    assert _payload(result.messages, "MEMORY_V2_EVIDENCE_PINNED_CONTEXT")["pinned_task_state"] == task_state


@pytest.mark.parametrize("provider", ["openai", "anthropic", "hyperspace", "gemini", "ollama"])
def test_pressure_preserves_the_verified_current_native_batch(provider):
    from test_compiler import _pending_native_tool_request

    baseline_request = _pending_native_tool_request(provider=provider, window=128_000)
    current_events = [{**dict(event), "store_seq": event["store_seq"] + 1000}
                      for event in baseline_request.semantic_events]
    pending = tuple({**dict(item), "store_seq": item["store_seq"] + 1000}
                    for item in baseline_request.pending_task_inputs)
    baseline = ContextCompiler().compile(replace(
        baseline_request, semantic_events=tuple(current_events), pending_task_inputs=pending,
    ))
    request = replace(baseline_request, semantic_events=tuple(_events(64) + current_events), pending_task_inputs=pending)

    result = ContextCompiler().compile(request)

    _assert_reduction(result, 64)
    assert result.to_dict()["messages"][-len(baseline.messages):] == baseline.to_dict()["messages"]


def test_pressure_preserves_current_cross_provider_context_page_bytes():
    from test_compiler import _pending_native_tool_request

    current = _pending_native_tool_request(provider="openai", source_provider="anthropic", window=128_000)
    data = "CURRENT PAGE 文\n".encode() * 100
    page = render_context_content_page(MemoryToolContentPage(
        ref=ResourceRef("artifact", "current-reader-source", 1), media_type="text/plain", data=data,
        offset=0, total_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
    ))
    current_events = []
    for event in current.semantic_events:
        changed = {**dict(event), "store_seq": event["store_seq"] + 1000, "tool_name": "context_content_read"}
        if changed["type"] == "tool_call":
            changed["arguments"] = {"ref": page["ref"], "offset": 0, "limit": 8192}
        else:
            raw = _json_bytes(page)
            changed.update(result=page, result_bytes=len(raw), result_sha256=hashlib.sha256(raw).hexdigest(),
                           model_projection={"result": page, "metadata": {"projection_policy": "context_page"}})
        current_events.append(changed)
    result_event = current_events[-1]
    pending = ({**dict(current.pending_task_inputs[0]), "store_seq": result_event["store_seq"],
                "content_bytes": result_event["result_bytes"], "content_sha256": result_event["result_sha256"]},)
    request = replace(current, semantic_events=tuple(_events(64) + current_events), pending_task_inputs=pending)

    result = ContextCompiler().compile(request)

    _assert_reduction(result, 64)
    exchange = next(item for item in _history(result)["tool_exchanges"] if item["call_id"] == "call-1")
    assert exchange["result"] == page
    assert len(_history(result)["tool_exchanges"]) == 65


def test_irreducible_reference_floor_reports_local_projection_limit():
    with pytest.raises(ContextBudgetExceededError) as raised:
        ContextCompiler().compile(_request(_events(1800, argument="small"), window=2_000_000))

    assert getattr(raised.value, "code", None) == "context_semantic_history_projection_exceeded"
    assert "larger context window" not in str(raised.value)
    assert "token context window" not in str(raised.value)
