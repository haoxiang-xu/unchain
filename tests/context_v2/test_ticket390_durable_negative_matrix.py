"""Exercise receipt failures through SQLite, coordinator and provider continuation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace

import pytest

from test_context_provider_turn_cross_provider import (
    _run_mixed_anthropic_compatible_tool_turn,
    _runtime,
)
from unchain.context.coordinator import ContextCompileCoordinatorError
from unchain.journal import (
    ArtifactRef,
    JournalAppendRequest,
    OperationRef,
    SemanticEventDraft,
)
from unchain.journal.provider_result import (
    ProviderTurnResultEnvelope,
    build_provider_turn_result_event_payload,
)
from unchain.providers.context_assembler import ProviderContextProjectionError


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def _rewrite_event(tmp_path, journal, event):
    event_bytes = _canonical(event.to_dict())
    with sqlite3.connect(tmp_path / "memory_v2" / "context_v2.sqlite3") as connection:
        connection.execute(
            "UPDATE events SET event_json = ?, event_sha256 = ? "
            "WHERE execution_id = ? AND event_id = ?",
            (event_bytes, hashlib.sha256(event_bytes).hexdigest(),
             journal.execution_id, event.event_id),
        )


@pytest.mark.parametrize(
    "mutation", ["duplicate_result", "stale_iteration", "all_stale_iteration"]
)
def test_conflicting_current_tool_results_stop_before_second_provider_send(
    tmp_path, mutation
):
    """An invalid authoritative batch cannot degrade into neutral history."""

    repositories, sends, calls = [], [], []
    runtime = _runtime(tmp_path, repository_observer=repositories.append)

    def mutate_first_result_after_second_tool_starts():
        if len(calls) != 2:
            return
        journal = repositories[0]
        result_events = [item for item in journal.capture_snapshot().events
                         if item.event_type == "tool_result"]
        event = result_events[0]
        if mutation == "duplicate_result":
            draft = SemanticEventDraft(
                event_id="ticket-390-duplicate-tool-result",
                event_type=event.event_type,
                attempt=event.attempt,
                operation_id="ticket-390-duplicate-tool-result-operation",
                payload=event.payload,
                resource_refs=event.resource_refs,
            )
            journal.append(request=draft.to_append_request())
            return

        events_to_change = (result_events if mutation == "all_stale_iteration"
                            else [event])
        for result_event in events_to_change:
            payload = {
                **result_event.payload,
                "iteration": result_event.payload["iteration"] + 1,
            }
            draft = SemanticEventDraft(
                event_id=result_event.event_id,
                event_type=result_event.event_type,
                attempt=result_event.attempt,
                operation_id=result_event.operation.operation_id,
                payload=payload,
                resource_refs=result_event.resource_refs,
            )
            changed = replace(result_event, payload=payload, operation=draft.operation)
            event_bytes = _canonical(changed.to_dict())
            with sqlite3.connect(tmp_path / "memory_v2" / "context_v2.sqlite3") as db:
                db.execute(
                    "UPDATE events SET event_json=?, event_sha256=? "
                    "WHERE execution_id=? AND event_id=?",
                    (event_bytes, hashlib.sha256(event_bytes).hexdigest(),
                     journal.execution_id, changed.event_id),
                )
                db.execute(
                    "UPDATE operations SET payload_sha256=? "
                    "WHERE execution_id=? AND operation_id=?",
                    (draft.operation.payload_sha256, journal.execution_id,
                     draft.operation.operation_id),
                )
        # Ensure the negative test reaches the strict current-turn consumer.
        journal.capture_snapshot()

    with pytest.raises(ContextCompileCoordinatorError,
                       match="current native tool batch"):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime,
            provider="anthropic",
            send_calls=sends,
            tool_calls=calls,
            call_specs=(("mixed-call-1", "probe", {"query": "durable"}),
                        ("mixed-call-2", "probe", {"query": "durable-2"})),
            after_tool=mutate_first_result_after_second_tool_starts,
        )
    assert calls == ["durable", "durable-2"]
    assert len(sends) == 1


def test_late_provider_result_receipt_stops_durable_continuation(tmp_path):
    repositories, sends, calls = [], [], []
    runtime = _runtime(tmp_path, repository_observer=repositories.append)

    def duplicate_result():
        journal = repositories[0]
        original = next(event for event in journal.capture_snapshot().events
                        if event.event_type == "provider.turn_result")
        appended = journal.append(request=JournalAppendRequest(
            event_id="late-result-event", event_type=original.event_type,
            attempt=original.attempt,
            operation=OperationRef("late-result-operation", "f" * 64),
            payload=original.payload, resource_refs=original.resource_refs,
        ))
        assert appended.event.store_seq > original.store_seq

    with pytest.raises(ContextCompileCoordinatorError,
                       match="conflicting provider turn results"):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime, provider="anthropic", send_calls=sends, tool_calls=calls,
            after_tool=duplicate_result,
        )
    assert calls == ["durable"]
    assert len(sends) == 1


@pytest.mark.parametrize("mutation", [
    "subject", "route", "provider", "text", "call_id", "call_name",
    "arguments", "order", "visible_output", "schema",
])
def test_mutated_durable_result_stops_before_second_provider_send(tmp_path, mutation):
    """Rewrite an isolated durable receipt coherently to reach strict consumers.

    Recompute disk digests so failures prove the continuation contract rather
    than SQLite's generic byte-tampering check. No production profile is used.
    """

    repositories, sends, calls, mutations = [], [], [], []
    runtime = _runtime(tmp_path, repository_observer=repositories.append)

    def mutate_result():
        journal = repositories[0]
        if mutation == "provider":
            wire = next(item for item in journal.capture_snapshot().events
                        if item.event_type == "provider.wire_snapshot")
            _rewrite_event(tmp_path, journal, replace(
                wire, payload={**wire.payload, "provider": "gemini"}))
            mutations.append(mutation)
            return
        event = next(item for item in journal.capture_snapshot().events
                     if item.event_type == "provider.turn_result")
        artifact = ArtifactRef.from_dict(event.payload["result_artifact"])
        raw = json.loads(journal.read_full_verified(artifact=artifact))
        envelope = ProviderTurnResultEnvelope.from_dict(raw)
        if mutation == "subject":
            raw["subject"]["attempt"]["attempt_id"] = "foreign-attempt"
        elif mutation == "route":
            raw["route_sha256"] = "e" * 64
        elif mutation == "text":
            raw["result"]["assistant_messages"][0]["content"][0]["text"] = "changed"
        elif mutation in {"call_id", "call_name", "arguments"}:
            block = raw["result"]["assistant_messages"][0]["content"][1]
            key, value = {
                "call_id": ("id", "foreign-call"),
                "call_name": ("name", "renamed_probe"),
                "arguments": ("input", {"query": "changed"}),
            }[mutation]
            block[key] = value
        elif mutation == "order":
            raw["result"]["assistant_messages"][0]["content"][1:] = list(reversed(
                raw["result"]["assistant_messages"][0]["content"][1:]))
        elif mutation == "schema":
            raw["schema"] = "unsupported.v999"
        else:
            assert mutation == "visible_output"

        raw["result_sha256"] = hashlib.sha256(_canonical(raw["result"])).hexdigest()
        content = _canonical(raw)
        changed_artifact = journal.put(
            content=content, media_type="application/json", preview="",
            operation=OperationRef("mutated-result-artifact", hashlib.sha256(content).hexdigest()),
        )
        if mutation == "schema":
            payload = {**event.payload, "result_artifact": changed_artifact.to_dict()}
        else:
            changed_envelope = ProviderTurnResultEnvelope.from_dict(raw)
            payload = build_provider_turn_result_event_payload(
                envelope=changed_envelope, artifact=changed_artifact)
        if mutation == "visible_output":
            payload["visible_output"] = not payload["visible_output"]
        changed_event = replace(event, payload=payload, resource_refs=(changed_artifact.ref,))
        _rewrite_event(tmp_path, journal, changed_event)
        mutations.append(mutation)

    expected = (ProviderContextProjectionError if mutation == "text"
                else ContextCompileCoordinatorError)
    with pytest.raises(expected):
        _run_mixed_anthropic_compatible_tool_turn(
            runtime, provider="anthropic", send_calls=sends, tool_calls=calls,
            call_specs=(("mixed-call-1", "probe", {"query": "durable"}),
                        ("mixed-call-2", "probe", {"query": "durable-2"})),
            after_tool=lambda: mutate_result() if len(calls) == 2 else None,
        )
    assert calls == ["durable", "durable-2"]
    assert mutations == [mutation]
    assert len(sends) == 1


def test_invalid_replay_format_stops_durable_continuation_at_assembler(tmp_path):
    from unchain.kernel.harness import BaseRuntimeHarness

    class InvalidReplayFormat(BaseRuntimeHarness):
        def __init__(self):
            super().__init__(name="invalid-replay-format", phases=("before_model",),
                             order=10_000)
            self.changed = False

        def build_delta(self, context):
            bucket = context.state.component_state.get("provider_replay", {})
            frame = bucket.get("frame")
            if frame is not None:
                frame["format"] = "unsupported.v999"
                self.changed = True
            return None

    base = _runtime(tmp_path)
    corruption = InvalidReplayFormat()

    class RuntimeWithInvalidReplay:
        def build_harnesses(self):
            return [*base.build_harnesses(), corruption]

        def compose_event_callback(self, callback):
            return base.compose_event_callback(callback)

    sends, calls = [], []
    with pytest.raises(ProviderContextProjectionError, match="replay format"):
        _run_mixed_anthropic_compatible_tool_turn(
            RuntimeWithInvalidReplay(), provider="anthropic", send_calls=sends,
            tool_calls=calls,
        )
    assert corruption.changed
    assert calls == ["durable"]
    assert len(sends) == 1
