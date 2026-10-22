from __future__ import annotations

from datetime import datetime, timezone

from unchain.events import RuntimeEvent
from unchain.events.bridge import RuntimeEventBridge


def _id_factory():
    counter = {"value": 0}

    def next_id() -> str:
        counter["value"] += 1
        return f"evt-{counter['value']}"

    return next_id


def _clock() -> datetime:
    return datetime(2026, 5, 26, 12, 34, 56, 789000, tzinfo=timezone.utc)


def test_v4_bridge_emits_session_started_with_seq():
    bridge = RuntimeEventBridge(
        session_id="thread-1",
        root_run_id="run-root",
        id_factory=_id_factory(),
        clock=_clock,
    )

    event = bridge.emit_session_started({"model": "gpt-5"})

    assert isinstance(event, RuntimeEvent)
    assert event.to_dict()["schema_version"] == "v4"
    assert event.event_id == "evt-1"
    assert event.seq == 1
    assert event.type == "session.started"
    assert event.surface.slot == "debug"


def test_v4_bridge_normalizes_raw_events_with_incrementing_seq():
    bridge = RuntimeEventBridge(
        session_id="thread-1",
        root_agent_id="developer",
        id_factory=_id_factory(),
        clock=_clock,
    )

    started = bridge.normalize(
        {
            "type": "run_started",
            "run_id": "run-root",
            "iteration": 0,
            "provider": "openai",
            "model": "gpt-5",
        }
    )
    tool = bridge.normalize(
        {
            "type": "tool_call",
            "run_id": "run-root",
            "iteration": 0,
            "tool_name": "read",
            "call_id": "call-1",
        }
    )

    assert bridge.root_run_id == "run-root"
    assert started[0].type == "run.started"
    assert started[0].seq == 1
    assert tool[0].type == "step.started"
    assert tool[0].seq == 2


def test_v4_bridge_records_dropped_unknown_events():
    bridge = RuntimeEventBridge(
        session_id="thread-1",
        root_run_id="run-root",
        id_factory=_id_factory(),
        clock=_clock,
    )

    assert bridge.normalize({"type": "unknown_event", "value": 1}) == []

    diagnostics = bridge.diagnostics()
    assert diagnostics["dropped_event_count"] == 1
    assert diagnostics["dropped_events"][0]["type"] == "unknown_event"


def test_v4_bridge_never_echoes_rejected_provider_retry_content():
    bridge = RuntimeEventBridge(session_id="thread-1", root_run_id="run-root")
    assert bridge.normalize({
        "type": "provider_retry", "run_id": "run-root", "iteration": 1,
        "provider": "gemini", "http_status": 503,
        "retry_ordinal": 1, "max_retries": 2, "delay_ms": 500,
        "response_body": "private-provider-content",
    }) == []
    assert bridge.diagnostics() == {
        "dropped_event_count": 1,
        "dropped_events": [{"type": "provider_retry", "event": {"type": "provider_retry"}}],
    }
