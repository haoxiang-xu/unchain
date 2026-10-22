from __future__ import annotations

import pytest

from unchain.events.normalizer import RuntimeEventNormalizerContext, normalize_raw_event


def _context():
    return RuntimeEventNormalizerContext(
        session_id="thread-1",
        root_run_id="run-root",
        root_agent_id="developer",
    )


@pytest.mark.parametrize("marker", ["retry_ordinal", "max_retries"])
def test_retry_formats_cannot_fall_through_with_hybrid_fields(marker):
    grouped = {
        "type": "provider_retry", "run_id": "run-root", "iteration": 2,
        "provider": "gemini", "http_status": 503, "provider_status": "UNAVAILABLE",
        "attempt_failed": 1, "next_attempt": 2, "max_attempts": 3,
        "delay_ms": 500, "remaining_ms": 500,
    }
    [valid] = normalize_raw_event(grouped, context=_context())
    assert valid.payload["remaining_ms"] == 500
    assert valid.payload["next_attempt"] == 2
    # Even a malformed marker selects the strict ordinal-format validator;
    # the event cannot escape into the other format's projection.
    assert normalize_raw_event({**grouped, marker: None}, context=_context()) == []


def test_v4_normalizes_only_bounded_gemini_retry_progress():
    raw = {
        "type": "provider_retry", "run_id": "run-root", "iteration": 2,
        "timestamp": 1790970507.0, "provider": "gemini", "http_status": 503,
        "retry_ordinal": 1, "max_retries": 2, "delay_ms": 500,
    }
    [event] = normalize_raw_event(raw, context=_context())
    assert event.type == "step.delta"
    assert event.turn_id == "run-root:turn-2"
    assert event.links.step_id == "model:run-root:turn-2:response"
    assert event.payload == {
        "step_id": "model:run-root:turn-2:response",
        "step_type": "model_response", "kind": "provider_retry",
        "provider": "gemini", "http_status": 503,
        "retry_ordinal": 1, "max_retries": 2, "delay_ms": 500,
    }
    for mutation in (
        {"response_body": "private-provider-content"},
        {"provider": "openai"}, {"http_status": 200},
        {"retry_ordinal": 3}, {"delay_ms": -1},
    ):
        assert normalize_raw_event({**raw, **mutation}, context=_context()) == []


def test_v4_normalizes_tool_call_to_step_started():
    events = normalize_raw_event(
        {
            "type": "tool_call",
            "run_id": "run-root",
            "iteration": 2,
            "tool_name": "write",
            "call_id": "call-1",
            "arguments": {"path": "a.py"},
        },
        context=_context(),
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "step.started"
    assert event.turn_id == "run-root:turn-2"
    assert event.links.step_id == "tool:call-1"
    assert event.links.tool_call_id == "call-1"
    assert event.surface.slot == "trace_inline"
    assert event.surface.scope == "turn"
    assert event.payload == {
        "step_id": "tool:call-1",
        "step_type": "tool",
        "tool_name": "write",
        "call_id": "call-1",
        "arguments": {"path": "a.py"},
    }


@pytest.mark.parametrize(
    ("result", "durable_result_outcome", "expected_status"),
    [
        ({"ok": True}, "success", "success"),
        ({"ok": False, "error": "missing"}, "error", "error"),
        ({"denied": True}, "denied", "denied"),
        ({"cancelled": True}, "cancelled", "cancelled"),
    ],
)
def test_v4_normalizes_verified_durable_tool_result_outcome(
    result,
    durable_result_outcome,
    expected_status,
):
    [event] = normalize_raw_event(
        {
            "type": "tool_result",
            "run_id": "run-root",
            "iteration": 2,
            "tool_name": "lookup",
            "call_id": "call-1",
            "result": result,
            "durable_result_outcome": durable_result_outcome,
        },
        context=_context(),
    )

    assert event.type == "step.completed"
    assert event.payload["status"] == expected_status


@pytest.mark.parametrize(
    ("result", "expected_status"),
    [
        pytest.param(
            {"ok": True, "status": {"state": "ready"}},
            "success",
            id="object-status",
        ),
        pytest.param(
            {"ok": True, "status": ["ready"]},
            "success",
            id="array-status",
        ),
        pytest.param(
            {"ok": False, "error": "missing", "status": {}},
            "error",
            id="error-with-object-status",
        ),
    ],
)
def test_v4_normalizes_non_string_tool_result_status(
    result,
    expected_status,
):
    [event] = normalize_raw_event(
        {
            "type": "tool_result",
            "run_id": "run-root",
            "iteration": 2,
            "tool_name": "lookup",
            "call_id": "call-1",
            "result": result,
        },
        context=_context(),
    )

    assert event.type == "step.completed"
    assert event.payload["status"] == expected_status


def test_v4_normalizes_code_diff_confirmation_to_interaction_requested():
    events = normalize_raw_event(
        {
            "type": "tool_confirmation_requested",
            "run_id": "run-root",
            "iteration": 1,
            "confirmation_id": "confirm-1",
            "tool_name": "write",
            "toolkit_id": "core",
            "call_id": "call-1",
            "arguments": {"path": "a.py"},
            "interact_type": "code_diff",
            "interact_config": {"unified_diff": "--- a.py\n+++ a.py"},
            "description": "Edit a.py",
        },
        context=_context(),
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "interaction.requested"
    assert event.links.interaction_id == "confirm-1"
    assert event.links.tool_call_id == "call-1"
    assert event.surface.slot == "trace_inline"
    assert event.payload["interaction_id"] == "confirm-1"
    assert event.payload["kind"] == "code_diff"
    assert event.payload["renderer"] == "code_diff"
    assert event.payload["blocking"] is True
    assert event.payload["target"] == {
        "tool_call_id": "call-1",
        "tool_name": "write",
        "toolkit_id": "core",
        "arguments": {"path": "a.py"},
    }
    assert event.payload["config"] == {"unified_diff": "--- a.py\n+++ a.py"}


def test_v4_normalizes_human_input_to_choice_interaction():
    events = normalize_raw_event(
        {
            "type": "human_input_requested",
            "run_id": "run-root",
            "iteration": 3,
            "request_id": "input-1",
            "kind": "selection",
            "title": "Choose",
            "question": "Pick one",
            "selection_mode": "single",
            "options": [{"label": "A", "value": "a"}],
        },
        context=_context(),
    )

    assert events[0].type == "interaction.requested"
    assert events[0].links.interaction_id == "input-1"
    assert events[0].payload["kind"] == "choice"
    assert events[0].payload["renderer"] == "single"
    assert events[0].payload["selection_mode"] == "single"
    assert events[0].payload["prompt"] == "Pick one"
    assert events[0].payload["options"] == [{"label": "A", "value": "a"}]
    assert events[0].payload["config"]["selection_mode"] == "single"


def test_v4_preserves_durable_request_on_legacy_human_event() -> None:
    durable_request = {
        "interaction_id": "interaction-human-1",
        "kind": "human_input",
        "request_digest": "digest-human-1",
    }
    events = normalize_raw_event(
        {
            "type": "human_input_requested",
            "run_id": "run-root",
            "iteration": 3,
            "interaction_id": "interaction-human-1",
            "interaction_request": durable_request,
            "request_id": "call-user",
            "kind": "selector",
            "title": "Choose",
            "question": "Pick one",
            "selection_mode": "single",
            "options": [{"label": "A", "value": "a"}],
        },
        context=_context(),
    )

    assert events[0].links.interaction_id == "interaction-human-1"
    assert events[0].payload["interaction_kind"] == "human_input"
    assert events[0].payload["request"] == durable_request


def test_v4_normalizes_durable_tool_interaction_request() -> None:
    durable_request = {
        "schema_version": 1,
        "interaction_id": "interaction-tool-1",
        "kind": "tool_approval",
        "request_digest": "digest-tool-1",
        "schema_digest": "schema-tool-1",
        "payload": {
            "type": "tool_confirmation_request",
            "tool_name": "write_file",
            "call_id": "call-1",
            "arguments": {"path": "a.py"},
            "description": "Write a.py",
        },
    }

    events = normalize_raw_event(
        {
            "type": "interaction_requested",
            "run_id": "run-root",
            "iteration": 2,
            "interaction_request": durable_request,
        },
        context=_context(),
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "interaction.requested"
    assert event.links.interaction_id == "interaction-tool-1"
    assert event.links.tool_call_id == "call-1"
    assert event.payload["interaction_id"] == "interaction-tool-1"
    assert event.payload["interaction_kind"] == "tool_approval"
    assert event.payload["renderer"] == "confirmation"
    assert event.payload["target"]["tool_name"] == "write_file"
    assert event.payload["request"] == durable_request


def test_v4_normalizes_durable_max_budget_interaction_request() -> None:
    durable_request = {
        "schema_version": 1,
        "interaction_id": "interaction-max-1",
        "kind": "max_budget",
        "request_digest": "digest-max-1",
        "schema_digest": "schema-max-1",
        "payload": {
            "effective_max": 6,
            "suggested_extra_iterations": 6,
            "decision": {"iteration": 6, "max_iterations": 6},
        },
    }

    events = normalize_raw_event(
        {
            "type": "interaction_requested",
            "run_id": "run-root",
            "iteration": 6,
            "interaction_request": durable_request,
        },
        context=_context(),
    )

    assert len(events) == 1
    event = events[0]
    assert event.type == "interaction.requested"
    assert event.links.interaction_id == "interaction-max-1"
    assert event.payload["kind"] == "continuation"
    assert event.payload["interaction_kind"] == "max_budget"
    assert event.payload["config"]["effective_max"] == 6
    assert event.payload["request"] == durable_request


def test_v4_normalizes_tool_denied_to_interaction_resolved():
    events = normalize_raw_event(
        {
            "type": "tool_denied",
            "run_id": "run-root",
            "iteration": 1,
            "confirmation_id": "confirm-1",
            "call_id": "call-1",
            "reason": "no",
        },
        context=_context(),
    )

    assert events[0].type == "interaction.resolved"
    assert events[0].links.interaction_id == "confirm-1"
    assert events[0].payload == {
        "interaction_id": "confirm-1",
        "outcome": "denied",
        "response": None,
        "reason": "no",
    }


def test_v4_artifact_surface_uses_run_summary_for_workspace_change_set():
    artifact = {
        "schema_version": "unchain.artifact.v1",
        "artifact_id": "workspace_change_set:run-root",
        "kind": "workspace_change_set",
        "title": "Workspace changes",
        "snapshot": {
            "change_set_id": "wcs_run-root",
            "totals": {"files": 1},
        },
        "presentation": {
            "surface": "run_summary",
            "group": "files",
            "collapsed": False,
        },
    }

    events = normalize_raw_event(
        {
            "type": "artifact_created",
            "run_id": "run-root",
            "iteration": 4,
            "artifact_id": "workspace_change_set:run-root",
            "artifact": artifact,
        },
        context=_context(),
    )

    assert events[0].type == "artifact.created"
    assert events[0].turn_id == "run-root:turn-4"
    assert events[0].links.artifact_id == "workspace_change_set:run-root"
    assert events[0].links.workspace_change_set_id == "wcs_run-root"
    assert events[0].surface.slot == "run_summary"
    assert events[0].surface.scope == "run"
    assert events[0].surface.group == "files"
    assert events[0].surface.default_state == "expanded"
    assert events[0].payload == artifact


RETRY_RAW = {
    "type": "provider_retry",
    "run_id": "run-root",
    "iteration": 0,
    "provider": "gemini",
    "attempt_failed": 2,
    "next_attempt": 3,
    "max_attempts": 11,
    "delay_ms": 4000,
    "remaining_ms": 3000,
    "http_status": 503,
    "provider_status": "UNAVAILABLE",
}


def test_v4_maps_provider_retry_to_a_model_response_step_delta():
    """BC-386-6: no new V4 event type; the retry rides the model response step."""

    events = normalize_raw_event(dict(RETRY_RAW), context=_context())
    assert len(events) == 1
    event = events[0]
    assert event.type == "step.delta"
    assert event.turn_id == "run-root:turn-0"
    assert event.links.step_id == "model:run-root:turn-0:response"
    assert event.surface.slot == "trace_inline"
    assert event.visibility == "user"
    assert event.payload == {
        "step_id": "model:run-root:turn-0:response",
        "step_type": "model_response",
        "kind": "provider_retry",
        "provider": "gemini",
        "attempt_failed": 2,
        "next_attempt": 3,
        "max_attempts": 11,
        "delay_ms": 4000,
        "remaining_ms": 3000,
        "http_status": 503,
        "provider_status": "UNAVAILABLE",
    }


@pytest.mark.parametrize("change", [
    {"http_status": None, "provider_status": ""},
])
def test_v4_provider_retry_without_http_evidence_keeps_closed_fields(change):
    raw = dict(RETRY_RAW)
    raw.update(change)
    event = normalize_raw_event(raw, context=_context())[0]
    assert event.payload["http_status"] is None
    assert event.payload["provider_status"] == ""


@pytest.mark.parametrize("change", [
    {"attempt_failed": "2"}, {"next_attempt": 0}, {"max_attempts": True},
    {"delay_ms": -1}, {"remaining_ms": 1.5}, {"http_status": "503"},
    {"provider_status": 7}, {"next_attempt": 12},
])
def test_v4_drops_a_malformed_provider_retry(change):
    raw = dict(RETRY_RAW)
    raw.update(change)
    assert normalize_raw_event(raw, context=_context()) == []


def test_v4_provider_retry_never_copies_unknown_fields():
    raw = dict(RETRY_RAW)
    raw["message"] = "PRIVATE provider text"
    raw["toolkit_id"] = "core"
    event = normalize_raw_event(raw, context=_context())[0]
    assert "message" not in event.payload and "toolkit_id" not in event.payload
    assert "PRIVATE" not in repr(event.payload)
