from __future__ import annotations

import pytest
from types import SimpleNamespace

from unchain.context.artifacts import ArtifactService
from unchain.context.ports import BoundArtifactRepository
from unchain.context.projector import (
    CanonicalSemanticEventProjector,
    ShadowObservedToolEventAdapter,
)
from unchain.events.normalizer import (
    RuntimeEventNormalizerContext,
    normalize_raw_event,
)
from unchain.context.tool_harness import ContextToolAuthorityHarness
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.kernel.types import ToolCall as RuntimeToolCall
from unchain.input import build_ask_user_question_tool
from unchain.journal import AttemptRef, GenerationRef
from unchain.journal.ports import BoundToolReceiptIndex
from unchain.tools import Tool, Toolkit, tool
from unchain.tools.base import ToolContext
from unchain.tools.execution import ToolExecutionHarness


def _echo(value: str = "ok"):
    return value


def test_timeline_merge_policy_is_developer_configurable_through_public_apis():
    assert Tool(name="legacy").timeline_merge_policy == "approved"
    assert Tool.from_callable(_echo).timeline_merge_policy == "approved"
    assert Tool(name="direct", timeline_merge_policy="always").timeline_merge_policy == "always"
    assert Tool.from_callable(_echo, timeline_merge_policy="never").timeline_merge_policy == "never"
    assert tool(func=_echo, timeline_merge_policy="no_feedback").timeline_merge_policy == "no_feedback"
    assert tool(_echo, timeline_merge_policy="never").timeline_merge_policy == "never"

    @Tool(name="decorator", timeline_merge_policy="always")
    def tool_class_decorator():
        return "ok"

    @tool(timeline_merge_policy="always")
    def decorated():
        return "ok"

    toolkit = Toolkit()
    registered = toolkit.register(_echo, timeline_merge_policy="never")
    registered_obj = toolkit.register(
        Tool.from_callable(_echo), timeline_merge_policy="no_feedback"
    )
    via_toolkit_method = toolkit.tool(_echo, timeline_merge_policy="always")

    @toolkit.tool(timeline_merge_policy="never")
    def toolkit_decorated():
        return "ok"

    assert decorated.timeline_merge_policy == "always"
    assert tool_class_decorator.timeline_merge_policy == "always"
    assert registered.timeline_merge_policy == "never"
    assert registered_obj.timeline_merge_policy == "no_feedback"
    assert via_toolkit_method.timeline_merge_policy == "always"
    assert toolkit_decorated.timeline_merge_policy == "never"


def test_timeline_merge_policy_survives_tool_decorator_clone_and_existing_registration():
    decorated = Tool(name="configured", timeline_merge_policy="always")(_echo)
    assert decorated.timeline_merge_policy == "always"
    registered = Toolkit().register(decorated)
    assert registered.timeline_merge_policy == "always"


@pytest.mark.parametrize("bad", ["sometimes", "", None, 3])
def test_timeline_merge_policy_rejects_invalid_developer_values(bad):
    with pytest.raises(ValueError, match="timeline_merge_policy"):
        Tool(name="invalid", timeline_merge_policy=bad)

    with pytest.raises(ValueError, match="timeline_merge_policy"):
        Toolkit().register(_echo, timeline_merge_policy=bad)

    with pytest.raises(ValueError, match="timeline_merge_policy"):
        Toolkit().register(Tool(name="registered"), timeline_merge_policy=bad)


def test_question_tool_defaults_to_never_but_developer_can_override():
    assert build_ask_user_question_tool().timeline_merge_policy == "never"
    toolkit = Toolkit()
    question = toolkit.register(
        build_ask_user_question_tool(), timeline_merge_policy="always"
    )
    assert question.timeline_merge_policy == "always"


def test_timeline_merge_policy_does_not_change_provider_schema():
    default = Tool.from_callable(_echo, name="sample")
    for provider in ("openai", "anthropic", "ollama", "gemini"):
        assert default.to_provider_json(provider) == Tool.from_callable(
            _echo, name="sample", timeline_merge_policy="always"
        ).to_provider_json(provider)


def test_timeline_merge_policy_survives_raw_normalizer_and_canonical_projection():
    context = RuntimeEventNormalizerContext(session_id="s", root_run_id="r")
    raw = {
        "type": "tool_call",
        "run_id": "r",
        "iteration": 1,
        "tool_name": "lookup",
        "call_id": "c1",
        "arguments": {"value": "x"},
        "timeline_merge_policy": "always",
    }
    normalized = normalize_raw_event(raw, context=context)
    assert normalized[0].payload["timeline_merge_policy"] == "always"

    artifacts = ArtifactService(
        _ArtifactRepository(), sanitizer=lambda content, media_type: content
    )
    projector = CanonicalSemanticEventProjector(
        attempt=AttemptRef(GenerationRef("e", "g"), "r"),
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    projected = projector(raw)
    assert projected.payload["timeline_merge_policy"] == "always"
    malformed = projector(
        {**raw, "call_id": "c2", "timeline_merge_policy": ["always"]}
    )
    assert malformed.payload["timeline_merge_policy"] == "never"

    observed = ShadowObservedToolEventAdapter(
        attempt=AttemptRef(GenerationRef("e", "g"), "r"),
        journal=_ReceiptIndex("e"),
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    observed_call = observed.project_tool_call(raw)
    assert observed_call.payload["timeline_merge_policy"] == "always"
    observed_malformed = observed.project_tool_call(
        {**raw, "call_id": "c3", "timeline_merge_policy": ["always"]}
    )
    assert observed_malformed.payload["timeline_merge_policy"] == "never"


def test_present_malformed_policy_is_conservatively_never_across_python_transport():
    context = RuntimeEventNormalizerContext(session_id="s", root_run_id="r")
    malformed = {
        "type": "tool_call",
        "run_id": "r",
        "iteration": 1,
        "tool_name": "lookup",
        "call_id": "malformed-call",
        "arguments": {},
        "timeline_merge_policy": ["always"],
    }
    normalized = normalize_raw_event(malformed, context=context)
    assert normalized[0].payload["timeline_merge_policy"] == "never"

    artifacts = ArtifactService(
        _ArtifactRepository(), sanitizer=lambda content, media_type: content
    )
    projector = CanonicalSemanticEventProjector(
        attempt=AttemptRef(GenerationRef("e", "g"), "r"),
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    canonical = projector(malformed)
    assert canonical.payload["timeline_merge_policy"] == "never"

    observed = ShadowObservedToolEventAdapter(
        attempt=AttemptRef(GenerationRef("e", "g"), "r"),
        journal=_ReceiptIndex("e"),
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    shadow = observed.project_tool_call(malformed)
    assert shadow.payload["timeline_merge_policy"] == "never"

    durable_request = {
        "interaction_id": "interaction",
        "kind": "tool_approval",
        "payload": {"call_id": "malformed-call", "tool_name": "lookup"},
    }
    interaction = normalize_raw_event(
        {
            "type": "interaction_requested",
            "run_id": "r",
            "interaction_request": durable_request,
            "timeline_merge_policy": ["always"],
        },
        context=context,
    )[0]
    assert interaction.payload["timeline_merge_policy"] == "never"
    assert interaction.payload["request"] == durable_request


def test_normalizer_keeps_interaction_policy_outside_durable_request_and_fails_closed():
    context = RuntimeEventNormalizerContext(session_id="s", root_run_id="r")
    request = {
        "interaction_id": "i1",
        "kind": "tool_approval",
        "payload": {"call_id": "c1", "tool_name": "lookup", "arguments": {}},
    }
    raw = {
        "type": "interaction_requested",
        "run_id": "r",
        "interaction_request": request,
        "timeline_merge_policy": "always",
    }
    normalized = normalize_raw_event(raw, context=context)
    assert normalized[0].payload["timeline_merge_policy"] == "always"
    assert normalized[0].payload["request"] == request
    assert "timeline_merge_policy" not in normalized[0].payload["request"]

    invalid = {
        "type": "tool_call",
        "run_id": "r",
        "iteration": 1,
        "tool_name": "lookup",
        "call_id": "c2",
        "arguments": {},
        "timeline_merge_policy": "future-mode",
    }
    old = {key: value for key, value in invalid.items() if key != "timeline_merge_policy"}
    invalid_payload = normalize_raw_event(invalid, context=context)[0].payload
    old_payload = normalize_raw_event(old, context=context)[0].payload
    assert invalid_payload["timeline_merge_policy"] == "never"
    assert "timeline_merge_policy" not in old_payload


class _RecordingLoop:
    def __init__(self):
        self.events = []

    def emit_event(self, callback, event_type, run_id, *, iteration, **payload):
        self.events.append(
            {
                "type": event_type,
                "run_id": run_id,
                "iteration": iteration,
                **payload,
            }
        )


def test_legacy_and_context_harnesses_emit_declared_raw_call_policy(monkeypatch):
    import unchain.tools.execution as execution_module

    loop = _RecordingLoop()
    toolkit = Toolkit()
    toolkit.register(_echo, name="legacy_lookup", timeline_merge_policy="always")
    state = RunState(iteration=1)
    call = RuntimeToolCall(call_id="legacy-call", name="legacy_lookup", arguments={})
    harness_context = HarnessContext(
        state=state,
        phase="on_tool_call",
        event={"run_id": "run", "toolkit": toolkit, "tool_call": call, "loop": loop},
    )

    def stop_after_emit(**kwargs):
        raise RuntimeError("stop after captured producer event")

    monkeypatch.setattr(execution_module, "prepare_tool_confirmation", stop_after_emit)
    with pytest.raises(RuntimeError, match="stop after captured"):
        ToolExecutionHarness()._on_tool_call(ToolContext(harness_context, "test"))
    assert loop.events[0]["timeline_merge_policy"] == "always"

    toolkit.register(_echo, name="context_lookup", timeline_merge_policy="never")
    call = RuntimeToolCall(call_id="context-call", name="context_lookup", arguments={})
    context_loop = _RecordingLoop()

    class _StopRuntime:
        def prepare_tool_execution(self, context):
            raise RuntimeError("stop after captured producer event")

    context_v2 = HarnessContext(
        state=RunState(iteration=2),
        phase="on_tool_call",
        event={
            "run_id": "run",
            "toolkit": toolkit,
            "tool_call": call,
            "loop": context_loop,
        },
    )
    with pytest.raises(RuntimeError, match="stop after captured"):
        ContextToolAuthorityHarness(runtime=_StopRuntime()).build_delta(context_v2)
    assert context_loop.events[0]["timeline_merge_policy"] == "never"


class _ArtifactRepository(BoundArtifactRepository):
    def __init__(self):
        super().__init__("e")

    def put(self, *, content, media_type, operation, preview=""):
        raise AssertionError("tool-call projection does not write artifacts")

    def read_verified(self, *, artifact, offset=0, limit=65_536):
        raise AssertionError("tool-call projection does not read artifacts")

    def read_full_verified(self, *, artifact):
        raise AssertionError("tool-call projection does not read artifacts")


class _ReceiptIndex(BoundToolReceiptIndex):
    def append(self, *, request):
        raise AssertionError("tool-call projection does not append receipts")

    def read(self, *, after=None, limit=100):
        raise AssertionError("tool-call projection does not read journal pages")

    def capture_snapshot(self, *, max_events=10_000, max_bytes=32 * 1024 * 1024):
        raise AssertionError("tool-call projection does not capture snapshots")

    def lookup_tool_execution_receipts(self, *, attempt, call_id):
        raise AssertionError("tool-call projection does not look up receipts")
