from __future__ import annotations

import pytest

from unchain.context.artifacts import ArtifactService
from unchain.context.ports import BoundArtifactRepository
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.events.normalizer import (
    RuntimeEventNormalizerContext,
    normalize_raw_event,
)
from unchain.input import build_ask_user_question_tool
from unchain.journal import AttemptRef, GenerationRef
from unchain.tools import Tool, Toolkit, tool


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


def test_normalizer_keeps_interaction_policy_outside_durable_request_and_drops_invalid():
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
    assert "timeline_merge_policy" not in invalid_payload
    assert "timeline_merge_policy" not in old_payload


class _ArtifactRepository(BoundArtifactRepository):
    def __init__(self):
        super().__init__("e")

    def put(self, *, content, media_type, operation, preview=""):
        raise AssertionError("tool-call projection does not write artifacts")

    def read_verified(self, *, artifact, offset=0, limit=65_536):
        raise AssertionError("tool-call projection does not read artifacts")

    def read_full_verified(self, *, artifact):
        raise AssertionError("tool-call projection does not read artifacts")
