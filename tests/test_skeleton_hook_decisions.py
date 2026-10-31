"""CP16 Hook decision contract, independent of model and tool dispatch."""
from __future__ import annotations

import pytest

from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hooks_runtime import apply_after_decision, apply_before_decision


def test_before_decisions_modify_only_arguments_accumulate_approval_and_can_block():
    arguments = {"query": "raw", "limit": 1}
    modified, approval, blocked = apply_before_decision(
        arguments,
        False,
        {"decision": "continue", "arguments": {"query": "safe", "limit": 1}, "require_approval": True},
        ("hooks", 0),
    )
    assert (modified, approval, blocked) == ({"query": "safe", "limit": 1}, True, None)
    arguments["query"] = "mutated outside"
    assert modified["query"] == "safe"

    unchanged, still_requires_approval, blocked = apply_before_decision(
        modified,
        approval,
        {"decision": "block", "reason": "policy denied"},
        ("hooks", 1),
    )
    assert unchanged == {"query": "safe", "limit": 1}
    assert still_requires_approval is True
    assert blocked == {"code": "hook_blocked", "reason": "policy denied"}


def test_after_replace_becomes_the_only_input_to_later_after_hooks():
    raw = {"secret": "sentinel", "visible": "original"}
    current = apply_after_decision(raw, {"decision": "replace", "result": {"visible": "redacted"}}, ("hooks", 0))
    later = apply_after_decision(current, {"decision": "continue"}, ("hooks", 1))
    assert later == {"visible": "redacted"}
    assert raw == {"secret": "sentinel", "visible": "original"}


@pytest.mark.parametrize("event, decision", [
    ("before_tool_call", {"decision": "replace", "result": {}}),
    ("before_tool_call", {"decision": "continue", "tool_name": "other"}),
    ("before_tool_call", {"decision": "block", "reason": "", "require_approval": False}),
    ("after_tool_call", {"decision": "continue", "result": {}}),
    ("after_tool_call", {"decision": "replace", "result": {"ok": True}, "unknown": True}),
])
def test_hook_decisions_are_closed_and_phase_specific(event, decision):
    with pytest.raises(SkeletonValidationError):
        if event == "before_tool_call":
            apply_before_decision({}, False, decision, ("hooks", 0))
        else:
            apply_after_decision({}, decision, ("hooks", 0))
