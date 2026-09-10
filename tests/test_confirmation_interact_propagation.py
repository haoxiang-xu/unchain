"""Test that execute_confirmable_tool_call propagates policy interact fields."""
from __future__ import annotations

from unittest.mock import MagicMock

from unchain.tools.confirmation import execute_confirmable_tool_call
from unchain.tools.models import ToolConfirmationPolicy, ToolConfirmationRequest
from unchain.tools import Tool


def _make_tool_call(name: str = "write", args: dict | None = None, call_id: str = "c-1"):
    tool_call = MagicMock()
    tool_call.name = name
    tool_call.call_id = call_id
    tool_call.arguments = args if args is not None else {"path": "foo.py", "content": "x"}
    return tool_call


def _make_tool_obj(resolver_return):
    tool_obj = Tool.from_callable(lambda path, content: {"ok": True}, name="write")
    tool_obj.requires_confirmation = True
    tool_obj.observe = False
    tool_obj.description = "write"
    tool_obj.render_component = None
    tool_obj.confirmation_resolver = MagicMock(return_value=resolver_return)
    return tool_obj


def test_policy_interact_fields_reach_request():
    captured = {}

    def on_tool_confirm(req: ToolConfirmationRequest):
        captured["req"] = req
        return {"approved": True, "modified_arguments": None}

    policy = ToolConfirmationPolicy(
        requires_confirmation=True,
        description="Edit foo.py",
        interact_type="code_diff",
        interact_config={
            "title": "Edit foo.py",
            "operation": "edit",
            "path": "foo.py",
            "unified_diff": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new\n",
            "truncated": False,
            "total_lines": 5,
            "displayed_lines": 5,
            "fallback_description": "edit foo.py (+1 -1)",
        },
    )
    toolkit = MagicMock()
    toolkit.get.return_value = _make_tool_obj(policy)
    toolkit.execute.return_value = {"ok": True}

    outcome = execute_confirmable_tool_call(
        toolkit=toolkit,
        tool_call=_make_tool_call(),
        on_tool_confirm=on_tool_confirm,
        loop=None,
        callback=None,
        run_id="run-1",
        iteration=0,
    )

    assert outcome.denied is False
    req = captured["req"]
    assert req.interact_type == "code_diff"
    assert req.interact_config == policy.interact_config


def test_default_policy_keeps_request_defaults():
    captured = {}

    def on_tool_confirm(req: ToolConfirmationRequest):
        captured["req"] = req
        return {"approved": True, "modified_arguments": None}

    toolkit = MagicMock()
    toolkit.get.return_value = _make_tool_obj(ToolConfirmationPolicy())
    toolkit.execute.return_value = {"ok": True}

    execute_confirmable_tool_call(
        toolkit=toolkit,
        tool_call=_make_tool_call(),
        on_tool_confirm=on_tool_confirm,
        loop=None,
        callback=None,
        run_id="run-1",
        iteration=0,
    )
    req = captured["req"]
    assert req.interact_type == "confirmation"
    assert req.interact_config is None
