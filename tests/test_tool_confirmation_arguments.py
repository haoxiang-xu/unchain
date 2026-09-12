"""Provider argument encoding must not change shell authorization (#264)."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from unchain.kernel.types import ToolCall
from unchain.toolkits import CoreToolkit
from unchain.tools import Toolkit, execute_confirmable_tool_call
from unchain.tools.confirmation import prepare_tool_confirmation


@pytest.fixture
def shell_toolkit(tmp_path):
    core = CoreToolkit(workspace_root=str(tmp_path))
    toolkit = Toolkit()
    for tool in core.tools.values():
        toolkit.register(tool)
    yield toolkit
    core.shutdown()


@pytest.mark.parametrize("encoding", ["object", "json"])
@pytest.mark.parametrize("approved", [False, True])
def test_mutating_shell_waits_for_the_same_visible_arguments(shell_toolkit, tmp_path, encoding, approved):
    command = "'approved' | Set-Content probe.txt" if sys.platform.startswith("win") else "printf approved > probe.txt"
    arguments = {"action": "run", "command": command}
    raw = json.dumps(arguments) if encoding == "json" else arguments
    call = ToolCall(call_id="shell-approval", name="shell", arguments=raw)
    preparation = prepare_tool_confirmation(toolkit=shell_toolkit, tool_call=call)
    assert not (tmp_path / "probe.txt").exists()
    assert preparation.needs_confirmation_response
    assert preparation.request.arguments == arguments
    assert preparation.effective_arguments == arguments
    observed = []

    def confirm(request):
        assert not (tmp_path / "probe.txt").exists()
        assert request.arguments == arguments
        observed.append(request.call_id)
        return {"approved": approved}

    outcome = execute_confirmable_tool_call(
        toolkit=shell_toolkit, tool_call=call, prepared_confirmation=preparation,
        on_tool_confirm=confirm, loop=None, callback=None, run_id="run-shell", iteration=0,
    )
    assert observed == ["shell-approval"]
    assert outcome.denied is not approved
    assert (tmp_path / "probe.txt").exists() is approved


@pytest.mark.parametrize("arguments", [{}, {"action": "run"}, {"action": "run", "command": ""}, {"action": "run", "command": 7}])
def test_unreadable_shell_command_never_grants_confirmation_exemption(shell_toolkit, arguments):
    call = ToolCall(call_id="empty-shell", name="shell", arguments=arguments)
    assert prepare_tool_confirmation(toolkit=shell_toolkit, tool_call=call).requires_confirmation


@pytest.mark.parametrize("arguments", ['{"action":', '[]', 'null', 'true', '"pwd"', []])
def test_malformed_or_non_object_arguments_fail_before_execution(shell_toolkit, arguments):
    call = ToolCall(call_id="invalid-shell", name="shell", arguments=arguments)
    preparation = prepare_tool_confirmation(toolkit=shell_toolkit, tool_call=call)
    assert preparation.resolver_error
    assert preparation.request is None


def test_json_read_only_shell_uses_the_same_policy(shell_toolkit):
    command = "Get-Location" if sys.platform.startswith("win") else "pwd"
    call = ToolCall(call_id="read-only", name="shell", arguments=json.dumps({"action": "run", "command": command}))
    preparation = prepare_tool_confirmation(toolkit=shell_toolkit, tool_call=call)
    assert preparation.resolver_error is None
    assert preparation.requires_confirmation is False


@pytest.mark.parametrize("command", ["mkdir probe && touch probe/file", "cat > probe.txt <<'EOF'\nhello\nEOF"])
def test_real_openai_adapter_produces_a_confirmable_shell_request(shell_toolkit, command):
    from unchain.providers import OpenAIModelIO
    from unchain.providers.base import ModelTurnRequest
    arguments = {"action": "run", "command": command}

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def __iter__(self):
            yield SimpleNamespace(type="response.completed", response=SimpleNamespace(
                id="response-shell", usage={"input_tokens": 1, "output_tokens": 1},
                output=[{"type": "function_call", "call_id": "provider-call", "name": "shell", "arguments": json.dumps(arguments)}],
            ))

    model_io = OpenAIModelIO(
        model="gpt-4.1-mini", api_key="test-key",
        client_factory=lambda **_kwargs: SimpleNamespace(responses=SimpleNamespace(create=lambda **_request: Stream())),
    )
    turn = model_io.fetch_turn(ModelTurnRequest(messages=[{"role": "user", "content": "Create the probe"}], toolkit=shell_toolkit))
    assert len(turn.tool_calls) == 1
    assert isinstance(turn.tool_calls[0].arguments, str)
    preparation = prepare_tool_confirmation(toolkit=shell_toolkit, tool_call=turn.tool_calls[0])
    assert preparation.needs_confirmation_response
    assert preparation.request.call_id == "provider-call"
    assert preparation.request.arguments == arguments
