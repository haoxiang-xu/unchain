"""New catalog identities retain function/thinking replay after a cold import."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import httpx
from openai import OpenAI

from unchain.agent import Agent, MemoryModule, ToolsModule
from unchain.memory import JsonFileSessionStore, MemoryManager
from unchain.providers import AnthropicModelIO, OpenAIModelIO
from tests.test_provider_replay import (
    _anthropic_factory, _anthropic_thinking_tool_events, _openai_factory,
)
from tests.test_anthropic_text_replay import (
    test_sdk_text_blocks_survive_tool_replay_and_followup as _check_sdk_followup,
)

MODELS = [
    ("openai", "gpt-6.1-sol", "medium"),
    ("openai", "gpt-6.1-terra", "medium"),
    ("anthropic", "claude-sonnet-5-5", "high"),
    ("anthropic", "claude-haiku-5-5", "medium"),
    ("anthropic", "claude-fable-5-2", "high"),
]


def _outputs(provider, kind, call_id="call-1"):
    if provider == "openai":
        if kind == "tool":
            return [
                {"type": "reasoning", "id": f"rs-{call_id}", "summary": [], "encrypted_content": f"cipher-{call_id}"},
                {"type": "function_call", "id": f"fc-{call_id}", "call_id": call_id,
                 "name": "demo_tool", "arguments": '{"x":2}'},
            ]
        return [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": kind}]}]
    if kind == "tool":
        events = _anthropic_thinking_tool_events(signature=f"sig-{call_id}")
        events[4].content_block.id = call_id
        return events
    return [SimpleNamespace(type="content_block_delta", index=0,
                            delta=SimpleNamespace(type="text_delta", text=kind)),
            SimpleNamespace(type="content_block_stop", index=0)]


@pytest.mark.parametrize("provider,model,effort", MODELS)
def test_new_model_cold_resume_second_tool_and_second_message(tmp_path, provider, model, effort):
    effects, wires = [], []

    def demo_tool(x: int):
        effects.append(x)
        return x + 1

    def agent(outputs):
        make_client = (_openai_factory if provider == "openai" else _anthropic_factory)(outputs, wires)

        def factory(**settings):
            client = make_client(**settings)
            transport = client.responses if provider == "openai" else client.messages
            method = "create" if provider == "openai" else "stream"
            send = getattr(transport, method)

            def strict_send(**wire):
                expected = ({"model", "input", "instructions", "max_output_tokens", "truncation",
                             "store", "include", "reasoning", "stream", "tools"}
                            if provider == "openai" else
                            {"model", "messages", "max_tokens", "output_config", "tools"})
                # System instructions are owned by the runtime, not user extensions.
                assert set(wire) in (expected, expected | {"system"})
                assert wire["model"] == model
                assert wire["reasoning" if provider == "openai" else "output_config"] == {"effort": effort}
                assert wire["max_output_tokens" if provider == "openai" else "max_tokens"] == 128_000
                return send(**wire)

            setattr(transport, method, strict_send)
            return client

        memory = MemoryManager(store=JsonFileSessionStore(tmp_path))
        io_type = OpenAIModelIO if provider == "openai" else AnthropicModelIO
        return Agent(
            name="catalog-cold-replay", provider=provider, model=model,
            modules=(ToolsModule(tools=(demo_tool,)), MemoryModule(memory=memory)),
            model_io_factory=lambda spec, context: io_type(model=model, api_key="fixture", client_factory=factory),
        ), memory

    first, memory = agent([_outputs(provider, "tool")])
    stopped = first.run("call a tool", session_id="catalog-session", max_iterations=1,
                        payload={"temperature": 0.2, "top_p": 0.5, "top_k": 4, "unknown": True})
    assert stopped.status == "max_iterations"
    assert effects == [2]
    saved = memory.store.load("catalog-session")["execution_checkpoint"]
    assert saved["replay_frame"]["complete"] is True

    resumed, new_memory = agent([_outputs(provider, "tool", "call-2"), _outputs(provider, "done"),
                                 _outputs(provider, "second answer")])
    completed = resumed.run("continue", session_id="catalog-session", max_iterations=3)
    assert completed.status == "completed"
    assert effects == [2, 2]  # one initial effect and one new effect, no duplicate
    replay = wires[1]["input" if provider == "openai" else "messages"]
    assert ("cipher-call-1" if provider == "openai" else "sig-call-1") in json.dumps(replay)
    assert "execution_checkpoint" not in new_memory.store.load("catalog-session")
    followup = resumed.run("second normal message", session_id="catalog-session", max_iterations=2)
    assert followup.status == "completed"
    assert effects == [2, 2]
    assert len(wires) == 4


@pytest.mark.parametrize("model", [entry[1] for entry in MODELS if entry[0] == "anthropic"])
def test_new_claude_sdk_blocks_survive_two_tool_interactions(model):
    _check_sdk_followup(model)


@pytest.mark.parametrize("model,effort", [
    (model, effort) for model, efforts in (
        ("gpt-6.1-sol", ["low", "medium", "high", "xhigh", "max"]),
        ("gpt-6.1-terra", ["none", "low", "medium", "high", "xhigh", "max"]),
    ) for effort in efforts
])
def test_pinned_openai_sdk_serializes_documented_efforts_without_typing_rejection(model, effort):
    observed = []

    def handle(request):
        assert request.url.path == "/v1/responses"
        body = json.loads(request.content)
        assert body == {
            "model": model, "input": "hello", "max_output_tokens": 64,
            "reasoning": {"effort": effort}, "stream": False,
        }
        observed.append(body)
        return httpx.Response(200, json={"id": "resp-serialized", "object": "response", "output": []})

    with OpenAI(api_key="fixture", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        client.responses.create(model=model, input="hello", max_output_tokens=64,
                                reasoning={"effort": effort}, stream=False)
    assert len(observed) == 1


def _process_phase(state_dir, provider, model, phase):
    """Invoked in fresh interpreters so no in-memory checkpoint can help."""
    state_dir = Path(state_dir)
    effects_file = state_dir / "effects.jsonl"
    wires = []

    def demo_tool(x: int):
        with effects_file.open("a") as effects:
            effects.write(json.dumps(x) + "\n")
        return x + 1

    outputs = ([_outputs(provider, "tool")] if phase == 0 else
               [_outputs(provider, "tool", "call-2"), _outputs(provider, "done"), _outputs(provider, "second")])
    factory = (_openai_factory if provider == "openai" else _anthropic_factory)(outputs, wires)
    io_type = OpenAIModelIO if provider == "openai" else AnthropicModelIO
    memory = MemoryManager(store=JsonFileSessionStore(state_dir / "sessions"))
    agent = Agent(
        name="catalog-process-restart", provider=provider, model=model,
        modules=(ToolsModule(tools=(demo_tool,)), MemoryModule(memory=memory)),
        model_io_factory=lambda spec, context: io_type(model=model, api_key="fixture", client_factory=factory),
    )
    result = agent.run("initial" if phase == 0 else "continue", session_id="restart", max_iterations=1 if phase == 0 else 3)
    assert result.status == ("max_iterations" if phase == 0 else "completed")
    assert all(wire["model"] == model for wire in wires)
    if phase:
        first_wire = json.dumps(wires[0])
        assert ("cipher-call-1" if provider == "openai" else "sig-call-1") in first_wire
        assert agent.run("second normal message", session_id="restart", max_iterations=2).status == "completed"
    assert effects_file.read_text().splitlines() == (["2"] if phase == 0 else ["2", "2"])


@pytest.mark.parametrize("provider,model,_effort", MODELS)
def test_installed_model_checkpoint_survives_interpreter_restart(tmp_path, provider, model, _effort):
    env = dict(os.environ)
    # Preserve the selected runtime source/site-packages environment; each
    # child reports failures rather than reusing any parent Python objects.
    for phase in (0, 1):
        subprocess.run([
            sys.executable, "-c",
            "import sys; from tests.test_ticket_387_replay import _process_phase; "
            "_process_phase(sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]))",
            str(tmp_path), provider, model, str(phase),
        ], cwd=Path(__file__).resolve().parents[1], env=env, check=True, capture_output=True, text=True)


@pytest.mark.parametrize("provider,model,effort", MODELS)
def test_catalog_identity_and_defaults_reach_verified_durable_wire(provider, model, effort):
    from unchain.providers import ModelTurnRequest
    from unchain.providers.prepared_request_factory import resolve_prepared_provider_request_payload
    from unchain.providers.wire_preparer import prepare_provider_wire
    from unchain.providers.wire_envelope import ProviderWireEnvelope
    from tests.test_provider_wire_preparer import ATTEMPT, _prepared_turn_for_request

    io_type = OpenAIModelIO if provider == "openai" else AnthropicModelIO
    model_io = io_type(model=model, api_key="fixture", client_factory=lambda **_: None)
    request = resolve_prepared_provider_request_payload(model_io=model_io, request=ModelTurnRequest(
        messages=[{"role": "user", "content": "hello"}],
        payload={"temperature": 0.1, "unknown": True},
    ))
    prepared, draft = _prepared_turn_for_request(model_io=model_io, request_payload=request)
    envelope = prepare_provider_wire(prepared, model_io=model_io, attempt=ATTEMPT,
                                     iteration=7, transport_target_sha256="a" * 64)
    restored = ProviderWireEnvelope.from_dict(envelope.to_dict())
    assert restored.verify_against_catalog(draft.catalog) is restored
    wire = restored.request_copy()
    expected = ({"model", "input", "instructions", "max_output_tokens", "truncation", "store", "include", "reasoning", "stream", "tools"}
                if provider == "openai" else {"model", "messages", "max_tokens", "output_config", "tools"})
    assert set(wire) == expected
    assert wire["model"] == model
    assert wire["reasoning" if provider == "openai" else "output_config"] == {"effort": effort}
