"""Read evidence must survive later provider requests, not only its first turn."""
from __future__ import annotations

import hashlib
import json

import pytest

from unchain.context import ContextCompileRequest, ContextCompiler, resolve_context_budget
from unchain.journal import ResourceRef
from unchain.toolkits.builtin.core import CoreToolkit
from unchain.tools.output_management import ToolOutputManager


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "ollama"])
@pytest.mark.parametrize("characters", [4800, 13000])
def test_complete_small_result_survives_later_historical_provider_turn(provider, characters):
    source = {"content": "x" * characters + "-complete-end-447"}
    raw = json.dumps(source, sort_keys=True, separators=(",", ":")).encode()
    ref = ResourceRef("artifact", "historical-small-447", 1)
    digest = hashlib.sha256(raw).hexdigest()
    projection = ToolOutputManager.active_default().project(
        raw, full_output_ref=ref.to_dict(), digest=digest,
        content_bytes=len(raw), requested_policy="head_tail",
    )
    assert projection.payload["inline"] is True
    result = ContextCompiler().compile(ContextCompileRequest(
        case="ticket-447-later-turn", provider=provider, model="synthetic",
        source_messages=({"role": "system", "content": "current policy"},
                         {"role": "user", "content": "Use the earlier read evidence."}),
        budget=resolve_context_budget(context_window_tokens=32768),
        semantic_events=(
            {"type": "tool_call", "event_id": "read-call-447", "store_seq": 1,
             "call_id": "read-447", "tool_name": "read", "arguments": {"path": "source.txt"}},
            {"type": "tool_result", "event_id": "read-result-447", "store_seq": 2,
             "call_id": "read-447", "tool_name": "read", "result": source,
             "result_bytes": len(raw), "result_sha256": digest,
             "full_output_ref": ref.to_dict(),
             "model_projection": {"result": projection.payload,
                                  "metadata": {"projection_policy": "head_tail"}}},
        ),
    ))
    history = next(message["content"] for message in result.messages
                   if str(message.get("content", "")).startswith("[MEMORY_V2_EVIDENCE_HISTORY]"))
    exchange = json.loads(history.split("\n", 2)[2])["tool_exchanges"][0]
    assert json.loads(exchange["result"]["preview"]) == source
    assert exchange["result"]["inline"] is True
    assert "read_request" not in exchange["result"]


def test_core_read_guidance_reaches_merged_builder_system_prompt(tmp_path):
    from unchain.agent import Agent, AgentBuilder, AgentCallContext
    from unchain.agent.model_io import ModelIOFactoryRegistry
    from unchain.tools import render_tool_prompt_block

    agent = Agent(name="ticket-447-prompt")
    builder = AgentBuilder(agent=agent, spec=agent.spec, state=agent.state,
                           call_context=AgentCallContext(mode="run", input_messages=[]),
                           model_io_registry=ModelIOFactoryRegistry())
    builder.add_tool(CoreToolkit(workspace_root=tmp_path))
    prompt = render_tool_prompt_block(builder.toolkit)
    assert "EOF" in prompt
    assert "byte" in prompt and "line" in prompt
    assert "reuse" in prompt.lower() and "missing" in prompt.lower()
    assert "read the full source" in prompt.lower()
    assert "edit-safety checks require it" in prompt.lower()
