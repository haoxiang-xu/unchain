"""Opt-in, credentialed release qualification; never runs in default tests.

RUN_PROVIDER_SMOKE=1 python -m pytest tests/evals/test_provider_smoke.py -m live_provider -q
Model overrides: SMOKE_GEMINI_MODEL / SMOKE_OPENAI_MODEL / SMOKE_ANTHROPIC_MODEL.
"""
from __future__ import annotations

import json
import os

import pytest

from unchain.agent.model_io import ModelIOFactoryRegistry
from unchain.kernel import KernelLoop
from unchain.providers import ModelTurnRequest
from unchain.schemas import ResponseFormat
from unchain.toolkits import CoreToolkit
from unchain.tools import Toolkit
from unchain.tools.execution import ToolExecutionHarness

pytestmark = [
    pytest.mark.live_provider,
    pytest.mark.skipif(
        os.getenv("RUN_PROVIDER_SMOKE") != "1", reason="live provider smoke is opt-in"
    ),
]


@pytest.fixture(params=["gemini", "openai", "anthropic"])
def model_io(request):
    provider = request.param
    key = os.getenv(f"{provider.upper()}_API_KEY")
    if provider == "gemini":
        key = key or os.getenv("GOOGLE_API_KEY")
    if not key:
        pytest.skip(f"{provider} credential unavailable")
    defaults = {
        "gemini": "gemini-3.6-flash",
        "openai": "gpt-4.1",
        "anthropic": "claude-sonnet-4-6",
    }
    return ModelIOFactoryRegistry().create(
        provider=provider,
        model=os.getenv(f"SMOKE_{provider.upper()}_MODEL", defaults[provider]),
        api_key=key,
    )


def test_live_streamed_text(model_io):
    events = []
    result = model_io.fetch_turn(
        ModelTurnRequest(
            messages=[{"role": "user", "content": "Reply with the word ready."}],
            emit_stream=True,
            callback=events.append,
        )
    )
    assert result.final_text.strip()
    assert any(
        event.get("type") == "token_delta" and event.get("delta") for event in events
    )


def test_live_structured_output(model_io):
    result = model_io.fetch_turn(
        ModelTurnRequest(
            messages=[
                {"role": "user", "content": "Return an object with ok equal to true."}
            ],
            response_format=ResponseFormat(
                "smoke",
                {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            ),
        )
    )
    assert json.loads(result.final_text) == {"ok": True}


def test_live_core_read_roundtrip(model_io, tmp_path):
    path = tmp_path / "smoke.txt"
    marker = "PUPU_GEMINI_SMOKE_163"
    path.write_text(marker)
    core = CoreToolkit(workspace_root=str(tmp_path))
    toolkit = Toolkit(tools={"read": core.tools["read"]})
    events = []
    result = KernelLoop(model_io=model_io, harnesses=[ToolExecutionHarness()]).run(
        [
            {
                "role": "user",
                "content": f"Use the read tool to read {path}, then reply with its exact content.",
            }
        ],
        toolkit=toolkit,
        provider=model_io.provider,
        model=model_io.model,
        callback=events.append,
        max_iterations=4,
    )
    assert result.status == "completed"
    assert any(event.get("type") == "tool_result" for event in events)
    assert marker in str(result.messages)


def test_live_repeated_long_context_cache(model_io):
    # Stable prefix exceeds Gemini Pro's 4096-token implicit-cache minimum.
    prefix = "\n".join(
        f"Reference item {n}: amber birch cedar delta evergreen forest."
        for n in range(2000)
    )
    request = ModelTurnRequest(
        messages=[
            {"role": "system", "content": prefix},
            {"role": "user", "content": "Reply only with ready."},
        ]
    )
    model_io.fetch_turn(request)
    repeated = [model_io.fetch_turn(request) for _ in range(2)]
    assert any(
        result.cache_read_input_tokens > 0 for result in repeated
    ), "No implicit cache hit observed; qualification remains incomplete"
