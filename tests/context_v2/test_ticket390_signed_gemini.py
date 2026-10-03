"""Signed Gemini mixed-content regression on the complete durable runtime."""

from types import SimpleNamespace

import httpx
from google.genai import types
from google.genai.errors import ClientError

from test_context_provider_turn_cross_provider import _runtime
from unchain.execution import ExecutionRuntime
from unchain.kernel.loop import KernelLoop
from unchain.memory import InMemorySessionStore
from unchain.providers import GeminiModelIO
from unchain.tools import Toolkit


def _run_signed_gemini_mixed_tool_turn(
    tmp_path,
    *,
    runtime=None,
    runtime_factory=None,
    session_store=None,
    session_id="signed-gemini-execution",
    send_calls=None,
    tool_calls=None,
    transient_second_send=False,
    retry_config=None,
):
    runtime = runtime or (runtime_factory or _runtime)(tmp_path)
    sends = send_calls if send_calls is not None else []
    tools = tool_calls if tool_calls is not None else []
    responses = [
        {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "Inspect the workspace first."},
            {"function_call": {"id": "gemini-signed-call", "name": "probe",
                               "args": {"query": "durable"}},
             "thought_signature": b"ticket390-synthetic-signature"},
        ]}, "finish_reason": "STOP"}]},
        {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "complete"},
        ]}, "finish_reason": "STOP"}]},
    ]

    def client_factory(**_kwargs):
        def stream(**kwargs):
            sends.append(kwargs)
            if transient_second_send and len(sends) == 2:
                raise ClientError(
                    429,
                    {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
                    response=httpx.Response(429),
                )
            return iter([types.GenerateContentResponse.model_validate(responses.pop(0))])
        return SimpleNamespace(models=SimpleNamespace(generate_content_stream=stream),
                               close=lambda: None)

    model = GeminiModelIO(model="gemini-ticket390-signed", api_key="test-key",
                          client_factory=client_factory, default_payloads={},
                          model_capabilities={})
    toolkit = Toolkit()

    def probe(query: str = ""):
        tools.append(query)
        return {"query": query, "status": "complete"}

    toolkit.register(probe, name="probe")
    loop = KernelLoop(model_io=model, harnesses=list(runtime.build_harnesses()),
                      retry_config=retry_config,
                      execution_runtime=ExecutionRuntime(
                          session_store or InMemorySessionStore()
                      ))
    result = loop.run(
        messages=[{"role": "user", "content": "call probe"}],
        callback=runtime.compose_event_callback(None),
        session_id=session_id, provider="gemini", model=model.model,
        toolkit=toolkit, run_id="signed-gemini-attempt", max_iterations=2,
    )
    return result, sends, tools


def test_signed_gemini_mixed_turn_retains_visible_text(tmp_path, *, runtime_factory=None):
    result, sends, tools = _run_signed_gemini_mixed_tool_turn(
        tmp_path,
        runtime_factory=runtime_factory,
    )
    assert result.status == "completed"
    assert tools == ["durable"]
    assert len(sends) == 2
    replayed = next(content for content in sends[1]["contents"]
                    if content.role == "model" and any(part.function_call for part in content.parts))
    assert replayed.parts[0].text == "Inspect the workspace first."
    assert replayed.parts[1].thought_signature == b"ticket390-synthetic-signature"
