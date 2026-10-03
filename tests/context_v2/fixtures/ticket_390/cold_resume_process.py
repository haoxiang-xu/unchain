"""Run one half of the ticket 390 cold-resume fixture in a fresh process."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from test_context_provider_turn_cross_provider import _runtime

from unchain.execution import ExecutionRuntime
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.providers import (
    AnthropicModelIO,
    GeminiModelIO,
    HyperspaceModelIO,
)
from unchain.runtime import build_runtime_loop
from unchain.tools import Toolkit


def _model(*, phase: str, provider: str, sends: list[dict]):
    call_id = f"{provider}-process-cold-call"
    if provider == "gemini":
        from google.genai import types

        response = (
            {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "Inspect the durable workspace."},
                {"function_call": {"id": call_id, "name": "probe",
                                   "args": {"query": "process"}},
                 "thought_signature": b"process-cold-signature"},
            ]}, "finish_reason": "STOP"}]}
            if phase == "seed"
            else {"candidates": [{"content": {"role": "model", "parts": [
                {"text": "process cold resume complete"},
            ]}, "finish_reason": "STOP"}]}
        )

        class _Models:
            def generate_content_stream(self, **kwargs):
                sends.append(copy.deepcopy(kwargs))
                return iter([types.GenerateContentResponse.model_validate(response)])

        class _Client:
            models = _Models()

            def close(self):
                return None

        return GeminiModelIO(
            model=f"phase2-process-{provider}", api_key="test-key",
            client_factory=lambda **_kwargs: _Client(), default_payloads={},
            model_capabilities={},
        )

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            content = (
                [
                    {"type": "thinking", "thinking": "plan", "signature": "signed"},
                    {"type": "text", "text": "Inspect the durable workspace."},
                    {
                        "type": "tool_use", "id": call_id,
                        "name": "probe", "input": {"query": "process"},
                    },
                ]
                if phase == "seed"
                else [{"type": "text", "text": "process cold resume complete"}]
            )
            return SimpleNamespace(id=f"process-cold-{phase}", content=content)

    class _Messages:
        def stream(self, **kwargs):
            sends.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        messages = _Messages()

    adapter = AnthropicModelIO if provider == "anthropic" else HyperspaceModelIO
    return adapter(
        model=f"phase2-process-{provider}", api_key="test-key",
        client_factory=lambda **_kwargs: _Client(), default_payloads={},
        model_capabilities={},
    )


def _toolkit(invocations: list[str]):
    toolkit = Toolkit()

    def probe(query=""):
        invocations.append(query)
        return {"query": query, "status": "complete"}

    toolkit.register(probe, name="probe", requires_confirmation=True)
    return toolkit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("seed", "resume"))
    parser.add_argument("provider", choices=("anthropic", "hyperspace", "gemini"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    session_id = "phase2-process-cold"
    sends: list[dict] = []
    invocations: list[str] = []
    runtime = _runtime(args.root)
    store = JsonFileSessionStore(args.root / "sessions")
    loop = build_runtime_loop(
        harnesses=list(runtime.build_harnesses()),
        model_io=_model(phase=args.phase, provider=args.provider, sends=sends),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
        execution_runtime=ExecutionRuntime(store),
        semantic_context_owner=runtime.owner_id,
    )
    if args.phase == "seed":
        result = loop.run(
            messages=[{"role": "user", "content": "call the probe"}],
            callback=runtime.compose_event_callback(None), session_id=session_id,
            provider=args.provider, model=f"phase2-process-{args.provider}",
            toolkit=_toolkit(invocations), run_id="phase2-process-attempt",
            max_iterations=2,
        )
    elif args.phase == "resume":
        result = loop.resume_interaction(
            session_id=session_id, response={"approved": True},
            callback=runtime.compose_event_callback(None),
            toolkit=_toolkit(invocations),
        )
    second_message_result = None
    if args.phase == "resume":
        second_message_result = loop.run(
            messages=[{"role": "user", "content": "continue the same chat"}],
            callback=runtime.compose_event_callback(None), session_id=session_id,
            provider=args.provider, model=f"phase2-process-{args.provider}",
            toolkit=Toolkit(), run_id=f"phase2-process-{args.provider}-second-message",
            max_iterations=1,
        )
    replay = None
    if args.phase == "resume" and sends:
        replay = []
        if args.provider == "gemini":
            for content in sends[0].get("contents", []):
                if content.role != "model":
                    continue
                for part in content.parts:
                    call = part.function_call
                    if part.text or call:
                        replay.append({
                            "text": part.text,
                            "call_id": call.id if call else None,
                            "call_name": call.name if call else None,
                            "arguments": call.args if call else None,
                            "signature": part.thought_signature == b"process-cold-signature",
                        })
        else:
            for message in sends[0].get("messages", []):
                if message.get("role") == "assistant" and isinstance(message.get("content"), list):
                    replay.extend({
                        "type": block.get("type"),
                        "text": block.get("text"),
                        "call_id": block.get("id"),
                        "call_name": block.get("name"),
                        "arguments": block.get("input"),
                        "signature": block.get("signature") == "signed",
                    } for block in message["content"] if isinstance(block, dict))
    print(json.dumps({
        "status": result.status,
        "second_message_status": (
            second_message_result.status if second_message_result is not None else None
        ),
        "send_count": len(sends),
        "invocations": invocations,
        "replayed_assistant": replay,
        "second_message_has_source_native_call": (
            (
                any(part.function_call for content in sends[1].get("contents", [])
                    for part in content.parts)
                if args.provider == "gemini"
                else "tool_use" in repr(sends[1].get("messages", []))
            )
            if len(sends) > 1 else None
        ),
        "second_message_history": (
            repr(sends[1].get("contents" if args.provider == "gemini" else "messages"))
            if len(sends) > 1 else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
