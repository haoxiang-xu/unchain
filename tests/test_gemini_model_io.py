from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from google.genai import types
from unchain.providers import GeminiModelIO, ModelTurnRequest, get_model_adapter_class
from unchain.providers.gemini_schema import sanitize_gemini_schema
from unchain.schemas import ResponseFormat
from unchain.tools import Tool, Toolkit


def make_io(chunks, captured=None):
    captured = captured if captured is not None else []

    def factory(**kwargs):
        assert kwargs["http_options"]["retry_options"]["attempts"] == 1

        def send(**request):
            assert set(request) == {"model", "contents", "config"}
            assert isinstance(request["config"], types.GenerateContentConfig)
            assert request["config"].automatic_function_calling.disable is True
            captured.append(request)
            return iter(types.GenerateContentResponse.model_validate(c) for c in chunks)

        return SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=send), close=lambda: None
        )

    return GeminiModelIO(
        model="gemini-2.5-flash", api_key="test-key", client_factory=factory
    )


def chunk(parts, usage=None):
    return {
        "candidates": [
            {"content": {"role": "model", "parts": parts}, "finish_reason": "STOP"}
        ],
        **({"usage_metadata": usage} if usage else {}),
    }


def test_stream_usage_and_structured_output():
    events, captured = [], []
    io = make_io(
        [
            chunk([{"text": '{"ok":'}]),
            chunk(
                [{"text": "true}"}],
                {
                    "prompt_token_count": 100,
                    "cached_content_token_count": 80,
                    "candidates_token_count": 4,
                    "thoughts_token_count": 3,
                    "total_token_count": 107,
                },
            ),
        ],
        captured,
    )
    result = io.fetch_turn(
        ModelTurnRequest(
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "hello"},
            ],
            emit_stream=True,
            callback=events.append,
            response_format=ResponseFormat(
                "answer",
                {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                    "additionalProperties": False,
                },
            ),
        )
    )
    assert result.final_text == '{"ok":true}'
    assert [e["delta"] for e in events if e["type"] == "token_delta"] == [
        '{"ok":',
        "true}",
    ]
    assert (
        result.input_tokens,
        result.output_tokens,
        result.consumed_tokens,
        result.cache_read_input_tokens,
    ) == (100, 7, 107, 80)
    assert result.provider_call_usage.input_uncached_tokens == 20
    assert captured[0]["config"].system_instruction == "Be helpful"
    assert captured[0]["config"].response_mime_type == "application/json"
    json.dumps(result.provider_replay_frame)


def test_native_tool_response_and_thought_signature_replay():
    from unchain.tools.messages import GeminiMessageBuilder
    from unchain.providers.context_assembler import _segments_for, _validate_tool_pairs

    io = make_io(
        [
            chunk(
                [
                    {
                        "function_call": {"name": "read", "args": {"path": "a.txt"}},
                        "thought_signature": "c2ln",
                    }
                ]
            )
        ]
    )
    result = io.fetch_turn(
        ModelTurnRequest(messages=[{"role": "user", "content": "read"}])
    )
    call = result.tool_calls[0]
    raw = result.provider_replay_frame["items"][-1]
    assert raw["parts"][0]["thought_signature"] == "c2ln"
    assert raw["parts"][0]["function_call"]["id"] == call.call_id
    response = GeminiMessageBuilder().build_tool_result_message(
        tool_call=call, tool_result={"text": "file content"}
    )
    captured = []
    make_io([chunk([{"text": "done"}])], captured).fetch_turn(
        ModelTurnRequest(messages=[*result.provider_replay_frame["items"], response])
    )
    assert captured[0]["contents"][1].parts[0].thought_signature == b"sig"
    assert captured[0]["contents"][2].parts[0].function_response.id == call.call_id
    _validate_tool_pairs("gemini", [raw, response])
    assert _segments_for("gemini.contents.v1", [raw])[0].requires_replay
    with pytest.raises(ValueError):
        bad = copy.deepcopy(response)
        bad["parts"][0]["function_response"]["id"] = "wrong"
        _validate_tool_pairs("gemini", [raw, bad])


def test_schema_projection_and_other_providers_unchanged():
    def read(path: str):
        return {"path": path}

    tool = Tool(name="read", description="Read", func=read)
    before = {p: tool.to_provider_json(p) for p in ("openai", "anthropic", "ollama")}
    native = tool.to_provider_json("gemini")
    assert set(native) == {"name", "description", "parameters"}
    assert "additionalProperties" not in native["parameters"]
    types.FunctionDeclaration.model_validate(native)
    assert {p: tool.to_provider_json(p) for p in before} == before
    schema = {
        "$defs": {"x": {"type": "string", "format": "uri"}},
        "type": "object",
        "properties": {"x": {"anyOf": [{"$ref": "#/$defs/x"}, {"type": "null"}]}},
        "additionalProperties": False,
    }
    assert sanitize_gemini_schema(schema)["properties"]["x"] == {
        "type": "string",
        "nullable": True,
    }
    for bad in [
        {"anyOf": [{"type": "string"}, {"type": "integer"}]},
        {"$ref": "#/missing"},
        {"$defs": {"x": {"$ref": "#/$defs/x"}}, "$ref": "#/$defs/x"},
    ]:
        with pytest.raises(ValueError):
            sanitize_gemini_schema(bad)


def test_media_and_unknown_field_fail_before_send():
    captured = []
    io = make_io([chunk([{"text": "image"}])], captured)
    io.fetch_turn(
        ModelTurnRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aGk=",
                            },
                        }
                    ],
                }
            ]
        )
    )
    assert captured[0]["contents"][0].parts[0].inline_data.data == b"hi"
    with pytest.raises(ValueError):
        io.fetch_turn(
            ModelTurnRequest(messages=[{"role": "user", "parts": [{"unknown": "x"}]}])
        )
    assert len(captured) == 1


def test_both_registries_and_provider_key_fallback(monkeypatch):
    from unchain.agent.model_io import ModelIOFactoryRegistry

    monkeypatch.setenv("GOOGLE_API_KEY", "google-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_model_adapter_class("gemini") is GeminiModelIO
    assert isinstance(
        ModelIOFactoryRegistry().create(
            provider="gemini", model="gemini-2.5-flash", api_key=None
        ),
        GeminiModelIO,
    )


def test_every_builtin_tool_schema_validates(tmp_path):
    from unchain.toolkits import CoreToolkit
    from unchain.toolkits.builtin.plan.plan import PlanToolkit
    from unchain.toolkits.builtin.agent_reach.agent_reach import AgentReachToolkit

    for toolkit in (
        CoreToolkit(workspace_root=str(tmp_path)),
        PlanToolkit(workspace_root=str(tmp_path)),
        AgentReachToolkit(),
    ):
        for declaration in toolkit.to_provider_json("gemini"):
            assert set(declaration) == {"name", "description", "parameters"}
            types.FunctionDeclaration.model_validate(declaration)
            assert "additionalProperties" not in json.dumps(declaration)


def test_durable_exact_transport_and_cold_reopen(tmp_path):
    from unchain.context.provider_execution import (
        ContextProviderTurnExecutionService,
        official_provider_transport_target_sha256,
    )
    from unchain.journal import AttemptRef, GenerationRef
    from unchain.persistence import SQLiteContextV2Store
    from unchain.providers.durable_turn_runtime import DurableProviderTurnMode
    from unchain.retry import RetryConfig

    attempt = AttemptRef(
        GenerationRef("gemini-execution", "gemini-generation"), "gemini-attempt"
    )

    def service():
        store = SQLiteContextV2Store(
            database_path=tmp_path / "context.sqlite3",
            object_directory=tmp_path / "objects",
        )
        return ContextProviderTurnExecutionService(
            attempt=attempt,
            store=store.bind_execution("gemini-execution"),
            mode=DurableProviderTurnMode.ENFORCE,
            transport_target_sha256=official_provider_transport_target_sha256(),
        )

    from unchain.toolkits import CoreToolkit
    captured = []
    io = make_io([chunk([{"text": "durable"}])], captured)
    request = ModelTurnRequest(
        messages=[{"role": "user", "content": "hello"}],
        iteration=1,
        run_id=attempt.attempt_id,
        toolkit=CoreToolkit(workspace_root=str(tmp_path)),
        response_format=ResponseFormat(
            "answer", {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        ),
    )
    first = service().fetch_prepared(
        model_io=io, request=request, retry_config=RetryConfig(max_retries=0)
    )
    assert first.final_text == "durable"
    # Re-open the durable store; a completed identical turn must not send twice.
    second = service().fetch_prepared(
        model_io=io, request=request, retry_config=RetryConfig(max_retries=0)
    )
    assert second.final_text == "durable"
    assert len(captured) == 1


def test_kernel_tool_roundtrip_uses_real_core_read(tmp_path):
    from unchain.kernel import KernelLoop
    from unchain.toolkits import CoreToolkit

    path = tmp_path / "hello.txt"
    path.write_text("fixture text")
    responses = iter(
        [
            [
                chunk(
                    [
                        {
                            "function_call": {
                                "name": "read",
                                "args": {"path": str(path)},
                            },
                            "thought_signature": "c2ln",
                        }
                    ]
                )
            ],
            [chunk([{"text": "finished"}])],
        ]
    )
    sends = []

    def factory(**kwargs):
        def send(**request):
            sends.append(request)
            return iter(
                types.GenerateContentResponse.model_validate(c) for c in next(responses)
            )

        return SimpleNamespace(models=SimpleNamespace(generate_content_stream=send))

    io = GeminiModelIO(model="gemini-2.5-flash", api_key="test", client_factory=factory)
    from unchain.tools.execution import ToolExecutionHarness

    loop = KernelLoop(model_io=io, harnesses=[ToolExecutionHarness()])
    result = loop.run(
        [{"role": "user", "content": "read the file"}],
        provider="gemini",
        model=io.model,
        toolkit=CoreToolkit(workspace_root=str(tmp_path)),
    )
    assert result.status == "completed"
    assert len(sends) == 2
    assert sends[1]["contents"][1].parts[0].thought_signature == b"sig"
    assert "fixture text" in str(
        sends[1]["contents"][2].parts[0].function_response.response
    )
