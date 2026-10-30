from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

from google.genai import types

from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.code_runtime import FileCodeSourceResolver, PythonEnvironmentProfile, SkeletonCodeRuntime
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.skeleton.kernel_runtime import SkeletonKernelHandler, build_node_bundle


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def _kernel(*, provider, protocol, model, endpoint, credential_ref, prompt, output_tokens):
    return {
        "schema_version": 1,
        "profile": {
            "id": provider + ".profile",
            "provider": provider,
            "model": model,
            "protocol": protocol,
            "endpoint": endpoint,
            "credential_ref": credential_ref,
            "context_window": 8192,
            "output_limit": 1024,
            "capabilities": {"json_schema": True, "images": False, "files": False},
        },
        "prompt": prompt,
        "max_iterations": 2,
        "output_tokens": output_tokens,
        "history": "root",
        "overflow": "error",
        "json_schema": copy.deepcopy(_SCHEMA),
    }


def _agent(node_id, kernel, inputs=()):
    return {
        "id": node_id,
        "type": "agent",
        "config": {
            "execution": {
                "inputs": list(inputs),
                "outputs": [{"name": "answer", "source": {"kind": "result", "path": ["answer"]}}],
            },
            "kernel": kernel,
        },
    }


def _binding(name, node_id):
    return {
        "name": name,
        "value": {"kind": "ref", "node_id": node_id, "path": ["answer"]},
        "missing": "error",
    }


def _document():
    alpha = _kernel(
        provider="openai", protocol="openai-responses", model="openai-model",
        endpoint="https://openai.example/v1", credential_ref="openai-credential",
        prompt="PROMPT_ALPHA_ONLY", output_tokens=101,
    )
    beta = _kernel(
        provider="anthropic", protocol="anthropic", model="anthropic-model",
        endpoint="https://anthropic.example/v1", credential_ref="anthropic-credential",
        prompt="PROMPT_BETA_ONLY", output_tokens=102,
    )
    gamma = _kernel(
        provider="gemini", protocol="gemini", model="gemini-model",
        endpoint="https://gemini.example/v1", credential_ref="gemini-credential",
        prompt="PROMPT_GAMMA_ONLY", output_tokens=103,
    )
    return {
        "schema_version": 1,
        "kind": "agent",
        "id": "mixed-kernel-runtime",
        "name": "Mixed Kernel Runtime",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": []}}},
            _agent("alpha", alpha),
            _agent("beta", beta, (_binding("alpha", "alpha"),)),
            _agent("gamma", gamma, (_binding("beta", "beta"),)),
            {
                "id": "postprocess",
                "type": "code",
                "config": {
                    "execution": {
                        "inputs": [_binding("answer", "gamma")],
                        "outputs": [{"name": "final", "source": {"kind": "result", "path": ["final"]}}],
                    },
                    "code": {
                        "source": {"kind": "file", "path": "postprocess.py"},
                        "environment_profile_id": "mixed-code",
                        "timeout_ms": 5000,
                        "max_result_bytes": 1048576,
                        "max_log_bytes": 65536,
                        "result_schema": {
                            "type": "object",
                            "properties": {"final": {"type": "string"}},
                            "required": ["final"],
                            "additionalProperties": False,
                        },
                    },
                },
            },
            {
                "id": "end",
                "type": "end",
                "config": {
                    "execution": {
                        "inputs": [{
                            "name": "final",
                            "value": {"kind": "ref", "node_id": "postprocess", "path": ["final"]},
                            "missing": "error",
                        }],
                        "outputs": [{"name": "final", "source": {"kind": "input", "path": ["final"]}}],
                    }
                },
            },
        ],
        "edges": [
            {"id": "start-alpha", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "alpha", "target_port_id": "in"},
            {"id": "alpha-beta", "kind": "flow", "source_node_id": "alpha", "source_port_id": "out", "target_node_id": "beta", "target_port_id": "in"},
            {"id": "beta-gamma", "kind": "flow", "source_node_id": "beta", "source_port_id": "out", "target_node_id": "gamma", "target_port_id": "in"},
            {"id": "gamma-code", "kind": "flow", "source_node_id": "gamma", "source_port_id": "out", "target_node_id": "postprocess", "target_port_id": "in"},
            {"id": "code-end", "kind": "flow", "source_node_id": "postprocess", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


class _OpenAIStream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                id="openai-response",
                output=[{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": '{"answer":"openai-answer"}'}
                ]}],
                usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            ),
        )


class _AnthropicStream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage={"input_tokens": 8}))
        yield SimpleNamespace(type="content_block_start", index=0,
                              content_block=SimpleNamespace(type="text", text=""))
        yield SimpleNamespace(type="content_block_delta", index=0,
                              delta=SimpleNamespace(type="text_delta", text='{"answer":"anthropic-answer"}'))
        yield SimpleNamespace(type="content_block_stop", index=0)
        yield SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"),
                              usage=SimpleNamespace(output_tokens=4))
        yield SimpleNamespace(type="message_stop")


def test_mixed_kernel_nodes_use_native_profiles_and_complete_through_postprocess(tmp_path):
    calls = {"openai": [], "anthropic": [], "gemini": []}

    def openai_factory(**client):
        calls["openai"].append({"client": copy.deepcopy(client)})

        def create(**request):
            calls["openai"][-1]["request"] = copy.deepcopy(request)
            return _OpenAIStream()

        return SimpleNamespace(responses=SimpleNamespace(create=create))

    def anthropic_factory(**client):
        calls["anthropic"].append({"client": copy.deepcopy(client)})

        def stream(**request):
            calls["anthropic"][-1]["request"] = copy.deepcopy(request)
            return _AnthropicStream()

        return SimpleNamespace(messages=SimpleNamespace(stream=stream))

    def gemini_factory(**client):
        calls["gemini"].append({"client": copy.deepcopy(client)})

        def generate_content_stream(**request):
            calls["gemini"][-1]["request"] = {
                "model": request["model"],
                "contents": [item.model_dump(mode="json", exclude_none=True) for item in request["contents"]],
                "config": request["config"].model_dump(mode="json", exclude_none=True),
            }
            return iter([types.GenerateContentResponse.model_validate({
                "candidates": [{"content": {"role": "model", "parts": [
                    {"text": '{"answer":"gemini-answer"}'}
                ]}, "finish_reason": "STOP"}],
                "usage_metadata": {"prompt_token_count": 9, "candidates_token_count": 5},
            })])

        return SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=generate_content_stream),
            close=lambda: None,
        )

    document = _document()
    plan = prepare_agent_graph(document)
    disk = SQLiteContextV2Store(database_path=tmp_path / "state.sqlite", object_directory=tmp_path / "objects")
    compiler = SQLiteContextCompilerV2Store(context_store=disk)
    journal = disk.bind_execution("mixed-execution")
    attempt = AttemptRef(GenerationRef("mixed-execution", "mixed-generation"), "root")
    store = WorkflowExecutionStore(journal, plan, graph_id="mixed-graph", attempt=attempt)

    def bind_profile(config):
        profile = copy.deepcopy(config["profile"])
        factories = {
            "openai": openai_factory,
            "anthropic": anthropic_factory,
            "gemini": gemini_factory,
        }
        transport = {
            "openai-responses": "openai",
            "anthropic": "anthropic",
            "gemini": "gemini",
        }[profile["protocol"]]
        return bind_kernel_profile(
            config,
            profile_resolver=lambda profile_id: profile if profile_id == profile["id"] else None,
            credential_resolver=lambda credential_ref: "credential-for-" + credential_ref,
            client_factories={transport: factories[transport]},
        )

    def bundle(*, attempt, view, config):
        return build_node_bundle(
            journal=journal,
            compiler_store=compiler,
            view=view,
            config=config,
            parent_attempt_id="root",
            artifact_sanitizer=lambda content, _media: content,
            payload_sanitizer=lambda _kind, payload: payload,
            partial_attempt_sink=lambda *_args: None,
        )

    handler = SkeletonKernelHandler(store, bind_profile=bind_profile, bundle_builder=bundle)
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "postprocess.py").write_text(
        "def main(input, context):\n return {'final':'postprocessed:'+input['answer']}\n",
        encoding="utf-8",
    )
    code_handler = SkeletonCodeRuntime(
        plan,
        source_resolver=FileCodeSourceResolver(code_root),
        profile_resolver=lambda profile_id: (
            PythonEnvironmentProfile.venv("mixed-code", sys.executable)
            if profile_id == "mixed-code" else None
        ),
        staging_root=tmp_path / "code-staging",
    )
    runner = DurableWorkflowRunner(
        store,
        {
            "agent": handler,
            "code": code_handler,
        },
    )
    runner.create({})
    result = runner.run()

    assert result.status == "completed"
    assert result.outputs == {"final": "postprocessed:gemini-answer"}
    assert result.completed_node_ids == ("start", "alpha", "beta", "gamma", "postprocess", "end")
    assert {name: len(sent) for name, sent in calls.items()} == {"openai": 1, "anthropic": 1, "gemini": 1}

    assert calls["openai"][0]["client"] == {
        "api_key": "credential-for-openai-credential",
        "base_url": "https://openai.example/v1",
        "max_retries": 0,
    }
    assert calls["openai"][0]["request"]["model"] == "openai-model"
    assert calls["openai"][0]["request"]["max_output_tokens"] == 101
    assert calls["openai"][0]["request"]["text"]["format"]["strict"] is True

    assert calls["anthropic"][0]["client"]["api_key"] == "credential-for-anthropic-credential"
    assert calls["anthropic"][0]["client"]["base_url"] == "https://anthropic.example/v1"
    assert calls["anthropic"][0]["client"]["max_retries"] == 0
    assert calls["anthropic"][0]["request"]["model"] == "anthropic-model"
    assert calls["anthropic"][0]["request"]["max_tokens"] == 102
    assert calls["anthropic"][0]["request"]["output_config"]["format"]["type"] == "json_schema"

    assert calls["gemini"][0]["client"] == {
        "api_key": "credential-for-gemini-credential",
        "http_options": {
            "timeout": 120000,
            "base_url": "https://gemini.example/v1",
            "retry_options": {"attempts": 1},
        },
    }
    assert calls["gemini"][0]["request"]["model"] == "gemini-model"
    assert calls["gemini"][0]["request"]["config"]["max_output_tokens"] == 103
    assert calls["gemini"][0]["request"]["config"]["response_json_schema"] == _SCHEMA

    serialized = {provider: str(calls[provider][0]["request"]) for provider in calls}
    assert "PROMPT_ALPHA_ONLY" not in serialized["anthropic"]
    assert "PROMPT_ALPHA_ONLY" not in serialized["gemini"]
    assert "PROMPT_BETA_ONLY" not in serialized["gemini"]
    assert "openai-answer" not in serialized["gemini"]

    assert runner.run().outputs == result.outputs
    assert {name: len(sent) for name, sent in calls.items()} == {"openai": 1, "anthropic": 1, "gemini": 1}
