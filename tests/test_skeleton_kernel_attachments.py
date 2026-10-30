from __future__ import annotations

import base64
import copy
from types import SimpleNamespace

import pytest

from unchain.context import ArtifactService, HostResolvedAttachment
from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.skeleton.kernel_runtime import SkeletonKernelHandler, build_node_bundle


_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
_IMAGE_BYTES = b"native-image-only"
_IMAGE_DATA = base64.b64encode(_IMAGE_BYTES).decode("ascii")


def _kernel(*, provider, protocol, model, prompt, images):
    return {
        "schema_version": 1,
        "profile": {
            "id": f"{provider}.attachment-profile",
            "provider": provider,
            "model": model,
            "protocol": protocol,
            "endpoint": f"https://{provider}.example/v1",
            "credential_ref": f"{provider}-credential",
            "context_window": 8192,
            "output_limit": 1024,
            "capabilities": {"json_schema": True, "images": images, "files": False},
        },
        "prompt": prompt,
        "max_iterations": 1,
        "output_tokens": 128,
        "history": "root",
        "overflow": "error",
        "json_schema": copy.deepcopy(_SCHEMA),
    }


def _agent(node_id, kernel):
    return {
        "id": node_id,
        "type": "agent",
        "config": {
            "execution": {
                "inputs": [],
                "outputs": [{"name": "answer", "source": {"kind": "result", "path": ["answer"]}}],
            },
            "kernel": kernel,
        },
    }


def _document(*, alpha_images=True):
    return {
        "schema_version": 1,
        "kind": "agent",
        "id": "kernel-attachment-isolation",
        "name": "Kernel attachment isolation",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": []}}},
            _agent("alpha", _kernel(
                provider="openai", protocol="openai-responses", model="openai-image-model",
                prompt="ALPHA_PRIVATE_PROMPT", images=alpha_images,
            )),
            _agent("beta", _kernel(
                provider="anthropic", protocol="anthropic", model="anthropic-image-model",
                prompt="BETA_PRIVATE_PROMPT", images=True,
            )),
            {
                "id": "end",
                "type": "end",
                "config": {"execution": {"inputs": [{
                    "name": "answer",
                    "value": {"kind": "ref", "node_id": "beta", "path": ["answer"]},
                    "missing": "error",
                }], "outputs": [{"name": "answer", "source": {"kind": "input", "path": ["answer"]}}]}},
            },
        ],
        "edges": [
            {"id": "start-alpha", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "alpha", "target_port_id": "in"},
            {"id": "alpha-beta", "kind": "flow", "source_node_id": "alpha", "source_port_id": "out", "target_node_id": "beta", "target_port_id": "in"},
            {"id": "beta-end", "kind": "flow", "source_node_id": "beta", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
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
                id="attachment-openai-response",
                output=[{"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": '{"answer":"openai-private-answer"}'}
                ]}],
                usage={"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
            ),
        )


class _AnthropicStream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="message_start", message=SimpleNamespace(usage={"input_tokens": 8}))
        yield SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text", text=""))
        yield SimpleNamespace(type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text='{"answer":"anthropic-private-answer"}'))
        yield SimpleNamespace(type="content_block_stop", index=0)
        yield SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=SimpleNamespace(output_tokens=4))
        yield SimpleNamespace(type="message_stop")


def _runtime(tmp_path, *, alpha_images=True):
    calls = {"openai": [], "anthropic": []}

    def openai_factory(**client):
        def create(**request):
            calls["openai"].append({"client": copy.deepcopy(client), "request": copy.deepcopy(request)})
            return _OpenAIStream()
        return SimpleNamespace(responses=SimpleNamespace(create=create))

    def anthropic_factory(**client):
        def stream(**request):
            calls["anthropic"].append({"client": copy.deepcopy(client), "request": copy.deepcopy(request)})
            return _AnthropicStream()
        return SimpleNamespace(messages=SimpleNamespace(stream=stream))

    document = _document(alpha_images=alpha_images)
    plan = prepare_agent_graph(document)
    disk = SQLiteContextV2Store(
        database_path=tmp_path / "state.sqlite",
        object_directory=tmp_path / "objects",
    )
    journal = disk.bind_execution("attachment-execution")
    compiler = SQLiteContextCompilerV2Store(context_store=disk)
    attempt = AttemptRef(GenerationRef("attachment-execution", "attachment-generation"), "root")
    store = WorkflowExecutionStore(journal, plan, graph_id="attachment-graph", attempt=attempt)

    def bind_profile(config):
        profile = copy.deepcopy(config["profile"])
        transport = {"openai-responses": "openai", "anthropic": "anthropic"}[profile["protocol"]]
        return bind_kernel_profile(
            config,
            profile_resolver=lambda profile_id: profile if profile_id == profile["id"] else None,
            credential_resolver=lambda credential_ref: f"credential-for-{credential_ref}",
            client_factories={transport: {"openai": openai_factory, "anthropic": anthropic_factory}[transport]},
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

    artifacts = ArtifactService(journal, sanitizer=lambda content, _media: content)
    artifact = artifacts.persist(
        _IMAGE_BYTES,
        media_type="image/png",
        operation_id="attachment-fixture-image",
    )
    attachment = HostResolvedAttachment(
        artifact=artifact,
        kind="image",
        name="node-alpha.png",
        media_type="image/png",
    )
    grants = []

    def attachments_for_node(context, resolved):
        grants.append((context.node_id, copy.deepcopy(resolved)))
        return (attachment,) if context.node_id in {"alpha", "beta"} else ()

    handler = SkeletonKernelHandler(
        store,
        bind_profile=bind_profile,
        bundle_builder=bundle,
        attachments_for_node=attachments_for_node,
    )
    return DurableWorkflowRunner(store, {"agent": handler}), journal, calls, grants


def test_node_attachment_is_durable_native_openai_and_isolated_from_anthropic_node(tmp_path):
    runner, journal, calls, grants = _runtime(tmp_path)
    runner.create({})
    result = runner.run()

    assert result.status == "completed"
    assert result.outputs == {"answer": "anthropic-private-answer"}
    assert [node_id for node_id, _resolved in grants] == ["alpha", "beta"]
    assert {provider: len(events) for provider, events in calls.items()} == {"openai": 1, "anthropic": 1}

    openai_wire = calls["openai"][0]["request"]
    openai_serialized = str(openai_wire)
    assert "input_image" in openai_serialized
    assert f"data:image/png;base64,{_IMAGE_DATA}" in openai_serialized
    assert openai_wire["model"] == "openai-image-model"

    anthropic_wire = calls["anthropic"][0]["request"]
    anthropic_serialized = str(anthropic_wire)
    assert anthropic_wire["model"] == "anthropic-image-model"
    assert "'type': 'image'" in anthropic_serialized
    assert "'media_type': 'image/png'" in anthropic_serialized
    assert _IMAGE_DATA in anthropic_serialized
    assert "openai-private-answer" not in anthropic_serialized
    assert "ALPHA_PRIVATE_PROMPT" not in anthropic_serialized

    journal_messages = [event for event in journal.capture_snapshot().events if event.event_type == "message.user"]
    alpha_message = next(event for event in journal_messages if event.payload["message"].get("attachments"))
    beta_message = next(event for event in journal_messages if event is not alpha_message)
    assert alpha_message.payload["message"]["attachments"][0]["media_type"] == "image/png"
    assert beta_message.payload["message"]["attachments"][0]["media_type"] == "image/png"


def test_unsupported_node_attachment_never_reaches_native_provider(tmp_path):
    runner, journal, calls, grants = _runtime(tmp_path, alpha_images=False)
    runner.create({})

    result = runner.run()

    assert result.status == "failed"
    assert [node_id for node_id, _resolved in grants] == ["alpha"]
    assert calls == {"openai": [], "anthropic": []}
    event_types = [event.event_type for event in journal.capture_snapshot().events]
    assert "context.current_input" not in event_types
