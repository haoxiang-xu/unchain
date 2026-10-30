"""Skeleton Kernel context compression stays inside the durable node boundary."""
from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

from unchain.journal import AttemptRef, GenerationRef, SemanticEventDraft
from unchain.persistence import SQLiteContextV2Store
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.durable_store import WorkflowExecutionStore
from unchain.skeleton.kernel_profile import bind_kernel_profile
from unchain.skeleton.kernel_runtime import SkeletonKernelHandler, build_node_bundle


_EXECUTION = "compression-execution"
_GENERATION = "compression-generation"
_ROOT = AttemptRef(GenerationRef(_EXECUTION, _GENERATION), "root")
_SIBLING = AttemptRef(GenerationRef(_EXECUTION, _GENERATION), "sibling")
_SENTINEL = "SIBLING_PRIVATE_SENTINEL"
_SUMMARY = "root decisions and unresolved work"


def _document(*, overflow: str) -> dict:
    kernel = {
        "schema_version": 1,
        "profile": {
            "id": "compression.openai",
            "provider": "openai",
            "model": "compression-test-model",
            "protocol": "openai-responses",
            "endpoint": "https://example.test/v1",
            "credential_ref": "compression-key",
            "context_window": 4096,
            "output_limit": 128,
            "capabilities": {"json_schema": True, "images": False, "files": False},
        },
        "prompt": "Answer from the supplied context.",
        "max_iterations": 1,
        "output_tokens": 64,
        "history": "root",
        "overflow": overflow,
        "json_schema": None,
    }
    return {
        "schema_version": 1,
        "kind": "agent",
        "id": "compression-kernel",
        "name": "Compression Kernel",
        "nodes": [
            {"id": "start", "type": "start", "config": {"execution": {"outputs": []}}},
            {
                "id": "agent",
                "type": "agent",
                "config": {"execution": {
                    "inputs": [],
                    "outputs": [{"name": "answer", "source": {"kind": "result", "path": ["message"]}}],
                }, "kernel": kernel},
            },
            {
                "id": "end",
                "type": "end",
                "config": {"execution": {
                    "inputs": [{
                        "name": "answer",
                        "value": {"kind": "ref", "node_id": "agent", "path": ["answer"]},
                        "missing": "error",
                    }],
                    "outputs": [{"name": "answer", "source": {"kind": "input", "path": ["answer"]}}],
                }},
            },
        ],
        "edges": [
            {"id": "start-agent", "kind": "flow", "source_node_id": "start", "source_port_id": "out", "target_node_id": "agent", "target_port_id": "in"},
            {"id": "agent-end", "kind": "flow", "source_node_id": "agent", "source_port_id": "out", "target_node_id": "end", "target_port_id": "in"},
        ],
    }


class _OpenAIStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.completed", response=self.response)


def _response(identifier: str, text: str):
    return SimpleNamespace(
        id=identifier,
        output=[{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        usage={"input_tokens": 8, "output_tokens": 3, "total_tokens": 11},
    )


class _StrictOpenAIClient:
    def __init__(self):
        self.requests = []

        def create(**request):
            copied = deepcopy(request)
            self.requests.append(copied)
            encoded = json.dumps(copied, ensure_ascii=False, sort_keys=True)
            if "Summarize the supplied conversation records" in encoded:
                return _OpenAIStream(_response("summary", _SUMMARY))
            return _OpenAIStream(_response("main", "main answer"))

        self.responses = SimpleNamespace(create=create)


class _FailingOpenAIClient:
    def __init__(self):
        self.requests = []

        def create(**request):
            self.requests.append(deepcopy(request))
            raise RuntimeError("summary transport interrupted")

        self.responses = SimpleNamespace(create=create)


def _append(journal, event_id: str, event_type: str, attempt: AttemptRef, **payload):
    return journal.append(request=SemanticEventDraft(
        event_id=event_id,
        event_type=event_type,
        attempt=attempt,
        operation_id="operation-" + event_id,
        payload=payload,
    ).to_append_request()).event


def _seed_public_history(journal) -> tuple:
    large = "historical evidence " * 420
    first = _append(
        journal, "root-user", "message.user", _ROOT,
        run_id=_ROOT.attempt_id,
        message={"role": "user", "content": large},
    )
    second = _append(
        journal, "root-assistant", "message.assistant", _ROOT,
        run_id=_ROOT.attempt_id,
        message={"role": "assistant", "content": large},
    )
    _append(
        journal, "sibling-private", "message.user", _SIBLING,
        run_id=_SIBLING.attempt_id,
        parent_run_id=_ROOT.attempt_id,
        message={"role": "user", "content": _SENTINEL},
    )
    return first, second


def _runtime(tmp_path, *, overflow: str, client: _StrictOpenAIClient):
    disk = SQLiteContextV2Store(
        database_path=tmp_path / "journal.sqlite3",
        object_directory=tmp_path / "objects",
    )
    compiler = SQLiteContextCompilerV2Store(context_store=disk)
    journal = disk.bind_execution(_EXECUTION)
    history = _seed_public_history(journal)
    store = WorkflowExecutionStore(
        journal,
        prepare_agent_graph(_document(overflow=overflow)),
        graph_id="compression-graph",
        attempt=_ROOT,
    )

    def bind(config):
        profile = deepcopy(config["profile"])
        return bind_kernel_profile(
            config,
            profile_resolver=lambda profile_id: profile if profile_id == profile["id"] else None,
            credential_resolver=lambda credential_ref: "test-key" if credential_ref == "compression-key" else None,
            client_factories={"openai": lambda **_kwargs: client},
        )

    def bundle(*, attempt, view, config, **kwargs):
        return build_node_bundle(
            journal=journal,
            compiler_store=compiler,
            view=view,
            config=config,
            parent_attempt_id=_ROOT.attempt_id,
            artifact_sanitizer=lambda content, _media: content,
            payload_sanitizer=lambda _kind, payload: payload,
            partial_attempt_sink=lambda *_args: None,
            **kwargs,
        )

    handler = SkeletonKernelHandler(
        store,
        bind_profile=bind,
        bundle_builder=bundle,
        tools_for_node=lambda _context: (),
    )
    return store, handler, history


def test_explicit_compression_uses_private_summary_then_main_kernel_wire(tmp_path) -> None:
    client = _StrictOpenAIClient()
    store, handler, history = _runtime(tmp_path, overflow="compress", client=client)
    original = [(event.event_id, dict(event.payload)) for event in history]

    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"current": "request"})
    result = runner.run()

    assert result.status == "completed"
    assert result.outputs == {"answer": "main answer"}
    summary_wires = [
        request for request in client.requests
        if "Summarize the supplied conversation records" in json.dumps(request)
    ]
    main_wires = [request for request in client.requests if request not in summary_wires]
    assert len(summary_wires) >= 2
    assert len(main_wires) == 1
    main_wire = json.dumps(main_wires[0])
    assert _SUMMARY in main_wire
    # The governing guidance legitimately says "historical evidence". Check
    # the actual old message bodies, rather than that shared phrase.
    assert all(event.payload["message"]["content"] not in main_wire for event in history)
    assert all(_SENTINEL not in json.dumps(request) for request in client.requests)
    assert all(request["model"] == "compression-test-model" for request in client.requests)
    assert all(request["max_output_tokens"] == 128 for request in summary_wires)
    assert all("tools" not in request for request in summary_wires)
    assert all("format" not in request.get("text", {}) for request in summary_wires)

    serialized_source = json.dumps(
        [dict(dict(event.payload)["message"]) for event in history],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    records = [
        json.loads(request["input"][1]["content"])["records"]
        for request in summary_wires
    ]
    prefix = ""
    source_chunks = []
    for record in records:
        if len(prefix) < len(serialized_source):
            source_chunks.append(record)
            prefix += record
    assert prefix == serialized_source
    assert len(source_chunks) < len(records)  # a fold call follows the source chunks

    events = store.journal.capture_snapshot().events
    assert [(event.event_id, dict(event.payload)) for event in events[:2]] == original
    compression = [event for event in events if event.event_type.startswith("context.compression_")]
    assert [event.event_type for event in compression] == [
        "context.compression_intent", "context.compression_result",
    ]
    assert all(event.attempt != _ROOT for event in compression)
    assert all(event.attempt.attempt_id != _SIBLING.attempt_id for event in compression)
    private_summary_attempts = [
        event for event in events
        if event.event_type == "skeleton.kernel_result.saved"
        and event.attempt.attempt_id.startswith("compression-attempt.")
    ]
    assert len(private_summary_attempts) == len(summary_wires)
    assert all(event.payload.get("parent_run_id") is None for event in private_summary_attempts)


def test_hook_raw_seal_is_excluded_from_real_compression_requests(tmp_path) -> None:
    client = _StrictOpenAIClient()
    store, handler, _history = _runtime(tmp_path, overflow="compress", client=client)
    # The paired PuPu host test proves the producer writes this indexed event
    # after a real Tool effect. This tests the compiler/compressor consumer gate.
    _append(store.journal, "hook-raw-private", "tool.hook_raw.sealed", _ROOT,
            call_id="tool-call", private="HOOK_RAW_SENTINEL")

    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"current": "request"})
    result = runner.run()

    summary_wires = [request for request in client.requests
                     if "Summarize the supplied conversation records" in json.dumps(request)]
    assert result.status == "completed"
    assert summary_wires
    assert all("HOOK_RAW_SENTINEL" not in json.dumps(request) for request in summary_wires)
    assert all("HOOK_RAW_SENTINEL" not in json.dumps(request) for request in client.requests)


def test_default_overflow_never_calls_the_summary_worker(tmp_path) -> None:
    client = _StrictOpenAIClient()
    store, handler, _history = _runtime(tmp_path, overflow="error", client=client)

    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"current": "request"})
    result = runner.run()

    assert result.status == "completed"
    assert result.outputs == {"answer": "main answer"}
    assert len(client.requests) == 1
    assert "Summarize the supplied conversation records" not in json.dumps(client.requests[0])
    assert not [
        event for event in store.journal.capture_snapshot().events
        if event.event_type.startswith("context.compression_")
    ]


def test_cold_recovery_never_retries_an_unfinished_compression_intent(tmp_path) -> None:
    failing_client = _FailingOpenAIClient()
    store, handler, _history = _runtime(
        tmp_path,
        overflow="compress",
        client=failing_client,
    )
    runner = DurableWorkflowRunner(store, {"agent": handler})
    runner.create({"current": "request"})
    assert runner.run().status == "outcome_unknown"
    assert len(failing_client.requests) == 1
    assert [
        event.event_type for event in store.journal.capture_snapshot().events
        if event.event_type.startswith("context.compression_")
    ] == ["context.compression_intent"]

    cold_client = _StrictOpenAIClient()
    cold_store, cold_handler, _history = _runtime(
        tmp_path,
        overflow="compress",
        client=cold_client,
    )
    cold_runner = DurableWorkflowRunner(cold_store, {"agent": cold_handler})
    assert cold_runner.recover().status == "outcome_unknown"
    assert cold_runner.run().status == "outcome_unknown"
    assert cold_client.requests == []
