from __future__ import annotations

import json
from pathlib import Path

import pytest

from unchain.context import ArtifactService, ContextCompileCoordinator
from unchain.context.ingress import ContextInputIngress, HostResolvedCurrentInput
from unchain.context.node_memory_view import NodeMemoryView
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.context.request_factory import JournalContextRequestFactory
from unchain.context.checkpoints import CheckpointRequest
from unchain.journal import AttemptRef, DurableEventSink, GenerationRef, SemanticEventDraft
from unchain.journal.snapshot import journal_event_sha256
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


_EXECUTION = "execution-node-compression"
_GENERATION = "generation-node-compression"
_DEFINITION = "e" * 64
_PROFILE = "f" * 64
_ROOT_RAW = "ROOT_HISTORY_RAW_MUST_BE_COMPRESSED"
_SIBLING_RAW = "SIBLING_HISTORY_RAW_MUST_STAY_PRIVATE"
_SUMMARY = "AUTHORIZED_NODE_COMPRESSION_SUMMARY"


def _attempt(name: str) -> AttemptRef:
    return AttemptRef(GenerationRef(_EXECUTION, _GENERATION), name)


def _context(attempt: AttemptRef) -> HarnessContext:
    state = RunState()
    state.session_state.session_id = _EXECUTION
    state.provider_state.provider = "openai"
    state.provider_state.model = "gpt-node-compression"
    state.provider_state.max_context_window_tokens = 8_192
    state.seed_messages([{"role": "system", "content": "node instructions"}])
    return HarnessContext(
        state=state,
        phase="before_model",
        event={"run_id": attempt.attempt_id, "toolkit": None},
    )


def _append(journal, event_id: str, event_type: str, attempt: AttemptRef, **payload):
    return journal.append(
        request=SemanticEventDraft(
            event_id=event_id,
            event_type=event_type,
            attempt=attempt,
            operation_id="operation-" + event_id,
            payload=payload,
        ).to_append_request()
    ).event


def _components(journal, artifacts, attempt: AttemptRef, *, parent_run_id: str | None):
    def payload_sanitizer(_event_type, payload):
        result = dict(payload)
        if parent_run_id is not None:
            result["parent_run_id"] = parent_run_id
        return result

    projector = CanonicalSemanticEventProjector(
        attempt=attempt,
        artifacts=artifacts,
        payload_sanitizer=payload_sanitizer,
    )
    sink = DurableEventSink(journal, attempt, projector)
    return projector, sink, ContextInputIngress(
        attempt=attempt,
        projector=projector,
        sink=sink,
    )


def _fixture(tmp_path: Path):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects",
    )
    journal = store.bind_execution(_EXECUTION)
    artifacts = ArtifactService(journal, sanitizer=lambda content, _media: content)
    root = _attempt("root")
    node = _attempt("node")
    sibling = _attempt("sibling")

    root_projector, root_sink, root_ingress = _components(
        journal,
        artifacts,
        root,
        parent_run_id=None,
    )
    del root_projector
    root_user = root_ingress.persist(
        HostResolvedCurrentInput(
            attempt=root,
            content=_ROOT_RAW + (" x" * 18_000),
        )
    ).event
    root_assistant = _append(
        journal,
        "root-assistant",
        "message.assistant",
        root,
        run_id=root.attempt_id,
        message={"role": "assistant", "content": "root response" + (" y" * 4_000)},
    )
    view = NodeMemoryView(
        node,
        _DEFINITION,
        tuple(
            (event.event_id, event.store_seq, journal_event_sha256(event))
            for event in (root_user, root_assistant)
        ),
    )
    view.persist(journal)

    _node_projector, _node_sink, node_ingress = _components(
        journal,
        artifacts,
        node,
        parent_run_id=root.attempt_id,
    )
    _sibling_projector, _sibling_sink, sibling_ingress = _components(
        journal,
        artifacts,
        sibling,
        parent_run_id=root.attempt_id,
    )
    sibling_ingress.persist(
        HostResolvedCurrentInput(
            attempt=sibling,
            content=_SIBLING_RAW + (" z" * 16_000),
        )
    )
    node_ingress.persist(
        HostResolvedCurrentInput(attempt=node, content="current node input")
    )

    capabilities = SQLiteContextCompilerV2Store(context_store=store).bind_execution(
        _EXECUTION,
        artifacts=artifacts,
    )
    request_factory = JournalContextRequestFactory(
        attempt=node,
        journal=journal,
        model_window_fallback=lambda _provider, _model: 8_192,
        node_memory_view=view,
    )
    return {
        "store": store,
        "journal": journal,
        "artifacts": artifacts,
        "capabilities": capabilities,
        "root": root,
        "node": node,
        "sibling": sibling,
        "view": view,
        "request_factory": request_factory,
    }


def _service(*, journal, attempt: AttemptRef, calls: list[dict], profile: str = _PROFILE):
    # Delayed until the production service module is installed by this CP11
    # slice; no fake protocol substitutes for its durable receipts.
    from unchain.context.compression import ContextCompressionService

    def summarize(*, source_messages, target_tokens, operation_id):
        calls.append(
            {
                "source_messages": tuple(dict(message) for message in source_messages),
                "target_tokens": target_tokens,
                "operation_id": operation_id,
            }
        )
        return _SUMMARY

    return ContextCompressionService(
        journal=journal,
        attempt=attempt,
        profile_digest=profile,
        summarize=summarize,
    )


def _coordinator(fixture, *, compression_service=None):
    return ContextCompileCoordinator(
        journal=fixture["journal"],
        checkpoint_repository=fixture["capabilities"].checkpoints,
        build_repository=fixture["capabilities"].context_builds,
        partial_attempt_sink=lambda _request, _error: None,
        artifacts=fixture["artifacts"],
        node_memory_view=fixture["view"],
        compression_service=compression_service,
    )


def _durable_compression_proof(journal) -> tuple[CheckpointRequest, dict]:
    intent = next(
        event
        for event in journal.capture_snapshot().events
        if event.event_type == "context.compression_intent"
    )
    result = next(
        event
        for event in journal.capture_snapshot().events
        if event.event_type == "context.compression_result"
    )
    payload = dict(intent.payload)
    checkpoint = CheckpointRequest.from_dict(payload["checkpoint"])
    operation_id = result.event_id.removeprefix("event.").removesuffix(".result")
    return checkpoint, {
        "schema": "unchain.context_compression.v1",
        "operation_id": operation_id,
        "checkpoint_request_id": checkpoint.request_id,
        "profile_digest": payload["profile_digest"],
        "target_tokens": payload["target_limit"],
        "summary": result.payload["summary"],
    }


def test_node_compression_is_opt_in_and_default_context_never_calls_summary(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    calls: list[dict] = []
    unused_service = _service(
        journal=fixture["journal"],
        attempt=fixture["node"],
        calls=calls,
    )
    assert unused_service is not None

    result = _coordinator(fixture).compile(
        fixture["request_factory"](_context(fixture["node"]))
    )

    assert calls == []
    assert _SUMMARY not in json.dumps(result.to_dict()["messages"], sort_keys=True)


def test_node_compression_uses_only_authorized_history_and_reuses_cold_receipt(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    calls: list[dict] = []
    service = _service(
        journal=fixture["journal"],
        attempt=fixture["node"],
        calls=calls,
    )
    first = _coordinator(fixture, compression_service=service).compile(
        fixture["request_factory"](_context(fixture["node"]))
    )

    assert len(calls) == 1
    call = calls[0]
    assert 32 <= call["target_tokens"] <= 256
    assert call["operation_id"]
    source = json.dumps(call["source_messages"], ensure_ascii=False, sort_keys=True)
    assert _ROOT_RAW in source
    assert _SIBLING_RAW not in source
    messages = json.dumps(first.to_dict()["messages"], ensure_ascii=False, sort_keys=True)
    assert _SUMMARY in messages
    assert _ROOT_RAW not in messages
    assert _SIBLING_RAW not in messages

    reopened_store = SQLiteContextV2Store(
        database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects",
    )
    reopened_journal = reopened_store.bind_execution(_EXECUTION)
    reopened_artifacts = ArtifactService(
        reopened_journal,
        sanitizer=lambda content, _media: content,
    )
    reopened_capabilities = SQLiteContextCompilerV2Store(
        context_store=reopened_store
    ).bind_execution(_EXECUTION, artifacts=reopened_artifacts)
    cold_calls: list[dict] = []
    cold_service = _service(
        journal=reopened_journal,
        attempt=fixture["node"],
        calls=cold_calls,
    )
    cold_fixture = {
        **fixture,
        "store": reopened_store,
        "journal": reopened_journal,
        "artifacts": reopened_artifacts,
        "capabilities": reopened_capabilities,
        "view": NodeMemoryView.from_dict(fixture["view"].to_dict()),
    }
    cold_fixture["request_factory"] = JournalContextRequestFactory(
        attempt=cold_fixture["node"],
        journal=reopened_journal,
        model_window_fallback=lambda _provider, _model: 8_192,
        node_memory_view=cold_fixture["view"],
    )
    second = _coordinator(cold_fixture, compression_service=cold_service).compile(
        cold_fixture["request_factory"](_context(cold_fixture["node"]))
    )

    assert cold_calls == []
    assert second.envelope == first.envelope


def test_node_compression_refuses_receipt_reuse_when_profile_changes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    calls: list[dict] = []
    first_service = _service(
        journal=fixture["journal"],
        attempt=fixture["node"],
        calls=calls,
    )
    _coordinator(fixture, compression_service=first_service).compile(
        fixture["request_factory"](_context(fixture["node"]))
    )

    changed_calls: list[dict] = []
    changed_service = _service(
        journal=fixture["journal"],
        attempt=fixture["node"],
        calls=changed_calls,
        profile="a" * 64,
    )
    with pytest.raises(Exception):
        _coordinator(fixture, compression_service=changed_service).compile(
            fixture["request_factory"](_context(fixture["node"]))
        )
    assert changed_calls == []


def test_node_compression_rejects_tampered_source_and_checkpoint_proof(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    calls: list[dict] = []
    service = _service(
        journal=fixture["journal"],
        attempt=fixture["node"],
        calls=calls,
    )
    _coordinator(fixture, compression_service=service).compile(
        fixture["request_factory"](_context(fixture["node"]))
    )
    checkpoint, proof = _durable_compression_proof(fixture["journal"])

    with pytest.raises(Exception):
        service.compress(
            checkpoint,
            ({"role": "user", "content": "forged source"},),
            target_tokens=proof["target_tokens"],
        )
    with pytest.raises(Exception):
        service.source_snapshot(
            {**proof, "checkpoint_request_id": "forged-checkpoint"}
        )
    assert len(calls) == 1
