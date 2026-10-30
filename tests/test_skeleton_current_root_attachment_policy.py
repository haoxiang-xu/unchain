from __future__ import annotations

import pytest

from unchain.context import ArtifactService, HostResolvedAttachment, HostResolvedCurrentInput
from unchain.context.coordinator import ContextCompileCoordinator
from unchain.context.compiler import JournalMessageProjectionError
from unchain.context.ingress import ContextInputIngress
from unchain.context.model_projection import ModelContextProjection
from unchain.context.node_memory_view import NodeMemoryView
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.journal import AttemptRef, DurableEventSink, SemanticEventDraft, journal_event_sha256
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store

from tests.context_v2.test_node_memory_view import _attempt, _context, _open_store, _runtime
from tests.test_skeleton_kernel_attachments import _runtime as _kernel_runtime


def _root_input(journal, attempt, content, attachments):
    artifacts = ArtifactService(journal, sanitizer=lambda content, _media: content)
    projector = CanonicalSemanticEventProjector(attempt=attempt, artifacts=artifacts,
        payload_sanitizer=lambda _kind, payload: payload)
    sink = DurableEventSink(journal, attempt, projector)
    ingress = ContextInputIngress(attempt=attempt, projector=projector, sink=sink)
    return ingress.persist(HostResolvedCurrentInput(
        attempt=attempt, content=content, message_index=0, attachments=attachments,
    )).event


def _pdf(journal, operation_id):
    artifacts = ArtifactService(journal, sanitizer=lambda content, _media: content)
    content = b"%PDF-1.7\ncurrent-root-policy\n%%EOF"
    artifact = artifacts.persist(content, media_type="application/pdf", operation_id=operation_id)
    return HostResolvedAttachment(artifact=artifact, kind="file", name="current.pdf", media_type="application/pdf")


def test_v1_wire_remains_exact_and_v2_policy_is_closed():
    node = _attempt("node")
    legacy = NodeMemoryView(node, "a" * 64)
    wire = {"schema": "unchain.node_memory_view.v1", "attempt": node.to_dict(),
            "definition_digest": "a" * 64, "history": []}
    assert legacy.to_dict() == wire
    assert NodeMemoryView.from_dict(wire) == legacy
    protected = NodeMemoryView(node, "a" * 64, current_root_attempt_id="root")
    assert protected.to_dict() == {**wire, "schema": "unchain.node_memory_view.v2", "current_root_attempt_id": "root"}
    assert NodeMemoryView.from_dict(protected.to_dict()) == protected
    assert protected.digest != legacy.digest


@pytest.mark.parametrize("root", [None, "", " root", "root ", 1, "node"])
def test_v2_rejects_missing_or_invalid_current_root(root):
    wire = NodeMemoryView(_attempt("node"), "a" * 64, current_root_attempt_id="root").to_dict()
    wire["current_root_attempt_id"] = root
    with pytest.raises(ValueError):
        NodeMemoryView.from_dict(wire)


@pytest.mark.parametrize("mutation", ["extra", "missing", "v1-policy", "unknown-version"])
def test_policy_decoder_rejects_nonclosed_shapes(mutation):
    wire = NodeMemoryView(_attempt("node"), "a" * 64, current_root_attempt_id="root").to_dict()
    if mutation == "extra":
        wire["attachments"] = []
    elif mutation == "missing":
        wire.pop("current_root_attempt_id")
    elif mutation == "v1-policy":
        wire["schema"] = "unchain.node_memory_view.v1"
    else:
        wire["schema"] = "unchain.node_memory_view.v3"
    with pytest.raises(ValueError):
        NodeMemoryView.from_dict(wire)


@pytest.mark.parametrize("current_text", ["current root plaintext", ""])
def test_effective_compilation_keeps_past_attachments_and_exact_original_receipts(tmp_path, current_text):
    disk = _open_store(tmp_path)
    journal = disk.bind_execution("execution-node-memory-view")
    previous = _attempt("previous-root")
    root = _attempt("current-root")
    node = _attempt("node")
    attachment = _pdf(journal, "policy-original-pdf")
    previous_input = _root_input(journal, previous, "past root plaintext", (attachment,))
    current_input = _root_input(journal, root, current_text, (attachment,))
    history = (previous_input, current_input)
    view = NodeMemoryView(node, "a" * 64,
        tuple((event.event_id, event.store_seq, journal_event_sha256(event)) for event in history),
        root.attempt_id)
    view.persist(journal)
    runtime = _runtime(disk, node, view, parent_run_id=root.attempt_id)
    runtime["ingress"].persist(HostResolvedCurrentInput(
        attempt=node, content="explicit node input", message_index=0,
    ))
    capabilities = SQLiteContextCompilerV2Store(context_store=disk).bind_execution(
        journal.execution_id, artifacts=runtime["artifacts"],
    )
    runtime["coordinator"] = ContextCompileCoordinator(journal=journal,
        checkpoint_repository=capabilities.checkpoints,
        build_repository=capabilities.context_builds,
        partial_attempt_sink=lambda *_args: None,
        model_projection=ModelContextProjection(runtime["artifacts"]),
        node_memory_view=view)
    before = journal.capture_snapshot()
    first = runtime["coordinator"].compile(runtime["request_factory"](_context(node)))
    messages = [dict(message) for message in first.messages]
    old_message = next(message for message in messages if isinstance(message["content"], (list, tuple)))
    assert old_message["content"][0] == {"type": "text", "text": "past root plaintext"}
    assert old_message["content"][1]["type"] == "pdf"
    assert sum(isinstance(message["content"], (list, tuple)) for message in messages) == 1
    assert any(message["content"] == "explicit node input" for message in messages)
    assert any(message["content"] == current_text for message in messages) is bool(current_text)
    after = journal.capture_snapshot()
    for original in history:
        assert after.events[original.store_seq - 1].to_dict() == original.to_dict()
        assert original.payload["message"]["attachments"]
    assert view.select(before)[:2] == history
    cold = NodeMemoryView.from_dict(view.to_dict())
    assert cold.select(after)[:2] == history
    replay = runtime["coordinator"].compile(runtime["request_factory"](_context(node)))
    assert replay.messages == first.messages
    assert replay.envelope == first.envelope


def test_actual_native_node_attachments_do_not_duplicate_current_root_upload(tmp_path):
    runner, journal, calls, _grants = _kernel_runtime(tmp_path)
    root = runner.store.attempt
    upload = _pdf(journal, "native-unbound-root-pdf")
    _root_input(journal, root, "root plaintext is still readable", (upload,))
    runner.create({})
    result = runner.run()
    assert result.status == "completed"
    assert len(calls["openai"]) == len(calls["anthropic"]) == 1
    openai_wire = calls["openai"][0]["request"]
    anthropic_wire = calls["anthropic"][0]["request"]
    assert "root plaintext is still readable" in str(openai_wire)
    assert "root plaintext is still readable" in str(anthropic_wire)
    assert "input_file" not in str(openai_wire)
    assert "application/pdf" not in str(anthropic_wire)
    assert str(openai_wire).count("'type': 'input_image'") == 1
    assert str(anthropic_wire).count("'type': 'image'") == 1
    for event in journal.capture_snapshot().events:
        if event.event_type == "context.node_memory_view":
            wire = event.to_dict()["payload"]
            assert wire["schema"] == "unchain.node_memory_view.v2"
            assert wire["current_root_attempt_id"] == root.attempt_id


@pytest.mark.parametrize("policy", [None, "foreign-root", "root"])
def test_cold_kernel_view_requires_exact_persisted_root_policy(tmp_path, policy):
    runner, journal, calls, _grants = _kernel_runtime(tmp_path)
    runner.create({})
    handler = runner.handlers["agent"]
    node = AttemptRef(runner.store.attempt.generation, "test-node-selector")
    saved = NodeMemoryView(node, handler.definition_digest, current_root_attempt_id=policy)
    saved.persist(journal)
    before = journal.capture_snapshot()
    if policy == "root":
        assert handler._view(node, handler.configs["alpha"]) == saved
    else:
        with pytest.raises(ValueError, match="node memory view definition changed"):
            handler._view(node, handler.configs["alpha"])
    assert journal.capture_snapshot().to_dict() == before.to_dict()
    assert calls == {"openai": [], "anthropic": []}


def test_effective_filter_cannot_hide_invalid_persisted_attachment_authority(tmp_path):
    disk = _open_store(tmp_path)
    journal = disk.bind_execution("execution-node-memory-view")
    root, node = _attempt("root"), _attempt("node")
    attachment = _pdf(journal, "malformed-policy-pdf")
    original = _root_input(journal, root, "current root", (attachment,))
    payload = original.to_dict()["payload"]
    payload["attachment_refs"] = []
    malformed = journal.append(request=SemanticEventDraft(
        event_id="malformed-attachment-root-input", event_type="message.user", attempt=root,
        operation_id="malformed-attachment-root-operation", payload=payload,
        resource_refs=original.resource_refs,
    ).to_append_request()).event
    view = NodeMemoryView(node, "a" * 64,
        ((malformed.event_id, malformed.store_seq, journal_event_sha256(malformed)),), root.attempt_id)
    view.persist(journal)
    runtime = _runtime(disk, node, view, parent_run_id=root.attempt_id)
    runtime["ingress"].persist(HostResolvedCurrentInput(attempt=node,
        content="valid explicitly bound node input", message_index=0))
    with pytest.raises(JournalMessageProjectionError) as rejected:
        runtime["coordinator"].compile(runtime["request_factory"](_context(node)))
    assert rejected.value.reason == "message_payload_invalid"


def test_uncreated_root_history_store_is_rejected_without_native_call(tmp_path):
    runner, journal, calls, _grants = _kernel_runtime(tmp_path)
    handler = runner.handlers["agent"]
    node = AttemptRef(runner.store.attempt.generation, "uncreated-node-selector")
    before = journal.capture_snapshot()
    with pytest.raises(ValueError, match="node memory view requires a created graph"):
        handler._view(node, handler.configs["alpha"])
    assert journal.capture_snapshot().to_dict() == before.to_dict()
    assert calls == {"openai": [], "anthropic": []}
