from __future__ import annotations

from pathlib import Path

import pytest

from unchain.context import ArtifactService
from unchain.context.coordinator import ContextCompileCoordinator
from unchain.context.factory import ContextExecutionBundle, ContextExecutionBundleError
from unchain.context.handoff import DurableHandoffRecorder, HandoffService
from unchain.context.ingress import ContextInputIngress, HostResolvedCurrentInput
from unchain.context.node_memory_view import NodeMemoryView
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.context.request_factory import (
    JournalContextRequestFactory,
    JournalContextRequestFactoryError,
)
from unchain.context.tool_boundary import DurableToolBoundary
from unchain.journal import AttemptRef, DurableEventSink, GenerationRef, SemanticEventDraft
from unchain.journal.snapshot import journal_event_sha256
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


_EXECUTION = "execution-node-memory-view"
_GENERATION = "generation-node-memory-view"
_DEFINITION = "a" * 64


def _attempt(name: str) -> AttemptRef:
    return AttemptRef(GenerationRef(_EXECUTION, _GENERATION), name)


def _open_store(root: Path) -> SQLiteContextV2Store:
    return SQLiteContextV2Store(
        database_path=root / "context_v2.sqlite3",
        object_directory=root / "objects",
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


def _context(attempt: AttemptRef) -> HarnessContext:
    state = RunState()
    state.session_state.session_id = _EXECUTION
    state.provider_state.provider = "openai"
    state.provider_state.model = "synthetic"
    state.provider_state.max_context_window_tokens = 16_384
    state.seed_messages([{"role": "system", "content": "node system"}])
    return HarnessContext(
        state=state,
        phase="before_model",
        event={"run_id": attempt.attempt_id, "toolkit": None},
    )


def _runtime(
    store: SQLiteContextV2Store,
    attempt: AttemptRef,
    view: NodeMemoryView | None,
    *,
    parent_run_id: str | None = None,
):
    journal = store.bind_execution(_EXECUTION)
    artifacts = ArtifactService(journal, sanitizer=lambda content, media_type: content)
    capabilities = SQLiteContextCompilerV2Store(context_store=store).bind_execution(
        _EXECUTION,
        artifacts=artifacts,
    )
    def scoped_payload(event_type, payload):
        result = dict(payload)
        if parent_run_id is not None:
            if result.get("parent_run_id", parent_run_id) != parent_run_id:
                raise ValueError("child event changed its parent run")
            result["parent_run_id"] = parent_run_id
        return result

    projector = CanonicalSemanticEventProjector(
        attempt=attempt,
        artifacts=artifacts,
        payload_sanitizer=scoped_payload,
    )
    sink = DurableEventSink(journal, attempt, projector)
    ingress = ContextInputIngress(attempt=attempt, projector=projector, sink=sink)
    request_factory = JournalContextRequestFactory(
        attempt=attempt,
        journal=journal,
        model_window_fallback=lambda provider, model: 16_384,
        node_memory_view=view,
    )
    coordinator = ContextCompileCoordinator(
        journal=journal,
        checkpoint_repository=capabilities.checkpoints,
        build_repository=capabilities.context_builds,
        partial_attempt_sink=lambda request, error: None,
        node_memory_view=view,
    )
    return {
        "journal": journal,
        "artifacts": artifacts,
        "projector": projector,
        "sink": sink,
        "ingress": ingress,
        "request_factory": request_factory,
        "coordinator": coordinator,
    }


def _messages(result) -> list[str]:
    return [dict(message)["content"] for message in result.messages]


def _append_scoped(runtime, event_id: str, event_type: str, **payload):
    attempt = runtime["projector"].attempt
    sanitized = runtime["projector"]._sanitize_payload(
        event_type,
        {"run_id": attempt.attempt_id, **payload},
        protected={"run_id": attempt.attempt_id},
    )
    return _append(runtime["journal"], event_id, event_type, attempt, **sanitized)


def _public_history(journal, root: AttemptRef) -> tuple:
    first = _append(
        journal,
        "event-root-user",
        "message.user",
        root,
        run_id=root.attempt_id,
        message={"role": "user", "content": "public root request"},
    )
    second = _append(
        journal,
        "event-root-assistant",
        "message.assistant",
        root,
        run_id=root.attempt_id,
        message={"role": "assistant", "content": "public root response"},
    )
    return first, second


def test_node_memory_view_compiles_fixed_root_history_and_own_events_only(tmp_path: Path):
    store = _open_store(tmp_path)
    journal = store.bind_execution(_EXECUTION)
    root = _attempt("root-attempt")
    node = _attempt("node-attempt")
    sibling = _attempt("sibling-attempt")
    history = _public_history(journal, root)
    sibling_runtime = _runtime(
        store,
        sibling,
        None,
        parent_run_id=root.attempt_id,
    )
    sibling_before = sibling_runtime["ingress"].persist(
        HostResolvedCurrentInput(
            attempt=sibling,
            content="sibling private before",
            message_index=0,
        )
    )
    assert sibling_before.event.payload["parent_run_id"] == root.attempt_id
    view = NodeMemoryView(
        node,
        _DEFINITION,
        tuple((event.event_id, event.store_seq, journal_event_sha256(event)) for event in history),
    )
    persisted = view.persist(journal)
    assert all(
        event.attempt != node
        for event in journal.capture_snapshot().events
        if event.event_id != persisted.event.event_id
    )

    runtime = _runtime(store, node, view, parent_run_id=root.attempt_id)
    own_first = runtime["ingress"].persist(
        HostResolvedCurrentInput(attempt=node, content="node first input", message_index=0)
    )
    assert own_first.event.payload["parent_run_id"] == root.attempt_id
    first_request = runtime["request_factory"](_context(node))
    first = runtime["coordinator"].compile(first_request)
    assert _messages(first) == [
        "node system",
        "public root request",
        "public root response",
        "node first input",
    ]
    assert "sibling private before" not in str(first.messages)

    own_assistant = _append_scoped(
        runtime,
        "event-node-private-assistant",
        "message.assistant",
        message={"role": "assistant", "content": "node private response"},
    )
    assert own_assistant.payload["parent_run_id"] == root.attempt_id
    sibling_between = sibling_runtime["ingress"].persist(
        HostResolvedCurrentInput(
            attempt=sibling,
            content="sibling private between",
            message_index=1,
        )
    )
    assert sibling_between.event.payload["parent_run_id"] == root.attempt_id
    runtime["ingress"].persist(
        HostResolvedCurrentInput(attempt=node, content="node second input", message_index=1)
    )
    second_request = runtime["request_factory"](_context(node))
    second = runtime["coordinator"].compile(second_request)
    assert _messages(second) == [
        "node system",
        "public root request",
        "public root response",
        "node first input",
        "node private response",
        "node second input",
    ]
    assert "sibling private before" not in str(second.messages)
    assert "sibling private between" not in str(second.messages)

    reopened = _open_store(tmp_path)
    cold_view = NodeMemoryView.from_dict(view.to_dict())
    cold = _runtime(reopened, node, cold_view, parent_run_id=root.attempt_id)
    rebuilt = cold["coordinator"].compile(cold["request_factory"](_context(node)))
    assert _messages(rebuilt) == _messages(second)
    assert rebuilt.envelope == second.envelope

    _append_scoped(
        runtime,
        "event-node-private-final",
        "final_message",
        content="node private final",
    )
    _append_scoped(
        runtime,
        "event-node-private-tool",
        "tool_result",
        call_id="node-private-call",
        result={"content": "node private tool text"},
    )
    assert any(
        event.event_type == "context.node_memory_view"
        for event in runtime["journal"].capture_snapshot().events
    )
    root_runtime = _runtime(store, root, None)
    root_runtime["ingress"].persist(
        HostResolvedCurrentInput(
            attempt=root,
            content="later root input",
            message_index=1,
        )
    )
    root_result = root_runtime["coordinator"].compile(
        root_runtime["request_factory"](_context(root))
    )
    assert _messages(root_result) == [
        "node system",
        "public root request",
        "public root response",
        "later root input",
    ]
    for private_text in (
        "node first input",
        "node private response",
        "node private final",
        "node private tool text",
        "sibling private before",
        "sibling private between",
    ):
        assert private_text not in str(root_result.messages)


def test_node_memory_view_rejects_wrong_rebound_and_late_bindings(tmp_path: Path):
    store = _open_store(tmp_path)
    journal = store.bind_execution(_EXECUTION)
    root = _attempt("root-attempt")
    node = _attempt("node-attempt")
    history = _public_history(journal, root)
    view = NodeMemoryView(
        node,
        _DEFINITION,
        tuple((event.event_id, event.store_seq, journal_event_sha256(event)) for event in history),
    )
    view.persist(journal)
    runtime = _runtime(store, node, view, parent_run_id=root.attempt_id)
    runtime["ingress"].persist(
        HostResolvedCurrentInput(attempt=node, content="node input", message_index=0)
    )

    with pytest.raises(JournalContextRequestFactoryError, match="attempt mismatch"):
        JournalContextRequestFactory(
            attempt=node,
            journal=journal,
            model_window_fallback=lambda provider, model: 16_384,
            node_memory_view=NodeMemoryView(_attempt("wrong-attempt"), _DEFINITION),
        )

    rebound = NodeMemoryView(node, "b" * 64, view.history)
    rebound_factory = JournalContextRequestFactory(
        attempt=node,
        journal=journal,
        model_window_fallback=lambda provider, model: 16_384,
        node_memory_view=rebound,
    )
    with pytest.raises(JournalContextRequestFactoryError, match="node memory view is invalid"):
        rebound_factory(_context(node))

    late = _attempt("late-attempt")
    _append(
        journal,
        "event-late-node-input",
        "message.user",
        late,
        run_id=late.attempt_id,
        message={"role": "user", "content": "late input"},
    )
    with pytest.raises(ValueError, match="precede"):
        NodeMemoryView(late, _DEFINITION).persist(journal)


def test_node_memory_view_bundle_rejects_mismatched_selector_and_legacy_is_none(tmp_path: Path):
    store = _open_store(tmp_path)
    journal = store.bind_execution(_EXECUTION)
    root = _attempt("root-attempt")
    node = _attempt("node-attempt")
    history = _public_history(journal, root)
    view = NodeMemoryView(
        node,
        _DEFINITION,
        tuple((event.event_id, event.store_seq, journal_event_sha256(event)) for event in history),
    )
    view.persist(journal)
    runtime = _runtime(store, node, view, parent_run_id=root.attempt_id)
    mismatched_factory = JournalContextRequestFactory(
        attempt=node,
        journal=runtime["journal"],
        model_window_fallback=lambda provider, model: 16_384,
        node_memory_view=NodeMemoryView(node, "b" * 64, view.history),
    )
    handoffs = HandoffService(runtime["artifacts"])
    with pytest.raises(ContextExecutionBundleError, match="same node memory view"):
        ContextExecutionBundle(
            attempt=node,
            journal=runtime["journal"],
            projector=runtime["projector"],
            durable_event_sink=runtime["sink"],
            coordinator=runtime["coordinator"],
            artifacts=runtime["artifacts"],
            handoffs=handoffs,
            ingress=runtime["ingress"],
            request_factory=mismatched_factory,
            tool_boundary=DurableToolBoundary(
                attempt=node,
                projector=runtime["projector"],
                sink=runtime["sink"],
            ),
            handoff_recorder=DurableHandoffRecorder(
                attempt=node,
                handoffs=handoffs,
                projector=runtime["projector"],
                sink=runtime["sink"],
            ),
            partial_attempt_sink=lambda event, error: None,
        )

    legacy = _runtime(store, _attempt("legacy-attempt"), None)
    assert legacy["request_factory"].node_memory_view is None
    assert legacy["coordinator"].node_memory_view is None
