from __future__ import annotations

import hashlib
import json
from pathlib import Path

from unchain.context import (
    ArtifactService,
    ContextCompileCoordinator,
    ContextInputIngress,
    DurableToolApprovalState,
    DurableToolBoundary,
    DurableToolCompletionDraft,
    DurableToolExecutionRequest,
    DurableToolExecutionSubject,
    DurableToolExecutor,
    DurableToolRouteKind,
    HostResolvedCurrentInput,
    JournalContextRequestFactory,
)
from unchain.context.node_memory_view import NodeMemoryView
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.execution import ExecutionRuntime
from unchain.journal import AttemptRef, DurableEventSink, GenerationRef
from unchain.journal.snapshot import journal_event_sha256
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.memory import InMemorySessionStore
from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store
from unchain.tools.output_management import ToolOutputManager


_EXECUTION = "execution-managed-preview"
_GENERATION = "generation-managed-preview"
_DEFINITION = "d" * 64
_RAW_CURRENT = "PRIVATE_TOOL_BYTES_CURRENT_FALLBACK"
_RAW_SIBLING = "PRIVATE_TOOL_BYTES_SIBLING_FALLBACK"


def _attempt(name: str) -> AttemptRef:
    return AttemptRef(GenerationRef(_EXECUTION, _GENERATION), name)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _harness(attempt: AttemptRef) -> HarnessContext:
    state = RunState()
    state.session_state.session_id = _EXECUTION
    state.provider_state.provider = "openai"
    state.provider_state.model = "gpt-managed-preview"
    state.provider_state.max_context_window_tokens = 16_384
    state.seed_messages([{"role": "system", "content": "fixed node instructions"}])
    return HarnessContext(
        state=state,
        phase="before_model",
        event={"run_id": attempt.attempt_id, "toolkit": None},
    )


def _node_components(
    *,
    journal,
    artifacts: ArtifactService,
    attempt: AttemptRef,
    parent_run_id: str | None,
):
    def sanitize_payload(_event_type, payload):
        result = dict(payload)
        if parent_run_id is not None:
            result["parent_run_id"] = parent_run_id
        return result

    projector = CanonicalSemanticEventProjector(
        attempt=attempt,
        artifacts=artifacts,
        payload_sanitizer=sanitize_payload,
    )
    projector.bind_tool_output_manager(
        ToolOutputManager.active_default(attempt_id=attempt.attempt_id)
    )
    sink = DurableEventSink(journal, attempt, projector)
    return projector, sink, DurableToolBoundary(
        attempt=attempt,
        projector=projector,
        sink=sink,
    )


def _persist_managed_artifact_only_result(
    *,
    boundary: DurableToolBoundary,
    guard,
    result: dict[str, str],
) -> None:
    arguments = {"query": "safe"}
    call_id = "call-" + boundary.attempt.attempt_id
    intent = boundary.sink(
        {
            "type": "tool_call",
            "run_id": boundary.attempt.attempt_id,
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": call_id,
            "arguments": arguments,
            # Deliberately omit source_provider: native rehydration must
            # decline this pair and use the neutral managed projection.
        }
    )
    subject = DurableToolExecutionSubject(
        intent_cursor=intent.cursor,
        original_arguments_sha256=_digest(arguments),
        effective_arguments_sha256=_digest(arguments),
        approval_state=DurableToolApprovalState.NOT_REQUIRED,
        approval_request_sha256="",
        approval_receipt_sha256="",
        route_kind=DurableToolRouteKind.NORMAL,
        route_manifest_sha256="1" * 64,
        terminal_handler_manifest_sha256="2" * 64,
        execution_fence=guard.fence,
    )
    request = DurableToolExecutionRequest(
        tool_name="lookup",
        call_id=call_id,
        iteration=0,
        subject=subject,
        tool_result_policy="artifact_only",
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=guard,
    )
    receipt = executor.execute(
        request=request,
        guard=guard,
        invocation=executor.bind_invocation(
            request=request,
            effective_arguments=arguments,
            terminal_handler=lambda _arguments: DurableToolCompletionDraft(
                result=result
            ),
        ),
    )
    assert receipt.model_projection is not None
    assert receipt.model_projection["result"]["projection"] == "artifact_only"


def test_managed_artifact_only_preview_is_private_in_native_batch_fallback(
    tmp_path: Path,
) -> None:
    store = SQLiteContextV2Store(
        database_path=tmp_path / "context.sqlite3",
        object_directory=tmp_path / "objects",
    )
    journal = store.bind_execution(_EXECUTION)
    artifacts = ArtifactService(journal, sanitizer=lambda content, _media: content)
    root = _attempt("root")
    current = _attempt("node-current")
    sibling = _attempt("node-sibling")

    root_projector, root_sink, _root_boundary = _node_components(
        journal=journal,
        artifacts=artifacts,
        attempt=root,
        parent_run_id=None,
    )
    root_input = ContextInputIngress(
        attempt=root,
        projector=root_projector,
        sink=root_sink,
    ).persist(
        HostResolvedCurrentInput(
            attempt=root,
            content="public root history",
        )
    ).event
    view = NodeMemoryView(
        current,
        _DEFINITION,
        ((root_input.event_id, root_input.store_seq, journal_event_sha256(root_input)),),
    )
    view.persist(journal)

    _current_projector, _current_sink, current_boundary = _node_components(
        journal=journal,
        artifacts=artifacts,
        attempt=current,
        parent_run_id=root.attempt_id,
    )
    _sibling_projector, _sibling_sink, sibling_boundary = _node_components(
        journal=journal,
        artifacts=artifacts,
        attempt=sibling,
        parent_run_id=root.attempt_id,
    )
    guard = ExecutionRuntime(InMemorySessionStore()).acquire(_EXECUTION, "owner")
    try:
        _persist_managed_artifact_only_result(
            boundary=current_boundary,
            guard=guard,
            result={"payload": _RAW_CURRENT},
        )
        _persist_managed_artifact_only_result(
            boundary=sibling_boundary,
            guard=guard,
            result={"payload": _RAW_SIBLING},
        )

        capabilities = SQLiteContextCompilerV2Store(context_store=store).bind_execution(
            _EXECUTION,
            artifacts=artifacts,
        )
        coordinator = ContextCompileCoordinator(
            journal=journal,
            checkpoint_repository=capabilities.checkpoints,
            build_repository=capabilities.context_builds,
            partial_attempt_sink=lambda _request, _error: None,
            artifacts=artifacts,
            node_memory_view=view,
        )
        request_factory = JournalContextRequestFactory(
            attempt=current,
            journal=journal,
            model_window_fallback=lambda _provider, _model: 16_384,
            node_memory_view=view,
        )

        request = request_factory(_harness(current))
        assert request.pending_task_inputs is not None
        assert request.pending_task_inputs[0]["type"] == "tool_result"
        compiled = coordinator.compile(request)
        wire_messages = json.dumps(
            compiled.to_dict()["messages"],
            ensure_ascii=False,
            sort_keys=True,
        )

        assert _RAW_CURRENT not in wire_messages
        assert _RAW_SIBLING not in wire_messages
        assert "artifact_only" in wire_messages
        # The missing source-provider identity makes native reconstruction
        # unavailable; the result must remain a managed neutral projection.
        assert "function_call_output" not in wire_messages
        assert "delivered_as_native_current_tool_result" not in wire_messages
    finally:
        guard.release()
