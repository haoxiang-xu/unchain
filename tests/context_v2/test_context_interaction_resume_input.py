from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from unchain.agent import (
    AgentBuilder,
    AgentCallContext,
    AgentSpec,
    AgentState,
    ContextModule,
)
from unchain.agent.model_io import ModelIOFactoryRegistry
from unchain.context import (
    ArtifactService,
    ContextCompileCoordinator,
    ContextExecutionBundle,
    ContextInputIngress,
    ContextRuntime,
    DurableContextRuntimeFactory,
    DurableHandoffRecorder,
    DurableToolBoundary,
    HandoffService,
    HostResolvedCurrentInput,
    HostResolvedInteractionInput,
    JournalContextRequestFactory,
)
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.journal import ArtifactRef, AttemptRef, DurableEventSink, GenerationRef
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.persistence.sqlite_context_compiler_v2 import (
    SQLiteContextCompilerV2Store,
)
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store


EXECUTION_ID = "execution-interaction-resume"
GENERATION_ID = "generation-interaction-resume"
INTERACTION_ID = "interaction-human-resume"
PAUSED_ATTEMPT = AttemptRef(
    GenerationRef(EXECUTION_ID, GENERATION_ID),
    "attempt-paused",
)
RESUMED_ATTEMPT = AttemptRef(
    GenerationRef(EXECUTION_ID, GENERATION_ID),
    "attempt-resumed",
)


class _CaptureModelIO:
    provider = "openai"
    model = "gpt-resume-test"

    def __init__(self) -> None:
        self.requests = []

    def fetch_turn(self, request):
        self.requests.append(request)
        raise AssertionError("model must not start after bootstrap persistence failure")


def _store(root: Path) -> SQLiteContextV2Store:
    return SQLiteContextV2Store(
        database_path=root / "context_v2.sqlite3",
        object_directory=root / "objects",
    )


def _seed_paused_interaction(root: Path) -> None:
    store = _store(root)
    journal = store.bind_execution(EXECUTION_ID)
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    projector = CanonicalSemanticEventProjector(
        attempt=PAUSED_ATTEMPT,
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    sink = DurableEventSink(journal, PAUSED_ATTEMPT, projector)
    receipt = sink(
        {
            "type": "interaction_requested",
            "run_id": PAUSED_ATTEMPT.attempt_id,
            "iteration": 1,
            "interaction_request": {
                "interaction_id": INTERACTION_ID,
                "kind": "human_input",
                "question": "Choose a framework",
            },
        }
    )
    assert receipt is not None
    assert receipt.event.event_type == "interaction.requested"


def _seed_paused_ask_interaction(root: Path, *, call_id: str) -> None:
    store = _store(root)
    journal = store.bind_execution(EXECUTION_ID)
    artifacts = ArtifactService(
        journal,
        sanitizer=lambda content, media_type: content,
    )
    projector = CanonicalSemanticEventProjector(
        attempt=PAUSED_ATTEMPT,
        artifacts=artifacts,
        payload_sanitizer=lambda event_type, payload: payload,
    )
    sink = DurableEventSink(journal, PAUSED_ATTEMPT, projector)
    call_receipt = sink(
        {
            "type": "tool_call",
            "run_id": PAUSED_ATTEMPT.attempt_id,
            "iteration": 1,
            "tool_name": "ask_user_question",
            "call_id": call_id,
            "arguments": {"question": "Choose a framework"},
            "source_provider": "openai",
        }
    )
    request_receipt = sink(
        {
            "type": "interaction_requested",
            "run_id": PAUSED_ATTEMPT.attempt_id,
            "iteration": 1,
            "interaction_request": {
                "interaction_id": INTERACTION_ID,
                "kind": "human_input",
                "payload": {"request_id": call_id},
            },
        }
    )
    assert call_receipt is not None
    assert call_receipt.event.event_type == "tool_call"
    assert request_receipt is not None
    assert request_receipt.event.event_type == "interaction.requested"


def _history_payload(messages) -> dict:
    message = next(
        item
        for item in messages
        if "MEMORY_V2_EVIDENCE_HISTORY" in str(item.get("content") or "")
    )
    return json.loads(message["content"].split("\n", 2)[2])


def _runtime(
    root: Path,
    *,
    current_input_resolver,
    sanitizer=lambda content, media_type: content,
):
    context_store = _store(root)
    compiler_store = SQLiteContextCompilerV2Store(context_store=context_store)
    bundles = {}

    def build(attempt):
        journal = context_store.bind_execution(attempt.generation.execution_id)
        artifacts = ArtifactService(journal, sanitizer=sanitizer)
        compiler = compiler_store.bind_execution(
            attempt.generation.execution_id,
            artifacts=artifacts,
        )
        projector = CanonicalSemanticEventProjector(
            attempt=attempt,
            artifacts=artifacts,
            payload_sanitizer=lambda event_type, payload: payload,
        )
        sink = DurableEventSink(journal, attempt, projector)
        handoffs = HandoffService(artifacts)
        bundle = ContextExecutionBundle(
            attempt=attempt,
            journal=journal,
            projector=projector,
            durable_event_sink=sink,
            coordinator=ContextCompileCoordinator(
                journal=journal,
                checkpoint_repository=compiler.checkpoints,
                build_repository=compiler.context_builds,
                partial_attempt_sink=lambda request, error: None,
            ),
            artifacts=artifacts,
            handoffs=handoffs,
            ingress=ContextInputIngress(
                attempt=attempt,
                projector=projector,
                sink=sink,
            ),
            request_factory=JournalContextRequestFactory(
                attempt=attempt,
                journal=journal,
                model_window_fallback=lambda provider, model: 16_384,
            ),
            tool_boundary=DurableToolBoundary(
                attempt=attempt,
                projector=projector,
                sink=sink,
            ),
            handoff_recorder=DurableHandoffRecorder(
                attempt=attempt,
                handoffs=handoffs,
                projector=projector,
                sink=sink,
            ),
            partial_attempt_sink=lambda event, error: None,
        )
        bundles[attempt.attempt_id] = bundle
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id="context-v2-interaction-resume",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, execution_id: GENERATION_ID,
            current_input_resolver=current_input_resolver,
        ),
    )
    return runtime, bundles


def _context(*, phase: str) -> HarnessContext:
    state = RunState()
    state.seed_messages(
        [
            {"role": "system", "content": "Use the explicit resume answer."},
            {"role": "user", "content": "stale transcript content"},
        ]
    )
    state.session_state.session_id = EXECUTION_ID
    state.provider_state.provider = "openai"
    state.provider_state.model = "gpt-resume-test"
    state.provider_state.max_context_window_tokens = 16_384
    return HarnessContext(
        state=state,
        phase=phase,
        event={"run_id": RESUMED_ATTEMPT.attempt_id, "toolkit": None},
    )


def _resolved_input(context, attempt):
    del context
    return HostResolvedInteractionInput(
        attempt=attempt,
        interaction_id=INTERACTION_ID,
        response={"selected_values": ["react"], "other_text": None},
        submitted_by="ui:test",
    )


def test_normal_pause_resumes_in_new_attempt_and_rebuilds_from_restart_snapshot(
    tmp_path: Path,
) -> None:
    _seed_paused_interaction(tmp_path)
    runtime, bundles = _runtime(
        tmp_path,
        current_input_resolver=_resolved_input,
    )
    runtime.bind_context(_context(phase="bootstrap"))
    first_bundle = bundles[RESUMED_ATTEMPT.attempt_id]
    first_snapshot = first_bundle.journal.capture_snapshot()
    assert [event.event_type for event in first_snapshot.events] == [
        "interaction.requested",
        "interaction.resolved",
    ]

    cold_runtime, cold_bundles = _runtime(
        tmp_path,
        current_input_resolver=lambda context, attempt: None,
    )
    cold_runtime.bind_context(_context(phase="bootstrap"))
    cold_bundle = cold_bundles[RESUMED_ATTEMPT.attempt_id]
    request = cold_bundle.request_factory(_context(phase="before_model"))
    result = cold_bundle.coordinator.compile(request)

    assert result.envelope.status.value == "complete"
    assert request.pending_task_inputs is not None
    [pending] = request.pending_task_inputs
    assert pending["type"] == "interaction_resolved"
    artifact = ArtifactRef(
        ref=pending["content_ref"],
        media_type="application/json",
        byte_length=pending["content_bytes"],
        sha256=pending["content_sha256"],
        preview=pending["preview"],
    )
    content = cold_bundle.artifacts.read_full(
        artifact,
        remaining_budget_bytes=pending["content_bytes"],
    )
    assert json.loads(content) == {
        "interaction_id": INTERACTION_ID,
        "response": {"other_text": None, "selected_values": ["react"]},
        "submitted_by": "ui:test",
    }


def test_resolution_persistence_failure_stops_compiler_and_model(
    tmp_path: Path,
) -> None:
    _seed_paused_interaction(tmp_path)

    def fail_sanitizer(content, media_type):
        del content, media_type
        raise OSError("injected interaction artifact persistence failure")

    runtime, _bundles = _runtime(
        tmp_path,
        current_input_resolver=_resolved_input,
        sanitizer=fail_sanitizer,
    )
    model_io = _CaptureModelIO()
    builder = AgentBuilder(
        agent=object(),
        spec=AgentSpec(
            name="interaction-resume-agent",
            provider="openai",
            model=model_io.model,
        ),
        state=AgentState(),
        call_context=AgentCallContext(
            mode="run",
            input_messages=copy.deepcopy(
                [
                    {"role": "system", "content": "Resume safely."},
                    {"role": "user", "content": "stale transcript"},
                ]
            ),
            session_id=EXECUTION_ID,
            run_id=RESUMED_ATTEMPT.attempt_id,
            max_context_window_tokens=16_384,
        ),
        model_io_registry=ModelIOFactoryRegistry(),
    )
    builder.set_model_io(model_io)
    ContextModule(runtime=runtime).configure(builder)
    prepared = builder.build()

    with pytest.raises(
        OSError,
        match="interaction artifact persistence failure",
    ):
        prepared.run()

    assert model_io.requests == []
    assert [
        event.event_type
        for event in _store(tmp_path)
        .bind_execution(EXECUTION_ID)
        .capture_snapshot()
        .events
    ] == ["interaction.requested"]


def test_cold_resume_answer_remains_visible_after_a_subsequent_user_turn(
    tmp_path: Path,
) -> None:
    call_id = "call-human-cold-resume"
    _seed_paused_ask_interaction(tmp_path, call_id=call_id)
    runtime, bundles = _runtime(
        tmp_path,
        current_input_resolver=_resolved_input,
    )
    runtime.bind_context(_context(phase="bootstrap"))
    first_bundle = bundles[RESUMED_ATTEMPT.attempt_id]
    first_snapshot = first_bundle.journal.capture_snapshot()
    assert [event.event_type for event in first_snapshot.events] == [
        "tool_call",
        "interaction.requested",
        "interaction.resolved",
    ]

    cold_runtime, cold_bundles = _runtime(
        tmp_path,
        current_input_resolver=lambda context, attempt: None,
    )
    cold_context = _context(phase="bootstrap")
    cold_runtime.bind_context(cold_context)
    cold_bundle = cold_bundles[RESUMED_ATTEMPT.attempt_id]
    first_request = cold_bundle.request_factory(_context(phase="before_model"))
    first_result = cold_bundle.coordinator.compile(first_request)
    assert first_result.envelope.status.value == "complete"
    assert first_result.diagnostics["atomic_call_ids"] == ()

    cold_bundle.ingress.persist(
        HostResolvedCurrentInput(
            attempt=RESUMED_ATTEMPT,
            content="Use that choice for the next step",
            message_index=1,
        )
    )
    next_request = cold_bundle.request_factory(_context(phase="before_model"))
    next_result = cold_bundle.coordinator.compile(next_request)

    assert next_result.envelope.status.value == "complete"
    assert next_result.diagnostics["atomic_call_ids"] == ()
    assert any(
        message.get("role") == "user"
        and message.get("content") == "Use that choice for the next step"
        for message in next_result.messages
    )
    history = _history_payload(next_result.messages)
    [resolved] = history["resolved_human_interactions"]
    assert resolved["interaction_id"] == INTERACTION_ID
    assert resolved["call_id"] == call_id
    assert resolved["tool_name"] == "ask_user_question"
    assert resolved["response"]["content_ref"]["kind"] == "artifact"
    assert "react" in resolved["response"]["preview"]
    assert "unfinished_tool_pairs" not in str(next_result.messages)
    assert not any(
        message.get("type") in {"function_call_output", "tool_result"}
        for message in next_result.messages
    )
    assert [
        event.event_type
        for event in _store(tmp_path)
        .bind_execution(EXECUTION_ID)
        .capture_snapshot()
        .events
    ] == [
        "tool_call",
        "interaction.requested",
        "interaction.resolved",
        "message.user",
    ]
