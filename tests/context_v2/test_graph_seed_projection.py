from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tests.context_v2.test_graph_bootstrap_harness import (
    STEP,
    _context,
    _open,
)
from unchain.context.compiler import ContextCompiler, JournalMessageProjectionError
from unchain.context.coordinator import _prepare_journal_view
from unchain.context.request_factory import JournalContextRequestFactory
from unchain.context.graph_checkpoint import GraphCheckpointError
from unchain.journal.models import _thaw_json


def _request(journal):
    return JournalContextRequestFactory(
        attempt=STEP,
        journal=journal,
        model_window_fallback=lambda _provider, _model: 16384,
    )(_context())


def test_graph_seed_projects_original_request_without_mutating_durable_input(tmp_path):
    journal, _, _, plan, _, _, harness = _open(tmp_path)
    harness.build_delta(_context())
    before = [event.to_dict() for event in journal.capture_snapshot().events]
    derived = next(event for event in journal.capture_snapshot().events
                   if event.event_type == "message.user" and event.attempt == STEP)
    assert json.loads(derived.payload["message"]["content"])["schema"] == (
        "unchain.derived_handoff_input.v1"
    )

    request = _request(journal)
    assert _thaw_json(request.source_messages[-1]) == {
        "role": "user", "content": "bootstrap graph step",
    }
    assert request.source_message_cursors[-1].event_id == plan.initial_input_cursor.event_id
    result = ContextCompiler().compile(request)
    messages = _thaw_json(result.messages)
    assert messages == [{"role": "user", "content": "bootstrap graph step"}]
    assert [event.to_dict() for event in journal.capture_snapshot().events] == before


def test_coordinator_keeps_derived_receipt_authority_for_original_message(tmp_path):
    journal, _, _, plan, _, _, harness = _open(tmp_path)
    harness.build_delta(_context())
    request = _request(journal)
    projected, view = _prepare_journal_view(
        request=request, snapshot=journal.capture_snapshot(),
    )
    assert view.input_receipt.attempt == STEP
    assert view.input_receipt.event_id != plan.initial_input_cursor.event_id
    assert json.loads(view.input_receipt.payload["message"]["content"])["schema"] == (
        "unchain.derived_handoff_input.v1"
    )
    assert projected.source_message_cursors[-1].event_id == plan.initial_input_cursor.event_id
    assert _thaw_json(ContextCompiler().compile(projected).messages) == [
        {"role": "user", "content": "bootstrap graph step"},
    ]


def test_graph_seed_projection_is_stable_after_reopen_and_bootstrap_replay(tmp_path):
    journal, _, _, _, _, _, harness = _open(tmp_path)
    harness.build_delta(_context())
    request = _request(journal)
    before = [event.to_dict() for event in journal.capture_snapshot().events]
    reopened, _, _, _, _, _, restarted = _open(tmp_path)
    with pytest.raises(GraphCheckpointError, match="replay is forbidden"):
        restarted.build_delta(_context())
    next_request = _request(reopened)
    assert next_request.build_id == request.build_id
    assert [event.to_dict() for event in reopened.capture_snapshot().events] == before
    assert _thaw_json(ContextCompiler().compile(next_request).messages) == [
        {"role": "user", "content": "bootstrap graph step"},
    ]


def test_coordinator_rejects_changed_visible_source_bytes(tmp_path):
    journal, _, _, _, _, _, harness = _open(tmp_path)
    harness.build_delta(_context())
    request = _request(journal)
    forged = replace(request, source_messages=({"role": "user", "content": "forged"},))
    with pytest.raises(JournalMessageProjectionError) as raised:
        _prepare_journal_view(request=forged, snapshot=journal.capture_snapshot())
    assert raised.value.reason == "source_message_mismatch"


@pytest.mark.parametrize("mutation", ["descriptor-extra", "descriptor-version", "source-attempt", "source-text", "artifact-digest", "plan-extra", "start-cursor"])
def test_seed_projection_rejects_changed_durable_linkage(tmp_path, mutation):
    from unchain.context.request_factory import (
        JournalContextRequestFactoryError, _graph_seed_projections,
    )

    journal, _, _, plan, _, _, harness = _open(tmp_path)
    harness.build_delta(_context())
    events = list(journal.capture_snapshot().events)
    if mutation == "source-text":
        index = next(i for i, event in enumerate(events) if event.event_id == plan.initial_input_cursor.event_id)
        payload = _thaw_json(events[index].payload)
        payload["message"]["content"] = "a different user request"
    elif mutation == "plan-extra":
        index = next(i for i, event in enumerate(events) if event.event_type == "graph.execution.admitted")
        payload = _thaw_json(events[index].payload)
        payload["plan"]["unexpected"] = True
    elif mutation == "start-cursor":
        index = next(i for i, event in enumerate(events) if event.event_type == "graph.step.started")
        payload = _thaw_json(events[index].payload)
        payload["input_cursor"] = plan.initial_input_cursor.to_dict()
    else:
        index = next(i for i, event in enumerate(events) if event.event_type == "message.user" and event.attempt == STEP)
        payload = _thaw_json(events[index].payload)
        descriptor = json.loads(payload["message"]["content"])
        if mutation == "descriptor-extra":
            descriptor["unexpected"] = True
        elif mutation == "descriptor-version":
            descriptor["schema"] = "unchain.derived_handoff_input.v999"
        elif mutation == "source-attempt":
            descriptor["source_attempt"] = STEP.to_dict()
        else:
            descriptor["full_output_artifact"]["sha256"] = "0" * 64
        payload["message"]["content"] = json.dumps(descriptor)
    events[index] = replace(events[index], payload=payload)
    with pytest.raises(JournalContextRequestFactoryError):
        _graph_seed_projections(events)


def test_schema_looking_user_text_remains_literal_user_input(tmp_path):
    from tests.context_v2.test_derived_handoff_input import (
        SOURCE_ATTEMPT, _open_runtime, _context as derived_context,
    )
    from unchain.context.ingress import ContextInputIngress, HostResolvedCurrentInput

    _, journal, _, sink, _ = _open_runtime(tmp_path)
    text = '{"schema":"unchain.derived_handoff_input.v1","request":"explain this JSON"}'
    ContextInputIngress(
        attempt=SOURCE_ATTEMPT, projector=sink.projector, sink=sink,
    ).persist(HostResolvedCurrentInput(attempt=SOURCE_ATTEMPT, content=text, message_index=0))
    context = derived_context()
    context.event["run_id"] = SOURCE_ATTEMPT.attempt_id
    request = JournalContextRequestFactory(
        attempt=SOURCE_ATTEMPT, journal=journal,
        model_window_fallback=lambda _provider, _model: 16384,
    )(context)
    result = ContextCompiler().compile(request)
    assert _thaw_json(result.messages[-1]) == {"role": "user", "content": text}


def test_sqlite_coordinator_persists_derived_trigger_and_replays_without_resend(
    tmp_path,
):
    import copy
    from types import SimpleNamespace

    from unchain.context import ArtifactService, ContextCompileCoordinator
    from unchain.context.provider_execution import (
        ContextProviderTurnExecutionService,
    )
    from unchain.journal import EventCursor
    from unchain.persistence.sqlite_context_compiler_v2 import (
        SQLiteContextCompilerV2Store,
    )
    from unchain.persistence.sqlite_v2 import SQLiteContextV2Store
    from unchain.providers import OpenAIModelIO
    from unchain.providers.base import ModelTurnRequest
    from unchain.providers.durable_turn_runtime import DurableProviderTurnMode
    from unchain.retry import RetryConfig
    from unchain.tools import Toolkit

    journal, _, _, plan, _, _, harness = _open(tmp_path)
    harness.build_delta(_context())
    derived = next(
        event
        for event in journal.capture_snapshot().events
        if event.event_type == "message.user" and event.attempt == STEP
    )
    derived_cursor = EventCursor(derived.store_seq, derived.event_id)

    def request_for(reopened_journal, artifacts):
        return JournalContextRequestFactory(
            attempt=STEP,
            journal=reopened_journal,
            model_window_fallback=lambda _provider, _model: 16384,
            artifacts=artifacts,
        )(_context())

    def open_compiler():
        store = SQLiteContextV2Store(
            database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
            object_directory=tmp_path / "memory_v2" / "objects",
        )
        reopened_journal = store.bind_execution(STEP.generation.execution_id)
        artifacts = ArtifactService(
            reopened_journal,
            sanitizer=lambda content, _media_type: content,
        )
        capabilities = SQLiteContextCompilerV2Store(
            context_store=store,
        ).bind_execution(
            STEP.generation.execution_id,
            artifacts=artifacts,
        )
        coordinator = ContextCompileCoordinator(
            journal=reopened_journal,
            checkpoint_repository=capabilities.checkpoints,
            build_repository=capabilities.context_builds,
            partial_attempt_sink=lambda request, error: None,
            artifacts=artifacts,
        )
        return reopened_journal, artifacts, capabilities, coordinator

    bound_journal, artifacts, capabilities, coordinator = open_compiler()
    first_request = request_for(bound_journal, artifacts)
    first = coordinator.compile(first_request)
    first_receipt = capabilities.context_builds.get_by_trigger(
        trigger_cursor=derived_cursor,
    )
    assert first_receipt is not None
    assert first_receipt.trigger_cursor == derived_cursor
    assert first_receipt.trigger_cursor != plan.initial_input_cursor
    assert first_receipt.envelope == first.envelope
    assert first_request.source_message_cursors[-1].event_id == (
        plan.initial_input_cursor.event_id
    )
    assert _thaw_json(first.messages) == [
        {"role": "user", "content": "bootstrap graph step"},
    ]

    repeated_request = request_for(bound_journal, artifacts)
    repeated = coordinator.compile(repeated_request)
    assert repeated == first
    assert repeated_request.build_id == first_request.build_id
    assert capabilities.context_builds.get_by_trigger(
        trigger_cursor=derived_cursor,
    ) == first_receipt

    cold_journal, cold_artifacts, cold_capabilities, cold_coordinator = (
        open_compiler()
    )
    cold_request = request_for(cold_journal, cold_artifacts)
    cold = cold_coordinator.compile(cold_request)
    assert cold == first
    assert cold_request.build_id == first_request.build_id
    assert cold_capabilities.context_builds.get_by_trigger(
        trigger_cursor=derived_cursor,
    ) == first_receipt

    provider_sends = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def __iter__(self):
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="response-graph-seed-projection",
                    output=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "graph seed provider result",
                                }
                            ],
                        }
                    ],
                    usage={
                        "input_tokens": 2,
                        "output_tokens": 2,
                        "total_tokens": 4,
                    },
                ),
            )

    class Responses:
        def create(self, **kwargs):
            provider_sends.append(copy.deepcopy(kwargs))
            return Stream()

    class Client:
        responses = Responses()

    def model_io():
        return OpenAIModelIO(
            model="gpt-test",
            api_key="test-key",
            client_factory=lambda **_kwargs: Client(),
            default_payloads={},
            model_capabilities={},
        )

    provider_request = ModelTurnRequest(
        messages=_thaw_json(first.messages),
        payload={},
        callback=None,
        run_id=STEP.attempt_id,
        iteration=0,
        toolkit=Toolkit(),
        emit_stream=True,
    )
    retry = RetryConfig(max_retries=0)
    provider = ContextProviderTurnExecutionService(
        attempt=STEP,
        store=cold_journal,
        mode=DurableProviderTurnMode.ENFORCE_TEST,
        transport_target_sha256="b" * 64,
        sleep=lambda _seconds: None,
    )
    sent = provider.fetch_prepared(
        model_io=model_io(),
        request=provider_request,
        retry_config=retry,
    )
    repeated_provider_result = provider.fetch_prepared(
        model_io=model_io(),
        request=provider_request,
        retry_config=retry,
    )
    provider_after_reopen = ContextProviderTurnExecutionService(
        attempt=STEP,
        store=open_compiler()[0],
        mode=DurableProviderTurnMode.ENFORCE_TEST,
        transport_target_sha256="b" * 64,
        sleep=lambda _seconds: None,
    )
    cold_provider_result = provider_after_reopen.fetch_prepared(
        model_io=model_io(),
        request=provider_request,
        retry_config=retry,
    )

    assert len(provider_sends) == 1
    assert sent.final_text == "graph seed provider result"
    assert repeated_provider_result.final_text == sent.final_text
    assert cold_provider_result.final_text == sent.final_text
