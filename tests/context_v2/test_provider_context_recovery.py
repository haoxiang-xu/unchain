from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from unchain.context.provider_recovery import (
    ContextLimitRecoveryError,
    apply_provider_context_recovery,
    load_provider_context_recovery,
    prepare_provider_context_recovery,
    provider_recovery_build_trigger,
)
from unchain.context.budget import resolve_context_budget
from unchain.context.models import ContextCompileRequest
from unchain.journal import AttemptRef, GenerationRef, JournalAppendRequest, OperationRef
from unchain.persistence import SQLiteContextV2Store


ATTEMPT = AttemptRef(GenerationRef("recovery-execution", "recovery-generation"), "recovery-attempt")


def _journal(tmp_path):
    store = SQLiteContextV2Store(database_path=tmp_path / "context.sqlite3", object_directory=tmp_path / "objects")
    journal = store.bind_execution(ATTEMPT.generation.execution_id)
    journal.append(request=JournalAppendRequest(
        event_id="input-recovery", event_type="message.user", attempt=ATTEMPT,
        operation=OperationRef("input-operation", "a" * 64),
        payload={"message": {"role": "user", "content": "keep this input"}},
    ))
    return journal


def _record(journal, *, iteration=0):
    return prepare_provider_context_recovery(
        journal=journal, attempt=ATTEMPT, iteration=iteration, provider="openai", model="test-model",
        envelope={"build_id": "rejected-build", "budget": {"available_input_tokens": 8000}, "estimated_input_tokens": 4000},
        messages=[{"role": "user", "content": "keep this input"}],
    )


def test_recovery_record_is_durable_and_scoped(tmp_path):
    journal = _journal(tmp_path)
    first = _record(journal)
    recovered = load_provider_context_recovery(journal=journal, attempt=ATTEMPT, iteration=0, provider="openai", model="test-model")
    assert recovered == first
    assert first.payload["ordinal"] == 1
    assert first.payload["available_input_tokens"] == 3000
    assert first.payload["source_input_event_id"] == "input-recovery"
    assert "keep this input" not in str(dict(first.payload))
    assert load_provider_context_recovery(journal=journal, attempt=ATTEMPT, iteration=1, provider="openai", model="test-model") is None
    with pytest.raises(ContextLimitRecoveryError):
        load_provider_context_recovery(journal=journal, attempt=ATTEMPT, iteration=0, provider="anthropic", model="test-model")


def test_recovery_changes_input_allowance_without_falsifying_model_window(tmp_path):
    journal = _journal(tmp_path)
    event = _record(journal)
    request = ContextCompileRequest(case="test", source_messages=({"role": "user", "content": "keep this input"},),
        budget=resolve_context_budget(context_window_tokens=10000), provider="openai", model="test-model",
        build_id="original-build", execution_id=ATTEMPT.generation.execution_id,
        generation_id=ATTEMPT.generation.generation_id, attempt_id=ATTEMPT.attempt_id)
    reduced = apply_provider_context_recovery(request, event)
    assert reduced.budget.available_input_tokens == 3000
    assert reduced.budget.context_window_tokens == request.budget.context_window_tokens
    assert reduced.budget.output_reserve_tokens == request.budget.output_reserve_tokens
    assert reduced.source_messages == request.source_messages
    assert reduced.build_id != request.build_id
    assert apply_provider_context_recovery(request, event) == reduced
    with pytest.raises(ContextLimitRecoveryError):
        apply_provider_context_recovery(replace(request, model="other-model"), event)


def test_recovery_has_a_durable_two_send_limit(tmp_path):
    journal = _journal(tmp_path)
    first = _record(journal)
    second = _record(journal)
    assert second.payload["ordinal"] == 2
    assert second.payload["parent_recovery_event_id"] == first.event_id
    assert second.payload["available_input_tokens"] < first.payload["available_input_tokens"]
    with pytest.raises(ContextLimitRecoveryError):
        _record(journal)


def test_changed_canonical_input_invalidates_old_recovery(tmp_path):
    journal = _journal(tmp_path)
    _record(journal)
    journal.append(request=JournalAppendRequest(
        event_id="new-input", event_type="message.user", attempt=ATTEMPT,
        operation=OperationRef("new-input-operation", "b" * 64),
        payload={"message": {"role": "user", "content": "changed input"}},
    ))
    with pytest.raises(ContextLimitRecoveryError):
        load_provider_context_recovery(journal=journal, attempt=ATTEMPT, iteration=0, provider="openai", model="test-model")


@pytest.mark.parametrize("field,value", [
    ("schema", "unchain.context_provider_recovery.v999"), ("ordinal", 3),
    ("iteration", True), ("before_message_bytes", 0), ("provider", ""),
    ("unexpected", "extra"), ("source_input_store_seq", False),
    ("parent_recovery_event_id", "forged-parent"),
])
def test_malformed_recovery_record_is_rejected(tmp_path, field, value):
    journal = _journal(tmp_path)
    event = _record(journal)
    payload = dict(event.payload)
    payload[field] = value
    malformed = replace(event, payload=payload)
    fake = SimpleNamespace(capture_snapshot=lambda: SimpleNamespace(events=[journal.capture_snapshot().events[0], malformed]))
    with pytest.raises(ContextLimitRecoveryError):
        load_provider_context_recovery(journal=fake, attempt=ATTEMPT, iteration=0, provider="openai", model="test-model")


@pytest.mark.parametrize("envelope", [
    {"build_id": "", "budget": {"available_input_tokens": 8000}, "estimated_input_tokens": 4000},
    {"build_id": "x" * 513, "budget": {"available_input_tokens": 8000}, "estimated_input_tokens": 4000},
    {"build_id": "b", "budget": {"available_input_tokens": True}, "estimated_input_tokens": 4000},
])
def test_invalid_producer_does_not_write_partial_recovery(tmp_path, envelope):
    journal = _journal(tmp_path)
    with pytest.raises(ContextLimitRecoveryError):
        prepare_provider_context_recovery(journal=journal, attempt=ATTEMPT, iteration=0,
            provider="openai", model="test-model", envelope=envelope, messages=[{"role": "user", "content": "input"}])
    assert len(journal.capture_snapshot().events) == 1


def test_compiler_recovery_trigger_has_independently_verified_input_and_build_chain(tmp_path):
    journal = _journal(tmp_path)
    source = journal.capture_snapshot().events[0]
    identity = {"domain": "unchain.context_build.trigger.v1", "attempt": ATTEMPT.to_dict(),
        "trigger": {"event_id": source.event_id, "store_seq": source.store_seq},
        "provider": "openai", "model": "test-model"}
    base = "context-build-" + hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode()).hexdigest()
    event = prepare_provider_context_recovery(journal=journal, attempt=ATTEMPT, iteration=0,
        provider="openai", model="test-model", envelope={"build_id": base,
        "budget": {"available_input_tokens": 8000}, "estimated_input_tokens": 4000}, messages=[{"role": "user", "content": "input"}])
    request = ContextCompileRequest(case="test", source_messages=({"role": "user", "content": "input"},),
        budget=resolve_context_budget(context_window_tokens=10000), provider="openai", model="test-model",
        build_id=base, execution_id=ATTEMPT.generation.execution_id,
        generation_id=ATTEMPT.generation.generation_id, attempt_id=ATTEMPT.attempt_id)
    reduced = apply_provider_context_recovery(request, event)
    assert provider_recovery_build_trigger(reduced, journal.capture_snapshot(), source) == event
    with pytest.raises(ContextLimitRecoveryError):
        provider_recovery_build_trigger(replace(reduced, build_id="context-recovery-build-" + "0" * 64), journal.capture_snapshot(), source)
    with pytest.raises(ContextLimitRecoveryError):
        provider_recovery_build_trigger(replace(reduced, budget=request.budget), journal.capture_snapshot(), source)


def test_confirmed_context_diagnostic_roundtrips_with_old_bytes_unchanged():
    from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic
    from unchain.providers.request_lease import ProviderRequestLease
    from test_provider_diagnostic_branch_compatibility import lease_wire
    old = lease_wire('http_v1')
    old['failure_diagnostic']['provider_code'] = 'context_length_exceeded'
    restored = ProviderRequestLease.from_durable_dict(old)
    assert restored.to_dict() == old
    assert restored.failure_diagnostic.input_context_rejected is False
    diagnostic = ProviderFailureDiagnostic(400, 'context_length_exceeded', input_context_rejected=True)
    current = dict(old, schema='unchain.provider_request_lease.v5', failure_diagnostic=diagnostic.to_dict())
    assert ProviderRequestLease.from_durable_dict(current).to_dict() == current
    for changed in [dict(current, schema=old['schema']), dict(old, failure_diagnostic=diagnostic.to_dict())]:
        with pytest.raises(ValueError):
            ProviderRequestLease.from_durable_dict(changed)


@pytest.mark.parametrize("path", ["graph", "derived_child"])
def test_recovery_recompiles_real_graph_and_derived_child_inputs(tmp_path, path):
    from unchain.context.artifacts import ArtifactService
    from unchain.context.coordinator import ContextCompileCoordinator
    from unchain.context.request_factory import JournalContextRequestFactory
    from unchain.context.provider_recovery import verify_provider_context_reduction
    from unchain.journal.runtime import SemanticEventDraft
    from unchain.persistence.sqlite_context_compiler_v2 import SQLiteContextCompilerV2Store
    if path == "graph":
        from tests.context_v2.test_graph_bootstrap_harness import _open, _context, STEP, ORCHESTRATION
        attempt, old_attempt = STEP, ORCHESTRATION
        store = SQLiteContextV2Store(database_path=tmp_path / 'memory_v2' / 'context_v2.sqlite3',
                                    object_directory=tmp_path / 'memory_v2' / 'objects')
        journal = store.bind_execution(attempt.generation.execution_id)
    else:
        from tests.context_v2.test_derived_handoff_input import (
            _open_runtime, _context, _append_source_events, _input, CONSUMER_ATTEMPT, SOURCE_ATTEMPT,
        )
        store, journal, artifacts, source_sink, derived = _open_runtime(tmp_path)
        attempt, old_attempt = CONSUMER_ATTEMPT, SOURCE_ATTEMPT
    for index, role in enumerate(('user', 'assistant')):
        draft = SemanticEventDraft(event_id=f'history-{index}', event_type=f'message.{role}',
            attempt=old_attempt, operation_id=f'history-operation-{index}',
            payload={'run_id': old_attempt.attempt_id, 'message': {'role': role, 'content': 'Earlier completed work. ' * 250}})
        journal.append(request=JournalAppendRequest(event_id=draft.event_id, event_type=draft.event_type,
            attempt=draft.attempt, operation=draft.operation, payload=draft.payload))
    if path == 'graph':
        journal, _, _, _, _, _, bootstrap = _open(tmp_path)
        bootstrap.build_delta(_context())
        artifacts = ArtifactService(journal, sanitizer=lambda content, media_type: content)
    else:
        derived.persist(_input(_append_source_events(source_sink)))
    context = _context()
    factory = JournalContextRequestFactory(attempt=attempt, journal=journal, artifacts=artifacts,
                                          model_window_fallback=lambda provider, model: 16384)
    capabilities = SQLiteContextCompilerV2Store(context_store=store).bind_execution(
        attempt.generation.execution_id, artifacts=artifacts)
    coordinator = ContextCompileCoordinator(journal=journal, artifacts=artifacts,
        checkpoint_repository=capabilities.checkpoints, build_repository=capabilities.context_builds,
        partial_attempt_sink=lambda request, error: None)
    request = factory(context)
    rejected = coordinator.compile(request)
    event = prepare_provider_context_recovery(journal=journal, attempt=attempt, iteration=0,
        provider=request.provider, model=request.model, envelope=rejected.envelope.to_dict(), messages=rejected.messages)
    result = coordinator.compile(apply_provider_context_recovery(factory(context), event))
    verify_provider_context_reduction(result, event)
    assert result.envelope.build_id != rejected.envelope.build_id
    assert result.messages[-1] == rejected.messages[-1]
