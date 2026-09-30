from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from unchain.journal import AttemptRef, GenerationRef, JournalAppendRequest, OperationRef
from unchain.kernel import KernelLoop
from unchain.kernel.harness import HarnessContext
from unchain.persistence.sqlite_v2 import SQLiteContextV2Store
from unchain.skills.activation import ActivationQueue
from unchain.skills.harness import SkillActivationHarness
from unchain.skills.registry import SkillRegistry, SkillsConfig
from unchain.skills.rendering import is_active_skills_message, parse_active_skills_block


def binding(root, generation='g', attempt='a'):
    store = SQLiteContextV2Store(database_path=root / 'state.db', object_directory=root / 'objects')
    return AttemptRef(GenerationRef('chat', generation), attempt), store.bind_execution('chat')


def user(bound, text, key):
    attempt, journal = bound
    journal.append(request=JournalAppendRequest(
        event_id=key, event_type='message.user', attempt=attempt,
        operation=OperationRef(key, hashlib.sha256(text.encode()).hexdigest()),
        payload={'message': {'role': 'user', 'content': text}},
    ))


def step(root, bound, text):
    registry = SkillRegistry(SkillsConfig(project_root=root, include_user_dirs=False))
    harness = SkillActivationHarness(registry=registry, queue=ActivationQueue())
    harness.journal_binding = lambda context: bound
    state = KernelLoop().seed_state([{'role':'user','content':text}], provider='openai', model='gpt-4.1')
    delta = harness.build_delta(HarnessContext(state=state, phase='before_model', event={}))
    if delta is not None:
        state.apply_delta(delta)
    return [e for m in state.latest_messages() if is_active_skills_message(m) for e in parse_active_skills_block(m['content'])]


def write(root, body):
    path = root / '.agents/skills/demo/SKILL.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('---\nname: demo\ndescription: Demo\n---\n' + body)
    return path


def test_journal_restores_skill_on_fresh_turn_and_cold_binding(tmp_path):
    path = write(tmp_path, 'OLD BODY')
    first = binding(tmp_path)
    user(first, '/demo', 'user-1')
    assert [e.body for e in step(tmp_path, first, '/demo')] == ['OLD BODY']
    path.unlink()
    cold = binding(tmp_path, attempt='b')
    user(cold, 'continue', 'user-2')
    assert [e.body for e in step(tmp_path, cold, 'continue')] == ['OLD BODY']


def test_journal_turn_cursor_distinguishes_identical_text_after_context_trim(tmp_path):
    write(tmp_path, 'OLD BODY')
    first = binding(tmp_path)
    user(first, '/demo', 'user-1')
    step(tmp_path, first, '/demo')
    write(tmp_path, 'NEW BODY')
    # Same user receipt, fresh context: replay must keep the snapshot.
    assert [e.body for e in step(tmp_path, binding(tmp_path), '/demo')] == ['OLD BODY']
    new = binding(tmp_path, attempt='b')
    user(new, '/demo', 'user-2')
    assert [e.body for e in step(tmp_path, new, '/demo')] == ['NEW BODY']
    reset = binding(tmp_path, generation='reset', attempt='c')
    user(reset, 'continue', 'user-3')
    assert step(tmp_path, reset, 'continue') == []


@pytest.mark.parametrize('payload', [
    {'schema':'unchain.skills.activation_snapshot.v2','block':'bad'},
    {'schema':'unchain.skills.activation_snapshot.v1','block':'bad'},
    {'schema':'unchain.skills.activation_snapshot.v1','block':'bad','extra':True},
])
def test_journal_snapshot_admission_is_closed_and_fail_closed(tmp_path, payload):
    from unchain.skills.activation import SkillActivationStateError
    bound = binding(tmp_path)
    attempt, journal = bound
    journal.append(request=JournalAppendRequest(
        event_id='bad-snapshot', event_type='skills.activation_snapshot', attempt=attempt,
        operation=OperationRef('bad-snapshot', '0' * 64), payload=payload,
    ))
    with pytest.raises(SkillActivationStateError):
        step(tmp_path, bound, 'continue')


def test_same_turn_snapshot_write_is_idempotent(tmp_path):
    write(tmp_path, 'BODY')
    bound = binding(tmp_path)
    user(bound, '/demo', 'user-1')
    first = step(tmp_path, bound, '/demo')
    for _ in range(3):
        assert step(tmp_path, binding(tmp_path), '/demo') == first
    assert sum(e.event_type == 'skills.activation_snapshot'
               for e in bound[1].capture_snapshot().events) == 1


def test_snapshot_receipt_passes_canonical_semantic_operation_validation(tmp_path):
    from unchain.journal import SemanticEventDraft
    write(tmp_path, 'BODY')
    bound = binding(tmp_path)
    user(bound, '/demo', 'user-1')
    step(tmp_path, bound, '/demo')
    event = bound[1].capture_snapshot().events[-1]
    assert event.event_type == 'skills.activation_snapshot'
    expected = SemanticEventDraft(
        event_id=event.event_id, event_type=event.event_type, attempt=event.attempt,
        operation_id=event.operation.operation_id, payload=event.payload,
        resource_refs=event.resource_refs,
    )
    assert event.operation == expected.operation
