"""Resume persisted after-hook phases without rewriting their pinned history.

The writer functions are exact exports from the two frozen wheels; the rest
of the stack is the real candidate core. These are regression tests, not a
substitute for the separate full-wheel/fresh-process upgrade qualification.
"""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

import tests.test_skeleton_kernel_interaction as base
from unchain.context.artifacts import ArtifactService
from unchain.context.tool_boundary import DurableToolBoundary
from unchain.execution import ExecutionRuntime
from unchain.journal import JournalAppendRequest, build_operation_ref
from unchain.journal.models import _thaw_json
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.skeleton.durable_execution import DurableWorkflowRunner
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hook_phase_journal import HookPhaseJournal, _digest
from unchain.skeleton.hook_raw_seal import HookRawSealStore
from unchain.skeleton.hooks_runtime import (
    SkeletonAfterToolHookDispatcher,
    apply_after_decision,
)


def _writer(name):
    fixture = json.loads((Path(__file__).parent / 'fixtures/hook_denial_upgrade_writers.json').read_text())
    record = fixture['writers'][name]
    source = record['source']
    assert sha256(source.encode()).hexdigest() == record['function_sha256']
    namespace = {'__name__': 'unchain.skeleton._frozen_test_writer',
        '__package__': 'unchain.skeleton', 'deepcopy': deepcopy,
        'SkeletonValidationError': SkeletonValidationError,
        'apply_after_decision': apply_after_decision}
    exec(compile(source, 'frozen-' + name + '-after-hook.py', 'exec'), namespace)
    return namespace['__call__']


class _Cut(BaseException):
    pass


def _objects(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob('*') if path.is_file()}


def _stage(tmp_path, monkeypatch, *, writer, cut, foreign_artifact=False, replacement_fields=None):
    document = base._document_with_before_hook()
    hooks = next(node for node in document['nodes'] if node['type'] == 'hooks_pool')['config']['hooks']
    hooks[0]['event'] = 'after_tool_call'
    hooks.append({**deepcopy(hooks[0]), 'id': 'second', 'order': 20})
    effects, invocations = [], []

    def hook_runner(*, binding, inputs, **_kwargs):
        invocations.append({'id': binding.hook['id'], 'inputs': deepcopy(inputs)})
        return {'decision': 'replace', 'result': {
            'visible': 'safe-' + binding.hook['id'], **(replacement_fields or {})}}

    sessions = JsonFileSessionStore(tmp_path / 'sessions')

    def stack(client, guard):
        return base._stack(tmp_path, memory_runtime=KernelMemoryRuntime.from_config(store=sessions),
            client=client, guard=guard, tool_calls=effects, document=document,
            hook_workflow_runner=hook_runner)

    with ExecutionRuntime(sessions).scope('interaction-session') as guard:
        store, handler = stack(base._StrictOpenAIClient([base._tool_response()]), guard)
        runner = DurableWorkflowRunner(store, {'agent': handler})
        runner.create({})
        assert runner.run().status == 'waiting'
    active = store.load().state['active']
    owner, method = {
        'raw': (HookRawSealStore, 'record'),
        'phase-one': (HookPhaseJournal, 'record'),
        'phase-two': (HookPhaseJournal, 'record'),
        'artifact': (ArtifactService, 'artifactize_tool_result'),
        'receipt': (DurableToolBoundary, 'persist_prepared_result'),
    }[cut]
    real = getattr(owner, method)

    def interrupted(*args, **kwargs):
        if foreign_artifact:
            # Constructed corruption control: the operation claims a result
            # inconsistent with the successfully recorded hook decisions.
            args = (args[0], {'foreign': 'not the pinned decision'})
        result = real(*args, **kwargs)
        if cut == 'phase-two' and len(invocations) != 2:
            return result
        raise _Cut(cut)

    with ExecutionRuntime(sessions).scope('interaction-session') as guard:
        store, handler = stack(base._StrictOpenAIClient([]), guard)
        runner = DurableWorkflowRunner(store, {'agent': handler})
        with monkeypatch.context() as old:
            old.setattr(SkeletonAfterToolHookDispatcher, '__call__', _writer(writer))
            old.setattr(owner, method, interrupted)
            with pytest.raises(_Cut):
                runner.resume_kernel_interaction(active['wait']['id'], False,
                    node_invocation_id=active['node_invocation_id'])
    return sessions, stack, store, effects, invocations


def _resume(sessions, stack):
    client = base._StrictOpenAIClient([base._final_response()])
    with ExecutionRuntime(sessions).scope('interaction-session') as guard:
        store, handler = stack(client, guard)
        runner = DurableWorkflowRunner(store, {'agent': handler})
        result = runner.recover()
        assert result.status == 'waiting'
        active = store.load().state['active']
        result = runner.resume_kernel_interaction(active['wait']['id'], False,
            node_invocation_id=active['node_invocation_id'])
        if result.status == 'result_received':
            result = runner.run()
        if result.status == 'completed':
            assert runner.run().status == 'completed'
    return store, result, client


@pytest.mark.parametrize('writer,cut,historical', [
    ('legacy', 'raw', False),
    ('legacy', 'phase-one', False),
    ('legacy', 'phase-two', False),
    ('legacy', 'artifact', True),
    ('legacy', 'receipt', True),
    ('protected', 'raw', False),
    ('protected', 'phase-two', False),
    ('protected', 'artifact', False),
])
def test_resume_pinned_denial_writers_without_repeating_or_rewriting(tmp_path, monkeypatch, writer, cut, historical):
    sessions, stack, old, effects, invocations = _stage(tmp_path, monkeypatch, writer=writer, cut=cut)
    old_events = [event.to_dict() for event in old.journal.capture_snapshot().events]
    old_objects = _objects(tmp_path / 'objects')
    store, result, client = _resume(sessions, stack)
    assert result.status == 'completed'
    # The standalone core helper reaches End; host-owned root-memory commit
    # is exercised by the separate production-host qualification.
    assert result.outputs == {'answer': 'approved result'}
    assert effects == []
    assert [entry['id'] for entry in invocations] == ['policy', 'second']
    assert invocations[1]['inputs']['raw_result'] == {}
    events = store.journal.capture_snapshot().events
    assert [event.to_dict() for event in events[:len(old_events)]] == old_events
    current_objects = _objects(tmp_path / 'objects')
    assert {name: current_objects[name] for name in old_objects} == old_objects
    receipts = [event for event in events if event.event_type == 'tool_result'
        and event.payload.get('call_id') == 'approval-call']
    assert len(receipts) == 1
    payload = _thaw_json(receipts[0].payload)
    expected = {'visible': 'safe-second'}
    if not historical:
        expected['denied'] = True
    assert payload['result'] == expected
    assert payload['execution_subject']['approval_state'] == 'denied'
    native = [item for item in client.requests[0]['input']
        if item.get('type') == 'function_call_output' and item.get('call_id') == 'approval-call']
    assert len(client.requests) == len(native) == 1
    descriptor = json.loads(native[0]['output'])
    content = json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()
    assert descriptor['preview'].encode() == content
    assert descriptor['content_sha256'] == payload['result_sha256'] == sha256(content).hexdigest()


def test_unrelated_phase_input_cannot_be_admitted_as_legacy_denial(tmp_path, monkeypatch):
    sessions, stack, old, effects, invocations = _stage(tmp_path, monkeypatch,
        writer='legacy', cut='phase-two')
    event = next(event for event in old.journal.capture_snapshot().events
        if event.event_type == 'skeleton.hook_phase.receipt' and event.payload.get('hook_id') == 'second')
    payload = _thaw_json(event.payload)
    payload['input_sha256'] = '0' * 64
    identity = {key: value for key, value in payload.items() if key != 'decision'}
    operation_id = 'skeleton.hook_phase.' + _digest(identity)
    old.journal.append(request=JournalAppendRequest(event_id='event.' + operation_id,
        event_type=event.event_type, attempt=event.attempt,
        operation=build_operation_ref(operation_id, domain=event.event_type, payload=payload), payload=payload))
    _store, result, client = _resume(sessions, stack)
    assert result.status == 'outcome_unknown'
    assert effects == [] and len(invocations) == 2 and client.requests == []


def test_foreign_result_claim_is_not_silently_reused_or_overwritten(tmp_path, monkeypatch):
    sessions, stack, _old, effects, invocations = _stage(tmp_path, monkeypatch,
        writer='legacy', cut='artifact', foreign_artifact=True)
    objects = _objects(tmp_path / 'objects')
    _store, result, client = _resume(sessions, stack)
    assert result.status == 'outcome_unknown'
    assert effects == [] and len(invocations) == 2 and client.requests == []
    current = _objects(tmp_path / 'objects')
    assert {name: current[name] for name in objects} == objects


@pytest.mark.parametrize('legacy_value', [None, False, 1])
def test_claimed_legacy_business_marker_keeps_its_exact_json_type(tmp_path, monkeypatch, legacy_value):
    sessions, stack, _old, effects, invocations = _stage(tmp_path, monkeypatch,
        writer='legacy', cut='artifact', replacement_fields={'denied': legacy_value})
    old_objects = _objects(tmp_path / 'objects')
    store, result, client = _resume(sessions, stack)
    assert result.status == 'completed'
    assert effects == [] and len(invocations) == 2
    receipt = next(event for event in store.journal.capture_snapshot().events
        if event.event_type == 'tool_result' and event.payload.get('call_id') == 'approval-call')
    content = json.dumps({'visible': 'safe-second', 'denied': legacy_value},
        sort_keys=True, separators=(',', ':')).encode()
    assert receipt.payload['result_sha256'] == sha256(content).hexdigest()
    assert receipt.payload['execution_subject']['approval_state'] == 'denied'
    native = next(item for item in client.requests[0]['input']
        if item.get('type') == 'function_call_output' and item.get('call_id') == 'approval-call')
    assert json.loads(native['output'])['preview'].encode() == content
    current = _objects(tmp_path / 'objects')
    assert {name: current[name] for name in old_objects} == old_objects
