"""An explicit disabled Hook is retained in the graph and absent at dispatch."""
from copy import deepcopy

import pytest

from unchain.skeleton.agent_preparation import prepare_agent_graph
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hook_runtime import SkeletonHookRuntime
from unchain.skeleton.hooks_config import attached_hooks, validate_hooks_pool_config
from unchain.skeleton.workflow_runtime import PublishedWorkflow
from tests.test_skeleton_hook_runtime import _agent_document, _entry, _with_pool, _workflow


def test_omitted_enabled_remains_the_exact_legacy_entry_without_a_stored_default():
    entry = _entry()
    assert validate_hooks_pool_config({'hooks': [entry]}, ('pool',)) == (entry,)
    assert 'enabled' not in entry


@pytest.mark.parametrize('enabled', [True, False])
def test_explicit_enabled_is_a_stored_boolean_and_does_not_mutate_the_document(enabled):
    entry = {**_entry(), 'enabled': enabled}
    before = deepcopy(entry)
    assert validate_hooks_pool_config({'hooks': [entry]}, ('pool',)) == (entry,)
    assert entry == before
    plan = prepare_agent_graph(_with_pool(_agent_document(), [entry]))
    assert plan.snapshot()['nodes'][-1]['config']['hooks'] == [entry]
    assert bool(attached_hooks(plan.snapshot())) is enabled


@pytest.mark.parametrize('bad', [None, 0, 1, '', 'false', [], {}])
def test_nonboolean_enabled_is_rejected_at_its_exact_field(bad):
    with pytest.raises(SkeletonValidationError) as error:
        validate_hooks_pool_config({'hooks': [{**_entry(), 'enabled': bad}]}, ('pool',))
    assert (error.value.code, error.value.path) == ('invalid_type', ('pool', 'hooks', 0, 'enabled'))


def test_disabled_workflow_is_never_resolved_while_active_order_and_identity_are_preserved():
    disabled = {**_entry(hook_id='disabled', order=0), 'enabled': False,
                'workflow': {'workflow_id': 'unavailable-workflow', 'revision_id': 'old'}}
    legacy = _entry(hook_id='legacy', order=20)
    active = {**_entry(hook_id='active', order=10), 'enabled': True}
    plan = prepare_agent_graph(_with_pool(_agent_document(), [legacy, disabled, active]))
    requests = []
    def resolve(identity, revision):
        requests.append((identity, revision))
        assert identity != 'unavailable-workflow'
        return PublishedWorkflow(identity, revision, _workflow(identity))
    runtime = SkeletonHookRuntime(plan, resolve)
    assert [b.hook['id'] for b in runtime.bindings_for_node('left')] == ['active', 'legacy']
    assert requests == [('redaction-workflow', 'rev-1')] * 2
    assert plan.snapshot()['nodes'][-1]['config']['hooks'][1] == disabled


@pytest.mark.parametrize('phase', ['before_tool_call', 'after_tool_call'])
def test_disabled_hook_does_not_invoke_the_workflow_on_the_real_kernel_tool_path(tmp_path, phase):
    import tests.test_skeleton_kernel_interaction as base
    from unchain.execution import ExecutionRuntime
    from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    document = base._document_with_before_hook()
    hook = next(n for n in document['nodes'] if n['type'] == 'hooks_pool')['config']['hooks'][0]
    hook.update(enabled=False, event=phase, force_approval=True)
    effects = []
    callbacks = []
    def forbidden_hook(**kwargs):
        callbacks.append(kwargs)
        raise AssertionError('disabled Hook was invoked')
    session = JsonFileSessionStore(tmp_path / 'sessions')
    client = base._StrictOpenAIClient([base._tool_response(), base._final_response()])
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        store, handler = base._stack(tmp_path, memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=client, guard=guard, tool_calls=effects, document=document,
            hook_workflow_runner=forbidden_hook, tool_requires_confirmation=False)
        runner = DurableWorkflowRunner(store, {'agent': handler})
        runner.create({})
        assert runner.run().status == 'completed'
    assert callbacks == []
    assert effects == ['approved value']
    assert len(client.requests) == 2


def test_actual_runtime_advertises_and_returns_a_detached_closed_enabled_contract():
    import unchain.skeleton.hooks_config as config
    from unchain.runtime.runtime_protocol import runtime_protocol_manifest
    contract = getattr(config, 'hook_entry_contract', lambda: None)()
    assert contract == {'schema': 'unchain.skeleton_hook_entry.v1',
                        'required_fields': ['event', 'force_approval', 'id', 'order', 'tool_name', 'workflow'],
                        'optional_fields': ['enabled'], 'omitted_enabled': True}
    contract['optional_fields'].clear()
    assert config.hook_entry_contract()['optional_fields'] == ['enabled']
    protocol = next(p for p in runtime_protocol_manifest()['protocols'] if p['id'] == 'skeleton_execution')
    assert 'hook_enable_v1' in protocol['features']


def test_disabling_a_hook_never_removes_the_tools_own_approval_requirement(tmp_path):
    import tests.test_skeleton_kernel_interaction as base
    from unchain.execution import ExecutionRuntime
    from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    document = base._document_with_before_hook()
    next(n for n in document['nodes'] if n['type'] == 'hooks_pool')['config']['hooks'][0]['enabled'] = False
    effects = []
    session = JsonFileSessionStore(tmp_path / 'sessions')
    client = base._StrictOpenAIClient([base._tool_response()])
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        store, handler = base._stack(tmp_path, memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=client, guard=guard, tool_calls=effects, document=document,
            tool_requires_confirmation=True)
        runner = DurableWorkflowRunner(store, {'agent': handler}); runner.create({})
        assert runner.run().status == 'waiting'
        assert runner.run().status == 'waiting'
    assert effects == []
    assert len(client.requests) == 1


def test_waiting_run_keeps_enabled_snapshot_while_a_new_run_can_use_disabled_definition(tmp_path):
    import tests.test_skeleton_kernel_interaction as base
    from unchain.execution import ExecutionRuntime
    from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    document = base._document_with_before_hook(force_approval=True)
    next(n for n in document['nodes'] if n['type'] == 'hooks_pool')['config']['hooks'][0]['enabled'] = True
    callbacks, effects = [], []
    def policy(**kwargs):
        callbacks.append(deepcopy(kwargs['inputs']))
        return {'decision': 'continue', 'require_approval': True}
    session = JsonFileSessionStore(tmp_path / 'sessions')
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        first, handler = base._stack(tmp_path, memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=base._StrictOpenAIClient([base._tool_response()]), guard=guard, tool_calls=effects,
            document=document, hook_workflow_runner=policy, tool_requires_confirmation=False)
        runner = DurableWorkflowRunner(first, {'agent': handler}); runner.create({})
        assert runner.run().status == 'waiting'
    fixed = first.plan.snapshot()
    changed = deepcopy(document)
    next(n for n in changed['nodes'] if n['type'] == 'hooks_pool')['config']['hooks'][0]['enabled'] = False
    wait = first.load().state['active']
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        resumed, handler = base._stack(tmp_path, memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=base._StrictOpenAIClient([base._final_response()]), guard=guard, tool_calls=effects,
            document=fixed, hook_workflow_runner=policy, tool_requires_confirmation=False)
        runner = DurableWorkflowRunner(resumed, {'agent': handler})
        runner.resume_kernel_interaction(wait['wait']['id'], True, node_invocation_id=wait['node_invocation_id'])
        assert runner.run().status == 'completed'
        assert runner.run().status == 'completed'
    assert len(callbacks) == 1
    assert effects == ['approved value']
    fresh_root = tmp_path / 'fresh'; fresh_root.mkdir()
    fresh_session = JsonFileSessionStore(fresh_root / 'sessions')
    with ExecutionRuntime(fresh_session).scope('interaction-session') as guard:
        fresh, handler = base._stack(fresh_root, memory_runtime=KernelMemoryRuntime.from_config(store=fresh_session),
            client=base._StrictOpenAIClient([base._tool_response(), base._final_response()]), guard=guard,
            tool_calls=effects, document=changed, hook_workflow_runner=policy, tool_requires_confirmation=False)
        runner = DurableWorkflowRunner(fresh, {'agent': handler}); runner.create({})
        assert runner.run().status == 'completed'
    assert len(callbacks) == 1
    assert effects == ['approved value', 'approved value']
