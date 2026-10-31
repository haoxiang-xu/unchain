"""Exact set membership at the real Skeleton/Context V2 Hook boundary."""
from copy import deepcopy
import pytest
from unchain.skeleton.errors import SkeletonValidationError
from unchain.skeleton.hooks_config import validate_hooks_pool_config
from tests.test_skeleton_hook_runtime import _entry

EXPECTED = {'schema': 'unchain.skeleton_hook_tool_set.v1', 'field': 'tool_name',
            'legacy_types': ['null', 'string'], 'set_type': 'array', 'minimum_size': 1,
            'duplicates': 'reject', 'names': 'nonblank_exact', 'normalization': 'none'}

@pytest.mark.parametrize('names', [['read'], ['read', 'write'], ['write ', 'Write', 'wr*']])
def test_nonempty_unique_exact_sets_are_preserved_without_normalization(names):
    entry={**_entry(), 'tool_name': deepcopy(names)}; original=deepcopy(entry)
    assert validate_hooks_pool_config({'hooks':[entry]}, ('pool',)) == (original,)
    assert entry == original

@pytest.mark.parametrize('bad,code,suffix', [([], 'invalid_value', ()), ([''], 'invalid_value', (0,)),
    (['  '], 'invalid_value', (0,)), (['a','a'], 'duplicate_name', (1,)),
    (['a',None], 'invalid_type', (1,)), ([False], 'invalid_type', (0,)),
    ([['read']], 'invalid_type', (0,)), ([{'name':'read'}], 'invalid_type', (0,))])
def test_bad_set_is_rejected_at_the_real_set_or_member_path(bad,code,suffix):
    with pytest.raises(SkeletonValidationError) as rejected:
        validate_hooks_pool_config({'hooks':[{**_entry(),'tool_name':bad}]}, ('pool',))
    assert (rejected.value.code,rejected.value.path)==(code,('pool','hooks',0,'tool_name')+suffix)


def test_new_contract_and_feature_are_real_and_enable_contract_stays_compatible():
    import unchain.skeleton.hooks_config as module
    from unchain.runtime.runtime_protocol import runtime_protocol_manifest
    getter=getattr(module,'hook_tool_set_contract',None)
    assert callable(getter)
    answer=getter();assert answer==EXPECTED;answer['legacy_types'].clear();assert getter()==EXPECTED
    assert module.hook_entry_contract()=={'schema':'unchain.skeleton_hook_entry.v1',
        'required_fields':['event','force_approval','id','order','tool_name','workflow'],
        'optional_fields':['enabled'],'omitted_enabled':True}
    protocol=next(p for p in runtime_protocol_manifest()['protocols'] if p['id']=='skeleton_execution')
    assert protocol['minor']==4 and {'hook_tool_set_v1', 'hook_embedded_workflow_v1'} <= set(protocol['features'])

@pytest.mark.parametrize('phase', ['before_tool_call','after_tool_call'])
@pytest.mark.parametrize('names,expected_calls', [(['dangerous_tool','another'],1),(['other'],0),(['dangerous_*'],0)])
def test_selected_and_unselected_names_on_the_actual_kernel_path(tmp_path,phase,names,expected_calls):
    import tests.test_skeleton_kernel_interaction as base
    from unchain.execution import ExecutionRuntime
    from unchain.memory import JsonFileSessionStore,KernelMemoryRuntime
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    doc=base._document_with_before_hook();entry=doc['nodes'][-1]['config']['hooks'][0]
    entry.update(tool_name=names,event=phase)
    effects=[];calls=[]
    def policy(**kwargs):
        calls.append(deepcopy(kwargs['inputs']))
        return {'decision':'continue','require_approval':False} if phase=='before_tool_call' else {'decision':'continue'}
    session=JsonFileSessionStore(tmp_path/'sessions');client=base._StrictOpenAIClient([base._tool_response(),base._final_response()])
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        store,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=session),client=client,
            guard=guard,tool_calls=effects,document=doc,hook_workflow_runner=policy,tool_requires_confirmation=False)
        runner=DurableWorkflowRunner(store,{'agent':handler});runner.create({});assert runner.run().status=='completed'
    assert len(calls)==expected_calls and effects==['approved value'] and len(client.requests)==2
    if calls: assert calls[0]['tool_name']=='dangerous_tool'


def test_frozen_waiting_set_survives_a_later_definition_change_and_before_is_not_replayed(tmp_path):
    import tests.test_skeleton_kernel_interaction as base
    from unchain.execution import ExecutionRuntime
    from unchain.memory import JsonFileSessionStore,KernelMemoryRuntime
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    doc=base._document_with_before_hook(force_approval=True)
    doc['nodes'][-1]['config']['hooks'][0]['tool_name']=['dangerous_tool','another']
    effects=[];calls=[]
    def policy(**kwargs):
        calls.append(deepcopy(kwargs['inputs']));return {'decision':'continue','require_approval':True}
    session=JsonFileSessionStore(tmp_path/'sessions')
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        first,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=base._StrictOpenAIClient([base._tool_response()]),guard=guard,tool_calls=effects,document=doc,
            hook_workflow_runner=policy,tool_requires_confirmation=False)
        runner=DurableWorkflowRunner(first,{'agent':handler});runner.create({});assert runner.run().status=='waiting'
    fixed=first.plan.snapshot();changed=deepcopy(doc);changed['nodes'][-1]['config']['hooks'][0]['tool_name']=['other']
    active=first.load().state['active']
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        resumed,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=base._StrictOpenAIClient([base._final_response()]),guard=guard,tool_calls=effects,document=fixed,
            hook_workflow_runner=policy,tool_requires_confirmation=False)
        runner=DurableWorkflowRunner(resumed,{'agent':handler});runner.resume_kernel_interaction(active['wait']['id'],True,node_invocation_id=active['node_invocation_id'])
        assert runner.run().status=='completed';assert runner.run().status=='completed'
    assert len(calls)==1 and effects==['approved value']
    fresh=tmp_path/'new';fresh.mkdir();new_session=JsonFileSessionStore(fresh/'sessions')
    with ExecutionRuntime(new_session).scope('interaction-session') as guard:
        store,handler=base._stack(fresh,memory_runtime=KernelMemoryRuntime.from_config(store=new_session),
            client=base._StrictOpenAIClient([base._tool_response(),base._final_response()]),guard=guard,tool_calls=effects,
            document=changed,hook_workflow_runner=policy,tool_requires_confirmation=False)
        runner=DurableWorkflowRunner(store,{'agent':handler});runner.create({});assert runner.run().status=='completed'
    assert len(calls)==1 and effects==['approved value','approved value']

@pytest.mark.parametrize('names', [['dangerous_tool','another'],['other']])
def test_set_matching_does_not_remove_the_tools_native_approval(tmp_path,names):
    import tests.test_skeleton_kernel_interaction as base
    from unchain.execution import ExecutionRuntime
    from unchain.memory import JsonFileSessionStore,KernelMemoryRuntime
    from unchain.skeleton.durable_execution import DurableWorkflowRunner
    doc=base._document_with_before_hook();doc['nodes'][-1]['config']['hooks'][0]['tool_name']=names
    effects=[]
    def policy(**_kwargs):return {'decision':'continue','require_approval':False}
    session=JsonFileSessionStore(tmp_path/'sessions');client=base._StrictOpenAIClient([base._tool_response()])
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        store,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=session),client=client,
            guard=guard,tool_calls=effects,document=doc,hook_workflow_runner=policy,tool_requires_confirmation=True)
        runner=DurableWorkflowRunner(store,{'agent':handler});runner.create({});assert runner.run().status=='waiting'
        assert runner.run().status=='waiting'
    assert effects==[] and len(client.requests)==1


def test_disabled_set_stays_structurally_valid_but_creates_no_hook_binding():
    from unchain.skeleton.agent_preparation import prepare_agent_graph
    from unchain.skeleton.hooks_config import attached_hooks
    from tests.test_skeleton_hook_runtime import _agent_document,_with_pool
    entry={**_entry(),'enabled':False,'tool_name':['read','write']}
    plan=prepare_agent_graph(_with_pool(_agent_document(),[entry]))
    assert plan.snapshot()['nodes'][-1]['config']['hooks']==[entry]
    assert attached_hooks(plan.snapshot())=={}
