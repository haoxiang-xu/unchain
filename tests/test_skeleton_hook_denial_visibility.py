"""Authority remains visible across after-hook redaction.

Uses the existing real Skeleton Kernel/Context/approval stack. The SDK reply and
Hook Workflow runner are finite test adapters: this is not the final real-store
published Workflow/desktop acceptance.
"""
from copy import deepcopy
from hashlib import sha256
import json

import pytest
import tests.test_skeleton_kernel_interaction as base
from unchain.execution import ExecutionRuntime
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime
from unchain.skeleton.durable_execution import DurableWorkflowRunner

def canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()

@pytest.mark.parametrize('approved',[False,True])
@pytest.mark.parametrize('mode',['none','continue','replace','forge-denied-false','replace-twice'])
def test_approval_fact_survives_after_hook_without_disabling_result_replacement(tmp_path,approved,mode):
    document=base._document() if mode=='none' else base._document_with_before_hook()
    if mode!='none':
        next(n for n in document['nodes'] if n['type']=='hooks_pool')['config']['hooks'][0]['event']='after_tool_call'
    if mode=='replace-twice':
        hooks=next(n for n in document['nodes'] if n['type']=='hooks_pool')['config']['hooks']
        hooks.append({**deepcopy(hooks[0]),'id':'second','order':20})
    hook_inputs=[];effects=[]
    def run_hook(*,binding,inputs,parent_context,parent_store):
        hook_inputs.append(deepcopy(inputs))
        if mode=='continue':return {'decision':'continue'}
        if mode=='replace-twice' and len(hook_inputs)==2:
            assert inputs['raw_result']=={}
            assert inputs['visible_result']==({'visible':'REDACTED','denied':True} if not approved else {'visible':'REDACTED'})
            return {'decision':'replace','result':{'visible':'FINAL'}}
        result={'visible':'REDACTED'}
        if mode=='forge-denied-false':result['denied']=False
        return {'decision':'replace','result':result}
    store=JsonFileSessionStore(tmp_path/'sessions')
    first_client=base._StrictOpenAIClient([base._tool_response()])
    with ExecutionRuntime(store).scope('interaction-session') as guard:
        first,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=store),
            client=first_client,guard=guard,tool_calls=effects,document=document,
            hook_workflow_runner=None if mode=='none' else run_hook)
        runner=DurableWorkflowRunner(first,{'agent':handler});runner.create({})
        assert runner.run().status=='waiting'
    assert effects==[] and hook_inputs==[]
    active=first.load().state['active']
    final_response=base._final_response()
    final_response.output[0]['content'][0]['text']='{"answer":"terminal summary"}'
    final_client=base._StrictOpenAIClient([final_response])
    with ExecutionRuntime(store).scope('interaction-session') as guard:
        resumed,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=store),
            client=final_client,guard=guard,tool_calls=effects,document=document,
            hook_workflow_runner=None if mode=='none' else run_hook)
        runner=DurableWorkflowRunner(resumed,{'agent':handler})
        receipt={'approved':approved,'modified_arguments':None,'reason':'no' if not approved else ''}
        assert runner.resume_kernel_interaction(active['wait']['id'],receipt,
            node_invocation_id=active['node_invocation_id']).status=='result_received'
        assert runner.run().outputs=={'answer':'terminal summary'}
        assert runner.run().outputs=={'answer':'terminal summary'}
    assert effects==(['approved value'] if approved else [])
    assert len(hook_inputs)==(0 if mode=='none' else 2 if mode=='replace-twice' else 1)
    events=resumed.journal.capture_snapshot().events
    result=next(e for e in events if e.event_type=='tool_result' and e.payload.get('tool_name')=='dangerous_tool')
    assert result.payload['execution_subject']['approval_state']==('approved' if approved else 'denied')
    if mode in ['replace','forge-denied-false','replace-twice']:
        expected={'visible':'FINAL' if mode=='replace-twice' else 'REDACTED'}
        if not approved:expected['denied']=True
        elif mode=='forge-denied-false':expected['denied']=False
    else:
        expected={'applied':'approved value'} if approved else {'denied':True,'tool':'dangerous_tool','reason':'no'}
    assert dict(result.payload['result'])==expected
    assert result.payload['result_sha256']==sha256(canonical(expected)).hexdigest()
    outputs=[x for x in final_client.requests[0]['input'] if x.get('type')=='function_call_output'
             and x.get('call_id')=='approval-call']
    assert len(outputs)==1
    model_result=json.loads(outputs[0]['output'])
    assert model_result['inline'] is True
    assert model_result['content_sha256']==sha256(canonical(expected)).hexdigest()
    assert len(final_client.requests)==1  # second terminal run did not call the SDK again


def test_after_replacement_for_a_nonapproval_tool_is_not_rewritten_from_business_data(tmp_path):
    document=base._document_with_before_hook()
    next(n for n in document['nodes'] if n['type']=='hooks_pool')['config']['hooks'][0]['event']='after_tool_call'
    calls=[]
    def run_hook(**_kwargs):
        return {'decision':'replace','result':{'denied':False,'visible':'public'}}
    session=JsonFileSessionStore(tmp_path/'sessions')
    client=base._StrictOpenAIClient([base._tool_response(),base._final_response()])
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        store,handler=base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=client,guard=guard,tool_calls=calls,document=document,
            hook_workflow_runner=run_hook,tool_requires_confirmation=False)
        runner=DurableWorkflowRunner(store,{'agent':handler});runner.create({})
        assert runner.run().status=='completed'
    result=next(e for e in store.journal.capture_snapshot().events if e.event_type=='tool_result')
    assert result.payload['execution_subject']['approval_state']=='not_required'
    assert dict(result.payload['result'])=={'denied':False,'visible':'public'}
    assert calls==['approved value']


@pytest.mark.parametrize('crash_phase',['raw_seal','phase_receipt','final_result'])
def test_denied_after_hook_recovers_without_tool_or_hook_reexecution(tmp_path,monkeypatch,crash_phase):
    from unchain.skeleton.hook_raw_seal import HookRawSealStore
    from unchain.skeleton.hook_phase_journal import HookPhaseJournal
    from unchain.context.tool_boundary import DurableToolBoundary
    class PowerLoss(BaseException):
        pass
    document=base._document_with_before_hook()
    next(n for n in document['nodes'] if n['type']=='hooks_pool')['config']['hooks'][0]['event']='after_tool_call'
    effects=[];hooks=[]
    def run_hook(**kwargs):
        hooks.append(deepcopy(kwargs['inputs']))
        return {'decision':'replace','result':{'visible':'safe'}}
    session=JsonFileSessionStore(tmp_path/'sessions')
    def stack(client,guard):
        return base._stack(tmp_path,memory_runtime=KernelMemoryRuntime.from_config(store=session),
            client=client,guard=guard,tool_calls=effects,document=document,hook_workflow_runner=run_hook)
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        first,handler=stack(base._StrictOpenAIClient([base._tool_response()]),guard)
        runner=DurableWorkflowRunner(first,{'agent':handler});runner.create({})
        assert runner.run().status=='waiting'
    active=first.load().state['active']
    owner,method=(HookRawSealStore,'record') if crash_phase=='raw_seal' else (HookPhaseJournal,'record') if crash_phase=='phase_receipt' else (DurableToolBoundary,'persist_prepared_result')
    original=getattr(owner,method)
    def interrupted(*args,**kwargs):
        result=original(*args,**kwargs)
        raise PowerLoss(crash_phase)
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        warm,handler=stack(base._StrictOpenAIClient([]),guard)
        runner=DurableWorkflowRunner(warm,{'agent':handler})
        with monkeypatch.context() as fault:
            fault.setattr(owner,method,interrupted)
            with pytest.raises(PowerLoss):
                runner.resume_kernel_interaction(active['wait']['id'],False,node_invocation_id=active['node_invocation_id'])
    client=base._StrictOpenAIClient([base._final_response()])
    with ExecutionRuntime(session).scope('interaction-session') as guard:
        cold,handler=stack(client,guard)
        runner=DurableWorkflowRunner(cold,{'agent':handler})
        recovered=runner.recover()
        # The durable interaction checkpoint remains the entry point after a
        # process dies inside its continuation. Recovery restores that wait;
        # re-presenting its same denied response consumes its existing receipt.
        assert recovered.status=='waiting'
        active=cold.load().state['active']
        assert runner.resume_kernel_interaction(active['wait']['id'],False,
            node_invocation_id=active['node_invocation_id']).status=='result_received'
        recovered=runner.run()
    assert recovered.status=='completed'
    assert effects==[] and len(hooks)==1 and len(client.requests)==1
    results=[e for e in cold.journal.capture_snapshot().events if e.event_type=='tool_result' and e.payload.get('call_id')=='approval-call']
    assert len(results)==1
    assert dict(results[0].payload['result'])=={'visible':'safe','denied':True}
    assert results[0].payload['execution_subject']['approval_state']=='denied'
