"""Strict display contract fed by an actual cold-resume Journal receipt."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from unchain.events.bridge import RuntimeEventBridge
from unchain.events.subagent_settlement import validate_subagent_settlement_projection
from unchain.journal import EventCursor

SOURCE=json.loads((Path(__file__).parent/'fixtures/skeleton_cold_subagent_settlement.json').read_text())

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def projection(status='completed'):
    event=SOURCE['settlement_event'];payload=event['payload']
    body={'schema':'unchain.skeleton_child_completion.v1','status':payload['status'],
        'outputs':deepcopy(payload['outputs']),'error':payload['error'],'summary':payload['summary']}
    content=canonical(body)
    assert sha256(content).hexdigest()==payload['result_sha256'] and len(content)==payload['result_bytes']
    if status!='completed':
        body.update(status=status,outputs=None,error='child '+status,summary='child '+status)
        content=canonical(body)
    return {'schema':'unchain.skeleton_subagent_settlement_projection.v1',
        'cursor':EventCursor(event['store_seq'],event['event_id']).to_dict(),
        'consumer_attempt':deepcopy(payload['consumer_attempt']),'source_attempt':deepcopy(payload['source_attempt']),
        'call_id':payload['call_id'],'tool_name':payload['tool_name'],'status':status,
        'result':{'full_output_ref':deepcopy(payload['full_output_ref']),
            'content_bytes':len(content),'content_sha256':sha256(content).hexdigest(),
            'preview':content.decode('utf-8'),'inline':True}}

def test_valid_projection_has_independent_values_and_actual_digest():
    value=projection();parsed=validate_subagent_settlement_projection(value,execution_id=SOURCE['execution_id'])
    assert parsed==value and parsed is not value
    parsed['result']['preview']='changed'
    assert value['result']['preview']!='changed'

@pytest.mark.parametrize('status',['completed','failed','cancelled','outcome_unknown'])
def test_v4_preserves_terminal_kind_and_binds_original_call(status):
    value=projection(status)
    bridge=RuntimeEventBridge(session_id=SOURCE['execution_id'],root_run_id=SOURCE['root_run_id'])
    result=bridge.normalize({'type':'subagent_tool_settled','settlement':value})
    assert len(result)==1
    event=result[0]
    assert event.type==('run.completed' if status=='completed' else 'run.failed')
    assert event.payload['status']==status and event.payload['settlement']==value
    assert event.run_id==value['source_attempt']['attempt_id']
    assert event.links.parent_run_id==value['consumer_attempt']['attempt_id']
    assert event.links.tool_call_id==value['call_id']

@pytest.mark.parametrize('mutation',[
    lambda v:v.update(extra=True),lambda v:v.pop('cursor'),lambda v:v.update(schema='future'),
    lambda v:v['cursor'].update(store_seq=True),lambda v:v['cursor'].update(extra=True),
    lambda v:v['source_attempt'].update(attempt_id=v['consumer_attempt']['attempt_id']),
    lambda v:v['source_attempt']['generation'].update(generation_id='foreign'),
    lambda v:v.update(call_id=' '),lambda v:v.update(tool_name=[]),lambda v:v.update(status='waiting'),
    lambda v:v['result'].update(extra=True),lambda v:v['result'].update(content_bytes=True),
    lambda v:v['result'].update(content_bytes=1.5),lambda v:v['result'].update(content_sha256='0'*64),
    lambda v:v['result'].update(inline=False),lambda v:v['result'].update(preview='{}'),
    lambda v:v['result']['full_output_ref'].update(kind='message'),
    lambda v:v['result']['full_output_ref'].update(fragment='other'),
    lambda v:v['result'].update(inline='yes'),lambda v:v['result'].update(preview='x'*8193),
])
def test_malformed_or_contradictory_receipt_is_refused(mutation):
    value=projection();mutation(value)
    with pytest.raises((TypeError,ValueError)):
        validate_subagent_settlement_projection(value,execution_id=SOURCE['execution_id'])
    bridge=RuntimeEventBridge(session_id=SOURCE['execution_id'],root_run_id=SOURCE['root_run_id'])
    assert bridge.normalize({'type':'subagent_tool_settled','settlement':value})==[]

def test_foreign_execution_cannot_be_normalized_into_this_chat():
    with pytest.raises(ValueError):
        validate_subagent_settlement_projection(projection(),execution_id='another-chat')
    assert RuntimeEventBridge(session_id='another-chat').normalize({'type':'subagent_tool_settled','settlement':projection()})==[]

def test_large_result_keeps_only_a_bounded_noninline_projection():
    value=projection();value['result'].update(content_bytes=20000,content_sha256='a'*64,preview='x'*8192,inline=False)
    assert validate_subagent_settlement_projection(value)==value

def test_body_status_cannot_be_swapped_even_after_recomputing_its_digest():
    value=projection();body=json.loads(value['result']['preview']);body['status']='failed'
    content=canonical(body);value['result'].update(content_bytes=len(content),content_sha256=sha256(content).hexdigest(),preview=content.decode())
    with pytest.raises(ValueError):validate_subagent_settlement_projection(value)
