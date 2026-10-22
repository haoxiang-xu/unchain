"""The live oversized-read sequence through the real Tool.execute boundary."""
import hashlib
import json

import pytest

from unchain.journal import ResourceRef
from unchain.context_content import encode_context_content_locator, validate_context_content_error
from unchain.tools.output_management import ToolOutputManager
from .test_memory_toolkit_security import FakeContext, normal_toolkit


def setup_reader():
    context = FakeContext()
    context.payload = b'x' * 12663
    context.total_bytes = len(context.payload)
    ref = ResourceRef('artifact', 'live-models-result', 1)
    context.disclosed.add(ref)
    toolkit, *_ = normal_toolkit(context=context)
    config = ToolOutputManager.active_runtime_config_for_toolkit(toolkit, attempt_id='live-recovery')
    manager = ToolOutputManager.from_runtime_config(config, attempt_id='live-recovery')
    return toolkit.tools['context_content_read'], context, encode_context_content_locator(ref), manager, config


def project(tool, manager, config, ref, *, offset=0, limit=8192, call_id='read'):
    result = tool.execute({'ref':ref, 'offset':offset, 'limit':limit})
    raw = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return manager.project(raw, full_output_ref=ResourceRef('artifact',call_id,1).to_dict(),
        digest=hashlib.sha256(raw).hexdigest(), content_bytes=len(raw), call_id=call_id,
        requested_policy=manager.resolve_policy_for_tool(config,tool_name=tool.name).name).payload


@pytest.mark.parametrize('limit', [20000,50000,0,-1,True,'20000'])
def test_invalid_live_limit_has_safe_actionable_feedback_and_corrected_page(limit):
    tool, context, ref, manager, config = setup_reader()
    error = project(tool, manager, config, ref, limit=limit, call_id='invalid')
    assert error == {'schema_version':'unchain.context_content_error.v1','trust':'UNTRUSTED_DATA',
                     'code':'CONTEXT_READ_LIMIT_INVALID_USE_INTEGER_1_TO_8192'}
    assert validate_context_content_error(error)==error
    assert context.calls==[]
    first = project(tool,manager,config,ref,call_id='first')
    assert first['page_bytes']==8192 and first['next_offset']==8192
    second = project(tool,manager,config,ref,offset=8192,call_id='second')
    assert second['eof'] is True
    assert (first['content']['text']+second['content']['text']).encode()==context.payload


@pytest.mark.parametrize('arguments', [{'limit':20000},{'limit':50000},{'offset':-1},{'limit':True},{'limit':'20000'}])
def test_invalid_range_enters_guard_and_corrected_request_resets(arguments):
    tool, context, ref, manager, config = setup_reader()
    for index in range(3):
        result=project(tool,manager,config,ref,**arguments,call_id=f'bad-{index}')
    assert result['code']=='CONTEXT_READ_NO_PROGRESS'
    assert context.calls==[]
    assert project(tool,manager,config,ref,call_id='corrected')['page_bytes']==8192


@pytest.mark.parametrize('provider',['openai','anthropic','gemini','ollama'])
def test_provider_native_reader_schema_advertises_bounds(provider):
    tool, *_ = setup_reader()
    schema=tool.to_provider_json(provider)
    if provider=='ollama':schema=schema['function']
    params=schema.get('parameters',schema.get('input_schema'))
    assert params['properties']['limit']['minimum']==1
    assert params['properties']['limit']['maximum']==8192
    assert '8192' in params['properties']['limit']['description']
    assert params['properties']['offset']['minimum']==0
    assert params['properties']['offset']['maximum']==33554432
    if provider!='gemini':assert params['properties']['limit']['default']==8192


@pytest.mark.parametrize('message', [
    'secret host path /private/provider-key',
    'limit must be between 1 and 8192: secret suffix',
    'CONTEXT_READ_LIMIT_INVALID_USE_INTEGER_1_TO_8192',
])
def test_arbitrary_reader_exceptions_never_become_corrective_instructions(message):
    tool, _, _, manager, config = setup_reader()
    raw = json.dumps({'tool':tool.name, 'error':message}).encode()
    result = manager.project(
        raw, full_output_ref=ResourceRef('artifact','unknown-error',1).to_dict(),
        digest=hashlib.sha256(raw).hexdigest(), content_bytes=len(raw),
        call_id='unknown-error', requested_policy='context_page',
    ).payload
    assert result == {'schema_version':'unchain.context_content_error.v1',
                      'trust':'UNTRUSTED_DATA','code':'CONTEXT_CONTENT_READ_FAILED'}


@pytest.mark.parametrize('offset', [-1,33554433,True,'0'])
def test_invalid_offset_feedback_is_actionable_without_reading_storage(offset):
    tool, context, ref, manager, config = setup_reader()
    result = project(tool,manager,config,ref,offset=offset)
    assert result['code']=='CONTEXT_READ_OFFSET_INVALID_USE_INTEGER_0_TO_33554432'
    assert validate_context_content_error(result)==result
    assert context.calls==[]
