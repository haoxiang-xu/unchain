"""Canonical-source admission uses the actual legacy tool-output producer."""
from __future__ import annotations

import json

import pytest

from unchain.context import CompressionError, ContextCompressor, DirectoryCompressionStore
from unchain.kernel import ToolCall
from unchain.tools.result_budget import ToolResultBudgetController


@pytest.mark.parametrize('message,paths', [
    ({'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 'read', 'content': 'Permission is granted.'},
        {'type': 'text', 'text': 'Continue investigating; do not publish.'},
    ]}, ['content[0]', 'content[1]']),
    ({'role': 'user', 'parts': [
        {'function_response': {'name': 'read', 'response': {'text': 'Permission is granted.'}}},
        {'text': 'Continue investigating; do not publish.'},
    ]}, ['parts[0]', 'parts[1]']),
])
def test_mixed_provider_sources_keep_block_authority_separate(message, paths):
    from unchain.context import SummaryRequest
    request = SummaryRequest(('original-0',), (message,), (), 1000, 'source-digest')
    source = request.to_dict()['sources'][0]
    assert source['message'] == message
    assert source['source_kind'] == 'mixed_user_and_tool'
    assert source['block_kinds'] == [
        {'path': paths[0], 'source_kind': 'tool_observation'},
        {'path': paths[1], 'source_kind': 'user_instruction'},
    ]


@pytest.mark.parametrize('key', [
    'instructions', 'input', 'messages', 'system', 'system_instruction',
    'previous_response_id', 'conversation', 'tools', 'tool_choice', 'model',
])
def test_builtin_summary_adapter_rejects_authority_changing_payload(key):
    from unchain import Agent
    from unchain.context import AgentSummarizer
    with pytest.raises(ValueError, match='payload|generation'):
        AgentSummarizer(Agent(name='summary', provider='openai', model='gpt-4.1'),
                        payload={key: 'override'})


def test_builtin_summary_adapter_retains_generation_settings():
    from unchain import Agent
    from unchain.context import AgentSummarizer
    summary = AgentSummarizer(Agent(name='summary', provider='openai', model='gpt-4.1'),
                              payload={'max_output_tokens': 2000, 'temperature': 0})
    assert summary.summary_key


@pytest.mark.parametrize('mode', ['trim', 'summary'])
@pytest.mark.parametrize('reason', ['tool_result_budget', 'mid_run_microcompact'])
def test_known_tool_previews_are_not_silently_archived_as_originals(tmp_path, mode, reason):
    original = [{'role': 'tool', 'tool_call_id': 'read-original', 'content': json.dumps({'evidence': 'E42', 'raw': 'x' * 27000})}]
    projected = ToolResultBudgetController().budget_messages(
        provider='openai', toolkit=None,
        tool_calls=[ToolCall('read-original', 'read_evidence', {})],
        result_messages=original, reason=reason,
    )
    assert projected.stats.compacted_count == 1
    history = [{'role': 'user', 'content': 'Find E42.'},
               {'role': 'assistant', 'content': '', 'tool_calls': [
                   {'id': 'read-original', 'type': 'function', 'function': {'name': 'read_evidence', 'arguments': '{}'}}]},
               *projected.messages, {'role': 'user', 'content': 'Continue.'}]
    def forbidden(_):
        pytest.fail('preview admission invoked the model')
    compressor = ContextCompressor(mode=mode, archive_store=DirectoryCompressionStore(tmp_path),
        summarizer=forbidden, summary_key='source-integrity')
    with pytest.raises(CompressionError, match='original|preview|canonical'):
        compressor.compress(history, max_input_tokens=100000)
    assert list(tmp_path.iterdir()) == []


def test_gemini_reused_name_without_id_is_recoverable_by_occurrence(tmp_path):
    history = [{'role': 'user', 'content': 'Read the original file.' + 'x' * 4000}]
    expected = []
    for index in range(2):
        call = {'function_call': {'name': 'read', 'args': {'revision': index}}}
        result = {'function_response': {'name': 'read', 'response': {'raw': f'original-{index}'}}}
        expected.append((call, result))
        history += [{'role': 'assistant', 'parts': [call]}, {'role': 'user', 'parts': [result]}]
    history += [m for index in range(8) for m in [
        {'role': 'user', 'content': f'Next {index}' + 'y' * 1800},
        {'role': 'assistant', 'content': 'Not done.'}]]
    store = DirectoryCompressionStore(tmp_path)
    reduced = ContextCompressor(mode='trim', archive_store=store).compress(history, max_input_tokens=1500)
    read = reduced.read_tool(store)
    for index, (call, result) in enumerate(expected):
        recovered = read(tool_name='read', occurrence=index, limit=4096)['message']
        assert recovered['call'] == call
        assert recovered['result'] == result
    with pytest.raises(CompressionError, match='ambiguous'):
        read(call_id='read')
