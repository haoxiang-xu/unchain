"""Pinned real SDK, HTTP parsing and durable SQLite response-outcome regressions."""
from __future__ import annotations

import json
import sqlite3

import httpx
import pytest
from google import genai

from unchain.context.provider_execution import ContextProviderTurnExecutionService, official_provider_transport_target_sha256
from unchain.journal import AttemptRef, GenerationRef
from unchain.persistence import SQLiteContextV2Store
from unchain.providers import GeminiModelIO, ModelTurnRequest
from unchain.providers.durable_turn_runtime import DurableProviderTurnMode, DurableProviderTurnTerminalError, DurableProviderTurnUncertainError
from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic
from unchain.retry import RetryConfig
from unchain.tools import Toolkit

PRIVATE = 'private response must not become a diagnostic'
STOP = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'complete'}]}, 'finishReason': 'STOP'}]}
PARTIAL = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'partial'}]}}]}


def sse(*chunks):
    return ''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)


def harness(tmp_path, response, *, cleanup_fault=None):
    sends, closes = [], []

    def handler(request):
        sends.append(request.content)
        return response(len(sends))

    def factory(**kwargs):
        options = kwargs.pop('http_options')
        options['client_args'] = {'transport': httpx.MockTransport(handler)}
        client = genai.Client(**kwargs, http_options=options)
        original_close = client.close
        closed = False

        def close():
            nonlocal closed
            if closed:
                return
            closed = True
            original_close()
            closes.append('client')
            if cleanup_fault == 'client':
                raise RuntimeError(PRIVATE)

        client.close = close
        if cleanup_fault == 'stream':
            original_send = client.models.generate_content_stream

            class Stream:
                def __init__(self, native):
                    self.native = native

                def __iter__(self):
                    return iter(self.native)

                def close(self):
                    self.native.close()
                    closes.append('stream')
                    raise RuntimeError(PRIVATE)

            client.models.generate_content_stream = lambda **kw: Stream(original_send(**kw))
        return client

    attempt = AttemptRef(GenerationRef('outcome-execution', 'outcome-generation'), 'outcome-attempt')
    model = GeminiModelIO(model='gemini-3.6-flash', api_key='offline-fixture', client_factory=factory)

    def run(max_retries=2):
        # Recreate store/service so recovery genuinely reloads disk records.
        store = SQLiteContextV2Store(database_path=tmp_path/'journal.sqlite3', object_directory=tmp_path/'objects')
        service = ContextProviderTurnExecutionService(attempt=attempt, store=store.bind_execution('outcome-execution'), mode=DurableProviderTurnMode.ENFORCE, transport_target_sha256=official_provider_transport_target_sha256(), sleep=lambda _s: None)
        return service.fetch_prepared(model_io=model, request=ModelTurnRequest(messages=[{'role': 'user', 'content': 'hello'}], run_id=attempt.attempt_id, toolkit=Toolkit()), retry_config=RetryConfig(max_retries=max_retries, base_delay_ms=0, max_delay_ms=0))

    def records():
        with sqlite3.connect(tmp_path/'journal.sqlite3') as db:
            leases = [json.loads(r[0]) for r in db.execute('SELECT lease_json FROM provider_request_lease_revisions ORDER BY rowid')]
            count = db.execute('SELECT COUNT(*) FROM provider_turn_result_receipts').fetchone()[0]
        return leases, count

    return run, sends, closes, records


def stream(body):
    return httpx.Response(200, headers={'Content-Type': 'text/event-stream'}, content=body)


@pytest.mark.parametrize('reason', ['SAFETY', 'MALFORMED_FUNCTION_CALL', 'RECITATION', 'MISSING_THOUGHT_SIGNATURE'])
def test_known_finish_is_terminal_and_survives_reopen(tmp_path, reason):
    run, sends, _, records = harness(tmp_path, lambda _n: stream(sse({'candidates': [{'finishReason': reason, 'finishMessage': PRIVATE}]})))
    for _ in range(2):
        with pytest.raises(DurableProviderTurnTerminalError) as caught:
            run()
        assert caught.value.diagnostic.http_status is None
        assert caught.value.diagnostic.provider_code == reason
        assert reason in str(caught.value)
    assert len(sends) == 1
    leases, count = records()
    assert leases[-1]['status'] == 'failed' and not leases[-1]['retryable']
    assert count == 0 and PRIVATE not in json.dumps(leases)


@pytest.mark.parametrize('body,reason', [
    (sse({'promptFeedback': {'blockReason': 'SAFETY', 'blockReasonMessage': PRIVATE}}), 'PROMPT_BLOCKED'),
    (sse({'candidates': [{'finishReason': 'STOP'}]}), 'EMPTY_RESPONSE'),
    (sse({'candidates': [{'content': {'role': 'model', 'parts': [{'text': PRIVATE, 'thought': True}]}, 'finishReason': 'STOP'}]}), 'EMPTY_RESPONSE'),
])
def test_explicit_empty_or_blocked_response_is_known_failure(tmp_path, body, reason):
    run, sends, _, records = harness(tmp_path, lambda _n: stream(body))
    with pytest.raises(DurableProviderTurnTerminalError) as caught:
        run()
    assert caught.value.diagnostic.provider_code == reason
    assert len(sends) == 1 and records()[1] == 0
    assert PRIVATE not in str(caught.value)


@pytest.mark.parametrize('body', [
    sse(PARTIAL),
    sse({'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'partial'}]}, 'finishReason': 'FINISH_REASON_UNSPECIFIED'}]}),
    sse(PARTIAL)+ 'data: {invalid\n\n',
    sse(PARTIAL, {'error': {'code': 503, 'status': 'UNAVAILABLE', 'message': PRIVATE}}),
])
def test_incomplete_or_interrupted_stream_never_completes_or_resends(tmp_path, body):
    run, sends, _, records = harness(tmp_path, lambda _n: stream(body))
    for _ in range(2):
        with pytest.raises(DurableProviderTurnUncertainError):
            run()
    leases, count = records()
    assert len(sends) == 1 and count == 0
    assert leases[-1]['status'] == 'started'


@pytest.mark.parametrize('fault', ['stream', 'client'])
@pytest.mark.parametrize('status', [200, 429, 503])
def test_cleanup_cannot_replace_success_or_http_failure(tmp_path, fault, status, caplog):
    def response(ordinal):
        if status != 200 and ordinal == 1:
            return httpx.Response(status, json={'error': {'code': status, 'status': 'UNAVAILABLE' if status == 503 else 'RESOURCE_EXHAUSTED', 'message': PRIVATE}})
        return stream(sse(STOP))
    run, sends, closes, records = harness(tmp_path, response, cleanup_fault=fault)
    assert run().final_text == 'complete'
    expected_sends = 1 if status == 200 else 2
    assert len(sends) == expected_sends
    assert 'client' in closes
    assert run().final_text == 'complete' and len(sends) == expected_sends
    leases, count = records()
    assert leases[-1]['status'] == 'completed' and count == 1
    assert PRIVATE not in caplog.text and PRIVATE not in json.dumps(leases)


def test_google_http_status_survives_without_raw_body(tmp_path):
    run, sends, _, records = harness(tmp_path, lambda _n: httpx.Response(404, json={'error': {'code': 404, 'status': 'NOT_FOUND', 'message': PRIVATE}}))
    for _ in range(2):
        with pytest.raises(DurableProviderTurnTerminalError) as caught:
            run()
        assert caught.value.diagnostic.provider_status == 'NOT_FOUND'
        assert caught.value.diagnostic.provider_code == ''
    assert len(sends) == 1 and PRIVATE not in json.dumps(records()[0])


def test_response_diagnostic_is_closed_and_versioned():
    wire = {'schema': 'unchain.provider_failure_diagnostic.v2', 'http_status': None, 'provider_code': 'SAFETY', 'parameter': ''}
    assert ProviderFailureDiagnostic.from_dict(wire).to_dict() == wire
    for changed in [dict(wire, extra=PRIVATE), dict(wire, provider_code=PRIVATE), dict(wire, http_status=200), dict(wire, parameter='messages'), dict(wire, schema='unchain.provider_failure_diagnostic.v1')]:
        with pytest.raises(ValueError):
            ProviderFailureDiagnostic.from_dict(changed)
    old = ProviderFailureDiagnostic(400, 'invalid_request_error').to_dict()
    assert old['schema'] == 'unchain.provider_failure_diagnostic.v1'
    assert ProviderFailureDiagnostic.from_dict(old).to_dict() == old


def test_real_sdk_signed_tool_continuation_with_durable_recovery(tmp_path):
    import base64
    from unchain.tools.messages import GeminiMessageBuilder

    signature = base64.b64encode(b'opaque-fixture-signature').decode()
    tool_response = {'candidates': [{'content': {'role': 'model', 'parts': [
        {'text': 'Inspect first.'},
        {'functionCall': {'id': 'signed-call', 'name': 'probe', 'args': {'query': 'durable'}}, 'thoughtSignature': signature},
    ]}, 'finishReason': 'STOP'}]}
    sends = []

    def handler(request):
        sends.append(json.loads(request.content))
        return stream(sse(tool_response if len(sends) == 1 else STOP))

    def factory(**kwargs):
        options = kwargs.pop('http_options')
        options['client_args'] = {'transport': httpx.MockTransport(handler)}
        return genai.Client(**kwargs, http_options=options)

    toolkit = Toolkit()
    toolkit.register(lambda query: {'result': query}, name='probe')
    model = GeminiModelIO(model='gemini-3.6-flash', api_key='offline-fixture', client_factory=factory)
    attempt = AttemptRef(GenerationRef('signed-execution', 'signed-generation'), 'signed-attempt')

    def run(messages, iteration):
        store = SQLiteContextV2Store(database_path=tmp_path/'journal.sqlite3', object_directory=tmp_path/'objects')
        service = ContextProviderTurnExecutionService(attempt=attempt, store=store.bind_execution('signed-execution'), mode=DurableProviderTurnMode.ENFORCE, transport_target_sha256=official_provider_transport_target_sha256(), sleep=lambda _s: None)
        return service.fetch_prepared(model_io=model, request=ModelTurnRequest(messages=messages, iteration=iteration, run_id=attempt.attempt_id, toolkit=toolkit), retry_config=RetryConfig(max_retries=0))

    first = run([{'role': 'user', 'content': 'call probe'}], 0)
    assert first.tool_calls[0].arguments == {'query': 'durable'}
    response = GeminiMessageBuilder().build_tool_result_message(tool_call=first.tool_calls[0], tool_result={'result': 'durable'})
    continuation = [*first.provider_replay_frame['items'], response]
    assert run(continuation, 1).final_text == 'complete'
    native = next(c for c in sends[1]['contents'] if c['role'] == 'model')
    assert native['parts'][0] == {'text': 'Inspect first.'}
    assert native['parts'][1]['thoughtSignature'] == signature
    assert native['parts'][1]['functionCall'] == {'id': 'signed-call', 'name': 'probe', 'args': {'query': 'durable'}}
    assert sends[1]['contents'][-1]['parts'][0]['functionResponse']['id'] == 'signed-call'
    assert run(continuation, 1).final_text == 'complete'
    assert len(sends) == 2
