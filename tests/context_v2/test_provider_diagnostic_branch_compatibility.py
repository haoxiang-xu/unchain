"""Read both release branches' durable diagnostics without changing lease hashes."""

import copy
import hashlib
import json

import pytest

from unchain.context import ContextRuntime  # Initialize the provider/context entry point.
from unchain.journal import AttemptRef, GenerationRef, OperationRef
from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic
from unchain.providers.request_lease import (
    ProviderRequestLease, ProviderRequestLeaseCoordinator,
    ProviderRequestStatus, ProviderRequestSubject, ProviderRequestTransitionError,
)
from unchain.persistence import SQLiteContextV2Store


HTTP_V1 = {
    'schema': 'unchain.provider_failure_diagnostic.v1', 'http_status': 400,
    'provider_code': 'invalid_request_error', 'parameter': 'tools[0].parameters',
}
RESPONSE_V2 = {
    'schema': 'unchain.provider_failure_diagnostic.v2', 'http_status': None,
    'provider_code': 'MALFORMED_FUNCTION_CALL', 'parameter': '',
}
HTTP_V2 = {
    'schema': 'unchain.provider_failure_diagnostic.v2', 'http_status': 404,
    'provider_code': '', 'parameter': '', 'provider_status': 'NOT_FOUND',
    'replacement_model': 'gemini-3.8-flash',
}
UNCERTAINTY = {
    'schema': 'unchain.provider_uncertainty_diagnostic.v1',
    'reason': 'stream_incomplete', 'phase': 'reading_response',
    'http_status': None, 'provider_code': '',
}
ATTEMPT = AttemptRef(GenerationRef('diagnostic-compat', 'generation-compat'), 'attempt-compat')
SUBJECT = ProviderRequestSubject(ATTEMPT, 0, 'a' * 64, 'primary', 0)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def lease_wire(kind):
    wire = {
        'schema': 'unchain.provider_request_lease.v2', 'subject': SUBJECT.to_dict(),
        'route_sha256': 'b' * 64, 'status': 'started', 'revision': 2,
        'visible_output': False, 'retryable': False, 'classification': '',
        'operation': OperationRef('compat-annotation', 'c' * 64).to_dict(),
        'result_binding': None, 'predecessor_sha256': None,
    }
    if kind == 'uncertainty':
        wire.update(schema='unchain.provider_request_lease.v4', uncertainty_diagnostic=copy.deepcopy(UNCERTAINTY))
    elif kind != 'started':
        wire.update(schema='unchain.provider_request_lease.v4' if kind == 'http_v2' else 'unchain.provider_request_lease.v3', status='failed', classification='non_retryable', failure_diagnostic=copy.deepcopy({'http_v1': HTTP_V1, 'response': RESPONSE_V2, 'http_v2': HTTP_V2}[kind]))
    return wire


@pytest.mark.parametrize('wire', [HTTP_V1, RESPONSE_V2, HTTP_V2], ids=['legacy_http_v1', '390_response_v2', '386_http_v2'])
def test_both_branches_diagnostics_keep_exact_canonical_bytes(wire):
    restored = ProviderFailureDiagnostic.from_dict(copy.deepcopy(wire))
    assert canonical(restored.to_dict()) == canonical(wire)


@pytest.mark.parametrize('kind', ['started', 'http_v1', 'response', 'http_v2', 'uncertainty'])
def test_both_branches_leases_keep_exact_bytes_and_hash(kind):
    wire = lease_wire(kind)
    for read in [ProviderRequestLease.from_dict, ProviderRequestLease.from_durable_dict]:
        restored = read(copy.deepcopy(wire))
        assert canonical(restored.to_dict()) == canonical(wire)
        assert restored.lease_sha256 == hashlib.sha256(canonical(wire)).hexdigest()


@pytest.mark.parametrize('wire', [
    dict(RESPONSE_V2, provider_status='NOT_FOUND', replacement_model='gemini-3.8-flash'),
    dict(RESPONSE_V2, schema=HTTP_V1['schema']),
    dict(RESPONSE_V2, http_status=503),
    dict(RESPONSE_V2, provider_code='UNAVAILABLE'),
    dict(HTTP_V2, http_status=None),
    dict(HTTP_V2, provider_status='', replacement_model=''),
    dict(HTTP_V2, provider_code='MALFORMED_FUNCTION_CALL'),
    dict(HTTP_V2, schema=HTTP_V1['schema']),
    dict(HTTP_V2, private_body='PRIVATE'),
    {k: v for k, v in HTTP_V2.items() if k != 'replacement_model'},
])
def test_diagnostic_schema_collisions_do_not_accept_hybrid_records(wire):
    with pytest.raises((TypeError, ValueError)):
        ProviderFailureDiagnostic.from_dict(wire)


def invalid_lease_wires():
    response, extended, uncertain = (lease_wire(k) for k in ['response', 'http_v2', 'uncertainty'])
    return [
        dict(response, schema=extended['schema']),
        dict(extended, schema=response['schema']),
        dict(extended, failure_diagnostic=copy.deepcopy(HTTP_V1)),
        dict(uncertain, schema=response['schema']),
        dict(uncertain, failure_diagnostic=copy.deepcopy(HTTP_V2)),
        dict(extended, uncertainty_diagnostic=copy.deepcopy(UNCERTAINTY)),
        dict(uncertain, status='failed', classification='non_retryable'),
        dict(extended, status='started', classification=''),
        dict(uncertain, retryable=True),
        dict(response, schema='unchain.provider_request_lease.v2'),
        dict(uncertain, schema='unchain.provider_request_lease.v99'),
    ]


@pytest.mark.parametrize('wire', invalid_lease_wires())
def test_lease_schema_collision_requires_exact_payload_and_status(wire):
    with pytest.raises((TypeError, ValueError)):
        ProviderRequestLease.from_durable_dict(wire)


@pytest.mark.parametrize('kind', ['http_v1', 'response', 'http_v2', 'uncertainty'])
def test_sqlite_cold_read_keeps_lease_hash_and_never_authorizes_uncertain_retry(tmp_path, kind):
    def repository():
        return SQLiteContextV2Store(database_path=tmp_path / 'context.sqlite3', object_directory=tmp_path / 'objects').bind_execution(ATTEMPT.generation.execution_id)
    first = repository()
    coordinator = ProviderRequestLeaseCoordinator(first)
    started = coordinator.claim_initial(attempt=ATTEMPT, iteration=0, envelope_sha256='a' * 64, route='primary', route_sha256='b' * 64, operation=OperationRef('compat-start', 'd' * 64))
    wire = lease_wire(kind)
    annotated = ProviderRequestLease.from_dict(wire)
    first.compare_and_swap(subject=SUBJECT, expected_revision=started.revision, replacement=annotated)
    reopened = repository()
    restored = reopened.load(subject=SUBJECT)
    assert restored.lease_sha256 == hashlib.sha256(canonical(wire)).hexdigest()
    assert canonical(restored.to_dict()) == canonical(wire)
    assert restored.status is (ProviderRequestStatus.STARTED if kind == 'uncertainty' else ProviderRequestStatus.FAILED)
    with pytest.raises(ProviderRequestTransitionError):
        ProviderRequestLeaseCoordinator(reopened).claim_retry(restored, operation=OperationRef('forbidden-retry', 'e' * 64))
