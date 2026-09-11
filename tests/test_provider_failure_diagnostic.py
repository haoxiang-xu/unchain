from __future__ import annotations

from types import SimpleNamespace

import pytest

from unchain.providers.failure_diagnostic import ProviderFailureDiagnostic


@pytest.mark.parametrize("status,code,parameter", [
    (400, "invalid_function_parameters", "tools[3].parameters"),
    (401, "invalid_api_key", ""),
    (403, "permission_denied", "model"),
    (404, "model_not_found", "model"),
])
def test_http_error_has_closed_safe_projection(status, code, parameter):
    error = RuntimeError("secret body and credentials must never be stringified")
    error.status_code = status
    error.body = {"error": {"code": code, "param": parameter, "message": str(error)}}
    diagnostic = ProviderFailureDiagnostic.from_exception(error)
    assert diagnostic.to_dict() == {
        "schema": "unchain.provider_failure_diagnostic.v1",
        "http_status": status, "provider_code": code, "parameter": parameter,
    }
    assert str(error) not in diagnostic.summary()
    assert ProviderFailureDiagnostic.from_dict(diagnostic.to_dict()) == diagnostic


def test_unknown_fields_and_secrets_are_not_copied():
    secret = "opaque_private_value"
    error = RuntimeError(secret)
    error.response = SimpleNamespace(status_code=400, headers={"authorization": secret})
    error.body = {"code": secret, "param": "tools[0]." + secret, "message": secret}
    diagnostic = ProviderFailureDiagnostic.from_exception(error)
    assert diagnostic.provider_code == diagnostic.parameter == ""
    assert secret not in repr(diagnostic.to_dict()) + diagnostic.summary()


@pytest.mark.parametrize("change", [
    {"schema": "unchain.provider_failure_diagnostic.v999"},
    {"message": "private"}, {"http_status": True}, {"http_status": "400"},
    {"http_status": 200}, {"provider_code": "private"},
    {"parameter": "tools[0].private"}, {"parameter": None},
])
def test_persisted_diagnostic_rejects_invalid_or_unknown_fields(change):
    payload = ProviderFailureDiagnostic(400).to_dict()
    payload.update(change)
    with pytest.raises((ValueError, TypeError)):
        ProviderFailureDiagnostic.from_dict(payload)


@pytest.mark.parametrize("status", [None, True, "400", 200])
def test_missing_http_evidence_never_invents_status(status):
    error = RuntimeError("HTTP 401 invalid_api_key")
    error.status_code = status
    assert ProviderFailureDiagnostic.from_exception(error) is None


@pytest.mark.parametrize('status', [400, 401, 403, 404, 429, 503])
def test_google_sdk_error_has_safe_http_diagnostic(status):
    from google.genai.errors import APIError

    error = APIError(status, {'error': {'code': status, 'message': 'private prompt and key'}})
    diagnostic = ProviderFailureDiagnostic.from_exception(error)
    assert diagnostic.http_status == status
    assert diagnostic.provider_code == diagnostic.parameter == ''
    assert 'private' not in repr(diagnostic.to_dict()) + diagnostic.summary()
