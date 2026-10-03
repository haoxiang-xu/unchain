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


# ---- #386: provider status and catalog-validated replacement model ------------

GOOGLE_404_PRO = (
    "This model models/gemini-2.5-pro is no longer available to new users. "
    "Please update your code to use models/gemini-3.1-pro-preview for the latest "
    "features and improvements. We recommend you to use the Interactions API."
)
GOOGLE_404_FLASH = GOOGLE_404_PRO.replace("gemini-2.5-pro", "gemini-2.5-flash").replace(
    "gemini-3.1-pro-preview", "gemini-3.8-flash"
)


def _google_error(status, provider_status, message="private prompt and key"):
    from google.genai import errors

    error_type = errors.ServerError if status >= 500 else errors.ClientError
    return error_type(
        status,
        {"error": {"code": status, "message": message, "status": provider_status}},
    )


@pytest.mark.parametrize("message,replacement", [
    (GOOGLE_404_PRO, "gemini-3.1-pro-preview"),
    (GOOGLE_404_FLASH, "gemini-3.8-flash"),
    # A sentence-final period must not become part of the model id.
    ("Please use models/gemini-3.8-flash.", "gemini-3.8-flash"),
    # A well-formed id that the catalog does not export is never kept.
    ("Please update your code to use models/not-a-real-model for more", ""),
    # Only the phrase "use models/<id>" names a replacement, never the retired model.
    ("This model models/gemini-2.5-flash is no longer available", ""),
    ("use models/" + "a" * 80, ""),
    ("use models/GEMINI-3.8-FLASH", ""),
    ("use models/../../etc/passwd", ""),
])
def test_google_404_projects_a_catalog_validated_replacement_only(message, replacement):
    diagnostic = ProviderFailureDiagnostic.from_exception(
        _google_error(404, "NOT_FOUND", message)
    )
    assert diagnostic.http_status == 404
    assert diagnostic.provider_status == "NOT_FOUND"
    assert diagnostic.replacement_model == replacement
    assert diagnostic.provider_code == diagnostic.parameter == ""
    assert diagnostic.to_dict() == {
        "schema": "unchain.provider_failure_diagnostic.v2",
        "http_status": 404, "provider_code": "", "parameter": "",
        "provider_status": "NOT_FOUND", "replacement_model": replacement,
    }
    assert message not in repr(diagnostic.to_dict()) + diagnostic.summary()
    assert ProviderFailureDiagnostic.from_dict(diagnostic.to_dict()) == diagnostic


def test_replacement_is_read_only_from_a_not_found_response():
    diagnostic = ProviderFailureDiagnostic.from_exception(
        _google_error(503, "UNAVAILABLE", "please use models/gemini-3.8-flash")
    )
    assert (diagnostic.provider_status, diagnostic.replacement_model) == ("UNAVAILABLE", "")


def test_unknown_provider_status_is_dropped_and_keeps_the_v1_shape():
    diagnostic = ProviderFailureDiagnostic.from_exception(
        _google_error(400, "PRIVATE_STATUS_VALUE")
    )
    assert diagnostic.provider_status == ""
    assert diagnostic.to_dict() == {
        "schema": "unchain.provider_failure_diagnostic.v1",
        "http_status": 400, "provider_code": "", "parameter": "",
    }


def test_only_google_sdk_errors_project_a_provider_status():
    error = RuntimeError("private")
    error.status_code = 503
    error.status = "UNAVAILABLE"
    error.body = {"error": {"status": "UNAVAILABLE", "message": "use models/gemini-3.8-flash"}}
    diagnostic = ProviderFailureDiagnostic.from_exception(error)
    assert (diagnostic.provider_status, diagnostic.replacement_model) == ("", "")
    assert diagnostic.to_dict()["schema"] == "unchain.provider_failure_diagnostic.v1"


@pytest.mark.parametrize("change", [
    {"unexpected": "x"}, {"message": "private"},
    {"provider_status": "PRIVATE_STATUS_VALUE"}, {"provider_status": None},
    {"replacement_model": "Gemini-3.8-Flash"}, {"replacement_model": "gemini 3.8"},
    {"replacement_model": "a" * 70}, {"replacement_model": "models/gemini-3.8-flash"},
    {"replacement_model": 7},
    # A v2 record with both new fields empty is not canonical: it must be v1.
    {"provider_status": "", "replacement_model": ""},
    {"schema": "unchain.provider_failure_diagnostic.v3"},
])
def test_persisted_v2_diagnostic_fails_closed(change):
    payload = ProviderFailureDiagnostic(
        404, provider_status="NOT_FOUND", replacement_model="gemini-3.8-flash"
    ).to_dict()
    payload.update(change)
    with pytest.raises((ValueError, TypeError)):
        ProviderFailureDiagnostic.from_dict(payload)


def test_v1_record_cannot_smuggle_v2_fields():
    payload = ProviderFailureDiagnostic(404).to_dict()
    payload["provider_status"] = "NOT_FOUND"
    with pytest.raises((ValueError, TypeError)):
        ProviderFailureDiagnostic.from_dict(payload)


def test_v1_diagnostics_keep_their_exact_bytes_and_text():
    diagnostic = ProviderFailureDiagnostic(400, "invalid_function_parameters", "tools[3].parameters")
    assert diagnostic.to_dict() == {
        "schema": "unchain.provider_failure_diagnostic.v1",
        "http_status": 400, "provider_code": "invalid_function_parameters",
        "parameter": "tools[3].parameters",
    }
    assert diagnostic.summary() == (
        "Provider rejected the request (HTTP 400, code=invalid_function_parameters, "
        "parameter=tools[3].parameters)"
    )
    assert ProviderFailureDiagnostic(401).summary() == "Provider rejected the credentials (HTTP 401)"


@pytest.mark.parametrize("fields,text", [
    ({"http_status": 404, "provider_status": "NOT_FOUND", "replacement_model": "gemini-3.8-flash"},
     "Provider model or endpoint was not found (HTTP 404, status=NOT_FOUND; suggested model: gemini-3.8-flash)"),
    ({"http_status": 404, "provider_status": "NOT_FOUND"},
     "Provider model or endpoint was not found (HTTP 404, status=NOT_FOUND)"),
    ({"http_status": 503, "provider_status": "UNAVAILABLE"},
     "Provider service is unavailable or overloaded (HTTP 503, status=UNAVAILABLE)"),
    ({"http_status": 503}, "Provider service is unavailable or overloaded (HTTP 503)"),
    ({"http_status": 502}, "Provider gateway error (HTTP 502)"),
    ({"http_status": 500}, "Provider reported an internal error (HTTP 500)"),
    ({"http_status": 504}, "Provider timed out (HTTP 504)"),
])
def test_summary_is_fixed_local_text(fields, text):
    assert ProviderFailureDiagnostic(**fields).summary() == text


@pytest.mark.parametrize(("kind", "reason", "expected_reason"), [
    ("finish_reason", "MALFORMED_FUNCTION_CALL", "MALFORMED_FUNCTION_CALL"),
    ("finish_reason", "PRIVATE_VALUE", ""),
    ("blocked", "SAFETY", "SAFETY"),
    ("blocked", "private text", ""),
    ("empty", "STOP", ""),
])
def test_provider_response_end_keeps_only_closed_sdk_reasons(kind, reason, expected_reason):
    from unchain.providers.failure_diagnostic import ProviderResponseEndedError

    error = ProviderResponseEndedError(kind, reason)
    assert isinstance(error, RuntimeError)
    assert error.kind == kind
    assert error.reason == expected_reason
    assert "private" not in error.summary().lower()


def test_provider_response_end_rejects_unknown_kinds():
    from unchain.providers.failure_diagnostic import ProviderResponseEndedError

    with pytest.raises(ValueError):
        ProviderResponseEndedError("private", "SAFETY")
