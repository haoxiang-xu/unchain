import httpx
import pytest

from unchain.toolkits.builtin.core import web_fetch as web


@pytest.mark.parametrize("status", [304, 400, 401, 403, 404, 429, 500, 503])
def test_failed_fetch_retains_bounded_body_and_status(monkeypatch, status):
    client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(
        status, text="<html><body>Access denied: request blocked. " + "x" * 20_000 + "</body></html>",
        headers={"content-type": "text/html", "retry-after": "30"}, request=request,
    ))
    monkeypatch.setattr(web.httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setattr(web, "validate_public_url", lambda url: (url, None))
    result, content = web.WebFetchService().fetch("https://example.com/blocked")
    assert result["ok"] is False
    assert result["status_code"] == status
    assert result["error"] == f"HTTP {status}"
    assert "Access denied" in result["result"]
    assert result["returned_chars"] == len(result["result"]) <= 4096
    assert result["truncated"] is True
    assert result["next_offset"] is None
    assert result["retryable"] is (status == 429 or status >= 500)
    assert result["retry_after"] == "30"
    assert result["retry_advice"]
    assert content is None
    assert set(result) == {
        "ok", "url", "final_url", "host", "status_code", "content_type",
        "file_kind", "result", "content_length", "returned_chars", "truncated",
        "next_offset", "cached", "redirect", "skipped", "error", "retryable",
        "retry_after", "retry_advice",
    }
