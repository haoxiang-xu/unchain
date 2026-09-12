from __future__ import annotations

import html
import ipaddress
import json
import re
import socket
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

_USER_AGENT = "unchain-code-toolkit/1.0"
_FETCH_TIMEOUT_SECONDS = 60.0
_MAX_CONTENT_BYTES = 10 * 1024 * 1024
_MAX_REDIRECTS = 10
_CACHE_TTL_SECONDS = 15 * 60
_CACHE_MAX_BYTES = 50 * 1024 * 1024
_DEFAULT_EXTRACT_MAX_INPUT_CHARS = 100_000
_TEXTUAL_CONTENT_TYPES = {
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/javascript",
    "application/x-javascript",
    "application/ecmascript",
    "application/x-www-form-urlencoded",
}
_SKIPPED_BINARY_TYPES = {
    "application/pdf": "pdf",
}
_PRIVATE_HOST_SUFFIXES = (
    ".internal",
    ".intranet",
    ".lan",
    ".local",
    ".localhost",
    ".home",
    ".corp",
)


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _normalize_content_type(content_type: str | None) -> str:
    raw = str(content_type or "").strip().lower()
    if not raw:
        return ""
    return raw.split(";", 1)[0].strip()


def _is_public_ip(address: str) -> bool:
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return not (
        parsed.is_private
        or parsed.is_loopback
        or parsed.is_link_local
        or parsed.is_multicast
        or parsed.is_reserved
        or parsed.is_unspecified
    )


def validate_public_url(url: str) -> tuple[str | None, dict[str, Any] | None]:
    if not isinstance(url, str) or not url.strip():
        return None, {"ok": False, "error": "url is required", "url": url}

    try:
        parsed = urlsplit(url.strip())
    except Exception:
        return None, {"ok": False, "error": "url is invalid", "url": url}

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return None, {"ok": False, "error": "url scheme must be http or https", "url": url}
    if parsed.username or parsed.password:
        return None, {"ok": False, "error": "url must not include credentials", "url": url}

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return None, {"ok": False, "error": "url hostname is required", "url": url}
    if hostname == "localhost" or hostname.endswith(_PRIVATE_HOST_SUFFIXES):
        return None, {"ok": False, "error": "private or localhost URLs are not allowed", "url": url}
    if "." not in hostname:
        try:
            ipaddress.ip_address(hostname)
        except ValueError:
            return None, {"ok": False, "error": "url hostname must be a public domain", "url": url}

    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        ipaddress.ip_address(hostname)
        addresses = {hostname}
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return None, {"ok": False, "error": "url hostname could not be resolved", "url": url}
        addresses = {entry[4][0] for entry in infos if isinstance(entry[4], tuple) and entry[4]}
        if not addresses:
            return None, {"ok": False, "error": "url hostname could not be resolved", "url": url}

    if any(not _is_public_ip(address) for address in addresses):
        return None, {"ok": False, "error": "private or non-public network targets are not allowed", "url": url}

    normalized_scheme = "https" if scheme == "http" else scheme
    normalized_port = parsed.port
    if scheme == "http" and normalized_port == 80:
        normalized_port = None
    default_port = 443 if normalized_scheme == "https" else 80
    include_port = normalized_port is not None and normalized_port != default_port
    netloc = hostname if not include_port else f"{hostname}:{normalized_port}"
    path = parsed.path or "/"
    normalized_url = urlunsplit((normalized_scheme, netloc, path, parsed.query, ""))
    return normalized_url, None


def is_safe_redirect(original_url: str, redirect_url: str) -> bool:
    try:
        original = urlsplit(original_url)
        redirect = urlsplit(redirect_url)
    except Exception:
        return False

    if redirect.scheme.lower() != original.scheme.lower():
        return False
    if redirect.username or redirect.password:
        return False
    if (redirect.port or (443 if redirect.scheme == "https" else 80)) != (
        original.port or (443 if original.scheme == "https" else 80)
    ):
        return False

    def _strip_www(hostname: str) -> str:
        return hostname.lower().removeprefix("www.")

    return _strip_www(redirect.hostname or "") == _strip_www(original.hostname or "")


def _file_kind_for(content_type: str, body: bytes, final_url: str) -> str:
    normalized = _normalize_content_type(content_type)
    if normalized in _SKIPPED_BINARY_TYPES:
        return _SKIPPED_BINARY_TYPES[normalized]
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith("text/") or normalized in _TEXTUAL_CONTENT_TYPES:
        return "text"
    if normalized in {"application/rss+xml", "application/atom+xml"}:
        return "text"
    if b"\x00" in body[:8192]:
        return "binary"
    lowered_url = final_url.lower()
    if lowered_url.endswith(".pdf"):
        return "pdf"
    return "text"


class _MarkdownHTMLParser(HTMLParser):
    _BLOCK_TAGS = {"article", "blockquote", "div", "header", "footer", "main", "nav", "p", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._href_stack: list[tuple[str, int]] = []
        self._list_depth = 0
        self._in_pre = False
        self._in_code = False

    def _append(self, text: str) -> None:
        if text:
            self._parts.append(text)

    def _newline(self, count: int = 1) -> None:
        if not self._parts:
            return
        existing = len(re.findall(r"\n+$", self._parts[-1])) if self._parts else 0
        needed = max(0, count - existing)
        if needed:
            self._parts.append("\n" * needed)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_map = {key.lower(): value for key, value in attrs}
        if tag in self._BLOCK_TAGS:
            self._newline(2)
            return
        if tag == "br":
            self._newline(1)
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._newline(2)
            level = int(tag[1])
            self._append("#" * level + " ")
            return
        if tag in {"ul", "ol"}:
            self._newline(2)
            self._list_depth += 1
            return
        if tag == "li":
            self._newline(1)
            self._append("  " * max(0, self._list_depth - 1) + "- ")
            return
        if tag == "pre":
            self._newline(2)
            self._append("```\n")
            self._in_pre = True
            return
        if tag == "code" and not self._in_pre:
            self._append("`")
            self._in_code = True
            return
        if tag == "a":
            href = str(attrs_map.get("href") or "").strip()
            self._href_stack.append((href, len(self._parts)))

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._BLOCK_TAGS:
            self._newline(2)
            return
        if tag in {"ul", "ol"}:
            self._list_depth = max(0, self._list_depth - 1)
            self._newline(2)
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._newline(2)
            return
        if tag == "pre":
            self._newline(1)
            self._append("```")
            self._newline(2)
            self._in_pre = False
            return
        if tag == "code" and self._in_code:
            self._append("`")
            self._in_code = False
            return
        if tag == "a" and self._href_stack:
            href, part_index = self._href_stack.pop()
            if href and part_index < len(self._parts):
                anchor_text = "".join(self._parts[part_index:]).strip()
                if href and href != anchor_text:
                    self._append(f" ({href})")

    def handle_data(self, data: str) -> None:
        if not data:
            return
        if self._in_pre:
            self._append(data)
            return
        normalized = re.sub(r"\s+", " ", data)
        if normalized.strip():
            self._append(html.unescape(normalized))

    def get_markdown(self) -> str:
        raw = "".join(self._parts)
        raw = raw.replace("\xa0", " ")
        raw = re.sub(r"[ \t]+\n", "\n", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_markdown(content: str) -> str:
    parser = _MarkdownHTMLParser()
    parser.feed(content)
    parser.close()
    return parser.get_markdown()


def decode_response_body(body: bytes, content_type: str) -> tuple[str, str]:
    normalized = _normalize_content_type(content_type)
    text = body.decode("utf-8", errors="replace")
    if normalized in {"text/html", "application/xhtml+xml"}:
        return html_to_markdown(text), normalized
    if normalized == "application/json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text, normalized
        return json.dumps(parsed, ensure_ascii=False, indent=2), normalized
    return text, normalized


def _read_response_body(response: httpx.Response) -> bytes:
    collected = bytearray()
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        remaining = _MAX_CONTENT_BYTES - len(collected)
        if remaining <= 0:
            raise ValueError(f"response exceeded {_MAX_CONTENT_BYTES} bytes")
        collected.extend(chunk[:remaining])
        if len(chunk) > remaining:
            raise ValueError(f"response exceeded {_MAX_CONTENT_BYTES} bytes")
    return bytes(collected)


@dataclass
class _CacheEntry:
    payload: dict[str, Any]
    content: str
    size: int
    expires_at: float


class _TTLPageCache:
    def __init__(self, *, max_bytes: int, ttl_seconds: int) -> None:
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._current_bytes = 0

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._current_bytes -= entry.size

    def get(self, key: str) -> _CacheEntry | None:
        self._purge_expired()
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def set(self, key: str, *, payload: dict[str, Any], content: str) -> None:
        self._purge_expired()
        size = max(1, len(content.encode("utf-8", errors="replace")))
        if size > self._max_bytes:
            return
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._current_bytes -= existing.size
        while self._entries and self._current_bytes + size > self._max_bytes:
            _, evicted = self._entries.popitem(last=False)
            self._current_bytes -= evicted.size
        self._entries[key] = _CacheEntry(
            payload=dict(payload),
            content=content,
            size=size,
            expires_at=time.time() + self._ttl_seconds,
        )
        self._current_bytes += size


def _read_error_response_body(response: httpx.Response) -> bytes:
    # Read one sentinel byte to report truncation without downloading an
    # arbitrarily large error page. Error bodies are diagnostics, not content.
    collected = bytearray()
    limit = 16_384 + 1
    for chunk in response.iter_bytes():
        collected.extend(chunk[:limit - len(collected)])
        if len(collected) >= limit:
            break
    return bytes(collected)


class WebFetchService:
    def __init__(self) -> None:
        self._cache = _TTLPageCache(max_bytes=_CACHE_MAX_BYTES, ttl_seconds=_CACHE_TTL_SECONDS)

    def fetch(self, url: str) -> tuple[dict[str, Any], str | None]:
        normalized_url, error = validate_public_url(url)
        if error is not None or normalized_url is None:
            result = dict(error or {})
            result.setdefault("mode", "raw")
            result.setdefault("cached", False)
            result.setdefault("redirect", None)
            result.setdefault("skipped", False)
            result.setdefault("content_length", 0)
            result.setdefault("returned_chars", 0)
            result.setdefault("next_offset", None)
            result.setdefault("file_kind", "")
            return result, None

        cached = self._cache.get(normalized_url)
        if cached is not None:
            payload = dict(cached.payload)
            payload["cached"] = True
            return payload, cached.content

        try:
            payload, content = self._request(normalized_url)
        except Exception as exc:
            return (
                {
                    "ok": False,
                    "url": url,
                    "final_url": normalized_url,
                    "host": urlsplit(normalized_url).hostname or "",
                    "status_code": 0,
                    "content_type": "",
                    "file_kind": "",
                    "result": "",
                    "content_length": 0,
                    "returned_chars": 0,
                    "truncated": False,
                    "next_offset": None,
                    "cached": False,
                    "redirect": None,
                    "skipped": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                None,
            )

        if payload.get("ok") and not payload.get("skipped") and isinstance(content, str):
            cache_payload = dict(payload)
            cache_payload["cached"] = False
            self._cache.set(normalized_url, payload=cache_payload, content=content)
        return payload, content

    def _request(self, normalized_url: str) -> tuple[dict[str, Any], str | None]:
        with httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(_FETCH_TIMEOUT_SECONDS),
            headers={
                "Accept": "text/html, text/plain, application/json, application/xml;q=0.9, */*;q=0.8",
                "User-Agent": _USER_AGENT,
            },
        ) as client:
            current_url = normalized_url
            for _ in range(_MAX_REDIRECTS + 1):
                with client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise RuntimeError("redirect response missing Location header")
                        redirect_url = urljoin(current_url, location)
                        if is_safe_redirect(current_url, redirect_url):
                            current_url = redirect_url
                            continue
                        return (
                            {
                                "ok": False,
                                "url": normalized_url,
                                "final_url": current_url,
                                "host": urlsplit(current_url).hostname or "",
                                "status_code": response.status_code,
                                "content_type": "",
                                "file_kind": "",
                                "result": "",
                                "content_length": 0,
                                "returned_chars": 0,
                                "truncated": False,
                                "next_offset": None,
                                "cached": False,
                                "redirect": {
                                    "requires_rerun": True,
                                    "redirect_url": redirect_url,
                                    "status_code": response.status_code,
                                },
                                "skipped": False,
                                "error": "",
                            },
                            None,
                        )
                    body = (
                        _read_response_body(response)
                        if 200 <= response.status_code < 300
                        else _read_error_response_body(response)
                    )
                    final_url = str(response.url)
                    final_host = response.url.host or ""
                    status_code = response.status_code
                    content_type = _normalize_content_type(response.headers.get("content-type"))
                    retry_after = str(response.headers.get("retry-after") or "")[:128]
                    break
            else:
                raise RuntimeError(f"too many redirects (>{_MAX_REDIRECTS})")

        file_kind = _file_kind_for(content_type, body, final_url)
        if not 200 <= status_code < 300:
            content, normalized_type = (
                decode_response_body(body[:16_384], content_type)
                if file_kind == "text"
                else ("", content_type)
            )
            excerpt = content[:4096]
            retryable = status_code in {408, 425, 429} or status_code >= 500
            return (
                {
                    "ok": False,
                    "url": normalized_url,
                    "final_url": final_url,
                    "host": final_host,
                    "status_code": status_code,
                    "content_type": normalized_type,
                    "file_kind": file_kind,
                    "result": excerpt,
                    "content_length": len(content),
                    "returned_chars": len(excerpt),
                    "truncated": len(body) > 16_384 or len(content) > len(excerpt),
                    "next_offset": None,
                    "cached": False,
                    "redirect": None,
                    "skipped": file_kind != "text",
                    "error": f"HTTP {status_code}",
                    "retryable": retryable,
                    "retry_after": retry_after,
                    "retry_advice": (
                        "Temporary HTTP failure. Respect Retry-After; retry only within "
                        "the bounded fetch budget, then use another source or explain the failure."
                        if retryable else
                        "Do not repeat this URL unchanged. Use another source or explain "
                        "that the site denied or could not serve the request."
                    ),
                },
                None,
            )
        if file_kind != "text":
            return (
                {
                    "ok": False,
                    "url": normalized_url,
                    "final_url": final_url,
                    "host": final_host,
                    "status_code": status_code,
                    "content_type": content_type,
                    "file_kind": file_kind,
                    "result": "",
                    "content_length": len(body),
                    "returned_chars": 0,
                    "truncated": False,
                    "next_offset": None,
                    "cached": False,
                    "redirect": None,
                    "skipped": True,
                    "error": "",
                },
                None,
            )

        content, normalized_type = decode_response_body(body, content_type)
        return (
            {
                "ok": True,
                "url": normalized_url,
                "final_url": final_url,
                "host": final_host,
                "status_code": status_code,
                "content_type": normalized_type,
                "file_kind": file_kind,
                "result": "",
                "content_length": len(content),
                "returned_chars": 0,
                "truncated": False,
                "next_offset": None,
                "cached": False,
                "redirect": None,
                "skipped": False,
                "error": "",
            },
            content,
        )


def run_extract_model(
    *,
    url: str,
    content: str,
    prompt: str,
    extract_model_config: dict[str, Any],
    execution_context: Any = None,
) -> str:
    from ....agent.model_io import ModelIOFactoryRegistry
    from ....kernel.model_io import ModelTurnRequest
    from ....kernel.run_ledger import (
        provider_call_route,
        record_model_turn,
        record_unobserved_model_attempt,
        request_sha256,
    )
    from ....tools.toolkit import Toolkit

    provider = str(extract_model_config.get("provider") or "").strip().lower()
    model = str(extract_model_config.get("model") or "").strip()
    api_key = extract_model_config.get("api_key")
    if not provider or not model:
        raise ValueError("extract_model runtime config requires non-empty provider and model")

    payload = extract_model_config.get("payload")
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("extract_model payload must be a dict when provided")
    request_payload = dict(payload)
    request_payload.setdefault("store", False)

    max_input_chars = extract_model_config.get("max_input_chars", _DEFAULT_EXTRACT_MAX_INPUT_CHARS)
    try:
        resolved_max_input_chars = max(1000, int(max_input_chars))
    except (TypeError, ValueError):
        resolved_max_input_chars = _DEFAULT_EXTRACT_MAX_INPUT_CHARS

    truncated_content = content[:resolved_max_input_chars]
    user_prompt = (
        "You are extracting information from a fetched web page.\n"
        "Use only the provided page content. If the page is insufficient, say so briefly.\n\n"
        f"URL: {url}\n\n"
        f"User request:\n{prompt.strip()}\n\n"
        "Page content:\n"
        "---\n"
        f"{truncated_content}\n"
        "---"
    )

    registry = ModelIOFactoryRegistry()
    model_io = registry.create(provider=provider, model=model, api_key=api_key if isinstance(api_key, str) else None)
    request = ModelTurnRequest(
        messages=[
            {
                "role": "system",
                "content": "Extract only the information requested from the provided web page content.",
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        payload=request_payload,
        toolkit=Toolkit(),
        emit_stream=False,
        run_id="web_fetch_extract",
        iteration=0,
    )
    run_state = getattr(execution_context, "run_state", None)
    iteration = max(0, int(getattr(execution_context, "iteration", 0) or 0))
    request_digest = request_sha256(
        state=run_state,
        payload=request.payload,
        toolkit=request.toolkit,
        response_format=request.response_format,
        openai_text_format=request.openai_text_format,
        provider=provider,
        model=model,
        messages=request.messages,
    )
    provider_turn_ownership = getattr(
        getattr(run_state, "run_ledger", None),
        "provider_turn_ownership",
        None,
    )
    if provider_turn_ownership is not None:
        call_id = str(getattr(execution_context, "call_id", "") or "")
        if not call_id:
            raise RuntimeError(
                "durable web extract requires its stable tool call_id"
            )
        from ....retry import RetryConfig

        turn = provider_turn_ownership.fetch_turn(
            state=run_state,
            model_io=model_io,
            request=request,
            occurrence_id=f"web_extract:{call_id}",
            purpose="web_extract",
            iteration=iteration,
            request_sha256=request_digest,
            retry_config=RetryConfig(),
            provider=provider,
            model=model,
        )
        if turn.tool_calls:
            raise RuntimeError("extract model attempted to call tools")
        result = (turn.final_text or _last_assistant_text(turn.assistant_messages)).strip()
        if not result:
            raise RuntimeError("extract model returned empty output")
        return result
    occurred_at = (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )
    try:
        turn = model_io.fetch_turn(request)
    except Exception:
        completed_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        if run_state is not None:
            record_unobserved_model_attempt(
                run_state,
                iteration=iteration,
                retry_ordinal=0,
                purpose="web_extract",
                request_digest=request_digest,
                payload=request_payload,
                started_at=occurred_at,
                completed_at=completed_at,
                route=provider_call_route(provider),
                status="uncertain",
                provider=provider,
                model=model,
            )
        raise
    if run_state is not None:
        completed_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        record_model_turn(
            run_state,
            turn,
            iteration=iteration,
            retry_ordinal=0,
            purpose="web_extract",
            request_digest=request_digest,
            payload=request_payload,
            started_at=occurred_at,
            completed_at=completed_at,
            route=provider_call_route(provider),
            provider=provider,
            model=model,
        )
    if turn.tool_calls:
        raise RuntimeError("extract model attempted to call tools")
    result = (turn.final_text or _last_assistant_text(turn.assistant_messages)).strip()
    if not result:
        raise RuntimeError("extract model returned empty output")
    return result


__all__ = [
    "WebFetchService",
    "run_extract_model",
]
