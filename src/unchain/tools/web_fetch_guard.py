"""Bound repeated failed fetches using the persisted native tool transcript."""

import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def web_fetch_failure_limit(messages: list[dict[str, Any]]) -> str | None:
    # Reuse the native OpenAI/Anthropic/Gemini/Ollama result reader. The full
    # transcript survives suspend/restart, so no volatile counter or new durable
    # schema can accidentally reset this budget during approval recovery.
    from ..kernel.microcompact import _collect_result_records, _is_tool_result_message

    start = max((
        index for index, message in enumerate(messages)
        if message.get("role") == "user" and not _is_tool_result_message(message)
    ), default=-1)
    records = _collect_result_records(messages[start + 1:], tool_calls=[])
    failures: dict[str, int] = {}
    total = 0
    seen: set[str] = set()
    for record in records:
        if record.tool_name != "web_fetch":
            continue
        if record.call_id:
            if record.call_id in seen:
                continue
            seen.add(record.call_id)
        payload = record.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ok") is True:
            failures.clear()
            total = 0
            continue
        # An admitted redirect requires a fresh fetch; it is not a failed page.
        if payload.get("ok") is not False or payload.get("redirect"):
            continue
        url = payload.get("final_url") or payload.get("url")
        if not isinstance(url, str) or not url:
            continue
        try:
            parts = urlsplit(url)
            key = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))
        except ValueError:
            key = url
        failures[key] = failures.get(key, 0) + 1
        total += 1
        status = payload.get("status_code")
    if total >= 6 or any(count >= 3 for count in failures.values()):
        detail = f"HTTP {status}" if isinstance(status, int) and status > 0 else "fetch errors"
        return (
            f"I stopped web fetching after repeated failures ({detail}): "
            "the retry limit is 3 failures for the same URL or 6 failed fetches "
            "without a successful page, checked after each tool batch. The failing "
            "sources could not be retrieved, so I cannot verify the requested "
            "information from them. Provide accessible links or the page contents to continue."
        )
    return None
