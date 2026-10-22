from __future__ import annotations

import hashlib
import json

import pytest

from unchain.context_content import (
    canonical_context_content_result_bytes,
    present_context_output_ref,
    render_context_content_error,
    render_context_content_page,
)
from unchain.journal import ResourceRef
from unchain.memory.toolkit.models import MemoryToolContentPage
from unchain.tools.output_management import (
    TOOL_OUTPUT_MANAGEMENT_SCHEMA,
    TOOL_OUTPUT_POLICY_MAP_SCHEMA,
    ToolOutputManagementError,
    ToolOutputManager,
    ToolOutputPolicyVersionError,
    ToolOutputReadError,
)
from unchain.kernel.types import ToolCall
from unchain.tools import Toolkit, get_provider_message_builder
from unchain.tools.tool import Tool


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _context_page() -> dict[str, object]:
    data = b"ticket-382-page-sentinel"
    return render_context_content_page(
        MemoryToolContentPage(
            ref=ResourceRef("artifact", "ticket-382-source", 1),
            media_type="text/plain",
            data=data,
            offset=0,
            total_bytes=len(data),
            sha256=_digest(data),
        )
    )


def test_active_manager_projects_once_and_disables_legacy_budget():
    manager = ToolOutputManager.active_default(
        attempt_id="attempt-a",
        preview_chars=4,
        inline_chars=8,
    )
    raw = b"0123456789"
    receipt = manager.project(
        raw,
        full_output_ref={"uri": "pupu://artifact/a@1"},
        digest=_digest(raw),
        content_bytes=len(raw),
        call_id="call-a",
    )

    assert manager.legacy_budget_enabled is False
    assert receipt.payload["projection"] == "default"
    assert receipt.payload["inline"] is False
    assert receipt.payload["preview"] == "0123"
    assert receipt.metadata["projection_version"] == "v1"
    assert manager.project(
        raw,
        full_output_ref={"uri": "pupu://artifact/a@1"},
        digest=_digest(raw),
        content_bytes=len(raw),
        call_id="call-a",
    ) == receipt


def test_snapshot_is_closed_and_unknown_explicit_policy_fails_closed():
    manager = ToolOutputManager.active_default()
    snapshot = manager.runtime_snapshot()
    assert snapshot["schema"] == TOOL_OUTPUT_MANAGEMENT_SCHEMA
    assert ToolOutputManager.from_runtime_config(
        {"tool_output_management": snapshot}
    ).legacy_budget_enabled is False

    raw = b"value"
    with pytest.raises(ToolOutputPolicyVersionError):
        manager.project(
            raw,
            full_output_ref={"uri": "pupu://artifact/a@1"},
            digest=_digest(raw),
            content_bytes=len(raw),
            requested_policy="does-not-exist",
        )

    invalid = dict(snapshot)
    invalid["unexpected"] = True
    with pytest.raises(ToolOutputManagementError):
        ToolOutputManager.from_runtime_config({"tool_output_management": invalid})


def test_reserved_context_page_policy_does_not_change_default_snapshots():
    manager = ToolOutputManager.active_default()

    assert [policy["name"] for policy in manager.runtime_snapshot()["policies"]] == [
        "artifact_only",
        "default",
        "head_tail",
    ]


def test_closed_tool_policy_map_selects_each_declared_tool_policy():
    manager = ToolOutputManager.active_default()
    config = {
        "tool_output_policy_map": {
            "schema": TOOL_OUTPUT_POLICY_MAP_SCHEMA,
            "policies": {"large_search": "artifact_only"},
        }
    }

    assert manager.resolve_policy_for_tool(
        config, tool_name="large_search"
    ).name == "artifact_only"
    assert manager.resolve_policy_for_tool(config, tool_name="small_read").name == "default"

    with pytest.raises(ToolOutputManagementError):
        manager.resolve_policy_for_tool(
            {"tool_output_policy_map": {"schema": "invalid", "policies": {}}},
            tool_name="large_search",
        )


def test_active_runtime_requires_native_tool_output_policy_declarations():
    class UndeclaredTool:
        pass

    class Toolkit:
        tools = {"large_search": UndeclaredTool()}

    config = ToolOutputManager.active_runtime_config_for_toolkit(
        Toolkit(),
        attempt_id="attempt-toolkit-policy",
    )
    manager = ToolOutputManager.from_runtime_config(config)

    assert manager.resolve_policy_for_tool(
        config,
        tool_name="large_search",
    ).name == "default"
    assert "tool_output_policy_map" not in config


def test_large_default_projection_uses_the_canonical_context_locator():
    manager = ToolOutputManager.active_default(preview_chars=4, inline_chars=8)
    raw = b"0123456789"
    projection = manager.project(
        raw,
        full_output_ref=ResourceRef("artifact", "ticket-382-output", 1).to_dict(),
        digest=_digest(raw),
        content_bytes=len(raw),
    )

    assert projection.payload == {
        "schema_version": "unchain.tool_output.paged.v1",
        "projection": "paged",
        "preview": {
            "head": "0123",
            "tail": "6789",
            "omitted_bytes": 2,
        },
        "content_bytes": 10,
        "content_sha256": _digest(raw),
        "full_output_ref": (
            "unchain://context/v1/"
            "eyJmcmFnbWVudCI6IiIsImlkIjoidGlja2V0LTM4Mi1vdXRwdXQiLCJraW5kIjoi"
            "YXJ0aWZhY3QiLCJyZXZpc2lvbiI6MX0"
        ),
        "read_request": {
            "tool": "context_content_read",
            "arguments": {
                "ref": (
                    "unchain://context/v1/"
                    "eyJmcmFnbWVudCI6IiIsImlkIjoidGlja2V0LTM4Mi1vdXRwdXQiLCJraW5kIjoi"
                    "YXJ0aWZhY3QiLCJyZXZpc2lvbiI6MX0"
                ),
                "offset": 0,
                "limit": 8192,
            },
        },
    }


def test_context_page_policy_is_reserved_and_preserves_one_canonical_page():
    toolkit = Toolkit(
        {
            "context_content_read": Tool(
                name="context_content_read",
                description="read",
                func=lambda: None,
                output_policy="context_page",
            )
        }
    )
    config = ToolOutputManager.active_runtime_config_for_toolkit(
        toolkit,
        attempt_id="ticket-382-context-page",
    )
    manager = ToolOutputManager.from_runtime_config(config)
    page = _context_page()
    raw = _canonical_json(page)

    assert config["tool_output_policy_map"] == {
        "schema": TOOL_OUTPUT_POLICY_MAP_SCHEMA,
        "policies": {"context_content_read": "context_page"},
    }
    projection = manager.project(
        raw,
        full_output_ref=ResourceRef("artifact", "ticket-382-page-output", 1).to_dict(),
        digest=_digest(raw),
        content_bytes=len(raw),
        requested_policy=manager.resolve_policy_for_tool(
            config,
            tool_name="context_content_read",
        ).name,
    )
    assert projection.payload == page
    assert "full_output_ref" not in projection.payload
    assert projection.metadata["projection_policy"] == "context_page"
    assert projection.metadata["projection_bytes"] == len(raw)


def test_context_page_policy_preserves_only_the_fixed_reader_failure_shape():
    toolkit = Toolkit(
        {
            "context_content_read": Tool(
                name="context_content_read",
                description="read",
                func=lambda: None,
                output_policy="context_page",
            )
        }
    )
    config = ToolOutputManager.active_runtime_config_for_toolkit(
        toolkit,
        attempt_id="ticket-382-context-error",
    )
    manager = ToolOutputManager.from_runtime_config(config)
    error = render_context_content_error()
    raw = canonical_context_content_result_bytes(error)

    projection = manager.project(
        raw,
        full_output_ref=ResourceRef("artifact", "ticket-382-page-error", 1).to_dict(),
        digest=_digest(raw),
        content_bytes=len(raw),
        requested_policy=manager.resolve_policy_for_tool(
            config,
            tool_name="context_content_read",
        ).name,
    )
    assert projection.payload == error

    tool_error = {"error": "undisclosed ref", "tool": "context_content_read"}
    tool_error_raw = _canonical_json(tool_error)
    recovered = manager.project(
        tool_error_raw,
        full_output_ref=ResourceRef("artifact", "ticket-382-page-error", 1).to_dict(),
        digest=_digest(tool_error_raw),
        content_bytes=len(tool_error_raw),
        requested_policy=manager.resolve_policy_for_tool(
            config,
            tool_name="context_content_read",
        ).name,
    )
    assert recovered.payload == error

    no_progress = {"error": "CONTEXT_READ_NO_PROGRESS", "tool": "context_content_read"}
    no_progress_raw = _canonical_json(no_progress)
    no_progress_projection = manager.project(
        no_progress_raw,
        full_output_ref=ResourceRef("artifact", "ticket-382-page-error", 1).to_dict(),
        digest=_digest(no_progress_raw),
        content_bytes=len(no_progress_raw),
        requested_policy=manager.resolve_policy_for_tool(
            config,
            tool_name="context_content_read",
        ).name,
    )
    assert no_progress_projection.payload["code"] == "CONTEXT_READ_NO_PROGRESS"

    untrusted = dict(error)
    untrusted["message"] = "unbounded details"
    invalid_raw = _canonical_json(untrusted)
    with pytest.raises(ToolOutputManagementError, match="context content page"):
        manager.project(
            invalid_raw,
            full_output_ref=ResourceRef("artifact", "ticket-382-page-error", 1).to_dict(),
            digest=_digest(invalid_raw),
            content_bytes=len(invalid_raw),
            requested_policy=manager.resolve_policy_for_tool(
                config,
                tool_name="context_content_read",
            ).name,
        )


def test_context_page_policy_rejects_other_tools_and_noncanonical_pages():
    with pytest.raises(ToolOutputManagementError, match="reserved"):
        toolkit = Toolkit(
            {
                "untrusted_reader": Tool(
                    name="untrusted_reader",
                    description="read",
                    func=lambda: None,
                    output_policy="context_page",
                )
            }
        )
        ToolOutputManager.active_runtime_config_for_toolkit(
            toolkit,
            attempt_id="ticket-382-invalid-context-page",
        )

    toolkit = Toolkit(
        {
            "context_content_read": Tool(
                name="context_content_read",
                description="read",
                func=lambda: None,
                output_policy="context_page",
            )
        }
    )
    config = ToolOutputManager.active_runtime_config_for_toolkit(
        toolkit,
        attempt_id="ticket-382-invalid-page",
    )
    manager = ToolOutputManager.from_runtime_config(config)
    noncanonical = json.dumps(_context_page()).encode("utf-8")
    with pytest.raises(ToolOutputManagementError, match="context content page"):
        manager.project(
            noncanonical,
            full_output_ref=ResourceRef(
                "artifact", "ticket-382-invalid-page-output", 1
            ).to_dict(),
            digest=_digest(noncanonical),
            content_bytes=len(noncanonical),
            requested_policy="context_page",
        )

    oversized_page = _context_page()
    oversized_page["media_type"] = "text/" + ("x" * 12_288)
    oversized = _canonical_json(oversized_page)
    with pytest.raises(ToolOutputManagementError, match="context content page"):
        manager.project(
            oversized,
            full_output_ref=ResourceRef(
                "artifact", "ticket-382-oversized-page-output", 1
            ).to_dict(),
            digest=_digest(oversized),
            content_bytes=len(oversized),
            requested_policy="context_page",
        )


@pytest.mark.parametrize(
    "provider", ("openai", "anthropic", "gemini", "hyperspace", "ollama")
)
def test_context_page_reaches_each_provider_without_a_second_output_ref(
    provider: str,
):
    toolkit = Toolkit(
        {
            "context_content_read": Tool(
                name="context_content_read",
                description="read",
                func=lambda: None,
                output_policy="context_page",
            )
        }
    )
    config = ToolOutputManager.active_runtime_config_for_toolkit(
        toolkit,
        attempt_id=f"ticket-382-context-page-{provider}",
    )
    manager = ToolOutputManager.from_runtime_config(config)
    page = _context_page()
    raw = _canonical_json(page)
    projection = manager.project(
        raw,
        full_output_ref=ResourceRef("artifact", "ticket-382-page-output", 1).to_dict(),
        digest=_digest(raw),
        content_bytes=len(raw),
        requested_policy="context_page",
    )

    messages = get_provider_message_builder(provider).build_tool_result_messages(
        tool_call=ToolCall(
            call_id="ticket-382-page-call",
            name="context_content_read",
            arguments={},
        ),
        tool_result=projection.payload,
    )
    assert messages
    assert "ticket-382-page-sentinel" in str(messages)
    assert "full_output_ref" not in str(messages)


@pytest.mark.parametrize("provider", ("openai", "anthropic", "hyperspace", "ollama"))
def test_declared_policy_projects_a_provider_valid_result_for_each_provider(
    provider: str,
):
    """Exercise provider-native encoding of a sealed Toolkit projection.

    Real normal, graph, resume, and subagent entrypoints are covered by the
    provider-boundary tests; this test is intentionally limited to the common
    provider-message builder contract.
    """
    toolkit = Toolkit(
        {
            "large_search": Tool(
                name="large_search",
                description="search",
                func=lambda: None,
                output_policy="artifact_only",
            )
        }
    )
    config = ToolOutputManager.active_runtime_config_for_toolkit(
        toolkit,
        attempt_id=f"provider-projection-{provider}",
    )
    manager = ToolOutputManager.from_runtime_config(
        config,
        attempt_id=f"provider-projection-{provider}",
    )
    raw = b"result that is intentionally not model-inline"
    projection = manager.project(
        raw,
        full_output_ref={"uri": "pupu://artifact/tool-output@1"},
        digest=_digest(raw),
        content_bytes=len(raw),
        call_id="call-output",
        requested_policy=manager.resolve_policy_for_tool(
            config,
            tool_name="large_search",
        ).name,
    )

    assert projection.payload["projection"] == "artifact_only"
    assert projection.payload["full_output_ref"] == {
        "uri": "pupu://artifact/tool-output@1"
    }
    assert projection.payload["content_sha256"] == _digest(raw)
    assert projection.payload["content_bytes"] == len(raw)
    messages = get_provider_message_builder(provider).build_tool_result_messages(
        tool_call=ToolCall(
            call_id="call-output",
            name="large_search",
            arguments={},
        ),
        tool_result=projection.payload,
    )
    assert messages
    assert "intentionally not model-inline" not in str(messages)
    assert "pupu://artifact/tool-output@1" in str(messages)


@pytest.mark.parametrize("source_ref", (None, {}, "", 0))
def test_invalid_source_receipt_fails_closed(source_ref):
    manager = ToolOutputManager.active_default()
    raw = b"value"

    with pytest.raises(ToolOutputManagementError):
        manager.project(
            raw,
            full_output_ref=source_ref,
            digest=_digest(raw),
            content_bytes=len(raw),
        )


def test_page_continuation_cannot_switch_source_artifact():
    manager = ToolOutputManager.active_default()
    first = manager.read_page(
        source_ref={"uri": "pupu://artifact/a@1"}, offset=0, limit=10
    )
    next_page = manager.read_page(
        source_ref={"uri": "pupu://artifact/a@1"},
        offset=10,
        limit=10,
        continuation=first,
    )
    assert next_page.offset == 10
    with pytest.raises(ToolOutputReadError):
        manager.read_page(
            source_ref={"uri": "pupu://artifact/b@1"},
            offset=10,
            limit=10,
            continuation=first,
        )


def test_manager_owns_provider_valid_historical_compaction():
    manager = ToolOutputManager.active_default()
    openai = manager.compact_historical_message(
        {"type": "function_call_output", "call_id": "call-a", "output": "raw"},
        call_ids={"call-a"},
    )
    assert openai["output"] == (
        '{"memory_v2_compacted": true, "call_ids": ["call-a"], '
        '"note": "Full tool output is available in the durable context journal."}'
    )

    gemini = manager.compact_historical_message(
        {
            "role": "user",
            "parts": [
                {"function_response": {"name": "search", "response": {"raw": True}}}
            ],
        },
        call_ids={"call-a"},
    )
    assert gemini["parts"][0]["function_response"]["response"] == {
        "memory_v2_compacted": True,
        "call_ids": ["call-a"],
    }


def _compacted_payload(message: dict) -> object:
    if "output" in message:
        return json.loads(message["output"])
    if message.get("role") == "tool":
        return json.loads(message["content"])
    if isinstance(message.get("content"), list):
        return json.loads(message["content"][0]["content"])
    return message["parts"][0]["function_response"]["response"]


@pytest.mark.parametrize(
    "provider", ("openai", "anthropic", "gemini", "hyperspace", "ollama")
)
def test_historical_compaction_preserves_a_strict_source_read_handle(provider):
    descriptor = present_context_output_ref(
        ResourceRef("artifact", "ticket-382-historical-source", 1),
        {
            "content_bytes": 20_000,
            "content_sha256": _digest(b"historical-source"),
            "preview_head": "head",
            "preview_tail": "tail",
            "omitted_bytes": 19_992,
        },
    )
    message = get_provider_message_builder(provider).build_tool_result_message(
        tool_call=ToolCall(call_id="call-history", name="large_search", arguments={}),
        tool_result=descriptor,
    )

    compacted = ToolOutputManager.active_default().compact_historical_message(
        message,
        call_ids={"call-history"},
    )

    payload = _compacted_payload(compacted)
    assert payload["memory_v2_compacted"] is True
    assert payload["context_content"] == [
        {
            "full_output_ref": descriptor["full_output_ref"],
            "read_request": descriptor["read_request"],
        }
    ]


def test_historical_compaction_preserves_page_range_and_reread_request():
    page = _context_page()
    compacted = ToolOutputManager.active_default().compact_historical_message(
        {
            "type": "function_call_output",
            "call_id": "call-page-history",
            "output": json.dumps(page, ensure_ascii=False),
        },
        call_ids={"call-page-history"},
    )

    payload = _compacted_payload(compacted)
    assert payload["context_content"] == [
        {
            "ref": page["ref"],
            "offset": page["offset"],
            "page_bytes": page["page_bytes"],
            "next_offset": page["next_offset"],
            "eof": page["eof"],
            "read_request": {
                "tool": "context_content_read",
                "arguments": {"ref": page["ref"], "offset": 0, "limit": 8192},
            },
            "next_read": page["next_read"],
        }
    ]


@pytest.mark.parametrize(
    "provider", ("openai", "anthropic", "gemini", "hyperspace", "ollama")
)
def test_historical_compaction_is_idempotent_for_context_page(provider):
    page = _context_page()
    message = get_provider_message_builder(provider).build_tool_result_message(
        tool_call=ToolCall(call_id="call-page-history", name="context_content_read", arguments={}),
        tool_result=page,
    )

    once = ToolOutputManager.active_default().compact_historical_message(
        message,
        call_ids={"call-page-history"},
    )
    twice = ToolOutputManager.active_default().compact_historical_message(
        once,
        call_ids={"call-page-history"},
    )

    assert _compacted_payload(twice)["context_content"] == _compacted_payload(once)[
        "context_content"
    ]
