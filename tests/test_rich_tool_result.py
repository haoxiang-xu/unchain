"""S0 — rich tool result (image回流) contract tests.

Contract (architect-frozen, single-directional gate): a tool result dict MAY
carry a reserved ``content_blocks`` key. When absent, provider builders behave
EXACTLY as before (zero migration). When present, each provider surfaces the
blocks in its native shape. Block vocabulary is append-only; today: text/image.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from unchain.kernel.types import ToolCall
from unchain.tools import get_provider_message_builder
from unchain.tools.messages import (
    coalesce_provider_tool_result_messages,
    iter_result_image_blocks,
    redact_result_image_data,
)
from unchain.tools.result_budget import ToolResultBudgetConfig, ToolResultBudgetController


def _tool_call():
    return SimpleNamespace(call_id="call_1", name="demo_tool")


def _rich_result():
    return {
        "status": "ok",
        "content_blocks": [
            {"type": "text", "text": "screenshot taken"},
            {
                "type": "image",
                "media_type": "image/png",
                "data_b64": "aW1n",
                "width": 1512,
                "height": 982,
            },
        ],
    }


# ── zero-migration: no content_blocks == byte-identical legacy behavior ──────

def test_anthropic_without_content_blocks_is_unchanged():
    msg = get_provider_message_builder("anthropic").build_tool_result_message(
        tool_call=_tool_call(), tool_result={"ok": True}
    )
    assert msg == {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": json.dumps({"ok": True}, ensure_ascii=False),
        }],
    }


def test_openai_without_content_blocks_is_unchanged():
    msg = get_provider_message_builder("openai").build_tool_result_message(
        tool_call=_tool_call(), tool_result={"ok": True}
    )
    assert msg == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": json.dumps({"ok": True}, ensure_ascii=False),
    }


def test_gemini_without_content_blocks_is_unchanged():
    msg = get_provider_message_builder("gemini").build_tool_result_message(
        tool_call=_tool_call(), tool_result={"ok": True}
    )
    assert msg == {
        "role": "user",
        "parts": [{"function_response": {"name": "demo_tool", "response": {"ok": True}}}],
    }


def test_ollama_without_content_blocks_is_unchanged():
    msg = get_provider_message_builder("ollama").build_tool_result_message(
        tool_call=_tool_call(), tool_result={"ok": True}
    )
    assert msg == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps({"ok": True}, ensure_ascii=False),
    }


# ── plural is the new contract; singular proxies to messages[0] ──────────────

def test_builder_returns_list_and_singular_proxies():
    builder = get_provider_message_builder("anthropic")
    messages = builder.build_tool_result_messages(
        tool_call=_tool_call(), tool_result={"ok": True}
    )
    assert isinstance(messages, list) and len(messages) == 1
    assert builder.build_tool_result_message(
        tool_call=_tool_call(), tool_result={"ok": True}
    ) == messages[0]


# ── content_blocks present: native surfacing per provider ────────────────────

def test_anthropic_image_becomes_native_source_block():
    messages = get_provider_message_builder("anthropic").build_tool_result_messages(
        tool_call=_tool_call(), tool_result=_rich_result()
    )
    assert len(messages) == 1
    tool_result_block = messages[0]["content"][0]
    assert tool_result_block["type"] == "tool_result"
    assert tool_result_block["tool_use_id"] == "call_1"
    # content is now a block array, not a JSON string
    assert tool_result_block["content"] == [
        {"type": "text", "text": "screenshot taken"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "aW1n"},
        },
    ]


def test_openai_image_is_placeholder_text_and_single_message():
    # llm-expert ruling (M2): image回流 is an honest placeholder; real image
    # output (M3) goes in-band in function_call_output.output, not here.
    messages = get_provider_message_builder("openai").build_tool_result_messages(
        tool_call=_tool_call(), tool_result=_rich_result()
    )
    assert len(messages) == 1
    assert messages[0]["type"] == "function_call_output"
    assert messages[0]["call_id"] == "call_1"
    assert "screenshot taken" in messages[0]["output"]
    assert "[image omitted: 1512x982 png]" in messages[0]["output"]


def test_gemini_image_becomes_inline_data_part():
    messages = get_provider_message_builder("gemini").build_tool_result_messages(
        tool_call=_tool_call(), tool_result=_rich_result()
    )
    assert len(messages) == 1
    parts = messages[0]["parts"]
    assert parts[0]["function_response"]["name"] == "demo_tool"
    assert "screenshot taken" in json.dumps(parts[0]["function_response"]["response"])
    assert parts[1] == {"inline_data": {"mime_type": "image/png", "data": "aW1n"}}


def test_ollama_image_is_omitted_placeholder():
    messages = get_provider_message_builder("ollama").build_tool_result_messages(
        tool_call=_tool_call(), tool_result=_rich_result()
    )
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert "screenshot taken" in messages[0]["content"]
    assert "[image omitted: 1512x982 png]" in messages[0]["content"]


def test_empty_or_invalid_content_blocks_falls_back_to_legacy():
    # content_blocks present but empty/invalid => treat as no blocks (legacy).
    for bad in ([], "nope", [123, "x"]):
        result = {"ok": True, "content_blocks": bad}
        msg = get_provider_message_builder("anthropic").build_tool_result_message(
            tool_call=_tool_call(), tool_result=result
        )
        assert msg["content"][0]["content"] == json.dumps(result, ensure_ascii=False)


# ── host interception hook ───────────────────────────────────────────────────

def test_iter_result_image_blocks_finds_images():
    blocks = iter_result_image_blocks(_rich_result())
    assert len(blocks) == 1
    assert blocks[0]["data_b64"] == "aW1n"
    assert iter_result_image_blocks({"ok": True}) == []
    assert iter_result_image_blocks("not a dict") == []


def test_redact_result_image_data_strips_base64_in_place():
    result = _rich_result()
    redacted = redact_result_image_data(result)
    # same object mutated
    assert redacted is result
    image = result["content_blocks"][1]
    assert "data_b64" not in image
    # a reference marker remains so the host can wire an artifact/URL
    assert image["type"] == "image"
    assert image.get("data_omitted") is True
    assert image.get("byte_len") == len("aW1n")
    # non-image blocks untouched, business fields untouched
    assert result["content_blocks"][0] == {"type": "text", "text": "screenshot taken"}
    assert result["status"] == "ok"


def test_redact_is_safe_on_plain_results():
    assert redact_result_image_data({"ok": True}) == {"ok": True}
    assert redact_result_image_data("x") == "x"


# ── regression red-lines: coalesce/budget over block arrays ──────────────────

def _anthropic_tool_result_message(call_id, tool_result):
    return get_provider_message_builder("anthropic").build_tool_result_messages(
        tool_call=SimpleNamespace(call_id=call_id, name="computer"),
        tool_result=tool_result,
    )[0]


def _budget(message, call_id="c1", config=None):
    return ToolResultBudgetController(config or ToolResultBudgetConfig()).budget_messages(
        provider="anthropic",
        toolkit=None,
        tool_calls=[ToolCall(call_id=call_id, name="computer", arguments={})],
        result_messages=[message],
    )


def test_budget_preserves_screenshot_base64_over_4000_chars():
    # RED-LINE: a real screenshot base64 (>>4000 chars) must not be counted
    # toward the result budget, compacted, or rewritten into a bare dict.
    big_b64 = "A" * 500_000
    message = _anthropic_tool_result_message(
        "c1",
        {
            "status": "ok",
            "content_blocks": [
                {"type": "text", "text": "screenshot taken"},
                {"type": "image", "media_type": "image/png", "data_b64": big_b64,
                 "width": 1512, "height": 982},
            ],
        },
    )
    outcome = _budget(message)
    out_content = outcome.messages[0]["content"][0]["content"]
    # still a valid Anthropic block array (never a bare dict / string)
    assert isinstance(out_content, list)
    assert all(isinstance(b, dict) and b.get("type") in ("text", "image") for b in out_content)
    image = next(b for b in out_content if b.get("type") == "image")
    assert image["source"]["data"] == big_b64  # base64 intact, uncompacted
    assert outcome.stats.compacted_count == 0


def test_budget_compacts_huge_text_but_preserves_image():
    big_text = "x" * 50_000
    message = _anthropic_tool_result_message(
        "c1",
        {
            "content_blocks": [
                {"type": "text", "text": big_text},
                {"type": "image", "media_type": "image/png", "data_b64": "aW1n"},
            ],
        },
    )
    outcome = _budget(message)
    out_content = outcome.messages[0]["content"][0]["content"]
    assert isinstance(out_content, list)
    image = next(b for b in out_content if b.get("type") == "image")
    assert image["source"]["data"] == "aW1n"  # image preserved verbatim
    text_block = next(b for b in out_content if b.get("type") == "text")
    assert len(text_block["text"]) < len(big_text)  # text was compacted
    assert outcome.stats.compacted_count >= 1


def test_budget_text_only_block_array_stays_valid_block_array_on_compaction():
    # Broader red-line: even a text-only block array must not compact into a
    # bare dict (illegal Anthropic tool_result.content).
    big_text = "y" * 50_000
    message = _anthropic_tool_result_message("c1", {"content_blocks": [{"type": "text", "text": big_text}]})
    outcome = _budget(message)
    out_content = outcome.messages[0]["content"][0]["content"]
    assert isinstance(out_content, list)
    assert all(isinstance(b, dict) and b.get("type") == "text" for b in out_content)
    assert len(out_content[0]["text"]) < len(big_text)


def test_coalesce_merges_anthropic_block_array_tool_results():
    builder = get_provider_message_builder("anthropic")
    m1 = builder.build_tool_result_messages(
        tool_call=SimpleNamespace(call_id="a", name="t"), tool_result=_rich_result()
    )[0]
    m2 = builder.build_tool_result_messages(
        tool_call=SimpleNamespace(call_id="b", name="t"), tool_result={"ok": True}
    )[0]
    coalesced = coalesce_provider_tool_result_messages("anthropic", [m1, m2])
    # two adjacent tool_result user messages coalesce into one user message
    assert len(coalesced) == 1
    assert coalesced[0]["role"] == "user"
    assert len(coalesced[0]["content"]) == 2
    assert all(b["type"] == "tool_result" for b in coalesced[0]["content"])
    # the nested image block survives coalescing intact
    assert coalesced[0]["content"][0]["content"][1]["type"] == "image"
