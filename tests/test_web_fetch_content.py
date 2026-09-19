import json
import tempfile

import pytest

from unchain.kernel.types import ToolCall
from unchain.tools.messages import get_provider_message_builder
from unchain.tools.prompting import render_tool_prompt_block
from unchain.toolkits import CoreToolkit
from unchain.toolkits.builtin.core import web_fetch as web
from unchain.toolkits.builtin.core.web_backend import past_end_notice, truncation_notice

_SKIP_TAGS = (
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "iframe",
    "object",
    "canvas",
    "audio",
    "video",
)


def _body_for(tag: str) -> str:
    if tag == "script":
        return '{"featureFlags":["a","b"]}'
    if tag == "style":
        return "p { color: red }"
    return f"SECRET-{tag}"


@pytest.mark.parametrize("tag", _SKIP_TAGS)
def test_html_to_markdown_skips_non_content_elements(tag):
    body = _body_for(tag)
    html_doc = (
        "<html><head><title>Title</title></head><body>"
        "<p>before</p>"
        f'<{tag} attr="x">{body}</{tag}>'
        "<p>after</p>"
        "</body></html>"
    )
    md = web.html_to_markdown(html_doc)
    assert "SECRET" not in md
    assert "featureFlags" not in md
    assert "color: red" not in md
    assert "before" in md
    assert "after" in md
    assert "Title" in md


def test_html_to_markdown_skips_nested_and_unbalanced():
    html_doc = (
        "<html><body>"
        "<svg><svg><text>X</text></svg></svg>"
        "<p>visible</p>"
        "</body></html>"
    )
    md = web.html_to_markdown(html_doc)
    assert "X" not in md
    assert "visible" in md

    # A stray end tag with no matching opener must not suppress following text.
    html_doc_unbalanced = "<html><body></script><p>still visible</p></body></html>"
    md_unbalanced = web.html_to_markdown(html_doc_unbalanced)
    assert "still visible" in md_unbalanced


def test_html_to_markdown_void_elements_never_swallow_text():
    html_doc = (
        '<html><body><meta charset="utf-8">'
        '<link rel="stylesheet" href="x.css">'
        '<img src="a.png">'
        '<input type="text">'
        "<p>kept text</p>"
        "</body></html>"
    )
    md = web.html_to_markdown(html_doc)
    assert "kept text" in md

    html_doc_br_hr = "<html><body><p>first</p><br><hr><p>second</p></body></html>"
    md_br_hr = web.html_to_markdown(html_doc_br_hr)
    assert "first" in md_br_hr
    assert "second" in md_br_hr


def test_html_to_markdown_keeps_existing_structure():
    html_doc = (
        "<html><head><title>Doc Title</title></head><body>\n"
        "<h2>Section Heading</h2>\n"
        "<ul><li>First item</li><li>Second item</li></ul>\n"
        '<pre><code>print("hi")\n'
        "x = 1\n"
        "</code></pre>\n"
        '<p>See <a href="https://example.com/ref">the reference</a> for more.</p>\n'
        "</body></html>"
    )
    md = web.html_to_markdown(html_doc)
    # Captured by running the CURRENT (pre-fix) implementation; this string
    # must stay identical before and after the skip-set fix.
    expected = (
        'Doc Title\n\n## Section Heading\n\n- First item\n- Second item\n\n'
        '```\nprint("hi")\nx = 1\n```\n\nSee the reference (https://example.com/ref) for more.'
    )
    assert md == expected


def _build_github_shaped_fixture() -> tuple[str, str]:
    description = "A lightweight cross-platform desktop AI client built with React and Electron."

    importmap_body = (
        '{"imports":{'
        + ",".join(
            f'"pkg-{i}":"https://github.githubassets.com/assets/pkg-{i}-{"a" * 20}.js"'
            for i in range(40)
        )
        + "}}"
    )
    assert len(importmap_body) > 1200

    style_body = "".join(f".c{i} {{ color: #{i:06x}; padding: {i}px; }}\n" for i in range(20))
    assert len(style_body) > 400

    flag_names = [f"feature_flag_{i}_{'x' * 20}" for i in range(120)]
    embedded_json = (
        '{"locale":"en","featureFlags":[' + ",".join(f'"{name}"' for name in flag_names) + "]}"
    )
    assert len(embedded_json) > 3500

    title = f"GitHub - o/r: {description} · GitHub"

    nav = '<nav><ul><li><a href="/o">Code</a></li><li><a href="/o/issues">Issues</a></li></ul></nav>'
    h1 = "<h1>o/r</h1>"
    summary_p = f"<p>{description}</p>"
    paragraph = (
        "This project ships a modular runtime with tool-using agents, a memory "
        "subsystem, and provider adapters for several model vendors. "
    )
    article = "<article>" + "".join(f"<p>{paragraph}Paragraph {i}.</p>" for i in range(20)) + "</article>"
    assert len(article) > 2800
    footer = "<footer>© GitHub, Inc.</footer>"

    html_doc = (
        "<html><head>"
        f'<script type="importmap">{importmap_body}</script>'
        f"<style>{style_body}</style>"
        f'<script type="application/json" data-target="react-app.embeddedData">{embedded_json}</script>'
        f"<title>{title}</title>"
        "</head><body>"
        f"{nav}{h1}{summary_p}{article}{footer}"
        "</body></html>"
    )
    return html_doc, description


def test_github_shaped_page_puts_description_first():
    html_doc, description = _build_github_shaped_fixture()

    # measured on unfixed converter at e583e9a: 11861 chars
    baseline = 11861

    md = web.html_to_markdown(html_doc)
    assert "featureFlags" not in md
    assert 0 <= md.find(description) < 300
    assert len(md) < baseline / 2


_PAGE_100 = "".join(str(i % 10) for i in range(100))


def _toolkit_with_fake_page(monkeypatch, page: str):
    tmp = tempfile.mkdtemp()
    toolkit = CoreToolkit(workspace_root=tmp)
    monkeypatch.setattr(web, "validate_public_url", lambda url: (url, None))

    calls = {"count": 0}

    def fake_request(url: str):
        calls["count"] += 1
        return (
            {
                "ok": True,
                "url": url,
                "final_url": url,
                "host": "example.com",
                "status_code": 200,
                "content_type": "text/plain",
                "file_kind": "text",
                "result": "",
                "content_length": len(page),
                "returned_chars": 0,
                "truncated": False,
                "next_offset": None,
                "cached": False,
                "redirect": None,
                "skipped": False,
                "error": "",
            },
            page,
        )

    monkeypatch.setattr(toolkit._web_backend.web_fetch_service, "_request", fake_request)
    return toolkit, calls


def test_raw_page_truncated_result_carries_notice(monkeypatch):
    toolkit, _ = _toolkit_with_fake_page(monkeypatch, _PAGE_100)
    result = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": 0, "max_chars": 40},
    )
    assert result["result"] == _PAGE_100[:40] + truncation_notice(0, 40, 100, 40)
    assert result["returned_chars"] == 40
    assert result["truncated"] is True
    assert result["next_offset"] == 40


def test_raw_page_continuation_to_end_has_no_notice(monkeypatch):
    toolkit, _ = _toolkit_with_fake_page(monkeypatch, _PAGE_100)
    result = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": 40, "max_chars": 100},
    )
    assert result["result"] == _PAGE_100[40:]
    assert "[web_fetch notice" not in result["result"]
    assert result["returned_chars"] == 60
    assert result["truncated"] is False
    assert result["next_offset"] is None


@pytest.mark.parametrize("offset", [100, 250])
def test_raw_page_past_end_returns_notice(monkeypatch, offset):
    toolkit, _ = _toolkit_with_fake_page(monkeypatch, _PAGE_100)
    result = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": offset, "max_chars": 40},
    )
    assert result["ok"] is True
    assert result["result"] == past_end_notice(offset, 100)
    assert result["returned_chars"] == 0
    assert result["truncated"] is False
    assert result["next_offset"] is None


def test_raw_page_cached_second_call_carries_its_own_notice(monkeypatch):
    toolkit, calls = _toolkit_with_fake_page(monkeypatch, _PAGE_100)
    first = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": 0, "max_chars": 40},
    )
    assert first["cached"] is False
    assert first["result"] == _PAGE_100[:40] + truncation_notice(0, 40, 100, 40)

    second = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": 50, "max_chars": 20},
    )
    assert second["cached"] is True
    assert second["result"] == _PAGE_100[50:70] + truncation_notice(50, 70, 100, 70)
    assert second["returned_chars"] == 20
    assert second["truncated"] is True
    assert second["next_offset"] == 70
    assert calls["count"] == 1


def test_raw_page_success_key_set_is_exact(monkeypatch):
    toolkit, _ = _toolkit_with_fake_page(monkeypatch, _PAGE_100)
    result = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": 0, "max_chars": 40},
    )
    assert set(result) == {
        "ok", "url", "final_url", "host", "status_code", "content_type",
        "file_kind", "result", "content_length", "returned_chars", "truncated",
        "next_offset", "cached", "redirect", "skipped", "error", "mode",
    }


def test_web_fetch_provider_json_description_carries_continuation_guidance():
    with tempfile.TemporaryDirectory() as tmp:
        toolkit = CoreToolkit(workspace_root=tmp)
        description = toolkit.tools["web_fetch"].to_provider_json("openai")["description"]
        assert "offset=next_offset" in description


def test_web_fetch_prompt_block_carries_notice_and_continuation_guidance():
    with tempfile.TemporaryDirectory() as tmp:
        toolkit = CoreToolkit(workspace_root=tmp)
        rendered = render_tool_prompt_block(toolkit)
        assert "[web_fetch notice" in rendered
        assert "offset=next_offset" in rendered


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "ollama"])
def test_web_fetch_wire_result_carries_notice_for_all_providers(monkeypatch, provider):
    toolkit, _ = _toolkit_with_fake_page(monkeypatch, _PAGE_100)
    result = toolkit.execute(
        "web_fetch",
        {"url": "https://example.com/page", "mode": "raw", "offset": 0, "max_chars": 40},
    )
    assert result["truncated"] is True

    call = ToolCall(
        call_id="c1",
        name="web_fetch",
        arguments={"url": "https://example.com/page", "mode": "raw", "offset": 0, "max_chars": 40},
    )
    builder = get_provider_message_builder(provider)
    messages = builder.build_tool_result_messages(tool_call=call, tool_result=result)
    serialized = json.dumps(messages, default=str, ensure_ascii=False)
    assert "[web_fetch notice" in serialized
    assert "offset=40" in serialized
