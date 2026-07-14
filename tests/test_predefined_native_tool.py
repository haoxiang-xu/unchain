"""S1 — predefined provider-native tool passthrough + beta header.

Anthropic computer use needs a predefined tool shape
``{"type":"computer_20251124","name":"computer",...}`` plus the request header
``anthropic-beta: computer-use-2025-11-24``. A Tool may declare per-provider
native specs and required betas; on providers without a native spec it falls
back to the generic function schema (same execute).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

from unchain.kernel import ModelTurnRequest
from unchain.kernel.provider_replay import tool_schema_digest, tool_schema_manifest
from unchain.providers import AnthropicModelIO
from unchain.tools import Toolkit
from unchain.tools.tool import Tool


COMPUTER_SPEC = {
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": 1512,
    "display_height_px": 982,
}
COMPUTER_BETA = "computer-use-2025-11-24"


def _computer_tool() -> Tool:
    def computer(action: str = "screenshot"):
        return {"ok": True, "action": action}

    return Tool(
        name="computer",
        func=computer,
        provider_native_specs={"anthropic": COMPUTER_SPEC},
        required_betas={"anthropic": [COMPUTER_BETA]},
    )


# ── to_provider_json: native on anthropic, generic elsewhere ─────────────────

def test_anthropic_returns_native_spec():
    tool = _computer_tool()
    assert tool.to_provider_json("anthropic") == COMPUTER_SPEC


def test_native_spec_is_deep_copied():
    tool = _computer_tool()
    produced = tool.to_provider_json("anthropic")
    produced["display_width_px"] = 1
    # mutating the returned dict must not corrupt the tool's stored spec
    assert tool.to_provider_json("anthropic")["display_width_px"] == 1512


def test_non_anthropic_falls_back_to_generic_function_schema():
    tool = _computer_tool()
    for provider in ("openai", "gemini", "ollama"):
        schema = tool.to_provider_json(provider)
        assert "type" not in schema or schema.get("type") != "computer_20251124"
        # generic schema carries the tool name and an executable function shape
        name = schema.get("name") or schema.get("function", {}).get("name")
        assert name == "computer"


def test_plain_tool_unaffected_without_native_spec():
    def echo(x: str):
        return {"x": x}

    tool = Tool.from_callable(echo, name="echo")
    anthropic_schema = tool.to_provider_json("anthropic")
    assert anthropic_schema["name"] == "echo"
    assert "input_schema" in anthropic_schema


# ── required betas aggregation ───────────────────────────────────────────────

def test_toolkit_aggregates_required_betas():
    toolkit = Toolkit()
    toolkit.register(_computer_tool())
    assert toolkit.required_betas("anthropic") == [COMPUTER_BETA]
    assert toolkit.required_betas("openai") == []


def test_toolkit_required_betas_dedupes_and_is_stable():
    def a():
        return {}

    def b():
        return {}

    t1 = Tool(name="a", func=a, required_betas={"anthropic": [COMPUTER_BETA, "x-beta"]})
    t2 = Tool(name="b", func=b, required_betas={"anthropic": [COMPUTER_BETA]})
    toolkit = Toolkit()
    toolkit.register(t1)
    toolkit.register(t2)
    assert toolkit.required_betas("anthropic") == [COMPUTER_BETA, "x-beta"]


# ── provider replay compatibility: native spec must not break manifest/digest ─

def test_native_spec_is_replay_compatible():
    toolkit = Toolkit()
    toolkit.register(_computer_tool())
    manifest = tool_schema_manifest(toolkit, "anthropic")
    assert "computer" in manifest
    # digest is stable and does not raise on the native spec
    assert tool_schema_digest(toolkit, "anthropic") == tool_schema_digest(toolkit, "anthropic")


# ── exposure preservation: re-registering by reference keeps native spec ─────

def test_native_spec_survives_toolkit_reregistration():
    # exposure builds the active toolkit by re-registering the same Tool object;
    # native spec + betas must survive that path.
    full = Toolkit()
    full.register(_computer_tool())
    exposed = Toolkit()
    exposed.register(full.get("computer"))
    assert exposed.to_provider_json("anthropic") == [COMPUTER_SPEC]
    assert exposed.required_betas("anthropic") == [COMPUTER_BETA]


# ── decorator / from_callable threading ──────────────────────────────────────

def test_from_callable_threads_native_spec_and_betas():
    def computer(action: str = "screenshot"):
        return {"ok": True}

    tool = Tool.from_callable(
        computer,
        name="computer",
        provider_native_specs={"anthropic": COMPUTER_SPEC},
        required_betas={"anthropic": [COMPUTER_BETA]},
    )
    assert tool.to_provider_json("anthropic") == COMPUTER_SPEC
    assert tool.required_betas == {"anthropic": [COMPUTER_BETA]}


# ── anthropic provider wires the beta header ─────────────────────────────────

class _FakeStream:
    def __init__(self, events):
        self._events = list(events)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(self._events)


class _FakeMessages:
    def __init__(self, *, captured_kwargs):
        self._captured = captured_kwargs

    def stream(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeStream([
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage={"input_tokens": 1, "output_tokens": 0}),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(type="text_delta", text="ok"),
            ),
            SimpleNamespace(type="message_delta", usage={"input_tokens": 1, "output_tokens": 1}),
        ])


class _FakeClient:
    def __init__(self, *, captured_kwargs, **kwargs):
        self.messages = _FakeMessages(captured_kwargs=captured_kwargs)


def _run_turn(toolkit):
    captured = {}
    io = AnthropicModelIO(
        model="claude-sonnet-4-5",
        api_key="test-key",
        client_factory=lambda api_key, **kwargs: _FakeClient(captured_kwargs=captured),
    )
    io.fetch_turn(
        ModelTurnRequest(
            messages=[{"role": "user", "content": "go"}],
            toolkit=toolkit,
            run_id="run-1",
        )
    )
    return captured


def test_anthropic_sets_beta_header_when_native_tool_present():
    toolkit = Toolkit()
    toolkit.register(_computer_tool())
    captured = _run_turn(toolkit)
    assert captured["extra_headers"]["anthropic-beta"] == COMPUTER_BETA
    # the native spec is the tool schema handed to the SDK (the provider may
    # additionally stamp cache_control on the last tool, so match on a subset).
    computer = next(t for t in captured["tools"] if t.get("type") == "computer_20251124")
    for key, value in COMPUTER_SPEC.items():
        assert computer[key] == value


def test_merge_beta_header_preserves_existing_and_dedupes():
    merge = AnthropicModelIO._merge_beta_header
    assert merge(None, [COMPUTER_BETA]) == COMPUTER_BETA
    # existing header value is preserved (first) and not overwritten
    assert merge("prompt-caching-2024", [COMPUTER_BETA]) == f"prompt-caching-2024,{COMPUTER_BETA}"
    # dedupe: a required beta already present is not repeated
    assert merge(f"x, {COMPUTER_BETA}", [COMPUTER_BETA, "y"]) == f"x,{COMPUTER_BETA},y"


def test_anthropic_omits_beta_header_without_required_betas():
    def echo(x: str):
        return {"x": x}

    toolkit = Toolkit()
    toolkit.register(Tool.from_callable(echo, name="echo"))
    captured = _run_turn(toolkit)
    assert "extra_headers" not in captured or "anthropic-beta" not in captured.get(
        "extra_headers", {}
    )
