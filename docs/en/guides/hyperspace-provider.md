# SAP Hyperspace Provider

`HyperspaceModelIO` connects unchain agents to **SAP Hyperspace** — an internal LLM proxy that exposes an Anthropic-compatible Messages API at a configurable base URL (default `http://localhost:6655/anthropic`).

Because Hyperspace speaks the Anthropic wire protocol, `HyperspaceModelIO` subclasses `AnthropicModelIO` and only overrides the default `client_factory` to point at the configured base URL. All Anthropic features are inherited automatically:

- Streaming `token_delta` / `reasoning` events via `request.callback`
- Extended thinking blocks
- Tool use round-trip with prompt caching (`cache_control: ephemeral`)
- Cache-aware token usage (`cache_read_input_tokens`, `cache_creation_input_tokens`)

## Installation

The `anthropic` Python SDK is already a runtime dependency of unchain — nothing extra to install.

## Quick start (high-level Agent API)

```python
from unchain import Agent

agent = Agent(
    name="translator",
    provider="hyperspace",
    model="hyperspace--claude-opus-4-6",
    api_key="<your-hyperspace-key>",   # or set HYPERSPACE_API_KEY env var
    instructions="You are a careful technical translator.",
)

result = agent.run("Translate 'hello world' to Spanish.")
print(result.messages[-1]["content"])
```

Environment variables read by `_create_hyperspace`:

| Variable | Purpose | Default |
|----------|---------|---------|
| `HYPERSPACE_API_KEY` | API key sent as `x-api-key` | required |
| `HYPERSPACE_BASE_URL` | Override the Anthropic-compatible endpoint | `http://localhost:6655/anthropic` |

## Direct provider use (low-level)

```python
from unchain.providers import HyperspaceModelIO
from unchain.kernel import ModelTurnRequest

io = HyperspaceModelIO(
    model="hyperspace--claude-opus-4-6",
    api_key="<your-hyperspace-key>",
    base_url="http://localhost:6655/anthropic",  # optional
)

events: list = []
turn = io.fetch_turn(
    ModelTurnRequest(
        messages=[
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Translate 'hello' to French."},
        ],
        callback=events.append,
        emit_stream=True,
        run_id="demo",
    )
)
print(turn.final_text)  # "bonjour"
```

## Registered models

| Model key (unchain) | provider_model (sent to Hyperspace) | Notes |
|---------------------|-------------------------------------|-------|
| `hyperspace--claude-opus-4-6` | `anthropic--claude-opus-4-6` | tools, thinking |
| `hyperspace--claude-opus-4-7` | `anthropic--claude-opus-4-7` | tools, thinking |
| `hyperspace--claude-sonnet-4-6` | `anthropic--claude-sonnet-4-6` | tools, thinking |
| `hyperspace--claude-haiku-4-5` | `anthropic--claude-haiku-4-5` | tools, thinking |

The `hyperspace--*` prefix avoids any clash with native Anthropic registry entries; the `provider_model` field rewrites the actual model string sent on the wire to SAP's `anthropic--*` naming.

## Authentication

The `anthropic` SDK already sends `X-Api-Key: <key>` and `anthropic-version: 2023-06-01` by default — both are exactly what Hyperspace expects, so no custom headers are required.

## Tests

### Kimi K2.7 Code replay compatibility

For `kimi-k2.7-code` at `https://api.moonshot.ai/anthropic` or
`https://api.moonshot.cn/anthropic`, the adapter preserves unsigned thinking in
provider-private replay using `kimi.unsigned-thinking.v1`. The persisted profile
is bound to the exact model and endpoint; the next request must have the same
live adapter profile. Native Anthropic, other models and other proxy endpoints
continue to require signed thinking. This is not a generic unsigned-thinking
switch. Payload model overrides and mismatched/unknown replay profiles fail
before provider execution. Existing unprofiled frames retain the old strict
rules; broken historical checkpoints are not rewritten.

`tests/test_kimi_replay.py` covers repeated tools, route identity rejection and
cold restart, including the official Context v2 approval path. These deterministic
tests do not substitute for real provider qualification.

See `tests/test_hyperspace_model_io.py` for the full test suite (request building, streaming, tool use, extended thinking, validation).

```bash
PYTHONPATH=src pytest tests/test_hyperspace_model_io.py -v
```
