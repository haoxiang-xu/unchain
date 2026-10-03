from __future__ import annotations

import json

import anthropic
import httpx

from unchain.providers import AnthropicModelIO
from unchain.runtime import build_runtime_loop
from unchain.tools import Toolkit


MODEL = "claude-sonnet-5"


def _stream_body(*, content: list[dict], stop_reason: str) -> str:
    events = [{
        "type": "message_start",
        "message": {
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        },
    }]
    for index, block in enumerate(content):
        start = dict(block)
        if block["type"] == "text":
            start["text"] = ""
        else:
            start["input"] = {}
        events.append({
            "type": "content_block_start",
            "index": index,
            "content_block": start,
        })
        if block["type"] == "text":
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"]),
            }
        events.extend([
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ])
    events.extend([
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ])
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    )


def test_real_anthropic_sdk_parsed_text_replays_parallel_tools_without_wire_pollution():
    responses = [
        _stream_body(
            content=[
                {"type": "text", "text": "I will call both tools."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "add_one",
                    "input": {"value": 1},
                },
                {
                    "type": "tool_use",
                    "id": "toolu_2",
                    "name": "add_one",
                    "input": {"value": 2},
                },
            ],
            stop_reason="tool_use",
        ),
        _stream_body(
            content=[{"type": "text", "text": "Both calls are done."}],
            stop_reason="end_turn",
        ),
    ]
    requests: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=responses.pop(0),
            request=request,
        )

    def client_factory(**kwargs):
        return anthropic.Anthropic(
            **kwargs,
            base_url="https://example.test",
            http_client=httpx.Client(transport=httpx.MockTransport(respond)),
        )

    effects: list[int] = []

    def add_one(value: int) -> int:
        effects.append(value)
        return value + 1

    toolkit = Toolkit()
    toolkit.register(add_one, name="add_one")
    result = build_runtime_loop(model_io=AnthropicModelIO(
        model=MODEL,
        api_key="fixture",
        client_factory=client_factory,
    )).run(
        [{"role": "user", "content": "Call both tools."}],
        provider="anthropic",
        model=MODEL,
        toolkit=toolkit,
        max_iterations=3,
    )

    assert result.status == "completed"
    assert result.messages[-1] == {"role": "assistant", "content": "Both calls are done."}
    assert sorted(effects) == [1, 2]
    assert len(requests) == 2
    replayed = requests[1]["messages"][-2]["content"]
    assert [block["type"] for block in replayed] == ["text", "tool_use", "tool_use"]
    assert "parsed_output" not in replayed[0]
    results = requests[1]["messages"][-1]["content"]
    assert [block["tool_use_id"] for block in results] == ["toolu_1", "toolu_2"]
