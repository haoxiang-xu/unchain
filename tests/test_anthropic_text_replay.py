from __future__ import annotations

import copy
import json

import anthropic
import httpx
import pytest
from anthropic.types import (
    CitationCharLocation,
    ParsedTextBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)

from unchain.agent import Agent, MemoryModule, ToolsModule
from unchain.kernel import ModelTurnRequest
from unchain.kernel.provider_replay import set_provider_replay_frame
from unchain.kernel.state import RunState
from unchain.memory import JsonFileSessionStore, MemoryManager
from unchain.providers import AnthropicModelIO
from unchain.providers.context_assembler import ProviderContextProjectionError
from unchain.providers.message_contract import ProviderMessageContractError
from unchain.providers.model_turn_runtime import build_model_turn_request
from unchain.runtime import build_runtime_loop
from unchain.tools import Toolkit


def _anthropic_sse(events: list[dict]) -> bytes:
    return b"".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode("utf-8")
        for event in events
    )


def _real_sdk_client_factory(*, responses: list[bytes], observed_requests: list[dict]):
    response_bodies = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        observed_requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=response_bodies.pop(0),
        )

    def factory(**kwargs):
        return anthropic.Anthropic(
            **kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

    return factory


def _real_sdk_parallel_tool_turn() -> bytes:
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_real_sdk_tools",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 8, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking both pages."}},
        {"type": "content_block_stop", "index": 0},
    ]
    for index in (1, 2):
        events.extend([
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": f"toolu_real_sdk_{index}",
                    "name": "web_fetch",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps({"url": f"https://example.com/{index}"}),
                },
            },
            {"type": "content_block_stop", "index": index},
        ])
    events.extend([
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"output_tokens": 18},
        },
        {"type": "message_stop"},
    ])
    return _anthropic_sse(events)


def _real_sdk_final_text_turn() -> bytes:
    return _anthropic_sse([
        {
            "type": "message_start",
            "message": {
                "id": "msg_real_sdk_final",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Both pages are ready."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 7},
        },
        {"type": "message_stop"},
    ])


def _tool_blocks(*, call_id: str, texts: list[TextBlock]):
    # Model the 0.83 tool wire shape even when CI has a newer SDK whose
    # default dump adds toolset_name=None outside this provider's contract.
    tool_block = ToolUseBlock(
        type="tool_use", id=call_id, name="demo_tool", input={"x": 2},
    ).model_dump(exclude_unset=True)
    tool_block["caller"] = None
    return [
        ThinkingBlock(type="thinking", thinking="plan", signature="signed-data"),
        *texts,
        tool_block,
    ]


def _client_factory(responses, requests):
    class _Stream:
        def __init__(self, response):
            self.response = response

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return {"id": "msg_test", "content": self.response}

    class _Messages:
        def stream(self, **kwargs):
            requests.append(copy.deepcopy(kwargs))
            return _Stream(responses.pop(0))

    class _Client:
        def __init__(self, **kwargs):
            self.messages = _Messages()

    return _Client


@pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5-5"])
def test_sdk_text_blocks_survive_tool_replay_and_followup(model):
    texts = [
        TextBlock(type="text", text="I will use the tool."),
        TextBlock(type="text", text=""),
        TextBlock(type="text", text=" More context.", citations=[CitationCharLocation(
            type="char_location", cited_text="source", document_index=0,
            document_title="doc", start_char_index=0, end_char_index=6,
        )]),
    ]
    expected_texts = [block.model_dump() for block in texts]
    assert expected_texts[0]["citations"] is None
    responses = [
        _tool_blocks(call_id="toolu_1", texts=texts),
        [TextBlock(type="text", text="first done")],
        _tool_blocks(call_id="toolu_2", texts=texts),
        [TextBlock(type="text", text="second done")],
    ]
    requests = []
    model_io = AnthropicModelIO(
        model=model,
        api_key="fixture",
        client_factory=_client_factory(responses, requests),
    )
    toolkit = Toolkit()
    effects = []

    def demo_tool(x: int):
        effects.append(x)
        return x + 1

    toolkit.register(demo_tool, name="demo_tool")
    messages = [{"role": "user", "content": "first request"}]
    first = build_runtime_loop(model_io=model_io).run(
        messages, provider="anthropic", model=model, toolkit=toolkit, max_iterations=3
    )
    assert first.status == "completed"
    assert first.messages[-1] == {"role": "assistant", "content": "first done"}
    assert effects == [2]
    replayed = requests[1]["messages"][-2]["content"]
    assert [block for block in replayed if block["type"] == "text"] == expected_texts
    assert replayed[0]["signature"] == "signed-data"
    assert "signed-data" not in json.dumps(first.messages)

    followup = build_runtime_loop(model_io=model_io).run(
        [*first.messages, {"role": "user", "content": "second request"}],
        provider="anthropic", model=model, toolkit=toolkit, max_iterations=3,
    )
    assert followup.status == "completed"
    assert followup.messages[-1] == {"role": "assistant", "content": "second done"}
    assert effects == [2, 2]
    replayed = requests[3]["messages"][-2]["content"]
    assert [block for block in replayed if block["type"] == "text"] == expected_texts
    assert replayed[0]["signature"] == "signed-data"
    assert "signed-data" not in json.dumps(followup.messages)


def test_sdk_text_only_final_text_is_unchanged():
    requests = []
    model_io = AnthropicModelIO(
        model="claude-sonnet-5", api_key="fixture",
        client_factory=_client_factory([[TextBlock(type="text", text="  Hello  ")]], requests),
    )
    turn = model_io.fetch_turn(ModelTurnRequest(
        messages=[{"role": "user", "content": "hello"}], toolkit=Toolkit(),
    ))
    assert turn.final_text == "Hello"
    assert turn.assistant_messages == [{"role": "assistant", "content": "Hello"}]


def test_real_sdk_text_block_does_not_poison_parallel_tool_followup():
    observed_requests: list[dict] = []
    tool_urls: list[str] = []
    model_io = AnthropicModelIO(
        model="claude-opus-4-6",
        api_key="offline-no-secret",
        client_factory=_real_sdk_client_factory(
            responses=[_real_sdk_parallel_tool_turn(), _real_sdk_final_text_turn()],
            observed_requests=observed_requests,
        ),
        default_payloads={},
        model_capabilities={},
    )
    toolkit = Toolkit()

    def web_fetch(url: str):
        tool_urls.append(url)
        return {"url": url, "status": "ready"}

    toolkit.register(web_fetch, name="web_fetch")

    completed = build_runtime_loop(model_io=model_io).run(
        [{"role": "user", "content": "Fetch both pages."}],
        provider="anthropic",
        model="claude-opus-4-6",
        toolkit=toolkit,
        max_iterations=3,
    )

    assert completed.status == "completed"
    assert completed.messages[-1] == {
        "role": "assistant",
        "content": "Both pages are ready.",
    }
    assert tool_urls == ["https://example.com/1", "https://example.com/2"]
    assert len(observed_requests) == 2
    replayed_content = observed_requests[1]["messages"][-2]["content"]
    replayed_text = next(block for block in replayed_content if block["type"] == "text")
    assert replayed_text == {
        "type": "text",
        "text": "Checking both pages.",
        "citations": None,
    }
    assert "parsed_output" not in json.dumps(observed_requests[1])


def test_sdk_populated_parsed_output_never_reaches_semantic_or_replay():
    parsed_text = ParsedTextBlock[dict](
        type="text",
        text="I will use the tool.",
        citations=None,
        parsed_output={"internal": {"answer": "ready"}},
    )
    requests = []
    model_io = AnthropicModelIO(
        model="claude-opus-4-6",
        api_key="fixture",
        client_factory=_client_factory(
            [_tool_blocks(call_id="toolu_parsed_output", texts=[parsed_text])],
            requests,
        ),
    )

    turn = model_io.fetch_turn(ModelTurnRequest(
        messages=[{"role": "user", "content": "Use the tool."}],
        toolkit=Toolkit(),
    ))

    semantic_text = next(
        block for block in turn.assistant_messages[0]["content"]
        if block["type"] == "text"
    )
    replay_text = next(
        block for block in turn.provider_replay_frame["items"][-1]["content"]
        if block["type"] == "text"
    )
    assert semantic_text == replay_text == {
        "type": "text",
        "text": "I will use the tool.",
        "citations": None,
    }


def test_legacy_parsed_output_replays_strictly_but_is_removed_from_outbound_copy():
    recorded_requests = []
    source_model = AnthropicModelIO(
        model="claude-opus-4-6",
        api_key="fixture",
        client_factory=_client_factory(
            [_tool_blocks(
                call_id="toolu_legacy_replay",
                texts=[TextBlock(type="text", text="I will use the tool.")],
            )],
            recorded_requests,
        ),
    )
    recorded_turn = source_model.fetch_turn(ModelTurnRequest(
        messages=[{"role": "user", "content": "Use the tool."}], toolkit=Toolkit(),
    ))
    historical_assistant = copy.deepcopy(recorded_turn.assistant_messages[0])
    historical_frame = copy.deepcopy(recorded_turn.provider_replay_frame)
    historical_assistant["content"][0]["parsed_output"] = {"old": "sdk-state"}
    historical_frame["items"][-1]["content"][1]["parsed_output"] = {
        "old": "sdk-state"
    }

    state = RunState()
    state.seed_messages([
        {"role": "user", "content": "Use the tool."},
        historical_assistant,
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": "toolu_legacy_replay",
            "content": "ready",
        }]},
    ])
    state.provider_state.provider = "anthropic"
    set_provider_replay_frame(state, historical_frame)

    assembled = build_model_turn_request(state)

    assembled_assistant = next(
        message for message in assembled.messages
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
    )
    assert assembled_assistant["content"][1]["parsed_output"] == {"old": "sdk-state"}
    assert historical_frame["items"][-1]["content"][1]["parsed_output"] == {
        "old": "sdk-state"
    }

    outbound_requests = []
    outbound_model = AnthropicModelIO(
        model="claude-opus-4-6",
        api_key="fixture",
        client_factory=_client_factory(
            [[TextBlock(type="text", text="complete")]], outbound_requests
        ),
    )
    outbound_model.fetch_turn(assembled)

    outbound_assistant = next(
        message for message in outbound_requests[0]["messages"]
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
    )
    assert outbound_assistant["content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "signed-data"},
        {"type": "text", "text": "I will use the tool.", "citations": None},
        {
            "type": "tool_use",
            "id": "toolu_legacy_replay",
            "name": "demo_tool",
            "input": {"x": 2},
        },
    ]
    assert historical_assistant["content"][0]["parsed_output"] == {"old": "sdk-state"}


def test_sdk_text_blocks_survive_cold_checkpoint_without_duplicate_tool(tmp_path):
    texts = [TextBlock(type="text", text=""), TextBlock(type="text", text="working")]
    expected_texts = [block.model_dump() for block in texts]
    effects = []

    def demo_tool(x: int):
        effects.append(x)
        return x + 1

    first_memory = MemoryManager(store=JsonFileSessionStore(tmp_path))
    first_agent = Agent(
        name="anthropic-sdk-text-cold", provider="anthropic", model="claude-sonnet-5",
        modules=(ToolsModule(tools=(demo_tool,)), MemoryModule(memory=first_memory)),
        model_io_factory=lambda spec, context: AnthropicModelIO(
            model="claude-sonnet-5", api_key="fixture",
            client_factory=_client_factory([_tool_blocks(call_id="toolu_1", texts=texts)], []),
        ),
    )
    stopped = first_agent.run("call tool", session_id="sdk-text-cold", max_iterations=1)
    assert stopped.status == "max_iterations"
    assert effects == [2]
    checkpoint = first_memory.store.load("sdk-text-cold")["execution_checkpoint"]
    replayed = checkpoint["replay_frame"]["items"][-2]["content"]
    assert [block for block in replayed if block["type"] == "text"] == expected_texts

    requests = []
    resumed_memory = MemoryManager(store=JsonFileSessionStore(tmp_path))
    resumed_agent = Agent(
        name="anthropic-sdk-text-cold", provider="anthropic", model="claude-sonnet-5",
        modules=(ToolsModule(tools=(demo_tool,)), MemoryModule(memory=resumed_memory)),
        model_io_factory=lambda spec, context: AnthropicModelIO(
            model="claude-sonnet-5", api_key="fixture",
            client_factory=_client_factory([[TextBlock(type="text", text="done")]], requests),
        ),
    )
    completed = resumed_agent.run("continue", session_id="sdk-text-cold", max_iterations=1)
    assert completed.status == "completed"
    assert effects == [2]
    replayed = next(
        message["content"] for message in requests[0]["messages"]
        if message.get("role") == "assistant"
        and isinstance(message.get("content"), list)
        and any(block.get("type") == "thinking" for block in message["content"])
    )
    assert [block for block in replayed if block["type"] == "text"] == expected_texts
    assert replayed[0]["signature"] == "signed-data"
    assert "signed-data" not in json.dumps(completed.messages)
    assert "execution_checkpoint" not in resumed_memory.store.load("sdk-text-cold")


def test_unknown_text_extension_is_rejected_before_provider_continuation():
    text = TextBlock(type="text", text="working")
    text.__pydantic_extra__ = {"extension": {"nested": [1, None]}}
    requests = []
    responses = [_tool_blocks(call_id="toolu_1", texts=[text])]
    model_io = AnthropicModelIO(
        model="claude-sonnet-5", api_key="fixture",
        client_factory=_client_factory(responses, requests),
    )
    effects = []
    toolkit = Toolkit()

    def demo_tool(x: int):
        effects.append(x)
        return x + 1

    toolkit.register(demo_tool, name="demo_tool")
    with pytest.raises(ProviderMessageContractError, match="unknown fields: extension"):
        build_runtime_loop(model_io=model_io).run(
            [{"role": "user", "content": "call tool"}],
            provider="anthropic", model="claude-sonnet-5", toolkit=toolkit,
            max_iterations=3,
        )
    assert effects == [2]
    assert len(requests) == 1


@pytest.mark.parametrize("mutation", [
    "text", "citations", "arguments", "name", "id", "signature",
])
def test_sdk_text_replay_rejects_mutation_before_provider_continuation(mutation):
    responses = [_tool_blocks(
        call_id="toolu_1", texts=[TextBlock(type="text", text="working")],
    )]
    requests = []
    model_io = AnthropicModelIO(
        model="claude-sonnet-5", api_key="fixture",
        client_factory=_client_factory(responses, requests),
    )
    turn = model_io.fetch_turn(ModelTurnRequest(
        messages=[{"role": "user", "content": "call tool"}], toolkit=Toolkit(),
    ))
    semantic = copy.deepcopy(turn.assistant_messages[0])
    frame = copy.deepcopy(turn.provider_replay_frame)
    if mutation == "text":
        semantic["content"][0]["text"] = "changed"
    elif mutation == "citations":
        semantic["content"][0]["citations"] = []
    elif mutation == "arguments":
        semantic["content"][1]["input"] = {"x": 3}
    elif mutation == "name":
        semantic["content"][1]["name"] = "other_tool"
    elif mutation == "id":
        semantic["content"][1]["id"] = "toolu_other"
    else:
        frame["items"][-1]["content"][0].pop("signature")

    state = RunState()
    state.seed_messages([
        {"role": "user", "content": "call tool"},
        semantic,
        {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "toolu_1", "content": "3",
        }]},
    ])
    state.provider_state.provider = "anthropic"
    set_provider_replay_frame(state, frame)
    with pytest.raises(ProviderContextProjectionError):
        build_model_turn_request(state)
    assert len(requests) == 1
