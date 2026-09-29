from __future__ import annotations

import copy
import json

import pytest
from anthropic.types import CitationCharLocation, TextBlock, ThinkingBlock, ToolUseBlock

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


def _tool_blocks(*, call_id: str, texts: list[TextBlock]):
    return [
        ThinkingBlock(type="thinking", thinking="plan", signature="signed-data"),
        *texts,
        ToolUseBlock(type="tool_use", id=call_id, name="demo_tool", input={"x": 2}),
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
