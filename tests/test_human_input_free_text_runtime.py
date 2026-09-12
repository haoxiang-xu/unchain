"""Real question-tool suspension, error results and durable free-text resume."""

import json

import pytest

from unchain.agent import Agent, MemoryModule, ToolsModule
from unchain.input import build_ask_user_question_tool
from unchain.interaction.runtime import DurableInteractionRuntime
from unchain.kernel import ModelTurnResult, ToolCall
from unchain.memory import JsonFileSessionStore, KernelMemoryRuntime


def _arguments(mode="single"):
    return {
        "title": "Project folder",
        "question": "What folder path should I use?",
        "selection_mode": mode,
        "options": [],
        "allow_other": True,
        "other_label": "Folder path",
    }


def _question(call_id, arguments):
    return ModelTurnResult(
        assistant_messages=[{
            "type": "function_call", "call_id": call_id,
            "name": "ask_user_question", "arguments": json.dumps(arguments),
        }],
        tool_calls=[ToolCall(call_id=call_id, name="ask_user_question", arguments=arguments)],
        response_id=f"response-{call_id}",
    )


def _done():
    return ModelTurnResult(
        assistant_messages=[{"role": "assistant", "content": "done"}],
        tool_calls=[], final_text="done",
    )


class _Model:
    provider = "openai"
    model = "gpt-5"

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def fetch_turn(self, request):
        self.requests.append(request)
        assert self.turns, "unexpected provider turn (question must not replay)"
        return self.turns.pop(0)


def _agent(model, store=None):
    modules = [ToolsModule(tools=(build_ask_user_question_tool(),))]
    if store is not None:
        modules.append(MemoryModule(memory=KernelMemoryRuntime.from_config(store=store)))
    return Agent(
        name="free-text-regression", provider="openai", model="gpt-5",
        modules=tuple(modules), model_io_factory=lambda _spec, _context: model,
    )


def _reply(request_id, text):
    return {"request_id": request_id, "selected_values": ["__other__"], "other_text": text}


def _outputs(messages):
    return [json.loads(m["output"]) for m in messages if m.get("type") == "function_call_output"]


@pytest.mark.parametrize("mode", ["single", "multiple"])
def test_two_free_text_questions_resume_with_distinct_request_ids(mode):
    model = _Model([_question("folder-1", _arguments(mode)), _question("folder-2", _arguments(mode)), _done(), _done()])
    agent = _agent(model)
    first = agent.run("ask for the destination", session_id="two-questions", max_iterations=5)
    assert first.status == "awaiting_human_input"
    assert first.human_input_request["options"] == []
    assert set(first.human_input_request) == {
        "request_id", "kind", "title", "question", "selection_mode", "options",
        "allow_other", "other_label", "other_placeholder", "min_selected", "max_selected",
    }
    assert first.human_input_request["kind"] == "selector"

    with pytest.raises(ValueError, match="other_text"):
        agent.resume_human_input(conversation=first.messages, continuation=first.continuation,
                                 response=_reply("folder-1", "  "), session_id="two-questions")
    assert len(model.requests) == 1
    second = agent.resume_human_input(
        conversation=first.messages, continuation=first.continuation,
        response=_reply("folder-1", " /Users/example/projects/first "), session_id="two-questions",
    )
    assert second.status == "awaiting_human_input"
    assert second.human_input_request["request_id"] == "folder-2"
    with pytest.raises(ValueError, match="request_id"):
        agent.resume_human_input(conversation=second.messages, continuation=second.continuation,
                                 response=_reply("folder-1", "wrong question"), session_id="two-questions")
    assert len(model.requests) == 2
    result = agent.resume_human_input(
        conversation=second.messages, continuation=second.continuation,
        response=_reply("folder-2", "~/projects/second"), session_id="two-questions",
    )
    assert result.status == "completed"
    # OpenAI remote continuation sends each answer as a delta, not full history.
    outputs = [output for request in model.requests[1:] for output in _outputs(request.messages)]
    assert outputs == [
        {"submitted": True, "selected_values": ["__other__"], "other_text": "/Users/example/projects/first"},
        {"submitted": True, "selected_values": ["__other__"], "other_text": "~/projects/second"},
    ]
    assert agent.run("a normal follow-up", session_id="two-questions").status == "completed"


@pytest.mark.parametrize("invalid", [
    {**_arguments(), "allow_other": False},
    {**_arguments(), "unexpected": "not in the advertised schema"},
    {**_arguments(), "options": [{"label": "A", "value": "a", "unknown": True}]},
])
def test_invalid_question_returns_error_tool_result_without_suspending(invalid):
    model = _Model([_question("bad-question", invalid), _done()])
    events = []
    result = _agent(model).run("ask me", max_iterations=3, callback=events.append)
    assert result.status == "completed"
    assert result.human_input_request is None
    outputs = _outputs(model.requests[-1].messages)
    assert len(outputs) == 1
    assert set(outputs[0]) == {"error", "tool"}
    assert outputs[0]["tool"] == "ask_user_question"
    assert outputs[0]["error"]
    assert all(e.get("type") != "human_input_requested" for e in events)


@pytest.mark.parametrize("mode", ["single", "multiple"])
def test_free_text_receipt_survives_fresh_disk_store_and_agent(mode, tmp_path):
    session_id = "disk-free-text"
    first_model = _Model([_question("folder-cold", _arguments(mode))])
    first = _agent(first_model, JsonFileSessionStore(tmp_path)).run(
        "ask for a folder", session_id=session_id, max_iterations=4,
    )
    assert first.status == "awaiting_human_input"

    runtime = DurableInteractionRuntime(KernelMemoryRuntime.from_config(store=JsonFileSessionStore(tmp_path)))
    pending = runtime.load_active(session_id)
    assert pending.request.payload == first.human_input_request
    reply = _reply("folder-cold", " /Users/example/恢复项目 ")
    receipt = runtime.record_receipt(
        session_id, interaction_id=pending.request.interaction_id, response=reply,
        submitted_by="ui:test", expected_revision=pending.session_snapshot.revision,
    )
    retried = runtime.record_receipt(
        session_id, interaction_id=pending.request.interaction_id, response=reply,
        submitted_by="ui:test", expected_revision=pending.session_snapshot.revision,
    )
    assert retried.receipt == receipt.receipt
    assert retried.session_snapshot.revision == receipt.session_snapshot.revision

    second_model = _Model([_done()])
    result = _agent(second_model, JsonFileSessionStore(tmp_path)).resume_human_input(session_id=session_id)
    assert result.status == "completed"
    assert len(second_model.requests) == 1
    assert _outputs(second_model.requests[0].messages) == [{
        "submitted": True, "selected_values": ["__other__"], "other_text": "/Users/example/恢复项目",
    }]
