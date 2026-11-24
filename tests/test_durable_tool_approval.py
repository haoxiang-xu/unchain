from __future__ import annotations

import pytest

from unchain.interaction.durable import (
    INTERACTION_JOURNAL_KEY,
    InteractionIntegrityError,
    InteractionNotPendingError,
)
from unchain.interaction.runtime import DurableInteractionRuntime
from unchain.kernel import ModelTurnResult
from unchain.kernel.model_tool_boundary import (
    FinalModelToolPreparation,
    _issue_final_model_tool_boundary,
)
from unchain.kernel.types import ToolCall
from unchain.memory import InMemorySessionStore, KernelMemoryRuntime
from unchain.runtime import build_runtime_loop
from unchain.tools import Toolkit
from unchain.tools.exposure import ToolExposureRuntime, ToolOptimizerConfig


class _QueueModelIO:
    provider = "openai"
    model = "gpt-5"

    def __init__(self, turns: list[ModelTurnResult]):
        self.turns = list(turns)
        self.requests = []

    def fetch_turn(self, request):
        self.requests.append(request)
        if not self.turns:
            raise AssertionError("unexpected model turn")
        return self.turns.pop(0)


def _tool_turn(*calls: ToolCall) -> ModelTurnResult:
    return ModelTurnResult(
        assistant_messages=[
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in calls
        ],
        tool_calls=list(calls),
        response_id="resp-tools",
    )


def _final_turn() -> ModelTurnResult:
    return ModelTurnResult(
        assistant_messages=[{"role": "assistant", "content": "done"}],
        tool_calls=[],
        final_text="done",
        response_id="resp-final",
    )


def _toolkit(safe_calls: list[int], dangerous_calls: list[str]) -> Toolkit:
    toolkit = Toolkit()

    def safe_tool() -> dict:
        safe_calls.append(1)
        return {"safe": True}

    def dangerous_tool(value: str) -> dict:
        dangerous_calls.append(value)
        return {"written": value}

    toolkit.register(safe_tool, name="safe_tool")
    toolkit.register(
        dangerous_tool,
        name="dangerous_tool",
        requires_confirmation=True,
    )
    return toolkit


def _register_final_boundary(loop, toolkit: Toolkit) -> None:
    loop.register_harness(
        _issue_final_model_tool_boundary(
            prepare=lambda context: FinalModelToolPreparation(
                model_toolkit=toolkit,
                execution_toolkit=toolkit,
                execution_binding=object(),
            ),
            validate=lambda context, preparation, turn: turn,
        )
    )


def test_boundary_enabled_sync_tool_approval_resume_fails_closed() -> None:
    session_id = "final-boundary-sync-approval"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    model = _QueueModelIO(
        [
            _tool_turn(
                ToolCall(
                    call_id="danger-1",
                    name="dangerous_tool",
                    arguments={"value": "must-not-run"},
                )
            ),
            _final_turn(),
        ]
    )
    loop = build_runtime_loop(
        model_io=model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    _register_final_boundary(loop, toolkit)
    loop._final_model_tool_boundary = None

    with pytest.raises(
        InteractionIntegrityError,
        match="authenticated durable tool binding recovery",
    ):
        loop.run(
            [{"role": "user", "content": "run it"}],
            session_id=session_id,
            provider="openai",
            model="gpt-5",
            toolkit=toolkit,
            on_tool_confirm=lambda request: {"approved": True},
        )

    assert dangerous_calls == []
    assert len(model.requests) == 1


def test_boundary_enabled_cold_tool_approval_resume_fails_closed() -> None:
    session_id = "final-boundary-cold-approval"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(
                        call_id="danger-1",
                        name="dangerous_tool",
                        arguments={"value": "must-not-run"},
                    )
                )
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    _register_final_boundary(first_loop, toolkit)
    suspended = first_loop.run(
        [{"role": "user", "content": "run it"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=toolkit,
    )
    assert suspended.status == "awaiting_interaction"
    assert suspended.run_bundle is not None
    assert suspended.run_bundle["lifecycle"]["status"] == "suspended"
    assert suspended.run_bundle["metrics"]["all"]["iterations"] == 1
    iteration_events = [
        event
        for event in suspended.run_bundle["metrics"]["events"]
        if event["kind"] == "iteration"
    ]
    assert len(iteration_events) == 1
    assert iteration_events[0]["outcome"] == "uncertain"

    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store)
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response={"approved": True},
        expected_revision=pending.session_snapshot.revision,
    )
    resume_model = _QueueModelIO([_final_turn()])
    resume_loop = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    _register_final_boundary(resume_loop, toolkit)
    resume_loop._final_model_tool_boundary = None

    with pytest.raises(
        InteractionIntegrityError,
        match="authenticated durable tool binding recovery",
    ):
        resume_loop.resume_interaction(
            session_id=session_id,
            response=None,
            toolkit=toolkit,
        )

    assert dangerous_calls == []
    assert resume_model.requests == []


def test_tool_approval_cold_resume_does_not_repeat_prior_batch_tools() -> None:
    session_id = "durable-tool-approval"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    first_model = _QueueModelIO(
        [
            _tool_turn(
                ToolCall(call_id="safe-1", name="safe_tool", arguments={}),
                ToolCall(
                    call_id="danger-1",
                    name="dangerous_tool",
                    arguments={"value": "approved"},
                ),
            )
        ]
    )
    first_memory = KernelMemoryRuntime.from_config(store=store)
    first_loop = build_runtime_loop(
        model_io=first_model,
        memory_runtime=first_memory,
    )

    suspended = first_loop.run(
        [{"role": "user", "content": "run both"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=toolkit,
        max_iterations=3,
    )
    assert suspended.status == "awaiting_interaction"
    assert suspended.interaction_request is not None
    assert safe_calls == [1]
    assert dangerous_calls == []

    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store),
        clock_ms=lambda: 100,
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response={"approved": True},
        submitted_by="ui:test",
        expected_revision=pending.session_snapshot.revision,
    )

    resume_model = _QueueModelIO([_final_turn()])
    resume_loop = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    resumed = resume_loop.resume_interaction(
        session_id=session_id,
        response=None,
        toolkit=toolkit,
    )

    assert resumed.status == "completed"
    assert safe_calls == [1]
    assert dangerous_calls == ["approved"]
    assert len(resume_model.requests) == 1
    journal = store.load(session_id)[INTERACTION_JOURNAL_KEY]
    assert journal["active_id"] is None
    assert journal["entries"][pending.request.interaction_id]["application"] is not None


def test_sync_tool_callback_is_adapter_over_durable_receipt() -> None:
    session_id = "durable-tool-sync"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(
                        call_id="danger-1",
                        name="dangerous_tool",
                        arguments={"value": "sync"},
                    )
                ),
                _final_turn(),
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )

    def approve(request):
        state = store.load(session_id)
        assert state["execution_checkpoint"]["status"] == "awaiting_interaction"
        journal = state[INTERACTION_JOURNAL_KEY]
        entry = journal["entries"][journal["active_id"]]
        assert entry["receipt"] is None
        assert dangerous_calls == []
        assert request.tool_name == "dangerous_tool"
        return {"approved": True}

    result = loop.run(
        [{"role": "user", "content": "run it"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=toolkit,
        on_tool_confirm=approve,
        max_iterations=3,
    )
    assert result.status == "completed"
    assert dangerous_calls == ["sync"]
    entry = next(iter(store.load(session_id)[INTERACTION_JOURNAL_KEY]["entries"].values()))
    assert entry["receipt"]["submitted_by"] == "callback:on_tool_confirm"
    assert entry["application"] is not None


def test_sync_callback_cannot_reapply_receipt_consumed_by_cold_worker() -> None:
    session_id = "durable-tool-callback-takeover"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    original_model = _QueueModelIO(
        [
            _tool_turn(
                ToolCall(
                    call_id="danger-1",
                    name="dangerous_tool",
                    arguments={"value": "once"},
                )
            ),
            _final_turn(),
        ]
    )
    original_loop = build_runtime_loop(
        model_io=original_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    cold_model = _QueueModelIO([_final_turn()])

    def cold_worker_consumes_then_callback_returns(_request):
        interaction = DurableInteractionRuntime(
            KernelMemoryRuntime.from_config(store=store)
        )
        pending = interaction.load_active(session_id)
        interaction.record_receipt(
            session_id,
            interaction_id=pending.request.interaction_id,
            response=True,
            expected_revision=pending.session_snapshot.revision,
        )
        result = build_runtime_loop(
            model_io=cold_model,
            memory_runtime=KernelMemoryRuntime.from_config(store=store),
        ).resume_interaction(
            session_id=session_id,
            response=None,
            toolkit=toolkit,
        )
        assert result.status == "completed"
        return True

    with pytest.raises(InteractionNotPendingError, match="already applied"):
        original_loop.run(
            [{"role": "user", "content": "run it"}],
            session_id=session_id,
            provider="openai",
            model="gpt-5",
            toolkit=toolkit,
            on_tool_confirm=cold_worker_consumes_then_callback_returns,
            max_iterations=3,
        )

    assert dangerous_calls == ["once"]
    assert len(cold_model.requests) == 1
    assert len(original_model.requests) == 1


def test_tool_wait_is_durable_before_response_event_callback() -> None:
    session_id = "durable-tool-response-event-failure"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(call_id="safe-1", name="safe_tool", arguments={}),
                    ToolCall(
                        call_id="danger-1",
                        name="dangerous_tool",
                        arguments={"value": "approved"},
                    ),
                )
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )

    def failing_event_sink(event: dict) -> None:
        if event.get("type") == "response_received":
            raise RuntimeError("event sink unavailable")

    with pytest.raises(RuntimeError, match="event sink unavailable"):
        first_loop.run(
            [{"role": "user", "content": "run both"}],
            session_id=session_id,
            provider="openai",
            model="gpt-5",
            toolkit=toolkit,
            callback=failing_event_sink,
            max_iterations=3,
        )

    persisted = store.load(session_id)
    assert persisted["execution_checkpoint"]["status"] == "awaiting_interaction"
    journal = persisted[INTERACTION_JOURNAL_KEY]
    interaction_id = journal["active_id"]
    assert interaction_id
    assert journal["entries"][interaction_id]["receipt"] is None
    assert safe_calls == [1]
    assert dangerous_calls == []

    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store)
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response=True,
        expected_revision=pending.session_snapshot.revision,
    )
    resume_model = _QueueModelIO([_final_turn()])
    resumed = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    ).resume_interaction(
        session_id=session_id,
        response=None,
        toolkit=toolkit,
    )

    assert resumed.status == "completed"
    assert safe_calls == [1]
    assert dangerous_calls == ["approved"]
    assert len(resume_model.requests) == 1


def test_changed_tool_schema_cannot_consume_old_approval() -> None:
    session_id = "durable-tool-schema-change"
    store = InMemorySessionStore()
    original_calls: list[str] = []
    original = Toolkit()

    def original_tool(value: str) -> dict:
        original_calls.append(value)
        return {"value": value}

    original.register(
        original_tool,
        name="dangerous_tool",
        requires_confirmation=True,
    )
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(
                        call_id="danger-1",
                        name="dangerous_tool",
                        arguments={"value": "old"},
                    )
                )
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    first_loop.run(
        [{"role": "user", "content": "run"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=original,
    )
    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store)
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response=True,
        expected_revision=pending.session_snapshot.revision,
    )

    changed_calls: list[str] = []
    changed = Toolkit()

    def changed_tool(value: str, force: bool = False) -> dict:
        changed_calls.append(value)
        return {"value": value, "force": force}

    changed.register(
        changed_tool,
        name="dangerous_tool",
        requires_confirmation=True,
    )
    resume_model = _QueueModelIO([_final_turn()])
    resume_loop = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    with pytest.raises(InteractionIntegrityError, match="prompt/tool schema"):
        resume_loop.resume_interaction(
            session_id=session_id,
            response=None,
            toolkit=changed,
        )
    assert changed_calls == []
    assert resume_model.requests == []


def test_removed_confirmation_policy_cannot_consume_old_approval() -> None:
    session_id = "durable-tool-confirmation-removed"
    store = InMemorySessionStore()
    original = Toolkit()
    original.register(
        lambda value: {"value": value},
        name="dangerous_tool",
        requires_confirmation=True,
    )
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(
                        call_id="danger-1",
                        name="dangerous_tool",
                        arguments={"value": "old"},
                    )
                )
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    suspended = first_loop.run(
        [{"role": "user", "content": "run"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=original,
    )
    assert suspended.status == "awaiting_interaction"

    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store)
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response=True,
        expected_revision=pending.session_snapshot.revision,
    )

    changed_calls: list[str] = []
    changed = Toolkit()

    def changed_tool(value: str) -> dict:
        changed_calls.append(value)
        return {"value": value}

    changed.register(
        changed_tool,
        name="dangerous_tool",
        requires_confirmation=False,
    )
    resume_model = _QueueModelIO([_final_turn()])
    resume_loop = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )

    with pytest.raises(InteractionIntegrityError, match="confirmation policy"):
        resume_loop.resume_interaction(
            session_id=session_id,
            response=None,
            toolkit=changed,
        )

    assert changed_calls == []
    assert resume_model.requests == []


def test_changed_model_io_cannot_consume_old_approval() -> None:
    session_id = "durable-tool-model-change"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(
                        call_id="danger-1",
                        name="dangerous_tool",
                        arguments={"value": "old"},
                    )
                )
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    suspended = first_loop.run(
        [{"role": "user", "content": "run"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=toolkit,
    )
    assert suspended.status == "awaiting_interaction"

    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store)
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response=True,
        expected_revision=pending.session_snapshot.revision,
    )

    changed_model = _QueueModelIO([_final_turn()])
    changed_model.model = "gpt-6"
    changed_loop = build_runtime_loop(
        model_io=changed_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )

    with pytest.raises(InteractionIntegrityError, match="provider/model"):
        changed_loop.resume_interaction(
            session_id=session_id,
            response=None,
            toolkit=toolkit,
        )

    assert dangerous_calls == []
    assert changed_model.requests == []


def test_deferred_tool_plugin_cannot_bypass_durable_approval() -> None:
    session_id = "durable-deferred-tool"
    store = InMemorySessionStore()
    dangerous_calls: list[str] = []
    full = Toolkit()
    full.register(lambda: {"safe": True}, name="safe_tool")

    def dangerous_tool(path: str) -> dict:
        dangerous_calls.append(path)
        return {"wrote": path}

    full.register(
        dangerous_tool,
        name="dangerous_tool",
        requires_confirmation=True,
        search_hint="dangerous write",
    )

    class _SelectorModel:
        provider = "openai"
        model = "gpt-5"

        def fetch_turn(self, _request):
            return ModelTurnResult(
                assistant_messages=[
                    {"role": "assistant", "content": '{"tool_names":["safe_tool"]}'}
                ],
                tool_calls=[],
                final_text='{"tool_names":["safe_tool"]}',
            )

    def exposure():
        runtime = ToolExposureRuntime(
            config=ToolOptimizerConfig(max_direct_tools=1, trigger_tool_count=1),
            full_toolkit=full,
            model_io=_SelectorModel(),
            provider="openai",
            model="gpt-5",
            messages=[{"role": "user", "content": "write"}],
        )
        exposed = runtime.prepare()
        return exposed, runtime.build_plugins()

    exposed, plugins = exposure()
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO(
            [
                _tool_turn(
                    ToolCall(
                        call_id="deferred-1",
                        name="tool_execute_deferred",
                        arguments={
                            "tool_name": "dangerous_tool",
                            "arguments": {"path": "secret.txt"},
                        },
                    )
                )
            ]
        ),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    suspended = first_loop.run(
        [{"role": "user", "content": "write"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=exposed,
        tool_runtime_plugins=plugins,
    )
    assert suspended.status == "awaiting_interaction"
    assert suspended.interaction_request["payload"]["tool_name"] == "dangerous_tool"
    assert dangerous_calls == []

    interaction = DurableInteractionRuntime(
        KernelMemoryRuntime.from_config(store=store)
    )
    pending = interaction.load_active(session_id)
    interaction.record_receipt(
        session_id,
        interaction_id=pending.request.interaction_id,
        response={"approved": False, "reason": "no"},
        expected_revision=pending.session_snapshot.revision,
    )

    resumed_exposed, resumed_plugins = exposure()
    resume_model = _QueueModelIO([_final_turn()])
    resume_loop = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    resumed = resume_loop.resume_interaction(
        session_id=session_id,
        response=None,
        toolkit=resumed_exposed,
        tool_runtime_plugins=resumed_plugins,
    )
    assert resumed.status == "completed"
    assert dangerous_calls == []
    assert len(resume_model.requests) == 1


def test_resume_announces_next_approval_in_same_batch() -> None:
    """A batch with two approvals: resuming the first must announce the second.

    The live loop emits ``interaction_requested`` for every wait it opens;
    the resume path opened the second wait silently, so a host whose
    receipts require the canonical request (PuPu Skeleton) could never
    accept that approval.
    """
    session_id = "durable-tool-approval-batch"
    store = InMemorySessionStore()
    safe_calls: list[int] = []
    dangerous_calls: list[str] = []
    toolkit = _toolkit(safe_calls, dangerous_calls)
    first_loop = build_runtime_loop(
        model_io=_QueueModelIO([
            _tool_turn(
                ToolCall(call_id="danger-1", name="dangerous_tool", arguments={"value": "one"}),
                ToolCall(call_id="danger-2", name="dangerous_tool", arguments={"value": "two"}),
            )
        ]),
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    )
    first_events: list[dict] = []
    suspended = first_loop.run(
        [{"role": "user", "content": "run both"}],
        session_id=session_id,
        provider="openai",
        model="gpt-5",
        toolkit=toolkit,
        max_iterations=3,
        callback=first_events.append,
    )
    assert suspended.status == "awaiting_interaction"
    first_id = suspended.interaction_request["interaction_id"]
    assert [e["interaction_request"]["interaction_id"] for e in first_events
        if e.get("type") == "interaction_requested"] == [first_id]

    interaction = DurableInteractionRuntime(KernelMemoryRuntime.from_config(store=store), clock_ms=lambda: 100)
    pending = interaction.load_active(session_id)
    interaction.record_receipt(session_id, interaction_id=first_id, response={"approved": True},
        submitted_by="ui:test", expected_revision=pending.session_snapshot.revision)

    resume_events: list[dict] = []
    resume_model = _QueueModelIO([])
    resumed = build_runtime_loop(
        model_io=resume_model,
        memory_runtime=KernelMemoryRuntime.from_config(store=store),
    ).resume_interaction(session_id=session_id, response=None, toolkit=toolkit,
        callback=resume_events.append)

    assert resumed.status == "awaiting_interaction"
    second_id = resumed.interaction_request["interaction_id"]
    assert second_id != first_id
    assert dangerous_calls == ["one"]
    assert resume_model.requests == []
    announced = [e for e in resume_events if e.get("type") == "interaction_requested"]
    assert [e["interaction_request"]["interaction_id"] for e in announced] == [second_id]
    assert announced[0]["interaction_request"] == resumed.interaction_request
