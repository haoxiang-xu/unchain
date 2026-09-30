from __future__ import annotations

import pickle

import pytest

from tests.context_v2.test_context_runtime_factory import (
    _bundle,
    _context,
    _current_input,
)
from unchain.context import (
    ContextRuntime,
    DurableContextRuntimeFactory,
    DurableToolExecutorContractError,
)
from unchain.events import RuntimeEventBridge
from unchain.execution import ExecutionRuntime
from unchain.kernel.harness import HarnessContext
from unchain.kernel.types import ToolCall
from unchain.memory import InMemorySessionStore
from unchain.tools import Toolkit


def _factory_tool_runtime(
    *,
    execution_id: str = "execution-1",
    attempt_id: str = "attempt-1",
    call_id: str = "call-1",
    persist_intent: bool = True,
    tool_result: dict | None = None,
    tool_output_management_active: bool = False,
):
    bundles = {}

    def build(attempt):
        bundle = _bundle(attempt)
        bundles[(attempt.generation.execution_id, attempt.attempt_id)] = bundle
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id=f"context-v2-{attempt_id}",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, resolved_execution_id: (
                f"generation-{resolved_execution_id}"
            ),
            current_input_resolver=_current_input,
        ),
        tool_output_management_active=tool_output_management_active,
    )
    guard = ExecutionRuntime(InMemorySessionStore()).acquire(
        execution_id,
        f"owner-{attempt_id}",
    )
    bootstrap = _context(
        session_id=execution_id,
        run_id=attempt_id,
        current_input=None,
    )
    bootstrap.event["execution_guard"] = guard
    runtime.build_harnesses()[0].build_delta(bootstrap)
    bundle = bundles[(execution_id, attempt_id)]

    invocations = []
    forged_handler_calls = []
    toolkit = Toolkit()

    @toolkit.tool(name="lookup")
    def lookup(query: str):
        invocations.append(query)
        return dict(tool_result or {"seen": query})

    arguments = {"query": "journal-owned"}
    if persist_intent:
        bundle.durable_event_sink(
            {
                "type": "tool_call",
                "run_id": attempt_id,
                "iteration": 0,
                "tool_name": "lookup",
                "call_id": call_id,
                "arguments": arguments,
                "source_provider": "openai",
            }
        )
    context = HarnessContext(
        state=bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": attempt_id,
            "execution_guard": guard,
            "toolkit": toolkit,
            "tool_call": ToolCall(
                call_id=call_id,
                name="lookup",
                arguments=dict(arguments),
            ),
            # None of these host-provided lookalikes are execution authority.
            "request": object(),
            "effective_arguments": {"query": "host-forged"},
            "terminal_handler": lambda value: forged_handler_calls.append(value),
        },
    )
    return (
        runtime,
        bundle,
        guard,
        context,
        toolkit,
        invocations,
        forged_handler_calls,
        bundles,
    )


def _uncommitted_runtime() -> ContextRuntime:
    return ContextRuntime._for_test(
        owner_id="context-v2-uncommitted",
        request_factory=lambda context: None,
        durable_event_sink=lambda event: None,
        partial_attempt_sink=lambda event, error: None,
        compiler=None,
    )


def test_prepare_and_execute_derive_tool_authority_from_journal_and_toolkit():
    (
        runtime,
        bundle,
        guard,
        context,
        _toolkit,
        invocations,
        forged_handler_calls,
        _bundles,
    ) = _factory_tool_runtime()
    try:
        permit = runtime.prepare_tool_execution(context)

        assert invocations == []
        receipt = runtime.execute_prepared_tool(context, permit)

        assert receipt.visible_result == {"seen": "journal-owned"}
        assert invocations == ["journal-owned"]
        assert forged_handler_calls == []
        assert [event.event_type for event in bundle.journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
        assert bundle.journal.events[0].payload["source_provider"] == "openai"
    finally:
        guard.release()


def test_tool_authority_harness_projects_only_the_durable_completion():
    (
        runtime,
        bundle,
        guard,
        context,
        _toolkit,
        invocations,
        _forged_handler_calls,
        _bundles,
    ) = _factory_tool_runtime(attempt_id="attempt-harness")
    context.state.provider_state.provider = "openai"
    emitted = []

    class Loop:
        @staticmethod
        def emit_event(callback, event_type, run_id, *, iteration, **extra):
            callback(
                {
                    "type": event_type,
                    "run_id": run_id,
                    "iteration": iteration,
                    **extra,
                }
            )

    context.event.update({"loop": Loop(), "callback": emitted.append})
    try:
        harness = runtime.build_harnesses()[1]
        delta = harness.build_delta(context)

        assert delta is not None
        batch = delta.state_updates["tool_batch_state"]
        assert invocations == ["journal-owned"]
        assert batch.executed_call_ids == ["call-1"]
        assert "journal-owned" in repr(batch.result_messages)
        assert delta.trace["durable_tool_completion"] is True
        assert [event.event_type for event in bundle.journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
        assert emitted == [
            {
                "type": "tool_call",
                "run_id": "attempt-harness",
                "iteration": 0,
                "tool_name": "lookup",
                "call_id": "call-1",
                "arguments": {"query": "journal-owned"},
                "source_provider": "openai",
            },
            {
                "type": "tool_result",
                "run_id": "attempt-harness",
                "iteration": 0,
                "tool_name": "lookup",
                "call_id": "call-1",
                "result": {"seen": "journal-owned"},
                "durable_result_outcome": "success",
            },
        ]
    finally:
        guard.release()


@pytest.mark.parametrize(
    ("tool_result", "expected_status"),
    [
        ({"ok": True, "value": "done"}, "success"),
        ({"ok": False, "error": "missing"}, "error"),
        ({"denied": True}, "denied"),
        ({"cancelled": True}, "cancelled"),
    ],
)
def test_projected_durable_tool_result_preserves_terminal_outcome_through_host_callback(
    tool_result,
    expected_status,
):
    (
        runtime,
        bundle,
        guard,
        context,
        _toolkit,
        _invocations,
        _forged_handler_calls,
        _bundles,
    ) = _factory_tool_runtime(
        attempt_id=f"attempt-outcome-{expected_status}",
        tool_result=tool_result,
        tool_output_management_active=True,
    )
    emitted = []
    context.state.provider_state.provider = "openai"
    context.event["callback"] = runtime.compose_event_callback(emitted.append)
    try:
        runtime.bind_execution_toolkit(context)
        delta = runtime.build_harnesses()[1].build_delta(context)

        assert delta is not None
        assert len(
            [event for event in bundle.journal.events if event.event_type == "tool_result"]
        ) == 1
        raw_result = next(event for event in emitted if event["type"] == "tool_result")
        assert raw_result["durable_result_outcome"] == expected_status
        assert raw_result["result"] != tool_result

        bridge = RuntimeEventBridge(
            session_id="session-outcome",
            root_run_id=context.event["run_id"],
        )
        [runtime_event] = bridge.normalize(raw_result)
        assert runtime_event.type == "step.completed"
        assert runtime_event.payload["status"] == expected_status
    finally:
        guard.release()


def test_prepare_fails_closed_without_durable_journal_intent():
    (
        runtime,
        bundle,
        guard,
        context,
        _toolkit,
        invocations,
        _forged_handler_calls,
        _bundles,
    ) = _factory_tool_runtime(persist_intent=False)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="journal|intent|durable",
        ):
            runtime.prepare_tool_execution(context)

        assert invocations == []
        assert bundle.journal.events == []
    finally:
        guard.release()


def test_prepared_tool_permit_is_one_shot():
    (
        runtime,
        bundle,
        guard,
        context,
        _toolkit,
        invocations,
        _forged_handler_calls,
        _bundles,
    ) = _factory_tool_runtime()
    try:
        permit = runtime.prepare_tool_execution(context)
        first = runtime.execute_prepared_tool(context, permit)

        with pytest.raises(
            DurableToolExecutorContractError,
            match="permit|one-shot|consumed",
        ):
            runtime.execute_prepared_tool(context, permit)

        assert first.visible_result == {"seen": "journal-owned"}
        assert invocations == ["journal-owned"]
        event_types = [event.event_type for event in bundle.journal.events]
        assert event_types.count("tool.started") == 1
        assert event_types.count("tool_result") == 1
    finally:
        guard.release()


def test_prepared_tool_permit_rejects_lookalike_and_wrong_runtime():
    lookalike_fixture = _factory_tool_runtime(attempt_id="attempt-lookalike")
    runtime, _bundle, guard, context, _toolkit, invocations, *_rest = (
        lookalike_fixture
    )

    class PermitLookalike:
        pass

    try:
        permit = runtime.prepare_tool_execution(context)
        lookalike = PermitLookalike()
        try:
            vars(lookalike).update(vars(permit))
        except TypeError:
            pass

        with pytest.raises(
            DurableToolExecutorContractError,
            match="permit",
        ):
            runtime.execute_prepared_tool(context, lookalike)
        assert invocations == []
    finally:
        guard.release()

    first = _factory_tool_runtime(attempt_id="attempt-runtime")
    second = _factory_tool_runtime(attempt_id="attempt-runtime")
    runtime, _bundle, guard, context, _toolkit, invocations, *_rest = first
    (
        other_runtime,
        _other_bundle,
        other_guard,
        other_context,
        _other_toolkit,
        other_invocations,
        *_other_rest,
    ) = second
    try:
        permit = runtime.prepare_tool_execution(context)

        with pytest.raises(
            DurableToolExecutorContractError,
            match="permit|runtime|authority|binding",
        ):
            other_runtime.execute_prepared_tool(other_context, permit)
        assert invocations == []
        assert other_invocations == []
    finally:
        guard.release()
        other_guard.release()


def test_prepared_tool_permit_rejects_a_different_context_and_attempt():
    context_fixture = _factory_tool_runtime(attempt_id="attempt-context")
    runtime, _bundle, guard, context, _toolkit, invocations, *_rest = (
        context_fixture
    )
    context_clone = HarnessContext(
        state=context.state,
        phase=context.phase,
        event=dict(context.event),
    )
    try:
        permit = runtime.prepare_tool_execution(context)
        with pytest.raises(
            DurableToolExecutorContractError,
            match="permit|context|binding",
        ):
            runtime.execute_prepared_tool(context_clone, permit)
        assert invocations == []
    finally:
        guard.release()

    attempt_fixture = _factory_tool_runtime(attempt_id="attempt-original")
    runtime, _bundle, guard, context, toolkit, invocations, *_rest = (
        attempt_fixture
    )
    second_guard = ExecutionRuntime(InMemorySessionStore()).acquire(
        "execution-1",
        "owner-other-attempt",
    )
    second_bootstrap = _context(
        session_id="execution-1",
        run_id="attempt-other",
        current_input=None,
    )
    second_bootstrap.event["execution_guard"] = second_guard
    runtime.build_harnesses()[0].build_delta(second_bootstrap)
    second_context = HarnessContext(
        state=second_bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": "attempt-other",
            "execution_guard": second_guard,
            "toolkit": toolkit,
            "tool_call": ToolCall(
                call_id="call-1",
                name="lookup",
                arguments={"query": "journal-owned"},
            ),
        },
    )
    try:
        permit = runtime.prepare_tool_execution(context)
        with pytest.raises(
            DurableToolExecutorContractError,
            match="permit|attempt|context|binding",
        ):
            runtime.execute_prepared_tool(second_context, permit)
        assert invocations == []
    finally:
        guard.release()
        second_guard.release()


def test_uncommitted_runtime_cannot_prepare_tools():
    runtime = _uncommitted_runtime()
    context = _context(
        session_id="execution-uncommitted",
        run_id="attempt-uncommitted",
        current_input=None,
    )

    with pytest.raises(
        DurableToolExecutorContractError,
        match="factory|committed|durable",
    ):
        runtime.prepare_tool_execution(context)
    with pytest.raises(
        DurableToolExecutorContractError,
        match="factory|committed|durable|permit",
    ):
        runtime.execute_prepared_tool(context, object())


def test_public_contract_rejects_host_supplied_execution_material():
    (
        runtime,
        _bundle,
        guard,
        context,
        _toolkit,
        invocations,
        forged_handler_calls,
        _bundles,
    ) = _factory_tool_runtime()
    try:
        assert not hasattr(runtime, "execute_tool")
        with pytest.raises(TypeError):
            runtime.prepare_tool_execution(context, request=object())

        permit = runtime.prepare_tool_execution(context)
        with pytest.raises(TypeError):
            runtime.execute_prepared_tool(
                context,
                permit,
                effective_arguments={"query": "host-forged"},
                terminal_handler=lambda value: forged_handler_calls.append(value),
            )

        receipt = runtime.execute_prepared_tool(context, permit)
        assert receipt.visible_result == {"seen": "journal-owned"}
        assert invocations == ["journal-owned"]
        assert forged_handler_calls == []
        assert "journal-owned" not in repr(permit)
        with pytest.raises((TypeError, pickle.PicklingError, AttributeError)):
            pickle.dumps(permit)
    finally:
        guard.release()
