from __future__ import annotations

import copy
import json

import pytest

from tests.context_v2.test_durable_tool_executor import (
    _execution_fixture,
    _guard,
    _invocation,
)
from tests.context_v2.test_context_runtime_factory import (
    _bundle,
    _context,
    _current_input,
)
from unchain.context import (
    ContextRuntime,
    DurableContextRuntimeFactory,
    DurableToolCompletionDraft,
    DurableToolExecutor,
    DurableToolExecutorContractError,
)
from unchain.context.tool_harness import ContextToolAuthorityHarness
from unchain.context.tool_transitions import (
    DurableToolStateTransitionEnvelope,
    build_declared_tool_state_transition,
    resolve_subagent_transition_cas,
)
from unchain.context.tool_routes import ContextToolRouteResolver
from unchain.kernel.state import RunState
from unchain.kernel.harness import HarnessContext
from unchain.kernel.types import ToolCall
from unchain.subagents.types import SubagentState
from unchain.tools.types import ToolBatchState
from unchain.tools import Toolkit
from unchain.tools.runtime import ToolRuntimeOutcome


TERMINAL_HANDOFF_KEYS = [
    "subagent_state",
    "transcript",
    "run_status",
    "pending_tool_calls",
    "tool_batch_state",
    "last_continuation",
    "next_model_input",
]


def _terminal_contract(*, allowed_state_keys=None):
    return {
        "schema": "unchain.tool_completion_contract.v1",
        "state_transition": "subagent_terminal_handoff.v1",
        "allowed_state_keys": list(
            TERMINAL_HANDOFF_KEYS if allowed_state_keys is None else allowed_state_keys
        ),
    }


def _handoff_variant_contract(*, variants=None):
    return {
        "schema": "unchain.tool_completion_contract.v1",
        "state_transition_variants": list(
            variants
            if variants is not None
            else [
                {
                    "state_transition": "subagent_snapshot.v1",
                    "allowed_state_keys": ["subagent_state"],
                },
                {
                    "state_transition": "subagent_terminal_handoff.v1",
                    "allowed_state_keys": list(TERMINAL_HANDOFF_KEYS),
                },
            ]
        ),
    }


def _base_subagent_state() -> SubagentState:
    return SubagentState(
        root_agent_id="root",
        active_agent_id="root",
        active_lineage=["root"],
    )


def _terminal_subagent_state() -> SubagentState:
    return SubagentState(
        root_agent_id="root",
        active_agent_id="root.researcher.1",
        active_lineage=["root", "root.researcher.1"],
        handoff_stack=[
            {
                "from_agent_id": "root",
                "to_agent_id": "root.researcher.1",
                "lineage": ["root", "root.researcher.1"],
                "template": "researcher",
            }
        ],
    )


def _base_run_state() -> RunState:
    return RunState(
        transcript=[{"role": "user", "content": "old"}],
        pending_tool_calls=[
            ToolCall(
                call_id="call-1",
                name="handoff_to_subagent",
                arguments={"target": "researcher"},
            )
        ],
        tool_batch_state=ToolBatchState(
            result_messages=[{"role": "tool", "content": "old"}],
            should_observe=True,
            executed_call_ids=["old-call"],
        ),
        run_status="running",
        last_continuation={"type": "old"},
        next_model_input=[{"role": "user", "content": "old"}],
        subagent_state=_base_subagent_state(),
    )


def _terminal_base_state():
    state = _base_run_state()
    return {
        "subagent_state": state.subagent_state,
        "transcript": state.transcript,
        "run_status": state.run_status,
        "pending_tool_calls": state.pending_tool_calls,
        "tool_batch_state": state.tool_batch_state,
        "last_continuation": state.last_continuation,
        "next_model_input": state.next_model_input,
    }


def _canonical_terminal_base_state():
    return {
        "subagent_state": _base_subagent_state().to_dict(),
        "transcript": [{"role": "user", "content": "old"}],
        "run_status": "running",
        "pending_tool_calls": [
            {
                "call_id": "call-1",
                "name": "handoff_to_subagent",
                "arguments": {"target": "researcher"},
            }
        ],
        "tool_batch_state": {
            "result_messages": [{"role": "tool", "content": "old"}],
            "should_observe": True,
            "awaiting_human_input": False,
            "human_input_request": None,
            "human_input_tool_call_id": None,
            "executed_call_ids": ["old-call"],
        },
        "last_continuation": {"type": "old"},
        "next_model_input": [{"role": "user", "content": "old"}],
    }


def _terminal_updates():
    return {
        "subagent_state": _terminal_subagent_state(),
        "transcript": [
            {"role": "user", "content": "investigate"},
            {"role": "assistant", "content": "finished"},
        ],
        "run_status": "completed",
        "pending_tool_calls": [],
        "tool_batch_state": {},
        "last_continuation": None,
        "next_model_input": None,
    }


def _canonical_terminal_updates():
    return {
        **_terminal_updates(),
        "subagent_state": _terminal_subagent_state().to_dict(),
        "tool_batch_state": {
            "result_messages": [],
            "should_observe": False,
            "awaiting_human_input": False,
            "human_input_request": None,
            "human_input_tool_call_id": None,
            "executed_call_ids": [],
        },
    }


def _child_wait_updates():
    marker = json.dumps({
        "schema": "unchain.skeleton_subagent_wait_marker.v1",
        "wait_id": "subagent.wait-1",
        "request": {"schema": "pupu.skeleton_subagent_wait.v1"},
    }, sort_keys=True, separators=(",", ":"))
    return {
        **_terminal_updates(),
        "transcript": [
            {"role": "user", "content": "investigate"},
            {"role": "assistant", "content": marker},
        ],
        "run_status": "awaiting_human_input",
    }


def test_terminal_handoff_manifest_builds_one_exact_typed_transition() -> None:
    draft = build_declared_tool_state_transition(
        completion_contract=_terminal_contract(),
        state_updates=_terminal_updates(),
        base_state=_terminal_base_state(),
    )

    assert draft is not None
    assert draft.kind == "subagent_terminal_handoff"
    assert draft.base_state == _canonical_terminal_base_state()
    assert draft.next_state == _canonical_terminal_updates()
    assert draft.handoff_refs == ()


def test_terminal_handoff_manifest_accepts_only_the_internal_child_wait_marker() -> None:
    draft = build_declared_tool_state_transition(
        completion_contract=_terminal_contract(),
        state_updates=_child_wait_updates(),
        base_state=_terminal_base_state(),
    )

    assert draft is not None
    assert draft.next_state["run_status"] == "awaiting_human_input"
    assert json.loads(draft.next_state["transcript"][-1]["content"])["wait_id"] == "subagent.wait-1"

    forged = _child_wait_updates()
    forged["transcript"][-1]["content"] = '{"schema":"unknown"}'
    with pytest.raises(DurableToolExecutorContractError, match="run status"):
        build_declared_tool_state_transition(
            completion_contract=_terminal_contract(),
            state_updates=forged,
            base_state=_terminal_base_state(),
        )


@pytest.mark.parametrize(
    ("state_updates", "expected_kind"),
    [
        ({"subagent_state": _terminal_subagent_state()}, "subagent_snapshot"),
        (_terminal_updates(), "subagent_terminal_handoff"),
    ],
)
def test_handoff_variant_contract_selects_one_exact_transition_by_update_key_set(
    state_updates,
    expected_kind,
) -> None:
    draft = build_declared_tool_state_transition(
        completion_contract=_handoff_variant_contract(),
        state_updates=state_updates,
        base_state=_terminal_base_state(),
    )

    assert draft is not None
    assert draft.kind == expected_kind
    if expected_kind == "subagent_snapshot":
        assert draft.base_state == _base_subagent_state().to_dict()
        assert draft.next_state == _terminal_subagent_state().to_dict()
    else:
        assert draft.base_state == _canonical_terminal_base_state()
        assert draft.next_state == _canonical_terminal_updates()


@pytest.mark.parametrize(
    "state_updates",
    [
        {
            "subagent_state": _terminal_subagent_state(),
            "transcript": [],
        },
        {**_terminal_updates(), "metadata": {"forged": True}},
    ],
)
def test_handoff_variant_contract_rejects_partial_or_extra_update_key_sets(
    state_updates,
) -> None:
    with pytest.raises(
        DurableToolExecutorContractError,
        match="exact|variant|transition|manifest",
    ):
        build_declared_tool_state_transition(
            completion_contract=_handoff_variant_contract(),
            state_updates=state_updates,
            base_state=_terminal_base_state(),
        )


@pytest.mark.parametrize(
    "variants",
    [
        [
            {
                "state_transition": "subagent_snapshot.v1",
                "allowed_state_keys": ["subagent_state"],
            }
        ],
        [
            {
                "state_transition": "subagent_snapshot.v1",
                "allowed_state_keys": ["subagent_state"],
            },
            {
                "state_transition": "subagent_snapshot.v1",
                "allowed_state_keys": ["subagent_state"],
            },
        ],
        [
            {
                "state_transition": "subagent_snapshot.v1",
                "allowed_state_keys": ["subagent_state"],
            },
            {
                "state_transition": "subagent_terminal_handoff.v1",
                "allowed_state_keys": [*TERMINAL_HANDOFF_KEYS, "metadata"],
            },
        ],
        [
            {
                "state_transition": "subagent_snapshot.v1",
                "allowed_state_keys": ["subagent_state"],
            },
            {
                "state_transition": "unknown.v1",
                "allowed_state_keys": list(TERMINAL_HANDOFF_KEYS),
            },
        ],
    ],
)
def test_handoff_variant_contract_rejects_missing_duplicate_or_forged_variants(
    variants,
) -> None:
    with pytest.raises(
        DurableToolExecutorContractError,
        match="variant|allowed keys|unsupported|manifest",
    ):
        build_declared_tool_state_transition(
            completion_contract=_handoff_variant_contract(variants=variants),
            state_updates={"subagent_state": _terminal_subagent_state()},
            base_state=_terminal_base_state(),
        )


@pytest.mark.parametrize(
    "allowed_state_keys",
    [
        TERMINAL_HANDOFF_KEYS[:-1],
        [*TERMINAL_HANDOFF_KEYS, "metadata"],
        [*TERMINAL_HANDOFF_KEYS[:-1], "subagent_state"],
    ],
)
def test_terminal_handoff_manifest_rejects_partial_extra_or_duplicate_keys(
    allowed_state_keys,
) -> None:
    with pytest.raises(
        DurableToolExecutorContractError,
        match="allowed keys|manifest|transition",
    ):
        build_declared_tool_state_transition(
            completion_contract=_terminal_contract(
                allowed_state_keys=allowed_state_keys,
            ),
            state_updates=_terminal_updates(),
            base_state=_terminal_base_state(),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda updates: updates.pop("next_model_input"),
        lambda updates: updates.update({"metadata": {"forged": True}}),
        lambda updates: updates.update({"run_status": "running"}),
        lambda updates: updates.update(
            {"pending_tool_calls": [{"call_id": "still-open"}]}
        ),
        lambda updates: updates.update({"last_continuation": {"type": "still-open"}}),
        lambda updates: updates.update(
            {"next_model_input": [{"role": "user", "content": "again"}]}
        ),
    ],
)
def test_terminal_handoff_transition_rejects_partial_extra_or_nonterminal_state(
    mutate,
) -> None:
    updates = copy.deepcopy(_terminal_updates())
    mutate(updates)

    with pytest.raises(
        DurableToolExecutorContractError,
        match="terminal|transition|state",
    ):
        build_declared_tool_state_transition(
            completion_contract=_terminal_contract(),
            state_updates=updates,
            base_state=_terminal_base_state(),
        )


def test_terminal_handoff_is_sealed_cold_recovered_and_cas_materialized() -> None:
    (
        executor,
        request,
        guard,
        journal,
        _order,
        _intent,
        _arguments,
    ) = _execution_fixture()
    draft = build_declared_tool_state_transition(
        completion_contract=_terminal_contract(),
        state_updates=_terminal_updates(),
        base_state=_terminal_base_state(),
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"status": "completed"},
                    state_transition=draft,
                ),
            ),
        )

        assert receipt.transition is not None
        assert receipt.transition.kind == "subagent_terminal_handoff"
        assert receipt.transition.handoff_refs == ()
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
            "tool_result",
        ]
        stored = executor.artifacts.read_full(
            receipt.transition.next_state_artifact,
            remaining_budget_bytes=(receipt.transition.next_state_artifact.byte_length),
        )
        assert json.loads(stored.decode("utf-8")) == (_canonical_terminal_updates())

        cold_executor = DurableToolExecutor(
            boundary=executor.boundary,
            artifacts=executor.artifacts,
            execution_guard=guard,
        )
        recovered = cold_executor.execute(
            request=request,
            guard=guard,
            invocation=None,
        )
        transition = DurableToolStateTransitionEnvelope.from_dict(
            recovered.transition.to_dict()
        )
        assert recovered.reused is True
        assert transition.kind == "subagent_terminal_handoff"

        state = _base_run_state()
        updates = resolve_subagent_transition_cas(
            artifacts=executor.artifacts,
            transition=transition,
            current_state=state,
        )

        assert updates is not None
        assert set(updates) == set(TERMINAL_HANDOFF_KEYS)
        assert type(updates["subagent_state"]) is SubagentState
        assert type(updates["tool_batch_state"]) is ToolBatchState
        state._apply_state_updates(updates)
        assert state.subagent_state == _terminal_subagent_state()
        assert state.transcript == _terminal_updates()["transcript"]
        assert state.run_status == "completed"
        assert state.pending_tool_calls == []
        assert state.tool_batch_state == ToolBatchState()
        assert state.last_continuation is None
        assert state.next_model_input is None
        assert (
            resolve_subagent_transition_cas(
                artifacts=executor.artifacts,
                transition=transition,
                current_state=state,
            )
            is None
        )

        conflict = RunState(
            subagent_state=SubagentState(
                root_agent_id="root",
                active_agent_id="root.other.1",
                active_lineage=["root", "root.other.1"],
            )
        )
        with pytest.raises(
            DurableToolExecutorContractError,
            match="state transition CAS conflict",
        ):
            resolve_subagent_transition_cas(
                artifacts=executor.artifacts,
                transition=transition,
                current_state=conflict,
            )
    finally:
        guard.release()


def test_terminal_route_binds_the_complete_pre_execution_state() -> None:
    class TerminalHandoffPlugin:
        def can_handle(self, *, tool_call, context):
            del tool_call, context
            return True

        def durable_runtime_manifest(self, *, tool_call, context):
            del tool_call, context
            return {
                "handler": "terminal-handoff",
                "terminal_handler": True,
                "completion_contract": _terminal_contract(),
            }

        def execute(self, *, tool_call, context):
            del tool_call, context
            return ToolRuntimeOutcome(
                handled=True,
                tool_result={"status": "completed"},
                state_updates=_terminal_updates(),
            )

    tool_call = ToolCall(
        call_id="call-1",
        name="handoff_to_subagent",
        arguments={"target": "researcher"},
    )
    toolkit = Toolkit()
    toolkit.register(lambda: {"legacy": True}, name=tool_call.name)
    context = _context(
        session_id="execution-1",
        run_id="attempt-1",
    )
    base = _base_run_state()
    context.state.transcript = copy.deepcopy(base.transcript)
    context.state.run_status = base.run_status
    context.state.pending_tool_calls = list(base.pending_tool_calls)
    context.state.tool_batch_state = base.tool_batch_state.copy()
    context.state.last_continuation = copy.deepcopy(base.last_continuation)
    context.state.next_model_input = copy.deepcopy(base.next_model_input)
    context.state.subagent_state = base.subagent_state.copy()
    context.event.update(
        {
            "toolkit": toolkit,
            "tool_runtime_plugins": [TerminalHandoffPlugin()],
            "tool_call": tool_call,
        }
    )

    route = ContextToolRouteResolver().resolve(context, tool_call)
    draft = route.terminal_handler(tool_call.arguments)

    assert draft.state_transition is not None
    assert draft.state_transition.kind == "subagent_terminal_handoff"
    assert draft.state_transition.base_state == _canonical_terminal_base_state()


def test_context_runtime_and_harness_apply_all_terminal_fields_atomically() -> None:
    calls = 0

    class TerminalHandoffPlugin:
        def can_handle(self, *, tool_call, context):
            del tool_call, context
            return True

        def durable_runtime_manifest(self, *, tool_call, context):
            del tool_call, context
            return {
                "handler": "terminal-handoff",
                "terminal_handler": True,
                "completion_contract": _terminal_contract(),
            }

        def execute(self, *, tool_call, context):
            del tool_call, context
            nonlocal calls
            calls += 1
            return ToolRuntimeOutcome(
                handled=True,
                tool_result={"status": "completed"},
                state_updates=_terminal_updates(),
            )

    bundles = {}

    def build(attempt):
        bundle = _bundle(attempt)
        bundles[(attempt.generation.execution_id, attempt.attempt_id)] = bundle
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id="context-v2-terminal-handoff",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, execution_id: "generation-1",
            current_input_resolver=_current_input,
        ),
    )
    guard = _guard()
    bootstrap = _context(
        session_id="execution-1",
        run_id="attempt-1",
        current_input=None,
    )
    base = _base_run_state()
    bootstrap.state.transcript = copy.deepcopy(base.transcript)
    bootstrap.state.run_status = base.run_status
    bootstrap.state.pending_tool_calls = list(base.pending_tool_calls)
    bootstrap.state.tool_batch_state = base.tool_batch_state.copy()
    bootstrap.state.last_continuation = copy.deepcopy(base.last_continuation)
    bootstrap.state.next_model_input = copy.deepcopy(base.next_model_input)
    bootstrap.state.subagent_state = base.subagent_state.copy()
    bootstrap.state.provider_state.provider = "openai"
    bootstrap.event["execution_guard"] = guard
    runtime.build_harnesses()[0].build_delta(bootstrap)
    bundle = bundles[("execution-1", "attempt-1")]
    tool_call = base.pending_tool_calls[0]
    bundle.durable_event_sink(
        {
            "type": "tool_call",
            "run_id": "attempt-1",
            "iteration": 0,
            "tool_name": tool_call.name,
            "call_id": tool_call.call_id,
            "arguments": tool_call.arguments,
            "source_provider": "openai",
        }
    )
    toolkit = Toolkit()
    toolkit.register(lambda: {"legacy": True}, name=tool_call.name)
    context = HarnessContext(
        state=bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": "attempt-1",
            "execution_guard": guard,
            "toolkit": toolkit,
            "tool_runtime_plugins": [TerminalHandoffPlugin()],
            "tool_call": tool_call,
        },
    )
    harness = ContextToolAuthorityHarness(runtime=runtime)
    try:
        delta = harness.build_delta(context)

        assert delta is not None
        assert set(delta.state_updates) == set(TERMINAL_HANDOFF_KEYS)
        context.state.apply_delta(delta)
        assert context.state.subagent_state == _terminal_subagent_state()
        assert context.state.transcript == _terminal_updates()["transcript"]
        assert context.state.run_status == "completed"
        assert context.state.pending_tool_calls == []
        assert context.state.tool_batch_state == ToolBatchState()
        assert context.state.last_continuation is None
        assert context.state.next_model_input is None
        assert calls == 1

        replay = harness.build_delta(context)
        assert replay is not None
        assert replay.state_updates == {}
        assert calls == 1
        assert [event.event_type for event in bundle.journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
            "tool_result",
        ]
    finally:
        guard.release()


def test_terminal_handoff_cas_rejects_drift_in_each_bound_state_field() -> None:
    (
        executor,
        request,
        guard,
        _journal,
        _order,
        _intent,
        _arguments,
    ) = _execution_fixture()
    draft = build_declared_tool_state_transition(
        completion_contract=_terminal_contract(),
        state_updates=_terminal_updates(),
        base_state=_terminal_base_state(),
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"status": "completed"},
                    state_transition=draft,
                ),
            ),
        )
        transition = receipt.transition
        assert transition is not None

        def drift_subagent(state):
            state.subagent_state.active_agent_id = "root.other.1"

        def drift_transcript(state):
            state.transcript.append({"role": "assistant", "content": "concurrent"})

        def drift_run_status(state):
            state.run_status = "failed"

        def drift_pending_calls(state):
            state.pending_tool_calls = []

        def drift_tool_batch(state):
            state.tool_batch_state = ToolBatchState()

        def drift_continuation(state):
            state.last_continuation = {"type": "changed"}

        def drift_next_input(state):
            state.next_model_input = [{"role": "user", "content": "changed"}]

        for mutate in (
            drift_subagent,
            drift_transcript,
            drift_run_status,
            drift_pending_calls,
            drift_tool_batch,
            drift_continuation,
            drift_next_input,
        ):
            state = _base_run_state()
            mutate(state)
            with pytest.raises(
                DurableToolExecutorContractError,
                match="state transition CAS conflict",
            ):
                resolve_subagent_transition_cas(
                    artifacts=executor.artifacts,
                    transition=transition,
                    current_state=state,
                )
    finally:
        guard.release()
