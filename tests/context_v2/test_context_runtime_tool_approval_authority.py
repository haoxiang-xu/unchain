from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import unchain.context as context_api
from unchain.capabilities import CapabilityOutcome, RequestSuspendOp
from tests.context_v2.test_context_runtime_factory import (
    _bundle,
    _context,
    _current_input,
)
from unchain.context import (
    ContextExecutionBundleError,
    ContextRuntime,
    DurableContextRuntimeFactory,
    DurableToolApprovalState,
    DurableToolExecutorContractError,
)
from unchain.execution import ExecutionRuntime
from unchain.interaction import (
    INTERACTION_JOURNAL_KEY,
    INTERACTION_KIND_TOOL_APPROVAL,
    InteractionIntegrityError,
    InteractionNotPendingError,
    InteractionRequest,
    build_interaction_receipt,
)
from unchain.interaction.durable import (
    strict_json_digest,
    validate_interaction_journal,
)
from unchain.interaction.runtime import DurableInteractionRuntime
from unchain.kernel.harness import HarnessContext
from unchain.kernel.loop import KernelLoop
from unchain.kernel.types import ToolCall
from unchain.memory import (
    InMemorySessionStore,
    KernelMemoryRuntime,
)
from unchain.memory.checkpoint_state import build_execution_checkpoint
from unchain.tools import Toolkit


def _approval_tool_runtime(
    *,
    execution_id: str = "execution-approval",
    attempt_id: str = "attempt-approval",
    interaction_store=None,
    bind_interaction_runtime: bool = True,
    timeline_merge_policy: str | None = None,
):
    bundles = {}

    def build(attempt):
        bundle = _bundle(attempt)
        bundles[(attempt.generation.execution_id, attempt.attempt_id)] = bundle
        return bundle

    store = InMemorySessionStore()
    memory_runtime = KernelMemoryRuntime.from_config(store=store)
    interaction_memory_runtime = (
        memory_runtime
        if interaction_store is None
        else KernelMemoryRuntime.from_config(store=interaction_store)
    )
    interaction_runtime = DurableInteractionRuntime(
        interaction_memory_runtime,
        clock_ms=lambda: 123,
    )
    bootstrap_interaction_runtime = (
        interaction_runtime if bind_interaction_runtime else None
    )
    loop = KernelLoop(
        interaction_runtime=bootstrap_interaction_runtime,
    )
    runtime = ContextRuntime.from_factory(
        owner_id=f"context-v2-{attempt_id}",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, resolved_execution_id: (
                f"generation-{resolved_execution_id}"
            ),
            current_input_resolver=_current_input,
        ),
    )
    guard = ExecutionRuntime(store).acquire(
        execution_id,
        f"owner-{attempt_id}",
    )
    bootstrap = _context(
        session_id=execution_id,
        run_id=attempt_id,
        current_input=None,
    )
    bootstrap.state.provider_state.provider = "openai"
    bootstrap.state.provider_state.model = "gpt-5"
    bootstrap.state.memory_state["session_revision"] = (
        memory_runtime.load_session_snapshot(execution_id).revision
    )
    bootstrap.event["execution_guard"] = guard
    bootstrap.event["loop"] = loop
    try:
        runtime.build_harnesses()[0].build_delta(bootstrap)
    except Exception:
        guard.release()
        raise
    bundle = bundles[(execution_id, attempt_id)]

    invocations: list[str] = []
    policy = {
        "requires_confirmation": True,
        "description": "Approve lookup",
    }

    def confirmation_resolver(arguments, execution_context):
        del arguments, execution_context
        return copy.deepcopy(policy)

    toolkit = Toolkit()

    @toolkit.tool(
        name="lookup",
        description="Lookup a value",
        requires_confirmation=True,
        confirmation_resolver=confirmation_resolver,
    )
    def lookup(query: str):
        invocations.append(query)
        return {"seen": query}

    arguments = {"query": "journal-owned"}
    intent_event = {
            "type": "tool_call",
            "run_id": attempt_id,
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-approval",
            "arguments": arguments,
        }
    if timeline_merge_policy is not None:
        intent_event["timeline_merge_policy"] = timeline_merge_policy
        intent_event["source_provider"] = "openai"
    bundle.durable_event_sink(intent_event)
    context = HarnessContext(
        state=bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": attempt_id,
            "execution_guard": guard,
            "toolkit": toolkit,
            "tool_call": ToolCall(
                call_id="call-approval",
                name="lookup",
                arguments=dict(arguments),
            ),
            "tool_calls": [
                ToolCall(
                    call_id="call-approval",
                    name="lookup",
                    arguments=dict(arguments),
                )
            ],
            "payload": {"source": "approval-test"},
            "max_iterations": 6,
            "loop": loop,
        },
    )
    return {
        "runtime": runtime,
        "bundle": bundle,
        "guard": guard,
        "context": context,
        "toolkit": toolkit,
        "invocations": invocations,
        "policy": policy,
        "memory_runtime": memory_runtime,
        "interaction_runtime": interaction_runtime,
        "loop": loop,
        "bootstrap": bootstrap,
        "execution_id": execution_id,
        "attempt_id": attempt_id,
    }


def _assert_durable_approval_pending(value):
    pending_type = getattr(
        context_api,
        "DurableToolApprovalPending",
        None,
    )
    assert pending_type is not None, (
        "Context V2 must expose a typed DurableToolApprovalPending result"
    )
    assert type(value) is pending_type
    request = InteractionRequest.from_dict(value.interaction_request)
    assert isinstance(value.continuation, dict)
    return value, request


def _persist_approval_pending(fixture, pending, response):
    pending, request = _assert_durable_approval_pending(pending)
    state = fixture["context"].state
    state.run_status = "awaiting_interaction"
    state.last_continuation = copy.deepcopy(pending.continuation)
    state.suspend_state.signal_kind = "tool_approval"
    state.suspend_state.payload = {
        "interaction_request": request.to_dict(),
        "continuation": copy.deepcopy(pending.continuation),
    }
    checkpoint = build_execution_checkpoint(
        state,
        status="awaiting_interaction",
        run_id=fixture["attempt_id"],
    )
    _, persisted = fixture["memory_runtime"].save_execution_checkpoint_snapshot(
        fixture["execution_id"],
        checkpoint,
        interaction_request=request.to_dict(),
        expected_revision=request.created_revision,
        execution_fence=fixture["guard"].fence,
    )
    receipt_snapshot = fixture["interaction_runtime"].record_receipt(
        fixture["execution_id"],
        interaction_id=request.interaction_id,
        response=response,
        expected_revision=persisted.revision,
        execution_fence=fixture["guard"].fence,
    )
    assert receipt_snapshot.receipt is not None
    return request, receipt_snapshot.receipt


_AUTO_RESUME_FIELD = object()


def _resume_approval_context(
    fixture,
    *,
    toolkit=None,
    raw_request=_AUTO_RESUME_FIELD,
    raw_response=_AUTO_RESUME_FIELD,
):
    original = fixture["context"]
    if raw_request is _AUTO_RESUME_FIELD or raw_response is _AUTO_RESUME_FIELD:
        durable_snapshot = fixture["interaction_runtime"].require_receipt(
            fixture["execution_id"]
        )
        if raw_request is _AUTO_RESUME_FIELD:
            raw_request = durable_snapshot.request.to_dict()
        if raw_response is _AUTO_RESUME_FIELD:
            assert durable_snapshot.receipt is not None
            raw_response = copy.deepcopy(durable_snapshot.receipt.response)
    event = {
        **original.event,
        "toolkit": toolkit or fixture["toolkit"],
        "tool_call": ToolCall(
            call_id="call-approval",
            name="lookup",
            arguments={"query": "journal-owned"},
        ),
    }
    if raw_request is not None:
        event["interaction_request"] = copy.deepcopy(raw_request)
    if raw_response is not None:
        event["interaction_response"] = copy.deepcopy(raw_response)
    return HarnessContext(
        state=original.state,
        phase="on_tool_call",
        event=event,
    )


def test_prepare_confirmation_returns_runtime_owned_approval_pending():
    fixture = _approval_tool_runtime()
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        pending, request = _assert_durable_approval_pending(pending)

        assert fixture["invocations"] == []
        assert request.kind == INTERACTION_KIND_TOOL_APPROVAL
        assert request.session_id == fixture["execution_id"]
        assert request.source_run_id == fixture["attempt_id"]
        assert request.payload["tool_name"] == "lookup"
        assert request.payload["call_id"] == "call-approval"
        assert request.payload["arguments"] == {
            "query": "journal-owned"
        }
        assert pending.continuation["type"] == (
            "tool_approval_continuation"
        )
        assert pending.continuation["interaction_id"] == (
            request.interaction_id
        )
        assert pending.continuation["paused_call_id"] == "call-approval"
        assert [event.event_type for event in fixture["bundle"].journal.events] == [
            "tool_call"
        ]
        with pytest.raises(DurableToolExecutorContractError, match="permit"):
            fixture["runtime"].execute_prepared_tool(
                fixture["context"],
                pending,
            )
    finally:
        fixture["guard"].release()


def test_tool_authority_harness_applies_runtime_owned_approval_suspension():
    fixture = _approval_tool_runtime(attempt_id="attempt-harness-pending")
    try:
        harness = fixture["runtime"].build_harnesses()[1]
        outcome = harness.build_delta(fixture["context"])

        assert isinstance(outcome, CapabilityOutcome)
        assert outcome.delta is not None
        assert outcome.delta.state_updates["run_status"] == (
            "awaiting_interaction"
        )
        assert outcome.delta.state_updates["last_continuation"]["type"] == (
            "tool_approval_continuation"
        )
        assert len(outcome.delta.context_ops) == 1
        assert isinstance(outcome.delta.context_ops[0], RequestSuspendOp)
        assert fixture["invocations"] == []
        assert [event.event_type for event in fixture["bundle"].journal.events] == [
            "tool_call"
        ]
    finally:
        fixture["guard"].release()


def test_approval_resume_reuses_original_timeline_policy_after_tool_config_changes():
    fixture = _approval_tool_runtime(
        attempt_id="attempt-policy-resume",
        timeline_merge_policy="never",
    )
    tool = fixture["toolkit"].get("lookup")
    try:
        harness = fixture["runtime"].build_harnesses()[1]
        approval = fixture["runtime"].prepare_tool_execution(fixture["context"])
        approval, _request = _assert_durable_approval_pending(approval)
        _persist_approval_pending(
            fixture,
            approval,
            {"approved": True, "modified_arguments": None, "reason": ""},
        )

        # A developer may reconfigure the live toolkit between pause and resume.
        # The same durable call ID must retain its original journal intent bytes.
        tool.timeline_merge_policy = "always"
        resumed = _resume_approval_context(fixture)
        resumed.state.run_status = "running"
        resumed.event["callback"] = fixture["runtime"].compose_event_callback(None)
        outcome = harness.build_delta(resumed)
        assert outcome is not None
        intent = next(
            event
            for event in fixture["bundle"].journal.events
            if event.event_type == "tool_call"
        )
        assert intent.payload["timeline_merge_policy"] == "never"
    finally:
        fixture["guard"].release()


def test_prepare_confirmation_requires_durable_receipt_not_raw_event():
    fixture = _approval_tool_runtime(attempt_id="attempt-raw-forgery")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        _, request = _assert_durable_approval_pending(pending)
        forged_resume = _resume_approval_context(
            fixture,
            raw_request=request.to_dict(),
            raw_response={
                "approved": True,
                "modified_arguments": {"query": "raw-forged"},
                "reason": "",
            },
        )

        with pytest.raises(
            (
                InteractionIntegrityError,
                InteractionNotPendingError,
                DurableToolExecutorContractError,
            ),
            match="durable|receipt|interaction|pending",
        ):
            fixture["runtime"].prepare_tool_execution(forged_resume)

        assert fixture["invocations"] == []
        assert [event.event_type for event in fixture["bundle"].journal.events] == [
            "tool_call"
        ]
    finally:
        fixture["guard"].release()


def test_forged_event_interaction_runtime_cannot_supply_approval_receipt():
    fixture = _approval_tool_runtime(attempt_id="attempt-forged-runtime")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        _, request = _assert_durable_approval_pending(pending)
        forged_receipt = build_interaction_receipt(
            request,
            {
                "approved": True,
                "modified_arguments": {"query": "forged"},
                "reason": "",
            },
            submitted_at_ms=124,
        )
        forged_runtime = SimpleNamespace(
            require_receipt=lambda session_id: SimpleNamespace(
                request=request,
                receipt=forged_receipt,
            )
        )
        forged_resume = _resume_approval_context(
            fixture,
            raw_request=request.to_dict(),
            raw_response=forged_receipt.response,
        )
        forged_resume.event["loop"] = SimpleNamespace(
            interaction_runtime=forged_runtime,
        )

        with pytest.raises(
            DurableToolExecutorContractError,
            match="raw|durable|receipt|authority",
        ):
            fixture["runtime"].prepare_tool_execution(forged_resume)

        assert fixture["invocations"] == []
        assert "tool.started" not in {
            event.event_type for event in fixture["bundle"].journal.events
        }
    finally:
        fixture["guard"].release()


def test_event_loop_replacement_still_uses_bootstrap_interaction_runtime():
    fixture = _approval_tool_runtime(attempt_id="attempt-loop-replaced")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": True,
                "modified_arguments": None,
                "reason": "",
            },
        )
        resume_context = _resume_approval_context(fixture)
        resume_context.event["loop"] = SimpleNamespace(
            interaction_runtime=SimpleNamespace(
                require_receipt=lambda session_id: (_ for _ in ()).throw(
                    AssertionError(
                        f"untrusted event runtime was used for {session_id}"
                    )
                )
            )
        )

        permit = fixture["runtime"].prepare_tool_execution(resume_context)
        completion = fixture["runtime"].execute_prepared_tool(
            resume_context,
            permit,
        )

        assert completion.visible_result == {"seen": "journal-owned"}
        assert fixture["invocations"] == ["journal-owned"]
    finally:
        fixture["guard"].release()


def test_bootstrap_rejects_official_interaction_runtime_with_different_store():
    with pytest.raises(
        ContextExecutionBundleError,
        match="interaction.*store|same.*store|authority",
    ):
        _approval_tool_runtime(
            attempt_id="attempt-wrong-interaction-store",
            interaction_store=InMemorySessionStore(),
        )


def test_approval_fails_closed_when_bootstrap_bound_no_interaction_runtime():
    fixture = _approval_tool_runtime(
        attempt_id="attempt-no-interaction-runtime",
        bind_interaction_runtime=False,
    )
    try:
        fixture["context"].event["loop"] = SimpleNamespace(
            interaction_runtime=fixture["interaction_runtime"],
        )

        with pytest.raises(
            DurableToolExecutorContractError,
            match="bootstrap.*interaction|interaction.*authority|runtime",
        ):
            fixture["runtime"].prepare_tool_execution(fixture["context"])

        assert fixture["invocations"] == []
        assert "tool.started" not in {
            event.event_type for event in fixture["bundle"].journal.events
        }
    finally:
        fixture["guard"].release()


def test_repeated_bootstrap_rejects_interaction_runtime_identity_drift():
    fixture = _approval_tool_runtime(attempt_id="attempt-runtime-drift")
    try:
        replacement_runtime = DurableInteractionRuntime(
            fixture["memory_runtime"],
            clock_ms=lambda: 456,
        )
        fixture["bootstrap"].event["loop"] = KernelLoop(
            interaction_runtime=replacement_runtime,
        )

        with pytest.raises(
            ContextExecutionBundleError,
            match="interaction.*changed|authority.*changed|identity",
        ):
            fixture["runtime"].build_harnesses()[0].build_delta(
                fixture["bootstrap"]
            )
    finally:
        fixture["guard"].release()


def test_cold_context_runtime_can_bind_new_official_runtime_on_same_store():
    fixture = _approval_tool_runtime(attempt_id="attempt-cold-runtime")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        request, receipt = _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": True,
                "modified_arguments": None,
                "reason": "",
            },
        )
        shared_journal = fixture["bundle"].journal
        cold_bundles = {}

        def build_cold(attempt):
            bundle = _bundle(attempt, journal=shared_journal)
            cold_bundles[(
                attempt.generation.execution_id,
                attempt.attempt_id,
            )] = bundle
            return bundle

        cold_memory_runtime = KernelMemoryRuntime.from_config(
            store=fixture["memory_runtime"].store,
        )
        cold_interaction_runtime = DurableInteractionRuntime(
            cold_memory_runtime,
            clock_ms=lambda: 789,
        )
        cold_loop = KernelLoop(
            interaction_runtime=cold_interaction_runtime,
        )
        cold_runtime = ContextRuntime.from_factory(
            owner_id="context-v2-cold-runtime",
            execution_factory=DurableContextRuntimeFactory(
                bundle_builder=build_cold,
                generation_resolver=lambda context, execution_id: (
                    f"generation-{execution_id}"
                ),
                current_input_resolver=_current_input,
            ),
        )
        cold_bootstrap = _context(
            session_id=fixture["execution_id"],
            run_id=fixture["attempt_id"],
            current_input=None,
        )
        cold_bootstrap.state.provider_state.provider = "openai"
        cold_bootstrap.state.provider_state.model = "gpt-5"
        cold_bootstrap.state.memory_state["session_revision"] = (
            cold_memory_runtime.load_session_snapshot(
                fixture["execution_id"]
            ).revision
        )
        cold_bootstrap.event.update(
            execution_guard=fixture["guard"],
            loop=cold_loop,
        )
        cold_runtime.build_harnesses()[0].build_delta(cold_bootstrap)
        cold_context = HarnessContext(
            state=cold_bootstrap.state,
            phase="on_tool_call",
            event={
                **fixture["context"].event,
                "execution_guard": fixture["guard"],
                "interaction_request": request.to_dict(),
                "interaction_response": copy.deepcopy(receipt.response),
                "loop": cold_loop,
            },
        )

        permit = cold_runtime.prepare_tool_execution(cold_context)
        completion = cold_runtime.execute_prepared_tool(
            cold_context,
            permit,
        )

        assert completion.visible_result == {"seen": "journal-owned"}
        assert fixture["invocations"] == ["journal-owned"]
        assert cold_bundles[
            (fixture["execution_id"], fixture["attempt_id"])
        ].journal is shared_journal
    finally:
        fixture["guard"].release()


def test_bootstrap_runtime_instance_method_monkeypatch_cannot_forge_receipt():
    fixture = _approval_tool_runtime(attempt_id="attempt-method-monkeypatch")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        _, request = _assert_durable_approval_pending(pending)
        forged_receipt = build_interaction_receipt(
            request,
            {
                "approved": True,
                "modified_arguments": {"query": "forged"},
                "reason": "",
            },
            submitted_at_ms=124,
        )
        fixture["interaction_runtime"].require_receipt = (
            lambda session_id: SimpleNamespace(
                request=request,
                receipt=forged_receipt,
            )
        )
        forged_resume = _resume_approval_context(
            fixture,
            raw_request=request.to_dict(),
            raw_response=forged_receipt.response,
        )

        with pytest.raises(
            DurableToolExecutorContractError,
            match="raw|durable|receipt|authority|changed",
        ):
            fixture["runtime"].prepare_tool_execution(forged_resume)

        assert fixture["invocations"] == []
        assert "tool.started" not in {
            event.event_type for event in fixture["bundle"].journal.events
        }
    finally:
        fixture["guard"].release()


@pytest.mark.parametrize("mutation", ["memory_runtime", "store"])
def test_bootstrap_interaction_memory_authority_mutation_fails_closed(mutation):
    fixture = _approval_tool_runtime(
        attempt_id=f"attempt-interaction-{mutation}-mutation"
    )
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        request, receipt = _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": True,
                "modified_arguments": None,
                "reason": "",
            },
        )
        bound_memory_runtime = fixture["interaction_runtime"].memory_runtime
        if mutation == "memory_runtime":
            fixture["interaction_runtime"].memory_runtime = (
                KernelMemoryRuntime.from_config(
                    store=bound_memory_runtime.store,
                )
            )
        else:
            original_store = bound_memory_runtime.store
            replacement_store = InMemorySessionStore()
            replacement_store._sessions = copy.deepcopy(
                original_store._sessions
            )
            replacement_store._revisions = copy.deepcopy(
                original_store._revisions
            )
            bound_memory_runtime.memory_manager.store = replacement_store

        with pytest.raises(
            DurableToolExecutorContractError,
            match="interaction.*authority|memory.*changed|store.*changed",
        ):
            fixture["runtime"].prepare_tool_execution(
                _resume_approval_context(
                    fixture,
                    raw_request=request.to_dict(),
                    raw_response=receipt.response,
                )
            )

        assert fixture["invocations"] == []
        assert "tool.started" not in {
            event.event_type for event in fixture["bundle"].journal.events
        }
    finally:
        fixture["guard"].release()


def test_approved_modified_arguments_bind_the_execution_subject():
    fixture = _approval_tool_runtime(attempt_id="attempt-modified")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        request, approval_receipt = _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": True,
                "modified_arguments": {"query": "durably-modified"},
                "reason": "",
            },
        )

        resume_context = _resume_approval_context(fixture)
        permit = fixture["runtime"].prepare_tool_execution(resume_context)
        completion = fixture["runtime"].execute_prepared_tool(
            resume_context,
            permit,
        )

        assert fixture["invocations"] == ["durably-modified"]
        assert completion.visible_result == {"seen": "durably-modified"}
        subject = completion.execution_subject
        assert subject.approval_state is DurableToolApprovalState.APPROVED
        assert subject.original_arguments_sha256 == strict_json_digest(
            {"query": "journal-owned"}
        )
        assert subject.effective_arguments_sha256 == strict_json_digest(
            {"query": "durably-modified"}
        )
        assert subject.approval_request_sha256 == request.request_digest
        assert subject.approval_receipt_sha256 == (
            approval_receipt.receipt_digest
        )
        assert [event.event_type for event in fixture["bundle"].journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        fixture["guard"].release()


def test_denied_approval_persists_synthetic_completion_without_invocation():
    fixture = _approval_tool_runtime(attempt_id="attempt-denied")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        request, approval_receipt = _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": False,
                "modified_arguments": None,
                "reason": "not allowed",
            },
        )

        resume_context = _resume_approval_context(fixture)
        permit = fixture["runtime"].prepare_tool_execution(resume_context)
        completion = fixture["runtime"].execute_prepared_tool(
            resume_context,
            permit,
        )

        assert fixture["invocations"] == []
        assert completion.visible_result == {
            "denied": True,
            "tool": "lookup",
            "reason": "not allowed",
        }
        subject = completion.execution_subject
        assert subject.approval_state.value == "denied"
        assert subject.approval_request_sha256 == request.request_digest
        assert subject.approval_receipt_sha256 == (
            approval_receipt.receipt_digest
        )
        assert [event.event_type for event in fixture["bundle"].journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
        assert completion.completion_artifact.byte_length > 0
    finally:
        fixture["guard"].release()


@pytest.mark.parametrize("approved", [False, True])
def test_answered_approval_does_not_bind_the_next_confirmable_call(approved):
    fixture = _approval_tool_runtime(
        attempt_id=f"attempt-next-approval-{approved}"
    )
    try:
        first_pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        first_request, _first_receipt = _persist_approval_pending(
            fixture,
            first_pending,
            {
                "approved": approved,
                "modified_arguments": None,
                "reason": "" if approved else "not allowed",
            },
        )
        session_snapshot = fixture["memory_runtime"].load_session_snapshot(
            fixture["execution_id"]
        )
        fixture["context"].state.memory_state["session_revision"] = (
            session_snapshot.revision
        )
        fixture["context"].state.iteration = 1

        next_call = ToolCall(
            call_id="call-next-approval",
            name="lookup",
            arguments={"query": "next-call"},
        )
        fixture["bundle"].durable_event_sink(
            {
                "type": "tool_call",
                "run_id": fixture["attempt_id"],
                "iteration": 1,
                "tool_name": next_call.name,
                "call_id": next_call.call_id,
                "arguments": copy.deepcopy(next_call.arguments),
            }
        )
        next_context = HarnessContext(
            state=fixture["context"].state,
            phase="on_tool_call",
            event={
                **fixture["context"].event,
                "tool_call": next_call,
                "tool_calls": [next_call],
            },
        )

        next_pending = fixture["runtime"].prepare_tool_execution(next_context)
        next_pending, next_request = _assert_durable_approval_pending(
            next_pending
        )

        assert next_request.interaction_id != first_request.interaction_id
        assert next_request.payload["call_id"] == next_call.call_id
        assert next_request.payload["arguments"] == {"query": "next-call"}
        assert fixture["invocations"] == []

        state = next_context.state
        state.run_status = "awaiting_interaction"
        state.last_continuation = copy.deepcopy(next_pending.continuation)
        state.suspend_state.signal_kind = "tool_approval"
        state.suspend_state.payload = {
            "interaction_request": next_request.to_dict(),
            "continuation": copy.deepcopy(next_pending.continuation),
        }
        checkpoint = build_execution_checkpoint(
            state,
            status="awaiting_interaction",
            run_id=fixture["attempt_id"],
        )
        fixture["memory_runtime"].save_execution_checkpoint_snapshot(
            fixture["execution_id"],
            checkpoint,
            interaction_request=next_request.to_dict(),
            expected_revision=next_request.created_revision,
            execution_fence=fixture["guard"].fence,
        )

        journal = validate_interaction_journal(
            fixture["memory_runtime"].load_session_state(
                fixture["execution_id"]
            )[INTERACTION_JOURNAL_KEY]
        )
        assert journal["active_id"] == next_request.interaction_id
        assert journal["entries"][first_request.interaction_id][
            "application"
        ] == {
            "schema_version": 1,
            "receipt_id": journal["entries"][first_request.interaction_id][
                "receipt"
            ]["receipt_id"],
            "applied_checkpoint_id": checkpoint["checkpoint_id"],
        }
        assert journal["entries"][next_request.interaction_id]["receipt"] is None
    finally:
        fixture["guard"].release()


@pytest.mark.parametrize("present_field", ["request", "response"])
def test_incomplete_tool_approval_resume_fields_fail_closed(present_field):
    fixture = _approval_tool_runtime(
        attempt_id=f"attempt-incomplete-resume-{present_field}"
    )
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        request, receipt = _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": True,
                "modified_arguments": None,
                "reason": "",
            },
        )
        resume_context = _resume_approval_context(
            fixture,
            raw_request=None,
            raw_response=None,
        )
        if present_field == "request":
            resume_context.event["interaction_request"] = request.to_dict()
        else:
            resume_context.event["interaction_response"] = copy.deepcopy(
                receipt.response
            )

        with pytest.raises(
            DurableToolExecutorContractError,
            match="resume fields are incomplete",
        ):
            fixture["runtime"].prepare_tool_execution(resume_context)

        assert fixture["invocations"] == []
        assert "tool.started" not in {
            event.event_type for event in fixture["bundle"].journal.events
        }
    finally:
        fixture["guard"].release()


def test_resume_approval_uses_durable_receipt_not_raw_response():
    fixture = _approval_tool_runtime(attempt_id="attempt-durable-response")
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        request, _receipt = _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": False,
                "modified_arguments": None,
                "reason": "durably denied",
            },
        )
        resume_context = _resume_approval_context(
            fixture,
            raw_request=request.to_dict(),
            raw_response={
                "approved": True,
                "modified_arguments": {"query": "forged"},
                "reason": "",
            },
        )

        permit = fixture["runtime"].prepare_tool_execution(resume_context)
        completion = fixture["runtime"].execute_prepared_tool(
            resume_context,
            permit,
        )

        assert completion.visible_result == {
            "denied": True,
            "tool": "lookup",
            "reason": "durably denied",
        }
        assert fixture["invocations"] == []
    finally:
        fixture["guard"].release()


@pytest.mark.parametrize("drift_kind", ["policy", "route"])
def test_approval_route_or_policy_drift_fails_closed(drift_kind):
    fixture = _approval_tool_runtime(
        attempt_id=f"attempt-{drift_kind}-drift"
    )
    try:
        pending = fixture["runtime"].prepare_tool_execution(
            fixture["context"]
        )
        _persist_approval_pending(
            fixture,
            pending,
            {
                "approved": True,
                "modified_arguments": None,
                "reason": "",
            },
        )

        if drift_kind == "policy":
            fixture["policy"]["description"] = "Changed approval policy"
        else:
            def changed_lookup(query: str):
                fixture["invocations"].append(f"changed:{query}")
                return {"changed": query}

            fixture["toolkit"].get("lookup").func = changed_lookup

        resume_context = _resume_approval_context(fixture)
        with pytest.raises(
            (InteractionIntegrityError, DurableToolExecutorContractError),
            match="approval|interaction|policy|route|manifest|binding|drift",
        ):
            fixture["runtime"].prepare_tool_execution(resume_context)

        assert fixture["invocations"] == []
        assert [event.event_type for event in fixture["bundle"].journal.events] == [
            "tool_call"
        ]
    finally:
        fixture["guard"].release()
