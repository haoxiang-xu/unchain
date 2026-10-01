from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest

from tests.context_v2.test_durable_tool_boundary import (
    _boundary,
    _persist_intent,
    _subject_for_intent,
)
from tests.context_v2.test_context_runtime_factory import (
    _bundle,
    _context,
    _current_input,
)
from unchain.context import (
    ContextExecutionBundleError,
    DurableToolCompletionDraft,
    DurableToolCompletionEnvelope,
    DurableToolApprovalState,
    DurableToolBoundaryCorruptError,
    DurableToolCompletionReceipt,
    DurableToolExecutionRequest,
    DurableToolExecutionSubject,
    DurableToolAuthorization,
    DurableToolExecutionDisposition,
    DurableToolExecutor,
    DurableToolExecutorContractError,
    DurableToolInvocationFailedError,
    DurableToolResultReceipt,
    ContextRuntime,
    DurableContextRuntimeFactory,
    ToolCompletionArtifactization,
    ToolResultArtifactization,
    ArtifactIntegrityError,
)
from unchain.execution import (
    ExecutionFence,
    ExecutionGuard,
    ExecutionLease,
    ExecutionLeaseNotOwnedError,
    ExecutionRuntime,
    _BorrowedExecutionGuard,
    _borrow_execution_guard,
)
from unchain.context.tool_executor import _assert_official_execution_guard_active
from unchain.context.artifacts import MAX_INLINE_TOOL_RESULT_BYTES
from unchain.context_content import encode_context_content_locator
from unchain.context.tool_harness import ContextToolAuthorityHarness
from unchain.context.tool_transitions import (
    DurableToolStateTransitionEnvelope,
    DurableToolStateTransitionDraft,
    resolve_subagent_transition_cas,
)
from unchain.subagents.plugin import _BatchExecutionGuard
from unchain.subagents.types import SubagentState
from unchain.memory import InMemorySessionStore
from unchain.kernel.harness import HarnessContext
from unchain.kernel.state import RunState
from unchain.kernel.types import ToolCall
from unchain.journal import AttemptRef, EventCursor, GenerationRef
from unchain.journal import ArtifactRef, ResourceRef
from unchain.tools import Toolkit
from unchain.tools.output_management import ToolOutputManager
from unchain.tools.runtime import ToolRuntimeOutcome


def _guard():
    return ExecutionRuntime(InMemorySessionStore()).acquire(
        "execution-1",
        "owner-1",
    )


def _execution_fixture():
    boundary, journal, order = _boundary()
    intent, arguments = _persist_intent(boundary)
    guard = _guard()
    subject = _subject_for_intent(
        intent,
        arguments,
        execution_fence=guard.fence,
    )
    request = DurableToolExecutionRequest(
        tool_name="lookup",
        call_id="call-1",
        iteration=0,
        subject=subject,
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=guard,
    )
    return executor, request, guard, journal, order, intent, arguments


def _invocation(
    executor,
    request,
    callback,
    *,
    effective_arguments=None,
):
    return executor.bind_invocation(
        request=request,
        effective_arguments=(
            {"query": "safe"}
            if effective_arguments is None
            else effective_arguments
        ),
        terminal_handler=callback,
    )


def test_executor_orders_side_effect_artifacts_and_terminal_journal() -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        def invoke(effective_arguments):
            assert effective_arguments == {"query": "safe"}
            order.append("invoke")
            return DurableToolCompletionDraft(
                result={"ok": True},
                should_observe=True,
            )

        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(executor, request, invoke),
        )

        assert order == [
            "journal:tool_call",
            "journal:tool.started",
            "invoke",
            "artifact:put",
            "artifact:put",
            "journal:tool_result",
        ]
        assert receipt.visible_result == {"ok": True}
        assert receipt.should_observe is True
        assert receipt.reused is False
        result_event = journal.events[-1]
        assert result_event.event_type == "tool_result"
        assert result_event.payload["completion_ref"] == (
            receipt.completion_artifact.ref.to_dict()
        )
        assert set(result_event.resource_refs) == {
            receipt.result_artifact.ref,
            receipt.completion_artifact.ref,
        }
    finally:
        guard.release()


def test_executor_carries_the_request_policy_into_the_durable_projection(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    executor._boundary.projector.bind_tool_output_manager(
        ToolOutputManager.active_default(attempt_id="run-1")
    )
    request = replace(request, tool_result_policy="artifact_only")
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"large": "x" * (MAX_INLINE_TOOL_RESULT_BYTES + 1)},
                ),
            ),
        )

        projection = journal.events[-1].payload["model_projection"]
        assert projection["result"]["projection"] == "artifact_only"
        assert projection["metadata"]["projection_policy"] == "artifact_only"
        assert receipt.model_projection == projection

        cold_executor = DurableToolExecutor(
            boundary=executor.boundary,
            artifacts=executor.artifacts,
            execution_guard=guard,
        )
        original_read = executor.artifacts.read_full

        def read_without_result_artifact(artifact, **kwargs):
            if artifact.ref == receipt.result_artifact.ref:
                raise AssertionError(
                    "cold recovery must not reread the large result artifact"
                )
            return original_read(artifact, **kwargs)

        monkeypatch.setattr(
            executor.artifacts,
            "read_full",
            read_without_result_artifact,
        )
        recovered = cold_executor.execute(
            request=request,
            guard=guard,
            invocation=None,
        )
        assert recovered.reused is True
        assert recovered.model_projection == projection
    finally:
        guard.release()


def test_projection_receipts_preserve_prior_positional_arguments() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        completion = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True},
                ),
            ),
        )
        result = DurableToolResultReceipt(
            completion.attempt,
            completion.tool_name,
            completion.call_id,
            completion.iteration,
            completion.execution_subject,
            completion.execution_subject.sha256,
            completion.visible_result,
            completion.result_artifact,
            completion.journal_cursor,
            completion.completion_artifact,
            False,
        )
        rebuilt_completion = DurableToolCompletionReceipt(
            completion.attempt,
            completion.tool_name,
            completion.call_id,
            completion.iteration,
            completion.execution_subject,
            completion.current_execution_fence,
            completion.visible_result,
            completion.should_observe,
            completion.result_artifact,
            completion.completion_artifact,
            completion.journal_cursor,
            None,
            False,
        )

        assert result.completion_artifact == completion.completion_artifact
        assert result.model_projection is None
        assert rebuilt_completion.transition is None
        assert rebuilt_completion.reused is False
        assert rebuilt_completion.model_projection is None
    finally:
        guard.release()


def test_runtime_uses_durable_projection_without_rereading_tool_artifact(
    monkeypatch,
) -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    executor._boundary.projector.bind_tool_output_manager(
        ToolOutputManager.active_default(attempt_id="attempt-1")
    )
    request = replace(request, tool_result_policy="artifact_only")
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"large": "result"},
                ),
            ),
        )
        bundle = _bundle(receipt.attempt)
        manager = ToolOutputManager.active_default(
            attempt_id=receipt.attempt.attempt_id
        )
        object.__setattr__(bundle, "tool_output_manager", manager)
        bundle.projector.bind_tool_output_manager(manager)
        runtime = ContextRuntime._for_test(
            owner_id="context-test",
            request_factory=lambda _context: None,
            durable_event_sink=lambda _event: None,
            partial_attempt_sink=lambda _event, _error: None,
        )
        runtime._attempt_bundles[("execution-1", "attempt-1")] = bundle
        state = RunState()
        state.session_state.session_id = "execution-1"
        context = HarnessContext(
            state=state,
            phase="tool",
            event={"run_id": "attempt-1"},
        )

        def unexpected_read(*_args, **_kwargs):
            raise AssertionError("runtime must reuse the durable projection")

        monkeypatch.setattr(bundle.artifacts, "read_full", unexpected_read)

        projected = runtime.project_tool_result_for_model(context, receipt)
        locator = encode_context_content_locator(receipt.result_artifact.ref)
        assert projected["projection"] == "artifact_only"
        assert projected["full_output_ref"] == locator
        assert projected["read_request"] == {
            "tool": "context_content_read",
            "arguments": {"ref": locator, "offset": 0, "limit": 8192},
        }
        assert receipt.model_projection["result"]["full_output_ref"] != locator
    finally:
        guard.release()


def test_executor_seals_transition_artifact_and_returns_bound_receipt() -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    base_state = SubagentState(
        root_agent_id="root",
        active_agent_id="root",
        active_lineage=["root"],
    ).to_dict()
    next_state = SubagentState(
        root_agent_id="root",
        active_agent_id="root.researcher.1",
        active_lineage=["root", "root.researcher.1"],
    ).to_dict()
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True},
                    state_transition=DurableToolStateTransitionDraft(
                        kind="subagent_snapshot",
                        base_state=base_state,
                        next_state=next_state,
                    ),
                ),
            ),
        )

        assert receipt.transition is not None
        assert receipt.transition.parent_attempt == receipt.attempt
        assert receipt.transition.call_id == receipt.call_id
        assert receipt.transition.execution_subject_sha256 == (
            receipt.execution_subject.sha256
        )
        assert receipt.transition.kind == "subagent_snapshot"
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
            "tool_result",
        ]
        seal = journal.events[-2]
        assert receipt.transition.next_state_artifact.ref in seal.resource_refs
        stored = executor.artifacts.read_full(
            receipt.transition.next_state_artifact,
            remaining_budget_bytes=(
                receipt.transition.next_state_artifact.byte_length
            ),
        )
        assert json.loads(stored.decode("utf-8")) == next_state
        assert order[-5:] == [
            "artifact:put",
            "artifact:put",
            "artifact:put",
            "journal:tool.subagent_completion.sealed",
            "journal:tool_result",
        ]
    finally:
        guard.release()


def test_transition_state_sanitizer_mutation_fails_closed_before_seal() -> None:
    boundary, journal, _order = _boundary(redact=True)
    intent, arguments = _persist_intent(boundary)
    guard = _guard()
    subject = _subject_for_intent(
        intent,
        arguments,
        execution_fence=guard.fence,
    )
    request = DurableToolExecutionRequest(
        tool_name="lookup",
        call_id="call-1",
        iteration=0,
        subject=subject,
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=guard,
    )
    try:
        with pytest.raises(
            ArtifactIntegrityError,
            match="control-plane JSON sanitizer changed",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True},
                        state_transition=DurableToolStateTransitionDraft(
                            kind="subagent_snapshot",
                            base_state={"active_agent_id": "root"},
                            next_state={"active_agent_id": "secret-value"},
                        ),
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_reuse_reads_and_verifies_the_sealed_next_state_artifact() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True},
                    state_transition=DurableToolStateTransitionDraft(
                        kind="subagent_snapshot",
                        base_state=SubagentState(
                            root_agent_id="root",
                            active_agent_id="root",
                            active_lineage=["root"],
                        ).to_dict(),
                        next_state=SubagentState(
                            root_agent_id="root",
                            active_agent_id="root.worker.1",
                            active_lineage=["root", "root.worker.1"],
                        ).to_dict(),
                    ),
                ),
            ),
        )
        next_artifact = receipt.transition.next_state_artifact
        repository = executor.artifacts._repository
        repository.by_id[next_artifact.ref.resource_id] = (
            next_artifact,
            b"{}",
        )

        with pytest.raises(
            (ArtifactIntegrityError, DurableToolExecutorContractError),
            match="next-state|next state|byte_length|sha256|artifact",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=None,
            )
    finally:
        guard.release()


def test_terminal_transition_without_same_parent_seal_is_corrupt() -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True},
                    state_transition=DurableToolStateTransitionDraft(
                        kind="subagent_snapshot",
                        base_state=SubagentState().to_dict(),
                        next_state=SubagentState(
                            root_agent_id="root",
                            active_agent_id="root.worker.1",
                            active_lineage=["root", "root.worker.1"],
                        ).to_dict(),
                    ),
                ),
            ),
        )
        journal.events[:] = [
            event
            for event in journal.events
            if event.event_type != "tool.subagent_completion.sealed"
        ]

        with pytest.raises(
            (DurableToolBoundaryCorruptError, DurableToolExecutorContractError),
            match="transition.*seal|seal.*transition|stateful.*seal",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=None,
            )
    finally:
        guard.release()


def test_terminal_seal_without_completion_transition_is_corrupt() -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True},
                    state_transition=DurableToolStateTransitionDraft(
                        kind="subagent_snapshot",
                        base_state=SubagentState().to_dict(),
                        next_state=SubagentState(
                            root_agent_id="root",
                            active_agent_id="root.worker.1",
                            active_lineage=["root", "root.worker.1"],
                        ).to_dict(),
                    ),
                ),
            ),
        )
        stateless = DurableToolCompletionEnvelope(
            attempt=receipt.attempt,
            tool_name=receipt.tool_name,
            call_id=receipt.call_id,
            iteration=receipt.iteration,
            execution_subject=receipt.execution_subject,
            execution_subject_sha256=receipt.execution_subject.sha256,
            result_artifact=receipt.result_artifact,
            visible_result=receipt.visible_result,
            should_observe=receipt.should_observe,
            transition=None,
        )
        stateless_artifactization = (
            executor.artifacts.artifactize_tool_completion(
                stateless.to_dict(),
                operation_id="artifact.test-stateless-completion",
            )
        )
        result_event = journal.events[-1]
        result_payload = dict(result_event.payload)
        result_payload.update(
            {
                "completion_ref": (
                    stateless_artifactization.artifact.ref.to_dict()
                ),
                "completion_bytes": (
                    stateless_artifactization.artifact.byte_length
                ),
                "completion_sha256": stateless_artifactization.artifact.sha256,
                "completion_preview": stateless_artifactization.artifact.preview,
            }
        )
        journal.events[-1] = replace(
            result_event,
            payload=result_payload,
            resource_refs=(
                receipt.result_artifact.ref,
                stateless_artifactization.artifact.ref,
            ),
        )

        with pytest.raises(
            (DurableToolBoundaryCorruptError, DurableToolExecutorContractError),
            match="seal.*transition|transition.*seal|stateful.*seal",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=None,
            )
    finally:
        guard.release()


def test_transition_operation_id_is_derived_from_execution_subject() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        next_state_artifact = executor.artifacts.persist_exact_json(
            SubagentState().to_dict(),
            operation_id="artifact.test-next-state",
            operation_binding={"kind": "test"},
        )

        with pytest.raises(
            DurableToolExecutorContractError,
            match="operation.*execution subject|operation.*derived",
        ):
            DurableToolStateTransitionEnvelope(
                parent_attempt=executor.boundary.attempt,
                call_id=request.call_id,
                execution_subject_sha256=request.subject.sha256,
                operation_id="transition.subagent-completion." + "0" * 64,
                kind="subagent_snapshot",
                base_state_sha256="1" * 64,
                next_state_artifact=next_state_artifact,
            )
    finally:
        guard.release()


def test_nonempty_handoff_refs_fail_closed_without_descriptor_authority() -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="handoff refs.*descriptor",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True},
                        state_transition=DurableToolStateTransitionDraft(
                            kind="subagent_snapshot",
                            base_state=SubagentState().to_dict(),
                            next_state=SubagentState().to_dict(),
                            handoff_refs=(
                                ResourceRef(
                                    "artifact",
                                    "unverified-handoff",
                                    1,
                                ),
                            ),
                        ),
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_tool_authority_harness_applies_transition_with_exact_cas() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    base = SubagentState(
        root_agent_id="root",
        active_agent_id="root",
        active_lineage=["root"],
    )
    next_state = SubagentState(
        root_agent_id="root",
        active_agent_id="root.worker.1",
        active_lineage=["root", "root.worker.1"],
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True},
                    state_transition=DurableToolStateTransitionDraft(
                        kind="subagent_snapshot",
                        base_state=base.to_dict(),
                        next_state=next_state.to_dict(),
                    ),
                ),
            ),
        )

        class Runtime:
            def prepare_tool_execution(self, context):
                del context
                return object()

            def execute_prepared_tool(self, context, permit):
                del context, permit
                return receipt

            def _bundle_for_context(self, context):
                del context
                return SimpleNamespace(
                    attempt=receipt.attempt,
                    artifacts=executor.artifacts,
                )

            def materialize_tool_transition(self, context, durable_receipt):
                return resolve_subagent_transition_cas(
                    artifacts=executor.artifacts,
                    transition=durable_receipt.transition,
                    current_state=context.state.subagent_state,
                )

        harness = ContextToolAuthorityHarness(runtime=Runtime())

        def build_context(state):
            run_state = _context(
                session_id="execution-1",
                run_id="attempt-1",
            ).state
            run_state.provider_state.provider = "openai"
            run_state.subagent_state = state
            return HarnessContext(
                state=run_state,
                phase="on_tool_call",
                event={
                    "run_id": "attempt-1",
                    "tool_call": ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"query": "safe"},
                    ),
                },
            )

        applied = harness.build_delta(build_context(base.copy()))
        assert applied.state_updates["subagent_state"] == next_state

        idempotent = harness.build_delta(build_context(next_state.copy()))
        assert "subagent_state" not in idempotent.state_updates

        with pytest.raises(
            DurableToolExecutorContractError,
            match="state transition CAS conflict",
        ):
            harness.build_delta(
                build_context(
                    SubagentState(
                        root_agent_id="root",
                        active_agent_id="root.other.1",
                        active_lineage=["root", "root.other.1"],
                    )
                )
            )
    finally:
        guard.release()


def test_bound_invocation_canonicalizes_json_scalar_subclasses_before_handler(
) -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    observed: dict[str, object] = {}

    class ForgedText(str):
        def __str__(self) -> str:
            return "DANGEROUS"

    def invoke(effective_arguments):
        query = effective_arguments["query"]
        observed["type"] = type(query)
        observed["value"] = query
        observed["string"] = str(query)
        return DurableToolCompletionDraft(result={"ok": True})

    try:
        executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                invoke,
                effective_arguments={"query": ForgedText("safe")},
            ),
        )

        assert observed == {
            "type": str,
            "value": "safe",
            "string": "safe",
        }
    finally:
        guard.release()


def test_executor_only_persists_and_returns_sanitized_tool_content() -> None:
    boundary, journal, _order = _boundary(redact=True)
    intent, arguments = _persist_intent(boundary)
    guard = _guard()
    subject = _subject_for_intent(
        intent,
        arguments,
        execution_fence=guard.fence,
    )
    request = DurableToolExecutionRequest(
        tool_name="lookup",
        call_id="call-1",
        iteration=0,
        subject=subject,
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=guard,
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"output": "secret-value"}
                ),
            ),
        )

        assert receipt.visible_result == {"output": "[REDACTED]"}
        serialized_journal = json.dumps(
            [event.to_dict() for event in journal.events],
            sort_keys=True,
        )
        assert "secret-value" not in serialized_journal
        repository = executor.artifacts._repository
        assert all(
            b"secret-value" not in content
            for _artifact, content in repository.by_id.values()
        )
    finally:
        guard.release()


def test_executor_reuses_verified_completion_without_invoking_again() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    calls = 0
    try:
        def invoke(_effective_arguments):
            nonlocal calls
            calls += 1
            return DurableToolCompletionDraft(result={"ok": True})

        first = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(executor, request, invoke),
        )
        second = executor.execute(
            request=request,
            guard=guard,
            invocation=None,
        )

        assert calls == 1
        assert first.reused is False
        assert second.reused is True
        assert second.visible_result == first.visible_result
        assert second.result_artifact == first.result_artifact
        assert second.completion_artifact == first.completion_artifact
    finally:
        guard.release()


def test_cold_executor_reuses_nfc_canonical_result_and_completion_previews() -> None:
    attempt = AttemptRef(GenerationRef("e", "g"), "a")
    boundary, _journal, _order = _boundary(attempt=attempt)
    intent, arguments = _persist_intent(
        boundary,
        tool_name="t",
        call_id="c",
        arguments={"q": "safe"},
    )
    guard = ExecutionRuntime(InMemorySessionStore()).acquire("e", "o")
    subject = _subject_for_intent(
        intent,
        arguments,
        approval_state=DurableToolApprovalState.NOT_REQUIRED,
        approval_request_sha256="",
        approval_receipt_sha256="",
        execution_fence=guard.fence,
    )
    request = DurableToolExecutionRequest(
        tool_name="t",
        call_id="c",
        iteration=0,
        subject=subject,
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=guard,
    )
    calls = 0
    try:
        def invoke(_effective_arguments):
            nonlocal calls
            calls += 1
            return DurableToolCompletionDraft(result={"value": "e\u0301"})

        first = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                invoke,
                effective_arguments=arguments,
            ),
        )
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

        assert calls == 1
        assert first.result_artifact.preview == '{"value":"é"}'
        assert "é" in first.completion_artifact.preview
        assert "e\u0301" not in first.completion_artifact.preview
        assert recovered.reused is True
        assert recovered.visible_result == {"value": "e\u0301"}
        assert recovered.result_artifact == first.result_artifact
        assert recovered.completion_artifact == first.completion_artifact
    finally:
        guard.release()


def test_executor_rejects_json_type_change_in_reuse_authorization(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True}
                ),
            ),
        )
        original = executor.boundary.authorize_execution
        handler_calls = 0

        def mutate_authorization(**kwargs):
            authorization = original(**kwargs)
            object.__setattr__(
                authorization,
                "visible_result",
                {"ok": 1},
            )
            return authorization

        def invoke(_effective_arguments):
            nonlocal handler_calls
            handler_calls += 1
            return DurableToolCompletionDraft(result={"bad": True})

        monkeypatch.setattr(
            executor.boundary,
            "authorize_execution",
            mutate_authorization,
        )
        with pytest.raises(
            DurableToolExecutorContractError,
            match="reuse authorization",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        assert handler_calls == 0
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        guard.release()


def test_executor_reuses_terminal_completion_after_guard_reacquires() -> None:
    executor, request, guard, _journal, _order, intent, arguments = (
        _execution_fixture()
    )
    calls = 0
    try:
        def invoke(_effective_arguments):
            nonlocal calls
            calls += 1
            return DurableToolCompletionDraft(result={"ok": True})

        executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(executor, request, invoke),
        )
        guard.release_for_wait()
        guard.reacquire()
        resumed_request = DurableToolExecutionRequest(
            tool_name=request.tool_name,
            call_id=request.call_id,
            iteration=request.iteration,
            subject=_subject_for_intent(
                intent,
                arguments,
                execution_fence=guard.fence,
            ),
        )

        reused = executor.execute(
            request=resumed_request,
            guard=guard,
            invocation=_invocation(
                executor,
                resumed_request,
                lambda _effective_arguments: (_ for _ in ()).throw(
                    AssertionError("recovery must not invoke the tool")
                ),
            ),
        )

        assert calls == 1
        assert reused.reused is True
        assert reused.execution_subject == request.subject
        assert reused.current_execution_fence == guard.fence
    finally:
        guard.release()


def test_executor_requires_the_subject_to_match_a_live_execution_guard() -> None:
    executor, request, guard, journal, order, intent, arguments = (
        _execution_fixture()
    )
    forged = DurableToolExecutionRequest(
        tool_name=request.tool_name,
        call_id=request.call_id,
        iteration=request.iteration,
        subject=_subject_for_intent(
            intent,
            arguments,
            execution_fence=type(guard.fence)(
                guard.fence.execution_id,
                "forged-owner",
                guard.fence.fencing_token + 1,
            ),
        ),
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="live execution fence",
        ):
            executor.execute(
                request=forged,
                guard=guard,
                invocation=_invocation(
                    executor,
                    forged,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"bad": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        guard.release()


def test_execution_request_rejects_subject_subclasses() -> None:
    _executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )

    class ForgedSubject(DurableToolExecutionSubject):
        pass

    forged_subject = ForgedSubject(
        **{
            item.name: getattr(request.subject, item.name)
            for item in fields(DurableToolExecutionSubject)
        }
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="execution subject",
        ):
            DurableToolExecutionRequest(
                tool_name=request.tool_name,
                call_id=request.call_id,
                iteration=request.iteration,
                subject=forged_subject,
            )
    finally:
        guard.release()


def test_executor_rechecks_exact_subject_before_any_started_claim() -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )

    class ForgedSubject(DurableToolExecutionSubject):
        pass

    invocation = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"bad": True}
        ),
    )
    forged_subject = ForgedSubject(
        **{
            item.name: getattr(request.subject, item.name)
            for item in fields(DurableToolExecutionSubject)
        }
    )
    object.__setattr__(request, "subject", forged_subject)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="execution subject",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=invocation,
            )

        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        guard.release()


@pytest.mark.parametrize("nested_field", ["intent_cursor", "execution_fence"])
def test_execution_request_rejects_nested_subject_subclasses(
    nested_field,
) -> None:
    _executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )

    class ForgedCursor(EventCursor):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    class ForgedFence(ExecutionFence):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    forged_subject = replace(request.subject)
    if nested_field == "intent_cursor":
        nested_value = ForgedCursor(
            request.subject.intent_cursor.store_seq,
            request.subject.intent_cursor.event_id,
        )
    else:
        nested_value = ForgedFence(
            request.subject.execution_fence.execution_id,
            request.subject.execution_fence.owner_id,
            request.subject.execution_fence.fencing_token,
        )
    object.__setattr__(forged_subject, nested_field, nested_value)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="execution subject",
        ):
            DurableToolExecutionRequest(
                tool_name=request.tool_name,
                call_id=request.call_id,
                iteration=request.iteration,
                subject=forged_subject,
            )
    finally:
        guard.release()


@pytest.mark.parametrize("nested_field", ["intent_cursor", "execution_fence"])
def test_executor_rechecks_nested_subject_types_before_started(
    nested_field,
) -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    handler_calls = 0

    class ForgedCursor(EventCursor):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    class ForgedFence(ExecutionFence):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(result={"bad": True})

    invocation = _invocation(executor, request, invoke)
    if nested_field == "intent_cursor":
        nested_value = ForgedCursor(
            request.subject.intent_cursor.store_seq,
            request.subject.intent_cursor.event_id,
        )
    else:
        nested_value = ForgedFence(
            request.subject.execution_fence.execution_id,
            request.subject.execution_fence.owner_id,
            request.subject.execution_fence.fencing_token,
        )
    object.__setattr__(request.subject, nested_field, nested_value)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="execution subject",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=invocation,
            )

        assert handler_calls == 0
        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        guard.release()


def test_executor_rejects_forged_authorization_before_invoking_handler(
    monkeypatch,
) -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    handler_calls = 0

    class ForgedAuthorization(DurableToolAuthorization):
        pass

    forged = ForgedAuthorization(
        disposition=DurableToolExecutionDisposition.EXECUTE,
        tool_name=request.tool_name,
        call_id=request.call_id,
        iteration=request.iteration,
        execution_subject=request.subject,
        started_cursor=EventCursor(2, "forged-started"),
        _authority=executor.boundary._authority,
    )

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(result={"bad": True})

    monkeypatch.setattr(
        executor.boundary,
        "authorize_execution",
        lambda **_kwargs: forged,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="authorization",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        assert handler_calls == 0
        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        guard.release()


@pytest.mark.parametrize(
    "nested_field",
    ["started_cursor", "current_execution_fence"],
)
def test_executor_rejects_nested_authorization_subclasses_before_handler(
    monkeypatch,
    nested_field,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.authorize_execution
    handler_calls = 0

    class ForgedCursor(EventCursor):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    class ForgedFence(ExecutionFence):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    def mutate_authorization(**kwargs):
        authorization = original(**kwargs)
        if nested_field == "started_cursor":
            nested_value = ForgedCursor(
                authorization.started_cursor.store_seq,
                authorization.started_cursor.event_id,
            )
        else:
            nested_value = ForgedFence(
                authorization.current_execution_fence.execution_id,
                authorization.current_execution_fence.owner_id,
                authorization.current_execution_fence.fencing_token,
            )
        object.__setattr__(authorization, nested_field, nested_value)
        return authorization

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(result={"bad": True})

    monkeypatch.setattr(
        executor.boundary,
        "authorize_execution",
        mutate_authorization,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="authorization",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        assert handler_calls == 0
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


@pytest.mark.parametrize("nested_record", ["cursor", "fence"])
def test_executor_rejects_mutated_fields_inside_exact_authorization_cursor(
    monkeypatch,
    nested_record,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.authorize_execution
    handler_calls = 0

    class ForgedText(str):
        __hash__ = str.__hash__

        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    def mutate_authorization(**kwargs):
        authorization = original(**kwargs)
        if nested_record == "cursor":
            object.__setattr__(
                authorization.started_cursor,
                "event_id",
                ForgedText("forged-started-event"),
            )
        else:
            object.__setattr__(
                authorization.current_execution_fence,
                "owner_id",
                ForgedText("forged-owner"),
            )
        return authorization

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(result={"bad": True})

    monkeypatch.setattr(
        executor.boundary,
        "authorize_execution",
        mutate_authorization,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="authorization",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        assert handler_calls == 0
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_executor_rejects_route_manifest_change_before_started_claim() -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    invocation = replace(
        _invocation(
            executor,
            request,
            lambda _effective_arguments: DurableToolCompletionDraft(
                result={"bad": True}
            ),
        ),
        route_manifest_sha256="a" * 64,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="route",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=invocation,
            )
        assert [event.event_type for event in journal.events] == ["tool_call"]
    finally:
        guard.release()


def test_executor_rejects_replaced_invocation_callback_before_started() -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    dangerous_calls = 0

    def dangerous(_effective_arguments):
        nonlocal dangerous_calls
        dangerous_calls += 1
        return DurableToolCompletionDraft(result={"dangerous": True})

    safe = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"safe": True}
        ),
    )
    forged = replace(safe, terminal_handler=dangerous)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="runtime binding",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=forged,
            )

        assert dangerous_calls == 0
        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        guard.release()


def test_executor_uses_bound_snapshot_if_invocation_object_is_mutated() -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    dangerous_calls = 0

    def dangerous(_effective_arguments):
        nonlocal dangerous_calls
        dangerous_calls += 1
        return DurableToolCompletionDraft(result={"dangerous": True})

    invocation = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"safe": True}
        ),
    )
    object.__setattr__(invocation, "terminal_handler", dangerous)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="durable subject",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=invocation,
            )

        assert dangerous_calls == 0
        assert [event.event_type for event in journal.events] == ["tool_call"]
    finally:
        guard.release()


def test_executor_rejects_effective_arguments_mismatch_before_started_claim() -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="effective arguments",
        ):
            _invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"bad": True}
                ),
                effective_arguments={"query": "DANGEROUS"},
            )

        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        guard.release()


def test_executor_rejects_forged_execution_guard_subclass_before_started() -> None:
    executor, request, real_guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )

    class ForgedGuard(ExecutionGuard):
        def __init__(self, fence):
            self._forged_fence = fence

        def assert_active(self):
            return ExecutionLease(
                execution_id=self._forged_fence.execution_id,
                owner_id=self._forged_fence.owner_id,
                fencing_token=self._forged_fence.fencing_token,
                acquired_at_ms=1,
                expires_at_ms=2,
            )

    forged_guard = ForgedGuard(request.subject.execution_fence)
    invocation = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"bad": True}
        ),
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="bootstrap binding",
        ):
            executor.execute(
                request=request,
                guard=forged_guard,
                invocation=invocation,
            )

        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        real_guard.release()


def test_executor_rejects_different_official_root_with_same_fence() -> None:
    executor, request, real_guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )
    unrelated_official_guard = _guard()
    invocation = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"bad": True}
        ),
    )
    try:
        assert unrelated_official_guard.fence == real_guard.fence
        with pytest.raises(
            DurableToolExecutorContractError,
            match="bootstrap binding",
        ):
            executor.execute(
                request=request,
                guard=unrelated_official_guard,
                invocation=invocation,
            )
        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        unrelated_official_guard.release()
        real_guard.release()


def test_official_borrowed_guard_verification_returns_child_fence() -> None:
    root = _guard()
    child = _borrow_execution_guard(
        root,
        session_id="execution-1:child-1",
    )
    try:
        verified = _assert_official_execution_guard_active(child)
        assert verified.execution_id == "execution-1:child-1"
        assert verified.owner_id == root.fence.owner_id
        assert verified.fencing_token == root.fence.fencing_token
    finally:
        child.release()
        root.release()


@pytest.mark.parametrize("use_batch_guard", [False, True])
def test_executor_accepts_projected_child_guard_for_child_attempt(
    use_batch_guard,
) -> None:
    child_attempt = AttemptRef(
        GenerationRef("execution-1:child-1", "generation-child-1"),
        "attempt-child-1",
    )
    boundary, journal, order = _boundary(attempt=child_attempt)
    intent, arguments = _persist_intent(boundary)
    root = _guard()
    child = _borrow_execution_guard(
        root,
        session_id=child_attempt.generation.execution_id,
    )
    guard = (
        _BatchExecutionGuard(child, threading.Event())
        if use_batch_guard
        else child
    )
    subject = _subject_for_intent(
        intent,
        arguments,
        execution_fence=_assert_official_execution_guard_active(guard),
    )
    request = DurableToolExecutionRequest(
        tool_name="lookup",
        call_id="call-1",
        iteration=0,
        subject=subject,
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=guard,
    )
    try:
        receipt = executor.execute(
            request=request,
            invocation=_invocation(
                executor,
                request,
                lambda effective_arguments: DurableToolCompletionDraft(
                    result={"arguments": effective_arguments}
                ),
            ),
        )
        assert receipt.attempt == child_attempt
        assert receipt.current_execution_fence.execution_id == (
            child_attempt.generation.execution_id
        )
        assert receipt.visible_result == {"arguments": arguments}
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
        assert order.count("artifact:put") == 2
    finally:
        child.release()
        root.release()


def test_forged_foreign_borrowed_guard_and_guard_cycle_fail_closed() -> None:
    root = _guard()
    foreign = _BorrowedExecutionGuard(
        parent=root,
        session_id="foreign-child",
    )
    batch = _BatchExecutionGuard(root, threading.Event())
    batch._delegate = batch
    waiting = _borrow_execution_guard(
        root,
        session_id="execution-1:waiting-child",
    )
    waiting.release_for_wait()
    abort_event = threading.Event()
    abort_event.set()
    aborted_batch = _BatchExecutionGuard(root, abort_event)
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="outside its root execution",
        ):
            _assert_official_execution_guard_active(foreign)
        with pytest.raises(
            DurableToolExecutorContractError,
            match="cyclic",
        ):
            _assert_official_execution_guard_active(batch)
        with pytest.raises(ExecutionLeaseNotOwnedError):
            _assert_official_execution_guard_active(waiting)
        with pytest.raises(
            DurableToolExecutorContractError,
            match="aborted",
        ):
            _assert_official_execution_guard_active(aborted_batch)
    finally:
        waiting.release()
        foreign.release()
        root.release()


def test_batch_guard_requires_real_abort_event_and_frozen_root_chain() -> None:
    class FakeEvent:
        @staticmethod
        def is_set():
            return False

    root_a = _guard()
    root_b = _guard()
    fake_event_batch = _BatchExecutionGuard(root_a, FakeEvent())
    boundary, journal, order = _boundary()
    intent, arguments = _persist_intent(boundary)
    batch = _BatchExecutionGuard(root_a, threading.Event())
    subject = _subject_for_intent(
        intent,
        arguments,
        execution_fence=_assert_official_execution_guard_active(batch),
    )
    request = DurableToolExecutionRequest(
        tool_name="lookup",
        call_id="call-1",
        iteration=0,
        subject=subject,
    )
    executor = DurableToolExecutor(
        boundary=boundary,
        artifacts=boundary.projector.artifacts,
        execution_guard=batch,
    )
    invocation = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"bad": True}
        ),
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="abort capability",
        ):
            _assert_official_execution_guard_active(fake_event_batch)

        assert root_a.fence == root_b.fence
        batch._delegate = root_b
        with pytest.raises(
            DurableToolExecutorContractError,
            match="(?:root|chain) changed",
        ):
            executor.execute(
                request=request,
                invocation=invocation,
            )
        assert [event.event_type for event in journal.events] == ["tool_call"]
        assert order == ["journal:tool_call"]
    finally:
        root_a.release()
        root_b.release()


def test_deep_official_batch_chain_does_not_overflow_python_recursion() -> None:
    root = _guard()
    guard = root
    for _index in range(1_200):
        guard = _BatchExecutionGuard(guard, threading.Event())
    try:
        verified = _assert_official_execution_guard_active(guard)
        assert verified == root.fence
    finally:
        root.release()


def test_completion_artifact_failure_leaves_uncertain_started_receipt(
    monkeypatch,
) -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )

    def fail_completion(*args, **kwargs):
        del args, kwargs
        raise OSError("completion store unavailable")

    monkeypatch.setattr(
        executor.artifacts,
        "artifactize_tool_completion",
        fail_completion,
    )
    try:
        with pytest.raises(OSError, match="completion store unavailable"):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
        assert order[-1] == "artifact:put"
    finally:
        guard.release()


def test_executor_never_returns_completion_from_a_duck_typed_result_receipt(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )

    def forge_receipt(
        _authorization,
        *,
        artifactization,
        completion_artifactization,
    ):
        return SimpleNamespace(
            artifact=artifactization.artifact,
            completion_artifact=completion_artifactization.artifact,
            visible_result=artifactization.visible_result,
            cursor=EventCursor(3, "forged-result"),
            duplicate=False,
        )

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        forge_receipt,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="result receipt",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_executor_rejects_mutated_duplicate_flag_on_fresh_result_receipt(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.persist_prepared_result

    def mutate_duplicate(
        authorization,
        *,
        artifactization,
        completion_artifactization,
    ):
        receipt = original(
            authorization,
            artifactization=artifactization,
            completion_artifactization=completion_artifactization,
        )
        object.__setattr__(receipt, "duplicate", True)
        return receipt

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        mutate_duplicate,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="result receipt",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        guard.release()


@pytest.mark.parametrize("nested_field", ["cursor", "artifact"])
def test_executor_rejects_nested_result_receipt_subclasses(
    monkeypatch,
    nested_field,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.persist_prepared_result

    class ForgedCursor(EventCursor):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    class ForgedArtifact(ArtifactRef):
        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    def mutate_receipt(
        authorization,
        *,
        artifactization,
        completion_artifactization,
    ):
        receipt = original(
            authorization,
            artifactization=artifactization,
            completion_artifactization=completion_artifactization,
        )
        if nested_field == "cursor":
            nested_value = ForgedCursor(
                receipt.cursor.store_seq,
                receipt.cursor.event_id,
            )
        else:
            nested_value = ForgedArtifact(
                ref=receipt.artifact.ref,
                media_type=receipt.artifact.media_type,
                byte_length=receipt.artifact.byte_length,
                sha256=receipt.artifact.sha256,
                preview=receipt.artifact.preview,
            )
        object.__setattr__(receipt, nested_field, nested_value)
        return receipt

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        mutate_receipt,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="result receipt",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        guard.release()


def test_executor_rejects_mutated_fields_inside_exact_result_cursor(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.persist_prepared_result

    class ForgedText(str):
        __hash__ = str.__hash__

        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    def mutate_receipt(
        authorization,
        *,
        artifactization,
        completion_artifactization,
    ):
        receipt = original(
            authorization,
            artifactization=artifactization,
            completion_artifactization=completion_artifactization,
        )
        object.__setattr__(
            receipt.cursor,
            "event_id",
            ForgedText("forged-result-event"),
        )
        return receipt

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        mutate_receipt,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="result receipt",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        guard.release()


@pytest.mark.parametrize(
    "nested_record",
    ["attempt", "artifact", "resource"],
)
def test_executor_rejects_mutated_fields_inside_exact_result_records(
    monkeypatch,
    nested_record,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.persist_prepared_result

    class ForgedText(str):
        __hash__ = str.__hash__

        def __eq__(self, _other):
            return True

        def __ne__(self, _other):
            return False

    def mutate_receipt(
        authorization,
        *,
        artifactization,
        completion_artifactization,
    ):
        receipt = original(
            authorization,
            artifactization=artifactization,
            completion_artifactization=completion_artifactization,
        )
        if nested_record == "attempt":
            isolated_attempt = AttemptRef.from_dict(
                receipt.attempt.to_dict()
            )
            object.__setattr__(receipt, "attempt", isolated_attempt)
            object.__setattr__(
                isolated_attempt.generation,
                "generation_id",
                ForgedText("forged-generation"),
            )
        elif nested_record == "artifact":
            isolated_artifact = ArtifactRef.from_dict(
                receipt.artifact.to_dict()
            )
            object.__setattr__(receipt, "artifact", isolated_artifact)
            object.__setattr__(
                isolated_artifact,
                "sha256",
                ForgedText("a" * 64),
            )
        else:
            isolated_artifact = ArtifactRef.from_dict(
                receipt.artifact.to_dict()
            )
            object.__setattr__(receipt, "artifact", isolated_artifact)
            object.__setattr__(
                isolated_artifact.ref,
                "resource_id",
                ForgedText("forged-resource"),
            )
        return receipt

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        mutate_receipt,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="result receipt",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        guard.release()


def test_executor_rejects_json_type_change_in_result_receipt(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.boundary.persist_prepared_result

    def mutate_receipt(
        authorization,
        *,
        artifactization,
        completion_artifactization,
    ):
        receipt = original(
            authorization,
            artifactization=artifactization,
            completion_artifactization=completion_artifactization,
        )
        object.__setattr__(receipt, "visible_result", {"ok": 1})
        return receipt

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        mutate_receipt,
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="result receipt",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool_result",
        ]
    finally:
        guard.release()


def test_bound_invocation_is_consumed_after_one_execute_attempt() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    invocation = _invocation(
        executor,
        request,
        lambda _effective_arguments: DurableToolCompletionDraft(
            result={"ok": True}
        ),
    )
    try:
        executor.execute(
            request=request,
            guard=guard,
            invocation=invocation,
        )
        assert executor._invocation_bindings == {}
        replayed = executor.execute(
            request=request,
            guard=guard,
            invocation=invocation,
        )
        assert replayed.reused is True
    finally:
        guard.release()


def test_same_bound_invocation_has_only_one_concurrent_consumer() -> None:
    executor, request, guard, _journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    handler_calls = 0

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(result={"ok": True})

    invocation = _invocation(executor, request, invoke)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    executor.execute,
                    request=request,
                    guard=guard,
                    invocation=invocation,
                )
                for _index in range(2)
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=2))
                except Exception as error:
                    outcomes.append(error)

        successes = [value for value in outcomes if not isinstance(value, Exception)]
        failures = [value for value in outcomes if isinstance(value, Exception)]
        assert len(successes) == 2
        assert failures == []
        assert sorted(receipt.reused for receipt in successes) == [False, True]
        assert handler_calls == 1
    finally:
        guard.release()


def test_tool_result_journal_failure_never_returns_visible_completion() -> None:
    executor, request, guard, journal, order, _intent, _arguments = (
        _execution_fixture()
    )

    def invoke(_effective_arguments):
        journal.fail_next = True
        return DurableToolCompletionDraft(result={"ok": True})

    try:
        with pytest.raises(OSError, match="journal unavailable"):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
        assert order[-2:] == ["artifact:put", "artifact:put"]
    finally:
        guard.release()


def test_seal_survives_result_failure_and_cold_executor_finalizes_without_handler(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    handler_calls = 0
    original_append = journal.append
    fail_result_once = True

    def append(*, request):
        nonlocal fail_result_once
        if request.event_type == "tool_result" and fail_result_once:
            fail_result_once = False
            raise OSError("journal unavailable after seal")
        return original_append(request=request)

    monkeypatch.setattr(journal, "append", append)

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(
            result={"ok": True},
            state_transition=DurableToolStateTransitionDraft(
                kind="subagent_snapshot",
                base_state=SubagentState(
                    root_agent_id="root",
                    active_agent_id="root",
                    active_lineage=["root"],
                ).to_dict(),
                next_state=SubagentState(
                    root_agent_id="root",
                    active_agent_id="root.researcher.1",
                    active_lineage=["root", "root.researcher.1"],
                ).to_dict(),
            ),
        )

    try:
        with pytest.raises(OSError, match="after seal"):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        assert handler_calls == 1
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
        ]

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

        assert handler_calls == 1
        assert recovered.reused is True
        assert recovered.transition is not None
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
            "tool_result",
        ]
    finally:
        guard.release()


def test_concurrent_cold_finalizers_accept_duplicate_terminal_receipt(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    handler_calls = 0
    original_append = journal.append
    fail_result_once = True

    def append(*, request):
        nonlocal fail_result_once
        if request.event_type == "tool_result" and fail_result_once:
            fail_result_once = False
            raise OSError("journal unavailable after seal")
        return original_append(request=request)

    monkeypatch.setattr(journal, "append", append)

    def invoke(_effective_arguments):
        nonlocal handler_calls
        handler_calls += 1
        return DurableToolCompletionDraft(
            result={"ok": True},
            state_transition=DurableToolStateTransitionDraft(
                kind="subagent_snapshot",
                base_state=SubagentState().to_dict(),
                next_state=SubagentState(
                    root_agent_id="root",
                    active_agent_id="root.worker.1",
                    active_lineage=["root", "root.worker.1"],
                ).to_dict(),
            ),
        )

    try:
        with pytest.raises(OSError, match="after seal"):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )

        cold_executors = tuple(
            DurableToolExecutor(
                boundary=executor.boundary,
                artifacts=executor.artifacts,
                execution_guard=guard,
            )
            for _index in range(2)
        )
        finalize_barrier = threading.Barrier(2)
        original_persist = executor.boundary.persist_prepared_result

        def synchronized_persist(*args, **kwargs):
            finalize_barrier.wait(timeout=2)
            return original_persist(*args, **kwargs)

        monkeypatch.setattr(
            executor.boundary,
            "persist_prepared_result",
            synchronized_persist,
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    cold_executor.execute,
                    request=request,
                    guard=guard,
                    invocation=None,
                )
                for cold_executor in cold_executors
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=3))
                except Exception as error:
                    outcomes.append(error)

        successes = [
            outcome for outcome in outcomes if not isinstance(outcome, Exception)
        ]
        failures = [
            outcome for outcome in outcomes if isinstance(outcome, Exception)
        ]
        assert failures == []
        assert len(successes) == 2
        assert all(receipt.reused for receipt in successes)
        assert successes[0].journal_cursor == successes[1].journal_cursor
        assert handler_calls == 1
        assert [event.event_type for event in journal.events].count(
            "tool_result"
        ) == 1
    finally:
        guard.release()


def test_context_runtime_cold_finalize_skips_route_approval_and_handler(
    monkeypatch,
) -> None:
    bundles = {}

    def build(attempt):
        bundle = _bundle(attempt)
        bundles[(attempt.generation.execution_id, attempt.attempt_id)] = bundle
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id="context-v2-cold-finalize-first",
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
    bootstrap.event["execution_guard"] = guard
    runtime.build_harnesses()[0].build_delta(bootstrap)
    bundle = bundles[("execution-1", "attempt-1")]
    calls = 0

    class StatefulPlugin:
        def can_handle(self, *, tool_call, context):
            del tool_call, context
            return True

        def durable_runtime_manifest(self, *, tool_call, context):
            del tool_call, context
            return {
                "handler": "stateful",
                "terminal_handler": True,
                "completion_contract": {
                    "schema": "unchain.tool_completion_contract.v1",
                    "state_transition": "subagent_snapshot.v1",
                    "allowed_state_keys": ["subagent_state"],
                },
            }

        def execute(self, *, tool_call, context):
            del tool_call, context
            nonlocal calls
            calls += 1
            return ToolRuntimeOutcome(
                tool_result={"ok": True},
                state_updates={
                    "subagent_state": SubagentState(
                        root_agent_id="root",
                        active_agent_id="root.worker.1",
                        active_lineage=["root", "root.worker.1"],
                    )
                },
            )

    toolkit = Toolkit()
    toolkit.register(lambda: {"legacy": True}, name="delegate_to_subagent")
    call = ToolCall(
        call_id="call-stateful",
        name="delegate_to_subagent",
        arguments={},
    )
    bundle.durable_event_sink(
        {
            "type": "tool_call",
            "run_id": "attempt-1",
            "iteration": 0,
            "tool_name": call.name,
            "call_id": call.call_id,
            "arguments": {},
        }
    )
    context = HarnessContext(
        state=bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": "attempt-1",
            "execution_guard": guard,
            "toolkit": toolkit,
            "tool_runtime_plugins": [StatefulPlugin()],
            "tool_call": call,
        },
    )
    original_append = bundle.journal.append
    fail_result_once = True

    def append(*, request):
        nonlocal fail_result_once
        if request.event_type == "tool_result" and fail_result_once:
            fail_result_once = False
            raise OSError("journal unavailable after seal")
        return original_append(request=request)

    monkeypatch.setattr(bundle.journal, "append", append)
    try:
        permit = runtime.prepare_tool_execution(context)
        with pytest.raises(OSError, match="after seal"):
            runtime.execute_prepared_tool(context, permit)
        assert calls == 1
        assert [event.event_type for event in bundle.journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
        ]

        cold_runtime = ContextRuntime.from_factory(
            owner_id="context-v2-cold-finalize-second",
            execution_factory=DurableContextRuntimeFactory(
                bundle_builder=lambda attempt: bundle,
                generation_resolver=lambda context, execution_id: "generation-1",
                current_input_resolver=_current_input,
            ),
        )
        cold_bootstrap = _context(
            session_id="execution-1",
            run_id="attempt-1",
            current_input=None,
        )
        cold_bootstrap.event["execution_guard"] = guard
        cold_runtime.build_harnesses()[0].build_delta(cold_bootstrap)
        cold_context = HarnessContext(
            state=cold_bootstrap.state,
            phase="on_tool_call",
            event={
                "run_id": "attempt-1",
                "execution_guard": guard,
                "tool_call": call,
            },
        )
        for forged_call in (
            ToolCall(
                call_id=call.call_id,
                name="forged_tool_name",
                arguments={},
            ),
            ToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments={"forged": True},
            ),
        ):
            forged_context = HarnessContext(
                state=cold_bootstrap.state,
                phase="on_tool_call",
                event={
                    "run_id": "attempt-1",
                    "execution_guard": guard,
                    "tool_call": forged_call,
                },
            )
            with pytest.raises(
                DurableToolExecutorContractError,
                match="host tool call.*journal intent",
            ):
                cold_runtime.prepare_tool_execution(forged_context)

        recovery_permit = cold_runtime.prepare_tool_execution(cold_context)
        recovered = cold_runtime.execute_prepared_tool(
            cold_context,
            recovery_permit,
        )

        assert calls == 1
        assert recovered.reused is True
        assert recovered.transition is not None
        assert [event.event_type for event in bundle.journal.events] == [
            "tool_call",
            "tool.started",
            "tool.subagent_completion.sealed",
            "tool_result",
        ]
    finally:
        guard.release()


def test_context_runtime_latches_transition_cas_conflict_after_tool_result() -> None:
    bundles = {}
    partials = []

    def build(attempt):
        bundle = replace(
            _bundle(attempt),
            partial_attempt_sink=lambda event, error: partials.append(
                (event, error)
            ),
        )
        bundles[(attempt.generation.execution_id, attempt.attempt_id)] = bundle
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id="context-v2-transition-conflict",
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
    bootstrap.state.subagent_state = SubagentState(
        root_agent_id="root",
        active_agent_id="root",
        active_lineage=["root"],
    )
    bootstrap.state.provider_state.provider = "openai"
    bootstrap.event["execution_guard"] = guard
    runtime.build_harnesses()[0].build_delta(bootstrap)
    bundle = bundles[("execution-1", "attempt-1")]

    class ConflictingPlugin:
        def can_handle(self, *, tool_call, context):
            del tool_call, context
            return True

        def durable_runtime_manifest(self, *, tool_call, context):
            del tool_call, context
            return {
                "handler": "conflicting-stateful",
                "terminal_handler": True,
                "completion_contract": {
                    "schema": "unchain.tool_completion_contract.v1",
                    "state_transition": "subagent_snapshot.v1",
                    "allowed_state_keys": ["subagent_state"],
                },
            }

        def execute(self, *, tool_call, context):
            del tool_call, context
            bootstrap.state.subagent_state = SubagentState(
                root_agent_id="root",
                active_agent_id="root.other.1",
                active_lineage=["root", "root.other.1"],
            )
            return ToolRuntimeOutcome(
                tool_result={"ok": True},
                state_updates={
                    "subagent_state": SubagentState(
                        root_agent_id="root",
                        active_agent_id="root.worker.1",
                        active_lineage=["root", "root.worker.1"],
                    )
                },
            )

    toolkit = Toolkit()
    toolkit.register(lambda: {"legacy": True}, name="delegate_to_subagent")
    call = ToolCall(
        call_id="call-transition-conflict",
        name="delegate_to_subagent",
        arguments={},
    )
    bundle.durable_event_sink(
        {
            "type": "tool_call",
            "run_id": "attempt-1",
            "iteration": 0,
            "tool_name": call.name,
            "call_id": call.call_id,
            "arguments": {},
        }
    )
    context = HarnessContext(
        state=bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": "attempt-1",
            "execution_guard": guard,
            "toolkit": toolkit,
            "tool_runtime_plugins": [ConflictingPlugin()],
            "tool_call": call,
        },
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="state transition CAS conflict",
        ) as raised:
            ContextToolAuthorityHarness(runtime=runtime).build_delta(context)

        assert bundle.journal.events[-1].event_type == "tool_result"
        assert len(partials) == 1
        partial_event, partial_error = partials[0]
        assert partial_error is raised.value
        assert partial_event["type"] == "tool.transition.partial"
        assert partial_event["run_id"] == "attempt-1"
        assert partial_event["call_id"] == call.call_id

        with pytest.raises(DurableToolExecutorContractError) as blocked:
            runtime.compile_context(context)
        assert blocked.value is raised.value
    finally:
        guard.release()


def test_guard_loss_after_terminal_journal_returns_nothing_and_restart_reuses(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, intent, arguments = (
        _execution_fixture()
    )
    original = executor.boundary.persist_prepared_result
    calls = 0

    def persist_then_lose_guard(*args, **kwargs):
        receipt = original(*args, **kwargs)
        guard.release_for_wait()
        return receipt

    monkeypatch.setattr(
        executor.boundary,
        "persist_prepared_result",
        persist_then_lose_guard,
    )
    try:
        def invoke(_effective_arguments):
            nonlocal calls
            calls += 1
            return DurableToolCompletionDraft(result={"ok": True})

        with pytest.raises(ExecutionLeaseNotOwnedError):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(executor, request, invoke),
            )
        assert journal.events[-1].event_type == "tool_result"

        guard.reacquire()
        resumed_request = DurableToolExecutionRequest(
            tool_name=request.tool_name,
            call_id=request.call_id,
            iteration=request.iteration,
            subject=_subject_for_intent(
                intent,
                arguments,
                execution_fence=guard.fence,
            ),
        )
        monkeypatch.setattr(
            executor.boundary,
            "persist_prepared_result",
            original,
        )
        recovered = executor.execute(
            request=resumed_request,
            guard=guard,
            invocation=_invocation(
                executor,
                resumed_request,
                lambda _effective_arguments: (_ for _ in ()).throw(
                    AssertionError("restart must not invoke completed tool")
                ),
            ),
        )
        assert calls == 1
        assert recovered.reused is True
    finally:
        guard.release()


def test_completion_sanitizer_cannot_change_protected_semantics(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.artifacts.artifactize_tool_completion

    def forge_completion(completion, *, operation_id):
        persisted = original(completion, operation_id=operation_id)
        forged = dict(persisted.completion)
        forged["should_observe"] = not forged["should_observe"]
        return ToolCompletionArtifactization(
            artifact=persisted.artifact,
            completion=forged,
        )

    monkeypatch.setattr(
        executor.artifacts,
        "artifactize_tool_completion",
        forge_completion,
    )
    try:
        with pytest.raises(
            ArtifactIntegrityError,
            match="completion.*artifact",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_completion_value_must_be_derived_from_verified_artifact_bytes(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.artifacts.artifactize_tool_completion

    def forge_split_brain(completion, *, operation_id):
        forged = dict(completion)
        forged["should_observe"] = not forged["should_observe"]
        persisted = original(forged, operation_id=operation_id)
        return ToolCompletionArtifactization(
            artifact=persisted.artifact,
            completion=completion,
        )

    monkeypatch.setattr(
        executor.artifacts,
        "artifactize_tool_completion",
        forge_split_brain,
    )
    try:
        with pytest.raises(
            ArtifactIntegrityError,
            match="completion.*artifact",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True},
                        should_observe=True,
                    ),
                ),
            )
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_final_receipt_does_not_alias_completion_artifactization(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.artifacts.artifactize_tool_completion
    captured: dict[str, ArtifactRef] = {}

    def capture_completion(completion, *, operation_id):
        persisted = original(completion, operation_id=operation_id)
        captured["artifact"] = persisted.artifact
        return persisted

    monkeypatch.setattr(
        executor.artifacts,
        "artifactize_tool_completion",
        capture_completion,
    )
    try:
        receipt = executor.execute(
            request=request,
            guard=guard,
            invocation=_invocation(
                executor,
                request,
                lambda _effective_arguments: DurableToolCompletionDraft(
                    result={"ok": True}
                ),
            ),
        )
        captured_artifact = captured["artifact"]
        durable_resource_id = journal.events[-1].payload[
            "completion_ref"
        ]["id"]

        assert receipt.completion_artifact is not captured_artifact
        object.__setattr__(
            captured_artifact.ref,
            "resource_id",
            "forged-completion-ref",
        )
        assert receipt.completion_artifact.ref.resource_id == (
            durable_resource_id
        )
    finally:
        guard.release()


def test_result_artifact_visible_value_must_be_derived_from_verified_bytes(
    monkeypatch,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    original = executor.artifacts.artifactize_tool_result

    def forge_visible(result, *, operation_id):
        persisted = original(result, operation_id=operation_id)
        return ToolResultArtifactization(
            artifact=persisted.artifact,
            visible_result={"forged": True},
            result_bytes=persisted.result_bytes,
            result_sha256=persisted.result_sha256,
        )

    monkeypatch.setattr(
        executor.artifacts,
        "artifactize_tool_result",
        forge_visible,
    )
    try:
        with pytest.raises(
            ArtifactIntegrityError,
            match="visible result",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: DurableToolCompletionDraft(
                        result={"ok": True}
                    ),
                ),
            )
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


@pytest.mark.parametrize(
    "raw_outcome",
    [
        {"result": {"ok": True}, "result_messages": [{"role": "tool"}]},
        {"result": {"ok": True}, "message_mutations": [{"op": "delete"}]},
        {"result": {"ok": True}, "callback": lambda event: event},
    ],
)
def test_executor_rejects_untyped_or_message_mutating_outcomes(
    raw_outcome,
) -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )
    try:
        with pytest.raises(
            DurableToolExecutorContractError,
            match="typed completion draft",
        ):
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    lambda _effective_arguments: raw_outcome,
                ),
            )

        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_tool_exception_is_replaced_with_a_secret_safe_typed_failure() -> None:
    executor, request, guard, journal, _order, _intent, _arguments = (
        _execution_fixture()
    )

    def fail_with_secret(_effective_arguments):
        raise RuntimeError("secret-value must not escape")

    try:
        with pytest.raises(DurableToolInvocationFailedError) as raised:
            executor.execute(
                request=request,
                guard=guard,
                invocation=_invocation(
                    executor,
                    request,
                    fail_with_secret,
                ),
            )
        assert "secret-value" not in str(raised.value)
        assert "secret-value" not in repr(raised.value)
        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        assert [event.event_type for event in journal.events] == [
            "tool_call",
            "tool.started",
        ]
    finally:
        guard.release()


def test_context_runtime_owns_executor_delegation_and_failure_latch() -> None:
    bundles = {}

    def build(attempt):
        bundle = _bundle(attempt)
        bundles[(attempt.generation.execution_id, attempt.attempt_id)] = bundle
        return bundle

    runtime = ContextRuntime.from_factory(
        owner_id="context-v2",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, execution_id: "generation-1",
            current_input_resolver=_current_input,
        ),
    )
    guard = _guard()
    bootstrap = _context(session_id="execution-1", run_id="attempt-1")
    bootstrap.event["execution_guard"] = guard
    runtime.build_harnesses()[0].build_delta(bootstrap)
    bundle = bundles[("execution-1", "attempt-1")]
    replacement_bootstrap = HarnessContext(
        state=bootstrap.state,
        phase="bootstrap",
        event={
            "run_id": "attempt-1",
            "execution_guard": _BatchExecutionGuard(
                guard,
                threading.Event(),
            ),
        },
    )
    with pytest.raises(
        ContextExecutionBundleError,
        match="authority regressed or changed",
    ):
        runtime.build_harnesses()[0].build_delta(replacement_bootstrap)
    missing_guard_bootstrap = HarnessContext(
        state=bootstrap.state,
        phase="bootstrap",
        event={"run_id": "attempt-1"},
    )
    with pytest.raises(
        ContextExecutionBundleError,
        match="omitted its execution guard",
    ):
        runtime.build_harnesses()[0].build_delta(missing_guard_bootstrap)
    arguments = {"query": "safe"}
    bundle.durable_event_sink(
        {
            "type": "tool_call",
            "run_id": "attempt-1",
            "iteration": 0,
            "tool_name": "lookup",
            "call_id": "call-1",
            "arguments": arguments,
        }
    )
    toolkit = Toolkit()
    toolkit.register(
        lambda query: {"ok": query == "safe"},
        name="lookup",
    )
    context = HarnessContext(
        state=bootstrap.state,
        phase="on_tool_call",
        event={
            "run_id": "attempt-1",
            "execution_guard": guard,
            "toolkit": toolkit,
            "tool_call": ToolCall(
                call_id="call-1",
                name="lookup",
                arguments=arguments,
            ),
        },
    )
    try:
        permit = runtime.prepare_tool_execution(context)
        receipt = runtime.execute_prepared_tool(context, permit)
        assert receipt.visible_result == {"ok": True}

        bundle.durable_event_sink(
            {
                "type": "tool_call",
                "run_id": "attempt-1",
                "iteration": 0,
                "tool_name": "lookup",
                "call_id": "call-2",
                "arguments": arguments,
            }
        )
        failed_context = HarnessContext(
            state=bootstrap.state,
            phase="on_tool_call",
            event={
                "run_id": "attempt-1",
                "execution_guard": guard,
                "toolkit": toolkit,
                "tool_call": ToolCall(
                    call_id="call-2",
                    name="lookup",
                    arguments=arguments,
                ),
            },
        )
        failed_permit = runtime.prepare_tool_execution(failed_context)
        unrelated_guard = _guard()
        failed_context.event["execution_guard"] = unrelated_guard
        guard.release()
        with pytest.raises(ExecutionLeaseNotOwnedError):
            runtime.execute_prepared_tool(
                failed_context,
                failed_permit,
            )
        with pytest.raises(ExecutionLeaseNotOwnedError):
            runtime.compile_context(failed_context)
        assert [event.event_type for event in bundle.journal.events].count(
            "tool.started"
        ) == 1
    finally:
        if "unrelated_guard" in locals():
            unrelated_guard.release()
        guard.release()
