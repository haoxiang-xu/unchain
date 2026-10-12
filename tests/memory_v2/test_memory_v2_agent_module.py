from __future__ import annotations

import importlib.util
from dataclasses import replace
from types import SimpleNamespace

import pytest

from unchain.agent.builder import AgentBuilder, AgentCallContext
from unchain.agent.model_io import ModelIOFactoryRegistry
from unchain.memory import (
    MEMORY_CANDIDATE_PROPOSE,
    MEMORY_CONTEXT_READ,
    MEMORY_EXECUTION_COMPLETE,
    MEMORY_V2_CAPABILITIES,
    MEMORY_V2_MODULE_KEY,
    MEMORY_WORKSPACE_READ,
    MemoryAttachment,
    MemoryAttachmentRequest,
    MemoryV2Module,
    MemoryV2ModuleError,
)
from unchain.agent.spec import AgentSpec, AgentState
from unchain.kernel import KernelLoop
from unchain.kernel.harness import HarnessContext
from unchain.kernel.types import KernelRunResult
from unchain.memory.curator import RunCaptureStatus, SourceRunStatus
from unchain.memory.curator.host import MemoryAgentHostAdapter
from unchain.memory.toolkit import MemoryToolkitRunBinding
from unchain.memory.toolkit.policy import (
    MEMORY_LEARNING_POLICY_VERSION,
    MEMORY_PROPOSAL_POLICY_VERSION,
    MemoryLearningPromptHarness,
)
from unchain.runtime import AgentRuntimeContext, ExecutionIdentity, ModuleGrant
from unchain.tools import render_tool_prompt_block

from .test_curator_coordinator import FakeCurationRepository, candidate, completion
from .test_memory_agent_host import (
    _enabled_host,
    _normal_binding,
    _normal_capabilities,
)


def test_official_memory_v2_module_has_a_package_local_import_path():
    assert importlib.util.find_spec("unchain.memory.module") is not None


class _CompletionFactory:
    def __init__(self, value):
        self.value = value
        self.results = []
        self.enqueue_results = []

    def build(self, *, result):
        self.results.append(result)
        return self.value

    def record_enqueue_result(self, *, completion, receipt):
        self.enqueue_results.append((completion, receipt))


class _AttachmentFactory:
    binding_id = "binding-a"

    def __init__(
        self,
        *,
        binding=None,
        completion_factory=None,
    ):
        self.binding = binding or _normal_binding()
        self.completion_factory = completion_factory
        self.requests = []

    def attach(self, request):
        self.requests.append(request)
        return MemoryAttachment(
            binding=self.binding,
            capabilities=_normal_capabilities(),
            completion_factory=self.completion_factory,
        )


_DEFAULT_RUNTIME_CONTEXT = object()


def _grant(
    capabilities=MEMORY_V2_CAPABILITIES,
    *,
    delegable_capabilities=None,
    authority=...,
):
    selected = frozenset(capabilities)
    if delegable_capabilities is None:
        delegable_capabilities = selected.difference({MEMORY_EXECUTION_COMPLETE})
    if authority is ...:
        authority = (
            "completion-authority-a"
            if MEMORY_EXECUTION_COMPLETE in selected
            else None
        )
    return ModuleGrant(
        module_key=MEMORY_V2_MODULE_KEY,
        capabilities=selected,
        delegable_capabilities=frozenset(delegable_capabilities),
        authority=authority,
    )


def _runtime_context(
    *,
    session_id="session-a",
    attempt_id="attempt-a",
    run_id="run-a",
    run_lineage=None,
    grant=None,
):
    return AgentRuntimeContext(
        identity=ExecutionIdentity(
            execution_id=session_id,
            attempt_id=attempt_id,
            run_id=run_id,
            run_lineage=tuple(run_lineage or (run_id,)),
        ),
        module_grants=(() if grant is False else (grant or _grant(),)),
    )


def _builder(
    *,
    session_id="session-a",
    attempt_id="attempt-a",
    run_id="run-a",
    mode="run",
    run_lineage=None,
    grant=None,
    runtime_context=_DEFAULT_RUNTIME_CONTEXT,
):
    if runtime_context is _DEFAULT_RUNTIME_CONTEXT:
        runtime_context = _runtime_context(
            session_id=session_id,
            attempt_id=attempt_id,
            run_id=run_id,
            run_lineage=run_lineage,
            grant=grant,
        )
    return AgentBuilder(
        agent=SimpleNamespace(name="normal-agent"),
        spec=AgentSpec(
            name="normal-agent",
            provider="openai",
            model="gpt-test",
        ),
        state=AgentState(),
        call_context=AgentCallContext(
            mode=mode,
            session_id=session_id,
            execution_owner_id=attempt_id,
            run_id=run_id,
            runtime_context=runtime_context,
        ),
        model_io_registry=ModelIOFactoryRegistry(),
    )


def _result(status="completed"):
    return KernelRunResult(
        messages=[{"role": "assistant", "content": status}],
        status=status,
    )


def test_default_closed_module_does_not_call_factory_or_change_builder():
    repository = FakeCurationRepository((candidate("candidate-a"),))
    factory = _AttachmentFactory(completion_factory=_CompletionFactory(completion()))
    module = MemoryV2Module(
        host=MemoryAgentHostAdapter(repository),
        attachment_factory=factory,
    )
    builder = _builder()

    module.configure(builder)

    assert factory.requests == []
    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []
    assert repository.jobs == {}


def test_default_module_without_a_host_is_a_noop():
    builder = _builder()

    MemoryV2Module().configure(builder)

    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []


def test_enabled_attachment_adds_granted_normal_toolkit_and_completion_hook():
    host, _, _, _ = _enabled_host()
    completion_factory = _CompletionFactory(completion())
    factory = _AttachmentFactory(completion_factory=completion_factory)
    builder = _builder(mode="resume_interaction")

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert tuple(
        name for name in builder.toolkit.tools if name.startswith("memory_")
    ) == (
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_propose",
    )
    assert len(builder.run_hooks) == 1
    assert sum(
        isinstance(harness, MemoryLearningPromptHarness)
        for harness in builder.harnesses
    ) == 1
    runtime_context = builder.call_context.runtime_context
    assert factory.requests == [
        MemoryAttachmentRequest(
            agent_name="normal-agent",
            mode="resume_interaction",
            identity=runtime_context.identity,
            grant=runtime_context.grant_for(MEMORY_V2_MODULE_KEY),
        )
    ]


def test_resume_keeps_stable_execution_lineage_across_a_new_attempt_run():
    host, _, _, _ = _enabled_host()
    factory = _AttachmentFactory(
        binding=MemoryToolkitRunBinding(
            "binding-a",
            "session-a",
            "resume-attempt",
            "resume-run",
        ),
        completion_factory=_CompletionFactory(
            completion(attempt_id="resume-attempt", run_id="resume-run")
        ),
    )
    builder = _builder(
        mode="resume_interaction",
        attempt_id="resume-attempt",
        run_id="resume-run",
        run_lineage=("root-run-a", "resume-run"),
    )

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert len(builder.run_hooks) == 1
    assert factory.requests[0].run_id == "resume-run"
    assert factory.requests[0].root_run_id == "root-run-a"
    assert factory.requests[0].parent_run_id == "root-run-a"
    assert factory.requests[0].grant.allows(MEMORY_EXECUTION_COMPLETE)
    assert factory.requests[0].grant.authority == "completion-authority-a"


def test_descendant_attachment_gets_only_delegable_tools_without_completion():
    host, _, _, _ = _enabled_host()
    child_grant = _grant(
        {
            MEMORY_CONTEXT_READ,
            MEMORY_WORKSPACE_READ,
            MEMORY_CANDIDATE_PROPOSE,
        }
    )
    builder = _builder(
        session_id="child-session",
        attempt_id="child-attempt",
        run_id="child-run",
        run_lineage=("root-run", "child-run"),
        grant=child_grant,
    )
    factory = _AttachmentFactory(
        binding=MemoryToolkitRunBinding(
            "binding-a",
            "child-session",
            "child-attempt",
            "child-run",
        )
    )

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert {name for name in builder.toolkit.tools if name.startswith("memory_")} == {
        "memory_list",
        "memory_search",
        "memory_read",
        "memory_propose",
    }
    assert "memory_upsert" not in builder.toolkit.tools
    assert "memory_promote" not in builder.toolkit.tools
    assert builder.run_hooks == []
    assert factory.requests[0].identity.parent_run_id == "root-run"
    assert factory.requests[0].grant is child_grant
    assert factory.requests[0].grant.authority is None


def test_descendant_attachment_rejects_ungranted_completion_authority():
    host, _, _, _ = _enabled_host()
    completion_factory = _CompletionFactory(completion())
    builder = _builder(
        session_id="child-session",
        attempt_id="child-attempt",
        run_id="child-run",
        run_lineage=("root-run", "child-run"),
        grant=_grant(
            {
                MEMORY_CONTEXT_READ,
                MEMORY_WORKSPACE_READ,
                MEMORY_CANDIDATE_PROPOSE,
            }
        ),
    )

    with pytest.raises(
        MemoryV2ModuleError,
        match="completion authority without a grant",
    ):
        MemoryV2Module(
            host=host,
            attachment_factory=_AttachmentFactory(
                binding=MemoryToolkitRunBinding(
                    "binding-a",
                    "child-session",
                    "child-attempt",
                    "child-run",
                ),
                completion_factory=completion_factory,
            ),
        ).configure(builder)

    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []
    assert completion_factory.results == []


def test_any_descendant_lineage_gets_granted_tools_without_a_completion_hook():
    host, _, _, _ = _enabled_host()
    builder = _builder(
        run_lineage=("root-run", "run-a"),
        grant=_grant({MEMORY_CANDIDATE_PROPOSE}),
    )

    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(),
    ).configure(builder)

    assert "memory_propose" in builder.toolkit.tools
    assert "memory_list" not in builder.toolkit.tools
    assert builder.run_hooks == []
    assert not any(
        isinstance(harness, MemoryLearningPromptHarness)
        for harness in builder.harnesses
    )


def test_enabled_attachment_rejects_missing_runtime_context():
    host, repository, _, _ = _enabled_host()
    builder = _builder(runtime_context=None)

    with pytest.raises(MemoryV2ModuleError, match="AgentRuntimeContext"):
        MemoryV2Module(
            host=host,
            attachment_factory=_AttachmentFactory(
                completion_factory=_CompletionFactory(completion())
            ),
        ).configure(builder)

    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []
    assert repository.jobs == {}


def test_missing_memory_grant_is_a_noop():
    host, repository, _, _ = _enabled_host()
    builder = _builder(grant=False)
    factory = _AttachmentFactory(
        completion_factory=_CompletionFactory(completion())
    )

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert factory.requests == []
    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []
    assert repository.jobs == {}


def test_grant_actually_clips_the_toolkit_surface():
    host, _, _, _ = _enabled_host()
    builder = _builder(grant=_grant({MEMORY_WORKSPACE_READ}))

    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(),
    ).configure(builder)

    assert tuple(builder.toolkit.tools) == (
        "memory_list",
        "memory_search",
        "memory_read",
    )
    assert builder.run_hooks == []


@pytest.mark.parametrize(
    ("mode", "run_lineage"),
    (
        ("run", None),
        ("graph", None),
        ("resume_human_input", None),
        ("resume_interaction", None),
        ("delegate", ("root-run", "run-a")),
    ),
)
def test_proposal_policy_follows_the_grant_across_agent_paths(mode, run_lineage):
    host, _, _, _ = _enabled_host()
    builder = _builder(
        mode=mode,
        run_lineage=run_lineage,
        grant=_grant({MEMORY_CANDIDATE_PROPOSE}),
    )

    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(),
    ).configure(builder)

    assert tuple(builder.toolkit.tools) == ("memory_propose",)
    rendered = render_tool_prompt_block(builder.toolkit)
    assert rendered.count(MEMORY_PROPOSAL_POLICY_VERSION) == 1


def test_read_only_grant_never_receives_the_proposal_policy():
    host, _, _, _ = _enabled_host()
    builder = _builder(grant=_grant({MEMORY_WORKSPACE_READ}))

    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(),
    ).configure(builder)

    assert "memory_propose" not in builder.toolkit.tools
    assert MEMORY_PROPOSAL_POLICY_VERSION not in render_tool_prompt_block(
        builder.toolkit
    )


def test_recipe_tool_filter_removes_the_proposal_policy_with_its_tool():
    host, _, _, _ = _enabled_host()
    builder = _builder(
        grant=_grant({MEMORY_WORKSPACE_READ, MEMORY_CANDIDATE_PROPOSE})
    )
    builder.spec = replace(
        builder.spec,
        allowed_tools=("memory_list", "memory_search", "memory_read"),
    )

    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(),
    ).configure(builder)
    builder._apply_allowed_tools_filter()

    assert "memory_propose" not in builder.toolkit.tools
    assert MEMORY_PROPOSAL_POLICY_VERSION not in render_tool_prompt_block(
        builder.toolkit
    )


def test_root_learning_instruction_tracks_provider_visible_proposal_tool():
    host, _, _, _ = _enabled_host()
    builder = _builder()
    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(
            completion_factory=_CompletionFactory(completion())
        ),
    ).configure(builder)
    (harness,) = [
        item
        for item in builder.harnesses
        if isinstance(item, MemoryLearningPromptHarness)
    ]
    state = KernelLoop().seed_state(
        [{"role": "user", "content": "Remember that I prefer concise reports."}],
        provider="openai",
        model="gpt-test",
    )

    delta = harness.build_delta(
        HarnessContext(
            state=state,
            phase="before_model",
            event={"toolkit": builder.toolkit},
        )
    )
    assert delta is not None
    state.apply_delta(delta)
    learning = [
        message
        for message in state.latest_messages()
        if MEMORY_LEARNING_POLICY_VERSION in str(message.get("content") or "")
    ]
    assert len(learning) == 1
    assert "Before returning the final answer" in learning[0]["content"]
    assert "Never weaken" in learning[0]["content"]

    builder.spec = replace(builder.spec, allowed_tools=("memory_list",))
    builder._apply_allowed_tools_filter()
    removal = harness.build_delta(
        HarnessContext(
            state=state,
            phase="before_model",
            event={"toolkit": builder.toolkit},
        )
    )
    assert removal is not None
    state.apply_delta(removal)
    assert MEMORY_LEARNING_POLICY_VERSION not in str(state.latest_messages())


@pytest.mark.parametrize(
    "capabilities",
    (
        {MEMORY_CANDIDATE_PROPOSE},
        {MEMORY_EXECUTION_COMPLETE},
        {MEMORY_WORKSPACE_READ, MEMORY_EXECUTION_COMPLETE},
    ),
)
def test_root_learning_instruction_requires_proposal_and_completion_authority(
    capabilities,
):
    host, _, _, _ = _enabled_host()
    builder = _builder(grant=_grant(capabilities))
    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(
            completion_factory=(
                _CompletionFactory(completion())
                if MEMORY_EXECUTION_COMPLETE in capabilities
                else None
            )
        ),
    ).configure(builder)

    assert not any(
        isinstance(harness, MemoryLearningPromptHarness)
        for harness in builder.harnesses
    )


def test_completion_capability_requires_explicit_authority():
    identity = _runtime_context().identity

    with pytest.raises(ValueError, match="requires an authority"):
        MemoryAttachmentRequest(
            agent_name="normal-agent",
            mode="run",
            identity=identity,
            grant=_grant(
                {MEMORY_EXECUTION_COMPLETE},
                authority=None,
            ),
        )


def test_submit_interaction_is_a_noop_without_runtime_context():
    host, _, _, _ = _enabled_host()
    factory = _AttachmentFactory(
        completion_factory=_CompletionFactory(completion())
    )
    builder = _builder(mode="submit_interaction", runtime_context=None)

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert factory.requests == []
    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []


def test_suspended_result_is_not_inferred_as_a_root_completion():
    repository = FakeCurationRepository((candidate("candidate-a"),))
    host, _, worker_capabilities, invoker = _enabled_host(repository)
    completion_factory = _CompletionFactory(None)
    builder = _builder()
    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(completion_factory=completion_factory),
    ).configure(builder)
    suspended = _result("awaiting_interaction")

    returned = builder.run_hooks[0](suspended)

    assert returned is None
    assert completion_factory.results == [suspended]
    assert repository.jobs == {}
    assert repository.isolations == []
    assert worker_capabilities.calls == []
    assert invoker.calls == []


def test_no_candidate_root_completion_is_a_noop_without_running_curator_model():
    host, repository, worker_capabilities, invoker = _enabled_host()
    completion_factory = _CompletionFactory(completion())
    builder = _builder()
    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(completion_factory=completion_factory),
    ).configure(builder)
    result = _result()

    returned = builder.run_hooks[0](result)

    assert returned is None
    assert completion_factory.results == [result]
    assert len(completion_factory.enqueue_results) == 1
    recorded_completion, receipt = completion_factory.enqueue_results[0]
    assert recorded_completion == completion()
    assert receipt.enabled is True
    assert receipt.reason == "no_pending_candidates"
    assert repository.jobs == {}
    assert worker_capabilities.calls == []
    assert invoker.calls == []


def test_replayed_root_completion_keeps_stable_identity_and_enqueues_once():
    repository = FakeCurationRepository((candidate("candidate-a"),))
    host, _, worker_capabilities, invoker = _enabled_host(repository)
    completion_factory = _CompletionFactory(completion())
    factory = _AttachmentFactory(completion_factory=completion_factory)
    first = _builder()
    second = _builder()
    module = MemoryV2Module(host=host, attachment_factory=factory)
    module.configure(first)
    module.configure(second)

    first.run_hooks[0](_result())
    second.run_hooks[0](_result())

    assert len(repository.jobs) == 1
    assert [item[1].reason for item in completion_factory.enqueue_results] == [
        "completed_root_run",
        "already_enqueued",
    ]
    assert len(factory.requests) == 2
    assert factory.requests[0] == factory.requests[1]
    assert worker_capabilities.calls == []
    assert invoker.calls == []


@pytest.mark.parametrize(
    ("run_status", "capture_status", "reason"),
    (
        (SourceRunStatus.FAILED, RunCaptureStatus.COMPLETE, "source_run_failed"),
        (
            SourceRunStatus.CANCELLED,
            RunCaptureStatus.COMPLETE,
            "source_run_cancelled",
        ),
        (
            SourceRunStatus.COMPLETED,
            RunCaptureStatus.PARTIAL,
            "source_capture_partial",
        ),
        (
            SourceRunStatus.COMPLETED,
            RunCaptureStatus.UNAVAILABLE,
            "source_capture_unavailable",
        ),
    ),
)
def test_non_eligible_terminal_captures_are_isolated_without_a_worker(
    run_status,
    capture_status,
    reason,
):
    repository = FakeCurationRepository((candidate("candidate-a"),))
    host, _, worker_capabilities, invoker = _enabled_host(repository)
    terminal = completion(
        run_status=run_status,
        capture_status=capture_status,
    )
    builder = _builder()
    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(
            completion_factory=_CompletionFactory(terminal)
        ),
    ).configure(builder)

    builder.run_hooks[0](_result())

    assert repository.jobs == {}
    assert len(repository.isolations) == 1
    assert repository.isolations[0][1] == reason
    assert worker_capabilities.calls == []
    assert invoker.calls == []


def test_attachment_identity_drift_fails_before_tools_or_hooks_are_added():
    host, repository, _, _ = _enabled_host()
    builder = _builder()
    factory = _AttachmentFactory(
        binding=replace(_normal_binding(), run_id="different-run"),
        completion_factory=_CompletionFactory(completion()),
    )

    with pytest.raises(MemoryV2ModuleError, match="run_id"):
        MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []
    assert repository.jobs == {}


def test_completion_identity_drift_fails_without_enqueuing():
    repository = FakeCurationRepository((candidate("candidate-a"),))
    host, _, _, _ = _enabled_host(repository)
    builder = _builder()
    terminal = completion(run_id="different-run")
    MemoryV2Module(
        host=host,
        attachment_factory=_AttachmentFactory(
            completion_factory=_CompletionFactory(terminal)
        ),
    ).configure(builder)

    with pytest.raises(MemoryV2ModuleError, match="run identity"):
        builder.run_hooks[0](_result())

    assert repository.jobs == {}
