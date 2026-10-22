from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from unchain.memory import (
    MEMORY_CANDIDATE_PROPOSE,
    MEMORY_EXECUTION_COMPLETE,
    MEMORY_V2_CAPABILITIES,
    MEMORY_V2_MODULE_KEY,
    MEMORY_WORKSPACE_READ,
    MemoryAttachmentRequest,
    MemoryV2Module,
)
from unchain.journal import ResourceRef
from unchain.kernel import ModelTurnResult
from unchain.kernel.types import ToolCall
from unchain.memory.curator.host import MemoryAgentHostAdapter, MemoryAgentHostConfig
from unchain.memory.toolkit import (
    CandidateProposalRequest,
    MemoryToolkitRunBinding,
)
from unchain.memory.toolkit.policy import MEMORY_LEARNING_POLICY_VERSION
from unchain.persistence.sqlite_memory_host_v2 import (
    SQLiteMemoryAttachmentFactory,
    SQLiteMemoryHostV2Error,
)
from unchain.runtime import ExecutionIdentity, ModuleGrant

from .test_memory_v2_agent_module import _CompletionFactory, _builder, _result
from .test_sqlite_memory_host_v2 import (
    _Clock,
    _ReviewCodec,
    _binding,
    _completion,
    _open_stack,
    _proposal,
)
from .test_memory_toolkit_security import FakeChat


class _Resolver:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.requests = []

    def resolve(self, request):
        self.requests.append(request)
        return self.callback(request)


class _NeverRunModel:
    def run(self, request, *, toolkit, binding):
        del request, toolkit, binding
        raise AssertionError("normal attachment must not run the Memory Agent model")


def _request(
    *,
    agent_name="normal-agent",
    mode="run",
    session_id="session-a",
    attempt_id="attempt-a",
    run_id="run-a",
    run_lineage=None,
    capabilities=MEMORY_V2_CAPABILITIES,
    authority=...,
):
    selected = frozenset(capabilities)
    if authority is ...:
        authority = (
            "completion-authority-a"
            if MEMORY_EXECUTION_COMPLETE in selected
            else None
        )
    return MemoryAttachmentRequest(
        agent_name=agent_name,
        mode=mode,
        identity=ExecutionIdentity(
            execution_id=session_id,
            attempt_id=attempt_id,
            run_id=run_id,
            run_lineage=tuple(run_lineage or (run_id,)),
        ),
        grant=ModuleGrant(
            module_key=MEMORY_V2_MODULE_KEY,
            capabilities=selected,
            delegable_capabilities=selected.difference(
                {MEMORY_EXECUTION_COMPLETE}
            ),
            authority=authority,
        ),
    )


def _attachment_factory(
    repository,
    workspace,
    consolidation_factory,
    *,
    resolver=None,
    long_term=None,
    allowed_long_term_refs=(),
):
    return SQLiteMemoryAttachmentFactory(
        binding_id="binding-chat-a",
        repository=repository,
        workspace=workspace,
        references=consolidation_factory.references,
        context=consolidation_factory.context,
        completion_factory_resolver=resolver,
        proposal_source_refs_resolver=(
            lambda _request: tuple(consolidation_factory.context.source_refs)
        ),
        long_term=long_term,
        allowed_long_term_refs=allowed_long_term_refs,
    )


def _enabled_host(repository, consolidation_factory, clock):
    return MemoryAgentHostAdapter(
        repository,
        capability_factory=consolidation_factory,
        model_invoker=_NeverRunModel(),
        config=MemoryAgentHostConfig(enabled=True),
        clock_ms=clock,
    )


def test_default_closed_module_never_attaches_or_binds_a_candidate_run(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    source_ref = ResourceRef("context_event", "event-a", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    resolver = _Resolver(lambda _request: _CompletionFactory(_completion(_binding())))
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
        resolver=resolver,
    )
    builder = _builder()

    MemoryV2Module(
        host=MemoryAgentHostAdapter(repository),
        attachment_factory=factory,
    ).configure(builder)

    assert resolver.requests == []
    assert builder.toolkit.tools == {}
    assert builder.run_hooks == []
    with sqlite3.connect(tmp_path / "context_v2.sqlite3") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM curation_run_scopes").fetchone()[0]
            == 0
        )


def test_root_attachment_exposes_exact_normal_tools_and_persists_current_run_candidate(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    binding = _binding()
    source_ref = ResourceRef("context_event", "event-a", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    consolidation_factory.context.source_refs.add(source_ref)
    completion_factory = _CompletionFactory(_completion(binding))
    resolver = _Resolver(lambda _request: completion_factory)
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
        resolver=resolver,
    )
    host = _enabled_host(repository, consolidation_factory, clock)
    builder = _builder()
    class _LearningModelIO:
        provider = "openai"
        model = "gpt-test"

        def __init__(self) -> None:
            self.requests = []

        def fetch_turn(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                assert any(
                    MEMORY_LEARNING_POLICY_VERSION
                    in str(message.get("content") or "")
                    for message in request.messages
                )
                call = ToolCall(
                    call_id="memory-proposal-a",
                    name="memory_propose",
                    arguments={
                        "path": "/decisions/context-policy.md",
                        "description": (
                            "Confirmed context policy for long-running agent tasks"
                        ),
                        "content": "Keep the canonical execution journal.",
                    },
                )
                return ModelTurnResult(
                    assistant_messages=[
                        {
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": call.arguments,
                        }
                    ],
                    tool_calls=[call],
                    response_id="memory-proposal-turn",
                )
            if len(self.requests) == 2:
                return ModelTurnResult(
                    assistant_messages=[{"role": "assistant", "content": "done"}],
                    tool_calls=[],
                    final_text="done",
                    response_id="memory-final-turn",
                )
            raise AssertionError("unexpected model turn")

    model_io = _LearningModelIO()
    builder.call_context.input_messages = [
        {
            "role": "user",
            "content": "I prefer that long-running tasks keep a canonical journal.",
        }
    ]
    builder.call_context.max_iterations = 2
    builder.set_model_io(model_io)

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert tuple(
        name for name in builder.toolkit.tools if name.startswith("memory_")
    ) == ("memory_list", "memory_search", "memory_read", "memory_propose")
    assert "memory_upsert" not in builder.toolkit.tools
    assert "memory_promote" not in builder.toolkit.tools
    assert "memory_update_task_state" not in builder.toolkit.tools
    for provider in ("openai", "anthropic", "gemini", "ollama"):
        schema = builder.toolkit.tools["memory_propose"].to_provider_json(provider)
        parameters = schema.get("parameters") or schema.get("input_schema")
        if parameters is None:
            parameters = schema["function"]["parameters"]
        assert "source_refs" not in parameters["properties"]
        assert "kind" not in parameters["properties"]
    assert len(builder.run_hooks) == 1
    assert resolver.requests == [
        _request(
            mode="run",
            session_id=binding.session_id,
            attempt_id=binding.attempt_id,
            run_id=binding.run_id,
        )
    ]

    result = builder.build().run()
    assert result.status == "completed"
    assert len(model_io.requests) == 2
    job = repository.find_job_by_trigger(
        trigger_key=_completion(binding).trigger_key
    )
    assert job is not None
    candidate_ref = job.candidates[0].candidate_ref
    assert (
        repository.read_candidate_content(
            ref=candidate_ref,
            offset=0,
            limit=256,
        ).data
        == b"Keep the canonical execution journal."
    )

    assert completion_factory.results == [result]
    assert job.trigger == _completion(binding)


def test_non_root_resolver_none_adds_tools_without_a_terminal_hook(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    source_ref = ResourceRef("context_event", "event-child", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    resolver = _Resolver(lambda _request: None)
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
        resolver=resolver,
    )
    host = _enabled_host(repository, consolidation_factory, clock)
    builder = _builder(
        session_id="child-session",
        attempt_id="child-attempt",
        run_id="child-run",
        run_lineage=("root-run", "child-run"),
        grant=ModuleGrant(
            module_key=MEMORY_V2_MODULE_KEY,
            capabilities=frozenset(
                {MEMORY_WORKSPACE_READ, MEMORY_CANDIDATE_PROPOSE}
            ),
            delegable_capabilities=frozenset(
                {MEMORY_WORKSPACE_READ, MEMORY_CANDIDATE_PROPOSE}
            ),
        ),
    )

    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    assert tuple(
        name for name in builder.toolkit.tools if name.startswith("memory_")
    ) == ("memory_list", "memory_search", "memory_read", "memory_propose")
    assert builder.run_hooks == []
    assert resolver.requests[0].run_id == "child-run"


def test_graph_step_attachment_persists_root_lineage_for_coordinator_completion(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    source_ref = ResourceRef("context_event", "event-graph-step", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
    )
    graph_attachment = factory.attach(
        _request(
            agent_name="graph-step-agent",
            session_id="session-a",
            attempt_id="graph-step-attempt",
            run_id="graph-step-run",
            run_lineage=("graph-root-run", "graph-step-run"),
            capabilities={MEMORY_CANDIDATE_PROPOSE},
        )
    )
    candidate = graph_attachment.capabilities.candidates.propose(
        request=_proposal(source_ref)
    )
    root_binding = MemoryToolkitRunBinding(
        binding_id="binding-chat-a",
        session_id="session-a",
        attempt_id="graph-root-attempt",
        run_id="graph-root-run",
    )
    factory.attach(
        _request(
            agent_name="graph-root-agent",
            session_id=root_binding.session_id,
            attempt_id=root_binding.attempt_id,
            run_id=root_binding.run_id,
        )
    )

    pending = repository.list_pending_candidates(
        completion=_completion(root_binding),
        limit=20,
    )
    assert [item.candidate_ref for item in pending] == [candidate.candidate_ref]
    assert pending[0].source_agent_run_id == "graph-step-run"
    with sqlite3.connect(tmp_path / "context_v2.sqlite3") as connection:
        rows = connection.execute(
            """
            SELECT attempt_id, run_id, root_run_id
            FROM curation_run_scopes
            ORDER BY attempt_id
            """
        ).fetchall()
    assert rows == [
        ("graph-root-attempt", "graph-root-run", "graph-root-run"),
        ("graph-step-attempt", "graph-step-run", "graph-root-run"),
    ]

    with pytest.raises(ValueError, match="requires an authority"):
        _request(
            agent_name="invalid-completion-agent",
            attempt_id="invalid-attempt",
            run_id="invalid-run",
            capabilities={MEMORY_EXECUTION_COMPLETE},
            authority=None,
        )


def test_suspended_completion_factory_none_never_enqueues(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    source_ref = ResourceRef("context_event", "event-a", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    completion_factory = _CompletionFactory(None)
    resolver = _Resolver(lambda _request: completion_factory)
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
        resolver=resolver,
    )
    host = _enabled_host(repository, consolidation_factory, clock)
    builder = _builder()
    MemoryV2Module(host=host, attachment_factory=factory).configure(builder)

    suspended = _result("awaiting_interaction")
    builder.run_hooks[0](suspended)

    assert completion_factory.results == [suspended]
    assert (
        repository.find_job_by_trigger(trigger_key=_completion(_binding()).trigger_key)
        is None
    )


def test_cold_factory_reuses_run_scope_workspace_and_candidate_receipt(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    binding = _binding()
    source_ref = ResourceRef("context_event", "event-a", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    workspace.write_markdown(
        path="/facts/existing.md",
        description="Existing durable chat memory visible after restart",
        content="persisted workspace content",
        expected_space_revision=1,
        source_refs=(source_ref,),
        operation_id="seed-workspace-before-attachment",
    )
    request = _request(
        session_id=binding.session_id,
        attempt_id=binding.attempt_id,
        run_id=binding.run_id,
        capabilities={MEMORY_WORKSPACE_READ, MEMORY_CANDIDATE_PROPOSE},
    )
    first_factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
    )
    first = first_factory.attach(request)
    proposal = _proposal(source_ref)
    candidate = first.capabilities.candidates.propose(request=proposal)

    reopened_repository, reopened_workspace, reopened_consolidation = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    reopened_factory = _attachment_factory(
        reopened_repository,
        reopened_workspace,
        reopened_consolidation,
    )
    reopened = reopened_factory.attach(request)

    assert reopened.capabilities.candidates.propose(request=proposal) == candidate
    listing = reopened.capabilities.chat.list_entries(
        path="/",
        recursive=True,
        limit=20,
    )
    assert [entry.path for entry in listing["entries"]] == ["/facts/existing.md"]
    entry = listing["entries"][0]
    page = reopened.capabilities.chat.read_content(
        ref=ResourceRef("memory", entry.entry_id, entry.revision, entry.space_id),
        offset=0,
        limit=256,
    )
    assert page.data == b"persisted workspace content"


def test_attachment_scope_and_resolver_drift_fail_closed(tmp_path: Path) -> None:
    clock = _Clock()
    source_ref = ResourceRef("context_event", "event-a", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )
    with pytest.raises(SQLiteMemoryHostV2Error, match="repository_binding"):
        SQLiteMemoryAttachmentFactory(
            binding_id="foreign-binding",
            repository=repository,
            workspace=workspace,
            references=consolidation_factory.references,
            context=consolidation_factory.context,
        )

    resolver = _Resolver(lambda _request: object())
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
        resolver=resolver,
    )
    request = _request()
    with pytest.raises(SQLiteMemoryHostV2Error, match="completion_factory_invalid"):
        factory.attach(request)
    with sqlite3.connect(tmp_path / "context_v2.sqlite3") as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM curation_run_scopes").fetchone()[0]
            == 0
        )


def test_long_term_refs_are_only_injected_when_explicit_and_distinct(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    source_ref = ResourceRef("context_event", "event-a", 1)
    repository, workspace, consolidation_factory = _open_stack(
        tmp_path,
        clock=clock,
        source_ref=source_ref,
    )

    long_term = FakeChat(
        binding_id="binding-chat-a",
        space_id="space-long-term-a",
    )
    allowed = ResourceRef("memory", "preference-a", 1, long_term.space_id)
    factory = _attachment_factory(
        repository,
        workspace,
        consolidation_factory,
        long_term=long_term,
        allowed_long_term_refs=(allowed,),
    )
    attachment = factory.attach(
        _request(capabilities={MEMORY_WORKSPACE_READ})
    )
    assert attachment.capabilities.long_term.space_id == long_term.space_id
    assert attachment.capabilities.allowed_long_term_refs == (allowed,)

    with pytest.raises(SQLiteMemoryHostV2Error, match="long_term_scope"):
        _attachment_factory(
            repository,
            workspace,
            consolidation_factory,
            long_term=type(
                "SameChat",
                (),
                {
                    "binding_id": "binding-chat-a",
                    "space_id": workspace.space.space_id,
                    "space_revision": 1,
                },
            )(),
            allowed_long_term_refs=(
                ResourceRef(
                    "memory",
                    "entry-a",
                    1,
                    workspace.space.space_id,
                ),
            ),
        )
