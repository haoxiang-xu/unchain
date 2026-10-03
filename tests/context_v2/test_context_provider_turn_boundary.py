from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from unchain.agent import Agent, AgentCallContext, ToolsModule
from unchain.agent.modules import ContextModule
from unchain.context.artifacts import ArtifactService
from unchain.context.coordinator import ContextCompileCoordinator
from unchain.context.factory import (
    ContextExecutionBundle,
    ContextExecutionBundleError,
    DurableContextRuntimeFactory,
)
from unchain.context.handoff import DurableHandoffRecorder, HandoffService
from unchain.context.ingress import ContextInputIngress, HostResolvedCurrentInput
from unchain.context.projector import CanonicalSemanticEventProjector
from unchain.context.provider_execution import ContextProviderTurnExecutionService
from unchain.context.request_factory import JournalContextRequestFactory
from unchain.context.runtime import ContextRuntime
from unchain.context.tool_boundary import DurableToolBoundary
from unchain.execution import ExecutionRuntime
from unchain.journal import DurableEventSink
from unchain.kernel.loop import KernelLoop
from unchain.kernel.model_tool_boundary import FinalModelToolBoundary
from unchain.memory import InMemorySessionStore
from unchain.persistence import SQLiteContextV2Store
from unchain.providers import OpenAIModelIO
from unchain.providers.durable_turn_runtime import DurableProviderTurnMode
from unchain.subagents import SubagentExecutor, SubagentPolicy, SubagentTemplate
from unchain.subagents.plugin import SubagentToolPlugin
from unchain.tools import Tool, Toolkit


TARGET_SHA256 = "c" * 64


class _CheckpointRepository:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id

    def prepare(self, **kwargs):
        raise AssertionError(kwargs)

    def commit(self, **kwargs):
        raise AssertionError(kwargs)

    def get_by_operation(self, **_kwargs):
        return None


class _BuildRepository:
    def __init__(self, execution_id: str) -> None:
        self.execution_id = execution_id
        self._operations = {}
        self._triggers = {}

    def record(self, *, envelope, operation, trigger_cursor):
        from unchain.context.ports import ContextBuildReceipt

        receipt = ContextBuildReceipt(
            envelope=envelope,
            operation=operation,
            trigger_cursor=trigger_cursor,
        )
        self._operations[operation.operation_id] = receipt
        self._triggers[trigger_cursor] = receipt
        return receipt

    def get_by_operation(self, *, operation):
        return self._operations.get(operation.operation_id)

    def get_by_trigger(self, *, trigger_cursor):
        return self._triggers.get(trigger_cursor)


def _current_input(context, attempt):
    users = [
        message
        for message in context.latest_messages()
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    if not users:
        return None
    return HostResolvedCurrentInput(
        attempt=attempt,
        content=str(users[-1].get("content") or ""),
    )


def _bundle_builder(tmp_path, *, with_provider_service: bool):
    store = SQLiteContextV2Store(
        database_path=tmp_path / "memory_v2" / "context_v2.sqlite3",
        object_directory=tmp_path / "memory_v2" / "objects",
    )
    bundles = {}

    def build(attempt):
        repository = store.bind_execution(attempt.generation.execution_id)
        artifacts = ArtifactService(
            repository,
            sanitizer=lambda content, media_type: content,
        )
        projector = CanonicalSemanticEventProjector(
            attempt=attempt,
            artifacts=artifacts,
            payload_sanitizer=lambda event_type, payload: payload,
        )
        sink = DurableEventSink(repository, attempt, projector)
        handoffs = HandoffService(artifacts)
        bundle = ContextExecutionBundle(
            attempt=attempt,
            journal=repository,
            projector=projector,
            durable_event_sink=sink,
            coordinator=ContextCompileCoordinator(
                journal=repository,
                checkpoint_repository=_CheckpointRepository(
                    attempt.generation.execution_id
                ),
                build_repository=_BuildRepository(attempt.generation.execution_id),
                partial_attempt_sink=lambda request, error: None,
                provider_turn_result_reader=repository.read_full_verified,
            ),
            artifacts=artifacts,
            handoffs=handoffs,
            ingress=ContextInputIngress(
                attempt=attempt,
                projector=projector,
                sink=sink,
            ),
            request_factory=JournalContextRequestFactory(
                attempt=attempt,
                journal=repository,
                model_window_fallback=lambda provider, model: 16_384,
            ),
            tool_boundary=DurableToolBoundary(
                attempt=attempt,
                projector=projector,
                sink=sink,
            ),
            handoff_recorder=DurableHandoffRecorder(
                attempt=attempt,
                handoffs=handoffs,
                projector=projector,
                sink=sink,
            ),
            partial_attempt_sink=lambda event, error: None,
            provider_turn_service=(
                ContextProviderTurnExecutionService(
                    attempt=attempt,
                    store=repository,
                    mode=DurableProviderTurnMode.ENFORCE_TEST,
                    transport_target_sha256=TARGET_SHA256,
                    sleep=lambda _seconds: None,
                )
                if with_provider_service
                else None
            ),
        )
        bundles[attempt.attempt_id] = bundle
        return bundle

    return build, bundles


def _runtime(
    tmp_path,
    *,
    enabled: bool,
    with_provider_service: bool,
    tool_output_management_active: bool = False,
):
    build, bundles = _bundle_builder(
        tmp_path,
        with_provider_service=with_provider_service,
    )
    runtime = ContextRuntime.from_factory(
        owner_id="context-v2-provider-turn",
        execution_factory=DurableContextRuntimeFactory(
            bundle_builder=build,
            generation_resolver=lambda context, execution_id: (
                f"generation-{execution_id}"
            ),
            current_input_resolver=_current_input,
        ),
        provider_turns_enabled=enabled,
        tool_output_management_active=tool_output_management_active,
    )
    return runtime, bundles


def _model_io(send_calls: list[dict]):
    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id="response-context-boundary",
                    output=[
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "boundary result"}
                            ],
                        }
                    ],
                    usage={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                ),
            )

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream()

    class _Client:
        responses = _Responses()

    model_io = OpenAIModelIO(
        model="gpt-boundary",
        api_key="test-key",
        client_factory=lambda **_kwargs: _Client(),
        default_payloads={},
        model_capabilities={},
    )
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    return model_io


def _tool_then_text_model_io(
    send_calls: list[dict],
    *,
    tool_name: str,
    call_id: str,
):
    class _Stream:
        def __init__(self, send_number: int) -> None:
            self._send_number = send_number

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            output = (
                [
                    {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": '{"query":"durable"}',
                    }
                ]
                if self._send_number == 1
                else [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "tool complete"}],
                    }
                ]
            )
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id=f"response-tool-boundary-{self._send_number}",
                    output=output,
                    usage={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                ),
            )

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _Stream(len(send_calls))

    class _Client:
        responses = _Responses()

    model_io = OpenAIModelIO(
        model="gpt-boundary",
        api_key="test-key",
        client_factory=lambda **_kwargs: _Client(),
        default_payloads={},
        model_capabilities={},
    )
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    return model_io


def test_default_factory_runtime_does_not_install_the_provider_boundary(tmp_path):
    runtime, _bundles = _runtime(
        tmp_path,
        enabled=False,
        with_provider_service=False,
    )

    assert not any(
        type(component) is FinalModelToolBoundary
        for component in runtime.build_harnesses()
    )


def test_enabled_runtime_rejects_a_bundle_without_provider_service(tmp_path):
    runtime, _bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=False,
    )
    loop = KernelLoop(harnesses=list(runtime.build_harnesses()))
    state = loop.seed_state(
        [{"role": "user", "content": "bind"}],
        provider="openai",
        model="gpt-boundary",
        session_id="execution-boundary-missing",
    )

    with pytest.raises(ContextExecutionBundleError, match="provider turn service"):
        loop._dispatch_bootstrap(
            state,
            payload={},
            toolkit=Toolkit(),
            callback=None,
            verbose=False,
            response_format=None,
            run_id="attempt-boundary-missing",
            resume_mode=False,
        )


def test_enabled_runtime_owns_the_final_kernel_provider_send(tmp_path):
    runtime, bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=True,
    )
    send_calls: list[dict] = []
    model_io = _model_io(send_calls)
    loop = KernelLoop(
        model_io=model_io,
        harnesses=list(runtime.build_harnesses()),
    )

    result = loop.run(
        messages=[{"role": "user", "content": "use durable provider"}],
        callback=runtime.compose_event_callback(None),
        session_id="execution-boundary",
        provider="openai",
        model="gpt-boundary",
        toolkit=Toolkit(),
        run_id="attempt-boundary",
        max_iterations=1,
    )

    assert result.messages[-1]["content"] == "boundary result"
    assert len(send_calls) == 1
    assert type(loop.final_model_tool_boundary) is FinalModelToolBoundary
    event_types = [
        event.event_type
        for event in bundles["attempt-boundary"].journal.capture_snapshot().events
    ]
    assert "provider.wire_snapshot" in event_types
    assert "provider.turn_result" in event_types
    receipts = bundles["attempt-boundary"].run_bundle_ledger.load_receipts(
        root_run_id="attempt-boundary",
        owner_run_id="attempt-boundary",
        attempt_id="attempt-boundary",
    )
    assert len(receipts) == 1
    assert receipts[0].status == "completed"
    assert receipts[0].timing.started_at is not None
    assert receipts[0].timing.completed_at is not None
    assert result.run_bundle is not None
    assert result.run_bundle["aggregation"]["all_call_ids"] == [
        receipts[0].provider_call_id
    ]


def test_graph_agent_mode_uses_the_same_durable_provider_boundary(tmp_path):
    runtime, bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=True,
    )
    send_calls: list[dict] = []
    agent = Agent(
        name="graph-provider-agent",
        provider="openai",
        model="gpt-boundary",
        modules=(ContextModule(runtime=runtime),),
        model_io_factory=lambda spec, context: _model_io(send_calls),
    )
    prepared = agent._prepare(
        AgentCallContext(
            mode="graph",
            input_messages=[{"role": "user", "content": "graph durable"}],
            session_id="execution-boundary-graph",
            run_id="attempt-boundary-graph",
            max_iterations=1,
        )
    )

    result = prepared.run()

    assert result.messages[-1]["content"] == "boundary result"
    assert len(send_calls) == 1
    assert "attempt-boundary-graph" in bundles


def test_graph_tool_turn_projects_artifact_only_before_second_provider_turn(tmp_path):
    runtime, bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=True,
        tool_output_management_active=True,
    )
    send_calls: list[dict] = []
    raw_result_marker = "GRAPH_RAW_TOOL_RESULT_MUST_NOT_REACH_SECOND_PROVIDER_TURN"

    def graph_probe(query: str = "") -> dict[str, str]:
        return {"query": query, "status": raw_result_marker}

    agent = Agent(
        name="graph-output-agent",
        provider="openai",
        model="gpt-boundary",
        modules=(
            ContextModule(runtime=runtime),
            ToolsModule(
                tools=(
                    Tool.from_callable(
                        graph_probe,
                        name="graph_probe",
                        output_policy="artifact_only",
                    ),
                )
            ),
        ),
        model_io_factory=lambda _spec, _context: _tool_then_text_model_io(
            send_calls,
            tool_name="graph_probe",
            call_id="call-graph-output",
        ),
    )

    execution_runtime = ExecutionRuntime(InMemorySessionStore())
    with execution_runtime.scope("execution-output-graph") as guard:
        result = agent._prepare(
            AgentCallContext(
                mode="graph",
                input_messages=[{"role": "user", "content": "graph tool output"}],
                session_id="execution-output-graph",
                run_id="attempt-output-graph",
                max_iterations=2,
                execution_guard=guard,
            )
        ).run()

    assert result.status == "completed"
    assert len(send_calls) == 2
    assert raw_result_marker not in repr(send_calls[1])
    assert "artifact_only" in repr(send_calls[1])
    assert "context.tool_output_snapshot" in [
        event.event_type
        for event in bundles["attempt-output-graph"].journal.capture_snapshot().events
    ]


def test_subagent_fork_uses_a_distinct_durable_attempt_boundary(tmp_path):
    runtime, bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=True,
    )
    send_calls: list[dict] = []
    parent = Agent(
        name="parent-provider-agent",
        provider="openai",
        model="gpt-boundary",
        modules=(ContextModule(runtime=runtime),),
        model_io_factory=lambda spec, context: _model_io(send_calls),
    )
    child = parent.fork_for_subagent(
        subagent_name="child-provider-agent",
        mode="delegate",
        parent_name="parent-provider-agent",
        lineage=["parent-provider-agent", "child-provider-agent"],
        task="inspect durable state",
        instructions="",
        expected_output="short answer",
        disabled_module_keys=("memory",),
    )

    result = child.run(
        "inspect",
        session_id="execution-boundary-child",
        run_id="attempt-boundary-child",
        max_iterations=1,
    )

    assert result.messages[-1]["content"] == "boundary result"
    assert len(send_calls) == 1
    assert "attempt-boundary-child" in bundles


def test_subagent_tool_turn_projects_artifact_only_before_second_provider_turn(
    tmp_path,
):
    runtime, bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=True,
        tool_output_management_active=True,
    )
    send_calls: list[dict] = []
    raw_result_marker = "SUBAGENT_RAW_TOOL_RESULT_MUST_NOT_REACH_SECOND_PROVIDER_TURN"

    def child_probe(query: str = "") -> dict[str, str]:
        return {"query": query, "status": raw_result_marker}

    parent = Agent(
        name="parent-output-agent",
        provider="openai",
        model="gpt-boundary",
        modules=(
            ContextModule(runtime=runtime),
            ToolsModule(
                tools=(
                    Tool.from_callable(
                        child_probe,
                        name="child_probe",
                        output_policy="head_tail",
                    ),
                )
            ),
        ),
    )
    template_agent = Agent(
        name="specialist-output-agent",
        provider="openai",
        model="gpt-boundary",
        modules=(
            ToolsModule(
                tools=(
                    Tool.from_callable(
                        child_probe,
                        name="child_probe",
                        output_policy="artifact_only",
                    ),
                )
            ),
        ),
        model_io_factory=lambda _spec, _context: _tool_then_text_model_io(
            send_calls,
            tool_name="child_probe",
            call_id="call-child-output",
        ),
    )
    template = SubagentTemplate(
        name="specialist",
        description="Template-owned tool output policy",
        agent=template_agent,
    )
    plugin = SubagentToolPlugin(
        parent_agent=parent,
        templates=(template,),
        policy=SubagentPolicy(),
        executor=SubagentExecutor(),
    )
    child, _, _ = plugin._build_subagent(
        template=template,
        child_id="parent-output-agent.specialist.1",
        mode="delegate",
        target="specialist",
        lineage=["parent-output-agent", "parent-output-agent.specialist.1"],
        task="inspect output",
        instructions="",
        expected_output="short answer",
    )

    execution_runtime = ExecutionRuntime(InMemorySessionStore())
    with execution_runtime.scope("execution-output-child") as guard:
        result = child.run(
            "inspect",
            session_id="execution-output-child",
            run_id="attempt-output-child",
            max_iterations=2,
            _execution_guard=guard,
        )

    assert result.status == "completed"
    assert len(send_calls) == 2
    assert raw_result_marker not in repr(send_calls[1])
    assert "artifact_only" in repr(send_calls[1])
    snapshots = [
        event
        for event in bundles["attempt-output-child"].journal.capture_snapshot().events
        if event.event_type == "context.tool_output_snapshot"
    ]
    assert len(snapshots) == 1
    assert snapshots[0].payload["runtime_config"]["tool_output_policy_map"] == {
        "schema": "unchain.tool_output_policy_map.v1",
        "policies": {"child_probe": "artifact_only"},
    }


def test_tool_bearing_turn_persists_result_and_continues_through_same_boundary(
    tmp_path,
):
    runtime, bundles = _runtime(
        tmp_path,
        enabled=True,
        with_provider_service=True,
        tool_output_management_active=True,
    )
    send_calls: list[dict] = []
    tool_calls: list[str] = []

    class _ToolThenTextStream:
        def __init__(self, send_number: int) -> None:
            self._send_number = send_number

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __iter__(self):
            output = (
                [
                    {
                        "type": "function_call",
                        "call_id": "call-provider-probe",
                        "name": "probe",
                        "arguments": '{"query":"durable"}',
                    }
                ]
                if self._send_number == 1
                else [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "tool complete"}],
                    }
                ]
            )
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    id=f"response-tool-boundary-{self._send_number}",
                    output=output,
                    usage={
                        "input_tokens": 3,
                        "output_tokens": 2,
                        "total_tokens": 5,
                    },
                ),
            )

    class _Responses:
        def create(self, **kwargs):
            send_calls.append(copy.deepcopy(kwargs))
            return _ToolThenTextStream(len(send_calls))

    class _Client:
        responses = _Responses()

    model_io = OpenAIModelIO(
        model="gpt-boundary",
        api_key="test-key",
        client_factory=lambda **_kwargs: _Client(),
        default_payloads={},
        model_capabilities={},
    )
    model_io.fetch_turn = lambda request: (_ for _ in ()).throw(
        AssertionError("legacy provider path was called")
    )
    toolkit = Toolkit()

    raw_result_marker = "RAW_TOOL_RESULT_MUST_NOT_REACH_SECOND_PROVIDER_TURN"

    def probe(query: str = "") -> dict[str, str]:
        tool_calls.append(query)
        return {"query": query, "status": raw_result_marker}

    toolkit.register(probe, name="probe", output_policy="artifact_only")
    loop = KernelLoop(
        model_io=model_io,
        harnesses=list(runtime.build_harnesses()),
        execution_runtime=ExecutionRuntime(InMemorySessionStore()),
    )
    emitted: list[dict] = []

    result = loop.run(
        messages=[{"role": "user", "content": "call the probe"}],
        callback=runtime.compose_event_callback(emitted.append),
        session_id="execution-boundary-tool",
        provider="openai",
        model="gpt-boundary",
        toolkit=toolkit,
        run_id="attempt-boundary-tool",
        max_iterations=2,
    )

    assert result.messages[-1]["content"] == "tool complete"
    assert tool_calls == ["durable"]
    assert len(send_calls) == 2
    event_types = [
        event.event_type
        for event in bundles["attempt-boundary-tool"].journal.capture_snapshot().events
    ]
    assert event_types.count("provider.wire_snapshot") == 2
    assert event_types.count("provider.turn_result") == 2
    assert event_types.count("tool_call") == 1
    assert event_types.count("tool_result") == 1
    tool_results = [
        event
        for event in emitted
        if event.get("type") == "tool_result"
        and event.get("call_id") == "call-provider-probe"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["run_id"] == "attempt-boundary-tool"
    assert tool_results[0]["tool_name"] == "probe"
    assert isinstance(tool_results[0].get("result"), dict)
    assert raw_result_marker not in repr(send_calls[1])
    assert "artifact_only" in repr(send_calls[1])
