from __future__ import annotations

import json
from pathlib import Path

import pytest

import unchain.context  # noqa: F401 - initialize the public context/provider graph
from unchain.agent import SkillsModule
from unchain.kernel import HarnessContext, KernelLoop
from unchain.kernel.types import ModelTurnResult, ToolCall
from unchain.memory.checkpoint_state import (
    build_execution_checkpoint,
    merge_checkpoint_transcript_with_incoming,
)
from unchain.optimizers import SlidingWindowOptimizer, SlidingWindowOptimizerConfig
from unchain.providers.wire_preparer import _normalize_openai_messages
from unchain.runtime import build_runtime_loop
from unchain.skills import (
    ACTIVE_SKILLS_SNAPSHOT_VERSION,
    SKILLS_CONTRACT_VERSION,
    SkillsConfig,
)
from unchain.skills.harness import SkillActivationHarness, SkillCatalogHarness
from unchain.skills.rendering import (
    is_active_skills_message,
    is_skill_catalog_message,
    parse_active_skills_block,
)
from unchain.tools import SkillDescriptor, Toolkit


class _FakeBuilder:
    def __init__(self, toolkit: Toolkit | None = None) -> None:
        self.toolkit = toolkit or Toolkit()
        self.harnesses = []

    def add_tool(self, entry):
        self.toolkit.register(entry)

    def add_harness(self, harness):
        self.harnesses.append(harness)


class _FakeModelIO:
    """Returns the scripted results in order and records every request."""

    def __init__(self, *results: ModelTurnResult) -> None:
        self.results = list(results)
        self.requests = []

    def fetch_turn(self, request):
        self.requests.append(request)
        return self.results.pop(0)


def _tool_call_result(name: str, arguments: dict) -> ModelTurnResult:
    return ModelTurnResult(
        assistant_messages=[
            {"type": "function_call", "call_id": "call_1", "name": name, "arguments": json.dumps(arguments)}
        ],
        tool_calls=[ToolCall(call_id="call_1", name=name, arguments=arguments)],
        final_text="",
        response_id="resp_tool",
    )


def _final_result(text: str = "done") -> ModelTurnResult:
    return ModelTurnResult(
        assistant_messages=[{"role": "assistant", "content": text}],
        tool_calls=[],
        final_text=text,
        response_id="resp_final",
    )


def _project(tmp_path: Path, *names: str) -> Path:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    for name in names or ("demo",):
        skill_dir = root / ".unchain" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} skill.\n---\nDo the {name} thing.\n",
            encoding="utf-8",
        )
    return root


def _config(tmp_path: Path, root: Path, **overrides) -> SkillsConfig:
    return SkillsConfig(project_root=root, include_user_dirs=False, home=tmp_path / "home", **overrides)


def _messages_by_kind(messages):
    return (
        [m for m in messages if is_skill_catalog_message(m)],
        [m for m in messages if is_active_skills_message(m)],
    )


# --- module wiring ---------------------------------------------------------------


def test_module_registers_tool_and_two_harnesses(tmp_path):
    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root, tool_name="load_skill")).configure(builder)

    assert "load_skill" in builder.toolkit.tools
    assert builder.toolkit.tools["load_skill"].always_load is True
    assert [type(h) for h in builder.harnesses] == [SkillCatalogHarness, SkillActivationHarness]
    assert [h.order for h in builder.harnesses] == [260, 262]
    assert builder.harnesses[0].tool_name == "load_skill"


def test_module_omits_tool_when_no_model_invocable_skill_exists(tmp_path):
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    quiet = root / ".unchain" / "skills" / "quiet"
    quiet.mkdir(parents=True)
    (quiet / "SKILL.md").write_text(
        "---\nname: quiet\ndescription: user only.\ndisable-model-invocation: true\n---\nbody\n",
        encoding="utf-8",
    )
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)

    assert "skill" not in builder.toolkit.tools
    # User-only skills stay explicitly invocable; no catalog is rendered.
    loop = KernelLoop(harnesses=list(builder.harnesses))
    state = loop.seed_state([{"role": "user", "content": "/quiet go"}], provider="openai", model="gpt-4.1")
    loop.dispatch_phase(state, phase="before_model", event={"toolkit": builder.toolkit})
    catalog, active = _messages_by_kind(state.latest_messages())
    assert catalog == [] and len(active) == 1


def test_module_accepts_dict_and_none_config(tmp_path, monkeypatch):
    monkeypatch.chdir(_project(tmp_path))
    assert SkillsModule().config == SkillsConfig()
    assert SkillsModule({"tool_name": "s"}).config.tool_name == "s"


def test_module_sees_toolkit_skills_added_after_configure(tmp_path):
    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    builder.toolkit.skills = (SkillDescriptor("late", "Added later.", "late body", source_id="tk"),)

    names = [s.name for s in builder.harnesses[0].registry.list().skills]
    assert names == ["demo", "late"]


def test_contract_versions_are_exported():
    assert SKILLS_CONTRACT_VERSION == 1
    assert ACTIVE_SKILLS_SNAPSHOT_VERSION == 1


# --- end to end through the kernel loop ----------------------------------------------


def test_kernel_loop_catalog_then_tool_activation_then_projection(tmp_path):
    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    model_io = _FakeModelIO(_tool_call_result("skill", {"name": "demo"}), _final_result())
    loop = build_runtime_loop(model_io=model_io, harnesses=list(builder.harnesses))
    state = loop.seed_state(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "use the demo skill"}],
        provider="openai",
        model="gpt-4.1",
    )

    loop.step_once(state, toolkit=builder.toolkit)

    first_request = model_io.requests[0].messages
    catalog, active = _messages_by_kind(first_request)
    assert len(catalog) == 1 and active == []
    assert "- `demo`: demo skill." in catalog[0]["content"]

    # The tool ran and the activation was projected in the same iteration.
    catalog, active = _messages_by_kind(state.latest_messages())
    assert len(active) == 1
    (entry,) = parse_active_skills_block(active[0]["content"])
    assert entry.identity.name == "demo" and entry.activation == "tool:call_1"
    assert entry.body == "Do the demo thing."

    loop.step_once(state, toolkit=builder.toolkit)
    second_request = model_io.requests[1].messages
    catalog, active = _messages_by_kind(second_request)
    assert len(catalog) == 1 and len(active) == 1
    assert [m["role"] for m in second_request[:3]] == ["system", "system", "system"]
    assert second_request[0] == {"role": "system", "content": "base"}
    # The tool result is an envelope only; the body lives once, in <active_skills>.
    envelopes = [m for m in second_request if "<skill_loaded" in json.dumps(m, ensure_ascii=False)]
    assert len(envelopes) == 1
    envelope_text = json.dumps(envelopes[0], ensure_ascii=False)
    assert "demo" in envelope_text and 'status=' in envelope_text
    assert "Do the demo thing." not in envelope_text
    assert json.dumps(second_request, ensure_ascii=False).count("Do the demo thing.") == 1


def test_kernel_loop_explicit_user_invocation_reaches_first_request(tmp_path):
    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    model_io = _FakeModelIO(_final_result())
    loop = KernelLoop(model_io=model_io, harnesses=list(builder.harnesses))
    user = {"role": "user", "content": "/demo do it"}
    state = loop.seed_state([user], provider="openai", model="gpt-4.1")

    loop.step_once(state, toolkit=builder.toolkit)

    request = model_io.requests[0].messages
    catalog, active = _messages_by_kind(request)
    assert len(catalog) == 1 and len(active) == 1
    assert request[-1] == user
    (entry,) = parse_active_skills_block(active[0]["content"])
    assert entry.activation.startswith("user:")


# --- durability -------------------------------------------------------------------


def test_blocks_survive_execution_checkpoint_and_resume_merge(tmp_path):
    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    loop = KernelLoop(harnesses=list(builder.harnesses))
    state = loop.seed_state(
        [{"role": "system", "content": "agent instructions"}, {"role": "user", "content": "/demo"}],
        provider="openai",
        model="gpt-4.1",
        session_id="session-skills",
    )
    loop.dispatch_phase(state, phase="before_model", event={"toolkit": builder.toolkit})

    checkpoint = build_execution_checkpoint(state, status="awaiting_human_input", run_id="run-1")
    persisted = checkpoint["transcript"]
    catalog, active = _messages_by_kind(persisted)
    # The catalog is stateless (re-rendered each step, like <tools>); only the
    # activation snapshot is persisted.
    assert catalog == [] and len(active) == 1

    # Fresh process: the host resends only the leading agent instruction plus new input.
    restored = merge_checkpoint_transcript_with_incoming(
        persisted,
        [{"role": "system", "content": "agent instructions"}, {"role": "user", "content": "continue"}],
    )
    catalog, active = _messages_by_kind(restored)
    assert catalog == [] and len(active) == 1
    assert restored[-1] == {"role": "user", "content": "continue"}

    # Re-running the harnesses on the restored transcript changes nothing and
    # re-reads no file (the source is deleted first).
    (root / ".unchain" / "skills" / "demo" / "SKILL.md").unlink()
    resumed = KernelLoop(harnesses=list(builder.harnesses)).seed_state(
        restored, provider="openai", model="gpt-4.1", session_id="session-skills"
    )
    activation_harness = builder.harnesses[1]
    assert activation_harness.build_delta(HarnessContext(state=resumed, phase="before_model", event={})) is None
    (entry,) = parse_active_skills_block(_messages_by_kind(resumed.latest_messages())[1][0]["content"])
    assert entry.body == "Do the demo thing."


def test_sliding_window_compaction_keeps_both_blocks(tmp_path):
    from unchain.kernel.delta import HarnessDelta

    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    loop = KernelLoop(
        harnesses=[
            *builder.harnesses,
            SlidingWindowOptimizer(SlidingWindowOptimizerConfig(max_window_tokens=400)),
        ]
    )
    state = loop.seed_state([{"role": "user", "content": "/demo start"}], provider="openai", model="gpt-4.1")
    loop.dispatch_phase(state, phase="before_model", event={"toolkit": builder.toolkit})
    state.apply_delta(
        HarnessDelta.append(
            created_by="test",
            messages=[
                {"role": "assistant", "content": "A" * 800},
                {"role": "user", "content": "B" * 800},
                {"role": "assistant", "content": "C" * 800},
                {"role": "user", "content": "recent"},
            ],
        )
    )

    loop.dispatch_phase(state, phase="before_model", event={"toolkit": builder.toolkit})

    messages = state.latest_messages()
    catalog, active = _messages_by_kind(messages)
    assert len(catalog) == 1 and len(active) == 1
    assert "A" * 800 not in json.dumps(messages)
    assert messages[-1] == {"role": "user", "content": "recent"}


def test_context_v2_compile_keeps_both_blocks_as_system_prefix(tmp_path):
    """Context V2 copies every source system message (compiler.py `_is_system`),
    so both blocks are outside the turn-compaction path by construction."""

    from unchain.context import ContextCompileRequest, ContextCompiler, resolve_context_budget
    from unchain.context.runtime import ContextRuntime

    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    window = 16_384

    def request_factory(context):
        return ContextCompileRequest(
            case="runtime-path",
            source_messages=tuple(context.latest_messages()),
            current_generation="generation-1",
            fixed_overhead_tokens=0,
            budget=resolve_context_budget(context_window_tokens=window),
            provider="openai",
            model="gpt-test",
            build_id=f"build-{context.state.iteration}",
            execution_id="execution-1",
            generation_id="generation-1",
            attempt_id="attempt-1",
        )

    runtime = ContextRuntime._for_test(
        owner_id="context-v2",
        compiler=ContextCompiler(),
        request_factory=request_factory,
        durable_event_sink=lambda event: None,
        partial_attempt_sink=lambda event, error: None,
    )
    loop = KernelLoop(harnesses=list(builder.harnesses))
    state = loop.seed_state(
        [{"role": "system", "content": "base"}, {"role": "user", "content": "/demo start"}],
        provider="openai",
        model="gpt-test",
        session_id="execution-1",
        max_context_window_tokens=window,
    )
    loop.dispatch_phase(state, phase="before_model", event={"toolkit": builder.toolkit})

    result = runtime.compile_context(HarnessContext(state=state, phase="before_model", event={"run_id": "attempt-1"}))
    compiled = result.to_dict()["messages"]
    catalog, active = _messages_by_kind(compiled)
    assert len(catalog) == 1 and len(active) == 1
    assert compiled[0] == {"role": "system", "content": "base"}
    assert [m["role"] for m in compiled[:3]] == ["system", "system", "system"]
    assert compiled[-1] == {"role": "user", "content": "/demo start"}


# --- provider wire ------------------------------------------------------------------


def test_blocks_are_plain_text_system_messages_on_provider_wire(tmp_path):
    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    loop = KernelLoop(harnesses=list(builder.harnesses))
    state = loop.seed_state([{"role": "user", "content": "/demo go"}], provider="openai", model="gpt-4.1")
    loop.dispatch_phase(state, phase="before_model", event={"toolkit": builder.toolkit})
    messages = state.latest_messages()
    for message in messages:
        assert set(message) == {"role", "content"}

    openai_wire = _normalize_openai_messages(messages)
    assert [m["role"] for m in openai_wire[:2]] == ["system", "system"]
    assert openai_wire[0]["content"].startswith("<available_skills>")
    assert openai_wire[1]["content"].startswith("<active_skills>")

    from unchain.providers.native import _translate_content_blocks_for_anthropic

    anthropic_messages = [dict(m) for m in messages if m["role"] != "system"]
    _translate_content_blocks_for_anthropic(anthropic_messages)
    assert anthropic_messages == [{"role": "user", "content": "/demo go"}]


def test_tool_envelope_reports_already_active_and_superseded(tmp_path):
    from unchain.kernel.delta import HarnessDelta

    root = _project(tmp_path)
    builder = _FakeBuilder()
    SkillsModule(_config(tmp_path, root)).configure(builder)
    tool_obj = builder.toolkit.tools["skill"]
    loop = KernelLoop(harnesses=list(builder.harnesses))
    state = loop.seed_state([{"role": "user", "content": "hi"}], provider="openai", model="gpt-4.1")

    first = tool_obj.func(name="demo")
    assert 'status="activated"' in first
    loop.dispatch_phase(state, phase="after_tool_batch", event={"toolkit": builder.toolkit, "tool_calls": []})
    assert len(_messages_by_kind(state.latest_messages())[1]) == 1

    second = tool_obj.func(name="demo")
    assert 'status="already_active"' in second
    loop.dispatch_phase(state, phase="after_tool_batch", event={"toolkit": builder.toolkit, "tool_calls": []})
    (entry,) = parse_active_skills_block(_messages_by_kind(state.latest_messages())[1][0]["content"])

    (root / ".unchain" / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo skill.\n---\nDo the NEW demo thing.\n", encoding="utf-8"
    )
    third = tool_obj.func(name="demo")
    assert 'status="superseded"' in third
    loop.dispatch_phase(state, phase="after_tool_batch", event={"toolkit": builder.toolkit, "tool_calls": []})
    active = _messages_by_kind(state.latest_messages())[1]
    assert len(active) == 1
    (updated,) = parse_active_skills_block(active[0]["content"])
    assert updated.body == "Do the NEW demo thing." and updated.revision != entry.revision
    assert tool_obj.func(name="Nope") == 'Error: invalid skill name "Nope"'
    assert tool_obj.func(name="missing") == 'Error: unknown skill "missing"'
