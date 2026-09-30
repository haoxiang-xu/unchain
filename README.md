<div align="center">
  <img src="./assets/unchain.png" alt="unchain" style="height: 100px; margin-bottom: 32px;" />
  <h1>unchain</h1>
  <p>A lightweight Python agent framework for building tool-using AI runtimes</p>
</div>

[![Version](https://img.shields.io/badge/version-0.2.0-2563eb)](pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-0ea5e9)](LICENSE)

Docs: [English](docs/README.en.md) | [中文](docs/README.zh-CN.md)

---

## What is unchain?

unchain is a modular agent framework that separates **agent configuration** from **execution**. It provides a harness-driven kernel loop, pluggable provider support, multi-tier memory, and a rich tool system — all composable through a clean module API.

Everything lives under one package: `src/unchain/`. Agent behaviour is composed by passing modules (`ToolsModule`, `MemoryModule`, `PoliciesModule`, `OptimizersModule`, `SubagentModule`, `ToolDiscoveryModule`) into `Agent`.

## Install

```bash
pip install -e ".[dev]"
```

**Prerequisites**: Python 3.12+

## Quick Start

```python
from unchain import Agent
from unchain.agent import ToolsModule
from unchain.toolkits import CoreToolkit

agent = Agent(
    name="coder",
    provider="openai",
    model="gpt-5",
    instructions="You are a coding assistant.",
    modules=(
        ToolsModule(tools=(CoreToolkit(workspace_root="."),)),
    ),
)

result = agent.run("Inspect the repo and explain what matters.")
print(result.messages[-1]["content"])
```

### Custom Tools

```python
from unchain import Agent
from unchain.agent import ToolsModule
from unchain.tools import tool

@tool
def fetch_price(ticker: str) -> dict:
    """Fetch the latest stock price for a given ticker symbol."""
    return {"ticker": ticker, "price": 142.50}

agent = Agent(
    name="analyst",
    provider="anthropic",
    model="claude-sonnet-4",
    modules=(ToolsModule(tools=(fetch_price,)),),
)
```

### Subagents (multi-agent work)

`Team` is no longer a top-level primitive. Compose multi-agent flows by giving a planner agent a `SubagentModule` so it can spawn typed children:

```python
from unchain import Agent
from unchain.agent import SubagentModule
from unchain.subagents import SubagentPolicy, SubagentTemplate

planner = Agent(
    name="planner",
    provider="openai",
    model="gpt-5",
    instructions="Plan the work, then delegate.",
    modules=(
        SubagentModule(
            templates=(
                SubagentTemplate(name="researcher", instructions="Research the question."),
                SubagentTemplate(name="writer", instructions="Write a report from the research."),
            ),
            policy=SubagentPolicy(max_depth=4, max_total_agents=8),
        ),
    ),
)

result = planner.run("Write a report on recent AI trends.")
```

The planner gets `delegate_to_subagent`, `handoff_to_subagent`, and `spawn_worker_batch` tools; see [docs/en/skills/agent-and-team.md](docs/en/skills/agent-and-team.md) for the full surface.

## Architecture

```
Agent.run(messages)
  |
  v
AgentBuilder --> PreparedAgent --> KernelLoop.run()
                                      |
                    +------ step_once() loop ------+
                    |                              |
                    v                              v
            dispatch_phase()              fetch_model_turn()
            (harnesses run)               (provider SDK call)
                    |                              |
                    v                              v
          Memory / Optimizers             Tool Execution
          Context Injection               Confirmation Gates
                    |                     Human Input Suspend
                    v
             HarnessDelta applied to RunState
                    |
                    v
              KernelRunResult
```

### Kernel Phases

Harnesses plug into these runtime phases:

| Phase | When | Example |
|-------|------|---------|
| `bootstrap` | Run initialization | Load memory |
| `before_model` | Before each LLM call | Inject context, optimize window |
| `after_model` | After LLM response | Process response |
| `on_tool_call` | Per tool invocation | Confirmation gates |
| `after_tool_batch` | All tools in batch done | Commit observations |
| `before_commit` | Before memory persist | Finalize state |
| `on_suspend` / `on_resume` | Pause/resume | Human input flow |

### Delta-Based State

All state mutations flow through immutable `HarnessDelta` objects:

```python
HarnessDelta.append(
    created_by="my_harness",
    messages=[{"role": "assistant", "content": "..."}],
    state_updates={"my_key": value},
)
```

## Providers

| Provider | Models | Streaming | Tools | Reasoning |
|----------|--------|-----------|-------|-----------|
| OpenAI | gpt-5, gpt-4.1, gpt-5-codex | Yes | Yes | Yes (gpt-5) |
| Anthropic | claude-sonnet-4, claude-opus-4-6, claude-haiku-3.5 | Yes | Yes | Yes (thinking) |
| Google Gemini | gemini-2.5-pro, gemini-2.5-flash | Yes | Yes | Yes |
| Ollama | Any local model | Yes | Yes | Model-dependent |
| SAP Hyperspace | hyperspace--claude-opus-4-6/4-7, hyperspace--claude-sonnet-4-6, hyperspace--claude-haiku-4-5 | Yes | Yes | Yes (thinking) |

Configure via environment variables:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
```

## Built-in Toolkits

| Toolkit | Description |
|---------|-------------|
| **CoreToolkit** | File read/write, directory listing, text search, line editing, shell execution, LSP, web fetch, structured user questions |
| **PlanToolkit** | Workspace-backed structured planning with draft, update, read, list, and finalize tools |
| **AgentReachToolkit** | Read-only Agent Reach helpers for status checks, web pages, and YouTube metadata |
| **MCPToolkit** | Bridge MCP (Model Context Protocol) servers as native toolkits |

Toolkits are also discoverable from local directories and entry-point plugins.

## Skills

Drop a `SKILL.md` (or flat `<name>.md`) file into `.unchain/skills/<name>/` or `.agents/skills/<name>/`, add `SkillsModule()` to the agent's modules, and it's discovered automatically — no registration code needed. `SkillsModule` renders a name+description catalog, exposes a `skill` tool the model can call to activate one by name, and lets the user activate one directly with `/name` in their own message. Activations are projected into a durable `<active_skills>` system block that survives compaction, resume, and cold restart.

See [Agent Skills](docs/en/skills/agent-skills.md) for the full frontmatter contract, discovery roots, and wire format.

## Memory System

Two-tier memory with pluggable backends:

```python
from unchain.memory import MemoryConfig, MemoryManager

config = MemoryConfig(
    last_n_turns=8,
    vector_top_k=4,
    long_term_enabled=True,
)
```

| Tier | Storage | Purpose |
|------|---------|---------|
| **Short-term** | In-memory / JSON sessions | Recent conversation context |
| **Long-term** | Qdrant vector DB | Semantic recall, fact extraction, profiles |

Strategies: `LastNTurns`, `SummaryToken`, `Hybrid`

## Modules

Agents are composed through pluggable modules:

```python
from unchain.agent import Agent, ToolsModule, MemoryModule, PoliciesModule
from unchain.toolkits import CoreToolkit

core_toolkit = CoreToolkit(workspace_root=".")

agent = Agent(
    name="assistant",
    provider="openai",
    model="gpt-5",
    modules=(
        ToolsModule(tools=(core_toolkit,)),
        MemoryModule(memory=memory_manager),
        PoliciesModule(max_iterations=32),
    ),
)
```

| Module | Purpose |
|--------|---------|
| `ToolsModule` | Register tools and toolkits |
| `MemoryModule` | Attach memory manager |
| `PoliciesModule` | Set iteration limits, subagent policies |
| `OptimizersModule` | Context window optimizers |
| `SubagentModule` | Register subagent templates |

## Subagents

Agents can delegate, handoff, or spawn parallel workers:

```python
from unchain.agent import SubagentModule
from unchain.subagents import SubagentTemplate

templates = [
    SubagentTemplate(name="researcher", agent=research_agent, allowed_modes=("delegate",)),
    SubagentTemplate(name="coder", agent=code_agent, allowed_modes=("worker",), parallel_safe=True),
]

parent = Agent(
    name="coordinator",
    modules=(SubagentModule(templates=templates),),
    ...
)
```

## Context Optimizers

Pluggable strategies for managing the context window:

| Optimizer | Strategy |
|-----------|----------|
| `LastNOptimizer` | Keep last N messages |
| `LlmSummaryOptimizer` | Summarize older messages with LLM |
| `ToolHistoryCompactionOptimizer` | Compress verbose tool results |
| `PinnedContextOptimizer` | Preserve pinned file context |
| `SlidingWindowOptimizer` | Token-aware context window truncation |

## Package Layout

```
src/
  unchain/
    __init__.py               # Public API: Agent
    agent/                    # Agent, AgentBuilder, PreparedAgent, modules
    kernel/                   # KernelLoop, RuntimeHarness, RuntimePhase, types
    providers/                # OpenAI, Anthropic, Ollama ModelIO
    tools/                    # Tool, Toolkit, registry, catalog, discovery, execution
    toolkits/                 # CoreToolkit, PlanToolkit, AgentReachToolkit, MCPToolkit
    memory/                   # MemoryManager, KernelMemoryRuntime, harnesses
    optimizers/               # Context window / token compaction harnesses
    subagents/                # Subagent executor + delegation tools
    retry/                    # Provider-agnostic retry layer
    schemas/                  # ResponseFormat (structured output)
    input/                    # Human input + media
    character/                # Persona / instruction helpers
    runtime/                  # Legacy Broth runtime kept only for LegacyBrothModelIO
    types/                    # Shared type aliases
tests/
  evals/                      # Scenario-based evaluation framework
docs/
  en/                         # English docs (skills chapters + API reference)
  zh-CN/                      # Chinese docs
```

## Public API

```python
# Top-level
from unchain import Agent

# Agent + modules
from unchain.agent import (
    AgentBuilder, PreparedAgent,
    ToolsModule, MemoryModule, PoliciesModule,
    OptimizersModule, SubagentModule, ToolDiscoveryModule,
)

# Kernel
from unchain.kernel import KernelLoop, KernelRunResult, RunState, RuntimeHarness, RuntimePhase

# Tools
from unchain.tools import (
    Tool, Toolkit, tool, ToolParameter, ToolkitRegistry,
    ToolkitCatalogRuntime, ToolkitCatalogConfig,
    ToolDiscoveryRuntime, ToolDiscoveryConfig,
)

# Providers
from unchain.providers import OpenAIModelIO, AnthropicModelIO, OllamaModelIO

# Toolkits
from unchain.toolkits import AgentReachToolkit, CoreToolkit, MCPToolkit, PlanToolkit

# Memory
from unchain.memory import MemoryConfig, MemoryManager, KernelMemoryRuntime

# Subagents
from unchain.subagents import SubagentPolicy, SubagentTemplate, SubagentExecutor

# Schemas
from unchain.schemas import ResponseFormat
```

## Testing

```bash
# Setup
./scripts/init_python312_venv.sh

# Run tests
./run_tests.sh

# Or directly
PYTHONPATH=src pytest tests/ -q
```

## Documentation

Full bilingual docs with skills chapters and API reference:

**Skills Chapters**: Architecture Overview, Agent & Subagents, Runtime Engine, Tool System Patterns, Memory System, Creating Toolkits, Testing Conventions

**API Reference**: Agents, Runtime, Interaction, Durable Jobs, Tools, Toolkits, Memory, Input/Workspace/Schemas

**Appendices**: Class Index (55+ classes), Export Index, Glossary, Return Shapes

See [docs/README.en.md](docs/README.en.md) for the full reading guide.

## License

[Apache 2.0](LICENSE)
