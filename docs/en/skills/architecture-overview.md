# Architecture Overview

Canonical English skill chapter for the `architecture-overview` topic.

## Role and boundaries

This chapter explains how the package is layered, which modules are foundational, and how execution and data move from user code into the kernel loop and back out.

## Dependency view

- `unchain.tools` stays foundational and dependency-light.
- `unchain.toolkits` depends on `tools` and the input/workspace primitives.
- `unchain.kernel` defines the run loop, runtime hook protocol, delta application, and shared types; it depends on `tools` (for the `Toolkit` it hands to providers) but not on memory or providers directly.
- `unchain.providers` implements `ModelAdapter` against vendor SDKs; the kernel only sees the adapter protocol. `ModelIO` remains a compatibility alias.
- `unchain.memory`, `unchain.optimizers`, `unchain.subagents`, `unchain.retry` are independent hook/runtime layers that plug into the kernel through the hook protocol.
- `unchain.agent` is the orchestration layer: it composes modules, builds a `PreparedAgent`, and delegates to the run loop (`KernelLoop` internally).

## Core objects

- `Agent` as the public orchestration entry point.
- `AgentBuilder` and `PreparedAgent` for the build → run pipeline.
- `RunLoop` as the execution concept. The concrete class is still `KernelLoop`.
- `RuntimeHook` (and `RuntimePhase`) as the extension surface. `RuntimeHarness` remains a compatibility alias.
- `ModelAdapter` as the provider boundary. `ModelIO` remains a compatibility alias.
- `RunDelta` context operations as the only shared state/context mutation path.
- `Tool`, `Toolkit`, `ToolkitRegistry`, `ToolkitCatalogRuntime`, `ToolDiscoveryRuntime` as the tool layer.
- `MemoryManager` and `KernelMemoryRuntime` as the memory layer.

## Execution and state flow

- User code constructs `Agent(name=..., modules=(...))`.
- `Agent.run()` builds an `AgentCallContext`, asks each module to `configure(builder)` on a fresh `AgentBuilder`, calls `builder.build()` to get a `PreparedAgent`, then calls `prepared.run()`.
- `PreparedAgent.run()` enters `KernelLoop.run()`, which dispatches hook phases, fetches one model turn via `ModelAdapter.fetch_turn()`, executes tools, and commits memory until the loop completes or suspends.
- Suspension yields a `KernelRunResult` with a continuation payload that `Agent.resume_human_input()` re-enters on the next call.

## Configuration surface

- Provider/model/api key (on `Agent`).
- Modules: `ToolsModule`, `MemoryModule`, `PoliciesModule`, `OptimizersModule`, `SubagentModule`, `InteractionModule`, `JobsModule`, `CharacterModule`, `ToolDiscoveryModule`, `ToolOptimizerModule`, `SkillsModule`.
- Per-call overrides on `Agent.run()` (`max_iterations`, `payload`, `callback`, `on_tool_confirm`, ...).

## Extension points

- Implement `ModelAdapter` to add providers (under `providers/`).
- Implement `RuntimeHook` to add per-phase behavior (memory commit, optimization, retry, subagents).
- Add builtin or plugin toolkits through `toolkit.toml` manifests.
- Swap memory stores/adapters without changing the orchestration API.

## Common gotchas

- The top-level public API is intentionally tiny: only `Agent`. Everything else lives in subpackages.
- A fresh run loop (`KernelLoop`) is built per `Agent.run()`; module state lives in `AgentState`, not on the loop.
- Provider calls go through `ModelAdapter` implementations under `providers/`; the old provider compatibility layer has been removed.

## Related class references

- [Agents API](../api/agents.md)
- [Runtime API](../api/runtime.md)
- [Tool System API](../api/tools.md)
- [Memory API](../api/memory.md)

## Source entry points

- `src/unchain/__init__.py`
- `src/unchain/agent/`
- `src/unchain/kernel/`
- `src/unchain/tools/`
- `src/unchain/toolkits/`

## Package Layout

```text
src/unchain/
├── __init__.py          # Public API: Agent (lazy)
├── agent/               # Orchestration layer
│   ├── agent.py         #   Agent — user-facing class
│   ├── builder.py       #   AgentCallContext, AgentBuilder, PreparedAgent
│   ├── spec.py          #   AgentSpec (frozen), AgentState
│   ├── model_io.py      #   ModelIOFactoryRegistry compatibility registry
│   └── modules/         #   ToolsModule, MemoryModule, PoliciesModule,
│                        #   OptimizersModule, SubagentModule, InteractionModule,
│                        #   JobsModule, CharacterModule, ToolDiscoveryModule,
│                        #   ToolOptimizerModule
├── kernel/              # Execution engine
│   ├── loop.py          #   KernelLoop — concrete run-loop implementation
│   ├── harness.py       #   RuntimeHook/RuntimeHarness protocol + RuntimePhase + HarnessContext
│   ├── state.py         #   RunState — mutable per-run state
│   ├── types.py         #   ToolCall, TokenUsage, ModelTurnResult, KernelRunResult
│   └── model_io.py      #   ModelAdapter/ModelIO protocol + ModelTurnRequest
├── providers/           # ModelAdapter implementations
│   ├── base.py          #   ModelAdapter/ModelIO protocol + ModelTurnRequest
│   ├── native.py        #   shared native adapter substrate
│   ├── model_io.py      #   legacy compatibility shim
│   ├── openai.py        #   OpenAIModelIO
│   ├── anthropic.py     #   AnthropicModelIO
│   ├── ollama.py        #   OllamaModelIO
│   └── hyperspace.py    #   HyperspaceModelIO
├── tools/               # Tool primitives + discovery
│   ├── tool.py          #   Tool — wrapped callable with metadata
│   ├── toolkit.py       #   Toolkit — dict container of Tools
│   ├── decorators.py    #   @tool decorator
│   ├── models.py        #   ToolParameter, confirmation types, history optimizers
│   ├── registry.py      #   ToolkitRegistry — discovers toolkits from 3 sources
│   ├── catalog.py       #   ToolkitCatalogRuntime — toolkit-level lazy activation
│   ├── discovery.py     #   ToolDiscoveryRuntime — per-tool deferred load
│   ├── execution.py     #   ToolExecutionHook/ToolExecutionHarness — confirm/observe
│   └── prompting.py     #   ToolPromptHook/ToolPromptHarness — tool spec rendering
├── skills/              # SKILL.md registry, catalog/activation harnesses, skill tool
├── toolkits/            # Builtin + MCP toolkits
│   ├── base.py          #   BuiltinToolkit — workspace-safe base
│   ├── mcp.py           #   MCPToolkit — MCP server bridge
│   └── builtin/         #   CoreToolkit, PlanToolkit, AgentReachToolkit
├── memory/              # Two-tier memory
│   ├── manager.py       #   MemoryManager — orchestrates stores + strategies
│   ├── runtime.py       #   KernelMemoryRuntime — kernel-side facade
│   ├── effects.py       #   RunDelta helpers for memory state/context effects
│   ├── short_term.py    #   Short-term recall hook
│   ├── long_term.py     #   LongTermExtractor, profile stores
│   ├── qdrant.py        #   Qdrant vector adapter
│   └── tool_history.py  #   Tool call history compaction
├── optimizers/          # Context window / token compaction harnesses
│   └── ...              #   LastN, LlmSummary, SlidingWindow, ToolHistoryCompaction, ToolPairSafety
├── subagents/           # Sub-agent execution and delegation tools
│   ├── executor.py      #   SubagentExecutor
│   ├── runtime_tools.py #   build_delegate_to_subagent_tool, ...
│   └── plugin.py        #   SubagentToolPlugin
├── retry/               # Provider-agnostic retry layer
│   ├── classifier.py    #   is_retryable
│   ├── backoff.py       #   compute_delay_ms
│   ├── executor.py      #   execute_with_retry
│   └── wrapper.py       #   fetch_turn_with_retry
├── runtime/             # Model resources and default payload loading
│   └── resources/       #   Model capability/default payload JSON
├── input/               # Human input + media
├── character/           # Agent persona / instruction helpers
├── schemas/             # ResponseFormat for structured output
└── types/               # Shared type aliases
```

## Import Hierarchy

The dependency direction flows **downward** — upper layers import from lower layers, never the reverse.

```text
Layer 0  (public API)         unchain                → exports Agent
Layer 1  (orchestration)      unchain.agent          → imports kernel, tools, toolkits, memory, optimizers, subagents
Layer 2  (engine)             unchain.kernel         → imports tools (for Toolkit), defines hooks/state/types
Layer 2' (provider adapters)  unchain.providers      → imports tools, kernel.types
Layer 3  (tool system)        unchain.tools          → no internal unchain deps (foundation)
Layer 3  (toolkit impls)      unchain.toolkits       → imports tools, input, workspace primitives
Layer 3  (memory)             unchain.memory         → imports tools (for tool_history), kernel (for hooks)
Layer 3  (optimizers/...)     unchain.optimizers / .subagents / .retry → import kernel (for hooks), tools
Layer 4  (primitives)         unchain.input, .character, .schemas, .types
```

**Rule**: `unchain.tools` is the foundation — it has zero internal dependencies. The kernel depends only on it. Everything else either implements a kernel protocol (`RuntimeHook`, `ModelAdapter`) or composes modules at the agent layer.

## Data Flow: Request → Response

```text
User code
  │
  ▼
Agent.run(messages, payload, ..., on_tool_confirm, ...)
  │  1. Normalize messages (str → list[dict]).
  │  2. Build AgentCallContext capturing per-call options.
  │  3. _prepare(): each module configures a fresh AgentBuilder.
  │  4. builder.build() → PreparedAgent (KernelLoop + merged Toolkit + hooks).
  │  5. prepared.run() → KernelLoop.run().
  │
  ▼
KernelLoop.run(messages, ...)
  │
  │  while not terminal:
  │    step_once():
  │      ┌─ dispatch_phase("bootstrap")            ─ harness setup
  │      ├─ dispatch_phase("before_model")         ─ optimizers / context prep
  │      ├─ ModelAdapter.fetch_turn(ModelTurnRequest) ─ provider call
  │      ├─ dispatch_phase("after_model")          ─ post-model hooks
  │      ├─ dispatch_phase("on_tool_call")         ─ confirmation gate
  │      ├─ ToolExecutionHook runs tool calls
  │      ├─ dispatch_phase("after_tool_batch")     ─ observation injection
  │      ├─ dispatch_phase("before_commit")        ─ memory commit hook
  │      └─ memory.commit_messages()
  │
  │    on suspension:
  │      dispatch_phase("on_suspend")              ─ state contributions
  │      dispatch_phase("suspend_persist")         ─ reserved durable barrier
  │      return KernelRunResult(status="awaiting_human_input", continuation=...)
  │
  ▼
KernelRunResult
  │  fields: messages, status, continuation, human_input_request,
  │          consumed_tokens, input_tokens, output_tokens, ...
  │
  ▼
User code reads the result; if suspended, calls Agent.resume_human_input().
```

## RuntimePhase reference

The kernel exposes nine ordered extension phases. Two additional phases are
reserved durability barriers owned by the runtime; ordinary harnesses cannot
register them.

| Phase | When | Typical use |
| --- | --- | --- |
| `bootstrap` | Before the first iteration. | Initialize per-run resources, attach state to `RunState`. |
| `before_model` | Before each `ModelAdapter.fetch_turn()`. | Context window pruning, tool history compaction, retry setup. |
| `after_model` | After each model turn returns. | Token accounting, response inspection, custom logging. |
| `on_tool_call` | Before tool execution. | Confirmation gating, argument rewriting, permission checks. |
| `after_tool_batch` | After all tool calls in a turn complete. | Observation turn injection, tool result post-processing. |
| `before_commit` | Before memory commit at end of iteration. | Memory write hooks, summarization triggers. |
| `on_suspend` | When the loop yields control to the caller. | Checkpoint catalog/discovery state, save resume tokens. |
| `on_resume` | When `resume_human_input()` re-enters the loop. | Restore state from continuation payload. |
| `run_finalizing` | After a terminal status is chosen, before durable finalization. | Add final state contributions and artifacts. |

After every `on_suspend`, the reserved `suspend_persist` barrier snapshots all
contributions. After every `run_finalizing`, the reserved `finalize_persist`
barrier performs the final semantic commit or checkpoint transition. This keeps
late, high-order extension hooks from mutating state after it was declared
durable.

## Component Relationships

| Component                  | Depends On                                     | Depended On By                  |
| -------------------------- | ---------------------------------------------- | ------------------------------- |
| `Tool` / `Toolkit`         | — (self-contained)                             | Almost everything               |
| `BuiltinToolkit`           | `Toolkit`, workspace primitives                | Builtin toolkit implementations |
| `ToolkitRegistry`          | `Toolkit`, filesystem                          | Catalog/Discovery runtimes      |
| `ToolkitCatalogRuntime`    | `ToolkitRegistry`, `Toolkit`                   | Agent (via `ToolsModule`)       |
| `ToolDiscoveryRuntime`     | `ToolkitRegistry`, `Toolkit`                   | Agent (via `ToolDiscoveryModule`) |
| `MemoryManager`            | session/vector stores, context strategies      | `KernelMemoryRuntime`           |
| `KernelMemoryRuntime`      | `MemoryManager`, hook protocol                 | KernelLoop (via `MemoryModule`) |
| `ModelAdapter` (protocol)  | provider SDKs (lazy)                           | `KernelLoop`                    |
| `KernelLoop`               | `ModelAdapter`, hooks, `Toolkit`               | `PreparedAgent`                 |
| `PreparedAgent`            | `KernelLoop`, `Toolkit`, defaults              | `Agent`                         |
| `AgentBuilder`             | modules, `KernelLoop`                          | `PreparedAgent`                 |
| `Agent`                    | modules, `AgentBuilder`                        | User code                       |

## Key Design Principles

1. **Minimal public surface** — Only `Agent` is a top-level export. Everything else is imported from subpackages.

2. **Fresh kernel per run** — `Agent.run()` builds a new `KernelLoop` each time. No leftover state between runs unless you keep a `MemoryModule`.

3. **Modular agent assembly** — Agents are composed by passing `modules=(ToolsModule(...), MemoryModule(...), ...)`. Each module gets one shot at the `AgentBuilder` during `_prepare()`.

4. **Tools are data** — A `Tool` is metadata + a callable. Parameter schemas are auto-inferred from Python type hints and docstrings.

5. **Three toolkit discovery sources** — Builtin (shipped with unchain), local (user directories), plugins (entry points). All use the same `toolkit.toml` manifest.

6. **Memory is optional and layered** — Short-term context strategies and long-term vector-backed profiles are independently configurable, both through `MemoryModule`.

7. **Provider-agnostic core** — `KernelLoop` only knows the `ModelAdapter` protocol. Provider-specific projection happens inside each adapter implementation.

## Related Skills

- [creating-builtin-toolkits.md](creating-builtin-toolkits.md) — How to add a new builtin toolkit
- [tool-system-patterns.md](tool-system-patterns.md) — Tool definition and registration patterns
- [memory-system.md](memory-system.md) — Memory tiers and configuration
- [runtime-engine.md](runtime-engine.md) — KernelLoop execution details
- [agent-and-team.md](agent-and-team.md) — Agent and subagent orchestration
- [testing-conventions.md](testing-conventions.md) — Test patterns and eval framework
