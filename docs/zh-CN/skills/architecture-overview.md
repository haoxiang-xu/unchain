# 架构总览

`architecture-overview` 主题的正式简体中文 skills 章节。

## 角色与边界

本章说明包是怎么分层的、哪些模块是地基，以及执行/数据流是如何从用户代码流入 kernel loop 又流回去的。

## 依赖关系

- `unchain.tools` 是地基，依赖最少。
- `unchain.toolkits` 依赖 `tools` 和 input/workspace 原语。
- `unchain.kernel` 定义执行 loop、runtime harness 协议和共享类型；依赖 `tools`（用于 `Toolkit`）但不直接依赖 memory 或 providers。
- `unchain.providers` 用厂商 SDK 实现 `ModelIO`；kernel 只看协议。
- `unchain.memory` / `unchain.optimizers` / `unchain.subagents` / `unchain.retry` 都是独立的 harness/runtime 层，通过 harness 协议挂进 kernel。
- `unchain.agent` 是编排层：组合 modules、构建 `PreparedAgent`、把执行委派给 `KernelLoop`。

## 核心对象

- `Agent` 作为公共编排入口。
- `AgentBuilder` 与 `PreparedAgent` 构成 build → run pipeline。
- `KernelLoop` 作为执行引擎。
- `RuntimeHarness`（与 `RuntimePhase`）作为扩展面。
- `ModelIO` 作为 provider 边界。
- `Tool`、`Toolkit`、`ToolkitRegistry`、`ToolkitCatalogRuntime`、`ToolDiscoveryRuntime` 作为工具层。
- `MemoryManager` 与 `KernelMemoryRuntime` 作为记忆层。

## 执行流与状态流

- 用户代码构造 `Agent(name=..., modules=(...))`。
- `Agent.run()` 构造 `AgentCallContext`，让每个 module 在新的 `AgentBuilder` 上 `configure(builder)`，调 `builder.build()` 拿到 `PreparedAgent`，最后调 `prepared.run()`。
- `PreparedAgent.run()` 进入 `KernelLoop.run()`，循环 dispatch harness 阶段、调一次 `ModelIO.fetch_turn()`、跑工具、commit memory，直到完成或暂停。
- 暂停会返回带 continuation payload 的 `KernelRunResult`，下次 `Agent.resume_human_input()` 重新进入。

## 配置面

- Provider/model/api key（在 `Agent` 上）。
- Modules：`ToolsModule`、`MemoryModule`、`PoliciesModule`、`OptimizersModule`、`SubagentModule`、`InteractionModule`、`JobsModule`、`CharacterModule`、`ToolDiscoveryModule`、`ToolOptimizerModule`、`SkillsModule`。
- `Agent.run()` 的 per-call 覆盖（`max_iterations`、`payload`、`callback`、`on_tool_confirm`、…）。

## 扩展点

- 实现 `ModelIO` 加 provider（在 `providers/` 下）。
- 实现 `RuntimeHarness` 加 phase 行为（memory commit、optimization、retry、subagents）。
- 通过 `toolkit.toml` manifest 加 builtin 或 plugin toolkit。
- 在不改编排 API 的前提下替换 memory store/adapter。

## 常见陷阱

- 顶层公共 API 故意保持极小：只有 `Agent`。其他一切都在子包里。
- 每次 `Agent.run()` 都构造新的 `KernelLoop`；module 状态存在 `AgentState`，不存在 loop 上。
- Provider 调用统一通过 `providers/` 下的 `ModelIO` 实现；旧的 provider 兼容层已经移除。

## 关联 class 参考

- [Agents API](../api/agents.md)
- [Runtime API](../api/runtime.md)
- [Tool System API](../api/tools.md)
- [Memory API](../api/memory.md)

## 源码入口

- `src/unchain/__init__.py`
- `src/unchain/agent/`
- `src/unchain/kernel/`
- `src/unchain/tools/`
- `src/unchain/toolkits/`

## 包结构

```text
src/unchain/
├── __init__.py          # 公共 API：Agent（懒加载）
├── agent/               # 编排层
│   ├── agent.py         #   Agent — 用户面向类
│   ├── builder.py       #   AgentCallContext, AgentBuilder, PreparedAgent
│   ├── spec.py          #   AgentSpec (frozen), AgentState
│   ├── model_io.py      #   ModelIOFactoryRegistry
│   └── modules/         #   ToolsModule, MemoryModule, PoliciesModule,
│                        #   OptimizersModule, SubagentModule, InteractionModule,
│                        #   JobsModule, CharacterModule, ToolDiscoveryModule,
│                        #   ToolOptimizerModule
├── kernel/              # 执行引擎
│   ├── loop.py          #   KernelLoop — step-once 主循环
│   ├── harness.py       #   RuntimeHarness 协议 + RuntimePhase + HarnessContext
│   ├── state.py         #   RunState — 单次 run 的可变状态
│   ├── types.py         #   ToolCall, TokenUsage, ModelTurnResult, KernelRunResult
│   └── model_io.py      #   ModelIO 协议 + ModelTurnRequest
├── providers/           # ModelIO adapter
│   ├── base.py          #   ModelIO 协议 + ModelTurnRequest
│   ├── native.py        #   共享 native adapter 底座
│   ├── model_io.py      #   legacy compatibility shim
│   ├── openai.py        #   OpenAIModelIO
│   ├── anthropic.py     #   AnthropicModelIO
│   ├── ollama.py        #   OllamaModelIO
│   └── hyperspace.py    #   HyperspaceModelIO
├── tools/               # Tool 原语 + 发现
│   ├── tool.py          #   Tool — 带元数据的 callable 包装
│   ├── toolkit.py       #   Toolkit — Tool 字典容器
│   ├── decorators.py    #   @tool 装饰器
│   ├── models.py        #   ToolParameter、确认类型、history optimizer
│   ├── registry.py      #   ToolkitRegistry — 三种来源发现 toolkit
│   ├── catalog.py       #   ToolkitCatalogRuntime — toolkit 级懒激活
│   ├── discovery.py     #   ToolDiscoveryRuntime — 工具级 deferred load
│   ├── execution.py     #   ToolExecutionHarness — 跑工具，处理 confirm/observe
│   └── prompting.py     #   ToolPromptHarness — prompt 端工具 spec 渲染
├── skills/              # SKILL.md registry、catalog/activation harness、skill 工具
├── toolkits/            # Builtin + MCP toolkits
│   ├── base.py          #   BuiltinToolkit — workspace-safe 基类
│   ├── mcp.py           #   MCPToolkit — MCP server bridge
│   └── builtin/         #   CoreToolkit, PlanToolkit, AgentReachToolkit
├── memory/              # 两层记忆
│   ├── manager.py       #   MemoryManager — 调度 store + strategy
│   ├── runtime.py       #   KernelMemoryRuntime — kernel 端 facade
│   ├── short_term.py    #   短期上下文策略
│   ├── long_term.py     #   LongTermExtractor、profile store
│   ├── qdrant.py        #   Qdrant 向量适配
│   └── tool_history.py  #   工具调用历史压缩
├── optimizers/          # 上下文窗口/token 压缩 harness
│   └── ...              #   LastN, LlmSummary, SlidingWindow, ToolHistoryCompaction, ToolPairSafety
├── subagents/           # 子 agent 执行与委派工具
│   ├── executor.py      #   SubagentExecutor
│   ├── runtime_tools.py #   build_delegate_to_subagent_tool, ...
│   └── plugin.py        #   SubagentToolPlugin
├── retry/               # provider 无关的重试层
│   ├── classifier.py    #   is_retryable
│   ├── backoff.py       #   compute_delay_ms
│   ├── executor.py      #   execute_with_retry
│   └── wrapper.py       #   fetch_turn_with_retry
├── runtime/             # Model resource 与默认 payload 加载
│   └── resources/       #   Model capability/default payload JSON
├── input/               # 人类输入 + media
├── character/           # Agent persona / instruction 工具
├── schemas/             # ResponseFormat（结构化输出）
└── types/               # 共享类型别名
```

## 导入层级

依赖方向**向下**流 —— 上层从下层 import，从来不反过来。

```text
Layer 0  (公共 API)        unchain                → 导出 Agent
Layer 1  (编排)            unchain.agent          → 导入 kernel、tools、toolkits、memory、optimizers、subagents
Layer 2  (引擎)            unchain.kernel         → 导入 tools（拿 Toolkit），定义 harness/state/types
Layer 2' (provider 适配)   unchain.providers      → 导入 tools、kernel.types
Layer 3  (工具系统)        unchain.tools          → 无 unchain 内部依赖（地基）
Layer 3  (toolkit 实现)    unchain.toolkits       → 导入 tools、input、workspace 原语
Layer 3  (memory)          unchain.memory         → 导入 tools（tool_history 用）、kernel（harness 用）
Layer 3  (optimizers/...)  unchain.optimizers / .subagents / .retry → 导入 kernel（harness 用）、tools
Layer 4  (原语)            unchain.input、.character、.schemas、.types
```

**规则**：`unchain.tools` 是地基 —— 没有任何 unchain 内部依赖。kernel 只依赖它。其他要么实现 kernel 的协议（harness、ModelIO），要么在 agent 层组合 module。

## 数据流：请求 → 响应

```text
用户代码
  │
  ▼
Agent.run(messages, payload, ..., on_tool_confirm, ...)
  │  1. 规范化 messages（str → list[dict]）。
  │  2. 构造 AgentCallContext，捕获 per-call 选项。
  │  3. _prepare()：每个 module 在新的 AgentBuilder 上配置自己。
  │  4. builder.build() → PreparedAgent（KernelLoop + 合并的 Toolkit + harnesses）。
  │  5. prepared.run() → KernelLoop.run()。
  │
  ▼
KernelLoop.run(messages, ...)
  │
  │  while not terminal:
  │    step_once():
  │      ┌─ dispatch_phase("bootstrap")            ─ harness setup
  │      ├─ dispatch_phase("before_model")         ─ optimizer / 上下文准备
  │      ├─ ModelIO.fetch_turn(ModelTurnRequest)   ─ provider 调用
  │      ├─ dispatch_phase("after_model")          ─ post-model hook
  │      ├─ dispatch_phase("on_tool_call")         ─ confirmation gate
  │      ├─ ToolExecutionHarness 执行 tool calls
  │      ├─ dispatch_phase("after_tool_batch")     ─ observation 注入
  │      ├─ dispatch_phase("before_commit")        ─ memory commit hook
  │      └─ memory.commit_messages()
  │
  │    暂停时：
  │      dispatch_phase("on_suspend")              ─ 汇总状态 contribution
  │      dispatch_phase("suspend_persist")         ─ 保留的持久化 barrier
  │      return KernelRunResult(status="awaiting_human_input", continuation=...)
  │
  ▼
KernelRunResult
  │  字段：messages, status, continuation, human_input_request,
  │        consumed_tokens, input_tokens, output_tokens, ...
  │
  ▼
用户代码读结果；如果暂停则调 Agent.resume_human_input()。
```

## RuntimePhase 速查

kernel 对外提供 9 个有序扩展阶段；另外两个 phase 是 runtime 独占的持久化
barrier，普通 harness 不能注册。

| 阶段 | 时机 | 典型用途 |
| --- | --- | --- |
| `bootstrap` | 第一次迭代前。 | 初始化 per-run 资源，把状态挂到 `RunState`。 |
| `before_model` | 每次 `ModelIO.fetch_turn()` 前。 | 上下文窗口剪枝、tool history 压缩、retry setup。 |
| `after_model` | 每次模型 turn 返回后。 | Token accounting、响应检查、自定义 logging。 |
| `on_tool_call` | 工具执行前。 | 确认 gate、参数改写、权限检查。 |
| `after_tool_batch` | 一轮所有工具调用完成后。 | observation turn 注入、工具结果后处理。 |
| `before_commit` | 迭代末尾 memory commit 前。 | memory 写 hook、摘要触发。 |
| `on_suspend` | loop 把控制权交还给调用方时。 | checkpoint catalog/discovery state，保存 resume token。 |
| `on_resume` | `resume_human_input()` 重新进入 loop 时。 | 从 continuation payload 恢复状态。 |
| `run_finalizing` | 已决定终态、但尚未 durable finalize 时。 | 补充最终状态与 artifact。 |

每次 `on_suspend` 全部执行完后，保留的 `suspend_persist` barrier 才会把所有
contribution 写入 checkpoint；每次 `run_finalizing` 完成后，保留的
`finalize_persist` barrier 才执行最终 semantic commit 或 checkpoint 转换。
因此高 order 的扩展 harness 不会在“已经宣称持久化”之后继续改状态。

## 组件关系

| 组件                       | 依赖                                            | 被谁依赖                          |
| -------------------------- | ----------------------------------------------- | --------------------------------- |
| `Tool` / `Toolkit`         | — （自包含）                                   | 几乎所有                          |
| `BuiltinToolkit`           | `Toolkit`、workspace 原语                       | 内置 toolkit 实现                 |
| `ToolkitRegistry`          | `Toolkit`、文件系统                             | Catalog/Discovery runtime         |
| `ToolkitCatalogRuntime`    | `ToolkitRegistry`、`Toolkit`                    | Agent（通过 `ToolsModule`）       |
| `ToolDiscoveryRuntime`     | `ToolkitRegistry`、`Toolkit`                    | Agent（通过 `ToolDiscoveryModule`）|
| `MemoryManager`            | session/vector store、context strategy          | `KernelMemoryRuntime`             |
| `KernelMemoryRuntime`      | `MemoryManager`、harness 协议                   | KernelLoop（通过 `MemoryModule`） |
| `ModelIO`（协议）          | provider SDK（懒加载）                          | `KernelLoop`                      |
| `KernelLoop`               | `ModelIO`、harnesses、`Toolkit`                 | `PreparedAgent`                   |
| `PreparedAgent`            | `KernelLoop`、`Toolkit`、defaults               | `Agent`                           |
| `AgentBuilder`             | modules、`KernelLoop`                           | `PreparedAgent`                   |
| `Agent`                    | modules、`AgentBuilder`                         | 用户代码                          |

## 关键设计原则

1. **极小的公共表面** —— 顶层只导出 `Agent`。其他都从子包里 import。

2. **每次 run 都新建 kernel** —— `Agent.run()` 每次构造新的 `KernelLoop`。除非保留 `MemoryModule`，否则 run 之间没有残留状态。

3. **Module 化 agent 装配** —— agent 通过 `modules=(ToolsModule(...), MemoryModule(...), ...)` 组合。每个 module 在 `_prepare()` 时拿到 `AgentBuilder` 一次。

4. **Tool 是数据** —— `Tool` 就是元数据 + callable。参数 schema 从 Python type hint 和 docstring 自动推断。

5. **三种 toolkit 发现来源** —— Builtin（unchain 内置）、local（用户目录）、plugins（entry points）。都用同一份 `toolkit.toml` manifest。

6. **Memory 可选且分层** —— 短期上下文策略和长期向量化 profile 各自独立配置，都通过 `MemoryModule`。

7. **Provider 无关的核心** —— `KernelLoop` 只认识 `ModelIO` 协议。provider 特定投影发生在每个 `ModelIO` 实现内部。

## 相关 skills

- [creating-builtin-toolkits.md](creating-builtin-toolkits.md) — 怎么加新 builtin toolkit
- [tool-system-patterns.md](tool-system-patterns.md) — Tool 定义和注册模式
- [memory-system.md](memory-system.md) — 记忆层级与配置
- [runtime-engine.md](runtime-engine.md) — KernelLoop 执行细节
- [agent-and-team.md](agent-and-team.md) — Agent 与子 agent 编排
- [testing-conventions.md](testing-conventions.md) — 测试模式与 eval framework
