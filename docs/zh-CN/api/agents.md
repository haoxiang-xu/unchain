# Agents API 参考

通过 `Agent` 类、`AgentBuilder` 管线、不可变 `AgentSpec`/`AgentState`，以及可插拔的 `AgentModule` 系统（tools、memory、policies、optimizers、subagents）实现模块化 Agent 组合。

| 指标 | 值 |
| --- | --- |
| 类数量 | 3 |
| Dataclass | 5 |
| 协议 | 1 |
| Agent 模块 | 6 |

## 覆盖地图

| 类 | 源码 | 导出 | 类型 |
| --- | --- | --- | --- |
| `AgentSpec` | `src/unchain/agent/spec.py` | subpackage | dataclass (frozen) |
| `AgentState` | `src/unchain/agent/spec.py` | subpackage | dataclass |
| `Agent` | `src/unchain/agent/agent.py` | top-level | class |
| `CompletionEvaluation` | `src/unchain/runtime/completion.py` | subpackage | dataclass (frozen) |
| `CompletionPolicy` | `src/unchain/runtime/completion.py` | subpackage | dataclass (frozen) |
| `AgentCallContext` | `src/unchain/agent/builder.py` | subpackage | dataclass |
| `PreparedAgent` | `src/unchain/agent/builder.py` | subpackage | dataclass |
| `AgentBuilder` | `src/unchain/agent/builder.py` | subpackage | dataclass |
| `AgentModule` | `src/unchain/agent/modules/base.py` | subpackage | protocol |
| `BaseAgentModule` | `src/unchain/agent/modules/base.py` | subpackage | dataclass (frozen) |
| `ToolsModule` | `src/unchain/agent/modules/tools.py` | subpackage | dataclass (frozen) |
| `MemoryModule` | `src/unchain/agent/modules/memory.py` | subpackage | dataclass (frozen) |
| `PoliciesModule` | `src/unchain/agent/modules/policies.py` | subpackage | dataclass (frozen) |
| `OptimizersModule` | `src/unchain/agent/modules/optimizers.py` | subpackage | dataclass (frozen) |
| `SubagentModule` | `src/unchain/agent/modules/subagents.py` | subpackage | dataclass (frozen) |
| `InteractionModule` | `src/unchain/agent/modules/interaction.py` | subpackage | dataclass (frozen) |

### `src/unchain/agent/spec.py`

不可变 Agent 规格与可变 Agent 状态。

## AgentSpec

Frozen dataclass，持有 Agent 实例的不可变配置。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/spec.py` |
| 继承/协议 | `-` |
| 导出状态 | 通过 `unchain.agent` 导出。 |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `str` | 构造时必需。 |
| `instructions` | `str` | 默认值：`""`。 |
| `provider` | `str` | 默认值：`"openai"`。 |
| `model` | `str` | 默认值：`"gpt-5"`。 |
| `api_key` | `str \| None` | 默认值：`None`。 |
| `modules` | `tuple[Any, ...]` | 默认值：`()`。 |
| `allowed_tools` | `tuple[str, ...] \| None` | 默认值：`None`。 |

## AgentState

可变 dataclass，用于每个 Agent 的运行时状态。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/spec.py` |
| 继承/协议 | `-` |
| 导出状态 | 通过 `unchain.agent` 导出。 |
| 对象类型 | Dataclass。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `module_state` | `dict[str, Any]` | 默认值：`field(default_factory=dict)`。 |

### `src/unchain/agent/agent.py`

顶层 `Agent` 门面：持有配置、规范化消息、通过 `AgentBuilder` 准备 kernel loop，并暴露 `run`/`resume_human_input`/`clone`/`fork_for_subagent`/`as_tool` 入口。

## Agent

面向用户的 Agent 类，组合模块、构建 `PreparedAgent`，并将执行委托给 `KernelLoop`。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/agent.py` |
| 继承/协议 | `-` |
| 导出状态 | 通过 `unchain` 顶层导出。 |
| 对象类型 | 类；公开。 |

### 构造表面

- `__init__(self, *, name: str, instructions: str='', provider: str='openai', model: str='gpt-5', api_key: str | None=None, modules: tuple[Any, ...]=(), allowed_tools: tuple[str, ...] | None=None, model_io_factory: Callable[..., Any] | None=None)`

### 属性

- `@property name` -> `str`
- `@property instructions` -> `str`
- `@property provider` -> `str`
- `@property model` -> `str`
- `@property allowed_tools` -> `tuple[str, ...] | None`

### 公共方法

#### `__init__(self, *, name: str, instructions: str='', provider: str='openai', model: str='gpt-5', api_key: str | None=None, modules: tuple[Any, ...]=(), allowed_tools: tuple[str, ...] | None=None, model_io_factory: Callable[..., Any] | None=None)`

初始化 Agent，创建 `AgentSpec`、`AgentState` 和 `ModelIOFactoryRegistry`。

- 类型：构造函数
- 错误：如果 `name` 为空或非字符串则抛出 `ValueError`。

#### `run(self, messages: str | list[dict[str, Any]], *, payload: dict[str, Any] | None=None, response_format: Any=None, callback: Callable[[dict[str, Any]], None] | None=None, verbose: bool=False, max_iterations: int | None=None, max_context_window_tokens: int | None=None, previous_response_id: str | None=None, on_tool_confirm: Callable[..., Any] | None=None, on_human_input: Callable[..., Any] | None=None, on_max_iterations: Callable[..., Any] | None=None, session_id: str | None=None, memory_namespace: str | None=None, run_id: str | None=None, tool_runtime_config: dict[str, Any] | None=None) -> KernelRunResult`

主入口。规范化消息、通过 `AgentBuilder` 准备 `PreparedAgent`，然后运行 kernel loop。

- 类型：方法
- 返回：`KernelRunResult`

#### `resume_human_input(self, *, conversation: list[dict[str, Any]], continuation: dict[str, Any], response: dict[str, Any] | Any, payload: dict[str, Any] | None=None, response_format: Any=None, callback: Callable[[dict[str, Any]], None] | None=None, verbose: bool=False, on_tool_confirm: Callable[..., Any] | None=None, on_human_input: Callable[..., Any] | None=None, on_max_iterations: Callable[..., Any] | None=None, session_id: str | None=None, memory_namespace: str | None=None, run_id: str | None=None, tool_runtime_config: dict[str, Any] | None=None) -> KernelRunResult`

在收集到人类输入后恢复暂停的运行。

- 类型：方法
- 返回：`KernelRunResult`

#### `clone(self, *, name: str | None=None, instructions: str | None=None, modules: tuple[Any, ...] | None=None, model: str | None=None, allowed_tools: tuple[str, ...] | None=None) -> Agent`

创建一个选择性覆盖字段的新 `Agent`。

- 类型：方法
- 返回：新的 `Agent` 实例。

#### `fork_for_subagent(self, *, subagent_name: str, mode: str, parent_name: str, lineage: list[str], task: str, instructions: str, expected_output: str, memory_policy: str, model: str | None=None, allowed_tools: tuple[str, ...] | None=None) -> Agent`

为子代理执行创建子 Agent。叠加委托指令，当 `memory_policy` 为 `"ephemeral"` 时可选地剥离 `MemoryModule`。

- 类型：方法
- 返回：新的 `Agent` 实例。

#### `as_tool(self, *, name: str | None=None, description: str | None=None, max_iterations: int | None=None) -> Tool`

将该 Agent 包装为可调用的 `Tool`，委托到 `self.run()`。

- 类型：方法
- 返回：`Tool`

### 生命周期与运行时角色

- 构造阶段校验 identity，构建 `AgentSpec` 和 `AgentState`。
- `run()` 规范化消息，创建 `AgentCallContext`，调用 `_prepare()` 让每个模块的 `configure()` 作用于 `AgentBuilder`，然后调用 `builder.build()` 获得 `PreparedAgent`，最后调用 `prepared.run()`。
- 模块组合行为：`ToolsModule` 注册工具，`MemoryModule` 挂载 memory，`PoliciesModule` 设置默认值，`OptimizersModule` 添加 harness，`SubagentModule` 添加委托工具。

### 最小调用示例

```python
from unchain import Agent
from unchain.agent import ToolsModule, MemoryModule

agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant.",
    modules=(
        ToolsModule(tools=(my_tool,)),
        MemoryModule(memory=my_memory_config),
    ),
)
result = agent.run("Hello!")
```

### `src/unchain/agent/builder.py`

Agent 准备管线：`AgentCallContext` 捕获调用端选项，`AgentBuilder` 累积模块贡献，`PreparedAgent` 持有最终组装好的 kernel loop。

## AgentCallContext

Dataclass，捕获传给 `Agent.run()` 或 `Agent.resume_human_input()` 的每次调用选项。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/builder.py` |
| 继承/协议 | `-` |
| 导出状态 | 通过 `unchain.agent` 导出。 |
| 对象类型 | Dataclass。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `mode` | `str` | `"run"` 或 `"resume_human_input"`。 |
| `input_messages` | `list[dict[str, Any]] \| None` | 默认值：`None`。 |
| `conversation` | `list[dict[str, Any]] \| None` | 默认值：`None`。 |
| `continuation` | `dict[str, Any] \| None` | 默认值：`None`。 |
| `response` | `dict[str, Any] \| Any` | 默认值：`None`。 |
| `payload` | `dict[str, Any] \| None` | 默认值：`None`。 |
| `response_format` | `ResponseFormat \| None` | 默认值：`None`。 |
| `callback` | `Callable[[dict[str, Any]], None] \| None` | 默认值：`None`。 |
| `verbose` | `bool` | 默认值：`False`。 |
| `max_iterations` | `int \| None` | 默认值：`None`。 |
| `max_context_window_tokens` | `int \| None` | 默认值：`None`。 |
| `previous_response_id` | `str \| None` | 默认值：`None`。 |
| `on_tool_confirm` | `Callable[..., Any] \| None` | 默认值：`None`。 |
| `on_human_input` | `Callable[..., Any] \| None` | 默认值：`None`。 |
| `on_max_iterations` | `Callable[..., Any] \| None` | 默认值：`None`。 |
| `session_id` | `str \| None` | 默认值：`None`。 |
| `memory_namespace` | `str \| None` | 默认值：`None`。 |
| `run_id` | `str \| None` | 默认值：`None`。 |
| `tool_runtime_config` | `dict[str, Any] \| None` | 默认值：`None`。 |

## AgentBuilder

可变 builder，模块通过它注册 tools、harness、memory、默认值和 hook。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/builder.py` |
| 继承/协议 | `-` |
| 导出状态 | 通过 `unchain.agent` 导出。 |
| 对象类型 | Dataclass。 |

### 公共方法

| 方法 | 说明 |
| --- | --- |
| `add_tool(entry)` | 注册 `Tool`、`Toolkit` 或 callable。 |
| `add_harness(harness)` | 挂载运行时 harness。 |
| `attach_memory_runtime(memory_runtime)` | 挂载 `KernelMemoryRuntime`。 |
| `set_model_io(model_io)` | 覆盖 `ModelIO` 实例。 |
| `set_model_io_factory(factory)` | 设置延迟 `ModelIO` 创建工厂。 |
| `add_run_hook(hook)` | 追加运行后 hook。 |
| `add_tool_runtime_plugin(plugin)` | 追加工具运行时插件。 |
| `set_payload_defaults(payload)` | 合并默认 payload。 |
| `set_response_format_default(response_format)` | 设置默认响应格式。 |
| `set_max_iterations_default(max_iterations)` | 设置默认最大迭代数。 |
| `set_max_context_window_tokens_default(tokens)` | 设置默认上下文窗口限制。 |
| `set_on_tool_confirm_default(on_tool_confirm)` | 设置默认工具确认回调。 |
| `set_on_human_input_default(on_human_input)` | 设置默认人类输入回调。 |
| `set_on_max_iterations_default(on_max_iterations)` | 设置默认最大迭代回调。 |
| `build()` | 完成构建并返回 `PreparedAgent`。 |

## PreparedAgent

已组装完毕、可执行的 Agent。持有 `KernelLoop`、合并后的 `Toolkit` 和解析后的默认值。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/builder.py` |
| 继承/协议 | `-` |
| 导出状态 | 通过 `unchain.agent` 导出。 |
| 对象类型 | Dataclass。 |

### 公共方法

| 方法 | 返回 | 说明 |
| --- | --- | --- |
| `run()` | `KernelRunResult` | 执行 kernel loop。 |
| `resume_human_input()` | `KernelRunResult` | 从人类输入暂停恢复。 |

### `src/unchain/agent/modules/`

可插拔模块，在准备阶段配置 `AgentBuilder`。

## AgentModule（协议）

所有 Agent 模块必须满足的协议。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/base.py` |
| 对象类型 | 协议。 |

### 必需接口

| 属性/方法 | 类型 | 说明 |
| --- | --- | --- |
| `name` | `str` | 模块标识符。 |
| `configure(builder)` | `-> None` | 在 Agent 准备阶段调用。 |

## ToolsModule

将 tools、toolkits 和 callable 注册到 builder。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/tools.py` |
| 继承/协议 | `BaseAgentModule` |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `tools` | `tuple[Any, ...]` | 默认值：`()`。 |
| `name` | `str` | 默认值：`"tools"`。 |

## MemoryModule

将 memory 挂载到 builder。接受 `KernelMemoryRuntime`、`MemoryManager`、`MemoryConfig` 或原始 dict。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/memory.py` |
| 继承/协议 | `BaseAgentModule` |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `memory` | `KernelMemoryRuntime \| MemoryManager \| MemoryConfig \| dict[str, Any] \| None` | 默认值：`None`。 |
| `store` | `SessionStore \| None` | 默认值：`None`。 |
| `name` | `str` | 默认值：`"memory"`。 |

## PoliciesModule

设置默认 payload、响应格式、最大迭代数、上下文窗口 token 数、工具确认回调，
以及可选 completion policy。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/policies.py` |
| 继承/协议 | `BaseAgentModule` |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `payload` | `dict[str, Any]` | 默认值：`{}`。 |
| `response_format` | `ResponseFormat \| None` | 默认值：`None`。 |
| `max_iterations` | `int \| None` | 默认值：`None`。 |
| `max_context_window_tokens` | `int \| None` | 默认值：`None`。 |
| `on_tool_confirm` | `Callable[..., Any] \| None` | 默认值：`None`。 |
| `completion_policy` | `CompletionPolicy \| None` | 默认值：`None`；显式启用的有界 repair policy。 |
| `name` | `str` | 默认值：`"policies"`。 |

### Completion policy

`completion_policy` 是显式启用能力。为 `None` 时，`PreparedAgent` 会直接返回
第一次 completed 的 `KernelRunResult`，不会额外 validation 或 repair turn。传入
policy 后，`PreparedAgent` 在 kernel run 完成后委托给 `CompletionPolicyRunner`；
kernel loop 本身仍然是有界 run loop。

```python
from unchain.agent import CompletionEvaluation, CompletionPolicy, PoliciesModule

def validate(result):
    final_text = str(result.messages[-1].get("content") or "")
    if "acceptance criteria" in final_text:
        return CompletionEvaluation(complete=True)
    return CompletionEvaluation(
        complete=False,
        feedback="Revise the answer and include acceptance criteria.",
    )

PoliciesModule(
    completion_policy=CompletionPolicy(
        validator=validate,
        max_repair_turns=1,
    )
)
```

## OptimizersModule

将运行时 harness（如上下文窗口优化器）挂载到 builder。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/optimizers.py` |
| 继承/协议 | `BaseAgentModule` |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `harnesses` | `tuple[object, ...]` | 默认值：`()`。 |
| `name` | `str` | 默认值：`"optimizers"`。 |

## SubagentModule

注册委托、handoff 和 worker-batch 工具，以及 `SubagentToolPlugin`。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/subagents.py` |
| 继承/协议 | `BaseAgentModule` |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `templates` | `tuple[SubagentTemplate, ...]` | 默认值：`()`。 |
| `policy` | `SubagentPolicy` | 默认值：`SubagentPolicy()`。 |
| `executor` | `SubagentExecutor \| None` | 默认值：`None`。 |
| `name` | `str` | 默认值：`"subagents"`。 |

## InteractionModule

把 `FyiChannel` 接入 Agent 的唯一入口：`configure()` 在 `fyi_channel` 不为 `None` 时向 builder 注册 `FyiInjectionHarness(channel=fyi_channel)`（`before_model` 阶段，`order=180`），使 mid-run 的 fyi 消息在当前 iteration 就对模型可见。用法与三原语的完整语义见 [Interject 原语指南](../guides/interject.md)。

| 项目 | 细节 |
| --- | --- |
| 源码 | `src/unchain/agent/modules/interaction.py` |
| 继承/协议 | `BaseAgentModule` |
| 对象类型 | Dataclass (frozen)。 |

### 字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `fyi_channel` | `FyiChannel \| None` | 默认值：`None`。为 `None` 时不注册任何 harness。 |
| `name` | `str` | 默认值：`"interaction"`。 |

### 示例

```python
from unchain import Agent
from unchain.agent import InteractionModule
from unchain.interaction import FyiChannel

fyi_channel = FyiChannel()  # 每个 run 一个新实例

agent = Agent(
    name="assistant",
    provider="openai",
    model="gpt-5",
    modules=(InteractionModule(fyi_channel=fyi_channel),),
)
```

`SteerBuffer`（run 结束后合并追加请求）与 `ProgressDigest`/`build_btw_prompt`（旁路问答）不经过 `AgentModule`，由调用方在 `run()` 之外自行编排——同样在 [Interject 原语指南](../guides/interject.md) 中有完整示例。
