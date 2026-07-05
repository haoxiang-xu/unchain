# 使用 Interject 原语：fyi / steer / btw

本指南说明如何在一次 `Agent.run()` 执行期间处理"用户中途又发来一条消息"这件事。unchain 提供三个互不依赖的原语，分别对应三种不同的语义，不做自动路由——调用方（CLI、UI 层）自己决定一条消息属于哪一种。

## 前提条件

- 理解 `Agent.run()` 的基本执行模型与 `callback` 事件流（参见 [架构总览](../skills/architecture-overview.md)）
- 了解 harness / module 机制的基本形态（参见 [添加新的 Kernel Harness](add-harness.md)）

## 参考文件

| 文件 | 职责 |
|------|------|
| `src/unchain/interaction/fyi.py` | `FyiChannel`、`FyiMessage`、`wrap_fyi`、`FyiInjectionHarness` |
| `src/unchain/interaction/steer.py` | `SteerBuffer`、`merge_steered_texts` |
| `src/unchain/interaction/btw.py` | `ProgressDigest`、`build_btw_prompt` |
| `src/unchain/agent/modules/interaction.py` | `InteractionModule` —— 把 `FyiChannel` 接入 Agent 的唯一入口 |
| `src/unchain/cli/repl.py` | `unchain-repl` 参考实现：`/btw` `/fyi` `/steer` 三个前缀 |
| `src/unchain/events/normalizer.py` | 原始 `fyi_injected` 事件到 `interaction.fyi_injected` 的规范化 |

## 语义速览

| 原语 | 注入时机 | 语义 |
|------|---------|------|
| `fyi` | 当前 run 的下一个 iteration 边界（`before_model` 阶段） | 给当前任务补充信息，不打断、不重开 run |
| `steer` | 当前 run 结束之后 | 把 run 期间收到的多条追加请求合并成一条，立即开下一轮 |
| `btw` | 立即（旁路） | 不接触主循环，用一个独立的侧路 Agent 基于进度摘要即时回答 |

## 步骤

### 1. fyi：往当前 run 里插一句话

`FyiChannel` 是线程安全的邮箱：任意线程 `post`，agent 循环在每个 `before_model` 阶段 `drain`。要让它生效，必须通过 `InteractionModule` 把 channel 接到 Agent 上——`InteractionModule.configure()` 会注册 `FyiInjectionHarness(channel=fyi_channel)`（`order=180`），该 harness 用 `InsertMessagesOp(target=ContextTarget.CONVERSATION, ...)` 把消息插入对话，因此消息在**当前这一轮**就对模型可见，而不必等到下一次 transcript 重建。

```python
import threading

from unchain import Agent
from unchain.agent import InteractionModule
from unchain.interaction import FyiChannel

fyi_channel = FyiChannel()  # 每个 run 一个新实例，见下方"调用方责任"

agent = Agent(
    name="assistant",
    provider="openai",
    model="gpt-5",
    instructions="You are a helpful assistant.",
    modules=(InteractionModule(fyi_channel=fyi_channel),),
)

def run_in_background() -> None:
    result = agent.run("帮我写一份项目周报", callback=print)
    print(result.status, result.messages[-1])

worker = threading.Thread(target=run_in_background, daemon=True)
worker.start()

# run 进行中，用户又发来一句话
fyi_channel.post("记得把上周的两个 bug 修复也算进去")

worker.join()
```

`FyiChannel.post()` 返回 `message_id`（`fyi_` 前缀 + uuid），`pending_count()` 可用于展示"还有几条待注入"。`post` 的 `origin` 默认为 `"user"`；`origin="system"` 会走另一套措辞（用于 §3 的 btw 场景，见下文"调用方责任"）。

注入发生时，回调事件流上会收到一条原始事件 `{"type": "fyi_injected", "count": ..., "messages": [...]}`。这是**原始**事件——规范化为 `interaction.fyi_injected`（`slot="trace_inline"`，`scope="turn"`，`visibility="user"`）是一个独立的、需要显式调用的步骤：

```python
from unchain.events.normalizer import RuntimeEventNormalizerContext, normalize_raw_event

context = RuntimeEventNormalizerContext(session_id="thread-1", root_run_id="run-1")
events = normalize_raw_event(raw_event, context=context)
```

如果你的 UI 层直接消费原始 callback 事件（而不经过 normalizer），需要自己识别 `"fyi_injected"`。

### 2. steer：run 结束后合并追加请求，开下一轮

`SteerBuffer` 不在 run 期间生效，而是在 run 结束后 `drain_merged()`：单条消息原样返回；多条会编号合并成一段文本，交给下一轮 `Agent.run()` 当作新任务。这是 `unchain-repl` 里 "steer 链" 的核心模式（`src/unchain/cli/repl.py` 的 while 循环）：

**关键点：steer 链是同一个对话接着做，不是几个互不相关的新任务。** 下一轮 run 必须看到完整的历史消息（上一轮的 `result.messages` + 新合并的 steer 文本），不能只传 `merged` 这一句——否则上一轮 run 里生效的约束（比如一句 `/fyi "只用中文回复"`）到下一轮就会被遗忘，行为不连贯。`Agent.run()` 的 `messages` 参数接受 `str | list[dict]`（见 `src/unchain/agent/agent.py`），首轮传字符串，后续每一轮都传拼接好的消息列表：

```python
from unchain import Agent
from unchain.interaction import SteerBuffer

steer_buffer = SteerBuffer()  # 每个任务一个新实例；steer 链内的多轮 run 复用同一个

def build_followup_messages(prior_messages, merged):
    """下一轮 run 的输入 = 上一轮完整历史 + 新合并的 steer 文本这一个 user turn。"""
    return [*prior_messages, {"role": "user", "content": merged}]

current_task = "调研三家 CI 服务商的定价"  # 首轮：纯字符串
while True:
    agent = Agent(name="assistant", provider="openai", model="gpt-5")
    # 假设有另一个线程在 run 期间调用了：
    #   steer_buffer.post("顺便把 GitHub Actions 也加进对比")
    #   steer_buffer.post("最后给一个推荐结论")
    result = agent.run(current_task)

    merged = steer_buffer.drain_merged()
    if merged is None:
        break  # 没有追加请求，结束
    print(merged)  # "The user sent several follow-up requests while the previous task was running. Address all of them, in order:\n1. 顺便把 GitHub Actions 也加进对比\n2. 最后给一个推荐结论"
    # 后续轮：携带完整历史的消息列表，而不是只传 merged
    current_task = build_followup_messages(result.messages, merged)
```

`merge_steered_texts(["顺便把 GitHub Actions 也加进对比", "最后给一个推荐结论"])` 会产出一段带编号列表的合并文本；只有一条时原样返回，不加任何前缀。

`src/unchain/cli/repl.py` 里的参考实现就是这个模式：`_run_worker` 把每轮 run 的 `KernelRunResult` 存进一个 `result_holder`，`main()` 用 `build_followup_messages(result_holder[0].messages, merged)` 拼出下一轮的输入；如果上一轮 run 失败（`result_holder` 为空），退化为只传 `merged`。

### 3. btw：旁路即时问答，不干扰主循环

`ProgressDigest` 是一个有界、线程安全、**永不抛异常**的事件收集器：把它和主 callback 一起挂在 run 上，它会从 `tool_name`/`content` 等事件字段里提炼出简短摘要。`build_btw_prompt` 用这份摘要构造一个侧路 Agent 的 system+user 消息——侧路 Agent 只负责"基于已知进度回答问题"，明确被告知不要自己动手做任务。

```python
from unchain import Agent
from unchain.interaction import ProgressDigest, build_btw_prompt

digest = ProgressDigest()

def callback(event: dict) -> None:
    digest(event)          # 从不抛异常，可放心和主 callback 一起跑
    # ... 主 callback 的其他逻辑（渲染等）

main_agent = Agent(name="assistant", provider="openai", model="gpt-5")
result = main_agent.run("重构支付模块", callback=callback)
```

主 run 进行中，随时可以起一个独立的侧路 Agent 回答用户的旁路提问：

```python
from unchain.kernel.lifecycle_events import last_assistant_text

side_agent = Agent(name="side_assistant", provider="openai", model="gpt-5", instructions="")
messages = build_btw_prompt(
    original_task="重构支付模块",
    digest_summary=digest.summary(),
    question="现在做到哪一步了？",
)
side_result = side_agent.run(messages, max_iterations=1)
answer = last_assistant_text(side_result.messages)
```

用 `last_assistant_text(side_result.messages)` 而不是裸 `side_result.messages[-1]["content"]`——后者假设最后一条消息一定是助手文本消息，一旦模型走了工具调用或返回了非字符串 content，索引/取值就会直接抛异常；`last_assistant_text` 从后往前找最后一条 `role == "assistant"` 且 `content` 是非空字符串的消息，找不到就返回空字符串，复制粘贴到自己代码里更稳妥（`src/unchain/cli/repl.py::_side_answer` 就是这么用的）。

侧路 Agent 完全独立于主 Agent，`max_iterations=1` 是常见选择——它只需要回答一次，不需要多轮工具调用。

## 调用方责任

- **每个对话/任务新建 `FyiChannel` / `SteerBuffer` 实例，这是硬要求。** 两者都不知道 `run_id`，如果跨对话复用同一个实例，上一个对话未被 drain 的消息会串到下一个不相关的对话里。但注意"对话"的粒度是**任务**，不是**run**：同一个任务内，steer 链把多轮 `agent.run()` 串成一次对话时，在这些 run 之间复用同一个 `FyiChannel` / `SteerBuffer` / `ProgressDigest` 实例是有意为之的设计——配合 §2 里 run 间的消息历史续接，语义上就是"同一对话接着做"，跨 run 未 drain 的 fyi 在下一轮的第一个 iteration 边界被注入是预期行为，不是串话。`unchain-repl` 的做法是每次用户发起新任务时才新建（见 `src/unchain/cli/repl.py::main`），而不是每次 run 都新建。
- **btw 答完之后，建议往 `FyiChannel` 补一条 `origin="system"` 备注**，让主 Agent 保持知情（否则主 Agent 完全不知道旁路问答发生过）：

  ```python
  fyi_channel.post(
      f"Q: 现在做到哪一步了？ A: {answer}",
      origin="system",
  )
  ```

  `origin="system"` 会触发 `wrap_fyi` 里的"侧助手已经回复过"措辞，而不是"用户又发来一条消息"的措辞——两者提示模型的动作不同（后者可能要求调整计划，前者通常不需要动作）。

- **`FyiChannel` / `SteerBuffer` 是活对象，必须通过 module 构造参数接入，不能塞进 `payload` 或 `run()` 的 kwargs。**
  - `Agent.run(..., payload={"fyi_channel": fyi_channel})` ——`payload` 是给 provider 请求体用的：OpenAI/Anthropic provider 会把它展开成 SDK 调用的具名参数，塞入 `fyi_channel` 这类未知 key 会立刻抛 `TypeError: Responses.create() got an unexpected keyword argument`；Ollama provider 则会在 JSON 序列化活对象时抛 TypeError。总之是响亮的即时报错，不是静默失败——但无论哪种，`payload` 都不是传活对象（如 FyiChannel）的通道。
  - `Agent.run(..., fyi_channel=fyi_channel)` ——`run()` 的参数是固定签名，未知 kwarg 直接 `TypeError`。
  - 正确方式始终是构造 `Agent` 时传入 `modules=(InteractionModule(fyi_channel=fyi_channel),)`（`SteerBuffer`/`ProgressDigest` 目前没有对应的 module 接入点，由调用方在 `run()` 之外自行编排，如上面 §2、§3 所示）。

## CLI 参考实现

`unchain-repl`（`src/unchain/cli/repl.py`）是这三个原语的最小可运行示例：运行中输入 `/btw <问题>`、`/fyi <文本>`、`/steer <文本>` 分别对应三种语义；`route_input()` 只认这三个显式前缀，没有自动判断消息类型的路由器——这是有意为之的设计,把"这句话是什么意思"的判断权交给调用方（用户自己敲前缀，或未来由上层 UI 决定）。

```bash
unchain-repl --provider ollama --model llama3
```

## 测试

```bash
PYTHONPATH=src pytest tests/ -q --tb=short -k "interaction_fyi or interaction_steer or interaction_btw or cli_repl_routing"
```

完整测试套件：

```bash
PYTHONPATH=src pytest tests/ -q --tb=short
```

## 相关文档

- [Agents API 参考](../api/agents.md) —— `InteractionModule` 字段与接入方式
- [添加新的 Kernel Harness](add-harness.md) —— harness 协议与阶段调度
- [架构总览](../skills/architecture-overview.md) —— kernel loop 执行模型
