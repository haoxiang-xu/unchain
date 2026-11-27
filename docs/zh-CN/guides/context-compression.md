# 选择本地裁剪或 Agent 摘要

`unchain.context.ContextCompressor` 提供两种明确可选的 Python 接口：

- `mode="trim"`：本地二分搜索可保留的最近历史，不增加模型请求。
- `mode="summary"`：将旧历史交给独立 Agent，生成有证据引用的交接摘要。

两种模式都会先归档完整原文，再返回下一次模型请求使用的视图。系统和开发者
指令、最新用户轮次保持原样；完整工具调用与结果不能拆开。所有工具参数、输出、
错误和引用都保留在归档原文中；调用和结果的对应表由程序生成。

```python
from unchain import Agent
from unchain.context import AgentSummarizer, ContextCompressor, DirectoryCompressionStore

store = DirectoryCompressionStore("./context-archives")
local = ContextCompressor(mode="trim", archive_store=store)
view = local.compress(full_history, max_input_tokens=16_000, fixed_overhead_tokens=700)

summary_agent = Agent(name="summarizer", provider=provider, model=model, api_key=api_key)
semantic = ContextCompressor(
    mode="summary", archive_store=store,
    summarizer=AgentSummarizer(summary_agent),
    summary_reserve_tokens=2_400,
    max_summary_input_tokens=64_000,
)
view = semantic.compress(full_history, max_input_tokens=16_000)
```

`max_input_tokens` 是已经扣除模型输出预留后的输入额度；工具定义等固定开销
通过 `fixed_overhead_tokens` 计一次。`summary_reserve_tokens` 包含摘要和归档标记。
超预算会明确失败，不会静默截断证据或切换成裁剪模式。

Agent 摘要器会复制指定 Agent 的模型配置，移除模块和工具，使用独立系统提示词，
运行一次。也可以传入自定义摘要函数，此时必须通过 `summary_key` 标识其模型、
提示词和参数；配置改变时更换该 key。内置 Agent 摘要器自动生成这一身份。
可以用 `AgentSummarizer(agent, payload=...)` 传入对应 provider 的输出上限等参数。
内置适配器只接受输出上限、采样、停止词、惩罚和 reasoning/thinking 等生成设置，
拒绝覆盖指令、输入、历史状态、模型或工具的参数；其他 provider 控制请用自定义摘要函数。

提示词参考 OpenAI 公开的
[Session Memory 示例](https://developers.openai.com/cookbook/examples/agents_sdk/session_memory)，
要求按时间整理、检查矛盾、标记 `UNVERIFIED`、精确保留错误和工具结果。
这并非 Responses 服务端压缩未公开的内部提示词。

摘要必须返回严格 JSON，包含 `schema="unchain.context_summary.v1"` 以及
`goal`、`constraints`、`decisions`、`progress`、`open_items`、`next_steps` 六个数组。
每项只有 `text` 和 `source_ids`，引用必须指向本次归档中的真实消息。
`request.to_dict()` 给模型提供 `s0`、`s1` 等短编号，并标注来源权限；用户文本和工具结果
混在一条消息时，会分别标注内容块。程序将合法短编号映射回归档的完整编号，不猜测或修复
拼错的引用。自定义函数也可直接引用 `request.source_ids`。允许一层完整 JSON 代码围栏，
拒绝额外解释文字、多个对象、重复字段和未知引用。
摘要作为不受信任的历史观察注入，不升级成系统指令。引用合法不等于内容理解正确，
真实摘要质量仍需要对选定模型做评估。

有未完成的工具调用或审批时，摘要模式会推迟并明确报错，等待结果或拒绝记录完成后
再重试。没有压缩压力时直接返回原文，不调用摘要模型。摘要请求超过配置的输入额度
也会报错，原始工具输出不会被偷偷截断。

```python
from unchain.agent import ToolsModule

# 仅当 view.archive_id 非空时创建；这个工具只绑定这一份归档。
reader = view.read_tool(store)
main = Agent(name="worker", provider=provider, model=model, api_key=api_key,
             modules=(ToolsModule(tools=(reader,)),))
result = main.run(list(view.messages))
```

模型可以调用 `context_archive_read()` 读取证据编号和调用链的 JSON 目录，再用
`context_archive_read(source_id="...")` 读取原始消息。目录的 `tool_catalog` 还列出工具名称
和调用次序。用 `context_archive_read(call_id="...")` 可以精确读取一次调用及其结果；也可用
`context_archive_read(tool_name="run_checks", occurrence=0)` 读取第一次检查，`occurrence=1`
读取第二次。返回的调用和结果是 provider 原始内容，并附来源编号，避免把初次失败和后续
重跑混淆。消息编号和调用选择条件不能混用；有歧义的调用编号明确报错。
返回值按 Unicode 字符分页，
通过 `offset=next_offset` 继续读取；拼接完整 JSON 文本后可核对 `sha256`。
小消息会额外返回完整的 `message`，大消息不会一次塞满上下文，原文仍完整保留。
程序也可直接调用
`store.get(view.archive_id).read(source_id)`。归档保存失败不会返回裁剪后的视图；
相同原文和策略会复用持久化结果，冷重启后也不重复调用摘要模型。读取时校验版本、
原文和内容摘要，损坏或错身份会报错。按原始聊天记录的访问权限保护归档目录。

每次都从应用 journal 取得完整原始历史来压缩。不要把上次 `view.messages` 再喂给
压缩器；接口会拒绝这种用法，防止摘要套摘要丢失原始证据。
如果其他 harness 已经把工具结果缩成预览，必须先从 journal 或产物补回原文。
接口会拒绝已知的 Unchain 工具预览格式，即使没有压缩压力也会检查；它无法识别任意
调用方自行截断的文本，也无法恢复输入前已经丢弃的数据。不要默认把上一个 Agent 返回的
模型视图当成完整原始 journal。

现有 Context V2 裁剪也已使用二分法跳过不可能放下的历史后缀，但它的检查点协议
和消费证明不变。本接口不会自动切换 PuPu 或 `ContextModule` 的模式；它适用于应用
自己管理历史、再传给 Agent 的边界。不要对 V2 已编译的模型视图叠加裁剪。
