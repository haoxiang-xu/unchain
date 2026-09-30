# Agent Skills

`agent-skills` 主题的正式简体中文 skills 章节。

## 角色与边界

本章覆盖 `SkillsModule`：发现 `SKILL.md` 形式的指令文件和 toolkit 内嵌的 skill 描述符，渲染 name+description 目录，暴露 `skill` 激活工具，从用户自己的话里解析 `/name` 显式调用，并把每一次激活投影进一个持久的 `<active_skills>` system 块——这个块能扛过压缩、resume、retry/replay 和冷重启。

skill 是一段可复用的任务特定指令，用一个简短的 kebab-case 名字和一句话描述来定位。skill 是被发现的，而不是手工注册的：在已知目录下放一个 `SKILL.md` 文件，或者在 toolkit manifest 里声明 `[[skills]]`，下次配置 agent 时 `SkillsModule` 就会找到它。

**跨 harness 兼容性。** `<name>/SKILL.md` 这种目录形状——一个带 `name` 和 `description` 的 YAML frontmatter 块，后面跟 Markdown 指令正文——和 Claude Code、Codex、DeepSeek Harness 的 skill 目录用的是同一套约定。为其中任一 harness 写的 skill 目录，原样丢进 `.unchain/skills/<name>/` 就能被 Unchain 发现：frontmatter 契约是一个有界的、兼容的子集（见下文 "SKILL.md frontmatter"），无法识别的 frontmatter key 会被保留为惰性元数据，而不是被拒绝。

## 依赖关系

- `unchain.skills.frontmatter` —— 不依赖任何第三方库的安全 YAML 子集解析器（`parse_skill_file`）；不知道 registry 或 kernel 的存在。
- `unchain.skills.models` —— 共享词汇表：`SkillIdentity`、`SkillSummary`、`LoadedSkill`、`ActiveSkill`、`SkillDiagnostic`、`SkillInventory`，以及 name/description 校验和 revision 哈希。不做任何 I/O。
- `unchain.skills.registry`（`SkillRegistry`、`SkillsConfig`）—— 扫描分级的文件系统根目录和运行时 `Toolkit.skills` 列表，合并并解决冲突。依赖 `frontmatter` 和 `models`。
- `unchain.skills.rendering` —— 把 `models.py` 里的记录转成 `<available_skills>` / `<active_skills>` 的线上文本，反过来也能严格解析回去。不知道 kernel 或 registry 的存在。
- `unchain.skills.activation`（`ActiveSkillSet`、`ActivationQueue`）—— 持久激活状态机：解析已持久化的块，合并新的激活，再重新渲染。
- `unchain.skills.harness`（`SkillCatalogHarness`、`SkillActivationHarness`）—— 通过 `KernelLoop.dispatch_phase` 提交这两个块的 `RuntimeHarness` 实现，复用了和 `ToolPromptHarness` 一样的"一个定界 system 块"模式。
- `unchain.skills.tools`（`build_skill_tool`）—— 从一个 registry + queue 构建模型可调用的 `skill` 工具。
- `unchain.agent.modules.skills`（`SkillsModule`）—— 在 configure 时把以上所有东西接进 `AgentBuilder` 的 `BaseAgentModule`。
- `unchain.tools.models.SkillDescriptor` 与 `unchain.tools.toolkit.Toolkit(skills=...)` —— toolkit 内嵌 skill 的载体，由 `unchain.tools.registry.ToolkitRegistry` 从 `[[skills]]` manifest 表解析而来，也可以直接构造。
- `unchain.context.composition` 与 `unchain.runtime.runtime_protocol` —— 在 Context Composition 遥测里给这两个渲染块记账，并对外宣告 `skills` runtime protocol。

## 核心对象

- `SkillsConfig` —— 发现配置（根目录、reserved commands、工具名、description 预算）。
- `SkillRegistry` —— 每次调用都不缓存的扫描器：`list()`、`get(name)`、`resolve(token)`。
- `SkillIdentity` / `SkillSummary` / `LoadedSkill` / `ActiveSkill` / `SkillDiagnostic` / `SkillInventory` —— 规范记录（`unchain.skills.models`）。
- `SkillDescriptor` —— toolkit 内嵌的一个 skill（`unchain.tools.models`）。
- `ActivationQueue` / `ActiveSkillSet` —— 激活状态机制（`unchain.skills.activation`）。
- `SkillCatalogHarness` / `SkillActivationHarness` —— `before_model` / `after_tool_batch` 两个 harness。
- `SkillsModule` —— agent module。

## 执行流与状态流

- `SkillsModule.configure(builder)` 构建一个 `SkillRegistry`，list 一次 inventory，只有当时至少存在一个 model-invocable 的 skill 才注册 `skill` 工具（每次 run 只决定一次——见下文 D7）。
- 每个 `before_model` 步骤，`SkillCatalogHarness` 都会重新扫描 registry，从最新的 inventory 重新渲染 `<available_skills>`（无状态——就像 `<tools>` 块一样，不会被持久化）。
- 模型调用 `skill(name=...)`，或者用户在自己的消息里写 `/name`；两条路径最终都变成一条待处理的激活。
- `SkillActivationHarness`（order 262，排在 `SkillCatalogHarness` 的 order 260 之后）解析已有的 `<active_skills>` 块，合并待处理的激活，再重新渲染。这个块会同时写进 working messages 和 `state.transcript`，因此能扛过 checkpoint/resume。
- 在 `after_tool_batch` 阶段，本轮迭代中 `skill` 工具排队的激活会被取出并提交，所以工具结果和被激活的正文落在同一次迭代里。

## 配置面

完整字段表见下文 "SkillsConfig"。简单说：扫描哪些根目录（`include_project_dirs`、`extra_dirs`、`include_user_dirs`、`project_root`、`home`）、目录 description 的截断预算、工具的名字，以及一组永远不能解析成 skill 的 `/name` token。

## 扩展点

- 在 `.unchain/skills/` 或 `.agents/skills/` 下放 `SKILL.md`（或扁平的 `<name>.md`）文件——不需要改代码。
- 把 `extra_dirs` 指向任何其他目录（比如同步过来的 `.claude/skills` 或 `.codex/skills` checkout），显式地让它可被发现。
- 在 toolkit 的 `toolkit.toml` manifest 里声明 `[[skills]]`，随 toolkit 一起分发一个 skill。
- 直接构造 `SkillDescriptor` 实例并传给 `Toolkit(skills=(...))`，作为程序化来源。

## 常见陷阱

- `skill` 工具是否存在，是在 configure 时根据当时的 inventory 快照一次性决定的——运行期间新增到磁盘上的 skill 仍会出现在目录里（每步重新渲染），但如果 configure 时一个都没有，不会追溯性地给已配置好的 agent 加上这个工具。
- 文件系统里的 `SKILL.md`/`<name>.md` skill 永远不带 `tools` 列表或别名——这两个字段只存在于 toolkit 内嵌的 `SkillDescriptor` skill 上。
- 没有解析出任何已知 skill 的 `/name` token 会原样留作普通字面文本；不会有任何东西删改用户的消息。
- 目录或激活状态一变，prompt-cache 前缀就会跟着变，这和任何其他会改写靠前 system 消息的 harness 是一样的代价。

## 关联 class 参考

- [Tool System API](../api/tools.md) —— `SkillDescriptor`、`Toolkit`。
- [Runtime API](../api/runtime.md) —— `RuntimeProtocol` manifest。
- [Tool System Patterns](tool-system-patterns.md) —— 这个特性所依托的更广泛的 tool/toolkit 模型。

## 源码入口

- `src/unchain/skills/frontmatter.py`
- `src/unchain/skills/models.py`
- `src/unchain/skills/registry.py`
- `src/unchain/skills/rendering.py`
- `src/unchain/skills/activation.py`
- `src/unchain/skills/harness.py`
- `src/unchain/skills/tools.py`
- `src/unchain/agent/modules/skills.py`
- `src/unchain/tools/models.py`（`SkillDescriptor`）
- `src/unchain/context/composition.py`
- `src/unchain/runtime/runtime_protocol.py`

## 详细参考

以下各节覆盖确切的线上格式、配置字段和生命周期保证。每一个渲染示例都是对应函数实际产出的原样复现。

## SKILL.md frontmatter

skill 文件是一个带 YAML frontmatter 的 Markdown 文件：

```
---
name: release-notes
description: Draft release notes from the merged PRs since the last tag.
---
1. Collect merged PRs since the last release tag with `git log`.
2. Group entries by type: feat, fix, docs, chore.
3. Write one Markdown bullet per PR, linking the PR number.
```

`name` 必填，必须匹配 `^[a-z0-9]+(?:-[a-z0-9]+)*$`（小写 kebab-case，1–64 个字符）——而且必须和目录名（对 `<name>/SKILL.md` 而言）或文件 stem（对扁平的 `<name>.md` 而言）完全一致，否则这个候选项会被丢弃，并记一条 `invalid` 诊断。`description` 必填、不能为空，最多 1024 个字符（原样持久化；只有渲染目录那一行会折叠空白并截断）。两个可选的策略 key，`disable-model-invocation` 和 `user-invocable`，分别控制这个 skill 能否被模型激活（`skill` 工具）和/或被用户激活（`/name`）。任何其他 frontmatter key 都会原样保留为惰性元数据（通过 `SkillRegistry.get(name).metadata` 能拿到，目录里拿不到）。

解析器（`unchain.skills.frontmatter.parse_skill_file`，决策 D1）是一个有界的、不依赖第三方库的安全 YAML 子集——刻意不做成通用解析器：

**接受：**
- 列 0 开始的 `key: value` 行，嵌套映射和 `- item` 块列表纯靠缩进区分（都被捕获成普通的 `dict`/`list[str]`，不做特殊类型化）。
- 裸标量、`'单引号'`、`"双引号"` 标量；双引号标量支持 `\"`、`\\`、`\n`、`\t` 转义。
- 引号外的行内 `# comment` 文本。
- 块标量 `|`、`|-`、`|+`（literal）和 `>`、`>-`、`>+`（folded——多行用空格拼接，空行变成一个换行），带三种 chomping 变体（`clip`/默认、`strip`、`keep`）。
- flow 序列 `[a, "b", 'c']`（每个元素按同样的规则解析成标量）。
- 空值（`key:` 后面什么都没有，下面也没有缩进内容）变成 `""`。
- `true`/`false`/`yes`/`no`/`on`/`off` 在 `fields` 里保持为普通字符串——布尔强转（`coerce_bool`）是后面单独做的一步，而且只对那两个策略 key 生效。
- 重复 key：只有 `name`、`description`、`disable-model-invocation`、`user-invocable` 这四个 key 重复时会被 `SkillParseError` 拒绝；其他任何重复 key 都是后出现的覆盖前面的。

**拒绝**（全部抛出 `SkillParseError`）：
- 没有起始的 `---` 行，或者 frontmatter 围栏没有闭合（缺少结尾的 `---`）。
- 缩进里用了 tab 字符。
- YAML tag（`!...`）、锚点（`&x`）、别名引用（`*x`）。
- 顶层出现一行既不是 `key: value` 也不是块标量/嵌套块的延续行。
- flow 映射（`{a: 1}`）——flow 序列支持，flow 映射不支持。

输出是 `ParsedSkillFile(fields: dict[str, object], body: str)`，其中 `body` 是闭合 `---` 之后的全部内容，去掉首尾的空行（body 自己的 `---` 行，比如出现在一个 fenced 代码块里，永远不会被误认成第二个 frontmatter 分隔符，因为只有开头之后遇到的第一条 `---` 行会闭合 frontmatter）。

## 发现根目录与优先级排序

`SkillRegistry.roots()` 返回一份固定的、分级的文件系统根目录列表，外加一个给 toolkit 内嵌 skill 用的虚拟根。名字冲突时永远是 rank 更小的赢：

| Rank | 来源标签 | 位置 | 由谁控制 |
|---|---|---|---|
| 100 | `project-unchain` | `<project_root>/.unchain/skills` | `include_project_dirs` |
| 200 | `project-agents` | `<project_root>/.agents/skills` | `include_project_dirs` |
| 300 | `custom` | `extra_dirs` 里的每一项，按给定顺序 | `extra_dirs` |
| 400 | `user-unchain` | `<home>/.unchain/skills` | `include_user_dirs` |
| 500 | `user-agents` | `<home>/.agents/skills` | `include_user_dirs` |
| 600 | `toolkit`（或描述符自己声明的 `source`） | 运行时 toolkit 上的 `Toolkit.skills` | 始终扫描 |

`include_project_dirs=False` 会把 100 和 200 两个 rank 一起跳过（比如 PuPu 在没有 workspace 时运行）。`include_user_dirs=False` 会跳过 400 和 500。**不会隐式扫描 `.claude/skills` 或 `.codex/skills`**——如果想让 Unchain 发现为那些 harness 写的 skill，显式把 `extra_dirs` 指向它们。

`project_root` 解析规则：如果 `SkillsConfig.project_root` 显式设置了，就原样使用，不做 `.git` 向上查找。如果是 `None`，`resolve_project_root(Path.cwd())` 会从当前工作目录往上找最近一个含 `.git` 条目的祖先目录；找不到就退回 cwd 本身。

每个根目录**只扫描一层**（不递归 `**/SKILL.md`）：每个直接子项要么是一个含 `SKILL.md` 文件的子目录（skill 名字 = 目录名，base directory = 该子目录），要么是直接躺在根目录里的一个扁平 `<name>.md` 文件（skill 名字 = 文件 stem，base directory = **根目录本身**，不是每个 skill 独立的目录）。直接躺在根目录里（不在任何子目录下）的裸 `SKILL.md` 文件会被忽略，dotfile/以点开头的目录同样忽略。条目按解析后的真实路径去重——如果一个根目录解析后的真实路径已经扫描过（比如两个配置的根目录相同，或者一个符号链接指回了已扫描过的树），会被静默跳过。

**获胜者选择。** 当两个或更多被发现的 skill 共享同一个规范名字时，获胜者是 `(rank, identity.key)` 元组最小的那个——先比 rank，rank 相同再按字典序比 `source:source_id:name` 这个 identity key，作为确定性的平局判定。每一个没有获胜的候选项都会得到一条 `shadowed` 诊断，指明获胜来源。

**别名（aliases）** 只存在于 toolkit 内嵌的 `SkillDescriptor` skill 上（文件系统里的 `SKILL.md`/`<name>.md` skill 永远是 `aliases=()`——frontmatter 里没有对应的 key）。一个和某个获胜规范名字相同的别名会被丢弃，并记一条 `alias_shadowed` 诊断；一个被两个不同获胜身份同时声明的别名也会被丢弃，并记一条 `alias_ambiguous` 诊断。

**保留命令**（`SkillsConfig.reserved_commands`）只作用于 `/name` 的解析：一个名字和保留命令冲突的 skill 仍然会出现在目录里，也仍然能通过 `skill` 工具加载，但用户文本里的 `/reserved-name` 永远不会解析到它（list 时会记一条 `reserved` 诊断）。

## 来源身份（identity）与 revision

每个被发现的 skill 都有一个 `SkillIdentity(source, source_id, name)`。它的 `key` 属性是 `f"{source}:{source_id}:{name}"`——在任何需要把一个 skill 和另一个位置的同名 skill 区分开的地方都用这个持久的、带来源信息的身份（registry 从不只按名字合并描述符）。

- 文件系统来源：`source` 是上表里的根标签（`project-unchain`、`project-agents`、`custom`、`user-unchain`、`user-agents`）；`source_id` 是 `SKILL.md`（或 `<name>.md`）文件解析后的绝对路径。
- Toolkit 来源：`source` 默认是 `"toolkit"`（可以在 `SkillDescriptor` 上覆盖）；对 manifest 声明的 `[[skills]]`，`source_id` 是该 toolkit 的 id（由 `ToolkitRegistry` 的 manifest 加载器设置）；对直接构造的 `Toolkit(skills=...)`，`source_id` 就是调用方传给 `SkillDescriptor(source_id=...)` 的值（不传的话默认是 `""`——给每个程序化来源传一个不同的 `source_id`，避免身份冲突）。

**Revision**（`compute_skill_revision`）是 `"sha256:" + sha256(canonical_json({"body", "tools", "model_invocable", "user_invocable"})).hexdigest()`——一个前缀 `sha256:` 的 64 字符十六进制摘要。它刻意与 `name`、`description`、`source`、文件路径无关：只有真正会改变激活后行为的那部分才会影响 revision。`compute_inventory_revision` 类似地把每个获胜 skill 的 `(identity.key, revision)` 对排序后哈希成一个与顺序无关的 inventory revision。

对文件系统 skill 来说，`tools` 永远是 `()`——frontmatter 解析器没有把 `tools:` 映射到 toolkit 工具列表这个概念上；`SKILL.md` frontmatter 里的任何 `tools:` key 都只会被保留成惰性元数据，不会成为 revision 或 `<skill_content tools="...">` 里用到的那个 `tools`。只有 toolkit 内嵌的 `SkillDescriptor.tools` 才会填充这个字段。

## SkillsConfig

```python
from unchain.skills import SkillsConfig
```

| 字段 | 类型 | 默认值 | 含义 |
|---|---|---|---|
| `project_root` | `str \| Path \| None` | `None` | 显式的项目根目录；`None` 时通过最近的 `.git` 祖先解析。 |
| `include_project_dirs` | `bool` | `True` | 扫描 rank 100/200（项目根目录下的 `.unchain/skills`、`.agents/skills`）。 |
| `extra_dirs` | `tuple[str \| Path, ...]` | `()` | 额外的根目录，rank 300，按给定顺序扫描。 |
| `include_user_dirs` | `bool` | `True` | 扫描 rank 400/500（`home` 下的 `.unchain/skills`、`.agents/skills`）。 |
| `catalog_description_max_length` | `int` | `500` | 目录里每条 description 截断前的最大字符数，超出用 `…` 截断。必须 `>= 3`。 |
| `tool_name` | `str` | `"skill"` | 模型可调用的激活工具的名字。必须是非空字符串。 |
| `reserved_commands` | `tuple[str, ...]` | `()` | `/name` 永远不能解析成的一组 token，即使某个 skill 恰好叫这个名字。 |
| `home` | `str \| Path \| None` | `None` | 覆盖 rank 400/500 用到的用户 home 目录；`None` 时用 `Path.home()`。 |
| `extra_skills` | `tuple[SkillDescriptor, ...]` | `()` | 不依附任何 toolkit 的程序化 skill（例如宿主已安装的 skill pack）。它们以 rank 600 与 toolkit 内嵌 skill 一起进入注册表，各自保留自己的 `source` / `source_id` 身份。 |

`reserved_commands` 在构造时会被规范化：每一项都会被 strip、去掉开头的 `/`、转小写，空结果会被丢弃——所以 `SkillsConfig(reserved_commands=("/Foo", "bar", "//baz"))` 存下来的是 `("foo", "bar", "baz")`。`extra_dirs` 里的每一项都会被强转成 `Path`。`SkillsConfig.coerce(value)` 接受 `None`（默认值）、一个已有的 `SkillsConfig`，或者一个包含这些字段的普通 `dict`——`SkillsModule(config=...)` 内部就是这么用的，所以 `SkillsModule({"tool_name": "load_skill"})` 也能用。

## `<available_skills>` 目录块

`SkillCatalogHarness`（order 260，phase `before_model`）每一步都会从最新的 registry inventory 渲染出一条 `<available_skills>` system 消息——它**不会**被持久化进 transcript，和 `<tools>` 风格的 prompt 块一样，每步重新推导而不是存起来。只有 model-invocable 的 skill 会出现，按名字排序。如果一个都没有，就不渲染任何块（之前插入过的块也会被移除）。

单个 skill（`release-notes`，`tool_name="skill"`）的渲染示例：

```
<available_skills>
# unchain generated skills catalog v1
A skill is a reusable set of task-specific instructions. The following skills are available in this session:

- `release-notes`: Draft release notes from the merged PRs since the last tag.

If the user names a skill, or the task clearly matches a skill's description, call the `skill` tool with the exact skill name before taking task actions. Load all applicable skills, then follow their full instructions. This catalog contains summaries only; do not infer or follow a skill's instructions until it has been loaded.
Skills a user invokes directly, and skills you load, appear in the <active_skills> system block; follow that block and do not load a skill that is already listed there.
</available_skills>
```

## `skill` 工具

`SkillsModule` 只有在 discovery inventory **在 configure 时**至少有一个 model-invocable 的 skill 时才会注册 `skill` 工具（`build_skill_tool`）——这是每次 run 一次性的决定（D7）。这个工具是用 `always_load=True` 构建的，所以即便其他工具都处于 `ToolDiscoveryRuntime`/`ToolkitCatalogRuntime` 的懒加载模式下，它也始终留在线上。

调用 `skill(name="release-notes")` 会校验名字、通过 registry 解析它（拒绝一个格式合法但未知或非 model-invocable 的名字）、加载完整正文，并把激活排进队列——它从不直接返回正文本身。排队的激活会在下一次 `after_tool_batch` 被 `SkillActivationHarness` 提交，所以工具结果和被投影的 `<active_skills>` 条目落在同一次迭代里。工具结果是一个无正文的 envelope：

```
<skill_loaded name="release-notes" revision="sha256:f6ade60e95853dfb585e71ed497d5c3b2abc0e9495ff57141fc72abcfd7449a0" status="activated">
Skill "release-notes" is now active. Its full instructions are in the <active_skills> system block; follow them for the rest of this run.
</skill_loaded>
```

`status` 有以下几种：

| Status | 含义 | 句子 |
|---|---|---|
| `activated` | 这个身份在本次 run 里第一次被激活。 | `Skill "{name}" is now active. Its full instructions are in the <active_skills> system block; follow them for the rest of this run.` |
| `already_active` | 同一个身份、同一个 revision，已经处于激活状态。 | `Skill "{name}" was already active at this revision; its instructions are already in the <active_skills> system block.` |
| `superseded` | 同一个身份、但 revision 不同（更新了）。 | `Skill "{name}" was re-activated at a new revision; the <active_skills> system block now holds the updated instructions.` |

错误以纯文本 `Error: ...` 的形式返回，例如 `Error: invalid skill name "Nope"`、`Error: unknown skill "missing"`、`Error: skill "x" is not available for model invocation`。

## `/name` 显式调用

任何真实的用户消息都能直接激活一个 skill，不需要工具调用：在文本里任意位置查找匹配 `/([A-Za-z0-9_-]+)`（一个 `/`，后面跟字母、数字、`_` 或 `-`）的、以空白分隔的 token，按文本出现顺序、大小写不敏感地去重。每个 token 会先按大小写不敏感的方式对规范 skill 名字解析，再对别名解析；`reserved_commands` 在解析之前就会被跳过，永远不会激活任何东西；无法识别的 token 会原样留作普通字面文本——**用户的消息永远不会被修改**、删改或以任何方式改写。

"真实用户消息"指的是 transcript 里最新一条 `role: "user"` 且内容是纯文本（或文本 content block）、本身又不是合成 skill 消息（`<skill_content...>` / `<skill_loaded...>`）的条目——只包含工具结果的 user 轮次在寻找"最新真实文本"时会被跳过。

如果解析出的 skill 设了 `user-invocable: false`，就不会发生激活，取而代之会记一条 `user_invocation_denied` 诊断（这个 skill 仍然可以是 model-invocable，反过来也一样——这两个策略 flag 彼此独立）。否则这个 skill 会被加载并以 `user:<ordinal>:<digest>` 的 provenance 激活，其中 `<ordinal>` 是此刻持久 transcript 中真实用户消息的数量，`<digest>` 是用户消息原文 SHA-256 的前 16 个十六进制字符。同一个 `<ordinal>:<digest>` 会作为 `turn=…` 写进块头，记录最后一个被解析过 `/name` 的真实用户轮次。之后的每一步，harness 先把当前轮次身份与这个标记比较——tool loop、retry、interaction resume、冷重启重新处理的都是**同一个**轮次，标记一致，harness 不会再碰注册表或文件系统（即使期间出现了更高优先级的来源或新版本）。只有真正的新轮次（序号更大，或文本不同）才会实时解析；因此在后面的轮次里重复发送字节相同的文本是一次新的显式激活，会在版本变化时 supersede，版本未变时只刷新条目的 provenance。

## 持久化的 `<active_skills>` 块

`SkillActivationHarness`（order 262，phase 是 `before_model` 和 `after_tool_batch`）是整个 skills 体系里唯一必须扛过压缩和进程重启的那部分状态，所以它被投影成自己独立的一条定界 system 消息，并且——这一点很关键——被直接写进 `state.transcript`（不只是 working messages），因为 `ContextCompilerHarness` 每一步都会重建整个消息列表，而执行 checkpoint 持久化的是 `state.transcript`，不是临时的 model-context 编辑。

渲染示例，一个已激活的 skill：

```
<active_skills>
# unchain generated active skills v1 turn=1:0123456789abcdef
These skills were activated in this run. Follow their instructions; they stay in force until the run ends.

<skill_content name="release-notes" source="project-unchain" source_id="/repo/.unchain/skills/release-notes/SKILL.md" revision="sha256:f6ade60e95853dfb585e71ed497d5c3b2abc0e9495ff57141fc72abcfd7449a0" activation="user:7e4fca1cce098c11" tools="">
<skill_resources>
Base directory for this skill: /repo/.unchain/skills/release-notes
Resolve relative paths mentioned by this skill against the base directory before using them. Load referenced resources only as needed.
</skill_resources>

<skill_instructions>
1. Collect merged PRs since the last release tag with `git log`.
2. Group entries by type: feat, fix, docs, chore.
3. Write one Markdown bullet per PR, linking the PR number.
</skill_instructions>
</skill_content>
</active_skills>
```

属性值都做了 XML 转义（`&`、`"`、`<`、`>`）；`activation` 对 `/name` 调用是 `user:<ordinal>:<digest>`，对 `skill` 工具调用在能匹配到 call id 时是 `tool:<call_id>`（匹配不到时退回字面量 `tool`）。如果一个 skill 正文里含有字面的 `</skill_instructions>` 或 `</skill_content>`，会在激活时被拒绝，并记一条 `body_delimiter` 诊断，而不是被持久化成一种会破坏严格解析器的形式。没有 base directory 的 skill（`base_dir=None` 的 toolkit skill）得到的是 `<skill_resources>This skill has no resource directory.</skill_resources>`，而不是 `Base directory for this skill: ...` 那一行。

**为什么这能扛过一切：**

- 它是一条 `role: "system"` 消息。Context V2 的 `ContextCompilerHarness` 在每次 compile 时都会拷贝每一条来源的 system 消息，只对非 system 轮次做基于预算的压缩；`SlidingWindow`/`LastN` optimizer 同样只碰非 system 的部分。所以一个 system-role 的块从结构上就在两种 context 模式下都躲开了轮次压缩路径。
- 它被写进 `state.transcript`，而这正是执行 checkpoint 持久化、`merge_checkpoint_transcript_with_incoming` 在 resume 时恢复的东西——所以 suspend/resume、retry、replay，以及冷重启（纯粹从一份持久化的 checkpoint dict 重建）看到的都是同一个块，Unchain 不需要重读任何一个 skill 源文件。
- 每一遍只有消息里**第一个** `<active_skills>` 块会被当作权威（`ActiveSkillSet.from_messages`）；重复的块会在下一次渲染时被折叠成一个。
- 每个身份（`source:source_id:name`）恰好一条记录，用 dict 键住——以同一 revision 重新激活同一个身份是空操作（`already_active`）；以不同 revision 重新激活它（比如 skill 文件被改过之后，用户又显式输入了一次 `/name`）会**原地**替换那条记录（它在块里的位置不变），并报告 `superseded`。两轮之间编辑或删除底层源文件，对一个已激活的条目不会有任何影响——这个块从不会被拿去对着文件系统重新读取，只会被重新解析、和新的激活合并。
- 一个没有 checkpoint 可以 resume 的全新 run，一开始完全没有 `<active_skills>` 块（空集合）——激活状态不会跨无关的 run 泄漏。
- 块的 header 带有明确的版本号（`# unchain generated active skills v1`）。如果一个已持久化的块版本不是 `v1`，解析会直接失败并抛出 `SkillActivationStateError`（fail closed），而不是默默忽略或按别的方式重新解释——一个损坏的或未来格式的快照会直接中止这次 run，而不是被瞎猜着继续跑。

运行时还会把一份摘要镜像写进 `state.component_bucket("skills")`，形如 `{"version": 1, "active": {...}, "diagnostics": [...]}`，供内省使用；这份镜像永远是从渲染出的块反推重建的，反过来永远不成立——这个块本身才是唯一的持久权威。

## Toolkit 内嵌的 skills

一个 toolkit manifest（`toolkit.toml`）可以直接通过 `[[skills]]` 表分发一个或多个 skill。取自 `src/unchain/toolkits/builtin/plan/toolkit.toml`：

```toml
[[skills]]
name = "plan"
title = "Plan First"
description = "Draft a step-by-step plan and wait for confirmation before executing."
body = "Before doing any work: draft a step-by-step plan using the planning tools ({tools}), present it to me, and wait for my explicit confirmation before executing."
tools = ["plan_start", "plan_update", "plan_read", "plan_finalize", "plan_list"]
```

识别的 key：`name`、`description`、`body` 必填；`tools`（一个工具名列表，必须已经在同一份 manifest 的 `[[tools]]` 里声明过）、`disable-model-invocation`、`user-invocable`、`aliases` 可选。`name` 和 `aliases` 里的每一项都必须是 kebab-case，最多 64 个字符。任何其他 key——包括上面这个 PuPu 专用的 `title`，或者旧 manifest 里的 `phase` key——都会被加载器直接忽略；`plan/toolkit.toml` 为了 PuPu 自己的展示需要保留了 `title`，不再声明 `phase`。

`{tools}` 会在 `body` 里被替换成声明的工具名，每个都用反引号包住、用逗号分隔，而且是**每次 registry 扫描时**都会替换（`SkillRegistry.list()`），不是在 manifest 解析时替换一次就固定下来。对上面这个 `plan` skill 来说，渲染出的正文是：

```
Before doing any work: draft a step-by-step plan using the planning tools (`plan_start`, `plan_update`, `plan_read`, `plan_finalize`, `plan_list`), present it to me, and wait for my explicit confirmation before executing.
```

`ToolkitRegistry` 把每张 `[[skills]]` 表解析成 `SkillDescriptor(source="toolkit", source_id=<toolkit id>, base_dir=<toolkit 根路径>)`，并把得到的元组挂到 `ToolkitDescriptor.skills` 上；`ToolkitRegistry.instantiate_toolkit()` 会把这个元组拷贝到实例化出来的 `Toolkit.skills` 上。`AgentBuilder.add_tool(toolkit)`（决策 D10）会把每一个传入的 `toolkit.skills` 描述符追加到 `builder.toolkit.skills` 上——不做去重；如果两个来源声明了同一个名字，由 skills registry 的 rank/identity-key 解析机制来决定谁赢。

skill 也可以完全脱离 manifest，程序化构造后挂到任意 `Toolkit` 上：

```python
from unchain.tools import SkillDescriptor, Toolkit

toolkit = Toolkit(
    skills=(
        SkillDescriptor(
            name="echo-helper",
            description="Echo any text back verbatim.",
            body="When asked to echo something, call `echo` with the exact text.",
            tools=("echo",),
            source="toolkit",
            source_id="echo-toolkit",
        ),
    ),
)
```

给每一个程序化构造的来源都传一个明确、彼此不同的 `source_id`——默认值是 `""`，两个 `source`、`source_id`、`name` 都相同的描述符是无法区分的同一个身份。

## 诊断（Diagnostics）

`SkillDiagnostic(kind, name, source, source_id, message)` 是关于发现或调用问题的一条非致命提示；`unchain.skills.models` 里的 `SKILL_DIAGNOSTIC_KINDS` 定义了十种合法 kind，其中 registry 和 activation harness 目前实际会产出八种：

| Kind | 什么时候产出 |
|---|---|
| `invalid` | frontmatter 解析失败、名字和文件/目录不匹配、name 或 description 不合法，或者某个布尔策略字段无法强转。 |
| `shadowed` | 某个被发现的 skill 在名字冲突里输给了 rank 更低（或字典序更靠前）的获胜者。 |
| `alias_shadowed` | 某个别名和一个已获胜 skill 的规范名字相同，因而被丢弃。 |
| `alias_ambiguous` | 某个别名被两个不同的获胜身份同时声明，因而被丢弃。 |
| `reserved` | 某个获胜 skill 的名字和 `reserved_commands` 里的一项冲突（`/name` 永远解析不到它）。 |
| `inaccessible` | `SKILL.md`/`<name>.md` 文件无法被解析路径、读取，或按 UTF-8 解码；或者一个 skill 的正文在激活时无法被重新读取。 |
| `body_delimiter` | 一次激活因为正文里含有字面的 `</skill_instructions>` 或 `</skill_content>` 而被拒绝。 |
| `user_invocation_denied` | 一个 `/name` token 解析到了一个 `user-invocable: false` 的 skill。 |

还有两种 kind，`unknown_skill` 和 `model_invocation_denied`，被声明为合法的 `SkillDiagnostic` kind，但目前在 registry、activation、工具代码里的任何地方都没有被实际产出——它们是为未来预留的。

## Context Composition 归因

`context/composition.py::_classify_message` 给这两个渲染块在 `"skills"` 分类下各自留了一个专属的 taxonomy 槽位：一条内容以 `<available_skills>` 开头的 system 消息会被分类成 `("skills", "catalog_metadata")`，以 `<active_skills>` 开头的会被分类成 `("skills", "loaded_body")`。其他任何消息的分类都不受影响。（`"skills"` 这个分类下还定义了第三个 subtype，`expanded_invocation`，是给一个无关的、保护隐私的 context-composition 提示用的——两个 skills harness 都不会产出它。）

## Runtime protocol `skills` 1.0

`unchain.runtime.runtime_protocol` 在按规范顺序排列的 runtime protocol manifest 里对外宣告了一个带版本号的 `skills` 条目：

```python
RuntimeProtocol(
    id="skills",
    major=1,
    minor=0,
    features=(
        "active_skills_snapshot_v1",
        "catalog_v1",
        "skill_md_registry_v1",
        "skill_tool_v1",
        "toolkit_embedded_skills_v1",
        "user_invocation_v1",
    ),
)
```

`unchain.skills` 还导出了 `SKILLS_PROTOCOL_ID = "skills"`、`SKILLS_CONTRACT_VERSION = 1`（描述符/inventory/解析的形状）和 `ACTIVE_SKILLS_SNAPSHOT_VERSION = 1`（`<active_skills>` 块的格式，和这个块自己的 `v1` header 对应）。

## 完整使用示例

```python
from unchain import Agent
from unchain.agent import ToolsModule, SkillsModule
from unchain.skills import SkillsConfig
from unchain.toolkits import CoreToolkit

agent = Agent(
    name="assistant",
    provider="openai",
    model="gpt-5",
    modules=(
        ToolsModule(tools=(CoreToolkit(workspace_root="."),)),
        SkillsModule(
            SkillsConfig(
                project_root=".",
                reserved_commands=("btw", "fyi", "queue"),
            )
        ),
    ),
)

result = agent.run("Please /release-notes for the v2.3 tag.")
```

`SkillsConfig(project_root=".")` 会扫描 `./.unchain/skills` 和 `./.agents/skills`（rank 100/200），再加上用户的 `~/.unchain/skills` 和 `~/.agents/skills`（rank 400/500，因为 `include_user_dirs` 默认是 `True`）；`reserved_commands=("btw", "fyi", "queue")` 意味着即使某个 skill 不小心叫了这几个名字之一，也永远不会通过 `/name` 激活（仍然可以通过 `skill` 工具被列出和加载）。示例消息里的 `/release-notes` token 会直接从用户这一轮里解析并激活对应的 skill，不需要先经过一次模型往返。

## 已知限制

- 目录或激活状态每变一次，prompt-cache 前缀就会跟着变——这和任何其他会改写靠前 system 消息的 harness（比如 `ToolPromptHarness`）承担的是同一种代价。
- `skill` 工具是否存在，是每次 run 在 configure 时决定的（D7）：一个在运行期间才变成 model-invocable 的 skill（因为文件被编辑，或者 toolkit 变了），不会追溯性地给一个已经配置好的 agent 加上这个工具。
- 每个根目录只扫描一层——不支持递归的 `**/SKILL.md` 发现，这和参考实现（`dsh`）harness 的行为一致。
- `/name` 激活的轮次身份按持久 transcript 中真实用户消息的数量计数。运行中途若 transcript 被改写并移除了较早的用户消息（例如做摘要压缩），这个数量会变小；之后的新轮次只要文本不同仍会被识别，但紧接着发送字节相同的重复文本可能被当成上一轮的重放（漏掉一次重新激活，绝不会激活错的东西）。

## 相关 skills

- [Tool System Patterns](tool-system-patterns.md) —— 这个特性所依托的 Tool/Toolkit 定义
- [Architecture Overview](architecture-overview.md) —— `SkillsModule` 在 agent 构建流水线里的位置
- [Runtime Engine](runtime-engine.md) —— `KernelLoop`、`RuntimeHarness` 各阶段，以及 checkpoint/resume 的机制
