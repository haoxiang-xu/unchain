# Unchain 百科文档

`unchain` 现在采用双层文档体系：仓库入口页保持精简，完整的中英双语百科正文覆盖 skills 章节和 `src/unchain` 下全部生产类。

Language switch: [English](README.en.md) | [简体中文](README.zh-CN.md)

## 阅读路径

- 先读 skills 章节建立架构和执行流，再进入 API 参考定位具体类。
- 当你需要全量覆盖检查、导出符号检索或返回结构速查时，使用附录。
- builtin toolkit 包内 README 故意保持简短，应把它们视为入口提示，而不是正式参考。

## Skills 章节

- [架构总览](zh-CN/skills/architecture-overview.md)
- [Agent 与 Subagents](zh-CN/skills/agent-and-team.md)
- [Runtime Engine](zh-CN/skills/runtime-engine.md)
- [工具系统模式](zh-CN/skills/tool-system-patterns.md)
- [Memory 系统](zh-CN/skills/memory-system.md)
- [创建内置 Toolkit](zh-CN/skills/creating-builtin-toolkits.md)
- [Agent Skills](zh-CN/skills/agent-skills.md)
- [测试约定](zh-CN/skills/testing-conventions.md)

## 操作指南

- [添加 Kernel Harness](zh-CN/guides/add-harness.md)
- [添加新模型](zh-CN/guides/add-model.md)
- [添加新 Provider](zh-CN/guides/add-provider.md)
- [SAP Hyperspace Provider](zh-CN/guides/hyperspace-provider.md)
- [添加新工具](zh-CN/guides/add-tool.md)
- [添加新 Toolkit](zh-CN/guides/add-toolkit.md)
- [调试流式问题](zh-CN/guides/debug-stream.md)
- [探索架构](zh-CN/guides/explore.md)
- [同步包](zh-CN/guides/sync-packages.md)
- [运行测试](zh-CN/guides/test.md)

## API 参考

- [Agents API 参考](zh-CN/api/agents.md)
- [Runtime API 参考](zh-CN/api/runtime.md)
- [Interaction API 参考](zh-CN/api/interaction.md)
- [耐久任务 API 参考](zh-CN/api/jobs.md)
- [工具系统 API 参考](zh-CN/api/tools.md)
- [Toolkit 实现参考](zh-CN/api/toolkits.md)
- [Memory API 参考](zh-CN/api/memory.md)
- [Input、Workspace 与 Schema 参考](zh-CN/api/input-workspace-schemas.md)

## 附录

- [类索引](zh-CN/appendix/class-index.md)
- [导出索引](zh-CN/appendix/export-index.md)
- [术语表](zh-CN/appendix/glossary.md)
- [返回结构与状态流速查](zh-CN/appendix/return-shapes-and-state-flow.md)

## 覆盖承诺

- `src/unchain` 下的生产类会按参考领域建立索引。
- 各包 `__init__` 中的公开导出会被交叉链接到参考树。
- 中英文文档保持相同的章节和页面布局。
