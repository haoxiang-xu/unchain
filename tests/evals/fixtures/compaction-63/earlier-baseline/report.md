已用真实模型完成测试。结论：当前 summary 实现不能判定通过。编码案例在明确的诊断配置下完成了三轮压缩和最终交付，但默认配置的格式兼容、摘要语义和精确回查均暴露问题。

实际调用模型为 **Claude Sonnet 4.6（claude-sonnet-4-6）**，通过已登录的 Claude CLI、关闭内置工具、每次只运行一轮推理；模型决定的工具调用交给实际 Unchain kernel 执行。OpenAI CLI 的两个连接探针均被后端拒绝为不支持的模型，没有算作 OpenAI 测试通过。没有读取或导出 API 密钥。

测试固定在候选 wheel，而非变化中的源码：
`/Users/red/.codex/artifacts/unchain-compression-implementation-20261023/wheels/unchain-0.2.0-py3-none-any.whl`
SHA256：`3f5e71a07d982372051604350a01ece985d630b02b7566ce598be1baab9944b1`。
本轮没有修改生产源码、没有提交或部署。

每个业务案例设三个连续压缩点，逐步完成任务，中途修正要求，最后要求恢复最早日志中的证据。每次压缩重新概括完整原始历史中要被移出的前缀，不是摘要再摘要。初始接手历史由真实本地文件读取及测试输出构造，两组相同；初始接手过程不算真实模型表现。后续摘要、工具选择和续做均调用真实模型。

默认配置先失败了：两次真实摘要都返回带 Markdown 围栏的 JSON，生产解析器直接拒绝。后续诊断的六次摘要也全部带围栏。因此下面的成功结果，依赖只剥去一层外部围栏的诊断适配，不能当作默认配置通过。

Agent 还有两层独立于 summary 的工具结果裁剪：历史结果的 mid_run_microcompact，以及新结果的 tool_result_budget。测试中原本约 2.7 万字符的结果被缩成预览，导致对照历史变化、第二轮压力未触发。正式诊断两组都禁用历史裁剪、把新结果预算提高到高于夹具大小。这里能确认“summary 拿到的不再是工具原文”；上游是否另有原文存储、是否能被当前 Agent 回查，仍需单独接好契约。不能宣称 summary 归档自动覆盖了这部分原文。

最终固定诊断（attempt-4）的结果：

| 案例 | Summary 条件 | 完整历史业务对照 |
| --- | --- | --- |
| 事故排查、验证、回放、报告 | 前两轮推进成功；第三轮摘要引用 ID 拼错，被校验器拒绝；没有生成最终报告 | 完成最终报告；6 项回放边界检查通过 |
| 修复代码、修正排序、测试、交付 | 三次压缩全部实际触发；最终交付完成；5 项公开测试及 100 组独立随机检查通过 | 同样完成；5 项测试和 100 组独立检查通过 |

这是两个受控案例的一次执行，不是成功率估计。Summary 组为测试未决审批保护，额外插入了一组显式审批请求/拒绝记录；完整历史组通过相同业务指令收到拒绝。不能把它当成逐条消息完全一致的因果消融。

具体保真度发现：

- **原文和链保存通过当前样本检查。** 独立检查了最终诊断中 5 份被接受的归档，逐条等于 canonical 原文前缀，摘要引用所指 source_id 存在，原文/内容哈希一致；另写解析器核对实际 Anthropic tool_use/tool_result 与审批记录，没有复用生产 `_chain`。调用关系两侧完整。这里检查的是原文与引用结构，不代表摘要每句话都得到证据支持。
- **关键事实大多保住，但不能给满分。** 事故中的错误码、证据哈希、proof nonce、实际证据 call ID、验证收据都正确。编码任务第三轮保住最新模块哈希、修正后的排序和 5 项测试通过状态。第二轮编码摘要没有明写“负数退款合法”这一约束，第三轮从完整原文中又恢复；未观察到最终代码因此出错。
- **未完成事项基本保住，但摘要会擅自加阻塞条件。** 两个案例均把“日志字段当时不属于交付内容”推成“必须让用户正式提供/授权这个字段，才能继续”。原任务没有这个额外要求。事故第一轮后的 worker 接受了这种说法；下一轮摘要再引用 worker 的话，反复强化。即使每次从完整原文重新摘要，先前摘要影响了后续 assistant 发言，错误仍可能进入 canonical 历史。
- **存在引用合法、却没有支持对应事实的情况。** 编码第二轮 open_items 中的新模块哈希是正确的，但该条只引用了不含此哈希的初始用户消息；真正支持它的是工具结果。当前校验器只验证引用 ID 存在，不能验证语义支持。
- **第三轮事故摘要被硬错误阻断。** 模型把合法的 `message-7d814946485eabf5-12` 写成 `message-7d814946485reabf5-12`。校验器正确拒绝，未把错误引用交给 Agent；没有修补该 ID、重跑到通过，或者把拒绝的原始输出计作成功摘要。
- **完成代码任务确实使用了归档。** Summary 编码组共发起 4 次真实 archive_read，最终阶段读取目录及原文分页，取回 audit_token，保存 delivery.json。最新排序要求得到执行，被拒绝的 P-publish 保持 denied。
- **但“原文存着”不等于能准确找回。** 另外预先固定了一个独立回查任务：从第三轮编码视图恢复最初代码、最初失败测试 stderr、哈希、实际读文件/测试 call ID。模型进行了 7 次归档读取，在 8 轮预算内回查失败。原始代码和读文件 call ID 正确；stderr、哈希和测试 call ID 则来自中途的 `live-019-0`，而非最初的 `setup-coding_handoff-3`。返回的 stderr 与后期失败结果逐字相同，证明是选错时间阶段，而不是单纯换行差异。正确最初结果仍完好在 source message 2 的后半段。没有延长预算或再次重试。
- **恢复/未决保护通过样本检查。** 5 次被接受的压缩都冷重开目录 store，复用相同视图且不再调用模型。两个案例在未决审批下都先拒绝压缩且调用模型 0 次；加入拒绝回应后再尝试摘要。事故之后因引用错误停止，不能把它记作第三轮成功。

保留的全部尝试与边界：

| 尝试 | 真实调用数 | CLI 标价估算 USD | 结果/用途 |
| --- | ---: | ---: | --- |
| attempt-1 | 13 | 0.219923 | worker 适配器对“文字加动作 JSON”过严；初始模型也未按夹具要求读完日志，未覆盖压缩压力。无合格的多轮保真度结论 |
| attempt-2 | 6 | 0.280389 | 默认 summary 两次都被围栏格式拒绝；对照触发默认历史裁剪，不满足完整历史假设 |
| attempt-3 | 28 | 1.515085 | 去掉单层围栏、关历史裁剪；新结果预算仍让后续压力不足；事故对照另有未明确 enum 的夹具缺陷，编码对照完成 |
| attempt-4 | 35 | 2.593287 | 修正诊断覆盖和明确既有 oracle enum 后固定执行；编码三轮完成，事故第三轮引用错误，两个业务对照完成 |
| 独立原文回查 | 8 | 0.388080 | 选错原始测试结果的时间阶段；失败保留，无重试 |

合计 90 次评估调用，CLI 标价估算 $4.996764；加上一次成功连通探针约 $4.998333。它是 CLI 回报的标价估算，不代表订阅实际扣费。最终诊断六次 summary 请求各约 19–25 秒，共估算 $0.699297。失败样本没有被后续成功覆盖，也没有合并平均值掩盖。

下一步应由这些失败样本驱动修正，优先级如下（本轮未实施）：

1. **把原文存储与上游裁剪串好。** 工具原始结果先留档、再投影到模型上下文；summary 和回查必须共用能够定位原始结果的引用。避免摘要层只能看到预览。
2. **让模型少手抄结构。** 采用明确的结构化输出契约，兼容可验证的外层围栏；把长哈希式 source_id 的生成与映射交给宿主，模型输出较短的受限引用。未知引用仍应拒绝；不要模糊纠正到“最像”的证据。失败应保住原历史并给可诊断结果。
3. **明确事实的来源与权限。** 用户要求、工具观察、assistant 判断分别记录；日志里的描述不能变成新的用户约束。把本次“额外授权条件”及“正确事实引用错来源”纳入回归，不只检查 JSON 和 ID。
4. **改善归档检索粒度。** 提供按 tool_call_id、工具名、文件路径、时间阶段检索，以及直接读取单个工具结果的能力。当前一个 source message 含多份结果，最初 stderr 藏在 3.3 万字符 JSON 后面，仅靠字符分页和大消息 source_id 容易取错。

修改后应重放同样失败样本，再增加不同模型和重复运行；本轮不声称统计可靠性、生产准入或原生 Anthropic/OpenAI SDK 传输兼容性。当前能证明的是：无损原文归档方案可工作，三轮后完成任务有成功实例，但摘要表述、引用生成和精确检索仍未达标。

证据入口：

- [最终矩阵](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/matrix-report.json)、[冻结计划](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/plan.json)、[逐次模型调用](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/call-ledger.json)
- [独立原文/调用链核对](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/independent-archive-review.json)、[逐轮语义审阅](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/semantic-review.json)
- [事故第三轮原始摘要](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/calls/007-summary/assistant.txt)、[实际模型输入](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/calls/007-summary/request.json)
- [代码交付](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/coding_handoff-summary_normalized/workspace/delivery.json)、[独立 100 组检查](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/attempt-4/coding_handoff-summary_normalized/workspace/independent-oracle.json)
- [独立回查结果](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/retrieval-probe/report.json)、[时间阶段错选分析](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/retrieval-probe/review.json)、[原文回查工具记录](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/retrieval-probe/tool-audit.json)
- [机器可读结论](/Users/red/.codex/artifacts/unchain-summary-fidelity-20261023/verdict.json)
