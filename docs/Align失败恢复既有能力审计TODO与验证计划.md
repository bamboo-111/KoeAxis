# Align 失败恢复既有能力审计 TODO 与验证计划

## 1. 目标与边界

本文是下一轮 Goal A 的唯一权威计划。目标是复用此前已经实现和验证过的能力，查明它们为什么没有恢复 Sayonara Lara 的 22 个非 OP/ED 对白 Align 失败，并把缺失的生产执行链路接通。

本计划不是新一轮全量 Align 调参，不要求为了数字归零把所有条目强制标记为 coarse，也不承担最终网页视觉重做。预计执行 3-6 小时；8 小时是方案检查点。

## 2. 固定输入和保护规则

- [x] 读取并验证 `reports/align_recovery_goal_baseline.json` 的提交、指标和证据 SHA-256。
- [x] 原始完整回归工作区只读：`workspaces/full-regression-sayonara-lara-02-20260715-234059/`。
- [x] 后置诊断工作区只读：`workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234/`。
- [x] 所有真实写入实验必须复制到新的派生工作区，禁止覆盖上述两个基线。
- [x] OP/ED 使用 `SKIPPED_MUSIC_REGION`，不计入对白恢复成功率。
- [x] 参考 ASS 只用于只读评估和人工比较，不得作为 transcript、Qwen Align 或 MFA 输入。
- [x] 不重新转录整集，不重复完整下游 LLM，除非受修改影响且进入最终晋升验收。
- [x] 不恢复 LLM Split、MFA 全量替代或已退役 Align 变体。
- [x] 不把整个 MiMo/通用 LLM 校对提前到 Align 前。

## 3. 阶段 TODO

### A0 基线冻结与执行副本

- [x] 校验 6 个固定证据文件的大小和 SHA-256。
- [x] 记录当前 Git 分支、提交、脏工作树和虚拟环境。
- [x] 建立新的只含必要 manifest、音频引用和报告的派生实验工作区。
- [x] 验证副本仍为 128 输入、104 完成、24 失败；排除音乐后为 91 `completed_exact`、0 `completed_coarse`、22 `failed`。
- [x] 生成 `reports/align_recovery_a0_baseline.json/.md`。

验收：原始两个工作区哈希不变；新实验可重复创建；未运行 GPU/LLM 前基线指标完全一致。

### A1 既有能力与生产可达性矩阵

- [x] 盘点 Qwen 主 Align 的调用入口、失败守卫和输出状态。
- [x] 盘点 `asr-short-window` 的触发条件、窗口结果、内容守恒拒绝和 24 条实际执行记录。
- [x] 盘点 coverage、零时长 token、dense-zero、局部插值和短应答相关参数，关联 P6 九变体结论。
- [x] 盘点 MFA local 候选、runner、内容/时间守卫、写回和启动成本。
- [x] 盘点 proofread-realign 中已成功的 Qwen clamp、MFA local 和 mixed-language original-timing 边界，但不把它误用于首次 Align。
- [x] 盘点 Web 恢复队列动作到生产 executor 的调用链。
- [x] 为每项输出 `IMPLEMENTED_REACHABLE`、`IMPLEMENTED_NOT_REACHABLE`、`REJECTED_BY_EVIDENCE`、`NOT_APPLICABLE` 四选一结论。
- [x] 生成 `reports/align_recovery_capability_matrix.json/.md`。

验收：所有既有能力均有代码符号、CLI/Web 入口、默认值、测试、历史证据和实际可达性，不再以“记得以前做过”代替证据。

### A2 22 条失败执行轨迹和根因

- [x] 对 22 个主对白失败逐条记录原 transcript、规范化字符数、语言、原时间窗、token、coverage、错误和邻接上下文。
- [x] 读取 short-window fallback 的每个窗口 ASR 文本、Align 状态、token 数和内容拒绝原因。
- [x] 区分 transcript 正确性未知、transcript 已确认正确、ASR 改写、无 token、低 coverage、零时长/密集零 token、语言不匹配、邻接吸附和音频不可辨识。
- [x] 18/18 个短应答全部进入 `short_response` 路由，不得仅按高置信子集抽样。
- [x] 每条记录 `attempted_capabilities`、`skipped_capabilities`、`skip_reason`、`final_root_cause` 和 `confidence`。
- [x] 对不确定 transcript 生成待人工音频核验清单，不自动用参考 ASS 替换。
- [x] 生成 `reports/align_recovery_failure_trace.json/.md`。

验收：22/22 有唯一主根因和完整执行轨迹；不存在无法解释的 no-op 或仅写状态未执行。

### A3 生产恢复执行器接线

- [x] 将 `retry_align` 从状态记录改为真实、可测试、可审计的 executor 调用。
- [x] executor 只消费失败对白和已批准策略，不重新运行整集。
- [x] Qwen retry 默认复用原 transcript；若存在人工 verified text，显式记录来源和前后差异。
- [x] short-window ASR 只能提供音频定位/核验证据，不能静默改写 transcript。
- [x] `route_language` 必须影响实际策略选择；没有可用 backend 时明确返回不可执行原因。
- [x] MFA local 仅在日语、环境可用、候选适用和守卫通过时执行。
- [x] 每次执行写入策略、命令/参数、耗时、状态、token、coverage、内容守恒、时间合法性和失败原因。
- [x] 所有写回前备份 aligned manifest、checkpoint 和 event log。
- [x] 增加幂等、并发冲突、进程中断和重复请求测试。

验收：Web/API/共享服务调用同一个 executor；真实动作不是“请求已记录”；失败仍保留原状态和证据。

### A4 短应答 VAD 与 coarse 安全门

- [x] VAD 定位必须保留 backend、阈值、region、原 segment 范围和耗时。
- [x] `completed_coarse` 前要求 transcript 已人工核验或由明确可信规则确认，不能仅凭 VAD 有语音就接受。
- [x] coarse 区间必须在原 segment 内、正时长、与未修改邻居不严重重叠且不改变文本顺序。
- [x] 短应答区间不得被扩展为整个长 segment 来掩盖定位失败。
- [x] 若多个 VAD region 无法唯一对应 transcript，保持 failed 并进入人工选择，不自动合并全部 region。
- [x] coarse 必须保留原失败原因、接受证据、操作者和可撤销备份。
- [x] final quality 对 coarse 保持 WARN 语义，不伪装成 exact。
- [x] 增加边界、邻接、空 transcript、多 region、重复接受和撤销测试。

验收：每个 coarse 都能回答“文本为什么可信、音频范围为什么可信、为什么不会吞并邻句”。

### A5 定向真实恢复与首次晋升

- [x] 先在派生 Sayonara Lara 工作区对 22 条执行，不重跑 ASR。
- [x] 每轮只改变一个策略组：生产接线、Qwen 原文 retry、MFA local、VAD/coarse 或语言路由。
- [x] 输出恢复前后 exact/coarse/failed、短应答失败、内容守恒、非法时间、重叠和耗时。
- [x] 不允许以放宽 coverage 阈值或状态映射作为恢复收益。
- [x] exact 必须有可靠 token 内容和时间；coarse 必须通过 A4 硬门。
- [x] 未解决条目保留 failed，并记录是否需要新算法 Goal B。
- [x] 生成 `reports/align_recovery_sayonara_result.json/.md`。

验收：所有可由既有能力解决的条目均已实际处理；failed 有可测量下降，或每个未下降条目都有可复核的证据拒绝结论。不能只给建议。

### A6 双数据集回归与下游重算

- [x] 从既有 Konoato01 和 Madougushi02 证据解析固定输入，不猜测路径。
- [x] 只运行受修改影响的定向恢复和必要下游。
- [x] 两套数据不得新增短对白缺失、内容守恒 FAIL、非法时间、严重重叠或 `<0.20` 高置信回退。
- [x] Sayonara Lara 恢复后重跑 Split、必要翻译/复核、quality-gate、Normalize 和 Export；无关 LLM 阶段可复用未失效产物。
- [x] 正式 quality 只能按真实重算结果变化；若仍 FAIL，报告剩余阻塞，不强制改写状态。
- [x] 生成双数据集晋升报告和 Sayonara Lara 最终质量对比。

验收：至少一套目标指标改善，其他验证集无关键回退；否则本轮候选不晋升，但保留失败报告。

### A7 Web/API 收口与最终验收

- [x] Web 恢复队列展示实际执行状态、耗时、策略、退出原因和前后状态。
- [x] 支持 exact/coarse/failed 筛选和失败根因筛选。
- [x] 页面刷新后从工作区状态恢复，不依赖内存或日志推断。
- [x] 不进行全站视觉重构，只补真实恢复工作流必要字段和控件。
- [x] 使用 Playwright 验证代表性 short response、exact success、coarse acceptance、failed rejection 和质量重算。
- [x] 更新 README、ARCHITECTURE、PIPELINE、STATUS 和 Web/API 契约。
- [x] 生成 `reports/align_recovery_final_acceptance.json/.md`。

验收：浏览器操作和 CLI/API 得到相同结果；没有只改显示、不改 manifest 的假恢复。

## 4. 每轮验证

最低命令：

```powershell
.\.venv312\Scripts\python.exe -m compileall qwen_asr optimizer tests -q
.\.venv312\Scripts\python.exe -m pytest tests/test_align_cleanup.py tests/test_alignment_state.py tests/test_recovery_service.py tests/test_web_workspace_api.py -q
.\.venv312\Scripts\python.exe -m ruff check qwen_asr optimizer tests
node --check qwen_asr/web/static/workbench.js
git diff --check
```

阶段收口：

```powershell
.\scripts\local_check.ps1
```

此外必须验证：

- [x] 22 条真实失败执行轨迹完整；
- [x] 两个保留工作区哈希不变；
- [x] 派生工作区具备执行前备份和审计；
- [x] 密钥不进入 API、日志、报告和 Git diff；
- [x] Web 真实浏览器除故意触发的 `VAD_NO_SPEECH` HTTP 409 外无控制台错误和失败请求；
- [x] exact、coarse、failed 和音乐排除状态语义一致。

## 5. Goal A 完成定义

只有 A0-A7 全部通过才可完成 Goal。完成不要求 22 条全部 exact，也不预设业务质量门必须 PASS；但必须完成既有能力审计、22 条逐条归因、真实 executor 接线、所有适用恢复动作、双数据集晋升判断、下游重算、Web 验收和中文报告。

以下任一情况存在时不得完成：仍有恢复按钮只写状态不执行；仍有失败条目没有执行轨迹；coarse 缺少 transcript 和时间证据；为了归零放宽内容/时间硬门；重复此前无收益参数矩阵；原始工作区被覆盖；只运行单元测试而没有真实数据。

若审计后确认剩余项需要新的定位或多语言算法，必须列出 Goal B 的精确候选和证据，但不要在 Goal A 中无边界扩展。
