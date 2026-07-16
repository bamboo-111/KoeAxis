# Align 失败恢复 Goal A 最终验收

- 状态：PASS（Goal A 完成）
- 正式业务质量门：FAIL
- 正式 Sayonara 结果：`92 exact / 0 coarse / 21 failed`
- Web 验收副本：`92 exact / 1 coarse / 20 failed`，其中 coarse 只用于交互验收

## 最终结论

Goal A 可以验收完成。既有恢复能力已经从“代码存在或只记录请求”收口为 CLI、Web/API 共用的真实单段 executor；所有适用动作均在派生工作区执行，失败保留原状态，成功写回前备份，目标级撤销不会覆盖后续其他恢复。完成 Goal A 不表示字幕业务质量门已通过：正式 quality 仍为 FAIL，21 条剩余失败继续保留，没有通过放宽 coverage、伪造 exact 或批量 coarse 来归零。

## A0–A2：基线、能力与根因

- 6/6 固定证据文件的大小和 SHA-256 全部 MATCH；两个只读 Sayonara 基线工作区未修改。
- 基线为 128 输入、104 completed、24 failed；排除 OP/ED 后为 91 exact、0 coarse、22 failed，其中 18 条是短应答。
- 22/22 失败都有逐条执行轨迹；根因汇总为：19 条 Qwen timing 失败后又被 short-window 内容守恒拒绝，2 条 token coverage 低于安全门，1 条语言路由不匹配。
- 参考 ASS 始终只读，没有作为 transcript、Qwen 或 MFA 输入。

## A3–A4：生产执行链与 coarse 硬门

- 新增 `qwen_asr/recovery_executor.py` 和 `recover-align` CLI；Web `retry_align` 调用相同 executor，不再只写 `retry_requested`。
- Qwen retry 默认使用原 transcript；verified text 必须先核验并显式启用。
- `route_language` 真实影响 backend；MFA local 仅用于日语候选。
- Qwen/MFA exact 都通过统一 token timing/content validator。
- aligned manifest、checkpoint、events 写回前备份；失败执行保留原 manifest；工作区锁、冲突、中断清锁、幂等和重复请求均有测试。
- coarse 要求 transcript 已核验，VAD 唯一区域或人工明确选择，区间在原段内、正时长、短应答不扩成长段、邻接重叠通过硬门，并保留原失败、VAD、操作者和备份证据。
- `undo_recovery` 只恢复目标 segment，保留后续其他恢复。

## A5：Sayonara 真实恢复

恢复前为 91 exact / 0 coarse / 22 failed。真实 Qwen 原 transcript retry 对失败条目逐条执行；适用的 MFA local 候选也真实执行。

临时曾得到 93 exact / 20 failed，但正式质量复核发现 `segment_000036` coverage 为 0.164931，低于 0.20 门槛，因此目标级撤销。最终只保留：

- `segment_000086`
- 文本：`何？足が息が苦しい。`
- backend：`mfa-local-recovery`
- content score：1.0
- coverage：0.453669
- 统一 timing validator：PASS

正式结果为 92 exact / 0 coarse / 21 failed。18 个短应答全部真实执行 VAD：9 个唯一单区域、5 个多区域需人工选择、4 个无语音；未经 transcript 核验的条目没有写入 coarse。

## A6：回归与下游

- Konoato01：100 exact / 18 failed 前后不变。
- Madougushi02：116 exact / 11 failed 前后不变。
- 两套数据均未新增短对白缺失、内容变化、非法时间、越界 token、非单调或严重重叠。
- Sayonara Split 从 370 变为 371；Translate 47/47 批次、371/371；MiMo 124/124 完成、9 条应用、0 unresolved；proofread-realign 4 candidates、3 completed、1 fallback、0 failed。
- 正式 quality 仍为 FAIL：6 PASS / 3 WARN / 6 FAIL；alignment health 为 92 exact / 0 coarse / 21 failed，时间异常 0；normalize/export 内容检查和 SRT 合法性 PASS。
- 诊断 normalize/export 明确标记 `diagnostic_only=true`、`quality_gate_bypassed=true`，只在隔离派生工作区生成 371 条，不冒充正式交付。
- ASS 指标有有限真实改善：split/mimo mean 0.813372、`<0.20` 17；normalized/export mean 0.818335、`<0.20` 16。但四份 ASS 报告总体仍 FAIL。

## A7：Web/API 验收

Playwright 真实浏览器完成以下代表流程：

- `segment_000013`：保存核验文本后，以唯一 VAD 区域接受为 coarse；刷新后状态持久化。
- `segment_000093`：局部 VAD 返回预期 HTTP 409 / `VAD_NO_SPEECH`，页面未崩溃、manifest 未被改写。
- `segment_000086`：审校 exact 筛选可见 3 个 completed_exact cue。
- 音频 Range 请求返回 HTTP 206。
- 从流水线页真实启动 quality-gate；返回码 1 对应诚实的质量 FAIL，刷新后仍显示 FAIL。
- 除故意触发的 VAD 409 外，未发现意外控制台错误或失败请求。

截图、trace 和网络日志位于 `output/playwright/align-recovery-a7/`；结构化证据见 `reports/align_recovery_web_acceptance.json`。

## 最终验证

- 定向 pytest：30 passed in 2.41s
- `scripts/local_check.ps1`：616 passed in 11.66s，Ruff 与 diff 检查通过
- 独立完整 pytest：616 passed in 10.88s
- compileall、Ruff、Node 语法、`git diff --check`：PASS
- 实时 `/api/v1/contract` 与 `reports/web_api_contract_v1.json`：语义一致
- 固定证据 SHA-256：6/6 MATCH
- 密钥模式扫描：0 命中

## Goal B 候选

1. 短应答的音频定位与 transcript 可信确认：18 条短应答仍 failed，VAD 唯一区域并不等于文本可信。
2. 英语/混合语言失败段的生产 backend 与路由：已有 1 条明确语言路由不匹配，MFA local 按设计仅支持日语。
3. 剩余 21 条低 coverage/dense-zero 的新对齐方法：既有 Qwen、short-window 与适用 MFA 已真实执行，当前拒绝来自可测量的内容/时间硬门，不是接线缺失。

最终判定：接受 Goal A；质量 FAIL 和 Goal B 候选均如实保留。
