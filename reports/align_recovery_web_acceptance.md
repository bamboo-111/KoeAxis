# Align 恢复 Web 验收

- 状态：PASS
- 范围：仅隔离验收副本 `align-recovery-web-acceptance-20260716-160300`
- 正式 Sayonara 工作区：未修改

## 验收结果

- 页面与工作区加载正常；旧 Web 进程造成的 HTML/JS 版本混用通过重启监听进程解决，无需修改源代码。
- 恢复页显示精确、粗略、失败和短应答指标，根因筛选 `short_response` 在验收操作后返回 17 条。
- `segment_000013` 先保存人工核验文本 `うん。`，再使用唯一 VAD 区域 `153407–154344 ms` 接受为 `completed_coarse`；确认框、备份、事件和刷新持久化均正常。
- 验收副本状态从 `92 exact / 0 coarse / 21 failed` 变为 `92 exact / 1 coarse / 20 failed`。该 coarse 仅用于 Web 交互验收，不属于正式数据结论。
- `segment_000093` 的局部 VAD 返回预期的 HTTP 409 / `VAD_NO_SPEECH`，页面展示错误信息且未崩溃、未改写 manifest。
- 字幕审校 `exact` 筛选可看到 `segment_000086` 的 3 个 `completed_exact` cue：`何？`、`足が息が`、`苦しい。`。
- 音频媒体请求返回 HTTP 206 Partial Content。
- 从流水线页真实启动 `quality-gate` 后，进程以返回码 1 结束；这是质量门 FAIL 的预期语义。刷新后仍为 `7 PASS / 2 WARN / 6 FAIL`，对齐健康为 `92 exact / 1 coarse / 20 failed / 0 timing errors`。
- 唯一浏览器控制台错误是故意触发的 VAD 409 网络错误；未发现其他控制台错误或失败请求。

## 证据

- 恢复页截图：`output/playwright/align-recovery-a7/recovery-coarse-and-failed.png`
- 质量页截图：`output/playwright/align-recovery-a7/quality-fail-after-web-run.png`
- Playwright trace：`output/playwright/align-recovery-a7/trace/trace-1784189476133.trace`
- 网络日志：`output/playwright/align-recovery-a7/trace/trace-1784189476133.network`
- 结构化报告：`reports/align_recovery_web_acceptance.json`

结论：A7 的必要 Web 操作链已由真实浏览器覆盖，质量门没有被绕过或伪装为 PASS。
