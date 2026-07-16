# 文档状态

更新时间：2026-07-16

本页区分当前规范、进行中的整理计划和历史实验记录。历史文档中的失败、候选参数或实验入口不能自动视为当前生产默认。

## 当前状态

项目文件与代码整理已经完成，最终状态为 `COMPLETE`。

- P2-A 审计和审批包已经生成；P2-D 实际归档或删除仍是可选动作，必须逐路径批准。
- P3 工具与 benchmark 边界已经完成并通过 dry-run、Ruff 和 smoke 验证。
- P6 Ruff 硬门和 mypy 评估已经完成。
- P5-0 approval revision 2 已按 `P5-0-A`、`P5-0-B`、`P5-0-D`、`P5-0-E` 顺序执行；四份 execution manifest 均为 `PASS`。`P5-0-C` 保持 `KEEP_LOCAL_FALLBACK`。
- P5 保留模块的职责拆分、重复 helper 审计和兼容性门禁已完成；P7 盘点、基线对比、文档一致性、生产 smoke 和收口报告也已完成。
- `reports/p5_0_compileall_environment_incident.json` 所列 8,931 个精确缓存路径已获得独立批准，逐路径备份并删除；剩余 0，删除目录 0。执行证据见 `reports/p5_0_compileall_environment_incident_execution.json`。
- 最新证据：MFA `3.4.0`、MFA local fallback `53 passed`、完整 `557 passed in 10.71s`、CLI/Web/质量门生产 smoke `68 passed`，Ruff、source-only compileall、CLI/import/help、review-scope 和 `git diff --check` 均通过。
- Web 前后端整理 W0-W8 已收口：版本化 API、Align 三态、22 个对白失败恢复队列、370 cue 审校草稿/撤销、质量证据跳转、受控预览下载、阶段启动、移动端导航与密钥零泄漏均已实现。
- 最新 Web 收口证据：Web 定向 `48 passed in 0.66s`，完整 `604 passed in 10.97s`；`scripts/local_check.ps1` 再次得到 `604 passed in 10.51s`；全范围 Ruff、compileall、Node 语法和 `git diff --check` 通过。
- Playwright 验证 1440×900、1280×720、390×844，控制台 0 error/0 warning、失败请求 0、媒体 Range 206、370 cue 缓存切换 33.7ms，字幕预览/下载 UTF-8 正常。
- Align 失败恢复 Goal A 已完成 A0-A7 的实现与真实数据验收：CLI/Web 共用真实 executor，Qwen retry 默认原 transcript，verified text 显式启用，MFA local 仅日语，coarse 受 transcript/VAD/邻接/短范围硬门保护，目标级 undo 保留后续其他恢复。
- Sayonara Lara 主对白从 `91 exact / 0 coarse / 22 failed` 改善到正式数据 `92 exact / 0 coarse / 21 failed`；唯一正式 exact 晋升为 `segment_000086`。18 个短应答全部真实执行 VAD，未核验文本没有写入 coarse。正式 quality 仍为 FAIL，时间异常为 0，normalized/export ASS 指标有一项真实改善。
- 双数据集定向回归 PASS：Konoato01 保持 `100 exact / 18 failed`，Madougushi02 保持 `116 exact / 11 failed`，未新增短对白缺失、内容守恒失败、非法时间、越界 token、非单调或严重重叠。
- Web 隔离验收副本验证 `segment_000013` 的核验 + coarse、`segment_000093` 的 VAD_NO_SPEECH 拒绝、`segment_000086` exact 筛选、音频 Range 206、刷新持久化和真实 quality-gate FAIL；该副本的 `1 coarse` 仅用于交互验收，不计入正式结果。证据见 `reports/align_recovery_web_acceptance.md`。
- Align 恢复最终验证：定向测试 `30 passed`；`scripts/local_check.ps1` 为 `616 passed in 11.66s`；独立完整 pytest 为 `616 passed in 10.88s`；compileall、全范围 Ruff、Node 语法、`git diff --check` 和 6/6 固定证据 SHA-256 均通过。

当前验收证据以 `reports/final_acceptance.md` 为准。该报告现为 `COMPLETE`，P0-P7 必做项和最终验收均已通过。

Web 专项验收以 `reports/web_frontend_backend_final_acceptance.md`、同名 JSON、`reports/web_api_contract_v1.json` 和 `reports/web_playwright_screenshot_index.md` 为准。

Align 恢复专项以 `reports/align_recovery_final_acceptance.md`、同名 JSON、`reports/align_recovery_sayonara_result.json`、`reports/align_recovery_dual_dataset_regression.json` 和 `reports/align_recovery_web_acceptance.json` 为准。

## 当前规范

以下文档描述当前仓库结构和操作方式：

- `README.md`：项目入口、默认策略、代码结构、验证和打包命令。
- `docs/ARCHITECTURE.md`：模块职责、本地状态边界和兼容规则。
- `docs/PIPELINE.md`：阶段顺序、产物、统一检查和可选 MFA 环境重建。
- `docs/WEBUI.md`：WebUI 模块、请求和状态接口。
- `docs/api-credentials.template.md`：凭证配置模板；本地凭证文件不进入版本控制。
- `docs/项目文件与代码整理计划.md`：当前整理范围、阶段门和验收要求。

若文档与运行行为冲突，以当前 CLI 参数解析、测试和实际代码为行为证据，并将文档冲突视为待修复问题，不能静默选择较方便的解释。

## 历史与证据文档

- `docs/字幕流程完整改造计划与验收清单.md`：字幕质量、MFA、LLM split、MiMo 和重对齐实验的历史计划及证据。文中的 `COMPLETED_REJECTED` 表示实验已完成但候选不得晋升，不表示相关实验入口仍应作为生产能力保留。
- `docs/Goal模式完整任务提示词.md`：历史任务编排和验收提示，不是当前运行参数或完成状态的权威来源。
- `docs/PERFORMANCE_OPTIMIZATION.md`：早期性能设计说明。涉及默认值、模块路径或性能结论时，应与当前 `README.md`、`docs/PIPELINE.md`、CLI 和 benchmark 报告交叉验证。

历史报告、工作区和 benchmark run 是可复现证据。未经明确路径级批准，不得移动、归档、删除或改写其结论。

## 状态更新规则

1. 当前默认值或阶段顺序变化时，同步更新 README、ARCHITECTURE、PIPELINE、CLI help 测试和 WebUI 文案。
2. 历史实验被退役时，保留结论和证据路径，在当前文档中删除生产入口描述，但不把历史失败改写成成功。
3. 每次最终验收更新都必须保留 `Status`、最新完整测试结果、未完成门和未获批准动作。
4. 未经授权的归档、删除、虚拟环境移动、stage、commit 和 push 不得因文档勾选而自动执行。
