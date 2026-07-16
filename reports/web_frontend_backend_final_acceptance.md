# Web 前后端整理最终验收报告

生成时间：2026-07-16 12:21:39 +08:00

状态：`PASS`（实现与验收通过；真实 Sayonara Lara 样例的业务质量门按证据继续保持 `FAIL`）

## 1. 交付范围

- 默认 `/` 已从 81 KB 兼容单页切换为拆分的结构化工作台；旧配置页保留在 `/legacy`。
- API 使用 `api_version / schema_version / data / error` 稳定 envelope，契约见 `reports/web_api_contract_v1.json`。
- 工作区、阶段、任务、Align、恢复、审校、质量、证据、媒体、导出均有版本化接口。
- 任务状态持久化到 `workspaces/.web-state/job.json` 和工作区 `reports/web_job.json`；服务重启后可把失联 PID 纠正为 `interrupted`。
- Align 共享状态为 `completed_exact / completed_coarse / failed`；音乐区域单独使用 `SKIPPED_MUSIC_REGION`。
- 所有 failed 对白进入恢复队列；音乐区域不计入对白失败。恢复服务复用生产 VAD，并记录 actor、时间、输入、策略、前后状态和证据。
- 审校支持 370 cue、整集音频定位、日中只读 ASS 对照、独立草稿保存、revision 冲突、时间校验、自动备份、JSONL 审计和上次编辑撤销。
- 脏审校草稿不会静默覆盖正式 manifest；quality-gate、normalize、export 会显示 outdated。
- 质量 WARN/FAIL 均有恢复项、cue 或受控报告目标；报告只能从当前质量 inventory 打开。
- 导出列表、UTF-8 预览和 attachment 下载均受 inventory 约束；导出成功不会覆盖质量门 FAIL。
- 新工作台可从阶段表继续可独立运行阶段；后端复用 CLI `build_command`，先检查输入和环境凭据，拒绝嵌套凭据字段。

## 2. 真实数据验收

只读验收工作区：

- `workspaces/full-regression-sayonara-lara-02-20260715-234059`
- `workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234`

已验证：

- 原始 Align：128 输入，104 完成，24 失败。
- 排除 OP/ED 后：113 对白，91 `completed_exact`，0 `completed_coarse`，22 `failed`。
- 音乐区域：15 个 segment、36 个 cue，状态为 `SKIPPED_MUSIC_REGION`。
- 恢复队列：22 个对白失败，18 个短应答优先。
- 审校/Normalize/Export：370 cue，非正时长 0，重叠 0。
- 后置修复精确恢复仍为 0/24；Web 没有伪装为成功。
- 最终质量：7 PASS、3 WARN、4 FAIL；正式状态继续为 `FAIL`。
- 导出 inventory：5 个 SRT/VTT/normalized 产物，全部保持 `quality_gate_failed`。

未对两个保留工作区执行审校保存、恢复写动作或阶段启动；浏览器验收只读。写路径由临时测试目录的单元/API 测试覆盖。

## 3. API 与安全

新增或冻结的主要接口：

- `GET /api/v1/contract`
- `GET /api/v1/job`
- `GET /api/v1/workspaces`
- `GET /api/v1/workspace`
- `GET /api/v1/workspace/stages`
- `POST /api/v1/workspace/stage/start`
- `GET /api/v1/workspace/align`
- `GET /api/v1/workspace/recovery`
- `POST /api/v1/workspace/recovery/action`
- `GET /api/v1/workspace/review`
- `POST /api/v1/workspace/review/edit`
- `POST /api/v1/workspace/review/undo`
- `GET /api/v1/workspace/media`
- `GET /api/v1/workspace/quality`
- `GET /api/v1/workspace/quality-evidence`
- `GET /api/v1/workspace/exports`
- `GET /api/v1/workspace/export-file`

安全证据：

- 兼容页不再包含 API 凭据输入、DOM 字段或启动 payload 字段。
- 当前 localStorage 设置会过滤凭据型字段；历史 v2/v3 设置迁移后删除。
- Web 启动 payload 和嵌套 stage settings 携带凭据字段时，在创建进程前返回结构化 400。
- MiMo 使用 `MIMO_API_KEY`，DeepSeek 官方地址使用 `DEEPSEEK_API_KEY`，其他兼容接口使用 `LLM_API_KEY`。
- 命令持久化仍执行敏感参数脱敏；版本化响应、HTML、截图和报告不含凭据值。
- 工作区必须是 `workspaces/` 下一级目录；媒体仅允许所选或 manifest-linked 工作区；质量/导出文件必须来自当前 inventory。

## 4. 自动化测试

- Web 定向 pytest：`48 passed in 0.66s`。
- 全量 pytest：`604 passed in 10.97s`。
- `scripts/local_check.ps1`：`604 passed in 10.51s`，并通过 Ruff 与 `git diff --check`。
- 全范围 Ruff：`All checks passed!`。
- `python -m compileall qwen_asr optimizer tests -q`：通过。
- `node --check qwen_asr/web/static/workbench.js`：通过。
- `git diff --check`：通过；仅保留工作树既有 CRLF 提示。

新增定向覆盖包括：

- API envelope、错误码、UTF-8、损坏 JSON、非法路径。
- 任务中断/重启恢复和命令脱敏。
- Align 三态与音乐区域排除。
- 22 项恢复队列、VAD/语言/重试/coarse 动作及备份。
- 审校草稿、revision、重叠、备份、审计、撤销和正式源不变。
- 质量 cue/恢复/报告目标与受控证据路径。
- 阶段启动 CLI payload parity、缺输入、缺环境凭据、嵌套凭据拒绝和结构化 job 响应。
- SRT/VTT UTF-8 预览、download attachment 和 Range media。

## 5. Playwright 验收

- 视口：1440×900、1280×720、390×844。
- 控制台：0 error、0 warning。
- 网络：失败请求 0；音频请求 206 Partial Content。
- 390px body 宽度 375px，导航按钮和质量操作均位于视口内。
- 1280px body 宽度 1265px，无横向溢出。
- 370 cue 缓存视图切换：33.7ms。
- 3.5 秒 job 轮询期间 cue DOM、370 行和 scrollTop=900 均保持不变。
- 键盘 Tab 顺序：跳到主要内容 → 流水线 → 恢复队列 → 字幕审校；焦点 outline 均为 solid。
- 对比度：按钮 13.56、FAIL 7.58、muted 6.61、焦点色 7.72。
- 字幕预览：200、`text/plain; charset=utf-8`、日文/中文均存在。
- 字幕下载：200、`Content-Disposition: attachment`、34034 bytes。
- 质量报告预览：200、`application/json`、UTF-8；readability 跳到 Cue 353，播放器定位 1351.199 秒。

截图索引见 `reports/web_playwright_screenshot_index.md`。

## 6. 风险与明确边界

- 真实样例仍有 22 个对白 Align 失败，因此业务质量门正确保持 FAIL；这不是 Web 验收失败。
- `retry_align` 当前记录结构化请求；实际模型执行仍通过现有阶段任务入口调度，不在浏览器只读验收中启动 GPU 任务。
- `proofread-realign` 由 MiMo 流程内部管理，不暴露伪造的独立启动按钮。
- 审校草稿不会自动晋升为正式产物；用户必须在明确的后续流水线策略中处理草稿，避免静默覆盖。
- 工作树包含用户既有大量修改和新增文件；本轮未 reset、restore、stage、commit 或 push。
- 历史工作区、媒体、模型、虚拟环境和既有报告未删除或整理。

## 7. 产物

- `reports/web_frontend_backend_final_acceptance.md`
- `reports/web_frontend_backend_final_acceptance.json`
- `reports/web_api_contract_v1.json`
- `reports/web_playwright_screenshot_index.md`
- `reports/web_w0_audit.md`
- `reports/web_w0_audit.json`
