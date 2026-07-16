# Web 前后端整理 W0 基线审计报告

审计编号：`web-w0-20260716-104206`
结论：W0 基线盘点与行为冻结通过，已继续进入 W1/W2 第一轮实现。

## 1. 当前实现基线

- 后端使用 `http.server.ThreadingHTTPServer`，路由、任务进程、文件选择、删除和工具动作集中在 `qwen_asr/web/server.py`。
- 命令构造集中在 `qwen_asr/web/commands.py`，继续调用 CLI；CLI 和 `PipelineRunner` 仍是业务规则权威来源。
- 工作区状态集中在 `qwen_asr/web/status.py`，数据来自 manifest 是否存在、数量、`ArtifactState`、`progress.json` 和日志尾部。
- 页面为 `qwen_asr/web/templates/index.html` 单文件，当前大小 81,362 字节，HTML、CSS、请求和状态渲染耦合。
- 旧接口没有统一版本、响应 envelope、稳定错误码和 API 契约文件。

## 2. 旧接口冻结清单

GET：`/`、`/favicon.ico`、`/api/status`、`/api/job`、`/api/suggest-workdir`、`/api/workspaces`。

POST：`/api/start`、`/api/stop`、各类文件/目录选择接口、单个/批量工作区删除、glossary 标准化。

旧接口在本轮保持兼容；新增结构化能力放在 `/api/v1/*`，避免通过破坏旧 payload 推进重构。

## 3. 功能盘点

真实可用：命令构造、任务启动/停止、状态轮询、路径选择、工作区删除保护、阶段产物计数、日志尾部、glossary 标准化。

部分可用：质量门只有摘要而无证据跳转；导出只有参数配置而无结构化产物审阅；刷新可恢复落盘产物，但活动任务仍依赖服务进程内存；旧工作区列表按四位数字前缀过滤，不能覆盖全部有效证据工作区。

缺失：Align 三态、OP/ED 排除、完整失败对白恢复队列、短应答优先、cue/播放器/参考字幕联动、恢复操作审计、结构化质量证据与受控下载。

已否定且不得整体恢复：生产 LLM split、全量 MFA 主对齐、质量门自动绕过。依据为 `reports/rejected_feature_inventory.*` 和现有整理文档。

## 4. 机器可读契约

契约快照：`reports/web_api_contract_v1.json`。

本轮新增 `/api/v1/contract`、`/api/v1/workspaces`、`/api/v1/workspace`、`/api/v1/workspace/align`、`/api/v1/workspace/recovery`、`/api/v1/workspace/quality`、`/api/v1/workspace/exports`。

统一响应包含 `api_version`、`schema_version`、`data`、`error`。路径只允许 `workspaces` 下一级目录；损坏 JSON 返回结构化状态；密钥不进入任何响应字段。

## 5. 基线测试

首次修改前运行：

`84 passed in 2.89s`

覆盖 `tests/test_webui.py`、`test_pipeline_runner.py`、`test_artifact_state.py`、`test_align_quality.py`、`test_export_paths.py`。

第一轮实现后：新增 API 测试与既有 Web 测试共 `51 passed in 0.20s`；Ruff、compileall 和 `git diff --check` 通过，仅保留工作树既有 CRLF 提示。

## 6. 真实工作区核对

只读调用 `workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234`：

- 原始 Align：128 输入，104 `completed_exact`，0 `completed_coarse`，24 `failed`；
- 排除 OP/ED：113 对白，91 `completed_exact`，22 `failed`；
- 音乐区域：15 个 segment、36 个 cue；
- recovery：22 个失败对白全部入队，18 个短应答优先；
- normalized：370 cue；
- 后置精确恢复：0；
- quality：`FAIL`，导出存在时仍保持 `quality_gate_failed`。

上述数字均由 manifest、项目 metadata 和报告结构推导，没有写入样例区间、片段编号或对白文本硬编码。

## 7. 安全与后续

本轮未修改两个保留工作区、历史产物、媒体、模型、虚拟环境和用户已有改动；未 stage、commit 或 push。备份位于 `backups/web-w0-w1-api-20260716-104206`。

下一步继续 W1–W3：把任务状态从服务进程内存迁移到可恢复结构化状态，补齐 API schema/中断恢复测试，并实现恢复动作、审计记录和下游失效机制。
