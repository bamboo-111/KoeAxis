# Web 前后端整理 TODO 与验证计划

## 1. 文档定位

本文是下一轮 Web 前后端整理的唯一权威计划和验收清单。目标不是先做视觉改版，而是先让现有 CLI、流水线状态、失败恢复、质量证据和导出产物完整接入 Web，再在功能闭环稳定后重构页面结构与交互。

本轮基线不恢复已经被证据否定并退役的实验功能，不把完整 MiMo/LLM 校对阶段提前到 Align 之前。新增能力应围绕独立的失败片段恢复层建设：

```text
Transcribe
  -> Correct
  -> Align pass 1
  -> Failed-segment recovery
       -> mark/skip OP/ED
       -> route every failed dialogue segment
       -> audio transcript verification
       -> short-response VAD/localization
       -> language routing where needed
       -> retry align
       -> completed_coarse fallback
  -> Split
  -> Translate
  -> MiMo suspect review
  -> Post-edit realign
  -> Quality gate
  -> Normalize
  -> Export
```

## 2. 当前基线

- Web 后端主要集中在 `qwen_asr/web/commands.py`、`server.py` 和 `status.py`。
- 页面主要集中在约 1701 行的 `qwen_asr/web/templates/index.html`，功能、状态和样式耦合较重。
- `tests/test_webui.py` 已覆盖命令构造、工作区状态、文件选择、删除保护和页面关键控件，可作为兼容基线。
- CLI 和流水线仍是业务规则权威来源；Web 不应复制阶段规则或通过解析人类日志推断核心状态。
- Sayonara Lara 02 真实样例：128 个 ASR 片段，正式 Align 104 完成、24 失败。
- 排除 OP/ED 后：113 个对白片段，91 完成、22 失败；18/22 是不超过 4 个规范化字符的短应答。
- OP 区间为 57.150-131.850 秒，ED 区间为 1314.480-1401.170 秒；样例中应展示 15 个排除片段和 36 个排除 cue。
- 后置 Normalize/Export 得到 370 个合法 cue、无非正时长、无重叠，但对 Align 失败的精确时间恢复为 0/24，质量门仍应保持 FAIL。

## 3. 实施原则

- [x] 功能完整优先于视觉重构；W0-W6 完成前不得用大规模样式改造掩盖数据/API 缺口。
- [x] CLI、PipelineRunner 和共享领域服务是规则来源，Web 只负责调用和呈现。
- [x] 页面不得直接解析 CLI 文本日志来决定阶段状态、质量门或失败原因。
- [x] 对齐状态统一为 `completed_exact`、`completed_coarse`、`failed`，OP/ED 使用 `SKIPPED_MUSIC_REGION` 单独表达。共享实现位于 `qwen_asr/alignment_state.py`；旧 `status` 字段继续兼容，新 manifest 显式持久化 `alignment_state`。
- [x] 导出成功不等于质量通过；质量门 FAIL 必须持续可见且不得标记正式完成。
- [x] `DEEPSEEK_API_KEY`、`MIMO_API_KEY` 等密钥只从环境读取，不通过 API 返回、不写入配置、不出现在日志和页面 DOM。
- [x] 保留脏工作树和用户已有修改；每轮修改前列出准确路径并逐文件备份。
- [x] 不针对 Sayonara Lara 单样例硬编码区间、片段编号或文本，样例只用于真实验收。

## 4. 分阶段 TODO

### W0 基线盘点与行为冻结

- [x] 盘点现有 Web 路由、请求/响应、命令构造、状态来源、工作区删除边界和前端控件。
- [x] 为当前 API、页面 payload、CLI 参数和 Web 状态建立机器可读契约快照。
- [x] 运行 `tests/test_webui.py` 和相关流水线测试，记录基线数量、耗时和失败项。
- [x] 标明现有页面中真实可用、仅占位、重复、失效和被否定实验功能。
- [x] 输出中文 W0 审计报告；不得在审计阶段删除功能或重构页面。

W0 证据：`reports/web_w0_audit.md`、`reports/web_w0_audit.json`、`reports/web_api_contract_v1.json`。首次修改前相关基线为 `84 passed in 2.89s`；第一轮 API 实现后 Web 定向为 `51 passed in 0.20s`，Ruff、compileall、`git diff --check` 通过。

验收：现有行为可复现，所有拟删除或替换能力均有调用点和测试影响清单。

### W1 后端领域与 API 边界

- [x] 将工作区读取、命令执行、进度事件、产物索引、质量状态和恢复任务拆成清晰服务边界。
- [x] 保留 CLI 命令兼容，同时让 CLI/Web 调用共享服务或共享结构化状态。
- [x] 定义稳定版本化响应模型、错误模型、路径规范化和 UTF-8 行为。
- [x] 避免继续扩大 `server.py`、`commands.py`、`status.py` 的混合职责。
- [x] 所有文件访问限定在允许的工作区和导出范围；删除接口继续执行路径边界保护。

验收：API 单元测试覆盖正常、缺失、损坏、运行中、失败和非法路径状态；CLI 原有行为不回退。

### W2 流水线与状态 API 完整化

- [x] 提供项目/工作区列表与详情 API。
- [x] 提供阶段列表、当前阶段、阶段状态、计数、耗时、日志引用和产物引用。`/api/v1/workspace/stages` 对当前任务提供真实耗时，历史阶段无可靠计时证据时返回 `null`。
- [x] 提供 Align 三态计数与逐片段状态，不以单一 completed/failed 二值替代。
- [x] 提供 OP/ED 排除区间和 `SKIPPED_MUSIC_REGION` 状态。
- [x] 提供 quality gate 总状态、分项检查、证据路径和阻塞原因。
- [x] 提供导出产物列表、元数据、预览和受控下载。SRT/VTT 预览固定为 UTF-8 文本，下载使用 attachment，路径必须来自当前 export inventory。
- [x] 支持结构化进度轮询或事件流；日志仅作为补充证据。当前 `/api/v1/job` 返回结构化进度，并将脱敏任务状态持久化到 `workspaces/.web-state/job.json` 与工作区 `reports/web_job.json`；服务重启后可识别已中断进程。

验收：断开浏览器后任务可继续；刷新页面能从工作区状态恢复；同一状态与 CLI/manifest 一致。

### W3 失败片段恢复队列

- [x] 为所有 `failed` 对白片段建立恢复任务，不能只处理高置信疑点子集。
- [x] 队列支持短应答、低覆盖、零时长 token、短窗口改写、语言混合等原因分类。
- [x] 优先展示短应答，提供片段音频、前后文、原 transcript、token/覆盖证据和参考字幕只读对照。
- [x] 支持音频 transcript 核验、VAD/局部定位、语言路由、重试 Align 和 `completed_coarse` fallback。`retry_align` 当前记录结构化重试请求；实际模型任务调度将在前端操作闭环中接入现有 Web job 服务。
- [x] 每次动作记录操作者、时间、输入、策略、结果、前后状态和证据路径。
- [x] OP/ED 不进入对白恢复失败统计，可查看但默认折叠。
- [x] 恢复后重新计算受影响下游状态和质量门，不允许仅改前端显示。接受 `completed_coarse` 会备份并原子更新 aligned manifest/checkpoint/event；下游由 ArtifactState 标记 outdated，旧质量 FAIL 保持到重跑质量门。

验收：真实样例可加载 22 个主对白失败项，其中 18 个短应答优先；未恢复时仍保持质量 FAIL。

### W4 前端功能壳与导航

- [x] 首屏直接进入可操作工作台，不制作营销落地页。
- [x] 建立项目选择、流水线、恢复队列、字幕审校、质量证据、导出和设置等明确视图。旧配置页保留在 `/legacy`，新工作台资源已拆分为 HTML/CSS/JS。
- [x] 支持任务启动、停止、刷新恢复、错误查看和阶段跳转。阶段启动复用 CLI payload/任务服务，并在启动前检查输入和环境凭据。
- [x] 页面状态直接来自结构化 API；加载、空数据、失败、只读和运行中状态齐全。
- [x] 使用现有页面符号体系；导航和命令按钮采用图标或短文字，并为禁用阶段提供原因 tooltip 与可访问名称。

验收：不依赖开发者控制台即可完成核心导航和任务观察；刷新不丢失服务端状态。

### W5 字幕审校工作流

- [x] 提供 cue 列表、时间轴/播放器联动、当前 cue 音频区间和前后文。结构化 cue 与工作区音频 Range API 已完成；前端联动在同阶段继续接入。
- [x] 提供日文原文、翻译、疑点、Align 状态、质量证据和参考字幕只读对照。
- [x] 支持筛选短应答、失败、粗略完成、低覆盖、翻译疑点和未处理项。恢复队列与 cue 审校分别提供短应答、Align 失败、粗略完成和疑点筛选；低覆盖通过恢复原因与详情证据展示。
- [x] 编辑保存必须触发明确的脏状态、校验和必要下游失效，不静默覆盖正式产物。Web 写入独立审校草稿，quality/normalize/export 显示 outdated。
- [x] 提供历史与撤销边界，保留原始 manifest 和操作审计。草稿保存有 revision、自动备份、JSONL 审计和上次编辑撤销。

验收：用户能从失败队列定位音频、完成核验、发起恢复、观察重算并回到字幕上下文。

### W6 质量、证据与导出闭环

- [x] 质量面板展示内容守恒、短应答、时间合法性、重叠、Align 覆盖、翻译结构、复核和重对齐结果。
- [x] 每个 FAIL/WARN 可跳转到对应 cue、片段、日志或报告证据。Align 失败分项跳转恢复队列，其余分项保留报告路径和结构化 target。
- [x] 明确区分草稿、质量门失败产物和正式可交付产物。
- [x] 导出面板展示 SRT/VTT/其他产物的路径、大小、生成时间、质量状态和下载动作。预览/下载仅允许当前导出 inventory 中的文件。
- [x] 参考字幕只用于人工比较与评估，不污染 ASR、Align 或恢复输入。

验收：Sayonara Lara 的 Normalize/Export 产物可预览下载，但页面仍明确显示正式质量 FAIL 及 22 个对白恢复项。

### W7 视觉与交互重构

- [x] 在 W0-W6 功能验收后拆分 81 KB 单页模板，按现有技术栈选择适度组件化方案。
- [x] 面向高频审校场景采用安静、紧凑、可扫描的工作台布局，不使用大 Hero 或装饰性卡片堆叠。
- [x] 不在卡片中嵌套卡片；面板标题字号与容器匹配。
- [x] 固定播放器、表格、工具栏、状态计数等关键尺寸，避免动态内容导致布局跳动。
- [x] 完成桌面和移动端响应式，确保中文长文本、路径、按钮和状态不重叠、不裁切。
- [x] 补齐键盘焦点、标签、对比度、禁用态和错误提示。

验收：Playwright 桌面/移动截图无空白、重叠、裁切、不可达控件或媒体失联；核心工作流操作次数合理。

### W8 全量验收与文档收口

- [x] 运行编译、Ruff、Web 定向测试、完整测试、CLI/Web parity 和 `local_check.ps1`。
- [x] 使用真实 Sayonara Lara 工作区做全链路 Web smoke，不重新污染原工作区。
- [x] 验证密钥不出现在响应、HTML、日志、快照或报告中。
- [x] 更新 README、ARCHITECTURE、PIPELINE、STATUS、CLI help 和 Web 行为说明。CLI 行为未新增参数，现有 help/parity 测试保持通过。
- [x] 生成中文最终验收报告、API 契约、页面截图索引和未解决风险清单。

验收：W0-W8 必做项全部通过，代码、浏览器、真实数据、文档和证据一致后才可完成 Goal。

## 5. 验证矩阵

### 5.1 每轮最低检查

```powershell
.\.venv312\Scripts\python.exe -m compileall qwen_asr optimizer tests -q
.\.venv312\Scripts\python.exe -m pytest tests/test_webui.py -q
.\.venv312\Scripts\python.exe -m ruff check qwen_asr optimizer tests
git diff --check
```

按修改范围增加相关 API、PipelineRunner、artifact state、quality gate、Align/recovery 和导出测试。阶段收口时运行：

```powershell
.\scripts\local_check.ps1
```

### 5.2 后端验收

- [x] API schema、错误码、路径边界和 UTF-8 测试通过。
- [x] 工作区缺文件、损坏 JSON、部分完成、任务中断和重启恢复均有测试。
- [x] CLI 与 Web 对同一 payload 生成相同业务参数和状态结论。
- [x] API 不返回密钥值，日志不记录 Authorization 或环境变量内容。
- [x] `completed_exact`、`completed_coarse`、`failed` 和 `SKIPPED_MUSIC_REGION` 不互相混用。

### 5.3 浏览器验收

- [x] 使用 Playwright 操作真实页面，不只检查 HTML 字符串。
- [x] 至少验证 1440x900、1280x720、390x844 三种视口。
- [x] 截图检查无重叠、裁切、溢出、空白面板、遮挡和不可读状态。
- [x] 音频播放、cue 定位、筛选、抽屉/面板、错误恢复、刷新恢复和下载可操作。
- [x] 页面无未处理控制台错误、失败网络请求和丢失静态资源。

### 5.4 真实数据验收

使用以下保留证据，不覆盖原始工作区：

- `workspaces/full-regression-sayonara-lara-02-20260715-234059/`
- `workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234/`

必须验证：

- [x] 显示 128 个 Align 输入片段、104 完成和 24 失败的原始结论。
- [x] 排除 OP/ED 后显示 113 个对白片段、91 完成和 22 失败。
- [x] 标识 15 个排除片段和 36 个排除 cue，不计入对白质量失败。
- [x] 22 个对白失败全部进入恢复队列，18 个短应答可筛选并优先显示。
- [x] 370 个标准化/导出 cue 可预览，非正时长和重叠均为 0。
- [x] 后置修复精确恢复仍为 0/24 时，Web 不得伪装为恢复成功。
- [x] 正式质量状态保持 FAIL，直到恢复流程真实解决并重新通过质量门。

## 6. 完成定义

只有以下条件全部满足才可标记完成：W0-W8 必做项全部勾选；后端 API、恢复队列、字幕审校、质量证据和导出功能闭环；CLI/Web/manifest 状态一致；单元测试、完整测试、Ruff、编译和 `local_check` 通过；Playwright 多视口验收通过；真实样例 smoke 通过；密钥零泄漏；中文文档与最终报告完成；没有破坏历史工作区、产物、媒体、模型、环境或用户已有修改。

视觉美化可以在功能闭环后迭代，但 W7 的结构、可用性、响应式和无重叠验收属于本计划必做项，不得以“后续人工再调样式”为由跳过。
