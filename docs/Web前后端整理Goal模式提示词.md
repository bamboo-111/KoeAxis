# Web 前后端整理 Goal 模式提示词

## 使用方法

在新的 Codex 任务中启用 Goal 模式后，将下方代码块中的完整内容作为第一条消息发送。该 Goal 以 Web 功能完整为第一优先级，在后端、恢复队列、审校、质量和导出闭环完成后再进行视觉与交互重构。

```text
请创建并持续执行一个长期 Goal，完整实现和验证 E:\project\qwen3-asr 的 Web 前后端整理任务。

唯一权威计划与验收清单是：

E:\project\qwen3-asr\docs\Web前后端整理TODO与验证计划.md

必须持续完成其中 W0-W8 的全部必做项。不要只完成后端 API、页面壳、某个恢复接口或一次视觉改版后就把 Goal 标记为 complete。只有代码、测试、真实浏览器验收、真实工作区 smoke、文档和最终中文报告全部通过，才能完成 Goal。

不要设置 token_budget。任务需要跨多次自动续跑；每次从最新未完成 TODO 继续，不要因上下文压缩重新开始，不要重复已有完整证据且未受后续修改影响的工作。

一、强制规则

1. 所有计划、进度、审计、验收和最终报告使用中文。
2. 每轮修改前在 commentary 中列出本轮将修改、新增和删除的准确文件路径。
3. 每轮修改前把所有将修改或删除的已有文件备份到 `backups\主题-YYYYMMDD-HHMMSS\`；新文件也要在备份目录登记。
4. 不得回滚脏工作树或用户已有修改，不得使用 git reset、checkout、restore 清理现状。
5. 未经明确授权，不得删除或移动历史工作区、字幕产物、导入数据、媒体、模型、虚拟环境、MFA 环境和用户文件。
6. 处理中文时不得用 PowerShell here-string 向 Python 传中文源码或匹配字符串；使用 apply_patch、UTF-8 文件、Unicode escape、行号或原生 PowerShell。
7. 终端中文乱码时先按 UTF-8 明确读取验证，不得直接判断源文件损坏。
8. 项目 Python 优先使用 `E:\project\qwen3-asr\.venv312\Scripts\python.exe`。
9. 不得 stage、commit 或 push，除非用户另行明确授权。
10. 工作超过 60 秒时持续提供简短中文进度更新。
11. 正常可从代码、manifest、metadata、报告和工作区查明的信息不要停下来询问用户。
12. 不要只给建议、伪代码或审计报告；审计后必须继续实施、测试和验收。

二、产品与架构约束

1. 功能完整优先于视觉重构。先完成 W0-W6，再完成 W7 的结构与视觉整理，最后 W8 收口。
2. CLI、PipelineRunner 和共享领域服务是业务规则权威来源。Web 不得复制流水线规则，也不得依靠解析人类日志决定核心状态。
3. 后端必须提供结构化的工作区、阶段、Align、恢复队列、质量门、证据和导出 API。
4. Align 使用三态：`completed_exact`、`completed_coarse`、`failed`。OP/ED 使用 `SKIPPED_MUSIC_REGION`，不计入对白失败门禁。
5. 所有 failed 对白片段都必须进入恢复队列，短应答优先；不能只处理高置信疑点子集。
6. 恢复层位于第一次 Align 之后、Split 之前。不要把整个 MiMo/LLM 校对阶段提前到 Align 前。
7. 不要整体恢复此前被证据否定并退役的实验功能。确需重新引入时，必须先证明通用需求、边界和回归收益，并在计划中记录。
8. Normalize/Export 只保证格式与输出，不承担 Align 质量修复。导出成功不得覆盖质量门 FAIL。
9. 参考 ASS 仅用于人工对照和评估，不得污染 ASR、Align 或恢复输入。
10. `DEEPSEEK_API_KEY` 和 `MIMO_API_KEY` 只从环境读取。不得保存、返回、渲染、打印或写入测试快照；DeepSeek 保持项目既有 `deepseek-v4-pro` 和禁止思考配置，除非用户另行调整。
11. 首屏必须是实际工作台，不制作营销落地页。功能闭环应覆盖项目选择、流水线、失败恢复、字幕审校、音频/参考对照、证据、质量门、导出和设置。
12. 页面不得嵌套卡片或使用大 Hero；使用现有图标库，保持紧凑、可扫描、面向重复审校工作。

三、开始时的只读审计

在首次修改前读取并核对：

1. `docs\Web前后端整理TODO与验证计划.md`；
2. 根目录和适用的 AGENTS.md；
3. git status、git diff 和当前未提交文件；
4. `qwen_asr\web\commands.py`、`server.py`、`status.py`、`static_html.py`、`templates\index.html`；
5. `tests\test_webui.py` 和相关 pipeline、artifact、quality、align、export 测试；
6. CLI、PipelineRunner、manifest、artifact state 和 quality gate 当前结构；
7. 最近备份与 `reports\recent_full_flow_cleanup.*`；
8. 两个 Sayonara Lara 保留工作区及关键报告；
9. 当前 Web 启动方式、端口、浏览器可访问性和静态资源；
10. 当前页面功能、占位功能、重复功能、已失效功能和被否定实验入口。

完成 W0 审计报告后立即继续第一轮实现，不要停在计划阶段。

四、真实验收基线

保留并只读使用：

E:\project\qwen3-asr\workspaces\full-regression-sayonara-lara-02-20260715-234059
E:\project\qwen3-asr\workspaces\post-repair-full-flow-sayonara-lara-02-20260716-002234

必须在 Web 中真实验证：

- Align 原始状态为 128 个输入、104 完成、24 失败；
- 排除 OP/ED 后为 113 个对白片段、91 完成、22 失败；
- 15 个片段和 36 个 cue 标记为音乐区域排除；
- 22 个对白失败全部进入恢复队列；
- 其中 18 个不超过 4 个规范化字符的短应答可筛选并优先显示；
- 370 个标准化和导出 cue 可查看，无非正时长和重叠；
- 后置修复精确恢复为 0/24 时不得显示为恢复成功；
- 质量状态保持 FAIL，直到恢复结果真实通过质量门。

不得为该样例硬编码区间、编号、计数或文本。所有数据必须来自通用 manifest、报告和 API。

五、持续执行顺序

严格以主计划 W0-W8 为检查清单：

W0 冻结基线并输出审计；
W1 建立后端领域与 API 边界；
W2 补齐流水线、三态 Align、OP/ED、质量和导出状态；
W3 建立完整失败片段恢复队列与审计；
W4 建立可操作的前端功能壳和导航；
W5 完成字幕、音频、参考对照和恢复审校工作流；
W6 完成质量证据与导出闭环；
W7 在功能稳定后拆分单页并完成视觉、响应式和可访问性重构；
W8 完成全部测试、真实浏览器、真实数据、文档和最终报告。

允许依据依赖调整同一阶段内部顺序，但不得跳过必做项。每完成一项，及时更新主计划中的 TODO 状态、证据路径和实际测试结果。后续修改使旧证据失效时，只重跑受影响范围及必要下游。

六、验证要求

每轮至少运行与修改范围相符的编译、导入、定向测试和 Ruff。常用最低命令：

.\.venv312\Scripts\python.exe -m compileall qwen_asr optimizer tests -q
.\.venv312\Scripts\python.exe -m pytest tests/test_webui.py -q
.\.venv312\Scripts\python.exe -m ruff check qwen_asr optimizer tests
git diff --check

阶段收口运行：

.\scripts\local_check.ps1

不能只靠 HTML 字符串测试。必须启动本地 Web 服务并用 Playwright 操作真实页面，检查 1440x900、1280x720、390x844 视口；保存截图并检查无重叠、裁切、溢出、空白面板、遮挡、控制台错误、失败请求和媒体失联。验证刷新恢复、任务观察、失败筛选、音频/cue 联动、证据跳转、质量状态和导出下载。

必须增加 CLI/Web parity、API schema、路径边界、UTF-8、损坏/缺失状态、任务中断恢复和密钥零泄漏测试。单元测试通过不代表 Web 工作流通过；真实浏览器和真实工作区 smoke 都是硬门。

七、完成条件

只有全部满足时才能将 Goal 标记为 complete：

1. 主计划 W0-W8 所有必做 TODO 已完成；
2. 后端领域/API 边界清晰，Web 不再依赖日志解析核心状态；
3. 流水线、Align 三态、OP/ED 排除、质量门和导出 API 完整；
4. 22 个主对白失败在真实样例中全部进入恢复队列，18 个短应答可优先处理；
5. 字幕审校、音频、参考对照、恢复操作、证据、质量和导出形成闭环；
6. CLI、Web、manifest 和质量结论一致；
7. Normalize/Export 成功不会错误覆盖质量 FAIL；
8. 编译、Ruff、Web 定向测试、完整测试和 local_check 全部通过；
9. Playwright 多视口真实浏览器验收通过；
10. Sayonara Lara 保留工作区真实 smoke 通过且无样例硬编码；
11. API、页面、日志、快照和报告中没有密钥泄漏；
12. README、ARCHITECTURE、PIPELINE、STATUS、CLI help 和 Web 行为一致；
13. 中文最终报告列出改动、验证、截图、真实数据结果、剩余风险和产物路径；
14. 历史工作区、产物、媒体、模型、环境和用户已有修改未被破坏；
15. 未经授权未 stage、commit 或 push。

如果仍有明确、安全且在范围内的工作，就继续执行，不要提前结束 Goal。只有同一实质阻塞连续至少三轮重复、已穷尽安全替代路径且确实无法继续时，才按 Goal 规则标记 blocked。
```
