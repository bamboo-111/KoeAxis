# Align 失败恢复 Goal 模式提示词

## 使用方法

在新的 Codex 任务中启用 Goal 模式后，将下方代码块内的完整内容作为第一条消息发送。该提示词执行 Goal A：优先复用和接通既有恢复能力，不重复此前已经完成或被否定的广泛实验。

```text
请创建并持续执行一个长期 Goal，完成 E:\project\qwen3-asr 的 Align 失败恢复既有能力审计、生产接线和真实数据验收。

唯一权威计划是：

E:\project\qwen3-asr\docs\Align失败恢复既有能力审计TODO与验证计划.md

固定机器基线是：

E:\project\qwen3-asr\reports\align_recovery_goal_baseline.json

必须持续完成计划 A0-A7 的全部必做项。不要只完成只读审计、22 条分类、一个 recovery API、单元测试或一次定向实验后就标记 complete。只有既有能力矩阵、逐条执行轨迹、真实 executor、coarse 安全门、真实恢复、双数据集回归、下游质量重算、Web 浏览器验收和中文最终报告全部通过，才能完成 Goal。

不要设置 token_budget。预计执行 3-6 小时，8 小时是方案复核检查点，不是自动停止。每次续跑从最新未完成项继续，不重复已有完整证据且未受后续修改影响的工作。

一、强制规则

1. 所有计划、进度、审计、实验和最终报告使用中文。
2. 每轮修改前在 commentary 中列出准确的修改、新增和删除文件。
3. 每轮修改前把所有将修改或删除的已有文件备份到 `backups\主题-YYYYMMDD-HHMMSS\`；新文件写入该轮 `NEW_FILES.txt`。
4. 不得回滚用户已有修改，不使用 git reset、checkout、restore 清理脏工作树。
5. 未经明确路径授权，不得删除或移动历史工作区、报告、字幕产物、媒体、模型、虚拟环境、MFA 环境和用户文件。
6. 处理中文时不得用 PowerShell here-string 向 Python 传中文源码或匹配字符串；使用 apply_patch、UTF-8 文件、Unicode escape、行号或原生 PowerShell。
7. 终端乱码时先明确按 UTF-8 读取验证，不得判断源文件损坏。
8. 项目 Python 使用 `E:\project\qwen3-asr\.venv312\Scripts\python.exe`。
9. 不得 stage、commit 或 push，除非用户再次明确授权。
10. 工作超过 60 秒时持续提供简短中文进度。
11. 能从代码、manifest、metadata、历史报告和工作区查明的信息不要停下来询问用户。
12. 不要只给建议、伪代码或审计结论；必须继续实现、测试、运行真实恢复并产出证据。

二、固定范围

这是 Goal A，不是重新进行完整字幕流程研发。

必须处理：

- Sayonara Lara 排除 OP/ED 后的 22 个对白 Align 失败；
- 其中 18 个不超过 4 个规范化字符的短应答；
- 既有 Qwen、ASR short-window（配置值 `asr-short-window`）、MFA local、VAD/coarse、语言路由和 recovery queue 的生产可达性；
- Web `retry_align` 和 `route_language` 从状态记录到真实 executor 的接线；
- coarse 的 transcript、邻接边界、时间和审计硬门；
- exact/coarse/failed 的真实质量重算。

明确禁止：

- 不重新运行 P6 已完成且无收益的 9 组广泛 Align 参数矩阵；
- 不重新尝试 MFA 全量替代；
- 不恢复 LLM Split 或已退役实验功能；
- 不把整个 MiMo 或通用 LLM 校对提前到 Align 前；
- 不把 OP/ED 歌词作为对白恢复目标；音乐区域继续使用 `SKIPPED_MUSIC_REGION`；
- 不使用参考 ASS 文本作为 ASR、Qwen Align 或 MFA 输入；
- 不静默接受 short-window ASR 改写后的 transcript；
- 不批量把 failed 改为 completed_coarse 来制造数字改善；
- 不进行全站 Web 视觉重构，只补恢复执行所需的字段和交互。

三、开始时必须完成的只读审计

首次修改前读取：

1. 权威 TODO 与机器基线；
2. 根目录及适用 AGENTS.md；
3. git status、当前分支、提交和 diff；
4. `qwen_asr/commands/align.py`、`align.py`、`alignment_state.py`；
5. `recovery_service.py`、Web workspace/recovery API 和相关测试；
6. MFA candidates、runner、guards、writeback 和历史实验报告；
7. proofread-realign 的 Qwen clamp、MFA local 和 mixed-language 守卫；
8. P6 Align 9 变体矩阵、MFA/Qwen A/B 和最终回归结论；
9. Sayonara Lara 两个只读工作区及后置修复/OPED 报告；
10. Konoato01、Madougushi02 的固定验证输入和历史证据。

校验基线 JSON 中 6 个证据 SHA-256。若不一致，先查明是否是用户后续修改，不得自动覆盖或恢复。

四、核心工程判断

当前已知：

- Qwen 首次 Align 后仍有 24 失败，排除 OP/ED 后为 22；
- 18/22 是短应答；
- short-window fallback 在诊断中对 24 条均尝试，但精确恢复 0/24；
- short-window 已有内容守恒，改写 transcript 时会拒绝，不能放宽该门；
- MFA 全量替代已被双数据集否定，只允许审计 MFA local；
- Web recovery queue 已能分类、VAD 和写 coarse；
- `retry_align` 当前只记录 `retry_requested`，没有执行 backend；
- `route_language` 当前只保存状态，没有派发执行策略；
- coarse 写回已有备份，但需要 transcript 核验和邻接质量硬门；
- post-repair 只改了 2/24 文本，精确时间恢复为 0，不能通过提前整个 MiMo 解决。

首先验证这些判断，不要直接假设它们仍然正确。然后建立“能力、代码、入口、默认值、测试、历史证据、真实可达性、实际结果”的机器矩阵。

五、执行要求

按以下阶段持续推进并更新权威 TODO：

- A0：基线冻结与执行副本；
- A1：既有能力与生产可达性矩阵；
- A2：22 条失败执行轨迹和根因；
- A3：生产恢复执行器接线；
- A4：短应答 VAD 与 coarse 安全门；
- A5：定向真实恢复与首次晋升；
- A6：双数据集回归与下游重算；
- A7：Web/API 收口与最终验收。

1. 原始两个 Sayonara Lara 工作区保持只读，所有写实验在新派生工作区执行。
2. 22/22 条生成逐条执行轨迹，18/18 短应答进入短应答路由。
3. 每条必须记录 attempted、skipped、skip reason、root cause 和 confidence。
4. 把 recovery 动作接到共享生产 executor，不在 Web 内复制 Align 规则。
5. Qwen retry 默认使用原 transcript；verified text 必须有人工动作和审计。
6. MFA local 只处理适用日语候选，并通过内容、时间和环境守卫。
7. VAD coarse 必须证明 transcript 可信、区间可信、不吞并邻句、可撤销。
8. exact 必须有可靠 token；coarse 必须无 token 冒充且保留 WARN；failed 必须保留真实原因。
9. 每轮只改变一个策略组，输出前后计数、内容、时间、短应答和耗时。
10. 对 Sayonara Lara 完成后，用 Konoato01 和 Madougushi02 做必要的双数据集回归。
11. 只重跑受修改影响的下游；最终 quality、normalize 和 export 必须按真实状态重算。
12. 若质量门仍 FAIL，准确报告剩余失败，不得伪造 PASS。

六、验证要求

每轮至少运行：

.\.venv312\Scripts\python.exe -m compileall qwen_asr optimizer tests -q
.\.venv312\Scripts\python.exe -m pytest tests/test_align_cleanup.py tests/test_alignment_state.py tests/test_recovery_service.py tests/test_web_workspace_api.py -q
.\.venv312\Scripts\python.exe -m ruff check qwen_asr optimizer tests
node --check qwen_asr/web/static/workbench.js
git diff --check

阶段收口运行：

.\scripts\local_check.ps1

必须使用真实数据，不得只运行 pytest。使用 Playwright 验证代表性短应答、exact 恢复、coarse 接受、失败拒绝、刷新恢复和质量重算。检查浏览器 console、失败请求、音频定位、manifest 状态和 API 一致性。

七、完成条件

只有全部满足才可将 Goal 标记 complete：

1. A0-A7 全部完成；
2. 22/22 有机器执行轨迹和唯一主根因；
3. 18/18 短应答经过适用恢复路由；
4. 既有能力全部有生产可达性结论；
5. retry_align 和适用语言路由执行真实 backend，不再只写状态；
6. exact、coarse、failed 由真实证据决定；
7. coarse 具备 transcript、VAD/时间、邻接和审计硬门；
8. 所有可由既有能力恢复的条目已实际处理；
9. 未解决条目有证据拒绝和 Goal B 候选，不进行无边界扩展；
10. 双数据集没有新增短对白缺失、内容损失、非法时间或严重重叠；
11. Sayonara Lara 下游和质量门已真实重算；
12. CLI、API、Web、manifest 和报告一致；
13. 编译、定向测试、完整测试、Ruff、Node 和 diff 检查通过；
14. Playwright 和真实数据验收通过；
15. 原始工作区、媒体、环境和用户改动未受损；
16. 中文最终报告列出恢复前后指标、耗时、未解决项和 Goal B 建议。

完成不要求 22 条全部变成 completed_exact，也不预设最终业务质量必须 PASS；但不能只做审计而不执行已有能力，也不能在仍有安全、明确、范围内工作时提前结束。
```
