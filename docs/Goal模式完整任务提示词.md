# Goal 模式完整任务提示词

## 使用方法

在新的 Codex 对话中启用 Goal 模式后，将下方代码块内的完整内容作为第一条消息发送。

这不是单阶段提示词。它要求一个 Goal 持续执行完整主计划，包含 MFA 全量 A/B、稳定基线、align → split 根因审计、逐项优化、疑点音频复核、修改后重对齐、显式质量门和全部回归。不得在完成 MFA 实验或任一中间阶段后把 Goal 标记为完成。

```text
请创建并持续执行一个长期 Goal，目标是完整实现并验证 E:\project\qwen3-asr 的字幕识别流程改造计划。

Goal 的唯一完成标准是：

完整执行并通过以下主计划文档中的全部必做阶段、验收标准和 TODO：

E:\project\qwen3-asr\docs\字幕流程完整改造计划与验收清单.md

不要只执行第一阶段，不要在 MFA 实验完成后停止，也不要把“代码写完”“单元测试通过”“某一集运行完成”当成整个 Goal 完成。只有主计划中的必做任务、两套可靠 ASS 的逐阶段验收、后续回归、最终默认策略和中文总结报告全部完成后，才能将 Goal 标记为 complete。

不要为 Goal 设置 token_budget。任务需要跨多次自动续跑持续推进；每次续跑都从当前未完成阶段继续，不要重新开始，不要重复已完成的实验，也不要因为上下文压缩而丢弃已有进度。

一、强制工作规则

1. 所有计划、进度说明、审计报告、实验报告和最终报告必须使用中文，禁止使用英语计划和报告。
2. 开始任何代码修改前，必须在 commentary 中列出本轮准备修改和新增的准确文件。
3. 每轮修改前，必须把所有将修改的已有文件备份到：

   backups\主题-YYYYMMDD-HHMMSS\

4. 新文件没有原文件可备份时，也必须建立该轮备份目录，并写入文件清单，说明哪些文件是新增文件。
5. 不得修改、删除、清理或覆盖历史工作区、实验产物、导入数据、视频、ASS、虚拟环境、MFA 环境、模型和用户已有改动，除非用户明确点名授权。
6. 当前工作树有大量未提交改动。不得使用 git reset、git checkout、git restore 或其他方式回滚它们。
7. 处理中文文本时，不得用 PowerShell here-string 直接向 Python 传中文源码或中文匹配字符串。
8. 涉及中文匹配、替换或生成时，使用 apply_patch、UTF-8 文件、Unicode escape、行号或原生 PowerShell 安全方式。
9. 终端显示中文乱码时，不得判断源文件损坏；必须先用明确 UTF-8 编码读取并验证。
10. 项目虚拟环境为 E:\project\qwen3-asr\.venv312，所有项目 Python、编译和测试优先使用该环境。
11. 禁止针对单个样例写硬编码修复。每次修改必须定位通用根因，并用至少两套可靠 ASS 验证。
12. 不恢复 MiMo 全量复核。正常流程只复核疑点；全量模式只作为明确的诊断实验保留。
13. 可靠 ASS 只能作为质量标准和评估输入，不得作为初始 ASR、MFA 或 Qwen forced alignment 的输入文本。
14. 不得重新运行不同 ASR 后与固定基线对比。MFA 和 Qwen 的 A/B 必须使用完全相同、哈希固定的 transcript。
15. 每一轮修改完成后，至少运行语法或编译检查、针对性测试、完整测试，以及与修改阶段相匹配的两套可靠 ASS 真实 A/B。
16. 单元测试通过不代表字幕质量通过。必须同时满足主计划定义的内容守恒、ASS 质量、时间合法性、短应答和回归验收标准。
17. 正常可继续的工作不得停下来询问用户。应优先从代码、manifest、metadata、历史报告和工作区中自行查明信息。
18. 只有缺少无法从本地发现且会实质改变方案的必要信息时才向用户提问。
19. 不要只提供建议、伪代码、审计结论或下一步计划；必须实际修改、测试、运行验证并产出报告。
20. Goal 未完成时不得标记 complete。只有同一阻塞原因在至少三次连续 Goal 续跑中重复出现、已穷尽安全替代方案且无法继续时，才可以按 Goal 规则标记 blocked。

二、开始时必须完成的只读审计

在第一次修改前，完整读取并核对：

1. E:\project\qwen3-asr\docs\字幕流程完整改造计划与验收清单.md；
2. 项目根目录及可能存在的 AGENTS.md；
3. git status 和 git diff；
4. qwen_asr、optimizer、tests 中当前相关实现；
5. backups 中最近各轮备份；
6. workspaces 中已有干净回归、MFA 局部实验和质量报告；
7. 当前测试数量和结果；
8. 两套可靠 ASS 的固定 transcript、源音频、偏移和文件哈希；
9. 当前 Qwen align 的入口、输出结构和 fallback；
10. 当前 MFA 局部实现、micromamba 调用方式、模型位置和可复用代码；
11. 当前 split prompt、split 输入适配和后处理逻辑；
12. 当前内容守恒、ASS 质量、疑点路由、复核重对齐和最终质量门接线。

审计后建立内部执行顺序和 TODO 状态，但不要停留在只读审计。立即说明第一轮修改文件、创建备份并开始执行主计划。

三、已知项目状态

- 工作目录：E:\project\qwen3-asr
- 最近完整测试结果：259 passed。
- ASR 默认最大切片：15 秒。
- LLM 默认并发：5。
- correct 主执行路径已经改成确定性文本清理。
- 翻译阶段已经支持 translation、asr_suspect、needs_audio_review、suspect_types、reason、confidence。
- MiMo 默认复核范围已经是 suspects。
- 已有 content-quality、ass-quality、ass-quality-diff、quality-suspects、final-quality、proofread-realign、tuning-matrix、mfa-align-experiment 等模块。
- 当前 quality-gate 虽然有定义和隐式拦截，但尚未完整成为正式显式阶段。
- 当前两套可靠 ASS 的最终质量仍为 FAIL。
- MFA 局部实验只证明过少量失败条目可以恢复，尚未证明 MFA 全量优于 Qwen3-ForcedAligner。
- 当前局部 MFA 每条独立启动约耗时 145–155 秒，全量对齐必须建立批量 corpus，不能逐字幕启动 MFA。

MFA 环境：

E:\project\qwen3-asr\tools\mfa-env

micromamba：

E:\project\qwen3-asr\tools\micromamba\extract\Library\bin\micromamba.exe

MFA root：

E:\project\qwen3-asr\tools\mfa-root

已安装模型：

- japanese_mfa 声学模型；
- japanese_mfa 词典。

不要直接调用 tools\mfa-env\Scripts\mfa.exe。应设置 MFA_ROOT_DIR，并通过以下方式调用：

tools\micromamba\extract\Library\bin\micromamba.exe run -p tools\mfa-env mfa

四、固定验证集

验证集一：Konoato01

视频：

D:\Users\下载\[ANi] 畫完這個再去死 - 01 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4

可靠 ASS：

D:\Users\下载\Kore Kaite Shine - 01 extracted subtitles\01.chs-jpn.ass

参考偏移：

ASR/SRT 时间相对 ASS 约 +510ms

已有干净回归目录：

E:\project\qwen3-asr\workspaces\clean-regression-20260712-002943\konoato01

验证集二：Madougushi02

可靠 ASS：

E:\project\qwen3-asr\tmp_manual_tests\compare-madougushi02-official-20260710-213319\official\JPSC.ass

参考偏移：

ASR/SRT 时间 - 6180ms ≈ ASS 时间

已有干净回归目录：

E:\project\qwen3-asr\workspaces\clean-regression-20260712-002943\madougushi02

Madougushi 的视频或完整音频路径必须从已有 metadata、manifest 或历史工作区解析，不得猜测，也不得因为终端路径乱码判断文件损坏。

五、完整执行阶段

必须按依赖关系持续完成以下所有阶段。允许在证据表明顺序需微调时调整局部顺序，但不得跳过任何必做阶段。

阶段 1：实现 MFA 全量对齐 backend，并与 Qwen3-ForcedAligner 做对等 A/B。

1. 提供 --align-backend qwen|mfa；
2. 实验完成前默认保持 qwen；
3. MFA 路径必须完全绕过 Qwen forced align；
4. Qwen 路径必须完全绕过 MFA；
5. 两者使用相同固定 transcript 和哈希；
6. MFA 按整集构建批量 corpus；
7. 第一轮以 15 秒原始 ASR segment 为 corpus 单元；
8. 不得逐字幕启动 MFA；
9. 不得用 ASS、翻译或后续校对文本污染 MFA 输入；
10. 解析 MFA 输出并转换成统一 aligned manifest；
11. 显式记录 alignment_backend、alignment_unit、覆盖率、<unk>、失败和非法时间；
12. 两个 backend 进入完全相同的 split；
13. 两套验证集分别生成 align 和 split 的 content-quality、ass-quality 和诊断报告；
14. 生成中文 MFA/Qwen 全量 A/B 报告；
15. 按证据选择：MFA 全量替代、Qwen 主对齐加 MFA 局部 fallback、或 MFA 暂不进入生产流程。

阶段 1 的结论不能只看平均分。若平均分提高但短对白缺失、低于 0.20、内容守恒 FAIL 或非法时间增加，则 MFA 全量替代不通过。

阶段 2：建立两套稳定基线。

固定 transcript、deterministic-correct、aligned、split、translated、proofread、proofread-realigned、normalized 和 export。为每阶段保存文件哈希、条目数、规范化日文字符数、短应答数、独有文本数、时间合法性、ASS 平均分、低于 0.45、低于 0.20、短对白缺失、相邻重复和内容守恒状态。相同命令重复运行时指标必须稳定。

阶段 3：完整审计 align → split 失败。

审计两套数据的全部 became-fail、全部短对白缺失、最大 20 条 score-drop、最大 20 条 matched-text-shortened、短应答吞并、边界重复、0 时间、1ms token 簇、长时间覆盖和非单调 token。逐条区分 align 内容、align 时间、token 粒度、split prompt、规则切分、后处理和评估误匹配。输出中文根因报告。

阶段 4：逐项优化 align。

分别实验局部插值最大间隔、零时间 token 默认和最大时长、单侧锚点、可靠 token 判定、覆盖率阈值、1ms 簇、短应答时长、最终选定 backend 及其输入标准化。每轮只改变一个变量组。两套数据均不得新增低于 0.20、短对白缺失、内容守恒 FAIL 或非法时间。

阶段 5：审计并优化 split prompt、输入适配和后处理。

至少比较当前 prompt、强化内容守恒 prompt、纯规则 split，以及适用时的 LLM split 加规则后处理。Prompt 只能选择句界，不得改写、删除、新增、翻译或纠正 ASR。Split 后规范化日文必须与 align 完全一致，内容和短应答保持率必须为 100%。平均分提高但短应答消失仍判定失败。

阶段 6：完善翻译疑点标注。

保持新结构和旧格式兼容，确保翻译不修改日文。完善人名/实体、否定、疑问、数量、主客体、碎片、语义矛盾、未翻译、上下串联、短应答、时间和内容守恒疑点。结构缺失时质量门 FAIL。

阶段 7：完善仅疑点音频复核。

普通模式只使用翻译、结构和质量门疑点；可靠 ASS 评估模式可额外使用 ASS 低分和阶段回归。所有修改必须有音频证据，拒绝空值、纯英文替换日文、异常缩短、独有文本无证据删除和短应答吞并。有可靠 ASS 时，修改后局部分数不得下降。必须修复 Madougushi 当前 MiMo 后短对白缺失从 12 增至 13 的回归。

阶段 8：完善复核后局部重对齐。

根据 MFA 全量 A/B 选择 backend 顺序。只重对齐日文被修改的条目，未修改条目时间不得变化。检查内容一致性、<unk>、时间单调、边界、严重重叠和可靠 ASS 局部分数。Fallback 进入 WARN，未解决超过门槛时进入 FAIL。

阶段 9：把 quality-gate 改为正式显式阶段。

正式顺序必须为：

proofread-realign → quality-gate → normalize → export

统一 CLI、PipelineRunner、checkpoint、artifact state 和 Web UI 的状态。检查逐阶段内容守恒、短应答、独有文本、重复、对齐覆盖率、1ms 簇、断句可读性、翻译结构、复核证据、重对齐、checkpoint 和 SRT 合法性。FAIL 时不得生成或标记正式完成产物，但必须保留草稿和中文诊断报告。Normalize/export 不承担质量提升职责，若改变规范化日文则 FAIL。

阶段 10：完成全部回归和默认策略决策。

按顺序回归：

1. Konoato01；
2. Madougushi02；
3. Madougushi01；
4. BDEP4；
5. SLEP2；
6. 其他已下载验证集。

最终生成中文决策报告，列出 Qwen 全量、MFA 全量、混合 fallback、align 参数、split prompt、split 后处理、疑点复核确认错误率、未解决率、单位耗时和最终质量门结果。高置信失败应逐步向不超过 5 收敛。若尚未达到，必须继续处理可修复的高置信失败，不能仅报告未达标后结束 Goal。

六、每轮参数或实现晋升标准

任一实现或默认参数只有同时满足以下条件才能晋升：

1. 两套可靠 ASS 均不得新增低于 0.20 的条目；
2. 两套均不得增加短对白缺失；
3. 两套均不得增加内容守恒 FAIL；
4. 两套均不得增加非法时间或严重重叠；
5. 至少一套目标指标明确改善；
6. 另一套不得明显回退；
7. 修改原因可以归因于本轮变量；
8. 语法检查、针对性测试和完整测试通过；
9. 若不通过，回退本轮失败方案，但不得回滚其他已有改动；
10. 失败实验也要保留中文报告，避免后续重复尝试。

七、持续执行与进度管理

1. 将主计划中的阶段和 TODO 作为 Goal 的持续检查清单。
2. 每完成一个阶段，更新该阶段的实际状态、证据路径、测试结果和未完成项。
3. 每次自动续跑先读取上次进度、最新报告和工作树，再继续下一个未完成项。
4. 不要因为一次工具运行时间长而停止；使用可恢复的 checkpoint 和等待机制继续。
5. 不要重复执行已经有完整证据且未受后续修改影响的实验。
6. 后续修改影响旧结论时，只重跑受影响的阶段和必要下游验证。
7. 工作超过 60 秒时持续提供简短中文进度更新。
8. 不要在中间阶段输出类似“任务已完成”的结论。

八、每轮最低验证命令

使用项目虚拟环境运行：

.venv312\Scripts\python.exe -m compileall qwen_asr optimizer tests -q
.venv312\Scripts\python.exe -m pytest -q

除此之外，必须运行与本轮修改对应的真实数据验证和质量报告。仅运行 pytest 不足以晋升字幕质量修改。

九、Goal 完成条件

只有全部满足时才能调用 Goal 完成：

1. 主计划所有必做阶段执行完成；
2. MFA/Qwen 全量 A/B 有两套可靠 ASS 的完整中文报告和明确结论；
3. 两套稳定基线已固定；
4. align → split 失败已完成逐条根因分类；
5. align 和 split 优化通过双数据集晋升标准；
6. 正常流程仅复核疑点；
7. 修改后重对齐和 fallback 策略已由证据确定；
8. quality-gate 已成为正式显式阶段；
9. FAIL 时正式产物确实被阻止；
10. Konoato01 和 Madougushi02 的最终指标无关键回退，并相对当前基线有实质改善；
11. Madougushi01、BDEP4、SLEP2 和其他可用验证集完成回归；
12. 项目编译、针对性测试和完整测试全部通过；
13. 所有最终报告为中文；
14. 原始数据、历史工作区和用户已有改动未被破坏；
15. 最终中文报告列出完成项、最终默认参数、所有验证结果、剩余不确定性和产物路径；
16. 主计划中的所有必做 TODO 已勾选，或仅剩文档明确标记为未来可选扩展的项目。

如果尚有安全、明确且在范围内的下一步，就继续执行，不要结束 Goal。
```
