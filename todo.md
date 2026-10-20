# TODO（2026-10-17 整理，同日逐项排查后更新；第二轮改进后再次更新）

范围：刺刺广播及日语视频字幕工作流的下一轮改进。原则：本流程优先换取正确率，速度与 token 是第二目标；改动先在一期新素材上与基线 A/B，再推广（测试素材一次只跑一期，24–30 期已用，下次用 **31 期**）。

基线：**文本 LLM 自 2026-10-17 起为 `gemini-3.8-flash`（Gemini 原生 v1beta，并发 4）**，MiMo 听音校对仍为 `mimo-v2.6-pro`。依据：`output/radio-cici-30-gemini38-20261017/`（对照 `radio-cici-30-mimo-20261017/` 同断句翻译、`radio-cici-30-mimosplit-20261017` 断句）。
旧 MiMo 基线：`output/radio-cici-25-tokenplan-20261016/`（工作区 `workspaces/radio-25-radio-cici-25-tokenplan-20261016/`）。
第二轮改进后的验证：`output/radio-cici-26-perf-20261017/`（工作区 `workspaces/radio-26-radio-cici-26-perf-20261017/`）。

### 断句 / 翻译思考等级对比（2026-10-20，gemini-3.8-flash-high，生产默认 low）
- 断句 LLM 合并（30 期 gold 300 s，每档 3 次）：正确条 low 82/82/83、medium 80/81/79、high 80/81/79；错切 15–16 → 13，但漏切换人 21 → 22，正确条时间精度（≤50 ms）43–44 → 40–41。整期合并耗时约 5 s / 70 s / 350 s。**无收益，保持 low。**
- 翻译（#15 / #18 同一断句，每档 2 次，claude-sonnet-4-6 对照字幕组已校译文盲评 2 遍，共 422 行）：严重错误 low 21.3、medium 17.7、high 19.0；同档两次之间评分差最多 13 分，与档间差同量级；两次都稳定的差别只有 3 行（low 错）对 1 行（medium 错）。耗时（10 分钟音频）约 35 s / 70 s / 170–270 s。**medium 可能略好但不显著，high 无额外收益；默认保持 low。**
- 注意：`--llm-extra-body-json` 对 run 中的 correct / split / translate 同时生效，单独给翻译调高需新增参数。
- 实验脚本与产物在会话 scratch（`think_split.py`、`think_tr.py`、`think_judge.py`），#15 / #18 底座工作区 `workspaces/radio-{15,18}-thinking-exp-20261020/`。

### Gemini 文本校对（text-proofread，2026-10-19）
- 已加入基线：v6 提示词 + medium，位于 proofread-realign 之后、quality-gate 之前；togeradi / japanese-generic 预设默认开启。54 期实验：零误改、术语表 9/9、跨片段成对错误能修；high 无额外收益。
- 54 期副本实跑（`workspaces/radio-54-textproofread-verify-20261019/`）：148 个请求 209 s（新中转并发 5 + 旧中转并发 2，旧中转 35 次中 4 次失败后自动转投），0 失败，改 28 行，只改中文、日文与时间轴不变，质量门 0 FAIL / 3 WARN（与正式产物相同）。
- 术语表 `samples/翻译对照.xlsx` 补充：声ログ/コエログ → 声log，トゲ占いの館/トゲウラナイノヤカタ → 刺刺占卜馆（53 期起由青春咨询所改版）。
- [ ] 词义选错、漏谓语（54 期 364 背带裤、333 灯、416、589、368、399）仍漏改，需新思路（如对照音频或二次问答）。
- [x] 2026-10-19 全部 Gemini 调用（split / correct / translate / text-proofread）改走项目中转公网 HTTPS（`GEMINI_BASE_URL=https://113.20.8.48`，默认模型 `gemini-3.8-flash-high`，`GEMINI_CONCURRENCY=5`，无需 SSH 隧道）；旧中转 ohmyapi（连接失败约 38%、并发 4 时 429）停用，凭据留在 `GEMINI_API_KEY_OLD` / `GEMINI_BASE_URL_OLD`。新中转偶发 TLS EOF（54 期 90 次中 3 次），由重试 / 中转池处理。
- [x] 术语表「广播用」列备注会拼上右侧「动画职位」列内容：2026-10-20 已修（注释列遇到下一块原文列即停）。

### 节目背景提示词 / 翻译提示词 v2 / LLM 纠错（2026-10-20，默认均未启用）
- 新增 `--program-context FILE`（可重复，自动拼进 correct / translate / text-proofread 的系统提示词，上限 8000 字）与 `--episode-title`（auto＝媒体文件名，仅在给了背景文件时附加；off；或直接写标题）。示例：`samples/program-context/togeradi.md`、`girls-band-cry.md`；预设未默认带（需要时在预设 `files` 里加 `{"flag": "--program-context", "path": ...}`，可多条）。Web `/legacy`「LLM / 翻译 / 校正」组可填（一行一个路径）。
- 新增 `--translate-prompt v2`（明显误听按本意翻并标 asr_suspect、跨条不重复、短附和/双关规则）与 `--translate-context-lines N`（每批前后 N 行只读上下文）；新增 `--correct-mode llm`（恢复 ASRCorrector：带背景 + 术语表的 LLM 纠错，再做规则清理；默认 rules 仍只做规则清理）。
- #15 / #18 对照（同一断句，各配置 2 次，claude-sonnet-4-6 盲评 2 遍，422 行）：严重错误（两期之和）v1 23.0、v1+背景 22.5、v2+5 行上下文 21.5、v2+上下文+背景 24.2；综合分（3×严重+一般）100.7 / 96.0 / 95.3 / 102.3。同配置两次之间最多差 13 分，**差异不显著，默认保持 v1、0 行上下文、rules**。耗时无明显差别（约 35–47 s / 期）。
- 个例：背景文件修好了「絶筆ツアー→Zepp 巡演」，v2 修好了「優秀勉強→备考」；「マニュイ」（招糕了）四种配置都没翻对；v2+背景在 #18 两次都把「三日神さん」改成「三香神」（背景和术语表里都没有，属模型乱改）。asr_suspect 数量 v2/背景略降（如 #18 124 → 113）。
- [x] 主要错误来源仍是 ASR 误识别被照字面翻译（约 70%）→ 2026-10-20 做了 LLM 纠错改进，见下节。
- 实验脚本：scratch `think_tr2.py`、`think_judge2.py`、`think_agg2.py`、`keylines2.py`。

### LLM 纠错改进（2026-10-20，togeradi 预设已启用）
- 新增纠错 v2（`--correct-prompt v2`，`--correct-mode llm` 时默认）：模型只返回「原片段→改后」局部替换（片段 ≤24 字、须逐字出现在原文）和置信度，程序替换，低于 `--correct-apply-threshold`（0.8）的只记在报告里；每批带前后 `--correct-context-segments`（默认 1）段只读上下文；`--correct-effort` 可单独给纠错设思考等级。v1（整段改写）仍可用。Web `/legacy` 与 CI 预设 run_args 均可配置。
- 纠错层面（#15/#18 原始转写，每配置 2 次，Gemini high 对照字幕组译文逐条判定，原文共 74 处误识别）：v1+背景 改对 47.5 / 改坏 23；第一版 v2（要求发音几乎相同、不改写法，medium）改对 40 / 改坏 7.5；放宽后的 v2 改对 46 / 改坏 25。
- 端到端（同一 ASR，从 correct 起重跑对齐/断句，翻译 v1 low；每配置 2 次纠错，claude-sonnet-4-6 按 45 s 时间窗盲评 2 遍，8 个版本同评）：综合分（3×严重+一般，两期合计）rules 基线 130.5、v1 102.3、第一版 v2 111.6、放宽 v2 100.3；严重错误 30.6 / 23.3 / 24.8 / 21.5。Gemini 评审结论相同（95.3 / 71.3 / 80.8 / 71.8）。**LLM 纠错让译文错误减少约 20–25%**；纠错层面"改坏"的多数对翻译影响不大，漏修的伤害更大。
- 采用：togeradi 预设加 `--correct-mode llm`（v2、上下文 1、low）和两份节目背景文件。japanese-generic 未改（无术语表/背景，未验证）。CLI 默认仍是 `rules`。
- 耗时：每期纠错约 10–20 s（low；medium 约 40–80 s 且无额外收益）。
- [ ] 节目背景同时进入 text-proofread 提示词，这一点未单独验证；下次正式期观察文本校对改动量。
- [ ] 纠错仍会改错个别词（如 優秀→一緒に、三日神→三上），可考虑后续让 MiMo 听音校对复核 LLM 纠错改过的段。
- 实验脚本：scratch `corr_exp.py`、`corr_judge.py`、`corr_agg.py`、`corr_e2e.py`、`think_judge3.py`、`think_agg4.py`；产物 `corr/`、`corr2/`、`corr-e2e/`、`think-tr3/`、`think-tr4/`。

### 30 期 Gemini 对比结论与待办（2026-10-17）
- 翻译：3.8-flash 对 ASR 错字/专名的还原明显好于 MiMo（打首獄門同好会、東京ガーデンシアター、声ログ）；MiMo 第 6–19 条整批译文错位一条且无任何检查报警。日常句两者接近。
- 断句：3.8-flash 语义合并后句中悬空 90 vs 107、<1 s 碎片 139 vs 156；MiMo 独有的 48 处切分大多切在词中（「よろしく｜お願いいたします」）。
- 速度：3.8-flash 并发 4 翻译 102 s（单请求中位 9.1 s），MiMo 并发 10 为 52.6 s；中转站单 Key 同时 5 个请求即 429（中转站自身限流，`retry-after: 60`）。
- 待办：① 翻译“整批错位”检测（条数对但内容错开时重译）；② 429 时按 `retry-after` 全局冷却而不是几秒内重试；③ 3.8-flash 断句不可切专名（トゲナシトゲアリ）与「&」后；④ ~~思考强度对比~~（2026-10-20 已做，见上；`--llm-extra-body-json '{"generationConfig":{"thinkingConfig":{"thinkingLevel":"high"}}}'`）；⑤ gemini-4-argon 需更高并发额度后再完整跑一次；⑥ 对齐：念邮箱（togeradi@interfm.jp）的段落 qwen 与短窗回退都失败，需 `recover-align --strategy mfa-local --mfa-padding-ms 200` 手动恢复。

---

## 0. 第二轮改进结果（26 期，2026-10-17，桌面程序已关闭）

| 项 | 25 期（改前） | 26 期（改后） |
|---|---|---|
| 全流程墙钟 | 约 11 分钟 | 8.2 分钟 |
| transcribe | 209.7s（batch 5） | 75.7s（batch 16，按时长排序） |
| GPU 阶段（不含 MFA 恢复） | 266s | 134s |
| MFA 对齐恢复 | 1 段 | 3 段，耗时 127s（已改为裁剪词典，重放降到 24–34s，见第 4 节） |
| translate 缓存命中 | 冷启动，估计约 66% | 90%（157k/175k） |
| mimo-proofread 缓存命中 | 16%（57k/361k） | 64%（178k/279k） |
| 全期输入 / 未命中输入 | 549k / 约 400k+ | 482k / 约 147k |
| 质量门 | 0 FAIL / 3 WARN（46 条 introduced_duplicate） | 0 FAIL / 4 WARN（introduced_duplicate 0 条） |

注：translate 已改为“极小预热请求”，27 期整期复测命中 90.1%（168k/186k），耗时 62.9s（26 期 71.4s）。27 期 mimo-proofread 命中 62.9%（193k/307k），出现 15 次 RateLimitError 重试。27 期全程 5.2 分钟（音频 28.8 分钟，没有对齐失败），产物在 `output/radio-cici-27-mfa-20261017/`，质量门 0 FAIL / 3 WARN。Token Plan 面板消耗请用户对照读数（26 期 02:23–02:31，27 期 03:12–03:17）。

---

## 1. 说话人切换：换人处断句

### 排查结论
- **MOSS-Transcribe-Diarize 可用**（详见 `experiments/speaker-change/README.md`）：
  - 广岛那期两处人工确认的换人，MOSS 都判对了，切点误差在 0.1–0.5s 以内；它还把待确认的「広島東橋で」识别成更通顺的「広島飛ばしで」。
  - 刺刺广播 25 期前 6 分钟：110 条字幕中标出 25 条，人工看约 15 条以上是真换人，说明“一条字幕里有两人说话”在刺刺广播里很常见（粗估一成以上）。误报主要来自边界差几百毫秒。
- **速度（空闲 GPU 复测，25 期前 360s）**：60s / 120s / 180s 分块都约 100s（实时率约 0.28），整段 360s 一次跑 160s（0.45），显存峰值约 7GB。一期 32 分钟约 9 分钟。此前“慢 7 倍”是因为当时在玩游戏占用了资源。建议 120s 分块。
- pyannote：已安装但无模型权重，未测。
- MOSS 的文字也可以作为第二份转写，用来辅助“漏识别 / 听错”检测（例：Qwen「前公演」，MOSS「舞公演」，推测原话是「毎公演」，待听音确认）。

### 待办
- [ ] 建小型标注集：广岛 #30/#34 + 25 期前 6 分钟的 25 条候选人工判定真 / 假，再加不该切的反例。
- [ ] 改进 `flag_cues.py` 的误报过滤：按 MOSS 切点到 Qwen 词边界的距离判断，只在切点两侧各有 ≥0.5s 语音时才算。
- [ ] 决定接入方式：align 之后产出“强制断点”，split / normalize 必须遵守；MOSS 作为可选的 GPU 阶段（约 +9 分钟 / 期）。
- [x] 补测 pyannote Community-1（2026-10-17）：一期约 30s、显存 1.6GB，广岛两处换人都检出；25 期前 6 分钟对快速接话、短句召回不如 MOSS。详见 `experiments/speaker-change/README.md`。
- [ ] 标注 `experiments/speaker-change/runs/cici25-full/labels-0-360/label_sheet.tsv`（38 条候选），然后算 pyannote / MOSS 的精确率与召回率，决定是否采用组合策略。
- [ ] 小样本方案见 `experiments/speaker-change/few-shot-plan.md`（第二版：说话人分离 + 声纹注册匹配，不自训模型）。候选：pyannote Community-1（需 HF token）、pyannoteAI API（付费）、MOSS + 开源嵌入模型。待用户提供：每位主持人 1–3 段 10–30s 单人片段（或确认我挑出的候选），以及 2–3 期各 5–10 分钟的换人测试标注；并确认是否输出说话人标签。
- [ ] 说话人判定（`enroll_diarize.py`，注册滑窗 + ReDimNet-M）继续提速，现状 30 期整期约 18s（2026-10-17 已从约 78s 优化，详见 `experiments/speaker-change/README.md` 性能优化一节）：
  - ① pyannote 分割模型导出 ONNX，自己实现切块、多人类别转换、计数聚合，不再导入 pyannote/lightning（约 −4~5s；需逐帧对齐验证；接入生产后如果进程常驻，这项收益变小）；
  - ② ReDimNet 整段只算一次特征再按窗口汇总（窗口声纹 8s → 约 2s；模型内有自注意力、按窗口做幅度归一化，只能近似，须在 30 期 gold + #15/#16/#18 silver 上重新评测）；
  - ③ 跳过完全没有人声的窗口（电台静音少，收益有限）。
  - 注：生产接入（2026-10-18）只用 ReDimNet 滑窗，不再需要 pyannote，①只对实验脚本有意义；生产里 30 期约 12s。
- [x] 说话人接入生产（2026-10-18）：`--speaker-enroll samples/speaker-enroll-togeradi.json`（togeradi preset 已加）。align 后算分数，split 按高置信换人切开 + LLM 合并禁止跨人，translate 带说话人提示。30 期 gold 正确条 66–71 → 83、漏切 32–35 → 23。详见 `experiments/speaker-change/README.md` 接入生产一节。
- [x] 导出带说话人的 ASS（2026-10-18）：`--speaker-enroll` 时 export 同目录另写同名 `.ass`，每位成员一个样式（1.理名/2.夕莉/5.朱李，取自已校 #15），低把握行 Name 标“说话人待确认”。30 期逐条准确率约 94.5%。
- [x] 端到端验收（2026-10-18，54 期，按 CI gpu/api 两阶段跑 togeradi preset，文本 LLM gemini-3.8-flash 并发 4，MiMo mimo-v2.6-pro）：GPU 阶段 1 段对齐失败（节目口号），自动恢复用 MFA 救回；API 阶段约 4 分钟；产出 SRT/ASS 各 661 条，在场成员自动判为 理名+夕莉，待确认 73 行；final quality WARN（与 26/27 期同级）。
- [x] Gemini 代替 MiMo 做听音校对（2026-10-18，54 期同一译文）：新服务商（ohmyapi）不支持音频（原生接口 400 UNSUPPORTED_BILLING_MODE，OpenAI 兼容接口无此通道），旧服务商 qhaigc 两种接口都支持。用 qhaigc gemini-3.8-flash 跑同样两段式：MiMo 采纳 33 处（修正 #89/#145-146/#512-513 译文错位、#182 漏主语等），Gemini 只 7 处且 2 处改错（年份 2026→2024、直译变生硬），耗时 410s vs 113s。保留 MiMo。
- [ ] 说话人后续：① 说话人提示让 LLM 合并偏保守（同人未完句不合并，详见实验 README“改坏”分析；弱化提示已试，无改善）；② 翻译带说话人的效果没有评测手段，只做了抽查；③ 重叠处的叠加行、按重叠区提高阈值都试过无效，插话附和只考虑对附和词单独处理；④ ASS 齐声台词目前只出一行（已校稿是每人复制一行）。
- [x] **54 期说话人识别问题（用户反馈，2026-10-18）**，已处理（详见 `experiments/speaker-change/README.md`“54 期说话人问题修复”）：
  - 「声ログ」三人被判成同一人 / 朱李差 / 易误判成理名：根因是整期在场判断把朱李排除（占比 2.4%、p95 0.46）。新增局部在场区间（30s 块内朱李占优即在该区间参与打分）和“高置信占比 ≥5% 即在场”；54 期 3:18–4:41 朱李 0→28 行，误判理名 37→7 行，30 期 gold / 15/16/18 期 silver 准确率不变。
  - 连续重复句：按“遍”判定说话人，两遍各自确信且不同人才切开（30 期 gold 漏切 23→21，改坏 0）；不确定时保持一行并标“说话人待确认”。
  - 「やります」×4：被对齐挤进下一句的末遍移回重复链、时间留给下一句；只含同一短语的相邻字幕合并为一条（「、」连接，normalize 不再拆）。
  - 剩余：手机录音段仍有约 1/3 行把握低；54 期无人工标注，只能间接核对；43/44 期在场规则变化无标注无法量化。
- [x] Web 支持说话人（2026-10-18）：`/legacy` 运行设置加「说话人注册 JSON」「在场成员」，run / batch-run / align / split / export 命令带 `--speaker-enroll`，工作台阶段启动复用保存的设置；导出列表、状态识别 `.ass`。
- [x] Web 运行设置加入 ASR 上下文文件（`--asr-context-file`，transcribe/run/batch-run）和节目记忆（`--program-memory-file`，normalize/run/batch-run），文件参数与 togeradi preset 一致。
- [x] `run` / `batch-run` 新增 `--align-recovery auto`：align 留下失败段时在同一进程里执行 CI 的恢复（Qwen bf16 → 本地 MFA → 审计过的 VAD 粗时间），恢复干净后继续后续阶段；54 期 segment_000041 实测由 MFA 救回。Web 运行设置加「对齐失败自动恢复」「校对后重对齐 MFA 回退」两个复选框（默认关），映射到 Run / Batch Run。
- [x] Web 与 CI 对齐（2026-10-18）：Run / Batch Run 传 MiMo 参数；校对方式加「纠错 + MiMo 音频校对」；`/legacy` 加对齐回退、听音范围、性能报告控件；Web 任务默认 `QWEN_ASR_MFA_BACKGROUND_PREFETCH=1`；配置抽屉加「CI 预设 → 载入」（`/api/ci-presets`，用 CI 的 `build_run_argv` + CLI 解析器换算成 Web 设置，测试保证载入后 Web 生成的 run 参数与 CI 只差 align_recovery / device / base URL / workdir）。

---

## 2. Token 效率（Token Plan）

### 已做
- 记录缓存命中：`pipeline_performance.*.json` 的每个请求和 `network_summary.token_totals` 含 `cached_tokens`。
- **translate 预热**（`optimizer/translator.py`）：并发前先发一个只含 system prompt 的极小请求（`max_tokens=16`，返回空内容也无妨），等 2s 再并发全部批次。实测预热约 4s，随后请求命中 1024/1061。（26 期整期用的是旧版“先单跑第一批”，命中 90%，但多花约 13s。）
- **mimo-proofread prompt 重排**（`qwen_asr/mimo_requests.py`、`qwen_asr/mimo_audio.py`）：术语表改为“常驻条目在前、本段命中条目在后”，并放在固定规则之后、段信息/条目之前；stage2 不再重复发送目标条目（邻近条目里已包含）。
- **mimo-proofread 预热**（`qwen_asr/mimo_proofread.py`）：stage1/stage2 先单发第一个请求，完成后等 2s 再并发。
- **A/B（25 期，同一输入，均为冷缓存）**：命中率 16% → 61%，未命中输入 304k → 127k（−58%），总输入 −9%，输出 −8%；结果差异与“旧 prompt 跑两次”之间的随机差异同量级，未见系统性变差；新版没有复现 DVD 重复。
- 缓存保留时间：同一 prompt 3 小时后重跑仍命中 87%。

### 待办
- [ ] 下一期整期复测“极小预热请求”版 translate 的耗时与命中率。
- [ ] mimo-proofread 每次仍有约 1/3 未命中（邻近条目、段信息、音频）；如需再降，可考虑减少邻近条目字段。
- [ ] 批量处理尽量放在北京时间 0:00–8:00（消耗 ×0.8）。

---

## 3. 25 期问题排查

### 结论
- `introduced_duplicate` 46 条中 45 条是误报（原话本来就重复，align 切短后落到相邻边界）；唯一真问题是第 398/399 条 DVD 重复（MiMo 给第 434 条补了下一条开头已有的「DVD」）。
- 对齐失败的 1 段是「小エログ。」（栏目名「コエログ」识别错），已由本地 MFA 恢复；成品第 106 条「小江戸」同类错误。
- 覆盖率 0.673 不是 25 期的问题：24/25 期、51–54 期、广岛都在 0.58–0.70，阈值 0.95 对电台节目不合适。
- 开头第 3 条「皆さんこんばんはトゲな」原文没错（有人抢先喊口号），但译文“我是刺刺”是补出来的。

### 已做
- [x] **修正 `introduced_duplicate`**（`qwen_asr/content_quality.py`）：若“X+X”在源阶段对应时间 ±1s 内本来就连续出现，则不报。复核：25 期 46 → 1（剩下的正是 DVD），24 期 25 → 0，广岛 0。
- [x] **MiMo 补全守卫**（`qwen_asr/mimo_guards.py::original_neighbor_duplication_guard`，接入 `mimo_application.py` / `mimo_proofread.py`）：建议原文追加的内容等于下一条开头、或前加的内容等于上一条结尾时拒绝；同一批次里邻条也被改写时按改写后的文字判断（允许“边界移动”）；重复片段只是更长片假名/拉丁词的前缀时（「アリ」vs「アリーナ」）不算。26 期实跑拒绝 20 条，人工核对 19 条正确、1 条误拒（已修规则）。

### 待办
- [ ] 覆盖率检查：改成只报覆盖率特别低的个别分段（如 <0.3），或用 VAD 去掉静音后再算；不再每期都 WARN。
- [ ] 节目记忆候选规则：「小江戸 / 小エログ」→「コエログ」（栏目名，限在“声ログ”环节附近）。
- [ ] 翻译 prompt：说一半被打断的句子不要补全。
- [ ] 抽样通读 25/26 期译文（梗、多人抢话段）。

---

## 4. GPU 转写

### 已做（空闲 GPU 实测，25 期 172 段）
- **按时长排序分批**（`qwen_asr/batching.py`，CLI `--batch-order duration|original`，默认 duration）：补齐浪费 19.1% → 1.2%，batch 5 推理 146/141s → 135/133s（约 −7%）；同顺序重复跑结果完全一致，换顺序有 9/172 段出现标点级差异。
- **按空闲显存放大 batch**（`qwen_asr/commands/transcribe_profile.py`）：short_form 且未显式指定 `--batch-size` 时，空闲显存 ≥9GB → 16，≥7.5GB → 12，≥6GB → 8，否则保持 5；记录在 transcribe profile 的 `vram_scaling`。实测（排序后）：

| batch | 推理合计 | 显存峰值（reserved） | 与 batch 5 原序文本差异 |
|---|---|---|---|
| 5 | 135s | 5.0GB | 9 段 |
| 8 | 94s | 5.2GB | 7 段 |
| 12 | 68s | 5.5GB | 6 段 |
| 16 | 57s | 6.0GB | 3 段 |
| 24 | 52s | 6.6GB | 4 段 |

  差异都是标点或个别语气词。玩游戏、空闲显存不足时自动回到 batch 5。

### 已做：MFA 对齐恢复提速（2026-10-17）
- 耗时分解：每次 `mfa align` 约 120s，其中加载 54 万词条的 `japanese_mfa.dict` 并规范化约 70s，建库等其余准备约 35s，真正对齐只有约 15s。
- **按批裁剪词典**（`qwen_asr/mfa_runner.py`）：只保留“是本批 .lab 文本子串（含 NFKC 小写形式）”的词条（剩 14–159 条，裁剪 0.2s），作为 DICTIONARY_PATH 传给 MFA，单次降到 12–20s。
  - 验证：13 个历史片段（含 5 个出 `<unk>` 的）词级时间戳与完整词典逐词一致；26 期 3 段恢复重放结果、token 全部一致。
  - 回退：裁剪版运行失败或缺输出时，自动用完整词典重跑一次；设 `QWEN_ASR_MFA_PRUNE_DICTIONARY=0` 可始终使用完整词典。
  - 每次结果都记录 `dictionary`（mode / kept_entries / fallback）。
- **后台预取（可开关，默认关）**：设 `QWEN_ASR_MFA_BACKGROUND_PREFETCH=1` 后，`start_alignment_recovery_mfa_prefetch` 在后台线程跑 MFA，与 Qwen 重试并行；`mfa-local` 会等同一工作区正在进行的预取完成。
- `mfa version` 检查结果在进程内缓存，每次 MFA 尝试省约 3s。
- 效果（26 期 3 段失败重放）：原 127s 预取 + 重试 → 同步 33.9s / 后台 24.2s。
- 27 期整期没有对齐失败，未触发 MFA。

### 待办
- [ ] 下一期出现对齐失败时，确认整期日志里的 `dictionary.mode=pruned` 与实际耗时。
- [ ] MFA 数据库按语料目录名（固定为 `corpus`）放在 `tools/mfa-root` 下；批处理脚本在 GPU 锁外跑 MFA，两期同时出现对齐失败时可能互相覆盖（旧问题，此前未遇到）。需要多期并行时给 MFA 加文件锁。
- [ ] align 清理耗时 11.7s，确认能否减少清理次数。
- [x] **ASR 候选 A/B（28 期，2026-10-17）**：`neosophie/Qwen3-ASR-1.7B-JA`（Qwen3-ASR-1.7B 的日语专有名词微调版，已下载到 `.model-cache`）。
  - 用法：`--model` 必须传本地快照路径。原因是 qwen_asr 包加载 processor 时不带 cache_dir，传仓库 id 会报离线找不到文件。
  - 对齐：两版都是 166 段精确对齐、1 段粗略对齐（同一段 133），配合 ForcedAligner 没有问题。
  - 速度：transcribe 98.2s，原版 68.4s（+44%）。
  - 文本：167 段中 141 段内容不同。JA 版更偏书面体，会省略语气词，数字写成阿拉伯数字，总字数 −3%。
  - MiMo 听音盲评（每段两种顺序各评一次，只统计两次结论一致的）：
    - JA 更好 31 段，原版更好 26 段，持平 27 段，两次不一致 57 段；
    - 判定的内容错误数：JA 187，原版 237；
    - 但 JA 漏内容的段更多（20 段，原版 12 段）。
    - 例：JA 对成员名「朱李」、惯用语「ほとぼりが冷める」更准；原版对「米津玄師」等更准，且漏句更少。
  - 结论：互有胜负，不是明确的上位替代。抽听表见 `output/radio-cici-28-asrja-20261017/ASR对比-28期.md`，待用户抽听核实。
  - 可选方向：保留原版做主转写，把 JA 版结果作为第二份候选交给 mimo-proofread 参考（每期 +约 100s GPU）。

- [x] **CI/CD WebUI（2026-10-17）**：新包 `qwen_asr/ci/`（控制端 server + GPU Worker 主动拉取 + 浏览器 SPA），文档 `docs/CI.md`，部署样例 `deploy/`（仅示例，部署者调整）。
  - 29 期真实端到端：首跑 336s（GPU 137s + API 188s）；`reset_from=translate` 重跑 117s，13 项 checks 与 SRT/reports.zip/manifests.zip 均上报。
  - 未验证：浏览器界面实测、Linux Worker、Docker/HTTPS 反代、多 Worker 并发。
  - 待查：重跑时 mimo-proofread 只发约 6 次请求，疑似复用旧 checkpoint，需确认是否按内容校验。
  - 后续可选：服务端审阅/对齐恢复编辑、S3 BlobStore、webhook 重试、JA ASR 候选作为预设选项。

---

## 5. 小项现状（保留）
| 项 | 现状 |
|---|---|
| ASR 漏识别检测 | 未实现。可借助 MOSS 第二份转写做对照（见第 1 项）。 |
| 质量检查 FAIL 自动修复 | 未实现。 |
| 旧成品「赤色の共済/共催」 | 节目记忆规则已有；`output/radio-cici-20261013-v2/` 中 45 期 1 处、47/48/49 期各 2 处、50 期 1 处，共 8 处未修。修改旧成品需用户同意（先备份）。 |
| Web 界面 `--program-memory-file` | 未实现（Web 代码中没有相关引用）。 |
| 翻译表译名建议 | 等用户决定（不改 `samples/翻译对照.xlsx`）。 |
| 切片 | 等用户从 `output/radio-cici-clips-v2-20261015/45-53期切片候选汇总.md` 挑选。 |
| 测试产物 | `$PI_SCRATCH_DIR/cliptest/` 与 `workspaces/radio-clip-45-01-cliptest/` 仍在，等用户决定是否删除。 |
| mimo-v2.5 下线（2026-10-21） | 本项目用 v2.6，不受影响。 |
