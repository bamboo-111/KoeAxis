# KoeAxis 简历指标审计与可复现基准测试

审计时间：2026-07-13。仓库：`E:\project\qwen3-asr`。本报告只审计现有代码和实验产物，并在 `benchmarks/` 下新增独立 benchmark 结果；未修改核心代码、历史 workspaces、samples、模型缓存或日志。

## 1. 代码与架构审计

完整流程由 `qwen_asr.commands.stages.cmd_run` 调用 `PipelineRunner` 串联：prepare、transcribe、correct、align、split、translate、mimo-proofread、proofread-realign、normalize、export。核心证据：`qwen_asr/commands/stages.py:1259` 到 `1283`，`qwen_asr/pipeline_runner.py:33` 到 `112`，阶段顺序来自 `qwen_asr/stages.py`。

prepare：`qwen_asr/commands/stages.py:341` 到 `419`，负责抽音频、VAD、切 segment、可选 eager 导出并写 `segments.json`。解决输入媒体标准化和可复现切片问题。本次 benchmark 已真实运行，证据见 `E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/*/manifests/segments.json` 与各 `benchmark.log`。

transcribe：`qwen_asr/commands/stages.py:422` 到 `590`，调用 `QwenASRTranscriber`；`qwen_asr/asr.py:56` 到 `232` 支持批量 ASR、OOM 异常识别、显存 probe 和每批日志。已真实运行，证据见 `E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/*/transcribe.profile.json`。

align：`qwen_asr/commands/stages.py:593` 到 `760`，支持 Qwen forced align 和 MFA backend，并逐段写 event/checkpoint。已在补充 benchmark 中真实运行，证据见 `E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/*/manifests/aligned_segments.json`。

split：`qwen_asr/commands/stages.py:1112` 到 `1151` 通过 `qwen_asr.optimizer_bridge.run_split_stage` 接入规则或 LLM split。当前代码支持；本次补充 benchmark 未运行 split，历史工作区有 100 个 `split_segments.json`，但本轮未复跑验证 split 性能。

translate：`qwen_asr/commands/stages.py:1154` 到 `1175` 接入 `run_translate_stage`。当前代码支持；本次未调用外部 LLM，不报告翻译性能。

normalize/export：`qwen_asr/commands/stages.py:1068` 到 `1109` 做时间轴后处理，`975` 到 `1065` 导出 SRT/VTT。当前代码支持；本次未运行 normalize/export benchmark，历史工作区存在 normalized/export 产物。

自适应 batch：`qwen_asr/batching.py:24` 到 `173` 按时长桶 `0-15s/15-30s/30-60s/60-120s/120s+` 分组；`target_batch_audio_seconds` 在 `BatchPlanner._take_adaptive_segments` 控制每批总音频秒数；`single_long_segment_threshold` 在 `next_batch` 把长 segment 单独跑；`report_oom` 在 OOM 后降低 `current_max_batch_items` 并收缩目标音频秒数。`qwen_asr/commands/stages.py:80` 到 `140` 负责自动选择默认参数，`211` 到 `339` 写 `transcribe.profile.json` 和 `recommendation.next_run`。已真实运行，见本次 adaptive profile。

checkpoint/resume/原子写入：`qwen_asr/storage.py:29` 到 `53` 使用临时文件后 replace；`qwen_asr/commands/stages.py:564` 到 `574`、`735` 到 `744` 每段写 event/checkpoint/manifest；`1438` 到 `1452` 可从 checkpoint+events 恢复；`1596` 到 `1601` 实现 resume skip。本次 resume 实测：`baseline_a_fixed_b1_r1` 重跑 transcribe 0.787 秒跳过模型加载，日志 `E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/baseline_a_fixed_b1_r1/resume_test.log`。

batch-run：`qwen_asr/batch_runner.py:29` 到 `116` 支持批量任务、prepare 并发、失败继续/快速失败和 summary 输出；本轮只做代码和测试验证，未运行 batch-run。测试证据：`tests/test_batch_runner.py`。

相较早期 Voxlign 或旧版本：当前 Git 历史只有 `12874df Initial KoeAxis subtitle pipeline`，仓库内没有可直接 diff 的 Voxlign 代码，因此“替换/废弃了哪些流程”当前无法验证。可从现有代码确认当前版本保留了 Qwen ASR、forced alignment、optimizer split/translate、normalize/export；新增或强化了 adaptive batch、profile、checkpoint/event recovery、MFA 实验 backend、quality gate 等能力，但不能声称相对 Voxlign 的定量提升。

## 2. 现有实验数据盘点

现有工作区只读盘点：129 个含 manifest 的工作区，127 个含 `segments.json` 的媒体工作区，累计 7,986 个 prepare segment，segment 总时长 168,589.576 秒（46.830 小时）；完成 transcript 7,763 条、aligned 6,575 条、split 33,379 条、translated 23,502 条、normalized 18,486 条。

语言统计来自 transcript manifest 的 `language` 字段：Japanese 7,734、English 19、Chinese 2、unknown 8。segment 时长分布：min 2.652s、p50 13.245s、p75 15.600s、p90 56.667s、p95 103.750s、max 144.290s。

历史 profile：38 个 profile JSON；aggregate `oom_retry_count=0`，最大已完成 batch size 为 3，最大每批音频 295.1s，长段单独 batch 共 100 个。证据示例：`workspaces/0999-auto-profile-20260610-144415-iter1/transcribe.profile.json`、`workspaces/history-glossary-llm-full-20260610-v3/episode-21/transcribe.profile.json`。

错误和恢复记录：发现 85 个 checkpoint JSON、85 个 events JSONL，806 个 align failed status；只读扫描日志未发现 OOM/retry/resume 字样。注意这只说明现存日志无文本命中，不证明从未发生。

质量异常统计限制：无人工标注，不能报告 WER/准确率。自动检查发现空转录 8、负时间 0、aligned overlap 7,271、超 15s 字幕 465、align failed 806。overlap 可能包含合法重叠/多 token 或 manifest 语义差异，需要人工抽样复核后才能作为质量结论。

## 3. 补充 Benchmark

输入：`E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/input_180s.mp3`，由 `samples/test.mp3` 前 180 秒裁剪得到。每个方案 2 次，独立 workdir，均为模型已缓存状态，不是冷启动下载测试。共同切片：5 个 segment，总音频 175.891s，segment hash 在 CSV 中一致。

| variant | runs | success | avg total s | avg transcribe s | avg align s | avg total RTF | peak cuda reserved MB | avg batch size | max batch size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_a_fixed_b1 | 2 | 2 | 66.555 | 41.71 | 18.42 | 0.3783 | 4586.0 | 1.0 | 1 |
| baseline_b_fixed_b2 | 2 | 2 | 59.98 | 35.383 | 18.508 | 0.341 | 4998.0 | 1.667 | 2 |
| current_adaptive_auto | 2 | 2 | 60.061 | 35.621 | 18.084 | 0.3415 | 4584.0 | 1.25 | 2 |

CSV：`benchmarks/benchmark_results.csv`。完整命令：`benchmarks/reproduction_commands.md`。环境：`benchmarks/environment.txt`。日志：`E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/*/benchmark.log`。

可验证结论：在这个 180s 小样本上，fixed batch size 2 和 adaptive 的总耗时接近；fixed batch size 1 更慢。不能把该小样本外推为全量性能提升，也不能声称显著提升。

## 4. 简历可用技术点

1. 自适应 ASR batch 调度：按 segment 时长分桶、长段单跑、按每批音频秒数控批，并记录 `recommendation.next_run`。证据：`qwen_asr/batching.py:24`、`qwen_asr/commands/stages.py:211`、本次 `transcribe.profile.json`。

2. 可恢复推理流水线：transcribe/align 每段写 JSONL event、checkpoint 和 manifest，支持 resume 跳过已完成段。证据：`qwen_asr/commands/stages.py:564`、`735`、`1438`，本次 resume 0.787s 跳过。

3. 可复现实验记录：profile 记录 batch size、音频秒数、显存 probe、OOM 次数、batch 建议；本次生成 CSV/日志/命令。证据：`benchmarks/benchmark_results.csv`、`benchmarks/reproduction_commands.md`。

4. 批量媒体 runner：支持 manifest/多媒体输入、prepare 并发、失败汇总。证据：`qwen_asr/batch_runner.py:29` 到 `116`、`tests/test_batch_runner.py`；本轮未真实运行 batch-run。

## 5. Resume Audit 结论

### A. 已验证事实

- 本地环境为 RTX 4070 SUPER 12GB、PyTorch 2.11.0+cu128、qwen-asr 0.0.6、CUDA 12.8 可用。
- 本次独立 benchmark 6/6 成功，输入 1 个、总音频 175.891s、5 个 segment。
- 本次 benchmark 无 OOM、无 retry，ASR profile 峰值 CUDA reserved 约 6.7GB。
- resume 测试可跳过已完成 transcribe，额外耗时 0.787s。
- 相关单元测试 `tests/test_transcribe_batching.py tests/test_asr_batch_profiling.py tests/test_pipeline_runner.py -q`：30 passed。

### B. 尚未验证的推测

- adaptive batch 相对 fixed batch 的稳定全量加速幅度：当前只有 180s 小样本和历史 profile，不足以写百分比提升。
- WER/准确率：没有人工标注文本，当前无法验证。
- 相较 Voxlign 的替换/废弃细节：当前 Git 历史无可比较旧代码，当前无法验证。
- batch-run 在本次补充 benchmark 中未运行，只有代码与单元测试证据。

### C. 最适合写入简历的 3 条 bullet

- 为 Qwen3-ASR 字幕流水线实现按时长分桶的自适应 batch 调度，在 180s 样本上完成 2×3 组可复现 benchmark，记录 batch、RTF 与显存 profile。
- 设计 transcribe/align 的 event+checkpoint 恢复机制，本地实测已完成任务 resume 0.787s 跳过模型加载并保留逐段产物。
- 建立本地量化实验审计报告，盘点 127 个媒体工作区、7,986 个切片和 46.83 小时音频，仅输出统计值与证据路径。

### D. 每条 bullet 对应证据路径

- Bullet 1：`qwen_asr/batching.py:24`、`qwen_asr/commands/stages.py:211`、`benchmarks/benchmark_results.csv`、`E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/current_adaptive_auto_r1/transcribe.profile.json`。
- Bullet 2：`qwen_asr/storage.py:29`、`qwen_asr/commands/stages.py:564`、`qwen_asr/commands/stages.py:735`、`E:\project\qwen3-asr\benchmarks\resume-audit-20260713-232937/runs/baseline_a_fixed_b1_r1/resume_test.log`。
- Bullet 3：`benchmarks/resume_audit.md`、`benchmarks/environment.txt`、`benchmarks/reproduction_commands.md`。

### E. 不应写入简历的夸大表述

- 不写“显著提升/大幅优化吞吐”，当前没有全量 A/B 统计显著性。
- 不写“降低显存 X%”，当前只有峰值显存记录，没有严格同条件显存降幅结论。
- 不写“提升准确率/WER”，无人工标注。
- 不写“生产级产品”，该仓库更适合作为自用/研究工具。
- 不写“完全独立手写”，如涉及 AI 辅助开发，应如实表述。

### F. 后续最小补实验清单

- 用同一全量媒体跑 fixed b1、fixed 稳定 batch、adaptive 各 3 次，补齐冷启动/热启动区分。
- 增加独立 GPU 采样进程，记录 prepare/transcribe/align 全阶段峰值显存。
- 为 30 到 50 个随机片段人工标注或校对，再报告 WER/错误类型。
- 运行 batch-run 真实多媒体任务，验证失败继续和 summary 语义。
- 对 aligned overlap 和 align failed 样本做人工抽样，区分真实错误和 manifest 语义差异。
