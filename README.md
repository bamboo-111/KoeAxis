# KoeAxis / 声译轴

KoeAxis 是面向本地音频转写、时间轴、翻译与人工审阅字幕候选分支的工作流；底层继续使用 Qwen3-ASR 与 Qwen ForcedAligner。

本项目用于本地离线视频转写与字幕生成，针对 `Qwen/Qwen3-ASR-1.7B` 和 `Qwen/Qwen3-ForcedAligner-0.6B` 做了顺序式显存使用设计，适合本地单卡或工作站环境。

## 特性

- 支持常见音频/视频媒体输入，`ffmpeg` 抽取 16kHz / mono / WAV 音频
- 可选 `ffmpeg afftdn` 音频降噪，适合背景噪声明显的素材
- CPU VAD + 静音优先切片，默认最大 15 秒、最小 2 秒
- ASR 与 ForcedAligner 分阶段执行，不同时驻留显存
- 每段结果实时落盘，支持 resume
- 导出 `JSON / TXT / SRT / VTT`
- 支持全局时间戳 offset 处理
- 内置本地 `optimizer/` 断句与翻译逻辑
- split、correct、translate 与 MiMo API 默认并发数统一为 5
- 提供本地网页控制台处理全流程与中间继续

## 推荐环境

- Python 3.10 - 3.12
- NVIDIA GPU，显存 > 6G
- CUDA 版本需与本地 `torch` 匹配
- `ffmpeg` 需在 `PATH` 中可用


## 安装

推荐直接使用一键安装脚本：

```bat
start.bat install
```

安装完成后：

```bat
start.bat web
```

或进入 CLI 环境：

```bat
start.bat cli
```

兼容旧入口：

```bat
start.bat web
```

如果需要手动安装，再用下面这套：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

如果要启用 FlashAttention 2，请自行安装兼容版本，并通过 CLI 显式传入：

```bash
--attn-implementation flash_attention_2
```

它不是默认依赖，也不是必需依赖。

## 离线模型缓存

`transcribe`、`align`、`run` 默认都会把模型缓存目录设为项目内的 `.\.model-cache`，避免 `--local-files-only` 时误碰用户目录缓存。

首次离线运行前，请先把下面两个模型准备到该目录，或用 `--model-cache-dir D:\path\to\cache` 指向你已有的缓存：

```text
Qwen/Qwen3-ASR-1.7B
Qwen/Qwen3-ForcedAligner-0.6B
```

如果使用默认的 `--local-files-only`，程序会在加载模型前检查缓存目录是否存在、是否可写、是否为空。空缓存会直接报错；如需允许在线下载，请显式传 `--no-local-files-only`。

## 命令

### 0. 预检查

```bash
python main.py preflight --workdir .\work --media input.mp4
```

`preflight` 会做轻量检查，包括输入文件、模型缓存目录、`local_files_only`、`device/dtype` 和 CUDA 可用性。
`transcribe`、`align`、`run`、`batch-run` 默认也会在正式执行前复用这套检查；如需跳过，可显式传 `--skip-preflight`。

### 1. 准备音频和切片

```bash
python main.py prepare --media input.mp4 --workdir .\work
python main.py prepare --media input.mp4 --workdir .\work --denoise --denoise-level 12
```

默认只生成 `audio/source.wav` 和 `segments.json`，不再一次性导出全部 segment wav。
如需保留旧行为，可加：

```bash
--eager-segment-export
```

### 2. 转写

```bash
python main.py transcribe --workdir .\work --model Qwen/Qwen3-ASR-1.7B
```

推荐参数：

```bash
python main.py transcribe ^
  --workdir .\work ^
  --model Qwen/Qwen3-ASR-1.7B ^
  --model-cache-dir .\.model-cache ^
  --dtype fp16 ^
  --device cuda ^
  --batch-size 5
```

新的批调度默认使用 `--batch-mode adaptive`，会按 segment 时长分桶，并结合总音频时长预算组批。若要强制回退到旧的固定条数行为，可显式指定：

```bash
--batch-mode fixed
```

在 `adaptive` 模式下，如果你不显式传 `--batch-size`、`--target-batch-audio-seconds`、`--single-long-segment-threshold`，系统会根据 `segments.json` 的时长分布自动选择一组更合适的默认值。

如需手动限制每批总音频时长，可再配合：

```bash
--target-batch-audio-seconds 180
```

如果素材里存在明显超长段，可继续配合：

```bash
--single-long-segment-threshold 90
```

当 segment 时长达到该阈值后，`adaptive` 模式会优先把它单独作为一批执行。

如需输出批级内存与调度报告，可启用：

```bash
--profile-batches
```

执行后会在 `workdir/transcribe.profile.json` 写出批次汇总，包含每批的时长分布、是否触发 OOM 重试、以及 `before/after/cleanup` 内存探针快照。
同时会附带 `recommendation.next_run`，给出下一轮可直接尝试的 `batch_size`、`target_batch_audio_seconds` 和 `single_long_segment_threshold` 建议值。

多语言混杂素材建议不要传 `--language`，让模型自动识别。只有整段语言很确定时才使用：

```bash
--language Japanese
```

如果当前 `torch` + GPU 栈对 BF16 更稳定，可改用：

```bash
--dtype bf16
```

### 3. 对齐

```bash
python main.py align --workdir .\work --model Qwen/Qwen3-ForcedAligner-0.6B
```

模型缓存默认使用项目内 `.\.model-cache`。如果你已经把模型放在其它位置，可显式传 `--model-cache-dir D:\path\to\cache` 覆盖。

如需调整 Align 阶段的显存清理频率，可显式指定：

```bash
--cleanup-interval 4
```

默认不会每个 segment 都做一次完整清理，而是按间隔清理并在阶段结束时做一次 full cleanup。

注意：默认部署策略是 `15s silence-first segmentation`。15 秒是找不到合适静音时的切段上限，并不是每 15 秒机械硬切；切段仍优先选择静音位置。每段默认保留 300ms padding，相邻窗口可能覆盖同一小段音频，后续对齐桥接会只删除时间确实重叠且文字完全一致的边界重复，同时保留双方独有内容。长窗口会提高漏识、显存压力和字幕漂移风险，如需改回更长窗口应显式传入 `--max-segment-seconds`。

### 4. 导出字幕

```bash
python main.py export --workdir .\work --format srt
python main.py export --workdir .\work --format both
```

如果没有对齐结果，但希望按 segment 级时间导出较粗糙字幕：

```bash
python main.py export --workdir .\work --format srt --coarse-subtitles
```

### 5. 一键运行

```bash
python main.py run --media input.mp4 --workdir .\work --denoise --with-align --format both
```

### 5.1 批量运行

```bash
python main.py batch-run --workdir .\batch-run samples\test.mp3 samples\test.mp3 --format srt
```

也支持从清单读取：

```bash
python main.py batch-run --workdir .\batch-run --manifest tasks.json
```

批量模式特性：

- 每个媒体文件独立 workdir
- `prepare` 默认有限并发（`--prepare-workers 2`）
- GPU 相关阶段默认串行
- 默认失败继续下一个；可用 `--fail-fast`
- 汇总输出写入 `workdir\summary\batch-summary.json` 和 `batch-summary.txt`

### 6. 使用内置 optimizer 断句

规则断句：

```bash
python main.py split --workdir .\work
```

生产 split 仅支持规则实现。历史 LLM split 模式已经完成实验并被否定，相关 CLI、Web、prompt 和运行分支均已退役；翻译 LLM、MiMo 和通用 `llm_client` 不受影响。

翻译：

```bash
$env:LLM_API_KEY = "<your-provider-key>"
python main.py translate ^
  --workdir .\work ^
  --llm-model your-model ^
  --llm-base-url http://127.0.0.1:8000/v1 ^
  --target-language 简体中文
```

凭据只从环境变量读取：MiMo 使用 `MIMO_API_KEY`，DeepSeek 官方接口使用 `DEEPSEEK_API_KEY`，其他通用兼容接口使用 `LLM_API_KEY`。Web 页面不会显示、保存或提交凭据。

导出时会优先使用：

1. `translated_segments.json`
2. `split_segments.json`
3. `aligned_segments.json`
4. `transcript_segments.json`

也可以显式指定：

```bash
python main.py export --workdir .\work --format srt --source split
```

### 7. 规范化时间轴

```bash
python main.py normalize --workdir .\work --source split
```

常用参数：

```bash
python main.py normalize ^
  --workdir .\work ^
  --source translated ^
  --extend-ms 350 ^
  --snap-gap-ms 200 ^
  --min-blank-ms 300
```

### 8. 本地网页

```bat
start.bat web
```

启动后访问：

```text
http://127.0.0.1:8765
```

网页支持：

- `/` 默认进入结构化字幕工作台；旧参数配置页保留在 `/legacy`
- 查看阶段顺序、输入/输出计数、真实任务耗时、日志与产物，并从工作台直接继续可独立运行阶段
- Align 使用 `completed_exact / completed_coarse / failed` 三态；OP/ED 使用 `SKIPPED_MUSIC_REGION`，不计入对白失败
- 所有 failed 对白进入恢复队列，支持 transcript 核验、语言路由、局部 VAD、重试请求与 `completed_coarse` fallback
- 370 cue 级审校、整集音频定位、只读参考 ASS 对照、独立草稿保存、自动备份、审计与撤销
- 质量面板可跳恢复项、审校 cue 或受控报告；导出面板支持 UTF-8 预览与 attachment 下载
- `Stop` 终止当前任务，刷新后从服务端持久化任务状态恢复
- 媒体路径支持手动输入，也可以通过本机文件选择器回填真实绝对路径
- 默认工作目录位于 `workspaces/编号-输入源名称/`
- 默认最终字幕导出到原始媒体文件同级同名路径，也支持自定义导出路径
- 非敏感页面参数自动保存；API 凭据只从 Web 服务进程环境读取，不进入 HTML、localStorage、payload、命令日志或报告
- 工作线程数可配置（`1+`）；split 使用规则并发，LLM 参数仅用于 translate、MiMo 和包含这些阶段的 run
- 通过独立子进程执行各阶段，保持 GPU 阶段之间的显存释放策略

## 输出目录

```text
workspaces/0001-source-name/
  project.json
  progress.json
  audio/
    source.wav
    segments/
      segment_000001.wav
      segment_000002.wav
  manifests/
    segments.json
    transcript_segments.json
    aligned_segments.json
    split_segments.json
    translated_segments.json
    normalized_segments.json
  drafts/
    transcript.txt
    subtitles.normalized.srt
  export-cache/
    subtitles.srt
    subtitles.vtt
  logs/

原始媒体同级目录/
  source-name.srt
  source-name.vtt
```

## 默认策略

- `batch_size` 在 CLI 解析阶段默认为未指定；adaptive 模式按当前片段时长分布自动选择 3、4 或 5，fixed 模式未显式指定时使用 5
- 默认 `batch_mode=adaptive`
- 若当前批大小触发显存不足，ASR 会自动降到更小的 batch 重试，并沿用降级后的值继续后续批次
- 若显存紧张或希望严格贴近逐段结果，可手动改回 `--batch-size 1`
- `adaptive` 模式会优先把时长接近的 segment 放进同一批，并受总音频时长预算约束
- 超过 `single_long_segment_threshold` 的长 segment 会自动单独跑，减少长尾段拖垮整批
- ASR 与 ForcedAligner 不同时加载
- 默认模型缓存目录为项目内 `.\.model-cache`，避免 `--local-files-only` 时误碰用户目录缓存
- 每段完成后释放局部 tensor
- 每阶段结束后执行：
  - `del model`
  - `gc.collect()`
  - `torch.cuda.empty_cache()`
- 不将全部音频长期保存在 GPU

## 中间结果与 Resume

- `segments.json` 是 prepare 阶段的权威切片定义
- `transcript_segments.json` 是 ASR 阶段的权威结果
- `aligned_segments.json` 是对齐阶段的权威结果
- `split_segments.json` 是 optimizer 断句结果
- `translated_segments.json` 是翻译结果
- `normalized_segments.json` 是时间轴规范化结果
- `transcribe` / `align` 现在会同时写事件日志和 checkpoint：
- `transcript_events.jsonl` / `transcript_checkpoint.json`
- `aligned_events.jsonl` / `aligned_checkpoint.json`
- 阶段结束仍会保留兼容的完整 `transcript_segments.json` / `aligned_segments.json`
- 已完成 segment 在默认 `--resume` 下会跳过
- 使用 `--force` 可覆盖已有结果

## 适配层说明

`qwen-asr` 的具体 API 可能随版本变化。项目已将模型调用封装到：

- `QwenASRTranscriber`
- `QwenForcedAligner`

如果你安装的版本与当前假定接口不同，请查看以下 `TODO` 标记位置并按本地版本修正：

- `qwen_asr/asr.py`
- `qwen_asr/align.py`

当前默认假设存在近似以下调用：

```python
Qwen3ASRModel.from_pretrained(...).transcribe(...)
Qwen3ForcedAlignerModel.from_pretrained(...).align(...)
```

## 代码结构

入口层：

- `main.py`: CLI 兼容入口，转发到 `qwen_asr.cli`
- `webapp.py`: WebUI 兼容入口，转发到 `qwen_asr.web.server`
- `start.bat`: 统一入口，支持 `web` / `cli` / `install`

核心流水线：

- `qwen_asr/cli.py`: argparse 参数定义与命令分发
- `qwen_asr/commands/`: `prepare / transcribe / correct / align / split / translate / normalize / export / run` 阶段实现
- `qwen_asr/stages.py`: 阶段顺序、输入/输出产物、下游清理定义
- `qwen_asr/artifact_state.py`: 产物完整性、缺失输入、过期状态、清理路径判断
- `qwen_asr/progress.py`: `progress.json` 读写
- `qwen_asr/models.py`: dataclass 数据结构和 `WorkPaths`
- `qwen_asr/storage.py`: JSON 落盘、UTF-8-SIG 读取和原子写

媒体与字幕处理：

- `qwen_asr/audio.py`: ffmpeg 抽音频、切片导出
- `qwen_asr/vad.py`: VAD 与 speech/silence 区间处理
- `qwen_asr/segmenter.py`: 基于 VAD 的保守切片逻辑
- `qwen_asr/asr.py`: Qwen3-ASR 推理适配层
- `qwen_asr/align.py`: Qwen3-ForcedAligner 推理适配层
- `qwen_asr/subtitle.py`: SRT/VTT 导出
- `qwen_asr/normalize.py`: 时间轴规范化后处理
- `qwen_asr/optimizer_bridge.py`: optimizer 适配层

WebUI：

- `qwen_asr/web/server.py`: HTTP 路由、任务生命周期、停止任务
- `qwen_asr/web/commands.py`: Web payload 到 CLI 命令的构造
- `qwen_asr/web/status.py`: 状态、进度、日志摘要
- `qwen_asr/web/static_html.py`: HTML 模板加载
- `qwen_asr/web/templates/index.html`: WebUI 页面模板

Optimizer：

- `optimizer/splitter.py`: 本地规则断句兼容入口
- `optimizer/translator.py`: LLM 翻译
- `optimizer/llm_client.py`: OpenAI-compatible LLM 客户端
- `optimizer/llm_config.py`: LLM 配置 dataclass
- `optimizer/text_utils.py`: 文本工具兼容导出层
- `optimizer/asr_cleanup.py`: ASR/字幕清理
- `optimizer/text_metrics.py`: CJK/多语言字数统计
- `optimizer/fixed_terms.py`: 固定词和邮箱类规则修正

项目辅助：

- `docs/STATUS.md`: 当前规范、进行中计划和历史实验文档的状态入口
- `docs/ARCHITECTURE.md`: 架构说明
- `docs/PIPELINE.md`: 阶段与产物说明
- `docs/WEBUI.md`: WebUI 模块和接口说明
- `scripts/install.bat`: 一键安装脚本
- `scripts/start-webui.bat`: 网页启动脚本
- `scripts/start-cli.bat`: CLI 启动脚本
- `scripts/build-package.bat`: 生成分发 zip，并检查关键模板/模块是否入包
- `samples/`: 示例媒体和表格
- `tests/`: 轻量回归测试

## 验证与打包

常用回归命令：

```bat
.venv312\Scripts\python.exe -m compileall main.py webapp.py qwen_asr optimizer tests
.venv312\Scripts\python.exe -m pytest tests
.venv312\Scripts\python.exe main.py --help
.venv312\Scripts\python.exe -c "import webapp; import qwen_asr.cli; import qwen_asr.final_quality; print('ok')"
```

统一本地检查入口：

```powershell
.\scripts\local_check.ps1
```

开发环境依赖固定在 `requirements-dev.txt`。首次配置或依赖版本变化后运行：

```powershell
.venv312\Scripts\python.exe -m pip install -r requirements-dev.txt
```

`local_check.ps1` 默认将 Ruff 作为硬门。`-SkipRuff` 只允许用于诊断依赖安装问题，不能作为最终验收结果。

项目整理与归档盘点：

```bat
.venv312\Scripts\python.exe scripts\project_inventory.py
```

构建分发包：

```bat
scripts\build-package.bat
```

生成结果：

```text
dist\qwen3-asr-package.zip
```

## 已知边界

- 当前实现优先稳定部署，不做流式
- 默认 VAD 走 CPU
- overlap 支持保留为参数，但默认关闭
- 若启用 overlap，文本去重建议在上游模型输出稳定后再针对真实语料微调
