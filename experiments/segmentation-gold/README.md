# 断句 / 时间轴 / 说话人人工标准（gold）

用户在 Aegisub 中逐条修正，作为断句、时间轴和说话人的唯一评测基准。规则和 LLM 方案都要在这份标准上打分，赢了才进生产。

## 切片

- `make_gold_slice.py`：从工作区 `audio/source.wav` 截取切片音频（与流水线同一时间轴，不用 mp3，避免解码延迟），并生成起点 ASS。
  - 文字和时间取自 `manifests/split_segments.json`（规则切分 + LLM 语义合并，时间直接来自对齐 token，未经 normalize 延长）。
  - 说话人栏用 pyannote 预填；特效栏「多人」表示 pyannote 在这一条里听到 ≥2 个人。
  - ASS 时间相对切片开头；`[Script Info]` 里的 `KoeAxis Slice Offset` 是切片在整期中的起点（秒）。
- `cici30/radio-cici-30-gemini38-20261017_375-675s.{ass,wav}`：30 期 374.57–675.21s，118 条，按“短条数 + 换人次数”最多自动选取（跳过首尾各 90s）。切片音频（节目原声）不入库（`.gitignore` 忽略 `experiments/**/*.wav`），需要时用 `make_gold_slice.py` 从 30 期工作区重新截取；评分只需要 ASS 和工作区的 `aligned_segments.json`。

## 标注约定（实际采用）

- 每一行是一个**说话单元**：一个人连续说的一段，时间对准实际发音（不做后延）。断点要全，**不需要**按最终字幕合并——成品字幕由相邻单元合并而来。
- 说话人栏填真实名字（30 期：朱李、理名）；听不清是谁填 `?`。仍是 S00/S01/S02 的行视为未标注，不参与评分。
- **空文本行** = 应删除的语气词 / 无意义附和（保留时间和说话人，仅作记录）。特效栏可写备注（如「无效笑声」）。
- 文字以原识别为基础，可以改正；听不清的内容写 `？？？`（视为有内容的单元，不是语气词）。
- 时间允许与别人重叠（附和、齐声）。

## 评分：`score_gold.py`

```
python experiments/segmentation-gold/score_gold.py <gold.ass> <workspace>/manifests/aligned_segments.json \
    纯规则=<ws>/manifests/split_segments.json Gemini成品=<ws>/manifests/normalized_segments.json ... \
    --baseline 纯规则 [--show 名字] [--until 秒]
```

- 候选可以是 manifest json（`start_time/end_time` 毫秒 + `original_subtitle`）或 SRT，时间为整期绝对时间。
- 候选按字符对齐映射到共用的 ASR token 流；每个 token 按时间重叠 + 文字包含 + 前后连续归到某个 gold 行，所以 gold 改过文字不影响评分。
- 相邻两个有内容 token：同一单元却切开 = **错切**；不同说话人却在同一条 = **漏切**（附和也必须单独成行）；同一人的不同单元切不切都行，只计数。
- 落在空文本行里的 token 仍然显示 = **未删语气词**；整条只有这类 token = **语气词独立条**。
- **正确条**：恰好由同一人若干完整单元组成、且不含语气词。只对正确条统计起止时间误差（≤20/50/100ms），以及结尾超出 gold >0.3s 的条数。
- **改坏**：基线中正确的条，在该版本里不再被任何正确条包含。
- 已知局限：ASR 把齐声重复（两人同时说同一句）合成一份 token 时，会被记成一次漏切。
