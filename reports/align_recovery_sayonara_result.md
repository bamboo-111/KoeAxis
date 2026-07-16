# Sayonara Lara Align 定向真实恢复结果

生成时间：2026-07-16T07:06:56.357393+00:00

## 前后指标

| 指标 | 恢复前 | 恢复后 |
|---|---:|---:|
| completed_exact | 91 | 92 |
| completed_coarse | 0 | 0 |
| failed | 22 | 21 |
| short_failed | 18 | 18 |

## 策略执行

- Qwen 原 transcript retry：20 条；exact=0。
- MFA local：记录 22 条适用/跳过结论；临时 exact=2，正式质量复核后保留 exact=1。
- 短应答 VAD：18 条；localized=14。
- 未经 transcript 可信核验的 VAD 结果均未写成 completed_coarse。

## 判定

临时恢复得到 93 exact / 20 failed；正式质量复核发现 `segment_000036` coverage 为 0.164931，低于 0.20 接受门槛，因此通过目标级 `undo_recovery` 撤销该条，同时保留后续 `segment_000086` 恢复。

正式结果为新增 exact=1、coarse=0：`segment_000086`（coverage 0.453669、content score 1.0）保留为 exact；其余 21 条继续保持 failed，并保留 backend 拒绝、不可执行或 transcript 未核验的证据。未放宽内容、时间或状态门。
