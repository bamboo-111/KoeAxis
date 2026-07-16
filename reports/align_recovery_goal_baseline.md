# Align 失败恢复下一 Goal 基线

## 状态

- 材料状态：`READY`
- 目标类型：Goal A，既有能力审计与生产接线
- 基线提交：`59b78594ecc8dff5aae6a1fbd20ed93f674b1f0f`
- 基线分支：`codex/web-pipeline-workbench`
- 预计用时：3-6 小时；8 小时为方案复核检查点，不是自动停止或 token 限制。

## 为什么不重新做全量优化

此前已经完成 Qwen/MFA 全量 A/B、9 组 Align 参数矩阵、短窗口 ASR fallback、局部 MFA、修改后重对齐、短应答保护、零时长 token 守卫、内容守恒和双数据集回归。下一 Goal 首先要查明这些能力在首次 Align 失败时是否真正可达、为什么没有产生恢复，而不是重新运行相同实验。

当前代码已经暴露出三个明确接线缺口：

1. Web `retry_align` 只写入 `retry_requested`，没有执行 Qwen、MFA local 或其他生产 backend。
2. `route_language` 只保存语言状态，没有据此派发可执行恢复策略。
3. `accept_completed_coarse` 能写回 VAD 区间，但仍需补齐 transcript 已核验、相邻边界和质量重算硬门。

这些是下一 Goal 的优先对象。不能用批量接受 coarse、放宽内容守恒或降低 coverage 阈值来制造表面改善。

## 固定真实基线

| 指标 | 数值 |
|---|---:|
| Align 输入 | 128 |
| 原始完成 | 104 |
| 原始失败 | 24 |
| OP/ED segment | 15 |
| OP/ED 中失败 | 2 |
| 主对白 segment | 113 |
| `completed_exact` | 91 |
| `completed_coarse` | 0 |
| 主对白失败 | 22 |
| 不超过 4 字符的短应答失败 | 18 |
| 英文失败 | 1 |
| 其他失败 | 3 |
| 后置修复精确恢复 | 0/24 |
| 最终质量门 | `FAIL` |

OP/ED 继续使用 `SKIPPED_MUSIC_REGION`，不进入对白恢复目标。参考 ASS 继续只读用于评估和人工比较，不能成为 transcript 或 Align 输入。

## 已有能力决策

| 能力 | 当前状态 | 下一 Goal 处理 |
|---|---|---|
| Qwen 主 Align | 生产默认 | 保持，增加失败执行轨迹 |
| ASR short-window fallback | 生产可达、默认关闭、0/24 恢复 | 审计失败原因，不允许改写原 transcript |
| 9 组 Align 参数矩阵 | `COMPLETED_REJECTED` | 不重复运行 |
| MFA 全量替代 | `COMPLETED_REJECTED` | 不恢复 |
| MFA local fallback | `KEEP_LOCAL_FALLBACK` | 审计能否接入失败对白恢复 |
| proofread-realign | 后置修改文本专用 | 不提前代替首次失败恢复 |
| VAD coarse | 已有写回能力 | 补证据、安全和质量硬门 |
| Web 恢复队列 | 功能可用 | 把状态动作接到真实 executor |

## 下一 Goal 的完成口径

完成不等于强行让 22 条全部成为 `completed_exact`。必须做到：

- 22/22 主对白失败有机器可读执行轨迹和根因；
- 18/18 短应答经过适用的短应答恢复路由；
- 所有既有恢复能力均明确为生产可达、证据拒绝或不适用；
- 不再存在“按钮或状态存在但 executor 未执行”的无操作路径；
- 能可靠恢复的条目实际晋升为 exact 或有充分证据的 coarse；
- 不能可靠恢复的条目保持 failed 并说明原因；
- 第二套真实数据没有新增短应答缺失、内容损失、非法时间或严重重叠；
- 所有受影响下游状态和质量门真实重算；
- Web 准确显示结果，不做全站视觉重构。

若 8 小时后仍无法解释 22 条的执行路径，应停止无目标参数尝试，先生成阻塞审计并重新确定 Goal B，而不是继续扩大实验范围。

## 权威入口

- TODO 与验收：`docs/Align失败恢复既有能力审计TODO与验证计划.md`
- Goal 提示词：`docs/Align失败恢复Goal模式提示词.md`
- 机器基线：`reports/align_recovery_goal_baseline.json`
