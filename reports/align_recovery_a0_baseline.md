# Align 恢复 A0 基线冻结报告

- 状态：PASS
- 派生工作区：`workspaces\align-recovery-sayonara-lara-02-20260716-143833`
- 分支/提交：`codex/web-pipeline-workbench` / `59b78594ecc8dff5aae6a1fbd20ed93f674b1f0f`

## 固定证据校验

| 证据 | 字节 | SHA-256 状态 |
|---|---|---|
| workspaces/full-regression-sayonara-lara-02-20260715-234059/manifests/transcript_segments.json | 58175 | MATCH |
| workspaces/full-regression-sayonara-lara-02-20260715-234059/manifests/aligned_segments.json | 566667 | MATCH |
| workspaces/full-regression-sayonara-lara-02-20260715-234059/manifests/split_segments.json | 56416 | MATCH |
| workspaces/full-regression-sayonara-lara-02-20260715-234059/reports/final_quality.json | 4967 | MATCH |
| workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234/reports/post_repair_effectiveness_excluding_oped.json | 3206 | MATCH |
| reports/web_frontend_backend_final_acceptance.json | 2707 | MATCH |

## 对话基线

| 指标 | 值 |
|---|---|
| raw_align_input | 128 |
| raw_completed | 104 |
| raw_failed | 24 |
| music_region_segments | 15 |
| music_region_failed | 2 |
| dialogue_segments | 113 |
| completed_exact | 91 |
| completed_coarse | 0 |
| failed_dialogue | 22 |
| short_failures_le_4_chars | 18 |

原始两个工作区未被修改；派生工作区只复制必要 manifest、状态文件和 OP/ED 证据，音频路径继续只读引用源工作区。
