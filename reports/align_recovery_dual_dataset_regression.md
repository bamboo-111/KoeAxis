# Align 恢复双数据集回归

状态：PASS

| 数据集 | exact 前→后 | failed 前→后 | 短失败前→后 | 内容守恒 | 非法时间回退 |
|---|---:|---:|---:|---|---|
| konoato01 | 100→100 | 18→18 | 6→6 | PASS | PASS |
| madougushi02 | 116→116 | 11→11 | 9→9 | PASS | PASS |

两个固定 transcript 的 SHA-256 均与稳定快照一致；所有执行发生在新派生副本，原始 clean-regression 工作区未写入。
