# 最近完整流程诊断清理记录

## 结论

- 状态：`PASS`
- 清理编号：`recent-full-flow-cleanup-20260716-165733`
- 备份目录：`backups/recent-full-flow-cleanup-20260716-165733/`
- 仅退役 4 个本次测试的一次性脚本或缓存文件。
- 完整流程工作区、参考字幕、manifest、日志、报告、标准化结果和导出字幕全部保留。
- 未修改 Web 生产代码、字幕流程生产代码、媒体、模型、虚拟环境或历史产物。
- 未执行 Git stage、commit 或 push。

## 已退役文件

1. `workspaces/full-regression-sayonara-lara-02-20260715-234059/scripts/extract_reference_tracks.py`
2. `workspaces/full-regression-sayonara-lara-02-20260715-234059/scripts/build_full_regression_report.py`
3. `workspaces/full-regression-sayonara-lara-02-20260715-234059/scripts/__pycache__/extract_reference_tracks.cpython-312.pyc`
4. `workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234/scripts/run_downstream_diagnostic.py`

这些文件已逐个备份并在 `backup_manifest.json` 中记录原始大小与 SHA-256。空的 `scripts/` 和 `__pycache__/` 目录可随精确文件删除后一并移除；没有删除任何非空目录。

## 保留证据

以下两个工作区继续作为 Web API、恢复队列、质量门和导出界面的真实数据验收基线：

- `workspaces/full-regression-sayonara-lara-02-20260715-234059/`
- `workspaces/post-repair-full-flow-sayonara-lara-02-20260716-002234/`

关键事实保持不变：正式 Align 为 104/128 完成、24 失败；排除 OP/ED 后为 91/113 完成、22 失败，其中 18 条为不超过 4 个规范化字符的短应答。后置 Normalize/Export 可修复时间合法性和重叠，但没有恢复任何失败片段的精确对齐，因此 Web 必须把失败片段恢复作为独立工作流展示，不能把最终导出成功等同于质量门通过。

## 后续入口

- 计划与验收：`docs/Web前后端整理TODO与验证计划.md`
- Goal 提示词：`docs/Web前后端整理Goal模式提示词.md`
- 机器清单：`reports/recent_full_flow_cleanup.json`

## 验证结果

- 4 个批准退役文件均已不存在。
- 7 个抽查的关键报告、SRT/VTT 导出和总结文件全部存在。
- 6 个备份对象的源文件 SHA-256 与备份 SHA-256 全部一致。
- 清理 JSON、关键诊断 JSON、中文文档 UTF-8 和关键标题/标记检查通过。
- `tests/test_webui.py`：`45 passed in 0.17s`。
- `git diff --check`：通过；仅有当前工作树既有的 LF/CRLF 转换提示。
