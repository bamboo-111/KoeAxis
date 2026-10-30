from __future__ import annotations

from typing import Any


def render_mfa_alignment_experiment_markdown(report: dict[str, Any]) -> str:
    environment = report.get("environment", {}) if isinstance(report.get("environment"), dict) else {}
    lines = [
        "# MFA 3.0 局部对齐实验诊断",
        "",
        f"- 状态：{report.get('status', '')}",
        f"- 原因：{report.get('reason', '')}",
        f"- MFA 可用：{environment.get('available', False)}",
        f"- mfa 可执行文件：{environment.get('executable', '')}",
        f"- 调用方式：{environment.get('invocation', '')}",
        f"- 调用命令：{' '.join(str(value) for value in environment.get('command', [])) if isinstance(environment.get('command'), list) else ''}",
        f"- Python 包版本：{environment.get('package_version', '')}",
        f"- 版本输出：{environment.get('version_output', '')}",
        f"- 候选数量：{report.get('candidate_count', 0)}",
        "",
        "## 通过标准",
        "",
    ]
    criteria = report.get("pass_criteria", {})
    if isinstance(criteria, dict):
        for key, value in criteria.items():
            lines.append(f"- {key}：{value}")
    lines.extend(["", "## 局部候选", ""])
    lines.append("| 序号 | 严重度 | 来源 | 原因 | 时间 | 文本 |")
    lines.append("|---:|---|---|---|---|---|")
    for index, item in enumerate(report.get("candidates", []), 1):
        if not isinstance(item, dict):
            continue
        time_text = format_time_range(item.get("start_ms"), item.get("end_ms"))
        text = str(item.get("text", "")).replace("|", "｜")
        lines.append(
            f"| {index} | {item.get('severity', '')} | {item.get('source', '')} | "
            f"{item.get('reason', '')} | {time_text} | {text[:80]} |"
        )
    lines.append("")
    local_run = report.get("local_alignment_run", {})
    if isinstance(local_run, dict) and local_run.get("enabled"):
        lines.extend(["## 局部 MFA 实跑", ""])
        lines.append("| 序号 | 状态 | usable | lab 来源 | lab 文本 | guard | 分数 | dry-run | Δcurrent | MFA 时间 | 原因 |")
        lines.append("|---:|---|---|---|---|---|---:|---|---:|---|---|")
        for index, item in enumerate(local_run.get("items", []), 1):
            if not isinstance(item, dict):
                continue
            guard = item.get("local_ass_guard", {}) if isinstance(item.get("local_ass_guard"), dict) else {}
            dry_run = item.get("writeback_dry_run", {}) if isinstance(item.get("writeback_dry_run"), dict) else {}
            lab_text = str(item.get("lab_text", "")).replace("|", "｜")
            time_text = format_time_range(guard.get("mfa_start_ms"), guard.get("mfa_end_ms"))
            reasons = ",".join(str(value) for value in (dry_run.get("reasons") or guard.get("reasons", [])) if value)
            lines.append(
                f"| {index} | {item.get('status', '')} | {item.get('usable', '')} | "
                f"{item.get('lab_text_source', '')} | {lab_text[:40]} | {guard.get('status', '')} | "
                f"{guard.get('text_score', '')} | {dry_run.get('status', '')} | "
                f"{dry_run.get('score_delta_vs_current', '')} | {time_text} | {reasons} |"
            )
        lines.append("")
    writeback = report.get("local_writeback", {})
    if isinstance(writeback, dict) and writeback.get("enabled"):
        lines.extend(["## 局部 MFA 写回评估", ""])
        lines.append(f"- 模式：{writeback.get('mode', '')}")
        lines.append(f"- 状态：{writeback.get('status', '')}")
        lines.append(f"- 应用数量：{writeback.get('applied_count', 0)}")
        lines.append(f"- 拒绝数量：{writeback.get('rejected_count', 0)}")
        if writeback.get("output_manifest"):
            lines.append(f"- 输出 manifest：{writeback.get('output_manifest')}")
        lines.extend(["", "| 序号 | 状态 | id | manifest 文本 | MFA 文本 | 分数 | 原时间 | 新时间 | 原因 |"])
        lines.append("|---:|---|---|---|---|---:|---|---|---|")
        for index, item in enumerate(writeback.get("items", []), 1):
            if not isinstance(item, dict):
                continue
            reasons = ",".join(str(value) for value in item.get("reasons", []) if value)
            old_time = format_time_range(item.get("old_start_ms"), item.get("old_end_ms"))
            new_time = format_time_range(item.get("new_start_ms"), item.get("new_end_ms"))
            manifest_text = str(item.get("manifest_text", "")).replace("|", "｜")
            mfa_text = str(item.get("mfa_text", "")).replace("|", "｜")
            lines.append(
                f"| {index} | {item.get('status', '')} | {item.get('subtitle_id', '')} | "
                f"{manifest_text[:40]} | {mfa_text[:40]} | {item.get('manifest_text_score', '')} | "
                f"{old_time} | {new_time} | {reasons} |"
            )
        lines.append("")
    return "\n".join(lines)


def format_time_range(start_ms: Any, end_ms: Any) -> str:
    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        return ""
    return f"{format_ms(start_ms)}-{format_ms(end_ms)}"


def format_ms(value: int) -> str:
    total_seconds, ms = divmod(max(0, value), 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"
