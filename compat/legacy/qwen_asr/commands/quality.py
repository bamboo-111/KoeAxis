from __future__ import annotations

import argparse

from qwen_asr.content_quality import evaluate_content_conservation
from qwen_asr.final_quality import evaluate_final_quality
from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress


def cmd_content_quality(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report = evaluate_content_conservation(
        work_paths,
        include_export=bool(getattr(args, "include_export", False)),
    )
    write_progress(
        work_paths,
        stage="content-quality",
        status="completed" if report["status"] != "FAIL" else "failed",
        summary=(
            f"content-quality {report['status']}: "
            f"{report['summary']['fail_count']} FAIL, {report['summary']['warn_count']} WARN"
        ),
    )
    return 0 if report["status"] != "FAIL" else 1


def cmd_quality_gate(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report = evaluate_final_quality(
        work_paths,
        include_export=bool(getattr(args, "include_export", False)),
        require_srt=bool(getattr(args, "require_srt", False)),
    )
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    write_progress(
        work_paths,
        stage="quality-gate",
        status="completed" if report["status"] != "FAIL" else "failed",
        done=int(summary.get("pass_count", 0) or 0),
        total=(
            int(summary.get("pass_count", 0) or 0)
            + int(summary.get("warn_count", 0) or 0)
            + int(summary.get("fail_count", 0) or 0)
        ),
        summary=(
            f"聚合质量门 {report['status']}："
            f"{summary.get('fail_count', 0)} FAIL，{summary.get('warn_count', 0)} WARN"
        ),
    )
    return 0 if report["status"] != "FAIL" else 1
