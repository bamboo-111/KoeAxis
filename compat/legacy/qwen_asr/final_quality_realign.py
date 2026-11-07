from __future__ import annotations

from typing import Any

from qwen_asr.final_quality_common import fail, normalize_status, passed, skip, warn
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def proofread_realign_check(work_paths: WorkPaths) -> dict[str, Any]:
    if not work_paths.mimo_proofread_manifest.exists():
        return skip("proofread_realign", "未运行音频复核，无需重对齐")
    payload = read_json(work_paths.mimo_proofread_manifest, default={})
    if not isinstance(payload, dict):
        return fail("proofread_realign", "音频复核字幕清单格式无效")
    pending = [
        str(key)
        for key, item in payload.items()
        if isinstance(item, dict)
        and bool(item.get("needs_realign"))
        and str(item.get("realign_status", "")).strip() != "completed"
    ]
    failed = [
        str(key)
        for key, item in payload.items()
        if isinstance(item, dict) and str(item.get("realign_status", "")).strip() == "failed"
    ]
    if pending or failed:
        return fail(
            "proofread_realign",
            f"音频复核修改后的重对齐未完成：待处理 {len(pending)} 条，失败 {len(failed)} 条",
            pending_ids=pending[:20],
            failed_ids=failed[:20],
        )
    report_path = work_paths.workdir / "reports" / "proofread_realign.json"
    if report_path.exists():
        report = read_json(report_path, default={})
        if isinstance(report, dict):
            status = normalize_status(report.get("status", "PASS"))
            if status == "FAIL":
                return fail("proofread_realign", "重对齐报告为失败", report=str(report_path))
            if status == "WARN":
                fallback_count = int(report.get("fallback_count", 0) or 0)
                mfa_completed_count = int(report.get("mfa_completed_count", 0) or 0)
                mfa_unusable_count = int(report.get("mfa_unusable_count", 0) or 0)
                mfa_rejected_count = int(report.get("mfa_rejected_count", 0) or 0)
                return warn(
                    "proofread_realign",
                    (
                        f"重对齐报告为警告：降级 {fallback_count} 条，"
                        f"MFA 成功 {mfa_completed_count} 条，"
                        f"MFA 不可用 {mfa_unusable_count} 条，"
                        f"MFA 拒绝 {mfa_rejected_count} 条"
                    ),
                    report=str(report_path),
                    fallback_count=fallback_count,
                    mfa_completed_count=mfa_completed_count,
                    mfa_unusable_count=mfa_unusable_count,
                    mfa_rejected_count=mfa_rejected_count,
                )
    return passed("proofread_realign", "音频复核修改后的重对齐状态通过")
