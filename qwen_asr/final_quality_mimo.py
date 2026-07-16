from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr.final_quality_common import fail, normalize_status, passed, skip
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def mimo_checkpoint_check(work_paths: WorkPaths) -> dict[str, Any]:
    pending_audio_review = pending_audio_review_count(work_paths)
    if not work_paths.mimo_proofread_report.exists():
        if pending_audio_review:
            return fail(
                "mimo_checkpoint",
                f"存在 {pending_audio_review} 条音频复核疑点，但未发现 MiMo 复核报告",
                pending_audio_review=pending_audio_review,
            )
        return skip("mimo_checkpoint", "未运行 MiMo 疑点音频复核")
    report = read_json(work_paths.mimo_proofread_report, default={})
    if not isinstance(report, dict):
        if (
            isinstance(report, list)
            and report
            and all(isinstance(item, dict) and item.get("status") == "completed" for item in report)
        ):
            return passed("mimo_checkpoint", f"旧版 MiMo 报告通过：{len(report)} 条")
        return fail("mimo_checkpoint", "MiMo 报告缺失或格式无效")

    mode = str(report.get("mode", "") or "")
    if mode == "two-stage-nearby":
        candidate_count = int(report.get("audio_review_candidate_count", report.get("stage1_suspect_count", 0)) or 0)
        completed_count = mimo_two_stage_completed_count(report, work_paths)
        failed_count = int(report.get("stage1_failed", 0) or 0) + int(report.get("stage2_failed", 0) or 0)
        unresolved_count = int(report.get("unresolved_count", 0) or 0)
        suspect_report_count = quality_suspect_applied_count(work_paths)
        expected_candidate_count = pending_audio_review or suspect_report_count
        if expected_candidate_count and candidate_count < expected_candidate_count:
            return fail(
                "mimo_checkpoint",
                (f"MiMo 复核候选少于已标注疑点：疑点 {expected_candidate_count}，MiMo 候选 {candidate_count}"),
                expected_candidate_count=expected_candidate_count,
                candidate_count=candidate_count,
                pending_audio_review=pending_audio_review,
                suspect_report_count=suspect_report_count,
            )
        if failed_count or completed_count < candidate_count:
            return fail(
                "mimo_checkpoint",
                f"MiMo 疑点复核未完成：候选 {candidate_count}，完成 {completed_count}，失败 {failed_count}",
                candidate_count=candidate_count,
                completed_count=completed_count,
                failed_count=failed_count,
            )
        missing_evidence_count = mimo_applied_without_evidence_count(work_paths)
        status = "WARN" if unresolved_count or missing_evidence_count else "PASS"
        return {
            "name": "mimo_checkpoint",
            "status": status,
            "message": (
                f"MiMo 疑点复核 {status}：候选 {candidate_count}，完成 {completed_count}"
                f"，未解决 {unresolved_count}，缺少应用证据 {missing_evidence_count}"
            ),
            "candidate_count": candidate_count,
            "completed_count": completed_count,
            "unresolved_count": unresolved_count,
            "missing_evidence_count": missing_evidence_count,
        }

    status = normalize_status(report.get("status", "PASS"))
    failed_count = int(report.get("failed_count", 0) or 0)
    if status == "FAIL" or failed_count:
        return fail("mimo_checkpoint", f"MiMo 报告失败：{failed_count} 条失败", failed_count=failed_count)
    return passed("mimo_checkpoint", f"MiMo 报告通过：模式 {mode or 'unknown'}")


def mimo_two_stage_completed_count(report: dict[str, Any], work_paths: WorkPaths) -> int:
    stage2_report: Any = None
    stage2_report_value = str(report.get("stage2_report", "") or "").strip()
    if stage2_report_value:
        stage2_report_path = Path(stage2_report_value)
        if not stage2_report_path.is_absolute():
            stage2_report_path = work_paths.mimo_proofread_dir / stage2_report_path
        if stage2_report_path.is_file():
            stage2_report = read_json(stage2_report_path, default=None)
    if isinstance(stage2_report, list):
        reviewed: set[str] = set()
        for item in stage2_report:
            if not isinstance(item, dict) or item.get("status") != "completed":
                continue
            target_ids = item.get("target_ids", [])
            if isinstance(target_ids, list) and target_ids:
                reviewed.update(str(value) for value in target_ids if str(value).strip())
            else:
                item_id = str(item.get("id", "")).strip()
                if item_id:
                    reviewed.add(item_id)
        if reviewed:
            return len(reviewed)
    return int(report.get("stage2_completed", 0) or 0)


def pending_audio_review_count(work_paths: WorkPaths) -> int:
    if not work_paths.translated_manifest.exists():
        return 0
    payload = read_json(work_paths.translated_manifest, default={})
    if not isinstance(payload, dict):
        return 0
    return sum(
        1
        for item in payload.values()
        if isinstance(item, dict) and (bool(item.get("needs_audio_review")) or bool(item.get("asr_suspect")))
    )


def quality_suspect_applied_count(work_paths: WorkPaths) -> int:
    candidates = [
        work_paths.workdir / "reports" / "quality_suspects.json",
        work_paths.workdir / "quality_suspects.json",
    ]
    counts: list[int] = []
    for path in candidates:
        if not path.exists():
            continue
        payload = read_json(path, default={})
        if isinstance(payload, dict):
            counts.append(int(payload.get("applied_count", payload.get("candidate_count", 0)) or 0))
    return max(counts, default=0)


def mimo_applied_without_evidence_count(work_paths: WorkPaths) -> int:
    if not work_paths.mimo_proofread_manifest.exists():
        return 0
    payload = read_json(work_paths.mimo_proofread_manifest, default={})
    if not isinstance(payload, dict):
        return 0
    missing = 0
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        history = item.get("proofread_history", [])
        if not isinstance(history, list):
            continue
        for entry in history:
            if not isinstance(entry, dict):
                continue
            source = str(entry.get("source", ""))
            changes = entry.get("changes", {})
            if (
                source.startswith("mimo-")
                and isinstance(changes, dict)
                and changes
                and not isinstance(entry.get("evidence"), dict)
            ):
                missing += 1
    return missing
