from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from qwen_asr.mimo_candidates import collect_stage1_suspects, translated_duration_ms
from qwen_asr.storage import write_json_atomic


def write_outputs(
    manifest_path: Path,
    report_path: Path,
    srt_path: Path,
    branch: dict[str, Any],
    report: list[dict[str, Any]],
) -> None:
    write_json_atomic(manifest_path, branch)
    write_json_atomic(report_path, report)
    srt_path.write_text(to_srt(branch), encoding="utf-8")


def write_two_stage_outputs(
    manifest_path: Path,
    report_path: Path,
    stage1_report_path: Path,
    stage2_report_path: Path,
    srt_path: Path,
    branch: dict[str, Any],
    stage1_report: list[dict[str, Any]],
    stage2_report: list[dict[str, Any]],
    started: float | None = None,
    translated: dict[str, Any] | None = None,
) -> None:
    elapsed_ms = int((time.monotonic() - started) * 1000) if started is not None else 0
    suspect_ids = collect_stage1_suspects(stage1_report)
    total_duration_ms = translated_duration_ms(translated or branch)
    video_minutes = total_duration_ms / 60000 if total_duration_ms > 0 else 0.0
    completed_audio = [item for item in stage2_report if item.get("status") == "completed"]
    failed_audio = [item for item in stage2_report if item.get("status") == "failed"]
    reviewed_count = stage2_reviewed_candidate_count(completed_audio)
    applied_count = sum(int(item.get("applied_count", 0) or 0) for item in completed_audio)
    rejected_count = sum(int(item.get("rejected_count", 0) or 0) for item in completed_audio)
    unresolved_count = len(failed_audio) + sum(
        1 for item in completed_audio
        if int(item.get("suggestion_count", 0) or 0) == 0 and str(item.get("raw_content", "")).strip()
    )
    write_json_atomic(manifest_path, branch)
    write_json_atomic(stage1_report_path, stage1_report)
    write_json_atomic(stage2_report_path, stage2_report)
    write_json_atomic(
        report_path,
        {
            "mode": "two-stage-nearby",
            "stage1_report": str(stage1_report_path),
            "stage2_report": str(stage2_report_path),
            "stage1_completed": sum(1 for item in stage1_report if item.get("status") == "completed"),
            "stage1_failed": sum(1 for item in stage1_report if item.get("status") == "failed"),
            "stage1_suspect_count": len(suspect_ids),
            "stage2_completed": reviewed_count,
            "stage2_completed_batches": len(completed_audio),
            "stage2_failed": len(failed_audio),
            "audio_review_candidate_count": len(suspect_ids),
            "audio_review_candidates_per_minute": (
                round(len(suspect_ids) / video_minutes, 3) if video_minutes > 0 else 0.0
            ),
            "audio_review_elapsed_ms": elapsed_ms,
            "audio_review_ms_per_completed_candidate": (
                round(elapsed_ms / reviewed_count, 1) if reviewed_count else 0.0
            ),
            "audio_review_applied_count": applied_count,
            "audio_review_rejected_count": rejected_count,
            "audio_review_unresolved_count": unresolved_count,
        },
    )
    srt_path.write_text(to_srt(branch), encoding="utf-8")


def stage2_reviewed_candidate_count(completed_audio: list[dict[str, Any]]) -> int:
    reviewed: set[str] = set()
    for item in completed_audio:
        target_ids = item.get("target_ids", [])
        if isinstance(target_ids, list) and target_ids:
            reviewed.update(str(value) for value in target_ids if str(value).strip())
        else:
            item_id = str(item.get("id", "")).strip()
            if item_id:
                reviewed.add(item_id)
    return len(reviewed)


def replace_report_item(
    report: list[dict[str, Any]],
    item_id: str,
    item: dict[str, Any],
    *,
    key: str = "segment_id",
) -> list[dict[str, Any]]:
    return [
        existing for existing in report
        if str(existing.get(key, "")) != item_id
    ] + [item]


def to_srt(items: dict[str, Any]) -> str:
    lines: list[str] = []
    index = 1
    for _, item in sorted(items.items(), key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else str(pair[0])):
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_time", 0))
        end = int(item.get("end_time", start + 1))
        original = str(item.get("original_subtitle", "")).strip()
        translated = str(item.get("translated_subtitle", "")).strip()
        text = original if not translated else f"{original}\n{translated}"
        lines.extend([str(index), f"{srt_time(start)} --> {srt_time(end)}", text, ""])
        index += 1
    return "\n".join(lines).strip() + "\n"


def srt_time(ms: int) -> str:
    hours, rem = divmod(max(0, ms), 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
