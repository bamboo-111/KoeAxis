from __future__ import annotations

from typing import Any

from qwen_asr.align import validate_aligned_token_timing
from qwen_asr.alignment_state import derive_alignment_state, overlaps_music_region, read_music_region_evidence
from qwen_asr.final_quality_common import fail, float_or_none, passed, skip, warn
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


class SimpleToken:
    def __init__(self, *, text: str, start_time: float, end_time: float) -> None:
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


def alignment_health_check(work_paths: WorkPaths) -> dict[str, Any]:
    if not work_paths.aligned_manifest.exists():
        return skip("alignment_health", "未运行 align 阶段")
    payload = read_json(work_paths.aligned_manifest, default=[])
    if not isinstance(payload, list) or not payload:
        return fail("alignment_health", "aligned_segments.json 缺失或为空")

    completed_exact = 0
    completed_coarse = 0
    skipped_music_region = 0
    failed: list[str] = []
    timing_errors: list[dict[str, Any]] = []
    one_ms_segments: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    music_regions, music_evidence_path, _, music_evidence_error = read_music_region_evidence(work_paths.workdir)
    for item in payload:
        if not isinstance(item, dict):
            continue
        segment_id = str(item.get("segment_id", ""))
        if overlaps_music_region(item, music_regions):
            skipped_music_region += 1
            continue
        alignment_state = derive_alignment_state(item)
        if alignment_state == "failed":
            failed.append(segment_id or f"#{len(failed) + 1}")
            continue
        if alignment_state == "completed_coarse":
            completed_coarse += 1
            continue
        completed_exact += 1
        start = float_or_none(item.get("global_start_time"))
        end = float_or_none(item.get("global_end_time"))
        raw_tokens = item.get("tokens", [])
        tokens = [token for token in raw_tokens if isinstance(token, dict) and str(token.get("text", "")).strip()]
        if start is None or end is None or end <= start:
            timing_errors.append({"segment_id": segment_id, "error": "invalid segment bounds"})
            continue
        token_objects = [
            SimpleToken(
                text=str(token.get("text", "")),
                start_time=float(token.get("start_time", 0.0)),
                end_time=float(token.get("end_time", token.get("start_time", 0.0))),
            )
            for token in tokens
        ]
        timing_error = validate_aligned_token_timing(token_objects, start, end)
        if timing_error:
            timing_errors.append({"segment_id": segment_id, "error": timing_error})
        coverage = alignment_coverage(token_objects, start, end)
        coverage_values.append(coverage)
        one_ms_count, max_one_ms_run = one_ms_token_stats(token_objects)
        if max_one_ms_run >= 3 or one_ms_count >= 5:
            one_ms_segments.append(
                {
                    "segment_id": segment_id,
                    "one_ms_token_count": one_ms_count,
                    "max_one_ms_run": max_one_ms_run,
                }
            )

    average_coverage = sum(coverage_values) / max(1, len(coverage_values))
    low_coverage = bool(coverage_values) and average_coverage < 0.95
    if failed or timing_errors:
        return fail(
            "alignment_health",
            (
                f"对齐健康检查 FAIL：精确完成 {completed_exact} 条，粗略完成 {completed_coarse} 条，"
                f"失败 {len(failed)} 条，"
                f"时间异常 {len(timing_errors)} 条，平均覆盖率 {average_coverage:.3f}"
            ),
            completed_count=completed_exact + completed_coarse,
            completed_exact_count=completed_exact,
            completed_coarse_count=completed_coarse,
            failed_count=len(failed),
            skipped_music_region_count=skipped_music_region,
            music_region_evidence_path=music_evidence_path,
            music_region_evidence_error=music_evidence_error,
            timing_error_count=len(timing_errors),
            one_ms_cluster_count=len(one_ms_segments),
            average_coverage=round(average_coverage, 6),
            failed_segment_ids=failed[:20],
            timing_errors=timing_errors[:20],
            one_ms_segments=one_ms_segments[:20],
        )
    if completed_coarse or low_coverage or one_ms_segments:
        return warn(
            "alignment_health",
            (
                f"对齐健康检查 WARN：精确完成 {completed_exact} 条，粗略完成 {completed_coarse} 条，"
                f"平均覆盖率 {average_coverage:.3f}，1ms token 簇 {len(one_ms_segments)} 条"
            ),
            completed_count=completed_exact + completed_coarse,
            completed_exact_count=completed_exact,
            completed_coarse_count=completed_coarse,
            failed_count=0,
            skipped_music_region_count=skipped_music_region,
            music_region_evidence_path=music_evidence_path,
            music_region_evidence_error=music_evidence_error,
            timing_error_count=0,
            one_ms_cluster_count=len(one_ms_segments),
            average_coverage=round(average_coverage, 6),
            one_ms_segments=one_ms_segments[:20],
        )
    return passed(
        "alignment_health",
        f"对齐健康检查通过：精确完成 {completed_exact} 条，平均覆盖率 {average_coverage:.3f}",
        completed_count=completed_exact,
        completed_exact_count=completed_exact,
        completed_coarse_count=0,
        failed_count=0,
        skipped_music_region_count=skipped_music_region,
        music_region_evidence_path=music_evidence_path,
        music_region_evidence_error=music_evidence_error,
        timing_error_count=0,
        one_ms_cluster_count=0,
        average_coverage=round(average_coverage, 6),
    )


def alignment_coverage(tokens: list[SimpleToken], start: float, end: float) -> float:
    duration = max(0.0, end - start)
    if duration <= 0:
        return 0.0
    covered = sum(max(0.0, min(end, token.end_time) - max(start, token.start_time)) for token in tokens)
    return min(1.0, covered / duration)


def one_ms_token_stats(tokens: list[SimpleToken]) -> tuple[int, int]:
    count = 0
    run = 0
    max_run = 0
    for token in tokens:
        duration_ms = max(0.0, token.end_time - token.start_time) * 1000
        if 0 < duration_ms <= 1.0:
            count += 1
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return count, max_run
