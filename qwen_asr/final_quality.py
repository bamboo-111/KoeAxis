from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_asr import final_quality_alignment as _alignment
from qwen_asr import final_quality_ass as _ass
from qwen_asr import final_quality_content as _content
from qwen_asr import final_quality_mimo as _mimo
from qwen_asr import final_quality_postproofread as _postproofread
from qwen_asr import final_quality_readability as _readability
from qwen_asr import final_quality_realign as _realign
from qwen_asr.content_quality import evaluate_content_conservation
from qwen_asr import final_quality_srt as _srt
from qwen_asr import final_quality_stage as _stage
from qwen_asr import final_quality_translation as _translation
from qwen_asr import final_quality_common as _common
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic

_normalize_status = _common.normalize_status
_pass = _common.passed
_warn = _common.warn
_fail = _common.fail
_skip = _common.skip


SRT_TIMESTAMP_RE = _srt.SRT_TIMESTAMP_RE
validate_srt = _srt.validate_srt
_srt_legality_check = _srt.srt_legality_check
_parse_srt_timestamp = _srt.parse_srt_timestamp
_srt_issue = _srt.srt_issue
POST_PROOFREAD_MIN_RETENTION = _postproofread.POST_PROOFREAD_MIN_RETENTION
SHORT_RESPONSE_GUARD_TEXTS = _postproofread.SHORT_RESPONSE_GUARD_TEXTS
MIN_ORDINARY_SUBTITLE_DURATION_MS = _readability.MIN_ORDINARY_SUBTITLE_DURATION_MS
MIN_PROTECTED_SHORT_SUBTITLE_DURATION_MS = _readability.MIN_PROTECTED_SHORT_SUBTITLE_DURATION_MS
PROTECTED_SHORT_SUBTITLE_NORMALIZED = _readability.PROTECTED_SHORT_SUBTITLE_NORMALIZED


def cmd_quality_gate(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report = evaluate_final_quality(
        work_paths,
        include_export=bool(getattr(args, "include_export", False)),
        require_srt=bool(getattr(args, "require_srt", False)),
    )
    return 0 if report["status"] != "FAIL" else 1


def evaluate_final_quality(
    work_paths: WorkPaths,
    *,
    include_export: bool = False,
    require_srt: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(_content_quality_check(work_paths, include_export=include_export))
    checks.append(_alignment_health_check(work_paths))
    checks.append(_normalize_export_content_check(work_paths))
    checks.append(_subtitle_readability_check(work_paths))
    checks.extend(_ass_quality_checks(work_paths))
    checks.append(_translation_completeness_check(work_paths))
    checks.append(_translation_structure_check(work_paths))
    checks.append(_mimo_checkpoint_check(work_paths))
    checks.append(_post_proofread_guard_check(work_paths))
    checks.append(_proofread_realign_check(work_paths))
    checks.append(_stage_checkpoint_check(work_paths))
    if include_export or require_srt or work_paths.subtitles_srt.exists():
        checks.append(_srt_legality_check(work_paths, require_srt=require_srt))

    status = _rollup_status(checks)
    report = {
        "status": status,
        "include_export": include_export,
        "require_srt": require_srt,
        "checks": checks,
        "summary": {
            "fail_count": sum(item["status"] == "FAIL" for item in checks),
            "warn_count": sum(item["status"] == "WARN" for item in checks),
            "pass_count": sum(item["status"] == "PASS" for item in checks),
        },
    }
    write_json_atomic(work_paths.final_quality_report, report)
    return report


def _content_quality_check(work_paths: WorkPaths, *, include_export: bool) -> dict[str, Any]:
    report = evaluate_content_conservation(work_paths, include_export=include_export)
    return {
        "name": "content_quality",
        "status": report["status"],
        "message": (
            f"内容守恒 {report['status']}："
            f"{report['summary']['fail_count']} 个失败，{report['summary']['warn_count']} 个警告"
        ),
        "report": str(work_paths.content_quality_report),
    }


def _alignment_health_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _alignment.alignment_health_check(work_paths)


def _normalize_export_content_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _content.normalize_export_content_check(work_paths)


def _subtitle_readability_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _readability.subtitle_readability_check(work_paths, manifest_key_sort=_manifest_key_sort)


def _is_protected_short_subtitle(text: str) -> bool:
    return _readability.is_protected_short_subtitle(text)


def _best_pre_normalize_stage_text(work_paths: WorkPaths) -> tuple[str, str | None]:
    return _content.best_pre_normalize_stage_text(work_paths)


def _normalized_stage_text(path: Path) -> str | None:
    return _content.normalized_stage_text(path)


def _manifest_item_text(kind: str, item: dict[str, Any]) -> str:
    return _content.manifest_item_text(kind, item)


def _srt_stage_text(path: Path) -> str | None:
    return _content.srt_stage_text(path)


def _ass_quality_checks(work_paths: WorkPaths) -> list[dict[str, Any]]:
    return _ass.ass_quality_checks(work_paths)


def _translation_structure_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _translation.translation_structure_check(work_paths)


def _translation_completeness_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _translation.translation_completeness_check(work_paths, manifest_key_sort=_manifest_key_sort)


def _mimo_checkpoint_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _mimo.mimo_checkpoint_check(work_paths)


def _mimo_two_stage_completed_count(report: dict[str, Any], work_paths: WorkPaths) -> int:
    return _mimo.mimo_two_stage_completed_count(report, work_paths)


def _pending_audio_review_count(work_paths: WorkPaths) -> int:
    return _mimo.pending_audio_review_count(work_paths)


def _quality_suspect_applied_count(work_paths: WorkPaths) -> int:
    return _mimo.quality_suspect_applied_count(work_paths)


def _mimo_applied_without_evidence_count(work_paths: WorkPaths) -> int:
    return _mimo.mimo_applied_without_evidence_count(work_paths)


def _post_proofread_guard_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _postproofread.post_proofread_guard_check(work_paths)


def _is_mimo_original_change(entry: Any) -> bool:
    return _postproofread.is_mimo_original_change(entry)


def _post_proofread_original_change_issue(subtitle_id: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    return _postproofread.post_proofread_original_change_issue(subtitle_id, entry)


def _post_proofread_content_regressed(before: str, after: str) -> bool:
    return _postproofread.post_proofread_content_regressed(before, after)


def _post_guard_issue(subtitle_id: str, issue_type: str, message: str, **extra: Any) -> dict[str, Any]:
    return _postproofread.post_guard_issue(subtitle_id, issue_type, message, **extra)


def _proofread_realign_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _realign.proofread_realign_check(work_paths)


def _stage_checkpoint_check(work_paths: WorkPaths) -> dict[str, Any]:
    return _stage.stage_checkpoint_check(work_paths)


def _has_checkpoint_artifact(work_paths: WorkPaths, stage: str) -> bool:
    return _stage.has_checkpoint_artifact(work_paths, stage)


def _rollup_status(checks: list[dict[str, Any]]) -> str:
    if any(item["status"] == "FAIL" for item in checks):
        return "FAIL"
    if any(item["status"] == "WARN" for item in checks):
        return "WARN"
    return "PASS"


_SimpleToken = _alignment.SimpleToken


def _float_or_none(value: Any) -> float | None:
    return _alignment.float_or_none(value)


def _int_or_none(value: Any) -> int | None:
    return _readability.int_or_none(value)


def _subtitle_display_text(item: dict[str, Any]) -> str:
    return _readability.subtitle_display_text(item)


def _alignment_coverage(tokens: list[_SimpleToken], start: float, end: float) -> float:
    return _alignment.alignment_coverage(tokens, start, end)


def _one_ms_token_stats(tokens: list[_SimpleToken]) -> tuple[int, int]:
    return _alignment.one_ms_token_stats(tokens)


def _manifest_key_sort(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _subtitle_readability_issue(
    severity: str,
    stage: str,
    key: str,
    kind: str,
    message: str,
) -> dict[str, Any]:
    return _readability.subtitle_readability_issue(severity, stage, key, kind, message)
