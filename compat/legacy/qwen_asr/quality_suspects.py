from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


QUALITY_SUSPECT_TYPES = {
    "short-dialogue-missing": "ass_short_dialogue_missing",
    "short-dialogue-timing-shifted": "ass_short_dialogue_timing_shifted",
}
QUALITY_LOW_SCORE_TYPE = "ass_low_score"
QUALITY_FAIL_SCORE_TYPE = "ass_fail_score"
QUALITY_DIFF_SUSPECT_TYPES = {
    "became-fail": "ass_stage_became_fail",
    "became-low": "ass_stage_became_low",
    "diagnostic-added": "ass_stage_diagnostic_added",
    "score-drop": "ass_stage_score_drop",
}
QUALITY_DIFF_MIN_SCORE_DROP = 0.50
REALIGN_ONLY_SUSPECT_TYPES = {
    "ass_short_dialogue_timing_shifted",
    QUALITY_LOW_SCORE_TYPE,
    QUALITY_FAIL_SCORE_TYPE,
}


def cmd_apply_quality_suspects(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report_paths = [Path(value) for value in _list_arg(getattr(args, "ass_quality_report", []) or [])]
    diff_report_paths = [Path(value) for value in _list_arg(getattr(args, "ass_quality_diff_report", []) or [])]
    if not report_paths and not diff_report_paths:
        raise RuntimeError("At least one quality report is required.")
    translated = read_json(work_paths.translated_manifest, default={})
    if not isinstance(translated, dict) or not translated:
        raise RuntimeError("translated_segments.json is missing or empty")

    result: dict[str, Any] = {
        "translated": translated,
        "report": {
            "source": "quality-suspects",
            "candidate_count": 0,
            "applied_count": 0,
            "reports": [],
        },
    }
    for report_path in report_paths:
        report = read_json(report_path, default={})
        if not isinstance(report, dict):
            raise RuntimeError(f"ASS quality report is not an object: {report_path}")
        stage_result = apply_quality_suspects_to_translated(
            result["translated"],
            report,
            max_distance_ms=int(getattr(args, "quality_suspect_max_distance_ms", 8000)),
        )
        result = _merge_apply_results(result, stage_result, report_path=report_path, source="ass-quality")
    for report_path in diff_report_paths:
        report = read_json(report_path, default={})
        if not isinstance(report, dict):
            raise RuntimeError(f"ASS quality diff report is not an object: {report_path}")
        stage_result = apply_quality_diff_suspects_to_translated(
            result["translated"],
            report,
            max_distance_ms=int(getattr(args, "quality_suspect_max_distance_ms", 8000)),
        )
        result = _merge_apply_results(result, stage_result, report_path=report_path, source="ass-quality-diff")
    write_json_atomic(work_paths.translated_manifest, result["translated"])
    output = str(getattr(args, "quality_suspect_report_output", "") or "").strip()
    if output:
        write_json_atomic(Path(output), result["report"])
    print(
        "\u8d28\u91cf\u95e8\u7591\u70b9\u5df2\u6ce8\u5165\uff1a"
        f"{result['report']['applied_count']}/{result['report']['candidate_count']}"
    )
    return 0


def _list_arg(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    return []


def _merge_apply_results(
    aggregate: dict[str, Any],
    stage_result: dict[str, Any],
    *,
    report_path: Path,
    source: str,
) -> dict[str, Any]:
    stage_report = dict(stage_result.get("report", {}))
    stage_report["path"] = str(report_path)
    stage_report["source"] = source
    aggregate["translated"] = stage_result["translated"]
    aggregate_report = aggregate["report"]
    aggregate_report["candidate_count"] = int(aggregate_report.get("candidate_count", 0)) + int(
        stage_report.get("candidate_count", 0)
    )
    aggregate_report["applied_count"] = int(aggregate_report.get("applied_count", 0)) + int(
        stage_report.get("applied_count", 0)
    )
    reports = aggregate_report.setdefault("reports", [])
    if isinstance(reports, list):
        reports.append(stage_report)
    return aggregate


def apply_quality_suspects_to_translated(
    translated: dict[str, Any],
    ass_quality_report: dict[str, Any],
    *,
    max_distance_ms: int = 8000,
) -> dict[str, Any]:
    rows = ass_quality_report.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    updated = {
        str(key): dict(value) if isinstance(value, dict) else value
        for key, value in translated.items()
    }
    candidates: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    subtitle_index = _translated_time_index(updated)

    for row in rows:
        if not isinstance(row, dict):
            continue
        matched_types = _quality_row_suspect_types(row, ass_quality_report)
        if not matched_types:
            continue
        target_start = int(row.get("target_start_ms", row.get("ass_start_ms", 0)) or 0)
        target_end = int(row.get("target_end_ms", row.get("ass_end_ms", target_start)) or target_start)
        matches = _quality_row_subtitle_matches(updated, subtitle_index, row, target_start, target_end)
        for match in matches:
            candidate = {
                "ass_index": row.get("index"),
                "ass_text": row.get("ass_text", ""),
                "diagnostics": matched_types,
                "target_start_ms": target_start,
                "target_end_ms": target_end,
                "matched_subtitle_id": match["subtitle_id"],
                "distance_ms": match["distance_ms"],
                "route": match.get("route", "primary"),
            }
            candidates.append(candidate)
            if not match["subtitle_id"] or match["distance_ms"] > max_distance_ms:
                continue
            subtitle_id = str(match["subtitle_id"])
            item = updated.get(subtitle_id)
            if not isinstance(item, dict):
                continue
            _merge_suspect_metadata(
                item,
                suspect_types=matched_types,
                reason=_quality_reason(row, matched_types),
            )
            applied.append(candidate)

    return {
        "translated": updated,
        "report": {
            "source": "ass-quality",
            "candidate_count": len(candidates),
            "applied_count": len(applied),
            "max_distance_ms": max_distance_ms,
            "candidates": candidates,
            "applied": applied,
        },
    }


def apply_quality_diff_suspects_to_translated(
    translated: dict[str, Any],
    ass_quality_diff_report: dict[str, Any],
    *,
    max_distance_ms: int = 8000,
) -> dict[str, Any]:
    issues = ass_quality_diff_report.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    updated = {
        str(key): dict(value) if isinstance(value, dict) else value
        for key, value in translated.items()
    }
    candidates: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    subtitle_index = _translated_time_index(updated)

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        matched_types = _quality_diff_suspect_types(issue)
        if not matched_types:
            continue
        target_start = int(issue.get("target_start_ms", issue.get("ass_start_ms", 0)) or 0)
        target_end = int(issue.get("target_end_ms", issue.get("ass_end_ms", target_start)) or target_start)
        match = _nearest_subtitle(subtitle_index, target_start, target_end)
        candidate = {
            "ass_index": issue.get("index"),
            "ass_text": issue.get("ass_text", ""),
            "transition": issue.get("transition", ""),
            "issue_type": issue.get("type", ""),
            "diagnostics": matched_types,
            "target_start_ms": target_start,
            "target_end_ms": target_end,
            "matched_subtitle_id": match["subtitle_id"],
            "distance_ms": match["distance_ms"],
        }
        candidates.append(candidate)
        if not match["subtitle_id"] or match["distance_ms"] > max_distance_ms:
            continue
        subtitle_id = str(match["subtitle_id"])
        item = updated.get(subtitle_id)
        if not isinstance(item, dict):
            continue
        _merge_suspect_metadata(
            item,
            suspect_types=matched_types,
            reason=_quality_diff_reason(issue, matched_types),
        )
        applied.append(candidate)

    return {
        "translated": updated,
        "report": {
            "source": "ass-quality-diff",
            "candidate_count": len(candidates),
            "applied_count": len(applied),
            "max_distance_ms": max_distance_ms,
            "candidates": candidates,
            "applied": applied,
        },
    }


def _quality_diff_suspect_types(issue: dict[str, Any]) -> list[str]:
    issue_type = str(issue.get("type", "") or "")
    severity = str(issue.get("severity", "") or "")
    score_drop = _safe_float(issue.get("score_drop", 0.0))
    current_diagnostics = issue.get("current_diagnostics", [])
    if not isinstance(current_diagnostics, list):
        current_diagnostics = []
    mapped: list[str] = []
    if issue_type in {"became-fail", "became-low", "diagnostic-added"}:
        mapped.append(QUALITY_DIFF_SUSPECT_TYPES[issue_type])
    elif issue_type == "score-drop" and (severity == "FAIL" or score_drop >= QUALITY_DIFF_MIN_SCORE_DROP):
        mapped.append(QUALITY_DIFF_SUSPECT_TYPES[issue_type])
    for diagnostic in current_diagnostics:
        if diagnostic in QUALITY_SUSPECT_TYPES:
            mapped.append(QUALITY_SUSPECT_TYPES[str(diagnostic)])
    return list(dict.fromkeys(mapped))


def _quality_row_suspect_types(row: dict[str, Any], report: dict[str, Any]) -> list[str]:
    diagnostics = row.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        diagnostics = []
    mapped = [
        QUALITY_SUSPECT_TYPES[diagnostic]
        for diagnostic in diagnostics
        if diagnostic in QUALITY_SUSPECT_TYPES
    ]
    level = str(row.get("level", "") or "").upper()
    score = _safe_float(row.get("score", 1.0))
    thresholds = report.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}
    low_score_threshold = _safe_float(thresholds.get("low_score", 0.45)) or 0.45
    fail_score_threshold = _safe_float(thresholds.get("fail_score", 0.20)) or 0.20
    if level == "FAIL" or score < fail_score_threshold:
        mapped.append(QUALITY_FAIL_SCORE_TYPE)
    elif level == "LOW" or score < low_score_threshold:
        mapped.append(QUALITY_LOW_SCORE_TYPE)
    return list(dict.fromkeys(mapped))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _translated_time_index(translated: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subtitle_id in sorted((str(key) for key in translated if str(key).isdigit()), key=int):
        item = translated.get(subtitle_id)
        if not isinstance(item, dict):
            continue
        try:
            start_ms = int(item.get("start_time", 0) or 0)
            end_ms = int(item.get("end_time", start_ms) or start_ms)
        except (TypeError, ValueError):
            continue
        rows.append({"subtitle_id": subtitle_id, "start_ms": start_ms, "end_ms": max(start_ms + 1, end_ms)})
    return rows


def _quality_row_subtitle_match(
    translated: dict[str, Any],
    index: list[dict[str, Any]],
    row: dict[str, Any],
    target_start: int,
    target_end: int,
) -> dict[str, Any]:
    key = _preferred_quality_row_key(row)
    if key and isinstance(translated.get(key), dict):
        return {"subtitle_id": key, "distance_ms": 0}
    return _nearest_subtitle(index, target_start, target_end)


def _quality_row_subtitle_matches(
    translated: dict[str, Any],
    index: list[dict[str, Any]],
    row: dict[str, Any],
    target_start: int,
    target_end: int,
) -> list[dict[str, Any]]:
    primary = dict(_quality_row_subtitle_match(translated, index, row, target_start, target_end))
    primary["route"] = "primary"
    matches = [primary]
    extra_key = _short_missing_diagnostic_key(row)
    if extra_key and extra_key != primary["subtitle_id"] and isinstance(translated.get(extra_key), dict):
        matches.append({"subtitle_id": extra_key, "distance_ms": 0, "route": "diagnostic"})
    return matches


def _preferred_quality_row_key(row: dict[str, Any]) -> str:
    diagnostics = row.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        diagnostics = []
    diagnostic_key = str(row.get("diagnostic_matched_key", "") or "").strip()
    if diagnostic_key and "short-dialogue-timing-shifted" in diagnostics:
        return diagnostic_key
    return str(row.get("matched_key", "") or "").strip()


def _short_missing_diagnostic_key(
    row: dict[str, Any],
    *,
    min_diagnostic_score: float = 0.50,
    min_score_gain: float = 0.15,
) -> str:
    diagnostics = row.get("diagnostics", [])
    if not isinstance(diagnostics, list) or "short-dialogue-missing" not in diagnostics:
        return ""
    diagnostic_key = str(row.get("diagnostic_matched_key", "") or "").strip()
    if not diagnostic_key:
        return ""
    try:
        score = float(row.get("score", 0.0) or 0.0)
        diagnostic_score = float(row.get("diagnostic_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return ""
    if diagnostic_score >= min_diagnostic_score and diagnostic_score >= score + min_score_gain:
        return diagnostic_key
    return ""


def _nearest_subtitle(index: list[dict[str, Any]], target_start: int, target_end: int) -> dict[str, Any]:
    best = {"subtitle_id": "", "distance_ms": 10**12}
    for item in index:
        distance = _time_distance_ms(
            int(item["start_ms"]),
            int(item["end_ms"]),
            target_start,
            target_end,
        )
        if distance < int(best["distance_ms"]):
            best = {"subtitle_id": str(item["subtitle_id"]), "distance_ms": distance}
    return best


def _time_distance_ms(start_ms: int, end_ms: int, target_start: int, target_end: int) -> int:
    if min(end_ms, target_end) > max(start_ms, target_start):
        return 0
    if end_ms <= target_start:
        return target_start - end_ms
    return start_ms - target_end


def _merge_suspect_metadata(item: dict[str, Any], *, suspect_types: list[str], reason: str) -> None:
    existing_types = item.get("suspect_types", [])
    if not isinstance(existing_types, list):
        existing_types = []
    merged_types = list(dict.fromkeys([str(value) for value in existing_types + suspect_types if str(value).strip()]))
    existing_reason = str(item.get("suspect_reason", "")).strip()
    needs_audio_review = _suspect_types_need_audio_review(merged_types)
    if needs_audio_review:
        item["asr_suspect"] = True
        item["needs_audio_review"] = True
        item["suspect_confidence"] = min(float(item.get("suspect_confidence", 1.0) or 1.0), 0.25)
    else:
        item["needs_realign"] = True
        item["realign_status"] = "pending"
        item.setdefault("asr_suspect", False)
        item.setdefault("needs_audio_review", False)
    item["suspect_types"] = merged_types
    item["suspect_reason"] = "; ".join(part for part in [existing_reason, reason] if part)


def _suspect_types_need_audio_review(suspect_types: list[str]) -> bool:
    clean = {str(value).strip() for value in suspect_types if str(value).strip()}
    if "ass_short_dialogue_timing_shifted" in clean and clean.issubset(REALIGN_ONLY_SUSPECT_TYPES):
        return False
    return bool(clean)


def _quality_reason(row: dict[str, Any], suspect_types: list[str]) -> str:
    ass_text = str(row.get("ass_text", "")).strip()
    return (
        "ASS quality gate suspect: "
        f"{','.join(suspect_types)}"
        f" ass_index={row.get('index')} ass_text={ass_text}"
    )


def _quality_diff_reason(issue: dict[str, Any], suspect_types: list[str]) -> str:
    ass_text = str(issue.get("ass_text", "")).strip()
    return (
        "ASS stage diff suspect: "
        f"{','.join(suspect_types)}"
        f" transition={issue.get('transition')}"
        f" issue={issue.get('type')}"
        f" ass_index={issue.get('index')}"
        f" ass_text={ass_text}"
    )
