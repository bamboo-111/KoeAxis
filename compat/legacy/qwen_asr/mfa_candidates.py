from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def collect_alignment_experiment_candidates(
    work_paths: WorkPaths,
    *,
    ass_quality_report_paths: list[Path],
    ass_quality_diff_report_paths: list[Path],
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    candidates.extend(candidates_from_content_quality(work_paths.content_quality_report))
    candidates.extend(
        candidates_from_proofread_realign(
            work_paths.workdir / "reports" / "proofread_realign.json"
        )
    )
    for path in ass_quality_report_paths:
        candidates.extend(candidates_from_ass_quality(path))
    for path in ass_quality_diff_report_paths:
        candidates.extend(candidates_from_ass_quality_diff(path))
    candidates.extend(candidates_from_mimo_manifest(work_paths.mimo_proofread_manifest))
    return dedupe_and_rank_candidates(candidates)[: max(0, max_candidates)]


def candidates_from_content_quality(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    for issue in payload.get("issues", []):
        if not isinstance(issue, dict):
            continue
        kind = str(issue.get("type", ""))
        if kind not in {
            "missing_short_response",
            "short_response_timing_shifted",
            "missing_unique_text",
            "alignment_fallback_too_short",
        }:
            continue
        result.append(
            candidate(
                source="content-quality",
                reason=kind,
                severity=str(issue.get("severity", "WARN")),
                start_ms=issue.get("start_ms"),
                end_ms=issue.get("end_ms"),
                text=str(issue.get("text", "")),
                details=issue,
            )
        )
    return result


def candidates_from_proofread_realign(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "")) not in {"fallback", "failed"}:
            continue
        result.append(
            candidate(
                source="proofread-realign",
                reason=str(item.get("status", "")),
                severity="FAIL" if item.get("status") == "failed" else "WARN",
                subtitle_id=str(item.get("id", "")),
                start_ms=item.get("before_start_time"),
                end_ms=item.get("before_end_time"),
                text="",
                details=item,
            )
        )
    return result


def candidates_from_ass_quality(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        if not isinstance(row, dict):
            continue
        diagnostics = [str(value) for value in row.get("diagnostics", []) if str(value)]
        if not any(value.startswith("short-dialogue") for value in diagnostics):
            continue
        result.append(
            candidate(
                source=f"ass-quality:{payload.get('source', '')}",
                reason=",".join(diagnostics),
                severity="FAIL" if "short-dialogue-missing" in diagnostics else "WARN",
                start_ms=row.get("ass_start_ms"),
                end_ms=row.get("ass_end_ms"),
                text=str(row.get("ass_text", "")),
                details={
                    "ass_index": row.get("ass_index"),
                    "score": row.get("score"),
                    "current_score": row.get("score"),
                    "diagnostics": diagnostics,
                },
            )
        )
    return result


def candidates_from_ass_quality_diff(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    for issue in payload.get("issues", []):
        if not isinstance(issue, dict):
            continue
        issue_type = str(issue.get("type", ""))
        current_diagnostics = [
            str(value) for value in issue.get("current_diagnostics", []) if str(value)
        ]
        if issue_type not in {
            "became-fail",
            "became-low",
            "score-drop",
            "matched-text-shortened",
        } and not any(value.startswith("short-dialogue") for value in current_diagnostics):
            continue
        result.append(
            candidate(
                source="ass-quality-diff",
                reason=issue_type,
                severity=str(issue.get("severity", "WARN")),
                start_ms=issue.get("ass_start_ms"),
                end_ms=issue.get("ass_end_ms"),
                text=str(issue.get("ass_text", "")),
                details={
                    "ass_index": issue.get("ass_index"),
                    "transition": issue.get("transition"),
                    "target_start_ms": issue.get("target_start_ms"),
                    "target_end_ms": issue.get("target_end_ms"),
                    "previous_score": issue.get("previous_score"),
                    "current_score": issue.get("current_score"),
                    "score_drop": issue.get("score_drop"),
                    "current_diagnostics": current_diagnostics,
                },
            )
        )
    return result


def candidates_from_mimo_manifest(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    result: list[dict[str, Any]] = []
    for subtitle_id, item in payload.items():
        if not isinstance(item, dict):
            continue
        history = item.get("proofread_history", [])
        has_mimo_change = isinstance(history, list) and any(
            isinstance(entry, dict) and str(entry.get("source", "")).startswith("mimo-")
            for entry in history
        )
        if not has_mimo_change and not item.get("needs_realign"):
            continue
        result.append(
            candidate(
                source="mimo-proofread",
                reason="mimo-change-needs-alignment-check",
                severity="WARN",
                subtitle_id=str(subtitle_id),
                start_ms=item.get("start_time"),
                end_ms=item.get("end_time"),
                text=str(item.get("original_subtitle", "")),
                details={
                    "needs_realign": item.get("needs_realign"),
                    "realign_status": item.get("realign_status"),
                    "realign_method": item.get("realign_method"),
                },
            )
        )
    return result


def candidate(
    *,
    source: str,
    reason: str,
    severity: str,
    start_ms: Any = None,
    end_ms: Any = None,
    text: str = "",
    subtitle_id: str = "",
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source": source,
        "reason": reason,
        "severity": severity if severity in {"FAIL", "WARN"} else "WARN",
        "subtitle_id": subtitle_id,
        "start_ms": _int_or_none(start_ms),
        "end_ms": _int_or_none(end_ms),
        "text": text,
        "details": details or {},
    }


def dedupe_and_rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for item in candidates:
        key = (
            item.get("subtitle_id") or "",
            item.get("start_ms"),
            item.get("end_ms"),
            item.get("text") or "",
            item.get("reason") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return sorted(unique, key=candidate_sort_key)


def candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, str]:
    severity_rank = 0 if item.get("severity") == "FAIL" else 1
    source_rank = {
        "proofread-realign": 0,
        "content-quality": 1,
        "ass-quality-diff": 2,
        "mimo-proofread": 3,
    }.get(str(item.get("source", "")).split(":")[0], 4)
    start_ms = item.get("start_ms")
    return (
        severity_rank,
        source_rank,
        int(start_ms) if isinstance(start_ms, int) else 10**12,
        str(item.get("subtitle_id", "")),
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
