from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_asr.mfa_guards import int_or_none, local_ass_match_score, range_distance_ms
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


def apply_mfa_local_writeback(
    work_paths: WorkPaths,
    local_runs: list[dict[str, Any]],
    *,
    mode: str = "off",
    output_path: Path | None = None,
) -> dict[str, Any]:
    mode = mode if mode in {"off", "propose", "apply"} else "off"
    if mode == "off":
        return {"enabled": False, "mode": mode, "status": "SKIP", "items": []}
    if not work_paths.split_manifest.exists():
        return {
            "enabled": True,
            "mode": mode,
            "status": "SKIP",
            "reason": "split-manifest-missing",
            "source_manifest": str(work_paths.split_manifest),
            "items": [],
        }
    manifest = read_json(work_paths.split_manifest, default={})
    if not isinstance(manifest, dict):
        return {
            "enabled": True,
            "mode": mode,
            "status": "SKIP",
            "reason": "split-manifest-invalid",
            "source_manifest": str(work_paths.split_manifest),
            "items": [],
        }
    mutable_manifest = json.loads(json.dumps(manifest, ensure_ascii=False))
    items: list[dict[str, Any]] = []
    applied_count = 0
    for run in local_runs:
        if not isinstance(run, dict):
            continue
        decision = build_mfa_writeback_decision(mutable_manifest, run)
        if mode == "apply" and decision.get("status") == "APPLY":
            item_id = str(decision.get("subtitle_id", ""))
            row = mutable_manifest.get(item_id)
            if isinstance(row, dict):
                row["start_time"] = decision["new_start_ms"]
                row["end_time"] = decision["new_end_ms"]
                row["mfa_local_writeback"] = {
                    "source": "mfa-align-experiment",
                    "previous_start_ms": decision.get("old_start_ms"),
                    "previous_end_ms": decision.get("old_end_ms"),
                    "mfa_text": decision.get("mfa_text", ""),
                    "text_score": decision.get("manifest_text_score"),
                }
                decision["status"] = "APPLIED"
                applied_count += 1
        items.append(decision)
    output = output_path or work_paths.workdir / "experiments" / "mfa-align-experiment" / "split_segments.mfa-writeback.json"
    if mode == "apply" and applied_count:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, mutable_manifest)
    return {
        "enabled": True,
        "mode": mode,
        "status": "APPLIED" if applied_count else "NOOP",
        "source_manifest": str(work_paths.split_manifest),
        "output_manifest": str(output) if mode == "apply" and applied_count else "",
        "candidate_count": len(items),
        "applied_count": applied_count,
        "rejected_count": sum(1 for item in items if item.get("status") == "REJECT"),
        "proposed_count": sum(1 for item in items if item.get("status") == "APPLY"),
        "items": items,
    }


def build_mfa_writeback_decision(manifest: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    candidate = run.get("candidate", {}) if isinstance(run.get("candidate"), dict) else {}
    guard = run.get("local_ass_guard", {}) if isinstance(run.get("local_ass_guard"), dict) else {}
    dry_run = run.get("writeback_dry_run", {}) if isinstance(run.get("writeback_dry_run"), dict) else {}
    reasons: list[str] = []
    if run.get("status") != "completed":
        reasons.append("local-run-not-completed")
    if guard.get("status") != "PASS":
        reasons.append("local-guard-not-pass")
    if dry_run.get("status") != "PASS":
        reasons.append("writeback-dry-run-not-pass")
    new_start = int_or_none(guard.get("mfa_start_ms"))
    new_end = int_or_none(guard.get("mfa_end_ms"))
    if new_start is None or new_end is None or new_end <= new_start:
        reasons.append("invalid-mfa-time")
    target = find_writeback_manifest_target(manifest, candidate)
    if target is None:
        reasons.append("manifest-target-not-found")
        target_id = ""
        target_row: dict[str, Any] = {}
    else:
        target_id, target_row = target
    mfa_text = str(guard.get("mfa_text") or run.get("lab_text") or "")
    manifest_text = str(target_row.get("original_subtitle", "")) if target_row else ""
    manifest_text_score = local_ass_match_score(manifest_text, mfa_text) if manifest_text and mfa_text else 0.0
    if target_row and manifest_text_score < 0.7:
        reasons.append("manifest-text-mismatch")
    old_start = int_or_none(target_row.get("start_time")) if target_row else None
    old_end = int_or_none(target_row.get("end_time")) if target_row else None
    if target_row and new_start is not None and new_end is not None and old_start is not None and old_end is not None:
        if range_distance_ms(old_start, old_end, new_start, new_end) > 2500:
            reasons.append("mfa-time-too-far-from-manifest-target")
    return {
        "status": "REJECT" if reasons else "APPLY",
        "reasons": reasons,
        "subtitle_id": target_id,
        "manifest_text": manifest_text,
        "mfa_text": mfa_text,
        "manifest_text_score": round(manifest_text_score, 6),
        "old_start_ms": old_start,
        "old_end_ms": old_end,
        "new_start_ms": new_start,
        "new_end_ms": new_end,
        "candidate": candidate,
    }


def find_writeback_manifest_target(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    details = candidate.get("details", {}) if isinstance(candidate.get("details"), dict) else {}
    target_start = int_or_none(details.get("target_start_ms"))
    target_end = int_or_none(details.get("target_end_ms"))
    if target_start is None or target_end is None:
        target_start = int_or_none(candidate.get("start_ms"))
        target_end = int_or_none(candidate.get("end_ms"))
    if target_start is None or target_end is None:
        return None
    best: tuple[int, str, dict[str, Any]] | None = None
    for subtitle_id, row in manifest.items():
        if not isinstance(row, dict):
            continue
        row_start = int_or_none(row.get("start_time"))
        row_end = int_or_none(row.get("end_time"))
        if row_start is None or row_end is None:
            continue
        distance = range_distance_ms(target_start, target_end, row_start, row_end)
        if distance > 2500:
            continue
        midpoint_delta = abs(((target_start + target_end) // 2) - ((row_start + row_end) // 2))
        score = distance * 10 + midpoint_delta
        if best is None or score < best[0]:
            best = (score, str(subtitle_id), row)
    if best is None:
        return None
    return best[1], best[2]
