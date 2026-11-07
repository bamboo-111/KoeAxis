from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr.storage import read_json


def load_existing_branch(path: Path, translated: dict[str, Any]) -> dict[str, Any]:
    existing = read_json(path, default=None)
    if isinstance(existing, dict) and existing:
        return {key: dict(value) if isinstance(value, dict) else value for key, value in existing.items()}
    return {key: dict(value) if isinstance(value, dict) else value for key, value in translated.items()}


def load_existing_report(path: Path) -> list[dict[str, Any]]:
    existing = read_json(path, default=[])
    if isinstance(existing, list):
        return [item for item in existing if isinstance(item, dict)]
    return []


def completed_segment_ids(report: list[dict[str, Any]], *, key: str = "segment_id") -> set[str]:
    return {
        str(item.get(key))
        for item in report
        if isinstance(item, dict) and item.get("status") == "completed"
    }


def pending_review_ids(
    review_ids: list[str],
    stage2_report: list[dict[str, Any]],
    *,
    resume: bool,
) -> list[str]:
    completed_stage2 = completed_segment_ids(stage2_report, key="id")
    return [
        subtitle_id for subtitle_id in review_ids
        if not (resume and subtitle_id in completed_stage2)
    ]
