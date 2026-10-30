from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from qwen_asr.storage import read_json

ALIGN_STATES = ("completed_exact", "completed_coarse", "failed")
MUSIC_REGION_STATE = "SKIPPED_MUSIC_REGION"


def derive_alignment_state(item: Any) -> str:
    explicit = str(_value(item, "alignment_state", _value(item, "align_state", ""))).strip()
    if explicit in ALIGN_STATES:
        return explicit
    if str(_value(item, "status", "completed")) != "completed":
        return "failed"
    if str(_value(item, "alignment_unit", "token")) != "token":
        return "completed_coarse"
    tokens = _value(item, "tokens", [])
    if not isinstance(tokens, list) or not tokens:
        return "completed_coarse"
    if not any(_positive_token(token) for token in tokens):
        return "completed_coarse"
    return "completed_exact"


def compatibility_status(alignment_state: str) -> str:
    return "failed" if alignment_state == "failed" else "completed"


def overlaps_music_region(item: Any, intervals: list[dict[str, Any]]) -> dict[str, Any] | None:
    start_ms = seconds_to_ms(_value(item, "global_start_time", None))
    end_ms = seconds_to_ms(_value(item, "global_end_time", None))
    if start_ms is None or end_ms is None:
        return None
    for interval in intervals:
        if start_ms < interval["end_ms"] and end_ms > interval["start_ms"]:
            return interval
    return None


def read_music_region_evidence(
    workdir: Path,
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any], str | None]:
    reports_dir = workdir / "reports"
    if not reports_dir.is_dir():
        return [], None, {}, None
    errors = []
    for path in sorted(reports_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = read_json(path, default={})
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(payload, dict):
            continue
        raw_intervals = payload.get("intervals")
        if not isinstance(raw_intervals, dict):
            continue
        intervals = []
        for name, value in raw_intervals.items():
            if not isinstance(value, dict):
                continue
            start_ms = int_or_none(value.get("start_ms"))
            end_ms = int_or_none(value.get("end_ms"))
            if start_ms is None or end_ms is None or end_ms <= start_ms:
                continue
            intervals.append({"name": str(name), "start_ms": start_ms, "end_ms": end_ms})
        if intervals:
            summary = {
                "alignment": payload.get("alignment", {}),
                "subtitle_cues": payload.get("subtitle_cues", {}),
                "postrepair_on_main_dialogue_failures": payload.get(
                    "postrepair_on_main_dialogue_failures", {}
                ),
            }
            return intervals, str(path), summary, None
    return [], None, {}, "; ".join(errors) if errors else None


def seconds_to_ms(value: Any) -> int | None:
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError):
        return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_token(token: Any) -> bool:
    try:
        return float(_value(token, "end_time", 0)) > float(_value(token, "start_time", 0))
    except (TypeError, ValueError):
        return False


def _value(item: Any, key: str, default: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)
