from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def fallback_original_timing(
    item: dict[str, Any],
    subtitle_id: str,
    start_ms: int,
    end_ms: int,
    clip_path: Path,
    error: str,
    mfa_row: dict[str, Any] | None = None,
    *,
    method: str = "original-timing",
) -> dict[str, Any]:
    item["start_time"] = start_ms
    item["end_time"] = end_ms
    item["needs_realign"] = False
    item["realign_status"] = "completed"
    item["realign_source"] = "proofread-realign-fallback"
    item["realign_method"] = method
    item["realign_clip_path"] = str(clip_path)
    item["realign_warning"] = error
    if mfa_row:
        item["realign_mfa_status"] = mfa_row.get("mfa_status", mfa_row.get("status"))
        item["realign_mfa_reason"] = mfa_row.get("reason")
        item["realign_mfa_result"] = mfa_row.get("mfa_result", mfa_row)
    item.pop("realign_error", None)
    row = {
        "id": subtitle_id,
        "status": "fallback",
        "method": method,
        "warning": error,
        "before_start_time": start_ms,
        "before_end_time": end_ms,
        "after_start_time": start_ms,
        "after_end_time": end_ms,
        "token_count": 0,
        "clip_path": str(clip_path),
    }
    if mfa_row:
        row["mfa_status"] = mfa_row.get("mfa_status", mfa_row.get("status"))
        row["mfa_reason"] = mfa_row.get("reason")
        row["mfa_result"] = mfa_row.get("mfa_result", mfa_row)
    return row


def should_keep_mixed_language_original_timing(item: dict[str, Any]) -> bool:
    text = str(item.get("original_subtitle", "") or "").strip()
    if len(text) < 40:
        return False
    start_ms = max(0, int(item.get("start_time", 0) or 0))
    end_ms = max(start_ms, int(item.get("end_time", start_ms) or start_ms))
    if end_ms - start_ms < 4000:
        return False
    latin_count = sum(character.isascii() and character.isalpha() for character in text)
    japanese_count = sum(is_japanese_character(character) for character in text)
    return latin_count >= 20 and japanese_count >= 8


def timing_candidate_guard(
    manifest: dict[str, Any],
    *,
    subtitle_id: str,
    start_ms: int,
    end_ms: int,
    clip_start_ms: int,
    clip_end_ms: int,
    max_overlap_ms: int = 120,
) -> dict[str, Any]:
    if end_ms <= start_ms:
        return {"accepted": False, "reason": "non-positive-duration"}
    if start_ms < clip_start_ms or end_ms > clip_end_ms:
        return {
            "accepted": False,
            "reason": "outside-realign-clip",
            "clip_start_ms": clip_start_ms,
            "clip_end_ms": clip_end_ms,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
    previous_end: int | None = None
    next_start: int | None = None
    current_key = sort_key(subtitle_id)
    for key, value in manifest.items():
        if str(key) == str(subtitle_id) or not isinstance(value, dict):
            continue
        row_sort_key = sort_key(str(key))
        try:
            row_start = int(value.get("start_time", 0) or 0)
            row_end = int(value.get("end_time", row_start) or row_start)
        except (TypeError, ValueError):
            continue
        if row_end <= row_start:
            continue
        if row_sort_key < current_key:
            previous_end = row_end if previous_end is None else max(previous_end, row_end)
        elif row_sort_key > current_key:
            next_start = row_start if next_start is None else min(next_start, row_start)
    previous_overlap = max(0, (previous_end or start_ms) - start_ms)
    next_overlap = max(0, end_ms - (next_start or end_ms))
    if previous_overlap > max_overlap_ms or next_overlap > max_overlap_ms:
        return {
            "accepted": False,
            "reason": "severe-neighbor-overlap",
            "previous_end_ms": previous_end,
            "next_start_ms": next_start,
            "previous_overlap_ms": previous_overlap,
            "next_overlap_ms": next_overlap,
            "max_overlap_ms": max_overlap_ms,
            "start_ms": start_ms,
            "end_ms": end_ms,
        }
    return {
        "accepted": True,
        "reason": "timing-ok",
        "previous_end_ms": previous_end,
        "next_start_ms": next_start,
        "previous_overlap_ms": previous_overlap,
        "next_overlap_ms": next_overlap,
        "start_ms": start_ms,
        "end_ms": end_ms,
    }


def clamp_timing_candidate_to_neighbors(
    manifest: dict[str, Any],
    *,
    subtitle_id: str,
    start_ms: int,
    end_ms: int,
    clip_start_ms: int,
    clip_end_ms: int,
    timing_guard: dict[str, Any],
    min_duration_ms: int = 500,
    max_display_pad_ms: int = 120,
) -> dict[str, Any]:
    if timing_guard.get("reason") != "severe-neighbor-overlap":
        return {"accepted": False, "reason": "not-clampable", "timing_guard": timing_guard}
    clamped_start = start_ms
    clamped_end = end_ms
    previous_end = timing_guard.get("previous_end_ms")
    next_start = timing_guard.get("next_start_ms")
    if isinstance(previous_end, int):
        clamped_start = max(clamped_start, previous_end)
    if isinstance(next_start, int):
        clamped_end = min(clamped_end, next_start)
    if clamped_end - clamped_start < min_duration_ms:
        expanded = expand_display_range_to_min_duration(
            clamped_start,
            clamped_end,
            min_duration_ms=min_duration_ms,
            max_display_pad_ms=max_display_pad_ms,
            lower_bound_ms=previous_end if isinstance(previous_end, int) else clip_start_ms,
            upper_bound_ms=next_start if isinstance(next_start, int) else clip_end_ms,
        )
        if expanded is None:
            return {
                "accepted": False,
                "reason": "clamped-duration-too-short",
                "start_ms": clamped_start,
                "end_ms": clamped_end,
                "min_duration_ms": min_duration_ms,
                "max_display_pad_ms": max_display_pad_ms,
                "timing_guard": timing_guard,
            }
        clamped_start, clamped_end = expanded
    clamped_guard = timing_candidate_guard(
        manifest,
        subtitle_id=subtitle_id,
        start_ms=clamped_start,
        end_ms=clamped_end,
        clip_start_ms=clip_start_ms,
        clip_end_ms=clip_end_ms,
    )
    if not clamped_guard["accepted"]:
        clamped_guard["reason"] = f"clamp-{clamped_guard['reason']}"
        clamped_guard["original_timing_guard"] = timing_guard
        return clamped_guard
    clamped_guard["reason"] = "timing-clamped-to-neighbors"
    clamped_guard["original_start_ms"] = start_ms
    clamped_guard["original_end_ms"] = end_ms
    clamped_guard["start_ms"] = clamped_start
    clamped_guard["end_ms"] = clamped_end
    clamped_guard["original_timing_guard"] = timing_guard
    return clamped_guard


def expand_display_range_to_min_duration(
    start_ms: int,
    end_ms: int,
    *,
    min_duration_ms: int,
    max_display_pad_ms: int,
    lower_bound_ms: int,
    upper_bound_ms: int,
) -> tuple[int, int] | None:
    duration = end_ms - start_ms
    deficit = min_duration_ms - duration
    if deficit <= 0:
        return start_ms, end_ms
    if deficit > max_display_pad_ms:
        return None
    add_after = min(deficit, max(0, upper_bound_ms - end_ms))
    end_ms += add_after
    deficit -= add_after
    if deficit:
        add_before = min(deficit, max(0, start_ms - lower_bound_ms))
        start_ms -= add_before
        deficit -= add_before
    if deficit or end_ms - start_ms < min_duration_ms:
        return None
    return start_ms, end_ms


def clamp_display_range_to_original_window(
    start_ms: int,
    end_ms: int,
    *,
    original_start_ms: int,
    original_end_ms: int,
    max_duration_ms: int = 8000,
) -> dict[str, Any]:
    if end_ms - start_ms <= max_duration_ms:
        return {"accepted": False, "reason": "duration-ok"}
    original_duration = original_end_ms - original_start_ms
    if original_duration <= 0 or original_duration > max_duration_ms:
        return {
            "accepted": False,
            "reason": "original-window-not-usable",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "original_start_ms": original_start_ms,
            "original_end_ms": original_end_ms,
            "max_duration_ms": max_duration_ms,
        }
    if original_start_ms < start_ms or original_end_ms > end_ms:
        return {
            "accepted": False,
            "reason": "original-window-outside-token-range",
            "start_ms": start_ms,
            "end_ms": end_ms,
            "original_start_ms": original_start_ms,
            "original_end_ms": original_end_ms,
        }
    return {
        "accepted": True,
        "reason": "display-clamped-to-original-window",
        "warning": "qwen timing clamped to original display window: overlong-duration",
        "start_ms": original_start_ms,
        "end_ms": original_end_ms,
        "original_token_start_ms": start_ms,
        "original_token_end_ms": end_ms,
        "max_duration_ms": max_duration_ms,
    }


def mfa_content_score(reference: str, candidate: str) -> float:
    normalized_reference = normalize_mfa_content(reference)
    normalized_candidate = normalize_mfa_content(candidate)
    if not normalized_reference or not normalized_candidate:
        return 0.0
    return SequenceMatcher(None, normalized_reference, normalized_candidate, autojunk=False).ratio()


def normalize_mfa_content(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(character for character in value if character.isalnum() or is_japanese_character(character))


def is_japanese_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def sort_key(value: Any) -> tuple[int, str]:
    text = str(value)
    return (int(text), "") if text.isdigit() else (10**9, text)


def safe_id(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in str(value)) or "item"
