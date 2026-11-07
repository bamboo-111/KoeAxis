from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr.storage import read_json


def read_mfa_words(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return []
    tiers = payload.get("tiers", {})
    if not isinstance(tiers, dict):
        return []
    words_tier = tiers.get("words", {})
    if not isinstance(words_tier, dict):
        return []
    entries = words_tier.get("entries", [])
    result: list[dict[str, Any]] = []
    if not isinstance(entries, list):
        return result
    for entry in entries:
        if not isinstance(entry, list) or len(entry) < 3:
            continue
        try:
            start_ms = int(round(float(entry[0]) * 1000))
            end_ms = int(round(float(entry[1]) * 1000))
        except (TypeError, ValueError):
            continue
        result.append({"start_ms": start_ms, "end_ms": end_ms, "text": str(entry[2])})
    return result


def globalize_mfa_words(
    words: list[dict[str, Any]],
    *,
    clip_start_ms: int,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for word in words:
        if not isinstance(word.get("start_ms"), int) or not isinstance(
            word.get("end_ms"),
            int,
        ):
            continue
        result.append(
            {
                "start_ms": clip_start_ms + int(word["start_ms"]),
                "end_ms": clip_start_ms + int(word["end_ms"]),
                "text": word.get("text", ""),
            }
        )
    return result


def evaluate_mfa_words(words: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_count = sum(
        1
        for word in words
        if str(word.get("text", "")).strip().lower() in {"<unk>", "unk"}
    )
    timed_count = sum(
        1
        for word in words
        if isinstance(word.get("start_ms"), int)
        and isinstance(word.get("end_ms"), int)
        and int(word["end_ms"]) > int(word["start_ms"])
    )
    known_timed_count = sum(
        1
        for word in words
        if str(word.get("text", "")).strip().lower() not in {"<unk>", "unk"}
        and isinstance(word.get("start_ms"), int)
        and isinstance(word.get("end_ms"), int)
        and int(word["end_ms"]) > int(word["start_ms"])
    )
    return {
        "word_count": len(words),
        "timed_count": timed_count,
        "unknown_count": unknown_count,
        "known_timed_count": known_timed_count,
        "usable": known_timed_count > 0,
    }
