from __future__ import annotations

from typing import Any


def collect_stage1_suspects(stage1_report: list[dict[str, Any]]) -> list[str]:
    suspect_ids: set[str] = set()
    for item in stage1_report:
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        for subtitle_id in item.get("suspect_ids", []):
            subtitle_id_text = str(subtitle_id).strip()
            if subtitle_id_text:
                suspect_ids.add(subtitle_id_text)
    return sorted(suspect_ids, key=int)


def translated_manifest_has_suspect_metadata(translated: dict[str, Any]) -> bool:
    for value in translated.values():
        if not isinstance(value, dict):
            continue
        if any(key in value for key in ("asr_suspect", "needs_audio_review", "suspect_types", "suspect_confidence")):
            return True
    return False


def suspect_types_need_audio_review(suspect_types: Any) -> bool:
    if not isinstance(suspect_types, list):
        return False
    clean = {str(value).strip() for value in suspect_types if str(value).strip()}
    if "ass_short_dialogue_timing_shifted" in clean and clean.issubset(
        {"ass_short_dialogue_timing_shifted", "ass_low_score", "ass_fail_score"}
    ):
        return False
    return bool(clean)


def build_manifest_suspect_report(
    translated: dict[str, Any],
    *,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    suspect_ids: list[str] = []
    suggestions: list[dict[str, Any]] = []
    for subtitle_id in sorted((str(key) for key in translated if str(key).isdigit()), key=int):
        item = translated.get(subtitle_id)
        if not isinstance(item, dict):
            continue
        confidence = coerce_confidence(item.get("suspect_confidence"), default=1.0)
        asr_suspect = coerce_bool(item.get("asr_suspect"))
        needs_audio_review = coerce_bool(item.get("needs_audio_review"))
        suspect_types = item.get("suspect_types", [])
        has_audio_review_types = suspect_types_need_audio_review(suspect_types)
        is_suspect = asr_suspect or needs_audio_review or has_audio_review_types or confidence < confidence_threshold
        if is_suspect:
            suspect_ids.append(subtitle_id)
        suggestions.append(
            {
                "id": subtitle_id,
                "error_type": "asr_suspect" if asr_suspect else ("needs_context" if needs_audio_review else ""),
                "original": str(item.get("original_subtitle", "")).strip(),
                "translation": str(item.get("translated_subtitle", "")).strip(),
                "suggested_translation": "",
                "asr_suspect": asr_suspect,
                "suggested_original": "",
                "needs_audio_review": needs_audio_review,
                "reason": str(item.get("suspect_reason", "")).strip(),
                "confidence": confidence,
                "suspect_types": suspect_types if isinstance(suspect_types, list) else [],
            }
        )
    return [
        {
            "segment_id": "translation-manifest",
            "status": "completed",
            "subtitle_ids": [item["id"] for item in suggestions],
            "suggestion_count": len(suggestions),
            "applied_count": 0,
            "suspect_ids": suspect_ids,
            "suspect_count": len(suspect_ids),
            "glossary_count": 0,
            "usage": {},
            "elapsed_ms": 0,
            "suggestions": suggestions,
            "source": "translation-manifest",
        }
    ]


def translated_duration_ms(translated: dict[str, Any]) -> int:
    starts: list[int] = []
    ends: list[int] = []
    for value in translated.values():
        if not isinstance(value, dict):
            continue
        try:
            starts.append(int(value.get("start_time", 0)))
            ends.append(int(value.get("end_time", 0)))
        except (TypeError, ValueError):
            continue
    if not starts or not ends:
        return 0
    return max(0, max(ends) - min(starts))


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1"}


def coerce_confidence(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))
