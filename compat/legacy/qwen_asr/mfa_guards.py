from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from typing import Any


def local_mfa_ass_guard(
    candidate: dict[str, Any],
    lab_text: str,
    global_word_ranges: list[dict[str, Any]],
    word_quality: dict[str, Any],
) -> dict[str, Any]:
    candidate_start = int_or_none(candidate.get("start_ms"))
    candidate_end = int_or_none(candidate.get("end_ms"))
    mfa_text = "".join(str(item.get("text", "")) for item in global_word_ranges)
    mfa_start = min((int(item["start_ms"]) for item in global_word_ranges), default=None)
    mfa_end = max((int(item["end_ms"]) for item in global_word_ranges), default=None)
    text_score = local_ass_match_score(str(candidate.get("text", "")), mfa_text or lab_text)
    overlaps = (
        candidate_start is not None
        and candidate_end is not None
        and mfa_start is not None
        and mfa_end is not None
        and min(candidate_end, mfa_end) > max(candidate_start, mfa_start)
    )
    distance_ms = (
        range_distance_ms(candidate_start, candidate_end, mfa_start, mfa_end)
        if candidate_start is not None
        and candidate_end is not None
        and mfa_start is not None
        and mfa_end is not None
        else None
    )
    unknown_count = int(word_quality.get("unknown_count", 0) or 0)
    usable = bool(word_quality.get("usable"))
    passed = usable and unknown_count == 0 and text_score >= 0.45 and overlaps
    reasons: list[str] = []
    if not usable:
        reasons.append("mfa-output-unusable")
    if unknown_count:
        reasons.append("mfa-unknown-word")
    if text_score < 0.45:
        reasons.append("local-text-score-low")
    if not overlaps:
        reasons.append("mfa-time-outside-candidate-window")
    return {
        "status": "PASS" if passed else "FAIL",
        "reference_text": str(candidate.get("text", "")),
        "lab_text": lab_text,
        "mfa_text": mfa_text,
        "text_score": round(text_score, 6),
        "candidate_start_ms": candidate_start,
        "candidate_end_ms": candidate_end,
        "mfa_start_ms": mfa_start,
        "mfa_end_ms": mfa_end,
        "time_overlaps_candidate": overlaps,
        "time_distance_ms": distance_ms,
        "unknown_count": unknown_count,
        "reasons": reasons,
    }


def mfa_writeback_dry_run(
    candidate: dict[str, Any],
    local_guard: dict[str, Any],
) -> dict[str, Any]:
    details = candidate.get("details", {}) if isinstance(candidate.get("details"), dict) else {}
    previous_score = float_or_none(details.get("previous_score"))
    current_score_value = (
        details.get("current_score") if "current_score" in details else details.get("score")
    )
    current_score = float_or_none(current_score_value)
    guard_score = float_or_none(local_guard.get("text_score"))
    status = "SKIP"
    reasons: list[str] = []
    if local_guard.get("status") != "PASS":
        reasons.append("local-guard-not-pass")
    if previous_score is None or current_score is None:
        reasons.append("missing-baseline-score")
    if guard_score is None:
        reasons.append("missing-guard-score")
    if not reasons:
        assert previous_score is not None
        assert current_score is not None
        assert guard_score is not None
        if guard_score + 1e-9 < current_score:
            reasons.append("would-lower-current-score")
        if previous_score >= 0.45 and guard_score < 0.45:
            reasons.append("would-drop-below-low-threshold")
        status = "PASS" if not reasons else "FAIL"
    return {
        "status": status,
        "previous_score": previous_score,
        "current_score": current_score,
        "mfa_text_score": guard_score,
        "score_delta_vs_current": (
            round(guard_score - current_score, 6)
            if guard_score is not None and current_score is not None
            else None
        ),
        "score_delta_vs_previous": (
            round(guard_score - previous_score, 6)
            if guard_score is not None and previous_score is not None
            else None
        ),
        "candidate_start_ms": local_guard.get("candidate_start_ms"),
        "candidate_end_ms": local_guard.get("candidate_end_ms"),
        "mfa_start_ms": local_guard.get("mfa_start_ms"),
        "mfa_end_ms": local_guard.get("mfa_end_ms"),
        "reasons": reasons,
    }


def local_ass_match_score(reference: str, candidate: str) -> float:
    normalized_reference = normalize_local_match_text(reference)
    normalized_candidate = normalize_local_match_text(candidate)
    if not normalized_reference or not normalized_candidate:
        return 0.0
    if len(normalized_candidate) < max(1, round(len(normalized_reference) * 0.7)):
        return SequenceMatcher(
            None,
            normalized_reference,
            normalized_candidate,
            autojunk=False,
        ).ratio()
    if (
        len(normalized_reference) <= 4
        and len(normalized_candidate) > len(normalized_reference) + 6
    ):
        return SequenceMatcher(
            None,
            normalized_reference,
            normalized_candidate,
            autojunk=False,
        ).ratio()
    return local_partial_ratio(normalized_reference, normalized_candidate)


def local_partial_ratio(reference: str, candidate: str) -> float:
    if not reference or not candidate:
        return 0.0
    if len(reference) > len(candidate):
        reference, candidate = candidate, reference
    if len(candidate) <= len(reference) * 1.3:
        return SequenceMatcher(None, reference, candidate, autojunk=False).ratio()
    window_min = max(1, round(len(reference) * 0.7))
    window_max = min(len(candidate), round(len(reference) * 1.3))
    best = 0.0
    for size in range(window_min, window_max + 1):
        for start in range(0, len(candidate) - size + 1):
            score = SequenceMatcher(
                None,
                reference,
                candidate[start : start + size],
                autojunk=False,
            ).ratio()
            if score > best:
                best = score
    return best


def normalize_local_match_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(
        character for character in value if character.isalnum() or is_japanese_character(character)
    )


def is_japanese_character(character: str) -> bool:
    return (
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
    )


def range_distance_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a >= start_b and end_b >= start_a:
        return 0
    if end_a < start_b:
        return start_b - end_a
    return start_a - end_b


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
