from __future__ import annotations

from typing import Any


def score_candidate_payload(
    *,
    dialogue: Any,
    source_text: str,
    source_start_ms: int,
    source_end_ms: int,
    matched_segment_count: int,
    covered_duration: int,
    normalize_text: Any,
) -> dict[str, Any]:
    time_overlap = interval_overlap_score(dialogue.start_ms, dialogue.end_ms, source_start_ms, source_end_ms)
    boundary = boundary_score(dialogue.start_ms, dialogue.end_ms, source_start_ms, source_end_ms)
    length_ratio = length_ratio_score(dialogue.text, source_text, normalize_text=normalize_text)
    merge_penalty = min(0.4, max(0, matched_segment_count - 1) * 0.08)
    token_coverage = min(1.0, covered_duration / max(1, dialogue.end_ms - dialogue.start_ms))
    score = max(
        0.0,
        min(
            1.0,
            (time_overlap * 0.45)
            + (boundary * 0.20)
            + (length_ratio * 0.15)
            + (token_coverage * 0.20)
            - merge_penalty,
        ),
    )
    reasons: list[str] = []
    if time_overlap < 0.45:
        reasons.append("time weak")
    if matched_segment_count >= 3:
        reasons.append(f"merged {matched_segment_count} splits")
    if token_coverage < 0.55:
        reasons.append("sparse tokens")
    if length_ratio < 0.45:
        reasons.append("length drift")
    if not source_text:
        reasons.append("empty source")
    return {
        "score": round(score, 4),
        "time_overlap_score": round(time_overlap, 4),
        "boundary_score": round(boundary, 4),
        "length_ratio_score": round(length_ratio, 4),
        "merge_penalty": round(merge_penalty, 4),
        "token_coverage_score": round(token_coverage, 4),
        "reasons": reasons or ["ok"],
    }


def interval_overlap_score(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    overlap = overlap_ms(start_a, end_a, start_b, end_b)
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0:
        return 0.0
    return max(0.0, min(1.0, overlap / union))


def boundary_score(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    duration = max(800, end_a - start_a)
    boundary_delta = abs(start_a - start_b) + abs(end_a - end_b)
    normalized = boundary_delta / (duration * 1.6)
    return max(0.0, min(1.0, 1.0 - normalized))


def length_ratio_score(chinese_text: str, source_text: str, *, normalize_text: Any) -> float:
    left = len(normalize_text(chinese_text))
    right = len(normalize_text(source_text))
    if left == 0 or right == 0:
        return 0.0
    shorter = min(left, right)
    longer = max(left, right)
    return shorter / longer


def overlap_ms(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))
