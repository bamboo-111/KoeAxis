from __future__ import annotations

import re
from typing import Any, Callable

from qwen_asr.mimo_candidates import coerce_bool, coerce_confidence


def apply_branch_updates(
    branch: dict[str, Any],
    updates: dict[str, dict[str, Any]],
    *,
    source: str,
) -> int:
    applied = 0
    for subtitle_id, fields in updates.items():
        item = branch.get(subtitle_id)
        if not isinstance(item, dict):
            continue
        evidence = fields.get("__proofread_evidence")
        if not isinstance(evidence, dict):
            evidence = {}
        changed: dict[str, dict[str, str]] = {}
        for field, suggested in fields.items():
            if field not in {"original_subtitle", "translated_subtitle"}:
                continue
            value = str(suggested).strip()
            before = str(item.get(field, "")).strip()
            if not value or value == before:
                continue
            item[field] = value
            changed[field] = {"before": before, "after": value}
        if not changed:
            continue
        if "original_subtitle" in changed:
            item["needs_realign"] = True
            item["realign_status"] = "pending"
        history = item.setdefault("proofread_history", [])
        if not isinstance(history, list):
            history = []
            item["proofread_history"] = history
        entry: dict[str, Any] = {"source": source, "changes": changed}
        if evidence:
            entry["evidence"] = evidence
        history.append(entry)
        applied += 1
    return applied


def normalize_qa_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "id": item.get("id", item.get("i", "")),
        "error_type": item.get("error_type", item.get("t", "")),
        "original": item.get("original", item.get("o", "")),
        "translation": item.get("translation", item.get("tr", "")),
        "suggested_translation": item.get("suggested_translation", item.get("s", "")),
        "asr_suspect": item.get("asr_suspect", item.get("a", False)),
        "suggested_original": item.get(
            "suggested_original",
            item.get("suspected_original", item.get("so", "")),
        ),
        "needs_audio_review": item.get("needs_audio_review", item.get("n", False)),
        "reason": item.get("reason", item.get("r", "")),
        "confidence": item.get("confidence", item.get("c", 0.0)),
    }
    normalized["id"] = str(normalized.get("id", "")).strip()
    error_type = str(normalized.get("error_type", "")).strip()
    allowed = {"translation_error", "term_error", "asr_suspect", "needs_context", "style_only"}
    normalized["error_type"] = error_type if error_type in allowed else ""
    normalized["asr_suspect"] = coerce_bool(normalized.get("asr_suspect"))
    normalized["needs_audio_review"] = coerce_bool(normalized.get("needs_audio_review"))
    normalized["confidence"] = coerce_confidence(normalized.get("confidence"), default=0.0)
    for key in ("original", "translation", "suggested_translation", "suggested_original", "reason"):
        normalized[key] = str(normalized.get(key, "")).strip()
    normalized["suggested_translation"] = clean_placeholder_value(normalized["suggested_translation"])
    normalized["suggested_original"] = clean_placeholder_value(normalized["suggested_original"])
    return normalized


def clean_placeholder_value(value: str) -> str:
    text = str(value or "").strip()
    if text.casefold() in {"none", "null", "nil", "n/a", "na", "なし", "無し", "无", "無"}:
        return ""
    return text


def safe_suggestion_value(value: str, current: str, *, field: str) -> str:
    text = clean_placeholder_value(value)
    if not text or text == current:
        return ""
    if field == "original_subtitle" and contains_japanese(current) and not contains_japanese(text):
        return ""
    if (
        field == "translated_subtitle"
        and contains_cjk(current)
        and not contains_cjk(text)
        and re.search(r"[A-Za-z]{2,}", text)
    ):
        return ""
    return text


def ass_acceptance_guard(
    item: dict[str, Any],
    *,
    current_original: str,
    suggested_original: str,
    min_improvement: float = 0.05,
) -> dict[str, Any]:
    if not suggested_original:
        return {"accepted": True, "reason": "no-original-change"}
    ass_text = extract_guard_ass_text(item)
    if not ass_text:
        return {"accepted": True, "reason": "no-ass-reference"}
    from qwen_asr.ass_quality import ass_match_score, normalize_for_match

    current_score = ass_match_score(ass_text, current_original)
    suggested_score = ass_match_score(ass_text, suggested_original)
    fragment_guard = ass_fragment_replacement_guard(
        ass_text=ass_text,
        current_original=current_original,
        suggested_original=suggested_original,
        current_score=current_score,
        suggested_score=suggested_score,
        normalize=normalize_for_match,
    )
    if not fragment_guard["accepted"]:
        return fragment_guard
    if suggested_score >= 0.95:
        return {
            "accepted": True,
            "reason": "ass-high-score",
            "ass_text": ass_text,
            "current_score": round(current_score, 6),
            "suggested_score": round(suggested_score, 6),
        }
    if current_score >= 0.75 and suggested_score + 1e-9 >= current_score:
        return {
            "accepted": True,
            "reason": "ass-no-regression",
            "ass_text": ass_text,
            "current_score": round(current_score, 6),
            "suggested_score": round(suggested_score, 6),
        }
    if suggested_score + 1e-9 >= current_score + min_improvement:
        return {
            "accepted": True,
            "reason": "ass-improved",
            "ass_text": ass_text,
            "current_score": round(current_score, 6),
            "suggested_score": round(suggested_score, 6),
        }
    return {
        "accepted": False,
        "reason": "ass-score-not-improved",
        "ass_text": ass_text,
        "current_score": round(current_score, 6),
        "suggested_score": round(suggested_score, 6),
    }


def ass_fragment_replacement_guard(
    *,
    ass_text: str,
    current_original: str,
    suggested_original: str,
    current_score: float,
    suggested_score: float,
    normalize: Callable[[str], str],
    min_reference_units: int = 12,
    min_current_units: int = 6,
    max_suggested_reference_ratio: float = 0.75,
    min_current_score: float = 0.20,
    high_suggested_score: float = 0.75,
    min_overlap_ratio: float = 0.50,
) -> dict[str, Any]:
    reference_signal = normalize(ass_text)
    current_signal = normalize(current_original)
    suggested_signal = normalize(suggested_original)
    if (
        len(reference_signal) < min_reference_units
        or len(current_signal) < min_current_units
        or not suggested_signal
        or current_score < min_current_score
        or suggested_score >= high_suggested_score
        or len(suggested_signal) >= len(reference_signal) * max_suggested_reference_ratio
    ):
        return {"accepted": True, "reason": "not-long-reference-fragment"}
    overlap = longest_common_substring_len(current_signal, suggested_signal)
    min_pair_units = min(len(current_signal), len(suggested_signal))
    if min_pair_units and overlap <= min_pair_units * min_overlap_ratio:
        return {
            "accepted": False,
            "reason": "ass-long-reference-fragment-replacement",
            "ass_text": ass_text,
            "current_score": round(current_score, 6),
            "suggested_score": round(suggested_score, 6),
            "reference_units": len(reference_signal),
            "current_units": len(current_signal),
            "suggested_units": len(suggested_signal),
            "overlap": overlap,
        }
    return {"accepted": True, "reason": "fragment-overlap-ok", "overlap": overlap}


def original_content_deletion_guard(
    *,
    current_original: str,
    suggested_original: str,
    ass_guard: dict[str, Any],
    min_current_units: int = 4,
    min_dropped_units: int = 3,
) -> dict[str, Any]:
    if not suggested_original or suggested_original == current_original:
        return {"accepted": True, "reason": "no-original-change"}
    if str(ass_guard.get("reason", "")) not in {"no-ass-reference", "no-original-change"}:
        return {"accepted": True, "reason": "ass-reference-accepted"}
    current_signal = japanese_signal(current_original)
    suggested_signal = japanese_signal(suggested_original)
    if len(current_signal) < min_current_units or not suggested_signal:
        return {"accepted": True, "reason": "short-original"}
    if current_signal in suggested_signal:
        return {"accepted": True, "reason": "content-preserved"}
    overlap = longest_common_substring_len(current_signal, suggested_signal)
    dropped_units = len(current_signal) - overlap
    if dropped_units >= min_dropped_units:
        return {
            "accepted": False,
            "reason": "original-content-dropped-without-ass-reference",
            "current_signal": current_signal,
            "suggested_signal": suggested_signal,
            "overlap": overlap,
        }
    return {"accepted": True, "reason": "minor-original-edit", "overlap": overlap}


def original_high_risk_replacement_guard(
    *,
    current_original: str,
    suggested_original: str,
    ass_guard: dict[str, Any],
    max_short_units: int = 3,
    min_expanded_units: int = 12,
) -> dict[str, Any]:
    if not suggested_original or suggested_original == current_original:
        return {"accepted": True, "reason": "no-original-change"}
    if str(ass_guard.get("reason", "")) not in {"no-ass-reference", "no-original-change"}:
        return {"accepted": True, "reason": "ass-reference-accepted"}
    current_signal = japanese_signal(current_original)
    suggested_signal = japanese_signal(suggested_original)
    if not current_signal or not suggested_signal:
        return {"accepted": True, "reason": "empty-signal"}
    if is_protected_short_response_signal(current_signal) and current_signal != suggested_signal:
        return {
            "accepted": False,
            "reason": "protected-short-response-replaced-without-ass-reference",
            "current_signal": current_signal,
            "suggested_signal": suggested_signal,
            "current_units": len(current_signal),
            "suggested_units": len(suggested_signal),
        }
    if len(current_signal) <= max_short_units and len(suggested_signal) >= min_expanded_units:
        return {
            "accepted": False,
            "reason": "short-response-expanded-without-ass-reference",
            "current_signal": current_signal,
            "suggested_signal": suggested_signal,
            "current_units": len(current_signal),
            "suggested_units": len(suggested_signal),
        }
    return {
        "accepted": True,
        "reason": "not-high-risk-original-replacement",
        "current_units": len(current_signal),
        "suggested_units": len(suggested_signal),
    }


def is_protected_short_response_signal(signal: str) -> bool:
    return signal in {
        "\u306f\u3044",
        "\u3048",
        "\u3046\u3093",
        "\u3046\u3046\u3093",
        "\u3044\u3044\u3048",
        "\u99c4\u76ee",
        "\u3060\u3081",
        "\u30c0\u30e1",
        "\u306f\u3042",
        "\u304a",
        "\u304a\u304a",
    }


def original_no_ass_substantial_rewrite_guard(
    *,
    current_original: str,
    suggested_original: str,
    ass_guard: dict[str, Any],
) -> dict[str, Any]:
    if not suggested_original or suggested_original == current_original:
        return {"accepted": True, "reason": "no-original-change"}
    if str(ass_guard.get("reason", "")) != "no-ass-reference":
        return {"accepted": True, "reason": "ass-reference-accepted"}
    current_signal = japanese_signal(current_original)
    suggested_signal = japanese_signal(suggested_original)
    if not current_signal or not suggested_signal:
        return {"accepted": True, "reason": "empty-signal"}
    if current_signal == suggested_signal:
        return {"accepted": True, "reason": "same-original-signal"}
    overlap = longest_common_substring_len(current_signal, suggested_signal)
    return {
        "accepted": False,
        "reason": "no-ass-reference-original-change",
        "current_signal": current_signal,
        "suggested_signal": suggested_signal,
        "current_units": len(current_signal),
        "suggested_units": len(suggested_signal),
        "overlap": overlap,
    }


def translation_shortening_guard(
    *,
    current_translation: str,
    suggested_translation: str,
    min_current_units: int = 8,
    max_ratio: float = 0.34,
) -> dict[str, Any]:
    if not suggested_translation or suggested_translation == current_translation:
        return {"accepted": True, "reason": "no-translation-change"}
    current_units = cjk_signal_len(current_translation)
    suggested_units = cjk_signal_len(suggested_translation)
    if current_units < min_current_units or suggested_units == 0:
        return {"accepted": True, "reason": "short-translation"}
    if suggested_units < current_units * max_ratio:
        return {
            "accepted": False,
            "reason": "translation-abnormally-shortened",
            "current_units": current_units,
            "suggested_units": suggested_units,
        }
    return {"accepted": True, "reason": "translation-length-ok"}


def japanese_signal(text: str) -> str:
    return "".join(
        char
        for char in str(text or "")
        if 0x3040 <= ord(char) <= 0x30FF
        or 0x3400 <= ord(char) <= 0x9FFF
        or 0xFF66 <= ord(char) <= 0xFF9D
    )


def cjk_signal_len(text: str) -> int:
    return sum(1 for char in str(text or "") if 0x3400 <= ord(char) <= 0x9FFF)


def longest_common_substring_len(left: str, right: str) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    best = 0
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            value = previous[index - 1] + 1 if left_char == right_char else 0
            current.append(value)
            if value > best:
                best = value
        previous = current
    return best


def extract_guard_ass_text(item: dict[str, Any]) -> str:
    reasons = [
        str(item.get("suspect_reason", "") or ""),
        str(item.get("reason", "") or ""),
    ]
    matches: list[str] = []
    for reason in reasons:
        matches.extend(re.findall(r"ass_text=([^;]+)", reason))
    return matches[-1].strip() if matches else ""


def contains_japanese(text: str) -> bool:
    return any(
        0x3040 <= ord(char) <= 0x30FF
        or 0x3400 <= ord(char) <= 0x9FFF
        or 0xFF66 <= ord(char) <= 0xFF9D
        for char in str(text or "")
    )


def contains_cjk(text: str) -> bool:
    return any(0x3400 <= ord(char) <= 0x9FFF for char in str(text or ""))
