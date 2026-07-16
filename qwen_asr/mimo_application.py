from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qwen_asr.mimo_candidates import coerce_bool, coerce_confidence
from qwen_asr.mimo_guards import (
    ass_acceptance_guard,
    original_content_deletion_guard,
    original_high_risk_replacement_guard,
    original_no_ass_substantial_rewrite_guard,
    safe_suggestion_value,
    translation_shortening_guard,
)


@dataclass(slots=True)
class SuggestionApplication:
    updates: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    rejection: dict[str, Any] | None = None


def evaluate_stage2_suggestion(
    *,
    subtitle_id: str,
    normalized: dict[str, Any],
    current_item: dict[str, Any],
    apply_confidence_threshold: float,
) -> SuggestionApplication:
    reviewed_id = str(normalized.get("id", "")).strip()
    if reviewed_id != subtitle_id:
        return SuggestionApplication()
    confidence = coerce_confidence(normalized.get("confidence"), default=1.0)
    needs_audio_review = coerce_bool(normalized.get("needs_audio_review"))
    asr_suspect = coerce_bool(normalized.get("asr_suspect"))
    suggested_original = str(normalized.get("suggested_original", "")).strip()
    suggested = str(normalized.get("suggested_translation", "")).strip()
    current_original = str(current_item.get("original_subtitle", "")).strip()
    current = str(current_item.get("translated_subtitle", "")).strip()
    suggested_original = safe_suggestion_value(
        suggested_original,
        current_original,
        field="original_subtitle",
    )
    suggested = safe_suggestion_value(
        suggested,
        current,
        field="translated_subtitle",
    )

    ass_guard = ass_acceptance_guard(
        current_item,
        current_original=current_original,
        suggested_original=suggested_original,
    )
    if not ass_guard["accepted"]:
        return SuggestionApplication(
            rejection={
                "id": reviewed_id,
                "field": "original_subtitle",
                "reason": ass_guard["reason"],
                "ass_text": ass_guard.get("ass_text", ""),
                "current_score": ass_guard.get("current_score"),
                "suggested_score": ass_guard.get("suggested_score"),
                "suggested_original": suggested_original,
            }
        )

    deletion_guard = original_content_deletion_guard(
        current_original=current_original,
        suggested_original=suggested_original,
        ass_guard=ass_guard,
    )
    if not deletion_guard["accepted"]:
        return SuggestionApplication(
            rejection={
                "id": reviewed_id,
                "field": "original_subtitle",
                "reason": deletion_guard["reason"],
                "current_signal": deletion_guard.get("current_signal", ""),
                "suggested_signal": deletion_guard.get("suggested_signal", ""),
                "overlap": deletion_guard.get("overlap", 0),
                "suggested_original": suggested_original,
            }
        )

    high_risk_guard = original_high_risk_replacement_guard(
        current_original=current_original,
        suggested_original=suggested_original,
        ass_guard=ass_guard,
    )
    if not high_risk_guard["accepted"]:
        return SuggestionApplication(
            rejection={
                "id": reviewed_id,
                "field": "original_subtitle",
                "reason": high_risk_guard["reason"],
                "current_signal": high_risk_guard.get("current_signal", ""),
                "suggested_signal": high_risk_guard.get("suggested_signal", ""),
                "current_units": high_risk_guard.get("current_units", 0),
                "suggested_units": high_risk_guard.get("suggested_units", 0),
                "suggested_original": suggested_original,
            }
        )

    no_ass_rewrite_guard = original_no_ass_substantial_rewrite_guard(
        current_original=current_original,
        suggested_original=suggested_original,
        ass_guard=ass_guard,
    )
    if not no_ass_rewrite_guard["accepted"]:
        return SuggestionApplication(
            rejection={
                "id": reviewed_id,
                "field": "original_subtitle",
                "reason": no_ass_rewrite_guard["reason"],
                "current_signal": no_ass_rewrite_guard.get("current_signal", ""),
                "suggested_signal": no_ass_rewrite_guard.get("suggested_signal", ""),
                "current_units": no_ass_rewrite_guard.get("current_units", 0),
                "suggested_units": no_ass_rewrite_guard.get("suggested_units", 0),
                "overlap": no_ass_rewrite_guard.get("overlap", 0),
                "suggested_original": suggested_original,
            }
        )

    shortening_guard = translation_shortening_guard(
        current_translation=current,
        suggested_translation=suggested,
    )
    if not shortening_guard["accepted"]:
        return SuggestionApplication(
            rejection={
                "id": reviewed_id,
                "field": "translated_subtitle",
                "reason": shortening_guard["reason"],
                "current_units": shortening_guard.get("current_units", 0),
                "suggested_units": shortening_guard.get("suggested_units", 0),
                "suggested_translation": suggested,
            }
        )

    fields: dict[str, Any] = {}
    original_apply_threshold = max(0.9, apply_confidence_threshold)
    if (
        suggested_original
        and suggested_original != current_original
        and asr_suspect
        and not needs_audio_review
        and suggested
        and confidence >= original_apply_threshold
    ):
        fields["original_subtitle"] = suggested_original
    if suggested and suggested != current and not needs_audio_review and confidence >= apply_confidence_threshold:
        fields["translated_subtitle"] = suggested
    if not fields:
        return SuggestionApplication()

    evidence = {
        "confidence": confidence,
        "asr_suspect": asr_suspect,
        "needs_audio_review": needs_audio_review,
        "reason": str(normalized.get("reason", "")).strip(),
        "ass_guard": ass_guard,
        "before": {
            "original_subtitle": current_original,
            "translated_subtitle": current,
        },
        "suggested": {
            "original_subtitle": suggested_original,
            "translated_subtitle": suggested,
        },
    }
    fields["__proofread_evidence"] = evidence
    return SuggestionApplication(updates=fields, evidence=evidence)
