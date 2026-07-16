from __future__ import annotations

from qwen_asr.mimo_application import evaluate_stage2_suggestion


def test_stage2_application_accepts_supported_original_and_translation_update() -> None:
    current = {
        "original_subtitle": "\u3042",
        "translated_subtitle": "\u554a",
        "suspect_reason": "ASS stage diff suspect: ass_text=\u306f\u3044",
    }
    normalized = {
        "id": "1",
        "suggested_original": "\u306f\u3044",
        "suggested_translation": "\u597d\u7684",
        "asr_suspect": True,
        "needs_audio_review": False,
        "confidence": 0.95,
        "reason": "audio clear",
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="1",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.rejection is None
    assert result.updates["original_subtitle"] == "\u306f\u3044"
    assert result.updates["translated_subtitle"] == "\u597d\u7684"
    assert result.evidence["ass_guard"]["accepted"] is True
    assert result.updates["__proofread_evidence"] == result.evidence


def test_stage2_application_rejects_unbacked_original_content_deletion() -> None:
    current = {
        "original_subtitle": "\u305d\u3093\u306a\u30ba\u30e0\u30eb\u30c9",
        "translated_subtitle": "\u90a3\u6837\u7684\u7956\u7a46\u5c14\u5fb7",
        "suspect_reason": "manifest suspect without ass_text",
    }
    normalized = {
        "id": "1",
        "suggested_original": "\u30ba\u30e0\u30eb\u30c9\u3055\u3093\uff01",
        "suggested_translation": "\u7956\u7a46\u5c14\u5fb7\u5148\u751f",
        "asr_suspect": True,
        "needs_audio_review": False,
        "confidence": 0.95,
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="1",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.updates == {}
    assert result.rejection is not None
    assert result.rejection["reason"] == "original-content-dropped-without-ass-reference"


def test_stage2_application_requires_audio_review_to_be_cleared_before_apply() -> None:
    current = {
        "original_subtitle": "\u3042",
        "translated_subtitle": "\u554a",
        "suspect_reason": "ASS stage diff suspect: ass_text=\u306f\u3044",
    }
    normalized = {
        "id": "1",
        "suggested_original": "\u306f\u3044",
        "suggested_translation": "\u597d\u7684",
        "asr_suspect": True,
        "needs_audio_review": True,
        "confidence": 0.95,
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="1",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.rejection is None
    assert result.updates == {}
