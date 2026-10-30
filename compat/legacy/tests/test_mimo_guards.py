from __future__ import annotations

from qwen_asr.mimo_guards import (
    apply_branch_updates,
    normalize_qa_item,
    original_high_risk_replacement_guard,
    safe_suggestion_value,
    translation_shortening_guard,
)


def test_apply_branch_updates_records_evidence_and_realign_marker() -> None:
    branch = {
        "1": {
            "original_subtitle": "\u3042",
            "translated_subtitle": "\u662f",
        }
    }

    applied = apply_branch_updates(
        branch,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u597d\u7684",
                "__proofread_evidence": {"confidence": 0.95},
            }
        },
        source="unit-test",
    )

    assert applied == 1
    assert branch["1"]["original_subtitle"] == "\u306f\u3044"
    assert branch["1"]["translated_subtitle"] == "\u597d\u7684"
    assert branch["1"]["needs_realign"] is True
    assert branch["1"]["realign_status"] == "pending"
    assert branch["1"]["proofread_history"][0]["evidence"] == {"confidence": 0.95}


def test_normalize_and_safety_guards_reject_placeholders_and_wrong_script() -> None:
    normalized = normalize_qa_item({"i": " 7 ", "s": "None", "so": "\u306f\u3044", "c": "0.8"})

    assert normalized["id"] == "7"
    assert normalized["suggested_translation"] == ""
    assert normalized["suggested_original"] == "\u306f\u3044"
    assert safe_suggestion_value("hello", "\u306f\u3044", field="original_subtitle") == ""


def test_original_and_translation_guards_reject_high_risk_edits() -> None:
    original_decision = original_high_risk_replacement_guard(
        current_original="\u3046\u3093\u3002",
        suggested_original="\u5965\u69d8\u3002",
        ass_guard={"accepted": True, "reason": "no-ass-reference"},
    )
    translation_decision = translation_shortening_guard(
        current_translation="\u8fd9\u662f\u4e00\u53e5\u5f88\u957f\u7684\u7ffb\u8bd1",
        suggested_translation="\u597d",
    )

    assert original_decision["accepted"] is False
    assert original_decision["reason"] == "protected-short-response-replaced-without-ass-reference"
    assert translation_decision["accepted"] is False
    assert translation_decision["reason"] == "translation-abnormally-shortened"
