from __future__ import annotations

from qwen_asr.mimo_candidates import (
    build_manifest_suspect_report,
    collect_stage1_suspects,
    suspect_types_need_audio_review,
    translated_manifest_has_suspect_metadata,
)


def test_manifest_candidate_selection_keeps_audio_scope_precise() -> None:
    translated = {
        "1": {
            "original_subtitle": "hai",
            "translated_subtitle": "yes",
            "suspect_types": ["ass_short_dialogue_timing_shifted", "ass_fail_score"],
            "suspect_confidence": 1.0,
        },
        "2": {
            "original_subtitle": "eto",
            "translated_subtitle": "um",
            "suspect_types": ["ass_short_dialogue_missing", "ass_fail_score"],
            "suspect_confidence": 1.0,
        },
        "3": {
            "original_subtitle": "bad",
            "translated_subtitle": "bad",
            "suspect_types": [],
            "suspect_confidence": 0.2,
        },
    }

    assert translated_manifest_has_suspect_metadata(translated)
    assert not suspect_types_need_audio_review(["ass_short_dialogue_timing_shifted", "ass_fail_score"])
    report = build_manifest_suspect_report(translated, confidence_threshold=0.75)

    assert collect_stage1_suspects(report) == ["2", "3"]
    assert report[0]["source"] == "translation-manifest"


def test_manifest_candidates_skip_low_yield_only_types_but_keep_them_in_report() -> None:
    translated = {
        "1": {"original_subtitle": "うん", "translated_subtitle": "嗯", "asr_suspect": True,
              "needs_audio_review": True, "suspect_types": ["short_response"], "suspect_confidence": 0.6},
        "2": {"original_subtitle": "ぜひということで", "translated_subtitle": "那么", "asr_suspect": True,
              "suspect_types": ["fragment", "semantic"], "suspect_confidence": 0.6},
        "3": {"original_subtitle": "棘ナシとギャアリ", "translated_subtitle": "无刺有刺", "asr_suspect": True,
              "suspect_types": ["fragment", "entity"], "suspect_confidence": 0.6},
    }

    report = build_manifest_suspect_report(translated, confidence_threshold=0.75)

    assert collect_stage1_suspects(report) == ["3"]
    assert [item["id"] for item in report[0]["suggestions"]] == ["1", "2", "3"]
