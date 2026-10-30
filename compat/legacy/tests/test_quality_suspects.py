from __future__ import annotations

from qwen_asr.mimo_proofread import _build_manifest_suspect_report, _collect_stage1_suspects
from qwen_asr.quality_suspects import apply_quality_diff_suspects_to_translated, apply_quality_suspects_to_translated


def test_quality_suspects_mark_translated_entries_for_audio_review() -> None:
    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1800,
            "original_subtitle": "\u306f\u3044",
            "translated_subtitle": "\u662f",
        },
        "2": {
            "start_time": 5000,
            "end_time": 6000,
            "original_subtitle": "\u6b21\u306e\u53f0\u8a5e",
            "translated_subtitle": "\u4e0b\u4e00\u53e5",
        },
    }
    ass_report = {
        "rows": [
            {
                "index": 7,
                "ass_text": "\u306f\u3044",
                "target_start_ms": 1200,
                "target_end_ms": 1500,
                "diagnostics": ["short-dialogue-missing"],
            }
        ]
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)
    updated = result["translated"]

    assert result["report"]["candidate_count"] == 1
    assert result["report"]["applied_count"] == 1
    assert updated["1"]["asr_suspect"] is True
    assert updated["1"]["needs_audio_review"] is True
    assert updated["1"]["suspect_types"] == ["ass_short_dialogue_missing"]
    stage1 = _build_manifest_suspect_report(updated, confidence_threshold=0.75)
    assert _collect_stage1_suspects(stage1) == ["1"]


def test_quality_suspects_skip_far_entries() -> None:
    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1800,
            "original_subtitle": "\u306f\u3044",
            "translated_subtitle": "\u662f",
        },
    }
    ass_report = {
        "rows": [
            {
                "index": 7,
                "ass_text": "\u306f\u3044",
                "target_start_ms": 12000,
                "target_end_ms": 12500,
                "diagnostics": ["short-dialogue-timing-shifted"],
            }
        ]
    }

    result = apply_quality_suspects_to_translated(translated, ass_report, max_distance_ms=2000)

    assert result["report"]["candidate_count"] == 1
    assert result["report"]["applied_count"] == 0
    assert "asr_suspect" not in result["translated"]["1"]


def test_quality_suspects_mark_low_score_rows_for_audio_review() -> None:
    translated = {
        "3": {
            "start_time": 1000,
            "end_time": 2000,
            "original_subtitle": "\u805e\u304d\u9055\u3048\u305f\u53f0\u8a5e",
            "translated_subtitle": "\u542c\u9519\u7684\u53f0\u8bcd",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u6b63\u3057\u3044\u53f0\u8a5e",
                "target_start_ms": 1100,
                "target_end_ms": 1900,
                "diagnostics": [],
                "score": 0.12,
                "level": "FAIL",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)
    updated = result["translated"]["3"]

    assert result["report"]["candidate_count"] == 1
    assert result["report"]["applied_count"] == 1
    assert updated["asr_suspect"] is True
    assert updated["needs_audio_review"] is True
    assert updated["suspect_types"] == ["ass_fail_score"]
    assert "ass_text=" in updated["suspect_reason"]


def test_quality_suspects_skip_normal_score_rows_without_diagnostics() -> None:
    translated = {
        "3": {
            "start_time": 1000,
            "end_time": 2000,
            "original_subtitle": "\u554f\u984c\u306a\u3044\u53f0\u8a5e",
            "translated_subtitle": "\u6ca1\u95ee\u9898\u7684\u53f0\u8bcd",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u554f\u984c\u306a\u3044\u53f0\u8a5e",
                "target_start_ms": 1100,
                "target_end_ms": 1900,
                "diagnostics": [],
                "score": 0.91,
                "level": "OK",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)

    assert result["report"]["candidate_count"] == 0
    assert "asr_suspect" not in result["translated"]["3"]


def test_quality_suspects_prefers_report_matched_key_for_low_score_rows() -> None:
    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1800,
            "original_subtitle": "\u6642\u9593\u7684\u306b\u8fd1\u3044",
            "translated_subtitle": "\u65f6\u95f4\u4e0a\u66f4\u8fd1",
        },
        "9": {
            "start_time": 5000,
            "end_time": 5800,
            "original_subtitle": "\u5b9f\u969b\u306e\u4f4e\u5206\u5b57\u5e55",
            "translated_subtitle": "\u5b9e\u9645\u4f4e\u5206\u5b57\u5e55",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u6b63\u3057\u3044\u53f0\u8a5e",
                "target_start_ms": 1100,
                "target_end_ms": 1700,
                "matched_key": "9",
                "diagnostics": [],
                "score": 0.12,
                "level": "FAIL",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)

    assert result["report"]["applied_count"] == 1
    assert "asr_suspect" not in result["translated"]["1"]
    assert result["translated"]["9"]["suspect_types"] == ["ass_fail_score"]


def test_quality_suspects_prefers_diagnostic_key_for_short_dialogue_rows() -> None:
    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1800,
            "original_subtitle": "\u6642\u9593\u7684\u306b\u8fd1\u3044",
            "translated_subtitle": "\u65f6\u95f4\u4e0a\u66f4\u8fd1",
        },
        "9": {
            "start_time": 5000,
            "end_time": 5800,
            "original_subtitle": "\u3046\u3093",
            "translated_subtitle": "\u55ef",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u3046\u3093",
                "target_start_ms": 1100,
                "target_end_ms": 1700,
                "matched_key": "1",
                "diagnostic_matched_key": "9",
                "diagnostics": ["short-dialogue-timing-shifted"],
                "score": 0.0,
                "level": "FAIL",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)

    assert result["report"]["applied_count"] == 1
    assert "asr_suspect" not in result["translated"]["1"]
    assert result["translated"]["9"]["suspect_types"] == ["ass_short_dialogue_timing_shifted", "ass_fail_score"]
    assert result["translated"]["9"]["asr_suspect"] is False
    assert result["translated"]["9"]["needs_audio_review"] is False
    assert result["translated"]["9"]["needs_realign"] is True
    assert result["translated"]["9"]["realign_status"] == "pending"


def test_quality_suspects_routes_timing_shifted_low_score_to_realign_only() -> None:
    translated = {
        "9": {
            "start_time": 5000,
            "end_time": 5800,
            "original_subtitle": "\u3046\u3093",
            "translated_subtitle": "\u55ef",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u3046\u3093",
                "target_start_ms": 1100,
                "target_end_ms": 1700,
                "diagnostic_matched_key": "9",
                "diagnostics": ["short-dialogue-timing-shifted"],
                "score": 0.12,
                "level": "FAIL",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)
    updated = result["translated"]["9"]
    stage1 = _build_manifest_suspect_report(result["translated"], confidence_threshold=0.75)

    assert result["report"]["applied_count"] == 1
    assert updated["suspect_types"] == ["ass_short_dialogue_timing_shifted", "ass_fail_score"]
    assert updated["asr_suspect"] is False
    assert updated["needs_audio_review"] is False
    assert updated["needs_realign"] is True
    assert _collect_stage1_suspects(stage1) == []


def test_quality_suspects_adds_diagnostic_candidate_for_short_missing_with_better_wide_match() -> None:
    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1800,
            "original_subtitle": "\u6642\u9593\u7684\u306b\u8fd1\u3044",
            "translated_subtitle": "\u65f6\u95f4\u4e0a\u66f4\u8fd1",
        },
        "9": {
            "start_time": 5000,
            "end_time": 5800,
            "original_subtitle": "\u3046\u3093",
            "translated_subtitle": "\u55ef",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u3046\u3093",
                "target_start_ms": 1100,
                "target_end_ms": 1700,
                "matched_key": "1",
                "diagnostic_matched_key": "9",
                "diagnostics": ["short-dialogue-missing"],
                "score": 0.0,
                "diagnostic_score": 0.67,
                "level": "FAIL",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)

    assert result["report"]["candidate_count"] == 2
    assert result["report"]["applied_count"] == 2
    assert result["translated"]["1"]["suspect_types"] == ["ass_short_dialogue_missing", "ass_fail_score"]
    assert result["translated"]["9"]["suspect_types"] == ["ass_short_dialogue_missing", "ass_fail_score"]
    assert [item["route"] for item in result["report"]["applied"]] == ["primary", "diagnostic"]


def test_quality_suspects_does_not_add_weak_diagnostic_candidate_for_short_missing() -> None:
    translated = {
        "1": {
            "start_time": 1000,
            "end_time": 1800,
            "original_subtitle": "\u6642\u9593\u7684\u306b\u8fd1\u3044",
            "translated_subtitle": "\u65f6\u95f4\u4e0a\u66f4\u8fd1",
        },
        "9": {
            "start_time": 5000,
            "end_time": 5800,
            "original_subtitle": "\u3046\u3093",
            "translated_subtitle": "\u55ef",
        },
    }
    ass_report = {
        "thresholds": {"low_score": 0.45, "fail_score": 0.20},
        "rows": [
            {
                "index": 8,
                "ass_text": "\u3046\u3093",
                "target_start_ms": 1100,
                "target_end_ms": 1700,
                "matched_key": "1",
                "diagnostic_matched_key": "9",
                "diagnostics": ["short-dialogue-missing"],
                "score": 0.0,
                "diagnostic_score": 0.28,
                "level": "FAIL",
            }
        ],
    }

    result = apply_quality_suspects_to_translated(translated, ass_report)

    assert result["report"]["candidate_count"] == 1
    assert result["report"]["applied_count"] == 1
    assert result["translated"]["1"]["suspect_types"] == ["ass_short_dialogue_missing", "ass_fail_score"]
    assert "asr_suspect" not in result["translated"]["9"]


def test_quality_diff_suspects_mark_stage_regressions_for_audio_review() -> None:
    translated = {
        "10": {
            "start_time": 540000,
            "end_time": 541000,
            "original_subtitle": "\u3058\u3083\u306d\u3002",
            "translated_subtitle": "\u518d\u89c1",
        },
    }
    diff_report = {
        "issues": [
            {
                "index": 108,
                "type": "became-fail",
                "severity": "FAIL",
                "transition": "aligned -> split",
                "ass_text": "\u3058\u3083\u3042\u306d",
                "target_start_ms": 540100,
                "target_end_ms": 540800,
                "score_drop": 0.85,
                "current_diagnostics": ["short-dialogue-missing"],
            }
        ]
    }

    result = apply_quality_diff_suspects_to_translated(translated, diff_report)
    updated = result["translated"]["10"]

    assert result["report"]["candidate_count"] == 1
    assert result["report"]["applied_count"] == 1
    assert updated["asr_suspect"] is True
    assert updated["needs_audio_review"] is True
    assert updated["suspect_types"] == ["ass_stage_became_fail", "ass_short_dialogue_missing"]
    assert "aligned -> split" in updated["suspect_reason"]


def test_quality_diff_suspects_skip_small_warn_score_drop() -> None:
    translated = {
        "10": {
            "start_time": 540000,
            "end_time": 541000,
            "original_subtitle": "\u3058\u3083\u306d\u3002",
            "translated_subtitle": "\u518d\u89c1",
        },
    }
    diff_report = {
        "issues": [
            {
                "index": 108,
                "type": "score-drop",
                "severity": "WARN",
                "transition": "aligned -> split",
                "ass_text": "\u3058\u3083\u3042\u306d",
                "target_start_ms": 540100,
                "target_end_ms": 540800,
                "score_drop": 0.12,
                "current_diagnostics": [],
            }
        ]
    }

    result = apply_quality_diff_suspects_to_translated(translated, diff_report)

    assert result["report"]["candidate_count"] == 0
    assert "asr_suspect" not in result["translated"]["10"]
