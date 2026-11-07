from __future__ import annotations

from pathlib import Path

from qwen_asr.ass_quality import (
    SubtitleCue,
    ass_match_score,
    estimate_global_offset_ms,
    evaluate_ass_quality,
    load_aligned_segment_cues,
    parse_srt_cues,
    select_reference_dialogues,
)
from qwen_asr.history_glossary import AssDialogue
from qwen_asr.models import WorkPaths


def test_select_reference_dialogues_keeps_main_japanese_and_excludes_songs() -> None:
    dialogues = [
        AssDialogue(0, 1000, "OP - JP", "\u3046\u305f"),
        AssDialogue(1000, 2000, "Text - JP", "\u306f\u3044"),
        AssDialogue(2000, 3000, "Text - CN", "\u4f60\u597d"),
        AssDialogue(3000, 4000, "Text - JP - UP", "\u3048\uff1f"),
    ]

    selected = select_reference_dialogues(dialogues)

    assert [item.text for item in selected] == ["\u306f\u3044", "\u3048\uff1f"]


def test_estimate_global_offset_uses_high_confidence_matches() -> None:
    dialogues = [
        AssDialogue(1000, 1800, "Text - JP", "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059"),
        AssDialogue(3000, 3800, "Text - JP", "\u6b21\u306e\u53f0\u8a5e\u3067\u3059"),
    ]
    cues = [
        SubtitleCue("\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059", 7180, 7980),
        SubtitleCue("\u6b21\u306e\u53f0\u8a5e\u3067\u3059", 9180, 9980),
    ]

    assert estimate_global_offset_ms(dialogues, cues) == 6180


def test_estimate_global_offset_uses_densest_anchor_cluster() -> None:
    dialogues = [
        AssDialogue(1000, 1800, "Text - JP", "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059"),
        AssDialogue(3000, 3800, "Text - JP", "\u6b21\u306e\u53f0\u8a5e\u3067\u3059"),
        AssDialogue(5000, 5800, "Text - JP", "\u5225\u306e\u9577\u3044\u53f0\u8a5e\u3067\u3059"),
    ]
    cues = [
        SubtitleCue("\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059", -199000, -198200),
        SubtitleCue("\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059", 7180, 7980),
        SubtitleCue("\u6b21\u306e\u53f0\u8a5e\u3067\u3059", 9180, 9980),
        SubtitleCue("\u5225\u306e\u9577\u3044\u53f0\u8a5e\u3067\u3059", 11180, 11980),
    ]

    assert estimate_global_offset_ms(dialogues, cues) == 6180


def test_evaluate_ass_quality_reports_low_scores() -> None:
    dialogues = [
        AssDialogue(1000, 1800, "Text - JP", "\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059"),
        AssDialogue(3000, 3800, "Text - JP", "\u6b21\u306e\u53f0\u8a5e\u3067\u3059"),
    ]
    cues = [
        SubtitleCue("\u3053\u308c\u306f\u30c6\u30b9\u30c8\u3067\u3059", 1000, 1800, "cue-1"),
        SubtitleCue("\u307e\u3063\u305f\u304f\u9055\u3046", 3000, 3800, "cue-2"),
    ]

    report = evaluate_ass_quality(
        ass_path=Path("reference.ass"),
        source="split",
        dialogues=dialogues,
        cues=cues,
        offset_ms=0,
        window_ms=300,
        diagnostic_window_ms=2000,
        low_score_threshold=0.45,
        fail_score_threshold=0.05,
        max_cases=2,
    )

    assert report["status"] == "FAIL"
    assert report["summary"]["score_ge_045"] == 1
    assert report["summary"]["score_lt_045"] == 1
    assert len(report["rows"]) == 2
    assert report["rows"][0]["matched_key"] == "cue-1"
    assert report["rows"][1]["matched_key"] == "cue-2"
    assert "diagnostic_matched_key" in report["rows"][1]


def test_evaluate_ass_quality_marks_short_and_overlong_diagnostics() -> None:
    dialogues = [
        AssDialogue(1000, 1300, "Text - JP", "\u306f\u3044"),
        AssDialogue(3000, 3600, "Text - JP", "\u3042\u306e\u30dd\u30b9\u30bf\u30fc"),
    ]
    cues = [
        SubtitleCue("", 1000, 1300),
        SubtitleCue("\u3053\u308c\u3001\u3042\u3001\u3042\u306e\u30dd\u30b9\u30bf\u30fc\u3001\u30b3\u30df\u30c6\u30a3\u30a2\u306e\u672c", 3000, 4300),
    ]

    report = evaluate_ass_quality(
        ass_path=Path("reference.ass"),
        source="split",
        dialogues=dialogues,
        cues=cues,
        offset_ms=0,
        window_ms=300,
        diagnostic_window_ms=2000,
        low_score_threshold=0.45,
        fail_score_threshold=0.20,
        max_cases=2,
    )

    assert report["summary"]["short_dialogue_low_score"] == 1
    assert report["summary"]["overlong_match"] == 1
    assert "short-dialogue-low-score" in report["rows"][0]["diagnostics"]
    assert "short-dialogue-missing" in report["rows"][0]["diagnostics"]
    assert "overlong-match" in report["rows"][1]["diagnostics"]


def test_overlong_match_requires_non_low_score_match() -> None:
    dialogues = [
        AssDialogue(1000, 1300, "Text - JP", "\u8352\u5510\u7121\u7a3d"),
    ]
    cues = [
        SubtitleCue("\u5f8c\u982d\u90e8\u3001\u4e71\u66b4\u8001\u7e3e\u3001\u7121\u610f\u5473\u306e\u53e3\u305a\u3044\u3067\u3059\u3002", 1000, 2600),
    ]

    report = evaluate_ass_quality(
        ass_path=Path("reference.ass"),
        source="split",
        dialogues=dialogues,
        cues=cues,
        offset_ms=0,
        window_ms=300,
        diagnostic_window_ms=2000,
        low_score_threshold=0.45,
        fail_score_threshold=0.20,
        max_cases=2,
    )

    assert report["summary"]["short_dialogue_low_score"] == 1
    assert report["summary"]["overlong_match"] == 0
    assert "overlong-match" not in report["rows"][0]["diagnostics"]


def test_evaluate_ass_quality_marks_timing_shifted_short_dialogue() -> None:
    dialogues = [
        AssDialogue(1000, 1300, "Text - JP", "\u306f\u3044"),
    ]
    cues = [
        SubtitleCue("\u306f\u3044", 2600, 3000),
    ]

    report = evaluate_ass_quality(
        ass_path=Path("reference.ass"),
        source="split",
        dialogues=dialogues,
        cues=cues,
        offset_ms=0,
        window_ms=300,
        diagnostic_window_ms=3000,
        low_score_threshold=0.45,
        fail_score_threshold=0.20,
        max_cases=2,
    )

    assert report["summary"]["short_dialogue_timing_shifted"] == 1
    assert report["summary"]["short_dialogue_missing"] == 0
    assert "short-dialogue-timing-shifted" in report["rows"][0]["diagnostics"]
    assert report["rows"][0]["diagnostic_distance_ms"] == 1300


def test_short_dialogue_wide_match_requires_complete_diagnostic_text() -> None:
    dialogues = [
        AssDialogue(1000, 1400, "Text - JP", "\u306f\u2026 \u306f\u3044"),
    ]
    cues = [
        SubtitleCue("\u306f\u3044", 2600, 3000),
    ]

    report = evaluate_ass_quality(
        ass_path=Path("reference.ass"),
        source="split",
        dialogues=dialogues,
        cues=cues,
        offset_ms=0,
        window_ms=300,
        diagnostic_window_ms=3000,
        low_score_threshold=0.45,
        fail_score_threshold=0.20,
        max_cases=2,
    )

    assert report["summary"]["short_dialogue_timing_shifted"] == 0
    assert report["summary"]["short_dialogue_missing"] == 1
    assert "short-dialogue-missing" in report["rows"][0]["diagnostics"]


def test_short_dialogue_embedded_in_long_candidate_is_not_perfect_match() -> None:
    assert ass_match_score("\u306f\u3044", "\u306f\u3044") == 1.0
    assert ass_match_score(
        "\u306f\u3044",
        "\u5144\u3055\u3093\u305f\u3061\u306d\u3001\u3084\u3063\u3068\u5916\u306b\u51fa\u3089\u308c\u308b\u308f\u3002\u306f\u3044\u3002",
    ) < 0.45


def test_long_dialogue_is_not_matched_by_single_token() -> None:
    assert ass_match_score("\u8d64\u798f\u3093\u3061\u306b\u6cca\u307e\u3063\u305f\u3063\u3066\u3053\u3068\u306b\u3057\u3066", "\u3068") < 0.45
    assert ass_match_score("\u8352\u5510\u7121\u7a3d", "\u7121") < 0.45


def test_ass_match_treats_hiragana_and_katakana_as_equivalent() -> None:
    assert ass_match_score("\u306d\u3048 \u30b7\u30bf\u30e9", "\u306d\u3048\u3001\u3057\u305f\u3089") == 1.0
    assert ass_match_score("\u30df\u30cf\u30a4\u30eb", "\u307f\u306f\u3044\u308b") == 1.0
    assert ass_match_score("\u3053\u308c \u307e\u305f\u501f\u308a\u308b\uff01", "\u30ab\u30ec\u30de\u30bf\u501f\u308a\u308b\u3002") < 0.45


def test_load_aligned_segment_cues_uses_manifest_segment_text(tmp_path: Path) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    work_paths.aligned_manifest.write_text(
        "["
        "{"
        '"global_start_time": 1.0,'
        '"global_end_time": 2.5,'
        '"text": "\\u8d64\\u798f\\u3093\\u3061\\u306b\\u6cca\\u307e\\u3063\\u305f",'
        '"tokens": [{"text": "\\u3068", "start_time": 1.2, "end_time": 1.3}]'
        "}"
        "]",
        encoding="utf-8",
    )

    cues = load_aligned_segment_cues(work_paths)

    assert cues == [SubtitleCue("\u8d64\u798f\u3093\u3061\u306b\u6cca\u307e\u3063\u305f", 1000, 2500, "")]


def test_parse_srt_cues_reads_first_text_line_as_japanese(tmp_path: Path) -> None:
    path = tmp_path / "sample.srt"
    path.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "\u306f\u3044\n"
        "\u662f\n",
        encoding="utf-8",
    )

    cues = parse_srt_cues(path)

    assert cues == [SubtitleCue("\u306f\u3044", 1000, 2000, "1")]
