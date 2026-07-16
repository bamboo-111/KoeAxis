from __future__ import annotations

from qwen_asr.mfa_experiment import _local_ass_match_score
from qwen_asr.mfa_experiment import _local_mfa_ass_guard
from qwen_asr.mfa_experiment import _mfa_writeback_dry_run
from qwen_asr.mfa_experiment import _range_distance_ms
from qwen_asr.mfa_guards import local_ass_match_score
from qwen_asr.mfa_guards import local_mfa_ass_guard
from qwen_asr.mfa_guards import mfa_writeback_dry_run
from qwen_asr.mfa_guards import normalize_local_match_text
from qwen_asr.mfa_guards import range_distance_ms


def test_local_mfa_ass_guard_passes_matching_text_and_overlapping_time() -> None:
    candidate = {"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044"}
    words = [{"start_ms": 1020, "end_ms": 1240, "text": "\u306f\u3044"}]
    quality = {"usable": True, "unknown_count": 0}

    result = local_mfa_ass_guard(candidate, "\u306f\u3044", words, quality)

    assert result["status"] == "PASS"
    assert result["text_score"] == 1.0
    assert result["time_overlaps_candidate"] is True
    assert result["reasons"] == []


def test_local_mfa_ass_guard_reports_unknown_and_non_overlapping_time() -> None:
    candidate = {"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044"}
    words = [{"start_ms": 2000, "end_ms": 2100, "text": "<unk>"}]
    quality = {"usable": False, "unknown_count": 1}

    result = local_mfa_ass_guard(candidate, "\u306f\u3044", words, quality)

    assert result["status"] == "FAIL"
    assert result["time_distance_ms"] == 700
    assert result["reasons"] == [
        "mfa-output-unusable",
        "mfa-unknown-word",
        "local-text-score-low",
        "mfa-time-outside-candidate-window",
    ]


def test_mfa_writeback_dry_run_passes_when_guard_improves_current_score() -> None:
    candidate = {"details": {"previous_score": 0.2, "current_score": 0.0}}
    guard = {
        "status": "PASS",
        "text_score": 1.0,
        "candidate_start_ms": 1000,
        "candidate_end_ms": 1300,
        "mfa_start_ms": 1020,
        "mfa_end_ms": 1240,
    }

    result = mfa_writeback_dry_run(candidate, guard)

    assert result["status"] == "PASS"
    assert result["score_delta_vs_current"] == 1.0
    assert result["reasons"] == []


def test_mfa_writeback_dry_run_rejects_score_drop() -> None:
    candidate = {"details": {"previous_score": 0.9, "current_score": 0.8}}
    guard = {"status": "PASS", "text_score": 0.7}

    result = mfa_writeback_dry_run(candidate, guard)

    assert result["status"] == "FAIL"
    assert result["reasons"] == ["would-lower-current-score"]


def test_local_match_score_normalizes_punctuation_width_and_case() -> None:
    assert normalize_local_match_text("\uff28\uff21\uff29\u3001") == "hai"
    assert local_ass_match_score("\u306f\u3044\u3002", "\u306f\u3044") == 1.0


def test_range_distance_ms_returns_zero_for_overlap_and_gap_for_disjoint_ranges() -> None:
    assert range_distance_ms(100, 200, 150, 250) == 0
    assert range_distance_ms(100, 200, 250, 300) == 50
    assert range_distance_ms(250, 300, 100, 200) == 50


def test_mfa_experiment_legacy_guard_aliases_use_guard_module() -> None:
    candidate = {"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044"}
    words = [{"start_ms": 1020, "end_ms": 1240, "text": "\u306f\u3044"}]
    quality = {"usable": True, "unknown_count": 0}
    guard = local_mfa_ass_guard(candidate, "\u306f\u3044", words, quality)

    assert _local_mfa_ass_guard(candidate, "\u306f\u3044", words, quality) == guard
    assert _mfa_writeback_dry_run(
        {"details": {"previous_score": 0.2, "current_score": 0.0}},
        guard,
    ) == mfa_writeback_dry_run(
        {"details": {"previous_score": 0.2, "current_score": 0.0}},
        guard,
    )
    assert _local_ass_match_score("\u306f\u3044", "\u306f\u3044") == local_ass_match_score(
        "\u306f\u3044",
        "\u306f\u3044",
    )
    assert _range_distance_ms(100, 200, 250, 300) == range_distance_ms(
        100,
        200,
        250,
        300,
    )
