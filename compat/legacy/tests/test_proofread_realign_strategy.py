from __future__ import annotations

from pathlib import Path

from qwen_asr.proofread_realign import _safe_id as legacy_safe_id
from qwen_asr.proofread_realign import _timing_candidate_guard as legacy_timing_candidate_guard
from qwen_asr.proofread_realign_strategy import (
    clamp_display_range_to_original_window,
    clamp_timing_candidate_to_neighbors,
    fallback_original_timing,
    mfa_content_score,
    safe_id,
    should_keep_mixed_language_original_timing,
    timing_candidate_guard,
)


def test_fallback_original_timing_updates_manifest_item_and_row(tmp_path: Path) -> None:
    item = {"needs_realign": True, "realign_error": "old"}
    clip = tmp_path / "clip.wav"
    mfa_row = {"mfa_status": "rejected", "reason": "mismatch", "mfa_result": {"score": 0.1}}

    row = fallback_original_timing(
        item,
        "7",
        1000,
        1600,
        clip,
        "fallback-used",
        mfa_row=mfa_row,
        method="mixed-language-original-timing",
    )

    assert item["start_time"] == 1000
    assert item["end_time"] == 1600
    assert item["needs_realign"] is False
    assert item["realign_status"] == "completed"
    assert item["realign_method"] == "mixed-language-original-timing"
    assert "realign_error" not in item
    assert row["status"] == "fallback"
    assert row["mfa_status"] == "rejected"


def test_mixed_language_original_timing_requires_long_text_and_duration() -> None:
    item = {
        "original_subtitle": "abcdefghijklmnopqrstuvwxyz extra words "
        + "\u3042\u3044\u3046\u3048\u304a\u304b\u304d\u304f",
        "start_time": 1000,
        "end_time": 6000,
    }

    assert should_keep_mixed_language_original_timing(item) is True
    assert should_keep_mixed_language_original_timing({**item, "end_time": 3000}) is False


def test_timing_guard_and_clamp_keep_legacy_alias() -> None:
    manifest = {
        "1": {"start_time": 1000, "end_time": 2000},
        "2": {"start_time": 2200, "end_time": 2600},
        "3": {"start_time": 2800, "end_time": 3600},
    }

    guard = timing_candidate_guard(
        manifest,
        subtitle_id="2",
        start_ms=1900,
        end_ms=3000,
        clip_start_ms=1500,
        clip_end_ms=3300,
    )
    clamped = clamp_timing_candidate_to_neighbors(
        manifest,
        subtitle_id="2",
        start_ms=1900,
        end_ms=3000,
        clip_start_ms=1500,
        clip_end_ms=3300,
        timing_guard=guard,
    )

    assert guard["accepted"] is False
    assert guard["reason"] == "severe-neighbor-overlap"
    assert clamped["accepted"] is True
    assert clamped["reason"] == "timing-clamped-to-neighbors"
    assert clamped["start_ms"] == 2000
    assert clamped["end_ms"] == 2800
    assert legacy_timing_candidate_guard(
        manifest,
        subtitle_id="2",
        start_ms=1900,
        end_ms=3000,
        clip_start_ms=1500,
        clip_end_ms=3300,
    ) == guard


def test_clamp_display_range_to_original_window_accepts_only_usable_original_window() -> None:
    result = clamp_display_range_to_original_window(
        1000,
        12000,
        original_start_ms=3000,
        original_end_ms=7000,
    )

    assert result["accepted"] is True
    assert result["reason"] == "display-clamped-to-original-window"
    assert result["start_ms"] == 3000
    assert result["end_ms"] == 7000


def test_content_score_and_safe_id_compatibility() -> None:
    assert mfa_content_score("\u306f\u3044\u3002", "\u306f\u3044") == 1.0
    assert safe_id("a/b:c") == "a_b_c"
    assert safe_id("") == "item"
    assert legacy_safe_id("") == "item"
