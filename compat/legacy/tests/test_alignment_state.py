from __future__ import annotations

import json

from qwen_asr.alignment_state import (
    compatibility_status,
    derive_alignment_state,
    overlaps_music_region,
    read_music_region_evidence,
)
from qwen_asr.models import AlignedSegment, AlignedToken


def test_legacy_alignment_rows_derive_three_states() -> None:
    assert derive_alignment_state({"status": "failed", "tokens": []}) == "failed"
    assert derive_alignment_state({"status": "completed", "alignment_unit": "segment", "tokens": []}) == "completed_coarse"
    assert derive_alignment_state(
        {"status": "completed", "tokens": [{"start_time": 1.0, "end_time": 1.2}]}
    ) == "completed_exact"


def test_aligned_segment_persists_explicit_state_with_legacy_status() -> None:
    segment = AlignedSegment(
        segment_id="s1",
        audio_path="s1.wav",
        global_start_time=0.0,
        global_end_time=1.0,
        text="x",
        tokens=[],
        status="completed",
        alignment_state="completed_coarse",
    )

    assert segment.alignment_state == "completed_coarse"
    assert segment.status == "completed"
    assert compatibility_status("failed") == "failed"


def test_token_evidence_defaults_to_completed_exact() -> None:
    segment = AlignedSegment(
        segment_id="s1",
        audio_path="s1.wav",
        global_start_time=0.0,
        global_end_time=1.0,
        text="x",
        tokens=[AlignedToken(text="x", start_time=0.1, end_time=0.2)],
        status="completed",
    )

    assert segment.alignment_state == "completed_exact"


def test_music_region_evidence_is_generic_and_overlap_based(tmp_path) -> None:  # noqa: ANN001
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "evidence.json").write_text(
        json.dumps(
            {
                "intervals": {"opening": {"start_ms": 1000, "end_ms": 2000}},
                "subtitle_cues": {"oped": 3},
            }
        ),
        encoding="utf-8",
    )

    intervals, path, summary, error = read_music_region_evidence(tmp_path)

    assert error is None
    assert path is not None and path.endswith("evidence.json")
    assert summary["subtitle_cues"]["oped"] == 3
    assert overlaps_music_region(
        {"global_start_time": 1.5, "global_end_time": 2.5}, intervals
    ) == {"name": "opening", "start_ms": 1000, "end_ms": 2000}
