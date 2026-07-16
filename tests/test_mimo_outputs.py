from __future__ import annotations

import json
from pathlib import Path

from qwen_asr.mimo_outputs import (
    stage2_reviewed_candidate_count,
    srt_time,
    to_srt,
    write_two_stage_outputs,
)


def test_write_two_stage_outputs_preserves_summary_schema(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    report = tmp_path / "report.json"
    stage1 = tmp_path / "stage1.json"
    stage2 = tmp_path / "stage2.json"
    srt = tmp_path / "out.srt"
    branch = {
        "1": {
            "start_time": 0,
            "end_time": 1000,
            "original_subtitle": "\u306f\u3044",
            "translated_subtitle": "\u597d\u7684",
        }
    }

    write_two_stage_outputs(
        manifest,
        report,
        stage1,
        stage2,
        srt,
        branch,
        [{"status": "completed", "suspect_ids": ["1"]}],
        [{"status": "completed", "id": "1", "applied_count": 1, "rejected_count": 2}],
        started=None,
        translated=branch,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "two-stage-nearby"
    assert payload["stage2_completed"] == 1
    assert payload["stage2_completed_batches"] == 1
    assert payload["audio_review_applied_count"] == 1
    assert payload["audio_review_rejected_count"] == 2
    assert "00:00:00,000 --> 00:00:01,000" in srt.read_text(encoding="utf-8")


def test_stage2_reviewed_candidate_count_dedupes_batch_target_ids() -> None:
    assert stage2_reviewed_candidate_count(
        [
            {"status": "completed", "target_ids": ["1", "2"]},
            {"status": "completed", "target_ids": ["2", "3"]},
            {"status": "completed", "id": "4"},
        ]
    ) == 4


def test_to_srt_orders_numeric_keys_and_formats_time() -> None:
    text = to_srt(
        {
            "2": {"start_time": 1000, "end_time": 2000, "original_subtitle": "\u3044\u3044", "translated_subtitle": ""},
            "1": {"start_time": 0, "end_time": 500, "original_subtitle": "\u306f\u3044", "translated_subtitle": "\u597d"},
        }
    )

    assert text.startswith("1\n00:00:00,000 --> 00:00:00,500\n\u306f\u3044\n\u597d")
    assert "2\n00:00:01,000 --> 00:00:02,000\n\u3044\u3044" in text
    assert srt_time(3_661_234) == "01:01:01,234"
