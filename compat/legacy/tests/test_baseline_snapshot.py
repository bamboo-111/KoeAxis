from __future__ import annotations

import hashlib
from pathlib import Path

from tools.baseline_snapshot import build_baseline_snapshot, render_baseline_snapshot_markdown
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_baseline_snapshot_records_hashes_and_stage_metrics(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "\u306f\u3044",
                "status": "completed",
            },
            {
                "segment_id": "segment_000002",
                "global_start_time": 1.0,
                "global_end_time": 2.0,
                "text": "\u72ec\u81ea",
                "status": "completed",
            },
        ],
    )
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044"},
            "2": {"start_time": 1000, "end_time": 2000, "original_subtitle": "\u72ec\u81ea"},
        },
    )

    report = build_baseline_snapshot(paths)

    transcript = next(item for item in report["stages"] if item["stage"] == "transcript")
    split = next(item for item in report["stages"] if item["stage"] == "split")
    proofread_realigned = next(item for item in report["stages"] if item["stage"] == "proofread-realigned")
    expected_hash = hashlib.sha256(paths.transcript_manifest.read_bytes()).hexdigest().upper()
    assert transcript["sha256"] == expected_hash
    assert transcript["item_count"] == 2
    assert transcript["normalized_japanese_chars"] == 4
    assert transcript["short_response_count"] == 1
    assert transcript["time"]["valid"] is True
    assert split["unique_text_count"] == 2
    assert proofread_realigned["exists"] is False


def test_baseline_snapshot_marks_invalid_time_and_renders_markdown(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"start_time": 1000, "end_time": 900, "original_subtitle": "\u306f\u3044"},
        },
    )

    report = build_baseline_snapshot(paths)
    split = next(item for item in report["stages"] if item["stage"] == "split")
    markdown = render_baseline_snapshot_markdown(report)

    assert split["time"]["valid"] is False
    assert split["time"]["invalid_count"] == 1
    assert "\u7a33\u5b9a\u57fa\u7ebf\u5feb\u7167" in markdown
    assert "split" in markdown


def test_baseline_snapshot_srt_uses_first_subtitle_line_for_export_metrics(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.subtitles_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.subtitles_srt.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "\u306f\u3044\n"
        "\u4e2d\u6587\u8bd1\u6587\n",
        encoding="utf-8",
    )

    report = build_baseline_snapshot(paths)
    export = next(item for item in report["stages"] if item["stage"] == "export")

    assert export["item_count"] == 1
    assert export["normalized_japanese_chars"] == 2
