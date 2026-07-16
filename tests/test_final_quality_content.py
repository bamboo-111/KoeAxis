from __future__ import annotations

from pathlib import Path

from qwen_asr import final_quality
from qwen_asr.final_quality_content import (
    best_pre_normalize_stage_text,
    manifest_item_text,
    normalize_export_content_check,
    normalized_stage_text,
    srt_stage_text,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_normalized_stage_text_reads_transcript_lists_and_subtitle_dicts(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.json"
    subtitles = tmp_path / "subtitles.json"
    write_json_atomic(transcript, [{"text": "\u306f\u3044"}, {"original_subtitle": "\u3046\u3093"}, "bad"])
    write_json_atomic(
        subtitles,
        {
            "1": {"original_subtitle": "\u306f\u3044"},
            "2": {"text": "\u3046\u3093"},
            "bad": "skip",
        },
    )

    assert normalized_stage_text(transcript) == "\u306f\u3044\u3046\u3093"
    assert normalized_stage_text(subtitles) == "\u306f\u3044\u3046\u3093"
    assert final_quality._normalized_stage_text(transcript) == "\u306f\u3044\u3046\u3093"


def test_manifest_item_text_prefers_stage_specific_fields() -> None:
    assert manifest_item_text("transcript", {"text": "a", "original_subtitle": "b"}) == "a"
    assert manifest_item_text("subtitle", {"text": "a", "original_subtitle": "b"}) == "b"
    assert final_quality._manifest_item_text("subtitle", {"text": "a"}) == "a"


def test_srt_stage_text_reads_subtitle_lines(tmp_path: Path) -> None:
    srt = tmp_path / "sample.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n\u306f\u3044\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\n\u3046\u3093\n",
        encoding="utf-8",
    )

    assert srt_stage_text(srt) == "\u306f\u3044\u3046\u3093"
    assert final_quality._srt_stage_text(srt) == "\u306f\u3044\u3046\u3093"


def test_best_pre_normalize_stage_text_prefers_proofread_realigned(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "\u5206\u5272"}})
    write_json_atomic(paths.mimo_proofread_manifest, {"1": {"original_subtitle": "\u4fee\u6b63"}})

    assert best_pre_normalize_stage_text(paths) == ("proofread-realigned", "\u4fee\u6b63")
    assert final_quality._best_pre_normalize_stage_text(paths) == ("proofread-realigned", "\u4fee\u6b63")


def test_normalize_export_content_check_skips_when_outputs_are_missing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    check = normalize_export_content_check(paths)

    assert check["status"] == "PASS"
    assert {item["name"] for item in check["checks"]} == {"normalize_content", "export_content"}
    assert all(item["skipped"] for item in check["checks"])


def test_normalize_export_content_check_passes_matching_normalize_and_export(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})
    write_json_atomic(paths.normalized_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})
    paths.subtitles_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n\u306f\u3044\n", encoding="utf-8")

    check = normalize_export_content_check(paths)

    assert check["status"] == "PASS"
    assert [item["status"] for item in check["checks"]] == ["PASS", "PASS"]
    assert check["checks"][0]["source_stage"] == "split"
    assert check["checks"][1]["source_stage"] == "normalized"


def test_normalize_export_content_check_fails_changed_normalize_text(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})
    write_json_atomic(paths.normalized_manifest, {"1": {"original_subtitle": "\u3044\u3044\u3048"}})

    check = normalize_export_content_check(paths)

    assert check["status"] == "FAIL"
    assert any(item["name"] == "normalize_content" and item["status"] == "FAIL" for item in check["checks"])


def test_normalize_export_content_check_fails_changed_export_text(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})
    paths.subtitles_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n\u3044\u3044\u3048\n", encoding="utf-8")

    check = normalize_export_content_check(paths)

    assert check["status"] == "FAIL"
    assert any(item["name"] == "export_content" and item["status"] == "FAIL" for item in check["checks"])
