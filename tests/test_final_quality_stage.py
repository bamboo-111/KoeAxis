from __future__ import annotations

from pathlib import Path

from qwen_asr import final_quality
from qwen_asr.final_quality_stage import has_checkpoint_artifact, stage_checkpoint_check
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_stage_checkpoint_passes_empty_workdir(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    check = stage_checkpoint_check(paths)

    assert check["status"] == "PASS"
    assert final_quality._stage_checkpoint_check(paths) == check


def test_has_checkpoint_artifact_detects_stage_outputs(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.transcript_checkpoint_path, {"done": 0})
    write_json_atomic(paths.aligned_events_path, [])
    write_json_atomic(paths.split_manifest, {})
    write_json_atomic(paths.translated_srt, {"not": "real srt"})
    write_json_atomic(paths.mimo_proofread_report, {})
    write_json_atomic(paths.workdir / "reports" / "proofread_realign.json", {})
    write_json_atomic(paths.normalized_srt, {})
    paths.subtitles_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    assert has_checkpoint_artifact(paths, "transcribe")
    assert has_checkpoint_artifact(paths, "align")
    assert has_checkpoint_artifact(paths, "split")
    assert has_checkpoint_artifact(paths, "translate")
    assert has_checkpoint_artifact(paths, "mimo-proofread")
    assert has_checkpoint_artifact(paths, "proofread-realign")
    assert has_checkpoint_artifact(paths, "normalize")
    assert has_checkpoint_artifact(paths, "export")
    assert not has_checkpoint_artifact(paths, "unknown")
    assert final_quality._has_checkpoint_artifact(paths, "export")


def test_stage_checkpoint_fails_for_incomplete_split_artifact(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {})

    check = stage_checkpoint_check(paths)

    assert check["status"] == "FAIL"
    assert "split" in check["stages"]


def test_stage_checkpoint_fails_for_incomplete_translate_artifact(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""}})
    write_json_atomic(paths.translated_manifest, {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""}})

    check = stage_checkpoint_check(paths)

    assert check["status"] == "FAIL"
    assert "translate" in check["stages"]
