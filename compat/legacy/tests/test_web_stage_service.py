from __future__ import annotations

import json
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic
from qwen_asr.web import stage_service


def test_stage_view_reports_order_status_counts_and_artifacts(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    paths.segments_manifest.write_text(json.dumps([{"segment_id": "s1"}]), encoding="utf-8")
    paths.audio_path.write_bytes(b"wav")
    paths.transcript_manifest.write_text(
        json.dumps([{"segment_id": "s1", "status": "completed", "text": "x"}]), encoding="utf-8"
    )
    paths.logs_dir.mkdir()
    (paths.logs_dir / "transcribe.log").write_text("done\n", encoding="utf-8")
    monkeypatch.setattr(stage_service, "load_job", lambda: {})

    view = stage_service.build_stage_view(paths)

    assert view["stages"][0]["name"] == "prepare"
    prepare = next(item for item in view["stages"] if item["name"] == "prepare")
    transcribe = next(item for item in view["stages"] if item["name"] == "transcribe")
    align = next(item for item in view["stages"] if item["name"] == "align")
    assert prepare["status"] == "complete"
    assert prepare["runnable"] is True
    assert transcribe["status"] == "complete"
    assert transcribe["output_count"] == 1
    assert transcribe["log"]["exists"] is True
    assert align["status"] == "pending"
    assert align["runnable"] is True
    assert any(item["kind"] == "transcript_manifest" for item in transcribe["artifacts"])
    proofread_realign = next(item for item in view["stages"] if item["name"] == "proofread-realign")
    assert proofread_realign["runnable"] is False
    assert proofread_realign["start_block_reason"] == "managed_by_pipeline"


def test_stage_view_exposes_current_job_duration(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    monkeypatch.setattr(
        stage_service,
        "load_job",
        lambda: {"stage": "align", "status": "running", "started_at": 10.0, "pid": 1},
    )
    monkeypatch.setattr(stage_service.time, "time", lambda: 12.5)

    view = stage_service.build_stage_view(paths)
    align = next(item for item in view["stages"] if item["name"] == "align")

    assert view["current_stage"] == "align"
    assert align["status"] == "running"
    assert align["duration_seconds"] == 2.5


def test_stage_view_surfaces_corrupt_progress(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    paths.progress_path.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(stage_service, "load_job", lambda: {})

    view = stage_service.build_stage_view(paths)

    assert view["progress"] == {}
    assert "JSONDecodeError" in view["progress_error"]


def test_stage_view_prioritizes_review_outdated_over_complete(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.translated_manifest, {"1": {"translated_subtitle": "done"}})
    write_json_atomic(paths.final_quality_report, {"status": "WARN"})
    write_json_atomic(paths.normalized_manifest, {"1": {"original_subtitle": "x"}})
    paths.normalized_srt.write_text("subtitle", encoding="utf-8")
    paths.subtitles_srt.write_text("subtitle", encoding="utf-8")
    write_json_atomic(
        paths.workdir / "drafts" / "web-review.json",
        {"schema_version": 1, "dirty": True, "cues": {}, "undo_stack": []},
    )
    monkeypatch.setattr(stage_service, "load_job", lambda: {})

    view = stage_service.build_stage_view(paths)

    for stage_name in ("quality-gate", "normalize", "export"):
        stage = next(item for item in view["stages"] if item["name"] == stage_name)
        assert stage["outdated"] is True
        assert stage["status"] == "outdated"
