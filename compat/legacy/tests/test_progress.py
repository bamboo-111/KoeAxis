from __future__ import annotations

from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress, write_progress
from qwen_asr.stages import StageStatus


def test_progress_write_read(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    written = write_progress(
        paths,
        stage="translate",
        status="running",
        done=3,
        total=8,
        current="batch 1",
        summary="3/8 translated subtitles",
    )

    loaded = read_progress(paths)
    assert loaded == written
    assert loaded["stage"] == "translate"
    assert loaded["done"] == 3
    assert loaded["updated_at"].endswith("Z")
    assert set(loaded) == {"stage", "status", "done", "total", "current", "updated_at", "summary"}


def test_progress_defaults_are_stable(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    written = write_progress(paths, stage="prepare", status="running")

    assert written["done"] is None
    assert written["total"] is None
    assert written["current"] == ""
    assert written["summary"] == ""


def test_progress_accepts_stage_status_enum(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    written = write_progress(paths, stage="export", status=StageStatus.COMPLETED)

    assert written["status"] == "completed"
    assert read_progress(paths)["status"] == "completed"
