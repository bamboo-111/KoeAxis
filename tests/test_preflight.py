from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.preflight import run_preflight


def test_preflight_reports_missing_media(tmp_path: Path) -> None:
    args = argparse.Namespace(
        media=str(tmp_path / "missing.mp3"),
        video=None,
        model_cache_dir=str(tmp_path / ".model-cache"),
        local_files_only=False,
        dtype="fp16",
        device="cuda",
        dry_run_check=False,
    )
    result = run_preflight(args, WorkPaths.from_workdir(tmp_path / "work"), "preflight")
    assert not result.ok
    assert any(issue.code == "missing_media" for issue in result.issues)


def test_preflight_reports_empty_local_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    args = argparse.Namespace(
        media=None,
        video=None,
        model_cache_dir=str(cache_dir),
        local_files_only=True,
        dtype="fp16",
        device="cpu",
        dry_run_check=False,
    )
    result = run_preflight(args, WorkPaths.from_workdir(tmp_path / "work"), "transcribe")
    assert any("or use --no-local-files-only" in issue.message for issue in result.issues)


def test_preflight_rejects_fp16_cpu(tmp_path: Path) -> None:
    args = argparse.Namespace(
        media=None,
        video=None,
        model_cache_dir=str(tmp_path / "cache"),
        local_files_only=False,
        dtype="fp16",
        device="cpu",
        dry_run_check=False,
    )
    result = run_preflight(args, WorkPaths.from_workdir(tmp_path / "work"), "transcribe")
    assert any(issue.code == "invalid_dtype_device" for issue in result.issues)
