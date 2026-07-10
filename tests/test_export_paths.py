from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.commands.stages import cmd_export
from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT
from qwen_asr.storage import write_json_atomic


def _args(media_path: Path, *, export_mode: str = "source", export_path: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(
        force=False,
        format="srt",
        source="transcript",
        max_subtitle_duration=6.0,
        min_subtitle_duration=1.0,
        max_chars_per_line_zh=18,
        max_chars_per_line_en=42,
        max_lines=2,
        pause_split_seconds=0.8,
        coarse_subtitles=False,
        optimizer_root=str(DEFAULT_OPTIMIZER_ROOT),
        export_mode=export_mode,
        export_path=export_path,
        media_path=str(media_path),
    )


def _write_transcript(paths: WorkPaths) -> None:
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "segment.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "hello",
                "status": "completed",
            }
        ],
    )


def test_export_writes_to_source_sibling_by_default(tmp_path: Path) -> None:
    media = tmp_path / "media" / "demo.mp3"
    media.parent.mkdir()
    media.write_bytes(b"media")
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "0001-demo")
    _write_transcript(paths)

    assert cmd_export(_args(media), paths) == 0

    assert paths.subtitles_srt.exists()
    assert media.with_suffix(".srt").exists()


def test_export_custom_directory_uses_source_name(tmp_path: Path) -> None:
    media = tmp_path / "demo.mp3"
    media.write_bytes(b"media")
    out_dir = tmp_path / "exports"
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "0001-demo")
    _write_transcript(paths)

    assert cmd_export(_args(media, export_mode="custom", export_path=str(out_dir)), paths) == 0

    assert (out_dir / "demo.srt").exists()


def test_export_custom_file_uses_exact_path(tmp_path: Path) -> None:
    media = tmp_path / "demo.mp3"
    media.write_bytes(b"media")
    out_file = tmp_path / "exports" / "custom-name.srt"
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "0001-demo")
    _write_transcript(paths)

    assert cmd_export(_args(media, export_mode="custom", export_path=str(out_file)), paths) == 0

    assert out_file.exists()
