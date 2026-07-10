from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.commands import stages
from qwen_asr.models import AlignedSegment, WorkPaths
from qwen_asr.storage import write_json_atomic


def test_cmd_align_uses_cleanup_interval(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    transcripts = [
        {
            "segment_id": f"segment_{index:06d}",
            "audio_path": f"audio-{index}.wav",
            "global_start_time": float(index),
            "global_end_time": float(index + 1),
            "text": f"text-{index}",
            "language": "Japanese",
            "status": "completed",
        }
        for index in range(1, 6)
    ]
    write_json_atomic(work_paths.transcript_manifest, transcripts)

    cleanup_flags: list[bool] = []

    class FakeAligner:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load(self) -> None:
            return None

        def run_segment(self, transcript, cleanup: bool = True) -> AlignedSegment:
            cleanup_flags.append(cleanup)
            return AlignedSegment(
                segment_id=transcript.segment_id,
                audio_path=transcript.audio_path,
                global_start_time=transcript.global_start_time,
                global_end_time=transcript.global_end_time,
                text=transcript.text,
                language=transcript.language,
                tokens=[],
                status="completed",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenForcedAligner", FakeAligner)

    args = argparse.Namespace(
        force=False,
        resume=True,
        cleanup_interval=4,
        model="align-model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_align(args, work_paths)

    assert status == 0
    assert cleanup_flags == [False, False, False, True, False]


def test_cmd_align_defaults_to_project_model_cache(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    write_json_atomic(
        work_paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "text",
                "language": "Japanese",
                "status": "completed",
            }
        ],
    )
    seen_init: dict[str, object] = {}

    class FakeAligner:
        def __init__(self, *args, **kwargs) -> None:
            seen_init.update(kwargs)

        def load(self) -> None:
            return None

        def run_segment(self, transcript, cleanup: bool = True) -> AlignedSegment:
            return AlignedSegment(
                segment_id=transcript.segment_id,
                audio_path=transcript.audio_path,
                global_start_time=transcript.global_start_time,
                global_end_time=transcript.global_end_time,
                text=transcript.text,
                language=transcript.language,
                tokens=[],
                status="completed",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenForcedAligner", FakeAligner)

    args = argparse.Namespace(
        force=False,
        resume=True,
        cleanup_interval=4,
        model="align-model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_align(args, work_paths)

    assert status == 0
    assert seen_init["model_cache_dir"] == str(stages.DEFAULT_MODEL_CACHE_DIR)
    assert args.model_cache_dir == str(stages.DEFAULT_MODEL_CACHE_DIR)


def test_cmd_align_reports_empty_local_model_cache(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    cache_dir = tmp_path / "empty-cache"
    write_json_atomic(
        work_paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "text",
                "language": "Japanese",
                "status": "completed",
            }
        ],
    )

    args = argparse.Namespace(
        force=False,
        resume=True,
        cleanup_interval=4,
        model="align-model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        keep_raw_model_output=False,
        model_cache_dir=str(cache_dir),
        local_files_only=True,
        skip_preflight=False,
    )

    try:
        stages.cmd_align(args, work_paths)
    except RuntimeError as exc:
        assert "Model cache directory is empty" in str(exc)
    else:
        raise AssertionError("empty local cache should fail before model load")


def _seed_model_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "model-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / ".test-cache-marker").write_text("ok", encoding="ascii")
    monkeypatch.setattr(stages, "DEFAULT_MODEL_CACHE_DIR", cache_dir)
