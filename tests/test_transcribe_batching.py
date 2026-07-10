from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.asr import ASRBatchOOMError
from qwen_asr.batching import BatchPlanner
from qwen_asr.commands import stages
from qwen_asr.models import TranscriptSegment, WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


def test_cmd_transcribe_reduces_batch_size_after_oom(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    segments = [
        {
            "segment_id": f"segment_{index:06d}",
            "audio_path": f"audio-{index}.wav",
            "source_audio_path": "source.wav",
            "global_start_time": float(index),
            "global_end_time": float(index + 1),
            "duration": 1.0,
            "status": "prepared",
        }
        for index in range(1, 9)
    ]
    write_json_atomic(work_paths.segments_manifest, segments)

    calls: list[int] = []

    class FakeTranscriber:
        def __init__(self, *args, batch_size: int = 1, **kwargs) -> None:
            self.batch_size = batch_size

        def load(self) -> None:
            return None

        def run_batch(self, batch) -> list[TranscriptSegment]:
            calls.append(len(batch))
            if len(batch) > 3:
                raise ASRBatchOOMError("CUDA out of memory")
            return [
                TranscriptSegment(
                    segment_id=item.segment_id,
                    audio_path=item.audio_path,
                    global_start_time=item.global_start_time,
                    global_end_time=item.global_end_time,
                    text=f"text-{item.segment_id}",
                    language="Japanese",
                    status="completed",
                )
                for item in batch
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenASRTranscriber", FakeTranscriber)

    args = argparse.Namespace(
        batch_size=5,
        force=False,
        resume=True,
        model="model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        max_new_tokens=512,
        language=None,
        batch_mode="fixed",
        target_batch_audio_seconds=None,
        single_long_segment_threshold=90.0,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_transcribe(args, work_paths)

    assert status == 0
    assert calls == [5, 4, 3, 3, 2]


def test_webui_transcribe_command_uses_default_batch_size() -> None:
    command = stages  # placeholder to keep import ordering local
    del command
    from qwen_asr.web.commands import build_command

    built = build_command(
        {
            "stage": "transcribe",
            "workdir": "work-test",
            "asr_model": "asr",
            "dtype": "fp16",
            "device": "cuda",
        }
    )

    assert built[built.index("--batch-mode") + 1] == "adaptive"
    assert "--batch-size" not in built
    assert "--single-long-segment-threshold" not in built


def test_adaptive_batch_planner_groups_similar_durations() -> None:
    segments = [
        _segment("segment_000001", 12.0),
        _segment("segment_000002", 13.0),
        _segment("segment_000003", 48.0),
        _segment("segment_000004", 49.0),
        _segment("segment_000005", 92.0),
    ]

    planner = BatchPlanner(
        segments,
        mode="adaptive",
        max_batch_items=3,
        target_audio_seconds=40.0,
    )

    batch1 = planner.next_batch()
    assert batch1 is not None
    assert [item.segment_id for item in batch1.segments] == ["segment_000001", "segment_000002"]
    planner.mark_success(batch1)

    batch2 = planner.next_batch()
    assert batch2 is not None
    assert [item.segment_id for item in batch2.segments] == ["segment_000003"]


def test_adaptive_batch_planner_reduces_limits_after_oom() -> None:
    segments = [
        _segment("segment_000001", 55.0),
        _segment("segment_000002", 57.0),
        _segment("segment_000003", 59.0),
    ]

    planner = BatchPlanner(
        segments,
        mode="adaptive",
        max_batch_items=3,
        target_audio_seconds=180.0,
    )

    batch = planner.next_batch()
    assert batch is not None
    planner.report_oom(batch)

    assert planner.current_max_batch_items == 2
    assert planner.current_target_audio_seconds == 144.0


def test_adaptive_batch_planner_forces_long_segment_to_singleton() -> None:
    segments = [
        _segment("segment_000001", 12.0),
        _segment("segment_000002", 15.0),
        _segment("segment_000003", 95.0),
        _segment("segment_000004", 96.0),
    ]

    planner = BatchPlanner(
        segments,
        mode="adaptive",
        max_batch_items=3,
        target_audio_seconds=180.0,
        single_long_segment_threshold=90.0,
    )

    batch1 = planner.next_batch()
    assert batch1 is not None
    assert [item.segment_id for item in batch1.segments] == ["segment_000001", "segment_000002"]
    planner.mark_success(batch1)

    batch2 = planner.next_batch()
    assert batch2 is not None
    assert [item.segment_id for item in batch2.segments] == ["segment_000003"]
    assert batch2.singleton_reason == "long_segment_threshold"
    assert batch2.duration_spread_ratio == 1.0


def test_cmd_transcribe_writes_profile_report_when_enabled(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    segments = [
        {
            "segment_id": "segment_000001",
            "audio_path": "audio-1.wav",
            "source_audio_path": "source.wav",
            "global_start_time": 0.0,
            "global_end_time": 12.0,
            "duration": 12.0,
            "status": "prepared",
        },
        {
            "segment_id": "segment_000002",
            "audio_path": "audio-2.wav",
            "source_audio_path": "source.wav",
            "global_start_time": 12.0,
            "global_end_time": 24.0,
            "duration": 12.0,
            "status": "prepared",
        },
    ]
    write_json_atomic(work_paths.segments_manifest, segments)

    class FakeTranscriber:
        def __init__(self, *args, batch_size: int = 1, **kwargs) -> None:
            self.batch_size = batch_size

        def load(self) -> None:
            return None

        def run_batch(self, batch) -> list[TranscriptSegment]:
            return [
                TranscriptSegment(
                    segment_id=item.segment_id,
                    audio_path=item.audio_path,
                    global_start_time=item.global_start_time,
                    global_end_time=item.global_end_time,
                    text=f"text-{item.segment_id}",
                    language="Japanese",
                    status="completed",
                )
                for item in batch
            ]

        def consume_last_batch_memory_probes(self) -> list[dict[str, object]]:
            return [
                {"phase": "before_inference", "batch_size": 2, "segment_ids": ["segment_000001", "segment_000002"]},
                {"phase": "after_inference", "batch_size": 2, "segment_ids": ["segment_000001", "segment_000002"]},
                {"phase": "after_cleanup", "batch_size": 2, "segment_ids": ["segment_000001", "segment_000002"]},
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenASRTranscriber", FakeTranscriber)

    args = argparse.Namespace(
        batch_size=5,
        force=False,
        resume=True,
        profile_batches=True,
        model="model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        max_new_tokens=512,
        language=None,
        batch_mode="adaptive",
        target_batch_audio_seconds=60.0,
        single_long_segment_threshold=90.0,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_transcribe(args, work_paths)

    assert status == 0
    profile = read_json(work_paths.transcribe_profile_path)
    assert profile["summary"]["completed_batch_count"] == 1
    assert profile["summary"]["oom_retry_count"] == 0
    assert profile["batches"][0]["memory_probes"][0]["phase"] == "before_inference"
    assert profile["recommendation"]["next_run"]["batch_mode"] == "adaptive"
    assert profile["recommendation"]["next_run"]["batch_size"] == 2
    assert profile["recommendation"]["signals"]["oom_retry_count"] == 0


def test_auto_select_transcribe_batch_defaults_prefers_long_form_profile() -> None:
    selection = stages._auto_select_transcribe_batch_defaults(
        [
            _segment("segment_000001", 112.0),
            _segment("segment_000002", 108.0),
            _segment("segment_000003", 95.0),
            _segment("segment_000004", 88.0),
            _segment("segment_000005", 76.0),
        ]
    )

    assert selection["profile"] == "long_form"
    assert selection["batch_size"] == 3
    assert selection["target_batch_audio_seconds"] == 300.0
    assert selection["single_long_segment_threshold"] == 90.0


def test_cmd_transcribe_auto_resolves_batch_defaults(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    segments = [
        {
            "segment_id": "segment_000001",
            "audio_path": "audio-1.wav",
            "source_audio_path": "source.wav",
            "global_start_time": 0.0,
            "global_end_time": 112.0,
            "duration": 112.0,
            "status": "prepared",
        },
        {
            "segment_id": "segment_000002",
            "audio_path": "audio-2.wav",
            "source_audio_path": "source.wav",
            "global_start_time": 112.0,
            "global_end_time": 220.0,
            "duration": 108.0,
            "status": "prepared",
        },
    ]
    write_json_atomic(work_paths.segments_manifest, segments)

    seen_init: dict[str, object] = {}

    class FakeTranscriber:
        def __init__(self, *args, batch_size: int = 1, **kwargs) -> None:
            seen_init["batch_size"] = batch_size

        def load(self) -> None:
            return None

        def run_batch(self, batch) -> list[TranscriptSegment]:
            return [
                TranscriptSegment(
                    segment_id=item.segment_id,
                    audio_path=item.audio_path,
                    global_start_time=item.global_start_time,
                    global_end_time=item.global_end_time,
                    text=f"text-{item.segment_id}",
                    language="Japanese",
                    status="completed",
                )
                for item in batch
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenASRTranscriber", FakeTranscriber)

    args = argparse.Namespace(
        batch_size=None,
        force=False,
        resume=True,
        profile_batches=False,
        model="model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        max_new_tokens=512,
        language=None,
        batch_mode="adaptive",
        target_batch_audio_seconds=None,
        single_long_segment_threshold=None,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_transcribe(args, work_paths)

    assert status == 0
    assert args.batch_size == 3
    assert args.target_batch_audio_seconds == 300.0
    assert args.single_long_segment_threshold == 90.0
    assert seen_init["batch_size"] == 3


def test_cmd_transcribe_defaults_to_project_model_cache(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    write_json_atomic(
        work_paths.segments_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio-1.wav",
                "source_audio_path": "source.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "duration": 1.0,
                "status": "prepared",
            }
        ],
    )
    seen_init: dict[str, object] = {}

    class FakeTranscriber:
        def __init__(self, *args, **kwargs) -> None:
            seen_init.update(kwargs)

        def load(self) -> None:
            return None

        def run_batch(self, batch) -> list[TranscriptSegment]:
            return [
                TranscriptSegment(
                    segment_id=item.segment_id,
                    audio_path=item.audio_path,
                    global_start_time=item.global_start_time,
                    global_end_time=item.global_end_time,
                    text=f"text-{item.segment_id}",
                    language="Japanese",
                    status="completed",
                )
                for item in batch
            ]

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenASRTranscriber", FakeTranscriber)

    args = argparse.Namespace(
        batch_size=5,
        force=False,
        resume=True,
        profile_batches=False,
        model="model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        max_new_tokens=512,
        language=None,
        batch_mode="adaptive",
        target_batch_audio_seconds=None,
        single_long_segment_threshold=None,
        keep_raw_model_output=False,
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_transcribe(args, work_paths)

    assert status == 0
    assert seen_init["model_cache_dir"] == str(stages.DEFAULT_MODEL_CACHE_DIR)
    assert args.model_cache_dir == str(stages.DEFAULT_MODEL_CACHE_DIR)


def test_cmd_transcribe_reports_empty_local_model_cache(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    cache_dir = tmp_path / "empty-cache"
    write_json_atomic(
        work_paths.segments_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio-1.wav",
                "source_audio_path": "source.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "duration": 1.0,
                "status": "prepared",
            }
        ],
    )

    args = argparse.Namespace(
        batch_size=5,
        force=False,
        resume=True,
        profile_batches=False,
        model="model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        max_new_tokens=512,
        language=None,
        batch_mode="adaptive",
        target_batch_audio_seconds=None,
        single_long_segment_threshold=None,
        keep_raw_model_output=False,
        model_cache_dir=str(cache_dir),
        local_files_only=True,
        skip_preflight=False,
    )

    try:
        stages.cmd_transcribe(args, work_paths)
    except RuntimeError as exc:
        assert "Model cache directory is empty" in str(exc)
    else:
        raise AssertionError("empty local cache should fail before model load")


def _seed_model_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "model-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / ".test-cache-marker").write_text("ok", encoding="ascii")
    monkeypatch.setattr(stages, "DEFAULT_MODEL_CACHE_DIR", cache_dir)


def _segment(segment_id: str, duration: float):
    return stages.AudioSegment(
        segment_id=segment_id,
        audio_path=f"{segment_id}.wav",
        source_audio_path="source.wav",
        global_start_time=0.0,
        global_end_time=duration,
        duration=duration,
        status="prepared",
    )
