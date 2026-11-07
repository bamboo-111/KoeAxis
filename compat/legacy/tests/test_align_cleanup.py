from __future__ import annotations

import argparse
import wave
from pathlib import Path

from qwen_asr.commands import stages
from qwen_asr.models import AlignedSegment, AlignedToken, TranscriptSegment, WorkPaths
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
    assert seen_init["keep_failed_tokens"] is False
    assert args.model_cache_dir == str(stages.DEFAULT_MODEL_CACHE_DIR)


def test_cmd_align_returns_failed_when_any_eligible_segment_fails(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    write_json_atomic(
        work_paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio-1.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "text one",
                "language": "Japanese",
                "status": "completed",
            },
            {
                "segment_id": "segment_000002",
                "audio_path": "audio-2.wav",
                "global_start_time": 1.0,
                "global_end_time": 2.0,
                "text": "text two",
                "language": "Japanese",
                "status": "completed",
            },
        ],
    )

    class FakeAligner:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load(self) -> None:
            return None

        def run_segment(self, transcript, cleanup: bool = True) -> AlignedSegment:
            if transcript.segment_id == "segment_000001":
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
            return AlignedSegment(
                segment_id=transcript.segment_id,
                audio_path=transcript.audio_path,
                global_start_time=transcript.global_start_time,
                global_end_time=transcript.global_end_time,
                text=transcript.text,
                language=transcript.language,
                tokens=[],
                status="failed",
                error="alignment returned no tokens",
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
    payload = stages.read_json(work_paths.aligned_manifest, default=[])

    assert status == 1
    assert [item["status"] for item in payload] == ["completed", "failed"]


def test_cmd_align_diagnostics_mode_keeps_failed_evidence(tmp_path: Path, monkeypatch) -> None:
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
                status="failed",
                error="alignment returned no tokens",
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
        align_diagnostics_mode="capture-failed",
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_align(args, work_paths)

    assert status == 1
    assert seen_init["keep_raw_model_output"] is True
    assert seen_init["keep_failed_tokens"] is True


def test_cmd_align_asr_short_window_fallback_writes_completed_alignment(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "segment.wav"
    _write_silent_wav(audio_path, duration_seconds=2.0)
    write_json_atomic(
        work_paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": str(audio_path),
                "global_start_time": 10.0,
                "global_end_time": 12.0,
                "text": "original",
                "language": "Japanese",
                "status": "completed",
            }
        ],
    )

    class FakeAligner:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load(self) -> None:
            return None

        def run_segment(self, transcript, cleanup: bool = True) -> AlignedSegment:
            if transcript.segment_id == "segment_000001":
                return AlignedSegment(
                    segment_id=transcript.segment_id,
                    audio_path=transcript.audio_path,
                    global_start_time=transcript.global_start_time,
                    global_end_time=transcript.global_end_time,
                    text=transcript.text,
                    language=transcript.language,
                    tokens=[],
                    raw_model_output={"raw": "failed"},
                    status="failed",
                    error="alignment returned no tokens",
                )
            return AlignedSegment(
                segment_id=transcript.segment_id,
                audio_path=transcript.audio_path,
                global_start_time=transcript.global_start_time,
                global_end_time=transcript.global_end_time,
                text=transcript.text,
                language=transcript.language,
                tokens=[
                    AlignedToken(
                        text=transcript.text,
                        start_time=transcript.global_start_time,
                        end_time=transcript.global_end_time,
                    )
                ],
                status="completed",
            )

        def close(self) -> None:
            return None

    class FakeTranscriber:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load(self) -> None:
            return None

        def run_segment(self, segment, cleanup: bool = True) -> TranscriptSegment:
            text = "orig" if segment.segment_id.endswith("w01") else "inal"
            return TranscriptSegment(
                segment_id=segment.segment_id,
                audio_path=segment.audio_path,
                global_start_time=segment.global_start_time,
                global_end_time=segment.global_end_time,
                text=text,
                language="Japanese",
                status="completed",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenForcedAligner", FakeAligner)
    monkeypatch.setattr(stages, "QwenASRTranscriber", FakeTranscriber)

    args = argparse.Namespace(
        force=False,
        resume=True,
        cleanup_interval=4,
        model="align-model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        keep_raw_model_output=False,
        align_diagnostics_mode="capture-failed",
        align_fallback="asr-short-window",
        align_fallback_window_seconds=1.0,
        asr_reference_model="asr-model",
        asr_reference_max_new_tokens=128,
        asr_reference_language="Japanese",
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_align(args, work_paths)
    payload = stages.read_json(work_paths.aligned_manifest, default=[])

    assert status == 0
    assert payload[0]["status"] == "completed"
    assert payload[0]["text"] == "original"
    assert payload[0]["raw_model_output"]["align_fallback"]["fallback"] == "asr-short-window"
    assert payload[0]["raw_model_output"]["align_fallback"]["completed_window_alignments"] == 2
    assert len(payload[0]["tokens"]) == 2


def test_cmd_align_asr_short_window_fallback_rejects_changed_transcript(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_model_cache(tmp_path, monkeypatch)
    audio_path = tmp_path / "segment.wav"
    _write_silent_wav(audio_path, duration_seconds=2.0)
    write_json_atomic(
        work_paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": str(audio_path),
                "global_start_time": 10.0,
                "global_end_time": 12.0,
                "text": "original sentence",
                "language": "Japanese",
                "status": "completed",
            }
        ],
    )

    class FakeAligner:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load(self) -> None:
            return None

        def run_segment(self, transcript, cleanup: bool = True) -> AlignedSegment:
            if transcript.segment_id == "segment_000001":
                return AlignedSegment(
                    segment_id=transcript.segment_id,
                    audio_path=transcript.audio_path,
                    global_start_time=transcript.global_start_time,
                    global_end_time=transcript.global_end_time,
                    text=transcript.text,
                    language=transcript.language,
                    tokens=[],
                    status="failed",
                    error="alignment returned no tokens",
                )
            return AlignedSegment(
                segment_id=transcript.segment_id,
                audio_path=transcript.audio_path,
                global_start_time=transcript.global_start_time,
                global_end_time=transcript.global_end_time,
                text=transcript.text,
                language=transcript.language,
                tokens=[
                    AlignedToken(
                        text=transcript.text,
                        start_time=transcript.global_start_time,
                        end_time=transcript.global_end_time,
                    )
                ],
                status="completed",
            )

        def close(self) -> None:
            return None

    class FakeTranscriber:
        def __init__(self, *args, **kwargs) -> None:
            return None

        def load(self) -> None:
            return None

        def run_segment(self, segment, cleanup: bool = True) -> TranscriptSegment:
            return TranscriptSegment(
                segment_id=segment.segment_id,
                audio_path=segment.audio_path,
                global_start_time=segment.global_start_time,
                global_end_time=segment.global_end_time,
                text="unrelated",
                language="Japanese",
                status="completed",
            )

        def close(self) -> None:
            return None

    monkeypatch.setattr(stages, "QwenForcedAligner", FakeAligner)
    monkeypatch.setattr(stages, "QwenASRTranscriber", FakeTranscriber)

    args = argparse.Namespace(
        force=False,
        resume=True,
        cleanup_interval=4,
        model="align-model",
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        keep_raw_model_output=False,
        align_diagnostics_mode="capture-failed",
        align_fallback="asr-short-window",
        align_fallback_window_seconds=1.0,
        asr_reference_model="asr-model",
        asr_reference_max_new_tokens=128,
        asr_reference_language="Japanese",
        model_cache_dir=None,
        local_files_only=True,
        skip_preflight=True,
    )

    status = stages.cmd_align(args, work_paths)
    payload = stages.read_json(work_paths.aligned_manifest, default=[])

    assert status == 1
    assert payload[0]["status"] == "failed"
    assert payload[0]["text"] == "original sentence"
    assert "changed transcript content" in payload[0]["error"]


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


def _write_silent_wav(path: Path, *, duration_seconds: float) -> None:
    sample_rate = 16000
    frames = int(sample_rate * duration_seconds)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)
