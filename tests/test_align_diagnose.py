from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path

from tools import align_diagnose
from tools.align_diagnose import collect_audio_metrics, compute_quality_metrics, cmd_align_diagnose
from qwen_asr.models import AlignedToken, WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


def test_compute_quality_metrics_reports_all_failure_metrics() -> None:
    tokens = [
        AlignedToken("a", 10.0, 10.0),
        AlignedToken("b", 10.0, 10.0),
        AlignedToken("cdefghij", 10.0, 10.08),
    ]

    metrics = compute_quality_metrics(tokens, 10.0, 20.0)

    assert metrics["quality_error"] is not None
    assert metrics["covered_duration"] == 0.08
    assert metrics["coverage_ratio"] == 0.008
    assert metrics["zero_duration_count"] == 2
    assert metrics["positive_token_count"] == 1
    assert metrics["max_zero_run"] == 2
    assert metrics["local_max_cps"] > 35


def test_collect_audio_metrics_reports_silence_and_clipping(tmp_path: Path) -> None:
    wav_path = tmp_path / "probe.wav"
    _write_wav(wav_path, [0] * 1600 + [32767] * 1600)

    metrics = collect_audio_metrics(wav_path)

    assert metrics["sample_rate"] == 16000
    assert metrics["channels"] == 1
    assert metrics["silence_ratio_100ms"] == 0.5
    assert metrics["low_energy_ratio_100ms"] == 0.5
    assert metrics["clipping_ratio"] == 0.5
    assert metrics["leading_low_energy_seconds"] == 0.1


def test_align_diagnose_dry_run_writes_only_plan(tmp_path: Path) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_transcripts(work_paths, count=2)
    _seed_align_outputs(work_paths)

    args = _args(dry_run_plan=True, segments="segment_000001,segment_000002")
    status = cmd_align_diagnose(args, work_paths)

    assert status == 0
    diagnose_dirs = list((tmp_path / "diagnostics").glob("align-diagnose-*"))
    assert len(diagnose_dirs) == 1
    assert (diagnose_dirs[0] / "experiment_plan.tsv").exists()
    assert not (diagnose_dirs[0] / "diagnose_runs.jsonl").exists()
    assert read_json(work_paths.aligned_manifest, default=[]) == _aligned_payload()


def test_align_diagnose_keeps_raw_output_and_failed_tokens(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_transcripts(work_paths, count=1)
    _seed_align_outputs(work_paths)
    _write_wav(tmp_path / "audio-1.wav", [1200] * 16000)
    monkeypatch.setattr(align_diagnose, "QwenForcedAligner", FakeAligner)

    args = _args(dry_run_plan=False, segments="segment_000001")
    status = cmd_align_diagnose(args, work_paths)

    assert status == 0
    diagnose_dir = next((tmp_path / "diagnostics").glob("align-diagnose-*"))
    run_rows = [
        json.loads(line)
        for line in (diagnose_dir / "diagnose_runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(run_rows) == 1
    assert run_rows[0]["quality_failed"] is True
    assert run_rows[0]["raw_output_snapshot"]["path"].endswith(".json")
    assert Path(run_rows[0]["raw_output_snapshot"]["path"]).exists()
    token_rows = _read_tsv(Path(run_rows[0]["tokens_path"]))
    assert len(token_rows) == 4
    assert token_rows[0]["text"] == "a"
    assert read_json(work_paths.aligned_manifest, default=[]) == _aligned_payload()
    assert not work_paths.aligned_events_path.exists()
    assert not work_paths.aligned_checkpoint_path.exists()


def test_align_diagnose_can_write_asr_reference(tmp_path: Path, monkeypatch) -> None:
    work_paths = WorkPaths.from_workdir(tmp_path)
    _seed_transcripts(work_paths, count=1)
    _seed_align_outputs(work_paths)
    _write_wav(tmp_path / "audio-1.wav", [1200] * 16000)
    monkeypatch.setattr(align_diagnose, "QwenForcedAligner", FakeAligner)
    monkeypatch.setattr(align_diagnose, "QwenASRTranscriber", FakeTranscriber)

    args = _args(dry_run_plan=False, segments="segment_000001")
    args.with_asr_reference = True
    args.asr_model = "fake-asr"
    args.asr_language = None
    args.asr_max_new_tokens = 128
    args.asr_window_seconds = 0.5
    status = cmd_align_diagnose(args, work_paths)

    assert status == 0
    diagnose_dir = next((tmp_path / "diagnostics").glob("align-diagnose-*"))
    assert (diagnose_dir / "asr_reference.tsv").exists()
    run_rows = [
        json.loads(line)
        for line in (diagnose_dir / "diagnose_runs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reference = run_rows[0]["asr_reference"]
    assert reference["original"]["text"] == "text-1"
    assert reference["window_merged"]["window_count"] == 2
    assert reference["window_merged"]["similarity_to_manifest"] > 0
    assert list((diagnose_dir / "asr_windows").glob("*.wav"))


class FakeAligner:
    def __init__(self, *args, **kwargs) -> None:
        self._model = FakeModel()

    def load(self) -> None:
        return None

    def close(self) -> None:
        return None


class FakeModel:
    def align(self, **kwargs):
        return {
            "tokens": [
                {"text": "a", "start": 0.0, "end": 0.0},
                {"text": "b", "start": 0.0, "end": 0.0},
                {"text": "c", "start": 0.0, "end": 0.0},
                {"text": "d", "start": 0.0, "end": 0.08},
            ],
            "meta": {"model": "fake"},
        }


class FakeTranscriber:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def load(self) -> None:
        return None

    def run_segment(self, segment, cleanup: bool = True):
        from qwen_asr.models import TranscriptSegment

        text = "text-1" if "original" in segment.segment_id else "text"
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


def _args(*, dry_run_plan: bool, segments: str) -> argparse.Namespace:
    return argparse.Namespace(
        model="fake-aligner",
        segments=segments,
        sample_size=30,
        text_mode="asr",
        repeat=1,
        dry_run_plan=dry_run_plan,
        dtype="fp16",
        device="cpu",
        attn_implementation=None,
        model_cache_dir=None,
        local_files_only=True,
        with_asr_reference=False,
        asr_model="fake-asr",
        asr_language=None,
        asr_max_new_tokens=128,
        asr_window_seconds=3.0,
    )


def _seed_transcripts(work_paths: WorkPaths, *, count: int) -> None:
    rows = []
    for index in range(1, count + 1):
        rows.append(
            {
                "segment_id": f"segment_{index:06d}",
                "audio_path": f"audio-{index}.wav",
                "global_start_time": 10.0 * index,
                "global_end_time": 10.0 * index + 10.0,
                "text": f"text-{index}",
                "language": "Japanese",
                "status": "completed",
            }
        )
    write_json_atomic(work_paths.transcript_manifest, rows)


def _seed_align_outputs(work_paths: WorkPaths) -> None:
    write_json_atomic(work_paths.aligned_manifest, _aligned_payload())


def _aligned_payload() -> list[dict[str, object]]:
    return [
        {
            "segment_id": "segment_000001",
            "audio_path": "audio-1.wav",
            "global_start_time": 10.0,
            "global_end_time": 20.0,
            "text": "text-1",
            "language": "Japanese",
            "tokens": [],
            "status": "failed",
            "error": "alignment token timing unreliable: covered 0.080s of 10.000s",
        }
    ]


def _write_wav(path: Path, samples: list[int]) -> None:
    import struct

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
