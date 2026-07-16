from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from qwen_asr.commands import stages
from qwen_asr.commands.transcribe_profile import (
    auto_select_transcribe_batch_defaults,
    build_transcribe_recommendation,
    consume_batch_memory_probes,
    prepare_model_cache_dir,
    resolve_model_cache_dir,
    resolve_transcribe_batch_defaults,
    write_transcribe_profile,
)
from qwen_asr.models import AudioSegment, WorkPaths
from qwen_asr.storage import read_json


def _segment(segment_id: str, duration: float) -> AudioSegment:
    return AudioSegment(
        segment_id=segment_id,
        audio_path=f"{segment_id}.wav",
        source_audio_path="source.wav",
        global_start_time=0.0,
        global_end_time=duration,
        duration=duration,
    )


def test_auto_select_transcribe_batch_defaults_prefers_long_form_profile() -> None:
    selection = auto_select_transcribe_batch_defaults(
        [_segment("a", 100.0), _segment("b", 120.0), _segment("c", 140.0)]
    )

    assert selection["profile"] == "long_form"
    assert selection["batch_size"] == 3
    assert stages._auto_select_transcribe_batch_defaults([_segment("a", 100.0)])["profile"] == "long_form"


def test_resolve_transcribe_batch_defaults_mutates_auto_defaults() -> None:
    args = argparse.Namespace(
        batch_mode="adaptive",
        batch_size=None,
        target_batch_audio_seconds=None,
        single_long_segment_threshold=None,
    )

    resolved = resolve_transcribe_batch_defaults(args, [_segment("a", 10.0), _segment("b", 20.0)])

    assert resolved["profile"] == "short_form"
    assert args.batch_size == 5
    assert args.target_batch_audio_seconds == 220.0
    assert args.single_long_segment_threshold == 110.0
    assert stages._resolve_transcribe_batch_defaults(args, [_segment("a", 10.0)])["profile"] == "short_form"


def test_resolve_transcribe_batch_defaults_keeps_fixed_defaults() -> None:
    args = argparse.Namespace(
        batch_mode="fixed",
        batch_size=None,
        target_batch_audio_seconds=None,
        single_long_segment_threshold=None,
    )

    resolved = resolve_transcribe_batch_defaults(args, [_segment("a", 10.0)])

    assert resolved["profile"] == "fixed"
    assert args.batch_size == 5
    assert args.target_batch_audio_seconds is None


def test_model_cache_dir_defaults_and_local_only_empty_guard(tmp_path: Path) -> None:
    args = argparse.Namespace(model_cache_dir=str(tmp_path / "cache"))

    assert resolve_model_cache_dir(args) == str(tmp_path / "cache")
    prepare_model_cache_dir(str(tmp_path / "cache"), local_files_only=False)

    with pytest.raises(RuntimeError, match="Model cache directory is empty"):
        prepare_model_cache_dir(str(tmp_path / "empty"), local_files_only=True)

    stages._prepare_model_cache_dir(str(tmp_path / "nonempty"), local_files_only=False)


def test_consume_batch_memory_probes_uses_optional_hook() -> None:
    class Transcriber:
        def consume_last_batch_memory_probes(self) -> list[dict[str, object]]:
            return [{"device": "cuda:0"}]

    assert consume_batch_memory_probes(Transcriber()) == [{"device": "cuda:0"}]
    assert consume_batch_memory_probes(object()) == []
    assert stages._consume_batch_memory_probes(Transcriber()) == [{"device": "cuda:0"}]


def test_write_transcribe_profile_records_recommendation(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    args = argparse.Namespace(
        profile_batches=True,
        batch_mode="adaptive",
        batch_size=4,
        target_batch_audio_seconds=200.0,
        single_long_segment_threshold=90.0,
    )
    segments = [_segment("a", 10.0), _segment("b", 95.0)]
    reports = [
        {
            "status": "completed",
            "batch_size": 2,
            "total_duration": 105.0,
            "duration_spread_ratio": 9.5,
            "max_duration": 95.0,
            "singleton_reason": "long_segment_threshold",
        },
        {
            "status": "oom_retry",
            "batch_size": 4,
            "total_duration": 200.0,
            "duration_spread_ratio": 1.0,
            "max_duration": 100.0,
        },
    ]

    write_transcribe_profile(paths, args, segments, reports, {"profile": "mixed"})
    payload = read_json(paths.transcribe_profile_path)

    assert payload["summary"]["oom_retry_count"] == 1
    assert payload["summary"]["singleton_reasons"] == {"long_segment_threshold": 1}
    assert payload["recommendation"]["next_run"]["batch_size"] == 3


def test_build_transcribe_recommendation_without_reports_keeps_configured_values() -> None:
    args = argparse.Namespace(
        batch_mode="adaptive",
        batch_size=4,
        target_batch_audio_seconds=None,
        single_long_segment_threshold=90.0,
    )

    recommendation = build_transcribe_recommendation(args, [], [])

    assert recommendation["next_run"]["batch_size"] == 4
    assert recommendation["next_run"]["target_batch_audio_seconds"] is None
