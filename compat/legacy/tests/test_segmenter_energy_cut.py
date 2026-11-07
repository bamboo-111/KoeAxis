from __future__ import annotations

from pathlib import Path

from qwen_asr.models import SpeechRegion
from qwen_asr.segmenter import EnergyProfile, SegmenterConfig, build_segments

FRAME = 0.02


def _profile(duration: float, dips: dict[float, float], loud: float = -20.0) -> EnergyProfile:
    db = [loud] * int(duration / FRAME)
    for at, level in dips.items():
        for index in range(int((at - 0.1) / FRAME), int((at + 0.1) / FRAME)):
            db[index] = level
    return EnergyProfile(db=db, frame_seconds=FRAME)


def _build(tmp_path: Path, profile: EnergyProfile | None, duration: float = 30.0):
    return build_segments(
        speech_regions=[SpeechRegion(start_time=0.0, end_time=duration)],
        silence_regions=[],
        audio_duration=duration,
        source_audio_path=tmp_path / "source.wav",
        segments_dir=tmp_path,
        config=SegmenterConfig(),
        energy_profile=profile,
    )


def test_continuous_speech_without_profile_keeps_hard_cut(tmp_path: Path) -> None:
    segments = _build(tmp_path, None)

    assert [segment.logical_end_time for segment in segments] == [15.0, 30.0]
    assert segments[0].global_end_time == 15.3


def test_continuous_speech_cuts_at_breath_dip_with_narrow_padding(tmp_path: Path) -> None:
    segments = _build(tmp_path, _profile(30.0, {11.5: -50.0}))

    assert segments[0].logical_end_time == 11.5
    assert segments[0].global_end_time == 11.58
    assert segments[1].global_start_time == 11.42
    assert segments[0].global_start_time == 0.0
    assert segments[-1].global_end_time == 30.0


def test_shallow_dip_is_not_treated_as_a_pause(tmp_path: Path) -> None:
    segments = _build(tmp_path, _profile(30.0, {11.5: -25.0}))

    assert segments[0].logical_end_time == 15.0


def test_dip_before_search_window_is_ignored(tmp_path: Path) -> None:
    segments = _build(tmp_path, _profile(30.0, {4.0: -50.0}))

    assert segments[0].logical_end_time == 15.0
