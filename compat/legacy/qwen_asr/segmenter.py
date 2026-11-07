from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

from qwen_asr.models import AudioSegment, SilenceRegion, SpeechRegion

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SegmenterConfig:
    max_segment_seconds: float = 15.0
    min_segment_seconds: float = 2.0
    preferred_silence_ms: int = 800
    min_silence_ms: int = 500
    padding_ms: int = 300
    overlap_ms: int = 0
    # When a window has no usable VAD silence (continuous talk, e.g. radio), cut at the quietest
    # point in the tail of the window instead of mid-word at the hard limit. 0 disables.
    energy_cut_min_depth_db: float = 10.0
    energy_cut_search_ratio: float = 0.4
    # Padding at an energy cut stays inside the dip so neighbours do not re-transcribe the same words.
    energy_cut_padding_ms: int = 80


@dataclass(slots=True)
class EnergyProfile:
    """Smoothed frame loudness in dB; frame ``i`` covers ``[i * frame_seconds, (i + 1) * frame_seconds)``."""

    db: list[float]
    frame_seconds: float = 0.02

    def window(self, start: float, end: float) -> tuple[int, list[float]]:
        first = max(0, int(start / self.frame_seconds))
        last = min(len(self.db), int(end / self.frame_seconds))
        return first, self.db[first:last]


def compute_energy_profile(
    audio_path: Path,
    frame_seconds: float = 0.02,
    smooth_seconds: float = 0.2,
) -> EnergyProfile:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    hop = max(1, int(round(frame_seconds * sample_rate)))
    usable = len(mono) // hop * hop
    if usable == 0:
        return EnergyProfile(db=[], frame_seconds=frame_seconds)
    power = (mono[:usable].reshape(-1, hop) ** 2).mean(axis=1)
    width = max(1, int(round(smooth_seconds / frame_seconds)))
    smoothed = np.convolve(power, np.ones(width) / width, mode="same")
    return EnergyProfile(db=(10.0 * np.log10(smoothed + 1e-12)).tolist(), frame_seconds=frame_seconds)


def build_segments(
    speech_regions: list[SpeechRegion],
    silence_regions: list[SilenceRegion],
    audio_duration: float,
    source_audio_path: Path,
    segments_dir: Path,
    config: SegmenterConfig,
    energy_profile: EnergyProfile | None = None,
) -> list[AudioSegment]:
    if not speech_regions:
        LOGGER.warning("No speech detected. Creating one full-audio segment.")
        return [
            _make_segment(
                segment_index=1,
                start_time=0.0,
                end_time=audio_duration,
                logical_start=0.0,
                logical_end=audio_duration,
                source_audio_path=source_audio_path,
                segments_dir=segments_dir,
                config=config,
                audio_duration=audio_duration,
            )
        ]

    energy_cuts: set[float] = set()
    logical_segments = _build_logical_segments(
        speech_regions=speech_regions,
        silence_regions=silence_regions,
        audio_duration=audio_duration,
        config=config,
        energy_profile=energy_profile,
        energy_cuts=energy_cuts,
    )
    merged = _normalize_logical_segments(logical_segments, audio_duration, config.min_segment_seconds)
    if energy_cuts:
        LOGGER.info("Segmenter used %d energy-dip cuts where VAD found no silence", len(energy_cuts))
    return [
        _make_segment(
            segment_index=index,
            start_time=start_time,
            end_time=end_time,
            logical_start=start_time,
            logical_end=end_time,
            source_audio_path=source_audio_path,
            segments_dir=segments_dir,
            config=config,
            audio_duration=audio_duration,
            start_padding_ms=config.energy_cut_padding_ms if _cut_key(start_time) in energy_cuts else None,
            end_padding_ms=config.energy_cut_padding_ms if _cut_key(end_time) in energy_cuts else None,
        )
        for index, (start_time, end_time) in enumerate(merged, start=1)
    ]


def _build_logical_segments(
    speech_regions: list[SpeechRegion],
    silence_regions: list[SilenceRegion],
    audio_duration: float,
    config: SegmenterConfig,
    energy_profile: EnergyProfile | None = None,
    energy_cuts: set[float] | None = None,
) -> list[tuple[float, float]]:
    logical_segments: list[tuple[float, float]] = []
    overlap_seconds = max(0.0, config.overlap_ms / 1000.0)

    cursor = speech_regions[0].start_time
    final_end = speech_regions[-1].end_time

    while cursor < final_end:
        latest_end = min(cursor + config.max_segment_seconds, final_end)
        if final_end - cursor <= config.max_segment_seconds:
            logical_segments.append((cursor, final_end))
            break

        cut_time = _find_cut_time(
            current_start=cursor,
            current_end=latest_end,
            silence_regions=silence_regions,
            config=config,
        )
        if cut_time is None:
            cut_time = _find_energy_cut_time(cursor, latest_end, energy_profile, config)
            if cut_time is not None and energy_cuts is not None:
                energy_cuts.add(_cut_key(cut_time))
        if cut_time is None:
            cut_time = latest_end
        cut_time = max(cursor + config.min_segment_seconds, cut_time)
        cut_time = min(cut_time, latest_end)

        if cut_time <= cursor:
            cut_time = latest_end

        logical_segments.append((cursor, cut_time))
        next_cursor = max(0.0, cut_time - overlap_seconds)
        if next_cursor <= cursor:
            next_cursor = cut_time
        cursor = next_cursor

    if not logical_segments:
        logical_segments.append((0.0, min(audio_duration, config.max_segment_seconds)))
    return logical_segments


def _find_cut_time(
    current_start: float,
    current_end: float,
    silence_regions: list[SilenceRegion],
    config: SegmenterConfig,
) -> float | None:
    preferred = config.preferred_silence_ms / 1000.0
    minimum = config.min_silence_ms / 1000.0
    target = current_end

    candidates_preferred = [
        silence
        for silence in silence_regions
        if (current_start + config.min_segment_seconds) <= silence.midpoint <= current_end
        and silence.duration >= preferred
    ]
    if candidates_preferred:
        return min(candidates_preferred, key=lambda item: abs(item.midpoint - target)).midpoint

    candidates_min = [
        silence
        for silence in silence_regions
        if (current_start + config.min_segment_seconds) <= silence.midpoint <= current_end
        and silence.duration >= minimum
    ]
    if candidates_min:
        return min(candidates_min, key=lambda item: abs(item.midpoint - target)).midpoint

    return None


def _find_energy_cut_time(
    current_start: float,
    current_end: float,
    energy_profile: EnergyProfile | None,
    config: SegmenterConfig,
) -> float | None:
    if energy_profile is None or not energy_profile.db or config.energy_cut_min_depth_db <= 0:
        return None
    _, whole = energy_profile.window(current_start, current_end)
    if not whole:
        return None
    span = current_end - current_start
    search_start = max(
        current_start + config.min_segment_seconds,
        current_end - span * max(0.0, min(1.0, config.energy_cut_search_ratio)),
    )
    first, tail = energy_profile.window(search_start, current_end)
    if not tail:
        return None
    index = min(range(len(tail)), key=tail.__getitem__)
    floor = tail[index]
    if statistics.median(whole) - floor < config.energy_cut_min_depth_db:
        return None
    # Cut in the middle of the quiet run, not at its edge next to the preceding word.
    left = right = index
    while left > 0 and tail[left - 1] <= floor + 1.0:
        left -= 1
    while right + 1 < len(tail) and tail[right + 1] <= floor + 1.0:
        right += 1
    return round((first + (left + right + 1) / 2) * energy_profile.frame_seconds, 3)


def _cut_key(value: float) -> float:
    return round(value, 3)


def _normalize_logical_segments(
    segments: list[tuple[float, float]],
    audio_duration: float,
    min_segment_seconds: float,
) -> list[tuple[float, float]]:
    normalized: list[tuple[float, float]] = []
    for start_time, end_time in segments:
        start_time = max(0.0, start_time)
        end_time = min(audio_duration, end_time)
        if end_time <= start_time:
            continue
        if normalized and start_time < normalized[-1][1]:
            prev_start, prev_end = normalized[-1]
            normalized[-1] = (prev_start, max(prev_end, end_time))
            continue
        if normalized and (end_time - start_time) < min_segment_seconds:
            prev_start, _ = normalized[-1]
            normalized[-1] = (prev_start, end_time)
        else:
            normalized.append((start_time, end_time))
    return normalized


def _make_segment(
    segment_index: int,
    start_time: float,
    end_time: float,
    logical_start: float,
    logical_end: float,
    source_audio_path: Path,
    segments_dir: Path,
    config: SegmenterConfig,
    audio_duration: float,
    start_padding_ms: int | None = None,
    end_padding_ms: int | None = None,
) -> AudioSegment:
    start_padding = (config.padding_ms if start_padding_ms is None else start_padding_ms) / 1000.0
    end_padding = (config.padding_ms if end_padding_ms is None else end_padding_ms) / 1000.0
    padded_start = max(0.0, start_time - start_padding)
    padded_end = min(audio_duration, end_time + end_padding)
    segment_id = f"segment_{segment_index:06d}"
    return AudioSegment(
        segment_id=segment_id,
        audio_path=str((segments_dir / f"{segment_id}.wav").resolve()),
        source_audio_path=str(source_audio_path.resolve()),
        global_start_time=round(padded_start, 3),
        global_end_time=round(padded_end, 3),
        duration=round(padded_end - padded_start, 3),
        logical_start_time=round(logical_start, 3),
        logical_end_time=round(logical_end, 3),
    )
