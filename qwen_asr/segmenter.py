from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from qwen_asr.models import AudioSegment, SilenceRegion, SpeechRegion

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SegmenterConfig:
    max_segment_seconds: float = 60.0
    min_segment_seconds: float = 8.0
    preferred_silence_ms: int = 800
    min_silence_ms: int = 500
    padding_ms: int = 300
    overlap_ms: int = 0


def build_segments(
    speech_regions: list[SpeechRegion],
    silence_regions: list[SilenceRegion],
    audio_duration: float,
    source_audio_path: Path,
    segments_dir: Path,
    config: SegmenterConfig,
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

    logical_segments = _build_logical_segments(
        speech_regions=speech_regions,
        silence_regions=silence_regions,
        audio_duration=audio_duration,
        config=config,
    )
    merged = _normalize_logical_segments(logical_segments, audio_duration, config.min_segment_seconds)
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
        )
        for index, (start_time, end_time) in enumerate(merged, start=1)
    ]


def _build_logical_segments(
    speech_regions: list[SpeechRegion],
    silence_regions: list[SilenceRegion],
    audio_duration: float,
    config: SegmenterConfig,
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
) -> float:
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

    return target


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
) -> AudioSegment:
    padding = config.padding_ms / 1000.0
    padded_start = max(0.0, start_time - padding)
    padded_end = min(audio_duration, end_time + padding)
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
