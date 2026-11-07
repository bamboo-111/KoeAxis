"""Display-duration adjustment helpers for subtitle splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter_readability import (
    READABILITY_MERGE_MAX_CJK,
    is_dialogue_standalone_response,
    is_protected_short_display_response,
    segment_duration,
)
from optimizer.text_utils import count_words


@dataclass(frozen=True)
class DisplayDurationConfig:
    protected_short_min_duration: int = 120
    ordinary_subtitle_min_duration: int = 500
    max_subtitle_display_duration: int = 8000


DEFAULT_DISPLAY_DURATION = DisplayDurationConfig()

PROTECTED_SHORT_MIN_DURATION = DEFAULT_DISPLAY_DURATION.protected_short_min_duration
ORDINARY_SUBTITLE_MIN_DURATION = DEFAULT_DISPLAY_DURATION.ordinary_subtitle_min_duration
MAX_SUBTITLE_DISPLAY_DURATION = DEFAULT_DISPLAY_DURATION.max_subtitle_display_duration

MergeSegments = Callable[[ASRDataSeg, ASRDataSeg], ASRDataSeg]


def merge_display_segments(left: ASRDataSeg, right: ASRDataSeg) -> ASRDataSeg:
    """Merge display fragments while preserving text and optional translation."""
    translated_text = ""
    if left.translated_text or right.translated_text:
        translated_text = f"{left.translated_text}{right.translated_text}"
    return ASRDataSeg(
        f"{left.text}{right.text}",
        min(left.start_time, right.start_time),
        max(left.end_time, right.end_time),
        translated_text=translated_text,
    )


def extend_protected_short_display_durations(
    segments: List[ASRDataSeg],
    config: DisplayDurationConfig = DEFAULT_DISPLAY_DURATION,
    merge_segments: MergeSegments = merge_display_segments,
) -> List[ASRDataSeg]:
    """Use adjacent silence to keep short subtitles readable."""
    result = merge_zero_gap_short_display_fragments(list(segments), config, merge_segments)
    for index, segment in enumerate(result):
        if not is_protected_short_display_response(segment.text):
            continue
        extend_segment_display_duration(
            result,
            index,
            config.protected_short_min_duration,
        )

    for index, segment in enumerate(result):
        if is_protected_short_display_response(segment.text):
            continue
        extend_segment_display_duration(
            result,
            index,
            config.ordinary_subtitle_min_duration,
        )

    redistribute_zero_gap_short_display_durations(result, config)
    for index, segment in enumerate(result):
        extend_segment_display_duration(
            result,
            index,
            minimum_display_duration(segment, config),
        )
    cap_long_display_durations(result, config)

    return result


def cap_long_display_durations(
    segments: List[ASRDataSeg],
    config: DisplayDurationConfig = DEFAULT_DISPLAY_DURATION,
) -> None:
    for segment in segments:
        if segment_duration(segment) > config.max_subtitle_display_duration:
            segment.end_time = segment.start_time + config.max_subtitle_display_duration


def merge_zero_gap_short_display_fragments(
    segments: List[ASRDataSeg],
    config: DisplayDurationConfig = DEFAULT_DISPLAY_DURATION,
    merge_segments: MergeSegments = merge_display_segments,
) -> List[ASRDataSeg]:
    result: List[ASRDataSeg] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        next_seg = segments[index + 1] if index + 1 < len(segments) else None
        if next_seg is None or next_seg.start_time != current.end_time:
            result.append(current)
            index += 1
            continue

        current_protected = is_protected_short_display_response(current.text)
        next_protected = is_protected_short_display_response(next_seg.text)
        duration = segment_duration(current)
        merged_text = f"{current.text}{next_seg.text}"
        can_merge_text = count_words(merged_text) <= READABILITY_MERGE_MAX_CJK
        should_merge = (
            can_merge_text
            and (
                (
                    current_protected
                    and next_protected
                    and duration < config.protected_short_min_duration
                )
                or (
                    not current_protected
                    and duration < config.ordinary_subtitle_min_duration
                    and not is_dialogue_standalone_response(current.text)
                )
            )
        )
        if should_merge:
            result.append(merge_segments(current, next_seg))
            index += 2
            continue

        result.append(current)
        index += 1
    return result


def redistribute_zero_gap_short_display_durations(
    segments: List[ASRDataSeg],
    config: DisplayDurationConfig = DEFAULT_DISPLAY_DURATION,
) -> None:
    for index, segment in enumerate(segments[:-1]):
        minimum = minimum_display_duration(segment, config)
        duration = segment_duration(segment)
        if duration >= minimum:
            continue
        next_seg = segments[index + 1]
        if next_seg.start_time != segment.end_time:
            continue

        next_minimum = minimum_display_duration(next_seg, config)
        needed = minimum - duration
        available = max(0, segment_duration(next_seg) - next_minimum)
        shift = min(needed, available)
        if shift <= 0:
            continue
        segment.end_time += shift
        next_seg.start_time = segment.end_time


def minimum_display_duration(
    segment: ASRDataSeg,
    config: DisplayDurationConfig = DEFAULT_DISPLAY_DURATION,
) -> int:
    if is_protected_short_display_response(segment.text):
        return config.protected_short_min_duration
    return config.ordinary_subtitle_min_duration


def extend_segment_display_duration(
    segments: List[ASRDataSeg],
    index: int,
    minimum_duration: int,
) -> None:
    segment = segments[index]
    duration = segment_duration(segment)
    if duration >= minimum_duration:
        return

    needed = minimum_duration - duration
    next_start = segments[index + 1].start_time if index + 1 < len(segments) else None
    if next_start is not None:
        extend = min(needed, max(0, next_start - segment.end_time))
        if extend > 0:
            segment.end_time += extend
            needed -= extend
    if needed <= 0:
        return

    prev_end = segments[index - 1].end_time if index > 0 else None
    if prev_end is None:
        segment.start_time = max(0, segment.start_time - needed)
        return

    shift = min(needed, max(0, segment.start_time - prev_end))
    if shift > 0:
        segment.start_time -= shift
