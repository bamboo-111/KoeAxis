"""Timing allocation helpers for rule-based subtitle splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter_readability import (
    is_protected_short_utterance,
    normalize_filler_text,
    segment_duration,
)


@dataclass(frozen=True)
class InlineTimingConfig:
    short_utterance_min_duration: int = 1500


DEFAULT_INLINE_TIMING = InlineTimingConfig()

INLINE_SHORT_UTTERANCE_MIN_DURATION = DEFAULT_INLINE_TIMING.short_utterance_min_duration


def parts_to_timed_segments(
    segment: ASRDataSeg,
    parts: List[str],
    config: InlineTimingConfig = DEFAULT_INLINE_TIMING,
) -> List[ASRDataSeg]:
    duration = max(1, segment.end_time - segment.start_time)
    weights = [max(1, len(normalize_filler_text(part))) for part in parts]
    total = max(1, sum(weights))
    result: List[ASRDataSeg] = []
    cursor = segment.start_time
    consumed = 0
    for index, (part, weight) in enumerate(zip(parts, weights, strict=True)):
        consumed += weight
        if index == len(parts) - 1:
            end_time = segment.end_time
        else:
            end_time = segment.start_time + round(duration * consumed / total)
            remaining = len(parts) - index - 1
            end_time = max(cursor + 1, min(segment.end_time - remaining, end_time))
        result.append(ASRDataSeg(part.strip(), cursor, end_time))
        cursor = end_time
    return extend_inline_short_utterances(result, config)


def extend_inline_short_utterances(
    segments: List[ASRDataSeg],
    config: InlineTimingConfig = DEFAULT_INLINE_TIMING,
) -> List[ASRDataSeg]:
    if len(segments) < 2:
        return segments
    result = list(segments)
    for index, segment in enumerate(result[:-1]):
        if not is_protected_short_utterance(segment.text):
            continue
        duration = segment_duration(segment)
        if duration >= config.short_utterance_min_duration:
            continue
        next_seg = result[index + 1]
        needed = config.short_utterance_min_duration - duration
        available = max(0, segment_duration(next_seg) - 1)
        shift = min(needed, available)
        if shift <= 0:
            continue
        segment.end_time += shift
        next_seg.start_time = segment.end_time
    return result


def split_text_evenly_with_timing(
    segment: ASRDataSeg,
    num_parts: int,
) -> List[ASRDataSeg]:
    """Split plain text into equal character spans with proportional timing."""
    if num_parts <= 1:
        return [segment]

    text = segment.text
    duration = segment.end_time - segment.start_time
    part_len = len(text) // num_parts
    parts: List[ASRDataSeg] = []
    for index in range(num_parts):
        if index < num_parts - 1:
            part_text = text[index * part_len : (index + 1) * part_len].strip()
        else:
            part_text = text[index * part_len :].strip()
        if not part_text:
            continue
        part_start = segment.start_time + int(duration * (index / num_parts))
        part_end = segment.start_time + int(duration * ((index + 1) / num_parts))
        parts.append(ASRDataSeg(part_text, part_start, part_end))
    return parts if parts else [segment]
