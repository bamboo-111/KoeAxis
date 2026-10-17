from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optimizer.asr_data import ASRData, ASRDataSeg
from optimizer.splitter_display import DisplayDurationConfig, redistribute_zero_gap_short_display_durations
from qwen_asr.subtitle_layout import reflow_subtitle_segments, strip_display_punctuation


@dataclass(slots=True)
class NormalizeParams:
    extend_ms: int = 350
    snap_gap_ms: int = 200
    min_blank_ms: int = 300
    min_duration_ms: int = 500


def normalize_asr_data(
    asr_data: ASRData,
    params: NormalizeParams | None = None,
    *,
    timing_tokens: list[ASRDataSeg] | None = None,
    layout_events: list[dict[str, Any]] | None = None,
) -> ASRData:
    params = params or NormalizeParams()
    _validate_params(params)

    if not asr_data.segments:
        return ASRData([])

    segments = reflow_subtitle_segments(
        asr_data.segments,
        timing_tokens=timing_tokens,
        events=layout_events,
        display_extension_ms=params.extend_ms,
    )

    normalized = []
    extend_ms = params.extend_ms
    snap_gap_ms = params.snap_gap_ms
    min_blank_ms = params.min_blank_ms

    for index, seg in enumerate(segments):
        display_start = seg.start_time
        display_end = seg.end_time + extend_ms
        normalized.append(
            {
                "index": index + 1,
                "start_time": display_start,
                "end_time": display_end,
                "voice_end_time": seg.end_time,
                "text": seg.text,
                "translated_text": seg.translated_text,
            }
        )

    for index in range(len(normalized) - 1):
        current = normalized[index]
        next_item = normalized[index + 1]
        gap = next_item["start_time"] - current["end_time"]

        if gap < snap_gap_ms:
            current["end_time"] = next_item["start_time"]
        elif gap <= min_blank_ms:
            current["end_time"] = max(
                current["voice_end_time"],
                next_item["start_time"] - min_blank_ms,
            )

    result: list[ASRDataSeg] = []
    for item in normalized:
        end_time = item["end_time"]
        if end_time <= item["start_time"]:
            end_time = item["voice_end_time"]
        result.append(
            ASRDataSeg(
                text=item["text"],
                translated_text=item["translated_text"],
                start_time=int(item["start_time"]),
                end_time=int(end_time),
            )
        )

    for index in range(len(result) - 1):
        if result[index].end_time > result[index + 1].start_time:
            result[index].end_time = result[index + 1].start_time

    for index, seg in enumerate(result):
        if seg.end_time - seg.start_time >= params.min_duration_ms:
            continue
        max_end_time = (
            result[index + 1].start_time
            if index + 1 < len(result)
            else seg.start_time + params.min_duration_ms
        )
        seg.end_time = min(seg.start_time + params.min_duration_ms, max_end_time)

    result = _merge_non_positive_segments(result)
    for index, seg in enumerate(result):
        missing = params.min_duration_ms - (seg.end_time - seg.start_time)
        if missing > 0:
            # No room after the cue: start it earlier inside the free gap before it.
            previous_end = result[index - 1].end_time if index > 0 else 0
            seg.start_time = max(previous_end, seg.start_time - missing)
    redistribute_zero_gap_short_display_durations(
        result,
        DisplayDurationConfig(
            protected_short_min_duration=params.min_duration_ms,
            ordinary_subtitle_min_duration=params.min_duration_ms,
        ),
    )
    _borrow_short_display_from_previous(result, params.min_duration_ms)
    for segment in result:
        segment.text = strip_display_punctuation(segment.text)
        segment.translated_text = strip_display_punctuation(segment.translated_text)
    return ASRData([segment for segment in result if segment.text or segment.translated_text])


def _borrow_short_display_from_previous(segments: list[ASRDataSeg], minimum: int) -> None:
    """Give a still-short cue time from the end of an adjacent longer previous cue (display only)."""
    for index in range(1, len(segments)):
        current = segments[index]
        previous = segments[index - 1]
        needed = minimum - (current.end_time - current.start_time)
        if needed <= 0 or previous.end_time != current.start_time:
            continue
        shift = min(needed, max(0, (previous.end_time - previous.start_time) - minimum))
        if shift > 0:
            previous.end_time -= shift
            current.start_time -= shift


def _validate_params(params: NormalizeParams) -> None:
    if params.extend_ms < 0:
        raise ValueError(f"extend_ms cannot be negative: {params.extend_ms}")
    if params.snap_gap_ms < 0:
        raise ValueError(f"snap_gap_ms cannot be negative: {params.snap_gap_ms}")
    if params.min_blank_ms < params.snap_gap_ms:
        raise ValueError(
            f"min_blank_ms ({params.min_blank_ms}) must be >= snap_gap_ms ({params.snap_gap_ms})"
        )
    if params.min_duration_ms < 0:
        raise ValueError(f"min_duration_ms cannot be negative: {params.min_duration_ms}")


def _merge_non_positive_segments(segments: list[ASRDataSeg]) -> list[ASRDataSeg]:
    cleaned: list[ASRDataSeg] = []
    index = 0
    while index < len(segments):
        current = segments[index]
        if current.end_time > current.start_time:
            cleaned.append(current)
            index += 1
            continue

        if index + 1 < len(segments):
            next_item = segments[index + 1]
            next_item.text = f"{current.text}{next_item.text}"
            if current.translated_text:
                next_item.translated_text = f"{current.translated_text}{next_item.translated_text}"
            next_item.start_time = min(current.start_time, next_item.start_time)
        elif cleaned:
            previous = cleaned[-1]
            previous.text = f"{previous.text}{current.text}"
            if current.translated_text:
                previous.translated_text = f"{previous.translated_text}{current.translated_text}"
            previous.end_time = max(previous.end_time, current.end_time)
        index += 1
    return [item for item in cleaned if item.end_time > item.start_time]
