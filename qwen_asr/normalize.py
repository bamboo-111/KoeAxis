from __future__ import annotations

from dataclasses import dataclass

from optimizer.asr_data import ASRData, ASRDataSeg


@dataclass(slots=True)
class NormalizeParams:
    extend_ms: int = 350
    snap_gap_ms: int = 200
    min_blank_ms: int = 300
    min_duration_ms: int = 300


def normalize_asr_data(
    asr_data: ASRData,
    params: NormalizeParams | None = None,
) -> ASRData:
    params = params or NormalizeParams()
    _validate_params(params)

    if not asr_data.segments:
        return ASRData([])

    normalized = []
    extend_ms = params.extend_ms
    snap_gap_ms = params.snap_gap_ms
    min_blank_ms = params.min_blank_ms

    for index, seg in enumerate(asr_data.segments):
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

    return ASRData(_merge_non_positive_segments(result))


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
