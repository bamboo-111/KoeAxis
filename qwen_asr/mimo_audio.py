from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

from qwen_asr.glossary import read_xlsx_glossary
from qwen_asr.storage import ensure_directory


def load_glossary(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    return [
        {
            "group": entry.group,
            "source": entry.source,
            "target": entry.target,
            "note": entry.note,
        }
        for entry in read_xlsx_glossary(path)
    ]


def subtitle_entries_for_segment(
    translated: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    start_ms = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
    end_ms = int(round(float(segment.get("global_end_time", 0.0)) * 1000))
    return {
        str(key): value
        for key, value in translated.items()
        if isinstance(value, dict)
        and int(value.get("end_time", 0)) > start_ms
        and int(value.get("start_time", 0)) < end_ms
    }


def filter_glossary(
    glossary: list[dict[str, str]],
    subtitle_entries: dict[str, Any],
    limit: int,
) -> list[dict[str, str]]:
    if not glossary or limit <= 0:
        return []
    haystack = "\n".join(
        f"{item.get('original_subtitle', '')}\n{item.get('translated_subtitle', '')}"
        for item in subtitle_entries.values()
        if isinstance(item, dict)
    )
    exact_matches = [entry for entry in glossary if entry["source"] and entry["source"] in haystack]
    general = [
        entry
        for entry in glossary
        if entry not in exact_matches and entry.get("group") in {"names", "show_terms", "通用日语"}
    ]
    return (exact_matches + general)[:limit]


def segment_for_subtitle_id(
    subtitle_id: str,
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
) -> dict[str, Any]:
    item = translated.get(subtitle_id)
    if not isinstance(item, dict):
        raise KeyError(f"subtitle id not found: {subtitle_id}")
    start_ms = int(item.get("start_time", 0))
    for segment in segments:
        segment_start = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
        segment_end = int(round(float(segment.get("global_end_time", 0.0)) * 1000))
        if segment_start <= start_ms < segment_end:
            return segment
    raise RuntimeError(f"No audio segment covers subtitle id {subtitle_id}")


def build_nearby_audio_batches(
    *,
    subtitle_ids: list[str],
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
    context_subtitles: int,
    batch_size: int,
    max_gap_s: float,
) -> list[list[str]]:
    candidates: list[tuple[str, str, int, int]] = []
    for subtitle_id in subtitle_ids:
        item = translated.get(subtitle_id)
        if not isinstance(item, dict):
            continue
        try:
            segment = segment_for_subtitle_id(subtitle_id, segments, translated)
        except RuntimeError:
            continue
        entries = nearby_entries_for_subtitle_id(subtitle_id, translated, context_subtitles)
        start_ms = min(int(entry.get("start_time", 0)) for entry in entries.values())
        end_ms = max(int(entry.get("end_time", 0)) for entry in entries.values())
        candidates.append((subtitle_id, str(segment.get("segment_id", "")), start_ms, end_ms))

    candidates.sort(key=lambda row: (row[1], row[2], int(row[0])))
    batches: list[list[str]] = []
    current: list[str] = []
    current_segment_id = ""
    current_end_ms = 0
    max_gap_ms = int(round(max_gap_s * 1000))

    for subtitle_id, segment_id, start_ms, end_ms in candidates:
        can_join = (
            current
            and segment_id == current_segment_id
            and len(current) < batch_size
            and start_ms - current_end_ms <= max_gap_ms
        )
        if not can_join:
            if current:
                batches.append(current)
            current = [subtitle_id]
            current_segment_id = segment_id
            current_end_ms = end_ms
            continue
        current.append(subtitle_id)
        current_end_ms = max(current_end_ms, end_ms)

    if current:
        batches.append(current)
    return batches


def nearby_entries_for_subtitle_id(
    subtitle_id: str,
    translated: dict[str, Any],
    context_subtitles: int,
) -> dict[str, Any]:
    if not subtitle_id.isdigit():
        raise ValueError(f"subtitle id must be numeric: {subtitle_id}")
    center = int(subtitle_id)
    start = max(1, center - context_subtitles)
    end = center + context_subtitles
    entries: dict[str, Any] = {}
    for index in range(start, end + 1):
        key = str(index)
        item = translated.get(key)
        if isinstance(item, dict):
            entries[key] = item
    return entries


def nearby_entries_for_subtitle_ids(
    subtitle_ids: list[str],
    translated: dict[str, Any],
    context_subtitles: int,
) -> dict[str, Any]:
    numeric_ids = [int(subtitle_id) for subtitle_id in subtitle_ids if subtitle_id.isdigit()]
    if not numeric_ids:
        return {}
    start = max(1, min(numeric_ids) - context_subtitles)
    end = max(numeric_ids) + context_subtitles
    entries: dict[str, Any] = {}
    for index in range(start, end + 1):
        key = str(index)
        item = translated.get(key)
        if isinstance(item, dict):
            entries[key] = item
    return entries


def write_nearby_audio_clip(
    *,
    subtitle_id: str,
    segment: dict[str, Any],
    entries: dict[str, Any],
    audio_path: Path,
    clips_dir: Path,
    padding_s: float,
) -> tuple[Path, dict[str, float]]:
    if not entries:
        raise ValueError(f"No nearby entries for subtitle id {subtitle_id}")
    ensure_directory(clips_dir)
    source_audio_path = Path(str(segment.get("source_audio_path", "")))
    try:
        using_source_audio = source_audio_path.exists() and audio_path.resolve() == source_audio_path.resolve()
    except OSError:
        using_source_audio = False
    segment_global_start_ms = 0 if using_source_audio else int(
        round(float(segment.get("global_start_time", 0.0)) * 1000)
    )
    min_start_ms = min(int(item.get("start_time", 0)) for item in entries.values())
    max_end_ms = max(int(item.get("end_time", 0)) for item in entries.values())
    local_start_s = max(0.0, (min_start_ms - segment_global_start_ms) / 1000.0 - padding_s)
    local_end_s = max(local_start_s + 0.2, (max_end_ms - segment_global_start_ms) / 1000.0 + padding_s)

    clip_path = clips_dir / f"subtitle_{subtitle_id}_nearby.wav"
    with wave.open(str(audio_path), "rb") as source:
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        start_frame = max(0, min(total_frames, int(round(local_start_s * frame_rate))))
        end_frame = max(start_frame + 1, min(total_frames, int(round(local_end_s * frame_rate))))
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        params = source.getparams()

    with wave.open(str(clip_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)

    duration_s = (end_frame - start_frame) / max(frame_rate, 1)
    return clip_path, {
        "start_s": start_frame / max(frame_rate, 1),
        "end_s": end_frame / max(frame_rate, 1),
        "duration_s": duration_s,
    }
