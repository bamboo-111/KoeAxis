"""Adapters for timestamped ASR output formats.

Some ASR tools emit karaoke-style SRT where each subtitle block repeats the
whole segment and wraps the currently active word in a ``<font>`` tag.  The
optimizer pipeline expects plain word or sentence segments instead, so this
module normalizes JSON and SRT outputs into ``ASRData``.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

from optimizer.asr_data import ASRData, ASRDataSeg


_SRT_TIME_PATTERN = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*"
    r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})"
)
_FONT_PATTERN = re.compile(
    r"<font\b[^>]*>(.*?)</font>", re.IGNORECASE | re.DOTALL
)
_TAG_PATTERN = re.compile(r"<[^>]+>")


def load_timestamp_output(path: str | Path) -> ASRData:
    """Load timestamped JSON or SRT output as optimizer ``ASRData``."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".json":
        return timestamp_json_to_asr_data(json.loads(file_path.read_text(encoding="utf-8")))
    if suffix == ".srt":
        return timestamp_srt_to_asr_data(file_path.read_text(encoding="utf-8"))

    raise ValueError(f"Unsupported input format: {suffix}. Use .json or .srt")


def timestamp_json_to_asr_data(data: dict[str, Any]) -> ASRData:
    """Convert Whisper-style JSON into word-level ``ASRData``.

    The word list is preferred because it has clean timestamps without the
    repeated karaoke text found in SRT output.  Segment-level text is used as a
    fallback when no word timestamps are present.
    """
    segments: list[ASRDataSeg] = []

    for segment in data.get("segments", []):
        words = segment.get("words") or []
        for word in words:
            text = str(word.get("word", ""))
            if not text.strip():
                continue
            start = _seconds_to_ms(word.get("start", segment.get("start", 0)))
            end = _seconds_to_ms(word.get("end", segment.get("end", 0)))
            segments.append(ASRDataSeg(text=text, start_time=start, end_time=_safe_end(start, end)))

    if segments:
        return ASRData(segments)

    for segment in data.get("segments", []):
        text = str(segment.get("text", ""))
        if not text.strip():
            continue
        start = _seconds_to_ms(segment.get("start", 0))
        end = _seconds_to_ms(segment.get("end", 0))
        segments.append(ASRDataSeg(text=text, start_time=start, end_time=_safe_end(start, end)))

    return ASRData(segments)


def timestamp_srt_to_asr_data(srt_text: str) -> ASRData:
    """Convert karaoke SRT into word-level ``ASRData``.

    If no karaoke ``<font>`` tags are found, this falls back to the normal SRT
    parser in ``ASRData``.
    """
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    segments: list[ASRDataSeg] = []
    saw_karaoke_tag = False

    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue

        match = _SRT_TIME_PATTERN.match(lines[1].strip())
        if not match:
            continue

        text = "\n".join(lines[2:])
        highlighted = _FONT_PATTERN.findall(text)
        if not highlighted:
            continue

        saw_karaoke_tag = True
        start, end = _parse_srt_time_match(match)
        word_text = "".join(_strip_tags(part) for part in highlighted).strip()
        if word_text:
            segments.append(
                ASRDataSeg(
                    text=word_text,
                    start_time=start,
                    end_time=_safe_end(start, end),
                )
            )

    if saw_karaoke_tag:
        return ASRData(segments)

    return ASRData.from_srt(srt_text)


def _seconds_to_ms(value: Any) -> int:
    return int(round(float(value) * 1000))


def _safe_end(start: int, end: int) -> int:
    return end if end > start else start + 20


def _strip_tags(text: str) -> str:
    return html.unescape(_TAG_PATTERN.sub("", text))


def _parse_srt_time_match(match: re.Match[str]) -> tuple[int, int]:
    nums = list(map(int, match.groups()))
    start = nums[0] * 3600000 + nums[1] * 60000 + nums[2] * 1000 + nums[3]
    end = nums[4] * 3600000 + nums[5] * 60000 + nums[6] * 1000 + nums[7]
    return start, end
