from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from qwen_asr.models import AlignedSegment, AlignedToken, TranscriptSegment
from optimizer.text_utils import clean_subtitle_text

LOGGER = logging.getLogger(__name__)

PUNCTUATION = set("。！？!?；;：:,，、.")


@dataclass(slots=True)
class SubtitleConfig:
    max_subtitle_duration: float = 6.0
    min_subtitle_duration: float = 1.0
    max_chars_per_line_zh: int = 18
    max_chars_per_line_en: int = 42
    max_lines: int = 2
    pause_split_seconds: float = 0.8


@dataclass(slots=True)
class SubtitleCue:
    index: int
    start_time: float
    end_time: float
    text: str


def build_cues_from_aligned_segments(
    aligned_segments: list[AlignedSegment],
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    index = 1
    for segment in aligned_segments:
        if segment.status != "completed" or not segment.tokens:
            continue
        groups = _group_tokens(segment.tokens, config)
        for text, start_time, end_time in groups:
            cues.append(SubtitleCue(index=index, start_time=start_time, end_time=end_time, text=text))
            index += 1
    return cues


def build_coarse_cues_from_transcripts(
    transcripts: list[TranscriptSegment],
    config: SubtitleConfig,
) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    index = 1
    for segment in transcripts:
        if segment.status != "completed" or not segment.text.strip():
            continue
        text_lines = _wrap_text(segment.text.strip(), config)
        cues.append(
            SubtitleCue(
                index=index,
                start_time=segment.global_start_time,
                end_time=min(
                    segment.global_start_time + config.max_subtitle_duration,
                    max(segment.global_start_time + config.min_subtitle_duration, segment.global_end_time),
                ),
                text=text_lines,
            )
        )
        index += 1
    return cues


def export_srt(cues: list[SubtitleCue]) -> str:
    blocks = []
    for cue in cues:
        blocks.append(
            f"{cue.index}\n{_format_timestamp(cue.start_time, for_vtt=False)} --> {_format_timestamp(cue.end_time, for_vtt=False)}\n{cue.text}\n"
        )
    return "\n".join(blocks).strip() + "\n"


def export_vtt(cues: list[SubtitleCue]) -> str:
    blocks = ["WEBVTT\n"]
    for cue in cues:
        blocks.append(
            f"{_format_timestamp(cue.start_time, for_vtt=True)} --> {_format_timestamp(cue.end_time, for_vtt=True)}\n{cue.text}\n"
        )
    return "\n".join(blocks).strip() + "\n"


def export_vtt_from_optimizer_asr_data(asr_data: Any) -> str:
    blocks = ["WEBVTT\n"]
    for segment in asr_data.segments:
        text = clean_subtitle_text(segment.text)
        translated = clean_subtitle_text(getattr(segment, "translated_text", ""))
        if translated:
            text = f"{text}\n{translated}"
        blocks.append(
            f"{_format_ms_timestamp(segment.start_time)} --> {_format_ms_timestamp(segment.end_time)}\n{text}\n"
        )
    return "\n".join(blocks).strip() + "\n"


def _group_tokens(tokens: list[AlignedToken], config: SubtitleConfig) -> list[tuple[str, float, float]]:
    groups: list[tuple[str, float, float]] = []
    current_tokens: list[AlignedToken] = []
    for token in tokens:
        if not current_tokens:
            current_tokens.append(token)
            continue

        tentative = current_tokens + [token]
        duration = tentative[-1].end_time - tentative[0].start_time
        gap = max(0.0, token.start_time - current_tokens[-1].end_time)
        text = _tokens_to_text(tentative)
        should_split = False

        if gap >= config.pause_split_seconds and duration >= config.min_subtitle_duration:
            should_split = True
        elif duration >= config.max_subtitle_duration:
            should_split = True
        elif _is_punctuation_boundary(current_tokens[-1].text) and _fits_display(text, config):
            should_split = True
        elif not _fits_display(text, config):
            should_split = True

        if should_split:
            groups.append(
                (
                    _wrap_text(_tokens_to_text(current_tokens), config),
                    current_tokens[0].start_time,
                    current_tokens[-1].end_time,
                )
            )
            current_tokens = [token]
        else:
            current_tokens.append(token)

    if current_tokens:
        groups.append(
            (
                _wrap_text(_tokens_to_text(current_tokens), config),
                current_tokens[0].start_time,
                current_tokens[-1].end_time,
            )
        )
    return groups


def _tokens_to_text(tokens: list[AlignedToken]) -> str:
    if not tokens:
        return ""
    pieces = [token.text for token in tokens]
    if any(" " in piece for piece in pieces):
        return "".join(pieces).strip()
    if any(_contains_cjk(piece) for piece in pieces):
        return "".join(pieces).strip()
    return " ".join(piece.strip() for piece in pieces if piece.strip()).strip()


def _is_punctuation_boundary(text: str) -> bool:
    return bool(text) and text[-1] in PUNCTUATION


def _fits_display(text: str, config: SubtitleConfig) -> bool:
    lines = _wrap_text(text, config).splitlines()
    return len(lines) <= config.max_lines


def _wrap_text(text: str, config: SubtitleConfig) -> str:
    text = clean_subtitle_text(re.sub(r"\s+", " ", text).strip())
    if not text:
        return ""
    if _contains_cjk(text):
        return _wrap_cjk(text, config.max_chars_per_line_zh, config.max_lines)
    return _wrap_words(text, config.max_chars_per_line_en, config.max_lines)


def _wrap_cjk(text: str, max_chars: int, max_lines: int) -> str:
    lines = [text[index : index + max_chars] for index in range(0, len(text), max_chars)]
    return "\n".join(lines[:max_lines])


def _wrap_words(text: str, max_chars: int, max_lines: int) -> str:
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        lines.append(current)
        current = word
        if len(lines) >= max_lines:
            break
    if current and len(lines) < max_lines:
        lines.append(current)
    return "\n".join(lines[:max_lines])


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _format_timestamp(seconds: float, for_vtt: bool) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    separator = "." if for_vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _format_ms_timestamp(milliseconds: int) -> str:
    total_ms = max(0, int(milliseconds))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
