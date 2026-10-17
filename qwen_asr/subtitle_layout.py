"""Content-preserving bilingual cue layout and display punctuation cleanup."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter_readability import is_dialogue_standalone_response

# A dot directly followed by an ASCII letter/digit is a domain/abbreviation dot ("interfm.jp"), not a sentence end.
_SENTENCE_END = re.compile(r"(?:[。！？!?]+|(?<!\d)\.(?![\dA-Za-z]))[」』）》】\"']*")
_SOURCE_TAILS = (
    "立ち", "実際", "ちなみに", "この", "その", "あの", "ぜひ", "ので", "から", "けど", "という",
    "と", "が", "を", "に", "で", "は", "の", "て",
)
_TARGET_TAILS = (
    "这个", "那个", "然后", "顺便说一下", "一定要", "拍到的照片", "将身体", "可以", "和", "与", "的", "把", "将",
)
_SOURCE_HEADS = ("位置", "は", "が", "を", "に", "です", "ます", "でした", "ました", "取りました")
_LEXICAL_PATTERN = re.compile(
    r"https?://[^\s，。！？、；：））」』]+"
    r"|[A-Za-z0-9._+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    r"|\b[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}\b"
    r"|(?<!\d)\d+(?:[.,]\d+)+(?!\d)"
    r"|\b[A-Za-z]+(?:['’\-][A-Za-z]+)+\b"
)


def content_key(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum())


def ends_sentence(text: str) -> bool:
    return any(not text[match.end():].strip() for match in _SENTENCE_END.finditer(text))


def strip_display_punctuation(text: str) -> str:
    """Remove sentence punctuation, not decimal points, email/URL syntax or word spelling."""
    protected: set[int] = set()
    for match in _LEXICAL_PATTERN.finditer(text):
        end = match.end()
        if text[match.start():end].startswith(("http://", "https://")):
            while end > match.start() and text[end - 1] in ".!?;":
                end -= 1
        protected.update(range(match.start(), end))
    characters = [
        " " if index not in protected and unicodedata.category(char).startswith("P") else char
        for index, char in enumerate(text)
    ]
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def _sentence_parts(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        parts.append(text[start:match.end()])
        start = match.end()
    if text[start:].strip():
        parts.append(text[start:])
    elif parts:
        parts[-1] += text[start:]
    return parts or [text]


def _paired_parts(segment: ASRDataSeg) -> tuple[list[str], list[str]] | None:
    source = _sentence_parts(segment.text)
    target = _sentence_parts(segment.translated_text) if segment.translated_text else [""] * len(source)
    # A verb suffix following punctuation is not a new dialogue turn.
    index = 1
    while len(source) > len(target) and index < len(source):
        if re.fullmatch(r"\s*(?:して(?:ます|います)|です|ます)[。.!！?？]*\s*", source[index]):
            source[index - 1] += source.pop(index)
        elif index == 1 and content_key(source[0]) in {"じゃん", "じゃーん"}:
            source[0] += source.pop(1)
        else:
            index += 1
    if len(source) != len(target) or len(source) < 2:
        return None
    if any(content_key(part) in {"これ", "それ", "あれ", "この", "その", "あの"} for part in source[:-1]):
        return None
    return source, target


def _split_segment(
    segment: ASRDataSeg,
    timing_tokens: list[ASRDataSeg],
    events: list[dict[str, Any]],
    display_extension_ms: int,
) -> list[ASRDataSeg]:
    paired = _paired_parts(segment)
    if paired is None:
        return [segment]
    source, target = paired
    duration = segment.end_time - segment.start_time
    if duration < len(source) * 120 or duration + display_extension_ms < len(source) * 500:
        return [segment]
    minimum = min(500, duration // len(source))
    lengths = [max(1, len(content_key(part))) for part in source]
    total = sum(lengths)
    tokens = sorted(
        (token for token in timing_tokens if token.end_time > segment.start_time and token.start_time < segment.end_time),
        key=lambda token: (token.start_time, token.end_time),
    )
    token_text = "".join(content_key(token.text) for token in tokens)
    source_key = "".join(content_key(part) for part in source)
    match_start = token_text.find(source_key) if source_key else -1
    token_ends: dict[int, int] = {}
    offset = 0
    for token in tokens:
        offset += len(content_key(token.text))
        if token.end_time > token.start_time:
            token_ends[offset] = token.end_time
    boundaries = [segment.start_time]
    cumulative = 0
    methods = []
    for index, length in enumerate(lengths[:-1]):
        cumulative += length
        boundary = segment.start_time + round(duration * cumulative / total)
        method = "proportional-display"
        if match_start >= 0 and match_start + cumulative in token_ends:
            boundary = token_ends[match_start + cumulative]
            method = "aligned-token-end"
        bounded = max(
            boundaries[-1] + minimum,
            min(boundary, segment.end_time - minimum * (len(source) - index - 1)),
        )
        if bounded != boundary and method == "aligned-token-end":
            method = "aligned-token-end-adjusted-display"
        boundaries.append(bounded)
        methods.append(method)
    boundaries.append(segment.end_time)
    events.append({
        "operation": "sentence_split", "source": segment.text, "count": len(source), "timing_methods": methods,
    })
    return [
        ASRDataSeg(original, boundaries[index], boundaries[index + 1], translated_text=translation)
        for index, (original, translation) in enumerate(zip(source, target, strict=True))
    ]


def _can_join(left: ASRDataSeg, right: ASRDataSeg) -> bool:
    if is_dialogue_standalone_response(left.text) or is_dialogue_standalone_response(right.text):
        return False
    if ends_sentence(left.text):
        return False
    if right.start_time - left.end_time > 3000 or right.start_time < left.start_time:
        return False
    if right.end_time - left.start_time > 8000:
        return False
    source_length = len(content_key(left.text + right.text))
    target_length = len(content_key(left.translated_text + right.translated_text))
    if max(source_length, target_length) > 40:
        return False
    return (
        content_key(left.text).endswith(_SOURCE_TAILS)
        or content_key(right.text).startswith(_SOURCE_HEADS)
        or content_key(left.translated_text).endswith(_TARGET_TAILS)
        or content_key(right.translated_text) in {"和", "与", "的", "然后"}
    )


def reflow_subtitle_segments(
    segments: list[ASRDataSeg],
    *,
    timing_tokens: list[ASRDataSeg] | None = None,
    events: list[dict[str, Any]] | None = None,
    display_extension_ms: int = 0,
) -> list[ASRDataSeg]:
    """Only change cue boundaries: source and translation content/order must survive."""
    events = events if events is not None else []
    split: list[ASRDataSeg] = []
    for segment in segments:
        clone = ASRDataSeg(segment.text, segment.start_time, segment.end_time, segment.translated_text)
        split.extend(_split_segment(clone, timing_tokens or [], events, display_extension_ms))
    result: list[ASRDataSeg] = []
    for segment in split:
        if result and _can_join(result[-1], segment):
            left = result.pop()
            events.append({"operation": "fragment_merge", "left": left.text, "right": segment.text})
            result.append(ASRDataSeg(
                left.text + segment.text,
                left.start_time,
                max(left.end_time, segment.end_time),
                left.translated_text + segment.translated_text,
            ))
        else:
            result.append(segment)
    for field in ("text", "translated_text"):
        before = content_key("".join(getattr(segment, field) for segment in segments))
        after = content_key("".join(getattr(segment, field) for segment in result))
        if before != after:
            raise RuntimeError(f"Subtitle layout changed {field} content")
    return result
