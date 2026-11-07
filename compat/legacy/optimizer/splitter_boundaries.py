"""Inline text boundary rules for subtitle splitting."""

from __future__ import annotations

import re
from typing import List

from optimizer.splitter_readability import normalize_filler_text
from optimizer.text_utils import count_words


STRONG_SENTENCE_END = re.compile(r"[\u3002\uff01\uff1f!?][\u300d\u300f\uff09\u300b\u3011]*\s*$")
INLINE_STRONG_BOUNDARY = re.compile(
    r"([\u3002\uff01\uff1f!?]+[\u300d\u300f\uff09\u300b\u3011]*)(?=\S)"
)
INLINE_SHORT_RESPONSE_BOUNDARY = re.compile(
    r"^(\s*(?:\u306f\u3044|\u3046\u3093|\u3048\u3048|\u3044\u3044\u3048|\u3044\u3084)[\u3001,]+)"
)
SHORT_RESPONSE_SPLIT_FOLLOWERS = (
    "\u79c1",
    "\u50d5",
    "\u4ffa",
    "\u3042\u305f\u3057",
    "\u308f\u305f\u3057",
)


def inline_dialogue_parts(text: str) -> List[str]:
    parts = split_inline_strong_boundaries(text)
    result: List[str] = []
    for part in parts:
        result.extend(split_inline_short_response_boundary(part))
    return [part for part in result if part.strip()]


def split_inline_strong_boundaries(text: str) -> List[str]:
    parts: List[str] = []
    cursor = 0
    for match in INLINE_STRONG_BOUNDARY.finditer(text):
        end = match.end()
        part = text[cursor:end]
        if part.strip():
            parts.append(part)
        cursor = end
    tail = text[cursor:]
    if tail.strip():
        parts.append(tail)
    return parts or [text]


def split_inline_short_response_boundary(text: str) -> List[str]:
    match = INLINE_SHORT_RESPONSE_BOUNDARY.match(text)
    if not match:
        return [text]
    first = match.group(1)
    rest = text[match.end() :]
    normalized_rest = normalize_filler_text(rest)
    if not normalized_rest.startswith(SHORT_RESPONSE_SPLIT_FOLLOWERS):
        return [text]
    if count_words(rest) < 3:
        return [text]
    return [first, rest]
