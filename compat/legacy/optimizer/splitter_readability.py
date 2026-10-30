"""Readability and short-response rules for subtitle splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass

from optimizer.asr_data import ASRDataSeg
from optimizer.text_utils import count_words


@dataclass(frozen=True)
class ReadabilityRuleConfig:
    filler_merge_max_gap: int = 220
    filler_merge_max_cjk: int = 24
    readability_merge_max_gap: int = 80
    readability_merge_max_cjk: int = 32
    readability_min_duration: int = 500
    tail_fragment_merge_max_gap: int = 8000


DEFAULT_READABILITY_RULES = ReadabilityRuleConfig()

FILLER_MERGE_MAX_GAP = DEFAULT_READABILITY_RULES.filler_merge_max_gap
FILLER_MERGE_MAX_CJK = DEFAULT_READABILITY_RULES.filler_merge_max_cjk
READABILITY_MERGE_MAX_GAP = DEFAULT_READABILITY_RULES.readability_merge_max_gap
READABILITY_MERGE_MAX_CJK = DEFAULT_READABILITY_RULES.readability_merge_max_cjk
READABILITY_MIN_DURATION = DEFAULT_READABILITY_RULES.readability_min_duration
TAIL_FRAGMENT_MERGE_MAX_GAP = DEFAULT_READABILITY_RULES.tail_fragment_merge_max_gap

_MATCH_PUNCTUATION = re.compile(r"[\u3002\u3001\uff0c\uff01\uff1f,.!?;:\uff1b\uff1a\u30fb\s]")
_FILLER_ONLY_TOKENS = (
    "\u3042\u306e",
    "\u307e\u3042",
    "\u306d",
    "\u3088",
    "\u3055",
)
_DIALOGUE_STANDALONE_RESPONSES = {
    "\u306f\u3044",
    "\u306f\u3044\u306f\u3044",
    "\u3046\u3093",
    "\u3046\u3046\u3093",
    "\u3048\u3048",
    "\u3044\u3084",
    "\u3044\u3044\u3048",
    "\u305d\u3046",
    "\u305d\u3046\u305d\u3046",
    "\u3042\u3042",
    "\u3048",
    "\u3042",
    "\u306a\u306b",
    "\u4f55",
    "\u306a\u3093\u3067",
    "\u3069\u3046\u3057\u3066",
}
_PROTECTED_SHORT_DISPLAY_RESPONSES = {
    "\u306f\u3044",
    "\u3046\u3093",
    "\u3046\u3046\u3093",
    "\u3048",
    "\u3042",
    "\u3044\u3084",
    "\u3044\u3044\u3048",
    "\u3060\u3081",
    "\u30c0\u30e1",
    "\u304a",
    "\u304a\u304a",
    "\u306f\u3042",
}
_SHORT_FILLER_PREFIXES = ("\u306d", "\u3048", "\u3046\u3093")
_READABILITY_SUFFIXES = (
    "\u3067\u3059",
    "\u3067\u3057\u305f",
    "\u307e\u3059",
    "\u307e\u3057\u305f",
    "\u3067\u3057\u305f",
    "\u3051\u3069",
    "\u3051\u308c\u3069",
    "\u3051\u308c\u3069\u3082",
    "\u304b\u3089",
    "\u306e\u3067",
    "\u306e\u306b",
    "\u3057",
    "\u3066",
    "\u305f",
    "\u3060",
    "\u3063\u3066",
    "\u3068\u3044\u3046",
    "\u3068\u304b",
    "\u306a\u3069",
    "\u306a",
    "\u306d",
    "\u3088",
    "\u308f",
    "\u304b",
    "\u306b",
    "\u3067",
    "\u3068",
    "\u304c",
    "\u3092",
    "\u306f",
    "\u3082",
    "\u306e",
    "\u3078",
    "\u304b\u3089",
    "\u306e\u3067",
    "\u3053\u3068",
)
_READABILITY_PREFIXES = (
    "\u3068\u3044\u3046\u3053\u3068\u3067",
    "\u305d\u3057\u3066",
    "\u305d\u308c\u3067",
    "\u3060\u304b\u3089",
    "\u3042\u3068",
    "\u3067\u3082",
    "\u3058\u3083\u3042",
    "\u305f\u3060",
    "\u307e\u305f",
    "\u306a\u306e\u3067",
    "\u3059\u308b\u3068",
    "\u3068\u3053\u308d\u3067",
    "\u3061\u306a\u307f\u306b",
)
_READABILITY_STANDALONE_WEAK = (
    "\u3054\u3056\u3044\u307e\u3059",
    "\u3088\u308d\u3057\u304f",
    "\u304a\u9858\u3044\u3057\u307e\u3059",
)
_NUMERIC_UNIT_FRAGMENTS = {
    "\u5ea6",
    "\u65e5",
    "\u6708",
    "\u5e74",
    "\u6642",
    "\u5206",
    "\u79d2",
    "\u500b",
    "\u56de",
    "\u4eba",
    "\u679a",
    "\u672c",
}
_READABILITY_TRAILING_FRAGMENTS = (
    "\u3093\u3067\u3059",
    "\u3093\u3067\u3059\u304c",
    "\u3067\u3057\u305f\u304c",
    "\u307e\u3059\u304c",
    "\u3051\u3069",
    "\u3051\u308c\u3069",
    "\u3051\u308c\u3069\u3082",
    "\u304b\u3089",
    "\u306e\u3067",
    "\u306e\u306b",
    "\u3057",
    "\u3066",
    "\u3063\u3066",
    "\u3068\u3044\u3046",
    "\u3068\u304b",
)
_READABILITY_LEADING_FRAGMENTS = (
    "\u305d\u3057\u3066",
    "\u305d\u308c\u3067",
    "\u3060\u304b\u3089",
    "\u3067\u3082",
    "\u3058\u3083\u3042",
    "\u305f\u3060",
    "\u307e\u305f",
    "\u306a\u306e\u3067",
    "\u3059\u308b\u3068",
    "\u3068\u3053\u308d\u3067",
    "\u3061\u306a\u307f\u306b",
)


def normalize_filler_text(text: str) -> str:
    """Normalize a candidate subtitle fragment for filler-word checks."""
    return _MATCH_PUNCTUATION.sub("", text.strip())


def is_filler_only(text: str) -> bool:
    """Return whether text is only short spoken filler particles."""
    normalized = normalize_filler_text(text)
    if is_dialogue_standalone_response(normalized):
        return False
    if not normalized or count_words(normalized) > 6:
        return False

    remaining = normalized
    tokens = sorted(_FILLER_ONLY_TOKENS, key=len, reverse=True)
    while remaining:
        for token in tokens:
            if remaining.startswith(token):
                remaining = remaining[len(token):]
                break
        else:
            return False
    return True


def is_dialogue_standalone_response(text: str) -> bool:
    """Return whether a short fragment is likely a separate dialogue turn."""
    normalized = normalize_filler_text(text)
    return normalized in _DIALOGUE_STANDALONE_RESPONSES


def starts_with_short_filler(text: str) -> bool:
    """Return whether a very short fragment starts with a filler particle."""
    normalized = normalize_filler_text(text)
    if not normalized or is_filler_only(normalized) or count_words(normalized) > 4:
        return False
    return any(normalized.startswith(prefix) for prefix in _SHORT_FILLER_PREFIXES)


def can_merge_filler(
    left: ASRDataSeg,
    right: ASRDataSeg,
    config: ReadabilityRuleConfig = DEFAULT_READABILITY_RULES,
) -> bool:
    """Check whether merging adjacent filler-related fragments is conservative."""
    if is_dialogue_standalone_response(left.text) or is_dialogue_standalone_response(
        right.text
    ):
        return False
    gap = max(0, right.start_time - left.end_time)
    merged_text = f"{left.text}{right.text}"
    return gap <= config.filler_merge_max_gap and count_words(
        merged_text
    ) <= config.filler_merge_max_cjk


def segment_duration(seg: ASRDataSeg) -> int:
    """Return non-negative segment duration in milliseconds."""
    return max(0, seg.end_time - seg.start_time)


def is_numeric_fragment(text: str) -> bool:
    """Return whether a fragment is part of a compact numeric expression."""
    normalized = normalize_filler_text(text)
    return bool(normalized) and (
        normalized.isdigit() or normalized in _NUMERIC_UNIT_FRAGMENTS
    )


def is_readability_short(
    seg: ASRDataSeg,
    config: ReadabilityRuleConfig = DEFAULT_READABILITY_RULES,
) -> bool:
    """Return whether a segment is a structural fragment, not merely short."""
    text = normalize_filler_text(seg.text)
    if is_dialogue_standalone_response(text):
        return False
    duration = segment_duration(seg)
    return (
        duration < config.readability_min_duration
        or text in _READABILITY_STANDALONE_WEAK
        or is_structural_readability_fragment(text)
    )


def is_structural_readability_fragment(text: str) -> bool:
    """Return whether text is a mergeable structural display fragment."""
    normalized = normalize_filler_text(text)
    return (
        normalized in _READABILITY_SUFFIXES
        or normalized in _READABILITY_PREFIXES
        or normalized.endswith(_READABILITY_TRAILING_FRAGMENTS)
        or normalized.startswith(_READABILITY_LEADING_FRAGMENTS)
        or is_numeric_fragment(normalized)
    )


def is_tail_fragment(text: str) -> bool:
    """Return whether text is a dangling Japanese tail fragment."""
    normalized = normalize_filler_text(text)
    if (
        not normalized
        or is_dialogue_standalone_response(normalized)
        or is_protected_short_utterance(normalized)
        or count_words(normalized) > 3
    ):
        return False
    if is_numeric_fragment(normalized):
        return False
    return normalized in _READABILITY_SUFFIXES


def is_protected_short_utterance(text: str) -> bool:
    """Protect short complete utterances from readability smoothing."""
    normalized = normalize_filler_text(text)
    if not normalized or count_words(normalized) > 4:
        return False
    if is_numeric_fragment(normalized) or is_filler_only(normalized):
        return False
    if normalized in _READABILITY_STANDALONE_WEAK:
        return False
    if normalized in _READABILITY_SUFFIXES or normalized in _READABILITY_PREFIXES:
        return False
    if normalized.endswith(_READABILITY_TRAILING_FRAGMENTS):
        return False
    if normalized.startswith(_READABILITY_LEADING_FRAGMENTS):
        return False
    return True


def is_protected_short_display_response(text: str) -> bool:
    normalized = normalize_filler_text(text)
    return normalized in _PROTECTED_SHORT_DISPLAY_RESPONSES


def can_merge_readability(
    left: ASRDataSeg,
    right: ASRDataSeg,
    config: ReadabilityRuleConfig = DEFAULT_READABILITY_RULES,
) -> bool:
    """Check whether merging short display fragments keeps subtitle size sane."""
    left_structural = is_structural_readability_fragment(left.text)
    right_structural = is_structural_readability_fragment(right.text)
    if (
        is_dialogue_standalone_response(left.text)
        or is_dialogue_standalone_response(right.text)
        or (is_protected_short_utterance(left.text) and not right_structural)
        or is_protected_short_utterance(right.text)
    ):
        return False
    gap = max(0, right.start_time - left.end_time)
    merged_text = f"{left.text}{right.text}"
    gap_limit = (
        config.tail_fragment_merge_max_gap
        if left_structural or right_structural
        else config.readability_merge_max_gap
    )
    return gap <= gap_limit and count_words(merged_text) <= config.readability_merge_max_cjk


def prefer_merge_next(
    prev_seg: ASRDataSeg | None,
    current: ASRDataSeg,
    next_seg: ASRDataSeg | None,
) -> bool:
    """Choose merge direction for a short fragment."""
    if next_seg is None:
        return False
    if prev_seg is None:
        return True

    current_text = normalize_filler_text(current.text)
    next_text = normalize_filler_text(next_seg.text)
    if is_numeric_fragment(current_text) and is_numeric_fragment(next_text):
        return True
    if (
        current_text in _READABILITY_PREFIXES
        or current_text.startswith(_READABILITY_LEADING_FRAGMENTS)
    ):
        return True
    if (
        current_text in _READABILITY_SUFFIXES
        or current_text.endswith(_READABILITY_TRAILING_FRAGMENTS)
    ):
        return False

    prev_gap = max(0, current.start_time - prev_seg.end_time)
    next_gap = max(0, next_seg.start_time - current.end_time)
    return next_gap + 100 < prev_gap
