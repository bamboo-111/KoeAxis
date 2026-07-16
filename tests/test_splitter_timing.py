from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import _parts_to_timed_segments
from optimizer.splitter_timing import (
    InlineTimingConfig,
    extend_inline_short_utterances,
    parts_to_timed_segments,
    split_text_evenly_with_timing,
)


def test_parts_to_timed_segments_allocates_by_normalized_text_weight() -> None:
    segment = ASRDataSeg("\u3042\u3044\u3046\u3048\u304a\u304b\u304d\u304f\u3051\u3053", 100, 500)

    result = parts_to_timed_segments(
        segment,
        ["\u3042\u3044\u3046\u3048\u304a", "\u304b\u304d\u304f\u3051\u3053"],
    )

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u3042\u3044\u3046\u3048\u304a", 100, 300),
        ("\u304b\u304d\u304f\u3051\u3053", 300, 500),
    ]


def test_parts_to_timed_segments_preserves_one_millisecond_minimum_parts() -> None:
    segment = ASRDataSeg("abc", 10, 12)

    result = parts_to_timed_segments(segment, ["a", "b", "c"])

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("a", 10, 11),
        ("b", 11, 12),
        ("c", 12, 12),
    ]


def test_inline_short_utterance_borrows_from_following_segment() -> None:
    segments = [
        ASRDataSeg("\u305d\u3046\u3088\u3002", 0, 300),
        ASRDataSeg("\u307e\u3060\u6b8b\u3055\u308c\u3066\u3044\u308b\u306e\u3002", 300, 3000),
    ]

    result = extend_inline_short_utterances(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u305d\u3046\u3088\u3002", 0, 1500),
        ("\u307e\u3060\u6b8b\u3055\u308c\u3066\u3044\u308b\u306e\u3002", 1500, 3000),
    ]


def test_inline_timing_config_can_lower_short_utterance_minimum() -> None:
    segments = [
        ASRDataSeg("\u305d\u3046\u3088\u3002", 0, 300),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 300, 1000),
    ]
    config = InlineTimingConfig(short_utterance_min_duration=500)

    result = extend_inline_short_utterances(segments, config)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u305d\u3046\u3088\u3002", 0, 500),
        ("\u6b21\u3067\u3059\u3002", 500, 1000),
    ]


def test_splitter_compatibility_alias_uses_timing_module() -> None:
    segment = ASRDataSeg("\u3042\u3044\u3046\u3048\u304a\u304b\u304d\u304f\u3051\u3053", 100, 500)

    parts = ["\u3042\u3044\u3046\u3048\u304a", "\u304b\u304d\u304f\u3051\u3053"]
    legacy = _parts_to_timed_segments(segment, parts)
    direct = parts_to_timed_segments(segment, parts)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in legacy] == [
        (seg.text, seg.start_time, seg.end_time) for seg in direct
    ]


def test_split_text_evenly_with_timing_preserves_proportional_ranges() -> None:
    segment = ASRDataSeg("abcdefghij", 100, 500)

    result = split_text_evenly_with_timing(segment, 3)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("abc", 100, 233),
        ("def", 233, 366),
        ("ghij", 366, 500),
    ]
