from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import _extend_protected_short_display_durations
from optimizer.splitter_display import (
    DisplayDurationConfig,
    cap_long_display_durations,
    extend_protected_short_display_durations,
    merge_zero_gap_short_display_fragments,
    minimum_display_duration,
)


def test_protected_short_display_duration_extends_only_into_gap() -> None:
    segments = [
        ASRDataSeg("\u524d\u3067\u3059\u3002", 0, 1000),
        ASRDataSeg("\u3046\u3093\u3002", 1100, 1101),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 1300, 1800),
    ]

    result = extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u524d\u3067\u3059\u3002", 0, 1000),
        ("\u3046\u3093\u3002", 1100, 1220),
        ("\u6b21\u3067\u3059\u3002", 1300, 1800),
    ]


def test_zero_gap_ordinary_fragment_merges_with_following_clause() -> None:
    segments = [
        ASRDataSeg("\u30d0\u30b6\u30fc\u30eb\u306f\u3069\u3046", 851215, 851614),
        ASRDataSeg("\u3067\u3057\u305f\u304b\u3002", 851614, 852114),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 852654, 853774),
    ]

    result = merge_zero_gap_short_display_fragments(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        (
            "\u30d0\u30b6\u30fc\u30eb\u306f\u3069\u3046"
            "\u3067\u3057\u305f\u304b\u3002",
            851215,
            852114,
        ),
        ("\u6b21\u3067\u3059\u3002", 852654, 853774),
    ]


def test_display_config_can_raise_minimum_duration() -> None:
    segment = ASRDataSeg("\u4e09\u65e5", 100, 300)
    config = DisplayDurationConfig(ordinary_subtitle_min_duration=700)

    assert minimum_display_duration(segment, config) == 700


def test_long_display_duration_is_capped_by_config() -> None:
    segments = [
        ASRDataSeg("\u3042\u3042\u3002", 83129, 98129),
    ]
    config = DisplayDurationConfig(max_subtitle_display_duration=3000)

    cap_long_display_durations(segments, config)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in segments] == [
        ("\u3042\u3042\u3002", 83129, 86129),
    ]


def test_splitter_compatibility_alias_uses_display_module() -> None:
    segments = [
        ASRDataSeg("\u524d\u3067\u3059\u3002", 0, 1000),
        ASRDataSeg("\u4e09\u65e5", 1100, 1420),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 1800, 2400),
    ]

    result = _extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u524d\u3067\u3059\u3002", 0, 1000),
        ("\u4e09\u65e5", 1100, 1600),
        ("\u6b21\u3067\u3059\u3002", 1800, 2400),
    ]
