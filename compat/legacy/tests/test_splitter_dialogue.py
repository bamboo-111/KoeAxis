from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import SubtitleSplitter
from optimizer.splitter import _extend_protected_short_display_durations


def _splitter() -> SubtitleSplitter:
    return SubtitleSplitter(thread_num=1, model="", base_url="", api_key="")


def test_dialogue_short_response_is_not_merged_as_filler() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("それじゃ早速練習始めるわよ", 0, 1200),
            ASRDataSeg("はい", 1300, 1500),
            ASRDataSeg("ママちゃんどうしたの", 1600, 2600),
        ]

        result = splitter._smooth_short_fillers(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "それじゃ早速練習始めるわよ",
            "はい",
            "ママちゃんどうしたの",
        ]
    finally:
        splitter.stop()


def test_dialogue_short_response_is_not_merged_for_readability() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("聞こえてますか", 0, 900),
            ASRDataSeg("うん", 950, 1100),
            ASRDataSeg("大丈夫", 1160, 1700),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "聞こえてますか",
            "うん",
            "大丈夫",
        ]
    finally:
        splitter.stop()


def test_non_dialogue_filler_can_still_merge() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("今日は", 0, 500),
            ASRDataSeg("あの", 520, 650),
        ]

        result = splitter._smooth_short_fillers(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == ["今日はあの"]
    finally:
        splitter.stop()


def test_complete_short_utterance_is_not_merged_for_readability() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("ちょっと待って", 0, 700),
            ASRDataSeg("大丈夫", 760, 1100),
            ASRDataSeg("先に行くね", 1160, 1900),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "ちょっと待って",
            "大丈夫",
            "先に行くね",
        ]
    finally:
        splitter.stop()


def test_short_sentence_starting_with_filler_is_not_merged_left() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("聞こえてる", 0, 800),
            ASRDataSeg("うん大丈夫", 850, 1300),
        ]

        result = splitter._smooth_short_fillers(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == ["聞こえてる", "うん大丈夫"]
    finally:
        splitter.stop()


def test_structural_fragment_can_still_merge_for_readability() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("行きたいんだ", 0, 800),
            ASRDataSeg("けど", 840, 980),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == ["行きたいんだけど"]
    finally:
        splitter.stop()


def test_rule_split_uses_strong_sentence_boundaries_even_when_short() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("はい、", 0, 300),
            ASRDataSeg("お守りします。", 300, 900),
            ASRDataSeg("それでは", 900, 1300),
            ASRDataSeg("奥様。", 1300, 1800),
        ]

        result = splitter._process_by_rules(segments)  # noqa: SLF001

        assert [(segment.text, segment.start_time, segment.end_time) for segment in result] == [
            ("はい、お守りします。", 0, 900),
            ("それでは奥様。", 900, 1800),
        ]
        assert "".join(segment.text for segment in result) == "".join(
            segment.text for segment in segments
        )
    finally:
        splitter.stop()


def test_rule_split_splits_inline_strong_sentence_boundaries() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("まだ生きていらっしゃる。そうよ。まだ残されているの。", 0, 3000),
        ]

        result = splitter._process_by_rules(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "まだ生きていらっしゃる。",
            "そうよ。",
            "まだ残されているの。",
        ]
        assert result[1].end_time - result[1].start_time >= 1500
        assert all(
            left.end_time <= right.start_time
            for left, right in zip(result, result[1:], strict=False)
        )
    finally:
        splitter.stop()


def test_rule_split_splits_short_response_before_first_person_clause() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("はい、私星野先生の代わり。", 0, 1600),
        ]

        result = splitter._process_by_rules(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "はい、",
            "私星野先生の代わり。",
        ]
    finally:
        splitter.stop()


def test_tail_fragment_merges_left_across_large_token_gap() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("\u3053\u308c\u3059\u3054\u3044\u9762\u767d\u304b\u3063", 261510, 263510),
            ASRDataSeg("\u305f\u3002", 270950, 271270),
            ASRDataSeg("\u7d9a\u304d\u306a\u3044\u306e\uff1f", 271910, 272550),
        ]

        result = splitter._merge_tail_fragments(segments)  # noqa: SLF001

        assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
            ("\u3053\u308c\u3059\u3054\u3044\u9762\u767d\u304b\u3063\u305f\u3002", 261510, 271270),
            ("\u7d9a\u304d\u306a\u3044\u306e\uff1f", 271910, 272550),
        ]
    finally:
        splitter.stop()


def test_tail_fragment_merge_preserves_protected_short_response() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("\u5b8c\u6210\u3057\u305f\u3089\u898b\u305b\u3066\u3002", 534372, 537572),
            ASRDataSeg("\u306f\u3044\u3002", 540117, 540437),
        ]

        result = splitter._merge_tail_fragments(segments)  # noqa: SLF001

        assert [seg.text for seg in result] == [
            "\u5b8c\u6210\u3057\u305f\u3089\u898b\u305b\u3066\u3002",
            "\u306f\u3044\u3002",
        ]
    finally:
        splitter.stop()


def test_tail_fragment_merge_does_not_merge_complete_imperative() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("\u8ab0\u304b\u304c\u79c1\u306b\u5bb6\u3092\u8ffd\u3044\u304b\u3051\u308b\u304b\u3057\u3089\u3002", 1200, 2400),
            ASRDataSeg("\u5f85\u3066\u3002", 8413, 8733),
        ]

        result = splitter._merge_tail_fragments(segments)  # noqa: SLF001

        assert [seg.text for seg in result] == [
            "\u8ab0\u304b\u304c\u79c1\u306b\u5bb6\u3092\u8ffd\u3044\u304b\u3051\u308b\u304b\u3057\u3089\u3002",
            "\u5f85\u3066\u3002",
        ]
    finally:
        splitter.stop()


def test_structural_readability_fragment_merges_across_token_gap() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("\u305d\u308c", 1000, 1800),
            ASRDataSeg("\u306b", 4200, 4280),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
            ("\u305d\u308c\u306b", 1000, 4280),
        ]
    finally:
        splitter.stop()


def test_protected_short_display_duration_extends_only_into_gap() -> None:
    segments = [
        ASRDataSeg("\u524d\u3067\u3059\u3002", 0, 1000),
        ASRDataSeg("\u3046\u3093\u3002", 1100, 1101),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 1300, 1800),
    ]

    result = _extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u524d\u3067\u3059\u3002", 0, 1000),
        ("\u3046\u3093\u3002", 1100, 1220),
        ("\u6b21\u3067\u3059\u3002", 1300, 1800),
    ]


def test_ordinary_short_display_duration_extends_to_readable_minimum() -> None:
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


def test_zero_gap_ordinary_fragment_merges_with_following_clause() -> None:
    segments = [
        ASRDataSeg("\u30d0\u30b6\u30fc\u30eb\u306f\u3069\u3046", 851215, 851614),
        ASRDataSeg("\u3067\u3057\u305f\u304b\u3002", 851614, 852114),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 852654, 853774),
    ]

    result = _extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        (
            "\u30d0\u30b6\u30fc\u30eb\u306f\u3069\u3046"
            "\u3067\u3057\u305f\u304b\u3002",
            851215,
            852114,
        ),
        ("\u6b21\u3067\u3059\u3002", 852654, 853774),
    ]


def test_zero_gap_protected_short_block_merges_then_extends() -> None:
    segments = [
        ASRDataSeg("\u898b\u3066\u3082\u3044\u3044\u666f\u8272\u3060\u306a\u3002", 576521, 582601),
        ASRDataSeg("\u306f\u3042\u3002", 582601, 582681),
        ASRDataSeg("\u304a\u304a", 582681, 582801),
        ASRDataSeg("\u8d64\u670d\u3060\u3002", 583801, 584601),
    ]

    result = _extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u898b\u3066\u3082\u3044\u3044\u666f\u8272\u3060\u306a\u3002", 576521, 582601),
        ("\u306f\u3042\u3002\u304a\u304a", 582601, 583101),
        ("\u8d64\u670d\u3060\u3002", 583801, 584601),
    ]


def test_zero_gap_short_fragment_borrows_boundary_from_long_following_clause() -> None:
    segments = [
        ASRDataSeg("\u66f8\u304d\u5c3d\u304f\u3059\u307e\u3067\u3002", 1327426, 1335346),
        ASRDataSeg("\u3042\u3052\u305f\u3002", 1335346, 1335704),
        ASRDataSeg(
            "\u77e5\u3089\u306d\u5834\u6240\u306e\u77e5\u3089\u306d\u672a\u6765"
            "\u306e\u77e5\u3089\u306d\u3042\u306a\u305f\u306b"
            "\u7a81\u304d\u523a\u3055\u3063\u3066\u308b\u6c17\u306a\u304f\u306a\u3063\u3066",
            1335704,
            1346824,
        ),
    ]

    result = _extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u66f8\u304d\u5c3d\u304f\u3059\u307e\u3067\u3002", 1327426, 1335346),
        ("\u3042\u3052\u305f\u3002", 1335346, 1335846),
        (
            "\u77e5\u3089\u306d\u5834\u6240\u306e\u77e5\u3089\u306d\u672a\u6765"
            "\u306e\u77e5\u3089\u306d\u3042\u306a\u305f\u306b"
            "\u7a81\u304d\u523a\u3055\u3063\u3066\u308b\u6c17\u306a\u304f\u306a\u3063\u3066",
            1335846,
            1343846,
        ),
    ]


def test_long_display_duration_is_capped_without_text_change() -> None:
    segments = [
        ASRDataSeg("\u3042\u3042\u3002", 83129, 98129),
        ASRDataSeg("\u6b21\u3067\u3059\u3002", 100000, 101000),
    ]

    result = _extend_protected_short_display_durations(segments)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
        ("\u3042\u3042\u3002", 83129, 91129),
        ("\u6b21\u3067\u3059\u3002", 100000, 101000),
    ]


def test_readability_smoothing_preserves_complete_short_phrase() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("\u63a2\u3057\u3055", 290472, 290792),
            ASRDataSeg("\u8ff7\u3063\u3066\u3044\u308b\u3002", 291512, 291992),
            ASRDataSeg("\u3042\u306a\u305f", 291992, 292152),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001
        result = splitter._merge_tail_fragments(result)  # noqa: SLF001
        result = splitter._smooth_readability_segments(result)  # noqa: SLF001

        assert [(seg.text, seg.start_time, seg.end_time) for seg in result] == [
            ("\u63a2\u3057\u3055", 290472, 290792),
            ("\u8ff7\u3063\u3066\u3044\u308b\u3002", 291512, 291992),
            ("\u3042\u306a\u305f", 291992, 292152),
        ]
    finally:
        splitter.stop()
