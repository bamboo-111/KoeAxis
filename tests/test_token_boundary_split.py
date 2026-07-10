from __future__ import annotations

import pytest

from optimizer.token_boundary_split import (
    _clamp_overlapping_segments,  # noqa: PLC2701
    _fallback_token_groups,  # noqa: PLC2701
    _records_from_aligned_item,  # noqa: PLC2701
    _records_to_segments,  # noqa: PLC2701
    _token_counts_to_segments,  # noqa: PLC2701
    _token_groups_to_segments,  # noqa: PLC2701
    parse_end_ids,
    parse_token_counts_output,
    parse_token_delimited_output,
)
from optimizer.asr_data import ASRDataSeg


def test_parse_end_ids_requires_increasing_known_ids_and_last_token() -> None:
    allowed = ["A0", "A1", "A2", "A3"]

    assert parse_end_ids("END=A1,A3", allowed) == ["A1", "A3"]

    with pytest.raises(ValueError):
        parse_end_ids("END=A2,A1,A3", allowed)
    with pytest.raises(ValueError):
        parse_end_ids("END=A1,A4", allowed)
    with pytest.raises(ValueError):
        parse_end_ids("END=A1,A2", allowed)


def test_records_to_segments_uses_token_boundaries() -> None:
    item = {
        "segment_id": "segment_000001",
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": 1.0,
        "tokens": [
            {"text": "それ", "start_time": 0.0, "end_time": 0.2},
            {"text": "じゃ", "start_time": 0.2, "end_time": 0.3},
            {"text": "はい", "start_time": 0.5, "end_time": 0.7},
        ],
    }

    records = _records_from_aligned_item(item)
    result = _records_to_segments(records, ["A1", "A2"])

    assert [segment.text for segment in result] == ["それじゃ", "はい"]
    assert [(segment.start_time, segment.end_time) for segment in result] == [(0, 300), (500, 700)]


def test_parse_token_delimited_output_preserves_exact_tokens() -> None:
    expected = ["それ", "じゃ", "はい"]

    assert parse_token_delimited_output("それ|じゃ<br>はい", expected) == [["それ", "じゃ"], ["はい"]]

    with pytest.raises(ValueError):
        parse_token_delimited_output("それ|じゃ<br>いいえ", expected)
    with pytest.raises(ValueError):
        parse_token_delimited_output("それ|じゃ", expected)


def test_parse_token_counts_output_requires_positive_counts_and_exact_sum() -> None:
    assert parse_token_counts_output("COUNTS=2,1", 3) == [2, 1]

    with pytest.raises(ValueError):
        parse_token_counts_output("2,1", 3)
    with pytest.raises(ValueError):
        parse_token_counts_output("COUNTS=2,0,1", 3)
    with pytest.raises(ValueError):
        parse_token_counts_output("COUNTS=2,2", 3)


def test_token_groups_to_segments_uses_group_boundaries() -> None:
    item = {
        "segment_id": "segment_000001",
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": 1.0,
        "tokens": [
            {"text": "それ", "start_time": 0.0, "end_time": 0.2},
            {"text": "じゃ", "start_time": 0.2, "end_time": 0.3},
            {"text": "はい", "start_time": 0.5, "end_time": 0.7},
        ],
    }

    records = _records_from_aligned_item(item)
    result = _token_groups_to_segments(records, [["それ", "じゃ"], ["はい"]])

    assert [segment.text for segment in result] == ["それじゃ", "はい"]
    assert [(segment.start_time, segment.end_time) for segment in result] == [(0, 300), (500, 700)]


def test_token_counts_to_segments_uses_count_boundaries() -> None:
    item = {
        "segment_id": "segment_000001",
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": 1.0,
        "tokens": [
            {"text": "それ", "start_time": 0.0, "end_time": 0.2},
            {"text": "じゃ", "start_time": 0.2, "end_time": 0.3},
            {"text": "はい", "start_time": 0.5, "end_time": 0.7},
        ],
    }

    records = _records_from_aligned_item(item)
    result = _token_counts_to_segments(records, [2, 1])

    assert [segment.text for segment in result] == ["それじゃ", "はい"]
    assert [(segment.start_time, segment.end_time) for segment in result] == [(0, 300), (500, 700)]


def test_fallback_token_groups_preserves_all_tokens() -> None:
    item = {
        "segment_id": "segment_000001",
        "status": "completed",
        "global_start_time": 0.0,
        "global_end_time": 1.0,
        "tokens": [
            {"text": "あ", "start_time": 0.0, "end_time": 0.1},
            {"text": "い", "start_time": 0.1, "end_time": 0.2},
            {"text": "う", "start_time": 0.7, "end_time": 0.8},
            {"text": "え", "start_time": 0.8, "end_time": 0.9},
        ],
    }

    records = _records_from_aligned_item(item)
    groups = _fallback_token_groups(records, max_word_count_cjk=3, max_word_count_english=3)

    assert [token for group in groups for token in group] == ["あ", "い", "う", "え"]
    assert len(groups) >= 2


def test_unreliable_zero_duration_tokens_get_proportional_fallback() -> None:
    item = {
        "segment_id": "segment_000001",
        "status": "completed",
        "global_start_time": 10.0,
        "global_end_time": 11.0,
        "tokens": [
            {"text": "あ", "start_time": 10.0, "end_time": 10.0},
            {"text": "れ", "start_time": 10.0, "end_time": 10.0},
            {"text": "だ", "start_time": 10.0, "end_time": 10.0},
            {"text": "よ", "start_time": 10.0, "end_time": 10.0},
        ],
    }

    records = _records_from_aligned_item(item)

    assert [record.duration_tick for record in records] == [None, None, None, None]
    assert records[0].start_ms == 10000
    assert records[-1].end_ms == 11000


def test_clamp_overlapping_segments_keeps_order_and_positive_duration() -> None:
    segments = [
        ASRDataSeg("a", 0, 1000),
        ASRDataSeg("b", 900, 1200),
        ASRDataSeg("c", 1100, 1300),
    ]

    result = _clamp_overlapping_segments(segments)

    assert [(item.text, item.start_time, item.end_time) for item in result] == [
        ("a", 0, 900),
        ("b", 900, 1100),
        ("c", 1100, 1300),
    ]


def test_clamp_overlapping_segments_does_not_reorder_text() -> None:
    segments = [
        ASRDataSeg("first", 100, 300),
        ASRDataSeg("second", 0, 200),
    ]

    result = _clamp_overlapping_segments(segments)

    assert [item.text for item in result] == ["first", "second"]
    assert result[1].start_time == 300
    assert result[1].end_time == 301
