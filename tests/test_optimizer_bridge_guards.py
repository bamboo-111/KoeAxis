from __future__ import annotations

import pytest

from optimizer.asr_data import ASRDataSeg
from qwen_asr.optimizer_bridge import (
    _extract_protected_short_responses as legacy_extract_protected_short_responses,
    _protected_short_response_segments as legacy_protected_short_response_segments,
    _segment_range_distance_ms as legacy_segment_range_distance_ms,
    _segments_normalized_text as legacy_segments_normalized_text,
    _validate_split_content_preserved as legacy_validate_split_content_preserved,
)
from qwen_asr.optimizer_bridge_adapter import _new_asr_data_seg, _normalize_content
from qwen_asr.optimizer_bridge_guards import (
    extract_protected_short_responses,
    protected_short_response_segments,
    segment_range_distance_ms,
    segments_normalized_text,
    validate_split_content_preserved,
)


def test_validate_split_content_preserved_rejects_deleted_text() -> None:
    with pytest.raises(RuntimeError, match="Split stage changed aligned text content"):
        validate_split_content_preserved(
            [ASRDataSeg("\u306f\u3044\u6b21", 0, 1000)],
            [ASRDataSeg("\u306f\u3044", 0, 500)],
            normalize_content=_normalize_content,
        )

    with pytest.raises(RuntimeError):
        legacy_validate_split_content_preserved(
            [ASRDataSeg("\u306f\u3044\u6b21", 0, 1000)],
            [ASRDataSeg("\u306f\u3044", 0, 500)],
        )


def test_protected_short_response_segments_filters_contextual_tokens() -> None:
    segments = [
        ASRDataSeg("\u524d", 0, 300),
        ASRDataSeg("\u306f\u3044", 350, 700),
        ASRDataSeg("\u6b21", 720, 1000),
        ASRDataSeg("\u30c0\u30e1\uff01", 2000, 2400),
    ]

    protected = protected_short_response_segments(segments, normalize_content=_normalize_content)

    assert [item["text"] for item in protected] == ["\u30c0\u30e1\uff01"]
    assert legacy_protected_short_response_segments(segments) == protected


def test_extract_protected_short_responses_splits_merged_segment_and_preserves_text() -> None:
    source = [
        ASRDataSeg("\u524d\u3067\u3059", 0, 700),
        ASRDataSeg("\u306f\u3044", 1200, 1450),
        ASRDataSeg("\u6b21\u3067\u3059", 2000, 2600),
    ]
    result = [ASRDataSeg("\u524d\u3067\u3059\u306f\u3044\u6b21\u3067\u3059", 0, 2600)]

    extracted = extract_protected_short_responses(
        source,
        result,
        normalize_content=_normalize_content,
        new_asr_data_seg=_new_asr_data_seg,
    )

    assert [(item.text, item.start_time, item.end_time) for item in extracted] == [
        ("\u524d\u3067\u3059", 0, 1200),
        ("\u306f\u3044", 1200, 1450),
        ("\u6b21\u3067\u3059", 1450, 2600),
    ]
    assert legacy_extract_protected_short_responses(source, result)[1].text == "\u306f\u3044"


def test_segments_normalized_text_and_distance_keep_legacy_aliases() -> None:
    segments = [ASRDataSeg(" A ", 0, 100), ASRDataSeg("b", 200, 300)]

    assert segments_normalized_text(segments, normalize_content=_normalize_content) == "ab"
    assert legacy_segments_normalized_text(segments) == "ab"
    assert segment_range_distance_ms(0, 100, 50, 150) == 0
    assert segment_range_distance_ms(0, 100, 150, 200) == 50
    assert legacy_segment_range_distance_ms(0, 100, 150, 200) == 50
