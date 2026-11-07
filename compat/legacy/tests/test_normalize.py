from __future__ import annotations

from optimizer.asr_data import ASRData, ASRDataSeg
from qwen_asr.normalize import NormalizeParams, normalize_asr_data


def test_normalize_merges_zero_duration_created_by_same_start_neighbor() -> None:
    source = ASRData(
        [
            ASRDataSeg("今踊ってみたいっぱい", 984160, 984161),
            ASRDataSeg("上がってるもんね", 984160, 984481),
        ]
    )

    result = normalize_asr_data(source, NormalizeParams(extend_ms=350, snap_gap_ms=200, min_blank_ms=300))

    assert len(result.segments) == 1
    assert result.segments[0].start_time == 984160
    assert result.segments[0].end_time > result.segments[0].start_time
    assert result.segments[0].text == "今踊ってみたいっぱい上がってるもんね"


def _assert_readable(result: ASRData) -> None:
    segments = result.segments
    assert all(seg.end_time - seg.start_time >= 500 for seg in segments), [
        (seg.text, seg.start_time, seg.end_time) for seg in segments
    ]
    assert all(left.end_time <= right.start_time for left, right in zip(segments, segments[1:], strict=False))


def test_normalize_short_cue_borrows_from_longer_previous_cue_when_next_has_no_slack() -> None:
    data = ASRData([
        ASRDataSeg("黄色いリバイだ。", 28210, 30500, translated_text="黄色的Revi。"),
        ASRDataSeg("はい", 30850, 31020, translated_text="好"),
        ASRDataSeg("あ", 31020, 31170, translated_text="啊"),
        ASRDataSeg("次の話です。", 31520, 34000, translated_text="下一个话题。"),
    ])

    result = normalize_asr_data(data)

    assert [seg.text for seg in result.segments] == ["黄色いリバイだ", "はい", "あ", "次の話です"]
    _assert_readable(result)


def test_normalize_short_cue_starts_earlier_into_free_gap() -> None:
    data = ASRData([
        ASRDataSeg("挿すよ。", 27339, 29739, translated_text="插进去哦。"),
        ASRDataSeg("リアルはこんな感じ。", 31579, 31600, translated_text="实际是这样的感觉。"),
        ASRDataSeg("やるわこんな感じ。", 31950, 32100, translated_text="要做了 这样的感觉。"),
    ])

    result = normalize_asr_data(data)

    _assert_readable(result)
    assert result.segments[1].start_time < 31579
