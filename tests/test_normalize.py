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
