from __future__ import annotations

from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import _repair_piece_ranges as legacy_repair_piece_ranges
from qwen_asr.optimizer_bridge import aligned_manifest_to_asr_data as legacy_aligned_manifest_to_asr_data
from qwen_asr.optimizer_bridge_adapter import (
    _normalize_content,
    _owned_segment_bounds_ms,
    _repair_piece_ranges,
    _restore_transcript_surface_to_pieces,
    _sanitized_aligned_payload,
    aligned_manifest_to_asr_data,
)
from qwen_asr.storage import write_json_atomic


class Segment:
    def __init__(self, text: str, start_time: int, end_time: int) -> None:
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class ASRDataSeg:
    def __init__(self, text: str, start_time: int, end_time: int) -> None:
        self.text = text
        self.start_time = start_time
        self.end_time = end_time


class ASRData:
    def __init__(self, segments: list[ASRDataSeg]) -> None:
        self.segments = segments


def test_owned_segment_bounds_split_overlaps_at_midpoints() -> None:
    payload = [
        {"global_start_time": 0.0, "global_end_time": 10.0},
        {"global_start_time": 8.0, "global_end_time": 20.0},
        {"global_start_time": 18.0, "global_end_time": 25.0},
    ]

    assert _owned_segment_bounds_ms(payload, 1) == (9000, 19000)


def test_restore_transcript_surface_preserves_punctuation() -> None:
    pieces = [
        {"text": "Hello", "start_ms": 0, "end_ms": 800},
        {"text": "world", "start_ms": 1000, "end_ms": 1800},
        {"text": "Yes", "start_ms": 2000, "end_ms": 2800},
    ]

    restored = _restore_transcript_surface_to_pieces("Hello, world! Yes?", pieces)

    assert [piece["text"] for piece in restored] == ["Hello, ", "world! ", "Yes?"]
    assert _normalize_content("".join(str(piece["text"]) for piece in restored)) == "helloworldyes"


def test_sanitized_payload_groups_source_by_reference_owned_ranges() -> None:
    source = [
        Segment("first", 0, 500),
        Segment("second", 600, 1100),
        Segment("tail", 1500, 1900),
    ]
    reference = [
        {"segment_id": "segment_000001", "global_start_time": 0.0, "global_end_time": 1.2},
        {"segment_id": "segment_000002", "global_start_time": 1.0, "global_end_time": 2.0},
    ]

    payload = _sanitized_aligned_payload(source, reference)

    assert [item["segment_id"] for item in payload] == ["segment_000001", "segment_000002"]
    assert [item["text"] for item in payload] == ["firstsecond", "tail"]
    assert all(item["status"] == "completed" for item in payload)


def test_repair_piece_ranges_interpolates_missing_ranges_and_keeps_legacy_alias() -> None:
    pieces = [
        {"text": "a", "start_ms": 100, "end_ms": 300},
        {"text": "b", "start_ms": 300, "end_ms": 300},
        {"text": "c", "start_ms": 300, "end_ms": 300},
        {"text": "d", "start_ms": 700, "end_ms": 900},
    ]

    result = _repair_piece_ranges(pieces, 0, 1000)

    assert result == [(100, 300), (300, 500), (500, 700), (700, 900)]
    assert legacy_repair_piece_ranges(pieces, 0, 1000) == result


def test_aligned_manifest_adapter_falls_back_to_transcript_and_keeps_legacy_alias(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 1.0,
                "global_end_time": 3.0,
                "text": "fallback text",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 1.0,
                "global_end_time": 3.0,
                "text": "bad timing",
                "status": "failed",
                "error": "alignment failed",
                "tokens": [],
            }
        ],
    )

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)
    legacy_result = legacy_aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [(segment.text, segment.start_time, segment.end_time) for segment in result.segments] == [
        ("fallback text", 1000, 3000)
    ]
    assert [(segment.text, segment.start_time, segment.end_time) for segment in legacy_result.segments] == [
        ("fallback text", 1000, 3000)
    ]
