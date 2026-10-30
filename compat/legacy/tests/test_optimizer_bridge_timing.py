from __future__ import annotations

from pathlib import Path
import pytest

from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import (
    _postprocess_split_segments,
    _protected_short_response_segments,
    _repair_piece_ranges,
    _sanitized_aligned_payload,
    _segments_normalized_text,
    _extract_protected_short_responses,
    _validate_aligned_manifest_for_split,
    _validate_split_content_preserved,
    _validate_split_short_responses_preserved,
    aligned_manifest_to_asr_data,
    run_split_stage,
)
from qwen_asr import optimizer_bridge as bridge_compat
from qwen_asr import optimizer_bridge_stages
from qwen_asr.storage import write_json_atomic
from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import SubtitleSplitter, _merge_asr_segments


def test_validate_aligned_manifest_for_split_rejects_bad_timing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 6.5,
                "global_end_time": 94.1,
                "text": "あれちゃ story",
                "language": "Japanese",
                "status": "completed",
                "tokens": [
                    {"text": "あれ", "start_time": 6.5, "end_time": 6.5},
                    {"text": "ちゃ", "start_time": 6.5, "end_time": 6.5},
                    {"text": "story", "start_time": 6.5, "end_time": 6.5},
                    {"text": "ずっと", "start_time": 6.5, "end_time": 6.5},
                ],
            }
        ],
    )

    with pytest.raises(RuntimeError, match="Alignment timing is unreliable"):
        _validate_aligned_manifest_for_split(paths)


def test_rule_split_bridge_postprocess_merges_tail_fragments() -> None:
    splitter = SubtitleSplitter(thread_num=1, model="", base_url="", api_key="")
    try:
        segments = [
            ASRDataSeg("\u3053\u308c", 261510, 261590),
            ASRDataSeg("\u3059\u3054\u3044\u9762\u767d\u304b\u3063", 262790, 263510),
            ASRDataSeg("\u305f\u3002", 270950, 271270),
        ]

        result = _postprocess_split_segments(splitter, segments)

        assert [segment.text for segment in result] == [
            "\u3053\u308c",
            "\u3059\u3054\u3044\u9762\u767d\u304b\u3063\u305f\u3002",
        ]
    finally:
        splitter.stop()


def test_rule_split_bridge_postprocess_extends_protected_short_display() -> None:
    splitter = SubtitleSplitter(thread_num=1, model="", base_url="", api_key="")
    try:
        segments = [
            ASRDataSeg("\u524d\u3067\u3059\u3002", 300, 1000),
            ASRDataSeg("\u3046\u3093\u3002", 1100, 1101),
            ASRDataSeg("\u6b21\u3067\u3059\u3002", 1300, 1800),
        ]

        result = _postprocess_split_segments(splitter, segments)

        assert [(segment.text, segment.start_time, segment.end_time) for segment in result] == [
            ("\u524d\u3067\u3059\u3002", 300, 1000),
            ("\u3046\u3093\u3002", 1100, 1220),
            ("\u6b21\u3067\u3059\u3002", 1300, 1800),
        ]
    finally:
        splitter.stop()


def test_rule_split_bridge_postprocess_extends_ordinary_short_display() -> None:
    splitter = SubtitleSplitter(thread_num=1, model="", base_url="", api_key="")
    try:
        segments = [
            ASRDataSeg("\u524d\u3067\u3059\u3002", 300, 1000),
            ASRDataSeg("\u4e09\u65e5", 1100, 1420),
            ASRDataSeg("\u6b21\u3067\u3059\u3002", 1800, 2400),
        ]

        result = _postprocess_split_segments(splitter, segments)

        assert [(segment.text, segment.start_time, segment.end_time) for segment in result] == [
            ("\u524d\u3067\u3059\u3002", 300, 1000),
            ("\u4e09\u65e5", 1100, 1600),
            ("\u6b21\u3067\u3059\u3002", 1800, 2400),
        ]
    finally:
        splitter.stop()


def test_merge_asr_segments_preserves_repeated_boundary_text() -> None:
    merged = _merge_asr_segments(
        ASRDataSeg("\u3075\u3056\u3051\u308b\u306a\u3002", 1000, 1400),
        ASRDataSeg("\u306a", 1380, 1500),
    )

    assert merged.text == "\u3075\u3056\u3051\u308b\u306a\u3002\u306a"


def test_aligned_manifest_to_asr_data_falls_back_to_transcript_for_failed_align(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 6.5,
                "global_end_time": 21.5,
                "text": "あられちゃう。",
                "language": "Japanese",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 6.5,
                "global_end_time": 21.5,
                "text": "あられちゃう。",
                "language": "Japanese",
                "status": "failed",
                "error": "alignment token timing unreliable: covered 0.640s of 15.600s",
                "tokens": [],
            }
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert len(result.segments) == 1
    assert result.segments[0].text == "あられちゃう。"
    assert result.segments[0].start_time == 6500
    assert result.segments[0].end_time == 21500


def test_aligned_manifest_to_asr_data_falls_back_for_completed_unreliable_timing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 10.0,
                "global_end_time": 20.0,
                "text": "fallback text",
                "language": "English",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 10.0,
                "global_end_time": 20.0,
                "text": "bad timing text",
                "language": "English",
                "status": "completed",
                "tokens": [
                    {"text": "abcdefghi", "start_time": 10.0, "end_time": 10.24},
                    {"text": "tail", "start_time": 19.0, "end_time": 20.0},
                ],
            }
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [(segment.text, segment.start_time, segment.end_time) for segment in result.segments] == [
        ("fallback text", 10000, 20000)
    ]


def test_aligned_manifest_to_asr_data_preserves_completed_tail_at_failed_neighbor(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000002",
                "audio_path": "audio.wav",
                "global_start_time": 10.0,
                "global_end_time": 20.0,
                "text": "fallback text",
                "language": "English",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 0.0,
                "global_end_time": 12.0,
                "text": "good tail",
                "language": "English",
                "status": "completed",
                "tokens": [
                    {"text": "good", "start_time": 8.0, "end_time": 9.0},
                    {"text": "tail", "start_time": 10.5, "end_time": 11.0},
                ],
            },
            {
                "segment_id": "segment_000002",
                "audio_path": "audio.wav",
                "global_start_time": 10.0,
                "global_end_time": 20.0,
                "text": "fallback text",
                "language": "English",
                "status": "failed",
                "error": "alignment token timing unreliable: local density 9 chars in 0.240s (37.5 cps)",
                "tokens": [],
            },
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [(seg.text, seg.start_time, seg.end_time) for seg in result.segments] == [
        ("good", 8000, 9000),
        ("tail", 10500, 11000),
        ("fallback text", 11000, 20000),
    ]


def test_aligned_manifest_to_asr_data_preserves_zero_duration_short_response(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.transcript_manifest, [])
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 2.0,
                "text": "before yes after",
                "status": "completed",
                "tokens": [
                    {"text": "before", "start_time": 0.1, "end_time": 0.8},
                    {"text": "yes", "start_time": 0.9, "end_time": 0.9},
                    {"text": "after", "start_time": 1.0, "end_time": 1.8},
                ],
            }
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [segment.text for segment in result.segments] == ["before", "yes", "after"]
    assert all(segment.end_time > segment.start_time for segment in result.segments)


def test_aligned_manifest_to_asr_data_narrows_overlong_short_fallback(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 0.0,
                "global_end_time": 15.0,
                "text": "\u306f\u3044\u3002",
                "language": "Japanese",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 0.0,
                "global_end_time": 15.0,
                "text": "\u3042\u3002\u306f\u3044\u3002",
                "language": "Japanese",
                "status": "completed",
                "tokens": [
                    {"text": "\u3042\u3002", "start_time": 0.0, "end_time": 7.0},
                    {"text": "\u306f\u3044\u3002", "start_time": 7.0, "end_time": 15.0},
                ],
            }
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [(segment.text, segment.end_time - segment.start_time) for segment in result.segments] == [
        ("\u306f\u3044\u3002", 500),
    ]
    assert result.segments[0].start_time == 7250
    assert result.segments[0].end_time == 7750


def test_repair_piece_ranges_uses_local_gap_for_zero_tokens() -> None:
    ranges = _repair_piece_ranges(
        [
            {"text": "a", "start_ms": 100, "end_ms": 300},
            {"text": "b", "start_ms": 300, "end_ms": 300},
            {"text": "c", "start_ms": 300, "end_ms": 300},
            {"text": "d", "start_ms": 700, "end_ms": 900},
        ],
        0,
        1000,
    )

    assert ranges == [(100, 300), (300, 500), (500, 700), (700, 900)]
    assert all(end - start > 1 for start, end in ranges)


def test_aligned_manifest_to_asr_data_restores_transcript_punctuation(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 3.0,
                "text": "Hello, world! Yes?",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 3.0,
                "text": "Hello, world! Yes?",
                "status": "completed",
                "tokens": [
                    {"text": "Hello", "start_time": 0.0, "end_time": 0.8},
                    {"text": "world", "start_time": 1.0, "end_time": 1.8},
                    {"text": "Yes", "start_time": 2.0, "end_time": 2.8},
                ],
            }
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [segment.text for segment in result.segments] == ["Hello, ", "world! ", "Yes?"]
    assert "".join(segment.text for segment in result.segments) == "Hello, world! Yes?"


def test_aligned_manifest_to_asr_data_deduplicates_only_exact_overlap_prefix(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.transcript_manifest, [])
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 10.3,
                "text": "alpha shared",
                "status": "completed",
                "tokens": [
                    {"text": "alpha", "start_time": 8.0, "end_time": 9.0},
                    {"text": "shared", "start_time": 9.4, "end_time": 10.2},
                ],
            },
            {
                "segment_id": "segment_000002",
                "global_start_time": 9.7,
                "global_end_time": 20.0,
                "text": "shared unique",
                "status": "completed",
                "tokens": [
                    {"text": "shared", "start_time": 9.8, "end_time": 10.4},
                    {"text": "unique", "start_time": 10.5, "end_time": 12.5},
                ],
            },
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [segment.text for segment in result.segments] == ["alpha", "shared", "unique"]
    assert all(
        left.end_time <= right.start_time for left, right in zip(result.segments, result.segments[1:], strict=False)
    )


def test_aligned_manifest_to_asr_data_rejects_stale_completed_content(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.aligned_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 5.0,
                "text": "keep this response",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 0.0,
                "global_end_time": 5.0,
                "text": "unrelated",
                "status": "completed",
                "tokens": [
                    {"text": "unrelated", "start_time": 0.0, "end_time": 5.0},
                ],
            }
        ],
    )

    class ASRDataSeg:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class ASRData:
        def __init__(self, segments: list[ASRDataSeg]) -> None:
            self.segments = segments

    result = aligned_manifest_to_asr_data(paths, ASRData, ASRDataSeg)

    assert [(segment.text, segment.start_time, segment.end_time) for segment in result.segments] == [
        ("keep this response", 0, 5000),
    ]


def test_validate_split_content_preserved_rejects_deleted_text() -> None:
    class Segment:
        def __init__(self, text: str) -> None:
            self.text = text

    with pytest.raises(RuntimeError, match="Split stage changed aligned text content"):
        _validate_split_content_preserved(
            [Segment("before"), Segment("yes"), Segment("after")],
            [Segment("before"), Segment("after")],
        )


def test_validate_split_short_responses_preserved_accepts_nearby_match() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    _validate_split_short_responses_preserved(
        [Segment("\u306f\u3044\u3002", 1000, 1400)],
        [Segment("\u306f\u3044", 1080, 1480)],
    )


def test_validate_split_short_responses_preserved_rejects_moved_short_response() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    with pytest.raises(RuntimeError, match="protected short responses"):
        _validate_split_short_responses_preserved(
            [Segment("\u306f\u3044\u3002", 1000, 1400)],
            [Segment("\u3058\u3083\u306d\u3002", 1200, 1600), Segment("\u306f\u3044", 5000, 5400)],
        )


def test_validate_split_short_responses_ignores_one_ms_source_artifacts() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    _validate_split_short_responses_preserved(
        [Segment("\u3048\uff1f", 566613, 566614)],
        [],
    )


def test_validate_split_short_responses_accepts_extracted_adjacent_result() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    _validate_split_short_responses_preserved(
        [
            Segment("\u3042\u306a\u305f\u304c\u3002", 1171739, 1172219),
            Segment("\u3046\u3093\uff1f", 1172219, 1172379),
        ],
        [
            Segment("\u3042\u306a\u305f\u304c\u3002", 1171739, 1172219),
            Segment("\u3046\u3093\uff1f", 1172219, 1172379),
        ],
    )


def test_protected_short_responses_ignore_contextual_dame_token() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    protected = _protected_short_response_segments(
        [
            Segment("\u306f", 1000, 1080),
            Segment("\u30c0\u30e1", 1080, 1240),
            Segment("\u3088\u3002", 1240, 1480),
            Segment("\u30c0\u30e1\uff01", 3000, 3400),
        ]
    )

    assert [item["text"] for item in protected] == ["\u30c0\u30e1\uff01"]


def test_protected_short_responses_ignore_attached_hai_and_iya() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    protected = _protected_short_response_segments(
        [
            Segment("\u306f\u3044", 1000, 1040),
            Segment("\u5148\u751f\u3001", 1040, 1320),
            Segment("\u3044\u3084", 3000, 3240),
            Segment("\u3067\u3001", 3240, 3400),
            Segment("\u306f\u3044\u3002", 5000, 5400),
            Segment("\u3044\u3084\uff01", 7000, 7400),
        ]
    )

    assert [item["text"] for item in protected] == ["\u306f\u3044\u3002", "\u3044\u3084\uff01"]


def test_protected_short_responses_ignore_punctuated_left_attached_context() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    protected = _protected_short_response_segments(
        [
            Segment("\u306a\u3093\u3066", 1000, 1400),
            Segment("\u3044\u3084\u3002", 1400, 1560),
            Segment("\u5bb6", 2200, 2280),
        ]
    )

    assert protected == []


def test_protected_short_responses_ignore_left_attached_repeated_hai() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    protected = _protected_short_response_segments(
        [
            Segment("\u306f\u3044", 1000, 1320),
            Segment("\u306f\u3044\u3002", 1480, 1720),
            Segment("\u4eca\u5ea6", 3200, 3520),
        ]
    )

    assert protected == []


def test_extract_protected_short_responses_splits_merged_segment() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int, translated_text: str = "") -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time
            self.translated_text = translated_text

    source = [
        Segment("\u305d\u3093\u306a\u306b", 1000, 1200),
        Segment("\u8fd4\u3059", 1200, 1400),
        Segment("\u306e", 800, 900),
        Segment("\u3044\u3084\u3002", 1500, 1900),
    ]
    result = [Segment("\u305d\u3093\u306a\u306b\u8fd4\u3059\u306e\u3044\u3084\u3002", 1000, 1900)]

    extracted = _extract_protected_short_responses(source, result)

    assert [(item.text, item.start_time, item.end_time) for item in extracted] == [
        ("\u305d\u3093\u306a\u306b\u8fd4\u3059\u306e", 1000, 1500),
        ("\u3044\u3084\u3002", 1500, 1900),
    ]


def test_extract_protected_short_responses_excludes_leading_punctuation_from_span() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int, translated_text: str = "") -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time
            self.translated_text = translated_text

    source = [Segment("\u3046\u3093\uff1f", 1172219, 1172379)]
    result = [Segment("\u3042\u306a\u305f\u304c\u3002\u3046\u3093\uff1f", 1171739, 1172379)]

    extracted = _extract_protected_short_responses(source, result)

    assert [(item.text, item.start_time, item.end_time) for item in extracted] == [
        ("\u3042\u306a\u305f\u304c\u3002", 1171739, 1172219),
        ("\u3046\u3093\uff1f", 1172219, 1172379),
    ]


def test_extract_protected_short_responses_can_run_after_display_merge() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int, translated_text: str = "") -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time
            self.translated_text = translated_text

    source = [Segment("\u3046\u3093\uff1f", 1172219, 1172379)]
    merged = [Segment("\u3042\u306a\u305f\u304c\u3002\u3046\u3093\uff1f", 1171739, 1172379)]

    extracted = _extract_protected_short_responses(source, merged)

    _validate_split_short_responses_preserved(source, extracted)


def test_extract_protected_short_responses_keeps_segment_when_text_missing() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int, translated_text: str = "") -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time
            self.translated_text = translated_text

    source = [Segment("\u306f\u3044\u3002", 1000, 1400)]
    result = [Segment("\u3058\u3083\u306d\u3002", 1000, 1400)]

    assert _extract_protected_short_responses(source, result) == result


def test_extract_protected_short_responses_keeps_normalized_content() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int, translated_text: str = "") -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time
            self.translated_text = translated_text

    source = [Segment("\u306f\u3044\u3002", 1000, 1400)]
    result = [Segment("\u306f\u3044\u306f\u3044\u3002", 1000, 1600)]

    extracted = _extract_protected_short_responses(source, result)

    assert _segments_normalized_text(extracted) == _segments_normalized_text(result)


def test_sanitized_aligned_payload_uses_content_preserving_source() -> None:
    class Segment:
        def __init__(self, text: str, start_time: int, end_time: int) -> None:
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    source = [
        Segment("first. ", 0, 900),
        Segment("failed transcript", 900, 1500),
        Segment("unique?", 1500, 2400),
    ]
    reference = [
        {
            "segment_id": "segment_000001",
            "global_start_time": 0.0,
            "global_end_time": 1.2,
            "status": "completed",
            "tokens": [{"text": "stale", "start_time": 0.0, "end_time": 1.2}],
        },
        {
            "segment_id": "segment_000002",
            "global_start_time": 1.0,
            "global_end_time": 2.4,
            "status": "failed",
            "tokens": [],
        },
    ]

    payload = _sanitized_aligned_payload(source, reference)

    assert [item["segment_id"] for item in payload] == ["segment_000001", "segment_000002"]
    assert [token["text"] for item in payload for token in item["tokens"]] == [
        "first. ",
        "failed transcript",
        "unique?",
    ]
    assert "".join(item["text"] for item in payload) == "first. failed transcriptunique?"
    assert all(item["status"] == "completed" for item in payload)


def test_run_split_stage_rejects_retired_split_mode(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    with pytest.raises(ValueError, match="Only 'rule' is available"):
        run_split_stage(paths, split_mode="token-counts")


def test_optimizer_bridge_stage_exports_delegate_to_stage_module() -> None:
    assert bridge_compat.run_split_stage is optimizer_bridge_stages.run_split_stage
    assert bridge_compat.run_translate_stage is optimizer_bridge_stages.run_translate_stage
    assert bridge_compat._postprocess_split_segments is optimizer_bridge_stages.postprocess_split_segments
    assert bridge_compat._load_optimizer_types is optimizer_bridge_stages.load_optimizer_types
