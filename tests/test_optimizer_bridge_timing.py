from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import _validate_aligned_manifest_for_split, aligned_manifest_to_asr_data
from qwen_asr.storage import write_json_atomic


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


def test_aligned_manifest_to_asr_data_clips_completed_tokens_at_failed_neighbor(tmp_path: Path) -> None:
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
        ("fallback text", 10000, 20000),
    ]
