from __future__ import annotations

from qwen_asr.align import QwenForcedAligner, validate_aligned_token_timing
from qwen_asr.models import AlignedToken, TranscriptSegment


def test_validate_aligned_token_timing_rejects_repeated_zero_timestamps() -> None:
    tokens = [
        AlignedToken("あれ", 6.5, 6.5),
        AlignedToken("ちゃ", 6.5, 6.5),
        AlignedToken("story", 6.5, 6.5),
        AlignedToken("ずっと", 70.34, 70.34),
    ]

    error = validate_aligned_token_timing(tokens, 6.5, 94.1)

    assert error is not None
    assert "covered" in error


def test_validate_aligned_token_timing_allows_sparse_zero_tokens_with_good_coverage() -> None:
    tokens = [
        AlignedToken("回っ", 36.5, 36.5),
        AlignedToken("てる", 37.06, 37.3),
        AlignedToken("そんな", 37.62, 38.1),
        AlignedToken("の", 41.62, 41.62),
        AlignedToken("は", 41.62, 43.22),
        AlignedToken("見える", 66.74, 66.82),
    ]

    error = validate_aligned_token_timing(tokens, 36.5, 67.1)

    assert error is None


def test_validate_aligned_token_timing_rejects_local_density_collapse() -> None:
    tokens = [
        AlignedToken("あら", 611.71, 611.79),
        AlignedToken("あ", 611.79, 611.79),
        AlignedToken("られ", 611.79, 611.87),
        AlignedToken("れ", 611.87, 611.87),
        AlignedToken("ちゃん", 611.87, 611.87),
        AlignedToken("みっ", 611.87, 611.95),
        AlignedToken("ちゃう", 611.95, 612.03),
        AlignedToken("なん", 612.5, 613.3),
        AlignedToken("で", 613.3, 614.0),
        AlignedToken("わかった", 614.0, 615.0),
    ]

    error = validate_aligned_token_timing(tokens, 611.55, 616.2)

    assert error is not None
    assert "local density" in error


def test_run_segment_marks_unreliable_alignment_failed() -> None:
    aligner = QwenForcedAligner("fake")

    class FakeModel:
        def align(self, **kwargs):
            return {
                "tokens": [
                    {"text": "あれ", "start": 0.0, "end": 0.0},
                    {"text": "ちゃ", "start": 0.0, "end": 0.0},
                    {"text": "story", "start": 0.0, "end": 0.0},
                    {"text": "ずっと", "start": 0.0, "end": 0.0},
                ]
            }

    aligner._model = FakeModel()  # noqa: SLF001
    transcript = TranscriptSegment(
        segment_id="segment_000001",
        audio_path="audio.wav",
        global_start_time=6.5,
        global_end_time=94.1,
        text="あれちゃ story",
        language="Japanese",
    )

    result = aligner.run_segment(transcript)

    assert result.status == "failed"
    assert result.tokens == []
    assert result.error is not None
    assert "unreliable" in result.error
