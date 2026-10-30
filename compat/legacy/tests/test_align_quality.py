from __future__ import annotations

from qwen_asr.align import (
    AlignTimingValidationConfig,
    QwenForcedAligner,
    repair_local_token_collapse,
    validate_aligned_token_timing,
)
from qwen_asr.models import AlignedToken, TranscriptSegment
from qwen_asr.optimizer_bridge_stages import mark_alignment_repaired_suspects


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


def test_validate_aligned_token_timing_uses_explicit_thresholds() -> None:
    low_coverage_tokens = [
        AlignedToken("あいうえお", 0.0, 0.1),
        AlignedToken("かきくけこ", 0.2, 0.3),
    ]
    dense_tokens = [
        AlignedToken("あら", 1.0, 1.08),
        AlignedToken("あ", 1.08, 1.08),
        AlignedToken("られ", 1.08, 1.16),
        AlignedToken("ちゃん", 1.16, 1.16),
        AlignedToken("みっ", 1.16, 1.24),
        AlignedToken("ちゃう", 1.24, 1.32),
    ]

    assert validate_aligned_token_timing(low_coverage_tokens, 0.0, 10.0) is not None
    assert (
        validate_aligned_token_timing(
            low_coverage_tokens,
            0.0,
            10.0,
            config=AlignTimingValidationConfig(min_coverage_ratio=0.01),
        )
        is None
    )
    assert validate_aligned_token_timing(dense_tokens, 1.0, 2.0) is not None
    assert (
        validate_aligned_token_timing(
            dense_tokens,
            1.0,
            2.0,
            config=AlignTimingValidationConfig(local_collapse_max_cps=200.0),
        )
        is None
    )


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


def test_run_segment_can_keep_failed_tokens_for_diagnostics() -> None:
    aligner = QwenForcedAligner("fake", keep_raw_model_output=True, keep_failed_tokens=True)

    class FakeModel:
        def align(self, **kwargs):
            return {
                "tokens": [
                    {"text": "a", "start": 0.0, "end": 0.0},
                    {"text": "b", "start": 0.0, "end": 0.0},
                    {"text": "c", "start": 0.0, "end": 0.0},
                ]
            }

    aligner._model = FakeModel()  # noqa: SLF001
    transcript = TranscriptSegment(
        segment_id="segment_000001",
        audio_path="audio.wav",
        global_start_time=1.0,
        global_end_time=9.0,
        text="abc",
        language="Japanese",
    )

    result = aligner.run_segment(transcript)

    assert result.status == "failed"
    assert [token.text for token in result.tokens] == ["a", "b", "c"]
    assert result.raw_model_output == {
        "tokens": [
            {"text": "a", "start": 0.0, "end": 0.0},
            {"text": "b", "start": 0.0, "end": 0.0},
            {"text": "c", "start": 0.0, "end": 0.0},
        ]
    }
    assert result.error is not None


def test_coverage_is_judged_against_text_length_not_segment_length() -> None:
    # A jingle fills the first 10 s of the VAD segment; the spoken line is correctly aligned at the end.
    tokens = [
        AlignedToken("トゲラジ", 10.72, 11.2),
        AlignedToken("川崎", 11.2, 11.68),
        AlignedToken("から", 11.68, 11.84),
        AlignedToken("世界", 11.84, 12.16),
        AlignedToken("へ", 12.16, 12.4),
    ]
    assert validate_aligned_token_timing(tokens, 0.0, 12.6) is None
    # A single short interjection inside a long segment is not a coverage failure either.
    assert validate_aligned_token_timing([AlignedToken("あ", 1.04, 1.12)], 0.0, 7.1) is None
    # Text squeezed into a small fraction of the time it needs still fails.
    crammed = [AlignedToken("あいうえおかきくけこ" * 6, 0.0, 1.0)]
    assert "covered" in (validate_aligned_token_timing(crammed, 0.0, 15.0) or "")


def _collapsed_segment_tokens() -> list[AlignedToken]:
    return [
        AlignedToken("バンコク", 0.16, 0.56),
        AlignedToken("で", 0.56, 0.8),
        AlignedToken("応援", 1.52, 1.76),
        AlignedToken("でき", 1.76, 1.92),
        AlignedToken("た", 1.92, 2.08),
        AlignedToken("ありがとう", 4.48, 4.72),
        AlignedToken("ござい", 5.6, 5.92),
        AlignedToken("ます", 5.92, 6.24),
        AlignedToken("ありがとう", 6.72, 6.72),
        AlignedToken("ござい", 6.72, 6.72),
        AlignedToken("ますい", 6.72, 6.72),
        AlignedToken("や", 6.72, 6.88),
        AlignedToken("自動", 7.12, 7.36),
        AlignedToken("翻訳", 7.36, 7.84),
        AlignedToken("いや", 8.24, 8.4),
        AlignedToken("そう", 8.4, 8.72),
    ]


def test_repair_local_token_collapse_retimes_only_the_collapsed_span() -> None:
    tokens = _collapsed_segment_tokens()
    assert "local density" in (validate_aligned_token_timing(tokens, 0.0, 10.0) or "")

    repaired = repair_local_token_collapse(tokens, 0.0, 10.0)

    assert repaired is not None
    new_tokens, spans = repaired
    assert validate_aligned_token_timing(new_tokens, 0.0, 10.0) is None
    assert [token.text for token in new_tokens] == [token.text for token in tokens]
    assert len(spans) == 1
    assert spans[0]["collapsed_text"] == "ありがとうございますい"
    low, high = spans[0]["token_start"], spans[0]["token_end"]
    for index, (old, new) in enumerate(zip(tokens, new_tokens, strict=True)):
        if index < low or index > high:
            assert (new.start_time, new.end_time) == (old.start_time, old.end_time)
    starts = [token.start_time for token in new_tokens]
    assert starts == sorted(starts)


def test_repair_refuses_when_most_of_the_segment_collapsed() -> None:
    tokens = [AlignedToken(text, 1.0, 1.0) for text in ("あら", "あ", "られ", "れ", "ちゃん", "みっ")]
    tokens.append(AlignedToken("ちゃう", 1.0, 1.08))
    assert repair_local_token_collapse(tokens, 0.0, 5.0) is None


def test_run_segment_accepts_repaired_alignment_with_recovery_evidence() -> None:
    aligner = QwenForcedAligner("fake")
    tokens = _collapsed_segment_tokens()

    class FakeModel:
        def align(self, **kwargs):
            return {"tokens": [{"text": t.text, "start": t.start_time, "end": t.end_time} for t in tokens]}

    aligner._model = FakeModel()  # noqa: SLF001
    transcript = TranscriptSegment(
        segment_id="segment_000133",
        audio_path="audio.wav",
        global_start_time=100.0,
        global_end_time=110.0,
        text="".join(token.text for token in tokens),
        language="Japanese",
    )

    result = aligner.run_segment(transcript, cleanup=False)

    assert result.status == "completed"
    assert result.alignment_state == "completed_exact"
    assert result.recovery is not None
    assert result.recovery["method"] == "align-local-repair"
    assert "local density" in result.recovery["original_error"]
    assert result.recovery["repaired_spans"][0]["start_time"] >= 100.0


def test_run_segment_repair_can_be_disabled() -> None:
    aligner = QwenForcedAligner(
        "fake", timing_validation_config=AlignTimingValidationConfig(repair_local_collapse=False)
    )
    tokens = _collapsed_segment_tokens()

    class FakeModel:
        def align(self, **kwargs):
            return {"tokens": [{"text": t.text, "start": t.start_time, "end": t.end_time} for t in tokens]}

    aligner._model = FakeModel()  # noqa: SLF001
    transcript = TranscriptSegment("s", "audio.wav", 0.0, 10.0, "x", "Japanese")

    assert aligner.run_segment(transcript, cleanup=False).status == "failed"


def test_mark_alignment_repaired_suspects_flags_overlapping_subtitles() -> None:
    payload = {
        "1": {"start_time": 5000, "end_time": 6500, "suspect_types": ["entity"], "suspect_reason": "名字"},
        "2": {"start_time": 6500, "end_time": 7400},
        "3": {"start_time": 8000, "end_time": 9000},
    }
    aligned = [
        {"segment_id": "a", "recovery": {"method": "vad-local"}},
        {
            "segment_id": "b",
            "recovery": {
                "method": "align-local-repair",
                "repaired_spans": [{"start_time": 6.2, "end_time": 7.1, "collapsed_text": "ありがとう"}],
            },
        },
    ]

    assert mark_alignment_repaired_suspects(payload, aligned) == 2
    assert payload["1"]["suspect_types"] == ["entity", "alignment_repaired"]
    assert payload["1"]["needs_audio_review"] is True
    assert payload["1"]["suspect_reason"].startswith("名字; ")
    assert payload["2"]["suspect_types"] == ["alignment_repaired"]
    assert "needs_audio_review" not in payload["3"]
