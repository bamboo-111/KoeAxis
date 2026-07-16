from __future__ import annotations

from pathlib import Path

from qwen_asr.content_quality import Cue, _compare_stages, evaluate_content_conservation, normalize_japanese
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_normalize_japanese_ignores_formatting_but_keeps_dialogue() -> None:
    assert normalize_japanese(" はい！ Ａ。") == "はいa"


def test_gate_detects_missing_short_response() -> None:
    source = [Cue("前です。", 0, 900), Cue("はい。", 900, 1200), Cue("後です。", 1200, 2100)]
    target = [Cue("前です。", 0, 900), Cue("後です。", 1200, 2100)]
    result = _compare_stages("split", source, "proofread", target)
    assert any(item["type"] == "missing_short_response" and item["severity"] == "FAIL" for item in result["issues"])


def test_gate_detects_short_response_inside_transcript_chunk() -> None:
    source = [Cue("前です。はい。後です。", 0, 2100)]
    target = [Cue("前です。", 0, 900), Cue("後です。", 1200, 2100)]
    result = _compare_stages("transcript", source, "split", target)
    assert any(item["type"] == "missing_short_response" and item["text"] == "はい" for item in result["issues"])


def test_gate_uses_local_time_for_short_response_inside_chunk() -> None:
    source = [Cue("前です。はい。後です。", 0, 2100)]
    target = [Cue("前です。", 0, 650), Cue("はい。", 700, 1000), Cue("後です。", 1450, 2100)]
    result = _compare_stages("align", source, "split", target)
    assert not any(item["type"] == "missing_short_response" for item in result["issues"])


def test_gate_deduplicates_identical_missing_short_response() -> None:
    source = [Cue("はい。はい。", 0, 1000)]
    target: list[Cue] = []
    result = _compare_stages("align", source, "split", target)
    missing = [item for item in result["issues"] if item["type"] == "missing_short_response"]
    assert len(missing) == 2


def test_gate_warns_when_short_response_exists_at_shifted_time() -> None:
    source = [Cue("はい。", 1000, 1300)]
    target = [Cue("はい。", 3000, 3300)]
    result = _compare_stages("transcript", source, "align", target)
    assert not any(item["type"] == "missing_short_response" for item in result["issues"])
    assert any(item["type"] == "short_response_timing_shifted" and item["severity"] == "WARN" for item in result["issues"])


def test_gate_detects_missing_unique_text() -> None:
    source = [Cue("ここだけの台詞", 0, 1000), Cue("続き", 1000, 2000)]
    target = [Cue("続き", 1000, 2000)]
    result = _compare_stages("align", source, "split", target)
    assert any(item["type"] == "missing_unique_text" for item in result["issues"])


def test_gate_warns_when_audio_evidence_replaces_text_in_proofread_window() -> None:
    source = [Cue("\u3067\u3001\u3053\u3053\u304b\u3089\u306e", 0, 1000)]
    target = [
        Cue(
            "\u3067",
            0,
            1000,
            metadata={
                "proofread_history": [
                    {
                        "source": "mimo-nearby-audio",
                        "changes": {"original_subtitle": {"before": "\u3067\u3001\u3053\u3053\u304b\u3089\u306e", "after": "\u3067"}},
                        "evidence": {"confidence": 0.95, "reason": "audio"},
                    }
                ]
            },
        )
    ]
    result = _compare_stages("split", source, "proofread", target)
    assert not any(item["type"] == "missing_unique_text" for item in result["issues"])
    assert any(
        item["type"] == "proofread_audio_evidence_changed_text" and item["severity"] == "WARN"
        for item in result["issues"]
    )


def test_gate_fails_when_proofread_replaces_text_without_audio_evidence() -> None:
    source = [Cue("\u3067\u3001\u3053\u3053\u304b\u3089\u306e", 0, 1000)]
    target = [Cue("\u3067", 0, 1000)]
    result = _compare_stages("split", source, "proofread", target)
    assert any(item["type"] == "missing_unique_text" and item["severity"] == "FAIL" for item in result["issues"])


def test_gate_detects_introduced_duplicate() -> None:
    source = [Cue("重複しない", 0, 1000)]
    target = [Cue("重複しない", 0, 700), Cue("重複しない", 700, 1400)]
    result = _compare_stages("align", source, "split", target)
    assert any(item["type"] == "introduced_duplicate" for item in result["issues"])


def test_gate_detects_repeated_boundary_suffix() -> None:
    source = [Cue("これは境界です", 0, 1800)]
    target = [Cue("これは境界", 0, 1000), Cue("境界です", 900, 1800)]
    result = _compare_stages("align", source, "split", target)
    assert any(item["type"] == "introduced_duplicate" and item["text"] == "境界" for item in result["issues"])


def test_gate_does_not_treat_split_unit_as_new_duplicate() -> None:
    source = [Cue("はい。続けます。はい。", 0, 2000)]
    target = [Cue("はい。", 0, 400), Cue("続けます。", 400, 1400), Cue("はい。", 1400, 2000)]
    result = _compare_stages("align", source, "split", target)
    assert not any(item["type"] == "introduced_duplicate" for item in result["issues"])


def test_gate_does_not_report_duplicate_inherited_from_source() -> None:
    source = [Cue("境界", 0, 1000), Cue("境界です", 900, 1800)]
    target = [Cue("境界", 0, 1000), Cue("境界です", 900, 1800)]
    result = _compare_stages("proofread", source, "export", target)
    assert not any(item["type"] == "introduced_duplicate" for item in result["issues"])


def test_gate_detects_alignment_result_that_is_too_short() -> None:
    source = [Cue("これは十分に長い元の認識です", 0, 2000, "s1")]
    target = [Cue("これは", 0, 500, "s1")]
    result = _compare_stages("transcript", source, "align", target)
    assert any(item["type"] == "alignment_fallback_too_short" for item in result["issues"])


def test_gate_accepts_content_preserving_split() -> None:
    source = [Cue("前です。はい。後です。", 0, 2000)]
    target = [Cue("前です。", 0, 700), Cue("はい。", 700, 1000), Cue("後です。", 1000, 2000)]
    result = _compare_stages("align", source, "split", target)
    assert result["content_retention"] == 1.0
    assert not result["issues"]


def test_gate_compares_split_against_actual_aligned_split_source(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "s1",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "\u3046\u3093\u3002",
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "audio_path": "s1.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "\u3046\u3093\u3002\u306f\u3044\u3002",
                "status": "completed",
                "tokens": [
                    {"text": "\u3046\u3093", "start_time": 0.0, "end_time": 0.3},
                    {"text": "\u306f\u3044", "start_time": 0.7, "end_time": 1.0},
                ],
            }
        ],
    )
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 0,
                "end_time": 500,
                "original_subtitle": "\u3046\u3093\u3002",
            }
        },
    )

    report = evaluate_content_conservation(paths)

    assert not any(
        item["type"] == "missing_short_response" and item.get("text") == "\u306f\u3044"
        for item in report["issues"]
    )
