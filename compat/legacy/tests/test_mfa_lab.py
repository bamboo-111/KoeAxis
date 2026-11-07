from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_experiment import (
    _choose_mfa_lab_text,
    _clean_mfa_lab_text,
    _looks_like_japanese_for_mfa,
    _nearest_manifest_text,
    _needs_manifest_lab_fallback,
    _normalize_mfa_candidate_lab_text,
)
from qwen_asr.mfa_lab import (
    choose_mfa_lab_text,
    clean_mfa_lab_text,
    looks_like_japanese_for_mfa,
    nearest_manifest_text,
    needs_manifest_lab_fallback,
    normalize_mfa_candidate_lab_text,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_clean_mfa_lab_text_removes_punctuation_symbols_and_marks() -> None:
    text = "\u306f\u3044!!\n\u2606\u3099 test"

    assert clean_mfa_lab_text(text) == "\u306f\u3044 test"
    assert _clean_mfa_lab_text(text) == clean_mfa_lab_text(text)


def test_normalize_mfa_candidate_lab_text_keeps_short_response_tail() -> None:
    text = clean_mfa_lab_text("\u306f \u00b4 \u306f\u3044")

    assert normalize_mfa_candidate_lab_text(text) == "\u306f\u3044"
    assert _normalize_mfa_candidate_lab_text(text) == "\u306f\u3044"
    assert normalize_mfa_candidate_lab_text("\u4eca\u65e5 \u306f\u3044") == "\u4eca\u65e5 \u306f\u3044"


def test_needs_manifest_fallback_flags_empty_symbols_and_foreign_punctuation() -> None:
    assert needs_manifest_lab_fallback("")
    assert needs_manifest_lab_fallback("\u2606")
    assert needs_manifest_lab_fallback("\u306f\u3044!")
    assert not needs_manifest_lab_fallback("\u306f\u3044\u3002")
    assert _needs_manifest_lab_fallback("\u2606") is True


def test_looks_like_japanese_for_mfa_requires_kana() -> None:
    assert looks_like_japanese_for_mfa("\u306f\u3044")
    assert not looks_like_japanese_for_mfa("漢字")
    assert _looks_like_japanese_for_mfa("\u30cf\u30a4")


def test_nearest_manifest_text_uses_closest_japanese_manifest(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "far": {
                "start_time": 8000,
                "end_time": 8300,
                "original_subtitle": "\u3046\u3093",
            },
            "near": {
                "start_time": 1000,
                "end_time": 1300,
                "original_subtitle": "\u306f\u3044",
            },
        },
    )

    candidate = {"start_ms": 1050, "end_ms": 1250, "text": "..."}

    assert nearest_manifest_text(paths, candidate) == "\u306f\u3044"
    assert _nearest_manifest_text(paths, candidate) == "\u306f\u3044"


def test_choose_mfa_lab_text_prefers_normalized_candidate_then_manifest(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1300,
                "original_subtitle": "\u306f\u3044",
            }
        },
    )

    normalized = choose_mfa_lab_text(paths, {"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u00b4 \u306f\u3044"})
    fallback = choose_mfa_lab_text(paths, {"start_ms": 1000, "end_ms": 1300, "text": "..."})

    assert normalized == {"text": "\u306f\u3044", "source": "candidate-normalized"}
    assert fallback == {"text": "\u306f\u3044", "source": "nearest-manifest"}
    assert _choose_mfa_lab_text(paths, {"start_ms": 1000, "end_ms": 1300, "text": "..."}) == fallback
