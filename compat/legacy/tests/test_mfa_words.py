from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_experiment import _evaluate_mfa_words
from qwen_asr.mfa_experiment import _globalize_mfa_words
from qwen_asr.mfa_experiment import _read_mfa_words
from qwen_asr.mfa_words import evaluate_mfa_words
from qwen_asr.mfa_words import globalize_mfa_words
from qwen_asr.mfa_words import read_mfa_words
from qwen_asr.storage import write_json_atomic


def test_read_mfa_words_parses_words_tier_entries(tmp_path: Path) -> None:
    path = tmp_path / "clip.json"
    write_json_atomic(
        path,
        {
            "tiers": {
                "words": {
                    "entries": [
                        [0.12, 0.34, "\u306f\u3044"],
                        ["bad", 0.50, "skip"],
                        [0.60],
                    ]
                }
            }
        },
    )

    assert read_mfa_words(path) == [
        {"start_ms": 120, "end_ms": 340, "text": "\u306f\u3044"},
    ]


def test_read_mfa_words_returns_empty_for_missing_tier(tmp_path: Path) -> None:
    path = tmp_path / "clip.json"
    write_json_atomic(path, {"tiers": {"phones": {"entries": []}}})

    assert read_mfa_words(path) == []


def test_globalize_mfa_words_offsets_valid_ranges_only() -> None:
    words = [
        {"start_ms": 20, "end_ms": 80, "text": "\u306f"},
        {"start_ms": "bad", "end_ms": 90, "text": "skip"},
    ]

    assert globalize_mfa_words(words, clip_start_ms=1000) == [
        {"start_ms": 1020, "end_ms": 1080, "text": "\u306f"},
    ]


def test_evaluate_mfa_words_counts_known_timed_and_unknown_words() -> None:
    words = [
        {"start_ms": 100, "end_ms": 200, "text": "\u306f"},
        {"start_ms": 210, "end_ms": 220, "text": "<unk>"},
        {"start_ms": 300, "end_ms": 300, "text": "\u3044"},
        {"text": "untimed"},
    ]

    assert evaluate_mfa_words(words) == {
        "word_count": 4,
        "timed_count": 2,
        "unknown_count": 1,
        "known_timed_count": 1,
        "usable": True,
    }


def test_mfa_experiment_legacy_aliases_use_words_module(tmp_path: Path) -> None:
    path = tmp_path / "clip.json"
    write_json_atomic(
        path,
        {"tiers": {"words": {"entries": [[0.1, 0.2, "\u306f"]]}}},
    )
    words = [{"start_ms": 10, "end_ms": 20, "text": "\u306f"}]

    assert _read_mfa_words(path) == read_mfa_words(path)
    assert _globalize_mfa_words(words, clip_start_ms=100) == globalize_mfa_words(
        words,
        clip_start_ms=100,
    )
    assert _evaluate_mfa_words(words) == evaluate_mfa_words(words)
