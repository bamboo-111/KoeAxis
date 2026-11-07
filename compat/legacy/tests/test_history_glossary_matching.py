from __future__ import annotations

from types import SimpleNamespace

from qwen_asr.history_glossary import _boundary_score as legacy_boundary_score
from qwen_asr.history_glossary import _length_ratio_score as legacy_length_ratio_score
from qwen_asr.history_glossary import _overlap_ms as legacy_overlap_ms
from qwen_asr.history_glossary_matching import (
    boundary_score,
    interval_overlap_score,
    length_ratio_score,
    overlap_ms,
    score_candidate_payload,
)
from qwen_asr.history_glossary_rules import normalize_glossary_text


def test_overlap_and_boundary_scores_keep_legacy_aliases() -> None:
    assert overlap_ms(1000, 2000, 1500, 2500) == 500
    assert legacy_overlap_ms(1000, 2000, 1500, 2500) == 500
    assert interval_overlap_score(1000, 2000, 1500, 2500) == 1 / 3
    assert boundary_score(1000, 2000, 1000, 2000) == 1.0
    assert legacy_boundary_score(1000, 2000, 1000, 2000) == 1.0


def test_length_ratio_score_uses_glossary_normalization() -> None:
    assert length_ratio_score(" 無 刺！", "無刺", normalize_text=normalize_glossary_text) == 1.0
    assert legacy_length_ratio_score(" 無 刺！", "無刺") == 1.0
    assert length_ratio_score("", "無刺", normalize_text=normalize_glossary_text) == 0.0


def test_score_candidate_payload_reports_penalties_and_reasons() -> None:
    dialogue = SimpleNamespace(start_ms=1000, end_ms=2000, text="無刺有刺")

    payload = score_candidate_payload(
        dialogue=dialogue,
        source_text="トゲトゲ",
        source_start_ms=2600,
        source_end_ms=3400,
        matched_segment_count=3,
        covered_duration=200,
        normalize_text=normalize_glossary_text,
    )

    assert payload["score"] < 0.5
    assert payload["merge_penalty"] == 0.16
    assert "time weak" in payload["reasons"]
    assert "merged 3 splits" in payload["reasons"]
    assert "sparse tokens" in payload["reasons"]
