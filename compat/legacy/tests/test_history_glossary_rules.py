from __future__ import annotations

from types import SimpleNamespace

from qwen_asr.glossary import GlossaryEntry
from qwen_asr.history_glossary import _normalize_glossary_text as legacy_normalize_glossary_text
from qwen_asr.history_glossary import _score_to_level as legacy_score_to_level
from qwen_asr.history_glossary_rules import (
    guess_glossary_group,
    is_llm_glossary_entry_allowed,
    looks_like_glossary_candidate,
    normalize_glossary_text,
    score_to_level,
)


def test_normalize_glossary_text_removes_spacing_and_outer_punctuation() -> None:
    assert normalize_glossary_text("  \u30c8 \u30b2\u30e9\u30b8\u3002 ") == "\u30c8\u30b2\u30e9\u30b8"
    assert legacy_normalize_glossary_text("  A B! ") == "AB"


def test_glossary_candidate_requires_short_high_quality_pair() -> None:
    item = SimpleNamespace(
        matched_segment_count=1,
        length_ratio_score=0.8,
        time_overlap_score=0.9,
        token_coverage_score=0.95,
    )

    assert looks_like_glossary_candidate("\u30b3\u30a8\u30ed\u30b0", "\u58f0log", item) is True
    assert looks_like_glossary_candidate("\u30b3\u30a8\u30ed\u30b0", "voice", item) is False
    assert looks_like_glossary_candidate("\u30b3\u30a8\u30ed\u30b0", "\u58f0log", SimpleNamespace(**{**item.__dict__, "matched_segment_count": 3})) is False


def test_guess_glossary_group_prefers_show_terms_and_names() -> None:
    assert guess_glossary_group("\u30b3\u30a8\u30ed\u30b0", "\u58f0log") == "show_terms"
    assert guess_glossary_group("\u7406\u540d", "\u7406\u540d") == "names"
    assert guess_glossary_group("\u307e\u305f\u6765\u9031\u4f1a\u3044\u307e\u3057\u3087\u3046", "\u4e0b\u5468\u518d\u89c1") == "fixed_phrases"


def test_llm_glossary_entry_filter_rejects_sentence_and_contextual_role_phrase() -> None:
    assert is_llm_glossary_entry_allowed(
        GlossaryEntry(group="show_terms", source="\u30c8\u30b2\u30e9\u30b8", target="\u65e0\u523a\u6709\u523a")
    )
    assert not is_llm_glossary_entry_allowed(
        GlossaryEntry(group="fixed_phrases", source="\u304d\u3087\u3046\u306f\u305f\u306e\u3057\u3044", target="\u6211\u4eca\u5929\u771f\u7684\u5f88\u5f00\u5fc3")
    )
    assert not is_llm_glossary_entry_allowed(
        GlossaryEntry(group="show_terms", source="\u76e3\u7763\u6731\u674e", target="\u5bfc\u6f14\u6731\u674e")
    )


def test_score_to_level_keeps_legacy_alias() -> None:
    assert score_to_level(0.8, 0.75) == "high"
    assert score_to_level(0.6, 0.75) == "medium"
    assert score_to_level(0.4, 0.75) == "low"
    assert legacy_score_to_level(0.8, 0.75) == "high"
