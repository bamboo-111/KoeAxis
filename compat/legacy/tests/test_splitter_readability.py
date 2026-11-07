from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import _is_protected_short_utterance
from optimizer.splitter_readability import (
    ReadabilityRuleConfig,
    can_merge_filler,
    can_merge_readability,
    is_dialogue_standalone_response,
    is_protected_short_display_response,
    is_protected_short_utterance,
    is_readability_short,
    is_structural_readability_fragment,
    starts_with_short_filler,
)


def test_dialogue_standalone_response_is_not_filler_merge_candidate() -> None:
    left = ASRDataSeg("\u805e\u3053\u3048\u3066\u307e\u3059\u304b", 0, 900)
    right = ASRDataSeg("\u3046\u3093", 950, 1100)

    assert is_dialogue_standalone_response(right.text)
    assert not can_merge_filler(left, right)
    assert starts_with_short_filler(right.text)


def test_structural_fragment_can_merge_across_tail_gap() -> None:
    left = ASRDataSeg("\u305d\u308c", 1000, 1800)
    right = ASRDataSeg("\u306b", 4200, 4280)

    assert is_structural_readability_fragment(right.text)
    assert is_readability_short(right)
    assert can_merge_readability(left, right)


def test_complete_short_utterance_is_protected_from_readability_merge() -> None:
    left = ASRDataSeg("\u3061\u3087\u3063\u3068\u5f85\u3063\u3066", 0, 700)
    right = ASRDataSeg("\u5927\u4e08\u592b", 760, 1100)

    assert is_protected_short_utterance(right.text)
    assert _is_protected_short_utterance(right.text)
    assert not can_merge_readability(left, right)


def test_display_short_responses_have_dedicated_protection_set() -> None:
    assert is_protected_short_display_response("\u306f\u3042\u3002")
    assert not is_protected_short_display_response("\u4e09\u65e5")


def test_config_can_tighten_filler_merge_gap() -> None:
    left = ASRDataSeg("\u4eca\u65e5\u306f", 0, 500)
    right = ASRDataSeg("\u3042\u306e", 520, 650)
    strict = ReadabilityRuleConfig(filler_merge_max_gap=10)

    assert can_merge_filler(left, right)
    assert not can_merge_filler(left, right, strict)
