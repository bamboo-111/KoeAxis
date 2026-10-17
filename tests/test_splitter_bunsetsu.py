from __future__ import annotations

import pytest

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import SubtitleSplitter
from optimizer.splitter_bunsetsu import allowed_cut_indices, bunsetsu_available

pytestmark = pytest.mark.skipif(not bunsetsu_available(), reason="fugashi + unidic-lite not installed")


def _tokens(words: list[str], step: int = 200, gaps: dict[int, int] | None = None) -> list[ASRDataSeg]:
    """Word tokens with contiguous timing; ``gaps`` adds a pause (ms) before token index."""
    gaps = gaps or {}
    segments = []
    cursor = 0
    for index, word in enumerate(words):
        cursor += gaps.get(index, 0)
        segments.append(ASRDataSeg(word, cursor, cursor + step))
        cursor += step
    return segments


def _splitter(**kwargs) -> SubtitleSplitter:
    return SubtitleSplitter(thread_num=1, model="", base_url="", api_key="", max_word_count_cjk=10, **kwargs)


def test_allowed_cuts_keep_auxiliaries_and_compounds_attached() -> None:
    words = ["食べ", "たい", "もの", "を", "東南", "アジア", "で", "探し", "て", "いる"]
    allowed = allowed_cut_indices(_tokens(words))
    assert allowed is not None
    assert 0 not in allowed  # 食べ|たい
    assert 4 not in allowed  # 東南|アジア
    assert 7 not in allowed  # 探し|て
    assert 8 not in allowed  # て|いる
    assert 3 in allowed  # を|東南
    assert 6 in allowed  # で|探し


def test_allowed_cuts_returns_none_for_non_japanese() -> None:
    assert allowed_cut_indices(_tokens(["hello ", "world ", "again "])) is None


def test_long_segment_split_avoids_cutting_inside_bunsetsu() -> None:
    # The biggest timing gap sits inside 行き|たい; production used to cut exactly there.
    words = ["いつか", "ライブ", "に", "行き", "たい", "と", "いう", "夢", "は", "叶わ", "なかっ", "た"]
    segments = _tokens(words, gaps={4: 120})

    pieces = _splitter()._split_long_segment(segments)

    texts = [piece.text for piece in pieces]
    assert "".join(texts) == "".join(words)
    assert len(texts) >= 2
    assert all(not text.startswith(("たい", "なかっ", "た")) for text in texts[1:])

    legacy = _splitter(bunsetsu_boundaries=False)._split_long_segment(segments)
    assert any(piece.text.startswith("たい") for piece in legacy[1:])


def test_short_pause_inside_bunsetsu_does_not_force_a_group_break() -> None:
    splitter = _splitter()
    words = ["報告", "ツイート", "を", "さし", "て", "くれ", "て", "て", "さ"]
    segments = _tokens(words, gaps={4: 900})  # さし … てくれて

    groups = splitter._group_by_time_gaps(segments, max_gap=500, check_large_gaps=True)
    assert len(groups) == 1

    long_pause = _tokens(words, gaps={4: 2000})
    assert len(splitter._group_by_time_gaps(long_pause, max_gap=500, check_large_gaps=True)) == 2
