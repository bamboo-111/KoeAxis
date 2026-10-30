from __future__ import annotations

from optimizer.splitter import _inline_dialogue_parts
from optimizer.splitter_boundaries import (
    STRONG_SENTENCE_END,
    inline_dialogue_parts,
    split_inline_short_response_boundary,
    split_inline_strong_boundaries,
)


def test_strong_sentence_boundary_splits_inline_text() -> None:
    text = (
        "\u307e\u3060\u751f\u304d\u3066\u3044\u3089\u3063\u3057\u3083\u308b\u3002"
        "\u305d\u3046\u3088\u3002"
        "\u307e\u3060\u6b8b\u3055\u308c\u3066\u3044\u308b\u306e\u3002"
    )

    assert split_inline_strong_boundaries(text) == [
        "\u307e\u3060\u751f\u304d\u3066\u3044\u3089\u3063\u3057\u3083\u308b\u3002",
        "\u305d\u3046\u3088\u3002",
        "\u307e\u3060\u6b8b\u3055\u308c\u3066\u3044\u308b\u306e\u3002",
    ]


def test_short_response_splits_before_first_person_clause() -> None:
    text = "\u306f\u3044\u3001\u79c1\u661f\u91ce\u5148\u751f\u306e\u4ee3\u308f\u308a\u3002"

    assert split_inline_short_response_boundary(text) == [
        "\u306f\u3044\u3001",
        "\u79c1\u661f\u91ce\u5148\u751f\u306e\u4ee3\u308f\u308a\u3002",
    ]


def test_short_response_boundary_requires_known_follower() -> None:
    text = "\u306f\u3044\u3001\u304a\u5b88\u308a\u3057\u307e\u3059\u3002"

    assert split_inline_short_response_boundary(text) == [text]


def test_inline_dialogue_parts_keeps_splitter_compatibility_alias() -> None:
    text = "\u306f\u3044\u3001\u79c1\u304c\u884c\u304d\u307e\u3059\u3002\u6b21\u3067\u3059\u3002"

    assert inline_dialogue_parts(text) == _inline_dialogue_parts(text)


def test_strong_sentence_end_accepts_closing_bracket() -> None:
    assert STRONG_SENTENCE_END.search("\u7d42\u308f\u308a\u3067\u3059\u3002\u300d")
