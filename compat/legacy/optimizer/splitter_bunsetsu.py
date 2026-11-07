"""Japanese bunsetsu (文節) boundaries used to restrict where the rule splitter may cut.

The aligner's tokens are already words, so cutting between tokens can still split a bunsetsu
(食べ|たい, 出かけ|ない, 東南|アジア, 4日|前). A morphological analyser marks which token
boundaries start a new bunsetsu; the splitter only cuts there (or after punctuation), and still
uses its usual timing signal to choose among the allowed positions.

fugashi + unidic-lite are optional: without them every boundary is allowed (previous behaviour).
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any, Sequence

logger = logging.getLogger("optimizer.splitter")

_FUNCTION_POS = {"助詞", "助動詞", "接尾辞", "補助記号"}
_NOUN_LIKE_POS = {"名詞", "代名詞", "接頭辞"}
_SURU_STARTS = ("し", "さ", "す", "せ", "でき")
_PUNCT_END = re.compile(r"[\u3002\uff01\uff1f!?\u3001\uff0c,\u2026]\s*$")
_JAPANESE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


@lru_cache(maxsize=1)
def _tagger() -> Any | None:
    try:
        import fugashi  # type: ignore

        return fugashi.Tagger()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("文節境界不可用（fugashi/unidic-lite 未安装或加载失败），按原规则切分: %s", exc)
        return None


def bunsetsu_available() -> bool:
    return _tagger() is not None


def _starts_bunsetsu(pos1: str, pos2: str, surface: str, previous: tuple[str, str, str] | None) -> bool:
    if previous is None:
        return True
    prev_pos1, prev_pos2, prev_surface = previous
    if prev_pos1 == "補助記号":
        return True
    if pos1 in _FUNCTION_POS:
        return False
    if prev_pos1 == "接頭辞":
        return False
    if pos1 == "名詞" and prev_pos1 in _NOUN_LIKE_POS:
        return False  # compound noun, number + counter: 東南アジア, 4日前
    if pos1 == "動詞" and prev_pos1 == "名詞" and surface.startswith(_SURU_STARTS):
        return False  # サ変: 参加し, 選択させ
    if pos2 == "非自立可能" and pos1 in {"動詞", "形容詞"} and prev_pos1 == "助詞" and prev_surface in {"て", "で"}:
        return False  # ている, てください, てしまう, てほしい
    return True


def bunsetsu_start_offsets(text: str) -> set[int] | None:
    """Character offsets in ``text`` where a new bunsetsu begins, or None if unavailable."""
    tagger = _tagger()
    if tagger is None:
        return None
    starts: set[int] = set()
    cursor = 0
    previous: tuple[str, str, str] | None = None
    for word in tagger(text):
        surface = word.surface
        index = text.find(surface, cursor)
        if index < 0:
            continue
        feature = word.feature
        pos1 = getattr(feature, "pos1", "") or ""
        pos2 = getattr(feature, "pos2", "") or ""
        if _starts_bunsetsu(pos1, pos2, surface, previous):
            starts.add(index)
        previous = (pos1, pos2, surface)
        cursor = index + len(surface)
    return starts


def allowed_cut_indices(segments: Sequence[Any]) -> set[int] | None:
    """Indices ``j`` such that cutting between ``segments[j]`` and ``segments[j + 1]`` is allowed.

    Returns None (no restriction) when the analyser is unavailable or the text is not Japanese.
    """
    if len(segments) < 2:
        return None
    texts = [str(getattr(segment, "text", "")) for segment in segments]
    joined = "".join(texts)
    if not _JAPANESE.search(joined):
        return None
    starts = bunsetsu_start_offsets(joined)
    if starts is None:
        return None
    allowed: set[int] = set()
    offset = 0
    for index, text in enumerate(texts[:-1]):
        offset += len(text)
        if _PUNCT_END.search(text) or offset in starts or not texts[index + 1].strip():
            allowed.add(index)
    return allowed
