"""Consecutive cues that only repeat one short phrase ("やります。" "やります。" "やります。").

The rule split cuts at every sentence end, so a phrase said several times in a row ends up as one cue per copy, each
with a sliver of time (often an even split of a collapsed alignment). When nobody confidently changes between the
copies they are joined into one cue ("やります、やります、やります"), which also keeps normalize from splitting them
again (it only splits cues with several sentences).

A last copy that the aligner squeezed into the start of the next sentence (its tokens have (almost) no duration) is
moved over to the chain; the next cue keeps its start time, so the time stays with the sentence actually spoken there.

Only cue boundaries and the punctuation between copies change; the spoken content and its order are preserved.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Sequence

from qwen_asr.subtitle_layout import content_key

MAX_PHRASE_CHARS = 12
# 3+ characters: two-character backchannels ("そう", "うん") are often said by different people in turn
MIN_PHRASE_CHARS = 3
MAX_GAP_MS = 1000
MAX_CHAIN_MS = 8000
DEGENERATE_MS = 120  # a copy shorter than this, or with a token under SQUEEZED_MS_PER_CHAR per character, was squeezed
SQUEEZED_MS_PER_CHAR = 50
_TRAILING = re.compile(r"[\s。．.、，,！!？?…]*$")


def _phrase(text: str) -> str:
    key = content_key(text)
    return key if MIN_PHRASE_CHARS <= len(key) <= MAX_PHRASE_CHARS else ""


def _prefix_end(text: str, key: str) -> int:
    """Index in text just after the characters spelling ``key`` and the punctuation after them; -1 if text does not
    start with key."""
    seen = ""
    for index, char in enumerate(text):
        seen += content_key(char)
        if seen == key:
            end = index + 1
            while end < len(text) and not content_key(text[end]):
                end += 1
            return end
        if not key.startswith(seen):
            return -1
    return -1


def _confident_change(labels: Sequence[Any] | None, a: int, b: int, min_margin: float) -> bool:
    if labels is None:
        return False
    left, right = labels[a], labels[b]
    return bool(left.speaker and right.speaker and left.speaker != right.speaker
                and left.margin >= min_margin and right.margin >= min_margin)


def _join(texts: list[str]) -> str:
    stems = [_TRAILING.sub("", text.strip()) for text in texts]
    tail = texts[-1].strip()[len(stems[-1]):]
    return "、".join(stems) + tail


def move_degenerate_copies(
    cues: list[Any],
    token_spans: Sequence[list[tuple[int, int]]],
    new_cue: Callable[[str, int, int], Any],
) -> tuple[list[Any], int]:
    """Move a repeated phrase at the start of a cue into the previous cue when its tokens have no real duration.

    token_spans[j]: (start_ms, end_ms) of the tokens in cue j, in text order, with their char counts given by the
    content key, i.e. a list aligned with ``content_key(cue.text)`` characters: one (start, end) per character.
    """
    out = list(cues)
    moved = 0
    for j in range(1, len(out)):
        phrase = _repeat_unit(str(out[j - 1].text))
        text = str(out[j].text)
        key = content_key(text)
        if not phrase or len(key) <= len(phrase) or not key.startswith(phrase):
            continue
        spans = token_spans[j]
        if len(spans) < len(phrase):
            continue
        copy = spans[: len(phrase)]
        duration = max(end for _, end in copy) - min(start for start, _ in copy)
        # a token of the copy spoken faster than SQUEEZED_MS_PER_CHAR per character cannot be real speech: the
        # aligner ran out of audio and crushed it (zero-length tokens are widened to a few frames upstream)
        per_char = [(end - start) / copy.count((start, end)) for start, end in set(copy)]
        if duration >= DEGENERATE_MS and min(per_char) >= SQUEEZED_MS_PER_CHAR:
            continue
        cut = _prefix_end(text, phrase)
        if cut <= 0 or not text[cut:].strip():
            continue
        prev = out[j - 1]
        out[j - 1] = new_cue(_join([str(prev.text), text[:cut]]), int(prev.start_time), int(prev.end_time))
        out[j] = new_cue(text[cut:].strip(), int(out[j].start_time), int(out[j].end_time))
        moved += 1
    return out, moved


def merge_repeat_chains(
    cues: list[Any],
    new_cue: Callable[[str, int, int], Any],
    *,
    labels: Sequence[Any] | None = None,
    min_margin: float = 0.1,
) -> tuple[list[Any], int]:
    """Join runs of adjacent cues whose text is the same short phrase (several copies inside one cue count too),
    unless the speaker confidently changes between them. Returns the cues and the number of joins."""
    out: list[Any] = []
    joins = 0
    j = 0
    while j < len(cues):
        unit = _repeat_unit(str(cues[j].text))
        k = j + 1
        if unit:
            while (
                k < len(cues)
                and _repeat_unit(str(cues[k].text)) == unit
                and int(cues[k].start_time) - int(cues[k - 1].end_time) <= MAX_GAP_MS
                and int(cues[k].end_time) - int(cues[j].start_time) <= MAX_CHAIN_MS
                and not _confident_change(labels, k - 1, k, min_margin)
            ):
                k += 1
        if k - j > 1:
            texts = [str(cue.text) for cue in cues[j:k]]
            out.append(new_cue(_soften_inner_stops(_join(texts)), int(cues[j].start_time), int(cues[k - 1].end_time)))
            joins += k - j - 1
        elif _periodic_unit(content_key(str(cues[j].text))) and _soften_inner_stops(str(cues[j].text)) != cues[j].text:
            # "ありがとうございます。ありがとうございます。" as one line: normalize would split it again by sentence
            out.append(new_cue(_soften_inner_stops(str(cues[j].text)), int(cues[j].start_time), int(cues[j].end_time)))
        else:
            out.append(cues[j])
        j = k
    return out, joins


def _soften_inner_stops(text: str) -> str:
    """Full stops between copies become commas (the last one stays); content characters are untouched."""
    return re.sub(r"[。．｡](?=.*\w)", "、", text)


def _periodic_unit(key: str) -> str:
    """The shortest unit (MIN..MAX_PHRASE_CHARS) that ``key`` is two or more whole copies of, else ''."""
    for size in range(MIN_PHRASE_CHARS, min(MAX_PHRASE_CHARS, len(key) // 2) + 1):
        if len(key) % size == 0 and key[:size] * (len(key) // size) == key:
            return key[:size]
    return ""


def rejoin_split_copies(cues: list[Any], new_cue: Callable[[str, int, int], Any]) -> tuple[list[Any], int]:
    """Undo length-limit cuts that fall inside a copy of a repeated phrase ("ありがとうございますありがとう" |
    "ございますありがとうございます"), so later steps see whole copies. Returns the cues and the number of joins."""
    out: list[Any] = []
    joins = 0
    for cue in cues:
        if out:
            prev = out[-1]
            left = content_key(str(prev.text))
            unit = _periodic_unit(left + content_key(str(cue.text)))
            if (
                unit
                and len(left) % len(unit)
                and int(cue.start_time) - int(prev.end_time) <= MAX_GAP_MS
                and int(cue.end_time) - int(prev.start_time) <= MAX_CHAIN_MS
            ):
                out[-1] = new_cue(str(prev.text).rstrip() + str(cue.text).strip(), int(prev.start_time),
                                  int(cue.end_time))
                joins += 1
                continue
        out.append(cue)
    return out, joins


def _repeat_unit(text: str) -> str:
    """The phrase a cue repeats: the unit it is whole copies of, else its whole key if short."""
    return _periodic_unit(content_key(text)) or _phrase(text)


def char_spans(tokens: Sequence[Any], cues: Sequence[Any]) -> list[list[tuple[int, int]]]:
    """Per cue, one (start_ms, end_ms) per content character of its tokens (.text/.start_time/.end_time), in text
    order; a token spelled with n characters contributes n copies of its span."""
    from qwen_asr.speaker.assign import map_tokens_to_cues

    where = map_tokens_to_cues([str(t.text) for t in tokens], [str(c.text) for c in cues])
    placed: list[list[tuple[int, int, int, int]]] = [[] for _ in cues]
    for order, (token, hit) in enumerate(zip(tokens, where, strict=True)):
        if hit is not None:
            placed[hit[0]].append((hit[1], order, int(token.start_time), int(token.end_time)))
    out: list[list[tuple[int, int]]] = []
    for items in placed:
        spans: list[tuple[int, int]] = []
        for _, order, start, end in sorted(items):
            spans += [(start, end)] * max(len(content_key(str(tokens[order].text))), 1)
        out.append(spans)
    return out
