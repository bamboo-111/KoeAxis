"""Speaker labels for word tokens and subtitle cues, and splitting cues where the speaker confidently changes.

Every token gets the member scores averaged over its time span (``SpeakerTrack.span_scores``). Inside a cue,
consecutive tokens with the same best member form runs; a run whose mean margin (best minus runner-up score) is
below ``min_margin`` is merged into the neighbouring run whose member scores higher on it, repeatedly, so only
confident speaker changes remain. Overlapping speech and boundary words, where both voices are present, have small
margins and therefore do not produce cuts. On episode 30's hand-made gold slice this (min_margin 0.1) cut 47 cues,
reduced missed speaker changes from 33/79 to 23/79 and broke no correct cue.

The cue text is never changed: a cut is placed before the first character of the next run's first token, and the
inner boundary times are that token's start / the previous token's end.
"""
from __future__ import annotations

import difflib
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

DEFAULT_MIN_MARGIN = 0.1
# Repeated phrases inside a cue ("ありがとうございますありがとうございます") are cut between copies when both copies
# confidently belong to different members. On episode 30's gold slice (rule split, no LLM merge) 0.08 recovered 2 of
# the 23 missed speaker changes and broke nothing; 0.05 added wrong cuts, 0.12 and above changed nothing.
DEFAULT_REPEAT_MARGIN: float | None = 0.08
UNKNOWN = ""


@dataclass(frozen=True)
class CueSpeaker:
    speaker: str  # UNKNOWN when there are no scored tokens
    margin: float  # best minus runner-up mean score; < min_margin means "not sure"


def _norm_chars(text: str) -> list[tuple[str, int]]:
    """(normalised char, index in text) for the alphanumeric content of text (NFKC + casefold, per character)."""
    out = []
    for index, ch in enumerate(text):
        for n in unicodedata.normalize("NFKC", ch).casefold():
            if n.isalnum():
                out.append((n, index))
    return out


def map_tokens_to_cues(token_texts: Sequence[str], cue_texts: Sequence[str]) -> list[tuple[int, int] | None]:
    """For each token: (cue index, position in that cue's text of the token's first matched char), or None."""
    tchars, towner = [], []
    for i, text in enumerate(token_texts):
        chars = _norm_chars(text)
        tchars += [c for c, _ in chars]
        towner += [i] * len(chars)
    cchars, cref = [], []
    for j, text in enumerate(cue_texts):
        for c, pos in _norm_chars(text):
            cchars.append(c)
            cref.append((j, pos))
    votes: list[dict[int, list[int]]] = [{} for _ in token_texts]
    matcher = difflib.SequenceMatcher(None, tchars, cchars, autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for k in range(size):
            j, pos = cref[b + k]
            votes[towner[a + k]].setdefault(j, []).append(pos)
    lengths = [len(_norm_chars(text)) for text in token_texts]
    out: list[tuple[int, int] | None] = []
    for i, vote in enumerate(votes):
        if not vote:
            out.append(None)
            continue
        j = max(vote, key=lambda c: len(vote[c]))
        out.append((j, min(vote[j])) if len(vote[j]) * 2 >= lengths[i] else None)
    return out


def _runs(labels: list[int]) -> list[tuple[int, int]]:
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i))
            start = i
    return out


def _margin(scores: np.ndarray, member: int) -> float:
    if scores.shape[1] < 2:
        return float("inf")
    others = np.delete(scores, member, axis=1).max(1)
    return float((scores[:, member] - others).mean())


def vote_labels(scores: np.ndarray, min_margin: float = DEFAULT_MIN_MARGIN) -> list[int]:
    """Member index per token of one cue after merging unconfident runs into their neighbours."""
    labels = [int(row.argmax()) for row in scores]
    while True:
        runs = _runs(labels)
        if len(runs) <= 1:
            return labels
        weak = [(_margin(scores[a:b], labels[a]), a, b) for a, b in runs]
        weak = [item for item in weak if item[0] < min_margin]
        if not weak:
            return labels
        _, a, b = min(weak)
        k = runs.index((a, b))
        neighbours = [labels[runs[x][0]] for x in (k - 1, k + 1) if 0 <= x < len(runs)]
        mean = scores[a:b].mean(0)
        target = max(neighbours, key=lambda m: mean[m])
        labels[a:b] = [target] * (b - a)


def cue_speaker(scores: np.ndarray, members: Sequence[str]) -> CueSpeaker:
    if not len(scores):
        return CueSpeaker(UNKNOWN, 0.0)
    mean = scores.mean(0)
    best = int(mean.argmax())
    if len(members) < 2:
        return CueSpeaker(members[best], float("inf"))
    return CueSpeaker(members[best], float(mean[best] - np.delete(mean, best).max()))


def _key(texts: Sequence[str]) -> str:
    return "".join(c for text in texts for c, _ in _norm_chars(text))


def repeat_cuts(token_texts: Sequence[str], scores: np.ndarray, min_margin: float, *, max_tokens: int = 8,
                min_chars: int = 3, similarity: float = 0.75) -> list[int]:
    """Token positions inside one cue where a repeated phrase is said again by somebody else.

    Two adjacent copies A = tokens[i:k], B = tokens[k:m] (normalised texts at least ``similarity`` alike, e.g.
    "ありがとうございます" twice or "うますぎて" / "うますぎ") are cut apart at k only when each copy on its own
    confidently belongs to a different member (mean-score margin >= ``min_margin`` on both). Run voting alone cannot
    do this: the copies are short, so one of them is usually merged into the other.
    """
    n = len(token_texts)
    if scores.ndim != 2 or scores.shape[1] < 2 or n < 2:
        return []
    candidates: list[tuple[float, int, int, int]] = []  # (match quality, cut k, copy A start i, copy B end m)
    for k in range(1, n):
        best: tuple[float, int, int] | None = None
        for i in range(max(0, k - max_tokens), k):
            a = _key(token_texts[i:k])
            if len(a) < min_chars:
                continue
            for m in range(k + 1, min(n, k + max_tokens) + 1):
                b = _key(token_texts[k:m])
                if len(b) < min_chars or len(b) > len(a) * 1.5 or len(a) > len(b) * 1.5:
                    continue
                # a copy that starts like the previous one is a better cut than one shifted by a character
                quality = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
                if quality >= similarity and (best is None or quality + 0.1 * (a[0] == b[0]) > best[0]):
                    best = (quality + 0.1 * (a[0] == b[0]), i, m)
        if best is not None:
            candidates.append((best[0], k, best[1], best[2]))
    # Copy boundaries first, from the text alone: in a periodic run ("ありがとうございますありがとうございます") every
    # shifted position matches as well as the true one, so the best and then earliest match wins and the others within
    # half a copy of it are dropped. Only then the voices decide, so a mid-word position can never be cut.
    chosen: list[tuple[int, int, int]] = []
    for _, k, i, m in sorted(candidates, key=lambda item: (-item[0], item[1])):
        spacing = max(min_chars, min(len(_key(token_texts[i:k])), len(_key(token_texts[k:m]))) / 2)
        if all(len(_key(token_texts[min(k, o):max(k, o)])) >= spacing for o, _, _ in chosen):
            chosen.append((k, i, m))
    out: list[int] = []
    for k, i, m in chosen:
        left, right = cue_speaker(scores[i:k], range(scores.shape[1])), cue_speaker(scores[k:m], range(scores.shape[1]))
        if left.speaker != right.speaker and min(left.margin, right.margin) >= min_margin:
            out.append(k)
    return sorted(out)


@dataclass
class SpeakerAssigner:
    """Speaker decisions for one episode: word tokens (text, start_ms, end_ms) scored on a SpeakerTrack."""

    members: list[str]
    token_texts: list[str]
    token_times: list[tuple[int, int]]  # ms
    token_scores: np.ndarray  # (tokens, members)
    min_margin: float = DEFAULT_MIN_MARGIN
    repeat_margin: float | None = DEFAULT_REPEAT_MARGIN

    @classmethod
    def from_track(cls, track: Any, tokens: Sequence[Any], min_margin: float = DEFAULT_MIN_MARGIN,
                   pad: float = 0.05, repeat_margin: float | None = None) -> SpeakerAssigner:
        """tokens: objects with .text, .start_time, .end_time in ms (optimizer ASRDataSeg)."""
        texts = [str(t.text) for t in tokens]
        times = [(int(t.start_time), int(t.end_time)) for t in tokens]
        scores = np.zeros((len(texts), len(track.members)), np.float32)
        for i, (start, end) in enumerate(times):
            scores[i] = track.span_scores(start / 1000, end / 1000, pad)
        if repeat_margin is None:
            repeat_margin = DEFAULT_REPEAT_MARGIN
        return cls(list(track.members), texts, times, scores, min_margin, repeat_margin)

    def _tokens_by_cue(self, cue_texts: Sequence[str]) -> tuple[dict[int, list[int]], list[tuple[int, int] | None]]:
        where = map_tokens_to_cues(self.token_texts, cue_texts)
        by_cue: dict[int, list[int]] = {}
        for i, hit in enumerate(where):
            if hit is not None:
                by_cue.setdefault(hit[0], []).append(i)
        return by_cue, where

    def label_cues(self, cues: Sequence[Any]) -> list[CueSpeaker]:
        """Majority voice of each cue (mean token scores), with its margin."""
        by_cue, _ = self._tokens_by_cue([str(c.text) for c in cues])
        return [cue_speaker(self.token_scores[by_cue.get(j, [])], self.members) for j in range(len(cues))]

    def split_cues(self, cues: Sequence[Any], new_cue: Callable[[str, int, int], Any], *,
                   voice_runs: bool = True) -> tuple[list[Any], int]:
        """Cut cues at confident speaker changes. new_cue(text, start_ms, end_ms) builds a piece.

        voice_runs=False only cuts between copies of a repeated phrase (``repeat_cuts``).
        Returns the new cue list and the number of cuts.
        """
        if len(self.members) < 2:
            return list(cues), 0
        by_cue, where = self._tokens_by_cue([str(c.text) for c in cues])
        out: list[Any] = []
        cuts = 0
        for j, cue in enumerate(cues):
            idx = by_cue.get(j, [])
            text = str(cue.text)
            labels = vote_labels(self.token_scores[idx], self.min_margin) if idx and voice_runs else []
            starts = {a for a, _ in _runs(labels)[1:]}
            if self.repeat_margin is not None and idx:
                texts = [self.token_texts[i] for i in idx]
                runs = set(starts)
                for k in repeat_cuts(texts, self.token_scores[idx], self.repeat_margin):
                    # a voice cut a few characters away already separates the two copies
                    if all(len(_key(texts[min(k, s):max(k, s)])) >= 3 for s in runs):
                        starts.add(k)
            # cut positions (char index of each later run's first token); runs whose text position does not move
            # forward (char order disagrees with time order) are not cut
            bounds: list[tuple[int, int, int]] = []  # (char cut, end_ms of previous token, start_ms of next token)
            last = 0
            for a2 in sorted(starts):
                pos = where[idx[a2]][1]  # type: ignore[index]
                if pos <= last or not text[last:pos].strip() or not text[pos:].strip():
                    continue
                bounds.append((pos, self.token_times[idx[a2 - 1]][1], self.token_times[idx[a2]][0]))
                last = pos
            if not bounds:
                out.append(cue)
                continue
            start_char, start_ms = 0, int(cue.start_time)
            for pos, prev_end, next_start in bounds:
                end_ms = max(min(prev_end, next_start), start_ms + 1)
                out.append(new_cue(text[start_char:pos].strip(), start_ms, end_ms))
                start_char, start_ms = pos, max(next_start, end_ms)
                cuts += 1
            out.append(new_cue(text[start_char:].strip(), start_ms, max(int(cue.end_time), start_ms + 1)))
        return out, cuts
