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


@dataclass
class SpeakerAssigner:
    """Speaker decisions for one episode: word tokens (text, start_ms, end_ms) scored on a SpeakerTrack."""

    members: list[str]
    token_texts: list[str]
    token_times: list[tuple[int, int]]  # ms
    token_scores: np.ndarray  # (tokens, members)
    min_margin: float = DEFAULT_MIN_MARGIN

    @classmethod
    def from_track(cls, track: Any, tokens: Sequence[Any], min_margin: float = DEFAULT_MIN_MARGIN,
                   pad: float = 0.05) -> SpeakerAssigner:
        """tokens: objects with .text, .start_time, .end_time in ms (optimizer ASRDataSeg)."""
        texts = [str(t.text) for t in tokens]
        times = [(int(t.start_time), int(t.end_time)) for t in tokens]
        scores = np.zeros((len(texts), len(track.members)), np.float32)
        for i, (start, end) in enumerate(times):
            scores[i] = track.span_scores(start / 1000, end / 1000, pad)
        return cls(list(track.members), texts, times, scores, min_margin)

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

    def split_cues(self, cues: Sequence[Any], new_cue: Callable[[str, int, int], Any]) -> tuple[list[Any], int]:
        """Cut cues at confident speaker changes. new_cue(text, start_ms, end_ms) builds a piece.

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
            labels = vote_labels(self.token_scores[idx], self.min_margin) if idx else []
            runs = _runs(labels)
            # cut positions (char index of each later run's first token); runs whose text position does not move
            # forward (char order disagrees with time order) are not cut
            bounds: list[tuple[int, int, int]] = []  # (char cut, end_ms of previous token, start_ms of next token)
            last = 0
            for _, (a2, _) in zip(runs, runs[1:], strict=False):
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
