"""Optional Qwen3-ASR context prompt: a bare term list plus a guard against the model copying it.

A/B on a 167-segment radio episode: a sentence-style prompt was copied into 10 segments, a bare
``用語: a, b, c`` list into 1. The guard below catches that copy (three or more prompt terms in a
comma/読点 separated run, or the ``用語:`` header itself) so the caller can re-transcribe the
segment without context. Only terms with a fixed reading (band, station, show, corner names)
belong in the list; member names and e-mail addresses did not help and raise the leak risk.
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

CONTEXT_PREFIX = "用語: "
MIN_LEAK_RUN = 3
_TERM_SPLIT_RE = re.compile(r"[,，、\n\r]+")
_PREFIX_LEAK_RE = re.compile(r"用語\s*[:：]")


def parse_context_terms(raw: str | Iterable[str] | None) -> list[str]:
    """Split a comma/読点/newline separated string (or iterable of such) into unique terms."""
    if raw is None:
        return []
    chunks = [raw] if isinstance(raw, str) else list(raw)
    terms: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        for piece in _TERM_SPLIT_RE.split(str(chunk)):
            term = piece.strip()
            if not term or term.startswith("#"):
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(term)
    return terms


def load_context_terms(terms: str | None = None, terms_file: str | Path | None = None) -> list[str]:
    """Terms from a UTF-8 file (one or more per line, ``#`` comments) followed by inline terms."""
    chunks: list[str] = []
    if terms_file:
        path = Path(terms_file)
        if not path.exists():
            raise FileNotFoundError(f"ASR context terms file not found: {path}")
        chunks.extend(
            line for line in path.read_text(encoding="utf-8-sig").splitlines() if not line.strip().startswith("#")
        )
    if terms:
        chunks.append(terms)
    return parse_context_terms(chunks)


def build_context_prompt(terms: Iterable[str]) -> str:
    terms = list(terms)
    return CONTEXT_PREFIX + ", ".join(terms) if terms else ""


def detect_context_leak(text: str, terms: Iterable[str], *, min_run: int = MIN_LEAK_RUN) -> bool:
    """True when the transcript looks like a copy of the prompt rather than speech."""
    terms = tuple(terms)
    if not text or not terms:
        return False
    if _PREFIX_LEAK_RE.search(text):
        return True
    return _leak_pattern(terms, max(2, int(min_run))).search(text) is not None


def merge_context_edits(base_text: str, context_text: str, terms: Iterable[str]) -> str:
    """Keep the no-context transcript and copy over only the context edits that land on a prompt term.

    The context prompt also perturbs decoding far away from the terms: on radio episodes 43/44 it
    changed ~55% of segments, mostly in term-unrelated words, and a blind listening judge could not
    reliably tell whether those changes helped. Restricting the edit to the term spans kept the term
    fixes while the segments differing from the plain transcript dropped from 127/159 to 18/19.
    A prompt copy (see ``detect_context_leak``) keeps the plain transcript unchanged.
    """
    terms = tuple(terms)
    if not terms or not context_text or context_text == base_text or detect_context_leak(context_text, terms):
        return base_text
    spans = [
        (match.start(), match.end())
        for match in _term_pattern(terms).finditer(context_text)
    ]
    if not spans:
        return base_text
    pieces: list[str] = []
    matcher = difflib.SequenceMatcher(None, base_text, context_text, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            pieces.append(base_text[i1:i2])
            continue
        if j2 > j1:
            on_term = any(j1 < end and j2 > start for start, end in spans)
        else:  # pure deletion: only inside a term span
            on_term = any(start < j1 < end for start, end in spans)
        pieces.append(context_text[j1:j2] if on_term else base_text[i1:i2])
    return "".join(pieces)


@lru_cache(maxsize=16)
def _term_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True))
    return re.compile(alternation, re.IGNORECASE)


@lru_cache(maxsize=16)
def _leak_pattern(terms: tuple[str, ...], min_run: int) -> re.Pattern[str]:
    alternation = "|".join(re.escape(term) for term in sorted(set(terms), key=len, reverse=True))
    term = f"(?:{alternation})"
    return re.compile(rf"{term}(?:\s*[,，、]\s*{term}){{{min_run - 1},}}", re.IGNORECASE)
