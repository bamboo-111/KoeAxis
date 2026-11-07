from __future__ import annotations

import re

from optimizer.fixed_terms import (
    _normalize_export_email_variants,
    _normalize_radio_addresses,
    _normalize_radio_fixed_terms,
)

_ASCII_LETTER_SPELLING_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z])(?:\s+([A-Za-z])){2,}(?![A-Za-z0-9])"
)

_DATE_COMPACT_RE = re.compile(r"(?<!\d)(\d{1,4})\s*([年月日])")

_SPLIT_NUMBER_BEFORE_UNIT_RE = re.compile(r"(?<!\d)(\d(?:\s+\d)+)\s*([年月日时分秒])")

_SPLIT_NUMBER_AFTER_UNIT_RE = re.compile(r"([年月日时分秒])\s*(\d(?:\s+\d)+)(?=\s*[年月日时分秒]|$)")

_EMAIL_SPACING_RE = re.compile(
    r"(?i)([a-z0-9](?:[a-z0-9._+-]|\s){1,})\s*@\s*([a-z0-9](?:[a-z0-9.-]|\s){1,}\.\s*[a-z](?:[a-z]|\s){1,})"
)

_AT_EMAIL_SPACING_RE = re.compile(
    r"(?i)\b([a-z0-9._+-]{3,})\s+at\s+([a-z0-9](?:[a-z0-9.-]|\s){1,}\.\s*[a-z](?:[a-z]|\s){1,})"
)

def clean_subtitle_text(text: str) -> str:
    """Normalize low-risk subtitle text artifacts without changing wording."""
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"[ \t\u3000]+", " ", cleaned)
    cleaned = _compact_split_numbers(cleaned)
    cleaned = _compact_date_units(cleaned)
    cleaned = _compact_spelled_ascii(cleaned)
    cleaned = _compact_email_spacing(cleaned)
    cleaned = _normalize_export_email_variants(cleaned)
    cleaned = _normalize_radio_fixed_terms(cleaned)
    cleaned = _cleanup_punctuation_spacing(cleaned)
    return cleaned.strip()

def clean_asr_correction_text(text: str) -> str:
    """Apply deterministic ASR cleanup for recurring radio subtitle artifacts."""
    cleaned = clean_subtitle_text(text)
    cleaned = _normalize_radio_addresses(cleaned)
    cleaned = _normalize_radio_fixed_terms(cleaned)
    return clean_subtitle_text(cleaned)

def _compact_split_numbers(text: str) -> str:
    def before_unit(match: re.Match[str]) -> str:
        return f"{match.group(1).replace(' ', '')}{match.group(2)}"

    def after_unit(match: re.Match[str]) -> str:
        return f"{match.group(1)}{match.group(2).replace(' ', '')}"

    previous = None
    current = text
    while previous != current:
        previous = current
        current = _SPLIT_NUMBER_BEFORE_UNIT_RE.sub(before_unit, current)
        current = _SPLIT_NUMBER_AFTER_UNIT_RE.sub(after_unit, current)
    return current

def _compact_date_units(text: str) -> str:
    compacted = _DATE_COMPACT_RE.sub(r"\1\2", text)
    compacted = re.sub(r"(?<=\d[年月日])\s+(?=\d)", "", compacted)
    return compacted

def _compact_spelled_ascii(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        letters = re.findall(r"[A-Za-z]", match.group(0))
        return "".join(letters)

    return _ASCII_LETTER_SPELLING_RE.sub(repl, text)

def _compact_email_spacing(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        local = re.sub(r"\s+", "", match.group(1))
        domain = re.sub(r"\s+", "", match.group(2))
        return f"{local}@{domain}"

    compacted = _AT_EMAIL_SPACING_RE.sub(repl, text)
    compacted = _EMAIL_SPACING_RE.sub(repl, compacted)
    return re.sub(r"\s*@\s*", "@", compacted)

def _cleanup_punctuation_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?，。！？、；：])", r"\1", text)
    text = re.sub(r"([（「『《【])\s+", r"\1", text)
    text = re.sub(r"\s+([）」』》】])", r"\1", text)
    return text
