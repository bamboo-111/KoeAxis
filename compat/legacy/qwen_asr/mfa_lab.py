from __future__ import annotations

import unicodedata
from typing import Any

from qwen_asr.mfa_guards import int_or_none, range_distance_ms
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


SHORT_MFA_CANDIDATE_RESPONSES = {
    "\u306f\u3044",
    "\u3048",
    "\u3048\u3048",
    "\u3046\u3093",
    "\u3046\u3046\u3093",
    "\u3044\u3044\u3048",
    "\u99c4\u76ee",
    "\u3060\u3081",
}


def clean_mfa_lab_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text.replace("\n", " ").replace("\r", " "))
    kept: list[str] = []
    for char in normalized:
        if char.isspace():
            kept.append(" ")
            continue
        category = unicodedata.category(char)
        if category[0] in {"P", "S", "M"}:
            kept.append(" ")
            continue
        kept.append(char)
    return " ".join("".join(kept).split()).strip()


def choose_mfa_lab_text(work_paths: WorkPaths, candidate: dict[str, Any]) -> dict[str, str]:
    candidate_text = str(candidate.get("text", "")).strip()
    cleaned_candidate = clean_mfa_lab_text(candidate_text)
    normalized_candidate = normalize_mfa_candidate_lab_text(cleaned_candidate)
    if normalized_candidate and normalized_candidate != cleaned_candidate:
        return {"text": normalized_candidate, "source": "candidate-normalized"}
    if looks_like_japanese_for_mfa(cleaned_candidate):
        return {"text": candidate_text, "source": "candidate"}
    fallback = nearest_manifest_text(work_paths, candidate)
    if fallback:
        return {"text": fallback, "source": "nearest-manifest"}
    return {"text": candidate_text, "source": "candidate"}


def normalize_mfa_candidate_lab_text(cleaned_text: str) -> str:
    parts = [part for part in cleaned_text.split() if part]
    if len(parts) < 2:
        return cleaned_text
    last = parts[-1]
    if last not in SHORT_MFA_CANDIDATE_RESPONSES:
        return cleaned_text
    prefix = parts[:-1]
    if all(is_isolated_kana_fragment(part) for part in prefix):
        return last
    return cleaned_text


def is_isolated_kana_fragment(text: str) -> bool:
    return 0 < len(text) <= 2 and all("\u3040" <= char <= "\u309f" for char in text)


def needs_manifest_lab_fallback(text: str) -> bool:
    if not text.strip():
        return True
    normalized = unicodedata.normalize("NFKC", text)
    for char in normalized:
        if char.isspace():
            continue
        category = unicodedata.category(char)
        if category[0] in {"S", "M"}:
            return True
        if category[0] == "P" and char not in {"\u3001", "\u3002", "\uff01", "\uff1f"}:
            return True
    return False


def looks_like_japanese_for_mfa(text: str) -> bool:
    if not text.strip():
        return False
    kana_count = sum(1 for char in text if "\u3040" <= char <= "\u30ff")
    return kana_count > 0


def nearest_manifest_text(work_paths: WorkPaths, candidate: dict[str, Any]) -> str:
    start_ms = candidate.get("start_ms")
    end_ms = candidate.get("end_ms")
    if not isinstance(start_ms, int) or not isinstance(end_ms, int):
        return ""
    best: tuple[int, str] | None = None
    for path in [work_paths.mimo_proofread_manifest, work_paths.translated_manifest, work_paths.split_manifest]:
        payload = read_json(path, default={})
        items = payload.values() if isinstance(payload, dict) else payload if isinstance(payload, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_start = int_or_none(item.get("start_time"))
            item_end = int_or_none(item.get("end_time"))
            text = str(item.get("original_subtitle", "")).strip()
            if item_start is None or item_end is None or not text or not looks_like_japanese_for_mfa(text):
                continue
            distance = range_distance_ms(start_ms, end_ms, item_start, item_end)
            if best is None or distance < best[0]:
                best = (distance, text)
    if best is None:
        return ""
    return best[1] if best[0] <= 2500 else ""
