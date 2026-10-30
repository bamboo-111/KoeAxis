from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGN_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

# Text LLM baseline (split / translate / correct). Switched from MiMo after the 2026-10 comparison on
# episode 30: gemini-3.8-flash recovered ASR-garbled names better, produced fewer mid-word split cuts and
# no shifted translation batch. The native Gemini endpoint is selected by the ``/v1beta`` URL
# (optimizer.gemini_native); point ``GEMINI_BASE_URL`` at a relay, otherwise Google is used.
# 2026-10-19: the project relay (https://113.20.8.48) serves this model only as ``gemini-3.8-flash-high``;
# it still honours ``thinkingLevel`` (low/medium/high), and without one it thinks like the plain name did.
DEFAULT_LLM_MODEL = "gemini-3.8-flash-high"
GOOGLE_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def default_gemini_base_url(env_value: str | None = None) -> str:
    """``GEMINI_BASE_URL`` as a native ``/v1beta`` root, or the Google endpoint when unset."""
    raw = (os.environ.get("GEMINI_BASE_URL", "") if env_value is None else env_value).strip().rstrip("/")
    if not raw:
        return GOOGLE_GEMINI_BASE_URL
    lowered = raw.lower()
    if "/v1beta" in lowered or "/v1alpha" in lowered:
        return raw
    if lowered.endswith("/v1"):
        raw = raw[: -len("/v1")]
    return f"{raw}/v1beta"


DEFAULT_LLM_BASE_URL = default_gemini_base_url()
# Most Gemini relays cap a key at ~5 in-flight requests; 4 kept episode 30 free of HTTP 429.
DEFAULT_LLM_CONCURRENCY = 4

# Translate prompt versions (optimizer/prompts/translate/*.md) and read-only neighbour context per batch.
TRANSLATE_PROMPT_VERSIONS = ("v1", "v2")
DEFAULT_TRANSLATE_PROMPT = "v1"
DEFAULT_TRANSLATE_CONTEXT_LINES = 0
# correct: "rules" = deterministic cleanup only (default since 2026-07); "llm" = ASRCorrector first.
DEFAULT_CORRECT_MODE = "rules"
# llm correct: v1 = whole-segment rewrite; v2 = span edits with confidence + read-only neighbour segments.
# #15/#18 end-to-end (2026-10-20): llm correct cut translation errors ~20-25% vs rules; v2 ~= v1, auditable.
CORRECT_PROMPT_VERSIONS = ("v1", "v2")
DEFAULT_CORRECT_PROMPT = "v2"
DEFAULT_CORRECT_CONTEXT_SEGMENTS = 1
DEFAULT_CORRECT_APPLY_THRESHOLD = 0.8
# "" = follow --disable-thinking / --llm-extra-body-json; low/medium/high sets thinking for correct only.
DEFAULT_CORRECT_EFFORT = ""

# MiMo audio proofread stays on MiMo (separate key, endpoint and concurrency).
DEFAULT_MIMO_MODEL = "mimo-v2.6-pro"
DEFAULT_MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_MIMO_CONCURRENCY = 10
# Split semantic merge only picks adjacent-cue groups; for MiMo pipelines Flash merged more half-sentences
# than Pro on episodes 46/47 with no failed batches, at about a third of the price.
DEFAULT_SPLIT_LLM_MODEL = "mimo-v2.6-flash"

DEFAULT_LLM_TIMEOUT = 120.0
# MiMo's "thinking disabled"; the Gemini adapter maps it to thinkingLevel=low.
DEFAULT_LLM_EXTRA_BODY_JSON = '{"thinking":{"type":"disabled"}}'
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / ".model-cache"
DEFAULT_MAX_SEGMENT_SECONDS = 15.0
DEFAULT_MIN_SEGMENT_SECONDS = 2.0
