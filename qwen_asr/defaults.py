from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_ALIGN_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
DEFAULT_LLM_MODEL = "mimo-v2.5"
DEFAULT_LLM_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_LLM_TIMEOUT = 120.0
DEFAULT_LLM_EXTRA_BODY_JSON = '{"thinking":{"type":"disabled"}}'
DEFAULT_LLM_CONCURRENCY = 5
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / ".model-cache"
DEFAULT_MAX_SEGMENT_SECONDS = 15.0
DEFAULT_MIN_SEGMENT_SECONDS = 2.0
