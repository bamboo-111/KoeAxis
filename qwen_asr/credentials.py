from __future__ import annotations

import os


def resolve_llm_api_key(explicit_key: str | None, base_url: str | None) -> str:
    """Resolve a generic LLM key without ever persisting it in project state."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    normalized_url = (base_url or "").lower()
    if "xiaomimimo.com" in normalized_url:
        return os.environ.get("MIMO_API_KEY", "").strip()
    if "qhaigc.net" in normalized_url:
        return os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("QUANHEX_API_KEY", "").strip()
    return os.environ.get("LLM_API_KEY", "").strip()


def resolve_mimo_api_key(explicit_key: str | None = None) -> str:
    """MiMo audio proofread is intentionally isolated from Gemini-compatible keys."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    return os.environ.get("MIMO_API_KEY", "").strip()
