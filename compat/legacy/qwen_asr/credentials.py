from __future__ import annotations

import os
from urllib.parse import urlparse


def resolve_llm_api_key(explicit_key: str | None, base_url: str | None) -> str:
    """Resolve a generic LLM key without ever persisting it in project state."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    normalized_url = (base_url or "").lower()
    if "xiaomimimo.com" in normalized_url:
        return os.environ.get("MIMO_API_KEY", "").strip()
    if "deepseek.com" in normalized_url:
        return os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if "qhaigc.net" in normalized_url or _is_gemini_url(normalized_url):
        return os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("QUANHEX_API_KEY", "").strip()
    return os.environ.get("LLM_API_KEY", "").strip()


def _host(url: str) -> str:
    url = url.strip().lower()
    if url.startswith("gemini+"):
        url = url[len("gemini+"):]
    return urlparse(url).netloc


def _is_gemini_url(normalized_url: str) -> bool:
    """Native Gemini endpoints, or the host configured in ``GEMINI_BASE_URL``."""
    if not normalized_url:
        return False
    if normalized_url.startswith("gemini+") or "generativelanguage.googleapis.com" in normalized_url:
        return True
    if "/v1beta" in normalized_url or "/v1alpha" in normalized_url:
        return True
    gemini_host = _host(os.environ.get("GEMINI_BASE_URL", ""))
    return bool(gemini_host) and gemini_host == _host(normalized_url)


def resolve_mimo_api_key(explicit_key: str | None = None) -> str:
    """MiMo audio proofread is intentionally isolated from Gemini-compatible keys."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    return os.environ.get("MIMO_API_KEY", "").strip()


MIMO_PAYG_HOST = "api.xiaomimimo.com"
MIMO_TOKEN_PLAN_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


def is_mimo_token_plan_key(api_key: str | None) -> bool:
    key = (api_key or "").strip()
    return key.startswith("tp-") or key.startswith("ttp-")


def resolve_llm_base_url(base_url: str | None, api_key: str | None) -> str:
    """Route MiMo Token Plan keys (tp-/ttp-) to the Token Plan host.

    Token Plan keys are rejected by the pay-as-you-go host, so a stored or default
    ``api.xiaomimimo.com`` URL is rewritten to ``MIMO_BASE_URL`` (if set) or the CN
    Token Plan endpoint. Any other URL is returned unchanged.
    """
    url = (base_url or "").strip()
    if MIMO_PAYG_HOST in url.lower() and is_mimo_token_plan_key(api_key):
        return os.environ.get("MIMO_BASE_URL", "").strip() or MIMO_TOKEN_PLAN_BASE_URL
    return url
