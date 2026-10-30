from __future__ import annotations

from qwen_asr.credentials import resolve_llm_api_key, resolve_mimo_api_key


def test_mimo_key_does_not_fall_back_to_quanhex(monkeypatch) -> None:
    monkeypatch.delenv("MIMO_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "quanhex-key")
    assert resolve_mimo_api_key() == ""


def test_llm_key_uses_provider_specific_environment(monkeypatch) -> None:
    monkeypatch.setenv("MIMO_API_KEY", "mimo-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("QUANHEX_API_KEY", "quanhex-key")
    assert resolve_llm_api_key(None, "https://api.xiaomimimo.com/v1") == "mimo-key"
    assert resolve_llm_api_key(None, "https://api.qhaigc.net") == "quanhex-key"


def test_llm_key_uses_deepseek_environment(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")
    assert resolve_llm_api_key(None, "https://api.deepseek.com") == "deepseek-key"


def test_token_plan_key_routes_payg_url_to_token_plan_host(monkeypatch) -> None:
    from qwen_asr.credentials import MIMO_TOKEN_PLAN_BASE_URL, resolve_llm_base_url

    monkeypatch.delenv("MIMO_BASE_URL", raising=False)
    payg = "https://api.xiaomimimo.com/v1"
    assert resolve_llm_base_url(payg, "tp-abc") == MIMO_TOKEN_PLAN_BASE_URL
    assert resolve_llm_base_url(payg, "ttp-abc") == MIMO_TOKEN_PLAN_BASE_URL
    assert resolve_llm_base_url(payg, "sk-abc") == payg
    assert resolve_llm_base_url("https://api.deepseek.com", "tp-abc") == "https://api.deepseek.com"
    sgp = "https://token-plan-sgp.xiaomimimo.com/v1"
    assert resolve_llm_base_url(sgp, "tp-abc") == sgp
    monkeypatch.setenv("MIMO_BASE_URL", sgp)
    assert resolve_llm_base_url(payg, "tp-abc") == sgp
