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
