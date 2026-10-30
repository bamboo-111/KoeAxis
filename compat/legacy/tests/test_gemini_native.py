from __future__ import annotations

import json

import httpx
import pytest

from optimizer import gemini_native, llm_client
from optimizer.exceptions import LLMAuthError, LLMConnectionError, LLMRateLimitError
from qwen_asr.credentials import resolve_llm_api_key


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://relay.example/v1beta", True),
        ("https://relay.example/v1beta/", True),
        ("https://generativelanguage.googleapis.com", True),
        ("gemini+https://relay.example", True),
        ("https://api.xiaomimimo.com/v1", False),
        ("https://relay.example/v1", False),
        ("", False),
    ],
)
def test_is_gemini_native_url(url, expected):
    assert gemini_native.is_gemini_native_url(url) is expected


def test_native_api_root_variants():
    assert gemini_native.native_api_root("https://h.example/v1beta/") == "https://h.example/v1beta"
    assert gemini_native.native_api_root("gemini+https://h.example") == "https://h.example/v1beta"
    assert gemini_native.native_api_root("https://h.example/v1beta/models/x") == "https://h.example/v1beta"


def test_build_payload_maps_roles_json_and_mimo_thinking():
    payload = gemini_native.build_payload(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "user", "content": "u3"},
        ],
        temperature=0.3,
        require_json=True,
        disable_thinking=False,
        extra_body={"thinking": {"type": "disabled"}, "unknown": 1,
                    "generationConfig": {"topP": 0.9}},
        max_tokens=64,
    )
    assert payload["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert [c["role"] for c in payload["contents"]] == ["user", "model", "user"]
    assert payload["contents"][2]["parts"] == [{"text": "u2"}, {"text": "u3"}]
    config = payload["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["maxOutputTokens"] == 64
    assert config["topP"] == 0.9
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}
    assert "unknown" not in payload and "thinking" not in payload


def _transport(status: int, body: dict, seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler)


def test_generate_content_converts_response_and_skips_thoughts():
    seen: list = []
    body = {
        "responseId": "r1",
        "modelVersion": "gemini-3.8-flash-n",
        "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [
            {"text": "thinking...", "thought": True},
            {"text": '```json\n{"1": "你好"}\n```', "thoughtSignature": "x"},
        ]}}],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                          "thoughtsTokenCount": 7, "totalTokenCount": 22, "cachedContentTokenCount": 4},
    }
    response = gemini_native.generate_content(
        base_url="https://h.example/v1beta", api_key="k", model="gemini-3.8-flash",
        payload={"contents": []}, timeout=5, transport=_transport(200, body, seen),
    )
    assert str(seen[0].url) == "https://h.example/v1beta/models/gemini-3.8-flash:generateContent"
    assert seen[0].headers["x-goog-api-key"] == "k"
    assert response.choices[0].message.content == '```json\n{"1": "你好"}\n```'
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 12
    assert response.usage.completion_tokens_details.reasoning_tokens == 7
    assert response.usage.prompt_tokens_details.cached_tokens == 4


@pytest.mark.parametrize(
    ("status", "error"),
    [(401, LLMAuthError), (429, LLMRateLimitError), (503, LLMRateLimitError), (400, LLMConnectionError)],
)
def test_generate_content_maps_errors(status, error):
    with pytest.raises(error):
        gemini_native.generate_content(
            base_url="https://h.example/v1beta", api_key="k", model="m", payload={},
            timeout=5, transport=_transport(status, {"error": {"message": "boom"}}, []),
        )


def test_call_llm_routes_native_urls(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return gemini_native._to_chat_completion(
            {"candidates": [{"content": {"parts": [{"text": '{"1": "译文"}'}]}}]}, kwargs["model"],
        )

    monkeypatch.setattr(llm_client, "gemini_generate_content", fake_generate)
    response = llm_client.call_llm(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        model="gemini-3.8-flash", temperature=0.3, base_url="https://h.example/v1beta",
        api_key="k", require_json=True, max_tokens=32,
    )
    assert json.loads(response.choices[0].message.content) == {"1": "译文"}
    assert captured["payload"]["generationConfig"]["maxOutputTokens"] == 32
    assert "/no_think" not in json.dumps(captured["payload"])


def test_resolve_key_uses_gemini_key_for_native_or_configured_host(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gem")
    monkeypatch.setenv("LLM_API_KEY", "generic")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://relay.example/")
    assert resolve_llm_api_key(None, "https://other.example/v1beta") == "gem"
    assert resolve_llm_api_key(None, "https://relay.example/v1") == "gem"
    assert resolve_llm_api_key(None, "https://other.example/v1") == "generic"
