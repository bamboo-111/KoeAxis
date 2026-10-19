"""Gemini native ``generateContent`` adapter for :func:`optimizer.llm_client.call_llm`.

Some relays (and Google itself) only expose Gemini models through the native
``/v1beta/models/{model}:generateContent`` endpoint. Selecting it is URL-driven
so every caller of ``call_llm`` keeps working unchanged:

* a base URL whose path contains ``/v1beta`` or ``/v1alpha``
  (e.g. ``https://relay.example/v1beta``), or
* any ``generativelanguage.googleapis.com`` URL, or
* an explicit ``gemini+https://host`` scheme prefix (path defaults to ``/v1beta``).

The response is converted into an OpenAI ``ChatCompletion`` so downstream
cleanup, validation and performance accounting stay identical.
"""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
from openai.types.chat import ChatCompletion

from optimizer.exceptions import (
    LLMAuthError,
    LLMConnectionError,
    LLMRateLimitError,
)

GEMINI_SCHEME_PREFIX = "gemini+"
# Top-level request fields accepted by generateContent; anything else in
# llm_extra_body (e.g. MiMo's ``thinking``) would be rejected with HTTP 400.
_NATIVE_TOP_LEVEL_FIELDS = {
    "generationConfig",
    "safetySettings",
    "systemInstruction",
    "tools",
    "toolConfig",
    "cachedContent",
    "labels",
}
DISABLED_THINKING_LEVEL = "low"
THINKING_LEVELS = ("low", "medium", "high")
AUDIO_MIME_SUBTYPES = {"mp3": "mpeg", "m4a": "mp4"}


def is_gemini_native_url(base_url: str | None) -> bool:
    url = (base_url or "").strip().lower()
    if not url:
        return False
    if url.startswith(GEMINI_SCHEME_PREFIX):
        return True
    parsed = urlparse(url)
    if parsed.netloc.endswith("generativelanguage.googleapis.com"):
        return True
    return "/v1beta" in parsed.path or "/v1alpha" in parsed.path


def native_api_root(base_url: str) -> str:
    """Return ``scheme://host/<version>`` without a trailing slash."""
    url = base_url.strip()
    if url.lower().startswith(GEMINI_SCHEME_PREFIX):
        url = url[len(GEMINI_SCHEME_PREFIX):]
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    for marker in ("/models", "/openai"):
        index = path.find(marker)
        if index != -1:
            path = path[:index]
    if not path:
        path = "/v1beta"
    return urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _message_parts(content: Any) -> list[dict[str, Any]]:
    """Native parts of one message: text items joined as before, OpenAI ``input_audio`` as ``inlineData``."""
    if not isinstance(content, list):
        return [{"text": _message_text(content)}]
    parts: list[dict[str, Any]] = []
    texts: list[str] = []
    for item in content:
        audio = item.get("input_audio") if isinstance(item, dict) and item.get("type") == "input_audio" else None
        if not isinstance(audio, dict):
            texts.append(_message_text([item]))
            continue
        if texts:
            parts.append({"text": "\n".join(texts)})
            texts = []
        audio_format = str(audio.get("format") or "wav").lower().lstrip(".")
        parts.append({"inlineData": {"mimeType": f"audio/{AUDIO_MIME_SUBTYPES.get(audio_format, audio_format)}",
                                     "data": str(audio.get("data", ""))}})
    if texts or not parts:
        parts.append({"text": "\n".join(texts)})
    return parts


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_payload(
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    require_json: bool,
    disable_thinking: bool,
    extra_body: dict[str, Any] | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    system_parts: list[dict[str, str]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            text = _message_text(message.get("content"))
            if text:
                system_parts.append({"text": text})
            continue
        gemini_role = "model" if role == "assistant" else "user"
        parts = _message_parts(message.get("content"))
        if contents and contents[-1]["role"] == gemini_role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": gemini_role, "parts": parts})

    generation_config: dict[str, Any] = {"temperature": temperature}
    if require_json:
        generation_config["responseMimeType"] = "application/json"
    if max_tokens is not None:
        generation_config["maxOutputTokens"] = int(max_tokens)

    extra = dict(extra_body or {})
    thinking = extra.pop("thinking", None)
    if disable_thinking or (isinstance(thinking, dict) and thinking.get("type") == "disabled"):
        generation_config["thinkingConfig"] = {"thinkingLevel": DISABLED_THINKING_LEVEL}
    # OpenAI-style effort (what the OpenAI-compatible relays honour) wins over the generic disable flag.
    effort = str(extra.pop("reasoning_effort", "") or "").strip().lower()
    if effort in THINKING_LEVELS:
        generation_config["thinkingConfig"] = {"thinkingLevel": effort}

    payload: dict[str, Any] = {"contents": contents, "generationConfig": generation_config}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    passthrough = {key: value for key, value in extra.items() if key in _NATIVE_TOP_LEVEL_FIELDS}
    return _deep_merge(payload, passthrough)


def _to_chat_completion(data: dict[str, Any], model: str) -> ChatCompletion:
    candidates = data.get("candidates") or []
    text = ""
    finish_reason = "stop"
    if candidates:
        candidate = candidates[0] or {}
        parts = ((candidate.get("content") or {}).get("parts")) or []
        text = "".join(
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict) and not part.get("thought")
        )
        reason = str(candidate.get("finishReason") or "STOP").upper()
        finish_reason = "length" if reason == "MAX_TOKENS" else (
            "content_filter" if reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"} else "stop"
        )
    usage_meta = data.get("usageMetadata") or {}
    prompt_tokens = int(usage_meta.get("promptTokenCount") or 0)
    output_tokens = int(usage_meta.get("candidatesTokenCount") or 0)
    reasoning_tokens = int(usage_meta.get("thoughtsTokenCount") or 0)
    cached_tokens = usage_meta.get("cachedContentTokenCount")
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens + reasoning_tokens,
        "total_tokens": int(usage_meta.get("totalTokenCount") or (prompt_tokens + output_tokens + reasoning_tokens)),
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached_tokens)}
    return ChatCompletion.model_validate({
        "id": str(data.get("responseId") or "gemini-native"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(data.get("modelVersion") or model),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": text or None},
        }],
        "usage": usage,
    })


def _error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("status") or error)
    return json.dumps(body, ensure_ascii=False)[:500]


def generate_content(
    *,
    base_url: str,
    api_key: str,
    model: str,
    payload: dict[str, Any],
    timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> ChatCompletion:
    url = f"{native_api_root(base_url)}/models/{model}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, headers=headers, json=payload)
    except httpx.TimeoutException as e:
        raise LLMConnectionError(detail=f"请求超时: {e}") from e
    except httpx.HTTPError as e:
        raise LLMConnectionError(detail=str(e)) from e

    status = response.status_code
    if status >= 400:
        message = _error_message(response)
        lowered = message.lower()
        if status in {401, 403}:
            raise LLMAuthError(detail=f"HTTP {status}: {message}")
        if status == 429 or status >= 500 or "upstream" in lowered or "try again later" in lowered:
            raise LLMRateLimitError(detail=f"上游暂时性错误 (HTTP {status}): {message}")
        raise LLMConnectionError(detail=f"HTTP {status}: {message}")
    try:
        data = response.json()
    except ValueError as e:
        raise LLMConnectionError(detail=f"Gemini 响应不是 JSON: {response.text[:200]}") from e
    return _to_chat_completion(data, model)
