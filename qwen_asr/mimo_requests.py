from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any, Callable


def maybe_disable_thinking_text(text: str, config: Any) -> str:
    if not config.disable_thinking:
        return text
    return "/no_think\n" + text


def chat_completion_create(
    *,
    client: Any,
    config: Any,
    messages: list[dict[str, Any]],
) -> Any:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.extra_body is not None:
        kwargs["extra_body"] = config.extra_body
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        if "extra_body" not in kwargs:
            raise
        kwargs.pop("extra_body", None)
        return client.chat.completions.create(**kwargs)


def compact_schema_prompt(config: Any) -> str:
    if not config.compact_output:
        return (
            "Schema: {id, error_type, original, translation, suggested_original, suggested_translation, "
            "asr_suspect, needs_audio_review, reason, confidence}.\n"
            "Field rules: error_type must be one of translation_error, term_error, asr_suspect, needs_context, style_only; "
            "asr_suspect and needs_audio_review must be booleans; confidence must be 0.0 to 1.0.\n"
        )
    return (
        "Use compact JSON keys to save tokens: "
        "{i:id,t:error_type,o:original,tr:translation,so:suggested_original,s:suggested_translation,"
        "a:asr_suspect,n:needs_audio_review,r:reason,c:confidence}.\n"
        "Return booleans for a and n. Use t values: translation_error, term_error, asr_suspect, needs_context, style_only. "
        "Keep r short, max 18 Chinese characters or 12 English words.\n"
    )


def call_mimo(
    *,
    client: Any,
    config: Any,
    segment: dict[str, Any],
    audio_path: Path,
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
) -> tuple[str, Any]:
    audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    prompt = (
        "You are proofreading Chinese subtitles for one Japanese audio segment.\n"
        "Use the audio as the source of truth, then compare Japanese original text, "
        "Chinese translation, and glossary.\n"
        "Return ONLY a JSON array. Include only subtitle IDs that need correction.\n"
        "If no correction is needed, return []. Keep IDs and timestamps unchanged.\n"
        "Object schema: {id, original, translation, suggested_translation, reason, confidence}. Keep reason short.\n"
        "Do not add markdown or prose outside JSON.\n"
        f"Segment JSON: {json.dumps(segment, ensure_ascii=True)}\n"
        f"Glossary JSON: {json.dumps(glossary_entries, ensure_ascii=True)}\n"
        f"Subtitle entries JSON: {json.dumps(subtitle_entries, ensure_ascii=True)}"
    )
    response = chat_completion_create(
        client=client,
        config=config,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": maybe_disable_thinking_text(prompt, config)},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": audio_path.suffix.lstrip(".") or "wav",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or "", response.usage


def call_mimo_text_stage1(
    *,
    client: Any,
    config: Any,
    segment: dict[str, Any],
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
) -> tuple[str, Any]:
    prompt = (
        "You are stage 1 of a two-stage subtitle QA pipeline.\n"
        "You do NOT have audio.\n"
        "Inputs: Japanese ASR subtitle text, Chinese translation, subtitle timing and neighboring context, and glossary.\n"
        "Task 1: correct Chinese translation errors that are strongly supported by the provided text and context.\n"
        "Task 2: flag Japanese ASR text as suspicious only when there is concrete textual evidence: "
        "the Japanese is grammatically broken or semantically impossible; the Chinese translation cannot reasonably "
        "follow from the Japanese text; a proper noun, number, date, venue, player name, work title, or technical term "
        "looks likely misrecognized; or neighboring subtitles strongly imply a different phrase.\n"
        "Do not guess a corrected Japanese phrase unless context strongly supports it.\n"
        "If audio is required to decide, set needs_audio_review to true and leave suggested_translation empty unless "
        "the Chinese fix is already safe.\n"
        "Return ONLY a valid JSON array. Include an object if either translation correction is needed OR audio review is needed.\n"
        "Do not include markdown or prose.\n"
        f"{compact_schema_prompt(config)}"
        f"Segment JSON: {json.dumps(segment, ensure_ascii=True)}\n"
        f"Glossary JSON: {json.dumps(glossary_entries, ensure_ascii=True)}\n"
        f"Subtitle entries JSON: {json.dumps(subtitle_entries, ensure_ascii=True)}"
    )
    response = chat_completion_create(
        client=client,
        config=config,
        messages=[{"role": "user", "content": maybe_disable_thinking_text(prompt, config)}],
    )
    return response.choices[0].message.content or "", response.usage


def call_mimo_nearby_audio(
    *,
    client: Any,
    config: Any,
    segment: dict[str, Any],
    target_ids: list[str],
    target_entries: dict[str, Any],
    nearby_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    clip_path: Path,
    clip_meta: dict[str, float],
) -> tuple[str, Any]:
    audio_data = base64.b64encode(clip_path.read_bytes()).decode("ascii")
    prompt = (
        "You are stage 2 of a two-stage subtitle QA pipeline.\n"
        "You HAVE a short nearby audio clip. The target subtitle IDs were flagged as suspicious by text-only QA.\n"
        "Use the audio as the source of truth.\n"
        "Tasks, in order:\n"
        "1. Listen specifically for each target ID's Japanese wording. Multiple target IDs may be present in one clip.\n"
        "2. Decide independently for each target ID whether original_subtitle is correct, incomplete, or a mishearing.\n"
        "If it is incomplete or misheard and the audio is clear, put the complete corrected Japanese in suggested_original.\n"
        "3. Pay special attention to proper nouns: player names, place names, venue or stadium names, work titles, "
        "band or member names, dates, and numbers.\n"
        "4. Use nearby subtitle text only as context. Do not rewrite non-target IDs.\n"
        "5. If the target is singing, music, or a nonverbal sound, do not rewrite it merely to force dialogue.\n"
        "6. If the target phrase is unclear in this clip, set needs_audio_review to true and do not invent wording.\n"
        "When suggested_original changes the meaning, also provide a matching suggested_translation.\n"
        "Return ONLY a valid JSON array. Include only target IDs that need correction or remain unresolved. "
        "Use one JSON object per subtitle ID.\n"
        "Do not include markdown or prose.\n"
        f"{compact_schema_prompt(config)}"
        f"Target IDs JSON: {json.dumps(target_ids, ensure_ascii=True)}\n"
        f"Clip local start/end seconds JSON: {json.dumps(clip_meta, ensure_ascii=True)}\n"
        f"Segment JSON: {json.dumps(segment, ensure_ascii=True)}\n"
        f"Glossary JSON: {json.dumps(glossary_entries, ensure_ascii=True)}\n"
        f"Target entries JSON: {json.dumps(target_entries, ensure_ascii=True)}\n"
        f"Nearby entries JSON: {json.dumps(nearby_entries, ensure_ascii=True)}"
    )
    response = chat_completion_create(
        client=client,
        config=config,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": maybe_disable_thinking_text(prompt, config)},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": clip_path.suffix.lstrip(".") or "wav",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or "", response.usage


def call_mimo_with_retries(
    *,
    client: Any,
    config: Any,
    segment: dict[str, Any],
    audio_path: Path,
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any]:
    attempt = 0
    delay = max(0.0, base_delay)
    while True:
        attempt += 1
        try:
            return call_mimo(
                client=client,
                config=config,
                segment=segment,
                audio_path=audio_path,
                subtitle_entries=subtitle_entries,
                glossary_entries=glossary_entries,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt > max(1, max_retries) or not is_transient_error(exc):
                raise
            wait_seconds = min(max_delay, delay or 1.0)
            print(
                f"  transient MiMo error on attempt {attempt}/{max_retries}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(max_delay, max(wait_seconds * 2.0, 1.0))


def call_mimo_text_stage1_with_retries(
    *,
    client: Any,
    config: Any,
    segment: dict[str, Any],
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any]:
    attempt = 0
    delay = max(0.0, base_delay)
    while True:
        attempt += 1
        try:
            return call_mimo_text_stage1(
                client=client,
                config=config,
                segment=segment,
                subtitle_entries=subtitle_entries,
                glossary_entries=glossary_entries,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt > max(1, max_retries) or not is_transient_error(exc):
                raise
            wait_seconds = min(max_delay, delay or 1.0)
            print(
                f"  transient MiMo stage1 error on attempt {attempt}/{max_retries}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(max_delay, max(wait_seconds * 2.0, 1.0))


def call_mimo_nearby_audio_with_retries(
    *,
    client: Any,
    config: Any,
    segment: dict[str, Any],
    target_ids: list[str],
    target_entries: dict[str, Any],
    nearby_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    clip_path: Path,
    clip_meta: dict[str, float],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any]:
    attempt = 0
    delay = max(0.0, base_delay)
    while True:
        attempt += 1
        try:
            return call_mimo_nearby_audio(
                client=client,
                config=config,
                segment=segment,
                target_ids=target_ids,
                target_entries=target_entries,
                nearby_entries=nearby_entries,
                glossary_entries=glossary_entries,
                clip_path=clip_path,
                clip_meta=clip_meta,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt > max(1, max_retries) or not is_transient_error(exc):
                raise
            wait_seconds = min(max_delay, delay or 1.0)
            print(
                f"  transient MiMo stage2 error on attempt {attempt}/{max_retries}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(max_delay, max(wait_seconds * 2.0, 1.0))


def is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "503",
        "502",
        "504",
        "429",
        "rate limit",
        "timeout",
        "timed out",
        "upstream",
        "\u670d\u52a1\u5f02\u5e38",
        "\u7a0d\u540e\u91cd\u8bd5",
        "server_error",
    )
    return any(marker in text for marker in transient_markers)


def request_suggestions_with_parse_retries(
    request: Callable[[], tuple[str, Any]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any, list[dict[str, Any]]]:
    attempts = max(1, int(max_retries))
    delay = min(2.0, max(0.5, float(base_delay) if base_delay > 0 else 0.5))
    last_error: Exception | None = None
    last_content = ""
    for attempt in range(1, attempts + 1):
        content, usage = request()
        last_content = content
        try:
            return content, usage, parse_suggestions(content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt >= attempts:
                break
            wait_seconds = min(2.0, max_delay, delay)
            print(
                f"  invalid MiMo JSON on attempt {attempt}/{attempts}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(2.0, max_delay, max(wait_seconds * 2.0, 0.5))
    preview = last_content.strip().replace("\r", " ").replace("\n", " ")[:160]
    raise ValueError(f"MiMo returned invalid JSON after {attempts} attempts: {last_error}; preview={preview!r}")


def parse_suggestions(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    else:
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        object_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if object_matches:
            parsed = [json.loads(match) for match in object_matches]
        else:
            raise
    if not isinstance(parsed, list):
        if isinstance(parsed, dict):
            parsed = [parsed]
        else:
            raise ValueError("MiMo response is not a JSON array")
    return [item for item in parsed if isinstance(item, dict)]


def usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return {"repr": repr(usage)}
