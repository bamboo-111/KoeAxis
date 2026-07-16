from __future__ import annotations

import types

from qwen_asr.mimo_requests import (
    chat_completion_create,
    is_transient_error,
    parse_suggestions,
    request_suggestions_with_parse_retries,
    usage_to_dict,
)


def test_request_suggestions_retries_invalid_json(monkeypatch) -> None:
    responses = iter([("", {"attempt": 1}), ("[]", {"attempt": 2})])
    monkeypatch.setattr("qwen_asr.mimo_requests.time.sleep", lambda _seconds: None)

    content, usage, suggestions = request_suggestions_with_parse_retries(
        lambda: next(responses),
        max_retries=2,
        base_delay=0.0,
        max_delay=0.0,
    )

    assert content == "[]"
    assert usage == {"attempt": 2}
    assert suggestions == []


def test_parse_suggestions_accepts_unclosed_fence_and_single_object() -> None:
    assert parse_suggestions('```json\n[{"id": "125", "suggested_translation": "ok"}]') == [
        {"id": "125", "suggested_translation": "ok"}
    ]
    assert parse_suggestions('prefix {"id": "1"} suffix') == [{"id": "1"}]


def test_chat_completion_retries_without_extra_body() -> None:
    seen_kwargs: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            seen_kwargs.append(dict(kwargs))
            if "extra_body" in kwargs:
                raise RuntimeError("extra_body unsupported")
            return "ok"

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=FakeCompletions()))
    config = types.SimpleNamespace(
        model="mimo",
        temperature=0.0,
        max_tokens=128,
        extra_body={"enable_thinking": False},
    )

    assert chat_completion_create(client=client, config=config, messages=[]) == "ok"
    assert "extra_body" in seen_kwargs[0]
    assert "extra_body" not in seen_kwargs[1]


def test_transient_error_and_usage_conversion() -> None:
    class Usage:
        def model_dump(self):
            return {"total_tokens": 3}

    assert is_transient_error(RuntimeError("server_error 503"))
    assert is_transient_error(RuntimeError("\u670d\u52a1\u5f02\u5e38"))
    assert not is_transient_error(RuntimeError("invalid prompt"))
    assert usage_to_dict(Usage()) == {"total_tokens": 3}
    assert usage_to_dict(None) == {}
