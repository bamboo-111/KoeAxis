from __future__ import annotations

import threading
import time

import pytest

from optimizer import gemini_native
from optimizer.exceptions import LLMAuthError, LLMRateLimitError
from qwen_asr.gemini_pool import GeminiEndpoint, GeminiRelayPool

MESSAGES = [{"role": "user", "content": "hi"}]


def _reply(text: str):
    return gemini_native._to_chat_completion(
        {"candidates": [{"content": {"parts": [{"text": text}]}}], "usageMetadata": {}}, "m")


def _endpoints(*specs):
    return [GeminiEndpoint(name, f"https://{name}.example/v1beta", "k", f"model-{name}", cap) for name, cap in specs]


def test_build_payload_maps_input_audio_and_reasoning_effort():
    payload = gemini_native.build_payload(
        [{"role": "user", "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "mp3"}},
        ]}],
        temperature=0.0, require_json=False, disable_thinking=True,
        extra_body={"reasoning_effort": "medium"}, max_tokens=None,
    )
    assert payload["contents"] == [{"role": "user", "parts": [
        {"text": "a\nb"}, {"inlineData": {"mimeType": "audio/mpeg", "data": "QUJD"}}]}]
    assert payload["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert "reasoning_effort" not in payload


def test_pool_respects_per_relay_inflight_caps():
    active = {"a": 0, "b": 0}
    peak = {"a": 0, "b": 0}
    lock = threading.Lock()

    def sender(endpoint, payload, timeout):
        with lock:
            active[endpoint.name] += 1
            peak[endpoint.name] = max(peak[endpoint.name], active[endpoint.name])
        time.sleep(0.05)
        with lock:
            active[endpoint.name] -= 1
        return _reply(endpoint.name)

    pool = GeminiRelayPool(_endpoints(("a", 2), ("b", 1)), sender=sender, hedge_initial_s=10)
    threads = [threading.Thread(target=pool.create, kwargs={"messages": MESSAGES}) for _ in range(9)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert peak["a"] <= 2 and peak["b"] <= 1
    stats = pool.stats()
    assert stats["a"]["succeeded"] + stats["b"]["succeeded"] == 9
    assert stats["b"]["succeeded"] >= 1


def test_pool_fails_over_to_other_relay():
    def sender(endpoint, payload, timeout):
        if endpoint.name == "a":
            raise LLMRateLimitError(detail="HTTP 503 upstream")
        return _reply("from-b")

    pool = GeminiRelayPool(_endpoints(("a", 4), ("b", 1)), sender=sender)
    response = pool.create(messages=MESSAGES)
    assert response.choices[0].message.content == "from-b"
    assert pool.last_served()["endpoint"] == "b"


def test_pool_hedges_slow_request_and_first_success_wins():
    release = threading.Event()

    def sender(endpoint, payload, timeout):
        if endpoint.name == "a":
            release.wait(2)
            return _reply("slow-a")
        return _reply("fast-b")

    pool = GeminiRelayPool(_endpoints(("a", 1), ("b", 1)), sender=sender, hedge_initial_s=0.05)
    started = time.monotonic()
    response = pool.create(messages=MESSAGES)
    release.set()
    assert response.choices[0].message.content == "fast-b"
    assert time.monotonic() - started < 1.5
    served = pool.last_served()
    assert served["hedged"] and served["won_by_hedge"]
    assert pool.stats()["b"]["hedges_sent"] == 1


def test_pool_rests_failing_relay_and_drops_rejected_key():
    calls = {"a": 0, "b": 0}

    def sender(endpoint, payload, timeout):
        calls[endpoint.name] += 1
        if endpoint.name == "a":
            raise LLMAuthError(detail="HTTP 401")
        raise LLMRateLimitError(detail="HTTP 429")

    pool = GeminiRelayPool(_endpoints(("a", 1), ("b", 1)), sender=sender, failure_threshold=2, cooldown_s=0.3)
    with pytest.raises(RuntimeError, match="429"):
        pool.create(messages=MESSAGES)
    assert calls == {"a": 1, "b": 1}  # failed over once, no blind same-relay retry
    with pytest.raises(RuntimeError):
        pool.create(messages=MESSAGES)
    assert pool.stats()["a"]["disabled"] is True
    assert calls == {"a": 1, "b": 2}  # b now rests after two failures in a row
    started = time.monotonic()
    with pytest.raises(RuntimeError):
        pool.create(messages=MESSAGES)
    assert time.monotonic() - started >= 0.25  # waited for the cool-down instead of hammering b
    assert calls == {"a": 1, "b": 3}


def test_failover_waits_for_a_busy_relay_instead_of_giving_up():
    release = threading.Event()

    def sender(endpoint, payload, timeout):
        if endpoint.name == "a":
            raise LLMRateLimitError(detail="HTTP 503 upstream")
        if payload["contents"][0]["parts"][0]["text"] == "hold":
            release.wait(2)
        return _reply("from-b")

    pool = GeminiRelayPool(_endpoints(("a", 1), ("b", 1)), sender=sender, hedge_initial_s=10)
    holder = threading.Thread(target=pool.create, kwargs={"messages": [{"role": "user", "content": "hold"}]})
    holder.start()
    time.sleep(0.05)  # b's only slot is now taken
    threading.Timer(0.2, release.set).start()
    response = pool.create(messages=MESSAGES)  # a fails, then waits for b's slot
    holder.join()
    assert response.choices[0].message.content == "from-b"


def test_exhausted_relays_raise_a_transient_error_for_callers():
    from qwen_asr.mimo_requests import is_transient_error

    def sender(endpoint, payload, timeout):
        raise LLMRateLimitError(detail="connection reset")

    pool = GeminiRelayPool(_endpoints(("a", 1)), sender=sender)
    with pytest.raises(RuntimeError) as info:
        pool.create(messages=MESSAGES)
    assert is_transient_error(info.value)
