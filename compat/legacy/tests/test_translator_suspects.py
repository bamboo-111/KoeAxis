from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from optimizer.translator import SubtitleTranslator, _augment_translation_suspects


def test_translation_validator_accepts_structured_suspect_payload() -> None:
    valid, error = SubtitleTranslator._validate_translation_result(  # noqa: SLF001
        {"1": "source"},
        {
            "1": {
                "translation": "target",
                "asr_suspect": True,
                "needs_audio_review": True,
                "suspect_types": ["fragment"],
                "reason": "broken source",
                "confidence": 0.4,
            }
        },
    )

    assert valid
    assert error == ""


def test_translate_chunk_stores_suspect_metadata(monkeypatch) -> None:
    translator = SubtitleTranslator(
        thread_num=1,
        batch_num=1,
        model="model",
        base_url="http://example.invalid/v1",
        api_key="key",
        target_language="Chinese",
    )
    monkeypatch.setattr(
        translator,
        "_agent_loop",
        lambda _prompt, _subtitle: {
            "1": {
                "translation": "target",
                "asr_suspect": True,
                "needs_audio_review": True,
                "suspect_types": ["name"],
                "reason": "uncertain name",
                "confidence": 0.35,
            }
        },
    )
    segment = ASRDataSeg("\u305d\u308c\u306f\u666e\u901a\u306e\u6587\u3067\u3059", 0, 1000)

    try:
        translator._translate_chunk([segment])  # noqa: SLF001
    finally:
        translator.stop()

    assert segment.translated_text == "target"
    assert segment.asr_suspect is True
    assert segment.needs_audio_review is True
    assert segment.suspect_types == ["name"]
    assert segment.suspect_reason == "uncertain name"
    assert segment.suspect_confidence == 0.35


def test_translate_chunk_adds_short_response_suspect_for_legacy_payload(monkeypatch) -> None:
    translator = SubtitleTranslator(
        thread_num=1,
        batch_num=1,
        model="model",
        base_url="http://example.invalid/v1",
        api_key="key",
        target_language="Chinese",
    )
    monkeypatch.setattr(translator, "_agent_loop", lambda _prompt, _subtitle: {"1": "\u662f"})
    segment = ASRDataSeg("\u306f\u3044", 0, 300)

    try:
        translator._translate_chunk([segment])  # noqa: SLF001
    finally:
        translator.stop()

    assert segment.translated_text == "\u662f"
    assert segment.asr_suspect is True
    assert segment.needs_audio_review is True
    assert "short_response" in segment.suspect_types
    assert segment.suspect_confidence < 1.0


def test_augment_translation_suspects_merges_rule_hits_with_llm_metadata() -> None:
    segment = ASRDataSeg("\u3053\u308c\u306f3\u500b\u3058\u3083\u306a\u3044\u306e\uff1f", 0, 900)

    meta = _augment_translation_suspects(
        segment,
        "\u8fd9\u662f4\u4e2a",
        {
            "asr_suspect": False,
            "needs_audio_review": False,
            "suspect_types": ["semantic"],
            "suspect_reason": "llm unsure",
            "suspect_confidence": 0.9,
        },
    )

    assert meta["asr_suspect"] is True
    assert meta["needs_audio_review"] is True
    assert meta["suspect_types"] == ["semantic", "question", "negation", "quantity"]
    assert "llm unsure" in meta["suspect_reason"]
    assert meta["suspect_confidence"] < 0.9


def test_augment_translation_suspects_marks_untranslated_and_name() -> None:
    segment = ASRDataSeg("\u30df\u30ab\u3055\u3093", 0, 800)

    meta = _augment_translation_suspects(segment, "\u30df\u30ab\u3055\u3093", {})

    assert meta["asr_suspect"] is True
    assert meta["needs_audio_review"] is True
    assert "untranslated" in meta["suspect_types"]
    assert "name" in meta["suspect_types"]


def test_translation_payload_strips_instruction_echo_prefix() -> None:
    from optimizer.translator import _normalize_translation_payload

    assert _normalize_translation_payload("翻译为：好的，那么请做歌曲介绍。")[0] == "好的，那么请做歌曲介绍。"
    assert _normalize_translation_payload({"translation": "译文: 大家晚上好"})[0] == "大家晚上好"
    assert _normalize_translation_payload("翻译这首歌很难")[0] == "翻译这首歌很难"
    assert _normalize_translation_payload("翻译：")[0] == "翻译："


def _fake_response(content: str):
    from types import SimpleNamespace

    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _translator() -> SubtitleTranslator:
    return SubtitleTranslator(
        thread_num=1, batch_num=2, model="model", base_url="http://example.invalid/v1", api_key="key",
        target_language="简体中文",
    )


def test_agent_loop_keeps_complete_batch_when_only_residual_kana_remains(monkeypatch) -> None:
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _fake_response('{"1": "大家晚上好", "2": "小エログ"}')

    monkeypatch.setattr("optimizer.translator.call_llm", fake_call_llm)
    translator = _translator()
    try:
        result = translator._agent_loop("system", {"1": "皆さんこんばんは", "2": "小エログ"})  # noqa: SLF001
    finally:
        translator.stop()

    assert len(calls) == 5
    assert result == {"1": "大家晚上好", "2": "小エログ"}


def test_agent_loop_still_rejects_structurally_incomplete_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        "optimizer.translator.call_llm", lambda **_kwargs: _fake_response('{"1": "大家晚上好"}')
    )
    translator = _translator()
    try:
        result = translator._agent_loop("system", {"1": "皆さんこんばんは", "2": "はい"})  # noqa: SLF001
    finally:
        translator.stop()

    assert result is None


def test_compact_clean_payload_and_prompt_contract() -> None:
    from pathlib import Path

    from optimizer.translator import _normalize_translation_payload

    translated, meta = _normalize_translation_payload({"translation": "你好", "confidence": 0.95})
    assert translated == "你好"
    assert meta["asr_suspect"] is False and meta["needs_audio_review"] is False
    assert meta["suspect_types"] == [] and meta["suspect_reason"] == ""
    assert meta["suspect_confidence"] == 0.95
    valid, error = SubtitleTranslator._validate_translation_result(  # noqa: SLF001
        {"1": "こんにちは", "2": "えっ"},
        {"1": {"translation": "你好", "confidence": 0.95}, "2": {"translation": "诶", "asr_suspect": True}},
    )
    assert valid and error == ""
    # Clean entries skip the four empty suspect fields: translation output tokens were ~47% lower on episode 43.
    prompt = Path("optimizer/prompts/translate/standard.md").read_text(encoding="utf-8")
    assert 'output only {"translation": ..., "confidence": ...}' in prompt


def test_parallel_translate_prewarms_cache_before_fan_out(monkeypatch) -> None:
    import threading

    monkeypatch.setattr("optimizer.translator.PREWARM_DELAY", 0.0)
    monkeypatch.setattr("optimizer.translator.SUBMIT_INTERVAL", 0.0)
    translator = SubtitleTranslator(
        thread_num=4, batch_num=1, model="model", base_url="http://example.invalid/v1", api_key="key",
        target_language="简体中文",
    )
    events: list[str] = []
    lock = threading.Lock()
    prewarm_calls: list[dict] = []

    def fake_call_llm(**kwargs):
        prewarm_calls.append(kwargs)
        with lock:
            events.append("prewarm")
        raise RuntimeError("empty content")

    def fake_chunk(chunk):
        with lock:
            events.append(chunk[0].text)

    monkeypatch.setattr("optimizer.translator.call_llm", fake_call_llm)
    monkeypatch.setattr(translator, "_translate_chunk", fake_chunk)
    chunks = [[ASRDataSeg(text=f"t{i}", start_time=i, end_time=i + 1)] for i in range(3)]
    assert translator._parallel_translate(chunks, 3) == 3  # noqa: SLF001
    assert events[0] == "prewarm"
    assert sorted(events[1:]) == ["t0", "t1", "t2"]
    assert prewarm_calls[0]["messages"][0]["content"] == translator._system_prompt()  # noqa: SLF001
    assert prewarm_calls[0]["max_tokens"] == 16
