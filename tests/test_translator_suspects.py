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
