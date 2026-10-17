from __future__ import annotations

from qwen_asr.mimo_application import evaluate_stage2_suggestion


def test_stage2_application_accepts_supported_original_and_translation_update() -> None:
    current = {
        "original_subtitle": "\u3042",
        "translated_subtitle": "\u554a",
        "suspect_reason": "ASS stage diff suspect: ass_text=\u306f\u3044",
    }
    normalized = {
        "id": "1",
        "suggested_original": "\u306f\u3044",
        "suggested_translation": "\u597d\u7684",
        "asr_suspect": True,
        "needs_audio_review": False,
        "confidence": 0.95,
        "reason": "audio clear",
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="1",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.rejection is None
    assert result.updates["original_subtitle"] == "\u306f\u3044"
    assert result.updates["translated_subtitle"] == "\u597d\u7684"
    assert result.evidence["ass_guard"]["accepted"] is True
    assert result.updates["__proofread_evidence"] == result.evidence


def test_stage2_application_rejects_unbacked_original_content_deletion() -> None:
    current = {
        "original_subtitle": "\u305d\u3093\u306a\u30ba\u30e0\u30eb\u30c9",
        "translated_subtitle": "\u90a3\u6837\u7684\u7956\u7a46\u5c14\u5fb7",
        "suspect_reason": "manifest suspect without ass_text",
    }
    normalized = {
        "id": "1",
        "suggested_original": "\u30ba\u30e0\u30eb\u30c9\u3055\u3093\uff01",
        "suggested_translation": "\u7956\u7a46\u5c14\u5fb7\u5148\u751f",
        "asr_suspect": True,
        "needs_audio_review": False,
        "confidence": 0.95,
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="1",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.updates == {}
    assert result.rejection is not None
    assert result.rejection["reason"] == "original-content-dropped-without-ass-reference"


def test_stage2_application_requires_audio_review_to_be_cleared_before_apply() -> None:
    current = {
        "original_subtitle": "\u3042",
        "translated_subtitle": "\u554a",
        "suspect_reason": "ASS stage diff suspect: ass_text=\u306f\u3044",
    }
    normalized = {
        "id": "1",
        "suggested_original": "\u306f\u3044",
        "suggested_translation": "\u597d\u7684",
        "asr_suspect": True,
        "needs_audio_review": True,
        "confidence": 0.95,
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="1",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.rejection is None
    assert result.updates == {}


def test_stage2_application_rejects_translation_that_reintroduces_kana() -> None:
    current = {
        "original_subtitle": "インタFM.jpまで送ってください。",
        "translated_subtitle": "FM.jp。",
    }
    normalized = {
        "id": "750",
        "suggested_translation": "请发送到インタFM.jp。",
        "asr_suspect": False,
        "needs_audio_review": False,
        "confidence": 0.95,
    }

    result = evaluate_stage2_suggestion(
        subtitle_id="750",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
    )

    assert result.updates == {}
    assert result.rejection is not None
    assert result.rejection["reason"] == "translation-introduces-japanese-kana"


def test_stage2_application_rejects_original_completed_with_next_cue_head() -> None:
    current = {"original_subtitle": "3月25日水曜日にリリースするライブブルーレイ&", "translated_subtitle": "即3月25日星期三发售的现场蓝光＆"}
    normalized = {
        "id": "434",
        "suggested_original": "3月25日水曜日にリリースするライブブルーレイ＆DVD",
        "suggested_translation": "即3月25日星期三发售的现场蓝光＆DVD",
        "asr_suspect": True,
        "needs_audio_review": False,
        "confidence": 0.95,
    }
    result = evaluate_stage2_suggestion(
        subtitle_id="434",
        normalized=normalized,
        current_item=current,
        apply_confidence_threshold=0.9,
        next_item={"original_subtitle": "DVDトゲナシトゲアリFifth One Man Live"},
    )
    assert result.rejection is not None
    assert result.rejection["reason"] == "original-appends-next-subtitle-head"
    assert result.rejection["duplicated_text"] == "dvd"


def test_stage2_application_rejects_original_prefixed_with_previous_cue_tail() -> None:
    result = evaluate_stage2_suggestion(
        subtitle_id="2",
        normalized={
            "id": "2",
            "suggested_original": "ありがとうございます。それでは次のコーナー",
            "suggested_translation": "谢谢。那么下一个环节",
            "asr_suspect": True,
            "confidence": 0.95,
        },
        current_item={"original_subtitle": "それでは次のコーナー", "translated_subtitle": "那么下一个环节"},
        apply_confidence_threshold=0.9,
        previous_item={"original_subtitle": "本当にありがとうございます。"},
    )
    assert result.rejection is not None
    assert result.rejection["reason"] == "original-prepends-previous-subtitle-tail"


def test_stage2_application_keeps_fix_when_neighbors_do_not_repeat_it() -> None:
    result = evaluate_stage2_suggestion(
        subtitle_id="2",
        normalized={
            "id": "2",
            "suggested_original": "コエログのコーナーです",
            "suggested_translation": "这是声音日志环节",
            "asr_suspect": True,
            "confidence": 0.95,
        },
        current_item={"original_subtitle": "小エログのコーナーです", "translated_subtitle": "这是小色日志环节",
                      "suspect_reason": "ASS stage diff suspect: ass_text=コエログのコーナーです"},
        apply_confidence_threshold=0.9,
        previous_item={"original_subtitle": "続いては"},
        next_item={"original_subtitle": "メッセージを紹介します"},
    )
    assert result.rejection is None
    assert result.updates["original_subtitle"] == "コエログのコーナーです"


def test_neighbor_guard_ignores_prefix_of_longer_katakana_word() -> None:
    from qwen_asr.mimo_guards import original_neighbor_duplication_guard

    result = original_neighbor_duplication_guard(
        current_original="ライブブルーレイand DVD トゲナシトゲ",
        suggested_original="ライブブルーレイand DVD トゲナシトゲアリ",
        next_original="アリーナワンマンライブメイドの時に付入されています。",
    )
    assert result["accepted"] is True
