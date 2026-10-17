from __future__ import annotations

import pytest

from optimizer.asr_data import ASRData, ASRDataSeg
from qwen_asr.normalize import NormalizeParams, normalize_asr_data
from qwen_asr.subtitle_layout import content_key, reflow_subtitle_segments, strip_display_punctuation


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("「你好，世界！」……（再见。）", "你好 世界 再见"),
        ("スバルと撮った？バイバイ。", "スバルと撮った バイバイ"),
        ("イエー！", "イエー"),
        ("价格是2.5，邮箱a.b@example.com。", "价格是2.5 邮箱a.b@example.com"),
        ("Don't re-write this!", "Don't re-write this"),
        ("See https://example.com/a?x=1.", "See https://example.com/a?x=1"),
        ("土下座 at interfm.jp まで。", "土下座 at interfm.jp まで"),
    ],
)
def test_display_punctuation_preserves_lexical_spelling(source: str, expected: str) -> None:
    assert strip_display_punctuation(source) == expected
    assert content_key(source) == content_key(expected)


def test_layout_merges_japanese_compound_and_its_continuation() -> None:
    source = [
        ASRDataSeg("こんな感じで立ち", 33000, 35000, "像这样站着"),
        ASRDataSeg("位置も出てて、実際", 36800, 38100, "位置也出来了，实际上"),
        ASRDataSeg("はこんな感じです。", 39500, 40600, "是这样的。"),
    ]
    result = reflow_subtitle_segments(source)
    assert len(result) == 1
    assert result[0].text == "こんな感じで立ち位置も出てて、実際はこんな感じです。"
    assert result[0].translated_text == "像这样站着位置也出来了，实际上是这样的。"
    assert (result[0].start_time, result[0].end_time) == (33000, 40600)
    assert source[0].text == "こんな感じで立ち"


def test_layout_merges_dangling_demonstratives_and_connector() -> None:
    source = [
        ASRDataSeg("ちなみに", 16000, 16800, "顺便说一下"),
        ASRDataSeg("この", 17400, 18200, "这个"),
        ASRDataSeg("QRコードを読み取る。", 18900, 20700, "扫描二维码。"),
        ASRDataSeg("ぜひ", 26000, 26800, "一定要"),
        ASRDataSeg("楽しんでね。", 26800, 28000, "玩得开心哦。"),
    ]
    result = reflow_subtitle_segments(source)
    assert [item.text for item in result] == ["ちなみにこのQRコードを読み取る。", "ぜひ楽しんでね。"]


def test_layout_splits_bilingual_sentences_at_aligned_word_ends() -> None:
    source = [ASRDataSeg("行け。カメラマン朱李さん。痛い。", 1000, 5000, "走吧。摄影师朱李。好痛。")]
    tokens = [
        ASRDataSeg("行け。", 1000, 1600),
        ASRDataSeg("カメラマン", 1800, 2800),
        ASRDataSeg("朱李さん。", 2800, 3800),
        ASRDataSeg("痛い。", 4000, 5000),
    ]
    events = []
    result = reflow_subtitle_segments(source, timing_tokens=tokens, events=events)
    assert [item.text for item in result] == ["行け。", "カメラマン朱李さん。", "痛い。"]
    assert [item.translated_text for item in result] == ["走吧。", "摄影师朱李。", "好痛。"]
    assert [item.end_time for item in result] == [1600, 3800, 5000]
    assert events[0]["timing_methods"] == ["aligned-token-end", "aligned-token-end"]


def test_layout_does_not_guess_mismatched_translation_sentence_boundaries() -> None:
    source = [ASRDataSeg("第一句。第二句。", 0, 3000, "两句译在一起")]
    result = reflow_subtitle_segments(source)
    assert len(result) == 1
    assert result[0].text == source[0].text
    assert result[0].translated_text == source[0].translated_text


def test_layout_does_not_split_inside_spoken_email_domain() -> None:
    source = [ASRDataSeg("状況などを土下座 at interfm.jp まで", 0, 4000, "情况等，请发送到 dogeza@interfm.jp")]
    result = reflow_subtitle_segments(source)
    assert [item.text for item in result] == [source[0].text]
    assert [item.translated_text for item in result] == [source[0].translated_text]


def test_layout_rejoins_verb_suffix_before_splitting_bilingual_sentences() -> None:
    source = [ASRDataSeg("バイバイ。してます。でも誰もいない。", 0, 6000, "在说拜拜。但是一个人都没有。")]
    result = reflow_subtitle_segments(source)
    assert [item.text for item in result] == ["バイバイ。してます。", "でも誰もいない。"]
    assert [item.translated_text for item in result] == ["在说拜拜。", "但是一个人都没有。"]


@pytest.mark.parametrize("gap", [3500, 10000])
def test_layout_does_not_bridge_long_silence(gap: int) -> None:
    source = [ASRDataSeg("この", 0, 1000, "这个"), ASRDataSeg("写真です。", 1000 + gap, 2500 + gap, "照片。")]
    assert len(reflow_subtitle_segments(source)) == 2


def test_layout_protects_complete_sentences_and_short_replies() -> None:
    source = [
        ASRDataSeg("終わりました。", 0, 2000, "结束了。"),
        ASRDataSeg("ちなみに", 2000, 2800, "顺便说一下"),
        ASRDataSeg("うん。", 2800, 2920, "嗯。"),
        ASRDataSeg("この", 2920, 3500, "这个"),
    ]
    assert len(reflow_subtitle_segments(source)) == 4


def test_normalize_cleans_both_lines_and_preserves_bilingual_content() -> None:
    source = ASRData([
        ASRDataSeg("ぜひ", 1000, 1500, "一定要"),
        ASRDataSeg("楽しんでね！", 1500, 3000, "玩得开心哦！"),
        ASRDataSeg("はい。", 3200, 3320, "好。"),
        ASRDataSeg("次です。", 3320, 4500, "下一位。"),
    ])
    result = normalize_asr_data(source)
    assert result.segments[0].text == "ぜひ楽しんでね"
    assert result.segments[0].translated_text == "一定要玩得开心哦"
    assert result.segments[1].text == "はい"
    assert result.segments[1].end_time - result.segments[1].start_time >= 500
    for field in ("text", "translated_text"):
        assert content_key("".join(getattr(item, field) for item in source.segments)) == content_key("".join(getattr(item, field) for item in result.segments))
    assert all(left.end_time <= right.start_time for left, right in zip(result.segments, result.segments[1:], strict=False))
    assert source.segments[1].text == "楽しんでね！"


def test_normalize_omits_punctuation_only_cue_without_dropping_words() -> None:
    source = ASRData([ASRDataSeg("……", 0, 1000), ASRDataSeg("次です。", 1500, 2500)])
    assert [item.text for item in normalize_asr_data(source).segments] == ["次です"]


def test_normalize_preserves_min_duration_configuration() -> None:
    source = ASRData([ASRDataSeg("はい。", 0, 120), ASRDataSeg("次です。", 120, 2000)])
    result = normalize_asr_data(source, NormalizeParams(min_duration_ms=300))
    assert result.segments[0].end_time - result.segments[0].start_time == 300


def test_normalize_can_split_short_sentences_using_display_extension() -> None:
    source = ASRData([ASRDataSeg("いけてるこれ。いけてるかこれ。", 1000, 1800, "这个还行吧。这个还行吗。")])
    result = normalize_asr_data(source)
    assert [segment.text for segment in result.segments] == ["いけてるこれ", "いけてるかこれ"]
    assert all(segment.end_time - segment.start_time >= 500 for segment in result.segments)
    assert result.segments[0].end_time == result.segments[1].start_time


def test_layout_does_not_create_dangling_demonstrative() -> None:
    source = ASRData([ASRDataSeg("これ。できた。", 1000, 1530, "这个。拍好了。")])
    result = normalize_asr_data(source)
    assert len(result.segments) == 1
    assert result.segments[0].text == "これ できた"
    assert result.segments[0].translated_text == "这个 拍好了"
