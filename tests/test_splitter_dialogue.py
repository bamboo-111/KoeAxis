from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from optimizer.splitter import SubtitleSplitter


def _splitter() -> SubtitleSplitter:
    return SubtitleSplitter(thread_num=1, model="", base_url="", api_key="")


def test_dialogue_short_response_is_not_merged_as_filler() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("それじゃ早速練習始めるわよ", 0, 1200),
            ASRDataSeg("はい", 1300, 1500),
            ASRDataSeg("ママちゃんどうしたの", 1600, 2600),
        ]

        result = splitter._smooth_short_fillers(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "それじゃ早速練習始めるわよ",
            "はい",
            "ママちゃんどうしたの",
        ]
    finally:
        splitter.stop()


def test_dialogue_short_response_is_not_merged_for_readability() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("聞こえてますか", 0, 900),
            ASRDataSeg("うん", 950, 1100),
            ASRDataSeg("大丈夫", 1160, 1700),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "聞こえてますか",
            "うん",
            "大丈夫",
        ]
    finally:
        splitter.stop()


def test_non_dialogue_filler_can_still_merge() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("今日は", 0, 500),
            ASRDataSeg("あの", 520, 650),
        ]

        result = splitter._smooth_short_fillers(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == ["今日はあの"]
    finally:
        splitter.stop()


def test_complete_short_utterance_is_not_merged_for_readability() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("ちょっと待って", 0, 700),
            ASRDataSeg("大丈夫", 760, 1100),
            ASRDataSeg("先に行くね", 1160, 1900),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == [
            "ちょっと待って",
            "大丈夫",
            "先に行くね",
        ]
    finally:
        splitter.stop()


def test_short_sentence_starting_with_filler_is_not_merged_left() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("聞こえてる", 0, 800),
            ASRDataSeg("うん大丈夫", 850, 1300),
        ]

        result = splitter._smooth_short_fillers(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == ["聞こえてる", "うん大丈夫"]
    finally:
        splitter.stop()


def test_structural_fragment_can_still_merge_for_readability() -> None:
    splitter = _splitter()
    try:
        segments = [
            ASRDataSeg("行きたいんだ", 0, 800),
            ASRDataSeg("けど", 840, 980),
        ]

        result = splitter._smooth_readability_segments(segments)  # noqa: SLF001

        assert [segment.text for segment in result] == ["行きたいんだけど"]
    finally:
        splitter.stop()
