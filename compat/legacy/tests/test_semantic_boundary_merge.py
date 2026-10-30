from __future__ import annotations

from optimizer.asr_data import ASRDataSeg
from qwen_asr.semantic_boundary_merge import _merge_batch


def test_semantic_merge_rejects_complete_sentence_boundary() -> None:
    source = [ASRDataSeg("行け。", 0, 1000), ASRDataSeg("カメラマン朱李さん。", 1000, 3000)]
    result = _merge_batch(source, [[1, 2]], max_chars=36)
    assert result == source


def test_semantic_merge_preserves_standalone_response() -> None:
    source = [ASRDataSeg("ちなみに", 0, 1000), ASRDataSeg("うん、", 1000, 1500)]
    assert _merge_batch(source, [[1, 2]], max_chars=36) == source


def test_semantic_merge_allows_compound_and_keeps_translation() -> None:
    source = [ASRDataSeg("立ち", 0, 1000, "站立"), ASRDataSeg("位置", 1000, 2000, "位置")]
    result = _merge_batch(source, [[1, 2]], max_chars=36)
    assert len(result) == 1
    assert result[0].text == "立ち位置"
    assert result[0].translated_text == "站立位置"


def test_semantic_singleton_keeps_its_translation() -> None:
    source = [ASRDataSeg("終わり。", 0, 1000, "结束。")]
    result = _merge_batch(source, [[1]], max_chars=36)
    assert result[0].translated_text == "结束。"


def test_semantic_merge_keeps_valid_runs_of_an_overlong_group() -> None:
    source = [
        ASRDataSeg("私は結構やりたい", 0, 1000),
        ASRDataSeg("こととか食べたいものを行きたい", 1000, 2500),
        ASRDataSeg("場所をインスタとかで", 2500, 3500),
        ASRDataSeg("めっちゃ保存してる。", 3500, 4500),
    ]
    result = _merge_batch(source, [[1, 2, 3, 4]], max_chars=28)
    assert [segment.text for segment in result] == [
        "私は結構やりたいこととか食べたいものを行きたい",
        "場所をインスタとかでめっちゃ保存してる。",
    ]
    assert "".join(segment.text for segment in result) == "".join(segment.text for segment in source)


def test_semantic_merge_group_split_still_respects_sentence_and_response_guards() -> None:
    source = [
        ASRDataSeg("行け。", 0, 1000),
        ASRDataSeg("カメラマン", 1000, 1500),
        ASRDataSeg("朱李さん", 1500, 2000),
        ASRDataSeg("うん、", 2000, 2300),
    ]
    result = _merge_batch(source, [[1, 2, 3, 4]], max_chars=36)
    assert [segment.text for segment in result] == ["行け。", "カメラマン朱李さん", "うん、"]


def test_split_llm_model_defaults_to_flash_for_mimo() -> None:
    import argparse

    from qwen_asr.commands.split import resolve_split_llm_model

    assert resolve_split_llm_model(argparse.Namespace(llm_model="mimo-v2.6-pro")) == "mimo-v2.6-flash"
    assert resolve_split_llm_model(argparse.Namespace(llm_model="mimo-v2.6-pro", split_llm_model="mimo-v2.6-pro")) == "mimo-v2.6-pro"
    assert resolve_split_llm_model(argparse.Namespace(llm_model="gpt-4o-mini")) == "gpt-4o-mini"
    assert resolve_split_llm_model(argparse.Namespace(llm_model=None)) is None


def _merge_kwargs() -> dict:
    return {
        "model": "m",
        "base_url": "http://x",
        "api_key": "k",
        "workers": 2,
        "max_chars": 36,
        "disable_thinking": True,
        "llm_extra_body": None,
        "timeout": 1.0,
    }


def test_semantic_merge_retries_a_failed_batch(monkeypatch) -> None:
    import qwen_asr.semantic_boundary_merge as merge
    from optimizer.exceptions import LLMConnectionError

    calls = []

    def flaky(batch, **_):
        calls.append(len(batch))
        if len(calls) == 1:
            raise LLMConnectionError(detail="Connection error.")
        return [[1, 2]]

    monkeypatch.setattr(merge, "_request_groups", flaky)
    monkeypatch.setattr(merge.time, "sleep", lambda _: None)
    source = [ASRDataSeg("今日は", 0, 500), ASRDataSeg("いい天気", 500, 1000)]

    result = merge.merge_adjacent_boundaries(source, **_merge_kwargs())

    assert calls == [2, 2]
    assert [seg.text for seg in result] == ["今日はいい天気"]


def test_semantic_merge_keeps_batch_unmerged_when_retries_run_out(monkeypatch) -> None:
    import qwen_asr.semantic_boundary_merge as merge

    def broken(batch, **_):
        raise RuntimeError("Semantic boundary merge coverage guard failed")

    monkeypatch.setattr(merge, "_request_groups", broken)
    monkeypatch.setattr(merge.time, "sleep", lambda _: None)
    source = [ASRDataSeg("今日は", 0, 500), ASRDataSeg("いい天気", 500, 1000)]

    result = merge.merge_adjacent_boundaries(source, **_merge_kwargs())

    assert [seg.text for seg in result] == ["今日は", "いい天気"]


def test_semantic_merge_does_not_hide_auth_errors(monkeypatch) -> None:
    import pytest

    import qwen_asr.semantic_boundary_merge as merge
    from optimizer.exceptions import LLMAuthError

    def unauthorized(batch, **_):
        raise LLMAuthError()

    monkeypatch.setattr(merge, "_request_groups", unauthorized)
    with pytest.raises(LLMAuthError):
        merge.merge_adjacent_boundaries([ASRDataSeg("a", 0, 1), ASRDataSeg("b", 1, 2)], **_merge_kwargs())
