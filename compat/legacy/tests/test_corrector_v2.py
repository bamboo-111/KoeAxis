from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr import corrector as corrector_module
from qwen_asr.corrector import ASRCorrector, apply_correction_edits
from qwen_asr.models import TranscriptSegment, WorkPaths


def _seg(seg_id: str, text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(segment_id=seg_id, audio_path="a.wav", global_start_time=start,
                             global_end_time=start + 10.0, text=text)


def test_apply_edits_minimal_spans_threshold_and_guards() -> None:
    chunk = [_seg("s1", "シリーさんがシリーさんに言った"), _seg("s2", "公開してます")]
    edits = [
        {"id": "s1", "from": "シリーさん", "to": "朱李さん", "confidence": 0.95, "reason": "member"},
        {"id": "s2", "from": "公開", "to": "後悔", "confidence": 0.5, "reason": "homophone"},
        {"id": "s2", "from": "存在しない", "to": "x", "confidence": 0.99},
        {"id": "s3", "from": "シリー", "to": "朱李", "confidence": 0.99},
        {"id": "s1", "from": "言った", "to": "言った", "confidence": 0.99},
        {"id": "s1", "from": "あ" * 30, "to": "い", "confidence": 0.99},
        "garbage",
    ]
    report = apply_correction_edits(chunk, edits, threshold=0.8)
    assert report["s1"].corrected_text == "朱李さんが朱李さんに言った"
    assert report["s1"].changed and report["s1"].reason.startswith("シリーさん→朱李さん (0.95, member)")
    assert report["s2"].corrected_text == "公開してます" and not report["s2"].changed
    assert "skipped: 公開→後悔 (0.50" in report["s2"].reason


def test_v2_sends_read_only_context_and_applies_edits(monkeypatch: pytest.MonkeyPatch) -> None:
    segments = [_seg(f"s{i}", f"テキスト{i}", start=i * 10.0) for i in range(5)]
    segments[2] = _seg("s2", "寿司さんが来た", start=20.0)
    seen: list[str] = []

    def fake_call(self, messages, validate, *_hints):
        user = messages[-1]["content"]
        seen.append(user)
        assert validate({"edits": []})[0] and not validate({"x": 1})[0]
        if "寿司さん" in user and "<asr_segments>" in user and "寿司さん" in user.split("<asr_segments>")[1]:
            return {"edits": [{"id": "s2", "from": "寿司さん", "to": "朱李さん", "confidence": 0.9}]}
        return {"edits": []}

    monkeypatch.setattr(ASRCorrector, "_call_with_retries", fake_call)
    corrector = ASRCorrector(model="m", base_url="http://x/v1", api_key="k", thread_num=1, batch_num=2,
                             prompt_version="v2", context_segments=1)
    try:
        fixed, report = corrector.correct(segments)
    finally:
        corrector.stop()
    assert [item.text for item in fixed][2] == "朱李さんが来た"
    assert sum(item.changed for item in report) == 1
    batch2 = next(text for text in seen if "\"s2\"" in text)
    assert "<context_before>テキスト1</context_before>" in batch2
    assert "<context_after>テキスト4</context_after>" in batch2
    first = next(text for text in seen if "\"s0\"" in text)
    assert "<context_before>" not in first and "<context_after>寿司さんが来た</context_after>" in first


def test_v2_prompt_includes_program_context_and_rejects_unknown_version() -> None:
    corrector = ASRCorrector(model="m", base_url="http://x/v1", api_key="k", program_context="成员：理名",
                             prompt_version="v2")
    try:
        prompt = corrector._build_system_prompt_v2()  # noqa: SLF001
    finally:
        corrector.stop()
    assert prompt.index("成员：理名") < prompt.index("<glossary>")
    assert '"edits"' in prompt
    with pytest.raises(ValueError):
        ASRCorrector(model="m", base_url="http://x/v1", api_key="k", prompt_version="v9")


def test_run_correction_stage_passes_v2_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path / "w")
    paths.transcript_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.transcript_manifest.write_text("[]", encoding="utf-8")
    seen: dict = {}

    class FakeCorrector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def correct(self, transcripts):
            return transcripts, []

        def stop(self):
            pass

    monkeypatch.setattr(corrector_module, "ASRCorrector", FakeCorrector)
    corrector_module.run_correction_stage(
        paths, [_seg("s1", "x")], llm_model="m", base_url="https://relay.example/v1beta", api_key="k",
        mode="llm", prompt_version="v2", context_segments=2, apply_threshold=0.7)
    assert (seen["prompt_version"], seen["context_segments"], seen["apply_threshold"]) == ("v2", 2, 0.7)
