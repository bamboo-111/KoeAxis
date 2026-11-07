from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from optimizer.asr_data import ASRDataSeg
from optimizer.translator import SubtitleTranslator
from qwen_asr import corrector as corrector_module
from qwen_asr import text_proofread as tp
from qwen_asr.cli import build_parser
from qwen_asr.glossary import _write_single_sheet_xlsx, build_glossary_prompt
from qwen_asr.models import TranscriptSegment, WorkPaths
from qwen_asr.program_context import build_program_context, program_context_block, program_context_from_args


def _workdir(tmp_path: Path, source_name: str = "【广播】#18 考试特辑") -> WorkPaths:
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "ep")
    paths.project_metadata.parent.mkdir(parents=True, exist_ok=True)
    paths.project_metadata.write_text(json.dumps({"source_name": source_name}, ensure_ascii=False), encoding="utf-8")
    return paths


def _files(tmp_path: Path) -> list[str]:
    show = tmp_path / "show.md"
    show.write_text("\ufeff# 节目\n成员：理名、夕莉\n", encoding="utf-8")
    anime = tmp_path / "anime.md"
    anime.write_text("角色：仁菜\n", encoding="utf-8")
    return [str(show), str(anime)]


def test_files_are_joined_in_order_with_auto_episode_title(tmp_path: Path) -> None:
    paths = _workdir(tmp_path)
    args = SimpleNamespace(program_context=_files(tmp_path), episode_title="auto")
    text = program_context_from_args(args, paths)
    assert text.index("成员：理名") < text.index("角色：仁菜") < text.index("本期标题")
    assert "【广播】#18 考试特辑" in text
    assert "\ufeff" not in text

    off = program_context_from_args(SimpleNamespace(program_context=_files(tmp_path), episode_title="off"), paths)
    assert "本期标题" not in off
    literal = program_context_from_args(SimpleNamespace(program_context=None, episode_title="特别篇"), paths)
    assert literal.endswith("特别篇")


def test_no_files_and_auto_title_adds_nothing(tmp_path: Path) -> None:
    paths = _workdir(tmp_path)
    assert program_context_from_args(SimpleNamespace(program_context=None, episode_title="auto"), paths) == ""
    assert program_context_from_args(SimpleNamespace(), paths) == ""
    assert program_context_block("") == ""


def test_missing_context_file_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_program_context([tmp_path / "missing.md"])


def test_cli_accepts_repeated_program_context() -> None:
    args = build_parser().parse_args([
        "translate", "--workdir", "w", "--target-language", "简体中文",
        "--program-context", "a.md", "--program-context", "b.md", "--translate-prompt", "v2",
        "--translate-context-lines", "3",
    ])
    assert args.program_context == ["a.md", "b.md"]
    assert (args.translate_prompt, args.translate_context_lines, args.episode_title) == ("v2", 3, "auto")
    run = build_parser().parse_args(["run", "--workdir", "w", "--media", "m.mp3", "--correct-mode", "llm"])
    assert run.correct_mode == "llm" and run.program_context is None


def _translator(**kwargs) -> SubtitleTranslator:
    return SubtitleTranslator(thread_num=1, batch_num=2, model="m", base_url="http://example.invalid/v1",
                              api_key="k", target_language="简体中文", custom_prompt="TERMS", **kwargs)


def test_translator_prompt_is_unchanged_without_context_and_gets_block_with_it() -> None:
    plain = _translator()
    with_ctx = _translator(prompt_version="v2", program_context="成员：理名")
    try:
        assert "<program_context>" not in plain._system_prompt()  # noqa: SLF001
        prompt = with_ctx._system_prompt()  # noqa: SLF001
        assert prompt.index("成员：理名") < prompt.index("<terminology>\nTERMS")
        assert "mishearing" in prompt
    finally:
        plain.stop()
        with_ctx.stop()
    with pytest.raises(ValueError):
        _translator(prompt_version="v9")


def test_translator_sends_read_only_neighbour_context(monkeypatch: pytest.MonkeyPatch) -> None:
    translator = _translator(context_lines=1)
    segments = [ASRDataSeg(f"s{i}", i * 1000, i * 1000 + 900) for i in range(5)]
    chunks = translator._split_chunks(segments)  # noqa: SLF001
    translator._chunk_context = translator._context_for_chunks(chunks)  # noqa: SLF001
    seen = {}

    def fake_loop(_prompt, subtitle, speakers=None, context=None):
        seen[next(iter(subtitle.values()))] = context
        return {key: {"translation": f"译{value}", "confidence": 0.9} for key, value in subtitle.items()}

    monkeypatch.setattr(translator, "_agent_loop", fake_loop)
    try:
        for chunk in chunks:
            translator._translate_chunk(chunk)  # noqa: SLF001
    finally:
        translator.stop()
    assert seen == {"s0": ([], ["s2"]), "s2": (["s1"], ["s4"]), "s4": (["s3"], [])}
    assert segments[3].translated_text == "译s3"


def test_glossary_notes_do_not_leak_from_the_block_on_the_right(tmp_path: Path) -> None:
    path = tmp_path / "g.xlsx"
    _write_single_sheet_xlsx(path, "Sheet1", [
        ["广播用", "中文", "注释", "", "日文", "中文", "注释"],
        ["声ログ", "声log", "成员录音", "", "企画", "策划", "核心创意岗位"],
    ])
    prompt = build_glossary_prompt(path)
    assert "- 声ログ => 声log; 说明：成员录音\n" in prompt + "\n"
    assert "- 企画 => 策划; 说明：核心创意岗位" in prompt


def test_correct_llm_mode_uses_program_context_then_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _workdir(tmp_path)
    paths.transcript_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.transcript_manifest.write_text("[]", encoding="utf-8")
    seen = {}

    class FakeCorrector:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def correct(self, transcripts):
            out, report = [], []
            for item in transcripts:
                new = item.text.replace("優秀勉強", "受験勉強")
                report.append(corrector_module.CorrectionReportItem(
                    segment_id=item.segment_id, original_text=item.text, corrected_text=new,
                    changed=new != item.text, reason="likely 受験勉強"))
                clone = corrector_module.TranscriptSegment(**corrector_module.asdict(item))
                clone.text = new
                out.append(clone)
            return out, report

        def stop(self):
            seen["stopped"] = True

    monkeypatch.setattr(corrector_module, "ASRCorrector", FakeCorrector)
    segment = TranscriptSegment(segment_id="seg_1", audio_path="a.wav", global_start_time=0.0,
                                global_end_time=1.0, text="優秀勉強する", status="completed")
    corrected, report = corrector_module.run_correction_stage(
        paths, [segment], llm_model="m", base_url="https://relay.example/v1beta", api_key="k",
        mode="llm", program_context="成员：理名")
    assert seen["program_context"] == "成员：理名" and seen["stopped"]
    assert corrected[0].text == "受験勉強する"
    assert (report[0].original_text, report[0].changed) == ("優秀勉強する", True)
    assert report[0].reason.startswith("llm: likely 受験勉強")

    rules, _ = corrector_module.run_correction_stage(paths, [segment], llm_model="m", base_url="", api_key="")
    assert rules[0].text == "優秀勉強する"


def test_correct_prompt_has_no_hardcoded_show_terms() -> None:
    corrector = corrector_module.ASRCorrector(model="m", base_url="http://x/v1", api_key="k",
                                              program_context="成员：理名")
    try:
        prompt = corrector._build_system_prompt()  # noqa: SLF001
    finally:
        corrector.stop()
    assert "togeraji" not in prompt
    assert prompt.index("成员：理名") < prompt.index("<glossary>")


def test_text_proofread_prompt_includes_program_context() -> None:
    config = tp.TextProofreadConfig(program="P", program_context="成员：理名")
    prompt = tp.build_prompt(config, glossary=[], before=[], targets=[], after=[])
    assert prompt.startswith("你是P的中文字幕校对编辑。\n<program_context>")
    plain = tp.build_prompt(tp.TextProofreadConfig(program="P"), glossary=[], before=[], targets=[], after=[])
    assert "<program_context>" not in plain
