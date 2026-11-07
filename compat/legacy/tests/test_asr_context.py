from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from qwen_asr.asr import QwenASRTranscriber
from qwen_asr.asr_context import (
    build_context_prompt,
    detect_context_leak,
    load_context_terms,
    merge_context_edits,
    parse_context_terms,
)
from qwen_asr.models import AudioSegment

TERMS = ["トゲナシトゲアリ", "トゲラジ", "InterFM", "輪廻の理", "ダイヤモンドダスト", "ガールズバンドクライ"]


def _segment(index: int) -> AudioSegment:
    return AudioSegment(
        segment_id=f"segment_{index:06d}",
        audio_path=f"segment_{index}.wav",
        source_audio_path="source.wav",
        global_start_time=float(index),
        global_end_time=float(index + 1),
        duration=1.0,
        status="prepared",
    )


def test_parse_and_load_context_terms(tmp_path: Path) -> None:
    assert parse_context_terms("トゲラジ, InterFM、トゲラジ\n# comment\ninterfm") == ["トゲラジ", "InterFM"]
    terms_file = tmp_path / "terms.txt"
    terms_file.write_text("# show terms\nトゲナシトゲアリ\nトゲラジ, InterFM\n", encoding="utf-8")
    assert load_context_terms("ガールズバンドクライ", terms_file) == [
        "トゲナシトゲアリ",
        "トゲラジ",
        "InterFM",
        "ガールズバンドクライ",
    ]
    assert build_context_prompt(["a", "b"]) == "用語: a, b"
    assert build_context_prompt([]) == ""


def test_detect_context_leak_flags_copied_term_runs_only() -> None:
    leaked = "アドレスはトゲラジアットインタFMドットJP, togeradi@interfm.jp. 輪廻の理, ダイヤモンドダスト, ガールズバンドクライ."
    assert detect_context_leak(leaked, TERMS)
    assert detect_context_leak("用語: トゲラジ", TERMS)
    assert not detect_context_leak("トゲナシトゲアリのトゲラジ、InterFMからお届けします。", ["トゲラジ"])
    assert not detect_context_leak("トゲナシトゲアリのトゲラジ、今週もInterFMから。", TERMS)
    assert not detect_context_leak("トゲラジ、トゲラジ", TERMS)
    assert not detect_context_leak(leaked, [])


def test_transcriber_without_terms_does_not_send_context(monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.asr._cleanup_torch", lambda full=False: None)
    calls: list[dict] = []

    class FakeModel:
        def transcribe(self, **kwargs):
            calls.append(kwargs)
            return [{"text": "a"}, {"text": "b"}]

    transcriber = QwenASRTranscriber(model_name="model", batch_size=2, language="Japanese")
    transcriber._model = FakeModel()
    results = transcriber.run_batch([_segment(1), _segment(2)])

    assert [item.text for item in results] == ["a", "b"]
    assert calls == [{"audio": ["segment_1.wav", "segment_2.wav"], "language": "Japanese"}]


def test_transcriber_retranscribes_leaked_segment_without_context(monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.asr._cleanup_torch", lambda full=False: None)
    calls: list[dict] = []

    class FakeModel:
        def transcribe(self, **kwargs):
            calls.append(kwargs)
            if isinstance(kwargs["audio"], list):
                return [
                    {"text": "トゲナシトゲアリのトゲラジ。"},
                    {"text": "輪廻の理, ダイヤモンドダスト, ガールズバンドクライ"},
                ]
            return [{"text": "本当の発話です。"}]

    transcriber = QwenASRTranscriber(
        model_name="model", batch_size=2, language="Japanese", context_terms=TERMS, context_mode="full"
    )
    transcriber._model = FakeModel()
    results = transcriber.run_batch([_segment(1), _segment(2)])

    assert [item.text for item in results] == ["トゲナシトゲアリのトゲラジ。", "本当の発話です。"]
    assert calls[0]["context"] == build_context_prompt(TERMS)
    assert calls[1] == {"audio": "segment_2.wav", "language": "Japanese"}
    assert transcriber.consume_context_leak_segments() == ["segment_000002"]
    assert transcriber.consume_context_leak_segments() == []


def test_single_segment_leak_is_retranscribed(monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.asr._cleanup_torch", lambda full=False: None)
    outputs = [[{"text": "用語: トゲラジ, InterFM"}], [{"text": "普通の発話"}]]
    seen_context: list[bool] = []

    class FakeModel:
        def transcribe(self, **kwargs):
            seen_context.append("context" in kwargs)
            return outputs.pop(0)

    transcriber = QwenASRTranscriber(model_name="model", context_terms=TERMS, context_mode="full")
    transcriber._model = FakeModel()
    result = transcriber.run_segment(_segment(3), cleanup=False)

    assert result.text == "普通の発話"
    assert seen_context == [True, False]


def test_mimo_proofread_command_passes_models(monkeypatch, tmp_path: Path) -> None:
    from qwen_asr.commands import mimo_proofread as command_module
    from qwen_asr.commands import stages as stage_helpers
    from qwen_asr.defaults import DEFAULT_MIMO_MODEL

    captured: list[list[str]] = []
    monkeypatch.setattr(stage_helpers, "resolve_mimo_api_key", lambda value: "key")
    monkeypatch.setattr(stage_helpers, "run_mimo_proofread", lambda command: captured.append(command) or 0)
    work_paths = SimpleNamespace(workdir=tmp_path, mimo_proofread_dir=tmp_path / "mimo")

    def option(command: list[str], flag: str) -> str | None:
        return command[command.index(flag) + 1] if flag in command else None

    command_module.cmd_mimo_proofread(argparse.Namespace(), work_paths)
    assert option(captured[-1], "--model") == DEFAULT_MIMO_MODEL

    # A Gemini text model must not leak into the MiMo proofread endpoint.
    command_module.cmd_mimo_proofread(argparse.Namespace(llm_model="gemini-3.8-flash"), work_paths)
    assert option(captured[-1], "--model") == DEFAULT_MIMO_MODEL

    command_module.cmd_mimo_proofread(argparse.Namespace(llm_model="mimo-v2.6-pro"), work_paths)
    assert option(captured[-1], "--model") == "mimo-v2.6-pro"
    assert option(captured[-1], "--audio-model") is None

    command_module.cmd_mimo_proofread(
        argparse.Namespace(
            llm_model="mimo-v2.6-pro",
            mimo_model="mimo-v2.6-flash",
            mimo_text_model="mimo-v2.6-pro",
            mimo_audio_model="mimo-v2.6-flash",
        ),
        work_paths,
    )
    assert option(captured[-1], "--model") == "mimo-v2.6-flash"
    assert option(captured[-1], "--text-model") == "mimo-v2.6-pro"
    assert option(captured[-1], "--audio-model") == "mimo-v2.6-flash"


def test_mimo_proofread_script_defaults_to_project_model() -> None:
    from qwen_asr.defaults import DEFAULT_MIMO_BASE_URL, DEFAULT_MIMO_MODEL
    from qwen_asr.mimo_proofread import _parse_args

    args = _parse_args(["--workdir", "w", "--api-key", "k"])
    assert args.model == DEFAULT_MIMO_MODEL
    assert args.base_url == DEFAULT_MIMO_BASE_URL
    assert args.text_model == ""
    assert args.audio_model == ""


def test_cli_and_web_expose_model_and_context_options() -> None:
    from qwen_asr.cli import build_parser
    from qwen_asr.web.commands import _asr_context_args, _mimo_proofread_args

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--workdir",
            "w",
            "--split-llm-model",
            "mimo-v2.6-flash",
            "--mimo-audio-model",
            "mimo-v2.6-flash",
            "--asr-context-terms",
            "トゲラジ,InterFM",
        ]
    )
    assert args.split_llm_model == "mimo-v2.6-flash"
    assert args.mimo_audio_model == "mimo-v2.6-flash"
    assert args.mimo_model is None
    assert args.asr_context_terms == "トゲラジ,InterFM"

    assert _asr_context_args({"asr_context_file": " terms.txt "}) == ["--asr-context-file", "terms.txt"]
    assert _asr_context_args({}) == []
    web_args = _mimo_proofread_args({"llm_model": "mimo-v2.6-pro", "mimo_audio_model": "mimo-v2.6-flash"})
    assert web_args[web_args.index("--mimo-model") + 1] == "mimo-v2.6-pro"
    assert web_args[web_args.index("--mimo-audio-model") + 1] == "mimo-v2.6-flash"


def test_split_command_prefers_split_llm_model(monkeypatch, tmp_path: Path) -> None:
    from qwen_asr.commands import split as split_module
    from qwen_asr.models import WorkPaths

    captured: dict = {}
    monkeypatch.setattr(split_module, "run_split_stage", lambda **kwargs: captured.update(kwargs))
    args = argparse.Namespace(
        force=False,
        resume=False,
        optimizer_root=str(tmp_path),
        llm_model="mimo-v2.6-pro",
        split_llm_model="mimo-v2.6-flash",
        llm_base_url="",
        llm_api_key=None,
        thread_num=1,
        max_word_count_cjk=18,
        max_word_count_english=18,
        prompt_limit_ratio=0.8,
        disable_thinking=True,
        llm_extra_body_json="",
        timeout=30.0,
    )
    split_module.cmd_split(args, WorkPaths.from_workdir(tmp_path))
    assert captured["llm_model"] == "mimo-v2.6-flash"


def test_merge_context_edits_keeps_only_term_edits() -> None:
    base = "インターフェームトゲな人気ありのトゲラジ川崎から世界へ。アイス、何味が好き？"
    ctx = "InterFM トゲナシトゲアリのトゲラジ、川崎から世界へ。愛して。何味が好き？"
    merged = merge_context_edits(base, ctx, TERMS)
    assert merged == "InterFM トゲナシトゲアリのトゲラジ川崎から世界へ。アイス、何味が好き？"
    assert merge_context_edits("エネルギー、元気", "NG、元気", TERMS) == "エネルギー、元気"
    assert merge_context_edits("本当の発話", "輪廻の理, ダイヤモンドダスト, ガールズバンドクライ", TERMS) == "本当の発話"
    assert merge_context_edits("同じ", "同じ", TERMS) == "同じ"
    assert merge_context_edits("とげラジです", "トゲラジです", []) == "とげラジです"


def test_transcriber_merge_mode_runs_both_passes_and_merges(monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.asr._cleanup_torch", lambda full=False: None)
    calls: list[bool] = []

    class FakeModel:
        def transcribe(self, **kwargs):
            with_context = "context" in kwargs
            calls.append(with_context)
            if with_context:
                return [{"text": "トゲナシトゲアリのNG。"}, {"text": "輪廻の理, ダイヤモンドダスト, ガールズバンドクライ"}]
            return [{"text": "トゲなしトゲありのエネルギー。"}, {"text": "本当の発話です。"}]

    transcriber = QwenASRTranscriber(model_name="model", batch_size=2, language="Japanese", context_terms=TERMS)
    transcriber._model = FakeModel()
    results = transcriber.run_batch([_segment(1), _segment(2)])

    assert calls == [True, False]
    assert [item.text for item in results] == ["トゲナシトゲアリのエネルギー。", "本当の発話です。"]
    assert transcriber.consume_context_term_edit_segments() == ["segment_000001"]
    assert transcriber.consume_context_leak_segments() == ["segment_000002"]


def test_transcriber_merge_mode_single_segment(monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.asr._cleanup_torch", lambda full=False: None)

    class FakeModel:
        def transcribe(self, **kwargs):
            return [{"text": "InterFMから。" if "context" in kwargs else "インターフェムから。"}]

    transcriber = QwenASRTranscriber(model_name="model", context_terms=TERMS)
    transcriber._model = FakeModel()
    assert transcriber.run_segment(_segment(4), cleanup=False).text == "InterFMから。"
