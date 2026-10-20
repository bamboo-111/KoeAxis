from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from qwen_asr.ci.config import DEFAULT_INPUT_FLAGS, Settings
from qwen_asr.ci.pipeline import build_run_argv, discover_presets
from qwen_asr.cli import build_parser
from qwen_asr.defaults import (
    DEFAULT_CORRECT_MODE,
    DEFAULT_TRANSLATE_CONTEXT_LINES,
    DEFAULT_TRANSLATE_PROMPT,
    TRANSLATE_PROMPT_VERSIONS,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic
from qwen_asr.web.ci_presets import PRESETS_DIR, preset_run_argv, preset_web_settings
from qwen_asr.web.commands import build_command
from qwen_asr.web.stage_start_service import StageStartError, _base_payload, build_workspace_stage_payload
from qwen_asr.web.static_html import INDEX_HTML, INDEX_TEMPLATE_PATH

LLM = {"llm_model": "m", "llm_base_url": "https://example.invalid/v1", "target_language": "简体中文"}


def _parse(command: list[str]):
    return build_parser().parse_args(command[2:])


def _context_files(tmp_path: Path) -> list[str]:
    paths = []
    for name in ("show.md", "series.md"):
        path = tmp_path / name
        path.write_text(f"# {name}\n", encoding="utf-8")
        paths.append(str(path))
    return paths


def _values(command: list[str], flag: str) -> list[str]:
    return [command[index + 1] for index, item in enumerate(command) if item == flag]


@pytest.mark.parametrize("stage", ["translate", "correct", "text-proofread"])
def test_stage_commands_pass_program_context_and_episode_title(tmp_path: Path, stage: str) -> None:
    files = _context_files(tmp_path)
    payload = {
        "stage": stage,
        "workdir": str(tmp_path),
        **LLM,
        "program_context_files": [files[0], " ", files[1]],
        "episode_title": "第29回",
        "translate_prompt": "v2",
        "translate_context_lines": "3",
        "correct_mode": "llm",
    }
    command = build_command(payload)
    assert command[2] == stage
    assert _values(command, "--program-context") == files
    args = _parse(command)
    assert args.program_context == files
    assert args.episode_title == "第29回"
    if stage == "translate":
        assert args.translate_prompt == "v2" and args.translate_context_lines == 3
        assert "--correct-mode" not in command
    elif stage == "correct":
        assert args.correct_mode == "llm"
        assert "--translate-prompt" not in command and "--translate-context-lines" not in command
    else:
        assert not {"--translate-prompt", "--translate-context-lines", "--correct-mode"} & set(command)


def test_stage_commands_omit_empty_prompt_context(tmp_path: Path) -> None:
    command = build_command({"stage": "translate", "workdir": str(tmp_path), **LLM, "episode_title": " "})
    assert "--program-context" not in command and "--episode-title" not in command
    args = _parse(command)
    assert args.program_context is None and args.episode_title == "auto"
    assert args.translate_prompt == DEFAULT_TRANSLATE_PROMPT


def test_program_context_accepts_newline_separated_text(tmp_path: Path) -> None:
    files = _context_files(tmp_path)
    command = build_command(
        {"stage": "text-proofread", "workdir": str(tmp_path), "program_context_files": f"{files[0]}\n\n {files[1]} \n"}
    )
    assert _values(command, "--program-context") == files


@pytest.mark.parametrize("stage", ["run", "batch-run"])
def test_run_and_batch_run_pass_prompt_context_flags(tmp_path: Path, stage: str) -> None:
    files = _context_files(tmp_path)
    settings = {
        "mediaPath": "media.mp3",
        "programContextFiles": "\n".join(files),
        "episodeTitle": "off",
        "translatePrompt": "v2",
        "translateContextLines": "2",
        "correctMode": "llm",
    }
    payload = _base_payload(WorkPaths.from_workdir(tmp_path), settings, {})
    payload.update(stage=stage, media_paths=["media.mp3"])
    assert payload["program_context_files"] == files
    command = build_command(payload)
    assert _values(command, "--program-context") == files
    assert _values(command, "--episode-title") == ["off"]
    assert _values(command, "--translate-prompt") == ["v2"]
    assert _values(command, "--translate-context-lines") == ["2"]
    assert _values(command, "--correct-mode") == ["llm"]
    args = _parse(command)
    assert args.program_context == files and args.correct_mode == "llm" and args.translate_prompt == "v2"


def test_run_defaults_come_from_qwen_asr_defaults(tmp_path: Path) -> None:
    payload = _base_payload(WorkPaths.from_workdir(tmp_path), {"mediaPath": "media.mp3"}, {})
    payload["stage"] = "run"
    command = build_command(payload)
    assert "--program-context" not in command
    args = _parse(command)
    assert args.episode_title == "auto"
    assert args.translate_prompt == DEFAULT_TRANSLATE_PROMPT
    assert args.translate_context_lines == DEFAULT_TRANSLATE_CONTEXT_LINES
    assert args.correct_mode == DEFAULT_CORRECT_MODE


@pytest.mark.parametrize("stage", ["translate", "correct", "text-proofread", "run", "batch-run"])
def test_missing_program_context_file_is_rejected(tmp_path: Path, stage: str) -> None:
    payload = _base_payload(
        WorkPaths.from_workdir(tmp_path),
        {"mediaPath": "media.mp3", "programContextFiles": str(tmp_path / "missing.md")},
        {},
    )
    payload.update(stage=stage, media_paths=["media.mp3"])
    with pytest.raises(ValueError, match="Program context file not found"):
        build_command(payload)


def test_invalid_prompt_options_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_command({"stage": "translate", "workdir": str(tmp_path), **LLM, "translate_prompt": "v9"})
    with pytest.raises(ValueError):
        build_command({"stage": "correct", "workdir": str(tmp_path), **LLM, "correct_mode": "magic"})


def test_workspace_stage_start_rejects_missing_program_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    paths = WorkPaths.from_workdir(tmp_path / "work")
    write_json_atomic(paths.segments_manifest, [{"segment_id": "1"}])
    write_json_atomic(paths.translated_manifest, {"1": {"original_subtitle": "x", "translated_subtitle": "y"}})
    with pytest.raises(StageStartError) as missing:
        build_workspace_stage_payload(
            paths, stage="text-proofread", settings={"programContextFiles": str(tmp_path / "nope.md")}
        )
    assert missing.value.code == "STAGE_PROGRAM_CONTEXT_MISSING"

    files = _context_files(tmp_path)
    payload = build_workspace_stage_payload(
        paths, stage="text-proofread", settings={"programContextFiles": "\n".join(files), "episodeTitle": "t"}
    )
    command = build_command(payload)
    assert _values(command, "--program-context") == files and _values(command, "--episode-title") == ["t"]


def test_build_run_argv_emits_every_program_context_file(tmp_path: Path) -> None:
    files = [("--program-context", tmp_path / "a.md"), ("--program-context", tmp_path / "b.md")]
    argv = build_run_argv(
        {"run_args": ["--with-translate"]},
        workdir=tmp_path / "w",
        media=tmp_path / "m.mp3",
        export_path=tmp_path / "o.srt",
        file_paths=files,
    )
    assert [argv[index + 1] for index, item in enumerate(argv) if item == "--program-context"] == [
        str(tmp_path / "a.md"),
        str(tmp_path / "b.md"),
    ]
    assert build_parser().parse_args(["run", *argv]).program_context == [str(path) for _, path in files]


def test_ci_accepts_program_context_uploads() -> None:
    assert "--program-context" in DEFAULT_INPUT_FLAGS
    assert "--program-context" in Settings().input_flags


def test_preset_round_trips_two_program_context_files(tmp_path: Path) -> None:
    samples = (PRESETS_DIR.parent / "samples" / "program-context").resolve()
    (tmp_path / "ctx.json").write_text(json.dumps({
        "id": "ctx",
        "files": [{"flag": "--program-context", "path": str(samples / "togeradi.md")},
                  {"flag": "--program-context", "path": str(samples / "girls-band-cry.md")}],
    }), encoding="utf-8")
    preset = discover_presets([tmp_path])["ctx"]
    context_files = [item.path for item in preset.files if item.flag == "--program-context"]
    assert [path.name for path in context_files] == ["togeradi.md", "girls-band-cry.md"]
    assert all(path.is_file() for path in context_files)
    assert preset_run_argv(preset).count("--program-context") == 2

    converted = preset_web_settings(preset)
    settings = converted["settings"]
    assert settings["programContextFiles"].split("\n") == [str(path) for path in context_files]
    assert settings["episodeTitle"] == "auto"
    assert settings["translatePrompt"] == DEFAULT_TRANSLATE_PROMPT
    assert settings["translateContextLines"] == str(DEFAULT_TRANSLATE_CONTEXT_LINES)
    assert settings["correctMode"] == DEFAULT_CORRECT_MODE
    assert "program_context" not in converted["unmapped"]

    payload = _base_payload(WorkPaths.from_workdir(Path("w")), {**settings, "mediaPath": "m.mp3"}, {})
    payload["stage"] = "run"
    args = _parse(build_command(payload))
    assert args.program_context == [str(path) for path in context_files]


def test_preset_run_args_map_prompt_options(tmp_path: Path) -> None:
    (tmp_path / "ctx.md").write_text("x", encoding="utf-8")
    (tmp_path / "custom.json").write_text(
        '{"id": "custom", "run_args": ["--translate-prompt", "v2", "--translate-context-lines", "4",'
        ' "--correct-mode", "llm", "--episode-title", "off"],'
        ' "files": [{"flag": "--program-context", "path": "ctx.md"}]}',
        encoding="utf-8",
    )
    converted = preset_web_settings(discover_presets([tmp_path])["custom"])
    settings = converted["settings"]
    assert settings["translatePrompt"] == "v2"
    assert settings["translateContextLines"] == "4"
    assert settings["correctMode"] == "llm"
    assert settings["episodeTitle"] == "off"
    assert settings["programContextFiles"] == str((tmp_path / "ctx.md").resolve())
    assert converted["unmapped"] == []


def _selected_option(select_id: str) -> str:
    select = re.search(rf'<select id="{select_id}">(.*?)</select>', INDEX_HTML).group(1)
    selected = re.findall(r'<option value="([^"]+)" selected>', select)
    assert len(selected) == 1, select
    return selected[0]


def test_index_html_defaults_equal_cli_defaults() -> None:
    assert _selected_option("translatePrompt") == DEFAULT_TRANSLATE_PROMPT
    assert _selected_option("correctMode") == DEFAULT_CORRECT_MODE
    prompt_select = re.search(r'<select id="translatePrompt">(.*?)</select>', INDEX_HTML).group(1)
    assert tuple(re.findall(r'<option value="([^"]+)"', prompt_select)) == tuple(TRANSLATE_PROMPT_VERSIONS)
    context_lines = re.search(r'<input id="translateContextLines"[^>]*value="([^"]*)"', INDEX_HTML).group(1)
    assert context_lines == str(DEFAULT_TRANSLATE_CONTEXT_LINES)
    assert re.search(r'<input id="episodeTitle"[^>]*value="auto"', INDEX_HTML)
    assert "{{" not in INDEX_HTML.split('id="programContextFiles"', 1)[1].split("</details>", 1)[0]
    # the template takes the defaults from qwen_asr.defaults instead of hard-coding them
    template = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    assert 'value="{{DEFAULT_TRANSLATE_CONTEXT_LINES}}"' in template
    assert "{{SELECTED_TRANSLATE_PROMPT_V1}}" in template and "{{SELECTED_CORRECT_MODE_RULES}}" in template


def test_index_has_prompt_context_controls() -> None:
    for snippet in (
        'id="programContextFiles"',
        'id="episodeTitle"',
        'id="translatePrompt"',
        'id="translateContextLines"',
        'id="correctMode"',
        "'programContextFiles', 'episodeTitle', 'translatePrompt', 'translateContextLines', 'correctMode'",
        "program_context_files: document.getElementById('programContextFiles').value.split('\\n')",
        "episode_title: document.getElementById('episodeTitle').value.trim()",
        "translate_prompt: document.getElementById('translatePrompt').value",
        "translate_context_lines: document.getElementById('translateContextLines').value",
        "correct_mode: document.getElementById('correctMode').value",
        "节目背景提示词：每行一个文件，按顺序合并后加入纠错 / 翻译 / 文本校对的提示词；本期标题 auto = 媒体文件名",
    ):
        assert snippet in INDEX_HTML, snippet
