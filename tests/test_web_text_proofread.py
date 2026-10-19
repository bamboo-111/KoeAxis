from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.ci.pipeline import discover_presets
from qwen_asr.cli import build_parser
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic
from qwen_asr.web.ci_presets import PRESETS_DIR, preset_web_settings
from qwen_asr.web.commands import build_command
from qwen_asr.web.server import missing_text_proofread_credential
from qwen_asr.web.stage_start_service import (
    StageStartError,
    _base_payload,
    build_workspace_stage_payload,
    gemini_relay_configured,
    stage_start_capability,
)
from qwen_asr.web.static_html import INDEX_HTML
from qwen_asr.web.status import build_progress, get_status

GEMINI_ENV = ("GEMINI_API_KEY",) + tuple(f"GEMINI_API_KEY_{index}" for index in range(1, 10))


@pytest.fixture
def no_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GEMINI_ENV:
        monkeypatch.delenv(name, raising=False)


def _translated_workspace(tmp_path: Path) -> WorkPaths:
    paths = WorkPaths.from_workdir(tmp_path / "work")
    write_json_atomic(paths.segments_manifest, [{"segment_id": "1"}])
    write_json_atomic(paths.translated_manifest, {"1": {"original_subtitle": "x", "translated_subtitle": "y"}})
    return paths


def _parse(command: list[str]):
    return build_parser().parse_args(command[2:])


def test_text_proofread_stage_command_passes_flags(tmp_path: Path) -> None:
    command = build_command(
        {
            "stage": "text-proofread",
            "workdir": str(tmp_path),
            "resume": False,
            "force": True,
            "glossary_xlsx": "glossary.xlsx",
            "text_proofread_effort": "high",
            "text_proofread_model": "gemini-x",
            "text_proofread_concurrency": "6",
            "text_proofread_timeout": "300",
            "text_proofread_apply_threshold": "0.9",
            "text_proofread_program": "电台节目",
            "text_proofread_hosts": "理名、夕莉",
        }
    )
    assert command[2] == "text-proofread"
    args = _parse(command)
    assert args.text_proofread_effort == "high"
    assert args.text_proofread_model == "gemini-x"
    assert args.text_proofread_concurrency == 6
    assert args.text_proofread_timeout == 300.0
    assert args.text_proofread_apply_threshold == 0.9
    assert args.text_proofread_program == "电台节目"
    assert args.text_proofread_hosts == "理名、夕莉"
    assert args.glossary_xlsx == "glossary.xlsx"
    assert args.force is True and args.resume is False
    assert command.count("--force") == 1 and command.count("--no-resume") == 1


def test_text_proofread_stage_defaults_and_bad_effort(tmp_path: Path) -> None:
    command = build_command({"stage": "text-proofread", "workdir": str(tmp_path)})
    args = _parse(command)
    assert args.text_proofread_effort == "medium"
    assert args.resume is True and args.force is False
    with pytest.raises(ValueError):
        build_command({"stage": "text-proofread", "workdir": str(tmp_path), "text_proofread_effort": "max"})


@pytest.mark.parametrize("stage", ["run", "batch-run"])
def test_run_and_batch_run_pass_text_proofread_only_when_enabled(tmp_path: Path, stage: str) -> None:
    payload = _base_payload(
        WorkPaths.from_workdir(tmp_path),
        {"mediaPath": "media.mp3", "textProofreadEffort": "low", "textProofreadHosts": "理名"},
        {},
    )
    payload.update(stage=stage, media_paths=["media.mp3"])
    without = build_command(payload)
    assert "--with-text-proofread" not in without
    assert not any(item.startswith("--text-proofread-") for item in without)

    payload["with_text_proofread"] = True
    command = build_command(payload)
    assert command.count("--with-text-proofread") == 1
    assert command[command.index("--text-proofread-effort") + 1] == "low"
    assert command[command.index("--text-proofread-hosts") + 1] == "理名"
    if stage == "run":
        args = _parse(command)
        assert args.with_text_proofread is True and args.text_proofread_effort == "low"


def test_togeradi_preset_round_trips_text_proofread_settings() -> None:
    preset = discover_presets([PRESETS_DIR])["togeradi"]
    converted = preset_web_settings(preset)
    settings = converted["settings"]
    assert settings["withTextProofread"] is True
    assert settings["textProofreadEffort"] == "medium"
    assert "トゲラジ" in settings["textProofreadProgram"]
    assert settings["textProofreadHosts"].startswith("理名")
    assert not any("text_proofread" in dest for dest in converted["unmapped"])

    payload = _base_payload(WorkPaths.from_workdir(Path("w")), {**settings, "mediaPath": "m.mp3"}, {})
    payload["stage"] = "run"
    args = _parse(build_command(payload))
    assert args.with_text_proofread is True
    assert args.text_proofread_program == settings["textProofreadProgram"]
    assert args.text_proofread_hosts == settings["textProofreadHosts"]


def test_stage_start_text_proofread_requires_gemini_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_gemini: None
) -> None:
    paths = _translated_workspace(tmp_path)
    assert stage_start_capability("text-proofread", [])["runnable"] is True
    with pytest.raises(StageStartError) as missing:
        build_workspace_stage_payload(paths, stage="text-proofread", settings={})
    assert missing.value.code == "STAGE_CREDENTIAL_MISSING"

    monkeypatch.setenv("GEMINI_API_KEY_3", "relay-key")
    payload = build_workspace_stage_payload(
        paths,
        stage="text-proofread",
        settings={"textProofreadEffort": "high", "textProofreadProgram": "节目", "textProofreadConcurrency": "2"},
    )
    assert payload["text_proofread_effort"] == "high"
    command = build_command(payload)
    assert command[2] == "text-proofread"
    assert command[command.index("--text-proofread-program") + 1] == "节目"
    assert command[command.index("--text-proofread-concurrency") + 1] == "2"


def test_stage_start_other_stages_ignore_text_proofread_setting(tmp_path: Path, no_gemini: None) -> None:
    paths = _translated_workspace(tmp_path)
    payload = build_workspace_stage_payload(paths, stage="quality-gate", settings={"withTextProofread": True})
    assert payload["stage"] == "quality-gate"


def test_gemini_relay_detection_and_api_start_check() -> None:
    assert gemini_relay_configured({}) is False
    assert gemini_relay_configured({"GEMINI_API_KEY": "  "}) is False
    assert gemini_relay_configured({"GEMINI_API_KEY_9": "k"}) is True
    assert missing_text_proofread_credential({"stage": "text-proofread"}, {})
    assert missing_text_proofread_credential({"stage": "run", "with_text_proofread": True}, {})
    assert missing_text_proofread_credential({"stage": "batch-run", "with_text_proofread": True}, {})
    assert missing_text_proofread_credential({"stage": "run", "with_text_proofread": False}, {}) == ""
    assert missing_text_proofread_credential({"stage": "translate", "with_text_proofread": True}, {}) == ""
    assert missing_text_proofread_credential({"stage": "text-proofread"}, {"GEMINI_API_KEY": "k"}) == ""


def test_status_and_progress_include_text_proofread(tmp_path: Path) -> None:
    paths = _translated_workspace(tmp_path)
    write_json_atomic(paths.text_proofread_manifest, {"1": {"translated_subtitle": "z"}})
    write_json_atomic(
        paths.text_proofread_report,
        {"status": "PASS", "applied_count": 3, "request_count": 10, "failed_count": 1},
    )
    status = get_status(str(paths.workdir))
    assert status["files"]["text_proofread"] is True
    assert status["counts"]["text_proofread"] == 1
    assert "text-proofread" in status["stages"]

    progress = build_progress(
        {"workdir": str(paths.workdir), "stage": "text-proofread", "status": "completed", "started_at": 0, "finished_at": 1}
    )
    assert progress["total"] == 10 and progress["completed"] == 9
    assert progress["summary"] == "text-proofread PASS: 3 applied, 1/10 requests failed"


def test_index_has_text_proofread_controls() -> None:
    for snippet in (
        'id="withTextProofread"',
        'id="textProofreadEffort"',
        '<option value="medium" selected>medium</option>',
        'id="textProofreadProgram"',
        'id="textProofreadHosts"',
        "'proofread-realign', 'text-proofread', 'quality-gate'",
        "with_text_proofread: document.getElementById('withTextProofread').checked",
        "text_proofread_effort: document.getElementById('textProofreadEffort').value",
        "文本校对（Gemini）",
    ):
        assert snippet in INDEX_HTML, snippet
