from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic
from qwen_asr.web.commands import build_command
from qwen_asr.web.stage_start_service import (
    StageStartError,
    build_workspace_stage_payload,
    stage_start_capability,
)


def test_stage_start_builds_shared_cli_payload_without_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "x"}})
    monkeypatch.setenv("MIMO_API_KEY", "environment-value")

    payload = build_workspace_stage_payload(
        paths,
        stage="translate",
        settings={
            "llmModel": "mimo-v2.5",
            "llmBaseUrl": "https://api.xiaomimimo.com/v1",
            "targetLanguage": "简体中文",
            "disableThinking": True,
        },
    )
    command = build_command(payload)

    assert payload["resume"] is True
    assert "llm_api_key" not in payload
    assert "--llm-api-key" not in command
    assert command[command.index("--llm-model") + 1] == "mimo-v2.5"
    assert "--disable-thinking" in command


def test_stage_start_passes_saved_speaker_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path / "work")
    write_json_atomic(paths.aligned_manifest, [{"segment_id": "1", "status": "completed"}])
    monkeypatch.setenv("MIMO_API_KEY", "environment-value")
    enroll = tmp_path / "enroll.json"
    enroll.write_text("{}", encoding="utf-8")

    payload = build_workspace_stage_payload(
        paths,
        stage="split",
        settings={
            "llmModel": "mimo-v2.5",
            "llmBaseUrl": "https://api.xiaomimimo.com/v1",
            "speakerEnroll": str(enroll),
            "speakers": "理名,夕莉",
            "asrContextFile": "ctx.txt",
            "programMemoryFile": "memory.json",
            "alignRecovery": True,
        },
    )
    assert payload["asr_context_file"] == "ctx.txt"
    assert payload["program_memory_file"] == "memory.json"
    assert payload["align_recovery"] == "auto"
    assert payload["proofread_realign_mfa_fallback"] == "off"
    command = build_command(payload)

    assert command[command.index("--speaker-enroll") + 1] == str(enroll)
    assert command[command.index("--speakers") + 1] == "理名,夕莉"


def test_stage_start_rejects_nested_credentials_and_missing_inputs(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    with pytest.raises(StageStartError) as credentials:
        build_workspace_stage_payload(
            paths,
            stage="split",
            settings={"llmApiKey": None},
        )
    assert credentials.value.code == "STAGE_SETTINGS_CONTAIN_CREDENTIALS"

    with pytest.raises(StageStartError) as missing:
        build_workspace_stage_payload(paths, stage="align", settings={})
    assert missing.value.code == "STAGE_INPUTS_MISSING"


def test_stage_start_requires_environment_credential_for_llm_stage(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "x"}})
    monkeypatch.delenv("MIMO_API_KEY", raising=False)

    with pytest.raises(StageStartError) as exc_info:
        build_workspace_stage_payload(
            paths,
            stage="translate",
            settings={"llmBaseUrl": "https://api.xiaomimimo.com/v1"},
        )
    assert exc_info.value.code == "STAGE_CREDENTIAL_MISSING"


def test_prepare_stage_uses_existing_project_media(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path / "workspace")
    media = tmp_path / "source.wav"
    media.write_bytes(b"audio")
    write_json_atomic(paths.project_metadata, {"original_media_path": str(media)})

    payload = build_workspace_stage_payload(paths, stage="prepare", settings={})

    assert payload["media_path"] == str(media)


def test_web_stage_export_is_srt_only_even_with_legacy_format_setting(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.normalized_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.normalized_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\ntext\n", encoding="utf-8")

    payload = build_workspace_stage_payload(
        paths,
        stage="export",
        settings={"format": "both"},
    )

    assert payload["format"] == "srt"


def test_proofread_realign_is_reported_as_pipeline_managed() -> None:
    assert stage_start_capability("proofread-realign", [])["reason"] == "managed_by_pipeline"


def test_translate_stage_defaults_to_gemini_38_flash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qwen_asr.defaults import DEFAULT_LLM_BASE_URL

    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "x"}})
    monkeypatch.setenv("GEMINI_API_KEY", "environment-value")

    payload = build_workspace_stage_payload(paths, stage="translate", settings={})
    command = build_command(payload)

    assert payload["llm_model"] == "gemini-3.8-flash"
    assert payload["llm_base_url"] == DEFAULT_LLM_BASE_URL
    assert "/v1beta" in payload["llm_base_url"]
    assert str(payload["thread_num"]) == "4"
    assert command[command.index("--llm-model") + 1] == "gemini-3.8-flash"
    assert "llm_api_key" not in payload
    assert "--llm-api-key" not in command
