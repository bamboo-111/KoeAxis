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


def test_proofread_realign_is_reported_as_pipeline_managed() -> None:
    assert stage_start_capability("proofread-realign", [])["reason"] == "managed_by_pipeline"
