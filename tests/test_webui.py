from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import webapp
from qwen_asr.defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
)
from qwen_asr.glossary import _write_single_sheet_xlsx
from qwen_asr.web import server
from qwen_asr.web.status import get_status
from qwen_asr.web.commands import (
    ROOT,
    WORKSPACES_DIR,
    build_command,
    normalize_asr_language,
    python_runtime,
    resolve_deletable_workspace,
    resolve_deletable_workspaces,
    suggest_workdir,
)
from qwen_asr.web.static_html import INDEX_HTML, INDEX_TEMPLATE_PATH, load_index_html
from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT


def test_webui_build_export_command_uses_project_root() -> None:
    command = build_command(
        {
            "stage": "export",
            "workdir": "work-test",
            "format": "srt",
            "source": "auto",
            "media_path": "input.mp3",
            "export_mode": "custom",
            "export_path": "out",
        }
    )

    assert command[:5] == [python_runtime(), str(ROOT / "main.py"), "export", "--workdir", "work-test"]
    assert command[command.index("--format") + 1] == "srt"
    assert command[command.index("--source") + 1] == "auto"
    assert command[command.index("--export-mode") + 1] == "custom"
    assert command[command.index("--export-path") + 1] == "out"
    assert command[command.index("--media-path") + 1] == "input.mp3"


def test_webui_build_prepare_command_uses_segment_defaults() -> None:
    command = build_command({"stage": "prepare", "workdir": "work-test", "media_path": "input.mp3"})

    assert "--max-segment-seconds" in command
    assert command[command.index("--max-segment-seconds") + 1] == str(DEFAULT_MAX_SEGMENT_SECONDS)
    assert "--min-segment-seconds" in command
    assert command[command.index("--min-segment-seconds") + 1] == str(DEFAULT_MIN_SEGMENT_SECONDS)
    assert command[command.index("--denoise-backend") + 1] == "mdx_net"
    assert command[command.index("--vad-backend") + 1] == "pyannote_onnx_v3"
    assert command[command.index("--pyannote-onnx-model") + 1] == "segmentation-3.0"


def test_webui_build_mimo_proofread_command() -> None:
    command = build_command(
        {
            "stage": "mimo-proofread",
            "workdir": "work-test",
            "llm_model": "mimo-v2.5",
            "llm_base_url": "https://api.xiaomimimo.com/v1",
            "llm_api_key": "sk-test",
            "llm_timeout": 240,
            "disable_thinking": True,
            "mimo_proofread_mode": "two-stage-nearby",
            "mimo_proofread_workers": 2,
            "mimo_nearby_batch_size": 5,
            "mimo_nearby_batch_max_gap_s": 20,
            "mimo_nearby_padding_s": 1.5,
            "mimo_nearby_context_subtitles": 1,
            "mimo_nearby_audio_workers": 1,
            "mimo_proofread_max_tokens": 4096,
            "mimo_compact_output": False,
            "mimo_proofread_output_dir": "out-mimo",
            "glossary_xlsx": "glossary.xlsx",
        }
    )

    assert command[:3] == [python_runtime(), str(ROOT / "main.py"), "mimo-proofread"]
    assert command[command.index("--mimo-proofread-mode") + 1] == "two-stage-nearby"
    assert command[command.index("--mimo-nearby-batch-size") + 1] == "5"
    assert command[command.index("--mimo-nearby-batch-max-gap-s") + 1] == "20.0"
    assert "--llm-api-key" not in command


def test_webui_build_run_command_uses_local_model_cache_by_default() -> None:
    command = build_command(
        {
            "stage": "run",
            "workdir": "work-test",
            "media_path": "input.mp3",
            "asr_model": "asr",
            "align_model": "align",
            "dtype": "fp16",
            "device": "cuda:0",
            "format": "srt",
            "source": "auto",
            "normalize_source": "auto",
        }
    )

    assert "--local-files-only" in command
    assert "--no-local-files-only" not in command
    assert command[command.index("--model-cache-dir") + 1] == str(DEFAULT_MODEL_CACHE_DIR)
    assert command[command.index("--align-cleanup-interval") + 1] == "4"
    assert command[command.index("--split-mode") + 1] == "token-counts"


def test_webui_build_run_includes_integrated_mimo_proofread_mode() -> None:
    command = build_command(
        {
            "stage": "run",
            "workdir": "work-test",
            "media_path": "input.mp3",
            "asr_model": "asr",
            "align_model": "align",
            "dtype": "fp16",
            "device": "cuda:0",
            "format": "srt",
            "source": "auto",
            "normalize_source": "auto",
            "with_mimo_proofread": True,
        }
    )
    assert "--with-mimo-proofread" in command


def test_webui_build_align_command_uses_cleanup_interval_default() -> None:
    command = build_command(
        {
            "stage": "align",
            "workdir": "work-test",
            "align_model": "align",
            "dtype": "fp16",
            "device": "cuda",
        }
    )

    assert command[command.index("--cleanup-interval") + 1] == "4"
    assert command[command.index("--model-cache-dir") + 1] == str(DEFAULT_MODEL_CACHE_DIR)


def test_webui_build_transcribe_command_uses_local_model_cache_by_default() -> None:
    command = build_command(
        {
            "stage": "transcribe",
            "workdir": "work-test",
            "asr_model": "asr",
            "dtype": "fp16",
            "device": "cuda",
        }
    )

    assert command[command.index("--model-cache-dir") + 1] == str(DEFAULT_MODEL_CACHE_DIR)
    assert "--local-files-only" in command
    assert "--no-local-files-only" not in command


def test_webui_build_batch_run_command_uses_media_list() -> None:
    command = build_command(
        {
            "stage": "batch-run",
            "workdir": "batch-work",
            "asr_model": "asr",
            "align_model": "align",
            "dtype": "fp16",
            "device": "cuda:0",
            "format": "srt",
            "source": "auto",
            "normalize_source": "auto",
            "media_paths": ["a.mp3", "b.mp4"],
        }
    )

    assert command[:5] == [python_runtime(), str(ROOT / "main.py"), "batch-run", "--workdir", "batch-work"]
    assert command[-2:] == ["a.mp3", "b.mp4"]
    assert command[command.index("--model-cache-dir") + 1] == str(DEFAULT_MODEL_CACHE_DIR)


def test_webui_build_batch_run_command_supports_manifest() -> None:
    command = build_command(
        {
            "stage": "batch-run",
            "workdir": "batch-work",
            "asr_model": "asr",
            "align_model": "align",
            "dtype": "fp16",
            "device": "cuda:0",
            "format": "srt",
            "source": "auto",
            "normalize_source": "auto",
            "batch_manifest": "tasks.jsonl",
        }
    )

    assert command[command.index("--manifest") + 1] == "tasks.jsonl"


def test_webui_python_runtime_prefers_project_venv() -> None:
    expected = ROOT / ".venv312" / "Scripts" / "python.exe"
    if expected.exists():
        assert python_runtime() == str(expected)
    else:
        assert python_runtime() == sys.executable


def test_default_optimizer_root_points_to_project_optimizer() -> None:
    assert DEFAULT_OPTIMIZER_ROOT == ROOT / "optimizer"
    assert DEFAULT_OPTIMIZER_ROOT.exists()


def test_webapp_root_reexports_dynamic_active_job() -> None:
    original = server.ACTIVE_JOB
    try:
        server.ACTIVE_JOB = {"status": "running"}
        assert webapp.ACTIVE_JOB == {"status": "running"}
    finally:
        server.ACTIVE_JOB = original


def test_pick_media_file_runs_dialog_in_child_process(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "input.mp3"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=f"{selected}\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server.pick_media_file() == {"cancelled": False, "path": str(selected.resolve())}


def test_pick_media_file_reports_cancelled(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server.pick_media_file() == {"cancelled": True}


def test_pick_media_files_runs_dialog_in_child_process(monkeypatch, tmp_path: Path) -> None:
    selected_a = tmp_path / "a.mp3"
    selected_b = tmp_path / "b.mp4"

    def fake_run(*args, **kwargs):
        stdout = f"{selected_a}\n{selected_b}\n"
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server.pick_media_files() == {
        "cancelled": False,
        "paths": [str(selected_a.resolve()), str(selected_b.resolve())],
    }


def test_pick_media_files_reports_cancelled(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server.pick_media_files() == {"cancelled": True, "paths": []}


def test_pick_batch_manifest_runs_dialog_in_child_process(monkeypatch, tmp_path: Path) -> None:
    selected = tmp_path / "tasks.jsonl"

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=f"{selected}\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server.pick_batch_manifest() == {"cancelled": False, "path": str(selected.resolve())}


def test_pick_batch_manifest_reports_cancelled(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="\n", stderr="")

    monkeypatch.setattr(server.subprocess, "run", fake_run)

    assert server.pick_batch_manifest() == {"cancelled": True}


def test_webui_status_reads_batch_summary(tmp_path: Path) -> None:
    summary_dir = tmp_path / "summary"
    summary_dir.mkdir(parents=True)
    (summary_dir / "batch-summary.json").write_text(
        '{"total":2,"succeeded":1,"failed":1,"tasks":[{"media":"a.mp3","workdir":"w1","status":"failed","error":"boom"}]}',
        encoding="utf-8",
    )
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "batch-run.log").write_text("line1\nline2\n", encoding="utf-8")

    status = get_status(str(tmp_path))

    assert status["batch_summary"]["total"] == 2
    assert "batch-run" in status["logs"]


def test_webui_normalize_asr_language_is_case_insensitive() -> None:
    assert normalize_asr_language("japanese") == "Japanese"
    assert normalize_asr_language("") == ""


def test_webui_index_html_loads_from_template_file() -> None:
    assert INDEX_TEMPLATE_PATH.exists()
    assert load_index_html() == INDEX_HTML
    assert "<!doctype html>" in INDEX_HTML


def test_webui_index_keeps_payload_control_ids() -> None:
    required_ids = [
        "mediaPath",
        "mediaPaths",
        "batchManifestPath",
        "workdir",
        "denoise",
        "denoiseBackend",
        "denoiseLevel",
        "denoiseProfile",
        "mdxModel",
        "mdxModelDir",
        "vadBackend",
        "pyannoteOnnxModel",
        "vadOnset",
        "vadOffset",
        "vadThreshold",
        "vadMinSpeechMs",
        "vadMinSilenceMs",
        "vadSpeechPadMs",
        "maxSegmentSeconds",
        "minSegmentSeconds",
        "preferredSilenceMs",
        "minSilenceMs",
        "paddingMs",
        "overlapMs",
        "asrModel",
        "asrLanguage",
        "alignModel",
        "dtype",
        "device",
        "localFilesOnly",
        "llmModel",
        "threadNum",
        "splitMode",
        "splitMaxWordCountCjk",
        "splitMaxWordCountEnglish",
        "splitPromptLimitRatio",
        "proofreadKind",
        "correctBatchNum",
        "translateBatchNum",
        "llmTimeout",
        "disableThinking",
        "llmExtraBodyJson",
        "llmBaseUrl",
        "llmApiKey",
        "mimoProofreadMode",
        "mimoProofreadWorkers",
        "mimoNearbyBatchSize",
        "mimoNearbyBatchMaxGapS",
        "mimoNearbyPaddingS",
        "mimoNearbyContextSubtitles",
        "mimoNearbyAudioWorkers",
        "mimoProofreadMaxTokens",
        "mimoCompactOutput",
        "mimoProofreadOutputDir",
        "targetLanguage",
        "normalizeSource",
        "normalizeExtendMs",
        "normalizeSnapGapMs",
        "normalizeMinBlankMs",
        "translateCustomPrompt",
        "glossaryXlsx",
        "format",
        "source",
        "exportMode",
        "exportPath",
        "withAlign",
        "withCorrect",
        "withNormalize",
        "withSplit",
        "withTranslate",
    ]

    for control_id in required_ids:
        assert f'id="{control_id}"' in INDEX_HTML or f"id='{control_id}'" in INDEX_HTML

    assert 'id="normalizeGlossaryButton"' in INDEX_HTML
    assert 'id="correctButton"' not in INDEX_HTML
    assert 'id="mimoProofreadButton"' not in INDEX_HTML


def test_webui_index_has_workflow_validation_and_log_tabs() -> None:
    assert "function validateBeforeStart" in INDEX_HTML
    assert "普通校对" in INDEX_HTML
    assert "MiMo + 音频校对" in INDEX_HTML
    assert "mimo-v2.5" in INDEX_HTML
    assert "https://api.xiaomimimo.com/v1" in INDEX_HTML
    assert "proofreadKind" in INDEX_HTML
    assert "startStage('correct')" not in INDEX_HTML
    assert "function renderPipeline" in INDEX_HTML
    assert "function renderLogs" in INDEX_HTML
    assert "logTabs" in INDEX_HTML
    assert "api/import-media" not in INDEX_HTML
    assert "mediaFile" not in INDEX_HTML
    assert "function pickMedia" in INDEX_HTML
    assert "function pickMediaList" in INDEX_HTML
    assert "function pickBatchManifest" in INDEX_HTML
    assert "function renderBatchSummary" in INDEX_HTML
    assert "function batchManifestPath" in INDEX_HTML
    assert "function parsedMediaPaths" in INDEX_HTML
    assert "function effectiveStage" in INDEX_HTML
    assert "/api/pick-media-list" in INDEX_HTML
    assert "/api/pick-batch-manifest" in INDEX_HTML
    assert "batch-run" in INDEX_HTML
    assert "mimo-proofread" in INDEX_HTML
    assert "function normalizeGlossary" in INDEX_HTML
    assert "/api/glossary-normalize" in INDEX_HTML
    assert "function suggestWorkdir" in INDEX_HTML
    assert "function deleteCurrentWorkspace" in INDEX_HTML
    assert "function deleteAllWorkspaces" in INDEX_HTML
    assert "Object.prototype.hasOwnProperty.call(logs, active)" in INDEX_HTML
    assert "/api/delete-workspaces" in INDEX_HTML


def test_webui_index_has_workbench_structure() -> None:
    required_nodes = [
        "configShell",
        "configRail",
        "configDrawer",
        "workspace",
        "jobSummary",
        "pipelinePanel",
        "artifactsPanel",
        "logsPanel",
        "pipelineTimeline",
    ]

    for node_id in required_nodes:
        assert f'id="{node_id}"' in INDEX_HTML or f"id='{node_id}'" in INDEX_HTML


def test_webui_index_has_config_drawer_state_functions() -> None:
    required_functions = [
        "function toggleConfig",
        "function applyConfigState",
        "function openConfigGroup",
        "function renderJobSummary",
        "function renderPipeline",
    ]

    for function_name in required_functions:
        assert function_name in INDEX_HTML

    assert "qwen3_asr_webui_ui_state_v1" in INDEX_HTML


def test_webui_index_does_not_ship_machine_specific_defaults() -> None:
    assert 'value="E:\\\\project' not in INDEX_HTML
    assert 'placeholder="E:\\\\project' not in INDEX_HTML
    assert "qwen3_asr_webui_settings_v2" not in INDEX_HTML
    assert "qwen3_asr_webui_settings_v3" in INDEX_HTML


def test_webui_suggest_workdir_uses_workspace_numbering(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.web.commands.WORKSPACES_DIR", tmp_path)
    (tmp_path / "0001-old").mkdir()

    suggested = suggest_workdir(r"D:\media\demo.video.mp4")

    assert suggested == tmp_path / "0002-demo.video"


def test_webui_resolve_deletable_workspace_rejects_outside_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.web.commands.WORKSPACES_DIR", tmp_path)
    allowed = tmp_path / "0001-demo"
    allowed.mkdir()

    assert resolve_deletable_workspace(str(allowed)) == allowed.resolve()
    try:
        resolve_deletable_workspace(str(tmp_path.parent / "backups"))
    except ValueError as exc:
        assert "workspaces" in str(exc)
    else:
        raise AssertionError("outside paths must be rejected")


def test_webui_resolve_deletable_workspaces_lists_all_projects(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.web.commands.WORKSPACES_DIR", tmp_path)
    allowed_a = tmp_path / "0001-demo"
    allowed_b = tmp_path / "0002-demo"
    ignored = tmp_path / "manual"
    allowed_a.mkdir()
    allowed_b.mkdir()
    ignored.mkdir()

    assert resolve_deletable_workspaces() == [allowed_a.resolve(), allowed_b.resolve()]
    assert resolve_deletable_workspaces([str(allowed_a), str(allowed_a), str(allowed_b)]) == [
        allowed_a.resolve(),
        allowed_b.resolve(),
    ]

    try:
        resolve_deletable_workspaces([str(tmp_path.parent / "backups")])
    except ValueError as exc:
        assert "workspaces" in str(exc)
    else:
        raise AssertionError("outside paths must be rejected")


def test_webui_delete_workspaces_deletes_all_deletable_projects(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.web.commands.WORKSPACES_DIR", tmp_path)
    monkeypatch.setattr("qwen_asr.web.server.WORKSPACES_DIR", tmp_path, raising=False)
    allowed_a = tmp_path / "0001-demo"
    allowed_b = tmp_path / "0002-demo"
    ignored = tmp_path / "manual"
    allowed_a.mkdir()
    allowed_b.mkdir()
    ignored.mkdir()

    result = server.delete_workspaces()

    assert result["count"] == 2
    assert not allowed_a.exists()
    assert not allowed_b.exists()
    assert ignored.exists()


def test_webui_delete_workspaces_rejects_non_list_selection() -> None:
    try:
        server.delete_workspaces("not-a-list")
    except RuntimeError as exc:
        assert "workdirs must be a list" in str(exc)
    else:
        raise AssertionError("non-list workdirs should be rejected")


def test_webui_normalize_glossary_xlsx_returns_output_and_count(tmp_path: Path) -> None:
    source = tmp_path / "glossary.xlsx"
    _write_single_sheet_xlsx(
        source,
        "Input",
        [["group", "source", "target", "note"], ["Terms", "A", "B", ""]],
    )

    result = server.normalize_glossary_xlsx(str(source))

    assert result["count"] == 1
    assert Path(result["output"]).exists()
    assert Path(result["output"]).name == "glossary.normalized.xlsx"


def test_webui_normalize_glossary_xlsx_rejects_empty_path() -> None:
    try:
        server.normalize_glossary_xlsx("")
    except RuntimeError as exc:
        assert "path is required" in str(exc)
    else:
        raise AssertionError("empty glossary path should be rejected")
