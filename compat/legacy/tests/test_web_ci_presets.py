from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.ci.pipeline import discover_presets
from qwen_asr.cli import build_parser
from qwen_asr.models import WorkPaths
from qwen_asr.web.ci_presets import PRESETS_DIR, list_web_presets, preset_run_argv, preset_web_settings
from qwen_asr.web.commands import build_command
from qwen_asr.web.server import job_environment
from qwen_asr.web.stage_start_service import _base_payload
from qwen_asr.web.static_html import INDEX_HTML

# Values the legacy page starts with that differ from the CLI defaults; a loaded preset must override them.
PAGE_DEFAULTS = {
    "denoise": True,
    "mimoNearbyBatchSize": "5",
    "mimoNearbyBatchMaxGapS": "20",
    "mimoProofreadMode": "segment-audio",
    "alignFallback": "off",
    "proofreadKind": "normal",
    "withCorrect": False,
    "withTranslate": False,
}
# Intended differences: CI recovers inside its GPU phase, the Web passes --align-recovery instead;
# device / base URL / workdir are machine settings.
ALLOWED_DIFFS = {"align_recovery", "device", "llm_base_url", "workdir"}


def _legacy_run_payload(settings: dict) -> dict:
    """What the /legacy page sends for Run (mirrors collectPayload in index.html)."""
    payload = _base_payload(
        WorkPaths.from_workdir(Path("ci-preset-workdir")),
        {**settings, "mediaPath": "ci-preset-media.mp3", "exportMode": "custom", "exportPath": "ci-preset.srt"},
        {},
    )
    kind = settings["proofreadKind"]
    payload.update(
        stage="run",
        llm_model=settings["llmModel"],
        llm_base_url="https://example.invalid/v1",
        thread_num=settings["threadNum"],
        target_language=settings["targetLanguage"],
        with_align=settings["withAlign"],
        with_split=settings["withSplit"],
        with_translate=settings["withTranslate"],
        with_normalize=settings["withNormalize"],
        with_correct=settings["withCorrect"] and kind in ("normal", "both"),
        with_mimo_proofread=settings["withCorrect"] and kind in ("mimo-audio", "both"),
        glossary_xlsx=settings["glossaryXlsx"],
        speaker_enroll=settings["speakerEnroll"],
        speakers=settings["speakers"],
        correct_batch_num=settings["correctBatchNum"],
        translate_batch_num=settings["translateBatchNum"],
        llm_timeout=settings["llmTimeout"],
        split_max_word_count_cjk=settings["splitMaxWordCountCjk"],
        split_max_word_count_english=settings["splitMaxWordCountEnglish"],
        align_fallback=settings["alignFallback"],
        align_fallback_window_seconds=settings["alignFallbackWindowSeconds"],
        translate_custom_prompt=settings["translateCustomPrompt"],
        disable_thinking=settings["disableThinking"],
        llm_extra_body_json=settings["llmExtraBodyJson"],
    )
    return payload


@pytest.mark.parametrize("preset_id", sorted(discover_presets([PRESETS_DIR])))
def test_loaded_ci_preset_gives_the_ci_run_arguments(preset_id: str) -> None:
    preset = discover_presets([PRESETS_DIR])[preset_id]
    converted = preset_web_settings(preset)
    assert converted["unmapped"] == [] and converted["unmapped_env"] == []

    ci = vars(build_parser().parse_args(["run", *preset_run_argv(preset)]))
    command = build_command(_legacy_run_payload({**PAGE_DEFAULTS, **converted["settings"]}))
    web = vars(build_parser().parse_args(command[2:]))

    diffs = {key: (ci.get(key), web.get(key)) for key in set(ci) | set(web) if ci.get(key) != web.get(key)}
    assert set(diffs) <= ALLOWED_DIFFS, diffs
    assert web["align_recovery"] == "auto"


def test_togeradi_preset_settings() -> None:
    togeradi = {item["id"]: item for item in list_web_presets()}["togeradi"]
    settings = togeradi["settings"]
    assert togeradi["missing_files"] == []
    assert settings["proofreadKind"] == "both" and settings["withCorrect"] is True
    assert settings["mimoProofreadMode"] == "two-stage-nearby" and settings["mimoNearbyBatchSize"] == "3"
    assert settings["proofreadRealignMfaFallback"] is True and settings["alignRecovery"] is True
    assert settings["programMemoryFile"].endswith("program-memory-togeradi.json")
    assert settings["asrContextFile"].endswith("asr-context-togeradi.txt")


def test_run_passes_mimo_settings_only_with_mimo_proofread() -> None:
    base = _legacy_run_payload({**PAGE_DEFAULTS, **preset_web_settings(discover_presets([PRESETS_DIR])["togeradi"])["settings"]})
    command = build_command(base)
    assert command[command.index("--mimo-nearby-batch-size") + 1] == "3"
    assert command.count("--glossary-xlsx") == 1  # MiMo options must not repeat run's own flags
    without = build_command({**base, "with_mimo_proofread": False})
    assert "--mimo-proofread-mode" not in without and "--with-correct" in without


def test_web_jobs_prefetch_mfa_in_background_like_ci_workers() -> None:
    assert job_environment({})["QWEN_ASR_MFA_BACKGROUND_PREFETCH"] == "1"
    assert job_environment({"QWEN_ASR_MFA_BACKGROUND_PREFETCH": "0"})["QWEN_ASR_MFA_BACKGROUND_PREFETCH"] == "0"


def test_index_has_ci_parity_controls() -> None:
    for snippet in (
        'id="ciPreset"',
        "/api/ci-presets",
        "applyCiPreset()",
        'id="alignFallback"',
        'id="alignFallbackWindowSeconds"',
        'id="performanceProfile"',
        'id="mimoAudioReviewScope"',
        '<option value="both">',
        "['normal', 'both'].includes",
        "['mimo-audio', 'both'].includes",
        "'alignFallback', 'alignFallbackWindowSeconds', 'performanceProfile'",
        "align_fallback: document.getElementById('alignFallback').value",
        "performance_profile: document.getElementById('performanceProfile').checked",
        "mimo_audio_review_scope: document.getElementById('mimoAudioReviewScope').value",
    ):
        assert snippet in INDEX_HTML, snippet


def test_every_preset_setting_is_a_saved_page_control() -> None:
    import re

    fields_block = INDEX_HTML.split("function settingFields()", 1)[1].split("];", 1)[0]
    saved = set(re.findall(r"'([A-Za-z]+)'", fields_block))
    for preset in list_web_presets():
        for field in preset["settings"]:
            assert f'id="{field}"' in INDEX_HTML, (preset["id"], field)
            assert field in saved, (preset["id"], field)
            assert f"getElementById('{field}')" in INDEX_HTML, (preset["id"], field)
