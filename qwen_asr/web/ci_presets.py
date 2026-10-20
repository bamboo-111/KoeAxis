"""CI presets (`ci-presets/*.json`) as Web UI run settings.

The preset is expanded exactly like a CI run (`build_run_argv`) and parsed by the real `main.py run`
parser, so every value - including CLI defaults the preset does not mention - is what CI would use.
Parsed options without a Web setting are reported as `unmapped` instead of being dropped silently.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_asr.ci.pipeline import Preset, PresetError, build_run_argv, discover_presets

ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = ROOT / "ci-presets"

# argparse dest -> Web setting id (values copied as they are)
DIRECT_FIELDS = {
    "denoise": "denoise",
    "denoise_backend": "denoiseBackend",
    "denoise_level": "denoiseLevel",
    "denoise_profile": "denoiseProfile",
    "mdx_model": "mdxModel",
    "vad_backend": "vadBackend",
    "pyannote_onnx_model": "pyannoteOnnxModel",
    "vad_onset": "vadOnset",
    "vad_offset": "vadOffset",
    "vad_threshold": "vadThreshold",
    "vad_min_speech_ms": "vadMinSpeechMs",
    "vad_min_silence_ms": "vadMinSilenceMs",
    "vad_speech_pad_ms": "vadSpeechPadMs",
    "max_segment_seconds": "maxSegmentSeconds",
    "min_segment_seconds": "minSegmentSeconds",
    "preferred_silence_ms": "preferredSilenceMs",
    "min_silence_ms": "minSilenceMs",
    "padding_ms": "paddingMs",
    "overlap_ms": "overlapMs",
    "model": "asrModel",
    "align_model": "alignModel",
    "dtype": "dtype",
    "disable_thinking": "disableThinking",
    "llm_extra_body_json": "llmExtraBodyJson",
    "custom_prompt": "translateCustomPrompt",
    "with_align": "withAlign",
    "with_split": "withSplit",
    "with_translate": "withTranslate",
    "with_normalize": "withNormalize",
    "align_fallback": "alignFallback",
    "align_fallback_window_seconds": "alignFallbackWindowSeconds",
    "mimo_proofread_mode": "mimoProofreadMode",
    "mimo_audio_review_scope": "mimoAudioReviewScope",
    "mimo_proofread_workers": "mimoProofreadWorkers",
    "mimo_nearby_batch_size": "mimoNearbyBatchSize",
    "mimo_nearby_batch_max_gap_s": "mimoNearbyBatchMaxGapS",
    "mimo_nearby_padding_s": "mimoNearbyPaddingS",
    "mimo_nearby_context_subtitles": "mimoNearbyContextSubtitles",
    "mimo_nearby_audio_workers": "mimoNearbyAudioWorkers",
    "mimo_proofread_max_tokens": "mimoProofreadMaxTokens",
    "mimo_compact_output": "mimoCompactOutput",
    "with_text_proofread": "withTextProofread",
    "text_proofread_effort": "textProofreadEffort",
    "text_proofread_model": "textProofreadModel",
    "text_proofread_concurrency": "textProofreadConcurrency",
    "text_proofread_timeout": "textProofreadTimeout",
    "text_proofread_apply_threshold": "textProofreadApplyThreshold",
    "text_proofread_program": "textProofreadProgram",
    "text_proofread_hosts": "textProofreadHosts",
    "performance_profile": "performanceProfile",
    "glossary_xlsx": "glossaryXlsx",
    "asr_context_file": "asrContextFile",
    "program_memory_file": "programMemoryFile",
    "episode_title": "episodeTitle",
    "translate_prompt": "translatePrompt",
    "translate_context_lines": "translateContextLines",
    "correct_mode": "correctMode",
    "speaker_enroll": "speakerEnroll",
    "speakers": "speakers",
    "llm_model": "llmModel",
    "thread_num": "threadNum",
    "target_language": "targetLanguage",
    "correct_batch_num": "correctBatchNum",
    "batch_num": "translateBatchNum",
    "max_word_count_cjk": "splitMaxWordCountCjk",
    "max_word_count_english": "splitMaxWordCountEnglish",
    "timeout": "llmTimeout",
    "normalize_source": "normalizeSource",
    "extend_ms": "normalizeExtendMs",
    "snap_gap_ms": "normalizeSnapGapMs",
    "min_blank_ms": "normalizeMinBlankMs",
    "source": "source",
}
# Handled specially below, or per-run / per-machine values a preset must not override.
SPECIAL_DESTS = {
    "with_correct", "with_mimo_proofread", "proofread_realign_mfa_fallback", "language", "align_recovery",
    "program_context",
}
IGNORED_DESTS = {"workdir", "media", "export_mode", "export_path", "format", "command", "func", "device", "mdx_model_dir", "model_cache_dir"}

_DUMMY = {"workdir": Path("ci-preset-workdir"), "media": Path("ci-preset-media.mp3"), "export_path": Path("ci-preset.srt")}


def _parse_run(argv: list[str]) -> argparse.Namespace:
    from qwen_asr.cli import build_parser

    return build_parser().parse_args(["run", *argv])


def preset_run_argv(preset: Preset) -> list[str]:
    """`main.py run` arguments CI would use for this preset (dummy workdir / media / export path)."""
    return build_run_argv(
        preset.public(),
        file_paths=[(item.flag, item.path) for item in preset.files],
        options=dict(preset.options),
        **_DUMMY,
    )


def preset_web_settings(preset: Preset) -> dict[str, Any]:
    args = _parse_run(preset_run_argv(preset))
    defaults = vars(_parse_run(["--workdir", "w", "--media", "m.mp3"]))
    settings: dict[str, Any] = {}
    for dest, field in DIRECT_FIELDS.items():
        value = getattr(args, dest, None)
        settings[field] = "" if value is None else value if isinstance(value, bool) else str(value)
    correct, mimo = bool(args.with_correct), bool(args.with_mimo_proofread)
    settings["withCorrect"] = correct or mimo
    settings["proofreadKind"] = "both" if correct and mimo else "mimo-audio" if mimo else "normal"
    settings["proofreadRealignMfaFallback"] = args.proofread_realign_mfa_fallback == "local"
    settings["asrLanguage"] = str(args.language or "")
    # every --program-context entry (preset files and run_args), in order, one path per line
    settings["programContextFiles"] = "\n".join(str(item) for item in (args.program_context or []))
    settings["alignRecovery"] = True  # CI always runs alignment recovery after align
    unmapped = sorted(
        dest
        for dest, value in vars(args).items()
        if dest not in DIRECT_FIELDS and dest not in SPECIAL_DESTS and dest not in IGNORED_DESTS
        and value != defaults.get(dest)
    )
    env = {key: value for key, value in preset.env.items() if key != "QWEN_ASR_MFA_BACKGROUND_PREFETCH"}
    return {"settings": settings, "unmapped": unmapped, "unmapped_env": sorted(env)}


def list_web_presets(directory: Path = PRESETS_DIR) -> list[dict[str, Any]]:
    try:
        presets = discover_presets([directory])
    except PresetError as exc:
        raise RuntimeError(str(exc)) from exc
    result = []
    for preset in presets.values():
        missing = [str(item.path) for item in preset.files if not item.path.is_file()]
        result.append(
            {"id": preset.id, "title": preset.title, "description": preset.description,
             "missing_files": missing, **preset_web_settings(preset)}
        )
    return sorted(result, key=lambda item: item["id"])
