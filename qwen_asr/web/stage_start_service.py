from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from qwen_asr.artifact_state import ArtifactState
from qwen_asr.credentials import resolve_llm_api_key, resolve_mimo_api_key
from qwen_asr.defaults import (
    DEFAULT_ALIGN_MODEL,
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json
from qwen_asr.web.commands import SENSITIVE_PAYLOAD_FIELDS

STARTABLE_STAGES = frozenset(
    {
        "prepare",
        "transcribe",
        "correct",
        "align",
        "split",
        "translate",
        "mimo-proofread",
        "quality-gate",
        "normalize",
        "export",
    }
)


class StageStartError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def stage_start_capability(stage: str, missing_inputs: list[str]) -> dict[str, Any]:
    if stage not in STARTABLE_STAGES:
        return {"runnable": False, "reason": "managed_by_pipeline"}
    if stage != "prepare" and missing_inputs:
        return {"runnable": False, "reason": "missing_inputs"}
    return {"runnable": True, "reason": None}


def build_workspace_stage_payload(
    work_paths: WorkPaths,
    *,
    stage: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name = str(stage or "").strip()
    if name not in STARTABLE_STAGES:
        raise StageStartError("STAGE_NOT_STARTABLE", "stage cannot be started independently", status=409)
    safe_settings = _safe_settings(settings)
    state = ArtifactState(work_paths)
    missing_inputs = state.missing_inputs(name)
    if name != "prepare" and missing_inputs:
        raise StageStartError(
            "STAGE_INPUTS_MISSING",
            f"stage inputs are missing: {', '.join(missing_inputs)}",
            status=409,
        )
    metadata = _metadata(work_paths)
    payload = _base_payload(work_paths, safe_settings, metadata)
    payload["stage"] = name
    _validate_stage_environment(name, payload)
    return payload


def _safe_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    if settings is None:
        return {}
    if not isinstance(settings, dict):
        raise StageStartError("STAGE_SETTINGS_INVALID", "settings must be an object")
    for key in settings:
        normalized = _snake_case(str(key))
        if normalized in SENSITIVE_PAYLOAD_FIELDS or re.search(
            r"(?:api_?key|authorization|access_?token|auth_?token|secret)$",
            normalized,
            flags=re.IGNORECASE,
        ):
            raise StageStartError(
                "STAGE_SETTINGS_CONTAIN_CREDENTIALS",
                "API credentials must be configured through process environment variables.",
            )
    return settings


def _base_payload(
    work_paths: WorkPaths,
    settings: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    def value(key: str, default: Any = None) -> Any:
        return settings.get(key, default)

    media_path = str(value("mediaPath", metadata.get("original_media_path", "")) or "").strip()
    export_mode = str(value("exportMode", metadata.get("export_mode", "source")) or "source")
    export_path = str(value("exportPath", metadata.get("custom_export_path", "")) or "").strip()
    return {
        "workdir": str(work_paths.workdir),
        "resume": True,
        "force": False,
        "media_path": media_path,
        "denoise": bool(value("denoise", False)),
        "denoise_backend": value("denoiseBackend", "mdx_net"),
        "denoise_level": value("denoiseLevel", 12.0),
        "denoise_profile": value("denoiseProfile", "strong"),
        "mdx_model": value("mdxModel", "UVR-MDX-NET-Inst_HQ_3.onnx"),
        "mdx_model_dir": value("mdxModelDir", ""),
        "vad_backend": value("vadBackend", "pyannote_onnx_v3"),
        "pyannote_onnx_model": value("pyannoteOnnxModel", "segmentation-3.0"),
        "vad_onset": value("vadOnset", 0.5),
        "vad_offset": value("vadOffset", 0.35),
        "vad_threshold": value("vadThreshold", 0.5),
        "vad_min_speech_ms": value("vadMinSpeechMs", 180),
        "vad_min_silence_ms": value("vadMinSilenceMs", 250),
        "vad_speech_pad_ms": value("vadSpeechPadMs", 120),
        "max_segment_seconds": value("maxSegmentSeconds", DEFAULT_MAX_SEGMENT_SECONDS),
        "min_segment_seconds": value("minSegmentSeconds", DEFAULT_MIN_SEGMENT_SECONDS),
        "preferred_silence_ms": value("preferredSilenceMs", 800),
        "min_silence_ms": value("minSilenceMs", 500),
        "padding_ms": value("paddingMs", 300),
        "overlap_ms": value("overlapMs", 0),
        "asr_model": value("asrModel", DEFAULT_ASR_MODEL),
        "asr_language": value("asrLanguage", ""),
        "align_model": value("alignModel", DEFAULT_ALIGN_MODEL),
        "align_cleanup_interval": value("alignCleanupInterval", 4),
        "align_diagnostics_mode": value("alignDiagnosticsMode", "off"),
        "align_fallback": value("alignFallback", "off"),
        "align_fallback_window_seconds": value("alignFallbackWindowSeconds", 3.0),
        "model_cache_dir": value("modelCacheDir", str(DEFAULT_MODEL_CACHE_DIR)),
        "dtype": value("dtype", "fp16"),
        "device": value("device", "cuda:0"),
        "local_files_only": bool(value("localFilesOnly", True)),
        "batch_mode": value("batchMode", "adaptive"),
        "asr_batch_size": _optional(value("asrBatchSize")),
        "single_long_segment_threshold": _optional(value("singleLongSegmentThreshold")),
        "target_batch_audio_seconds": _optional(value("targetBatchAudioSeconds")),
        "profile_batches": bool(value("profileBatches", False)),
        "llm_model": value("llmModel", DEFAULT_LLM_MODEL),
        "llm_base_url": value("llmBaseUrl", DEFAULT_LLM_BASE_URL),
        "thread_num": value("threadNum", DEFAULT_LLM_CONCURRENCY),
        "split_max_word_count_cjk": value("splitMaxWordCountCjk", 18),
        "split_max_word_count_english": value("splitMaxWordCountEnglish", 18),
        "correct_batch_num": value("correctBatchNum", 8),
        "translate_batch_num": value("translateBatchNum", 20),
        "llm_timeout": value("llmTimeout", DEFAULT_LLM_TIMEOUT),
        "disable_thinking": bool(value("disableThinking", True)),
        "llm_extra_body_json": value("llmExtraBodyJson", DEFAULT_LLM_EXTRA_BODY_JSON),
        "target_language": value("targetLanguage", "简体中文"),
        "translate_custom_prompt": value("translateCustomPrompt", ""),
        "glossary_xlsx": value("glossaryXlsx", ""),
        "mimo_proofread_mode": value("mimoProofreadMode", "segment-audio"),
        "mimo_audio_review_scope": value("mimoAudioReviewScope", "suspects"),
        "mimo_proofread_workers": value("mimoProofreadWorkers", DEFAULT_LLM_CONCURRENCY),
        "mimo_nearby_batch_size": value("mimoNearbyBatchSize", 1),
        "mimo_nearby_batch_max_gap_s": value("mimoNearbyBatchMaxGapS", 8.0),
        "mimo_nearby_padding_s": value("mimoNearbyPaddingS", 1.5),
        "mimo_nearby_context_subtitles": value("mimoNearbyContextSubtitles", 1),
        "mimo_nearby_audio_workers": value("mimoNearbyAudioWorkers", DEFAULT_LLM_CONCURRENCY),
        "mimo_proofread_max_tokens": value("mimoProofreadMaxTokens", 4096),
        "mimo_compact_output": bool(value("mimoCompactOutput", False)),
        "mimo_proofread_output_dir": value("mimoProofreadOutputDir", ""),
        "quality_gate_include_export": bool(value("qualityGateIncludeExport", False)),
        "quality_gate_require_srt": bool(value("qualityGateRequireSrt", False)),
        "normalize_source": value("normalizeSource", "auto"),
        "normalize_extend_ms": value("normalizeExtendMs", 350),
        "normalize_snap_gap_ms": value("normalizeSnapGapMs", 200),
        "normalize_min_blank_ms": value("normalizeMinBlankMs", 300),
        "format": value("format", "both"),
        "source": value("source", "auto"),
        "export_mode": export_mode,
        "export_path": export_path,
    }


def _validate_stage_environment(stage: str, payload: dict[str, Any]) -> None:
    if stage == "prepare":
        media_path = Path(str(payload.get("media_path", "")))
        if not str(media_path) or not media_path.exists() or not media_path.is_file():
            raise StageStartError(
                "STAGE_MEDIA_NOT_FOUND",
                "prepare requires an existing media path in project metadata or saved Web settings",
                status=409,
            )
    if stage in {"translate", "correct"} and not resolve_llm_api_key(
        None,
        str(payload.get("llm_base_url", "")),
    ):
        raise StageStartError(
            "STAGE_CREDENTIAL_MISSING",
            "the provider API credential is not configured in the Web service environment",
            status=409,
        )
    if stage == "mimo-proofread" and not resolve_mimo_api_key():
        raise StageStartError(
            "STAGE_CREDENTIAL_MISSING",
            "MIMO_API_KEY is not configured in the Web service environment",
            status=409,
        )


def _metadata(work_paths: WorkPaths) -> dict[str, Any]:
    try:
        payload = read_json(work_paths.project_metadata, default={})
    except (OSError, UnicodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _optional(value: Any) -> Any:
    return None if value is None or value == "" else value
