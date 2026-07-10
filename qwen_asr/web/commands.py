from __future__ import annotations

import sys
import re
from pathlib import Path

from qwen_asr.defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
)

HOST = "127.0.0.1"
PORT = 8765
ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_DIR = ROOT / "workspaces"
PROJECT_PYTHON = ROOT / ".venv312" / "Scripts" / "python.exe"
DEFAULT_ASR_BATCH_SIZE = 5
SUPPORTED_ASR_LANGUAGES = (
    "",
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)

def normalize_asr_language(value: object) -> str:
    language = str(value or "").strip()
    if not language:
        return ""
    supported = {item.lower(): item for item in SUPPORTED_ASR_LANGUAGES if item}
    normalized = supported.get(language.lower())
    if normalized is None:
        raise ValueError(
            "Unsupported ASR language: "
            f"{language}. Choose one of: {', '.join(item for item in SUPPORTED_ASR_LANGUAGES if item)}"
        )
    return normalized

def _prepare_audio_args(payload: dict) -> list[str]:
    args: list[str] = [
        "--denoise-backend",
        str(payload.get("denoise_backend", "mdx_net") or "mdx_net"),
        "--denoise-level",
        str(max(0.0, float(payload.get("denoise_level", 12.0)))),
        "--denoise-profile",
        str(payload.get("denoise_profile", "strong") or "strong"),
        "--mdx-model",
        str(payload.get("mdx_model", "UVR-MDX-NET-Inst_HQ_3.onnx") or "UVR-MDX-NET-Inst_HQ_3.onnx"),
        "--vad-backend",
        str(payload.get("vad_backend", "pyannote_onnx_v3") or "pyannote_onnx_v3"),
        "--vad-threshold",
        str(max(0.0, min(1.0, float(payload.get("vad_threshold", 0.5))))),
        "--vad-onset",
        str(max(0.0, min(1.0, float(payload.get("vad_onset", 0.5))))),
        "--vad-offset",
        str(max(0.0, min(1.0, float(payload.get("vad_offset", 0.35))))),
        "--vad-min-speech-ms",
        str(max(0, int(payload.get("vad_min_speech_ms", 180)))),
        "--vad-min-silence-ms",
        str(max(0, int(payload.get("vad_min_silence_ms", 250)))),
        "--vad-speech-pad-ms",
        str(max(0, int(payload.get("vad_speech_pad_ms", 120)))),
        "--pyannote-onnx-model",
        str(payload.get("pyannote_onnx_model", "segmentation-3.0") or "segmentation-3.0"),
    ]
    mdx_model_dir = str(payload.get("mdx_model_dir", "") or "").strip()
    if mdx_model_dir:
        args.extend(["--mdx-model-dir", mdx_model_dir])
    return args

def build_command(payload: dict) -> list[str]:
    stage = payload["stage"]
    workdir = payload["workdir"]
    thread_num = str(max(1, int(payload.get("thread_num", 4))))
    disable_thinking = bool(payload.get("disable_thinking", True))
    llm_extra_body_json = str(payload.get("llm_extra_body_json", "") or "").strip()
    proofread_kind = str(payload.get("proofread_kind", "normal") or "normal").strip()
    segment_args = [
        "--max-segment-seconds",
        str(max(1.0, float(payload.get("max_segment_seconds", DEFAULT_MAX_SEGMENT_SECONDS)))),
        "--min-segment-seconds",
        str(max(0.0, float(payload.get("min_segment_seconds", DEFAULT_MIN_SEGMENT_SECONDS)))),
        "--preferred-silence-ms",
        str(max(0, int(payload.get("preferred_silence_ms", 800)))),
        "--min-silence-ms",
        str(max(0, int(payload.get("min_silence_ms", 500)))),
        "--padding-ms",
        str(max(0, int(payload.get("padding_ms", 300)))),
        "--overlap-ms",
        str(max(0, int(payload.get("overlap_ms", 0)))),
    ]
    resume = payload.get("resume")
    if resume is None:
        resume = stage == "run"
    command = [python_runtime(), str(ROOT / "main.py"), stage, "--workdir", workdir, "--log-level", "INFO"]
    model_cache_dir = str(payload.get("model_cache_dir", "") or "").strip() or str(DEFAULT_MODEL_CACHE_DIR)
    model_cache_args = ["--model-cache-dir", model_cache_dir]
    media_list = [str(item).strip() for item in payload.get("media_paths", []) if str(item).strip()]

    if payload.get("force"):
        command.append("--force")
    if resume is False:
        command.append("--no-resume")

    if stage == "mimo-proofread":
        command = [python_runtime(), str(ROOT / "main.py"), "mimo-proofread", "--workdir", workdir, "--log-level", "INFO"]
        command.extend(_mimo_proofread_args(payload))
    elif stage == "prepare":
        command.extend(["--media", payload["media_path"]])
        command.extend(segment_args)
        if payload.get("denoise"):
            command.append("--denoise")
        command.extend(_prepare_audio_args(payload))
    elif stage == "transcribe":
        command.extend(["--model", payload["asr_model"], "--dtype", payload["dtype"], "--device", payload["device"]])
        command.extend(model_cache_args)
        command.extend(["--batch-mode", str(payload.get("batch_mode", "adaptive"))])
        if payload.get("asr_batch_size") is not None:
            command.extend(["--batch-size", str(max(1, int(payload["asr_batch_size"])))])
        if payload.get("single_long_segment_threshold") is not None:
            command.extend(["--single-long-segment-threshold", str(max(1.0, float(payload["single_long_segment_threshold"])))])
        if payload.get("target_batch_audio_seconds") is not None:
            command.extend(["--target-batch-audio-seconds", str(max(1.0, float(payload["target_batch_audio_seconds"])))])
        if payload.get("profile_batches"):
            command.append("--profile-batches")
        command.append("--local-files-only" if payload.get("local_files_only", True) else "--no-local-files-only")
        language = normalize_asr_language(payload.get("asr_language", ""))
        if language:
            command.extend(["--language", language])
    elif stage == "align":
        command.extend(
            [
                "--model",
                payload["align_model"],
                *model_cache_args,
                "--dtype",
                payload["dtype"],
                "--device",
                payload["device"],
                "--cleanup-interval",
                str(max(1, int(payload.get("align_cleanup_interval", 4)))),
            ]
        )
        command.append("--local-files-only" if payload.get("local_files_only", True) else "--no-local-files-only")
    elif stage == "normalize":
        command.extend(
            [
                "--source",
                payload["normalize_source"],
                "--extend-ms",
                str(max(0, int(payload.get("normalize_extend_ms", 350)))),
                "--snap-gap-ms",
                str(max(0, int(payload.get("normalize_snap_gap_ms", 200)))),
                "--min-blank-ms",
                str(max(0, int(payload.get("normalize_min_blank_ms", 300)))),
            ]
        )
    elif stage == "split":
        command.extend(
            [
                "--thread-num",
                thread_num,
                "--max-word-count-cjk",
                str(max(1, int(payload.get("split_max_word_count_cjk", 25)))),
                "--max-word-count-english",
                str(max(1, int(payload.get("split_max_word_count_english", 18)))),
                "--prompt-limit-ratio",
                str(max(0.1, float(payload.get("split_prompt_limit_ratio", 0.8)))),
                "--split-mode",
                str(payload.get("split_mode", "token-counts") or "token-counts"),
                "--timeout",
                str(max(1.0, float(payload.get("llm_timeout", 120)))),
            ]
        )
        command.append("--disable-thinking" if disable_thinking else "--no-disable-thinking")
        if llm_extra_body_json:
            command.extend(["--llm-extra-body-json", llm_extra_body_json])
        if payload.get("llm_model") and payload.get("llm_base_url") and payload.get("llm_api_key"):
            command.extend(
                [
                    "--llm-model",
                    payload["llm_model"],
                    "--llm-base-url",
                    payload["llm_base_url"],
                ]
            )
    elif stage == "translate":
        command.extend(
            [
                "--thread-num",
                thread_num,
                "--batch-num",
                str(max(1, int(payload.get("translate_batch_num", 20)))),
                "--custom-prompt",
                str(payload.get("translate_custom_prompt", "")),
                "--timeout",
                str(max(1.0, float(payload.get("llm_timeout", 120)))),
                "--disable-thinking" if disable_thinking else "--no-disable-thinking",
                "--llm-model",
                payload["llm_model"],
                "--llm-base-url",
                payload["llm_base_url"],
                "--llm-api-key",
                payload["llm_api_key"],
                "--target-language",
                payload["target_language"],
            ]
        )
        if llm_extra_body_json:
            command.extend(["--llm-extra-body-json", llm_extra_body_json])
        if str(payload.get("glossary_xlsx", "")).strip():
            command.extend(["--glossary-xlsx", str(payload["glossary_xlsx"]).strip()])
    elif stage == "correct":
        command.extend(
            [
                "--thread-num",
                thread_num,
                "--batch-num",
                str(max(1, int(payload.get("correct_batch_num", 8)))),
                "--timeout",
                str(max(1.0, float(payload.get("llm_timeout", 120)))),
                "--disable-thinking" if disable_thinking else "--no-disable-thinking",
                "--llm-model",
                payload["llm_model"],
                "--llm-base-url",
                payload["llm_base_url"],
                "--llm-api-key",
                payload["llm_api_key"],
            ]
        )
        if llm_extra_body_json:
            command.extend(["--llm-extra-body-json", llm_extra_body_json])
        if str(payload.get("glossary_xlsx", "")).strip():
            command.extend(["--glossary-xlsx", str(payload["glossary_xlsx"]).strip()])
    elif stage == "export":
        command.extend(["--format", payload["format"], "--source", payload["source"]])
        command.extend(["--export-mode", payload.get("export_mode", "source")])
        if str(payload.get("export_path", "")).strip():
            command.extend(["--export-path", str(payload["export_path"]).strip()])
        if str(payload.get("media_path", "")).strip():
            command.extend(["--media-path", str(payload["media_path"]).strip()])
    elif stage == "run":
        command.extend(
            [
                "--media",
                payload["media_path"],
                "--model",
                payload["asr_model"],
                "--align-model",
                payload["align_model"],
                *model_cache_args,
                "--align-cleanup-interval",
                str(max(1, int(payload.get("align_cleanup_interval", 4)))),
                "--dtype",
                payload["dtype"],
                "--device",
                payload["device"],
                "--local-files-only" if payload.get("local_files_only", True) else "--no-local-files-only",
                "--batch-mode",
                str(payload.get("batch_mode", "adaptive")),
                "--format",
                payload["format"],
                "--source",
                payload["source"],
                "--export-mode",
                payload.get("export_mode", "source"),
                "--normalize-source",
                payload["normalize_source"],
                "--extend-ms",
                str(max(0, int(payload.get("normalize_extend_ms", 350)))),
                "--snap-gap-ms",
                str(max(0, int(payload.get("normalize_snap_gap_ms", 200)))),
                "--min-blank-ms",
                str(max(0, int(payload.get("normalize_min_blank_ms", 300)))),
                "--thread-num",
                thread_num,
                "--batch-num",
                str(max(1, int(payload.get("translate_batch_num", 20)))),
                "--correct-batch-num",
                str(max(1, int(payload.get("correct_batch_num", 8)))),
                "--custom-prompt",
                str(payload.get("translate_custom_prompt", "")),
                "--max-word-count-cjk",
                str(max(1, int(payload.get("split_max_word_count_cjk", 25)))),
                "--max-word-count-english",
                str(max(1, int(payload.get("split_max_word_count_english", 18)))),
                "--prompt-limit-ratio",
                str(max(0.1, float(payload.get("split_prompt_limit_ratio", 0.8)))),
                "--split-mode",
                str(payload.get("split_mode", "token-counts") or "token-counts"),
                "--timeout",
                str(max(1.0, float(payload.get("llm_timeout", 120)))),
                "--disable-thinking" if disable_thinking else "--no-disable-thinking",
            ]
        )
        if payload.get("asr_batch_size") is not None:
            command.extend(["--batch-size", str(max(1, int(payload["asr_batch_size"])))])
        if payload.get("single_long_segment_threshold") is not None:
            command.extend(["--single-long-segment-threshold", str(max(1.0, float(payload["single_long_segment_threshold"])))])
        if payload.get("target_batch_audio_seconds") is not None:
            command.extend(["--target-batch-audio-seconds", str(max(1.0, float(payload["target_batch_audio_seconds"])))])
        if payload.get("profile_batches"):
            command.append("--profile-batches")
        command.extend(segment_args)
        if str(payload.get("export_path", "")).strip():
            command.extend(["--export-path", str(payload["export_path"]).strip()])
        if payload.get("denoise"):
            command.append("--denoise")
        command.extend(_prepare_audio_args(payload))
        language = normalize_asr_language(payload.get("asr_language", ""))
        if language:
            command.extend(["--language", language])
        if llm_extra_body_json:
            command.extend(["--llm-extra-body-json", llm_extra_body_json])
        if str(payload.get("glossary_xlsx", "")).strip():
            command.extend(["--glossary-xlsx", str(payload["glossary_xlsx"]).strip()])
        if payload.get("with_align"):
            command.append("--with-align")
        if payload.get("with_correct"):
            command.append("--with-correct")
        if payload.get("with_mimo_proofread"):
            command.append("--with-mimo-proofread")
        if payload.get("with_split"):
            command.append("--with-split")
        if payload.get("with_translate"):
            command.append("--with-translate")
            command.extend(
                [
                    "--llm-model",
                    payload["llm_model"],
                    "--llm-base-url",
                    payload["llm_base_url"],
                    "--target-language",
                    payload["target_language"],
                ]
            )
        if payload.get("with_normalize"):
            command.append("--with-normalize")
    elif stage == "batch-run":
        command = [python_runtime(), str(ROOT / "main.py"), "batch-run", "--workdir", workdir, "--log-level", "INFO"]
        command.extend(model_cache_args)
        command.extend(
            [
                "--align-model",
                payload["align_model"],
                "--model",
                payload["asr_model"],
                "--dtype",
                payload["dtype"],
                "--device",
                payload["device"],
                "--batch-mode",
                str(payload.get("batch_mode", "adaptive")),
                "--format",
                payload["format"],
                "--source",
                payload["source"],
                "--export-mode",
                payload.get("export_mode", "source"),
                "--normalize-source",
                payload["normalize_source"],
                "--thread-num",
                thread_num,
                "--batch-num",
                str(max(1, int(payload.get("translate_batch_num", 20)))),
                "--correct-batch-num",
                str(max(1, int(payload.get("correct_batch_num", 8)))),
                "--max-word-count-cjk",
                str(max(1, int(payload.get("split_max_word_count_cjk", 25)))),
                "--max-word-count-english",
                str(max(1, int(payload.get("split_max_word_count_english", 18)))),
                "--prompt-limit-ratio",
                str(max(0.1, float(payload.get("split_prompt_limit_ratio", 0.8)))),
                "--timeout",
                str(max(1.0, float(payload.get("llm_timeout", 120)))),
                "--split-mode",
                str(payload.get("split_mode", "token-counts") or "token-counts"),
                "--align-cleanup-interval",
                str(max(1, int(payload.get("align_cleanup_interval", 4)))),
            ]
        )
        command.append("--local-files-only" if payload.get("local_files_only", True) else "--no-local-files-only")
        command.append("--disable-thinking" if disable_thinking else "--no-disable-thinking")
        if payload.get("denoise"):
            command.append("--denoise")
        command.extend(_prepare_audio_args(payload))
        if payload.get("with_align"):
            command.append("--with-align")
        if payload.get("with_correct"):
            command.append("--with-correct")
        if payload.get("with_mimo_proofread"):
            command.append("--with-mimo-proofread")
        if payload.get("with_split"):
            command.append("--with-split")
        if payload.get("with_translate"):
            command.append("--with-translate")
        if payload.get("with_normalize"):
            command.append("--with-normalize")
        if payload.get("profile_batches"):
            command.append("--profile-batches")
        if payload.get("asr_batch_size") is not None:
            command.extend(["--batch-size", str(max(1, int(payload["asr_batch_size"])))])
        if payload.get("single_long_segment_threshold") is not None:
            command.extend(["--single-long-segment-threshold", str(max(1.0, float(payload["single_long_segment_threshold"])))])
        if payload.get("target_batch_audio_seconds") is not None:
            command.extend(["--target-batch-audio-seconds", str(max(1.0, float(payload["target_batch_audio_seconds"])))])
        language = normalize_asr_language(payload.get("asr_language", ""))
        if language:
            command.extend(["--language", language])
        if llm_extra_body_json:
            command.extend(["--llm-extra-body-json", llm_extra_body_json])
        if str(payload.get("glossary_xlsx", "")).strip():
            command.extend(["--glossary-xlsx", str(payload["glossary_xlsx"]).strip()])
        if str(payload.get("export_path", "")).strip():
            command.extend(["--export-path", str(payload["export_path"]).strip()])
        if str(payload.get("batch_manifest", "")).strip():
            command.extend(["--manifest", str(payload["batch_manifest"]).strip()])
        if payload.get("with_translate") or payload.get("with_correct"):
            command.extend(
                [
                    "--llm-model",
                    payload["llm_model"],
                    "--llm-base-url",
                    payload["llm_base_url"],
                ]
            )
        if payload.get("with_translate"):
            command.extend(["--target-language", payload["target_language"]])
        command.extend(segment_args)
        command.extend(media_list)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    return command


def _mimo_proofread_args(payload: dict) -> list[str]:
    args = [
        "--mimo-proofread-mode", str(payload.get("mimo_proofread_mode", "segment-audio")),
        "--mimo-proofread-workers", str(max(1, int(payload.get("mimo_proofread_workers", 1)))),
        "--mimo-nearby-batch-size", str(max(1, int(payload.get("mimo_nearby_batch_size", 1)))),
        "--mimo-nearby-batch-max-gap-s", str(max(0.0, float(payload.get("mimo_nearby_batch_max_gap_s", 8.0)))),
        "--mimo-nearby-padding-s", str(max(0.0, float(payload.get("mimo_nearby_padding_s", 1.5)))),
        "--mimo-nearby-context-subtitles", str(max(0, int(payload.get("mimo_nearby_context_subtitles", 1)))),
        "--mimo-nearby-audio-workers", str(max(1, int(payload.get("mimo_nearby_audio_workers", 1)))),
        "--mimo-proofread-max-tokens", str(max(512, int(payload.get("mimo_proofread_max_tokens", 4096)))),
    ]
    if payload.get("mimo_compact_output"):
        args.append("--mimo-compact-output")
    if payload.get("resume") is False:
        args.append("--no-resume")
    if str(payload.get("glossary_xlsx", "")).strip():
        args.extend(["--glossary-xlsx", str(payload["glossary_xlsx"]).strip()])
    return args


def python_runtime() -> str:
    if PROJECT_PYTHON.exists():
        return str(PROJECT_PYTHON)
    return sys.executable


def suggest_workdir(media_path_value: str) -> Path:
    media_path = Path(str(media_path_value or "").strip())
    source_name = _safe_project_name(media_path.stem or "media")
    WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
    next_index = _next_workspace_index()
    return WORKSPACES_DIR / f"{next_index:04d}-{source_name}"


def list_workspaces() -> list[dict[str, object]]:
    if not WORKSPACES_DIR.exists():
        return []
    projects: list[dict[str, object]] = []
    for path in sorted(WORKSPACES_DIR.iterdir()):
        if not path.is_dir() or not _is_workspace_project(path):
            continue
        stat = path.stat()
        projects.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "modified_at": stat.st_mtime,
            }
        )
    return projects


def resolve_deletable_workspace(workdir_value: str) -> Path:
    target = Path(str(workdir_value or "").strip()).resolve()
    root = WORKSPACES_DIR.resolve()
    if target.parent != root:
        raise ValueError("Only first-level project directories under workspaces can be deleted.")
    if not _is_workspace_project(target):
        raise ValueError("Workspace project names must start with a four-digit numeric prefix.")
    return target


def resolve_deletable_workspaces(workdir_values: list[object] | None = None) -> list[Path]:
    if workdir_values is None:
        return [Path(item["path"]) for item in list_workspaces()]
    targets: list[Path] = []
    seen: set[Path] = set()
    for value in workdir_values:
        target = resolve_deletable_workspace(str(value))
        if target not in seen:
            targets.append(target)
            seen.add(target)
    return targets


def _next_workspace_index() -> int:
    existing = []
    if WORKSPACES_DIR.exists():
        for path in WORKSPACES_DIR.iterdir():
            if path.is_dir():
                match = re.match(r"^(\d{4})-", path.name)
                if match:
                    existing.append(int(match.group(1)))
    return (max(existing) + 1) if existing else 1


def _safe_project_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip(".-_")
    return cleaned[:80] or "media"


def _is_workspace_project(path: Path) -> bool:
    return bool(re.match(r"^\d{4}-", path.name))
