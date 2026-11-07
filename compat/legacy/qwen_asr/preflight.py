from __future__ import annotations

import argparse
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qwen_asr.defaults import DEFAULT_MODEL_CACHE_DIR
from qwen_asr.models import WorkPaths


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    level: str
    code: str
    message: str


@dataclass(slots=True)
class PreflightResult:
    stage: str
    issues: list[PreflightIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add_error(self, code: str, message: str) -> None:
        self.issues.append(PreflightIssue(level="error", code=code, message=message))

    def add_warning(self, code: str, message: str) -> None:
        self.issues.append(PreflightIssue(level="warning", code=code, message=message))


def run_preflight(args: argparse.Namespace, work_paths: WorkPaths, stage: str) -> PreflightResult:
    result = PreflightResult(stage=stage)
    _check_inputs(args, work_paths, stage, result)
    _check_model_runtime(args, result)
    return result


def format_preflight_messages(result: PreflightResult) -> list[str]:
    return [f"[{issue.level}] {issue.message}" for issue in result.issues]


def ensure_preflight(args: argparse.Namespace, work_paths: WorkPaths, stage: str) -> None:
    if bool(getattr(args, "skip_preflight", False)):
        return
    result = run_preflight(args, work_paths, stage)
    if not result.ok:
        errors = "\n".join(format_preflight_messages(result))
        raise RuntimeError(f"Preflight failed for {stage}:\n{errors}")


def _check_inputs(args: argparse.Namespace, work_paths: WorkPaths, stage: str, result: PreflightResult) -> None:
    if stage in {"prepare", "run", "preflight", "batch-run"}:
        media = getattr(args, "media", None) or getattr(args, "video", None)
        if media:
            media_path = Path(str(media)).resolve()
            if not media_path.exists():
                result.add_error("missing_media", f"Input media does not exist: {media_path}. Provide a valid --media path.")
        elif stage != "batch-run":
            result.add_error("missing_media_arg", "Missing --media. Provide an input media file before starting.")
    if stage == "transcribe" and not work_paths.segments_manifest.exists():
        result.add_error("missing_segments_manifest", "segments.json is missing. Run prepare first, then retry transcribe.")
    if stage == "align" and not work_paths.transcript_manifest.exists():
        result.add_error("missing_transcript_manifest", "transcript_segments.json is missing. Run transcribe first, then retry align.")


def _check_model_runtime(args: argparse.Namespace, result: PreflightResult) -> None:
    if not hasattr(args, "device"):
        return
    device = str(getattr(args, "device", "cuda")).strip().lower()
    dtype = str(getattr(args, "dtype", "fp16")).strip().lower()
    raw_cache_dir = getattr(args, "model_cache_dir", None)
    model_cache_dir = Path(str(raw_cache_dir or DEFAULT_MODEL_CACHE_DIR)).resolve()
    local_files_only = bool(getattr(args, "local_files_only", True))

    if dtype == "fp16" and device == "cpu":
        result.add_error("invalid_dtype_device", "dtype fp16 requires CUDA. Use --device cuda or switch to a supported dtype/device combination.")
    if dtype == "bf16" and device == "cpu":
        result.add_error("invalid_dtype_device", "dtype bf16 on CPU is not supported here. Use --device cuda or choose another dtype.")

    if device != "cpu" and not re.fullmatch(r"cuda(?::\d+)?", device):
        result.add_error("invalid_device", f"Unsupported device value: {device}. Use cpu, cuda, or cuda:N.")

    if device.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            result.add_error("missing_torch", f"torch is required for CUDA preflight: {exc}. Install runtime dependencies first.")
        else:
            if not torch.cuda.is_available():
                result.add_error("cuda_unavailable", "CUDA was requested but is not available. Verify your torch/CUDA install or switch to --device cpu.")

    try:
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        probe = model_cache_dir / f".write-test-{uuid.uuid4().hex}"
        probe.write_text("ok", encoding="ascii")
        probe.unlink(missing_ok=True)
    except OSError:
        result.add_error(
            "cache_not_writable",
            f"Model cache directory is not writable: {model_cache_dir}. Fix permissions or pass --model-cache-dir to a writable location.",
        )
        return

    if local_files_only and not any(model_cache_dir.iterdir()):
        result.add_error(
            "cache_empty",
            "Model cache directory is empty while local_files_only=True. Populate "
            f"{model_cache_dir}, pass --model-cache-dir to an existing cache, or use --no-local-files-only.",
        )

    if bool(getattr(args, "dry_run_check", False)):
        result.add_warning("dry_run_reserved", "dry-run preflight is reserved for lightweight reachability checks and does not warm up models.")
