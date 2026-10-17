from __future__ import annotations

import argparse
import os

from qwen_asr.defaults import DEFAULT_MIMO_CONCURRENCY, DEFAULT_MIMO_MODEL
from qwen_asr.models import WorkPaths


def resolve_mimo_proofread_model(args: argparse.Namespace) -> str:
    """--mimo-model wins, then --llm-model when it is a MiMo model; otherwise the MiMo default.

    The text LLM baseline is no longer MiMo, so a Gemini --llm-model must not leak into the MiMo endpoint.
    """
    explicit = getattr(args, "mimo_model", None)
    if explicit:
        return str(explicit)
    llm_model = getattr(args, "llm_model", None)
    if llm_model and str(llm_model).lower().startswith("mimo-"):
        return str(llm_model)
    return DEFAULT_MIMO_MODEL


def cmd_mimo_proofread(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    """Create an isolated MiMo candidate branch; formal subtitles remain unchanged."""
    from qwen_asr.commands import stages as stage_helpers

    stage_helpers._validate_mimo_diagnostic_scope(args)
    api_key = stage_helpers.resolve_mimo_api_key(getattr(args, "mimo_api_key", None))
    if not api_key:
        raise RuntimeError("MiMo audio proofread requires MIMO_API_KEY")
    command = [
        "--workdir", str(work_paths.workdir),
        "--output-dir", str(work_paths.mimo_proofread_dir),
        "--workers", str(max(1, int(getattr(args, "mimo_proofread_workers", DEFAULT_MIMO_CONCURRENCY)))),
        "--proofread-mode", str(getattr(args, "mimo_proofread_mode", "segment-audio")),
        "--audio-review-scope", str(getattr(args, "mimo_audio_review_scope", "suspects")),
        "--nearby-batch-size", str(max(1, int(getattr(args, "mimo_nearby_batch_size", 1)))),
        "--nearby-batch-max-gap-s", str(max(0.0, float(getattr(args, "mimo_nearby_batch_max_gap_s", 8.0)))),
        "--nearby-padding-s", str(max(0.0, float(getattr(args, "mimo_nearby_padding_s", 1.5)))),
        "--nearby-context-subtitles", str(max(0, int(getattr(args, "mimo_nearby_context_subtitles", 1)))),
        "--nearby-audio-workers",
        str(max(1, int(getattr(args, "mimo_nearby_audio_workers", DEFAULT_MIMO_CONCURRENCY)))),
        "--max-tokens", str(max(512, int(getattr(args, "mimo_proofread_max_tokens", 4096)))),
        "--stage2-apply-threshold", str(min(1.0, max(0.0, float(getattr(args, "mimo_stage2_apply_threshold", 0.9))))),
        "--timeout", str(max(1.0, float(getattr(args, "timeout", 240.0)))),
        "--llm-extra-body-json",
        str(getattr(args, "mimo_llm_extra_body_json", stage_helpers.DEFAULT_LLM_EXTRA_BODY_JSON) or ""),
        "--disable-thinking",
    ]
    if getattr(args, "mimo_diagnostic_all", False):
        command.append("--diagnostic-all")
    if getattr(args, "mimo_compact_output", False):
        command.append("--compact-output")
    command.append("--resume" if getattr(args, "resume", True) else "--no-resume")
    command.extend(["--model", resolve_mimo_proofread_model(args)])
    text_model = getattr(args, "mimo_text_model", None)
    if text_model:
        command.extend(["--text-model", str(text_model)])
    audio_model = getattr(args, "mimo_audio_model", None)
    if audio_model:
        command.extend(["--audio-model", str(audio_model)])
    if getattr(args, "glossary_xlsx", None):
        command.extend(["--glossary-xlsx", str(args.glossary_xlsx)])
    previous_key = os.environ.get("MIMO_API_KEY")
    os.environ["MIMO_API_KEY"] = api_key
    try:
        return stage_helpers.run_mimo_proofread(command)
    finally:
        if previous_key is None:
            os.environ.pop("MIMO_API_KEY", None)
        else:
            os.environ["MIMO_API_KEY"] = previous_key
