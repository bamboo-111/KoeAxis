from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qwen_asr.defaults import DEFAULT_SPLIT_LLM_MODEL
from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import run_split_stage
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


def resolve_split_llm_model(args: argparse.Namespace) -> str | None:
    """Explicit --split-llm-model wins; MiMo pipelines default the merge step to Flash; others reuse --llm-model."""
    explicit = getattr(args, "split_llm_model", None)
    if explicit:
        return str(explicit)
    llm_model = getattr(args, "llm_model", None)
    if llm_model and str(llm_model).lower().startswith("mimo-"):
        return DEFAULT_SPLIT_LLM_MODEL
    return llm_model


def speaker_stage_options(args: argparse.Namespace) -> dict:
    from qwen_asr.speaker.voiceprint import parse_speakers

    return {
        "speaker_enroll": getattr(args, "speaker_enroll", None) or None,
        "speakers": parse_speakers(getattr(args, "speakers", None)),
        "speaker_min_margin": float(getattr(args, "speaker_min_margin", 0.1)),
        "speaker_weights": getattr(args, "speaker_model_path", None) or None,
        "speaker_device": getattr(args, "speaker_device", "auto") or "auto",
    }


def cmd_split(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    if args.force:
        stage_helpers._clear_split_outputs(work_paths)
    if args.resume and not args.force and work_paths.split_manifest.exists():
        LOGGER.info("split_segments.json already exists. Skipping split stage.")
        count = stage_helpers._read_json_dict_count(work_paths.split_manifest)
        write_progress(
            work_paths,
            stage="split",
            status="skipped",
            done=count,
            total=count,
            summary="split_segments.json already exists",
        )
        return 0
    run_split_stage(
        work_paths=work_paths,
        optimizer_root=Path(args.optimizer_root),
        llm_model=resolve_split_llm_model(args),
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        thread_num=args.thread_num,
        max_word_count_cjk=args.max_word_count_cjk,
        max_word_count_english=args.max_word_count_english,
        prompt_limit_ratio=args.prompt_limit_ratio,
        disable_thinking=args.disable_thinking,
        llm_extra_body=stage_helpers._parse_json_object_argument("llm_extra_body_json", args.llm_extra_body_json),
        timeout=args.timeout,
        split_mode=getattr(args, "split_mode", "rule"),
        split_boundaries=getattr(args, "split_boundaries", "bunsetsu"),
        **speaker_stage_options(args),
    )
    count = stage_helpers._read_json_dict_count(work_paths.split_manifest)
    write_progress(
        work_paths,
        stage="split",
        status="running",
        done=count,
        total=count,
        summary=f"generated {count} split subtitles",
    )
    return 0
