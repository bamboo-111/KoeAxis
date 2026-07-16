from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import run_split_stage
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


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
        llm_model=args.llm_model,
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
