from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


def cmd_translate(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    if args.force:
        stage_helpers._clear_translate_outputs(work_paths)
    if args.resume and not args.force and stage_helpers._translation_manifest_complete_for_split(work_paths):
        LOGGER.info("translated_segments.json already exists. Skipping translate stage.")
        total = stage_helpers._read_json_dict_count(work_paths.split_manifest)
        done = stage_helpers._count_translated_items(work_paths.translated_manifest)
        write_progress(
            work_paths,
            stage="translate",
            status="skipped",
            done=done,
            total=total,
            summary="translated_segments.json already exists",
        )
        return 0
    stage_helpers.run_translate_stage(
        work_paths=work_paths,
        target_language=args.target_language,
        llm_model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        optimizer_root=Path(args.optimizer_root),
        thread_num=args.thread_num,
        batch_num=args.batch_num,
        custom_prompt=args.custom_prompt,
        glossary_xlsx=Path(args.glossary_xlsx) if args.glossary_xlsx else None,
        disable_thinking=args.disable_thinking,
        llm_extra_body=stage_helpers._parse_json_object_argument("llm_extra_body_json", args.llm_extra_body_json),
        timeout=args.timeout,
    )
    total = stage_helpers._read_json_dict_count(work_paths.split_manifest)
    done = stage_helpers._count_translated_items(work_paths.translated_manifest)
    complete = stage_helpers._translation_manifest_complete_for_split(work_paths)
    write_progress(
        work_paths,
        stage="translate",
        status="running" if complete else "failed",
        done=done,
        total=total,
        summary=f"{done}/{total or '?'} translated subtitles",
    )
    return 0 if complete else 1
