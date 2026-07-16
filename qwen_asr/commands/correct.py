from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


def cmd_correct(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    if args.force:
        stage_helpers._clear_correct_outputs(work_paths)
    if args.resume and not args.force and stage_helpers._correction_complete(work_paths):
        LOGGER.info("corrected_segments.json already exists. Skipping ASR correction stage.")
        total = stage_helpers._read_completed_list_count(work_paths.transcript_manifest)
        done = stage_helpers._read_completed_list_count(work_paths.corrected_manifest)
        write_progress(
            work_paths,
            stage="correct",
            status="skipped",
            done=done,
            total=total,
            summary="corrected_segments.json already exists",
        )
        return 0

    transcripts = stage_helpers.load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    if not transcripts:
        raise RuntimeError("transcript_segments.json is missing or empty. Run transcribe first.")

    corrected, report = stage_helpers.run_correction_stage(
        work_paths=work_paths,
        transcripts=transcripts,
        llm_model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        thread_num=args.thread_num,
        batch_num=getattr(args, "correct_batch_num", getattr(args, "batch_num", 8)),
        glossary_xlsx=Path(args.glossary_xlsx) if args.glossary_xlsx else None,
        disable_thinking=args.disable_thinking,
        llm_extra_body=stage_helpers._parse_json_object_argument("llm_extra_body_json", args.llm_extra_body_json),
        timeout=args.timeout,
    )
    stage_helpers.write_transcript_text(work_paths.transcript_text, corrected)
    changed_count = sum(1 for item in report if item.changed)
    failed_count = sum(1 for item in report if item.status != "completed")
    if changed_count:
        stage_helpers._clear_downstream_after_correction(work_paths)
    LOGGER.info(
        "ASR correction report written: %d changed, %d failed, %d total",
        changed_count,
        failed_count,
        len(report),
    )
    write_progress(
        work_paths,
        stage="correct",
        status="running",
        done=sum(1 for item in report if item.status == "completed"),
        total=len(report),
        summary=f"{changed_count} changed, {failed_count} failed, {len(report)} total",
    )
    return 0 if report and failed_count < len(report) else 1
