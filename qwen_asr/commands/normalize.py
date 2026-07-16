from __future__ import annotations

import argparse
import logging
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.normalize import NormalizeParams, normalize_asr_data
from qwen_asr.progress import write_progress
from qwen_asr.storage import write_json_atomic

LOGGER = logging.getLogger(__name__)


def cmd_normalize(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    if args.force:
        stage_helpers._clear_normalize_outputs(work_paths)
    gate_status = stage_helpers._require_quality_gate_before_formal_output(work_paths)
    if gate_status != 0:
        return gate_status
    if args.resume and not args.force and work_paths.normalized_manifest.exists():
        LOGGER.info("normalized_segments.json already exists. Skipping normalize stage.")
        count = stage_helpers._read_json_dict_count(work_paths.normalized_manifest)
        write_progress(
            work_paths,
            stage="normalize",
            status="skipped",
            done=count,
            total=count,
            summary="normalized_segments.json already exists",
        )
        return 0

    optimizer_root = Path(args.optimizer_root)
    source_asr_data = stage_helpers._load_normalize_source(args.source, work_paths, optimizer_root)
    if source_asr_data is None or not source_asr_data.segments:
        raise RuntimeError("No subtitle source available for normalize stage.")

    params = NormalizeParams(
        extend_ms=args.extend_ms,
        snap_gap_ms=args.snap_gap_ms,
        min_blank_ms=args.min_blank_ms,
    )
    result = normalize_asr_data(source_asr_data, params)
    write_json_atomic(work_paths.normalized_manifest, result.to_json())
    work_paths.normalized_srt.write_text(result.to_srt(), encoding="utf-8")
    LOGGER.info("Normalized %d subtitle segments", len(result.segments))
    write_progress(
        work_paths,
        stage="normalize",
        status="running",
        done=len(result.segments),
        total=len(result.segments),
        summary=f"Normalized {len(result.segments)} subtitle segments",
    )
    return 0
