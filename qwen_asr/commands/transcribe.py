from __future__ import annotations

import argparse
import logging
from time import perf_counter

from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


def cmd_transcribe(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    stage_helpers.ensure_preflight(args, work_paths, "transcribe")
    if args.force:
        stage_helpers._clear_transcribe_outputs(work_paths)

    segments = stage_helpers.load_segments(work_paths.segments_manifest)
    if not segments:
        raise RuntimeError("segments.json is missing or empty. Run prepare first.")

    resolved_defaults = stage_helpers._resolve_transcribe_batch_defaults(args, segments)
    model_cache_dir = stage_helpers._resolve_model_cache_dir(args)
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    LOGGER.info(
        "Resolved transcribe batch defaults: profile=%s batch_size=%s target_audio_seconds=%s single_long_segment_threshold=%s",
        resolved_defaults["profile"],
        args.batch_size,
        getattr(args, "target_batch_audio_seconds", None),
        getattr(args, "single_long_segment_threshold", None),
    )

    existing = {} if args.force else {
        item.segment_id: item
        for item in stage_helpers.load_transcripts(
            work_paths.transcript_manifest,
            work_paths.transcript_checkpoint_path,
            work_paths.transcript_events_path,
        )
    }
    if args.resume and not args.force and all(
        item.segment_id in existing and existing[item.segment_id].status == "completed"
        for item in segments
    ):
        LOGGER.info("All transcript segments already completed. Skipping ASR model load.")
        stage_helpers.write_transcript_text(work_paths.transcript_text, stage_helpers._ordered_transcripts(segments, existing))
        write_progress(
            work_paths,
            stage="transcribe",
            status="skipped",
            done=len(segments),
            total=len(segments),
            summary="All transcript segments already completed",
        )
        return 0

    manifest = list(existing.values())
    batch_reports: list[dict[str, object]] = []

    transcriber = stage_helpers.QwenASRTranscriber(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
        keep_raw_model_output=args.keep_raw_model_output,
        model_cache_dir=model_cache_dir,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
        profile_batches=getattr(args, "profile_batches", False),
    )
    transcriber.load()
    try:
        pending_segments = []
        for segment in segments:
            existing_item = existing.get(segment.segment_id)
            if stage_helpers._should_skip(existing_item, args):
                LOGGER.info("Skipping completed segment %s", segment.segment_id)
                continue
            if not stage_helpers.Path(segment.audio_path).exists() and work_paths.audio_path.exists():
                stage_helpers.ensure_segment_audio(work_paths.audio_path, segment)
            pending_segments.append(segment)

        planner = stage_helpers.BatchPlanner(
            pending_segments,
            mode=getattr(args, "batch_mode", "adaptive"),
            max_batch_items=args.batch_size,
            target_audio_seconds=getattr(args, "target_batch_audio_seconds", None),
            single_long_segment_threshold=getattr(args, "single_long_segment_threshold", 90.0),
        )
        LOGGER.info("ASR batch planner initialized: %s", planner.describe_limits())

        while True:
            planned_batch = planner.next_batch()
            if planned_batch is None:
                break
            batch = planned_batch.segments
            batch_begin = perf_counter()
            batch_report: dict[str, object] = {
                "mode": planned_batch.mode,
                "bucket_label": planned_batch.bucket_label,
                "batch_size": len(batch),
                "total_duration": planned_batch.total_duration,
                "min_duration": planned_batch.min_duration,
                "max_duration": planned_batch.max_duration,
                "duration_spread_ratio": planned_batch.duration_spread_ratio,
                "singleton_reason": planned_batch.singleton_reason,
                "segment_ids": [segment.segment_id for segment in batch],
                "planner_limits_before": planner.describe_limits(),
            }
            try:
                results = transcriber.run_batch(batch)
            except stage_helpers.ASRBatchOOMError:
                batch_report["status"] = "oom_retry"
                batch_report["elapsed_s"] = round(perf_counter() - batch_begin, 3)
                batch_report["memory_probes"] = stage_helpers._consume_batch_memory_probes(transcriber)
                if planner.current_max_batch_items <= 1:
                    batch_reports.append(batch_report)
                    stage_helpers._write_transcribe_profile(work_paths, args, segments, batch_reports, resolved_defaults)
                    raise
                planner.report_oom(planned_batch)
                batch_report["planner_limits_after"] = planner.describe_limits()
                batch_reports.append(batch_report)
                LOGGER.warning(
                    "ASR batch OOM detected. Retrying with planner limits: %s from %s.",
                    planner.describe_limits(),
                    batch[0].segment_id,
                )
                continue
            batch_report["status"] = "completed"
            batch_report["elapsed_s"] = round(perf_counter() - batch_begin, 3)
            batch_report["memory_probes"] = stage_helpers._consume_batch_memory_probes(transcriber)
            LOGGER.info(
                "ASR batch completed: mode=%s bucket=%s size=%d audio_s=%.2f min_segment_s=%.2f max_segment_s=%.2f spread=%.3f singleton_reason=%s elapsed_s=%.3f first_segment=%s last_segment=%s",
                planned_batch.mode,
                planned_batch.bucket_label,
                len(batch),
                planned_batch.total_duration,
                planned_batch.min_duration,
                planned_batch.max_duration,
                planned_batch.duration_spread_ratio,
                planned_batch.singleton_reason or "",
                perf_counter() - batch_begin,
                batch[0].segment_id,
                batch[-1].segment_id,
            )
            planner.mark_success(planned_batch)
            batch_report["planner_limits_after"] = planner.describe_limits()
            batch_reports.append(batch_report)
            for result in results:
                existing[result.segment_id] = result
                manifest = stage_helpers._ordered_transcripts(segments, existing)
                stage_helpers._append_manifest_event(
                    work_paths.transcript_events_path,
                    manifest_type="transcript",
                    item=result,
                )
                stage_helpers._write_manifest_checkpoint(
                    work_paths.transcript_checkpoint_path,
                    manifest,
                )
                stage_helpers.write_json_atomic(work_paths.transcript_manifest, stage_helpers.serialize_manifest(manifest))
                stage_helpers.write_transcript_text(work_paths.transcript_text, manifest)
                write_progress(
                    work_paths,
                    stage="transcribe",
                    status="running",
                    done=sum(1 for item in manifest if item.status == "completed"),
                    total=len(segments),
                    current=result.segment_id,
                    summary="transcribing segments",
                )
    finally:
        transcriber.close()

    manifest = stage_helpers._ordered_transcripts(segments, existing)
    successes = [item for item in manifest if item.status == "completed" and item.text.strip()]
    stage_helpers._write_transcribe_profile(work_paths, args, segments, batch_reports, resolved_defaults)
    return 0 if successes else 1
