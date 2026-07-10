from __future__ import annotations

import argparse
import logging
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from qwen_asr.align import QwenForcedAligner
from qwen_asr.asr import ASRBatchOOMError, QwenASRTranscriber
from qwen_asr.audio import ensure_segment_audio, export_segment_audio, extract_audio, load_audio_metadata
from qwen_asr.batching import BatchPlanner
from qwen_asr.corrector import run_correction_stage
from qwen_asr.credentials import resolve_mimo_api_key
from qwen_asr.artifact_state import ArtifactState
from qwen_asr.defaults import DEFAULT_MODEL_CACHE_DIR
from qwen_asr.models import (
    AlignedSegment,
    AlignedToken,
    AudioSegment,
    TranscriptSegment,
    WorkPaths,
)
from qwen_asr.normalize import NormalizeParams, normalize_asr_data
from qwen_asr.optimizer_bridge import (
    load_best_asr_data,
    load_specific_asr_data,
    run_split_stage,
    run_translate_stage,
)
from qwen_asr.pipeline_runner import PipelineRunner
from qwen_asr.preflight import ensure_preflight, format_preflight_messages, run_preflight
from qwen_asr.progress import write_progress
from qwen_asr.segmenter import SegmenterConfig, build_segments
from qwen_asr.storage import (
    append_jsonl,
    ensure_directory,
    load_jsonl,
    read_json,
    serialize_manifest,
    write_checkpoint_json,
    write_json_atomic,
)
from qwen_asr.mimo_proofread import main as run_mimo_proofread
from qwen_asr.subtitle import (
    SubtitleConfig,
    build_coarse_cues_from_transcripts,
    build_cues_from_aligned_segments,
    export_srt,
    export_vtt,
    export_vtt_from_optimizer_asr_data,
)
from qwen_asr.vad import create_vad_adapter, derive_silence_regions

LOGGER = logging.getLogger(__name__)


def _percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[index])


def _auto_select_transcribe_batch_defaults(segments: list[AudioSegment]) -> dict[str, object]:
    durations = sorted(max(0.0, float(segment.duration)) for segment in segments)
    if not durations:
        return {
            "profile": "empty",
            "batch_size": 5,
            "target_batch_audio_seconds": 220.0,
            "single_long_segment_threshold": 90.0,
            "reasons": ["No segments available; falling back to conservative defaults."],
        }

    p50 = _percentile(durations, 0.5)
    p75 = _percentile(durations, 0.75)
    p90 = _percentile(durations, 0.9)
    long_share = sum(1 for value in durations if value >= 90.0) / len(durations)
    ultra_share = sum(1 for value in durations if value >= 120.0) / len(durations)

    if p50 >= 85.0 or long_share >= 0.4:
        profile = "long_form"
        selected = {
            "batch_size": 3,
            "target_batch_audio_seconds": 300.0,
            "single_long_segment_threshold": 90.0,
        }
        reasons = ["Segment distribution is long-form heavy; prefer smaller batch caps and isolate long tails earlier."]
    elif p75 >= 75.0 or long_share >= 0.2:
        profile = "mixed"
        selected = {
            "batch_size": 4,
            "target_batch_audio_seconds": 260.0,
            "single_long_segment_threshold": 95.0,
        }
        reasons = ["Segment durations are mixed; keep moderate batch caps to balance throughput and padding risk."]
    else:
        profile = "short_form"
        selected = {
            "batch_size": 5,
            "target_batch_audio_seconds": 220.0,
            "single_long_segment_threshold": 110.0,
        }
        reasons = ["Most segments are short; allow a higher batch cap while keeping a long-tail escape hatch."]

    if ultra_share >= 0.1:
        selected["single_long_segment_threshold"] = min(float(selected["single_long_segment_threshold"]), 90.0)
        reasons.append("A noticeable share of segments is 120s+; tighten the long-segment threshold.")

    return {
        "profile": profile,
        "batch_size": int(selected["batch_size"]),
        "target_batch_audio_seconds": float(selected["target_batch_audio_seconds"]),
        "single_long_segment_threshold": float(selected["single_long_segment_threshold"]),
        "segment_stats": {
            "p50_duration": round(p50, 2),
            "p75_duration": round(p75, 2),
            "p90_duration": round(p90, 2),
            "long_segment_share": round(long_share, 3),
            "ultra_long_segment_share": round(ultra_share, 3),
        },
        "reasons": reasons,
    }


def _resolve_transcribe_batch_defaults(args: argparse.Namespace, segments: list[AudioSegment]) -> dict[str, object]:
    auto_selection = _auto_select_transcribe_batch_defaults(segments)
    batch_size_explicit = getattr(args, "batch_size", None) is not None
    target_explicit = getattr(args, "target_batch_audio_seconds", None) is not None
    threshold_explicit = getattr(args, "single_long_segment_threshold", None) is not None

    if getattr(args, "batch_mode", "adaptive") == "fixed":
        if not batch_size_explicit:
            args.batch_size = 5
        return {
            "profile": "fixed",
            "batch_size_source": "explicit" if batch_size_explicit else "auto",
            "target_audio_seconds_source": "n/a",
            "single_long_segment_threshold_source": "n/a",
            "auto_selection": auto_selection,
        }

    if not batch_size_explicit:
        args.batch_size = int(auto_selection["batch_size"])
    if not target_explicit:
        args.target_batch_audio_seconds = float(auto_selection["target_batch_audio_seconds"])
    if not threshold_explicit:
        args.single_long_segment_threshold = float(auto_selection["single_long_segment_threshold"])

    return {
        "profile": str(auto_selection["profile"]),
        "batch_size_source": "explicit" if batch_size_explicit else "auto",
        "target_audio_seconds_source": "explicit" if target_explicit else "auto",
        "single_long_segment_threshold_source": "explicit" if threshold_explicit else "auto",
        "auto_selection": auto_selection,
    }


def _resolve_model_cache_dir(args: argparse.Namespace) -> str:
    cache_dir = getattr(args, "model_cache_dir", None)
    if cache_dir:
        return str(cache_dir)
    resolved = str(DEFAULT_MODEL_CACHE_DIR)
    args.model_cache_dir = resolved
    return resolved


def _prepare_model_cache_dir(model_cache_dir: str, *, local_files_only: bool) -> None:
    path = Path(model_cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write-test"
    try:
        probe.write_text("ok", encoding="ascii")
    except OSError as exc:
        raise RuntimeError(f"Model cache directory is not writable: {path}") from exc
    finally:
        if probe.exists():
            probe.unlink()

    if local_files_only and not any(path.iterdir()):
        raise RuntimeError(
            "Model cache directory is empty while local_files_only=True. "
            f"Populate {path} first, pass --model-cache-dir to an existing cache, "
            "or use --no-local-files-only to allow downloading."
        )


def _consume_batch_memory_probes(transcriber: object) -> list[dict[str, object]]:
    consume = getattr(transcriber, "consume_last_batch_memory_probes", None)
    if callable(consume):
        return list(consume())
    return []


def _write_transcribe_profile(
    work_paths: WorkPaths,
    args: argparse.Namespace,
    segments: list[AudioSegment],
    batch_reports: list[dict[str, object]],
    resolved_defaults: dict[str, object],
) -> None:
    if not getattr(args, "profile_batches", False):
        return
    completed = [report for report in batch_reports if report.get("status") == "completed"]
    oom_retries = [report for report in batch_reports if report.get("status") == "oom_retry"]
    singleton_reasons = Counter(
        str(report.get("singleton_reason"))
        for report in completed
        if report.get("singleton_reason")
    )
    profile_payload = {
        "stage": "transcribe",
        "batch_mode": getattr(args, "batch_mode", "adaptive"),
        "configured_batch_size": args.batch_size,
        "configured_target_batch_audio_seconds": getattr(args, "target_batch_audio_seconds", None),
        "configured_single_long_segment_threshold": getattr(args, "single_long_segment_threshold", 90.0),
        "resolved_defaults": resolved_defaults,
        "segment_count": len(segments),
        "summary": {
            "batch_count": len(batch_reports),
            "completed_batch_count": len(completed),
            "oom_retry_count": len(oom_retries),
            "max_completed_batch_size": max((int(report["batch_size"]) for report in completed), default=0),
            "max_completed_audio_seconds": max((float(report["total_duration"]) for report in completed), default=0.0),
            "max_duration_spread_ratio": max((float(report["duration_spread_ratio"]) for report in completed), default=0.0),
            "singleton_batches": sum(1 for report in completed if report.get("singleton_reason")),
            "singleton_reasons": dict(singleton_reasons),
        },
        "recommendation": _build_transcribe_recommendation(args, completed, oom_retries),
        "batches": batch_reports,
    }
    write_json_atomic(work_paths.transcribe_profile_path, profile_payload)


def _build_transcribe_recommendation(
    args: argparse.Namespace,
    completed: list[dict[str, object]],
    oom_retries: list[dict[str, object]],
) -> dict[str, object]:
    configured_batch_size = int(args.batch_size)
    configured_target_audio_seconds = getattr(args, "target_batch_audio_seconds", None)
    configured_single_long_segment_threshold = float(getattr(args, "single_long_segment_threshold", 90.0))

    completed_batch_sizes = [int(report["batch_size"]) for report in completed]
    completed_audio_seconds = [float(report["total_duration"]) for report in completed]
    completed_max_duration = [float(report["max_duration"]) for report in completed]
    singleton_long_segments = [
        float(report["max_duration"])
        for report in completed
        if report.get("singleton_reason") == "long_segment_threshold"
    ]
    high_spread_batches = [
        report
        for report in completed
        if float(report.get("duration_spread_ratio", 0.0)) >= 2.5
    ]

    if oom_retries:
        recommended_batch_size = max(
            1,
            min(
                configured_batch_size,
                min(max(1, int(report["batch_size"]) - 1) for report in oom_retries),
            ),
        )
    elif completed_batch_sizes:
        recommended_batch_size = max(completed_batch_sizes)
    else:
        recommended_batch_size = configured_batch_size

    if configured_target_audio_seconds is None:
        if completed_audio_seconds:
            recommended_target_audio_seconds = round(max(1.0, max(completed_audio_seconds) * 0.9), 2)
        else:
            recommended_target_audio_seconds = None
    else:
        if oom_retries and completed_audio_seconds:
            recommended_target_audio_seconds = round(
                min(float(configured_target_audio_seconds), max(completed_audio_seconds) * 0.95),
                2,
            )
        elif completed_audio_seconds:
            recommended_target_audio_seconds = round(
                max(float(configured_target_audio_seconds), max(completed_audio_seconds)),
                2,
            )
        else:
            recommended_target_audio_seconds = float(configured_target_audio_seconds)

    if singleton_long_segments:
        recommended_single_long_segment_threshold = round(min(singleton_long_segments), 2)
    elif completed_max_duration:
        recommended_single_long_segment_threshold = round(
            max(configured_single_long_segment_threshold, max(completed_max_duration) * 1.1),
            2,
        )
    else:
        recommended_single_long_segment_threshold = configured_single_long_segment_threshold

    reasons: list[str] = []
    if oom_retries:
        reasons.append("Observed OOM retries; recommend a lower stable batch cap.")
    if high_spread_batches:
        reasons.append("Some completed batches still had high duration spread; tighter grouping or lower target audio seconds may help.")
    if singleton_long_segments:
        reasons.append("Long-tail segments were isolated into singleton batches; keep the long-segment threshold near the shortest isolated segment.")
    if not reasons:
        reasons.append("No OOM retries were observed; recommendations are based on the largest completed adaptive batches.")

    return {
        "next_run": {
            "batch_mode": getattr(args, "batch_mode", "adaptive"),
            "batch_size": recommended_batch_size,
            "target_batch_audio_seconds": recommended_target_audio_seconds,
            "single_long_segment_threshold": recommended_single_long_segment_threshold,
        },
        "signals": {
            "oom_retry_count": len(oom_retries),
            "high_spread_batch_count": len(high_spread_batches),
            "singleton_long_segment_count": len(singleton_long_segments),
        },
        "reasons": reasons,
    }

def cmd_prepare(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    ensure_preflight(args, work_paths, "prepare")
    ensure_directory(work_paths.workdir)
    ensure_directory(work_paths.segments_dir)
    ensure_directory(work_paths.logs_dir)

    if args.force:
        _clear_prepare_outputs(work_paths)

    media_path = _resolve_media_path(args)
    if not media_path.exists():
        raise FileNotFoundError(f"Media not found: {media_path}")
    _write_project_metadata(args, work_paths, media_path)

    audio_path = extract_audio(
        media_path,
        work_paths.audio_path,
        overwrite=args.force,
        denoise=args.denoise,
        denoise_level=getattr(args, "denoise_level", 12.0),
        denoise_backend=getattr(args, "denoise_backend", "mdx_net"),
        denoise_profile=getattr(args, "denoise_profile", "strong"),
        mdx_model=getattr(args, "mdx_model", "UVR-MDX-NET-Inst_HQ_3.onnx"),
        mdx_model_dir=getattr(args, "mdx_model_dir", None),
    )
    _, duration = load_audio_metadata(audio_path)

    vad = create_vad_adapter(
        getattr(args, "vad_backend", "pyannote_onnx_v3"),
        threshold=getattr(args, "vad_threshold", 0.5),
        onset=getattr(args, "vad_onset", 0.5),
        offset=getattr(args, "vad_offset", 0.35),
        min_speech_duration_ms=getattr(args, "vad_min_speech_ms", 180),
        min_silence_duration_ms=getattr(args, "vad_min_silence_ms", 250),
        speech_pad_ms=getattr(args, "vad_speech_pad_ms", 120),
        pyannote_model=getattr(args, "pyannote_onnx_model", "segmentation-3.0"),
    )
    speech_regions = vad.detect(audio_path)
    speech_duration = sum(max(0.0, item.end_time - item.start_time) for item in speech_regions)
    LOGGER.info(
        "VAD completed: backend=%s regions=%d speech_s=%.2f coverage=%.3f",
        getattr(args, "vad_backend", "pyannote_onnx_v3"),
        len(speech_regions),
        speech_duration,
        speech_duration / duration if duration > 0 else 0.0,
    )
    silence_regions = derive_silence_regions(speech_regions, duration)
    config = SegmenterConfig(
        max_segment_seconds=args.max_segment_seconds,
        min_segment_seconds=args.min_segment_seconds,
        preferred_silence_ms=args.preferred_silence_ms,
        min_silence_ms=args.min_silence_ms,
        padding_ms=args.padding_ms,
        overlap_ms=args.overlap_ms,
    )
    segments = build_segments(
        speech_regions=speech_regions,
        silence_regions=silence_regions,
        audio_duration=duration,
        source_audio_path=audio_path,
        segments_dir=work_paths.segments_dir,
        config=config,
    )
    eager_segment_export = bool(getattr(args, "eager_segment_export", False))
    for segment in segments:
        if eager_segment_export:
            export_segment_audio(audio_path, segment)
        segment.status = "prepared"
    write_json_atomic(work_paths.segments_manifest, serialize_manifest(segments))
    LOGGER.info("Prepared %d segments", len(segments))
    write_progress(
        work_paths,
        stage="prepare",
        status="running",
        done=len(segments),
        total=len(segments),
        summary=f"Prepared {len(segments)} segments",
    )
    return 0 if segments else 1


def cmd_transcribe(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    ensure_preflight(args, work_paths, "transcribe")
    if args.force:
        _clear_transcribe_outputs(work_paths)

    segments = load_segments(work_paths.segments_manifest)
    if not segments:
        raise RuntimeError("segments.json is missing or empty. Run prepare first.")

    resolved_defaults = _resolve_transcribe_batch_defaults(args, segments)
    model_cache_dir = _resolve_model_cache_dir(args)
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
        for item in load_transcripts(
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
        write_transcript_text(work_paths.transcript_text, _ordered_transcripts(segments, existing))
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

    transcriber = QwenASRTranscriber(
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
        pending_segments: list[AudioSegment] = []
        for segment in segments:
            existing_item = existing.get(segment.segment_id)
            if _should_skip(existing_item, args):
                LOGGER.info("Skipping completed segment %s", segment.segment_id)
                continue
            if not Path(segment.audio_path).exists() and work_paths.audio_path.exists():
                ensure_segment_audio(work_paths.audio_path, segment)
            pending_segments.append(segment)

        planner = BatchPlanner(
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
            except ASRBatchOOMError:
                batch_report["status"] = "oom_retry"
                batch_report["elapsed_s"] = round(perf_counter() - batch_begin, 3)
                batch_report["memory_probes"] = _consume_batch_memory_probes(transcriber)
                if planner.current_max_batch_items <= 1:
                    batch_reports.append(batch_report)
                    _write_transcribe_profile(work_paths, args, segments, batch_reports, resolved_defaults)
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
            batch_report["memory_probes"] = _consume_batch_memory_probes(transcriber)
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
                manifest = _ordered_transcripts(segments, existing)
                _append_manifest_event(
                    work_paths.transcript_events_path,
                    manifest_type="transcript",
                    item=result,
                )
                _write_manifest_checkpoint(
                    work_paths.transcript_checkpoint_path,
                    manifest,
                )
                write_json_atomic(work_paths.transcript_manifest, serialize_manifest(manifest))
                write_transcript_text(work_paths.transcript_text, manifest)
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

    manifest = _ordered_transcripts(segments, existing)
    successes = [item for item in manifest if item.status == "completed" and item.text.strip()]
    _write_transcribe_profile(work_paths, args, segments, batch_reports, resolved_defaults)
    return 0 if successes else 1


def cmd_align(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    ensure_preflight(args, work_paths, "align")
    cleanup_interval = max(1, int(getattr(args, "cleanup_interval", 4)))
    model_cache_dir = _resolve_model_cache_dir(args)

    if args.force:
        _clear_align_outputs(work_paths)

    transcripts = load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    if not transcripts:
        raise RuntimeError("transcript_segments.json is missing or empty. Run transcribe first.")

    existing = {} if args.force else {
        item.segment_id: item
        for item in load_aligned_segments(
            work_paths.aligned_manifest,
            work_paths.aligned_checkpoint_path,
            work_paths.aligned_events_path,
        )
    }
    eligible_transcripts = [item for item in transcripts if item.status == "completed" and item.text.strip()]
    if args.resume and not args.force and eligible_transcripts and all(
        item.segment_id in existing and existing[item.segment_id].status == "completed"
        for item in eligible_transcripts
    ):
        LOGGER.info("All alignment segments already completed. Skipping aligner model load.")
        write_progress(
            work_paths,
            stage="align",
            status="skipped",
            done=len(eligible_transcripts),
            total=len(eligible_transcripts),
            summary="All alignment segments already completed",
        )
        return 0

    aligner = QwenForcedAligner(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        keep_raw_model_output=args.keep_raw_model_output,
        model_cache_dir=model_cache_dir,
        local_files_only=args.local_files_only,
    )
    aligner.load()
    try:
        attempted = 0
        for transcript in transcripts:
            if transcript.status != "completed" or not transcript.text.strip():
                LOGGER.warning("Skipping transcript without usable text: %s", transcript.segment_id)
                continue
            transcript_audio_path = Path(transcript.audio_path)
            if not transcript_audio_path.exists() and work_paths.audio_path.exists():
                ensure_segment_audio(work_paths.audio_path, AudioSegment(
                    segment_id=transcript.segment_id,
                    audio_path=transcript.audio_path,
                    source_audio_path=str(work_paths.audio_path),
                    global_start_time=transcript.global_start_time,
                    global_end_time=transcript.global_end_time,
                    duration=max(0.0, transcript.global_end_time - transcript.global_start_time),
                ))
            existing_item = existing.get(transcript.segment_id)
            if _should_skip(existing_item, args):
                LOGGER.info("Skipping completed alignment %s", transcript.segment_id)
                continue
            attempted += 1
            cleanup_now = attempted % cleanup_interval == 0
            result = aligner.run_segment(transcript, cleanup=cleanup_now)
            existing[result.segment_id] = result
            manifest = _ordered_aligned(transcripts, existing)
            _append_manifest_event(
                work_paths.aligned_events_path,
                manifest_type="aligned",
                item=result,
            )
            _write_manifest_checkpoint(
                work_paths.aligned_checkpoint_path,
                manifest,
            )
            write_json_atomic(work_paths.aligned_manifest, serialize_manifest(manifest))
            write_progress(
                work_paths,
                stage="align",
                status="running",
                done=sum(1 for item in manifest if item.status == "completed"),
                total=len(eligible_transcripts),
                current=result.segment_id,
                summary="aligning segments",
            )
    finally:
        aligner.close()

    manifest = _ordered_aligned(transcripts, existing)
    successes = [item for item in manifest if item.status == "completed"]
    return 0 if successes else 1


def cmd_export(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    if args.force:
        _clear_export_outputs(work_paths)

    config = SubtitleConfig(
        max_subtitle_duration=args.max_subtitle_duration,
        min_subtitle_duration=args.min_subtitle_duration,
        max_chars_per_line_zh=args.max_chars_per_line_zh,
        max_chars_per_line_en=args.max_chars_per_line_en,
        max_lines=args.max_lines,
        pause_split_seconds=args.pause_split_seconds,
    )

    transcripts = load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    write_transcript_text(work_paths.transcript_text, transcripts)

    optimizer_root = Path(args.optimizer_root)
    optimizer_asr_data = None
    if args.source in {"auto", "normalized", "translated", "split", "transcript"}:
        optimizer_asr_data = _load_optimizer_export_source(args.source, work_paths, optimizer_root)

    if optimizer_asr_data is not None:
        if args.format in {"srt", "both"}:
            ensure_directory(work_paths.subtitles_srt.parent)
            work_paths.subtitles_srt.write_text(optimizer_asr_data.to_srt(), encoding="utf-8")
        if args.format in {"vtt", "both"}:
            ensure_directory(work_paths.subtitles_vtt.parent)
            work_paths.subtitles_vtt.write_text(
                export_vtt_from_optimizer_asr_data(optimizer_asr_data),
                encoding="utf-8",
            )
        ready = _finalize_exports(args, work_paths)
        write_progress(
            work_paths,
            stage="export",
            status="running",
            done=len(ready),
            total=1 if args.format in {"srt", "vtt"} else 2,
            current=", ".join(ready),
            summary=f"exported {', '.join(ready) if ready else 'no subtitle files'}",
        )
        return 0

    aligned = load_aligned_segments(
        work_paths.aligned_manifest,
        work_paths.aligned_checkpoint_path,
        work_paths.aligned_events_path,
    ) if args.source in {"auto", "aligned"} else []
    cues = []
    if aligned:
        cues = build_cues_from_aligned_segments(aligned, config)
    elif args.coarse_subtitles or args.source == "transcript":
        cues = build_coarse_cues_from_transcripts(transcripts, config)

    if not cues and args.format in {"srt", "vtt", "both"}:
        LOGGER.warning("No timestamped cues available. Only transcript.txt was written.")
        write_progress(
            work_paths,
            stage="export",
            status="running",
            done=0,
            total=1 if args.format in {"srt", "vtt"} else 2,
            current="transcript.txt",
            summary="No timestamped cues available",
        )
        return 0 if work_paths.transcript_text.exists() else 1

    if args.format in {"srt", "both"}:
        ensure_directory(work_paths.subtitles_srt.parent)
        work_paths.subtitles_srt.write_text(export_srt(cues), encoding="utf-8")
    if args.format in {"vtt", "both"}:
        ensure_directory(work_paths.subtitles_vtt.parent)
        work_paths.subtitles_vtt.write_text(export_vtt(cues), encoding="utf-8")
    ready = _finalize_exports(args, work_paths)
    write_progress(
        work_paths,
        stage="export",
        status="running",
        done=len(ready),
        total=1 if args.format in {"srt", "vtt"} else 2,
        current=", ".join(ready),
        summary=f"exported {', '.join(ready) if ready else 'no subtitle files'}",
    )
    return 0


def cmd_normalize(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    if args.force:
        _clear_normalize_outputs(work_paths)
    if args.resume and not args.force and work_paths.normalized_manifest.exists():
        LOGGER.info("normalized_segments.json already exists. Skipping normalize stage.")
        count = _read_json_dict_count(work_paths.normalized_manifest)
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
    source_asr_data = _load_normalize_source(args.source, work_paths, optimizer_root)
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


def cmd_split(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    if args.force:
        _clear_split_outputs(work_paths)
    if args.resume and not args.force and work_paths.split_manifest.exists():
        LOGGER.info("split_segments.json already exists. Skipping split stage.")
        count = _read_json_dict_count(work_paths.split_manifest)
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
        llm_extra_body=_parse_json_object_argument("llm_extra_body_json", args.llm_extra_body_json),
        timeout=args.timeout,
        split_mode=getattr(args, "split_mode", "token-boundary"),
    )
    count = _read_json_dict_count(work_paths.split_manifest)
    write_progress(
        work_paths,
        stage="split",
        status="running",
        done=count,
        total=count,
        summary=f"generated {count} split subtitles",
    )
    return 0


def cmd_translate(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    if args.force:
        _clear_translate_outputs(work_paths)
    if args.resume and not args.force and work_paths.translated_manifest.exists():
        LOGGER.info("translated_segments.json already exists. Skipping translate stage.")
        total = _read_json_dict_count(work_paths.split_manifest)
        done = _count_translated_items(work_paths.translated_manifest)
        write_progress(
            work_paths,
            stage="translate",
            status="skipped",
            done=done,
            total=total,
            summary="translated_segments.json already exists",
        )
        return 0
    run_translate_stage(
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
        llm_extra_body=_parse_json_object_argument("llm_extra_body_json", args.llm_extra_body_json),
        timeout=args.timeout,
    )
    total = _read_json_dict_count(work_paths.split_manifest)
    done = _count_translated_items(work_paths.translated_manifest)
    write_progress(
        work_paths,
        stage="translate",
        status="running",
        done=done,
        total=total,
        summary=f"{done}/{total or '?'} translated subtitles",
    )
    return 0


def cmd_correct(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    if args.force:
        _clear_correct_outputs(work_paths)
    if args.resume and not args.force and _correction_complete(work_paths):
        LOGGER.info("corrected_segments.json already exists. Skipping ASR correction stage.")
        total = _read_completed_list_count(work_paths.transcript_manifest)
        done = _read_completed_list_count(work_paths.corrected_manifest)
        write_progress(
            work_paths,
            stage="correct",
            status="skipped",
            done=done,
            total=total,
            summary="corrected_segments.json already exists",
        )
        return 0

    transcripts = load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    if not transcripts:
        raise RuntimeError("transcript_segments.json is missing or empty. Run transcribe first.")

    corrected, report = run_correction_stage(
        work_paths=work_paths,
        transcripts=transcripts,
        llm_model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        thread_num=args.thread_num,
        batch_num=getattr(args, "correct_batch_num", getattr(args, "batch_num", 8)),
        glossary_xlsx=Path(args.glossary_xlsx) if args.glossary_xlsx else None,
        disable_thinking=args.disable_thinking,
        llm_extra_body=_parse_json_object_argument("llm_extra_body_json", args.llm_extra_body_json),
        timeout=args.timeout,
    )
    write_transcript_text(work_paths.transcript_text, corrected)
    changed_count = sum(1 for item in report if item.changed)
    failed_count = sum(1 for item in report if item.status != "completed")
    if changed_count:
        _clear_downstream_after_correction(work_paths)
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


def cmd_run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    ensure_preflight(args, work_paths, "run")
    if args.with_correct:
        if not args.llm_model or not args.llm_base_url or not args.llm_api_key:
            raise RuntimeError("correct stage requires --llm-model, --llm-base-url, and --llm-api-key")
    if args.with_translate:
        if not args.target_language or not args.llm_model or not args.llm_base_url or not args.llm_api_key:
            raise RuntimeError("translate stage requires --target-language, --llm-model, --llm-base-url, and --llm-api-key")
    if getattr(args, "with_mimo_proofread", False):
        translated_ready = work_paths.translated_manifest.exists() and _translated_manifest_has_content(work_paths.translated_manifest)
        if not args.with_translate and not translated_ready:
            raise RuntimeError("MiMo audio proofread requires --with-translate or an existing non-empty translated_segments.json")
        if not resolve_mimo_api_key(getattr(args, "mimo_api_key", None)):
            raise RuntimeError("MiMo audio proofread requires MIMO_API_KEY")
    handlers = {
        "prepare": cmd_prepare,
        "transcribe": cmd_transcribe,
        "correct": cmd_correct,
        "align": cmd_align,
        "split": cmd_split,
        "translate": cmd_translate,
        "mimo-proofread": cmd_mimo_proofread,
        "normalize": cmd_normalize,
        "export": cmd_export,
    }
    return PipelineRunner(work_paths, handlers).run(args)


def cmd_mimo_proofread(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    """Create an isolated MiMo candidate branch; formal subtitles remain unchanged."""
    api_key = resolve_mimo_api_key(getattr(args, "mimo_api_key", None))
    if not api_key:
        raise RuntimeError("MiMo audio proofread requires MIMO_API_KEY")
    command = [
        "--workdir", str(work_paths.workdir),
        "--output-dir", str(work_paths.mimo_proofread_dir),
        "--workers", str(max(1, int(getattr(args, "mimo_proofread_workers", 1)))),
        "--proofread-mode", str(getattr(args, "mimo_proofread_mode", "segment-audio")),
        "--nearby-batch-size", str(max(1, int(getattr(args, "mimo_nearby_batch_size", 1)))),
        "--nearby-batch-max-gap-s", str(max(0.0, float(getattr(args, "mimo_nearby_batch_max_gap_s", 8.0)))),
        "--nearby-padding-s", str(max(0.0, float(getattr(args, "mimo_nearby_padding_s", 1.5)))),
        "--nearby-context-subtitles", str(max(0, int(getattr(args, "mimo_nearby_context_subtitles", 1)))),
        "--nearby-audio-workers", str(max(1, int(getattr(args, "mimo_nearby_audio_workers", 1)))),
        "--max-tokens", str(max(512, int(getattr(args, "mimo_proofread_max_tokens", 4096)))),
        "--timeout", str(max(1.0, float(getattr(args, "timeout", 240.0)))),
        "--disable-thinking",
    ]
    if getattr(args, "mimo_compact_output", False):
        command.append("--compact-output")
    if getattr(args, "resume", True):
        command.append("--resume")
    if getattr(args, "glossary_xlsx", None):
        command.extend(["--glossary-xlsx", str(args.glossary_xlsx)])
    previous_key = os.environ.get("MIMO_API_KEY")
    os.environ["MIMO_API_KEY"] = api_key
    try:
        return run_mimo_proofread(command)
    finally:
        if previous_key is None:
            os.environ.pop("MIMO_API_KEY", None)
        else:
            os.environ["MIMO_API_KEY"] = previous_key


def _translated_manifest_has_content(path: Path) -> bool:
    payload = read_json(path, default={})
    return isinstance(payload, dict) and bool(payload) and all(
        str(item.get("translated_subtitle", "")).strip() for item in payload.values() if isinstance(item, dict)
    )


def cmd_preflight(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    result = run_preflight(args, work_paths, "preflight")
    for line in format_preflight_messages(result):
        print(line)
    return 0 if result.ok else 1


def load_segments(path: Path) -> list[AudioSegment]:
    data = read_json(path, default=[])
    return [AudioSegment(**item) for item in data]


def load_transcripts(path: Path, checkpoint_path: Path | None = None, events_path: Path | None = None) -> list[TranscriptSegment]:
    data = read_json(path, default=[])
    if not data and checkpoint_path is not None and events_path is not None:
        data = _recover_manifest_from_checkpoint_and_events(checkpoint_path, events_path)
        if data:
            write_json_atomic(path, data)
    return [TranscriptSegment(**item) for item in data]


def load_aligned_segments(path: Path, checkpoint_path: Path | None = None, events_path: Path | None = None) -> list[AlignedSegment]:
    data = read_json(path, default=[])
    if not data and checkpoint_path is not None and events_path is not None:
        data = _recover_manifest_from_checkpoint_and_events(checkpoint_path, events_path)
        if data:
            write_json_atomic(path, data)
    segments: list[AlignedSegment] = []
    for item in data:
        tokens = [AlignedToken(**token) for token in item.get("tokens", [])]
        clone = dict(item)
        clone["tokens"] = tokens
        segments.append(AlignedSegment(**clone))
    return segments


def _resolve_media_path(args: argparse.Namespace) -> Path:
    media = getattr(args, "media", None) or getattr(args, "video", None)
    if not media:
        raise ValueError("--media is required")
    return Path(media).resolve()


def _resolve_export_media_path(args: argparse.Namespace, work_paths: WorkPaths) -> Path:
    media = getattr(args, "media_path", None) or getattr(args, "media", None) or getattr(args, "video", None)
    if media:
        return Path(media).resolve()
    metadata = _load_project_metadata(work_paths)
    original = str(metadata.get("original_media_path", "")).strip()
    if original:
        return Path(original).resolve()
    raise RuntimeError("Export source mode requires project.json original_media_path or --media-path.")


def _write_project_metadata(args: argparse.Namespace, work_paths: WorkPaths, media_path: Path) -> None:
    existing = _load_project_metadata(work_paths)
    export_mode = str(getattr(args, "export_mode", None) or existing.get("export_mode") or "source")
    custom_export_path = str(
        getattr(args, "export_path", None)
        or existing.get("custom_export_path")
        or ""
    )
    payload = {
        **existing,
        "original_media_path": str(media_path),
        "source_name": media_path.stem,
        "created_at": existing.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "export_mode": export_mode,
        "custom_export_path": custom_export_path,
    }
    write_json_atomic(work_paths.project_metadata, payload)


def _load_project_metadata(work_paths: WorkPaths) -> dict:
    payload = read_json(work_paths.project_metadata, default={})
    return payload if isinstance(payload, dict) else {}


def _finalize_exports(args: argparse.Namespace, work_paths: WorkPaths) -> list[str]:
    metadata = _load_project_metadata(work_paths)
    export_mode = str(getattr(args, "export_mode", None) or metadata.get("export_mode") or "source")
    export_path = str(getattr(args, "export_path", None) or metadata.get("custom_export_path") or "").strip()
    media_path = _resolve_export_media_path(args, work_paths)
    if export_mode not in {"source", "custom"}:
        raise RuntimeError(f"Unsupported export mode: {export_mode}")
    if export_mode == "custom" and not export_path:
        raise RuntimeError("Custom export mode requires --export-path.")

    targets = _export_targets(
        format_name=args.format,
        media_path=media_path,
        export_mode=export_mode,
        export_path=Path(export_path) if export_path else None,
    )
    ready: list[str] = []
    cache_paths = {"srt": work_paths.subtitles_srt, "vtt": work_paths.subtitles_vtt}
    for suffix, cache_path in cache_paths.items():
        target = targets.get(suffix)
        if target is None or not cache_path.exists():
            continue
        ensure_directory(target.parent)
        shutil.copy2(cache_path, target)
        ready.append(suffix)
        LOGGER.info("Saved %s export: %s", suffix, target)

    payload = {
        **metadata,
        "original_media_path": str(media_path),
        "source_name": media_path.stem,
        "created_at": metadata.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "export_mode": export_mode,
        "custom_export_path": export_path,
        "last_exported": {key: str(value) for key, value in targets.items() if key in ready},
    }
    write_json_atomic(work_paths.project_metadata, payload)
    return ready


def _export_targets(
    *,
    format_name: str,
    media_path: Path,
    export_mode: str,
    export_path: Path | None,
) -> dict[str, Path]:
    requested = ("srt", "vtt") if format_name == "both" else (format_name,)
    if export_mode == "source":
        base = media_path.with_suffix("")
        return {suffix: base.with_suffix(f".{suffix}") for suffix in requested}
    assert export_path is not None
    if _looks_like_file_path(export_path):
        if len(requested) == 1:
            suffix = requested[0]
            return {suffix: export_path if export_path.suffix else export_path.with_suffix(f".{suffix}")}
        return {suffix: export_path.with_suffix(f".{suffix}") for suffix in requested}
    return {suffix: export_path / f"{media_path.stem}.{suffix}" for suffix in requested}


def _looks_like_file_path(path: Path) -> bool:
    if path.exists():
        return path.is_file()
    return bool(path.suffix)


def write_transcript_text(path: Path, transcripts: list[TranscriptSegment]) -> None:
    ensure_directory(path.parent)
    merged_lines: list[str] = []
    previous = ""
    for item in transcripts:
        if item.status != "completed" or not item.text.strip():
            continue
        current = item.text.strip()
        if previous:
            current = _dedupe_boundary_text(previous, current)
        if current:
            merged_lines.append(current)
            previous = current
    path.write_text("\n".join(merged_lines).strip() + ("\n" if merged_lines else ""), encoding="utf-8")


def _should_skip(existing_item: TranscriptSegment | AlignedSegment | None, args: argparse.Namespace) -> bool:
    if args.force:
        return False
    if not args.resume:
        return False
    return existing_item is not None and existing_item.status == "completed"


def _ordered_transcripts(
    segments: list[AudioSegment],
    transcript_map: dict[str, TranscriptSegment],
) -> list[TranscriptSegment]:
    return [transcript_map[item.segment_id] for item in segments if item.segment_id in transcript_map]


def _ordered_aligned(
    transcripts: list[TranscriptSegment],
    aligned_map: dict[str, AlignedSegment],
) -> list[AlignedSegment]:
    return [aligned_map[item.segment_id] for item in transcripts if item.segment_id in aligned_map]


def _chunked(items: list[AudioSegment], size: int) -> list[list[AudioSegment]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _load_optimizer_export_source(source: str, work_paths: WorkPaths, optimizer_root: Path):
    if source in {"normalized", "translated", "split", "transcript"}:
        return load_specific_asr_data(work_paths, source=source, optimizer_root=optimizer_root)
    return load_best_asr_data(work_paths, optimizer_root=optimizer_root)


def _load_normalize_source(source: str, work_paths: WorkPaths, optimizer_root: Path):
    if source == "auto":
        for candidate in ("translated", "split", "transcript"):
            data = load_specific_asr_data(work_paths, source=candidate, optimizer_root=optimizer_root)
            if data is not None and getattr(data, "segments", None):
                return data
        return None
    return load_specific_asr_data(work_paths, source=source, optimizer_root=optimizer_root)


def _clear_prepare_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "prepare")
    if work_paths.segments_dir.exists():
        for file_path in work_paths.segments_dir.glob("*.wav"):
            file_path.unlink(missing_ok=True)


def _clear_transcribe_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "transcribe")
    _clear_downstream_after_correction(work_paths)


def _clear_align_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "align")


def _clear_split_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "split")


def _clear_translate_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "translate")


def _clear_normalize_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "normalize")


def _clear_export_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "export")


def _clear_correct_outputs(work_paths: WorkPaths) -> None:
    _clear_stage_outputs(work_paths, "correct")


def _clear_downstream_after_correction(work_paths: WorkPaths) -> None:
    ArtifactState(work_paths).delete_downstream_outputs("correct")


def _clear_stage_outputs(work_paths: WorkPaths, stage: str) -> None:
    ArtifactState(work_paths).delete_stage_outputs(stage)


def _correction_complete(work_paths: WorkPaths) -> bool:
    if not work_paths.corrected_manifest.exists():
        return False
    transcripts = load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    eligible_count = sum(
        1 for item in transcripts
        if item.status == "completed" and item.text.strip()
    )
    report = read_json(work_paths.corrected_manifest, default=[])
    if not isinstance(report, list):
        return False
    completed_count = sum(
        1 for item in report
        if item.get("status", "completed") == "completed"
    )
    return eligible_count > 0 and completed_count >= eligible_count


def _read_completed_list_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default=[])
    if not isinstance(payload, list):
        return 0
    return sum(1 for item in payload if item.get("status", "completed") == "completed")


def _read_json_dict_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default={})
    return len(payload) if isinstance(payload, dict) else 0


def _count_translated_items(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return 0
    return sum(
        1
        for item in payload.values()
        if isinstance(item, dict) and str(item.get("translated_subtitle", "")).strip()
    )


def _dedupe_boundary_text(previous_text: str, current_text: str) -> str:
    previous_tokens = previous_text.split()
    current_tokens = current_text.split()
    max_overlap = min(len(previous_tokens), len(current_tokens), 20)
    for overlap in range(max_overlap, 0, -1):
        if previous_tokens[-overlap:] == current_tokens[:overlap]:
            return " ".join(current_tokens[overlap:]).strip()

    if previous_text and current_text.startswith(previous_text):
        return current_text[len(previous_text) :].strip()

    max_chars = min(len(previous_text), len(current_text), 30)
    for overlap in range(max_chars, 0, -1):
        if previous_text[-overlap:] == current_text[:overlap]:
            return current_text[overlap:].strip()

    return current_text


def _parse_json_object_argument(name: str, value: str | None) -> dict | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        import json

        parsed = json.loads(stripped)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise ValueError(f"{name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return parsed


def _append_manifest_event(path: Path, *, manifest_type: str, item: TranscriptSegment | AlignedSegment) -> None:
    append_jsonl(
        path,
        {
            "type": manifest_type,
            "segment_id": item.segment_id,
            "payload": serialize_manifest([item])[0],
        },
    )


def _write_manifest_checkpoint(path: Path, manifest: list[TranscriptSegment] | list[AlignedSegment]) -> None:
    write_checkpoint_json(path, serialize_manifest(manifest))


def _recover_manifest_from_checkpoint_and_events(checkpoint_path: Path, events_path: Path) -> list[dict]:
    payload = read_json(checkpoint_path, default=[])
    items = payload if isinstance(payload, list) else []
    indexed = {str(item.get("segment_id")): item for item in items if isinstance(item, dict)}
    for event in load_jsonl(events_path):
        if not isinstance(event, dict):
            continue
        segment_id = str(event.get("segment_id", "")).strip()
        row = event.get("payload")
        if segment_id and isinstance(row, dict):
            indexed[segment_id] = row
    return list(indexed.values())
