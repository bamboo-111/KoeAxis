from __future__ import annotations

import argparse
import logging
import unicodedata
import wave
from pathlib import Path

from qwen_asr.alignment_state import derive_alignment_state
from qwen_asr.models import AlignedSegment, AlignedToken, AudioSegment, TranscriptSegment, WorkPaths
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


def cmd_align(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    stage_helpers.ensure_preflight(args, work_paths, "align")
    cleanup_interval = max(1, int(getattr(args, "cleanup_interval", 4)))
    model_cache_dir = stage_helpers._resolve_model_cache_dir(args)
    if args.force:
        stage_helpers._clear_align_outputs(work_paths)

    transcripts = stage_helpers.load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    if not transcripts:
        raise RuntimeError("transcript_segments.json is missing or empty. Run transcribe first.")

    existing = {} if args.force else {
        item.segment_id: item
        for item in stage_helpers.load_aligned_segments(
            work_paths.aligned_manifest,
            work_paths.aligned_checkpoint_path,
            work_paths.aligned_events_path,
        )
    }
    eligible_transcripts = [item for item in transcripts if item.status == "completed" and item.text.strip()]
    if args.resume and not args.force and eligible_transcripts and all(
        item.segment_id in existing and existing[item.segment_id].status == "completed"
        and getattr(existing[item.segment_id], "alignment_backend", "qwen") == "qwen"
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

    aligner = stage_helpers.QwenForcedAligner(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        keep_raw_model_output=args.keep_raw_model_output
        or getattr(args, "align_diagnostics_mode", "off") == "capture-failed",
        keep_failed_tokens=getattr(args, "align_diagnostics_mode", "off") == "capture-failed",
        model_cache_dir=model_cache_dir,
        local_files_only=args.local_files_only,
        timing_validation_config=stage_helpers.AlignTimingValidationConfig(
            min_coverage_ratio=float(getattr(args, "align_min_coverage_ratio", 0.2)),
            bad_zero_run=int(getattr(args, "align_max_zero_run", 8)),
            dense_zero_ratio=float(getattr(args, "align_dense_zero_ratio", 0.5)),
            min_dense_coverage_ratio=float(getattr(args, "align_min_dense_coverage_ratio", 0.5)),
            local_collapse_min_chars=int(getattr(args, "align_local_collapse_min_chars", 8)),
            local_collapse_max_duration=float(getattr(args, "align_local_collapse_max_duration_ms", 500)) / 1000.0,
            local_collapse_max_cps=float(getattr(args, "align_local_collapse_max_cps", 35.0)),
            local_collapse_max_tokens=int(getattr(args, "align_local_collapse_max_tokens", 12)),
        ),
    )
    aligner.load()
    asr_reference_transcriber = None
    try:
        attempted = 0
        for transcript in transcripts:
            if transcript.status != "completed" or not transcript.text.strip():
                LOGGER.warning("Skipping transcript without usable text: %s", transcript.segment_id)
                continue
            transcript_audio_path = Path(transcript.audio_path)
            if not transcript_audio_path.exists() and work_paths.audio_path.exists():
                stage_helpers.ensure_segment_audio(work_paths.audio_path, AudioSegment(
                    segment_id=transcript.segment_id,
                    audio_path=transcript.audio_path,
                    source_audio_path=str(work_paths.audio_path),
                    global_start_time=transcript.global_start_time,
                    global_end_time=transcript.global_end_time,
                    duration=max(0.0, transcript.global_end_time - transcript.global_start_time),
                ))
            existing_item = existing.get(transcript.segment_id)
            if stage_helpers._should_skip(existing_item, args):
                LOGGER.info("Skipping completed alignment %s", transcript.segment_id)
                continue
            attempted += 1
            cleanup_now = attempted % cleanup_interval == 0
            result = aligner.run_segment(transcript, cleanup=cleanup_now)
            if result.status == "failed" and getattr(args, "align_fallback", "off") == "asr-short-window":
                asr_reference_transcriber = _ensure_align_fallback_asr_transcriber(
                    args,
                    model_cache_dir,
                    asr_reference_transcriber,
                )
                result = _run_asr_short_window_align_fallback(
                    args=args,
                    work_paths=work_paths,
                    transcript=transcript,
                    original_result=result,
                    aligner=aligner,
                    transcriber=asr_reference_transcriber,
                )
            existing[result.segment_id] = result
            manifest = stage_helpers._ordered_aligned(transcripts, existing)
            stage_helpers._append_manifest_event(
                work_paths.aligned_events_path,
                manifest_type="aligned",
                item=result,
            )
            stage_helpers._write_manifest_checkpoint(
                work_paths.aligned_checkpoint_path,
                manifest,
            )
            stage_helpers.write_json_atomic(work_paths.aligned_manifest, stage_helpers.serialize_manifest(manifest))
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
        if asr_reference_transcriber is not None:
            asr_reference_transcriber.close()
        aligner.close()

    manifest = stage_helpers._ordered_aligned(transcripts, existing)
    return _align_status_code(manifest, eligible_transcripts)


def _align_status_code(manifest: list[AlignedSegment], eligible_transcripts: list[TranscriptSegment]) -> int:
    completed = {
        item.segment_id
        for item in manifest
        if derive_alignment_state(item) in {"completed_exact", "completed_coarse"}
    }
    eligible = {item.segment_id for item in eligible_transcripts}
    return 0 if eligible and eligible.issubset(completed) else 1


def _ensure_align_fallback_asr_transcriber(
    args: argparse.Namespace,
    model_cache_dir: str | None,
    transcriber,
):
    from qwen_asr.commands import stages as stage_helpers

    if transcriber is not None:
        return transcriber
    fallback = stage_helpers.QwenASRTranscriber(
        model_name=getattr(args, "asr_reference_model", None) or stage_helpers.DEFAULT_ASR_MODEL,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        max_new_tokens=int(getattr(args, "asr_reference_max_new_tokens", 512)),
        language=getattr(args, "asr_reference_language", None),
        keep_raw_model_output=False,
        model_cache_dir=model_cache_dir,
        local_files_only=args.local_files_only,
        batch_size=1,
    )
    fallback.load()
    return fallback


def _run_asr_short_window_align_fallback(
    *,
    args: argparse.Namespace,
    work_paths: WorkPaths,
    transcript: TranscriptSegment,
    original_result: AlignedSegment,
    aligner,
    transcriber,
) -> AlignedSegment:
    from qwen_asr.commands import stages as stage_helpers

    window_seconds = max(0.5, float(getattr(args, "align_fallback_window_seconds", 3.0)))
    source_audio = Path(transcript.audio_path)
    output_dir = stage_helpers.ensure_directory(
        work_paths.workdir / "diagnostics" / "align-fallback" / "asr-short-window" / transcript.segment_id
    )
    windows = _export_align_fallback_windows(source_audio, transcript, output_dir, window_seconds)
    fallback_rows: list[dict[str, object]] = []
    completed_alignments: list[AlignedSegment] = []
    for window in windows:
        asr_result = transcriber.run_segment(window, cleanup=False)
        row: dict[str, object] = {
            "window_segment_id": window.segment_id,
            "audio_path": window.audio_path,
            "global_start_time": window.global_start_time,
            "global_end_time": window.global_end_time,
            "asr_status": asr_result.status,
            "asr_text": asr_result.text,
            "asr_error": asr_result.error,
        }
        if asr_result.status != "completed" or not asr_result.text.strip():
            row["align_status"] = "skipped"
            row["align_error"] = "empty ASR reference text"
            fallback_rows.append(row)
            continue
        window_transcript = TranscriptSegment(
            segment_id=window.segment_id,
            audio_path=window.audio_path,
            global_start_time=window.global_start_time,
            global_end_time=window.global_end_time,
            text=asr_result.text,
            language=asr_result.language or transcript.language,
            status="completed",
        )
        alignment = aligner.run_segment(window_transcript, cleanup=False)
        row["align_status"] = alignment.status
        row["align_error"] = alignment.error
        row["token_count"] = len(alignment.tokens)
        fallback_rows.append(row)
        if alignment.status == "completed" and alignment.tokens:
            completed_alignments.append(alignment)
    _cleanup_align_fallback_torch()

    tokens = sorted(
        [token for alignment in completed_alignments for token in alignment.tokens],
        key=lambda token: (token.start_time, token.end_time),
    )
    merged_text = "".join(str(row.get("asr_text") or "").strip() for row in fallback_rows)
    content_error = _validate_fallback_alignment_content(transcript.text, tokens)
    metadata = {
        "fallback": "asr-short-window",
        "original_error": original_result.error,
        "window_seconds": window_seconds,
        "window_count": len(windows),
        "completed_window_alignments": len(completed_alignments),
        "merged_asr_text": merged_text,
        "content_error": content_error,
        "windows": fallback_rows,
    }
    timing_error = stage_helpers.validate_aligned_token_timing(
        tokens,
        transcript.global_start_time,
        transcript.global_end_time,
    )
    if not tokens or timing_error or content_error:
        return AlignedSegment(
            segment_id=transcript.segment_id,
            audio_path=transcript.audio_path,
            global_start_time=transcript.global_start_time,
            global_end_time=transcript.global_end_time,
            text=transcript.text,
            language=transcript.language,
            tokens=original_result.tokens,
            raw_model_output=_merge_fallback_metadata(original_result.raw_model_output, metadata),
            status="failed",
            error=(
                content_error
                or timing_error
                or original_result.error
                or "ASR short-window fallback produced no aligned tokens"
            ),
        )
    return AlignedSegment(
        segment_id=transcript.segment_id,
        audio_path=transcript.audio_path,
        global_start_time=transcript.global_start_time,
        global_end_time=transcript.global_end_time,
        text=transcript.text,
        language=transcript.language,
        tokens=tokens,
        raw_model_output=_merge_fallback_metadata(original_result.raw_model_output, metadata),
        status="completed",
    )


def _validate_fallback_alignment_content(
    transcript_text: str,
    tokens: list[AlignedToken],
) -> str | None:
    reference = _normalize_alignment_content(transcript_text)
    aligned = _normalize_alignment_content("".join(token.text for token in tokens))
    if not reference:
        return None
    if aligned != reference:
        return (
            "ASR short-window fallback changed transcript content: "
            f"reference_chars={len(reference)} aligned_chars={len(aligned)}"
        )
    return None


def _normalize_alignment_content(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _export_align_fallback_windows(
    audio_path: Path,
    transcript: TranscriptSegment,
    output_dir: Path,
    window_seconds: float,
) -> list[AudioSegment]:
    from qwen_asr.commands import stages as stage_helpers

    stage_helpers.ensure_directory(output_dir)
    with wave.open(str(audio_path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        total_frames = source.getnframes()
        total_duration = total_frames / sample_rate if sample_rate else 0.0
        windows: list[AudioSegment] = []
        start = 0.0
        index = 1
        while start < total_duration:
            end = min(total_duration, start + window_seconds)
            start_frame = int(round(start * sample_rate))
            end_frame = int(round(end * sample_rate))
            source.setpos(start_frame)
            frames = source.readframes(max(0, end_frame - start_frame))
            window_path = output_dir / f"{transcript.segment_id}-w{index:02d}.wav"
            with wave.open(str(window_path), "wb") as target:
                target.setnchannels(channels)
                target.setsampwidth(sample_width)
                target.setframerate(sample_rate)
                target.writeframes(frames)
            windows.append(
                AudioSegment(
                    segment_id=f"{transcript.segment_id}-w{index:02d}",
                    audio_path=str(window_path),
                    source_audio_path=str(audio_path),
                    global_start_time=round(transcript.global_start_time + start, 3),
                    global_end_time=round(transcript.global_start_time + end, 3),
                    duration=round(max(0.0, end - start), 3),
                    status="completed",
                )
            )
            start = end
            index += 1
    return windows


def _merge_fallback_metadata(raw_model_output: object, metadata: dict[str, object]) -> dict[str, object]:
    if isinstance(raw_model_output, dict):
        return {**raw_model_output, "align_fallback": metadata}
    return {"raw_model_output": raw_model_output, "align_fallback": metadata}


def _cleanup_align_fallback_torch() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
