from __future__ import annotations

import argparse
import logging

from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress

LOGGER = logging.getLogger(__name__)


def cmd_prepare(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    stage_helpers.ensure_preflight(args, work_paths, "prepare")
    stage_helpers.ensure_directory(work_paths.workdir)
    stage_helpers.ensure_directory(work_paths.segments_dir)
    stage_helpers.ensure_directory(work_paths.logs_dir)

    if args.force:
        stage_helpers._clear_prepare_outputs(work_paths)

    media_path = stage_helpers._resolve_media_path(args)
    if not media_path.exists():
        raise FileNotFoundError(f"Media not found: {media_path}")
    stage_helpers._write_project_metadata(args, work_paths, media_path)

    audio_path = stage_helpers.extract_audio(
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
    _, duration = stage_helpers.load_audio_metadata(audio_path)

    vad = stage_helpers.create_vad_adapter(
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
    silence_regions = stage_helpers.derive_silence_regions(speech_regions, duration)
    config = stage_helpers.SegmenterConfig(
        max_segment_seconds=args.max_segment_seconds,
        min_segment_seconds=args.min_segment_seconds,
        preferred_silence_ms=args.preferred_silence_ms,
        min_silence_ms=args.min_silence_ms,
        padding_ms=args.padding_ms,
        overlap_ms=args.overlap_ms,
    )
    segments = stage_helpers.build_segments(
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
            stage_helpers.export_segment_audio(audio_path, segment)
        segment.status = "prepared"
    stage_helpers.write_json_atomic(work_paths.segments_manifest, stage_helpers.serialize_manifest(segments))
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
