from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress
from qwen_asr.storage import ensure_directory, read_json, write_json_atomic
from qwen_asr.subtitle import (
    SubtitleConfig,
    build_coarse_cues_from_transcripts,
    build_cues_from_aligned_segments,
    export_srt,
    export_vtt,
    export_vtt_from_optimizer_asr_data,
)

LOGGER = logging.getLogger(__name__)


def cmd_export(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    if args.force:
        stage_helpers._clear_export_outputs(work_paths)
    gate_status = stage_helpers._require_quality_gate_before_formal_output(work_paths)
    if gate_status != 0:
        return gate_status

    config = SubtitleConfig(
        max_subtitle_duration=args.max_subtitle_duration,
        min_subtitle_duration=args.min_subtitle_duration,
        max_chars_per_line_zh=args.max_chars_per_line_zh,
        max_chars_per_line_en=args.max_chars_per_line_en,
        max_lines=args.max_lines,
        pause_split_seconds=args.pause_split_seconds,
    )

    transcripts = stage_helpers.load_transcripts(
        work_paths.transcript_manifest,
        work_paths.transcript_checkpoint_path,
        work_paths.transcript_events_path,
    )
    stage_helpers.write_transcript_text(work_paths.transcript_text, transcripts)

    optimizer_root = Path(args.optimizer_root)
    optimizer_asr_data = None
    if args.source in {"auto", "normalized", "mimo", "translated", "split", "transcript"}:
        optimizer_asr_data = stage_helpers._load_optimizer_export_source(args.source, work_paths, optimizer_root)

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

    aligned = (
        stage_helpers.load_aligned_segments(
            work_paths.aligned_manifest,
            work_paths.aligned_checkpoint_path,
            work_paths.aligned_events_path,
        )
        if args.source in {"auto", "aligned"}
        else []
    )
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


def _resolve_export_media_path(args: argparse.Namespace, work_paths: WorkPaths) -> Path:
    media = getattr(args, "media_path", None) or getattr(args, "media", None) or getattr(args, "video", None)
    if media:
        return Path(media).resolve()
    metadata = _load_project_metadata(work_paths)
    original = str(metadata.get("original_media_path", "")).strip()
    if original:
        return Path(original).resolve()
    raise RuntimeError("Export source mode requires project.json original_media_path or --media-path.")


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
        if not _same_file_path(cache_path, target):
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


def _same_file_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.absolute() == right.absolute()


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
