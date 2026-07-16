from __future__ import annotations

import argparse
import logging
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path

from qwen_asr.commands import transcribe_profile as _transcribe_profile
from qwen_asr.align import AlignTimingValidationConfig, QwenForcedAligner, validate_aligned_token_timing
from qwen_asr.artifact_state import ArtifactState
from qwen_asr.asr import ASRBatchOOMError, QwenASRTranscriber
from qwen_asr.audio import ensure_segment_audio, export_segment_audio, extract_audio, load_audio_metadata
from qwen_asr.batching import BatchPlanner
from qwen_asr.corrector import run_correction_stage
from qwen_asr.credentials import resolve_mimo_api_key
from qwen_asr.defaults import (
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_MODEL_CACHE_DIR,
)
from qwen_asr.final_quality import evaluate_final_quality
from qwen_asr.mimo_proofread import main as run_mimo_proofread
from qwen_asr.models import (
    AlignedSegment,
    AlignedToken,
    AudioSegment,
    TranscriptSegment,
    WorkPaths,
)
from qwen_asr.optimizer_bridge import (
    load_best_asr_data,
    load_specific_asr_data,
    run_split_stage,
    run_translate_stage,
)
from qwen_asr.pipeline_runner import PipelineRunner
from qwen_asr.preflight import ensure_preflight
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
from qwen_asr.vad import create_vad_adapter, derive_silence_regions

LOGGER = logging.getLogger(__name__)


def _percentile(sorted_values: list[float], ratio: float) -> float:
    return _transcribe_profile.percentile(sorted_values, ratio)


def _auto_select_transcribe_batch_defaults(segments: list[AudioSegment]) -> dict[str, object]:
    return _transcribe_profile.auto_select_transcribe_batch_defaults(segments)


def _resolve_transcribe_batch_defaults(args: argparse.Namespace, segments: list[AudioSegment]) -> dict[str, object]:
    return _transcribe_profile.resolve_transcribe_batch_defaults(args, segments)


def _resolve_model_cache_dir(args: argparse.Namespace) -> str:
    return _transcribe_profile.resolve_model_cache_dir(args, DEFAULT_MODEL_CACHE_DIR)


def _prepare_model_cache_dir(model_cache_dir: str, *, local_files_only: bool) -> None:
    _transcribe_profile.prepare_model_cache_dir(model_cache_dir, local_files_only=local_files_only)


def _consume_batch_memory_probes(transcriber: object) -> list[dict[str, object]]:
    return _transcribe_profile.consume_batch_memory_probes(transcriber)


def _write_transcribe_profile(
    work_paths: WorkPaths,
    args: argparse.Namespace,
    segments: list[AudioSegment],
    batch_reports: list[dict[str, object]],
    resolved_defaults: dict[str, object],
) -> None:
    _transcribe_profile.write_transcribe_profile(work_paths, args, segments, batch_reports, resolved_defaults)


def _build_transcribe_recommendation(
    args: argparse.Namespace,
    completed: list[dict[str, object]],
    oom_retries: list[dict[str, object]],
) -> dict[str, object]:
    return _transcribe_profile.build_transcribe_recommendation(args, completed, oom_retries)

def cmd_prepare(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.prepare import cmd_prepare as _cmd_prepare

    return _cmd_prepare(args, work_paths)


def cmd_transcribe(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.transcribe import cmd_transcribe as _cmd_transcribe

    return _cmd_transcribe(args, work_paths)


def cmd_align(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.align import cmd_align as _cmd_align

    return _cmd_align(args, work_paths)


def cmd_export(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.export import cmd_export as _cmd_export

    return _cmd_export(args, work_paths)


def cmd_normalize(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.normalize import cmd_normalize as _cmd_normalize

    return _cmd_normalize(args, work_paths)


def cmd_split(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.split import cmd_split as _cmd_split

    return _cmd_split(args, work_paths)


def cmd_translate(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.translate import cmd_translate as _cmd_translate

    return _cmd_translate(args, work_paths)


def cmd_correct(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.correct import cmd_correct as _cmd_correct

    return _cmd_correct(args, work_paths)


def cmd_run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.run import cmd_run as _cmd_run

    return _cmd_run(args, work_paths)


def cmd_mimo_proofread(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.mimo_proofread import cmd_mimo_proofread as _cmd_mimo_proofread

    return _cmd_mimo_proofread(args, work_paths)


def _validate_mimo_diagnostic_scope(args: argparse.Namespace) -> None:
    if (
        str(getattr(args, "mimo_audio_review_scope", "suspects") or "suspects") == "all"
        and not bool(getattr(args, "mimo_diagnostic_all", False))
    ):
        raise RuntimeError(
            "MiMo full audio review scope is diagnostic-only; pass --mimo-diagnostic-all with "
            "--mimo-audio-review-scope all for an explicit diagnostic experiment."
        )


def cmd_content_quality(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.quality import cmd_content_quality as _cmd_content_quality

    return _cmd_content_quality(args, work_paths)


def cmd_quality_gate(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.quality import cmd_quality_gate as _cmd_quality_gate

    return _cmd_quality_gate(args, work_paths)


def _require_quality_gate_before_formal_output(work_paths: WorkPaths) -> int:
    report = evaluate_final_quality(work_paths, include_export=False, require_srt=False)
    status = str(report.get("status", "WARN") or "WARN")
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    write_progress(
        work_paths,
        stage="quality-gate",
        status="completed" if status != "FAIL" else "failed",
        done=int(summary.get("pass_count", 0) or 0),
        total=(
            int(summary.get("pass_count", 0) or 0)
            + int(summary.get("warn_count", 0) or 0)
            + int(summary.get("fail_count", 0) or 0)
        ),
        summary=(
            f"聚合质量门 {status}："
            f"{summary.get('fail_count', 0)} FAIL，{summary.get('warn_count', 0)} WARN"
        ),
    )
    return 0 if status != "FAIL" else 1


def cmd_proofread_realign(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.proofread_realign import cmd_proofread_realign as _cmd_proofread_realign

    return _cmd_proofread_realign(args, work_paths)


def _translated_manifest_has_content(path: Path) -> bool:
    payload = read_json(path, default={})
    return isinstance(payload, dict) and bool(payload) and all(
        str(item.get("translated_subtitle", "")).strip() for item in payload.values() if isinstance(item, dict)
    )


def cmd_preflight(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands.preflight import cmd_preflight as _cmd_preflight

    return _cmd_preflight(args, work_paths)


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
    if source in {"normalized", "mimo", "translated", "split", "transcript"}:
        return load_specific_asr_data(work_paths, source=source, optimizer_root=optimizer_root)
    return load_best_asr_data(work_paths, optimizer_root=optimizer_root)


def _load_normalize_source(source: str, work_paths: WorkPaths, optimizer_root: Path):
    if source == "auto":
        for candidate in ("mimo", "translated", "split", "transcript"):
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


def _translation_manifest_complete_for_split(work_paths: WorkPaths) -> bool:
    payload = read_json(work_paths.translated_manifest, default={})
    if not isinstance(payload, dict) or not payload:
        return False
    split_payload = read_json(work_paths.split_manifest, default={})
    if isinstance(split_payload, dict) and split_payload:
        expected_keys = {str(key) for key in split_payload.keys()}
        if not expected_keys.issubset({str(key) for key in payload.keys()}):
            return False
        items = [payload.get(key) for key in expected_keys]
    else:
        items = list(payload.values())
    if not items or not all(isinstance(item, dict) for item in items):
        return False
    return all(str(item.get("translated_subtitle", "")).strip() for item in items)


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
