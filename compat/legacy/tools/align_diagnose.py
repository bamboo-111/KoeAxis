from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import wave
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from qwen_asr.align import QwenForcedAligner, _cleanup_torch, _sanitize_raw_output, validate_aligned_token_timing
from qwen_asr.asr import QwenASRTranscriber
from qwen_asr.defaults import DEFAULT_ALIGN_MODEL
from qwen_asr.models import AlignedToken, AudioSegment, TranscriptSegment, WorkPaths
from qwen_asr.storage import ensure_directory, read_json

SILENCE_RMS_THRESHOLD = 0.003
LOW_RMS_THRESHOLD = 0.01
CLIPPING_THRESHOLD = 0.98


def cmd_align_diagnose(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    transcripts = _load_transcript_map(work_paths)
    selected_ids = _select_segment_ids(args, work_paths, transcripts)
    output_dir = _make_output_dir(work_paths)
    ensure_directory(output_dir)

    plan_rows = _build_experiment_plan(work_paths, transcripts, selected_ids, args.text_mode)
    _write_tsv(output_dir / "experiment_plan.tsv", plan_rows)
    if args.dry_run_plan:
        print(f"Align diagnose plan: {output_dir}")
        return 0

    raw_dir = ensure_directory(output_dir / "raw_outputs")
    token_dir = ensure_directory(output_dir / "tokens")
    asr_window_dir = ensure_directory(output_dir / "asr_windows") if args.with_asr_reference else None
    run_rows: list[dict[str, Any]] = []
    audio_rows: list[dict[str, Any]] = []
    asr_reference_rows: list[dict[str, Any]] = []

    aligner = QwenForcedAligner(
        model_name=args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        keep_raw_model_output=True,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
    )
    asr_transcriber = _load_asr_transcriber(args) if args.with_asr_reference else None
    aligner.load()
    try:
        for segment_id in selected_ids:
            transcript = transcripts[segment_id]
            audio_path = _resolve_audio_path(work_paths, transcript.audio_path)
            audio_metrics = collect_audio_metrics(audio_path)
            asr_reference = _run_asr_reference(
                transcriber=asr_transcriber,
                transcript=transcript,
                audio_path=audio_path,
                output_dir=asr_window_dir,
                window_seconds=args.asr_window_seconds,
            ) if asr_transcriber is not None and asr_window_dir is not None else None
            if asr_reference is not None:
                asr_reference_rows.append(_flatten_asr_reference(transcript.segment_id, asr_reference))
            audio_rows.append({"segment_id": segment_id, "audio_path": str(audio_path), **audio_metrics})
            text = _resolve_text(transcript.text, args.text_mode)
            for repeat_index in range(max(1, int(args.repeat))):
                run_id = f"{segment_id}-r{repeat_index + 1:02d}"
                row = _run_diagnostic_alignment(
                    aligner=aligner,
                    transcript=transcript,
                    audio_path=audio_path,
                    text=text,
                    text_mode=args.text_mode,
                    run_id=run_id,
                    repeat_index=repeat_index + 1,
                    raw_dir=raw_dir,
                    token_dir=token_dir,
                    model_args=args,
                    audio_metrics=audio_metrics,
                    asr_reference=asr_reference,
                )
                run_rows.append(row)
    finally:
        aligner.close()
        if asr_transcriber is not None:
            asr_transcriber.close()

    _write_jsonl(output_dir / "diagnose_runs.jsonl", run_rows)
    _write_tsv(output_dir / "audio_metrics.tsv", audio_rows)
    if asr_reference_rows:
        _write_tsv(output_dir / "asr_reference.tsv", asr_reference_rows)
    _write_tsv(output_dir / "summary.tsv", _build_summary_rows(run_rows))
    print(f"Align diagnose output: {output_dir}")
    return 0 if run_rows else 1


def _run_diagnostic_alignment(
    *,
    aligner: QwenForcedAligner,
    transcript: TranscriptSegment,
    audio_path: Path,
    text: str,
    text_mode: str,
    run_id: str,
    repeat_index: int,
    raw_dir: Path,
    token_dir: Path,
    model_args: argparse.Namespace,
    audio_metrics: dict[str, Any],
    asr_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_output: Any = None
    raw_error: str | None = None
    try:
        raw_output = aligner._model.align(  # noqa: SLF001
            audio=str(audio_path),
            text=text,
            language=transcript.language,
        )
    except Exception as exc:  # pragma: no cover - covered through integration behavior
        raw_error = str(exc)

    sanitized_raw = _sanitize_raw_output(raw_output)
    raw_path = raw_dir / f"{run_id}.json"
    raw_path.write_text(json.dumps(sanitized_raw, ensure_ascii=False, indent=2), encoding="utf-8")

    extraction = extract_token_diagnostics(raw_output, transcript.global_start_time)
    token_path = token_dir / f"{run_id}.tokens.tsv"
    _write_tsv(token_path, extraction["token_rows"])
    tokens = [
        AlignedToken(
            text=str(row["text"]),
            start_time=float(row["offset_start"]),
            end_time=float(row["offset_end"]),
        )
        for row in extraction["token_rows"]
    ]
    quality = compute_quality_metrics(tokens, transcript.global_start_time, transcript.global_end_time)
    if raw_error and not quality["quality_error"]:
        quality["quality_error"] = raw_error

    input_snapshot = {
        "segment_id": transcript.segment_id,
        "audio_path": str(audio_path),
        "global_start_time": transcript.global_start_time,
        "global_end_time": transcript.global_end_time,
        "segment_duration": max(0.0, transcript.global_end_time - transcript.global_start_time),
        "text": text,
        "text_mode": text_mode,
        "char_count": len(text),
        "chars_per_second": _safe_ratio(len(text), transcript.global_end_time - transcript.global_start_time),
        "language": transcript.language,
        "model": model_args.model,
        "dtype": model_args.dtype,
        "device": model_args.device,
        "attn_implementation": model_args.attn_implementation,
        "local_files_only": model_args.local_files_only,
        **_input_audio_fields(audio_metrics),
    }
    classification = classify_run(quality, audio_metrics, input_snapshot)
    return {
        "run_id": run_id,
        "segment_id": transcript.segment_id,
        "repeat_index": repeat_index,
        "experiment_type": "A",
        "input_snapshot": input_snapshot,
        "raw_output_snapshot": {"path": str(raw_path), "error": raw_error},
        "token_extraction_snapshot": extraction["snapshot"],
        "quality_metrics": quality,
        "audio_metrics": audio_metrics,
        "asr_reference": asr_reference,
        "classification": classification,
        "quality_failed": bool(quality["quality_error"]),
        "tokens_path": str(token_path),
    }


def extract_token_diagnostics(raw_output: Any, global_offset: float) -> dict[str, Any]:
    token_rows, source = _find_token_rows(raw_output)
    output_rows: list[dict[str, Any]] = []
    for index, item in enumerate(token_rows or []):
        text, raw_start, raw_end = _coerce_token_row(item)
        offset_start = round(global_offset + raw_start, 3)
        offset_end = round(global_offset + raw_end, 3)
        output_rows.append(
            {
                "index": index,
                "text": text,
                "raw_start": round(raw_start, 3),
                "raw_end": round(raw_end, 3),
                "raw_duration": round(max(0.0, raw_end - raw_start), 3),
                "offset_start": offset_start,
                "offset_end": offset_end,
                "offset_duration": round(max(0.0, offset_end - offset_start), 3),
                "zero_duration": offset_end <= offset_start,
            }
        )
    positive = [row for row in output_rows if not row["zero_duration"]]
    snapshot = {
        "token_source": source,
        "token_count": len(output_rows),
        "positive_token_count": len(positive),
        "zero_duration_count": len(output_rows) - len(positive),
        "raw_time_range": _time_range(output_rows, "raw_start", "raw_end"),
        "offset_time_range": _time_range(output_rows, "offset_start", "offset_end"),
    }
    return {"snapshot": snapshot, "token_rows": output_rows}


def compute_quality_metrics(
    tokens: list[AlignedToken],
    global_start_time: float,
    global_end_time: float,
) -> dict[str, Any]:
    zero_count = 0
    zero_run = 0
    max_zero_run = 0
    positive_start: float | None = None
    positive_end: float | None = None
    previous_start: float | None = None
    monotonic_violation_index: int | None = None

    for index, token in enumerate(tokens):
        if token.end_time <= token.start_time:
            zero_count += 1
            zero_run += 1
            max_zero_run = max(max_zero_run, zero_run)
        else:
            zero_run = 0
            positive_start = token.start_time if positive_start is None else min(positive_start, token.start_time)
            positive_end = token.end_time if positive_end is None else max(positive_end, token.end_time)
        if previous_start is not None and token.start_time < previous_start and monotonic_violation_index is None:
            monotonic_violation_index = index
        previous_start = token.start_time

    segment_duration = max(0.0, global_end_time - global_start_time)
    covered_duration = max(0.0, (positive_end or global_start_time) - (positive_start or global_start_time))
    return {
        "quality_error": validate_aligned_token_timing(tokens, global_start_time, global_end_time),
        "segment_duration": round(segment_duration, 3),
        "covered_duration": round(covered_duration, 3),
        "coverage_ratio": round(_safe_ratio(covered_duration, segment_duration), 4),
        "token_count": len(tokens),
        "positive_token_count": len(tokens) - zero_count,
        "zero_duration_count": zero_count,
        "zero_ratio": round(_safe_ratio(zero_count, len(tokens)), 4),
        "positive_token_ratio": round(_safe_ratio(len(tokens) - zero_count, len(tokens)), 4),
        "max_zero_run": max_zero_run,
        "local_max_cps": round(_local_max_cps(tokens), 3),
        "monotonic_violation_index": monotonic_violation_index,
    }


def collect_audio_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"audio_error": "audio file missing"}
    try:
        with wave.open(str(path), "rb") as handle:
            frame_count = handle.getnframes()
            sample_rate = handle.getframerate()
            channel_count = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(frame_count)
    except wave.Error as exc:
        return {"audio_error": str(exc)}

    base = {
        "audio_error": None,
        "wav_duration": round(_safe_ratio(frame_count, sample_rate), 3),
        "sample_rate": sample_rate,
        "channels": channel_count,
        "sample_width_bytes": sample_width,
    }
    if sample_width != 2:
        return {**base, "metrics_error": "only 16-bit PCM audio metrics are supported"}

    sample_count = len(frames) // 2
    if sample_count == 0:
        return {**base, "metrics_error": "empty audio"}

    import struct

    values = list(struct.unpack(f"<{sample_count}h", frames))
    if channel_count > 1:
        values = values[::channel_count]
    rms_values = _window_rms(values, sample_rate)
    abs_values = [abs(value) for value in values]
    full_rms = math.sqrt(sum(value * value for value in values) / max(1, len(values))) / 32768.0
    peak = max(abs_values, default=0) / 32768.0
    clipping_count = sum(1 for value in abs_values if value / 32768.0 >= CLIPPING_THRESHOLD)
    silence_flags = [value < SILENCE_RMS_THRESHOLD for value in rms_values]
    low_flags = [value < LOW_RMS_THRESHOLD for value in rms_values]
    return {
        **base,
        "metrics_error": None,
        "rms": round(full_rms, 6),
        "rms_min": round(min(rms_values, default=0.0), 6),
        "rms_median": round(statistics.median(rms_values) if rms_values else 0.0, 6),
        "rms_p90": round(_percentile(rms_values, 0.9), 6),
        "rms_max": round(max(rms_values, default=0.0), 6),
        "peak": round(peak, 6),
        "clipping_ratio": round(_safe_ratio(clipping_count, len(abs_values)), 6),
        "silence_ratio_100ms": round(_safe_ratio(sum(silence_flags), len(silence_flags)), 4),
        "low_energy_ratio_100ms": round(_safe_ratio(sum(low_flags), len(low_flags)), 4),
        "leading_low_energy_seconds": round(_edge_duration(low_flags, from_start=True), 3),
        "trailing_low_energy_seconds": round(_edge_duration(low_flags, from_start=False), 3),
        "longest_low_energy_seconds": round(_longest_true_run(low_flags) * 0.1, 3),
    }


def _load_asr_transcriber(args: argparse.Namespace) -> QwenASRTranscriber:
    transcriber = QwenASRTranscriber(
        model_name=args.asr_model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.asr_max_new_tokens,
        language=args.asr_language,
        keep_raw_model_output=False,
        model_cache_dir=args.model_cache_dir,
        local_files_only=args.local_files_only,
        batch_size=1,
    )
    transcriber.load()
    return transcriber


def _run_asr_reference(
    *,
    transcriber: QwenASRTranscriber | None,
    transcript: TranscriptSegment,
    audio_path: Path,
    output_dir: Path | None,
    window_seconds: float,
) -> dict[str, Any]:
    if transcriber is None or output_dir is None:
        return {}
    original_segment = AudioSegment(
        segment_id=f"{transcript.segment_id}-asr-original",
        audio_path=str(audio_path),
        source_audio_path=str(audio_path),
        global_start_time=transcript.global_start_time,
        global_end_time=transcript.global_end_time,
        duration=max(0.0, transcript.global_end_time - transcript.global_start_time),
        status="completed",
    )
    original = transcriber.run_segment(original_segment, cleanup=False)
    windows = _export_fixed_windows(audio_path, transcript, output_dir, max(0.5, float(window_seconds)))
    window_results = [transcriber.run_segment(window, cleanup=False) for window in windows]
    _cleanup_torch()

    manifest_text = transcript.text.strip()
    original_text = original.text.strip()
    window_text = "".join(item.text.strip() for item in window_results)
    return {
        "original": {
            "status": original.status,
            "text": original_text,
            "char_count": len(original_text),
            "chars_per_second": round(_safe_ratio(len(original_text), original_segment.duration), 4),
            "similarity_to_manifest": round(_text_similarity(manifest_text, original_text), 4),
            "error": original.error,
        },
        "windows": [
            {
                "segment_id": item.segment_id,
                "audio_path": item.audio_path,
                "global_start_time": item.global_start_time,
                "global_end_time": item.global_end_time,
                "duration": item.duration,
                "status": result.status,
                "text": result.text,
                "char_count": len(result.text),
                "chars_per_second": round(_safe_ratio(len(result.text), item.duration), 4),
                "error": result.error,
            }
            for item, result in zip(windows, window_results, strict=True)
        ],
        "window_merged": {
            "text": window_text,
            "char_count": len(window_text),
            "chars_per_second": round(
                _safe_ratio(len(window_text), max(0.0, transcript.global_end_time - transcript.global_start_time)),
                4,
            ),
            "similarity_to_manifest": round(_text_similarity(manifest_text, window_text), 4),
            "completed_window_count": sum(1 for item in window_results if item.status == "completed"),
            "window_count": len(window_results),
        },
        "manifest": {
            "text": manifest_text,
            "char_count": len(manifest_text),
            "chars_per_second": round(
                _safe_ratio(len(manifest_text), max(0.0, transcript.global_end_time - transcript.global_start_time)),
                4,
            ),
        },
    }


def _export_fixed_windows(
    audio_path: Path,
    transcript: TranscriptSegment,
    output_dir: Path,
    window_seconds: float,
) -> list[AudioSegment]:
    ensure_directory(output_dir)
    with wave.open(str(audio_path), "rb") as source:
        sample_rate = source.getframerate()
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        total_frames = source.getnframes()
        total_duration = _safe_ratio(total_frames, sample_rate)
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


def _flatten_asr_reference(segment_id: str, reference: dict[str, Any]) -> dict[str, Any]:
    original = reference.get("original") or {}
    merged = reference.get("window_merged") or {}
    manifest = reference.get("manifest") or {}
    return {
        "segment_id": segment_id,
        "manifest_chars": manifest.get("char_count"),
        "manifest_cps": manifest.get("chars_per_second"),
        "original_status": original.get("status"),
        "original_chars": original.get("char_count"),
        "original_cps": original.get("chars_per_second"),
        "original_similarity_to_manifest": original.get("similarity_to_manifest"),
        "window_count": merged.get("window_count"),
        "completed_window_count": merged.get("completed_window_count"),
        "window_merged_chars": merged.get("char_count"),
        "window_merged_cps": merged.get("chars_per_second"),
        "window_merged_similarity_to_manifest": merged.get("similarity_to_manifest"),
    }


def classify_run(quality: dict[str, Any], audio_metrics: dict[str, Any], input_snapshot: dict[str, Any]) -> str:
    if not quality.get("quality_error"):
        return "ok"
    if quality.get("monotonic_violation_index") is not None:
        return "adapter_mismatch"
    if float(audio_metrics.get("low_energy_ratio_100ms") or 0.0) >= 0.6:
        return "window_context"
    if (
        float(audio_metrics.get("peak") or 0.0) >= 0.8
        and int(quality.get("max_zero_run") or 0) > 0
    ):
        return "audio_chain"
    if float(input_snapshot.get("chars_per_second") or 0.0) <= 1.0:
        return "text_mismatch"
    if int(quality.get("max_zero_run") or 0) > 0 or float(quality.get("local_max_cps") or 0.0) >= 35.0:
        return "model_or_audio_collapse"
    return "unknown"


def _load_transcript_map(work_paths: WorkPaths) -> dict[str, TranscriptSegment]:
    payload = read_json(work_paths.transcript_manifest, default=[])
    return {str(item["segment_id"]): TranscriptSegment(**item) for item in payload}


def _select_segment_ids(
    args: argparse.Namespace,
    work_paths: WorkPaths,
    transcripts: dict[str, TranscriptSegment],
) -> list[str]:
    requested = _parse_segment_ids(args.segments)
    if requested:
        return [segment_id for segment_id in requested if segment_id in transcripts]
    ranked = _rank_segments(work_paths, transcripts)
    return ranked[: max(1, int(args.sample_size))]


def _rank_segments(work_paths: WorkPaths, transcripts: dict[str, TranscriptSegment]) -> list[str]:
    aligned_payload = read_json(work_paths.aligned_manifest, default=[])
    if not aligned_payload:
        return list(transcripts)
    rows: list[tuple[int, float, str]] = []
    for item in aligned_payload:
        segment_id = str(item.get("segment_id", ""))
        if segment_id not in transcripts:
            continue
        duration = max(0.0, float(item.get("global_end_time", 0.0)) - float(item.get("global_start_time", 0.0)))
        text = str(item.get("text") or "")
        failed = item.get("status") != "completed" or not item.get("tokens")
        low_cps = _safe_ratio(len(text), duration) <= 1.0
        score = (100 if failed else 0) + (10 if low_cps else 0)
        rows.append((-score, duration, segment_id))
    return [segment_id for _, _, segment_id in sorted(rows)]


def _build_experiment_plan(
    work_paths: WorkPaths,
    transcripts: dict[str, TranscriptSegment],
    selected_ids: list[str],
    text_mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for segment_id in selected_ids:
        transcript = transcripts[segment_id]
        rows.append(
            {
                "experiment_type": "A",
                "workdir": str(work_paths.workdir),
                "segment_id": segment_id,
                "audio_path": str(_resolve_audio_path(work_paths, transcript.audio_path)),
                "text_mode": text_mode,
                "global_start_time": transcript.global_start_time,
                "global_end_time": transcript.global_end_time,
                "text_chars": len(transcript.text),
            }
        )
    return rows


def _build_summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter((row["experiment_type"], row["classification"]) for row in run_rows)
    return [
        {"experiment_type": experiment_type, "classification": classification, "count": count}
        for (experiment_type, classification), count in sorted(counter.items())
    ]


def _make_output_dir(work_paths: WorkPaths) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return work_paths.workdir / "diagnostics" / f"align-diagnose-{stamp}"


def _resolve_audio_path(work_paths: WorkPaths, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = work_paths.workdir / path
    if candidate.exists():
        return candidate
    return path


def _resolve_text(text: str, text_mode: str) -> str:
    if text_mode == "normalized":
        return "".join(char for char in text.strip() if char not in " \t\r\n")
    return text


def _parse_segment_ids(value: str | None) -> list[str]:
    if not value:
        return []
    path = Path(value)
    if path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return [item.strip() for item in value.split(",") if item.strip()]


def _find_token_rows(raw_output: Any) -> tuple[list[Any] | None, str | None]:
    current = raw_output
    if isinstance(current, list):
        if not current:
            return [], "list.empty"
        current = current[0]
    if isinstance(current, dict):
        for key in ("tokens", "words", "timestamps"):
            if current.get(key) is not None:
                return list(current.get(key) or []), f"dict.{key}"
        return None, None
    for attr in ("items", "tokens"):
        if hasattr(current, attr):
            value = getattr(current, attr)
            if not callable(value):
                return list(value or []), f"attr.{attr}"
    return None, None


def _coerce_token_row(item: Any) -> tuple[str, float, float]:
    if isinstance(item, dict):
        text = item.get("text") or item.get("token") or item.get("word") or ""
        start_time = float(item.get("start") or item.get("start_time") or 0.0)
        end_time = float(item.get("end") or item.get("end_time") or start_time)
        return str(text), start_time, end_time
    text = str(getattr(item, "text", getattr(item, "token", "")))
    start_time = float(getattr(item, "start_time", getattr(item, "start", 0.0)))
    end_time = float(getattr(item, "end_time", getattr(item, "end", start_time)))
    return text, start_time, end_time


def _local_max_cps(tokens: list[AlignedToken]) -> float:
    normalized = [token for token in tokens if token.text.strip() and token.end_time >= token.start_time]
    max_cps = 0.0
    for start_index, first in enumerate(normalized):
        chars = 0
        window_start = first.start_time
        window_end = first.end_time
        for token in normalized[start_index : start_index + 12]:
            chars += len(token.text.strip())
            window_end = max(window_end, token.end_time)
            duration = window_end - window_start
            if duration > 0:
                max_cps = max(max_cps, chars / duration)
    return max_cps


def _time_range(rows: list[dict[str, Any]], start_key: str, end_key: str) -> dict[str, float | None]:
    if not rows:
        return {"start": None, "end": None, "duration": None}
    start = min(float(row[start_key]) for row in rows)
    end = max(float(row[end_key]) for row in rows)
    return {"start": start, "end": end, "duration": round(max(0.0, end - start), 3)}


def _window_rms(values: list[int], sample_rate: int) -> list[float]:
    window = max(1, sample_rate // 10)
    result = []
    for index in range(0, len(values), window):
        chunk = values[index : index + window]
        if not chunk:
            continue
        result.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)) / 32768.0)
    return result


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return ordered[index]


def _edge_duration(flags: list[bool], *, from_start: bool) -> float:
    iterable = flags if from_start else list(reversed(flags))
    count = 0
    for flag in iterable:
        if not flag:
            break
        count += 1
    return count * 0.1


def _longest_true_run(flags: list[bool]) -> int:
    current = 0
    longest = 0
    for flag in flags:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _input_audio_fields(audio_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "wav_duration": audio_metrics.get("wav_duration"),
        "sample_rate": audio_metrics.get("sample_rate"),
        "channels": audio_metrics.get("channels"),
        "sample_width_bytes": audio_metrics.get("sample_width_bytes"),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return float(numerator) / float(denominator)


def _text_similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_directory(path.parent)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_arg_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", default=DEFAULT_ALIGN_MODEL)
    parser.add_argument("--segments", default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--text-mode", choices=["asr", "normalized", "manual"], default="asr")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--dry-run-plan", action="store_true")
    parser.add_argument("--with-asr-reference", action="store_true")
    parser.add_argument("--asr-model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--asr-language", default=None)
    parser.add_argument("--asr-max-new-tokens", type=int, default=512)
    parser.add_argument("--asr-window-seconds", type=float, default=3.0)
    return parser
