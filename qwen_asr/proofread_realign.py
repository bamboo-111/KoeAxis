from __future__ import annotations

import argparse
import wave
from pathlib import Path
from time import perf_counter
from typing import Any

from qwen_asr.align import QwenForcedAligner
from qwen_asr.mfa_experiment import detect_mfa_environment, run_local_mfa_alignment_experiments
from qwen_asr.models import TranscriptSegment, WorkPaths
from qwen_asr import proofread_realign_strategy as _strategy
from qwen_asr.storage import read_json, write_json_atomic


DEFAULT_REALIGN_PADDING_MS = 800
_fallback_original_timing = _strategy.fallback_original_timing
_should_keep_mixed_language_original_timing = _strategy.should_keep_mixed_language_original_timing
_timing_candidate_guard = _strategy.timing_candidate_guard
_clamp_timing_candidate_to_neighbors = _strategy.clamp_timing_candidate_to_neighbors
_expand_display_range_to_min_duration = _strategy.expand_display_range_to_min_duration
_clamp_display_range_to_original_window = _strategy.clamp_display_range_to_original_window
_mfa_content_score = _strategy.mfa_content_score
_normalize_mfa_content = _strategy.normalize_mfa_content
_is_japanese_character = _strategy.is_japanese_character
_sort_key = _strategy.sort_key
_safe_id = _strategy.safe_id


def run_proofread_realign_stage(
    args: argparse.Namespace,
    work_paths: WorkPaths,
    *,
    aligner_factory: type[QwenForcedAligner] = QwenForcedAligner,
) -> dict[str, Any]:
    manifest_path = _proofread_realign_manifest_path(args, work_paths)
    report_path = _proofread_realign_report_path(args, work_paths)
    manifest = read_json(manifest_path, default={})
    if not isinstance(manifest, dict) or not manifest:
        report = _empty_report("missing-mimo-proofread-manifest")
        _write_report(report_path, report)
        return report

    _clear_punctuation_only_realign_flags(manifest)
    retry_method = str(getattr(args, "proofread_realign_retry_method", "none") or "none")
    all_pending = _pending_items(manifest, retry_method=retry_method)
    max_items = int(getattr(args, "proofread_realign_max_items", 0) or 0)
    pending = all_pending[:max_items] if max_items > 0 else all_pending
    primary = str(getattr(args, "proofread_realign_primary", "qwen-first") or "qwen-first")
    if not pending:
        write_json_atomic(manifest_path, manifest)
        report = {
            "status": "PASS",
            "manifest_path": str(manifest_path),
            "candidate_count": len(all_pending),
            "pending_count": 0,
            "completed_count": 0,
            "failed_count": 0,
            "items": [],
        }
        _write_report(report_path, report)
        return report

    output_dir = _proofread_realign_diagnostics_dir(args, work_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows: list[dict[str, Any]] = []
    alignment_pending = pending
    if primary != "original-timing":
        alignment_pending = []
        for subtitle_id, item in pending:
            if _should_keep_mixed_language_original_timing(item):
                start_ms = max(0, int(item.get("start_time", 0) or 0))
                end_ms = max(start_ms + 1, int(item.get("end_time", start_ms + 1) or start_ms + 1))
                rows.append(
                    _fallback_original_timing(
                        item,
                        subtitle_id,
                        start_ms,
                        end_ms,
                        output_dir / f"{_safe_id(subtitle_id)}.wav",
                        "mixed-language-original-timing",
                        method="mixed-language-original-timing",
                    )
                )
            else:
                alignment_pending.append((subtitle_id, item))

    if alignment_pending and primary != "original-timing" and not work_paths.audio_path.exists():
        report = _failure_report(alignment_pending, "source-audio-missing", str(work_paths.audio_path))
        report["manifest_path"] = str(manifest_path)
        _write_report(report_path, report)
        return report

    if primary == "original-timing":
        for subtitle_id, item in pending:
            start_ms = max(0, int(item.get("start_time", 0) or 0))
            end_ms = max(start_ms + 1, int(item.get("end_time", start_ms + 1) or start_ms + 1))
            rows.append(
                _fallback_original_timing(
                    item,
                    subtitle_id,
                    start_ms,
                    end_ms,
                    output_dir / f"{_safe_id(subtitle_id)}.wav",
                    "explicit-original-timing-primary",
                )
            )
    elif primary == "mfa-local":
        for subtitle_id, item in alignment_pending:
            rows.append(
                _realign_one_mfa_only(
                    work_paths,
                    output_dir,
                    subtitle_id,
                    item,
                    fallback=str(getattr(args, "proofread_realign_fallback", "original-timing") or "original-timing"),
                    mfa_padding_ms=int(getattr(args, "proofread_realign_mfa_padding_ms", 700)),
                    mfa_min_content_score=float(getattr(args, "proofread_realign_mfa_min_content_score", 0.70)),
                    manifest=manifest,
                )
            )
    else:
        if alignment_pending:
            aligner = aligner_factory(
                str(getattr(args, "proofread_realign_model", None) or getattr(args, "align_model", None) or getattr(args, "model", "")),
                dtype=str(getattr(args, "dtype", "fp16")),
                device=str(getattr(args, "device", "cuda")),
                attn_implementation=getattr(args, "attn_implementation", None),
                keep_raw_model_output=bool(getattr(args, "keep_raw_model_output", False)),
                keep_failed_tokens=True,
                model_cache_dir=getattr(args, "model_cache_dir", None),
                local_files_only=bool(getattr(args, "local_files_only", True)),
            )
            try:
                aligner.load()
                for subtitle_id, item in alignment_pending:
                    rows.append(
                        _realign_one(
                            aligner,
                            work_paths.audio_path,
                            output_dir,
                            subtitle_id,
                            item,
                            language=str(getattr(args, "proofread_realign_language", None) or getattr(args, "language", None) or "Japanese"),
                            padding_ms=int(getattr(args, "proofread_realign_padding_ms", DEFAULT_REALIGN_PADDING_MS)),
                            fallback=str(getattr(args, "proofread_realign_fallback", "original-timing") or "original-timing"),
                            mfa_fallback=str(getattr(args, "proofread_realign_mfa_fallback", "off") or "off"),
                            mfa_padding_ms=int(getattr(args, "proofread_realign_mfa_padding_ms", 700)),
                            mfa_min_content_score=float(getattr(args, "proofread_realign_mfa_min_content_score", 0.70)),
                            work_paths=work_paths,
                            manifest=manifest,
                        )
                    )
            finally:
                aligner.close()

    write_json_atomic(manifest_path, manifest)
    completed = sum(row["status"] == "completed" for row in rows)
    fallback_count = sum(row["status"] == "fallback" for row in rows)
    mfa_completed = sum(row.get("method") == "mfa-local" and row["status"] == "completed" for row in rows)
    mfa_unusable = sum(row.get("mfa_status") == "unusable" for row in rows)
    mfa_rejected = sum(row.get("mfa_status") == "rejected" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    report = {
        "status": "FAIL" if failed else ("WARN" if fallback_count or mfa_unusable or mfa_rejected else "PASS"),
        "manifest_path": str(manifest_path),
        "diagnostics_dir": str(output_dir),
        "primary": primary,
        "retry_method": retry_method,
        "candidate_count": len(all_pending),
        "pending_count": len(pending),
        "completed_count": completed,
        "fallback_count": fallback_count,
        "mfa_completed_count": mfa_completed,
        "mfa_unusable_count": mfa_unusable,
        "mfa_rejected_count": mfa_rejected,
        "failed_count": failed,
        "elapsed_ms": int((perf_counter() - started) * 1000),
        "items": rows,
    }
    _write_report(report_path, report)
    return report


def has_unrealigned_proofread_changes(work_paths: WorkPaths) -> bool:
    manifest = read_json(work_paths.mimo_proofread_manifest, default={})
    if not isinstance(manifest, dict):
        return False
    return any(_needs_realign(item) for item in manifest.values() if isinstance(item, dict))


def _pending_items(manifest: dict[str, Any], *, retry_method: str = "none") -> list[tuple[str, dict[str, Any]]]:
    return [
        (str(key), item)
        for key, item in sorted(manifest.items(), key=lambda pair: _sort_key(pair[0]))
        if isinstance(item, dict) and _needs_realign(item, retry_method=retry_method)
    ]


def _needs_realign(item: dict[str, Any], *, retry_method: str = "none") -> bool:
    if bool(item.get("needs_realign")) and str(item.get("realign_status", "")).strip() != "completed":
        return True
    return (
        retry_method == "original-timing"
        and str(item.get("realign_status", "")).strip() == "completed"
        and str(item.get("realign_method", "")).strip() == "original-timing"
    )


def _clear_punctuation_only_realign_flags(manifest: dict[str, Any]) -> None:
    for item in manifest.values():
        if not isinstance(item, dict) or not bool(item.get("needs_realign")):
            continue
        change = _latest_original_change(item)
        if change is None:
            continue
        before, after = change
        if _normalize_mfa_content(before) != _normalize_mfa_content(after):
            continue
        item["needs_realign"] = False
        item["realign_status"] = "completed"
        item["realign_method"] = "punctuation-only"
        item["realign_source"] = "proofread-realign-skip"


def _latest_original_change(item: dict[str, Any]) -> tuple[str, str] | None:
    history = item.get("proofread_history", [])
    if not isinstance(history, list):
        return None
    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes", {})
        if not isinstance(changes, dict):
            continue
        original_change = changes.get("original_subtitle")
        if not isinstance(original_change, dict):
            continue
        return (
            str(original_change.get("before", "") or ""),
            str(original_change.get("after", "") or ""),
        )
    return None


def _realign_one(
    aligner: QwenForcedAligner,
    source_audio: Path,
    output_dir: Path,
    subtitle_id: str,
    item: dict[str, Any],
    *,
    language: str,
    padding_ms: int,
    fallback: str,
    mfa_fallback: str,
    mfa_padding_ms: int,
    mfa_min_content_score: float,
    work_paths: WorkPaths,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    start_ms = max(0, int(item.get("start_time", 0) or 0))
    end_ms = max(start_ms + 1, int(item.get("end_time", start_ms + 1) or start_ms + 1))
    text = str(item.get("original_subtitle", "")).strip()
    if not text:
        item["realign_status"] = "failed"
        item["realign_error"] = "empty-original-subtitle"
        return {"id": subtitle_id, "status": "failed", "error": "empty-original-subtitle"}

    clip_start_ms = max(0, start_ms - max(0, padding_ms))
    clip_end_ms = end_ms + max(0, padding_ms)
    clip_path = output_dir / f"{_safe_id(subtitle_id)}.wav"
    clip_start_ms, clip_end_ms = _crop_wav(source_audio, clip_path, clip_start_ms, clip_end_ms)
    transcript = TranscriptSegment(
        segment_id=f"proofread_{subtitle_id}",
        audio_path=str(clip_path),
        global_start_time=clip_start_ms / 1000.0,
        global_end_time=clip_end_ms / 1000.0,
        text=text,
        language=language,
    )
    result = aligner.run_segment(transcript, cleanup=False)
    if result.status != "completed" or not result.tokens:
        error = result.error or "no-aligned-tokens"
        if mfa_fallback == "local":
            mfa_row = _try_mfa_local_realign(
                work_paths,
                item,
                subtitle_id,
                start_ms,
                end_ms,
                text,
                padding_ms=mfa_padding_ms,
                min_content_score=mfa_min_content_score,
                qwen_error=error,
                manifest=manifest,
            )
            if mfa_row["status"] == "completed":
                return mfa_row
            error = f"{error}; mfa-local {mfa_row.get('reason') or mfa_row.get('status')}"
        else:
            mfa_row = None
        if fallback == "original-timing":
            return _fallback_original_timing(
                item,
                subtitle_id,
                start_ms,
                end_ms,
                clip_path,
                error,
                mfa_row=mfa_row,
            )
        item["realign_status"] = "failed"
        item["realign_error"] = error
        return {"id": subtitle_id, "status": "failed", "error": error, "clip_path": str(clip_path)}

    token_start_ms = min(int(round(token.start_time * 1000)) for token in result.tokens)
    token_end_ms = max(int(round(token.end_time * 1000)) for token in result.tokens)
    if token_end_ms <= token_start_ms:
        token_end_ms = token_start_ms + 1
    recognized_text = "".join(str(token.text) for token in result.tokens)
    content_score = _mfa_content_score(text, recognized_text)
    if content_score < mfa_min_content_score:
        error = f"qwen-content-mismatch score={content_score:.6f}"
        if fallback == "original-timing":
            return _fallback_original_timing(item, subtitle_id, start_ms, end_ms, clip_path, error)
        item["realign_status"] = "failed"
        item["realign_error"] = error
        return {
            "id": subtitle_id,
            "status": "failed",
            "error": error,
            "qwen_recognized_text": recognized_text,
            "qwen_content_score": round(content_score, 6),
            "qwen_min_content_score": mfa_min_content_score,
            "clip_path": str(clip_path),
        }
    timing_guard = _timing_candidate_guard(
        manifest,
        subtitle_id=subtitle_id,
        start_ms=token_start_ms,
        end_ms=token_end_ms,
        clip_start_ms=clip_start_ms,
        clip_end_ms=clip_end_ms,
    )
    realign_method: str | None = None
    realign_warning: str | None = None
    if not timing_guard["accepted"]:
        clamped = _clamp_timing_candidate_to_neighbors(
            manifest,
            subtitle_id=subtitle_id,
            start_ms=token_start_ms,
            end_ms=token_end_ms,
            clip_start_ms=clip_start_ms,
            clip_end_ms=clip_end_ms,
            timing_guard=timing_guard,
        )
        if clamped["accepted"]:
            token_start_ms = int(clamped["start_ms"])
            token_end_ms = int(clamped["end_ms"])
            timing_guard = clamped
            realign_method = "qwen-clamped"
            realign_warning = "qwen timing clamped to neighbor boundaries: severe-neighbor-overlap"
        else:
            error = str(timing_guard["reason"])
            if fallback == "original-timing":
                return _fallback_original_timing(item, subtitle_id, start_ms, end_ms, clip_path, error)
            item["realign_status"] = "failed"
            item["realign_error"] = error
            return {"id": subtitle_id, "status": "failed", "error": error, "timing_guard": timing_guard}
    duration_clamp = _clamp_display_range_to_original_window(
        token_start_ms,
        token_end_ms,
        original_start_ms=start_ms,
        original_end_ms=end_ms,
    )
    if duration_clamp["accepted"]:
        token_start_ms = int(duration_clamp["start_ms"])
        token_end_ms = int(duration_clamp["end_ms"])
        timing_guard = {
            **timing_guard,
            "duration_clamp": duration_clamp,
            "start_ms": token_start_ms,
            "end_ms": token_end_ms,
        }
        realign_method = "qwen-clamped"
        realign_warning = str(duration_clamp["warning"])
    item["start_time"] = token_start_ms
    item["end_time"] = token_end_ms
    item["needs_realign"] = False
    item["realign_status"] = "completed"
    item["realign_source"] = "proofread-realign"
    if realign_method:
        item["realign_method"] = realign_method
    else:
        item.pop("realign_method", None)
    if realign_warning:
        item["realign_warning"] = realign_warning
    else:
        item.pop("realign_warning", None)
    item["realign_clip_path"] = str(clip_path)
    item["realign_tokens"] = [
        {
            "text": token.text,
            "start_time": int(round(token.start_time * 1000)),
            "end_time": int(round(token.end_time * 1000)),
        }
        for token in result.tokens
    ]
    item.pop("realign_error", None)
    return {
        "id": subtitle_id,
        "status": "completed",
        "before_start_time": start_ms,
        "before_end_time": end_ms,
        "after_start_time": token_start_ms,
        "after_end_time": token_end_ms,
        "token_count": len(result.tokens),
        "qwen_recognized_text": recognized_text,
        "qwen_content_score": round(content_score, 6),
        "timing_guard": timing_guard,
        "clip_path": str(clip_path),
        **({"method": realign_method} if realign_method else {}),
        **({"warning": realign_warning} if realign_warning else {}),
    }


def _realign_one_mfa_only(
    work_paths: WorkPaths,
    output_dir: Path,
    subtitle_id: str,
    item: dict[str, Any],
    *,
    fallback: str,
    mfa_padding_ms: int,
    mfa_min_content_score: float,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    start_ms = max(0, int(item.get("start_time", 0) or 0))
    end_ms = max(start_ms + 1, int(item.get("end_time", start_ms + 1) or start_ms + 1))
    text = str(item.get("original_subtitle", "")).strip()
    if not text:
        item["realign_status"] = "failed"
        item["realign_error"] = "empty-original-subtitle"
        return {"id": subtitle_id, "status": "failed", "error": "empty-original-subtitle"}
    mfa_row = _try_mfa_local_realign(
        work_paths,
        item,
        subtitle_id,
        start_ms,
        end_ms,
        text,
        padding_ms=mfa_padding_ms,
        min_content_score=mfa_min_content_score,
        qwen_error="mfa-local-primary",
        manifest=manifest,
    )
    if mfa_row["status"] == "completed":
        return mfa_row
    if fallback == "original-timing":
        mfa_result = mfa_row.get("mfa_result", {}) if isinstance(mfa_row.get("mfa_result"), dict) else {}
        clip_path = Path(str(mfa_result.get("clip") or output_dir / f"{_safe_id(subtitle_id)}.wav"))
        return _fallback_original_timing(
            item,
            subtitle_id,
            start_ms,
            end_ms,
            clip_path,
            f"mfa-local {mfa_row.get('reason') or mfa_row.get('status')}",
            mfa_row=mfa_row,
        )
    item["realign_status"] = "failed"
    item["realign_error"] = str(mfa_row.get("reason") or mfa_row.get("status") or "mfa-local-failed")
    return {
        "id": subtitle_id,
        "status": "failed",
        "error": item["realign_error"],
        "mfa_status": mfa_row.get("mfa_status", mfa_row.get("status")),
        "mfa_result": mfa_row.get("mfa_result", mfa_row),
    }


def _try_mfa_local_realign(
    work_paths: WorkPaths,
    item: dict[str, Any],
    subtitle_id: str,
    start_ms: int,
    end_ms: int,
    text: str,
    *,
    padding_ms: int,
    min_content_score: float,
    qwen_error: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    environment = detect_mfa_environment(run_version_check=True)
    candidate = {
        "source": "proofread-realign",
        "reason": "qwen-forced-align-failed",
        "severity": "WARN",
        "subtitle_id": subtitle_id,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "text": text,
        "details": {"qwen_error": qwen_error},
    }
    results = run_local_mfa_alignment_experiments(
        work_paths,
        [candidate],
        environment=environment,
        max_run_candidates=1,
        padding_ms=padding_ms,
    )
    mfa_result = results[0] if results else {"status": "skipped", "reason": "no-mfa-result"}
    global_ranges = mfa_result.get("global_word_ranges", []) if isinstance(mfa_result, dict) else []
    usable_ranges = [
        token for token in global_ranges
        if isinstance(token, dict)
        and isinstance(token.get("start_ms"), int)
        and isinstance(token.get("end_ms"), int)
        and int(token["end_ms"]) > int(token["start_ms"])
        and str(token.get("text", "")).strip().lower() not in {"<unk>", "unk"}
    ]
    if not mfa_result.get("usable") or not usable_ranges:
        return {
            "id": subtitle_id,
            "status": "mfa-unusable",
            "mfa_status": "unusable",
            "reason": str(mfa_result.get("reason", "mfa-output-unusable")),
            "before_start_time": start_ms,
            "before_end_time": end_ms,
            "qwen_error": qwen_error,
            "mfa_result": mfa_result,
        }
    recognized_text = "".join(str(token.get("text", "")) for token in usable_ranges)
    content_score = _mfa_content_score(text, recognized_text)
    if content_score < min_content_score:
        return {
            "id": subtitle_id,
            "status": "mfa-rejected",
            "mfa_status": "rejected",
            "reason": "mfa-content-mismatch",
            "before_start_time": start_ms,
            "before_end_time": end_ms,
            "qwen_error": qwen_error,
            "mfa_recognized_text": recognized_text,
            "mfa_content_score": round(content_score, 6),
            "mfa_min_content_score": min_content_score,
            "mfa_result": mfa_result,
        }
    token_start_ms = min(int(token["start_ms"]) for token in usable_ranges)
    token_end_ms = max(int(token["end_ms"]) for token in usable_ranges)
    timing_guard = _timing_candidate_guard(
        manifest,
        subtitle_id=subtitle_id,
        start_ms=token_start_ms,
        end_ms=token_end_ms,
        clip_start_ms=max(0, start_ms - max(0, padding_ms)),
        clip_end_ms=end_ms + max(0, padding_ms),
    )
    if not timing_guard["accepted"]:
        return {
            "id": subtitle_id,
            "status": "mfa-rejected",
            "mfa_status": "rejected",
            "reason": str(timing_guard["reason"]),
            "before_start_time": start_ms,
            "before_end_time": end_ms,
            "qwen_error": qwen_error,
            "mfa_recognized_text": recognized_text,
            "mfa_content_score": round(content_score, 6),
            "mfa_min_content_score": min_content_score,
            "timing_guard": timing_guard,
            "mfa_result": mfa_result,
        }
    item["start_time"] = token_start_ms
    item["end_time"] = token_end_ms
    item["needs_realign"] = False
    item["realign_status"] = "completed"
    item["realign_source"] = "proofread-realign-mfa"
    item["realign_method"] = "mfa-local"
    item["realign_warning"] = f"qwen forced align failed: {qwen_error}"
    item["realign_tokens"] = [
        {
            "text": str(token.get("text", "")),
            "start_time": int(token["start_ms"]),
            "end_time": int(token["end_ms"]),
        }
        for token in usable_ranges
    ]
    item["realign_mfa_result"] = {
        "clip": mfa_result.get("clip"),
        "lab": mfa_result.get("lab"),
        "lab_text": mfa_result.get("lab_text"),
        "lab_text_source": mfa_result.get("lab_text_source"),
        "elapsed_ms": mfa_result.get("elapsed_ms"),
        "word_quality": mfa_result.get("word_quality"),
        "recognized_text": recognized_text,
        "content_score": round(content_score, 6),
        "min_content_score": min_content_score,
    }
    item.pop("realign_error", None)
    return {
        "id": subtitle_id,
        "status": "completed",
        "method": "mfa-local",
        "mfa_status": "usable",
        "before_start_time": start_ms,
        "before_end_time": end_ms,
        "after_start_time": token_start_ms,
        "after_end_time": token_end_ms,
        "token_count": len(usable_ranges),
        "qwen_error": qwen_error,
        "mfa_recognized_text": recognized_text,
        "mfa_content_score": round(content_score, 6),
        "mfa_min_content_score": min_content_score,
        "timing_guard": timing_guard,
        "mfa_result": mfa_result,
    }


def _crop_wav(source: Path, target: Path, start_ms: int, end_ms: int) -> tuple[int, int]:
    with wave.open(str(source), "rb") as reader:
        frame_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        channels = reader.getnchannels()
        total_frames = reader.getnframes()
        total_ms = int(round(total_frames * 1000 / frame_rate)) if frame_rate else 0
        start_ms = min(max(0, start_ms), total_ms)
        end_ms = min(max(start_ms + 1, end_ms), total_ms)
        start_frame = int(start_ms * frame_rate / 1000)
        end_frame = max(start_frame + 1, int(end_ms * frame_rate / 1000))
        reader.setpos(start_frame)
        frames = reader.readframes(end_frame - start_frame)
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(frame_rate)
        writer.writeframes(frames)
    return start_ms, end_ms


def _proofread_realign_manifest_path(args: argparse.Namespace, work_paths: WorkPaths) -> Path:
    value = str(getattr(args, "proofread_realign_manifest", "") or "").strip()
    return Path(value) if value else work_paths.mimo_proofread_manifest


def _proofread_realign_diagnostics_dir(args: argparse.Namespace, work_paths: WorkPaths) -> Path:
    value = str(getattr(args, "proofread_realign_diagnostics_dir", "") or "").strip()
    return Path(value) if value else work_paths.workdir / "diagnostics" / "proofread-realign"


def _proofread_realign_report_path(args: argparse.Namespace, work_paths: WorkPaths) -> Path:
    value = str(getattr(args, "proofread_realign_report_output", "") or "").strip()
    return Path(value) if value else work_paths.workdir / "reports" / "proofread_realign.json"


def _write_report(output: Path, report: dict[str, Any]) -> None:
    write_json_atomic(output, report)


def _empty_report(reason: str) -> dict[str, Any]:
    return {
        "status": "PASS",
        "reason": reason,
        "pending_count": 0,
        "completed_count": 0,
        "failed_count": 0,
        "items": [],
    }


def _failure_report(pending: list[tuple[str, dict[str, Any]]], reason: str, detail: str) -> dict[str, Any]:
    return {
        "status": "FAIL",
        "reason": reason,
        "detail": detail,
        "pending_count": len(pending),
        "completed_count": 0,
        "failed_count": len(pending),
        "items": [{"id": subtitle_id, "status": "failed", "error": reason} for subtitle_id, _item in pending],
    }
