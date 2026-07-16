from __future__ import annotations

import json
import os
import shutil
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from qwen_asr.align import AlignTimingValidationConfig, QwenForcedAligner, validate_aligned_token_timing
from qwen_asr.alignment_state import derive_alignment_state
from qwen_asr.defaults import DEFAULT_ALIGN_MODEL, DEFAULT_MODEL_CACHE_DIR
from qwen_asr.mfa_environment import detect_mfa_environment
from qwen_asr.mfa_guards import local_ass_match_score
from qwen_asr.mfa_runner import run_local_mfa_alignment_experiments
from qwen_asr.models import AlignedSegment, AlignedToken, TranscriptSegment, WorkPaths
from qwen_asr.storage import append_jsonl, read_json, write_json_atomic


RECOVERY_STRATEGIES = ("auto", "qwen", "mfa-local")
SUPPORTED_QWEN_LANGUAGES = {"auto", "japanese", "english", "chinese"}
LOCK_STALE_SECONDS = 2 * 60 * 60


class RecoveryExecutionError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 409) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def execute_alignment_recovery(
    work_paths: WorkPaths,
    *,
    segment_id: str,
    strategy: str = "auto",
    language_route: str | None = None,
    verified_text: str | None = None,
    use_verified_text: bool = False,
    actor: str = "local-user",
    settings: dict[str, Any] | None = None,
    qwen_runner: Callable[[TranscriptSegment], AlignedSegment | dict[str, Any]] | None = None,
    mfa_runner: Callable[..., list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    settings = settings if isinstance(settings, dict) else {}
    normalized_strategy = str(strategy or "auto").strip().lower()
    if normalized_strategy not in RECOVERY_STRATEGIES:
        raise RecoveryExecutionError("RECOVERY_STRATEGY_INVALID", f"unsupported recovery strategy: {strategy}")
    with _execution_lock(work_paths, segment_id):
        aligned = _read_list(work_paths.aligned_manifest, "aligned manifest")
        transcripts = _read_list(work_paths.transcript_manifest, "transcript manifest")
        target = next((item for item in aligned if str(item.get("segment_id")) == segment_id), None)
        transcript = next((item for item in transcripts if str(item.get("segment_id")) == segment_id), None)
        if target is None or transcript is None:
            raise RecoveryExecutionError("RECOVERY_SEGMENT_MISSING", "aligned or transcript segment is missing", status=404)
        if derive_alignment_state(target) != "failed":
            raise RecoveryExecutionError("RECOVERY_SEGMENT_NOT_FAILED", "recovery only accepts a currently failed segment")
        original_text = str(transcript.get("text", target.get("text", "")))
        selected_text = original_text
        text_source = "original_transcript"
        if use_verified_text:
            selected_text = str(verified_text or "").strip()
            if not selected_text:
                raise RecoveryExecutionError(
                    "VERIFIED_TEXT_NOT_AVAILABLE",
                    "use_verified_text requires an explicitly verified non-empty transcript",
                )
            text_source = "human_verified_text"
        route = _normalize_language(language_route or transcript.get("language") or target.get("language") or "auto")
        if route == "auto":
            route = _normalize_language(transcript.get("language") or target.get("language") or "auto")
        selected_strategy = _select_strategy(normalized_strategy, route)
        started = time.monotonic()
        attempt = {
            "segment_id": segment_id,
            "requested_strategy": normalized_strategy,
            "strategy": selected_strategy,
            "language_route": route,
            "text_source": text_source,
            "original_text": original_text,
            "selected_text": selected_text,
            "text_changed": _normalize_text(original_text) != _normalize_text(selected_text),
            "actor": actor,
            "started_at": _utc_now(),
        }
        try:
            if selected_strategy == "qwen":
                result = _execute_qwen(
                    target,
                    transcript,
                    selected_text=selected_text,
                    language_route=route,
                    settings=settings,
                    runner=qwen_runner,
                )
            else:
                result = _execute_mfa_local(
                    work_paths,
                    aligned,
                    target,
                    selected_text=selected_text,
                    language_route=route,
                    settings=settings,
                    runner=mfa_runner,
                )
        except KeyboardInterrupt:
            raise
        except RecoveryExecutionError:
            raise
        except Exception as exc:  # model/provider/runtime boundary
            result = {
                "status": "failed",
                "alignment_state": "failed",
                "error_code": "RECOVERY_BACKEND_EXCEPTION",
                "error": f"{type(exc).__name__}: {exc}",
            }
        elapsed_ms = round((time.monotonic() - started) * 1000)
        attempt.update(result)
        attempt["elapsed_ms"] = elapsed_ms
        attempt["finished_at"] = _utc_now()
        if result.get("alignment_state") != "completed_exact":
            attempt["original_state_preserved"] = True
            return attempt
        replacement = result.get("aligned_segment")
        if not isinstance(replacement, dict):
            raise RecoveryExecutionError("RECOVERY_RESULT_INVALID", "exact recovery result has no aligned segment")
        backup_dir = backup_alignment_state(work_paths, segment_id)
        original_error = str(target.get("error", "") or target.get("alignment_failure_reason", "") or "")
        replacement["recovery"] = {
            "method": selected_strategy,
            "actor": actor,
            "accepted_at": _utc_now(),
            "text_source": text_source,
            "original_text": original_text,
            "selected_text": selected_text,
            "original_error": original_error,
            "language_route": route,
            "elapsed_ms": elapsed_ms,
            "backup_path": str(backup_dir),
            "backend_evidence": result.get("backend_evidence", {}),
        }
        replacement["alignment_failure_reason"] = original_error
        replacement["alignment_state"] = "completed_exact"
        replacement["status"] = "completed"
        aligned[aligned.index(target)] = replacement
        _write_aligned_state(work_paths, aligned, replacement)
        attempt["backup_path"] = str(backup_dir)
        attempt["writeback"] = "completed"
        attempt.pop("aligned_segment", None)
        return attempt


def backup_alignment_state(work_paths: WorkPaths, segment_id: str) -> Path:
    backup_dir = (
        work_paths.workdir
        / "reports"
        / "recovery-backups"
        / f"{segment_id}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    registered = []
    for path in (work_paths.aligned_manifest, work_paths.aligned_checkpoint_path, work_paths.aligned_events_path):
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
            registered.append(str(path))
    (backup_dir / "BACKED_UP_FILES.txt").write_text("\n".join(registered) + "\n", encoding="utf-8")
    return backup_dir


def restore_alignment_backup(work_paths: WorkPaths, backup_dir: Path, *, segment_id: str) -> dict[str, Any]:
    resolved = backup_dir.resolve()
    allowed_root = (work_paths.workdir / "reports" / "recovery-backups").resolve()
    if not resolved.is_relative_to(allowed_root) or not resolved.is_dir():
        raise RecoveryExecutionError("RECOVERY_BACKUP_INVALID", "backup path is outside this workspace")
    source_manifest = resolved / work_paths.aligned_manifest.name
    if not source_manifest.exists():
        raise RecoveryExecutionError("RECOVERY_BACKUP_EMPTY", "backup contains no aligned manifest")
    backup_rows = _read_list(source_manifest, "backup aligned manifest")
    restored_target = next((row for row in backup_rows if str(row.get("segment_id")) == segment_id), None)
    if restored_target is None:
        raise RecoveryExecutionError("RECOVERY_BACKUP_TARGET_MISSING", "backup does not contain the target segment")
    current_rows = _read_list(work_paths.aligned_manifest, "aligned manifest")
    current_target = next((row for row in current_rows if str(row.get("segment_id")) == segment_id), None)
    if current_target is None:
        raise RecoveryExecutionError("RECOVERY_SEGMENT_MISSING", "current manifest does not contain the target segment")
    safety_backup = backup_alignment_state(work_paths, f"{segment_id}-pre-undo")
    current_rows[current_rows.index(current_target)] = restored_target
    write_json_atomic(work_paths.aligned_manifest, current_rows)
    write_json_atomic(work_paths.aligned_checkpoint_path, current_rows)
    append_jsonl(
        work_paths.aligned_events_path,
        {
            "type": "aligned_recovery_undo",
            "segment_id": segment_id,
            "payload": restored_target,
            "restored_from": str(resolved),
            "safety_backup": str(safety_backup),
        },
    )
    return {
        "status": "restored",
        "segment_id": segment_id,
        "backup_path": str(resolved),
        "safety_backup": str(safety_backup),
        "restored": [str(work_paths.aligned_manifest), str(work_paths.aligned_checkpoint_path)],
    }


def _execute_qwen(
    target: dict[str, Any],
    transcript: dict[str, Any],
    *,
    selected_text: str,
    language_route: str,
    settings: dict[str, Any],
    runner: Callable[[TranscriptSegment], AlignedSegment | dict[str, Any]] | None,
) -> dict[str, Any]:
    if language_route not in SUPPORTED_QWEN_LANGUAGES:
        return {
            "status": "not_executable",
            "alignment_state": "failed",
            "error_code": "LANGUAGE_BACKEND_UNAVAILABLE",
            "error": f"qwen recovery has no approved route for language {language_route}",
        }
    segment = TranscriptSegment(
        segment_id=str(transcript["segment_id"]),
        audio_path=str(transcript.get("audio_path", target.get("audio_path", ""))),
        global_start_time=float(transcript.get("global_start_time", target["global_start_time"])),
        global_end_time=float(transcript.get("global_end_time", target["global_end_time"])),
        text=selected_text,
        language=_display_language(language_route),
        status="completed",
    )
    if runner is None:
        aligner = QwenForcedAligner(
            model_name=str(settings.get("model", DEFAULT_ALIGN_MODEL)),
            dtype=str(settings.get("dtype", "fp16")),
            device=str(settings.get("device", "cuda")),
            attn_implementation=settings.get("attn_implementation"),
            keep_raw_model_output=True,
            keep_failed_tokens=True,
            model_cache_dir=str(settings.get("model_cache_dir", DEFAULT_MODEL_CACHE_DIR)),
            local_files_only=bool(settings.get("local_files_only", True)),
            timing_validation_config=AlignTimingValidationConfig(),
        )
        result_value = aligner.run_segment(segment, cleanup=True)
    else:
        result_value = runner(segment)
    result = asdict(result_value) if isinstance(result_value, AlignedSegment) else dict(result_value)
    if derive_alignment_state(result) != "completed_exact":
        return {
            "status": "failed",
            "alignment_state": "failed",
            "error_code": "QWEN_RETRY_FAILED",
            "error": str(result.get("error") or result.get("alignment_failure_reason") or "qwen retry did not return exact tokens"),
            "backend_evidence": _safe_backend_evidence(result),
        }
    guard = _exact_result_guard(result, target, selected_text)
    if guard is not None:
        return {
            "status": "rejected",
            "alignment_state": "failed",
            "error_code": guard[0],
            "error": guard[1],
            "backend_evidence": _safe_backend_evidence(result),
        }
    result.update(
        {
            "segment_id": str(target["segment_id"]),
            "text": selected_text,
            "language": _display_language(language_route),
            "alignment_backend": "qwen-recovery",
            "alignment_unit": "token",
            "alignment_state": "completed_exact",
            "status": "completed",
            "error": None,
        }
    )
    return {
        "status": "completed_exact",
        "alignment_state": "completed_exact",
        "token_count": len(result.get("tokens", [])),
        "coverage": _coverage(result),
        "backend_evidence": _safe_backend_evidence(result),
        "aligned_segment": result,
    }


def _execute_mfa_local(
    work_paths: WorkPaths,
    aligned: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    selected_text: str,
    language_route: str,
    settings: dict[str, Any],
    runner: Callable[..., list[dict[str, Any]]] | None,
) -> dict[str, Any]:
    if language_route != "japanese":
        return {
            "status": "not_executable",
            "alignment_state": "failed",
            "error_code": "MFA_LANGUAGE_NOT_APPLICABLE",
            "error": "mfa-local is approved only for Japanese recovery candidates",
        }
    source_audio = _resolve_source_audio(work_paths, target)
    if source_audio is None:
        return {
            "status": "not_executable",
            "alignment_state": "failed",
            "error_code": "MFA_SOURCE_AUDIO_MISSING",
            "error": "source audio for mfa-local could not be resolved",
        }
    environment = detect_mfa_environment(run_version_check=True)
    if not environment.get("available"):
        return {
            "status": "not_executable",
            "alignment_state": "failed",
            "error_code": "MFA_ENVIRONMENT_UNAVAILABLE",
            "error": str(environment.get("reason") or "MFA environment is unavailable"),
            "backend_evidence": {"environment": environment},
        }
    candidate = {
        "source": "align-recovery",
        "reason": "initial-align-failed",
        "severity": "WARN",
        "subtitle_id": str(target["segment_id"]),
        "start_ms": round(float(target["global_start_time"]) * 1000),
        "end_ms": round(float(target["global_end_time"]) * 1000),
        "text": selected_text,
        "details": {"original_error": target.get("error")},
    }
    active_runner = runner or run_local_mfa_alignment_experiments
    run_paths = replace(work_paths, audio_path=source_audio)
    results = active_runner(
        run_paths,
        [candidate],
        environment=environment,
        max_run_candidates=1,
        padding_ms=max(0, int(settings.get("mfa_padding_ms", 700))),
    )
    mfa_result = results[0] if results else {"status": "skipped", "reason": "no-mfa-result"}
    ranges = [
        row
        for row in mfa_result.get("global_word_ranges", [])
        if isinstance(row, dict)
        and isinstance(row.get("start_ms"), int)
        and isinstance(row.get("end_ms"), int)
        and row["end_ms"] > row["start_ms"]
        and str(row.get("text", "")).strip().lower() not in {"<unk>", "unk"}
    ]
    if not mfa_result.get("usable") or not ranges:
        return {
            "status": "failed",
            "alignment_state": "failed",
            "error_code": "MFA_OUTPUT_UNUSABLE",
            "error": str(mfa_result.get("reason") or "mfa-local output is unusable"),
            "backend_evidence": _safe_mfa_evidence(mfa_result, environment),
        }
    recognized = "".join(str(row.get("text", "")) for row in ranges)
    score = local_ass_match_score(selected_text, recognized)
    minimum = float(settings.get("mfa_min_content_score", 0.70))
    if score < minimum:
        return {
            "status": "rejected",
            "alignment_state": "failed",
            "error_code": "MFA_CONTENT_MISMATCH",
            "error": f"mfa-local content score {score:.3f} is below {minimum:.3f}",
            "backend_evidence": {**_safe_mfa_evidence(mfa_result, environment), "recognized_text": recognized, "content_score": score},
        }
    tokens = [
        {"text": str(row.get("text", "")), "start_time": row["start_ms"] / 1000.0, "end_time": row["end_ms"] / 1000.0}
        for row in ranges
    ]
    replacement = {
        **target,
        "text": selected_text,
        "tokens": tokens,
        "status": "completed",
        "error": None,
        "alignment_backend": "mfa-local-recovery",
        "alignment_unit": "token",
        "alignment_state": "completed_exact",
        "alignment_coverage": _coverage({**target, "tokens": tokens}),
        "alignment_unknown_count": 0,
    }
    guard = _exact_result_guard(replacement, target, selected_text, require_token_text_identity=False)
    if guard is None:
        guard = _neighbor_guard(aligned, target, tokens)
    if guard is not None:
        return {
            "status": "rejected",
            "alignment_state": "failed",
            "error_code": guard[0],
            "error": guard[1],
            "backend_evidence": _safe_mfa_evidence(mfa_result, environment),
        }
    return {
        "status": "completed_exact",
        "alignment_state": "completed_exact",
        "token_count": len(tokens),
        "coverage": replacement["alignment_coverage"],
        "backend_evidence": {
            **_safe_mfa_evidence(mfa_result, environment),
            "recognized_text": recognized,
            "content_score": round(score, 6),
            "min_content_score": minimum,
        },
        "aligned_segment": replacement,
    }


def _exact_result_guard(
    result: dict[str, Any],
    target: dict[str, Any],
    selected_text: str,
    *,
    require_token_text_identity: bool = True,
) -> tuple[str, str] | None:
    tokens = result.get("tokens") if isinstance(result.get("tokens"), list) else []
    if not tokens:
        return "EXACT_TOKENS_REQUIRED", "exact recovery requires non-empty tokens"
    starts = []
    ends = []
    token_text = []
    for token in tokens:
        if not isinstance(token, dict):
            return "EXACT_TOKEN_INVALID", "exact recovery returned a non-object token"
        try:
            start = float(token["start_time"])
            end = float(token["end_time"])
        except (KeyError, TypeError, ValueError):
            return "EXACT_TOKEN_INVALID", "exact recovery returned invalid token timestamps"
        if end <= start:
            return "EXACT_TOKEN_NON_POSITIVE", "exact recovery returned a non-positive token"
        starts.append(start)
        ends.append(end)
        token_text.append(str(token.get("text", "")))
    if starts != sorted(starts):
        return "EXACT_TOKEN_NON_MONOTONIC", "exact recovery returned non-monotonic tokens"
    segment_start = float(target["global_start_time"])
    segment_end = float(target["global_end_time"])
    if min(starts) < segment_start or max(ends) > segment_end:
        return "EXACT_TOKEN_OUT_OF_RANGE", "exact recovery tokens exceed the original segment"
    timing_error = validate_aligned_token_timing(
        [
            AlignedToken(
                text=str(token.get("text", "")),
                start_time=float(token["start_time"]),
                end_time=float(token["end_time"]),
            )
            for token in tokens
        ],
        segment_start,
        segment_end,
        config=AlignTimingValidationConfig(),
    )
    if timing_error:
        return "EXACT_TIMING_UNRELIABLE", timing_error
    if require_token_text_identity and _normalize_text("".join(token_text)) != _normalize_text(selected_text):
        return "EXACT_CONTENT_MISMATCH", "exact recovery token text does not preserve the selected transcript"
    return None


def _neighbor_guard(
    aligned: list[dict[str, Any]], target: dict[str, Any], tokens: list[dict[str, Any]]
) -> tuple[str, str] | None:
    index = aligned.index(target)
    start = min(float(token["start_time"]) for token in tokens)
    end = max(float(token["end_time"]) for token in tokens)
    previous = aligned[index - 1] if index > 0 else None
    following = aligned[index + 1] if index + 1 < len(aligned) else None
    if previous is not None and derive_alignment_state(previous) != "failed":
        previous_end = float(previous.get("global_end_time", start))
        if start < previous_end:
            return "RECOVERY_NEIGHBOR_OVERLAP", "recovery token range overlaps the previous completed segment"
    if following is not None and derive_alignment_state(following) != "failed":
        following_start = float(following.get("global_start_time", end))
        if end > following_start:
            return "RECOVERY_NEIGHBOR_OVERLAP", "recovery token range overlaps the next completed segment"
    return None


def _select_strategy(strategy: str, language_route: str) -> str:
    if strategy != "auto":
        return strategy
    return "qwen"


def _normalize_language(value: Any) -> str:
    normalized = str(value or "auto").strip().lower()
    aliases = {"ja": "japanese", "jp": "japanese", "en": "english", "zh": "chinese"}
    return aliases.get(normalized, normalized)


def _display_language(value: str) -> str | None:
    return {"japanese": "Japanese", "english": "English", "chinese": "Chinese", "auto": None}.get(value, value)


def _resolve_source_audio(work_paths: WorkPaths, target: dict[str, Any]) -> Path | None:
    if work_paths.audio_path.exists():
        return work_paths.audio_path
    segment_audio = Path(str(target.get("audio_path", "")))
    candidate = segment_audio.parent.parent / "source.wav"
    return candidate if candidate.exists() else None


def _coverage(item: dict[str, Any]) -> float:
    tokens = item.get("tokens") if isinstance(item.get("tokens"), list) else []
    if not tokens:
        return 0.0
    start = min(float(token["start_time"]) for token in tokens)
    end = max(float(token["end_time"]) for token in tokens)
    duration = max(0.001, float(item["global_end_time"]) - float(item["global_start_time"]))
    return round(max(0.0, end - start) / duration, 6)


def _safe_backend_evidence(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "error": result.get("error"),
        "token_count": len(result.get("tokens", [])) if isinstance(result.get("tokens"), list) else 0,
        "coverage": _coverage(result) if isinstance(result.get("tokens"), list) and result.get("tokens") else 0.0,
        "raw_model_output": result.get("raw_model_output"),
    }


def _safe_mfa_evidence(result: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    return {
        "environment_available": bool(environment.get("available")),
        "status": result.get("status"),
        "reason": result.get("reason"),
        "elapsed_ms": result.get("elapsed_ms"),
        "clip": result.get("clip"),
        "lab": result.get("lab"),
        "lab_text": result.get("lab_text"),
        "word_quality": result.get("word_quality"),
        "stdout_tail": result.get("stdout_tail"),
        "stderr_tail": result.get("stderr_tail"),
    }


def _write_aligned_state(work_paths: WorkPaths, aligned: list[dict[str, Any]], replacement: dict[str, Any]) -> None:
    write_json_atomic(work_paths.aligned_manifest, aligned)
    write_json_atomic(work_paths.aligned_checkpoint_path, aligned)
    append_jsonl(
        work_paths.aligned_events_path,
        {"type": "aligned_recovery", "segment_id": replacement["segment_id"], "payload": replacement},
    )


def _read_list(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        payload = read_json(path, default=[])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecoveryExecutionError("RECOVERY_MANIFEST_CORRUPT", f"{label}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, list):
        raise RecoveryExecutionError("RECOVERY_MANIFEST_INVALID", f"{label} must be a JSON list")
    return [item for item in payload if isinstance(item, dict)]


@contextmanager
def _execution_lock(work_paths: WorkPaths, segment_id: str) -> Iterator[None]:
    path = work_paths.workdir / "reports" / "recovery-executor.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
        path.unlink()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RecoveryExecutionError("RECOVERY_CONFLICT", "another recovery execution is already active", status=409) from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "segment_id": segment_id, "started_at": _utc_now()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
