from __future__ import annotations

import json
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_asr.alignment_state import derive_alignment_state, overlaps_music_region, read_music_region_evidence
from qwen_asr.models import WorkPaths
from qwen_asr.storage import append_jsonl, read_json, write_json_atomic
from qwen_asr.vad import create_vad_adapter

RECOVERY_REPORT_NAME = "failed_segment_recovery.json"
RECOVERY_SCHEMA_VERSION = 1
ACTIONS = (
    "verify_transcript",
    "localize_vad",
    "route_language",
    "retry_align",
    "accept_completed_coarse",
)


class RecoveryError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def build_recovery_view(work_paths: WorkPaths) -> dict[str, Any]:
    aligned = _read_aligned(work_paths)
    transcripts = _read_optional_list(work_paths.transcript_manifest)
    transcript_by_id = {str(item.get("segment_id", "")): item for item in transcripts}
    intervals, evidence_path, evidence_summary, evidence_error = read_music_region_evidence(work_paths.workdir)
    state, state_error = _read_state(work_paths)
    saved_tasks = state.get("tasks", {}) if isinstance(state.get("tasks"), dict) else {}
    reference_sources = _reference_sources(work_paths)
    items = []
    for index, item in enumerate(aligned):
        if derive_alignment_state(item) != "failed" or overlaps_music_region(item, intervals):
            continue
        segment_id = str(item.get("segment_id", ""))
        saved = saved_tasks.get(segment_id, {}) if isinstance(saved_tasks.get(segment_id), dict) else {}
        normalized_chars = _normalized_char_count(str(item.get("text", "")))
        previous_item = aligned[index - 1] if index > 0 else None
        next_item = aligned[index + 1] if index + 1 < len(aligned) else None
        transcript = transcript_by_id.get(segment_id, {})
        task = {
            "recovery_id": segment_id,
            "segment_id": segment_id,
            "status": str(saved.get("status", "pending")),
            "priority": "short_response" if normalized_chars <= 4 else "standard",
            "normalized_char_count": normalized_chars,
            "reason_codes": _reason_codes(item, normalized_chars),
            "text": str(item.get("text", "")),
            "original_transcript": str(transcript.get("text", item.get("text", ""))),
            "verified_text": saved.get("verified_text"),
            "language": item.get("language"),
            "language_route": saved.get("language_route"),
            "start_ms": _seconds_to_ms(item.get("global_start_time")),
            "end_ms": _seconds_to_ms(item.get("global_end_time")),
            "audio_path": str(item.get("audio_path", "")),
            "token_count": len(item.get("tokens", [])) if isinstance(item.get("tokens"), list) else 0,
            "coverage": item.get("alignment_coverage"),
            "error": str(item.get("error", "") or ""),
            "context": {
                "previous": _context_row(previous_item),
                "next": _context_row(next_item),
            },
            "vad_proposal": saved.get("vad_proposal"),
            "reference_sources": reference_sources,
            "available_actions": list(ACTIONS),
            "last_action_at": saved.get("last_action_at"),
        }
        items.append(task)
    items.sort(key=lambda item: (item["priority"] != "short_response", item["start_ms"] or 0, item["segment_id"]))
    resolved = [value for value in saved_tasks.values() if isinstance(value, dict) and value.get("status") == "completed_coarse"]
    return {
        "status": "available",
        "total": len(items),
        "short_response_count": sum(1 for item in items if item["priority"] == "short_response"),
        "items": items,
        "resolved": resolved,
        "audit": state.get("audit", []) if isinstance(state.get("audit"), list) else [],
        "state_path": str(_state_path(work_paths)),
        "state_error": state_error,
        "music_region_evidence": evidence_path,
        "music_region_evidence_summary": evidence_summary,
        "music_region_evidence_error": evidence_error,
        "policy": "every failed dialogue segment is queued; music regions are excluded",
    }


def perform_recovery_action(
    work_paths: WorkPaths,
    *,
    segment_id: str,
    action: str,
    payload: dict[str, Any] | None = None,
    actor: str = "web-local-user",
) -> dict[str, Any]:
    normalized_action = str(action or "").strip()
    if normalized_action not in ACTIONS:
        raise RecoveryError("RECOVERY_ACTION_INVALID", f"unsupported recovery action: {normalized_action}")
    payload = payload if isinstance(payload, dict) else {}
    view = build_recovery_view(work_paths)
    task = next((item for item in view["items"] if item["segment_id"] == segment_id), None)
    if task is None:
        raise RecoveryError("RECOVERY_TASK_NOT_FOUND", "failed dialogue recovery task does not exist", status=404)
    state, _ = _read_state(work_paths)
    state.setdefault("schema_version", RECOVERY_SCHEMA_VERSION)
    state.setdefault("tasks", {})
    state.setdefault("audit", [])
    saved = state["tasks"].setdefault(segment_id, {"segment_id": segment_id, "status": "pending"})
    before = dict(saved)
    action_input: dict[str, Any] = {}
    result: dict[str, Any]
    if normalized_action == "verify_transcript":
        verified_text = str(payload.get("verified_text", "")).strip()
        if not verified_text:
            raise RecoveryError("VERIFIED_TEXT_REQUIRED", "verified_text is required")
        saved["verified_text"] = verified_text
        saved["status"] = "verified"
        action_input = {"verified_text": verified_text}
        result = {"verified": True}
    elif normalized_action == "route_language":
        language = str(payload.get("language", "")).strip()
        if not language:
            raise RecoveryError("LANGUAGE_REQUIRED", "language is required")
        saved["language_route"] = language
        saved["status"] = "language_routed"
        action_input = {"language": language}
        result = {"language": language}
    elif normalized_action == "retry_align":
        strategy = str(payload.get("strategy", "qwen"))
        saved["retry_strategy"] = strategy
        saved["status"] = "retry_requested"
        action_input = {"strategy": strategy}
        result = {"retry_requested": True, "strategy": strategy}
    elif normalized_action == "localize_vad":
        proposal = _localize_with_vad(task, payload)
        saved["vad_proposal"] = proposal
        saved["status"] = "localized"
        action_input = {
            "backend": proposal["backend"],
            "threshold": proposal.get("threshold"),
        }
        result = proposal
    else:
        proposal = _coarse_proposal(task, saved, payload)
        backup_path = _apply_completed_coarse(work_paths, task, proposal)
        saved["status"] = "completed_coarse"
        saved["result"] = {
            "alignment_state": "completed_coarse",
            "start_ms": proposal["start_ms"],
            "end_ms": proposal["end_ms"],
            "backup_path": backup_path,
        }
        action_input = {"start_ms": proposal["start_ms"], "end_ms": proposal["end_ms"]}
        result = dict(saved["result"])
    now = _utc_now()
    saved["last_action"] = normalized_action
    saved["last_action_at"] = now
    saved["actor"] = actor
    audit_entry = {
        "id": f"{segment_id}:{int(time.time() * 1000)}",
        "segment_id": segment_id,
        "actor": actor,
        "timestamp": now,
        "action": normalized_action,
        "input": action_input,
        "strategy": result.get("backend", result.get("strategy", normalized_action)),
        "result": result,
        "before": before,
        "after": dict(saved),
        "evidence_path": str(_state_path(work_paths)),
    }
    state["audit"].append(audit_entry)
    state["updated_at"] = now
    write_json_atomic(_state_path(work_paths), state)
    return {
        "task": dict(saved),
        "audit": audit_entry,
        "recovery": build_recovery_view(work_paths),
    }


def _localize_with_vad(task: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(task.get("audio_path", "")))
    if not audio_path.exists():
        raise RecoveryError("RECOVERY_AUDIO_MISSING", "segment audio file does not exist", status=409)
    backend = str(payload.get("backend", "pyannote_onnx_v3"))
    threshold = float(payload.get("threshold", 0.5))
    adapter = create_vad_adapter(
        backend=backend,
        threshold=threshold,
        onset=float(payload.get("onset", 0.5)),
        offset=float(payload.get("offset", 0.35)),
        min_speech_duration_ms=max(1, int(payload.get("min_speech_duration_ms", 80))),
        min_silence_duration_ms=max(1, int(payload.get("min_silence_duration_ms", 120))),
        speech_pad_ms=max(0, int(payload.get("speech_pad_ms", 80))),
    )
    try:
        regions = adapter.detect(audio_path)
    except Exception as exc:  # provider/runtime boundary
        raise RecoveryError("VAD_LOCALIZATION_FAILED", str(exc), status=409) from exc
    if not regions:
        raise RecoveryError("VAD_NO_SPEECH", "VAD found no speech in the failed segment", status=409)
    local_start = min(float(region.start_time) for region in regions)
    local_end = max(float(region.end_time) for region in regions)
    segment_start = int(task["start_ms"])
    segment_end = int(task["end_ms"])
    start_ms = max(segment_start, segment_start + round(local_start * 1000))
    end_ms = min(segment_end, segment_start + round(local_end * 1000))
    if end_ms <= start_ms:
        raise RecoveryError("VAD_BOUNDS_INVALID", "VAD produced non-positive global bounds", status=409)
    return {
        "backend": backend,
        "threshold": threshold,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "region_count": len(regions),
    }


def _coarse_proposal(task: dict[str, Any], saved: dict[str, Any], payload: dict[str, Any]) -> dict[str, int]:
    source = payload if payload.get("start_ms") is not None and payload.get("end_ms") is not None else saved.get("vad_proposal")
    if not isinstance(source, dict):
        raise RecoveryError("COARSE_BOUNDS_REQUIRED", "run localize_vad or provide start_ms/end_ms first")
    try:
        start_ms = int(source["start_ms"])
        end_ms = int(source["end_ms"])
        segment_start = int(task["start_ms"])
        segment_end = int(task["end_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RecoveryError("COARSE_BOUNDS_INVALID", "coarse bounds must be integer milliseconds") from exc
    if start_ms < segment_start or end_ms > segment_end or end_ms <= start_ms:
        raise RecoveryError("COARSE_BOUNDS_OUT_OF_RANGE", "coarse bounds must stay inside the failed segment")
    return {"start_ms": start_ms, "end_ms": end_ms}


def _apply_completed_coarse(
    work_paths: WorkPaths,
    task: dict[str, Any],
    proposal: dict[str, int],
) -> str:
    aligned = _read_aligned(work_paths)
    target = next((item for item in aligned if str(item.get("segment_id", "")) == task["segment_id"]), None)
    if target is None:
        raise RecoveryError("ALIGNED_SEGMENT_NOT_FOUND", "aligned manifest segment is missing", status=409)
    backup_dir = _backup_alignment_state(work_paths, task["segment_id"])
    original_error = str(target.get("error", "") or "")
    target.update(
        {
            "global_start_time": proposal["start_ms"] / 1000.0,
            "global_end_time": proposal["end_ms"] / 1000.0,
            "tokens": [],
            "status": "completed",
            "error": None,
            "alignment_backend": "vad-local",
            "alignment_unit": "segment",
            "alignment_state": "completed_coarse",
            "alignment_failure_reason": original_error,
            "recovery": {
                "method": "vad-local",
                "accepted_at": _utc_now(),
                "original_error": original_error,
            },
        }
    )
    write_json_atomic(work_paths.aligned_manifest, aligned)
    write_json_atomic(work_paths.aligned_checkpoint_path, aligned)
    append_jsonl(
        work_paths.aligned_events_path,
        {"type": "aligned", "segment_id": task["segment_id"], "payload": target},
    )
    return str(backup_dir)


def _backup_alignment_state(work_paths: WorkPaths, segment_id: str) -> Path:
    backup_dir = (
        work_paths.workdir
        / "reports"
        / "recovery-backups"
        / f"{segment_id}-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
    )
    backup_dir.mkdir(parents=True, exist_ok=False)
    registered = []
    for path in (
        work_paths.aligned_manifest,
        work_paths.aligned_checkpoint_path,
        work_paths.aligned_events_path,
    ):
        if path.exists():
            shutil.copy2(path, backup_dir / path.name)
            registered.append(str(path))
    (backup_dir / "BACKED_UP_FILES.txt").write_text("\n".join(registered) + "\n", encoding="utf-8")
    return backup_dir


def _read_aligned(work_paths: WorkPaths) -> list[dict[str, Any]]:
    try:
        payload = read_json(work_paths.aligned_manifest, default=[])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RecoveryError("ALIGNED_MANIFEST_CORRUPT", f"{type(exc).__name__}: {exc}", status=409) from exc
    if not isinstance(payload, list):
        raise RecoveryError("ALIGNED_MANIFEST_INVALID", "aligned manifest must be a JSON list", status=409)
    return [item for item in payload if isinstance(item, dict)]


def _read_optional_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = read_json(path, default=[])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _read_state(work_paths: WorkPaths) -> tuple[dict[str, Any], str | None]:
    path = _state_path(work_paths)
    try:
        payload = read_json(path, default={})
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return (payload, None) if isinstance(payload, dict) else ({}, "expected JSON object")


def _state_path(work_paths: WorkPaths) -> Path:
    return work_paths.workdir / "reports" / RECOVERY_REPORT_NAME


def _reference_sources(work_paths: WorkPaths) -> list[dict[str, str]]:
    root = work_paths.workdir / "references"
    if not root.is_dir():
        return []
    return [
        {"name": path.name, "path": str(path), "mode": "read_only"}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".ass", ".ssa", ".srt", ".vtt"}
    ]


def _reason_codes(item: dict[str, Any], normalized_chars: int) -> list[str]:
    reasons = []
    if normalized_chars <= 4:
        reasons.append("short_response")
    tokens = item.get("tokens", []) if isinstance(item.get("tokens"), list) else []
    if not tokens:
        reasons.append("no_tokens")
    if any(_token_non_positive(token) for token in tokens if isinstance(token, dict)):
        reasons.append("zero_duration_token")
    coverage = item.get("alignment_coverage")
    if isinstance(coverage, (int, float)) and coverage < 0.5:
        reasons.append("low_coverage")
    text = str(item.get("text", ""))
    if _mixed_language(text):
        reasons.append("mixed_language")
    error = str(item.get("error", "")).lower()
    if "changed transcript" in error or "content" in error:
        reasons.append("short_window_rewrite")
    if "timing" in error or "duration" in error:
        reasons.append("timing_unreliable")
    return reasons or ["align_failed"]


def _token_non_positive(token: dict[str, Any]) -> bool:
    try:
        return float(token.get("end_time", 0)) <= float(token.get("start_time", 0))
    except (TypeError, ValueError):
        return True


def _mixed_language(text: str) -> bool:
    has_latin = any("LATIN" in unicodedata.name(char, "") for char in text if char.isalpha())
    has_cjk = any(
        "CJK" in unicodedata.name(char, "") or "HIRAGANA" in unicodedata.name(char, "") or "KATAKANA" in unicodedata.name(char, "")
        for char in text
    )
    return has_latin and has_cjk


def _normalized_char_count(value: str) -> int:
    return sum(1 for char in value if unicodedata.category(char)[0] in {"L", "N"})


def _context_row(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    return {
        "segment_id": str(item.get("segment_id", "")),
        "text": str(item.get("text", "")),
        "start_ms": _seconds_to_ms(item.get("global_start_time")),
        "end_ms": _seconds_to_ms(item.get("global_end_time")),
    }


def _seconds_to_ms(value: Any) -> int | None:
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
