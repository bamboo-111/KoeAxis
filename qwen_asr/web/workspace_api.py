from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_asr.alignment_state import (
    ALIGN_STATES,
    MUSIC_REGION_STATE,
    derive_alignment_state,
    overlaps_music_region,
    read_music_region_evidence,
    seconds_to_ms,
)
from qwen_asr.final_quality_alignment import alignment_health_check
from qwen_asr.models import WorkPaths
from qwen_asr.recovery_service import RecoveryError, build_recovery_view, perform_recovery_action
from qwen_asr.review_service import (
    ReviewError,
    build_review_view,
    resolve_workspace_media,
    save_review_edit,
    undo_review_edit,
)
from qwen_asr.storage import read_json
from qwen_asr.web.commands import WORKSPACES_DIR
from qwen_asr.web.stage_service import build_stage_view
from qwen_asr.web.stage_start_service import StageStartError, build_workspace_stage_payload

API_VERSION = "v1"
SCHEMA_VERSION = 1


class WorkspaceApiError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status

    def as_payload(self) -> dict[str, Any]:
        return envelope(error={"code": self.code, "message": str(self)})


def envelope(*, data: Any = None, error: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "data": data,
        "error": error,
    }


def api_contract() -> dict[str, Any]:
    return envelope(
        data={
            "align_states": list(ALIGN_STATES),
            "music_region_state": MUSIC_REGION_STATE,
            "endpoints": {
                "GET /api/v1/contract": {"query": [], "response": "contract"},
                "GET /api/v1/job": {"query": [], "response": "job_state"},
                "GET /api/v1/workspaces": {"query": [], "response": "workspace_summary[]"},
                "GET /api/v1/workspace": {"query": ["workdir"], "response": "workspace_detail"},
                "GET /api/v1/workspace/stages": {"query": ["workdir"], "response": "stage_view"},
                "POST /api/v1/workspace/stage/start": {
                    "body": ["workdir", "stage", "settings"],
                    "response": "job_state",
                },
                "GET /api/v1/workspace/align": {"query": ["workdir"], "response": "align_state"},
                "GET /api/v1/workspace/recovery": {"query": ["workdir"], "response": "recovery_queue"},
                "GET /api/v1/workspace/review": {"query": ["workdir"], "response": "review_view"},
                "POST /api/v1/workspace/review/edit": {
                    "body": ["workdir", "cue_id", "original", "translation", "start_ms", "end_ms", "expected_revision", "actor"],
                    "response": "review_edit_result",
                },
                "POST /api/v1/workspace/review/undo": {
                    "body": ["workdir", "expected_revision", "actor"],
                    "response": "review_undo_result",
                },
                "GET /api/v1/workspace/media": {"query": ["workdir", "path"], "response": "binary_range"},
                "POST /api/v1/workspace/recovery/action": {
                    "body": ["workdir", "segment_id", "action", "payload", "actor"],
                    "response": "recovery_action_result",
                },
                "GET /api/v1/workspace/quality": {"query": ["workdir"], "response": "quality_gate"},
                "GET /api/v1/workspace/quality-evidence": {
                    "query": ["workdir", "path"],
                    "response": "quality_evidence_file",
                },
                "GET /api/v1/workspace/exports": {"query": ["workdir"], "response": "export_artifact[]"},
                "GET /api/v1/workspace/export-file": {
                    "query": ["workdir", "path", "download"],
                    "response": "subtitle_file",
                },
            },
            "error_model": {"code": "ASCII_STABLE_CODE", "message": "UTF-8 message"},
            "security": {
                "workspace_scope": "first-level directories below workspaces",
                "secrets": "environment-only and never serialized",
            },
        }
    )


def list_workspace_summaries() -> dict[str, Any]:
    root = WORKSPACES_DIR.resolve()
    if not root.exists():
        return envelope(data=[])
    items = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not _looks_like_workspace(path):
            continue
        metadata, metadata_error = _safe_read_dict(path / "project.json")
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "workdir": str(path.resolve()),
                "source_name": str(metadata.get("source_name", path.name)),
                "modified_at": stat.st_mtime,
                "metadata_status": "corrupt" if metadata_error else "available",
            }
        )
    return envelope(data=items)


def get_workspace_detail(workdir_value: str) -> dict[str, Any]:
    work_paths = _resolve_workspace(workdir_value)
    metadata, metadata_error = _safe_read_dict(work_paths.project_metadata)
    align = _build_align_state(work_paths)
    recovery = envelope(data=build_recovery_view(work_paths))
    quality = _build_quality_gate(work_paths)
    exports = _build_exports(work_paths, metadata)
    normalized, normalized_error = _safe_read_mapping(work_paths.normalized_manifest)
    return envelope(
        data={
            "workdir": str(work_paths.workdir),
            "name": work_paths.workdir.name,
            "source_name": str(metadata.get("source_name", work_paths.workdir.name)),
            "metadata_status": "corrupt" if metadata_error else "available",
            "metadata_error": metadata_error,
            "stages": build_stage_view(work_paths),
            "align": align["data"],
            "recovery": recovery["data"],
            "quality": quality["data"],
            "exports": exports["data"],
            "cues": {
                "normalized": len(normalized),
                "status": "corrupt" if normalized_error else "available" if normalized else "missing",
                "error": normalized_error,
            },
        }
    )


def get_align_state(workdir_value: str) -> dict[str, Any]:
    return _build_align_state(_resolve_workspace(workdir_value))


def get_stage_view(workdir_value: str) -> dict[str, Any]:
    return envelope(data=build_stage_view(_resolve_workspace(workdir_value)))


def prepare_workspace_stage_start(
    workdir_value: str,
    *,
    stage: str,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        return build_workspace_stage_payload(
            _resolve_workspace(workdir_value),
            stage=stage,
            settings=settings,
        )
    except StageStartError as exc:
        raise WorkspaceApiError(exc.code, str(exc), status=exc.status) from exc


def get_recovery_queue(workdir_value: str) -> dict[str, Any]:
    work_paths = _resolve_workspace(workdir_value)
    return envelope(data=build_recovery_view(work_paths))


def apply_recovery_action(
    workdir_value: str,
    *,
    segment_id: str,
    action: str,
    payload: dict[str, Any] | None = None,
    actor: str = "web-local-user",
) -> dict[str, Any]:
    try:
        result = perform_recovery_action(
            _resolve_workspace(workdir_value),
            segment_id=segment_id,
            action=action,
            payload=payload,
            actor=actor,
        )
    except RecoveryError as exc:
        raise WorkspaceApiError(exc.code, str(exc), status=exc.status) from exc
    return envelope(data=result)


def get_quality_gate(workdir_value: str) -> dict[str, Any]:
    return _build_quality_gate(_resolve_workspace(workdir_value))


def get_quality_evidence_path(workdir_value: str, path_value: str) -> Path:
    work_paths = _resolve_workspace(workdir_value)
    quality = _build_quality_gate(work_paths)["data"]
    allowed: set[Path] = set()
    evidence_path = quality.get("evidence_path")
    if evidence_path:
        allowed.add(Path(str(evidence_path)).resolve())
    for check in quality.get("checks", []):
        if not isinstance(check, dict):
            continue
        target = check.get("target")
        if isinstance(target, dict) and target.get("path"):
            allowed.add(Path(str(target["path"])).resolve())
    target = Path(str(path_value or "").strip()).resolve()
    if target not in allowed or not _is_relative_to(target, work_paths.workdir.resolve()):
        raise WorkspaceApiError(
            "QUALITY_EVIDENCE_OUT_OF_SCOPE",
            "quality evidence path is not in the selected workspace quality inventory",
            status=403,
        )
    if not target.exists() or not target.is_file():
        raise WorkspaceApiError("QUALITY_EVIDENCE_NOT_FOUND", "quality evidence file does not exist", status=404)
    if target.suffix.lower() not in {".json", ".jsonl", ".log", ".md", ".txt", ".csv"}:
        raise WorkspaceApiError("QUALITY_EVIDENCE_TYPE_NOT_ALLOWED", "quality evidence file type is not allowed", status=403)
    return target


def get_review_view(workdir_value: str) -> dict[str, Any]:
    return envelope(data=build_review_view(_resolve_workspace(workdir_value)))


def apply_review_edit(
    workdir_value: str,
    *,
    cue_id: str,
    original: str,
    translation: str,
    start_ms: Any,
    end_ms: Any,
    expected_revision: Any = None,
    actor: str = "web-local-user",
) -> dict[str, Any]:
    try:
        result = save_review_edit(
            _resolve_workspace(workdir_value),
            cue_id=cue_id,
            original=original,
            translation=translation,
            start_ms=start_ms,
            end_ms=end_ms,
            expected_revision=expected_revision,
            actor=actor,
        )
    except ReviewError as exc:
        raise WorkspaceApiError(exc.code, str(exc), status=exc.status) from exc
    return envelope(data=result)


def apply_review_undo(
    workdir_value: str,
    *,
    expected_revision: Any = None,
    actor: str = "web-local-user",
) -> dict[str, Any]:
    try:
        result = undo_review_edit(
            _resolve_workspace(workdir_value),
            expected_revision=expected_revision,
            actor=actor,
        )
    except ReviewError as exc:
        raise WorkspaceApiError(exc.code, str(exc), status=exc.status) from exc
    return envelope(data=result)


def get_workspace_media_path(workdir_value: str, path_value: str) -> Path:
    try:
        return resolve_workspace_media(_resolve_workspace(workdir_value), path_value)
    except ReviewError as exc:
        raise WorkspaceApiError(exc.code, str(exc), status=exc.status) from exc


def get_exports(workdir_value: str) -> dict[str, Any]:
    work_paths = _resolve_workspace(workdir_value)
    metadata, _ = _safe_read_dict(work_paths.project_metadata)
    return _build_exports(work_paths, metadata)


def get_export_file_path(workdir_value: str, path_value: str) -> Path:
    work_paths = _resolve_workspace(workdir_value)
    metadata, _ = _safe_read_dict(work_paths.project_metadata)
    allowed = {Path(item["path"]).resolve() for item in _build_exports(work_paths, metadata)["data"]}
    target = Path(str(path_value or "").strip()).resolve()
    if target not in allowed:
        raise WorkspaceApiError(
            "EXPORT_PATH_OUT_OF_SCOPE",
            "export path is not in the selected workspace export inventory",
            status=403,
        )
    return target


def _resolve_workspace(workdir_value: str) -> WorkPaths:
    raw = str(workdir_value or "").strip()
    if not raw:
        raise WorkspaceApiError("WORKDIR_REQUIRED", "workdir is required")
    target = Path(raw).resolve()
    root = WORKSPACES_DIR.resolve()
    if target.parent != root:
        raise WorkspaceApiError(
            "WORKDIR_OUT_OF_SCOPE",
            "workdir must be a first-level directory under workspaces",
            status=403,
        )
    if not target.exists() or not target.is_dir():
        raise WorkspaceApiError("WORKSPACE_NOT_FOUND", "workspace does not exist", status=404)
    return WorkPaths.from_workdir(target)


def _looks_like_workspace(path: Path) -> bool:
    return (path / "project.json").exists() or (path / "manifests").is_dir() or (path / "progress.json").exists()


def _build_align_state(work_paths: WorkPaths) -> dict[str, Any]:
    aligned, manifest_error = _safe_read_list(work_paths.aligned_manifest)
    intervals, evidence_path, evidence_summary, evidence_error = read_music_region_evidence(work_paths.workdir)
    rows: list[dict[str, Any]] = []
    raw_counts: Counter[str] = Counter()
    dialogue_counts: Counter[str] = Counter()
    excluded_count = 0
    for item in aligned:
        state = derive_alignment_state(item)
        raw_counts[state] += 1
        region = overlaps_music_region(item, intervals)
        dialogue_state = MUSIC_REGION_STATE if region else state
        if region:
            excluded_count += 1
        else:
            dialogue_counts[state] += 1
        rows.append(
            {
                "segment_id": str(item.get("segment_id", "")),
                "state": dialogue_state,
                "raw_state": state,
                "text": str(item.get("text", "")),
                "language": item.get("language"),
                "start_ms": seconds_to_ms(item.get("global_start_time")),
                "end_ms": seconds_to_ms(item.get("global_end_time")),
                "audio_path": str(item.get("audio_path", "")),
                "token_count": len(item.get("tokens", [])) if isinstance(item.get("tokens"), list) else 0,
                "coverage": item.get("alignment_coverage"),
                "error": str(item.get("error", "") or ""),
                "music_region": region,
            }
        )
    return envelope(
        data={
            "manifest_status": "corrupt" if manifest_error else "available" if aligned else "missing",
            "manifest_error": manifest_error,
            "source": str(work_paths.aligned_manifest),
            "raw_counts": _state_counts(raw_counts),
            "dialogue_counts": _state_counts(dialogue_counts),
            "excluded_music_region_count": excluded_count,
            "music_regions": intervals,
            "music_region_evidence": evidence_path,
            "music_region_evidence_summary": evidence_summary,
            "music_region_evidence_error": evidence_error,
            "segments": rows,
        }
    )


def _build_quality_gate(work_paths: WorkPaths) -> dict[str, Any]:
    candidates = [work_paths.final_quality_report]
    candidates.extend(sorted((work_paths.workdir / "reports").glob("final_quality*.json")))
    seen: set[Path] = set()
    errors: list[dict[str, str]] = []
    for path in reversed(candidates):
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        payload, error = _safe_read_dict(path)
        if error:
            errors.append({"path": str(path), "error": error})
            continue
        if payload:
            checks = payload.get("checks", [])
            if isinstance(checks, dict):
                checks = [{"name": key, **value} if isinstance(value, dict) else {"name": key, "value": value} for key, value in checks.items()]
            checks = checks if isinstance(checks, list) else []
            if not checks and str(payload.get("status", "")).upper() == "FAIL":
                checks.append(
                    {
                        "name": "reported_quality_status",
                        "status": "FAIL",
                        "message": "Saved quality report status is FAIL.",
                    }
                )
            live_alignment = alignment_health_check(work_paths)
            checks = [item for item in checks if not isinstance(item, dict) or item.get("name") != "alignment_health"]
            checks.append(live_alignment)
            checks = [{**item, "target": _quality_target(item, path)} for item in checks if isinstance(item, dict)]
            summary = {
                "pass_count": sum(str(item.get("status", "")).upper() == "PASS" for item in checks),
                "warn_count": sum(str(item.get("status", "")).upper() == "WARN" for item in checks),
                "fail_count": sum(str(item.get("status", "")).upper() == "FAIL" for item in checks),
            }
            status = "FAIL" if summary["fail_count"] else "WARN" if summary["warn_count"] else "PASS"
            current_payload = {**payload, "status": status, "summary": summary, "checks": checks}
            return envelope(
                data={
                    "status": status,
                    "summary": summary,
                    "checks": checks,
                    "blocking_reasons": _quality_blocking_reasons(current_payload),
                    "evidence_path": str(path),
                    "evidence_status": str(payload.get("status", "UNKNOWN")).upper(),
                    "read_errors": errors,
                }
            )
    return envelope(
        data={
            "status": "UNKNOWN",
            "summary": {},
            "checks": [],
            "blocking_reasons": [],
            "evidence_path": None,
            "read_errors": errors,
        }
    )


def _build_exports(work_paths: WorkPaths, metadata: dict[str, Any]) -> dict[str, Any]:
    candidates: set[Path] = set()
    for path in (work_paths.subtitles_srt, work_paths.subtitles_vtt, work_paths.normalized_srt):
        if path.exists():
            candidates.add(path.resolve())
    for directory in (work_paths.workdir / "exports", work_paths.workdir / "export-cache"):
        if directory.is_dir():
            candidates.update(path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".srt", ".vtt"})
    last_exported = metadata.get("last_exported", {})
    if isinstance(last_exported, dict):
        for raw in last_exported.values():
            path = Path(str(raw)).resolve()
            if path.exists() and path.is_file() and path.suffix.lower() in {".srt", ".vtt"}:
                candidates.add(path)
    quality = _build_quality_gate(work_paths)["data"]
    items = []
    for path in sorted(candidates):
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "format": path.suffix.lower().lstrip("."),
                "path": str(path),
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
                "quality_status": quality.get("status", "UNKNOWN"),
                "delivery_state": "formal" if quality.get("status") in {"PASS", "WARN"} else "quality_gate_failed",
                "previewable": True,
            }
        )
    return envelope(data=items)


def _quality_blocking_reasons(payload: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = []
    checks = payload.get("checks", [])
    if isinstance(checks, dict):
        iterable = checks.items()
    elif isinstance(checks, list):
        iterable = ((str(item.get("name", item.get("id", "check"))), item) for item in checks if isinstance(item, dict))
    else:
        iterable = []
    for name, check in iterable:
        if not isinstance(check, dict):
            continue
        if str(check.get("status", "")).upper() == "FAIL":
            reasons.append({"name": name, "message": check.get("message", check.get("reason", "")), "evidence": check.get("evidence")})
    return reasons


def _quality_target(check: dict[str, Any], fallback_path: Path) -> dict[str, Any]:
    name = str(check.get("name", ""))
    if name == "alignment_health":
        return {
            "view": "recovery",
            "segment_ids": list(check.get("failed_segment_ids", [])),
        }
    issues = [item for item in check.get("issues", []) if isinstance(item, dict)]
    cue_ids = [str(item["key"]) for item in issues if item.get("key") is not None]
    if cue_ids:
        return {"view": "review", "cue_ids": list(dict.fromkeys(cue_ids))}
    report = check.get("report") or check.get("evidence")
    return {"view": "evidence", "path": str(report or fallback_path)}


def _state_counts(counter: Counter[str]) -> dict[str, int]:
    return {state: int(counter.get(state, 0)) for state in ALIGN_STATES}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_read_list(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    try:
        payload = read_json(path, default=[])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, list):
        return [], "expected JSON list"
    return [item for item in payload if isinstance(item, dict)], None


def _safe_read_dict(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        payload = read_json(path, default={})
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return {}, "expected JSON object"
    return payload, None


def _safe_read_mapping(path: Path) -> tuple[dict[str, Any], str | None]:
    return _safe_read_dict(path)
