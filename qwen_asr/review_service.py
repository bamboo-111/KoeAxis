from __future__ import annotations

import json
import shutil
import threading
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qwen_asr.alignment_state import (
    MUSIC_REGION_STATE,
    derive_alignment_state,
    overlaps_music_region,
    read_music_region_evidence,
)
from qwen_asr.history_glossary import parse_ass_dialogues
from qwen_asr.models import WorkPaths
from qwen_asr.storage import append_jsonl, read_json, write_json_atomic

MEDIA_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".mp4", ".webm"}
REVIEW_SCHEMA_VERSION = 1
MAX_UNDO_DEPTH = 100
REVIEW_LOCK = threading.Lock()


class ReviewError(RuntimeError):
    def __init__(self, code: str, message: str, *, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


def build_review_view(work_paths: WorkPaths) -> dict[str, Any]:
    draft, draft_error = _load_review_draft(work_paths)
    if draft is not None:
        source_path = review_draft_path(work_paths)
        source_payload = draft["cues"]
        source_error = None
    else:
        source_path, source_payload, source_error = _select_cue_source(work_paths)
    translated = _safe_mapping(work_paths.translated_manifest)
    aligned = _safe_list(work_paths.aligned_manifest)
    intervals, evidence_path, _, evidence_error = read_music_region_evidence(work_paths.workdir)
    references = _load_references(work_paths)
    cues = []
    for key, value in source_payload.items():
        if not isinstance(value, dict):
            continue
        start_ms = _int_or_none(value.get("start_time"))
        end_ms = _int_or_none(value.get("end_time"))
        if start_ms is None or end_ms is None:
            continue
        original = str(value.get("original_subtitle", value.get("text", "")))
        translation = str(value.get("translated_subtitle", ""))
        translated_item = translated.get(str(key), {}) if isinstance(translated.get(str(key)), dict) else {}
        aligned_item = _best_aligned_segment(aligned, start_ms, end_ms)
        alignment_state = derive_alignment_state(aligned_item) if aligned_item else "unknown"
        if aligned_item and overlaps_music_region(aligned_item, intervals):
            alignment_state = MUSIC_REGION_STATE
        flags = _review_flags(value, translated_item, alignment_state)
        if draft is not None and str(key) in set(draft.get("edited_cue_ids", [])):
            flags.append("web_review_edited")
        cues.append(
            {
                "cue_id": str(key),
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": end_ms - start_ms,
                "original": original,
                "translation": translation,
                "segment_id": str(aligned_item.get("segment_id", "")) if aligned_item else None,
                "alignment_state": alignment_state,
                "flags": flags,
                "reference": _reference_matches(references, start_ms, end_ms),
            }
        )
    cues.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["cue_id"]))
    audio_path = _review_audio_path(work_paths, aligned)
    return {
        "source": str(source_path) if source_path else None,
        "base_source": draft.get("base_source") if draft else str(source_path) if source_path else None,
        "source_error": source_error,
        "draft_error": draft_error,
        "review_state": _review_state(work_paths, draft),
        "cue_count": len(cues),
        "audio_path": str(audio_path) if audio_path else None,
        "music_region_evidence": evidence_path,
        "music_region_evidence_error": evidence_error,
        "reference_sources": [
            {"name": item["name"], "path": item["path"], "cue_count": len(item["dialogues"]), "mode": "read_only"}
            for item in references
        ],
        "cues": cues,
    }


def review_draft_path(work_paths: WorkPaths) -> Path:
    return work_paths.workdir / "drafts" / "web-review.json"


def review_history_path(work_paths: WorkPaths) -> Path:
    return work_paths.workdir / "reports" / "web_review_history.jsonl"


def review_is_dirty(work_paths: WorkPaths) -> bool:
    draft, _ = _load_review_draft(work_paths)
    return bool(draft and draft.get("dirty"))


def save_review_edit(
    work_paths: WorkPaths,
    *,
    cue_id: str,
    original: str,
    translation: str,
    start_ms: Any,
    end_ms: Any,
    expected_revision: Any = None,
    actor: str = "web-local-user",
) -> dict[str, Any]:
    with REVIEW_LOCK:
        draft, draft_error = _load_review_draft(work_paths)
        if draft_error:
            raise ReviewError("REVIEW_DRAFT_CORRUPT", draft_error, status=409)
        if draft is None:
            source_path, source_payload, source_error = _select_cue_source(work_paths)
            if source_path is None or not source_payload:
                raise ReviewError("REVIEW_SOURCE_MISSING", source_error or "review source is missing", status=404)
            draft = _new_review_draft(source_path, source_payload)
        _check_revision(draft, expected_revision)
        cue_key = str(cue_id or "").strip()
        cues = draft["cues"]
        if cue_key not in cues or not isinstance(cues[cue_key], dict):
            raise ReviewError("REVIEW_CUE_NOT_FOUND", "review cue does not exist", status=404)
        after = _validated_edit(cues, cue_key, original, translation, start_ms, end_ms)
        before = _cue_snapshot(cues[cue_key])
        if before == after:
            return {"changed": False, "review": build_review_view(work_paths)}
        backup_path = _backup_review_draft(work_paths, int(draft.get("revision", 0)))
        _apply_cue_snapshot(cues[cue_key], after)
        revision = int(draft.get("revision", 0)) + 1
        changed_at = time.time()
        entry = {
            "cue_id": cue_key,
            "before": before,
            "after": after,
            "revision": revision,
            "actor": _actor(actor),
            "changed_at": changed_at,
        }
        undo_stack = [item for item in draft.get("undo_stack", []) if isinstance(item, dict)]
        undo_stack.append(entry)
        draft["undo_stack"] = undo_stack[-MAX_UNDO_DEPTH:]
        draft["edited_cue_ids"] = sorted({str(item.get("cue_id", "")) for item in draft["undo_stack"]})
        draft["dirty"] = bool(draft["undo_stack"])
        draft["revision"] = revision
        draft["updated_at"] = changed_at
        draft["updated_by"] = entry["actor"]
        write_json_atomic(review_draft_path(work_paths), draft)
        append_jsonl(
            review_history_path(work_paths),
            {
                "action": "edit",
                **entry,
                "draft_path": str(review_draft_path(work_paths)),
                "backup_path": backup_path,
            },
        )
        return {
            "changed": True,
            "revision": revision,
            "backup_path": backup_path,
            "audit_path": str(review_history_path(work_paths)),
            "review": build_review_view(work_paths),
        }


def undo_review_edit(
    work_paths: WorkPaths,
    *,
    expected_revision: Any = None,
    actor: str = "web-local-user",
) -> dict[str, Any]:
    with REVIEW_LOCK:
        draft, draft_error = _load_review_draft(work_paths)
        if draft_error:
            raise ReviewError("REVIEW_DRAFT_CORRUPT", draft_error, status=409)
        if draft is None:
            raise ReviewError("REVIEW_DRAFT_MISSING", "review draft does not exist", status=404)
        _check_revision(draft, expected_revision)
        undo_stack = [item for item in draft.get("undo_stack", []) if isinstance(item, dict)]
        if not undo_stack:
            raise ReviewError("REVIEW_NOTHING_TO_UNDO", "review draft has no edit to undo", status=409)
        edit = undo_stack.pop()
        cue_id = str(edit.get("cue_id", ""))
        cue = draft["cues"].get(cue_id)
        if not isinstance(cue, dict) or not isinstance(edit.get("before"), dict):
            raise ReviewError("REVIEW_HISTORY_INVALID", "review undo history is invalid", status=409)
        backup_path = _backup_review_draft(work_paths, int(draft.get("revision", 0)))
        after_undo = _cue_snapshot(cue)
        _apply_cue_snapshot(cue, edit["before"])
        revision = int(draft.get("revision", 0)) + 1
        changed_at = time.time()
        draft["undo_stack"] = undo_stack
        draft["edited_cue_ids"] = sorted({str(item.get("cue_id", "")) for item in undo_stack})
        draft["dirty"] = bool(undo_stack)
        draft["revision"] = revision
        draft["updated_at"] = changed_at
        draft["updated_by"] = _actor(actor)
        write_json_atomic(review_draft_path(work_paths), draft)
        append_jsonl(
            review_history_path(work_paths),
            {
                "action": "undo",
                "cue_id": cue_id,
                "before": after_undo,
                "after": edit["before"],
                "undone_edit_revision": edit.get("revision"),
                "revision": revision,
                "actor": draft["updated_by"],
                "changed_at": changed_at,
                "draft_path": str(review_draft_path(work_paths)),
                "backup_path": backup_path,
            },
        )
        return {
            "changed": True,
            "revision": revision,
            "backup_path": backup_path,
            "audit_path": str(review_history_path(work_paths)),
            "review": build_review_view(work_paths),
        }


def resolve_workspace_media(work_paths: WorkPaths, path_value: str) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        raise ReviewError("MEDIA_PATH_REQUIRED", "media path is required")
    target = Path(raw).resolve()
    allowed_roots = [work_paths.workdir.resolve(), *_linked_workspace_roots(work_paths)]
    if not any(_is_relative_to(target, root) for root in allowed_roots):
        raise ReviewError(
            "MEDIA_PATH_OUT_OF_SCOPE",
            "media must be inside the selected workspace or a manifest-linked workspace",
            status=403,
        )
    if not target.exists() or not target.is_file():
        raise ReviewError("MEDIA_NOT_FOUND", "media file does not exist", status=404)
    if target.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ReviewError("MEDIA_TYPE_NOT_ALLOWED", "file type is not allowed", status=403)
    return target


def _load_review_draft(work_paths: WorkPaths) -> tuple[dict[str, Any] | None, str | None]:
    path = review_draft_path(work_paths)
    if not path.exists():
        return None, None
    try:
        payload = read_json(path, default=None)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "review draft root must be an object"
    if payload.get("schema_version") != REVIEW_SCHEMA_VERSION:
        return None, "review draft schema version is unsupported"
    if not isinstance(payload.get("cues"), dict):
        return None, "review draft cues must be an object"
    return payload, None


def _new_review_draft(source_path: Path, source_payload: dict[str, Any]) -> dict[str, Any]:
    now = time.time()
    return {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "base_source": str(source_path.resolve()),
        "base_source_modified_at": source_path.stat().st_mtime,
        "created_at": now,
        "updated_at": now,
        "updated_by": None,
        "revision": 0,
        "dirty": False,
        "history_limit": MAX_UNDO_DEPTH,
        "edited_cue_ids": [],
        "undo_stack": [],
        "cues": deepcopy(source_payload),
    }


def _review_state(work_paths: WorkPaths, draft: dict[str, Any] | None) -> dict[str, Any]:
    if draft is None:
        return {
            "draft_exists": False,
            "dirty": False,
            "revision": 0,
            "can_undo": False,
            "undo_depth": 0,
            "history_limit": MAX_UNDO_DEPTH,
            "draft_path": str(review_draft_path(work_paths)),
            "audit_path": str(review_history_path(work_paths)),
        }
    undo_stack = [item for item in draft.get("undo_stack", []) if isinstance(item, dict)]
    return {
        "draft_exists": True,
        "dirty": bool(draft.get("dirty")),
        "revision": int(draft.get("revision", 0)),
        "can_undo": bool(undo_stack),
        "undo_depth": len(undo_stack),
        "history_limit": int(draft.get("history_limit", MAX_UNDO_DEPTH)),
        "draft_path": str(review_draft_path(work_paths)),
        "audit_path": str(review_history_path(work_paths)),
        "updated_at": draft.get("updated_at"),
        "updated_by": draft.get("updated_by"),
    }


def _check_revision(draft: dict[str, Any], expected_revision: Any) -> None:
    if expected_revision is None:
        return
    try:
        expected = int(expected_revision)
    except (TypeError, ValueError) as exc:
        raise ReviewError("REVIEW_REVISION_INVALID", "expected revision must be an integer") from exc
    current = int(draft.get("revision", 0))
    if expected != current:
        raise ReviewError(
            "REVIEW_REVISION_CONFLICT",
            f"review draft changed: expected revision {expected}, current revision {current}",
            status=409,
        )


def _validated_edit(
    cues: dict[str, Any],
    cue_id: str,
    original: str,
    translation: str,
    start_ms: Any,
    end_ms: Any,
) -> dict[str, Any]:
    original_text = str(original or "").strip()
    translation_text = str(translation or "").strip()
    if not original_text:
        raise ReviewError("REVIEW_ORIGINAL_REQUIRED", "original subtitle must not be empty")
    if len(original_text) > 10000 or len(translation_text) > 10000:
        raise ReviewError("REVIEW_TEXT_TOO_LONG", "review text exceeds 10000 characters")
    start = _required_int(start_ms, "start_ms")
    end = _required_int(end_ms, "end_ms")
    if start < 0 or end <= start:
        raise ReviewError("REVIEW_TIME_INVALID", "cue time must satisfy 0 <= start_ms < end_ms")
    for other_id, other in cues.items():
        if str(other_id) == cue_id or not isinstance(other, dict):
            continue
        other_start = _int_or_none(other.get("start_time"))
        other_end = _int_or_none(other.get("end_time"))
        if other_start is None or other_end is None:
            continue
        if start < other_end and other_start < end:
            raise ReviewError(
                "REVIEW_TIME_OVERLAP",
                f"cue time overlaps cue {other_id}",
                status=409,
            )
    return {
        "start_ms": start,
        "end_ms": end,
        "original": original_text,
        "translation": translation_text,
    }


def _required_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ReviewError("REVIEW_TIME_INVALID", f"{field} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ReviewError("REVIEW_TIME_INVALID", f"{field} must be an integer") from exc


def _cue_snapshot(cue: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_ms": _int_or_none(cue.get("start_time")),
        "end_ms": _int_or_none(cue.get("end_time")),
        "original": str(cue.get("original_subtitle", cue.get("text", ""))),
        "translation": str(cue.get("translated_subtitle", "")),
    }


def _apply_cue_snapshot(cue: dict[str, Any], snapshot: dict[str, Any]) -> None:
    cue["start_time"] = int(snapshot["start_ms"])
    cue["end_time"] = int(snapshot["end_ms"])
    cue["original_subtitle"] = str(snapshot.get("original", ""))
    cue["translated_subtitle"] = str(snapshot.get("translation", ""))


def _backup_review_draft(work_paths: WorkPaths, revision: int) -> str | None:
    source = review_draft_path(work_paths)
    if not source.exists():
        return None
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = work_paths.workdir / "reports" / "review-backups" / f"revision-{revision}-{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source, backup_dir / source.name)
    (backup_dir / "BACKED_UP_FILES.txt").write_text(f"{source}\n", encoding="utf-8")
    return str(backup_dir)


def _actor(value: str) -> str:
    actor = str(value or "web-local-user").strip()
    return actor[:128] or "web-local-user"


def _select_cue_source(work_paths: WorkPaths) -> tuple[Path | None, dict[str, Any], str | None]:
    candidates = (
        work_paths.normalized_manifest,
        work_paths.mimo_proofread_manifest,
        work_paths.translated_manifest,
        work_paths.split_manifest,
    )
    errors = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = read_json(path, default={})
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(payload, dict) and payload:
            return path, payload, None
    return None, {}, "; ".join(errors) if errors else None


def _safe_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = read_json(path, default={})
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_list(path: Path) -> list[dict[str, Any]]:
    try:
        payload = read_json(path, default=[])
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _review_audio_path(work_paths: WorkPaths, aligned: list[dict[str, Any]]) -> Path | None:
    if work_paths.audio_path.exists():
        return work_paths.audio_path
    for item in aligned:
        segment_path = Path(str(item.get("audio_path", "")))
        if segment_path.parent.name == "segments" and segment_path.parent.parent.name == "audio":
            source = segment_path.parent.parent / "source.wav"
            if source.exists():
                return source.resolve()
    return None


def _linked_workspace_roots(work_paths: WorkPaths) -> list[Path]:
    roots: set[Path] = set()
    workspaces_root = work_paths.workdir.parent.resolve()
    for manifest in (work_paths.aligned_manifest, work_paths.transcript_manifest, work_paths.segments_manifest):
        for item in _safe_list(manifest):
            raw = str(item.get("audio_path", "")).strip()
            if not raw:
                continue
            path = Path(raw).resolve()
            for parent in path.parents:
                if parent.parent == workspaces_root:
                    roots.add(parent)
                    break
    return sorted(roots)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _load_references(work_paths: WorkPaths) -> list[dict[str, Any]]:
    root = work_paths.workdir / "references"
    if not root.is_dir():
        return []
    results = []
    for path in sorted(root.glob("*.ass")):
        try:
            dialogues = parse_ass_dialogues(path)
        except (OSError, UnicodeError, ValueError):
            continue
        results.append({"name": path.name, "path": str(path), "dialogues": dialogues})
    return results


def _best_aligned_segment(
    aligned: list[dict[str, Any]], start_ms: int, end_ms: int
) -> dict[str, Any] | None:
    best = None
    best_overlap = 0
    for item in aligned:
        segment_start = _seconds_to_ms(item.get("global_start_time"))
        segment_end = _seconds_to_ms(item.get("global_end_time"))
        if segment_start is None or segment_end is None:
            continue
        overlap = max(0, min(end_ms, segment_end) - max(start_ms, segment_start))
        if overlap > best_overlap:
            best = item
            best_overlap = overlap
    return best


def _reference_matches(
    references: list[dict[str, Any]], start_ms: int, end_ms: int
) -> list[dict[str, Any]]:
    matches = []
    midpoint = (start_ms + end_ms) // 2
    for source in references:
        best = None
        best_score: tuple[int, int] | None = None
        for dialogue in source["dialogues"]:
            overlap = max(0, min(end_ms, dialogue.end_ms) - max(start_ms, dialogue.start_ms))
            distance = abs(((dialogue.start_ms + dialogue.end_ms) // 2) - midpoint)
            score = (overlap, -distance)
            if best_score is None or score > best_score:
                best = dialogue
                best_score = score
        if best is not None and (best_score[0] > 0 or -best_score[1] <= 3000):
            matches.append(
                {
                    "source": source["name"],
                    "start_ms": best.start_ms,
                    "end_ms": best.end_ms,
                    "style": best.style,
                    "text": best.text,
                    "time_overlap_ms": best_score[0],
                }
            )
    return matches


def _review_flags(
    cue: dict[str, Any], translated: dict[str, Any], alignment_state: str
) -> list[str]:
    flags = []
    if alignment_state in {"failed", "completed_coarse"}:
        flags.append(alignment_state)
    if alignment_state == MUSIC_REGION_STATE:
        flags.append("music_region")
    if bool(translated.get("needs_realign")):
        flags.append("needs_realign")
    realign_status = str(translated.get("realign_status", ""))
    if realign_status and realign_status != "completed":
        flags.append(f"realign_{realign_status}")
    if not str(cue.get("translated_subtitle", "")).strip():
        flags.append("translation_missing")
    return flags


def _seconds_to_ms(value: Any) -> int | None:
    try:
        return round(float(value) * 1000)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
