from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from qwen_asr.artifact_state import ArtifactState
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress
from qwen_asr.stages import STAGE_DEFINITIONS
from qwen_asr.storage import read_json
from qwen_asr.web.job_state import load_job
from qwen_asr.web.stage_start_service import stage_start_capability


def build_stage_view(work_paths: WorkPaths) -> dict[str, Any]:
    artifact_state = ArtifactState(work_paths)
    progress, progress_error = _safe_progress(work_paths)
    job = load_job() or {}
    current_stage = str(progress.get("stage") or job.get("stage") or "")
    stages = []
    for index, (name, definition) in enumerate(STAGE_DEFINITIONS.items()):
        state_error = None
        try:
            complete = artifact_state.is_complete(name)
            missing_inputs = artifact_state.missing_inputs(name)
            outdated = artifact_state.is_outdated(name)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            complete = False
            missing_inputs = []
            outdated = False
            state_error = f"{type(exc).__name__}: {exc}"
        artifacts = _artifact_rows(work_paths, definition.output_attrs)
        start_capability = stage_start_capability(name, missing_inputs)
        has_outputs = any(item["exists"] for item in artifacts)
        stage_status = _stage_status(
            name=name,
            current_stage=current_stage,
            job=job,
            complete=complete,
            outdated=outdated,
            missing_inputs=missing_inputs,
            has_outputs=has_outputs,
            state_error=state_error,
        )
        log_path = work_paths.logs_dir / f"{name}.log"
        stages.append(
            {
                "index": index,
                "name": name,
                "status": stage_status,
                "complete": complete,
                "outdated": outdated,
                "missing_inputs": missing_inputs,
                "state_error": state_error,
                "input_count": _attribute_count(work_paths, definition.input_attrs, definition.any_input_groups),
                "output_count": _attribute_count(work_paths, definition.output_attrs, ()),
                "duration_seconds": _job_duration(name, job),
                "log": {
                    "path": str(log_path),
                    "exists": log_path.exists(),
                    "size_bytes": log_path.stat().st_size if log_path.exists() else 0,
                },
                "artifacts": artifacts,
                "runnable": start_capability["runnable"],
                "start_block_reason": start_capability["reason"],
            }
        )
    return {
        "current_stage": current_stage or None,
        "progress": progress,
        "progress_error": progress_error,
        "job": {key: value for key, value in job.items() if key != "command"},
        "stages": stages,
    }


def _stage_status(
    *,
    name: str,
    current_stage: str,
    job: dict[str, Any],
    complete: bool,
    outdated: bool,
    missing_inputs: list[str],
    has_outputs: bool,
    state_error: str | None,
) -> str:
    if state_error:
        return "error"
    if name == current_stage and job.get("status") in {"running", "stopping", "failed", "interrupted"}:
        return str(job["status"])
    if outdated:
        return "outdated"
    if complete:
        return "complete"
    if has_outputs:
        return "failed"
    if missing_inputs:
        return "blocked"
    return "pending"


def _artifact_rows(work_paths: WorkPaths, attrs: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for attr in attrs:
        path = getattr(work_paths, attr)
        rows.append(
            {
                "kind": attr,
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
                "modified_at": path.stat().st_mtime if path.exists() else None,
            }
        )
    return rows


def _attribute_count(
    work_paths: WorkPaths,
    attrs: tuple[str, ...],
    groups: tuple[tuple[str, ...], ...],
) -> int | None:
    paths = [getattr(work_paths, attr) for attr in attrs]
    for group in groups:
        paths.extend(getattr(work_paths, attr) for attr in group if getattr(work_paths, attr).exists())
    counts = [_path_count(path) for path in paths if path.exists()]
    return max(counts) if counts else None


def _path_count(path: Path) -> int:
    if not path.is_file():
        return 0
    if path.suffix.lower() == ".json":
        try:
            payload = read_json(path, default=None)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return 0
        if isinstance(payload, (list, dict)):
            return len(payload)
    return 1


def _job_duration(stage: str, job: dict[str, Any]) -> float | None:
    if str(job.get("stage", "")) != stage:
        return None
    try:
        start = float(job["started_at"])
    except (KeyError, TypeError, ValueError):
        return None
    end_value = job.get("finished_at") if job.get("status") not in {"running", "stopping"} else time.time()
    try:
        end = float(end_value)
    except (TypeError, ValueError):
        return None
    return round(max(0.0, end - start), 3)


def _safe_progress(work_paths: WorkPaths) -> tuple[dict[str, Any], str | None]:
    try:
        payload = read_progress(work_paths) or {}
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    return payload if isinstance(payload, dict) else {}, None
