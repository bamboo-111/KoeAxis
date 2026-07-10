from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.stages import StageStatus
from qwen_asr.storage import read_json, write_json_atomic


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_progress(
    work_paths: WorkPaths,
    *,
    stage: str,
    status: str | StageStatus,
    done: int | None = None,
    total: int | None = None,
    current: str = "",
    summary: str = "",
) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "status": str(status),
        "done": done,
        "total": total,
        "current": current,
        "updated_at": utc_now_iso(),
        "summary": summary,
    }
    write_json_atomic(work_paths.progress_path, payload)
    return payload


def read_progress(work_paths: WorkPaths) -> dict[str, Any] | None:
    payload = read_json(work_paths.progress_path, default=None)
    return payload if isinstance(payload, dict) else None


def progress_path_for(workdir: Path) -> Path:
    return WorkPaths.from_workdir(workdir).progress_path
