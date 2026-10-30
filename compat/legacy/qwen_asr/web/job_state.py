from __future__ import annotations

import ctypes
import os
import time
from pathlib import Path
from typing import Any

from qwen_asr.storage import read_json, write_json_atomic
from qwen_asr.web.commands import WORKSPACES_DIR

GLOBAL_JOB_STATE_PATH = WORKSPACES_DIR / ".web-state" / "job.json"
WORKSPACE_JOB_STATE_RELATIVE_PATH = Path("reports") / "web_job.json"
SECRET_FLAGS = {
    "--api-key",
    "--llm-api-key",
    "--deepseek-api-key",
    "--mimo-api-key",
}


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in job.items() if not key.startswith("_")}
    command = result.get("command")
    if isinstance(command, list):
        result["command"] = redact_command(command)
    return result


def persist_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = public_job(job)
    payload["state_updated_at"] = time.time()
    write_json_atomic(GLOBAL_JOB_STATE_PATH, payload)
    workspace_path = _workspace_job_path(payload.get("workdir"))
    if workspace_path is not None:
        write_json_atomic(workspace_path, payload)
    return payload


def load_job(*, reconcile: bool = True) -> dict[str, Any] | None:
    try:
        payload = read_json(GLOBAL_JOB_STATE_PATH, default=None)
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if reconcile and payload.get("status") in {"running", "stopping"}:
        pid = _as_int(payload.get("pid"))
        if pid is None or not pid_is_running(pid):
            payload["status"] = "interrupted"
            payload["finished_at"] = time.time()
            payload["returncode"] = None
            payload["message"] = "Saved Web job process is no longer running."
            persist_job(payload)
    return payload


def redact_command(command: list[Any]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for raw in command:
        value = str(raw)
        if redact_next:
            redacted.append("***")
            redact_next = False
            continue
        redacted.append(value)
        if value.lower() in SECRET_FLAGS:
            redact_next = True
    return redacted


def pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _workspace_job_path(workdir_value: Any) -> Path | None:
    raw = str(workdir_value or "").strip()
    if not raw:
        return None
    workdir = Path(raw).resolve()
    root = WORKSPACES_DIR.resolve()
    if workdir.parent != root or not workdir.is_dir():
        return None
    return workdir / WORKSPACE_JOB_STATE_RELATIVE_PATH


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
