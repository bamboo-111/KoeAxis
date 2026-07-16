from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import shutil
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from qwen_asr.glossary import write_normalized_glossary_xlsx
from qwen_asr.web.commands import (
    HOST,
    PORT,
    ROOT,
    build_command,
    list_workspaces,
    resolve_deletable_workspace,
    resolve_deletable_workspaces,
    suggest_workdir,
)
from qwen_asr.web.static_html import INDEX_HTML, WORKBENCH_HTML, load_static_asset
from qwen_asr.web.status import build_progress, get_status
from qwen_asr.web.job_state import load_job, persist_job, public_job
from qwen_asr.web.workspace_api import (
    WorkspaceApiError,
    apply_recovery_action,
    apply_review_edit,
    apply_review_undo,
    api_contract,
    envelope,
    get_align_state,
    get_exports,
    get_export_file_path,
    get_quality_gate,
    get_quality_evidence_path,
    get_recovery_queue,
    get_review_view,
    get_stage_view,
    get_workspace_detail,
    get_workspace_media_path,
    list_workspace_summaries,
    prepare_workspace_stage_start,
)

JOB_LOCK = threading.Lock()
ACTIVE_JOB: dict | None = None

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(WORKBENCH_HTML)
            return
        if parsed.path == "/legacy":
            self._send_html(INDEX_HTML)
            return
        if parsed.path.startswith("/static/"):
            try:
                data, content_type = load_static_asset(parsed.path.removeprefix("/static/"))
            except FileNotFoundError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(data, content_type)
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if parsed.path == "/api/status":
            query = parse_qs(parsed.query)
            workdir = query.get("workdir", [""])[0]
            self._send_json(get_status(workdir))
            return
        if parsed.path == "/api/job":
            self._send_json(get_active_job())
            return
        if parsed.path == "/api/suggest-workdir":
            query = parse_qs(parsed.query)
            media = query.get("media", [""])[0]
            self._send_json({"workdir": str(suggest_workdir(media).resolve())})
            return
        if parsed.path == "/api/workspaces":
            self._send_json({"workspaces": list_workspaces()})
            return
        if parsed.path == "/api/v1/workspace/media":
            query = parse_qs(parsed.query)
            try:
                path = get_workspace_media_path(
                    query.get("workdir", [""])[0],
                    query.get("path", [""])[0],
                )
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            self._send_file(path)
            return
        if parsed.path == "/api/v1/workspace/export-file":
            query = parse_qs(parsed.query)
            try:
                path = get_export_file_path(
                    query.get("workdir", [""])[0],
                    query.get("path", [""])[0],
                )
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            download = query.get("download", ["0"])[0] == "1"
            self._send_file(
                path,
                download=download,
                content_type=None if download else _subtitle_preview_content_type(path),
            )
            return
        if parsed.path == "/api/v1/workspace/quality-evidence":
            query = parse_qs(parsed.query)
            try:
                path = get_quality_evidence_path(
                    query.get("workdir", [""])[0],
                    query.get("path", [""])[0],
                )
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            self._send_file(path)
            return
        if parsed.path.startswith("/api/v1/"):
            self._handle_v1_get(parsed)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _handle_v1_get(self, parsed) -> None:
        query = parse_qs(parsed.query)
        workdir = query.get("workdir", [""])[0]
        routes = {
            "/api/v1/contract": lambda: api_contract(),
            "/api/v1/job": lambda: get_structured_job(),
            "/api/v1/workspaces": lambda: list_workspace_summaries(),
            "/api/v1/workspace": lambda: get_workspace_detail(workdir),
            "/api/v1/workspace/stages": lambda: get_stage_view(workdir),
            "/api/v1/workspace/align": lambda: get_align_state(workdir),
            "/api/v1/workspace/recovery": lambda: get_recovery_queue(workdir),
            "/api/v1/workspace/review": lambda: get_review_view(workdir),
            "/api/v1/workspace/quality": lambda: get_quality_gate(workdir),
            "/api/v1/workspace/exports": lambda: get_exports(workdir),
        }
        handler = routes.get(parsed.path)
        if handler is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            self._send_json(handler())
        except WorkspaceApiError as exc:
            self._send_json(exc.as_payload(), status=exc.status)

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/v1/workspace/stage/start":
            payload = self._read_json()
            try:
                command_payload = prepare_workspace_stage_start(
                    str(payload.get("workdir", "")),
                    stage=str(payload.get("stage", "")),
                    settings=payload.get("settings"),
                )
                job = start_job(command_payload)
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            except RuntimeError as exc:
                self._send_json(envelope(error={"code": "JOB_CONFLICT", "message": str(exc)}), status=409)
                return
            self._send_json(envelope(data=job), status=202)
            return
        if parsed.path == "/api/v1/workspace/review/edit":
            payload = self._read_json()
            try:
                result = apply_review_edit(
                    str(payload.get("workdir", "")),
                    cue_id=str(payload.get("cue_id", "")),
                    original=str(payload.get("original", "")),
                    translation=str(payload.get("translation", "")),
                    start_ms=payload.get("start_ms"),
                    end_ms=payload.get("end_ms"),
                    expected_revision=payload.get("expected_revision"),
                    actor=str(payload.get("actor", "web-local-user")),
                )
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/v1/workspace/review/undo":
            payload = self._read_json()
            try:
                result = apply_review_undo(
                    str(payload.get("workdir", "")),
                    expected_revision=payload.get("expected_revision"),
                    actor=str(payload.get("actor", "web-local-user")),
                )
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/v1/workspace/recovery/action":
            payload = self._read_json()
            try:
                result = apply_recovery_action(
                    str(payload.get("workdir", "")),
                    segment_id=str(payload.get("segment_id", "")),
                    action=str(payload.get("action", "")),
                    payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                    actor=str(payload.get("actor", "web-local-user")),
                )
            except WorkspaceApiError as exc:
                self._send_json(exc.as_payload(), status=exc.status)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/start":
            payload = self._read_json()
            try:
                job = start_job(payload)
            except (KeyError, TypeError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=409)
                return
            self._send_json(job, status=202)
            return
        if parsed.path == "/api/stop":
            stopped = stop_job()
            self._send_json(stopped, status=200 if stopped.get("status") != "idle" else 409)
            return
        if parsed.path == "/api/pick-media":
            try:
                result = pick_media_file()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/pick-media-list":
            try:
                result = pick_media_files()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/pick-batch-manifest":
            try:
                result = pick_batch_manifest()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/pick-glossary-xlsx":
            try:
                result = pick_glossary_xlsx()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/pick-output-directory":
            try:
                result = pick_output_directory()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/pick-model-cache-dir":
            try:
                result = pick_model_cache_dir()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/pick-export-file":
            try:
                result = pick_export_file()
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/delete-workspace":
            payload = self._read_json()
            try:
                result = delete_workspace(str(payload.get("workdir", "")))
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/delete-workspaces":
            payload = self._read_json()
            try:
                result = delete_workspaces(payload.get("workdirs"))
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        if parsed.path == "/api/glossary-normalize":
            payload = self._read_json()
            try:
                result = normalize_glossary_xlsx(str(payload.get("xlsx", "")))
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            self._send_json(result)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args):  # noqa: A003
        return

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, payload: str) -> None:
        data = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(
        self,
        path: Path,
        *,
        download: bool = False,
        content_type: str | None = None,
    ) -> None:
        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        range_header = self.headers.get("Range", "")
        partial = False
        if range_header.startswith("bytes="):
            try:
                start_raw, end_raw = range_header.removeprefix("bytes=").split("-", 1)
                start = int(start_raw) if start_raw else 0
                end = int(end_raw) if end_raw else end
                if start < 0 or end < start or start >= size:
                    raise ValueError
                end = min(end, size - 1)
                partial = True
            except ValueError:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", content_type or _content_type_for_path(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(path.name)}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            self.wfile.write(handle.read(length))


def _content_type_for_path(path: Path) -> str:
    if path.suffix.lower() == ".srt":
        return "application/x-subrip; charset=utf-8"
    if path.suffix.lower() == ".vtt":
        return "text/vtt; charset=utf-8"
    guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"{guessed}; charset=utf-8" if guessed.startswith("text/") else guessed


def _subtitle_preview_content_type(path: Path) -> str:
    if path.suffix.lower() in {".srt", ".vtt"}:
        return "text/plain; charset=utf-8"
    return _content_type_for_path(path)

def start_job(payload: dict) -> dict:
    global ACTIVE_JOB
    with JOB_LOCK:
        if ACTIVE_JOB and ACTIVE_JOB["status"] == "running":
            raise RuntimeError("Another job is already running.")
        command = build_command(payload)
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ACTIVE_JOB = {
            "id": str(int(time.time() * 1000)),
            "stage": payload["stage"],
            "workdir": payload["workdir"],
            "command": command,
            "pid": process.pid,
            "_process": process,
            "status": "running",
            "started_at": time.time(),
            "returncode": None,
        }
        thread = threading.Thread(target=_wait_job, args=(process,), daemon=True)
        thread.start()
        return persist_job(ACTIVE_JOB)

def _wait_job(process: subprocess.Popen) -> None:
    global ACTIVE_JOB
    returncode = process.wait()
    with JOB_LOCK:
        if ACTIVE_JOB and ACTIVE_JOB["pid"] == process.pid:
            if ACTIVE_JOB.get("status") == "stopping":
                ACTIVE_JOB["status"] = "stopped"
            else:
                ACTIVE_JOB["status"] = "completed" if returncode == 0 else "failed"
            ACTIVE_JOB["returncode"] = returncode
            ACTIVE_JOB["finished_at"] = time.time()
            ACTIVE_JOB.pop("_process", None)
            persist_job(ACTIVE_JOB)

def stop_job() -> dict:
    global ACTIVE_JOB
    with JOB_LOCK:
        if not ACTIVE_JOB or ACTIVE_JOB.get("status") != "running":
            return {"status": "idle", "message": "No running job."}
        pid = ACTIVE_JOB["pid"]
        ACTIVE_JOB["status"] = "stopping"
        proc = ACTIVE_JOB.get("_process")
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif proc is not None:
            proc.terminate()
    finally:
        with JOB_LOCK:
            if ACTIVE_JOB and ACTIVE_JOB["pid"] == pid:
                ACTIVE_JOB["status"] = "stopped"
                ACTIVE_JOB["returncode"] = -1
                ACTIVE_JOB["finished_at"] = time.time()
                ACTIVE_JOB.pop("_process", None)
                job = public_job(ACTIVE_JOB)
                job["progress"] = build_progress(job)
                persist_job(job)
            else:
                job = None
    if job is not None:
        return job
    return {"status": "idle"}

def get_active_job() -> dict:
    with JOB_LOCK:
        if not ACTIVE_JOB:
            saved = load_job()
            if saved is None:
                return {"status": "idle"}
            saved["progress"] = build_progress(saved)
            return saved
        job = public_job(ACTIVE_JOB)
        job["progress"] = build_progress(job)
        return job


def get_structured_job() -> dict:
    from qwen_asr.web.workspace_api import envelope

    return envelope(data=get_active_job())


def pick_media_file() -> dict:
    selected = _run_tk_file_picker(
        title="Select media file",
        filetypes="[('Media files', '*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mkv *.mov *.webm *.avi'), ('All files', '*.*')]",
    )
    if not selected:
        return {"cancelled": True}
    path = Path(selected).resolve()
    return {"cancelled": False, "path": str(path)}


def pick_media_files() -> dict:
    selected_output = _run_tk_file_picker(
        title="Select media files",
        filetypes="[('Media files', '*.mp3 *.wav *.m4a *.aac *.flac *.ogg *.mp4 *.mkv *.mov *.webm *.avi'), ('All files', '*.*')]",
        multiple=True,
    )
    selected = [line.strip() for line in selected_output.splitlines() if line.strip()]
    if not selected:
        return {"cancelled": True, "paths": []}
    return {"cancelled": False, "paths": [str(Path(item).resolve()) for item in selected]}


def pick_batch_manifest() -> dict:
    selected = _run_tk_file_picker(
        title="Select batch manifest",
        filetypes="[('Batch manifests', '*.json *.jsonl'), ('All files', '*.*')]",
    )
    if not selected:
        return {"cancelled": True}
    return {"cancelled": False, "path": str(Path(selected).resolve())}


def pick_glossary_xlsx() -> dict:
    selected = _run_tk_file_picker(
        title="Select glossary xlsx",
        filetypes="[('Excel files', '*.xlsx *.xls'), ('All files', '*.*')]",
    )
    if not selected:
        return {"cancelled": True}
    return {"cancelled": False, "path": str(Path(selected).resolve())}


def pick_output_directory() -> dict:
    selected = _run_tk_directory_picker(title="Select output directory")
    if not selected:
        return {"cancelled": True}
    return {"cancelled": False, "path": str(Path(selected).resolve())}


def pick_model_cache_dir() -> dict:
    selected = _run_tk_directory_picker(title="Select model cache directory")
    if not selected:
        return {"cancelled": True}
    return {"cancelled": False, "path": str(Path(selected).resolve())}


def pick_export_file() -> dict:
    selected = _run_tk_save_file_picker(
        title="Select export file",
        filetypes="[('Subtitle files', '*.srt *.vtt'), ('All files', '*.*')]",
        defaultextension=".srt",
    )
    if not selected:
        return {"cancelled": True}
    return {"cancelled": False, "path": str(Path(selected).resolve())}


def _run_tk_file_picker(*, title: str, filetypes: str, multiple: bool = False) -> str:
    function_name = "askopenfilenames" if multiple else "askopenfilename"
    result_expr = "'\\n'.join(selected)" if multiple else "selected or ''"
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "root.lift()\n"
        "root.focus_force()\n"
        "root.update()\n"
        "try:\n"
        f"    selected = filedialog.{function_name}(\n"
        f"        title={title!r},\n"
        "        parent=root,\n"
        f"        filetypes={filetypes},\n"
        "    )\n"
        f"    print({result_expr})\n"
        "finally:\n"
        "    root.destroy()\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("File picker timed out.") from exc
    except Exception as exc:  # pragma: no cover - depends on local desktop GUI availability
        raise RuntimeError(f"File picker is unavailable: {exc}") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "File picker failed.").strip()
        raise RuntimeError(message)
    return result.stdout.strip()


def _run_tk_directory_picker(*, title: str) -> str:
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "root.lift()\n"
        "root.focus_force()\n"
        "root.update()\n"
        "try:\n"
        "    selected = filedialog.askdirectory(\n"
        f"        title={title!r},\n"
        "        parent=root,\n"
        "    )\n"
        "    print(selected or '')\n"
        "finally:\n"
        "    root.destroy()\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("File picker timed out.") from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Directory picker is unavailable: {exc}") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "Directory picker failed.").strip()
        raise RuntimeError(message)
    return result.stdout.strip()


def _run_tk_save_file_picker(*, title: str, filetypes: str, defaultextension: str) -> str:
    script = (
        "import tkinter as tk\n"
        "from tkinter import filedialog\n"
        "root = tk.Tk()\n"
        "root.withdraw()\n"
        "root.attributes('-topmost', True)\n"
        "root.lift()\n"
        "root.focus_force()\n"
        "root.update()\n"
        "try:\n"
        "    selected = filedialog.asksaveasfilename(\n"
        f"        title={title!r},\n"
        "        parent=root,\n"
        f"        filetypes={filetypes},\n"
        f"        defaultextension={defaultextension!r},\n"
        "    )\n"
        "    print(selected or '')\n"
        "finally:\n"
        "    root.destroy()\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("File picker timed out.") from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"File picker is unavailable: {exc}") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "File picker failed.").strip()
        raise RuntimeError(message)
    return result.stdout.strip()


def delete_workspace(workdir_value: str) -> dict:
    try:
        target = resolve_deletable_workspace(workdir_value)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    if not target.exists():
        return {"deleted": False, "path": str(target), "message": "Workspace does not exist."}
    if not target.is_dir():
        raise RuntimeError("Workspace target is not a directory.")
    shutil.rmtree(target)
    return {"deleted": True, "path": str(target)}


def delete_workspaces(workdir_values: object = None) -> dict:
    if workdir_values is not None and not isinstance(workdir_values, list):
        raise RuntimeError("workdirs must be a list when provided.")
    try:
        targets = resolve_deletable_workspaces(workdir_values)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    deleted: list[str] = []
    missing: list[str] = []
    for target in targets:
        if not target.exists():
            missing.append(str(target))
            continue
        if not target.is_dir():
            raise RuntimeError(f"Workspace target is not a directory: {target}")
        shutil.rmtree(target)
        deleted.append(str(target))
    return {"deleted": deleted, "missing": missing, "count": len(deleted)}


def normalize_glossary_xlsx(xlsx_value: str) -> dict:
    xlsx = str(xlsx_value or "").strip()
    if not xlsx:
        raise RuntimeError("Glossary xlsx path is required.")
    try:
        result = write_normalized_glossary_xlsx(Path(xlsx))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise RuntimeError(str(exc)) from exc
    return {"output": str(result.output_path), "count": result.entry_count}


def main() -> int:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"http://{HOST}:{PORT}")
    server.serve_forever()
    return 0
