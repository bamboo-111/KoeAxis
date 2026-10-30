"""CI worker: pulls runs from the control server over HTTPS and executes them locally.

    python -m qwen_asr.ci worker --server https://ci.example.com --token kxw_... [--id gpu-01]

Scheduling on one worker: a run is GPU phase (prepare→align + alignment recovery) followed by API phase
(split→export). The worker only claims a new run while no own run is in its GPU phase, so GPU work is
serial while the API phase of run N overlaps the GPU phase of run N+1 (bounded by ``max_runs``). A
file lock (``<data>/gpu.lock``) also serializes GPU use across several worker processes on one host.

Workspace layout (``<data>/runs/<workspace_key>/``): ``work/`` pipeline workdir (reused by retries with
the same workspace key), ``media/`` source file, ``inputs/<run>/`` preset/user files, ``ci/<run>/`` spec
and export. Everything here is replaceable: the phase command (``--phase-command``) and the per-worker
CLI defaults (``--default-run-args``) are plain configuration.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from qwen_asr.ci.phase import EVENT_PREFIX
from qwen_asr.ci.pipeline import API_PHASE, GPU_PHASE, STAGES, build_run_argv, safe_filename

logger = logging.getLogger("koeaxis.worker")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class WorkerConfig:
    server: str
    token: str
    worker_id: str = field(default_factory=lambda: safe_filename(socket.gethostname(), "worker")[:64])
    name: str = ""
    data_dir: Path = Path("ci-worker")
    max_runs: int = 2                      # concurrent runs (only one of them in its GPU phase)
    poll_interval_s: float = 10.0
    heartbeat_interval_s: float = 15.0
    log_flush_interval_s: float = 2.0
    default_run_args: list[str] = field(default_factory=list)   # e.g. --device cuda:0 --dtype fp16
    env: dict[str, str] = field(default_factory=dict)
    preset_ids: list[str] | None = None
    phase_command: list[str] = field(default_factory=lambda: [sys.executable, "-B", "-m", "qwen_asr.ci", "phase"])
    model_cache_dir: Path | None = None    # resolve "org/name" model ids to local HF snapshots
    project_root: Path = PROJECT_ROOT
    upload_manifests: bool = True
    request_timeout_s: float = 120.0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.name = self.name or self.worker_id


def resolve_model_reference(value: str, cache_dir: Path | None) -> str:
    """Map a Hugging Face id ("org/name") to the newest local snapshot directory when one exists."""
    if not value or not cache_dir or Path(value).exists() or value.count("/") != 1:
        return value
    snapshots = Path(cache_dir) / ("models--" + value.replace("/", "--")) / "snapshots"
    if not snapshots.is_dir():
        return value
    candidates = sorted((path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return str(candidates[-1]) if candidates else value


def parse_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def kill_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------- run state
class RunState:
    """Stage timeline, metrics and checks of one run, merged from phase events and report files."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stages: dict[str, dict[str, Any]] = {
            name: {"name": name, "phase": phase, "status": "pending"} for name, phase in STAGES
        }
        self.metrics: dict[str, Any] = {}
        self.checks: list[dict[str, Any]] = []
        self.summary = ""
        self.phase = ""
        self.dirty = True

    def on_event(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.dirty = True
            kind = event.get("event")
            if kind == "stage" and event.get("stage") in self.stages:
                stage = self.stages[event["stage"]]
                status = str(event.get("status") or "")
                stage["status"] = status
                if status == "running":
                    stage["started_at"] = event.get("at")
                    stage.pop("error", None)
                else:
                    stage["finished_at"] = event.get("at")
                for key in ("elapsed_s", "error", "return_code"):
                    if event.get(key) is not None:
                        stage[key] = event[key]
                if isinstance(event.get("summary"), dict):
                    stage["detail"] = event["summary"]
                self.summary = f"{event['stage']} {status}"
            elif kind == "phase":
                self.phase = f"{event.get('phase')}:{event.get('status')}"
            elif kind == "reset":
                self.summary = f"reset from {event.get('stage')}"
            elif kind == "gpu_lock":
                self.summary = "GPU lock acquired"

    def merge_performance(self, report: dict[str, Any]) -> None:
        with self.lock:
            self.dirty = True
            for item in report.get("stages") or []:
                stage = self.stages.get(str(item.get("stage")))
                if not stage:
                    continue
                stage["elapsed_s"] = round(float(item.get("wall_time_s") or 0), 1)
                if item.get("summary"):
                    stage["summary"] = str(item["summary"])[:500]
                if stage["status"] in {"pending", "running"} and item.get("status"):
                    stage["status"] = str(item["status"])
                network = item.get("network_summary") or {}
                tokens = network.get("token_totals") or {}
                if network.get("request_count"):
                    stage["requests"] = network.get("request_count")
                    stage["request_failures"] = network.get("failure_count")
                    stage["tokens"] = {key: tokens.get(key) for key in ("input_tokens", "cached_tokens", "output_tokens")}
            self._recompute_metrics()

    def _recompute_metrics(self) -> None:
        totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0}
        requests = failures = 0
        phase_time = {GPU_PHASE: 0.0, API_PHASE: 0.0}
        for stage in self.stages.values():
            for key in totals:
                totals[key] += int((stage.get("tokens") or {}).get(key) or 0)
            requests += int(stage.get("requests") or 0)
            failures += int(stage.get("request_failures") or 0)
            phase_time[stage["phase"]] += float(stage.get("elapsed_s") or 0)
        self.metrics.update({
            "gpu_phase_s": round(phase_time[GPU_PHASE], 1),
            "api_phase_s": round(phase_time[API_PHASE], 1),
            "llm_requests": requests,
            "llm_request_failures": failures,
            **totals,
            "cache_hit_rate": round(totals["cached_tokens"] / totals["input_tokens"], 3) if totals["input_tokens"] else None,
        })

    def set_checks(self, report: dict[str, Any]) -> None:
        with self.lock:
            self.dirty = True
            checks = [{"name": "final_quality", "status": report.get("status"),
                       "message": "聚合质量门"}]
            for item in report.get("checks") or []:
                if isinstance(item, dict):
                    checks.append({"name": item.get("name"), "status": item.get("status"),
                                   "message": str(item.get("message") or "")[:500]})
            self.checks = checks
            self.metrics["quality_status"] = report.get("status")

    def finalize(self) -> None:
        with self.lock:
            for stage in self.stages.values():
                if stage["status"] == "pending":
                    stage["status"] = "skipped"
                elif stage["status"] == "running":
                    stage["status"] = "cancelled"
            self.dirty = True

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            self.dirty = False
            return {"stages": [dict(stage) for stage in self.stages.values()], "metrics": dict(self.metrics),
                    "checks": list(self.checks), "summary": self.summary[:500]}


# ------------------------------------------------------------------------------------------- client
class ServerClient:
    def __init__(self, config: WorkerConfig, client: httpx.Client | None = None) -> None:
        self.config = config
        self.http = client or httpx.Client(base_url=config.server.rstrip("/"), timeout=config.request_timeout_s)
        self.headers = {"Authorization": f"Bearer {config.token}", "X-KoeAxis-Worker": config.worker_id}

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        delay = 2.0
        for attempt in range(5):
            try:
                headers = {**self.headers, **(kwargs.get("headers") or {})}
                response = self.http.request(method, url, **{**kwargs, "headers": headers})
                if response.status_code >= 500 and attempt < 4:
                    raise httpx.HTTPStatusError("server error", request=response.request, response=response)
                response.raise_for_status()
                return response
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                status = getattr(getattr(exc, "response", None), "status_code", 0)
                if 400 <= status < 500 or attempt == 4:
                    raise
                logger.warning("%s %s failed (%s); retrying", method, url, exc)
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError("unreachable")

    def claim(self, info: dict[str, Any]) -> dict[str, Any] | None:
        response = self._request("POST", "/api/v1/worker/claim", json={
            "name": self.config.name, "info": info, "preset_ids": self.config.preset_ids})
        return None if response.status_code == 204 else response.json()["run"]

    def download(self, url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        with self.http.stream("GET", url, headers=self.headers) as response:
            response.raise_for_status()
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(1024 * 1024):
                    handle.write(chunk)
        os.replace(tmp, target)

    def heartbeat(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/worker/runs/{run_id}/heartbeat", json=payload).json()

    def append_log(self, run_id: str, text: str) -> None:
        self._request("POST", f"/api/v1/worker/runs/{run_id}/log", content=text.encode("utf-8"),
                      headers={"Content-Type": "text/plain; charset=utf-8"})

    def upload_artifact(self, run_id: str, path: Path, name: str, kind: str) -> None:
        with path.open("rb") as handle:
            self._request("POST", f"/api/v1/worker/runs/{run_id}/artifacts",
                          files={"file": (name, handle, "application/octet-stream")}, data={"name": name, "kind": kind})

    def complete(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/worker/runs/{run_id}/complete", json=payload).json()


# ------------------------------------------------------------------------------------------- execution
class RunExecutor(threading.Thread):
    def __init__(self, worker: Worker, run: dict[str, Any]) -> None:
        super().__init__(name=f"run-{run['id']}", daemon=True)
        self.worker, self.run_spec, self.config, self.client = worker, run, worker.config, worker.client
        self.run_id = run["id"]
        self.state = RunState()
        self.in_gpu_phase = True
        self.cancelled = threading.Event()
        self.process: subprocess.Popen | None = None
        self._log_buffer: list[str] = []
        self._log_lock = threading.Lock()
        self._stop_aux = threading.Event()
        base = self.config.data_dir / "runs" / safe_filename(run["workspace_key"], run["id"])
        self.paths = {
            "base": base, "work": base / "work", "media": base / "media", "inputs": base / "inputs" / self.run_id,
            "ci": base / "ci" / self.run_id,
        }

    # -------------------------------------------------------------------- logging / heartbeat
    def log(self, text: str) -> None:
        line = text if text.endswith("\n") else text + "\n"
        with self._log_lock:
            self._log_buffer.append(line)
        logger.debug("[%s] %s", self.run_id, line.rstrip())

    def flush_log(self) -> None:
        with self._log_lock:
            data, self._log_buffer = "".join(self._log_buffer), []
        while data:
            chunk, data = data[: 1024 * 1024], data[1024 * 1024:]
            try:
                self.client.append_log(self.run_id, chunk)
            except Exception as exc:  # noqa: BLE001 - keep running; the log is also on disk
                logger.warning("log upload failed for %s: %s", self.run_id, exc)
                return

    def send_heartbeat(self) -> None:
        try:
            reply = self.client.heartbeat(self.run_id, self.state.snapshot())
        except Exception as exc:  # noqa: BLE001
            logger.warning("heartbeat failed for %s: %s", self.run_id, exc)
            return
        if reply.get("cancel_requested") and not self.cancelled.is_set():
            self.log("[ci] cancel requested by server; stopping")
            self.cancelled.set()
            if self.process is not None:
                kill_process_tree(self.process)

    def _aux_loop(self) -> None:
        last_heartbeat = 0.0
        while not self._stop_aux.wait(self.config.log_flush_interval_s):
            self.flush_log()
            if time.monotonic() - last_heartbeat >= self.config.heartbeat_interval_s:
                last_heartbeat = time.monotonic()
                self.send_heartbeat()

    # -------------------------------------------------------------------- preparation
    def prepare_files(self) -> tuple[Path, list[tuple[str, Path]]]:
        media_info = self.run_spec["media"]
        media = self.paths["media"] / safe_filename(media_info["name"] or "media")
        if not (media.exists() and media.stat().st_size == media_info.get("size")
                and sha256_file(media) == media_info.get("sha256")):
            self.log(f"[ci] downloading media {media_info['name']} ({media_info.get('size')} bytes)")
            self.client.download(media_info["url"], media)
        files = []
        for item in self.run_spec.get("inputs") or []:
            target = self.paths["inputs"] / f"{item['index']:02d}" / safe_filename(item["name"] or "input")
            self.client.download(item["url"], target)
            if item.get("sha256") and sha256_file(target) != item["sha256"]:
                raise RuntimeError(f"checksum mismatch for input {item['name']}")
            files.append((item["flag"], target))
        return media, files

    def build_spec(self, media: Path, files: list[tuple[str, Path]]) -> tuple[Path, Path]:
        options = dict(self.run_spec.get("options") or {})
        for key in ("asr_model", "align_model"):
            if options.get(key):
                options[key] = resolve_model_reference(options[key], self.config.model_cache_dir)
        preset = dict(self.run_spec.get("preset") or {})
        export_path = self.paths["ci"] / "export" / (safe_filename(self.run_spec["title"], self.run_id) + ".srt")
        argv = build_run_argv(preset, workdir=self.paths["work"], media=media, export_path=export_path,
                              file_paths=files, options=options, extra_args=self.run_spec.get("extra_args") or [],
                              worker_args=self.config.default_run_args)
        spec_path = self.paths["ci"] / "spec.json"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(json.dumps({"run_id": self.run_id, "argv": argv,
                                         "reset_from": self.run_spec.get("reset_from")},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        return spec_path, export_path

    # -------------------------------------------------------------------- phases
    def run_phase(self, spec_path: Path, phase: str) -> int:
        command = [*self.config.phase_command, "--spec", str(spec_path), "--phase", phase,
                   "--gpu-lock", str(self.config.data_dir / "gpu.lock")]
        env = {**os.environ, **(self.run_spec.get("env") or {}), **self.config.env,
               "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "KOEAXIS_RUN_ID": self.run_id}
        self.log(f"[ci] phase {phase}: {' '.join(shlex.quote(part) for part in command)}")
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        self.process = subprocess.Popen(
            command, cwd=self.config.project_root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace", bufsize=1, **kwargs)
        assert self.process.stdout is not None
        for line in self.process.stdout:
            event = parse_event(line)
            if event is not None:
                self.state.on_event(event)
                if event.get("event") == "stage" and event.get("status") != "running":
                    self.send_heartbeat()
                elif event.get("event") == "reset":
                    removed = [str(item) for item in event.get("removed") or []]
                    shown = ", ".join(removed[:20]) + (f" ... (+{len(removed) - 20})" if len(removed) > 20 else "")
                    self.log(f"[ci] reset_from={event.get('stage')}: removed {len(removed)} item(s)"
                             f"{': ' + shown if removed else ''}\n")
                continue
            self.log(line)
        code = self.process.wait()
        self.process = None
        self._merge_reports(phase)
        return code

    def _work_paths(self):
        from qwen_asr.models import WorkPaths  # lightweight dataclass; knows both workdir layouts

        return WorkPaths.from_workdir(self.paths["work"])

    def _merge_reports(self, phase: str) -> None:
        # the phase command writes performance reports here (see phase.build_args)
        performance = self.paths["work"] / "reports" / f"pipeline_performance.{phase}.json"
        sources = [(performance, self.state.merge_performance)]
        if phase == API_PHASE:
            sources.append((self._work_paths().final_quality_report, self.state.set_checks))
        for path, handler in sources:
            if path.exists():
                try:
                    handler(json.loads(path.read_text(encoding="utf-8")))
                except (OSError, ValueError) as exc:
                    self.log(f"[ci] cannot read {path.name}: {exc}")

    # -------------------------------------------------------------------- artifacts
    def upload_artifacts(self, export_path: Path) -> None:
        uploads: list[tuple[Path, str, str]] = []
        if export_path.exists():
            uploads.append((export_path, export_path.name, "subtitle"))
            subtitle_count = sum(1 for line in export_path.read_text(encoding="utf-8-sig").splitlines() if "-->" in line)
            self.state.metrics["subtitle_count"] = subtitle_count
            speaker_ass = export_path.with_suffix(".ass")  # speaker-styled ASS, written when --speaker-enroll is set
            if speaker_ass.exists():
                uploads.append((speaker_ass, speaker_ass.name, "subtitle"))
        work = self.paths["work"]
        paths = self._work_paths()
        bundles = {"reports.zip": [work / "reports", work / "diagnostics", work / "logs", paths.progress_path,
                                   paths.content_quality_report, paths.final_quality_report]}
        if self.config.upload_manifests:
            # manifests/ (project layout) or the JSON/SRT/text files at the workdir root (flat layout);
            # audio and per-segment clips are never uploaded.
            loose = [path for path in work.iterdir()
                     if path.is_file() and path.suffix in {".json", ".jsonl", ".srt", ".txt"}] if work.is_dir() else []
            bundles["manifests.zip"] = [work / "manifests", *loose]
        for name, sources in bundles.items():
            archive = self.paths["ci"] / name
            if self._zip(archive, sources, work):
                uploads.append((archive, name, "report" if name == "reports.zip" else "data"))
        for path, name, kind in uploads:
            try:
                self.client.upload_artifact(self.run_id, path, name, kind)
                self.log(f"[ci] uploaded artifact {name} ({path.stat().st_size} bytes)")
            except Exception as exc:  # noqa: BLE001
                self.log(f"[ci] artifact upload failed for {name}: {exc}")

    @staticmethod
    def _zip(archive: Path, sources: list[Path], root: Path) -> bool:
        files = []
        for source in sources:
            if source.is_file():
                files.append(source)
            elif source.is_dir():
                files.extend(path for path in source.rglob("*") if path.is_file())
        files = list(dict.fromkeys(path.resolve() for path in files))
        if not files:
            return False
        archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in files:
                bundle.write(path, path.relative_to(root.resolve()).as_posix())
        return True

    # -------------------------------------------------------------------- main
    def run(self) -> None:  # noqa: C901 - linear orchestration
        aux = threading.Thread(target=self._aux_loop, name=f"aux-{self.run_id}", daemon=True)
        aux.start()
        status, error = "failed", ""
        started = time.monotonic()
        export_path: Path | None = None
        try:
            self.log(f"[ci] worker {self.config.worker_id} starting run {self.run_id} "
                     f"(workspace {self.run_spec['workspace_key']}, reset_from={self.run_spec.get('reset_from')})")
            media, files = self.prepare_files()
            spec_path, export_path = self.build_spec(media, files)
            if export_path.exists():
                export_path.unlink()
            code = 1
            for phase in (GPU_PHASE, API_PHASE):
                if self.cancelled.is_set():
                    break
                self.in_gpu_phase = phase == GPU_PHASE
                code = self.run_phase(spec_path, phase)
                self.in_gpu_phase = False
                if code != 0:
                    error = f"{phase} phase exited with code {code}"
                    break
            if self.cancelled.is_set():
                status, error = "cancelled", "cancelled"
            elif code == 0:
                status = "succeeded"
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            self.log(f"[ci] ERROR {error}")
        finally:
            self.in_gpu_phase = False
            self.state.metrics["wall_time_s"] = round(time.monotonic() - started, 1)
            self.state.finalize()
            if export_path is not None and status != "cancelled":
                self.upload_artifacts(export_path)
            self._stop_aux.set()
            aux.join(timeout=10)
            self.flush_log()
            payload = self.state.snapshot()
            payload.update(status=status, error=error[:2000])
            if status == "succeeded":
                quality = self.state.metrics.get("quality_status")
                payload["summary"] = f"完成：质量门 {quality or '-'}，{self.state.metrics.get('subtitle_count', '?')} 条字幕"
            elif error:
                payload["summary"] = error[:500]
            try:
                self.client.complete(self.run_id, payload)
            except Exception as exc:  # noqa: BLE001
                logger.error("cannot report completion of %s: %s", self.run_id, exc)


class Worker:
    def __init__(self, config: WorkerConfig, client: ServerClient | None = None) -> None:
        self.config = config
        self.client = client or ServerClient(config)
        self.active: dict[str, RunExecutor] = {}
        self.stop = threading.Event()
        config.data_dir.mkdir(parents=True, exist_ok=True)

    def info(self) -> dict[str, Any]:
        return {
            "host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "max_runs": self.config.max_runs, "active": sorted(self.active), "pid": os.getpid(),
            "default_run_args": self.config.default_run_args, "presets": self.config.preset_ids,
            "free_disk_gb": round(shutil.disk_usage(self.config.data_dir).free / 1024**3, 1),
        }

    def can_claim(self) -> bool:
        self.active = {key: value for key, value in self.active.items() if value.is_alive()}
        return len(self.active) < self.config.max_runs and not any(item.in_gpu_phase for item in self.active.values())

    def poll_once(self) -> RunExecutor | None:
        if not self.can_claim():
            return None
        run = self.client.claim(self.info())
        if not run:
            return None
        executor = RunExecutor(self, run)
        self.active[run["id"]] = executor
        executor.start()
        logger.info("claimed run %s (%s)", run["id"], run.get("title"))
        return executor

    def serve(self, *, once: bool = False) -> None:
        logger.info("worker %s polling %s", self.config.worker_id, self.config.server)
        while not self.stop.is_set():
            try:
                executor = self.poll_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("claim failed: %s", exc)
                executor = None
            if once and executor is not None:
                executor.join()
                return
            if executor is None:
                self.stop.wait(self.config.poll_interval_s)
        for executor in self.active.values():
            executor.join()
