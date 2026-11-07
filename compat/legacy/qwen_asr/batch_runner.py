from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from qwen_asr.models import WorkPaths
from qwen_asr.pipeline_runner import PipelineRunner
from qwen_asr.preflight import ensure_preflight
from qwen_asr.storage import ensure_directory, load_jsonl, write_json_atomic
from qwen_asr.web.commands import suggest_workdir

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchTask:
    task_id: int
    media: str
    workdir: str
    requested_workdir: str | None = None


def run_batch_command(args: argparse.Namespace, handlers: dict[str, object]) -> int:
    batch_root = Path(args.workdir).resolve()
    ensure_directory(batch_root)
    summary_dir = batch_root / "summary"
    logs_dir = batch_root / "logs"
    ensure_directory(summary_dir)
    ensure_directory(logs_dir)
    batch_log = logs_dir / "batch-run.log"

    tasks = _load_tasks(args)
    if not tasks:
        raise RuntimeError("batch-run requires at least one media input from positional paths or --manifest.")

    started_at = _now_iso()
    _append_batch_log(batch_log, f"batch started total={len(tasks)} workdir={batch_root}")
    summary_rows: list[dict[str, object]] = [_initial_summary_row(task) for task in tasks]
    prepare_workers = max(1, int(getattr(args, "prepare_workers", 2)))
    _append_batch_log(batch_log, f"prepare phase started workers={prepare_workers}")
    failures = 0

    with ThreadPoolExecutor(max_workers=prepare_workers) as executor:
        future_map = {
            executor.submit(_prepare_only, args, task, handlers): (task, perf_counter())
            for task in tasks
        }
        for future in as_completed(future_map):
            task, prepare_started = future_map[future]
            row = summary_rows[task.task_id]
            row["started_at"] = _now_iso()
            try:
                status = future.result()
                if status != 0:
                    raise RuntimeError(f"prepare returned exit code {status}")
                row["prepare_status"] = "completed"
                row["status"] = "prepared"
                row["elapsed_s"] = round(perf_counter() - prepare_started, 3)
                row["finished_at"] = _now_iso()
                _append_batch_log(batch_log, f"task {task.task_id + 1} prepare completed workdir={task.workdir}")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                failures += 1
                row["prepare_status"] = "failed"
                row["status"] = "failed"
                row["failed_stage"] = "prepare"
                row["error"] = str(exc)
                row["elapsed_s"] = round(perf_counter() - prepare_started, 3)
                row["finished_at"] = _now_iso()
                _append_batch_log(batch_log, f"task {task.task_id + 1} prepare failed error={exc}")
                if getattr(args, "fail_fast", False):
                    _write_batch_summary(summary_dir, summary_rows, started_at=started_at, finished_at=_now_iso())
                    return 1

    for task in tasks:
        row = summary_rows[task.task_id]
        if row.get("prepare_status") == "failed":
            continue
        started = perf_counter()
        row["started_at"] = _now_iso()
        _append_batch_log(batch_log, f"task {task.task_id + 1} run started workdir={task.workdir} media={task.media}")
        try:
            task_args = _task_args(args, task)
            ensure_preflight(task_args, WorkPaths.from_workdir(Path(task.workdir)), "run")
            status = PipelineRunner(WorkPaths.from_workdir(Path(task.workdir)), handlers).run(task_args)
            row["status"] = "completed" if status == 0 else "failed"
            row["failed_stage"] = "" if status == 0 else "run"
            row["elapsed_s"] = round(perf_counter() - started, 3)
            row["finished_at"] = _now_iso()
            _append_batch_log(batch_log, f"task {task.task_id + 1} run {row['status']} elapsed_s={row['elapsed_s']}")
            if status != 0:
                failures += 1
                if getattr(args, "fail_fast", False):
                    _write_batch_summary(summary_dir, summary_rows, started_at=started_at, finished_at=_now_iso())
                    return 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures += 1
            row["status"] = "failed"
            row["failed_stage"] = "run"
            row["elapsed_s"] = round(perf_counter() - started, 3)
            row["error"] = str(exc)
            row["finished_at"] = _now_iso()
            _append_batch_log(batch_log, f"task {task.task_id + 1} run failed error={exc}")
            if getattr(args, "fail_fast", False):
                _write_batch_summary(summary_dir, summary_rows, started_at=started_at, finished_at=_now_iso())
                return 1

    finished_at = _now_iso()
    _write_batch_summary(summary_dir, summary_rows, started_at=started_at, finished_at=finished_at)
    _append_batch_log(batch_log, f"batch finished failures={failures} summary={summary_dir / 'batch-summary.json'}")
    return 0 if failures == 0 else 1


def _prepare_only(args: argparse.Namespace, task: BatchTask, handlers: dict[str, object]) -> int:
    task_args = _task_args(args, task)
    ensure_preflight(task_args, WorkPaths.from_workdir(Path(task.workdir)), "prepare")
    return handlers["prepare"](task_args, WorkPaths.from_workdir(Path(task.workdir)))


def _task_args(args: argparse.Namespace, task: BatchTask) -> argparse.Namespace:
    task_args = argparse.Namespace(**vars(args))
    task_args.command = "run"
    task_args.media = task.media
    task_args.workdir = task.workdir
    return task_args


def _load_tasks(args: argparse.Namespace) -> list[BatchTask]:
    tasks: list[BatchTask] = []
    used_workdirs: set[str] = set()
    for media in list(getattr(args, "media_files", []) or []):
        media_path = str(Path(media).resolve())
        requested = str(suggest_workdir(str(media)).resolve())
        tasks.append(_make_task(len(tasks), media_path, requested, used_workdirs, None))
    manifest_path = getattr(args, "manifest", None)
    if manifest_path:
        path = Path(str(manifest_path))
        if path.suffix.lower() == ".jsonl":
            entries = load_jsonl(path)
        else:
            with path.open("r", encoding="utf-8-sig") as handle:
                entries = json.load(handle)
        if not isinstance(entries, list):
            raise RuntimeError("--manifest must contain a JSON array or JSONL rows.")
        for entry in entries:
            if not isinstance(entry, dict) or not str(entry.get("media", "")).strip():
                raise RuntimeError("Each batch manifest item must include a non-empty media field.")
            media = str(Path(str(entry["media"])).resolve())
            requested = str(Path(str(entry.get("workdir") or suggest_workdir(media))).resolve())
            tasks.append(_make_task(len(tasks), media, requested, used_workdirs, requested if entry.get("workdir") else None))
    return tasks


def _make_task(task_id: int, media: str, requested_workdir: str, used_workdirs: set[str], explicit_workdir: str | None) -> BatchTask:
    workdir = _unique_workdir(Path(requested_workdir), used_workdirs)
    return BatchTask(
        task_id=task_id,
        media=media,
        workdir=str(workdir),
        requested_workdir=explicit_workdir if explicit_workdir and str(Path(explicit_workdir).resolve()) != str(workdir) else None,
    )


def _unique_workdir(base: Path, used_workdirs: set[str]) -> Path:
    candidate = base.resolve()
    index = 2
    while str(candidate).lower() in used_workdirs:
        candidate = candidate.with_name(f"{base.name}-{index}").resolve()
        index += 1
    used_workdirs.add(str(candidate).lower())
    return candidate


def _initial_summary_row(task: BatchTask) -> dict[str, object]:
    row: dict[str, object] = {
        "task_id": task.task_id,
        "media": task.media,
        "workdir": task.workdir,
        "prepare_status": "pending",
        "status": "pending",
        "failed_stage": "",
        "error": "",
        "elapsed_s": 0.0,
        "started_at": "",
        "finished_at": "",
    }
    if task.requested_workdir:
        row["requested_workdir"] = task.requested_workdir
    return row


def _write_batch_summary(summary_dir: Path, rows: list[dict[str, object]], *, started_at: str | None = None, finished_at: str | None = None) -> None:
    normalized_rows = [_normalize_summary_row(row) for row in rows]
    payload = {
        "total": len(normalized_rows),
        "succeeded": sum(1 for row in normalized_rows if row.get("status") == "completed"),
        "failed": sum(1 for row in normalized_rows if row.get("status") == "failed"),
        "skipped": sum(1 for row in normalized_rows if row.get("status") == "skipped"),
        "started_at": started_at or "",
        "finished_at": finished_at or "",
        "tasks": normalized_rows,
    }
    write_json_atomic(summary_dir / "batch-summary.json", payload)
    lines = [
        f"total={payload['total']}",
        f"succeeded={payload['succeeded']}",
        f"failed={payload['failed']}",
        f"skipped={payload['skipped']}",
        "status\tstage\telapsed_s\tworkdir\tmedia\terror",
    ]
    for row in normalized_rows:
        lines.append(
            "\t".join(
                [
                    str(row.get("status", "unknown")),
                    str(row.get("failed_stage") or row.get("prepare_status") or ""),
                    str(row.get("elapsed_s", "")),
                    str(row["workdir"]),
                    str(row["media"]),
                    str(row.get("error", "")),
                ]
            )
        )
    (summary_dir / "batch-summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _normalize_summary_row(row: dict[str, object]) -> dict[str, object]:
    normalized = dict(row)
    status = str(normalized.get("status") or normalized.get("prepare_status") or "pending")
    if status == "prepared":
        status = "pending"
    normalized["status"] = status
    normalized.setdefault("prepare_status", "")
    normalized.setdefault("failed_stage", "")
    normalized.setdefault("error", "")
    normalized.setdefault("elapsed_s", 0.0)
    normalized.setdefault("started_at", "")
    normalized.setdefault("finished_at", "")
    return normalized


def _append_batch_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
    path.write_text(existing + f"{_now_iso()} {message}\n", encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
