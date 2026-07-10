from __future__ import annotations

import re
import time
from pathlib import Path

from qwen_asr.artifact_state import stage_statuses
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress
from qwen_asr.storage import read_json

def get_status(workdir_value: str) -> dict:
    if not workdir_value:
        return {"error": "workdir is required"}
    work_paths = WorkPaths.from_workdir(Path(workdir_value))
    if not work_paths.workdir.exists():
        return {
            "workdir": str(work_paths.workdir),
            "exists": False,
            "files": {},
            "counts": {},
            "logs": {},
            "stages": stage_statuses(work_paths),
            "batch_summary": None,
        }
    status = {
        "workdir": str(work_paths.workdir),
        "exists": True,
        "files": {
            "audio": work_paths.audio_path.exists(),
            "segments": work_paths.segments_manifest.exists(),
            "transcript": work_paths.transcript_manifest.exists(),
            "corrected": work_paths.corrected_manifest.exists(),
            "aligned": work_paths.aligned_manifest.exists(),
            "split": work_paths.split_manifest.exists(),
            "translated": work_paths.translated_manifest.exists(),
            "normalized": work_paths.normalized_manifest.exists(),
            "srt": work_paths.subtitles_srt.exists(),
            "vtt": work_paths.subtitles_vtt.exists(),
        },
        "counts": {},
        "logs": {},
        "stages": stage_statuses(work_paths),
        "batch_summary": _read_batch_summary(work_paths.workdir),
    }
    counts = {
        "segments": work_paths.segments_manifest,
        "transcript": work_paths.transcript_manifest,
        "corrected": work_paths.corrected_manifest,
        "aligned": work_paths.aligned_manifest,
        "split": work_paths.split_manifest,
        "translated": work_paths.translated_manifest,
        "normalized": work_paths.normalized_manifest,
    }
    for key, path in counts.items():
        if not path.exists():
            continue
        payload = read_json(path, default=[] if key in {"segments", "transcript", "corrected", "aligned"} else {})
        status["counts"][key] = len(payload) if isinstance(payload, list) else len(payload.keys())

    for log_name in (
        "prepare",
        "transcribe",
        "correct",
        "align",
        "normalize",
        "split",
        "translate",
        "export",
        "mimo-proofread",
        "run",
        "batch-run",
    ):
        log_path = work_paths.logs_dir / f"{log_name}.log"
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            status["logs"][log_name] = "\n".join(lines[-20:])
    return status


def _read_batch_summary(workdir: Path) -> dict | None:
    summary_path = workdir / "summary" / "batch-summary.json"
    if not summary_path.exists():
        return None
    payload = read_json(summary_path, default=None)
    return payload if isinstance(payload, dict) else None

def build_progress(job: dict) -> dict:
    work_paths = WorkPaths.from_workdir(Path(job["workdir"]))
    stage = job["stage"]
    started_at = job.get("started_at", time.time())
    finished_at = job.get("finished_at")
    end_time = finished_at if job.get("status") != "running" and finished_at else time.time()
    elapsed = max(0.0, end_time - started_at)
    progress = {
        "stage": stage,
        "status": job["status"],
        "elapsed_seconds": round(elapsed, 1),
        "summary": "",
        "completed": None,
        "total": None,
        "current": "",
    }
    saved_progress = read_progress(work_paths)
    if saved_progress and saved_progress.get("stage"):
        progress.update(
            {
                "stage": saved_progress.get("stage", stage),
                "status": saved_progress.get("status", job["status"]),
                "summary": saved_progress.get("summary", ""),
                "completed": saved_progress.get("done"),
                "total": saved_progress.get("total"),
                "current": saved_progress.get("current", ""),
                "updated_at": saved_progress.get("updated_at", ""),
            }
        )
        return progress

    log_path = work_paths.logs_dir / f"{stage}.log"
    recent_lines: list[str] = []
    if log_path.exists():
        recent_lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-50:]

    if stage == "prepare":
        total = _read_count(work_paths.segments_manifest)
        progress["completed"] = total
        progress["total"] = total
        progress["summary"] = f"prepared {total or 0} segments" if total else "extracting audio / running VAD"
        progress["current"] = _find_last_matching(recent_lines, ["Running CPU VAD", "Extracting audio", "Extracting and denoising audio", "Prepared "])
        return progress

    if stage == "transcribe":
        total = _read_count(work_paths.segments_manifest)
        completed = _read_completed_list_count(work_paths.transcript_manifest)
        progress["completed"] = completed
        progress["total"] = total
        progress["summary"] = f"{completed}/{total or '?'} transcript segments"
        progress["current"] = _find_last_matching(recent_lines, ["Transcribing segment_", "Skipping completed segment", "All transcript segments"])
        return progress

    if stage == "align":
        total = _read_completed_list_count(work_paths.transcript_manifest)
        completed = _read_completed_list_count(work_paths.aligned_manifest)
        progress["completed"] = completed
        progress["total"] = total
        progress["summary"] = f"{completed}/{total or '?'} aligned segments"
        progress["current"] = _find_last_matching(recent_lines, ["Aligning segment_", "Skipping completed alignment", "All alignment segments"])
        return progress

    if stage == "correct":
        total = _read_completed_list_count(work_paths.transcript_manifest)
        completed = _read_completed_list_count(work_paths.corrected_manifest)
        changed = _count_changed_corrections(work_paths.corrected_manifest)
        progress["completed"] = completed
        progress["total"] = total
        progress["summary"] = f"{completed}/{total or '?'} corrected segments, {changed} changed"
        progress["current"] = _find_last_matching(recent_lines, ["correcting ASR text", "ASR correction report", "corrected_segments.json already exists"])
        return progress

    if stage == "split":
        output_count = _read_json_dict_count(work_paths.split_manifest)
        if job["status"] == "completed":
            progress["completed"] = output_count
            progress["total"] = output_count
            progress["summary"] = f"completed {output_count or 0} split subtitles"
        else:
            progress["completed"] = None
            progress["total"] = None
            progress["summary"] = f"generated {output_count or 0} split subtitles"
        progress["current"] = _find_last_matching(recent_lines, ["规则分割", "开始调用 API", "split_segments.json already exists"])
        return progress

    if stage == "translate":
        total = _read_json_dict_count(work_paths.split_manifest)
        completed = _count_translated_items(work_paths.translated_manifest)
        if job["status"] == "running" and completed == 0:
            completed = _estimate_translate_completed_from_log(recent_lines, total)
        if job["status"] == "completed" and total:
            progress["completed"] = total
            progress["total"] = total
            progress["summary"] = f"completed {total} translated subtitles"
        else:
            progress["completed"] = completed
            progress["total"] = total
            progress["summary"] = f"{completed}/{total or '?'} translated subtitles"
        progress["current"] = _find_last_matching(recent_lines, ["正在翻译字幕", "translated_segments.json already exists"])
        return progress

    if stage == "normalize":
        output_count = _read_json_dict_count(work_paths.normalized_manifest)
        if job["status"] == "completed":
            progress["completed"] = output_count
            progress["total"] = output_count
            progress["summary"] = f"completed {output_count or 0} normalized subtitles"
        else:
            progress["summary"] = f"generated {output_count or 0} normalized subtitles"
        progress["current"] = _find_last_matching(recent_lines, ["Normalized ", "normalized_segments.json already exists"])
        return progress

    if stage == "export":
        source = "normalized" if work_paths.normalized_manifest.exists() else "translated" if work_paths.translated_manifest.exists() else "split" if work_paths.split_manifest.exists() else "aligned" if work_paths.aligned_manifest.exists() else "transcript"
        progress["summary"] = f"export source: {source}"
        ready = []
        if work_paths.subtitles_srt.exists():
            ready.append("srt")
        if work_paths.subtitles_vtt.exists():
            ready.append("vtt")
        progress["current"] = ", ".join(ready) if ready else _find_last_matching(recent_lines, ["No timestamped cues", "Saved"])
        return progress

    if stage == "mimo-proofread":
        progress["summary"] = "MiMo audio proofread running"
        progress["current"] = recent_lines[-1] if recent_lines else ""
        return progress

    if stage == "run":
        progress["summary"] = _summarize_run(work_paths)
        progress["current"] = _find_last_matching(
            recent_lines,
            [
                "Running CPU VAD",
                "Extracting audio",
                "Extracting and denoising audio",
                "Prepared ",
                "Transcribing segment_",
                "correcting ASR text",
                "ASR correction report",
                "Aligning segment_",
                "Normalized ",
                "规则分割",
                "正在翻译字幕",
                "All transcript segments already completed",
                "All alignment segments already completed",
                "normalized_segments.json already exists",
                "split_segments.json already exists",
                "translated_segments.json already exists",
                "Command failed",
            ],
        )
        return progress

    return progress

def _read_count(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = read_json(path, default=[])
    return len(payload) if isinstance(payload, list) else len(payload.keys())

def _read_completed_list_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default=[])
    if not isinstance(payload, list):
        return 0
    return sum(1 for item in payload if item.get("status", "completed") == "completed")

def _read_json_dict_count(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default={})
    return len(payload.keys()) if isinstance(payload, dict) else 0

def _count_translated_items(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        return 0
    return sum(1 for item in payload.values() if str(item.get("translated_subtitle", "")).strip())

def _estimate_translate_completed_from_log(lines: list[str], total: int) -> int:
    if total <= 0:
        return 0

    batch_sizes = [
        int(match.group(1))
        for line in lines
        for match in [re.search(r"正在翻译字幕：(\d+)\s*条", line)]
        if match
    ]
    if not batch_sizes:
        return 0

    successful_requests = sum(
        1
        for line in lines
        if "HTTP Request:" in line and "chat/completions" in line and "200 OK" in line
    )
    if successful_requests <= 0:
        return 0

    completed = sum(batch_sizes[:successful_requests])
    return min(total, completed)

def _count_changed_corrections(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default=[])
    if not isinstance(payload, list):
        return 0
    return sum(1 for item in payload if item.get("changed"))

def _count_aligned_tokens(path: Path) -> int:
    if not path.exists():
        return 0
    payload = read_json(path, default=[])
    if not isinstance(payload, list):
        return 0
    return sum(len(item.get("tokens", [])) for item in payload if item.get("status") == "completed")

def _find_last_matching(lines: list[str], needles: list[str]) -> str:
    for line in reversed(lines):
        if any(needle in line for needle in needles):
            return line
    return ""

def _summarize_run(work_paths: WorkPaths) -> str:
    steps = []
    if work_paths.segments_manifest.exists():
        steps.append("prepare")
    if work_paths.transcript_manifest.exists():
        steps.append("transcribe")
    if work_paths.corrected_manifest.exists():
        steps.append("correct")
    if work_paths.aligned_manifest.exists():
        steps.append("align")
    if work_paths.split_manifest.exists():
        steps.append("split")
    if work_paths.translated_manifest.exists():
        steps.append("translate")
    if work_paths.normalized_manifest.exists():
        steps.append("normalize")
    if work_paths.subtitles_srt.exists() or work_paths.subtitles_vtt.exists():
        steps.append("export")
    return "completed: " + ", ".join(steps) if steps else "running pipeline"
