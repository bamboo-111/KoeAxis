"""Automatic alignment recovery used by CI runs after the `align` stage.

For every segment whose alignment failed: retry Qwen (bf16), then local MFA (prefetched once for all
segments, optionally on a background thread); segments every exact strategy rejected get audited
segment-level ("coarse") timing derived from VAD, marked as unreviewed ASR text in the recovery state.
Every attempt is appended to `reports/workflow_alignment_recovery.json` / `workflow_coarse_acceptance.json`.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.recovery_executor import execute_alignment_recovery, start_alignment_recovery_mfa_prefetch
from qwen_asr.storage import read_json, write_json_atomic

ACTOR = "ci-recovery"
STRATEGIES = (("qwen", "bf16"), ("mfa-local", "fp16"))


def failed_alignment_count(paths: WorkPaths) -> int:
    return sum(item.get("status") == "failed" for item in read_json(paths.aligned_manifest, default=[]))


def run_alignment_recovery(
    paths: WorkPaths,
    *,
    settings: dict[str, Any],
    language_route: str = "Japanese",
    gpu_lock: Callable[[], AbstractContextManager[Any]] | None = None,
    log: Callable[..., None] = print,
) -> dict[str, Any]:
    """Recover failed alignments in place; returns a summary for the run report."""
    gpu_lock = gpu_lock or nullcontext
    started = time.monotonic()
    initial = failed_alignment_count(paths)
    summary: dict[str, Any] = {"failed_initial": initial}
    if not initial:
        summary.update(failed_remaining=0, elapsed_s=0.0)
        return summary
    _clear_dead_recovery_lock(paths, log)
    future = start_alignment_recovery_mfa_prefetch(paths, language_route=language_route)
    with gpu_lock():
        _recover_exact(paths, settings=settings, language_route=language_route, log=log)
    try:
        prefetch = future.result()
        summary["mfa_prefetch"] = {
            **{key: value for key, value in prefetch.items() if key != "batches"},
            "dictionary": [row.get("dictionary") for row in prefetch.get("batches", [])],
            "mfa_elapsed_ms": [row.get("elapsed_ms") for row in prefetch.get("batches", [])],
        }
    except Exception as exc:  # noqa: BLE001 - recorded; recovery itself already ran uncached
        summary["mfa_prefetch"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if failed_alignment_count(paths):
        summary["coarse"] = accept_coarse_timing(paths, log=log)
    summary["failed_remaining"] = failed_alignment_count(paths)
    summary["elapsed_s"] = round(time.monotonic() - started, 1)
    return summary


def _recover_exact(paths: WorkPaths, *, settings: dict[str, Any], language_route: str, log: Callable[..., None]) -> None:
    report = paths.workdir / "reports" / "workflow_alignment_recovery.json"
    attempts = read_json(report, default=[])
    for target in read_json(paths.aligned_manifest, default=[]):
        if target.get("status") != "failed":
            continue
        segment_id = target["segment_id"]
        tried = {
            (row.get("segment_id"), row.get("strategy"))
            for row in attempts
            if "already active" not in str(row.get("error") or "")
        }
        for strategy, dtype in STRATEGIES:
            if (segment_id, strategy) in tried:
                continue
            try:
                result = execute_alignment_recovery(
                    paths,
                    segment_id=segment_id,
                    strategy=strategy,
                    language_route=language_route,
                    actor=ACTOR,
                    settings={**settings, "dtype": dtype},
                )
            except Exception as exc:  # noqa: BLE001 - one failed attempt must not stop the others
                result = {"segment_id": segment_id, "strategy": strategy, "status": "error",
                          "error": f"{type(exc).__name__}: {exc}"}
            attempts.append(result)
            write_json_atomic(report, attempts)
            log("RECOVERY", segment_id, strategy, result.get("status"), result.get("error") or "")
            if result.get("alignment_state") == "completed_exact":
                break


def accept_coarse_timing(paths: WorkPaths, *, log: Callable[..., None] = print) -> list[dict[str, Any]]:
    """Segment-level timing for segments every exact strategy rejected (audited, marked unreviewed)."""
    from qwen_asr.recovery_service import RecoveryError, build_recovery_view, perform_recovery_action

    report = paths.workdir / "reports" / "workflow_coarse_acceptance.json"
    entries = read_json(report, default=[])
    new_entries: list[dict[str, Any]] = []
    for task in build_recovery_view(paths)["items"]:
        segment_id = task["segment_id"]
        entry: dict[str, Any] = {"segment_id": segment_id, "error": task["error"], "text": task["text"],
                                 "start_ms": task["start_ms"], "end_ms": task["end_ms"]}
        try:
            perform_recovery_action(
                paths, segment_id=segment_id, action="verify_transcript", actor=ACTOR,
                payload={"verified_text": task["original_transcript"] or task["text"],
                         "source": "asr_unreviewed_batch_auto"},
            )
            start, end = task["start_ms"], task["end_ms"]
            short = task.get("priority") == "short_response"
            limit = min(6000, round((end - start) * 0.8)) - 1
            # Short replies must not stretch over most of a long segment: pick ONE VAD region whose length is
            # closest to the expected speaking time; without a usable region fall back to an expected-length
            # window at the segment start (flagged in the report for review).
            expected = max(600, 220 * max(1, len(str(task["original_transcript"] or task["text"]).strip())))
            try:
                vad = perform_recovery_action(paths, segment_id=segment_id, action="localize_vad", actor=ACTOR,
                                              payload={})["task"]["vad_proposal"]
                regions = vad.get("regions") or []
                entry["vad_regions"] = len(regions)
                if regions and not short:
                    start = min(region["start_ms"] for region in regions)
                    end = max(region["end_ms"] for region in regions)
                elif regions:
                    fitting = [r for r in regions if r["end_ms"] - r["start_ms"] <= limit] or regions
                    pick = min(fitting, key=lambda r: abs((r["end_ms"] - r["start_ms"]) - expected))
                    start, end = pick["start_ms"], min(pick["end_ms"], pick["start_ms"] + limit)
                    entry["short_region_pick"] = pick.get("index")
            except RecoveryError as exc:
                entry["vad_error"] = f"{exc.code}: {exc}"
            if short and end - start > limit:
                end = start + min(expected, limit)
                entry["short_fallback_window"] = True
            previous = (task.get("context") or {}).get("previous") or {}
            following = (task.get("context") or {}).get("next") or {}
            if previous.get("end_ms") is not None:
                start = max(start, int(previous["end_ms"]))
            if following.get("start_ms") is not None:
                end = min(end, int(following["start_ms"]))
            entry["coarse_span_ms"] = {"start_ms": start, "end_ms": end}
            result = perform_recovery_action(paths, segment_id=segment_id, action="accept_completed_coarse",
                                             actor=ACTOR, payload={"start_ms": start, "end_ms": end})
            entry.update(status="completed_coarse", bounds=result["task"]["result"])
        except RecoveryError as exc:
            entry.update(status="failed", reason=f"{exc.code}: {exc}")
        entries.append(entry)
        new_entries.append(entry)
        write_json_atomic(report, entries)
        log("COARSE", segment_id, entry["status"], entry.get("reason", ""))
    return new_entries


def _clear_dead_recovery_lock(paths: WorkPaths, log: Callable[..., None]) -> None:
    lock = paths.workdir / "reports" / "recovery-executor.lock"
    if not lock.exists():
        return
    try:
        pid = int(json.loads(lock.read_text(encoding="utf-8")).get("pid"))
    except (OSError, ValueError, TypeError):
        pid = None
    if pid and _pid_alive(pid):
        return
    lock.unlink(missing_ok=True)
    log("RECOVERY_LOCK_CLEARED", pid)


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def recovery_settings(args: Any, model_cache_default: Path) -> dict[str, Any]:
    return {
        "model_cache_dir": str(getattr(args, "model_cache_dir", None) or model_cache_default),
        "device": str(getattr(args, "device", None) or "cuda:0"),
        "local_files_only": bool(getattr(args, "local_files_only", True)),
    }
