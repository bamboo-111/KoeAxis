#!/usr/bin/env python3
"""Build a progress snapshot (JSON) of a KoeAxis fixed-chain run for a web UI.

Inputs, all read-only, from the run output directory:
  work/progress.json     done/total/current/updated_at. `current` is the LAST COMPLETED
                         stage, so the running stage is STAGES[done].
  launch-options.json    its mtime is the run start (override with --started-at).
  performance.json       only present after the run ends; used for final stage times.

Estimates use bounded, earlier completed runs on the same known device. Missing
duration or unknown cold ASR latency suppresses the countdown.

Usage:
  python tools/server/progress_snapshot.py --run-dir /opt/koeaxis/runs/experiments/SERVER-LINUX-RUN-9
      [--baseline other/performance.json] [--started-at 2026-11-02T07:51:59+00:00]
"""
import argparse
import datetime as dt
import json
import sys
import time
import math
import os
import statistics
import heapq
import wave
from pathlib import Path

STAGES = ["prepare", "transcribe", "correct", "align", "speaker", "split", "translate",
          "audio_review", "proofread_realign", "quality", "normalize", "export"]
LABELS = {"prepare": "音频准备", "transcribe": "云端转写", "correct": "文本校正", "align": "强制对齐",
          "speaker": "说话人识别", "split": "断句", "translate": "翻译", "audio_review": "音频复核",
          "proofread_realign": "校对重对齐", "quality": "质量检查", "normalize": "规范化", "export": "导出"}
# SERVER-LINUX-RUN-8 wall seconds (stage.prepare ... stage.export), rounded.
DEFAULT_SECONDS = {"prepare": 76, "transcribe": 1, "correct": 1, "align": 358, "speaker": 766,
                   "split": 122, "translate": 81, "audio_review": 243, "proofread_realign": 34,
                   "quality": 5, "normalize": 5, "export": 13}
STARTUP_SECONDS = 7  # input_signature before the first stage


def parse_time(text):
    return dt.datetime.fromisoformat(text).timestamp()


def baseline_from_performance(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = dict(DEFAULT_SECONDS)
    for row in data.get("stages", []):
        name = row.get("name", "").removeprefix("stage.")
        if name in out and "wall_seconds" in row:
            out[name] = max(1.0, float(row["wall_seconds"]))
    return out

def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def positive(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n > 0 else None
    except (TypeError, ValueError):
        return None


def media_info(run, config=None):
    run = Path(run)
    source = read_json(run / "work/media-source.json")
    config = config or {}
    config = config.get("config", config)
    config = config if isinstance(config, dict) else {}
    device = str(source.get("device") or config.get("device") or "").lower()
    device = "cpu" if device == "cpu" else "gpu" if device.startswith(("cuda", "gpu")) else None
    duration = positive(source.get("duration_s"))
    if duration is None:
        try:
            with wave.open(str(run / "work/audio.wav"), "rb") as audio:
                duration = positive(audio.getnframes() / audio.getframerate())
        except (OSError, EOFError, ValueError, wave.Error, ZeroDivisionError):
            pass
    if duration is None:
        duration = positive(read_json(run / "work/cloud-asr-usage.json").get("audio_seconds"))
    return duration, device


def historical_baseline(run, started, duration=None, device=None, config=None):
    """Inspect at most 200 recent sibling runs; exclude current/future completions."""
    run = Path(run)
    found_duration, found_device = media_info(run, config)
    duration, device = duration or found_duration, device or found_device
    samples = {s: [] for s in STAGES}
    count = 0
    try:
        with os.scandir(run.parent) as entries:
            candidates = heapq.nlargest(200, (e for e in entries if e.is_dir(follow_symlinks=False)),
                                       key=lambda e: e.stat(follow_symlinks=False).st_mtime_ns)
    except OSError:
        candidates = []
    for entry in candidates:
        try:
            other = Path(entry.path)
            if not entry.is_dir(follow_symlinks=False) or other.resolve() == run.resolve():
                continue
            prog = read_json(other / "work/progress.json")
            if prog.get("status") != "completed" or not prog.get("updated_at"):
                continue
            if parse_time(prog["updated_at"]) >= started:
                continue
            cfg = read_json(other / "job-config.json") or read_json(other / "launch-options.json")
            if other.name.startswith("WEBUI-"):
                cfg = read_json(run.parent.parent.parent / "webui/jobs" / (other.name[6:] + ".config.json")) or cfg
            seconds, kind = media_info(other, cfg)
            if not device or kind != device or not seconds:
                continue
            perf = read_json(other / "performance.json")
            rows = perf.get("stages", [])
            if not isinstance(rows, list):
                continue
            used = False
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name", "")).removeprefix("stage.")
                value = positive(row.get("wall_seconds"))
                if name in samples and name != "transcribe" and value:
                    samples[name].append(value / seconds)
                    used = True
            count += int(used)
        except (OSError, ValueError, TypeError, OverflowError):
            continue
    scale = duration / 1560 if duration else 1.0
    expected = {s: max(1.0, DEFAULT_SECONDS[s] * scale) for s in STAGES}
    if duration:
        for name, values in samples.items():
            if values:
                expected[name] = max(1.0, statistics.median(values) * duration)
    usage = read_json(run / "work/cloud-asr-usage.json")
    cache = "unknown"
    if usage.get("status") == "completed" and usage.get("backend") != "local":
        requests = usage.get("requests")
        hits = usage.get("cache_hits")
        if isinstance(requests, int) and isinstance(hits, int):
            cache = "cold_or_mixed" if requests > 0 else "warm" if hits > 0 else "unknown"
    return expected, {"duration_s": duration, "device": device, "history_runs": count,
                      "fallback_stages": [s for s in STAGES if s != "transcribe" and not samples[s]],
                      "basis": "same-device median per media second; missing stages use CPU reference" if count else
                               "26-minute CPU reference scaled by duration (low confidence)",
                      "asr_cache": cache}


def build_snapshot(progress, expected, started_at, now, run_state=None, estimate=None):
    """progress: parsed progress.json (or None); run_state: optional 'failed' override."""
    total = len(STAGES)
    progress = progress or {}
    done = max(0, min(int(progress.get("done", 0)), total))
    status = progress.get("status", "pending")
    if run_state in ("failed", "cancelled", "completed"):
        status = run_state
    finished = status == "completed" or (done >= total and status not in ("failed", "cancelled"))
    if not finished and done == total:
        done = total - 1
    if finished:
        done = total
        if progress.get("updated_at"):
            now = min(now, parse_time(progress["updated_at"]))
    stage_start = parse_time(progress["updated_at"]) if progress.get("updated_at") and done else started_at + STARTUP_SECONDS
    elapsed = max(0.0, now - started_at)

    done_expected = sum(expected[s] for s in STAGES[:done])
    speed = 1.0
    if estimate is None and done and done_expected > 0:
        speed = min(2.0, max(0.5, (stage_start - started_at - STARTUP_SECONDS) / done_expected))

    stages = []
    for i, name in enumerate(STAGES):
        state = "done" if i < done else ("running" if i == done and not finished and status not in ("failed", "cancelled") else
                                          "failed" if i == done and status in ("failed", "cancelled") else "pending")
        stages.append({"name": name, "label": LABELS[name], "state": state,
                       "expected_s": round(expected[name] * (speed if state != "done" else 1.0), 1)})

    current = None
    remaining = 0.0
    if finished:
        overall_percent = 100.0
    else:
        idx = done
        name = STAGES[idx]
        exp = expected[name] * speed
        stage_elapsed = max(0.0, now - stage_start)
        overrun = stage_elapsed >= exp
        stage_percent = 99.0 if overrun else round(100.0 * stage_elapsed / exp, 1)
        stage_remaining = max(exp * 0.05, exp - stage_elapsed) if not overrun else 0.0
        current = {"name": name, "label": LABELS[name], "index": idx + 1, "total": total,
                   "percent": stage_percent, "elapsed_s": round(stage_elapsed), "expected_s": round(exp),
                   "remaining_s": round(stage_remaining), "overrun": overrun}
        later = sum(expected[s] for s in STAGES[idx + 1:]) * speed
        remaining = stage_remaining + later
        total_expected = sum(expected[s] for s in STAGES[:idx]) + sum(expected[s] for s in STAGES[idx:])
        done_part = done_expected + expected[name] * min(0.99, stage_elapsed / exp)
        overall_percent = round(100.0 * done_part / total_expected, 1)
    uncertain = bool(estimate is not None and (not estimate.get("duration_s") or done < 2))
    if estimate is not None and estimate.get("device") != "cpu":
        uncertain = uncertain or any(s in estimate.get("fallback_stages", []) for s in STAGES[done:])
    uncertain = uncertain or bool(current and current["overrun"])
    if current and (uncertain or status in ("failed", "cancelled")):
        current["remaining_s"] = None

    return {
        "schema": 1,
        "run_status": "completed" if finished else status,
        "stages": stages,
        "stage_done": done,
        "stage_total": total,
        "current": current,
        "overall": {"percent": overall_percent, "elapsed_s": round(elapsed),
                    "remaining_s": None if status in ("failed", "cancelled") or (uncertain and not finished) else round(remaining),
                    "eta_unix_s": None if finished or status in ("failed", "cancelled") or uncertain else round(now + remaining)},
        "estimate": {**(estimate or {"basis": "baseline stage durations scaled by observed speed"}), "speed_factor": round(speed, 2),
                     "stage_percent_is_time_based": True},
        "updated_at": progress.get("updated_at"),
        "now_unix_s": round(now),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--started-at", default=None)
    ap.add_argument("--state", choices=["failed", "cancelled", "completed"], default=None)
    args = ap.parse_args()
    run = Path(args.run_dir)
    try:
        progress = json.loads((run / "work" / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        progress = None
    if args.started_at:
        started = parse_time(args.started_at)
    else:
        marker = run / "launch-options.json"
        started = marker.stat().st_mtime if marker.exists() else time.time()
    config = read_json(run / "job-config.json") or read_json(run / "launch-options.json")
    expected, estimate = historical_baseline(run, started, config=config)
    if args.baseline:
        expected = baseline_from_performance(args.baseline)
        estimate["basis"] = "explicit performance baseline (unscaled)"
    snap = build_snapshot(progress, expected, started, time.time(), args.state, estimate)
    sys.stdout.write(json.dumps(snap, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
