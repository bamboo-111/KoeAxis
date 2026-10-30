"""Explicit transcript-driven full alignment; no baseline manifests as execution inputs."""
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def load_rows(path):
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("transcript must be a nonempty list")
    ids = set()
    for row in rows:
        if not isinstance(row, dict) or not all(k in row for k in (
                "segment_id", "audio_path", "global_start_time", "global_end_time", "text", "status")):
            raise ValueError("invalid transcript row")
        ident = row["segment_id"]
        if not isinstance(ident, str) or not ident or ident in ids:
            raise ValueError("missing or duplicate segment id")
        ids.add(ident)
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

@contextmanager
def timed(records, phase, **metadata):
    started = time.perf_counter()
    status = 'error'
    try:
        yield
        status = 'ok'
    finally:
        records.append({'phase': phase, 'elapsed_ms': (time.perf_counter() - started) * 1000,
                        'status': status, **metadata})


def align_rows(rows, initial, recover, mfa):
    """All initial results are generated first, then failures use only these neighbors."""
    aligned = [initial(row) for row in rows]
    if len(aligned) != len(rows) or any(a.get("segment_id") != r["segment_id"] for r, a in zip(rows, aligned)):
        raise ValueError("initial alignment identity mismatch")
    initial_rows = list(aligned)
    branches = []
    for index, row in enumerate(rows):
        if aligned[index].get("status") != "failed":
            continue
        short = recover(row, initial_rows[index], initial_rows)
        result = short["aligned_row"]
        mfa_report = None
        if short["enter_mfa"]:
            mfa_report = mfa(row, short)
            result = mfa_report["aligned_row"]
        if result.get("segment_id") != row["segment_id"]:
            raise ValueError("recovered alignment identity mismatch")
        aligned[index] = result
        branches.append({"segment_id": row["segment_id"], "short_window": short,
                         "mfa": mfa_report})
    return aligned, branches


def protect_legacy(root):
    root = Path(root).resolve()
    def inside(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        try:
            return Path(os.fsdecode(value)).resolve().is_relative_to(root)
        except (OSError, ValueError):
            return False
    def hook(event, args):
        paths = []
        if event == "open" and args:
            mode = args[1] if len(args) > 1 else ""
            flags = args[2] if len(args) > 2 else 0
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                    isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                paths = [args[0]]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"} and args:
            paths = [args[0]]
        elif event in {"os.rename", "os.link", "os.replace"} and len(args) >= 2:
            paths = list(args[:2])
        elif event == "os.symlink" and len(args) >= 2:
            paths = [args[1]]
        if any(inside(p) for p in paths):
            raise PermissionError("legacy write forbidden: " + event)
    sys.addaudithook(hook)


def real(recover, rows, work, source_audio, api_mfa):
    recover._legacy()
    from qwen_asr.commands import stages
    from qwen_asr.models import TranscriptSegment, dataclass_to_dict
    work = Path(work)
    recover.assert_outside(work, "chain work")
    work.mkdir(parents=True, exist_ok=True)
    if not Path(source_audio).is_file():
        raise ValueError("source audio missing")
    if any(not Path(r["audio_path"]).is_file() for r in rows):
        raise ValueError("explicit segment audio missing")
    before = recover.legacy_snapshot()
    old_env = recover._offline_env(work)
    old_cwd, old_progress = Path.cwd(), recover.PROGRESS
    recover.PROGRESS = work / "progress.json"
    protect_legacy(recover.LEGACY)
    timings = []
    try:
        os.chdir(work)
        model = stages.QwenForcedAligner(
            model_name=str(recover.ALIGN_SNAP), dtype="fp16", device="cuda", attn_implementation=None,
            keep_raw_model_output=False, keep_failed_tokens=False, model_cache_dir=str(recover.CACHE),
            local_files_only=True, timing_validation_config=stages.AlignTimingValidationConfig(
                min_coverage_ratio=0.2, bad_zero_run=8, dense_zero_ratio=0.5,
                min_dense_coverage_ratio=0.5, local_collapse_min_chars=8,
                local_collapse_max_duration=0.5, local_collapse_max_cps=35.0,
                local_collapse_max_tokens=12))
        initial = []
        try:
            with timed(timings, 'align_load'):
                model.load()
            with timed(timings, 'align_inference_and_loop'):
                for i, row in enumerate(rows):
                    if row["status"] != "completed" or not row["text"].strip():
                        raise ValueError("unusable transcript: " + row["segment_id"])
                    item = TranscriptSegment(segment_id=row["segment_id"], audio_path=row["audio_path"],
                        global_start_time=row["global_start_time"], global_end_time=row["global_end_time"],
                        text=row["text"], language=row.get("language") or "Japanese", status=row["status"])
                    initial.append(dataclass_to_dict(model.run_segment(item, cleanup=(i + 1) % 4 == 0)))
                    if (i + 1) % 25 == 0 or i + 1 == len(rows):
                        recover.progress("initial-alignment", done=i + 1, total=len(rows))
        finally:
            with timed(timings, 'align_release'):
                recover._close_model(model)
        recover._dump(work / "initial_aligned_segments.json", initial)
        initial_iter = iter(initial)
        def short(row, initial_row, neighbors):
            segment_work = work / "recovery" / row["segment_id"]
            segment_work.mkdir(parents=True, exist_ok=True)
            with timed(timings, 'short_window_total', segment_id=row['segment_id']):
                return recover.run_short_window(segment_work, row, None, initial_row, neighbors, source_audio)
        def mfa(row, short_report):
            with timed(timings, 'mfa_call_total', segment_id=row['segment_id']):
                return api_mfa(short_report["work_dir"], row["segment_id"], source_audio)
        aligned, branches = align_rows(rows, lambda row: next(initial_iter), short, mfa)
        recover._dump(work / "aligned_segments.json", aligned)
        recover._dump(work / "branches.json", branches)
    finally:
        try:
            recover._dump(work / "alignment_timing.json", {'schema': 'alignment-timing-v1', 'clock': 'perf_counter', 'records': timings, 'short_window_breakdown': 'unknown', 'mfa_breakdown': 'unknown'})
        except OSError:
            pass  # Diagnostics must not mask the original business outcome.
        os.chdir(old_cwd)
        recover._restore_env(old_env)
        recover.PROGRESS = old_progress
    after = recover.legacy_snapshot()
    changed = recover.snapshot_diff(before, after)
    if changed:
        raise RuntimeError("legacy snapshot changed: " + str(changed))
    return {"count": len(aligned), "initial_failed": sum(r.get("status") == "failed" for r in initial),
            "failed": sum(r.get("status") != "completed" for r in aligned), "branches": branches,
            "legacy_before": before, "legacy_after": after, "legacy_changes": changed,
            "aligned_path": str(work / "aligned_segments.json"), "aligned_sha256": digest(work / "aligned_segments.json")}
