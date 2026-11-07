"""Bind the existing short-window and MFA functions. Do not copy a second driver.

Writes stay under the caller work directory. The legacy repository is read only.
"""
import hashlib
import math
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

LEGACY = Path(__file__).resolve().parents[3] / "compat" / "legacy"
WS = LEGACY / "workspaces/radio-54-togeradi-20261018"
CACHE = Path(__file__).resolve().parents[3] / "models"
ASR_SNAP = CACHE / "models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5"
ALIGN_SNAP = CACHE / "models--Qwen--Qwen3-ForcedAligner-0.6B/snapshots/c7cbfc2048c462b0d63a45797104fc9db3ad62b7"
MFA_EXE = LEGACY / "tools/mfa-env/Scripts/mfa.exe"
MFA_DICT = LEGACY / "tools/mfa-root/pretrained_models/dictionary/japanese_mfa.dict"
MFA_ACOUSTIC = LEGACY / "tools/mfa-root/pretrained_models/acoustic/japanese_mfa.zip"
SOURCE_WAV = WS / "audio/source.wav"
SEGMENT_WAV = WS / "audio/segments/segment_000041.wav"
TRANSCRIPT = WS / "manifests/transcript_segments.json"
RAW_TRANSCRIPT = WS / "manifests/transcript_segments.raw.json"
ALIGNED = WS / "manifests/aligned_segments.json"
BASELINE = Path(__file__).resolve().parents[3] / "runs/experiments/REAL-ASR-001/repair-1/segment-000041-baseline.json"
SEGMENT_ID = "segment_000041"
EXPECTED_TEXT = "interfm.jpトゲなしトゲありのトゲラジ川崎から世界へ。"
PROGRESS = Path(__file__).resolve().parents[3] / "runs/experiments/REAL-ALIGN-RECOVERY-001/progress.json"
WATCH = (
    LEGACY / "tools/mfa-root",
    LEGACY / "tools/mfa-env",
    LEGACY / ".model-cache",
    WS / "diagnostics",
    WS / "manifests",
    WS / "audio",
)


def configure_resources(resources):
    """Override only this invocation's locally loaded recovery module."""
    global CACHE, ASR_SNAP, ALIGN_SNAP, MFA_EXE, MFA_DICT, MFA_ACOUSTIC
    names = {'cache': ('CACHE', True), 'asr_model': ('ASR_SNAP', True),
             'align_model': ('ALIGN_SNAP', True), 'mfa_exe': ('MFA_EXE', False),
             'mfa_dictionary': ('MFA_DICT', False), 'mfa_acoustic': ('MFA_ACOUSTIC', False)}
    if not isinstance(resources, dict) or set(resources) != set(names):
        raise ValueError('all explicit alignment resources are required')
    values = {}
    for key, (name, directory) in names.items():
        value = resources[key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('alignment resource must be absolute: ' + key)
        path = Path(value).resolve()
        if not (path.is_dir() if directory else path.is_file()):
            raise ValueError('alignment resource missing: ' + key)
        values[name] = path
    CACHE, ASR_SNAP, ALIGN_SNAP = values['CACHE'], values['ASR_SNAP'], values['ALIGN_SNAP']
    MFA_EXE, MFA_DICT, MFA_ACOUSTIC = values['MFA_EXE'], values['MFA_DICT'], values['MFA_ACOUSTIC']


def outside_legacy(path):
    resolved = Path(path).resolve()
    legacy = LEGACY.resolve()
    return resolved != legacy and legacy not in resolved.parents


def assert_outside(path, label):
    if not outside_legacy(path):
        raise RuntimeError(label + " is inside the legacy repository: " + str(path))


def progress(step, **extra):
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"steps": []}
    if PROGRESS.is_file():
        try:
            payload = json.loads(PROGRESS.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"steps": []}
    if not isinstance(payload, dict):
        payload = {"steps": []}
    steps = payload.get("steps")
    if not isinstance(steps, list):
        steps = []
    steps.append({"step": step, **extra})
    payload["steps"] = steps
    payload["last_step"] = step
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, PROGRESS)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def segment_row(path, segment_id):
    for item in _load_json(path):
        if isinstance(item, dict) and item.get("segment_id") == segment_id:
            return item
    return None


def fingerprint(root):
    root = Path(root)
    rows = []
    if not root.exists():
        return {"exists": False, "count": 0, "digest": None}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            path = Path(dirpath) / name
            try:
                stat = path.stat()
            except OSError as exc:
                rows.append(str(path.relative_to(root)) + "|err|" + type(exc).__name__)
                continue
            rows.append("%s|%s|%s" % (path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return {"exists": True, "count": len(rows), "digest": digest}


def git_status():
    completed = subprocess.run(
        ["git", "status", "--short"], cwd=LEGACY, capture_output=True, text=True, check=False)
    text = completed.stdout.replace("\r\n", "\n")
    if not text.endswith("\n") and text:
        text += "\n"
    lines = [line for line in text.split("\n") if line]
    return {"returncode": completed.returncode, "lines": len(lines),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def legacy_snapshot():
    return {"git": git_status(), "trees": {str(path): fingerprint(path) for path in WATCH}}


def snapshot_diff(before, after):
    changed = []
    if before["git"] != after["git"]:
        changed.append({"tree": "git-status", "before": before["git"], "after": after["git"]})
    for key, left in before["trees"].items():
        right = after["trees"].get(key)
        if left != right:
            changed.append({"tree": key, "before": left, "after": right})
    return changed


def _trim(value, limit=500):
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[-limit:]


def plain(value):
    """Host json_bytes accepts only None, str, bool, int, finite float, list, and dict."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else None
    if type(value) is dict:
        return {str(key): plain(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [plain(item) for item in value]
    module = type(value).__module__
    if type(module) is str and module.startswith("numpy") and hasattr(value, "item"):
        try:
            return plain(value.item())
        except Exception:
            return str(value)
    return str(value)


def _public_mfa(attempt):
    if not isinstance(attempt, dict):
        return {"raw_type": type(attempt).__name__}
    evidence = attempt.get("backend_evidence") if isinstance(attempt.get("backend_evidence"), dict) else {}
    return {
        "status": attempt.get("status"),
        "alignment_state": attempt.get("alignment_state"),
        "error_code": attempt.get("error_code"),
        "error": attempt.get("error"),
        "elapsed_ms": attempt.get("elapsed_ms"),
        "writeback": attempt.get("writeback"),
        "text_source": attempt.get("text_source"),
        "selected_text": attempt.get("selected_text"),
        "original_text": attempt.get("original_text"),
        "language_route": attempt.get("language_route"),
        "strategy": attempt.get("strategy"),
        "recognized_text": evidence.get("recognized_text"),
        "content_score": evidence.get("content_score"),
        "reason": evidence.get("reason"),
        "stderr_tail": _trim(str(evidence.get("stderr_tail") or evidence.get("stderr") or "")),
    }


def _tokens(value):
    rows = []
    for token in value or []:
        if isinstance(token, dict):
            rows.append({"text": token.get("text"), "start_time": token.get("start_time"),
                         "end_time": token.get("end_time")})
    return rows


def compare_result(actual_tokens, actual_fields, baseline):
    expected = baseline["tokens"]
    diffs = []
    for key in ("text", "status", "alignment_backend", "alignment_state"):
        if actual_fields.get(key) != baseline.get(key):
            diffs.append({"field": key, "actual": actual_fields.get(key), "expected": baseline.get(key)})
    if len(actual_tokens) != len(expected):
        diffs.append({"field": "token_count", "actual": len(actual_tokens), "expected": len(expected)})
    for index, (got, want) in enumerate(zip(actual_tokens, expected)):
        for key in ("text", "start_time", "end_time"):
            if got.get(key) != want.get(key):
                diffs.append({"field": "tokens[%d].%s" % (index, key),
                              "actual": got.get(key), "expected": want.get(key)})
    return diffs


def _close_model(model):
    try:
        model.close()
    except Exception:
        pass
    import gc
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _offline_env(work):
    home = work / "home"
    tmp = work / "tmp"
    for path in (home, tmp, work / "hf-home", work / "numba-cache", work / "mpl-cache", work / "torch-home"):
        assert_outside(path, "writable dir")
        path.mkdir(parents=True, exist_ok=True)
    previous = {key: os.environ.get(key) for key in (
        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "NUMBA_CACHE_DIR", "MPLCONFIGDIR",
        "TORCH_HOME", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES", "HOME", "USERPROFILE",
        "TEMP", "TMP", "MFA_ROOT_DIR")}
    updates = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HOME": str(work / "hf-home"),
        "HF_HUB_CACHE": str(CACHE),
        "HUGGINGFACE_HUB_CACHE": str(CACHE),
        "NUMBA_CACHE_DIR": str(work / "numba-cache"),
        "MPLCONFIGDIR": str(work / "mpl-cache"),
        "TORCH_HOME": str(work / "torch-home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "HOME": str(home),
        "USERPROFILE": str(home),
        "TEMP": str(tmp),
        "TMP": str(tmp),
    }
    for key, value in updates.items():
        assert_outside(value, key) if key in {"HF_HOME", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "TORCH_HOME", "HOME", "USERPROFILE", "TEMP", "TMP"} else None
        os.environ[key] = value
    sys.dont_write_bytecode = True
    return previous


def _restore_env(previous):
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def copy_mfa_root(work):
    root = work / "mfa-root"
    assert_outside(root, "mfa root")
    for source, relative in (
        (MFA_DICT, "pretrained_models/dictionary/japanese_mfa.dict"),
        (MFA_ACOUSTIC, "pretrained_models/acoustic/japanese_mfa.zip"),
    ):
        if not source.is_file():
            raise RuntimeError("missing read-only MFA resource " + str(source))
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        assert_outside(target, "copied MFA resource")
    return root


def _legacy():
    root = str(LEGACY)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_short_window(work, transcript_row=None, raw_row=None, initial_row=None, aligned_rows=None, source_audio=None):
    """One model at a time. Reuses the legacy window export, ASR, aligner and content guard."""
    _legacy()
    from qwen_asr.commands import stages as stage_helpers
    from qwen_asr.commands.align import (
        _export_align_fallback_windows,
        _validate_fallback_alignment_content,
    )
    from qwen_asr.models import TranscriptSegment, dataclass_to_dict

    legacy_default = transcript_row is None
    if legacy_default:
        transcript_row = segment_row(TRANSCRIPT, SEGMENT_ID)
        raw_row = segment_row(RAW_TRANSCRIPT, SEGMENT_ID) if RAW_TRANSCRIPT.is_file() else None
    if transcript_row is None:
        raise RuntimeError("transcript segment missing")
    if legacy_default and transcript_row.get("text") != EXPECTED_TEXT:
        raise RuntimeError("transcript text is not the fixed B segment")
    segment_id = transcript_row["segment_id"]
    segment_wav = Path(transcript_row["audio_path"])
    transcript = TranscriptSegment(
        segment_id=segment_id,
        audio_path=str(segment_wav),
        global_start_time=float(transcript_row["global_start_time"]),
        global_end_time=float(transcript_row["global_end_time"]),
        text=transcript_row["text"],
        language=transcript_row.get("language") or "Japanese",
        status="completed",
    )
    if initial_row is None:
        progress("align-initial-load")
        aligner = stage_helpers.QwenForcedAligner(
            model_name=str(ALIGN_SNAP), dtype="fp16", device="cuda", attn_implementation=None,
            keep_raw_model_output=False, keep_failed_tokens=False, model_cache_dir=str(CACHE),
            local_files_only=True,
            timing_validation_config=stage_helpers.AlignTimingValidationConfig(
                min_coverage_ratio=0.2, bad_zero_run=8, dense_zero_ratio=0.5,
                min_dense_coverage_ratio=0.5, local_collapse_min_chars=8,
                local_collapse_max_duration=0.5, local_collapse_max_cps=35.0,
                local_collapse_max_tokens=12))
        aligner.load()
        try:
            initial = aligner.run_segment(transcript, cleanup=False)
            initial_dict = dataclass_to_dict(initial)
        finally:
            _close_model(aligner)
    else:
        initial_dict = initial_row
    progress("align-initial-done", status=initial_dict.get("status"), error=initial_dict.get("error"))
    report = {
        "input": "baseline transcript segment_000041" if legacy_default else "explicit transcript segment",
        "transcript_text": transcript_row.get("text"),
        "raw_text": None if raw_row is None else raw_row.get("text"),
        "raw_differs_from_transcript": None if raw_row is None else raw_row.get("text") != transcript_row.get("text"),
        "branch": "initial-align",
        "initial": {
            "status": initial_dict.get("status"),
            "error": initial_dict.get("error"),
            "alignment_state": initial_dict.get("alignment_state"),
            "token_count": len(initial_dict.get("tokens") or []),
        },
        "models": ["aligner"],
        "one_model_at_a_time": True,
    }
    if initial_dict.get("status") != "failed":
        report["enter_short_window"] = False
        report["enter_mfa"] = False
        report["mfa_skipped_reason"] = "initial align did not fail; original rule does not enter short-window"
        report["aligned"] = {
            "text": initial_dict.get("text"),
            "status": initial_dict.get("status"),
            "alignment_backend": initial_dict.get("alignment_backend"),
            "alignment_state": initial_dict.get("alignment_state"),
            "tokens": _tokens(initial_dict.get("tokens")),
            "error": initial_dict.get("error"),
        }
        return plain(report)

    progress("short-window-export")
    output_dir = work / "diagnostics" / "align-fallback" / "asr-short-window" / segment_id
    assert_outside(output_dir, "short-window output")
    windows = _export_align_fallback_windows(segment_wav, transcript, output_dir, 3.0)
    asr_rows = []
    progress("short-window-asr-load")
    # Effective baseline arg is --language Japanese, not the constructor default None.
    transcriber = stage_helpers.QwenASRTranscriber(
        model_name=str(ASR_SNAP),
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        max_new_tokens=512,
        language=transcript.language,
        keep_raw_model_output=False,
        model_cache_dir=str(CACHE),
        local_files_only=True,
        batch_size=1,
    )
    transcriber.load()
    try:
        for window in windows:
            asr_result = transcriber.run_segment(window, cleanup=False)
            asr_rows.append({
                "window_segment_id": window.segment_id,
                "global_start_time": window.global_start_time,
                "global_end_time": window.global_end_time,
                "asr_status": asr_result.status,
                "asr_text": asr_result.text,
                "asr_error": asr_result.error,
                "audio_path": window.audio_path,
                "language": asr_result.language or transcript.language,
            })
    finally:
        _close_model(transcriber)
    progress("short-window-align-load", windows=len(asr_rows))
    aligner = stage_helpers.QwenForcedAligner(
        model_name=str(ALIGN_SNAP),
        dtype="fp16",
        device="cuda",
        attn_implementation=None,
        keep_raw_model_output=False,
        keep_failed_tokens=False,
        model_cache_dir=str(CACHE),
        local_files_only=True,
        timing_validation_config=stage_helpers.AlignTimingValidationConfig(
            min_coverage_ratio=0.2, bad_zero_run=8, dense_zero_ratio=0.5,
            min_dense_coverage_ratio=0.5, local_collapse_min_chars=8,
            local_collapse_max_duration=0.5, local_collapse_max_cps=35.0,
            local_collapse_max_tokens=12,
        ),
    )
    aligner.load()
    completed = []
    try:
        for row in asr_rows:
            if row["asr_status"] != "completed" or not str(row["asr_text"] or "").strip():
                row["align_status"] = "skipped"
                row["align_error"] = "empty ASR reference text"
                row["token_count"] = 0
                continue
            window_transcript = TranscriptSegment(
                segment_id=row["window_segment_id"],
                audio_path=row["audio_path"],
                global_start_time=row["global_start_time"],
                global_end_time=row["global_end_time"],
                text=row["asr_text"],
                language=row["language"],
                status="completed",
            )
            alignment = aligner.run_segment(window_transcript, cleanup=False)
            aligned = dataclass_to_dict(alignment)
            row["align_status"] = aligned.get("status")
            row["align_error"] = aligned.get("error")
            row["token_count"] = len(aligned.get("tokens") or [])
            if aligned.get("status") == "completed" and aligned.get("tokens"):
                completed.extend(aligned["tokens"])
    finally:
        _close_model(aligner)
    tokens = sorted(completed, key=lambda token: (token["start_time"], token["end_time"]))
    from qwen_asr.models import AlignedToken
    content_error = _validate_fallback_alignment_content(
        transcript.text,
        [AlignedToken(text=token["text"], start_time=token["start_time"], end_time=token["end_time"])
         for token in tokens],
    )
    merged = "".join(str(row.get("asr_text") or "").strip() for row in asr_rows)
    token_objects = [AlignedToken(**token) for token in tokens]
    timing_error = stage_helpers.validate_aligned_token_timing(
        token_objects, transcript.global_start_time, transcript.global_end_time)
    failed = (not tokens) or bool(content_error) or bool(timing_error)
    report["enter_short_window"] = True
    report["models"] = ["aligner", "asr", "aligner"]
    report["short_window"] = {
        "fallback": "asr-short-window",
        "original_error": initial_dict.get("error"),
        "window_seconds": 3.0,
        "window_count": len(windows),
        "completed_window_alignments": sum(1 for row in asr_rows if row.get("align_status") == "completed" and row.get("token_count")),
        "merged_asr_text": merged,
        "content_error": content_error,
        "windows": [{key: row[key] for key in (
            "window_segment_id", "global_start_time", "global_end_time", "asr_status",
            "asr_text", "align_status", "align_error", "token_count") if key in row} for row in asr_rows],
        "branch_note": "sequential one-model; original helper keeps aligner and ASR resident together",
    }
    report["enter_mfa"] = bool(failed)
    report["mfa_skipped_reason"] = None if failed else "short-window content guard passed; original rule accepts it"
    if not legacy_default:
        from qwen_asr.models import AlignedSegment
        from qwen_asr.commands.align import _merge_fallback_metadata
        metadata = {k: v for k, v in report["short_window"].items() if k != "branch_note"}
        metadata["windows"] = [{k: v for k, v in row.items() if k != "language"} for row in asr_rows]
        error = content_error or timing_error or initial_dict.get("error") or "ASR short-window fallback produced no aligned tokens"
        full = AlignedSegment(
            segment_id=segment_id, audio_path=str(segment_wav),
            global_start_time=transcript.global_start_time, global_end_time=transcript.global_end_time,
            text=transcript.text, language=transcript.language,
            tokens=[AlignedToken(**item) for item in (initial_dict.get("tokens") or [])] if failed else token_objects,
            raw_model_output=_merge_fallback_metadata(initial_dict.get("raw_model_output"), metadata),
            status="failed" if failed else "completed", error=error if failed else None)
        full_row = dataclass_to_dict(full)
        report["aligned_row"] = full_row
    if failed:
        error = content_error or timing_error or initial_dict.get("error") or "ASR short-window fallback produced no aligned tokens"
        failed_row = full_row if not legacy_default else {
            "segment_id": segment_id,
            "audio_path": str(segment_wav),
            "global_start_time": transcript.global_start_time,
            "global_end_time": transcript.global_end_time,
            "text": transcript.text,
            "language": transcript.language,
            "tokens": initial_dict.get("tokens") or [],
            "status": "failed",
            "error": error,
            "alignment_backend": "qwen",
            "alignment_unit": "token",
            "alignment_state": "failed",
            "alignment_failure_reason": error,
            "raw_model_output": {"align_fallback": report["short_window"]},
        }
        _write_recovery_inputs(work, transcript_row, failed_row, aligned_rows, source_audio)
        report["work_dir"] = str(work)
        report["aligned"] = {
            "text": transcript.text,
            "status": "failed",
            "alignment_backend": "qwen",
            "alignment_state": "failed",
            "tokens": _tokens(initial_dict.get("tokens")),
            "error": error,
        }
    else:
        report["aligned"] = {
            "text": transcript.text,
            "status": "completed",
            "alignment_backend": "qwen",
            "alignment_state": "completed_exact",
            "tokens": _tokens(tokens),
            "error": None,
        }
    progress("short-window-done", content_error=content_error, enter_mfa=report["enter_mfa"])
    return plain(report)


def _write_recovery_inputs(work, transcript_row, failed_row, aligned_rows=None, source_audio=None):
    _legacy()
    from qwen_asr.models import WorkPaths
    paths = WorkPaths.from_workdir(work)
    paths = replace(paths, audio_path=Path(source_audio) if source_audio else SOURCE_WAV)
    assert_outside(work, "recovery workdir")
    paths.transcript_manifest.parent.mkdir(parents=True, exist_ok=True)
    neighbors = []
    if aligned_rows is None:
        aligned_rows = _load_json(ALIGNED)
    segment_id = failed_row["segment_id"]
    index = next(i for i, row in enumerate(aligned_rows) if row.get("segment_id") == segment_id)
    if index > 0:
        neighbors.append(aligned_rows[index - 1])
    neighbors.append(failed_row)
    if index + 1 < len(aligned_rows):
        neighbors.append(aligned_rows[index + 1])
    _dump(paths.transcript_manifest, [transcript_row])
    _dump(paths.aligned_manifest, neighbors)
    return str(work)


def _dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8", newline="\n")


def mfa_process_env(work, mfa_root):
    """Redirect MFA and its interpreter caches out of the legacy repository."""
    work = Path(work)
    mfa_root = Path(mfa_root)
    dirs = {
        "HOME": work / "home",
        "USERPROFILE": work / "home",
        "TEMP": work / "tmp",
        "TMP": work / "tmp",
        "APPDATA": work / "appdata",
        "LOCALAPPDATA": work / "localappdata",
        "XDG_CACHE_HOME": work / "xdg-cache",
        "XDG_CONFIG_HOME": work / "xdg-config",
        "XDG_DATA_HOME": work / "xdg-data",
        "PYTHONPYCACHEPREFIX": work / "pyc",
    }
    for path in list(dirs.values()) + [mfa_root]:
        assert_outside(path, "mfa writable path")
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MFA_ROOT_DIR"] = str(mfa_root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key, path in dirs.items():
        env[key] = str(path)
    prefixes = []
    env_root = MFA_EXE.parent.parent
    for relative in ("Library/bin", "Library/usr/bin", "Library/mingw-w64/bin", "Scripts", "bin", "."):
        candidate = env_root / relative
        if candidate.is_dir():
            prefixes.append(str(candidate))
    current = env.get("PATH", "")
    env["PATH"] = os.pathsep.join(prefixes + ([current] if current else []))
    return env


def assert_mfa_version_clean(work, mfa_root):
    before = legacy_snapshot()
    completed = subprocess.run(
        [str(MFA_EXE), "version"], cwd=str(work), env=mfa_process_env(work, mfa_root),
        capture_output=True, text=True, timeout=60, check=False)
    changed = snapshot_diff(before, legacy_snapshot())
    progress("mfa-version-preflight", returncode=completed.returncode, changed=len(changed))
    if changed:
        raise RuntimeError("mfa version wrote inside the legacy repository")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "")[-300:]
        raise RuntimeError("mfa version failed: " + tail)


def run_mfa(work, segment_id=SEGMENT_ID, source_audio=SOURCE_WAV, include_row=False):
    """Call execute_alignment_recovery(strategy=mfa-local) with an injected root."""
    _legacy()
    import qwen_asr.mfa_environment as mfa_env
    import qwen_asr.mfa_runner as mfa_runner
    from qwen_asr.models import WorkPaths
    from qwen_asr.recovery_executor import execute_alignment_recovery

    work = Path(work)
    mfa_root = copy_mfa_root(work)
    home = work / "home"
    tmp = work / "tmp"
    for path in (work, mfa_root, home, tmp):
        assert_outside(path, "mfa writable path")
    if not MFA_EXE.is_file():
        raise RuntimeError("mfa.exe missing")
    assert_mfa_version_clean(work, mfa_root)
    paths = replace(WorkPaths.from_workdir(work), audio_path=Path(source_audio))
    original_root = mfa_env.find_project_mfa_root
    original_command = mfa_env.build_project_mfa_command
    original_env_run = mfa_env.subprocess.run
    original_runner_run = mfa_runner.subprocess.run

    def patched_root(project_root=None):
        return mfa_root

    def patched_command(project_root=None):
        return [str(MFA_EXE)], str(MFA_EXE), "direct"

    def guarded_run(command, **kwargs):
        passed = kwargs.get("env")
        env = dict(passed) if isinstance(passed, dict) else mfa_process_env(work, mfa_root)
        redirect = mfa_process_env(work, mfa_root)
        for key in (
            "MFA_ROOT_DIR", "HOME", "USERPROFILE", "TEMP", "TMP", "APPDATA", "LOCALAPPDATA",
            "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
            "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX", "PATH"):
            env[key] = redirect[key]
        kwargs["env"] = env
        kwargs["cwd"] = str(work)
        kwargs.setdefault("timeout", 600)
        completed = original_runner_run(command, **kwargs)
        if getattr(completed, "returncode", 0) not in (0, None):
            progress("mfa-subprocess-failed", returncode=completed.returncode,
                     stderr_tail=_trim(getattr(completed, "stderr", "") or ""),
                     stdout_tail=_trim(getattr(completed, "stdout", "") or ""))
        return completed

    def guarded_mfa_runner(work_paths, candidates, *, environment, max_run_candidates, padding_ms):
        injected = dict(environment)
        injected["available"] = True
        injected["command"] = [str(MFA_EXE)]
        injected["executable"] = str(MFA_EXE)
        injected["invocation"] = "direct"
        injected["root_dir"] = str(mfa_root)
        return mfa_runner.run_local_mfa_alignment_experiments(
            work_paths, candidates, environment=injected,
            max_run_candidates=max_run_candidates, padding_ms=padding_ms,
            run_command=guarded_run, environ_factory=lambda: os.environ.copy())

    configured = {
        "mfa_root": str(mfa_root),
        "mfa_exe_read_only": str(MFA_EXE),
        "cwd": str(work),
        "home": str(home),
        "temp": str(tmp),
        "source_audio_read_only": str(source_audio),
        "all_writable_outside_legacy": True,
    }
    progress("mfa-start", root=str(mfa_root))
    previous_cwd = Path.cwd()
    mfa_env.find_project_mfa_root = patched_root
    mfa_env.build_project_mfa_command = patched_command
    mfa_env.subprocess.run = guarded_run
    os.chdir(work)
    try:
        attempt = execute_alignment_recovery(
            paths, segment_id=segment_id, strategy="mfa-local",
            language_route="japanese", actor="real-align-recovery",
            settings={"mfa_padding_ms": 700, "mfa_min_content_score": 0.70, "local_files_only": True},
            mfa_runner=guarded_mfa_runner)
    finally:
        mfa_env.find_project_mfa_root = original_root
        mfa_env.build_project_mfa_command = original_command
        mfa_env.subprocess.run = original_env_run
        os.chdir(previous_cwd)
    aligned = segment_row(paths.aligned_manifest, segment_id) or {}
    public = _public_mfa(attempt)
    public["configured_paths"] = configured
    public["tokens"] = _tokens(aligned.get("tokens"))
    public["alignment_backend"] = aligned.get("alignment_backend")
    public["alignment_state"] = aligned.get("alignment_state") or public.get("alignment_state")
    public["text"] = aligned.get("text")
    public["status_field"] = aligned.get("status")
    if include_row:
        public["aligned_row"] = aligned
    public["lab_text"] = None
    evidence = attempt.get("backend_evidence") if isinstance(attempt, dict) else None
    if isinstance(evidence, dict):
        lab_text = evidence.get("lab_text")
        if not lab_text and isinstance(evidence.get("lab"), str):
            lab_path = Path(evidence["lab"])
            if lab_path.is_file() and outside_legacy(lab_path):
                lab_text = lab_path.read_text(encoding="utf-8").strip()
        public["lab_text"] = lab_text or None
    progress("mfa-done", status=public.get("status"), alignment_state=public.get("alignment_state"))
    return plain(public)


def run_real(work):
    work = Path(work)
    assert_outside(work, "real workdir")
    work.mkdir(parents=True, exist_ok=True)
    before = legacy_snapshot()
    previous_env = _offline_env(work)
    previous_cwd = Path.cwd()
    os.chdir(work)
    error = None
    short = None
    mfa = None
    try:
        progress("legacy-snapshot-before", git_lines=before["git"]["lines"])
        short = run_short_window(work)
        if short.get("enter_mfa"):
            mfa = run_mfa(work)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()[-4000:]}
        progress("real-exception", error_type=error["type"])
    finally:
        os.chdir(previous_cwd)
        _restore_env(previous_env)
    after = legacy_snapshot()
    changed = snapshot_diff(before, after)
    progress("legacy-snapshot-after", changed=len(changed), git_lines=after["git"]["lines"])
    baseline = _load_json(BASELINE)
    aligned = (mfa if mfa and mfa.get("tokens") else None) or (short or {}).get("aligned") or {}
    tokens = aligned.get("tokens") or []
    fields = {
        "text": aligned.get("text") or (short or {}).get("transcript_text"),
        "status": aligned.get("status_field") or aligned.get("status"),
        "alignment_backend": aligned.get("alignment_backend"),
        "alignment_state": aligned.get("alignment_state"),
    }
    if mfa and mfa.get("alignment_state") == "completed_exact":
        fields["status"] = "completed"
        fields["alignment_backend"] = mfa.get("alignment_backend") or "mfa-local-recovery"
        fields["alignment_state"] = "completed_exact"
        fields["text"] = mfa.get("text") or fields["text"]
    diffs = compare_result(tokens, fields, baseline) if short else [{"field": "run", "actual": "exception", "expected": "completed"}]
    short_window = (short or {}).get("short_window") or {}
    if short_window.get("content_error") != (baseline.get("short_window") or {}).get("content_error"):
        diffs.append({"field": "short_window.content_error",
                      "actual": short_window.get("content_error"),
                      "expected": (baseline.get("short_window") or {}).get("content_error")})
    if short_window.get("merged_asr_text") != (baseline.get("short_window") or {}).get("merged_asr_text"):
        diffs.append({"field": "short_window.merged_asr_text",
                      "actual": short_window.get("merged_asr_text"),
                      "expected": (baseline.get("short_window") or {}).get("merged_asr_text")})
    recovery = baseline.get("recovery") or {}
    if mfa:
        for key in ("recognized_text", "content_score", "lab_text"):
            if mfa.get(key) != recovery.get(key):
                diffs.append({"field": "recovery." + key, "actual": mfa.get(key),
                              "expected": recovery.get(key)})
    verdict = "blocked" if error or changed or not short else ("identical" if not diffs else "differs")
    return {
        "models_run": short is not None,
        "mfa_executed": mfa is not None,
        "legacy_clean": not changed,
        "legacy_changes": changed,
        "git_after": after["git"],
        "error": error,
        "short_window_report": short,
        "mfa_report": mfa,
        "comparison_fields": fields,
        "token_count": len(tokens),
        "tokens": tokens,
        "diffs": diffs,
        "segment_verdict": verdict,
        "chain": "segment_000041 only; not the full ASR-to-subtitle chain",
    }
