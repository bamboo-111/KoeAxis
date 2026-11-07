from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import unicodedata
from collections.abc import Callable, Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from qwen_asr.mfa_guards import local_mfa_ass_guard, mfa_writeback_dry_run
from qwen_asr.mfa_lab import choose_mfa_lab_text, clean_mfa_lab_text
from qwen_asr.mfa_words import evaluate_mfa_words, globalize_mfa_words, read_mfa_words
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic

# Each `mfa align --clean` call spends ~110 s loading the Japanese dictionary/tokenizer before aligning a few
# seconds of audio, independent of how many clips it aligns. Callers that know several candidates up front
# prefetch them in ONE call (one speaker directory per clip, so per-clip speaker adaptation is identical to an
# individual run); later single-candidate calls are served from this cache keyed by audio, clip range and text.
BATCH_CACHE_FILE = "batch-cache.json"

# Almost all of that start-up is spent loading and normalising the 544k-entry `japanese_mfa.dict`; the actual
# alignment takes seconds. MFA only ever looks up words that occur in the .lab text, so a dictionary holding
# just the entries that are substrings of the batch's lab texts gives identical word timings (verified on 13
# historical clips, including <unk> cases) in ~15 s instead of ~120 s. Set QWEN_ASR_MFA_PRUNE_DICTIONARY=0
# (or environment["prune_dictionary"]=False) to always use the full dictionary; a failed pruned run is
# retried once with the full dictionary.
MFA_MODEL_NAME = "japanese_mfa"
PRUNE_DICTIONARY_ENV = "QWEN_ASR_MFA_PRUNE_DICTIONARY"
PRUNED_DICTIONARY_FILE = "pruned_dictionary.dict"
PRUNE_MAX_WORD_CHARS = 24

def mfa_experiment_dir(work_paths: WorkPaths) -> Path:
    return work_paths.workdir / "experiments" / "mfa-align-experiment"


def mfa_clip_range(candidate: dict[str, Any], padding_ms: int) -> tuple[int, int]:
    start_ms = max(0, int(candidate["start_ms"]) - max(0, padding_ms))
    end_ms = max(start_ms + 1, int(candidate["end_ms"]) + max(0, padding_ms))
    return start_ms, end_ms


def mfa_cache_key(audio_path: Path, start_ms: int, end_ms: int, lab_text: str) -> str:
    try:
        stat = audio_path.stat()
        stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
    except OSError:
        stamp = "missing"
    raw = json.dumps([str(audio_path.resolve()), stamp, start_ms, end_ms, lab_text], ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _runnable(candidate: dict[str, Any]) -> bool:
    return (
        isinstance(candidate.get("start_ms"), int)
        and isinstance(candidate.get("end_ms"), int)
        and candidate.get("end_ms", 0) > candidate.get("start_ms", 0)
        and bool(str(candidate.get("text", "")).strip())
    )


def _mfa_align_command(
    command: list[str],
    corpus_dir: Path,
    output_dir: Path,
    *,
    single_speaker: bool,
    dictionary: str = MFA_MODEL_NAME,
) -> list[str]:
    return [
        *command,
        "align",
        str(corpus_dir),
        dictionary,
        MFA_MODEL_NAME,
        str(output_dir),
        "--clean",
        *(["--single_speaker"] if single_speaker else []),
        "--num_jobs",
        "1",
        "--overwrite",
        "--output_format",
        "json",
    ]


def prune_dictionary_enabled(environment: dict[str, Any]) -> bool:
    value = environment.get("prune_dictionary")
    if value is None:
        value = os.environ.get(PRUNE_DICTIONARY_ENV, "1")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "off", "no", ""}


def full_mfa_dictionary_path(environment: dict[str, Any]) -> Path | None:
    root_dir = str(environment.get("root_dir") or "").strip()
    if not root_dir:
        return None
    path = Path(root_dir) / "pretrained_models" / "dictionary" / f"{MFA_MODEL_NAME}.dict"
    return path if path.is_file() else None


def _lab_substrings(lab_texts: Iterable[str], max_chars: int) -> set[str]:
    substrings: set[str] = set()
    for raw in lab_texts:
        for text in {raw, unicodedata.normalize("NFKC", raw).lower()}:
            length = len(text)
            for start in range(length):
                for end in range(start + 1, min(length, start + max_chars) + 1):
                    substrings.add(text[start:end])
    return substrings


def write_pruned_mfa_dictionary(
    full_dictionary: Path,
    lab_texts: Iterable[str],
    output_path: Path,
    *,
    max_chars: int = PRUNE_MAX_WORD_CHARS,
) -> dict[str, Any]:
    """Keep only entries whose word (or its lowercase form) is a substring of some lab text."""
    started = time.monotonic()
    substrings = _lab_substrings(lab_texts, max_chars)
    kept = 0
    total = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with full_dictionary.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8", newline="\n") as target:
        for line in source:
            total += 1
            word = line.split("\t", 1)[0]
            if word in substrings or word.lower() in substrings:
                target.write(line)
                kept += 1
    return {
        "path": str(output_path),
        "kept_entries": kept,
        "total_entries": total,
        "prune_ms": int((time.monotonic() - started) * 1000),
    }


def _run_mfa_align(
    command: list[str],
    corpus_dir: Path,
    output_dir: Path,
    *,
    lab_texts: list[str],
    work_dir: Path,
    single_speaker: bool,
    environment: dict[str, Any],
    run_command: Callable[..., Any],
    environ_factory: Callable[[], dict[str, str]],
    timeout: int,
    expected_outputs: list[Path],
) -> tuple[Any, dict[str, Any]]:
    """Run `mfa align`, preferring a per-batch pruned dictionary; retry once with the full one on failure."""
    env = _mfa_env(environment, environ_factory)
    info: dict[str, Any] = {"mode": "full", "fallback": False}
    full_dictionary = full_mfa_dictionary_path(environment) if prune_dictionary_enabled(environment) else None
    if full_dictionary is not None:
        try:
            pruned = write_pruned_mfa_dictionary(full_dictionary, lab_texts, work_dir / PRUNED_DICTIONARY_FILE)
        except (OSError, UnicodeError) as exc:
            info["prune_error"] = f"{type(exc).__name__}: {exc}"
        else:
            info = {"mode": "pruned", "fallback": False, **pruned}
            completed = run_command(
                _mfa_align_command(
                    command, corpus_dir, output_dir, single_speaker=single_speaker, dictionary=pruned["path"]
                ),
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
            if completed.returncode == 0 and all(path.exists() for path in expected_outputs):
                return completed, info
            info = {
                **info,
                "mode": "full",
                "fallback": True,
                "pruned_returncode": completed.returncode,
                "pruned_stderr_tail": (completed.stderr or "")[-1000:],
            }
    completed = run_command(
        _mfa_align_command(command, corpus_dir, output_dir, single_speaker=single_speaker),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return completed, info


def _mfa_env(environment: dict[str, Any], environ_factory: Callable[[], dict[str, str]]) -> dict[str, str]:
    env = environ_factory()
    root_dir = environment.get("root_dir")
    if root_dir:
        env["MFA_ROOT_DIR"] = str(root_dir)
    return env


def read_mfa_batch_cache(work_paths: WorkPaths) -> dict[str, Any]:
    cache = read_json(mfa_experiment_dir(work_paths) / BATCH_CACHE_FILE, default={})
    return cache if isinstance(cache, dict) else {}


def prefetch_local_mfa_alignments(
    work_paths: WorkPaths,
    candidates: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    padding_ms: int,
    run_command: Callable[..., Any] | None = None,
    monotonic: Callable[[], float] | None = None,
    environ_factory: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Align every not-yet-cached candidate in one MFA call and cache the per-clip words."""
    run_command = run_command or subprocess.run
    monotonic = monotonic or time.monotonic
    environ_factory = environ_factory or os.environ.copy
    if not environment.get("available"):
        return {"status": "skipped", "reason": "mfa-unavailable", "aligned_count": 0}
    command = environment.get("command")
    if not isinstance(command, list) or not command:
        return {"status": "skipped", "reason": "mfa-command-missing", "aligned_count": 0}
    if not work_paths.audio_path.exists():
        return {"status": "skipped", "reason": "source-audio-missing", "aligned_count": 0}
    experiment_dir = mfa_experiment_dir(work_paths)
    cache = read_mfa_batch_cache(work_paths)
    batch_dir = experiment_dir / f"batch_{datetime.now():%Y%m%d-%H%M%S-%f}"
    corpus_dir = batch_dir / "corpus"
    output_dir = batch_dir / "out"
    planned: list[dict[str, Any]] = []
    already_cached = 0
    seen: set[str] = set()
    for index, candidate in enumerate(candidates, 1):
        if not _runnable(candidate):
            continue
        start_ms, end_ms = mfa_clip_range(candidate, padding_ms)
        lab_choice = choose_mfa_lab_text(work_paths, candidate)
        lab_text = clean_mfa_lab_text(lab_choice["text"])
        if not lab_text:
            continue
        key = mfa_cache_key(work_paths.audio_path, start_ms, end_ms, lab_text)
        if key in cache:
            already_cached += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        name = f"clip_{index:03d}"
        speaker_dir = corpus_dir / name
        speaker_dir.mkdir(parents=True, exist_ok=True)
        clip_path = speaker_dir / f"{name}.wav"
        lab_path = speaker_dir / f"{name}.lab"
        lab_path.write_text(lab_text, encoding="utf-8")
        clip_result = ffmpeg_extract_clip(
            work_paths.audio_path, clip_path, start_ms=start_ms, end_ms=end_ms, run_command=run_command
        )
        if clip_result["status"] != "completed":
            continue
        planned.append(
            {
                "key": key,
                "name": name,
                "clip": str(clip_path),
                "lab": str(lab_path),
                "lab_text": lab_text,
                "lab_text_source": lab_choice["source"],
            }
        )
    if not planned:
        return {"status": "completed", "aligned_count": 0, "already_cached": already_cached}
    output_dir.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    completed, dictionary_info = _run_mfa_align(
        [str(value) for value in command],
        corpus_dir,
        output_dir,
        lab_texts=[row["lab_text"] for row in planned],
        work_dir=batch_dir,
        single_speaker=False,
        environment=environment,
        run_command=run_command,
        environ_factory=environ_factory,
        timeout=600 + 30 * len(planned),
        expected_outputs=[output_dir / row["name"] / f"{row['name']}.json" for row in planned],
    )
    elapsed_ms = int((monotonic() - started) * 1000)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "reason": "mfa-align-failed",
            "aligned_count": 0,
            "planned_count": len(planned),
            "elapsed_ms": elapsed_ms,
            "batch_dir": str(batch_dir),
            "dictionary": dictionary_info,
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
    for row in planned:
        json_path = output_dir / row["name"] / f"{row['name']}.json"
        cache[row["key"]] = {
            **{key: value for key, value in row.items() if key not in {"key", "name"}},
            "output_json": str(json_path),
            "output_exists": json_path.exists(),
            "words": read_mfa_words(json_path),
            "elapsed_ms": elapsed_ms,
            "batch_dir": str(batch_dir),
            "batch_size": len(planned),
            "dictionary": dictionary_info,
            "stdout_tail": (completed.stdout or "")[-2000:],
            "stderr_tail": (completed.stderr or "")[-2000:],
        }
    write_json_atomic(experiment_dir / BATCH_CACHE_FILE, cache)
    return {
        "status": "completed",
        "aligned_count": len(planned),
        "already_cached": already_cached,
        "elapsed_ms": elapsed_ms,
        "batch_dir": str(batch_dir),
        "dictionary": dictionary_info,
    }


def run_local_mfa_alignment_experiments(
    work_paths: WorkPaths,
    candidates: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    max_run_candidates: int,
    padding_ms: int,
    run_command: Callable[..., Any] | None = None,
    monotonic: Callable[[], float] | None = None,
    environ_factory: Callable[[], dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    run_command = run_command or subprocess.run
    monotonic = monotonic or time.monotonic
    environ_factory = environ_factory or os.environ.copy
    if not environment.get("available"):
        return [
            {
                "status": "skipped",
                "reason": "mfa-unavailable",
            }
        ]
    if not work_paths.audio_path.exists():
        return [
            {
                "status": "skipped",
                "reason": "source-audio-missing",
                "audio_path": str(work_paths.audio_path),
            }
        ]
    command = environment.get("command")
    if not isinstance(command, list) or not command:
        return [
            {
                "status": "skipped",
                "reason": "mfa-command-missing",
            }
        ]
    experiment_dir = mfa_experiment_dir(work_paths)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runnable = [item for item in candidates if _runnable(item)][: max(0, max_run_candidates)]
    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(runnable, 1):
        results.append(
            run_one_local_mfa_alignment(
                work_paths,
                candidate,
                environment=environment,
                command=[str(value) for value in command],
                experiment_dir=experiment_dir,
                index=index,
                padding_ms=padding_ms,
                run_command=run_command,
                monotonic=monotonic,
                environ_factory=environ_factory,
            )
        )
    return results


def run_one_local_mfa_alignment(
    work_paths: WorkPaths,
    candidate: dict[str, Any],
    *,
    environment: dict[str, Any],
    command: list[str],
    experiment_dir: Path,
    index: int,
    padding_ms: int,
    run_command: Callable[..., Any] | None = None,
    monotonic: Callable[[], float] | None = None,
    environ_factory: Callable[[], dict[str, str]] | None = None,
) -> dict[str, Any]:
    run_command = run_command or subprocess.run
    monotonic = monotonic or time.monotonic
    environ_factory = environ_factory or os.environ.copy
    candidate_dir = experiment_dir / f"candidate_{index:03d}"
    corpus_dir = candidate_dir / "corpus"
    output_dir = candidate_dir / "out"
    start_ms, end_ms = mfa_clip_range(candidate, padding_ms)
    clip_path = corpus_dir / "clip.wav"
    lab_path = corpus_dir / "clip.lab"
    lab_choice = choose_mfa_lab_text(work_paths, candidate)
    lab_text = clean_mfa_lab_text(lab_choice["text"])
    if not lab_text:
        return {
            "status": "skipped",
            "reason": "empty-lab-text-after-cleaning",
            "candidate": candidate,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "lab": str(lab_path),
        }
    cached = read_mfa_batch_cache(work_paths).get(mfa_cache_key(work_paths.audio_path, start_ms, end_ms, lab_text))
    if isinstance(cached, dict) and cached.get("output_exists"):
        result = _build_mfa_result(
            candidate,
            start_ms=start_ms,
            end_ms=end_ms,
            lab_text=lab_text,
            lab_text_source=str(cached.get("lab_text_source") or lab_choice["source"]),
            words=list(cached.get("words") or []),
            returncode=0,
            elapsed_ms=int(cached.get("elapsed_ms") or 0),
            clip=str(cached.get("clip", "")),
            lab=str(cached.get("lab", "")),
            output_json=str(cached.get("output_json", "")),
            stdout=str(cached.get("stdout_tail", "")),
            stderr=str(cached.get("stderr_tail", "")),
        )
        result["batch_cache"] = {"batch_dir": cached.get("batch_dir"), "batch_size": cached.get("batch_size")}
        return result
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    lab_path.write_text(lab_text, encoding="utf-8")
    clip_result = ffmpeg_extract_clip(
        work_paths.audio_path,
        clip_path,
        start_ms=start_ms,
        end_ms=end_ms,
        run_command=run_command,
    )
    if clip_result["status"] != "completed":
        return {
            "status": "failed",
            "reason": "ffmpeg-extract-failed",
            "candidate": candidate,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "clip": str(clip_path),
            "details": clip_result,
        }
    started = monotonic()
    json_path = output_dir / "clip.json"
    completed, dictionary_info = _run_mfa_align(
        command,
        corpus_dir,
        output_dir,
        lab_texts=[lab_text],
        work_dir=candidate_dir,
        single_speaker=True,
        environment=environment,
        run_command=run_command,
        environ_factory=environ_factory,
        timeout=600,
        expected_outputs=[json_path],
    )
    elapsed_ms = int((monotonic() - started) * 1000)
    result = _build_mfa_result(
        candidate,
        start_ms=start_ms,
        end_ms=end_ms,
        lab_text=lab_text,
        lab_text_source=lab_choice["source"],
        words=read_mfa_words(json_path),
        returncode=completed.returncode,
        elapsed_ms=elapsed_ms,
        clip=str(clip_path),
        lab=str(lab_path),
        output_json=str(json_path),
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
    )
    result["dictionary"] = dictionary_info
    return result


def _build_mfa_result(
    candidate: dict[str, Any],
    *,
    start_ms: int,
    end_ms: int,
    lab_text: str,
    lab_text_source: str,
    words: list[dict[str, Any]],
    returncode: int,
    elapsed_ms: int,
    clip: str,
    lab: str,
    output_json: str,
    stdout: str,
    stderr: str,
) -> dict[str, Any]:
    word_quality = evaluate_mfa_words(words)
    global_word_ranges = globalize_mfa_words(words, clip_start_ms=start_ms)
    local_guard = local_mfa_ass_guard(candidate, lab_text, global_word_ranges, word_quality)
    writeback_dry_run = mfa_writeback_dry_run(candidate, local_guard)
    if returncode != 0:
        return {
            "status": "failed",
            "reason": "mfa-align-failed",
            "candidate": candidate,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "elapsed_ms": elapsed_ms,
            "clip": clip,
            "lab": lab,
            "output_json": output_json,
            "stdout": stdout,
            "stderr": stderr,
            "words": words,
            "word_quality": word_quality,
            "local_ass_guard": local_guard,
            "writeback_dry_run": writeback_dry_run,
        }
    return {
        "status": "completed",
        "usable": word_quality["usable"],
        "reason": "" if word_quality["usable"] else "mfa-output-unusable",
        "candidate": candidate,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "elapsed_ms": elapsed_ms,
        "clip": clip,
        "lab": lab,
        "lab_text": lab_text,
        "lab_text_source": lab_text_source,
        "output_json": output_json,
        "words": words,
        "word_quality": word_quality,
        "global_word_ranges": global_word_ranges,
        "local_ass_guard": local_guard,
        "writeback_dry_run": writeback_dry_run,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def ffmpeg_extract_clip(
    source_audio: Path,
    clip_path: Path,
    *,
    start_ms: int,
    end_ms: int,
    run_command: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    run_command = run_command or subprocess.run
    start_seconds = max(0, start_ms) / 1000.0
    duration_seconds = max(0.001, end_ms - start_ms) / 1000.0
    completed = run_command(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(source_audio),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(clip_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
