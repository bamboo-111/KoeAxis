from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qwen_asr.mfa_guards import local_mfa_ass_guard, mfa_writeback_dry_run
from qwen_asr.mfa_lab import choose_mfa_lab_text, clean_mfa_lab_text
from qwen_asr.mfa_words import evaluate_mfa_words, globalize_mfa_words, read_mfa_words
from qwen_asr.models import WorkPaths


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
    experiment_dir = work_paths.workdir / "experiments" / "mfa-align-experiment"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    runnable = [
        item for item in candidates
        if isinstance(item.get("start_ms"), int)
        and isinstance(item.get("end_ms"), int)
        and item.get("end_ms", 0) > item.get("start_ms", 0)
        and str(item.get("text", "")).strip()
    ][: max(0, max_run_candidates)]
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
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_ms = max(0, int(candidate["start_ms"]) - max(0, padding_ms))
    end_ms = max(start_ms + 1, int(candidate["end_ms"]) + max(0, padding_ms))
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
    env = environ_factory()
    root_dir = environment.get("root_dir")
    if root_dir:
        env["MFA_ROOT_DIR"] = str(root_dir)
    completed = run_command(
        [
            *command,
            "align",
            str(corpus_dir),
            "japanese_mfa",
            "japanese_mfa",
            str(output_dir),
            "--clean",
            "--single_speaker",
            "--num_jobs",
            "1",
            "--overwrite",
            "--output_format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    elapsed_ms = int((monotonic() - started) * 1000)
    json_path = output_dir / "clip.json"
    words = read_mfa_words(json_path)
    word_quality = evaluate_mfa_words(words)
    global_word_ranges = globalize_mfa_words(words, clip_start_ms=start_ms)
    local_guard = local_mfa_ass_guard(candidate, lab_text, global_word_ranges, word_quality)
    writeback_dry_run = mfa_writeback_dry_run(candidate, local_guard)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "reason": "mfa-align-failed",
            "candidate": candidate,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "elapsed_ms": elapsed_ms,
            "clip": str(clip_path),
            "lab": str(lab_path),
            "output_json": str(json_path),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
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
        "clip": str(clip_path),
        "lab": str(lab_path),
        "lab_text": lab_text,
        "lab_text_source": lab_choice["source"],
        "output_json": str(json_path),
        "words": words,
        "word_quality": word_quality,
        "global_word_ranges": global_word_ranges,
        "local_ass_guard": local_guard,
        "writeback_dry_run": writeback_dry_run,
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-2000:],
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
