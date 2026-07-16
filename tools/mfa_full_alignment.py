from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from qwen_asr.align import validate_aligned_token_timing
from qwen_asr.mfa_experiment import (
    _clean_mfa_lab_text,
    _evaluate_mfa_words,
    _project_mfa_root,
    _read_mfa_words,
    detect_mfa_environment,
)
from qwen_asr.models import AlignedSegment, AlignedToken, TranscriptSegment, WorkPaths
from qwen_asr.storage import write_json_atomic


DEFAULT_MFA_ACOUSTIC_MODEL = "japanese_mfa"
DEFAULT_MFA_DICTIONARY = "japanese_mfa"


def run_mfa_full_alignment(
    work_paths: WorkPaths,
    transcripts: list[TranscriptSegment],
    *,
    num_jobs: int = 1,
    acoustic_model: str = DEFAULT_MFA_ACOUSTIC_MODEL,
    dictionary: str = DEFAULT_MFA_DICTIONARY,
) -> tuple[list[AlignedSegment], dict[str, Any]]:
    environment = detect_mfa_environment(run_version_check=True)
    if not environment.get("available"):
        raise RuntimeError("MFA command is unavailable")
    command = environment.get("command")
    if not isinstance(command, list) or not command:
        raise RuntimeError("MFA command is missing")

    experiment_dir = work_paths.workdir / "experiments" / "mfa-full-align"
    corpus_dir = experiment_dir / "corpus"
    output_dir = experiment_dir / "out"
    report_path = work_paths.workdir / "reports" / "mfa_full_align.json"
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible = [item for item in transcripts if item.status == "completed" and item.text.strip()]
    input_items = _write_mfa_corpus(corpus_dir, eligible)
    started = time.monotonic()
    env = os.environ.copy()
    root_dir = environment.get("root_dir") or str(_project_mfa_root() or "")
    if root_dir:
        env["MFA_ROOT_DIR"] = str(root_dir)
    completed = subprocess.run(
        [
            *[str(value) for value in command],
            "align",
            str(corpus_dir),
            dictionary,
            acoustic_model,
            str(output_dir),
            "--clean",
            "--single_speaker",
            "--num_jobs",
            str(max(1, int(num_jobs))),
            "--overwrite",
            "--output_format",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=None,
        env=env,
    )
    elapsed_ms = int((time.monotonic() - started) * 1000)
    aligned = _build_aligned_segments(eligible, input_items, output_dir, completed.returncode)
    report = _build_mfa_full_report(
        environment=environment,
        input_items=input_items,
        aligned=aligned,
        returncode=completed.returncode,
        elapsed_ms=elapsed_ms,
        stdout=completed.stdout,
        stderr=completed.stderr,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
    )
    write_json_atomic(report_path, report)
    return aligned, report


def _write_mfa_corpus(corpus_dir: Path, transcripts: list[TranscriptSegment]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for transcript in transcripts:
        base = _safe_mfa_stem(transcript.segment_id)
        source_audio = Path(transcript.audio_path)
        wav_path = corpus_dir / f"{base}.wav"
        lab_path = corpus_dir / f"{base}.lab"
        lab_text = _clean_mfa_lab_text(transcript.text)
        status = "ready"
        reason = ""
        if not lab_text:
            status = "rejected"
            reason = "empty-lab-text-after-cleaning"
        elif not source_audio.exists():
            status = "rejected"
            reason = "source-audio-missing"
        if status == "ready":
            shutil.copy2(source_audio, wav_path)
            lab_path.write_text(lab_text, encoding="utf-8")
        items[transcript.segment_id] = {
            "segment_id": transcript.segment_id,
            "stem": base,
            "status": status,
            "reason": reason,
            "source_audio": str(source_audio),
            "wav": str(wav_path),
            "lab": str(lab_path),
            "lab_text": lab_text,
            "original_text": transcript.text,
        }
    return items


def _build_aligned_segments(
    transcripts: list[TranscriptSegment],
    input_items: dict[str, dict[str, Any]],
    output_dir: Path,
    returncode: int,
) -> list[AlignedSegment]:
    results: list[AlignedSegment] = []
    for transcript in transcripts:
        input_item = input_items.get(transcript.segment_id, {})
        if input_item.get("status") != "ready":
            results.append(_failed_segment(transcript, str(input_item.get("reason", "mfa-input-rejected"))))
            continue
        json_path = _find_mfa_output_json(output_dir, str(input_item.get("stem", "")))
        if json_path is None:
            reason = "mfa-align-failed" if returncode != 0 else "mfa-output-missing"
            results.append(_failed_segment(transcript, reason))
            continue
        words = _read_mfa_words(json_path)
        word_quality = _evaluate_mfa_words(words)
        tokens = _words_to_tokens(words, transcript.global_start_time)
        coverage = _coverage_ratio(tokens, transcript.global_start_time, transcript.global_end_time)
        unknown_count = int(word_quality.get("unknown_count", 0))
        if not word_quality.get("usable"):
            results.append(_failed_segment(transcript, "mfa-output-unusable", coverage=coverage, unknown_count=unknown_count))
            continue
        timing_error = validate_aligned_token_timing(
            tokens,
            transcript.global_start_time,
            transcript.global_end_time,
        )
        if timing_error:
            results.append(_failed_segment(transcript, timing_error, coverage=coverage, unknown_count=unknown_count))
            continue
        results.append(
            AlignedSegment(
                segment_id=transcript.segment_id,
                audio_path=transcript.audio_path,
                global_start_time=transcript.global_start_time,
                global_end_time=transcript.global_end_time,
                text=transcript.text,
                language=transcript.language,
                tokens=tokens,
                status="completed",
                alignment_backend="mfa",
                alignment_unit="word",
                alignment_coverage=coverage,
                alignment_unknown_count=unknown_count,
            )
        )
    return results


def _failed_segment(
    transcript: TranscriptSegment,
    reason: str,
    *,
    coverage: float | None = None,
    unknown_count: int = 0,
) -> AlignedSegment:
    return AlignedSegment(
        segment_id=transcript.segment_id,
        audio_path=transcript.audio_path,
        global_start_time=transcript.global_start_time,
        global_end_time=transcript.global_end_time,
        text=transcript.text,
        language=transcript.language,
        tokens=[],
        status="failed",
        error=reason,
        alignment_backend="mfa",
        alignment_unit="word",
        alignment_coverage=coverage,
        alignment_unknown_count=unknown_count,
        alignment_failure_reason=reason,
    )


def _words_to_tokens(words: list[dict[str, Any]], global_start_time: float) -> list[AlignedToken]:
    tokens: list[AlignedToken] = []
    offset = float(global_start_time)
    for word in words:
        text = str(word.get("text", "")).strip()
        if not text:
            continue
        try:
            start_ms = int(word.get("start_ms", 0))
            end_ms = int(word.get("end_ms", start_ms))
        except (TypeError, ValueError):
            continue
        tokens.append(
            AlignedToken(
                text=text,
                start_time=round(offset + start_ms / 1000.0, 3),
                end_time=round(offset + end_ms / 1000.0, 3),
            )
        )
    return tokens


def _coverage_ratio(tokens: list[AlignedToken], start_time: float, end_time: float) -> float:
    duration = max(0.0, end_time - start_time)
    if duration <= 0:
        return 0.0
    positive = [token for token in tokens if token.end_time > token.start_time]
    if not positive:
        return 0.0
    covered = max(token.end_time for token in positive) - min(token.start_time for token in positive)
    return round(max(0.0, min(1.0, covered / duration)), 6)


def _find_mfa_output_json(output_dir: Path, stem: str) -> Path | None:
    direct = output_dir / f"{stem}.json"
    if direct.exists():
        return direct
    matches = list(output_dir.rglob(f"{stem}.json"))
    return matches[0] if matches else None


def _safe_mfa_stem(segment_id: str) -> str:
    kept = [char if char.isalnum() or char in {"_", "-"} else "_" for char in segment_id]
    return "".join(kept).strip("_") or "segment"


def _build_mfa_full_report(
    *,
    environment: dict[str, Any],
    input_items: dict[str, dict[str, Any]],
    aligned: list[AlignedSegment],
    returncode: int,
    elapsed_ms: int,
    stdout: str,
    stderr: str,
    corpus_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    completed = [item for item in aligned if item.status == "completed"]
    failed = [item for item in aligned if item.status != "completed"]
    token_count = sum(len(item.tokens) for item in aligned)
    unknown_count = sum(int(item.alignment_unknown_count) for item in aligned)
    illegal_time_count = sum(
        1
        for item in aligned
        for token in item.tokens
        if token.end_time <= token.start_time
        or token.start_time < item.global_start_time
        or token.end_time > item.global_end_time
    )
    return {
        "alignment_backend": "mfa",
        "alignment_unit": "word",
        "status": "completed" if returncode == 0 else "failed",
        "returncode": returncode,
        "elapsed_ms": elapsed_ms,
        "corpus_dir": str(corpus_dir),
        "output_dir": str(output_dir),
        "environment": environment,
        "summary": {
            "input_count": len(input_items),
            "input_rejected_count": sum(1 for item in input_items.values() if item.get("status") != "ready"),
            "completed_count": len(completed),
            "failed_count": len(failed),
            "token_count": token_count,
            "unknown_count": unknown_count,
            "illegal_time_count": illegal_time_count,
            "average_coverage": round(
                sum(float(item.alignment_coverage or 0.0) for item in aligned) / max(1, len(aligned)),
                6,
            ),
        },
        "failures": [
            {
                "segment_id": item.segment_id,
                "reason": item.alignment_failure_reason or item.error or "",
                "coverage": item.alignment_coverage,
                "unknown_count": item.alignment_unknown_count,
            }
            for item in failed
        ],
        "inputs": list(input_items.values()),
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
    }
