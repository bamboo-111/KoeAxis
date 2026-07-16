from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from qwen_asr.defaults import DEFAULT_MODEL_CACHE_DIR
from qwen_asr.models import AudioSegment, WorkPaths
from qwen_asr.storage import write_json_atomic


def percentile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * ratio))))
    return float(sorted_values[index])


def auto_select_transcribe_batch_defaults(segments: list[AudioSegment]) -> dict[str, object]:
    durations = sorted(max(0.0, float(segment.duration)) for segment in segments)
    if not durations:
        return {
            "profile": "empty",
            "batch_size": 5,
            "target_batch_audio_seconds": 220.0,
            "single_long_segment_threshold": 90.0,
            "reasons": ["No segments available; falling back to conservative defaults."],
        }

    p50 = percentile(durations, 0.5)
    p75 = percentile(durations, 0.75)
    p90 = percentile(durations, 0.9)
    long_share = sum(1 for value in durations if value >= 90.0) / len(durations)
    ultra_share = sum(1 for value in durations if value >= 120.0) / len(durations)

    if p50 >= 85.0 or long_share >= 0.4:
        profile = "long_form"
        selected = {
            "batch_size": 3,
            "target_batch_audio_seconds": 300.0,
            "single_long_segment_threshold": 90.0,
        }
        reasons = ["Segment distribution is long-form heavy; prefer smaller batch caps and isolate long tails earlier."]
    elif p75 >= 75.0 or long_share >= 0.2:
        profile = "mixed"
        selected = {
            "batch_size": 4,
            "target_batch_audio_seconds": 260.0,
            "single_long_segment_threshold": 95.0,
        }
        reasons = ["Segment durations are mixed; keep moderate batch caps to balance throughput and padding risk."]
    else:
        profile = "short_form"
        selected = {
            "batch_size": 5,
            "target_batch_audio_seconds": 220.0,
            "single_long_segment_threshold": 110.0,
        }
        reasons = ["Most segments are short; allow a higher batch cap while keeping a long-tail escape hatch."]

    if ultra_share >= 0.1:
        selected["single_long_segment_threshold"] = min(float(selected["single_long_segment_threshold"]), 90.0)
        reasons.append("A noticeable share of segments is 120s+; tighten the long-segment threshold.")

    return {
        "profile": profile,
        "batch_size": int(selected["batch_size"]),
        "target_batch_audio_seconds": float(selected["target_batch_audio_seconds"]),
        "single_long_segment_threshold": float(selected["single_long_segment_threshold"]),
        "segment_stats": {
            "p50_duration": round(p50, 2),
            "p75_duration": round(p75, 2),
            "p90_duration": round(p90, 2),
            "long_segment_share": round(long_share, 3),
            "ultra_long_segment_share": round(ultra_share, 3),
        },
        "reasons": reasons,
    }


def resolve_transcribe_batch_defaults(args: argparse.Namespace, segments: list[AudioSegment]) -> dict[str, object]:
    auto_selection = auto_select_transcribe_batch_defaults(segments)
    batch_size_explicit = getattr(args, "batch_size", None) is not None
    target_explicit = getattr(args, "target_batch_audio_seconds", None) is not None
    threshold_explicit = getattr(args, "single_long_segment_threshold", None) is not None

    if getattr(args, "batch_mode", "adaptive") == "fixed":
        if not batch_size_explicit:
            args.batch_size = 5
        return {
            "profile": "fixed",
            "batch_size_source": "explicit" if batch_size_explicit else "auto",
            "target_audio_seconds_source": "n/a",
            "single_long_segment_threshold_source": "n/a",
            "auto_selection": auto_selection,
        }

    if not batch_size_explicit:
        args.batch_size = int(auto_selection["batch_size"])
    if not target_explicit:
        args.target_batch_audio_seconds = float(auto_selection["target_batch_audio_seconds"])
    if not threshold_explicit:
        args.single_long_segment_threshold = float(auto_selection["single_long_segment_threshold"])

    return {
        "profile": str(auto_selection["profile"]),
        "batch_size_source": "explicit" if batch_size_explicit else "auto",
        "target_audio_seconds_source": "explicit" if target_explicit else "auto",
        "single_long_segment_threshold_source": "explicit" if threshold_explicit else "auto",
        "auto_selection": auto_selection,
    }


def resolve_model_cache_dir(args: argparse.Namespace, default_model_cache_dir: Path | str = DEFAULT_MODEL_CACHE_DIR) -> str:
    cache_dir = getattr(args, "model_cache_dir", None)
    if cache_dir:
        return str(cache_dir)
    resolved = str(default_model_cache_dir)
    args.model_cache_dir = resolved
    return resolved


def prepare_model_cache_dir(model_cache_dir: str, *, local_files_only: bool) -> None:
    path = Path(model_cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write-test"
    try:
        probe.write_text("ok", encoding="ascii")
    except OSError as exc:
        raise RuntimeError(f"Model cache directory is not writable: {path}") from exc
    finally:
        if probe.exists():
            probe.unlink()

    if local_files_only and not any(path.iterdir()):
        raise RuntimeError(
            "Model cache directory is empty while local_files_only=True. "
            f"Populate {path} first, pass --model-cache-dir to an existing cache, "
            "or use --no-local-files-only to allow downloading."
        )


def consume_batch_memory_probes(transcriber: object) -> list[dict[str, object]]:
    consume = getattr(transcriber, "consume_last_batch_memory_probes", None)
    if callable(consume):
        return list(consume())
    return []


def write_transcribe_profile(
    work_paths: WorkPaths,
    args: argparse.Namespace,
    segments: list[AudioSegment],
    batch_reports: list[dict[str, object]],
    resolved_defaults: dict[str, object],
) -> None:
    if not getattr(args, "profile_batches", False):
        return
    completed = [report for report in batch_reports if report.get("status") == "completed"]
    oom_retries = [report for report in batch_reports if report.get("status") == "oom_retry"]
    singleton_reasons = Counter(
        str(report.get("singleton_reason"))
        for report in completed
        if report.get("singleton_reason")
    )
    profile_payload = {
        "stage": "transcribe",
        "batch_mode": getattr(args, "batch_mode", "adaptive"),
        "configured_batch_size": args.batch_size,
        "configured_target_batch_audio_seconds": getattr(args, "target_batch_audio_seconds", None),
        "configured_single_long_segment_threshold": getattr(args, "single_long_segment_threshold", 90.0),
        "resolved_defaults": resolved_defaults,
        "segment_count": len(segments),
        "summary": {
            "batch_count": len(batch_reports),
            "completed_batch_count": len(completed),
            "oom_retry_count": len(oom_retries),
            "max_completed_batch_size": max((int(report["batch_size"]) for report in completed), default=0),
            "max_completed_audio_seconds": max((float(report["total_duration"]) for report in completed), default=0.0),
            "max_duration_spread_ratio": max((float(report["duration_spread_ratio"]) for report in completed), default=0.0),
            "singleton_batches": sum(1 for report in completed if report.get("singleton_reason")),
            "singleton_reasons": dict(singleton_reasons),
        },
        "recommendation": build_transcribe_recommendation(args, completed, oom_retries),
        "batches": batch_reports,
    }
    write_json_atomic(work_paths.transcribe_profile_path, profile_payload)


def build_transcribe_recommendation(
    args: argparse.Namespace,
    completed: list[dict[str, object]],
    oom_retries: list[dict[str, object]],
) -> dict[str, object]:
    configured_batch_size = int(args.batch_size)
    configured_target_audio_seconds = getattr(args, "target_batch_audio_seconds", None)
    configured_single_long_segment_threshold = float(getattr(args, "single_long_segment_threshold", 90.0))

    completed_batch_sizes = [int(report["batch_size"]) for report in completed]
    completed_audio_seconds = [float(report["total_duration"]) for report in completed]
    completed_max_duration = [float(report["max_duration"]) for report in completed]
    singleton_long_segments = [
        float(report["max_duration"])
        for report in completed
        if report.get("singleton_reason") == "long_segment_threshold"
    ]
    high_spread_batches = [
        report
        for report in completed
        if float(report.get("duration_spread_ratio", 0.0)) >= 2.5
    ]

    if oom_retries:
        recommended_batch_size = max(
            1,
            min(
                configured_batch_size,
                min(max(1, int(report["batch_size"]) - 1) for report in oom_retries),
            ),
        )
    elif completed_batch_sizes:
        recommended_batch_size = max(completed_batch_sizes)
    else:
        recommended_batch_size = configured_batch_size

    if configured_target_audio_seconds is None:
        if completed_audio_seconds:
            recommended_target_audio_seconds = round(max(1.0, max(completed_audio_seconds) * 0.9), 2)
        else:
            recommended_target_audio_seconds = None
    else:
        if oom_retries and completed_audio_seconds:
            recommended_target_audio_seconds = round(
                min(float(configured_target_audio_seconds), max(completed_audio_seconds) * 0.95),
                2,
            )
        elif completed_audio_seconds:
            recommended_target_audio_seconds = round(
                max(float(configured_target_audio_seconds), max(completed_audio_seconds)),
                2,
            )
        else:
            recommended_target_audio_seconds = float(configured_target_audio_seconds)

    if singleton_long_segments:
        recommended_single_long_segment_threshold = round(min(singleton_long_segments), 2)
    elif completed_max_duration:
        recommended_single_long_segment_threshold = round(
            max(configured_single_long_segment_threshold, max(completed_max_duration) * 1.1),
            2,
        )
    else:
        recommended_single_long_segment_threshold = configured_single_long_segment_threshold

    reasons: list[str] = []
    if oom_retries:
        reasons.append("Observed OOM retries; recommend a lower stable batch cap.")
    if high_spread_batches:
        reasons.append("Some completed batches still had high duration spread; tighter grouping or lower target audio seconds may help.")
    if singleton_long_segments:
        reasons.append("Long-tail segments were isolated into singleton batches; keep the long-segment threshold near the shortest isolated segment.")
    if not reasons:
        reasons.append("No OOM retries were observed; recommendations are based on the largest completed adaptive batches.")

    return {
        "next_run": {
            "batch_mode": getattr(args, "batch_mode", "adaptive"),
            "batch_size": recommended_batch_size,
            "target_batch_audio_seconds": recommended_target_audio_seconds,
            "single_long_segment_threshold": recommended_single_long_segment_threshold,
        },
        "signals": {
            "oom_retry_count": len(oom_retries),
            "high_spread_batch_count": len(high_spread_batches),
            "singleton_long_segment_count": len(singleton_long_segments),
        },
        "reasons": reasons,
    }
