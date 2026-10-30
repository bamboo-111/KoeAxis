from __future__ import annotations

import contextvars
from contextlib import contextmanager
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - optional when profiling is disabled
    psutil = None  # type: ignore[assignment]


_ACTIVE_STAGE: contextvars.ContextVar["StageSpan | None"] = contextvars.ContextVar(
    "qwen_asr_active_performance_stage",
    default=None,
)
_ACTIVE_STAGE_SHARED: StageSpan | None = None
_ACTIVE_STAGE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return _round(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return _round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _safe_process_tree() -> list[psutil.Process]:
    if psutil is None:
        return []
    root = psutil.Process(os.getpid())
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except (psutil.Error, OSError):
        pass
    return processes


def _process_snapshot() -> dict[str, int]:
    rss = 0
    read_bytes = 0
    write_bytes = 0
    process_count = 0
    for process in _safe_process_tree():
        try:
            rss += int(process.memory_info().rss)
            io = process.io_counters()
            read_bytes += int(io.read_bytes)
            write_bytes += int(io.write_bytes)
            process_count += 1
        except (psutil.Error, OSError):
            continue
    return {
        "rss_bytes": rss,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "process_count": process_count,
    }


def _gpu_snapshot() -> dict[str, int | None]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {"allocated_bytes": None, "reserved_bytes": None}
    try:
        cuda = torch.cuda
        if not cuda.is_available():
            return {"allocated_bytes": None, "reserved_bytes": None}
        return {
            "allocated_bytes": int(cuda.memory_allocated()),
            "reserved_bytes": int(cuda.memory_reserved()),
        }
    except (AttributeError, RuntimeError):
        return {"allocated_bytes": None, "reserved_bytes": None}


def _classify_io(operation: str, path: Path) -> str:
    name = path.name.lower()
    if operation == "append" or name.endswith(".jsonl"):
        return "event_append"
    if "checkpoint" in name:
        return f"checkpoint_{operation}"
    if "manifest" in name or name.endswith("segments.json"):
        return f"manifest_{operation}"
    return f"json_{operation}"


def record_io(operation: str, path: Path, byte_count: int, duration_s: float) -> None:
    span = _current_stage()
    if span is None:
        return
    span.record_io(
        kind=_classify_io(operation, path),
        path=path,
        byte_count=max(0, int(byte_count)),
        duration_s=max(0.0, float(duration_s)),
    )


@contextmanager
def measure_operation(kind: str, **metadata: Any):
    span = _current_stage()
    started = time.perf_counter()
    status = "completed"
    error_type: str | None = None
    try:
        yield
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        raise
    finally:
        if span is not None:
            span.record_operation(
                kind=kind,
                duration_s=time.perf_counter() - started,
                status=status,
                error_type=error_type,
                metadata=metadata,
            )


def record_network_request(
    *,
    provider: str,
    model: str,
    duration_s: float,
    success: bool,
    attempts: int,
    timeout: bool,
    input_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    thinking_disabled: bool,
    no_think_sent: bool,
    extra_body_sent: bool,
    error_type: str | None = None,
    cached_tokens: int | None = None,
) -> None:
    span = _current_stage()
    if span is None:
        return
    span.record_network_request(
        {
            "provider": provider,
            "model": model,
            "duration_s": _round(duration_s),
            "success": success,
            "attempts": max(1, int(attempts)),
            "retry_count": max(0, int(attempts) - 1),
            "timeout": timeout,
            "input_tokens": input_tokens,
            "cached_tokens": cached_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "thinking_disabled": thinking_disabled,
            "no_think_sent": no_think_sent,
            "extra_body_sent": extra_body_sent,
            "error_type": error_type,
        }
    )


def usage_tokens(usage: Any) -> tuple[int | None, int | None, int | None]:
    if usage is None:
        return None, None, None
    input_tokens = _optional_int(getattr(usage, "prompt_tokens", None))
    output_tokens = _optional_int(getattr(usage, "completion_tokens", None))
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = _optional_int(getattr(details, "reasoning_tokens", None)) if details is not None else None
    return input_tokens, output_tokens, reasoning_tokens


def usage_cached_tokens(usage: Any) -> int | None:
    """Prompt-cache hits (``prompt_tokens_details.cached_tokens``); None when not reported."""
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    return _optional_int(getattr(details, "cached_tokens", None)) if details is not None else None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _current_stage() -> StageSpan | None:
    span = _ACTIVE_STAGE.get()
    if span is not None:
        return span
    with _ACTIVE_STAGE_LOCK:
        return _ACTIVE_STAGE_SHARED


def collect_workload_snapshot(work_paths: Any) -> dict[str, Any]:
    from collections import Counter

    from qwen_asr.alignment_state import derive_alignment_state
    from qwen_asr.storage import read_json

    errors: dict[str, str] = {}

    def load(name: str, path: Path, default: Any) -> Any:
        try:
            return read_json(path, default=default)
        except (OSError, UnicodeError, ValueError) as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
            return default

    segments = load("segments", work_paths.segments_manifest, [])
    transcripts = load("transcripts", work_paths.transcript_manifest, [])
    aligned = load("aligned", work_paths.aligned_manifest, [])
    split = load("split", work_paths.split_manifest, {})
    translated = load("translated", work_paths.translated_manifest, {})
    mimo = load("mimo", work_paths.mimo_proofread_manifest, {})
    normalized = load("normalized", work_paths.normalized_manifest, {})
    progress = load("progress", work_paths.progress_path, {})
    segment_rows = [item for item in segments if isinstance(item, dict)] if isinstance(segments, list) else []
    transcript_rows = [item for item in transcripts if isinstance(item, dict)] if isinstance(transcripts, list) else []
    aligned_rows = [item for item in aligned if isinstance(item, dict)] if isinstance(aligned, list) else []
    transcript_status = Counter(str(item.get("status", "unknown")) for item in transcript_rows)
    alignment_status = Counter(derive_alignment_state(item) for item in aligned_rows)
    audio_seconds = sum(
        max(
            0.0,
            float(item.get("duration", 0.0) or 0.0)
            or float(item.get("global_end_time", 0.0) or 0.0) - float(item.get("global_start_time", 0.0) or 0.0),
        )
        for item in segment_rows
    )
    return {
        "audio_seconds": _round(audio_seconds, 3),
        "segment_count": len(segment_rows),
        "transcript_count": len(transcript_rows),
        "transcript_status": dict(sorted(transcript_status.items())),
        "aligned_count": len(aligned_rows),
        "alignment_status": dict(sorted(alignment_status.items())),
        "split_cue_count": len(split) if isinstance(split, dict) else 0,
        "translated_cue_count": len(translated) if isinstance(translated, dict) else 0,
        "mimo_cue_count": len(mimo) if isinstance(mimo, dict) else 0,
        "normalized_cue_count": len(normalized) if isinstance(normalized, dict) else 0,
        "progress": {
            "stage": progress.get("stage"),
            "status": progress.get("status"),
            "done": progress.get("done"),
            "total": progress.get("total"),
        }
        if isinstance(progress, dict)
        else {},
        "read_errors": errors,
    }


@dataclass(slots=True)
class _IoAggregate:
    count: int = 0
    bytes: int = 0
    duration_s: float = 0.0

    def add(self, byte_count: int, duration_s: float) -> None:
        self.count += 1
        self.bytes += byte_count
        self.duration_s += duration_s

    def to_json(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "bytes": self.bytes,
            "duration_s": _round(self.duration_s),
        }


@dataclass(slots=True)
class StageSpan:
    recorder: "PipelinePerformanceRecorder"
    name: str
    status: str = "running"
    summary: str = ""
    started_at: str = field(default_factory=_utc_now)
    _wall_start: float = field(default_factory=time.perf_counter)
    _cpu_start: float = field(default_factory=time.process_time)
    _process_start: dict[str, int] = field(default_factory=_process_snapshot)
    _io: dict[str, _IoAggregate] = field(default_factory=dict)
    _operations: list[dict[str, Any]] = field(default_factory=list)
    _network_requests: list[dict[str, Any]] = field(default_factory=list)
    _workload: dict[str, Any] = field(default_factory=dict)
    _paths: set[str] = field(default_factory=set)
    _peak_rss_bytes: int = 0
    _peak_process_count: int = 0
    _peak_gpu_allocated_bytes: int | None = None
    _peak_gpu_reserved_bytes: int | None = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _token: contextvars.Token["StageSpan | None"] | None = None
    _data_lock: threading.Lock = field(default_factory=threading.Lock)

    def __enter__(self) -> "StageSpan":
        global _ACTIVE_STAGE_SHARED
        self._token = _ACTIVE_STAGE.set(self)
        with _ACTIVE_STAGE_LOCK:
            _ACTIVE_STAGE_SHARED = self
        self._sample()
        self._thread = threading.Thread(
            target=self._sample_loop,
            name=f"performance-{self.name}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        global _ACTIVE_STAGE_SHARED
        if exc is not None:
            self.status = "failed"
            self.summary = self.summary or f"{type(exc).__name__}: {exc}"
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.recorder.sample_interval_s * 2))
        self._sample()
        if self._token is not None:
            _ACTIVE_STAGE.reset(self._token)
        with _ACTIVE_STAGE_LOCK:
            if _ACTIVE_STAGE_SHARED is self:
                _ACTIVE_STAGE_SHARED = None
        self.recorder.finish_stage(self)
        return False

    def set_result(self, status: str, summary: str = "") -> None:
        self.status = status
        self.summary = summary

    def record_io(self, *, kind: str, path: Path, byte_count: int, duration_s: float) -> None:
        with self._data_lock:
            aggregate = self._io.setdefault(kind, _IoAggregate())
            aggregate.add(byte_count, duration_s)
            if len(self._paths) < 200:
                self._paths.add(str(path))

    def record_operation(
        self,
        *,
        kind: str,
        duration_s: float,
        status: str,
        error_type: str | None,
        metadata: dict[str, Any],
    ) -> None:
        with self._data_lock:
            self._operations.append(
                {
                    "kind": kind,
                    "duration_s": _round(duration_s),
                    "status": status,
                    "error_type": error_type,
                    "metadata": metadata,
                }
            )

    def record_network_request(self, payload: dict[str, Any]) -> None:
        with self._data_lock:
            self._network_requests.append(payload)

    def set_workload(self, payload: dict[str, Any]) -> None:
        with self._data_lock:
            self._workload = dict(payload)

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.recorder.sample_interval_s):
            self._sample()

    def _sample(self) -> None:
        process = _process_snapshot()
        self._peak_rss_bytes = max(self._peak_rss_bytes, process["rss_bytes"])
        self._peak_process_count = max(self._peak_process_count, process["process_count"])
        gpu = _gpu_snapshot()
        allocated = gpu["allocated_bytes"]
        reserved = gpu["reserved_bytes"]
        if allocated is not None:
            self._peak_gpu_allocated_bytes = max(self._peak_gpu_allocated_bytes or 0, allocated)
        if reserved is not None:
            self._peak_gpu_reserved_bytes = max(self._peak_gpu_reserved_bytes or 0, reserved)

    def to_json(self) -> dict[str, Any]:
        process_end = _process_snapshot()
        wall_time_s = time.perf_counter() - self._wall_start
        cpu_time_s = time.process_time() - self._cpu_start
        with self._data_lock:
            instrumented_io = {key: value.to_json() for key, value in sorted(self._io.items())}
            instrumented_paths = sorted(self._paths)
            operations = list(self._operations)
            network_requests = list(self._network_requests)
            workload = dict(self._workload)
        timing_breakdown = _timing_breakdown(
            wall_time_s=wall_time_s,
            operations=operations,
            instrumented_io=instrumented_io,
            network_requests=network_requests,
        )
        return {
            "stage": self.name,
            "status": self.status,
            "summary": self.summary,
            "started_at": self.started_at,
            "finished_at": _utc_now(),
            "wall_time_s": _round(wall_time_s),
            "cpu_time_s": _round(cpu_time_s),
            "cpu_to_wall_ratio": _round(cpu_time_s / wall_time_s) if wall_time_s > 0 else 0.0,
            "peak_host_ram_mb": _round(self._peak_rss_bytes / (1024 * 1024), 3),
            "peak_process_count": self._peak_process_count,
            "peak_gpu_allocated_mb": (
                _round(self._peak_gpu_allocated_bytes / (1024 * 1024), 3)
                if self._peak_gpu_allocated_bytes is not None
                else None
            ),
            "peak_gpu_reserved_mb": (
                _round(self._peak_gpu_reserved_bytes / (1024 * 1024), 3)
                if self._peak_gpu_reserved_bytes is not None
                else None
            ),
            "process_io": {
                "read_bytes": max(0, process_end["read_bytes"] - self._process_start["read_bytes"]),
                "write_bytes": max(0, process_end["write_bytes"] - self._process_start["write_bytes"]),
            },
            "instrumented_io": instrumented_io,
            "instrumented_paths": instrumented_paths,
            "operations": operations,
            "network_requests": network_requests,
            "network_summary": _summarize_network_requests(network_requests),
            "workload": workload,
            "timing_breakdown": timing_breakdown,
        }


def _summarize_network_requests(requests: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(item["duration_s"]) for item in requests]
    token_fields = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")
    token_totals = {
        key: sum(int(item[key]) for item in requests if item.get(key) is not None)
        for key in token_fields
    }
    token_unavailable = {
        key: sum(1 for item in requests if item.get(key) is None)
        for key in token_fields
    }
    return {
        "request_count": len(requests),
        "success_count": sum(1 for item in requests if item.get("success")),
        "failure_count": sum(1 for item in requests if not item.get("success")),
        "retry_count": sum(int(item.get("retry_count", 0)) for item in requests),
        "timeout_count": sum(1 for item in requests if item.get("timeout")),
        "latency_s": {
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "max": _round(max(latencies)) if latencies else None,
        },
        "token_totals": token_totals,
        "token_unavailable_count": token_unavailable,
        "thinking_disabled_count": sum(1 for item in requests if item.get("thinking_disabled")),
        "no_think_sent_count": sum(1 for item in requests if item.get("no_think_sent")),
        "extra_body_sent_count": sum(1 for item in requests if item.get("extra_body_sent")),
    }


def _timing_breakdown(
    *,
    wall_time_s: float,
    operations: list[dict[str, Any]],
    instrumented_io: dict[str, dict[str, Any]],
    network_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    operation_seconds: dict[str, float] = {}
    for item in operations:
        kind = str(item.get("kind", "unknown"))
        operation_seconds[kind] = operation_seconds.get(kind, 0.0) + float(item.get("duration_s", 0.0) or 0.0)
    operation_seconds = {key: _round(value) for key, value in sorted(operation_seconds.items())}
    model_seconds = sum(
        value
        for key, value in operation_seconds.items()
        if key in {"model_load", "model_inference", "model_cleanup"}
    )
    io_seconds = sum(float(item.get("duration_s", 0.0) or 0.0) for item in instrumented_io.values())
    network_latency_sum_s = sum(float(item.get("duration_s", 0.0) or 0.0) for item in network_requests)
    remaining = max(0.0, wall_time_s - model_seconds - io_seconds)
    return {
        "stage_wall_s": _round(wall_time_s),
        "operation_seconds": operation_seconds,
        "model_lifecycle_s": _round(model_seconds),
        "instrumented_io_s": _round(io_seconds),
        "network_request_latency_sum_s": _round(network_latency_sum_s),
        "network_parallelism_ratio": _round(network_latency_sum_s / wall_time_s) if wall_time_s > 0 else 0.0,
        "remaining_including_network_and_orchestration_s": _round(remaining),
        "remaining_ratio": _round(remaining / wall_time_s) if wall_time_s > 0 else 0.0,
        "note": "Network request latency can overlap under concurrency and is not subtracted from remaining time.",
    }
class PipelinePerformanceRecorder:
    def __init__(
        self,
        output_path: Path,
        *,
        enabled: bool,
        sample_interval_s: float = 0.2,
    ) -> None:
        self.output_path = output_path
        self.enabled = enabled
        self.sample_interval_s = max(0.05, float(sample_interval_s))
        self.started_at = _utc_now()
        self._wall_start = time.perf_counter()
        self._stages: list[dict[str, Any]] = []

    def stage(self, name: str) -> StageSpan | _NullStageSpan:
        if not self.enabled:
            return _NullStageSpan()
        return StageSpan(self, name)

    def finish_stage(self, span: StageSpan) -> None:
        self._stages.append(span.to_json())
        self.write_report()

    def write_report(self) -> None:
        if not self.enabled:
            return
        payload = {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "started_at": self.started_at,
            "total_wall_time_s": _round(time.perf_counter() - self._wall_start),
            "sample_interval_s": self.sample_interval_s,
            "environment": {
                "python": sys.version.split()[0],
                "executable": sys.executable,
                "platform": platform.platform(),
                "pid": os.getpid(),
            },
            "stages": self._stages,
        }
        from qwen_asr.storage import write_json_atomic

        write_json_atomic(self.output_path, payload)


class _NullStageSpan:
    def __enter__(self) -> "_NullStageSpan":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def set_result(self, status: str, summary: str = "") -> None:
        del status, summary

    def set_workload(self, payload: dict[str, Any]) -> None:
        del payload
