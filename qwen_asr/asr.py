from __future__ import annotations

import ctypes
import gc
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from qwen_asr.models import AudioSegment, TranscriptSegment
from qwen_asr.vendor_qwen import get_qwen3_asr_model_class

LOGGER = logging.getLogger(__name__)


class ASRBatchOOMError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BatchMemorySnapshot:
    host_private_mb: float | None
    host_rss_mb: float | None
    system_available_mb: float | None
    cuda_allocated_mb: float | None
    cuda_reserved_mb: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "host_private_mb": self.host_private_mb,
            "host_rss_mb": self.host_rss_mb,
            "system_available_mb": self.system_available_mb,
            "cuda_allocated_mb": self.cuda_allocated_mb,
            "cuda_reserved_mb": self.cuda_reserved_mb,
        }


@dataclass(frozen=True, slots=True)
class BatchMemoryProbe:
    phase: str
    batch_size: int
    segment_ids: list[str]
    snapshot: BatchMemorySnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "batch_size": self.batch_size,
            "segment_ids": self.segment_ids,
            **self.snapshot.to_dict(),
        }


class QwenASRTranscriber:
    def __init__(
        self,
        model_name: str,
        dtype: str = "fp16",
        device: str = "cuda",
        attn_implementation: str | None = None,
        max_new_tokens: int = 512,
        language: str | None = None,
        keep_raw_model_output: bool = False,
        model_cache_dir: str | None = None,
        local_files_only: bool = True,
        batch_size: int = 1,
        profile_batches: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.keep_raw_model_output = keep_raw_model_output
        self.model_cache_dir = model_cache_dir
        self.local_files_only = local_files_only
        self.batch_size = max(1, int(batch_size))
        self.profile_batches = profile_batches
        self._model: Any = None
        self._last_batch_probes: list[BatchMemoryProbe] = []

    def load(self) -> None:
        LOGGER.info("Loading ASR model: %s", self.model_name)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for transcription") from exc

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        try:
            Qwen3ASRModel = get_qwen3_asr_model_class()
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen-asr package is required for transcription. Install dependencies first."
            ) from exc

        kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": _normalize_device_map(self.device),
            "max_inference_batch_size": self.batch_size,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.model_cache_dir:
            kwargs["cache_dir"] = self.model_cache_dir
        kwargs["local_files_only"] = self.local_files_only

        with _offline_model_loading(self.local_files_only):
            self._model = Qwen3ASRModel.from_pretrained(self.model_name, **kwargs)

    def run_segment(self, segment: AudioSegment, cleanup: bool = True) -> TranscriptSegment:
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")

        LOGGER.info("Transcribing %s", segment.segment_id)
        raw_output = None
        text = ""
        language = self.language
        try:
            raw_output = self._model.transcribe(
                audio=segment.audio_path,
                language=self.language,
            )
            text, language = _extract_transcript(raw_output, self.language)
            return TranscriptSegment(
                segment_id=segment.segment_id,
                audio_path=segment.audio_path,
                global_start_time=segment.global_start_time,
                global_end_time=segment.global_end_time,
                text=text.strip(),
                language=language,
                raw_model_output=_sanitize_raw_output(raw_output) if self.keep_raw_model_output else None,
                status="completed",
            )
        except Exception as exc:
            LOGGER.exception("ASR failed for %s", segment.segment_id)
            return TranscriptSegment(
                segment_id=segment.segment_id,
                audio_path=segment.audio_path,
                global_start_time=segment.global_start_time,
                global_end_time=segment.global_end_time,
                text="",
                language=language,
                raw_model_output=_sanitize_raw_output(raw_output) if self.keep_raw_model_output else None,
                status="failed",
                error=str(exc),
            )
        finally:
            if cleanup:
                _cleanup_torch()

    def run_batch(self, segments: list[AudioSegment]) -> list[TranscriptSegment]:
        if not segments:
            return []
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")
        self._last_batch_probes = []
        if len(segments) == 1:
            segment = segments[0]
            self._log_batch_memory_probe("before_inference", segments)
            result = self.run_segment(segment, cleanup=False)
            self._log_batch_memory_probe("after_inference", segments)
            _cleanup_torch()
            self._log_batch_memory_probe("after_cleanup", segments)
            return [result]

        batch_label = ", ".join(segment.segment_id for segment in segments)
        LOGGER.info("Transcribing batch: %s", batch_label)
        self._log_batch_memory_probe("before_inference", segments)
        try:
            raw_outputs = self._model.transcribe(
                audio=[segment.audio_path for segment in segments],
                language=self.language,
            )
            self._log_batch_memory_probe("after_inference", segments)
            normalized_outputs = _normalize_batch_outputs(raw_outputs, len(segments))
            results = [
                _build_transcript_segment(
                    segment=segment,
                    raw_output=raw_output,
                    fallback_language=self.language,
                    keep_raw_model_output=self.keep_raw_model_output,
                )
                for segment, raw_output in zip(segments, normalized_outputs, strict=True)
            ]
            return results
        except Exception as exc:
            if len(segments) > 1 and _is_oom_error(exc):
                LOGGER.warning("Batch ASR hit OOM for %s", batch_label)
                raise ASRBatchOOMError(str(exc)) from exc
            LOGGER.exception("Batch ASR failed for %s. Falling back to per-segment transcription.", batch_label)
            return [self.run_segment(segment, cleanup=False) for segment in segments]
        finally:
            _cleanup_torch()
            self._log_batch_memory_probe("after_cleanup", segments)

    def close(self) -> None:
        self._model = None
        _cleanup_torch(full=True)

    def consume_last_batch_memory_probes(self) -> list[dict[str, Any]]:
        probes = [probe.to_dict() for probe in self._last_batch_probes]
        self._last_batch_probes = []
        return probes

    def _log_batch_memory_probe(self, phase: str, segments: list[AudioSegment]) -> None:
        if not self.profile_batches:
            return
        snapshot = capture_batch_memory_snapshot()
        probe = BatchMemoryProbe(
            phase=phase,
            batch_size=len(segments),
            segment_ids=[segment.segment_id for segment in segments],
            snapshot=snapshot,
        )
        self._last_batch_probes.append(probe)
        payload = {
            "event": "asr_batch_memory_probe",
            **probe.to_dict(),
        }
        LOGGER.info("%s", json.dumps(payload, ensure_ascii=True, sort_keys=True))


def _extract_transcript(raw_output: Any, fallback_language: str | None) -> tuple[str, str | None]:
    if raw_output is None:
        return "", fallback_language
    if isinstance(raw_output, list):
        if not raw_output:
            return "", fallback_language
        return _extract_transcript(raw_output[0], fallback_language)
    if isinstance(raw_output, str):
        return raw_output, fallback_language
    if isinstance(raw_output, dict):
        text = raw_output.get("text") or raw_output.get("transcript") or ""
        language = raw_output.get("language") or fallback_language
        return str(text), language
    if hasattr(raw_output, "text"):
        return str(getattr(raw_output, "text")), getattr(raw_output, "language", fallback_language)
    return str(raw_output), fallback_language


def _normalize_batch_outputs(raw_output: Any, expected_count: int) -> list[Any]:
    if isinstance(raw_output, list):
        if len(raw_output) == expected_count:
            return raw_output
        if len(raw_output) == 1 and expected_count == 1:
            return raw_output
        raise RuntimeError(
            f"ASR batch output size mismatch: expected {expected_count}, got {len(raw_output)}"
        )
    if expected_count == 1:
        return [raw_output]
    raise RuntimeError("ASR batch output is not a list for multi-segment transcription")


def _build_transcript_segment(
    *,
    segment: AudioSegment,
    raw_output: Any,
    fallback_language: str | None,
    keep_raw_model_output: bool,
) -> TranscriptSegment:
    text, language = _extract_transcript(raw_output, fallback_language)
    return TranscriptSegment(
        segment_id=segment.segment_id,
        audio_path=segment.audio_path,
        global_start_time=segment.global_start_time,
        global_end_time=segment.global_end_time,
        text=text.strip(),
        language=language,
        raw_model_output=_sanitize_raw_output(raw_output) if keep_raw_model_output else None,
        status="completed",
    )


def _sanitize_raw_output(raw_output: Any) -> Any:
    if raw_output is None:
        return None
    if isinstance(raw_output, (str, int, float, bool)):
        return raw_output
    if isinstance(raw_output, dict):
        return {str(key): _sanitize_raw_output(value) for key, value in raw_output.items()}
    if isinstance(raw_output, (list, tuple)):
        return [_sanitize_raw_output(item) for item in raw_output]
    if hasattr(raw_output, "__dict__"):
        return {str(key): _sanitize_raw_output(value) for key, value in vars(raw_output).items()}
    return str(raw_output)


def _cleanup_torch(full: bool = False) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    gc.collect()
    if torch.cuda.is_available():
        if full:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _normalize_device_map(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


@contextmanager
def _offline_model_loading(enabled: bool):
    if not enabled:
        yield
        return

    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _is_oom_error(exc: BaseException) -> bool:
    messages: list[str] = []
    current: BaseException | None = exc
    while current is not None:
        messages.append(str(current).lower())
        current = current.__cause__ if isinstance(current.__cause__, BaseException) else None
    joined = " || ".join(messages)
    return (
        "out of memory" in joined
        or "cuda out of memory" in joined
        or "cublas_status_alloc_failed" in joined
        or "cuda error: out of memory" in joined
    )


def capture_batch_memory_snapshot() -> BatchMemorySnapshot:
    host_private_mb, host_rss_mb = _read_host_memory_mb()
    cuda_allocated_mb, cuda_reserved_mb = _read_cuda_memory_mb()
    return BatchMemorySnapshot(
        host_private_mb=host_private_mb,
        host_rss_mb=host_rss_mb,
        system_available_mb=_read_system_available_memory_mb(),
        cuda_allocated_mb=cuda_allocated_mb,
        cuda_reserved_mb=cuda_reserved_mb,
    )


def _read_cuda_memory_mb() -> tuple[float | None, float | None]:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None, None
    if not torch.cuda.is_available():
        return None, None
    try:
        return _bytes_to_mb(torch.cuda.memory_allocated()), _bytes_to_mb(torch.cuda.memory_reserved())
    except Exception:  # pragma: no cover
        return None, None


def _read_system_available_memory_mb() -> float | None:
    if not hasattr(ctypes, "windll"):
        return None
    memory_status = _MemoryStatusEx()
    memory_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
        return None
    return _bytes_to_mb(memory_status.ullAvailPhys)


def _read_host_memory_mb() -> tuple[float | None, float | None]:
    if not hasattr(ctypes, "windll"):
        return None, None

    process_memory_counters = _ProcessMemoryCountersEx()
    cb = ctypes.sizeof(_ProcessMemoryCountersEx)
    process_handle = ctypes.windll.kernel32.GetCurrentProcess()
    success = ctypes.windll.psapi.GetProcessMemoryInfo(
        process_handle,
        ctypes.byref(process_memory_counters),
        cb,
    )
    if not success:
        return None, None
    return (
        _bytes_to_mb(process_memory_counters.PrivateUsage),
        _bytes_to_mb(process_memory_counters.WorkingSetSize),
    )


def _bytes_to_mb(value: int) -> float:
    return round(float(value) / (1024 * 1024), 2)


class _ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]
