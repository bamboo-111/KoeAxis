from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
import os
from typing import Any

from qwen_asr.models import AlignedSegment, AlignedToken, TranscriptSegment
from qwen_asr.vendor_qwen import get_qwen3_forced_aligner_class

LOGGER = logging.getLogger(__name__)

BAD_ZERO_RUN = 8
MIN_COVERAGE_RATIO = 0.2
MIN_DENSE_COVERAGE_RATIO = 0.5
DENSE_ZERO_RATIO = 0.5
LOCAL_COLLAPSE_MIN_CHARS = 8
LOCAL_COLLAPSE_MAX_DURATION = 0.5
LOCAL_COLLAPSE_MAX_CPS = 35.0
LOCAL_COLLAPSE_MAX_TOKENS = 12


class QwenForcedAligner:
    def __init__(
        self,
        model_name: str,
        dtype: str = "fp16",
        device: str = "cuda",
        attn_implementation: str | None = None,
        keep_raw_model_output: bool = False,
        model_cache_dir: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self.keep_raw_model_output = keep_raw_model_output
        self.model_cache_dir = model_cache_dir
        self.local_files_only = local_files_only
        self._model: Any = None

    def load(self) -> None:
        LOGGER.info("Loading aligner model: %s", self.model_name)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for alignment") from exc

        torch_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.dtype)
        if torch_dtype is None:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

        try:
            Qwen3ForcedAligner = get_qwen3_forced_aligner_class()
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen-asr package is required for alignment. Install dependencies first."
            ) from exc

        kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": _normalize_device_map(self.device),
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.model_cache_dir:
            kwargs["cache_dir"] = self.model_cache_dir
        kwargs["local_files_only"] = self.local_files_only

        with _offline_model_loading(self.local_files_only):
            self._model = Qwen3ForcedAligner.from_pretrained(self.model_name, **kwargs)

    def run_segment(self, transcript_segment: TranscriptSegment, cleanup: bool = True) -> AlignedSegment:
        if self._model is None:
            raise RuntimeError("Aligner model is not loaded")

        LOGGER.info("Aligning %s", transcript_segment.segment_id)
        raw_output = None
        try:
            raw_output = self._model.align(
                audio=transcript_segment.audio_path,
                text=transcript_segment.text,
                language=transcript_segment.language,
            )
            tokens = _extract_tokens(raw_output, transcript_segment.global_start_time)
            timing_error = validate_aligned_token_timing(
                tokens,
                transcript_segment.global_start_time,
                transcript_segment.global_end_time,
            )
            if timing_error:
                raise RuntimeError(timing_error)
            return AlignedSegment(
                segment_id=transcript_segment.segment_id,
                audio_path=transcript_segment.audio_path,
                global_start_time=transcript_segment.global_start_time,
                global_end_time=transcript_segment.global_end_time,
                text=transcript_segment.text,
                language=transcript_segment.language,
                tokens=tokens,
                raw_model_output=_sanitize_raw_output(raw_output) if self.keep_raw_model_output else None,
                status="completed",
            )
        except Exception as exc:
            LOGGER.exception("Alignment failed for %s", transcript_segment.segment_id)
            return AlignedSegment(
                segment_id=transcript_segment.segment_id,
                audio_path=transcript_segment.audio_path,
                global_start_time=transcript_segment.global_start_time,
                global_end_time=transcript_segment.global_end_time,
                text=transcript_segment.text,
                language=transcript_segment.language,
                tokens=[],
                raw_model_output=_sanitize_raw_output(raw_output) if self.keep_raw_model_output else None,
                status="failed",
                error=str(exc),
            )
        finally:
            if cleanup:
                _cleanup_torch()

    def close(self) -> None:
        self._model = None
        _cleanup_torch(full=True)


def _extract_tokens(raw_output: Any, global_offset: float) -> list[AlignedToken]:
    if isinstance(raw_output, list):
        if not raw_output:
            return []
        return _extract_tokens(raw_output[0], global_offset)
    token_rows = None
    if isinstance(raw_output, dict):
        token_rows = raw_output.get("tokens") or raw_output.get("words") or raw_output.get("timestamps")
    elif hasattr(raw_output, "items"):
        token_rows = getattr(raw_output, "items")
    elif hasattr(raw_output, "tokens"):
        token_rows = getattr(raw_output, "tokens")
    if token_rows is None:
        return []

    tokens: list[AlignedToken] = []
    for item in token_rows:
        if isinstance(item, dict):
            text = item.get("text") or item.get("token") or item.get("word") or ""
            start_time = float(item.get("start") or item.get("start_time") or 0.0)
            end_time = float(item.get("end") or item.get("end_time") or start_time)
        else:
            text = str(getattr(item, "text", getattr(item, "token", "")))
            start_time = float(getattr(item, "start_time", getattr(item, "start", 0.0)))
            end_time = float(getattr(item, "end_time", getattr(item, "end", start_time)))
        tokens.append(
            AlignedToken(
                text=str(text),
                start_time=round(global_offset + start_time, 3),
                end_time=round(global_offset + end_time, 3),
            )
        )
    return tokens


def validate_aligned_token_timing(
    tokens: list[AlignedToken],
    global_start_time: float,
    global_end_time: float,
) -> str | None:
    if not tokens:
        return "alignment returned no tokens"

    zero_count = 0
    zero_run = 0
    max_zero_run = 0
    positive_start: float | None = None
    positive_end: float | None = None
    previous_start: float | None = None

    for token in tokens:
        if token.end_time <= token.start_time:
            zero_count += 1
            zero_run += 1
            max_zero_run = max(max_zero_run, zero_run)
        else:
            zero_run = 0
            positive_start = token.start_time if positive_start is None else min(positive_start, token.start_time)
            positive_end = token.end_time if positive_end is None else max(positive_end, token.end_time)
        if previous_start is not None and token.start_time < previous_start:
            return "alignment token timestamps are not monotonic"
        previous_start = token.start_time

    segment_duration = max(0.0, global_end_time - global_start_time)
    covered_duration = max(0.0, (positive_end or global_start_time) - (positive_start or global_start_time))
    if segment_duration > 0 and covered_duration / segment_duration < MIN_COVERAGE_RATIO:
        return (
            "alignment token timing unreliable: "
            f"covered {covered_duration:.3f}s of {segment_duration:.3f}s"
        )

    zero_ratio = zero_count / max(len(tokens), 1)
    if max_zero_run > BAD_ZERO_RUN:
        return f"alignment token timing unreliable: zero-duration run {max_zero_run}"
    if (
        segment_duration > 0
        and zero_ratio >= DENSE_ZERO_RATIO
        and covered_duration / segment_duration < MIN_DENSE_COVERAGE_RATIO
    ):
        return (
            "alignment token timing unreliable: "
            f"zero-duration ratio {zero_ratio:.2f} with covered {covered_duration:.3f}s of {segment_duration:.3f}s"
        )
    collapse_error = _validate_local_token_density(tokens)
    if collapse_error:
        return collapse_error
    return None


def _validate_local_token_density(tokens: list[AlignedToken]) -> str | None:
    normalized_tokens = [
        token
        for token in tokens
        if token.text.strip() and token.end_time >= token.start_time
    ]
    for start_index, first in enumerate(normalized_tokens):
        chars = 0
        window_start = first.start_time
        window_end = first.end_time
        for token in normalized_tokens[start_index : start_index + LOCAL_COLLAPSE_MAX_TOKENS]:
            chars += len(token.text.strip())
            window_end = max(window_end, token.end_time)
            duration = window_end - window_start
            if duration <= 0:
                continue
            cps = chars / duration
            if (
                chars >= LOCAL_COLLAPSE_MIN_CHARS
                and duration <= LOCAL_COLLAPSE_MAX_DURATION
                and cps >= LOCAL_COLLAPSE_MAX_CPS
            ):
                return (
                    "alignment token timing unreliable: "
                    f"local density {chars} chars in {duration:.3f}s ({cps:.1f} cps)"
                )
    return None


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
