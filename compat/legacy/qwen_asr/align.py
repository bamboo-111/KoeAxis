from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qwen_asr.model_runtime import (
    cleanup_torch as _cleanup_torch,
    normalize_device_map as _normalize_device_map,
    offline_model_loading as _offline_model_loading,
    sanitize_raw_output as _sanitize_raw_output,
)
from qwen_asr.models import AlignedSegment, AlignedToken, TranscriptSegment
from qwen_asr.performance import measure_operation
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
# Coverage is judged against the shorter of the VAD segment and the time the text needs at this
# generous speaking rate, so a short utterance inside a long segment (jingle, music, silence) passes.
COVERAGE_EXPECTED_CPS = 6.0
# Local collapse repair: re-time only the collapsed tokens between trustworthy neighbours.
REPAIR_MAX_CHAR_RATIO = 0.35
REPAIR_TARGET_CPS = 15.0
REPAIR_NATURAL_CPS = 8.0
REPAIR_MAX_EXPAND_TOKENS = 3
COLLAPSED_TOKEN_MAX_DURATION = 0.04


@dataclass(frozen=True, slots=True)
class AlignTimingValidationConfig:
    min_coverage_ratio: float = MIN_COVERAGE_RATIO
    bad_zero_run: int = BAD_ZERO_RUN
    min_dense_coverage_ratio: float = MIN_DENSE_COVERAGE_RATIO
    dense_zero_ratio: float = DENSE_ZERO_RATIO
    local_collapse_min_chars: int = LOCAL_COLLAPSE_MIN_CHARS
    local_collapse_max_duration: float = LOCAL_COLLAPSE_MAX_DURATION
    local_collapse_max_cps: float = LOCAL_COLLAPSE_MAX_CPS
    local_collapse_max_tokens: int = LOCAL_COLLAPSE_MAX_TOKENS
    coverage_expected_cps: float = COVERAGE_EXPECTED_CPS
    repair_local_collapse: bool = True
    repair_max_char_ratio: float = REPAIR_MAX_CHAR_RATIO
    repair_target_cps: float = REPAIR_TARGET_CPS
    repair_natural_cps: float = REPAIR_NATURAL_CPS
    repair_max_expand_tokens: int = REPAIR_MAX_EXPAND_TOKENS


class QwenForcedAligner:
    def __init__(
        self,
        model_name: str,
        dtype: str = "fp16",
        device: str = "cuda",
        attn_implementation: str | None = None,
        keep_raw_model_output: bool = False,
        keep_failed_tokens: bool = False,
        model_cache_dir: str | None = None,
        local_files_only: bool = True,
        timing_validation_config: AlignTimingValidationConfig | None = None,
    ) -> None:
        self.model_name = model_name
        self.dtype = dtype
        self.device = device
        self.attn_implementation = attn_implementation
        self.keep_raw_model_output = keep_raw_model_output
        self.keep_failed_tokens = keep_failed_tokens
        self.model_cache_dir = model_cache_dir
        self.local_files_only = local_files_only
        self.timing_validation_config = timing_validation_config or AlignTimingValidationConfig()
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
            raise RuntimeError("qwen-asr package is required for alignment. Install dependencies first.") from exc

        kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "device_map": _normalize_device_map(self.device),
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        if self.model_cache_dir:
            kwargs["cache_dir"] = self.model_cache_dir
        kwargs["local_files_only"] = self.local_files_only

        with measure_operation(
            "model_load",
            component="align",
            model=self.model_name,
            device=self.device,
            dtype=self.dtype,
            local_files_only=self.local_files_only,
        ):
            with _offline_model_loading(self.local_files_only):
                self._model = Qwen3ForcedAligner.from_pretrained(self.model_name, **kwargs)

    def run_segment(self, transcript_segment: TranscriptSegment, cleanup: bool = True) -> AlignedSegment:
        if self._model is None:
            raise RuntimeError("Aligner model is not loaded")

        LOGGER.info("Aligning %s", transcript_segment.segment_id)
        raw_output = None
        tokens: list[AlignedToken] = []
        try:
            with measure_operation(
                "model_inference",
                component="align",
                model=self.model_name,
                batch_size=1,
                audio_seconds=round(
                    max(0.0, transcript_segment.global_end_time - transcript_segment.global_start_time),
                    3,
                ),
            ):
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
                config=self.timing_validation_config,
            )
            recovery = None
            if timing_error and self.timing_validation_config.repair_local_collapse:
                repair = repair_local_token_collapse(
                    tokens,
                    transcript_segment.global_start_time,
                    transcript_segment.global_end_time,
                    config=self.timing_validation_config,
                )
                if repair is not None:
                    tokens, spans = repair
                    LOGGER.info(
                        "Repaired %d locally collapsed span(s) in %s: %s",
                        len(spans),
                        transcript_segment.segment_id,
                        timing_error,
                    )
                    recovery = {
                        "method": "align-local-repair",
                        "original_error": timing_error,
                        "repaired_spans": spans,
                    }
                    timing_error = None
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
                recovery=recovery,
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
                tokens=tokens if self.keep_failed_tokens else [],
                raw_model_output=_sanitize_raw_output(raw_output) if self.keep_raw_model_output else None,
                status="failed",
                error=str(exc),
            )
        finally:
            if cleanup:
                with measure_operation("model_cleanup", component="align", full=False):
                    _cleanup_torch()

    def close(self) -> None:
        self._model = None
        with measure_operation("model_cleanup", component="align", full=True):
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
        token_rows = raw_output.items
    elif hasattr(raw_output, "tokens"):
        token_rows = raw_output.tokens
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
    *,
    config: AlignTimingValidationConfig | None = None,
) -> str | None:
    config = config or AlignTimingValidationConfig()
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
    reference_duration = _coverage_reference_duration(tokens, segment_duration, config)
    if reference_duration > 0 and covered_duration / reference_duration < config.min_coverage_ratio:
        return f"alignment token timing unreliable: covered {covered_duration:.3f}s of {segment_duration:.3f}s"

    zero_ratio = zero_count / max(len(tokens), 1)
    if max_zero_run > config.bad_zero_run:
        return f"alignment token timing unreliable: zero-duration run {max_zero_run}"
    if (
        reference_duration > 0
        and zero_ratio >= config.dense_zero_ratio
        and covered_duration / reference_duration < config.min_dense_coverage_ratio
    ):
        return (
            "alignment token timing unreliable: "
            f"zero-duration ratio {zero_ratio:.2f} with covered {covered_duration:.3f}s of {segment_duration:.3f}s"
        )
    collapse_error = _validate_local_token_density(tokens, config=config)
    if collapse_error:
        return collapse_error
    return None


def _coverage_reference_duration(
    tokens: list[AlignedToken],
    segment_duration: float,
    config: AlignTimingValidationConfig,
) -> float:
    """Duration the tokens are expected to cover: the segment, capped by what the text needs."""
    if config.coverage_expected_cps <= 0:
        return segment_duration
    chars = sum(len(token.text.strip()) for token in tokens)
    return min(segment_duration, chars / config.coverage_expected_cps)


def _local_density_windows(
    tokens: list[AlignedToken],
    *,
    config: AlignTimingValidationConfig,
):
    """Yield (first_index, last_index, chars, duration) for every over-dense token window."""
    indexed = [
        (index, token)
        for index, token in enumerate(tokens)
        if token.text.strip() and token.end_time >= token.start_time
    ]
    for position, (first_index, first) in enumerate(indexed):
        chars = 0
        window_start = first.start_time
        window_end = first.end_time
        for index, token in indexed[position : position + config.local_collapse_max_tokens]:
            chars += len(token.text.strip())
            window_end = max(window_end, token.end_time)
            duration = window_end - window_start
            if duration <= 0:
                continue
            if (
                chars >= config.local_collapse_min_chars
                and duration <= config.local_collapse_max_duration
                and chars / duration >= config.local_collapse_max_cps
            ):
                yield first_index, index, chars, duration


def _validate_local_token_density(
    tokens: list[AlignedToken],
    *,
    config: AlignTimingValidationConfig,
) -> str | None:
    for _first, _last, chars, duration in _local_density_windows(tokens, config=config):
        cps = chars / duration
        return f"alignment token timing unreliable: local density {chars} chars in {duration:.3f}s ({cps:.1f} cps)"
    return None


def _collapsed_token_indices(tokens: list[AlignedToken], *, config: AlignTimingValidationConfig) -> list[int]:
    """Tokens squeezed to (near) zero length inside an over-dense window; the rest keep their timing."""
    collapsed: set[int] = set()
    for first, last, _chars, _duration in _local_density_windows(tokens, config=config):
        for index in range(first, last + 1):
            token = tokens[index]
            length = max(len(token.text.strip()), 1)
            duration = token.end_time - token.start_time
            if duration <= COLLAPSED_TOKEN_MAX_DURATION or length / max(duration, 1e-6) >= config.local_collapse_max_cps:
                collapsed.add(index)
    return sorted(collapsed)


def _index_runs(indices: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index <= runs[-1][1] + 2:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def repair_local_token_collapse(
    tokens: list[AlignedToken],
    global_start_time: float,
    global_end_time: float,
    *,
    config: AlignTimingValidationConfig | None = None,
) -> tuple[list[AlignedToken], list[dict[str, Any]]] | None:
    """Re-time locally collapsed tokens between their trustworthy neighbours.

    The aligner usually places most of a segment correctly and squeezes only a few tokens (repeats,
    chants, words that were probably not spoken) into ~0s. Instead of discarding the whole segment,
    spread just those tokens over the time between the surrounding tokens, widening the window by a
    few neighbours when it is too tight. Returns None when the collapse is too large or the repaired
    tokens still fail validation, so the caller keeps the original failure.
    """
    config = config or AlignTimingValidationConfig()
    collapsed = _collapsed_token_indices(tokens, config=config)
    if not collapsed:
        return None
    total_chars = sum(len(token.text.strip()) for token in tokens)
    collapsed_chars = sum(len(tokens[index].text.strip()) for index in collapsed)
    if total_chars <= 0 or collapsed_chars / total_chars > config.repair_max_char_ratio:
        return None

    repaired = [AlignedToken(token.text, token.start_time, token.end_time) for token in tokens]
    count = len(repaired)
    spans: list[dict[str, Any]] = []
    for run_start, run_end in _index_runs(collapsed):
        low, high = run_start, run_end
        for _ in range(config.repair_max_expand_tokens + 1):
            left, right, chars = _repair_window(repaired, low, high, global_start_time, global_end_time)
            if right - left > 0 and chars / (right - left) <= config.repair_target_cps:
                break
            if low == 0 and high == count - 1:
                break
            low, high = max(0, low - 1), min(count - 1, high + 1)
        left, right, chars = _repair_window(repaired, low, high, global_start_time, global_end_time)
        if right - left <= 0:
            return None
        # A wide window (pause/music around the collapse) is not filled: the tokens get the time a
        # natural speaking rate needs, placed around where the aligner collapsed them.
        needed = chars / config.repair_natural_cps if config.repair_natural_cps > 0 else right - left
        if needed < right - left:
            anchor = tokens[run_start].start_time
            left = min(max(anchor - needed / 2, left), right - needed)
            right = left + needed
        cursor = left
        for index in range(low, high + 1):
            duration = (right - left) * max(len(repaired[index].text.strip()), 1) / chars
            repaired[index] = AlignedToken(repaired[index].text, round(cursor, 3), round(cursor + duration, 3))
            cursor += duration
        spans.append(
            {
                "token_start": low,
                "token_end": high,
                "collapsed_text": "".join(tokens[index].text for index in range(run_start, run_end + 1)),
                "text": "".join(token.text for token in repaired[low : high + 1]),
                "start_time": round(left, 3),
                "end_time": round(right, 3),
            }
        )
    if validate_aligned_token_timing(repaired, global_start_time, global_end_time, config=config):
        return None
    return repaired, spans


def _repair_window(
    tokens: list[AlignedToken],
    low: int,
    high: int,
    global_start_time: float,
    global_end_time: float,
) -> tuple[float, float, int]:
    left = tokens[low - 1].end_time if low > 0 else global_start_time
    right = tokens[high + 1].start_time if high + 1 < len(tokens) else global_end_time
    chars = sum(max(len(tokens[index].text.strip()), 1) for index in range(low, high + 1))
    return left, right, chars
