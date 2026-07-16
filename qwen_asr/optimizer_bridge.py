from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_asr import optimizer_bridge_adapter as _adapter
from qwen_asr import optimizer_bridge_guards as _guards
from qwen_asr import optimizer_bridge_stages as _stage_calls
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json

DEFAULT_OPTIMIZER_ROOT = _stage_calls.DEFAULT_OPTIMIZER_ROOT
ALIGN_LOCAL_INTERPOLATION_MAX_GAP_MS = _adapter.ALIGN_LOCAL_INTERPOLATION_MAX_GAP_MS
ALIGN_ZERO_TOKEN_DEFAULT_DURATION_MS = _adapter.ALIGN_ZERO_TOKEN_DEFAULT_DURATION_MS
ALIGN_ZERO_TOKEN_MAX_DURATION_MS = _adapter.ALIGN_ZERO_TOKEN_MAX_DURATION_MS
FALLBACK_SHORT_TEXT_MAX_NORMALIZED_CHARS = _adapter.FALLBACK_SHORT_TEXT_MAX_NORMALIZED_CHARS
FALLBACK_SHORT_TEXT_MIN_DURATION_MS = _adapter.FALLBACK_SHORT_TEXT_MIN_DURATION_MS
FALLBACK_SHORT_TEXT_MAX_DURATION_MS = _adapter.FALLBACK_SHORT_TEXT_MAX_DURATION_MS
FALLBACK_SHORT_TEXT_LONG_DURATION_MS = _adapter.FALLBACK_SHORT_TEXT_LONG_DURATION_MS
FALLBACK_SHORT_RESPONSE_NORMALIZED = _adapter.FALLBACK_SHORT_RESPONSE_NORMALIZED
aligned_manifest_to_asr_data = _adapter.aligned_manifest_to_asr_data
_owned_segment_bounds_ms = _adapter._owned_segment_bounds_ms
_sanitized_aligned_payload = _adapter._sanitized_aligned_payload
_sanitized_payload_item = _adapter._sanitized_payload_item
_asr_segments_to_tokens = _adapter._asr_segments_to_tokens
_chunks_to_asr_segments = _adapter._chunks_to_asr_segments
_remove_exact_boundary_duplicate = _adapter._remove_exact_boundary_duplicate
_narrow_overlong_short_text_ranges = _adapter._narrow_overlong_short_text_ranges
_is_fallback_short_response_text = _adapter._is_fallback_short_response_text
_repair_piece_ranges = _adapter._repair_piece_ranges
_raw_piece_range = _adapter._raw_piece_range
_fill_missing_range_run = _adapter._fill_missing_range_run
_enforce_monotonic_ranges = _adapter._enforce_monotonic_ranges
_normalize_content = _adapter._normalize_content
_restore_transcript_surface_to_pieces = _adapter._restore_transcript_surface_to_pieces
_new_asr_data_seg = _adapter._new_asr_data_seg
_validate_aligned_manifest_for_split = _adapter._validate_aligned_manifest_for_split
SPLIT_PROTECTED_SHORT_RESPONSE_NORMALIZED = _guards.SPLIT_PROTECTED_SHORT_RESPONSE_NORMALIZED
SPLIT_SHORT_RESPONSE_MAX_DURATION_MS = _guards.SPLIT_SHORT_RESPONSE_MAX_DURATION_MS
SPLIT_SHORT_RESPONSE_MIN_DURATION_MS = _guards.SPLIT_SHORT_RESPONSE_MIN_DURATION_MS
SPLIT_SHORT_RESPONSE_MAX_DISTANCE_MS = _guards.SPLIT_SHORT_RESPONSE_MAX_DISTANCE_MS
SPLIT_SHORT_RESPONSE_ISOLATION_GAP_MS = _guards.SPLIT_SHORT_RESPONSE_ISOLATION_GAP_MS
SPLIT_CONTEXT_SENSITIVE_SHORT_RESPONSE_NORMALIZED = _guards.SPLIT_CONTEXT_SENSITIVE_SHORT_RESPONSE_NORMALIZED
run_split_stage = _stage_calls.run_split_stage
_postprocess_split_segments = _stage_calls.postprocess_split_segments
_extend_protected_short_display_segments = _stage_calls.extend_protected_short_display_segments
run_translate_stage = _stage_calls.run_translate_stage
_merge_translation_suspect_metadata = _stage_calls.merge_translation_suspect_metadata
_load_optimizer_types = _stage_calls.load_optimizer_types


def _validate_split_content_preserved(source_segments: list[Any], result_segments: list[Any]) -> None:
    _guards.validate_split_content_preserved(
        source_segments,
        result_segments,
        normalize_content=_normalize_content,
    )


def _extract_protected_short_responses(source_segments: list[Any], result_segments: list[Any]) -> list[Any]:
    return _guards.extract_protected_short_responses(
        source_segments,
        result_segments,
        normalize_content=_normalize_content,
        new_asr_data_seg=_new_asr_data_seg,
    )


def _segments_normalized_text(segments: list[Any]) -> str:
    return _guards.segments_normalized_text(segments, normalize_content=_normalize_content)


def _split_segment_by_protected_items(segment: Any, protected_items: list[dict[str, Any]]) -> list[Any]:
    return _guards.split_segment_by_protected_items(
        segment,
        protected_items,
        normalize_content=_normalize_content,
        new_asr_data_seg=_new_asr_data_seg,
    )


def _split_text_part_by_protected_item(part: dict[str, Any], item: dict[str, Any]) -> list[dict[str, Any]]:
    return _guards.split_text_part_by_protected_item(part, item, normalize_content=_normalize_content)


def _find_protected_text_span(text: str, protected_text: str) -> tuple[int, int] | None:
    return _guards.find_protected_text_span(text, protected_text, normalize_content=_normalize_content)


def _validate_split_short_responses_preserved(source_segments: list[Any], result_segments: list[Any]) -> None:
    _guards.validate_split_short_responses_preserved(
        source_segments,
        result_segments,
        normalize_content=_normalize_content,
    )


def _protected_short_response_segments(segments: list[Any], *, require_standalone: bool = True) -> list[dict[str, Any]]:
    return _guards.protected_short_response_segments(
        segments,
        require_standalone=require_standalone,
        normalize_content=_normalize_content,
    )


def _is_standalone_protected_short_response(
    segments: list[Any],
    index: int,
    normalized: str,
    text: str,
    start_ms: int,
    end_ms: int,
) -> bool:
    return _guards.is_standalone_protected_short_response(segments, index, normalized, text, start_ms, end_ms)


def _neighbor_end_ms(segments: list[Any], index: int) -> int | None:
    return _guards.neighbor_end_ms(segments, index)


def _neighbor_start_ms(segments: list[Any], index: int) -> int | None:
    return _guards.neighbor_start_ms(segments, index)


def _segment_time_ms(value: Any) -> int | None:
    return _guards.segment_time_ms(value)


def _segment_range_distance_ms(
    first_start_ms: int,
    first_end_ms: int,
    second_start_ms: int,
    second_end_ms: int,
) -> int:
    return _guards.segment_range_distance_ms(first_start_ms, first_end_ms, second_start_ms, second_end_ms)


def load_best_asr_data(
    work_paths: WorkPaths,
    optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT,
):
    ASRData, ASRDataSeg, *_ = _load_optimizer_types(optimizer_root)
    if work_paths.normalized_manifest.exists():
        return ASRData.from_json(read_json(work_paths.normalized_manifest, default={}))
    if work_paths.mimo_proofread_manifest.exists():
        return ASRData.from_json(read_json(work_paths.mimo_proofread_manifest, default={}))
    if work_paths.translated_manifest.exists():
        return ASRData.from_json(read_json(work_paths.translated_manifest, default={}))
    if work_paths.split_manifest.exists():
        return ASRData.from_json(read_json(work_paths.split_manifest, default={}))
    if work_paths.transcript_manifest.exists():
        return transcript_manifest_to_asr_data(work_paths, ASRData, ASRDataSeg)
    return None


def load_specific_asr_data(
    work_paths: WorkPaths,
    source: str,
    optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT,
):
    ASRData, ASRDataSeg, *_ = _load_optimizer_types(optimizer_root)
    if source == "normalized" and work_paths.normalized_manifest.exists():
        return ASRData.from_json(read_json(work_paths.normalized_manifest, default={}))
    if source == "mimo" and work_paths.mimo_proofread_manifest.exists():
        return ASRData.from_json(read_json(work_paths.mimo_proofread_manifest, default={}))
    if source == "translated" and work_paths.translated_manifest.exists():
        return ASRData.from_json(read_json(work_paths.translated_manifest, default={}))
    if source == "split" and work_paths.split_manifest.exists():
        return ASRData.from_json(read_json(work_paths.split_manifest, default={}))
    if source == "aligned" and work_paths.aligned_manifest.exists():
        return aligned_manifest_to_asr_data(work_paths, ASRData, ASRDataSeg)
    if source == "transcript" and work_paths.transcript_manifest.exists():
        return transcript_manifest_to_asr_data(work_paths, ASRData, ASRDataSeg)
    return None


def transcript_manifest_to_asr_data(work_paths: WorkPaths, ASRData: Any, ASRDataSeg: Any):
    transcript_payload = read_json(work_paths.transcript_manifest, default=[])
    segments = []
    for item in transcript_payload:
        if item.get("status") != "completed":
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        start_ms = int(round(float(item.get("global_start_time", 0.0)) * 1000))
        end_ms = int(round(float(item.get("global_end_time", 0.0)) * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        segments.append(ASRDataSeg(text=text, start_time=start_ms, end_time=end_ms))
    return ASRData(segments)
