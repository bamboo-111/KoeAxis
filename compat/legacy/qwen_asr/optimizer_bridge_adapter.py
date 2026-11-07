from __future__ import annotations

import logging
import unicodedata
from typing import Any

from qwen_asr.align import validate_aligned_token_timing
from qwen_asr.models import AlignedToken, WorkPaths
from qwen_asr.storage import read_json

LOGGER = logging.getLogger(__name__)

ALIGN_LOCAL_INTERPOLATION_MAX_GAP_MS = 800
ALIGN_ZERO_TOKEN_DEFAULT_DURATION_MS = 160
ALIGN_ZERO_TOKEN_MAX_DURATION_MS = 500
FALLBACK_SHORT_TEXT_MAX_NORMALIZED_CHARS = 4
FALLBACK_SHORT_TEXT_MIN_DURATION_MS = 500
FALLBACK_SHORT_TEXT_MAX_DURATION_MS = 900
FALLBACK_SHORT_TEXT_LONG_DURATION_MS = 3000
FALLBACK_SHORT_RESPONSE_NORMALIZED = {
    "\u306f\u3044",
    "\u3046\u3093",
    "\u3046\u3046\u3093",
    "\u3048",
    "\u3042",
    "\u3044\u3084",
    "\u3044\u3044\u3048",
    "\u3060\u3081",
    "\u30c0\u30e1",
}


def aligned_manifest_to_asr_data(work_paths: WorkPaths, ASRData: Any, ASRDataSeg: Any):
    aligned_payload = read_json(work_paths.aligned_manifest, default=[])
    transcript_payload = read_json(work_paths.transcript_manifest, default=[])
    transcript_by_id = {
        str(item.get("segment_id")): item
        for item in transcript_payload
        if isinstance(item, dict)
    }
    chunks: list[dict[str, Any]] = []
    failures: list[str] = []
    for index, item in enumerate(aligned_payload):
        owned_start_ms, owned_end_ms = _owned_segment_bounds_ms(aligned_payload, index)
        segment_start_ms = int(round(float(item.get("global_start_time", 0.0)) * 1000))
        segment_end_ms = int(round(float(item.get("global_end_time", 0.0)) * 1000))
        if item.get("status") != "completed":
            segment_id = str(item.get("segment_id", "<unknown>"))
            transcript = transcript_by_id.get(segment_id)
            if not transcript:
                failures.append(f"{segment_id}: status={item.get('status')} error={item.get('error')}")
                continue
            text = str(transcript.get("text", "")).strip()
            if not text:
                failures.append(f"{segment_id}: status={item.get('status')} empty transcript")
                continue
            if owned_end_ms <= owned_start_ms:
                failures.append(f"{segment_id}: transcript timing is not positive")
                continue
            chunks.append(
                {
                    "segment_id": segment_id,
                    "segment_start_ms": segment_start_ms,
                    "segment_end_ms": segment_end_ms,
                    "owned_start_ms": owned_start_ms,
                    "owned_end_ms": owned_end_ms,
                    "pieces": [{"text": text, "start_ms": owned_start_ms, "end_ms": owned_end_ms}],
                }
            )
            LOGGER.warning(
                "Using transcript-level timing for %s because forced alignment failed: %s",
                segment_id,
                item.get("error"),
            )
            continue
        tokens = [
            AlignedToken(
                text=str(token.get("text", "")),
                start_time=float(token.get("start_time", 0.0)),
                end_time=float(token.get("end_time", token.get("start_time", 0.0))),
            )
            for token in item.get("tokens", [])
            if str(token.get("text", "")).strip()
        ]
        segment_id = str(item.get("segment_id", "<unknown>"))
        transcript = transcript_by_id.get(segment_id)
        transcript_text = str(transcript.get("text", "")).strip() if transcript else ""
        error = validate_aligned_token_timing(
            tokens,
            float(item.get("global_start_time", 0.0)),
            float(item.get("global_end_time", 0.0)),
        )
        if error:
            if transcript_text and owned_end_ms > owned_start_ms:
                chunks.append(
                    {
                        "segment_id": segment_id,
                        "segment_start_ms": segment_start_ms,
                        "segment_end_ms": segment_end_ms,
                        "owned_start_ms": owned_start_ms,
                        "owned_end_ms": owned_end_ms,
                        "pieces": [
                            {
                                "text": transcript_text,
                                "start_ms": owned_start_ms,
                                "end_ms": owned_end_ms,
                            }
                        ],
                    }
                )
                LOGGER.warning(
                    "Using transcript-level timing for %s because completed alignment timing is unreliable: %s",
                    segment_id,
                    error,
                )
                continue
            failures.append(f"{segment_id}: {error}")
            continue
        aligned_text = "".join(token.text for token in tokens)
        if transcript_text and _normalize_content(transcript_text) != _normalize_content(aligned_text):
            chunks.append(
                {
                    "segment_id": segment_id,
                    "segment_start_ms": segment_start_ms,
                    "segment_end_ms": segment_end_ms,
                    "owned_start_ms": owned_start_ms,
                    "owned_end_ms": owned_end_ms,
                    "pieces": [
                        {
                            "text": transcript_text,
                            "start_ms": owned_start_ms,
                            "end_ms": owned_end_ms,
                        }
                    ],
                }
            )
            LOGGER.warning(
                "Using transcript-level timing for %s because completed alignment changed content",
                segment_id,
            )
            continue
        pieces: list[dict[str, Any]] = []
        for token in item.get("tokens", []):
            text = str(token.get("text", "")).strip()
            if not text:
                continue
            start_ms = int(round(float(token.get("start_time", 0.0)) * 1000))
            end_ms = int(round(float(token.get("end_time", 0.0)) * 1000))
            pieces.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
        if transcript_text:
            pieces = _restore_transcript_surface_to_pieces(transcript_text, pieces)
        chunks.append(
            {
                "segment_id": segment_id,
                "segment_start_ms": segment_start_ms,
                "segment_end_ms": segment_end_ms,
                "owned_start_ms": owned_start_ms,
                "owned_end_ms": owned_end_ms,
                "pieces": pieces,
            }
        )
    if failures:
        preview = "; ".join(failures[:5])
        more = f"; +{len(failures) - 5} more" if len(failures) > 5 else ""
        raise RuntimeError(f"Alignment timing is unreliable and cannot fall back to transcript timing: {preview}{more}")
    segments = _chunks_to_asr_segments(chunks, ASRDataSeg)
    return ASRData(segments)


def _owned_segment_bounds_ms(
    aligned_payload: list[dict[str, Any]],
    index: int,
) -> tuple[int, int]:
    item = aligned_payload[index]
    start_ms = int(round(float(item.get("global_start_time", 0.0)) * 1000))
    end_ms = int(round(float(item.get("global_end_time", 0.0)) * 1000))
    owned_start_ms = start_ms
    owned_end_ms = end_ms

    if index > 0:
        previous_end_ms = int(
            round(float(aligned_payload[index - 1].get("global_end_time", 0.0)) * 1000)
        )
        if previous_end_ms > start_ms:
            owned_start_ms = (previous_end_ms + start_ms) // 2
    if index + 1 < len(aligned_payload):
        next_start_ms = int(
            round(float(aligned_payload[index + 1].get("global_start_time", 0.0)) * 1000)
        )
        if end_ms > next_start_ms:
            owned_end_ms = (end_ms + next_start_ms) // 2

    if owned_end_ms <= owned_start_ms:
        owned_end_ms = owned_start_ms + 1
    return owned_start_ms, owned_end_ms


def _sanitized_aligned_payload(
    source_segments: list[Any],
    reference_payload: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not source_segments:
        return []
    if not reference_payload:
        start_ms = min(int(segment.start_time) for segment in source_segments)
        end_ms = max(int(segment.end_time) for segment in source_segments)
        return [_sanitized_payload_item("sanitized_000001", start_ms, end_ms, source_segments)]

    result: list[dict[str, Any]] = []
    cursor = 0
    for index, item in enumerate(reference_payload):
        owned_start_ms, owned_end_ms = _owned_segment_bounds_ms(reference_payload, index)
        is_last = index == len(reference_payload) - 1
        grouped: list[Any] = []
        while cursor < len(source_segments):
            segment = source_segments[cursor]
            midpoint_ms = (int(segment.start_time) + int(segment.end_time)) // 2
            if not is_last and midpoint_ms >= owned_end_ms:
                break
            grouped.append(segment)
            cursor += 1
        if grouped:
            result.append(
                _sanitized_payload_item(
                    str(item.get("segment_id") or f"sanitized_{index + 1:06d}"),
                    owned_start_ms,
                    owned_end_ms,
                    grouped,
                )
            )

    if cursor < len(source_segments):
        remaining = source_segments[cursor:]
        if result:
            last = result[-1]
            last["tokens"].extend(_asr_segments_to_tokens(remaining))
            last["text"] = "".join(str(token["text"]) for token in last["tokens"])
            last["global_end_time"] = max(
                float(last["global_end_time"]),
                max(int(segment.end_time) for segment in remaining) / 1000.0,
            )
        else:
            start_ms = min(int(segment.start_time) for segment in remaining)
            end_ms = max(int(segment.end_time) for segment in remaining)
            result.append(_sanitized_payload_item("sanitized_000001", start_ms, end_ms, remaining))
    return result


def _sanitized_payload_item(
    segment_id: str,
    start_ms: int,
    end_ms: int,
    segments: list[Any],
) -> dict[str, Any]:
    tokens = _asr_segments_to_tokens(segments)
    return {
        "segment_id": segment_id,
        "global_start_time": start_ms / 1000.0,
        "global_end_time": end_ms / 1000.0,
        "text": "".join(str(token["text"]) for token in tokens),
        "status": "completed",
        "tokens": tokens,
    }


def _asr_segments_to_tokens(segments: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "text": str(segment.text),
            "start_time": int(segment.start_time) / 1000.0,
            "end_time": int(segment.end_time) / 1000.0,
        }
        for segment in segments
        if str(segment.text).strip()
    ]


def _chunks_to_asr_segments(chunks: list[dict[str, Any]], ASRDataSeg: Any) -> list[Any]:
    result: list[Any] = []
    previous_text = ""
    previous_end_ms: int | None = None
    previous_segment_end_ms: int | None = None

    for chunk in chunks:
        pieces = list(chunk["pieces"])
        segment_start_ms = int(chunk["segment_start_ms"])
        segment_end_ms = int(chunk["segment_end_ms"])
        overlaps_previous = (
            previous_segment_end_ms is not None
            and previous_segment_end_ms > segment_start_ms
        )
        if overlaps_previous and previous_text:
            pieces = _remove_exact_boundary_duplicate(previous_text, pieces)

        ranges = _repair_piece_ranges(
            pieces,
            int(chunk["owned_start_ms"]),
            int(chunk["owned_end_ms"]),
        )
        ranges = _narrow_overlong_short_text_ranges(pieces, ranges)
        for piece, (start_ms, end_ms) in zip(pieces, ranges, strict=True):
            if previous_end_ms is not None and start_ms < previous_end_ms:
                start_ms = previous_end_ms
            if end_ms <= start_ms:
                end_ms = start_ms + 1
            result.append(
                _new_asr_data_seg(
                    ASRDataSeg,
                    text=str(piece["text"]),
                    translated_text="",
                    start_time=start_ms,
                    end_time=end_ms,
                )
            )
            previous_end_ms = end_ms

        chunk_text = "".join(str(piece["text"]) for piece in pieces)
        if chunk_text:
            previous_text = chunk_text
        previous_segment_end_ms = segment_end_ms
    return result


def _remove_exact_boundary_duplicate(
    previous_text: str,
    pieces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous = _normalize_content(previous_text)
    if not previous:
        return pieces

    best_count = 0
    prefix = ""
    for count, piece in enumerate(pieces[:40], 1):
        prefix += _normalize_content(str(piece.get("text", "")))
        if not prefix or len(prefix) > 80:
            break
        if previous.endswith(prefix):
            best_count = count
    return pieces[best_count:]


def _narrow_overlong_short_text_ranges(
    pieces: list[dict[str, Any]],
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not pieces or len(pieces) != len(ranges):
        return ranges

    narrowed: list[tuple[int, int]] = []
    for piece, (start_ms, end_ms) in zip(pieces, ranges, strict=True):
        duration = end_ms - start_ms
        normalized_length = len(_normalize_content(str(piece.get("text", ""))))
        if (
            _is_fallback_short_response_text(str(piece.get("text", "")))
            and duration > FALLBACK_SHORT_TEXT_LONG_DURATION_MS
        ):
            target_duration = min(
                FALLBACK_SHORT_TEXT_MAX_DURATION_MS,
                max(FALLBACK_SHORT_TEXT_MIN_DURATION_MS, normalized_length * ALIGN_ZERO_TOKEN_DEFAULT_DURATION_MS),
            )
            center = start_ms + duration // 2
            new_start = max(start_ms, center - target_duration // 2)
            new_end = min(end_ms, new_start + target_duration)
            if (
                new_end - new_start < FALLBACK_SHORT_TEXT_MIN_DURATION_MS
                and end_ms - start_ms >= FALLBACK_SHORT_TEXT_MIN_DURATION_MS
            ):
                new_end = min(end_ms, new_start + FALLBACK_SHORT_TEXT_MIN_DURATION_MS)
                new_start = max(start_ms, new_end - FALLBACK_SHORT_TEXT_MIN_DURATION_MS)
            narrowed.append((new_start, max(new_start + 1, new_end)))
            continue
        narrowed.append((start_ms, end_ms))
    return narrowed


def _is_fallback_short_response_text(text: str) -> bool:
    normalized = _normalize_content(text)
    return (
        bool(normalized)
        and len(normalized) <= FALLBACK_SHORT_TEXT_MAX_NORMALIZED_CHARS
        and normalized in FALLBACK_SHORT_RESPONSE_NORMALIZED
    )


def _repair_piece_ranges(
    pieces: list[dict[str, Any]],
    owned_start_ms: int,
    owned_end_ms: int,
) -> list[tuple[int, int]]:
    if not pieces:
        return []

    owned_end_ms = max(owned_end_ms, owned_start_ms + len(pieces))
    raw_ranges = [_raw_piece_range(piece) for piece in pieces]
    repaired: list[tuple[int, int] | None] = [None] * len(pieces)

    for index, (start_ms, end_ms) in enumerate(raw_ranges):
        if end_ms > start_ms:
            start_ms = max(owned_start_ms, min(start_ms, owned_end_ms - 1))
            end_ms = max(start_ms + 1, min(end_ms, owned_end_ms))
            repaired[index] = (start_ms, end_ms)

    index = 0
    while index < len(pieces):
        if repaired[index] is not None:
            index += 1
            continue
        run_start = index
        while index < len(pieces) and repaired[index] is None:
            index += 1
        run_end = index - 1
        _fill_missing_range_run(
            repaired,
            run_start,
            run_end,
            owned_start_ms,
            owned_end_ms,
        )

    return _enforce_monotonic_ranges(repaired, owned_start_ms, owned_end_ms)


def _raw_piece_range(piece: dict[str, Any]) -> tuple[int, int]:
    start_ms = int(piece.get("start_ms", 0))
    end_ms = int(piece.get("end_ms", start_ms))
    return start_ms, end_ms


def _fill_missing_range_run(
    repaired: list[tuple[int, int] | None],
    run_start: int,
    run_end: int,
    owned_start_ms: int,
    owned_end_ms: int,
) -> None:
    left = repaired[run_start - 1] if run_start > 0 else None
    right = repaired[run_end + 1] if run_end + 1 < len(repaired) else None
    count = run_end - run_start + 1

    if left and right:
        gap_start = left[1]
        gap_end = right[0]
        gap = gap_end - gap_start
        if gap >= count and gap <= ALIGN_LOCAL_INTERPOLATION_MAX_GAP_MS:
            cursor = gap_start
            for index in range(run_start, run_end + 1):
                remaining = run_end - index + 1
                step = max(1, round((gap_end - cursor) / remaining))
                end_ms = min(gap_end - (remaining - 1), cursor + step)
                repaired[index] = (cursor, max(cursor + 1, end_ms))
                cursor = repaired[index][1]
            return

    cursor = left[1] if left else owned_start_ms
    limit = right[0] if right else owned_end_ms
    duration = min(
        ALIGN_ZERO_TOKEN_MAX_DURATION_MS,
        max(1, ALIGN_ZERO_TOKEN_DEFAULT_DURATION_MS),
    )
    for index in range(run_start, run_end + 1):
        remaining = run_end - index
        latest_end = max(cursor + 1, limit - remaining)
        end_ms = min(cursor + duration, latest_end)
        if right and end_ms > right[0] - remaining:
            end_ms = right[0] - remaining
        repaired[index] = (cursor, max(cursor + 1, end_ms))
        cursor = repaired[index][1]


def _enforce_monotonic_ranges(
    repaired: list[tuple[int, int] | None],
    owned_start_ms: int,
    owned_end_ms: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = owned_start_ms
    total = len(repaired)
    for index, value in enumerate(repaired):
        remaining = total - index - 1
        latest_end = max(cursor + 1, owned_end_ms - remaining)
        if value is None:
            start_ms = cursor
            end_ms = min(cursor + ALIGN_ZERO_TOKEN_DEFAULT_DURATION_MS, latest_end)
        else:
            start_ms = max(cursor, min(value[0], latest_end - 1))
            end_ms = max(start_ms + 1, min(value[1], latest_end))
        ranges.append((start_ms, end_ms))
        cursor = end_ms
    return ranges


def _normalize_content(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _restore_transcript_surface_to_pieces(
    transcript_text: str,
    pieces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not pieces:
        return pieces
    targets = [_normalize_content(str(piece.get("text", ""))) for piece in pieces]
    if not all(targets) or "".join(targets) != _normalize_content(transcript_text):
        return pieces

    restored: list[dict[str, Any]] = []
    cursor = 0
    for target, piece in zip(targets, pieces, strict=True):
        start = cursor
        matched = ""
        while cursor < len(transcript_text) and len(matched) < len(target):
            matched += _normalize_content(transcript_text[cursor])
            cursor += 1
            if not target.startswith(matched):
                return pieces
        if matched != target:
            return pieces
        while cursor < len(transcript_text) and not _normalize_content(transcript_text[cursor]):
            cursor += 1
        surface = transcript_text[start:cursor]
        if not surface.strip():
            return pieces
        restored.append({**piece, "text": surface})

    if cursor != len(transcript_text):
        return pieces
    return restored


def _new_asr_data_seg(
    ASRDataSeg: Any,
    *,
    text: str,
    translated_text: str,
    start_time: int,
    end_time: int,
) -> Any:
    try:
        return ASRDataSeg(
            text=text,
            translated_text=translated_text,
            start_time=start_time,
            end_time=end_time,
        )
    except TypeError:
        return ASRDataSeg(text=text, start_time=start_time, end_time=end_time)


def _validate_aligned_manifest_for_split(work_paths: WorkPaths) -> None:
    aligned_payload = read_json(work_paths.aligned_manifest, default=[])
    if not aligned_payload:
        raise RuntimeError("aligned_segments.json is missing or empty. Run align first.")

    failures: list[str] = []
    for item in aligned_payload:
        segment_id = str(item.get("segment_id", "<unknown>"))
        if item.get("status") != "completed":
            failures.append(f"{segment_id}: status={item.get('status')} error={item.get('error')}")
            continue
        tokens = [
            AlignedToken(
                text=str(token.get("text", "")),
                start_time=float(token.get("start_time", 0.0)),
                end_time=float(token.get("end_time", token.get("start_time", 0.0))),
            )
            for token in item.get("tokens", [])
            if str(token.get("text", "")).strip()
        ]
        error = validate_aligned_token_timing(
            tokens,
            float(item.get("global_start_time", 0.0)),
            float(item.get("global_end_time", 0.0)),
        )
        if error:
            failures.append(f"{segment_id}: {error}")

    if failures:
        preview = "; ".join(failures[:5])
        more = f"; +{len(failures) - 5} more" if len(failures) > 5 else ""
        raise RuntimeError(f"Alignment timing is unreliable. Re-run/fix align before split: {preview}{more}")
