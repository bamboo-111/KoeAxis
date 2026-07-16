from __future__ import annotations

import logging
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)

SPLIT_PROTECTED_SHORT_RESPONSE_NORMALIZED = {
    "\u306f\u3044",
    "\u3046\u3093",
    "\u3046\u3046\u3093",
    "\u3048",
    "\u3044\u3084",
    "\u3044\u3044\u3048",
    "\u3060\u3081",
    "\u30c0\u30e1",
}
SPLIT_SHORT_RESPONSE_MAX_DURATION_MS = 1800
SPLIT_SHORT_RESPONSE_MIN_DURATION_MS = 120
SPLIT_SHORT_RESPONSE_MAX_DISTANCE_MS = 2500
SPLIT_SHORT_RESPONSE_ISOLATION_GAP_MS = 500
SPLIT_CONTEXT_SENSITIVE_SHORT_RESPONSE_NORMALIZED = {
    "\u306f\u3044",
    "\u3044\u3084",
    "\u3060\u3081",
    "\u30c0\u30e1",
}


def validate_split_content_preserved(
    source_segments: list[Any],
    result_segments: list[Any],
    *,
    normalize_content: Callable[[str], str],
) -> None:
    source_text = normalize_content("".join(str(segment.text) for segment in source_segments))
    result_text = normalize_content("".join(str(segment.text) for segment in result_segments))
    if source_text != result_text:
        raise RuntimeError(
            "Split stage changed aligned text content: "
            f"source_chars={len(source_text)} result_chars={len(result_text)}"
        )


def extract_protected_short_responses(
    source_segments: list[Any],
    result_segments: list[Any],
    *,
    normalize_content: Callable[[str], str],
    new_asr_data_seg: Callable[..., Any],
) -> list[Any]:
    protected = protected_short_response_segments(source_segments, normalize_content=normalize_content)
    if not protected or not result_segments:
        return result_segments
    output: list[Any] = []
    cursor = 0
    for segment in result_segments:
        segment_start = segment_time_ms(getattr(segment, "start_time", None))
        segment_end = segment_time_ms(getattr(segment, "end_time", None))
        if segment_start is None or segment_end is None or segment_end <= segment_start:
            output.append(segment)
            continue
        contained: list[dict[str, Any]] = []
        while cursor < len(protected) and protected[cursor]["end_ms"] <= segment_start:
            cursor += 1
        scan = cursor
        while scan < len(protected) and protected[scan]["start_ms"] < segment_end:
            item = protected[scan]
            if item["start_ms"] >= segment_start and item["end_ms"] <= segment_end:
                contained.append(item)
            scan += 1
        split_segments = split_segment_by_protected_items(
            segment,
            contained,
            normalize_content=normalize_content,
            new_asr_data_seg=new_asr_data_seg,
        )
        output.extend(split_segments)
    if segments_normalized_text(output, normalize_content=normalize_content) != segments_normalized_text(
        result_segments,
        normalize_content=normalize_content,
    ):
        LOGGER.warning("Protected short-response extraction skipped because it would change split text content.")
        return result_segments
    return output


def segments_normalized_text(segments: list[Any], *, normalize_content: Callable[[str], str]) -> str:
    return normalize_content("".join(str(getattr(segment, "text", "")) for segment in segments))


def split_segment_by_protected_items(
    segment: Any,
    protected_items: list[dict[str, Any]],
    *,
    normalize_content: Callable[[str], str],
    new_asr_data_seg: Callable[..., Any],
) -> list[Any]:
    if not protected_items:
        return [segment]
    segment_start = segment_time_ms(getattr(segment, "start_time", None))
    segment_end = segment_time_ms(getattr(segment, "end_time", None))
    if segment_start is None or segment_end is None:
        return [segment]
    parts = [{"text": str(getattr(segment, "text", "")), "start_ms": segment_start, "end_ms": segment_end}]
    changed = False
    for item in protected_items:
        next_parts: list[dict[str, Any]] = []
        for part in parts:
            split_parts = split_text_part_by_protected_item(part, item, normalize_content=normalize_content)
            if len(split_parts) > 1 or split_parts[0] != part:
                changed = True
            next_parts.extend(split_parts)
        parts = next_parts
    if not changed:
        return [segment]
    segment_class = segment.__class__
    result: list[Any] = []
    for part in parts:
        text = str(part["text"])
        if not text.strip():
            continue
        start_ms = int(part["start_ms"])
        end_ms = int(part["end_ms"])
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        result.append(new_asr_data_seg(segment_class, text=text, translated_text="", start_time=start_ms, end_time=end_ms))
    return result or [segment]


def split_text_part_by_protected_item(
    part: dict[str, Any],
    item: dict[str, Any],
    *,
    normalize_content: Callable[[str], str],
) -> list[dict[str, Any]]:
    start_ms = int(part["start_ms"])
    end_ms = int(part["end_ms"])
    item_start = int(item["start_ms"])
    item_end = int(item["end_ms"])
    if item_start < start_ms or item_end > end_ms:
        return [part]
    text = str(part["text"])
    span = find_protected_text_span(text, str(item["text"]), normalize_content=normalize_content)
    if span is None:
        return [part]
    before_text = text[: span[0]]
    protected_text = text[span[0] : span[1]]
    after_text = text[span[1] :]
    pieces: list[dict[str, Any]] = []
    if before_text.strip() and item_start > start_ms:
        pieces.append({"text": before_text, "start_ms": start_ms, "end_ms": item_start})
    pieces.append({"text": protected_text, "start_ms": item_start, "end_ms": max(item_start + 1, item_end)})
    if after_text.strip() and end_ms > item_end:
        pieces.append({"text": after_text, "start_ms": item_end, "end_ms": end_ms})
    return pieces


def find_protected_text_span(
    text: str,
    protected_text: str,
    *,
    normalize_content: Callable[[str], str],
) -> tuple[int, int] | None:
    normalized_target = normalize_content(protected_text)
    if not normalized_target:
        return None
    for start in range(len(text)):
        normalized = ""
        content_start: int | None = None
        for end in range(start, len(text)):
            character_normalized = normalize_content(text[end])
            if character_normalized and content_start is None:
                content_start = end
            normalized += character_normalized
            if not normalized or content_start is None:
                continue
            if normalized == normalized_target:
                span_end = end + 1
                while span_end < len(text) and normalize_content(text[span_end]) == "":
                    span_end += 1
                return content_start, span_end
            if not normalized_target.startswith(normalized):
                break
    return None


def validate_split_short_responses_preserved(
    source_segments: list[Any],
    result_segments: list[Any],
    *,
    normalize_content: Callable[[str], str],
) -> None:
    source_items = protected_short_response_segments(source_segments, normalize_content=normalize_content)
    if not source_items:
        return
    result_items = protected_short_response_segments(
        result_segments,
        require_standalone=False,
        normalize_content=normalize_content,
    )
    used_result_indexes: set[int] = set()
    failures: list[str] = []
    for source in source_items:
        best_index: int | None = None
        best_distance: int | None = None
        for index, result in enumerate(result_items):
            if index in used_result_indexes or result["normalized"] != source["normalized"]:
                continue
            distance = segment_range_distance_ms(
                source["start_ms"],
                source["end_ms"],
                result["start_ms"],
                result["end_ms"],
            )
            if distance > SPLIT_SHORT_RESPONSE_MAX_DISTANCE_MS:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is None:
            failures.append(
                f"{source['text']}@{source['start_ms']}-{source['end_ms']}"
            )
            continue
        used_result_indexes.add(best_index)
    if failures:
        preview = "; ".join(failures[:5])
        more = f"; +{len(failures) - 5} more" if len(failures) > 5 else ""
        raise RuntimeError(
            "Split stage moved or swallowed protected short responses: "
            f"{preview}{more}"
        )


def protected_short_response_segments(
    segments: list[Any],
    *,
    require_standalone: bool = True,
    normalize_content: Callable[[str], str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        text = str(getattr(segment, "text", ""))
        normalized = normalize_content(text)
        if normalized not in SPLIT_PROTECTED_SHORT_RESPONSE_NORMALIZED:
            continue
        start_ms = segment_time_ms(getattr(segment, "start_time", None))
        end_ms = segment_time_ms(getattr(segment, "end_time", None))
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue
        if end_ms - start_ms < SPLIT_SHORT_RESPONSE_MIN_DURATION_MS:
            continue
        if end_ms - start_ms > SPLIT_SHORT_RESPONSE_MAX_DURATION_MS:
            continue
        if require_standalone and not is_standalone_protected_short_response(
            segments,
            index,
            normalized,
            text,
            start_ms,
            end_ms,
        ):
            continue
        result.append(
            {
                "text": text,
                "normalized": normalized,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
    return result


def is_standalone_protected_short_response(
    segments: list[Any],
    index: int,
    normalized: str,
    text: str,
    start_ms: int,
    end_ms: int,
) -> bool:
    if normalized not in SPLIT_CONTEXT_SENSITIVE_SHORT_RESPONSE_NORMALIZED:
        return True
    previous_end = neighbor_end_ms(segments, index - 1)
    next_start = neighbor_start_ms(segments, index + 1)
    isolated_left = previous_end is None or start_ms - previous_end >= SPLIT_SHORT_RESPONSE_ISOLATION_GAP_MS
    isolated_right = next_start is None or next_start - end_ms >= SPLIT_SHORT_RESPONSE_ISOLATION_GAP_MS
    if not isolated_left:
        return False
    if any(mark in text for mark in ("。", "！", "？", "!", "?")):
        return True
    return isolated_left and isolated_right


def neighbor_end_ms(segments: list[Any], index: int) -> int | None:
    if index < 0 or index >= len(segments):
        return None
    return segment_time_ms(getattr(segments[index], "end_time", None))


def neighbor_start_ms(segments: list[Any], index: int) -> int | None:
    if index < 0 or index >= len(segments):
        return None
    return segment_time_ms(getattr(segments[index], "start_time", None))


def segment_time_ms(value: Any) -> int | None:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def segment_range_distance_ms(
    first_start_ms: int,
    first_end_ms: int,
    second_start_ms: int,
    second_end_ms: int,
) -> int:
    if min(first_end_ms, second_end_ms) > max(first_start_ms, second_start_ms):
        return 0
    if first_end_ms <= second_start_ms:
        return second_start_ms - first_end_ms
    return first_start_ms - second_end_ms
