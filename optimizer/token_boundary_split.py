from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Sequence

from optimizer.asr_data import ASRData, ASRDataSeg
from optimizer.exceptions import LLMConnectionError, LLMError, LLMRateLimitError, LLMResponseError
from optimizer.llm_client import DEFAULT_TIMEOUT, call_llm
from optimizer.prompts import get_prompt
from optimizer.split_by_llm import CONNECTION_BASE_DELAY, MAX_DELAY, RATE_LIMIT_BASE_DELAY, BACKOFF_FACTOR
from optimizer.text_utils import count_words, is_mainly_cjk

LOGGER = logging.getLogger("optimizer.token_boundary_split")

BLOCK_SIZE = 16
MAX_TICK = 200
MAX_STEPS = 3
BAD_ZERO_RATIO = 0.25
BAD_ZERO_RUN = 8


@dataclass(slots=True)
class TokenBoundaryRecord:
    token_id: str
    text: str
    start_ms: int
    end_ms: int
    duration_tick: int | None
    gap_tick: int | None


def split_aligned_payload_by_token_boundaries(
    aligned_payload: Sequence[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    disable_thinking: bool = False,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    thread_num: int = 1,
) -> ASRData:
    tasks: list[tuple[int, dict[str, Any], list[TokenBoundaryRecord]]] = []
    for item in aligned_payload:
        if item.get("status") != "completed":
            continue
        records = _records_from_aligned_item(item)
        if not records:
            continue
        tasks.append((len(tasks), item, records))

    if not tasks:
        return ASRData([])

    workers = max(1, int(thread_num))
    results: dict[int, list[ASRDataSeg]] = {}

    def run_task(task: tuple[int, dict[str, Any], list[TokenBoundaryRecord]]) -> tuple[int, list[ASRDataSeg]]:
        index, item, records = task
        end_ids = _request_token_boundaries(
            records,
            segment_id=str(item.get("segment_id", "")),
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
        return index, _records_to_segments(records, end_ids)

    LOGGER.info("token-boundary split starting: segments=%d workers=%d", len(tasks), workers)
    if workers == 1:
        for task in tasks:
            index, segments = run_task(task)
            results[index] = segments
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            future_map = {executor.submit(run_task, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                index, segments = future.result()
                results[index] = segments

    result: list[ASRDataSeg] = []
    for index in range(len(tasks)):
        result.extend(results[index])
    return ASRData(_clamp_overlapping_segments(result))


def split_aligned_payload_by_token_delimited_text(
    aligned_payload: Sequence[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    disable_thinking: bool = False,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    thread_num: int = 1,
) -> ASRData:
    tasks: list[tuple[int, dict[str, Any], list[TokenBoundaryRecord]]] = []
    for item in aligned_payload:
        if item.get("status") != "completed":
            continue
        records = _records_from_aligned_item(item)
        if records:
            tasks.append((len(tasks), item, records))

    if not tasks:
        return ASRData([])

    workers = max(1, int(thread_num))
    results: dict[int, list[ASRDataSeg]] = {}

    def run_task(task: tuple[int, dict[str, Any], list[TokenBoundaryRecord]]) -> tuple[int, list[ASRDataSeg]]:
        index, item, records = task
        groups = _request_token_delimited_split(
            records,
            segment_id=str(item.get("segment_id", "")),
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
        return index, _token_groups_to_segments(records, groups)

    LOGGER.info("token-delimited split starting: segments=%d workers=%d", len(tasks), workers)
    if workers == 1:
        for task in tasks:
            index, segments = run_task(task)
            results[index] = segments
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            future_map = {executor.submit(run_task, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                index, segments = future.result()
                results[index] = segments

    result: list[ASRDataSeg] = []
    for index in range(len(tasks)):
        result.extend(results[index])
    return ASRData(_clamp_overlapping_segments(result))


def split_aligned_payload_by_token_counts(
    aligned_payload: Sequence[dict[str, Any]],
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    disable_thinking: bool = False,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    thread_num: int = 1,
) -> ASRData:
    tasks: list[tuple[int, dict[str, Any], list[TokenBoundaryRecord]]] = []
    for item in aligned_payload:
        if item.get("status") != "completed":
            continue
        records = _records_from_aligned_item(item)
        if records:
            tasks.append((len(tasks), item, records))

    if not tasks:
        return ASRData([])

    workers = max(1, int(thread_num))
    results: dict[int, list[ASRDataSeg]] = {}

    def run_task(task: tuple[int, dict[str, Any], list[TokenBoundaryRecord]]) -> tuple[int, list[ASRDataSeg]]:
        index, item, records = task
        counts = _request_token_counts_split(
            records,
            segment_id=str(item.get("segment_id", "")),
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
        return index, _token_counts_to_segments(records, counts)

    LOGGER.info("token-counts split starting: segments=%d workers=%d", len(tasks), workers)
    if workers == 1:
        for task in tasks:
            index, segments = run_task(task)
            results[index] = segments
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
            future_map = {executor.submit(run_task, task): task[0] for task in tasks}
            for future in as_completed(future_map):
                index, segments = future.result()
                results[index] = segments

    result: list[ASRDataSeg] = []
    for index in range(len(tasks)):
        result.extend(results[index])
    return ASRData(_clamp_overlapping_segments(result))


def _records_from_aligned_item(item: dict[str, Any]) -> list[TokenBoundaryRecord]:
    tokens = [token for token in item.get("tokens", []) if str(token.get("text", "")).strip()]
    if not tokens:
        return []

    segment_start_ms = int(round(float(item.get("global_start_time", 0.0)) * 1000))
    segment_end_ms = int(round(float(item.get("global_end_time", 0.0)) * 1000))
    raw_ranges: list[tuple[int, int]] = []
    zero_run = 0
    max_zero_run = 0
    zero_count = 0
    for token in tokens:
        start_ms = int(round(float(token.get("start_time", 0.0)) * 1000))
        end_ms = int(round(float(token.get("end_time", 0.0)) * 1000))
        raw_ranges.append((start_ms, end_ms))
        if end_ms <= start_ms:
            zero_count += 1
            zero_run += 1
            max_zero_run = max(max_zero_run, zero_run)
        else:
            zero_run = 0

    timing_unreliable = (zero_count / max(len(tokens), 1)) > BAD_ZERO_RATIO or max_zero_run > BAD_ZERO_RUN
    repaired_ranges = _proportional_ranges(tokens, segment_start_ms, segment_end_ms) if timing_unreliable else _repair_ranges(raw_ranges, segment_start_ms, segment_end_ms)

    records: list[TokenBoundaryRecord] = []
    for index, token in enumerate(tokens):
        start_ms, end_ms = repaired_ranges[index]
        next_start = repaired_ranges[index + 1][0] if index + 1 < len(repaired_ranges) else None
        duration_tick = None if timing_unreliable or raw_ranges[index][1] <= raw_ranges[index][0] else _clip_tick(round((end_ms - start_ms) / 10))
        gap_tick = None
        if next_start is not None and not timing_unreliable:
            gap_tick = _clip_tick(round(max(0, next_start - end_ms) / 10))
        records.append(
            TokenBoundaryRecord(
                token_id=_format_token_id(index),
                text=str(token.get("text", "")).strip(),
                start_ms=start_ms,
                end_ms=end_ms,
                duration_tick=duration_tick,
                gap_tick=gap_tick,
            )
        )
    return records


def _repair_ranges(raw_ranges: list[tuple[int, int]], segment_start_ms: int, segment_end_ms: int) -> list[tuple[int, int]]:
    repaired: list[tuple[int, int]] = []
    count = len(raw_ranges)
    fallback = _proportional_ranges([{} for _ in raw_ranges], segment_start_ms, segment_end_ms)
    for index, (start_ms, end_ms) in enumerate(raw_ranges):
        if end_ms > start_ms:
            start_ms = max(segment_start_ms, start_ms)
            end_ms = min(max(start_ms + 1, end_ms), segment_end_ms)
            repaired.append((start_ms, end_ms))
            continue
        prev_end = repaired[-1][1] if repaired else segment_start_ms
        next_start = None
        for next_index in range(index + 1, count):
            candidate_start, candidate_end = raw_ranges[next_index]
            if candidate_end > candidate_start and candidate_start > prev_end:
                next_start = candidate_start
                break
        if next_start is not None and next_start > prev_end:
            end_ms = min(next_start, prev_end + max(1, (next_start - prev_end) // 2))
            repaired.append((prev_end, end_ms))
        else:
            repaired.append(fallback[index])
    return _ensure_monotonic_ranges(repaired, segment_start_ms, segment_end_ms)


def _proportional_ranges(tokens: Sequence[Any], segment_start_ms: int, segment_end_ms: int) -> list[tuple[int, int]]:
    count = max(1, len(tokens))
    duration = max(count, segment_end_ms - segment_start_ms)
    ranges = []
    for index in range(count):
        start_ms = segment_start_ms + round(duration * index / count)
        end_ms = segment_start_ms + round(duration * (index + 1) / count)
        ranges.append((int(start_ms), int(max(start_ms + 1, end_ms))))
    return _ensure_monotonic_ranges(ranges, segment_start_ms, max(segment_end_ms, ranges[-1][1]))


def _ensure_monotonic_ranges(ranges: list[tuple[int, int]], segment_start_ms: int, segment_end_ms: int) -> list[tuple[int, int]]:
    fixed: list[tuple[int, int]] = []
    cursor = segment_start_ms
    for start_ms, end_ms in ranges:
        start_ms = max(cursor, start_ms)
        end_ms = max(start_ms + 1, end_ms)
        if end_ms > segment_end_ms and segment_end_ms > start_ms:
            end_ms = segment_end_ms
        fixed.append((start_ms, end_ms))
        cursor = end_ms
    return fixed


def _request_token_boundaries(
    records: list[TokenBoundaryRecord],
    *,
    segment_id: str,
    model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    disable_thinking: bool,
    llm_extra_body: dict[str, Any] | None,
    timeout: float,
) -> list[str]:
    system_prompt = get_prompt(
        "split/token_boundary",
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
    )
    user_prompt = f"# segment={segment_id} unit=10ms block={BLOCK_SIZE} count={len(records)}\n{_format_records(records)}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    allowed_ids = [record.token_id for record in records]
    last_error = ""
    rate_limit_delay = RATE_LIMIT_BASE_DELAY
    connection_delay = CONNECTION_BASE_DELAY
    for step in range(MAX_STEPS):
        try:
            response = call_llm(
                messages=messages,
                model=model,
                temperature=0.1,
                base_url=base_url,
                api_key=api_key,
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
            )
        except LLMRateLimitError as exc:
            LOGGER.warning("token-boundary split rate limited attempt=%d: %s", step + 1, exc)
            time.sleep(rate_limit_delay)
            rate_limit_delay = min(rate_limit_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except LLMConnectionError as exc:
            LOGGER.warning("token-boundary split connection error attempt=%d: %s", step + 1, exc)
            time.sleep(connection_delay)
            connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue

        text = response.choices[0].message.content or ""
        try:
            end_ids = parse_end_ids(text, allowed_ids)
            LOGGER.info("token-boundary split segment=%s tokens=%d parts=%d", segment_id, len(records), len(end_ids))
            return end_ids
        except ValueError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Invalid output: {last_error}. Output only END=... with valid increasing IDs."})
    raise RuntimeError(f"token-boundary split failed for {segment_id}: {last_error or 'no valid response'}")


def _request_token_delimited_split(
    records: list[TokenBoundaryRecord],
    *,
    segment_id: str,
    model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    disable_thinking: bool,
    llm_extra_body: dict[str, Any] | None,
    timeout: float,
) -> list[list[str]]:
    system_prompt = get_prompt(
        "split/token_delimited",
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
    )
    source = "|".join(record.text for record in records)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": source}]
    expected = [record.text for record in records]
    last_error = ""
    rate_limit_delay = RATE_LIMIT_BASE_DELAY
    connection_delay = CONNECTION_BASE_DELAY
    for step in range(MAX_STEPS):
        try:
            response = call_llm(
                messages=messages,
                model=model,
                temperature=0.1,
                base_url=base_url,
                api_key=api_key,
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
            )
        except LLMRateLimitError as exc:
            LOGGER.warning("token-delimited split rate limited attempt=%d: %s", step + 1, exc)
            time.sleep(rate_limit_delay)
            rate_limit_delay = min(rate_limit_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except LLMConnectionError as exc:
            LOGGER.warning("token-delimited split connection error attempt=%d: %s", step + 1, exc)
            time.sleep(connection_delay)
            connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except LLMResponseError as exc:
            last_error = str(exc)
            LOGGER.warning("token-delimited split invalid response attempt=%d segment=%s: %s", step + 1, segment_id, exc)
            continue
        except LLMError as exc:
            last_error = str(exc)
            LOGGER.warning("token-delimited split LLM error attempt=%d segment=%s: %s", step + 1, segment_id, exc)
            time.sleep(connection_delay)
            connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue

        text = response.choices[0].message.content or ""
        try:
            groups = parse_token_delimited_output(text, expected)
            LOGGER.info("token-delimited split segment=%s tokens=%d parts=%d", segment_id, len(records), len(groups))
            return groups
        except ValueError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Invalid output: {last_error}. Output the complete original token-delimited text, only inserting <br> between tokens."})
    LOGGER.warning("token-delimited split failed for %s; using local fallback: %s", segment_id, last_error or "no valid response")
    return _fallback_token_groups(records, max_word_count_cjk=max_word_count_cjk, max_word_count_english=max_word_count_english)


def _request_token_counts_split(
    records: list[TokenBoundaryRecord],
    *,
    segment_id: str,
    model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    disable_thinking: bool,
    llm_extra_body: dict[str, Any] | None,
    timeout: float,
) -> list[int]:
    system_prompt = get_prompt(
        "split/token_counts",
        max_word_count_cjk=max_word_count_cjk,
        max_word_count_english=max_word_count_english,
    )
    source = "|".join(record.text for record in records)
    user_prompt = f"COUNT={len(records)}\n{source}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    last_error = ""
    rate_limit_delay = RATE_LIMIT_BASE_DELAY
    connection_delay = CONNECTION_BASE_DELAY
    for step in range(MAX_STEPS):
        try:
            response = call_llm(
                messages=messages,
                model=model,
                temperature=0.1,
                base_url=base_url,
                api_key=api_key,
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
            )
        except LLMRateLimitError as exc:
            LOGGER.warning("token-counts split rate limited attempt=%d: %s", step + 1, exc)
            time.sleep(rate_limit_delay)
            rate_limit_delay = min(rate_limit_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except LLMConnectionError as exc:
            LOGGER.warning("token-counts split connection error attempt=%d: %s", step + 1, exc)
            time.sleep(connection_delay)
            connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue
        except LLMResponseError as exc:
            last_error = str(exc)
            LOGGER.warning("token-counts split invalid response attempt=%d segment=%s: %s", step + 1, segment_id, exc)
            continue
        except LLMError as exc:
            last_error = str(exc)
            LOGGER.warning("token-counts split LLM error attempt=%d segment=%s: %s", step + 1, segment_id, exc)
            time.sleep(connection_delay)
            connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
            continue

        text = response.choices[0].message.content or ""
        try:
            counts = parse_token_counts_output(text, len(records))
            LOGGER.info("token-counts split segment=%s tokens=%d parts=%d", segment_id, len(records), len(counts))
            return counts
        except ValueError as exc:
            last_error = str(exc)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Invalid output: {last_error}. Output only COUNTS=... with positive integers that sum to {len(records)}."})
    LOGGER.warning("token-counts split failed for %s; using local fallback: %s", segment_id, last_error or "no valid response")
    return [len(group) for group in _fallback_token_groups(records, max_word_count_cjk=max_word_count_cjk, max_word_count_english=max_word_count_english)]


def parse_token_delimited_output(output: str, expected_tokens: Sequence[str]) -> list[list[str]]:
    cleaned = re.sub(r"\n+", "", output.strip())
    cleaned = re.sub(r"\s*<br\s*/?>\s*", "<br>", cleaned, flags=re.IGNORECASE)
    raw_groups = [part for part in cleaned.split("<br>") if part.strip()]
    groups: list[list[str]] = []
    flattened: list[str] = []
    for raw_group in raw_groups:
        tokens = [token.strip() for token in raw_group.split("|") if token.strip()]
        if tokens:
            groups.append(tokens)
            flattened.extend(tokens)
    expected = [str(token).strip() for token in expected_tokens if str(token).strip()]
    if flattened != expected:
        raise ValueError("token sequence changed")
    return groups


def parse_token_counts_output(output: str, expected_total: int) -> list[int]:
    match = re.search(r"COUNTS\s*=\s*([0-9,\s]+)", output.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError("missing COUNTS=...")
    counts = [int(item) for item in re.split(r"[,\s]+", match.group(1).strip()) if item]
    if not counts:
        raise ValueError("empty COUNTS list")
    if any(count <= 0 for count in counts):
        raise ValueError("COUNTS must be positive")
    if sum(counts) != expected_total:
        raise ValueError(f"COUNTS sum must be {expected_total}")
    return counts


def _fallback_token_groups(
    records: list[TokenBoundaryRecord],
    *,
    max_word_count_cjk: int,
    max_word_count_english: int,
) -> list[list[str]]:
    limit = max_word_count_cjk if is_mainly_cjk(_join_token_text([record.text for record in records])) else max_word_count_english
    limit = max(1, int(limit))
    groups: list[list[str]] = []
    current: list[str] = []
    current_units = 0
    for index, record in enumerate(records):
        token_units = max(1, count_words(record.text))
        should_split = bool(current) and (
            current_units + token_units > limit
            or (record.gap_tick is not None and record.gap_tick >= 30 and current_units >= max(1, limit // 2))
        )
        if should_split:
            groups.append(current)
            current = []
            current_units = 0
        current.append(record.text)
        current_units += token_units
        if record.gap_tick is not None and record.gap_tick >= 60 and current_units >= max(1, limit // 3):
            groups.append(current)
            current = []
            current_units = 0
        elif index == len(records) - 1 and current:
            groups.append(current)
    return groups


def parse_end_ids(output: str, allowed_ids: Sequence[str]) -> list[str]:
    match = re.search(r"END\s*=\s*([A-Z0-9,\s]+)", output.strip(), flags=re.IGNORECASE)
    if not match:
        raise ValueError("missing END=...")
    ids = [item.strip().upper() for item in match.group(1).split(",") if item.strip()]
    if not ids:
        raise ValueError("empty END list")
    allowed = [item.upper() for item in allowed_ids]
    positions = {item: index for index, item in enumerate(allowed)}
    previous = -1
    for token_id in ids:
        if token_id not in positions:
            raise ValueError(f"unknown token id: {token_id}")
        position = positions[token_id]
        if position <= previous:
            raise ValueError("END ids must be strictly increasing")
        previous = position
    if ids[-1] != allowed[-1]:
        raise ValueError(f"last END must be {allowed[-1]}")
    return ids


def _records_to_segments(records: list[TokenBoundaryRecord], end_ids: list[str]) -> list[ASRDataSeg]:
    id_to_index = {record.token_id.upper(): index for index, record in enumerate(records)}
    result: list[ASRDataSeg] = []
    start_index = 0
    for end_id in end_ids:
        end_index = id_to_index[end_id.upper()]
        group = records[start_index : end_index + 1]
        text = _join_token_text([record.text for record in group])
        result.append(ASRDataSeg(text=text, start_time=group[0].start_ms, end_time=group[-1].end_ms))
        start_index = end_index + 1
    return result


def _token_groups_to_segments(records: list[TokenBoundaryRecord], groups: list[list[str]]) -> list[ASRDataSeg]:
    result: list[ASRDataSeg] = []
    cursor = 0
    for group in groups:
        count = len(group)
        record_group = records[cursor : cursor + count]
        if record_group:
            result.append(
                ASRDataSeg(
                    text=_join_token_text([record.text for record in record_group]),
                    start_time=record_group[0].start_ms,
                    end_time=record_group[-1].end_ms,
                )
            )
        cursor += count
    return result


def _token_counts_to_segments(records: list[TokenBoundaryRecord], counts: list[int]) -> list[ASRDataSeg]:
    groups: list[list[str]] = []
    cursor = 0
    for count in counts:
        group = records[cursor : cursor + count]
        groups.append([record.text for record in group])
        cursor += count
    return _token_groups_to_segments(records, groups)


def _clamp_overlapping_segments(segments: list[ASRDataSeg]) -> list[ASRDataSeg]:
    if not segments:
        return []
    ordered = list(segments)
    for index in range(len(ordered) - 1):
        current = ordered[index]
        next_item = ordered[index + 1]
        if current.end_time > next_item.start_time:
            if next_item.start_time > current.start_time:
                current.end_time = max(current.start_time + 1, next_item.start_time)
            else:
                next_item.start_time = current.end_time
                if next_item.end_time <= next_item.start_time:
                    next_item.end_time = next_item.start_time + 1
    return ordered


def _format_records(records: list[TokenBoundaryRecord]) -> str:
    parts = []
    for record in records:
        timing = "?"
        if record.duration_tick is not None:
            timing = _format_tick(record.duration_tick)
            if record.gap_tick is not None:
                timing = f"{timing},{_format_tick(record.gap_tick)}"
        parts.append(f"{record.token_id}:{record.text}/{timing}")
    return " ".join(parts)


def _format_token_id(index: int) -> str:
    return f"{_block_label(index // BLOCK_SIZE)}{index % BLOCK_SIZE}"


def _block_label(index: int) -> str:
    letters = []
    value = index
    while True:
        letters.append(chr(ord("A") + (value % 26)))
        value = value // 26 - 1
        if value < 0:
            break
    return "".join(reversed(letters))


def _clip_tick(value: int) -> int:
    return max(0, min(MAX_TICK, int(value)))


def _format_tick(value: int) -> str:
    return f"{MAX_TICK}+" if value >= MAX_TICK else str(value)


def _join_token_text(tokens: list[str]) -> str:
    if not tokens:
        return ""
    merged = "".join(tokens) if is_mainly_cjk("".join(tokens)) else " ".join(tokens)
    return re.sub(r"\s+", " ", merged).strip()


__all__ = [
    "TokenBoundaryRecord",
    "parse_end_ids",
    "parse_token_counts_output",
    "parse_token_delimited_output",
    "split_aligned_payload_by_token_boundaries",
    "split_aligned_payload_by_token_counts",
    "split_aligned_payload_by_token_delimited_text",
]
