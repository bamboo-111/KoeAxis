from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from optimizer.exceptions import LLMAuthError
from optimizer.splitter_boundaries import STRONG_SENTENCE_END
from optimizer.splitter_readability import is_dialogue_standalone_response
from optimizer.llm_client import call_llm

# The merge only joins cues; a batch that still fails after retries keeps its rule-split cues.
MERGE_BATCH_RETRIES = 2

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """Select merge boundaries for adjacent subtitle cues.
Return JSON only in this shape: {\"groups\":[[1,2],[3],[4,5]]}.
Every input id must appear exactly once, in order, with no duplicates or omissions.
Only merge adjacent ids when they form one incomplete phrase or a word was split across cues.
Keep complete sentences, speaker changes, short replies, questions, and reactions separate.
Never return, correct, translate, or rewrite subtitle text.
"""


def merge_adjacent_boundaries(
    segments: list[Any],
    *,
    model: str,
    base_url: str,
    api_key: str,
    workers: int,
    max_chars: int,
    disable_thinking: bool,
    llm_extra_body: dict[str, Any] | None,
    timeout: float,
    batch_size: int = 50,
) -> list[Any]:
    if not segments:
        return []
    batches = [segments[index : index + batch_size] for index in range(0, len(segments), batch_size)]
    grouped: dict[int, list[list[int]]] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _request_groups_or_keep,
                batch,
                model=model,
                base_url=base_url,
                api_key=api_key,
                max_chars=max_chars,
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
            ): index
            for index, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            grouped[futures[future]] = future.result()

    result: list[Any] = []
    for index, batch in enumerate(batches):
        result.extend(_merge_batch(batch, grouped[index], max_chars=max_chars))
    source_text = "".join(str(segment.text) for segment in segments)
    output_text = "".join(str(segment.text) for segment in result)
    if output_text != source_text:
        raise RuntimeError("Semantic boundary merge content guard failed")
    LOGGER.info(
        "Constrained semantic boundary merge completed: %d -> %d cues; exact content preserved",
        len(segments),
        len(result),
    )
    return result


def _request_groups_or_keep(batch: list[Any], *, retries: int = MERGE_BATCH_RETRIES, **kwargs: Any) -> list[list[int]]:
    """One failed request (network stall, bad JSON, guard) must not fail the whole split stage."""
    for attempt in range(retries + 1):
        try:
            return _request_groups(batch, **kwargs)
        except LLMAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the unmerged batch
            LOGGER.warning(
                "Semantic boundary merge batch of %d cues failed (attempt %d/%d): %s",
                len(batch),
                attempt + 1,
                retries + 1,
                exc,
            )
            if attempt < retries:
                time.sleep(2**attempt)
    LOGGER.warning("Semantic boundary merge kept %d cues unmerged after %d attempts", len(batch), retries + 1)
    return [[index] for index in range(1, len(batch) + 1)]


def _request_groups(
    batch: list[Any],
    *,
    model: str,
    base_url: str,
    api_key: str,
    max_chars: int,
    disable_thinking: bool,
    llm_extra_body: dict[str, Any] | None,
    timeout: float,
) -> list[list[int]]:
    payload = [
        {
            "id": index,
            "start_ms": int(segment.start_time),
            "end_ms": int(segment.end_time),
            "text": str(segment.text),
        }
        for index, segment in enumerate(batch, 1)
    ]
    prompt = f"Prefer merged text no longer than {max_chars} characters.\n" + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    response = call_llm(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.0,
        base_url=base_url,
        api_key=api_key,
        disable_thinking=disable_thinking,
        require_json=True,
        llm_extra_body=llm_extra_body,
        timeout=timeout,
    )
    content = response.choices[0].message.content or ""
    groups = json.loads(content).get("groups")
    if not isinstance(groups, list) or not all(isinstance(group, list) and group for group in groups):
        raise RuntimeError("Semantic boundary merge returned invalid groups")
    normalized = [[int(value) for value in group] for group in groups]
    expected = list(range(1, len(batch) + 1))
    flattened = [value for group in normalized for value in group]
    if flattened != expected:
        raise RuntimeError("Semantic boundary merge coverage guard failed")
    if any(group != list(range(group[0], group[-1] + 1)) for group in normalized):
        raise RuntimeError("Semantic boundary merge returned non-adjacent ids")
    return normalized


def _merge_batch(batch: list[Any], groups: list[list[int]], *, max_chars: int) -> list[Any]:
    result: list[Any] = []
    for group in groups:
        selected = [batch[index - 1] for index in group]
        if len(selected) == 1 or _merge_allowed(selected, max_chars=max_chars):
            result.append(_merged_segment(selected))
            continue
        # The model often chains 3-4 cues into one group; when the whole chain breaks a guard,
        # keep the longest guard-passing runs (left to right) instead of discarding every merge.
        run: list[Any] = [selected[0]]
        for segment in selected[1:]:
            if _merge_allowed([*run, segment], max_chars=max_chars):
                run.append(segment)
                continue
            result.append(_merged_segment(run))
            run = [segment]
        result.append(_merged_segment(run))
    return result


def _merge_allowed(selected: list[Any], *, max_chars: int) -> bool:
    text = "".join(str(segment.text) for segment in selected)
    duration = int(selected[-1].end_time) - int(selected[0].start_time)
    gaps = [
        int(selected[index + 1].start_time) - int(selected[index].end_time)
        for index in range(len(selected) - 1)
    ]
    crosses_sentence = any(STRONG_SENTENCE_END.search(str(segment.text)) for segment in selected[:-1])
    contains_response = any(is_dialogue_standalone_response(str(segment.text)) for segment in selected)
    return not (
        len(text) > max_chars or duration > 10000 or any(gap > 1000 for gap in gaps)
        or crosses_sentence or contains_response
    )


def _merged_segment(selected: list[Any]) -> Any:
    if len(selected) == 1:
        return selected[0]
    first = selected[0]
    return type(first)(
        text="".join(str(segment.text) for segment in selected),
        translated_text="".join(str(segment.translated_text) for segment in selected),
        start_time=int(first.start_time),
        end_time=int(selected[-1].end_time),
    )
