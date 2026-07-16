from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json_repair

from qwen_asr.glossary import build_glossary_prompt
from qwen_asr.models import TranscriptSegment, WorkPaths
from optimizer.exceptions import LLMConnectionError, LLMRateLimitError
from optimizer.llm_config import LLMConfig
from optimizer.llm_client import DEFAULT_TIMEOUT, call_llm
from optimizer.text_utils import clean_asr_correction_text
from qwen_asr.storage import serialize_manifest, write_json_atomic

LOGGER = logging.getLogger(__name__)

MAX_STEPS = 5
RATE_LIMIT_BASE_DELAY = 5.0
CONNECTION_BASE_DELAY = 3.0
MAX_DELAY = 60.0
BACKOFF_FACTOR = 2.0
GLOBAL_COOLDOWN_SECONDS = 10.0
SUBMIT_INTERVAL = 0.5


@dataclass(slots=True)
class CorrectionReportItem:
    segment_id: str
    original_text: str
    corrected_text: str
    changed: bool
    reason: str = ""
    status: str = "completed"
    error: str | None = None


class ASRCorrector:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        thread_num: int = 4,
        batch_num: int = 8,
        glossary_xlsx: Path | str | None = None,
        disable_thinking: bool = True,
        llm_extra_body: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.llm_config = LLMConfig(
            model=model,
            base_url=base_url,
            api_key=api_key,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
        self.thread_num = max(1, thread_num)
        self.batch_num = max(1, batch_num)
        self.disable_thinking = disable_thinking
        self.llm_extra_body = llm_extra_body
        self.timeout = timeout
        self.glossary_prompt = build_glossary_prompt(glossary_xlsx) if glossary_xlsx else ""
        self.is_running = True
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(max_workers=self.thread_num)
        self._cooldown_until = 0.0
        self._cooldown_lock = threading.Lock()

    def correct(self, transcripts: list[TranscriptSegment]) -> tuple[list[TranscriptSegment], list[CorrectionReportItem]]:
        eligible = [
            item for item in transcripts
            if item.status == "completed" and item.text.strip()
        ]
        if not eligible:
            return transcripts, []

        chunks = [
            eligible[i : i + self.batch_num]
            for i in range(0, len(eligible), self.batch_num)
        ]
        report_by_id: dict[str, CorrectionReportItem] = {}
        success_count = self._parallel_correct(chunks, report_by_id)
        if success_count == 0:
            raise RuntimeError(f"ASR correction failed: all {len(chunks)} batches failed")

        corrected_by_id = {
            item.segment_id: item.corrected_text
            for item in report_by_id.values()
            if item.status == "completed"
        }
        corrected_transcripts: list[TranscriptSegment] = []
        for item in transcripts:
            clone = TranscriptSegment(**asdict(item))
            if clone.segment_id in corrected_by_id:
                clone.text = corrected_by_id[clone.segment_id]
            corrected_transcripts.append(clone)

        report = [
            report_by_id[item.segment_id]
            for item in eligible
            if item.segment_id in report_by_id
        ]
        LOGGER.info("ASR correction completed: %d/%d batches succeeded", success_count, len(chunks))
        return corrected_transcripts, report

    def _parallel_correct(
        self,
        chunks: list[list[TranscriptSegment]],
        report_by_id: dict[str, CorrectionReportItem],
    ) -> int:
        if not self.executor:
            raise RuntimeError("Correction thread pool is not initialized")

        future_to_chunk = {}
        for index, chunk in enumerate(chunks):
            if not self.is_running:
                break
            future = self.executor.submit(self._correct_chunk, chunk)
            future_to_chunk[future] = chunk
            if index < len(chunks) - 1:
                time.sleep(SUBMIT_INTERVAL)

        success_count = 0
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                result = future.result()
                report_by_id.update(result)
                success_count += 1
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.error(
                    "ASR correction batch failed (%s - %s): %s",
                    chunk[0].segment_id,
                    chunk[-1].segment_id,
                    exc,
                )
                for item in chunk:
                    report_by_id[item.segment_id] = CorrectionReportItem(
                        segment_id=item.segment_id,
                        original_text=item.text,
                        corrected_text=item.text,
                        changed=False,
                        reason="batch failed; original text kept",
                        status="failed",
                        error=str(exc),
                    )
        return success_count

    def _correct_chunk(self, chunk: list[TranscriptSegment]) -> dict[str, CorrectionReportItem]:
        input_dict = {
            item.segment_id: item.text
            for item in chunk
        }
        time_start = chunk[0].global_start_time
        time_end = chunk[-1].global_end_time
        LOGGER.info(
            "[+] correcting ASR text: %d segments (%.1fs - %.1fs)",
            len(chunk),
            time_start,
            time_end,
        )

        result_dict = self._agent_loop(input_dict)
        if result_dict is None:
            raise RuntimeError("Correction agent loop did not return a valid result")

        report: dict[str, CorrectionReportItem] = {}
        for item in chunk:
            payload = result_dict.get(item.segment_id)
            if isinstance(payload, dict):
                corrected = str(payload.get("corrected_text", item.text)).strip()
                reason = str(payload.get("reason", "")).strip()
            else:
                corrected = str(payload or item.text).strip()
                reason = ""
            corrected = clean_asr_correction_text(corrected)
            if not corrected:
                corrected = item.text
                reason = reason or "empty correction ignored; original text kept"
            report[item.segment_id] = CorrectionReportItem(
                segment_id=item.segment_id,
                original_text=item.text,
                corrected_text=corrected,
                changed=corrected != item.text,
                reason=reason,
            )
        return report

    def _agent_loop(self, input_dict: dict[str, str]) -> dict[str, Any] | None:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": (
                    "<asr_segments>"
                    f"{json.dumps(input_dict, ensure_ascii=False)}"
                    "</asr_segments>"
                ),
            },
        ]
        last_result: dict[str, Any] | None = None
        rate_limit_delay = RATE_LIMIT_BASE_DELAY
        connection_delay = CONNECTION_BASE_DELAY

        for step in range(MAX_STEPS):
            if not self.is_running:
                break
            self._wait_for_cooldown()
            if not self.is_running:
                break

            try:
                response = call_llm(
                    messages=messages,
                    model=self.llm_config.model,
                    temperature=0.1,
                    base_url=self.llm_config.base_url,
                    api_key=self.llm_config.api_key,
                    disable_thinking=self.llm_config.disable_thinking,
                    require_json=True,
                    llm_extra_body=self.llm_config.llm_extra_body,
                    timeout=self.llm_config.timeout,
                )
            except LLMRateLimitError as exc:
                LOGGER.warning(
                    "LLM rate limited during correction (try %d): %s; waiting %.1fs",
                    step + 1,
                    exc,
                    rate_limit_delay,
                )
                self._set_global_cooldown(max(rate_limit_delay, GLOBAL_COOLDOWN_SECONDS))
                self._sleep_interruptible(rate_limit_delay)
                rate_limit_delay = min(rate_limit_delay * BACKOFF_FACTOR, MAX_DELAY)
                continue
            except LLMConnectionError as exc:
                LOGGER.warning(
                    "LLM connection/upstream error during correction (try %d): %s; waiting %.1fs",
                    step + 1,
                    exc,
                    connection_delay,
                )
                self._sleep_interruptible(connection_delay)
                connection_delay = min(connection_delay * BACKOFF_FACTOR, MAX_DELAY)
                continue
            except Exception as exc:  # pylint: disable=broad-exception-caught
                LOGGER.warning("LLM correction call failed (try %d): %s", step + 1, exc)
                continue

            result_text = response.choices[0].message.content
            if not result_text:
                LOGGER.warning("LLM correction returned empty response (try %d)", step + 1)
                continue

            parsed = json_repair.loads(result_text.strip())
            if not isinstance(parsed, dict):
                messages.append({"role": "assistant", "content": result_text})
                messages.append({"role": "user", "content": "Output must be a JSON object keyed by segment_id."})
                continue

            last_result = parsed
            is_valid, error_message = self._validate_result(input_dict, parsed)
            if is_valid:
                return parsed

            LOGGER.warning("ASR correction validation failed (try %d): %s", step + 1, error_message)
            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {error_message}\n"
                        "Fix the errors and output ONLY a JSON object containing every original segment_id."
                    ),
                }
            )

        if last_result is not None:
            filtered = {
                key: value
                for key, value in last_result.items()
                if key in input_dict
            }
            if filtered:
                return filtered
        return None

    def _build_system_prompt(self) -> str:
        glossary = self.glossary_prompt.strip() or "No glossary supplied."
        return f"""You are an ASR post-correction engine for subtitle transcripts.

Your ONLY task is to fix obvious speech recognition mistakes in the original transcript language.

ABSOLUTE REQUIREMENTS:
1. Do NOT translate.
2. Do NOT rewrite for style, summarize, expand, or make the text more formal.
3. Do NOT merge, split, add, remove, or rename segment IDs.
4. If unsure, keep the original text exactly.
5. Output ONLY a raw JSON object. No markdown, notes, or code fences.
6. Each value must be an object with "corrected_text" and "reason".

Fix only high-confidence issues:
- misrecognized proper nouns, show names, song titles, member names, venues, fixed phrases
- broken dates/numbers, spelled-out English letters, emails, URLs, hashtags
- obvious katakana or homophone ASR errors when context is clear

High-priority fixed terms for this project:
- トゲナシトゲアリ, トゲラジ, ハッシュタグトゲラジ
- 仁菜, 朱李, ルパ, ダイヤモンドダスト
- InterFM / interfm.jp
- The program address is togeraji@interfm.jp. Correct split or misheard forms such as "t o g e r a j i at interfm jp",
  "d o g e a d i at interfm jp", "トゲラジ at intfm.jp", and similar variants to "togeraji@interfm.jp".

Use this glossary as recognition guidance, not as translation instructions:
<glossary>
{glossary}
</glossary>

Input format: {{"segment_000001": "text", "segment_000002": "text"}}
Output format: {{"segment_000001": {{"corrected_text": "text", "reason": "short reason or kept"}}, "segment_000002": {{"corrected_text": "text", "reason": "short reason or kept"}}}}
"""

    def _validate_result(self, input_dict: dict[str, str], result: dict[str, Any]) -> tuple[bool, str]:
        expected = set(input_dict.keys())
        actual = set(str(key) for key in result.keys())
        missing = expected - actual
        extra = actual - expected
        errors: list[str] = []
        if missing:
            errors.append(f"Missing keys {sorted(missing)}")
        if extra:
            errors.append(f"Extra keys {sorted(extra)}")
        for key in expected & actual:
            value = result.get(key)
            if isinstance(value, dict):
                corrected = value.get("corrected_text")
            else:
                corrected = value
            if corrected is None:
                errors.append(f"{key} is missing corrected_text")
        return (not errors), "; ".join(errors)

    def _set_global_cooldown(self, seconds: float) -> None:
        with self._cooldown_lock:
            deadline = time.monotonic() + seconds
            if deadline > self._cooldown_until:
                self._cooldown_until = deadline

    def _wait_for_cooldown(self) -> None:
        while self.is_running:
            with self._cooldown_lock:
                remaining = self._cooldown_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))

    def _sleep_interruptible(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while self.is_running and time.monotonic() < deadline:
            time.sleep(min(0.5, deadline - time.monotonic()))

    def stop(self) -> None:
        self.is_running = False
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


def run_correction_stage(
    work_paths: WorkPaths,
    transcripts: list[TranscriptSegment],
    llm_model: str,
    base_url: str,
    api_key: str,
    thread_num: int = 4,
    batch_num: int = 8,
    glossary_xlsx: Path | str | None = None,
    disable_thinking: bool = True,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[list[TranscriptSegment], list[CorrectionReportItem]]:
    if not work_paths.raw_transcript_manifest.exists():
        shutil.copy2(work_paths.transcript_manifest, work_paths.raw_transcript_manifest)

    del llm_model, base_url, api_key, thread_num, batch_num
    del glossary_xlsx, disable_thinking, llm_extra_body, timeout

    corrected: list[TranscriptSegment] = []
    report: list[CorrectionReportItem] = []
    for item in transcripts:
        clone = TranscriptSegment(**asdict(item))
        if clone.status == "completed" and clone.text.strip():
            cleaned = clean_asr_correction_text(clone.text).strip() or clone.text
            changed = cleaned != clone.text
            clone.text = cleaned
            report.append(
                CorrectionReportItem(
                    segment_id=clone.segment_id,
                    original_text=item.text,
                    corrected_text=cleaned,
                    changed=changed,
                    reason=(
                        "deterministic text cleanup"
                        if changed
                        else "kept; no deterministic cleanup needed"
                    ),
                )
            )
        corrected.append(clone)

    write_json_atomic(work_paths.transcript_manifest, serialize_manifest(corrected))
    write_json_atomic(work_paths.corrected_manifest, [asdict(item) for item in report])
    return corrected, report
