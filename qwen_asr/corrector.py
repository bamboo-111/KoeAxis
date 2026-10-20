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

from qwen_asr.credentials import resolve_llm_api_key
from qwen_asr.defaults import CORRECT_PROMPT_VERSIONS
from qwen_asr.glossary import build_glossary_prompt
from qwen_asr.models import TranscriptSegment, WorkPaths
from qwen_asr.program_context import program_context_block
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
# v2 edits: longest misheard span the model may replace (longer rewrites are dropped as guesses)
MAX_EDIT_SPAN = 24


def _validate_edits_shape(result: dict[str, Any]) -> tuple[bool, str]:
    edits = result.get("edits")
    if not isinstance(edits, list):
        return False, 'missing "edits" list'
    return True, ""


def apply_correction_edits(
    chunk: list[TranscriptSegment],
    edits: list[Any],
    threshold: float,
) -> dict[str, CorrectionReportItem]:
    """Apply v2 span edits (``{"id", "from", "to", "confidence", "reason"}``) at or above ``threshold``.

    Edits for unknown ids, spans not found in the text, spans longer than ``MAX_EDIT_SPAN`` or no-op edits are
    dropped; edits below the threshold are recorded in the reason as skipped.
    """
    texts = {item.segment_id: item.text for item in chunk}
    applied: dict[str, list[str]] = {key: [] for key in texts}
    skipped: dict[str, list[str]] = {key: [] for key in texts}
    valid: list[tuple[float, str, str, str, str]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        seg_id = str(edit.get("id", "")).strip()
        src = str(edit.get("from", "") or "")
        dst = str(edit.get("to", "") or "")
        if seg_id not in texts or not src or src == dst or len(src) > MAX_EDIT_SPAN:
            continue
        try:
            confidence = float(edit.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        valid.append((confidence, seg_id, src, dst, str(edit.get("reason", "") or "").strip()))
    valid.sort(key=lambda item: -item[0])
    for confidence, seg_id, src, dst, reason in valid:
        label = f"{src}→{dst} ({confidence:.2f}{', ' + reason if reason else ''})"
        if confidence < threshold:
            skipped[seg_id].append(label)
        elif src in texts[seg_id]:
            texts[seg_id] = texts[seg_id].replace(src, dst)
            applied[seg_id].append(label)
    report: dict[str, CorrectionReportItem] = {}
    for item in chunk:
        reasons = applied[item.segment_id][:]
        if skipped[item.segment_id]:
            reasons.append("skipped: " + "; ".join(skipped[item.segment_id]))
        report[item.segment_id] = CorrectionReportItem(
            segment_id=item.segment_id,
            original_text=item.text,
            corrected_text=texts[item.segment_id],
            changed=texts[item.segment_id] != item.text,
            reason="; ".join(reasons) or "kept",
        )
    return report


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
        program_context: str = "",
        prompt_version: str = "v1",
        context_segments: int = 0,
        apply_threshold: float = 0.8,
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
        self.program_context = program_context or ""
        if prompt_version not in CORRECT_PROMPT_VERSIONS:
            raise ValueError(f"unknown correct prompt version: {prompt_version}")
        self.prompt_version = prompt_version
        self.context_segments = max(0, int(context_segments))
        self.apply_threshold = float(apply_threshold)
        self._chunk_context: dict[int, tuple[list[TranscriptSegment], list[TranscriptSegment]]] = {}
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
        self._chunk_context = {}
        if self.context_segments > 0:
            n = self.context_segments
            for index, chunk in enumerate(chunks):
                start = index * self.batch_num
                before = eligible[max(0, start - n) : start]
                after = eligible[start + len(chunk) : start + len(chunk) + n]
                self._chunk_context[id(chunk)] = (before, after)
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
        if self.prompt_version == "v2":
            return self._correct_chunk_v2(chunk)
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
        last_result = self._call_with_retries(
            messages,
            lambda parsed: self._validate_result(input_dict, parsed),
            "Output must be a JSON object keyed by segment_id.",
            "Fix the errors and output ONLY a JSON object containing every original segment_id.",
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

    def _call_with_retries(
        self,
        messages: list[dict[str, str]],
        validate: Any,
        not_object_hint: str,
        fix_hint: str,
    ) -> dict[str, Any] | None:
        """Call the LLM until ``validate(parsed)`` passes; returns the last parsed object (valid or not) or None."""
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
                messages.append({"role": "user", "content": not_object_hint})
                continue

            last_result = parsed
            is_valid, error_message = validate(parsed)
            if is_valid:
                return parsed

            LOGGER.warning("ASR correction validation failed (try %d): %s", step + 1, error_message)
            messages.append({"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)})
            messages.append({"role": "user", "content": f"Error: {error_message}\n{fix_hint}"})

        return last_result

    def _build_system_prompt(self) -> str:
        glossary = self.glossary_prompt.strip() or "No glossary supplied."
        context = program_context_block(self.program_context)
        context = f"{context}\n" if context else ""
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
- misrecognized proper nouns, show names, song titles, member names, venues, fixed phrases (check the glossary
  and the program context; a misheard name usually sounds similar to the real one)
- broken dates/numbers, spelled-out English letters, emails, URLs, hashtags
- obvious katakana or homophone ASR errors when context is clear
Keep plausible spoken Japanese as it is: fillers, repetitions, casual or broken grammar are not errors.

{context}Use this glossary as recognition guidance, not as translation instructions:
<glossary>
{glossary}
</glossary>

Input format: {{"segment_000001": "text", "segment_000002": "text"}}
Output format: {{"segment_000001": {{"corrected_text": "text", "reason": "short reason or kept"}}, "segment_000002": {{"corrected_text": "text", "reason": "short reason or kept"}}}}
"""

    # ---- v2: minimal span edits with confidence, read-only neighbour segments ----

    def _correct_chunk_v2(self, chunk: list[TranscriptSegment]) -> dict[str, CorrectionReportItem]:
        texts = {item.segment_id: item.text for item in chunk}
        before, after = self._chunk_context.get(id(chunk), ([], []))
        LOGGER.info(
            "[+] correcting ASR text (v2): %d segments (%.1fs - %.1fs)",
            len(chunk),
            chunk[0].global_start_time,
            chunk[-1].global_end_time,
        )
        parts = []
        if before:
            parts.append("<context_before>" + "\n".join(item.text for item in before) + "</context_before>")
        parts.append("<asr_segments>" + json.dumps(texts, ensure_ascii=False) + "</asr_segments>")
        if after:
            parts.append("<context_after>" + "\n".join(item.text for item in after) + "</context_after>")
        messages = [
            {"role": "system", "content": self._build_system_prompt_v2()},
            {"role": "user", "content": "\n".join(parts)},
        ]
        result = self._call_with_retries(
            messages,
            _validate_edits_shape,
            'Output must be a JSON object: {"edits": [...]}.',
            'Output ONLY {"edits": [...]} (an empty list when nothing needs fixing).',
        )
        if result is None or not isinstance(result.get("edits"), list):
            raise RuntimeError("Correction agent loop did not return a valid edit list")
        return apply_correction_edits(chunk, result["edits"], self.apply_threshold)

    def _build_system_prompt_v2(self) -> str:
        glossary = self.glossary_prompt.strip() or "No glossary supplied."
        context = program_context_block(self.program_context)
        context = f"{context}\n" if context else ""
        return f"""You are an ASR post-correction engine for Japanese radio / video transcripts.
Find what the speech recognizer got WRONG and return minimal replacements. Do not translate or restyle.

Read the segments as a conversation: use the neighbouring segments, the program context and the glossary to
work out what the speakers actually said. Typical recognition errors:
- names and show terms heard as other words (寿司さん→朱李さん, シリーさん→朱李さん, 小エログ→声ログ);
- set phrases, greetings and common words garbled into nonsense or into the wrong word;
- homophones / near-homophones that contradict the context (公開する→後悔する);
- a dropped or misheard particle, okurigana or conjugation that changes the meaning;
- a numeral or letter standing in for a name or word that sounds alike.

Rules:
1. The replacement must be something that plausibly SOUNDS like the recognized text (same or similar
   mora). Do not insert content that was not said, and do not replace a sensible word with a "better" one.
2. "from" must be copied verbatim from the segment text and be as short as possible (a word or short phrase,
   at most {MAX_EDIT_SPAN} characters); "to" is its replacement. One edit per distinct error; an edit
   replaces every occurrence of "from" in that segment.
3. Member names and show terms may be written in the form used by the glossary / program context
   (リナ→理名). Leave fillers, repetitions, casual speech, punctuation, spacing and letter case alone.
4. Fix every error you are reasonably sure about; give confidence 0.9+ when the context makes it certain,
   0.6–0.8 when plausible, lower when guessing. Do not guess at unintelligible passages.
5. <context_before> / <context_after> are read-only; edit only ids inside <asr_segments>.
6. Output ONLY raw JSON: {{"edits": [{{"id": "segment_000001", "from": "シリーさん", "to": "朱李さん",
   "confidence": 0.95, "reason": "member name"}}]}}. Use {{"edits": []}} when nothing needs fixing.

{context}Use this glossary as recognition guidance, not as translation instructions:
<glossary>
{glossary}
</glossary>
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
    mode: str = "rules",
    program_context: str = "",
    prompt_version: str = "v1",
    context_segments: int = 0,
    apply_threshold: float = 0.8,
) -> tuple[list[TranscriptSegment], list[CorrectionReportItem]]:
    """``rules``: deterministic cleanup only. ``llm``: ASRCorrector fixes obvious misrecognitions first
    (``prompt_version`` v1 rewrites whole segments; v2 returns span edits applied at ``apply_threshold``)."""
    if not work_paths.raw_transcript_manifest.exists():
        shutil.copy2(work_paths.transcript_manifest, work_paths.raw_transcript_manifest)

    original_text = {item.segment_id: item.text for item in transcripts}
    llm_reports: dict[str, CorrectionReportItem] = {}
    if mode == "llm":
        resolved_key = resolve_llm_api_key(api_key, base_url)
        if not (llm_model and base_url and resolved_key):
            raise RuntimeError("correct --correct-mode llm needs --llm-model, --llm-base-url and an API key")
        corrector = ASRCorrector(
            model=llm_model,
            base_url=base_url,
            api_key=resolved_key,
            thread_num=thread_num,
            batch_num=batch_num,
            glossary_xlsx=glossary_xlsx,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
            program_context=program_context,
            prompt_version=prompt_version,
            context_segments=context_segments,
            apply_threshold=apply_threshold,
        )
        try:
            transcripts_llm, llm_report = corrector.correct(transcripts)
        finally:
            corrector.stop()
        llm_reports = {item.segment_id: item for item in llm_report}
        LOGGER.info(
            "LLM correction: %d changed, %d failed, %d total",
            sum(item.changed for item in llm_report),
            sum(item.status != "completed" for item in llm_report),
            len(llm_report),
        )
        transcripts = transcripts_llm
    elif mode != "rules":
        raise ValueError(f"unknown correct mode: {mode}")

    corrected: list[TranscriptSegment] = []
    report: list[CorrectionReportItem] = []
    for item in transcripts:
        clone = TranscriptSegment(**asdict(item))
        if clone.status == "completed" and clone.text.strip():
            cleaned = clean_asr_correction_text(clone.text).strip() or clone.text
            cleanup_changed = cleaned != clone.text
            clone.text = cleaned
            before = original_text.get(clone.segment_id, item.text)
            llm_item = llm_reports.get(clone.segment_id)
            reasons = []
            if llm_item is not None and llm_item.changed:
                reasons.append(f"llm: {llm_item.reason or 'corrected'}")
            if cleanup_changed:
                reasons.append("deterministic text cleanup")
            report.append(
                CorrectionReportItem(
                    segment_id=clone.segment_id,
                    original_text=before,
                    corrected_text=cleaned,
                    changed=cleaned != before,
                    reason="; ".join(reasons) or "kept; no deterministic cleanup needed",
                    status=llm_item.status if llm_item is not None else "completed",
                    error=llm_item.error if llm_item is not None else None,
                )
            )
        corrected.append(clone)

    write_json_atomic(work_paths.transcript_manifest, serialize_manifest(corrected))
    write_json_atomic(work_paths.corrected_manifest, [asdict(item) for item in report])
    return corrected, report
