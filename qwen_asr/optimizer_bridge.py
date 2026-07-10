from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qwen_asr.align import validate_aligned_token_timing
from qwen_asr.glossary import build_glossary_prompt
from qwen_asr.models import AlignedToken
from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress
from qwen_asr.storage import read_json, write_json_atomic

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPTIMIZER_ROOT = PROJECT_ROOT / "optimizer"


def run_split_stage(
    work_paths: WorkPaths,
    optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT,
    llm_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    thread_num: int = 4,
    max_word_count_cjk: int = 25,
    max_word_count_english: int = 18,
    prompt_limit_ratio: float = 0.8,
    disable_thinking: bool = False,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = 120.0,
    split_mode: str = "token-counts",
) -> None:
    ASRData, ASRDataSeg, SubtitleSplitter = _load_optimizer_types(optimizer_root)
    source = aligned_manifest_to_asr_data(work_paths, ASRData, ASRDataSeg)
    if not source.segments:
        raise RuntimeError("No aligned token data available for split stage.")

    if llm_model and base_url and api_key and split_mode in {"token-counts", "token-delimited", "token-boundary"}:
        try:
            aligned_payload = read_json(work_paths.aligned_manifest, default=[])
            if split_mode == "token-counts":
                from optimizer.token_boundary_split import split_aligned_payload_by_token_counts

                result = split_aligned_payload_by_token_counts(
                    aligned_payload,
                    model=llm_model,
                    base_url=base_url,
                    api_key=api_key,
                    max_word_count_cjk=max_word_count_cjk,
                    max_word_count_english=max_word_count_english,
                    disable_thinking=disable_thinking,
                    llm_extra_body=llm_extra_body,
                    timeout=timeout,
                    thread_num=thread_num,
                )
            elif split_mode == "token-delimited":
                from optimizer.token_boundary_split import split_aligned_payload_by_token_delimited_text

                result = split_aligned_payload_by_token_delimited_text(
                    aligned_payload,
                    model=llm_model,
                    base_url=base_url,
                    api_key=api_key,
                    max_word_count_cjk=max_word_count_cjk,
                    max_word_count_english=max_word_count_english,
                    disable_thinking=disable_thinking,
                    llm_extra_body=llm_extra_body,
                    timeout=timeout,
                    thread_num=thread_num,
                )
            else:
                from optimizer.token_boundary_split import split_aligned_payload_by_token_boundaries

                result = split_aligned_payload_by_token_boundaries(
                    aligned_payload,
                    model=llm_model,
                    base_url=base_url,
                    api_key=api_key,
                    max_word_count_cjk=max_word_count_cjk,
                    max_word_count_english=max_word_count_english,
                    disable_thinking=disable_thinking,
                    llm_extra_body=llm_extra_body,
                    timeout=timeout,
                    thread_num=thread_num,
                )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.warning("%s split failed; falling back to text LLM split: %s", split_mode, exc)
            result = _run_text_split(
                source=source,
                SubtitleSplitter=SubtitleSplitter,
                thread_num=thread_num,
                llm_model=llm_model,
                base_url=base_url,
                api_key=api_key,
                max_word_count_cjk=max_word_count_cjk,
                max_word_count_english=max_word_count_english,
                prompt_limit_ratio=prompt_limit_ratio,
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
            )
    elif llm_model and base_url and api_key and split_mode == "text":
        result = _run_text_split(
            source=source,
            SubtitleSplitter=SubtitleSplitter,
            thread_num=thread_num,
            llm_model=llm_model,
            base_url=base_url,
            api_key=api_key,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            prompt_limit_ratio=prompt_limit_ratio,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
    else:
        LOGGER.info("No LLM config supplied or split_mode=rule. Falling back to optimizer rule-based split.")
        splitter = SubtitleSplitter(
            thread_num=1,
            model="",
            base_url="",
            api_key="",
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            timeout=timeout,
        )
        try:
            processed = splitter._process_by_rules(source.segments)  # noqa: SLF001
            result = ASRData(processed)
        finally:
            splitter.stop()

    write_json_atomic(work_paths.split_manifest, result.to_json())
    work_paths.split_srt.write_text(result.to_srt(), encoding="utf-8")


def _run_text_split(
    *,
    source: Any,
    SubtitleSplitter: Any,
    thread_num: int,
    llm_model: str,
    base_url: str,
    api_key: str,
    max_word_count_cjk: int,
    max_word_count_english: int,
    prompt_limit_ratio: float,
    disable_thinking: bool,
    llm_extra_body: dict[str, Any] | None,
    timeout: float,
):
    if llm_model and base_url and api_key:
        splitter = SubtitleSplitter(
            thread_num=thread_num,
            model=llm_model,
            base_url=base_url,
            api_key=api_key,
            max_word_count_cjk=max_word_count_cjk,
            max_word_count_english=max_word_count_english,
            prompt_limit_ratio=prompt_limit_ratio,
            disable_thinking=disable_thinking,
            llm_extra_body=llm_extra_body,
            timeout=timeout,
        )
        try:
            return splitter.split_subtitle(source)
        finally:
            splitter.stop()
    raise RuntimeError("Text split requires llm_model, base_url, and api_key")


def run_translate_stage(
    work_paths: WorkPaths,
    target_language: str,
    llm_model: str,
    base_url: str,
    api_key: str,
    optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT,
    thread_num: int = 4,
    batch_num: int = 20,
    custom_prompt: str = "",
    glossary_xlsx: Path | str | None = None,
    disable_thinking: bool = True,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> None:
    ASRData, _, _, SubtitleTranslator = _load_optimizer_types(
        optimizer_root,
        need_translator=True,
    )
    if not work_paths.split_manifest.exists():
        raise RuntimeError("split_segments.json is missing. Run split first.")

    asr_data = ASRData.from_json(read_json(work_paths.split_manifest, default={}))
    prompt_parts = [custom_prompt.strip()] if custom_prompt.strip() else []
    if glossary_xlsx:
        glossary_prompt = build_glossary_prompt(glossary_xlsx)
        if glossary_prompt:
            prompt_parts.append(glossary_prompt)

    translator = SubtitleTranslator(
        thread_num=thread_num,
        batch_num=batch_num,
        model=llm_model,
        base_url=base_url,
        api_key=api_key,
        target_language=target_language,
        custom_prompt="\n\n".join(prompt_parts),
        disable_thinking=disable_thinking,
        llm_extra_body=llm_extra_body,
        timeout=timeout,
        progress_callback=lambda done, total, current: write_progress(
            work_paths,
            stage="translate",
            status="running",
            done=done,
            total=total,
            current=current,
            summary=f"{done}/{total or '?'} translated subtitles",
        ),
    )
    try:
        result = translator.translate(asr_data)
    finally:
        translator.stop()

    write_json_atomic(work_paths.translated_manifest, result.to_json())
    work_paths.translated_srt.write_text(result.to_srt(), encoding="utf-8")


def aligned_manifest_to_asr_data(work_paths: WorkPaths, ASRData: Any, ASRDataSeg: Any):
    aligned_payload = read_json(work_paths.aligned_manifest, default=[])
    transcript_payload = read_json(work_paths.transcript_manifest, default=[])
    transcript_by_id = {
        str(item.get("segment_id")): item
        for item in transcript_payload
        if isinstance(item, dict)
    }
    segments = []
    failures: list[str] = []
    for index, item in enumerate(aligned_payload):
        next_segment_start_ms = _next_segment_start_ms(aligned_payload, index)
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
            start_ms = int(round(float(transcript.get("global_start_time", item.get("global_start_time", 0.0))) * 1000))
            end_ms = int(round(float(transcript.get("global_end_time", item.get("global_end_time", 0.0))) * 1000))
            if next_segment_start_ms is not None:
                end_ms = min(end_ms, next_segment_start_ms)
            if end_ms <= start_ms:
                failures.append(f"{segment_id}: transcript timing is not positive")
                continue
            segments.append(ASRDataSeg(text=text, start_time=start_ms, end_time=end_ms))
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
        error = validate_aligned_token_timing(
            tokens,
            float(item.get("global_start_time", 0.0)),
            float(item.get("global_end_time", 0.0)),
        )
        if error:
            failures.append(f"{item.get('segment_id', '<unknown>')}: {error}")
            continue
        for token in item.get("tokens", []):
            text = str(token.get("text", "")).strip()
            if not text:
                continue
            start_ms = int(round(float(token.get("start_time", 0.0)) * 1000))
            end_ms = int(round(float(token.get("end_time", 0.0)) * 1000))
            if next_segment_start_ms is not None:
                end_ms = min(end_ms, next_segment_start_ms)
            if end_ms <= start_ms:
                continue
            segments.append(ASRDataSeg(text=text, start_time=start_ms, end_time=end_ms))
    if failures:
        preview = "; ".join(failures[:5])
        more = f"; +{len(failures) - 5} more" if len(failures) > 5 else ""
        raise RuntimeError(f"Alignment timing is unreliable and cannot fall back to transcript timing: {preview}{more}")
    return ASRData(_make_segments_monotonic(segments, ASRDataSeg))


def _next_segment_start_ms(aligned_payload: list[dict[str, Any]], index: int) -> int | None:
    if index + 1 >= len(aligned_payload):
        return None
    return int(round(float(aligned_payload[index + 1].get("global_start_time", 0.0)) * 1000))


def _make_segments_monotonic(segments: list[Any], ASRDataSeg: Any) -> list[Any]:
    result = []
    for index, segment in enumerate(segments):
        start_time = int(segment.start_time)
        end_time = int(segment.end_time)
        if index + 1 < len(segments):
            end_time = min(end_time, int(segments[index + 1].start_time))
        if end_time <= start_time:
            continue
        result.append(
            _new_asr_data_seg(
                ASRDataSeg,
                text=segment.text,
                translated_text=getattr(segment, "translated_text", ""),
                start_time=start_time,
                end_time=end_time,
            )
        )
    return result


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


def load_best_asr_data(
    work_paths: WorkPaths,
    optimizer_root: Path = DEFAULT_OPTIMIZER_ROOT,
):
    ASRData, ASRDataSeg, *_ = _load_optimizer_types(optimizer_root)
    if work_paths.normalized_manifest.exists():
        return ASRData.from_json(read_json(work_paths.normalized_manifest, default={}))
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


def _load_optimizer_types(
    optimizer_root: Path,
    need_translator: bool = False,
):
    optimizer_root = optimizer_root.resolve()
    if not optimizer_root.exists():
        if DEFAULT_OPTIMIZER_ROOT.exists():
            LOGGER.warning("Optimizer root not found: %s; falling back to %s", optimizer_root, DEFAULT_OPTIMIZER_ROOT)
            optimizer_root = DEFAULT_OPTIMIZER_ROOT
        else:
            raise FileNotFoundError(f"Optimizer root not found: {optimizer_root}")

    from optimizer.asr_data import ASRData, ASRDataSeg  # type: ignore
    from optimizer.splitter import SubtitleSplitter  # type: ignore

    if need_translator:
        from optimizer.translator import SubtitleTranslator  # type: ignore

        return ASRData, ASRDataSeg, SubtitleSplitter, SubtitleTranslator
    return ASRData, ASRDataSeg, SubtitleSplitter
