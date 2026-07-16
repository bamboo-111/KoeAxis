from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qwen_asr import optimizer_bridge_adapter as _adapter
from qwen_asr import optimizer_bridge_guards as _guards
from qwen_asr.glossary import build_glossary_prompt
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
    max_word_count_cjk: int = 18,
    max_word_count_english: int = 18,
    prompt_limit_ratio: float = 0.8,
    disable_thinking: bool = False,
    llm_extra_body: dict[str, Any] | None = None,
    timeout: float = 120.0,
    split_mode: str = "rule",
) -> None:
    del llm_model, base_url, api_key, thread_num, prompt_limit_ratio, disable_thinking, llm_extra_body
    if split_mode != "rule":
        raise ValueError(f"Unsupported split mode: {split_mode}. Only 'rule' is available.")
    ASRData, ASRDataSeg, SubtitleSplitter = load_optimizer_types(optimizer_root)
    source = _adapter.aligned_manifest_to_asr_data(work_paths, ASRData, ASRDataSeg)
    if not source.segments:
        raise RuntimeError("No aligned token data available for split stage.")

    LOGGER.info("Using the production rule split implementation.")
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
        processed = postprocess_split_segments(splitter, processed)
        result = ASRData(processed)
    finally:
        splitter.stop()

    result.segments = _guards.extract_protected_short_responses(
        source.segments,
        result.segments,
        normalize_content=_adapter._normalize_content,
        new_asr_data_seg=_adapter._new_asr_data_seg,
    )
    result.segments = extend_protected_short_display_segments(result.segments)
    result.segments = _guards.extract_protected_short_responses(
        source.segments,
        result.segments,
        normalize_content=_adapter._normalize_content,
        new_asr_data_seg=_adapter._new_asr_data_seg,
    )
    _guards.validate_split_content_preserved(
        source.segments,
        result.segments,
        normalize_content=_adapter._normalize_content,
    )
    _guards.validate_split_short_responses_preserved(
        source.segments,
        result.segments,
        normalize_content=_adapter._normalize_content,
    )
    write_json_atomic(work_paths.split_manifest, result.to_json())
    work_paths.split_srt.write_text(result.to_srt(), encoding="utf-8")


def postprocess_split_segments(splitter: Any, segments: list[Any]) -> list[Any]:
    processed = splitter._smooth_short_fillers(segments)  # noqa: SLF001
    processed = splitter._smooth_readability_segments(processed)  # noqa: SLF001
    merge_tail = getattr(splitter, "_merge_tail_fragments", None)
    if callable(merge_tail):
        processed = merge_tail(processed)
    processed = splitter._smooth_readability_segments(processed)  # noqa: SLF001
    return extend_protected_short_display_segments(processed)


def extend_protected_short_display_segments(segments: list[Any]) -> list[Any]:
    try:
        from optimizer.splitter import _extend_protected_short_display_durations
    except ImportError:
        return segments
    return _extend_protected_short_display_durations(segments)


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
    ASRData, _, _, SubtitleTranslator = load_optimizer_types(
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

    translated_payload = result.to_json()
    merge_translation_suspect_metadata(translated_payload, result.segments)
    write_json_atomic(work_paths.translated_manifest, translated_payload)
    work_paths.translated_srt.write_text(result.to_srt(), encoding="utf-8")


def merge_translation_suspect_metadata(payload: dict[str, Any], segments: list[Any]) -> None:
    for index, segment in enumerate(segments, 1):
        item = payload.get(str(index))
        if not isinstance(item, dict):
            continue
        asr_suspect = bool(getattr(segment, "asr_suspect", False))
        needs_audio_review = bool(getattr(segment, "needs_audio_review", False))
        suspect_types = getattr(segment, "suspect_types", [])
        if not isinstance(suspect_types, list):
            suspect_types = [str(suspect_types)] if str(suspect_types).strip() else []
        reason = str(getattr(segment, "suspect_reason", "")).strip()
        confidence = getattr(segment, "suspect_confidence", 1.0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 1.0
        item["asr_suspect"] = asr_suspect
        item["needs_audio_review"] = needs_audio_review
        item["suspect_types"] = [str(value).strip() for value in suspect_types if str(value).strip()]
        item["suspect_reason"] = reason
        item["suspect_confidence"] = confidence


def load_optimizer_types(
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
