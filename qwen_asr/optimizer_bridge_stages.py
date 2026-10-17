from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from qwen_asr import optimizer_bridge_adapter as _adapter
from qwen_asr import optimizer_bridge_guards as _guards
from qwen_asr.glossary import build_glossary_prompt
from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress
from qwen_asr.credentials import resolve_llm_api_key
from qwen_asr.semantic_boundary_merge import merge_adjacent_boundaries
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
    split_boundaries: str = "bunsetsu",
    speaker_enroll: Path | str | None = None,
    speakers: list[str] | None = None,
    speaker_min_margin: float = 0.1,
    speaker_weights: Path | str | None = None,
    speaker_device: str = "auto",
) -> None:
    del prompt_limit_ratio
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
        bunsetsu_boundaries=split_boundaries != "off",
    )
    try:
        processed = splitter._process_by_rules(source.segments)  # noqa: SLF001
        processed = postprocess_split_segments(splitter, processed)
        assigner = None
        if speaker_enroll:
            assigner = load_speaker_assigner(
                work_paths,
                source.segments,
                speaker_enroll=speaker_enroll,
                speakers=speakers,
                min_margin=speaker_min_margin,
                weights=speaker_weights,
                device=speaker_device,
            )
        if assigner is not None:
            processed, cuts = assigner.split_cues(
                processed,
                lambda text, start, end: _adapter._new_asr_data_seg(
                    ASRDataSeg, text=text, translated_text="", start_time=start, end_time=end
                ),
            )
            attach_speakers(assigner, processed)
            LOGGER.info("Speaker split: %d cuts at confident speaker changes (%s)", cuts, ", ".join(assigner.members))
        resolved_api_key = resolve_llm_api_key(api_key, base_url)
        if llm_model and base_url and resolved_api_key:
            processed = merge_adjacent_boundaries(
                processed,
                model=llm_model,
                base_url=base_url,
                api_key=resolved_api_key,
                workers=thread_num,
                max_chars=max(32, max_word_count_cjk * 2),
                disable_thinking=disable_thinking,
                llm_extra_body=llm_extra_body,
                timeout=timeout,
                speaker_min_margin=speaker_min_margin,
            )
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
    payload = result.to_json()
    if assigner is not None:
        for index, label in enumerate(assigner.label_cues(result.segments), 1):
            item = payload.get(str(index))
            if isinstance(item, dict):
                item["speaker"] = label.speaker if label.margin >= speaker_min_margin else ""
                item["speaker_margin"] = round(min(label.margin, 1.0), 3)
    write_json_atomic(work_paths.split_manifest, payload)
    work_paths.split_srt.write_text(result.to_srt(), encoding="utf-8")


def load_speaker_assigner(
    work_paths: WorkPaths,
    tokens: list[Any],
    *,
    speaker_enroll: Path | str,
    speakers: list[str] | None,
    min_margin: float,
    weights: Path | str | None,
    device: str,
):
    """Speaker decisions from the cached track (computed after align); None when unavailable (logged).

    Speaker labels are auxiliary: a missing model or a scoring failure falls back to splitting without them.
    """
    from qwen_asr.speaker.assign import SpeakerAssigner
    from qwen_asr.speaker.voiceprint import ensure_speaker_track

    try:
        track = ensure_speaker_track(
            work_paths.audio_path,
            work_paths.aligned_manifest.parent,
            speaker_enroll,
            speakers=speakers,
            weights=weights,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 - speaker labels must never fail the split stage
        LOGGER.warning("Speaker identification unavailable, splitting without it: %s", exc)
        return None
    return SpeakerAssigner.from_track(track, tokens, min_margin=min_margin)


def attach_speakers(assigner: Any, segments: list[Any]) -> None:
    """Set .speaker / .speaker_margin on each cue for the LLM merge (prompt hint + cross-speaker merge guard)."""
    for segment, label in zip(segments, assigner.label_cues(segments), strict=True):
        segment.speaker = label.speaker
        segment.speaker_margin = label.margin


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

    split_payload = read_json(work_paths.split_manifest, default={})
    asr_data = ASRData.from_json(split_payload)
    with_speakers = attach_manifest_speakers(split_payload, asr_data.segments)
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
    if with_speakers:
        for index, segment in enumerate(result.segments, 1):
            item = translated_payload.get(str(index))
            if isinstance(item, dict) and hasattr(segment, "speaker"):
                item["speaker"] = segment.speaker
    mark_alignment_repaired_suspects(translated_payload, read_json(work_paths.aligned_manifest, default=[]))
    write_json_atomic(work_paths.translated_manifest, translated_payload)
    work_paths.translated_srt.write_text(result.to_srt(), encoding="utf-8")


def attach_manifest_speakers(payload: Any, segments: list[Any]) -> bool:
    """Copy split_segments.json "speaker" labels onto the loaded segments (matched by time and text)."""
    if not isinstance(payload, dict):
        return False
    labels: dict[tuple[int, int, str], str] = {}
    for item in payload.values():
        if isinstance(item, dict) and "speaker" in item:
            key = (int(item.get("start_time", 0)), int(item.get("end_time", 0)), str(item.get("original_subtitle", "")))
            labels[key] = str(item.get("speaker") or "")
    if not labels:
        return False
    for segment in segments:
        segment.speaker = labels.get((int(segment.start_time), int(segment.end_time), str(segment.text)), "")
    return True


ALIGNMENT_REPAIRED_SUSPECT_TYPE = "alignment_repaired"


def mark_alignment_repaired_suspects(payload: dict[str, Any], aligned_payload: Any) -> int:
    """Send subtitles overlapping a locally repaired alignment span to audio review.

    Collapsed alignment usually means the transcript there was not really spoken (duplicated
    phrases, garbled chants), so these lines are ASR suspects as well as timing estimates.
    """
    spans: list[tuple[int, int, str]] = []
    for item in aligned_payload if isinstance(aligned_payload, list) else []:
        recovery = item.get("recovery") if isinstance(item, dict) else None
        if not isinstance(recovery, dict) or recovery.get("method") != "align-local-repair":
            continue
        for span in recovery.get("repaired_spans") or []:
            try:
                start_ms = int(round(float(span["start_time"]) * 1000))
                end_ms = int(round(float(span["end_time"]) * 1000))
            except (KeyError, TypeError, ValueError):
                continue
            spans.append((start_ms, end_ms, str(span.get("collapsed_text", ""))))
    marked = 0
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        try:
            start_ms, end_ms = int(item.get("start_time", 0)), int(item.get("end_time", 0))
        except (TypeError, ValueError):
            continue
        hits = [text for span_start, span_end, text in spans if start_ms < span_end and end_ms > span_start]
        if not hits:
            continue
        types = [str(value) for value in item.get("suspect_types") or [] if str(value).strip()]
        if ALIGNMENT_REPAIRED_SUSPECT_TYPE not in types:
            types.append(ALIGNMENT_REPAIRED_SUSPECT_TYPE)
        item["suspect_types"] = types
        item["needs_audio_review"] = True
        reason = str(item.get("suspect_reason", "")).strip()
        note = "对齐局部塌缩已修补，可能有未说出的文本: " + " / ".join(hits)
        item["suspect_reason"] = f"{reason}; {note}" if reason else note
        marked += 1
    return marked


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
