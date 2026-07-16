from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from qwen_asr.defaults import DEFAULT_LLM_EXTRA_BODY_JSON
from qwen_asr import mimo_audio as _mimo_audio
from qwen_asr import mimo_inputs as _mimo_inputs
from qwen_asr.storage import ensure_directory


@dataclass(frozen=True)
class MiMoConfig:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    temperature: float
    disable_thinking: bool
    extra_body: dict[str, Any] | None
    compact_output: bool


@dataclass(frozen=True)
class SegmentTask:
    index: int
    total: int
    segment: dict[str, Any]
    subtitle_entries: dict[str, Any]
    glossary_entries: list[dict[str, str]]
    audio_path: Path


@dataclass(frozen=True)
class SegmentResult:
    segment_id: str
    report_item: dict[str, Any]
    updates: dict[str, dict[str, str]]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workdir = Path(args.workdir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else workdir / "experiments" / "mimo-proofread"
    ensure_directory(output_dir)

    segments, translated = _load_pipeline_inputs(workdir)
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("segments.json is missing or empty")
    if not isinstance(translated, dict) or not translated:
        raise RuntimeError("translated_segments.json is missing or empty")

    glossary = _load_glossary(Path(args.glossary_xlsx).resolve() if args.glossary_xlsx else None)
    client = OpenAI(
        base_url=_normalize_base_url(args.base_url),
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=0,
    )
    config = MiMoConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        disable_thinking=args.disable_thinking,
        extra_body=_parse_extra_body(args.llm_extra_body_json),
        compact_output=args.compact_output,
    )

    manifest_path = output_dir / "mimo_proofread_segments.json"
    report_path = output_dir / "mimo_proofread_report.json"
    srt_path = output_dir / "subtitles.mimo-proofread.srt"
    if args.proofread_mode == "two-stage-nearby":
        return _run_two_stage_nearby(
            args=args,
            client=client,
            config=config,
            segments=segments,
            translated=translated,
            glossary=glossary,
            output_dir=output_dir,
            manifest_path=manifest_path,
            report_path=report_path,
            srt_path=srt_path,
        )

    branch = _load_existing_branch(manifest_path, translated)
    report = _load_existing_report(report_path)
    completed_segments = _completed_segment_ids(report)
    segment_limit = args.segment_limit if args.segment_limit and args.segment_limit > 0 else None
    selected_segments = segments[:segment_limit]

    tasks: list[SegmentTask] = []
    for index, segment in enumerate(selected_segments, start=1):
        segment_id = str(segment.get("segment_id", f"segment_{index:06d}"))
        if args.resume and segment_id in completed_segments:
            print(f"[{index}/{len(selected_segments)}] {segment_id} skipped existing checkpoint", flush=True)
            continue
        report = [item for item in report if str(item.get("segment_id")) != segment_id]
        audio_path = Path(str(segment.get("audio_path", "")))
        if not audio_path.exists():
            report.append({
                "segment_id": segment_id,
                "status": "skipped",
                "error": f"audio not found: {audio_path}",
            })
            _write_outputs(manifest_path, report_path, srt_path, branch, report)
            continue

        covered = _subtitle_entries_for_segment(translated, segment)
        if not covered:
            report.append({
                "segment_id": segment_id,
                "status": "skipped",
                "error": "no translated subtitle entries in segment window",
            })
            _write_outputs(manifest_path, report_path, srt_path, branch, report)
            continue

        relevant_glossary = _filter_glossary(glossary, covered, args.max_glossary_entries)
        tasks.append(
            SegmentTask(
                index=index,
                total=len(selected_segments),
                segment=segment,
                subtitle_entries=covered,
                glossary_entries=relevant_glossary,
                audio_path=audio_path,
            )
        )

    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_to_task = {
                executor.submit(
                    _process_segment_task,
                    task=task,
                    client=client,
                    config=config,
                    max_retries=args.max_retries,
                    base_delay=args.retry_base_delay,
                    max_delay=args.retry_max_delay,
                    keep_raw=args.keep_raw,
                ): task
                for task in tasks
            }
            for future in as_completed(future_to_task):
                result = future.result()
                report.append(result.report_item)
                if result.report_item.get("status") == "completed":
                    completed_segments.add(result.segment_id)
                    _apply_branch_updates(branch, result.updates, source="mimo-segment-audio")
                _write_outputs(manifest_path, report_path, srt_path, branch, report)

    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(f"srt={srt_path}")
    return 0


def _load_pipeline_inputs(workdir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return _mimo_inputs.load_pipeline_inputs(workdir)


def _validate_translated_manifest_complete_for_split(
    translated: dict[str, Any],
    split: dict[str, Any],
) -> None:
    _mimo_inputs.validate_translated_manifest_complete_for_split(translated, split)


def _manifest_key_sort(value: str) -> tuple[int, int | str]:
    return _mimo_inputs.manifest_key_sort(value)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an independent MiMo audio proofread branch.")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--base-url", default="https://api.xiaomimimo.com/v1")
    parser.add_argument("--api-key", default=os.environ.get("MIMO_API_KEY", ""))
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--glossary-xlsx", default="")
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--llm-extra-body-json", default=DEFAULT_LLM_EXTRA_BODY_JSON)
    parser.add_argument("--compact-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--segment-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--proofread-mode",
        choices=["segment-audio", "two-stage-nearby"],
        default="segment-audio",
        help="segment-audio keeps the original full segment audio review; two-stage-nearby runs text QA then nearby audio review for suspicious lines.",
    )
    parser.add_argument("--nearby-padding-s", type=float, default=1.5)
    parser.add_argument("--nearby-context-subtitles", type=int, default=1)
    parser.add_argument("--nearby-audio-workers", type=int, default=1)
    parser.add_argument(
        "--nearby-batch-size",
        type=int,
        default=1,
        help="Batch adjacent suspect subtitle IDs into one nearby audio request. Use 1 to keep per-subtitle review.",
    )
    parser.add_argument(
        "--nearby-batch-max-gap-s",
        type=float,
        default=8.0,
        help="Maximum gap between suspect subtitles in one nearby audio batch.",
    )
    parser.add_argument("--stage1-confidence-threshold", type=float, default=0.75)
    parser.add_argument("--stage1-apply-threshold", type=float, default=0.8)
    parser.add_argument("--stage2-apply-threshold", type=float, default=0.9)
    parser.add_argument(
        "--audio-review-scope",
        choices=["suspects", "all"],
        default="suspects",
        help="Review only text-stage suspects or every subtitle with nearby audio.",
    )
    parser.add_argument(
        "--diagnostic-all",
        action="store_true",
        help="Allow --audio-review-scope all for explicit diagnostic experiments only.",
    )
    parser.add_argument("--max-glossary-entries", type=int, default=80)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=8.0)
    parser.add_argument("--retry-max-delay", type=float, default=90.0)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("MiMo audio proofread requires --api-key or MIMO_API_KEY")
    if args.audio_review_scope == "all" and not args.diagnostic_all:
        parser.error("--audio-review-scope all is diagnostic-only; pass --diagnostic-all explicitly")
    return args


def _process_segment_task(
    *,
    task: SegmentTask,
    client: OpenAI,
    config: MiMoConfig,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    keep_raw: bool,
) -> SegmentResult:
    segment_id = str(task.segment.get("segment_id", f"segment_{task.index:06d}"))
    started = time.monotonic()
    print(
        f"[{task.index}/{task.total}] {segment_id} start "
        f"subtitles={len(task.subtitle_entries)} glossary={len(task.glossary_entries)}",
        flush=True,
    )
    try:
        raw_content, usage, suggestions = _request_suggestions_with_parse_retries(
            lambda: _call_mimo_with_retries(
                client=client,
                config=config,
                segment=task.segment,
                audio_path=task.audio_path,
                subtitle_entries=task.subtitle_entries,
                glossary_entries=task.glossary_entries,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            ),
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[{task.index}/{task.total}] {segment_id} failed: {exc}", flush=True)
        return SegmentResult(
            segment_id=segment_id,
            report_item={
                "segment_id": segment_id,
                "status": "failed",
                "subtitle_ids": list(task.subtitle_entries.keys()),
                "error": str(exc),
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            },
            updates={},
        )

    updates: dict[str, dict[str, str]] = {}
    for item in suggestions:
        subtitle_id = str(item.get("id", "")).strip()
        suggested = str(item.get("suggested_translation", "")).strip()
        if subtitle_id and subtitle_id in task.subtitle_entries and suggested:
            current = str(task.subtitle_entries[subtitle_id].get("translated_subtitle", "")).strip()
            if suggested != current:
                updates[subtitle_id] = {"translated_subtitle": suggested}

    print(
        f"[{task.index}/{task.total}] {segment_id} "
        f"subtitles={len(task.subtitle_entries)} suggestions={len(suggestions)} applied={len(updates)}",
        flush=True,
    )
    return SegmentResult(
        segment_id=segment_id,
        report_item={
            "segment_id": segment_id,
            "status": "completed",
            "subtitle_ids": list(task.subtitle_entries.keys()),
            "suggestion_count": len(suggestions),
            "applied_count": len(updates),
            "glossary_count": len(task.glossary_entries),
            "usage": _usage_to_dict(usage),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "suggestions": suggestions,
            "raw_content": raw_content if keep_raw else "",
        },
        updates=updates,
    )


def _run_two_stage_nearby(
    *,
    args: argparse.Namespace,
    client: OpenAI,
    config: MiMoConfig,
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
    glossary: list[dict[str, str]],
    output_dir: Path,
    manifest_path: Path,
    report_path: Path,
    srt_path: Path,
) -> int:
    stage1_report_path = output_dir / "mimo_stage1_text_suspect_report.json"
    stage2_report_path = output_dir / "mimo_stage2_nearby_audio_report.json"
    clips_dir = output_dir / "nearby-audio-clips"
    ensure_directory(clips_dir)

    branch = _load_existing_branch(manifest_path, translated)
    stage1_report = _load_existing_report(stage1_report_path)
    stage2_report = _load_existing_report(stage2_report_path)
    started = time.monotonic()

    segment_limit = args.segment_limit if args.segment_limit and args.segment_limit > 0 else None
    selected_segments = segments[:segment_limit]

    if _translated_manifest_has_suspect_metadata(translated):
        stage1_report = _build_manifest_suspect_report(
            translated,
            confidence_threshold=args.stage1_confidence_threshold,
        )
        _write_two_stage_outputs(
            manifest_path,
            report_path,
            stage1_report_path,
            stage2_report_path,
            srt_path,
            branch,
            stage1_report,
            stage2_report,
            started=started,
            translated=translated,
        )
    else:
        completed_stage1 = _completed_segment_ids(stage1_report)
        stage1_tasks: list[SegmentTask] = []
        for index, segment in enumerate(selected_segments, start=1):
            segment_id = str(segment.get("segment_id", f"segment_{index:06d}"))
            if args.resume and segment_id in completed_stage1:
                print(f"[stage1 {index}/{len(selected_segments)}] {segment_id} skipped existing checkpoint", flush=True)
                continue
            covered = _subtitle_entries_for_segment(translated, segment)
            if not covered:
                stage1_report = _replace_report_item(
                    stage1_report,
                    segment_id,
                    {
                        "segment_id": segment_id,
                        "status": "skipped",
                        "error": "no translated subtitle entries in segment window",
                    },
                )
                _write_two_stage_outputs(
                    manifest_path,
                    report_path,
                    stage1_report_path,
                    stage2_report_path,
                    srt_path,
                    branch,
                    stage1_report,
                    stage2_report,
                    started=started,
                    translated=translated,
                )
                continue
            stage1_tasks.append(
                SegmentTask(
                    index=index,
                    total=len(selected_segments),
                    segment=segment,
                    subtitle_entries=covered,
                    glossary_entries=_filter_glossary(glossary, covered, args.max_glossary_entries),
                    audio_path=Path(str(segment.get("audio_path", ""))),
                )
            )

        if stage1_tasks:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                future_to_task = {
                    executor.submit(
                        _process_stage1_text_task,
                        task=task,
                        client=client,
                        config=config,
                        max_retries=args.max_retries,
                        base_delay=args.retry_base_delay,
                        max_delay=args.retry_max_delay,
                        keep_raw=args.keep_raw,
                        suspect_confidence_threshold=args.stage1_confidence_threshold,
                        apply_confidence_threshold=args.stage1_apply_threshold,
                    ): task
                    for task in stage1_tasks
                }
                for future in as_completed(future_to_task):
                    result = future.result()
                    stage1_report = _replace_report_item(stage1_report, result.segment_id, result.report_item)
                    _write_two_stage_outputs(
                        manifest_path,
                        report_path,
                        stage1_report_path,
                        stage2_report_path,
                        srt_path,
                        branch,
                        stage1_report,
                        stage2_report,
                        started=started,
                        translated=translated,
                    )

    suspect_ids = _collect_stage1_suspects(stage1_report)
    review_ids = (
        sorted(
            (str(key) for key, value in translated.items() if str(key).isdigit() and isinstance(value, dict)),
            key=int,
        )
        if args.audio_review_scope == "all"
        else suspect_ids
    )
    pending_review_ids = _pending_review_ids(review_ids, stage2_report, resume=args.resume)
    print(
        f"stage2 nearby audio review: scope={args.audio_review_scope} "
        f"targets={len(review_ids)} suspects={len(suspect_ids)} pending={len(pending_review_ids)}",
        flush=True,
    )

    if pending_review_ids:
        review_batches = _build_nearby_audio_batches(
            subtitle_ids=pending_review_ids,
            segments=selected_segments,
            translated=translated,
            context_subtitles=max(0, args.nearby_context_subtitles),
            batch_size=max(1, args.nearby_batch_size),
            max_gap_s=max(0.0, args.nearby_batch_max_gap_s),
        )
        with ThreadPoolExecutor(max_workers=max(1, args.nearby_audio_workers)) as executor:
            future_to_batch = {
                executor.submit(
                    _process_stage2_nearby_audio_batch_task,
                    target_ids=target_ids,
                    client=client,
                    config=config,
                    segments=selected_segments,
                    translated=translated,
                    glossary=glossary,
                    clips_dir=clips_dir,
                    context_subtitles=max(0, args.nearby_context_subtitles),
                    padding_s=max(0.0, args.nearby_padding_s),
                    max_glossary_entries=args.max_glossary_entries,
                    max_retries=args.max_retries,
                    base_delay=args.retry_base_delay,
                    max_delay=args.retry_max_delay,
                    keep_raw=args.keep_raw,
                    apply_confidence_threshold=args.stage2_apply_threshold,
                ): target_ids
                for target_ids in review_batches
            }
            for future in as_completed(future_to_batch):
                target_ids = future_to_batch[future]
                results = future.result()
                for result in results:
                    report_id = str(result.report_item.get("id", result.segment_id))
                    stage2_report = _replace_report_item(stage2_report, report_id, result.report_item, key="id")
                    if result.report_item.get("status") != "completed":
                        continue
                    _apply_branch_updates(branch, result.updates, source="mimo-nearby-audio")
                print(f"stage2 batch completed ids={','.join(target_ids)}", flush=True)
                _write_two_stage_outputs(
                    manifest_path,
                    report_path,
                    stage1_report_path,
                    stage2_report_path,
                    srt_path,
                    branch,
                    stage1_report,
                    stage2_report,
                    started=started,
                    translated=translated,
                )

    _write_two_stage_outputs(
        manifest_path,
        report_path,
        stage1_report_path,
        stage2_report_path,
        srt_path,
        branch,
        stage1_report,
        stage2_report,
        started=started,
        translated=translated,
    )
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(f"stage1_report={stage1_report_path}")
    print(f"stage2_report={stage2_report_path}")
    print(f"srt={srt_path}")
    return 0


def _process_stage1_text_task(
    *,
    task: SegmentTask,
    client: OpenAI,
    config: MiMoConfig,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    keep_raw: bool,
    suspect_confidence_threshold: float,
    apply_confidence_threshold: float,
) -> SegmentResult:
    segment_id = str(task.segment.get("segment_id", f"segment_{task.index:06d}"))
    started = time.monotonic()
    print(
        f"[stage1 {task.index}/{task.total}] {segment_id} start "
        f"subtitles={len(task.subtitle_entries)} glossary={len(task.glossary_entries)}",
        flush=True,
    )
    try:
        raw_content, usage, suggestions = _request_suggestions_with_parse_retries(
            lambda: _call_mimo_text_stage1_with_retries(
                client=client,
                config=config,
                segment=task.segment,
                subtitle_entries=task.subtitle_entries,
                glossary_entries=task.glossary_entries,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            ),
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"[stage1 {task.index}/{task.total}] {segment_id} failed: {exc}", flush=True)
        raw_value = locals().get("raw_content", "")
        return SegmentResult(
            segment_id=segment_id,
            report_item={
                "segment_id": segment_id,
                "status": "failed",
                "subtitle_ids": list(task.subtitle_entries.keys()),
                "error": str(exc),
                "raw_content": raw_value if keep_raw else "",
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            },
            updates={},
        )

    suspect_ids: set[str] = set()
    normalized_suggestions: list[dict[str, Any]] = []
    for item in suggestions:
        normalized = _normalize_qa_item(item)
        subtitle_id = str(normalized.get("id", "")).strip()
        if not subtitle_id or subtitle_id not in task.subtitle_entries:
            continue
        normalized_suggestions.append(normalized)
        confidence = _coerce_confidence(normalized.get("confidence"), default=1.0)
        asr_suspect = _coerce_bool(normalized.get("asr_suspect"))
        needs_audio_review = _coerce_bool(normalized.get("needs_audio_review"))
        error_type = str(normalized.get("error_type", "")).strip()
        if (
            asr_suspect
            or needs_audio_review
            or error_type == "needs_context"
            or confidence < suspect_confidence_threshold
        ):
            suspect_ids.add(subtitle_id)

    print(
        f"[stage1 {task.index}/{task.total}] {segment_id} "
        f"suggestions={len(normalized_suggestions)} suspects={len(suspect_ids)} applied=0",
        flush=True,
    )
    return SegmentResult(
        segment_id=segment_id,
        report_item={
            "segment_id": segment_id,
            "status": "completed",
            "subtitle_ids": list(task.subtitle_entries.keys()),
            "suggestion_count": len(normalized_suggestions),
            "applied_count": 0,
            "stage1_text_updates_disabled": True,
            "suspect_ids": sorted(suspect_ids, key=int),
            "suspect_count": len(suspect_ids),
            "glossary_count": len(task.glossary_entries),
            "usage": _usage_to_dict(usage),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "suggestions": normalized_suggestions,
            "raw_content": raw_content if keep_raw else "",
        },
        updates={},
    )


def _process_stage2_nearby_audio_task(
    *,
    subtitle_id: str,
    client: OpenAI,
    config: MiMoConfig,
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
    glossary: list[dict[str, str]],
    clips_dir: Path,
    context_subtitles: int,
    padding_s: float,
    max_glossary_entries: int,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    keep_raw: bool,
    apply_confidence_threshold: float,
) -> SegmentResult:
    return _process_stage2_nearby_audio_batch_task(
        target_ids=[subtitle_id],
        client=client,
        config=config,
        segments=segments,
        translated=translated,
        glossary=glossary,
        clips_dir=clips_dir,
        context_subtitles=context_subtitles,
        padding_s=padding_s,
        max_glossary_entries=max_glossary_entries,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        keep_raw=keep_raw,
        apply_confidence_threshold=apply_confidence_threshold,
    )[0]


def _process_stage2_nearby_audio_batch_task(
    *,
    target_ids: list[str],
    client: OpenAI,
    config: MiMoConfig,
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
    glossary: list[dict[str, str]],
    clips_dir: Path,
    context_subtitles: int,
    padding_s: float,
    max_glossary_entries: int,
    max_retries: int,
    base_delay: float,
    max_delay: float,
    keep_raw: bool,
    apply_confidence_threshold: float,
) -> list[SegmentResult]:
    started = time.monotonic()
    if not target_ids:
        return []
    try:
        segment = _segment_for_subtitle_id(target_ids[0], segments, translated)
        entries = _nearby_entries_for_subtitle_ids(target_ids, translated, context_subtitles)
        target_entries = {
            subtitle_id: translated[subtitle_id]
            for subtitle_id in target_ids
            if subtitle_id in translated and isinstance(translated[subtitle_id], dict)
        }
        segment_audio_path = Path(str(segment.get("audio_path", "")))
        source_audio_path = Path(str(segment.get("source_audio_path", "")))
        audio_path = source_audio_path if source_audio_path.exists() else segment_audio_path
        if not audio_path.exists():
            raise FileNotFoundError(f"audio not found: {audio_path}")
        clip_path, clip_meta = _write_nearby_audio_clip(
            subtitle_id="-".join(target_ids),
            segment=segment,
            entries=entries,
            audio_path=audio_path,
            clips_dir=clips_dir,
            padding_s=padding_s,
        )
        relevant_glossary = _filter_glossary(glossary, entries, max_glossary_entries)
        raw_content, usage, suggestions = _request_suggestions_with_parse_retries(
            lambda: _call_mimo_nearby_audio_with_retries(
                client=client,
                config=config,
                segment=segment,
                target_ids=target_ids,
                target_entries=target_entries,
                nearby_entries=entries,
                glossary_entries=relevant_glossary,
                clip_path=clip_path,
                clip_meta=clip_meta,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
            ),
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        if len(target_ids) > 1:
            print(
                f"[stage2] subtitles {','.join(target_ids)} failed as batch; retrying individually: {exc}",
                flush=True,
            )
            fallback_results: list[SegmentResult] = []
            for subtitle_id in target_ids:
                for result in _process_stage2_nearby_audio_batch_task(
                    target_ids=[subtitle_id],
                    client=client,
                    config=config,
                    segments=segments,
                    translated=translated,
                    glossary=glossary,
                    clips_dir=clips_dir,
                    context_subtitles=context_subtitles,
                    padding_s=padding_s,
                    max_glossary_entries=max_glossary_entries,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    keep_raw=keep_raw,
                    apply_confidence_threshold=apply_confidence_threshold,
                ):
                    result.report_item["fallback_from_batch"] = list(target_ids)
                    result.report_item["batch_error"] = str(exc)
                    fallback_results.append(result)
            return fallback_results
        print(f"[stage2] subtitles {','.join(target_ids)} failed: {exc}", flush=True)
        return [
            SegmentResult(
                segment_id=subtitle_id,
                report_item={
                    "id": subtitle_id,
                    "status": "failed",
                    "target_ids": target_ids,
                    "error": str(exc),
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                },
                updates={},
            )
            for subtitle_id in target_ids
        ]

    suggestions_by_id: dict[str, list[dict[str, Any]]] = {subtitle_id: [] for subtitle_id in target_ids}
    for item in suggestions:
        normalized = _normalize_qa_item(item)
        reviewed_id = str(normalized.get("id", "")).strip()
        if reviewed_id not in suggestions_by_id or reviewed_id not in translated:
            continue
        suggestions_by_id[reviewed_id].append(normalized)

    results: list[SegmentResult] = []
    for subtitle_id in target_ids:
        updates: dict[str, dict[str, Any]] = {}
        applied_evidence: dict[str, dict[str, Any]] = {}
        rejections: list[dict[str, Any]] = []
        normalized_suggestions = suggestions_by_id.get(subtitle_id, [])
        for normalized in normalized_suggestions:
            reviewed_id = str(normalized.get("id", "")).strip()
            if reviewed_id != subtitle_id or reviewed_id not in translated:
                continue
            application = _evaluate_stage2_suggestion(
                subtitle_id=subtitle_id,
                normalized=normalized,
                current_item=translated[reviewed_id],
                apply_confidence_threshold=apply_confidence_threshold,
            )
            if application.rejection is not None:
                rejections.append(application.rejection)
                continue
            if application.updates:
                applied_evidence[reviewed_id] = application.evidence
                updates[reviewed_id] = application.updates
        results.append(
            SegmentResult(
                segment_id=subtitle_id,
                report_item={
                    "id": subtitle_id,
                    "status": "completed",
                    "target_ids": target_ids,
                    "clip_path": str(clip_path),
                    "clip_duration_s": clip_meta["duration_s"],
                    "segment_id": str(segment.get("segment_id", "")),
                    "suggestion_count": len(normalized_suggestions),
                    "applied_count": len(updates),
                    "rejected_count": len(rejections),
                    "glossary_count": len(relevant_glossary),
                    "usage": _usage_to_dict(usage),
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "suggestions": normalized_suggestions,
                    "rejections": rejections,
                    "applied_evidence": applied_evidence,
                    "raw_content": raw_content if keep_raw else "",
                },
                updates=updates,
            )
        )
        print(
            f"[stage2] subtitle {subtitle_id} batch={len(target_ids)} "
            f"suggestions={len(normalized_suggestions)} applied={len(updates)}",
            flush=True,
        )
    return results


def _load_existing_branch(path: Path, translated: dict[str, Any]) -> dict[str, Any]:
    from qwen_asr.mimo_checkpoints import load_existing_branch

    return load_existing_branch(path, translated)


def _apply_branch_updates(
    branch: dict[str, Any],
    updates: dict[str, dict[str, Any]],
    *,
    source: str,
) -> int:
    from qwen_asr.mimo_guards import apply_branch_updates

    return apply_branch_updates(branch, updates, source=source)


def _load_existing_report(path: Path) -> list[dict[str, Any]]:
    from qwen_asr.mimo_checkpoints import load_existing_report

    return load_existing_report(path)


def _completed_segment_ids(report: list[dict[str, Any]], *, key: str = "segment_id") -> set[str]:
    from qwen_asr.mimo_checkpoints import completed_segment_ids

    return completed_segment_ids(report, key=key)


def _pending_review_ids(
    review_ids: list[str],
    stage2_report: list[dict[str, Any]],
    *,
    resume: bool,
) -> list[str]:
    from qwen_asr.mimo_checkpoints import pending_review_ids

    return pending_review_ids(review_ids, stage2_report, resume=resume)


def _parse_extra_body(value: str) -> dict[str, Any] | None:
    return _mimo_inputs.parse_extra_body(value)


def _maybe_disable_thinking_text(text: str, config: MiMoConfig) -> str:
    from qwen_asr.mimo_requests import maybe_disable_thinking_text

    return maybe_disable_thinking_text(text, config)


def _chat_completion_create(
    *,
    client: OpenAI,
    config: MiMoConfig,
    messages: list[dict[str, Any]],
) -> Any:
    from qwen_asr.mimo_requests import chat_completion_create

    return chat_completion_create(client=client, config=config, messages=messages)


def _write_outputs(
    manifest_path: Path,
    report_path: Path,
    srt_path: Path,
    branch: dict[str, Any],
    report: list[dict[str, Any]],
) -> None:
    from qwen_asr.mimo_outputs import write_outputs

    write_outputs(manifest_path, report_path, srt_path, branch, report)


def _write_two_stage_outputs(
    manifest_path: Path,
    report_path: Path,
    stage1_report_path: Path,
    stage2_report_path: Path,
    srt_path: Path,
    branch: dict[str, Any],
    stage1_report: list[dict[str, Any]],
    stage2_report: list[dict[str, Any]],
    started: float | None = None,
    translated: dict[str, Any] | None = None,
) -> None:
    from qwen_asr.mimo_outputs import write_two_stage_outputs

    write_two_stage_outputs(
        manifest_path,
        report_path,
        stage1_report_path,
        stage2_report_path,
        srt_path,
        branch,
        stage1_report,
        stage2_report,
        started=started,
        translated=translated,
    )


def _stage2_reviewed_candidate_count(completed_audio: list[dict[str, Any]]) -> int:
    from qwen_asr.mimo_outputs import stage2_reviewed_candidate_count

    return stage2_reviewed_candidate_count(completed_audio)


def _replace_report_item(
    report: list[dict[str, Any]],
    item_id: str,
    item: dict[str, Any],
    *,
    key: str = "segment_id",
) -> list[dict[str, Any]]:
    from qwen_asr.mimo_outputs import replace_report_item

    return replace_report_item(report, item_id, item, key=key)


def _collect_stage1_suspects(stage1_report: list[dict[str, Any]]) -> list[str]:
    from qwen_asr.mimo_candidates import collect_stage1_suspects

    return collect_stage1_suspects(stage1_report)


def _translated_manifest_has_suspect_metadata(translated: dict[str, Any]) -> bool:
    from qwen_asr.mimo_candidates import translated_manifest_has_suspect_metadata

    return translated_manifest_has_suspect_metadata(translated)


def _suspect_types_need_audio_review(suspect_types: Any) -> bool:
    from qwen_asr.mimo_candidates import suspect_types_need_audio_review

    return suspect_types_need_audio_review(suspect_types)


def _build_manifest_suspect_report(
    translated: dict[str, Any],
    *,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    from qwen_asr.mimo_candidates import build_manifest_suspect_report

    return build_manifest_suspect_report(translated, confidence_threshold=confidence_threshold)


def _translated_duration_ms(translated: dict[str, Any]) -> int:
    from qwen_asr.mimo_candidates import translated_duration_ms

    return translated_duration_ms(translated)


def _normalize_base_url(base_url: str) -> str:
    return _mimo_inputs.normalize_base_url(base_url)


def _load_glossary(path: Path | None) -> list[dict[str, str]]:
    return _mimo_audio.load_glossary(path)


def _subtitle_entries_for_segment(
    translated: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    return _mimo_audio.subtitle_entries_for_segment(translated, segment)


def _filter_glossary(
    glossary: list[dict[str, str]],
    subtitle_entries: dict[str, Any],
    limit: int,
) -> list[dict[str, str]]:
    return _mimo_audio.filter_glossary(glossary, subtitle_entries, limit)


def _segment_for_subtitle_id(
    subtitle_id: str,
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
) -> dict[str, Any]:
    return _mimo_audio.segment_for_subtitle_id(subtitle_id, segments, translated)


def _build_nearby_audio_batches(
    *,
    subtitle_ids: list[str],
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
    context_subtitles: int,
    batch_size: int,
    max_gap_s: float,
) -> list[list[str]]:
    return _mimo_audio.build_nearby_audio_batches(
        subtitle_ids=subtitle_ids,
        segments=segments,
        translated=translated,
        context_subtitles=context_subtitles,
        batch_size=batch_size,
        max_gap_s=max_gap_s,
    )


def _nearby_entries_for_subtitle_id(
    subtitle_id: str,
    translated: dict[str, Any],
    context_subtitles: int,
) -> dict[str, Any]:
    return _mimo_audio.nearby_entries_for_subtitle_id(subtitle_id, translated, context_subtitles)


def _nearby_entries_for_subtitle_ids(
    subtitle_ids: list[str],
    translated: dict[str, Any],
    context_subtitles: int,
) -> dict[str, Any]:
    return _mimo_audio.nearby_entries_for_subtitle_ids(subtitle_ids, translated, context_subtitles)


def _write_nearby_audio_clip(
    *,
    subtitle_id: str,
    segment: dict[str, Any],
    entries: dict[str, Any],
    audio_path: Path,
    clips_dir: Path,
    padding_s: float,
) -> tuple[Path, dict[str, float]]:
    return _mimo_audio.write_nearby_audio_clip(
        subtitle_id=subtitle_id,
        segment=segment,
        entries=entries,
        audio_path=audio_path,
        clips_dir=clips_dir,
        padding_s=padding_s,
    )


def _normalize_qa_item(item: dict[str, Any]) -> dict[str, Any]:
    from qwen_asr.mimo_guards import normalize_qa_item

    return normalize_qa_item(item)


def _clean_placeholder_value(value: str) -> str:
    from qwen_asr.mimo_guards import clean_placeholder_value

    return clean_placeholder_value(value)


def _safe_suggestion_value(value: str, current: str, *, field: str) -> str:
    from qwen_asr.mimo_guards import safe_suggestion_value

    return safe_suggestion_value(value, current, field=field)


def _ass_acceptance_guard(
    item: dict[str, Any],
    *,
    current_original: str,
    suggested_original: str,
    min_improvement: float = 0.05,
) -> dict[str, Any]:
    from qwen_asr.mimo_guards import ass_acceptance_guard

    return ass_acceptance_guard(
        item,
        current_original=current_original,
        suggested_original=suggested_original,
        min_improvement=min_improvement,
    )


def _ass_fragment_replacement_guard(
    *,
    ass_text: str,
    current_original: str,
    suggested_original: str,
    current_score: float,
    suggested_score: float,
    normalize: Callable[[str], str],
    min_reference_units: int = 12,
    min_current_units: int = 6,
    max_suggested_reference_ratio: float = 0.75,
    min_current_score: float = 0.20,
    high_suggested_score: float = 0.75,
    min_overlap_ratio: float = 0.50,
) -> dict[str, Any]:
    from qwen_asr.mimo_guards import ass_fragment_replacement_guard

    return ass_fragment_replacement_guard(
        ass_text=ass_text,
        current_original=current_original,
        suggested_original=suggested_original,
        current_score=current_score,
        suggested_score=suggested_score,
        normalize=normalize,
        min_reference_units=min_reference_units,
        min_current_units=min_current_units,
        max_suggested_reference_ratio=max_suggested_reference_ratio,
        min_current_score=min_current_score,
        high_suggested_score=high_suggested_score,
        min_overlap_ratio=min_overlap_ratio,
    )


def _original_content_deletion_guard(
    *,
    current_original: str,
    suggested_original: str,
    ass_guard: dict[str, Any],
    min_current_units: int = 4,
    min_dropped_units: int = 3,
) -> dict[str, Any]:
    from qwen_asr.mimo_guards import original_content_deletion_guard

    return original_content_deletion_guard(
        current_original=current_original,
        suggested_original=suggested_original,
        ass_guard=ass_guard,
        min_current_units=min_current_units,
        min_dropped_units=min_dropped_units,
    )


def _original_high_risk_replacement_guard(
    *,
    current_original: str,
    suggested_original: str,
    ass_guard: dict[str, Any],
    max_short_units: int = 3,
    min_expanded_units: int = 12,
) -> dict[str, Any]:
    from qwen_asr.mimo_guards import original_high_risk_replacement_guard

    return original_high_risk_replacement_guard(
        current_original=current_original,
        suggested_original=suggested_original,
        ass_guard=ass_guard,
        max_short_units=max_short_units,
        min_expanded_units=min_expanded_units,
    )


def _is_protected_short_response_signal(signal: str) -> bool:
    from qwen_asr.mimo_guards import is_protected_short_response_signal

    return is_protected_short_response_signal(signal)


def _original_no_ass_substantial_rewrite_guard(
    *,
    current_original: str,
    suggested_original: str,
    ass_guard: dict[str, Any],
) -> dict[str, Any]:
    from qwen_asr.mimo_guards import original_no_ass_substantial_rewrite_guard

    return original_no_ass_substantial_rewrite_guard(
        current_original=current_original,
        suggested_original=suggested_original,
        ass_guard=ass_guard,
    )


def _evaluate_stage2_suggestion(
    *,
    subtitle_id: str,
    normalized: dict[str, Any],
    current_item: dict[str, Any],
    apply_confidence_threshold: float,
):
    from qwen_asr.mimo_application import evaluate_stage2_suggestion

    return evaluate_stage2_suggestion(
        subtitle_id=subtitle_id,
        normalized=normalized,
        current_item=current_item,
        apply_confidence_threshold=apply_confidence_threshold,
    )


def _translation_shortening_guard(
    *,
    current_translation: str,
    suggested_translation: str,
    min_current_units: int = 8,
    max_ratio: float = 0.34,
) -> dict[str, Any]:
    from qwen_asr.mimo_guards import translation_shortening_guard

    return translation_shortening_guard(
        current_translation=current_translation,
        suggested_translation=suggested_translation,
        min_current_units=min_current_units,
        max_ratio=max_ratio,
    )


def _japanese_signal(text: str) -> str:
    from qwen_asr.mimo_guards import japanese_signal

    return japanese_signal(text)


def _cjk_signal_len(text: str) -> int:
    from qwen_asr.mimo_guards import cjk_signal_len

    return cjk_signal_len(text)


def _longest_common_substring_len(left: str, right: str) -> int:
    from qwen_asr.mimo_guards import longest_common_substring_len

    return longest_common_substring_len(left, right)


def _extract_guard_ass_text(item: dict[str, Any]) -> str:
    from qwen_asr.mimo_guards import extract_guard_ass_text

    return extract_guard_ass_text(item)


def _contains_japanese(text: str) -> bool:
    from qwen_asr.mimo_guards import contains_japanese

    return contains_japanese(text)


def _contains_cjk(text: str) -> bool:
    from qwen_asr.mimo_guards import contains_cjk

    return contains_cjk(text)


def _compact_schema_prompt(config: MiMoConfig) -> str:
    from qwen_asr.mimo_requests import compact_schema_prompt

    return compact_schema_prompt(config)


def _coerce_bool(value: Any) -> bool:
    from qwen_asr.mimo_candidates import coerce_bool

    return coerce_bool(value)


def _coerce_confidence(value: Any, *, default: float) -> float:
    from qwen_asr.mimo_candidates import coerce_confidence

    return coerce_confidence(value, default=default)


def _call_mimo(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    audio_path: Path,
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
) -> tuple[str, Any]:
    from qwen_asr.mimo_requests import call_mimo

    return call_mimo(
        client=client,
        config=config,
        segment=segment,
        audio_path=audio_path,
        subtitle_entries=subtitle_entries,
        glossary_entries=glossary_entries,
    )


def _call_mimo_text_stage1(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
) -> tuple[str, Any]:
    from qwen_asr.mimo_requests import call_mimo_text_stage1

    return call_mimo_text_stage1(
        client=client,
        config=config,
        segment=segment,
        subtitle_entries=subtitle_entries,
        glossary_entries=glossary_entries,
    )


def _call_mimo_nearby_audio(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    target_ids: list[str],
    target_entries: dict[str, Any],
    nearby_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    clip_path: Path,
    clip_meta: dict[str, float],
) -> tuple[str, Any]:
    from qwen_asr.mimo_requests import call_mimo_nearby_audio

    return call_mimo_nearby_audio(
        client=client,
        config=config,
        segment=segment,
        target_ids=target_ids,
        target_entries=target_entries,
        nearby_entries=nearby_entries,
        glossary_entries=glossary_entries,
        clip_path=clip_path,
        clip_meta=clip_meta,
    )


def _call_mimo_with_retries(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    audio_path: Path,
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any]:
    from qwen_asr.mimo_requests import call_mimo_with_retries

    return call_mimo_with_retries(
        client=client,
        config=config,
        segment=segment,
        audio_path=audio_path,
        subtitle_entries=subtitle_entries,
        glossary_entries=glossary_entries,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


def _call_mimo_text_stage1_with_retries(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any]:
    from qwen_asr.mimo_requests import call_mimo_text_stage1_with_retries

    return call_mimo_text_stage1_with_retries(
        client=client,
        config=config,
        segment=segment,
        subtitle_entries=subtitle_entries,
        glossary_entries=glossary_entries,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


def _call_mimo_nearby_audio_with_retries(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    target_ids: list[str],
    target_entries: dict[str, Any],
    nearby_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
    clip_path: Path,
    clip_meta: dict[str, float],
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any]:
    from qwen_asr.mimo_requests import call_mimo_nearby_audio_with_retries

    return call_mimo_nearby_audio_with_retries(
        client=client,
        config=config,
        segment=segment,
        target_ids=target_ids,
        target_entries=target_entries,
        nearby_entries=nearby_entries,
        glossary_entries=glossary_entries,
        clip_path=clip_path,
        clip_meta=clip_meta,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


def _is_transient_error(exc: Exception) -> bool:
    from qwen_asr.mimo_requests import is_transient_error

    return is_transient_error(exc)


def _request_suggestions_with_parse_retries(
    request: Callable[[], tuple[str, Any]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> tuple[str, Any, list[dict[str, Any]]]:
    from qwen_asr.mimo_requests import request_suggestions_with_parse_retries

    return request_suggestions_with_parse_retries(
        request,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )


def _parse_suggestions(content: str) -> list[dict[str, Any]]:
    from qwen_asr.mimo_requests import parse_suggestions

    return parse_suggestions(content)


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    from qwen_asr.mimo_requests import usage_to_dict

    return usage_to_dict(usage)


def _to_srt(items: dict[str, Any]) -> str:
    from qwen_asr.mimo_outputs import to_srt

    return to_srt(items)


def _srt_time(ms: int) -> str:
    from qwen_asr.mimo_outputs import srt_time

    return srt_time(ms)


if __name__ == "__main__":
    raise SystemExit(main())
