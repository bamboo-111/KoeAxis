from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from qwen_asr.glossary import read_xlsx_glossary
from qwen_asr.storage import ensure_directory, read_json, write_json_atomic


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
    updates: dict[str, str]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    workdir = Path(args.workdir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else workdir / "experiments" / "mimo-proofread"
    ensure_directory(output_dir)

    segments = read_json(workdir / "manifests" / "segments.json", default=[])
    translated = read_json(workdir / "manifests" / "translated_segments.json", default={})
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
    completed_segments = {
        str(item.get("segment_id"))
        for item in report
        if isinstance(item, dict) and item.get("status") == "completed"
    }
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
                task = future_to_task[future]
                result = future.result()
                report.append(result.report_item)
                if result.report_item.get("status") == "completed":
                    completed_segments.add(result.segment_id)
                    for subtitle_id, suggested in result.updates.items():
                        if subtitle_id in branch and isinstance(branch[subtitle_id], dict):
                            branch[subtitle_id]["translated_subtitle"] = suggested
                _write_outputs(manifest_path, report_path, srt_path, branch, report)

    print(f"manifest={manifest_path}")
    print(f"report={report_path}")
    print(f"srt={srt_path}")
    return 0


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
    parser.add_argument("--llm-extra-body-json", default="")
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
    parser.add_argument("--stage2-apply-threshold", type=float, default=0.75)
    parser.add_argument("--max-glossary-entries", type=int, default=80)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=8.0)
    parser.add_argument("--retry-max-delay", type=float, default=90.0)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("MiMo audio proofread requires --api-key or MIMO_API_KEY")
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
        raw_content, usage = _call_mimo_with_retries(
            client=client,
            config=config,
            segment=task.segment,
            audio_path=task.audio_path,
            subtitle_entries=task.subtitle_entries,
            glossary_entries=task.glossary_entries,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        suggestions = _parse_suggestions(raw_content)
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

    updates: dict[str, str] = {}
    for item in suggestions:
        subtitle_id = str(item.get("id", "")).strip()
        suggested = str(item.get("suggested_translation", "")).strip()
        if subtitle_id and subtitle_id in task.subtitle_entries and suggested:
            current = str(task.subtitle_entries[subtitle_id].get("translated_subtitle", "")).strip()
            if suggested != current:
                updates[subtitle_id] = suggested

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

    segment_limit = args.segment_limit if args.segment_limit and args.segment_limit > 0 else None
    selected_segments = segments[:segment_limit]

    completed_stage1 = {
        str(item.get("segment_id"))
        for item in stage1_report
        if isinstance(item, dict) and item.get("status") == "completed"
    }
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
                task = future_to_task[future]
                result = future.result()
                stage1_report = _replace_report_item(stage1_report, result.segment_id, result.report_item)
                if result.report_item.get("status") == "completed":
                    for subtitle_id, suggested in result.updates.items():
                        if subtitle_id in branch and isinstance(branch[subtitle_id], dict):
                            branch[subtitle_id]["translated_subtitle"] = suggested
                _write_two_stage_outputs(
                    manifest_path,
                    report_path,
                    stage1_report_path,
                    stage2_report_path,
                    srt_path,
                    branch,
                    stage1_report,
                    stage2_report,
                )

    suspect_ids = _collect_stage1_suspects(stage1_report)
    completed_stage2 = {
        str(item.get("id"))
        for item in stage2_report
        if isinstance(item, dict) and item.get("status") == "completed"
    }
    pending_review_ids = [
        subtitle_id for subtitle_id in suspect_ids
        if not (args.resume and subtitle_id in completed_stage2)
    ]
    print(
        f"stage2 nearby audio review: suspect_ids={len(suspect_ids)} pending={len(pending_review_ids)}",
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
                    for update_id, suggested in result.updates.items():
                        if update_id in branch and isinstance(branch[update_id], dict):
                            branch[update_id]["translated_subtitle"] = suggested
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
        raw_content, usage = _call_mimo_text_stage1_with_retries(
            client=client,
            config=config,
            segment=task.segment,
            subtitle_entries=task.subtitle_entries,
            glossary_entries=task.glossary_entries,
            max_retries=max_retries,
            base_delay=base_delay,
            max_delay=max_delay,
        )
        suggestions = _parse_suggestions(raw_content)
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

    updates: dict[str, str] = {}
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
        suggested = str(normalized.get("suggested_translation", "")).strip()
        current = str(task.subtitle_entries[subtitle_id].get("translated_subtitle", "")).strip()
        if (
            suggested
            and suggested != current
            and not asr_suspect
            and not needs_audio_review
            and confidence >= apply_confidence_threshold
        ):
            updates[subtitle_id] = suggested

    print(
        f"[stage1 {task.index}/{task.total}] {segment_id} "
        f"suggestions={len(normalized_suggestions)} suspects={len(suspect_ids)} applied={len(updates)}",
        flush=True,
    )
    return SegmentResult(
        segment_id=segment_id,
        report_item={
            "segment_id": segment_id,
            "status": "completed",
            "subtitle_ids": list(task.subtitle_entries.keys()),
            "suggestion_count": len(normalized_suggestions),
            "applied_count": len(updates),
            "suspect_ids": sorted(suspect_ids, key=int),
            "suspect_count": len(suspect_ids),
            "glossary_count": len(task.glossary_entries),
            "usage": _usage_to_dict(usage),
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "suggestions": normalized_suggestions,
            "raw_content": raw_content if keep_raw else "",
        },
        updates=updates,
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
        audio_path = Path(str(segment.get("audio_path", "")))
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
        raw_content, usage = _call_mimo_nearby_audio_with_retries(
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
        )
        suggestions = _parse_suggestions(raw_content)
    except Exception as exc:  # pylint: disable=broad-exception-caught
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
        updates: dict[str, str] = {}
        normalized_suggestions = suggestions_by_id.get(subtitle_id, [])
        for normalized in normalized_suggestions:
            reviewed_id = str(normalized.get("id", "")).strip()
            if reviewed_id != subtitle_id or reviewed_id not in translated:
                continue
            confidence = _coerce_confidence(normalized.get("confidence"), default=1.0)
            suggested = str(normalized.get("suggested_translation", "")).strip()
            current = str(translated[reviewed_id].get("translated_subtitle", "")).strip()
            if suggested and suggested != current and confidence >= apply_confidence_threshold:
                updates[reviewed_id] = suggested
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
                    "glossary_count": len(relevant_glossary),
                    "usage": _usage_to_dict(usage),
                    "elapsed_ms": int((time.monotonic() - started) * 1000),
                    "suggestions": normalized_suggestions,
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
    existing = read_json(path, default=None)
    if isinstance(existing, dict) and existing:
        return {key: dict(value) if isinstance(value, dict) else value for key, value in existing.items()}
    return {key: dict(value) if isinstance(value, dict) else value for key, value in translated.items()}


def _load_existing_report(path: Path) -> list[dict[str, Any]]:
    existing = read_json(path, default=[])
    if isinstance(existing, list):
        return [item for item in existing if isinstance(item, dict)]
    return []


def _parse_extra_body(value: str) -> dict[str, Any] | None:
    if not value.strip():
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--llm-extra-body-json must be a JSON object")
    return parsed


def _maybe_disable_thinking_text(text: str, config: MiMoConfig) -> str:
    if not config.disable_thinking:
        return text
    return "/no_think\n" + text


def _chat_completion_create(
    *,
    client: OpenAI,
    config: MiMoConfig,
    messages: list[dict[str, Any]],
) -> Any:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.extra_body is not None:
        kwargs["extra_body"] = config.extra_body
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        if "extra_body" not in kwargs:
            raise
        kwargs.pop("extra_body", None)
        return client.chat.completions.create(**kwargs)


def _write_outputs(
    manifest_path: Path,
    report_path: Path,
    srt_path: Path,
    branch: dict[str, Any],
    report: list[dict[str, Any]],
) -> None:
    write_json_atomic(manifest_path, branch)
    write_json_atomic(report_path, report)
    srt_path.write_text(_to_srt(branch), encoding="utf-8")


def _write_two_stage_outputs(
    manifest_path: Path,
    report_path: Path,
    stage1_report_path: Path,
    stage2_report_path: Path,
    srt_path: Path,
    branch: dict[str, Any],
    stage1_report: list[dict[str, Any]],
    stage2_report: list[dict[str, Any]],
) -> None:
    write_json_atomic(manifest_path, branch)
    write_json_atomic(stage1_report_path, stage1_report)
    write_json_atomic(stage2_report_path, stage2_report)
    write_json_atomic(
        report_path,
        {
            "mode": "two-stage-nearby",
            "stage1_report": str(stage1_report_path),
            "stage2_report": str(stage2_report_path),
            "stage1_completed": sum(1 for item in stage1_report if item.get("status") == "completed"),
            "stage1_failed": sum(1 for item in stage1_report if item.get("status") == "failed"),
            "stage1_suspect_count": len(_collect_stage1_suspects(stage1_report)),
            "stage2_completed": sum(1 for item in stage2_report if item.get("status") == "completed"),
            "stage2_failed": sum(1 for item in stage2_report if item.get("status") == "failed"),
        },
    )
    srt_path.write_text(_to_srt(branch), encoding="utf-8")


def _replace_report_item(
    report: list[dict[str, Any]],
    item_id: str,
    item: dict[str, Any],
    *,
    key: str = "segment_id",
) -> list[dict[str, Any]]:
    return [
        existing for existing in report
        if str(existing.get(key, "")) != item_id
    ] + [item]


def _collect_stage1_suspects(stage1_report: list[dict[str, Any]]) -> list[str]:
    suspect_ids: set[str] = set()
    for item in stage1_report:
        if not isinstance(item, dict) or item.get("status") != "completed":
            continue
        for subtitle_id in item.get("suspect_ids", []):
            subtitle_id_text = str(subtitle_id).strip()
            if subtitle_id_text:
                suspect_ids.add(subtitle_id_text)
    return sorted(suspect_ids, key=int)


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/v1") else "/v1")


def _load_glossary(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    return [
        {
            "group": entry.group,
            "source": entry.source,
            "target": entry.target,
            "note": entry.note,
        }
        for entry in read_xlsx_glossary(path)
    ]


def _subtitle_entries_for_segment(
    translated: dict[str, Any],
    segment: dict[str, Any],
) -> dict[str, Any]:
    start_ms = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
    end_ms = int(round(float(segment.get("global_end_time", 0.0)) * 1000))
    return {
        str(key): value
        for key, value in translated.items()
        if isinstance(value, dict)
        and int(value.get("end_time", 0)) > start_ms
        and int(value.get("start_time", 0)) < end_ms
    }


def _filter_glossary(
    glossary: list[dict[str, str]],
    subtitle_entries: dict[str, Any],
    limit: int,
) -> list[dict[str, str]]:
    if not glossary or limit <= 0:
        return []
    haystack = "\n".join(
        f"{item.get('original_subtitle', '')}\n{item.get('translated_subtitle', '')}"
        for item in subtitle_entries.values()
        if isinstance(item, dict)
    )
    exact_matches = [
        entry for entry in glossary
        if entry["source"] and entry["source"] in haystack
    ]
    general = [
        entry for entry in glossary
        if entry not in exact_matches and entry.get("group") in {"names", "show_terms", "通用日语"}
    ]
    return (exact_matches + general)[:limit]


def _segment_for_subtitle_id(
    subtitle_id: str,
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
) -> dict[str, Any]:
    item = translated.get(subtitle_id)
    if not isinstance(item, dict):
        raise KeyError(f"subtitle id not found: {subtitle_id}")
    start_ms = int(item.get("start_time", 0))
    for segment in segments:
        segment_start = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
        segment_end = int(round(float(segment.get("global_end_time", 0.0)) * 1000))
        if segment_start <= start_ms < segment_end:
            return segment
    raise RuntimeError(f"No audio segment covers subtitle id {subtitle_id}")


def _build_nearby_audio_batches(
    *,
    subtitle_ids: list[str],
    segments: list[dict[str, Any]],
    translated: dict[str, Any],
    context_subtitles: int,
    batch_size: int,
    max_gap_s: float,
) -> list[list[str]]:
    candidates: list[tuple[str, str, int, int]] = []
    for subtitle_id in subtitle_ids:
        item = translated.get(subtitle_id)
        if not isinstance(item, dict):
            continue
        try:
            segment = _segment_for_subtitle_id(subtitle_id, segments, translated)
        except RuntimeError:
            continue
        entries = _nearby_entries_for_subtitle_id(subtitle_id, translated, context_subtitles)
        start_ms = min(int(entry.get("start_time", 0)) for entry in entries.values())
        end_ms = max(int(entry.get("end_time", 0)) for entry in entries.values())
        candidates.append((subtitle_id, str(segment.get("segment_id", "")), start_ms, end_ms))

    candidates.sort(key=lambda row: (row[1], row[2], int(row[0])))
    batches: list[list[str]] = []
    current: list[str] = []
    current_segment_id = ""
    current_end_ms = 0
    max_gap_ms = int(round(max_gap_s * 1000))

    for subtitle_id, segment_id, start_ms, end_ms in candidates:
        can_join = (
            current
            and segment_id == current_segment_id
            and len(current) < batch_size
            and start_ms - current_end_ms <= max_gap_ms
        )
        if not can_join:
            if current:
                batches.append(current)
            current = [subtitle_id]
            current_segment_id = segment_id
            current_end_ms = end_ms
            continue
        current.append(subtitle_id)
        current_end_ms = max(current_end_ms, end_ms)

    if current:
        batches.append(current)
    return batches


def _nearby_entries_for_subtitle_id(
    subtitle_id: str,
    translated: dict[str, Any],
    context_subtitles: int,
) -> dict[str, Any]:
    if not subtitle_id.isdigit():
        raise ValueError(f"subtitle id must be numeric: {subtitle_id}")
    center = int(subtitle_id)
    start = max(1, center - context_subtitles)
    end = center + context_subtitles
    entries: dict[str, Any] = {}
    for index in range(start, end + 1):
        key = str(index)
        item = translated.get(key)
        if isinstance(item, dict):
            entries[key] = item
    return entries


def _nearby_entries_for_subtitle_ids(
    subtitle_ids: list[str],
    translated: dict[str, Any],
    context_subtitles: int,
) -> dict[str, Any]:
    numeric_ids = [int(subtitle_id) for subtitle_id in subtitle_ids if subtitle_id.isdigit()]
    if not numeric_ids:
        return {}
    start = max(1, min(numeric_ids) - context_subtitles)
    end = max(numeric_ids) + context_subtitles
    entries: dict[str, Any] = {}
    for index in range(start, end + 1):
        key = str(index)
        item = translated.get(key)
        if isinstance(item, dict):
            entries[key] = item
    return entries


def _write_nearby_audio_clip(
    *,
    subtitle_id: str,
    segment: dict[str, Any],
    entries: dict[str, Any],
    audio_path: Path,
    clips_dir: Path,
    padding_s: float,
) -> tuple[Path, dict[str, float]]:
    if not entries:
        raise ValueError(f"No nearby entries for subtitle id {subtitle_id}")
    segment_global_start_ms = int(round(float(segment.get("global_start_time", 0.0)) * 1000))
    min_start_ms = min(int(item.get("start_time", 0)) for item in entries.values())
    max_end_ms = max(int(item.get("end_time", 0)) for item in entries.values())
    local_start_s = max(0.0, (min_start_ms - segment_global_start_ms) / 1000.0 - padding_s)
    local_end_s = max(local_start_s + 0.2, (max_end_ms - segment_global_start_ms) / 1000.0 + padding_s)

    clip_path = clips_dir / f"subtitle_{subtitle_id}_nearby.wav"
    with wave.open(str(audio_path), "rb") as source:
        frame_rate = source.getframerate()
        total_frames = source.getnframes()
        start_frame = max(0, min(total_frames, int(round(local_start_s * frame_rate))))
        end_frame = max(start_frame + 1, min(total_frames, int(round(local_end_s * frame_rate))))
        source.setpos(start_frame)
        frames = source.readframes(end_frame - start_frame)
        params = source.getparams()

    with wave.open(str(clip_path), "wb") as target:
        target.setparams(params)
        target.writeframes(frames)

    duration_s = (end_frame - start_frame) / max(frame_rate, 1)
    return clip_path, {
        "start_s": start_frame / max(frame_rate, 1),
        "end_s": end_frame / max(frame_rate, 1),
        "duration_s": duration_s,
    }


def _normalize_qa_item(item: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "id": item.get("id", item.get("i", "")),
        "error_type": item.get("error_type", item.get("t", "")),
        "original": item.get("original", item.get("o", "")),
        "translation": item.get("translation", item.get("tr", "")),
        "suggested_translation": item.get("suggested_translation", item.get("s", "")),
        "asr_suspect": item.get("asr_suspect", item.get("a", False)),
        "suspected_original": item.get("suspected_original", item.get("so", "")),
        "needs_audio_review": item.get("needs_audio_review", item.get("n", False)),
        "reason": item.get("reason", item.get("r", "")),
        "confidence": item.get("confidence", item.get("c", 0.0)),
    }
    normalized["id"] = str(normalized.get("id", "")).strip()
    error_type = str(normalized.get("error_type", "")).strip()
    allowed = {"translation_error", "term_error", "asr_suspect", "needs_context", "style_only"}
    normalized["error_type"] = error_type if error_type in allowed else ""
    normalized["asr_suspect"] = _coerce_bool(normalized.get("asr_suspect"))
    normalized["needs_audio_review"] = _coerce_bool(normalized.get("needs_audio_review"))
    normalized["confidence"] = _coerce_confidence(normalized.get("confidence"), default=0.0)
    for key in ("original", "translation", "suggested_translation", "suspected_original", "reason"):
        normalized[key] = str(normalized.get(key, "")).strip()
    return normalized


def _compact_schema_prompt(config: MiMoConfig) -> str:
    if not config.compact_output:
        return (
            "Schema: {id, error_type, original, translation, suggested_translation, "
            "asr_suspect, suspected_original, needs_audio_review, reason, confidence}.\n"
            "Field rules: error_type must be one of translation_error, term_error, asr_suspect, needs_context, style_only; "
            "asr_suspect and needs_audio_review must be booleans; confidence must be 0.0 to 1.0.\n"
        )
    return (
        "Use compact JSON keys to save tokens: "
        "{i:id,t:error_type,o:original,tr:translation,s:suggested_translation,a:asr_suspect,"
        "so:suspected_original,n:needs_audio_review,r:reason,c:confidence}.\n"
        "Return booleans for a and n. Use t values: translation_error, term_error, asr_suspect, needs_context, style_only. "
        "Keep r short, max 18 Chinese characters or 12 English words.\n"
    )


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "yes", "1"}


def _coerce_confidence(value: Any, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number))


def _call_mimo(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    audio_path: Path,
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
) -> tuple[str, Any]:
    audio_data = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    prompt = (
        "You are proofreading Chinese subtitles for one Japanese audio segment.\n"
        "Use the audio as the source of truth, then compare Japanese original text, "
        "Chinese translation, and glossary.\n"
        "Return ONLY a JSON array. Include only subtitle IDs that need correction.\n"
        "If no correction is needed, return []. Keep IDs and timestamps unchanged.\n"
        "Object schema: {id, original, translation, suggested_translation, reason, confidence}. Keep reason short.\n"
        "Do not add markdown or prose outside JSON.\n"
        f"Segment JSON: {json.dumps(segment, ensure_ascii=True)}\n"
        f"Glossary JSON: {json.dumps(glossary_entries, ensure_ascii=True)}\n"
        f"Subtitle entries JSON: {json.dumps(subtitle_entries, ensure_ascii=True)}"
    )
    response = _chat_completion_create(
        client=client,
        config=config,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _maybe_disable_thinking_text(prompt, config)},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": audio_path.suffix.lstrip(".") or "wav",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or "", response.usage


def _call_mimo_text_stage1(
    *,
    client: OpenAI,
    config: MiMoConfig,
    segment: dict[str, Any],
    subtitle_entries: dict[str, Any],
    glossary_entries: list[dict[str, str]],
) -> tuple[str, Any]:
    prompt = (
        "You are stage 1 of a two-stage subtitle QA pipeline.\n"
        "You do NOT have audio.\n"
        "Inputs: Japanese ASR subtitle text, Chinese translation, subtitle timing and neighboring context, and glossary.\n"
        "Task 1: correct Chinese translation errors that are strongly supported by the provided text and context.\n"
        "Task 2: flag Japanese ASR text as suspicious only when there is concrete textual evidence: "
        "the Japanese is grammatically broken or semantically impossible; the Chinese translation cannot reasonably "
        "follow from the Japanese text; a proper noun, number, date, venue, player name, work title, or technical term "
        "looks likely misrecognized; or neighboring subtitles strongly imply a different phrase.\n"
        "Do not guess a corrected Japanese phrase unless context strongly supports it.\n"
        "If audio is required to decide, set needs_audio_review to true and leave suggested_translation empty unless "
        "the Chinese fix is already safe.\n"
        "Return ONLY a valid JSON array. Include an object if either translation correction is needed OR audio review is needed.\n"
        "Do not include markdown or prose.\n"
        f"{_compact_schema_prompt(config)}"
        f"Segment JSON: {json.dumps(segment, ensure_ascii=True)}\n"
        f"Glossary JSON: {json.dumps(glossary_entries, ensure_ascii=True)}\n"
        f"Subtitle entries JSON: {json.dumps(subtitle_entries, ensure_ascii=True)}"
    )
    response = _chat_completion_create(
        client=client,
        config=config,
        messages=[{"role": "user", "content": _maybe_disable_thinking_text(prompt, config)}],
    )
    return response.choices[0].message.content or "", response.usage


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
    audio_data = base64.b64encode(clip_path.read_bytes()).decode("ascii")
    prompt = (
        "You are stage 2 of a two-stage subtitle QA pipeline.\n"
        "You HAVE a short nearby audio clip. The target subtitle IDs were flagged as suspicious by text-only QA.\n"
        "Use the audio as the source of truth.\n"
        "Tasks, in order:\n"
        "1. Listen specifically for each target ID's Japanese wording. Multiple target IDs may be present in one clip.\n"
        "2. Decide independently for each target ID whether original_subtitle is correct, incomplete, or a mishearing.\n"
        "3. Pay special attention to proper nouns: player names, place names, venue or stadium names, work titles, "
        "band or member names, dates, and numbers.\n"
        "4. Use nearby subtitle text only as context. Do not rewrite non-target IDs.\n"
        "5. If the target phrase is unclear in this short clip, set needs_audio_review to true and explain that a wider clip is needed.\n"
        "Return ONLY a valid JSON array. Include only target IDs that need correction or remain unresolved. "
        "Use one JSON object per subtitle ID.\n"
        "Do not include markdown or prose.\n"
        f"{_compact_schema_prompt(config)}"
        f"Target IDs JSON: {json.dumps(target_ids, ensure_ascii=True)}\n"
        f"Clip local start/end seconds JSON: {json.dumps(clip_meta, ensure_ascii=True)}\n"
        f"Segment JSON: {json.dumps(segment, ensure_ascii=True)}\n"
        f"Glossary JSON: {json.dumps(glossary_entries, ensure_ascii=True)}\n"
        f"Target entries JSON: {json.dumps(target_entries, ensure_ascii=True)}\n"
        f"Nearby entries JSON: {json.dumps(nearby_entries, ensure_ascii=True)}"
    )
    response = _chat_completion_create(
        client=client,
        config=config,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _maybe_disable_thinking_text(prompt, config)},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data,
                            "format": clip_path.suffix.lstrip(".") or "wav",
                        },
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or "", response.usage


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
    attempt = 0
    delay = max(0.0, base_delay)
    while True:
        attempt += 1
        try:
            return _call_mimo(
                client=client,
                config=config,
                segment=segment,
                audio_path=audio_path,
                subtitle_entries=subtitle_entries,
                glossary_entries=glossary_entries,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt > max(1, max_retries) or not _is_transient_error(exc):
                raise
            wait_seconds = min(max_delay, delay or 1.0)
            print(
                f"  transient MiMo error on attempt {attempt}/{max_retries}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(max_delay, max(wait_seconds * 2.0, 1.0))


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
    attempt = 0
    delay = max(0.0, base_delay)
    while True:
        attempt += 1
        try:
            return _call_mimo_text_stage1(
                client=client,
                config=config,
                segment=segment,
                subtitle_entries=subtitle_entries,
                glossary_entries=glossary_entries,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt > max(1, max_retries) or not _is_transient_error(exc):
                raise
            wait_seconds = min(max_delay, delay or 1.0)
            print(
                f"  transient MiMo stage1 error on attempt {attempt}/{max_retries}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(max_delay, max(wait_seconds * 2.0, 1.0))


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
    attempt = 0
    delay = max(0.0, base_delay)
    while True:
        attempt += 1
        try:
            return _call_mimo_nearby_audio(
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
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt > max(1, max_retries) or not _is_transient_error(exc):
                raise
            wait_seconds = min(max_delay, delay or 1.0)
            print(
                f"  transient MiMo stage2 error on attempt {attempt}/{max_retries}: "
                f"{exc}; retrying in {wait_seconds:.1f}s",
                flush=True,
            )
            time.sleep(wait_seconds)
            delay = min(max_delay, max(wait_seconds * 2.0, 1.0))


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc).lower()
    transient_markers = (
        "503",
        "502",
        "504",
        "429",
        "rate limit",
        "timeout",
        "timed out",
        "upstream",
        "服务异常",
        "稍后重试",
        "server_error",
    )
    return any(marker in text for marker in transient_markers)


def _parse_suggestions(content: str) -> list[dict[str, Any]]:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        object_matches = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
        if object_matches:
            parsed = [json.loads(match) for match in object_matches]
        else:
            raise
    if not isinstance(parsed, list):
        if isinstance(parsed, dict):
            parsed = [parsed]
        else:
            raise ValueError("MiMo response is not a JSON array")
    return [item for item in parsed if isinstance(item, dict)]


def _usage_to_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "dict"):
        return usage.dict()
    return {"repr": repr(usage)}


def _to_srt(items: dict[str, Any]) -> str:
    lines: list[str] = []
    index = 1
    for _, item in sorted(items.items(), key=lambda pair: int(pair[0]) if str(pair[0]).isdigit() else str(pair[0])):
        if not isinstance(item, dict):
            continue
        start = int(item.get("start_time", 0))
        end = int(item.get("end_time", start + 1))
        original = str(item.get("original_subtitle", "")).strip()
        translated = str(item.get("translated_subtitle", "")).strip()
        text = original if not translated else f"{original}\n{translated}"
        lines.extend([str(index), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""])
        index += 1
    return "\n".join(lines).strip() + "\n"


def _srt_time(ms: int) -> str:
    hours, rem = divmod(max(0, ms), 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


if __name__ == "__main__":
    raise SystemExit(main())
