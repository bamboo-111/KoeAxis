from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from qwen_asr.ass_quality import build_ass_quality_report
from qwen_asr.content_quality import SHORT_RESPONSES, normalize_japanese
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


STAGE_FILES: tuple[tuple[str, str], ...] = (
    ("transcript", "transcript_manifest"),
    ("correct", "corrected_manifest"),
    ("aligned", "aligned_manifest"),
    ("split", "split_manifest"),
    ("translated", "translated_manifest"),
    ("proofread", "mimo_proofread_manifest"),
    ("proofread-realigned", "mimo_proofread_manifest"),
    ("normalized", "normalized_manifest"),
    ("export", "subtitles_srt"),
)


def cmd_baseline_snapshot(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report = build_baseline_snapshot(
        work_paths,
        ass_path=Path(args.ass) if getattr(args, "ass", "") else None,
        ass_offset_ms=getattr(args, "ass_offset_ms", None),
        ass_source=str(getattr(args, "ass_source", "split")),
    )
    output = Path(getattr(args, "output", "") or work_paths.workdir / "reports" / "baseline_snapshot.json")
    markdown_output = getattr(args, "markdown_output", None)
    if markdown_output is None:
        markdown_output = output.with_suffix(".md")
    write_json_atomic(output, report)
    if markdown_output:
        Path(markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_output).write_text(render_baseline_snapshot_markdown(report), encoding="utf-8")
    print(f"基线快照已写入：{output}")
    return 0


def build_baseline_snapshot(
    work_paths: WorkPaths,
    *,
    ass_path: Path | None = None,
    ass_offset_ms: int | None = None,
    ass_source: str = "split",
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = []
    for stage, attr in STAGE_FILES:
        path = getattr(work_paths, attr)
        stages.append(_stage_snapshot(stage, path))

    content_quality = read_json(work_paths.content_quality_report, default={})
    final_quality = read_json(work_paths.final_quality_report, default={})
    ass_quality: dict[str, Any] | None = None
    if ass_path is not None:
        ass_quality = build_ass_quality_report(
            work_paths,
            ass_path=ass_path,
            source=ass_source,
            offset_ms=ass_offset_ms,
        )

    return {
        "workdir": str(work_paths.workdir),
        "stages": stages,
        "content_quality": _quality_summary(content_quality),
        "final_quality": _quality_summary(final_quality),
        "ass_quality": _ass_summary(ass_quality) if ass_quality is not None else None,
    }


def _stage_snapshot(stage: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    metrics = _load_stage_metrics(stage, path) if exists else _empty_metrics()
    return {
        "stage": stage,
        "path": str(path),
        "exists": exists,
        "sha256": _sha256(path) if exists else "",
        "size_bytes": path.stat().st_size if exists else 0,
        **metrics,
    }


def _load_stage_metrics(stage: str, path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".srt":
        items = _load_srt_items(path)
    else:
        payload = read_json(path, default=[] if stage in {"transcript", "correct", "aligned"} else {})
        items = _items_from_payload(stage, payload)
    normalized_texts = [normalize_japanese(str(item.get("text", ""))) for item in items]
    normalized_joined = "".join(normalized_texts)
    text_counter: dict[str, int] = {}
    for text in normalized_texts:
        if text:
            text_counter[text] = text_counter.get(text, 0) + 1
    return {
        "item_count": len(items),
        "normalized_japanese_chars": len(normalized_joined),
        "short_response_count": sum(1 for text in normalized_texts if _is_short_response(text)),
        "unique_text_count": sum(1 for count in text_counter.values() if count == 1),
        "time": _time_summary(items),
        "adjacent_duplicate_count": _adjacent_duplicate_count(normalized_texts),
    }


def _items_from_payload(stage: str, payload: Any) -> list[dict[str, Any]]:
    raw_items = payload.values() if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    items: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = _item_text(stage, item)
        start_ms, end_ms = _item_times_ms(item)
        items.append({"text": text, "start_ms": start_ms, "end_ms": end_ms})
    return items


def _item_text(stage: str, item: dict[str, Any]) -> str:
    if stage == "aligned":
        tokens = item.get("tokens", [])
        if isinstance(tokens, list) and tokens:
            return "".join(str(token.get("text", "")) for token in tokens if isinstance(token, dict))
        return str(item.get("text", ""))
    if stage in {"split", "translated", "proofread", "proofread-realigned", "normalized"}:
        return str(item.get("original_subtitle", item.get("text", "")))
    return str(item.get("text", item.get("original_subtitle", "")))


def _item_times_ms(item: dict[str, Any]) -> tuple[int | None, int | None]:
    if "start_time" in item or "end_time" in item:
        return _as_int_or_none(item.get("start_time")), _as_int_or_none(item.get("end_time"))
    start = _as_float_or_none(item.get("global_start_time"))
    end = _as_float_or_none(item.get("global_end_time"))
    return (
        int(round(start * 1000)) if start is not None else None,
        int(round(end * 1000)) if end is not None else None,
    )


def _load_srt_items(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    items: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[1].split("-->", 1)]
        items.append(
            {
                "text": lines[2],
                "start_ms": _parse_srt_time(start_raw),
                "end_ms": _parse_srt_time(end_raw),
            }
        )
    return items


def _parse_srt_time(value: str) -> int | None:
    try:
        clock, ms = value.split(",", 1)
        hours, minutes, seconds = [int(part) for part in clock.split(":")]
        return ((hours * 60 + minutes) * 60 + seconds) * 1000 + int(ms)
    except (TypeError, ValueError):
        return None


def _time_summary(items: list[dict[str, Any]]) -> dict[str, int | bool]:
    invalid = 0
    non_monotonic = 0
    previous_start: int | None = None
    for item in items:
        start = item.get("start_ms")
        end = item.get("end_ms")
        if not isinstance(start, int) or not isinstance(end, int) or end <= start or start < 0:
            invalid += 1
        if isinstance(start, int) and previous_start is not None and start < previous_start:
            non_monotonic += 1
        if isinstance(start, int):
            previous_start = start
    return {
        "valid": invalid == 0 and non_monotonic == 0,
        "invalid_count": invalid,
        "non_monotonic_count": non_monotonic,
    }


def _quality_summary(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict) or not report:
        return {"exists": False, "status": "", "summary": {}}
    return {
        "exists": True,
        "status": str(report.get("status", "")),
        "summary": report.get("summary", {}) if isinstance(report.get("summary"), dict) else {},
    }


def _ass_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    return {
        "status": report.get("status", ""),
        "source": report.get("source", ""),
        "offset_ms": report.get("offset_ms"),
        "selected_dialogue_count": report.get("selected_dialogue_count", 0),
        "source_cue_count": report.get("source_cue_count", 0),
        "summary": summary,
    }


def _empty_metrics() -> dict[str, Any]:
    return {
        "item_count": 0,
        "normalized_japanese_chars": 0,
        "short_response_count": 0,
        "unique_text_count": 0,
        "time": {"valid": False, "invalid_count": 0, "non_monotonic_count": 0},
        "adjacent_duplicate_count": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_short_response(text: str) -> bool:
    return bool(text) and len(text) <= 4 and text in {normalize_japanese(value) for value in SHORT_RESPONSES}


def _adjacent_duplicate_count(texts: list[str]) -> int:
    return sum(1 for previous, current in zip(texts, texts[1:], strict=False) if previous and previous == current)


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_baseline_snapshot_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 稳定基线快照",
        "",
        f"- 工作区：`{report.get('workdir', '')}`",
        "",
        "## 阶段文件",
        "",
        "| 阶段 | 存在 | 条目 | 规范化日文字符 | 短应答 | 独有文本 | 时间合法 | SHA256 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("stages", []):
        if not isinstance(item, dict):
            continue
        time = item.get("time", {}) if isinstance(item.get("time"), dict) else {}
        lines.append(
            "| "
            f"{item.get('stage', '')} | "
            f"{'是' if item.get('exists') else '否'} | "
            f"{item.get('item_count', 0)} | "
            f"{item.get('normalized_japanese_chars', 0)} | "
            f"{item.get('short_response_count', 0)} | "
            f"{item.get('unique_text_count', 0)} | "
            f"{'是' if time.get('valid') else '否'} | "
            f"`{str(item.get('sha256', ''))[:12]}` |"
        )
    lines.extend(["", "## 质量摘要", ""])
    for label, key in (("内容守恒", "content_quality"), ("最终质量门", "final_quality"), ("ASS 质量", "ass_quality")):
        value = report.get(key)
        if not isinstance(value, dict):
            lines.append(f"- {label}：未生成")
            continue
        lines.append(f"- {label}：{value.get('status', '') or '未生成'}")
    lines.append("")
    return "\n".join(lines)
