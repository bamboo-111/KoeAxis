from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from qwen_asr.content_quality import normalize_japanese
from qwen_asr.final_quality_common import fail, passed, skip, warn
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def normalize_export_content_check(work_paths: WorkPaths) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    normalized = normalized_stage_text(work_paths.normalized_manifest)
    if normalized is not None:
        source_name, source_text = best_pre_normalize_stage_text(work_paths)
        if source_text is None:
            checks.append(warn("normalize_content", "存在 normalized 产物，但找不到可比较的上游字幕源"))
        elif source_text != normalized:
            checks.append(
                fail(
                    "normalize_content",
                    "normalize 改变了规范化日文内容",
                    source_stage=source_name,
                    source_chars=len(source_text),
                    normalized_chars=len(normalized),
                )
            )
        else:
            checks.append(
                passed(
                    "normalize_content",
                    "normalize 未改变规范化日文内容",
                    source_stage=source_name,
                    chars=len(normalized),
                )
            )
    else:
        checks.append(skip("normalize_content", "未生成 normalized 产物，跳过 normalize 内容检查"))

    export_text = srt_stage_text(work_paths.subtitles_srt) if work_paths.subtitles_srt.exists() else None
    if export_text is not None:
        if normalized is not None:
            source_name = "normalized"
            source_text = normalized
        else:
            source_name, source_text = best_pre_normalize_stage_text(work_paths)
        if source_text is None:
            checks.append(warn("export_content", "存在 export SRT，但找不到可比较的上游字幕源"))
        elif source_text != export_text:
            checks.append(
                fail(
                    "export_content",
                    "export 改变了规范化日文内容",
                    source_stage=source_name,
                    source_chars=len(source_text),
                    export_chars=len(export_text),
                )
            )
        else:
            checks.append(
                passed(
                    "export_content",
                    "export 未改变规范化日文内容",
                    source_stage=source_name,
                    chars=len(export_text),
                )
            )
    else:
        checks.append(skip("export_content", "未生成 export SRT，跳过 export 内容检查"))

    failed = [item for item in checks if item["status"] == "FAIL"]
    warned = [item for item in checks if item["status"] == "WARN"]
    status = "FAIL" if failed else "WARN" if warned else "PASS"
    return {
        "name": "normalize_export_content",
        "status": status,
        "message": (f"normalize/export 内容检查 {status}：{len(failed)} 个失败，{len(warned)} 个警告"),
        "checks": checks,
    }


def best_pre_normalize_stage_text(work_paths: WorkPaths) -> tuple[str, str | None]:
    for name, path in (
        ("proofread-realigned", work_paths.mimo_proofread_manifest),
        ("translated", work_paths.translated_manifest),
        ("split", work_paths.split_manifest),
        ("transcript", work_paths.transcript_manifest),
    ):
        text = normalized_stage_text(path)
        if text is not None:
            return name, text
    return "", None


def normalized_stage_text(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = read_json(path, default={})
    if isinstance(payload, list):
        parts = [manifest_item_text("transcript", item) for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        parts = [manifest_item_text("subtitle", item) for item in payload.values() if isinstance(item, dict)]
    else:
        return None
    return "".join(normalize_japanese(part) for part in parts)


def manifest_item_text(kind: str, item: dict[str, Any]) -> str:
    if kind == "transcript":
        return str(item.get("text", item.get("original_subtitle", "")))
    return str(item.get("original_subtitle", item.get("text", "")))


def srt_stage_text(path: Path) -> str | None:
    if not path.exists():
        return None
    blocks = [
        block.strip() for block in re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig")) if block.strip()
    ]
    parts: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) >= 3 and "-->" in lines[1]:
            parts.append(lines[2])
    return "".join(normalize_japanese(part) for part in parts)
