from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qwen_asr.content_quality import normalize_japanese
from qwen_asr.final_quality_common import fail, int_or_none, skip
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


MIN_ORDINARY_SUBTITLE_DURATION_MS = 500
MIN_PROTECTED_SHORT_SUBTITLE_DURATION_MS = 120
PROTECTED_SHORT_SUBTITLE_NORMALIZED = {
    "\u306f\u3044",
    "\u3048",
    "\u3046\u3093",
    "\u3046\u3046\u3093",
    "\u3044\u3044\u3048",
    "\u99c4\u76ee",
    "\u3060\u3081",
    "\u30c0\u30e1",
    "\u9055\u3046",
    "\u75db\u3044",
    "\u3042",
    "\u3042\u3042",
    "\u3044\u3084",
    "\u304a",
    "\u304a\u304a",
    "\u306f\u3042",
}


def subtitle_readability_check(
    work_paths: WorkPaths,
    *,
    manifest_key_sort: Callable[[str], tuple[int, int | str]],
) -> dict[str, Any]:
    stages = [
        ("split", work_paths.split_manifest),
        ("translated", work_paths.translated_manifest),
        ("mimo-proofread", work_paths.mimo_proofread_manifest),
        ("normalized", work_paths.normalized_manifest),
    ]
    issues: list[dict[str, Any]] = []
    checked_stage_count = 0
    checked_item_count = 0
    for stage, path in stages:
        if not path.exists():
            continue
        payload = read_json(path, default={})
        if not isinstance(payload, dict) or not payload:
            issues.append(subtitle_readability_issue("FAIL", stage, "", "invalid_manifest", "字幕 manifest 缺失或为空"))
            continue
        checked_stage_count += 1
        previous_end: int | None = None
        for key in sorted((str(item_key) for item_key in payload.keys()), key=manifest_key_sort):
            item = payload.get(key)
            if not isinstance(item, dict):
                issues.append(subtitle_readability_issue("FAIL", stage, key, "invalid_item", "字幕条目不是字典"))
                continue
            checked_item_count += 1
            start_ms = int_or_none(item.get("start_time"))
            end_ms = int_or_none(item.get("end_time"))
            text = subtitle_display_text(item)
            if not text:
                issues.append(subtitle_readability_issue("FAIL", stage, key, "empty_text", "字幕文本为空"))
            if start_ms is None or end_ms is None:
                issues.append(subtitle_readability_issue("FAIL", stage, key, "missing_time", "字幕时间缺失或非法"))
                continue
            if start_ms < 0 or end_ms < 0:
                issues.append(subtitle_readability_issue("FAIL", stage, key, "negative_time", "字幕时间为负数"))
            duration_ms = end_ms - start_ms
            if duration_ms <= 0:
                issues.append(
                    subtitle_readability_issue(
                        "FAIL", stage, key, "non_positive_duration", "字幕结束时间不晚于开始时间"
                    )
                )
            elif duration_ms < MIN_PROTECTED_SHORT_SUBTITLE_DURATION_MS:
                issues.append(
                    subtitle_readability_issue(
                        "WARN",
                        stage,
                        key,
                        "protected_short_too_fast"
                        if is_protected_short_subtitle(text)
                        else "ordinary_subtitle_too_fast",
                        f"字幕时长低于可读阈值：{duration_ms}ms",
                    )
                )
            elif duration_ms < MIN_ORDINARY_SUBTITLE_DURATION_MS and not is_protected_short_subtitle(text):
                issues.append(
                    subtitle_readability_issue(
                        "WARN",
                        stage,
                        key,
                        "ordinary_subtitle_too_fast",
                        f"普通字幕低于 500ms：{duration_ms}ms",
                    )
                )
            elif duration_ms > 8000:
                issues.append(
                    subtitle_readability_issue(
                        "WARN",
                        stage,
                        key,
                        "very_long_duration",
                        f"字幕时长过长：{duration_ms}ms",
                    )
                )
            if len(text) > 80:
                issues.append(
                    subtitle_readability_issue(
                        "WARN",
                        stage,
                        key,
                        "long_text",
                        f"单条字幕文本过长：{len(text)} 字符",
                    )
                )
            if previous_end is not None and start_ms < previous_end:
                overlap = previous_end - start_ms
                severity = "FAIL" if overlap > 500 else "WARN"
                issues.append(
                    subtitle_readability_issue(
                        severity,
                        stage,
                        key,
                        "overlap",
                        f"与上一条重叠 {overlap}ms",
                    )
                )
            previous_end = end_ms

    if checked_stage_count == 0:
        return skip("subtitle_readability", "未生成字幕阶段 manifest，跳过断句/时间可读性检查")

    fail_count = sum(item["severity"] == "FAIL" for item in issues)
    warn_count = sum(item["severity"] == "WARN" for item in issues)
    if fail_count:
        return fail(
            "subtitle_readability",
            f"断句/时间可读性 FAIL：{fail_count} 个失败，{warn_count} 个警告",
            checked_stage_count=checked_stage_count,
            checked_item_count=checked_item_count,
            issues=issues[:50],
        )
    status = "WARN" if warn_count else "PASS"
    return {
        "name": "subtitle_readability",
        "status": status,
        "message": f"断句/时间可读性 {status}：检查 {checked_stage_count} 个阶段、{checked_item_count} 条字幕，{warn_count} 个警告",
        "checked_stage_count": checked_stage_count,
        "checked_item_count": checked_item_count,
        "issues": issues[:50],
    }


def is_protected_short_subtitle(text: str) -> bool:
    normalized = normalize_japanese(text)
    return normalized in PROTECTED_SHORT_SUBTITLE_NORMALIZED


def subtitle_display_text(item: dict[str, Any]) -> str:
    for key in ("original_subtitle", "text", "translated_subtitle"):
        text = str(item.get(key, "")).strip()
        if text:
            return text
    return ""


def subtitle_readability_issue(
    severity: str,
    stage: str,
    key: str,
    kind: str,
    message: str,
) -> dict[str, Any]:
    return {"severity": severity, "stage": stage, "key": key, "type": kind, "message": message}
