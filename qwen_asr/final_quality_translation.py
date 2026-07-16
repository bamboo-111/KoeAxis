from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qwen_asr.final_quality_common import fail, passed, skip
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def translation_structure_check(work_paths: WorkPaths) -> dict[str, Any]:
    if not work_paths.translated_manifest.exists():
        return skip("translation_structure", "未运行翻译阶段")
    payload = read_json(work_paths.translated_manifest, default={})
    if not isinstance(payload, dict) or not payload:
        return fail("translation_structure", "翻译产物缺失或不是字幕字典")

    missing_translation = 0
    structured_count = 0
    suspect_count = 0
    invalid_suspect_fields = 0
    for item in payload.values():
        if not isinstance(item, dict):
            missing_translation += 1
            continue
        if not str(item.get("translated_subtitle", "")).strip():
            missing_translation += 1
        structured_keys = {"asr_suspect", "needs_audio_review", "suspect_types", "confidence"}
        if any(key in item for key in structured_keys):
            structured_count += 1
        if bool(item.get("needs_audio_review")) or bool(item.get("asr_suspect")):
            suspect_count += 1
        if "suspect_types" in item and not isinstance(item.get("suspect_types"), list):
            invalid_suspect_fields += 1

    if missing_translation or invalid_suspect_fields:
        return fail(
            "translation_structure",
            f"翻译结构异常：缺译 {missing_translation} 条，疑点字段异常 {invalid_suspect_fields} 条",
            missing_translation=missing_translation,
            invalid_suspect_fields=invalid_suspect_fields,
        )
    if structured_count == 0:
        return fail("translation_structure", "translation artifact has no structured suspect fields")
    return {
        "name": "translation_structure",
        "status": "PASS",
        "message": f"翻译结构通过：结构化 {structured_count} 条，疑点 {suspect_count} 条",
        "structured_count": structured_count,
        "suspect_count": suspect_count,
    }


def translation_completeness_check(
    work_paths: WorkPaths,
    *,
    manifest_key_sort: Callable[[str], tuple[int, int | str]],
) -> dict[str, Any]:
    if not work_paths.translated_manifest.exists():
        return skip("translation_completeness", "未运行翻译阶段")
    payload = read_json(work_paths.translated_manifest, default={})
    if not isinstance(payload, dict) or not payload:
        return fail("translation_completeness", "翻译产物缺失或不是字幕字典")

    split_payload = read_json(work_paths.split_manifest, default={})
    if not isinstance(split_payload, dict) or not split_payload:
        blank_count = sum(
            1
            for item in payload.values()
            if not isinstance(item, dict) or not str(item.get("translated_subtitle", "")).strip()
        )
        if blank_count:
            return fail(
                "translation_completeness",
                f"翻译产物存在 {blank_count} 条空译文",
                blank_count=blank_count,
            )
        return passed("translation_completeness", f"翻译产物完整：{len(payload)} 条")

    expected_keys = {str(key) for key in split_payload.keys()}
    translated_keys = {str(key) for key in payload.keys()}
    missing_keys = sorted(expected_keys - translated_keys, key=manifest_key_sort)
    blank_keys = sorted(
        (
            key
            for key in expected_keys & translated_keys
            if not isinstance(payload.get(key), dict) or not str(payload[key].get("translated_subtitle", "")).strip()
        ),
        key=manifest_key_sort,
    )
    extra_count = len(translated_keys - expected_keys)
    if missing_keys or blank_keys:
        return fail(
            "translation_completeness",
            (
                "翻译产物未覆盖当前 split："
                f"split {len(expected_keys)} 条，translated {len(translated_keys)} 条，"
                f"缺失 {len(missing_keys)} 条，空译文 {len(blank_keys)} 条"
            ),
            split_count=len(expected_keys),
            translated_count=len(translated_keys),
            missing_count=len(missing_keys),
            blank_count=len(blank_keys),
            extra_count=extra_count,
            missing_keys=missing_keys[:20],
            blank_keys=blank_keys[:20],
        )
    status = "WARN" if extra_count else "PASS"
    message = (
        f"翻译产物覆盖当前 split：{len(expected_keys)} 条"
        if status == "PASS"
        else f"翻译产物覆盖当前 split，但存在 {extra_count} 条额外 key"
    )
    return {
        "name": "translation_completeness",
        "status": status,
        "message": message,
        "split_count": len(expected_keys),
        "translated_count": len(translated_keys),
        "extra_count": extra_count,
    }
