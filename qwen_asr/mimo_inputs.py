from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def load_pipeline_inputs(workdir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    work_paths = WorkPaths.from_workdir(workdir)
    segments = read_json(work_paths.segments_manifest, default=[])
    translated = read_json(work_paths.translated_manifest, default={})
    split = read_json(work_paths.split_manifest, default={})
    if isinstance(translated, dict) and isinstance(split, dict) and split:
        validate_translated_manifest_complete_for_split(translated, split)
    return (
        segments if isinstance(segments, list) else [],
        translated if isinstance(translated, dict) else {},
    )


def validate_translated_manifest_complete_for_split(
    translated: dict[str, Any],
    split: dict[str, Any],
) -> None:
    expected_keys = {str(key) for key in split.keys()}
    translated_keys = {str(key) for key in translated.keys()}
    missing_keys = sorted(expected_keys - translated_keys, key=manifest_key_sort)
    blank_keys = sorted(
        (
            key
            for key in expected_keys & translated_keys
            if not isinstance(translated.get(key), dict)
            or not str(translated[key].get("translated_subtitle", "")).strip()
        ),
        key=manifest_key_sort,
    )
    if missing_keys or blank_keys:
        details: list[str] = []
        if missing_keys:
            details.append(f"missing_keys={','.join(missing_keys[:10])}")
        if blank_keys:
            details.append(f"blank_translation_keys={','.join(blank_keys[:10])}")
        raise RuntimeError(
            "translated_segments.json is incomplete for current split_segments.json; "
            + "; ".join(details)
        )


def manifest_key_sort(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/") + ("" if base_url.rstrip("/").endswith("/v1") else "/v1")


def parse_extra_body(value: str) -> dict[str, Any] | None:
    if not value.strip():
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--llm-extra-body-json must be a JSON object")
    return parsed
