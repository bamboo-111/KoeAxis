from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.mimo_inputs import (
    load_pipeline_inputs,
    manifest_key_sort,
    normalize_base_url,
    parse_extra_body,
    validate_translated_manifest_complete_for_split,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_load_pipeline_inputs_supports_flat_workdir(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.segments_manifest, [{"segment_id": "segment_000001"}])
    write_json_atomic(paths.translated_manifest, {"1": {"translated_subtitle": "ok"}})

    segments, translated = load_pipeline_inputs(tmp_path)

    assert segments == [{"segment_id": "segment_000001"}]
    assert translated == {"1": {"translated_subtitle": "ok"}}


def test_load_pipeline_inputs_rejects_translated_manifest_missing_split_keys(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.segments_manifest, [{"segment_id": "segment_000001"}])
    write_json_atomic(paths.split_manifest, {"1": {}, "2": {}, "10": {}})
    write_json_atomic(paths.translated_manifest, {"1": {"translated_subtitle": "ok"}, "2": {"translated_subtitle": ""}})

    with pytest.raises(RuntimeError) as exc_info:
        load_pipeline_inputs(tmp_path)

    message = str(exc_info.value)
    assert "missing_keys=10" in message
    assert "blank_translation_keys=2" in message


def test_validate_translated_manifest_sorts_numeric_keys() -> None:
    with pytest.raises(RuntimeError) as exc_info:
        validate_translated_manifest_complete_for_split(
            {"1": {"translated_subtitle": "ok"}},
            {"10": {}, "2": {}, "1": {}},
        )

    assert "missing_keys=2,10" in str(exc_info.value)


def test_manifest_key_sort_and_base_url_normalization() -> None:
    assert sorted(["10", "2", "a"], key=manifest_key_sort) == ["2", "10", "a"]
    assert normalize_base_url("https://example.test") == "https://example.test/v1"
    assert normalize_base_url("https://example.test/v1") == "https://example.test/v1"


def test_parse_extra_body_accepts_empty_or_json_object() -> None:
    assert parse_extra_body("") is None
    assert parse_extra_body('{"thinking":{"type":"disabled"}}') == {"thinking": {"type": "disabled"}}


def test_parse_extra_body_rejects_non_object_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        parse_extra_body("[]")
