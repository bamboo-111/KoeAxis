from __future__ import annotations

from qwen_asr.final_quality import (
    _manifest_key_sort,
    _translation_completeness_check,
    _translation_structure_check,
)
from qwen_asr.final_quality_translation import translation_completeness_check, translation_structure_check
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_translation_checks_skip_when_translation_missing(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)

    structure = translation_structure_check(paths)
    completeness = translation_completeness_check(paths, manifest_key_sort=_manifest_key_sort)

    assert structure["status"] == "PASS"
    assert structure["skipped"] is True
    assert completeness["status"] == "PASS"
    assert completeness["skipped"] is True


def test_translation_structure_fails_missing_structured_fields(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.translated_manifest, {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"}})

    check = translation_structure_check(paths)

    assert check["status"] == "FAIL"
    assert "structured suspect fields" in check["message"]
    assert _translation_structure_check(paths)["status"] == "FAIL"


def test_translation_structure_fails_invalid_suspect_fields(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_audio_review": True,
                "suspect_types": "not-list",
            }
        },
    )

    check = translation_structure_check(paths)

    assert check["status"] == "FAIL"
    assert check["invalid_suspect_fields"] == 1


def test_translation_completeness_reports_missing_keys_sorted(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "10": {"original_subtitle": "later"},
            "2": {"original_subtitle": "middle"},
            "1": {"original_subtitle": "first"},
        },
    )
    write_json_atomic(
        paths.translated_manifest,
        {"1": {"translated_subtitle": "\u662f"}},
    )

    check = translation_completeness_check(paths, manifest_key_sort=_manifest_key_sort)

    assert check["status"] == "FAIL"
    assert check["missing_keys"] == ["2", "10"]
    assert _translation_completeness_check(paths)["missing_keys"] == ["2", "10"]


def test_translation_completeness_warns_extra_keys(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {"translated_subtitle": "\u662f"},
            "2": {"translated_subtitle": "\u5426"},
        },
    )

    check = translation_completeness_check(paths, manifest_key_sort=_manifest_key_sort)

    assert check["status"] == "WARN"
    assert check["extra_count"] == 1
