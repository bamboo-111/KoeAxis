from __future__ import annotations

from qwen_asr.final_quality import (
    _int_or_none,
    _is_protected_short_subtitle,
    _manifest_key_sort,
    _subtitle_display_text,
    _subtitle_readability_check,
    _subtitle_readability_issue,
)
from qwen_asr.final_quality_readability import (
    int_or_none,
    is_protected_short_subtitle,
    subtitle_display_text,
    subtitle_readability_check,
    subtitle_readability_issue,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_readability_check_fails_non_positive_duration(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 1000, "end_time": 1000, "original_subtitle": "\u306f\u3044"}},
    )

    check = subtitle_readability_check(paths, manifest_key_sort=_manifest_key_sort)

    assert check["status"] == "FAIL"
    assert check["issues"][0]["type"] == "non_positive_duration"
    assert _subtitle_readability_check(paths)["status"] == "FAIL"


def test_readability_check_warns_for_protected_short_duration(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 1000, "end_time": 1100, "original_subtitle": "\u306f\u3044"}},
    )

    check = subtitle_readability_check(paths, manifest_key_sort=_manifest_key_sort)

    assert check["status"] == "WARN"
    assert check["issues"][0]["type"] == "protected_short_too_fast"


def test_readability_check_sorts_manifest_keys_numerically(tmp_path) -> None:  # noqa: ANN001
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "10": {"start_time": 900, "end_time": 1500, "original_subtitle": "later"},
            "2": {"start_time": 500, "end_time": 1000, "original_subtitle": "middle"},
        },
    )

    check = subtitle_readability_check(paths, manifest_key_sort=_manifest_key_sort)

    assert check["issues"][0]["key"] == "10"
    assert check["issues"][0]["type"] == "overlap"


def test_readability_helper_compatibility_aliases() -> None:
    item = {"translated_subtitle": "\u662f", "text": "\u306f\u3044"}

    assert is_protected_short_subtitle("\u306f\u3044")
    assert _is_protected_short_subtitle("\u306f\u3044")
    assert int_or_none("12.4") == _int_or_none("12.4") == 12
    assert subtitle_display_text(item) == _subtitle_display_text(item) == "\u306f\u3044"
    assert subtitle_readability_issue("WARN", "split", "1", "kind", "msg") == _subtitle_readability_issue(
        "WARN",
        "split",
        "1",
        "kind",
        "msg",
    )
