from __future__ import annotations

from scripts.review_scope import (
    build_change_groups,
    build_source_test_mapping,
    group_for_path,
)


def test_source_test_mapping_uses_direct_and_command_integration_tests() -> None:
    untracked = [
        "qwen_asr/mimo_audio.py",
        "qwen_asr/commands/align.py",
    ]
    tests = {
        "tests/test_mimo_audio.py",
        "tests/test_pipeline_runner.py",
        "tests/test_align_cleanup.py",
        "tests/test_align_quality.py",
    }

    report = build_source_test_mapping(untracked, tests)

    assert report["production_module_count"] == 2
    assert report["mapped_count"] == 2
    assert report["unmapped_count"] == 0
    records = {item["production_path"]: item for item in report["records"]}
    assert records["qwen_asr/mimo_audio.py"]["coverage_kind"] == "direct"
    assert records["qwen_asr/commands/align.py"]["coverage_kind"] == "integration"


def test_source_test_mapping_reports_unmapped_module() -> None:
    report = build_source_test_mapping(["qwen_asr/new_boundary.py"], set())

    assert report["mapped_count"] == 0
    assert report["unmapped_count"] == 1
    assert report["records"][0]["status"] == "UNMAPPED"


def test_change_groups_keep_unknown_paths_visible() -> None:
    paths = ["qwen_asr/mimo_audio.py", "README.md", "1/unexplained.txt"]
    states = {path: "untracked" for path in paths}

    report = build_change_groups(paths, states)

    assert report["review_path_count"] == 3
    assert report["unclassified_count"] == 1
    assert group_for_path("qwen_asr/mimo_audio.py") == "translation_mimo"
    assert group_for_path("README.md") == "tests_docs_acceptance"
    assert group_for_path("docs/项目文件与代码整理计划.md") == "tests_docs_acceptance"
    assert group_for_path("samples/翻译对照.xlsx") == "tests_docs_acceptance"
    assert group_for_path("1/unexplained.txt") == "unclassified"
