from __future__ import annotations

from pathlib import Path

from qwen_asr import final_quality
from qwen_asr.final_quality_ass import ass_quality_checks, normalize_status
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_ass_quality_checks_skip_when_reports_directory_is_missing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    checks = ass_quality_checks(paths)

    assert checks == [
        {
            "name": "ass_quality",
            "status": "PASS",
            "skipped": True,
            "message": "没有发现 ASS 质量报告",
        }
    ]


def test_ass_quality_checks_skip_when_only_quality_suspect_reports_exist(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.workdir / "reports" / "ass_quality.translated.quality_suspects.json", {"status": "FAIL"})

    checks = ass_quality_checks(paths)

    assert checks[0]["status"] == "PASS"
    assert checks[0]["skipped"] is True


def test_ass_quality_checks_reads_status_and_summary(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    report = paths.workdir / "reports" / "ass_quality.translated.json"
    write_json_atomic(report, {"status": "warn", "summary": {"score_lt_045": 2, "score_lt_020": 1}})

    checks = ass_quality_checks(paths)

    assert checks[0]["status"] == "WARN"
    assert checks[0]["report"] == str(report)
    assert "低分 2" in checks[0]["message"]
    assert final_quality._ass_quality_checks(paths) == checks


def test_ass_quality_checks_reports_invalid_json_shape(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    report = paths.workdir / "reports" / "ass_quality.translated.json"
    write_json_atomic(report, ["bad"])

    checks = ass_quality_checks(paths)

    assert checks[0]["status"] == "FAIL"
    assert checks[0]["report"] == str(report)


def test_ass_quality_normalize_status_defaults_unknown_to_warn() -> None:
    assert normalize_status("pass") == "PASS"
    assert normalize_status("bad") == "WARN"
