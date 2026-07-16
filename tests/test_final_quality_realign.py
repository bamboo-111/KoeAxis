from __future__ import annotations

from pathlib import Path

from qwen_asr import final_quality
from qwen_asr.final_quality_realign import normalize_status, proofread_realign_check
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_proofread_realign_skips_when_mimo_manifest_is_missing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    check = proofread_realign_check(paths)

    assert check["status"] == "PASS"
    assert check["skipped"] is True


def test_proofread_realign_fails_invalid_mimo_manifest_shape(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_manifest, ["bad"])

    check = proofread_realign_check(paths)

    assert check["status"] == "FAIL"


def test_proofread_realign_fails_pending_and_failed_realign_items(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {"needs_realign": True, "realign_status": "pending"},
            "2": {"needs_realign": True, "realign_status": "failed"},
            "3": {"needs_realign": True, "realign_status": "completed"},
        },
    )

    check = proofread_realign_check(paths)

    assert check["status"] == "FAIL"
    assert check["pending_ids"] == ["1", "2"]
    assert check["failed_ids"] == ["2"]
    assert final_quality._proofread_realign_check(paths) == check


def test_proofread_realign_fails_failed_report(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_manifest, {"1": {"needs_realign": False}})
    write_json_atomic(paths.workdir / "reports" / "proofread_realign.json", {"status": "FAIL"})

    check = proofread_realign_check(paths)

    assert check["status"] == "FAIL"
    assert check["report"].endswith("proofread_realign.json")


def test_proofread_realign_warns_with_report_counters(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_manifest, {"1": {"needs_realign": False}})
    write_json_atomic(
        paths.workdir / "reports" / "proofread_realign.json",
        {
            "status": "WARN",
            "fallback_count": 1,
            "mfa_completed_count": 2,
            "mfa_unusable_count": 3,
            "mfa_rejected_count": 4,
        },
    )

    check = proofread_realign_check(paths)

    assert check["status"] == "WARN"
    assert check["fallback_count"] == 1
    assert check["mfa_completed_count"] == 2
    assert check["mfa_unusable_count"] == 3
    assert check["mfa_rejected_count"] == 4


def test_proofread_realign_passes_when_items_and_report_are_clean(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_manifest, {"1": {"needs_realign": True, "realign_status": "completed"}})
    write_json_atomic(paths.workdir / "reports" / "proofread_realign.json", {"status": "PASS"})

    check = proofread_realign_check(paths)

    assert check["status"] == "PASS"


def test_realign_normalize_status_defaults_unknown_values_to_warn() -> None:
    assert normalize_status("pass") == "PASS"
    assert normalize_status("unknown") == "WARN"
