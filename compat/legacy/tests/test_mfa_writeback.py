from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_experiment import (
    _build_mfa_writeback_decision,
    _find_writeback_manifest_target,
    apply_mfa_local_writeback as legacy_apply_mfa_local_writeback,
)
from qwen_asr.mfa_writeback import (
    apply_mfa_local_writeback,
    build_mfa_writeback_decision,
    find_writeback_manifest_target,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


def _completed_run() -> dict[str, object]:
    return {
        "status": "completed",
        "candidate": {
            "start_ms": 1000,
            "end_ms": 1300,
            "text": "\u306f\u3044",
            "details": {"target_start_ms": 1000, "target_end_ms": 1300},
        },
        "local_ass_guard": {
            "status": "PASS",
            "mfa_text": "\u306f\u3044",
            "mfa_start_ms": 1080,
            "mfa_end_ms": 1240,
        },
        "writeback_dry_run": {"status": "PASS"},
    }


def test_find_writeback_manifest_target_prefers_closest_midpoint() -> None:
    manifest = {
        "near": {"start_time": 1000, "end_time": 1300, "original_subtitle": "\u306f\u3044"},
        "far": {"start_time": 1500, "end_time": 1800, "original_subtitle": "\u306f\u3044"},
    }
    candidate = {"start_ms": 1010, "end_ms": 1290}

    target = find_writeback_manifest_target(manifest, candidate)
    legacy_target = _find_writeback_manifest_target(manifest, candidate)

    assert target is not None
    assert target[0] == "near"
    assert legacy_target is not None
    assert legacy_target[0] == "near"


def test_build_mfa_writeback_decision_rejects_invalid_guard() -> None:
    manifest = {"1": {"start_time": 1000, "end_time": 1300, "original_subtitle": "\u306f\u3044"}}
    run = _completed_run()
    run["local_ass_guard"] = {"status": "FAIL", "mfa_text": "\u306f\u3044", "mfa_start_ms": 1080, "mfa_end_ms": 1240}

    decision = build_mfa_writeback_decision(manifest, run)

    assert decision["status"] == "REJECT"
    assert "local-guard-not-pass" in decision["reasons"]
    assert _build_mfa_writeback_decision(manifest, run)["status"] == "REJECT"


def test_apply_mfa_local_writeback_proposes_without_writing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 1000, "end_time": 1300, "original_subtitle": "\u306f\u3044"}},
    )

    report = apply_mfa_local_writeback(paths, [_completed_run()], mode="propose")

    assert report["status"] == "NOOP"
    assert report["proposed_count"] == 1
    assert report["items"][0]["status"] == "APPLY"
    assert report["output_manifest"] == ""


def test_apply_mfa_local_writeback_applies_manifest_copy(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    output = tmp_path / "split.mfa.json"
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 1000, "end_time": 1300, "original_subtitle": "\u306f\u3044"}},
    )

    report = legacy_apply_mfa_local_writeback(paths, [_completed_run()], mode="apply", output_path=output)

    assert report["status"] == "APPLIED"
    assert report["applied_count"] == 1
    updated = read_json(output, default={})
    assert updated["1"]["start_time"] == 1080
    assert updated["1"]["end_time"] == 1240
    assert updated["1"]["mfa_local_writeback"]["mfa_text"] == "\u306f\u3044"


def test_apply_mfa_local_writeback_skips_invalid_manifest(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.split_manifest.write_text("[]", encoding="utf-8")

    report = apply_mfa_local_writeback(paths, [_completed_run()], mode="apply")

    assert report["status"] == "SKIP"
    assert report["reason"] == "split-manifest-invalid"
