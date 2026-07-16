from __future__ import annotations

from pathlib import Path

from qwen_asr import final_quality
from qwen_asr.final_quality_mimo import (
    mimo_applied_without_evidence_count,
    mimo_checkpoint_check,
    mimo_two_stage_completed_count,
    pending_audio_review_count,
    quality_suspect_applied_count,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_mimo_checkpoint_skips_when_no_audio_review_work_exists(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    check = mimo_checkpoint_check(paths)

    assert check["status"] == "PASS"
    assert check["skipped"] is True


def test_pending_audio_review_counts_audio_flags(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {"needs_audio_review": True},
            "2": {"asr_suspect": True},
            "3": {"needs_audio_review": False, "asr_suspect": False},
            "4": "bad",
        },
    )

    assert pending_audio_review_count(paths) == 2
    assert final_quality._pending_audio_review_count(paths) == 2


def test_mimo_checkpoint_fails_when_marked_suspects_have_no_report(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.translated_manifest, {"1": {"needs_audio_review": True}})

    check = mimo_checkpoint_check(paths)

    assert check["status"] == "FAIL"
    assert check["pending_audio_review"] == 1


def test_mimo_checkpoint_uses_quality_suspect_report_as_expected_count(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.workdir / "reports" / "quality_suspects.json", {"applied_count": 3})
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "audio_review_candidate_count": 2,
            "stage1_failed": 0,
            "stage2_failed": 0,
            "stage2_completed": 2,
        },
    )

    check = mimo_checkpoint_check(paths)

    assert quality_suspect_applied_count(paths) == 3
    assert check["status"] == "FAIL"
    assert check["expected_candidate_count"] == 3
    assert check["suspect_report_count"] == 3


def test_mimo_checkpoint_counts_completed_stage2_target_ids(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.mimo_proofread_dir.mkdir(parents=True, exist_ok=True)
    stage2_report = paths.mimo_proofread_dir / "stage2.json"
    write_json_atomic(
        stage2_report,
        [
            {"status": "completed", "target_ids": ["1", "2"]},
            {"status": "completed", "id": "3"},
            {"status": "failed", "target_ids": ["4"]},
        ],
    )

    count = mimo_two_stage_completed_count({"stage2_report": str(stage2_report), "stage2_completed": 1}, paths)

    assert count == 3
    assert final_quality._mimo_two_stage_completed_count({"stage2_report": str(stage2_report)}, paths) == 3


def test_mimo_checkpoint_warns_for_missing_application_evidence(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "proofread_history": [
                    {
                        "source": "mimo-stage2",
                        "changes": {"original_subtitle": {"before": "a", "after": "b"}},
                    },
                    {
                        "source": "human",
                        "changes": {"original_subtitle": {"before": "b", "after": "c"}},
                    },
                ]
            }
        },
    )
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "audio_review_candidate_count": 1,
            "stage1_failed": 0,
            "stage2_failed": 0,
            "stage2_completed": 1,
        },
    )

    check = mimo_checkpoint_check(paths)

    assert mimo_applied_without_evidence_count(paths) == 1
    assert check["status"] == "WARN"
    assert check["missing_evidence_count"] == 1


def test_mimo_checkpoint_accepts_legacy_completed_report_list(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_report, [{"status": "completed"}, {"status": "completed"}])

    check = mimo_checkpoint_check(paths)

    assert check["status"] == "PASS"


def test_mimo_checkpoint_fails_failed_non_two_stage_report(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_report, {"status": "FAIL", "failed_count": 2})

    check = mimo_checkpoint_check(paths)

    assert check["status"] == "FAIL"
    assert check["failed_count"] == 2
