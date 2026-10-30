from __future__ import annotations

from pathlib import Path

from qwen_asr import final_quality
from qwen_asr.final_quality_postproofread import (
    is_mimo_original_change,
    post_guard_issue,
    post_proofread_content_regressed,
    post_proofread_guard_check,
    post_proofread_original_change_issue,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def _entry(*, before: str = "はい", after: str = "いいえ", ass_guard: dict | None = None) -> dict:
    evidence = {} if ass_guard is None else {"ass_guard": ass_guard}
    return {
        "source": "mimo-nearby-audio",
        "changes": {"original_subtitle": {"before": before, "after": after}},
        "evidence": evidence,
    }


def test_post_proofread_guard_skips_when_mimo_manifest_is_missing(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    check = post_proofread_guard_check(paths)

    assert check["status"] == "PASS"
    assert check["skipped"] is True


def test_is_mimo_original_change_requires_mimo_source_and_original_change() -> None:
    assert is_mimo_original_change({"source": "mimo-stage2", "changes": {"original_subtitle": {}}})
    assert not is_mimo_original_change({"source": "human", "changes": {"original_subtitle": {}}})
    assert not is_mimo_original_change({"source": "mimo-stage2", "changes": {"translated_subtitle": {}}})
    assert final_quality._is_mimo_original_change({"source": "mimo-stage2", "changes": {"original_subtitle": {}}})


def test_post_proofread_change_issue_requires_ass_guard() -> None:
    issue = post_proofread_original_change_issue("1", _entry())

    assert issue is not None
    assert issue["type"] == "missing_ass_guard"
    assert final_quality._post_proofread_original_change_issue("1", _entry()) == issue


def test_post_proofread_change_issue_rejects_unaccepted_ass_guard() -> None:
    issue = post_proofread_original_change_issue(
        "1",
        _entry(ass_guard={"accepted": False, "reason": "low-score"}),
    )

    assert issue is not None
    assert issue["type"] == "ass_guard_rejected_but_applied"
    assert issue["reason"] == "low-score"


def test_post_proofread_change_issue_rejects_score_regression() -> None:
    issue = post_proofread_original_change_issue(
        "1",
        _entry(ass_guard={"accepted": True, "current_score": 0.8, "suggested_score": 0.7}),
    )

    assert issue is not None
    assert issue["type"] == "ass_score_regression"


def test_post_proofread_change_issue_requires_ass_support_for_content_regression() -> None:
    issue = post_proofread_original_change_issue(
        "1",
        _entry(
            before="はいです",
            after="で",
            ass_guard={"accepted": True, "reason": "manual", "current_score": 0.5, "suggested_score": 0.5},
        ),
    )

    assert issue is not None
    assert issue["type"] == "content_regression_without_ass_support"
    assert post_proofread_content_regressed("はいです", "で")
    assert final_quality._post_proofread_content_regressed("はいです", "で")


def test_post_proofread_change_issue_allows_supported_change() -> None:
    issue = post_proofread_original_change_issue(
        "1",
        _entry(ass_guard={"accepted": True, "reason": "ass-improved", "current_score": 0.3, "suggested_score": 0.8}),
    )

    assert issue is None


def test_post_proofread_guard_reports_first_issues_and_count(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            str(index): {
                "proofread_history": [
                    _entry(before="はいです", after="で", ass_guard={"accepted": True, "reason": "manual"})
                ]
            }
            for index in range(25)
        },
    )

    check = post_proofread_guard_check(paths)

    assert check["status"] == "FAIL"
    assert check["checked_change_count"] == 25
    assert check["issue_count"] == 25
    assert len(check["issues"]) == 20


def test_post_proofread_guard_passes_when_all_original_changes_are_supported(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "proofread_history": [
                    _entry(ass_guard={"accepted": True, "reason": "ass-improved", "current_score": 0.3, "suggested_score": 0.8})
                ]
            }
        },
    )

    check = post_proofread_guard_check(paths)

    assert check["status"] == "PASS"
    assert check["checked_change_count"] == 1


def test_post_guard_issue_keeps_extra_fields() -> None:
    issue = post_guard_issue("1", "kind", "message", value=3)

    assert issue == {"subtitle_id": "1", "type": "kind", "message": "message", "value": 3}
    assert final_quality._post_guard_issue("1", "kind", "message", value=3) == issue
