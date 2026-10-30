from __future__ import annotations

from pathlib import Path

from qwen_asr.mimo_checkpoints import (
    completed_segment_ids,
    load_existing_branch,
    load_existing_report,
    pending_review_ids,
)
from qwen_asr.storage import write_json_atomic


def test_load_existing_branch_prefers_checkpoint_and_copies_dict_values(tmp_path: Path) -> None:
    checkpoint = tmp_path / "branch.json"
    write_json_atomic(checkpoint, {"1": {"translated_subtitle": "done"}})
    translated = {"1": {"translated_subtitle": "source"}}

    branch = load_existing_branch(checkpoint, translated)
    branch["1"]["translated_subtitle"] = "changed"

    assert translated["1"]["translated_subtitle"] == "source"
    assert load_existing_branch(checkpoint, translated)["1"]["translated_subtitle"] == "done"


def test_load_existing_branch_falls_back_to_translated_when_checkpoint_empty(tmp_path: Path) -> None:
    translated = {"1": {"translated_subtitle": "source"}}
    branch = load_existing_branch(tmp_path / "missing.json", translated)
    branch["1"]["translated_subtitle"] = "changed"

    assert translated["1"]["translated_subtitle"] == "source"


def test_load_existing_report_filters_non_dict_items(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    write_json_atomic(report, [{"segment_id": "a"}, "bad", 1, {"segment_id": "b"}])

    assert load_existing_report(report) == [{"segment_id": "a"}, {"segment_id": "b"}]


def test_completed_and_pending_ids_honor_resume_flag() -> None:
    report = [
        {"segment_id": "segment_1", "status": "completed"},
        {"segment_id": "segment_2", "status": "failed"},
        {"segment_id": "segment_3", "status": "completed"},
    ]
    stage2 = [
        {"id": "1", "status": "completed"},
        {"id": "2", "status": "failed"},
    ]

    assert completed_segment_ids(report) == {"segment_1", "segment_3"}
    assert pending_review_ids(["1", "2", "3"], stage2, resume=True) == ["2", "3"]
    assert pending_review_ids(["1", "2", "3"], stage2, resume=False) == ["1", "2", "3"]
