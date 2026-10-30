from __future__ import annotations

from pathlib import Path

from tools.align_split_audit import build_align_split_audit_report, render_markdown_report
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_align_split_audit_includes_required_cases_and_classifies_split_loss(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 1.0,
                "global_end_time": 3.0,
                "text": "\u306f\u3044\u3002\u6b21\u3067\u3059\u3002",
                "tokens": [
                    {"text": "\u306f\u3044", "start_time": 1.0, "end_time": 1.4},
                    {"text": "\u6b21\u3067\u3059", "start_time": 1.5, "end_time": 2.5},
                ],
                "status": "completed",
            }
        ],
    )
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"start_time": 1500, "end_time": 2500, "original_subtitle": "\u6b21\u3067\u3059\u3002"},
        },
    )

    aligned_report_path = tmp_path / "aligned.ass.json"
    split_report_path = tmp_path / "split.ass.json"
    diff_report_path = tmp_path / "diff.json"
    write_json_atomic(
        aligned_report_path,
        {
            "summary": {"score_lt_020": 0, "short_dialogue_missing": 0},
            "rows": [
                {
                    "index": 1,
                    "target_start_ms": 1000,
                    "target_end_ms": 1400,
                    "ass_start_ms": 1000,
                    "ass_end_ms": 1400,
                    "ass_text": "\u306f\u3044",
                    "matched_text": "\u306f\u3044",
                    "score": 1.0,
                    "diagnostics": [],
                }
            ],
        },
    )
    write_json_atomic(
        split_report_path,
        {
            "summary": {"score_lt_020": 1, "short_dialogue_missing": 1},
            "rows": [
                {
                    "index": 1,
                    "target_start_ms": 1000,
                    "target_end_ms": 1400,
                    "ass_start_ms": 1000,
                    "ass_end_ms": 1400,
                    "ass_text": "\u306f\u3044",
                    "matched_text": "",
                    "score": 0.0,
                    "level": "fail",
                    "diagnostics": ["short-dialogue-missing"],
                }
            ],
        },
    )
    write_json_atomic(
        diff_report_path,
        {
            "issues": [
                {
                    "type": "became-fail",
                    "severity": "FAIL",
                    "index": 1,
                    "target_start_ms": 1000,
                    "target_end_ms": 1400,
                    "ass_start_ms": 1000,
                    "ass_end_ms": 1400,
                    "ass_text": "\u306f\u3044",
                    "score_drop": 1.0,
                }
            ]
        },
    )

    report = build_align_split_audit_report(
        paths,
        dataset="sample",
        aligned_ass_report=aligned_report_path,
        split_ass_report=split_report_path,
        diff_report=diff_report_path,
    )

    assert report["selection"]["all_became_fail_included"] is True
    assert report["selection"]["all_short_dialogue_missing_included"] is True
    assert report["summary"]["type_counts"]["became-fail"] == 1
    assert report["summary"]["type_counts"]["short-dialogue-missing"] == 1
    assert any(case["root_cause_detail"] == "split-mode-unknown-content-loss" for case in report["cases"])
    assert report["summary"]["stage_owner_counts"]["unknown"] >= 1
    assert "align -> split 根因审计报告" in render_markdown_report(report)


def test_align_split_audit_classifies_token_anomaly_before_split(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 1.0,
                "global_end_time": 2.0,
                "text": "\u306f\u3044",
                "tokens": [
                    {"text": "\u306f", "start_time": 1.0, "end_time": 1.001},
                    {"text": "\u3044", "start_time": 1.001, "end_time": 1.002},
                    {"text": "\u3002", "start_time": 1.002, "end_time": 1.003},
                ],
            }
        ],
    )
    write_json_atomic(paths.split_manifest, {"1": {"start_time": 1000, "end_time": 1003, "original_subtitle": "\u306f\u3044"}})
    aligned_report_path = tmp_path / "aligned.json"
    split_report_path = tmp_path / "split.json"
    diff_report_path = tmp_path / "diff.json"
    row = {
        "index": 1,
        "target_start_ms": 1000,
        "target_end_ms": 1400,
        "ass_start_ms": 1000,
        "ass_end_ms": 1400,
        "ass_text": "\u306f\u3044",
        "matched_text": "\u306f\u3044",
        "score": 0.8,
        "diagnostics": [],
    }
    write_json_atomic(aligned_report_path, {"summary": {}, "rows": [row]})
    write_json_atomic(split_report_path, {"summary": {}, "rows": [{**row, "score": 0.0, "matched_text": ""}]})
    write_json_atomic(diff_report_path, {"issues": [{**row, "type": "score-drop", "severity": "WARN", "score_drop": 0.8}]})

    report = build_align_split_audit_report(
        paths,
        dataset="sample",
        aligned_ass_report=aligned_report_path,
        split_ass_report=split_report_path,
        diff_report=diff_report_path,
    )

    assert report["cases"][0]["root_cause"] == "align token 结构异常"
    assert report["cases"][0]["stage_owner"] == "align"


def test_align_split_audit_uses_split_mode_for_rule_postprocess_attribution(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "segment_000001",
                "global_start_time": 1.0,
                "global_end_time": 3.0,
                "text": "\u306f\u3044\u3002\u6b21\u3067\u3059\u3002",
                "tokens": [
                    {"text": "\u306f\u3044", "start_time": 1.0, "end_time": 1.4},
                    {"text": "\u6b21\u3067\u3059", "start_time": 1.5, "end_time": 2.5},
                ],
                "status": "completed",
            }
        ],
    )
    write_json_atomic(paths.split_manifest, {"1": {"start_time": 1500, "end_time": 2500, "original_subtitle": "\u6b21\u3067\u3059\u3002"}})
    aligned_report_path = tmp_path / "aligned.json"
    split_report_path = tmp_path / "split.json"
    diff_report_path = tmp_path / "diff.json"
    aligned_row = {
        "index": 1,
        "target_start_ms": 1000,
        "target_end_ms": 1400,
        "ass_start_ms": 1000,
        "ass_end_ms": 1400,
        "ass_text": "\u306f\u3044",
        "matched_text": "\u306f\u3044",
        "score": 1.0,
        "diagnostics": [],
    }
    split_row = {**aligned_row, "matched_text": "", "score": 0.0, "diagnostics": ["short-dialogue-missing"]}
    write_json_atomic(aligned_report_path, {"summary": {}, "rows": [aligned_row]})
    write_json_atomic(split_report_path, {"summary": {}, "rows": [split_row]})
    write_json_atomic(diff_report_path, {"issues": [{**aligned_row, "type": "matched-text-shortened", "severity": "FAIL"}]})

    report = build_align_split_audit_report(
        paths,
        dataset="sample",
        aligned_ass_report=aligned_report_path,
        split_ass_report=split_report_path,
        diff_report=diff_report_path,
        audit_split_mode="rule",
    )

    case = report["cases"][0]
    assert case["stage_owner"] == "postprocess"
    assert case["root_cause_detail"] == "rule-or-postprocess-content-loss"
    assert report["summary"]["stage_owner_counts"]["postprocess"] >= 1
