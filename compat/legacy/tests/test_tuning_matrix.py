from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.storage import read_json, write_json_atomic
from tools.tuning_matrix import build_tuning_matrix, cmd_tuning_matrix


def test_build_tuning_matrix_has_stage_pass_criteria() -> None:
    matrix = build_tuning_matrix()

    assert len(matrix) >= 10
    assert matrix[0]["id"] == "prepare_vad"
    assert all(item["switches"] for item in matrix)
    assert all(item["variants"] for item in matrix)
    assert all(item["pass_criteria"] for item in matrix)


def test_cmd_tuning_matrix_summarizes_reports(tmp_path: Path) -> None:
    ass_report = tmp_path / "ass.json"
    content_report = tmp_path / "content.json"
    realign_report = tmp_path / "proofread_realign.json"
    output = tmp_path / "matrix.json"
    markdown = tmp_path / "matrix.md"
    write_json_atomic(
        ass_report,
        {
            "status": "FAIL",
            "offset_ms": 6180,
            "summary": {
                "score_lt_020": 3,
                "score_lt_045": 8,
                "short_dialogue_low_score": 2,
                "short_dialogue_timing_shifted": 1,
                "short_dialogue_missing": 1,
                "overlong_match": 4,
            },
        },
    )
    write_json_atomic(
        content_report,
        {
            "status": "WARN",
            "summary": {"fail_count": 0, "warn_count": 2},
        },
    )
    write_json_atomic(
        realign_report,
        {
            "status": "WARN",
            "pending_count": 2,
            "fallback_count": 1,
            "mfa_completed_count": 1,
            "mfa_unusable_count": 0,
            "mfa_rejected_count": 1,
        },
    )

    status = cmd_tuning_matrix(
        argparse.Namespace(
            output=str(output),
            markdown_output=str(markdown),
            ass_quality_report=[f"madougushi={ass_report}"],
            content_quality_report=[f"konoato={content_report}"],
            proofread_realign_report=[f"konoato={realign_report}"],
        )
    )

    payload = read_json(output)
    text = markdown.read_text(encoding="utf-8")
    assert status == 0
    assert payload["status"] == "FAIL"
    assert payload["current_baselines"]["ass_quality"][0]["label"] == "madougushi"
    assert payload["current_baselines"]["proofread_realign"][0]["mfa_rejected_count"] == 1
    assert "\u5b57\u5e55\u6d41\u7a0b\u9010\u9636\u6bb5\u8c03\u53c2\u77e9\u9635" in text
    assert "\u77ed\u5bf9\u767d\u4f4e\u5206 2" in text
    assert "\u77ed\u5bf9\u767d\u7591\u4f3c\u9519\u65f6 1" in text
    assert "\u77ed\u5bf9\u767d\u7591\u4f3c\u7f3a\u5931 1" in text
    assert "\u8fc7\u957f\u5339\u914d 4" in text
    assert "\u5185\u5bb9\u5b88\u6052 konoato" in text
    assert "\u590d\u6838\u540e\u91cd\u5bf9\u9f50 konoato" in text
    assert "MFA \u6210\u529f 1" in text
    assert "MFA \u62d2\u7edd 1" in text


def test_cmd_tuning_matrix_defaults_missing_mfa_counts_to_zero(tmp_path: Path) -> None:
    realign_report = tmp_path / "proofread_realign.json"
    output = tmp_path / "matrix.json"
    markdown = tmp_path / "matrix.md"
    write_json_atomic(
        realign_report,
        {
            "status": "WARN",
            "pending_count": 1,
            "fallback_count": 1,
        },
    )

    cmd_tuning_matrix(
        argparse.Namespace(
            output=str(output),
            markdown_output=str(markdown),
            ass_quality_report=[],
            content_quality_report=[],
            proofread_realign_report=[f"legacy={realign_report}"],
        )
    )

    text = markdown.read_text(encoding="utf-8")
    assert "MFA \u6210\u529f 0" in text
    assert "MFA \u4e0d\u53ef\u7528 0" in text
    assert "MFA \u62d2\u7edd 0" in text
