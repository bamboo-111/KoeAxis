from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_candidates import (
    candidates_from_ass_quality_diff,
    candidates_from_content_quality,
    candidates_from_mimo_manifest,
    candidates_from_proofread_realign,
    collect_alignment_experiment_candidates,
    dedupe_and_rank_candidates,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_candidates_from_content_quality_filters_alignment_issue_types(tmp_path: Path) -> None:
    report = tmp_path / "content_quality.json"
    write_json_atomic(
        report,
        {
            "issues": [
                {
                    "type": "missing_short_response",
                    "severity": "FAIL",
                    "start_ms": "1000",
                    "end_ms": 1300,
                    "text": "\u306f\u3044",
                },
                {"type": "ordinary_warning", "text": "\u30b9\u30ad\u30c3\u30d7"},
            ]
        },
    )

    result = candidates_from_content_quality(report)

    assert result == [
        {
            "source": "content-quality",
            "reason": "missing_short_response",
            "severity": "FAIL",
            "subtitle_id": "",
            "start_ms": 1000,
            "end_ms": 1300,
            "text": "\u306f\u3044",
            "details": {
                "type": "missing_short_response",
                "severity": "FAIL",
                "start_ms": "1000",
                "end_ms": 1300,
                "text": "\u306f\u3044",
            },
        }
    ]


def test_candidates_from_proofread_realign_keeps_failed_and_fallback_items(
    tmp_path: Path,
) -> None:
    report = tmp_path / "proofread_realign.json"
    write_json_atomic(
        report,
        {
            "items": [
                {
                    "id": "7",
                    "status": "failed",
                    "before_start_time": 2000,
                    "before_end_time": 2300,
                },
                {"id": "8", "status": "completed"},
            ]
        },
    )

    result = candidates_from_proofread_realign(report)

    assert result[0]["source"] == "proofread-realign"
    assert result[0]["reason"] == "failed"
    assert result[0]["severity"] == "FAIL"
    assert result[0]["subtitle_id"] == "7"


def test_candidates_from_ass_quality_diff_keeps_short_dialogue_regressions(
    tmp_path: Path,
) -> None:
    report = tmp_path / "ass_quality_diff.json"
    write_json_atomic(
        report,
        {
            "issues": [
                {
                    "type": "score-drop",
                    "severity": "WARN",
                    "ass_start_ms": 3000,
                    "ass_end_ms": 3300,
                    "ass_text": "\u3046\u3093",
                    "current_diagnostics": [],
                },
                {
                    "type": "unchanged",
                    "current_diagnostics": ["short-dialogue-missing"],
                    "ass_text": "\u306f\u3044",
                },
                {"type": "unchanged", "current_diagnostics": []},
            ]
        },
    )

    result = candidates_from_ass_quality_diff(report)

    assert [item["reason"] for item in result] == ["score-drop", "unchanged"]
    assert result[1]["details"]["current_diagnostics"] == ["short-dialogue-missing"]


def test_candidates_from_mimo_manifest_requires_change_or_realign_flag(tmp_path: Path) -> None:
    manifest = tmp_path / "mimo.json"
    write_json_atomic(
        manifest,
        {
            "1": {
                "start_time": 100,
                "end_time": 300,
                "original_subtitle": "\u306f\u3044",
                "proofread_history": [{"source": "mimo-stage2"}],
            },
            "2": {
                "start_time": 400,
                "end_time": 700,
                "original_subtitle": "\u3046\u3093",
                "needs_realign": True,
            },
            "3": {"original_subtitle": "\u9664\u5916"},
        },
    )

    result = candidates_from_mimo_manifest(manifest)

    assert [item["subtitle_id"] for item in result] == ["1", "2"]
    assert all(item["source"] == "mimo-proofread" for item in result)


def test_collect_alignment_candidates_dedupes_and_orders_sources(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.content_quality_report,
        {
            "issues": [
                {
                    "type": "missing_short_response",
                    "severity": "FAIL",
                    "start_ms": 1000,
                    "end_ms": 1300,
                    "text": "\u306f\u3044",
                }
            ]
        },
    )
    realign_report = paths.workdir / "reports" / "proofread_realign.json"
    write_json_atomic(
        realign_report,
        {
            "items": [
                {
                    "id": "9",
                    "status": "failed",
                    "before_start_time": 2000,
                    "before_end_time": 2300,
                }
            ]
        },
    )

    result = collect_alignment_experiment_candidates(
        paths,
        ass_quality_report_paths=[],
        ass_quality_diff_report_paths=[],
        max_candidates=10,
    )

    assert [item["source"] for item in result] == [
        "proofread-realign",
        "content-quality",
    ]


def test_dedupe_and_rank_candidates_prefers_single_matching_key() -> None:
    first = {
        "source": "content-quality",
        "reason": "missing_short_response",
        "severity": "FAIL",
        "subtitle_id": "",
        "start_ms": 1000,
        "end_ms": 1300,
        "text": "\u306f\u3044",
    }
    duplicate = dict(first)
    later = {
        "source": "mimo-proofread",
        "reason": "mimo-change-needs-alignment-check",
        "severity": "WARN",
        "subtitle_id": "2",
        "start_ms": 500,
        "end_ms": 800,
        "text": "\u3046\u3093",
    }

    result = dedupe_and_rank_candidates([later, first, duplicate])

    assert result == [first, later]
