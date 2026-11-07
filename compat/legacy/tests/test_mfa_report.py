from __future__ import annotations

from qwen_asr.mfa_experiment import _format_ms, _format_time_range
from qwen_asr.mfa_report import (
    format_ms,
    format_time_range,
    render_mfa_alignment_experiment_markdown,
)


def test_format_ms_and_range_keep_legacy_output() -> None:
    assert format_ms(3_723_004) == "01:02:03.004"
    assert format_ms(-50) == "00:00:00.000"
    assert format_time_range(1000, 2300) == "00:00:01.000-00:00:02.300"
    assert format_time_range("1000", 2300) == ""
    assert _format_ms(3_723_004) == format_ms(3_723_004)
    assert _format_time_range(1000, 2300) == format_time_range(1000, 2300)


def test_render_mfa_alignment_markdown_includes_candidates_runs_and_writeback() -> None:
    markdown = render_mfa_alignment_experiment_markdown(
        {
            "status": "READY",
            "reason": "",
            "candidate_count": 1,
            "environment": {
                "available": True,
                "executable": "mfa-bin",
                "invocation": "direct",
                "command": ["mfa-bin"],
                "package_version": "3.0.0",
                "version_output": "mfa 3.0",
            },
            "pass_criteria": {"ass_local_score": "stable"},
            "candidates": [
                {
                    "severity": "FAIL",
                    "source": "content-quality",
                    "reason": "missing_short_response",
                    "start_ms": 1000,
                    "end_ms": 1300,
                    "text": "a|b",
                }
            ],
            "local_alignment_run": {
                "enabled": True,
                "items": [
                    {
                        "status": "completed",
                        "usable": True,
                        "lab_text_source": "candidate",
                        "lab_text": "lab|text",
                        "local_ass_guard": {
                            "status": "PASS",
                            "text_score": 1.0,
                            "mfa_start_ms": 1020,
                            "mfa_end_ms": 1240,
                        },
                        "writeback_dry_run": {
                            "status": "PASS",
                            "score_delta_vs_current": 1.0,
                            "reasons": [],
                        },
                    }
                ],
            },
            "local_writeback": {
                "enabled": True,
                "mode": "apply",
                "status": "APPLIED",
                "applied_count": 1,
                "rejected_count": 0,
                "output_manifest": "split.mfa.json",
                "items": [
                    {
                        "status": "APPLIED",
                        "subtitle_id": "1",
                        "manifest_text": "manifest|text",
                        "mfa_text": "mfa|text",
                        "manifest_text_score": 1.0,
                        "old_start_ms": 1000,
                        "old_end_ms": 1300,
                        "new_start_ms": 1020,
                        "new_end_ms": 1240,
                        "reasons": [],
                    }
                ],
            },
        }
    )

    assert "# MFA 3.0" in markdown
    assert "mfa-bin" in markdown
    assert "00:00:01.000-00:00:01.300" in markdown
    assert "a｜b" in markdown
    assert "lab｜text" in markdown
    assert "manifest｜text" in markdown
    assert "split.mfa.json" in markdown
