from __future__ import annotations

from pathlib import Path

from qwen_asr import ass_quality_diff
from qwen_asr.ass_quality_diff import build_ass_quality_diff_report, parse_report_specs
from qwen_asr.storage import write_json_atomic


def _row(index: int, score: float, text: str, matched: str, diagnostics: list[str] | None = None) -> dict[str, object]:
    return {
        "index": index,
        "ass_start_ms": index * 1000,
        "ass_end_ms": index * 1000 + 500,
        "ass_text": text,
        "matched_text": matched,
        "matched_start_ms": index * 1000,
        "matched_end_ms": index * 1000 + 500,
        "ass_normalized_chars": len(text),
        "matched_normalized_chars": len(matched),
        "score": score,
        "level": "fail" if score < 0.20 else ("low" if score < 0.45 else ("warn" if score < 0.75 else "ok")),
        "diagnostics": diagnostics or [],
    }


def _report(source: str, rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": "FAIL",
        "source": source,
        "offset_ms": 0,
        "selected_dialogue_count": len(rows),
        "source_cue_count": len(rows),
        "summary": {
            "mean": round(sum(float(row["score"]) for row in rows) / len(rows), 6),
            "score_lt_045": sum(float(row["score"]) < 0.45 for row in rows),
            "score_lt_020": sum(float(row["score"]) < 0.20 for row in rows),
            "short_dialogue_missing": sum("short-dialogue-missing" in row["diagnostics"] for row in rows),
            "short_dialogue_timing_shifted": sum("short-dialogue-timing-shifted" in row["diagnostics"] for row in rows),
        },
        "rows": rows,
    }


def test_ass_quality_diff_keeps_stable_production_api() -> None:
    assert set(ass_quality_diff.__all__) == {
        "build_ass_quality_diff_report",
        "cmd_ass_quality_diff",
        "parse_report_specs",
        "render_markdown_report",
    }


def test_ass_quality_diff_reports_stage_regressions(tmp_path: Path) -> None:
    first = tmp_path / "transcript.json"
    second = tmp_path / "split.json"
    write_json_atomic(
        first,
        _report(
            "transcript",
            [
                _row(1, 0.90, "\u306f\u3044", "\u306f\u3044"),
                _row(2, 0.80, "\u9577\u3044\u53f0\u8a5e", "\u9577\u3044\u53f0\u8a5e"),
                _row(3, 0.70, "\u6b8b\u308b\u884c", "\u6b8b\u308b\u884c"),
            ],
        ),
    )
    write_json_atomic(
        second,
        _report(
            "split",
            [
                _row(1, 0.00, "\u306f\u3044", "", ["short-dialogue-low-score", "short-dialogue-missing"]),
                _row(2, 0.50, "\u9577\u3044\u53f0\u8a5e", "\u9577\u3044"),
                _row(3, 0.68, "\u6b8b\u308b\u884c", "\u6b8b\u308b\u884c"),
            ],
        ),
    )

    report = build_ass_quality_diff_report(parse_report_specs([f"transcript={first}", f"split={second}"]))

    assert report["status"] == "FAIL"
    assert report["summary"]["fail_issue_count"] >= 1
    issue_types = {item["type"] for item in report["issues"]}
    assert "became-fail" in issue_types
    assert "diagnostic-added" in issue_types
    assert "matched-text-shortened" in issue_types
    assert report["transitions"][0]["from"] == "transcript"
    assert report["transitions"][0]["to"] == "split"


def test_ass_quality_diff_passes_when_no_regression(tmp_path: Path) -> None:
    first = tmp_path / "aligned.json"
    second = tmp_path / "export.json"
    rows = [_row(1, 0.90, "\u306f\u3044", "\u306f\u3044")]
    write_json_atomic(first, _report("aligned", rows))
    write_json_atomic(second, _report("export", rows))

    report = build_ass_quality_diff_report(parse_report_specs([str(first), str(second)]))

    assert report["status"] == "PASS"
    assert report["summary"]["issue_count"] == 0
