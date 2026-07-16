from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_asr.storage import ensure_directory, read_json, write_json_atomic

__all__ = [
    "build_ass_quality_diff_report",
    "cmd_ass_quality_diff",
    "parse_report_specs",
    "render_markdown_report",
]


DEFAULT_SCORE_DROP_THRESHOLD = 0.15
DEFAULT_LENGTH_DROP_RATIO = 0.60
TRACKED_ADDED_DIAGNOSTICS = (
    "short-dialogue-missing",
    "short-dialogue-timing-shifted",
    "short-dialogue-low-score",
    "overlong-match",
)
LEVEL_RANK = {
    "ok": 0,
    "warn": 1,
    "low": 2,
    "fail": 3,
}


def cmd_ass_quality_diff(args: argparse.Namespace) -> int:
    report = build_ass_quality_diff_report(
        parse_report_specs(getattr(args, "report", []) or []),
        score_drop_threshold=float(getattr(args, "score_drop_threshold", DEFAULT_SCORE_DROP_THRESHOLD)),
        length_drop_ratio=float(getattr(args, "length_drop_ratio", DEFAULT_LENGTH_DROP_RATIO)),
        max_cases=int(getattr(args, "max_cases", 50)),
    )
    output = Path(getattr(args, "output", "") or "ass_quality_diff.json")
    write_json_atomic(output, report)
    markdown_output = str(getattr(args, "markdown_output", "") or "").strip()
    if markdown_output:
        path = Path(markdown_output)
        ensure_directory(path.parent)
        path.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"ASS \u9636\u6bb5\u5dee\u5206{_zh_status(str(report['status']))}\uff1a{output}")
    return 0 if report["status"] != "FAIL" else 1


def parse_report_specs(values: list[str]) -> list[tuple[str | None, Path]]:
    specs: list[tuple[str | None, Path]] = []
    for value in values:
        raw = str(value).strip()
        if not raw:
            continue
        if "=" in raw:
            label, path = raw.split("=", 1)
            specs.append((label.strip() or None, Path(path.strip())))
        else:
            specs.append((None, Path(raw)))
    if len(specs) < 2:
        raise ValueError("ass-quality-diff requires at least two --report entries.")
    return specs


def build_ass_quality_diff_report(
    report_specs: list[tuple[str | None, Path]],
    *,
    score_drop_threshold: float = DEFAULT_SCORE_DROP_THRESHOLD,
    length_drop_ratio: float = DEFAULT_LENGTH_DROP_RATIO,
    max_cases: int = 50,
) -> dict[str, Any]:
    stages = [_load_stage(label, path) for label, path in report_specs]
    transitions: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for previous, current in zip(stages, stages[1:], strict=False):
        transition = _compare_transition(
            previous,
            current,
            score_drop_threshold=score_drop_threshold,
            length_drop_ratio=length_drop_ratio,
        )
        transitions.append(transition["summary"])
        issues.extend(transition["issues"])

    issue_counts: dict[str, int] = {}
    severity_counts = {"FAIL": 0, "WARN": 0}
    for issue in issues:
        issue_counts[str(issue["type"])] = issue_counts.get(str(issue["type"]), 0) + 1
        severity = str(issue.get("severity", "WARN"))
        if severity in severity_counts:
            severity_counts[severity] += 1

    sorted_issues = sorted(
        issues,
        key=lambda item: (
            0 if item.get("severity") == "FAIL" else 1,
            -float(item.get("score_drop", 0.0) or 0.0),
            str(item.get("transition", "")),
            int(item.get("index", 0) or 0),
        ),
    )
    status = "FAIL" if severity_counts["FAIL"] else ("WARN" if issues else "PASS")
    return {
        "status": status,
        "thresholds": {
            "score_drop": score_drop_threshold,
            "length_drop_ratio": length_drop_ratio,
            "tracked_added_diagnostics": list(TRACKED_ADDED_DIAGNOSTICS),
        },
        "stages": [_stage_summary(stage) for stage in stages],
        "transitions": transitions,
        "summary": {
            "issue_count": len(issues),
            "fail_issue_count": severity_counts["FAIL"],
            "warn_issue_count": severity_counts["WARN"],
            "issue_counts": issue_counts,
        },
        "issues": sorted_issues[:max_cases],
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# ASS \u9636\u6bb5\u5dee\u5206\u62a5\u544a",
        "",
        f"- \u72b6\u6001\uff1a{_zh_status(str(report['status']))}",
        f"- \u95ee\u9898\u603b\u6570\uff1a{summary['issue_count']}",
        f"- \u5931\u8d25\u7ea7\u95ee\u9898\uff1a{summary['fail_issue_count']}",
        f"- \u8b66\u544a\u7ea7\u95ee\u9898\uff1a{summary['warn_issue_count']}",
        "",
        "## \u9636\u6bb5\u6982\u89c8",
        "",
        "|\u9636\u6bb5|ASS \u884c\u6570|\u5747\u5206|<0.45|<0.20|\u77ed\u53e5\u7f3a\u5931|\u77ed\u53e5\u9519\u65f6|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        summary_item = stage["summary"]
        lines.append(
            f"|{stage['label']}|{stage['selected_dialogue_count']}|{summary_item.get('mean', 0)}|"
            f"{summary_item.get('score_lt_045', 0)}|{summary_item.get('score_lt_020', 0)}|"
            f"{summary_item.get('short_dialogue_missing', 0)}|"
            f"{summary_item.get('short_dialogue_timing_shifted', 0)}|"
        )
    lines.extend(["", "## \u9636\u6bb5\u8fc7\u6e21", ""])
    for transition in report["transitions"]:
        lines.extend(
            [
                f"### {transition['from']} -> {transition['to']}",
                "",
                f"- \u95ee\u9898\u6570\uff1a{transition['issue_count']}",
                f"- \u5e73\u5747\u5206\u53d8\u5316\uff1a{transition['mean_score_delta']}",
                f"- \u65b0\u589e\u5931\u8d25\uff1a{transition['became_fail_count']}",
                f"- \u65b0\u589e\u4f4e\u5206\uff1a{transition['became_low_count']}",
                f"- \u77ed\u53e5\u65b0\u589e\u5f02\u5e38\uff1a{transition['added_short_dialogue_issue_count']}",
                "",
            ]
        )
    lines.extend(["## \u4e3b\u8981\u9000\u5316\u6837\u672c", ""])
    for item in report["issues"]:
        lines.extend(
            [
                f"### {item['transition']} / ASS {item['index']} / {item['type']}",
                "",
                f"- \u4e25\u91cd\u5ea6\uff1a{_zh_status(str(item['severity']))}",
                f"- ASS \u65f6\u95f4\uff1a{item['ass_start_ms']} - {item['ass_end_ms']}",
                f"- ASS\uff1a{item['ass_text']}",
                f"- \u4e0a\u4e00\u9636\u6bb5\uff1a{item['previous_label']} \u5206\u6570 {item['previous_score']} / {item['previous_matched_text']}",
                f"- \u5f53\u524d\u9636\u6bb5\uff1a{item['current_label']} \u5206\u6570 {item['current_score']} / {item['current_matched_text']}",
                f"- \u8bf4\u660e\uff1a{item['reason']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _load_stage(label: str | None, path: Path) -> dict[str, Any]:
    report = read_json(path)
    if not isinstance(report, dict):
        raise ValueError(f"Invalid ASS quality report: {path}")
    stage_label = label or str(report.get("source") or path.stem)
    rows = report.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError(f"ASS quality report has no row list: {path}")
    by_index = {int(row["index"]): row for row in rows if isinstance(row, dict) and "index" in row}
    return {
        "label": stage_label,
        "path": str(path),
        "report": report,
        "rows": by_index,
    }


def _stage_summary(stage: dict[str, Any]) -> dict[str, Any]:
    report = stage["report"]
    return {
        "label": stage["label"],
        "path": stage["path"],
        "status": report.get("status"),
        "source": report.get("source"),
        "offset_ms": report.get("offset_ms"),
        "selected_dialogue_count": report.get("selected_dialogue_count", len(stage["rows"])),
        "source_cue_count": report.get("source_cue_count", 0),
        "summary": report.get("summary", {}),
    }


def _compare_transition(
    previous: dict[str, Any],
    current: dict[str, Any],
    *,
    score_drop_threshold: float,
    length_drop_ratio: float,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    previous_rows: dict[int, dict[str, Any]] = previous["rows"]
    current_rows: dict[int, dict[str, Any]] = current["rows"]
    shared_indexes = sorted(set(previous_rows) & set(current_rows))
    score_deltas: list[float] = []
    became_fail_count = 0
    became_low_count = 0
    added_short_dialogue_issue_count = 0
    for index in shared_indexes:
        prev = previous_rows[index]
        curr = current_rows[index]
        previous_score = float(prev.get("score", 0.0) or 0.0)
        current_score = float(curr.get("score", 0.0) or 0.0)
        score_drop = round(previous_score - current_score, 6)
        score_deltas.append(round(current_score - previous_score, 6))
        prev_level = str(prev.get("level") or "")
        curr_level = str(curr.get("level") or "")
        previous_diagnostics = set(prev.get("diagnostics", []) or [])
        current_diagnostics = set(curr.get("diagnostics", []) or [])
        added_diagnostics = sorted(current_diagnostics - previous_diagnostics)
        tracked_added = [item for item in added_diagnostics if item in TRACKED_ADDED_DIAGNOSTICS]
        threshold_issue_added = False
        if previous_score >= 0.20 and current_score < 0.20:
            became_fail_count += 1
            threshold_issue_added = True
            issues.append(
                _issue(
                    "became-fail",
                    "FAIL",
                    previous,
                    current,
                    prev,
                    curr,
                    score_drop=score_drop,
                    reason="\u4e0a\u4e00\u9636\u6bb5\u672a\u8fbe\u5931\u8d25\u9608\u503c\uff0c\u5f53\u524d\u9636\u6bb5\u964d\u5230 0.20 \u4ee5\u4e0b\u3002",
                )
            )
        elif previous_score >= 0.45 and current_score < 0.45:
            became_low_count += 1
            threshold_issue_added = True
            issues.append(
                _issue(
                    "became-low",
                    "FAIL" if current_score < 0.20 else "WARN",
                    previous,
                    current,
                    prev,
                    curr,
                    score_drop=score_drop,
                    reason="\u4e0a\u4e00\u9636\u6bb5\u4ecd\u53ef\u7528\uff0c\u5f53\u524d\u9636\u6bb5\u964d\u5230 0.45 \u4ee5\u4e0b\u3002",
                )
            )
        elif score_drop >= score_drop_threshold:
            threshold_issue_added = True
            issues.append(
                _issue(
                    "score-drop",
                    "WARN",
                    previous,
                    current,
                    prev,
                    curr,
                    score_drop=score_drop,
                    reason=f"\u5339\u914d\u5206\u6570\u4e0b\u964d {score_drop}\uff0c\u8d85\u8fc7\u9608\u503c {score_drop_threshold}\u3002",
                )
            )
        if (
            not threshold_issue_added
            and LEVEL_RANK.get(curr_level, 0) > LEVEL_RANK.get(prev_level, 0)
            and score_drop > 0
        ):
            issues.append(
                _issue(
                    "level-worse",
                    "FAIL" if curr_level == "fail" else "WARN",
                    previous,
                    current,
                    prev,
                    curr,
                    score_drop=score_drop,
                    reason=f"\u884c\u7ea7\u522b\u4ece {prev_level or '-'} \u53d8\u4e3a {curr_level or '-'}\u3002",
                )
            )
        if tracked_added:
            if any(item.startswith("short-dialogue") for item in tracked_added):
                added_short_dialogue_issue_count += 1
            severity = "FAIL" if "short-dialogue-missing" in tracked_added else "WARN"
            issues.append(
                _issue(
                    "diagnostic-added",
                    severity,
                    previous,
                    current,
                    prev,
                    curr,
                    score_drop=score_drop,
                    reason="\u5f53\u524d\u9636\u6bb5\u65b0\u589e\u8bca\u65ad\uff1a" + ", ".join(tracked_added),
                )
            )
        previous_length = int(prev.get("matched_normalized_chars", 0) or 0)
        current_length = int(curr.get("matched_normalized_chars", 0) or 0)
        if previous_length > 0 and current_length < previous_length * length_drop_ratio and score_drop > 0:
            issues.append(
                _issue(
                    "matched-text-shortened",
                    "WARN",
                    previous,
                    current,
                    prev,
                    curr,
                    score_drop=score_drop,
                    reason=(
                        f"\u5339\u914d\u6587\u672c\u957f\u5ea6\u4ece {previous_length} \u964d\u5230 {current_length}\uff0c"
                        f"\u4f4e\u4e8e\u4e0a\u4e00\u9636\u6bb5\u7684 {length_drop_ratio:.0%}\u3002"
                    ),
                )
            )
    return {
        "summary": {
            "from": previous["label"],
            "to": current["label"],
            "shared_row_count": len(shared_indexes),
            "issue_count": len(issues),
            "mean_score_delta": round(sum(score_deltas) / len(score_deltas), 6) if score_deltas else 0.0,
            "became_fail_count": became_fail_count,
            "became_low_count": became_low_count,
            "added_short_dialogue_issue_count": added_short_dialogue_issue_count,
        },
        "issues": issues,
    }


def _issue(
    issue_type: str,
    severity: str,
    previous_stage: dict[str, Any],
    current_stage: dict[str, Any],
    previous_row: dict[str, Any],
    current_row: dict[str, Any],
    *,
    score_drop: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "type": issue_type,
        "severity": severity,
        "transition": f"{previous_stage['label']} -> {current_stage['label']}",
        "previous_label": previous_stage["label"],
        "current_label": current_stage["label"],
        "index": current_row.get("index"),
        "ass_start_ms": current_row.get("ass_start_ms"),
        "ass_end_ms": current_row.get("ass_end_ms"),
        "target_start_ms": current_row.get("target_start_ms", current_row.get("ass_start_ms")),
        "target_end_ms": current_row.get("target_end_ms", current_row.get("ass_end_ms")),
        "ass_text": current_row.get("ass_text"),
        "previous_score": previous_row.get("score"),
        "current_score": current_row.get("score"),
        "score_drop": score_drop,
        "previous_level": previous_row.get("level"),
        "current_level": current_row.get("level"),
        "previous_diagnostics": previous_row.get("diagnostics", []),
        "current_diagnostics": current_row.get("diagnostics", []),
        "previous_matched_text": previous_row.get("matched_text", ""),
        "current_matched_text": current_row.get("matched_text", ""),
        "previous_matched_start_ms": previous_row.get("matched_start_ms"),
        "previous_matched_end_ms": previous_row.get("matched_end_ms"),
        "current_matched_start_ms": current_row.get("matched_start_ms"),
        "current_matched_end_ms": current_row.get("matched_end_ms"),
        "reason": reason,
    }


def _zh_status(status: str) -> str:
    return {
        "PASS": "\u901a\u8fc7",
        "WARN": "\u8b66\u544a",
        "FAIL": "\u5931\u8d25",
    }.get(status, status)
