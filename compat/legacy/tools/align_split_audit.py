from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_asr.ass_quality import SubtitleCue, load_source_cues, normalize_for_match
from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT
from qwen_asr.storage import ensure_directory, read_json, write_json_atomic

AUDIT_TYPES = {
    "became-fail",
    "short-dialogue-missing",
    "score-drop",
    "matched-text-shortened",
    "short-dialogue-timing-shifted",
    "introduced-duplicate",
}


def cmd_align_split_audit(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report = build_align_split_audit_report(
        work_paths,
        dataset=str(getattr(args, "dataset", "") or work_paths.workdir.name),
        aligned_ass_report=Path(getattr(args, "aligned_ass_report", "")),
        split_ass_report=Path(getattr(args, "split_ass_report", "")),
        diff_report=Path(getattr(args, "diff_report", "")),
        max_score_drop=int(getattr(args, "max_score_drop", 20)),
        max_shortened=int(getattr(args, "max_shortened", 20)),
        audit_split_mode=str(getattr(args, "audit_split_mode", "unknown") or "unknown"),
    )
    output = Path(getattr(args, "output", "") or work_paths.workdir / "reports" / "align_split_audit.json")
    write_json_atomic(output, report)
    markdown_output = str(getattr(args, "markdown_output", "") or "").strip()
    if markdown_output:
        path = Path(markdown_output)
        ensure_directory(path.parent)
        path.write_text(render_markdown_report(report), encoding="utf-8")
    print(f"align -> split \u5ba1\u8ba1\u62a5\u544a\uff1a{output}")
    return 0


def build_align_split_audit_report(
    work_paths: WorkPaths,
    *,
    dataset: str,
    aligned_ass_report: Path,
    split_ass_report: Path,
    diff_report: Path,
    max_score_drop: int = 20,
    max_shortened: int = 20,
    audit_split_mode: str = "unknown",
) -> dict[str, Any]:
    aligned_report = _load_report(aligned_ass_report)
    split_report = _load_report(split_ass_report)
    diff = _load_report(diff_report)
    aligned_rows = _rows_by_index(aligned_report)
    split_rows = _rows_by_index(split_report)
    aligned_segments = _load_aligned_segments(work_paths)
    aligned_cues = load_source_cues(work_paths, source="aligned", optimizer_root=DEFAULT_OPTIMIZER_ROOT)
    split_cues = load_source_cues(work_paths, source="split", optimizer_root=DEFAULT_OPTIMIZER_ROOT)

    selected = _select_issues(
        diff.get("issues", []),
        split_rows,
        max_score_drop=max_score_drop,
        max_shortened=max_shortened,
    )
    cases = [
        _audit_case(
            dataset=dataset,
            issue=issue,
            aligned_row=aligned_rows.get(int(issue["index"])),
            split_row=split_rows.get(int(issue["index"])),
            aligned_cues=aligned_cues,
            split_cues=split_cues,
            aligned_segments=aligned_segments,
            audit_split_mode=audit_split_mode,
        )
        for issue in selected
    ]
    root_counts: dict[str, int] = {}
    stage_owner_counts: dict[str, int] = {}
    root_detail_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for case in cases:
        root_counts[case["root_cause"]] = root_counts.get(case["root_cause"], 0) + 1
        stage_owner_counts[case["stage_owner"]] = stage_owner_counts.get(case["stage_owner"], 0) + 1
        root_detail_counts[case["root_cause_detail"]] = root_detail_counts.get(case["root_cause_detail"], 0) + 1
        type_counts[case["type"]] = type_counts.get(case["type"], 0) + 1
    return {
        "dataset": dataset,
        "status": "PASS",
        "inputs": {
            "workdir": str(work_paths.workdir),
            "aligned_ass_report": str(aligned_ass_report),
            "split_ass_report": str(split_ass_report),
            "diff_report": str(diff_report),
            "audit_split_mode": audit_split_mode,
        },
        "selection": {
            "case_count": len(cases),
            "max_score_drop": max_score_drop,
            "max_shortened": max_shortened,
            "all_became_fail_included": _count_diff_type(diff, "became-fail") == type_counts.get("became-fail", 0),
            "all_short_dialogue_missing_included": _count_split_diagnostic(split_rows, "short-dialogue-missing")
            == type_counts.get("short-dialogue-missing", 0),
        },
        "summary": {
            "type_counts": type_counts,
            "root_cause_counts": root_counts,
            "stage_owner_counts": stage_owner_counts,
            "root_cause_detail_counts": root_detail_counts,
            "aligned_summary": aligned_report.get("summary", {}),
            "split_summary": split_report.get("summary", {}),
        },
        "cases": cases,
    }


def _select_issues(
    diff_issues: list[Any],
    split_rows: dict[int, dict[str, Any]],
    *,
    max_score_drop: int,
    max_shortened: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()

    def add(issue: dict[str, Any], issue_type: str | None = None) -> None:
        index = int(issue.get("index", 0) or 0)
        kind = str(issue_type or issue.get("type", ""))
        if not index or not kind or (index, kind) in seen:
            return
        payload = dict(issue)
        payload["type"] = kind
        selected.append(payload)
        seen.add((index, kind))

    typed = [item for item in diff_issues if isinstance(item, dict)]
    for item in typed:
        if item.get("type") == "became-fail":
            add(item)
    for index, row in sorted(split_rows.items()):
        diagnostics = set(row.get("diagnostics") or [])
        if "short-dialogue-missing" in diagnostics:
            add(_issue_from_split_row(index, row), "short-dialogue-missing")
        if "short-dialogue-timing-shifted" in diagnostics:
            add(_issue_from_split_row(index, row), "short-dialogue-timing-shifted")
    for item in [i for i in typed if i.get("type") == "introduced-duplicate"]:
        add(item)
    score_drop_items = sorted(
        [i for i in typed if i.get("type") == "score-drop"],
        key=lambda item: float(item.get("score_drop", 0.0) or 0.0),
        reverse=True,
    )[:max_score_drop]
    for item in score_drop_items:
        add(item)
    shortened_items = sorted(
        [i for i in typed if i.get("type") == "matched-text-shortened"],
        key=lambda item: float(item.get("score_drop", 0.0) or 0.0),
        reverse=True,
    )[:max_shortened]
    for item in shortened_items:
        add(item)
    return selected


def _issue_from_split_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": index,
        "ass_start_ms": row.get("ass_start_ms", 0),
        "ass_end_ms": row.get("ass_end_ms", 0),
        "target_start_ms": row.get("target_start_ms", 0),
        "target_end_ms": row.get("target_end_ms", 0),
        "ass_text": row.get("ass_text", ""),
        "current_score": row.get("score", 0.0),
        "current_matched_text": row.get("matched_text", ""),
        "current_matched_start_ms": row.get("matched_start_ms", 0),
        "current_matched_end_ms": row.get("matched_end_ms", 0),
        "current_diagnostics": row.get("diagnostics", []),
        "severity": "FAIL" if row.get("level") == "fail" else "WARN",
    }


def _audit_case(
    *,
    dataset: str,
    issue: dict[str, Any],
    aligned_row: dict[str, Any] | None,
    split_row: dict[str, Any] | None,
    aligned_cues: list[SubtitleCue],
    split_cues: list[SubtitleCue],
    aligned_segments: list[dict[str, Any]],
    audit_split_mode: str,
) -> dict[str, Any]:
    target_start = int(issue.get("target_start_ms") or issue.get("ass_start_ms") or 0)
    target_end = int(issue.get("target_end_ms") or issue.get("ass_end_ms") or target_start)
    aligned_local = _nearby_cues(aligned_cues, target_start, target_end)
    split_local = _nearby_cues(split_cues, target_start, target_end)
    token_metrics = _token_metrics(_nearby_aligned_segments(aligned_segments, target_start, target_end))
    classification = _classify_case(
        issue,
        aligned_row,
        split_row,
        aligned_local,
        split_local,
        token_metrics,
        audit_split_mode=audit_split_mode,
    )
    return {
        "dataset": dataset,
        "index": int(issue.get("index", 0) or 0),
        "type": str(issue.get("type", "")),
        "severity": str(issue.get("severity", "WARN")),
        "ass_time": [int(issue.get("ass_start_ms", 0) or 0), int(issue.get("ass_end_ms", 0) or 0)],
        "target_time": [target_start, target_end],
        "ass_text": str(issue.get("ass_text", "")),
        "aligned_score": _score(aligned_row),
        "split_score": _score(split_row),
        "score_drop": round(max(0.0, _score(aligned_row) - _score(split_row)), 6),
        "aligned_matched_text": str((aligned_row or {}).get("matched_text", "")),
        "split_matched_text": str((split_row or {}).get("matched_text", "")),
        "aligned_nearby_text": "".join(cue.text for cue in aligned_local),
        "split_nearby_text": "".join(cue.text for cue in split_local),
        "aligned_diagnostics": list((aligned_row or {}).get("diagnostics") or []),
        "split_diagnostics": list((split_row or {}).get("diagnostics") or []),
        "token_metrics": token_metrics,
        "root_cause": classification["root_cause"],
        "root_cause_detail": classification["root_cause_detail"],
        "stage_owner": classification["stage_owner"],
        "classification_confidence": classification["classification_confidence"],
        "recommendation": classification["recommendation"],
        "evidence": classification["evidence"],
    }


def _classify_case(
    issue: dict[str, Any],
    aligned_row: dict[str, Any] | None,
    split_row: dict[str, Any] | None,
    aligned_local: list[SubtitleCue],
    split_local: list[SubtitleCue],
    token_metrics: dict[str, Any],
    *,
    audit_split_mode: str = "unknown",
) -> dict[str, str]:
    issue_type = str(issue.get("type", ""))
    ass_text = normalize_for_match(str(issue.get("ass_text", "")))
    aligned_score = _score(aligned_row)
    split_score = _score(split_row)
    aligned_text = normalize_for_match("".join(cue.text for cue in aligned_local))
    split_text = normalize_for_match("".join(cue.text for cue in split_local))
    aligned_has_ass = bool(ass_text and ass_text in aligned_text)
    split_has_ass = bool(ass_text and ass_text in split_text)
    split_mode = _normalize_split_mode(audit_split_mode)
    split_diagnostics = set((split_row or {}).get("diagnostics") or [])

    if token_metrics["non_monotonic_count"] or token_metrics["one_ms_token_count"] >= 3 or token_metrics["zero_duration_token_count"] >= 3:
        return _classification(
            root_cause="align token 结构异常",
            root_cause_detail="align-token-structure",
            stage_owner="align",
            classification_confidence="high",
            recommendation="先进入 align 参数实验，处理 0/1ms token 簇和非单调 token，再评估 split。",
            evidence="目标窗口附近存在异常 token 时间结构。",
        )
    if aligned_score < 0.20 and split_score < 0.20:
        return _classification(
            root_cause="align 时间或内容已失败",
            root_cause_detail="align-low-score-inherited",
            stage_owner="align",
            classification_confidence="high",
            recommendation="优先审计 forced alignment 输入文本、覆盖率和局部 fallback，不应先调 split。",
            evidence="aligned 阶段已经低于 0.20，split 只是继承失败。",
        )
    if issue_type == "matched-text-shortened" or (aligned_has_ass and not split_has_ass):
        if split_mode == "rule":
            return _classification(
                root_cause="split 规则或后处理内容缩短",
                root_cause_detail="rule-or-postprocess-content-loss",
                stage_owner="postprocess",
                classification_confidence="medium",
                recommendation="检查规则 split 输出和后处理合并/删除规则；当前审计样本未使用 LLM prompt。",
                evidence="aligned 附近仍能看到目标文本，但 rule split 后目标文本缺失或明显缩短。",
            )
        return _classification(
            root_cause="split 内容缩短但模式未知",
            root_cause_detail="split-mode-unknown-content-loss",
            stage_owner="unknown",
            classification_confidence="low",
            recommendation="补充 audit split mode 或 LLM 原始返回记录后再区分 prompt 与后处理。",
            evidence="aligned 附近仍能看到目标文本，但 split 附近缺失或明显缩短。",
        )
    if issue_type in {"became-fail", "score-drop"} and aligned_score >= 0.45 and split_score < aligned_score:
        stage_owner = "rule" if split_mode == "rule" else "unknown"
        detail = "rule-boundary-error" if split_mode == "rule" else "split-mode-unknown-boundary-error"
        return _classification(
            root_cause="split 时间边界或切分点错误",
            root_cause_detail=detail,
            stage_owner=stage_owner,
            classification_confidence="medium" if split_mode != "unknown" else "low",
            recommendation="比较 split 前后局部时间边界，保护短句并限制跨句合并。",
            evidence="aligned 分数可用，split 后分数下降。",
        )
    if issue_type.startswith("short-dialogue") and ass_text and ass_text in normalize_for_match("".join(cue.text for cue in split_local)):
        return _classification(
            root_cause="ASS 评估窗口或短句时移",
            root_cause_detail="evaluator-window-or-timing-shift",
            stage_owner="evaluator",
            classification_confidence="medium",
            recommendation="扩大短句诊断窗口并审计偏移；若文本存在但远离目标时间，应归入时间问题。",
            evidence="短对白文本未消失，但未落入目标窗口。",
        )
    if issue_type.startswith("short-dialogue"):
        stage_owner = "rule" if split_mode == "rule" else "unknown"
        detail = "rule-short-dialogue-merged" if split_mode == "rule" else "split-mode-unknown-short-dialogue-merged"
        return _classification(
            root_cause="短应答被吞并",
            root_cause_detail=detail,
            stage_owner=stage_owner,
            classification_confidence="medium" if split_mode != "unknown" else "low",
            recommendation="在规则 split 和后处理中保护短应答，禁止合并后不可定位。",
            evidence="目标短对白在 split 局部文本中不可见。",
        )
    if split_diagnostics & {"introduced-duplicate", "duplicate-nearby"}:
        return _classification(
            root_cause="split 后处理制造相邻重复",
            root_cause_detail="postprocess-duplicate-boundary",
            stage_owner="postprocess",
            classification_confidence="medium",
            recommendation="审计后处理边界合并和相邻去重规则。",
            evidence="split ASS 诊断包含相邻重复相关标记。",
        )
    return _classification(
        root_cause="需要人工复核的评估差异",
        root_cause_detail="manual-review-required",
        stage_owner="unknown",
        classification_confidence="low",
        recommendation="保留为审计样本，结合音频和上下文确认是标准 ASS 差异还是真实流程问题。",
        evidence="现有结构化证据不足以自动唯一归因。",
    )


def _classification(
    *,
    root_cause: str,
    root_cause_detail: str,
    stage_owner: str,
    classification_confidence: str,
    recommendation: str,
    evidence: str,
) -> dict[str, str]:
    return {
        "root_cause": root_cause,
        "root_cause_detail": root_cause_detail,
        "stage_owner": stage_owner,
        "classification_confidence": classification_confidence,
        "recommendation": recommendation,
        "evidence": evidence,
    }


def _normalize_split_mode(value: str) -> str:
    mode = str(value or "unknown").strip().lower()
    if mode == "rule":
        return mode
    return "unknown"


def _nearby_cues(cues: list[SubtitleCue], start_ms: int, end_ms: int, margin_ms: int = 1200) -> list[SubtitleCue]:
    left = start_ms - margin_ms
    right = end_ms + margin_ms
    return [cue for cue in cues if cue.end_ms >= left and cue.start_ms <= right]


def _nearby_aligned_segments(items: list[dict[str, Any]], start_ms: int, end_ms: int, margin_ms: int = 1200) -> list[dict[str, Any]]:
    left = (start_ms - margin_ms) / 1000.0
    right = (end_ms + margin_ms) / 1000.0
    return [
        item for item in items
        if float(item.get("global_end_time", 0.0) or 0.0) >= left and float(item.get("global_start_time", 0.0) or 0.0) <= right
    ]


def _token_metrics(segments: list[dict[str, Any]]) -> dict[str, Any]:
    zero = one_ms = non_monotonic = long_span = token_count = 0
    previous_start: float | None = None
    for segment in segments:
        seg_start = float(segment.get("global_start_time", 0.0) or 0.0)
        seg_end = float(segment.get("global_end_time", 0.0) or 0.0)
        if seg_end - seg_start >= 12.0:
            long_span += 1
        for token in segment.get("tokens", []) or []:
            if not isinstance(token, dict):
                continue
            token_count += 1
            start = float(token.get("start_time", 0.0) or 0.0)
            end = float(token.get("end_time", 0.0) or 0.0)
            duration_ms = int(round((end - start) * 1000))
            if duration_ms <= 0:
                zero += 1
            if duration_ms == 1:
                one_ms += 1
            if previous_start is not None and start < previous_start:
                non_monotonic += 1
            previous_start = start
    return {
        "segment_count": len(segments),
        "token_count": token_count,
        "zero_duration_token_count": zero,
        "one_ms_token_count": one_ms,
        "non_monotonic_count": non_monotonic,
        "long_span_segment_count": long_span,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# align -> split 根因审计报告",
        "",
        f"- 数据集：{report['dataset']}",
        f"- 审计样本数：{report['selection']['case_count']}",
        f"- became-fail 全量纳入：{report['selection']['all_became_fail_included']}",
        f"- short-dialogue-missing 全量纳入：{report['selection']['all_short_dialogue_missing_included']}",
        "",
        "## 根因统计",
        "",
    ]
    for name, count in sorted(report["summary"]["root_cause_counts"].items()):
        lines.append(f"- {name}：{count}")
    lines.extend(["", "## 阶段归因统计", ""])
    for name, count in sorted(report["summary"].get("stage_owner_counts", {}).items()):
        lines.append(f"- {name}：{count}")
    lines.extend(["", "## 样本明细", ""])
    for case in report["cases"]:
        lines.extend(
            [
                f"### ASS {case['index']} / {case['type']}",
                "",
                f"- 根因：{case['root_cause']}",
                f"- 归因阶段：{case.get('stage_owner', 'unknown')} / {case.get('root_cause_detail', '')} / {case.get('classification_confidence', '')}",
                f"- 建议阶段：{case['recommendation']}",
                f"- ASS：{case['ass_text']}",
                f"- 时间：{case['target_time'][0]} - {case['target_time'][1]}",
                f"- 分数：aligned {case['aligned_score']} -> split {case['split_score']}",
                f"- 证据：{case['evidence']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _load_report(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid report: {path}")
    return payload


def _rows_by_index(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["index"]): row for row in report.get("rows", []) if isinstance(row, dict) and "index" in row}


def _load_aligned_segments(work_paths: WorkPaths) -> list[dict[str, Any]]:
    payload = read_json(work_paths.aligned_manifest, default=[])
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _score(row: dict[str, Any] | None) -> float:
    if not row:
        return 0.0
    return float(row.get("score", 0.0) or 0.0)


def _count_diff_type(diff: dict[str, Any], kind: str) -> int:
    return sum(1 for item in diff.get("issues", []) if isinstance(item, dict) and item.get("type") == kind)


def _count_split_diagnostic(rows: dict[int, dict[str, Any]], diagnostic: str) -> int:
    return sum(1 for row in rows.values() if diagnostic in set(row.get("diagnostics") or []))


def dumps_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
