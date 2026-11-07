from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_asr.storage import ensure_directory, read_json, write_json_atomic


def cmd_tuning_matrix(args: argparse.Namespace) -> int:
    matrix = build_tuning_matrix()
    ass_reports = [_load_labeled_report(value) for value in getattr(args, "ass_quality_report", [])]
    content_reports = [_load_labeled_report(value) for value in getattr(args, "content_quality_report", [])]
    proofread_realign_reports = [
        _load_labeled_report(value) for value in getattr(args, "proofread_realign_report", [])
    ]
    payload = {
        "status": _overall_status(ass_reports, content_reports, proofread_realign_reports),
        "purpose": "\u9010\u9636\u6bb5\u8c03\u53c2\u5bf9\u6bd4\u548c\u9a8c\u6536\u6807\u51c6",
        "stage_count": len(matrix),
        "stages": matrix,
        "current_baselines": {
            "ass_quality": ass_reports,
            "content_quality": content_reports,
            "proofread_realign": proofread_realign_reports,
        },
        "acceptance_rule": {
            "hard_rule": "\u4efb\u4e00\u786c\u95e8\u5931\u8d25\u5219\u4e0d\u5f97\u628a\u8be5\u7ec4\u5408\u6807\u4e3a\u6b63\u5f0f\u6210\u529f",
            "comparison_rule": "\u4e24\u4e2a\u53ef\u9760 ASS \u96c6\u7684\u4f4e\u5206\u9879\u4e0d\u5f97\u540c\u65f6\u9000\u5316\uff0c\u4e14\u5185\u5bb9\u5b88\u6052\u95e8\u5fc5\u987b\u901a\u8fc7",
        },
    }
    output = Path(getattr(args, "output", "") or "tuning_matrix.json")
    write_json_atomic(output, payload)
    markdown_output = str(getattr(args, "markdown_output", "") or "").strip()
    if markdown_output:
        path = Path(markdown_output)
        ensure_directory(path.parent)
        path.write_text(render_tuning_markdown(payload), encoding="utf-8")
    print(f"\u8c03\u53c2\u77e9\u9635\u5df2\u5199\u5165\uff1a{output}")
    return 0


def build_tuning_matrix() -> list[dict[str, Any]]:
    return [
        _stage(
            "prepare_vad",
            "\u5207\u7247\u4e0e VAD",
            ["--max-segment-seconds", "--preferred-silence-ms", "--min-silence-ms", "--padding-ms", "--overlap-ms", "--vad-threshold"],
            [
                "\u57fa\u7ebf\uff1a15s / preferred 800ms / min 500ms / padding 300ms / overlap 0ms",
                "\u8fb9\u754c\u589e\u5f3a\uff1apadding 500ms\uff0coverlap 300ms",
                "\u66f4\u7d27 VAD\uff1amin silence 300ms\uff0cpreferred 500ms",
            ],
            [
                "\u4e0d\u589e\u52a0\u660e\u663e\u8d1f\u65f6\u95f4\u6216\u91cd\u53e0",
                "\u5185\u5bb9\u5b88\u6052 transcript->align \u4e0d\u51fa FAIL",
                "ASS \u8bc4\u4f30 score_lt_020 \u4e0d\u9ad8\u4e8e\u57fa\u7ebf",
            ],
        ),
        _stage(
            "transcribe",
            "15s \u65e5\u8bed ASR",
            ["--language ja", "--max-segment-seconds", "--batch-mode", "--target-batch-audio-seconds"],
            [
                "15s \u4e3b\u8def\u5f84",
                "20s \u98ce\u9669\u8fb9\u754c\u5bf9\u7167",
                "30s \u4e0a\u4e0b\u6587\u5bf9\u7167",
            ],
            [
                "\u663e\u5f0f\u6307\u5b9a\u65e5\u8bed",
                "\u4e24\u4e2a ASS \u96c6 score_lt_020 \u4e0d\u9ad8\u4e8e\u57fa\u7ebf",
                "\u77ed\u5e94\u7b54\u5728 transcript \u4e2d\u53ef\u68c0\u51fa",
            ],
        ),
        _stage(
            "correct",
            "\u786e\u5b9a\u6027 correct",
            ["--with-correct", "--no-with-correct"],
            ["\u542f\u7528\u786e\u5b9a\u6027\u6e05\u7406", "\u8df3\u8fc7 correct \u76f4\u63a5 align"],
            [
                "\u4e0d\u5f97\u5728\u65e0\u97f3\u9891\u8bc1\u636e\u65f6\u6539\u5199\u8bed\u4e49",
                "\u89c4\u8303\u5316\u65e5\u6587\u5185\u5bb9\u4fdd\u7559\u7387 >= 0.995",
                "\u4e0d\u65b0\u589e ASS score_lt_020",
            ],
        ),
        _stage(
            "align",
            "\u5bf9\u9f50\u4e0e\u5c40\u90e8 token \u63d2\u503c",
            [
                "--align-fallback",
                "--align-fallback-window-seconds",
                "align_local_interpolation_max_gap_ms",
                "align_zero_token_default_duration_ms",
                "align_zero_token_max_duration_ms",
            ],
            [
                "\u5c40\u90e8\u63d2\u503c 800ms / zero 160ms / max 500ms",
                "\u5c40\u90e8\u63d2\u503c 500ms / zero 120ms / max 400ms",
                "\u98ce\u9669\u6bb5\u542f\u7528 asr-short-window",
            ],
            [
                "\u4e0d\u51fa\u73b0 1ms token \u7c07",
                "\u65f6\u95f4\u5355\u8c03\uff0c\u65e0\u5f02\u5e38\u8d85\u9ad8 cps",
                "\u56de\u9000\u6587\u672c\u957f\u5ea6\u4e0d\u5f97\u660e\u663e\u77ed\u4e8e\u539f transcript",
            ],
        ),
        _stage(
            "split",
            "\u65ad\u53e5",
            ["--max-word-count-cjk", "--max-word-count-english"],
            ["rule"],
            [
                "\u5185\u5bb9\u5b88\u6052 align->split \u5fc5\u987b PASS",
                "\u306f\u3044 / \u3048 / \u99c4\u76ee \u7b49\u77ed\u53e5\u4e0d\u5f97\u88ab\u541e",
                "ASS short_dialogue_low_score \u4e0d\u5f97\u9ad8\u4e8e\u57fa\u7ebf",
                "ASS overlong_match \u4e0d\u5f97\u9ad8\u4e8e\u57fa\u7ebf",
                "\u4e0d\u65b0\u589e\u76f8\u90bb\u8fb9\u754c\u91cd\u590d WARN \u8d85\u8fc7\u57fa\u7ebf",
            ],
        ),
        _stage(
            "sentence_postprocess",
            "\u65ad\u53e5\u540e\u5904\u7406",
            ["split_fragment_max_duration_ms", "split_fragment_max_gap_ms", "split_min_readable_duration_ms"],
            ["200ms / 80ms / 500ms", "300ms / 120ms / 600ms"],
            [
                "\u4e0d\u53ef\u8bfb\u788e\u7247\u5e94\u5408\u5e76",
                "\u6709\u6548\u77ed\u53e5\u4e0d\u5f97\u56e0\u65f6\u957f\u77ed\u88ab\u5220\u9664",
                "SRT \u5e8f\u53f7\u548c\u65f6\u95f4\u5408\u6cd5",
            ],
        ),
        _stage(
            "translate_suspects",
            "\u7ffb\u8bd1\u5e76\u6807\u6ce8\u7591\u70b9",
            ["--batch-num", "--thread-num", "structured suspect schema"],
            ["\u7ed3\u6784\u5316\u8fd4\u56de", "\u517c\u5bb9\u65e7\u7eaf\u8bd1\u6587\u8fd4\u56de"],
            [
                "\u7ed3\u6784\u5316\u89e3\u6790\u6210\u529f\u7387 100%",
                "\u7591\u70b9\u5bc6\u5ea6\u5728\u53ef\u590d\u6838\u8303\u56f4\u5185",
                "\u4eba\u540d\u88ab\u8bd1\u6210\u666e\u901a\u540d\u8bcd\u5fc5\u987b\u8fdb\u7591\u70b9",
            ],
        ),
        _stage(
            "suspect_audio_review",
            "\u4ec5\u7591\u70b9\u97f3\u9891\u590d\u6838",
            ["--mimo-audio-review-scope suspects", "--mimo-nearby-padding-s", "--mimo-nearby-context-subtitles"],
            ["suspects \u9ed8\u8ba4", "all \u4ec5\u8bca\u65ad"],
            [
                "\u9ed8\u8ba4\u4e0d\u89e6\u53d1\u5168\u91cf\u590d\u6838",
                "\u8bb0\u5f55\u5019\u9009\u6570\u3001\u5019\u9009\u5bc6\u5ea6\u3001\u8017\u65f6\u3001\u786e\u8ba4\u9519\u8bef\u7387\u3001\u672a\u89e3\u51b3\u7387",
                "\u672a\u5b8c\u6210 checkpoint \u5fc5\u987b\u8ba9\u8d28\u91cf\u95e8 FAIL",
            ],
        ),
        _stage(
            "realign_after_review",
            "\u590d\u6838\u4fee\u6539\u540e\u91cd\u5bf9\u9f50",
            [
                "modified-entry realign",
                "--proofread-realign-fallback",
                "--proofread-realign-mfa-fallback",
                "--proofread-realign-mfa-min-content-score",
            ],
            [
                "\u53ea\u91cd\u5bf9\u9f50\u88ab\u4fee\u6539\u6761\u76ee",
                "\u5fc5\u8981\u65f6\u91cd\u8dd1\u76f8\u90bb\u5c0f\u7a97\u53e3",
                "Qwen \u5931\u8d25\u540e\u5c40\u90e8 MFA fallback",
                "MFA \u6587\u672c\u4e00\u81f4\u6027\u5206\u6570\u9608\u503c 0.70 / 0.85",
            ],
            [
                "\u4fee\u6539\u540e\u65f6\u95f4\u8986\u76d6\u5bf9\u5e94\u97f3\u9891",
                "\u4e0d\u7834\u574f\u672a\u4fee\u6539\u6761\u76ee",
                "MFA usable=true \u4e14\u6587\u672c\u4e00\u81f4\u6027\u5206\u6570\u8fbe\u6807\u624d\u53ef\u63a5\u53d7",
                "MFA unusable / rejected \u5fc5\u987b\u8fdb\u5165 WARN \u7edf\u8ba1",
                "\u590d\u6838\u540e ASS score_lt_020 \u5e94\u4e0b\u964d\u6216\u4e0d\u53d8",
            ],
        ),
        _stage(
            "quality_gate",
            "\u8d28\u91cf\u95e8",
            ["ass-quality", "content-quality", "SRT legality", "checkpoint completeness"],
            ["PASS", "WARN", "FAIL"],
            [
                "FAIL \u65f6\u4e0d\u5f97\u6807\u8bb0\u6b63\u5f0f\u5b8c\u6210",
                "\u4e24\u4e2a ASS \u96c6\u90fd\u5fc5\u987b\u4e0d\u9000\u5316",
                "\u6700\u7ec8\u9ad8\u7f6e\u4fe1\u5931\u8d25\u6570\u5411 <=5 \u6536\u655b",
                "\u77ed\u5bf9\u767d\u4f4e\u5206\u548c\u8fc7\u957f\u5339\u914d\u6c61\u67d3\u5fc5\u987b\u5355\u72ec\u5bf9\u6bd4",
            ],
        ),
    ]


def render_tuning_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# \u5b57\u5e55\u6d41\u7a0b\u9010\u9636\u6bb5\u8c03\u53c2\u77e9\u9635",
        "",
        f"- \u603b\u4f53\u72b6\u6001\uff1a{_zh_status(str(payload['status']))}",
        f"- \u9636\u6bb5\u6570\uff1a{payload['stage_count']}",
        "",
        "## \u5f53\u524d\u57fa\u7ebf",
        "",
    ]
    for report in payload["current_baselines"]["ass_quality"]:
        summary = report.get("summary", {})
        lines.append(
            f"- ASS {report['label']}\uff1a\u72b6\u6001 {_zh_status(str(report.get('status', '')))}"
            f"\uff0c\u504f\u79fb {report.get('offset_ms', '')} ms"
            f"\uff0c<0.20 {summary.get('score_lt_020', '')}"
            f"\uff0c<0.45 {summary.get('score_lt_045', '')}"
            f"\uff0c\u77ed\u5bf9\u767d\u4f4e\u5206 {summary.get('short_dialogue_low_score', '')}"
            f"\uff0c\u77ed\u5bf9\u767d\u7591\u4f3c\u9519\u65f6 {summary.get('short_dialogue_timing_shifted', '')}"
            f"\uff0c\u77ed\u5bf9\u767d\u7591\u4f3c\u7f3a\u5931 {summary.get('short_dialogue_missing', '')}"
            f"\uff0c\u8fc7\u957f\u5339\u914d {summary.get('overlong_match', '')}"
        )
    for report in payload["current_baselines"]["content_quality"]:
        summary = report.get("summary", {})
        lines.append(
            f"- \u5185\u5bb9\u5b88\u6052 {report['label']}\uff1a\u72b6\u6001 {_zh_status(str(report.get('status', '')))}"
            f"\uff0cFAIL {summary.get('fail_count', '')}"
            f"\uff0cWARN {summary.get('warn_count', '')}"
        )
    for report in payload["current_baselines"].get("proofread_realign", []):
        lines.append(
            f"- \u590d\u6838\u540e\u91cd\u5bf9\u9f50 {report['label']}\uff1a\u72b6\u6001 {_zh_status(str(report.get('status', '')))}"
            f"\uff0c\u5f85\u5904\u7406 {_int_report_value(report, 'pending_count')}"
            f"\uff0c\u964d\u7ea7 {_int_report_value(report, 'fallback_count')}"
            f"\uff0cMFA \u6210\u529f {_int_report_value(report, 'mfa_completed_count')}"
            f"\uff0cMFA \u4e0d\u53ef\u7528 {_int_report_value(report, 'mfa_unusable_count')}"
            f"\uff0cMFA \u62d2\u7edd {_int_report_value(report, 'mfa_rejected_count')}"
        )
    lines.extend(["", "## \u8c03\u53c2\u9879\u548c\u901a\u8fc7\u6807\u51c6", ""])
    for index, stage in enumerate(payload["stages"], 1):
        lines.extend(
            [
                f"### {index}. {stage['name_zh']}",
                "",
                f"- \u9636\u6bb5 ID\uff1a{stage['id']}",
                f"- \u5f00\u5173\uff1a{', '.join(stage['switches'])}",
                f"- \u5019\u9009\uff1a{'; '.join(stage['variants'])}",
                "- \u901a\u8fc7\u6807\u51c6\uff1a" + "\uff1b".join(stage["pass_criteria"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _stage(
    stage_id: str,
    name_zh: str,
    switches: list[str],
    variants: list[str],
    pass_criteria: list[str],
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "name_zh": name_zh,
        "switches": switches,
        "variants": variants,
        "pass_criteria": pass_criteria,
    }


def _load_labeled_report(value: str) -> dict[str, Any]:
    label, path_text = _split_label_path(value)
    path = Path(path_text)
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        payload = {}
    result = dict(payload)
    result["label"] = label or path.stem
    result["path"] = str(path)
    return result


def _split_label_path(value: str) -> tuple[str, str]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), path.strip()
    return "", value.strip()


def _int_report_value(report: dict[str, Any], key: str) -> int:
    try:
        return int(report.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _overall_status(
    ass_reports: list[dict[str, Any]],
    content_reports: list[dict[str, Any]],
    proofread_realign_reports: list[dict[str, Any]],
) -> str:
    reports = ass_reports + content_reports + proofread_realign_reports
    if any(report.get("status") == "FAIL" for report in reports):
        return "FAIL"
    if any(report.get("status") == "WARN" for report in reports):
        return "WARN"
    return "PASS" if reports else "WARN"


def _zh_status(status: str) -> str:
    return {
        "PASS": "\u901a\u8fc7",
        "WARN": "\u8b66\u544a",
        "FAIL": "\u5931\u8d25",
    }.get(status, status)
