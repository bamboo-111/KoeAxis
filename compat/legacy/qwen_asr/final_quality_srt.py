from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from qwen_asr import final_quality_common as _common
from qwen_asr.models import WorkPaths

_pass = _common.passed
_warn = _common.warn
_fail = _common.fail
_skip = _common.skip

SRT_TIMESTAMP_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2}),(?P<ms>\d{3})$")


def srt_legality_check(work_paths: WorkPaths, *, require_srt: bool) -> dict[str, Any]:
    if not work_paths.subtitles_srt.exists():
        if require_srt:
            return _fail("srt_legality", "要求生成 SRT，但未找到导出的 SRT")
        return _skip("srt_legality", "未生成 SRT，跳过 SRT 合法性检查")
    issues = validate_srt(work_paths.subtitles_srt)
    fail_count = sum(item["severity"] == "FAIL" for item in issues)
    warn_count = sum(item["severity"] == "WARN" for item in issues)
    if fail_count:
        return _fail(
            "srt_legality",
            f"SRT 合法性失败：{fail_count} 个失败，{warn_count} 个警告",
            issues=issues[:50],
        )
    status = "WARN" if warn_count else "PASS"
    return {
        "name": "srt_legality",
        "status": status,
        "message": f"SRT 合法性 {status}：{warn_count} 个警告",
        "issues": issues[:50],
    }


def validate_srt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    blocks = [block for block in re.split(r"\r?\n\s*\r?\n", text.strip()) if block.strip()]
    issues: list[dict[str, Any]] = []
    previous_end: int | None = None
    expected_index = 1
    for block_number, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 3:
            issues.append(srt_issue("FAIL", block_number, "block_too_short", "字幕块行数不足"))
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            issues.append(srt_issue("FAIL", block_number, "bad_index", "字幕序号不是整数"))
            continue
        if index != expected_index:
            issues.append(
                srt_issue(
                    "FAIL", block_number, "non_continuous_index", f"字幕序号应为 {expected_index}，实际为 {index}"
                )
            )
        expected_index += 1
        if " --> " not in lines[1]:
            issues.append(srt_issue("FAIL", index, "bad_time_separator", "时间轴缺少标准分隔符"))
            continue
        start_raw, end_raw = lines[1].split(" --> ", 1)
        start_ms = parse_srt_timestamp(start_raw)
        end_ms = parse_srt_timestamp(end_raw)
        if start_ms is None or end_ms is None:
            issues.append(srt_issue("FAIL", index, "bad_timestamp", "时间戳格式非法"))
            continue
        if start_ms < 0 or end_ms < 0:
            issues.append(srt_issue("FAIL", index, "negative_time", "时间戳为负数"))
        if end_ms <= start_ms:
            issues.append(srt_issue("FAIL", index, "non_positive_duration", "结束时间不晚于开始时间"))
        if previous_end is not None and start_ms < previous_end:
            overlap = previous_end - start_ms
            severity = "FAIL" if overlap > 500 else "WARN"
            issues.append(srt_issue(severity, index, "overlap", f"与上一条重叠 {overlap}ms"))
        previous_end = end_ms
    if not blocks:
        issues.append(srt_issue("FAIL", 0, "empty_srt", "SRT 为空"))
    return issues


def parse_srt_timestamp(value: str) -> int | None:
    match = SRT_TIMESTAMP_RE.match(value.strip())
    if not match:
        return None
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    milliseconds = int(match.group("ms"))
    if minutes >= 60 or seconds >= 60:
        return None
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def srt_issue(severity: str, index: int, issue_type: str, message: str) -> dict[str, Any]:
    return {"severity": severity, "index": index, "type": issue_type, "message": message}
