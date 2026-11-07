from __future__ import annotations

from typing import Any

from qwen_asr.final_quality_common import fail, normalize_status, skip
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


def ass_quality_checks(work_paths: WorkPaths) -> list[dict[str, Any]]:
    reports_dir = work_paths.workdir / "reports"
    if not reports_dir.exists():
        return [skip("ass_quality", "没有发现 ASS 质量报告")]
    paths = [
        path
        for path in sorted(reports_dir.glob("ass_quality*.json"))
        if not path.name.endswith(".quality_suspects.json")
    ]
    if not paths:
        return [skip("ass_quality", "没有发现 ASS 质量报告")]
    checks: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path, default={})
        if not isinstance(payload, dict):
            checks.append(fail("ass_quality", f"ASS 质量报告无法读取：{path}", report=str(path)))
            continue
        status = normalize_status(payload.get("status", "WARN"))
        summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        checks.append(
            {
                "name": "ass_quality",
                "status": status,
                "message": (
                    f"ASS 质量报告 {path.name} 为 {status}"
                    f"，低分 {summary.get('score_lt_045', '')}"
                    f"，失败 {summary.get('score_lt_020', '')}"
                ),
                "report": str(path),
            }
        )
    return checks
