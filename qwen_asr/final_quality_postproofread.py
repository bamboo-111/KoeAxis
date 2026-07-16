from __future__ import annotations

from typing import Any

from qwen_asr.content_quality import normalize_japanese
from qwen_asr.final_quality_common import fail, float_or_none, passed, skip
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json


POST_PROOFREAD_MIN_RETENTION = 0.60
SHORT_RESPONSE_GUARD_TEXTS = ("はい", "え", "え?", "うん", "いいえ", "駄目", "だめ")


def post_proofread_guard_check(work_paths: WorkPaths) -> dict[str, Any]:
    if not work_paths.mimo_proofread_manifest.exists():
        return skip("post_proofread_guard", "未运行 MiMo proofread，跳过复核后内容/ASS 守卫")
    proofread = read_json(work_paths.mimo_proofread_manifest, default={})
    if not isinstance(proofread, dict) or not proofread:
        return fail("post_proofread_guard", "MiMo proofread manifest 缺失或格式无效")

    issues: list[dict[str, Any]] = []
    checked_changes = 0
    for subtitle_id, item in proofread.items():
        if not isinstance(item, dict):
            continue
        history = item.get("proofread_history", [])
        if not isinstance(history, list):
            continue
        for entry in history:
            if not is_mimo_original_change(entry):
                continue
            checked_changes += 1
            issue = post_proofread_original_change_issue(str(subtitle_id), entry)
            if issue:
                issues.append(issue)

    if issues:
        return fail(
            "post_proofread_guard",
            f"proofread 后内容/ASS 守卫失败：{len(issues)} 条",
            checked_change_count=checked_changes,
            issue_count=len(issues),
            issues=issues[:20],
        )
    return passed(
        "post_proofread_guard",
        f"proofread 后内容/ASS 守卫通过：检查 {checked_changes} 条原文修改",
        checked_change_count=checked_changes,
    )


def is_mimo_original_change(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    if not str(entry.get("source", "")).startswith("mimo-"):
        return False
    changes = entry.get("changes", {})
    return isinstance(changes, dict) and isinstance(changes.get("original_subtitle"), dict)


def post_proofread_original_change_issue(subtitle_id: str, entry: dict[str, Any]) -> dict[str, Any] | None:
    changes = entry.get("changes", {})
    change = changes.get("original_subtitle", {}) if isinstance(changes, dict) else {}
    before = str(change.get("before", "") if isinstance(change, dict) else "")
    after = str(change.get("after", "") if isinstance(change, dict) else "")
    evidence = entry.get("evidence", {})
    if not isinstance(evidence, dict):
        return post_guard_issue(subtitle_id, "missing_evidence", "原文修改缺少 proofread evidence")
    ass_guard = evidence.get("ass_guard")
    if not isinstance(ass_guard, dict) or not ass_guard:
        return post_guard_issue(subtitle_id, "missing_ass_guard", "原文修改缺少 ASS guard 证据")
    if not bool(ass_guard.get("accepted")):
        return post_guard_issue(
            subtitle_id,
            "ass_guard_rejected_but_applied",
            "ASS guard 未接受的原文修改被应用",
            reason=str(ass_guard.get("reason", "")),
        )
    current_score = float_or_none(ass_guard.get("current_score"))
    suggested_score = float_or_none(ass_guard.get("suggested_score"))
    if current_score is not None and suggested_score is not None and suggested_score < current_score:
        return post_guard_issue(
            subtitle_id,
            "ass_score_regression",
            "原文修改后 ASS 局部分数下降",
            current_score=current_score,
            suggested_score=suggested_score,
        )
    if post_proofread_content_regressed(before, after) and str(ass_guard.get("reason", "")) not in {
        "ass-improved",
        "ass-high-score",
    }:
        return post_guard_issue(
            subtitle_id,
            "content_regression_without_ass_support",
            "原文修改导致内容大幅缩短或短应答消失，且缺少 ASS 支撑",
            before_chars=len(normalize_japanese(before)),
            after_chars=len(normalize_japanese(after)),
            ass_reason=str(ass_guard.get("reason", "")),
        )
    return None


def post_proofread_content_regressed(before: str, after: str) -> bool:
    before_norm = normalize_japanese(before)
    after_norm = normalize_japanese(after)
    if before_norm and len(after_norm) / max(1, len(before_norm)) < POST_PROOFREAD_MIN_RETENTION:
        return True
    for text in SHORT_RESPONSE_GUARD_TEXTS:
        normalized = normalize_japanese(text)
        if normalized and normalized in before_norm and normalized not in after_norm:
            return True
    return False


def post_guard_issue(subtitle_id: str, issue_type: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"subtitle_id": subtitle_id, "type": issue_type, "message": message, **extra}
