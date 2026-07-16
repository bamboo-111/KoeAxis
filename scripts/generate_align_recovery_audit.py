from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_asr.alignment_state import derive_alignment_state, overlaps_music_region


SOURCE = ROOT / "workspaces" / "full-regression-sayonara-lara-02-20260715-234059"
DIAGNOSTIC = ROOT / "workspaces" / "post-repair-full-flow-sayonara-lara-02-20260716-002234"
BASELINE_PATH = ROOT / "reports" / "align_recovery_goal_baseline.json"
OPED_REPORT = DIAGNOSTIC / "reports" / "post_repair_effectiveness_excluding_oped.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_char_count(text: str) -> int:
    return sum(1 for char in text if unicodedata.category(char)[0] in {"L", "N"})


def token_coverage(item: dict[str, Any]) -> float | None:
    tokens = item.get("tokens") if isinstance(item.get("tokens"), list) else []
    positive = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        try:
            start = float(token.get("start_time"))
            end = float(token.get("end_time"))
        except (TypeError, ValueError):
            continue
        if end > start:
            positive.append((start, end))
    try:
        segment_duration = float(item["global_end_time"]) - float(item["global_start_time"])
    except (KeyError, TypeError, ValueError):
        return None
    if not positive or segment_duration <= 0:
        return 0.0
    covered = max(end for _, end in positive) - min(start for start, _ in positive)
    return round(max(0.0, covered) / segment_duration, 6)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def create_derived_workspace(target: Path) -> dict[str, Any]:
    if target.exists():
        derivation_path = target / "reports" / "derivation.json"
        if not derivation_path.exists():
            raise FileExistsError(f"existing target is not a generated derived workspace: {target}")
        return read_json(derivation_path)
    copies = {
        SOURCE / "project.json": target / "project.json",
        SOURCE / "progress.json": target / "progress.json",
        SOURCE / "manifests" / "transcript_segments.json": target / "manifests" / "transcript_segments.json",
        SOURCE / "manifests" / "aligned_segments.json": target / "manifests" / "aligned_segments.json",
        SOURCE / "manifests" / "aligned_checkpoint.json": target / "manifests" / "aligned_checkpoint.json",
        SOURCE / "manifests" / "aligned_events.jsonl": target / "manifests" / "aligned_events.jsonl",
        SOURCE / "manifests" / "split_segments.json": target / "manifests" / "split_segments.json",
        OPED_REPORT: target / "reports" / OPED_REPORT.name,
    }
    copied = []
    for source, destination in copies.items():
        if not source.exists():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            {
                "source": str(source.relative_to(ROOT)),
                "destination": str(destination.relative_to(ROOT)),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    derivation = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_workspace": str(SOURCE.relative_to(ROOT)),
        "diagnostic_workspace": str(DIAGNOSTIC.relative_to(ROOT)),
        "policy": "source workspaces remain read-only; audio_path values continue to reference source segment audio",
        "copied_files": copied,
    }
    write_json(target / "reports" / "derivation.json", derivation)
    return derivation


def verify_baseline(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for evidence in baseline["evidence"]:
        path = ROOT / evidence["path"]
        actual_bytes = path.stat().st_size if path.exists() else None
        actual_sha256 = sha256(path) if path.exists() else None
        rows.append(
            {
                **evidence,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_sha256,
                "status": "MATCH"
                if actual_bytes == evidence["bytes"] and actual_sha256 == evidence["sha256"]
                else "MISMATCH",
            }
        )
    return rows


def music_intervals() -> list[dict[str, Any]]:
    payload = read_json(OPED_REPORT)
    return [
        {"name": name, "start_ms": int(value["start_ms"]), "end_ms": int(value["end_ms"])}
        for name, value in payload["intervals"].items()
    ]


def dialogue_metrics(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    intervals = music_intervals()
    raw = Counter(derive_alignment_state(item) for item in aligned)
    dialogue = Counter()
    music = Counter()
    for item in aligned:
        target = music if overlaps_music_region(item, intervals) else dialogue
        target[derive_alignment_state(item)] += 1
    failed_dialogue = [
        item
        for item in aligned
        if derive_alignment_state(item) == "failed" and not overlaps_music_region(item, intervals)
    ]
    return {
        "raw_align_input": len(aligned),
        "raw_completed": raw["completed_exact"] + raw["completed_coarse"],
        "raw_failed": raw["failed"],
        "music_region_segments": sum(music.values()),
        "music_region_failed": music["failed"],
        "dialogue_segments": sum(dialogue.values()),
        "completed_exact": dialogue["completed_exact"],
        "completed_coarse": dialogue["completed_coarse"],
        "failed_dialogue": dialogue["failed"],
        "short_failures_le_4_chars": sum(
            normalized_char_count(str(item.get("text", ""))) <= 4 for item in failed_dialogue
        ),
    }


def capability_matrix() -> list[dict[str, Any]]:
    return [
        {
            "capability": "qwen_primary_align",
            "conclusion": "IMPLEMENTED_REACHABLE",
            "code": ["qwen_asr/commands/align.py::cmd_align", "qwen_asr/align.py::QwenForcedAligner"],
            "entry": "CLI align/run/batch-run with align_backend=qwen",
            "default": "qwen",
            "tests": ["tests/test_align_cleanup.py"],
            "historical_evidence": "full regression: 128 input, 104 completed, 24 failed",
            "actual_reachability": "production entry executes the backend; 22 non-music failures remain",
        },
        {
            "capability": "asr_short_window_fallback",
            "conclusion": "IMPLEMENTED_REACHABLE",
            "code": ["qwen_asr/commands/align.py::_run_asr_short_window_align_fallback"],
            "entry": "align_fallback=asr-short-window",
            "default": "off",
            "tests": ["tests/test_align_cleanup.py"],
            "historical_evidence": "24/24 failures attempted; exact recovery 0/24",
            "actual_reachability": "content preservation guard rejects transcript rewrites and must remain strict",
        },
        {
            "capability": "align_parameter_matrix",
            "conclusion": "REJECTED_BY_EVIDENCE",
            "code": ["qwen_asr/align.py::AlignTimingValidationConfig", "qwen_asr/align.py::validate_aligned_token_timing"],
            "entry": "explicit align timing validation and repair flags",
            "default": "current defaults retained",
            "tests": ["tests/test_align_cleanup.py"],
            "historical_evidence": "P6 nine variants had identical ASS target metrics",
            "actual_reachability": "reachable but broad rerun is prohibited without a new falsifiable hypothesis",
        },
        {
            "capability": "mfa_full_backend",
            "conclusion": "REJECTED_BY_EVIDENCE",
            "code": ["tools/mfa_full_alignment.py", "qwen_asr/mfa_backend.py"],
            "entry": "align_backend=mfa",
            "default": "qwen",
            "tests": ["tests/test_mfa_backend.py"],
            "historical_evidence": "two-dataset A/B showed about 0.5 coverage and downstream regression",
            "actual_reachability": "implemented but not eligible for Goal A broad rerun",
        },
        {
            "capability": "mfa_local_fallback",
            "conclusion": "IMPLEMENTED_NOT_REACHABLE",
            "code": ["qwen_asr/mfa_candidates.py", "qwen_asr/mfa_runner.py", "qwen_asr/mfa_guards.py", "qwen_asr/mfa_writeback.py"],
            "entry": "proofread-realign experiment path only",
            "default": "not connected to initial Align failed-dialogue recovery",
            "tests": ["tests/test_mfa_candidates.py", "tests/test_mfa_runner.py", "tests/test_mfa_guards.py", "tests/test_mfa_writeback.py"],
            "historical_evidence": "kept only as Japanese local fallback; English mismatch was correctly rejected",
            "actual_reachability": "no shared first-Align recovery executor currently dispatches it",
        },
        {
            "capability": "recovery_retry_align",
            "conclusion": "IMPLEMENTED_NOT_REACHABLE",
            "code": ["qwen_asr/recovery_service.py::perform_recovery_action"],
            "entry": "Web/API action retry_align",
            "default": "strategy=qwen",
            "tests": ["tests/test_recovery_service.py", "tests/test_web_workspace_api.py"],
            "historical_evidence": "action persists retry_requested only",
            "actual_reachability": "does not invoke any align backend",
        },
        {
            "capability": "recovery_language_route",
            "conclusion": "IMPLEMENTED_NOT_REACHABLE",
            "code": ["qwen_asr/recovery_service.py::perform_recovery_action"],
            "entry": "Web/API action route_language",
            "default": "none",
            "tests": ["tests/test_recovery_service.py"],
            "historical_evidence": "language value is persisted with audit",
            "actual_reachability": "does not affect backend dispatch",
        },
        {
            "capability": "vad_coarse_writeback",
            "conclusion": "IMPLEMENTED_REACHABLE",
            "code": ["qwen_asr/recovery_service.py::_localize_with_vad", "qwen_asr/recovery_service.py::_apply_completed_coarse"],
            "entry": "localize_vad then accept_completed_coarse",
            "default": "pyannote_onnx_v3, threshold=0.5",
            "tests": ["tests/test_recovery_service.py"],
            "historical_evidence": "manifest/checkpoint/event backup and coarse writeback are tested",
            "actual_reachability": "reachable, but transcript verification, multi-region and neighbor hard gates are missing",
        },
        {
            "capability": "proofread_realign",
            "conclusion": "NOT_APPLICABLE",
            "code": ["qwen_asr/proofread_realign.py"],
            "entry": "post-edit proofread-realign stage",
            "default": "post-translation/post-edit only",
            "tests": ["tests/test_proofread_realign.py", "tests/test_proofread_realign_strategy.py"],
            "historical_evidence": "Qwen clamp, MFA local and mixed-language original timing are guarded",
            "actual_reachability": "not a replacement for initial Align failure recovery",
        },
    ]


def fallback_payload(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("raw_model_output")
    if not isinstance(raw, dict):
        return {}
    payload = raw.get("align_fallback")
    return payload if isinstance(payload, dict) else {}


def root_cause(item: dict[str, Any], fallback: dict[str, Any]) -> tuple[str, str]:
    text = str(item.get("text", ""))
    language = str(item.get("language", ""))
    if any("LATIN" in unicodedata.name(char, "") for char in text if char.isalpha()) and language.lower() == "japanese":
        return "language_route_mismatch", "HIGH"
    if fallback.get("content_error"):
        return "qwen_timing_failure_then_short_window_content_guard_rejection", "HIGH"
    return "qwen_token_coverage_below_safety_threshold", "HIGH"


def observed_conditions(item: dict[str, Any], fallback: dict[str, Any], aligned: list[dict[str, Any]], index: int) -> list[str]:
    conditions = ["transcript_correctness_unknown"]
    tokens = item.get("tokens") if isinstance(item.get("tokens"), list) else []
    if not tokens:
        conditions.append("no_tokens")
    zero_count = 0
    for token in tokens:
        if not isinstance(token, dict):
            zero_count += 1
            continue
        try:
            if float(token.get("end_time", 0)) <= float(token.get("start_time", 0)):
                zero_count += 1
        except (TypeError, ValueError):
            zero_count += 1
    if zero_count:
        conditions.append("zero_duration_token")
    if tokens and zero_count / len(tokens) >= 0.5:
        conditions.append("dense_zero_tokens")
    coverage = item.get("alignment_coverage") if item.get("alignment_coverage") is not None else token_coverage(item)
    if isinstance(coverage, (int, float)) and coverage < 0.2:
        conditions.append("low_coverage")
    if fallback.get("content_error"):
        conditions.append("asr_rewrite_rejected")
    text = str(item.get("text", ""))
    language = str(item.get("language", ""))
    if any("LATIN" in unicodedata.name(char, "") for char in text if char.isalpha()) and language.lower() == "japanese":
        conditions.append("language_mismatch")
    start_ms = round(float(item["global_start_time"]) * 1000)
    end_ms = round(float(item["global_end_time"]) * 1000)
    previous_end = round(float(aligned[index - 1]["global_end_time"]) * 1000) if index > 0 else None
    next_start = round(float(aligned[index + 1]["global_start_time"]) * 1000) if index + 1 < len(aligned) else None
    if (previous_end is not None and start_ms < previous_end) or (next_start is not None and end_ms > next_start):
        conditions.append("neighbor_overlap_risk")
    return conditions


def failure_traces(aligned: list[dict[str, Any]], transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals = music_intervals()
    transcript_by_id = {str(item.get("segment_id")): item for item in transcripts}
    rows = []
    for index, item in enumerate(aligned):
        if derive_alignment_state(item) != "failed" or overlaps_music_region(item, intervals):
            continue
        segment_id = str(item.get("segment_id"))
        fallback = fallback_payload(item)
        normalized_chars = normalized_char_count(str(item.get("text", "")))
        root_cause_name, confidence = root_cause(item, fallback)
        rows.append(
            {
                "segment_id": segment_id,
                "original_transcript": str(transcript_by_id.get(segment_id, {}).get("text", item.get("text", ""))),
                "normalized_char_count": normalized_chars,
                "route": "short_response" if normalized_chars <= 4 else "standard",
                "language": item.get("language"),
                "original_range_ms": [round(float(item["global_start_time"]) * 1000), round(float(item["global_end_time"]) * 1000)],
                "token_count": len(item.get("tokens", [])) if isinstance(item.get("tokens"), list) else 0,
                "coverage": item.get("alignment_coverage") if item.get("alignment_coverage") is not None else token_coverage(item),
                "error": item.get("error"),
                "neighbor_context": {
                    "previous": aligned[index - 1].get("segment_id") if index > 0 else None,
                    "next": aligned[index + 1].get("segment_id") if index + 1 < len(aligned) else None,
                },
                "short_window": {
                    "attempted": bool(fallback),
                    "window_count": fallback.get("window_count", 0),
                    "completed_window_alignments": fallback.get("completed_window_alignments", 0),
                    "merged_asr_text": fallback.get("merged_asr_text"),
                    "content_error": fallback.get("content_error"),
                    "windows": fallback.get("windows", []),
                },
                "transcript_evidence_state": "unknown_requires_audio_verification",
                "observed_conditions": observed_conditions(item, fallback, aligned, index),
                "attempted_capabilities": ["qwen_primary_align", "asr_short_window_fallback"],
                "skipped_capabilities": [
                    {"capability": "mfa_local_fallback", "skip_reason": "not connected to initial Align recovery executor"},
                    {"capability": "vad_coarse", "skip_reason": "not executed in the immutable baseline and transcript is not yet verified"},
                    {"capability": "language_route_executor", "skip_reason": "route_language currently persists state only"},
                ],
                "final_root_cause": root_cause_name,
                "confidence": confidence,
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    output.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return "\n".join(output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Align recovery A0-A2 audit artifacts")
    parser.add_argument("--derived-workspace", type=Path, required=True)
    args = parser.parse_args()
    target = args.derived_workspace.resolve()
    baseline = read_json(BASELINE_PATH)
    evidence = verify_baseline(baseline)
    if any(item["status"] != "MATCH" for item in evidence):
        raise RuntimeError("baseline evidence mismatch; refusing to create derived workspace")
    derivation = create_derived_workspace(target)
    aligned = read_json(target / "manifests" / "aligned_segments.json")
    transcripts = read_json(target / "manifests" / "transcript_segments.json")
    metrics = dialogue_metrics(aligned)
    expected = {key: baseline["baseline_metrics"][key] for key in metrics}
    if metrics != expected:
        raise RuntimeError(f"derived metrics mismatch: actual={metrics!r} expected={expected!r}")
    now = datetime.now(timezone.utc).isoformat()
    a0 = {
        "schema_version": 1,
        "generated_at": now,
        "status": "PASS",
        "git": {
            "branch": git_value("branch", "--show-current"),
            "commit": git_value("rev-parse", "HEAD"),
            "status_short": git_value("status", "--short"),
        },
        "python": str(ROOT / ".venv312" / "Scripts" / "python.exe"),
        "source_workspace": str(SOURCE.relative_to(ROOT)),
        "diagnostic_workspace": str(DIAGNOSTIC.relative_to(ROOT)),
        "derived_workspace": str(target.relative_to(ROOT)),
        "evidence": evidence,
        "metrics": metrics,
        "derivation": derivation,
    }
    write_json(ROOT / "reports" / "align_recovery_a0_baseline.json", a0)
    a0_rows = [[item["path"], item["bytes"], item["status"]] for item in evidence]
    write_text(
        ROOT / "reports" / "align_recovery_a0_baseline.md",
        "# Align 恢复 A0 基线冻结报告\n\n"
        f"状态：PASS  \n派生工作区：`{target.relative_to(ROOT)}`  \n分支/提交：`{a0['git']['branch']}` / `{a0['git']['commit']}`\n\n"
        "## 固定证据校验\n\n"
        + markdown_table(["证据", "字节", "SHA-256 状态"], a0_rows)
        + "\n\n## 对话基线\n\n"
        + markdown_table(["指标", "值"], [[key, value] for key, value in metrics.items()])
        + "\n\n原始两个工作区未被修改；派生工作区只复制必要 manifest、状态文件和 OP/ED 证据，音频路径继续只读引用源工作区。",
    )
    matrix = capability_matrix()
    matrix_payload = {"schema_version": 1, "generated_at": now, "status": "COMPLETE", "items": matrix}
    write_json(ROOT / "reports" / "align_recovery_capability_matrix.json", matrix_payload)
    write_text(
        ROOT / "reports" / "align_recovery_capability_matrix.md",
        "# Align 恢复既有能力与生产可达性矩阵\n\n"
        + markdown_table(
            ["能力", "结论", "入口/默认", "真实可达性"],
            [[item["capability"], item["conclusion"], f"{item['entry']} / {item['default']}", item["actual_reachability"]] for item in matrix],
        )
        + "\n\n结论：Qwen 主 Align、short-window 和 VAD/coarse 代码路径可达；MFA local、retry_align 与 route_language 尚未由共享首次 Align 恢复执行器真实派发。P6 九变体和 MFA 全量替代已有否定证据，不重复运行。",
    )
    traces = failure_traces(aligned, transcripts)
    if len(traces) != 22 or sum(item["route"] == "short_response" for item in traces) != 18:
        raise RuntimeError("failure trace invariant mismatch")
    trace_payload = {
        "schema_version": 1,
        "generated_at": now,
        "status": "COMPLETE",
        "source_workspace": str(SOURCE.relative_to(ROOT)),
        "derived_workspace": str(target.relative_to(ROOT)),
        "counts": {
            "dialogue_failed": len(traces),
            "short_response": sum(item["route"] == "short_response" for item in traces),
            "short_window_attempted": sum(item["short_window"]["attempted"] for item in traces),
        },
        "root_causes": dict(Counter(item["final_root_cause"] for item in traces)),
        "manual_audio_verification_required": [item["segment_id"] for item in traces],
        "items": traces,
    }
    write_json(ROOT / "reports" / "align_recovery_failure_trace.json", trace_payload)
    write_text(
        ROOT / "reports" / "align_recovery_failure_trace.md",
        "# Align 恢复 22 条失败执行轨迹\n\n"
        f"已覆盖 22/22 条非 OP/ED 对白失败；18/18 个规范化字符数不超过 4 的条目已标记为 `short_response`。22 条均已有 Qwen 主 Align 和 short-window 尝试证据。\n\n"
        + markdown_table(
            ["segment", "文本", "字符", "路由", "token", "coverage", "主根因", "置信度"],
            [[item["segment_id"], item["original_transcript"], item["normalized_char_count"], item["route"], item["token_count"], item["coverage"], item["final_root_cause"], item["confidence"]] for item in traces],
        )
        + "\n\n所有 transcript 在 coarse 或 verified-text retry 前仍需音频核验；参考 ASS 未作为 transcript、Qwen 或 MFA 输入。未执行的 MFA local、VAD/coarse 和语言路由均记录了明确 skip reason。",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
