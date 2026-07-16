from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_asr.align import QwenForcedAligner  # noqa: E402
from qwen_asr.alignment_state import derive_alignment_state, overlaps_music_region, read_music_region_evidence  # noqa: E402
from qwen_asr.defaults import DEFAULT_ALIGN_MODEL, DEFAULT_MODEL_CACHE_DIR  # noqa: E402
from qwen_asr.mfa_environment import detect_mfa_environment  # noqa: E402
from qwen_asr.mfa_runner import run_local_mfa_alignment_experiments  # noqa: E402
from qwen_asr.models import WorkPaths  # noqa: E402
from qwen_asr.recovery_executor import execute_alignment_recovery  # noqa: E402
from qwen_asr.recovery_service import RecoveryError, build_recovery_view, perform_recovery_action  # noqa: E402
from qwen_asr.storage import read_json, write_json_atomic  # noqa: E402


TRACE_PATH = ROOT / "reports" / "align_recovery_failure_trace.json"
OUTPUT_JSON = ROOT / "reports" / "align_recovery_sayonara_result.json"
OUTPUT_MD = ROOT / "reports" / "align_recovery_sayonara_result.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def metrics(work_paths: WorkPaths) -> dict[str, int]:
    aligned = read_json(work_paths.aligned_manifest, default=[])
    intervals, _, _, _ = read_music_region_evidence(work_paths.workdir)
    counts = Counter()
    short_failed = 0
    for item in aligned:
        if overlaps_music_region(item, intervals):
            continue
        state = derive_alignment_state(item)
        counts[state] += 1
        if state == "failed":
            text = str(item.get("text", ""))
            normalized_chars = sum(character.isalnum() for character in text)
            short_failed += normalized_chars <= 4
    return {
        "completed_exact": counts["completed_exact"],
        "completed_coarse": counts["completed_coarse"],
        "failed": counts["failed"],
        "short_failed": short_failed,
    }


def run_qwen(work_paths: WorkPaths, segment_ids: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    attempts = []
    aligner = QwenForcedAligner(
        model_name=DEFAULT_ALIGN_MODEL,
        dtype="fp16",
        device="cuda",
        keep_raw_model_output=True,
        keep_failed_tokens=True,
        model_cache_dir=str(DEFAULT_MODEL_CACHE_DIR),
        local_files_only=True,
    )
    try:
        aligner.load()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return [
            {
                "segment_id": segment_id,
                "strategy": "qwen",
                "status": "not_executable",
                "alignment_state": "failed",
                "error_code": "QWEN_MODEL_UNAVAILABLE",
                "error": error,
            }
            for segment_id in segment_ids
        ], error
    try:
        for segment_id in segment_ids:
            started = time.monotonic()
            try:
                result = execute_alignment_recovery(
                    work_paths,
                    segment_id=segment_id,
                    strategy="qwen",
                    language_route="auto",
                    actor="goal-a-real-data",
                    qwen_runner=lambda segment: aligner.run_segment(segment, cleanup=False),
                )
            except Exception as exc:
                result = {
                    "segment_id": segment_id,
                    "strategy": "qwen",
                    "status": "executor_error",
                    "alignment_state": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            result.setdefault("wall_elapsed_ms", round((time.monotonic() - started) * 1000))
            attempts.append(result)
    finally:
        aligner.close()
    return attempts, None


def run_mfa_local(work_paths: WorkPaths, trace_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = {item["segment_id"]: item for item in build_recovery_view(work_paths)["items"]}
    candidates = []
    skipped = []
    for trace in trace_items:
        segment_id = trace["segment_id"]
        task = current.get(segment_id)
        if task is None:
            continue
        if str(task.get("language", "")).lower() != "japanese":
            skipped.append({"segment_id": segment_id, "strategy": "mfa-local", "status": "skipped", "reason": "non-japanese"})
            continue
        if trace["route"] == "short_response":
            skipped.append(
                {
                    "segment_id": segment_id,
                    "strategy": "mfa-local",
                    "status": "skipped",
                    "reason": "short-response routed to VAD; Japanese dictionary alignment is not a safe first choice",
                }
            )
            continue
        candidates.append(
            {
                "source": "align-recovery",
                "reason": "initial-align-failed",
                "severity": "WARN",
                "subtitle_id": segment_id,
                "start_ms": int(task["start_ms"]),
                "end_ms": int(task["end_ms"]),
                "text": task["original_transcript"],
                "details": {"original_error": task["error"]},
            }
        )
    if not candidates:
        return skipped
    environment = detect_mfa_environment(run_version_check=True)
    if not environment.get("available"):
        return skipped + [
            {
                "segment_id": candidate["subtitle_id"],
                "strategy": "mfa-local",
                "status": "not_executable",
                "reason": str(environment.get("reason") or "mfa-unavailable"),
            }
            for candidate in candidates
        ]
    aligned = read_json(work_paths.aligned_manifest, default=[])
    segment_audio = Path(str(next(item for item in aligned if item["segment_id"] == candidates[0]["subtitle_id"])["audio_path"]))
    source_audio = segment_audio.parent.parent / "source.wav"
    run_paths = replace(work_paths, audio_path=source_audio)
    batch_results = run_local_mfa_alignment_experiments(
        run_paths,
        candidates,
        environment=environment,
        max_run_candidates=len(candidates),
        padding_ms=700,
    )
    result_by_id = {
        str(item.get("candidate", {}).get("subtitle_id")): item
        for item in batch_results
        if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
    }
    attempts = []
    for candidate in candidates:
        segment_id = candidate["subtitle_id"]
        cached = result_by_id.get(segment_id, {"status": "skipped", "reason": "missing-batch-result"})
        try:
            result = execute_alignment_recovery(
                work_paths,
                segment_id=segment_id,
                strategy="mfa-local",
                language_route="Japanese",
                actor="goal-a-real-data",
                mfa_runner=lambda *args, cached=cached, **kwargs: [cached],
            )
        except Exception as exc:
            result = {
                "segment_id": segment_id,
                "strategy": "mfa-local",
                "status": "executor_error",
                "alignment_state": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        attempts.append(result)
    return skipped + attempts


def run_vad(work_paths: WorkPaths, trace_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts = []
    for trace in trace_items:
        if trace["route"] != "short_response":
            continue
        current = {item["segment_id"]: item for item in build_recovery_view(work_paths)["items"]}
        if trace["segment_id"] not in current:
            attempts.append({"segment_id": trace["segment_id"], "strategy": "vad", "status": "skipped", "reason": "already recovered exact"})
            continue
        try:
            result = perform_recovery_action(
                work_paths,
                segment_id=trace["segment_id"],
                action="localize_vad",
                payload={"backend": "pyannote_onnx_v3"},
                actor="goal-a-real-data",
            )
            proposal = result["task"]["vad_proposal"]
            attempts.append(
                {
                    "segment_id": trace["segment_id"],
                    "strategy": "vad",
                    "status": "localized",
                    "region_count": proposal["region_count"],
                    "unique_mapping": proposal["unique_mapping"],
                    "regions": proposal["regions"],
                    "elapsed_ms": proposal["elapsed_ms"],
                    "coarse_writeback": "not_attempted_transcript_unverified",
                }
            )
        except RecoveryError as exc:
            attempts.append(
                {
                    "segment_id": trace["segment_id"],
                    "strategy": "vad",
                    "status": "failed",
                    "error_code": exc.code,
                    "error": str(exc),
                }
            )
    return attempts


def markdown(payload: dict[str, Any]) -> str:
    before = payload["before"]
    after = payload["after"]
    lines = [
        "# Sayonara Lara Align 定向真实恢复结果",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 前后指标",
        "",
        "| 指标 | 恢复前 | 恢复后 |",
        "|---|---:|---:|",
    ]
    for key in ("completed_exact", "completed_coarse", "failed", "short_failed"):
        lines.append(f"| {key} | {before[key]} | {after[key]} |")
    lines.extend(
        [
            "",
            "## 策略执行",
            "",
            f"- Qwen 原 transcript retry：{len(payload['qwen_attempts'])} 条；exact={sum(item.get('alignment_state') == 'completed_exact' for item in payload['qwen_attempts'])}。",
            f"- MFA local：记录 {len(payload['mfa_attempts'])} 条适用/跳过结论；exact={sum(item.get('alignment_state') == 'completed_exact' for item in payload['mfa_attempts'])}。",
            f"- 短应答 VAD：{len(payload['vad_attempts'])} 条；localized={sum(item.get('status') == 'localized' for item in payload['vad_attempts'])}。",
            "- 未经 transcript 可信核验的 VAD 结果均未写成 completed_coarse。",
            "",
            "## 判定",
            "",
            payload["decision"],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Goal A targeted recovery on the derived Sayonara Lara workspace")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--phase", choices=["all", "qwen-only"], default="all")
    args = parser.parse_args()
    work_paths = WorkPaths.from_workdir(args.workdir.resolve())
    trace = read_json(TRACE_PATH)
    trace_items = trace["items"]
    phase_before = metrics(work_paths)
    previous = read_json(OUTPUT_JSON, default={}) if args.phase == "qwen-only" and OUTPUT_JSON.exists() else {}
    before = previous.get("before", phase_before)
    started = time.monotonic()
    currently_failed = {item["segment_id"] for item in build_recovery_view(work_paths)["items"]}
    qwen_attempts, qwen_load_error = run_qwen(
        work_paths,
        [item["segment_id"] for item in trace_items if item["segment_id"] in currently_failed],
    )
    if args.phase == "qwen-only":
        mfa_attempts = previous.get("mfa_attempts", [])
        vad_attempts = previous.get("vad_attempts", [])
    else:
        mfa_attempts = run_mfa_local(work_paths, trace_items)
        vad_attempts = run_vad(work_paths, trace_items)
    after = metrics(work_paths)
    exact_gain = after["completed_exact"] - before["completed_exact"]
    coarse_gain = after["completed_coarse"] - before["completed_coarse"]
    decision = (
        f"真实恢复新增 exact={exact_gain}、coarse={coarse_gain}。"
        "所有未解决条目继续保持 failed，并保留 backend 拒绝、不可执行或 transcript 未核验的证据；未放宽内容、时间或状态门。"
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "COMPLETE",
        "workspace": str(work_paths.workdir),
        "before": before,
        "phase_before": phase_before,
        "after": after,
        "elapsed_ms": int(previous.get("elapsed_ms", 0)) + round((time.monotonic() - started) * 1000),
        "qwen_model_load_error": qwen_load_error,
        "qwen_attempts": qwen_attempts,
        "mfa_attempts": mfa_attempts,
        "vad_attempts": vad_attempts,
        "decision": decision,
    }
    write_json_atomic(OUTPUT_JSON, payload)
    write_json_atomic(work_paths.workdir / "reports" / "align_recovery_execution.json", payload)
    OUTPUT_MD.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps({"before": before, "after": after, "elapsed_ms": payload["elapsed_ms"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
