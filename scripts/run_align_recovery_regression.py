from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_asr.align import QwenForcedAligner  # noqa: E402
from qwen_asr.alignment_state import derive_alignment_state  # noqa: E402
from qwen_asr.defaults import DEFAULT_ALIGN_MODEL, DEFAULT_MODEL_CACHE_DIR  # noqa: E402
from qwen_asr.models import WorkPaths  # noqa: E402
from qwen_asr.recovery_executor import execute_alignment_recovery  # noqa: E402
from qwen_asr.storage import read_json, write_json_atomic  # noqa: E402


SOURCES = {
    "konoato01": ROOT / "workspaces" / "clean-regression-20260712-002943" / "konoato01",
    "madougushi02": ROOT / "workspaces" / "clean-regression-20260712-002943" / "madougushi02",
}
EXPECTED_TRANSCRIPT_SHA256 = {
    "konoato01": "ea4780c8fd3fba11222bff01d1ddb1adbdc51048463eeaad2e09073296dfc725",
    "madougushi02": "eb7eb8f12a5dab2eb3d1e23a0d54b1cabc5a4527da0d9b640ec1d0a55c2d620a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_copy(source: Path, target: Path) -> None:
    if target.exists():
        expected = target / "transcript_segments.json"
        if expected.exists() and sha256(expected) == sha256(source / "transcript_segments.json"):
            return
        raise FileExistsError(f"existing target is not the expected immutable-input copy: {target}")
    target.mkdir(parents=True)
    for name in ("project.json", "progress.json", "transcript_segments.json", "aligned_segments.json"):
        shutil.copy2(source / name, target / name)


def normalized_chars(text: str) -> int:
    return sum(character.isalnum() for character in text)


def structural_metrics(aligned: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(derive_alignment_state(item) for item in aligned)
    short_failed = sum(
        derive_alignment_state(item) == "failed" and normalized_chars(str(item.get("text", ""))) <= 4
        for item in aligned
    )
    illegal = 0
    out_of_range = 0
    non_monotonic = 0
    severe_overlap = 0
    previous_end = None
    for item in aligned:
        try:
            start = float(item["global_start_time"])
            end = float(item["global_end_time"])
        except (KeyError, TypeError, ValueError):
            illegal += 1
            continue
        if end <= start:
            illegal += 1
        if previous_end is not None and start < previous_end - 0.1:
            severe_overlap += 1
        previous_end = end
        token_starts = []
        for token in item.get("tokens", []) if isinstance(item.get("tokens"), list) else []:
            try:
                token_start = float(token["start_time"])
                token_end = float(token["end_time"])
            except (KeyError, TypeError, ValueError):
                illegal += 1
                continue
            if token_end <= token_start:
                illegal += 1
            if token_start < start or token_end > end:
                out_of_range += 1
            token_starts.append(token_start)
        if token_starts != sorted(token_starts):
            non_monotonic += 1
    return {
        "segment_count": len(aligned),
        "completed_exact": counts["completed_exact"],
        "completed_coarse": counts["completed_coarse"],
        "failed": counts["failed"],
        "short_failed": short_failed,
        "illegal_time_count": illegal,
        "out_of_range_token_count": out_of_range,
        "non_monotonic_segment_count": non_monotonic,
        "severe_overlap_count": severe_overlap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run two-dataset recovery regression on immutable-input copies")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    datasets = {}
    for name, source in SOURCES.items():
        transcript = source / "transcript_segments.json"
        actual_sha = sha256(transcript)
        if actual_sha != EXPECTED_TRANSCRIPT_SHA256[name]:
            raise RuntimeError(f"{name} transcript SHA-256 mismatch: {actual_sha}")
        target = output_root / name
        create_copy(source, target)
        paths = WorkPaths.from_workdir(target)
        before_manifest = read_json(paths.aligned_manifest, default=[])
        datasets[name] = {
            "source": str(source),
            "workdir": str(target),
            "transcript_sha256": actual_sha,
            "before": structural_metrics(before_manifest),
            "before_text": {str(item.get("segment_id")): str(item.get("text", "")) for item in before_manifest},
            "attempts": [],
        }
    aligner = QwenForcedAligner(
        model_name=DEFAULT_ALIGN_MODEL,
        dtype="fp16",
        device="cuda",
        keep_raw_model_output=True,
        keep_failed_tokens=True,
        model_cache_dir=str(DEFAULT_MODEL_CACHE_DIR),
        local_files_only=True,
    )
    started = time.monotonic()
    aligner.load()
    try:
        for row in datasets.values():
            paths = WorkPaths.from_workdir(Path(row["workdir"]))
            failed_ids = [
                str(item["segment_id"])
                for item in read_json(paths.aligned_manifest, default=[])
                if derive_alignment_state(item) == "failed"
            ]
            for segment_id in failed_ids:
                result = execute_alignment_recovery(
                    paths,
                    segment_id=segment_id,
                    strategy="qwen",
                    language_route="auto",
                    actor="goal-a-dual-regression",
                    qwen_runner=lambda segment: aligner.run_segment(segment, cleanup=False),
                )
                row["attempts"].append(result)
    finally:
        aligner.close()
    overall_pass = True
    for row in datasets.values():
        paths = WorkPaths.from_workdir(Path(row["workdir"]))
        after_manifest = read_json(paths.aligned_manifest, default=[])
        after_text = {str(item.get("segment_id")): str(item.get("text", "")) for item in after_manifest}
        row["after"] = structural_metrics(after_manifest)
        row["content_changed_ids"] = sorted(
            segment_id for segment_id, text in row.pop("before_text").items() if after_text.get(segment_id) != text
        )
        row["checks"] = {
            "failed_not_increased": row["after"]["failed"] <= row["before"]["failed"],
            "short_failed_not_increased": row["after"]["short_failed"] <= row["before"]["short_failed"],
            "content_preserved": not row["content_changed_ids"],
            "illegal_time_not_increased": row["after"]["illegal_time_count"] <= row["before"]["illegal_time_count"],
            "out_of_range_not_increased": row["after"]["out_of_range_token_count"] <= row["before"]["out_of_range_token_count"],
            "non_monotonic_not_increased": row["after"]["non_monotonic_segment_count"] <= row["before"]["non_monotonic_segment_count"],
            "severe_overlap_not_increased": row["after"]["severe_overlap_count"] <= row["before"]["severe_overlap_count"],
        }
        overall_pass = overall_pass and all(row["checks"].values())
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overall_pass else "FAIL",
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "datasets": datasets,
    }
    json_path = ROOT / "reports" / "align_recovery_dual_dataset_regression.json"
    md_path = ROOT / "reports" / "align_recovery_dual_dataset_regression.md"
    write_json_atomic(json_path, payload)
    lines = ["# Align 恢复双数据集回归", "", f"状态：{payload['status']}", "", "| 数据集 | exact 前→后 | failed 前→后 | 短失败前→后 | 内容守恒 | 非法时间回退 |", "|---|---:|---:|---:|---|---|"]
    for name, row in datasets.items():
        lines.append(
            f"| {name} | {row['before']['completed_exact']}→{row['after']['completed_exact']} | "
            f"{row['before']['failed']}→{row['after']['failed']} | {row['before']['short_failed']}→{row['after']['short_failed']} | "
            f"{'PASS' if row['checks']['content_preserved'] else 'FAIL'} | "
            f"{'PASS' if row['checks']['illegal_time_not_increased'] else 'FAIL'} |"
        )
    lines.extend(["", "两个固定 transcript 的 SHA-256 均与稳定快照一致；所有执行发生在新派生副本，原始 clean-regression 工作区未写入。", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "datasets": {name: {"before": row["before"], "after": row["after"]} for name, row in datasets.items()}}, ensure_ascii=False))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
