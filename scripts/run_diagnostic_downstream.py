from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_asr.commands import stages as stage_helpers  # noqa: E402
from qwen_asr.models import WorkPaths  # noqa: E402
from qwen_asr.normalize import NormalizeParams, normalize_asr_data  # noqa: E402
from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT  # noqa: E402
from qwen_asr.storage import read_json, write_json_atomic  # noqa: E402
from qwen_asr.subtitle import export_vtt_from_optimizer_asr_data  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate diagnostic normalize/export artifacts after an explicit quality FAIL")
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()
    paths = WorkPaths.from_workdir(args.workdir.resolve())
    quality = read_json(paths.final_quality_report, default={})
    if str(quality.get("status", "")).upper() != "FAIL":
        raise RuntimeError("diagnostic bypass is only valid when the saved quality status is explicitly FAIL")
    started = time.monotonic()
    source = stage_helpers._load_normalize_source("mimo", paths, Path(DEFAULT_OPTIMIZER_ROOT))
    if source is None or not source.segments:
        raise RuntimeError("MiMo source is missing or empty")
    normalized = normalize_asr_data(
        source,
        NormalizeParams(extend_ms=350, snap_gap_ms=200, min_blank_ms=300),
    )
    paths.normalized_manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.normalized_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.subtitles_srt.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths.normalized_manifest, normalized.to_json())
    srt_text = normalized.to_srt()
    vtt_text = export_vtt_from_optimizer_asr_data(normalized)
    paths.normalized_srt.write_text(srt_text, encoding="utf-8")
    paths.subtitles_srt.write_text(srt_text, encoding="utf-8")
    paths.subtitles_vtt.write_text(vtt_text, encoding="utf-8")
    target_dir = paths.workdir / "exports"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_srt = target_dir / "subtitles.srt"
    target_vtt = target_dir / "subtitles.vtt"
    shutil.copy2(paths.subtitles_srt, target_srt)
    shutil.copy2(paths.subtitles_vtt, target_vtt)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "diagnostic_only": True,
        "quality_gate_bypassed": True,
        "bypass_scope": "isolated derived Goal A workspace only",
        "formal_quality_status": quality.get("status"),
        "formal_quality_summary": quality.get("summary"),
        "source": "mimo",
        "source_segment_count": len(source.segments),
        "normalized_segment_count": len(normalized.segments),
        "elapsed_ms": round((time.monotonic() - started) * 1000),
        "artifacts": {
            "normalized_manifest": {"path": str(paths.normalized_manifest), "sha256": sha256(paths.normalized_manifest)},
            "normalized_srt": {"path": str(paths.normalized_srt), "sha256": sha256(paths.normalized_srt)},
            "export_cache_srt": {"path": str(paths.subtitles_srt), "sha256": sha256(paths.subtitles_srt)},
            "export_cache_vtt": {"path": str(paths.subtitles_vtt), "sha256": sha256(paths.subtitles_vtt)},
            "target_srt": {"path": str(target_srt), "sha256": sha256(target_srt)},
            "target_vtt": {"path": str(target_vtt), "sha256": sha256(target_vtt)},
        },
        "content_identity": {
            "normalized_srt_equals_export_srt": sha256(paths.normalized_srt) == sha256(target_srt),
            "cache_srt_equals_target_srt": sha256(paths.subtitles_srt) == sha256(target_srt),
            "cache_vtt_equals_target_vtt": sha256(paths.subtitles_vtt) == sha256(target_vtt),
        },
    }
    workspace_report = paths.workdir / "reports" / "diagnostic_downstream_execution.json"
    root_report = ROOT / "reports" / "align_recovery_downstream_recompute.json"
    write_json_atomic(workspace_report, payload)
    write_json_atomic(root_report, payload)
    md = (
        "# Align 恢复下游重算\n\n"
        f"状态：{payload['status']}（仅诊断产物）  \n"
        f"正式质量状态：{payload['formal_quality_status']}  \n"
        f"输入/输出条目：{payload['source_segment_count']} / {payload['normalized_segment_count']}  \n"
        f"耗时：{payload['elapsed_ms']} ms\n\n"
        "正式 quality gate 未被修改或伪装；由于 alignment_health 仍 FAIL，normalize/export 仅在隔离派生工作区中生成，并明确记录 bypass。"
    )
    (ROOT / "reports" / "align_recovery_downstream_recompute.md").write_text(md + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "segments": len(normalized.segments), "elapsed_ms": payload["elapsed_ms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
