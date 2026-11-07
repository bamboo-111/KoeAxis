from __future__ import annotations

import argparse

from qwen_asr.models import WorkPaths
from qwen_asr.progress import write_progress
from qwen_asr.proofread_realign import run_proofread_realign_stage


def cmd_proofread_realign(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    report = run_proofread_realign_stage(args, work_paths)
    write_progress(
        work_paths,
        stage="proofread-realign",
        status="completed" if report["status"] != "FAIL" else "failed",
        summary=(
            f"proofread-realign {report['status']}: "
            f"{report.get('completed_count', 0)} completed, {report.get('failed_count', 0)} failed"
        ),
    )
    return 0 if report["status"] != "FAIL" else 1
