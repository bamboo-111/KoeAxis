from __future__ import annotations

import argparse

from qwen_asr.models import WorkPaths
from qwen_asr.preflight import format_preflight_messages, run_preflight


def cmd_preflight(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    result = run_preflight(args, work_paths, "preflight")
    for line in format_preflight_messages(result):
        print(line)
    return 0 if result.ok else 1
