from __future__ import annotations

import argparse
import json

from qwen_asr.models import WorkPaths
from qwen_asr.recovery_executor import execute_alignment_recovery


def cmd_recover_align(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    result = execute_alignment_recovery(
        work_paths,
        segment_id=str(args.segment_id),
        strategy=str(args.strategy),
        language_route=str(args.language_route or "auto"),
        verified_text=str(args.verified_text or "") or None,
        use_verified_text=bool(args.use_verified_text),
        actor=str(args.actor),
        settings={
            "model": args.model,
            "model_cache_dir": args.model_cache_dir,
            "dtype": args.dtype,
            "device": args.device,
            "attn_implementation": args.attn_implementation,
            "local_files_only": args.local_files_only,
            "mfa_padding_ms": args.mfa_padding_ms,
            "mfa_min_content_score": args.mfa_min_content_score,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("alignment_state") == "completed_exact" else 1
