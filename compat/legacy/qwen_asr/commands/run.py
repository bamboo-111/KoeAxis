from __future__ import annotations

import argparse

from qwen_asr.models import WorkPaths


def cmd_run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    from qwen_asr.commands import stages as stage_helpers

    stage_helpers.ensure_preflight(args, work_paths, "run")
    stage_helpers._validate_mimo_diagnostic_scope(args)
    if args.with_translate:
        if not args.target_language or not args.llm_model or not args.llm_base_url or not args.llm_api_key:
            raise RuntimeError("translate stage requires --target-language, --llm-model, --llm-base-url, and --llm-api-key")
    if getattr(args, "with_mimo_proofread", False):
        translated_ready = (
            work_paths.translated_manifest.exists()
            and stage_helpers._translated_manifest_has_content(work_paths.translated_manifest)
        )
        if not args.with_translate and not translated_ready:
            raise RuntimeError("MiMo audio proofread requires --with-translate or an existing non-empty translated_segments.json")
        if not stage_helpers.resolve_mimo_api_key(getattr(args, "mimo_api_key", None)):
            raise RuntimeError("MiMo audio proofread requires MIMO_API_KEY")
    handlers = {
        "prepare": stage_helpers.cmd_prepare,
        "transcribe": stage_helpers.cmd_transcribe,
        "correct": stage_helpers.cmd_correct,
        "align": stage_helpers.cmd_align,
        "split": stage_helpers.cmd_split,
        "translate": stage_helpers.cmd_translate,
        "mimo-proofread": stage_helpers.cmd_mimo_proofread,
        "proofread-realign": stage_helpers.cmd_proofread_realign,
        "normalize": stage_helpers.cmd_normalize,
        "export": stage_helpers.cmd_export,
    }
    return stage_helpers.PipelineRunner(work_paths, handlers).run(args)
