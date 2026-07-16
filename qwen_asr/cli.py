from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

from qwen_asr.ass_quality import cmd_ass_quality
from qwen_asr.ass_quality_diff import cmd_ass_quality_diff
from qwen_asr.batch_runner import run_batch_command
from qwen_asr.commands import (
    cmd_align,
    cmd_content_quality,
    cmd_correct,
    cmd_export,
    cmd_mimo_proofread,
    cmd_normalize,
    cmd_preflight,
    cmd_prepare,
    cmd_proofread_realign,
    cmd_quality_gate,
    cmd_run,
    cmd_split,
    cmd_transcribe,
    cmd_translate,
)
from qwen_asr.commands.recover_align import cmd_recover_align
from qwen_asr.credentials import resolve_llm_api_key
from qwen_asr.defaults import (
    DEFAULT_ALIGN_MODEL,
    DEFAULT_ASR_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_CONCURRENCY,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
)
from qwen_asr.glossary import write_normalized_glossary_xlsx
from qwen_asr.history_glossary import cmd_history_glossary
from qwen_asr.logging_utils import setup_logging
from qwen_asr.mfa_experiment import cmd_mfa_align_experiment
from qwen_asr.models import WorkPaths
from qwen_asr.optimizer_bridge import DEFAULT_OPTIMIZER_ROOT
from qwen_asr.progress import read_progress, write_progress
from qwen_asr.quality_suspects import cmd_apply_quality_suspects

LOGGER = logging.getLogger(__name__)

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "llm_api_key"):
        args.llm_api_key = resolve_llm_api_key(args.llm_api_key, getattr(args, "llm_base_url", None)) or None

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    if getattr(args, "command", "") == "glossary-normalize":
        try:
            return cmd_glossary_normalize(args)
        except Exception as exc:
            LOGGER.exception("Command failed")
            print(str(exc), file=sys.stderr)
            return 1

    if getattr(args, "command", "") == "apply-quality-suspects":
        try:
            work_paths = WorkPaths.from_workdir(Path(args.workdir))
            return cmd_apply_quality_suspects(args, work_paths)
        except Exception as exc:
            LOGGER.exception("Command failed")
            print(str(exc), file=sys.stderr)
            return 1

    if getattr(args, "command", "") == "ass-quality-diff":
        try:
            return cmd_ass_quality_diff(args)
        except Exception as exc:
            LOGGER.exception("Command failed")
            print(str(exc), file=sys.stderr)
            return 1

    work_paths = WorkPaths.from_workdir(Path(args.workdir))
    _apply_model_cache_default(args)
    log_file = work_paths.logs_dir / f"{args.command}.log"
    setup_logging(log_file=log_file, level=args.log_level)

    try:
        if getattr(args, "command", "") == "ass-quality":
            return cmd_ass_quality(args, work_paths)
        return _run_command_with_progress(args, work_paths)
    except Exception:
        command = getattr(args, "command", "unknown")
        if command != "ass-quality":
            write_progress(
                work_paths,
                stage=command,
                status="failed",
                current="Command failed",
                summary=f"{command} failed",
            )
        LOGGER.exception("Command failed")
        return 1


def _run_command_with_progress(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    command = getattr(args, "command", "")
    if command == "batch-run":
        handlers = {
            "prepare": cmd_prepare,
            "transcribe": cmd_transcribe,
            "correct": cmd_correct,
            "align": cmd_align,
            "split": cmd_split,
            "translate": cmd_translate,
            "mimo-proofread": cmd_mimo_proofread,
            "proofread-realign": cmd_proofread_realign,
            "quality-gate": cmd_quality_gate,
            "normalize": cmd_normalize,
            "export": cmd_export,
        }
        return run_batch_command(args, handlers)
    return _run_stage_with_progress(command, args.func, args, work_paths)


def _run_stage_with_progress(
    stage: str,
    handler: Callable[[argparse.Namespace, WorkPaths], int],
    args: argparse.Namespace,
    work_paths: WorkPaths,
) -> int:
    write_progress(work_paths, stage=stage, status="running", current="", summary=f"{stage} started")
    status = handler(args, work_paths)
    existing = read_progress(work_paths) or {}
    write_progress(
        work_paths,
        stage=stage,
        status="completed" if status == 0 else "failed",
        done=existing.get("done"),
        total=existing.get("total"),
        current=existing.get("current", ""),
        summary=existing.get("summary") or f"{stage} {'completed' if status == 0 else 'failed'}",
    )
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline Qwen3-ASR subtitle pipeline")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", required=True)
    common.add_argument("--log-level", default="INFO")
    common.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    common.add_argument("--force", action="store_true")
    common.add_argument("--skip-preflight", action="store_true")
    common.add_argument("--dry-run-check", action="store_true")

    model_common = argparse.ArgumentParser(add_help=False)
    model_common.add_argument(
        "--model-cache-dir",
        default=str(DEFAULT_MODEL_CACHE_DIR),
        help=f"Model cache directory. Defaults to project-local {DEFAULT_MODEL_CACHE_DIR}.",
    )
    model_common.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    model_common.add_argument("--device", default="cuda")
    model_common.add_argument("--attn-implementation", default=None)
    model_common.add_argument("--keep-raw-model-output", action="store_true")
    model_common.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)

    prepare = subparsers.add_parser("prepare", parents=[common])
    prepare.add_argument("--media")
    prepare.add_argument("--video", help="Deprecated alias for --media.")
    _add_prepare_audio_arguments(prepare)
    prepare.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SECONDS)
    prepare.add_argument("--min-segment-seconds", type=float, default=DEFAULT_MIN_SEGMENT_SECONDS)
    prepare.add_argument("--preferred-silence-ms", type=int, default=800)
    prepare.add_argument("--min-silence-ms", type=int, default=500)
    prepare.add_argument("--padding-ms", type=int, default=300)
    prepare.add_argument("--overlap-ms", type=int, default=0)
    prepare.add_argument("--eager-segment-export", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    preflight = subparsers.add_parser("preflight", parents=[common, model_common])
    preflight.add_argument("--media")
    preflight.add_argument("--video", help="Deprecated alias for --media.")
    preflight.set_defaults(func=cmd_preflight)

    transcribe = subparsers.add_parser("transcribe", parents=[common, model_common])
    transcribe.add_argument("--model", default=DEFAULT_ASR_MODEL)
    transcribe.add_argument("--batch-size", type=int, default=None)
    transcribe.add_argument("--batch-mode", choices=["fixed", "adaptive"], default="adaptive")
    transcribe.add_argument("--target-batch-audio-seconds", type=float, default=None)
    transcribe.add_argument("--single-long-segment-threshold", type=float, default=None)
    transcribe.add_argument("--profile-batches", action="store_true")
    transcribe.add_argument("--max-new-tokens", type=int, default=512)
    transcribe.add_argument("--language", default=None)
    transcribe.set_defaults(func=cmd_transcribe)

    align = subparsers.add_parser("align", parents=[common, model_common])
    align.add_argument("--model", default=DEFAULT_ALIGN_MODEL)
    align.add_argument("--cleanup-interval", type=int, default=4)
    align.add_argument(
        "--align-diagnostics-mode",
        choices=["off", "capture-failed"],
        default="off",
        help="When enabled, failed alignments keep raw output and extracted tokens for diagnosis.",
    )
    align.add_argument("--align-fallback", choices=["off", "asr-short-window"], default="off")
    align.add_argument("--align-fallback-window-seconds", type=float, default=3.0)
    _add_align_timing_arguments(align)
    align.add_argument("--asr-reference-model", default=DEFAULT_ASR_MODEL)
    align.add_argument("--asr-reference-max-new-tokens", type=int, default=512)
    align.add_argument("--asr-reference-language", default=None)
    align.set_defaults(func=cmd_align)

    recover_align = subparsers.add_parser("recover-align", parents=[common, model_common])
    recover_align.add_argument("--segment-id", required=True)
    recover_align.add_argument("--strategy", choices=["auto", "qwen", "mfa-local"], default="auto")
    recover_align.add_argument("--language-route", default="auto")
    recover_align.add_argument("--verified-text", default="")
    recover_align.add_argument("--use-verified-text", action="store_true")
    recover_align.add_argument("--actor", default="cli-local-user")
    recover_align.add_argument("--model", default=DEFAULT_ALIGN_MODEL)
    recover_align.add_argument("--mfa-padding-ms", type=int, default=700)
    recover_align.add_argument("--mfa-min-content-score", type=float, default=0.70)
    recover_align.set_defaults(func=cmd_recover_align)

    mfa_align_experiment = subparsers.add_parser("mfa-align-experiment", parents=[common])
    mfa_align_experiment.add_argument("--ass-quality-report", action="append", default=[])
    mfa_align_experiment.add_argument("--ass-quality-diff-report", action="append", default=[])
    mfa_align_experiment.add_argument("--max-candidates", type=int, default=40)
    mfa_align_experiment.add_argument("--run-local-alignments", action="store_true")
    mfa_align_experiment.add_argument("--max-run-candidates", type=int, default=3)
    mfa_align_experiment.add_argument("--local-padding-ms", type=int, default=700)
    mfa_align_experiment.add_argument("--mfa-local-writeback", choices=["off", "propose", "apply"], default="off")
    mfa_align_experiment.add_argument("--mfa-local-writeback-output", default="")
    mfa_align_experiment.add_argument("--output", default="")
    mfa_align_experiment.add_argument("--markdown-output", default=None)
    mfa_align_experiment.add_argument("--mfa-version-check", action=argparse.BooleanOptionalAction, default=True)
    mfa_align_experiment.set_defaults(func=cmd_mfa_align_experiment)

    normalize = subparsers.add_parser("normalize", parents=[common])
    normalize.add_argument("--source", choices=["auto", "normalized", "mimo", "translated", "split", "transcript"], default="auto")
    normalize.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    normalize.add_argument("--extend-ms", type=int, default=350)
    normalize.add_argument("--snap-gap-ms", type=int, default=200)
    normalize.add_argument("--min-blank-ms", type=int, default=300)
    normalize.set_defaults(func=cmd_normalize)

    glossary_normalize = subparsers.add_parser("glossary-normalize")
    glossary_normalize.add_argument("--xlsx", required=True)
    glossary_normalize.add_argument("--output", default=None)
    glossary_normalize.add_argument("--log-level", default="INFO")
    glossary_normalize.set_defaults(func=cmd_glossary_normalize)

    apply_quality_suspects = subparsers.add_parser("apply-quality-suspects")
    apply_quality_suspects.add_argument("--workdir", required=True)
    apply_quality_suspects.add_argument("--ass-quality-report", action="append", default=[])
    apply_quality_suspects.add_argument("--ass-quality-diff-report", action="append", default=[])
    apply_quality_suspects.add_argument("--quality-suspect-max-distance-ms", type=int, default=8000)
    apply_quality_suspects.add_argument("--quality-suspect-report-output", default="")
    apply_quality_suspects.add_argument("--log-level", default="INFO")
    apply_quality_suspects.set_defaults(func=cmd_apply_quality_suspects)

    ass_quality_diff = subparsers.add_parser("ass-quality-diff")
    ass_quality_diff.add_argument(
        "--report",
        action="append",
        required=True,
        help="ASS quality report path. Use label=path to override the stage label. Repeat in stage order.",
    )
    ass_quality_diff.add_argument("--score-drop-threshold", type=float, default=0.15)
    ass_quality_diff.add_argument("--length-drop-ratio", type=float, default=0.60)
    ass_quality_diff.add_argument("--max-cases", type=int, default=50)
    ass_quality_diff.add_argument("--output", required=True)
    ass_quality_diff.add_argument("--markdown-output", default=None)
    ass_quality_diff.add_argument("--log-level", default="INFO")
    ass_quality_diff.set_defaults(func=cmd_ass_quality_diff)

    content_quality = subparsers.add_parser("content-quality", parents=[common])
    content_quality.add_argument("--include-export", action="store_true")
    content_quality.set_defaults(func=cmd_content_quality)

    quality_gate = subparsers.add_parser("quality-gate", parents=[common])
    quality_gate.add_argument("--include-export", action="store_true")
    quality_gate.add_argument("--require-srt", action="store_true")
    quality_gate.set_defaults(func=cmd_quality_gate)

    proofread_realign = subparsers.add_parser("proofread-realign", parents=[common, model_common])
    proofread_realign.add_argument("--proofread-realign-model", default=DEFAULT_ALIGN_MODEL)
    proofread_realign.add_argument("--proofread-realign-padding-ms", type=int, default=800)
    proofread_realign.add_argument("--proofread-realign-language", default="Japanese")
    proofread_realign.add_argument("--proofread-realign-fallback", choices=["original-timing", "fail"], default="original-timing")
    proofread_realign.add_argument("--proofread-realign-mfa-fallback", choices=["off", "local"], default="off")
    proofread_realign.add_argument("--proofread-realign-mfa-padding-ms", type=int, default=700)
    proofread_realign.add_argument("--proofread-realign-mfa-min-content-score", type=float, default=0.70)
    proofread_realign.add_argument("--proofread-realign-primary", choices=["qwen-first", "mfa-local", "original-timing"], default="qwen-first")
    proofread_realign.add_argument("--proofread-realign-retry-method", choices=["none", "original-timing"], default="none")
    proofread_realign.add_argument("--proofread-realign-max-items", type=int, default=0)
    proofread_realign.add_argument("--proofread-realign-manifest", default="")
    proofread_realign.add_argument("--proofread-realign-diagnostics-dir", default="")
    proofread_realign.add_argument("--proofread-realign-report-output", default="")
    proofread_realign.set_defaults(func=cmd_proofread_realign)

    export = subparsers.add_parser("export", parents=[common])
    export.add_argument("--format", choices=["srt", "vtt", "both"], default="srt")
    export.add_argument("--source", choices=["auto", "normalized", "mimo", "translated", "split", "aligned", "transcript"], default="auto")
    export.add_argument("--export-mode", choices=["source", "custom"], default="source")
    export.add_argument("--export-path", default=None)
    export.add_argument("--media-path", default=None)
    export.add_argument("--max-subtitle-duration", type=float, default=6.0)
    export.add_argument("--min-subtitle-duration", type=float, default=1.0)
    export.add_argument("--max-chars-per-line-zh", type=int, default=18)
    export.add_argument("--max-chars-per-line-en", type=int, default=42)
    export.add_argument("--max-lines", type=int, default=2)
    export.add_argument("--pause-split-seconds", type=float, default=0.8)
    export.add_argument("--coarse-subtitles", action="store_true")
    export.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    export.set_defaults(func=cmd_export)

    split = subparsers.add_parser("split", parents=[common])
    split.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    split.add_argument("--thread-num", type=int, default=DEFAULT_LLM_CONCURRENCY)
    split.add_argument("--max-word-count-cjk", type=int, default=18)
    split.add_argument("--max-word-count-english", type=int, default=18)
    split.set_defaults(
        split_mode="rule",
        prompt_limit_ratio=0.8,
        llm_model="",
        llm_base_url="",
        llm_api_key="",
        disable_thinking=False,
        llm_extra_body_json="",
        timeout=120.0,
    )
    split.set_defaults(func=cmd_split)

    translate = subparsers.add_parser("translate", parents=[common])
    translate.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    translate.add_argument("--target-language", required=True)
    translate.add_argument("--thread-num", type=int, default=DEFAULT_LLM_CONCURRENCY)
    translate.add_argument("--batch-num", type=int, default=20)
    translate.add_argument("--custom-prompt", default="")
    translate.add_argument("--glossary-xlsx", default=None)
    _add_llm_arguments(translate, required=True, disable_thinking_default=True)
    translate.set_defaults(func=cmd_translate)

    correct = subparsers.add_parser("correct", parents=[common])
    correct.add_argument("--thread-num", type=int, default=DEFAULT_LLM_CONCURRENCY)
    correct.add_argument("--batch-num", type=int, default=8)
    correct.add_argument("--glossary-xlsx", default=None)
    _add_llm_arguments(correct, required=False, disable_thinking_default=True)
    correct.set_defaults(func=cmd_correct)

    mimo_proofread = subparsers.add_parser("mimo-proofread", parents=[common])
    _add_mimo_proofread_arguments(mimo_proofread)
    mimo_proofread.set_defaults(func=cmd_mimo_proofread)

    history_glossary = subparsers.add_parser("history-glossary", parents=[common, model_common])
    history_glossary.add_argument("--history-dir", required=True)
    history_glossary.add_argument("--output-xlsx", required=True)
    history_glossary.add_argument("--episode-filter", default=None)
    history_glossary.add_argument("--review-ass", default=None)
    history_glossary.add_argument("--export-matches", default=None)
    history_glossary.add_argument("--min-match-score", type=float, default=0.72)
    history_glossary.add_argument("--min-term-frequency", type=int, default=2)
    history_glossary.add_argument("--extractor-mode", choices=["curated", "llm"], default="curated")
    _add_prepare_audio_arguments(history_glossary, denoise_default=True)
    history_glossary.add_argument("--model", default=DEFAULT_ASR_MODEL)
    history_glossary.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    history_glossary.add_argument("--align-cleanup-interval", type=int, default=4)
    history_glossary.add_argument("--max-new-tokens", type=int, default=512)
    history_glossary.add_argument("--language", default=None)
    history_glossary.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SECONDS)
    history_glossary.add_argument("--min-segment-seconds", type=float, default=DEFAULT_MIN_SEGMENT_SECONDS)
    history_glossary.add_argument("--preferred-silence-ms", type=int, default=800)
    history_glossary.add_argument("--min-silence-ms", type=int, default=500)
    history_glossary.add_argument("--padding-ms", type=int, default=300)
    history_glossary.add_argument("--overlap-ms", type=int, default=0)
    history_glossary.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    history_glossary.add_argument("--max-word-count-cjk", type=int, default=18)
    history_glossary.add_argument("--max-word-count-english", type=int, default=18)
    _add_llm_arguments(history_glossary, required=False, disable_thinking_default=True)
    history_glossary.set_defaults(func=cmd_history_glossary, split_mode="rule", prompt_limit_ratio=0.8)

    ass_quality = subparsers.add_parser("ass-quality", parents=[common])
    ass_quality.add_argument("--ass", required=True)
    ass_quality.add_argument(
        "--source",
        choices=["transcript", "aligned", "split", "translated", "mimo", "normalized", "export"],
        default="export",
    )
    ass_quality.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    ass_quality.add_argument("--include-styles", default="Text - JP,Text - JP - UP")
    ass_quality.add_argument("--exclude-style-prefixes", default="OP,ED")
    ass_quality.add_argument("--offset-ms", type=int, default=None)
    ass_quality.add_argument("--window-ms", type=int, default=1200)
    ass_quality.add_argument("--diagnostic-window-ms", type=int, default=8000)
    ass_quality.add_argument("--low-score-threshold", type=float, default=0.45)
    ass_quality.add_argument("--fail-score-threshold", type=float, default=0.20)
    ass_quality.add_argument("--max-cases", type=int, default=30)
    ass_quality.add_argument("--output", default=None)
    ass_quality.add_argument("--markdown-output", default=None)
    ass_quality.set_defaults(func=cmd_ass_quality)

    run = subparsers.add_parser("run", parents=[common, model_common])
    run.add_argument("--media")
    run.add_argument("--video", help="Deprecated alias for --media.")
    _add_prepare_audio_arguments(run)
    run.add_argument("--model", default=DEFAULT_ASR_MODEL)
    run.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    run.add_argument("--align-cleanup-interval", type=int, default=4)
    run.add_argument(
        "--align-diagnostics-mode",
        choices=["off", "capture-failed"],
        default="off",
        help="Pass diagnostic capture mode to the align stage.",
    )
    run.add_argument("--align-fallback", choices=["off", "asr-short-window"], default="off")
    run.add_argument("--align-fallback-window-seconds", type=float, default=3.0)
    _add_align_timing_arguments(run)
    run.add_argument("--proofread-realign-model", default=DEFAULT_ALIGN_MODEL)
    run.add_argument("--proofread-realign-padding-ms", type=int, default=800)
    run.add_argument("--proofread-realign-language", default="Japanese")
    run.add_argument("--proofread-realign-fallback", choices=["original-timing", "fail"], default="original-timing")
    run.add_argument("--proofread-realign-mfa-fallback", choices=["off", "local"], default="off")
    run.add_argument("--proofread-realign-mfa-padding-ms", type=int, default=700)
    run.add_argument("--proofread-realign-mfa-min-content-score", type=float, default=0.70)
    run.add_argument("--proofread-realign-primary", choices=["qwen-first", "mfa-local", "original-timing"], default="qwen-first")
    run.add_argument("--proofread-realign-retry-method", choices=["none", "original-timing"], default="none")
    run.add_argument("--proofread-realign-max-items", type=int, default=0)
    run.add_argument("--proofread-realign-manifest", default="")
    run.add_argument("--proofread-realign-diagnostics-dir", default="")
    run.add_argument("--proofread-realign-report-output", default="")
    run.add_argument("--batch-size", type=int, default=None)
    run.add_argument("--batch-mode", choices=["fixed", "adaptive"], default="adaptive")
    run.add_argument("--target-batch-audio-seconds", type=float, default=None)
    run.add_argument("--single-long-segment-threshold", type=float, default=None)
    run.add_argument("--profile-batches", action="store_true")
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--language", default=None)
    run.add_argument("--with-align", action="store_true")
    run.add_argument("--with-correct", action="store_true")
    run.add_argument("--with-split", action="store_true")
    run.add_argument("--with-translate", action="store_true")
    run.add_argument("--with-mimo-proofread", action="store_true")
    run.add_argument("--with-normalize", action="store_true")
    run.add_argument("--format", choices=["srt", "vtt", "both"], default="srt")
    run.add_argument("--source", choices=["auto", "normalized", "mimo", "translated", "split", "aligned", "transcript"], default="auto")
    run.add_argument("--export-mode", choices=["source", "custom"], default="source")
    run.add_argument("--export-path", default=None)
    run.add_argument("--normalize-source", choices=["auto", "mimo", "translated", "split", "transcript"], default="auto")
    run.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SECONDS)
    run.add_argument("--min-segment-seconds", type=float, default=DEFAULT_MIN_SEGMENT_SECONDS)
    run.add_argument("--preferred-silence-ms", type=int, default=800)
    run.add_argument("--min-silence-ms", type=int, default=500)
    run.add_argument("--padding-ms", type=int, default=300)
    run.add_argument("--overlap-ms", type=int, default=0)
    run.add_argument("--eager-segment-export", action="store_true")
    run.add_argument("--max-subtitle-duration", type=float, default=6.0)
    run.add_argument("--min-subtitle-duration", type=float, default=1.0)
    run.add_argument("--max-chars-per-line-zh", type=int, default=18)
    run.add_argument("--max-chars-per-line-en", type=int, default=42)
    run.add_argument("--max-lines", type=int, default=2)
    run.add_argument("--pause-split-seconds", type=float, default=0.8)
    run.add_argument("--coarse-subtitles", action="store_true")
    run.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    run.add_argument("--target-language", default=None)
    run.add_argument("--thread-num", type=int, default=DEFAULT_LLM_CONCURRENCY)
    run.add_argument("--batch-num", type=int, default=20)
    run.add_argument("--correct-batch-num", type=int, default=8)
    run.add_argument("--custom-prompt", default="")
    run.add_argument("--glossary-xlsx", default=None)
    run.add_argument("--max-word-count-cjk", type=int, default=18)
    run.add_argument("--max-word-count-english", type=int, default=18)
    run.add_argument("--extend-ms", type=int, default=350)
    run.add_argument("--snap-gap-ms", type=int, default=200)
    run.add_argument("--min-blank-ms", type=int, default=300)
    _add_quality_suspect_arguments(run)
    _add_llm_arguments(run, required=False, disable_thinking_default=True)
    _add_mimo_proofread_arguments(run)
    run.set_defaults(func=cmd_run, split_mode="rule", prompt_limit_ratio=0.8)

    batch = subparsers.add_parser("batch-run", parents=[common, model_common])
    batch.add_argument("--manifest", default=None)
    batch.add_argument("--prepare-workers", type=int, default=2)
    batch.add_argument("--fail-fast", action="store_true")
    batch.add_argument("--media", default=None)
    batch.add_argument("--video", help="Deprecated alias for --media.")
    batch.add_argument("media_files", nargs="*")
    _add_prepare_audio_arguments(batch)
    batch.add_argument("--model", default=DEFAULT_ASR_MODEL)
    batch.add_argument("--align-model", default=DEFAULT_ALIGN_MODEL)
    batch.add_argument("--align-cleanup-interval", type=int, default=4)
    batch.add_argument(
        "--align-diagnostics-mode",
        choices=["off", "capture-failed"],
        default="off",
        help="Pass diagnostic capture mode to the align stage.",
    )
    batch.add_argument("--align-fallback", choices=["off", "asr-short-window"], default="off")
    batch.add_argument("--align-fallback-window-seconds", type=float, default=3.0)
    _add_align_timing_arguments(batch)
    batch.add_argument("--proofread-realign-model", default=DEFAULT_ALIGN_MODEL)
    batch.add_argument("--proofread-realign-padding-ms", type=int, default=800)
    batch.add_argument("--proofread-realign-language", default="Japanese")
    batch.add_argument("--proofread-realign-fallback", choices=["original-timing", "fail"], default="original-timing")
    batch.add_argument("--proofread-realign-mfa-fallback", choices=["off", "local"], default="off")
    batch.add_argument("--proofread-realign-mfa-padding-ms", type=int, default=700)
    batch.add_argument("--proofread-realign-mfa-min-content-score", type=float, default=0.70)
    batch.add_argument("--proofread-realign-primary", choices=["qwen-first", "mfa-local", "original-timing"], default="qwen-first")
    batch.add_argument("--proofread-realign-retry-method", choices=["none", "original-timing"], default="none")
    batch.add_argument("--proofread-realign-max-items", type=int, default=0)
    batch.add_argument("--proofread-realign-manifest", default="")
    batch.add_argument("--proofread-realign-diagnostics-dir", default="")
    batch.add_argument("--proofread-realign-report-output", default="")
    batch.add_argument("--batch-size", type=int, default=None)
    batch.add_argument("--batch-mode", choices=["fixed", "adaptive"], default="adaptive")
    batch.add_argument("--target-batch-audio-seconds", type=float, default=None)
    batch.add_argument("--single-long-segment-threshold", type=float, default=None)
    batch.add_argument("--profile-batches", action="store_true")
    batch.add_argument("--max-new-tokens", type=int, default=512)
    batch.add_argument("--language", default=None)
    batch.add_argument("--with-align", action="store_true")
    batch.add_argument("--with-correct", action="store_true")
    batch.add_argument("--with-split", action="store_true")
    batch.add_argument("--with-translate", action="store_true")
    batch.add_argument("--with-mimo-proofread", action="store_true")
    batch.add_argument("--with-normalize", action="store_true")
    batch.add_argument("--format", choices=["srt", "vtt", "both"], default="srt")
    batch.add_argument("--source", choices=["auto", "normalized", "mimo", "translated", "split", "aligned", "transcript"], default="auto")
    batch.add_argument("--export-mode", choices=["source", "custom"], default="source")
    batch.add_argument("--export-path", default=None)
    batch.add_argument("--normalize-source", choices=["auto", "mimo", "translated", "split", "transcript"], default="auto")
    batch.add_argument("--max-segment-seconds", type=float, default=DEFAULT_MAX_SEGMENT_SECONDS)
    batch.add_argument("--min-segment-seconds", type=float, default=DEFAULT_MIN_SEGMENT_SECONDS)
    batch.add_argument("--preferred-silence-ms", type=int, default=800)
    batch.add_argument("--min-silence-ms", type=int, default=500)
    batch.add_argument("--padding-ms", type=int, default=300)
    batch.add_argument("--overlap-ms", type=int, default=0)
    batch.add_argument("--eager-segment-export", action="store_true")
    batch.add_argument("--max-subtitle-duration", type=float, default=6.0)
    batch.add_argument("--min-subtitle-duration", type=float, default=1.0)
    batch.add_argument("--max-chars-per-line-zh", type=int, default=18)
    batch.add_argument("--max-chars-per-line-en", type=int, default=42)
    batch.add_argument("--max-lines", type=int, default=2)
    batch.add_argument("--pause-split-seconds", type=float, default=0.8)
    batch.add_argument("--coarse-subtitles", action="store_true")
    batch.add_argument("--optimizer-root", default=str(DEFAULT_OPTIMIZER_ROOT))
    batch.add_argument("--target-language", default=None)
    batch.add_argument("--thread-num", type=int, default=DEFAULT_LLM_CONCURRENCY)
    batch.add_argument("--batch-num", type=int, default=20)
    batch.add_argument("--correct-batch-num", type=int, default=8)
    batch.add_argument("--custom-prompt", default="")
    batch.add_argument("--glossary-xlsx", default=None)
    batch.add_argument("--max-word-count-cjk", type=int, default=18)
    batch.add_argument("--max-word-count-english", type=int, default=18)
    batch.add_argument("--extend-ms", type=int, default=350)
    batch.add_argument("--snap-gap-ms", type=int, default=200)
    batch.add_argument("--min-blank-ms", type=int, default=300)
    _add_quality_suspect_arguments(batch)
    _add_llm_arguments(batch, required=False, disable_thinking_default=True)
    _add_mimo_proofread_arguments(batch)
    batch.set_defaults(func=cmd_run, split_mode="rule", prompt_limit_ratio=0.8)

    return parser


def _add_prepare_audio_arguments(parser: argparse.ArgumentParser, *, denoise_default: bool = False) -> None:
    if denoise_default:
        parser.add_argument(
            "--denoise",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run source separation / denoise before VAD/ASR.",
        )
    else:
        parser.add_argument("--denoise", action="store_true", help="Run source separation / denoise before VAD/ASR.")
    parser.add_argument(
        "--denoise-backend",
        choices=["mdx_net", "ffmpeg"],
        default="mdx_net",
        help="Denoise backend. mdx_net separates vocals with an MDX-Net model; ffmpeg keeps the legacy filter chain.",
    )
    parser.add_argument("--denoise-level", type=float, default=12.0, help="Post-filter noise reduction level in dB.")
    parser.add_argument(
        "--denoise-profile",
        choices=["light", "medium", "strong", "speech"],
        default="strong",
        help="Legacy ffmpeg denoise profile.",
    )
    parser.add_argument("--mdx-model", default="UVR-MDX-NET-Inst_HQ_3.onnx")
    parser.add_argument("--mdx-model-dir", default=str(DEFAULT_MODEL_CACHE_DIR / "mdx-net"))
    parser.add_argument(
        "--vad-backend",
        choices=["pyannote_onnx_v3", "silero"],
        default="pyannote_onnx_v3",
        help="Voice activity detector used before segment building.",
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Silero threshold.")
    parser.add_argument("--vad-onset", type=float, default=0.5, help="pyannote onset threshold.")
    parser.add_argument("--vad-offset", type=float, default=0.35, help="pyannote offset threshold.")
    parser.add_argument("--vad-min-speech-ms", type=int, default=180)
    parser.add_argument("--vad-min-silence-ms", type=int, default=250)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=120)
    parser.add_argument("--pyannote-onnx-model", default="segmentation-3.0")


def _add_quality_suspect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quality-ass", default="")
    parser.add_argument("--quality-ass-source", choices=["split", "translated", "normalized", "export"], default="translated")
    parser.add_argument("--quality-ass-diff-sources", default="")
    parser.add_argument("--quality-suspect-include-main-ass-report", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-ass-offset-ms", type=int, default=None)
    parser.add_argument("--quality-ass-window-ms", type=int, default=1200)
    parser.add_argument("--quality-ass-diagnostic-window-ms", type=int, default=8000)
    parser.add_argument("--quality-ass-low-score-threshold", type=float, default=0.45)
    parser.add_argument("--quality-ass-fail-score-threshold", type=float, default=0.20)
    parser.add_argument("--quality-ass-max-cases", type=int, default=30)
    parser.add_argument("--quality-suspect-ass-report", default="")
    parser.add_argument("--quality-suspect-ass-diff-report", default="")
    parser.add_argument("--quality-suspect-max-distance-ms", type=int, default=8000)
    parser.add_argument("--quality-suspect-report-output", default="")


def _apply_model_cache_default(args: argparse.Namespace) -> None:
    if hasattr(args, "model_cache_dir") and not args.model_cache_dir:
        args.model_cache_dir = str(DEFAULT_MODEL_CACHE_DIR)


def cmd_glossary_normalize(args: argparse.Namespace) -> int:
    result = write_normalized_glossary_xlsx(
        input_path=Path(args.xlsx),
        output_path=Path(args.output) if args.output else None,
    )
    print(f"Normalized glossary: {result.output_path}")
    print(f"Entries: {result.entry_count}")
    return 0


def _add_llm_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool,
    disable_thinking_default: bool,
) -> None:
    parser.add_argument("--llm-model", required=False, default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", required=False, default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--llm-api-key", required=False, default=None)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=disable_thinking_default)
    parser.add_argument("--llm-extra-body-json", default=DEFAULT_LLM_EXTRA_BODY_JSON)
    parser.add_argument("--timeout", type=float, default=DEFAULT_LLM_TIMEOUT)


def _add_align_timing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--align-min-coverage-ratio", type=float, default=0.2)
    parser.add_argument("--align-max-zero-run", type=int, default=8)
    parser.add_argument("--align-dense-zero-ratio", type=float, default=0.5)
    parser.add_argument("--align-min-dense-coverage-ratio", type=float, default=0.5)
    parser.add_argument("--align-local-collapse-min-chars", type=int, default=8)
    parser.add_argument("--align-local-collapse-max-duration-ms", type=int, default=500)
    parser.add_argument("--align-local-collapse-max-cps", type=float, default=35.0)
    parser.add_argument("--align-local-collapse-max-tokens", type=int, default=12)


def _add_mimo_proofread_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mimo-api-key", default=None, help="Optional override; otherwise MIMO_API_KEY is used.")
    parser.add_argument("--mimo-proofread-mode", choices=["segment-audio", "two-stage-nearby"], default="segment-audio")
    parser.add_argument("--mimo-audio-review-scope", choices=["suspects", "all"], default="suspects")
    parser.add_argument(
        "--mimo-diagnostic-all",
        action="store_true",
        help="Allow --mimo-audio-review-scope all for explicit diagnostic experiments only.",
    )
    parser.add_argument("--mimo-proofread-workers", type=int, default=DEFAULT_LLM_CONCURRENCY)
    parser.add_argument("--mimo-nearby-batch-size", type=int, default=1)
    parser.add_argument("--mimo-nearby-batch-max-gap-s", type=float, default=8.0)
    parser.add_argument("--mimo-nearby-padding-s", type=float, default=1.5)
    parser.add_argument("--mimo-nearby-context-subtitles", type=int, default=1)
    parser.add_argument("--mimo-nearby-audio-workers", type=int, default=DEFAULT_LLM_CONCURRENCY)
    parser.add_argument("--mimo-proofread-max-tokens", type=int, default=4096)
    parser.add_argument("--mimo-stage2-apply-threshold", type=float, default=0.9)
    parser.add_argument("--mimo-llm-extra-body-json", default=DEFAULT_LLM_EXTRA_BODY_JSON)
    parser.add_argument("--mimo-compact-output", action=argparse.BooleanOptionalAction, default=False)


if __name__ == "__main__":
    sys.exit(main())
