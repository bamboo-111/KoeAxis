from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from qwen_asr import mfa_environment as _mfa_environment
from qwen_asr.mfa_candidates import (
    candidate as _candidate,
    candidate_sort_key as _candidate_sort_key,
    candidates_from_ass_quality as _candidates_from_ass_quality,
    candidates_from_ass_quality_diff as _candidates_from_ass_quality_diff,
    candidates_from_content_quality as _candidates_from_content_quality,
    candidates_from_mimo_manifest as _candidates_from_mimo_manifest,
    candidates_from_proofread_realign as _candidates_from_proofread_realign,
    collect_alignment_experiment_candidates,
    dedupe_and_rank_candidates as _dedupe_and_rank_candidates,
)
from qwen_asr.mfa_guards import (
    float_or_none as _float_or_none,
    int_or_none as _int_or_none,
    is_japanese_character as _is_japanese_character,
    local_ass_match_score as _local_ass_match_score,
    local_mfa_ass_guard as _local_mfa_ass_guard,
    local_partial_ratio as _local_partial_ratio,
    mfa_writeback_dry_run as _mfa_writeback_dry_run,
    normalize_local_match_text as _normalize_local_match_text,
    range_distance_ms as _range_distance_ms,
)
from qwen_asr.mfa_lab import (
    SHORT_MFA_CANDIDATE_RESPONSES,
    choose_mfa_lab_text as _choose_mfa_lab_text_from_lab,
    clean_mfa_lab_text as _clean_mfa_lab_text_from_lab,
    is_isolated_kana_fragment as _is_isolated_kana_fragment_from_lab,
    looks_like_japanese_for_mfa as _looks_like_japanese_for_mfa_from_lab,
    nearest_manifest_text as _nearest_manifest_text_from_lab,
    needs_manifest_lab_fallback as _needs_manifest_lab_fallback_from_lab,
    normalize_mfa_candidate_lab_text as _normalize_mfa_candidate_lab_text_from_lab,
)
from qwen_asr.mfa_report import (
    format_ms as _format_ms_from_report,
    format_time_range as _format_time_range_from_report,
    render_mfa_alignment_experiment_markdown,
)
from qwen_asr.mfa_runner import (
    ffmpeg_extract_clip as _ffmpeg_extract_clip_from_runner,
    run_local_mfa_alignment_experiments as _run_local_mfa_alignment_experiments_from_runner,
    run_one_local_mfa_alignment as _run_one_local_mfa_alignment_from_runner,
)
from qwen_asr.mfa_writeback import (
    apply_mfa_local_writeback as _apply_mfa_local_writeback_from_writeback,
    build_mfa_writeback_decision as _build_mfa_writeback_decision_from_writeback,
    find_writeback_manifest_target as _find_writeback_manifest_target_from_writeback,
)
from qwen_asr.mfa_words import (
    evaluate_mfa_words as _evaluate_mfa_words,
    globalize_mfa_words as _globalize_mfa_words,
    read_mfa_words as _read_mfa_words,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


DEFAULT_REPORT_NAME = "mfa_alignment_experiment.json"
DEFAULT_MARKDOWN_NAME = "mfa_alignment_experiment.md"


def cmd_mfa_align_experiment(args: argparse.Namespace, work_paths: WorkPaths) -> int:
    writeback_output = getattr(args, "mfa_local_writeback_output", "") or ""
    report = build_mfa_alignment_experiment_report(
        work_paths,
        ass_quality_report_paths=[Path(value) for value in getattr(args, "ass_quality_report", [])],
        ass_quality_diff_report_paths=[Path(value) for value in getattr(args, "ass_quality_diff_report", [])],
        max_candidates=int(getattr(args, "max_candidates", 40)),
        run_version_check=bool(getattr(args, "mfa_version_check", True)),
        run_local_alignments=bool(getattr(args, "run_local_alignments", False)),
        max_run_candidates=int(getattr(args, "max_run_candidates", 3)),
        local_padding_ms=int(getattr(args, "local_padding_ms", 700)),
        local_writeback_mode=str(getattr(args, "mfa_local_writeback", "off")),
        local_writeback_output=Path(writeback_output) if writeback_output else None,
    )
    output = Path(getattr(args, "output", "") or work_paths.workdir / "reports" / DEFAULT_REPORT_NAME)
    markdown_output = getattr(args, "markdown_output", None)
    if markdown_output is None:
        markdown_output = work_paths.workdir / "reports" / DEFAULT_MARKDOWN_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output, report)
    if markdown_output:
        Path(markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_output).write_text(render_mfa_alignment_experiment_markdown(report), encoding="utf-8")
    print(f"MFA 局部实验诊断已写入：{output}")
    return 0


def build_mfa_alignment_experiment_report(
    work_paths: WorkPaths,
    *,
    ass_quality_report_paths: list[Path] | None = None,
    ass_quality_diff_report_paths: list[Path] | None = None,
    max_candidates: int = 40,
    run_version_check: bool = True,
    run_local_alignments: bool = False,
    max_run_candidates: int = 3,
    local_padding_ms: int = 700,
    local_writeback_mode: str = "off",
    local_writeback_output: Path | None = None,
) -> dict[str, Any]:
    environment = detect_mfa_environment(run_version_check=run_version_check)
    candidates = collect_alignment_experiment_candidates(
        work_paths,
        ass_quality_report_paths=ass_quality_report_paths or [],
        ass_quality_diff_report_paths=ass_quality_diff_report_paths or [],
        max_candidates=max_candidates,
    )
    status = "READY" if environment["available"] else "SKIP"
    local_runs: list[dict[str, Any]] = []
    if run_local_alignments:
        local_runs = run_local_mfa_alignment_experiments(
            work_paths,
            candidates,
            environment=environment,
            max_run_candidates=max_run_candidates,
            padding_ms=local_padding_ms,
        )
    local_writeback = apply_mfa_local_writeback(
        work_paths,
        local_runs,
        mode=local_writeback_mode,
        output_path=local_writeback_output,
    )
    return {
        "status": status,
        "reason": "" if environment["available"] else "MFA 未安装或不可执行；已生成局部实验候选清单",
        "environment": environment,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "local_alignment_run": {
            "enabled": run_local_alignments,
            "max_run_candidates": max_run_candidates,
            "padding_ms": local_padding_ms,
            "run_count": len(local_runs),
            "success_count": sum(1 for item in local_runs if item.get("status") == "completed"),
            "usable_count": sum(1 for item in local_runs if item.get("usable") is True),
            "unusable_count": sum(1 for item in local_runs if item.get("usable") is False),
            "failed_count": sum(1 for item in local_runs if item.get("status") == "failed"),
            "skipped_count": sum(1 for item in local_runs if item.get("status") == "skipped"),
            "items": local_runs,
        },
        "local_writeback": local_writeback,
        "pass_criteria": {
            "short_response_timing_shifted": "不得高于 Qwen3 ForcedAligner 基线",
            "alignment_unreliable_count": "alignment token timing unreliable 数量应下降",
            "proofread_realign_fallback_count": "original-timing fallback 数量应下降",
            "ass_local_score": "局部 ASS 分数不得下降",
            "runtime": "只允许用于局部疑点；单位候选耗时必须可接受",
        },
        "recommended_scope": [
            "ASS diff 疑点",
            "内容守恒 missing_short_response / short_response_timing_shifted / missing_unique_text",
            "proofread-realign fallback 条目",
            "MiMo 已修改且 needs_realign 的条目",
            "align fallback 或 token timing unreliable 片段",
        ],
    }


def detect_mfa_environment(*, run_version_check: bool = True) -> dict[str, Any]:
    return _mfa_environment.detect_mfa_environment(
        run_version_check=run_version_check,
        project_mfa_command=_project_mfa_command,
        project_mfa_root=_project_mfa_root,
        path_mfa_lookup=shutil.which,
    )


def _project_mfa_command() -> tuple[list[str], str, str]:
    return _mfa_environment.build_project_mfa_command()


def _project_mfa_executable() -> str:
    return _mfa_environment.find_project_mfa_executable()


def _project_mfa_root() -> Path | None:
    return _mfa_environment.find_project_mfa_root()


def run_local_mfa_alignment_experiments(
    work_paths: WorkPaths,
    candidates: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    max_run_candidates: int,
    padding_ms: int,
) -> list[dict[str, Any]]:
    return _run_local_mfa_alignment_experiments_from_runner(
        work_paths,
        candidates,
        environment=environment,
        max_run_candidates=max_run_candidates,
        padding_ms=padding_ms,
        run_command=subprocess.run,
        monotonic=time.monotonic,
        environ_factory=os.environ.copy,
    )


def apply_mfa_local_writeback(
    work_paths: WorkPaths,
    local_runs: list[dict[str, Any]],
    *,
    mode: str = "off",
    output_path: Path | None = None,
) -> dict[str, Any]:
    return _apply_mfa_local_writeback_from_writeback(
        work_paths,
        local_runs,
        mode=mode,
        output_path=output_path,
    )


def _build_mfa_writeback_decision(manifest: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    return _build_mfa_writeback_decision_from_writeback(manifest, run)


def _find_writeback_manifest_target(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    return _find_writeback_manifest_target_from_writeback(manifest, candidate)


def _run_one_local_mfa_alignment(
    work_paths: WorkPaths,
    candidate: dict[str, Any],
    *,
    environment: dict[str, Any],
    command: list[str],
    experiment_dir: Path,
    index: int,
    padding_ms: int,
) -> dict[str, Any]:
    return _run_one_local_mfa_alignment_from_runner(
        work_paths,
        candidate,
        environment=environment,
        command=command,
        experiment_dir=experiment_dir,
        index=index,
        padding_ms=padding_ms,
        run_command=subprocess.run,
        monotonic=time.monotonic,
        environ_factory=os.environ.copy,
    )


def _clean_mfa_lab_text(text: str) -> str:
    return _clean_mfa_lab_text_from_lab(text)


def _choose_mfa_lab_text(work_paths: WorkPaths, candidate: dict[str, Any]) -> dict[str, str]:
    return _choose_mfa_lab_text_from_lab(work_paths, candidate)


def _normalize_mfa_candidate_lab_text(cleaned_text: str) -> str:
    return _normalize_mfa_candidate_lab_text_from_lab(cleaned_text)


def _is_isolated_kana_fragment(text: str) -> bool:
    return _is_isolated_kana_fragment_from_lab(text)


def _needs_manifest_lab_fallback(text: str) -> bool:
    return _needs_manifest_lab_fallback_from_lab(text)


def _looks_like_japanese_for_mfa(text: str) -> bool:
    return _looks_like_japanese_for_mfa_from_lab(text)


def _nearest_manifest_text(work_paths: WorkPaths, candidate: dict[str, Any]) -> str:
    return _nearest_manifest_text_from_lab(work_paths, candidate)


def _ffmpeg_extract_clip(source_audio: Path, clip_path: Path, *, start_ms: int, end_ms: int) -> dict[str, Any]:
    return _ffmpeg_extract_clip_from_runner(
        source_audio,
        clip_path,
        start_ms=start_ms,
        end_ms=end_ms,
        run_command=subprocess.run,
    )


def _format_time_range(start_ms: Any, end_ms: Any) -> str:
    return _format_time_range_from_report(start_ms, end_ms)


def _format_ms(value: int) -> str:
    return _format_ms_from_report(value)
