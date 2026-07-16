from __future__ import annotations

from argparse import Namespace

import pytest

from qwen_asr.commands.stages import _validate_mimo_diagnostic_scope
from qwen_asr.defaults import (
    DEFAULT_ALIGN_MODEL,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
)
from qwen_asr.cli import build_parser
from qwen_asr.mimo_proofread import _parse_args as parse_mimo_proofread_args


def _subparser(name: str):
    parser = build_parser()
    return parser._subparsers._group_actions[0].choices[name]  # noqa: SLF001


def test_mimo_can_be_selected_as_normalize_and_export_source() -> None:
    parser = build_parser()
    normalize_args = parser.parse_args(["normalize", "--workdir", "work", "--source", "mimo"])
    export_args = parser.parse_args(["export", "--workdir", "work", "--source", "mimo"])

    assert normalize_args.source == "mimo"
    assert export_args.source == "mimo"


def test_run_requires_diagnostic_flag_for_all_audio_review_scope() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["run", "--workdir", "work", "--mimo-audio-review-scope", "all"]
    )
    diagnostic_args = parser.parse_args(
        ["run", "--workdir", "work", "--mimo-audio-review-scope", "all", "--mimo-diagnostic-all"]
    )

    assert args.mimo_audio_review_scope == "all"
    assert diagnostic_args.mimo_diagnostic_all is True
    with pytest.raises(RuntimeError, match="diagnostic-only"):
        _validate_mimo_diagnostic_scope(args)
    _validate_mimo_diagnostic_scope(diagnostic_args)


def test_mimo_scope_suspects_does_not_require_diagnostic_flag() -> None:
    _validate_mimo_diagnostic_scope(Namespace(mimo_audio_review_scope="suspects", mimo_diagnostic_all=False))


def test_mimo_inner_cli_requires_diagnostic_flag_for_all_scope() -> None:
    with pytest.raises(SystemExit):
        parse_mimo_proofread_args(["--workdir", "work", "--api-key", "sk-test", "--audio-review-scope", "all"])

    args = parse_mimo_proofread_args(
        ["--workdir", "work", "--api-key", "sk-test", "--audio-review-scope", "all", "--diagnostic-all"]
    )

    assert args.audio_review_scope == "all"
    assert args.diagnostic_all is True


def test_mimo_proofread_uses_disabled_thinking_extra_body_by_default() -> None:
    parser = build_parser()
    args = parser.parse_args(["mimo-proofread", "--workdir", "work"])
    override = parser.parse_args(
        [
            "mimo-proofread",
            "--workdir",
            "work",
            "--mimo-llm-extra-body-json",
            '{"thinking":{"type":"enabled"}}',
        ]
    )

    assert args.mimo_llm_extra_body_json == DEFAULT_LLM_EXTRA_BODY_JSON
    assert override.mimo_llm_extra_body_json == '{"thinking":{"type":"enabled"}}'


def test_correct_command_no_longer_requires_llm_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["correct", "--workdir", "work"])

    assert args.command == "correct"
    assert args.llm_model == DEFAULT_LLM_MODEL


def test_cli_help_contains_correct_and_with_correct() -> None:
    parser = build_parser()

    top_help = parser.format_help()
    run_parser = _subparser("run")
    run_help = run_parser.format_help()

    assert "correct" in top_help
    assert "preflight" in top_help
    assert "batch-run" in top_help
    assert "ass-quality" in top_help
    assert "content-quality" in top_help
    assert "quality-gate" in top_help
    assert "proofread-realign" in top_help
    assert "ass-quality-diff" in top_help
    assert "tuning-matrix" not in top_help
    assert "baseline-snapshot" not in top_help
    assert "align-diagnose" not in top_help
    assert "align-split-audit" not in top_help
    assert "glossary-normalize" in top_help
    assert "history-glossary" in top_help
    assert "--with-correct" in run_help


def test_run_help_keeps_shared_stage_options_visible() -> None:
    help_text = _subparser("run").format_help()

    expected_options = [
        "--workdir",
        "--resume",
        "--no-resume",
        "--skip-preflight",
        "--dry-run-check",
        "--force",
        "--media",
        "--video",
        "--with-align",
        "--align-cleanup-interval",
        "--with-correct",
        "--with-split",
        "--with-translate",
        "--with-normalize",
        "--normalize-source",
        "--export-mode",
        "--export-path",
        "--batch-mode",
        "--denoise-backend",
        "--vad-backend",
        "--pyannote-onnx-model",
        "--target-batch-audio-seconds",
        "--single-long-segment-threshold",
        "--profile-batches",
        "--model-cache-dir",
        "--llm-model",
        "--local-files-only",
        "--no-local-files-only",
        "--llm-base-url",
        "--llm-api-key",
        "--disable-thinking",
        "--no-disable-thinking",
        "--llm-extra-body-json",
        "--timeout",
        "--eager-segment-export",
        "--quality-ass",
        "--quality-suspect-ass-report",
        "--quality-suspect-include-main-ass-report",
    ]

    for option in expected_options:
        assert option in help_text


def test_cli_parse_defaults_for_resume_force_and_llm_flags() -> None:
    parser = build_parser()

    run_args = parser.parse_args(["run", "--workdir", "work", "--media", "input.mp3"])
    split_args = parser.parse_args(["split", "--workdir", "work"])
    translate_args = parser.parse_args(
        [
            "translate",
            "--workdir",
            "work",
            "--llm-model",
            "model",
            "--llm-base-url",
            "http://localhost:8000/v1",
            "--llm-api-key",
            "key",
            "--target-language",
            "zh",
        ]
    )

    assert run_args.resume is True
    assert run_args.force is False
    assert run_args.disable_thinking is True
    assert run_args.llm_model == DEFAULT_LLM_MODEL
    assert run_args.llm_base_url == DEFAULT_LLM_BASE_URL
    assert run_args.llm_extra_body_json == DEFAULT_LLM_EXTRA_BODY_JSON
    assert run_args.local_files_only is True
    assert run_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)
    assert run_args.batch_mode == "adaptive"
    assert run_args.denoise is False
    assert run_args.denoise_backend == "mdx_net"
    assert run_args.vad_backend == "pyannote_onnx_v3"
    assert run_args.pyannote_onnx_model == "segmentation-3.0"
    assert run_args.batch_size is None
    assert run_args.align_cleanup_interval == 4
    assert "--align-backend" not in _subparser("run").format_help()
    assert "--mfa-num-jobs" not in _subparser("run").format_help()
    assert "--align-timing-repair" not in _subparser("run").format_help()
    assert "--align-min-token-ms" not in _subparser("run").format_help()
    assert run_args.target_batch_audio_seconds is None
    assert run_args.single_long_segment_threshold is None
    assert run_args.profile_batches is False
    assert run_args.split_mode == "rule"
    assert "--split-mode" not in _subparser("run").format_help()
    assert "--split-mode" not in _subparser("split").format_help()
    assert run_args.timeout == 120.0
    assert run_args.max_segment_seconds == DEFAULT_MAX_SEGMENT_SECONDS
    assert run_args.min_segment_seconds == DEFAULT_MIN_SEGMENT_SECONDS
    assert run_args.export_mode == "source"
    assert run_args.export_path is None
    assert run_args.skip_preflight is False
    assert run_args.dry_run_check is False
    assert run_args.eager_segment_export is False
    assert run_args.quality_ass == ""
    assert run_args.quality_ass_source == "translated"
    assert run_args.quality_ass_offset_ms is None
    assert run_args.quality_suspect_ass_report == ""
    assert run_args.quality_suspect_include_main_ass_report is True
    assert run_args.quality_suspect_max_distance_ms == 8000
    assert run_args.mimo_stage2_apply_threshold == 0.9
    assert run_args.proofread_realign_model == DEFAULT_ALIGN_MODEL
    assert run_args.proofread_realign_padding_ms == 800
    assert run_args.proofread_realign_language == "Japanese"
    assert split_args.disable_thinking is False
    assert translate_args.disable_thinking is True
    assert translate_args.batch_num == 20

    batch_args = parser.parse_args(["batch-run", "--workdir", "work", "a.mp3", "b.mp3"])
    assert batch_args.prepare_workers == 2
    assert batch_args.media_files == ["a.mp3", "b.mp3"]
    assert batch_args.max_segment_seconds == DEFAULT_MAX_SEGMENT_SECONDS
    assert batch_args.min_segment_seconds == DEFAULT_MIN_SEGMENT_SECONDS


def test_cli_parse_all_model_commands_use_project_model_cache_default() -> None:
    parser = build_parser()

    transcribe_args = parser.parse_args(["transcribe", "--workdir", "work"])
    align_args = parser.parse_args([
        "align",
        "--workdir",
        "work",
        "--align-diagnostics-mode",
        "capture-failed",
        "--align-fallback",
        "asr-short-window",
        "--align-fallback-window-seconds",
        "2.5",
        "--align-min-coverage-ratio",
        "0.3",
        "--align-max-zero-run",
        "6",
        "--align-dense-zero-ratio",
        "0.4",
        "--align-min-dense-coverage-ratio",
        "0.6",
        "--align-local-collapse-min-chars",
        "10",
        "--align-local-collapse-max-duration-ms",
        "700",
        "--align-local-collapse-max-cps",
        "40",
        "--align-local-collapse-max-tokens",
        "14",
    ])
    run_args = parser.parse_args([
        "run",
        "--workdir",
        "work",
        "--media",
        "input.mp3",
        "--align-diagnostics-mode",
        "capture-failed",
        "--align-fallback",
        "asr-short-window",
    ])

    assert transcribe_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)
    assert align_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)
    assert align_args.align_diagnostics_mode == "capture-failed"
    assert "--align-backend" not in _subparser("align").format_help()
    assert "--mfa-num-jobs" not in _subparser("align").format_help()
    assert align_args.align_fallback == "asr-short-window"
    assert align_args.align_fallback_window_seconds == 2.5
    align_help = _subparser("align").format_help()
    assert "--align-timing-repair" not in align_help
    assert "--align-min-token-ms" not in align_help
    assert "--align-local-interpolation-max-gap-ms" not in align_help
    assert "--align-zero-token-default-duration-ms" not in align_help
    assert "--align-zero-token-max-duration-ms" not in align_help
    assert align_args.align_min_coverage_ratio == 0.3
    assert align_args.align_max_zero_run == 6
    assert align_args.align_dense_zero_ratio == 0.4
    assert align_args.align_min_dense_coverage_ratio == 0.6
    assert align_args.align_local_collapse_min_chars == 10
    assert align_args.align_local_collapse_max_duration_ms == 700
    assert align_args.align_local_collapse_max_cps == 40
    assert align_args.align_local_collapse_max_tokens == 14
    assert run_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)
    assert run_args.align_diagnostics_mode == "capture-failed"
    assert run_args.align_fallback == "asr-short-window"


def test_cli_parse_model_cache_dir_can_be_overridden() -> None:
    parser = build_parser()

    args = parser.parse_args(["transcribe", "--workdir", "work", "--model-cache-dir", "cache"])

    assert args.model_cache_dir == "cache"


def test_glossary_normalize_parse_does_not_require_workdir() -> None:
    parser = build_parser()

    args = parser.parse_args(["glossary-normalize", "--xlsx", "glossary.xlsx"])

    assert args.command == "glossary-normalize"
    assert args.xlsx == "glossary.xlsx"
    assert not hasattr(args, "workdir")


def test_history_glossary_parse_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args(
        [
            "history-glossary",
            "--workdir",
            "work",
            "--history-dir",
            "D:\\history",
            "--output-xlsx",
            "glossary.xlsx",
        ]
    )

    assert args.command == "history-glossary"
    assert args.denoise is True
    assert args.denoise_backend == "mdx_net"
    assert args.vad_backend == "pyannote_onnx_v3"
    assert args.min_match_score == 0.72
    assert args.min_term_frequency == 2
    assert args.extractor_mode == "curated"
    assert args.align_cleanup_interval == 4
    assert args.max_segment_seconds == DEFAULT_MAX_SEGMENT_SECONDS
    assert args.min_segment_seconds == DEFAULT_MIN_SEGMENT_SECONDS
    assert args.batch_mode if hasattr(args, "batch_mode") else True


def test_ass_quality_parse_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "ass-quality",
        "--workdir",
        "work",
        "--ass",
        "reference.ass",
    ])

    assert args.command == "ass-quality"
    assert args.source == "export"
    assert args.include_styles == "Text - JP,Text - JP - UP"
    assert args.exclude_style_prefixes == "OP,ED"
    assert args.offset_ms is None
    assert args.diagnostic_window_ms == 8000


def test_apply_quality_suspects_parse_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "apply-quality-suspects",
        "--workdir",
        "work",
        "--ass-quality-report",
        "ass.json",
    ])

    assert args.command == "apply-quality-suspects"
    assert args.quality_suspect_max_distance_ms == 8000
    assert args.quality_suspect_report_output == ""


def test_content_quality_parse_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "content-quality",
        "--workdir",
        "work",
    ])

    assert args.command == "content-quality"
    assert args.include_export is False


def test_quality_gate_parse_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "quality-gate",
        "--workdir",
        "work",
    ])

    assert args.command == "quality-gate"
    assert args.include_export is False
    assert args.require_srt is False


def test_mfa_align_experiment_parse_local_run_options() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "mfa-align-experiment",
        "--workdir",
        "work",
        "--ass-quality-report",
        "ass.json",
        "--ass-quality-diff-report",
        "diff.json",
        "--run-local-alignments",
        "--max-run-candidates",
        "2",
        "--local-padding-ms",
        "500",
    ])

    assert args.command == "mfa-align-experiment"
    assert args.ass_quality_report == ["ass.json"]
    assert args.ass_quality_diff_report == ["diff.json"]
    assert args.run_local_alignments is True
    assert args.max_run_candidates == 2
    assert args.local_padding_ms == 500


def test_proofread_realign_parse_defaults() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "proofread-realign",
        "--workdir",
        "work",
    ])

    assert args.command == "proofread-realign"
    assert args.proofread_realign_model == DEFAULT_ALIGN_MODEL
    assert args.proofread_realign_padding_ms == 800
    assert args.proofread_realign_language == "Japanese"
    assert args.proofread_realign_fallback == "original-timing"
    assert args.proofread_realign_mfa_fallback == "off"
    assert args.proofread_realign_mfa_padding_ms == 700
    assert args.proofread_realign_mfa_min_content_score == 0.70
    assert args.proofread_realign_primary == "qwen-first"
    assert args.proofread_realign_retry_method == "none"
    assert args.proofread_realign_max_items == 0


def test_proofread_realign_parse_mfa_fallback_options() -> None:
    parser = build_parser()

    args = parser.parse_args([
        "proofread-realign",
        "--workdir",
        "work",
        "--proofread-realign-mfa-fallback",
        "local",
        "--proofread-realign-mfa-padding-ms",
        "500",
        "--proofread-realign-mfa-min-content-score",
        "0.8",
        "--proofread-realign-primary",
        "original-timing",
        "--proofread-realign-retry-method",
        "original-timing",
        "--proofread-realign-max-items",
        "1",
    ])

    assert args.command == "proofread-realign"
    assert args.proofread_realign_mfa_fallback == "local"
    assert args.proofread_realign_mfa_padding_ms == 500
    assert args.proofread_realign_mfa_min_content_score == 0.8
    assert args.proofread_realign_primary == "original-timing"
    assert args.proofread_realign_retry_method == "original-timing"
    assert args.proofread_realign_max_items == 1
