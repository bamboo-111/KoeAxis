from __future__ import annotations

from qwen_asr.defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_EXTRA_BODY_JSON,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_SEGMENT_SECONDS,
    DEFAULT_MIN_SEGMENT_SECONDS,
    DEFAULT_MODEL_CACHE_DIR,
)
from qwen_asr.cli import build_parser


def _subparser(name: str):
    parser = build_parser()
    return parser._subparsers._group_actions[0].choices[name]  # noqa: SLF001


def test_cli_help_contains_correct_and_with_correct() -> None:
    parser = build_parser()

    top_help = parser.format_help()
    run_parser = _subparser("run")
    run_help = run_parser.format_help()

    assert "correct" in top_help
    assert "preflight" in top_help
    assert "batch-run" in top_help
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
        "--split-mode",
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
    assert run_args.target_batch_audio_seconds is None
    assert run_args.single_long_segment_threshold is None
    assert run_args.profile_batches is False
    assert run_args.split_mode == "token-counts"
    assert run_args.timeout == 120.0
    assert run_args.max_segment_seconds == DEFAULT_MAX_SEGMENT_SECONDS
    assert run_args.min_segment_seconds == DEFAULT_MIN_SEGMENT_SECONDS
    assert run_args.export_mode == "source"
    assert run_args.export_path is None
    assert run_args.skip_preflight is False
    assert run_args.dry_run_check is False
    assert run_args.eager_segment_export is False
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
    align_args = parser.parse_args(["align", "--workdir", "work"])
    run_args = parser.parse_args(["run", "--workdir", "work", "--media", "input.mp3"])

    assert transcribe_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)
    assert align_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)
    assert run_args.model_cache_dir == str(DEFAULT_MODEL_CACHE_DIR)


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
