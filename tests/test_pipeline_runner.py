from __future__ import annotations

import argparse
from pathlib import Path

from optimizer.asr_data import ASRData, ASRDataSeg
from qwen_asr.commands.stages import cmd_export, cmd_normalize
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress
from qwen_asr.pipeline_runner import PipelineRunner
from qwen_asr.stages import StageResult, StageStatus
from qwen_asr.storage import read_json, write_json_atomic


def _write_valid_srt(path: Path) -> None:
    path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "\u306f\u3044\n",
        encoding="utf-8",
    )


def test_stage_order_with_correct(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []
    seen_args: dict[str, argparse.Namespace] = {}

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            seen_args[name] = args
            if name == "export":
                work_paths.subtitles_srt.write_text(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "\u4fee\u6b63\u5f8c\n",
                    encoding="utf-8",
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=True,
        with_align=True,
        with_split=True,
        with_translate=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        model="asr-model",
        max_new_tokens=256,
        language="Japanese",
        correct_batch_num=8,
        align_model="align-model",
        align_diagnostics_mode="capture-failed",
        align_fallback="asr-short-window",
        normalize_source="translated",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "correct", "align", "split", "translate", "normalize", "export")}

    status = PipelineRunner(paths, handlers).run(args)

    assert status == 0
    assert calls == ["prepare", "transcribe", "correct", "align", "split", "translate", "normalize", "export"]
    assert seen_args["correct"].batch_num == 8
    assert seen_args["align"].cleanup_interval == 4
    assert not hasattr(seen_args["align"], "align_backend")
    assert not hasattr(seen_args["align"], "mfa_num_jobs")
    assert seen_args["align"].align_diagnostics_mode == "capture-failed"
    assert seen_args["align"].align_fallback == "asr-short-window"
    assert seen_args["align"].asr_reference_model == "asr-model"
    assert seen_args["align"].asr_reference_max_new_tokens == 256
    assert seen_args["align"].asr_reference_language == "Japanese"
    assert seen_args["normalize"].source == "translated"


def test_stage_order_places_mimo_proofread_after_translate(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "mimo-proofread":
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {
                        "1": {
                            "start_time": 0,
                            "end_time": 1000,
                            "original_subtitle": "\u4fee\u6b63\u5f8c",
                            "translated_subtitle": "done",
                        }
                    },
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 0,
                        "stage2_completed": 0,
                    },
                )
            if name == "export":
                work_paths.subtitles_srt.write_text(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "\u4fee\u6b63\u5f8c\n",
                    encoding="utf-8",
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=True,
        with_split=True,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        correct_batch_num=8,
        align_model="align-model",
        normalize_source="translated",
    )
    handlers = {
        name: handler(name)
        for name in ("prepare", "transcribe", "correct", "align", "split", "translate", "mimo-proofread", "proofread-realign", "normalize", "export")
    }

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "align", "split", "translate", "mimo-proofread", "proofread-realign", "normalize", "export"]


def test_build_invocations_includes_explicit_quality_gate_before_normalize(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    args = argparse.Namespace(
        with_correct=False,
        with_align=True,
        with_split=True,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=True,
        align_model="align-model",
        normalize_source="mimo",
    )

    invocations = PipelineRunner(paths, {}).build_invocations(args)

    assert [item.name for item in invocations] == [
        "prepare",
        "transcribe",
        "align",
        "split",
        "translate",
        "mimo-proofread",
        "proofread-realign",
        "quality-gate",
        "normalize",
        "export",
    ]
    quality_args = next(item.args for item in invocations if item.name == "quality-gate")
    assert quality_args.include_export is False


def test_run_final_gate_records_status_summary(monkeypatch, tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    def fake_evaluate(*args, **kwargs):
        return {
            "status": "WARN",
            "summary": {"fail_count": 0, "warn_count": 2, "pass_count": 5},
            "checks": [],
        }

    monkeypatch.setattr("qwen_asr.pipeline_runner.evaluate_final_quality", fake_evaluate)

    status = PipelineRunner(paths, {})._run_final_gate(include_export=False, require_srt=False)
    progress = read_progress(paths)

    assert status == 0
    assert progress["stage"] == "quality-gate"
    assert progress["status"] == "completed"
    assert progress["done"] == 5
    assert progress["total"] == 7
    assert progress["summary"] == "\u805a\u5408\u8d28\u91cf\u95e8 WARN\uff1a0 FAIL\uff0c2 WARN"


def test_run_stops_before_export_when_quality_gate_fails(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "translate":
                write_json_atomic(
                    work_paths.translated_manifest,
                    {
                        "1": {
                            "start_time": 0,
                            "end_time": 1000,
                            "original_subtitle": "\u306f\u3044",
                            "translated_subtitle": "\u662f",
                            "asr_suspect": True,
                            "needs_audio_review": True,
                            "suspect_types": ["manual"],
                        }
                    },
                )
            if name == "export":
                _write_valid_srt(work_paths.subtitles_srt)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=False,
        with_normalize=False,
        force=True,
        format="srt",
        align_model="align-model",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "translate", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 1
    assert calls == ["prepare", "transcribe", "translate"]
    assert not paths.subtitles_srt.exists()
    assert read_json(paths.final_quality_report)["status"] == "FAIL"
    progress = read_progress(paths)
    assert progress["stage"] == "quality-gate"
    assert progress["status"] == "failed"


def test_standalone_normalize_is_blocked_by_failed_quality_gate(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.transcript_manifest,
        [{"segment_id": "s1", "global_start_time": 0.0, "global_end_time": 1.0, "text": "\u306f\u3044", "status": "completed"}],
    )
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u5225\u306e\u53f0\u8a5e"}},
    )

    status = cmd_normalize(argparse.Namespace(force=False), paths)

    assert status == 1
    assert not paths.normalized_manifest.exists()
    progress = read_progress(paths)
    assert progress["stage"] == "quality-gate"
    assert progress["status"] == "failed"


def test_standalone_export_is_blocked_by_failed_quality_gate(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.transcript_manifest,
        [{"segment_id": "s1", "global_start_time": 0.0, "global_end_time": 1.0, "text": "\u306f\u3044", "status": "completed"}],
    )
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u5225\u306e\u53f0\u8a5e"}},
    )

    status = cmd_export(argparse.Namespace(force=False), paths)

    assert status == 1
    assert not paths.subtitles_srt.exists()
    progress = read_progress(paths)
    assert progress["stage"] == "quality-gate"
    assert progress["status"] == "failed"


def test_run_injects_quality_suspects_before_mimo(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []
    ass_report = tmp_path / "ass_quality.json"
    injected_report = tmp_path / "quality_suspects.json"
    write_json_atomic(
        ass_report,
        {
            "rows": [
                {
                    "index": 1,
                    "ass_text": "\u306f\u3044",
                    "target_start_ms": 1000,
                    "target_end_ms": 1400,
                    "diagnostics": ["short-dialogue-missing"],
                }
            ]
        },
    )

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "translate":
                write_json_atomic(
                    work_paths.translated_manifest,
                    {
                        "1": {
                            "start_time": 900,
                            "end_time": 1500,
                            "original_subtitle": "\u306f\u3044",
                            "translated_subtitle": "\u662f",
                        }
                    },
                )
            if name == "mimo-proofread":
                translated = read_json(work_paths.translated_manifest)
                assert translated["1"]["needs_audio_review"] is True
                assert translated["1"]["suspect_types"] == ["ass_short_dialogue_missing"]
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {
                        "1": {
                            "start_time": 900,
                            "end_time": 1500,
                            "original_subtitle": "\u4fee\u6b63\u5f8c",
                            "translated_subtitle": "done",
                        }
                    },
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 1,
                    },
                    )
            if name == "export":
                work_paths.subtitles_srt.write_text(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "\u4fee\u6b63\u5f8c\n",
                    encoding="utf-8",
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=False,
        force=True,
        align_model="align-model",
        quality_suspect_ass_report=str(ass_report),
        quality_suspect_max_distance_ms=8000,
        quality_suspect_report_output=str(injected_report),
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export"]
    report = read_json(injected_report)
    assert report["applied_count"] == 1


def test_run_generates_ass_quality_report_for_quality_suspects(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []
    ass_path = tmp_path / "reference.ass"
    ass_path.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:01.40,Text - JP,,0,0,0,,\u306f\u3044\n",
        encoding="utf-8",
    )

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "translate":
                write_json_atomic(
                    work_paths.translated_manifest,
                    {
                        "1": {
                            "start_time": 900,
                            "end_time": 1500,
                            "original_subtitle": "\u5225\u306e\u53f0\u8a5e",
                            "translated_subtitle": "\u522b\u7684\u53f0\u8bcd",
                        }
                    },
                )
            if name == "mimo-proofread":
                translated = read_json(work_paths.translated_manifest)
                assert translated["1"]["needs_audio_review"] is True
                assert "ass_short_dialogue_missing" in translated["1"]["suspect_types"]
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {
                        "1": {
                            "start_time": 900,
                            "end_time": 1500,
                            "original_subtitle": "\u306f\u3044",
                            "translated_subtitle": "done",
                        }
                    },
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 1,
                    },
                    )
            if name == "export":
                _write_valid_srt(work_paths.subtitles_srt)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=False,
        force=True,
        align_model="align-model",
        quality_ass=str(ass_path),
        quality_ass_source="translated",
        quality_ass_offset_ms=0,
        quality_ass_window_ms=300,
        quality_ass_diagnostic_window_ms=1000,
        quality_ass_low_score_threshold=0.45,
        quality_ass_fail_score_threshold=0.20,
        quality_ass_max_cases=10,
        quality_suspect_ass_report="",
        quality_suspect_max_distance_ms=8000,
        quality_suspect_report_output="",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export"]
    assert (tmp_path / "reports" / "ass_quality.translated.quality_suspects.json").exists()
    assert (tmp_path / "reports" / "quality_suspects.json").exists()


def test_run_generates_ass_quality_diff_report_for_quality_suspects(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []
    ass_path = tmp_path / "reference.ass"
    ass_path.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:01.40,Text - JP,,0,0,0,,\u306f\u3044\n",
        encoding="utf-8",
    )
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 1.0,
                "global_end_time": 1.4,
                "text": "\u306f\u3044",
                "language": "Japanese",
                "status": "completed",
                "error": None,
            }
        ],
    )
    write_json_atomic(paths.split_manifest, ASRData([ASRDataSeg("\u5225\u306e\u53f0\u8a5e", 1000, 1400)]).to_json())

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "translate":
                write_json_atomic(
                    work_paths.translated_manifest,
                    {
                        "1": {
                            "start_time": 1000,
                            "end_time": 1400,
                            "original_subtitle": "\u5225\u306e\u53f0\u8a5e",
                            "translated_subtitle": "\u522b\u7684\u53f0\u8bcd",
                        }
                    },
                )
            if name == "mimo-proofread":
                translated = read_json(work_paths.translated_manifest)
                assert translated["1"]["needs_audio_review"] is True
                assert "ass_stage_became_fail" in translated["1"]["suspect_types"]
                write_json_atomic(work_paths.split_manifest, ASRData([ASRDataSeg("\u306f\u3044", 1000, 1400)]).to_json())
                translated["1"]["original_subtitle"] = "\u306f\u3044"
                write_json_atomic(work_paths.translated_manifest, translated)
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {"1": {"start_time": 1000, "end_time": 1400, "original_subtitle": "\u306f\u3044", "translated_subtitle": "done"}},
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 1,
                    },
                    )
            if name == "export":
                _write_valid_srt(work_paths.subtitles_srt)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=False,
        force=True,
        align_model="align-model",
        quality_ass=str(ass_path),
        quality_ass_source="translated",
        quality_ass_diff_sources="transcript,split",
        quality_ass_offset_ms=0,
        quality_ass_window_ms=300,
        quality_ass_diagnostic_window_ms=1000,
        quality_ass_low_score_threshold=0.45,
        quality_ass_fail_score_threshold=0.20,
        quality_ass_max_cases=10,
        quality_suspect_ass_report="",
        quality_suspect_ass_diff_report="",
        quality_suspect_include_main_ass_report=True,
        quality_suspect_max_distance_ms=8000,
        quality_suspect_report_output="",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export"]
    assert (tmp_path / "reports" / "ass_quality.transcript.quality_suspects.json").exists()
    assert (tmp_path / "reports" / "ass_quality.split.quality_suspects.json").exists()
    assert (tmp_path / "reports" / "ass_quality.diff.quality_suspects.json").exists()
    report = read_json(tmp_path / "reports" / "quality_suspects.json")
    assert report["applied_count"] >= 1
    assert [item["source"] for item in report["reports"]].count("ass-quality") == 1
    assert any(item["source"] == "ass-quality-diff" for item in report["reports"])


def test_run_can_inject_only_ass_quality_diff_suspects(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []
    ass_path = tmp_path / "reference.ass"
    ass_path.write_text(
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:01.40,Text - JP,,0,0,0,,\u306f\u3044\n",
        encoding="utf-8",
    )
    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "audio.wav",
                "global_start_time": 1.0,
                "global_end_time": 1.4,
                "text": "\u306f\u3044",
                "language": "Japanese",
                "status": "completed",
                "error": None,
            }
        ],
    )
    write_json_atomic(paths.split_manifest, ASRData([ASRDataSeg("\u5225\u306e\u53f0\u8a5e", 1000, 1400)]).to_json())

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "translate":
                write_json_atomic(
                    work_paths.translated_manifest,
                    {
                        "1": {
                            "start_time": 1000,
                            "end_time": 1400,
                            "original_subtitle": "\u5225\u306e\u53f0\u8a5e",
                            "translated_subtitle": "\u522b\u7684\u53f0\u8bcd",
                        }
                    },
                )
            if name == "mimo-proofread":
                translated = read_json(work_paths.translated_manifest)
                assert translated["1"]["needs_audio_review"] is True
                write_json_atomic(paths.split_manifest, ASRData([ASRDataSeg("\u306f\u3044", 1000, 1400)]).to_json())
                translated["1"]["original_subtitle"] = "\u306f\u3044"
                write_json_atomic(work_paths.translated_manifest, translated)
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {"1": {"start_time": 1000, "end_time": 1400, "original_subtitle": "\u306f\u3044", "translated_subtitle": "done"}},
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 1,
                    },
                )
            if name == "export":
                _write_valid_srt(work_paths.subtitles_srt)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=False,
        force=True,
        align_model="align-model",
        quality_ass=str(ass_path),
        quality_ass_source="translated",
        quality_ass_diff_sources="transcript,split",
        quality_ass_offset_ms=0,
        quality_ass_window_ms=300,
        quality_ass_diagnostic_window_ms=1000,
        quality_ass_low_score_threshold=0.45,
        quality_ass_fail_score_threshold=0.20,
        quality_ass_max_cases=10,
        quality_suspect_ass_report="",
        quality_suspect_ass_diff_report="",
        quality_suspect_include_main_ass_report=False,
        quality_suspect_max_distance_ms=8000,
        quality_suspect_report_output="",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "export"]
    report = read_json(tmp_path / "reports" / "quality_suspects.json")
    assert [item["source"] for item in report["reports"]].count("ass-quality") == 0
    assert [item["source"] for item in report["reports"]].count("ass-quality-diff") == 1


def test_run_stops_when_mimo_quality_gate_fails(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "mimo-proofread":
                write_json_atomic(work_paths.mimo_proofread_manifest, {"1": {"translated_subtitle": "done"}})
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 1,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 0,
                    },
                    )
            if name == "export":
                _write_valid_srt(work_paths.subtitles_srt)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        correct_batch_num=8,
        align_model="align-model",
        normalize_source="mimo",
    )
    handlers = {
        name: handler(name)
        for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "normalize", "export")
    }

    assert PipelineRunner(paths, handlers).run(args) == 1
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread"]


def test_run_stops_before_normalize_when_proofread_original_needs_realign(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "mimo-proofread":
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {
                        "1": {
                            "start_time": 0,
                            "end_time": 1000,
                            "original_subtitle": "\u4fee\u6b63\u5f8c",
                            "translated_subtitle": "\u4fee\u6b63\u540e",
                            "needs_realign": True,
                            "realign_status": "pending",
                        }
                    },
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 1,
                    },
                )
            if name == "export":
                work_paths.subtitles_srt.write_text(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "\u4fee\u6b63\u5f8c\n",
                    encoding="utf-8",
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        correct_batch_num=8,
        align_model="align-model",
        normalize_source="mimo",
    )
    handlers = {
        name: handler(name)
        for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "normalize", "export")
    }

    assert PipelineRunner(paths, handlers).run(args) == 1
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign"]


def test_run_allows_completed_proofread_realign(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "mimo-proofread":
                write_json_atomic(
                    work_paths.mimo_proofread_manifest,
                    {
                        "1": {
                            "start_time": 0,
                            "end_time": 1000,
                            "original_subtitle": "\u4fee\u6b63\u5f8c",
                            "translated_subtitle": "\u4fee\u6b63\u540e",
                            "needs_realign": True,
                            "realign_status": "completed",
                        }
                    },
                )
                write_json_atomic(
                    work_paths.mimo_proofread_report,
                    {
                        "mode": "two-stage-nearby",
                        "stage1_failed": 0,
                        "stage2_failed": 0,
                        "audio_review_candidate_count": 1,
                        "stage2_completed": 1,
                    },
                )
            if name == "export":
                work_paths.subtitles_srt.write_text(
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "\u4fee\u6b63\u5f8c\n",
                    encoding="utf-8",
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=True,
        with_mimo_proofread=True,
        with_normalize=True,
        skip_preflight=False,
        force=True,
        correct_batch_num=8,
        align_model="align-model",
        normalize_source="mimo",
    )
    handlers = {
        name: handler(name)
        for name in ("prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "normalize", "export")
    }

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "translate", "mimo-proofread", "proofread-realign", "normalize", "export"]


def test_run_stops_before_normalize_when_short_response_disappears(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "transcribe":
                write_json_atomic(
                    work_paths.transcript_manifest,
                    [
                        {
                            "segment_id": "s1",
                            "global_start_time": 0.0,
                            "global_end_time": 2.0,
                            "text": "前です。はい。後です。",
                            "status": "completed",
                        }
                    ],
                )
            elif name == "split":
                write_json_atomic(
                    work_paths.split_manifest,
                    {
                        "1": {"start_time": 0, "end_time": 800, "original_subtitle": "前です。"},
                        "2": {"start_time": 1200, "end_time": 2000, "original_subtitle": "後です。"},
                    },
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=True,
        with_translate=False,
        with_mimo_proofread=False,
        with_normalize=True,
        force=True,
        align_model="align-model",
        normalize_source="split",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "split", "normalize", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 1
    assert calls == ["prepare", "transcribe", "split"]
    assert paths.content_quality_report.exists()


def test_run_stops_before_export_without_normalize_when_content_disappears(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "transcribe":
                write_json_atomic(
                    work_paths.transcript_manifest,
                    [{"segment_id": "s1", "global_start_time": 0.0, "global_end_time": 1.0, "text": "駄目！", "status": "completed"}],
                )
            elif name == "split":
                write_json_atomic(work_paths.split_manifest, {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "別の台詞"}})
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=True,
        with_translate=False,
        with_mimo_proofread=False,
        with_normalize=False,
        force=True,
        align_model="align-model",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "split", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 1
    assert calls == ["prepare", "transcribe", "split"]
    report = read_json(paths.content_quality_report)
    assert report["status"] == "FAIL"


def test_run_stops_after_export_when_srt_is_invalid(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "export":
                work_paths.subtitles_srt.write_text(
                    "1\n"
                    "00:00:01,000 --> 00:00:00,900\n"
                    "\u306f\u3044\n",
                    encoding="utf-8",
                )
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=False,
        with_translate=False,
        with_mimo_proofread=False,
        with_normalize=False,
        force=True,
        align_model="align-model",
        format="srt",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 1
    assert calls == ["prepare", "transcribe", "export"]
    report = read_json(paths.final_quality_report)
    assert report["status"] == "FAIL"
    assert any(item["name"] == "srt_legality" and item["status"] == "FAIL" for item in report["checks"])


def test_run_records_content_quality_after_split_checkpoint(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    calls: list[str] = []

    def handler(name: str):
        def _run(args: argparse.Namespace, work_paths: WorkPaths) -> int:
            calls.append(name)
            if name == "transcribe":
                write_json_atomic(
                    work_paths.transcript_manifest,
                    [
                        {
                            "segment_id": "s1",
                            "global_start_time": 0.0,
                            "global_end_time": 1.0,
                            "text": "\u306f\u3044",
                            "status": "completed",
                        }
                    ],
                )
            elif name == "split":
                write_json_atomic(
                    work_paths.split_manifest,
                    {
                        "1": {
                            "start_time": 0,
                            "end_time": 1000,
                            "original_subtitle": "\u306f\u3044",
                        }
                    },
                )
            if name == "export":
                _write_valid_srt(work_paths.subtitles_srt)
            return 0

        return _run

    args = argparse.Namespace(
        with_correct=False,
        with_align=False,
        with_split=True,
        with_translate=False,
        with_mimo_proofread=False,
        with_normalize=False,
        force=True,
        align_model="align-model",
    )
    handlers = {name: handler(name) for name in ("prepare", "transcribe", "split", "export")}

    assert PipelineRunner(paths, handlers).run(args) == 0
    assert calls == ["prepare", "transcribe", "split", "export"]
    report = read_json(paths.content_quality_report)
    assert report["checked_stages"] == ["transcript", "split", "export"]
    assert report["status"] == "PASS"


def test_stage_result_defaults() -> None:
    result = StageResult(stage="prepare", status=StageStatus.SKIPPED, summary="already complete")

    assert result.return_code == 0
    assert result.status == StageStatus.SKIPPED
    assert result.summary == "already complete"
