from __future__ import annotations

import argparse
from pathlib import Path

from qwen_asr.commands.stages import cmd_quality_gate
from qwen_asr.final_quality import evaluate_final_quality, validate_srt
from qwen_asr.models import WorkPaths
from qwen_asr.progress import read_progress
from qwen_asr.storage import read_json, write_json_atomic


def test_final_quality_passes_minimal_transcript_only_workdir(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.transcript_manifest,
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

    report = evaluate_final_quality(paths)

    assert report["status"] == "PASS"
    assert paths.final_quality_report.exists()


def test_quality_gate_command_records_progress_counts(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.transcript_manifest,
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

    status = cmd_quality_gate(
        argparse.Namespace(include_export=False, require_srt=False),
        paths,
    )
    progress = read_progress(paths)

    assert status == 0
    assert progress["stage"] == "quality-gate"
    assert progress["status"] == "completed"
    assert progress["done"] == 11
    assert progress["total"] == 11
    assert progress["summary"] == "\u805a\u5408\u8d28\u91cf\u95e8 PASS\uff1a0 FAIL\uff0c0 WARN"


def test_final_quality_fails_when_translation_structure_is_incomplete(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.translated_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    assert any(item["name"] == "translation_structure" and item["status"] == "FAIL" for item in report["checks"])


def test_final_quality_fails_when_alignment_has_failed_segments(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "status": "failed",
                "error": "alignment returned no tokens",
                "tokens": [],
            }
        ],
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    check = next(item for item in report["checks"] if item["name"] == "alignment_health")
    assert check["status"] == "FAIL"
    assert check["failed_count"] == 1


def test_final_quality_warns_for_alignment_one_ms_clusters(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "global_start_time": 0.0,
                "global_end_time": 0.004,
                "status": "completed",
                "tokens": [
                    {"text": "a", "start_time": 0.0, "end_time": 0.001},
                    {"text": "b", "start_time": 0.001, "end_time": 0.002},
                    {"text": "c", "start_time": 0.002, "end_time": 0.003},
                    {"text": "d", "start_time": 0.003, "end_time": 0.004},
                ],
            }
        ],
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "alignment_health")
    assert report["status"] == "WARN"
    assert check["status"] == "WARN"
    assert check["one_ms_cluster_count"] == 1


def test_final_quality_fails_when_translated_manifest_misses_current_split_keys(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""},
            "2": {"original_subtitle": "\u3046\u3093", "translated_subtitle": ""},
        },
    )
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "asr_suspect": False,
                "needs_audio_review": False,
                "suspect_types": [],
            },
        },
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    check = next(item for item in report["checks"] if item["name"] == "translation_completeness")
    assert check["status"] == "FAIL"
    assert check["split_count"] == 2
    assert check["translated_count"] == 1
    assert check["missing_count"] == 1
    assert check["missing_keys"] == ["2"]


def test_final_quality_fails_for_manifest_non_positive_duration(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1000,
                "original_subtitle": "\u306f\u3044",
            },
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "subtitle_readability")
    assert report["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert any(item["type"] == "non_positive_duration" for item in check["issues"])


def test_final_quality_warns_for_manifest_readability_outliers(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    long_text = "\u3042" * 81
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 0,
                "end_time": 100,
                "original_subtitle": "\u306f\u3044",
            },
            "2": {
                "start_time": 1000,
                "end_time": 10000,
                "original_subtitle": "\u3046\u3093",
            },
            "3": {
                "start_time": 11000,
                "end_time": 13000,
                "original_subtitle": long_text,
            },
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "subtitle_readability")
    assert report["status"] == "WARN"
    assert check["status"] == "WARN"
    assert {item["type"] for item in check["issues"]} == {
        "protected_short_too_fast",
        "very_long_duration",
        "long_text",
    }


def test_final_quality_warns_for_ordinary_subtitle_below_500ms(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 0,
                "end_time": 400,
                "original_subtitle": "\u6b21\u3067\u3059",
            },
            "2": {
                "start_time": 500,
                "end_time": 900,
                "original_subtitle": "\u306f\u3044",
            },
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "subtitle_readability")
    issues = {item["key"]: item["type"] for item in check["issues"]}
    assert check["status"] == "WARN"
    assert issues == {"1": "ordinary_subtitle_too_fast"}


def test_final_quality_treats_non_response_two_char_subtitle_as_ordinary(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 0,
                "end_time": 80,
                "original_subtitle": "\u51fa\u756a",
            },
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "subtitle_readability")
    assert check["status"] == "WARN"
    assert check["issues"][0]["type"] == "ordinary_subtitle_too_fast"


def test_final_quality_stage_checkpoint_fails_for_empty_split_artifact(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.split_manifest, {})

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "stage_checkpoint")
    assert report["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert "split" in check["stages"]


def test_final_quality_stage_checkpoint_fails_for_incomplete_translate_artifact(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": ""},
            "2": {"original_subtitle": "\u3046\u3093", "translated_subtitle": ""},
        },
    )
    write_json_atomic(
        paths.translated_manifest,
        {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"}},
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "stage_checkpoint")
    assert report["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert "translate" in check["stages"]


def test_final_quality_fails_for_legacy_translation_shape(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.translated_manifest,
        {"1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"}},
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    assert any(item["name"] == "translation_structure" and item["status"] == "FAIL" for item in report["checks"])


def test_final_quality_fails_unfinished_mimo_checkpoint(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(paths.mimo_proofread_manifest, {"1": {"translated_subtitle": "\u662f"}})
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 1,
            "audio_review_candidate_count": 2,
            "stage2_completed": 1,
        },
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    assert any(item["name"] == "mimo_checkpoint" and item["status"] == "FAIL" for item in report["checks"])


def test_final_quality_counts_two_stage_batch_target_ids(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.mimo_proofread_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {"translated_subtitle": "\u662f"},
            "2": {"translated_subtitle": "\u55ef"},
        },
    )
    stage2_report = paths.mimo_proofread_dir / "stage2.json"
    write_json_atomic(
        stage2_report,
        [
            {
                "id": "1-2",
                "status": "completed",
                "target_ids": ["1", "2"],
                "applied_count": 0,
                "rejected_count": 0,
            }
        ],
    )
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage2_report": str(stage2_report),
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 2,
            "stage2_completed": 1,
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "mimo_checkpoint")
    assert check["status"] == "PASS"
    assert check["completed_count"] == 2


def test_final_quality_fails_when_audio_review_suspects_have_no_mimo_report(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_audio_review": True,
                "suspect_types": ["ass_short_dialogue_missing"],
                "confidence": 0.5,
            }
        },
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    check = next(item for item in report["checks"] if item["name"] == "mimo_checkpoint")
    assert check["status"] == "FAIL"
    assert check["pending_audio_review"] == 1


def test_final_quality_fails_when_mimo_candidates_do_not_cover_marked_suspects(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "needs_audio_review": True,
                "suspect_types": ["ass_short_dialogue_missing"],
                "confidence": 0.5,
            },
            "2": {
                "original_subtitle": "\u3048\uff1f",
                "translated_subtitle": "\u54a6\uff1f",
                "needs_audio_review": True,
                "suspect_types": ["ass_short_dialogue_missing"],
                "confidence": 0.5,
            },
        },
    )
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 1,
            "stage2_completed": 1,
        },
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    check = next(item for item in report["checks"] if item["name"] == "mimo_checkpoint")
    assert check["status"] == "FAIL"
    assert check["expected_candidate_count"] == 2


def test_final_quality_warns_when_mimo_applied_change_lacks_evidence(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u597d\u7684",
                "proofread_history": [
                    {
                        "source": "mimo-nearby-audio",
                        "changes": {
                            "original_subtitle": {
                                "before": "\u3042",
                                "after": "\u306f\u3044",
                            }
                        },
                    }
                ],
            }
        },
    )
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 1,
            "stage2_completed": 1,
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "mimo_checkpoint")
    assert check["status"] == "WARN"
    assert check["missing_evidence_count"] == 1


def test_final_quality_fails_when_post_proofread_original_change_lacks_ass_guard(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "proofread_history": [
                    {
                        "source": "mimo-nearby-audio",
                        "changes": {
                            "original_subtitle": {
                                "before": "\u306f\u3044\u3067\u3059",
                                "after": "\u306f\u3044",
                            }
                        },
                        "evidence": {"confidence": 0.99},
                    }
                ],
            }
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "post_proofread_guard")
    assert check["status"] == "FAIL"
    assert check["issues"][0]["type"] == "missing_ass_guard"


def test_final_quality_fails_when_post_proofread_ass_score_regresses(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "proofread_history": [
                    {
                        "source": "mimo-nearby-audio",
                        "changes": {
                            "original_subtitle": {
                                "before": "\u306f\u3044",
                                "after": "\u3044\u3044\u3048",
                            }
                        },
                        "evidence": {
                            "ass_guard": {
                                "accepted": True,
                                "reason": "ass-improved",
                                "current_score": 0.8,
                                "suggested_score": 0.7,
                            }
                        },
                    }
                ],
            }
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "post_proofread_guard")
    assert check["status"] == "FAIL"
    assert check["issues"][0]["type"] == "ass_score_regression"


def test_final_quality_allows_post_proofread_original_change_with_ass_guard(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "original_subtitle": "\u3044\u3044\u3048",
                "proofread_history": [
                    {
                        "source": "mimo-nearby-audio",
                        "changes": {
                            "original_subtitle": {
                                "before": "\u306f\u3044",
                                "after": "\u3044\u3044\u3048",
                            }
                        },
                        "evidence": {
                            "ass_guard": {
                                "accepted": True,
                                "reason": "ass-improved",
                                "current_score": 0.3,
                                "suggested_score": 0.8,
                            }
                        },
                    }
                ],
            }
        },
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "post_proofread_guard")
    assert check["status"] == "PASS"
    assert check["checked_change_count"] == 1


def test_final_quality_fails_pending_proofread_realign(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "original_subtitle": "\u4fee\u6b63\u5f8c",
                "translated_subtitle": "\u4fee\u6b63\u540e",
                "needs_realign": True,
                "realign_status": "pending",
            }
        },
    )
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 1,
            "stage2_completed": 1,
        },
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "FAIL"
    assert any(item["name"] == "proofread_realign" and item["status"] == "FAIL" for item in report["checks"])


def test_final_quality_warns_when_proofread_realign_uses_fallback(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.mimo_proofread_manifest,
        {
            "1": {
                "original_subtitle": "\u306f\u3044",
                "translated_subtitle": "\u662f",
                "start_time": 0,
                "end_time": 1000,
                "needs_realign": False,
                "realign_status": "completed",
                "realign_method": "original-timing",
            }
        },
    )
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 1,
            "stage2_completed": 1,
        },
    )
    write_json_atomic(
        tmp_path / "reports" / "proofread_realign.json",
        {
            "status": "WARN",
            "fallback_count": 1,
            "mfa_completed_count": 0,
            "mfa_unusable_count": 1,
            "mfa_rejected_count": 1,
            "failed_count": 0,
        },
    )

    report = evaluate_final_quality(paths)

    assert report["status"] == "WARN"
    check = next(item for item in report["checks"] if item["name"] == "proofread_realign")
    assert check["status"] == "WARN"
    assert check["fallback_count"] == 1
    assert check["mfa_unusable_count"] == 1
    assert check["mfa_rejected_count"] == 1


def test_final_quality_fails_required_missing_srt(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    report = evaluate_final_quality(paths, include_export=True, require_srt=True)

    assert report["status"] == "FAIL"
    assert any(item["name"] == "srt_legality" and item["status"] == "FAIL" for item in report["checks"])


def test_final_quality_fails_when_normalize_changes_japanese(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044"}},
    )
    write_json_atomic(
        paths.normalized_manifest,
        {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u3044\u3044\u3048"}},
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "normalize_export_content")
    assert report["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert any(item["name"] == "normalize_content" and item["status"] == "FAIL" for item in check["checks"])


def test_final_quality_fails_when_export_changes_japanese(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.normalized_manifest,
        {"1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044"}},
    )
    paths.subtitles_srt.parent.mkdir(parents=True, exist_ok=True)
    paths.subtitles_srt.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "\u3044\u3044\u3048\n",
        encoding="utf-8",
    )

    report = evaluate_final_quality(paths)

    check = next(item for item in report["checks"] if item["name"] == "normalize_export_content")
    assert report["status"] == "FAIL"
    assert check["status"] == "FAIL"
    assert any(item["name"] == "export_content" and item["status"] == "FAIL" for item in check["checks"])


def test_final_quality_ass_summary_message_uses_report_counts(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    report_path = tmp_path / "reports" / "ass_quality.export.json"
    write_json_atomic(
        report_path,
        {
            "status": "FAIL",
            "summary": {
                "score_lt_045": 7,
                "score_lt_020": 3,
            },
        },
    )

    report = evaluate_final_quality(paths)

    ass_check = next(item for item in report["checks"] if item["name"] == "ass_quality")
    assert "低分 7" in ass_check["message"]
    assert "失败 3" in ass_check["message"]


def test_validate_srt_detects_bad_index_and_overlap(tmp_path: Path) -> None:
    srt = tmp_path / "bad.srt"
    srt.write_text(
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "\u306f\u3044\n\n"
        "3\n"
        "00:00:01,400 --> 00:00:01,500\n"
        "\u3048\uff1f\n",
        encoding="utf-8",
    )

    issues = validate_srt(srt)

    assert any(item["type"] == "non_continuous_index" for item in issues)
    assert any(item["type"] == "overlap" and item["severity"] == "FAIL" for item in issues)


def test_final_quality_writes_report(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)

    evaluate_final_quality(paths)
    report = read_json(paths.final_quality_report)

    assert report["status"] == "PASS"
    assert report["summary"]["pass_count"] > 0
