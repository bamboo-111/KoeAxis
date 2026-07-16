from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import qwen_asr.commands.stages as stages
from qwen_asr.artifact_state import ArtifactState
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


def test_artifact_state_complete_missing_outdated_and_downstream(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)

    assert state.missing_inputs("transcribe") == ["segments_manifest"]
    assert not state.is_complete("prepare")

    paths.audio_path.write_bytes(b"wav")
    write_json_atomic(
        paths.segments_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "segment.wav",
                "source_audio_path": "audio.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "duration": 1.0,
                "status": "prepared",
            }
        ],
    )
    assert state.is_complete("prepare")
    assert state.missing_inputs("transcribe") == []

    write_json_atomic(
        paths.transcript_manifest,
        [
            {
                "segment_id": "segment_000001",
                "audio_path": "segment.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "hello",
                "status": "completed",
            }
        ],
    )
    assert state.is_complete("transcribe")

    paths.aligned_manifest.write_text("[]", encoding="utf-8")
    downstream = {path.name for path in state.downstream_existing_outputs("correct")}
    assert "aligned_segments.json" in downstream

    old_time = time.time() - 20
    os.utime(paths.transcript_manifest, (old_time, old_time))
    paths.aligned_manifest.write_text("[]", encoding="utf-8")
    assert not state.is_outdated("align")
    time.sleep(0.01)
    paths.transcript_manifest.write_text(paths.transcript_manifest.read_text(encoding="utf-8"), encoding="utf-8")
    assert state.is_outdated("align")


def test_artifact_state_any_input_groups_and_force_delete_paths(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)

    assert state.missing_inputs("normalize") == [
        "mimo_proofread_manifest|translated_manifest|split_manifest|transcript_manifest"
    ]

    write_json_atomic(paths.split_manifest, {"1": {"subtitle": "hello"}})

    assert state.missing_inputs("normalize") == []
    assert {path.name for path in state.force_delete_paths("translate")} == {
        "translated_segments.json",
        "subtitles.translated.srt",
    }


def test_mimo_proofread_invalidates_normalize_and_export(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)
    paths.normalized_manifest.write_text("{}", encoding="utf-8")
    paths.normalized_srt.write_text("", encoding="utf-8")
    paths.subtitles_srt.write_text("", encoding="utf-8")

    downstream = {path.name for path in state.downstream_existing_outputs("mimo-proofread")}

    assert "normalized_segments.json" in downstream
    assert "subtitles.normalized.srt" in downstream
    assert "subtitles.srt" in downstream


def test_dirty_review_draft_marks_quality_normalize_and_export_outdated(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)
    write_json_atomic(paths.final_quality_report, {"status": "WARN"})
    write_json_atomic(paths.normalized_manifest, {"1": {"original_subtitle": "x"}})
    paths.normalized_srt.write_text("subtitle", encoding="utf-8")
    paths.subtitles_srt.write_text("subtitle", encoding="utf-8")
    write_json_atomic(
        paths.workdir / "drafts" / "web-review.json",
        {"schema_version": 1, "dirty": True, "cues": {}, "undo_stack": []},
    )

    assert state.is_outdated("quality-gate")
    assert state.is_outdated("normalize")
    assert state.is_outdated("export")

    write_json_atomic(
        paths.workdir / "drafts" / "web-review.json",
        {"schema_version": 1, "dirty": False, "cues": {}, "undo_stack": []},
    )
    assert not state.is_outdated("quality-gate")


def test_artifact_state_does_not_mark_formal_outputs_complete_when_quality_gate_fails(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)
    write_json_atomic(paths.normalized_manifest, {"1": {"original_subtitle": "\u306f\u3044"}})
    paths.subtitles_srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n\u306f\u3044\n",
        encoding="utf-8",
    )
    write_json_atomic(paths.final_quality_report, {"status": "FAIL", "summary": {"fail_count": 1}})

    assert not state.is_complete("quality-gate")
    assert not state.is_complete("normalize")
    assert not state.is_complete("export")

    write_json_atomic(paths.final_quality_report, {"status": "WARN", "summary": {"warn_count": 1}})

    assert state.is_complete("quality-gate")
    assert state.is_complete("normalize")
    assert state.is_complete("export")


def test_mimo_proofread_incomplete_audio_review_is_not_complete(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)
    write_json_atomic(paths.mimo_proofread_manifest, {"1": {"translated_subtitle": "done"}})
    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 2,
            "stage2_completed": 1,
        },
    )

    assert not state.is_complete("mimo-proofread")

    write_json_atomic(
        paths.mimo_proofread_report,
        {
            "mode": "two-stage-nearby",
            "stage1_failed": 0,
            "stage2_failed": 0,
            "audio_review_candidate_count": 2,
            "stage2_completed": 2,
        },
    )

    assert state.is_complete("mimo-proofread")


def test_artifact_state_delete_stage_and_downstream_outputs(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)

    paths.corrected_manifest.write_text("[]", encoding="utf-8")
    paths.aligned_manifest.write_text("[]", encoding="utf-8")
    paths.split_manifest.write_text("{}", encoding="utf-8")
    paths.subtitles_srt.write_text("1\n", encoding="utf-8")

    state.delete_stage_outputs("correct")

    assert not paths.corrected_manifest.exists()
    assert paths.aligned_manifest.exists()

    state.delete_downstream_outputs("correct")

    assert not paths.aligned_manifest.exists()
    assert not paths.split_manifest.exists()
    assert not paths.subtitles_srt.exists()


def test_artifact_state_translate_requires_nonblank_translations(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)

    write_json_atomic(paths.translated_manifest, {"1": {"translated_subtitle": ""}})
    assert not state.is_complete("translate")

    write_json_atomic(paths.translated_manifest, {"1": {"translated_subtitle": "done"}})
    assert state.is_complete("translate")


def test_artifact_state_translate_requires_current_split_keys(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    state = ArtifactState(paths)

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
    assert not state.is_complete("translate")

    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"},
            "2": {"original_subtitle": "\u3046\u3093", "translated_subtitle": ""},
        },
    )
    assert not state.is_complete("translate")

    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {"original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"},
            "2": {"original_subtitle": "\u3046\u3093", "translated_subtitle": "\u55ef"},
        },
    )
    assert state.is_complete("translate")


def test_cmd_translate_resume_reruns_when_manifest_misses_split_keys(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044", "translated_subtitle": ""},
            "2": {"start_time": 1000, "end_time": 1500, "original_subtitle": "\u3046\u3093", "translated_subtitle": ""},
        },
    )
    write_json_atomic(
        paths.translated_manifest,
        {
            "1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"},
        },
    )
    called = {"value": False}

    def fake_run_translate_stage(**kwargs) -> None:
        called["value"] = True
        write_json_atomic(
            kwargs["work_paths"].translated_manifest,
            {
                "1": {"start_time": 0, "end_time": 1000, "original_subtitle": "\u306f\u3044", "translated_subtitle": "\u662f"},
                "2": {"start_time": 1000, "end_time": 1500, "original_subtitle": "\u3046\u3093", "translated_subtitle": "\u55ef"},
            },
        )

    monkeypatch.setattr(stages, "run_translate_stage", fake_run_translate_stage)
    args = argparse.Namespace(
        force=False,
        resume=True,
        target_language="zh",
        llm_model="model",
        llm_base_url="http://example.invalid",
        llm_api_key="key",
        optimizer_root=".",
        thread_num=1,
        batch_num=1,
        custom_prompt="",
        glossary_xlsx="",
        disable_thinking=True,
        llm_extra_body_json="",
        timeout=1.0,
    )

    assert stages.cmd_translate(args, paths) == 0
    assert called["value"] is True


def test_work_paths_use_structured_layout_under_workspaces(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "0001-demo")

    assert paths.audio_path.name == "source.wav"
    assert paths.audio_path.parent.name == "audio"
    assert paths.segments_dir.parts[-2:] == ("audio", "segments")
    assert paths.transcript_manifest.parts[-2:] == ("manifests", "transcript_segments.json")
    assert paths.transcript_text.parts[-2:] == ("drafts", "transcript.txt")
    assert paths.subtitles_srt.parts[-2:] == ("export-cache", "subtitles.srt")
    assert paths.project_metadata.name == "project.json"
