from __future__ import annotations

import os
import time
from pathlib import Path

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

    assert state.missing_inputs("normalize") == ["translated_manifest|split_manifest|transcript_manifest"]

    write_json_atomic(paths.split_manifest, {"1": {"subtitle": "hello"}})

    assert state.missing_inputs("normalize") == []
    assert {path.name for path in state.force_delete_paths("translate")} == {
        "translated_segments.json",
        "subtitles.translated.srt",
    }


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


def test_work_paths_use_structured_layout_under_workspaces(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path / "workspaces" / "0001-demo")

    assert paths.audio_path.name == "source.wav"
    assert paths.audio_path.parent.name == "audio"
    assert paths.segments_dir.parts[-2:] == ("audio", "segments")
    assert paths.transcript_manifest.parts[-2:] == ("manifests", "transcript_segments.json")
    assert paths.transcript_text.parts[-2:] == ("drafts", "transcript.txt")
    assert paths.subtitles_srt.parts[-2:] == ("export-cache", "subtitles.srt")
    assert paths.project_metadata.name == "project.json"
