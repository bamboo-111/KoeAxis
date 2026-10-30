from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.models import WorkPaths
from qwen_asr.review_service import (
    ReviewError,
    build_review_view,
    resolve_workspace_media,
    save_review_edit,
    undo_review_edit,
)
from qwen_asr.storage import load_jsonl, read_json, write_json_atomic


def _paths(tmp_path: Path) -> WorkPaths:
    workdir = tmp_path / "workspaces" / "sample"
    (workdir / "manifests").mkdir(parents=True)
    (workdir / "audio").mkdir()
    (workdir / "references").mkdir()
    paths = WorkPaths.from_workdir(workdir)
    paths.audio_path.write_bytes(b"audio")
    write_json_atomic(
        paths.normalized_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 2000,
                "original_subtitle": "こんにちは",
                "translated_subtitle": "你好",
            }
        },
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "global_start_time": 0.5,
                "global_end_time": 2.5,
                "status": "completed",
                "tokens": [{"text": "x", "start_time": 1.0, "end_time": 1.2}],
            }
        ],
    )
    (workdir / "references" / "reference.ass").write_text(
        "[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Text - JP,,0,0,0,,こんにちは\n",
        encoding="utf-8",
    )
    return paths


def test_review_view_combines_cue_align_audio_and_reference(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    view = build_review_view(paths)

    assert view["cue_count"] == 1
    assert view["audio_path"] == str(paths.audio_path)
    cue = view["cues"][0]
    assert cue["segment_id"] == "s1"
    assert cue["alignment_state"] == "completed_exact"
    assert cue["reference"][0]["text"] == "こんにちは"
    assert view["reference_sources"][0]["mode"] == "read_only"


def test_media_path_is_limited_to_workspace(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"x")

    assert resolve_workspace_media(paths, str(paths.audio_path)) == paths.audio_path.resolve()
    with pytest.raises(ReviewError) as exc_info:
        resolve_workspace_media(paths, str(outside))
    assert exc_info.value.code == "MEDIA_PATH_OUT_OF_SCOPE"


def test_media_path_allows_manifest_linked_workspace_audio(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    linked = tmp_path / "workspaces" / "source" / "audio" / "segments" / "s1.wav"
    linked.parent.mkdir(parents=True)
    linked.write_bytes(b"linked")
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "s1",
                "audio_path": str(linked),
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "status": "failed",
            }
        ],
    )

    assert resolve_workspace_media(paths, str(linked)) == linked.resolve()


def test_review_edit_uses_draft_audit_backup_and_undo_without_overwriting_source(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    source_before = paths.normalized_manifest.read_bytes()

    first = save_review_edit(
        paths,
        cue_id="1",
        original="こんにちは。",
        translation="你好",
        start_ms=1000,
        end_ms=2000,
        expected_revision=0,
        actor="tester",
    )
    second = save_review_edit(
        paths,
        cue_id="1",
        original="こんにちは。",
        translation="您好",
        start_ms=1000,
        end_ms=2000,
        expected_revision=1,
        actor="tester",
    )

    assert first["backup_path"] is None
    assert Path(second["backup_path"]).joinpath("web-review.json").exists()
    assert paths.normalized_manifest.read_bytes() == source_before
    draft = read_json(paths.workdir / "drafts" / "web-review.json")
    assert draft["dirty"] is True
    assert draft["revision"] == 2
    assert draft["cues"]["1"]["translated_subtitle"] == "您好"
    assert [row["action"] for row in load_jsonl(paths.workdir / "reports" / "web_review_history.jsonl")] == [
        "edit",
        "edit",
    ]

    undone_second = undo_review_edit(paths, expected_revision=2, actor="reviewer")
    assert undone_second["review"]["cues"][0]["translation"] == "你好"
    undone_first = undo_review_edit(paths, expected_revision=3, actor="reviewer")
    assert undone_first["review"]["cues"][0]["original"] == "こんにちは"
    assert undone_first["review"]["review_state"]["dirty"] is False
    assert undone_first["review"]["review_state"]["can_undo"] is False
    assert paths.normalized_manifest.read_bytes() == source_before


def test_review_edit_rejects_overlap_and_stale_revision(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    payload = read_json(paths.normalized_manifest)
    payload["2"] = {
        "start_time": 2100,
        "end_time": 3000,
        "original_subtitle": "次",
        "translated_subtitle": "下一条",
    }
    write_json_atomic(paths.normalized_manifest, payload)

    with pytest.raises(ReviewError) as overlap:
        save_review_edit(
            paths,
            cue_id="1",
            original="こんにちは",
            translation="你好",
            start_ms=1000,
            end_ms=2200,
            expected_revision=0,
        )
    assert overlap.value.code == "REVIEW_TIME_OVERLAP"

    save_review_edit(
        paths,
        cue_id="1",
        original="こんにちは。",
        translation="你好",
        start_ms=1000,
        end_ms=2000,
        expected_revision=0,
    )
    with pytest.raises(ReviewError) as conflict:
        save_review_edit(
            paths,
            cue_id="1",
            original="こんにちは！",
            translation="你好",
            start_ms=1000,
            end_ms=2000,
            expected_revision=0,
        )
    assert conflict.value.code == "REVIEW_REVISION_CONFLICT"
