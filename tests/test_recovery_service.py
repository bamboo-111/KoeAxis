from __future__ import annotations

from pathlib import Path

from qwen_asr.models import SpeechRegion, WorkPaths
from qwen_asr.recovery_service import build_recovery_view, perform_recovery_action
from qwen_asr.storage import read_json, write_json_atomic


def _workspace(tmp_path: Path) -> WorkPaths:
    workdir = tmp_path / "workspaces" / "sample"
    (workdir / "manifests").mkdir(parents=True)
    (workdir / "reports").mkdir()
    (workdir / "audio" / "segments").mkdir(parents=True)
    paths = WorkPaths.from_workdir(workdir)
    write_json_atomic(
        paths.transcript_manifest,
        [
            {"segment_id": "before", "text": "before", "status": "completed"},
            {"segment_id": "failed", "text": "うん。", "status": "completed"},
            {"segment_id": "music", "text": "song", "status": "completed"},
        ],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {
                "segment_id": "before",
                "audio_path": "before.wav",
                "global_start_time": 0.0,
                "global_end_time": 1.0,
                "text": "before",
                "status": "completed",
                "tokens": [{"text": "before", "start_time": 0.1, "end_time": 0.8}],
            },
            {
                "segment_id": "failed",
                "audio_path": str(workdir / "audio" / "segments" / "failed.wav"),
                "global_start_time": 1.0,
                "global_end_time": 3.0,
                "text": "うん。",
                "status": "failed",
                "tokens": [{"text": "うん", "start_time": 1.0, "end_time": 1.0}],
                "error": "alignment token timing unreliable",
            },
            {
                "segment_id": "music",
                "audio_path": "music.wav",
                "global_start_time": 4.0,
                "global_end_time": 5.0,
                "text": "song",
                "status": "failed",
                "tokens": [],
            },
        ],
    )
    write_json_atomic(
        workdir / "reports" / "music.json",
        {"intervals": {"op": {"start_ms": 3500, "end_ms": 5500}}},
    )
    return paths


def test_recovery_view_queues_all_failed_dialogue_and_adds_context(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)

    view = build_recovery_view(paths)

    assert view["total"] == 1
    assert view["short_response_count"] == 1
    task = view["items"][0]
    assert task["segment_id"] == "failed"
    assert task["context"]["previous"]["segment_id"] == "before"
    assert "zero_duration_token" in task["reason_codes"]
    assert "timing_unreliable" in task["reason_codes"]


def test_verify_and_route_actions_are_persisted_with_audit(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)

    perform_recovery_action(
        paths,
        segment_id="failed",
        action="verify_transcript",
        payload={"verified_text": "うん。"},
        actor="tester",
    )
    result = perform_recovery_action(
        paths,
        segment_id="failed",
        action="route_language",
        payload={"language": "Japanese"},
        actor="tester",
    )

    state = read_json(paths.workdir / "reports" / "failed_segment_recovery.json")
    assert state["tasks"]["failed"]["verified_text"] == "うん。"
    assert state["tasks"]["failed"]["language_route"] == "Japanese"
    assert len(state["audit"]) == 2
    assert result["audit"]["actor"] == "tester"


def test_vad_localization_and_coarse_acceptance_update_journal_with_backup(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    Path(build_recovery_view(paths)["items"][0]["audio_path"]).write_bytes(b"wav")

    class FakeVad:
        def detect(self, path):  # noqa: ANN001
            return [SpeechRegion(start_time=0.2, end_time=0.8)]

    monkeypatch.setattr("qwen_asr.recovery_service.create_vad_adapter", lambda **kwargs: FakeVad())
    localized = perform_recovery_action(
        paths,
        segment_id="failed",
        action="localize_vad",
        payload={"backend": "silero"},
    )
    completed = perform_recovery_action(
        paths,
        segment_id="failed",
        action="accept_completed_coarse",
    )

    assert localized["task"]["vad_proposal"]["start_ms"] == 1200
    manifest = read_json(paths.aligned_manifest)
    item = next(row for row in manifest if row["segment_id"] == "failed")
    assert item["alignment_state"] == "completed_coarse"
    assert item["status"] == "completed"
    assert paths.aligned_checkpoint_path.exists()
    assert paths.aligned_events_path.exists()
    backup = Path(completed["task"]["result"]["backup_path"])
    assert (backup / "aligned_segments.json").exists()
    assert completed["recovery"]["total"] == 0
    assert len(completed["recovery"]["resolved"]) == 1
