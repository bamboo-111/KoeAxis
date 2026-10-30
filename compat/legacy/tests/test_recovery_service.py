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
    perform_recovery_action(
        paths,
        segment_id="failed",
        action="verify_transcript",
        payload={"verified_text": "うん。"},
        actor="tester",
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


def test_retry_align_executes_backend_and_language_route_affects_dispatch(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    calls = []

    def fake_execute(work_paths, **kwargs):  # noqa: ANN001
        calls.append((work_paths, kwargs))
        return {
            "status": "failed",
            "alignment_state": "failed",
            "strategy": kwargs["strategy"],
            "language_route": kwargs["language_route"],
            "elapsed_ms": 12,
            "error": "still failed",
        }

    monkeypatch.setattr("qwen_asr.recovery_service.execute_alignment_recovery", fake_execute)
    route = perform_recovery_action(
        paths,
        segment_id="failed",
        action="route_language",
        payload={"language": "Japanese"},
    )
    retried = perform_recovery_action(
        paths,
        segment_id="failed",
        action="retry_align",
        payload={"strategy": "auto"},
    )

    assert route["task"]["language_route_plan"]["available_strategies"] == ["qwen", "mfa-local"]
    assert calls[0][1]["language_route"] == "Japanese"
    assert retried["task"]["status"] == "retry_failed"
    assert retried["task"]["execution"]["elapsed_ms"] == 12


def test_completed_coarse_requires_verified_transcript_and_unique_region_selection(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    Path(build_recovery_view(paths)["items"][0]["audio_path"]).write_bytes(b"wav")

    class FakeVad:
        def detect(self, path):  # noqa: ANN001
            return [SpeechRegion(start_time=0.2, end_time=0.5), SpeechRegion(start_time=0.8, end_time=1.1)]

    monkeypatch.setattr("qwen_asr.recovery_service.create_vad_adapter", lambda **kwargs: FakeVad())
    localized = perform_recovery_action(paths, segment_id="failed", action="localize_vad")

    assert localized["task"]["vad_proposal"]["requires_manual_region_selection"] is True
    try:
        perform_recovery_action(paths, segment_id="failed", action="accept_completed_coarse")
    except Exception as exc:  # RecoveryError is intentionally checked by stable code
        assert getattr(exc, "code", "") == "COARSE_TRANSCRIPT_NOT_VERIFIED"
    else:  # pragma: no cover
        raise AssertionError("coarse acceptance unexpectedly succeeded")

    perform_recovery_action(
        paths,
        segment_id="failed",
        action="verify_transcript",
        payload={"verified_text": "うん。"},
    )
    try:
        perform_recovery_action(paths, segment_id="failed", action="accept_completed_coarse")
    except Exception as exc:
        assert getattr(exc, "code", "") == "COARSE_REGION_SELECTION_REQUIRED"
    else:  # pragma: no cover
        raise AssertionError("multi-region coarse acceptance unexpectedly succeeded")

    completed = perform_recovery_action(
        paths,
        segment_id="failed",
        action="accept_completed_coarse",
        payload={"region_index": 0},
        actor="tester",
    )
    recovery = next(item for item in read_json(paths.aligned_manifest) if item["segment_id"] == "failed")["recovery"]
    assert completed["task"]["status"] == "completed_coarse"
    assert recovery["selection_source"] == "vad_region:0"
    assert recovery["transcript_verified_by"] == "web-local-user"


def test_completed_coarse_can_be_undone_from_recorded_backup(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    Path(build_recovery_view(paths)["items"][0]["audio_path"]).write_bytes(b"wav")

    class FakeVad:
        def detect(self, path):  # noqa: ANN001
            return [SpeechRegion(start_time=0.2, end_time=0.8)]

    monkeypatch.setattr("qwen_asr.recovery_service.create_vad_adapter", lambda **kwargs: FakeVad())
    perform_recovery_action(
        paths,
        segment_id="failed",
        action="verify_transcript",
        payload={"verified_text": "うん。"},
    )
    perform_recovery_action(paths, segment_id="failed", action="localize_vad")
    perform_recovery_action(paths, segment_id="failed", action="accept_completed_coarse")
    undone = perform_recovery_action(paths, segment_id="failed", action="undo_recovery", actor="tester")

    restored = next(item for item in read_json(paths.aligned_manifest) if item["segment_id"] == "failed")
    assert restored["status"] == "failed"
    assert undone["task"]["status"] == "undone"
    assert undone["recovery"]["total"] == 1
