from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr.models import AlignedSegment, AlignedToken, WorkPaths
from qwen_asr.recovery_executor import (
    MFA_BACKGROUND_PREFETCH_ENV,
    RecoveryExecutionError,
    backup_alignment_state,
    execute_alignment_recovery,
    mfa_background_prefetch_enabled,
    prefetch_alignment_recovery_mfa,
    restore_alignment_backup,
    start_alignment_recovery_mfa_prefetch,
    wait_for_alignment_recovery_mfa_prefetch,
)
from qwen_asr.storage import read_json, write_json_atomic


def _workspace(tmp_path: Path) -> WorkPaths:
    workdir = tmp_path / "workspaces" / "sample"
    (workdir / "manifests").mkdir(parents=True)
    (workdir / "reports").mkdir()
    (workdir / "audio" / "segments").mkdir(parents=True)
    paths = WorkPaths.from_workdir(workdir)
    segment_audio = workdir / "audio" / "segments" / "failed.wav"
    segment_audio.write_bytes(b"wav")
    write_json_atomic(
        paths.transcript_manifest,
        [{"segment_id": "failed", "audio_path": str(segment_audio), "global_start_time": 1.0, "global_end_time": 3.0, "text": "うん。", "language": "Japanese", "status": "completed"}],
    )
    write_json_atomic(
        paths.aligned_manifest,
        [
            {"segment_id": "before", "audio_path": "before.wav", "global_start_time": 0.0, "global_end_time": 0.9, "text": "前", "status": "completed", "tokens": [{"text": "前", "start_time": 0.1, "end_time": 0.8}]},
            {"segment_id": "failed", "audio_path": str(segment_audio), "global_start_time": 1.0, "global_end_time": 3.0, "text": "うん。", "language": "Japanese", "status": "failed", "tokens": [], "error": "alignment returned no tokens"},
            {"segment_id": "after", "audio_path": "after.wav", "global_start_time": 3.1, "global_end_time": 4.0, "text": "後", "status": "completed", "tokens": [{"text": "後", "start_time": 3.2, "end_time": 3.8}]},
        ],
    )
    return paths


def _qwen_success(segment):  # noqa: ANN001
    return AlignedSegment(
        segment_id=segment.segment_id,
        audio_path=segment.audio_path,
        global_start_time=segment.global_start_time,
        global_end_time=segment.global_end_time,
        text=segment.text,
        language=segment.language,
        tokens=[AlignedToken(text=segment.text.rstrip("。"), start_time=1.2, end_time=1.8)],
        status="completed",
    )


def test_qwen_recovery_defaults_to_original_transcript_and_writes_exact_with_backup(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    seen = []

    def runner(segment):  # noqa: ANN001
        seen.append(segment.text)
        return _qwen_success(segment)

    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="qwen",
        verified_text="はい。",
        use_verified_text=False,
        qwen_runner=runner,
    )

    item = next(row for row in read_json(paths.aligned_manifest) if row["segment_id"] == "failed")
    assert seen == ["うん。"]
    assert result["alignment_state"] == "completed_exact"
    assert item["alignment_state"] == "completed_exact"
    assert item["recovery"]["text_source"] == "original_transcript"
    assert Path(result["backup_path"], "aligned_segments.json").exists()


def test_real_qwen_recovery_loads_and_closes_aligner(tmp_path: Path, monkeypatch) -> None:
    paths = _workspace(tmp_path)
    calls: list[str] = []

    class FakeAligner:
        def __init__(self, **kwargs):
            del kwargs
            self.loaded = False

        def load(self) -> None:
            self.loaded = True
            calls.append("load")

        def run_segment(self, segment, cleanup=True):  # noqa: ANN001
            assert self.loaded is True
            assert cleanup is False
            calls.append("run")
            return AlignedSegment(
                segment_id=segment.segment_id,
                audio_path=segment.audio_path,
                global_start_time=segment.global_start_time,
                global_end_time=segment.global_end_time,
                text=segment.text,
                language=segment.language,
                tokens=[AlignedToken(text="うん", start_time=1.2, end_time=1.8)],
                status="completed",
            )

        def close(self) -> None:
            calls.append("close")
            self.loaded = False

    monkeypatch.setattr("qwen_asr.recovery_executor.QwenForcedAligner", FakeAligner)

    result = execute_alignment_recovery(paths, segment_id="failed", strategy="qwen")

    assert result["alignment_state"] == "completed_exact"
    assert calls == ["load", "run", "close"]


def test_auto_language_route_inherits_transcript_language(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    languages = []

    def runner(segment):  # noqa: ANN001
        languages.append(segment.language)
        return _qwen_success(segment)

    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="qwen",
        language_route="auto",
        qwen_runner=runner,
    )

    assert result["language_route"] == "japanese"
    assert languages == ["Japanese"]


def test_verified_text_requires_explicit_opt_in_and_is_audited(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="qwen",
        verified_text="はい。",
        use_verified_text=True,
        actor="tester",
        qwen_runner=_qwen_success,
    )

    item = next(row for row in read_json(paths.aligned_manifest) if row["segment_id"] == "failed")
    assert result["text_source"] == "human_verified_text"
    assert result["text_changed"] is True
    assert item["text"] == "はい。"
    assert item["recovery"]["actor"] == "tester"


def test_failed_retry_preserves_original_manifest(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    before = paths.aligned_manifest.read_bytes()

    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="qwen",
        qwen_runner=lambda segment: AlignedSegment(
            segment_id=segment.segment_id,
            audio_path=segment.audio_path,
            global_start_time=segment.global_start_time,
            global_end_time=segment.global_end_time,
            text=segment.text,
            language=segment.language,
            tokens=[],
            status="failed",
            error="no tokens",
        ),
    )

    assert result["alignment_state"] == "failed"
    assert result["original_state_preserved"] is True
    assert paths.aligned_manifest.read_bytes() == before


def test_execution_lock_rejects_concurrent_request_and_recovers_after_interrupt(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    lock = paths.workdir / "reports" / "recovery-executor.lock"
    lock.write_text("{}", encoding="ascii")
    with pytest.raises(RecoveryExecutionError, match="already active") as conflict:
        execute_alignment_recovery(paths, segment_id="failed", qwen_runner=_qwen_success)
    assert conflict.value.code == "RECOVERY_CONFLICT"
    lock.unlink()

    def interrupt(segment):  # noqa: ANN001
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        execute_alignment_recovery(paths, segment_id="failed", qwen_runner=interrupt)
    assert not lock.exists()


def test_mfa_local_rejects_non_japanese_route_without_running_backend(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="mfa-local",
        language_route="English",
        mfa_runner=lambda *args, **kwargs: pytest.fail("MFA backend should not run"),
    )
    assert result["error_code"] == "MFA_LANGUAGE_NOT_APPLICABLE"
    assert result["original_state_preserved"] is True


def test_mfa_local_applies_only_guarded_usable_japanese_result(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    (paths.workdir / "audio" / "source.wav").write_bytes(b"wav")
    monkeypatch.setattr(
        "qwen_asr.recovery_executor.detect_mfa_environment",
        lambda **kwargs: {"available": True, "command": ["mfa"], "root_dir": "root"},
    )

    def fake_mfa(*args, **kwargs):  # noqa: ANN001
        return [{"status": "completed", "usable": True, "elapsed_ms": 5, "global_word_ranges": [{"text": "うん", "start_ms": 1200, "end_ms": 1700}], "word_quality": {"usable": True}}]

    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="mfa-local",
        language_route="Japanese",
        mfa_runner=fake_mfa,
    )
    item = next(row for row in read_json(paths.aligned_manifest) if row["segment_id"] == "failed")
    assert result["alignment_state"] == "completed_exact"
    assert item["alignment_backend"] == "mfa-local-recovery"
    assert item["recovery"]["backend_evidence"]["content_score"] == 1.0


def test_mfa_local_rejects_content_match_when_timing_coverage_is_unreliable(
    tmp_path: Path, monkeypatch
) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    (paths.workdir / "audio" / "source.wav").write_bytes(b"wav")
    monkeypatch.setattr(
        "qwen_asr.recovery_executor.detect_mfa_environment",
        lambda **kwargs: {"available": True, "command": ["mfa"], "root_dir": "root"},
    )

    def fake_mfa(*args, **kwargs):  # noqa: ANN001
        return [{"status": "completed", "usable": True, "global_word_ranges": [{"text": "うん", "start_ms": 1200, "end_ms": 1250}], "word_quality": {"usable": True}}]

    before = paths.aligned_manifest.read_bytes()
    result = execute_alignment_recovery(
        paths,
        segment_id="failed",
        strategy="mfa-local",
        language_route="Japanese",
        mfa_runner=fake_mfa,
    )

    assert result["error_code"] == "EXACT_TIMING_UNRELIABLE"
    assert result["original_state_preserved"] is True
    assert paths.aligned_manifest.read_bytes() == before


def test_targeted_undo_preserves_later_unrelated_manifest_changes(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)
    backup = backup_alignment_state(paths, "failed")
    rows = read_json(paths.aligned_manifest)
    failed = next(row for row in rows if row["segment_id"] == "failed")
    failed.update({"status": "completed", "alignment_state": "completed_exact", "tokens": [{"text": "うん", "start_time": 1.2, "end_time": 1.8}]})
    after = next(row for row in rows if row["segment_id"] == "after")
    after["text"] = "later-change"
    write_json_atomic(paths.aligned_manifest, rows)
    write_json_atomic(paths.aligned_checkpoint_path, rows)

    result = restore_alignment_backup(paths, backup, segment_id="failed")

    restored = read_json(paths.aligned_manifest)
    assert next(row for row in restored if row["segment_id"] == "failed")["status"] == "failed"
    assert next(row for row in restored if row["segment_id"] == "after")["text"] == "later-change"
    assert Path(result["safety_backup"]).is_dir()


def test_prefetch_recovery_mfa_batches_failed_japanese_segments_without_writeback(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    paths = _workspace(tmp_path)
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"RIFF")
    before = read_json(paths.aligned_manifest)
    monkeypatch.setattr(
        "qwen_asr.recovery_executor.detect_mfa_environment",
        lambda **_: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    seen: list[tuple[Path, list[dict], int]] = []

    def prefetcher(work_paths, candidates, *, environment, padding_ms):  # noqa: ANN001, ANN202
        seen.append((work_paths.audio_path, candidates, padding_ms))
        return {"status": "completed", "aligned_count": len(candidates)}

    summary = prefetch_alignment_recovery_mfa(paths, settings={"mfa_padding_ms": 500}, prefetcher=prefetcher)

    assert summary["status"] == "completed" and summary["candidate_count"] == 1
    assert seen[0][0] == paths.audio_path and seen[0][2] == 500
    assert seen[0][1][0]["start_ms"] == 1000 and seen[0][1][0]["end_ms"] == 3000
    assert seen[0][1][0]["text"] == "うん。"
    assert read_json(paths.aligned_manifest) == before


def test_prefetch_recovery_mfa_skips_non_japanese_routes(tmp_path: Path) -> None:
    paths = _workspace(tmp_path)

    summary = prefetch_alignment_recovery_mfa(paths, language_route="english", prefetcher=lambda *a, **k: {})

    assert summary == {"status": "skipped", "reason": "no-failed-japanese-segments", "aligned_count": 0}


def test_background_prefetch_runs_off_thread_and_mfa_local_waits_for_it(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import threading

    paths = _workspace(tmp_path)
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"RIFF")
    monkeypatch.setattr(
        "qwen_asr.recovery_executor.detect_mfa_environment",
        lambda **_: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    release = threading.Event()
    threads: list[str] = []

    def prefetcher(work_paths, candidates, *, environment, padding_ms):  # noqa: ANN001, ANN202
        threads.append(threading.current_thread().name)
        release.wait(5)
        return {"status": "completed", "aligned_count": len(candidates)}

    future = start_alignment_recovery_mfa_prefetch(paths, background=True, prefetcher=prefetcher)
    assert not future.done()
    waiter = threading.Thread(target=wait_for_alignment_recovery_mfa_prefetch, args=(paths,))
    waiter.start()
    waiter.join(0.2)
    assert waiter.is_alive()  # mfa-local would block until the prefetch finishes
    release.set()
    waiter.join(5)
    assert not waiter.is_alive()
    assert future.result(5)["candidate_count"] == 1
    assert threads == ["mfa-recovery-prefetch"]
    wait_for_alignment_recovery_mfa_prefetch(paths)  # nothing pending any more: returns immediately


def test_prefetch_is_synchronous_unless_background_is_enabled(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    import threading

    paths = _workspace(tmp_path)
    paths.audio_path.parent.mkdir(parents=True, exist_ok=True)
    paths.audio_path.write_bytes(b"RIFF")
    monkeypatch.delenv(MFA_BACKGROUND_PREFETCH_ENV, raising=False)
    monkeypatch.setattr(
        "qwen_asr.recovery_executor.detect_mfa_environment",
        lambda **_: {"available": True, "command": ["mfa"], "root_dir": ""},
    )
    threads: list[str] = []

    def prefetcher(work_paths, candidates, *, environment, padding_ms):  # noqa: ANN001, ANN202
        threads.append(threading.current_thread().name)
        return {"status": "completed", "aligned_count": len(candidates)}

    future = start_alignment_recovery_mfa_prefetch(paths, prefetcher=prefetcher)

    assert future.done() and future.result()["aligned_count"] == 1
    assert threads == [threading.current_thread().name]
    monkeypatch.setenv(MFA_BACKGROUND_PREFETCH_ENV, "1")
    assert mfa_background_prefetch_enabled() is True
