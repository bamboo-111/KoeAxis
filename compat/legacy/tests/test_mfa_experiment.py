from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_experiment import (
    apply_mfa_local_writeback,
    build_mfa_alignment_experiment_report,
    detect_mfa_environment,
    run_local_mfa_alignment_experiments,
)
from qwen_asr.models import WorkPaths
from qwen_asr.storage import read_json, write_json_atomic


def test_detect_mfa_environment_reports_skip_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("qwen_asr.mfa_experiment._project_mfa_command", lambda: ([], "", ""))
    monkeypatch.setattr("qwen_asr.mfa_experiment.shutil.which", lambda _name: None)
    monkeypatch.setattr("qwen_asr.mfa_experiment._project_mfa_root", lambda: None)

    report = detect_mfa_environment(run_version_check=False)

    assert report["available"] is False
    assert report["executable"] == ""


def test_detect_mfa_environment_prefers_project_micromamba_command(monkeypatch) -> None:
    monkeypatch.setattr(
        "qwen_asr.mfa_experiment._project_mfa_command",
        lambda: (["micro", "run", "-p", "env", "mfa"], "micro", "micromamba-run"),
    )
    monkeypatch.setattr("qwen_asr.mfa_experiment.shutil.which", lambda _name: "path-mfa")
    monkeypatch.setattr("qwen_asr.mfa_experiment._project_mfa_root", lambda: None)

    report = detect_mfa_environment(run_version_check=False)

    assert report["available"] is True
    assert report["executable"] == "micro"
    assert report["invocation"] == "micromamba-run"
    assert report["command"] == ["micro", "run", "-p", "env", "mfa"]


def test_mfa_experiment_collects_local_alignment_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "qwen_asr.mfa_experiment.detect_mfa_environment",
        lambda run_version_check=True: {
            "available": False,
            "executable": "",
            "invocation": "",
            "command": [],
            "root_dir": "",
            "package_version": "",
            "version_output": "",
            "version_error": "",
        },
    )
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.content_quality_report,
        {
            "status": "FAIL",
            "issues": [
                {
                    "severity": "FAIL",
                    "type": "missing_short_response",
                    "text": "\u306f\u3044",
                    "start_ms": 1000,
                    "end_ms": 1300,
                }
            ],
        },
    )

    report = build_mfa_alignment_experiment_report(paths, max_candidates=10)

    assert report["status"] == "SKIP"
    assert report["candidate_count"] == 1
    assert report["candidates"][0]["source"] == "content-quality"
    assert report["candidates"][0]["reason"] == "missing_short_response"
    assert "ass_local_score" in report["pass_criteria"]


def test_run_local_mfa_alignment_experiment_extracts_and_parses_words(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append(command_list)
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[9])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "clip.json",
            {
                "tiers": {
                    "words": {
                        "entries": [
                            [0.12, 0.34, "\u306f\u3044"],
                        ],
                    },
                },
            },
        )
        return Completed(stdout="ok")

    monkeypatch.setattr("qwen_asr.mfa_experiment.subprocess.run", fake_run)

    result = run_local_mfa_alignment_experiments(
        paths,
        [
            {
                "source": "content-quality",
                "reason": "missing_short_response",
                "severity": "FAIL",
                "start_ms": 1000,
                "end_ms": 1300,
                "text": "\u306f\u3044",
                "details": {"previous_score": 0.2, "current_score": 0.0},
            }
        ],
        environment={
            "available": True,
            "command": ["micro", "run", "-p", "env", "mfa"],
            "root_dir": str(tmp_path / "mfa-root"),
        },
        max_run_candidates=1,
        padding_ms=100,
    )

    assert result[0]["status"] == "completed"
    assert result[0]["usable"] is True
    assert result[0]["word_quality"]["known_timed_count"] == 1
    assert result[0]["start_ms"] == 900
    assert result[0]["lab_text_source"] == "candidate"
    assert result[0]["global_word_ranges"] == [{"start_ms": 1020, "end_ms": 1240, "text": "\u306f\u3044"}]
    assert result[0]["local_ass_guard"]["status"] == "PASS"
    assert result[0]["local_ass_guard"]["text_score"] == 1.0
    assert result[0]["writeback_dry_run"]["status"] == "PASS"
    assert result[0]["writeback_dry_run"]["score_delta_vs_current"] == 1.0
    assert any(call[0] == "ffmpeg" for call in calls)
    assert any(call[:5] == ["micro", "run", "-p", "env", "mfa"] for call in calls)


def test_run_local_mfa_alignment_normalizes_short_response_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1300,
                "original_subtitle": "\u306f\u3044",
            },
        },
    )

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[9])
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.1, 0.3, "\u306f\u3044"]]}}},
        )
        return Completed()

    monkeypatch.setattr("qwen_asr.mfa_experiment.subprocess.run", fake_run)

    result = run_local_mfa_alignment_experiments(
        paths,
        [
            {
                "source": "ass-quality",
                "reason": "short-dialogue-missing",
                "severity": "FAIL",
                "start_ms": 1000,
                "end_ms": 1300,
                "text": "\u306f\u00b4 \u306f\u3044",
                "details": {"previous_score": 0.8, "current_score": 0.0},
            }
        ],
        environment={"available": True, "command": ["micro", "run", "-p", "env", "mfa"], "root_dir": ""},
        max_run_candidates=1,
        padding_ms=100,
    )

    assert result[0]["status"] == "completed"
    assert result[0]["lab_text"] == "\u306f\u3044"
    assert result[0]["lab_text_source"] == "candidate-normalized"
    assert result[0]["local_ass_guard"]["status"] == "PASS"
    assert result[0]["writeback_dry_run"]["status"] == "PASS"


def test_run_local_mfa_alignment_uses_nearest_manifest_text_for_unusable_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1300,
                "original_subtitle": "\u306f\u3044",
            },
        },
    )

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[9])
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.1, 0.3, "\u306f\u3044"]]}}},
        )
        return Completed()

    monkeypatch.setattr("qwen_asr.mfa_experiment.subprocess.run", fake_run)

    result = run_local_mfa_alignment_experiments(
        paths,
        [
            {
                "source": "ass-quality",
                "reason": "short-dialogue-missing",
                "severity": "FAIL",
                "start_ms": 1000,
                "end_ms": 1300,
                "text": "...",
                "details": {"previous_score": 0.0, "current_score": 0.0},
            }
        ],
        environment={"available": True, "command": ["micro", "run", "-p", "env", "mfa"], "root_dir": ""},
        max_run_candidates=1,
        padding_ms=100,
    )

    assert result[0]["status"] == "completed"
    assert result[0]["lab_text"] == "\u306f\u3044"
    assert result[0]["lab_text_source"] == "nearest-manifest"


def test_run_local_mfa_alignment_guard_rejects_unknown_words(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[9])
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.1, 0.3, "<unk>"]]}}},
        )
        return Completed()

    monkeypatch.setattr("qwen_asr.mfa_experiment.subprocess.run", fake_run)

    result = run_local_mfa_alignment_experiments(
        paths,
        [
            {
                "source": "ass-quality",
                "reason": "short-dialogue-missing",
                "severity": "FAIL",
                "start_ms": 1000,
                "end_ms": 1300,
                "text": "\u306f\u3044",
                "details": {"previous_score": 0.0, "current_score": 0.0},
            }
        ],
        environment={"available": True, "command": ["micro", "run", "-p", "env", "mfa"], "root_dir": ""},
        max_run_candidates=1,
        padding_ms=100,
    )

    assert result[0]["local_ass_guard"]["status"] == "FAIL"
    assert "mfa-unknown-word" in result[0]["local_ass_guard"]["reasons"]
    assert result[0]["writeback_dry_run"]["status"] == "SKIP"


def test_apply_mfa_local_writeback_writes_manifest_copy_when_text_matches(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1300,
                "original_subtitle": "\u306f\u3044",
            }
        },
    )
    output = tmp_path / "split.mfa.json"

    report = apply_mfa_local_writeback(
        paths,
        [
            {
                "status": "completed",
                "candidate": {
                    "start_ms": 1000,
                    "end_ms": 1300,
                    "text": "\u306f\u3044",
                    "details": {"target_start_ms": 1000, "target_end_ms": 1300},
                },
                "local_ass_guard": {
                    "status": "PASS",
                    "mfa_text": "\u306f\u3044",
                    "mfa_start_ms": 1080,
                    "mfa_end_ms": 1240,
                },
                "writeback_dry_run": {"status": "PASS"},
            }
        ],
        mode="apply",
        output_path=output,
    )

    assert report["status"] == "APPLIED"
    assert report["applied_count"] == 1
    updated = read_json(output, default={})
    assert updated["1"]["start_time"] == 1080
    assert updated["1"]["end_time"] == 1240
    assert updated["1"]["mfa_local_writeback"]["previous_start_ms"] == 1000


def test_apply_mfa_local_writeback_rejects_manifest_text_mismatch(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    write_json_atomic(
        paths.split_manifest,
        {
            "1": {
                "start_time": 1000,
                "end_time": 1300,
                "original_subtitle": "\u3058\u3083\u306d",
            }
        },
    )

    report = apply_mfa_local_writeback(
        paths,
        [
            {
                "status": "completed",
                "candidate": {
                    "start_ms": 1000,
                    "end_ms": 1300,
                    "text": "\u306f\u3044",
                    "details": {"target_start_ms": 1000, "target_end_ms": 1300},
                },
                "local_ass_guard": {
                    "status": "PASS",
                    "mfa_text": "\u306f\u3044",
                    "mfa_start_ms": 1080,
                    "mfa_end_ms": 1240,
                },
                "writeback_dry_run": {"status": "PASS"},
            }
        ],
        mode="apply",
        output_path=tmp_path / "split.mfa.json",
    )

    assert report["status"] == "NOOP"
    assert report["applied_count"] == 0
    assert report["items"][0]["status"] == "REJECT"
    assert "manifest-text-mismatch" in report["items"][0]["reasons"]
