from __future__ import annotations

from pathlib import Path

from qwen_asr.mfa_experiment import _ffmpeg_extract_clip, run_local_mfa_alignment_experiments as legacy_run_local
from qwen_asr.mfa_runner import ffmpeg_extract_clip, run_local_mfa_alignment_experiments
from qwen_asr.models import WorkPaths
from qwen_asr.storage import write_json_atomic


class Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_local_mfa_alignment_skips_missing_command(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")

    result = run_local_mfa_alignment_experiments(
        paths,
        [{"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044"}],
        environment={"available": True, "command": []},
        max_run_candidates=1,
        padding_ms=100,
    )

    assert result == [{"status": "skipped", "reason": "mfa-command-missing"}]


def test_ffmpeg_extract_clip_builds_expected_command(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        calls.append([str(value) for value in command])
        return Completed()

    result = ffmpeg_extract_clip(
        tmp_path / "source.wav",
        tmp_path / "clip.wav",
        start_ms=250,
        end_ms=1750,
        run_command=fake_run,
    )

    assert result["status"] == "completed"
    assert calls[0][0] == "ffmpeg"
    assert calls[0][calls[0].index("-ss") + 1] == "0.250"
    assert calls[0][calls[0].index("-t") + 1] == "1.500"


def test_run_local_mfa_alignment_injects_root_and_parses_words(tmp_path: Path) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[dict[str, object]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append({"command": command_list, "env": kwargs.get("env", {})})
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[command_list.index("--clean") - 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.12, 0.34, "\u306f\u3044"]]}}},
        )
        return Completed(stdout="ok")

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
        environment={"available": True, "command": ["mfa"], "root_dir": str(tmp_path / "mfa-root")},
        max_run_candidates=1,
        padding_ms=100,
        run_command=fake_run,
        monotonic=iter([1.0, 1.25]).__next__,
        environ_factory=lambda: {},
    )

    assert result[0]["status"] == "completed"
    assert result[0]["elapsed_ms"] == 250
    assert result[0]["global_word_ranges"] == [{"start_ms": 1020, "end_ms": 1240, "text": "\u306f\u3044"}]
    assert calls[1]["env"]["MFA_ROOT_DIR"] == str(tmp_path / "mfa-root")


def test_mfa_experiment_runner_compatibility_aliases(tmp_path: Path, monkeypatch) -> None:
    paths = WorkPaths.from_workdir(tmp_path)
    paths.audio_path.write_bytes(b"RIFF")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN202
        command_list = [str(value) for value in command]
        calls.append(command_list)
        if command_list[0] == "ffmpeg":
            Path(command_list[-1]).write_bytes(b"RIFF")
            return Completed()
        output_dir = Path(command_list[command_list.index("--clean") - 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_dir / "clip.json",
            {"tiers": {"words": {"entries": [[0.1, 0.2, "\u306f\u3044"]]}}},
        )
        return Completed()

    monkeypatch.setattr("qwen_asr.mfa_experiment.subprocess.run", fake_run)

    result = legacy_run_local(
        paths,
        [{"start_ms": 1000, "end_ms": 1300, "text": "\u306f\u3044", "details": {"previous_score": 0.0}}],
        environment={"available": True, "command": ["mfa"], "root_dir": ""},
        max_run_candidates=1,
        padding_ms=0,
    )

    assert result[0]["status"] == "completed"
    assert calls[0][0] == "ffmpeg"
    assert calls[1][0] == "mfa"
    assert _ffmpeg_extract_clip(tmp_path / "source.wav", tmp_path / "clip.wav", start_ms=0, end_ms=1)["status"] in {
        "completed",
        "failed",
    }
